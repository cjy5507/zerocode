//! `.jsonl` input history + one-shot rustyline migrator (Phase 3, Lane L6).
//!
//! `.zo/spec.md` §5.1 makes `.jsonl` the mandatory on-disk format
//! for the TUI input history and requires the legacy rustyline file to
//! be migrated once and then deleted. This module implements both:
//!
//! * [`History::load`] reads an existing `.jsonl`, or — if that file
//!   does not yet exist — probes for a sibling rustyline file named
//!   `history`, imports it via [`migrate_from_rustyline`], writes the
//!   `.jsonl` snapshot, and deletes the legacy file.
//! * [`History::append`] persists one JSON-Lines record per call.
//! * [`History::search`] does a reverse substring search.
//!
//! ## Living standard (mirrors L1)
//!
//! 1. Module layout: one file.
//! 2. Errors: a dedicated [`HistoryError`] `thiserror` enum with a
//!    generic `Adapter` catch-all.
//! 3. No async — all I/O is synchronous.
//! 4. Tests live at `crates/zo-cli/tests/tui_history.rs`.
//! 5. Every `pub` item carries a `///` doc comment.

use std::collections::VecDeque;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Default maximum number of entries retained in memory / on disk.
pub const DEFAULT_MAX: usize = 1000;

/// Errors surfaced by the history module.
#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    /// I/O failure while reading or writing a history file.
    #[error("history io error: {0}")]
    Io(#[from] io::Error),

    /// JSON (de)serialization failure on a `.jsonl` record.
    #[error("history json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Catch-all wrapper for downstream adapter errors.
    #[error("history {component}: {message}")]
    Adapter {
        /// Component that produced the failure.
        component: &'static str,
        /// Human-readable description.
        message: String,
    },
}

/// One persisted history record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HistoryRecord {
    /// RFC3339-ish timestamp (`seconds_since_epoch` as string for
    /// portability — avoids pulling in `chrono` for one field).
    pub ts: String,
    /// The user-entered text.
    pub text: String,
    /// The `AppMode` tag at submit time. Free-form string so the
    /// history file stays forward-compatible with new modes.
    pub mode: String,
}

impl HistoryRecord {
    /// Build a fresh record with a `SystemTime::now` timestamp.
    #[must_use]
    pub fn now(text: String, mode: impl Into<String>) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or_else(|_| "0".to_string(), |d| d.as_secs().to_string());
        Self {
            ts,
            text,
            mode: mode.into(),
        }
    }
}

/// In-memory ring buffer backed by a `.jsonl` file on disk.
#[derive(Debug)]
pub struct History {
    path: PathBuf,
    entries: VecDeque<HistoryRecord>,
    max: usize,
}

impl History {
    /// Load history from `path`.
    ///
    /// Behavior:
    /// * If `path` exists, parse it as a `.jsonl` stream. Any malformed
    ///   lines are silently skipped so a partially-corrupt file does not
    ///   brick the TUI.
    /// * If `path` does not exist, look for a sibling file literally
    ///   named `history` (the rustyline default) and — if present —
    ///   migrate it one-shot, write the new `.jsonl` snapshot, then
    ///   delete the legacy file.
    /// * Otherwise return an empty history bound to `path`.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, HistoryError> {
        Self::load_with_max(path, DEFAULT_MAX)
    }

    /// Like [`History::load`] but with an explicit eviction cap.
    pub fn load_with_max(path: impl Into<PathBuf>, max: usize) -> Result<Self, HistoryError> {
        let path = path.into();
        let max = max.max(1);
        let mut entries: VecDeque<HistoryRecord> = VecDeque::new();

        if path.exists() {
            let file = fs::File::open(&path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(rec) = serde_json::from_str::<HistoryRecord>(&line) {
                    entries.push_back(rec);
                }
            }
        } else if let Some(parent) = path.parent() {
            let legacy = parent.join("history");
            if legacy.exists() {
                let imported = migrate_from_rustyline(&legacy)?;
                for text in imported {
                    entries.push_back(HistoryRecord::now(text, "normal"));
                }
                // Persist the migrated snapshot before deleting the
                // legacy file — if the write fails we keep the legacy
                // file so the user has a recovery path.
                Self::write_all(&path, &entries)?;
                fs::remove_file(&legacy)?;
            }
        }

        while entries.len() > max {
            entries.pop_front();
        }

        Ok(Self { path, entries, max })
    }

    /// Append a new record to both memory and disk.
    pub fn append(&mut self, record: HistoryRecord) -> Result<(), HistoryError> {
        let mut encoded = serde_json::to_vec(&record)?;
        encoded.push(b'\n');
        self.entries.push_back(record);
        let mut evicted = false;
        while self.entries.len() > self.max {
            self.entries.pop_front();
            evicted = true;
        }

        // If we evicted, rewrite the whole file to stay in sync.
        if evicted {
            Self::write_all(&self.path, &self.entries)?;
            return Ok(());
        }

        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Reverse substring search. Most-recent matches come first.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&HistoryRecord> {
        if query.is_empty() {
            return self.entries.iter().rev().collect();
        }
        self.entries
            .iter()
            .rev()
            .filter(|rec| rec.text.contains(query))
            .collect()
    }

    /// Number of records currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if there are no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Underlying disk path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Eviction cap.
    #[must_use]
    pub const fn max(&self) -> usize {
        self.max
    }

    /// Read-only slice of records in insertion order.
    #[must_use]
    pub fn entries(&self) -> Vec<&HistoryRecord> {
        self.entries.iter().collect()
    }

    fn write_all(path: &Path, entries: &VecDeque<HistoryRecord>) -> Result<(), HistoryError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut contents = Vec::new();
        for rec in entries {
            serde_json::to_writer(&mut contents, rec)?;
            contents.push(b'\n');
        }
        runtime::file_ops::replace_file_atomic(path, &contents)?;
        Ok(())
    }
}

/// Parse a rustyline-format history file into plain text entries.
///
/// Rustyline's `DefaultHistory` writes an optional `#V2` header line
/// followed by one entry per line, escaping backslashes as `\\` and
/// newlines as `\n`. Lines beginning with `#` are treated as comments
/// / headers and skipped. Empty lines are dropped.
pub fn migrate_from_rustyline(legacy: &Path) -> io::Result<Vec<String>> {
    let file = fs::File::open(legacy)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        out.push(unescape_rustyline(&line));
    }
    Ok(out)
}

fn unescape_rustyline(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            #[allow(clippy::match_same_arms)]
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("prompt-history-{label}-{nanos}"))
    }

    #[cfg(unix)]
    #[test]
    fn prompt_compaction_failure_preserves_previous_history_bytes() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_path("compaction-failure");
        fs::create_dir_all(&dir).expect("create history directory");
        let path = dir.join("prompts.jsonl");
        let mut history = History::load_with_max(&path, 1).expect("load history");
        history
            .append(HistoryRecord::now("old".to_string(), "normal"))
            .expect("seed history");
        let before = fs::read(&path).expect("read seeded history");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555))
            .expect("make history directory read-only");

        let probe = dir.join("probe");
        if fs::write(&probe, b"probe").is_ok() {
            let _ = fs::remove_file(probe);
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755))
                .expect("restore history directory permissions");
            let _ = fs::remove_dir_all(dir);
            return;
        }

        let result = history.append(HistoryRecord::now("new".to_string(), "normal"));

        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755))
            .expect("restore history directory permissions");
        let after = fs::read(&path).expect("read history after failed compaction");
        let _ = fs::remove_dir_all(dir);
        assert!(result.is_err(), "allocating the sibling temp must fail");
        assert_eq!(after, before, "failed compaction must preserve prior history");
    }
}
