use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const DEFAULT_MAX: usize = 500;

#[derive(Debug, thiserror::Error)]
pub enum CommandHistoryError {
    #[error("command history io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("command history json error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UsageRecord {
    ts: u64,
    command: String,
}

#[derive(Debug)]
pub struct CommandHistory {
    path: PathBuf,
    entries: VecDeque<UsageRecord>,
    max: usize,
}

impl CommandHistory {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, CommandHistoryError> {
        let path = path.into();
        let mut entries = VecDeque::new();

        if path.exists() {
            let file = fs::File::open(&path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(rec) = serde_json::from_str::<UsageRecord>(&line) {
                    entries.push_back(rec);
                }
            }
        }

        while entries.len() > DEFAULT_MAX {
            entries.pop_front();
        }

        Ok(Self {
            path,
            entries,
            max: DEFAULT_MAX,
        })
    }

    pub fn record(&mut self, command: &str) -> Result<(), CommandHistoryError> {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        let rec = UsageRecord {
            ts,
            command: command.to_string(),
        };
        let mut encoded = serde_json::to_vec(&rec)?;
        encoded.push(b'\n');
        self.entries.push_back(rec);

        let mut evicted = false;
        while self.entries.len() > self.max {
            self.entries.pop_front();
            evicted = true;
        }

        if evicted {
            self.write_all()?;
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

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn frecency_scores(&self) -> HashMap<String, f64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        let mut scores: HashMap<String, f64> = HashMap::new();
        for rec in &self.entries {
            let age_secs = now.saturating_sub(rec.ts);
            let age_hours = age_secs as f64 / 3600.0;
            let decay = 1.0 / (1.0 + age_hours / 24.0);
            *scores.entry(rec.command.clone()).or_default() += decay;
        }
        scores
    }

    #[must_use]
    pub fn top_recent(&self, limit: usize) -> Vec<String> {
        let scores = self.frecency_scores();
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(limit).map(|(cmd, _)| cmd).collect()
    }

    fn write_all(&self) -> Result<(), CommandHistoryError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut contents = Vec::new();
        for rec in &self.entries {
            serde_json::to_writer(&mut contents, rec)?;
            contents.push(b'\n');
        }
        runtime::file_ops::replace_file_atomic(&self.path, &contents)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cmd-history-{label}-{nanos}.jsonl"))
    }

    #[test]
    fn records_and_scores() {
        let path = temp_path("scores");
        let mut hist = CommandHistory::load(&path).unwrap();
        hist.record("/commit").unwrap();
        hist.record("/commit").unwrap();
        hist.record("/status").unwrap();

        let scores = hist.frecency_scores();
        assert!(scores["/commit"] > scores["/status"]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn top_recent_returns_ranked() {
        let path = temp_path("recent");
        let mut hist = CommandHistory::load(&path).unwrap();
        hist.record("/commit").unwrap();
        hist.record("/commit").unwrap();
        hist.record("/commit").unwrap();
        hist.record("/status").unwrap();

        let top = hist.top_recent(2);
        assert_eq!(top[0], "/commit");
        assert!(top.len() <= 2);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn persists_and_reloads() {
        let path = temp_path("persist");
        {
            let mut hist = CommandHistory::load(&path).unwrap();
            hist.record("/commit").unwrap();
            hist.record("/diff").unwrap();
        }
        let hist = CommandHistory::load(&path).unwrap();
        assert_eq!(hist.entries.len(), 2);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn empty_history_returns_empty_scores() {
        let path = temp_path("empty");
        let hist = CommandHistory::load(&path).unwrap();
        assert!(hist.frecency_scores().is_empty());
        assert!(hist.top_recent(5).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn compaction_failure_preserves_previous_history_bytes() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_path("compaction-failure");
        fs::create_dir_all(&dir).expect("create history directory");
        let path = dir.join("commands.jsonl");
        let mut history = CommandHistory::load(&path).expect("load history");
        history.max = 1;
        history.record("/old").expect("seed history");
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

        let result = history.record("/new");

        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755))
            .expect("restore history directory permissions");
        let after = fs::read(&path).expect("read history after failed compaction");
        let _ = fs::remove_dir_all(dir);
        assert!(result.is_err(), "allocating the sibling temp must fail");
        assert_eq!(after, before, "failed compaction must preserve prior history");
    }
}
