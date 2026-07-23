//! Consumer for `ScheduleWakeup` requests — the missing other half.
//!
//! The `ScheduleWakeup` tool (tools crate) writes `.zo/wakeups/wakeup-*.json`
//! files describing a delayed re-invocation:
//! `{ delaySeconds: f64, reason, prompt, scheduledAt: "<epoch secs>", sessionId? }`.
//! Nothing read them back, so the promised "wake me in N seconds" never fired —
//! the file was written and orphaned.
//!
//! This module drains *due* files: the TUI session loop calls [`scan_due_wakeups`]
//! each idle pass, re-injects each due `prompt` as a fresh user turn, and deletes
//! the file. [`next_wakeup_due_in`] lets the idle `select!` sleep exactly until
//! the next one fires instead of waiting for a keystroke (the part that made the
//! alarm feel dead even when a key was eventually pressed).

use std::path::{Path, PathBuf};
use std::time::Duration;

const WAKEUPS_DIR: &str = "wakeups";

// Files past this grace period are garbage-collected regardless of ownership.
const STALE_WAKEUP_SECS: u64 = 7 * 24 * 60 * 60;

/// How soon to re-poll when an already-due wakeup could not be drained yet (e.g.
/// the turn queue was momentarily full). Short enough to feel instant, long
/// enough to never busy-loop at 0s.
const OVERDUE_RETRY: Duration = Duration::from_secs(1);

/// Current Unix time in whole seconds — matches the `scheduledAt` epoch-seconds
/// string `ScheduleWakeup` writes (`epoch_seconds_now`, which uses `as_secs()`).
pub(crate) fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(crate) fn wakeups_dir() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    runtime::zo_state_base(&cwd).join(".zo").join(WAKEUPS_DIR)
}

/// A due wakeup ready to be re-injected as a turn.
pub(crate) struct DueWakeup {
    pub prompt: String,
    pub reason: String,
    pub file: PathBuf,
}

/// Nearest scheduled wakeup for the TUI's live countdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScheduledWakeupInfo {
    pub due_in: Duration,
    pub reason: String,
}

/// The fields a consumer needs from a wakeup file.
struct Parsed {
    scheduled_at: u64,
    delay_seconds: f64,
    prompt: String,
    reason: String,
    session_id: Option<String>,
}

/// Absolute epoch-second the wakeup becomes due. Sub-second delay truncates —
/// `scheduledAt` itself only has 1-second resolution, so it cannot matter; the
/// `max(0.0)` rules out a negative cast and delays are bounded seconds in
/// practice, so the truncating `as u64` is exact for any real wakeup.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn due_at(p: &Parsed) -> u64 {
    p.scheduled_at
        .saturating_add(p.delay_seconds.max(0.0) as u64)
}

/// Parse a wakeup file. `None` on corrupt JSON, a missing/invalid `scheduledAt`,
/// or an empty `prompt` — all unrecoverable, so the caller deletes the file
/// rather than retrying a wakeup that can never fire.
fn parse_wakeup(content: &str) -> Option<Parsed> {
    let val: serde_json::Value = serde_json::from_str(content).ok()?;
    let scheduled_at = val.get("scheduledAt")?.as_str()?.parse::<u64>().ok()?;
    let delay_seconds = val
        .get("delaySeconds")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let prompt = val.get("prompt")?.as_str()?.to_string();
    if prompt.trim().is_empty() {
        return None;
    }
    let reason = val
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let session_id = match val.get("sessionId") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(session_id)) => Some(session_id.clone()),
        Some(_) => return None,
    };
    Some(Parsed {
        scheduled_at,
        delay_seconds,
        prompt,
        reason,
        session_id,
    })
}

fn belongs_to_session(parsed: &Parsed, session_id: Option<&str>) -> bool {
    // Pre-upgrade files have no owner and remain first-wins so scheduled work
    // is not lost while multiple versions coexist in the same directory.
    parsed.session_id.as_deref().is_none_or(|owner| session_id == Some(owner))
}

fn is_stale(parsed: &Parsed, now_epoch: u64) -> bool {
    now_epoch.saturating_sub(due_at(parsed)) > STALE_WAKEUP_SECS
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("json"))
        .collect()
}

/// Due wakeups in `.zo/wakeups`, FIFO by `scheduledAt`, leaving not-yet-due
/// files in place. Corrupt/unusable files are deleted as a side effect — they
/// can never fire, so retrying them forever would just leak.
pub(crate) fn scan_due_wakeups(now_epoch: u64, session_id: Option<&str>) -> Vec<DueWakeup> {
    scan_due_in_dir(&wakeups_dir(), now_epoch, session_id)
}

fn scan_due_in_dir(dir: &Path, now_epoch: u64, session_id: Option<&str>) -> Vec<DueWakeup> {
    let mut due: Vec<(u64, DueWakeup)> = Vec::new();
    for path in json_files(dir) {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Some(parsed) = parse_wakeup(&content) else {
            let _ = std::fs::remove_file(&path);
            continue;
        };
        if is_stale(&parsed, now_epoch) {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        if !belongs_to_session(&parsed, session_id) {
            continue;
        }
        if due_at(&parsed) <= now_epoch {
            due.push((
                parsed.scheduled_at,
                DueWakeup {
                    prompt: parsed.prompt,
                    reason: parsed.reason,
                    file: path,
                },
            ));
        }
    }
    due.sort_by_key(|(scheduled, _)| *scheduled);
    due.into_iter().map(|(_, w)| w).collect()
}

/// Duration the idle `select!` should sleep before re-checking wakeups, or
/// `None` when none are pending. A future wakeup sleeps until it fires; an
/// already-due one that the loop-top drain could not clear yet (e.g. a full turn
/// queue) still arms the timer on a short retry — without this, an overdue file
/// would never wake the idle prompt and would wait for a keystroke (the original
/// bug's tail).
pub(crate) fn next_wakeup_due_in(
    now_epoch: u64,
    session_id: Option<&str>,
) -> Option<Duration> {
    next_due_in_dir(&wakeups_dir(), now_epoch, session_id)
}

/// Nearest pending wakeup and its user-facing reason. Unlike the idle retry
/// timer, an overdue wakeup reports zero so the HUD can render `wake now`.
pub(crate) fn next_wakeup_info(
    now_epoch: u64,
    session_id: Option<&str>,
) -> Option<ScheduledWakeupInfo> {
    next_info_in_dir(&wakeups_dir(), now_epoch, session_id)
}

fn next_info_in_dir(
    dir: &Path,
    now_epoch: u64,
    session_id: Option<&str>,
) -> Option<ScheduledWakeupInfo> {
    json_files(dir)
        .into_iter()
        .filter_map(|path| {
            let content = std::fs::read_to_string(&path).ok()?;
            let parsed = parse_wakeup(&content)?;
            if is_stale(&parsed, now_epoch) {
                let _ = std::fs::remove_file(&path);
                return None;
            }
            if !belongs_to_session(&parsed, session_id) {
                return None;
            }
            Some((due_at(&parsed), parsed.reason.trim().to_string()))
        })
        .min_by_key(|(at, _)| *at)
        .map(|(at, reason)| ScheduledWakeupInfo {
            due_in: Duration::from_secs(at.saturating_sub(now_epoch)),
            reason,
        })
}

fn next_due_in_dir(dir: &Path, now_epoch: u64, session_id: Option<&str>) -> Option<Duration> {
    let mut earliest: Option<Duration> = None;
    for path in json_files(dir) {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Some(parsed) = parse_wakeup(&content) else {
            continue;
        };
        if is_stale(&parsed, now_epoch) {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        if !belongs_to_session(&parsed, session_id) {
            continue;
        }
        let at = due_at(&parsed);
        let wait = if at > now_epoch {
            Duration::from_secs(at - now_epoch)
        } else {
            OVERDUE_RETRY
        };
        earliest = Some(earliest.map_or(wait, |e| e.min(wait)));
    }
    earliest
}

/// Consume a fired wakeup file so it never fires twice. If the unlink fails
/// (rare: permissions / unmounted dir), blank the file instead — the next scan
/// then parses an empty prompt, treats it as unusable, and drops it. Without
/// this fallback an undeletable file would re-queue the same prompt every pass,
/// forever.
pub(crate) fn consume_wakeup_file(file: &Path) {
    if std::fs::remove_file(file).is_ok() {
        return;
    }
    let _ = std::fs::write(file, "");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wakeup_json(scheduled_at: &str, delay: f64, prompt: &str) -> String {
        wakeup_json_with_reason(scheduled_at, delay, prompt, "r")
    }

    fn wakeup_json_with_reason(
        scheduled_at: &str,
        delay: f64,
        prompt: &str,
        reason: &str,
    ) -> String {
        format!(
            r#"{{"delaySeconds":{delay},"reason":"{reason}","prompt":"{prompt}","scheduledAt":"{scheduled_at}"}}"#
        )
    }

    fn stamped_wakeup_json(
        scheduled_at: &str,
        delay: f64,
        prompt: &str,
        reason: &str,
        session_id: &str,
    ) -> String {
        format!(
            r#"{{"delaySeconds":{delay},"reason":"{reason}","prompt":"{prompt}","scheduledAt":"{scheduled_at}","sessionId":"{session_id}"}}"#
        )
    }

    #[test]
    fn wakeup_store_is_nested_under_zo_state_dir() {
        let _guard = crate::test_env_lock();
        let prior = std::env::var_os("ZO_STATE_DIR");
        let root = std::env::temp_dir().join(format!("zo-wakeups-{}", std::process::id()));
        std::env::set_var("ZO_STATE_DIR", &root);
        let actual = wakeups_dir();
        match prior {
            Some(value) => std::env::set_var("ZO_STATE_DIR", value),
            None => std::env::remove_var("ZO_STATE_DIR"),
        }

        assert_eq!(actual, root.join(".zo").join("wakeups"));
    }

    #[test]
    fn parses_fields_and_computes_due_at() {
        let p = parse_wakeup(&wakeup_json("1000", 300.0, "check CI")).expect("parses");
        assert_eq!(p.scheduled_at, 1000);
        assert_eq!(p.prompt, "check CI");
        assert_eq!(p.session_id, None);
        assert_eq!(due_at(&p), 1300);
    }

    #[test]
    fn rejects_corrupt_empty_or_bad_timestamp() {
        assert!(parse_wakeup("not json").is_none());
        // scheduledAt not a number
        assert!(parse_wakeup(&wakeup_json("x", 1.0, "p")).is_none());
        // empty/whitespace prompt
        assert!(parse_wakeup(&wakeup_json("1", 1.0, "   ")).is_none());
        // missing scheduledAt entirely
        assert!(parse_wakeup(r#"{"delaySeconds":1.0,"prompt":"p"}"#).is_none());
    }

    #[test]
    fn zero_and_fractional_delay_truncate_to_seconds() {
        assert_eq!(due_at(&parse_wakeup(&wakeup_json("100", 0.0, "p")).unwrap()), 100);
        // 0.9s rounds down — scheduledAt is 1s-resolution anyway.
        assert_eq!(due_at(&parse_wakeup(&wakeup_json("100", 0.9, "p")).unwrap()), 100);
    }

    #[test]
    fn scan_returns_only_due_files_fifo_and_drops_corrupt() {
        let dir = std::env::temp_dir().join(format!("zo-wakeup-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");

        // Two due (different scheduledAt → FIFO order), one future, one corrupt.
        std::fs::write(dir.join("wakeup-b.json"), wakeup_json("200", 10.0, "second")).unwrap();
        std::fs::write(dir.join("wakeup-a.json"), wakeup_json("100", 10.0, "first")).unwrap();
        std::fs::write(dir.join("wakeup-future.json"), wakeup_json("100000", 60.0, "later")).unwrap();
        std::fs::write(dir.join("wakeup-corrupt.json"), "{ not json").unwrap();

        let now = 100_000; // 100000s: the "100"/"200" files are long overdue.
        let due = scan_due_in_dir(&dir, now, None);
        let prompts: Vec<&str> = due.iter().map(|d| d.prompt.as_str()).collect();
        assert_eq!(prompts, vec!["first", "second"], "due files FIFO by scheduledAt");

        // Corrupt file is deleted; future file remains.
        assert!(!dir.join("wakeup-corrupt.json").exists(), "corrupt file dropped");
        assert!(dir.join("wakeup-future.json").exists(), "future file kept");

        // Consume the drained files (as the loop does after queueing each). Until
        // consumed they stay on disk, so next_due would (correctly) report a
        // short retry for them — drain first, then only the future file remains.
        for w in &due {
            super::consume_wakeup_file(&w.file);
        }

        // next_due_in now points at the future file (scheduledAt 100000 + 60 - now).
        let next = next_due_in_dir(&dir, now, None).expect("a future wakeup is pending");
        assert_eq!(next, Duration::from_secs(60));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stamped_wakeup_is_invisible_to_a_different_session() {
        let dir = std::env::temp_dir().join(format!(
            "zo-wakeup-foreign-session-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("wakeup-foreign.json");
        std::fs::write(
            &file,
            stamped_wakeup_json("100", 10.0, "foreign", "other work", "session-b"),
        )
        .unwrap();

        assert!(scan_due_in_dir(&dir, 1_000, Some("session-a")).is_empty());
        assert_eq!(next_info_in_dir(&dir, 100, Some("session-a")), None);
        assert_eq!(next_due_in_dir(&dir, 100, Some("session-a")), None);
        assert!(file.exists(), "a foreign session must not consume the file");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_removes_only_long_overdue_foreign_wakeups() {
        let dir = std::env::temp_dir().join(format!(
            "zo-wakeup-stale-foreign-session-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let stale = dir.join("wakeup-stale.json");
        let fresh = dir.join("wakeup-fresh.json");
        let now = 1_000_000;
        let stale_due = now - 8 * 24 * 60 * 60;
        let fresh_due = now - 24 * 60 * 60;
        std::fs::write(
            &stale,
            stamped_wakeup_json(
                &stale_due.to_string(),
                0.0,
                "stale foreign",
                "old work",
                "session-b",
            ),
        )
        .unwrap();
        std::fs::write(
            &fresh,
            stamped_wakeup_json(
                &fresh_due.to_string(),
                0.0,
                "fresh foreign",
                "recent work",
                "session-b",
            ),
        )
        .unwrap();

        assert!(scan_due_in_dir(&dir, now, Some("session-a")).is_empty());
        assert!(!stale.exists(), "an eight-days-overdue wakeup is stale");
        assert!(fresh.exists(), "a recently-due foreign wakeup must remain");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn own_session_stamped_wakeup_fires() {
        let dir =
            std::env::temp_dir().join(format!("zo-wakeup-own-session-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("wakeup-own.json"),
            stamped_wakeup_json("100", 10.0, "mine", "own work", "session-a"),
        )
        .unwrap();

        let due = scan_due_in_dir(&dir, 1_000, Some("session-a"));
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].prompt, "mine");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_unstamped_wakeup_still_fires_in_any_session() {
        let dir = std::env::temp_dir().join(format!(
            "zo-wakeup-legacy-session-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("wakeup-legacy.json"),
            wakeup_json("100", 10.0, "legacy"),
        )
        .unwrap();

        // 업그레이드 전 파일은 소유자가 없으므로 기존 first-wins 동작을 유지한다.
        assert_eq!(scan_due_in_dir(&dir, 1_000, Some("session-a")).len(), 1);
        assert_eq!(scan_due_in_dir(&dir, 1_000, Some("session-b")).len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mixed_directory_returns_nearest_eligible_wakeup() {
        let dir = std::env::temp_dir().join(format!(
            "zo-wakeup-mixed-session-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("wakeup-foreign.json"),
            stamped_wakeup_json("100", 5.0, "foreign", "foreign nearest", "session-b"),
        )
        .unwrap();
        std::fs::write(
            dir.join("wakeup-legacy.json"),
            wakeup_json_with_reason("100", 20.0, "legacy", "legacy eligible"),
        )
        .unwrap();
        std::fs::write(
            dir.join("wakeup-own.json"),
            stamped_wakeup_json("100", 30.0, "mine", "own later", "session-a"),
        )
        .unwrap();

        assert_eq!(
            next_info_in_dir(&dir, 100, Some("session-a")),
            Some(ScheduledWakeupInfo {
                due_in: Duration::from_secs(20),
                reason: "legacy eligible".to_string(),
            })
        );
        assert_eq!(
            next_due_in_dir(&dir, 100, Some("session-a")),
            Some(Duration::from_secs(20))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn next_due_arms_short_retry_for_overdue_then_none_when_empty() {
        let dir = std::env::temp_dir().join(format!("zo-wakeup-overdue-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");

        // No files → no timer.
        assert_eq!(next_due_in_dir(&dir, 100, None), None);

        // An already-due file that the drain could not clear (e.g. full queue)
        // must still arm the timer on a short retry — never None — so it does not
        // wait for a keystroke.
        std::fs::write(dir.join("wakeup-overdue.json"), wakeup_json("1", 1.0, "do it")).unwrap();
        assert_eq!(next_due_in_dir(&dir, 100_000, None), Some(OVERDUE_RETRY));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn next_info_returns_nearest_wakeup_and_trimmed_reason() {
        let dir =
            std::env::temp_dir().join(format!("zo-wakeup-info-nearest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");

        std::fs::write(
            dir.join("wakeup-later.json"),
            wakeup_json_with_reason("100", 60.0, "later", " later check "),
        )
        .unwrap();
        std::fs::write(
            dir.join("wakeup-nearest.json"),
            wakeup_json_with_reason("100", 20.0, "nearest", "  check CI  "),
        )
        .unwrap();

        assert_eq!(
            next_info_in_dir(&dir, 105, None),
            Some(ScheduledWakeupInfo {
                due_in: Duration::from_secs(15),
                reason: "check CI".to_string(),
            })
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn next_info_reports_overdue_as_zero() {
        let dir =
            std::env::temp_dir().join(format!("zo-wakeup-info-overdue-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("wakeup-overdue.json"),
            wakeup_json_with_reason("10", 5.0, "run", "status check"),
        )
        .unwrap();

        assert_eq!(
            next_info_in_dir(&dir, 100, None),
            Some(ScheduledWakeupInfo {
                due_in: Duration::ZERO,
                reason: "status check".to_string(),
            })
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn next_info_skips_corrupt_files() {
        let dir =
            std::env::temp_dir().join(format!("zo-wakeup-info-corrupt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("wakeup-corrupt.json"), "{ nope").unwrap();
        std::fs::write(
            dir.join("wakeup-valid.json"),
            wakeup_json_with_reason("100", 10.0, "run", "valid"),
        )
        .unwrap();

        assert_eq!(
            next_info_in_dir(&dir, 100, None),
            Some(ScheduledWakeupInfo {
                due_in: Duration::from_secs(10),
                reason: "valid".to_string(),
            })
        );
        assert!(dir.join("wakeup-corrupt.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn consume_removes_the_file() {
        let dir = std::env::temp_dir().join(format!("zo-wakeup-consume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("wakeup-x.json");
        std::fs::write(&file, wakeup_json("1", 1.0, "p")).unwrap();
        super::consume_wakeup_file(&file);
        assert!(!file.exists(), "consumed wakeup file is removed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
