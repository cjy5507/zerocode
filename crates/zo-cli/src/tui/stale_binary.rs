//! Detect when the running `zo` binary has been replaced on disk.
//!
//! The recurring "★미배포·재시작필요" churn has one root cause: a redeploy
//! swaps the file backing the *running* process, but nothing tells the user
//! their live session is still executing the old code. This module captures the
//! boot-time identity of [`current_exe`](std::env::current_exe) and, on a
//! throttled cadence, re-`stat`s that same path — a changed identity means a new
//! build is on disk and the session must restart to pick it up.
//!
//! Identity is the `(device, inode, mtime)` triple. `rm`+`cp` and `install -m`
//! both allocate a new inode; an in-place overwrite keeps the inode but bumps
//! mtime — comparing the whole triple catches all three, while never firing on a
//! mere re-`stat` of the same unchanged file. mtime is the *supplementary*
//! signal (a bare `touch` would move it without a real rebuild), but inode is the
//! primary one, so the combined compare is both sensitive and precise.
//!
//! Detection is fail-open at every step: if [`current_exe`](std::env::current_exe)
//! or the `stat` fails, the watch is simply disabled — it never blocks boot and
//! never raises a false "restart" nag.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A new build was found on disk backing the running process.
///
/// Carries only what the sidebar can honestly show: the *disk file's* mtime. The
/// running binary's build SHA / date are compile-time constants baked into this
/// process and describe the old code, not the replacement, so they cannot label
/// the new file — the disk mtime is the one true fact about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleBinaryInfo {
    /// mtime of the replacement file on disk, in seconds since the Unix epoch.
    pub disk_mtime: i64,
}

impl StaleBinaryInfo {
    /// The always-on sidebar warning line, e.g.
    /// `/restart · new build on disk (2026-07-10)`.
    ///
    /// Ordered action-first: the sidebar truncates long lines at narrow widths,
    /// and the date is the part that can afford to be cut — the earlier
    /// date-first ordering hid the `/restart` cue exactly when it mattered. The
    /// date is derived from [`Self::disk_mtime`]; the `/restart` cue names the
    /// action without this module owning that command (a separate concern).
    #[must_use]
    pub fn sidebar_label(&self) -> String {
        format!("/restart \u{00b7} new build on disk ({})", self.disk_date())
    }

    /// The one-shot transcript notice pushed at the first turn boundary after a
    /// redeploy is detected (see [`take_newly_stale`]) — a full sentence, since
    /// the transcript does not truncate like the sidebar does.
    #[must_use]
    pub fn transcript_notice(&self) -> String {
        format!(
            "[update] new zo build on disk ({}) \u{2014} run /restart to apply it to this session",
            self.disk_date()
        )
    }

    fn disk_date(&self) -> String {
        core_types::date::utc_date_from_unix_secs(u64::try_from(self.disk_mtime).unwrap_or(0))
    }
}

/// The `(device, inode, mtime)` triple identifying a file on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BinaryIdentity {
    dev: u64,
    ino: u64,
    mtime: i64,
}

impl BinaryIdentity {
    /// `stat` `path` into an identity, or `None` when it cannot be read (missing,
    /// permission denied, or the brief mid-deploy rename window). Callers treat
    /// `None` as *inconclusive*, never as "stale".
    #[cfg(unix)]
    fn read(path: &Path) -> Option<Self> {
        use std::os::unix::fs::MetadataExt as _;
        let meta = std::fs::metadata(path).ok()?;
        Some(Self {
            dev: meta.dev(),
            ino: meta.ino(),
            mtime: meta.mtime(),
        })
    }

    /// Non-unix fallback: no cheap `(dev, ino, mtime)` triple, so detection is
    /// disabled (fail-open). zo deploys to unix hosts; this keeps the crate
    /// compiling everywhere without a false signal.
    #[cfg(not(unix))]
    fn read(_path: &Path) -> Option<Self> {
        None
    }
}

/// Throttled, latch-once watch over one executable path.
///
/// Testable in isolation: [`Self::check_at`] takes an injected `now` and
/// re-`stat`s the stored path, while [`Self::observe`] is the pure latch
/// decision over an already-read identity — so a unit test can exercise the
/// `(dev, ino, mtime)` sensitivity with hand-built identities and drive the
/// clock, without depending on the real [`current_exe`](std::env::current_exe)
/// or on the OS to set an arbitrary mtime.
struct StaleWatch {
    path: PathBuf,
    baseline: BinaryIdentity,
    throttle: Duration,
    last_check: Option<Instant>,
    stale: Option<StaleBinaryInfo>,
    /// Whether the one-shot transcript notice has been handed out (see
    /// [`Self::take_newly_stale_at`]). Independent of the `stale` latch: the
    /// sidebar badge stays on forever, the notice fires exactly once.
    reported: bool,
}

impl StaleWatch {
    /// Capture the boot-time baseline for `path`. `None` (detection disabled)
    /// when the path cannot be `stat`ed — fail-open.
    fn capture(path: PathBuf, throttle: Duration) -> Option<Self> {
        let baseline = BinaryIdentity::read(&path)?;
        Some(Self {
            path,
            baseline,
            throttle,
            last_check: None,
            stale: None,
            reported: false,
        })
    }

    /// Capture the baseline for the running process's own executable.
    fn capture_current_exe(throttle: Duration) -> Option<Self> {
        Self::capture(std::env::current_exe().ok()?, throttle)
    }

    /// Re-`stat` the path (subject to the throttle) and latch `stale` the first
    /// time the on-disk identity diverges from the boot baseline.
    ///
    /// A deploy is one-way, so once latched the result never reverts and later
    /// calls skip the `stat` entirely. Between checks the throttle short-circuits
    /// so a burst of quick turns cannot `stat` the binary many times a second.
    /// Returns the current stale info (`None` = not stale, or not yet due).
    fn check_at(&mut self, now: Instant) -> Option<StaleBinaryInfo> {
        if let Some(info) = &self.stale {
            return Some(info.clone());
        }
        if let Some(last) = self.last_check {
            if now.duration_since(last) < self.throttle {
                return None;
            }
        }
        self.last_check = Some(now);
        // A vanished path (the brief unlink→create window of `install`/`cp`) is
        // inconclusive, not stale: wait for the next tick to see the new file.
        let current = BinaryIdentity::read(&self.path)?;
        self.observe(current)
    }

    /// Pure latch decision over a freshly-read identity. Splitting this from the
    /// filesystem read makes the triple's sensitivity deterministically testable
    /// (no reliance on the OS to move an mtime). Latches `stale` on the first
    /// divergence and stamps the disk mtime as the reported build time.
    fn observe(&mut self, current: BinaryIdentity) -> Option<StaleBinaryInfo> {
        if current == self.baseline {
            return None;
        }
        let info = StaleBinaryInfo {
            disk_mtime: current.mtime,
        };
        self.stale = Some(info.clone());
        Some(info)
    }

    /// [`Self::check_at`] plus a report-once latch: `Some` exactly the first
    /// time this is called while stale, `None` on every later call. Drives the
    /// one-shot transcript notice at turn boundaries without suppressing the
    /// always-on sidebar badge (which keeps reading the `stale` latch).
    fn take_newly_stale_at(&mut self, now: Instant) -> Option<StaleBinaryInfo> {
        let info = self.check_at(now)?;
        if self.reported {
            return None;
        }
        self.reported = true;
        Some(info)
    }
}

/// Minimum spacing between `stat` calls. The check runs on the idle tick and at
/// each turn boundary; without a floor a burst of quick turns would `stat` the
/// binary many times a second for no benefit. One deploy will not be missed by a
/// 5-second window.
const CHECK_THROTTLE: Duration = Duration::from_secs(5);

/// Process-global watch. `Some` = detection active; `None` = disabled
/// (fail-open, e.g. [`current_exe`](std::env::current_exe) unavailable).
static WATCH: OnceLock<Mutex<Option<StaleWatch>>> = OnceLock::new();

fn watch() -> &'static Mutex<Option<StaleWatch>> {
    WATCH.get_or_init(|| Mutex::new(StaleWatch::capture_current_exe(CHECK_THROTTLE)))
}

/// Pin the boot-time identity of [`current_exe`](std::env::current_exe).
///
/// Idempotent and optional — [`check`] lazily initializes on first use — but
/// calling it once at REPL startup fixes the baseline as early as possible,
/// before any redeploy can race the first HUD build. Fail-open: a
/// [`current_exe`](std::env::current_exe) error just leaves detection disabled.
pub fn init() {
    let _ = watch();
}

/// The current stale-binary status for the running process.
///
/// Throttled to one `stat` per [`CHECK_THROTTLE`] and latched once tripped, so
/// it is cheap to call on every HUD build. `None` = the running binary still
/// matches the file on disk, the throttle has not elapsed, or detection is
/// disabled.
#[must_use]
pub fn check() -> Option<StaleBinaryInfo> {
    // A poisoned lock (a panic mid-check) fails open rather than propagating.
    let mut guard = watch().lock().ok()?;
    guard.as_mut()?.check_at(Instant::now())
}

/// `Some` exactly once — the first call made after the running binary is found
/// replaced on disk. Turn boundaries call this to push a one-shot transcript
/// notice; the sidebar badge from [`check`] is unaffected and stays on.
#[must_use]
pub fn take_newly_stale() -> Option<StaleBinaryInfo> {
    let mut guard = watch().lock().ok()?;
    guard.as_mut()?.take_newly_stale_at(Instant::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("zo-stale-binary-{tag}-{nanos}"))
    }

    fn write_file(path: &Path, contents: &str) {
        fs::write(path, contents).expect("write temp binary stand-in");
    }

    /// A watch whose baseline is injected directly, so [`StaleWatch::observe`]
    /// can be driven with hand-built identities — the path is never read.
    fn watch_with_baseline(baseline: BinaryIdentity) -> StaleWatch {
        StaleWatch {
            path: PathBuf::from("/zo-stale-binary/does-not-exist"),
            baseline,
            throttle: Duration::from_secs(0),
            last_check: None,
            stale: None,
            reported: false,
        }
    }

    const BASE: BinaryIdentity = BinaryIdentity {
        dev: 1,
        ino: 42,
        mtime: 1_000,
    };

    #[test]
    fn identical_identity_is_not_stale() {
        let mut watch = watch_with_baseline(BASE);
        assert_eq!(watch.observe(BASE), None);
        // Idempotent: a second identical observation still reads clean.
        assert_eq!(watch.observe(BASE), None);
    }

    #[test]
    fn inode_change_is_stale() {
        // `rm` + `cp` / `install -m 755`: the path gets a brand-new inode.
        let mut watch = watch_with_baseline(BASE);
        let replaced = BinaryIdentity { ino: 99, ..BASE };
        assert_eq!(
            watch.observe(replaced),
            Some(StaleBinaryInfo { disk_mtime: 1_000 })
        );
    }

    #[test]
    fn mtime_change_is_stale() {
        // In-place overwrite keeps the inode but moves mtime; the triple still
        // diverges, so this is caught too (mtime is the supplementary signal).
        let mut watch = watch_with_baseline(BASE);
        let bumped = BinaryIdentity {
            mtime: 2_000,
            ..BASE
        };
        assert_eq!(
            watch.observe(bumped),
            Some(StaleBinaryInfo { disk_mtime: 2_000 })
        );
    }

    #[test]
    fn device_change_is_stale() {
        // A path that moved across filesystems (dev differs) is stale too — the
        // full triple is compared, not inode alone.
        let mut watch = watch_with_baseline(BASE);
        let moved = BinaryIdentity { dev: 7, ..BASE };
        assert!(watch.observe(moved).is_some());
    }

    #[test]
    fn end_to_end_inode_replacement_latches_stale() {
        // Real filesystem path: capture, replace (new inode), and confirm the
        // read+compare pipeline latches and stays latched even if the file later
        // vanishes (a deploy is one-way; the read is skipped once latched).
        let path = unique_tmp("e2e");
        write_file(&path, "old build");
        let mut watch =
            StaleWatch::capture(path.clone(), Duration::from_secs(0)).expect("baseline captured");
        assert_eq!(watch.check_at(Instant::now()), None);
        fs::remove_file(&path).expect("unlink old");
        write_file(&path, "new build");
        let stale = watch
            .check_at(Instant::now())
            .expect("a replaced inode must read as stale");
        // Delete the file entirely: the latch means later checks stay stale
        // without re-`stat`ing (they would otherwise read `None` = inconclusive).
        fs::remove_file(&path).expect("unlink new");
        assert_eq!(watch.check_at(Instant::now()), Some(stale));
    }

    #[test]
    fn missing_path_disables_detection() {
        // No file at boot → no baseline → detection disabled, never a false nag.
        let path = unique_tmp("missing");
        assert!(
            StaleWatch::capture(path, Duration::from_secs(0)).is_none(),
            "capturing a non-existent path must fail open (disabled), not panic"
        );
    }

    #[test]
    fn throttle_skips_the_stat_until_the_window_elapses() {
        let path = unique_tmp("throttle");
        write_file(&path, "v1");
        let throttle = Duration::from_secs(5);
        let mut watch = StaleWatch::capture(path.clone(), throttle).expect("baseline captured");
        let t0 = Instant::now();
        // First check performs the stat (no prior check to throttle against).
        assert_eq!(watch.check_at(t0), None);
        // Replace the file so a fresh stat *would* see a new inode.
        fs::remove_file(&path).expect("unlink");
        write_file(&path, "v2");
        // Within the throttle window the stat is skipped, so the change is not
        // yet observed — this is the load-bearing throttle behavior.
        assert_eq!(
            watch.check_at(t0 + Duration::from_secs(1)),
            None,
            "a check inside the throttle window must not re-stat"
        );
        // Past the window the stat runs again and the change is detected.
        assert!(
            watch.check_at(t0 + Duration::from_secs(6)).is_some(),
            "once the throttle elapses the replacement is detected"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn sidebar_label_leads_with_restart_then_names_the_disk_date() {
        // 2026-07-10 00:00:00 UTC.
        let info = StaleBinaryInfo {
            disk_mtime: 1_783_641_600,
        };
        let label = info.sidebar_label();
        // Action-first: narrow sidebars truncate the tail, and the date is the
        // expendable part — `/restart` must survive the cut.
        assert!(label.starts_with("/restart"), "{label:?}");
        assert!(label.contains("new build on disk"), "{label:?}");
        assert!(label.contains("2026-07-10"), "{label:?}");
    }

    #[test]
    fn transcript_notice_names_the_date_and_restart() {
        let info = StaleBinaryInfo {
            disk_mtime: 1_783_641_600,
        };
        let notice = info.transcript_notice();
        assert!(notice.starts_with("[update]"), "{notice:?}");
        assert!(notice.contains("2026-07-10"), "{notice:?}");
        assert!(notice.contains("/restart"), "{notice:?}");
    }

    #[test]
    fn take_newly_stale_fires_once_while_check_stays_latched() {
        let path = unique_tmp("take-once");
        write_file(&path, "old build");
        let mut watch =
            StaleWatch::capture(path.clone(), Duration::from_secs(0)).expect("baseline captured");
        // Not stale yet → nothing to report.
        assert_eq!(watch.take_newly_stale_at(Instant::now()), None);
        fs::remove_file(&path).expect("unlink old");
        write_file(&path, "new build");
        let info = watch
            .take_newly_stale_at(Instant::now())
            .expect("first call after replacement reports");
        // One-shot: the report never repeats …
        assert_eq!(watch.take_newly_stale_at(Instant::now()), None);
        // … while the sidebar's latch keeps reading stale.
        assert_eq!(watch.check_at(Instant::now()), Some(info));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn global_check_is_fail_open_and_idempotent() {
        // The process-global path stats the real test binary, which is not being
        // redeployed mid-test, so this can only return `None` — and must never
        // panic or wedge on repeated calls.
        init();
        assert_eq!(check(), None);
        assert_eq!(check(), None);
    }
}
