//! Single owner of the per-agent manifest directory walk, with a short-TTL
//! process-global cache.
//!
//! The agent store (`ZO_AGENT_STORE`, else the per-project state dir
//! `~/.zo/projects/<slug>/state/agents`) holds one `<id>.json` manifest per
//! spawned agent. Two live surfaces need the same
//! "newest-first list of manifests with their mtimes":
//!
//! * the HUD sidebar scan — [`crate::tui::workflow_progress`]'s sibling in
//!   `session::tui_loop::list_running_agents_since`, rebuilt every live snapshot
//!   (~500 ms mid-turn, ~3 s idle);
//! * the `Ctrl+O` workflow viewer's plain-fan-out fallback
//!   ([`crate::tui::workflow_progress::build_agents_fallback_since`]).
//!
//! Before this module each kept its **own** copy of the `read_dir` + per-entry
//! `metadata().modified()` (`stat`) + sort, so the same directory was walked from
//! scratch on every snapshot — and terminal manifests accumulate on disk forever,
//! so that per-entry `stat` cost grows without bound over a long session. This is
//! the single canonical enumeration, plus a time-to-live cache so consecutive
//! reads within the live cadence reuse one walk instead of re-`stat`-ing every
//! manifest.
//!
//! The cache holds only the directory metadata (paths + mtimes) — the part that
//! scales with the manifest count. Callers still `read_to_string` + parse each
//! manifest's **content** fresh on every call, so a status change written inside
//! an existing manifest is never hidden; only the *set of files* is reused, for
//! at most [`CACHE_TTL`]. The list is display-only, so a newly spawned agent
//! appearing at most `CACHE_TTL` late is imperceptible at the snapshot cadence.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// How long one enumeration of the manifest directory is reused before the next
/// caller re-walks it.
///
/// Chosen to cover the mid-turn live cadence (a snapshot roughly every 500 ms,
/// plus an overlapping `Ctrl+O` viewer refresh on the same directory) so the
/// repeated `stat`-storm collapses to one walk per window, while staying short
/// enough that a freshly spawned agent shows up within ~1.5 s. Idle snapshots run
/// further apart than this, so they re-walk a fresh directory (correct: nothing is
/// competing for the render loop while idle).
pub(crate) const CACHE_TTL: Duration = Duration::from_millis(1500);

/// One enumerated manifest: its path and last-modified time.
///
/// `SystemTime` is the richest form the filesystem exposes; the HUD scan uses it
/// directly and the workflow fallback narrows it to epoch seconds at its own
/// boundary, so both historical callers keep their behavior from one source.
pub type ManifestEntry = (PathBuf, SystemTime);

/// Pure, uncached enumeration of `<store>/<id>.json` manifests, newest-first.
///
/// Ordering is by modified time descending, then by path descending as a stable
/// tiebreak — identical to the two per-call copies this replaces. A missing or
/// unreadable directory (or an entry whose `metadata()`/`modified()` is
/// unavailable) yields an empty list / skips that entry, exactly as before.
#[must_use]
pub(crate) fn enumerate(store: &Path) -> Vec<ManifestEntry> {
    let Ok(entries) = std::fs::read_dir(store) else {
        return Vec::new();
    };
    let mut paths = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((path, modified))
        })
        .collect::<Vec<_>>();
    paths.sort_by(|(left_path, left_modified), (right_path, right_modified)| {
        right_modified
            .cmp(left_modified)
            .then_with(|| right_path.cmp(left_path))
    });
    paths
}

/// Newest-first manifest enumeration for `store`, served from the shared TTL
/// cache when a recent walk of the *same* store is still fresh.
///
/// Returns an `Arc<[ManifestEntry]>` so a cache hit is a cheap refcount bump
/// rather than a clone of (potentially thousands of) `PathBuf`s. Callers iterate
/// by reference; both reading a manifest's content (`read_to_string(path)`) and
/// computing its age work directly on the shared slice.
#[must_use]
pub fn newest_first_cached(store: &Path) -> Arc<[ManifestEntry]> {
    if let Some(hit) = cache_lookup(store) {
        return hit;
    }
    newest_first_fresh(store)
}

/// Newest-first manifest enumeration for `store`, bypassing the TTL cache and
/// replacing it with the fresh result. Use this at visibility boundaries (for
/// example opening Ctrl+O) where a just-created manifest must not be hidden by a
/// pre-spawn cached file set.
#[must_use]
pub fn newest_first_fresh(store: &Path) -> Arc<[ManifestEntry]> {
    // Enumerate outside the lock so a slow directory walk never serializes a
    // concurrent reader of a *different* store. Two simultaneous misses on the
    // same store may both walk once; that is rare (snapshots are sub-Hz) and
    // strictly cheaper than holding the mutex across the I/O.
    let entries: Arc<[ManifestEntry]> = Arc::from(enumerate(store));
    cache_store(store, &entries);
    entries
}

struct CacheSlot {
    store: PathBuf,
    captured_at: Instant,
    entries: Arc<[ManifestEntry]>,
}

fn cache() -> &'static Mutex<Option<CacheSlot>> {
    static CACHE: OnceLock<Mutex<Option<CacheSlot>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn cache_lookup(store: &Path) -> Option<Arc<[ManifestEntry]>> {
    cache_lookup_at(store, Instant::now())
}

/// Freshness decision with the clock injected: a slot hits while `now` is
/// strictly inside `captured_at + CACHE_TTL`. Split from [`cache_lookup`] so
/// the TTL policy is deterministic under test — asserting reuse through the
/// wall-clock wrapper raced the 1.5s TTL against test-runner load (a saturated
/// machine could stall between two calls long enough to expire the slot).
fn cache_lookup_at(store: &Path, now: Instant) -> Option<Arc<[ManifestEntry]>> {
    let guard = cache().lock().ok()?;
    let slot = guard.as_ref()?;
    (slot.store == store && now.duration_since(slot.captured_at) < CACHE_TTL)
        .then(|| Arc::clone(&slot.entries))
}

fn cache_store(store: &Path, entries: &Arc<[ManifestEntry]>) {
    if let Ok(mut guard) = cache().lock() {
        *guard = Some(CacheSlot {
            store: store.to_path_buf(),
            captured_at: Instant::now(),
            entries: Arc::clone(entries),
        });
    }
}

/// Drop any cached enumeration. Test-only: lets a test that writes manifests and
/// then re-reads the *same* store path observe its own writes regardless of a
/// neighbouring test's recent walk.
#[cfg(test)]
pub(crate) fn reset_cache_for_tests() {
    if let Ok(mut guard) = cache().lock() {
        *guard = None;
    }
}

/// The current slot's capture instant, if any. Test-only: lets the TTL tests
/// drive [`cache_lookup_at`] with `now` anchored to the slot's own timestamp
/// instead of racing the wall clock.
#[cfg(test)]
fn cache_captured_at_for_tests() -> Option<Instant> {
    cache().lock().ok()?.as_ref().map(|slot| slot.captured_at)
}

/// Serialize tests that exercise the process-global enumeration cache.
///
/// The cache itself is intentionally global for production, so any test in any
/// module that calls `reset_cache_for_tests`, `newest_first_cached`, or
/// `newest_first_fresh` must hold this same lock for the whole reset/read/read
/// sequence. Separate per-module locks still allow interleaving and make the TTL
/// reuse assertions flaky under Rust's parallel test runner.
#[cfg(test)]
pub(crate) fn cache_test_lock_for_tests() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
        super::cache_test_lock_for_tests()
    }

    fn unique_store(label: &str) -> PathBuf {
        let millis = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "zo-agent-manifests-{label}-{}-{millis}-{counter}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create store");
        dir
    }

    fn touch_json(store: &Path, name: &str) {
        std::fs::write(store.join(name), "{}").expect("write manifest");
    }

    #[test]
    fn enumerate_returns_only_json_newest_first() {
        let store = unique_store("order");
        // Write three manifests with strictly increasing mtimes so the order is
        // deterministic regardless of filesystem timestamp granularity.
        for name in ["a.json", "b.json", "c.json"] {
            touch_json(&store, name);
            std::thread::sleep(Duration::from_millis(10));
        }
        // A non-json file must be ignored entirely.
        std::fs::write(store.join("notes.txt"), "ignore me").expect("write txt");

        let entries = enumerate(&store);
        let names: Vec<String> = entries
            .iter()
            .map(|(path, _)| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(
            names,
            vec!["c.json", "b.json", "a.json"],
            "newest-first, json-only"
        );
        let _ = std::fs::remove_dir_all(&store);
    }

    #[test]
    fn enumerate_missing_dir_is_empty() {
        let store = std::env::temp_dir().join(format!(
            "zo-agent-manifests-missing-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(enumerate(&store).is_empty(), "missing dir → empty list");
    }

    #[test]
    fn cache_reuses_one_walk_within_ttl_then_sees_new_files_after_reset() {
        let _guard = cache_test_lock();
        let store = unique_store("ttl");
        reset_cache_for_tests();
        touch_json(&store, "first.json");

        let first = newest_first_cached(&store);
        assert_eq!(first.len(), 1, "cold read sees the one manifest");

        // A second manifest written *within* TTL is hidden because the cached
        // enumeration (the set of files) is reused — the cache's whole point.
        // Drive the freshness decision at the slot's own capture instant
        // (deterministically inside the TTL) instead of re-calling the
        // wall-clock wrapper: under a saturated parallel test run the gap
        // between two calls could exceed the 1.5s TTL and flake this hit.
        touch_json(&store, "second.json");
        let captured_at = cache_captured_at_for_tests().expect("cold read populated the slot");
        let cached = cache_lookup_at(&store, captured_at).expect("hit strictly inside the TTL");
        assert_eq!(
            cached.len(),
            1,
            "within TTL the prior enumeration is reused, new file not yet seen"
        );
        assert!(
            Arc::ptr_eq(&first, &cached),
            "a hit returns the same shared allocation, not a fresh walk"
        );

        // The instant the TTL lapses the slot stops hitting…
        assert!(
            cache_lookup_at(&store, captured_at + CACHE_TTL).is_none(),
            "at exactly captured_at + TTL the slot must be expired"
        );

        // …and clearing the cache re-walks and now sees both files.
        reset_cache_for_tests();
        let fresh = newest_first_cached(&store);
        assert_eq!(
            fresh.len(),
            2,
            "after reset the re-walk sees both manifests"
        );
        let _ = std::fs::remove_dir_all(&store);
    }

    #[test]
    fn cache_refreshes_when_store_path_changes() {
        let _guard = cache_test_lock();
        let store_a = unique_store("switch-a");
        let store_b = unique_store("switch-b");
        reset_cache_for_tests();
        touch_json(&store_a, "a.json");
        touch_json(&store_b, "b1.json");
        touch_json(&store_b, "b2.json");

        let a = newest_first_cached(&store_a);
        assert_eq!(a.len(), 1, "store A enumerated");
        // A different store must not return store A's cached slot, even within TTL.
        let b = newest_first_cached(&store_b);
        assert_eq!(
            b.len(),
            2,
            "store B re-enumerated, not served from A's slot"
        );

        let _ = std::fs::remove_dir_all(&store_a);
        let _ = std::fs::remove_dir_all(&store_b);
    }
}
