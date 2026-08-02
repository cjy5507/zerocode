//! Cross-restart persistence for the `/goal` and `/loop` controllers (the
//! loop-engineering "Memory" component). Serializable DTOs mirror the live
//! controller state with only `serde`-friendly types — wall-clock and snapshot
//! fields are reconstructed on load rather than stored, and `Instant`-based
//! schedules are re-armed relative to the new process start.
//!
//! Everything here is **fail-open**: a missing, unreadable, or version-mismatched
//! state file loads as empty (a session simply starts without restored
//! automation), and saves are best-effort (a write failure never blocks a turn).
//! Set `ZO_DISABLE_AUTOMATION_PERSIST=1` to turn the whole feature off.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use decision_core::{
    BlockTracker, ConvergenceLedger, CriteriaProgress, LoopBudget, PivotLedger, ProgressTracker,
};
use serde::{Deserialize, Serialize};

/// On-disk format version. Bumped on an incompatible schema change; a file with
/// any other version is ignored (fail-open to empty).
const VERSION: u32 = 1;
const DISABLE_ENV: &str = "ZO_DISABLE_AUTOMATION_PERSIST";

#[must_use]
pub(crate) fn current_version() -> u32 {
    VERSION
}

/// The full persisted automation state for one session.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct AutomationStatePersist {
    pub version: u32,
    #[serde(default)]
    pub goal: Option<GoalPersist>,
    #[serde(default)]
    pub loops: Vec<LoopPersist>,
}

/// A resumable goal. Validators are stored as their labels and reparsed on load;
/// the transient `last_report` is intentionally dropped.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GoalPersist {
    pub id: u64,
    pub text: String,
    pub checks: Vec<String>,
    pub max_turns: u32,
    pub turn_count: u32,
    /// `GoalRunState` rendered with `Debug` ("Active"/"Paused"/…). Only Active and
    /// Paused are ever persisted; an Active goal reloads as Paused (see policy).
    pub state: String,
    pub output_tokens_used: u64,
    pub token_budget: Option<u64>,
    pub progress: ProgressTracker,
    /// `--allow-writes` opt-in, so a restored goal keeps its permission mode
    /// instead of silently reverting to the read-only default. `#[serde(default)]`
    /// keeps pre-`--allow-writes` state files loadable (they load as `false`).
    #[serde(default)]
    pub allow_writes: bool,
    /// Wall-clock time (Unix epoch seconds) the goal was last persisted. Used to
    /// expire an abandoned, never-progressed goal on restore so a one-off `/goal`
    /// does not linger in the HUD forever across restarts. `#[serde(default)]`
    /// keeps pre-timestamp state files loadable (they load as `0`, which the
    /// restore policy treats as "legacy, unknown age").
    #[serde(default)]
    pub saved_at: u64,
    /// Runaway-guard ledgers, persisted so a restart cannot re-buy their
    /// budgets: without these a goal that was one blocked turn from
    /// escalation (or had spent its pivots / hit convergence churn) restarted
    /// every counter at zero — a restart-resume cycle made the guards
    /// unreachable. `#[serde(default)]` keeps pre-ledger state files loadable
    /// (they load as fresh ledgers, the old behavior). The checkpoint ledger
    /// is deliberately NOT persisted: a restored goal loads as Paused and the
    /// resume itself is the acknowledgement its pacing wants.
    #[serde(default)]
    pub blocks: BlockTracker,
    #[serde(default)]
    pub convergence: ConvergenceLedger,
    #[serde(default)]
    pub criteria: CriteriaProgress,
    #[serde(default)]
    pub pivots: PivotLedger,
}

/// A resumable recurring loop. Fixed-count loops are never persisted (their
/// prompts are eagerly drained at creation, so reviving one would re-fire it).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct LoopPersist {
    pub id: String,
    pub prompt: String,
    /// `LoopStatus` rendered with `Debug`; only Active/Paused are persisted.
    pub status: String,
    pub run_count: u32,
    /// Cumulative assistant output tokens charged against the loop's budget, so a
    /// restored token-bounded loop resumes counting from where it stopped instead
    /// of from zero. `#[serde(default)]` keeps pre-budget state files loadable.
    #[serde(default)]
    pub output_tokens: u64,
    /// The loop's resource ceiling (`--max-runs` / `--token-budget`). Without it a
    /// restored recurring loop reloaded unbounded and ran forever past its cap.
    /// Defaulted for backward compatibility — an absent budget reloads unbounded.
    #[serde(default)]
    pub budget: LoopBudget,
    /// `--until <check>` completion validators as their labels (reparsed on load,
    /// like `GoalPersist.checks`), so a restored loop keeps its "done when X" stop
    /// condition. `#[serde(default)]` keeps older state files loadable.
    #[serde(default)]
    pub until: Vec<String>,
    /// The `--until` stall tracker, so a restored loop keeps its no-progress
    /// streak instead of resetting it (and re-earning a full retry budget).
    /// `#[serde(default)]` keeps older state files loadable.
    #[serde(default)]
    pub progress: ProgressTracker,
    /// `--allow-writes` opt-in, so a restored loop keeps its permission mode
    /// instead of silently reverting to the read-only default. `#[serde(default)]`
    /// keeps pre-`--allow-writes` state files loadable (they load as `false`).
    #[serde(default)]
    pub allow_writes: bool,
    pub kind: LoopKindPersist,
}

/// Only the schedule *config* is stored; `Instant` due times and the watch
/// snapshot are reconstructed on load so a restart never replays a backlog.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum LoopKindPersist {
    Interval { every_secs: u64 },
    Watch { glob: String },
}

/// Project-scoped state file: `<base>/.zo/automation/state.json`. Keyed by
/// the workspace, NOT the per-run session id — `/goal` and `/loop` are a
/// project-level concept, so any session opened in this cwd resumes the same
/// automation (a fresh interactive `zo` restart restores it, not just
/// `zo serve`). Lives under `.zo/` (gitignored) so it never pollutes the
/// working tree.
fn state_path(cwd: &Path) -> PathBuf {
    runtime::zo_state_base(cwd)
        .join(".zo")
        .join("automation")
        .join("state.json")
}

fn disabled() -> bool {
    std::env::var(DISABLE_ENV).is_ok()
}

type StateWriteMutex = Mutex<()>;
type StateWriteLocks = Mutex<HashMap<PathBuf, Weak<StateWriteMutex>>>;

fn canonical_lock_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        path.parent()
            .and_then(|parent| parent.canonicalize().ok())
            .and_then(|parent| path.file_name().map(|name| parent.join(name)))
            .unwrap_or_else(|| path.to_path_buf())
    })
}

fn state_write_lock(path: &Path) -> Arc<StateWriteMutex> {
    static LOCKS: OnceLock<StateWriteLocks> = OnceLock::new();
    let key = canonical_lock_path(path);
    let locks = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    lock
}

/// Best-effort, atomic write of the automation state. An empty state removes any
/// stale file instead of writing churn (a project with no active automation
/// keeps no file). Silently does nothing when disabled or on failure —
/// persistence must never break a turn.
pub(crate) fn save(cwd: &Path, state: &AutomationStatePersist) {
    if disabled() {
        return;
    }
    let _ = save_with_hook(cwd, state, || {});
}

pub(super) fn save_with_hook(
    cwd: &Path,
    state: &AutomationStatePersist,
    before_write: impl FnOnce(),
) -> std::io::Result<()> {
    let path = state_path(cwd);
    let write_lock = state_write_lock(&path);
    let _guard = write_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    before_write();

    if state.goal.is_none() && state.loops.is_empty() {
        return match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        };
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(state).map_err(std::io::Error::other)?;
    crate::write_atomic(&path, json.as_bytes())
}

/// Load the persisted state, or an empty default on any error / version skew.
#[must_use]
pub(crate) fn load(cwd: &Path) -> AutomationStatePersist {
    if disabled() {
        return AutomationStatePersist::default();
    }
    std::fs::read_to_string(state_path(cwd))
        .ok()
        .and_then(|raw| serde_json::from_str::<AutomationStatePersist>(&raw).ok())
        .filter(|state| state.version == VERSION)
        .unwrap_or_default()
}
