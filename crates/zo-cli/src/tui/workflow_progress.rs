//! Reads the live workflow progress snapshot and joins it with the per-agent
//! manifests into a [`WorkflowView`] for the [`workflow_viewer`] modal.
//!
//! The engine's [`progress`] sink writes `.zo/workflows/_active.progress.json`
//! (phase topology + each phase's `agent_ids`). This module reads that file and,
//! for each `agent_id`, reads the per-agent manifest (`.zo/agents/<id>.json`,
//! the same files the sidebar polls) to attach live name/status/currentTool/
//! tokens/elapsed. Phase tallies come straight from the progress file, so they
//! stay accurate even when the manifest read budget is exhausted.
//!
//! Lives in the lib (not `tui_loop`) so [`App`]'s tick loop can re-read it while
//! the viewer is open — the same data is used by the host to open the modal and
//! by the app to refresh it.
//!
//! [`progress`]: tools
//! [`workflow_viewer`]: super::modals::workflow_viewer
//! [`App`]: super::app::App

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::Value;

use super::agent_session_filter::manifest_belongs_to_session;
use super::modals::{WorkflowAgentRow, WorkflowPhaseRow, WorkflowView};

/// Must match `tools::workflow_tools::progress::ACTIVE_PROGRESS_FILE`.
const ACTIVE_PROGRESS_FILE: &str = "_active.progress.json";

/// Cap on total manifest reads per poll so a hundreds-of-agents workflow cannot
/// stall the render thread. Phases past the budget still show accurate counts
/// (from the progress file), just without per-agent rows.
const MANIFEST_READ_BUDGET: usize = 128;

/// How much of a "finished agent" a spawned-but-still-running agent counts for in
/// the phase completion percent. Without it, a phase whose agents have not yet
/// reached a terminal state reads 0% for its entire duration (the recorded tally
/// only steps on `AgentDone`), so a single long phase — or a single long-running
/// agent — shows "0% done" the whole time even though work is clearly happening.
/// 0.3 lets the bar move off zero as agents spin up while still keeping a running
/// agent worth visibly less than a finished one; it is capped per phase so the
/// in-flight credit can never push a non-terminal phase to 100%.
pub(crate) const INFLIGHT_AGENT_FRACTION: f64 = 0.3;

/// How long to trust the workflow progress doc's just-spawned agent ids before
/// a matching live manifest appears. Claude tool execution can stamp the
/// workflow snapshot before the child manifest is visible to the HUD poll; if we
/// zero the doc-only running count immediately, the UI briefly claims no agents
/// are running even though the workflow has launched them. Keep this short so
/// the existing ghost-run reaper still wins once a stale/foreign snapshot stops
/// producing live manifests.
const DOC_ONLY_RUNNING_GRACE_SECS: u64 = 30;

/// On-disk progress document — a tolerant mirror of the engine's writer. Every
/// field defaults so a newer/older schema still loads (forward/backward compat).
#[derive(Deserialize, Default)]
struct ProgressDoc {
    /// Run id stamped by the engine. Joins the snapshot to its append-only event
    /// log (`<run_id>.events.jsonl`) for the viewer's event-log inspector.
    #[serde(default)]
    run_id: String,
    /// Foreground session id stamped by the workflow tool. New snapshots use it
    /// to prove doc-only just-spawned agents belong to this TUI session.
    #[serde(default)]
    parent_session_id: Option<String>,
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    synthesizing: bool,
    /// Epoch milliseconds of the writer's last emit (the engine stamps it every
    /// event). `0` when absent (an older snapshot) — staleness is skipped then.
    #[serde(default)]
    updated_at_ms: u64,
    #[serde(default)]
    phases: Vec<ProgressPhase>,
}

/// A `running` snapshot older than this is treated as a crashed/abandoned engine
/// and not shown. The progress file only advances at phase boundaries, so the gap
/// between writes can be a whole long-running phase. Keep the stale backstop
/// measured in hours so a healthy build/test/review phase that runs past the old
/// 20-minute window remains visible while still eventually reaping abandoned
/// snapshots.
const STALE_AFTER_MS: u64 = 6 * 60 * 60 * 1000;

#[derive(Deserialize, Default)]
struct ProgressPhase {
    #[serde(default)]
    id: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    round: u32,
    #[serde(default)]
    agent_ids: Vec<String>,
    #[serde(default)]
    completed: usize,
    #[serde(default)]
    failed: usize,
    #[serde(default)]
    still_running: usize,
    #[serde(default)]
    carried: usize,
    #[serde(default)]
    retried: usize,
    #[serde(default)]
    skipped: usize,
    #[serde(default)]
    findings: usize,
    #[serde(default)]
    blocked: usize,
    #[serde(default)]
    invalidated: usize,
    #[serde(default)]
    selective_retries: usize,
}

impl ProgressPhase {
    fn selective_event_count(&self) -> usize {
        self.carried
            .saturating_add(self.retried)
            .saturating_add(self.skipped)
            .saturating_add(self.findings)
            .saturating_add(self.blocked)
            .saturating_add(self.invalidated)
            .saturating_add(self.selective_retries)
    }
}

/// Per-agent manifest-read cache for [`read_view_cached`]. A manifest whose file
/// mtime is unchanged *and* whose last-parsed status is terminal is served from
/// the cache, skipping the read+parse on the render thread. Running/pending agents
/// — the live ones, and few (bounded by the concurrency cap) — are always
/// re-read, so live status / `currentTool` / elapsed never freeze.
#[derive(Default)]
pub struct WorkflowViewCache {
    rows: std::collections::HashMap<String, (SystemTime, WorkflowAgentRow)>,
}

/// One phase of a live workflow, condensed to the per-phase agent tally that
/// drives the sidebar's always-on "Fleet" view (a progress bar + status per
/// phase). Manifest-free — built straight from the progress snapshot, so it
/// rides the existing summary refresh with no extra file reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FleetPhase {
    /// Phase id/slug as stamped by the engine (e.g. `Review`, `Verify`).
    pub id: String,
    /// Exact Todo plan step owned by this real workflow phase. Synthetic
    /// agents-only summaries leave this `None` so their `agents` phase cannot
    /// be mistaken for a plan step with the same text.
    pub step_id: Option<String>,
    /// Exact manifest ids assigned to this phase. Used to select only an
    /// executor that the workflow engine actually attached to the step; empty
    /// when a malformed progress snapshot contains unsafe ids.
    pub agent_ids: Vec<String>,
    /// Phase status: `pending` | `running` | `done` | `resumed` | `failed`.
    pub status: String,
    /// Agents spawned for this phase (`agent_ids.len()`).
    pub total: usize,
    /// Agents that reached a successful terminal result.
    pub completed: usize,
    /// Agents that reached a failed terminal result.
    pub failed: usize,
    /// Agents still running in this phase.
    pub running: usize,
}

impl FleetPhase {
    /// Terminal agents (completed + failed) — the filled portion of the bar.
    #[must_use]
    pub const fn terminal(&self) -> usize {
        self.completed.saturating_add(self.failed)
    }
}

/// Small, manifest-free workflow snapshot for compact status surfaces.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkflowSummary {
    pub name: String,
    pub status: String,
    pub mode: String,
    pub current_phase: String,
    pub current_phase_status: String,
    pub current_phase_index: usize,
    pub total_phases: usize,
    /// Whole-workflow completion estimate, 0-100. Based on completed phases plus
    /// the current phase's recorded completed/failed agent tally when present.
    pub progress_percent: u8,
    /// Number of terminal phases (`done` / `resumed`) in the active snapshot.
    pub completed_phases: usize,
    pub next_phase: Option<String>,
    pub total_agents: usize,
    /// Agents that reached a successful terminal result.
    pub completed_agents: usize,
    /// Agents that reached a failed terminal result.
    pub failed_agents: usize,
    /// Agents recorded as still running by the progress snapshot. While a phase
    /// is running and the engine has not written a tally yet, this falls back to
    /// the current phase's spawned-but-not-finished count.
    pub running_agents: usize,
    /// Per-phase breakdown for the always-on Fleet view. Empty for a plain
    /// single-fan-out `SpawnMultiAgent` run (which has no phase structure); a
    /// multi-phase `Workflow` populates one entry per phase.
    pub phases: Vec<FleetPhase>,
}

/// Read the active workflow snapshot, or `None` when there is no progress file,
/// it is unreadable/corrupt, has no phases, or is not a live run (see
/// [`is_live`]).
#[must_use]
pub fn read_view() -> Option<WorkflowView> {
    read_view_since(0, None)
}

/// Read the active workflow/fan-out view when it belongs to the visible session.
/// Both the workflow progress file and plain fan-out manifests live in
/// workspace-global stores, so opening Ctrl+O must filter out fresh-but-foreign
/// rows from previous chats.
#[must_use]
pub fn read_view_since(started_after_secs: u64, session_id: Option<&str>) -> Option<WorkflowView> {
    // Open path (Ctrl+O): require a *live* run so a finished/stale snapshot never
    // opens a dead tree as if current.
    if let Some(doc) = load_doc(true)
        .filter(|doc| doc_updated_after(doc, started_after_secs))
        .filter(|doc| doc_session_matches(doc, session_id))
    {
        let store = agent_store_dir();
        let now = now_secs();
        let allow_doc_only_placeholders = doc_only_rows_allowed(&doc, session_id);
        let view = build_view(doc, |id| {
            read_agent_row_scoped(store.as_ref(), id, now, session_id, allow_doc_only_placeholders)
        });
        if !session_filter_enabled(session_id)
            || view_has_agents(&view)
            || (allow_doc_only_placeholders && view_has_progress_agents(&view))
        {
            return Some(view);
        }
    }
    // No workflow progress file → a plain `SpawnMultiAgent` fan-out. Synthesize a
    // view from the live agent manifests so `Ctrl+O` opens on it too.
    read_agents_fallback_since(true, started_after_secs, session_id)
}

/// Like [`read_view`] but for the already-open viewer's ~2 Hz refresh: reuses
/// unchanged terminal manifests from `cache` (so a large, mostly-finished
/// workflow stops re-parsing every manifest each tick) and — crucially — accepts
/// a *terminal* snapshot, so the tree flips to completed/cancelled and the
/// spinners stop when the run ends instead of freezing on the last running frame.
#[must_use]
pub fn read_view_cached(cache: &mut WorkflowViewCache) -> Option<WorkflowView> {
    read_view_cached_since(cache, 0, None)
}

/// Refresh the already-open workflow/fan-out view without borrowing a UI-owned
/// cache. Intended for background snapshot tasks: it follows a live run into its
/// terminal state like [`read_view_cached_since`], but keeps all disk IO off the
/// render tick that requested it.
#[must_use]
pub fn read_view_refresh_since(
    started_after_secs: u64,
    session_id: Option<&str>,
) -> Option<WorkflowView> {
    let mut cache = WorkflowViewCache::default();
    read_view_cached_since(&mut cache, started_after_secs, session_id)
}

pub fn read_view_cached_since(
    cache: &mut WorkflowViewCache,
    started_after_secs: u64,
    session_id: Option<&str>,
) -> Option<WorkflowView> {
    // Refresh path: the live gate is only for *opening* (read_view); once open,
    // follow the run into its terminal state.
    let Some(doc) = load_doc(false)
        .filter(|doc| doc_updated_after(doc, started_after_secs))
        .filter(|doc| doc_session_matches(doc, session_id))
    else {
        // Fan-out fallback: re-read fresh (the few manifests are cheap and running
        // rows must never be cached); the workflow-only cache is left untouched.
        return read_agents_fallback_since(false, started_after_secs, session_id);
    };
    let store = agent_store_dir();
    let now = now_secs();
    // Bound the cache to the current snapshot's agents (ids are unique per spawn,
    // never reused) so it cannot grow across workflow runs in a long session.
    let live_ids: std::collections::HashSet<&str> = doc
        .phases
        .iter()
        .flat_map(|p| p.agent_ids.iter().map(String::as_str))
        .collect();
    cache.rows.retain(|id, _| live_ids.contains(id.as_str()));
    let allow_doc_only_placeholders = doc_only_rows_allowed(&doc, session_id);
    let view = build_view(doc, |id| {
        read_agent_row_cached_scoped(
            store.as_ref(),
            id,
            now,
            cache,
            session_id,
            allow_doc_only_placeholders,
        )
    });
    if !session_filter_enabled(session_id)
        || view_has_agents(&view)
        || (allow_doc_only_placeholders && view_has_progress_agents(&view))
    {
        return Some(view);
    }
    read_agents_fallback_since(false, started_after_secs, session_id)
}

/// Read only the active workflow topology for the sidebar/HUD.
///
/// Unlike [`read_view`], this does not touch per-agent manifests, so it stays
/// cheap enough for compact status refreshes. It is intentionally live-gated:
/// finished workflow snapshots remain on disk, and the sidebar should not show a
/// stale completed workflow forever after the run has ended.
#[must_use]
pub fn read_summary() -> Option<WorkflowSummary> {
    read_summary_since(0, None)
}

/// Read only the active workflow topology when it belongs to the visible
/// session. The progress store is workspace-local, so a newly opened chat must
/// not inherit a still-fresh `running` snapshot from a previous session.
#[must_use]
pub fn read_summary_since(
    started_after_secs: u64,
    session_id: Option<&str>,
) -> Option<WorkflowSummary> {
    if let Some(doc) = load_doc(true)
        .filter(|doc| doc_updated_after(doc, started_after_secs))
        .filter(|doc| doc_session_matches(doc, session_id))
    {
        if let Some(summary) = summarize_doc(&doc) {
            return Some(reconcile_summary_with_manifests(
                &doc,
                summary,
                session_id,
            ));
        }
    }
    // No Workflow progress doc → a plain `SpawnMultiAgent` / pre-analysis fan-out.
    // Synthesize a one-phase summary from the live agent manifests so the always-on
    // Fleet shows for fan-out runs too — mirroring [`read_view_since`]'s fallback
    // (the reason `Ctrl+O` already opens on a fan-out while the sidebar Fleet, which
    // only read the progress doc, used to stay blank during pre-analysis).
    summarize_agents_fallback(started_after_secs, session_id)
}

/// Synthesize a [`WorkflowSummary`] for a plain agent fan-out (no `Workflow`
/// progress doc) from the live per-agent manifests, so the sidebar Fleet renders
/// during pre-analysis / `SpawnMultiAgent` runs. Returns `None` when nothing is
/// live (the `require_live` gate in [`build_agents_fallback_since`]), so an
/// ordinary turn with no spawned agents never shows a spurious Fleet.
fn summarize_agents_fallback(
    started_after_secs: u64,
    session_id: Option<&str>,
) -> Option<WorkflowSummary> {
    let store = agent_store_dir()?;
    let view = build_agents_fallback_since(&store, true, now_secs(), started_after_secs, session_id)?;
    Some(summary_from_agents_view(&view))
}

/// Map the synthetic single-phase fan-out [`WorkflowView`] onto the compact
/// [`WorkflowSummary`] the sidebar/HUD consume. Pure (no IO) so it is unit-testable
/// from a manifest fixture.
fn summary_from_agents_view(view: &WorkflowView) -> WorkflowSummary {
    // `build_agents_fallback_since` always synthesizes exactly one "agents" phase.
    let phase = view.phases.first();
    let id = phase.map_or_else(|| view.name.clone(), |p| p.id.clone());
    let status = phase.map_or_else(|| view.status.clone(), |p| p.status.clone());
    let total = phase.map_or(0, |p| p.total);
    let completed = phase.map_or(0, |p| p.completed);
    let failed = phase.map_or(0, |p| p.failed);
    let running = phase.map_or(0, |p| p.still_running);
    let agent_ids = phase.map_or_else(Vec::new, |p| {
        p.agents.iter().map(|agent| agent.id.clone()).collect()
    });

    let terminal = completed.saturating_add(failed);
    let progress_percent = if total == 0 {
        0
    } else {
        u8::try_from(terminal.saturating_mul(100) / total)
            .unwrap_or(100)
            .min(100)
    };
    let phase_terminal = matches!(status.as_str(), "done" | "resumed");

    WorkflowSummary {
        name: view.name.clone(),
        status: view.status.clone(),
        mode: view.mode.clone(),
        current_phase: id.clone(),
        current_phase_status: status.clone(),
        current_phase_index: 1,
        total_phases: 1,
        progress_percent,
        completed_phases: usize::from(phase_terminal),
        next_phase: None,
        total_agents: total,
        completed_agents: completed,
        failed_agents: failed,
        running_agents: running,
        phases: vec![FleetPhase {
            id,
            step_id: None,
            agent_ids,
            status,
            total,
            completed,
            failed,
            running,
        }],
    }
}

fn doc_session_matches(doc: &ProgressDoc, session_id: Option<&str>) -> bool {
    match (session_id, doc.parent_session_id.as_deref()) {
        (Some(expected), Some(actual)) => actual == expected,
        // Older snapshots do not carry a session stamp; keep the legacy manifest
        // scoped path for them, but do not grant them doc-only grace.
        _ => true,
    }
}

/// Validate identity metadata once on the workflow-poll path, not on every TUI
/// frame. A malformed list is unusable for exact attribution, but its original
/// length/tallies still drive the Fleet progress fields.
fn correlatable_agent_ids(agent_ids: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::with_capacity(agent_ids.len());
    if agent_ids
        .iter()
        .all(|id| !id.is_empty() && id.trim() == id && seen.insert(id.as_str()))
    {
        agent_ids.to_vec()
    } else {
        Vec::new()
    }
}

fn doc_updated_after(doc: &ProgressDoc, started_after_secs: u64) -> bool {
    if started_after_secs == 0 {
        return true;
    }
    if doc.updated_at_ms == 0 {
        return false;
    }
    doc.updated_at_ms / 1000 >= started_after_secs
}

fn manifest_value_created_after(value: &Value, started_after_secs: u64) -> bool {
    if started_after_secs == 0 {
        return true;
    }
    let created = value
        .get("createdAt")
        .or_else(|| value.get("created_at"))
        .and_then(|value| match value {
            Value::Number(number) => number.as_u64(),
            Value::String(text) => text.trim().parse::<u64>().ok(),
            _ => None,
        });
    created.is_some_and(|created| created >= started_after_secs)
}

fn summarize_doc(doc: &ProgressDoc) -> Option<WorkflowSummary> {
    let total_phases = doc.phases.len();
    let active_idx = doc
        .phases
        .iter()
        .position(|phase| phase.status == "running")
        .or_else(|| {
            doc.phases
                .iter()
                .position(|phase| phase.status == "pending")
        })
        .or_else(|| total_phases.checked_sub(1))?;
    let current = doc.phases.get(active_idx)?;
    let next_phase = doc
        .phases
        .iter()
        .skip(active_idx + 1)
        .find(|phase| !matches!(phase.status.as_str(), "done" | "resumed"))
        .map(|phase| phase.id.clone());
    let total_agents = doc.phases.iter().map(|phase| phase.agent_ids.len()).sum();
    let completed_phases = doc
        .phases
        .iter()
        .filter(|phase| phase_is_terminal(phase))
        .count();
    let completed_agents = doc.phases.iter().map(|phase| phase.completed).sum();
    let failed_agents = doc.phases.iter().map(|phase| phase.failed).sum();
    let running_agents = doc.phases.iter().map(phase_running_agents).sum();
    let _selective_events: usize = doc.phases.iter().map(ProgressPhase::selective_event_count).sum();

    let current_status = phase_display_status(current);

    Some(WorkflowSummary {
        name: doc.name.clone(),
        status: doc.status.clone(),
        mode: doc.mode.clone(),
        current_phase: current.id.clone(),
        current_phase_status: current_status,
        current_phase_index: active_idx + 1,
        total_phases,
        progress_percent: workflow_progress_percent(&doc.phases),
        completed_phases,
        next_phase,
        total_agents,
        completed_agents,
        failed_agents,
        running_agents,
        phases: doc
            .phases
            .iter()
            .map(|phase| FleetPhase {
                id: phase.id.clone(),
                step_id: Some(phase.id.clone()),
                agent_ids: correlatable_agent_ids(&phase.agent_ids),
                status: phase_display_status(phase),
                total: phase.agent_ids.len(),
                completed: phase.completed,
                failed: phase.failed,
                running: phase_running_agents(phase),
            })
            .collect(),
    })
}

fn reconcile_summary_with_manifests(
    doc: &ProgressDoc,
    summary: WorkflowSummary,
    session_id: Option<&str>,
) -> WorkflowSummary {
    let now = now_secs();
    let doc_only_running_grace = doc_only_running_grace_active(doc, now, session_id);
    let Some(store) = agent_store_dir() else {
        return reconcile_summary_with_live_count(summary, 0, doc_only_running_grace);
    };
    reconcile_summary_with_live_count(
        summary,
        live_running_manifest_count_in(&store, now, doc, session_id),
        doc_only_running_grace,
    )
}

fn doc_only_running_grace_active(
    doc: &ProgressDoc,
    now_secs: u64,
    session_id: Option<&str>,
) -> bool {
    doc.status == "running"
        && doc.updated_at_ms > 0
        && doc_session_matches(doc, session_id)
        && match session_id {
            Some(_) => doc.parent_session_id.is_some(),
            None => true,
        }
        && now_secs
            .saturating_sub(doc.updated_at_ms / 1000)
            <= DOC_ONLY_RUNNING_GRACE_SECS
}

fn reconcile_summary_with_live_count(
    mut summary: WorkflowSummary,
    live_running_count: usize,
    doc_only_running_grace: bool,
) -> WorkflowSummary {
    if summary.running_agents == 0 || live_running_count > 0 || doc_only_running_grace {
        return summary;
    }
    summary.running_agents = 0;
    for phase in &mut summary.phases {
        phase.running = 0;
    }
    summary
}

/// Count how many of a progress doc's agent ids have a *genuinely live*
/// manifest, using the same session-scope and freshness contract as the Ctrl+O
/// agent view. This keeps the HUD from showing doc-only ghost `running` counts
/// after every manifest is gone, terminal, stale, or owned by another session.
fn live_running_manifest_count_in(
    store: &Path,
    now: u64,
    doc: &ProgressDoc,
    session_id: Option<&str>,
) -> usize {
    doc.phases
        .iter()
        .flat_map(|phase| phase.agent_ids.iter())
        .take(MANIFEST_READ_BUDGET)
        .filter(|id| agent_manifest_is_live(store, id, now, session_id))
        .count()
}

fn agent_manifest_is_live(
    store: &Path,
    id: &str,
    now: u64,
    session_id: Option<&str>,
) -> bool {
    let path = store.join(format!("{id}.json"));
    let age = fs::metadata(&path)
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(u64::MAX, |modified| now.saturating_sub(modified.as_secs()));
    if age > AGENT_RUNNING_STALE_SECS {
        return false;
    }
    let Some(value) = read_agent_manifest_value(&path) else {
        return false;
    };
    if !manifest_belongs_to_session(&value, session_id, false) {
        return false;
    }
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    !is_terminal_status(status)
}

fn phase_is_terminal(phase: &ProgressPhase) -> bool {
    matches!(phase.status.as_str(), "done" | "resumed")
}

fn phase_recorded_total(phase: &ProgressPhase) -> usize {
    phase.agent_ids.len().max(
        phase
            .completed
            .saturating_add(phase.failed)
            .saturating_add(phase.still_running),
    )
}

fn phase_display_status(phase: &ProgressPhase) -> String {
    let mut status = phase.status.clone();
    let selective = phase.selective_event_count();
    if selective > 0 {
        let _ = write!(status, " · {selective} selective");
        if phase.findings > 0 {
            let _ = write!(status, " · {} findings", phase.findings);
        }
        if phase.blocked > 0 {
            let _ = write!(status, " · {} blocked", phase.blocked);
        }
    }
    status
}

fn phase_running_agents(phase: &ProgressPhase) -> usize {
    if phase_is_terminal(phase) {
        return 0;
    }
    if phase.still_running > 0 {
        return phase.still_running;
    }
    if phase.status == "running" {
        return phase
            .agent_ids
            .len()
            .saturating_sub(phase.completed.saturating_add(phase.failed));
    }
    0
}

fn phase_progress_percent(phase: &ProgressPhase) -> usize {
    if phase_is_terminal(phase) {
        return 100;
    }
    if phase.status == "pending" {
        return 0;
    }
    let total = phase_recorded_total(phase);
    if total == 0 {
        return 0;
    }
    let finished = phase.completed.saturating_add(phase.failed);
    // Within-agent partial credit so a running phase is not stuck at 0% until its
    // first agent finishes. Each in-flight agent counts as INFLIGHT_AGENT_FRACTION
    // of a finished one, capped at 0.9 of the unfinished remainder so the phase
    // can never reach 100% while still running. AgentDone bumps `completed` and
    // drops `still_running` in the same snapshot, so `finished + inflight_credit`
    // stays monotonic across a completion (no need for a display-side latch).
    let remaining = total.saturating_sub(finished);
    #[allow(
        clippy::cast_precision_loss,
        reason = "agent counts are tiny, far below f64's 53-bit exact-integer range"
    )]
    let (finished_f, inflight_f, remaining_f) = (
        finished as f64,
        phase_running_agents(phase) as f64,
        remaining as f64,
    );
    let inflight_credit = (inflight_f * INFLIGHT_AGENT_FRACTION).min(remaining_f * 0.9);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "percent is a small non-negative whole number; counts are tiny"
    )]
    let pct = ((finished_f + inflight_credit) * 100.0 / total as f64).floor() as usize;
    // Hold below 100 until the phase actually flips terminal (handled above), even
    // when finished == total but the status has not yet been rewritten.
    pct.min(99)
}

fn workflow_progress_percent(phases: &[ProgressPhase]) -> u8 {
    if phases.is_empty() {
        return 0;
    }
    let sum = phases.iter().map(phase_progress_percent).sum::<usize>();
    u8::try_from((sum / phases.len()).min(100)).unwrap_or(100)
}

/// Parse the progress snapshot. `require_live` gates on [`is_live`] (the open
/// path); the refresh path passes `false` so an already-open viewer can follow a
/// run into its terminal state rather than freezing on the last running frame.
fn load_doc(require_live: bool) -> Option<ProgressDoc> {
    let path = progress_path()?;
    let text = fs::read_to_string(&path).ok()?;
    let mut doc: ProgressDoc = serde_json::from_str(&text).ok()?;
    if doc.phases.is_empty() {
        return None;
    }
    // Phase-5: the snapshot is a fast O(1) poll *cache*; the append-only event log
    // is the run's source of truth. Overlay it *before* the visibility gate so the
    // gate and every consumer (sidebar summary + Ctrl+O viewer) decide on the same
    // event-derived status — a finished run whose final snapshot write was dropped
    // reads as terminal everywhere instead of lingering as "running".
    reconcile_doc_with_event_log(&mut doc);
    if !doc_visible(require_live, &doc.status, doc.updated_at_ms, now_ms()) {
        return None;
    }
    Some(doc)
}

/// Overlay the run's append-only event log onto a polled snapshot so every
/// consumer reports the same event-derived status (doc §16-P5: "snapshot은
/// cache로 격하"). The snapshot can lag the run or drop its final overwrite; the
/// log never loses the run's last word, so on disagreement the log wins.
///
/// Only a *running* snapshot is reconciled — a terminal one is already
/// authoritative — which bounds the extra read to live runs; the log is a single
/// small file (one line per topology event, not per token), read on the poll
/// path, never the draw path. The pure core is [`reconcile_doc`] (records
/// injected) so the overlay is unit-tested without an on-disk log.
fn reconcile_doc_with_event_log(doc: &mut ProgressDoc) {
    if doc.status != "running" || doc.run_id.is_empty() {
        return;
    }
    let records = tools::read_event_log(&doc.run_id);
    if !records.is_empty() {
        reconcile_doc(doc, &records);
    }
}

/// Pure core of [`reconcile_doc_with_event_log`]. Advances the workflow- and
/// phase-level status toward what the event log records, never regressing a
/// snapshot that is already ahead of a (possibly behind) log — see
/// [`status_rank`].
fn reconcile_doc(doc: &mut ProgressDoc, records: &[tools::WorkflowEventRecord]) {
    if let Some(terminal) = tools::event_log_terminal_status(records) {
        doc.status = terminal;
    }
    let phase_status = tools::event_phase_statuses(records);
    for phase in &mut doc.phases {
        if let Some(derived) = phase_status.get(&phase.id) {
            if status_rank(derived) > status_rank(&phase.status) {
                phase.status = derived.clone();
            }
        }
    }
}

/// Monotonic progress rank for a phase status, so reconciliation only advances a
/// phase toward its terminal state. The snapshot and the log are both written
/// per event, so they usually agree; on the rare disagreement (a dropped phase
/// write) the log is ahead, and ranking guarantees we never downgrade a phase
/// the snapshot already shows as `done`/`resumed`. Both are terminal (rank 2).
fn status_rank(status: &str) -> u8 {
    match status {
        "done" | "resumed" => 2,
        "running" => 1,
        _ => 0, // pending / unknown
    }
}

/// Whether a parsed doc should be surfaced: the open path requires a live run; the
/// refresh path shows whatever is in the active slot (it was confirmed live when
/// the viewer opened, so following it to completion is correct).
fn doc_visible(require_live: bool, status: &str, updated_at_ms: u64, now_ms: u64) -> bool {
    if require_live {
        return is_live(status, updated_at_ms, now_ms);
    }
    // Refresh may follow a live run into a terminal state, but a stale
    // `running` snapshot is an abandoned workflow and must not keep a dead
    // agent tree visible forever.
    status != "running" || is_live(status, updated_at_ms, now_ms)
}

/// Fold a parsed doc into a [`WorkflowView`], pulling each agent row via `get_row`
/// (plain or cached). The manifest read budget is shared across phases so a
/// hundreds-of-agents workflow cannot stall the render thread.
fn build_view(
    doc: ProgressDoc,
    mut get_row: impl FnMut(&str) -> Option<WorkflowAgentRow>,
) -> WorkflowView {
    let mut budget = MANIFEST_READ_BUDGET;
    let phases = doc
        .phases
        .into_iter()
        .map(|phase| {
            let total = phase.agent_ids.len();
            let agents = phase
                .agent_ids
                .iter()
                .filter_map(|id| {
                    if budget == 0 {
                        return None;
                    }
                    budget -= 1;
                    get_row(id)
                })
                .collect();
            WorkflowPhaseRow {
                step_id: Some(phase.id.clone()),
                plan_step: None,
                id: phase.id,
                kind: phase.kind,
                status: phase.status,
                round: phase.round,
                completed: phase.completed,
                failed: phase.failed,
                still_running: phase.still_running,
                total,
                agents,
            }
        })
        .collect();

    WorkflowView {
        run_id: doc.run_id,
        name: doc.name,
        description: doc.description,
        status: doc.status,
        mode: doc.mode,
        synthesizing: doc.synthesizing,
        phases,
    }
}

/// A manifest status that will not change again, so its parsed row is safe to
/// cache until the file's mtime moves.
fn is_terminal_status(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "stopped")
}

fn session_filter_enabled(session_id: Option<&str>) -> bool {
    session_id.is_some_and(|id| !id.trim().is_empty())
}

fn view_has_agents(view: &WorkflowView) -> bool {
    view.phases.iter().any(|phase| !phase.agents.is_empty())
}

fn view_has_progress_agents(view: &WorkflowView) -> bool {
    view.phases.iter().any(|phase| phase.total > 0)
}

fn doc_only_rows_allowed(doc: &ProgressDoc, session_id: Option<&str>) -> bool {
    !session_filter_enabled(session_id)
        || doc
            .parent_session_id
            .as_deref()
            .is_some_and(|actual| Some(actual) == session_id)
}

/// Join one agent's manifest with an mtime cache: a terminal-status row whose
/// manifest mtime is unchanged is returned from `cache` without reparsing;
/// everything else is read fresh so live rows never freeze.
fn read_agent_row_cached_scoped(
    store: Option<&PathBuf>,
    id: &str,
    now_secs: u64,
    cache: &mut WorkflowViewCache,
    session_id: Option<&str>,
    allow_missing_placeholder: bool,
) -> Option<WorkflowAgentRow> {
    let manifest_path = store.map(|s| s.join(format!("{id}.json")));
    let mtime = manifest_path
        .as_ref()
        .and_then(|p| fs::metadata(p).ok())
        .and_then(|m| m.modified().ok());
    if !session_filter_enabled(session_id) {
        if let (Some(mt), Some((cached_mt, cached_row))) = (mtime, cache.rows.get(id)) {
            if mt == *cached_mt && is_terminal_status(&cached_row.status) {
                return Some(cached_row.clone());
            }
        }
    }
    let Some(value) = manifest_path
        .as_ref()
        .and_then(|path| read_agent_manifest_value(path))
    else {
        return allow_missing_placeholder.then(|| doc_only_agent_row(id));
    };
    if !manifest_belongs_to_session(&value, session_id, false) {
        return None;
    }
    if let (Some(mt), Some((cached_mt, cached_row))) = (mtime, cache.rows.get(id)) {
        if mt == *cached_mt && is_terminal_status(&cached_row.status) {
            return Some(cached_row.clone());
        }
    }
    let row = agent_row_from_manifest(id, now_secs, &value);
    if let Some(mt) = mtime {
        cache.rows.insert(id.to_string(), (mt, row.clone()));
    }
    Some(row)
}

/// Join one agent's manifest. A missing/unreadable manifest degrades to a
/// `pending` placeholder rather than dropping the row.
fn read_agent_row(store: Option<&PathBuf>, id: &str, now_secs: u64) -> WorkflowAgentRow {
    let Some(store) = store else {
        return pending_agent_row(id);
    };
    read_agent_manifest_value(&store.join(format!("{id}.json"))).map_or_else(|| pending_agent_row(id), |value| agent_row_from_manifest(id, now_secs, &value))
}

fn read_agent_row_scoped(
    store: Option<&PathBuf>,
    id: &str,
    now_secs: u64,
    session_id: Option<&str>,
    allow_missing_placeholder: bool,
) -> Option<WorkflowAgentRow> {
    let Some(store) = store else {
        return allow_missing_placeholder.then(|| doc_only_agent_row(id));
    };
    let Some(value) = read_agent_manifest_value(&store.join(format!("{id}.json"))) else {
        return allow_missing_placeholder.then(|| doc_only_agent_row(id));
    };
    if !manifest_belongs_to_session(&value, session_id, false) {
        return None;
    }
    Some(agent_row_from_manifest(id, now_secs, &value))
}

fn read_agent_manifest_value(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str::<Value>(&text).ok()
}

fn doc_only_agent_row(id: &str) -> WorkflowAgentRow {
    WorkflowAgentRow {
        status: "running".to_string(),
        current_phase: Some("starting".to_string()),
        ..pending_agent_row(id)
    }
}

fn pending_agent_row(id: &str) -> WorkflowAgentRow {
    WorkflowAgentRow {
        id: id.to_string(),
        name: id.to_string(),
        status: "pending".to_string(),
        ..WorkflowAgentRow::default()
    }
}

fn agent_row_from_manifest(id: &str, now_secs: u64, value: &Value) -> WorkflowAgentRow {
    let mut row = pending_agent_row(id);
    if let Some(name) = value
        .get("label")
        .and_then(Value::as_str)
        .or_else(|| value.get("name").and_then(Value::as_str))
    {
        row.name = name.to_string();
    }
    if let Some(status) = value.get("status").and_then(Value::as_str) {
        row.status = status.to_string();
    }
    row.id = value
        .get("agentId")
        .and_then(Value::as_str)
        .map_or_else(|| id.to_string(), str::to_string);
    row.description = value
        .get("description")
        .and_then(Value::as_str)
        .map_or_else(String::new, str::to_string);
    row.subagent_type = value
        .get("subagentType")
        .and_then(Value::as_str)
        .map(str::to_string);
    row.current_tool = value
        .get("currentTool")
        .and_then(Value::as_str)
        .map(str::to_string);
    row.recent_tools = value
        .get("recentTools")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    row.model = manifest_model_label(value);
    row.route_reason = value
        .get("routeReason")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
        .map(str::to_string);
    row.tokens = value
        .get("tokenHistory")
        .and_then(Value::as_array)
        .map_or(0, |arr| arr.iter().filter_map(Value::as_u64).sum());
    row.tool_calls = value
        .get("toolCalls")
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok());
    row.output_file = value
        .get("outputFile")
        .and_then(Value::as_str)
        .map(str::to_string);
    row.error = value
        .get("error")
        .and_then(Value::as_str)
        .map(|error| compact_detail(error, 260));
    row.blocker = value
        .get("currentBlocker")
        .and_then(|blocker| blocker.get("detail"))
        .and_then(Value::as_str)
        .map(|detail| compact_detail(detail, 260));
    row.last_event = value
        .get("laneEvents")
        .and_then(Value::as_array)
        .and_then(|events| events.last())
        .and_then(lane_event_summary);
    // Live "what is the agent saying / doing right now" signals. `outputTail` is
    // the rolling streamed prose (bounded to the last few lines so the detail pane
    // stays compact); `currentPhase` is a transient wait/stream label; the
    // heartbeat is the agent's last manifest write.
    row.output_tail = value
        .get("outputTail")
        .and_then(Value::as_str)
        .map(|tail| last_tail_lines(tail, 3))
        .filter(|tail| !tail.is_empty());
    row.current_phase = value
        .get("currentPhase")
        .and_then(Value::as_str)
        .filter(|phase| !phase.is_empty())
        .map(str::to_string);
    row.idle_secs = value
        .get("lastActivityAt")
        .and_then(Value::as_u64)
        .map(|at| now_secs.saturating_sub(at));
    let created = value.get("createdAt").and_then(parse_epoch);
    let completed = value.get("completedAt").and_then(parse_epoch);
    row.elapsed_secs = match (created, completed) {
        (Some(start), Some(end)) => end.saturating_sub(start),
        (Some(start), None) => now_secs.saturating_sub(start),
        _ => 0,
    };
    row
}

fn lane_event_summary(event: &Value) -> Option<String> {
    let name = event
        .get("event")
        .and_then(Value::as_str)
        .or_else(|| event.get("status").and_then(Value::as_str))?;
    let detail = event
        .get("detail")
        .and_then(Value::as_str)
        .map(|detail| compact_detail(detail, 220));
    Some(match detail {
        Some(detail) if !detail.is_empty() => format!("{name}: {detail}"),
        _ => name.to_string(),
    })
}

/// Keep only the last `max_lines` non-blank lines of an agent's streamed output
/// tail, each trimmed, so the live "output" section in the agent detail pane shows
/// the most recent prose without unbounded growth (the manifest caps the tail at
/// ~2 KB; this caps the rendered slice).
fn last_tail_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn compact_detail(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut last_space = false;
    let mut truncated = false;
    for ch in text.chars() {
        if out.chars().count() >= max_chars {
            truncated = true;
            break;
        }
        if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    let mut out = out.trim().to_string();
    if truncated {
        out.push('…');
    }
    out
}

fn manifest_model_label(value: &Value) -> String {
    let model = string_field(value, &["model"]);
    let display = string_field(
        value,
        &[
            "resolvedModel",
            "resolved_model",
            "modelDisplayName",
            "model_display_name",
            "displayModel",
            "display_model",
            "modelName",
            "model_name",
        ],
    );
    let requested = string_field(value, &["requestedModel", "requested_model"]);

    // `model`/`resolvedModel` is the model that ACTUALLY ran. When it is only a
    // vague family alias ("gpt", "opus") we substitute a more specific id from
    // the manifest — but only one in the SAME provider family. A different-family
    // `requested` here means an on-wire pin was dropped and the agent ran on the
    // parent instead (e.g. a `gpt-5.5-fast` pin under an opus session that fell
    // back to opus); showing that requested id would relabel an opus run as gpt —
    // the very "gpt라 해놓고 실제로는 opus" mislabel we must not reintroduce.
    if let Some(model) = model {
        let model_short = short_model(&model);
        if is_generic_model_alias(&model_short) {
            if let Some(display) = &display {
                let display_short = short_model(display);
                if display_short != model_short && same_model_family(&display_short, &model_short) {
                    return display.clone();
                }
            }
            if let Some(requested) = &requested {
                let requested_short = short_model(requested);
                if requested_short != model_short && same_model_family(&requested_short, &model_short)
                {
                    return requested.clone();
                }
            }
        }
        return model;
    }
    if let Some(display) = &display {
        let display_short = short_model(display);
        if is_generic_model_alias(&display_short) {
            if let Some(requested) = &requested {
                let requested_short = short_model(requested);
                if requested_short != display_short
                    && same_model_family(&requested_short, &display_short)
                {
                    return requested.clone();
                }
            }
        }
        return display.clone();
    }
    requested.unwrap_or_default()
}

/// The provider family of a short model id (as produced by [`short_model`]).
/// Used to keep [`manifest_model_label`] from relabeling a run with a different
/// provider's id. `None` for ids we can't classify (custom/self-hosted), which
/// are then treated as non-matching so we never substitute across an unknown
/// boundary.
fn model_family(short: &str) -> Option<&'static str> {
    let s = short.trim();
    if s.is_empty() {
        return None;
    }
    if s == "opus" || s == "sonnet" || s == "haiku" || s.starts_with("claude") {
        return Some("anthropic");
    }
    if s == "gpt" || s.starts_with("gpt-") || s.starts_with("o1") || s.starts_with("o3") {
        return Some("openai");
    }
    if s.starts_with("gemini") {
        return Some("google");
    }
    if s.starts_with("grok") {
        return Some("xai");
    }
    None
}

/// Whether two short model ids belong to the same, known provider family.
/// Unknown families never match — better to show the honest resolved alias than
/// to risk relabeling a run with an unrelated id.
fn same_model_family(a: &str, b: &str) -> bool {
    matches!((model_family(a), model_family(b)), (Some(x), Some(y)) if x == y)
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Display label for a manifest model string.
///
/// This is the single shared model-label tokenizer for the whole TUI: the
/// sidebar, HUD badge, workflow modal, and agents detail view all route through
/// it so they show the same concrete model. It never collapses a resolved model
/// into a family alias such as `gpt` or `opus`; it removes provider prefixes and
/// extracts model-id tokens from human display names.
#[must_use]
pub fn short_model(model: &str) -> String {
    let mut model = model.trim();
    if model.is_empty() {
        return String::new();
    }

    for prefix in [
        "openai/",
        "openai:",
        "anthropic/",
        "anthropic:",
        "claude/",
        "claude:",
        "google/",
        "google:",
        "xai/",
        "xai:",
    ] {
        if let Some(rest) = model.strip_prefix(prefix) {
            model = rest.trim();
            break;
        }
    }

    let lower = model.to_ascii_lowercase();
    model_id_token(&lower).unwrap_or_else(|| model.to_string())
}

/// Whether a short model label is just a family alias (e.g. `gpt`, `opus`)
/// rather than a concrete resolved model id. Shared with the HUD badge so both
/// agree on when to prefer a resolved display name over the alias.
#[must_use]
pub fn is_generic_model_alias(label: &str) -> bool {
    matches!(label, "opus" | "sonnet" | "haiku" | "gpt")
}

fn model_id_token(text: &str) -> Option<String> {
    for prefix in ["claude-", "gpt-", "o3", "o1"] {
        if let Some(start) = text.find(prefix) {
            let raw = &text[start..];
            let mut token = raw
                .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '.'))
                .next()
                .unwrap_or(prefix)
                .trim_end_matches('-')
                .to_string();
            if !token.is_empty() {
                let suffix_start = token.len();
                append_model_suffix_words(&mut token, raw.get(suffix_start..).unwrap_or_default());
                return Some(token);
            }
        }
    }
    None
}

fn append_model_suffix_words(token: &mut String, rest: &str) {
    let rest = rest
        .trim_start_matches(|ch: char| ch.is_whitespace() || matches!(ch, '-' | '_' | ':' | '/'));
    for word in rest
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '.'))
        .filter(|word| !word.is_empty())
        .take(3)
    {
        let suffix = word.to_ascii_lowercase();
        if !is_model_suffix_word(&suffix) {
            break;
        }
        token.push('-');
        token.push_str(&suffix);
    }
}

fn is_model_suffix_word(word: &str) -> bool {
    matches!(
        word,
        "fast"
            | "mini"
            | "high"
            | "medium"
            | "low"
            | "max"
            | "turbo"
            | "preview"
            | "latest"
            | "nano"
            | "pro"
            | "flash"
            | "lite"
            | "thinking"
            | "reasoning"
    )
}

/// Accept an epoch as a JSON number or a numeric string (the manifest writes it
/// as a seconds string).
fn parse_epoch(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .unwrap_or(0)
}

/// Whether a progress snapshot represents a *live* workflow worth showing.
///
/// The engine never deletes `_active.progress.json`: a finished run leaves it on
/// disk with a terminal status (`completed`/`cancelled`/`budget_exhausted`), and
/// a crashed engine leaves a `running` snapshot that stops advancing. Both must
/// read as "not live" so `Ctrl+O` never opens a dead tree as if it were current.
/// `updated_at_ms == 0` means an older snapshot without the timestamp — trust the
/// status alone then.
fn is_live(status: &str, updated_at_ms: u64, now_ms: u64) -> bool {
    if status != "running" {
        return false;
    }
    updated_at_ms == 0 || now_ms.saturating_sub(updated_at_ms) <= STALE_AFTER_MS
}

/// Mirrors `tools::workflow_tools::cache::workflow_store_dir`: the
/// `ZO_WORKFLOW_STORE` override, else `<cwd>/.zo/workflows`.
fn progress_path() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("ZO_WORKFLOW_STORE") {
        return Some(PathBuf::from(dir).join(ACTIVE_PROGRESS_FILE));
    }
    let cwd = std::env::current_dir().ok()?;
    Some(
        cwd.join(".zo")
            .join("workflows")
            .join(ACTIVE_PROGRESS_FILE),
    )
}

/// Mirrors `tui_loop`'s agent-store resolution by delegating to the single
/// shared resolver: the `ZO_AGENT_STORE` override, else the per-project
/// state dir (`~/.zo/projects/<slug>/state/agents`). Reader and writer share
/// [`tools::agent_store_dir`], so they can never drift onto different paths.
fn agent_store_dir() -> Option<PathBuf> {
    tools::agent_store_dir().ok()
}

/// Freshness limits for the fan-out fallback, mirroring `tui_loop`'s sidebar gate
/// (`RUNNING_STALE_SECS` / `TERMINAL_GRACE_SECS`): a *running* agent's manifest is
/// silent during a model turn, so keep it visible longer than a *terminal* one
/// (which drops after a brief grace), but still reap abandoned rows promptly.
const AGENT_RUNNING_STALE_SECS: u64 = 5 * 60;
const AGENT_TERMINAL_GRACE_SECS: u64 = 8;

/// Synthesize a single-phase [`WorkflowView`] from the live `.zo/agents`
/// manifests, so `Ctrl+O` opens on a plain `SpawnMultiAgent` fan-out (which writes
/// no workflow progress file — only the dynamic workflow engine does).
///
/// `require_live` is the open vs. refresh split: the open path returns `None`
/// unless at least one agent is genuinely running (so a finished fan-out doesn't
/// open), while the refresh path returns any still-fresh row so an open viewer can
/// follow the agents to completion before going empty.
fn read_agents_fallback_since(
    require_live: bool,
    started_after_secs: u64,
    session_id: Option<&str>,
) -> Option<WorkflowView> {
    build_agents_fallback_since(
        &agent_store_dir()?,
        require_live,
        now_secs(),
        started_after_secs,
        session_id,
    )
}

/// Core of [`read_agents_fallback`], split out with the store dir and `now`
/// injected so it is testable without touching the process-wide `ZO_AGENT_STORE`
/// env var.
fn build_agents_fallback_since(
    store: &PathBuf,
    require_live: bool,
    now: u64,
    started_after_secs: u64,
    session_id: Option<&str>,
) -> Option<WorkflowView> {
    let mut rows: Vec<WorkflowAgentRow> = Vec::new();
    // Terminal agents that have aged past the 8s grace are held here, not dropped
    // outright: while the fan-out is still live they must keep counting toward the
    // fleet total/completed (see the re-admit below), or an early finisher
    // un-counts as it ages — the "0/5 → 1/5 → 0/4" flicker.
    let mut aged_terminal: Vec<WorkflowAgentRow> = Vec::new();
    let manifests = if require_live {
        super::agent_manifests::newest_first_fresh(store)
    } else {
        super::agent_manifests::newest_first_cached(store)
    };
    for (path, modified) in manifests.iter() {
        if rows.len() >= MANIFEST_READ_BUDGET {
            break;
        }
        let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Narrow the shared `SystemTime` mtime to epoch seconds at this boundary
        // so the fan-out fallback keeps its original `u64`-based age arithmetic.
        let modified_secs = modified
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let age = now.saturating_sub(modified_secs);
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if !manifest_belongs_to_session(&value, session_id, false) {
            continue;
        }
        if !manifest_value_created_after(&value, started_after_secs) {
            continue;
        }
        let row = read_agent_row(Some(store), id, now);
        if is_terminal_status(&row.status) {
            // A finished agent stops updating its manifest mtime, so it ages
            // immediately. Hold it aside if it is past the terminal grace rather
            // than dropping it: it must keep counting while the batch is live.
            if age > AGENT_TERMINAL_GRACE_SECS {
                aged_terminal.push(row);
            } else {
                rows.push(row);
            }
        } else if age <= AGENT_RUNNING_STALE_SECS {
            rows.push(row);
        }
        // else: a genuinely stale (zombie) running agent — dropped.
    }
    let still_running = rows
        .iter()
        .filter(|r| !is_terminal_status(&r.status))
        .count();
    // While any agent is still running, re-admit the aged terminal agents so the
    // fleet total/completed stay monotonic (no 5→4 / 1→0 flicker). Once nothing is
    // running the batch has settled, and the aged terminals fade with the grace.
    if still_running > 0 {
        rows.append(&mut aged_terminal);
    }
    if rows.is_empty() {
        return None;
    }
    let completed = rows.iter().filter(|r| r.status == "completed").count();
    let failed = rows.iter().filter(|r| r.status == "failed").count();
    // Open path: don't open a fan-out where nothing is running anymore.
    if require_live && still_running == 0 {
        return None;
    }
    let total = rows.len();
    // Fill the recorded tallies *and* set status from them, so the phase reads as
    // terminal once every agent is done — then `running_now`/`completed_now` (and
    // the spinners) settle correctly whether the phase is live or finished.
    let (phase_status, view_status) = if still_running > 0 {
        ("running", "running")
    } else {
        ("done", "completed")
    };
    Some(WorkflowView {
        // Synthetic view from per-agent manifests (no workflow run) → no event log.
        run_id: String::new(),
        name: "agents".to_string(),
        description: format!("{total} spawned agents"),
        status: view_status.to_string(),
        mode: "phases".to_string(),
        synthesizing: false,
        phases: vec![WorkflowPhaseRow {
            step_id: None,
            plan_step: None,
            id: "agents".to_string(),
            kind: "fanout".to_string(),
            status: phase_status.to_string(),
            round: 1,
            completed,
            failed,
            still_running,
            total,
            agents: rows,
        }],
    })
}

/// Flat per-agent snapshot for the Ctrl+G agents viewer: every manifest that
/// belongs to the visible session, terminal ones included — unlike the
/// workflow/fan-out views there is **no live gate**, so the viewer opens on a
/// finished batch too (Claude-Code style post-mortem browsing).
#[derive(Debug, Default, Clone)]
pub struct AgentRowsSnapshot {
    /// Non-terminal agents first, then newest `createdAt` first.
    pub rows: Vec<WorkflowAgentRow>,
    /// Session manifests hidden by the default freshness window
    /// (`started_after`) — the viewer's history toggle re-reads with
    /// `include_history` to reveal them.
    pub older_hidden: usize,
    /// Session-visible rows dropped by the manifest read budget (never a
    /// silent truncation — the viewer surfaces the count).
    pub capped: usize,
}

impl AgentRowsSnapshot {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.older_hidden == 0 && self.capped == 0
    }
}

/// Read the flat agent list for the agents viewer. `include_history` disables
/// the `started_after` freshness window (session scoping always applies), so a
/// resumed long-lived session can browse its earlier fleets on demand instead
/// of them crowding the default view.
#[must_use]
pub fn read_agent_rows_since(
    started_after_secs: u64,
    session_id: Option<&str>,
    include_history: bool,
) -> AgentRowsSnapshot {
    agent_store_dir().map_or_else(AgentRowsSnapshot::default, |store| {
        build_agent_rows_since(
            &store,
            now_secs(),
            started_after_secs,
            session_id,
            include_history,
        )
    })
}

/// Core of [`read_agent_rows_since`] with the store and clock injected for
/// tests. Iterates manifests newest-mtime-first, so when the read budget caps
/// the list it keeps the most recently active agents (live manifests are
/// rewritten constantly and always sort first).
fn build_agent_rows_since(
    store: &Path,
    now: u64,
    started_after_secs: u64,
    session_id: Option<&str>,
    include_history: bool,
) -> AgentRowsSnapshot {
    let manifests = super::agent_manifests::newest_first_cached(store);
    let mut snapshot = AgentRowsSnapshot::default();
    // Carry `createdAt` beside each row so the final order is stable spawn
    // order (creation time), not the mtime order manifests were scanned in.
    let mut dated: Vec<(u64, WorkflowAgentRow)> = Vec::new();
    for (path, modified) in manifests.iter() {
        let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if !manifest_belongs_to_session(&value, session_id, false) {
            continue;
        }
        if !include_history && !manifest_value_created_after(&value, started_after_secs) {
            snapshot.older_hidden += 1;
            continue;
        }
        if dated.len() >= MANIFEST_READ_BUDGET {
            snapshot.capped += 1;
            continue;
        }
        let modified_secs = modified
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let row = agent_row_from_manifest(id, now, &value);
        // A "running" manifest that stopped updating long ago is a crashed
        // zombie (same rule as the fan-out fallback); showing it alive forever
        // in a history-capable viewer would be a lie. Terminal rows stay.
        if !is_terminal_status(&row.status)
            && now.saturating_sub(modified_secs) > AGENT_RUNNING_STALE_SECS
        {
            continue;
        }
        let created = value
            .get("createdAt")
            .and_then(parse_epoch)
            .unwrap_or(modified_secs);
        dated.push((created, row));
    }
    dated.sort_by(|a, b| {
        let terminal_a = is_terminal_status(&a.1.status);
        let terminal_b = is_terminal_status(&b.1.status);
        terminal_a.cmp(&terminal_b).then_with(|| b.0.cmp(&a.0))
    });
    snapshot.rows = dated.into_iter().map(|(_, row)| row).collect();
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
        super::super::agent_manifests::cache_test_lock_for_tests()
    }

    #[test]
    fn short_model_maps_known_families() {
        assert_eq!(short_model("openai/gpt-5.5-fast"), "gpt-5.5-fast");
        assert_eq!(short_model("openai:gpt-5.5-fast"), "gpt-5.5-fast");
        assert_eq!(short_model("gpt-5.6-sol"), "gpt-5.6-sol");
        assert_eq!(short_model("OpenAI GPT-5.5 Fast"), "gpt-5.5-fast");
        assert_eq!(short_model("OpenAI:o3-mini-high"), "o3-mini-high");
        assert_eq!(short_model("OpenAI O3 Mini High"), "o3-mini-high");
        assert_eq!(short_model("anthropic/claude-opus-4-8"), "claude-opus-4-8");
        assert_eq!(short_model("anthropic:claude-opus-4-8"), "claude-opus-4-8");
        assert_eq!(short_model("mystery-model"), "mystery-model");
    }

    #[test]
    fn shared_model_tokenizer_matches_hud_and_sidebar_contract() {
        // The HUD badge (hud.rs) and the sidebar/workflow modal now both route
        // through this single tokenizer, so the cases each relied on must agree
        // here. Provider prefixes are stripped, suffix words are appended, and
        // family aliases are flagged so callers can prefer a resolved name.
        assert_eq!(short_model("  claude-opus-4-8  "), "claude-opus-4-8");
        assert_eq!(short_model("OpenAI:o3-mini-high"), "o3-mini-high");
        // When the id token is split off from human text, only recognized
        // suffix words are appended and trailing junk is dropped.
        assert_eq!(short_model("OpenAI GPT-5.5 Fast Experimental"), "gpt-5.5-fast");
        // A bare family name has no id token, so it falls back to the input.
        assert_eq!(short_model("gpt"), "gpt");
        assert_eq!(short_model(""), "");

        assert!(is_generic_model_alias("opus"));
        assert!(is_generic_model_alias("gpt"));
        assert!(!is_generic_model_alias("claude-opus-4-8"));
        assert!(!is_generic_model_alias(""));
        // The alias check operates on the tokenized label, mirroring how both
        // call sites invoke it (`is_generic_model_alias(&short_model(..))`).
        assert!(is_generic_model_alias(&short_model("anthropic:opus")));
        assert!(!is_generic_model_alias(&short_model(
            "anthropic/claude-opus-4-8"
        )));
    }

    #[test]
    fn parse_epoch_accepts_number_and_string() {
        assert_eq!(parse_epoch(&Value::from(1_700u64)), Some(1_700));
        assert_eq!(parse_epoch(&Value::from("1700")), Some(1_700));
        assert_eq!(parse_epoch(&Value::from("x")), None);
        assert_eq!(parse_epoch(&Value::Null), None);
    }

    #[test]
    fn agent_row_reads_explicit_tool_calls_without_counting_lane_events() {
        let dir = temp_store("tool-calls");
        fs::write(
            dir.join("legacy.json"),
            r#"{"name":"legacy","status":"running","laneEvents":[{"kind":"started"}],"createdAt":"100"}"#,
        )
        .unwrap();
        let legacy = read_agent_row(Some(&dir), "legacy", 160);
        assert_eq!(
            legacy.tool_calls, None,
            "laneEvents are lifecycle notes, not tool calls"
        );
        assert_eq!(legacy.elapsed_secs, 60);

        fs::write(
            dir.join("explicit.json"),
            r#"{"name":"explicit","status":"running","toolCalls":3,"laneEvents":[{"kind":"started"}]}"#,
        )
        .unwrap();
        let explicit = read_agent_row(Some(&dir), "explicit", 160);
        assert_eq!(explicit.tool_calls, Some(3));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_row_prefers_human_label_over_slug_name() {
        let dir = temp_store("label");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"find-architecture-srp","label":"find:architecture SRP","status":"running"}"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a0", 160);
        assert_eq!(row.name, "find:architecture SRP");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_row_prefers_resolved_model_over_generic_alias() {
        let dir = temp_store("resolved-model");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"a0","status":"running","model":"gpt","resolvedModel":"openai/gpt-5.5-fast"}"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a0", 160);
        assert_eq!(row.model, "openai/gpt-5.5-fast");
        assert_eq!(short_model(&row.model), "gpt-5.5-fast");

        fs::write(
            dir.join("a1.json"),
            r#"{"name":"a1","status":"running","model":"openai/gpt-5.5-fast","modelDisplayName":"OpenAI GPT-5.5 Fast"}"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a1", 160);
        assert_eq!(
            row.model, "openai/gpt-5.5-fast",
            "an already resolved model id should not be replaced by a display string"
        );
        assert_eq!(short_model(&row.model), "gpt-5.5-fast");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_row_prefers_requested_model_when_model_is_generic_alias() {
        let dir = temp_store("requested-model");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"a0","status":"running","model":"gpt","requestedModel":"openai/gpt-5.5-fast"}"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a0", 160);
        assert_eq!(row.model, "openai/gpt-5.5-fast");
        assert_eq!(short_model(&row.model), "gpt-5.5-fast");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_row_never_relabels_a_run_with_a_cross_family_requested_model() {
        // A `gpt-5.5-fast` pin that was dropped back to the opus parent: the
        // agent actually ran on opus (`model`/`resolvedModel`), but the manifest
        // still records the requested gpt id. The card must show what RAN (opus),
        // never the cross-family requested id — the "gpt라 해놓고 실제로는 opus"
        // mislabel. (Once the resolver honors the pin these agree; this guards the
        // still-gated frontmatter/env-override paths where they can still differ.)
        let dir = temp_store("cross-family-requested");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"a0","status":"running","model":"opus","resolvedModel":"opus","requestedModel":"gpt-5.5-fast"}"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a0", 160);
        assert_eq!(
            short_model(&row.model),
            "opus",
            "a cross-family requested id must not relabel an opus run as gpt"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_row_reads_detail_metadata_for_viewer() {
        let dir = temp_store("detail");
        fs::write(
            dir.join("a0.json"),
            r#"{
                "agentId":"agent-0",
                "name":"ops-risk",
                "description":"Check deployment risk",
                "subagentType":"analysis",
                "status":"failed",
                "outputFile":"/tmp/ops-risk.md",
                "currentBlocker":{"failureClass":"tool_runtime","detail":"tool   failed\nwhile reading"},
                "error":"gateway\nerror",
                "laneEvents":[{"event":"lane.failed","status":"failed","detail":"rate   limited"}]
            }"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a0", 160);
        assert_eq!(row.id, "agent-0");
        assert_eq!(row.description, "Check deployment risk");
        assert_eq!(row.subagent_type.as_deref(), Some("analysis"));
        assert_eq!(row.output_file.as_deref(), Some("/tmp/ops-risk.md"));
        assert_eq!(row.blocker.as_deref(), Some("tool failed while reading"));
        assert_eq!(row.error.as_deref(), Some("gateway error"));
        assert_eq!(row.last_event.as_deref(), Some("lane.failed: rate limited"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_row_reads_live_output_phase_and_heartbeat() {
        // The "what is the agent doing right now" signals the modal surfaces while
        // an agent is still running: the streamed output tail (bounded to the last
        // few lines), the transient phase, and the idle heartbeat.
        let dir = temp_store("live");
        fs::write(
            dir.join("a0.json"),
            r#"{
                "agentId":"agent-0",
                "name":"scan",
                "status":"running",
                "currentPhase":"thinking",
                "lastActivityAt":150,
                "outputTail":"first line\n\nsecond line\nthird line\nfourth line"
            }"#,
        )
        .unwrap();
        let row = read_agent_row(Some(&dir), "a0", 160);
        assert_eq!(row.status, "running");
        assert_eq!(row.current_phase.as_deref(), Some("thinking"));
        // now_secs(160) - lastActivityAt(150) = 10s idle.
        assert_eq!(row.idle_secs, Some(10));
        // Only the last 3 non-blank lines are kept.
        assert_eq!(
            row.output_tail.as_deref(),
            Some("second line\nthird line\nfourth line")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn last_tail_lines_keeps_last_nonblank_lines() {
        assert_eq!(last_tail_lines("a\nb\nc\nd", 2), "c\nd");
        assert_eq!(last_tail_lines("a\n\n  \nb", 3), "a\nb");
        assert_eq!(last_tail_lines("only", 3), "only");
        assert_eq!(last_tail_lines("", 3), "");
    }

    #[test]
    fn is_live_gates_terminal_and_stale_runs() {
        let now = 1_000_000_000u64;
        // A fresh running snapshot is live.
        assert!(is_live("running", now, now));
        // No timestamp (older writer) → trust the status.
        assert!(is_live("running", 0, now));
        // Terminal statuses are never live, however fresh.
        assert!(!is_live("completed", now, now));
        assert!(!is_live("cancelled", now, now));
        assert!(!is_live("budget_exhausted", now, now));
        // A running snapshot that stopped advancing past the backstop = dead.
        assert!(!is_live("running", now - STALE_AFTER_MS - 1, now));
        // Within the backstop it is still live (slow phase, not a crash).
        assert!(is_live("running", now - STALE_AFTER_MS + 1, now));
    }

    #[test]
    fn refresh_path_follows_run_to_terminal_open_path_does_not() {
        let now = 30_000_000u64;
        // Open path (require_live=true): a terminal snapshot must NOT open; live does.
        assert!(!doc_visible(true, "completed", now, now));
        assert!(!doc_visible(true, "cancelled", now, now));
        assert!(doc_visible(true, "running", now, now));
        // Refresh path (require_live=false): a terminal snapshot IS shown, so an
        // already-open viewer flips to completed and stops spinning (the regression
        // this guards). A fresh running snapshot is shown too.
        assert!(doc_visible(false, "completed", now, now));
        assert!(doc_visible(false, "running", now, now));
        // But a stale running snapshot is an abandoned workflow, not something an
        // open modal should keep repainting forever.
        assert!(!doc_visible(
            false,
            "running",
            now - STALE_AFTER_MS - 1,
            now
        ));
    }

    #[test]
    fn workflow_summary_session_scope_ignores_previous_run_snapshot() {
        let doc = ProgressDoc {
            name: "previous".to_string(),
            status: "running".to_string(),
            updated_at_ms: 200_000,
            phases: vec![ProgressPhase {
                id: "old-phase".to_string(),
                status: "running".to_string(),
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };

        assert!(
            !doc_updated_after(&doc, 300),
            "snapshot updated before this session should be hidden"
        );
        assert!(
            doc_updated_after(&doc, 150),
            "snapshot updated inside this session should remain visible"
        );
        assert!(
            !doc_updated_after(
                &ProgressDoc {
                    updated_at_ms: 0,
                    ..doc
                },
                150
            ),
            "legacy timestamp-free snapshots cannot be attributed to this session"
        );
    }

    fn temp_store(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zo-wfcache-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn cache_serves_terminal_row_on_unchanged_mtime() {
        let dir = temp_store("terminal");
        let store = Some(dir.clone());
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"a0","status":"completed","tokenHistory":[10,20]}"#,
        )
        .unwrap();
        let mut cache = WorkflowViewCache::default();
        let first =
            read_agent_row_cached_scoped(store.as_ref(), "a0", 0, &mut cache, None, true).expect("row");
        assert_eq!(first.status, "completed");
        assert_eq!(first.tokens, 30);
        // Poison the cached copy: a second read with the file untouched must serve
        // the (poisoned) cache, proving it skipped the parse.
        cache.rows.get_mut("a0").unwrap().1.name = "FROM_CACHE".to_string();
        let second =
            read_agent_row_cached_scoped(store.as_ref(), "a0", 0, &mut cache, None, true).expect("row");
        assert_eq!(second.name, "FROM_CACHE");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_synthesizes_a_view_from_running_manifests() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-live");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"running","createdAt":"100","currentTool":"grep","model":"claude-opus-4-8"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a1.json"),
            r#"{"name":"finder-1","status":"completed","createdAt":"100","model":"claude-sonnet-4-6"}"#,
        )
        .unwrap();
        // Open path: at least one agent is running, so a synthetic single-phase
        // view is produced (the model label flows through too).
        let view = build_agents_fallback_since(&dir, true, now_secs(), 0, None)
            .expect("live fan-out opens");
        assert_eq!(view.phases.len(), 1);
        assert_eq!(view.status, "running");
        let phase = &view.phases[0];
        assert_eq!(phase.step_id, None, "synthetic fan-out stays unscoped");
        assert_eq!(phase.plan_step, None);
        assert_eq!(phase.total, 2);
        assert_eq!(phase.status, "running");
        assert_eq!(phase.still_running, 1);
        assert_eq!(phase.completed, 1);
        assert_eq!(
            phase
                .agents
                .iter()
                .filter(|a| a.status == "running")
                .count(),
            1
        );
        assert!(phase.agents.iter().any(|a| a.model == "claude-opus-4-8"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_open_path_bypasses_stale_manifest_file_set_cache() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-fresh-open");
        super::super::agent_manifests::reset_cache_for_tests();
        assert!(
            super::super::agent_manifests::newest_first_cached(&dir).is_empty(),
            "prime an empty cached file set before agents spawn"
        );
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"running","createdAt":"100","currentTool":"grep"}"#,
        )
        .unwrap();
        assert!(
            super::super::agent_manifests::newest_first_cached(&dir).is_empty(),
            "the ordinary cached read still hides the just-created manifest"
        );

        let view = build_agents_fallback_since(&dir, true, now_secs(), 0, None)
            .expect("Ctrl+O open path must see just-created agent manifests");
        assert_eq!(view.phases[0].total, 1);
        assert_eq!(
            view.phases[0].agents[0].current_tool,
            Some("grep".to_string())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agents_fallback_summary_populates_one_fleet_phase_for_the_sidebar() {
        let _guard = manifest_cache_test_lock();
        // Regression: the sidebar Fleet stayed blank during pre-analysis/fan-out
        // because read_summary_since read only the progress doc. The synthetic
        // fan-out view must map onto a WorkflowSummary with a non-empty `phases`
        // (so push_workflow_section renders a Fleet bar) and matching tallies.
        let dir = temp_store("fallback-summary");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"running","createdAt":"100"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a1.json"),
            r#"{"name":"finder-1","status":"completed","createdAt":"100"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a2.json"),
            r#"{"name":"finder-2","status":"failed","createdAt":"100"}"#,
        )
        .unwrap();
        let view = build_agents_fallback_since(&dir, true, now_secs(), 0, None)
            .expect("live fan-out opens");
        let summary = summary_from_agents_view(&view);

        assert_eq!(summary.phases.len(), 1, "sidebar Fleet needs a phase to render");
        let fleet = &summary.phases[0];
        assert_eq!(fleet.total, 3);
        assert_eq!(fleet.completed, 1);
        assert_eq!(fleet.failed, 1);
        assert_eq!(fleet.running, 1);
        assert_eq!(
            fleet.step_id, None,
            "a synthetic agents phase must never claim a Todo step"
        );
        let mut agent_ids = fleet.agent_ids.clone();
        agent_ids.sort();
        assert_eq!(agent_ids, ["a0", "a1", "a2"]);
        assert_eq!(summary.total_agents, 3);
        assert_eq!(summary.completed_agents, 1);
        assert_eq!(summary.failed_agents, 1);
        assert_eq!(summary.running_agents, 1);
        assert_eq!(summary.status, "running");
        // 2 of 3 terminal ⇒ 66%.
        assert_eq!(summary.progress_percent, 66);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn correlation_agent_ids_require_nonblank_trimmed_unique_values() {
        let valid = vec!["a0".to_string(), "a1".to_string()];
        assert_eq!(correlatable_agent_ids(&valid), valid);

        for invalid in [
            vec![String::new()],
            vec![" a0 ".to_string()],
            vec!["a0".to_string(), "a0".to_string()],
        ] {
            assert!(
                correlatable_agent_ids(&invalid).is_empty(),
                "unsafe ids must not reach exact attribution: {invalid:?}"
            );
        }
    }

    #[test]
    fn ghost_running_doc_with_no_live_manifests_zeroes_the_hud_running_count() {
        let doc = ProgressDoc {
            name: "fanout".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "verify".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string(), "a1".to_string(), "a2".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        let raw = summarize_doc(&doc).expect("running doc summarizes");
        assert_eq!(raw.running_agents, 3, "doc-only summary reproduces the ghost count");
        let reconciled = reconcile_summary_with_live_count(raw, 0, false);
        assert_eq!(reconciled.running_agents, 0);
        assert_eq!(reconciled.phases[0].running, 0);

        let view = build_view(doc, |_| None);
        assert!(
            !view_has_agents(&view),
            "Ctrl+O would have no live manifest rows for this ghost doc"
        );
    }

    #[test]
    fn fresh_doc_only_workflow_spawn_keeps_running_count_during_manifest_birth_grace() {
        let now = 1_000u64;
        let doc = ProgressDoc {
            name: "workflow".to_string(),
            parent_session_id: Some("current-session".to_string()),
            status: "running".to_string(),
            mode: "phases".to_string(),
            updated_at_ms: now * 1000,
            phases: vec![ProgressPhase {
                id: "work".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string(), "a1".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        assert!(doc_only_running_grace_active(
            &doc,
            now,
            Some("current-session")
        ));
        assert!(!doc_only_running_grace_active(
            &doc,
            now,
            Some("other-session")
        ));
        let legacy_doc = ProgressDoc {
            name: "workflow".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            updated_at_ms: now * 1000,
            phases: vec![ProgressPhase {
                id: "work".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string(), "a1".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        assert!(doc_session_matches(&doc, Some("current-session")));
        assert!(!doc_session_matches(&doc, Some("other-session")));
        assert!(doc_session_matches(&legacy_doc, Some("current-session")));
        assert!(!doc_only_running_grace_active(
            &legacy_doc,
            now,
            Some("current-session")
        ));

        let raw = summarize_doc(&doc).expect("fresh running doc summarizes");
        let reconciled = reconcile_summary_with_live_count(raw, 0, true);
        assert_eq!(
            reconciled.running_agents, 2,
            "a just-spawned workflow must remain visibly running before manifests appear"
        );
        assert_eq!(reconciled.phases[0].running, 2);

        let expired_now = now + DOC_ONLY_RUNNING_GRACE_SECS + 1;
        assert!(!doc_only_running_grace_active(
            &doc,
            expired_now,
            Some("current-session")
        ));
    }

    #[test]
    fn session_stamped_workflow_view_opens_with_doc_only_agent_placeholders() {
        let doc = ProgressDoc {
            name: "workflow".to_string(),
            parent_session_id: Some("current-session".to_string()),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "implement".to_string(),
                kind: "fanout".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string(), "a1".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        let allow = doc_only_rows_allowed(&doc, Some("current-session"));
        assert!(allow);
        let view = build_view(doc, |id| {
            read_agent_row_scoped(None, id, 0, Some("current-session"), allow)
        });

        assert!(view_has_agents(&view), "Ctrl+O must have rows before manifests land");
        assert_eq!(view.phases[0].total, 2);
        assert_eq!(view.phases[0].agents.len(), 2);
        assert_eq!(view.phases[0].agents[0].status, "running");
        assert_eq!(view.phases[0].agents[0].current_phase.as_deref(), Some("starting"));
    }

    #[test]
    fn workflow_doc_only_placeholders_require_matching_session_stamp() {
        let foreign = ProgressDoc {
            name: "workflow".to_string(),
            parent_session_id: Some("other-session".to_string()),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "implement".to_string(),
                kind: "fanout".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        assert!(!doc_only_rows_allowed(&foreign, Some("current-session")));
        assert!(!doc_session_matches(&foreign, Some("current-session")));

        let legacy = ProgressDoc {
            name: "workflow".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "implement".to_string(),
                kind: "fanout".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        assert!(!doc_only_rows_allowed(&legacy, Some("current-session")));
        assert!(doc_session_matches(&legacy, Some("current-session")));
    }

    #[test]
    fn live_manifest_keeps_the_running_count_but_terminal_stale_and_foreign_do_not() {
        let dir = temp_store("live-reconcile");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"live","status":"running","parentSessionId":"sess-a"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a1.json"),
            r#"{"name":"done","status":"completed","parentSessionId":"sess-a"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a2.json"),
            r#"{"name":"foreign","status":"running","parentSessionId":"sess-b"}"#,
        )
        .unwrap();
        let doc = ProgressDoc {
            name: "fanout".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "verify".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string(), "a1".to_string(), "a2".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };

        assert_eq!(
            live_running_manifest_count_in(&dir, now_secs(), &doc, Some("sess-a")),
            1,
            "only the fresh non-terminal manifest for this session counts as live"
        );
        let raw = summarize_doc(&doc).expect("running doc summarizes");
        let reconciled = reconcile_summary_with_live_count(raw, 1, false);
        assert_eq!(
            reconciled.running_agents, 3,
            "one live manifest means the workflow doc is still alive, so preserve its recorded running count"
        );

        assert_eq!(
            live_running_manifest_count_in(&dir, now_secs(), &doc, Some("sess-missing")),
            0
        );
        let _ = fs::remove_dir_all(&dir);
    }


    #[test]
    fn fallback_does_not_read_unstamped_manifest_when_strict_scope_is_empty() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-unstamped-no-session");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"unstamped-live","status":"running","createdAt":"200"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a1.json"),
            r#"{"name":"foreign-live","status":"running","createdAt":"210","parentSessionId":"other"}"#,
        )
        .unwrap();

        assert!(
            build_agents_fallback_since(&dir, true, now_secs(), 150, Some("session-a")).is_none(),
            "unstamped rows cannot be safely attributed to session-a"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_keeps_finished_agents_counted_while_batch_is_live() {
        let _guard = manifest_cache_test_lock();
        // Regression ("0/5 → 1/5 → 0/4"): a finished agent stops touching its
        // manifest, so it ages immediately; past the 8s terminal grace the old
        // code dropped it from the live fan-out, shrinking the denominator
        // (total 2→1) and un-counting the completion (1→0) while the batch was
        // still running. It must stay counted until the whole batch settles.
        let dir = temp_store("fallback-aged-terminal");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"running","createdAt":"100"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a1.json"),
            r#"{"name":"finder-1","status":"completed","createdAt":"100"}"#,
        )
        .unwrap();
        // 60s in the "future": the running agent (5-min stale limit) survives; the
        // completed agent is well past the 8s terminal grace, so the old code
        // dropped it. With the fix it is re-admitted because a0 is still running.
        let aged_now = now_secs() + 60;
        let view = build_agents_fallback_since(&dir, true, aged_now, 0, None)
            .expect("a live batch with an aged-out finisher still opens");
        let phase = &view.phases[0];
        assert_eq!(phase.total, 2, "finished agent must still count toward the total");
        assert_eq!(phase.completed, 1, "the completion must not un-count as it ages");
        assert_eq!(phase.still_running, 1);

        // Once nothing is running, the aged terminals fade with the grace.
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"completed","createdAt":"100"}"#,
        )
        .unwrap();
        assert!(
            build_agents_fallback_since(&dir, true, now_secs() + 60, 0, None).is_none(),
            "a fully-aged, fully-terminal batch fades (nothing running)"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_view_ignores_previous_session_agent_manifests() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-session-scope");
        fs::write(
            dir.join("old.json"),
            r#"{"name":"old-session","status":"running","createdAt":"100","model":"openai/gpt-5.5-fast"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("new.json"),
            r#"{"name":"current-session","status":"running","createdAt":"200","model":"openai/gpt-5.5-fast"}"#,
        )
        .unwrap();

        let view = build_agents_fallback_since(&dir, true, now_secs(), 150, None)
            .expect("current fanout opens");
        let phase = &view.phases[0];

        assert_eq!(phase.total, 1);
        assert_eq!(phase.still_running, 1);
        assert_eq!(phase.agents[0].name, "current-session");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_view_filters_agent_manifests_by_parent_session_id() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-parent-session-scope");
        fs::write(
            dir.join("foreign.json"),
            r#"{"name":"foreign-session","status":"running","createdAt":"200","parentSessionId":"session-b"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("current.json"),
            r#"{"name":"current-session","status":"running","createdAt":"200","parentSessionId":"session-a"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("unstamped.json"),
            r#"{"name":"legacy-unstamped","status":"running","createdAt":"200"}"#,
        )
        .unwrap();

        let view = build_agents_fallback_since(&dir, true, now_secs(), 150, Some("session-a"))
            .expect("current session fanout opens");
        let phase = &view.phases[0];

        assert_eq!(phase.total, 1);
        assert_eq!(phase.agents[0].name, "current-session");
        assert!(
            build_agents_fallback_since(&dir, true, now_secs(), 150, Some("missing-session"))
                .is_none(),
            "foreign and unstamped manifests must not open another session's panel"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Ctrl+G 리더: live 게이트가 없어 전-종결 배치도 행으로 나온다 (fan-out
    /// 폴백은 `still_running==0` 이면 None — 이 뷰어는 사후 열람이 목적).
    #[test]
    fn agent_rows_include_terminal_agents_without_live_gate() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("agent-rows-terminal");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"done-one","status":"completed","createdAt":"200","parentSessionId":"sess-a"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("a1.json"),
            r#"{"name":"failed-one","status":"failed","createdAt":"210","parentSessionId":"sess-a"}"#,
        )
        .unwrap();

        let snapshot = build_agent_rows_since(&dir, now_secs(), 0, Some("sess-a"), false);
        assert_eq!(snapshot.rows.len(), 2, "terminal agents stay browsable");
        assert!(
            build_agents_fallback_since(&dir, true, now_secs(), 0, Some("sess-a")).is_none(),
            "sanity: the live-gated fan-out path would have hidden this batch"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// 기본 뷰는 `started_after` 이후만 — 재개된 장수 세션의 16시간 전 에이전트가
    /// first-class 로 나오던 문제. 숨긴 개수는 `older_hidden` 으로 정직하게 세고,
    /// history 토글이 같은 세션 범위 안에서만 되살린다.
    #[test]
    fn agent_rows_history_toggle_reveals_older_session_manifests() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("agent-rows-history");
        fs::write(
            dir.join("old.json"),
            r#"{"name":"yesterday","status":"completed","createdAt":"100","parentSessionId":"sess-a"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("new.json"),
            r#"{"name":"today","status":"completed","createdAt":"500","parentSessionId":"sess-a"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("foreign-old.json"),
            r#"{"name":"other","status":"completed","createdAt":"100","parentSessionId":"sess-b"}"#,
        )
        .unwrap();

        let fresh = build_agent_rows_since(&dir, now_secs(), 300, Some("sess-a"), false);
        assert_eq!(fresh.rows.len(), 1);
        assert_eq!(fresh.rows[0].name, "today");
        assert_eq!(fresh.older_hidden, 1, "the pre-window manifest is counted, not silent");

        let history = build_agent_rows_since(&dir, now_secs(), 300, Some("sess-a"), true);
        assert_eq!(history.rows.len(), 2, "history reveals the older session manifest");
        assert_eq!(history.older_hidden, 0);
        assert!(
            history.rows.iter().all(|row| row.name != "other"),
            "history never crosses the session boundary"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// 정렬: 비터미널 우선, 그 안에서 createdAt 내림차순 — 리스트 맨 위가 항상
    /// "지금 뛰는 것들, 최신 순".
    #[test]
    fn agent_rows_sort_running_first_then_newest_created() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("agent-rows-sort");
        fs::write(
            dir.join("d.json"),
            r#"{"name":"done-newest","status":"completed","createdAt":"900"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("r1.json"),
            r#"{"name":"run-older","status":"running","createdAt":"300"}"#,
        )
        .unwrap();
        fs::write(
            dir.join("r2.json"),
            r#"{"name":"run-newer","status":"running","createdAt":"400"}"#,
        )
        .unwrap();

        let snapshot = build_agent_rows_since(&dir, now_secs(), 0, None, false);
        let names: Vec<&str> = snapshot.rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["run-newer", "run-older", "done-newest"],
            "non-terminal rows lead even when a terminal row is newer"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn workflow_view_omits_foreign_session_agent_rows() {
        let doc = ProgressDoc {
            name: "session-scoped-workflow".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "read".to_string(),
                kind: "fanout".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["current".to_string(), "foreign".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };

        let view = build_view(doc, |id| match id {
            "current" => Some(WorkflowAgentRow {
                id: id.to_string(),
                name: "current-session".to_string(),
                status: "running".to_string(),
                ..WorkflowAgentRow::default()
            }),
            _ => None,
        });

        assert_eq!(view.phases[0].agents.len(), 1);
        assert_eq!(view.phases[0].step_id.as_deref(), Some("read"));
        assert_eq!(
            view.phases[0].plan_step, None,
            "the App joins Todo labels without adding reader IO"
        );
        assert_eq!(view.phases[0].agents[0].name, "current-session");
    }


    #[test]
    fn progress_summary_surfaces_selective_repair_states() {
        let doc = ProgressDoc {
            run_id: "run".to_string(),
            parent_session_id: None,
            name: "repair".to_string(),
            description: String::new(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            synthesizing: false,
            updated_at_ms: 1,
            phases: vec![ProgressPhase {
                id: "verify".to_string(),
                kind: "phase".to_string(),
                status: "running".to_string(),
                round: 1,
                completed: 1,
                failed: 0,
                still_running: 0,
                agent_ids: vec!["a".to_string()],
                carried: 2,
                retried: 1,
                skipped: 0,
                findings: 1,
                blocked: 1,
                invalidated: 1,
                selective_retries: 1,
            }],
        };
        let summary = summarize_doc(&doc).expect("summary");
        assert!(summary.current_phase_status.contains("selective"));
        assert!(summary.current_phase_status.contains("findings"));
        assert!(summary.current_phase_status.contains("blocked"));
        assert!(summary.phases[0].status.contains("selective"));
    }

    #[test]
    fn summary_reports_current_and_next_phase_without_manifests() {
        let doc = ProgressDoc {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![
                ProgressPhase {
                    id: "read".to_string(),
                    kind: "fanout".to_string(),
                    status: "done".to_string(),
                    agent_ids: vec!["a0".to_string(), "a1".to_string()],
                    completed: 2,
                    ..ProgressPhase::default()
                },
                ProgressPhase {
                    id: "test".to_string(),
                    kind: "fanout".to_string(),
                    status: "running".to_string(),
                    agent_ids: vec!["a2".to_string()],
                    ..ProgressPhase::default()
                },
                ProgressPhase {
                    id: "synthesize".to_string(),
                    kind: "single".to_string(),
                    status: "pending".to_string(),
                    agent_ids: Vec::new(),
                    ..ProgressPhase::default()
                },
            ],
            ..ProgressDoc::default()
        };

        let summary = summarize_doc(&doc).expect("workflow summary");
        assert_eq!(summary.name, "code-health");
        assert_eq!(summary.status, "running");
        assert_eq!(summary.mode, "phases");
        assert_eq!(summary.current_phase, "test");
        assert_eq!(summary.current_phase_status, "running");
        assert_eq!(summary.current_phase_index, 2);
        assert_eq!(summary.total_phases, 3);
        assert_eq!(summary.next_phase.as_deref(), Some("synthesize"));
        assert_eq!(summary.total_agents, 3);
        assert_eq!(summary.completed_phases, 1);
        assert_eq!(summary.completed_agents, 2);
        assert_eq!(summary.failed_agents, 0);
        assert_eq!(summary.running_agents, 1);
        let active_phase = summary
            .phases
            .iter()
            .find(|phase| phase.id == "test")
            .expect("active Fleet phase");
        assert_eq!(active_phase.step_id.as_deref(), Some("test"));
        assert_eq!(active_phase.agent_ids, ["a2"]);
        // Phase 1 done (100), phase 2 running with 1 in-flight agent of 1 total
        // earns INFLIGHT_AGENT_FRACTION (0.3 → 30%), phase 3 pending (0):
        // (100 + 30 + 0) / 3 = 43. The within-agent credit lifts the bar off 0
        // while the active phase has no finished agent yet (was a flat 33).
        assert_eq!(summary.progress_percent, 43);
    }

    #[test]
    fn summary_progress_includes_active_phase_recorded_tally() {
        let doc = ProgressDoc {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![
                ProgressPhase {
                    id: "read".to_string(),
                    status: "done".to_string(),
                    agent_ids: vec!["a0".to_string(), "a1".to_string()],
                    completed: 2,
                    ..ProgressPhase::default()
                },
                ProgressPhase {
                    id: "test".to_string(),
                    status: "running".to_string(),
                    agent_ids: vec!["a2".to_string(), "a3".to_string()],
                    completed: 1,
                    failed: 1,
                    ..ProgressPhase::default()
                },
            ],
            ..ProgressDoc::default()
        };

        let summary = summarize_doc(&doc).expect("workflow summary");
        // Phase 1 done (100). Phase 2 has all agents finished (1 done + 1 failed
        // == 2 total) but its status is still "running" (barrier not yet flipped),
        // so it is held at 99 — a non-terminal phase never reads 100 (honesty:
        // the work is staged but not committed). (100 + 99) / 2 = 99 (was 100).
        assert_eq!(summary.progress_percent, 99);
        assert_eq!(summary.completed_phases, 1);
        assert_eq!(summary.completed_agents, 3);
        assert_eq!(summary.failed_agents, 1);
        assert_eq!(summary.running_agents, 0);
    }

    #[test]
    fn single_running_phase_progress_is_nonzero_monotonic_and_below_100() {
        // The user-reported defect: a single fan-out phase reads 0% the whole time
        // until its first agent finishes. With within-agent credit the percent (a)
        // is > 0 as soon as agents are spawned/running, (b) never decreases as
        // agents finish, and (c) stays < 100 until the phase flips terminal.
        let total = 4usize;
        let ids: Vec<String> = (0..total).map(|i| format!("a{i}")).collect();
        let mut last = 0u8;
        let mut seen_nonzero_while_running = false;

        // Walk completed = 0..=total with the matching still_running, phase running.
        for done in 0..=total {
            let still_running = total - done;
            let status = "running"; // phase not yet past its barrier
            let doc = ProgressDoc {
                name: "scan".to_string(),
                status: "running".to_string(),
                mode: "phases".to_string(),
                phases: vec![ProgressPhase {
                    id: "scan".to_string(),
                    kind: "fanout".to_string(),
                    status: status.to_string(),
                    agent_ids: ids.clone(),
                    completed: done,
                    still_running,
                    ..ProgressPhase::default()
                }],
                ..ProgressDoc::default()
            };
            let pct = summarize_doc(&doc).expect("summary").progress_percent;
            assert!(pct >= last, "percent regressed: {last} -> {pct} at done={done}");
            assert!(pct < 100, "a running phase must stay below 100%: {pct} at done={done}");
            if still_running > 0 || done > 0 {
                assert!(pct > 0, "a phase with active/finished agents must be > 0%: {pct} at done={done}");
                seen_nonzero_while_running = true;
            }
            last = pct;
        }
        assert!(seen_nonzero_while_running, "test should exercise the running window");

        // Once the phase flips terminal, it reads 100%.
        let done_doc = ProgressDoc {
            name: "scan".to_string(),
            status: "completed".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "scan".to_string(),
                kind: "fanout".to_string(),
                status: "done".to_string(),
                agent_ids: ids,
                completed: total,
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        assert_eq!(
            summarize_doc(&done_doc).expect("summary").progress_percent,
            100,
            "a terminal phase reads 100%"
        );
    }

    #[test]
    fn summary_ignores_stale_running_count_on_terminal_phase() {
        let doc = ProgressDoc {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            phases: vec![ProgressPhase {
                id: "cleanup".to_string(),
                status: "done".to_string(),
                agent_ids: vec!["a0".to_string(), "a1".to_string()],
                completed: 2,
                still_running: 2,
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };

        let summary = summarize_doc(&doc).expect("workflow summary");
        assert_eq!(summary.progress_percent, 100);
        assert_eq!(summary.completed_phases, 1);
        assert_eq!(summary.completed_agents, 2);
        assert_eq!(
            summary.running_agents, 0,
            "terminal phases must not leak stale still_running into HUD/sidebar"
        );
    }

    #[test]
    fn fallback_open_path_skips_a_finished_fanout_but_refresh_shows_it() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-done");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"completed"}"#,
        )
        .unwrap();
        // Open path: nothing running → don't open a finished fan-out.
        assert!(build_agents_fallback_since(&dir, true, now_secs(), 0, None).is_none());
        // Refresh path: still-fresh terminal rows are shown so an already-open
        // viewer flips to completed instead of freezing.
        let view = build_agents_fallback_since(&dir, false, now_secs(), 0, None)
            .expect("refresh shows finished");
        assert_eq!(view.status, "completed");
        assert_eq!(view.phases[0].status, "done");
        assert_eq!(view.phases[0].completed, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_drops_stale_manifests_and_returns_none() {
        let _guard = manifest_cache_test_lock();
        let dir = temp_store("fallback-stale");
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"finder-0","status":"running"}"#,
        )
        .unwrap();
        // `now` far ahead of the file mtime → past RUNNING_STALE_SECS → dropped.
        let stale_now = now_secs() + AGENT_RUNNING_STALE_SECS + 100;
        assert!(build_agents_fallback_since(&dir, false, stale_now, 0, None).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_never_serves_a_running_row() {
        let dir = temp_store("running");
        let store = Some(dir.clone());
        fs::write(
            dir.join("a0.json"),
            r#"{"name":"live","status":"running","currentTool":"bash"}"#,
        )
        .unwrap();
        let mut cache = WorkflowViewCache::default();
        let first =
            read_agent_row_cached_scoped(store.as_ref(), "a0", 0, &mut cache, None, true).expect("row");
        assert_eq!(first.status, "running");
        // Even with the file untouched (same mtime), a running row is re-read, not
        // served from cache — so live status/currentTool/elapsed never freeze.
        cache.rows.get_mut("a0").unwrap().1.name = "FROM_CACHE".to_string();
        let second =
            read_agent_row_cached_scoped(store.as_ref(), "a0", 0, &mut cache, None, true).expect("row");
        assert_eq!(second.name, "live");
        let _ = fs::remove_dir_all(&dir);
    }

    fn event(seq: u64, event: tools::WorkflowEventKind) -> tools::WorkflowEventRecord {
        tools::WorkflowEventRecord {
            schema: 1,
            run_id: "r".to_string(),
            seq,
            ts_ms: seq + 1,
            event,
        }
    }

    #[test]
    fn reconcile_doc_overrides_stale_running_workflow_and_phase_status() {
        // Snapshot still says the run and its first phase are running (a dropped
        // final write); the event log recorded both as finished.
        let mut doc = ProgressDoc {
            run_id: "r".to_string(),
            status: "running".to_string(),
            phases: vec![
                ProgressPhase {
                    id: "read".to_string(),
                    status: "running".to_string(),
                    ..ProgressPhase::default()
                },
                ProgressPhase {
                    id: "synth".to_string(),
                    status: "pending".to_string(),
                    ..ProgressPhase::default()
                },
            ],
            ..ProgressDoc::default()
        };
        let records = vec![
            event(
                0,
                tools::WorkflowEventKind::PhaseDone {
                    id: "read".to_string(),
                    completed: 2,
                    failed: 0,
                    still_running: 0,
                    carried: 0,
                    retried: 0,
                    skipped: 0,
                },
            ),
            event(
                1,
                tools::WorkflowEventKind::Finished {
                    status: "completed".to_string(),
                },
            ),
        ];

        reconcile_doc(&mut doc, &records);
        assert_eq!(doc.status, "completed", "log's terminal status wins");
        assert_eq!(
            doc.phases[0].status, "done",
            "stale running phase corrected"
        );
        assert_eq!(
            doc.phases[1].status, "pending",
            "a phase the log never mentioned keeps its snapshot status"
        );
    }

    #[test]
    fn reconcile_doc_never_regresses_a_phase_ahead_of_the_log() {
        // Snapshot already shows the phase done; the log is one event behind
        // (only an enter). Ranking must keep the snapshot's terminal status.
        let mut doc = ProgressDoc {
            run_id: "r".to_string(),
            status: "running".to_string(),
            phases: vec![ProgressPhase {
                id: "read".to_string(),
                status: "done".to_string(),
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        let records = vec![event(
            0,
            tools::WorkflowEventKind::PhaseEnter {
                id: "read".to_string(),
                round: 1,
            },
        )];

        reconcile_doc(&mut doc, &records);
        assert_eq!(doc.phases[0].status, "done");
        assert_eq!(
            doc.status, "running",
            "no Finished event yet → the run stays running"
        );
    }

    #[test]
    fn reconciled_terminal_state_flows_into_the_built_view() {
        // The read model the Ctrl+O viewer renders: reconcile a stale-running
        // snapshot, then build the view the same way the reader does. The view
        // reads as completed (so the viewer stops spinning and shows the final
        // tree) even though the snapshot's last write said "running".
        let mut doc = ProgressDoc {
            run_id: "r".to_string(),
            name: "code-health".to_string(),
            status: "running".to_string(),
            phases: vec![ProgressPhase {
                id: "read".to_string(),
                status: "running".to_string(),
                agent_ids: vec!["a0".to_string()],
                ..ProgressPhase::default()
            }],
            ..ProgressDoc::default()
        };
        let records = vec![
            event(
                0,
                tools::WorkflowEventKind::PhaseDone {
                    id: "read".to_string(),
                    completed: 1,
                    failed: 0,
                    still_running: 0,
                    carried: 0,
                    retried: 0,
                    skipped: 0,
                },
            ),
            event(
                1,
                tools::WorkflowEventKind::Finished {
                    status: "completed".to_string(),
                },
            ),
        ];
        reconcile_doc(&mut doc, &records);

        let view = build_view(doc, |id| {
            Some(WorkflowAgentRow {
                id: id.to_string(),
                status: "completed".to_string(),
                ..WorkflowAgentRow::default()
            })
        });
        assert_eq!(view.status, "completed");
        assert_eq!(view.phases[0].status, "done");
    }
}
