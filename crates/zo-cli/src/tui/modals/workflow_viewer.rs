//! Live workflow progress viewer — the dynamic-workflow analogue of the
//! `/workflows` tree in Claude Code, organized as Plan → Executors → detail.
//!
//! Where the sidebar's `✦ N agents` line is a *flat* count (fine for
//! `SpawnMultiAgent`'s single fan-out), a dynamic [`Workflow`] is multi-phase:
//! fan-out → reduce → synthesize, possibly with hundreds of agents queued
//! behind a concurrency cap. This full-screen modal draws the **topology**: a
//! left Plan rail, the selected step's Executors, and a right inspector with
//! live status, current tool, failures, output, tokens, and elapsed time. A
//! plain fan-out is labeled as an unlinked run scope instead of a fake Plan.
//!
//! ## Data flow (why the host feeds it)
//!
//! The modal is a pure view. The host (`tui_loop`) polls the engine's progress
//! snapshot (`.zo/workflows/_active.progress.json`, written by
//! `tools::workflow_tools::progress`) and *joins* each phase's `agent_ids`
//! against the per-agent manifests (`.zo/agents/<id>.json`) the sidebar
//! already reads — producing a [`WorkflowView`]. The App then joins its existing
//! in-memory Todo snapshot by exact step id; the modal performs no disk IO.
//! While the modal is open the
//! host re-polls on the same ~tick the HUD uses and calls [`refresh`], so the
//! tree updates live without the modal touching disk. Selection/scroll survive
//! a refresh (clamped), so a growing agent list never yanks the cursor.
//!
//! [`Workflow`]: tools
//! [`refresh`]: WorkflowViewerModal::refresh

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::glyphs;
use super::super::hud::{TodoChecklistItem, TodoChecklistStatus};
use std::fmt::Write as _;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::super::spinner::format_elapsed;
use super::super::term::reduce_motion_enabled;
use super::draw_scrollbar;
use super::super::theme::Theme;
use super::super::workflow_progress::{INFLIGHT_AGENT_FRACTION, short_model};

/// The FLEET tree's share of the body, clamped between a floor that holds a
/// readable executor row and a ceiling past which the cells belong to the
/// inspector.
///
/// The viewer used to spend *two* fixed columns — a 40-cell Plan rail and a
/// 48-cell Executor list — on what is one question ("which executor?"), and
/// then truncated the executor names inside the narrower of the two while the
/// inspector sat on a hundred spare columns. One proportional tree answers the
/// question in less space and gives the names room to breathe.
const FLEET_PERCENT: u16 = 26;
pub(crate) const FLEET_MIN_WIDTH: u16 = 32;
const FLEET_MAX_WIDTH: u16 = 64;
/// Width the inspector needs before the tree may sit beside it rather than
/// above it.
const INSPECTOR_MIN_WIDTH: u16 = 44;
/// The `│` column between the tree and the inspector.
pub(crate) const COLUMN_RAIL_WIDTH: u16 = 1;
/// Columns of empty surface between two side-by-side panes. Pi separates panes
/// with space, not with a border, and without it a clipped executor row abutted
/// the detail column's first label (`22 toolsflow`).
pub(crate) const PANE_GUTTER: u16 = 1;
/// Body width at which the tree moves beside the inspector.
pub(crate) const SPLIT_LAYOUT_MIN_WIDTH: u16 =
    FLEET_MIN_WIDTH + COLUMN_RAIL_WIDTH + INSPECTOR_MIN_WIDTH + 2 * PANE_GUTTER;
/// Rows the stacked (narrow) arrangement gives the tree before the inspector
/// takes the rest.
const STACKED_FLEET_MAX_ROWS: u16 = 10;

/// Spinner frames for a running agent/phase. Advanced once per [`refresh`] so
/// the "is anything happening?" signal stays alive between polls.
const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

/// Cells the header's progress gauge fills.
const GAUGE_CELLS: usize = 12;

/// One agent row under a phase, joined from the per-agent manifest.
#[derive(Debug, Clone, Default)]
pub struct WorkflowAgentRow {
    /// Stable id from the manifest/progress file.
    pub id: String,
    /// Display name = the phase-coordinate slug the engine stamped
    /// (`read-engine`), or the manifest name.
    pub name: String,
    /// Delegated task description from the manifest.
    pub description: String,
    /// Agent harness/profile type, if available.
    pub subagent_type: Option<String>,
    /// `"running"` | `"completed"` | `"failed"` | `"pending"` | `"stopped"`.
    pub status: String,
    /// Tool the agent is currently running (manifest `currentTool`), shown live.
    pub current_tool: Option<String>,
    /// Rolling feed of the agent's most recent tool calls with argument
    /// briefs (manifest `recentTools`, oldest → newest). Rendered in the
    /// details pane as the live activity transcript.
    pub recent_tools: Vec<String>,
    /// Actual resolved model string from the manifest, or empty when unknown.
    pub model: String,
    /// Number of tool calls so far (`toolCalls` in the agent manifest).
    /// `None` means the manifest came from an older writer; do not infer from
    /// lifecycle lane events, which are not tool calls.
    pub tool_calls: Option<usize>,
    /// Output tokens accumulated (sum of the manifest `tokenHistory`).
    pub tokens: u64,
    /// Seconds since the manifest was last written.
    pub elapsed_secs: u64,
    /// Markdown output file written by the agent runtime.
    pub output_file: Option<String>,
    /// Terminal error captured in the manifest, if any.
    pub error: Option<String>,
    /// Current blocker detail, if any.
    pub blocker: Option<String>,
    /// Last compact lane event detail, if any.
    pub last_event: Option<String>,
    /// Rolling tail of the agent's latest streamed assistant text (manifest
    /// `outputTail`) — *what the agent is actually saying right now*. Present
    /// while running; `None`/empty for an agent that never streamed. Bounded to a
    /// few lines so the detail pane shows live prose without unbounded growth.
    pub output_tail: Option<String>,
    /// Transient wait/stream phase the agent is in (manifest `currentPhase`, e.g.
    /// `thinking`), shown in the agent line when no concrete tool is running.
    pub current_phase: Option<String>,
    /// Seconds since the agent's last manifest write (computed at row-build time
    /// from `lastActivityAt`). Drives an "active Ns ago" heartbeat so a stuck
    /// agent (no recent write) is visible. `None` for an older/heartbeat-less
    /// manifest.
    pub idle_secs: Option<u64>,
    /// Why the Smart router picked this agent's model (manifest `routeReason`),
    /// e.g. `auto:coding tier=strong` or a `learned-shadow-differs:<model>` /
    /// `quota-degraded` / exploration-slot suffix. `None` for explicit models,
    /// routing off, or legacy manifests. Rendered in the detail pane so
    /// auto-routing is explainable without JSONL archaeology (`/smart doctor`
    /// covers the aggregate view; this is the per-agent one).
    pub route_reason: Option<String>,
}

/// One phase row in the left rail.
#[derive(Debug, Clone, Default)]
pub struct WorkflowPhaseRow {
    /// Exact Todo step id for a real Workflow phase. `None` for the synthetic
    /// plain-`SpawnMultiAgent` group, which must never impersonate a Plan step.
    pub step_id: Option<String>,
    /// Exact Todo snapshot joined by `step_id` at the App boundary. Display-only
    /// and refreshed without any render-path IO.
    pub plan_step: Option<TodoChecklistItem>,
    pub id: String,
    /// `"fanout"` | `"over"` | `"single"`.
    pub kind: String,
    /// `"pending"` | `"running"` | `"done"` | `"resumed"`.
    pub status: String,
    /// Current round (1-based) for a `repeat` phase.
    pub round: u32,
    pub completed: usize,
    pub failed: usize,
    pub still_running: usize,
    /// Total agents spawned for this phase (`agent_ids.len()`).
    pub total: usize,
    /// The agents themselves, for the right pane when this phase is selected.
    pub agents: Vec<WorkflowAgentRow>,
}

impl WorkflowPhaseRow {
    fn plan_label(&self, prefer_active_form: bool) -> &str {
        if let Some(step) = &self.plan_step {
            if prefer_active_form
                && self.status == "running"
                && step.status == TodoChecklistStatus::InProgress
            {
                // The exact sentence rule the activity line and the chapter
                // notice use (`hud::step_sentence`), so the three NOW surfaces
                // can never word the same step differently.
                return crate::tui::hud::step_sentence(step);
            }
            return &step.content;
        }
        if self.step_id.is_some() {
            &self.id
        } else {
            "Run-level fan-out"
        }
    }

    fn scope(&self) -> ViewerScope {
        if self.plan_step.is_some() {
            ViewerScope::Plan
        } else if self.step_id.is_some() {
            ViewerScope::Workflow
        } else {
            ViewerScope::Run
        }
    }

    fn is_plan_scoped(&self) -> bool {
        self.scope() == ViewerScope::Plan
    }

    /// A phase past its barrier: its recorded `completed`/`failed`/`still_running`
    /// tallies are authoritative (the engine only fills them at `PhaseDone`).
    fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "done" | "resumed")
    }

    /// Agents running *right now*. While a phase executes the engine's recorded
    /// `still_running` is structurally 0 (it is only written at the barrier), so
    /// counting it makes the header read "0 running" for the whole run. Instead
    /// count the live per-agent rows — the same manifest-backed `status` the agent
    /// pane already trusts — and fall back to the recorded count once the phase is
    /// terminal (its rows may have been dropped past the manifest read budget).
    ///
    /// The live branch is itself subject to the shared 128-manifest read budget,
    /// so for a phase with more loaded-or-pending agents than the budget this is a
    /// *lower bound* (it can never over-count); it self-corrects to the engine's
    /// authoritative count once the phase is terminal.
    fn running_now(&self) -> usize {
        if self.is_terminal() {
            self.still_running
        } else {
            self.agents.iter().filter(|a| a.status == "running").count()
        }
    }

    /// Agents completed so far — live row count while running, recorded count once
    /// the phase is past its barrier (same single-source-of-truth rule as
    /// [`Self::running_now`]).
    fn completed_now(&self) -> usize {
        if self.is_terminal() {
            self.completed
        } else {
            self.agents
                .iter()
                .filter(|a| a.status == "completed")
                .count()
        }
    }

    fn failed_now(&self) -> usize {
        if self.is_terminal() {
            self.failed
        } else {
            self.agents.iter().filter(|a| a.status == "failed").count()
        }
    }

    fn finished_now(&self) -> usize {
        self.completed_now().saturating_add(self.failed_now())
    }

    fn progress_percent(&self) -> usize {
        if self.is_terminal() {
            return 100;
        }
        if self.total == 0 {
            return 0;
        }
        // Within-agent partial credit, mirroring `phase_progress_percent` in
        // workflow_progress.rs so the modal and the sidebar/HUD agree: a running
        // phase is not pinned at 0% before its first agent finishes. Capped below
        // 100 until the phase is terminal.
        let finished = self.finished_now();
        let remaining = self.total.saturating_sub(finished);
        #[allow(
            clippy::cast_precision_loss,
            reason = "agent counts are tiny, far below f64's 53-bit exact-integer range"
        )]
        let (finished_f, inflight_f, remaining_f) =
            (finished as f64, self.running_now() as f64, remaining as f64);
        let inflight_credit = (inflight_f * INFLIGHT_AGENT_FRACTION).min(remaining_f * 0.9);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "percent is a small non-negative whole number; counts are tiny"
        )]
        let pct = ((finished_f + inflight_credit) * 100.0 / self.total as f64).floor() as usize;
        pct.min(99)
    }
}

/// The full snapshot the host hands the modal each refresh.
#[derive(Debug, Clone, Default)]
pub struct WorkflowView {
    /// Run id, joining this view to its append-only event log for the inspector
    /// (`e`). Empty for a synthetic agents-only view (no workflow run).
    pub run_id: String,
    pub name: String,
    pub description: String,
    /// `"running"` | `"completed"` | `"cancelled"` | `"budget_exhausted"`.
    pub status: String,
    /// `"phases"` | `"pipeline"`.
    pub mode: String,
    pub phases: Vec<WorkflowPhaseRow>,
    /// True while the final synthesize agent runs.
    pub synthesizing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewerScope {
    Plan,
    Workflow,
    Run,
}

impl WorkflowView {
    fn attach_plan_items(&mut self, items: &[TodoChecklistItem]) {
        for phase in &mut self.phases {
            phase.plan_step = None;
        }
        for phase_idx in 0..self.phases.len() {
            let Some(step_id) = self.phases[phase_idx].step_id.clone() else {
                continue;
            };
            if step_id.is_empty()
                || step_id.trim() != step_id
                || self.phases[phase_idx].id != step_id
                || self
                    .phases
                    .iter()
                    .filter(|phase| phase.step_id.as_deref() == Some(step_id.as_str()))
                    .count()
                    != 1
            {
                continue;
            }
            let mut matching = items
                .iter()
                .filter(|item| item.step_id.as_deref() == Some(step_id.as_str()));
            let Some(item) = matching.next() else {
                continue;
            };
            if matching.next().is_none() {
                self.phases[phase_idx].plan_step = Some(item.clone());
            }
        }
    }

    fn scope(&self) -> ViewerScope {
        if !self.phases.is_empty()
            && self
                .phases
                .iter()
                .all(WorkflowPhaseRow::is_plan_scoped)
        {
            ViewerScope::Plan
        } else if self.phases.iter().any(|phase| phase.step_id.is_some()) {
            ViewerScope::Workflow
        } else {
            ViewerScope::Run
        }
    }

    fn plan_link_count(&self) -> usize {
        self.phases
            .iter()
            .filter(|phase| phase.is_plan_scoped())
            .count()
    }

    /// Total agents across all phases (the "M" in `N/M agents`).
    fn total_agents(&self) -> usize {
        self.phases.iter().map(|p| p.total).sum()
    }

    /// Agents currently running across all phases (the "N"). Live: counts the
    /// per-agent rows for a running phase, so the header tracks the spinning rows
    /// instead of the engine's post-barrier `still_running` (which stays 0 for the
    /// whole run). See [`WorkflowPhaseRow::running_now`].
    fn running_agents(&self) -> usize {
        self.phases.iter().map(WorkflowPhaseRow::running_now).sum()
    }

    fn failed_agents(&self) -> usize {
        self.phases.iter().map(WorkflowPhaseRow::failed_now).sum()
    }

    fn active_phase_index(&self) -> Option<usize> {
        self.phases
            .iter()
            .position(|phase| phase.status == "running")
            .or_else(|| {
                self.phases
                    .iter()
                    .position(|phase| phase.status == "pending")
            })
            .or_else(|| self.phases.len().checked_sub(1))
    }

    fn progress_percent(&self) -> usize {
        if self.phases.is_empty() {
            return 0;
        }
        let sum = self
            .phases
            .iter()
            .map(WorkflowPhaseRow::progress_percent)
            .sum::<usize>();
        (sum / self.phases.len()).min(100)
    }

}

/// Outcome of a key handled by [`WorkflowViewerModal`]. The viewer is read-only
/// (a live monitor), so the only exit is `Close`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowViewerAction {
    /// `Esc` / `q` / `Ctrl+C` — close the viewer.
    Close,
}

/// Full-screen live workflow progress viewer.
#[derive(Debug, Clone)]
pub struct WorkflowViewerModal {
    view: WorkflowView,
    /// Phase the cursor sits under (the group the selected executor belongs to).
    selected_phase: usize,
    /// Scroll offset into the flattened FLEET tree.
    fleet_scroll: u16,
    /// Selected agent inside [`Self::selected_phase`].
    selected_agent: usize,
    /// Spinner phase, advanced on each refresh.
    tick: usize,
    /// Phase-3 event-log inspector: when on, the modal shows the run's
    /// append-only event timeline instead of the phase/agent panes.
    events_mode: bool,
    /// Cached, pre-formatted timeline lines. Read once when the inspector is
    /// opened (and on refresh while open), never in the render path — the draw
    /// loop must stay non-blocking.
    events: Vec<String>,
    /// Scroll offset into the event timeline.
    events_scroll: u16,
    /// Tail of the selected agent's markdown output file, refreshed by the
    /// host tick (mtime-gated) — never read in the draw path. `(path,
    /// modified, lines)`. The runtime writes the file when the agent
    /// finishes, so this fills the details pane with the agent's actual
    /// result the moment it lands.
    output_tail: Option<(String, std::time::SystemTime, Vec<String>)>,
    /// Which detail sections the reader has folded away by clicking their
    /// heading. Modal state, not per-agent: a fold survives walking the fleet.
    folds: DetailFolds,
}

/// Pure geometry of one workflow-viewer frame.
struct WorkflowRegions {
    header: Rect,
    body: WorkflowBody,
    footer: Rect,
}

/// Which arrangement the body took, and where its panes landed. The side-by-side
/// and stacked arrangements differ only in the rects they produce, so they share
/// one variant — draw and hit-test treat them identically.
enum WorkflowBody {
    /// Event-log inspector (`Ctrl+E`): one scrolling timeline.
    Events(Rect),
    /// Too short for panes: one compact summary.
    Compact(Rect),
    /// The FLEET tree and the executor inspector. `rail` is the `│` between them
    /// and is `None` when the body stacked them instead.
    Panes {
        fleet: Rect,
        rail: Option<Rect>,
        inspector: Rect,
    },
}

/// One row of the FLEET tree.
///
/// The tree is the viewer's single selector: a phase is a *group header* over
/// the executors it spawned, not a column of its own. Flattening the two into
/// one list is what lets `↑`/`↓` walk the whole run — the old pane confined the
/// cursor to the selected phase, so an executor in phase 2 was unreachable
/// without first finding the `←`/`→` axis nobody knew was there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FleetRow {
    /// Group header for `view.phases[phase]`.
    Phase { phase: usize },
    /// `view.phases[phase].agents[agent]`, drawn indented under its group.
    Executor { phase: usize, agent: usize },
}

impl WorkflowViewerModal {
    #[must_use]
    pub fn new(view: WorkflowView) -> Self {
        let selected_phase = view.active_phase_index().unwrap_or(0);
        let selected_agent = view
            .phases
            .get(selected_phase)
            .and_then(|phase| {
                phase
                    .agents
                    .iter()
                    .position(|agent| agent.status == "running")
            })
            .unwrap_or(0);
        Self {
            view,
            selected_phase,
            fleet_scroll: 0,
            selected_agent,
            tick: 0,
            events_mode: false,
            events: Vec::new(),
            events_scroll: 0,
            output_tail: None,
            folds: DetailFolds::default(),
        }
    }

    /// Join the App's already-loaded Todo snapshot onto real Workflow phases.
    /// Synthetic fan-outs carry no `step_id`, so even a Todo named `agents`
    /// remains explicitly unscoped.
    pub fn attach_plan_items(&mut self, items: &[TodoChecklistItem]) {
        self.view.attach_plan_items(items);
    }

    /// Refresh the selected agent's output-file tail (host tick, never the
    /// draw path). mtime-gated: a stat per call, a read only when the file
    /// actually changed or the selection moved to a different agent.
    pub fn refresh_output_tail(&mut self) {
        // Deep enough to fill a tall terminal's output column. Forty lines was
        // sized for a hugged card; against a full-height column on a 60-row
        // screen it ran out of result halfway down and left the rest blank —
        // the read is mtime-gated on the host tick, so the extra lines cost a
        // stat per poll and a 64KB scan only when the file actually grew.
        const TAIL_LINES: usize = 240;
        const TAIL_BYTES: u64 = 64 * 1024;
        let Some(path) = self
            .selected_agent_row()
            .and_then(|agent| agent.output_file.clone())
        else {
            self.output_tail = None;
            return;
        };
        let Ok(meta) = std::fs::metadata(&path) else {
            // Not written yet (agent still running) — keep nothing cached so
            // the pane falls back to the live activity feed.
            self.output_tail = None;
            return;
        };
        let Ok(modified) = meta.modified() else {
            return;
        };
        if let Some((cached_path, cached_mtime, _)) = &self.output_tail {
            if *cached_path == path && *cached_mtime == modified {
                return;
            }
        }
        let lines = read_tail_lines(&path, meta.len(), TAIL_BYTES, TAIL_LINES);
        self.output_tail = Some((path, modified, lines));
    }

    /// Reload the cached event timeline from the run's append-only log. Called
    /// when the inspector opens and on refresh while it's open, so the render
    /// path only ever reads the in-memory `events`.
    fn reload_events(&mut self) {
        self.events = if self.view.run_id.is_empty() {
            Vec::new()
        } else {
            tools::event_timeline_lines(&tools::read_event_log(&self.view.run_id))
        };
        self.events_scroll = self.clamp_events_scroll(self.events_scroll);
    }

    fn clamp_events_scroll(&self, want: u16) -> u16 {
        let max = u16::try_from(self.events.len().saturating_sub(1)).unwrap_or(u16::MAX);
        want.min(max)
    }

    /// `true` when there is no active workflow to show.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.view.phases.is_empty()
    }

    /// Replace the snapshot with a fresh poll, preserving (and clamping) the
    /// cursor/scroll so a live update never jumps the view. Data refresh is
    /// throttled by the host; the spinner advances separately via
    /// [`Self::advance_spinner`] so animation stays smooth between polls.
    pub fn refresh(&mut self, mut view: WorkflowView, plan_items: &[TodoChecklistItem]) {
        // Phase-5: the view arrives already reconciled against the append-only
        // event log by the reader ([`workflow_progress`]'s `load_doc`), so a
        // snapshot that dropped its final write reads as terminal here *and* in
        // the sidebar — every consumer agrees. The viewer just renders the read
        // model it is handed.
        view.attach_plan_items(plan_items);
        // TodoWrite may drop completed rows, and the HUD clears an all-completed
        // list at turn settlement. Preserve a previously exact label per phase
        // for the same run, but only when the current Todo snapshot has zero
        // matches. Duplicate Todo or phase ids remain deliberately ambiguous.
        if !view.run_id.is_empty() && view.run_id == self.view.run_id {
            for phase_idx in 0..view.phases.len() {
                if view.phases[phase_idx].plan_step.is_some() {
                    continue;
                }
                let Some(step_id) = view.phases[phase_idx].step_id.clone() else {
                    continue;
                };
                let current_phase_is_unique = !step_id.is_empty()
                    && step_id.trim() == step_id
                    && view.phases[phase_idx].id == step_id
                    && view
                        .phases
                        .iter()
                        .filter(|phase| phase.step_id.as_deref() == Some(step_id.as_str()))
                        .count()
                        == 1;
                let current_todo_matches = plan_items
                    .iter()
                    .filter(|item| item.step_id.as_deref() == Some(step_id.as_str()))
                    .count();
                if !current_phase_is_unique || current_todo_matches != 0 {
                    continue;
                }

                let mut prior = self
                    .view
                    .phases
                    .iter()
                    .filter(|old| old.step_id.as_deref() == Some(step_id.as_str()));
                let Some(old) = prior.next() else {
                    continue;
                };
                if prior.next().is_some() || old.id != step_id || old.plan_step.is_none() {
                    continue;
                }
                let mut preserved = old.plan_step.clone();
                if let Some(step) = &mut preserved {
                    step.status = match view.phases[phase_idx].status.as_str() {
                        "done" | "resumed" => TodoChecklistStatus::Completed,
                        "running" => TodoChecklistStatus::InProgress,
                        _ => TodoChecklistStatus::Pending,
                    };
                }
                view.phases[phase_idx].plan_step = preserved;
            }
        }
        self.view = view;
        if self.view.run_id.is_empty() {
            self.events_mode = false;
        }
        let max_phase = self.view.phases.len().saturating_sub(1);
        if self.selected_phase > max_phase {
            self.selected_phase = max_phase;
            self.fleet_scroll = 0;
        }
        self.clamp_selected_agent();
        // Keep the open inspector live as the run appends events.
        if self.events_mode {
            self.reload_events();
        }
    }

    /// Advance the spinner one frame. Called every redraw (decoupled from the
    /// slower data refresh) so a running agent reads as alive.
    pub fn advance_spinner(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    #[must_use]
    pub const fn selected_phase(&self) -> usize {
        self.selected_phase
    }

    #[must_use]
    pub const fn selected_agent(&self) -> usize {
        self.selected_agent
    }

    /// Pre-select the agent with this id, scanning every phase. On a hit, moves
    /// the phase + agent cursor to it (resetting the agent scroll) and returns
    /// `true`; on a miss, leaves the default `(0, 0)` selection and returns
    /// `false`, so the caller lands on the aggregate view instead of a wrong
    /// agent. Used to focus the viewer on a clicked pinned-panel row.
    pub fn select_agent_by_id(&mut self, id: &str) -> bool {
        for (phase_idx, phase) in self.view.phases.iter().enumerate() {
            if let Some(agent_idx) = phase.agents.iter().position(|agent| agent.id == id) {
                self.selected_phase = phase_idx;
                self.selected_agent = agent_idx;
                self.fleet_scroll = 0;
                return true;
            }
        }
        false
    }

    fn select_prev_phase(&mut self) {
        if self.selected_phase > 0 {
            self.selected_phase -= 1;
            self.fleet_scroll = 0;
            self.selected_agent = 0;
        }
    }

    fn select_next_phase(&mut self) {
        if self.selected_phase + 1 < self.view.phases.len() {
            self.selected_phase += 1;
            self.fleet_scroll = 0;
            self.selected_agent = 0;
        }
    }

    /// Every executor in the run, in tree order, as `(phase, agent)`.
    fn executor_index(&self) -> Vec<(usize, usize)> {
        self.view
            .phases
            .iter()
            .enumerate()
            .flat_map(|(phase, row)| (0..row.agents.len()).map(move |agent| (phase, agent)))
            .collect()
    }

    /// Where the cursor sits in [`Self::executor_index`].
    fn executor_cursor(&self) -> usize {
        self.executor_index()
            .iter()
            .position(|&(phase, agent)| {
                phase == self.selected_phase && agent == self.selected_agent
            })
            .unwrap_or(0)
    }

    /// Move the cursor to the `idx`-th executor of the run, carrying the phase
    /// selection with it so the inspector and the group highlight agree.
    fn select_executor_at(&mut self, idx: usize) {
        let index = self.executor_index();
        if index.is_empty() {
            return;
        }
        let (phase, agent) = index[idx.min(index.len() - 1)];
        self.selected_phase = phase;
        self.selected_agent = agent;
    }

    /// The flattened tree: one header per phase, its executors under it.
    fn fleet_rows(&self) -> Vec<FleetRow> {
        let mut rows = Vec::new();
        for (phase, row) in self.view.phases.iter().enumerate() {
            rows.push(FleetRow::Phase { phase });
            rows.extend(
                (0..row.agents.len()).map(|agent| FleetRow::Executor { phase, agent }),
            );
        }
        rows
    }

    /// Row index of the cursor in [`Self::fleet_rows`] — the executor it is on,
    /// or the header of a phase that spawned nothing yet.
    fn fleet_cursor_row(&self, rows: &[FleetRow]) -> usize {
        rows.iter()
            .position(|row| match *row {
                FleetRow::Executor { phase, agent } => {
                    phase == self.selected_phase && agent == self.selected_agent
                }
                FleetRow::Phase { .. } => false,
            })
            .or_else(|| {
                rows.iter().position(|row| {
                    matches!(*row, FleetRow::Phase { phase } if phase == self.selected_phase)
                })
            })
            .unwrap_or(0)
    }

    fn selected_phase_row(&self) -> Option<&WorkflowPhaseRow> {
        self.view.phases.get(self.selected_phase)
    }

    fn selected_agent_row(&self) -> Option<&WorkflowAgentRow> {
        self.selected_phase_row()
            .and_then(|phase| phase.agents.get(self.selected_agent))
    }

    fn clamp_selected_agent(&mut self) {
        let max_agent = self
            .selected_phase_row()
            .map_or(0, |phase| phase.agents.len().saturating_sub(1));
        if self.selected_agent > max_agent {
            self.selected_agent = max_agent;
        }
    }

    /// Walk the cursor `rows` executors back, crossing phase groups.
    fn select_prev_agent(&mut self, rows: usize) {
        self.select_executor_at(self.executor_cursor().saturating_sub(rows));
    }

    /// Walk the cursor `rows` executors forward, crossing phase groups.
    fn select_next_agent(&mut self, rows: usize) {
        self.select_executor_at(self.executor_cursor().saturating_add(rows));
    }

    /// Scroll the fleet tree up by `rows` (mouse wheel). The offset is clamped to
    /// the content height at draw time, so an unbounded add is safe.
    pub fn scroll_agents_up(&mut self, rows: u16) {
        if self.events_mode {
            self.events_scroll = self.events_scroll.saturating_sub(rows);
            return;
        }
        self.fleet_scroll = self.fleet_scroll.saturating_sub(rows);
        self.select_prev_agent(usize::from(rows));
    }

    /// Scroll the fleet tree down by `rows` (mouse wheel).
    pub fn scroll_agents_down(&mut self, rows: u16) {
        if self.events_mode {
            self.events_scroll =
                self.clamp_events_scroll(self.events_scroll.saturating_add(rows));
            return;
        }
        self.fleet_scroll = self.fleet_scroll.saturating_add(rows);
        self.select_next_agent(usize::from(rows));
    }

    /// Handle one key. Returns `Some(Close)` to dismiss; `None` while navigating.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<WorkflowViewerAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Some(WorkflowViewerAction::Close);
        }
        // Ctrl+E toggles the event-log inspector without stealing printable
        // `e` from the live composer behind this monitor.
        if matches!(key.code, KeyCode::Char('e'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
        {
            if self.view.run_id.is_empty() {
                return None;
            }
            self.events_mode = !self.events_mode;
            if self.events_mode {
                self.reload_events();
                self.events_scroll = 0;
            }
            return None;
        }
        if self.events_mode {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => return Some(WorkflowViewerAction::Close),
                KeyCode::Up | KeyCode::Char('k') => {
                    self.events_scroll = self.events_scroll.saturating_sub(1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.events_scroll =
                        self.clamp_events_scroll(self.events_scroll.saturating_add(1));
                }
                KeyCode::PageUp => self.events_scroll = self.events_scroll.saturating_sub(10),
                KeyCode::PageDown => {
                    self.events_scroll =
                        self.clamp_events_scroll(self.events_scroll.saturating_add(10));
                }
                KeyCode::Home | KeyCode::Char('g') => self.events_scroll = 0,
                KeyCode::End | KeyCode::Char('G') => {
                    self.events_scroll = self.clamp_events_scroll(u16::MAX);
                }
                _ => {}
            }
            return None;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => return Some(WorkflowViewerAction::Close),
            // Executor navigation (primary axis). Most live views have one phase and
            // many executors, so ↑/↓ must move through useful rows instead of feeling
            // inert on the phase rail.
            KeyCode::Up | KeyCode::Char('k') => self.select_prev_agent(1),
            KeyCode::Down | KeyCode::Char('j') => self.select_next_agent(1),
            // Plan-step / phase navigation (secondary axis).
            KeyCode::Left | KeyCode::Char('h') => self.select_prev_phase(),
            KeyCode::Right | KeyCode::Char('l') => self.select_next_phase(),
            // Home/End walk the whole run, like ↑/↓ — confining them to the
            // selected phase made `G` a no-op on the last group and a jump to
            // the middle of the tree on every other one.
            KeyCode::Home | KeyCode::Char('g') => {
                self.select_executor_at(0);
                self.fleet_scroll = 0;
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.select_executor_at(usize::MAX);
            }
            KeyCode::PageUp => self.select_prev_agent(10),
            KeyCode::PageDown => self.select_next_agent(10),
            _ => {}
        }
        None
    }

    /// Route a left-click at absolute `(column, row)` given the same modal
    /// `area` the draw used. Returns `true` when the click changed something,
    /// so the host can redraw.
    ///
    /// * a FLEET **phase header** selects that group
    /// * a FLEET **executor row** selects that agent — the move `↑`/`↓` makes
    /// * an inspector **section heading** folds or unfolds that section
    ///
    /// Hit-testing recomputes the same pure layout [`Self::draw`] renders from
    /// (down to the columns of the split detail band and each section's wrapped
    /// row span), so a hit can never land one pane over.
    pub fn handle_click(&mut self, column: u16, row: u16, area: Rect, theme: &Theme) -> bool {
        let Some(regions) = self.regions(area, theme) else {
            return false;
        };
        let WorkflowBody::Panes {
            fleet, inspector, ..
        } = regions.body
        else {
            return false;
        };

        let fleet_inner = pane_inner(fleet, theme);
        if let Some(hit) = hit_row(fleet_inner, column, row) {
            let rows = self.fleet_rows();
            let offset = self.fleet_offset(fleet_inner.height, &rows);
            let Some(&target) = rows.get(usize::from(offset.saturating_add(hit))) else {
                return false;
            };
            return self.select_fleet_row(target);
        }

        self.toggle_detail_section_at(inspector, column, row, theme)
    }

    /// Move the selection onto a clicked tree row. Returns `false` when the row
    /// was already selected, so a re-click never costs a redraw.
    fn select_fleet_row(&mut self, target: FleetRow) -> bool {
        match target {
            FleetRow::Phase { phase } => {
                if phase == self.selected_phase {
                    return false;
                }
                self.selected_phase = phase;
                self.selected_agent = 0;
                true
            }
            FleetRow::Executor { phase, agent } => {
                if phase == self.selected_phase && agent == self.selected_agent {
                    return false;
                }
                self.selected_phase = phase;
                self.selected_agent = agent;
                true
            }
        }
    }

    /// Fold/unfold the detail section whose heading the click landed on.
    fn toggle_detail_section_at(
        &mut self,
        detail: Rect,
        column: u16,
        row: u16,
        theme: &Theme,
    ) -> bool {
        let Some(agent) = self.selected_agent_row() else {
            return false;
        };
        let inner = pane_inner(detail, theme);
        let Some(hit) = hit_row(inner, column, row) else {
            return false;
        };
        let section = match self.inspector_band(agent, inner) {
            // The output column pins its heading to row 0 and scrolls the tail
            // under it, so that row — and only that row — is the fold target.
            Some(band) if column >= band.output.x => self
                .detail_output_column(agent, band.output.width, theme)
                .section_at_row(usize::from(hit), band.output.width)
                .filter(|_| hit == 0),
            Some(band) => self
                .detail_meta_column(agent, band.meta.width, theme)
                .section_at_row(usize::from(hit), band.meta.width),
            None => {
                let mut stacked = self.detail_meta_column(agent, inner.width, theme);
                stacked.push(Line::from(""));
                stacked.append(self.detail_output_column(agent, inner.width, theme));
                stacked.section_at_row(usize::from(hit), inner.width)
            }
        };
        let Some(section) = section else {
            return false;
        };
        self.folds.toggle(section);
        true
    }

    /// Top-row offset of the FLEET tree for a viewport of `height` rows —
    /// shared by draw and hit-testing so a click can never land on the row
    /// above the one the reader aimed at.
    fn fleet_offset(&self, height: u16, rows: &[FleetRow]) -> u16 {
        let max_scroll = u16::try_from(rows.len())
            .unwrap_or(u16::MAX)
            .saturating_sub(height);
        let cursor = u16::try_from(self.fleet_cursor_row(rows)).unwrap_or(u16::MAX);
        visible_offset(self.fleet_scroll.min(max_scroll), cursor, height).min(max_scroll)
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        // The title names the *run*, nothing else. It used to spell out the
        // topology (`Run → Executors · spawned agents`), which the FLEET tree
        // then drew again two rows below and the inspector a third time — the
        // duplication that made a dense screen read as noise.
        let run_name = if self.view.name == "agents" {
            "spawned agents"
        } else {
            self.view.name.as_str()
        };
        let title = format!("agents · {}", short(run_name, 52));
        // Blank the transcript this modal floats over *and* lay down the
        // elevation-2 glass before any of the body draws. The host frosts the
        // backdrop for us in the real app, but relying on that left the viewer
        // with no surface of its own: on a terminal whose background already
        // matches the theme's, the modal simply changed subject mid-screen
        // instead of reading as a pane above the conversation. It is also what
        // makes the rows a short body never covers read as *this* surface
        // rather than as a hole punched through it.
        super::dialog_surface(frame, area, theme);
        frame.render_widget(
            CardFrame::new(SurfaceKind::Modal, theme)
                .title(super::modal_title(theme, title))
                .padding(Padding::symmetric(1, 0))
                .block(),
            area,
        );
        // Geometry comes from the pure pass, never from the block we just
        // rendered: `handle_click` calls the same function, so a hit can never
        // land on a pane the frame put somewhere else.
        let Some(regions) = self.regions(area, theme) else {
            return;
        };

        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                self.header_lines(theme, regions.header.width),
                regions.header.width,
            )),
            regions.header,
        );

        match regions.body {
            WorkflowBody::Events(area) => self.draw_event_log(frame, area, theme),
            WorkflowBody::Compact(area) => self.draw_compact_body(frame, area, theme),
            WorkflowBody::Panes {
                fleet,
                rail,
                inspector,
            } => {
                self.draw_fleet(frame, fleet, theme);
                if let Some(rail) = rail {
                    draw_column_rail(frame, rail, theme);
                }
                self.draw_agent_detail(frame, inspector, theme);
            }
        }

        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                vec![footer_line(
                    theme,
                    regions.footer.width,
                    self.events_mode,
                    !self.view.run_id.is_empty(),
                )],
                regions.footer.width,
            )),
            regions.footer,
        );
    }

    /// The dialog's inner content rect — the same one [`CardFrame::render`]
    /// returns for the block `draw` paints. A title never changes it (the top
    /// rule already costs the row it rides on), so the pure pass may omit it.
    fn modal_inner(area: Rect, theme: &Theme) -> Rect {
        CardFrame::new(SurfaceKind::Modal, theme)
            .padding(Padding::symmetric(1, 0))
            .block()
            .inner(area)
    }

    /// Pure geometry for one frame. `draw` paints from it and `handle_click`
    /// hit-tests against it, so the two can never drift apart.
    fn regions(&self, area: Rect, theme: &Theme) -> Option<WorkflowRegions> {
        let inner = Self::modal_inner(area, theme);
        if inner.height == 0 || inner.width == 0 {
            return None;
        }
        let [header, body, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);
        Some(WorkflowRegions {
            header,
            body: self.body_regions(body),
            footer,
        })
    }

    /// Rows the FLEET tree holds: one header per phase plus every executor.
    fn fleet_row_count(&self) -> usize {
        self.view.phases.len()
            + self
                .view
                .phases
                .iter()
                .map(|phase| phase.agents.len())
                .sum::<usize>()
    }

    /// Which arrangement the body takes at this size, and where its panes land.
    fn body_regions(&self, body: Rect) -> WorkflowBody {
        if self.events_mode {
            return WorkflowBody::Events(body);
        }
        if body.height < 10 {
            return WorkflowBody::Compact(body);
        }
        if body.width >= SPLIT_LAYOUT_MIN_WIDTH {
            let [fleet, rail, inspector] = Layout::horizontal([
                Constraint::Length(fleet_width(body.width)),
                Constraint::Length(COLUMN_RAIL_WIDTH),
                Constraint::Min(INSPECTOR_MIN_WIDTH),
            ])
            .spacing(PANE_GUTTER)
            .areas(body);
            // Both columns take the body's full height. The inspector used to be
            // hugged to its content, which on a 60-row terminal left thirty rows
            // of empty surface below a *truncated* output tail — the viewer was
            // starving the one section that had more to show.
            return WorkflowBody::Panes {
                fleet,
                rail: Some(rail),
                inspector,
            };
        }
        // `+ 2` for the card's own rules: sized to `+ 1` the tree lost a row to
        // scroll, and the row it lost was the phase header the executors hang
        // off — the stacked view then showed a flat list with no group at all.
        let fleet_height = u16::try_from(self.fleet_row_count().saturating_add(2))
            .unwrap_or(u16::MAX)
            .clamp(3, STACKED_FLEET_MAX_ROWS)
            .min(body.height.saturating_sub(6).max(3));
        let [fleet, inspector] =
            Layout::vertical([Constraint::Length(fleet_height), Constraint::Min(0)]).areas(body);
        WorkflowBody::Panes {
            fleet,
            rail: None,
            inspector,
        }
    }

    /// Phase-3 event-log inspector: the run's append-only timeline, rendered from
    /// the cached `events` (never re-reading the log in the draw path).
    fn draw_event_log(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let [title_area, list_area] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(
                vec![Line::styled(
                    format!(
                        "event log · {} events · ^E: back · \u{2191}\u{2193} scroll",
                        self.events.len()
                    ),
                    detail_secondary_style(theme),
                )],
                title_area.width,
            )),
            title_area,
        );
        let lines: Vec<Line<'static>> = if self.events.is_empty() {
            vec![Line::styled(
                "no events recorded for this run",
                detail_secondary_style(theme),
            )]
        } else {
            self.events
                .iter()
                .map(|line| Line::raw(line.clone()))
                .collect()
        };
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(lines, list_area.width))
                .scroll((self.events_scroll, 0))
                .style(detail_value_style(theme)),
            list_area,
        );
    }

    /// Low-height fallback: preserve the relationship and the selected
    /// executor instead of drawing three empty bordered panels.
    fn draw_compact_body(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let text_width = usize::from(area.width.saturating_sub(14)).max(4);
        let key = detail_label_style(theme);
        let value = detail_value_style(theme);
        let mut lines = Vec::new();
        if let Some(phase) = self.selected_phase_row() {
            let scope = match phase.scope() {
                ViewerScope::Plan => "Plan",
                ViewerScope::Workflow => "Workflow",
                ViewerScope::Run => "Run",
            };
            lines.push(detail_line(
                scope,
                &short(phase.plan_label(true), text_width),
                key,
                value,
            ));
            if let Some(agent) = self.selected_agent_row() {
                lines.push(detail_line(
                    "Executor",
                    &format!(
                        "{} · {} · {}",
                        selected_tally(self.selected_agent, phase.agents.len()),
                        short(&agent.name, text_width),
                        agent.status
                    ),
                    key,
                    agent_status_style(&agent.status, theme),
                ));
                let activity = agent
                    .current_tool
                    .as_deref()
                    .or(agent.current_phase.as_deref())
                    .unwrap_or("waiting for activity");
                lines.push(detail_line(
                    "Activity",
                    &short(activity, text_width),
                    key,
                    Style::new().fg(theme.palette.teal),
                ));
            } else {
                lines.push(detail_line("Executor", "none started", key, value));
            }
        } else {
            lines.push(Line::styled(
                "no workflow scope",
                detail_secondary_style(theme),
            ));
        }
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(lines, area.width)),
            area,
        );
    }

    /// The two header rows: a progress band and the NOW line under it.
    ///
    /// This replaced a `RUN SCOPE unlinked fan-out → EXECUTORS 1/2 → DETAIL
    /// deep-research…` breadcrumb that restated, in one line, the three things
    /// the tree and the inspector were already showing. What a reader cannot
    /// get from those two panes is the *aggregate*: how far along the run is,
    /// and whether it is honestly linked to a Plan step.
    ///
    /// Exactly two rows, always: `regions()` grants the header a fixed height of
    /// 2 and every hit test measures against it.
    fn header_lines(&self, theme: &Theme, width: u16) -> Vec<Line<'static>> {
        vec![self.progress_band(theme, width), self.now_line(theme, width)]
    }

    /// The header's second row: *what is happening right now* — the active
    /// phase's own sentence, its coordinate in the run, the tool one of its
    /// executors is running, and which phase is up next. Pure: built from the
    /// snapshot the host already fed in, with no IO.
    fn now_line(&self, theme: &Theme, width: u16) -> Line<'static> {
        let dim = detail_secondary_style(theme);
        let total = self.view.phases.len();
        // Only a live run has a "now". `active_phase_index` deliberately falls
        // back to the last phase for terminal views (selection needs an
        // anchor), so gate on the run status here — a finished run must not
        // spin or resurrect a stale activeForm.
        let active = (self.view.status == "running")
            .then(|| self.view.active_phase_index())
            .flatten()
            .and_then(|index| self.view.phases.get(index).map(|phase| (index, phase)));
        let Some((index, phase)) = active else {
            // Terminal / empty view: one muted summary, still one row.
            let glyph = if self.view.status == "completed" { "✓ " } else { "" };
            return crate::tui::text_metrics::truncate_line_to_cells(
                Line::from(Span::styled(format!("{glyph}{}", self.view.status), dim)),
                usize::from(width),
            );
        };
        let marker = if reduce_motion_enabled() {
            "▸"
        } else {
            SPINNER[self.tick % SPINNER.len()]
        };
        let mut spans = vec![
            Span::styled(format!("{marker} "), Style::new().fg(theme.palette.accent)),
            Span::styled(
                phase.plan_label(true).to_string(),
                Style::new()
                    .fg(theme.palette.fg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" — {}/{total}", index + 1), dim),
        ];
        if let Some(tool) = phase
            .agents
            .iter()
            .find(|agent| agent.status == "running")
            .and_then(|agent| agent.current_tool.as_deref())
        {
            spans.push(Span::styled(format!(" · ⟶ {tool}"), dim));
        }
        // No `next:` segment here: the progress band's right shoulder
        // (`scope_note`) already names the upcoming phase, and repeating it put
        // a second "synthesize" on screen that shadowed the tree's own group
        // rows in text searches.
        crate::tui::text_metrics::truncate_line_to_cells(Line::from(spans), usize::from(width))
    }

    fn progress_band(&self, theme: &Theme, width: u16) -> Line<'static> {
        let dim = detail_secondary_style(theme);
        let (glyph, glyph_style) = phase_status_glyph(
            if self.view.status == "running" { "running" } else { "done" },
            theme,
            self.tick,
        );
        let status_label = if self.view.synthesizing {
            "synthesizing".to_string()
        } else {
            self.view.status.clone()
        };
        let percent = self.view.progress_percent();
        let mut left = vec![
            Span::styled(format!("{glyph} "), glyph_style),
            Span::styled(
                status_label,
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("   "),
        ];
        left.extend(gauge_spans(percent, theme));
        left.push(Span::styled(
            format!(
                "   {} executors · {} running",
                self.view.total_agents(),
                self.view.running_agents()
            ),
            dim,
        ));
        let failed = self.view.failed_agents();
        if failed > 0 {
            left.push(Span::styled(
                format!(" · {failed} failed"),
                theme.diff_del_style(),
            ));
        }

        // The right shoulder carries the caveat, never the progress: a run that
        // is not linked to a Plan step must say so, but it must not be the first
        // thing the eye lands on.
        let right = self.scope_note(theme);
        spread_line(left, right, width)
    }

    /// The run's Plan-honesty note, right-aligned in the header.
    fn scope_note(&self, theme: &Theme) -> Vec<Span<'static>> {
        let dim = detail_secondary_style(theme);
        match self.view.scope() {
            ViewerScope::Plan => {
                let Some(next) = self
                    .view
                    .active_phase_index()
                    .and_then(|idx| next_phase_index(&self.view, idx))
                    .and_then(|idx| self.view.phases.get(idx))
                else {
                    return Vec::new();
                };
                vec![
                    Span::styled("next ", dim),
                    Span::styled(
                        short(next.plan_label(false), 28),
                        detail_value_style(theme),
                    ),
                ]
            }
            ViewerScope::Workflow if self.view.plan_link_count() > 0 => vec![Span::styled(
                format!(
                    "{}/{} phases Plan linked",
                    self.view.plan_link_count(),
                    self.view.phases.len()
                ),
                dim,
            )],
            ViewerScope::Workflow => vec![Span::styled(
                "Plan link unavailable",
                Style::new().fg(theme.palette.warn),
            )],
            ViewerScope::Run => vec![Span::styled(
                "not linked to a Plan step",
                Style::new().fg(theme.palette.warn),
            )],
        }
    }

    /// The FLEET tree: every phase group with its executors under it, in one
    /// scrolling selector.
    fn draw_fleet(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let rows = self.fleet_rows();
        let title = format!(
            " FLEET \u{2022} {} ",
            selected_tally(self.executor_cursor(), self.view.total_agents())
        );
        let body_width = pane_inner(area, theme).width;
        let lines: Vec<Line<'static>> = rows
            .iter()
            .map(|row| self.fleet_line(*row, theme, body_width))
            .collect();
        let inner = CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, pane_title_style(theme)))
            .border_style(pane_rule_style(theme))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        if lines.is_empty() {
            frame.render_widget(
                Paragraph::new(super::fit_body_rows(
                    vec![Line::from(Span::styled(
                        "no executors yet",
                        detail_secondary_style(theme),
                    ))],
                    inner.width,
                )),
                inner,
            );
            return;
        }
        let offset = self.fleet_offset(inner.height, &rows);
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(lines, inner.width)).scroll((offset, 0)),
            inner,
        );
        draw_scrollbar(frame, inner, offset, rows.len(), theme);
    }

    fn fleet_line(&self, row: FleetRow, theme: &Theme, width: u16) -> Line<'static> {
        match row {
            FleetRow::Phase { phase } => self.fleet_phase_line(phase, theme, width),
            FleetRow::Executor { phase, agent } => {
                self.fleet_executor_line(phase, agent, theme, width)
            }
        }
    }

    /// A phase group header: `▾ Synthesize findings   2/3`.
    ///
    /// It is the scope row the old 40-cell Plan rail spent a whole column on.
    /// As a header it costs one row, and the executors it owns are visibly
    /// *under* it instead of in a second pane the reader had to correlate by
    /// hand.
    fn fleet_phase_line(&self, idx: usize, theme: &Theme, width: u16) -> Line<'static> {
        let Some(phase) = self.view.phases.get(idx) else {
            return Line::from("");
        };
        let current = idx == self.selected_phase;
        let (icon, icon_style) = phase_status_glyph(&phase.status, theme, self.tick);
        let label_style = if current {
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            detail_secondary_style(theme)
        };
        let tally = if phase.total > 0 {
            format!("{}/{}", phase.completed_now(), phase.total)
        } else {
            "queued".to_string()
        };
        let caret = if theme.no_color {
            "v "
        } else {
            "\u{25be} "
        };
        // Plan honesty, per group. The header states the run's *aggregate*
        // (`1/2 phases Plan linked`); without a marker here a reader could not
        // tell which of the two this one is.
        let unlinked = phase.scope() == ViewerScope::Workflow && self.view.plan_link_count() > 0;
        let suffix = if unlinked { "  \u{00b7} unlinked" } else { "" };
        let label_room = usize::from(width)
            .saturating_sub(
                crate::tui::text_metrics::display_width(caret)
                    + 3
                    + crate::tui::text_metrics::display_width(&tally)
                    + crate::tui::text_metrics::display_width(suffix),
            )
            .max(8);
        let mut left = vec![
            Span::styled(caret.to_string(), detail_label_style(theme)),
            Span::styled(format!("{icon} "), icon_style),
            Span::styled(short(phase.plan_label(true), label_room), label_style),
        ];
        if unlinked {
            left.push(Span::styled(suffix, Style::new().fg(theme.palette.warn)));
        }
        spread_line(
            left,
            vec![Span::styled(tally, detail_label_style(theme))],
            width,
        )
    }

    /// An executor under its group, drawn by the same row builder the Ctrl+G
    /// fleet list uses so an agent reads identically in both.
    fn fleet_executor_line(
        &self,
        phase: usize,
        agent_idx: usize,
        theme: &Theme,
        width: u16,
    ) -> Line<'static> {
        let Some(agent) = self
            .view
            .phases
            .get(phase)
            .and_then(|row| row.agents.get(agent_idx))
        else {
            return Line::from("");
        };
        let selected = phase == self.selected_phase && agent_idx == self.selected_agent;
        let indent = FLEET_INDENT_COLS;
        let line = agent_list_line(
            agent,
            selected,
            theme,
            self.tick,
            width.saturating_sub(u16::try_from(indent).unwrap_or(0)),
        );
        let wash = if selected {
            theme
                .selection_bg()
                .map_or_else(Style::new, |bg| Style::new().bg(bg))
        } else {
            Style::new()
        };
        let mut spans = vec![Span::styled(" ".repeat(indent), wash)];
        spans.extend(line.spans);
        let line = Line::from(spans);
        if selected {
            super::wash_row(line, width, theme)
        } else {
            line
        }
    }

    /// The executor detail pane. Wide enough it splits into meta │ output, so
    /// the streamed result gets a column of its own instead of trailing a dozen
    /// short metadata rows down a 120-cell pane.
    fn draw_agent_detail(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let title = self.selected_agent_row().map_or_else(
            || " Executor detail ".to_string(),
            |agent| format!(" Executor · {} ", short(&agent.name, 32)),
        );
        let Some(agent) = self.selected_agent_row() else {
            let inner = CardFrame::new(SurfaceKind::Panel, theme)
                .title(Line::styled(title, pane_title_style(theme)))
                .border_style(pane_rule_style(theme))
                .render(frame, area);
            frame.render_widget(
                Paragraph::new(super::fit_body_rows(
                    vec![Line::from(Span::styled(
                        "select an executor",
                        detail_secondary_style(theme),
                    ))],
                    inner.width,
                )),
                inner,
            );
            return;
        };
        // The inspector takes the band it was given, whole. Hugging it to its
        // content was the "giant void": on a tall terminal the card closed after
        // a dozen meta rows and the output tail — the one section with more to
        // say — was clipped above thirty rows of empty surface.
        let inner = CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, pane_title_style(theme)))
            .border_style(pane_rule_style(theme))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let Some(band) = self.inspector_band(agent, inner) else {
            let mut column = self.detail_meta_column(agent, inner.width, theme);
            column.push(Line::from(""));
            column.append(self.detail_output_column(agent, inner.width, theme));
            frame.render_widget(
                Paragraph::new(super::wrap_body_rows(&column.into_lines(), inner.width, false)),
                inner,
            );
            return;
        };
        let meta = self.detail_meta_column(agent, band.meta.width, theme);
        let output = self.detail_output_column(agent, band.output.width, theme);
        // The rail stops where the *taller* of the two columns does, so it never
        // hangs below the content it separates.
        let content_rows = output
            .row_count(band.output.width)
            .max(meta.row_count(band.meta.width))
            .min(usize::from(band.rail.height));
        frame.render_widget(
            Paragraph::new(super::wrap_body_rows(
                &meta.into_lines(),
                band.meta.width,
                false,
            )),
            band.meta,
        );
        draw_detail_rail(
            frame,
            band.rail,
            theme,
            u16::try_from(content_rows).unwrap_or(u16::MAX),
        );
        draw_output_column(frame, band.output, output, 0);
    }

    /// How the inspector divides itself, or `None` to keep one column.
    ///
    /// A column is only worth reserving when something will be *in* it. The
    /// geometric split alone gave a silent executor a 60-cell empty column
    /// beside a meta card whose values were being clipped for want of the very
    /// cells sitting blank next to them — the void, moved sideways.
    fn inspector_band(&self, agent: &WorkflowAgentRow, inner: Rect) -> Option<DetailBand> {
        let landed = self
            .output_tail
            .as_ref()
            .filter(|(path, _, tail)| {
                agent.output_file.as_deref() == Some(path.as_str()) && !tail.is_empty()
            })
            .map(|(_, _, tail)| tail.as_slice());
        split_detail_band_for(
            inner,
            agent,
            landed,
            self.folds.folded(DetailSection::Output),
        )
    }

    /// Left column: the shared agent meta card. The executor's place in the run
    /// is the tree's job — this pane used to restate it three times
    /// (`flow`, then a whole `SCOPE CONTEXT` block of `scope`/`phase`/`step`).
    fn detail_meta_column(
        &self,
        agent: &WorkflowAgentRow,
        width: u16,
        theme: &Theme,
    ) -> DetailColumn {
        let mut column = DetailColumn::default();
        column.append(agent_meta_column(agent, width, theme, self.folds));
        column
    }

    /// Right column: the agent's streamed prose, or — once it lands — the tail
    /// of its markdown result file (refreshed mtime-gated by the host tick,
    /// never read at draw time).
    fn detail_output_column(
        &self,
        agent: &WorkflowAgentRow,
        width: u16,
        theme: &Theme,
    ) -> DetailColumn {
        let landed = self
            .output_tail
            .as_ref()
            .filter(|(path, _, tail)| {
                agent.output_file.as_deref() == Some(path.as_str()) && !tail.is_empty()
            })
            .map(|(_, _, tail)| tail.as_slice());
        agent_output_column(
            agent,
            width,
            theme,
            self.folds.folded(DetailSection::Output),
            landed,
        )
    }
}

/// Cell budgets for an agent row's two elastic columns, given the row's own
/// width and the widths the data already fixed.
///
/// `width` is the pane the row paints into, so the same row reads correctly in
/// the workflow viewer's ~40-cell agent pane and in the Ctrl+G list at full modal
/// width. The name takes the larger share of the slack because it is what the
/// reader scans the fleet by; the activity column is a live detail that can
/// afford to end in `…`.
///
/// Each floor is the fixed budget it replaced, so a narrow pane renders exactly
/// as it did before and only a pane with room to spare gets more: the change can
/// add cells to a row, never take them away.
fn agent_row_budgets(
    width: u16,
    model_label: &str,
    metrics: &str,
    has_activity: bool,
) -> (usize, usize) {
    let fixed = FIXED_CHROME
        + if has_activity { ACTIVITY_CHROME } else { 0 }
        + crate::tui::text_metrics::display_width(model_label)
        + crate::tui::text_metrics::display_width(metrics);
    let elastic = usize::from(width).saturating_sub(fixed);
    if !has_activity {
        return (elastic.clamp(NAME_FLOOR, NAME_CEILING), 0);
    }
    let name = (elastic * 3 / 5).clamp(NAME_FLOOR, NAME_CEILING);
    let activity = elastic
        .saturating_sub(name)
        .clamp(ACTIVITY_FLOOR, ACTIVITY_CEILING);
    (name, activity)
}

/// Marker (2) + status glyph (2) + the two-space gap before the model.
const FIXED_CHROME: usize = 6;
/// `  ⟶ ` ahead of the activity text.
const ACTIVITY_CHROME: usize = 5;
const NAME_FLOOR: usize = 24;
const NAME_CEILING: usize = 48;
const ACTIVITY_FLOOR: usize = 18;
const ACTIVITY_CEILING: usize = 44;

/// The secondary columns that survive at this width, and the two elastic
/// budgets that go with them.
///
/// The budgets have *floors*, and below a certain pane width their sum exceeds
/// the row: the line then overflows and the paint clips it — silently dropping
/// the rightmost column, which is the live `⟶ tool` the reader opened the
/// viewer for. Shedding a secondary column on purpose keeps the row inside the
/// pane and keeps what the row is *for*. Both dropped columns are stated in
/// full by the inspector one pane over; the live activity is not.
/// `wants` is what the name and the activity text would take *uncut*: a budget
/// is a ceiling, and testing the fit against the floor instead threw away the
/// metrics of a seventeen-cell name that had room to spare.
fn agent_row_columns<'a>(
    width: u16,
    model: &'a str,
    metrics: &'a str,
    wants: (usize, usize),
    has_activity: bool,
) -> (&'a str, &'a str, usize, usize) {
    let (want_name, want_activity) = wants;
    for (model, metrics) in [(model, metrics), (model, ""), ("", "")] {
        let (name, activity) = agent_row_budgets(width, model, metrics, has_activity);
        let total = FIXED_CHROME
            + if has_activity { ACTIVITY_CHROME } else { 0 }
            + crate::tui::text_metrics::display_width(model)
            + crate::tui::text_metrics::display_width(metrics)
            + name.min(want_name)
            + activity.min(want_activity);
        if total <= usize::from(width) {
            return (model, metrics, name, activity);
        }
    }
    let (name, activity) = agent_row_budgets(width, "", "", has_activity);
    ("", "", name, activity)
}

/// One agent row for a selection list: status glyph, name, model, metrics and
/// the live tool/phase arrow. Shared by the workflow viewer's agent pane and
/// the Ctrl+G agents viewer so a fleet reads identically in both.
pub(crate) fn agent_list_line(
    agent: &WorkflowAgentRow,
    selected: bool,
    theme: &Theme,
    tick: usize,
    width: u16,
) -> Line<'static> {
    // Pi select-list row: `→ ✓ name  model  metrics  ⟶ activity`. The agent's
    // own hue (`Theme::agent_color`) carries its identity, so a fleet stays
    // scannable by color even when the names truncate to the same prefix.
    //
    // The secondary columns are `muted`, not `dim`: `Typography::dim` carries
    // `Modifier::DIM` on top of the `dim` hue, which took the model/metrics
    // columns below the legibility floor on the dialog surface.
    let secondary = detail_secondary_style(theme);
    let wash = if selected {
        match theme.selection_bg() {
            Some(bg) => Style::new().bg(bg),
            None => Style::new(),
        }
    } else {
        Style::new()
    };
    let (icon, icon_style) = agent_status_glyph(&agent.status, theme, tick);
    let marker = if selected {
        super::cursor_marker(!theme.no_color)
    } else {
        super::blank_marker()
    };
    let marker_style = if selected {
        super::selected_style(theme)
    } else {
        secondary
    };
    // On the selected row the slate wash already says "this one", so the raw
    // identity hue on top of it (two of the eight are a hot pink and a magenta)
    // was pure noise. Calm it toward `fg` — still that agent's colour, no
    // longer shouting over the selection.
    let name_hue = if selected {
        theme.calm_identity(theme.agent_color(&agent.name))
    } else {
        theme.agent_color(&agent.name)
    };
    let name_style = Style::new()
        .fg(name_hue)
        .add_modifier(Modifier::BOLD)
        .patch(wash);
    // Metrics: tokens · tools · elapsed. The token total is only persisted to
    // the manifest when the agent finishes, so it reads 0 for the whole run —
    // showing a bare "0" looks like a broken counter, so drop it until known
    // (matching the sidebar, which hides its sparkline while empty).
    let elapsed = format_elapsed(agent.elapsed_secs);
    let metrics = match (agent.tokens > 0, agent.tool_calls) {
        (true, Some(tool_calls)) => format!(
            "  {} · {} tools · {elapsed}",
            fmt_tokens(agent.tokens),
            tool_calls,
        ),
        (true, None) => format!("  {} · {elapsed}", fmt_tokens(agent.tokens)),
        (false, Some(tool_calls)) => format!("  {tool_calls} tools · {elapsed}"),
        (false, None) => format!("  {elapsed}"),
    };
    let model_label = short_model(&agent.model);
    // Name and live-activity are the row's two elastic columns; everything else
    // (marker, glyph, model, metrics) has a width the data fixes. Budgeting them
    // from the pane instead of from constants is what stops a 120-cell list from
    // cutting `review-engine-and-th…` with sixty columns still empty to its right.
    let activity_text = (agent.status == "running")
        .then(|| {
            agent
                .current_tool
                .as_deref()
                .or(agent.current_phase.as_deref())
        })
        .flatten();
    let wants = (
        crate::tui::text_metrics::display_width(&agent.name),
        activity_text.map_or(0, crate::tui::text_metrics::display_width),
    );
    let (model_label, metrics, name_budget, activity_budget) = agent_row_columns(
        width,
        &model_label,
        &metrics,
        wants,
        activity_text.is_some(),
    );
    let mut spans = vec![
        Span::styled(marker.to_string(), marker_style),
        Span::styled(format!("{icon} "), icon_style.patch(wash)),
        Span::styled(short(&agent.name, name_budget), name_style),
    ];
    if !model_label.is_empty() {
        spans.push(Span::styled("  ", wash));
        spans.push(Span::styled(model_label.to_string(), secondary.patch(wash)));
    }
    spans.push(Span::styled(metrics.to_string(), secondary.patch(wash)));
    // Live activity for a running agent: the concrete tool if one is running,
    // otherwise the transient phase (e.g. `thinking`) so a between-tools agent
    // still reads as *doing something* instead of looking idle. Phase rides the
    // reasoning `violet` (waiting to think is not a warning) and only shows when
    // no tool is active — the writer clears the phase on tool start, so gating
    // on `current_tool.is_none()` suppresses flicker.
    if agent.status == "running" {
        if let Some(tool) = &agent.current_tool {
            // Tool/activity text takes the tool role hue (teal), matching the
            // transcript's tool rail.
            spans.push(Span::styled(
                format!("  ⟶ {}", short(tool, activity_budget)),
                Style::new().fg(theme.palette.teal).patch(wash),
            ));
        } else if let Some(phase) = &agent.current_phase {
            spans.push(Span::styled(
                format!("  ⟶ {}", short(phase, activity_budget)),
                Style::new().fg(theme.palette.violet).patch(wash),
            ));
        }
    }
    Line::from(spans)
}

// ── Agent detail band ───────────────────────────────────────────────────────
//
// Both viewers (Ctrl+O workflow, Ctrl+G agents) render the same agent detail
// card, and both used to stack it in one full-width column. The card's rows are
// ~40 cells of text; on a 200-column terminal that wasted every column past the
// first forty on every row, while the streamed output dribbled down the left
// edge and left a band of empty surface under it. Wide enough, the card now
// splits: meta on the left, the output tail on the right, where it takes the
// band's whole height.

/// Minimum width of the meta column before the band may split.
pub(crate) const DETAIL_META_MIN_WIDTH: u16 = 40;
/// Minimum width of the output column before the band may split.
pub(crate) const DETAIL_OUTPUT_MIN_WIDTH: u16 = 40;
/// The `│` rail drawn between the two columns.
const DETAIL_RAIL_WIDTH: u16 = 1;
/// Ceiling for the meta column: past it the rows are all padding, and the extra
/// width belongs to the output tail. Sixty-four cells is one label column (8) +
/// one gap + a 55-cell value, which holds every field the card emits without
/// clipping a normal task line.
const DETAIL_META_MAX_WIDTH: u16 = 64;
/// Share of the band the meta column takes, as a percentage. The old split
/// handed the meta column a fixed floor plus two fifths of the *slack*, which
/// on a wide terminal pinned it at its ceiling and left the label column
/// starved on everything narrower.
///
/// Just under half, not two fifths: a meta row is a `label value` pair whose
/// value is frequently a 60-cell route reason or task line, and at 40% of a
/// mid-width band it clipped one (`auto:coding tier=strong · learned-shad…`)
/// while the output column beside it had cells to spare. The ceiling still
/// hands a wide terminal's extra columns to the stream.
const DETAIL_META_PERCENT: u16 = 46;
/// Band width at which the two-column split forms: both columns at their floor,
/// the rail, and one column of Pi gutter air on each side of it.
pub(crate) const DETAIL_SPLIT_MIN_WIDTH: u16 =
    DETAIL_META_MIN_WIDTH + DETAIL_OUTPUT_MIN_WIDTH + DETAIL_RAIL_WIDTH + 2 * PANE_GUTTER;

/// The two columns of a split detail band and the rail between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DetailBand {
    /// Left column: status, task, model, metrics, activity feed.
    pub meta: Rect,
    /// One-column `│` rail.
    pub rail: Rect,
    /// Right column: the streamed output tail, filling the band's height.
    pub output: Rect,
}

/// Split `area` into `meta │ output`, or `None` when it is too narrow — the
/// caller then keeps the single-column stack. A narrow terminal must lose the
/// second column, never the card.
pub(crate) fn split_detail_band(area: Rect) -> Option<DetailBand> {
    if area.width < DETAIL_SPLIT_MIN_WIDTH || area.height == 0 {
        return None;
    }
    // Proportional, not "floor plus slack": the meta column is a fixed share of
    // the band, floored so a bash brief still fits and capped so a wide terminal
    // spends its extra columns on the stream rather than on padding.
    let headroom = area
        .width
        .saturating_sub(DETAIL_OUTPUT_MIN_WIDTH + DETAIL_RAIL_WIDTH + 2 * PANE_GUTTER);
    let meta_width = (area.width * DETAIL_META_PERCENT / 100)
        .clamp(DETAIL_META_MIN_WIDTH, DETAIL_META_MAX_WIDTH)
        .min(headroom.max(DETAIL_META_MIN_WIDTH));
    let [meta, rail, output] = Layout::horizontal([
        Constraint::Length(meta_width),
        Constraint::Length(DETAIL_RAIL_WIDTH),
        Constraint::Min(DETAIL_OUTPUT_MIN_WIDTH),
    ])
    .spacing(PANE_GUTTER)
    .areas(area);
    Some(DetailBand { meta, rail, output })
}

/// [`split_detail_band`], but only when the output has something to put in the
/// column it would take.
///
/// A column is worth reserving only when something will be *in* it. The
/// geometric split alone gave a silent agent a 60-cell empty column beside a
/// meta card whose values were being clipped for want of the very cells sitting
/// blank next to them — the void, moved sideways. Shared by both viewers so a
/// quiet agent reads the same in each.
pub(crate) fn split_detail_band_for(
    area: Rect,
    agent: &WorkflowAgentRow,
    landed_tail: Option<&[String]>,
    folded: bool,
) -> Option<DetailBand> {
    if folded {
        return None;
    }
    let has_output = landed_tail.is_some_and(|tail| !tail.is_empty())
        || agent
            .output_tail
            .as_deref()
            .is_some_and(|tail| !tail.trim().is_empty());
    // A running agent is one token away from streaming, and a column that
    // appears mid-stream jumps the layout under the reader — but *speculating*
    // must never cost the meta card a field it could otherwise show. Reserve it
    // only once meta is already at its ceiling, where the cells taken are the
    // ones it had no use for anyway.
    let anticipated = agent.status == "running"
        && area.width * DETAIL_META_PERCENT / 100 >= DETAIL_META_MAX_WIDTH;
    (has_output || anticipated)
        .then(|| split_detail_band(area))
        .flatten()
}

/// Paint the `│` rail between the two columns, in the resting `border` hue —
/// the sidebar's `draw_left_anchor_rule` tone.
///
/// It starts one row *below* the band so the two section rules read as separate
/// inlays instead of meeting in a box corner: Pi separates panes with air and a
/// stroke, never with a frame.
///
/// `content_rows` is how many rows the **output column** actually painted: the
/// rail is that column's left edge, so it ends where the column does. Drawn to
/// the band's full height instead it hangs below the last streamed row — a
/// dozen rows of stroke with nothing to their right — which is exactly what a
/// silent or folded stream looked like beside a full meta card.
pub(crate) fn draw_detail_rail(
    frame: &mut Frame<'_>,
    rail: Rect,
    theme: &Theme,
    content_rows: u16,
) {
    let span = rail.height.min(content_rows);
    if rail.width == 0 || span <= 1 {
        return;
    }
    let glyph = if theme.no_color {
        glyphs::VERTICAL_SEP_NC
    } else {
        glyphs::VERTICAL_SEP
    };
    let rows: Vec<Line<'static>> = (1..span).map(|_| Line::from(glyph)).collect();
    frame.render_widget(
        Paragraph::new(rows).style(Style::new().fg(theme.palette.border)),
        Rect {
            y: rail.y.saturating_add(1),
            height: span.saturating_sub(1),
            ..rail
        },
    );
}

/// A section of the detail card a click can fold away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetailSection {
    /// The rolling tool feed.
    Activity,
    /// The streamed output tail.
    Output,
}

/// Which foldable sections are collapsed. Lives on the *modal*, not on the
/// agent: a reader who folded the output keeps it folded while walking the
/// fleet, which is the whole point of folding it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DetailFolds {
    activity: bool,
    output: bool,
}

impl DetailFolds {
    #[must_use]
    pub(crate) const fn folded(self, section: DetailSection) -> bool {
        match section {
            DetailSection::Activity => self.activity,
            DetailSection::Output => self.output,
        }
    }

    pub(crate) fn toggle(&mut self, section: DetailSection) {
        match section {
            DetailSection::Activity => self.activity = !self.activity,
            DetailSection::Output => self.output = !self.output,
        }
    }
}

/// One rendered column of the detail card: the lines to paint plus where its
/// foldable headings landed.
///
/// Carrying the heading rows with the content is what lets a click be mapped
/// back to the section the reader aimed at *from the very lines the frame
/// painted* — a second, independent layout pass is exactly how hit-testing
/// drifts from the pixels.
#[derive(Debug, Default)]
pub(crate) struct DetailColumn {
    lines: Vec<Line<'static>>,
    headings: Vec<(usize, DetailSection)>,
}

impl DetailColumn {
    pub(crate) fn push(&mut self, line: Line<'static>) {
        self.lines.push(line);
    }

    fn push_heading(&mut self, section: DetailSection, line: Line<'static>) {
        self.headings.push((self.lines.len(), section));
        self.lines.push(line);
    }

    /// Append `other` below this column, shifting its heading rows.
    pub(crate) fn append(&mut self, other: Self) {
        let offset = self.lines.len();
        self.headings.extend(
            other
                .headings
                .into_iter()
                .map(|(row, section)| (row + offset, section)),
        );
        self.lines.extend(other.lines);
    }

    /// The lines, ready for a `Paragraph`.
    pub(crate) fn into_lines(self) -> Vec<Line<'static>> {
        self.lines
    }

    /// Peel the leading heading row off the body so a caller can pin it while
    /// the body scrolls underneath.
    ///
    /// A long tail scrolls its own `▾ LIVE OUTPUT ───` off the top otherwise —
    /// taking with it the only thing that says the section can be folded, on
    /// exactly the agent busy enough to want it folded.
    pub(crate) fn split_heading(self) -> (Option<Line<'static>>, Vec<Line<'static>>) {
        let mut lines = self.lines.into_iter();
        let leads_with_heading = self.headings.first().is_some_and(|(row, _)| *row == 0);
        if !leads_with_heading {
            return (None, lines.collect());
        }
        (lines.next(), lines.collect())
    }

    /// Terminal rows the column occupies once wrapped into `width`.
    pub(crate) fn row_count(&self, width: u16) -> usize {
        self.lines
            .iter()
            .map(|line| wrapped_rows(line, width))
            .sum()
    }

    /// The foldable section whose heading sits on `row` (0-based inside the
    /// column, scroll already applied), or `None` for a body row.
    pub(crate) fn section_at_row(&self, row: usize, width: u16) -> Option<DetailSection> {
        let mut cursor = 0usize;
        for (idx, line) in self.lines.iter().enumerate() {
            let rows = wrapped_rows(line, width);
            if row < cursor + rows {
                return self
                    .headings
                    .iter()
                    .find(|(heading, _)| *heading == idx)
                    .map(|(_, section)| *section);
            }
            cursor += rows;
        }
        None
    }
}

/// Paint an output column: its section heading pinned to the first row, the
/// tail scrolling underneath.
///
/// The body follows the live tail — the newest line lands on the column's last
/// row, which is what makes a streaming agent read as *live* instead of as a
/// stalled first paragraph — and `scroll_back` walks that anchor backwards
/// (`0` = following).
pub(crate) fn draw_output_column(
    frame: &mut Frame<'_>,
    area: Rect,
    column: DetailColumn,
    scroll_back: u16,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let width = area.width;
    let (heading, body) = column.split_heading();
    let Some(heading) = heading else {
        frame.render_widget(
            Paragraph::new(super::wrap_body_rows(&body, area.width, false)),
            area,
        );
        return;
    };
    let [head_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
    frame.render_widget(
        Paragraph::new(super::fit_body_rows(vec![heading], head_area.width)),
        head_area,
    );
    if body_area.height == 0 || body.is_empty() {
        return;
    }
    let rows = super::wrap_body_rows(&body, width, false);
    let scroll = u16::try_from(rows.len())
        .unwrap_or(u16::MAX)
        .saturating_sub(body_area.height)
        .saturating_sub(scroll_back);
    frame.render_widget(Paragraph::new(rows).scroll((scroll, 0)), body_area);
}

/// Rows one line occupies once wrapped into `width`, measured with the same
/// wrapper that paints it (`super::wrap_body_rows`) so a hit test can never
/// disagree with the frame.
fn wrapped_rows(line: &Line<'static>, width: u16) -> usize {
    if width == 0 {
        return 1;
    }
    crate::tui::text_metrics::wrap_line_to_cells(line, usize::from(width), false)
        .len()
        .max(1)
}

/// Disclosure caret for a foldable heading.
fn fold_caret(theme: &Theme, folded: bool) -> &'static str {
    match (theme.no_color, folded) {
        (false, true) => glyphs::CHEVRON_RIGHT,
        (false, false) => glyphs::CHEVRON_DOWN,
        (true, true) => glyphs::CHEVRON_RIGHT_NC,
        (true, false) => glyphs::CHEVRON_DOWN_NC,
    }
}

/// [`section_heading_rule`] for a section a click can fold: the rule's leading
/// `─` becomes a `▾`/`▸` disclosure caret, and a folded section states what it
/// is holding back (`▸ LIVE OUTPUT · 12 lines ─────`) instead of vanishing.
///
/// The span shape matches [`section_rule`] exactly (rule / label / rule), so
/// every contrast assertion pinned on a heading holds for a foldable one.
pub(crate) fn foldable_heading_rule(
    label: &str,
    width: u16,
    theme: &Theme,
    folded: bool,
    hidden_rows: usize,
) -> Line<'static> {
    let rule_style = Style::new().fg(theme.palette.border);
    let caret = fold_caret(theme, folded);
    let label = if folded {
        format!("{} \u{00b7} {hidden_rows} lines", label.to_uppercase())
    } else {
        label.to_uppercase()
    };
    let used = UnicodeWidthStr::width(caret) + UnicodeWidthStr::width(label.as_str()) + 2;
    let fill = usize::from(width).saturating_sub(used).max(4);
    Line::from(vec![
        Span::styled(format!("{caret} "), rule_style),
        Span::styled(
            label,
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {}", "\u{2500}".repeat(fill)), rule_style),
    ])
}

/// `tokens · tools · elapsed`, plus the heartbeat for a running agent.
///
/// The token total is only persisted when the agent finishes, so it reads 0 for
/// the whole run — a bare "0" looks like a broken counter, so it is dropped
/// until known.
fn agent_metrics_text(agent: &WorkflowAgentRow) -> String {
    let elapsed = format_elapsed(agent.elapsed_secs);
    let metrics = match (agent.tokens > 0, agent.tool_calls) {
        (true, Some(tool_calls)) => format!(
            "{} · {tool_calls} tools · {elapsed}",
            fmt_tokens(agent.tokens)
        ),
        (true, None) => format!("{} · {elapsed}", fmt_tokens(agent.tokens)),
        (false, Some(tool_calls)) => format!("{tool_calls} tools · {elapsed}"),
        (false, None) => elapsed,
    };
    match (agent.status == "running", agent.idle_secs) {
        // Heartbeat: how long since the agent last wrote its manifest. A
        // climbing value on a "running" agent is the signal it has stalled.
        (true, Some(idle)) => format!("{metrics} · active {idle}s ago"),
        _ => metrics,
    }
}

/// The meta half of the agent detail card, laid out over `width` cells (the
/// section rules fill to it; pass the column's width). Everything except the
/// streamed output, which the split layout gives a column of its own.
///
/// ## Reading order
///
/// What it is doing (`status`/failure/`tool`), what it was asked to do
/// (`task`), what is running it (`model`), how it is going (`metrics`) — then
/// the plumbing. `id`, `route`, `event` and above all the 90-cell `output` path
/// are reference material: they used to sit *above* the task and pushed the one
/// row a reader actually scans off a short pane.
///
/// ## Contrast ladder
///
/// Every row is `muted` label + `fg` value, and only a value that *means*
/// something takes a hue: failure in `error`, the running tool in `teal`, the
/// between-tools wait in `violet`, a path in `cyan`. The pane used to render
/// both columns through `key_hint`/`dim` — `dim` plus `Modifier::DIM` — so the
/// whole card sat at ~1.5:1 on the dialog surface and read as empty.
pub(crate) fn agent_meta_column(
    agent: &WorkflowAgentRow,
    width: u16,
    theme: &Theme,
    folds: DetailFolds,
) -> DetailColumn {
    let key = detail_label_style(theme);
    let value = detail_value_style(theme);
    let mut column = DetailColumn::default();
    column.push(detail_row(
        "status",
        &agent.status,
        key,
        agent_status_style(&agent.status, theme),
        width,
    ));
    if let Some(error) = &agent.error {
        column.push(detail_row("error", error, key, failure_style(theme), width));
    }
    if let Some(blocker) = &agent.blocker {
        column.push(detail_row(
            "blocker",
            blocker,
            key,
            failure_style(theme),
            width,
        ));
    }
    if let Some(tool) = &agent.current_tool {
        column.push(detail_row(
            "tool",
            tool,
            key,
            Style::new().fg(theme.palette.teal),
            width,
        ));
    } else if let Some(phase) = &agent.current_phase {
        // The between-tools wait (`thinking`) rides the reasoning violet, not
        // the `warn` yellow it used to: waiting to think is not a warning.
        column.push(detail_row(
            "activity",
            phase,
            key,
            Style::new().fg(theme.palette.violet),
            width,
        ));
    }
    if !agent.description.is_empty() {
        column.push(detail_row("task", &agent.description, key, value, width));
    }
    let model_label = short_model(&agent.model);
    if !model_label.is_empty() {
        column.push(detail_row("model", &model_label, key, value, width));
    }
    column.push(detail_row(
        "metrics",
        &agent_metrics_text(agent),
        key,
        value,
        width,
    ));
    append_plumbing_rows(&mut column, agent, width, theme);
    append_activity_section(&mut column, agent, width, theme, folds);
    column
}

/// The reference half of the card — what ran it and where its result went.
/// Split off the live half by one row of air: the reader scans the block above
/// every refresh and this one only when something is wrong.
fn append_plumbing_rows(
    column: &mut DetailColumn,
    agent: &WorkflowAgentRow,
    width: u16,
    theme: &Theme,
) {
    let key = detail_label_style(theme);
    let value = detail_value_style(theme);
    column.push(Line::from(""));
    if let Some(kind) = &agent.subagent_type {
        column.push(detail_row("type", kind, key, value, width));
    }
    if let Some(reason) = &agent.route_reason {
        column.push(detail_row("route", reason, key, value, width));
    }
    if let Some(event) = &agent.last_event {
        column.push(detail_row("event", event, key, value, width));
    }
    column.push(detail_row("id", &agent.id, key, value, width));
    if let Some(path) = &agent.output_file {
        // A path is a path everywhere in the TUI: `cyan`, the same hue the
        // transcript's file headers use. Clipped to one row from the *left*:
        // a wrapped path spilled its tail under the label column, and the tail
        // is the only part that names the file.
        column.push(detail_line(
            "output",
            &short_path(path, detail_value_room(width)),
            key,
            Style::new().fg(theme.palette.cyan),
        ));
    }
}

/// Live activity transcript: the manifest's rolling `recentTools` feed (oldest
/// → newest). The newest entry is the work happening right now for a running
/// agent, so it carries the tool hue; the history behind it stays `muted` —
/// quiet, but a readable quiet.
fn append_activity_section(
    column: &mut DetailColumn,
    agent: &WorkflowAgentRow,
    width: u16,
    theme: &Theme,
    folds: DetailFolds,
) {
    if agent.recent_tools.is_empty() {
        return;
    }
    let folded = folds.folded(DetailSection::Activity);
    column.push(Line::from(""));
    column.push_heading(
        DetailSection::Activity,
        foldable_heading_rule("activity", width, theme, folded, agent.recent_tools.len()),
    );
    if folded {
        return;
    }
    let running = agent.status == "running";
    let runs = fold_tool_runs(&agent.recent_tools);
    let newest = runs.len().saturating_sub(1);
    let verbs: Vec<String> = runs
        .iter()
        .map(|run| {
            if run.count > 1 {
                format!("{} \u{00d7}{}", run.tool, run.count)
            } else {
                run.tool.clone()
            }
        })
        .collect();
    // The verb column is as wide as the widest verb *present*, capped — the
    // arguments line up into a scannable second column without a fixed budget
    // charging a four-letter `bash` for eleven cells it does not use.
    let name_cols = verbs
        .iter()
        .map(|verb| crate::tui::text_metrics::display_width(verb))
        .max()
        .unwrap_or(0)
        .min(TOOL_NAME_MAX_COLS);
    // One row per *run*, always. A tool brief carrying a path is longer than any
    // column this card ever gets, and letting it wrap put its remainder at
    // column 0 of the next row — a list that has stopped being a list.
    let room = usize::from(width)
        .saturating_sub(BODY_INDENT_COLS + 2 + name_cols + 1)
        .max(12);
    for (i, (run, verb)) in runs.iter().zip(&verbs).enumerate() {
        let style = if running && i == newest {
            Style::new().fg(theme.palette.teal)
        } else {
            detail_secondary_style(theme)
        };
        let verb = short(verb, name_cols);
        let pad = name_cols.saturating_sub(crate::tui::text_metrics::display_width(&verb));
        column.push(Line::from(vec![
            Span::styled("  \u{00b7} ", detail_label_style(theme)),
            Span::styled(format!("{verb}{} ", " ".repeat(pad)), style),
            Span::styled(
                condense_tool_args(&run.args, room),
                detail_secondary_style(theme),
            ),
        ]));
    }
}

/// Ceiling on the tool-name column of an activity row. Past it a pathological
/// verb would push every argument off the card.
const TOOL_NAME_MAX_COLS: usize = 15;

/// A run of consecutive calls to the same tool, with each call's argument brief.
struct ToolRun {
    tool: String,
    count: usize,
    args: Vec<String>,
}

/// Fold `tool · arg` briefs into runs of the same tool.
///
/// Four `read_file` calls used to render as four rows of
/// `read_file /Users/joe/.zo/pro…4cc/memory/e…` — the middle-truncation ate
/// exactly the segment that differed, so the feed carried four rows and zero
/// bits. One row saying `read_file ×4` with the four *basenames* carries the
/// same four calls and actually distinguishes them.
fn fold_tool_runs(entries: &[String]) -> Vec<ToolRun> {
    let mut runs: Vec<ToolRun> = Vec::new();
    for entry in entries {
        let (tool, arg) = entry
            .split_once(" \u{00b7} ")
            .map_or((entry.trim(), ""), |(tool, arg)| (tool.trim(), arg.trim()));
        match runs.last_mut() {
            Some(run) if run.tool == tool => {
                run.count += 1;
                if !arg.is_empty() {
                    run.args.push(arg.to_string());
                }
            }
            _ => runs.push(ToolRun {
                tool: tool.to_string(),
                count: 1,
                args: if arg.is_empty() {
                    Vec::new()
                } else {
                    vec![arg.to_string()]
                },
            }),
        }
    }
    runs
}

/// The argument column of an activity row: distinct, shortened arguments joined
/// until they fill `room`, then `+N` for the rest.
fn condense_tool_args(args: &[String], room: usize) -> String {
    let mut seen: Vec<String> = Vec::new();
    for arg in args {
        let brief = brief_tool_arg(arg);
        if !brief.is_empty() && !seen.contains(&brief) {
            seen.push(brief);
        }
    }
    if seen.is_empty() {
        return String::new();
    }
    if seen.len() == 1 {
        return short_middle(&seen[0], room);
    }
    let mut out = String::new();
    let mut shown = 0usize;
    for brief in &seen {
        let candidate = if out.is_empty() {
            brief.clone()
        } else {
            format!("{out}, {brief}")
        };
        // Keep a few cells back for the `+N` tail so it never gets clipped off.
        if crate::tui::text_metrics::display_width(&candidate) > room.saturating_sub(5) {
            break;
        }
        out = candidate;
        shown += 1;
    }
    if shown == 0 {
        return short_middle(&seen[0], room);
    }
    if shown < seen.len() {
        let _ = write!(out, ", +{}", seen.len() - shown);
    }
    out
}

/// One argument, shortened to the part that identifies it.
///
/// A tool brief is dominated by absolute paths that share a 40-cell prefix
/// nobody reads: four `read_file` rows differing only past
/// `/Users/joe/.zo/projects/…/` truncated to four *identical* rows. Every
/// path-shaped token keeps its last two segments — the directory that gives the
/// file meaning, and the file — wherever it sits in the brief, so a redirect
/// target at the end of a bash line survives as readably as a leading path.
fn brief_tool_arg(arg: &str) -> String {
    let arg = arg.trim();
    if !arg.contains('/') {
        return arg.to_string();
    }
    arg.split(' ')
        .map(|token| {
            let (lead, core) = split_punct_prefix(token);
            let segments: Vec<&str> = core.split('/').filter(|part| !part.is_empty()).collect();
            if segments.len() <= 2 || !core.contains('/') {
                return token.to_string();
            }
            format!(
                "{lead}{}/{}",
                segments[segments.len() - 2],
                segments[segments.len() - 1]
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split a token's leading quote/bracket punctuation off its body, so a quoted
/// path shortens without losing the quote that says it is one.
fn split_punct_prefix(token: &str) -> (&str, &str) {
    let idx = token
        .find(|ch: char| !matches!(ch, '"' | '\'' | '(' | '[' | '<' | '`'))
        .unwrap_or(token.len());
    token.split_at(idx)
}

/// The output half of the detail card: *what the agent is actually saying*.
///
/// The manifest carries a rolling `outputTail` while the agent streams; once it
/// finishes, the host hands us the landed markdown file's tail in `landed_tail`
/// (read on the host tick, never in the draw path). The landed result wins when
/// both exist — it is the finished answer, the live buffer only its last few
/// lines. With neither, the section says so: an empty pane reads as a broken
/// viewer.
pub(crate) fn agent_output_column(
    agent: &WorkflowAgentRow,
    width: u16,
    theme: &Theme,
    folded: bool,
    landed_tail: Option<&[String]>,
) -> DetailColumn {
    let landed = landed_tail.unwrap_or_default();
    let (label, source) = if landed.is_empty() {
        let live: Vec<String> = agent
            .output_tail
            .as_deref()
            .map(|tail| tail.lines().map(str::to_string).collect())
            .unwrap_or_default();
        (
            if agent.status == "running" {
                "live output"
            } else {
                "output"
            },
            live,
        )
    } else {
        ("output tail", landed.to_vec())
    };
    let rows = render_output_rows(&source, width, theme);

    let mut column = DetailColumn::default();
    column.push_heading(
        DetailSection::Output,
        foldable_heading_rule(label, width, theme, folded, rows.len().max(1)),
    );
    if folded {
        return column;
    }
    if rows.is_empty() {
        let placeholder = if agent.status == "running" {
            format!(
                "  no output yet · started {} ago",
                format_elapsed(agent.elapsed_secs)
            )
        } else {
            "  no streamed output recorded".to_string()
        };
        column.push(Line::from(Span::styled(
            placeholder,
            detail_secondary_style(theme),
        )));
        return column;
    }
    for row in rows {
        let mut spans = vec![Span::raw(" ".repeat(BODY_INDENT_COLS))];
        spans.extend(row.spans);
        column.push(Line::from(spans));
    }
    column
}

/// An agent's output, rendered as the markdown it is.
///
/// An agent result file opens with `# Agent Task` / `- id: …` / `## Prompt`, and
/// the viewer used to paint those four characters of syntax verbatim — the one
/// pane that carries the actual answer was the only surface in the TUI that did
/// not render prose. It goes through the transcript's own renderer now, so a
/// heading reads as a heading and a fenced block as code.
///
/// Highlighting is off (`rendered_tail_for_width`): a running agent's tail is
/// re-rendered every frame, and syntect is stateful per line — the cost that
/// froze the spinner on long answers. Above the streaming renderer's size bound
/// it falls back to plain wrapped rows rather than spending the frame.
///
/// The gate is `has_strong_markdown_signal` — a heading, a fence or balanced
/// `**`. Without one the tail is *lines*, not a document, and running it
/// through pulldown would reflow a log into a paragraph: the same mistake in
/// the other direction. Agent result files open with `# Agent Task`, so the
/// case that prompted this renders; a plain stream of log lines does not.
fn render_output_rows(source: &[String], width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let room = u16::try_from(usize::from(width).saturating_sub(BODY_INDENT_COLS).max(8))
        .unwrap_or(width);
    let start = source.len().saturating_sub(OUTPUT_TAIL_MAX_ROWS);
    let tail = &source[start..];
    let text = tail.join("\n");
    if text.trim().is_empty() {
        return Vec::new();
    }
    let plain = || -> Vec<Line<'static>> {
        wrap_output_rows(tail, width)
            .into_iter()
            .map(|row| Line::from(Span::styled(row, detail_value_style(theme))))
            .collect()
    };
    let mut rows = if crate::tui::markdown::has_strong_markdown_signal(&text) {
        crate::tui::markdown::rendered_bounded_streaming_tail_for_width(&text, theme, room)
            .unwrap_or_else(plain)
    } else {
        plain()
    };
    let overflow = rows.len().saturating_sub(OUTPUT_TAIL_MAX_ROWS);
    if overflow > 0 {
        rows.drain(..overflow);
    }
    // A trailing newline (and the blank row a renderer closes a block with) is
    // not content: left in, it pushes the newest line off the column's last row
    // and a live stream stops looking live.
    while rows
        .last()
        .is_some_and(|line| line.spans.iter().all(|span| span.content.trim().is_empty()))
    {
        rows.pop();
    }
    rows
}

/// Indent of a body row under a section heading, in columns. Two cells, the
/// same step the activity feed's `· ` bullet sits on.
const BODY_INDENT_COLS: usize = 2;

/// Rows kept from an output tail. A cap there has to be — the manifest's rolling
/// buffer is bounded but a landed markdown file is not — but it belongs on the
/// **row** count, not on each row's length.
///
/// The old cap clipped every source line to 200 characters, and a live
/// `outputTail` is one unbroken 2000-character blob until the model emits a
/// newline: a fifty-five-row column rendered *one* row of it and left the rest
/// of the band empty. Four hundred rows is deeper than any terminal and still
/// cheap to wrap.
const OUTPUT_TAIL_MAX_ROWS: usize = 400;

/// Wrap an output tail into rows that each fit `width` exactly once indented.
///
/// Pre-wrapping (rather than handing long lines to `Wrap { trim: false }`) is
/// what lets the indent survive a wrap — ratatui restarts a wrapped row at
/// column 0 — and makes every measured row a real terminal row, so the hug, the
/// tail-following scroll and the fold hit-test all agree with the frame.
fn wrap_output_rows(source: &[String], width: u16) -> Vec<String> {
    let room = usize::from(width).saturating_sub(BODY_INDENT_COLS).max(8);
    let start = source.len().saturating_sub(OUTPUT_TAIL_MAX_ROWS);
    let mut rows: Vec<String> = Vec::new();
    for line in &source[start..] {
        if line.trim().is_empty() {
            rows.push(String::new());
        } else {
            rows.extend(wrap_to_width(line, room));
        }
    }
    let overflow = rows.len().saturating_sub(OUTPUT_TAIL_MAX_ROWS);
    if overflow > 0 {
        rows.drain(..overflow);
    }
    rows
}

/// Value style for a `status` row: the status glyph's own hue, with a failure
/// additionally BOLD so a dead executor is the first thing the eye lands on.
fn agent_status_style(status: &str, theme: &Theme) -> Style {
    let style = agent_status_glyph(status, theme, 0).1;
    if status == "failed" {
        style.add_modifier(Modifier::BOLD)
    } else {
        style
    }
}

/// Value style for an error/blocker row: `error`, BOLD.
fn failure_style(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.palette.error)
        .add_modifier(Modifier::BOLD)
}

/// Read the last `max_lines` lines of `path`, scanning at most `tail_bytes`
/// from the end so a huge output file never blocks the tick.
fn read_tail_lines(path: &str, file_len: u64, tail_bytes: u64, max_lines: usize) -> Vec<String> {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let start = file_len.saturating_sub(tail_bytes);
    if start > 0 && file.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = String::new();
    if file.read_to_string(&mut buf).is_err() {
        return Vec::new();
    }
    let mut lines: Vec<String> = buf
        .lines()
        .map(|line| line.trim_end().to_string())
        .collect();
    // A mid-line cut at the byte window start renders as garbage — drop it.
    if start > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    let overflow = lines.len().saturating_sub(max_lines);
    if overflow > 0 {
        lines.drain(..overflow);
    }
    lines
}

fn footer_line(
    theme: &Theme,
    width: u16,
    events_mode: bool,
    events_available: bool,
) -> Line<'static> {
    if events_mode {
        let full = super::key_hint_footer_reflowing(
            theme,
            &[
                ("↑/↓", "events"),
                ("PgUp/PgDn", "page"),
                ("^E", "back"),
                ("Esc", "close"),
            ],
        );
        if line_width(&full) <= usize::from(width) {
            return full;
        }
        let compact = super::key_hint_footer_with_separator(
            theme,
            &[("↑/↓", "events"), ("^E", "back"), ("Esc", "close")],
            " · ",
        );
        if line_width(&compact) <= usize::from(width) {
            return compact;
        }
        let minimal = super::key_hint_footer_with_separator(
            theme,
            &[("^E", "back"), ("Esc", "close")],
            " · ",
        );
        if line_width(&minimal) <= usize::from(width) {
            return minimal;
        }
        return super::key_hint_footer_reflowing(theme, &[("Esc", "close")]);
    }

    // `click` earns its cell in the footer: the panes have been clickable
    // since this line was written, but nothing said so — the reader's first
    // instinct (click the heading to fold the stream) looked broken.
    let mut full_hints = vec![
        ("↑/↓", "executor"),
        ("←/→", "phase"),
        ("PgUp/PgDn", "page"),
        ("click", "select/fold"),
    ];
    if events_available {
        full_hints.push(("^E", "events"));
    }
    full_hints.push(("Esc", "close"));
    let full = super::key_hint_footer_reflowing(theme, &full_hints);
    if line_width(&full) <= usize::from(width) {
        return full;
    }

    let compact = super::key_hint_footer_with_separator(
        theme,
        &[
            ("↑/↓", "executor"),
            ("←/→", "phase"),
            ("Esc", "close"),
        ],
        " · ",
    );
    if line_width(&compact) <= usize::from(width) {
        return compact;
    }

    let minimal = super::key_hint_footer_with_separator(
        theme,
        &[("↑/↓", "executor"), ("Esc", "close")],
        " · ",
    );
    if line_width(&minimal) <= usize::from(width) {
        return minimal;
    }
    super::key_hint_footer_reflowing(theme, &[("Esc", "close")])
}

/// A Pi section header inside a detail pane: a BOLD accent label riding a `─`
/// rule drawn in `border`, exactly like the dialog chrome one level up.
///
/// `width` is the column the rule fills to; pass `0` to emit a short fixed
/// rule when the caller does not know its width (the Paragraph clips anyway).
pub(crate) fn section_rule(label: &str, width: u16, theme: &Theme) -> Line<'static> {
    // `─ ` + label + ` ` + at least one trailing `─`: the shape needs 4 cells of
    // chrome, so below that there is no room for a title at all.
    const CHROME: usize = 4;

    let rule_style = Style::new().fg(theme.palette.border);
    let total = usize::from(width);
    if total <= CHROME {
        return Line::from(Span::styled("\u{2500}".repeat(total), rule_style));
    }
    // The label is user data (an agent name, a file path), so it is clamped here
    // rather than trusted: the rule renders with `wrap`, and an over-long title
    // used to fold onto a second row and steal a content row from the pane it
    // heads. `max(4)` on the fill was what let the row exceed `width`.
    let label = crate::tui::text_metrics::truncate_to_cells(label, total - CHROME);
    let fill = total - crate::tui::text_metrics::display_width(&label) - 3;
    Line::from(vec![
        Span::styled("\u{2500} ", rule_style),
        Span::styled(
            label,
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {}", "\u{2500}".repeat(fill)), rule_style),
    ])
}

/// [`section_rule`] for a *fixed* section title, uppercased the way the
/// sidebar's `section_heading_span` does it (`─ ACTIVITY ─────`).
///
/// Case is what separates the two kinds of heading the detail pane draws: an
/// uppercase inlay is a section of the card, a lowercase word in the label
/// column is a field of it. Callers whose title is user data (an agent name,
/// a file) keep their case and use [`section_rule`] directly.
pub(crate) fn section_heading_rule(label: &str, width: u16, theme: &Theme) -> Line<'static> {
    section_rule(&label.to_uppercase(), width, theme)
}

/// Rule color for a pane nested inside the viewer.
///
/// [`SurfaceKind::Panel`]'s default rule is `typography.dim` — `dim` *plus*
/// `Modifier::DIM` — which on the dialog surface is a rule nobody can see, so
/// the columns ran into each other and the modal never read as a stack of
/// surfaces. The panes take the same resting `border` the modal's own rules and
/// the sidebar's `draw_card_rules` use; hierarchy is carried by the *title*
/// instead (BOLD accent for the dialog, [`pane_title_style`] for a pane).
fn pane_rule_style(theme: &Theme) -> Style {
    Style::new().fg(theme.palette.border)
}

/// Title style for a nested pane: `muted` — readable, and a clear step below
/// the modal's own BOLD-accent dialog title.
fn pane_title_style(theme: &Theme) -> Style {
    Style::new().fg(theme.palette.muted)
}

/// The content rect of a nested pane — the very rect [`CardFrame::render`]
/// hands back for a [`SurfaceKind::Panel`], built from the same recipe so a hit
/// test cannot disagree with the frame about where a pane's rows start.
fn pane_inner(hugged: Rect, theme: &Theme) -> Rect {
    CardFrame::new(SurfaceKind::Panel, theme)
        .block()
        .inner(hugged)
}

/// The row inside `area` a click at `(column, row)` landed on, or `None` when
/// the point is outside it.
pub(crate) fn hit_row(area: Rect, column: u16, row: u16) -> Option<u16> {
    let inside = column >= area.x
        && column < area.x.saturating_add(area.width)
        && row >= area.y
        && row < area.y.saturating_add(area.height);
    inside.then(|| row.saturating_sub(area.y))
}

/// Columns an executor row is indented under its phase header. Two cells, the
/// same step every other nested list in the TUI uses.
const FLEET_INDENT_COLS: usize = 2;

/// Cells the FLEET tree takes out of a `body` of this width.
///
/// Proportional, not fixed: the old 48-cell executor pane cut
/// `deep-research·easycare-applycard-…` in half on a 300-column terminal while
/// the inspector had a hundred columns it did not need.
pub(crate) fn fleet_width(body_width: u16) -> u16 {
    let room = body_width.saturating_sub(COLUMN_RAIL_WIDTH + INSPECTOR_MIN_WIDTH + 2 * PANE_GUTTER);
    (body_width * FLEET_PERCENT / 100)
        .clamp(FLEET_MIN_WIDTH, FLEET_MAX_WIDTH)
        .min(room.max(FLEET_MIN_WIDTH))
}

/// `▰▰▰▰▰▰▱▱▱▱ 65%` — the run's progress as a bar the eye reads before the
/// number. A percentage alone is a fact; a bar is a glance.
fn gauge_spans(percent: usize, theme: &Theme) -> Vec<Span<'static>> {
    let color = !theme.no_color;
    let filled = (percent.min(100) * GAUGE_CELLS).div_ceil(100).min(GAUGE_CELLS);
    #[allow(
        clippy::cast_precision_loss,
        reason = "percent is 0..=100; f64 represents it exactly"
    )]
    let ratio = percent.min(100) as f64 / 100.0;
    vec![
        Span::styled(
            glyphs::card_gauge_fill(color).repeat(filled),
            Style::new().fg(theme.metric_color(ratio)),
        ),
        Span::styled(
            glyphs::card_gauge_empty(color).repeat(GAUGE_CELLS - filled),
            Style::new().fg(theme.palette.faint),
        ),
        Span::styled(
            format!(" {percent}%"),
            Style::new().fg(theme.palette.fg),
        ),
    ]
}

/// `left … right` — one row with `right` pushed to the far edge of `width`.
///
/// Clipped from the *left* group when the two cannot both fit: the right
/// shoulder is a short caveat or tally that loses all meaning half-drawn,
/// while the left is prose that reads fine truncated.
fn spread_line(
    left: Vec<Span<'static>>,
    right: Vec<Span<'static>>,
    width: u16,
) -> Line<'static> {
    let total = usize::from(width);
    let right_width: usize = right
        .iter()
        .map(|span| crate::tui::text_metrics::display_width(span.content.as_ref()))
        .sum();
    if right.is_empty() || right_width + 2 >= total {
        return Line::from(left);
    }
    let left_line = Line::from(left);
    let room = total - right_width;
    let left_line = crate::tui::text_metrics::truncate_line_to_cells(left_line, room);
    let used = line_width(&left_line);
    let mut spans = left_line.spans;
    spans.push(Span::raw(" ".repeat(room.saturating_sub(used))));
    spans.extend(right);
    Line::from(spans)
}

/// The `│` separating two *panes* — the fleet column from the inspector, the
/// file rail from the diff. Unlike the detail band's inner rail (which is one
/// column's left edge and stops with its content) this one runs the pane's
/// whole height, first row included: the panes it divides start with content,
/// and a rail that skipped row 0 left a notch at the top of the screen.
pub(crate) fn draw_column_rail(frame: &mut Frame<'_>, rail: Rect, theme: &Theme) {
    if rail.width == 0 || rail.height == 0 {
        return;
    }
    let glyph = if theme.no_color {
        glyphs::VERTICAL_SEP_NC
    } else {
        glyphs::VERTICAL_SEP
    };
    let rows: Vec<Line<'static>> = (0..rail.height).map(|_| Line::from(glyph)).collect();
    frame.render_widget(
        Paragraph::new(rows).style(Style::new().fg(theme.palette.border)),
        rail,
    );
}

/// Width of the label column in a detail row. The longest label the viewer
/// emits is `activity` (8 cells); at the old 7 it overflowed and pushed its
/// value out of the column every other row.
const DETAIL_LABEL_COLS: usize = 8;

/// The label column of a detail row: plain `muted`, **never** the DIM-modified
/// `key_hint`.
///
/// `Typography::key_hint` is `dim` *plus* `Modifier::DIM`, which a terminal
/// renders at roughly half intensity — on the Pi palette that turns a `#666666`
/// label into ~`#3a3a3a` over a `#1e1e24` surface (about 1.5:1). Every label in
/// the detail pane was invisible for that reason. `muted` with no modifier is
/// ~4.2:1 on the same surface: quiet, but actually readable.
#[must_use]
pub(crate) fn detail_label_style(theme: &Theme) -> Style {
    Style::new().fg(theme.palette.muted)
}

/// The value column of a detail row: the full body foreground.
///
/// Values are the content; they sit at the top of the contrast ladder
/// (label `muted` → value `fg` → semantic accents for status/failure/tool).
#[must_use]
pub(crate) fn detail_value_style(theme: &Theme) -> Style {
    Style::new().fg(theme.palette.fg)
}

/// Quiet-but-readable secondary prose — the activity history behind the newest
/// entry, list metrics, placeholders. One step below [`detail_value_style`] and
/// still clear of the invisible band.
///
/// Resolves to the same `muted` token as [`detail_label_style`] today; it is a
/// separate role because it answers a different question ("how loud is this
/// content?" vs "how loud is a label?") and a theme may well want to split them.
#[must_use]
fn detail_secondary_style(theme: &Theme) -> Style {
    Style::new().fg(theme.palette.muted)
}

fn detail_line(label: &str, value: &str, label_style: Style, value_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<DETAIL_LABEL_COLS$} "), label_style),
        Span::styled(value.to_string(), value_style),
    ])
}

/// Columns a detail row's value has left after its label in a `width`-cell
/// column.
fn detail_value_room(width: u16) -> usize {
    usize::from(width)
        .saturating_sub(DETAIL_LABEL_COLS + 1)
        .max(12)
}

/// [`detail_line`] clipped to the column it is drawn in — **exactly one row**.
///
/// The card is rendered through `Wrap { trim: false }`, so an over-long value
/// used to continue at column 0 of the next row, under the label grid rather
/// than beside it. Two or three of those in a row and the pane stops reading as
/// a table at all. Claude Code clips a field it cannot fit; so does this.
fn detail_row(
    label: &str,
    value: &str,
    label_style: Style,
    value_style: Style,
    width: u16,
) -> Line<'static> {
    detail_line(
        label,
        &short(value, detail_value_room(width)),
        label_style,
        value_style,
    )
}

fn selected_tally(selected: usize, len: usize) -> String {
    if len == 0 {
        "0/0".to_string()
    } else {
        format!("{}/{}", selected.saturating_add(1).min(len), len)
    }
}

fn next_phase_index(view: &WorkflowView, idx: usize) -> Option<usize> {
    view.phases
        .iter()
        .enumerate()
        .skip(idx + 1)
        .find(|(_, phase)| !phase.is_terminal())
        .map(|(idx, _)| idx)
}

/// Top row of a `height`-row viewport that keeps `selected` visible, moving as
/// little as possible from `current`.
pub(crate) fn visible_offset(current: u16, selected: u16, height: u16) -> u16 {
    if height == 0 || selected < current {
        return selected;
    }
    let bottom = current.saturating_add(height.saturating_sub(1));
    if selected > bottom {
        selected.saturating_sub(height.saturating_sub(1))
    } else {
        current
    }
}

/// `(glyph, style)` for a phase status.
fn phase_status_glyph(status: &str, theme: &Theme, tick: usize) -> (String, Style) {
    match status {
        "done" => ("✓".to_string(), theme.diff_add_style()),
        "running" => (
            SPINNER[if reduce_motion_enabled() {
                0
            } else {
                tick % SPINNER.len()
            }]
            .to_string(),
            Style::new().fg(theme.palette.accent),
        ),
        "resumed" => ("⟲".to_string(), Style::new().fg(theme.palette.muted)),
        _ => ("○".to_string(), Style::new().fg(theme.palette.muted)),
    }
}

/// `(glyph, style)` for an agent status.
/// Status glyph + color for an agent row.
///
/// The Pi status vocabulary: `✓` success, `✗` error, `⊘` warn (deliberately
/// stopped, not a failure), and the live spinner in cyan while running — cyan
/// rather than accent so "in flight" never reads the same as "selected".
fn agent_status_glyph(status: &str, theme: &Theme, tick: usize) -> (String, Style) {
    match status {
        "completed" => (
            "\u{2713}".to_string(),
            Style::new().fg(theme.palette.success),
        ),
        "running" => (
            SPINNER[if reduce_motion_enabled() {
                0
            } else {
                tick % SPINNER.len()
            }]
            .to_string(),
            Style::new().fg(theme.palette.cyan),
        ),
        "failed" => ("\u{2717}".to_string(), Style::new().fg(theme.palette.error)),
        "stopped" => ("\u{2298}".to_string(), Style::new().fg(theme.palette.warn)),
        _ => ("\u{25cb}".to_string(), Style::new().fg(theme.palette.muted)),
    }
}

/// Compact token count: `1234` → `1.2k`, `1_200_000` → `1.2M`. Token counts are
/// small enough that the f64 cast never loses precision.
#[allow(clippy::cast_precision_loss)]
fn fmt_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        format!("{tokens}")
    }
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum()
}

/// Truncate `text` to at most `max` **display columns**, appending `…` when
/// clipped. Width-aware (CJK/wide glyphs count as two columns) so the header
/// description, modal/pane titles, and agent names — which carry user/CJK text,
/// not just ASCII slugs — never overflow their column and clip the status/tally
/// that shares the line.
/// [`short`] for a path: the cut is taken from the **left** (`…/agents/x.md`).
/// A path's tail names the file; right-truncating one keeps only the prefix
/// every agent shares.
pub(crate) fn short_path(path: &str, max: usize) -> String {
    if UnicodeWidthStr::width(path) <= max {
        return path.to_string();
    }
    let budget = max.saturating_sub(1);
    let mut kept: Vec<char> = Vec::new();
    let mut width = 0;
    for ch in path.chars().rev() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > budget {
            break;
        }
        width += w;
        kept.push(ch);
    }
    let mut out = String::from("…");
    out.extend(kept.into_iter().rev());
    out
}

pub(crate) fn short(text: &str, max: usize) -> String {
    if UnicodeWidthStr::width(text) <= max {
        return text.to_string();
    }
    // Reserve one column for the ellipsis.
    let budget = max.saturating_sub(1);
    let mut out = String::new();
    let mut width = 0;
    for ch in text.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > budget {
            break;
        }
        width += w;
        out.push(ch);
    }
    out.push('…');
    out
}

/// [`short`] with the cut taken from the **middle** (`bash · echo "### …/x.tsx"`).
///
/// A tool brief carries its verb at the head and its subject — almost always a
/// path — at the tail, and right-truncating one keeps the prefix every entry
/// shares while throwing away the only part that identifies it.
pub(crate) fn short_middle(text: &str, max: usize) -> String {
    if UnicodeWidthStr::width(text) <= max {
        return text.to_string();
    }
    if max <= 4 {
        return short(text, max);
    }
    // One column for the ellipsis; the remainder leans to the head, which holds
    // the tool name.
    let budget = max.saturating_sub(1);
    let tail_budget = budget / 2;
    let head_budget = budget - tail_budget;
    let (head, rest) = split_at_width(text, head_budget);
    let mut tail_start = rest.len();
    let mut tail_width = 0;
    for (idx, ch) in rest.char_indices().rev() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if tail_width + w > tail_budget {
            break;
        }
        tail_width += w;
        tail_start = idx;
    }
    format!("{head}…{}", &rest[tail_start..])
}

/// Split `text` at the last char boundary that fits `max` display columns,
/// always consuming at least one char so a caller looping on the remainder
/// terminates.
fn split_at_width(text: &str, max: usize) -> (&str, &str) {
    let mut width = 0;
    for (idx, ch) in text.char_indices() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > max {
            if idx == 0 {
                return text.split_at(ch.len_utf8());
            }
            return text.split_at(idx);
        }
        width += w;
    }
    (text, "")
}

/// Greedy word wrap of one source line into rows of at most `room` display
/// columns. Whitespace runs collapse to a single space and a token wider than
/// the column is hard-broken, so a stream with no break opportunity still fills
/// the column instead of being cut off at its first row.
fn wrap_to_width(text: &str, room: usize) -> Vec<String> {
    let room = room.max(2);
    let mut rows: Vec<String> = Vec::new();
    let mut row = String::new();
    let mut row_width = 0usize;
    for word in text.split_whitespace() {
        let mut rest = word;
        loop {
            let rest_width = UnicodeWidthStr::width(rest);
            let gap = usize::from(row_width > 0);
            if row_width + gap + rest_width <= room {
                if gap == 1 {
                    row.push(' ');
                    row_width += 1;
                }
                row.push_str(rest);
                row_width += rest_width;
                break;
            }
            if rest_width <= room || room.saturating_sub(row_width + gap) < 2 {
                // It fits on a row of its own, or there is no useful room left
                // on this one: break the row and retry.
                rows.push(std::mem::take(&mut row));
                row_width = 0;
                continue;
            }
            // Wider than a whole row even alone: fill this one and carry on.
            let (head, tail) = split_at_width(rest, room - row_width - gap);
            if gap == 1 {
                row.push(' ');
            }
            row.push_str(head);
            rows.push(std::mem::take(&mut row));
            row_width = 0;
            rest = tail;
        }
    }
    if !row.is_empty() || rows.is_empty() {
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests;
