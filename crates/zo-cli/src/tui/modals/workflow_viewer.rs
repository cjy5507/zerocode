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
use ratatui::widgets::{Padding, Paragraph, Wrap};

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::hud::{TodoChecklistItem, TodoChecklistStatus};
use std::fmt::Write as _;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::super::spinner::format_elapsed;
use super::super::term::reduce_motion_enabled;
use super::draw_scrollbar;
use super::super::theme::Theme;
use super::super::workflow_progress::{INFLIGHT_AGENT_FRACTION, short_model};

/// Width (cells) of the left Plan rail. Wide enough for a real Todo label while
/// keeping the Executor list dominant in the workspace.
const PLAN_RAIL_WIDTH: u16 = 40;
const EXECUTOR_PANE_WIDTH: u16 = 48;
const DETAIL_PANE_MIN_WIDTH: u16 = 40;
const MEDIUM_PANE_MIN_WIDTH: u16 = 44;
const WIDE_LAYOUT_MIN_WIDTH: u16 =
    PLAN_RAIL_WIDTH + EXECUTOR_PANE_WIDTH + DETAIL_PANE_MIN_WIDTH;
const MEDIUM_LAYOUT_MIN_WIDTH: u16 = PLAN_RAIL_WIDTH + MEDIUM_PANE_MIN_WIDTH;

/// Spinner frames for a running agent/phase. Advanced once per [`refresh`] so
/// the "is anything happening?" signal stays alive between polls.
const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

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
                && !step.active_form.trim().is_empty()
            {
                return &step.active_form;
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
    /// Selected phase (left rail cursor).
    selected_phase: usize,
    /// Scroll offset into the selected phase's agent list (right pane).
    agent_scroll: u16,
    /// Selected agent row inside the selected phase.
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
            agent_scroll: 0,
            selected_agent,
            tick: 0,
            events_mode: false,
            events: Vec::new(),
            events_scroll: 0,
            output_tail: None,
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
        const TAIL_LINES: usize = 40;
        const TAIL_BYTES: u64 = 16 * 1024;
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
            self.agent_scroll = 0;
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
                self.agent_scroll = 0;
                return true;
            }
        }
        false
    }

    fn select_prev_phase(&mut self) {
        if self.selected_phase > 0 {
            self.selected_phase -= 1;
            self.agent_scroll = 0;
            self.selected_agent = 0;
        }
    }

    fn select_next_phase(&mut self) {
        if self.selected_phase + 1 < self.view.phases.len() {
            self.selected_phase += 1;
            self.agent_scroll = 0;
            self.selected_agent = 0;
        }
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

    fn select_prev_agent(&mut self, rows: usize) {
        self.selected_agent = self.selected_agent.saturating_sub(rows);
    }

    fn select_next_agent(&mut self, rows: usize) {
        let max_agent = self
            .selected_phase_row()
            .map_or(0, |phase| phase.agents.len().saturating_sub(1));
        self.selected_agent = self.selected_agent.saturating_add(rows).min(max_agent);
    }

    /// Scroll the agent pane up by `rows` (mouse wheel). The offset is clamped to
    /// the content height at draw time, so an unbounded add is safe.
    pub fn scroll_agents_up(&mut self, rows: u16) {
        if self.events_mode {
            self.events_scroll = self.events_scroll.saturating_sub(rows);
            return;
        }
        self.agent_scroll = self.agent_scroll.saturating_sub(rows);
        self.select_prev_agent(usize::from(rows));
    }

    /// Scroll the agent pane down by `rows` (mouse wheel).
    pub fn scroll_agents_down(&mut self, rows: u16) {
        if self.events_mode {
            self.events_scroll =
                self.clamp_events_scroll(self.events_scroll.saturating_add(rows));
            return;
        }
        self.agent_scroll = self.agent_scroll.saturating_add(rows);
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
            KeyCode::Home | KeyCode::Char('g') => {
                self.selected_agent = 0;
                self.agent_scroll = 0;
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.selected_agent = self
                    .selected_phase_row()
                    .map_or(0, |phase| phase.agents.len().saturating_sub(1));
            }
            KeyCode::PageUp => self.select_prev_agent(10),
            KeyCode::PageDown => self.select_next_agent(10),
            _ => {}
        }
        None
    }

    /// Draw the modal into `area`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let scope = match self.view.scope() {
            ViewerScope::Plan => "Plan → Executors",
            ViewerScope::Workflow => "Workflow → Executors",
            ViewerScope::Run => "Run → Executors",
        };
        let run_name = if self.view.name == "agents" {
            "spawned agents"
        } else {
            self.view.name.as_str()
        };
        let title = format!(" {scope} · {} ", short(run_name, 42));
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(title, theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }

        let [header_area, body_area, footer_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        frame.render_widget(
            Paragraph::new(vec![
                self.workflow_path_line(theme, header_area.width),
                self.header_line(theme),
            ]),
            header_area,
        );

        if self.events_mode {
            self.draw_event_log(frame, body_area, theme);
        } else if body_area.height < 10 {
            self.draw_compact_body(frame, body_area, theme);
        } else if body_area.width >= WIDE_LAYOUT_MIN_WIDTH {
            let [rail_area, agent_area, detail_area] = Layout::horizontal([
                Constraint::Length(PLAN_RAIL_WIDTH),
                Constraint::Length(EXECUTOR_PANE_WIDTH),
                Constraint::Min(1),
            ])
            .areas(body_area);
            self.draw_phase_rail(frame, rail_area, theme);
            self.draw_agent_pane(frame, agent_area, theme);
            self.draw_agent_detail(frame, detail_area, theme);
        } else if body_area.width >= MEDIUM_LAYOUT_MIN_WIDTH {
            let [rail_area, pane_area] =
                Layout::horizontal([Constraint::Length(PLAN_RAIL_WIDTH), Constraint::Min(1)])
                    .areas(body_area);
            let detail_height = (pane_area.height.saturating_mul(45) / 100)
                .max(6)
                .min(pane_area.height.saturating_sub(3));
            let [agent_area, detail_area] = Layout::vertical([
                Constraint::Length(pane_area.height.saturating_sub(detail_height)),
                Constraint::Min(0),
            ])
            .areas(pane_area);
            self.draw_phase_rail(frame, rail_area, theme);
            self.draw_agent_pane(frame, agent_area, theme);
            self.draw_agent_detail(frame, detail_area, theme);
        } else {
            let wanted_plan = u16::try_from(self.view.phases.len().saturating_add(2))
                .unwrap_or(u16::MAX)
                .clamp(3, 6);
            let plan_height = wanted_plan.min(body_area.height.saturating_sub(6).max(3));
            let remaining = body_area.height.saturating_sub(plan_height);
            if remaining < 9 {
                self.draw_compact_body(frame, body_area, theme);
            } else {
                // Reserve enough detail rows for flow/status/tool/metrics; the
                // Executor list absorbs the remaining height.
                let detail_height = (remaining.saturating_mul(2) / 5)
                    .max(6)
                    .min(remaining.saturating_sub(3));
                let agent_height = remaining.saturating_sub(detail_height);
                let [rail_area, agent_area, detail_area] = Layout::vertical([
                    Constraint::Length(plan_height),
                    Constraint::Length(agent_height),
                    Constraint::Min(0),
                ])
                .areas(body_area);
                self.draw_phase_rail(frame, rail_area, theme);
                self.draw_agent_pane(frame, agent_area, theme);
                self.draw_agent_detail(frame, detail_area, theme);
            }
        }

        frame.render_widget(
            Paragraph::new(footer_line(
                theme,
                footer_area.width,
                self.events_mode,
                !self.view.run_id.is_empty(),
            )),
            footer_area,
        );
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
            Paragraph::new(Line::styled(
                format!(
                    "event log · {} events · ^E: back · \u{2191}\u{2193} scroll",
                    self.events.len()
                ),
                theme.typography.dim,
            )),
            title_area,
        );
        let lines: Vec<Line<'static>> = if self.events.is_empty() {
            vec![Line::styled(
                "no events recorded for this run",
                theme.typography.dim,
            )]
        } else {
            self.events
                .iter()
                .map(|line| Line::raw(line.clone()))
                .collect()
        };
        frame.render_widget(
            Paragraph::new(lines)
                .scroll((self.events_scroll, 0))
                .style(theme.typography.body),
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
                theme.typography.key_hint,
                theme.typography.body,
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
                    theme.typography.key_hint,
                    agent_status_glyph(&agent.status, theme, self.tick).1,
                ));
                let activity = agent
                    .current_tool
                    .as_deref()
                    .or(agent.current_phase.as_deref())
                    .unwrap_or("waiting for activity");
                lines.push(detail_line(
                    "Activity",
                    &short(activity, text_width),
                    theme.typography.key_hint,
                    theme.typography.dim,
                ));
            } else {
                lines.push(detail_line(
                    "Executor",
                    "none started",
                    theme.typography.key_hint,
                    theme.typography.dim,
                ));
            }
        } else {
            lines.push(Line::styled("no workflow scope", theme.typography.dim));
        }
        frame.render_widget(Paragraph::new(lines), area);
    }

    /// The selected relationship is the visual anchor: users should understand
    /// the workspace by scanning this line without reading panel contents.
    fn workflow_path_line(&self, theme: &Theme, width: u16) -> Line<'static> {
        let scope = self.view.scope();
        let scope_label = match scope {
            ViewerScope::Plan => "PLAN",
            ViewerScope::Workflow => "WORKFLOW",
            ViewerScope::Run => "RUN SCOPE",
        };
        if width < 72 {
            return Line::from(vec![
                Span::styled(
                    scope_label,
                    Style::new()
                        .fg(theme.palette.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  →  ", theme.typography.dim),
                Span::styled("EXECUTORS", theme.typography.body),
                Span::styled("  →  ", theme.typography.dim),
                Span::styled("DETAIL", theme.typography.body),
            ]);
        }

        let phase_context = self.selected_phase_row().map_or_else(
            || "0/0".to_string(),
            |phase| {
                match phase.scope() {
                    ViewerScope::Plan => {
                        let link = if scope == ViewerScope::Workflow {
                            " · Plan linked"
                        } else {
                            ""
                        };
                        format!(
                            "phase {}/{} · {}{link}",
                            self.selected_phase + 1,
                            self.view.phases.len(),
                            short(phase.plan_label(true), 24)
                        )
                    }
                    ViewerScope::Workflow => format!(
                        "{}/{} · {} · Plan unlinked",
                        self.selected_phase + 1,
                        self.view.phases.len(),
                        short(phase.plan_label(false), 18)
                    ),
                    ViewerScope::Run => "unlinked fan-out".to_string(),
                }
            },
        );
        let executor_context = self.selected_phase_row().map_or_else(
            || "0/0".to_string(),
            |phase| selected_tally(self.selected_agent, phase.agents.len()),
        );
        let detail_context = self
            .selected_agent_row()
            .map_or_else(|| "none".to_string(), |agent| short(&agent.name, 24));
        Line::from(vec![
            Span::styled(
                scope_label,
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {phase_context}"), theme.typography.body),
            Span::styled("  →  ", theme.typography.dim),
            Span::styled("EXECUTORS", theme.typography.body),
            Span::styled(format!(" {executor_context}"), theme.typography.dim),
            Span::styled("  →  ", theme.typography.dim),
            Span::styled("DETAIL", theme.typography.body),
            Span::styled(format!(" {detail_context}"), theme.typography.dim),
        ])
    }

    /// Header: run status and outcomes, with scope honesty for synthetic runs.
    fn header_line(&self, theme: &Theme) -> Line<'static> {
        let dim = theme.typography.dim;
        let mut spans = Vec::new();
        let status_label = if self.view.synthesizing {
            "synthesizing".to_string()
        } else {
            self.view.status.clone()
        };
        spans.push(Span::styled(
            status_label,
            Style::new().fg(theme.palette.accent),
        ));
        spans.push(Span::styled(
            format!(
                "  ·  {}% done  ·  {} active / {} executors",
                self.view.progress_percent(),
                self.view.running_agents(),
                self.view.total_agents()
            ),
            dim,
        ));
        let failed = self.view.failed_agents();
        if failed > 0 {
            spans.push(Span::styled(
                format!("  ·  {failed} failed"),
                theme.diff_del_style(),
            ));
        }
        if self.view.scope() == ViewerScope::Plan {
            if let Some(idx) = self.view.active_phase_index() {
                if let Some(next_idx) = next_phase_index(&self.view, idx) {
                    if let Some(next) = self.view.phases.get(next_idx) {
                        spans.push(Span::styled("  ·  next ", dim));
                        spans.push(Span::styled(
                            short(next.plan_label(false), 24),
                            theme.typography.body,
                        ));
                    }
                }
            }
        } else {
            let warning = match self.view.scope() {
                ViewerScope::Workflow if self.view.plan_link_count() > 0 => format!(
                    "  ·  {}/{} phases Plan linked",
                    self.view.plan_link_count(),
                    self.view.phases.len()
                ),
                ViewerScope::Workflow => "  ·  Plan link unavailable".to_string(),
                ViewerScope::Run => "  ·  not linked to a Plan step".to_string(),
                ViewerScope::Plan => unreachable!(),
            };
            spans.push(Span::styled(warning, Style::new().fg(theme.palette.warn)));
        }
        Line::from(spans)
    }

    fn draw_phase_rail(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let title = match self.view.scope() {
            ViewerScope::Plan => " Plan steps ".to_string(),
            ViewerScope::Workflow if self.view.plan_link_count() > 0 => format!(
                " Workflow steps · {}/{} Plan ",
                self.view.plan_link_count(),
                self.view.phases.len()
            ),
            ViewerScope::Workflow => " Workflow steps · Plan unlinked ".to_string(),
            ViewerScope::Run => " Run scope · unlinked ".to_string(),
        };
        let inner = CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, theme.typography.dim))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let label_width = usize::from(inner.width.saturating_sub(12)).max(8);
        let lines: Vec<Line<'static>> = self
            .view
            .phases
            .iter()
            .enumerate()
            .map(|(idx, phase)| self.phase_rail_line(idx, phase, label_width, theme))
            .collect();
        let max_scroll = u16::try_from(lines.len())
            .unwrap_or(u16::MAX)
            .saturating_sub(inner.height);
        let selected = u16::try_from(self.selected_phase).unwrap_or(u16::MAX);
        let offset = visible_offset(0, selected, inner.height).min(max_scroll);
        frame.render_widget(Paragraph::new(lines).scroll((offset, 0)), inner);
        draw_scrollbar(frame, inner, offset, self.view.phases.len(), theme);
    }

    fn phase_rail_line(
        &self,
        idx: usize,
        phase: &WorkflowPhaseRow,
        label_width: usize,
        theme: &Theme,
    ) -> Line<'static> {
        let selected = idx == self.selected_phase;
        let caret = if selected { "›" } else { " " };
        let (icon, icon_style) = phase_status_glyph(&phase.status, theme, self.tick);
        let label_style = if selected {
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            theme.typography.body
        };
        // completed/total, live: `completed_now` counts finished rows while the
        // phase runs (the recorded `completed` is 0 until the barrier), so the
        // tally climbs as agents finish instead of jumping from 0/N to N/N.
        let tally = if phase.total > 0 {
            format!("{}/{}", phase.completed_now(), phase.total)
        } else {
            "queued".to_string()
        };
        Line::from(vec![
            Span::styled(format!("{caret} "), label_style),
            Span::styled(format!("{icon} "), icon_style),
            Span::styled(short(phase.plan_label(false), label_width), label_style),
            Span::raw("  "),
            Span::styled(tally, theme.typography.dim),
        ])
    }

    fn draw_agent_pane(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let phase = self.view.phases.get(self.selected_phase);
        let title = phase.map_or_else(
            || " Executors ".to_string(),
            |p| {
                let scope = match p.scope() {
                    ViewerScope::Plan => {
                        format!(
                            "workflow phase {}/{} · Plan linked",
                            self.selected_phase + 1,
                            self.view.phases.len()
                        )
                    }
                    ViewerScope::Workflow => format!(
                        "workflow {}/{} · Plan unlinked",
                        self.selected_phase + 1,
                        self.view.phases.len()
                    ),
                    ViewerScope::Run => "run-level".to_string(),
                };
                let raw = format!(
                    " Executors · {scope} · {} · {} ",
                    selected_tally(self.selected_agent, p.agents.len()),
                    p.plan_label(false)
                );
                short(&raw, usize::from(area.width.saturating_sub(2)))
            },
        );
        let inner = CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, theme.typography.dim))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let Some(phase) = phase else {
            return;
        };
        if phase.agents.is_empty() {
            let note = match phase.status.as_str() {
                "pending" => "queued — no executor started yet",
                "resumed" => "resumed from cache (no executor spawned)",
                _ => "no executors",
            };
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(note, theme.typography.dim))),
                inner,
            );
            return;
        }
        let lines: Vec<Line<'static>> = phase
            .agents
            .iter()
            .enumerate()
            .map(|(idx, agent)| self.agent_line(idx, agent, theme))
            .collect();
        // Clamp the scroll to the content: `Paragraph::scroll` does not clamp, so
        // PageDown past the end would otherwise push every row off the top into
        // blank space. Cosmetic (per-frame, `draw` is `&self`) but it self-heals
        // every redraw.
        let max_scroll = u16::try_from(lines.len())
            .unwrap_or(u16::MAX)
            .saturating_sub(inner.height);
        let selected = u16::try_from(self.selected_agent).unwrap_or(u16::MAX);
        let offset = visible_offset(self.agent_scroll.min(max_scroll), selected, inner.height)
            .min(max_scroll);
        frame.render_widget(Paragraph::new(lines).scroll((offset, 0)), inner);
        draw_scrollbar(frame, inner, offset, phase.agents.len(), theme);
    }

    fn agent_line(&self, idx: usize, agent: &WorkflowAgentRow, theme: &Theme) -> Line<'static> {
        agent_list_line(agent, idx == self.selected_agent, theme, self.tick)
    }

    fn draw_agent_detail(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let title = self.selected_agent_row().map_or_else(
            || " Executor detail ".to_string(),
            |agent| format!(" Executor · {} ", short(&agent.name, 32)),
        );
        let inner = CardFrame::new(SurfaceKind::Panel, theme)
            .title(Line::styled(title, theme.typography.dim))
            .render(frame, area);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let Some(agent) = self.selected_agent_row() else {
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    "select an executor",
                    theme.typography.dim,
                ))),
                inner,
            );
            return;
        };
        frame.render_widget(
            Paragraph::new(self.agent_detail_lines(agent, theme)).wrap(Wrap { trim: false }),
            inner,
        );
    }

    fn agent_detail_lines(&self, agent: &WorkflowAgentRow, theme: &Theme) -> Vec<Line<'static>> {
        let dim = theme.typography.dim;
        let key = theme.typography.key_hint;
        let mut lines = Vec::new();
        if let Some(phase) = self.selected_phase_row() {
            lines.push(detail_line(
                "flow",
                &executor_flow_text(&self.view, self.selected_phase, phase, agent),
                key,
                theme.typography.body,
            ));
        }
        lines.extend(agent_detail_body_lines(agent, theme));
        if let Some(phase) = self.selected_phase_row() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "scope context".to_string(),
                theme.typography.heading_2,
            )));
            lines.push(detail_line(
                "scope",
                &scope_detail_text(&self.view, self.selected_phase, phase),
                key,
                theme.typography.body,
            ));
            lines.push(detail_line(
                "phase",
                &phase_detail_text(phase),
                key,
                dim,
            ));
            lines.push(detail_line(
                "step",
                &phase_status_text(phase),
                key,
                phase_status_style(&phase.status, theme),
            ));
        }

        // Once the agent lands its markdown result, show its tail right here
        // (refreshed mtime-gated by the host tick, never read at draw time).
        if let Some((path, _, tail)) = &self.output_tail {
            if agent.output_file.as_deref() == Some(path.as_str()) && !tail.is_empty() {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "output tail".to_string(),
                    theme.typography.heading_2,
                )));
                for row in tail {
                    lines.push(Line::from(Span::styled(
                        format!("  {row}"),
                        theme.typography.body,
                    )));
                }
            }
        }
        lines
    }
}

/// One agent row for a selection list: status glyph, name, model, metrics and
/// the live tool/phase arrow. Shared by the workflow viewer's agent pane and
/// the Ctrl+G agents viewer so a fleet reads identically in both.
pub(crate) fn agent_list_line(
    agent: &WorkflowAgentRow,
    selected: bool,
    theme: &Theme,
    tick: usize,
) -> Line<'static> {
    let dim = theme.typography.dim;
    let (icon, icon_style) = agent_status_glyph(&agent.status, theme, tick);
    let label_style = if selected {
        Style::new()
            .fg(theme.palette.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        theme.typography.body
    };
    let mut spans = vec![
        Span::styled(if selected { "› " } else { "  " }, label_style),
        Span::styled(format!("{icon} "), icon_style),
        Span::styled(short(&agent.name, 24), label_style),
    ];
    let model_label = short_model(&agent.model);
    if !model_label.is_empty() {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(model_label, dim));
    }
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
    spans.push(Span::styled(metrics, dim));
    // Live activity for a running agent: the concrete tool if one is running,
    // otherwise the transient phase (e.g. `thinking`) so a between-tools agent
    // still reads as *doing something* instead of looking idle. Phase is shown
    // in the warn tone and only when no tool is active (the writer clears the
    // phase on tool start, so gating on `current_tool.is_none()` suppresses
    // flicker).
    if agent.status == "running" {
        if let Some(tool) = &agent.current_tool {
            spans.push(Span::styled(
                format!("  ⟶ {}", short(tool, 18)),
                Style::new().fg(theme.palette.accent),
            ));
        } else if let Some(phase) = &agent.current_phase {
            spans.push(Span::styled(
                format!("  ⟶ {}", short(phase, 18)),
                Style::new().fg(theme.palette.warn),
            ));
        }
    }
    Line::from(spans)
}

/// The agent-only detail card body. Status, failures, live activity, and
/// metrics lead so constrained panes still answer "what is this executor
/// doing, or why did it stop?" before the longer identity/task metadata. Shared by the
/// workflow viewer's detail pane (which prepends phase context and appends the
/// landed-file tail) and the Ctrl+G agents viewer.
#[allow(clippy::too_many_lines)] // one detail card: rows assembled in order
pub(crate) fn agent_detail_body_lines(
    agent: &WorkflowAgentRow,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let dim = theme.typography.dim;
    let key = theme.typography.key_hint;
    let mut lines = vec![detail_line(
        "status",
        &agent.status,
        key,
        agent_status_glyph(&agent.status, theme, 0).1,
    )];
    if let Some(error) = &agent.error {
        lines.push(detail_line(
            "error",
            &short(error, 220),
            key,
            theme.diff_del_style(),
        ));
    }
    if let Some(blocker) = &agent.blocker {
        lines.push(detail_line(
            "blocker",
            &short(blocker, 220),
            key,
            theme.diff_del_style(),
        ));
    }
    if let Some(tool) = &agent.current_tool {
        lines.push(detail_line(
            "tool",
            tool,
            key,
            Style::new().fg(theme.palette.accent),
        ));
    } else if let Some(phase) = &agent.current_phase {
        lines.push(detail_line(
            "activity",
            phase,
            key,
            Style::new().fg(theme.palette.warn),
        ));
    }
    let elapsed = format_elapsed(agent.elapsed_secs);
    let metrics = match (agent.tokens > 0, agent.tool_calls) {
        (true, Some(tool_calls)) => {
            format!(
                "{} · {tool_calls} tools · {elapsed}",
                fmt_tokens(agent.tokens)
            )
        }
        (true, None) => format!("{} · {elapsed}", fmt_tokens(agent.tokens)),
        (false, Some(tool_calls)) => format!("{tool_calls} tools · {elapsed}"),
        (false, None) => elapsed,
    };
    let metrics = match (agent.status == "running", agent.idle_secs) {
        // Heartbeat: how long since the agent last wrote its manifest. A
        // climbing value on a "running" agent is the signal it has stalled.
        (true, Some(idle)) => format!("{metrics} · active {idle}s ago"),
        _ => metrics,
    };
    lines.push(detail_line("metrics", &metrics, key, dim));
    if let Some(path) = &agent.output_file {
        lines.push(detail_line("output", path, key, dim));
    }
    lines.push(detail_line("id", &agent.id, key, dim));
    if !agent.description.is_empty() {
        lines.push(detail_line(
            "task",
            &short(&agent.description, 180),
            key,
            theme.typography.body,
        ));
    }
    if let Some(kind) = &agent.subagent_type {
        lines.push(detail_line("type", kind, key, dim));
    }
    let model_label = short_model(&agent.model);
    if !model_label.is_empty() {
        lines.push(detail_line("model", &model_label, key, dim));
    }
    if let Some(reason) = &agent.route_reason {
        lines.push(detail_line("route", &short(reason, 220), key, dim));
    }
    if let Some(event) = &agent.last_event {
        lines.push(detail_line("event", &short(event, 220), key, dim));
    }

    // Live activity transcript: the manifest's rolling `recentTools` feed
    // (oldest → newest). The newest entry is the work happening right now
    // for a running agent, so it carries the accent.
    if !agent.recent_tools.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "activity".to_string(),
            theme.typography.heading_2,
        )));
        let running = agent.status == "running";
        let newest = agent.recent_tools.len().saturating_sub(1);
        for (i, entry) in agent.recent_tools.iter().enumerate() {
            let style = if running && i == newest {
                Style::new().fg(theme.palette.accent)
            } else {
                dim
            };
            lines.push(Line::from(Span::styled(
                format!("  \u{00b7} {}", short(entry, 200)),
                style,
            )));
        }
    }

    // Live streamed prose — *what the agent is saying right now*. The manifest
    // carries a rolling `outputTail` while the agent runs (the finished-file
    // tail is `None` until it lands), so this is the only window into a
    // running agent's actual output. Mutually exclusive with the finished-file
    // section: a live row has `output_tail`, a landed row has the file.
    if agent.status == "running" {
        if let Some(tail) = &agent.output_tail {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "live output".to_string(),
                theme.typography.heading_2,
            )));
            for row in tail.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", short(row, 200)),
                    theme.typography.body,
                )));
            }
        }
    }
    lines
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
        let full = super::key_hint_footer(
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
        return super::key_hint_footer(theme, &[("Esc", "close")]);
    }

    let mut full_hints = vec![
        ("↑/↓", "executor"),
        ("←/→", "scope"),
        ("PgUp/PgDn", "page"),
    ];
    if events_available {
        full_hints.push(("^E", "events"));
    }
    full_hints.push(("Esc", "close"));
    let full = super::key_hint_footer(theme, &full_hints);
    if line_width(&full) <= usize::from(width) {
        return full;
    }

    let compact = super::key_hint_footer_with_separator(
        theme,
        &[
            ("↑/↓", "executor"),
            ("←/→", "scope"),
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
    super::key_hint_footer(theme, &[("Esc", "close")])
}

fn detail_line(label: &str, value: &str, label_style: Style, value_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<7} "), label_style),
        Span::styled(value.to_string(), value_style),
    ])
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

fn scope_detail_text(view: &WorkflowView, idx: usize, phase: &WorkflowPhaseRow) -> String {
    match phase.scope() {
        ViewerScope::Plan => format!("Plan linked · {}", phase.plan_label(false)),
        ViewerScope::Workflow => format!(
            "Plan unlinked · workflow phase {}/{} · {}",
            idx + 1,
            view.phases.len(),
            phase.plan_label(false)
        ),
        ViewerScope::Run => "Plan unlinked · Run-level fan-out".to_string(),
    }
}

fn phase_detail_text(phase: &WorkflowPhaseRow) -> String {
    let mut text = format!("{} · {}", phase.id, phase.kind);
    if phase.round > 0 {
        let _ = write!(text, " · round {}", phase.round);
    }
    text
}

fn executor_flow_text(
    view: &WorkflowView,
    idx: usize,
    phase: &WorkflowPhaseRow,
    agent: &WorkflowAgentRow,
) -> String {
    match phase.scope() {
        ViewerScope::Plan => format!("plan step -> {}", agent.name),
        ViewerScope::Workflow => format!(
            "workflow phase {}/{} -> {}",
            idx + 1,
            view.phases.len(),
            agent.name
        ),
        ViewerScope::Run => format!("run-level -> {}", agent.name),
    }
}

fn phase_status_text(phase: &WorkflowPhaseRow) -> String {
    if phase.total == 0 {
        return phase.status.clone();
    }
    format!(
        "{} · {}% done · {} done / {} running / {} failed / {} total",
        phase.status,
        phase.progress_percent(),
        phase.completed_now(),
        phase.running_now(),
        phase.failed_now(),
        phase.total
    )
}

fn phase_status_style(status: &str, theme: &Theme) -> Style {
    match status {
        "done" => theme.diff_add_style(),
        "running" => Style::new().fg(theme.palette.accent),
        "failed" | "cancelled" | "budget_exhausted" => theme.diff_del_style(),
        _ => theme.typography.dim,
    }
}

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
        "resumed" => ("⟲".to_string(), theme.typography.dim),
        _ => ("○".to_string(), theme.typography.dim),
    }
}

/// `(glyph, style)` for an agent status.
fn agent_status_glyph(status: &str, theme: &Theme, tick: usize) -> (String, Style) {
    match status {
        "completed" => ("✓".to_string(), theme.diff_add_style()),
        "running" => (
            SPINNER[if reduce_motion_enabled() {
                0
            } else {
                tick % SPINNER.len()
            }]
            .to_string(),
            Style::new().fg(theme.palette.accent),
        ),
        "failed" => ("✗".to_string(), theme.diff_del_style()),
        "stopped" => ("⊘".to_string(), theme.typography.dim),
        _ => ("○".to_string(), theme.typography.dim),
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

#[cfg(test)]
mod tests;
