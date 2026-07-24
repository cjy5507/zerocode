//! `App` — top-level TUI state and event loop.
//!
//! Owns the [`AppMode`] state machine, the [`Theme`], and the
//! [`AppEndpoint`]-shaped channel handles (render-block receiver +
//! agent-command sender). Drives the draw loop via
//! `crossterm::event::EventStream` + `tokio::select!` so terminal
//! events and streamed render blocks interleave cleanly.
//!
//! The transcript, HUD, and modal widgets in this module render the
//! shipped TUI surface. Channel wiring still lives outside this module,
//! but the draw path itself should describe actual runtime behavior
//! rather than staged lane placeholders.
//!
//! See `code-rules.md` R1 (render-block only), R2 (no ANSI), R8
//! (bounded channels), R9 (&Theme for all styling).

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use api::{ProviderKind, detect_provider_kind, provider_catalog, resolve_model_alias};
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use runtime::PermissionMode as RuntimePermissionMode;
use runtime::message_stream::BlockId;
use runtime::message_stream::{PermissionPrompt, RenderBlock, SystemLevel, UserQuestionPrompt};
use tokio::sync::{mpsc, oneshot};

use super::TuiError;
use super::blocks::tool_call;
use super::command_history::CommandHistory;
use super::history::History;
use super::heat::HeatState;
use super::hud::{self, HudState, PermissionMode};
use super::image_protocol::ImageProtocol;
use super::inline::{FinalizedTranscriptQueue, TerminalMode};
use super::term::TermProfile;
use super::input::{InputCommand, InputWidget};
use super::layout::LayoutRegions;
use super::modals::{
    AgentsViewerAction, AgentsViewerModal, ApiKeyConnectInfo, ApiKeyModal, ChoicePickerModal,
    CustomProviderWizardModal, DeepTierModal, DeepTierView, DiffViewerAction, DiffViewerModal,
    Effort, EffortPickerModal, FilePickerModal, Modal,
    ModalResult, ModalSelection, ModelPickerEntry, ModelPickerModal, PermissionPickerModal,
    ReportViewerBlock, ReportViewerModal,
    RewindViewerAction, RewindViewerModal, ReviewModal, SmartSettingsModal, TeamInboxViewerModal,
    ToolToggleModal, ToolToggleRow, UsageDashboardAction, UsageDashboardModal, UserQuestionModal,
    WorkflowView, WorkflowViewerAction, WorkflowViewerModal, RemoteOnboardingModal,
    RemoteOnboardingView,
};
use super::sidebar::{ChangedFile, GitStatusSnapshot, SidebarState};
use super::spinner::TurnActivity;
use super::startup::StartupScreen;
use super::theme::Theme;
use super::transcript::Transcript;

pub type RenderObserver = Arc<dyn Fn(&RenderBlock) + Send + Sync>;

mod agent_batch;
mod defaults;
mod history_nav;
mod keys;
mod mention_hint;
mod modal_geometry;
mod modals;
mod queue;
mod render;
mod run_loop;
mod search;
mod slash_hint;
mod stream_pace;
mod types;

#[cfg(test)]
mod tests;

pub use self::types::{
    AgentCommand, AgentResultMeta, AppAction, AppMode, ClipboardCopyTarget, ImageAttachment,
    QueueLimitError, QueuedMessage, ScheduledWakeHud, TranscriptViewRequest, WakeSource,
};

use self::defaults::default_hud_state;
use self::mention_hint::{apply_mention, draw_mention_hint, mention_trigger};
use self::modal_geometry::{
    anchored_modal_rect, centered_modal_rect, diff_modal_rect, effort_modal_rect,
    palette_modal_rect,
};
use self::slash_hint::{draw_slash_hint, slash_completion_for};

/// Session-installed closure polled each tick for the custom `statusLine`
/// command output (see [`App::set_status_line_poller`]).
pub type StatusLinePoller = Box<dyn Fn(&crate::tui::hud::HudState) -> Option<String>>;

/// Session-installed event-gated wakeup-file scanner. Outer `None` means no
/// scan is due; `Some(None)` means a due scan found no scheduled wakeup.
pub type ScheduledWakePoller = Box<dyn Fn() -> Option<Option<ScheduledWakeHud>>>;

/// Session-installed event-gated workspace status spawner.
pub type WorkspaceStatusPoller = Box<
    dyn Fn(&Path) -> Option<tokio::task::JoinHandle<Option<GitStatusSnapshot>>>,
>;

/// Session-installed provider of freshly-encoded MCP HUD rows (the
/// [`crate::tui::hud::McpHudStatus`] encoding). `None` = the session has no
/// MCP runtime state to read (keep the last synced rows).
pub type McpStatusPoller = Box<dyn Fn() -> Option<Vec<String>>>;

type ScanCancelToken = Arc<AtomicBool>;

/// Verdict sent after the App considers a spectator replacement. A stale
/// replacement is acknowledged without clearing the newer local transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckVerdict {
    Applied,
    Stale,
}

/// One ordered spectator ingress. A replacement is acknowledged only after it
/// is accepted or rejected against the monotonic spectator floor.
pub enum SpectatorEvent {
    Frame { frame_seq: u64, block: RenderBlock },
    Replace {
        blocks: Vec<RenderBlock>,
        post_boundary: VecDeque<RenderBlock>,
        next_seq: u64,
        ack: oneshot::Sender<AckVerdict>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HoveredCopyButton {
    block_id: BlockId,
    button: Rect,
}

/// Transcript block under a left-button press, retained only so a release on
/// the same block can preserve the existing click behavior. Character-level
/// drag-selection state lives in [`Transcript`] because drawing owns the cells
/// that are highlighted and copied.
#[derive(Debug, Clone, Copy)]
struct TranscriptPress {
    block: Option<BlockId>,
}

/// Transcript viewport interaction state, grouped off [`App`] (god-struct
/// decomposition). These fields all describe how the user is currently
/// interacting with the transcript surface rather than the transcript contents
/// themselves: follow-tail behavior, scrollbar dragging, copy-button hover,
/// and drag selection. Keeping them together makes the next App split easier
/// without moving rendering or input behavior yet.
struct TranscriptViewState {
    /// Whether new output should keep snapping the transcript to the tail.
    follow_output: bool,
    /// `true` between a left-button press on the transcript scrollbar and its
    /// release, so drag events scroll the view.
    scrollbar_dragging: bool,
    /// Copy affordance currently shown for the block under the mouse.
    hovered_copy_button: Option<HoveredCopyButton>,
    /// Block under the current transcript press, for click semantics on release.
    transcript_press: Option<TranscriptPress>,
}

impl Default for TranscriptViewState {
    fn default() -> Self {
        Self {
            follow_output: true,
            scrollbar_dragging: false,
            hovered_copy_button: None,
            transcript_press: None,
        }
    }
}

/// Startup launchpad state, grouped off [`App`] (god-struct decomposition). The
/// launchpad `screen` and the `shown_at` instant share one lifetime when shown
/// or dismissed together, so keeping them behind a few intent-revealing methods
/// (`show`, `dismiss`, `intro_elapsed`) removes the "set both / clear both"
/// duplication that was previously inlined at every call site.
#[derive(Default)]
struct StartupState {
    /// Startup launchpad shown before the first interactive turn. `None` once
    /// the launchpad is dismissed.
    screen: Option<StartupScreen>,
    /// When the startup launchpad first appeared. Static launchpads use this
    /// only for API compatibility; animated variants can use it to keep the idle
    /// render loop awake until their intro settles. `None` once dismissed.
    shown_at: Option<Instant>,
}

impl StartupState {
    /// Show `screen` and start the launchpad intro/repaint clock.
    fn show(&mut self, screen: StartupScreen) {
        self.screen = Some(screen);
        self.shown_at = Some(Instant::now());
    }

    /// Dismiss the launchpad and stop the intro/repaint clock.
    fn dismiss(&mut self) {
        self.screen = None;
        self.shown_at = None;
    }

    /// Elapsed time since the launchpad appeared, or `None` when it never has.
    fn intro_elapsed(&self) -> Option<Duration> {
        self.shown_at.map(|shown| shown.elapsed())
    }
}

fn new_scan_cancel_token() -> ScanCancelToken {
    Arc::new(AtomicBool::new(false))
}

fn scan_cancelled(cancel: &ScanCancelToken) -> bool {
    cancel.load(Ordering::Relaxed)
}

/// Background directory-scan handles, grouped off [`App`] (god-struct
/// decomposition). Both scans walk the workspace on a blocking worker so a
/// large repo never stalls the UI thread; each pairs a `JoinHandle` with a
/// cooperative cancel flag because `spawn_blocking` workers cannot be
/// force-stopped once running. `file_*` feeds the full `@` file-picker modal;
/// `workspace_*` feeds the inline `@`-mention completion list. All `None` when
/// no scan is pending — hence `#[derive(Default)]`.
#[derive(Default)]
struct BackgroundScans {
    /// In-flight background workspace scan feeding the `@` file picker.
    /// The BFS over the cwd runs on a blocking worker so a large repo
    /// never stalls the UI thread; the run loop polls this and lands the
    /// result via [`FilePickerModal::set_items`]. `None` when no scan is
    /// pending.
    file_task: Option<tokio::task::JoinHandle<Vec<String>>>,
    /// Cooperative cancellation flag for the full file-picker scan.
    file_cancel: Option<ScanCancelToken>,
    /// In-flight background workspace scan for the *inline* `@`-mention list
    /// (sister to `file_task`, which feeds the full picker modal). The
    /// scan must never run inline on the UI thread: the BFS walks up to 4000
    /// directories and typing `@` used to freeze the composer until it
    /// finished (the intermittent "input lags while typing" bug).
    workspace_task: Option<tokio::task::JoinHandle<Vec<String>>>,
    /// Cooperative cancellation flag for the inline workspace scan. Needed
    /// because `spawn_blocking` workers cannot be force-stopped once running.
    workspace_cancel: Option<ScanCancelToken>,
}

/// Transcript search (Ctrl+F) state, grouped off [`App`] (god-struct
/// decomposition). The query buffer, its match set, and the active-match
/// cursor form one cohesive responsibility (incremental find + highlight +
/// wrap-around navigation). All fields start empty — hence `#[derive(Default)]`.
#[derive(Default)]
struct SearchState {
    /// Search query buffer (populated in [`AppMode::Search`]).
    query: String,
    /// Transcript block indices matching the current search query, in
    /// document order. Recomputed incrementally as the query changes.
    matches: Vec<usize>,
    /// Index into [`Self::matches`] of the active match (scrolled to and
    /// highlighted). Meaningless when `matches` is empty.
    active_match: usize,
}

/// All slash-command / picker modal handles, grouped off [`App`] (god-struct
/// decomposition). At most one is `Some` at a time — the currently displayed
/// overlay, mirrored by the [`AppMode`] state machine. All `None` at rest,
/// hence `#[derive(Default)]`.
#[derive(Default)]
struct Modals {
    /// Interactive `/diff` viewer modal.
    diff_viewer: Option<DiffViewerModal>,
    /// Snapshot rewind/diff viewer modal when Ctrl+R is active.
    rewind: Option<RewindViewerModal>,
    /// Live workflow progress viewer modal when Ctrl+O is active.
    workflow: Option<WorkflowViewerModal>,
    /// Session agents viewer modal when Ctrl+G is active.
    agents: Option<AgentsViewerModal>,
    /// `TeamInbox` viewer modal when `/inbox` is active.
    team_inbox: Option<TeamInboxViewerModal>,
    /// Graphical token/cost dashboard when `/usage` is active.
    usage_dashboard: Option<UsageDashboardModal>,
}

/// Selection payloads for choice-style modals whose visible labels are separate
/// from the command/session tokens applied after selection. Keeping these
/// parallel vectors and the generic argument-picker command together avoids
/// scattering picker bookkeeping across the top-level [`App`] coordinator.
#[derive(Default)]
struct ChoiceModalState {
    /// Session IDs parallel to the `/resume` session picker's options; indexed
    /// by the selected row when the slot modal resolves.
    session_ids: Vec<String>,
    /// `command:provider` tokens parallel to the `/login` · `/connect` picker's
    /// options; indexed by the selected row when the slot modal resolves.
    login_provider_ids: Vec<String>,
    /// Slash command backing the arg-picker modal on the active slot (e.g.
    /// `"theme"`); the chosen label is re-submitted as `/<command> <label>`.
    arg_picker_command: String,
}

/// Slash-command / `@`-mention hint popup state, grouped off [`App`]
/// (god-struct decomposition). The keyboard cursor and the "explicitly
/// dismissed for this exact input" guard form one cohesive responsibility per
/// popup. All fields start empty — hence `#[derive(Default)]`.
#[derive(Default)]
struct HintsState {
    /// Keyboard cursor inside the slash-command hint popup. `None` when
    /// no item is highlighted.
    slash_cursor: Option<usize>,
    /// Exact input text for which the slash hint was explicitly dismissed.
    /// Editing the input makes the text differ and allows the hint to reopen.
    slash_hidden_for: Option<String>,
    /// Keyboard cursor inside the `@`-mention hint popup. `None` when no
    /// item is highlighted (mirrors `slash_cursor`).
    mention_cursor: Option<usize>,
    /// Exact input text for which the mention hint was explicitly dismissed.
    /// Editing the input makes the text differ and allows the hint to reopen.
    mention_hidden_for: Option<String>,
}

/// A rollback snapshot of the TUI plan-gate state, taken before a mode
/// transition mutates the [`App`] so the change can be undone if the runtime
/// permission change fails. See [`App::plan_mode_snapshot`].
#[derive(Clone, Copy)]
pub struct PlanModeSnapshot {
    plan_mode_active: bool,
    plan_prev_mode: Option<PermissionMode>,
    perm_mode: PermissionMode,
}

/// The top-level TUI application.
#[allow(clippy::struct_excessive_bools)]
pub struct App {
    /// Session-wide terminal/rendering strategy. Fullscreen is the default and
    /// preserves the existing draw path byte-for-byte.
    terminal_mode: TerminalMode,
    /// Settled inline transcript chunks awaiting `Terminal::insert_before`.
    finalized_transcript: FinalizedTranscriptQueue,
    /// Current mode (Normal / Modal*).
    mode: AppMode,
    /// Resolved theme.
    theme: Theme,
    /// Most recent layout — refreshed on every draw.
    regions: Option<LayoutRegions>,
    /// The rect the transcript body was actually drawn into on the most
    /// recent draw — `regions.transcript` minus the bottom overlay
    /// reservations (todo/queue/agent panels, search bar) and the startup
    /// banner offset. Scroll clamps and mouse hit-tests MUST use this
    /// viewport, not the full region: clamping against the full height made
    /// the last `bottom_reserved` rows unreachable behind the pinned panels
    /// (wheel-clip root cause). One frame of lag matches the reservation
    /// model, which likewise reads the previous frame's resolved scroll.
    transcript_draw_rect: Option<Rect>,
    /// Screen rect of the pinned live-agent panel from the most recent draw,
    /// `None` when the panel is not showing. A left-click inside it opens the
    /// live agent view (Ctrl+O), Claude-Code style.
    agent_panel_click_rect: Option<Rect>,
    /// Per-agent-row click targets from the most recent draw: `(row rect, agent
    /// id)` for each agent visible in the pinned panel. A left-click inside a
    /// row opens the live agent view focused on THAT agent; a click elsewhere in
    /// the panel falls back to `agent_panel_click_rect` (aggregate view).
    /// Rebuilt every frame alongside `agent_panel_click_rect`, so it never
    /// outlives the geometry it describes.
    agent_row_click_targets: Vec<(Rect, String)>,
    /// Agent id currently under the mouse in the pinned panel, or `None`. Drives
    /// the hover underline; updated only when the hovered target actually
    /// changes so raw mouse motion never forces a repaint.
    hovered_agent: Option<String>,
    /// Consecutive live-viewer refreshes that read back no snapshot. One empty
    /// read can be a torn manifest / progress-doc swap mid-write; the viewer
    /// only auto-closes after two in a row (see
    /// [`Self::apply_workflow_viewer_snapshot`]).
    workflow_viewer_empty_refreshes: u8,
    /// Render receiver from the agent task.
    rx: mpsc::Receiver<RenderBlock>,
    /// Optional non-blocking projection hook for the session-local remote UI.
    /// The callback must never wait; remote backpressure is isolated behind it.
    render_observer: Option<RenderObserver>,
    /// Ordered remote spectator ingress. Normal frames and snapshot replacement
    /// share this one bounded queue to prevent cross-channel stale reordering.
    spectator_rx: mpsc::Receiver<SpectatorEvent>,
    /// Lowest replacement boundary the App may accept. Own-turn completion
    /// advances this before the spectator gate is released.
    spectator_floor: u64,
    /// Agent-command sender to the agent task.
    cmd_tx: mpsc::Sender<AgentCommand>,
    /// Last observed Ctrl-C timestamp (for double-tap exit detection).
    last_ctrl_c: Option<Instant>,
    /// Last observed bare-Esc timestamp in Normal mode (for the
    /// Esc-Esc "rewind previous checkpoint" double-tap).
    last_esc: Option<Instant>,
    /// Count of render blocks drained — exposed for integration tests.
    blocks_drained: usize,
    /// Live streaming pacer: buffers the open prose block's arrived-but-unrevealed
    /// characters and drips them across a few frames so bursty providers read as a
    /// smooth type-in. `None` outside an actively streaming prose block. See
    /// [`stream_pace`].
    stream_pacer: Option<stream_pace::StreamPacer>,
    /// `true` once the main loop has been asked to exit.
    should_quit: bool,
    /// Scrollable transcript of [`RenderBlock`]s.
    transcript: Transcript,
    /// Transcript viewport interaction state (follow-tail, scrollbar drag,
    /// copy hover, and character-selection press state). See
    /// [`TranscriptViewState`].
    transcript_view: TranscriptViewState,
    /// Current HUD snapshot. Seeded with a neutral default so the
    /// foundation tests that don't push a real model still render.
    hud_state: HudState,
    /// Whether the TUI plan-mode gate is engaged. Plan mode maps to the
    /// runtime's read-only permission, so the runtime alone cannot tell a
    /// plain `read-only` session from a plan-first one — this flag carries
    /// that distinction so the HUD can label the badge `plan` (and the
    /// Shift+Tab cycle has a fourth, read-only-backed stop). Set by the
    /// Shift+Tab cycle and `/plan on`; cleared on `/plan off` or any cycle
    /// step that leaves the read-only stop.
    plan_mode_active: bool,
    /// The TUI permission mode to restore when plan mode is turned off via
    /// `/plan off`. `None` when plan mode was never explicitly entered via
    /// `/plan on` (the cycle does not record a prior mode), in which case the
    /// restore defaults to `Workspace`.
    plan_prev_mode: Option<PermissionMode>,
    /// Rollback snapshot for a Shift+Tab cycle in flight. The cycle mutates the
    /// plan-gate state during key handling (before the runtime permission change
    /// is applied by the host loop), so it records the pre-mutation state here;
    /// the host loop restores it if `apply_permission_change` fails and clears it
    /// on success. `None` outside a pending cycle.
    plan_cycle_rollback: Option<PlanModeSnapshot>,
    /// Owned input widget (Lane L7c).
    input: InputWidget,
    /// Whether the in-app input widget is currently the active source
    /// of prompt text.
    input_enabled: bool,
    /// Active permission prompt awaiting user decision, if any (L7c).
    ///
    /// Held by value because [`PermissionPrompt`] is not `Clone` — its
    /// `oneshot::Sender` must stay unique until the modal resolves.
    active_prompt: Option<PermissionPrompt>,
    /// Cursor row in the permission prompt's choice list (↑↓ navigation),
    /// defaulted to the safe Deny choice when a prompt opens.
    permission_selected: usize,
    /// Active `AskUserQuestion` prompt awaiting a modal answer.
    active_user_question: Option<UserQuestionPrompt>,
    /// All slash-command / picker modal handles, grouped off [`App`]. At most
    /// one is `Some` at a time (the active overlay); see [`Modals`].
    modals: Modals,
    /// The active modal overlay as a trait object, for modals migrated onto the
    /// unified [`Modal`] slot (P6). Exactly one of this or a legacy [`Modals`]
    /// field is ever `Some`; [`Self::set_active_modal`] enforces that. Its key
    /// dispatch and draw flow through one path instead of a per-modal arm.
    active_modal: Option<Box<dyn Modal>>,
    /// Startup launchpad state (screen plus intro/repaint clock),
    /// grouped off [`App`]. See [`StartupState`].
    startup: StartupState,
    /// Monotonic render tick used by lightweight widget animations.
    tick: u64,
    /// In-flight turn snapshot driving the activity spinner. `None`
    /// when no turn is running.
    turn_activity: Option<TurnActivity>,
    /// Time at which the canonical turn activity was torn down. Present after
    /// a turn ends so draw-time heat derivation can animate the cooling ramp.
    cooled_since: Option<Instant>,
    /// Highest sub-agent heartbeat (`lastActivityAt` epoch secs) observed so far
    /// this turn. A delegating main turn is parked awaiting its fan-out, so the
    /// main model emits no events and the spinner would flip to a false "no
    /// output" badge. When a live HUD poll shows this advance — an agent ran a
    /// tool, streamed, or changed phase — the stall clock is reset. It is a
    /// *conservative* signal: a genuinely hung / rate-limited swarm stops
    /// writing manifests, so the heartbeat freezes and the badge still surfaces.
    last_agent_heartbeat: u64,
    /// Whether the model wrote/updated the plan (`TodoWrite`) during the
    /// current turn. The live pinned plan panel reflects the plan the model is
    /// *actively maintaining*, so a stale all-pending plan left in the store by
    /// an earlier turn must not re-pin itself above the input on every later,
    /// unrelated turn (the "ghost plan" residue). Reset in
    /// [`Self::begin_turn_with_generation`],
    /// set in [`Self::apply_todo_tool_result`]. The sidebar todo section is
    /// independent and keeps showing persistent outstanding work.
    todo_touched_this_turn: bool,
    /// Live agent batches under in-flight Spawn-family tool calls, ordered by
    /// tool-call start. Runtime can announce several delegation calls before
    /// executing them serially, so each transcript row needs its own merge
    /// state instead of one replaceable "current" batch.
    agent_batches: Vec<agent_batch::ActiveAgentBatch>,
    /// Session-installed poller for the custom `statusLine` command. Called
    /// on a ~1s tick cadence; returns the freshest cached command output.
    status_line_poller: Option<StatusLinePoller>,
    /// Cached scheduler sources. File scans are event-gated by the session;
    /// loop deadlines arrive through the session sync seam.
    scheduled_wake_poller: Option<ScheduledWakePoller>,
    workspace_status_poller: Option<WorkspaceStatusPoller>,
    /// Live MCP status source (see [`Self::refresh_mcp_status`]): re-derives
    /// the HUD's per-server rows between the action-boundary HUD rebuilds so
    /// background discovery transitions surface while the prompt sits idle.
    mcp_status_poller: Option<McpStatusPoller>,
    /// Last time the MCP poller ran — the ~1s cadence gate for
    /// [`Self::refresh_mcp_status`], which is driven from the 30 fps tick.
    mcp_status_polled_at: Option<Instant>,
    scheduled_file_wake: Option<ScheduledWakeHud>,
    scheduled_loop_wake: Option<ScheduledWakeHud>,
    scheduled_wake: Option<ScheduledWakeHud>,
    /// Finalized first non-empty reasoning title per live reasoning block. This
    /// keeps the spinner stable after the first reasoning line is newline-
    /// terminated and avoids rescanning/copying long accumulated reasoning.
    reasoning_activity_titles: HashMap<BlockId, String>,
    /// Bounded source for an in-progress first reasoning line before it is
    /// newline-terminated. This preserves fragments that appear after a very
    /// long blank prefix without scanning the full accumulated `prior` again.
    reasoning_activity_open_titles: HashMap<BlockId, String>,
    /// Images pasted from the clipboard awaiting the next submit.
    pending_images: Vec<ImageAttachment>,
    /// Set when the next submit is a re-injected background agent result, so the
    /// submit path renders an [`RenderBlock::AgentResult`] card instead of a
    /// `You` user message. Staged when the tagged queued message pops (mirroring
    /// `pending_images`), consumed once at submit.
    pending_agent_result: Option<AgentResultMeta>,
    /// File the `/memory` command asked the host to open in `$EDITOR`. The host
    /// loop owns the terminal, so the command records the request here and the
    /// loop drains it after slash dispatch.
    pending_editor_file: Option<PathBuf>,
    /// Transcript dump the `/dump` command asked the host to open in `$PAGER`
    /// (or `$EDITOR` for `/dump edit`). Same contract as
    /// [`Self::pending_editor_file`]: recorded here, drained by the loop.
    pending_transcript_view: Option<TranscriptViewRequest>,
    /// Frecency tracker for slash command usage.
    command_history: CommandHistory,
    /// Prevent repeated non-fatal history persistence failures from flooding the
    /// transcript; the first failure still remains visible to the user.
    history_persistence_warning_shown: bool,
    /// Manifest-read cache for the viewer's ~2 Hz poll, so a long-open viewer
    /// stops re-parsing every (often terminal) per-agent manifest on the render
    /// thread each tick. See [`workflow_progress::read_view_cached`].
    workflow_view_cache: super::workflow_progress::WorkflowViewCache,
    /// Pre-rendered body lines for the Esc-Esc rewind confirmation card.
    /// `Some` only while [`AppMode::ModalConfirmRewind`] is active; cleared by
    /// [`Self::exit_modal`].
    rewind_confirm: Option<Vec<String>>,
    /// Choice-modal payloads kept alongside their visible picker labels.
    choice_modals: ChoiceModalState,
    /// Slash-command / `@`-mention hint popup state (see [`HintsState`]).
    hints: HintsState,
    /// Frecency tracker for `@`-file mentions (sister to `command_history`).
    mention_history: CommandHistory,
    /// Lazily-collected workspace file list backing `@`-mention completion;
    /// filled on first mention via `ensure_workspace_files`, reused after.
    workspace_files: Vec<String>,
    /// Input history loaded from disk.
    history: History,
    /// Current position in history browsing. `None` = not browsing.
    /// 0 = most recent entry, 1 = second most recent, etc.
    history_cursor: Option<usize>,
    /// Stashed user input before history browsing began.
    history_stash: String,
    /// Transcript search (Ctrl+F) state: query buffer, match set, and
    /// active-match cursor, grouped off [`App`]. See [`SearchState`].
    search: SearchState,
    /// Pager content shown in [`AppMode::Pager`] overlay.
    pager_content: Option<String>,
    /// Cached raw line model for pager content. Drawing slices this vector to
    /// the visible window instead of splitting and allocating the full output on
    /// every frame.
    pager_lines: Vec<String>,
    /// Vertical scroll offset within the pager overlay.
    pager_scroll: u16,
    /// Detected terminal image protocol for inline rendering.
    image_protocol: ImageProtocol,
    /// `true` when the terminal is known to handle CSI ?2026, so
    /// [`App::draw_frame`] brackets each frame in synchronized output.
    synchronized_output: bool,
    /// File-changes sidebar state (toggled with Ctrl+B).
    sidebar: SidebarState,
    /// Messages typed and queued by the user while a turn is in progress.
    /// Drained by the caller (turn controller) via [`App::take_queued_messages`].
    /// Each entry carries its own pasted images (Claude Code CLI parity).
    queued_messages: VecDeque<QueuedMessage>,
    /// Epoch-second lower bound for agent manifests that belong to this visible
    /// session. The agent store is workspace-global, so the UI must filter out
    /// fresh-but-foreign manifests from previous chats.
    agent_manifest_started_after: u64,
    /// Foreground session id for the live TUI. New agent manifests carry this
    /// id, allowing the HUD/agents detail view to hide agents from a different
    /// Claude/GPT session running in the same workspace.
    agent_manifest_session_id: Option<String>,
    /// Live-process counter from the exact foreground runtime used by this App.
    /// It is already scoped to the visible session when installed; render-time
    /// reads touch only the small live-counter map.
    background_process_count: runtime::task_registry::LiveBackgroundProcessCount,
    /// Project-local Markdown prompt commands discovered by the session layer.
    /// Live-discovered MCP prompts are merged in as synthesized entries
    /// (`mcp__server__prompt` names) so the slash hint/completion surfaces
    /// them; dispatch never reads this list, so the merge cannot reroute.
    prompt_commands: Vec<commands::PromptCommandDef>,
    /// Last `RuntimeMcpState::prompts_version` merged into
    /// [`Self::prompt_commands`]; `None` forces a re-merge on the next sync.
    mcp_prompts_version: Option<u64>,
    /// Background workspace-scan handles (file picker + inline mention),
    /// grouped to keep the scan lifecycle in one place. See [`BackgroundScans`].
    scans: BackgroundScans,
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("mode", &self.mode)
            .field("blocks_drained", &self.blocks_drained)
            .field("should_quit", &self.should_quit)
            .finish_non_exhaustive()
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.cancel_workspace_scan();
        self.cancel_file_scan();
    }
}

impl App {
    /// Maximum number of queued render blocks to ingest before one paint.
    ///
    /// Drawing is still frame-gated, so raising this does not repaint more
    /// often. It only keeps the bounded render channel from filling during
    /// provider/SSE bursts; otherwise the awaited parser send backpressures the
    /// HTTP body read and turns a fast stream into a cap-sized clump → draw
    /// pause → clump cadence.
    const MAX_DRAIN_PER_TICK: usize = 256;

    const MOUSE_SCROLL_ROWS: u16 = 3;
    const KEY_SCROLL_ROWS: u16 = 3;
    const HALF_PAGE_SCROLL_ROWS: u16 = 10;
    /// Window inside which a second Ctrl-C counts as "double-tap exit".
    pub const CTRL_C_DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(1000);

    /// Window inside which a second bare Esc counts as the "rewind previous
    /// checkpoint" double-tap (conversation + code together).
    pub const ESC_DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(600);

    /// How long the startup intro needs idle redraws before settling. The
    /// ignition sequence keeps a small settle buffer so the byte-identical
    /// final frame lands before the idle loop stops redrawing.
    const STARTUP_INTRO: Duration = if super::startup::INTRO_TOTAL_MS == 0 {
        Duration::from_millis(0)
    } else {
        Duration::from_millis(super::startup::INTRO_TOTAL_MS + 150)
    };

    fn cancel_workspace_scan(&mut self) {
        if let Some(cancel) = self.scans.workspace_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        if let Some(task) = self.scans.workspace_task.take() {
            task.abort();
        }
    }

    pub(super) fn cancel_file_scan(&mut self) {
        if let Some(cancel) = self.scans.file_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        if let Some(task) = self.scans.file_task.take() {
            task.abort();
        }
    }

    /// Construct a new application handle.
    ///
    /// Callers (Lane L7) own the channel pair and pass the receive
    /// side plus the command sender in.
    #[must_use]
    pub fn new(
        theme: Theme,
        rx: mpsc::Receiver<RenderBlock>,
        cmd_tx: mpsc::Sender<AgentCommand>,
    ) -> Self {
        let (_spectator_tx, spectator_rx) = mpsc::channel(1);
        Self::new_with_spectator(theme, rx, spectator_rx, cmd_tx)
    }

    /// Construct an App with a caller-owned ordered spectator ingress.
    #[must_use]
    pub fn new_with_spectator(
        theme: Theme,
        rx: mpsc::Receiver<RenderBlock>,
        spectator_rx: mpsc::Receiver<SpectatorEvent>,
        cmd_tx: mpsc::Sender<AgentCommand>,
    ) -> Self {
        Self {
            terminal_mode: TerminalMode::Fullscreen,
            finalized_transcript: FinalizedTranscriptQueue::default(),
            mode: AppMode::Normal,
            theme,
            regions: None,
            transcript_draw_rect: None,
            agent_panel_click_rect: None,
            agent_row_click_targets: Vec::new(),
            hovered_agent: None,
            workflow_viewer_empty_refreshes: 0,
            rx,
            render_observer: None,
            spectator_rx,
            spectator_floor: 0,
            cmd_tx,
            last_ctrl_c: None,
            last_esc: None,
            blocks_drained: 0,
            stream_pacer: None,
            should_quit: false,
            transcript: Transcript::new(),
            transcript_view: TranscriptViewState::default(),
            hud_state: default_hud_state(),
            plan_mode_active: false,
            plan_prev_mode: None,
            plan_cycle_rollback: None,
            input: InputWidget::new(),
            input_enabled: false,
            active_prompt: None,
            permission_selected: 0,
            active_user_question: None,
            modals: Modals::default(),
            active_modal: None,
            startup: StartupState::default(),
            tick: 0,
            turn_activity: None,
            cooled_since: None,
            last_agent_heartbeat: 0,
            todo_touched_this_turn: false,
            agent_batches: Vec::new(),
            status_line_poller: None,
            scheduled_wake_poller: None,
            workspace_status_poller: None,
            mcp_status_poller: None,
            mcp_status_polled_at: None,
            scheduled_file_wake: None,
            scheduled_loop_wake: None,
            scheduled_wake: None,
            reasoning_activity_titles: HashMap::new(),
            reasoning_activity_open_titles: HashMap::new(),
            pending_images: Vec::new(),
            pending_agent_result: None,
            pending_editor_file: None,
            pending_transcript_view: None,
            // Inert default: reads empty, writes discard. The session loop
            // attaches the real per-user history via `set_command_history`
            // once the data directory is known (see `tui_loop`).
            command_history: CommandHistory::load(PathBuf::from("/dev/null"))
                .expect("inert command history"),
            history_persistence_warning_shown: false,
            workflow_view_cache: super::workflow_progress::WorkflowViewCache::default(),
            rewind_confirm: None,
            choice_modals: ChoiceModalState::default(),
            hints: HintsState::default(),
            // Inert default: reads empty, writes discard. The session loop
            // attaches the real per-project history via `set_mention_history`
            // once the workspace `.zo/` dir is known (see `tui_loop`).
            mention_history: CommandHistory::load(PathBuf::from("/dev/null"))
                .expect("inert mention history"),
            workspace_files: Vec::new(),
            history: History::load(PathBuf::from("/tmp/.zo-dummy-history-nonexistent"))
                .unwrap_or_else(|_| {
                    History::load_with_max(PathBuf::from("/dev/null"), 1).expect("dummy history")
                }),
            history_cursor: None,
            history_stash: String::new(),
            search: SearchState::default(),
            pager_content: None,
            pager_lines: Vec::new(),
            pager_scroll: 0,
            image_protocol: TermProfile::current().image,
            synchronized_output: TermProfile::current().synchronized_output,
            sidebar: SidebarState::new(),
            queued_messages: VecDeque::new(),
            agent_manifest_started_after: epoch_seconds_now(),
            agent_manifest_session_id: None,
            background_process_count: runtime::task_registry::LiveBackgroundProcessCount::default(),
            prompt_commands: Vec::new(),
            mcp_prompts_version: None,
            scans: BackgroundScans::default(),
        }
    }

    /// Advance the render tick by one frame using wrapping arithmetic.
    pub fn advance_tick(&mut self) {
        // 인증 소스 라인은 전역 셀 스냅샷 — 락 한 번, 30fps에 무해.
        self.hud_state.auth_origin = api::latest_claude_auth_origin();
        self.tick = self.tick.wrapping_add(1);
        // Drip the streaming pacer one frame's worth: reveal the characters the
        // elapsed wall-clock time has earned, so a buffered burst types out
        // smoothly across frames. No-op when nothing is paced. Driven here (the
        // shared 30 fps grid) so both the idle loop and the in-turn render tick
        // advance the reveal on the same cadence.
        self.drip_stream();
        // Custom status line refresh — both the idle loop and the in-turn
        // select loop drive ticks, so the configured command stays live in
        // either state. ~1s cadence at the 30fps grid; the poller itself is
        // debounced and thread-offloaded, so this can never stall a frame.
        if self.tick.is_multiple_of(30) {
            if let Some(poller) = self.status_line_poller.as_ref() {
                let line = poller(&self.hud_state);
                self.hud_state.status_line = line;
            }
            // Live MCP rows for the in-turn render path; the idle loop drives
            // the same refresh from its tick arm (pre-draw) so idle changes
            // also trigger their own redraw.
            self.refresh_mcp_status(Instant::now());
        }
    }

    /// Install the session-side status line poller (settings `statusLine`).
    /// The closure receives the current [`HudState`] and returns the freshest
    /// cached command output; `None` clears the custom line.
    pub fn set_status_line_poller(&mut self, poller: StatusLinePoller) {
        self.status_line_poller = Some(poller);
    }

    /// Install the session-side wakeup scanner. Its closure decides from
    /// freshness events whether a filesystem scan is due.
    pub fn set_scheduled_wakeup_poller(&mut self, poller: ScheduledWakePoller) {
        self.scheduled_wake_poller = Some(poller);
    }

    /// Install the session-side event-gated workspace status source.
    pub fn set_workspace_status_poller(&mut self, poller: WorkspaceStatusPoller) {
        self.workspace_status_poller = Some(poller);
    }

    /// Install (or re-arm) the live MCP status source. Re-installed on every
    /// session sync because `/resume`, `/session switch`, and context reloads
    /// replace the runtime and its MCP state handle — a poller captured once
    /// at startup would keep reading the dead runtime's state forever.
    pub fn set_mcp_status_poller(&mut self, poller: McpStatusPoller) {
        self.mcp_status_poller = Some(poller);
    }

    /// Re-derive the HUD's MCP rows from live session state, at most once per
    /// second. Returns whether the rows changed (the caller's redraw signal).
    ///
    /// The full HUD snapshot is rebuilt only at action boundaries
    /// (`sync_app_context` in the session loop), so without this poll a
    /// background discovery that finishes while the prompt sits idle leaves
    /// the sidebar stuck on "discovering" — servers long connected, HUD never
    /// told — until the next user action. Driven from both idle and in-turn
    /// tick paths; the cadence gate makes double-driving harmless.
    pub fn refresh_mcp_status(&mut self, now: Instant) -> bool {
        const MCP_STATUS_POLL_INTERVAL: Duration = Duration::from_secs(1);
        let Some(poller) = self.mcp_status_poller.as_ref() else {
            return false;
        };
        if self
            .mcp_status_polled_at
            .is_some_and(|last| now.saturating_duration_since(last) < MCP_STATUS_POLL_INTERVAL)
        {
            return false;
        }
        self.mcp_status_polled_at = Some(now);
        let Some(rows) = poller() else {
            return false;
        };
        if rows == self.hud_state.mcp_servers {
            return false;
        }
        self.hud_state.mcp_servers = rows;
        true
    }

    fn spawn_workspace_status_snapshot(
        &self,
    ) -> Option<tokio::task::JoinHandle<Option<GitStatusSnapshot>>> {
        self.workspace_status_poller
            .as_ref()
            .and_then(|poller| poller(&self.hud_state.cwd))
    }

    /// Refresh the cached `ScheduleWakeup` source when its event source is due.
    /// Returns whether the current frame should redraw.
    pub fn refresh_scheduled_wakeup(&mut self) -> bool {
        let Some(poller) = self.scheduled_wake_poller.as_ref() else {
            return false;
        };
        let Some(next) = poller() else {
            return false;
        };
        let changed = next != self.scheduled_file_wake;
        self.scheduled_file_wake = next;
        self.publish_scheduled_wake();
        changed || self.scheduled_wake.is_some()
    }

    /// Replace the `/loop` candidate from the same session snapshot used to arm
    /// the idle timer.
    pub fn set_scheduled_loop_wake(&mut self, wake: Option<ScheduledWakeHud>) {
        self.scheduled_loop_wake = wake;
        self.publish_scheduled_wake();
    }

    /// A consumed wakeup must stop rendering `wake now` immediately. Any later
    /// pending file is discovered by the next one-second scan.
    pub fn clear_scheduled_file_wake(&mut self) {
        self.scheduled_file_wake = None;
        self.publish_scheduled_wake();
    }

    fn publish_scheduled_wake(&mut self) {
        self.scheduled_wake = match (&self.scheduled_file_wake, &self.scheduled_loop_wake) {
            (Some(wakeup), Some(loop_wake)) => Some(if wakeup.due_at_epoch <= loop_wake.due_at_epoch {
                wakeup.clone()
            } else {
                loop_wake.clone()
            }),
            (Some(wakeup), None) => Some(wakeup.clone()),
            (None, Some(loop_wake)) => Some(loop_wake.clone()),
            (None, None) => None,
        };
        self.hud_state.scheduled_wake = self.scheduled_wake.clone();
    }

    /// Mark a new turn as in progress so the activity spinner takes
    /// over the rule-top row. Idempotent if a turn is already active.
    ///
    /// CC parity: starting a turn keeps the tail pinned only when the user was
    /// already following. The submit path re-arms follow for messages the user
    /// sends themselves, so their own turns still snap to the tail — but
    /// auto-started turns (queued drain, loops, agent-result re-injection)
    /// must not yank a reader who scrolled up. In long agentic sessions turn
    /// boundaries fire constantly, and the old unconditional re-arm read as
    /// "mouse scrolling is broken while streaming".
    ///
    /// Begin a turn with the host's monotonic turn generation, used to select
    /// a stable prose-streaming Zo verb without adding randomness.
    pub fn begin_turn_with_generation(&mut self, turn_generation: u64) {
        // Runtime fallback notices are turn-scoped. A quota-parked turn emits
        // its active notice again after this reset; the first recovered turn
        // emits none and therefore returns the HUD to the configured model.
        self.hud_state.turn_fallback_model = None;
        self.hud_state.quota_fallback_model = None;
        if self.turn_activity.is_none() {
            self.cooled_since = None;
            self.clear_reasoning_activity_title_caches();
            let mut activity =
                TurnActivity::new_for_turn(std::time::Instant::now(), turn_generation);
            // Seed an immediate zo cue so the live indicator reads as active
            // cognition from frame zero, for every provider. Claude/GPT stream
            // reasoning almost at once and overwrite this within a tick; Gemini
            // (Code Assist) computes server-side before its first SSE frame, so
            // without this seed its longer pre-first-block wait shows only the
            // bare "Working" fallback and reads as a frozen turn. The first real
            // block (reasoning / text / tool) replaces it. Uses the Zo
            // metaphor (`Thinking`) instead of a generic `Thinking`, matching the
            // streaming reasoning cue.
            activity.set_current_action(crate::tui::blocks::reasoning::ZO_REVEAL_VERBS[0]);
            self.turn_activity = Some(activity);
        }
        // A new turn starts with no plan written *yet*. Until the model calls
        // `TodoWrite` this turn, any plan in the store is carryover from an
        // earlier turn and must not pin the live panel (see the field doc).
        self.todo_touched_this_turn = false;
        // Fresh stall baseline: the first agent poll of this turn (any heartbeat
        // > 0) counts as progress and resets the spinner's stall clock.
        self.last_agent_heartbeat = 0;
        // The live pinned todo panel owns the plan while the turn streams, so
        // tell the transcript to suppress its `Updated Plan` Todos block until
        // the turn settles (prevents the plan rendering twice).
        self.transcript.set_turn_active(true);
        if self.transcript_view.follow_output {
            self.transcript.scroll_to_bottom();
        }
    }

    /// Update the in-flight turn's streaming token counters. No-op if
    /// no turn is active.
    ///
    /// A `tokens_in` of `0` is treated as "no new input figure" and leaves the
    /// previous value intact, so the sub-agent token path
    /// (`update_turn_tokens(0, agent_tokens)`) cannot erase a real input count
    /// already supplied by a `RenderBlock::Usage` snapshot. Any token update is
    /// also observable progress, so it resets the stall clock.
    pub fn update_turn_tokens(&mut self, tokens_in: u32, tokens_out: u32) {
        if let Some(activity) = self.turn_activity.as_mut() {
            if tokens_in > 0 {
                activity.tokens_in = tokens_in;
            }
            // Fan-out aggregate (summed sub-agent output). Kept monotonic but
            // deliberately excluded from the throughput window — sub-agent bursts
            // are not main-model generation speed.
            activity.record_agent_output(tokens_out, std::time::Instant::now());
        }
    }

    /// Set the in-flight turn's user-facing activity line. Host-driven prelude
    /// work (for example automatic fan-out before the model stream starts)
    /// does not arrive as `RenderBlock`s, so it needs this direct path.
    pub fn set_turn_activity(&mut self, action: impl Into<String>) {
        if let Some(activity) = self.turn_activity.as_mut() {
            activity.set_current_action(action);
            // A new host-driven label IS observable progress (a fan-out phase
            // advanced, triage finished, the main turn started). Reset the stall
            // clock so the multi-second prelude can't leave the turn reading
            // "no output" the instant the main model takes over.
            activity.mark_event();
        }
    }

    /// Clear the activity spinner; called once the turn finishes.
    ///
    /// CC parity: a settled turn snaps to the tail only when the user was
    /// still following — so the common case still lands on the final answer
    /// ("마지막 출력되면 화면으로 내려가게"), and the pin keeps a deferred
    /// typewriter tail on screen as it finishes typing. A reader who scrolled
    /// up mid-stream keeps their place instead: the earlier unconditional
    /// snap, combined with auto-started follow-up turns, yanked the viewport
    /// at every turn boundary and read as broken mouse scrolling. Wheel-down
    /// to the tail, `End`, or submitting a message re-arms following.
    pub fn end_turn(&mut self) {
        self.settle_turn(true);
    }

    /// Tear down a turn that was prepared locally but never executed.
    ///
    /// This clears the same live state as [`Self::end_turn`] without starting a
    /// cooling animation for work that never ran.
    pub fn abort_turn(&mut self) {
        self.settle_turn(false);
    }

    fn settle_turn(&mut self, cool: bool) {
        // Mark any still-buffered streamed tail `done` so it finishes typing out
        // smoothly on the following idle ticks (a fast typewriter finish, not a
        // one-frame jump). The provider's final delta usually already set this;
        // this also seals a turn that ended without a terminal delta.
        self.finish_stream();
        if self.terminal_mode.is_inline() {
            // Native scrollback chunks cannot merge with a later paced tail.
            // Complete the already-finished block before ownership moves to
            // the finalized queue; fullscreen keeps its existing idle drip.
            self.flush_stream();
        }
        self.clear_reasoning_activity_title_caches();
        // Agent trees are persisted in the transcript side table; this live
        // merge state is only for the streaming turn that just ended.
        self.agent_batches.clear();
        if self.turn_activity.take().is_some() {
            self.cooled_since = if cool && !self.theme.no_color {
                Some(Instant::now())
            } else {
                None
            };
        }
        // The turn settled: the live panel is gone, so let the transcript show
        // its `Updated Plan` Todos block again as settled history.
        self.transcript.set_turn_active(false);
        // A plan whose every item is now complete is finished: delete it from
        // the store so it does not ghost in the sidebar / next turn's panel.
        self.clear_completed_plan_store();
        if self.transcript_view.follow_output {
            self.transcript.scroll_to_bottom();
        }
        self.finalize_inline_transcript();
    }

    /// Select the session terminal strategy before the first frame.
    pub fn set_terminal_mode(&mut self, mode: TerminalMode) {
        self.terminal_mode = mode;
        if mode.is_inline() {
            // Native scrollback owns selection and history in inline mode.
            self.sidebar.visible = false;
            self.synchronized_output = false;
        }
    }

    /// Current terminal strategy.
    #[must_use]
    pub const fn terminal_mode(&self) -> TerminalMode {
        self.terminal_mode
    }

    /// Seal any currently-live transcript blocks for native scrollback.
    ///
    /// `end_turn` calls this after draining the render channel. The session
    /// teardown and resumed-session bootstrap also call it so notices/history
    /// outside a model turn are not left behind in the viewport.
    pub fn finalize_inline_transcript(&mut self) {
        if self.terminal_mode.is_inline() {
            self.finalized_transcript.finalize(&mut self.transcript);
        }
    }

    fn clear_reasoning_activity_title_caches(&mut self) {
        self.reasoning_activity_titles.clear();
        self.reasoning_activity_open_titles.clear();
    }

    /// Read-only access to the active turn snapshot, if any.
    #[must_use]
    pub const fn turn_activity(&self) -> Option<&TurnActivity> {
        self.turn_activity.as_ref()
    }

    fn heat_state_at(&self, now: Instant) -> HeatState {
        HeatState::derive(self.turn_activity.is_some(), self.cooled_since, now)
    }

    fn heat_state(&self) -> HeatState {
        self.heat_state_at(Instant::now())
    }

    fn cooling_active_at(&self, now: Instant) -> bool {
        self.heat_state_at(now).is_cooling()
    }

    /// Current render tick.
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// Mutable access to the owned input widget (Lane L7c).
    pub fn input_mut(&mut self) -> &mut InputWidget {
        &mut self.input
    }

    /// Read-only access to the owned input widget (Lane L7c).
    #[must_use]
    pub const fn input(&self) -> &InputWidget {
        &self.input
    }

    /// Enable in-app input handling for persistent-session flows.
    pub fn enable_input(&mut self) {
        self.input_enabled = true;
    }

    /// Disable in-app input handling.
    pub fn disable_input(&mut self) {
        self.input_enabled = false;
    }

    /// `true` when the in-app input widget currently handles normal-mode keys.
    #[must_use]
    pub const fn input_enabled(&self) -> bool {
        self.input_enabled
    }

    /// Replace the project prompt-command entries shown in the command
    /// palette. The merged MCP subset (names starting `mcp__`) is owned by
    /// [`Self::set_mcp_prompt_commands`] and survives the replacement — this
    /// setter runs every loop tick and must not wipe it.
    pub fn set_prompt_commands(&mut self, commands: Vec<commands::PromptCommandDef>) {
        let merged_mcp: Vec<commands::PromptCommandDef> = self
            .prompt_commands
            .iter()
            .filter(|command| command.name.starts_with("mcp__"))
            .cloned()
            .collect();
        self.prompt_commands = commands;
        self.prompt_commands
            .retain(|command| !command.name.starts_with("mcp__"));
        self.prompt_commands.extend(merged_mcp);
    }

    /// Last MCP prompts version merged into the palette (see
    /// [`Self::set_mcp_prompt_commands`]).
    #[must_use]
    pub fn mcp_prompts_version(&self) -> Option<u64> {
        self.mcp_prompts_version
    }

    /// Merge live-discovered MCP prompts into the palette as synthesized
    /// prompt-command entries (names start with `mcp__`). Replaces only the
    /// previously merged MCP subset; project `.md` commands are untouched.
    pub fn set_mcp_prompt_commands(
        &mut self,
        version: u64,
        commands: Vec<commands::PromptCommandDef>,
    ) {
        self.mcp_prompts_version = Some(version);
        self.prompt_commands
            .retain(|command| !command.name.starts_with("mcp__"));
        self.prompt_commands.extend(commands);
    }

    /// Route an inbound [`RenderBlock`] to the correct surface.
    ///
    /// The ingest is a fixed pipeline of cohesive stages, each a
    /// single-responsibility helper: reap stale reasoning titles, update
    /// fallback identity and the turn spinner, bump the live sidebar tool
    /// activity, then route the block to its surface (modal / HUD / transcript).
    /// Keeping the stage order here is the pinned contract that later phases
    /// assemble against.
    pub fn push_block(&mut self, block: RenderBlock) {
        self.push_block_inner(block);
    }

    pub fn set_render_observer(
        &mut self,
        observer: Option<RenderObserver>,
    ) {
        self.render_observer = observer;
    }

    fn push_block_inner(&mut self, block: RenderBlock) {
        if let Some(observer) = self.render_observer.as_ref() {
            observer(&block);
        }
        self.blocks_drained = self.blocks_drained.saturating_add(1);
        self.startup.screen = None;
        self.reap_reasoning_titles_on_done(&block);
        self.update_fallback_model_state(&block);
        self.update_turn_activity(&block);
        self.update_live_tool_activity(&block);
        self.route_block_to_surface(block);
    }

    /// Stage 1: a finished reasoning block drops its rolling-title cache entries
    /// so a later turn reusing the same [`BlockId`] cannot echo a stale title.
    fn reap_reasoning_titles_on_done(&mut self, block: &RenderBlock) {
        if let RenderBlock::Reasoning { id, done: true, .. } = block {
            self.reasoning_activity_titles.remove(id);
            self.reasoning_activity_open_titles.remove(id);
        }
    }

    /// Latch the model actually serving this turn from stable runtime notices.
    /// The next [`Self::begin_turn_with_generation`] clears both fields before a
    /// parked quota turn can re-announce its sustained fallback.
    fn update_fallback_model_state(&mut self, block: &RenderBlock) {
        let RenderBlock::System { text, .. } = block else {
            return;
        };
        if text == core_types::REFUSAL_FALLBACK_WARN {
            self.hud_state.turn_fallback_model = Some("opus".to_string());
        } else if let Some(model) = core_types::parse_quota_fallback_model(text) {
            self.hud_state.quota_fallback_model = Some(model.to_string());
        }
    }

    /// Stage 2: drive the live turn spinner (streamed token estimate, stall
    /// clock, and the current-action title) from the inbound block. No-op when
    /// no turn is active.
    #[allow(clippy::too_many_lines)] // cohesive spinner-activity update
    fn update_turn_activity(&mut self, block: &RenderBlock) {
        // Streaming token estimate for the activity spinner. We can't
        // observe the runtime's UsageTracker mid-turn (it's borrowed by
        // the agent future), so we approximate output tokens at ~4 chars
        // per token from streamed text/reasoning deltas. The HUD still
        // shows the authoritative count once the turn completes.
        if let Some(activity) = self.turn_activity.as_mut() {
            let added_chars: usize = match block {
                RenderBlock::TextDelta { text, .. } | RenderBlock::Reasoning { text, .. } => {
                    text.chars().count()
                }
                _ => 0,
            };
            if added_chars > 0 {
                let added_tokens = u32::try_from(added_chars / 4).unwrap_or(u32::MAX);
                // This chars/4 estimate drives the visible count until the first
                // authoritative usage snapshot latches a baseline; after that the
                // baseline owns the count, but the estimate keeps feeding the
                // throughput window so the live tok/s rate stays alive between
                // (sparse) usage snapshots.
                activity.bump_output_estimate(added_tokens, std::time::Instant::now());
            }
            // Any transcript-bound block is observable progress: reset the
            // stall clock so the spinner keeps reading "working", not "stuck".
            if matches!(
                block,
                RenderBlock::TextDelta { .. }
                    | RenderBlock::Reasoning { .. }
                    | RenderBlock::ToolCall { .. }
                    | RenderBlock::ToolResult { .. }
                    // A mid-run `send_to_user` push is observable progress: the
                    // model actively produced content, so reset the stall clock
                    // and keep the spinner reading "working".
                    | RenderBlock::UserNotice { .. }
            ) {
                activity.mark_event();
            }

            match block {
                RenderBlock::ToolCall {
                    name,
                    summary,
                    status:
                        runtime::message_stream::ToolCallStatus::Pending
                        | runtime::message_stream::ToolCallStatus::Running,
                    ..
                } => {
                    // A running tool gets the longer stall grace — builds/tests/
                    // commands/fetches are silent for 90-120s by nature.
                    activity.set_tool_action(tool_call::activity_summary(name, summary));
                }
                RenderBlock::ToolResult { .. } => {
                    activity.set_current_action("Reading tool output; choosing next step");
                }
                RenderBlock::TextDelta { text, .. } if !text.is_empty() => {
                    activity.set_prose_streaming_action();
                }
                RenderBlock::Reasoning {
                    id,
                    text,
                    done: false,
                    ..
                } => {
                    // `text` is only the latest thinking DELTA. Keep spinner
                    // title reconstruction turn-local: runtime block IDs can be
                    // reused by later turns, so reading transcript state by
                    // BlockId can accidentally pick up a stale prior turn. The
                    // bounded open-title cache carries the current turn's
                    // unterminated first line; the finalized cache carries it
                    // after newline termination.
                    //
                    // Rolling title: a paragraph break in the delta means the
                    // model moved on to a new thought — retitle from the newest
                    // paragraph instead of freezing on the block's first line
                    // for the whole reasoning phase (GPT streams its summary
                    // parts as paragraphs on one block; long Claude thinking
                    // has the same shape).
                    let text: &str = if let Some((_, tail)) = text.rsplit_once("\n\n") {
                        self.reasoning_activity_titles.remove(id);
                        self.reasoning_activity_open_titles.remove(id);
                        tail
                    } else {
                        text
                    };
                    if let Some(title) = self.reasoning_activity_titles.get(id) {
                        activity.set_current_action(title.clone());
                    } else {
                        let source = if let Some(open) = self.reasoning_activity_open_titles.get(id) {
                            append_reasoning_title_source_fragment(open, text)
                        } else {
                            reasoning_title_source("", text).into_owned()
                        };
                        let title = crate::tui::blocks::reasoning::reasoning_summary_line(&source);
                        if title.is_empty() {
                            activity.set_current_action(reasoning_activity_summary(&source));
                        } else if reasoning_first_non_empty_line_is_terminated(&source) {
                            self.reasoning_activity_open_titles.remove(id);
                            self.reasoning_activity_titles.insert(*id, title.clone());
                            activity.set_current_action(title);
                        } else {
                            self.reasoning_activity_open_titles.insert(*id, source);
                            activity.set_current_action(title);
                        }
                    }
                }
                RenderBlock::CompactionProgress { streamed_chars } => {
                    // Live compaction heartbeat: the summary streams internally,
                    // so its char count is the only visible movement. Reset the
                    // stall clock and show it on the spinner.
                    activity.mark_event();
                    let tenths = streamed_chars / 100;
                    activity.set_current_action(format!(
                        "Compacting conversation… ↓ {}.{}k chars",
                        tenths / 10,
                        tenths % 10
                    ));
                }
                RenderBlock::System { text, .. }
                    if text.starts_with(core_types::QUIET_REASONING_LABEL) =>
                {
                    // The stream's quiet-reasoning heartbeat: keep-alive chunks
                    // are verifiably arriving while the model reasons without a
                    // visible delta. Latch the calm badge state — the stall
                    // clock keeps counting, but the badge now says "reasoning ·
                    // stream alive" instead of "no output", so a healthy
                    // multi-minute xhigh reasoning pass stops reading as a hang
                    // (live report: users Esc'd out and lost the whole pass).
                    activity.note_stream_alive_quiet();
                }
                RenderBlock::System { text, .. }
                    if text.starts_with(core_types::QUOTA_HOLD_NOTICE_PREFIX) =>
                {
                    // The runtime parked this turn until the model's quota
                    // window resets (hard 429 → wait band, up to ~15 minutes of
                    // deliberate silence). Say so on the spinner — the live
                    // report this fixes: the park read as "no output 5m" and
                    // the user concluded the session was wedged. The action
                    // line names the state; the latched badge keeps it honest
                    // while the stall clock counts.
                    activity.set_current_action(
                        "Rate-limited · holding for quota reset — esc to interrupt",
                    );
                    activity.note_quota_hold();
                }
                RenderBlock::System { text, .. } if text.starts_with("Compacting conversation") => {
                    // CC-style progress: surface the compaction in the spinner
                    // (which already renders "esc to interrupt") instead of leaving
                    // it on the prior "Drafting response" while the summary streams.
                    // mark_event here (only for this known progress notice, not
                    // every System line) so the stall clock does not fire during
                    // the summary round-trip.
                    activity.mark_event();
                    activity.set_current_action(text.clone());
                }
                RenderBlock::System { text, .. } if text.starts_with("Compacted conversation") => {
                    // Compaction finished — clear the "Compacting…" action so a
                    // subsequent model hang shows the stall indicator, not a stale
                    // compaction spinner.
                    activity.mark_event();
                    activity.set_current_action("Resuming after compaction");
                }
                _ => {}
            }
        }
    }

    /// Stage 3: bump the live sidebar tool activity — a starting tool increments
    /// the bash/read/edit counter (and opens an agent batch for delegation
    /// calls); a tool result clears the in-flight chip, applies a `TodoWrite`
    /// checklist, and seals the batch.
    fn update_live_tool_activity(&mut self, block: &RenderBlock) {
        // Live activity bump: increment the sidebar's bash/read/edit
        // counter the instant a tool starts running, so users see work
        // progressing instead of zeros while the session is borrowed.
        if let RenderBlock::ToolCall {
            tool_call_id,
            name,
            summary,
            status,
            ..
        } = block
        {
            let in_flight = matches!(
                status,
                runtime::message_stream::ToolCallStatus::Pending
                    | runtime::message_stream::ToolCallStatus::Running
            );
            let action = if in_flight {
                Some(tool_call::activity_summary(name, summary))
            } else {
                None
            };
            self.bump_tool_activity(name, action.as_deref());
            // A delegation call (Spawn family or the Workflow tool) opens a live
            // agent batch: the per-agent tree renders under this row and fills
            // from manifests + completions, so running sub-agents are visible
            // inline in the transcript and not just in the sidebar / viewer.
            if in_flight && tool_call::opens_agent_batch(name) {
                self.begin_agent_batch(&tool_call_id.0);
            }
        }
        // Tool completed → clear the "currently running" surface so the
        // sidebar stops claiming a stale tool is still in flight. The
        // `▸ <name>` chip in the sidebar reflects only true in-flight
        // work after this; the lifetime totals stay in `bash/read/edit`.
        if let RenderBlock::ToolResult {
            tool_call_id, body, ..
        } = block
        {
            self.hud_state.last_tool = None;
            // A `TodoWrite` result carries the authoritative new checklist in
            // its JSON output. Apply it to the HUD the instant the result lands
            // so the sidebar reflects the update immediately, instead of waiting
            // up to ~330 ms for the next live-snapshot poll to re-read the file.
            self.apply_todo_tool_result(body);
            // Seals the agent batch if this result belongs to its Spawn call —
            // the row flips to the `N agents finished` header.
            self.finish_agent_batch(&tool_call_id.0);
        }
    }

    /// Stage 4: route the block to its surface — blocking prompts open modals
    /// and stash their oneshot responders, live/rate-limit snapshots update the
    /// HUD ledger, active-turn prose is paced through the stream buffer, and
    /// everything else lands in the transcript in true arrival order.
    fn route_block_to_surface(&mut self, block: RenderBlock) {
        match block {
            RenderBlock::PermissionPrompt(prompt) => self.open_permission_modal(prompt),
            RenderBlock::UserQuestionPrompt(prompt) => self.open_user_question_modal(prompt),
            RenderBlock::Usage {
                ctx_tokens,
                cumulative,
                current,
            } => self.record_live_usage(ctx_tokens, cumulative, current),
            RenderBlock::RateLimit(snapshot) => {
                // Live ledger only — like Usage, this never enters the transcript.
                self.hud_state.rate_limit = Some(snapshot);
            }
            // An empty, not-yet-done text delta is a pure "open" signal with no
            // content; skip it so it never creates a phantom empty prose block.
            // The transcript opens a block on the first non-empty delta anyway.
            RenderBlock::TextDelta { text, done, .. } if text.is_empty() && !done => {}
            // Live streamed prose during an active turn is paced through the
            // stream buffer so a provider's sentence-sized bursts read as a smooth
            // type-in instead of whole chunks landing at once. There is no rate
            // ceiling and no hold-back (see `stream_pace`): the first glyphs show
            // on the arrival frame and a `done` tail settles in ~2-3 frames.
            RenderBlock::TextDelta { id, text, done } if self.turn_activity.is_some() => {
                Self::trace_stream_delta(text.chars().count());
                self.buffer_paced(id, text, done);
            }
            // Everything else — replayed/resumed prose (no active turn), tool
            // calls, system blocks — lands in the transcript in true arrival
            // order. A non-prose block first flushes any paced tail so the
            // order (prose, tool, prose) is preserved without a hold-back queue.
            other => {
                self.flush_stream();
                self.push_transcript_block_now(other);
            }
        }
    }

    /// Diagnostic: when `ZO_DELTA_TRACE` is set, append each live text
    /// delta's char count and inter-arrival gap (ms) to
    /// `~/.zo/logs/delta-trace.log`, one `chars<TAB>gap_ms` line per delta.
    /// Off → a single env probe and an immediate return (zero cost). Used to
    /// measure a provider's real streaming cadence (e.g. Anthropic burstiness:
    /// many tiny deltas landing in one frame vs an even per-token drip).
    fn trace_stream_delta(chars: usize) {
        use std::io::Write as _;
        static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
        if std::env::var_os("ZO_DELTA_TRACE").is_none() {
            return;
        }
        let now = std::time::Instant::now();
        let gap_ms = {
            let mut last = LAST
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let gap = last.map_or(0, |t| now.duration_since(t).as_millis());
            *last = Some(now);
            gap
        };
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let path = std::path::Path::new(&home).join(".zo/logs/delta-trace.log");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{chars}\t{gap_ms}");
        }
    }

    fn push_transcript_block_now(&mut self, block: RenderBlock) {
        // A steering delivery echo means that queued message just reached the live
        // turn — clear its pending "queued" entry so it does not re-submit as a
        // separate turn afterwards.
        if let RenderBlock::System { text, .. } = &block {
            if let Some(steer) = text.strip_prefix(runtime::STEERING_ECHO_PREFIX) {
                self.remove_queued_message_matching(steer);
            }
        }
        self.transcript.push(block);

        if self.transcript_view.follow_output {
            self.transcript.scroll_to_bottom();
        }
    }

    /// Replace or append a synthetic system block in the transcript. This is for
    /// host-driven live status such as automatic fan-out progress, where the
    /// central chat pane should keep showing current work without appending a
    /// fresh log line on every tick.
    pub fn upsert_system_block(
        &mut self,
        id: runtime::message_stream::BlockId,
        level: SystemLevel,
        text: String,
    ) {
        self.blocks_drained = self.blocks_drained.saturating_add(1);
        self.startup.screen = None;
        self.transcript.upsert_system(id, level, text);
        if self.transcript_view.follow_output {
            self.transcript.scroll_to_bottom();
        }
    }

    /// Apply a real mid-turn usage snapshot (from [`RenderBlock::Usage`]) to the
    /// HUD ledger: authoritative ctx tokens plus cost priced with the active
    /// model. This replaces the per-tick char-count estimate so the live ledger
    /// reflects the runtime's real `UsageTracker` instead of an approximation.
    fn record_live_usage(
        &mut self,
        ctx_tokens: u64,
        cumulative: runtime::TokenUsage,
        current: runtime::TokenUsage,
    ) {
        // A real usage snapshot is itself observable progress (the request
        // landed / a model iteration completed): reset the stall clock and feed
        // the spinner the *authoritative* token counts instead of leaving its
        // char-count estimate. `tokens_in` is otherwise always 0 for a plain
        // single-model turn; `tokens_out` otherwise drifts on CJK / code.
        if let Some(activity) = self.turn_activity.as_mut() {
            activity.mark_event();
            if current.input_tokens > 0 {
                activity.tokens_in = current.input_tokens;
            }
            // Drive the live count from the monotonic *cumulative* output minus a
            // turn-start baseline, not the per-iteration `current.output_tokens`
            // (which is smaller on every new round-trip and made the counter drop).
            activity.record_output_usage(
                cumulative.output_tokens,
                current.output_tokens,
                std::time::Instant::now(),
            );
        }
        // A mid-stream ctx snapshot (emitted at `message_start`, before any
        // output is generated) carries an empty cumulative: ctx is known but
        // cost is not. Advance the live ctx immediately so the ledger tracks
        // window occupancy the moment the request lands, but keep the prior
        // cost until the turn's real cumulative arrives — otherwise the live
        // preview would blink the cost back to $0.00 each iteration.
        if cumulative.total_tokens() == 0 {
            // The snapshot's `current` already knows this request's new/cached
            // split — advance it with ctx so the `⤷ N new · M cached` line
            // doesn't hold the previous turn's values until billing lands.
            if current.input_tokens > 0 || current.cache_read_input_tokens > 0 {
                self.hud_state.ctx_new_input = u64::from(current.input_tokens);
                self.hud_state.ctx_cached = u64::from(current.cache_read_input_tokens);
            }
            self.update_hud_usage(ctx_tokens, self.hud_state.cost_usd);
            return;
        }
        // Split the *current* context window into billed (new) vs cache-read so
        // the `⤷ N new · M cached` line is a true breakdown of the `ctx` figure
        // above it: `new + cached ≤ ctx_tokens`, same scale. This must come from
        // the latest request (`current`), not `cumulative` — the session-summed
        // cache reads run to the millions and would dwarf the per-request ctx
        // line (showing e.g. `8.6M cached` under a `191.9k / 1.0M` window).
        self.hud_state.ctx_new_input = u64::from(current.input_tokens);
        self.hud_state.ctx_cached = u64::from(current.cache_read_input_tokens);
        let pricing = runtime::pricing_for_model(&self.hud_state.model.alias);
        self.hud_state.cost_approx = pricing.is_none();
        let cost = cumulative
            .estimate_cost_usd_with_pricing(
                pricing.unwrap_or_else(runtime::ModelPricing::default_sonnet_tier),
            )
            .total_cost_usd();
        self.update_hud_usage(ctx_tokens, cost);
    }

    pub fn set_startup_screen(&mut self, startup_screen: StartupScreen) {
        self.startup.show(startup_screen);
    }

    pub fn dismiss_startup_screen(&mut self) {
        self.startup.dismiss();
    }

    /// `true` while the startup launchpad is visible and its bounded ignition
    /// intro still needs idle redraws. It becomes false after the settle buffer,
    /// returning the TUI to zero-CPU idle.
    fn startup_intro_active(&self) -> bool {
        // Under reduce-motion the launchpad renders settled from the first
        // frame (see `render`), so it never needs per-tick idle redraws.
        !crate::tui::term::reduce_motion_enabled()
            && self
                .startup
                .shown_at
                .is_some_and(|shown| shown.elapsed() < Self::STARTUP_INTRO)
    }

    /// Revert the given repo-relative file to `HEAD`, then refresh the
    /// open diff viewer from a fresh `git diff HEAD`.
    ///
    /// Reuses the same `git checkout HEAD -- <path>` that backs the
    /// snapshot/undo machinery (worktree restore from a known tree); on
    /// success the file drops out of the diff. Files not present in `HEAD`
    /// (brand-new adds) cannot be reverted this way — git errors and the
    /// note explains why rather than deleting anything from the TUI. When
    /// nothing changed remains the viewer closes.
    fn revert_diff_file(&mut self, path: &str) {
        match git_checkout_head(path) {
            Ok(()) => {
                let diff_text = git_diff_head();
                let files = super::modals::diff_viewer::parse_unified_diff(&diff_text);
                if files.is_empty() {
                    self.exit_modal();
                    self.push_diff_note(
                        SystemLevel::Info,
                        format!("Reverted {path} — no changes remain"),
                    );
                } else if let Some(modal) = self.modals.diff_viewer.as_mut() {
                    modal.set_files(files);
                    self.push_diff_note(SystemLevel::Info, format!("Reverted {path}"));
                }
            }
            Err(error) => {
                self.push_diff_note(
                    SystemLevel::Error,
                    format!("Could not revert {path}: {error}"),
                );
            }
        }
    }

    /// Push a one-line system note into the transcript (revert feedback).
    /// Uses a process-local monotonic id so the synthetic block never
    /// collides with runtime-streamed block ids.
    fn push_diff_note(&mut self, level: SystemLevel, text: String) {
        use std::sync::atomic::{AtomicU64, Ordering};
        // High base keeps these well clear of the runtime's id range.
        static NOTE_ID: AtomicU64 = AtomicU64::new(u64::MAX / 2);
        self.push_block(RenderBlock::System {
            id: runtime::message_stream::BlockId(NOTE_ID.fetch_add(1, Ordering::Relaxed)),
            level,
            text,
        });
    }

    /// Cursor row in the active permission prompt's choice list.
    #[must_use]
    pub const fn permission_selected(&self) -> usize {
        self.permission_selected
    }

    /// Move the permission-prompt cursor one row (↑/↓), clamped (no wrap) so the
    /// bottom Deny row can never wrap up to an allow. No-op without an active
    /// prompt. Returns the new index.
    pub fn move_permission_selection(&mut self, up: bool) -> usize {
        if let Some(prompt) = self.active_prompt.as_ref() {
            let len = prompt.choices.len();
            self.permission_selected =
                crate::tui::blocks::permission::move_selection(self.permission_selected, up, len);
        }
        self.permission_selected
    }

    /// Take ownership of the active permission prompt, clearing it
    /// from the [`App`]. Used by the host loop to resolve the prompt's
    /// oneshot responder exactly once.
    pub fn take_active_prompt(&mut self) -> Option<PermissionPrompt> {
        let taken = self.active_prompt.take();
        if taken.is_some() {
            self.mode = AppMode::Normal;
        }
        taken
    }

    /// Dismiss the active permission prompt only when it matches `id`.
    /// Remote approval uses this to close the losing TUI surface without
    /// disturbing a newer prompt that may already have replaced it.
    pub fn dismiss_permission_prompt(&mut self, id: BlockId) -> bool {
        if self.active_prompt.as_ref().map(|prompt| prompt.id) != Some(id) {
            return false;
        }
        self.active_prompt = None;
        self.mode = AppMode::Normal;
        true
    }

    /// Mutable access to the transcript viewport.
    ///
    /// Exposed so Lane L7 (or integration tests) can push
    /// [`RenderBlock`]s into the visible surface without going
    /// through the mpsc channel.
    pub fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.transcript
    }

    /// Read-only access to the visible transcript (tests / host introspection).
    #[must_use]
    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }

    /// Advance the local own-turn fence. It is monotonic so an older remote
    /// replacement can never reopen a transcript already settled by this App.
    pub fn advance_spectator_floor(&mut self, next_seq: u64) {
        self.spectator_floor = self.spectator_floor.max(next_seq);
    }

    /// Process one spectator event in FIFO order. A stale Replace is rejected
    /// without clearing the transcript; an accepted one replaces atomically.
    pub fn process_spectator_event(&mut self, event: SpectatorEvent) {
        match event {
            SpectatorEvent::Frame { frame_seq, block } => {
                if frame_seq < self.spectator_floor {
                    return;
                }
                self.push_block_inner(block);
            }
            SpectatorEvent::Replace { blocks, post_boundary, next_seq, ack } => {
                if next_seq < self.spectator_floor {
                    let _ = ack.send(AckVerdict::Stale);
                    return;
                }
                self.clear_transcript();
                for block in blocks {
                    self.push_block_inner(block);
                }
                for block in post_boundary {
                    self.push_block_inner(block);
                }
                self.advance_spectator_floor(next_seq);
                let _ = ack.send(AckVerdict::Applied);
            }
        }
    }

    /// Reset the visible session surface after commands like
    /// `/clear`, `/resume`, or `/session switch`.
    pub fn reset_session_view(&mut self) {
        self.discard_stream();
        self.transcript.clear();
        self.transcript_view.follow_output = true;
        self.input.clear();
        self.mode = AppMode::Normal;
        // Drop the sticky startup launchpad: live turns dismiss it on first
        // submit, but `/resume`/`/clear` reseed the transcript without a
        // submit, and the leftover banner painted over the reseeded blocks
        // (the "broken rendering after /resume" bug).
        self.startup.dismiss();
        self.active_prompt = None;
        self.active_user_question = None;
        // Migrated modals live on the unified slot; clear it alongside the other
        // prompt state so a reseed (`/clear`, `/resume`, `/new`) starts clean.
        self.active_modal = None;
        self.rewind_confirm = None;
        // Clear the live-ledger fields that belong to the previous session so
        // the sidebar does not show stale todos/agents/activity after `/clear`,
        // `/new`, or `/resume`. The next `build_hud_state` sync repopulates
        // them from the fresh session; until then they must read empty rather
        // than linger (set directly — `set_hud_state`'s monotonic-max would
        // otherwise pin the old counters).
        self.hud_state.todo_items.clear();
        self.hud_state.todo_summary = None;
        self.hud_state.agents.clear();
        self.hud_state.running_agents = 0;
        self.hud_state.workflow = None;
        self.hud_state.last_tool = None;
        self.hud_state.turn_fallback_model = None;
        self.hud_state.quota_fallback_model = None;
        self.hud_state.bash_count = 0;
        self.hud_state.read_count = 0;
        self.hud_state.edit_count = 0;
        self.hud_state.background_tasks = 0;
        self.hud_state.scheduled_wake = None;
        self.background_process_count =
            runtime::task_registry::LiveBackgroundProcessCount::default();
        self.scheduled_file_wake = None;
        self.scheduled_loop_wake = None;
        self.scheduled_wake = None;
    }

    /// Drop all visible transcript state so the surface can be
    /// repopulated for a new session context.
    pub fn clear_transcript(&mut self) {
        self.discard_stream();
        self.transcript.clear();
        self.transcript_view.hovered_copy_button = None;
        self.transcript_view.follow_output = true;
    }

    /// Re-enable auto-follow and scroll to the newest output. A model turn does
    /// this in [`Self::begin_turn_with_generation`], but slash commands and modal/shortcut state
    /// changes (`/goal`, `/model`, `/permissions`, …) append their confirmation
    /// without it: if the user had scrolled up, the confirmation landed
    /// off-screen and the command looked like a no-op. Call this when surfacing
    /// such immediate feedback so it is always visible.
    pub fn follow_latest(&mut self) {
        self.transcript_view.follow_output = true;
        self.transcript.scroll_to_bottom();
    }

    /// Replace the current HUD snapshot. Lane L7 hooks this into
    /// the live cost/usage/mcp counters.
    ///
    /// Bash/read/edit counters are *monotonically* preserved: the live
    /// `bump_tool_activity` path counts every `ToolCall` (including
    /// sub-agent tools that never land in the parent session, like
    /// `read_file` issued inside a `SpawnMultiAgent` worker), whereas
    /// the periodic rebuild in `build_hud_state` only sees parent-session
    /// `ContentBlock::ToolResult`s. Picking the max keeps the user-visible
    /// totals from snapping back to zero after every turn-boundary sync.
    pub fn set_hud_state(&mut self, mut hud: HudState) {
        hud.bash_count = hud.bash_count.max(self.hud_state.bash_count);
        hud.read_count = hud.read_count.max(self.hud_state.read_count);
        hud.edit_count = hud.edit_count.max(self.hud_state.edit_count);
        // These fields are owned by turn-scoped runtime notices, not the
        // session snapshot. Preserve them across periodic HUD rebuilds; the
        // canonical turn-start boundary clears them explicitly.
        hud.turn_fallback_model
            .clone_from(&self.hud_state.turn_fallback_model);
        hud.quota_fallback_model
            .clone_from(&self.hud_state.quota_fallback_model);
        // Preserve the live ctx/cost the streaming `RenderBlock::Usage` path
        // recorded this turn. The periodic rebuild can momentarily compute a
        // zero ctx/cost (e.g. before the runtime's UsageTracker has the first
        // billed turn, or on a tool-only iteration), and overwriting with that
        // zero is exactly what made the ledger read `ctx 0 / 1.0M  0%` and
        // `cost $0.00` even after real usage had streamed in. Only let the
        // rebuild raise these values, never reset them to zero.
        if hud.ctx_used == 0 {
            hud.ctx_used = self.hud_state.ctx_used;
        }
        if hud.cost_usd <= 0.0 {
            hud.cost_usd = self.hud_state.cost_usd;
        }
        if hud.perm_mode == PermissionMode::ReadOnly {
            if self.plan_mode_active {
                hud.perm_mode = PermissionMode::Plan;
            }
        } else {
            self.plan_mode_active = false;
        }
        // The process count is sourced only from the currently installed
        // session-scoped atomic handle; callers cannot inject a stale count.
        hud.background_tasks = self.background_task_count();
        hud.scheduled_wake.clone_from(&self.scheduled_wake);
        // Turn-boundary rebuilds carry a fresh manifest scan too — keep the
        // transcript's agent tree in lockstep with it.
        self.refresh_agent_batch(&hud.agents);
        self.hud_state = hud;
    }

    /// Hydrate the sidebar's session metadata for a `zo attach` client.
    ///
    /// Unlike the local REPL — which builds a full [`HudState`] from a live
    /// `LiveCli` — an attach client has no runtime, so the model / permission
    /// mode / cwd / branch arrive from the server's `session.info` response.
    /// The live-ledger fields (ctx, cost, tool counts, rate-limit) are *not*
    /// touched here: they keep updating from streamed `Usage` / `RateLimit` /
    /// `ToolCall` blocks via [`Self::push_block`].
    pub fn set_session_meta(
        &mut self,
        model_alias: &str,
        context_limit: u64,
        perm_mode: RuntimePermissionMode,
        cwd: PathBuf,
        git_branch: Option<String>,
    ) {
        // Attach metadata names the model already selected by the server, so its
        // display must not depend on which provider credentials exist locally.
        let trimmed_alias = model_alias.trim();
        let resolved_model = provider_catalog()
            .iter()
            .find(|entry| entry.alias.eq_ignore_ascii_case(trimmed_alias))
            .map_or_else(
                || resolve_model_alias(trimmed_alias),
                |entry| entry.canonical_model_id.to_string(),
            );
        let provider = match detect_provider_kind(&resolved_model) {
            ProviderKind::Anthropic => "claude",
            ProviderKind::Xai => "xai",
            ProviderKind::OpenAi => "openai",
            ProviderKind::Google => "google",
            ProviderKind::Ollama => "ollama",
        };
        self.hud_state.model = runtime::message_stream::ActiveModel {
            provider,
            alias: model_alias.to_string(),
            display_name: resolved_model,
            context_limit: u32::try_from(context_limit).unwrap_or(u32::MAX),
        };
        self.hud_state.ctx_limit = context_limit;
        self.hud_state.perm_mode = match perm_mode {
            // The runtime has no `Plan`: plan mode is read-only at the
            // runtime layer, so a read-only snapshot resolves to the `Plan`
            // badge only when the TUI plan-mode gate is engaged.
            RuntimePermissionMode::ReadOnly if self.plan_mode_active => PermissionMode::Plan,
            RuntimePermissionMode::ReadOnly => PermissionMode::ReadOnly,
            RuntimePermissionMode::WorkspaceWrite | RuntimePermissionMode::Prompt => {
                PermissionMode::Workspace
            }
            RuntimePermissionMode::DangerFullAccess | RuntimePermissionMode::Allow => {
                PermissionMode::All
            }
        };
        self.hud_state.cwd = cwd;
        self.hud_state.git_branch = git_branch;
    }

    /// Refresh the live context-usage and cost fields on the HUD
    /// without rebuilding the whole snapshot. Called from the render
    /// loop during a streaming turn so the percentage advances in
    /// real time instead of waiting for the next turn boundary.
    pub fn update_hud_usage(&mut self, ctx_used: u64, cost_usd: f64) {
        self.hud_state.ctx_used = ctx_used;
        self.hud_state.cost_usd = cost_usd;
    }

    /// Current session cwd used by live sidebar refresh tasks.
    #[must_use]
    pub fn hud_cwd(&self) -> PathBuf {
        self.hud_state.cwd.clone()
    }

    /// The HUD permission badge as currently shown (`Plan` resolves through
    /// the plan-mode gate). Exposed so the host loop / slash handlers can read
    /// the visible mode without reaching into the HUD snapshot.
    #[must_use]
    pub fn perm_mode(&self) -> PermissionMode {
        self.hud_state.perm_mode
    }

    /// Engage the TUI plan-mode gate: remember the currently-shown permission
    /// mode (so `/plan off` can restore it) and flip the HUD badge to `Plan`.
    ///
    /// Re-entrant: a second `enter_plan_mode` while already in plan mode keeps
    /// the originally-remembered prior mode rather than recording `Plan` as the
    /// thing to restore to. The runtime read-only switch is the caller's job;
    /// this only owns the TUI-visible gate state.
    pub fn enter_plan_mode(&mut self) {
        if !self.plan_mode_active {
            self.plan_prev_mode = Some(self.hud_state.perm_mode);
        }
        self.plan_mode_active = true;
        self.hud_state.perm_mode = PermissionMode::Plan;
    }

    /// Leave the TUI plan-mode gate, returning the permission mode to restore.
    ///
    /// Restores the mode remembered by [`Self::enter_plan_mode`], defaulting to
    /// [`PermissionMode::Workspace`] when none was recorded (e.g. plan mode was
    /// reached through the Shift+Tab cycle, which records no prior). The caller
    /// applies the corresponding runtime permission change.
    pub fn exit_plan_mode(&mut self) -> PermissionMode {
        let restored = self.plan_prev_mode.take().unwrap_or(PermissionMode::Workspace);
        self.plan_mode_active = false;
        self.hud_state.perm_mode = restored;
        restored
    }

    /// Whether the TUI plan-mode gate is currently engaged. Drives the
    /// Shift+Tab cycle's read-only-backed `Plan` stop.
    #[must_use]
    pub fn plan_mode_active(&self) -> bool {
        self.plan_mode_active
    }

    /// Mark the read-only-backed Shift+Tab stop as plain `ReadOnly` (cycle
    /// stepped off the `Plan` stop) or `Plan`. Used by the key handler so the
    /// next session-snapshot refresh resolves the read-only runtime mode to the
    /// right badge.
    pub fn set_plan_mode_active(&mut self, active: bool) {
        self.plan_mode_active = active;
    }

    /// Capture the exact TUI plan-gate state (`plan_mode_active` plus the
    /// remembered prior mode and the HUD badge) so a mode transition can be made
    /// transactional. The App is mutated before the runtime permission change is
    /// applied; if that change fails, [`Self::restore_plan_mode_snapshot`] rolls
    /// the App back to this snapshot so the UI Plan flag never diverges from the
    /// runtime and `plan_selected` is left unchanged.
    #[must_use]
    pub fn plan_mode_snapshot(&self) -> PlanModeSnapshot {
        PlanModeSnapshot {
            plan_mode_active: self.plan_mode_active,
            plan_prev_mode: self.plan_prev_mode,
            perm_mode: self.hud_state.perm_mode,
        }
    }

    /// Roll the TUI plan-gate state back to a [`Self::plan_mode_snapshot`],
    /// undoing an `enter_plan_mode` / `exit_plan_mode` / cycle mutation whose
    /// runtime permission change failed.
    pub fn restore_plan_mode_snapshot(&mut self, snapshot: PlanModeSnapshot) {
        self.plan_mode_active = snapshot.plan_mode_active;
        self.plan_prev_mode = snapshot.plan_prev_mode;
        self.hud_state.perm_mode = snapshot.perm_mode;
    }

    /// Record the pre-mutation plan-gate state before a Shift+Tab cycle flips it,
    /// so the host loop can roll it back if the runtime permission change fails.
    pub fn arm_plan_cycle_rollback(&mut self) {
        self.plan_cycle_rollback = Some(self.plan_mode_snapshot());
    }

    /// Take the armed Shift+Tab rollback snapshot, clearing it. The host loop
    /// applies it on `apply_permission_change` failure and discards it on success
    /// so the plan-gate mutation commits only alongside the runtime change.
    pub fn take_plan_cycle_rollback(&mut self) -> Option<PlanModeSnapshot> {
        self.plan_cycle_rollback.take()
    }

    /// Live update for tool activity (bash/read/edit). Called when a
    /// `ToolCall` arrives during streaming so the sidebar reflects
    /// work in progress instead of waiting for the turn to finish.
    pub fn bump_tool_activity(&mut self, tool_name: &str, current_action: Option<&str>) {
        let lower = tool_name.to_ascii_lowercase();
        if lower.contains("bash") || lower.contains("shell") {
            self.hud_state.bash_count = self.hud_state.bash_count.saturating_add(1);
        } else if lower.contains("read") || lower.contains("grep") || lower.contains("glob") {
            self.hud_state.read_count = self.hud_state.read_count.saturating_add(1);
        } else if lower.contains("edit") || lower.contains("write") {
            self.hud_state.edit_count = self.hud_state.edit_count.saturating_add(1);
        }
        if let Some(current_action) = current_action {
            // Surface the currently running action so users can see the
            // concrete effect in play, not just the tool category.
            self.hud_state.last_tool = Some(current_action.to_string());
        }
    }

    /// Immediately reflect a `TodoWrite` tool result in the HUD checklist.
    ///
    /// The tool's JSON output carries `new_todos` — the authoritative list it
    /// just persisted. Parsing it here lets the sidebar update the instant the
    /// result block lands, rather than waiting for the next ~330 ms live-snapshot
    /// poll to re-read `.zo-todos.json`. A non-`TodoWrite` body (no
    /// `new_todos` field) is ignored. When every item is `Completed`, the list
    /// is cleared immediately so a finished plan does not linger in the HUD or
    /// the live todo panel.
    fn apply_todo_tool_result(&mut self, body: &runtime::message_stream::ToolResultBody) {
        use runtime::message_stream::{TodoResultStatus, ToolResultBody};
        // The runtime formats a `TodoWrite`/`TaskList` result into the typed
        // `Todos` body (see `format_todos_result`); map it straight into the HUD
        // checklist so the sidebar updates the instant the result lands. Older
        // `Generic`/`Text` JSON bodies (and tests) are still parsed below.
        if let ToolResultBody::Todos(todos) = body {
            let items: Vec<crate::tui::hud::TodoChecklistItem> = todos
                .iter()
                .map(|item| crate::tui::hud::TodoChecklistItem {
                    // The provider-neutral typed result omits Zo's optional
                    // correlation id; the next store poll restores it.
                    step_id: None,
                    content: item.content.clone(),
                    active_form: item.active_form.clone(),
                    status: match item.status {
                        TodoResultStatus::Pending => crate::tui::hud::TodoChecklistStatus::Pending,
                        TodoResultStatus::InProgress => {
                            crate::tui::hud::TodoChecklistStatus::InProgress
                        }
                        TodoResultStatus::Completed => {
                            crate::tui::hud::TodoChecklistStatus::Completed
                        }
                    },
                })
                .collect();
            self.apply_todo_items_to_hud(items);
            self.todo_touched_this_turn = true;
            return;
        }
        let content = match body {
            ToolResultBody::Generic { content, .. } | ToolResultBody::Text { content, .. } => {
                content.as_str()
            }
            _ => return,
        };
        // Cheap guard: only attempt a parse when the payload actually looks like
        // a TodoWrite result, so ordinary tool output never pays JSON cost.
        if !content.contains("new_todos") {
            return;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(content) else {
            return;
        };
        let Some(new_todos) = value.get("new_todos").and_then(|v| v.as_array()) else {
            return;
        };
        let items: Vec<crate::tui::hud::TodoChecklistItem> = new_todos
            .iter()
            .filter_map(parse_todo_checklist_item)
            .collect();
        self.apply_todo_items_to_hud(items);
        self.todo_touched_this_turn = true;
    }

    fn apply_todo_items_to_hud(&mut self, items: Vec<crate::tui::hud::TodoChecklistItem>) {
        use crate::tui::hud::TodoChecklistStatus;
        if !items.is_empty()
            && items
                .iter()
                .all(|item| item.status == TodoChecklistStatus::Completed)
        {
            self.hud_state.todo_items.clear();
            self.hud_state.todo_summary = None;
            return;
        }
        self.hud_state.todo_summary = crate::tui::hud::active_todo_summary(&items);
        self.hud_state.todo_items = items;
    }

    /// Delete a finished plan from the session todo store at turn settle, so a
    /// fully-completed checklist does not linger as a "ghost" in the sidebar
    /// (and reappear in the live panel / transcript on the next, unrelated
    /// turn). This is the store-clear the panel/transcript suppression logic
    /// already documents but nothing previously performed: the rendering layer
    /// only *hid* completed snapshots, while the underlying `<session>.todos.json`
    /// kept the rows and every HUD poll resurrected them.
    ///
    /// Only fires when the plan is non-empty and *every* item is completed —
    /// genuinely outstanding work (pending/in-progress) survives across turns,
    /// matching the persistent-checklist contract. The HUD copy is cleared in
    /// the same breath so the sidebar drops it instantly, without waiting for
    /// the next ~330 ms store poll.
    fn clear_completed_plan_store(&mut self) {
        use crate::tui::hud::TodoChecklistStatus;
        let items = &self.hud_state.todo_items;
        if items.is_empty()
            || items
                .iter()
                .any(|item| item.status != TodoChecklistStatus::Completed)
        {
            return;
        }
        if let Some(store_path) =
            crate::tui::hud::todo_store_path_for_hud(Some(self.hud_state.cwd.as_path()))
        {
            // An empty JSON array is the store's "no todos" encoding (the writer
            // and the HUD reader both treat it as empty). Leave a real file in
            // place rather than removing it, so the next `TodoWrite` and any
            // concurrent reader see a well-formed, empty store.
            if let Err(err) = std::fs::write(&store_path, b"[]") {
                eprintln!(
                    "[zo] warning: failed to reset todo store at {}: {err}",
                    store_path.display()
                );
            }
        }
        self.hud_state.todo_items.clear();
        self.hud_state.todo_summary = None;
        self.todo_touched_this_turn = false;
    }

    /// Replace the running-agent count and the live todo list. Called
    /// from the render-tick during a turn so the HUD reflects newly
    /// spawned/finished agents and just-written `.zo-todos.json`
    /// updates without waiting for the next sync barrier.
    pub fn update_hud_live_snapshot(
        &mut self,
        running_agents: u16,
        todo_items: Vec<crate::tui::hud::TodoChecklistItem>,
        agents: Vec<crate::tui::hud::AgentTaskSummary>,
        workflow: Option<crate::tui::workflow_progress::WorkflowSummary>,
    ) {
        let had_agent_rows = !agents.is_empty();
        // Feed the transcript's live agent tree from the same scan — before the
        // sidebar's terminal filter, because the tree needs terminal rows for
        // its completion-order `⎿ Done` flips.
        self.refresh_agent_batch(&agents);
        let live_count = agents
            .iter()
            .filter(|agent| !agent_status_is_terminal(&agent.status))
            .count();
        self.hud_state.running_agents = if had_agent_rows {
            u16::try_from(live_count).unwrap_or(u16::MAX)
        } else {
            running_agents
        };
        // Finished plans should not linger in the HUD/live panel. Mixed lists
        // are kept; all-completed lists are cleared by `apply_todo_items_to_hud`.
        self.apply_todo_items_to_hud(todo_items);
        // A delegating main turn is not idle while its agents make progress.
        // Reset the spinner's stall clock whenever any live agent's heartbeat
        // (`lastActivityAt`) advances — a tool ran, output streamed, or a phase
        // changed — so the row reads "Delegating · N agents" instead of a false
        // "no output Ns". Conservative by construction: a hung / rate-limited
        // swarm stops bumping `lastActivityAt`, the max stays put, and the badge
        // still surfaces (each row also self-labels `stalled Nm ago`).
        let max_heartbeat = agents
            .iter()
            .filter(|agent| !agent_status_is_terminal(&agent.status))
            .filter_map(|agent| agent.last_activity_at)
            .max()
            .unwrap_or(0);
        if max_heartbeat > self.last_agent_heartbeat {
            self.last_agent_heartbeat = max_heartbeat;
            if let Some(activity) = self.turn_activity.as_mut() {
                activity.mark_event();
            }
        }
        // While ANY agent is still live, keep just-finished siblings in the
        // panel data (the manifest scan already drops them a few seconds
        // after completion): the sidebar tree / pinned panel show the
        // completed/failed flip live instead of a finishing agent vanishing
        // one frame after it completes. `running_agents` above still counts
        // non-terminal rows only. An all-terminal snapshot clears the panel
        // exactly as before — a stopped fleet must leave the frame.
        self.hud_state.agents = if live_count > 0 {
            agents
        } else {
            Vec::new()
        };
        self.hud_state.workflow = workflow.filter(|summary| summary.status.as_str() == "running");
        self.hud_state.background_tasks = self.background_task_count();
        self.hud_state.scheduled_wake = self.scheduled_wake.clone();
    }

    /// Install the foreground runtime's already-session-scoped live background
    /// process count. Session switches replace this handle before the next HUD
    /// snapshot, so counts cannot bleed between visible sessions.
    pub fn set_background_process_count(
        &mut self,
        count: runtime::task_registry::LiveBackgroundProcessCount,
    ) {
        self.background_process_count = count;
        self.hud_state.background_tasks = self.background_task_count();
    }

    /// Nonblocking, constant-time count for the visible session.
    #[must_use]
    pub fn background_task_count(&self) -> usize {
        self.background_process_count.load()
    }

    /// Whether the HUD currently tracks any live (non-terminal) sub-agents —
    /// e.g. to route the progress-viewer key to their feed when no workflow
    /// is active. Panel data may also carry just-finished agents inside the
    /// scanner's terminal grace window; those don't count as live.
    #[must_use]
    pub fn has_live_agents(&self) -> bool {
        self.hud_state
            .agents
            .iter()
            .any(|agent| !agent_status_is_terminal(&agent.status))
    }

    /// Attach a loaded `@`-file mention usage history.
    ///
    /// Sister to [`Self::set_command_history`]; the session loop calls this
    /// at startup once the per-project data directory is known so the
    /// `@`-mention hint floats recently used files to the top.
    pub fn set_mention_history(&mut self, history: CommandHistory) {
        self.mention_history = history;
    }

    /// Scope live agent manifests to the current visible session.
    pub fn set_agent_manifest_started_after(&mut self, epoch_seconds: u64) {
        self.agent_manifest_started_after = epoch_seconds;
    }

    pub fn set_agent_manifest_session_id(&mut self, session_id: impl Into<String>) {
        let session_id = session_id.into();
        self.agent_manifest_session_id = (!session_id.trim().is_empty()).then_some(session_id);
    }

    #[must_use]
    pub fn agent_manifest_session_id(&self) -> Option<&str> {
        self.agent_manifest_session_id.as_deref()
    }

    #[must_use]
    pub const fn agent_manifest_started_after(&self) -> u64 {
        self.agent_manifest_started_after
    }

    /// Current mode.
    #[must_use]
    pub const fn mode(&self) -> AppMode {
        self.mode
    }

    /// Theme reference for tests / widget callers.
    #[must_use]
    pub const fn theme(&self) -> &Theme {
        &self.theme
    }

    /// Replace the active theme. The next draw picks up the new styles.
    ///
    /// The per-block render cache is keyed on a theme-identity fingerprint
    /// (the transcript's layout pass re-keys on a `/theme` switch), so a stale
    /// palette can no longer survive a hit. This explicit drop stays as a fast
    /// path: it also discards any mid-stream incremental prefix so an in-flight
    /// answer repaints in the new palette on the very next frame rather than
    /// after the stream settles.
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        self.transcript.invalidate_render_cache();
    }

    // ── Sidebar (Ctrl+B) ────────────────────────────────────────────

    /// Toggle the file-changes sidebar on/off.
    pub fn toggle_sidebar(&mut self) {
        self.sidebar.toggle();
    }

    /// Toggle the running-agents tree (Ctrl+A). Visible 와 별개라
    /// 사이드바 자체가 닫혀 있을 때도 상태를 기억해, 다시 열 때
    /// 마지막 상태를 복원한다.
    pub fn toggle_sidebar_agents(&mut self) {
        self.sidebar.toggle_agents();
    }

    /// Store git status as the session baseline so only new changes show.
    pub fn capture_sidebar_baseline(&mut self, snapshot: &GitStatusSnapshot) {
        self.sidebar.capture_baseline(snapshot);
    }

    /// Replace the sidebar's changed-file list with the capped snapshot and true total.
    pub fn set_changed_files(&mut self, files: Vec<ChangedFile>, total: usize) {
        self.sidebar.set_changed_files(files, total);
    }

    /// Read-only access to the sidebar state.
    #[must_use]
    pub const fn sidebar(&self) -> &SidebarState {
        &self.sidebar
    }

    // ── Search (Ctrl+F) ─────────────────────────────────────────────
    // Search behavior (enter/exit, query, match navigation) lives in
    // the `search` submodule as a separate `impl App` block.

    // ── Pager overlay ───────────────────────────────────────────────

    /// Show content in the pager overlay. If the content is short
    /// enough to fit in the transcript, push it as a system report
    /// instead. Returns `true` if the pager was opened.
    pub fn show_in_pager(&mut self, content: String, viewport_height: u16) -> bool {
        let line_count = content.lines().count();
        if line_count > usize::from(viewport_height.saturating_sub(4)) {
            self.set_pager_content(content);
            self.pager_scroll = 0;
            self.mode = AppMode::Pager;
            true
        } else {
            false
        }
    }

    /// Open the Esc-Esc rewind confirmation card with the given pre-rendered
    /// body lines (built from [`super::super::session::live_cli::LiveCli::preview_rewind`]).
    /// The destructive rewind only fires once the user answers `y`; `n`/Esc
    /// cancels with no change.
    pub fn open_rewind_confirm(&mut self, lines: Vec<String>) {
        self.rewind_confirm = Some(lines);
        self.mode = AppMode::ModalConfirmRewind;
    }

    /// Body lines for the active rewind confirmation card, if any.
    #[must_use]
    pub fn rewind_confirm_lines(&self) -> Option<&[String]> {
        self.rewind_confirm.as_deref()
    }

    /// Force the pager overlay regardless of content length — used by
    /// commands that always want the full-screen view (e.g. the `?`
    /// keybinding help). Agent detail no longer routes here: Ctrl+G opens
    /// the structured [`AgentsViewerModal`] instead.
    pub fn open_pager(&mut self, content: String) {
        self.set_pager_content(content);
        self.pager_scroll = 0;
        self.mode = AppMode::Pager;
    }

    fn set_pager_content(&mut self, content: String) {
        self.pager_lines = content.lines().map(ToOwned::to_owned).collect();
        self.pager_content = Some(content);
    }

    /// Exit the pager overlay and return to Normal.
    pub fn exit_pager(&mut self) {
        self.pager_content = None;
        self.pager_lines.clear();
        self.pager_scroll = 0;
        self.mode = AppMode::Normal;
    }

    /// Scroll the pager down by `rows`.
    pub fn pager_scroll_down(&mut self, rows: u16) {
        self.pager_scroll = self.pager_scroll.saturating_add(rows);
    }

    /// Scroll the pager up by `rows`.
    pub fn pager_scroll_up(&mut self, rows: u16) {
        self.pager_scroll = self.pager_scroll.saturating_sub(rows);
    }

    /// Open a modal. Test / internal helper; a full mode machine with
    /// allowed transitions is added in Lane L8.
    pub fn enter_mode(&mut self, mode: AppMode) {
        self.mode = mode;
    }

    pub fn set_input_text(&mut self, text: &str) {
        self.hints.slash_hidden_for = None;
        self.hints.mention_hidden_for = None;
        self.input.clear();
        for ch in text.chars() {
            self.input.insert_char(ch);
        }
    }

    pub fn toggle_focus_mode(&mut self) {
        if self.mode == AppMode::Focus {
            self.mode = AppMode::Normal;
        } else {
            self.mode = AppMode::Focus;
        }
    }

    fn drain_ready_blocks_after(&mut self, already_drained: usize) -> usize {
        let mut drained = 0;
        while already_drained + drained < Self::MAX_DRAIN_PER_TICK {
            match self.rx.try_recv() {
                Ok(block) => {
                    self.push_block(block);
                    drained += 1;
                }
                Err(_) => break,
            }
        }
        drained
    }

    /// Drain render blocks from the mpsc buffer, capped at
    /// [`MAX_DRAIN_PER_TICK`] to spread large bursts across frames.
    pub fn drain_ready_blocks(&mut self) -> usize {
        self.drain_ready_blocks_after(0)
    }

    /// Drain every block that is already queued. This is intentionally reserved
    /// for turn shutdown, where correctness beats frame spreading: a trailing
    /// `TextDelta { done: true }` must be applied before `end_turn()` paints the
    /// final no-spinner frame, or the last assistant block can look unfinished
    /// even though the provider has returned.
    pub fn drain_ready_blocks_to_idle(&mut self) -> usize {
        let mut drained = 0usize;
        while let Ok(block) = self.rx.try_recv() {
            self.push_block(block);
            drained = drained.saturating_add(1);
        }
        drained
    }

    /// Push a render block that woke the loop, then drain the rest of the
    /// immediately-ready burst under the same per-frame cap.
    pub fn drain_ready_blocks_with_first(&mut self, first: RenderBlock) -> usize {
        self.push_block(first);
        1 + self.drain_ready_blocks_after(1)
    }

    /// The per-frame render-block drain cap ([`Self::MAX_DRAIN_PER_TICK`]).
    ///
    /// Exposed so tests assert burst truncation against the real constant
    /// rather than a hardcoded literal that silently rots when the cap is
    /// retuned.
    #[must_use]
    pub const fn max_drain_per_tick() -> usize {
        Self::MAX_DRAIN_PER_TICK
    }

    /// Await the next render block from the agent channel. Suitable
    /// for use as a `tokio::select!` arm so the TUI loop wakes
    /// immediately when new streaming content arrives — rather than
    /// batching on a timer tick.
    pub async fn recv_block(&mut self) -> Option<RenderBlock> {
        self.rx.recv().await
    }

    /// Handle a bracketed-paste event by inserting the pasted text into the
    /// input widget, or into secret/custom provider setup modals when those
    /// overlays are active. Applies in Normal mode — and in the coexisting
    /// `ModalWorkflow` live monitor — regardless of whether input is enabled;
    /// when disabled the user is composing a queued message. Search mode
    /// extends the query instead, mirroring its per-character key path.
    pub fn handle_paste(&mut self, text: &str) {
        if matches!(self.mode, AppMode::ModalModel) {
            if let Some(modal) = self.active_modal_as::<ModelPickerModal>() {
                modal.paste_text(text);
            }
            return;
        }
        if matches!(self.mode, AppMode::ModalApiKey) {
            if let Some(modal) = self.active_modal_as::<ApiKeyModal>() {
                modal.paste_text(text);
            }
            return;
        }
        if matches!(self.mode, AppMode::ModalCustomProvider) {
            if let Some(modal) = self.active_modal_as::<CustomProviderWizardModal>() {
                modal.paste_text(text);
            }
            return;
        }
        if matches!(self.mode, AppMode::ModalDeepTier) {
            if let Some(modal) = self.active_modal_as::<DeepTierModal>() {
                modal.paste_text(text);
            }
            return;
        }
        // Agents-viewer message box: IME-committed text (e.g. a composed
        // Hangul syllable) reaches the TUI as a paste in several terminals,
        // so dropping it here made the box look Korean-dead.
        if matches!(self.mode, AppMode::ModalAgents) {
            if let Some(modal) = self.modals.agents.as_mut() {
                modal.paste_text(text);
            }
            return;
        }
        // Transcript search: characters extend the query directly, so
        // IME-committed text must take the same route or the search bar is
        // dead for Hangul. Control characters (a multi-line clipboard paste)
        // are stripped — a query is a single line by construction.
        if matches!(self.mode, AppMode::Search) {
            let printable: String = text.chars().filter(|ch| !ch.is_control()).collect();
            if !printable.is_empty() {
                self.search.query.push_str(&printable);
                self.refresh_search();
            }
            return;
        }
        // `ModalWorkflow` is the read-only live monitor that coexists with the
        // composer: printable keys already fall through to it so the user can
        // steer while watching (see `workflow_modal_consumes_key`). IME-committed
        // text arrives as a paste and must take the same route — otherwise
        // steering is silently dead for Hangul exactly while a workflow runs.
        if matches!(self.mode, AppMode::Normal | AppMode::ModalWorkflow) {
            if let Some((media_type, data)) = pasted_image_data_url(text) {
                if let Err(error) = self.push_clipboard_image(media_type, data) {
                    self.report_queue_limit_error(error);
                }
            } else {
                self.input.insert_text(text);
            }
        }
    }

    /// Handle a paste event that owns its text buffer. Composer pastes move the
    /// buffer into the input widget so collapsed payloads do not allocate and
    /// copy a second multi-MB body while the event allocation is still live.
    pub fn handle_paste_owned(&mut self, text: String) {
        if matches!(self.mode, AppMode::Normal | AppMode::ModalWorkflow) {
            if let Some((media_type, data)) = pasted_image_data_url(&text) {
                if let Err(error) = self.push_clipboard_image(media_type, data) {
                    self.report_queue_limit_error(error);
                }
            } else {
                self.input.insert_text_owned(text);
            }
            return;
        }
        self.handle_paste(&text);
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) -> Result<AppAction, TuiError> {
        match self.mode {
            AppMode::Normal | AppMode::Focus => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    return Ok(self.handle_transcript_wheel(mouse, true));
                }
                MouseEventKind::ScrollDown => {
                    return Ok(self.handle_transcript_wheel(mouse, false));
                }
                // Drag the transcript scrollbar with the mouse (the wheel alone
                // worked before): a press on the right-edge scrollbar column
                // starts a drag, and motion maps the row onto the scroll range
                // until release.
                MouseEventKind::Down(MouseButton::Left) => {
                    if let Some(action) = self.handle_left_press(mouse) {
                        return Ok(action);
                    }
                }
                MouseEventKind::Moved => {
                    if self.update_hovered_copy_button(mouse) {
                        return Ok(AppAction::Redraw);
                    }
                    // Hovering an agent row underlines it; repaint only on a
                    // genuine target change so raw motion never floods frames.
                    if self.update_hovered_agent(mouse) {
                        return Ok(AppAction::Redraw);
                    }
                }
                MouseEventKind::Drag(MouseButton::Left) => {
                    if self.transcript_view.scrollbar_dragging {
                        self.scroll_transcript_to_mouse(mouse);
                    } else if self.transcript_view.transcript_press.is_some()
                        && self.drag_extend_char_selection(mouse)
                    {
                        return Ok(AppAction::Redraw);
                    }
                }
                MouseEventKind::Up(MouseButton::Left) => {
                    if let Some(action) = self.finish_left_release(mouse) {
                        return Ok(action);
                    }
                }
                _ => {}
            },
            AppMode::Pager => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.pager_scroll_up(Self::MOUSE_SCROLL_ROWS);
                }
                MouseEventKind::ScrollDown => {
                    self.pager_scroll_down(Self::MOUSE_SCROLL_ROWS);
                }
                _ => {}
            },
            // Live workflow viewer (Ctrl+O): the wheel scrolls the agent pane, so
            // a long fan-out is navigable by mouse, not just PgUp/PgDn.
            AppMode::ModalWorkflow => {
                if let Some(modal) = &mut self.modals.workflow {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => modal.scroll_agents_up(Self::MOUSE_SCROLL_ROWS),
                        MouseEventKind::ScrollDown => {
                            modal.scroll_agents_down(Self::MOUSE_SCROLL_ROWS);
                        }
                        _ => {}
                    }
                }
            }
            // Agents viewer (Ctrl+G): the wheel walks the list selection, and a
            // left-click on a list row selects that agent (the modal recomputes
            // the same pure layout the draw used, so hits can't drift).
            AppMode::ModalAgents => {
                let modal_area = self
                    .regions
                    .as_ref()
                    .map(|regions| diff_modal_rect(regions, regions.transcript));
                if let (Some(modal), Some(area)) = (self.modals.agents.as_mut(), modal_area) {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => modal.scroll_list(true, 1),
                        MouseEventKind::ScrollDown => modal.scroll_list(false, 1),
                        MouseEventKind::Down(MouseButton::Left) => {
                            modal.handle_click(mouse.column, mouse.row, area, &self.theme);
                        }
                        _ => {}
                    }
                }
            }
            // Every other modal is a selection-list picker (`/model`, `/resume`,
            // `@`-file, `/tools`, …). The wheel moves the highlighted row up/down,
            // matching the arrow keys — without this the wheel was silently
            // dropped over an open picker.
            mode if mode.is_modal() => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.scroll_active_picker(true);
                }
                MouseEventKind::ScrollDown => {
                    self.scroll_active_picker(false);
                }
                _ => {}
            },
            _ => {}
        }

        Ok(AppAction::None)
    }

    /// Route a transcript wheel notch. During a held left-drag selection the
    /// notch scrolls the view and re-extends the selection head to the cell
    /// under the (stationary) pointer — the content-row anchor makes the
    /// newly revealed rows part of the gesture, and the immediate repaint
    /// (`Redraw`, never the coalesced scroll path) mines each revealed row
    /// for the release copy, so the selection grows past one screenful.
    /// Outside a drag the wheel scrolls as before; a settled highlight
    /// survives scrolling because it tracks its content rows.
    fn handle_transcript_wheel(&mut self, mouse: MouseEvent, up: bool) -> AppAction {
        let drag_selecting = self.transcript_view.transcript_press.is_some()
            && self.transcript.char_selection_active();
        self.clear_hovered_copy_button();
        if !drag_selecting && self.mouse_over_slash_hint(mouse) {
            if up {
                self.select_prev_slash_hint();
            } else {
                self.select_next_slash_hint();
            }
            return AppAction::None;
        }

        // The wheel always scrolls the transcript, even when the pointer rests
        // over the composer (idle) or while typing. The input box never consumes
        // wheel events — only the slash-hint popup above does, since the wheel
        // navigates it.
        self.prepare_user_scroll();
        if up {
            self.transcript.scroll_up(Self::MOUSE_SCROLL_ROWS);
            self.transcript_view.follow_output = false;
        } else {
            self.transcript.scroll_down(Self::MOUSE_SCROLL_ROWS);
            self.refresh_follow_output();
        }
        if drag_selecting {
            // `scroll_down` can overshoot the max offset until a draw clamps
            // it; clamp now so the head lands on the real content row.
            self.prepare_user_scroll();
            self.drag_extend_char_selection(mouse);
            return AppAction::Redraw;
        }
        AppAction::None
    }

    /// Route a wheel notch to the active slot modal. Every list picker lives on
    /// the unified slot and owns its wheel semantics via [`Modal::scroll`]
    /// (dedicated `scroll_up`/`scroll_down`, an arrow-key synthesis, or a
    /// no-op); viewer modals are legacy and carry their own wheel handling
    /// elsewhere, so the slot is empty and this is a no-op for them.
    fn scroll_active_picker(&mut self, up: bool) {
        let rows = Self::MOUSE_SCROLL_ROWS as usize;
        if let Some(modal) = self.active_modal.as_mut() {
            modal.scroll(up, rows);
        }
    }

    /// `true` when the pointer is on the slash-hint popup, the only overlay that
    /// consumes wheel events (the wheel navigates it). Anywhere else — including
    /// over the composer — the wheel scrolls the transcript.
    fn mouse_over_slash_hint(&self, mouse: MouseEvent) -> bool {
        self.slash_hint_popup_rect()
            .is_some_and(|rect| Self::rect_contains(rect, mouse.column, mouse.row))
    }

    fn mouse_over_hud(&self, mouse: MouseEvent) -> bool {
        self.regions
            .is_some_and(|regions| Self::rect_contains(regions.hud, mouse.column, mouse.row))
    }

    /// Handle a left-button press. Returns `Some(action)` when the press is
    /// consumed (return it up the loop) or `None` to fall through. Extracted
    /// from [`Self::handle_mouse`] so that dispatcher stays within the
    /// line-length lint while this press logic — HUD toggle, scrollbar drag,
    /// agent-row click, aggregate-panel click, copy button, char selection —
    /// stays in one readable place.
    fn handle_left_press(&mut self, mouse: MouseEvent) -> Option<AppAction> {
        // A press anywhere drops the previous drag-selection highlight (kept on
        // screen after the last copy), so the old wash never lingers under a new
        // gesture.
        let cleared_selection = self.transcript.clear_char_selection();
        if self.mouse_over_hud(mouse) {
            self.toggle_sidebar();
            self.clear_hovered_copy_button();
            return Some(AppAction::Redraw);
        }
        if self.mouse_over_transcript_scrollbar(mouse) {
            self.transcript_view.scrollbar_dragging = true;
            self.clear_hovered_copy_button();
            self.scroll_transcript_to_mouse(mouse);
        } else if let Some(id) = self.agent_id_at_mouse(mouse) {
            // A click on a specific agent row opens the live agent view focused
            // on THAT agent. Checked before the whole-panel fallback below so a
            // row click is per-agent, while the header / overflow line / gaps
            // still open the aggregate view.
            self.clear_hovered_copy_button();
            return Some(AppAction::OpenAgentInViewer(id));
        } else if self.mouse_over_agent_panel(mouse) {
            // The pinned live-agent panel is an overlay, not selectable
            // transcript text: a click on it opens the live agent view (same
            // surface as Ctrl+O).
            self.clear_hovered_copy_button();
            return Some(AppAction::OpenWorkflowViewer);
        } else if let Some(text) = self.copy_text_if_mouse_over_copy_button(mouse) {
            return Some(AppAction::ClipboardCopyBlock(text));
        } else if let Some(rect) = self
            .transcript_view_rect()
            .filter(|rect| Self::rect_contains(*rect, mouse.column, mouse.row))
        {
            let block = self.block_id_at_mouse(mouse);
            self.clear_hovered_copy_button();
            self.transcript
                .begin_char_selection(mouse.column, mouse.row.saturating_sub(rect.y));
            self.transcript_view.transcript_press = Some(TranscriptPress { block });
        }
        cleared_selection.then_some(AppAction::Redraw)
    }

    /// `true` when the pointer is inside the pinned live-agent panel painted by
    /// the most recent frame — the click target that opens the live agent view.
    fn mouse_over_agent_panel(&self, mouse: MouseEvent) -> bool {
        self.agent_panel_click_rect
            .is_some_and(|rect| Self::rect_contains(rect, mouse.column, mouse.row))
    }
    /// The agent id whose pinned-panel row is under the pointer, or `None`.
    /// Scans the per-frame `agent_row_click_targets`; the first containing rect
    /// wins (rows never overlap). Reads in-memory state only — no disk I/O — so
    /// it is safe on the interaction hot path.
    fn agent_id_at_mouse(&self, mouse: MouseEvent) -> Option<String> {
        self.agent_row_click_targets
            .iter()
            .find(|(rect, _)| Self::rect_contains(*rect, mouse.column, mouse.row))
            .map(|(_, id)| id.clone())
    }

    /// Update `hovered_agent` to the agent row under the pointer. Returns `true`
    /// only when the hovered target actually changed, so the caller repaints on
    /// a genuine hover transition and raw motion within the same row (or over
    /// empty space, repeatedly) never forces a frame.
    fn update_hovered_agent(&mut self, mouse: MouseEvent) -> bool {
        let next = self.agent_id_at_mouse(mouse);
        if next == self.hovered_agent {
            return false;
        }
        self.hovered_agent = next;
        true
    }

    /// End a left-button gesture. A dragged character selection copies the
    /// visible cells mined by the last draw and keeps its highlight until the
    /// next press. Agent cards still open their viewer and collapsible blocks
    /// toggle, but a plain click on prose never writes to the clipboard.
    /// Clicks on empty space or a different release block do nothing.
    fn finish_left_release(&mut self, mouse: MouseEvent) -> Option<AppAction> {
        self.transcript_view.scrollbar_dragging = false;
        let press = self.transcript_view.transcript_press.take();
        let was_dragged = self.transcript.has_char_selection();
        if let Some(text) = self.transcript.finish_char_selection() {
            return Some(AppAction::ClipboardCopyBlock(text));
        }
        if was_dragged {
            return Some(AppAction::Redraw);
        }
        let anchor = press.and_then(|press| press.block)?;
        if self.block_id_at_mouse(mouse) != Some(anchor) {
            return None;
        }
        if self.block_opens_agent_view(anchor) {
            return Some(AppAction::OpenWorkflowViewer);
        }
        // Collapsed rows open in place on click (CC parity): a collapsed
        // tool-group summary reveals its rows, a tool row/result toggles the
        // clipped result body, a settled `step` trace expands the reasoning.
        if self.transcript.toggle_expand_for_click(anchor) {
            return Some(AppAction::Redraw);
        }
        // Copying is explicit via the hover copy button, drag selection, or
        // key bindings; a plain click must never touch the system clipboard.
        Some(AppAction::None)
    }

    /// Whether the transcript block is a spawn-family / `Workflow` tool card —
    /// the rows that host the inline agent tree, and therefore the ones a mouse
    /// click should answer by opening the live agent view.
    fn block_opens_agent_view(&self, id: BlockId) -> bool {
        self.transcript.blocks().iter().any(|block| {
            matches!(
                block,
                RenderBlock::ToolCall { id: block_id, name, .. }
                    if *block_id == id && tool_call::opens_agent_batch(name)
            )
        })
    }

    fn hovered_copy_block_id(&self) -> Option<BlockId> {
        self.transcript_view.hovered_copy_button.map(|hover| hover.block_id)
    }

    /// Stable id of the transcript block under the mouse, or `None` when the
    /// pointer is outside the transcript region or over empty space. A press
    /// retains this only for the release-over-same-block click semantics.
    fn block_id_at_mouse(&mut self, mouse: MouseEvent) -> Option<BlockId> {
        let rect = self.transcript_view_rect()?;
        if !Self::rect_contains(rect, mouse.column, mouse.row) {
            return None;
        }
        let row_in_viewport = mouse.row.saturating_sub(rect.y);
        self.transcript.block_id_at_viewport_row(
            row_in_viewport,
            &self.theme,
            rect.width,
            self.image_protocol,
        )
    }

    /// Extend a left-drag character selection to the pointer's cell.
    /// Coordinates outside the transcript are clamped to its nearest edge;
    /// the transcript stores the row as a content row against its current
    /// scroll, so a wheel notch mid-drag (which scrolls first, then lands
    /// here) grows the selection past the viewport instead of breaking it.
    /// Returns whether the highlight changed and needs a repaint.
    fn drag_extend_char_selection(&mut self, mouse: MouseEvent) -> bool {
        let Some(rect) = self.transcript_view_rect() else {
            return false;
        };
        if rect.width == 0 || rect.height == 0 {
            return false;
        }
        let right = rect.x.saturating_add(rect.width - 1);
        let bottom = rect.y.saturating_add(rect.height - 1);
        let col = mouse.column.clamp(rect.x, right);
        let row = mouse.row.clamp(rect.y, bottom);
        self.transcript
            .extend_char_selection(col, row.saturating_sub(rect.y))
    }

    fn update_hovered_copy_button(&mut self, mouse: MouseEvent) -> bool {
        let next = self.transcript_copy_affordance_at_mouse(mouse);
        if self.transcript_view.hovered_copy_button == next {
            return false;
        }
        self.transcript_view.hovered_copy_button = next;
        true
    }

    fn clear_hovered_copy_button(&mut self) -> bool {
        let had_hover = self.transcript_view.hovered_copy_button.is_some();
        self.transcript_view.hovered_copy_button = None;
        had_hover
    }

    fn copy_text_if_mouse_over_copy_button(&mut self, mouse: MouseEvent) -> Option<String> {
        if let Some(hover) = self.transcript_view.hovered_copy_button {
            if Self::rect_contains(hover.button, mouse.column, mouse.row) {
                return self.transcript.copy_text_for_block_id(hover.block_id);
            }
        }

        let hit = self.transcript_copy_affordance_at_mouse(mouse)?;
        if !Self::rect_contains(hit.button, mouse.column, mouse.row) {
            return None;
        }
        self.transcript_view.hovered_copy_button = Some(HoveredCopyButton {
            block_id: hit.block_id,
            button: hit.button,
        });
        self.transcript.copy_text_for_block_id(hit.block_id)
    }

    fn transcript_copy_affordance_at_mouse(
        &mut self,
        mouse: MouseEvent,
    ) -> Option<HoveredCopyButton> {
        let rect = self.transcript_view_rect()?;
        if !Self::rect_contains(rect, mouse.column, mouse.row) {
            return None;
        }
        let row_in_viewport = mouse.row.saturating_sub(rect.y);
        let hit = self.transcript.copy_affordance_at_viewport_row(
            row_in_viewport,
            rect,
            &self.theme,
            self.image_protocol,
        )?;
        Some(HoveredCopyButton {
            block_id: hit.block_id,
            button: hit.button,
        })
    }

    /// Return the active transcript viewport rectangle — the rect the body was
    /// actually drawn into (bottom overlay reservations and startup banner
    /// already subtracted). Falls back to deriving from the full region only
    /// before the first draw. Interaction paths (wheel, follow-tail, scrollbar
    /// drag, row→block mapping) must share the drawn viewport or scroll clamps
    /// leave the tail rows stranded behind the pinned panels.
    fn transcript_view_rect(&self) -> Option<Rect> {
        if let Some(rect) = self.transcript_draw_rect {
            return Some(rect);
        }
        let regions = self.regions?;
        let rect = regions.transcript;
        if self.startup.screen.is_some() && !self.transcript.is_empty() {
            let banner_h = super::startup::preferred_height(rect.width).min(rect.height);
            let below_h = rect.height.saturating_sub(banner_h);
            return Some(Rect::new(rect.x, rect.y + banner_h, rect.width, below_h));
        }
        Some(rect)
    }

    fn rect_contains(rect: Rect, column: u16, row: u16) -> bool {
        rect.width > 0
            && rect.height > 0
            && column >= rect.x
            && column < rect.x + rect.width
            && row >= rect.y
            && row < rect.y + rect.height
    }

    /// `true` when the pointer is on the transcript's right-edge scrollbar
    /// column (only drawn when content overflows). The scrollbar lives in the
    /// last column of `regions.transcript` (see `draw_scroll_indicator`).
    fn mouse_over_transcript_scrollbar(&self, mouse: MouseEvent) -> bool {
        if let Some(rect) = self.transcript_view_rect() {
            rect.width > 0
                && rect.height > 0
                && mouse.column == rect.x + rect.width - 1
                && mouse.row >= rect.y
                && mouse.row < rect.y + rect.height
        } else {
            false
        }
    }

    /// Map a scrollbar drag at the mouse's row onto the transcript scroll range.
    /// The row is clamped to the transcript so a drag that wanders off the
    /// column (or past the bottom into the input) still tracks sensibly.
    fn scroll_transcript_to_mouse(&mut self, mouse: MouseEvent) {
        let Some(rect) = self.transcript_view_rect() else {
            return;
        };
        if rect.height == 0 {
            return;
        }
        let row_in_viewport = mouse.row.saturating_sub(rect.y).min(rect.height - 1);
        // Normalize the tail sentinel first so the mapping starts from a real
        // offset (same precondition as the wheel path).
        self.prepare_user_scroll();
        self.transcript.scroll_to_viewport_row(
            row_in_viewport,
            rect.height,
            &self.theme,
            rect.width,
            self.image_protocol,
        );
        // Following the tail is on only when the drag lands at the bottom.
        self.refresh_follow_output();
    }

    /// Normalize the transcript's sentinel tail scroll to a real offset
    /// so the next `scroll_up` actually moves the viewport.
    fn prepare_user_scroll(&mut self) {
        if let Some(rect) = self.transcript_view_rect() {
            self.transcript.clamp_scroll_to_content(
                rect.height,
                &self.theme,
                rect.width,
                self.image_protocol,
            );
        }
    }

    fn refresh_follow_output(&mut self) {
        self.transcript_view.follow_output = self.transcript_view_rect().is_none_or(|rect| {
            self.transcript
                .is_at_bottom(rect.height, &self.theme, rect.width, self.image_protocol)
        });
    }
}

fn epoch_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn agent_status_is_terminal(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "stopped")
}

/// Parse one `new_todos` entry from a `TodoWrite` result into a renderable
/// checklist item. Tolerant of missing fields (falls back to `content` for the
/// active form, `Pending` for an unknown status) so a slightly-off payload
/// still surfaces rather than vanishing.
fn parse_todo_checklist_item(
    value: &serde_json::Value,
) -> Option<crate::tui::hud::TodoChecklistItem> {
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};
    let content = value.get("content").and_then(|v| v.as_str())?.to_string();
    let step_id = value
        .get("stepId")
        .or_else(|| value.get("step_id"))
        .and_then(|v| v.as_str())
        .filter(|id| !id.trim().is_empty())
        .map(str::to_string);
    let active_form = value
        .get("activeForm")
        .or_else(|| value.get("active_form"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map_or_else(|| content.clone(), str::to_string);
    let status = match value.get("status").and_then(|v| v.as_str()) {
        Some("in_progress") => TodoChecklistStatus::InProgress,
        Some("completed") => TodoChecklistStatus::Completed,
        _ => TodoChecklistStatus::Pending,
    };
    Some(TodoChecklistItem {
        step_id,
        content,
        status,
        active_form,
    })
}

/// The activity/spinner line shown while the model is reasoning. Echoes the
/// model's own first reasoning line (SSOT: [`crate::tui::blocks::reasoning::reasoning_summary_line`])
/// so a Korean (or any non-English) thinking stream shows its real topic instead
/// of collapsing every non-English reasoning to a hardcoded English bucket
/// ("Working through the request"). Falls back to a neutral cue only before any
/// usable line has streamed.
const MAX_REASONING_TITLE_SOURCE_CHARS: usize = 512;

fn reasoning_activity_summary(text: &str) -> String {
    let line = crate::tui::blocks::reasoning::reasoning_summary_line(text);
    if line.is_empty() {
        // No usable reasoning line yet — fall back to the Zo cue (`Thinking…`)
        // rather than a generic `Thinking…`, matching the streaming reasoning
        // line's metaphor.
        format!("{}…", crate::tui::blocks::reasoning::ZO_REVEAL_VERBS[0])
    } else {
        line
    }
}

fn append_reasoning_title_source_fragment(prior_source: &str, delta: &str) -> String {
    bounded_first_non_empty_line_source(prior_source.chars().chain(delta.chars()))
}

fn bounded_first_non_empty_line_source<I>(chars: I) -> String
where
    I: IntoIterator<Item = char>,
{
    let mut source = String::new();
    let mut remaining = MAX_REASONING_TITLE_SOURCE_CHARS;
    let mut started = false;
    for ch in chars {
        if !started {
            if ch.is_whitespace() {
                continue;
            }
            started = true;
        }
        if ch == '\n' {
            source.push(ch);
            break;
        }
        if remaining == 0 {
            break;
        }
        source.push(ch);
        remaining -= 1;
    }
    source
}

fn reasoning_first_non_empty_line_is_terminated(text: &str) -> bool {
    for line in text.split_inclusive('\n') {
        let terminated = line.ends_with('\n');
        let line = line.strip_suffix('\n').unwrap_or(line);
        if !line.trim().is_empty() {
            return terminated;
        }
    }
    false
}

/// Reconstruct only the slice of `prior + delta` that `reasoning_summary_line`
/// needs — the **first non-empty line** of the accumulated reasoning — without
/// copying the whole thought on every delta.
///
/// `prior` is the reasoning merged so far (excluding `delta`); `delta` is the
/// just-arrived chunk. The title is the first non-empty line of `prior + delta`,
/// so once `prior` already contains a completed non-empty line (a non-blank line
/// followed by a newline) that line is final and `delta` cannot change it —
/// return `prior` untouched (borrowed, zero-copy). Until then the first line is
/// still open, so only a bounded prefix of the last unterminated line plus the
/// new delta can affect the 72-character spinner title.
fn reasoning_title_source<'a>(prior: &'a str, delta: &str) -> std::borrow::Cow<'a, str> {
    // A completed non-empty line exists iff some non-blank line is followed by a
    // newline. Only scan the title budget: if the first line is longer than that,
    // a bounded prefix is already enough for `reasoning_summary_line`'s 72-char
    // title, and scanning the rest would reintroduce O(n²) CPU on no-newline
    // reasoning streams.
    let mut line_start = 0;
    let mut line_has_nonblank = false;
    let mut hit_scan_budget_with_blank_prefix = false;
    for (seen, (idx, ch)) in prior.char_indices().enumerate() {
        if seen >= MAX_REASONING_TITLE_SOURCE_CHARS {
            hit_scan_budget_with_blank_prefix = !line_has_nonblank;
            break;
        }
        if ch == '\n' {
            if line_has_nonblank {
                // First non-empty line is already terminated in `prior`: final.
                return std::borrow::Cow::Borrowed(prior);
            }
            line_start = idx + ch.len_utf8();
            line_has_nonblank = false;
        } else if !ch.is_whitespace() {
            line_has_nonblank = true;
        }
    }

    // No bounded, terminated non-empty line yet. Build at most the title-source
    // budget from the current candidate line plus the new delta; both scan and
    // allocation are now independent of the accumulated reasoning size. If the
    // scan budget was consumed entirely by blank prefix, let the fresh delta
    // have the budget so the eventual first non-empty title can appear.
    let copy_from = if hit_scan_budget_with_blank_prefix {
        prior.len()
    } else {
        line_start
    };
    std::borrow::Cow::Owned(bounded_first_non_empty_line_source(
        prior[copy_from..].chars().chain(delta.chars()),
    ))
}

mod paste;
use paste::pasted_image_data_url;

/// Build the active reasoning-effort badge for the idle rule above the input.
/// The caller uses the returned display width to stop the hairline one clear
/// cell before the badge while keeping the badge's right edge fixed.
fn effort_rule_badge(state: &HudState, theme: &Theme) -> Option<(Line<'static>, u16)> {
    let label = hud::effort_badge_label(state.effort, &state.model.alias)?;
    let effort = state.effort?;
    let badge = format!(" {label} ");
    let badge_width = u16::try_from(badge.chars().count()).unwrap_or(u16::MAX);
    let badge_style = match effort {
        Effort::Max | Effort::Ultra | Effort::Smart => Style::default()
            .fg(theme.palette.accent)
            .add_modifier(Modifier::BOLD),
        Effort::High | Effort::Xhigh => Style::default().fg(theme.palette.warn),
        Effort::Low | Effort::Medium | Effort::Off => Style::default().fg(theme.palette.dim),
    };
    Some((
        Line::from(Span::styled(badge, badge_style)),
        badge_width,
    ))
}

/// Draw a prebuilt effort badge at the unchanged right-aligned position.
fn draw_effort_rule_badge(
    frame: &mut ratatui::Frame<'_>,
    line_area: Rect,
    badge: Line<'static>,
    badge_width: u16,
) {
    let badge_rect = Rect::new(
        line_area.x + line_area.width.saturating_sub(badge_width),
        line_area.y,
        badge_width,
        1,
    );
    // Clear only the badge cells (not the whole input box) before painting.
    frame.render_widget(Clear, badge_rect);
    frame.render_widget(Paragraph::new(badge), badge_rect);
}

/// Restore one repo-relative path to its `HEAD` content in the process
/// cwd's git tree. Errors (e.g. path absent in `HEAD`) carry git's stderr
/// so the caller can surface why the revert was refused.
fn git_checkout_head(path: &str) -> Result<(), String> {
    let output = std::process::Command::new("git")
        .args(["checkout", "HEAD", "--", path])
        .output()
        .map_err(|error| format!("failed to run git: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

/// `git diff HEAD --no-color` text in the process cwd, empty on failure —
/// mirrors the `/diff` command's read so the refreshed viewer matches.
fn git_diff_head() -> String {
    std::process::Command::new("git")
        .args(["diff", "HEAD", "--no-color"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// Collect relative file paths under the current working directory for
/// the `@`-triggered [`FilePickerModal`].
///
/// Walks with the `ignore` crate's standard filters so the completion
/// list honours `.gitignore` / `.ignore` and skips hidden + VCS entries —
/// build/coverage/`target` output never pollutes the picker. The result is
/// capped so a huge tree can never stall the modal; paths use forward
/// slashes and are sorted for a stable, predictable list. Returns an empty
/// vec when the cwd is unreadable.
fn collect_workspace_files(cancel: &ScanCancelToken) -> Vec<String> {
    let Ok(root) = std::env::current_dir() else {
        return Vec::new();
    };
    collect_workspace_files_in(&root, cancel)
}

/// `@`-picker file walk rooted at an explicit directory — the testable core
/// of [`collect_workspace_files`], which only supplies the process cwd.
///
/// Uses `ignore::WalkBuilder`'s `standard_filters(true)`: `.gitignore`,
/// `.ignore`, global/exclude ignores, hidden-entry skipping, and the
/// `.git/` directory — the same filter set the glob/grep tools use, so the
/// completion list matches them and never surfaces ignored build output.
fn collect_workspace_files_in(root: &Path, cancel: &ScanCancelToken) -> Vec<String> {
    use ignore::WalkBuilder;

    /// Hard cap on returned files — keeps the modal responsive on large repos.
    const MAX_FILES: usize = 2000;

    if scan_cancelled(cancel) {
        return Vec::new();
    }

    let mut files: Vec<String> = Vec::new();
    let walker = WalkBuilder::new(root).standard_filters(true).build();
    for entry in walker {
        if scan_cancelled(cancel) {
            return Vec::new();
        }
        let Ok(entry) = entry else {
            continue;
        };
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        if let Ok(rel) = entry.path().strip_prefix(root) {
            files.push(rel.to_string_lossy().replace('\\', "/"));
            if files.len() >= MAX_FILES {
                break;
            }
        }
    }

    if scan_cancelled(cancel) {
        return Vec::new();
    }

    files.sort();
    files
}

#[cfg(test)]
mod frecency_wiring_tests {
    //! Regression guard for the slash/`@`-file frecency stores. The old
    //! constructor bound them to throwaway `/tmp` dummy paths and `mention_history`
    //! had no setter, so "recently used floats up" ran on empty data every
    //! session. These tests prove a recorded entry survives a reload and that
    //! the setters attach it where the hints read it.

    use super::{AgentCommand, App, CommandHistory, PathBuf, RenderBlock, Theme, mpsc};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn app() -> App {
        let (_block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
        let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
        App::new(Theme::no_color(), block_rx, cmd_tx)
    }

    fn temp_jsonl(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "zo-frecency-{label}-{}-{nanos}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn recorded_command_reloads_and_attaches() {
        let path = temp_jsonl("cmd");
        // First session: record a usage, then drop so it flushes to disk.
        {
            let mut hist = CommandHistory::load(&path).expect("load command history");
            hist.record("/commit").expect("record command");
        }
        // Next session: reload from the same path and attach via the setter.
        let reloaded = CommandHistory::load(&path).expect("reload command history");
        let mut app = app();
        app.set_command_history(reloaded);

        assert!(
            app.command_history
                .frecency_scores()
                .contains_key("/commit"),
            "recorded command must survive reload and be attached for frecency"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recorded_mention_reloads_and_attaches() {
        let path = temp_jsonl("mention");
        {
            let mut hist = CommandHistory::load(&path).expect("load mention history");
            hist.record("src/config.rs").expect("record mention");
        }
        let reloaded = CommandHistory::load(&path).expect("reload mention history");
        let mut app = app();
        // The setter must exist (it did not before this fix) and wire the data.
        app.set_mention_history(reloaded);

        assert!(
            app.mention_history
                .frecency_scores()
                .contains_key("src/config.rs"),
            "recorded mention must survive reload and be attached for frecency"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn default_stores_are_inert_not_tmp_backed() {
        // The constructor default must read empty (no leaked /tmp dummy data)
        // until the session loop attaches the real per-user/project files.
        let app = app();
        assert!(
            app.command_history.frecency_scores().is_empty(),
            "default command history must be empty, not /tmp-backed"
        );
        assert!(
            app.mention_history.frecency_scores().is_empty(),
            "default mention history must be empty, not /tmp-backed"
        );
    }
}
