//! Plain value types backing the [`App`](super::App) state machine.
//!
//! Split out from `app/mod.rs` so the enums and small structs that
//! describe modes, key-event outcomes, and clipboard payloads can be
//! read independently of the (much larger) `App` implementation.

use runtime::PermissionMode as RuntimePermissionMode;
use runtime::message_stream::{ActiveModel, AgentResultStatus};

use crate::tui::modals::{CustomProviderDraft, SmartSettingsCommit};

/// Scheduler that owns the next visible wakeup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeSource {
    Wakeup,
    Loop,
}

/// App-owned absolute deadline used by the HUD and sidebar countdowns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledWakeHud {
    pub due_at_epoch: u64,
    pub reason: String,
    pub source: WakeSource,
}

/// The TUI's top-level mode state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Normal input + transcript view.
    Normal,
    /// `/model` picker modal is open.
    ModalModel,
    /// `/permissions` picker modal is open.
    ModalPermissions,
    /// A blocking choice modal (e.g. permission prompt) is open.
    ModalChoice,
    /// A blocking `AskUserQuestion` modal is open.
    ModalQuestion,
    /// `/resume` session picker modal is open.
    ModalSession,
    /// `/login` · `/connect` provider picker modal is open.
    ModalLogin,
    /// `/connect` API-key setup modal for cloud OpenAI-compatible adapters.
    ModalApiKey,
    /// `/connect` custom OpenAI-compatible provider wizard is open.
    ModalCustomProvider,
    /// `/effort` interactive slider modal is open.
    ModalEffort,
    /// `/diff` interactive diff viewer modal is open.
    ModalDiff,
    /// `/hunks` human/agent attribution review modal is open.
    ModalHunks,
    /// `@`-triggered fuzzy file-reference picker is open.
    ModalFile,
    /// Interactive snapshot rewind/diff viewer is open (Ctrl+R).
    ModalRewind,
    /// Esc-Esc rewind confirmation card is open. Gates the destructive
    /// combined (code + conversation) rewind behind an explicit y/n so a
    /// mistaken double-tap cannot silently discard the latest turn's edits.
    ModalConfirmRewind,
    /// Live workflow progress viewer is open (Ctrl+O).
    ModalWorkflow,
    /// Session agents viewer is open (Ctrl+G): flat list + detail over every
    /// sub-agent manifest, live and finished.
    ModalAgents,
    /// `TeamInbox` viewer is open (`/inbox`).
    ModalTeamInbox,
    /// `/tools` runtime tool toggle modal is open.
    ModalTools,
    /// `/usage` graphical token/cost dashboard modal is open.
    ModalUsage,
    /// Generic slash-command report popup is open (`/mcp`, `/doctor`, …).
    ModalReport,
    /// `/smart` large Smart Router settings dashboard modal is open.
    ModalSmartSettings,
    /// `/tier` ordered deep-model pool picker is open.
    ModalDeepTier,
    /// `/remote` onboarding and status modal is open.
    ModalRemoteOnboarding,
    /// Generic single-select picker for a slash command's fixed-choice
    /// argument (e.g. `/theme`, `/plan`, `/fast`). Reuses `ChoicePickerModal`;
    /// the chosen label is re-submitted as `/<command> <label>` so the
    /// command's existing text handler applies it.
    ModalArgPick,
    /// Transcript text search overlay (Ctrl+F).
    Search,
    /// Full-screen pager for long output (scrollable overlay).
    Pager,
    /// Focus view — transcript only, no input/HUD. Toggle with F11 or `/focus`.
    Focus,
}

impl AppMode {
    /// `true` if any modal is currently on top of the transcript.
    #[must_use]
    pub const fn is_modal(self) -> bool {
        !matches!(self, Self::Normal | Self::Focus)
    }

    /// `true` if the mode is an overlay that takes over the full screen.
    #[must_use]
    pub const fn is_overlay(self) -> bool {
        matches!(self, Self::Pager | Self::Focus)
    }
}

impl std::fmt::Display for AppMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => f.write_str("Normal"),
            Self::ModalModel => f.write_str("ModalModel"),
            Self::ModalPermissions => f.write_str("ModalPermissions"),
            Self::ModalChoice => f.write_str("ModalChoice"),
            Self::ModalQuestion => f.write_str("ModalQuestion"),
            Self::ModalSession => f.write_str("ModalSession"),
            Self::ModalLogin => f.write_str("ModalLogin"),
            Self::ModalApiKey => f.write_str("ModalApiKey"),
            Self::ModalCustomProvider => f.write_str("ModalCustomProvider"),
            Self::ModalEffort => f.write_str("ModalEffort"),
            Self::ModalDiff => f.write_str("ModalDiff"),
            Self::ModalHunks => f.write_str("ModalHunks"),
            Self::ModalFile => f.write_str("ModalFile"),
            Self::ModalRewind => f.write_str("ModalRewind"),
            Self::ModalConfirmRewind => f.write_str("ModalConfirmRewind"),
            Self::ModalWorkflow => f.write_str("ModalWorkflow"),
            Self::ModalAgents => f.write_str("ModalAgents"),
            Self::ModalTeamInbox => f.write_str("ModalTeamInbox"),
            Self::ModalTools => f.write_str("ModalTools"),
            Self::ModalUsage => f.write_str("ModalUsage"),
            Self::ModalReport => f.write_str("ModalReport"),
            Self::ModalSmartSettings => f.write_str("ModalSmartSettings"),
            Self::ModalDeepTier => f.write_str("ModalDeepTier"),
            Self::ModalRemoteOnboarding => f.write_str("ModalRemoteOnboarding"),
            Self::ModalArgPick => f.write_str("ModalArgPick"),
            Self::Search => f.write_str("Search"),
            Self::Pager => f.write_str("Pager"),
            Self::Focus => f.write_str("Focus"),
        }
    }
}

/// Agent command currency flowing out of the TUI.
///
/// The enum stays local to the TUI crate so the input loop can emit
/// semantic commands without depending on the outer session wiring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentCommand {
    /// User cancelled the in-flight turn (first Ctrl-C).
    CancelTurn,
    /// User requested a clean shutdown.
    Quit,
    /// Mid-turn steering: the user typed a plain message and pressed Enter
    /// while a turn was in flight. The turn loop folds this into the next
    /// tool-result boundary so the model adjusts course within the same turn.
    Steer(String),
    /// A remote device cancelled the turn identified by this generation.
    RemoteCancelTurn { turn_generation: u64 },
    /// A remote device steered the turn identified by this generation.
    RemoteSteer {
        turn_generation: u64,
        text: String,
    },
}

/// Clipboard copy scope requested by a keybinding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardCopyTarget {
    /// Copy the same payload as `/copy` / `/copy last`.
    Last,
    /// Copy the full exported transcript.
    All,
}

/// Semantic action emitted by the app after handling a key event.
#[derive(Debug, Clone)]
pub enum AppAction {
    /// No externally-visible action; the app state changed locally.
    None,
    /// Submit a prompt from the in-app input widget.
    Submit(String),
    /// A model was picked from the modal.
    SelectModel(ActiveModel),
    /// Custom provider draft submitted from the `/connect` onboarding wizard.
    ConnectCustomProvider(CustomProviderDraft),
    /// User entered an API key for an OpenAI-compatible `/connect` preset.
    ConnectApiKey { provider: String, api_key: String },
    /// A permission mode was picked from the modal.
    SelectPermission(RuntimePermissionMode),
    /// User pressed Ctrl+V — the outer loop should read the clipboard
    /// (checking for images first, then text) and feed the result back
    /// into the app via [`super::App::handle_paste`] or
    /// [`super::App::push_clipboard_image`].
    ClipboardPaste,
    /// User requested a copy-to-clipboard operation.
    ClipboardCopy(ClipboardCopyTarget),
    /// User pressed an explicit block-copy control or dragged across rendered
    /// transcript content; copy that range's plain-text payload rather than the
    /// whole session or last message.
    ClipboardCopyBlock(String),
    /// App-local UI state changed and the terminal should repaint, but no
    /// outer-loop command is required. Used for mouse hover affordances and
    /// in-place view toggles so motion/click events do not masquerade as
    /// session actions.
    Redraw,
    /// A session was picked from the `/resume` modal.
    SelectSession(String),
    /// Launch $EDITOR for composing a message.
    Editor,
    /// User double-tapped Esc in Normal mode — the outer loop computes a
    /// rewind preview and opens the [`AppMode::ModalConfirmRewind`] card
    /// rather than rewinding immediately. The actual rewind only runs on
    /// [`Self::ConfirmRewind`], so a stray double-tap can be cancelled.
    RewindCheckpoint,
    /// User confirmed the Esc-Esc rewind in the confirmation card. The outer
    /// loop drives `rewind_turns(1)` + the git snapshot undo.
    ConfirmRewind,
    /// User opened the snapshot rewind viewer (Ctrl+R). The outer loop builds
    /// the timeline from the git snapshot stack and populates the modal.
    OpenRewindViewer,
    /// User opened the live workflow progress viewer (Ctrl+O). The outer loop
    /// reads the workflow progress snapshot, joins the per-agent manifests, and
    /// populates the modal — then keeps refreshing it while it is open.
    OpenWorkflowViewer,
    /// User clicked a specific agent's row in the pinned live-agent panel. Same
    /// surface as [`Self::OpenWorkflowViewer`], but the modal opens pre-selected
    /// to the clicked agent (by id). Falls back to the aggregate view when the
    /// id is not present in the assembled workflow snapshot.
    OpenAgentInViewer(String),
    /// User confirmed a rewind in the viewer — restore the worktree to the
    /// snapshot at this stack index.
    RewindTo(usize),
    /// Ack a `TeamInbox` update through the runtime session-consumer seam.
    AckTeamInboxUpdate(String),
    /// Insert a safe `TeamInbox` summary reference into the composer.
    IncludeTeamInboxUpdate(String),
    /// Refresh the `TeamInbox` viewer snapshot from the runtime store.
    RefreshTeamInboxViewer,
    /// User toggled a runtime tool from the `/tools` modal.
    ToggleTool { name: String, enabled: bool },
    /// User confirmed staged Smart Router settings from the `/smart` dashboard.
    SaveSmartSettings(SmartSettingsCommit),
    /// User requested a `/tier` mutation from the interactive picker.
    DeepTier(commands::DeepTierAction),
    /// Request the outer loop to quit.
    Quit,
}

impl PartialEq for AppAction {
    // Several variant arms share a `left == right` body but bind differently
    // typed payloads, so they cannot be collapsed into a single `|` pattern.
    #[allow(clippy::match_same_arms)]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None)
            | (Self::Quit, Self::Quit)
            | (Self::Editor, Self::Editor)
            | (Self::RewindCheckpoint, Self::RewindCheckpoint)
            | (Self::ConfirmRewind, Self::ConfirmRewind)
            | (Self::OpenRewindViewer, Self::OpenRewindViewer)
            | (Self::OpenWorkflowViewer, Self::OpenWorkflowViewer)
            | (Self::ClipboardPaste, Self::ClipboardPaste)
            | (Self::Redraw, Self::Redraw)
            | (Self::RefreshTeamInboxViewer, Self::RefreshTeamInboxViewer) => true,
            (Self::RewindTo(left), Self::RewindTo(right)) => left == right,
            (Self::OpenAgentInViewer(left), Self::OpenAgentInViewer(right)) => left == right,
            (Self::AckTeamInboxUpdate(left), Self::AckTeamInboxUpdate(right)) => left == right,
            (Self::IncludeTeamInboxUpdate(left), Self::IncludeTeamInboxUpdate(right)) => left == right,
            (Self::ClipboardCopy(left), Self::ClipboardCopy(right)) => left == right,
            (Self::ClipboardCopyBlock(left), Self::ClipboardCopyBlock(right)) => left == right,
            (Self::ConnectCustomProvider(left), Self::ConnectCustomProvider(right)) => left == right,
            (Self::Submit(left), Self::Submit(right))
            | (Self::SelectSession(left), Self::SelectSession(right)) => left == right,
            (
                Self::ConnectApiKey {
                    provider: left_provider,
                    api_key: left_key,
                },
                Self::ConnectApiKey {
                    provider: right_provider,
                    api_key: right_key,
                },
            ) => left_provider == right_provider && left_key == right_key,
            (
                Self::ToggleTool {
                    name: left_name,
                    enabled: left_enabled,
                },
                Self::ToggleTool {
                    name: right_name,
                    enabled: right_enabled,
                },
            ) => left_name == right_name && left_enabled == right_enabled,
            (Self::SaveSmartSettings(left), Self::SaveSmartSettings(right)) => left == right,
            (Self::DeepTier(left), Self::DeepTier(right)) => left == right,
            (Self::SelectPermission(left), Self::SelectPermission(right)) => left == right,
            (Self::SelectModel(left), Self::SelectModel(right)) => {
                left.provider == right.provider
                    && left.alias == right.alias
                    && left.display_name == right.display_name
                    && left.context_limit == right.context_limit
            }
            _ => false,
        }
    }
}

/// A bounded TUI input queue refused a new item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueLimitError {
    /// The type-ahead prompt queue is full.
    QueuedMessagesFull { limit: usize },
    /// The pending clipboard-image list is full.
    PendingImagesFull { limit: usize },
}

impl std::fmt::Display for QueueLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueuedMessagesFull { limit } => {
                write!(f, "queued message limit reached ({limit})")
            }
            Self::PendingImagesFull { limit } => {
                write!(f, "pending image limit reached ({limit})")
            }
        }
    }
}

impl std::error::Error for QueueLimitError {}

/// An image read from the clipboard, awaiting attachment to a user message.
///
/// `data` is already base64-encoded (ready for the Anthropic API).
#[derive(Debug, Clone)]
pub struct ImageAttachment {
    pub media_type: String,
    pub data: String,
}

/// A message composed while a turn was in flight and parked for submission once
/// the turn ends. Carries its own pasted images so an image-only or
/// text-plus-image entry survives the wait intact (Claude Code CLI parity:
/// typed input is queued and shown, then submitted as its own turn in order).
#[derive(Debug, Clone, Default)]
pub struct QueuedMessage {
    /// Raw composed text (may be empty when only images were attached).
    pub text: String,
    /// Clipboard images attached to this queued message.
    pub images: Vec<ImageAttachment>,
    /// True only for a `/goal` action/repair turn the controller enqueued. The
    /// REPL latches goal ownership at *pop* time from this flag (not at dispatch),
    /// so a user message typed ahead of the goal prompt — which pops first under
    /// FIFO — cannot have its verifier verdict mis-attributed to the goal.
    pub goal_owned: bool,
    /// `Some(loop_id)` for a `/loop`-owned run. Consulted at *pop* time so the
    /// loop controller can drop a stale run (the loop was `/loop stop|pause`d, or
    /// its budget is spent) instead of dispatching it — what makes `/loop`
    /// stoppable mid-flight rather than fire-and-forget.
    pub loop_id: Option<String>,
    /// `Some(meta)` when this queued turn is a re-injected **background
    /// sub-agent result**. The text still submits as a normal user-role turn (so
    /// the model reads the agent's result), but the transcript renders it as a
    /// collapsible [`RenderBlock::AgentResult`] card authored by the agent
    /// instead of an amber `You` message. `None` for ordinary user input.
    pub agent_result: Option<AgentResultMeta>,
    /// True when this entry also rode the mid-turn steering channel at emit
    /// time. A later plain-text mid-turn submit may steer only while every
    /// earlier queued entry is itself steered — steering past an unsteered
    /// entry (a slash command, an image turn) would break the FIFO order the
    /// queue preserves. Removed-on-fold semantics are unchanged.
    pub steered: bool,
}

/// Provenance for a re-injected background sub-agent result, carried on the
/// [`QueuedMessage`] so the submit path can render an agent-result card while
/// still sending the body to the model as an input turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentResultMeta {
    /// Sub-agent display label (e.g. `runtime-scout`).
    pub label: String,
    /// Completion status driving the card's header glyph / tint.
    pub status: AgentResultStatus,
}

/// A `/dump` request recorded by slash dispatch for the host loop: suspend
/// the TUI and open `path` in an external viewer once the loop owns the
/// terminal again. Same host-owns-the-terminal contract as the `/memory`
/// pending-editor file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptViewRequest {
    /// The transcript dump artifact to open.
    pub path: std::path::PathBuf,
    /// `true` → `$EDITOR` (`/dump edit`); `false` → `$PAGER` (`/dump`).
    pub edit: bool,
}
