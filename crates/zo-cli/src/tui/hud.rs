//! Bottom HUD — a quiet, right-aligned session summary.
//!
//! ```text
//!   claude-opus-4-8  workspace-write                       ctx 38%  $0.08
//! ```
//!
//! Whitespace establishes hierarchy instead of a full-width rule. The model
//! and permission mode anchor the left edge, live signals align right, and
//! color appears only when a state needs attention. Info remains available
//! on-demand via `/status`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use core_types::RateLimitSnapshot;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use runtime::message_stream::ActiveModel;
use serde::Deserialize;
use unicode_width::UnicodeWidthChar;

use super::app::ScheduledWakeHud;
use super::glyphs;
use super::heat::HeatState;
use super::modals::{Effort, effort_level_label};
use super::sidebar::permission_style;
use super::theme::Theme;
use super::workflow_progress::{WorkflowSummary, is_generic_model_alias, short_model};

const SESSION_BADGE_PALETTE: [Color; 8] = [
    Color::Red,
    Color::Green,
    Color::Yellow,
    Color::Blue,
    Color::Magenta,
    Color::Cyan,
    Color::LightRed,
    Color::LightBlue,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionIdentity {
    pub name: String,
    pub color: Color,
}

impl SessionIdentity {
    #[must_use]
    pub fn named(session_id: &str, name: Option<&str>) -> Option<Self> {
        let name = name.map(str::trim).filter(|name| !name.is_empty())?;
        Some(Self {
            name: name.to_string(),
            color: session_badge_color(session_id),
        })
    }
}

#[must_use]
pub fn session_color_index(session_id: &str) -> usize {
    let hash = session_id.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    usize::try_from(hash % 8).expect("session palette index is always in 0..8")
}

#[must_use]
pub fn session_badge_color(session_id: &str) -> Color {
    SESSION_BADGE_PALETTE[session_color_index(session_id)]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    ReadOnly,
    /// Plan mode: the runtime is read-only, but the session is in a
    /// plan-first gate (the model drafts a plan; the user approves it to
    /// resume editing). Maps to runtime `ReadOnly` but is labelled and
    /// styled distinctly so the HUD shows the gate is engaged.
    Plan,
    Workspace,
    All,
}

impl PermissionMode {
    /// Canonical user-facing permission mode label. Keep this in sync with the
    /// config/parser spelling (`read-only`, `workspace-write`,
    /// `danger-full-access`) so HUD, sidebar, and bottom statusline never show
    /// different names for the same mode.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Plan => "plan",
            Self::Workspace => "workspace-write",
            Self::All => "danger-full-access",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityPosture {
    SandboxActive,
    SandboxBlocked,
    SandboxOff,
}

impl SecurityPosture {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SandboxActive => "sandbox:on",
            Self::SandboxBlocked => "sandbox:blocked",
            Self::SandboxOff => "sandbox:off",
        }
    }
}

/// Status for one item in the live Todo checklist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TodoChecklistStatus {
    /// The item has not started yet.
    Pending,
    /// The item is currently active.
    InProgress,
    /// The item has been completed.
    Completed,
}

impl TodoChecklistStatus {
    /// Stable HUD ordering: the active task first, queued work next, completed
    /// items last. This mirrors the sidebar rendering contract and the
    /// `TodoWrite` persistence order.
    #[must_use]
    pub const fn hud_order(self) -> u8 {
        match self {
            Self::InProgress => 0,
            Self::Pending => 1,
            Self::Completed => 2,
        }
    }
}

/// Canonicalize todo rows before they reach the HUD: active work first,
/// pending next, completed last, preserving model order within each status.
#[must_use]
pub fn canonical_todo_items_for_hud(
    items: impl IntoIterator<Item = TodoChecklistItem>,
) -> Vec<TodoChecklistItem> {
    let mut indexed = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| (item.status.hud_order(), index, item))
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(order, index, _)| (*order, *index));
    indexed.into_iter().map(|(_, _, item)| item).collect()
}

/// Count the *incomplete* (in-progress or pending) todos. The "N todos active"
/// summary must exclude completed items so a finished-but-not-yet-cleared list
/// does not keep claiming work is active. Single owner of "what counts active".
#[must_use]
pub fn count_active_todos(items: &[TodoChecklistItem]) -> usize {
    items
        .iter()
        .filter(|item| item.status != TodoChecklistStatus::Completed)
        .count()
}

/// The "N todos active" summary line, or `None` when nothing is active (no
/// items, or every item completed). Shared by the sidebar HUD, the live
/// snapshot, and the immediate `TodoWrite` update so they never disagree.
#[must_use]
pub fn active_todo_summary(items: &[TodoChecklistItem]) -> Option<String> {
    let active = count_active_todos(items);
    (active > 0).then(|| format!("{active} todos active"))
}

/// A Todo item ready for TUI rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoChecklistItem {
    /// Stable plan-step id from `TodoWrite.stepId`. `None` for legacy and
    /// provider-normalized tool results that do not carry the extension.
    pub step_id: Option<String>,
    /// Stable task description.
    pub content: String,
    /// Current task state.
    pub status: TodoChecklistStatus,
    /// Present-tense form shown by `TodoWrite` while active.
    pub active_form: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredTodoStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Deserialize)]
struct StoredTodoItem {
    #[serde(rename = "stepId", alias = "step_id", default)]
    step_id: Option<String>,
    content: String,
    status: StoredTodoStatus,
    #[serde(rename = "activeForm", default)]
    active_form: String,
}

/// Load the current session todo store and canonicalize it for HUD/sidebar
/// rendering. Missing or malformed stores are treated as empty, matching the
/// HUD's display-only contract.
#[must_use]
pub fn load_todo_items_for_hud(store_path: &Path) -> Vec<TodoChecklistItem> {
    let Ok(raw) = std::fs::read_to_string(store_path) else {
        return Vec::new();
    };
    let Ok(items) = serde_json::from_str::<Vec<StoredTodoItem>>(&raw) else {
        return Vec::new();
    };
    let mapped = items
        .into_iter()
        .map(|item| {
            let active_form = if item.active_form.trim().is_empty() {
                item.content.clone()
            } else {
                item.active_form
            };
            TodoChecklistItem {
                step_id: item.step_id,
                content: item.content,
                status: match item.status {
                    StoredTodoStatus::Pending => TodoChecklistStatus::Pending,
                    StoredTodoStatus::InProgress => TodoChecklistStatus::InProgress,
                    StoredTodoStatus::Completed => TodoChecklistStatus::Completed,
                },
                active_form,
            }
        })
        .collect::<Vec<_>>();
    canonical_todo_items_for_hud(mapped)
}

/// Resolve the todo store used by the HUD. `ZO_TODO_STORE` is the
/// session-specific override; an empty value behaves as unset so the HUD does
/// not accidentally read a relative path from the process cwd.
#[must_use]
pub fn todo_store_path_for_hud(cwd: Option<&Path>) -> Option<PathBuf> {
    // Delegate to the shared resolver so the HUD follows the writer's
    // read-only-cwd fallback instead of showing an empty primary. An explicit
    // `ZO_TODO_STORE` is the session store; resolve from any base since the
    // resolver honors it first.
    if let Ok(path) = std::env::var("ZO_TODO_STORE") {
        if !path.trim().is_empty() {
            return Some(runtime::todo_store::resolve_readable_store(Path::new("")));
        }
    }
    Some(runtime::todo_store::resolve_readable_store(cwd?))
}

/// One language-server status row ready for TUI rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspStatusItem {
    /// Language or server key.
    pub language: String,
    /// Display status from the runtime registry.
    pub status: String,
}

/// Compact MCP server status shown in the sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpHudStatusKind {
    Discovering,
    Ready,
    /// Discovery timed out on an interactive OAuth bridge still waiting for the
    /// user to finish browser auth — recoverable, distinct from `Failed`.
    AuthPending,
    Failed,
}

/// A display-ready MCP server row encoded through [`HudState::mcp_servers`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpHudStatus {
    pub name: String,
    pub kind: McpHudStatusKind,
    pub message: Option<String>,
}

impl McpHudStatus {
    const SEP: char = '\u{1f}';

    #[must_use]
    pub fn ready(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: McpHudStatusKind::Ready,
            message: None,
        }
    }

    #[must_use]
    pub fn discovering(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: McpHudStatusKind::Discovering,
            message: None,
        }
    }

    #[must_use]
    pub fn failed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: McpHudStatusKind::Failed,
            message: Some(message.into()),
        }
    }

    #[must_use]
    pub fn auth_pending(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: McpHudStatusKind::AuthPending,
            message: Some(message.into()),
        }
    }

    #[must_use]
    pub fn encode(&self) -> String {
        let status = match self.kind {
            McpHudStatusKind::Discovering => "discovering",
            McpHudStatusKind::Ready => "ready",
            McpHudStatusKind::AuthPending => "auth_pending",
            McpHudStatusKind::Failed => "failed",
        };
        match &self.message {
            Some(message) => format!(
                "{}{}{}{}{}",
                self.name,
                Self::SEP,
                status,
                Self::SEP,
                message
            ),
            None => format!("{}{}{}", self.name, Self::SEP, status),
        }
    }

    #[must_use]
    pub fn decode(raw: &str) -> Self {
        let mut parts = raw.splitn(3, Self::SEP);
        let name = parts.next().unwrap_or_default().to_string();
        let kind = match parts.next() {
            Some("discovering") => McpHudStatusKind::Discovering,
            Some("auth_pending") => McpHudStatusKind::AuthPending,
            Some("failed") => McpHudStatusKind::Failed,
            _ => McpHudStatusKind::Ready,
        };
        let message = parts
            .next()
            .filter(|message| !message.trim().is_empty())
            .map(str::to_string);
        Self {
            name,
            kind,
            message,
        }
    }
}

/// Overall health of the configured MCP sources, derived once from the
/// per-server statuses so every surface (sidebar headline, its color, the
/// `/doctor` count) reads one classification instead of re-deriving its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpHealth {
    /// No MCP servers are configured.
    None,
    /// Every configured server finished discovery and is ready.
    Healthy,
    /// At least one server is still discovering or waiting on interactive
    /// browser auth, and none have failed — a transient, self-resolving state.
    Connecting,
    /// At least one server failed discovery: the user has fewer tools than
    /// configured and should act.
    Degraded,
}

/// Counts of MCP sources by lifecycle state plus their overall [`McpHealth`].
///
/// The single owner of "how many MCP sources are there, and are they OK". The
/// sidebar headline, its color, and the truncation hint all read this one value
/// object instead of each re-counting `mcp_servers` with its own rule — which is
/// what let the old denormalized `mcp_count` drift from the rendered rows and
/// stay green while a server was failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct McpSourcesSummary {
    pub total: usize,
    pub ready: usize,
    pub discovering: usize,
    pub auth_pending: usize,
    pub failed: usize,
}

impl McpSourcesSummary {
    /// Fold the encoded per-server rows (`HudState::mcp_servers`) into one
    /// summary. Decoding is centralized here so the count can never disagree
    /// with the rendered rows: both are derived from the same source list.
    #[must_use]
    pub fn from_encoded(servers: &[String]) -> Self {
        let mut summary = Self::default();
        for raw in servers {
            summary.total += 1;
            match McpHudStatus::decode(raw).kind {
                McpHudStatusKind::Ready => summary.ready += 1,
                McpHudStatusKind::Discovering => summary.discovering += 1,
                McpHudStatusKind::AuthPending => summary.auth_pending += 1,
                McpHudStatusKind::Failed => summary.failed += 1,
            }
        }
        summary
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Collapse the per-state counts into one overall verdict. Failure wins over
    /// in-flight, which wins over fully-ready — so a single failed source can
    /// never be masked by healthy siblings in the headline color.
    #[must_use]
    pub fn health(&self) -> McpHealth {
        if self.total == 0 {
            McpHealth::None
        } else if self.failed > 0 {
            McpHealth::Degraded
        } else if self.discovering > 0 || self.auth_pending > 0 {
            McpHealth::Connecting
        } else {
            McpHealth::Healthy
        }
    }
}

/// Per-agent line displayed in the sidebar tree under "✦ N agents".
///
/// 단순 count 만으로는 Claude Code 의 `↓ N background agents launched`
/// 평면 UX 와 차이가 없다. zo 는 manifest 의 name/status/elapsed
/// 까지 노출해 어떤 agent 가 얼마나 오래 돌고 있는지 한 화면에서 본다.
#[derive(Debug, Clone, Default)]
pub struct AgentTaskSummary {
    /// Manifest `agentId` — joins this summary to the transcript's live agent
    /// tree and the completion channel. Empty for legacy manifests.
    pub id: String,
    /// Manifest `toolCallId` — the delegation call that spawned this agent, so
    /// the transcript batch tree attributes it to the right Spawn-family call
    /// on concurrent multi-delegation turns. `None` for legacy manifests and
    /// host-spawned agents (those fall back to the collecting batch).
    pub tool_call_id: Option<String>,
    pub name: String,
    pub status: String,
    /// Actual resolved model string from the manifest's `model` field, so the
    /// user sees *which* model each agent is running. Rendering may shorten it,
    /// but the data stays bound to the spawned agent's real model. Empty when
    /// unknown.
    pub model: String,
    pub elapsed_secs: u64,
    /// Per-turn token sample sequence; rendered as a Sparkline in the
    /// sidebar agent row when non-empty. Empty in the current build
    /// because the per-agent token timeline collection is a separate
    /// chunk — the data path is wired through so once the producer side
    /// lands the sparkline lights up with no UI change required.
    pub token_history: Vec<u32>,
    /// Tool the agent is currently running (the manifest's `currentTool`),
    /// shown next to the agent row so the user sees live activity — *what*
    /// each agent is doing, not just that it's running. `None` between tools.
    pub current_tool: Option<String>,
    /// Transient wait/stream phase from the manifest's `currentPhase`
    /// (`waiting for api slot`, `rate-limited · resumes in ~90s`, `thinking`).
    /// Shown when no tool is active, so a quota-parked agent reads as alive.
    pub current_phase: Option<String>,
    /// Epoch seconds of the agent's last liveness signal (`lastActivityAt`).
    /// The agents detail view derives a heartbeat (`active 3s ago`) from it.
    pub last_activity_at: Option<u64>,
    /// Manifest `subagentType` (e.g. `Explore`) — drives the Claude Code style
    /// `N Explore agents finished` header when a batch is homogeneous.
    pub subagent_type: Option<String>,
    /// Manifest `toolCalls` running total, shown as `N tool uses` in the tree.
    pub tool_calls: Option<usize>,
    /// Total output tokens so far (sum of the manifest's `tokenHistory`).
    pub tokens: u64,
    /// Manifest `createdAt` epoch seconds — stable spawn-order key for the tree.
    pub created_at: Option<u64>,
    /// Last chars of the agent's streamed output (manifest `outputTail`, rolling
    /// buffer). Surfaced as a dim `⤷ …` sub-line under the agent's row in the
    /// pinned live tree / inline tree so the user sees *what each agent is
    /// saying*, not just which tool it is running. `None` when the agent has
    /// streamed nothing yet.
    pub output_tail: Option<String>,
    /// Why the Smart router picked this agent's model (manifest `routeReason`).
    /// Shown in the Ctrl+G agent detail so auto-routing is explainable, not
    /// opaque. `None` for explicit models / routing off / legacy manifests.
    pub route_reason: Option<String>,
}

impl AgentTaskSummary {
    /// The live-activity label for this agent's sidebar row: an active tool
    /// wins; otherwise the wait/stream phase explains the silence.
    #[must_use]
    pub fn activity_label(&self) -> Option<&str> {
        self.current_tool
            .as_deref()
            .or(self.current_phase.as_deref())
    }

    /// Whether the activity label names a *wait* state (no tool running, a
    /// phase shown instead) — rendered in the warning tone so a parked agent
    /// is visually distinct from one actively running a tool.
    #[must_use]
    pub fn activity_is_wait(&self) -> bool {
        self.current_tool.is_none() && self.current_phase.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct HudState {
    pub session_identity: Option<SessionIdentity>,
    pub model: ActiveModel,
    /// Model that re-served this turn after a safety-classifier refusal.
    /// Cleared at the next user-turn start.
    pub turn_fallback_model: Option<String>,
    /// Cross-provider model serving turns while the main model's quota cools.
    /// Re-announced by every parked turn and cleared at each turn start.
    pub quota_fallback_model: Option<String>,
    pub ctx_used: u64,
    pub ctx_limit: u64,
    /// New (uncached) input tokens — the portion of context not served from
    /// the prompt cache. `0` until a billed response arrives. Billed at full
    /// rate, so this (not the cached bulk) drives cost.
    pub ctx_new_input: u64,
    /// Cache-read input tokens — context served from the prompt cache at ~1/10
    /// the price. Typically the bulk of a long session's context.
    pub ctx_cached: u64,
    /// Input-token count at which full auto-compaction fires for the active
    /// model (live-runtime resolved: model-family policy — 80% of the window
    /// for Claude, 85% otherwise — plus env / settings overrides). `0` =
    /// unknown → gauges fall back to nominal-window percent.
    pub compact_threshold: u64,
    pub cost_usd: f64,
    /// `true` when the active model has no pricing-table entry and `cost_usd`
    /// was computed at the fallback Sonnet rate — rendered `~$` so the guess
    /// is not presented as authoritative.
    pub cost_approx: bool,
    pub cwd: PathBuf,
    pub git_branch: Option<String>,
    pub perm_mode: PermissionMode,
    pub security_posture: SecurityPosture,
    pub effort: Option<Effort>,
    /// The Architect contract's implementer model when `smart.execSwap` is
    /// armed for the current/most-recent turn's difficulty. Rendered alone —
    /// the anchor names whoever is editing, not the steering pair; `None`
    /// keeps the plain session model for native EXEC.
    pub architect_impl: Option<String>,
    /// Encoded per-server MCP rows (see [`McpHudStatus::encode`]). The single
    /// source of truth for MCP in the HUD: both the count and the rendered rows
    /// derive from this one list via [`McpSourcesSummary`], so they can never
    /// disagree. There is deliberately no separate `mcp_count` field — a
    /// denormalized count is exactly what let the headline drift from the rows.
    pub mcp_servers: Vec<String>,
    pub bash_count: u32,
    pub read_count: u32,
    pub edit_count: u32,
    pub changed_files: usize,
    pub todo_summary: Option<String>,
    pub todo_items: Vec<TodoChecklistItem>,
    /// Active `/goal` and `/loop` automation summaries for the sidebar.
    pub automation_lines: Vec<String>,
    pub lsp_servers: Vec<LspStatusItem>,
    pub running_agents: u16,
    /// Per-agent summaries — `running_agents` 의 길이/count 와 일치.
    /// 비어 있으면 sidebar 가 count line 만 표시 (legacy 동작).
    pub agents: Vec<AgentTaskSummary>,
    /// Lightweight workflow topology summary for the sidebar. This is separate
    /// from `agents`: dynamic workflows have explicit phases, while plain
    /// `SpawnMultiAgent` fan-out only has manifests.
    pub workflow: Option<WorkflowSummary>,
    /// Human-readable current tool activity, surfaced in the sidebar so
    /// users can see *what* is happening (not just that something is).
    pub last_tool: Option<String>,
    /// Unified 5h/7d rate-limit gauges (subscription / OAuth). `None` until a
    /// streamed response carries the unified headers; API-key sessions leave
    /// it `None` and the sidebar shows no gauge.
    pub rate_limit: Option<RateLimitSnapshot>,
    /// Cross-provider quota rows from `api::quota::provider_quota_views()` —
    /// the measured Anthropic windows plus a 429-estimated row per throttled
    /// non-Anthropic provider. The sidebar renders only the estimated rows from
    /// here (marked `est`); the measured Anthropic gauge keeps rendering from
    /// [`Self::rate_limit`] unchanged. Refreshed on the periodic HUD rebuild;
    /// empty when no provider currently carries a quota signal.
    pub provider_quotas: Vec<api::quota::ProviderQuotaView>,
    /// Which rung of the Claude credential chain is active (keychain OAuth /
    /// `zo login` OAuth / env API key). OAuth-first: the env rung is
    /// metered billing, so the sidebar renders it as a standing warning.
    pub auth_origin: Option<api::ClaudeAuthOrigin>,
    /// First output line of the user's configured `statusLine` command
    /// (settings key, Claude Code parity). When `Some`, [`compose`] renders it
    /// — ANSI SGR parsed into spans — instead of the stock segment row.
    pub status_line: Option<String>,
    /// Unread `TeamInbox` updates for this session's consumer (B4 badge).
    /// Computed by the same unread predicate as the turn-start digest
    /// (`runtime::team_inbox_unread_count`), fail-open to `0`. A count only —
    /// no update summary/body text ever reaches the HUD.
    pub team_inbox_unread: u64,
    /// Set once the running binary's on-disk file has been replaced by a new
    /// build (see [`crate::tui::stale_binary`]). Drives an always-on sidebar
    /// warning telling the user to `/restart` so the live session stops running
    /// stale code. `None` while the running binary still matches disk.
    pub stale_binary: Option<super::stale_binary::StaleBinaryInfo>,
    /// Active `run_in_background` Bash processes launched by this visible
    /// session in the current runtime. This comes from an ephemeral atomic
    /// tracker, not persisted `TaskRegistry` status; generic tasks, unstamped
    /// launches, and pre-restart records therefore fail closed to zero.
    pub background_tasks: usize,
    /// Nearest pending `ScheduleWakeup` or recurring `/loop` deadline. The App
    /// refreshes the source snapshot; renderers only subtract wall time.
    pub scheduled_wake: Option<ScheduledWakeHud>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HudViewModel {
    pub session_identity: Option<SessionIdentity>,
    pub model: String,
    pub context: String,
    pub cost: String,
    pub security: String,
    pub workflow: Option<String>,
    pub permission_mode: PermissionMode,
    pub security_posture: SecurityPosture,
    pub running_agents: u16,
    pub edits: u32,
    pub changed_files: usize,
    pub background_tasks: usize,
    pub scheduled_wake: Option<ScheduledWakeHud>,
}

impl HudViewModel {
    #[must_use]
    pub fn from_state(state: &HudState) -> Self {
        Self {
            session_identity: state.session_identity.clone(),
            model: model_short_name(state),
            context: format_context_tokens(state.ctx_used, state.ctx_limit),
            cost: format_cost(state.cost_usd, state.cost_approx),
            security: state.security_posture.label().to_string(),
            workflow: state.workflow.as_ref().map(workflow_hud_label),
            permission_mode: state.perm_mode,
            security_posture: state.security_posture,
            running_agents: state.running_agents,
            edits: state.edit_count,
            changed_files: state.changed_files,
            background_tasks: state.background_tasks,
            scheduled_wake: state.scheduled_wake.clone(),
        }
    }
}

/// Compact countdown text shared by the HUD and sidebar.
#[must_use]
pub fn format_scheduled_countdown(seconds: u64) -> String {
    if seconds == 0 {
        return "now".to_string();
    }
    if seconds >= 60 * 60 {
        return format!("{}h{:02}m", seconds / (60 * 60), (seconds / 60) % 60);
    }
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

pub(super) fn scheduled_countdown(wake: &ScheduledWakeHud) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_scheduled_countdown(wake.due_at_epoch.saturating_sub(now))
}

fn push_live_badges(spans: &mut Vec<Span<'static>>, view: &HudViewModel, theme: &Theme) {
    if let Some(identity) = view.session_identity.as_ref() {
        let badge_style = if theme.no_color {
            Style::default()
        } else {
            Style::new().fg(identity.color)
        };
        spans.push(Span::styled(format!("{HUD_GAP}●"), badge_style));
        spans.push(Span::styled(
            format!(" {}", identity.name),
            Style::new().fg(theme.palette.fg),
        ));
    }
    let live_style = Style::new().fg(theme.palette.cyan);
    if view.background_tasks > 0 {
        spans.push(Span::styled(
            format!("{HUD_GAP}bg {}", view.background_tasks),
            live_style,
        ));
    }
    if let Some(wake) = view.scheduled_wake.as_ref() {
        spans.push(Span::styled(
            format!("{HUD_GAP}wake {}", scheduled_countdown(wake)),
            live_style,
        ));
    }
}

const HUD_GUTTER: &str = "   ";
const HUD_GAP: &str = "  ";

/// Draw the bottom status bar.
///
/// `details_owned_elsewhere` reports whether another visible surface already
/// owns session/model/security/work status for this frame: the right-hand
/// ledger when it is painted, or the top activity row while a turn is running.
/// When true, the HUD keeps only the model, permission mode, context pressure,
/// and live signals that still need immediate attention.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &HudState,
    theme: &Theme,
    ledger_visible: bool,
    agent_panel_visible: bool,
) {
    draw_with_heat(
        frame,
        area,
        state,
        theme,
        ledger_visible,
        agent_panel_visible,
        HeatState::Cold,
    );
}

/// Draw the bottom status bar with a compact activity marker for the current
/// turn. Temperature is intentionally confined to one cell instead of washing
/// the full row with animated color.
pub fn draw_with_heat(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &HudState,
    theme: &Theme,
    ledger_visible: bool,
    agent_panel_visible: bool,
    heat_state: HeatState,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::new().bg(Color::Reset)),
        area,
    );

    if area.height >= 2 {
        if let Some(workflow_line) = compose_workflow_row(state, theme, area.width, heat_state) {
            let activity = Rect {
                height: 1,
                ..area
            };
            let session = Rect {
                y: area.y + 1,
                height: area.height - 1,
                ..area
            };
            frame.render_widget(Paragraph::new(workflow_line), activity);
            frame.render_widget(
                Paragraph::new(compose_with_overlays_and_heat(
                    state,
                    theme,
                    area.width,
                    ledger_visible,
                    agent_panel_visible,
                    false,
                    HeatState::Cold,
                )),
                session,
            );
            return;
        }
    }

    frame.render_widget(
        Paragraph::new(compose_with_overlays_and_heat(
            state,
            theme,
            area.width,
            ledger_visible,
            agent_panel_visible,
            true,
            heat_state,
        )),
        area,
    );
}

/// Dedicated workflow activity row for narrow terminals that receive a second
/// HUD line. The live activity sits above the quiet session summary, matching
/// the same information hierarchy used by the full-width layout.
fn compose_workflow_row(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    heat_state: HeatState,
) -> Option<Line<'static>> {
    let workflow = workflow_hud_label(state.workflow.as_ref()?);
    let marker = activity_marker_span(theme, heat_state, theme.palette.accent);
    let avail = usize::from(cols).saturating_sub(display_width(marker.content.as_ref()));
    let shown = truncate_hud_label(&workflow, avail);
    Some(Line::from(vec![
        marker,
        Span::styled(shown, Style::new().fg(theme.palette.fg)),
    ]))
}

/// [`compose`] with overlay awareness: when the pinned live-agent panel is
/// already on screen, the bottom bar's agent count is redundant, so the panel
/// owns that signal while every other segment remains unchanged.
#[must_use]
pub fn compose_with_overlays(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    ledger_visible: bool,
    agent_panel_visible: bool,
    show_workflow: bool,
) -> Line<'static> {
    compose_with_overlays_and_heat(
        state,
        theme,
        cols,
        ledger_visible,
        agent_panel_visible,
        show_workflow,
        HeatState::Cold,
    )
}

fn compose_with_overlays_and_heat(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    ledger_visible: bool,
    agent_panel_visible: bool,
    show_workflow: bool,
    heat_state: HeatState,
) -> Line<'static> {
    compose_impl(
        state,
        theme,
        cols,
        ledger_visible,
        !agent_panel_visible,
        show_workflow,
        heat_state,
    )
}

#[must_use]
pub fn compose(state: &HudState, theme: &Theme, cols: u16, ledger_visible: bool) -> Line<'static> {
    compose_impl(
        state,
        theme,
        cols,
        ledger_visible,
        true,
        true,
        HeatState::Cold,
    )
}

/// Compose the stock HUD. `show_workflow = false` suppresses the inline badge
/// when a dedicated activity row already owns the workflow phase.
#[allow(clippy::too_many_lines)]
fn compose_impl(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    ledger_visible: bool,
    show_agents: bool,
    show_workflow: bool,
    heat_state: HeatState,
) -> Line<'static> {
    let view = HudViewModel::from_state(state);

    if let Some(raw) = state.status_line.as_deref() {
        let first = raw.lines().next().unwrap_or("").trim_end();
        if !first.trim().is_empty() {
            return compose_custom_status_line(first, &view, theme, cols, heat_state);
        }
    }

    let model_style = Style::new().fg(theme.palette.fg);
    let detail_style = Style::new().fg(theme.palette.muted);
    let mut left_spans = hud_leader_spans(theme, heat_state);
    left_spans.push(Span::styled(view.model.clone(), model_style));
    left_spans.push(Span::raw(HUD_GAP));
    push_status_spans(&mut left_spans, &view, theme);

    if ledger_visible {
        if state.ctx_used > 0 && compact_percent(state.ctx_used, state.compact_threshold).is_some() {
            left_spans.push(Span::raw(HUD_GAP));
            left_spans.extend(context_usage_spans(state, &view.context, theme));
        }

        let mut right_spans = Vec::new();
        push_live_badges(&mut right_spans, &view, theme);
        push_agent_badge(&mut right_spans, &view, theme, show_agents);
        return compose_hud_row(left_spans, right_spans, cols);
    }

    left_spans.push(Span::raw(HUD_GAP));
    left_spans.extend(context_usage_spans(state, &view.context, theme));
    left_spans.push(Span::raw(HUD_GAP));
    left_spans.push(Span::styled(view.cost.clone(), detail_style));

    let mut right_spans = Vec::new();
    push_live_badges(&mut right_spans, &view, theme);
    push_agent_badge(&mut right_spans, &view, theme, show_agents);

    if show_workflow {
        if let Some(workflow) = &view.workflow {
            let left_width = spans_display_width(&left_spans);
            let right_width = spans_display_width(&right_spans);
            let avail = usize::from(cols).saturating_sub(left_width + right_width + 2);
            if cols >= 96 && avail >= 12 {
                right_spans.push(Span::styled(
                    format!("{HUD_GAP}{}", truncate_hud_label(workflow, avail)),
                    Style::new().fg(theme.palette.info),
                ));
            }
        }
    }

    push_change_badge(&mut right_spans, &left_spans, &view, theme, cols);
    compose_hud_row(left_spans, right_spans, cols)
}

fn push_agent_badge(
    spans: &mut Vec<Span<'static>>,
    view: &HudViewModel,
    theme: &Theme,
    show_agents: bool,
) {
    if !show_agents || view.running_agents == 0 {
        return;
    }
    let spark = glyphs::pick(!theme.no_color, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC);
    spans.push(Span::styled(
        format!("{HUD_GAP}{spark}{} agents", view.running_agents),
        Style::new().fg(theme.palette.info),
    ));
}

fn push_change_badge(
    right_spans: &mut Vec<Span<'static>>,
    left_spans: &[Span<'static>],
    view: &HudViewModel,
    theme: &Theme,
    cols: u16,
) {
    let label = if view.changed_files > 0 {
        format!("+{} files", view.changed_files)
    } else if view.edits > 0 {
        format!("+{} edits", view.edits)
    } else {
        return;
    };
    let needed = spans_display_width(left_spans)
        + spans_display_width(right_spans)
        + display_width(&label)
        + display_width(HUD_GAP);
    if needed < usize::from(cols) {
        right_spans.push(Span::styled(
            format!("{HUD_GAP}{label}"),
            Style::new().fg(theme.palette.teal),
        ));
    }
}

/// Keep live right-edge signals visible by trimming lower-priority stock
/// context from the left before the terminal gets a chance to clip the tail.
fn compose_hud_row(
    mut left_spans: Vec<Span<'static>>,
    right_spans: Vec<Span<'static>>,
    cols: u16,
) -> Line<'static> {
    let width = usize::from(cols);
    let right_width = spans_display_width(&right_spans).min(width);
    truncate_spans(&mut left_spans, width.saturating_sub(right_width));
    let used = spans_display_width(&left_spans) + right_width;
    if !right_spans.is_empty() && used < width {
        left_spans.push(Span::raw(" ".repeat(width - used)));
    }
    left_spans.extend(right_spans);
    truncate_spans(&mut left_spans, width);
    Line::from(left_spans)
}

fn push_status_spans(spans: &mut Vec<Span<'static>>, view: &HudViewModel, theme: &Theme) {
    if view.security_posture == SecurityPosture::SandboxBlocked {
        spans.push(Span::styled(
            view.security.clone(),
            security_posture_style(view.security_posture, theme),
        ));
        return;
    }
    if view
        .workflow
        .as_deref()
        .is_some_and(|workflow| workflow.contains("running"))
    {
        spans.push(Span::styled(
            "running",
            Style::new().fg(theme.palette.muted),
        ));
        spans.push(Span::raw(HUD_GAP));
    }
    spans.push(Span::styled(
        view.permission_mode.label(),
        hud_mode_style(view.permission_mode, theme),
    ));
}

fn hud_mode_style(mode: PermissionMode, theme: &Theme) -> Style {
    if mode == PermissionMode::All {
        permission_style(mode, theme)
    } else {
        Style::new().fg(theme.palette.muted)
    }
}

/// Render the user's custom status line after the persistent model anchor. ANSI
/// SGR is preserved in color mode and stripped under `NO_COLOR`.
fn compose_custom_status_line(
    first: &str,
    view: &HudViewModel,
    theme: &Theme,
    cols: u16,
    heat_state: HeatState,
) -> Line<'static> {
    let mut spans = hud_leader_spans(theme, heat_state);
    if !view.model.trim().is_empty() {
        spans.push(Span::styled(
            view.model.clone(),
            Style::new().fg(theme.palette.fg),
        ));
        spans.push(Span::raw(HUD_GAP));
    }
    let mut content = super::ansi_spans::ansi_spans(first);
    if theme.no_color {
        for span in &mut content {
            span.style = Style::default();
        }
    }
    spans.extend(content);
    let mut right_spans = Vec::new();
    push_live_badges(&mut right_spans, view, theme);
    compose_hud_row(spans, right_spans, cols)
}

fn hud_leader_spans(theme: &Theme, heat_state: HeatState) -> Vec<Span<'static>> {
    if heat_state == HeatState::Cold {
        vec![Span::raw(HUD_GUTTER)]
    } else {
        vec![activity_marker_span(
            theme,
            heat_state,
            theme.palette.muted,
        )]
    }
}

fn activity_marker_span(theme: &Theme, heat_state: HeatState, cold_color: Color) -> Span<'static> {
    let spark = glyphs::pick(!theme.no_color, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC);
    let color = match heat_state {
        HeatState::Cold => cold_color,
        HeatState::Hot => theme.heat().ember,
        HeatState::Cooling { ramp_idx } => theme.cooling_fill_color(ramp_idx),
    };
    Span::styled(
        format!(" {spark} "),
        chrome_style(theme, color),
    )
}

fn chrome_style(theme: &Theme, color: Color) -> Style {
    if theme.no_color || color == Color::Reset {
        Style::default()
    } else {
        Style::new().fg(color)
    }
}

/// Truncate a styled span run to `max_width` display cells, trimming from the
/// tail and appending an ellipsis (mirrors [`truncate_hud_label`] but preserves
/// per-span styling so the leading model badge survives). No-op when it fits.
fn truncate_spans(spans: &mut Vec<Span<'static>>, max_width: usize) {
    if spans_display_width(spans) <= max_width {
        return;
    }
    if max_width == 0 {
        spans.clear();
        return;
    }
    let budget = max_width.saturating_sub(1);
    let mut out: Vec<Span<'static>> = Vec::with_capacity(spans.len());
    let mut width = 0usize;
    for span in spans.drain(..) {
        if width >= budget {
            break;
        }
        let span_width = display_width(span.content.as_ref());
        if width + span_width <= budget {
            width += span_width;
            out.push(span);
            continue;
        }
        // Partial span: keep as many leading chars as fit in the remaining budget.
        let mut kept = String::new();
        for ch in span.content.chars() {
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if width + ch_width > budget {
                break;
            }
            kept.push(ch);
            width += ch_width;
        }
        if !kept.is_empty() {
            out.push(Span::styled(kept, span.style));
        }
        break;
    }
    out.push(Span::raw("…"));
    *spans = out;
}

fn spans_display_width(spans: &[Span<'_>]) -> usize {
    spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}


fn security_posture_style(posture: SecurityPosture, theme: &Theme) -> Style {
    let color = match posture {
        SecurityPosture::SandboxActive => theme.palette.success,
        SecurityPosture::SandboxBlocked => theme.palette.warn,
        SecurityPosture::SandboxOff => theme.palette.error,
    };
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

pub(crate) fn effort_badge_label(effort: Option<Effort>, model: &str) -> Option<String> {
    let effort = match effort {
        Some(Effort::Off) | None => return None,
        Some(effort) => effort,
    };
    let label = effort.canonical();
    // Smart carries a DYNAMIC band, not one static tier — its `level()` is
    // only the floor, so the single-tier clamp check below would silently
    // hide the escalation headroom. Always show the resolved band range
    // (which degenerates to one value when the model's ceiling collapses it).
    if let Some((floor, ceiling)) = effort.band_labels_for_model(model) {
        return Some(if floor == ceiling {
            format!("{label}→{floor}")
        } else {
            format!("{label}→{floor}~{ceiling}")
        });
    }
    // Show the model-specific Zo tier after capability clamping, matching
    // `/effort show`. Provider serializers may encode a higher internal GPT
    // tier as the supported xhigh wire value.
    if let Some(requested) = effort.level() {
        let effective = api::effective_effort_for_model(requested, model);
        if effective != requested {
            return Some(format!("{label}→{}", effort_level_label(effective)));
        }
    }
    Some(label.to_string())
}

fn model_short_name(state: &HudState) -> String {
    let alias = short_model(&state.model.alias);
    let display = short_model(&state.model.display_name);

    let base = if alias.is_empty() {
        display
    } else if display.is_empty() {
        alias
    } else if is_generic_model_alias(&alias) && display != alias {
        display
    } else {
        alias
    };
    // Architect contract active this turn: show the implementer alone — the
    // user asked to see the model actually editing, not a `main▸impl` pair.
    let mut label = match state.architect_impl.as_deref().map(short_model) {
        Some(impl_model) if !impl_model.is_empty() => impl_model,
        _ => base.clone(),
    };

    // A quota fallback is session-cooldown state, not a permanent model switch:
    // keep the configured session model visible beside the model actually on
    // the wire. It outranks the Architect anchor because quota is the active
    // provider-level override.
    if let Some(fallback) = state.quota_fallback_model.as_deref().map(short_model) {
        if !fallback.is_empty() {
            label = if base.is_empty() {
                format!("{fallback} (quota)")
            } else {
                format!("{base}→{fallback} (quota)")
            };
        }
    }

    // Refusal fallback is one-turn-only. ASCII `!` is a width-safe identity
    // mark (not an emoji) and remains legible when NO_COLOR strips styling.
    if let Some(fallback) = state.turn_fallback_model.as_deref().map(short_model) {
        if !fallback.is_empty() {
            label.push_str(" !");
            label.push_str(&fallback);
        }
    }
    label
}

fn workflow_hud_label(summary: &WorkflowSummary) -> String {
    let phase = truncate_hud_label(&summary.current_phase, 18);
    let terminal_agents = summary
        .completed_agents
        .saturating_add(summary.failed_agents);
    // Show the completion percent alone — the old "X% · Y% left" pair was
    // redundant (Y is always 100−X) and read as a broken "0%/100%" while a phase
    // had no finished agents yet; the percent now carries in-flight credit.
    let mut label = format!(
        "{}% · phase {}/{} {phase}",
        summary.progress_percent, summary.current_phase_index, summary.total_phases
    );
    if !summary.current_phase_status.is_empty() {
        label.push(' ');
        label.push_str(&summary.current_phase_status);
    }
    if let Some(next) = summary
        .next_phase
        .as_deref()
        .filter(|phase| !phase.trim().is_empty())
    {
        label.push_str(" \u{2192} ");
        label.push_str(&truncate_hud_label(next, 14));
    }
    if summary.total_agents > 0 {
        let _ = write!(
            label,
            " · agents {terminal_agents}/{}",
            summary.total_agents
        );
        if summary.running_agents > 0 {
            let _ = write!(label, " · {} running", summary.running_agents);
        }
    }
    label
}

fn truncate_hud_label(text: &str, max_width: usize) -> String {
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }

    let mut out = String::new();
    let mut width = 0usize;
    let budget = max_width.saturating_sub(1);
    for ch in text.chars() {
        let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + char_width > budget {
            break;
        }
        out.push(ch);
        width += char_width;
    }
    out.push('…');
    out
}

use crate::tui::text_metrics::display_width;

fn format_context_tokens(used: u64, limit: u64) -> String {
    if used == 0 {
        return "ctx pending".to_string();
    }
    let over_limit = limit > 0 && used > limit;
    let display_tokens = if over_limit { limit } else { used };
    let tokens = format_tokens(display_tokens, over_limit);
    format!("ctx {tokens}")
}

/// Map normalized context pressure to the shared semantic bands. Healthy usage
/// stays neutral; color enters only as compaction pressure becomes actionable.
#[must_use]
pub(crate) fn heat_band_color(ratio: f64, theme: &Theme) -> Color {
    if ratio < 0.75 {
        theme.palette.muted
    } else if ratio < 0.90 {
        theme.heat().ember
    } else {
        theme.heat().molten
    }
}

fn context_usage_spans(
    state: &HudState,
    fallback: &str,
    theme: &Theme,
) -> Vec<Span<'static>> {
    let muted = Style::new().fg(theme.palette.muted);
    if state.ctx_used == 0 {
        return vec![Span::styled(fallback.to_string(), muted)];
    }
    let Some(percent) = context_pressure_percent(state) else {
        return vec![Span::styled(fallback.to_string(), muted)];
    };

    let ratio = f64::from(u32::try_from(percent).unwrap_or(100)) / 100.0;
    vec![Span::styled(
        format!("ctx {percent}%"),
        Style::new().fg(heat_band_color(ratio, theme)),
    )]
}

/// Canonical live context pressure for every TUI surface. Prefer occupancy of
/// the auto-compaction ceiling; fall back to nominal window occupancy only when
/// that ceiling is unavailable.
pub(crate) fn context_pressure_percent(state: &HudState) -> Option<u64> {
    if state.ctx_used == 0 {
        return None;
    }
    compact_percent(state.ctx_used, state.compact_threshold).or_else(|| {
        (state.ctx_limit > 0)
            .then(|| (state.ctx_used.saturating_mul(100) / state.ctx_limit).min(100))
    })
}

/// Percent (0-100, saturating) of the auto-compaction threshold consumed.
/// `None` when the threshold is unknown.
pub(crate) fn compact_percent(used: u64, compact_threshold: u64) -> Option<u64> {
    if compact_threshold == 0 {
        return None;
    }
    Some((used.saturating_mul(100) / compact_threshold).min(100))
}

fn format_tokens(tokens: u64, over_limit: bool) -> String {
    let suffix = if over_limit { "+" } else { "" };
    if tokens == 0 {
        format!("~0{suffix}")
    } else if tokens < 1_000 {
        format!("~{tokens}{suffix}")
    } else if tokens < 1_000_000 {
        #[allow(clippy::cast_precision_loss)]
        let v = tokens as f64 / 1_000.0;
        format!("~{v:.1}k{suffix}")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let v = tokens as f64 / 1_000_000.0;
        if tokens.is_multiple_of(1_000_000) {
            format!("~{v:.1}M{suffix}")
        } else {
            format!("~{v:.2}M{suffix}")
        }
    }
}

fn format_cost(usd: f64, approx: bool) -> String {
    // `~` marks a fallback-rate estimate (model missing from the pricing
    // table), so the figure reads as a guess rather than a bill.
    let prefix = if approx { "~$" } else { "$" };
    if usd < 0.001 {
        format!("{prefix}0.00")
    } else {
        format!("{prefix}{usd:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(servers: &[McpHudStatus]) -> Vec<String> {
        servers.iter().map(McpHudStatus::encode).collect()
    }

    #[test]
    fn mcp_summary_counts_each_lifecycle_state() {
        let servers = encoded(&[
            McpHudStatus::ready("a"),
            McpHudStatus::ready("b"),
            McpHudStatus::discovering("c"),
            McpHudStatus::auth_pending("d", "browser auth"),
            McpHudStatus::failed("e", "timed out"),
        ]);
        let summary = McpSourcesSummary::from_encoded(&servers);
        assert_eq!(summary.total, 5);
        assert_eq!(summary.ready, 2);
        assert_eq!(summary.discovering, 1);
        assert_eq!(summary.auth_pending, 1);
        assert_eq!(summary.failed, 1);
        // total is exactly the rendered row count — the count can never drift
        // from the rows because both fold the same list.
        assert_eq!(summary.total, servers.len());
    }

    #[test]
    fn mcp_summary_health_ranks_failure_over_inflight_over_ready() {
        assert_eq!(McpSourcesSummary::default().health(), McpHealth::None);

        let ready = McpSourcesSummary::from_encoded(&encoded(&[McpHudStatus::ready("a")]));
        assert_eq!(ready.health(), McpHealth::Healthy);

        let connecting = McpSourcesSummary::from_encoded(&encoded(&[
            McpHudStatus::ready("a"),
            McpHudStatus::discovering("b"),
        ]));
        assert_eq!(connecting.health(), McpHealth::Connecting);

        let auth = McpSourcesSummary::from_encoded(&encoded(&[McpHudStatus::auth_pending(
            "a",
            "browser auth",
        )]));
        assert_eq!(auth.health(), McpHealth::Connecting);

        // A single failure outranks ready/in-flight siblings: the headline can
        // never stay green (or merely amber) while a source is down.
        let degraded = McpSourcesSummary::from_encoded(&encoded(&[
            McpHudStatus::ready("a"),
            McpHudStatus::discovering("b"),
            McpHudStatus::failed("c", "timed out"),
        ]));
        assert_eq!(degraded.health(), McpHealth::Degraded);
    }

    #[test]
    fn mcp_summary_is_empty_only_with_no_sources() {
        assert!(McpSourcesSummary::from_encoded(&[]).is_empty());
        assert!(!McpSourcesSummary::from_encoded(&encoded(&[McpHudStatus::ready("a")])).is_empty());
    }

    #[test]
    fn mcp_hud_status_encode_decode_roundtrips_each_kind() {
        let ready = McpHudStatus::ready("ctx7");
        assert_eq!(McpHudStatus::decode(&ready.encode()), ready);

        let discovering = McpHudStatus::discovering("ctx7");
        assert_eq!(McpHudStatus::decode(&discovering.encode()), discovering);

        let auth_pending = McpHudStatus::auth_pending("atlassian", "waiting for browser auth");
        let decoded = McpHudStatus::decode(&auth_pending.encode());
        assert_eq!(decoded.kind, McpHudStatusKind::AuthPending);
        assert_eq!(decoded, auth_pending);

        let failed = McpHudStatus::failed("atlassian", "initialize timed out");
        assert_eq!(McpHudStatus::decode(&failed.encode()), failed);
    }

    fn todo(status: TodoChecklistStatus) -> TodoChecklistItem {
        TodoChecklistItem {
            step_id: None,
            content: "task".to_string(),
            status,
            active_form: "doing task".to_string(),
        }
    }

    #[test]
    fn count_active_todos_excludes_completed() {
        let items = vec![
            todo(TodoChecklistStatus::Completed),
            todo(TodoChecklistStatus::InProgress),
            todo(TodoChecklistStatus::Pending),
            todo(TodoChecklistStatus::Completed),
        ];
        // Only the in-progress + pending count as active.
        assert_eq!(count_active_todos(&items), 2);
    }

    #[test]
    fn active_todo_summary_hides_when_all_completed_or_empty() {
        assert_eq!(active_todo_summary(&[]), None);
        let all_done = vec![
            todo(TodoChecklistStatus::Completed),
            todo(TodoChecklistStatus::Completed),
        ];
        // A finished-but-not-cleared list must not keep claiming work is active.
        assert_eq!(active_todo_summary(&all_done), None);
        let mixed = vec![
            todo(TodoChecklistStatus::Completed),
            todo(TodoChecklistStatus::InProgress),
        ];
        assert_eq!(
            active_todo_summary(&mixed),
            Some("1 todos active".to_string())
        );
    }

    fn sample_state() -> HudState {
        HudState {
            session_identity: None,
            model: ActiveModel {
                provider: "anthropic",
                alias: "opus".to_string(),
                display_name: "claude-opus-4-8".to_string(),
                context_limit: 1_000_000,
            },
            turn_fallback_model: None,
            quota_fallback_model: None,
            ctx_used: 12_400,
            ctx_limit: 1_000_000,
            ctx_new_input: 1_200,
            ctx_cached: 11_200,
            compact_threshold: 450_000,
            cost_usd: 0.08,
            cost_approx: false,
            cwd: PathBuf::from("/tmp"),
            git_branch: Some("main".to_string()),
            perm_mode: PermissionMode::Workspace,
            security_posture: SecurityPosture::SandboxActive,
            effort: None,
            architect_impl: None,
            mcp_servers: Vec::new(),
            bash_count: 0,
            read_count: 0,
            edit_count: 0,
            changed_files: 0,
            todo_summary: None,
            todo_items: Vec::new(),
            automation_lines: Vec::new(),
            lsp_servers: Vec::new(),
            running_agents: 0,
            agents: Vec::new(),
            workflow: None,
            last_tool: None,
            rate_limit: None,
            provider_quotas: Vec::new(),
            auth_origin: None,
            status_line: None,
            team_inbox_unread: 0,
            stale_binary: None,
            background_tasks: 0,
            scheduled_wake: None,
        }
    }

    fn running_workflow() -> WorkflowSummary {
        WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "read-code".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: 2,
            next_phase: None,
            total_agents: 4,
            progress_percent: 50,
            completed_phases: 0,
            completed_agents: 2,
            failed_agents: 0,
            running_agents: 2,
            phases: Vec::new(),
        }
    }

    #[test]
    fn draw_with_heat_resets_every_hud_cell_background_to_terminal_default() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.workflow = Some(running_workflow());
        let area = Rect::new(0, 0, 80, 2);
        let backend = ratatui::backend::TestBackend::new(area.width, area.height);
        let mut terminal = ratatui::Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| {
                frame.render_widget(
                    ratatui::widgets::Block::default()
                        .style(Style::new().bg(theme.palette.code_bg)),
                    area,
                );
                draw_with_heat(
                    frame,
                    area,
                    &state,
                    &theme,
                    false,
                    false,
                    HeatState::Cold,
                );
            })
            .expect("draw");
        let buffer = terminal.backend().buffer();

        assert!(
            matches!(theme.palette.code_bg, Color::Rgb(_, _, _) | Color::Indexed(_)),
            "test requires a colored code background, got {:?}",
            theme.palette.code_bg
        );
        assert!(
            buffer.content().iter().all(|cell| {
                cell.bg == Color::Reset && cell.bg != theme.palette.code_bg
            }),
            "every HUD cell must use the terminal background instead of {:?}",
            theme.palette.code_bg
        );
    }

    #[test]
    fn session_color_hash_is_stable_and_spreads_across_palette() {
        assert_eq!(session_color_index("session-stable"), session_color_index("session-stable"));
        let used = (0..64)
            .map(|index| session_color_index(&format!("session-{index}")))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(used.len(), SESSION_BADGE_PALETTE.len());
    }

    /// Concatenate every span's text so we can assert on the rendered line.
    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    fn line_display_width(line: &Line<'_>) -> usize {
        line.spans
            .iter()
            .map(|span| display_width(span.content.as_ref()))
            .sum()
    }

    /// CC 패리티: `statusLine` 명령 출력이 있으면 커스텀 상태 내용이
    /// 스톡 세그먼트를 대체한다. 단, 현재 모델명은 사용자가 항상 알아야
    /// 하므로 커스텀 상태줄 앞에도 고정 badge 로 유지한다.
    #[test]
    fn custom_status_line_keeps_model_badge() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.status_line = Some("\u{1b}[32m⌁ main\u{1b}[0m · $0.42".to_string());
        let line = compose(&state, &theme, 80, true);
        let text = line_text(&line);
        assert!(text.starts_with(HUD_GUTTER), "quiet left gutter: {text:?}");
        assert!(
            text.contains("claude-opus-4-8"),
            "custom status must still show current model: {text:?}"
        );
        assert!(text.contains("⌁ main · $0.42"), "{text:?}");
        let model_pos = text.find("claude-opus-4-8").expect("model badge position");
        let custom_pos = text.find("⌁ main").expect("custom status position");
        assert!(
            model_pos < custom_pos,
            "model badge should be early enough to scan before custom status: {text:?}"
        );
        assert!(
            !text.contains("write") && !text.contains("$0.08"),
            "other stock segments are still replaced by custom status: {text:?}"
        );
        // 빈 출력은 ledger-visible 조기 축약 규칙으로 폴백.
        state.status_line = Some(String::new());
        let fallback = line_text(&compose(&state, &theme, 80, true));
        assert!(fallback.starts_with(HUD_GUTTER), "{fallback:?}");
        assert!(
            fallback.contains("opus"),
            "model identity stays visible: {fallback:?}"
        );
        assert!(
            fallback.contains("write"),
            "compact HUD should show the live permission mode, not a literal status word: {fallback:?}"
        );
        assert!(!fallback.contains("status"), "{fallback:?}");
    }

    /// An overlong custom `statusLine` must be width-truncated to `cols`,
    /// never spilling past the right edge where the terminal would clip it (and
    /// with it the leading model badge anchor). Uses wide CJK glyphs so the
    /// truncation is exercised in display cells, not bytes.
    #[test]
    fn custom_status_line_truncates_overlong_input_to_cols() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        // ~90 wide cells of custom content (each CJK char = 2 cells) at a 40-col
        // terminal: far past the right edge.
        let long = "\u{c218}\u{c815}".repeat(45);
        state.status_line = Some(long);
        let cols = 40u16;
        let line = compose(&state, &theme, cols, true);
        let width = line_display_width(&line);
        assert!(
            width <= usize::from(cols),
            "custom status line must not exceed {cols} cells, got {width}"
        );
        // The model badge anchor survives the tail truncation.
        let text = line_text(&line);
        assert!(
            text.contains("claude-opus-4-8"),
            "model badge must remain visible after truncation: {text:?}"
        );
        // Tail was clipped, so an ellipsis marks the elision.
        assert!(text.contains('…'), "truncated tail keeps ellipsis: {text:?}");
    }

    #[test]
    fn hud_collapses_stock_segments_when_ledger_visible() {
        let theme = Theme::no_color();
        let state = sample_state();
        let text = line_text(&compose(&state, &theme, 120, true));
        assert!(
            text.starts_with(HUD_GUTTER),
            "quiet left gutter remains: {text:?}"
        );
        assert!(
            text.contains("claude-opus-4-8"),
            "current model identity must remain visible even in compact HUD: {text:?}"
        );
        assert!(
            text.contains("workspace-write"),
            "compact HUD must expose the canonical current permission mode instead of a hardcoded status word: {text:?}"
        );
        assert!(
            !text.contains("status"),
            "compact HUD must not render the old literal status label: {text:?}"
        );
        for duplicate in ["Anthropic", "tokens", "$", "edit", "files"] {
            assert!(
                !text.contains(duplicate),
                "ledger-visible HUD must not duplicate {duplicate:?}: {text:?}"
            );
        }
    }

    #[test]
    fn compact_hud_status_badge_tracks_permission_mode() {
        let theme = Theme::no_color();
        let mut state = sample_state();

        state.perm_mode = PermissionMode::ReadOnly;
        let read_only = line_text(&compose(&state, &theme, 120, true));
        assert!(read_only.contains("read-only"), "{read_only:?}");
        assert!(!read_only.contains("status"), "{read_only:?}");

        state.perm_mode = PermissionMode::Workspace;
        let write = line_text(&compose(&state, &theme, 120, true));
        assert!(write.contains("workspace-write"), "{write:?}");

        state.perm_mode = PermissionMode::All;
        let full_access = line_text(&compose(&state, &theme, 120, true));
        assert!(
            full_access.contains("danger-full-access"),
            "{full_access:?}"
        );
        assert!(!full_access.contains("status"), "{full_access:?}");
    }

    #[test]
    fn compact_hud_status_badge_surfaces_workflow_and_blocked_states() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.workflow = Some(WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "read-code".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: 2,
            next_phase: None,
            total_agents: 2,
            progress_percent: 50,
            completed_phases: 0,
            completed_agents: 0,
            failed_agents: 0,
            running_agents: 2,
            phases: Vec::new(),
        });

        let running = line_text(&compose(&state, &theme, 120, true));
        assert!(running.contains("running"), "{running:?}");

        state.security_posture = SecurityPosture::SandboxBlocked;
        let blocked = line_text(&compose(&state, &theme, 120, true));
        assert_eq!(
            blocked.matches("blocked").count(),
            1,
            "blocked renders exactly once in the compact HUD: {blocked:?}"
        );
        assert!(
            !blocked.contains("running"),
            "blocked wins over running: {blocked:?}"
        );

        let full = line_text(&compose(&state, &theme, 120, false));
        assert_eq!(
            full.matches("blocked").count(),
            1,
            "blocked renders exactly once in the full HUD: {full:?}"
        );
    }

    #[test]
    fn two_row_hud_workflow_row_shows_full_phase_and_percent() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.workflow = Some(running_workflow());

        let status = line_text(&compose(&state, &theme, 60, false));
        assert!(
            !status.contains("phase 1/2"),
            "narrow session row leaves workflow detail to its activity row: {status:?}"
        );
        assert!(status.starts_with(HUD_GUTTER), "quiet session gutter: {status:?}");

        let row = compose_workflow_row(&state, &theme, 60, HeatState::Cold)
            .expect("workflow row present");
        let text = line_text(&row);
        assert!(
            text.starts_with(&format!(" {} ", glyphs::ZO_SPARK_NC)),
            "activity marker preserves the shared three-cell gutter: {text:?}"
        );
        assert!(text.contains("50%"), "activity row shows percent: {text:?}");
        assert!(text.contains("phase 1/2"), "activity row shows phase: {text:?}");
        assert!(!text.contains("% left"), "no redundant remaining half: {text:?}");

        let color_theme = Theme::default_dark();
        let hot = compose_workflow_row(&state, &color_theme, 60, HeatState::Hot)
            .expect("hot workflow row");
        assert_eq!(hot.spans[0].style.fg, Some(color_theme.heat().ember));
        let cooling = compose_workflow_row(
            &state,
            &color_theme,
            60,
            HeatState::Cooling { ramp_idx: 3 },
        )
        .expect("cooling workflow row");
        assert_eq!(
            cooling.spans[0].style.fg,
            Some(color_theme.cooling_fill_color(3))
        );

        state.workflow = None;
        assert!(compose_workflow_row(&state, &theme, 60, HeatState::Cold).is_none());
    }

    #[test]
    fn running_workflow_keeps_permission_mode_visible() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.perm_mode = PermissionMode::All;
        state.workflow = Some(running_workflow());

        let line = compose(&state, &theme, 120, false);
        let permission = line
            .spans
            .iter()
            .find(|span| span.content == "danger-full-access")
            .expect("running workflow must not hide permission mode");
        assert_eq!(
            permission.style.fg,
            permission_style(PermissionMode::All, &theme).fg
        );
    }

    #[test]
    fn hud_shows_context_and_cost_when_ledger_hidden() {
        let theme = Theme::no_color();
        let state = sample_state();
        let text = line_text(&compose(&state, &theme, 120, false));
        assert!(
            text.contains("ctx 2%"),
            "HUD is the single authority; must show context pressure: {text:?}"
        );
        assert!(
            text.contains("$0.08"),
            "HUD is the single authority; must show cost: {text:?}"
        );
        assert!(
            text.contains("claude-opus-4-8"),
            "HUD keeps the resolved model id: {text:?}"
        );
        assert!(!text.contains('━') && !text.contains('▰') && !text.contains('▱'));
    }

    #[test]
    fn hud_view_model_is_the_display_contract() {
        let mut state = sample_state();
        state.ctx_used = 1_370_000;
        state.ctx_limit = 1_000_000;
        state.cost_usd = 0.42;
        state.security_posture = SecurityPosture::SandboxBlocked;
        state.running_agents = 2;
        state.edit_count = 5;
        state.changed_files = 7;
        state.workflow = Some(WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "read-code".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 2,
            total_phases: 4,
            next_phase: Some("synthesize".to_string()),
            total_agents: 12,
            progress_percent: 25,
            completed_phases: 1,
            completed_agents: 3,
            failed_agents: 0,
            running_agents: 9,
            phases: Vec::new(),
        });

        let view = HudViewModel::from_state(&state);

        assert_eq!(view.model, "claude-opus-4-8");
        assert_eq!(view.context, "ctx ~1.0M+");
        assert_eq!(view.cost, "$0.42");
        assert_eq!(view.security, "sandbox:blocked");
        assert_eq!(
            view.workflow.as_deref(),
            Some(
                "25% · phase 2/4 read-code running \u{2192} synthesize · agents 3/12 · 9 running"
            )
        );
        assert_eq!(view.permission_mode, PermissionMode::Workspace);
        assert_eq!(view.security_posture, SecurityPosture::SandboxBlocked);
        assert_eq!(view.running_agents, 2);
        assert_eq!(view.background_tasks, 0);
        assert_eq!(view.edits, 5);
        assert_eq!(view.changed_files, 7);
    }

    #[test]
    fn hud_token_formatter_keeps_precise_million_limits() {
        assert_eq!(format_tokens(1_050_000, false), "~1.05M");
        assert_eq!(format_tokens(1_000_000, false), "~1.0M");
    }

    #[test]
    fn scheduled_countdown_formats_minutes_hours_and_now() {
        assert_eq!(format_scheduled_countdown(9 * 60 + 42), "9:42");
        assert_eq!(format_scheduled_countdown(60 * 60 + 4 * 60), "1h04m");
        assert_eq!(format_scheduled_countdown(0), "now");
    }

    #[test]
    fn hud_shows_mode_and_model_without_provider_chrome() {
        let theme = Theme::no_color();
        let state = sample_state(); // perm_mode = Workspace, provider = anthropic
        let text = line_text(&compose(&state, &theme, 120, false));
        assert!(
            text.contains("workspace-write"),
            "mode badge shown when HUD is the status authority: {text:?}"
        );
        assert!(
            text.contains("claude-opus-4-8"),
            "resolved model id shown: {text:?}"
        );
        // The standalone provider segment was retired as redundant chrome —
        // the sidebar names the provider; the HUD spends the cells on ctx/cost.
        assert!(
            !text.contains("Anthropic"),
            "provider chrome stays off the HUD: {text:?}"
        );
    }

    #[test]
    fn hud_prioritizes_cost_and_security_over_provider_at_medium_width() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.security_posture = SecurityPosture::SandboxBlocked;
        let text = line_text(&compose(&state, &theme, 80, false));

        assert!(
            text.contains("claude-opus-4-8"),
            "resolved model id stays visible: {text:?}"
        );
        assert!(
            text.contains("$0.08"),
            "cost must not be crowded out by provider: {text:?}"
        );
        assert!(
            text.contains("sandbox:blocked"),
            "security must not be crowded out by provider: {text:?}"
        );
        assert!(
            !text.contains("Anthropic"),
            "provider is secondary at medium width: {text:?}"
        );
    }

    #[test]
    fn hud_shows_workflow_phase_when_space_allows() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.workflow = Some(WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "read-code".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 2,
            total_phases: 4,
            next_phase: Some("synthesize".to_string()),
            total_agents: 12,
            progress_percent: 25,
            completed_phases: 1,
            completed_agents: 3,
            failed_agents: 0,
            running_agents: 9,
            phases: Vec::new(),
        });

        let wide = line_text(&compose(&state, &theme, 220, false));
        assert!(
            wide.contains("phase 2/4 read-code running \u{2192} synthesize"),
            "workflow phase badge missing: {wide:?}"
        );
        assert!(
            wide.contains("25%"),
            "workflow progress percentage missing: {wide:?}"
        );
        assert!(
            !wide.contains("% left"),
            "the redundant '% left' half must no longer be shown: {wide:?}"
        );
        assert!(
            wide.contains("agents 3/12") && wide.contains("9 running"),
            "workflow agent tally missing: {wide:?}"
        );

        let narrow = line_text(&compose(&state, &theme, 80, false));
        assert!(
            !narrow.contains("phase 2/4"),
            "narrow HUD should leave workflow detail to sidebar: {narrow:?}"
        );
    }

    #[test]
    fn workflow_hud_label_truncates_long_phase_names() {
        let summary = WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "very-long-phase-name-that-would-crowd-the-hud".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 3,
            total_phases: 5,
            next_phase: Some("very-long-next-phase-name".to_string()),
            total_agents: 8,
            progress_percent: 40,
            completed_phases: 2,
            completed_agents: 4,
            failed_agents: 0,
            running_agents: 4,
            phases: Vec::new(),
        };

        let label = workflow_hud_label(&summary);
        assert!(label.starts_with("40% · phase 3/5 very-long-phase-n…"));
        assert!(label.contains(" running \u{2192} very-long-nex…"));
        assert!(label.contains("agents 4/8 · 4 running"));
    }

    #[test]
    fn workflow_hud_label_truncates_wide_phase_names_by_cell_width() {
        let summary = WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "한국어단계이름이매우김".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: 3,
            next_phase: Some("다음단계이름이매우김".to_string()),
            total_agents: 4,
            progress_percent: 0,
            completed_phases: 0,
            completed_agents: 0,
            failed_agents: 0,
            running_agents: 4,
            phases: Vec::new(),
        };

        let label = workflow_hud_label(&summary);
        let current = label
            .strip_prefix("0% · phase 1/3 ")
            .and_then(|rest| rest.split_once(" running "))
            .map(|(phase, _)| phase)
            .expect("workflow label includes current phase and status");
        let next = label
            .split(" \u{2192} ")
            .nth(1)
            .and_then(|rest| rest.split(" · agents ").next())
            .expect("workflow label includes next phase");

        assert!(display_width(current) <= 18, "{current:?} is too wide");
        assert!(display_width(next) <= 14, "{next:?} is too wide");
        assert!(label.contains("agents 0/4 · 4 running"));
        assert!(
            current.ends_with('…') && next.ends_with('…'),
            "wide phase labels should be visibly clipped: {label:?}"
        );
    }

    #[test]
    fn hud_fill_counts_wide_cells_before_padding_separator() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.workflow = Some(WorkflowSummary {
            name: "code-health".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "한국어단계이름이매우김".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: 3,
            next_phase: Some("다음단계이름이매우김".to_string()),
            total_agents: 4,
            progress_percent: 0,
            completed_phases: 0,
            completed_agents: 0,
            failed_agents: 0,
            running_agents: 4,
            phases: Vec::new(),
        });

        let line = compose(&state, &theme, 140, true);
        assert!(
            line_display_width(&line) <= 140,
            "HUD must not pad past terminal width: {:?}",
            line_text(&line)
        );
    }

    #[test]
    fn model_anchor_shows_only_the_swapped_implementer_while_exec_swap_is_armed() {
        let mut state = sample_state();
        state.model.alias = "fable".to_string();
        state.model.display_name = "claude-fable-5".to_string();
        assert_eq!(model_short_name(&state), "fable");

        // A live swap replaces the anchor outright — no `main▸impl` pair.
        state.architect_impl = Some("gpt-5.6-terra".to_string());
        assert_eq!(model_short_name(&state), "gpt-5.6-terra");
        assert!(!model_short_name(&state).contains('▸'));

        state.architect_impl = None;
        assert_eq!(model_short_name(&state), "fable");
    }

    #[test]
    fn model_anchor_surfaces_refusal_and_quota_fallback_states_width_safely() {
        let mut state = sample_state();
        state.model.alias = "fable".to_string();
        state.model.display_name = "claude-fable-5".to_string();

        let none = model_short_name(&state);
        assert_eq!(none, "fable");

        state.turn_fallback_model = Some("opus".to_string());
        let refusal = model_short_name(&state);
        assert_eq!(refusal, "fable !opus");

        state.turn_fallback_model = None;
        state.quota_fallback_model = Some("opus".to_string());
        let quota = model_short_name(&state);
        assert_eq!(quota, "fable→opus (quota)");

        state.turn_fallback_model = Some("opus".to_string());
        let both = model_short_name(&state);
        assert_eq!(both, "fable→opus (quota) !opus");

        for label in [none, refusal, quota, both] {
            assert_eq!(
                display_width(&label),
                label.chars().count(),
                "fallback label contains a zero- or wide-cell glyph: {label:?}"
            );
        }
    }

    #[test]
    fn model_short_name_preserves_resolved_model_ids() {
        let mut state = sample_state();
        state.model.alias = "opus".to_string();
        state.model.display_name = "claude-opus-4-8".to_string();
        assert_eq!(model_short_name(&state), "claude-opus-4-8");

        state.model.provider = "openai";
        state.model.alias = "openai:gpt-5.5-fast".to_string();
        state.model.display_name = "OpenAI GPT-5.5 Fast".to_string();

        assert_eq!(model_short_name(&state), "gpt-5.5-fast");

        state.model.alias = "gpt".to_string();
        state.model.display_name = "OpenAI GPT-5.5 Fast".to_string();
        assert_eq!(model_short_name(&state), "gpt-5.5-fast");

        state.model.alias.clear();
        state.model.display_name = "OpenAI:o3-mini-high".to_string();
        assert_eq!(model_short_name(&state), "o3-mini-high");

        state.model.display_name = "OpenAI O3 Mini High".to_string();
        assert_eq!(model_short_name(&state), "o3-mini-high");
    }

    #[test]
    fn effort_badge_label_hides_off_and_shows_active_effort() {
        assert_eq!(effort_badge_label(None, "opus"), None);
        assert_eq!(effort_badge_label(Some(Effort::Off), "opus"), None);
        assert_eq!(
            effort_badge_label(Some(Effort::Max), "opus").as_deref(),
            Some("max")
        );
        // Ultra is a static pin projected exactly like every other level:
        // opus clamps the real Ultra tier down to xhigh (no Anthropic ultra
        // wire value).
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "opus").as_deref(),
            Some("ultra→xhigh")
        );
    }

    #[test]
    fn effort_badge_shows_model_specific_gpt_ultra_projection() {
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "gpt-5.6-sol").as_deref(),
            Some("ultra")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "gpt-5.6-terra-2026-07-09").as_deref(),
            Some("ultra")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "gpt-5.6-luna").as_deref(),
            Some("ultra→xhigh")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "gpt-5.5").as_deref(),
            Some("ultra→xhigh")
        );
    }

    #[test]
    fn effort_badge_shows_smart_band_range_per_model() {
        // Sol/terra: the full internal selection band reaches Ultra.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "gpt-5.6-sol").as_deref(),
            Some("smart→xhigh~ultra")
        );
        // Fable/luna: ceiling tops out at `max`.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "claude-fable-5").as_deref(),
            Some("smart→xhigh~max")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "gpt-5.6-luna").as_deref(),
            Some("smart→xhigh~max")
        );
        // Legacy GPT: the ceiling collapses onto the floor — shown as a single
        // value, not a fake range.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "gpt-5.5").as_deref(),
            Some("smart→xhigh")
        );
        // Gemini: caps hard at high — degenerate single value.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "gemini-3.5-flash").as_deref(),
            Some("smart→high")
        );
        // Sonnet: no xhigh, but max is reachable — a genuine [high..max] band.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "claude-sonnet-5").as_deref(),
            Some("smart→high~max")
        );
    }

    #[test]
    fn effort_badge_shows_model_specific_gpt_max_projection() {
        assert_eq!(
            effort_badge_label(Some(Effort::Max), "gpt-5.5").as_deref(),
            Some("max→xhigh")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Max), "gpt-5.6-sol").as_deref(),
            Some("max")
        );
        // xhigh itself also passes through unclamped on GPT.
        assert_eq!(
            effort_badge_label(Some(Effort::Xhigh), "gpt-5.5").as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn compact_percent_measures_pressure_against_the_ceiling() {
        // Pressure is measured against the compaction ceiling, not the nominal
        // window: half-way to a 450k ceiling is 50% pressure even though it is
        // only ~22% of a 1M window. (The ceiling itself is policy-derived —
        // 80% of the window for Claude — and arrives here as a plain number.)
        assert_eq!(compact_percent(225_000, 450_000), Some(50));
        // Saturates at 100 past the ceiling; unknown threshold opts out.
        assert_eq!(compact_percent(600_000, 450_000), Some(100));
        assert_eq!(compact_percent(600_000, 0), None);
    }

    #[test]
    fn canonical_context_pressure_has_one_fallback_for_all_surfaces() {
        let mut state = sample_state();
        state.ctx_used = 225_000;
        state.ctx_limit = 1_000_000;
        state.compact_threshold = 450_000;
        assert_eq!(context_pressure_percent(&state), Some(50));

        state.compact_threshold = 0;
        assert_eq!(context_pressure_percent(&state), Some(22));
        state.ctx_used = 0;
        assert_eq!(context_pressure_percent(&state), None);
        state.ctx_used = 225_000;
        state.ctx_limit = 0;
        assert_eq!(context_pressure_percent(&state), None);
    }

    #[test]
    fn heat_band_color_obeys_all_four_boundaries() {
        let theme = Theme::default_dark();
        assert_eq!(heat_band_color(0.49, &theme), theme.palette.muted);
        assert_eq!(heat_band_color(0.50, &theme), theme.palette.muted);
        assert_eq!(heat_band_color(0.75, &theme), theme.heat().ember);
        assert_eq!(heat_band_color(0.90, &theme), theme.heat().molten);
    }

    #[test]
    fn hud_context_percent_is_text_first_and_uses_static_hierarchy() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.ctx_used = 171_000;
        state.compact_threshold = 450_000;
        let line = compose(&state, &theme, 120, false);
        let text = line_text(&line);

        assert!(text.contains("ctx 38%"), "context stays text-first: {text:?}");
        assert!(!text.contains('▰') && !text.contains('▱'), "gauge chrome is removed: {text:?}");

        let span = |content: &str| {
            line.spans
                .iter()
                .find(|span| span.content == content)
                .unwrap_or_else(|| panic!("missing span {content:?} in {line:?}"))
        };
        assert_eq!(span("ctx 38%").style.fg, Some(theme.palette.muted));
        assert_eq!(span("$0.08").style.fg, Some(theme.palette.muted));
        assert_eq!(span("workspace-write").style.fg, Some(theme.palette.muted));
        assert_eq!(span("claude-opus-4-8").style.fg, Some(theme.palette.fg));
        assert_eq!(line.spans[0].content.as_ref(), HUD_GUTTER);
        assert_eq!(line.spans[0].style.fg, None);
    }

    #[test]
    fn abbreviated_in_turn_hud_keeps_compact_context_text() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.ctx_used = 171_000;
        state.compact_threshold = 450_000;

        let line = compose(&state, &theme, 120, true);
        let text = line_text(&line);
        assert!(
            text.contains("ctx 38%"),
            "the abbreviated HUD retains context pressure: {text:?}"
        );
        assert!(
            !text.contains("$0.08"),
            "other full-HUD details remain abbreviated: {text:?}"
        );
        let context = line
            .spans
            .iter()
            .find(|span| span.content == "ctx 38%")
            .expect("context percentage span");
        assert_eq!(context.style.fg, Some(theme.palette.muted));
    }

    #[test]
    fn hud_activity_marker_alone_follows_heat_state() {
        let theme = Theme::default_dark();
        let state = sample_state();
        let hot = compose_impl(&state, &theme, 120, false, true, true, HeatState::Hot);
        assert_eq!(
            hot.spans[0].content.as_ref(),
            format!(" {} ", glyphs::ZO_SPARK)
        );
        assert_eq!(hot.spans[0].style.fg, Some(theme.heat().ember));
        assert_eq!(display_width(hot.spans[0].content.as_ref()), 3);
        assert!(!line_text(&hot).contains('━'), "heat must not color a full-width rule");

        let cooling = compose_impl(
            &state,
            &theme,
            120,
            false,
            true,
            true,
            HeatState::Cooling { ramp_idx: 3 },
        );
        assert_eq!(
            cooling.spans[0].style.fg,
            Some(theme.cooling_fill_color(3))
        );

        let cooled = compose_impl(
            &state,
            &theme,
            120,
            false,
            true,
            true,
            HeatState::Cooling { ramp_idx: 7 },
        );
        assert_eq!(
            cooled.spans[0].style.fg,
            Some(theme.heat().steel_dim)
        );

        let cold = compose_impl(&state, &theme, 120, false, true, true, HeatState::Cold);
        assert_eq!(cold.spans[0].content.as_ref(), HUD_GUTTER);
        assert_eq!(display_width(cold.spans[0].content.as_ref()), 3);
        assert_eq!(cold.spans[0].style.fg, None);
    }

    #[test]
    fn context_formatter_preserves_absolute_count_for_fallbacks() {
        assert_eq!(format_context_tokens(225_000, 1_000_000), "ctx ~225.0k");
    }

    #[test]
    fn unknown_model_pricing_renders_cost_as_approximate() {
        let mut state = sample_state();
        state.cost_usd = 0.42;
        state.cost_approx = true;
        assert_eq!(HudViewModel::from_state(&state).cost, "~$0.42");
    }

    #[test]
    fn background_task_badge_survives_agent_panel_overlay() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.running_agents = 2;
        state.background_tasks = 1;

        let text = line_text(&compose_with_overlays(
            &state, &theme, 120, true, true, true,
        ));
        assert!(text.contains("bg 1"), "background task badge missing: {text:?}");
        assert!(
            !text.contains("agents"),
            "agent chip should be masked by its expanded panel: {text:?}"
        );
    }

    #[test]
    fn agents_spark_degrades_under_no_color() {
        let mut state = sample_state();
        state.running_agents = 3;

        // Color: the Zo spark leads the agents indicator.
        let rich = line_text(&compose(&state, &Theme::default_dark(), 120, true));
        assert!(
            rich.contains("✦3 agents"),
            "spark is ✦ under color: {rich:?}"
        );

        // NO_COLOR: the spark degrades to its 1-cell ASCII sibling `+` (R10).
        let plain = line_text(&compose(&state, &Theme::no_color(), 120, true));
        assert!(
            !plain.contains('✦'),
            "no rich spark survives NO_COLOR: {plain:?}"
        );
        assert!(
            plain.contains("+3 agents"),
            "spark is + under NO_COLOR: {plain:?}"
        );
    }

    #[test]
    fn hud_uses_whitespace_in_color_and_no_color() {
        let state = sample_state();
        for theme in [Theme::default_dark(), Theme::no_color()] {
            let line = compose(&state, &theme, 120, false);
            let text = line_text(&line);
            assert!(text.starts_with(HUD_GUTTER), "quiet gutter: {text:?}");
            assert!(!text.contains('┗') && !text.contains('━'), "no anvil chrome: {text:?}");
            assert!(!text.contains('▰') && !text.contains('▱'), "no gauge chrome: {text:?}");
            assert!(line_display_width(&line) <= 120);
        }
    }


}
