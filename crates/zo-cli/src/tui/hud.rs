//! Bottom HUD — a quiet, right-aligned session summary.
//!
//! ```text
//!   opus             workspace-write                       ctx 38%  $0.08
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

/// The named session behind the `●` badge.
///
/// Carries the *id*, not a baked `Color`: the badge hue is theme chrome
/// ([`Theme::session_badge_color`]), so it has to be resolved against the live
/// theme at paint time. Baking it here meant the badge kept whichever palette
/// was current when the identity was built and ignored `/theme` entirely — and
/// the palette itself was eight raw ANSI colors, so a session could be tinted
/// the same `Red` a failure uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionIdentity {
    pub name: String,
    /// Session id the badge hue is keyed on (stable for the session's life).
    pub id: String,
}

impl SessionIdentity {
    #[must_use]
    pub fn named(session_id: &str, name: Option<&str>) -> Option<Self> {
        let name = name.map(str::trim).filter(|name| !name.is_empty())?;
        Some(Self {
            name: name.to_string(),
            id: session_id.to_string(),
        })
    }

    /// Badge hue under `theme` — `Reset` under `NO_COLOR`, where the `●` glyph
    /// carries the badge on its own.
    #[must_use]
    pub fn badge_color(&self, theme: &Theme) -> Color {
        theme.session_badge_color(&self.id)
    }
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
    /// Canonical user-facing permission mode label, shared by HUD, sidebar,
    /// and bottom statusline so no surface shows a different name for the
    /// same mode. Display labels, not parser ids: the config/CLI spelling
    /// (`danger-full-access`) stays what commands accept, while every screen
    /// shows the same vocabulary the trust dialog used ("Full access") — the
    /// caution is carried by the mode's warn styling, not by an alarm prefix
    /// echoed at the user for a choice they already made.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Plan => "plan",
            Self::Workspace => "workspace-write",
            Self::All => "full access",
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

/// The plan step the agent is on right now, as one shareable sentence source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NowStep<'a> {
    /// 1-based position of the step in the checklist.
    pub index: usize,
    /// Number of steps in the checklist.
    pub total: usize,
    /// Present-tense sentence to show (`active_form`, falling back to `content`).
    pub text: &'a str,
}

/// Present-tense sentence for one checklist item: trimmed `active_form`,
/// falling back to trimmed `content`. The single wording rule every NOW
/// surface shares — the activity line and chapter go through [`now_step`],
/// the workflow viewer header calls this directly on its joined plan step.
#[must_use]
pub(crate) fn step_sentence(item: &TodoChecklistItem) -> &str {
    let active_form = item.active_form.trim();
    if active_form.is_empty() {
        item.content.trim()
    } else {
        active_form
    }
}

/// The single owner of "what is the agent working on right now" as a sentence.
/// The activity line and the transcript chapter notice read this; the workflow
/// viewer header words its phase through the same [`step_sentence`] rule, so
/// the three surfaces can never word the same step differently. Borrows the
/// checklist — no clone on the render path.
#[must_use]
pub(crate) fn now_step(items: &[TodoChecklistItem]) -> Option<NowStep<'_>> {
    let (index, item) = items
        .iter()
        .enumerate()
        .find(|(_, item)| item.status == TodoChecklistStatus::InProgress)?;
    Some(NowStep {
        index: index + 1,
        total: items.len(),
        text: step_sentence(item),
    })
}

/// The NOW phrase both live surfaces print: `step 3/7 <sentence>`.
///
/// The coordinate is labeled on purpose. It shares a screen with the plan
/// card's `N/M done` tally, and two bare `N/M` counters of the same shape read
/// as one number contradicting itself — this one is a *position*, that one a
/// *completion count*. They legitimately disagree: a plan whose active step is
/// listed first while five others are finished shows position 1 against 5
/// done, which without the word reads as progress running backwards.
///
/// Owned here rather than formatted at each call site, for the reason
/// [`step_sentence`] is: the activity line and the transcript chapter must not
/// be able to word the same step differently.
#[must_use]
pub(crate) fn now_step_label(step: &NowStep<'_>) -> String {
    format!("step {}/{} {}", step.index, step.total, step.text)
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
    /// A transcript block holds the Tab focus, so an empty-composer Enter
    /// expands that block instead of submitting. Derived at paint time from
    /// `transcript.focused_idx()` — never mirrored at the mutation sites — so
    /// the chip cannot outlive the state it reports. Without it the focus is
    /// invisible whenever the focused block has scrolled off-screen, which is
    /// what made a claimed Enter read as a dead Enter key.
    pub block_focused: bool,
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
            Style::new().fg(identity.badge_color(theme))
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

/// Join `fields` into one run separated by thin rules, dropping empty ones.
///
/// Colour rides the *text*, never a background fill. Filled blocks were tried
/// and reverted: a saturated band cannot know what the user's terminal
/// background is, so on a real theme it fought the surface underneath and the
/// labels on top of it went muddy. Foreground hues sit on the background the
/// user already tuned for reading, which is the only one guaranteed to work.
fn join_fields(fields: Vec<Vec<Span<'static>>>, theme: &Theme) -> Vec<Span<'static>> {
    let mut out: Vec<Span<'static>> = Vec::new();
    for field in fields.into_iter().filter(|field| !field.is_empty()) {
        push_separator(&mut out, theme);
        out.extend(field);
    }
    out
}

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
        FooterOwnership {
            ledger_visible,
            agent_panel_visible,
            workflow: false,
        },
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
    owned: FooterOwnership,
    heat_state: HeatState,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    frame.render_widget(
        ratatui::widgets::Block::default().style(Style::new().bg(Color::Reset)),
        area,
    );

    // One row per line, top-down. `compose_rows` decides what fits in the
    // height the layout granted, so nothing here has to re-derive priorities.
    let rows = compose_rows_owned(state, theme, area.width, area.height, owned, heat_state);
    for (index, line) in rows.into_iter().enumerate() {
        let Ok(offset) = u16::try_from(index) else {
            break;
        };
        if offset >= area.height {
            break;
        }
        frame.render_widget(
            Paragraph::new(line),
            Rect {
                y: area.y + offset,
                height: 1,
                ..area
            },
        );
    }
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

    // One left-aligned run carries every field. The bar used to split — totals
    // on the left, model and context anchored right — which made the reader
    // cross a gap of blank cells to assemble a single sentence and left the
    // middle of a wide terminal empty. Nothing on this row is pinned to the
    // right edge any more.
    //
    // Reading order is the model first: it is the most consequential fact on
    // the bar. Fitting order is separate — see [`FIELD_DROP_ORDER`] — so the
    // leftmost field is not automatically the one that survives.
    //
    // Every slot is always pushed, empty when it has nothing to say, because
    // [`FIELD_DROP_ORDER`] indexes into this list by position.
    let model = view.model.trim();
    let fields: Vec<Vec<Span<'static>>> = vec![
        if model.is_empty() {
            Vec::new()
        } else {
            // Accent, and the only bold run the row carries in normal operation.
            vec![Span::styled(
                model.to_string(),
                Style::new()
                    .fg(theme.palette.accent)
                    .add_modifier(Modifier::BOLD),
            )]
        },
        status_spans(&view, theme),
        context_usage_spans(state, &view.context, theme, cols),
        quota_headroom_spans(state, theme),
        if ledger_visible {
            Vec::new()
        } else {
            vec![Span::styled(
                view.cost.clone(),
                Style::new().fg(theme.palette.violet),
            )]
        },
    ];

    // Built before the run and kept out of its budget. These are the only
    // fields on the bar that mean "right now", and the footer is the one
    // surface that always shows them — so a narrow terminal gives up the
    // session's standing facts to keep them, never the reverse.
    let mut live: Vec<Span<'static>> = Vec::new();
    push_live_badges(&mut live, &view, theme);

    let mut spans = hud_leader_spans(theme, heat_state);
    let reserved = spans_display_width(&spans) + spans_display_width(&live);
    spans.extend(fit_fields(
        fields,
        &FIELD_DROP_ORDER,
        theme,
        usize::from(cols).saturating_sub(reserved),
    ));
    spans.extend(live);
    push_agent_badge(&mut spans, &view, theme, show_agents);
    // The badge still belongs here when `show_workflow` survived: a visible
    // ledger does not mean the phase is on screen — a clipped sidebar shows
    // none of it, and then this row is the last surface left.
    push_workflow_badge(&mut spans, &view, theme, cols, show_workflow);
    if !ledger_visible {
        push_change_badge(&mut spans, &view, theme, cols);
    }
    compose_hud_row(spans, Vec::new(), cols)
}

/// Divider between footer fields.
///
/// Whitespace alone stopped being enough once the row carried a gauge, a badge,
/// and a permission mode: the fields ran together into one grey sentence. A thin
/// rule in the border colour groups them without competing with the data.
fn push_separator(spans: &mut Vec<Span<'static>>, theme: &Theme) {
    if spans.is_empty() {
        return;
    }
    spans.push(Span::styled(
        separator_glyph(theme),
        Style::new().fg(theme.palette.border),
    ));
}

fn separator_glyph(theme: &Theme) -> &'static str {
    if theme.no_color { " | " } else { " │ " }
}

/// Width [`join_fields`] will produce, rules included.
fn joined_width(fields: &[Vec<Span<'static>>], theme: &Theme) -> usize {
    let rule = display_width(separator_glyph(theme));
    fields
        .iter()
        .filter(|field| !field.is_empty())
        .enumerate()
        .map(|(index, field)| spans_display_width(field) + usize::from(index > 0) * rule)
        .sum()
}

/// Anchor-row fields, least important first: spend goes before window pressure,
/// which goes before the permission mode, and the model is given up last.
const FIELD_DROP_ORDER: [usize; 5] = [4, 3, 2, 1, 0];

/// Join `fields` into one left-aligned run, dropping whole fields — least
/// important first, per `drop_order` — until it fits `budget`.
///
/// Dropping a field whole is what makes a narrow terminal predictable: the row
/// sheds its cheapest fact entirely instead of ending mid-word, and anything
/// the caller keeps out of `budget` (the live signals) can never be crowded off
/// the bar by a field that matters less than it does.
fn fit_fields(
    mut fields: Vec<Vec<Span<'static>>>,
    drop_order: &[usize],
    theme: &Theme,
    budget: usize,
) -> Vec<Span<'static>> {
    for &index in drop_order {
        if joined_width(&fields, theme) <= budget {
            break;
        }
        if let Some(field) = fields.get_mut(index) {
            field.clear();
        }
    }
    join_fields(fields, theme)
}

/// Inline workflow phase on the anchor row.
///
/// Only drawn when nothing else is carrying the phase: not the footer's own
/// activity row, and not another live surface (the caller's
/// `workflow_owned_elsewhere`). Both the ledger and non-ledger paths go
/// through here, so the fallback cannot be reachable from only one of them.
fn push_workflow_badge(
    spans: &mut Vec<Span<'static>>,
    view: &HudViewModel,
    theme: &Theme,
    cols: u16,
    show_workflow: bool,
) {
    if !show_workflow {
        return;
    }
    let Some(workflow) = &view.workflow else {
        return;
    };
    // One left-aligned run, so what is already placed is the whole budget spent.
    let avail = usize::from(cols).saturating_sub(spans_display_width(spans) + 2);
    if cols >= WORKFLOW_BADGE_MIN_COLS && avail >= WORKFLOW_BADGE_MIN_LABEL {
        spans.push(Span::styled(
            format!("{HUD_GAP}{}", truncate_hud_label(workflow, avail)),
            Style::new().fg(theme.palette.info),
        ));
    }
}

/// Narrower than this and the anchor row has no room for a phase badge on top
/// of the model, context, and cost it already carries.
const WORKFLOW_BADGE_MIN_COLS: u16 = 96;
/// A phase label shorter than this reads as noise rather than information.
const WORKFLOW_BADGE_MIN_LABEL: usize = 12;

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
    spans: &mut Vec<Span<'static>>,
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
    let needed = spans_display_width(spans) + display_width(&label) + display_width(HUD_GAP);
    if needed < usize::from(cols) {
        spans.push(Span::styled(
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

// ── Footer rows ──────────────────────────────────────────────────────────
//
// The footer is a stack, not a single packed line. Each row answers one
// question, which is what lets a narrow terminal drop rows instead of
// silently truncating whichever field happened to sort last:
//
//   1. where am I        `~/2026/forge-code (main) • session`
//   2. what has it cost  `32.1%/1M  $1.284            opus 5 · xhigh`
//   3. what is happening `⠹ edit·hud.rs   agents 2/3   todo 4/7`
//
// Row 2 is the anchor and always renders: the left half accumulates over the
// session, the right half says what is running it and is right-aligned so the
// eye lands in one place. Row 3 exists only while something is live.

/// Rows the footer wants at `state`. Row 3 appears only while a turn, an
/// agent, a workflow, or a todo list is live, so an idle session stays quiet.
#[must_use]
pub fn desired_rows(state: &HudState) -> u16 {
    if activity_row_has_content(state) {
        ACTIVITY_ROW_INDEX + 1
    } else {
        ACTIVITY_ROW_INDEX
    }
}

/// Whether something is *live* — not merely whether the session has done work.
///
/// `edit_count` and `changed_files` are session totals that only ever go up, so
/// treating them as liveness gave the footer a third row from the first edit
/// onward and never gave it back. The dirty-tree size now sits beside the branch
/// it describes on the location row, where a running total belongs.
fn activity_row_has_content(state: &HudState) -> bool {
    state.workflow.is_some()
        || state.running_agents > 0
        || state.background_tasks > 0
        || state
            .todo_items
            .iter()
            .any(|item| item.status == TodoChecklistStatus::InProgress)
}

/// Build the footer rows for `cols`, tallest first.
///
/// `rows` is the height the layout actually granted; the caller may have less
/// room than [`desired_rows`] asked for. Dropping is by row, and in a fixed
/// order — the location row goes first, then the activity row — so what
/// survives a narrow terminal is predictable rather than a function of which
/// field the composer happened to push last.
#[must_use]
pub fn compose_rows(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    rows: u16,
    ledger_visible: bool,
    agent_panel_visible: bool,
    heat_state: HeatState,
) -> Vec<Line<'static>> {
    compose_rows_owned(
        state,
        theme,
        cols,
        rows,
        FooterOwnership {
            ledger_visible,
            agent_panel_visible,
            workflow: false,
        },
        heat_state,
    )
}

/// What other surfaces are already showing, so the footer never repeats them.
///
/// `workflow` is deliberately separate from `ledger_visible`: a *visible*
/// sidebar does not necessarily *carry* the phase — a clipped one shows none of
/// it — and in that case the footer is the only surface left that can.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FooterOwnership {
    /// The right ledger is on screen and owns the session metadata.
    pub ledger_visible: bool,
    /// The pinned agent panel is on screen and owns the fleet detail.
    pub agent_panel_visible: bool,
    /// Another live surface already names the workflow phase.
    pub workflow: bool,
}

/// [`compose_rows`] with the full ownership picture (see [`FooterOwnership`]).
#[must_use]
pub fn compose_rows_owned(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    rows: u16,
    owned: FooterOwnership,
    heat_state: HeatState,
) -> Vec<Line<'static>> {
    // The activity row owns the workflow phase when it is on screen. When the
    // granted height cannot fit it, the anchor row takes the phase back as an
    // inline badge — the phase must never fall off the screen entirely just
    // because the terminal is short.
    let activity = (rows > ACTIVITY_ROW_INDEX)
        .then(|| compose_activity_row(state, theme, cols, heat_state))
        .flatten();
    let anchor = compose_with_overlays_and_heat(
        state,
        theme,
        cols,
        owned.ledger_visible,
        owned.agent_panel_visible,
        activity.is_none() && !owned.workflow,
        heat_state,
    );
    match rows {
        0 => Vec::new(),
        1 => vec![anchor],
        _ => {
            let mut out = vec![compose_location_row(state, theme, cols), anchor];
            out.extend(activity);
            out
        }
    }
}

/// Index of the live-activity row within the footer stack. Also the height at
/// which it first fits, since rows are filled top-down.
const ACTIVITY_ROW_INDEX: u16 = 2;

/// Row 1 — `~/parent/project (branch) • session`, all dim.
fn compose_location_row(state: &HudState, theme: &Theme, cols: u16) -> Line<'static> {
    // Path, branch, and session are three different kinds of fact, so they get
    // three hues instead of one undifferentiated grey run. The palette already
    // names these roles: cyan is the cwd, teal the branch.
    // The same gutter every other footer row leads with, so the stack shares one
    // left margin instead of the path starting a column left of the rest.
    let mut left = vec![
        Span::raw(HUD_GUTTER),
        // Hue, not weight. Bold is spent on the anchor row's model and on the
        // two alarm states below; a bold path and a bold branch here gave the
        // footer three competing focal points and no single one.
        Span::styled(
            compact_cwd_for_footer(state),
            Style::new().fg(theme.palette.cyan),
        ),
    ];
    if let Some(branch) = state.git_branch.as_deref().map(str::trim) {
        if !branch.is_empty() {
            push_separator(&mut left, theme);
            left.push(Span::styled(
                branch.to_string(),
                Style::new().fg(theme.palette.teal),
            ));
        }
    }
    // Dirty-tree size belongs next to the branch it applies to, not buried in
    // the badge run on the anchor row where it competed with live signals.
    if state.changed_files > 0 {
        push_separator(&mut left, theme);
        left.push(Span::styled(
            format!("{} files", state.changed_files),
            Style::new().fg(theme.palette.warn),
        ));
    }
    if let Some(identity) = state.session_identity.as_ref() {
        let name = identity.name.trim();
        if !name.is_empty() {
            left.push(Span::styled(
                " • ",
                Style::new().fg(theme.palette.faint),
            ));
            let style = if theme.no_color {
                Style::default()
            } else {
                Style::new().fg(identity.badge_color(theme))
            };
            left.push(Span::styled(name.to_string(), style));
        }
    }
    // Tab-focus is a mode the composer does not otherwise show: the focused
    // block is routinely scrolled out of view, and then the only evidence of
    // it is an Enter that expands something invisible instead of sending.
    if state.block_focused {
        push_separator(&mut left, theme);
        left.push(Span::styled(
            "block focused".to_string(),
            Style::new().fg(theme.palette.accent),
        ));
        left.push(Span::styled(
            " · Esc".to_string(),
            Style::new().fg(theme.palette.dim),
        ));
    }
    // A newer build landed on disk while this session runs old code. The
    // sidebar has carried this warning all along, but inline mode hides the
    // sidebar — so inline sessions (the default) quietly kept running stale
    // binaries through a whole afternoon of deploys. The footer is the one
    // surface every mode always shows.
    if state.stale_binary.is_some() {
        push_separator(&mut left, theme);
        left.push(Span::styled(
            "⟳ new build on disk — /restart".to_string(),
            Style::new()
                .fg(theme.palette.warn)
                .add_modifier(Modifier::BOLD),
        ));
    }
    // The hints ride the same left-aligned run as the rest of the footer;
    // nothing on any row is pinned to the right edge any more.
    left.extend(footer_hint_spans(theme, cols));
    compose_hud_row(left, Vec::new(), cols)
}

/// Narrower than this and the path itself is worth more than a key hint.
const FOOTER_HINT_MIN_COLS: u16 = 72;

/// Tail of the location row: two keys the composer's own placeholder does not
/// already advertise, so the hint adds reach rather than repeating itself.
fn footer_hint_spans(theme: &Theme, cols: u16) -> Vec<Span<'static>> {
    if cols < FOOTER_HINT_MIN_COLS {
        return Vec::new();
    }
    let hint = if theme.no_color {
        "^B panel   ^C exit"
    } else {
        "⌃B panel   ⌃C exit"
    };
    vec![Span::styled(
        format!("{FIELD_GAP}{hint}"),
        Style::new().fg(theme.palette.dim),
    )]
}

/// `~/parent/project`, matching the sidebar's own abbreviation so the two
/// surfaces never disagree about where the session is.
fn compact_cwd_for_footer(state: &HudState) -> String {
    state.cwd.parent().and_then(|path| path.file_name()).map_or_else(
        || "~".to_string(),
        |parent| {
            let project = state
                .cwd
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            format!("~/{}/{project}", parent.to_string_lossy())
        },
    )
}

/// Row 3 — what is live right now. `None` when nothing is.
fn compose_activity_row(
    state: &HudState,
    theme: &Theme,
    cols: u16,
    heat_state: HeatState,
) -> Option<Line<'static>> {
    let view = HudViewModel::from_state(state);
    let mut spans = vec![activity_marker_span(theme, heat_state, theme.palette.dim)];

    // Each field carries its own hue so the row can be read by colour at a
    // glance — which of these is live matters more than the text itself.
    let mut parts: Vec<(String, Color)> = Vec::new();
    if let Some(workflow) = view.workflow.as_deref() {
        parts.push((workflow.to_string(), theme.palette.info));
    }
    if view.running_agents > 0 {
        parts.push((
            format!("agents {}", view.running_agents),
            theme.palette.cyan,
        ));
    }
    if view.background_tasks > 0 {
        parts.push((
            format!("bg {}", view.background_tasks),
            theme.palette.violet,
        ));
    }
    // Session totals (edits, changed files) are deliberately absent: they live
    // on the location row next to the branch they describe. Repeating them here
    // is what used to keep this row on screen long after anything was live.
    if parts.is_empty() {
        return None;
    }

    let budget = usize::from(cols);
    for (index, (text, color)) in parts.into_iter().enumerate() {
        let used = spans_display_width(&spans);
        let lead = if index == 0 { "" } else { FIELD_GAP };
        let remaining = budget.saturating_sub(used + display_width(lead));
        if remaining == 0 {
            break;
        }
        if !lead.is_empty() {
            spans.push(Span::raw(lead));
        }
        spans.push(Span::styled(
            truncate_hud_label(&text, remaining),
            Style::new().fg(color),
        ));
    }
    Some(Line::from(spans))
}

/// What this session is allowed to do.
///
/// A blocked sandbox displaces the mode entirely — when the sandbox is refusing
/// work, that is the answer to "why can it not do this", and the permission mode
/// is the less urgent half of the same question.
fn status_spans(view: &HudViewModel, theme: &Theme) -> Vec<Span<'static>> {
    if view.security_posture == SecurityPosture::SandboxBlocked {
        return vec![Span::styled(
            view.security.clone(),
            security_posture_style(view.security_posture, theme),
        )];
    }
    let mut spans = Vec::new();
    if view
        .workflow
        .as_deref()
        .is_some_and(|workflow| workflow.contains("running"))
    {
        spans.push(Span::styled("running", Style::new().fg(theme.palette.muted)));
        spans.push(Span::raw(HUD_GAP));
    }
    spans.push(Span::styled(
        view.permission_mode.label(),
        hud_mode_style(view.permission_mode, theme),
    ));
    spans
}

fn hud_mode_style(mode: PermissionMode, theme: &Theme) -> Style {
    if mode == PermissionMode::All {
        permission_style(mode, theme)
    } else {
        Style::new().fg(theme.palette.muted)
    }
}

fn security_posture_style(posture: SecurityPosture, theme: &Theme) -> Style {
    let color = match posture {
        SecurityPosture::SandboxActive => theme.palette.success,
        SecurityPosture::SandboxBlocked => theme.palette.warn,
        SecurityPosture::SandboxOff => theme.palette.error,
    };
    Style::new().fg(color).add_modifier(Modifier::BOLD)
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
    // Live signals stay on the left with everything else, and their width comes
    // out of the custom content's budget first: an overlong status line must
    // lose its own tail rather than push a live signal off the bar.
    let mut live = Vec::new();
    push_live_badges(&mut live, view, theme);

    let mut spans = hud_leader_spans(theme, heat_state);
    if !view.model.trim().is_empty() {
        // The same accent and weight the stock row gives it, so the model reads
        // as one field whichever composer drew the line.
        spans.push(Span::styled(
            view.model.clone(),
            Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD),
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
    truncate_spans(
        &mut spans,
        usize::from(cols).saturating_sub(spans_display_width(&live)),
    );
    spans.extend(live);
    compose_hud_row(spans, Vec::new(), cols)
}

fn hud_leader_spans(theme: &Theme, heat_state: HeatState) -> Vec<Span<'static>> {
    // The mark is drawn at rest too, in the brand accent. A blank gutter gave
    // the eye nothing to land on and the whole bar read as background noise;
    // the glyph is the same three cells either way, so nothing shifts when a
    // turn starts and the mark takes on its heat colour.
    vec![activity_marker_span(theme, heat_state, theme.palette.accent)]
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

/// Placeholder shown until the first usage report lands.
///
/// An em-dash, not the word `pending`: this sits where a number goes, and the
/// row already has a shape (label + empty gauge) that says "not measured yet".
/// A word there read as a *status* — as though something were stuck waiting —
/// and it was the widest thing on an otherwise numeric row.
const CTX_UNMEASURED: &str = "ctx \u{2014}";

fn format_context_tokens(used: u64, limit: u64) -> String {
    if used == 0 {
        return CTX_UNMEASURED.to_string();
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

/// Cells in the anchor row's context gauge.
const CTX_GAUGE_CELLS: usize = 10;

/// Pi separates footer fields with whitespace, not glyph dividers — the columns
/// read as columns without a rule between them.
const FIELD_GAP: &str = "   ";

/// Narrower than this and the bar costs more than it tells.
///
/// The bar plus its spelled-out label runs about eighteen columns, and at 80 it
/// was pushing spend off the right edge into an ellipsis. The percent alone
/// carries the same information, so below this the row keeps the number and
/// drops the picture.
const CTX_GAUGE_MIN_COLS: u16 = 100;

/// Colour for the gauge itself.
///
/// Unlike [`heat_band_color`], a healthy session still gets a colour: the gauge
/// is a status light that is always on, not a warning that only appears once
/// something is wrong. The bands above 75% are shared with every other surface
/// so "amber means compaction is near" reads the same everywhere.
fn ctx_gauge_color(ratio: f64, theme: &Theme) -> Color {
    if ratio < 0.75 {
        theme.palette.success
    } else {
        heat_band_color(ratio, theme)
    }
}

/// Context pressure: a label, a ten-cell gauge, and the percent.
///
/// The gauge is the one *drawn* object on an otherwise textual row, which is
/// what makes the bar readable at a glance without painting anything. Its lit
/// cells take the pressure band, the unlit ones stay faint, so the bar reads by
/// length as well as by colour.
fn context_usage_spans(
    state: &HudState,
    fallback: &str,
    theme: &Theme,
    cols: u16,
) -> Vec<Span<'static>> {
    // `None` before the first usage report — the gauge is still drawn, empty, so
    // the footer keeps its shape instead of collapsing to a word and then
    // jumping a dozen columns wide once the first token count lands.
    let percent = (state.ctx_used > 0)
        .then(|| context_pressure_percent(state))
        .flatten();
    let color = percent.map_or(theme.palette.faint, |percent| {
        ctx_gauge_color(f64::from(u32::try_from(percent).unwrap_or(100)) / 100.0, theme)
    });
    // Shown as headroom rather than consumption. "70%" beside a context bar is
    // genuinely ambiguous — half of readers take it for what is left — and the
    // number people act on is how much room remains before compaction.
    let remaining = percent.map(|percent| 100u64.saturating_sub(percent));

    // Spelled out when there is room: a bare `ctx` in front of a bar reads as
    // chrome, `Context` reads as a label for the number beside it.
    let label = if cols >= CTX_GAUGE_MIN_COLS {
        "Context "
    } else {
        "ctx "
    };
    let mut spans = vec![Span::styled(label, Style::new().fg(theme.palette.dim))];
    if cols >= CTX_GAUGE_MIN_COLS {
        let rich = !theme.no_color;
        let fill = glyphs::pick(rich, glyphs::GAUGE_HUD_FILL, glyphs::GAUGE_HUD_FILL_NC);
        let empty = glyphs::pick(rich, glyphs::GAUGE_HUD_EMPTY, glyphs::GAUGE_HUD_EMPTY_NC);
        // The bar measures what is LEFT, so it drains as the session fills —
        // the direction every fuel gauge trains people to read. Rounding up
        // keeps one lit cell while any headroom survives, so "nearly out" and
        // "actually out" stay distinguishable.
        let filled = remaining.map_or(0, |remaining| {
            usize::try_from(remaining)
                .unwrap_or(100)
                .saturating_mul(CTX_GAUGE_CELLS)
                .div_ceil(100)
                .clamp(1, CTX_GAUGE_CELLS)
        });
        spans.push(Span::styled(fill.repeat(filled), Style::new().fg(color)));
        spans.push(Span::styled(
            empty.repeat(CTX_GAUGE_CELLS - filled),
            Style::new().fg(theme.palette.faint),
        ));
        spans.push(Span::raw(" "));
    }
    spans.push(Span::styled(
        percent.map_or_else(
            // The fallback carries its own `ctx ` prefix for surfaces that render
            // it bare; here the label is already drawn, so strip it rather than
            // printing "Context ctx —".
            || fallback.trim().trim_start_matches("ctx ").trim().to_string(),
            |_| {
                format!(
                    "{}% left",
                    remaining.unwrap_or_default()
                )
            },
        ),
        Style::new().fg(color),
    ));
    spans
}

/// Every signed-in provider's tightest window, stated the way the context
/// gauge is.
///
/// One row per provider, not one row overall: the account that stops working
/// first is not always the one being spent right now, and a session that hops
/// between models needs to see all of them without looking away from the
/// footer. Within a provider only the nearest window shows — a comfortable
/// weekly allowance is not worth a slot while the five-hour window empties.
///
/// Worded `NN% left` deliberately. Providers report consumption and every
/// surface here shows headroom; a bare percent beside a provider name gives the
/// reader no way to tell which direction it runs.
fn quota_headroom_spans(state: &HudState, theme: &Theme) -> Vec<Span<'static>> {
    let mut tightest: Vec<&api::quota::ProviderQuotaView> = Vec::new();
    for view in &state.provider_quotas {
        if view.remaining_percent.is_none() {
            continue;
        }
        match tightest
            .iter_mut()
            .find(|kept| kept.provider == view.provider)
        {
            Some(kept) if view.remaining_percent < kept.remaining_percent => *kept = view,
            Some(_) => {}
            None => tightest.push(view),
        }
    }
    if tightest.is_empty() {
        return Vec::new();
    }

    let mut spans = Vec::with_capacity(tightest.len() * 3);
    for view in tightest {
        let Some(remaining) = view.remaining_percent else {
            continue;
        };
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", Style::new().fg(theme.palette.faint)));
        }
        // Colour keys off consumption, matching the rail and the context gauge.
        let used = 100u8.saturating_sub(remaining);
        let approx = if view.estimated { "~" } else { "" };
        spans.push(Span::styled(
            format!("{} ", view.provider.rate_limit_key()),
            Style::new()
                .fg(provider_tint(view.provider, theme))
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("{approx}{remaining}% left"),
            Style::new().fg(ctx_gauge_color(f64::from(used) / 100.0, theme)),
        ));
        // When it comes back, in the same clock language the rail uses. A
        // percentage says how bad it is; only this says how long it lasts.
        if let Some(reset) = view.resets_at_unix.and_then(|at| i64::try_from(at).ok()) {
            spans.push(Span::styled(
                format!(" ↺{}", core_types::date::local_weekday_clock(reset)),
                Style::new().fg(theme.palette.faint),
            ));
        }
    }
    spans
}

/// A stable hue per provider, so three quota figures on one row separate at a
/// glance instead of reading as one run of dim text.
///
/// The hue names *which account*; the percent beside it keeps the health ramp
/// (green through red). Two colour jobs on one row only work because they sit
/// on different words — the name never changes colour as the quota drains, and
/// the number never carries brand.
fn provider_tint(kind: api::ProviderKind, theme: &Theme) -> Color {
    match kind {
        api::ProviderKind::Anthropic => theme.palette.accent,
        api::ProviderKind::OpenAi => theme.palette.teal,
        api::ProviderKind::Google => theme.palette.info,
        api::ProviderKind::Xai => theme.palette.violet,
        api::ProviderKind::Ollama => theme.palette.muted,
    }
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
    fn location_row_carries_the_stale_binary_warning() {
        // The stale-binary cue lived only in the sidebar, which inline mode
        // (the default) hides — sessions ran a whole day of deploys behind
        // with no visible sign. The footer location row is the one surface
        // every mode shows, so the warning must ride it.
        let theme = Theme::pi();
        let mut state = sample_state();
        state.stale_binary = None;
        let clean: String = compose_location_row(&state, &theme, 160)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(!clean.contains("/restart"), "{clean}");

        state.stale_binary = Some(super::super::stale_binary::StaleBinaryInfo {
            disk_mtime: 1_753_000_000,
        });
        let warned: String = compose_location_row(&state, &theme, 160)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(
            warned.contains("new build on disk") && warned.contains("/restart"),
            "{warned}"
        );
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

    fn step(content: &str, active_form: &str, status: TodoChecklistStatus) -> TodoChecklistItem {
        TodoChecklistItem {
            step_id: None,
            content: content.to_string(),
            status,
            active_form: active_form.to_string(),
        }
    }

    #[test]
    fn now_step_picks_the_first_in_progress_item() {
        let items = vec![
            step("Write parser", "Writing parser", TodoChecklistStatus::Completed),
            step("Verify parser", "Verifying parser", TodoChecklistStatus::InProgress),
            step("Ship parser", "Shipping parser", TodoChecklistStatus::InProgress),
            step("Document parser", "Documenting parser", TodoChecklistStatus::Pending),
        ];
        let now = now_step(&items).expect("an in-progress step exists");
        assert_eq!(now.index, 2, "1-based position of the first in-progress item");
        assert_eq!(now.total, 4);
        assert_eq!(now.text, "Verifying parser");
    }

    /// The label has to name its number, because the plan card beside it shows
    /// a same-shaped `N/M done` tally that counts something else — and the two
    /// legitimately disagree, as here: position 1 while 2 of 3 are finished.
    #[test]
    fn the_now_label_names_its_number_a_step_position() {
        let items = vec![
            step("Ship it", "Shipping it", TodoChecklistStatus::InProgress),
            step("Write parser", "Writing parser", TodoChecklistStatus::Completed),
            step("Verify parser", "Verifying parser", TodoChecklistStatus::Completed),
        ];
        let now = now_step(&items).expect("an in-progress step exists");

        assert_eq!(now_step_label(&now), "step 1/3 Shipping it");
        assert!(
            !now_step_label(&now).starts_with("1/3"),
            "a bare coordinate reads as the plan card's done-tally"
        );
    }

    #[test]
    fn now_step_falls_back_to_content_when_active_form_is_blank() {
        let items = vec![step("Verify parser", "   ", TodoChecklistStatus::InProgress)];
        let now = now_step(&items).expect("an in-progress step exists");
        assert_eq!(now.text, "Verify parser", "blank active_form falls back to content");
    }

    #[test]
    fn now_step_returns_none_without_an_in_progress_item() {
        assert_eq!(now_step(&[]), None);
        let items = vec![
            step("Write parser", "Writing parser", TodoChecklistStatus::Completed),
            step("Verify parser", "Verifying parser", TodoChecklistStatus::Pending),
        ];
        assert_eq!(now_step(&items), None);
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
            block_focused: false,
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
                    FooterOwnership::default(),
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
        use crate::tui::theme::{SESSION_BADGE_HUES, session_badge_index};

        assert_eq!(
            session_badge_index("session-stable"),
            session_badge_index("session-stable")
        );
        let used = (0..64)
            .map(|index| session_badge_index(&format!("session-{index}")))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(used.len(), SESSION_BADGE_HUES.len());

        // The badge hue is now theme chrome, not a hardcoded ANSI slot: two
        // different sessions get different tints, and none of them is a raw
        // `Color::Red`/`Color::Green` that could read as a status.
        let theme = Theme::default_dark();
        let identity = SessionIdentity::named("session-123", Some("deploy watch"))
            .expect("named identity");
        assert_eq!(identity.badge_color(&theme), theme.session_badge_color("session-123"));
        assert!(!matches!(
            identity.badge_color(&theme),
            Color::Red | Color::Green | Color::Yellow | Color::Blue
        ));
    }

    /// The three-cell leader the anchor row starts with: the mark, spaced. Its
    /// width is what the footer stack's left margin depends on; the glyph itself
    /// differs between colour and `NO_COLOR`.
    fn leader_text(theme: &Theme) -> String {
        format!(
            " {} ",
            glyphs::pick(!theme.no_color, glyphs::ZO_SPARK, glyphs::ZO_SPARK_NC)
        )
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
        assert!(text.starts_with(&leader_text(&theme)), "row leads with the mark: {text:?}");
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
        assert!(fallback.starts_with(&leader_text(&theme)), "{fallback:?}");
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
            text.starts_with(&leader_text(&theme)),
            "row still leads with the mark: {text:?}"
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
            full_access.contains("full access"),
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

        // The phase badge keeps reporting its own status even while blocked —
        // the phase must never vanish from the screen. What `blocked wins over
        // running` is about is the *session status* segment, so drop the
        // workflow to isolate it.
        let mut without_workflow = state.clone();
        without_workflow.workflow = None;
        let blocked_only = line_text(&compose(&without_workflow, &theme, 120, true));
        assert!(
            !blocked_only.contains("running"),
            "blocked wins over running in the status segment: {blocked_only:?}"
        );

        let full = line_text(&compose(&state, &theme, 120, false));
        assert_eq!(
            full.matches("blocked").count(),
            1,
            "blocked renders exactly once in the full HUD: {full:?}"
        );
    }

    /// Live detail belongs to the activity row, not the anchor row.
    #[test]
    fn activity_row_carries_workflow_detail_off_the_anchor_row() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        state.workflow = Some(running_workflow());

        let anchor = line_text(&compose(&state, &theme, 60, false));
        assert!(
            !anchor.contains("phase 1/2"),
            "the anchor row leaves workflow detail to the activity row: {anchor:?}"
        );

        let row = compose_activity_row(&state, &theme, 60, HeatState::Cold)
            .expect("activity row present");
        let text = line_text(&row);
        assert!(
            text.starts_with(&format!(" {} ", glyphs::ZO_SPARK_NC)),
            "activity marker keeps the shared three-cell gutter: {text:?}"
        );
        assert!(text.contains("50%"), "activity row shows percent: {text:?}");
        assert!(text.contains("phase 1/2"), "activity row shows phase: {text:?}");

        let color_theme = Theme::default_dark();
        let hot = compose_activity_row(&state, &color_theme, 60, HeatState::Hot)
            .expect("hot activity row");
        assert_eq!(hot.spans[0].style.fg, Some(color_theme.heat().ember));
        let cooling = compose_activity_row(
            &state,
            &color_theme,
            60,
            HeatState::Cooling { ramp_idx: 3 },
        )
        .expect("cooling activity row");
        assert_eq!(
            cooling.spans[0].style.fg,
            Some(color_theme.cooling_fill_color(3))
        );
    }

    /// An idle session gets two rows; a live one gets the third.
    #[test]
    fn activity_row_appears_only_while_something_is_live() {
        let theme = Theme::no_color();
        let mut idle = sample_state();
        idle.workflow = None;
        idle.running_agents = 0;
        idle.background_tasks = 0;
        idle.edit_count = 0;
        idle.changed_files = 0;

        assert_eq!(desired_rows(&idle), 2);
        assert!(compose_activity_row(&idle, &theme, 60, HeatState::Cold).is_none());
        assert_eq!(
            compose_rows(&idle, &theme, 60, 3, false, false, HeatState::Cold).len(),
            2,
            "an idle session must not pad out a blank third row"
        );

        let mut live = idle.clone();
        live.running_agents = 2;
        assert_eq!(desired_rows(&live), 3);
        let rows = compose_rows(&live, &theme, 60, 3, false, false, HeatState::Cold);
        assert_eq!(rows.len(), 3);
        assert!(line_text(&rows[2]).contains("agents 2"), "{:?}", line_text(&rows[2]));
    }

    /// Row 1 names where the session is; the anchor row stays put underneath.
    #[test]
    fn location_row_shows_cwd_branch_and_session_name() {
        let theme = Theme::no_color();
        let mut state = sample_state();
        // A nested path so the `~/parent/project` abbreviation has both halves
        // to work with; built from the temp dir rather than a literal so the
        // test carries no machine-specific path.
        state.cwd = std::env::temp_dir().join("workspace").join("project");
        state.git_branch = Some("main".to_string());

        let rows = compose_rows(&state, &theme, 80, 2, false, false, HeatState::Cold);
        assert_eq!(rows.len(), 2);
        let location = line_text(&rows[0]);
        assert!(
            location.trim_start().starts_with("~/"),
            "cwd abbreviates to ~: {location:?}"
        );
        assert!(
            location.starts_with(HUD_GUTTER),
            "location shares the footer's left margin: {location:?}"
        );
        assert!(location.contains("main"), "branch is a column: {location:?}");
        // The right edge carries the key hint rather than sitting blank.
        assert!(location.trim_end().ends_with("exit"), "{location:?}");

        // Height 1 keeps the anchor row — the one that names the model — and
        // drops the location, never the other way round.
        let one = compose_rows(&state, &theme, 80, 1, false, false, HeatState::Cold);
        assert_eq!(one.len(), 1);
        assert_eq!(line_text(&one[0]), line_text(&compose(&state, &theme, 80, false)));
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
            .find(|span| span.content == "full access")
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
            text.contains("98% left"),
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
        // Scoped to the workflow badge: the context gauge legitimately reads
        // "N% left" now, and an unscoped search would catch it instead of the
        // duplicated progress half this guards.
        let workflow = wide.split("$0.08").nth(1).unwrap_or_default();
        assert!(
            !workflow.contains("% left"),
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
        // Ultra is a static pin projected exactly like every other level.
        // Anthropic has no `ultra` wire value, so opus lands on its real top
        // rung, `max`.
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "opus").as_deref(),
            Some("ultra→max")
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
        // Luna exposes no internal Ultra rung, so it shows its own ceiling.
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "gpt-5.6-luna").as_deref(),
            Some("ultra→max")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Ultra), "gpt-5.5").as_deref(),
            Some("ultra→xhigh")
        );
    }

    #[test]
    fn effort_badge_shows_smart_band_range_per_model() {
        // Sol/terra: `max`, not `ultra` — automatic escalation stops one rung
        // below the model's own ceiling; `ultra` needs the explicit pin.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "gpt-5.6-sol").as_deref(),
            Some("smart→xhigh~max")
        );
        // Fable/luna: the model's own ceiling is already `max`.
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
        // Sonnet 5 has xhigh; Sonnet 4.6 does not, so only 4.6 shows the
        // clamped [high..max] band.
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "claude-sonnet-5").as_deref(),
            Some("smart→xhigh~max")
        );
        assert_eq!(
            effort_badge_label(Some(Effort::Smart), "claude-sonnet-4-6").as_deref(),
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

    /// Context pressure is a gauge plus its percent, and each footer field
    /// carries its own hue.
    ///
    /// The gauge is the point: a bar is read at a glance where a number has to
    /// be parsed. The percent stays alongside it so the exact value is never
    /// lost, and the colour bands are shared with every other surface.
    #[test]
    fn hud_context_shows_a_coloured_gauge_beside_its_percent() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.ctx_used = 171_000;
        state.compact_threshold = 450_000;
        let line = compose(&state, &theme, 120, false);
        let text = line_text(&line);

        assert!(text.contains("62% left"), "the exact value survives: {text:?}");
        assert!(text.contains('▰') && text.contains('▱'), "gauge is drawn: {text:?}");

        let span = |content: &str| {
            line.spans
                .iter()
                .find(|span| span.content == content)
                .unwrap_or_else(|| panic!("missing span {content:?} in {line:?}"))
        };
        // Under 75% of the compaction ceiling there is still room, so the gauge
        // reads as healthy rather than as a warning.
        assert_eq!(span("62% left").style.fg, Some(theme.palette.success));
        assert_eq!(span("$0.08").style.fg, Some(theme.palette.violet));
        assert_eq!(span("workspace-write").style.fg, Some(theme.palette.muted));
        // The model leads the run in the brand accent and is the only bold field.
        let model = span("claude-opus-4-8");
        assert_eq!(model.style.fg, Some(theme.palette.accent));
        assert!(model.style.add_modifier.contains(Modifier::BOLD));
        // Colour rides the text: nothing on this row paints a background.
        assert!(
            line.spans.iter().all(|span| span.style.bg.is_none()),
            "the anchor row must not fill a background: {line:?}"
        );
        assert_eq!(line.spans[0].content.as_ref(), leader_text(&theme));
        assert_eq!(line.spans[0].style.fg, Some(theme.palette.accent));
    }

    /// The bar is the first thing dropped when the terminal is narrow: the
    /// percent alone carries the same information in a fraction of the columns.
    #[test]
    fn narrow_terminal_keeps_the_percent_and_drops_the_gauge() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.ctx_used = 171_000;
        state.compact_threshold = 450_000;

        let narrow = line_text(&compose(&state, &theme, CTX_GAUGE_MIN_COLS - 1, false));
        assert!(narrow.contains("62% left"), "{narrow:?}");
        assert!(!narrow.contains('▰'), "gauge dropped when it does not fit: {narrow:?}");
    }

    /// The footer carries one quota window — the tightest — worded the same
    /// way the context gauge is, because a bare percent beside a model name
    /// gives the reader no way to tell headroom from consumption.
    #[test]
    fn hud_shows_the_tightest_quota_window_as_headroom() {
        let theme = Theme::default_dark();
        let mut state = sample_state();
        state.provider_quotas = vec![
            api::quota::ProviderQuotaView {
                provider: api::ProviderKind::Anthropic,
                window_label: "7d".to_string(),
                remaining_percent: Some(61),
                resets_at_unix: None,
                estimated: false,
            },
            api::quota::ProviderQuotaView {
                provider: api::ProviderKind::Anthropic,
                window_label: "5h".to_string(),
                remaining_percent: Some(12),
                resets_at_unix: Some(1_786_240_800),
                estimated: false,
            },
            api::quota::ProviderQuotaView {
                provider: api::ProviderKind::OpenAi,
                window_label: "7d".to_string(),
                remaining_percent: Some(85),
                resets_at_unix: None,
                estimated: false,
            },
        ];

        let text = line_text(&compose(&state, &theme, 200, false));
        assert!(
            text.contains("anthropic 12% left"),
            "each provider shows its nearest window: {text:?}"
        );
        assert!(
            text.contains("openai 85% left"),
            "a second signed-in provider is shown too: {text:?}"
        );
        assert!(
            !text.contains("61% left"),
            "the roomier window of a provider already shown stays on the rail: {text:?}"
        );
        // A percentage says how bad it is; the instant says how long it lasts.
        assert!(
            text.contains('↺') && text.contains(':'),
            "the footer says when the window comes back: {text:?}"
        );
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
            text.contains("Context ") && text.contains("62% left"),
            "the abbreviated HUD retains context pressure: {text:?}"
        );
        assert!(
            !text.contains("$0.08"),
            "other full-HUD details remain abbreviated: {text:?}"
        );
        let context = line
            .spans
            .iter()
            .find(|span| span.content == "62% left")
            .expect("context percentage span");
        // 38% of the compaction ceiling still leaves room, so it reads healthy.
        assert_eq!(context.style.fg, Some(theme.palette.success));
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

        // At rest the mark stays, in the brand accent: the leader is the
        // footer's anchor for the eye, and it must not vanish between turns.
        // Its width is unchanged, so nothing shifts when heat arrives.
        let cold = compose_impl(&state, &theme, 120, false, true, true, HeatState::Cold);
        assert_eq!(cold.spans[0].content.as_ref(), leader_text(&theme));
        assert_eq!(display_width(cold.spans[0].content.as_ref()), 3);
        assert_eq!(cold.spans[0].style.fg, Some(theme.palette.accent));
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
            assert!(text.starts_with(&leader_text(&theme)), "row leads with the mark: {text:?}");
            assert!(!text.contains('┗') && !text.contains('━'), "no anvil chrome: {text:?}");
            // Fields are divided by a thin rule, and by nothing heavier: the
            // bar carries a gauge, a badge, and a mode, and whitespace alone let
            // them run together into one grey sentence.
            assert!(!text.contains('·') && !text.contains('•'), "no heavy dividers: {text:?}");
            assert!(line_display_width(&line) <= 120);
        }
    }


}
