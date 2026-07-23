use std::cell::Cell;
use std::collections::HashSet;
use std::io;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::future::join_all;
use runtime::{
    ConfigLoader, ContentBlock, ConversationMessage, ConversationRuntime, MessageRole,
    PermissionMode, Session,
};
use sha2::{Digest, Sha256};
use serde_json::json;

use super::auto_fanout::FailureSignal;
use super::automation::{GoalAdvance, GoalController, LoopCommandResult, LoopController};
use super::{
    BuiltRuntime, build_runtime_plugin_state_with_loader, runtime_bridge, startup_snapshot,
};
use super::turn_harness::{ModelLedTurnSetup, TurnHarness};
use crate::cli_args::{AllowedToolSet, CliOutputFormat};
use crate::formatting::format_auto_compaction_notice;
use crate::render::TerminalRenderer;
use crate::create_managed_session_handle;
use crate::session_registry::{SessionScope, create_managed_session_handle_at};
use zo_cli::tui::modals::Effort;

/// Map a session's persistence scope to its prompt mode: an ephemeral session
/// is a headless one-shot where nobody can answer mid-task questions.
fn prompt_mode_for(scope: SessionScope) -> runtime::PromptMode {
    match scope {
        SessionScope::Project => runtime::PromptMode::Interactive,
        SessionScope::Ephemeral => runtime::PromptMode::Autonomous,
    }
}

use super::session_preferences::{
    SessionPreferences, effort_from_preferences, has_effort_preference, load_project_preferences,
    load_session_preferences, merge_preferences, save_project_preferences,
    save_session_preferences,
};

/// Token ceiling for the one-shot rubric grader reply — a small JSON object, so
/// generous but bounded. No extended thinking is requested.
const RUBRIC_GRADER_MAX_TOKENS: u32 = 2048;
/// Byte budget for the worker's final output handed to the grader.
const RUBRIC_GRADER_OUTPUT_LIMIT: usize = 4000;
/// Byte budget for the bounded working-tree diff handed to the grader.
const RUBRIC_GRADER_DIFF_LIMIT: usize = 6000;

/// System prompt for the independent rubric grader (see [`LiveCli::grade_active_rubric`]).
const RUBRIC_GRADER_SYSTEM_PROMPT: &str = "You are an INDEPENDENT, strict rubric grader for a long-horizon goal loop. \
You did NOT perform the work — judge it adversarially against the stated success criteria and do not give the benefit of the doubt. \
A criterion is `met` only when the provided evidence clearly demonstrates it. \
Reply with ONLY a single JSON object and no other text: \
{\"criteria\":[{\"name\":\"<criterion>\",\"met\":true|false,\"note\":\"<brief why>\"}],\"pass\":true|false}. \
`pass` is true only if every criterion is met.";

/// Token ceiling for one independent verify lens reply (a small JSON object).
const INDEPENDENT_VERIFY_MAX_TOKENS: u32 = 1024;
/// Byte budget for the bounded working-tree diff handed to each verify lens.
const INDEPENDENT_VERIFY_DIFF_LIMIT: usize = 6000;

/// The three independent verification lenses (mirrors the deep-lane VERIFY panel,
/// but each runs as its own fresh-context judgement). `(name, question)`.
const INDEPENDENT_VERIFY_LENSES: [(&str, &str); 3] = [
    (
        "spec",
        "Does the change correctly and completely accomplish the stated goal? Reject if it is incomplete, incorrect, or solves the wrong thing.",
    ),
    (
        "regression",
        "Could the change break existing behavior, callers, or tests? Reject if it plausibly regresses something that worked before.",
    ),
    (
        "security",
        "Does the change introduce a security or safety problem — injection, unsafe input handling, leaked secrets, or a destructive operation? Reject if so.",
    ),
];

/// System prompt for one independent verify lens (see [`LiveCli::independent_verify_under_ultracode`]).
const INDEPENDENT_VERIFY_SYSTEM_PROMPT: &str = "You are an INDEPENDENT, strict, adversarial verifier examining ONE lens of a code change you did NOT write. \
Judge only the lens you are given, against the diff. Default to rejecting when the lens's concern is plausibly violated — do not give the benefit of the doubt. \
Reply with ONLY a single JSON object and no other text: {\"accepted\": true|false, \"issue\": \"<brief reason>\"}.";

/// Build the per-lens user prompt for one independent verify judgement.
fn build_lens_verify_prompt(objective: &str, lens_question: &str, diff: &str) -> String {
    format!(
        "Goal:\n{objective}\n\nLens to judge:\n{lens_question}\n\n\
         The change under review (bounded git diff):\n{diff}\n\n\
         Judge ONLY this lens against the diff and reply with the JSON object only."
    )
}

/// Minimum byte length for a solo conclusion to be worth contesting under
/// principle ②. A short reply ("done", a rename ack, a one-line answer) has no
/// competing-hypothesis surface, so it warns nothing and costs nothing.
const COMPETING_HYPOTHESES_MIN_BYTES: usize = 280;

/// Byte bound on the last user ask fed to `tools::assess_turn_complexity` by
/// the post-turn panel gate. Generous enough that the classifier's long-brief
/// rule (≥800 chars) still sees a long ask as such.
const POST_TURN_ASK_CLASSIFY_LIMIT: usize = 4_096;
/// Byte budget for the conclusion text handed to each competing-hypotheses lens.
const COMPETING_HYPOTHESES_CONCLUSION_LIMIT: usize = 6000;

/// The two competing-hypotheses lenses for a solo reasoning/decision turn
/// (principle ②). `(name, question)`. Two framings of one axis: the alternatives
/// the conclusion failed to rule out, and the single strongest surviving
/// objection — kept to two so the burst stays cheap.
const COMPETING_HYPOTHESES_LENSES: [(&str, &str); 2] = [
    (
        "alternatives",
        "List the competing explanations or approaches the conclusion did NOT consider or rule out. Reject (accepted=false) ONLY if a specific, materially plausible alternative was left unaddressed.",
    ),
    (
        "objection",
        "State the single strongest concrete objection to the conclusion. Reject (accepted=false) ONLY if that objection is materially plausible and the conclusion does not answer it.",
    ),
];

/// System prompt for one competing-hypotheses lens (principle ② self-critique).
/// Yields the same `{"accepted": ...}` shape the rubric parser reads, and is
/// biased to abstain toward `accepted:true` so a well-reasoned solo answer is not
/// nagged by invented objections.
const COMPETING_HYPOTHESES_SYSTEM_PROMPT: &str = "You are an INDEPENDENT, adversarial reviewer of a conclusion you did NOT write. \
Judge ONLY the lens you are given against the conclusion. A conclusion is SOUND unless a SPECIFIC, materially plausible competing explanation or objection was left unaddressed — give a well-reasoned conclusion the benefit of the doubt and do NOT invent objections. \
Reply with ONLY a single JSON object and no other text: {\"accepted\": true|false, \"issue\": \"<the specific unaddressed alternative/objection, or empty>\"}.";

/// Build the per-lens user prompt for one competing-hypotheses judgement. Feeds
/// the model's OWN conclusion (no diff) so a pure-reasoning/decision turn — the
/// case the diff panel skips — is still contested.
fn build_competing_hypotheses_prompt(conclusion: &str, lens_question: &str) -> String {
    format!(
        "The conclusion under review (the assistant's own answer this turn):\n{conclusion}\n\n\
         Lens to judge:\n{lens_question}\n\n\
         Judge ONLY this lens against the conclusion and reply with the JSON object only."
    )
}

/// Whether a solo conclusion is substantive enough to be worth contesting. Pure:
/// a short reply has no competing-hypothesis surface, so it is skipped (no cost).
fn conclusion_is_substantive(conclusion: &str) -> bool {
    conclusion.trim().len() >= COMPETING_HYPOTHESES_MIN_BYTES
}

/// Build the per-turn user prompt that hands the grader its evidence.
fn build_rubric_grader_prompt(
    objective: &str,
    criteria: &[String],
    output: &str,
    diff: &str,
) -> String {
    let criteria_list = criteria
        .iter()
        .enumerate()
        .map(|(index, criterion)| format!("{}. {criterion}", index + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Goal objective:\n{objective}\n\nSuccess criteria (the rubric):\n{criteria_list}\n\n\
         The worker's final output this turn:\n{output}\n\n\
         Working-tree diff (bounded):\n{diff}\n\n\
         Grade each criterion against this evidence and reply with the JSON object only."
    )
}

/// A raw `git diff HEAD` of the working tree. Best-effort: an empty string when
/// this is not a git repo or git fails. Callers should truncate only the copy
/// sent to a grader; dedupe hashes use the full raw diff so two changes sharing
/// the same prefix never suppress each other.
fn working_tree_diff(cwd: &std::path::Path) -> String {
    std::process::Command::new("git")
        .args(["diff", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default()
}

/// A bounded `git diff HEAD` of the working tree for the grader. Best-effort:
/// an empty string when this is not a git repo or git fails.
fn bounded_working_tree_diff(cwd: &std::path::Path, limit: usize) -> String {
    truncate_bytes(&working_tree_diff(cwd), limit)
}

/// Join the text blocks of the grader's [`api::MessageResponse`].
fn grader_response_text(response: &api::MessageResponse) -> String {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            api::OutputContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn post_turn_verify_diff_hash(diff: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(diff.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn reserve_post_turn_verify(seen: &Mutex<HashSet<String>>, diff_hash: &str) -> bool {
    let mut seen = seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    seen.insert(diff_hash.to_string())
}

fn record_decisive_post_turn_verify(
    seen: &Mutex<HashSet<String>>,
    diff_hash: &str,
    verdict: Option<bool>,
) -> Option<bool> {
    if verdict.is_none() {
        seen.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(diff_hash);
    }
    verdict
}

async fn send_lens_request(
    client: api::ProviderClient,
    model: String,
    prompt: String,
    system_prompt: &'static str,
) -> decision_core::LensVerdict {
    let request = api::MessageRequest {
        model: api::wire_model_id(&model),
        max_tokens: INDEPENDENT_VERIFY_MAX_TOKENS,
        messages: vec![api::InputMessage::user_text(prompt)],
        system: Some(vec![api::SystemBlock::text(system_prompt)]),
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    match client.send_message(&request).await {
        Ok(response) => match decision_core::parse_rubric_grade(&grader_response_text(&response)) {
            Some(true) => decision_core::LensVerdict::Accept,
            Some(false) => decision_core::LensVerdict::Reject,
            None => decision_core::LensVerdict::Abstain,
        },
        Err(_) => decision_core::LensVerdict::Abstain,
    }
}

/// Combine the deep-lane semantic verdict with the rubric grader's verdict
/// conservatively: any reject wins, else any accept, else no signal. Reuses the
/// goal gate's `AnyReject` fold so the two semantic sources cannot drift apart.
fn fold_semantic_with_rubric(semantic: Option<bool>, rubric: Option<bool>) -> Option<bool> {
    let lens = |value: Option<bool>| match value {
        Some(true) => decision_core::LensVerdict::Accept,
        Some(false) => decision_core::LensVerdict::Reject,
        None => decision_core::LensVerdict::Abstain,
    };
    decision_core::fold_lens_verdicts(
        &[lens(semantic), lens(rubric)],
        decision_core::ConsensusPolicy::AnyReject,
    )
}

/// Truncate `text` to at most `limit` bytes on a char boundary, marking elision.
fn truncate_bytes(text: &str, limit: usize) -> String {
    core_types::text::truncate_on_char_boundary(text, limit, "…[truncated]")
}

#[derive(Debug, Clone)]
pub(crate) struct SessionHandle {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ManagedSessionSummary {
    pub(crate) id: String,
    pub(crate) name: Option<String>,
    pub(crate) path: PathBuf,
    pub(crate) modified_epoch_millis: u128,
    pub(crate) message_count: usize,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) branch_name: Option<String>,
    pub(crate) first_user_text: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum UserPlanState {
    #[default]
    Inactive,
    Active,
}

impl UserPlanState {
    fn from_selected(selected: bool) -> Self {
        if selected { Self::Active } else { Self::Inactive }
    }

    fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }
}

pub(crate) struct LiveCli {
    /// Workspace directory captured when this session was created or
    /// rehydrated. Server sessions keep using this even if the process cwd
    /// changes while other sessions are alive.
    pub(crate) cwd: PathBuf,
    pub(crate) model: String,
    /// `true` when the session model is an explicit user pin (`--model` flag
    /// or `/model` pick). Mirrored onto the tool-dispatch context so spawn
    /// smart routing inherits the pinned model instead of re-routing.
    pub(crate) model_user_pinned: bool,
    pub(crate) allowed_tools: Option<AllowedToolSet>,
    pub(crate) permission_mode: PermissionMode,
    /// Authoritative, runtime-facing user-selected planning state. Plan
    /// enforces [`PermissionMode::ReadOnly`], so `permission_mode` alone cannot
    /// distinguish it from plain read-only. Only user-driven seams mutate it.
    user_plan_state: UserPlanState,
    pub(crate) system_prompt: Vec<String>,
    pub(crate) runtime: BuiltRuntime,
    pub(crate) tasks: Arc<runtime::task_registry::TaskRegistry>,
    pub(crate) session: SessionHandle,
    /// Extended thinking budget. `None` = thinking disabled.
    pub(crate) thinking_budget: Option<u32>,
    /// Current effort level. Drives the thinking budget and, for
    /// [`Effort::Smart`], the dynamic per-turn effort band.
    /// A custom numeric `/effort <n>` that matches no preset is tracked
    /// as `None` (the budget still applies via `thinking_budget`).
    pub(crate) effort: Option<Effort>,
    /// Git snapshot stack for /undo /redo (None if not in a git repo).
    pub(crate) snapshot_stack: Option<runtime::git_snapshot::SnapshotStack>,
    /// Where sessions spawned by this CLI are persisted. Interactive REPLs
    /// are project-scoped; one-shot `-p` / headless runs are ephemeral so
    /// they never pollute the target repo. Inherited by `/new` and `/fork`.
    pub(crate) session_scope: SessionScope,
    /// Explicit MCP config file supplied through `--mcp-config`.
    ///
    /// Kept on the session harness so any runtime rebuild (`/clear`,
    /// `/permission`, headless per-turn runtime rebuilds, etc.) preserves the
    /// same MCP server surface instead of silently falling back to default
    /// settings after startup.
    pub(crate) mcp_config: Option<PathBuf>,
    /// Free-text goal for the current session, set via `/goal <text>` and
    /// shown by `/goal` with no argument. In-memory only — it scopes the
    /// REPL's intent without touching the persisted session file.
    pub(crate) session_goal: Option<String>,
    /// Session-local `/goal` controller. Owns validator config, turn counts,
    /// and history; `session_goal` above mirrors the active text into the
    /// existing system-reminder path.
    pub(crate) goal_controller: GoalController,
    /// Session-local `/loop` controller for fixed-count, interval, and polling
    /// watch loops. TUI drains this into the same serialized prompt queue used
    /// for user-entered mid-turn messages.
    pub(crate) loop_controller: LoopController,
    /// One-shot ownership latch: `true` only while the turn the host just
    /// dispatched is a *goal-owned* turn (a goal action prompt or a queued
    /// repair). `advance_goal_after_turn` consumes it so an unrelated user,
    /// `/loop`, or workflow turn that happens to earn a verifier accept cannot
    /// complete the active goal. (The full multi-owner model is deferred; a bool
    /// suffices while goals are single-active and turns run serially.)
    pub(crate) goal_turn_pending: bool,
    /// Diff hashes already completed by the post-turn verifier in this session.
    /// The verifier reads `git diff HEAD`, so a dirty worktree that stays dirty
    /// across several follow-up turns would otherwise produce the same warning
    /// over and over. Hashes are recorded only after a decisive verifier result
    /// (`Some(true/false)`); API errors or abstentions remain retryable.
    post_turn_verified_diffs: Arc<Mutex<HashSet<String>>>,
    /// Whether THIS session has ever written non-empty automation state. The
    /// persist file is project-scoped (keyed by cwd, shared across concurrent
    /// sessions in the same project), so a session with no automation of its own
    /// must never let its empty save *delete* a sibling session's active goal.
    /// Only a session that itself wrote automation may remove the file on clear.
    /// `Cell` so the `&self` persist path can record it without a `&mut` ripple.
    automation_persisted: Cell<bool>,
    /// Model-switch handoff memory. When the user plans with one model and
    /// switches to another for implementation, this keeps the next model
    /// anchored to the same session decisions without exposing hidden thought.
    pub(crate) model_handoff_memory: Option<String>,
    /// Epoch-second lower bound for agent manifests shown in this session's HUD.
    /// `.zo/agents` is workspace-global; without this, a newly opened chat can
    /// inherit still-fresh `running` rows from a previous session and show ghost
    /// agents even though no current turn spawned them.
    pub(crate) agent_manifest_started_after: u64,
    /// Optional cap on the agentic tool-use loop per turn, from `--max-turns`
    /// on the headless `-p` path. `None` inherits the runtime's
    /// `DEFAULT_MAX_ITERATIONS` backstop; setting it caps worst-case cost lower
    /// on a one-shot run.
    pub(crate) max_turns: Option<usize>,
    /// Optional cap on model-requested tool calls per turn, from
    /// `--max-tool-calls` on the headless `-p` path. This bounds parallel
    /// tool bursts that fit inside a single agentic iteration.
    pub(crate) max_tool_calls: Option<usize>,
    /// The failure signal to escalate the *next* turn's route on (WI-B), or
    /// `None` when the last turn succeeded or escalation already stopped. Fed to
    /// [`RouteHint::escalate`](super::auto_fanout::RouteHint::escalate).
    pub(crate) route_escalation: Option<FailureSignal>,
    /// The failure signal the previous turn ended on, for the 2-consecutive
    /// guard in [`decide_escalation`](super::auto_fanout::decide_escalation).
    pub(crate) last_failure_signal: Option<FailureSignal>,
    /// Consecutive turns that ended by exhausting a turn budget (grind streak).
    /// Session-scoped and in-memory (a `/restart` starts fresh); arms
    /// [`grind_escalation`](super::grind_escalation) at turn entry and resets
    /// on any turn that ends clean.
    pub(crate) grind_streak: u32,
    /// Automatic continuations already spent on the current chain (see
    /// `grind_escalation::should_auto_continue`). Reset when a turn ends clean
    /// or when the user types their own message (a fresh chain).
    pub(crate) auto_continue_chain: u32,
    /// Armed when the previous turn ended with a verbalized low-confidence
    /// readout (see [`confidence_cascade`](super::confidence_cascade)), plus
    /// the model's stated reason for the escalated turn's directive. One-shot:
    /// consumed (taken) at the next turn entry. Session-scoped and in-memory.
    #[allow(clippy::option_option)]
    pub(crate) cascade_armed: Option<Option<String>>,
    /// True after a turn that consumed an armed cascade, so a still-low
    /// readout at that escalated turn's end does NOT immediately re-arm (one
    /// escalation per low streak; persistent uncertainty surfaces to the user
    /// through the directive's report contract instead of burning escalated
    /// turns in a loop). Cleared by any non-escalated turn.
    pub(crate) cascade_ran_last_turn: bool,
    /// Custom status line runner (settings `statusLine` command, CC parity).
    /// `Arc` so the TUI's poller closure shares the same debounced cache.
    pub(crate) status_line: std::sync::Arc<super::status_line::StatusLineRunner>,
    /// One-turn override for the offered tool set, set by a custom prompt/slash
    /// command that declares `allowed-tools`. When `Some`, the next turn's wire
    /// request is filtered to exactly this set (preferred over the session-global
    /// [`allowed_tools`](Self::allowed_tools)); it is cleared back to `None` at
    /// turn completion so the restriction never leaks into the following turn.
    pub(crate) turn_allowed_tools: Option<AllowedToolSet>,
    /// Cross-model verifier target for the always-on verification legs (the
    /// deep gate's VERIFY sub-turns and the post-turn lens panels), as
    /// `(model, provider client)`. Re-derived on every turn entry from the
    /// Smart Verifier-role route (see `smart_settings::route_deep_verify_candidates`)
    /// and cleared when routing yields nothing, so it can never outlive a model
    /// switch or a `/smart off`. Caching the constructed [`ProviderClient`]
    /// avoids re-resolving credentials (keychain/OAuth) on every turn.
    pub(crate) deep_verify_provider: Option<(String, api::AuthRoute, api::ProviderClient)>,
    /// Cached cross-provider client for the quota-fallback turn loop (installed
    /// via `runtime::set_quota_fallback_client`), as
    /// `(model, auth route, provider client)`.
    /// Re-derived on every turn entry from the different-provider route (see
    /// `smart_settings::route_quota_fallback_model`) and cleared when routing
    /// yields nothing, so it can never outlive a model switch or a
    /// `/smart quota-fallback off`. Caching the constructed [`ProviderClient`]
    /// avoids re-resolving credentials (keychain/OAuth) on every turn — the same
    /// pattern as [`Self::deep_verify_provider`].
    pub(crate) quota_fallback_provider: Option<(String, api::AuthRoute, api::ProviderClient)>,
    /// Cached implementer client for the Architect execution contract
    /// (installed via `runtime::set_exec_contract`), as
    /// `(model, auth route, provider client)`. Re-derived on every turn entry
    /// from the Smart Coding-role route (see
    /// `smart_settings::route_exec_impl_model`) and cleared when
    /// routing yields nothing, so it can never outlive a model switch or a
    /// `/smart policy classic`. Caching the constructed [`ProviderClient`]
    /// avoids re-resolving credentials on every turn — the same pattern as
    /// [`Self::deep_verify_provider`]. Populated only when `smart.execSwap`
    /// arms for the current turn's classified difficulty.
    pub(crate) exec_impl_provider: Option<(String, api::AuthRoute, api::ProviderClient)>,
    /// Pending `/restart` re-exec. Set by the `/restart` handler once its
    /// pre-flight gate passes and the session is persisted; consumed by
    /// `run_repl` *after* the TUI loop exits and the terminal is restored, at
    /// which point it replaces the process image with a fresh build resuming
    /// this session. `None` on every ordinary exit.
    pub(crate) pending_restart: Option<super::restart::RestartPlan>,
}

const SESSION_GOAL_REMINDER_PREFIX: &str = "[zo:session-goal]";
const MODEL_HANDOFF_REMINDER_PREFIX: &str = "[zo:model-handoff]";
const PLAN_MODE_REMINDER_PREFIX: &str = "[zo:plan-mode]";
const DEFAULT_EFFORT: Effort = Effort::High;

/// The per-turn contract injected while the user has explicitly selected Plan
/// (Shift+Tab plan stop or `/plan on`). The runtime is read-only in Plan, but a
/// bare read-only session cannot tell the model *why* — so without this the
/// model reaches for the write-gated `EnterPlanMode` tool every turn and hits a
/// deterministic permission denial. Stating that Plan is already active (and
/// that leaving it is a user-only action) removes that dead-end round-trip
/// without weakening read-only enforcement.
fn plan_mode_system_reminder() -> String {
    format!(
        "{PLAN_MODE_REMINDER_PREFIX} Plan mode is already active (the user selected it). \
         The session is read-only by the user's choice: explore with read-only tools, then \
         write and submit your plan with ExitPlanModeV2. \
         Do NOT call EnterPlanMode — plan mode is already on, so that call only fails. \
         Do NOT try to switch permission mode or edit settings to gain write access; \
         only the user leaves plan mode (Shift+Tab or `/plan off`), and submitting a plan \
         never restores write access on its own."
    )
}

fn session_goal_system_reminder(goal: &str) -> Option<String> {
    let goal = goal.trim();
    if goal.is_empty() {
        return None;
    }
    Some(format!(
        "{SESSION_GOAL_REMINDER_PREFIX} Current session goal: {goal}\n\
         Prioritize this goal when planning, delegating, verifying, and deciding whether the work is complete. \
         When goal mode is active, plan before acting: state the concrete plan, then execute, validate, and repair. \
         The user's latest message still takes precedence."
    ))
}

pub(super) fn model_handoff_system_reminder(
    previous_model: &str,
    next_model: &str,
    messages: &[ConversationMessage],
    session_goal: Option<&str>,
) -> Option<String> {
    if previous_model.trim().is_empty() || next_model.trim().is_empty() {
        return None;
    }
    let mut lines = vec![format!(
        "{MODEL_HANDOFF_REMINDER_PREFIX} Model handoff: this session switched from {previous_model} to {next_model}. Prior context remains active; this is not a new task."
    )];
    if let Some(goal) = session_goal.map(str::trim).filter(|goal| !goal.is_empty()) {
        lines.push(format!("Active session goal: {goal}"));
    }
    let recent = recent_handoff_context(messages, 4);
    if !recent.is_empty() {
        lines.push("Recent visible context to preserve:".to_string());
        lines.extend(recent.into_iter().map(|line| format!("- {line}")));
    }
    lines.push(
        "Prior user intent, plans, constraints, todos, and verification state remain relevant unless the latest user message changes them."
            .to_string(),
    );
    Some(lines.join("\n"))
}

fn recent_handoff_context(messages: &[ConversationMessage], limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for message in messages.iter().rev() {
        if out.len() >= limit {
            break;
        }
        let role = match message.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            _ => continue,
        };
        let text = message
            .blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        let text = compact_handoff_text(&text, 220);
        if !text.is_empty() {
            out.push(format!("{role}: {text}"));
        }
    }
    out.reverse();
    out
}

fn compact_handoff_text(text: &str, limit: usize) -> String {
    let mut collapsed = String::with_capacity(text.len().min(limit));
    let mut last_was_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
            }
            last_was_space = true;
        } else {
            collapsed.push(ch);
            last_was_space = false;
        }
    }
    let collapsed = collapsed.trim();
    let mut chars = collapsed.chars();
    let truncated = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}\u{2026}")
    } else {
        truncated
    }
}

/// Remove the default cwd todo store (`.zo-todos.json`) so a freshly
/// started session begins with an empty checklist instead of inheriting
/// stale todos from a previous session that shared this working directory.
///
/// Leaves an explicit `ZO_TODO_STORE` override in place — that path is
/// owned by tests or by callers that deliberately scope the store, and is
/// not the ghost-todo source this guards against.
fn clear_stale_default_todo_store_at(cwd: &Path) {
    // Capture the user-override state now, before we ever set the env var
    // ourselves for per-session scoping.
    if user_supplied_todo_store() {
        return;
    }
    let _ = std::fs::remove_file(cwd.join(".zo-todos.json"));
}

/// Whether the current `ZO_TODO_STORE` is a *user-provided* override (set at
/// launch or by a deliberate caller) rather than one of our own per-session
/// scoping writes. We must never clobber the former, but must always be free to
/// update the latter on a session switch.
///
/// Detected structurally instead of via a one-shot snapshot: the env counts as
/// a user override only when it is present AND its value is not one this process
/// previously wrote through [`scope_todo_store_to_session`]. This is independent
/// of call order — a lazily-captured `OnceLock` snapshot was order-dependent and
/// broke under a shared multi-test process (whichever test first set the var
/// poisoned every later session's scoping). An absent env is never an override.
fn user_supplied_todo_store() -> bool {
    match std::env::var_os("ZO_TODO_STORE") {
        None => false,
        Some(current) => !value_was_scoped_by_us(&current),
    }
}

/// Process-global record of every `ZO_TODO_STORE` value our own session
/// scoping has written, so [`user_supplied_todo_store`] can tell our writes
/// apart from a user/launch-provided override without a fragile snapshot.
fn scoped_store_values() -> &'static std::sync::Mutex<std::collections::HashSet<std::ffi::OsString>>
{
    use std::sync::OnceLock;
    static SCOPED: OnceLock<std::sync::Mutex<std::collections::HashSet<std::ffi::OsString>>> =
        OnceLock::new();
    SCOPED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

fn record_scoped_store_value(value: &std::ffi::OsStr) {
    if let Ok(mut set) = scoped_store_values().lock() {
        set.insert(value.to_os_string());
    }
}

fn value_was_scoped_by_us(value: &std::ffi::OsStr) -> bool {
    scoped_store_values()
        .lock()
        .map(|set| set.contains(value))
        .unwrap_or(false)
}

/// Scope this process's todo store to one session so two `zo` instances in
/// the same working directory (e.g. one on GPT, one on Claude) keep separate
/// checklists instead of clobbering a single shared `cwd/.zo-todos.json`.
///
/// The harness is identical for every model — this changes *where* the store
/// lives, not how it behaves — so the per-session file sits beside the session
/// transcript (`<session-path-stem>.todos.json`). All readers/writers already
/// honor `ZO_TODO_STORE`, so setting it here transparently isolates the
/// store for the runtime, the sidebar, and the `TodoWrite` tool alike.
///
/// A user-provided `ZO_TODO_STORE` (tests, explicit scoping) wins and is
/// left untouched — but our *own* prior session scoping is updated, so a
/// `/resume`/`/session switch` correctly follows the new session's store.
///
/// `fresh` clears any stale file at the target (new session); a resume leaves
/// the existing per-session todos in place so the restored checklist survives.
pub(super) fn scope_todo_store_to_session(session_path: &std::path::Path, fresh: bool) {
    if user_supplied_todo_store() {
        return;
    }
    let store_path = session_path.with_extension("todos.json");
    if fresh {
        let _ = std::fs::remove_file(&store_path);
    }
    // Record the value as our own scoping write BEFORE setting it, so a later
    // `user_supplied_todo_store()` recognizes it as ours (not a user override)
    // regardless of test/call order.
    record_scoped_store_value(store_path.as_os_str());
    std::env::set_var("ZO_TODO_STORE", &store_path);
}

fn epoch_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn build_runtime_with_optional_mcp_config(
    cwd: &std::path::Path,
    mcp_config: Option<&PathBuf>,
    session: Session,
    session_id: &str,
    model: String,
    system_prompt: Vec<String>,
    enable_tools: bool,
    emit_output: bool,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    thinking: Option<api::ThinkingConfig>,
    named_effort: Option<api::EffortLevel>,
    effort_band_ceiling: Option<api::EffortLevel>,
    tasks: runtime::task_registry::TaskRegistry,
    startup_auth_policy: crate::runtime_support::StartupAuthPolicy,
) -> Result<BuiltRuntime, Box<dyn std::error::Error>> {
    let Some(mcp_config) = mcp_config else {
        return crate::runtime_support::build_runtime_with_thinking_for_auth_policy(
            cwd,
            session,
            session_id,
            model,
            system_prompt,
            enable_tools,
            emit_output,
            allowed_tools,
            permission_mode,
            thinking,
            named_effort,
            effort_band_ceiling,
            Some(tasks),
            startup_auth_policy,
        );
    };

    let loader = ConfigLoader::default_for(cwd).with_mcp_config(mcp_config);
    let runtime_config = loader.load()?;
    let runtime_plugin_state = build_runtime_plugin_state_with_loader(
        cwd,
        &loader,
        &runtime_config,
        Some(tasks),
    )?;
    crate::runtime_support::build_runtime_with_plugin_state_auth_policy(
        session,
        session_id,
        model,
        system_prompt,
        enable_tools,
        emit_output,
        allowed_tools,
        permission_mode,
        runtime_plugin_state,
        thinking,
        named_effort,
        effort_band_ceiling,
        startup_auth_policy,
    )
}

/// Pure core of [`LiveCli::apply_system_prompt_overrides`]. `replace` discards
/// the base entirely; `append` is added as a trailing prompt segment after any
/// replacement. Kept free-standing so the precedence is unit-testable without
/// constructing a full runtime.
fn apply_prompt_overrides(
    base: Vec<String>,
    replace: Option<String>,
    append: Option<String>,
) -> Vec<String> {
    let mut prompt = match replace {
        Some(replacement) => vec![replacement],
        None => base,
    };
    if let Some(addition) = append {
        prompt.push(addition);
    }
    prompt
}

fn startup_auth_policy_for_scope(_scope: SessionScope) -> crate::runtime_support::StartupAuthPolicy {
    crate::runtime_support::StartupAuthPolicy::Require
}

impl LiveCli {
    /// Construct an interactive (project-scoped) CLI: sessions persist into
    /// the working tree's `.zo/sessions/`. This is the historical default
    /// used by the REPL/TUI path.
    #[allow(dead_code)]
    pub(crate) fn new(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_scoped(
            model,
            enable_tools,
            allowed_tools,
            permission_mode,
            SessionScope::Project,
        )
    }

    /// Construct a project-scoped CLI but keep startup auth mandatory.
    ///
    /// Headless slash-command entrypoints need project session access but must
    /// not hide auth failures behind the interactive TUI fallback.
    pub(crate) fn new_requiring_startup_auth(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_scoped_with_mcp_config_and_session_id(
            model,
            enable_tools,
            allowed_tools,
            permission_mode,
            SessionScope::Project,
            None,
            None,
            crate::runtime_support::StartupAuthPolicy::Require,
        )
    }

    /// Start a new visible agent/workflow scope for this live session.
    ///
    /// The on-disk agent/workflow stores are shared by every chat in the
    /// workspace, so session-changing commands must advance this lower bound.
    /// Plain transcript reseeds (rewind/undo) should not call it.
    ///
    /// Also re-stamps the runtime tool-context session id to the *current*
    /// session. The fast session-swap paths (`resume_session_fast`,
    /// `clear_session_report`/`/new`) only call `replace_session`, which swaps
    /// the transcript but never touches the shared tool context — whose session
    /// id is otherwise written just once at build time (`runtime_support`). Left
    /// stale, every `SpawnMultiAgent` member manifest is stamped with the
    /// pre-swap session id, so the TUI's strict session filter
    /// (`manifest_belongs_to_session`, `allow_unstamped = false`) drops all of
    /// them and the inline agent tree stays empty (`spawning…` forever). Every
    /// caller sets `self.session` to the new id before calling this, so the
    /// re-stamp tracks the live session; on the full-rebuild paths the id is
    /// already correct and this is a no-op.
    pub(crate) fn refresh_agent_manifest_scope(&mut self) -> u64 {
        self.agent_manifest_started_after = epoch_seconds_now();
        let session_id = self.session.id.clone();
        if let Some(runtime) = self.runtime.runtime.as_mut() {
            let context = runtime.tool_executor_mut().tool_registry_mut().context();
            context.set_session_id(&session_id);
            // Session swaps rebuild the shared ToolContext, so re-assert the
            // interactive-host background default alongside the session id —
            // every caller of this method is an interactive command handler
            // whose REPL re-injects detached agent completions.
            context.set_background_agent_default(true);
        }
        self.refresh_workspace_checkpoint_scope();
        self.agent_manifest_started_after
    }

    fn refresh_workspace_checkpoint_scope(&mut self) {
        let durable_dir = self
            .runtime
            .feature_config
            .checkpoint_durable()
            .then(|| self.session.path.with_extension("checkpoints"));
        if let Some(runtime) = self.runtime.runtime.as_mut() {
            let context = runtime.tool_executor_mut().tool_registry_mut().context();
            if let Err(error) = context.reset_workspace_checkpoint_session(durable_dir) {
                eprintln!("warning: failed to load workspace checkpoints: {error}");
            }
        }
    }

    /// Construct a CLI whose session persistence is governed by `scope`.
    ///
    /// Non-interactive entry points (`zo -p …`, headless slash commands)
    /// pass [`SessionScope::Ephemeral`] so a benchmark/CI run leaves the
    /// target repository clean — no `.zo/sessions/` is written into the
    /// working tree. Interactive callers use [`SessionScope::Project`].
    pub(crate) fn new_scoped(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        scope: SessionScope,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_scoped_with_mcp_config(
            model,
            enable_tools,
            allowed_tools,
            permission_mode,
            scope,
            None,
        )
    }

    pub(crate) fn new_scoped_with_mcp_config(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        scope: SessionScope,
        mcp_config: Option<PathBuf>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_scoped_with_mcp_config_and_session_id(
            model,
            enable_tools,
            allowed_tools,
            permission_mode,
            scope,
            mcp_config,
            None,
            startup_auth_policy_for_scope(scope),
        )
    }

    /// [`Self::new_scoped_with_mcp_config`] plus an explicit session id
    /// (`--session-id`, CC parity) so scripted/CI runs get deterministic
    /// session files. Refuses an id whose session file already exists.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_scoped_with_mcp_config_and_session_id(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        scope: SessionScope,
        mcp_config: Option<PathBuf>,
        explicit_session_id: Option<String>,
        startup_auth_policy: crate::runtime_support::StartupAuthPolicy,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cwd = crate::current_cli_cwd()?;
        Self::new_scoped_with_mcp_config_and_session_id_at(
            model,
            enable_tools,
            allowed_tools,
            permission_mode,
            scope,
            mcp_config,
            explicit_session_id,
            startup_auth_policy,
            cwd,
        )
    }

    pub(crate) fn new_scoped_at(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        scope: SessionScope,
        cwd: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_scoped_with_mcp_config_and_session_id_at(
            model,
            enable_tools,
            allowed_tools,
            permission_mode,
            scope,
            None,
            None,
            crate::runtime_support::StartupAuthPolicy::Require,
            cwd,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_scoped_with_mcp_config_and_session_id_at(
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        scope: SessionScope,
        mcp_config: Option<PathBuf>,
        explicit_session_id: Option<String>,
        startup_auth_policy: crate::runtime_support::StartupAuthPolicy,
        cwd: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // A fresh session owns no todos. The sidebar re-reads the todo store
        // (`.zo-todos.json`) on every render tick, so a file left behind by
        // an earlier session in this cwd would otherwise resurrect as ghost
        // items the moment the TUI starts. Clear it before the runtime — and
        // the first render — can read it. Honors `ZO_TODO_STORE` overrides
        // (tests, explicit scoping) by leaving non-default stores untouched.
        clear_stale_default_todo_store_at(&cwd);
        let system_prompt =
            crate::conversation_support::build_system_prompt_for_mode(&cwd, prompt_mode_for(scope))?;
        let mut session_state = Session::new();
        if let Some(id) = explicit_session_id {
            let id = id.trim();
            if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                return Err(format!(
                    "--session-id must be non-empty and use only [A-Za-z0-9-]: {id:?}"
                )
                .into());
            }
            id.clone_into(&mut session_state.session_id);
        }
        let session = create_managed_session_handle_at(&session_state.session_id, scope, &cwd)?;
        // Isolate this session's todo store so a second `zo` in the same cwd
        // (e.g. GPT and Claude side by side) does not share — and overwrite —
        // one `cwd/.zo-todos.json`. Same harness for every model; only the
        // store location is per-session. `fresh`: a brand-new session starts
        // with an empty checklist.
        scope_todo_store_to_session(&session.path, true);
        let preferences = load_project_preferences(&cwd)?;
        let model = if model == crate::DEFAULT_MODEL {
            preferences.model.clone().unwrap_or(model)
        } else {
            model
        };
        let (preferred_effort, preferred_budget) = effort_from_preferences(&preferences);
        let has_effort_preference = has_effort_preference(&preferences);
        let effort = if has_effort_preference {
            preferred_effort
        } else {
            Some(DEFAULT_EFFORT)
        };
        let thinking_budget = if has_effort_preference {
            preferred_budget
        } else {
            Some(DEFAULT_EFFORT.budget())
        };
        let tasks = Arc::new(runtime::task_registry::TaskRegistry::new());
        let mut runtime = build_runtime_with_optional_mcp_config(
            &cwd,
            mcp_config.as_ref(),
            session_state.with_persistence_path(session.path.clone()),
            &session.id,
            model.clone(),
            system_prompt.clone(),
            enable_tools,
            true,
            allowed_tools.clone(),
            permission_mode,
            thinking_budget.map(api::ThinkingConfig::enabled),
            effort.and_then(Effort::level),
            effort.and_then(Effort::band_ceiling),
            tasks.as_ref().clone(),
            startup_auth_policy,
        )?;
        // An ephemeral session is a headless one-shot: nobody is present to
        // answer a mid-run question, so the turn-end gate lints those too.
        runtime.set_autonomous_surface(matches!(scope, SessionScope::Ephemeral));
        let status_line = std::sync::Arc::new(super::status_line::StatusLineRunner::new(
            super::status_line::status_line_command_from_config(&cwd),
        ));
        let mut cli = Self {
            snapshot_stack: runtime::git_snapshot::SnapshotStack::try_new_at(&cwd),
            cwd,
            model,
            model_user_pinned: false,
            allowed_tools,
            permission_mode,
            user_plan_state: UserPlanState::default(),
            system_prompt,
            runtime,
            tasks,
            session,
            thinking_budget,
            effort,
            session_scope: scope,
            mcp_config,
            session_goal: None,
            goal_controller: GoalController::default(),
            loop_controller: LoopController::default(),
            goal_turn_pending: false,
            post_turn_verified_diffs: Arc::new(Mutex::new(HashSet::new())),
            automation_persisted: Cell::new(false),
            model_handoff_memory: None,
            agent_manifest_started_after: epoch_seconds_now(),
            max_turns: None,
            max_tool_calls: None,
            route_escalation: None,
            last_failure_signal: None,
            grind_streak: 0,
            auto_continue_chain: 0,
            cascade_armed: None,
            cascade_ran_last_turn: false,
            status_line,
            turn_allowed_tools: None,
            deep_verify_provider: None,
            quota_fallback_provider: None,
            exec_impl_provider: None,
            pending_restart: None,
        };
        cli.load_automation_state();
        cli.refresh_workspace_checkpoint_scope();
        cli.apply_session_system_reminders();
        cli.persist_session()?;
        Ok(cli)
    }

    /// Like [`Self::new_scoped`] but rehydrates an already-persisted
    /// [`Session`] instead of starting fresh — `zo serve` uses it to restore
    /// `.zo/sessions/` transcripts into its live pool on restart so old
    /// session ids keep working across a server bounce.
    ///
    /// A 1:1 copy of `new_scoped`, except it (1) keeps the loaded conversation
    /// (no `Session::new()`) and (2) does **not** clear the cwd's default todo
    /// store: that store belongs to the interactive user, and a server reviving
    /// many sessions at boot must never wipe it.
    pub(crate) fn new_scoped_with_session(
        session_state: Session,
        model: String,
        enable_tools: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        scope: SessionScope,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cwd = crate::current_cli_cwd()?;
        let system_prompt =
            crate::conversation_support::build_system_prompt_for_mode(&cwd, prompt_mode_for(scope))?;
        // Reuse the loaded session's id so the on-disk path is unchanged — this
        // updates the existing file's timestamp, never creates a new one.
        let session = create_managed_session_handle(&session_state.session_id, scope)?;
        let preferences = merge_preferences(
            load_session_preferences(&session.path),
            load_project_preferences(&cwd)?,
        );
        let model = if model == crate::DEFAULT_MODEL {
            preferences.model.clone().unwrap_or(model)
        } else {
            model
        };
        let (preferred_effort, preferred_budget) = effort_from_preferences(&preferences);
        let has_effort_preference = has_effort_preference(&preferences);
        let effort = if has_effort_preference {
            preferred_effort
        } else {
            Some(DEFAULT_EFFORT)
        };
        let thinking_budget = if has_effort_preference {
            preferred_budget
        } else {
            Some(DEFAULT_EFFORT.budget())
        };
        // /goal은 세션 헤더에 영속된다(resume 복원) — 빌더로 move되기 전에 캡처.
        let restored_session_goal = session_state.session_goal.clone();
        let tasks = Arc::new(runtime::task_registry::TaskRegistry::new());
        let mut runtime = build_runtime_with_optional_mcp_config(
            &cwd,
            None,
            session_state.with_persistence_path(session.path.clone()),
            &session.id,
            model.clone(),
            system_prompt.clone(),
            enable_tools,
            true,
            allowed_tools.clone(),
            permission_mode,
            thinking_budget.map(api::ThinkingConfig::enabled),
            effort.and_then(Effort::level),
            effort.and_then(Effort::band_ceiling),
            tasks.as_ref().clone(),
            startup_auth_policy_for_scope(scope),
        )?;
        runtime.set_autonomous_surface(matches!(scope, SessionScope::Ephemeral));
        let status_line = std::sync::Arc::new(super::status_line::StatusLineRunner::new(
            super::status_line::status_line_command_from_config(&cwd),
        ));
        let mut cli = Self {
            cwd,
            model,
            model_user_pinned: false,
            allowed_tools,
            permission_mode,
            user_plan_state: UserPlanState::default(),
            system_prompt,
            runtime,
            tasks,
            session,
            thinking_budget,
            effort,
            snapshot_stack: runtime::git_snapshot::SnapshotStack::try_new(),
            session_scope: scope,
            mcp_config: None,
            session_goal: restored_session_goal,
            goal_controller: GoalController::default(),
            loop_controller: LoopController::default(),
            goal_turn_pending: false,
            post_turn_verified_diffs: Arc::new(Mutex::new(HashSet::new())),
            automation_persisted: Cell::new(false),
            model_handoff_memory: preferences.model_handoff_memory,
            agent_manifest_started_after: epoch_seconds_now(),
            max_turns: None,
            max_tool_calls: None,
            route_escalation: None,
            last_failure_signal: None,
            grind_streak: 0,
            auto_continue_chain: 0,
            cascade_armed: None,
            cascade_ran_last_turn: false,
            status_line,
            turn_allowed_tools: None,
            deep_verify_provider: None,
            quota_fallback_provider: None,
            exec_impl_provider: None,
            pending_restart: None,
        };
        cli.load_automation_state();
        cli.refresh_workspace_checkpoint_scope();
        cli.apply_session_system_reminders();
        cli.persist_session()?;
        Ok(cli)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_runtime(
        &self,
        session: Session,
        session_id: &str,
        model: String,
        system_prompt: Vec<String>,
        enable_tools: bool,
        emit_output: bool,
        allowed_tools: Option<AllowedToolSet>,
        permission_mode: PermissionMode,
        thinking: Option<api::ThinkingConfig>,
    ) -> Result<BuiltRuntime, Box<dyn std::error::Error>> {
        let mut runtime = build_runtime_with_optional_mcp_config(
            &self.cwd,
            self.mcp_config.as_ref(),
            session,
            session_id,
            model,
            system_prompt,
            enable_tools,
            emit_output,
            allowed_tools,
            permission_mode,
            thinking,
            self.effort.and_then(Effort::level),
            self.effort.and_then(Effort::band_ceiling),
            self.tasks.as_ref().clone(),
            crate::runtime_support::StartupAuthPolicy::Require,
        )?;
        self.apply_spawn_model_context(&mut runtime);
        Ok(runtime)
    }

    fn apply_spawn_model_context(&self, runtime: &mut BuiltRuntime) {
        let Some(inner) = runtime.try_runtime_mut() else {
            return;
        };
        let context = inner.tool_executor_mut().tool_registry_mut().context();
        context.set_active_model(&self.model);
        context.set_active_model_pinned(self.model_user_pinned);
    }

    /// Record whether the session model is an explicit user pin and mirror it
    /// onto the live tool-dispatch context (shared `Arc` cell, so concurrent
    /// spawn dispatch observes it immediately).
    pub(crate) fn set_model_user_pinned(&mut self, pinned: bool) {
        self.model_user_pinned = pinned;
        if let Some(rt) = self.runtime.try_runtime_mut() {
            rt.tool_executor_mut()
                .tool_registry_mut()
                .context()
                .set_active_model_pinned(pinned);
        }
    }

    /// Record a turn's terminal error so the *next* turn's route can escalate
    /// (WI-B). Recognized, escalation-worthy failures bump the route one step;
    /// a 2nd consecutive identical failure stops escalating (honest failure, no
    /// loop). Unrecognized failures (auth/network) leave the route untouched.
    pub(crate) fn record_turn_failure(&mut self, error: &str) {
        if let Some(signal) = super::auto_fanout::classify_turn_failure(error) {
            self.route_escalation =
                super::auto_fanout::decide_escalation(self.last_failure_signal, signal);
            self.last_failure_signal = Some(signal);
        }
    }

    /// Record a turn that ended by exhausting a turn budget. The graceful stop
    /// returns `Ok`, so without this the WI-B ladder never saw it — worse, the
    /// clean-turn reset *cleared* prior escalation state, which is how an
    /// hours-long exhaust→"계속"→exhaust grind stayed invisible to routing.
    /// Feeds both the one-step route escalation and the grind streak that arms
    /// [`grind_escalation`](super::grind_escalation) on the next turn.
    pub(crate) fn record_turn_budget_exhausted(&mut self, kind: runtime::BudgetExhausted) {
        self.grind_streak = self.grind_streak.saturating_add(1);
        let signal = super::auto_fanout::failure_signal_for_budget(kind);
        self.route_escalation =
            super::auto_fanout::decide_escalation(self.last_failure_signal, signal);
        self.last_failure_signal = Some(signal);
    }

    /// Clear the escalation state after a turn that did not fail, so a future
    /// failure starts a fresh ladder rather than inheriting stale state.
    pub(crate) fn clear_turn_failure(&mut self) {
        self.route_escalation = None;
        self.last_failure_signal = None;
        self.grind_streak = 0;
    }

    /// Build an `api::ThinkingConfig` from the current thinking budget, if set.
    pub(crate) fn thinking_config(&self) -> Option<api::ThinkingConfig> {
        self.thinking_budget.map(api::ThinkingConfig::enabled)
    }

    /// Install a fresh [`runtime_bridge::LiveAsyncApiClient`] on the live
    /// runtime so an inline async operation (e.g. `/compact`) drives its
    /// round-trip through the async client and await-suspends instead of
    /// blocking the drive-loop task. Mirrors the per-turn install in
    /// [`super::turn_controller::drive_turn`]; the runtime built by
    /// `build_runtime`/`replace_runtime` carries no async client, so this must
    /// run before such an operation. No-op when the inner runtime has been taken.
    pub(crate) fn ensure_async_api_client(&mut self) {
        let live_client = {
            let api_client = self.runtime.api_client();
            std::sync::Arc::new(runtime_bridge::LiveAsyncApiClient::new(
                api_client.client(),
                api_client.model().to_string(),
                api_client.auth_route(),
                api_client.enable_tools(),
                self.allowed_tools.clone(),
                api_client.tool_registry(),
                self.thinking_config(),
                self.effort.and_then(Effort::level),
                self.effort.and_then(Effort::band_ceiling),
            ))
        };
        if let Some(runtime) = self.runtime.try_runtime_mut() {
            runtime.set_async_api_client(live_client);
        }
    }

    /// Select a named effort preset, updating both the thinking budget
    /// and the tracked level (so [`Effort::Smart`] turns resolve the
    /// dynamic effort band).
    ///
    /// The change always applies in-memory (it takes effect this session even
    /// when it cannot be persisted). Returns a ready-to-surface warning line
    /// when persisting the preference failed — worded like the model-switch
    /// warning — so a caller can tell the user the choice was not saved; `None`
    /// on success. Startup/resume callers that have no user to warn ignore it.
    pub(crate) fn set_effort(&mut self, effort: Effort) -> Option<String> {
        self.effort = Some(effort);
        self.thinking_budget = (effort.budget() > 0).then_some(effort.budget());
        self.effort_preference_persist_warning()
    }

    /// Bound the per-turn agentic loop (headless `--max-turns`). `None` inherits
    /// the runtime's `DEFAULT_MAX_ITERATIONS` backstop; applied to each turn's
    /// runtime in `prepare_turn_runtime`.
    pub(crate) fn set_max_turns(&mut self, max_turns: Option<usize>) {
        self.max_turns = max_turns;
    }

    /// Bound model-requested tool calls per turn (headless
    /// `--max-tool-calls`). `None` leaves tool calls unbounded; applied to each
    /// turn's runtime in `prepare_turn_runtime`.
    pub(crate) fn set_max_tool_calls(&mut self, max_tool_calls: Option<usize>) {
        self.max_tool_calls = max_tool_calls;
    }

    /// Apply a custom numeric `/effort <n>` budget. Snaps to a preset
    /// level when the budget matches one exactly; otherwise tracks no
    /// named level (`Smart`'s reminder only fires for the preset).
    ///
    /// Same contract as [`Self::set_effort`]: the budget always applies
    /// in-memory, and a persistence failure is returned as a surface-ready
    /// warning line rather than swallowed.
    pub(crate) fn set_effort_budget(&mut self, budget: u32) -> Option<String> {
        self.effort = Effort::from_budget((budget > 0).then_some(budget));
        self.thinking_budget = (budget > 0).then_some(budget);
        self.effort_preference_persist_warning()
    }

    /// Persist the session preferences and, on failure, format the single
    /// effort-persistence warning line shared by [`Self::set_effort`] and
    /// [`Self::set_effort_budget`]. Worded to mirror the model-switch warning
    /// (`apply_model_change`) so `/effort` and `/model` report failures the
    /// same way. One definition keeps the two setters' warning policy identical.
    fn effort_preference_persist_warning(&self) -> Option<String> {
        self.persist_session_preferences()
            .err()
            .map(|error| format!("Warning          effort preference was not saved: {error}"))
    }

    /// System prompt to send for the upcoming turn on the *headless*
    /// paths, which rebuild the runtime from scratch each turn: the stored
    /// base prompt plus the session-goal and model-handoff reminders.
    ///
    /// Deliberately effort-agnostic. Orchestration posture is taught once by
    /// the base prompt's delegation rubric (and the spawn tools' own
    /// descriptions), and the model applies it per ask — no mode appends a
    /// standing fan-out reminder on top.
    pub(crate) fn effective_system_prompt(&self) -> Vec<String> {
        let mut prompt = self.system_prompt.clone();
        if let Some(goal_reminder) = self
            .session_goal
            .as_deref()
            .and_then(session_goal_system_reminder)
        {
            prompt.push(goal_reminder);
        }
        if let Some(model_handoff) = self.model_handoff_memory.as_deref() {
            prompt.push(model_handoff.to_string());
        }
        if self.plan_selected() {
            prompt.push(plan_mode_system_reminder());
        }
        prompt
    }

    /// Apply `--system-prompt`/`--append-system-prompt` to the stored base
    /// prompt that every headless turn sends (via `effective_system_prompt`).
    /// `replace` swaps the whole base prompt; `append` adds a trailing
    /// segment. The parser makes the two mutually exclusive, but both are
    /// handled here so the order is well-defined if that ever changes.
    pub(crate) fn apply_system_prompt_overrides(
        &mut self,
        replace: Option<String>,
        append: Option<String>,
    ) {
        self.system_prompt =
            apply_prompt_overrides(std::mem::take(&mut self.system_prompt), replace, append);
    }

    /// Re-read project context files and runtime config, then swap the live
    /// runtime to use the refreshed base system prompt.
    pub(crate) fn reload_context(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        // Settings may have changed — refresh the custom status line command
        // alongside the prompt so `/reload` (and `/output-style`) pick both up.
        self.status_line
            .set_command(super::status_line::status_line_command_from_config(
                &self.cwd,
            ));
        let refreshed_prompt = crate::conversation_support::build_system_prompt_for_mode(
            &self.cwd,
            prompt_mode_for(self.session_scope),
        )?;
        let section_count = refreshed_prompt.len();
        let runtime = self.build_runtime(
            self.runtime.session().clone(),
            &self.session.id,
            self.model.clone(),
            refreshed_prompt.clone(),
            true,
            true,
            self.allowed_tools.clone(),
            self.permission_mode,
            self.thinking_config(),
        )?;
        self.replace_runtime(runtime)?;
        self.system_prompt = refreshed_prompt;
        self.apply_session_system_reminders();
        self.persist_session()?;
        Ok(format!(
            "Context reloaded\n  System prompt sections {section_count}\n  Session          {}",
            self.session.id
        ))
    }

    /// Sync the session-goal and model-handoff reminders onto the long-lived
    /// TUI runtime. Idempotent and surgical: it never disturbs the base prompt
    /// or a post-compaction reminder, and never accumulates duplicates across
    /// turns.
    pub(crate) fn apply_session_system_reminders(&mut self) {
        let goal_reminder = self
            .session_goal
            .as_deref()
            .and_then(session_goal_system_reminder);
        let model_handoff = self.model_handoff_memory.as_deref();
        let plan_selected = self.plan_selected();
        let plan_reminder = plan_selected.then(plan_mode_system_reminder);
        if let Some(runtime) = self.runtime.try_runtime_mut() {
            // Mirror the authoritative user-selected Plan flag onto the shared
            // tool-context cell so the live registry (and its clones, including
            // the request builder) drive the plan-mode tool surface from mode,
            // not prompt inference. Applied here so a runtime rebuild — which
            // routes through `replace_runtime` → this method — re-establishes it
            // on the fresh context rather than silently reverting to write-mode.
            runtime
                .tool_executor_mut()
                .tool_registry_mut()
                .context()
                .set_plan_selected(plan_selected);
            // Mirror the goal into the runtime session: persists it for
            // resume and feeds the TurnEnd (Stop) hook's `sessionGoal`.
            runtime.set_session_goal(self.session_goal.clone());
            runtime.replace_transient_system_reminder_by_prefix(
                SESSION_GOAL_REMINDER_PREFIX,
                goal_reminder.as_deref(),
            );
            runtime.replace_transient_system_reminder_by_prefix(
                MODEL_HANDOFF_REMINDER_PREFIX,
                model_handoff,
            );
            runtime.replace_transient_system_reminder_by_prefix(
                PLAN_MODE_REMINDER_PREFIX,
                plan_reminder.as_deref(),
            );
        }
    }

    pub(crate) fn plan_selected(&self) -> bool {
        self.user_plan_state.is_active()
    }

    /// Set the authoritative user-selected Plan state and refresh the per-turn
    /// contract so the model is told Plan is already active. Called only from
    /// user-driven seams (Shift+Tab plan stop, `/plan on|off`); the model never
    /// toggles this and can never use it to restore write access.
    pub(crate) fn set_plan_selected(&mut self, selected: bool) {
        let state = UserPlanState::from_selected(selected);
        if self.user_plan_state == state {
            return;
        }
        self.user_plan_state = state;
        self.apply_session_system_reminders();
    }

    #[cfg(test)]
    pub(crate) fn startup_banner(&self) -> String {
        self.startup_banner_with_timing(None)
    }

    // Only the `cfg(test)` banner wrapper above and bin tests consume the
    // plain-text banner today — the TUI path renders `startup_screen`.
    #[cfg(test)]
    pub(crate) fn startup_banner_with_timing(
        &self,
        startup_elapsed: Option<std::time::Duration>,
    ) -> String {
        startup_snapshot::build_startup_banner(
            &self.model,
            self.permission_mode,
            &self.session.id,
            startup_elapsed,
        )
    }

    pub(crate) fn startup_screen(
        &self,
        startup_elapsed: Option<std::time::Duration>,
    ) -> zo_cli::tui::StartupScreen {
        startup_snapshot::build_startup_screen(
            &self.model,
            self.permission_mode.as_str(),
            &self.session.id,
            &self.session.path,
            startup_elapsed,
        )
    }

    // Bin tests assert the boxed-input label; the TUI composes its own.
    #[cfg(test)]
    pub(crate) fn input_box_label(&self) -> String {
        startup_snapshot::input_box_label(&self.model, &self.session.id)
    }

    fn apply_goal_controller_to_built_runtime(&self, runtime: &mut BuiltRuntime) {
        if let Some(inner) = runtime.try_runtime_mut() {
            inner.set_deep_gate(self.goal_controller.deep_gate_config());
        }
    }

    pub(crate) fn start_goal_controller(
        &mut self,
        goal: String,
        options: commands::GoalOptions,
    ) -> (String, Option<String>) {
        // Goal-contract gate (decision_core::goal_contract): an ambiguous
        // success metric ("100프로 커버리지") with no objective check gets ONE
        // clarifying question BEFORE any turn is spent — the observed runaway
        // burned twelve hours on the wrong reading of exactly such a goal. A
        // goal with any objective `--check` (or an unambiguous text) is never
        // held back; the user is present at `/goal` time, so the text
        // round-trip is immediate. `ZO_GOAL_CONTRACT` < 2 disables the gate.
        if super::automation::goal_contract_level() >= 2
            && !super::automation::has_objective_checks(&options.checks)
        {
            if let decision_core::GoalAmbiguity::Ambiguous(cues) = decision_core::screen_goal(&goal)
            {
                return (
                    super::automation::build_goal_clarify_report(&goal, &cues),
                    None,
                );
            }
        }
        let report = self.goal_controller.start(goal, options);
        self.session_goal = self
            .goal_controller
            .active_goal_text()
            .map(std::string::ToString::to_string);
        self.apply_session_system_reminders();
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_deep_gate(self.goal_controller.deep_gate_config());
        }
        // Ownership is latched per-turn when the queued goal prompt pops (TUI) or
        // at the headless goal loop top — not here at dispatch — so a user message
        // typed ahead of this prompt cannot consume the goal's verifier verdict.
        let prompt = self.goal_controller.active_prompt();
        (report, prompt)
    }

    pub(crate) fn edit_goal_controller(&mut self, goal: String) -> String {
        let report = self.goal_controller.edit(goal);
        self.session_goal = self
            .goal_controller
            .active_goal_text()
            .map(std::string::ToString::to_string);
        self.apply_session_system_reminders();
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_deep_gate(self.goal_controller.deep_gate_config());
        }
        report
    }

    pub(crate) fn clear_goal_controller(&mut self) -> String {
        let report = self.goal_controller.clear();
        self.session_goal = None;
        self.apply_session_system_reminders();
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_deep_gate(None);
        }
        report
    }

    pub(crate) fn pause_goal_controller(&mut self) -> String {
        let report = self.goal_controller.pause();
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_deep_gate(None);
        }
        report
    }

    pub(crate) fn resume_goal_controller(&mut self) -> (String, Option<String>) {
        let Some((report, prompt)) = self.goal_controller.resume() else {
            return (
                "Goal resume\n  Status           no paused goal to resume".to_string(),
                None,
            );
        };
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_deep_gate(self.goal_controller.deep_gate_config());
        }
        // Ownership is latched per-turn at pop (TUI) / loop top (headless), not
        // here — see `start_goal_controller`.
        (report, Some(prompt))
    }

    pub(crate) fn verify_goal_controller(&mut self) -> String {
        // Manual `/goal verify` has no turn in hand, so there is no semantic
        // verdict to fold — pass `None` (defers to deterministic validators).
        let report = self.goal_controller.verify(&self.cwd, None);
        report.render("Goal validation")
    }

    /// Ownership latch, shared by the async and blocking advance paths. Returns
    /// `Some(turn_output_tokens)` when the completed turn is a goal-owned turn
    /// that should advance the goal, or `None` (→ `Idle`).
    ///
    /// `turn_output_tokens` is already THIS turn's own output delta (from
    /// `TurnSummary.turn_output_tokens`, measured within the turn's runtime
    /// instance), so it is charged directly — no host-side cumulative baseline,
    /// which used to drift across the per-turn runtime rebuild + compaction.
    fn goal_advance_precheck(&mut self, turn_output_tokens: u32) -> Option<u32> {
        // Ownership gate: only advance the goal when the turn that just completed
        // was a goal-owned turn. The latch is set per-turn at the point the turn
        // begins — at queue *pop* in the TUI (from the message's `goal_owned` tag)
        // and at the top of the headless goal loop — and consumed here, so it is
        // never stale: an unrelated user, `/loop`, or workflow turn always sees it
        // `false` and cannot consume the goal's verifier verdict or burn a turn.
        std::mem::take(&mut self.goal_turn_pending).then_some(turn_output_tokens)
    }

    /// Refresh the deep gate from the goal's current state after an advance. The
    /// ownership latch is NOT re-armed here: a queued repair turn carries the
    /// `goal_owned` tag and re-latches when it pops (TUI) / at the loop top
    /// (headless), so re-arming at dispatch would only risk a stale latch.
    fn goal_advance_finish(&mut self) {
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_deep_gate(self.goal_controller.deep_gate_config());
        }
    }

    /// Interactive advance: run the goal's validators (which may block on
    /// `cargo`/`git` for up to two minutes) on a worker thread via
    /// `spawn_blocking`, so the TUI event loop keeps drawing the spinner and
    /// handling input instead of freezing. Mirrors `deep_gate::command_is_green`.
    pub(crate) async fn advance_goal_after_turn(
        &mut self,
        semantic: Option<bool>,
        verifier_issues: Vec<String>,
        turn_output_tokens: u32,
    ) -> GoalAdvance {
        let Some(turn_output_tokens) = self.goal_advance_precheck(turn_output_tokens) else {
            return GoalAdvance::Idle;
        };
        let Some(validators) = self.goal_controller.active_goal_validators() else {
            return GoalAdvance::Idle;
        };
        // Grade the goal's `ModelRubric` success criteria (if any) with an
        // independent fresh-context model evaluator and fold its verdict into the
        // semantic signal. No rubric criterion ⇒ `semantic` is returned unchanged
        // at zero cost; a rubric is the only way a non-coding goal (no objective
        // cargo/git/grep check) is ever confirmed `Satisfied` rather than running
        // to its turn cap as "unverified".
        let semantic = self.grade_active_rubric(&validators, semantic).await;
        // Under UltraCode, fold in an INDEPENDENT adversarial verification of this
        // turn's change (three fresh-context lenses), strengthening the deep-lane
        // verdict that is a single self-reporting forward pass. Off-UltraCode this
        // is a no-op (the deep-lane `semantic` is returned unchanged).
        let semantic = self.independent_verify_under_ultracode(semantic).await;
        let cwd = self.cwd.clone();
        // A join error means the worker panicked or the runtime is shutting down
        // — never observed in practice (the validators are panic-safe). Skip this
        // advance rather than fabricate a verdict.
        let Ok(mut report) = tokio::task::spawn_blocking(move || {
            super::automation::run_validators(&cwd, &validators, semantic)
        })
        .await
        else {
            return GoalAdvance::Idle;
        };
        // Attach the verifier's concrete objections (empty unless it rejected) so
        // the repair prompt can name the exact defects to fix.
        report.semantic_issues = verifier_issues;
        let advance = self.goal_controller.record_turn_with_report(
            &self.cwd,
            &self.session.id,
            &report,
            turn_output_tokens,
        );
        self.goal_advance_finish();
        advance
    }

    /// Grade the active goal's `ModelRubric` success criteria with an
    /// independent, fresh-context model evaluator and fold the verdict into
    /// `semantic`.
    ///
    /// Returns `semantic` unchanged when the goal has no rubric criterion (the
    /// common case — zero added cost). Otherwise asks the model — in a fresh
    /// one-shot request with no tools and no session history, so it is not biased
    /// by the implementer's own turn — to judge this turn's work (the goal
    /// objective, the rubric criteria, the model's final output, and a bounded
    /// working-tree diff) and folds its per-criterion verdict via
    /// [`decision_core::parse_rubric_grade`]. The verdict is combined with the
    /// incoming deep-lane `semantic` conservatively (any reject wins), so a
    /// rubric veto blocks a stop and a rubric accept can satisfy a goal that has
    /// no objective validators.
    ///
    /// The request is a plain async API call (no spawned agent, no tools), so it
    /// never freezes the TUI. Any failure (no reply, parse miss, API error)
    /// yields no rubric signal and leaves `semantic` as the deep-lane verdict —
    /// fail-open, never a fabricated accept.
    async fn grade_active_rubric(
        &self,
        validators: &[super::automation::GoalValidator],
        semantic: Option<bool>,
    ) -> Option<bool> {
        let criteria: Vec<String> = validators
            .iter()
            .filter_map(|validator| match validator {
                super::automation::GoalValidator::ModelRubric { label } => Some(label.clone()),
                _ => None,
            })
            .collect();
        if criteria.is_empty() {
            return semantic;
        }
        let Some(objective) = self.goal_controller.active_goal_text().map(str::to_string) else {
            return semantic;
        };
        let output = self.last_assistant_text_bounded(RUBRIC_GRADER_OUTPUT_LIMIT);
        let cwd = self.cwd.clone();
        let diff =
            tokio::task::spawn_blocking(move || bounded_working_tree_diff(&cwd, RUBRIC_GRADER_DIFF_LIMIT))
                .await
                .unwrap_or_default();
        let request = api::MessageRequest {
            model: self.model.clone(),
            max_tokens: RUBRIC_GRADER_MAX_TOKENS,
            messages: vec![api::InputMessage::user_text(build_rubric_grader_prompt(
                &objective, &criteria, &output, &diff,
            ))],
            system: Some(vec![api::SystemBlock::text(RUBRIC_GRADER_SYSTEM_PROMPT)]),
            tools: None,
            tool_choice: None,
            stream: false,
            thinking: None,
            output_config: None,
            effort: None,
            effort_band_ceiling: None,
        };
        // Owned client, so the `&self` borrow ends before the await.
        let client = self.runtime.api_client().client();
        let rubric = match client.send_message(&request).await {
            Ok(response) => decision_core::parse_rubric_grade(&grader_response_text(&response)),
            Err(_) => None,
        };
        fold_semantic_with_rubric(semantic, rubric)
    }

    /// Under `UltraCode`, fold an INDEPENDENT adversarial verification of this goal
    /// turn's change into `semantic`.
    ///
    /// Where the deep-lane VERIFY is a single sub-turn that self-reports three
    /// lens booleans in one forward pass (so one blind spot biases all three),
    /// this issues three genuinely independent fresh-context judgements — spec,
    /// regression, security — each blind to the others and to the implementer's
    /// own reasoning, and folds them under `AnyReject`: a single credible
    /// objection blocks the goal from stopping. The panel verdict is then combined
    /// with the incoming deep-lane verdict conservatively (any reject wins), so
    /// this only ever *strengthens* the gate, never relaxes it.
    ///
    /// Gated to `UltraCode` and to turns that actually changed the working tree, so
    /// ordinary turns are unchanged and pay nothing. Each lens is a plain async API
    /// call with no tools (no spawned agent, no permission surface), so it never
    /// freezes the TUI; any failure abstains (fail-open), never a fabricated
    /// accept. Lives at the CLI layer because the verifier infrastructure cannot
    /// be reached from the `runtime`-crate deep gate without a circular dependency.
    async fn independent_verify_under_ultracode(&self, semantic: Option<bool>) -> Option<bool> {
        // Goal-completion FOLD: Ultracode only. This path can BLOCK a goal from
        // stopping, so it must never widen to default-High (that would gate every
        // High goal turn on the 3-lens panel). The non-blocking *warning* path
        // (`independent_verify_warning`) is what extends the panel to High/Max.
        if self.effort != Some(Effort::Smart) {
            return semantic;
        }
        self.run_independent_verify_panel(semantic).await
    }

    /// The independent three-lens panel itself (diff → spec / regression /
    /// security fresh-context lenses → `AnyReject` fold), with NO effort gate —
    /// the CALLER decides when to run it. `semantic` is the verdict the panel can
    /// only ever strengthen (a lens reject wins); an empty diff or any API / parse
    /// failure leaves it unchanged (fail-open).
    async fn run_independent_verify_panel(&self, semantic: Option<bool>) -> Option<bool> {
        let cwd = self.cwd.clone();
        let diff = tokio::task::spawn_blocking(move || {
            bounded_working_tree_diff(&cwd, INDEPENDENT_VERIFY_DIFF_LIMIT)
        })
        .await
        .unwrap_or_default();
        if diff.trim().is_empty() {
            // Nothing changed this turn — there is nothing to independently verify.
            return semantic;
        }
        let objective = self
            .goal_controller
            .active_goal_text()
            .map(str::to_string)
            .unwrap_or_default();
        let (client, model) = self.verify_lens_target();
        let panel = Self::run_independent_lenses(client, model, objective, diff).await;
        fold_semantic_with_rubric(semantic, panel)
    }

    /// The verification lens panels' `(client, model)`: the cross-model
    /// verifier when one is wired for this session (the same Smart
    /// Verifier-role target the deep gate's VERIFY legs run on), else the
    /// native main-model client. Reads the per-turn cache populated by the
    /// turn controller, so it is exactly as fresh as this turn's routing —
    /// which makes the "independent" spec/regression/security panel actually
    /// independent of the model that produced the change.
    fn verify_lens_target(&self) -> (api::ProviderClient, String) {
        match self.deep_verify_provider.as_ref() {
            Some((model, _, client)) => (client.clone(), model.clone()),
            None => (self.runtime.api_client().client(), self.model.clone()),
        }
    }

    /// The independent-verify lens loop as a `self`-free associated fn (mirrors
    /// `run_competing_lenses` for `post_turn_verify_future`): inputs are
    /// snapshotted so the lens round-trips run on a spawned task, never borrowing
    /// `self` or blocking `app.run`. Behaviour is byte-identical to the old loop.
    async fn run_independent_lenses(
        client: api::ProviderClient,
        model: String,
        objective: String,
        diff: String,
    ) -> Option<bool> {
        let lens_futures = INDEPENDENT_VERIFY_LENSES.iter().map(|(_lens, question)| {
            send_lens_request(
                client.clone(),
                model.clone(),
                build_lens_verify_prompt(&objective, question, &diff),
                INDEPENDENT_VERIFY_SYSTEM_PROMPT,
            )
        });
        let verdicts = join_all(lens_futures).await;
        decision_core::fold_lens_verdicts(&verdicts, decision_core::ConsensusPolicy::AnyReject)
    }

    /// Snapshot the inputs the post-turn verify panel needs and return a
    /// **spawnable** future that runs the LLM lens round-trips OFF the input-pump
    /// thread. `run_session_loop` spawns this after a non-goal turn instead of
    /// `.await`-ing the panels inline, so the composer and spinner stay live
    /// during the 2–3 lens calls — that is the fix for the "answer ends → input
    /// echoes one char then bursts" freeze (the panels only ever push a
    /// non-blocking warning, so a slightly-late arrival is fine). Mirrors the old
    /// inline sequence exactly: an edit turn (non-empty diff) gets the High+
    /// independent spec/regression/security panel, a no-diff Ultracode turn gets
    /// the competing-hypotheses panel; `None` means there is nothing to spawn.
    /// Record the CURRENT worktree diff as verified in the post-turn registry.
    /// Called when the reactive gate (or a goal verify leg) semantically
    /// accepted this turn's change: the panel is skipped for the turn itself,
    /// but without this record a later Smart turn over the still-dirty
    /// worktree would re-judge the exact same diff — "verified once, never
    /// re-verified" must survive turn boundaries.
    pub(crate) fn mark_worktree_diff_verified(&self) {
        let diff = working_tree_diff(&self.cwd);
        if diff.trim().is_empty() {
            return;
        }
        let hash = post_turn_verify_diff_hash(&diff);
        let _ = reserve_post_turn_verify(&self.post_turn_verified_diffs, &hash);
    }

    pub(crate) fn post_turn_verify_future(
        &self,
    ) -> Option<impl std::future::Future<Output = Option<String>> + Send + 'static> {
        let effort = self.effort?;
        // Smart (ultracode) only. This panel used to run on every High+ edit
        // turn ON TOP of the reactive gate's own verification, so one small
        // change was judged by several mid-tier agents in sequence — the
        // "과잉 검증" the verification redesign removes. Ordinary turns keep
        // exactly one verification: the reactive gate's smart verify leg.
        if effort != Effort::Smart {
            return None;
        }
        let raw_diff = working_tree_diff(&self.cwd);
        let (client, model) = self.verify_lens_target();
        let objective = self
            .goal_controller
            .active_goal_text()
            .map(str::to_string)
            .unwrap_or_default();
        let conclusion = self.last_assistant_text_bounded(COMPETING_HYPOTHESES_CONCLUSION_LIMIT);
        // The no-diff competing-hypotheses fallback is a Smart-only
        // orchestration behavior (mirrors `independent_verify_under_ultracode`
        // above) — the static `Ultra` pin has no orchestration hint, so it
        // stays out of this fallback even though it shares the High+ gate.
        let is_ultracode = effort == Effort::Smart;
        // Proportionality: the competing-hypotheses panel is for substantive
        // solo CONCLUSIONS (a diagnosis, a decision). A turn whose ask itself
        // read trivial/small — a lookup, a quick question — skips it even when
        // the answer runs long, riding the same complexity signal as the Smart
        // dynamic effort band. An empty/unreadable ask classifies Unknown and
        // keeps the panel (fail-up).
        let ask_reads_simple = matches!(
            tools::assess_turn_complexity(
                &self.last_user_text_bounded(POST_TURN_ASK_CLASSIFY_LIMIT)
            ),
            runtime::RouteTaskComplexity::Trivial | runtime::RouteTaskComplexity::Small
        );
        let verified_diffs = self.post_turn_verified_diffs.clone();
        // Capture the exact diff before returning the spawnable future. Creating
        // a future does not poll it, so reading `git diff` inside the async body
        // let the next user turn race in and contaminate this turn's verifier.
        Some(async move {
            if !raw_diff.trim().is_empty() {
                let diff_hash = post_turn_verify_diff_hash(&raw_diff);
                if !reserve_post_turn_verify(&verified_diffs, &diff_hash) {
                    return None;
                }
                let diff = truncate_bytes(&raw_diff, INDEPENDENT_VERIFY_DIFF_LIMIT);
                // Edit turn: the High+ independent spec/regression/security panel.
                let verdict = Self::run_independent_lenses(client, model, objective, diff).await;
                let verdict = record_decisive_post_turn_verify(
                    &verified_diffs,
                    &diff_hash,
                    verdict,
                );
                match verdict {
                    Some(false) => Some(
                        "Independent verification flagged this change — a spec / regression / \
                         security lens rejected it. Review before relying on it."
                            .to_string(),
                    ),
                    _ => None,
                }
            } else if is_ultracode && !ask_reads_simple && conclusion_is_substantive(&conclusion) {
                // No-diff Ultracode turn: the competing-hypotheses panel.
                Self::run_competing_lenses(client, model, conclusion).await
            } else {
                None
            }
        })
    }

    /// The competing-hypotheses lens loop as a `self`-free associated fn, so the
    /// post-turn panel can run it on a spawned task off the input-pump thread (see
    /// `post_turn_verify_future`): the caller snapshots `client`/`model`/
    /// `conclusion` and moves them in, so the LLM round-trips never borrow `self`
    /// and never block `app.run`. Behaviour is byte-identical to the old inline
    /// loop — the only change is that the inputs arrive pre-snapshotted.
    async fn run_competing_lenses(
        client: api::ProviderClient,
        model: String,
        conclusion: String,
    ) -> Option<String> {
        let lens_futures = COMPETING_HYPOTHESES_LENSES.iter().map(|(_lens, question)| {
            send_lens_request(
                client.clone(),
                model.clone(),
                build_competing_hypotheses_prompt(&conclusion, question),
                COMPETING_HYPOTHESES_SYSTEM_PROMPT,
            )
        });
        let verdicts = join_all(lens_futures).await;
        match decision_core::fold_lens_verdicts(&verdicts, decision_core::ConsensusPolicy::AnyReject)
        {
            Some(false) => Some(
                "Competing-hypotheses check flagged this conclusion — an independent lens found a \
                 specific alternative or objection it did not rule out. Re-examine before relying \
                 on it."
                    .to_string(),
            ),
            _ => None,
        }
    }

    /// The last assistant message's text content, truncated to `limit` bytes —
    /// the model's own account of what it did this turn, for the rubric grader.
    fn last_assistant_text_bounded(&self, limit: usize) -> String {
        let text = self
            .runtime
            .session()
            .messages
            .iter()
            .rev()
            .find(|message| message.role == MessageRole::Assistant)
            .map(|message| {
                message
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        truncate_bytes(&text, limit)
    }

    /// The most recent USER text — the turn's ask — from the session, bounded.
    /// Skips user-role messages whose blocks carry no text (tool-result
    /// carriers ride the user role in the wire format but are not the ask).
    fn last_user_text_bounded(&self, limit: usize) -> String {
        let text = self
            .runtime
            .session()
            .messages
            .iter()
            .rev()
            .filter(|message| message.role == MessageRole::User)
            .map(|message| {
                message
                    .blocks
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .find(|text| !text.trim().is_empty())
            .unwrap_or_default();
        truncate_bytes(&text, limit)
    }

    /// Blocking advance for the headless path (and tests): no TUI to freeze, so
    /// the validators run inline. Same decision as the async path. The goal's
    /// `ModelRubric` success criteria are graded here too (via a private
    /// current-thread runtime, since `main()` is sync), so a rubric-only headless
    /// goal can actually pass instead of being structurally `Unverifiable`.
    pub(crate) fn advance_goal_after_turn_blocking(
        &mut self,
        semantic: Option<bool>,
        turn_output_tokens: u32,
    ) -> GoalAdvance {
        let Some(turn_output_tokens) = self.goal_advance_precheck(turn_output_tokens) else {
            return GoalAdvance::Idle;
        };
        // Fold the rubric grade into the semantic verdict on the headless path too
        // (parity with the interactive `advance_goal_after_turn`). Without this a
        // rubric-only goal has `deterministic = None` and `semantic = None`, so it
        // is forever `Unverifiable` in scripted/CI runs — exactly where it matters.
        let semantic = self.grade_active_rubric_blocking(semantic);
        let advance = self.goal_controller.record_turn_and_advance(
            &self.cwd,
            &self.session.id,
            semantic,
            turn_output_tokens,
        );
        self.goal_advance_finish();
        advance
    }

    /// Blocking wrapper around [`Self::grade_active_rubric`] for the headless path.
    /// Builds a private current-thread runtime ONLY when the active goal actually
    /// carries rubric criteria (otherwise returns `semantic` untouched at zero
    /// cost), mirroring the `run_prompt_ndjson` `block_on` pattern — the headless
    /// path is a sync context (`main()` has no `#[tokio::main]`). Fail-open: any
    /// runtime-build error leaves the verdict unchanged rather than blocking the
    /// goal turn.
    fn grade_active_rubric_blocking(&self, semantic: Option<bool>) -> Option<bool> {
        let Some(validators) = self.goal_controller.active_goal_validators() else {
            return semantic;
        };
        let has_rubric = validators
            .iter()
            .any(|validator| matches!(validator, super::automation::GoalValidator::ModelRubric { .. }));
        if !has_rubric {
            return semantic;
        }
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return semantic;
        };
        rt.block_on(self.grade_active_rubric(&validators, semantic))
    }

    pub(crate) fn handle_loop_controller_command(
        &mut self,
        command: commands::LoopCommand,
    ) -> LoopCommandResult {
        self.loop_controller
            .handle_command(&self.cwd, &self.session.id, command)
    }

    pub(crate) fn drain_due_loop_prompts(
        &mut self,
        now: std::time::Instant,
    ) -> Vec<super::automation::QueuedLoopPrompt> {
        self.loop_controller
            .drain_due_prompts(&self.cwd, &self.session.id, now)
    }

    /// Pop-time gate for a loop-owned queued prompt: decides whether a dequeued
    /// `/loop` run still fires (or was stopped/paused/exhausted in the meantime).
    pub(crate) fn begin_loop_turn(&mut self, loop_id: &str) -> super::automation::LoopTurnGate {
        self.loop_controller
            .begin_loop_turn(&self.cwd, &self.session.id, loop_id)
    }

    /// Charge a completed `/loop` turn's output tokens against the loop's optional
    /// `--token-budget`, so a bounded recurring loop stops at its token ceiling.
    pub(crate) fn charge_loop_output(&mut self, loop_id: &str, output_tokens: u32) {
        self.loop_controller
            .charge_loop_output(loop_id, output_tokens);
    }

    /// After a `/loop`-owned turn, re-check the loop's `--until` completion
    /// condition (if any) and stop the loop once it is met — so a goal-driven loop
    /// ends when its objective is achieved instead of running to its budget cap.
    /// The objective validators may block on `cargo`/`grep`, so they run off the
    /// async loop via `spawn_blocking` (the TUI keeps drawing); the cheap state
    /// mutation stays on the event loop. A loop with no `--until` is a no-op.
    /// Evaluate a recurring loop's `--until` completion check after a turn. Returns
    /// a user-facing notice when the loop STOPS on a stall (the surprising,
    /// new-give-up case the user must see); `None` otherwise (still running, or a
    /// silent `--until`-met completion). Runs the validators off the async loop so
    /// the TUI never freezes.
    pub(crate) async fn check_loop_until_after_turn(&mut self, loop_id: &str) -> Option<String> {
        let until = self.loop_controller.loop_until_validators(loop_id)?;
        let cwd = self.cwd.clone();
        let report =
            tokio::task::spawn_blocking(move || super::automation::run_validators(&cwd, &until, None))
                .await
                .ok()?;
        if report.ok {
            self.loop_controller.complete_loop(loop_id);
            return None;
        }
        match self
            .loop_controller
            .observe_loop_stall(loop_id, &report.objective_failures)
        {
            super::automation::LoopStallVerdict::Continue => None,
            super::automation::LoopStallVerdict::Stalled => {
                // The `--until` check has failed with the same objective signature
                // repeatedly: the loop is stuck. Stop it honestly instead of firing
                // forever, and tell the user why.
                self.loop_controller.stall_loop(loop_id);
                Some(format!(
                    "Loop {loop_id} stopped — `--until` stalled (same failure repeated with no progress)."
                ))
            }
            super::automation::LoopStallVerdict::Blocked(need) => {
                // The `--until` check keeps failing on something outside the
                // loop's control — retrying cannot fix it. Stop and escalate the
                // specific blocker + remedy (mirrors the goal's Blocked terminal).
                self.loop_controller.block_loop(loop_id, need);
                Some(format!(
                    "Loop {loop_id} stopped — `--until` blocked; needs {}. Next: {}",
                    need.label(),
                    need.remedy()
                ))
            }
        }
    }

    pub(crate) fn next_loop_due_in(&self, now: std::time::Instant) -> Option<std::time::Duration> {
        self.loop_controller.next_due_in(now)
    }

    pub(crate) fn next_loop_wake(
        &self,
        now: std::time::Instant,
    ) -> Option<(std::time::Duration, String)> {
        self.loop_controller
            .next_due_info(now)
            .map(|(due_in, reason)| (due_in, reason.trim().to_string()))
    }

    /// Pause a loop whose just-finished turn exhausted a turn budget. Returns
    /// `true` when it actually transitioned (an Active loop), so the caller emits
    /// the digest note + notice exactly once.
    pub(crate) fn pause_loop_for_budget(&mut self, loop_id: &str) -> bool {
        self.loop_controller.pause_for_budget(loop_id)
    }

    /// Subscribe this session to the team inbox `digest` channel (best-effort,
    /// fail-open) so an autonomous loop's notices surface in the turn-start digest.
    pub(crate) fn seed_digest_subscription(&self) {
        let _ = runtime::ensure_session_channel_subscription(&self.cwd, &self.session.id, "digest");
    }

    /// Record an autonomous-loop notice into the team inbox `digest` channel — the
    /// "propose / report" surface an unattended loop leaves for the user's morning
    /// review. Best-effort and fail-open (inbox trouble never blocks a turn):
    /// ensure the store exists, subscribe this session (so the post is unread to
    /// it), then post. All three steps share one root so the injection reads the
    /// same store.
    pub(crate) fn record_automation_digest(&self, summary: &str) {
        let root = runtime::team_inbox_store_root(&self.cwd);
        if tools::ensure_team_inbox_store(&root).is_err() {
            return;
        }
        let _ = runtime::ensure_session_channel_subscription(&self.cwd, &self.session.id, "digest");
        let _ = tools::host_post_team_inbox_update(&root, "digest", "zo-loop", summary);
    }

    pub(crate) fn automation_hud_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(goal) = self.goal_controller.hud_label() {
            lines.push(goal);
        }
        if let Some(loop_label) = self.loop_controller.hud_label() {
            lines.push(loop_label);
        }
        lines
    }

    fn prepare_turn_runtime(
        &self,
        emit_output: bool,
    ) -> Result<(BuiltRuntime, crate::HookAbortMonitor), Box<dyn std::error::Error>> {
        let hook_abort_signal = runtime::HookAbortSignal::new();
        // Thread the active effort's thinking budget into the per-turn runtime
        // so the headless `run_turn` / `--print` JSON paths honor `/effort`
        // (the TUI path overrides thinking via `LiveAsyncApiClient`).
        let mut runtime = self
            .build_runtime(
                self.runtime.session().clone(),
                &self.session.id,
                self.model.clone(),
                self.effective_system_prompt(),
                true,
                emit_output,
                // A per-turn override (a headless prompt-command's `allowed-tools`
                // frontmatter) wins over the session-global set, mirroring the TUI
                // turn build site. `run_turn_capturing` clears it after the turn so
                // it never leaks into the next one.
                self.turn_allowed_tools
                    .clone()
                    .or_else(|| self.allowed_tools.clone()),
                self.permission_mode,
                self.thinking_config(),
            )?
            .with_hook_abort_signal(hook_abort_signal.clone());
        self.apply_goal_controller_to_built_runtime(&mut runtime);
        Self::discover_mcp_tools_for_headless_turn(&mut runtime);
        // `--max-turns` (headless) caps the agentic loop on the per-turn
        // runtime. The runtime is rebuilt each turn, so apply it here rather
        // than once at construction. `None` inherits the runtime's
        // `DEFAULT_MAX_ITERATIONS` backstop (no longer unbounded).
        if let (Some(max_turns), Some(inner)) = (self.max_turns, runtime.try_runtime_mut()) {
            inner.set_max_iterations(max_turns);
        }
        if let (Some(max_tool_calls), Some(inner)) =
            (self.max_tool_calls, runtime.try_runtime_mut())
        {
            inner.set_max_tool_calls(max_tool_calls);
        }
        // Cost circuit breakers (wall clock + output/input tokens): the
        // headless paths previously ran with NONE of these — the interactive
        // TUI wired them in `turn_controller` only, so the unattended path
        // most likely to run away was the one without a net. Same env-driven
        // defaults as the TUI (`ZO_TURN_*`, `0` disables).
        if let Some(inner) = runtime.try_runtime_mut() {
            let (deadline, output_budget, input_budget) = runtime::env_turn_budgets();
            if let Some(budget) = deadline {
                inner.set_deadline(std::time::Instant::now() + budget);
            }
            inner.set_turn_output_token_budget(output_budget);
            inner.set_turn_input_token_budget(input_budget);
        }
        let hook_abort_monitor = crate::HookAbortMonitor::spawn(hook_abort_signal);

        Ok((runtime, hook_abort_monitor))
    }

    #[cfg(test)]
    pub(crate) fn prepare_turn_runtime_for_test(
        &self,
        emit_output: bool,
    ) -> Result<(BuiltRuntime, crate::HookAbortMonitor), Box<dyn std::error::Error>> {
        self.prepare_turn_runtime(emit_output)
    }

    /// Refresh this turn's cross-provider quota fallback and quota-wait band on
    /// `runtime`. The interactive TUI wires both per turn in `turn_controller`;
    /// the headless `-p` paths (`run_turn` text + ndjson/json) build a fresh
    /// runtime each turn via `prepare_turn_runtime`, and the long-lived
    /// serve/ACP streaming path reuses one runtime across turns — all funnel
    /// here so the two quota policies stay in one place and take effect *next*
    /// turn after a `/smart` edit or model switch, never only at construction.
    ///
    /// The fallback client is re-derived every call (the top-ranked
    /// different-provider peer) and set-or-cleared: `None` (Smart off,
    /// `smart.quotaFallback` off, or no cross-provider peer) installs nothing,
    /// so routing that returns `None` clears any stale client rather than
    /// leaking it. The wait band is read every call too — NOT gated on a
    /// fallback client existing, since holding for an imminent reset is valid
    /// with no peer. Kept off `prepare_turn_runtime` because deriving the route
    /// caches the constructed provider client on `&mut self`, and that method is
    /// `&self`.
    pub(crate) fn install_quota_fallback_client(&mut self, runtime: &mut BuiltRuntime) {
        let client = super::turn_controller::quota_fallback_async_client(self);
        let wait_band = super::smart_settings::quota_wait_band();
        if let Some(inner) = runtime.try_runtime_mut() {
            inner.set_quota_fallback_client(client);
            inner.set_quota_wait_band(wait_band);
        }
    }

    /// Same quota policy as [`install_quota_fallback_client`], but applied to
    /// this session's *own* long-lived `self.runtime` (the serve/ACP streaming
    /// path). Kept separate because `quota_fallback_async_client` borrows
    /// `&mut self` to cache the derived provider client, which cannot overlap a
    /// `&mut self.runtime` borrow; deriving both values first releases that
    /// borrow before touching the runtime.
    pub(crate) fn install_quota_fallback_client_on_self(&mut self) {
        let client = super::turn_controller::quota_fallback_async_client(self);
        let wait_band = super::smart_settings::quota_wait_band();
        if let Some(inner) = self.runtime.try_runtime_mut() {
            inner.set_quota_fallback_client(client);
            inner.set_quota_wait_band(wait_band);
        }
    }

    fn discover_mcp_tools_for_headless_turn(runtime: &mut BuiltRuntime) {
        let Some(mcp_state) = runtime.mcp_state.clone() else {
            return;
        };
        let registry = runtime.api_client().tool_registry();
        super::mcp_runtime::discover_pending_mcp_tools_now(&mcp_state, &registry);
        Self::refresh_runtime_tool_requirements(runtime);
    }

    pub(crate) fn start_mcp_discovery_in_background(&self) {
        Self::start_mcp_discovery_for_runtime(&self.runtime);
    }

    fn start_mcp_discovery_for_runtime(runtime: &BuiltRuntime) {
        let Some(mcp_state) = runtime.mcp_state.as_ref() else {
            return;
        };
        let registry = runtime.api_client().tool_registry();
        super::mcp_runtime::discover_pending_mcp_tools_in_background(mcp_state, registry);
    }

    fn refresh_runtime_tool_requirements(runtime: &mut BuiltRuntime) {
        let specs = runtime.api_client().tool_registry().permission_specs(None).ok();
        if let (Some(specs), Some(inner)) = (specs, runtime.try_runtime_mut()) {
            inner.refresh_tool_requirements(specs);
        }
    }

    pub(crate) fn replace_runtime(
        &mut self,
        mut new_runtime: BuiltRuntime,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let previous_context = self.runtime.runtime.as_mut().map(|runtime| {
            runtime
                .tool_executor_mut()
                .tool_registry_mut()
                .context()
                .clone()
        });
        if let (Some(previous_context), Some(runtime)) =
            (previous_context.as_ref(), new_runtime.try_runtime_mut())
        {
            runtime
                .tool_executor_mut()
                .tool_registry_mut()
                .context()
                .copy_workspace_checkpoint_state_from(previous_context);
        }
        self.apply_goal_controller_to_built_runtime(&mut new_runtime);
        // Shut down all subsystems explicitly before the swap.
        // The Drop impl is a no-op for already-shut-down subsystems.
        self.runtime.shutdown_lsp()?;
        self.runtime.shutdown_mcp()?;
        self.runtime.shutdown_plugins()?;

        // Take the inner ConversationRuntime out BEFORE replacing the
        // BuiltRuntime, so the Drop impl on the old BuiltRuntime
        // doesn't try to drop tokio resources from an async context
        // (which panics). Leak it instead — /resume runs at most a
        // handful of times per process so the leak is acceptable.
        if let Some(old_inner) = self.runtime.runtime.take() {
            std::mem::forget(old_inner);
        }
        // Swap in the new runtime and drop the old shell where blocking is
        // allowed. The old shell may still own a tokio runtime — the LSP
        // state's `owned_runtime`, built at startup with no ambient runtime —
        // and dropping that from inside the TUI's ambient multi-thread runtime
        // panics ("cannot drop a runtime in a non-blocking context"). This is
        // the drop-side twin of the LSP `run_blocking` block_in_place guard;
        // together they make a Shift+Tab / `/permission` rebuild safe.
        let old_shell = std::mem::replace(&mut self.runtime, new_runtime);
        if matches!(
            tokio::runtime::Handle::try_current().map(|handle| handle.runtime_flavor()),
            Ok(tokio::runtime::RuntimeFlavor::MultiThread)
        ) {
            tokio::task::block_in_place(move || drop(old_shell));
        } else {
            drop(old_shell);
        }
        // The swapped-in runtime is freshly built, so every MCP server it owns is
        // seeded `pending` with discovery deferred to a background pass (see
        // `build_runtime_mcp_state`). Startup kicks that pass exactly once, so a
        // mid-session rebuild (`/resume`, `/reload`, `/permission`, Shift+Tab,
        // model/effort change) would otherwise leave every server stuck
        // `Discovering` forever. Restart discovery on the new state here — the
        // single choke point every rebuild funnels through — so newly-pending
        // servers become eligible again.
        Self::start_mcp_discovery_for_runtime(&self.runtime);
        self.apply_session_system_reminders();
        Ok(())
    }

    pub(crate) fn run_turn(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.run_turn_capturing(input).map(|_| ())
    }

    /// Like [`run_turn`](Self::run_turn) but returns the turn's assistant
    /// output-token count, so the headless goal loop can charge it against the
    /// goal's token budget.
    pub(crate) fn run_turn_capturing(
        &mut self,
        input: &str,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        // The shared headless boundary builds the runtime and installs the
        // cross-provider quota fallback for this sync turn (the runtime swaps to
        // it — driven on a scoped runtime by the sync loop's `block_on` bridge —
        // if the main model's quota is exhausted mid-turn), keeping text/json/
        // ndjson in parity.
        let (mut runtime, hook_abort_monitor) = self.prepare_headless_turn_runtime(true)?;
        // The per-turn allowed-tools override (if any) is now baked into `runtime`;
        // clear it so a subsequent queued message or goal-loop turn falls back to
        // the session-global set rather than inheriting this command's restriction.
        self.turn_allowed_tools = None;
        // Gap B: nudge the model toward the right route shape (this headless
        // path never host pre-spawns, so the nudge is the only orchestration
        // signal). Set on the per-turn runtime, keyed by prefix so it never
        // accumulates across turns.
        let _turn_setup = TurnHarness::setup_model_led_turn(
            &mut runtime,
            ModelLedTurnSetup {
                input,
                effort: self.effort,
                session_tokens: self.runtime.estimated_tokens(),
                system_prompt: &self.system_prompt,
                clear_stale_reactive_gate: false,
            },
        );
        let mut spinner = crate::render::Spinner::new();
        // The spinner is cosmetic status decorated with ANSI color; it belongs
        // on stderr, never on machine-readable stdout, and is suppressed when
        // stderr is not a terminal or NO_COLOR is set so headless/piped text
        // output carries no escape sequences.
        let mut spinner_out = io::stderr();
        let show_spinner = spinner_out.is_terminal() && !crate::render::no_color_env();
        if show_spinner {
            spinner.tick(
                "🦀 Thinking...",
                TerminalRenderer::new().color_theme(),
                &mut spinner_out,
            )?;
        }
        let mut permission_prompter = crate::CliPermissionPrompter::new(self.permission_mode);
        let restore_deep_gate =
            TurnHarness::install_automation_plan_gate_if_needed(input, &mut runtime);
        let result = runtime.run_turn(input, Some(&mut permission_prompter));
        TurnHarness::restore_deep_gate(&mut runtime, restore_deep_gate);
        hook_abort_monitor.stop();
        match result {
            Ok(summary) => {
                self.replace_runtime(runtime)?;
                if show_spinner {
                    spinner.finish(
                        "✨ Done",
                        TerminalRenderer::new().color_theme(),
                        &mut spinner_out,
                    )?;
                }
                println!();
                // This turn's own output delta (not the session cumulative) — the
                // amount the goal token budget should charge for this turn.
                let output_tokens = summary.turn_output_tokens;
                if let Some(event) = summary.auto_compaction {
                    println!(
                        "{}",
                        format_auto_compaction_notice(event.removed_message_count)
                    );
                }
                self.persist_appended_session()?;
                Ok(output_tokens)
            }
            Err(error) => {
                runtime.shutdown_plugins()?;
                if show_spinner {
                    spinner.fail(
                        "❌ Request failed",
                        TerminalRenderer::new().color_theme(),
                        &mut spinner_out,
                    )?;
                }
                Err(Box::new(error))
            }
        }
    }

    pub(crate) fn run_turn_with_output(
        &mut self,
        input: &str,
        output_format: CliOutputFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match output_format {
            CliOutputFormat::Text => self.run_turn(input),
            CliOutputFormat::Json => self.run_prompt_json(input),
            CliOutputFormat::Ndjson => self.run_prompt_ndjson(input),
        }
    }

    /// The single boundary every headless `-p` turn (text, json, ndjson) crosses
    /// to obtain its per-turn runtime: build it via
    /// [`prepare_turn_runtime`](Self::prepare_turn_runtime) *and* install the
    /// cross-provider quota fallback, so a quota-exhausted turn swaps providers
    /// mid-turn regardless of output format. Folding the install in here keeps the
    /// three formats in parity (the JSON path once forgot it). `&mut self`
    /// because deriving the fallback route caches a client on `self`;
    /// `prepare_turn_runtime` stays `&self`.
    fn prepare_headless_turn_runtime(
        &mut self,
        emit_output: bool,
    ) -> Result<(BuiltRuntime, crate::HookAbortMonitor), Box<dyn std::error::Error>> {
        let (mut runtime, hook_abort_monitor) = self.prepare_turn_runtime(emit_output)?;
        self.install_quota_fallback_client(&mut runtime);
        Ok((runtime, hook_abort_monitor))
    }

    fn run_prompt_ndjson(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        // The shared headless boundary builds the runtime and installs the
        // cross-provider quota fallback (streaming client swaps if the main
        // model's quota is exhausted mid-turn), keeping text/json/ndjson in
        // parity.
        let (mut runtime, hook_abort_monitor) = self.prepare_headless_turn_runtime(false)?;
        // Gap B: model-led route nudge on this headless streaming path (no host
        // pre-spawn here). Keyed by prefix so it never accumulates across turns.
        let _turn_setup = TurnHarness::setup_model_led_turn(
            &mut runtime,
            ModelLedTurnSetup {
                input,
                effort: self.effort,
                session_tokens: self.runtime.estimated_tokens(),
                system_prompt: &self.system_prompt,
                clear_stale_reactive_gate: false,
            },
        );
        // Headless live streaming: drive the turn through `run_turn_streaming`
        // so each RenderBlock is emitted as a typed ndjson line the moment it
        // arrives (mid-turn byte streaming), instead of replaying a post-hoc
        // summary. Permission prompts are auto-denied (no human attached).
        let live_client = TurnHarness::build_live_client(
            &runtime,
            self.allowed_tools.clone(),
            self.thinking_config(),
            self.effort.and_then(Effort::level),
            self.effort.and_then(Effort::band_ceiling),
        );
        let tokio_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let restore_deep_gate =
            TurnHarness::install_automation_plan_gate_if_needed(input, &mut runtime);
        // Capability parity with the TUI: a code-changing headless `-p` turn
        // gets reactive auto-verify (implement→verify→retry). Installed *after*
        // the automation gate so its `deep_gate().is_some()` guard yields to a
        // plan-first/`/goal` gate; a no-op for analysis prompts or `ZO_AUTO_VERIFY=0`.
        let restore_reactive_gate =
            TurnHarness::install_reactive_verify_gate_if_coding(input, &mut runtime);
        let started = std::time::Instant::now();
        let result = match runtime.runtime.as_mut() {
            Some(rt) => tokio_rt.block_on(super::ndjson_summary::drive_ndjson_stream(
                rt,
                live_client,
                input.to_string(),
                &self.model,
            )),
            None => Err("runtime not available".into()),
        };
        let duration_ms = started.elapsed().as_millis();
        // Restore in reverse install order (reactive first, then automation).
        TurnHarness::restore_deep_gate(&mut runtime, restore_reactive_gate);
        TurnHarness::restore_deep_gate(&mut runtime, restore_deep_gate);
        hook_abort_monitor.stop();
        let summary = result.map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
        self.replace_runtime(runtime)?;
        self.persist_appended_session()?;

        let stdout = std::io::stdout();
        super::ndjson_summary::write_ndjson_result_event(
            &summary,
            &self.model,
            &self.session.id,
            duration_ms,
            stdout.lock(),
        )
    }

    /// Cloneable handle to this session's mid-turn steering queue (track 5 pair
    /// sessions). The `zo serve` helm pushes a `session.steer` message here
    /// while a socket turn is in flight; the turn drains it at the next
    /// tool-result boundary — the exact same contract the local REPL's command
    /// pump uses (see [`ConversationRuntime::steering_handle`]). Returns `None`
    /// only when the inner runtime has been taken (mid-`/resume` rebuild).
    pub(crate) fn steering_handle(&self) -> Option<runtime::SteeringQueue> {
        self.runtime.runtime.as_ref().map(ConversationRuntime::steering_handle)
    }

    /// Drive one streaming turn headlessly against this session's **long-lived**
    /// runtime, forwarding every [`RenderBlock`](runtime::message_stream::RenderBlock)
    /// into `block_tx` as it streams. The caller owns the receiver and decides
    /// where blocks go — `zo serve` funnels them onto an attached client's
    /// socket.
    ///
    /// Unlike the one-shot `-p` paths ([`run_prompt_ndjson`](Self::run_prompt_ndjson)),
    /// this reuses `self.runtime` directly instead of rebuilding it per turn —
    /// the long-lived-session contract the interactive TUI turn path
    /// (`run_live_turn_with_images`) already follows. Rebuilding here is unsafe
    /// anyway: `build_runtime` panics when called from inside an ambient Tokio
    /// runtime, which is exactly where the server drives turns.
    ///
    /// Persists the session on success, like every other turn path. `async`
    /// (runs inside the caller's runtime) — it never builds its own. Errors are
    /// returned as `String` (a `Send` type) so the future stays `Send` and can
    /// be driven on a spawned `zo serve` task.
    pub(crate) async fn run_turn_streaming_to_channel(
        &mut self,
        input: &str,
        block_tx: tokio::sync::mpsc::Sender<runtime::message_stream::RenderBlock>,
        permission: super::socket_permission::SocketPrompterConfig,
        hook_abort_signal: runtime::HookAbortSignal,
        user_cancel_requested: Arc<AtomicBool>,
    ) -> Result<runtime::TurnSummary, String> {
        self.run_turn_streaming_to_channel_with_prompter(
            input,
            block_tx,
            super::ndjson_summary::StreamPrompter::Socket(permission),
            hook_abort_signal,
            user_cancel_requested,
        )
        .await
    }

    pub(crate) async fn run_turn_streaming_to_channel_with_prompter(
        &mut self,
        input: &str,
        block_tx: tokio::sync::mpsc::Sender<runtime::message_stream::RenderBlock>,
        prompter: super::ndjson_summary::StreamPrompter,
        hook_abort_signal: runtime::HookAbortSignal,
        user_cancel_requested: Arc<AtomicBool>,
    ) -> Result<runtime::TurnSummary, String> {
        self.runtime
            .set_hook_abort_signal(hook_abort_signal.clone());
        self.start_mcp_discovery_in_background();
        let hook_abort_monitor = crate::HookAbortMonitor::spawn(hook_abort_signal.clone());
        // Toggle the ultracode orchestration reminder on the long-lived runtime,
        // matching the interactive turn path so `/effort ultracode` behaves the
        // same whether attached over a socket or driven directly.
        self.apply_session_system_reminders();
        // Gap B: same model-led route nudge the headless `-p` paths get, so a
        // socket-attached turn is steered toward the right route shape too (serve
        // never host pre-spawns). Compute before the `&mut self.runtime` borrow,
        // then set-or-clear by prefix so it never accumulates across turns.
        let session_tokens = self.runtime.estimated_tokens();
        let _turn_setup = TurnHarness::setup_model_led_turn(
            &mut self.runtime,
            ModelLedTurnSetup {
                input,
                effort: self.effort,
                session_tokens,
                system_prompt: &self.system_prompt,
                clear_stale_reactive_gate: false,
            },
        );
        let live_client = TurnHarness::build_live_client(
            &self.runtime,
            self.allowed_tools.clone(),
            self.thinking_config(),
            self.effort.and_then(Effort::level),
            self.effort.and_then(Effort::band_ceiling),
        );
        // Quota parity with the TUI and headless `-p` paths: re-derive and
        // set-or-clear the cross-provider quota fallback and refresh the
        // quota-wait band on this long-lived runtime every turn (not just at
        // construction), so a `/smart` edit or model switch takes effect next
        // turn and routing that returns `None` clears any stale client rather
        // than leaking it. Computed before the `&mut self.runtime` borrow
        // because deriving the route caches the provider client on `self`.
        self.install_quota_fallback_client_on_self();
        // serve reuses one long-lived runtime across turns: clear a `Reactive`
        // gate stranded by a prior turn whose future was cancelled mid-`await`
        // (its restore never ran) BEFORE capturing the automation-gate baseline,
        // so neither installer mistakes the stale gate for prior state. A
        // persistent `/goal`/PlanFirst gate is left untouched.
        TurnHarness::clear_stale_reactive_gate(&mut self.runtime);
        let restore_deep_gate =
            TurnHarness::install_automation_plan_gate_if_needed(input, &mut self.runtime);
        // Capability parity with `-p` ndjson and the TUI: a code-changing
        // socket-attached turn gets reactive auto-verify (implement→verify→retry).
        // Installed *after* the automation gate so its `deep_gate().is_some()`
        // guard yields to a plan-first/`/goal` gate; a no-op for analysis prompts
        // or `ZO_AUTO_VERIFY=0`. Restored each turn (this runtime is long-lived,
        // so a stale gate must never leak into the next, possibly non-coding turn).
        let restore_reactive_gate =
            TurnHarness::install_reactive_verify_gate_if_coding(input, &mut self.runtime);
        // Cost circuit breakers, re-applied every turn like the TUI does:
        // serve's long-lived runtime previously carried NO deadline or token
        // budgets, so a socket-driven turn could run away unbounded. Re-arming
        // per turn also keeps a stale deadline from a prior turn from firing.
        if let Some(inner) = self.runtime.try_runtime_mut() {
            let (deadline, output_budget, input_budget) = runtime::env_turn_budgets();
            match deadline {
                Some(budget) => inner.set_deadline(std::time::Instant::now() + budget),
                None => inner.clear_deadline(),
            }
            inner.set_turn_output_token_budget(output_budget);
            inner.set_turn_input_token_budget(input_budget);
        }
        let result = match self.runtime.runtime.as_mut() {
            Some(rt) => {
                let message_count_before = rt.session().messages.len();
                let completed = {
                    let turn = super::ndjson_summary::drive_render_stream(
                        rt,
                        live_client,
                        input.to_string(),
                        &self.model,
                        block_tx,
                        prompter,
                    );
                    tokio::pin!(turn);
                    tokio::select! {
                        biased;
                        result = &mut turn => Some(result),
                        () = async {
                            while !hook_abort_signal.is_aborted() {
                                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                            }
                        } => None,
                    }
                };
                if user_cancel_requested.load(Ordering::SeqCst) {
                    Err(rt
                        .cancel_streaming_turn_by_user(
                            "turn cancelled by session.cancel_turn",
                            message_count_before,
                        )
                        .to_string())
                } else if hook_abort_signal.is_aborted() {
                    Err(rt
                        .cancel_streaming_turn_by_host(
                            "turn aborted because the serve host stopped",
                            message_count_before,
                        )
                        .to_string())
                } else {
                    completed.expect("turn completed when no stop signal is set")
                }
            }
            None => Err("runtime not available".to_string()),
        };
        hook_abort_monitor.stop();
        // Restore in reverse install order (reactive first, then automation).
        TurnHarness::restore_deep_gate(&mut self.runtime, restore_reactive_gate);
        TurnHarness::restore_deep_gate(&mut self.runtime, restore_deep_gate);
        let summary = result?;
        self.persist_appended_session()
            .map_err(|error| error.to_string())?;
        Ok(summary)
    }

    fn run_prompt_json(&mut self, input: &str) -> Result<(), Box<dyn std::error::Error>> {
        // The shared headless boundary builds the runtime and installs the
        // cross-provider quota fallback for this JSON turn (the runtime swaps to
        // it if the main model's quota is exhausted mid-turn), keeping text/json/
        // ndjson in parity — the install the JSON path once forgot.
        let (mut runtime, hook_abort_monitor) = self.prepare_headless_turn_runtime(false)?;
        // Gap B: model-led route nudge on this headless JSON path (no host
        // pre-spawn here). Keyed by prefix so it never accumulates across turns.
        let _turn_setup = TurnHarness::setup_model_led_turn(
            &mut runtime,
            ModelLedTurnSetup {
                input,
                effort: self.effort,
                session_tokens: self.runtime.estimated_tokens(),
                system_prompt: &self.system_prompt,
                clear_stale_reactive_gate: false,
            },
        );
        let restore_deep_gate =
            TurnHarness::install_automation_plan_gate_if_needed(input, &mut runtime);
        // JSON stdout must stay machine-parseable: never prompt interactively
        // (even on a TTY), so permission requests resolve to a structured deny
        // in the emitted result rather than a prompt interleaved with stdout.
        let mut permission_prompter =
            crate::CliPermissionPrompter::new_non_interactive(self.permission_mode);
        let started = std::time::Instant::now();
        let result = runtime.run_turn(input, Some(&mut permission_prompter));
        let duration_ms = started.elapsed().as_millis();
        TurnHarness::restore_deep_gate(&mut runtime, restore_deep_gate);
        hook_abort_monitor.stop();
        let summary = result?;
        self.replace_runtime(runtime)?;
        self.persist_appended_session()?;
        println!(
            "{}",
            prompt_result_json(&summary, &self.model, &self.session.id, duration_ms)
        );
        Ok(())
    }

    pub(crate) fn persist_session(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.runtime.session().save_to_path(&self.session.path)?;
        self.persist_session_preferences()?;
        self.save_automation_state();
        Ok(())
    }

    /// Persist an ordinary turn. Messages already reached the bound JSONL via
    /// `Session::push_message`; only header/compaction or in-place transcript
    /// changes fall back to a full snapshot.
    fn persist_appended_session(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.runtime
            .session()
            .persist_appended_state_to_path(&self.session.path)?;
        self.persist_session_preferences()?;
        self.save_automation_state();
        Ok(())
    }

    /// Run the append-aware ordinary-turn persistence check away from the async
    /// TUI drive loop. The clean path only validates the writer fingerprint and
    /// returns; a rare dirty turn can still serialize and atomically rewrite.
    pub(crate) async fn persist_appended_session_offloaded(
        &self,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let session = self.runtime.session().clone();
        let path = self.session.path.clone();
        tokio::task::spawn_blocking(move || session.persist_appended_state_to_path(&path))
            .await
            .map_err(|join_error| format!("session persist task failed: {join_error}"))??;
        self.persist_session_preferences()?;
        self.save_automation_state();
        Ok(())
    }

    /// Persist the `/goal` and `/loop` controller state under `.zo/automation`,
    /// scoped to PROJECT sessions only (an ephemeral session keeps the working
    /// tree clean). Best-effort; never blocks a turn. `pub(crate)` so the
    /// headless one-shot slash path (`zo "/goal …"`), which never runs a
    /// turn and therefore never reaches `persist_session`, can still persist a
    /// goal/loop mutation (e.g. a clear) instead of silently dropping it.
    pub(crate) fn save_automation_state(&self) {
        if self.session_scope != SessionScope::Project {
            return;
        }
        let state = super::automation::persist::AutomationStatePersist {
            version: super::automation::persist::current_version(),
            goal: self.goal_controller.snapshot_persist(),
            loops: self.loop_controller.snapshot_persist(),
        };
        let is_empty = state.goal.is_none() && state.loops.is_empty();
        if is_empty {
            // The state file is project-scoped (shared across sessions in the
            // same cwd). A session that has no automation of its own must not let
            // its empty save *delete* a sibling session's active goal. Only a
            // session that previously wrote automation may remove it on clear.
            if !self.automation_persisted.get() {
                return;
            }
        } else {
            self.automation_persisted.set(true);
        }
        super::automation::persist::save(&self.cwd, &state);
    }

    /// Restore persisted `/goal` and `/loop` state on session startup. Fail-open:
    /// a missing/corrupt/version-skewed file simply starts without restored
    /// automation. A restored goal reloads Paused (see `restore_persist`).
    fn load_automation_state(&mut self) {
        let state = super::automation::persist::load(&self.cwd);
        let had_state = state.goal.is_some() || !state.loops.is_empty();
        if let Some(goal) = state.goal {
            self.goal_controller.restore_persist(goal);
            self.session_goal = self
                .goal_controller
                .active_goal_text()
                .map(std::string::ToString::to_string);
            // No per-turn token baseline to reseed: the goal charges each turn's
            // own `TurnSummary.turn_output_tokens` delta, which the runtime
            // measures within the turn — so a resumed goal starts counting cleanly
            // from its next turn regardless of historical cumulative.
        }
        self.loop_controller.restore_persist(&self.cwd, state.loops);
        // This session now owns automation state restored from disk; allow it to
        // remove the file later if the user clears the goal/loops.
        if had_state {
            self.automation_persisted.set(true);
        }
    }

    pub(crate) fn persist_session_preferences(&self) -> Result<(), Box<dyn std::error::Error>> {
        let preferences = SessionPreferences {
            model: Some(self.model.clone()),
            effort: self.effort.map(|effort| effort.canonical().to_string()),
            effort_budget: self.thinking_budget,
            model_handoff_memory: self.model_handoff_memory.clone(),
        };
        save_session_preferences(&self.session.path, &preferences)?;
        if self.session_scope == SessionScope::Project {
            save_project_preferences(&self.cwd, &preferences)?;
        }
        Ok(())
    }

    /// Capture the current worktree as a checkpoint snapshot so that a
    /// later Esc-Esc can restore the code to this point.
    ///
    /// Snapshots are *post-state* checkpoints (the resulting tree of each
    /// turn, plus an initial baseline captured at session start), matching
    /// the `SnapshotStack::undo` contract where the top snapshot mirrors the
    /// current worktree. A no-op when not in a git repo. The `turn_number`
    /// is purely informational (surfaced in the rewind report).
    ///
    /// Best-effort: capture failures are swallowed so an unusual worktree
    /// state never blocks a turn. They simply mean Esc-Esc finds nothing to
    /// rewind to, which is reported honestly to the user.
    /// Dry-run preview of what an Esc-Esc rewind would revert: the tracked
    /// files the worktree restore would touch. Empty when there is no earlier
    /// snapshot or nothing changed. Drives the rewind confirmation modal so a
    /// mistaken double-tap (e.g. an Esc meant to deny a permission prompt) can
    /// be cancelled before any model-authored file is overwritten.
    pub(crate) fn preview_rewind(&self) -> Vec<PathBuf> {
        self.snapshot_stack
            .as_ref()
            .and_then(runtime::git_snapshot::SnapshotStack::preview_undo)
            .unwrap_or_default()
    }

    pub(crate) fn capture_code_checkpoint(&mut self) {
        let turn_number = self.runtime.session().messages.len();
        if let Some(stack) = self.snapshot_stack.as_mut() {
            let _ = stack.capture(turn_number);
        }
    }

    pub(crate) fn workspace_rewind_report(
        &mut self,
        action: &commands::WorkspaceRewindAction,
    ) -> Result<String, String> {
        let Some(runtime) = self.runtime.try_runtime_mut() else {
            return Err("runtime is not available".to_string());
        };
        let context = runtime.tool_executor_mut().tool_registry_mut().context();
        match action {
            commands::WorkspaceRewindAction::List => Ok(
                tools::render_workspace_checkpoint_list(&context.workspace_checkpoints()),
            ),
            commands::WorkspaceRewindAction::Restore { turn_index, force } => context
                .restore_workspace_to_before(*turn_index, *force)
                .map(|summary| tools::render_workspace_restore_summary(&summary))
                .map_err(|error| error.to_string()),
        }
    }

    /// Rewind the previous turn's *conversation and code together* — the
    /// Esc-Esc combined checkpoint. Drops the last user+assistant message
    /// pair (`rewind_turns(1)`) and restores the worktree to the previous
    /// snapshot (`SnapshotStack::undo`).
    ///
    /// Returns a [`RewindCheckpointReport`] describing what was restored.
    /// The conversation rewind and the code restore are reported
    /// independently so a partial outcome (e.g. code restore blocked by a
    /// user edit) is surfaced truthfully rather than silently dropped.
    pub(crate) fn rewind_last_checkpoint(&mut self) -> RewindCheckpointReport {
        let removed = self.runtime.rewind_turns(1);

        let code = match self.snapshot_stack.as_mut() {
            None => CodeRewindOutcome::NoRepo,
            Some(stack) => match stack.undo() {
                Ok(result) => CodeRewindOutcome::Restored {
                    turn: result.restored_turn,
                },
                // `< 2 snapshots` (nothing earlier to restore to) reads as
                // "no earlier code state"; any other error (e.g. a path the
                // user edited since the snapshot) is surfaced verbatim.
                Err(error) => {
                    if stack.depth() < 2 {
                        CodeRewindOutcome::NoEarlierState
                    } else {
                        CodeRewindOutcome::Blocked {
                            reason: error.to_string(),
                        }
                    }
                }
            },
        };

        RewindCheckpointReport {
            messages_removed: removed,
            code,
        }
    }
}

/// Outcome of the code-side of an Esc-Esc rewind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CodeRewindOutcome {
    /// Worktree restored to the previous snapshot.
    Restored { turn: usize },
    /// Not in a git repository — no code snapshots exist.
    NoRepo,
    /// In a git repo but no earlier snapshot to restore to.
    NoEarlierState,
    /// Restore refused (e.g. a tracked path changed since the snapshot).
    Blocked { reason: String },
}

/// Result of [`LiveCli::rewind_last_checkpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RewindCheckpointReport {
    /// Conversation messages dropped by `rewind_turns(1)`.
    pub(crate) messages_removed: usize,
    /// What happened to the code side.
    pub(crate) code: CodeRewindOutcome,
}

impl RewindCheckpointReport {
    /// `true` when neither the conversation nor the code moved — used by
    /// the caller to show a "nothing to rewind" notice instead of a
    /// success report.
    pub(crate) fn is_noop(&self) -> bool {
        self.messages_removed == 0 && !matches!(self.code, CodeRewindOutcome::Restored { .. })
    }
}

/// Build the `--output-format json` result object for a completed headless
/// turn.
///
/// The Claude-Code-SDK key set (`type`, `subtype`, `is_error`, `result`,
/// `session_id`, `num_turns`, `duration_ms`, `total_cost_usd`, `usage`) comes
/// from the shared [`sdk_result_object`](super::ndjson_summary::sdk_result_object)
/// builder, so this path and the `stream-json` terminal event cannot drift in
/// naming or typing. Everything layered on afterwards — `message`, `model`,
/// `iterations`, `num_tool_uses`, `tool_uses`/`tool_results`, cache events, the
/// human-formatted `estimated_cost` — is an ADDITIVE zo extra.
///
/// Kept pure (no `self`, no IO) so the schema is unit-testable and cannot
/// silently drift. `is_error` is always `false` here: a failed turn returns
/// `Err` upstream and exits non-zero before this serializer runs, but emitting
/// the field keeps the schema stable so a consumer can branch on status without
/// parsing exit codes.
fn prompt_result_json(
    summary: &runtime::TurnSummary,
    model: &str,
    session_id: &str,
    duration_ms: u128,
) -> serde_json::Value {
    let tool_uses = crate::collect_tool_uses(summary);
    let mut value =
        super::ndjson_summary::sdk_result_object(summary, model, session_id, duration_ms);
    let total_cost_usd = value["total_cost_usd"].as_f64().unwrap_or(0.0);
    if let Some(object) = value.as_object_mut() {
        // Additive zo extras layered on top of the SDK key set. `message`
        // duplicates the SDK `result` for backward-compatible consumers (the
        // deep-eval extractor reads `message`).
        object.insert(
            "message".into(),
            json!(crate::final_assistant_text(summary)),
        );
        object.insert("model".into(), json!(model));
        object.insert("iterations".into(), json!(summary.iterations));
        object.insert("num_tool_uses".into(), json!(tool_uses.len()));
        object.insert(
            "auto_compaction".into(),
            json!(summary.auto_compaction.map(|event| json!({
                "removed_messages": event.removed_message_count,
                "notice": format_auto_compaction_notice(event.removed_message_count),
            }))),
        );
        object.insert("tool_uses".into(), json!(tool_uses));
        object.insert(
            "tool_results".into(),
            json!(crate::collect_tool_results(summary)),
        );
        object.insert(
            "prompt_cache_events".into(),
            json!(crate::collect_prompt_cache_events(summary)),
        );
        object.insert(
            "estimated_cost".into(),
            json!(runtime::format_usd(total_cost_usd)),
        );
    }
    value
}

#[cfg(test)]
mod rubric_grader_tests {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::time::Duration;

    use futures_util::future::join_all;

    use super::{
        build_rubric_grader_prompt, fold_semantic_with_rubric, post_turn_verify_diff_hash,
        record_decisive_post_turn_verify, reserve_post_turn_verify, truncate_bytes,
        working_tree_diff,
    };

    async fn delayed_verdict(
        delay_ms: u64,
        verdict: decision_core::LensVerdict,
    ) -> decision_core::LensVerdict {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        verdict
    }

    #[test]
    fn fold_is_conservative_any_reject_wins() {
        // Rubric reject blocks a stop even when the deep-lane verifier accepted.
        assert_eq!(fold_semantic_with_rubric(Some(true), Some(false)), Some(false));
        assert_eq!(fold_semantic_with_rubric(Some(false), Some(true)), Some(false));
        // A rubric accept can satisfy a goal with no deep-lane verdict.
        assert_eq!(fold_semantic_with_rubric(None, Some(true)), Some(true));
        // Both accept ⇒ accept; both absent ⇒ no signal.
        assert_eq!(fold_semantic_with_rubric(Some(true), Some(true)), Some(true));
        assert_eq!(fold_semantic_with_rubric(None, None), None);
    }

    #[test]
    fn fold_is_fail_open_when_the_grader_gives_no_signal() {
        // A failed/garbled grade (rubric = None) must not erase the deep-lane
        // verdict — the turn keeps whatever signal it already had.
        assert_eq!(fold_semantic_with_rubric(Some(true), None), Some(true));
        assert_eq!(fold_semantic_with_rubric(Some(false), None), Some(false));
    }

    #[test]
    fn prompt_carries_objective_criteria_output_and_diff() {
        let prompt = build_rubric_grader_prompt(
            "Write a report",
            &["cites sources".to_string(), "has a summary".to_string()],
            "I wrote the report.",
            "diff --git a/report.md",
        );
        assert!(prompt.contains("Write a report"));
        assert!(prompt.contains("1. cites sources"));
        assert!(prompt.contains("2. has a summary"));
        assert!(prompt.contains("I wrote the report."));
        assert!(prompt.contains("diff --git a/report.md"));
    }

    #[tokio::test]
    async fn concurrent_lens_collection_preserves_order_and_any_reject_fold() {
        let verdicts = join_all([
            delayed_verdict(30, decision_core::LensVerdict::Accept),
            delayed_verdict(1, decision_core::LensVerdict::Reject),
            delayed_verdict(10, decision_core::LensVerdict::Accept),
        ])
        .await;

        assert_eq!(
            verdicts,
            vec![
                decision_core::LensVerdict::Accept,
                decision_core::LensVerdict::Reject,
                decision_core::LensVerdict::Accept,
            ],
            "join_all must preserve lens declaration order, not completion order"
        );
        assert_eq!(
            decision_core::fold_lens_verdicts(
                &verdicts,
                decision_core::ConsensusPolicy::AnyReject,
            ),
            Some(false),
            "AnyReject semantics must stay identical after concurrent collection"
        );
    }

    #[test]
    fn post_turn_verify_diff_snapshot_is_not_later_worktree_state() {
        fn git(dir: &std::path::Path, args: &[&str]) {
            let output = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .expect("git should run");
            assert!(
                output.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-post-turn-diff-{unique}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        git(&dir, &["init"]);
        git(&dir, &["config", "user.email", "test@example.com"]);
        git(&dir, &["config", "user.name", "Test User"]);
        let file = dir.join("file.txt");
        std::fs::write(&file, "base\n").expect("write base");
        git(&dir, &["add", "file.txt"]);
        git(&dir, &["commit", "-m", "init"]);

        std::fs::write(&file, "base\nturn-a\n").expect("write turn A");
        let snapshot = working_tree_diff(&dir);
        std::fs::write(&file, "base\nturn-a\nturn-b\n").expect("write turn B");
        let current = working_tree_diff(&dir);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(snapshot.contains("turn-a"));
        assert!(!snapshot.contains("turn-b"));
        assert!(current.contains("turn-b"));
    }

    #[test]
    fn post_turn_verify_dedupes_only_decisive_dirty_diff_results() {
        let seen = Mutex::new(HashSet::new());
        let first = post_turn_verify_diff_hash("diff --git a/file b/file\n+one");
        let second = post_turn_verify_diff_hash("diff --git a/file b/file\n+two");

        assert!(
            reserve_post_turn_verify(&seen, &first),
            "first sighting of a dirty diff should reserve verification"
        );
        assert!(
            !reserve_post_turn_verify(&seen, &first),
            "an overlapping future for the same dirty diff must not duplicate the verifier call"
        );
        assert_eq!(record_decisive_post_turn_verify(&seen, &first, None), None);
        assert!(
            reserve_post_turn_verify(&seen, &first),
            "API errors/abstentions must release the reservation and remain retryable"
        );

        assert_eq!(
            record_decisive_post_turn_verify(&seen, &first, Some(false)),
            Some(false)
        );
        assert!(
            !reserve_post_turn_verify(&seen, &first),
            "a decisive warning result should suppress duplicate warnings for the same dirty diff"
        );
        assert!(
            reserve_post_turn_verify(&seen, &second),
            "different diffs must still be verified"
        );
        assert_eq!(
            record_decisive_post_turn_verify(&seen, &second, Some(true)),
            Some(true)
        );
        assert!(
            !reserve_post_turn_verify(&seen, &second),
            "a decisive clean result also marks that exact diff complete"
        );
    }

    #[test]
    fn truncate_marks_elision_on_a_char_boundary() {
        assert_eq!(truncate_bytes("short", 100), "short");
        let truncated = truncate_bytes("héllo world", 3);
        assert!(truncated.ends_with("…[truncated]"));
        // Never splits a multi-byte char (would panic on a non-boundary slice).
        assert!(truncated.starts_with('h'));
    }

    #[test]
    fn lens_verify_prompt_carries_objective_question_and_diff() {
        let prompt = super::build_lens_verify_prompt(
            "Fix the parser",
            "Could the change break existing callers?",
            "diff --git a/parser.rs",
        );
        assert!(prompt.contains("Fix the parser"));
        assert!(prompt.contains("Could the change break existing callers?"));
        assert!(prompt.contains("diff --git a/parser.rs"));
    }

    #[test]
    fn there_are_three_distinct_independent_verify_lenses() {
        // spec / regression / security — distinct concerns, so a blind spot in one
        // does not silence the others (unlike a single self-reporting forward pass).
        let names: Vec<&str> = super::INDEPENDENT_VERIFY_LENSES
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(names, ["spec", "regression", "security"]);
    }

    // --- Principle ②: solo competing-hypothesis self-critique (no-diff panel) ---

    #[test]
    fn competing_hypotheses_prompt_carries_the_conclusion_and_lens() {
        // Feeds the model's OWN conclusion (no diff / no objective) — the no-diff
        // case the spec/regression/security diff panel skips.
        let prompt = super::build_competing_hypotheses_prompt(
            "The slowdown is caused by the reveal EWMA converging on the burst peak.",
            "State the single strongest concrete objection to the conclusion.",
        );
        assert!(prompt.contains("reveal EWMA converging on the burst peak"));
        assert!(prompt.contains("single strongest concrete objection"));
        assert!(prompt.contains("the assistant's own answer"));
    }

    #[test]
    fn there_are_two_competing_hypotheses_lenses() {
        // One axis, two framings (enumerate alternatives / strongest objection),
        // kept to two so the ultracode burst stays cheap.
        let names: Vec<&str> = super::COMPETING_HYPOTHESES_LENSES
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(names, ["alternatives", "objection"]);
    }

    #[test]
    fn conclusion_is_substantive_gates_on_length() {
        // A short reply has no competing-hypothesis surface → skipped (no cost).
        assert!(!super::conclusion_is_substantive("done"));
        assert!(!super::conclusion_is_substantive("   ok, fixed.   "));
        // A substantive causal/decision conclusion is worth contesting.
        let long = "x".repeat(super::COMPETING_HYPOTHESES_MIN_BYTES);
        assert!(super::conclusion_is_substantive(&long));
    }

    #[test]
    fn independent_panel_only_strengthens_never_relaxes_the_verdict() {
        // The panel folds into the deep-lane verdict under AnyReject, so an
        // independent lens reject blocks a deep-lane accept (strengthens), while a
        // panel that abstains (all lenses errored) leaves the verdict intact.
        assert_eq!(fold_semantic_with_rubric(Some(true), Some(false)), Some(false));
        assert_eq!(fold_semantic_with_rubric(Some(true), None), Some(true));
        // It never flips a deep-lane reject into an accept.
        assert_eq!(fold_semantic_with_rubric(Some(false), Some(true)), Some(false));
    }
}

#[cfg(test)]
mod prompt_json_tests {
    use super::prompt_result_json;

    fn summary(iterations: usize, output_tokens: u32) -> runtime::TurnSummary {
        runtime::TurnSummary {
            assistant_messages: Vec::new(),
            tool_results: Vec::new(),
            prompt_cache_events: Vec::new(),
            iterations,
            usage: runtime::TokenUsage {
                input_tokens: 1_000,
                output_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            turn_output_tokens: output_tokens,
            auto_compaction: None,
            microcompact: None,
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted: None,
        }
    }

    /// The benchmark-comparability fields must all be present and well-typed so
    /// a harness can compare Zo against Claude Code without reverse
    /// engineering the schema.
    #[test]
    fn json_carries_benchmark_comparability_fields() {
        let value = prompt_result_json(&summary(4, 250), "claude-sonnet-4-6", "sess-1", 1_234);

        assert_eq!(value["is_error"], false);
        assert_eq!(value["model"], "claude-sonnet-4-6");
        assert_eq!(value["duration_ms"], 1_234);
        assert_eq!(value["iterations"], 4);
        assert_eq!(value["num_tool_uses"], 0);
        assert_eq!(value["usage"]["input_tokens"], 1_000);
        assert_eq!(value["usage"]["output_tokens"], 250);
        // Cost must be a number (not a string) so it is directly comparable.
        assert!(value["total_cost_usd"].is_number());
        assert!(value["estimated_cost"].is_string());
    }

    #[test]
    fn num_tool_uses_matches_tool_uses_len() {
        let value = prompt_result_json(&summary(1, 10), "claude-opus-4-8", "sess-2", 0);
        let len = value["tool_uses"].as_array().map_or(0, Vec::len);
        assert_eq!(value["num_tool_uses"], len);
    }

    /// Both headless result paths (`--output-format json` via `prompt_result_json`
    /// and `stream-json` via `write_ndjson_result_event`) must emit the
    /// Claude-Code-SDK key set with byte-identical names AND value shapes, so an
    /// SDK consumer parses either path the same way. This pins them to the shared
    /// `sdk_result_object` builder and fails if either path drops, renames, or
    /// retypes an SDK key.
    #[test]
    fn both_paths_emit_matching_sdk_key_set() {
        let summary = summary(4, 250);
        let model = "claude-sonnet-4-6";
        let session_id = "sess-shared";
        let duration_ms = 1_234;

        // json path.
        let json_value = prompt_result_json(&summary, model, session_id, duration_ms);

        // stream-json terminal event path: capture the line the production
        // writer emits and parse it back.
        let mut buf = Vec::new();
        super::super::ndjson_summary::write_ndjson_result_event(
            &summary,
            model,
            session_id,
            duration_ms,
            &mut buf,
        )
        .expect("write_ndjson_result_event should succeed");
        let ndjson_value: serde_json::Value =
            serde_json::from_slice(&buf).expect("ndjson result line should be valid JSON");

        // The required SDK keys, by name. Both objects must carry every one.
        let sdk_keys = [
            "type",
            "subtype",
            "is_error",
            "result",
            "session_id",
            "num_turns",
            "duration_ms",
            "total_cost_usd",
            "usage",
        ];
        for key in sdk_keys {
            assert!(
                json_value.get(key).is_some(),
                "json path missing SDK key {key:?}"
            );
            assert!(
                ndjson_value.get(key).is_some(),
                "ndjson path missing SDK key {key:?}"
            );
            // Identical value (same name AND same shape/content) across paths.
            assert_eq!(
                json_value[key], ndjson_value[key],
                "SDK key {key:?} diverges between json and ndjson paths"
            );
        }

        // Spot-check the SDK contract values themselves.
        assert_eq!(json_value["type"], "result");
        assert_eq!(json_value["subtype"], "success");
        assert_eq!(json_value["is_error"], false);
        assert_eq!(json_value["session_id"], session_id);
        assert_eq!(json_value["num_turns"], 4);
        assert!(json_value["total_cost_usd"].is_number());
        assert!(json_value["result"].is_string());
    }
}

#[cfg(test)]
mod prompt_override_tests {
    use super::super::session_preferences::preferences_path;
    use super::{
        MODEL_HANDOFF_REMINDER_PREFIX, PLAN_MODE_REMINDER_PREFIX, SESSION_GOAL_REMINDER_PREFIX,
        SessionPreferences, apply_prompt_overrides, effort_from_preferences,
        model_handoff_system_reminder, plan_mode_system_reminder, session_goal_system_reminder,
    };
    use super::super::turn_harness::TurnHarness;
    use runtime::{ContentBlock, ConversationMessage, DeepGateConfig, DeepMode};
    use zo_cli::tui::modals::Effort;
    use std::path::PathBuf;

    #[test]
    fn replace_swaps_entire_base() {
        let out = apply_prompt_overrides(
            vec!["base-a".into(), "base-b".into()],
            Some("custom".into()),
            None,
        );
        assert_eq!(out, vec!["custom".to_string()]);
    }

    #[test]
    fn append_keeps_base_and_adds_segment() {
        let out = apply_prompt_overrides(vec!["base".into()], None, Some("extra".into()));
        assert_eq!(out, vec!["base".to_string(), "extra".to_string()]);
    }

    #[test]
    fn no_overrides_returns_base_unchanged() {
        let base = vec!["base".to_string()];
        let out = apply_prompt_overrides(base.clone(), None, None);
        assert_eq!(out, base);
    }

    #[test]
    fn replace_then_append_compose_in_order() {
        let out = apply_prompt_overrides(
            vec!["base".into()],
            Some("custom".into()),
            Some("extra".into()),
        );
        assert_eq!(out, vec!["custom".to_string(), "extra".to_string()]);
    }

    #[test]
    fn goal_reminder_is_marked_and_trimmed() {
        let reminder = session_goal_system_reminder("  ship the HUD fix  ").expect("reminder");
        assert!(reminder.starts_with(SESSION_GOAL_REMINDER_PREFIX));
        assert!(reminder.contains("ship the HUD fix"));
        assert!(reminder.contains("plan before acting"));
        assert!(!reminder.contains("  ship"));
    }

    #[test]
    fn empty_goal_has_no_reminder() {
        assert!(session_goal_system_reminder("   ").is_none());
    }

    /// The Plan contract must be self-marking and must state — deterministically,
    /// not by model luck — that plan mode is already active and `EnterPlanMode`
    /// must not be called. This is what prevents the duplicate write-gated
    /// `EnterPlanMode` denial the model otherwise hits every turn under a
    /// user-selected (read-only) Plan.
    #[test]
    fn plan_mode_reminder_states_plan_is_active_and_forbids_re_entering() {
        let reminder = plan_mode_system_reminder();
        assert!(reminder.starts_with(PLAN_MODE_REMINDER_PREFIX));
        assert!(reminder.contains("Plan mode is already active"));
        assert!(
            reminder.contains("Do NOT call EnterPlanMode"),
            "must explicitly forbid the write-gated re-entry tool: {reminder}"
        );
        assert!(
            reminder.contains("ExitPlanModeV2"),
            "must point at the read-only plan-submission tool: {reminder}"
        );
        assert!(
            reminder.contains("Shift+Tab") && reminder.contains("/plan off"),
            "must say only the user leaves plan mode: {reminder}"
        );
        // Must never invite the model to restore write access on its own.
        assert!(!reminder.to_lowercase().contains("switch to workspace"));
    }

    #[test]
    fn automation_plan_gate_token_is_scoped_and_restorable() {
        let reactive = DeepGateConfig {
            mode: DeepMode::Reactive,
            check_command: Some("cargo test".to_string()),
            max_attempts: 3,
        };
        let plan_first = DeepGateConfig {
            mode: DeepMode::PlanFirst,
            check_command: None,
            max_attempts: 2,
        };

        assert!(
            super::super::automation::automation_plan_gate_change("manual prompt", Some(&reactive))
                .is_none()
        );
        let change = super::super::automation::automation_plan_gate_change(
            "[zo:automation-plan-first] loop automation",
            Some(&reactive),
        )
        .expect("automation prompt should produce restore token");
        let restored_reactive = change
            .restore
            .expect("previous reactive config should be preserved");
        assert!(matches!(restored_reactive.mode, DeepMode::Reactive));
        assert_eq!(
            restored_reactive.check_command.as_deref(),
            Some("cargo test")
        );
        assert_eq!(restored_reactive.max_attempts, 3);
        assert!(matches!(
            change.install.as_ref().map(|config| config.mode),
            Some(DeepMode::PlanFirst)
        ));
        assert!(
            super::super::automation::automation_plan_gate_change(
                "[zo:automation-plan-first] loop automation",
                None,
            )
            .expect("automation prompt should produce restore token")
            .restore
            .is_none()
        );
        assert!(super::super::automation::should_install_automation_plan_gate(Some(&reactive)));
        assert!(!super::super::automation::should_install_automation_plan_gate(Some(&plan_first)));
    }

    #[test]
    fn model_handoff_reminder_preserves_recent_visible_context() {
        let messages = vec![
            ConversationMessage::user_text("Plan with ChatGPT, then implement with Opus."),
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: "Plan: inspect auth, persist preferences, verify tests.".to_string(),
            }]),
        ];
        let reminder = model_handoff_system_reminder(
            "gpt-5.5",
            "claude-opus-4-8",
            &messages,
            Some("ship multi-model TUI"),
        )
        .expect("handoff reminder");

        assert!(reminder.starts_with(MODEL_HANDOFF_REMINDER_PREFIX));
        assert!(reminder.contains("gpt-5.5"));
        assert!(reminder.contains("claude-opus-4-8"));
        assert!(reminder.contains("ship multi-model TUI"));
        assert!(reminder.contains("Plan: inspect auth"));
        let lower = reminder.to_lowercase();
        assert!(
            !lower.contains("continue the same work") && !lower.contains("without restarting"),
            "handoff reminder must describe state, not prompt visible continuation filler: {reminder}"
        );
    }

    #[test]
    fn preferences_path_sits_outside_sessions_dir() {
        let path = preferences_path(&PathBuf::from("/tmp/.zo/sessions/session-1.jsonl"));
        assert_eq!(
            path,
            PathBuf::from("/tmp/.zo/session-prefs/session-1.json")
        );
    }

    #[test]
    fn effort_preferences_preserve_off_without_defaulting_high() {
        let prefs = SessionPreferences {
            effort: Some("off".to_string()),
            effort_budget: None,
            ..SessionPreferences::default()
        };
        let (effort, budget) = effort_from_preferences(&prefs);
        assert_eq!(effort, Some(Effort::Off));
        assert_eq!(budget, None);
    }

    #[test]
    fn reactive_verify_gate_installs_only_for_coding_turns_without_a_gate() {
        let coding = "fix the bug in src/click/core.py";
        let analysis = "analyze and summarize the architecture";

        // Opted-in + coding + no existing gate → install.
        assert!(TurnHarness::reactive_verify_gate_wanted(coding, false, false));
        // Analysis prompt → never (stays single-pass, no added cost).
        assert!(!TurnHarness::reactive_verify_gate_wanted(
            analysis, false, false
        ));
        // A gate already present (e.g. `/goal` or plan-first automation) → yield,
        // so the two never split-brain.
        assert!(!TurnHarness::reactive_verify_gate_wanted(coding, false, true));
        // Explicit opt-out wins even for a coding turn with no gate.
        assert!(!TurnHarness::reactive_verify_gate_wanted(coding, true, false));
    }

    #[test]
    fn opt_out_fully_gates_a_coding_turn() {
        // The `ZO_AUTO_VERIFY` opt-out (parsed by `auto_verify_opted_out` as
        // `0`/`off`) must fully suppress the gate even on a code-changing turn
        // with no existing gate — the one case that would otherwise install.
        // Exercised via the pure helper so the test never mutates the
        // process-global env (`set_var` is `unsafe`, which the crate forbids).
        assert!(TurnHarness::reactive_verify_gate_wanted(
            "fix src/x.rs",
            false,
            false
        ));
        assert!(!TurnHarness::reactive_verify_gate_wanted(
            "fix src/x.rs",
            true,
            false
        ));
    }

    #[test]
    fn serve_and_ndjson_share_one_reactive_gate_policy() {
        // serve (`run_turn_streaming_to_channel`) and `-p` ndjson
        // (`run_prompt_ndjson`) both install via
        // `install_reactive_verify_gate_if_coding`, which delegates to this one
        // pure predicate — so a socket-attached turn and a headless `-p` turn make
        // the identical install decision for the same (prompt, opt-out,
        // existing-gate) inputs. Pinning it here guards the two transports against
        // ever diverging.
        for &(prompt, opted_out, has_gate, want) in &[
            ("fix the bug in src/click/core.py", false, false, true),
            (
                "analyze and summarize the architecture",
                false,
                false,
                false,
            ),
            ("fix the bug in src/click/core.py", false, true, false),
            ("fix the bug in src/click/core.py", true, false, false),
        ] {
            assert_eq!(
                TurnHarness::reactive_verify_gate_wanted(prompt, opted_out, has_gate),
                want,
                "prompt={prompt:?} opted_out={opted_out} has_gate={has_gate}"
            );
        }
    }

    #[test]
    fn headless_objective_check_defaults_to_verifier_only_and_honors_explicit_opt_in() {
        // Single test owns the process-global env var end-to-end so it never
        // races a sibling: the whole unset → set → blank → cleanup cycle is
        // observed under one test, mirroring the env discipline of the
        // resume-cache tests.
        let prev = std::env::var("ZO_AUTO_VERIFY_CMD").ok();

        // Unset ⇒ verifier-only (None): the host does NOT guess a cargo/npm/…
        // command from the cwd. This is the whole point — no host hallucination,
        // matching the interactive default.
        std::env::remove_var("ZO_AUTO_VERIFY_CMD");
        assert_eq!(
            TurnHarness::headless_objective_check_command(),
            None,
            "no explicit command ⇒ verifier-only, never a cwd-marker guess"
        );

        // Explicit opt-in is honored verbatim (the headless equivalent of
        // interactive `/auto <cmd>`).
        std::env::set_var("ZO_AUTO_VERIFY_CMD", "cargo test -p runtime");
        assert_eq!(
            TurnHarness::headless_objective_check_command().as_deref(),
            Some("cargo test -p runtime")
        );

        // Blank / whitespace is treated as unset, not an empty command.
        std::env::set_var("ZO_AUTO_VERIFY_CMD", "   ");
        assert_eq!(TurnHarness::headless_objective_check_command(), None);

        // Restore the prior environment so sibling tests are unaffected.
        match prev {
            Some(value) => std::env::set_var("ZO_AUTO_VERIFY_CMD", value),
            None => std::env::remove_var("ZO_AUTO_VERIFY_CMD"),
        }
    }
}

#[cfg(test)]
mod spawn_context_tests {
    use super::{BuiltRuntime, LiveCli, post_turn_verify_diff_hash, working_tree_diff};
    use super::super::turn_harness::TurnHarness;

    use futures_util::FutureExt;
    use zo_cli::tui::modals::Effort;
    use std::path::{Path, PathBuf};
    use std::sync::MutexGuard;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct CurrentDirGuard {
        original: PathBuf,
        _lock: MutexGuard<'static, ()>,
    }

    impl CurrentDirGuard {
        fn enter(path: &Path) -> Self {
            let lock = crate::test_cwd_lock()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let original = std::env::current_dir().expect("cwd should exist");
            std::env::set_current_dir(path).expect("set current dir");
            Self {
                original,
                _lock: lock,
            }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    struct ApiKeyGuard {
        previous: Option<std::ffi::OsString>,
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn remove(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = self.previous.take() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    impl ApiKeyGuard {
        fn set_dummy() -> Self {
            let previous = std::env::var_os("ANTHROPIC_API_KEY");
            std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-live-cli");
            Self { previous }
        }
    }

    impl Drop for ApiKeyGuard {
        fn drop(&mut self) {
            if let Some(value) = self.previous.take() {
                std::env::set_var("ANTHROPIC_API_KEY", value);
            } else {
                std::env::remove_var("ANTHROPIC_API_KEY");
            }
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        // Every test entering a temp cwd would otherwise persist its session
        // into the developer's real ~/.zo/projects/ (see
        // isolate_global_zo_home_for_tests).
        crate::isolate_global_zo_home_for_tests();
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_millis();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "zo-live-cli-{label}-{}-{millis}-{counter}",
            std::process::id()
        ))
    }

    fn spawn_parent_model(runtime: &mut BuiltRuntime) -> Option<String> {
        runtime.try_runtime_mut().and_then(|inner| {
            inner
                .tool_executor_mut()
                .tool_registry_mut()
                .context()
                .spawn_parent_model()
        })
    }

    fn live_spawn_parent_model(cli: &mut LiveCli) -> Option<String> {
        spawn_parent_model(&mut cli.runtime)
    }

    fn test_cli(label: &str) -> (LiveCli, PathBuf) {
        let temp_dir = unique_temp_dir(label);
        std::fs::create_dir_all(&temp_dir).expect("temp dir should exist");
        let _cwd = CurrentDirGuard::enter(&temp_dir);
        let _api_key = ApiKeyGuard::set_dummy();
        let cli = LiveCli::new(
            "sonnet".to_string(),
            true,
            None,
            runtime::PermissionMode::ReadOnly,
        )
        .expect("live cli should build");
        (cli, temp_dir)
    }

    fn with_smart_settings<T>(
        config_home: &Path,
        settings: &str,
        run: impl FnOnce() -> T,
    ) -> T {
        std::fs::write(config_home.join("settings.json"), settings)
            .expect("test smart settings should be writable");
        super::super::smart_settings::with_test_config_home(config_home, run)
    }

    fn git(dir: &std::path::Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git should run");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// Proportionality: a no-diff Smart turn whose ASK reads trivial/small (a
    /// lookup) must skip the competing-hypotheses panel entirely — the future
    /// resolves `None` on the first poll, proving no lens round-trip was even
    /// started (a vacuous pass via network-failure abstention stays pending).
    #[test]
    fn simple_ask_skips_the_no_diff_competing_hypotheses_panel() {
        use runtime::{ContentBlock, ConversationMessage};
        let (mut cli, _temp_dir) = test_cli("post-turn-verify-simple-ask");
        cli.effort = Some(Effort::Smart);
        // No git repo in the temp dir → empty diff → only the no-diff branch.
        {
            let runtime = cli.runtime.try_runtime_mut().expect("runtime");
            let session = runtime.session_mut();
            session
                .push_message(ConversationMessage::user_text("이 설정값이 뭔지 알려줘"))
                .expect("push user ask");
            session
                .push_message(ConversationMessage::assistant(vec![ContentBlock::Text {
                    // Longer than COMPETING_HYPOTHESES_MIN_BYTES so only the
                    // ask-complexity gate (not the length gate) can skip it.
                    text: "이 설정값은 라우팅 임계값입니다. ".repeat(30),
                }]))
                .expect("push assistant conclusion");
        }
        let fut = cli
            .post_turn_verify_future()
            .expect("Smart effort still constructs the future");
        assert_eq!(
            fut.now_or_never(),
            Some(None),
            "a simple lookup ask must resolve immediately with no panel"
        );
    }

    #[test]
    fn marked_verified_diff_is_never_panel_verified_again() {
        // "Verified once, never re-verified": after the reactive gate accepts a
        // turn's change, the worktree hash is recorded — a later Smart turn
        // over the same still-dirty worktree must skip the panel (resolve to
        // `None` without any lens round-trip).
        let (mut cli, temp_dir) = test_cli("post-turn-verify-marked");
        cli.effort = Some(Effort::Smart);

        git(&temp_dir, &["init"]);
        git(&temp_dir, &["config", "user.email", "test@example.com"]);
        git(&temp_dir, &["config", "user.name", "Test User"]);
        let file = temp_dir.join("file.txt");
        std::fs::write(&file, "base\n").expect("write base");
        git(&temp_dir, &["add", "."]);
        git(&temp_dir, &["commit", "-m", "init"]);
        std::fs::write(&file, "base\nverified-change\n").expect("dirty the tree");

        cli.mark_worktree_diff_verified();
        let fut = cli
            .post_turn_verify_future()
            .expect("Smart effort still constructs the future");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        assert_eq!(
            runtime.block_on(fut),
            None,
            "a diff marked verified must short-circuit before any lens call"
        );
    }

    #[test]
    fn post_turn_verify_future_snapshots_diff_before_polling() {
        let (mut cli, temp_dir) = test_cli("post-turn-verify-snapshot");
        // Smart (ultracode) — the panel no longer runs on plain High+ turns
        // (one verification per change; the reactive gate owns ordinary turns).
        cli.effort = Some(Effort::Smart);

        git(&temp_dir, &["init"]);
        git(&temp_dir, &["config", "user.email", "test@example.com"]);
        git(&temp_dir, &["config", "user.name", "Test User"]);
        let file = temp_dir.join("file.txt");
        std::fs::write(&file, "base\n").expect("write base");
        git(&temp_dir, &["add", "."]);
        git(&temp_dir, &["commit", "-m", "init"]);

        std::fs::write(&file, "base\nturn-a\n").expect("write turn A");
        let captured_diff = working_tree_diff(&temp_dir);
        assert!(captured_diff.contains("turn-a"));
        assert!(!captured_diff.contains("turn-b"));
        let captured_hash = post_turn_verify_diff_hash(&captured_diff);

        let fut = cli
            .post_turn_verify_future()
            .expect("dirty High-effort turn should create verifier future");

        std::fs::write(&file, "base\nturn-a\nturn-b\n").expect("write turn B");
        let later_worktree_diff = working_tree_diff(&temp_dir);
        assert!(later_worktree_diff.contains("turn-b"));
        let later_worktree_hash = post_turn_verify_diff_hash(&later_worktree_diff);
        assert_ne!(captured_hash, later_worktree_hash);

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        runtime.block_on(async {
            let _ = fut.now_or_never();
        });
        let seen = cli
            .post_turn_verified_diffs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            seen.contains(&captured_hash),
            "post-turn verifier must reserve the diff captured when the future was created"
        );
        assert!(
            !seen.contains(&later_worktree_hash),
            "post-turn verifier must not read a later worktree diff on first poll"
        );

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn active_model_is_applied_to_rebuilt_runtime_spawn_context() {
        let (mut cli, temp_dir) = test_cli("active-model-rebuild");
        assert_eq!(live_spawn_parent_model(&mut cli).as_deref(), Some("sonnet"));

        cli.model = "claude-opus-4-8".to_string();
        let mut rebuilt = cli
            .build_runtime(
                cli.runtime.session().clone(),
                &cli.session.id,
                cli.model.clone(),
                cli.system_prompt.clone(),
                true,
                true,
                cli.allowed_tools.clone(),
                cli.permission_mode,
                None,
            )
            .expect("runtime rebuild should preserve active spawn context");
        assert_eq!(
            spawn_parent_model(&mut rebuilt).as_deref(),
            Some("claude-opus-4-8")
        );

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn clear_stale_reactive_gate_clears_reactive_but_preserves_plan_first() {
        let (mut cli, temp_dir) = test_cli("stale-reactive-gate");

        // A `Reactive` gate stranded by a cancelled prior turn is cleared at
        // entry, so the long-lived serve runtime starts the next turn clean.
        if let Some(inner) = cli.runtime.try_runtime_mut() {
            inner.set_deep_gate(Some(runtime::DeepGateConfig {
                mode: runtime::DeepMode::Reactive,
                check_command: None,
                max_attempts: 2,
            }));
        }
        TurnHarness::clear_stale_reactive_gate(&mut cli.runtime);
        assert!(
            cli.runtime.deep_gate().is_none(),
            "a stranded reactive gate must be cleared"
        );

        // A persistent `PlanFirst` gate (e.g. an active `/goal`) is left intact.
        if let Some(inner) = cli.runtime.try_runtime_mut() {
            inner.set_deep_gate(Some(runtime::DeepGateConfig {
                mode: runtime::DeepMode::PlanFirst,
                check_command: Some("cargo test".to_string()),
                max_attempts: 2,
            }));
        }
        TurnHarness::clear_stale_reactive_gate(&mut cli.runtime);
        assert!(
            matches!(
                cli.runtime.deep_gate().map(|gate| gate.mode),
                Some(runtime::DeepMode::PlanFirst)
            ),
            "a persistent plan-first/goal gate must be preserved"
        );

        // No gate → no-op.
        if let Some(inner) = cli.runtime.try_runtime_mut() {
            inner.set_deep_gate(None);
        }
        TurnHarness::clear_stale_reactive_gate(&mut cli.runtime);
        assert!(cli.runtime.deep_gate().is_none());

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn clear_session_refreshes_agent_manifest_scope() {
        let (mut cli, temp_dir) = test_cli("agent-scope-clear");
        cli.agent_manifest_started_after = 1;

        let report = cli
            .clear_session_report(true)
            .expect("clear should create a fresh session");

        assert!(report.contains("Session cleared"));
        assert!(
            cli.agent_manifest_started_after > 1,
            "fresh sessions must not inherit previous workspace agent manifests"
        );

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn fast_session_swap_restamps_tool_context_session_id() {
        // Invisible-swarm regression: an in-place session swap (/clear, /new,
        // fast /resume) must re-stamp the shared tool-context session id, or
        // every SpawnMultiAgent member manifest keeps the *pre-swap* id and the
        // TUI's strict session filter (`allow_unstamped = false`) drops all of
        // them — leaving the inline agent tree stuck on `spawning…` forever.
        let (mut cli, temp_dir) = test_cli("agent-scope-restamp");

        let original_id = cli.session.id.clone();
        cli.clear_session_report(true)
            .expect("clear should create a fresh session");
        let fresh_id = cli.session.id.clone();
        assert_ne!(fresh_id, original_id, "clear must mint a new session id");

        let ctx_session_id = cli
            .runtime
            .try_runtime_mut()
            .expect("live runtime present after fast swap")
            .tool_executor_mut()
            .tool_registry_mut()
            .context()
            .session_id();
        assert_eq!(
            ctx_session_id.as_deref(),
            Some(fresh_id.as_str()),
            "spawned-agent manifests must be stamped with the current session id"
        );

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn clear_session_scopes_todo_store_to_fresh_session() {
        let _env_lock = crate::test_env_lock();
        let _todo_env = EnvVarGuard::remove("ZO_TODO_STORE");
        let (mut cli, temp_dir) = test_cli("todo-scope-clear");
        let old_store = std::env::var_os("ZO_TODO_STORE")
            .map(PathBuf::from)
            .expect("new cli should scope a per-session todo store");
        std::fs::write(
            &old_store,
            r#"[{"content":"old ghost plan","activeForm":"old ghost plan","status":"pending"}]"#,
        )
        .expect("old todo store should be writable");

        let report = cli
            .clear_session_report(true)
            .expect("clear should create a fresh session");

        let new_store = std::env::var_os("ZO_TODO_STORE")
            .map(PathBuf::from)
            .expect("clear should keep todo store scoped");
        assert!(report.contains("Session cleared"));
        assert_ne!(
            new_store, old_store,
            "a fresh session must not keep reading the previous session's todo store"
        );
        assert_eq!(new_store, cli.session.path.with_extension("todos.json"));
        assert!(
            !new_store.exists(),
            "fresh clear should remove any stale todo file at the new session path"
        );
        assert!(
            old_store.exists(),
            "clearing must not destroy the previous session's resumable todo store"
        );

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn model_pin_reports_preference_persistence_failure() {
        let (mut cli, temp_dir) = test_cli("model-pin-warning");
        let blocker = temp_dir.join("blocked");
        std::fs::write(&blocker, "not a directory").expect("create path blocker");
        cli.session.path = blocker.join("session.jsonl");

        let report = cli.apply_model_change("claude-opus-4-8");

        assert!(
            report.contains("Warning          model preference was not saved:"),
            "model switch report must surface the non-fatal persistence failure: {report}"
        );
        std::fs::remove_dir_all(temp_dir).expect("cleanup");
    }

    /// Point `session.path` under a regular file so `persist_session_preferences`
    /// (which writes next to it) fails deterministically — the same seam the
    /// model-pin persistence-failure test uses, with no env or global-state race.
    fn cli_with_unwritable_preferences(label: &str) -> (LiveCli, PathBuf) {
        let (mut cli, temp_dir) = test_cli(label);
        let blocker = temp_dir.join("blocked");
        std::fs::write(&blocker, "not a directory").expect("create path blocker");
        cli.session.path = blocker.join("session.jsonl");
        (cli, temp_dir)
    }

    /// A named `/effort` preset must still apply in-memory when persistence
    /// fails, and return a surface-ready warning worded like the model-switch
    /// warning — never swallow the error.
    #[test]
    fn set_effort_preset_surfaces_persistence_failure_but_still_applies() {
        let (mut cli, temp_dir) = cli_with_unwritable_preferences("effort-preset-warning");

        let warning = cli.set_effort(Effort::Max);

        assert_eq!(
            cli.effort,
            Some(Effort::Max),
            "the effort change must apply in-memory even when it cannot be saved",
        );
        let warning = warning.expect("a failed persist must return a warning, not None");
        assert!(
            warning.starts_with("Warning          effort preference was not saved:"),
            "warning must match the model-switch wording: {warning}"
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// A numeric `/effort <n>` budget must behave identically: applied
    /// in-memory, with a persistence failure surfaced rather than swallowed.
    #[test]
    fn set_effort_budget_surfaces_persistence_failure_but_still_applies() {
        let (mut cli, temp_dir) = cli_with_unwritable_preferences("effort-budget-warning");

        let warning = cli.set_effort_budget(16000);

        assert_eq!(
            cli.thinking_budget,
            Some(16000),
            "the numeric budget must apply in-memory even when it cannot be saved",
        );
        let warning = warning.expect("a failed persist must return a warning, not None");
        assert!(
            warning.starts_with("Warning          effort preference was not saved:"),
            "warning must match the model-switch wording: {warning}"
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// The happy path returns `None`: a successful persist must not fabricate a
    /// warning, so callers only surface a message on genuine failure.
    #[test]
    fn set_effort_returns_no_warning_when_persistence_succeeds() {
        let (mut cli, temp_dir) = test_cli("effort-no-warning");

        assert_eq!(
            cli.set_effort(Effort::High),
            None,
            "a successful persist must not surface a spurious warning",
        );
        assert_eq!(
            cli.set_effort_budget(24000),
            None,
            "a successful numeric persist must not surface a spurious warning",
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// Serve/ACP quota parity, negative half: when routing yields no
    /// cross-provider peer (an explicit Smart-off fixture), a turn-entry install
    /// on the long-lived streaming runtime must CLEAR any fallback client a
    /// prior turn left installed, never leak it. Drives the exact helper the
    /// socket/ACP streaming path (`run_turn_streaming_to_channel_with_prompter`)
    /// calls.
    #[test]
    fn streaming_quota_install_clears_stale_fallback_when_route_is_none() {
        let (mut cli, temp_dir) = test_cli("stream-quota-clear");
        {
            let client = TurnHarness::build_live_client(
                &cli.runtime,
                cli.allowed_tools.clone(),
                cli.thinking_config(),
                None,
                None,
            );
            let runtime = cli.runtime.try_runtime_mut().expect("runtime");
            runtime.set_quota_fallback_client(Some((client, "openai:gpt-5.6-sol".to_string())));
            assert_eq!(
                cli.runtime
                    .try_runtime_mut()
                    .expect("runtime")
                    .quota_fallback_model(),
                Some("openai:gpt-5.6-sol"),
                "precondition: a stale fallback is installed",
            );
        }

        with_smart_settings(
            &temp_dir,
            r#"{"smart":{"enabled":false}}"#,
            || {
                assert_eq!(
                    super::super::smart_settings::route_quota_fallback_model(
                        cli.runtime.api_client().model(),
                    ),
                    None,
                    "precondition: explicit Smart-off settings must yield no route",
                );

                // Same helper the serve/ACP streaming turn path invokes each turn.
                cli.install_quota_fallback_client_on_self();

                assert_eq!(
                    cli.runtime
                        .try_runtime_mut()
                        .expect("runtime")
                        .quota_fallback_model(),
                    None,
                    "a None route on turn entry must clear the stale fallback, not leak it",
                );
            },
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// Serve/ACP quota parity, wait-band half: the streaming turn-entry install
    /// refreshes the quota-wait band from settings every turn (not just at
    /// construction), so a `/smart` edit takes effect next turn. An explicit
    /// non-default test band replaces a stale runtime value, proving the refresh
    /// ran and is NOT gated on a fallback client existing.
    #[test]
    fn streaming_quota_install_refreshes_wait_band() {
        let (mut cli, temp_dir) = test_cli("stream-quota-waitband");
        let stale = std::time::Duration::from_secs(999 * 60);
        let configured = std::time::Duration::from_secs(7 * 60);
        cli.runtime
            .try_runtime_mut()
            .expect("runtime")
            .set_quota_wait_band(stale);

        with_smart_settings(
            &temp_dir,
            r#"{"smart":{"enabled":false,"quotaWaitBandMinutes":7}}"#,
            || {
                cli.install_quota_fallback_client_on_self();

                let band = cli
                    .runtime
                    .try_runtime_mut()
                    .expect("runtime")
                    .quota_wait_band();
                assert_ne!(
                    band, stale,
                    "the stale wait band must be overwritten by the per-turn settings read",
                );
                assert_eq!(
                    band, configured,
                    "the refreshed band must match the explicit test setting",
                );
            },
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// The headless `&mut BuiltRuntime` install variant shares the same quota
    /// policy as the self variant: it too clears a stale fallback on a None
    /// route and refreshes the wait band, so all turn paths stay in parity.
    #[test]
    fn headless_quota_install_clears_stale_fallback_and_refreshes_band() {
        let (mut cli, temp_dir) = test_cli("headless-quota-parity");
        let (mut runtime, _monitor) = cli
            .prepare_turn_runtime(false)
            .expect("prepare headless runtime");
        {
            let client = TurnHarness::build_live_client(
                &runtime,
                cli.allowed_tools.clone(),
                cli.thinking_config(),
                None,
                None,
            );
            let inner = runtime.try_runtime_mut().expect("runtime");
            inner.set_quota_fallback_client(Some((client, "openai:gpt-5.6-sol".to_string())));
            inner.set_quota_wait_band(std::time::Duration::from_secs(999 * 60));
        }

        let configured = std::time::Duration::from_secs(11 * 60);
        with_smart_settings(
            &temp_dir,
            r#"{"smart":{"enabled":false,"quotaWaitBandMinutes":11}}"#,
            || {
                assert_eq!(
                    super::super::smart_settings::route_quota_fallback_model(
                        cli.runtime.api_client().model(),
                    ),
                    None,
                    "precondition: explicit Smart-off settings must yield no route",
                );

                cli.install_quota_fallback_client(&mut runtime);

                let inner = runtime.try_runtime_mut().expect("runtime");
                assert_eq!(
                    inner.quota_fallback_model(),
                    None,
                    "headless install must also clear a stale fallback on a None route",
                );
                assert_eq!(
                    inner.quota_wait_band(),
                    configured,
                    "headless install must also refresh the explicit settings-derived band",
                );
            },
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// Positive read path: a fallback installed on the runtime is reported by
    /// `quota_fallback_model()`, and refreshing only the wait band leaves it
    /// installed — the invariant the serve/ACP helper relies on when routing
    /// yields a peer. Kept tiny (no routing fixture): asserts the accessor the
    /// clear-on-None tests read against actually observes an installed client.
    #[test]
    fn installed_quota_fallback_is_reported_and_survives_wait_band_refresh() {
        let (mut cli, temp_dir) = test_cli("quota-fallback-positive");
        let client = TurnHarness::build_live_client(
            &cli.runtime,
            cli.allowed_tools.clone(),
            cli.thinking_config(),
            None,
            None,
        );
        let inner = cli.runtime.try_runtime_mut().expect("runtime");
        inner.set_quota_fallback_client(Some((client, "openai:gpt-5.6-sol".to_string())));

        assert_eq!(
            inner.quota_fallback_model(),
            Some("openai:gpt-5.6-sol"),
            "an installed fallback must be reported by the accessor",
        );

        inner.set_quota_wait_band(std::time::Duration::from_secs(42 * 60));
        assert_eq!(
            inner.quota_fallback_model(),
            Some("openai:gpt-5.6-sol"),
            "refreshing only the wait band must not clear an installed fallback",
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn active_model_survives_permission_runtime_rebuild_spawn_context() {
        let (mut cli, temp_dir) = test_cli("active-model-permission");
        let _ = cli.apply_model_change("claude-opus-4-8");
        assert_eq!(
            live_spawn_parent_model(&mut cli).as_deref(),
            Some("claude-opus-4-8")
        );

        cli.apply_permission_change("danger-full-access")
            .expect("permission switch should rebuild runtime");

        assert_eq!(
            cli.permission_mode,
            runtime::PermissionMode::DangerFullAccess
        );
        assert_eq!(cli.model.as_str(), "claude-opus-4-8");
        assert_eq!(
            live_spawn_parent_model(&mut cli).as_deref(),
            Some("claude-opus-4-8")
        );

        std::fs::remove_dir_all(temp_dir).ok();
    }
}
