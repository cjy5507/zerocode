use std::collections::BTreeSet;
use std::time::Duration;

use runtime::{PermissionMode, RouteOutcomeRecord, RuntimeHookConfig, lsp_client::LspRegistry};

use super::agent_runtime::{build_agent_runtime, subagent_hook_context};
use super::completion::{AgentCompletion, notify_agent_completion, provider_error_class_metadata};
use super::manifest::{
    manifest_generation_is_current, persist_agent_stopped_state, persist_agent_terminal_state,
    persist_agent_terminal_state_with_history,
};
use super::unregister_agent_cancel_signal;
use super::{AgentOutput, final_assistant_text, final_structured_output};
use crate::ToolError;
use crate::context::ProbeRevertGuard;

#[derive(Debug, Clone)]
pub(crate) struct AgentJob {
    pub manifest: AgentOutput,
    pub prompt: String,
    pub system_prompt: Vec<String>,
    pub allowed_tools: BTreeSet<String>,
    /// Per-agent permission rules from the custom-agent definition. `None` for
    /// built-in types, which keeps [`agent_permission_policy`] rule-free.
    /// Carried (like `allowed_tools`) across the spawn's OS-thread boundary.
    pub permission_rules: Option<runtime::RuntimePermissionRuleConfig>,
    /// Per-agent permission mode. `None` defaults to `DangerFullAccess` in
    /// [`build_agent_runtime`] - byte-identical to the pre-override behavior.
    pub permission_mode: Option<PermissionMode>,
    /// Working directory for this agent's tools (worktree isolation). `None`
    /// runs in the process cwd. Carried from [`AgentInput::cwd`] into
    /// [`build_agent_runtime`], which loads it onto the executor's context.
    pub cwd: Option<std::path::PathBuf>,
    /// Parent session's LSP registry (a cheap `Arc`-clone), shared into the
    /// sub-agent's context by [`build_agent_runtime`] only when `cwd` is `None`
    /// (the agent runs in the parent tree, so the parent's servers - rooted
    /// there - give correct diagnostics). A worktree-isolated agent
    /// (`cwd` `Some`) skips it, since its tree has no matching server. `None` on
    /// the workflow-engine and test paths. `LspRegistry` is all `Arc`-backed, so
    /// it crosses the spawn OS-thread boundary safely.
    pub lsp: Option<LspRegistry>,
    /// Structured-output schema (workflow 8c). When set, `build_agent_runtime`
    /// enables `StructuredOutput` and `run_agent_job` captures that tool call's
    /// input as the structured result. Carried from [`AgentInput::schema`].
    pub schema: Option<serde_json::Value>,
    /// Carried from [`AgentInput::workflow_member`]: selects the workflow API
    /// semaphore (higher cap) over the shared one in [`run_agent_job`].
    pub workflow_member: bool,
    /// Optional per-agent wall-clock budget. `None` leaves the spawned turn
    /// unbounded; callers that need a hard kill must pass an explicit budget.
    pub time_budget: Option<Duration>,
    /// Optional thinking budget chosen by the sub-agent model router. `None`
    /// keeps the provider default; GPT hard/deep routes set this so gpt-5.5
    /// receives the effort tier that matches the delegated task.
    pub thinking_budget_tokens: Option<u32>,
    /// Named reasoning-effort tier the Smart router recommends for this
    /// agent's route (carried from [`AgentInput::route_effort`]). `None` (the
    /// default for every pre-existing spawn path) keeps
    /// [`ProviderRuntimeClient`](super::provider_client::ProviderRuntimeClient)'s
    /// behavior byte-identical.
    pub route_effort: Option<api::EffortLevel>,
    /// Per-call provider-request concurrency ceiling, carried from the flat
    /// `SpawnMultiAgent` `concurrency` argument. Caps the adaptive governor's
    /// admission to `min(live_limit, this)` so a tighter user value actually
    /// throttles real API concurrency. `None` = governor ceiling only.
    pub api_concurrency: Option<usize>,
    /// Ranked host-computed fallback models to try when the selected model's
    /// provider is rate-limited or parked in a long cool-down.
    pub route_fallback_models: Vec<String>,
    /// Parent-session MCP passthrough carried from [`AgentInput`]: the
    /// sub-agent's client advertises these tool schemas and its executor
    /// routes their calls back through the parent session's MCP runtime.
    pub mcp_passthrough: Option<crate::registry::McpPassthrough>,
    /// Hook config inherited from the parent runtime so spawned agents honor
    /// SubagentStart/SubagentStop and regular tool hooks across the thread boundary.
    pub hook_config: RuntimeHookConfig,
    /// Cooperative cancel signal shared with the parent-side agent registry.
    /// Foreground Ctrl+C aborts it, and the sync conversation loop exits at the
    /// next safe boundary instead of running another model/tool iteration.
    pub cancel_signal: runtime::HookAbortSignal,
    /// The worker agent id this agent's route need judges (Phase 4 verdict
    /// channel — planner-bound reviewer→worker pairing), carried verbatim from
    /// [`super::AgentInput::judged_agent`]. `None` = no recognized judged
    /// worker (see that field's doc for the exact absence conditions).
    pub judged_agent: Option<String>,
    /// The PARENT/session model this agent was spawned FROM (Phase 4 verdict
    /// channel — ad-hoc standalone review, source #3): the model whose route
    /// an ad-hoc reviewer's verdict is credited to when it judges the current
    /// turn's work rather than a specific sibling worker. `None` when the
    /// caller passed no parent model (e.g. a bare test harness) — a verdict
    /// recorder must skip recording rather than guess.
    pub parent_model: Option<String>,
    /// Mid-turn steering queue shared with the parent-side registry
    /// (`SendMessage` delivery). Created and registered at spawn time — before
    /// the detached thread exists — and installed into the sub-agent's runtime
    /// via `with_steering_queue`, so a send can never race the runtime build.
    pub steering: runtime::SteeringQueue,
    /// On-disk JSONL transcript for this agent's conversation
    /// (`<store>/<id>.session.jsonl`). Written incrementally while the turn
    /// runs and snapshotted in full when it ends, so a terminal agent can be
    /// resumed by `SendMessage` with its context intact. `None` only on bare
    /// test harnesses.
    pub transcript_path: Option<std::path::PathBuf>,
    /// Rehydrate the session from `transcript_path` instead of starting fresh
    /// — the `SendMessage` resume path. A missing/unreadable transcript fails
    /// the job rather than silently continuing without the prior context.
    pub resume: bool,
}

/// Per-turn cap on model turns for a spawned sub-agent. Reaching it no longer
/// fails a healthy long-running task immediately: [`run_agent_job`] grants a
/// small number of fresh continuation turns when the exhausted turn produced a
/// successful tool result. The per-turn cap still catches a locally stuck loop.
const SPAWNED_AGENT_MAX_ITERATIONS: usize = 64;

/// Maximum fresh turns granted after a recoverable budget cutoff. Two windows
/// let a legitimate implementation finish while bounding total provider spend;
/// hard-stop budgets (deadline, tool calls, verification treadmill) never earn
/// a continuation.
const SPAWNED_AGENT_MAX_BUDGET_CONTINUATIONS: u8 = 2;

/// Read-only agents cannot produce edit/plan events, so a substantial partial
/// deliverable can also earn a continuation. A short status sentence does not.
const SPAWNED_AGENT_MIN_CONTINUATION_TEXT_CHARS: usize = 256;

/// Hard cap on total model-requested tool calls for a spawned sub-agent. One
/// iteration can issue many parallel `tool_use` blocks, so the iteration cap
/// above does **not** bound tool-call volume on its own — the runaway that
/// prompted this reached 174 tool calls (2026-06-07). This complements
/// [`SPAWNED_AGENT_MAX_ITERATIONS`]: `256` is generous for a focused delegated
/// task while still hard-stopping a model that bursts tool calls every turn.
/// Enforcement and the graceful `failed` completion live in the conversation
/// loop (`check_tool_call_budget` → `record_turn_failed`).
const SPAWNED_AGENT_MAX_TOOL_CALLS: usize = 256;

/// Spawn one sub-agent on its own OS thread and return immediately (the thread is
/// detached; completion is reported via [`notify_agent_completion`]).
///
/// **OS-thread bounding lives at the call site, not here.** This spawns a real
/// thread eagerly per agent; the API semaphore acquired *inside* the turn loop
/// (`workflow_api_semaphore` / `agent_api_semaphore`) bounds concurrent provider
/// streams, not thread count. The workflow engine bounds live threads by spawning
/// in windows of `workflow_concurrency_limit()` and awaiting each batch before
/// the next (`engine::spawn_and_collect`), so a multi-phase workflow stays within
/// the cap *except* for stragglers that overran the phase timeout. The flat
/// `SpawnMultiAgent` fan-out has no such window - its cap is its own concurrency
/// setting (default 1). A true permit-before-spawn OS-thread gate would belong
/// here, but a blocking acquire risks wedging the engine behind a non-terminating
/// straggler, so it is deliberately not added; see `engine.rs` for the windowing
/// contract this relies on.
struct AgentWorkerRegistrationGuard {
    agent_id: String,
    run_generation: u64,
}

impl Drop for AgentWorkerRegistrationGuard {
    fn drop(&mut self) {
        unregister_agent_cancel_signal(&self.agent_id, self.run_generation);
        super::unregister_agent_steering(&self.agent_id, self.run_generation);
    }
}

pub(super) fn spawn_agent_job(job: AgentJob) -> Result<(), ToolError> {
    let thread_name = format!("zo-agent-{}", job.manifest.agent_id);
    std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let agent_id = job.manifest.agent_id.clone();
            let agent_name = job.manifest.name.clone();
            let _worker_registration = AgentWorkerRegistrationGuard {
                agent_id: agent_id.clone(),
                run_generation: job.manifest.run_generation,
            };
            // Owned here (not inside `run_agent_job`) so the budget total is
            // readable on *every* terminal branch — including a returned error or
            // an unwinding panic, where the inner function never hands anything
            // back. The provider client accumulates into it (and into the display
            // sparkline) through the clones passed into the runtime.
            let token_history: std::sync::Arc<std::sync::Mutex<Vec<u32>>> =
                std::sync::Arc::default();
            // Never-lossy output-token total (the workflow budget source),
            // separate from the display-capped `token_history`.
            let output_tokens_total: std::sync::Arc<std::sync::atomic::AtomicU64> =
                std::sync::Arc::default();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::manifest::with_agent_run_generation(job.manifest.run_generation, || {
                    run_agent_job(&job, &token_history, &output_tokens_total)
                })
            }));
            // Read after the turn fully returns (or unwinds) — every `fetch_add`
            // happens-before this load, so the total is exact.
            let output_tokens = output_tokens_total.load(std::sync::atomic::Ordering::Relaxed);
            match result {
                Ok(Ok(outcome)) => {
                    let structured = completion_structured_with_provider_error_class(
                        outcome.structured,
                        outcome.provider_error_class,
                    );
                    // `run_agent_job` already persisted the terminal manifest
                    // (completed, or failed-with-partial-work on a budget cutoff)
                    // and fired `SubagentStop`; here we only surface the in-memory
                    // completion, carrying whichever terminal status it settled on.
                    notify_agent_completion_with_route_outcome(
                        &job,
                        AgentCompletion {
                            agent_id: agent_id.clone(),
                            name: agent_name,
                            status: String::from(outcome.status),
                            result: Some(outcome.final_text),
                            structured,
                            error: outcome.error,
                            output_tokens,
                        },
                    );
                }
                Ok(Err(error)) => {
                    let provider_error_class = error.provider_error_class();
                    let error = error.to_string();
                    let status = if agent_error_is_cancelled(&error) {
                        let _ = persist_agent_stopped_state(&job.manifest, error.as_str());
                        "stopped"
                    } else {
                        let _ = persist_agent_terminal_state(
                            &job.manifest,
                            "failed",
                            None,
                            Some(error.clone()),
                        );
                        "failed"
                    };
                    notify_agent_completion_with_route_outcome(
                        &job,
                        AgentCompletion {
                            agent_id: agent_id.clone(),
                            name: agent_name,
                            status: status.to_string(),
                            result: None,
                            structured: provider_error_class.map(provider_error_class_metadata),
                            error: Some(error),
                            output_tokens,
                        },
                    );
                }
                Err(_) => {
                    let panic_msg = String::from("sub-agent thread panicked");
                    let _ = persist_agent_terminal_state(
                        &job.manifest,
                        "failed",
                        None,
                        Some(panic_msg.clone()),
                    );
                    notify_agent_completion_with_route_outcome(
                        &job,
                        AgentCompletion {
                            agent_id: agent_id.clone(),
                            name: agent_name,
                            status: String::from("failed"),
                            result: None,
                            structured: None,
                            error: Some(panic_msg),
                            output_tokens,
                        },
                    );
                }
            }
        })
        .map(|_| ())
        .map_err(|error| ToolError::Execution(error.to_string()))
}

fn reconcile_completion_terminal_status(
    completion: &mut AgentCompletion,
    status: &str,
    persisted_error: Option<&str>,
) {
    if !super::agent_output_status_is_terminal(status) || completion.status == status {
        return;
    }
    completion.status = status.to_string();
    if completion.error.is_none() {
        completion.error = persisted_error
            .map(str::to_string)
            .or_else(|| (status == "stopped").then(|| "agent stopped".to_string()));
    }
}

fn reconcile_completion_with_manifest(manifest: &AgentOutput, completion: &mut AgentCompletion) {
    let Some(stored) = super::manifest::load_agent_manifest_from_scanned_path(
        std::path::Path::new(&manifest.manifest_file),
    )
    .ok()
    else {
        return;
    };
    reconcile_completion_terminal_status(completion, &stored.status, stored.error.as_deref());
}

fn notify_agent_completion_with_route_outcome(job: &AgentJob, mut completion: AgentCompletion) {
    if !manifest_generation_is_current(&job.manifest) {
        return;
    }
    // An external stop can win the durable terminal transition while a provider
    // stream is already returning. Keep the richer worker payload, but the
    // user-facing status and route attribution must reflect the durable winner.
    reconcile_completion_with_manifest(&job.manifest, &mut completion);
    notify_agent_completion(completion.clone(), Some(job.manifest.run_generation));
    record_agent_route_outcome(job, &completion);
    record_agent_verdict_outcome(job, &completion);
}

/// Pure pass/fail projection of `semantic_verdict`'s label. `None` for
/// anything ambiguous (the `"retry"` label — status completed but no usable
/// structured verdict recovered, e.g. missing/malformed `StructuredOutput` —
/// or any other unrecognized label) — never a guess, per the shared
/// verdict-attribution doctrine.
fn verdict_passed(label: &str) -> Option<bool> {
    match label {
        "pass" => Some(true),
        "finding" => Some(false),
        _ => None,
    }
}

/// Phase 4 verdict channel — sources #2 and #3: a completed verifier/reviewer
/// agent's own structured verdict, folded back onto whatever it judged.
/// Reuses the workflow engine's EXISTING verdict recorder and
/// structured-verdict classifier verbatim (`workflow_tools::engine::
/// attribution`/`items::semantic_verdict`, already `pub(crate)` for this)
/// rather than a second parser — one classifier, multiple callers. Silent on
/// anything ambiguous: provenance over volume.
fn record_agent_verdict_outcome(job: &AgentJob, completion: &AgentCompletion) {
    if completion.status != "completed" {
        return;
    }
    let Some(verdict) = crate::workflow_tools::engine::items::semantic_verdict(
        &completion.status,
        completion.structured.as_ref(),
    ) else {
        return;
    };
    let Some(passed) = verdict_passed(verdict) else {
        return;
    };

    // Source #2: planner-bound reviewer→worker — an exact, host-resolved
    // binding to the JUDGED worker's own route (see
    // `AgentInput::judged_agent`'s doc for how the binding is established and
    // its exact absence conditions).
    if let Some(worker_agent_id) = job.judged_agent.as_deref() {
        crate::workflow_tools::engine::attribution::record_verdict_outcome_for_agent(
            worker_agent_id,
            passed,
            crate::workflow_tools::engine::attribution::VerdictKind::PassFail,
        );
        return;
    }

    // Source #3: an ad-hoc standalone review/verification agent judging the
    // PARENT turn's own work — conservative whitelist only (see
    // `is_ad_hoc_turn_review`), never a guess when the target is unclear.
    if !is_ad_hoc_turn_review(job) {
        return;
    }
    let Some(parent_model) = job.parent_model.as_deref() else {
        // No parent model available at this seam (e.g. a bare test harness,
        // or a caller that never threaded one through) — skip rather than
        // credit an unknown route.
        return;
    };
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let record = RouteOutcomeRecord::new(
        "main",
        "turn",
        crate::misc_tools::canonicalize_route_model_id(parent_model),
        if passed { "completed" } else { "failed" },
    )
    .with_signal("verdict")
    // Strict pass/fail judgement — same convention/weight as
    // `workflow_tools::engine::attribution::VerdictKind::PassFail`.
    .with_signal_weight(Some(1.0));
    let _ = runtime::record_route_outcome(&cwd, &record);
}

/// Substring markers a review/verification prompt uses to name the CURRENT
/// turn's own diff/working tree (Phase 4 verdict channel source #3's
/// whitelist). Faithful to this repo's own review prompts — mirrors
/// `workflow_tools::presets`'s `cross_model_verified` preset ("review the
/// current working tree", "Inspect the current diff") — plus their Korean
/// equivalents (the classifier keyword tables' existing parity convention).
const CURRENT_TURN_REVIEW_MARKERS: &[&str] = &[
    "current diff",
    "this diff",
    "working tree",
    "uncommitted",
    "이 diff",
    "현재 diff",
    "워킹트리",
    "미커밋",
];

/// Conservative whitelist for Phase 4 verdict channel source #3 (ad-hoc
/// standalone review): a bare reviewer/verifier spawn — no judged-agent
/// binding — is credited to the PARENT turn ONLY when it is unmistakably
/// reviewing the turn's OWN current work, never a delegated/unrelated task.
/// Requires BOTH:
/// - the resolved built-in type is the reviewer/verification harness
///   (`"code-reviewer"` or `"Verification"` — the SAME canonical strings
///   `route_outcome_target` already keys run-level outcomes by), and
/// - the prompt explicitly names the current turn's diff/working tree
///   (`CURRENT_TURN_REVIEW_MARKERS`).
///
/// Any other subagent type, or a prompt that never names the CURRENT
/// diff/tree, returns `false` — this fn only recognizes what it's certain of;
/// a delegated review of some OTHER (unrelated) diff/file must never be
/// credited to the main turn.
fn is_ad_hoc_turn_review(job: &AgentJob) -> bool {
    let subagent_type = job.manifest.subagent_type.as_deref().unwrap_or_default();
    if !matches!(subagent_type, "code-reviewer" | "Verification") {
        return false;
    }
    let haystack = job.prompt.to_ascii_lowercase();
    CURRENT_TURN_REVIEW_MARKERS
        .iter()
        .any(|marker| haystack.contains(marker))
}

fn record_agent_route_outcome(job: &AgentJob, completion: &AgentCompletion) {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let manifest = current_on_disk_manifest_or_spawn_time(&job.manifest);
    let record = route_outcome_record(&manifest, completion, job.route_effort);
    let _ = runtime::record_route_outcome(&cwd, &record);
}

/// Prefer the manifest's CURRENT on-disk state over the spawn-time in-memory
/// copy captured in `job.manifest` (**verified misattribution fix**): a
/// mid-run rate-limit/starvation swap (`record_agent_runtime_model`,
/// `manifest.rs:~301`, triggered by `provider_client.rs`'s
/// `switch_runtime_model`) updates ONLY the on-disk manifest file's
/// `resolvedModel`/`model` — the in-memory `job.manifest` this recorder used
/// to read verbatim never observes it, so a quota-pressure run credited the
/// model the agent had already swapped AWAY FROM. By the time this fires, the
/// terminal `persist_agent_*_state` call has already written the final
/// status to the SAME manifest file, so a fresh read here picks up both the
/// final model AND the terminal timestamps. Falls back to the in-memory
/// manifest verbatim on any read/parse failure (e.g. a test harness that
/// never wrote a real manifest file, or the file already having been cleaned
/// up) — never a hard failure.
fn current_on_disk_manifest_or_spawn_time(spawn_time_manifest: &AgentOutput) -> AgentOutput {
    super::manifest::load_agent_manifest_from_scanned_path(std::path::Path::new(
        &spawn_time_manifest.manifest_file,
    ))
    .unwrap_or_else(|_| spawn_time_manifest.clone())
}

fn route_outcome_record(
    manifest: &AgentOutput,
    completion: &AgentCompletion,
    route_effort: Option<api::EffortLevel>,
) -> RouteOutcomeRecord {
    RouteOutcomeRecord::new(
        "subagent",
        route_outcome_target(manifest),
        // P3 canonicalization-at-write: new records are keyed by the
        // canonical model id from the start, so they never need a read-time
        // merge (`claude-opus-4-8` vs a hand-typed `claude-opus-4.8`, etc.).
        crate::misc_tools::canonicalize_route_model_id(&selected_route_model(manifest)),
        completion_status(completion),
    )
    .with_requested_model(requested_route_model(manifest))
    .with_provider_error_class(provider_error_class_label(completion))
    .with_output_tokens(completion.output_tokens)
    .with_role(manifest.route_role.clone())
    .with_complexity(manifest.route_complexity.clone())
    .with_risk(manifest.route_risk.clone())
    .with_route_source(manifest.route_source.clone())
    .with_effort_level(effort_level_label(route_effort))
    .with_duration_ms(run_duration_ms(manifest))
}

/// `EffortLevel`'s lowercase wire token (`#[serde(rename_all = "lowercase")]`)
/// — the same projection `apply.rs` uses to smuggle it, reused here instead
/// of a parallel token table. `route_effort` is the Smart router's
/// RECOMMENDATION for this route (`AgentJob::route_effort`), not necessarily
/// the effort the provider request ultimately used (a starvation/budget
/// clamp can still lower it) — the closest available signal without
/// threading the provider client's actual per-request effort back through
/// the job boundary.
fn effort_level_label(route_effort: Option<api::EffortLevel>) -> Option<String> {
    route_effort
        .and_then(|effort| serde_json::to_value(effort).ok())
        .and_then(|value| value.as_str().map(str::to_string))
}

/// Best-effort wall-clock run duration from the manifest's own
/// `startedAt`/`completedAt` (epoch-second strings) — `None` on any missing/
/// unparsable timestamp rather than a guess.
fn run_duration_ms(manifest: &AgentOutput) -> Option<u64> {
    let started: u64 = manifest.started_at.as_deref()?.parse().ok()?;
    let completed: u64 = manifest.completed_at.as_deref()?.parse().ok()?;
    Some(completed.saturating_sub(started).saturating_mul(1000))
}

fn route_outcome_target(manifest: &AgentOutput) -> String {
    non_empty_owned(manifest.subagent_type.as_deref())
        .unwrap_or_else(|| "general-purpose".to_string())
}

fn selected_route_model(manifest: &AgentOutput) -> String {
    non_empty_owned(manifest.resolved_model.as_deref())
        .or_else(|| non_empty_owned(manifest.model.as_deref()))
        .unwrap_or_else(|| "unknown".to_string())
}

fn requested_route_model(manifest: &AgentOutput) -> Option<String> {
    non_empty_owned(manifest.requested_model.as_deref())
}

fn completion_status(completion: &AgentCompletion) -> String {
    completion.status.clone()
}

fn provider_error_class_label(completion: &AgentCompletion) -> Option<String> {
    completion
        .structured
        .as_ref()?
        .get("providerErrorClass")?
        .as_str()
        .and_then(|value| non_empty_owned(Some(value)))
}

fn non_empty_owned(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
}

#[cfg(test)]
pub(super) fn route_outcome_record_for_tests(
    manifest: &AgentOutput,
    completion: &AgentCompletion,
) -> RouteOutcomeRecord {
    route_outcome_record(manifest, completion, None)
}

#[cfg(test)]
pub(super) fn current_on_disk_manifest_or_spawn_time_for_tests(
    spawn_time_manifest: &AgentOutput,
) -> AgentOutput {
    current_on_disk_manifest_or_spawn_time(spawn_time_manifest)
}

/// A sub-agent turn's terminal outcome, handed back to [`spawn_agent_job`] for
/// notification. `status` is `"completed"` for a natural end and `"failed"` when
/// the turn stopped on an exhausted budget — in which case `final_text` still
/// carries the preserved partial work (the runtime no longer discards it), so
/// the parent and the transcript viewer both see what got done.
struct AgentJobOutcome {
    final_text: String,
    structured: Option<serde_json::Value>,
    status: &'static str,
    /// Why a non-`completed` outcome failed (e.g. the budget-exhausted kind).
    /// Carried into the `AgentCompletion` so the parent's notice names the
    /// real cause instead of fabricating "unknown error" for an agent that
    /// visibly returned a (partial) result.
    error: Option<String>,
    /// Provider classification survives a failed continuation so route health
    /// does not misclassify quota/transport failures as model-quality failures.
    provider_error_class: Option<api::ProviderErrorClass>,
}

fn agent_budget_can_auto_continue(kind: runtime::BudgetExhausted) -> bool {
    // A sub-agent continuation is a fresh model-loop window, not a fresh cost
    // budget. Only the local iteration guard is recoverable; token, deadline,
    // tool-call, and verification-treadmill cutoffs remain hard stops.
    kind == runtime::BudgetExhausted::Iterations
}

fn subagent_allows_text_progress(
    subagent_type: &str,
    permission_mode: Option<runtime::PermissionMode>,
) -> bool {
    permission_mode == Some(runtime::PermissionMode::ReadOnly)
        || matches!(
            subagent_type,
            "Explore"
                | "Plan"
                | "Verification"
                | "deep-research"
                | "code-reviewer"
                | "data-analyst"
        )
}

fn continuation_text_is_meaningful(text: &str) -> bool {
    text.chars().filter(|ch| !ch.is_whitespace()).count()
        >= SPAWNED_AGENT_MIN_CONTINUATION_TEXT_CHARS
}

fn agent_turn_has_successful_tool_result(summary: &runtime::TurnSummary) -> bool {
    summary
        .tool_results
        .iter()
        .flat_map(|message| &message.blocks)
        .any(|block| {
            matches!(
                block,
                runtime::session::ContentBlock::ToolResult {
                    is_error: false,
                    ..
                }
            )
        })
}

fn agent_turn_made_progress(
    summary: &runtime::TurnSummary,
    allow_text_progress: bool,
) -> bool {
    summary.progress_tool_results() > 0
        || (allow_text_progress
            && agent_turn_has_successful_tool_result(summary)
            && continuation_text_is_meaningful(&last_substantive_assistant_text(
                &summary.assistant_messages,
            )))
}

fn agent_turn_tool_calls(summary: &runtime::TurnSummary) -> usize {
    summary
        .assistant_messages
        .iter()
        .flat_map(|message| &message.blocks)
        .filter(|block| matches!(block, runtime::session::ContentBlock::ToolUse { .. }))
        .count()
}

#[derive(Default)]
struct AgentContinuationState {
    continuations_used: u8,
    cumulative_iterations: usize,
    cumulative_tool_calls: usize,
    cumulative_output_tokens: u32,
    cumulative_input_tokens: u32,
    latest_structured: Option<serde_json::Value>,
    latest_budget_text: String,
    latest_budget_result: String,
}

impl AgentContinuationState {
    fn observe(
        &mut self,
        summary: &runtime::TurnSummary,
        schema: Option<&serde_json::Value>,
    ) {
        self.cumulative_iterations = self
            .cumulative_iterations
            .saturating_add(summary.iterations);
        self.cumulative_tool_calls = self
            .cumulative_tool_calls
            .saturating_add(agent_turn_tool_calls(summary));
        self.cumulative_output_tokens = self
            .cumulative_output_tokens
            .saturating_add(summary.turn_output_tokens);
        // `usage.input_tokens` is cumulative for this runtime. Keep the high
        // water mark so compaction/accounting resets cannot create fake budget.
        self.cumulative_input_tokens = self
            .cumulative_input_tokens
            .max(summary.usage.input_tokens);
        if let Some(schema) = schema {
            if let Some(structured) = final_structured_output(summary, schema) {
                self.latest_structured = Some(structured);
            }
        }
        if let Some(kind) = summary.budget_exhausted {
            let text = last_substantive_assistant_text(&summary.assistant_messages);
            if !text.trim().is_empty() {
                // Preserve the latest checkpoint, even when it is concise. A
                // later short conclusion is more current than an older essay.
                self.latest_budget_text = text;
            }
            self.latest_budget_result = agent_budget_exhausted_result(
                &self.latest_budget_text,
                kind,
                self.cumulative_iterations,
                self.cumulative_tool_calls,
            );
        }
    }

    fn remaining_tool_calls(&self) -> usize {
        SPAWNED_AGENT_MAX_TOOL_CALLS.saturating_sub(self.cumulative_tool_calls)
    }

    fn remaining_output_tokens(&self, budget: Option<u32>) -> Option<u32> {
        budget.map(|budget| budget.saturating_sub(self.cumulative_output_tokens))
    }

    fn remaining_input_tokens(&self, budget: Option<u32>) -> Option<u32> {
        budget.map(|budget| budget.saturating_sub(self.cumulative_input_tokens))
    }

    fn token_budgets_have_headroom(
        &self,
        output_budget: Option<u32>,
        input_budget: Option<u32>,
    ) -> bool {
        output_budget.is_none_or(|budget| self.cumulative_output_tokens < budget)
            && input_budget.is_none_or(|budget| self.cumulative_input_tokens < budget)
    }
}

fn continuation_error_result(
    continuation: &AgentContinuationState,
    error: &str,
) -> Option<String> {
    let partial = if !continuation.latest_budget_text.trim().is_empty() {
        continuation.latest_budget_text.clone()
    } else if continuation.latest_structured.is_some() {
        "The agent returned a partial result via StructuredOutput.".to_string()
    } else if !continuation.latest_budget_result.trim().is_empty() {
        continuation.latest_budget_result.clone()
    } else {
        return None;
    };
    Some(format!(
        "{partial}\n\n[zo:auto-continue] The continuation failed: {error}"
    ))
}

fn completion_structured_with_provider_error_class(
    structured: Option<serde_json::Value>,
    provider_error_class: Option<api::ProviderErrorClass>,
) -> Option<serde_json::Value> {
    let Some(provider_error_class) = provider_error_class else {
        return structured;
    };
    let serde_json::Value::Object(mut metadata) =
        provider_error_class_metadata(provider_error_class)
    else {
        return structured;
    };
    match structured {
        None => Some(serde_json::Value::Object(metadata)),
        Some(serde_json::Value::Object(mut result)) => {
            result.extend(metadata);
            Some(serde_json::Value::Object(result))
        }
        Some(result) => {
            metadata.insert("structuredResult".to_string(), result);
            Some(serde_json::Value::Object(metadata))
        }
    }
}

fn should_auto_continue_budget(
    kind: Option<runtime::BudgetExhausted>,
    made_progress: bool,
    continuations_used: u8,
    remaining_tool_calls: usize,
    token_budgets_have_headroom: bool,
) -> bool {
    continuations_used < SPAWNED_AGENT_MAX_BUDGET_CONTINUATIONS
        && remaining_tool_calls > 0
        && token_budgets_have_headroom
        && kind.is_some_and(agent_budget_can_auto_continue)
        && made_progress
}

fn should_auto_continue_agent(
    summary: &runtime::TurnSummary,
    schema: Option<&serde_json::Value>,
    continuations_used: u8,
    remaining_tool_calls: usize,
    token_budgets_have_headroom: bool,
    allow_text_progress: bool,
) -> bool {
    // A schema agent that already submitted its deliverable is done; a schema
    // agent cut off before `StructuredOutput` is eligible like any other agent.
    schema.is_none_or(|schema| final_structured_output(summary, schema).is_none())
        && should_auto_continue_budget(
            summary.budget_exhausted,
            agent_turn_made_progress(summary, allow_text_progress),
            continuations_used,
            remaining_tool_calls,
            token_budgets_have_headroom,
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentTerminalDisposition {
    Completed,
    BudgetFailed(runtime::BudgetExhausted),
}

fn agent_terminal_disposition(
    summary: &runtime::TurnSummary,
    schema_requested: bool,
    has_structured: bool,
) -> AgentTerminalDisposition {
    match summary.budget_exhausted {
        Some(runtime::BudgetExhausted::Iterations) if schema_requested && has_structured => {
            AgentTerminalDisposition::Completed
        }
        Some(kind) => AgentTerminalDisposition::BudgetFailed(kind),
        None => AgentTerminalDisposition::Completed,
    }
}

fn agent_budget_continuation_prompt(
    kind: runtime::BudgetExhausted,
    continuation: u8,
) -> String {
    format!(
        "[zo:auto-continue] The previous sub-agent turn reached its {} after making progress. \
         Continue the same task from the preserved conversation and finish the requested work and \
         verification now. Do not restart or merely summarize. Continuation {continuation}/{}.",
        budget_exhausted_kind_label(kind),
        SPAWNED_AGENT_MAX_BUDGET_CONTINUATIONS
    )
}

// Cohesive terminal-settling core: build runtime → run turn → settle one of
// three outcomes (error, budget-exhausted-with-work, completed). The budget arm
// mirrors the completed arm's persist + `SubagentStop`, so keeping them in one
// scope reads more clearly than a helper that would need the generic runtime
// type threaded through it.
#[allow(clippy::too_many_lines)]
fn run_agent_job(
    job: &AgentJob,
    token_history: &std::sync::Arc<std::sync::Mutex<Vec<u32>>>,
    output_tokens_total: &std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> Result<AgentJobOutcome, runtime::RuntimeError> {
    let mut runtime = build_agent_runtime(job, token_history.clone(), output_tokens_total.clone())
        .map_err(runtime::RuntimeError::new)?
        .with_max_iterations(SPAWNED_AGENT_MAX_ITERATIONS)
        .with_max_tool_calls(SPAWNED_AGENT_MAX_TOOL_CALLS)
        .with_hook_abort_signal(job.cancel_signal.clone());
    // CC parity: hooks firing inside this sub-agent carry agent_id/agent_type.
    runtime.set_hook_agent_context(
        job.manifest.agent_id.clone(),
        job.manifest
            .subagent_type
            .clone()
            .unwrap_or_else(|| "general-purpose".to_string()),
    );
    // Stop(TurnEnd) is a main-agent contract; the `for_subagent()` hook view
    // already strips those rules — this is defense-in-depth so no future hook
    // source can re-loop a sub-agent's narrow task.
    runtime.set_max_stop_loops(0);
    runtime.fire_lifecycle_hook(
        runtime::HookEvent::SubagentStart,
        &subagent_hook_context(job, "running", None, None),
    );
    // Only explicit budgets are hard deadlines. Workflow phase agents pass
    // `None` because the user needs their real result instead of a synthetic
    // "agent exceeded its time budget" failure.
    if let Some(deadline) = agent_deadline(std::time::Instant::now(), job.time_budget) {
        runtime.set_deadline(deadline);
    }
    // Cost circuit breakers (output/input tokens), same env-driven defaults as
    // every other turn host. Sub-agents bill on their own runtimes, so without
    // these a parent could route unbounded generation — or a cache-dead
    // full-transcript re-bill loop — through its spawns and bypass its own
    // breakers entirely. The iteration/tool-call caps above bound *count*, not
    // token cost; a few iterations can each carry a six-figure token bill.
    let (_, output_budget, input_budget) = runtime::env_turn_budgets();
    runtime.set_turn_output_token_budget(output_budget);
    runtime.set_turn_input_token_budget(input_budget);
    // Debug mode: revert any `InstrumentLog` probes so a debugger sub-agent's
    // tracing never leaks into the working tree. The explicit call below covers
    // the normal success/returned-error paths; the guard - built from a clone of
    // the sink before the turn - covers the PANIC path, where an unwind inside
    // `run_turn` would otherwise skip the explicit call and drop the runtime
    // (and its sink) with markers still on disk. A no-op for agents that never
    // instrumented (the ledger is empty).
    let probe_guard = ProbeRevertGuard::new(runtime.tool_executor().probe_sink_handle());
    let mut next_prompt = job.prompt.clone();
    let schema_requested = job.schema.is_some();
    let allow_text_progress = subagent_allows_text_progress(
        job.manifest.subagent_type.as_deref().unwrap_or("general-purpose"),
        job.permission_mode,
    );
    let mut continuation = AgentContinuationState::default();
    let summary_result = loop {
        let result = runtime.run_turn(next_prompt, None);
        runtime.tool_executor().revert_probes();
        match result {
            Ok(summary) => {
                continuation.observe(&summary, job.schema.as_ref());
                let remaining_tool_calls = continuation.remaining_tool_calls();
                let token_budgets_have_headroom =
                    continuation.token_budgets_have_headroom(output_budget, input_budget);
                if should_auto_continue_agent(
                    &summary,
                    job.schema.as_ref(),
                    continuation.continuations_used,
                    remaining_tool_calls,
                    token_budgets_have_headroom,
                    allow_text_progress,
                ) {
                    continuation.continuations_used += 1;
                    let kind = summary
                        .budget_exhausted
                        .expect("continuation policy requires an exhausted budget");
                    runtime.set_max_tool_calls(remaining_tool_calls);
                    runtime.set_turn_output_token_budget(
                        continuation.remaining_output_tokens(output_budget),
                    );
                    runtime.set_turn_input_token_budget(
                        continuation.remaining_input_tokens(input_budget),
                    );
                    next_prompt =
                        agent_budget_continuation_prompt(kind, continuation.continuations_used);
                    continue;
                }
                break Ok(summary);
            }
            Err(error) => break Err(error),
        }
    };
    // Free any write leases this agent acquired (track 4-2) so the paths it
    // edited are immediately available to the next sequential agent instead of
    // waiting out the lease TTL. No-op unless the guard was opt-in enabled.
    runtime.tool_executor().release_write_leases();
    drop(probe_guard);
    // Full-snapshot the conversation for `SendMessage` resume, on EVERY exit
    // (completed, budget-cut, returned error). `push_message`'s incremental
    // append misses direct message mutations (steer folds, compaction
    // rewrites), so the terminal snapshot is the ground truth a resume
    // rehydrates. Best-effort: a failed write only degrades resumability.
    if let Some(path) = job.transcript_path.as_ref() {
        let _ = runtime.session().save_to_path(path);
    }
    let summary = match summary_result {
        Ok(summary) => summary,
        Err(error) => {
            let provider_error_class = error.provider_error_class();
            let message = error.to_string();
            let status = if agent_error_is_cancelled(&message) {
                "stopped"
            } else {
                "failed"
            };
            if let Some(final_text) = continuation_error_result(&continuation, &message) {
                let structured = continuation.latest_structured.clone();
                let history_snapshot = token_history
                    .lock()
                    .map(|guard| guard.clone())
                    .unwrap_or_default();
                if let Err(persist_error) = persist_agent_terminal_state_with_history(
                    &job.manifest,
                    status,
                    Some(final_text.as_str()),
                    Some(message.clone()),
                    history_snapshot,
                ) {
                    runtime.fire_lifecycle_hook(
                        runtime::HookEvent::SubagentStop,
                        &subagent_hook_context(
                            job,
                            "failed",
                            None,
                            Some(persist_error.as_str()),
                        ),
                    );
                    return Err(runtime::RuntimeError::new(persist_error));
                }
                runtime.fire_lifecycle_hook(
                    runtime::HookEvent::SubagentStop,
                    &subagent_hook_context(
                        job,
                        status,
                        Some(final_text.as_str()),
                        Some(message.as_str()),
                    ),
                );
                return Ok(AgentJobOutcome {
                    final_text,
                    structured,
                    status,
                    error: Some(message),
                    provider_error_class,
                });
            }
            runtime.fire_lifecycle_hook(
                runtime::HookEvent::SubagentStop,
                &subagent_hook_context(job, status, None, Some(message.as_str())),
            );
            return Err(error);
        }
    };
    // Budget exhausted mid-task: the runtime now preserves the turn's work and
    // returns Ok with this marker instead of erroring, so a cut-off agent no
    // longer vaporizes everything it did. Surface the partial result and mark
    // the agent `failed` WITH that result (not a silent `None`) — the parent and
    // the transcript viewer both see what got done, and a follow-up can continue
    // or narrow the task. Persistence + `SubagentStop` mirror the completed path
    // below, differing only in the status and the budget-banner text.
    let structured = continuation.latest_structured.clone();
    if let AgentTerminalDisposition::BudgetFailed(kind) =
        agent_terminal_disposition(&summary, schema_requested, structured.is_some())
    {
        let final_text = agent_budget_exhausted_result(
            &continuation.latest_budget_text,
            kind,
            continuation.cumulative_iterations,
            continuation.cumulative_tool_calls,
        );
        let budget_error = format!(
            "budget exhausted: {} — partial result preserved",
            budget_exhausted_kind_label(kind)
        );
        let history_snapshot = token_history
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        if let Err(error) = persist_agent_terminal_state_with_history(
            &job.manifest,
            "failed",
            Some(final_text.as_str()),
            Some(budget_error.clone()),
            history_snapshot,
        ) {
            runtime.fire_lifecycle_hook(
                runtime::HookEvent::SubagentStop,
                &subagent_hook_context(job, "failed", None, Some(error.as_str())),
            );
            return Err(runtime::RuntimeError::new(error));
        }
        runtime.fire_lifecycle_hook(
            runtime::HookEvent::SubagentStop,
            &subagent_hook_context(job, "failed", Some(final_text.as_str()), None),
        );
        return Ok(AgentJobOutcome {
            final_text,
            structured,
            status: "failed",
            error: Some(budget_error),
            provider_error_class: None,
        });
    }
    // Capture the `StructuredOutput` tool call's input only when a schema was
    // requested (8c); otherwise this is a plain free-text agent. Resolved BEFORE
    // the diagnostic fallback so an empty final text with a structured result is
    // reported as "returned via StructuredOutput" rather than a misleading
    // "no final response" warning.
    let structured = continuation.latest_structured;
    // A completed continuation can legitimately end through StructuredOutput
    // or with only a final tool result. Preserve the latest substantive text
    // from an earlier budget window instead of replacing it with a diagnostic.
    let current_final_text = final_assistant_text(&summary);
    let final_text_source = if current_final_text.trim().is_empty() {
        continuation.latest_budget_text
    } else {
        current_final_text
    };
    let final_text = agent_result_or_diagnostic(
        &final_text_source,
        structured.is_some(),
        continuation.cumulative_iterations,
        continuation.cumulative_tool_calls,
    );
    let history_snapshot = token_history
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default();
    if let Err(error) = persist_agent_terminal_state_with_history(
        &job.manifest,
        "completed",
        Some(final_text.as_str()),
        None,
        history_snapshot,
    ) {
        runtime.fire_lifecycle_hook(
            runtime::HookEvent::SubagentStop,
            &subagent_hook_context(job, "failed", None, Some(error.as_str())),
        );
        return Err(runtime::RuntimeError::new(error));
    }
    runtime.fire_lifecycle_hook(
        runtime::HookEvent::SubagentStop,
        &subagent_hook_context(job, "completed", Some(final_text.as_str()), None),
    );
    Ok(AgentJobOutcome {
        final_text,
        structured,
        status: "completed",
        error: None,
        provider_error_class: None,
    })
}

/// The caller-facing result text for a sub-agent turn that ran to completion.
///
/// Normally this is the turn's final assistant text ([`final_assistant_text`]).
/// A turn can, however, complete with no final text at all — the model's last
/// message was tool calls only, or it exhausted its iteration / tool-call budget
/// mid-task. A raw extract is then an empty string, which both the in-memory
/// completion surfaced to the spawn caller and the persisted `## Result` block
/// render as a silent blank (no "Final response" section). To keep a completion
/// from ever being a silent empty, fall back to a short diagnostic derived from
/// the turn signals so the caller can always tell what happened and how to
/// recover.
///
/// When `has_structured` is true the agent returned its result through the
/// `StructuredOutput` tool rather than a final text message — that is a normal,
/// successful completion, so we surface a short pointer to the structured
/// payload instead of a "no final response" warning. Pure and independent of the
/// summary type, so it is exhaustively unit-tested.
fn agent_result_or_diagnostic(
    final_text: &str,
    has_structured: bool,
    iterations: usize,
    tool_calls: usize,
) -> String {
    if !final_text.trim().is_empty() {
        return final_text.to_string();
    }
    if has_structured {
        return "(results returned via StructuredOutput tool)".to_string();
    }
    // A successful completion with no final text was not stopped by a budget
    // error, so we do NOT claim a budget cutoff or tell the caller to "re-run
    // with a higher budget" — the turn simply ended on a tool call.
    format!(
        "[no final response] The sub-agent completed {iterations} iteration(s) and \
         {tool_calls} tool call(s) without producing a final text response — it likely \
         ended on a tool call. Re-run with a narrower task."
    )
}

/// The last non-empty assistant TEXT the agent produced before the budget cut
/// it off — the partial work worth surfacing to the parent.
///
/// On a budget-exhausted turn the runtime deterministically appends its
/// synthetic `[budget] …` closer as the LAST assistant message (every budget
/// arm pushes it, and a failed push errors the turn out of the budget path
/// entirely), so `final_assistant_text` would return that notice — duplicating
/// the banner [`agent_budget_exhausted_result`] already prepends and hiding the
/// agent's real narration. Skip the trailing closer structurally (drop the last
/// message, no string matching) and reverse-scan the rest for text; a turn that
/// only ever issued tool calls yields the empty string, which the banner
/// builder turns into the transcript-pointer note.
fn last_substantive_assistant_text(messages: &[runtime::session::ConversationMessage]) -> String {
    let before_closer = &messages[..messages.len().saturating_sub(1)];
    before_closer
        .iter()
        .rev()
        .map(|message| {
            message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    runtime::session::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .find(|text| !text.trim().is_empty())
        .unwrap_or_default()
}

/// Short, lowercase label for a budget-exhaustion kind, used in the caller-
/// facing banner.
fn budget_exhausted_kind_label(kind: runtime::BudgetExhausted) -> &'static str {
    match kind {
        runtime::BudgetExhausted::Iterations => "iteration budget",
        runtime::BudgetExhausted::Deadline => "time budget",
        runtime::BudgetExhausted::ToolCalls => "tool-call budget",
        runtime::BudgetExhausted::OutputTokens => "output-token budget",
        runtime::BudgetExhausted::InputTokens => "input-token budget",
        runtime::BudgetExhausted::VerificationTreadmill => "verification loop",
    }
}

/// The caller-facing result for a sub-agent turn that stopped on an exhausted
/// budget (iteration cap, wall-clock deadline, or tool-call budget). Unlike
/// [`agent_result_or_diagnostic`], the turn's work IS preserved, so this leads
/// with a one-line budget banner and then the agent's final assistant text — or,
/// when the agent ended mid-task with no final prose, a note pointing at the
/// preserved transcript plus a narrow-retry hint. Pure and independent of the
/// summary type, so it is exhaustively unit-tested.
fn agent_budget_exhausted_result(
    final_text: &str,
    kind: runtime::BudgetExhausted,
    iterations: usize,
    tool_calls: usize,
) -> String {
    let banner = format!(
        "[budget exhausted: {} after {iterations} iteration(s), {tool_calls} tool call(s)]",
        budget_exhausted_kind_label(kind)
    );
    if final_text.trim().is_empty() {
        format!(
            "{banner}\nNo final text; partial work is recorded in the agent transcript. \
             Continue in a follow-up turn or re-run with a narrower task."
        )
    } else {
        format!("{banner}\n{final_text}")
    }
}

fn agent_error_is_cancelled(error: &str) -> bool {
    error.to_ascii_lowercase().contains("agent cancelled")
}

fn agent_deadline(
    now: std::time::Instant,
    time_budget: Option<Duration>,
) -> Option<std::time::Instant> {
    time_budget.map(|budget| now + budget)
}

#[cfg(test)]
mod tests {
    use super::{
        AgentContinuationState, AgentJob, AgentOutput, AgentTerminalDisposition,
        SPAWNED_AGENT_MAX_BUDGET_CONTINUATIONS, SPAWNED_AGENT_MAX_ITERATIONS,
        SPAWNED_AGENT_MAX_TOOL_CALLS, agent_budget_continuation_prompt,
        agent_budget_exhausted_result, agent_deadline, agent_result_or_diagnostic,
        agent_terminal_disposition, agent_turn_made_progress,
        completion_structured_with_provider_error_class, continuation_error_result,
        continuation_text_is_meaningful, is_ad_hoc_turn_review,
        persist_agent_terminal_state_with_history, reconcile_completion_terminal_status,
        record_agent_verdict_outcome, should_auto_continue_agent, should_auto_continue_budget,
        subagent_allows_text_progress, verdict_passed,
    };
    use super::super::{final_structured_output, completion::AgentCompletion};
    use crate::misc_tools::agent_tools::AgentActivityTelemetry;
    use core_types::usage::TokenUsage;
    use runtime::session::{ContentBlock, ConversationMessage, MessageRole};
    use std::time::Duration;

    fn test_message(role: MessageRole, blocks: Vec<ContentBlock>) -> ConversationMessage {
        ConversationMessage {
            role,
            blocks,
            usage: None,
            thought_signature: None,
            reasoning_replay: None,
            model: None,
        }
    }

    fn continuation_summary(
        assistant_messages: Vec<ConversationMessage>,
        tool_results: Vec<ConversationMessage>,
        budget_exhausted: Option<runtime::BudgetExhausted>,
    ) -> runtime::TurnSummary {
        runtime::TurnSummary {
            assistant_messages,
            tool_results,
            prompt_cache_events: Vec::new(),
            iterations: SPAWNED_AGENT_MAX_ITERATIONS + 1,
            usage: TokenUsage::default(),
            turn_output_tokens: 0,
            auto_compaction: None,
            microcompact: None,
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted,
        }
    }

    fn successful_tool_result(tool_name: &str) -> ConversationMessage {
        test_message(
            MessageRole::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: format!("{tool_name}-id"),
                tool_name: tool_name.to_string(),
                output: "ok".to_string(),
                is_error: false,
                images: Vec::new(),
            }],
        )
    }

    /// A minimal `completed` manifest backed by real temp files, so a
    /// persistence path test can read back exactly what a spawn caller and the
    /// `## Result` block would see.
    fn empty_result_manifest(dir: &std::path::Path, id: &str) -> AgentOutput {
        let manifest = AgentOutput {
            agent_id: id.to_string(),
            parent_session_id: None,
            tool_call_id: None,
            name: id.to_string(),
            label: None,
            description: "agent".to_string(),
            subagent_type: Some("Explore".to_string()),
            requested_model: None,
            resolved_model: None,
            route_reason: None,
            route_role: None,
            route_complexity: None,
            route_risk: None,
            route_source: None,
            model: None,
            status: "running".to_string(),
            output_file: dir.join(format!("{id}.md")).display().to_string(),
            manifest_file: dir.join(format!("{id}.json")).display().to_string(),
            created_at: "100".to_string(),
            owner_pid: None,
            run_generation: 0,
            started_at: Some("100".to_string()),
            completed_at: None,
            completion_published_at: None,
            lane_events: Vec::new(),
            current_blocker: None,
            error: None,
            token_history: Vec::new(),
            current_tool: None,
            recent_tools: Vec::new(),
            tool_calls: 0,
            current_phase: None,
            output_tail: String::new(),
            last_activity_at: None,
            activity: AgentActivityTelemetry::default(),
        };
        super::super::manifest::write_agent_manifest(&manifest).expect("write manifest");
        std::fs::write(&manifest.output_file, "# Agent\n").expect("seed output file");
        manifest
    }

    fn unique_dir(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-{label}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    /// Path-level: an empty-final-text turn WITH a structured result persists the
    /// `StructuredOutput` pointer as the completion result (never a "no final
    /// response" warning), mirroring `run_agent_job`'s exact ordering — compute
    /// `structured`, then derive `final_text`, then persist.
    #[test]
    fn empty_result_with_structured_output_persists_structured_pointer() {
        let dir = unique_dir("spawn-empty-structured");
        let manifest = empty_result_manifest(&dir, "structured");

        // run_agent_job resolves `structured` before the diagnostic fallback.
        let has_structured = true;
        let final_text = agent_result_or_diagnostic("", has_structured, 3, 4);
        assert_eq!(final_text, "(results returned via StructuredOutput tool)");

        persist_agent_terminal_state_with_history(
            &manifest,
            "completed",
            Some(final_text.as_str()),
            None,
            Vec::new(),
        )
        .expect("persist completed state");

        let reread: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("reread manifest"),
        )
        .expect("parse manifest");
        assert_eq!(reread.status, "completed");
        let output = std::fs::read_to_string(&manifest.output_file).expect("read output");
        assert!(
            output.contains("(results returned via StructuredOutput tool)"),
            "the persisted Final response points at the structured payload: {output}",
        );
        assert!(
            !output.contains("[no final response]"),
            "a structured completion must not persist a warning: {output}",
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Path-level: an empty-final-text turn WITHOUT a structured result persists
    /// the simplified diagnostic — no misleading budget wording — as a real
    /// `## Result` block so the spawn caller never sees a silent blank.
    #[test]
    fn empty_result_without_structured_persists_simplified_diagnostic() {
        let dir = unique_dir("spawn-empty-blank");
        let manifest = empty_result_manifest(&dir, "blank");

        let has_structured = false;
        let final_text = agent_result_or_diagnostic("", has_structured, 5, 9);
        assert!(final_text.contains("[no final response]"), "{final_text}");
        assert!(!final_text.contains("budget"), "{final_text}");

        persist_agent_terminal_state_with_history(
            &manifest,
            "completed",
            Some(final_text.as_str()),
            None,
            Vec::new(),
        )
        .expect("persist completed state");

        let output = std::fs::read_to_string(&manifest.output_file).expect("read output");
        assert!(
            output.contains("### Final response") && output.contains("[no final response]"),
            "the empty turn still persists a non-blank Final response: {output}",
        );
        assert!(
            !output.contains("higher budget") && !output.contains("after reaching its"),
            "the persisted diagnostic must not claim a budget cutoff: {output}",
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn agent_result_passes_through_a_real_final_response() {
        assert_eq!(
            agent_result_or_diagnostic("the answer is 42", false, 3, 5),
            "the answer is 42",
        );
    }

    /// Pure: a budget-exhausted turn WITH a final assistant text leads with the
    /// budget banner (kind + counts) and then the preserved text.
    #[test]
    fn agent_budget_exhausted_result_leads_with_banner_then_final_text() {
        let out = agent_budget_exhausted_result(
            "partial progress: edited foo.rs",
            runtime::BudgetExhausted::Iterations,
            64,
            120,
        );
        assert!(
            out.starts_with(
                "[budget exhausted: iteration budget after 64 iteration(s), 120 tool call(s)]"
            ),
            "{out}"
        );
        assert!(out.contains("partial progress: edited foo.rs"), "{out}");
    }

    /// Pure: with no final text, the banner is followed by a transcript pointer
    /// and a narrow-retry hint — never a silent blank.
    #[test]
    fn agent_budget_exhausted_result_points_at_transcript_when_no_final_text() {
        let out =
            agent_budget_exhausted_result("   ", runtime::BudgetExhausted::ToolCalls, 10, 256);
        assert!(
            out.starts_with(
                "[budget exhausted: tool-call budget after 10 iteration(s), 256 tool call(s)]"
            ),
            "{out}"
        );
        assert!(
            out.contains("partial work is recorded in the agent transcript"),
            "{out}"
        );
        assert!(out.contains("narrower task"), "{out}");
    }

    /// Pure: the deadline kind is labelled "time budget".
    #[test]
    fn agent_budget_exhausted_result_labels_the_deadline_kind() {
        let out = agent_budget_exhausted_result("x", runtime::BudgetExhausted::Deadline, 1, 1);
        assert!(out.contains("time budget"), "{out}");
    }

    /// Pure: the partial-work extractor must skip the runtime's synthetic
    /// trailing `[budget] …` closer (always the LAST assistant message on a
    /// budget-exhausted turn) and return the last REAL narration — feeding
    /// `final_assistant_text` here would surface the closer itself, duplicating
    /// the banner and hiding the work. A tool-use-only remainder yields the
    /// empty string, which the banner builder turns into the transcript pointer.
    #[test]
    fn last_substantive_assistant_text_skips_the_synthetic_closer() {
        use runtime::session::{ContentBlock, ConversationMessage};
        let text = |s: &str| {
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: s.to_string(),
            }])
        };
        let closer = text("[budget] Iteration budget exhausted after 64 iteration(s); …");

        // Real narration before the closer is surfaced.
        let messages = vec![text("did A; next is B"), closer.clone()];
        assert_eq!(super::last_substantive_assistant_text(&messages), "did A; next is B");

        // Empty/whitespace narration is skipped in the reverse scan.
        let messages = vec![text("real work note"), text("   "), closer.clone()];
        assert_eq!(super::last_substantive_assistant_text(&messages), "real work note");

        // Only the closer (no prior text) → empty → transcript-pointer branch.
        assert_eq!(super::last_substantive_assistant_text(&[closer]), "");
        assert_eq!(super::last_substantive_assistant_text(&[]), "");
    }

    /// Path-level: a budget-exhausted turn persists status `failed` WITH the
    /// partial-work result (not a silent `None`), so the parent and the `##
    /// Result` block both see what got done. Mirrors `run_agent_job`'s budget
    /// branch: build the partial text, then persist it as a failed terminal.
    #[test]
    fn budget_exhausted_persists_failed_status_with_partial_result() {
        let dir = unique_dir("spawn-budget-exhausted");
        let manifest = empty_result_manifest(&dir, "budget");

        let final_text = agent_budget_exhausted_result(
            "did some work",
            runtime::BudgetExhausted::Iterations,
            SPAWNED_AGENT_MAX_ITERATIONS,
            3,
        );
        persist_agent_terminal_state_with_history(
            &manifest,
            "failed",
            Some(final_text.as_str()),
            None,
            Vec::new(),
        )
        .expect("persist failed-with-partial state");

        let reread: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("reread manifest"),
        )
        .expect("parse manifest");
        assert_eq!(reread.status, "failed");
        let output = std::fs::read_to_string(&manifest.output_file).expect("read output");
        assert!(
            output.contains("[budget exhausted:") && output.contains("did some work"),
            "a budget cutoff must persist the partial work, not a blank: {output}",
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn agent_result_passes_through_final_text_even_when_structured_present() {
        // A real final text always wins, structured or not.
        assert_eq!(
            agent_result_or_diagnostic("the answer is 42", true, 3, 5),
            "the answer is 42",
        );
    }

    #[test]
    fn agent_result_reports_structured_output_for_blank_text() {
        // Empty final text but a StructuredOutput result: a normal, successful
        // completion — surface a pointer to the structured payload, NOT a warning.
        let out = agent_result_or_diagnostic("   \n  ", true, 4, 7);
        assert_eq!(out, "(results returned via StructuredOutput tool)", "{out}");
        assert!(!out.contains("[no final response]"), "{out}");
    }

    #[test]
    fn agent_result_diagnoses_a_blank_response_without_structured() {
        let out = agent_result_or_diagnostic("   \n  ", false, 4, 7);
        assert!(out.contains("[no final response]"), "{out}");
        assert!(out.contains("4 iteration(s)"), "{out}");
        assert!(out.contains("7 tool call(s)"), "{out}");
    }

    #[test]
    fn agent_result_empty_diagnostic_omits_budget_wording() {
        // A successful completion was not stopped by a budget error, so the
        // diagnostic must not claim a budget cutoff or tell the caller to raise
        // the budget — even at the iteration / tool-call ceilings.
        for out in [
            agent_result_or_diagnostic("", false, SPAWNED_AGENT_MAX_ITERATIONS, 3),
            agent_result_or_diagnostic("", false, 2, SPAWNED_AGENT_MAX_TOOL_CALLS),
        ] {
            assert!(out.contains("[no final response]"), "{out}");
            assert!(!out.contains("budget"), "must not mention a budget: {out}");
            assert!(
                !out.contains("after reaching its"),
                "must not claim a budget cutoff: {out}",
            );
            assert!(
                !out.contains("higher budget"),
                "must not advise raising the budget: {out}",
            );
        }
    }

    #[test]
    fn spawned_agents_default_to_no_wall_clock_deadline() {
        assert!(
            agent_deadline(std::time::Instant::now(), None).is_none(),
            "omitted budgets should wait for the actual agent result"
        );
    }

    #[test]
    fn spawned_agents_honor_explicit_wall_clock_budget() {
        let now = std::time::Instant::now();
        let deadline = agent_deadline(now, Some(Duration::from_secs(20 * 60)))
            .expect("explicit budget should set a deadline");
        assert_eq!(deadline.duration_since(now), Duration::from_secs(20 * 60));
    }

    #[test]
    fn spawned_agents_are_bounded_to_stop_runaways() {
        // A finite, modest cap turns a runaway sub-agent (the 174-tool-call
        // fan-out of 2026-06-07) into a graceful `failed` completion instead of
        // unbounded thrash. The enforcement + failure path are proven in
        // `runtime::conversation::tests::run_turn_errors_when_max_iterations_is_exceeded`.
        assert_eq!(
            SPAWNED_AGENT_MAX_ITERATIONS, 64,
            "spawned sub-agents must keep a finite iteration cap; usize::MAX reintroduces the runaway"
        );
    }

    #[test]
    fn spawned_agent_progress_requires_mutation_or_read_only_deliverable() {
        let read_only = continuation_summary(
            Vec::new(),
            vec![successful_tool_result("read_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        assert!(!agent_turn_made_progress(&read_only, false));
        assert!(!agent_turn_made_progress(&read_only, true));

        let mutation = continuation_summary(
            Vec::new(),
            vec![successful_tool_result("edit_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        assert!(agent_turn_made_progress(&mutation, false));

        let report = "substantive finding ".repeat(20);
        assert!(continuation_text_is_meaningful(&report));
        let filler_without_evidence = continuation_summary(
            vec![
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text {
                        text: report.clone(),
                    }],
                ),
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text {
                        text: "[budget] continuation required".to_string(),
                    }],
                ),
            ],
            Vec::new(),
            Some(runtime::BudgetExhausted::Iterations),
        );
        assert!(!agent_turn_made_progress(&filler_without_evidence, true));
        let read_only_report = continuation_summary(
            vec![
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text { text: report }],
                ),
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text {
                        text: "[budget] continuation required".to_string(),
                    }],
                ),
            ],
            vec![successful_tool_result("read_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        assert!(!agent_turn_made_progress(&read_only_report, false));
        assert!(agent_turn_made_progress(&read_only_report, true));
    }

    #[test]
    fn spawned_agent_schema_continues_until_structured_deliverable() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["verdict"],
            "properties": { "verdict": { "type": "string" } },
            "additionalProperties": false
        });
        let mutation = continuation_summary(
            Vec::new(),
            vec![successful_tool_result("edit_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        assert!(should_auto_continue_agent(
            &mutation,
            Some(&schema),
            0,
            SPAWNED_AGENT_MAX_TOOL_CALLS,
            true,
            false,
        ));

        let structured = continuation_summary(
            vec![test_message(
                MessageRole::Assistant,
                vec![ContentBlock::ToolUse {
                    id: "structured".to_string(),
                    name: "StructuredOutput".to_string(),
                    input: serde_json::json!({"verdict": "pass"}).to_string(),
                }],
            )],
            vec![test_message(
                MessageRole::Tool,
                vec![ContentBlock::ToolResult {
                    tool_use_id: "structured".to_string(),
                    tool_name: "StructuredOutput".to_string(),
                    output: "ok".to_string(),
                    is_error: false,
                    images: Vec::new(),
                }],
            )],
            Some(runtime::BudgetExhausted::Iterations),
        );
        assert!(!should_auto_continue_agent(
            &structured,
            Some(&schema),
            0,
            SPAWNED_AGENT_MAX_TOOL_CALLS,
            true,
            false,
        ));
        assert_eq!(
            agent_terminal_disposition(&structured, true, true),
            AgentTerminalDisposition::Completed,
            "a schema agent that submitted StructuredOutput at the iteration boundary is complete",
        );
        assert_eq!(
            agent_terminal_disposition(&structured, false, true),
            AgentTerminalDisposition::BudgetFailed(runtime::BudgetExhausted::Iterations),
            "non-schema agents must not turn an iteration cutoff into success",
        );

        let mut hard_stop = structured;
        hard_stop.budget_exhausted = Some(runtime::BudgetExhausted::InputTokens);
        assert_eq!(
            agent_terminal_disposition(&hard_stop, true, true),
            AgentTerminalDisposition::BudgetFailed(runtime::BudgetExhausted::InputTokens),
            "StructuredOutput must not bypass a hard token budget",
        );
    }

    #[test]
    fn spawned_agent_completion_respects_durable_stopped_status() {
        let rich_result = serde_json::json!({"verdict": "pass"});
        let mut completion = AgentCompletion {
            agent_id: "agent-status-race".to_string(),
            name: "status-race".to_string(),
            status: "completed".to_string(),
            result: Some("recovered rich result".to_string()),
            structured: Some(rich_result.clone()),
            error: None,
            output_tokens: 42,
        };

        reconcile_completion_terminal_status(&mut completion, "stopped", None);

        assert_eq!(completion.status, "stopped");
        assert_eq!(completion.result.as_deref(), Some("recovered rich result"));
        assert_eq!(completion.structured, Some(rich_result));
        assert_eq!(completion.error.as_deref(), Some("agent stopped"));
        assert_eq!(completion.output_tokens, 42);
    }

    #[test]
    #[allow(clippy::too_many_lines)] // cohesive acceptance/rejection contract matrix
    fn spawned_agent_schema_requires_unique_successful_valid_structured_output() {
        fn boundary(
            assistant: Vec<ContentBlock>,
            results: Vec<ContentBlock>,
        ) -> runtime::TurnSummary {
            continuation_summary(
                vec![test_message(MessageRole::Assistant, assistant)],
                vec![test_message(MessageRole::Tool, results)],
                Some(runtime::BudgetExhausted::Iterations),
            )
        }
        let schema = serde_json::json!({
            "type": "object",
            "required": ["verdict"],
            "properties": { "verdict": { "type": "string" } },
            "additionalProperties": false
        });
        let use_block = |id: &str, input: serde_json::Value| ContentBlock::ToolUse {
            id: id.to_string(),
            name: "StructuredOutput".to_string(),
            input: input.to_string(),
        };
        let result = |id: &str, name: &str, is_error| ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            tool_name: name.to_string(),
            output: "ok".to_string(),
            is_error,
            images: Vec::new(),
        };
        let progress = || result("progress", "edit_file", false);
        let continues = |summary: &runtime::TurnSummary, schema: &serde_json::Value| {
            should_auto_continue_agent(
                summary,
                Some(schema),
                0,
                SPAWNED_AGENT_MAX_TOOL_CALLS,
                true,
                false,
            )
        };
        let rejects = |summary: &runtime::TurnSummary, schema: &serde_json::Value| {
            assert!(final_structured_output(summary, schema).is_none());
            assert_eq!(
                agent_terminal_disposition(summary, true, false),
                AgentTerminalDisposition::BudgetFailed(runtime::BudgetExhausted::Iterations),
            );
        };

        let valid = boundary(
            vec![use_block("valid", serde_json::json!({"verdict": "pass"}))],
            vec![result("valid", "StructuredOutput", false)],
        );
        assert_eq!(
            final_structured_output(&valid, &schema),
            Some(serde_json::json!({"verdict": "pass"}))
        );
        assert!(!continues(&valid, &schema));
        assert_eq!(
            agent_terminal_disposition(&valid, true, true),
            AgentTerminalDisposition::Completed,
        );

        let duplicate_id = boundary(
            vec![
                use_block("duplicate", serde_json::json!({"verdict": "pass"})),
                use_block("duplicate", serde_json::json!({"verdict": 7})),
            ],
            vec![result("duplicate", "StructuredOutput", false), progress()],
        );
        rejects(&duplicate_id, &schema);
        assert!(continues(&duplicate_id, &schema));

        let different_ids = boundary(
            vec![
                use_block("first", serde_json::json!({"verdict": "pass"})),
                use_block("second", serde_json::json!({"verdict": "pass"})),
            ],
            vec![
                result("first", "StructuredOutput", false),
                result("second", "StructuredOutput", false),
                progress(),
            ],
        );
        rejects(&different_ids, &schema);
        assert!(continues(&different_ids, &schema));

        let error_result = boundary(
            vec![use_block("error", serde_json::json!({"verdict": "pass"}))],
            vec![result("error", "StructuredOutput", true), progress()],
        );
        rejects(&error_result, &schema);
        assert!(continues(&error_result, &schema));

        let mismatched_result = boundary(
            vec![use_block("mismatch", serde_json::json!({"verdict": "pass"}))],
            vec![result("mismatch", "read_file", false), progress()],
        );
        rejects(&mismatched_result, &schema);
        assert!(continues(&mismatched_result, &schema));

        let invalid_value = boundary(
            vec![use_block("invalid", serde_json::json!({"verdict": 7}))],
            vec![result("invalid", "StructuredOutput", false), progress()],
        );
        rejects(&invalid_value, &schema);
        assert!(continues(&invalid_value, &schema));

        let unsupported_schema = serde_json::json!({"type": "object", "oneOf": []});
        let unsupported_schema_value = boundary(
            vec![use_block("unsupported", serde_json::json!({"verdict": "pass"}))],
            vec![result("unsupported", "StructuredOutput", false), progress()],
        );
        assert_eq!(
            final_structured_output(&unsupported_schema_value, &unsupported_schema),
            Some(serde_json::json!({"verdict": "pass"})),
            "unsupported schema keywords must preserve the captured value verbatim"
        );
        assert!(!continues(&unsupported_schema_value, &unsupported_schema));
        assert_eq!(
            agent_terminal_disposition(&unsupported_schema_value, true, true),
            AgentTerminalDisposition::Completed,
        );
    }

    #[test]
    fn spawned_agent_text_progress_is_limited_to_read_only_agents() {
        assert!(subagent_allows_text_progress("Explore", None));
        assert!(subagent_allows_text_progress(
            "custom",
            Some(runtime::PermissionMode::ReadOnly),
        ));
        assert!(!subagent_allows_text_progress("general-purpose", None));
        assert!(!subagent_allows_text_progress("debugger", None));
    }

    #[test]
    fn spawned_agent_continuation_state_preserves_outputs_and_total_tool_calls() {
        let schema = serde_json::json!({"type": "object"});
        let partial = "verified partial result ".repeat(20);
        let first = continuation_summary(
            vec![
                test_message(
                    MessageRole::Assistant,
                    vec![
                        ContentBlock::Text {
                            text: partial.clone(),
                        },
                        ContentBlock::ToolUse {
                            id: "structured".to_string(),
                            name: "StructuredOutput".to_string(),
                            input: serde_json::json!({"verdict": "pass"}).to_string(),
                        },
                    ],
                ),
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text {
                        text: "[budget] continuation required".to_string(),
                    }],
                ),
            ],
            vec![test_message(
                MessageRole::Tool,
                vec![ContentBlock::ToolResult {
                    tool_use_id: "structured".to_string(),
                    tool_name: "StructuredOutput".to_string(),
                    output: "ok".to_string(),
                    is_error: false,
                    images: Vec::new(),
                }],
            )],
            Some(runtime::BudgetExhausted::Iterations),
        );
        let shorter = "shorter partial result ".repeat(13);
        let second = continuation_summary(
            vec![
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text {
                        text: shorter.clone(),
                    }],
                ),
                test_message(
                    MessageRole::Assistant,
                    vec![ContentBlock::Text {
                        text: "[budget] continuation required".to_string(),
                    }],
                ),
            ],
            Vec::new(),
            Some(runtime::BudgetExhausted::Iterations),
        );
        let completed = continuation_summary(Vec::new(), Vec::new(), None);

        let mut state = AgentContinuationState::default();
        state.observe(&first, Some(&schema));
        state.observe(&second, Some(&schema));
        state.observe(&completed, Some(&schema));

        assert_eq!(state.cumulative_tool_calls, 1);
        assert_eq!(state.remaining_tool_calls(), SPAWNED_AGENT_MAX_TOOL_CALLS - 1);
        assert_eq!(state.latest_budget_text, shorter);
        assert_eq!(
            state.latest_structured,
            Some(serde_json::json!({"verdict": "pass"}))
        );
        let failed = continuation_error_result(&state, "provider disconnected")
            .expect("partial result should survive a continuation error");
        assert!(failed.contains("shorter partial result"), "{failed}");
        assert!(failed.contains("provider disconnected"), "{failed}");
    }

    #[test]
    fn spawned_agent_provider_error_metadata_preserves_partial_payload() {
        let structured = completion_structured_with_provider_error_class(
            Some(serde_json::json!({"partial": "preserved"})),
            Some(api::ProviderErrorClass::RateLimit { retry_after: None }),
        )
        .expect("provider metadata should be attached");

        assert_eq!(structured["partial"], "preserved");
        assert_eq!(structured["providerErrorClass"], "rateLimit");
    }

    #[test]
    fn spawned_agent_mutation_only_window_preserves_fallback_on_continuation_error() {
        let mutation = continuation_summary(
            Vec::new(),
            vec![successful_tool_result("edit_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        let mut state = AgentContinuationState::default();
        state.observe(&mutation, None);

        let failed = continuation_error_result(&state, "provider disconnected")
            .expect("mutation progress must retain a parent-facing fallback");
        assert!(failed.contains("[budget exhausted:"), "{failed}");
        assert!(failed.contains("partial work is recorded in the agent transcript"), "{failed}");
        assert!(failed.contains("provider disconnected"), "{failed}");
    }

    #[test]
    fn spawned_agent_worker_enriches_an_externally_stopped_manifest() {
        let dir = unique_dir("spawn-external-stop-enrichment");
        let manifest = empty_result_manifest(&dir, "stopped");
        assert!(
            super::super::manifest::persist_agent_stopped_state(
                &manifest,
                "external stop requested",
            )
            .expect("persist external stop")
        );

        persist_agent_terminal_state_with_history(
            &manifest,
            "stopped",
            Some("recovered partial result"),
            Some("agent cancelled".to_string()),
            Vec::new(),
        )
        .expect("worker should enrich the stopped terminal state");

        let output = std::fs::read_to_string(&manifest.output_file).expect("read output");
        assert!(output.contains("recovered partial result"), "{output}");
        let reread: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(reread.status, "stopped");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spawned_agent_tool_budget_counts_issued_tool_uses_not_results() {
        let summary = continuation_summary(
            vec![test_message(
                MessageRole::Assistant,
                vec![
                    ContentBlock::ToolUse {
                        id: "first".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "first.rs"}).to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "second".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "second.rs"}).to_string(),
                    },
                ],
            )],
            vec![successful_tool_result("read_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );

        let mut state = AgentContinuationState::default();
        state.observe(&summary, None);

        assert_eq!(state.cumulative_tool_calls, 2);
        assert_eq!(state.remaining_tool_calls(), SPAWNED_AGENT_MAX_TOOL_CALLS - 2);
    }

    #[test]
    fn spawned_agent_continuation_keeps_token_budgets_cumulative() {
        let mut first = continuation_summary(
            Vec::new(),
            vec![successful_tool_result("edit_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        first.turn_output_tokens = 60;
        first.usage.input_tokens = 400;

        let mut state = AgentContinuationState::default();
        state.observe(&first, None);
        assert_eq!(state.remaining_output_tokens(Some(100)), Some(40));
        assert_eq!(state.remaining_input_tokens(Some(1_000)), Some(600));
        assert!(state.token_budgets_have_headroom(Some(100), Some(1_000)));

        let mut second = continuation_summary(
            Vec::new(),
            vec![successful_tool_result("edit_file")],
            Some(runtime::BudgetExhausted::Iterations),
        );
        second.turn_output_tokens = 40;
        second.usage.input_tokens = 1_000;
        state.observe(&second, None);

        assert_eq!(state.remaining_output_tokens(Some(100)), Some(0));
        assert_eq!(state.remaining_input_tokens(Some(1_000)), Some(0));
        assert!(!state.token_budgets_have_headroom(Some(100), Some(1_000)));
        assert_eq!(
            state.cumulative_iterations,
            2 * (SPAWNED_AGENT_MAX_ITERATIONS + 1),
        );
    }

    #[test]
    fn spawned_agents_auto_continue_iteration_cutoffs_with_progress() {
        assert!(should_auto_continue_budget(
            Some(runtime::BudgetExhausted::Iterations),
            true,
            0,
            SPAWNED_AGENT_MAX_TOOL_CALLS,
            true,
        ));
    }

    #[test]
    fn spawned_agents_do_not_continue_hard_stop_or_stalled_cutoffs() {
        for kind in [
            runtime::BudgetExhausted::InputTokens,
            runtime::BudgetExhausted::OutputTokens,
            runtime::BudgetExhausted::Deadline,
            runtime::BudgetExhausted::ToolCalls,
            runtime::BudgetExhausted::VerificationTreadmill,
        ] {
            assert!(!should_auto_continue_budget(
                Some(kind),
                true,
                0,
                SPAWNED_AGENT_MAX_TOOL_CALLS,
                true,
            ));
        }
        assert!(!should_auto_continue_budget(
            Some(runtime::BudgetExhausted::Iterations),
            false,
            0,
            SPAWNED_AGENT_MAX_TOOL_CALLS,
            true,
        ));
        assert!(!should_auto_continue_budget(
            Some(runtime::BudgetExhausted::InputTokens),
            true,
            SPAWNED_AGENT_MAX_BUDGET_CONTINUATIONS,
            SPAWNED_AGENT_MAX_TOOL_CALLS,
            true,
        ));
        assert!(!should_auto_continue_budget(
            Some(runtime::BudgetExhausted::Iterations),
            true,
            0,
            0,
            true,
        ));
        assert!(!should_auto_continue_budget(
            Some(runtime::BudgetExhausted::Iterations),
            true,
            0,
            SPAWNED_AGENT_MAX_TOOL_CALLS,
            false,
        ));
    }

    #[test]
    fn spawned_agent_continuation_prompt_requires_completion() {
        let prompt = agent_budget_continuation_prompt(runtime::BudgetExhausted::Iterations, 1);
        assert!(prompt.contains("Continue the same task"), "{prompt}");
        assert!(prompt.contains("finish the requested work"), "{prompt}");
        assert!(prompt.contains("Do not restart or merely summarize"), "{prompt}");
        assert!(prompt.contains("Continuation 1/2"), "{prompt}");
    }

    #[test]
    fn subagent_stops_at_max_tool_calls() {
        // The iteration cap alone does not bound tool-call volume: one model turn
        // can issue many parallel `tool_use` blocks. `run_agent_job` wires this
        // cap via `.with_max_tool_calls`, and the enforcement + graceful `failed`
        // completion are proven in
        // `runtime::conversation::tests::run_turn_errors_when_max_tool_calls_is_exceeded`.
        assert_eq!(
            SPAWNED_AGENT_MAX_TOOL_CALLS, 256,
            "spawned sub-agents must keep a finite tool-call cap; usize::MAX reintroduces the \
             unbounded tool-call burst (the 174-call runaway of 2026-06-07)"
        );
    }

    // ── Phase 4 verdict channel (sources #2 planner-bound, #3 ad-hoc) ──────

    #[test]
    fn verdict_passed_maps_pass_and_finding_only() {
        assert_eq!(verdict_passed("pass"), Some(true));
        assert_eq!(verdict_passed("finding"), Some(false));
        // "retry" (no usable structured verdict) and any unrecognized label
        // are ambiguous — never a signal.
        assert_eq!(verdict_passed("retry"), None);
        assert_eq!(verdict_passed("unknown-label"), None);
        assert_eq!(verdict_passed(""), None);
    }

    fn verdict_test_manifest(id: &str, subagent_type: &str) -> AgentOutput {
        AgentOutput {
            agent_id: id.to_string(),
            parent_session_id: None,
            tool_call_id: None,
            name: id.to_string(),
            label: None,
            description: "agent".to_string(),
            subagent_type: Some(subagent_type.to_string()),
            requested_model: None,
            resolved_model: Some("worker-model".to_string()),
            route_reason: None,
            route_role: None,
            route_complexity: None,
            route_risk: None,
            route_source: None,
            model: Some("worker-model".to_string()),
            status: "running".to_string(),
            output_file: format!("/tmp/{id}.md"),
            manifest_file: format!("/tmp/{id}.json"),
            created_at: "100".to_string(),
            owner_pid: None,
            run_generation: 0,
            started_at: Some("100".to_string()),
            completed_at: None,
            completion_published_at: None,
            lane_events: Vec::new(),
            current_blocker: None,
            error: None,
            token_history: Vec::new(),
            current_tool: None,
            recent_tools: Vec::new(),
            tool_calls: 0,
            current_phase: None,
            output_tail: String::new(),
            last_activity_at: None,
            activity: AgentActivityTelemetry::default(),
        }
    }

    #[test]
    fn ad_hoc_turn_review_requires_both_type_and_marker() {
        let reviewer = verdict_test_manifest("r1", "code-reviewer");
        let verifier = verdict_test_manifest("r2", "Verification");
        let explorer = verdict_test_manifest("r3", "Explore");

        let job_with = |manifest: AgentOutput, prompt: &str| AgentJob {
            manifest,
            prompt: prompt.to_string(),
            system_prompt: Vec::new(),
            allowed_tools: std::collections::BTreeSet::new(),
            permission_rules: None,
            permission_mode: None,
            cwd: None,
            lsp: None,
            schema: None,
            workflow_member: false,
            time_budget: None,
            thinking_budget_tokens: None,
            route_effort: None,
            api_concurrency: None,
            route_fallback_models: Vec::new(),
            mcp_passthrough: None,
            hook_config: runtime::RuntimeHookConfig::default(),
            cancel_signal: runtime::HookAbortSignal::new(),
            judged_agent: None,
            parent_model: None,
            steering: runtime::SteeringQueue::default(),
            transcript_path: None,
            resume: false,
        };

        // Right type + English marker.
        assert!(is_ad_hoc_turn_review(&job_with(
            reviewer.clone(),
            "Adversarially review the current working tree after the implementation."
        )));
        // Right type + Korean marker.
        assert!(is_ad_hoc_turn_review(&job_with(
            verifier.clone(),
            "이 diff를 검증해줘"
        )));
        // Right type, but no whitelist marker — some OTHER unrelated review.
        assert!(!is_ad_hoc_turn_review(&job_with(
            reviewer.clone(),
            "Review the design doc in docs/architecture.md"
        )));
        // Whitelist marker present, but not a reviewer/verifier type.
        assert!(!is_ad_hoc_turn_review(&job_with(
            explorer,
            "Inspect the current diff for stray debug prints"
        )));
        // Neither.
        assert!(!is_ad_hoc_turn_review(&job_with(
            reviewer,
            "Summarize the README"
        )));
    }

    /// Env isolation for the verdict-recording integration tests below:
    /// `ZO_AGENT_STORE` (the worker-manifest lookup `attribution`'s
    /// `record_verdict_outcome_for_agent` reads) and `ZO_STATE_DIR` (the
    /// route-outcome log's root, `runtime::zo_project_state_dir`) both
    /// redirect to fresh temp dirs — regardless of the real process cwd — so
    /// these tests never touch the developer's real `~/.zo`. Guarded by the
    /// crate-wide env-mutation lock (other modules assert on the same globals).
    struct VerdictTestEnv {
        _guard: std::sync::MutexGuard<'static, ()>,
        state_dir: std::path::PathBuf,
        agent_store: std::path::PathBuf,
        prior_state_dir: Option<std::ffi::OsString>,
        prior_agent_store: Option<std::ffi::OsString>,
    }

    impl VerdictTestEnv {
        fn setup(tag: &str) -> Self {
            let guard = crate::tests::env_lock()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let base = std::env::temp_dir().join(format!(
                "zo-verdict-spawn-{tag}-{}-{nanos}",
                std::process::id()
            ));
            let state_dir = base.join("state");
            let agent_store = base.join("agents");
            std::fs::create_dir_all(&state_dir).expect("state dir");
            std::fs::create_dir_all(&agent_store).expect("agent store dir");
            let prior_state_dir = std::env::var_os("ZO_STATE_DIR");
            let prior_agent_store = std::env::var_os(super::super::labels::AGENT_STORE_ENV);
            std::env::set_var("ZO_STATE_DIR", &state_dir);
            std::env::set_var(super::super::labels::AGENT_STORE_ENV, &agent_store);
            Self {
                _guard: guard,
                state_dir,
                agent_store,
                prior_state_dir,
                prior_agent_store,
            }
        }

        fn write_worker_manifest(&self, manifest: &AgentOutput) {
            std::fs::write(
                self.agent_store.join(format!("{}.json", manifest.agent_id)),
                serde_json::to_string(manifest).expect("manifest json"),
            )
            .expect("write worker manifest");
        }

        // Associated fn (not `&self`): the route-outcome log's root is fixed
        // by `ZO_STATE_DIR` (set in `setup`), and the cwd-derived slug
        // underneath it needs no other `self` state to resolve.
        fn read_outcomes() -> Vec<runtime::RouteOutcomeRecord> {
            let cwd = std::env::current_dir().expect("cwd");
            runtime::read_route_outcomes(&cwd).unwrap_or_default()
        }
    }

    impl Drop for VerdictTestEnv {
        fn drop(&mut self) {
            match self.prior_state_dir.take() {
                Some(value) => std::env::set_var("ZO_STATE_DIR", value),
                None => std::env::remove_var("ZO_STATE_DIR"),
            }
            match self.prior_agent_store.take() {
                Some(value) => std::env::set_var(super::super::labels::AGENT_STORE_ENV, value),
                None => std::env::remove_var(super::super::labels::AGENT_STORE_ENV),
            }
            let _ = std::fs::remove_dir_all(&self.state_dir);
            let _ = std::fs::remove_dir_all(&self.agent_store);
        }
    }

    fn passing_completion(structured: Option<serde_json::Value>) -> AgentCompletion {
        AgentCompletion {
            agent_id: "reviewer-agent".to_string(),
            name: "reviewer".to_string(),
            status: "completed".to_string(),
            result: Some("done".to_string()),
            structured,
            error: None,
            output_tokens: 0,
        }
    }

    fn reviewer_job(judged_agent: Option<&str>, parent_model: Option<&str>, prompt: &str) -> AgentJob {
        AgentJob {
            manifest: verdict_test_manifest("reviewer-agent", "code-reviewer"),
            prompt: prompt.to_string(),
            system_prompt: Vec::new(),
            allowed_tools: std::collections::BTreeSet::new(),
            permission_rules: None,
            permission_mode: None,
            cwd: None,
            lsp: None,
            schema: None,
            workflow_member: false,
            time_budget: None,
            thinking_budget_tokens: None,
            route_effort: None,
            api_concurrency: None,
            route_fallback_models: Vec::new(),
            mcp_passthrough: None,
            hook_config: runtime::RuntimeHookConfig::default(),
            cancel_signal: runtime::HookAbortSignal::new(),
            judged_agent: judged_agent.map(str::to_string),
            parent_model: parent_model.map(str::to_string),
            steering: runtime::SteeringQueue::default(),
            transcript_path: None,
            resume: false,
        }
    }

    #[test]
    fn planner_bound_verdict_records_against_the_judged_worker() {
        let env = VerdictTestEnv::setup("planner-bound");
        env.write_worker_manifest(&verdict_test_manifest("worker-1", "Refactor"));

        let job = reviewer_job(Some("worker-1"), None, "judge sibling worker-1's change");
        let completion = passing_completion(Some(serde_json::json!({"verdict": "pass", "coverage": "tests"})));
        record_agent_verdict_outcome(&job, &completion);

        let outcomes = VerdictTestEnv::read_outcomes();
        assert_eq!(outcomes.len(), 1, "exactly one verdict for the judged worker");
        let record = &outcomes[0];
        assert_eq!(record.route_key, "subagent:Refactor");
        assert_eq!(record.selected_model, "worker-model");
        assert_eq!(record.signal.as_deref(), Some("verdict"));
        assert_eq!(record.status, "completed");
    }

    #[test]
    fn planner_bound_verdict_is_silent_when_the_judged_worker_manifest_is_missing() {
        let _env = VerdictTestEnv::setup("planner-bound-missing");
        // Deliberately no `write_worker_manifest` call — the judged agent id
        // does not resolve to any manifest in the store.

        let job = reviewer_job(Some("nonexistent-worker"), None, "judge a worker");
        let completion = passing_completion(Some(serde_json::json!({"verdict": "pass", "coverage": "tests"})));
        record_agent_verdict_outcome(&job, &completion);

        assert!(
            VerdictTestEnv::read_outcomes().is_empty(),
            "a judged-agent binding to a worker with no manifest must record nothing"
        );
    }

    #[test]
    fn ad_hoc_review_records_against_the_main_turn_when_whitelisted() {
        let _env = VerdictTestEnv::setup("ad-hoc-whitelisted");

        let job = reviewer_job(
            None,
            Some("claude-opus-4-8"),
            "Inspect the current diff and relevant tests.",
        );
        let completion = passing_completion(Some(serde_json::json!({"verdict": "pass", "coverage": "tests"})));
        record_agent_verdict_outcome(&job, &completion);

        let outcomes = VerdictTestEnv::read_outcomes();
        assert_eq!(outcomes.len(), 1);
        let record = &outcomes[0];
        assert_eq!(record.route_key, "main:turn");
        assert_eq!(record.selected_model, "claude-opus-4-8");
        assert_eq!(record.signal.as_deref(), Some("verdict"));
        assert_eq!(record.status, "completed");
    }

    #[test]
    fn ad_hoc_review_is_silent_without_a_whitelist_marker() {
        let _env = VerdictTestEnv::setup("ad-hoc-no-marker");

        let job = reviewer_job(
            None,
            Some("claude-opus-4-8"),
            "Please look over the project and tell me what you think.",
        );
        let completion = passing_completion(Some(serde_json::json!({"verdict": "pass", "coverage": "tests"})));
        record_agent_verdict_outcome(&job, &completion);

        assert!(
            VerdictTestEnv::read_outcomes().is_empty(),
            "an ad-hoc review without a current-turn marker must never guess its target"
        );
    }

    #[test]
    fn ad_hoc_review_is_silent_without_a_parent_model() {
        let _env = VerdictTestEnv::setup("ad-hoc-no-parent-model");

        let job = reviewer_job(None, None, "Inspect the current diff and relevant tests.");
        let completion = passing_completion(Some(serde_json::json!({"verdict": "pass", "coverage": "tests"})));
        record_agent_verdict_outcome(&job, &completion);

        assert!(
            VerdictTestEnv::read_outcomes().is_empty(),
            "no parent model available at this seam must skip recording, never guess a route"
        );
    }

    #[test]
    fn unparseable_verdict_records_nothing_for_either_source() {
        let env = VerdictTestEnv::setup("unparseable");
        env.write_worker_manifest(&verdict_test_manifest("worker-2", "Refactor"));

        // No `structured` output at all — `semantic_verdict` cannot recover a
        // usable label, so neither the planner-bound nor the ad-hoc path may
        // record anything, even though the whitelist/binding both line up.
        let bound_job = reviewer_job(Some("worker-2"), None, "judge sibling worker-2's change");
        record_agent_verdict_outcome(&bound_job, &passing_completion(None));

        let ad_hoc_job = reviewer_job(
            None,
            Some("claude-opus-4-8"),
            "Inspect the current diff and relevant tests.",
        );
        record_agent_verdict_outcome(&ad_hoc_job, &passing_completion(None));

        assert!(
            VerdictTestEnv::read_outcomes().is_empty(),
            "an unparseable/missing structured verdict must never be recorded"
        );
    }

    #[test]
    fn still_running_completion_is_never_recorded() {
        let env = VerdictTestEnv::setup("still-running");
        env.write_worker_manifest(&verdict_test_manifest("worker-3", "Refactor"));

        let job = reviewer_job(Some("worker-3"), None, "judge sibling worker-3's change");
        let mut completion = passing_completion(Some(serde_json::json!({"verdict": "pass", "coverage": "tests"})));
        completion.status = "still_running".to_string();
        record_agent_verdict_outcome(&job, &completion);

        assert!(
            VerdictTestEnv::read_outcomes().is_empty(),
            "a still-running completion must never be recorded — only a terminal status may reach the verdict recorder"
        );
    }
}
