mod agent_tools;
mod audit;
mod config_tools;
pub(crate) mod council;
mod dispatch;
mod fanout_isolation;
mod notebook_tools;
mod retrieve_output;
mod session_recall;
mod skill_tools;
mod smart_router;
mod specs;

pub(crate) use dispatch::dispatch;
pub(crate) use specs::tool_specs;

// Re-export everything that was previously public from this module.
// These are used by lib.rs tests.
pub(crate) use smart_router::{
    canonicalize_route_model_id, smart_parent_model_for_agent,
};
pub use smart_router::{
    assess_agent_task, assess_turn_complexity, assess_turn_orchestration, smart_deep_tier_models,
    smart_deep_tier_models_for, smart_exec_swap, smart_setting_defaults, turn_has_write_intent,
    AgentTaskAssessment, DeepTierModelsSetting, SmartExecSwap, SmartSettingDefaults,
    TurnOrchestrationHint,
};
#[allow(unused_imports)]
pub(crate) use agent_tools::{
    agent_permission_policy, allowed_tools_for_subagent, cancel_and_salvage_agent,
    classify_lane_failure, execute_agent_with_parent_model_and_hooks, execute_agent_with_spawn,
    execute_agent_with_spawn_and_parent_model_and_hooks, final_assistant_text,
    normalize_subagent_type, persist_agent_terminal_state, push_output_block,
    wait_for_agent_completions_cancellable, wait_for_agent_completions_observed,
    wait_for_agent_completions_until_done, workflow_concurrency_limit,
    agent_activity_snapshot_by_id, AgentActivitySnapshot, AgentInput, AgentJob, AgentOutput,
    SubagentToolExecutor, AGENT_MODEL_ENV,
};
pub use agent_tools::{
    agent_store_dir, agent_worker_is_live, background_completion_matches_session,
    clear_background_agent, is_background_agent, mark_background_agent,
    reconcile_dead_agent_worker,
    notify_background_task_completion,
    loaded_custom_agents, LoadedCustomAgent,
    parent_session_belongs, provider_error_class_from_completion, provider_error_class_metadata,
    reap_orphaned_agents, register_agent_completion_channel,
    stop_running_agents_since,
    stop_running_agents_since_for_session, stop_running_agents_since_for_strict_session,
    wait_for_agent_completions, AgentCompletion, AGENT_STARVED_STATUS,
};
pub(crate) use audit::run_audit;
pub(crate) use config_tools::write_plan_artifact;
pub use config_tools::{
    execute_config, execute_enter_plan_mode, execute_exit_plan_mode, ConfigInput, ConfigOutput,
    ConfigValue, EnterPlanModeInput, ExitPlanModeInput, PlanModeOutput,
};
#[allow(unused_imports)]
pub(crate) use council::{
    CouncilInput, MAX_COUNCIL_CANDIDATES, MAX_COUNCIL_CANDIDATE_CHARS, MAX_COUNCIL_LLM_JUDGE_CALLS,
};
#[allow(unused_imports)]
pub(crate) use notebook_tools::{NotebookCellType, NotebookEditInput, NotebookEditMode};
pub(crate) use retrieve_output::{run_retrieve_tool_output, RetrieveToolOutputInput};
pub(crate) use session_recall::{run_session_recall, SessionRecallInput};
#[cfg(test)]
pub(crate) use skill_tools::execute_skill;
pub(crate) use skill_tools::{
    normalize_skill_slug, parse_skill_frontmatter_field, render_proposed_skill, write_atomic_new,
    write_atomic_replace, SkillDistillInput, SkillInput, SkillReviewInput,
};

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    epoch_seconds_now, mvp_tool_specs, to_pretty_json, GlobalToolRegistry, SearchableToolSpec,
    ToolContext, ToolError, ToolSpec,
};
use runtime::{lsp_client::LspRegistry, McpDegradedReport, PermissionMode, RuntimeHookConfig};

use super::{execute_tool_with_context, from_value, maybe_enforce_permission_check};
use runtime::permission_enforcer::PermissionEnforcer;

// --- Input structs (remaining in misc_tools) ---

#[derive(Debug, Deserialize)]
pub(crate) struct ToolSearchInput {
    pub query: String,
    pub max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SleepInput {
    pub duration_ms: u64,
    /// 런타임이 이미 비차단 `tokio::time::sleep` 으로 대기했음을 표시한다.
    /// live 디스패치(`conversation::run_turn`)는 `sleep_tool_execution_input`
    /// 이 이 플래그를 주입한 뒤 `tokio::time::sleep` 으로 먼저 잔다. set 이면
    /// `execute_sleep` 이 동기 `std::thread::sleep` 을 건너뛰어 ① 대기 2배,
    /// ② `block_in_place` 로 인한 spinner/elapsed freeze 를 막는다. 플래그가
    /// 없는 비-live 경로(테스트·직접 dispatch)에서는 종전대로 동기 슬립한다.
    #[serde(default, rename = "__zo_already_slept")]
    pub already_slept: bool,
}

/// Upper bound on a single `send_to_user` push, applied at the tool boundary
/// (the render block itself never truncates). A verbatim finding/diff can be
/// large, so the cap is generous; past it the message is truncated with a
/// trailing marker and the result flags `truncated: true` so the model knows.
pub(crate) const MAX_SEND_TO_USER_CHARS: usize = 16_000;

/// Input for `send_to_user` (and its legacy aliases `SendUserMessage` /
/// `Brief`). Only `message` is meaningful; the aliases' old `attachments` /
/// `status` fields are tolerated (serde ignores unknown keys) but no longer do
/// anything — attachments are a non-goal for the mid-run push.
#[derive(Debug, Deserialize)]
pub(crate) struct SendToUserInput {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub(crate) struct StructuredOutputInput(BTreeMap<String, Value>);

#[derive(Debug)]
pub(crate) struct AskUserQuestionInput {
    pub question: String,
    /// Short topic chip shown beside the prompt title (e.g. `Auth method`).
    pub header: Option<String>,
    /// Accepts both bare strings and `{label, description}` objects — see
    /// [`runtime::message_stream::QuestionOption`]'s `Deserialize`.
    pub options: Option<Vec<runtime::message_stream::QuestionOption>>,
    /// When `true` the user may pick several options and the answer is
    /// returned as a JSON array; the default (`false`) keeps the single-choice
    /// contract where `answer` is a lone string.
    pub multi_select: bool,
}

impl<'de> Deserialize<'de> for AskUserQuestionInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        normalize_ask_user_question(value).map_err(serde::de::Error::custom)
    }
}

/// Recover a well-formed `{question, header, options}` from the several shapes
/// models actually emit. Beyond the canonical flat form we tolerate:
/// - a harness-style `{ "questions": [ { … } ] }` envelope (take the first),
/// - the entire payload JSON-encoded into the `question` string
///   (`question = "{\"question\": …, \"options\": […]}"`),
/// - prose followed by a trailing JSON array of options dumped into `question`.
///
/// Top-level fields always win; recovered fields only fill gaps. The function is
/// fail-open: anything it cannot interpret is left as-is so a plain question
/// still renders.
fn normalize_ask_user_question(mut value: Value) -> Result<AskUserQuestionInput, String> {
    use runtime::message_stream::QuestionOption;

    // Unwrap a `{ "questions": [ {…} ] }` envelope to its first entry.
    if let Some(first) = value
        .get_mut("questions")
        .and_then(Value::as_array_mut)
        .and_then(|arr| arr.first_mut())
    {
        value = first.take();
    }

    let obj = value
        .as_object()
        .ok_or_else(|| "AskUserQuestion input must be a JSON object".to_string())?;

    let mut question = obj
        .get("question")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mut header = obj.get("header").and_then(Value::as_str).map(str::to_string);
    let mut options = obj
        .get("options")
        .and_then(|v| serde_json::from_value::<Vec<QuestionOption>>(v.clone()).ok())
        .filter(|opts| !opts.is_empty());
    // Tolerate both the canonical camelCase `multiSelect` and a snake_case
    // spelling; models emit either. Missing / non-bool defaults to single-select.
    let mut multi_select = bool_field(obj, "multiSelect").or_else(|| bool_field(obj, "multi_select"));

    // The whole payload JSON-encoded into the `question` string.
    if let Ok(inner) = serde_json::from_str::<Value>(question.trim()) {
        if let Some(inner_obj) = inner.as_object() {
            if let Some(inner_q) = inner_obj.get("question").and_then(Value::as_str) {
                if options.is_none() {
                    options = inner_obj
                        .get("options")
                        .and_then(|v| serde_json::from_value::<Vec<QuestionOption>>(v.clone()).ok())
                        .filter(|opts| !opts.is_empty());
                }
                if header.is_none() {
                    header = inner_obj
                        .get("header")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                if multi_select.is_none() {
                    multi_select = bool_field(inner_obj, "multiSelect")
                        .or_else(|| bool_field(inner_obj, "multi_select"));
                }
                question = inner_q.to_string();
            }
        }
    }

    // Prose followed by a trailing JSON array of options dumped into `question`.
    if options.is_none() {
        if let Some(idx) = question.find('[') {
            let (prose, tail) = question.split_at(idx);
            if let Ok(arr) = serde_json::from_str::<Vec<QuestionOption>>(tail.trim()) {
                if !arr.is_empty() {
                    options = Some(arr);
                    question = prose.trim().to_string();
                }
            }
        }
    }

    if question.trim().is_empty() {
        return Err("AskUserQuestion requires a non-empty `question`".to_string());
    }

    Ok(AskUserQuestionInput {
        question,
        header,
        options,
        multi_select: multi_select.unwrap_or(false),
    })
}

/// Read a boolean field, tolerating the stringy `"true"`/`"false"` some models
/// emit instead of a JSON bool. Returns `None` when the key is absent or the
/// value is neither a bool nor a recognisable boolean string.
fn bool_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<bool> {
    match obj.get(key)? {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct MemoryWriteInput {
    pub slug: String,
    pub summary: String,
    pub body: String,
    /// Write to the global per-project machine-local overlay (`memory.local/`)
    /// instead of the durable global project store (`memory/`). Defaults to the
    /// durable store. Recall merges both, so a local entry surfaces only on this
    /// machine.
    #[serde(default)]
    pub local: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RemoteTriggerInput {
    pub url: String,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub headers: Option<Value>,
    #[serde(default)]
    pub body: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TestingPermissionInput {
    pub action: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MonitorInput {
    pub process_id: Option<String>,
    pub command: Option<String>,
    #[serde(default = "default_monitor_lines")]
    pub lines: usize,
}

fn default_monitor_lines() -> usize {
    50
}

#[derive(Debug, Deserialize)]
pub(crate) struct SendMessageInput {
    pub to: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScheduleWakeupInput {
    #[serde(rename = "delaySeconds")]
    pub delay_seconds: f64,
    pub reason: String,
    pub prompt: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SyntheticOutputInput {
    pub tool_name: String,
    pub output: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SpawnMultiAgentInput {
    #[serde(deserialize_with = "deserialize_agents_lenient")]
    pub agents: Vec<Value>,
    /// Permit-before-spawn window: at most this many sub-agents run at once, so a
    /// flat fan-out can no longer create N OS threads eagerly (BUG-R11/D4). `None`
    /// uses the workflow execution window (`min(16, cores-2)`, the Claude Code
    /// model) — later members queue for a freed slot. Clamped to
    /// `[1, MAX_SPAWN_MULTI_AGENT_AGENTS]`. Lenient like `agents`: models emit
    /// `"3"` (string) or `3.0` (integral float) instead of a bare integer, and a
    /// strict parse rejected the whole fan-out call.
    #[serde(default, deserialize_with = "deserialize_option_usize_lenient")]
    pub concurrency: Option<usize>,
    /// Internal foreground session id stamped onto every spawned member
    /// manifest. Hidden from tool input; dispatch fills it from [`ToolContext`].
    #[serde(skip)]
    pub parent_session_id: Option<String>,
    /// Internal `tool_use` id of this fan-out call, copied onto every member's
    /// [`AgentInput`] so their manifests attribute to this call's transcript
    /// batch. Hidden from tool input; dispatch fills it from the smuggled
    /// `__zo_tool_call_id`.
    #[serde(skip)]
    pub tool_call_id: Option<String>,
    /// Internal parent-session MCP passthrough, copied onto every member's
    /// [`AgentInput`] (see that field). Hidden from tool input; dispatch fills
    /// it from [`ToolContext`].
    #[serde(skip)]
    pub mcp_passthrough: Option<crate::registry::McpPassthrough>,
    /// Parent's active permission mode, copied onto every member's
    /// [`AgentInput`] so the spawn clamp applies to fan-out members exactly
    /// like single `Agent` spawns. Hidden from tool input; dispatch fills it.
    #[serde(skip)]
    pub parent_permission_mode: Option<runtime::PermissionMode>,
}

/// Hard cap on `agents` per flat `SpawnMultiAgent` call — CC-scale queued
/// breadth (matches the workflow hard ceiling). The cap bounds how many
/// members one call may ENUMERATE; how many run at once is the spawn window
/// below, and real provider concurrency is bounded tighter still by the
/// adaptive per-provider rate governor.
pub(crate) const MAX_SPAWN_MULTI_AGENT_AGENTS: usize = 64;

/// Resolve the effective permit-before-spawn window. Defaults to the workflow
/// execution bound (`min(16, cores-2)`, the Claude Code model): a large
/// fan-out spawns that many members immediately and streams the rest in as
/// slots free up, instead of opening one OS thread per member eagerly
/// (BUG-R11/D4). An explicit `concurrency` request may tighten the window to 1
/// or widen it up to the hard cap — widening never raises real API concurrency,
/// which stays behind the adaptive per-provider governor either way.
pub(crate) fn effective_spawn_window(requested: Option<usize>, agent_count: usize) -> usize {
    let cap = MAX_SPAWN_MULTI_AGENT_AGENTS.min(agent_count.max(1));
    requested.map_or_else(
        || workflow_concurrency_limit().min(cap),
        |window| window.clamp(1, cap),
    )
}
/// Legacy foreground collection window used by compatibility call sites and
/// tests. Production `SpawnMultiAgent` result aggregation waits for terminal
/// completions instead of surfacing `still_running` after this window.
pub(crate) const SPAWN_MULTI_AGENT_WAIT_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// Deserialize `agents`, tolerating the stringified-JSON form some models
/// emit (`"[{…}]"`) instead of a real array. See [`coerce_agents`].
fn deserialize_agents_lenient<'de, D>(deserializer: D) -> Result<Vec<Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    coerce_agents(value).map_err(serde::de::Error::custom)
}

/// Coerce a raw `agents` JSON value into a list of agent objects.
///
/// Accepts, in increasing order of leniency:
/// - a real array (the schema-correct form),
/// - a single object (treated as a one-agent fan-out),
/// - a JSON-encoded string of either of the above — the failure seen in the
///   wild, where the model serialized the argument to a string and strict
///   serde rejected it with `invalid type: string, expected a sequence`.
fn coerce_agents(value: Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Array(items) => Ok(items),
        Value::Object(_) => Ok(vec![value]),
        Value::String(raw) => {
            let parsed: Value = crate::model_json::parse_model_json(&raw)
                .map_err(|err| format!("`agents` was a string but not valid JSON: {err}"))?;
            match parsed {
                Value::Array(items) => Ok(items),
                object @ Value::Object(_) => Ok(vec![object]),
                _ => Err("`agents` string must encode an array or object".to_string()),
            }
        }
        other => Err(format!(
            "`agents` must be an array of agent objects (got {})",
            json_type_name(&other)
        )),
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Deserialize an optional non-negative integer, tolerating the forms models
/// emit instead of a bare integer. Seen in the wild on `SpawnMultiAgent`'s
/// `concurrency` argument: the model sent `"3"` and strict serde rejected the
/// whole call with `invalid type: string "3", expected usize`.
fn deserialize_option_usize_lenient<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        None => Ok(None),
        Some(value) => coerce_optional_usize(&value).map_err(serde::de::Error::custom),
    }
}

/// Coerce a raw JSON value into an optional non-negative integer.
///
/// Accepts, in increasing order of leniency:
/// - a real non-negative integer (the schema-correct form),
/// - an integral float (`3.0`) — some models emit numbers that way,
/// - a numeric string (`"3"`, `" 4 "`, `"3.0"`) — the failure seen in the
///   wild, mirroring [`coerce_agents`],
/// - `null` or an empty/whitespace string, treated as unset.
fn coerce_optional_usize(value: &Value) -> Result<Option<usize>, String> {
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => number_as_usize(number)
            .map(Some)
            .ok_or_else(|| format!("expected a non-negative integer, got {number}")),
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            trimmed
                .parse::<usize>()
                .ok()
                .or_else(|| {
                    serde_json::from_str::<serde_json::Number>(trimmed)
                        .ok()
                        .as_ref()
                        .and_then(number_as_usize)
                })
                .map(Some)
                .ok_or_else(|| {
                    format!("expected a non-negative integer, got string {trimmed:?}")
                })
        }
        other => Err(format!(
            "expected a non-negative integer, got {}",
            json_type_name(other)
        )),
    }
}

/// A `serde_json::Number` as `usize` when it is an exact non-negative
/// integer — including the integral-float form (`3.0`) models sometimes emit.
/// Floats are round-tripped through their display form instead of an `as`
/// cast so nothing truncates silently.
fn number_as_usize(number: &serde_json::Number) -> Option<usize> {
    if let Some(int) = number.as_u64() {
        return usize::try_from(int).ok();
    }
    let float = number.as_f64()?;
    if float.fract() != 0.0 || !float.is_finite() || float < 0.0 {
        return None;
    }
    format!("{float:.0}").parse::<usize>().ok()
}

// --- Output structs (remaining in misc_tools) ---

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ToolSearchOutput {
    pub matches: Vec<String>,
    /// Full definitions (description + input schema) of the matched tools —
    /// the payload a deferred-tool lookup exists for. Empty when no match
    /// resolved to a registered definition.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub schemas: Vec<api::ToolDefinition>,
    pub query: String,
    pub normalized_query: String,
    #[serde(rename = "total_deferred_tools")]
    pub total_deferred_tools: usize,
    #[serde(rename = "pending_mcp_servers")]
    pub pending_mcp_servers: Option<Vec<String>>,
    #[serde(rename = "mcp_degraded", skip_serializing_if = "Option::is_none")]
    pub mcp_degraded: Option<McpDegradedReport>,
}

#[derive(Debug, Serialize)]
struct SleepOutput {
    duration_ms: u64,
    message: String,
}

#[derive(Debug, Serialize)]
struct StructuredOutputResult {
    data: String,
    structured_output: BTreeMap<String, Value>,
}

// --- run_* entry points ---

pub(crate) fn run_skill(input: SkillInput) -> Result<String, ToolError> {
    to_pretty_json(skill_tools::execute_skill(input)?)
}

pub(crate) fn run_skill_distill(input: &SkillDistillInput) -> Result<String, ToolError> {
    to_pretty_json(skill_tools::execute_skill_distill(input)?)
}

pub(crate) fn run_skill_review(input: &SkillReviewInput) -> Result<String, ToolError> {
    to_pretty_json(skill_tools::execute_skill_review(input)?)
}

const BACKGROUND_AGENT_NOTE: &str = "Agent is running in the background; you will be notified when \
    it finishes. Keep making progress on other independent work meanwhile — if it completes while \
    your turn is still running, its result reaches you mid-turn at your next tool boundary as a \
    task notification; if your turn has ended, it arrives as a follow-up message. Do NOT poll the \
    output file and do NOT idle-wait; when nothing is left but waiting, end your turn. To steer it \
    mid-run, or to follow up after it finishes without losing its context, call SendMessage with its \
    name (load the schema via ToolSearch if needed).";

fn running_background_agent_result(
    manifest: &AgentOutput,
    blocking_wait_timed_out: bool,
) -> Result<String, ToolError> {
    agent_tools::mark_background_agent(manifest.agent_id.clone());
    let note = if blocking_wait_timed_out {
        format!(
            "{BACKGROUND_AGENT_NOTE} The blocking wait timed out, but the agent keeps running."
        )
    } else {
        BACKGROUND_AGENT_NOTE.to_string()
    };
    to_pretty_json(json!({
        "agentId": manifest.agent_id,
        "name": manifest.name,
        "subagentType": manifest.subagent_type,
        "status": "running",
        "background": true,
        "outputFile": manifest.output_file,
        "note": note,
    }))
}

fn finish_blocking_agent_call(
    manifest: &AgentOutput,
    completion: Option<AgentCompletion>,
) -> Result<String, ToolError> {
    if completion
        .as_ref()
        .is_some_and(|completion| completion.status == "still_running")
    {
        // `notify_agent_completion` publishes every spawn through the existing
        // single-consumer channel. Marking this id is the only extra wiring a
        // detached spawn needs; `parent_session_id` already scopes the live
        // manifest consumed by the HUD and agents viewer.
        return running_background_agent_result(manifest, true);
    }
    to_pretty_json(match completion {
        Some(completion) => json!({
            "agentId": manifest.agent_id,
            "name": manifest.name,
            "subagentType": manifest.subagent_type,
            "status": completion.status,
            // Head/tail cap (background re-injection parity): the global
            // envelope truncation is a blind tail cut that loses exactly the
            // agent's conclusion.
            "result": completion
                .result
                .as_deref()
                .map(|r| core_types::text::elide_middle(r, 16_000)),
            "structured": completion.structured,
            "error": completion.error,
            "outputFile": manifest.output_file,
        }),
        // Defensive fallback for a backend that returns no completion at all:
        // hand back the output-file path and a live brief.
        None => json!({
            "agentId": manifest.agent_id,
            "name": manifest.name,
            "subagentType": manifest.subagent_type,
            "status": "still_running",
            "outputFile": manifest.output_file,
            "live": agent_tools::agent_live_brief(&manifest.manifest_file),
        }),
    })
}

pub(crate) fn run_agent(
    input: AgentInput,
    parent_model: Option<&str>,
    parent_lsp: Option<&LspRegistry>,
    hook_config: Option<&RuntimeHookConfig>,
) -> Result<String, ToolError> {
    // Background mode: detach and return AT spawn time so the main model can
    // keep working (and the user keep chatting) while the agent runs. The real
    // result is PUSHED back into the conversation by the foreground REPL when
    // the agent completes — the host marks this id via `mark_background_agent`;
    // a completion during a live turn is folded in mid-turn at the next
    // tool-result boundary (CC task-notification parity), and one arriving
    // while the host is idle re-injects as a fresh turn. Crucially this is
    // push-based, not poll-based: the `note` tells the model NOT to poll the
    // output file, which is exactly the sleep+cat polling the synchronous path
    // below was introduced to kill.
    if input.background.unwrap_or(false) {
        let manifest = agent_tools::execute_agent_with_parent_model_and_hooks(
            input,
            parent_model,
            parent_lsp,
            hook_config,
        )?;
        return running_background_agent_result(&manifest, false);
    }
    // Block until the sub-agent finishes and return its result inline — like
    // `SpawnMultiAgent` and Claude Code's `Task`. The old behavior returned at
    // spawn time (`status: "running"` + an output-file path), so the model had
    // to poll the file with `sleep`+`cat`; that polling multiplied foreground
    // requests and tripped the shared provider rate limit even though the
    // sub-agent runs on the same account quota.
    let (manifest, completion) =
        agent_tools::execute_agent_blocking(input, parent_model, parent_lsp, hook_config)?;
    finish_blocking_agent_call(&manifest, completion)
}

pub(crate) fn run_tool_search(
    input: &ToolSearchInput,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    to_pretty_json(execute_tool_search(input, ctx))
}

pub(crate) fn run_council(input: &CouncilInput) -> Result<String, ToolError> {
    to_pretty_json(council::execute_council(input)?)
}

pub(crate) fn run_notebook_edit(input: NotebookEditInput) -> Result<String, ToolError> {
    to_pretty_json(notebook_tools::execute_notebook_edit(input)?)
}

pub(crate) fn run_sleep(input: &SleepInput) -> Result<String, ToolError> {
    to_pretty_json(execute_sleep(input)?)
}

/// `send_to_user` (and legacy aliases): push verbatim content to the user
/// mid-run without ending the turn.
///
/// With a live channel the message becomes a `UserNotice` block and the tool
/// returns `{delivered:true}`; with no interactive surface (headless runs,
/// sub-agents whose fresh context carries no channel, or a channel that does
/// not support the push) it degrades to an inline echo — `{delivered:false,
/// message}` — so the content is never silently lost. A sub-agent thus reports
/// the text back to its parent instead of barging into the parent's TUI.
pub(crate) fn run_send_to_user(
    input: SendToUserInput,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    if input.message.trim().is_empty() {
        return Err(ToolError::InvalidInput("'message' must not be empty".into()));
    }

    // Cap at the input boundary (the render block never truncates). Truncate on
    // a char boundary so multi-byte content is never split mid-codepoint.
    let (message, truncated) = if input.message.chars().count() > MAX_SEND_TO_USER_CHARS {
        let kept: String = input.message.chars().take(MAX_SEND_TO_USER_CHARS).collect();
        (
            format!("{kept}\n\n[send_to_user: truncated at {MAX_SEND_TO_USER_CHARS} chars]"),
            true,
        )
    } else {
        (input.message, false)
    };

    match ctx
        .user_question_channel()
        .as_deref()
        .map(|channel| channel.send_to_user(&message))
    {
        Some(Ok(())) => to_pretty_json(json!({
            "delivered": true,
            "truncated": truncated,
            "info": "Pushed to the user; the turn continues.",
        })),
        // No channel, or a channel that cannot push: echo inline so the content
        // survives. Headless / sub-agent runs land here (fresh context, no TUI).
        Some(Err(_)) | None => to_pretty_json(json!({
            "delivered": false,
            "truncated": truncated,
            "message": message,
            "note": "no interactive surface; returned inline",
        })),
    }
}

pub(crate) fn run_synthetic_output(input: &SyntheticOutputInput) -> Result<String, ToolError> {
    to_pretty_json(json!({
        "tool_name": input.tool_name,
        "output": input.output,
        "injected": true,
    }))
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_spawn_multi_agent(
    input: &SpawnMultiAgentInput,
    parent_model: Option<&str>,
    parent_lsp: Option<&LspRegistry>,
    hook_config: Option<&RuntimeHookConfig>,
) -> Result<String, ToolError> {
    run_spawn_multi_agent_with_timeout_and_hooks(
        input,
        parent_model,
        parent_lsp,
        SPAWN_MULTI_AGENT_WAIT_TIMEOUT,
        None,
        false,
        hook_config,
    )
}

#[allow(dead_code, clippy::too_many_lines)]
pub(crate) fn run_spawn_multi_agent_with_timeout(
    input: &SpawnMultiAgentInput,
    parent_model: Option<&str>,
    parent_lsp: Option<&LspRegistry>,
    wait_timeout: Duration,
    agent_time_budget: Option<Duration>,
    workflow_member: bool,
) -> Result<String, ToolError> {
    run_spawn_multi_agent_with_timeout_and_hooks(
        input,
        parent_model,
        parent_lsp,
        wait_timeout,
        agent_time_budget,
        workflow_member,
        None,
    )
}

#[allow(clippy::too_many_lines)]
pub(crate) fn run_spawn_multi_agent_with_timeout_and_hooks(
    input: &SpawnMultiAgentInput,
    parent_model: Option<&str>,
    parent_lsp: Option<&LspRegistry>,
    wait_timeout: Duration,
    agent_time_budget: Option<Duration>,
    workflow_member: bool,
    hook_config: Option<&RuntimeHookConfig>,
) -> Result<String, ToolError> {
    if input.agents.is_empty() {
        return Err(ToolError::InvalidInput(
            "agents list must not be empty".to_string(),
        ));
    }
    if input.agents.len() > MAX_SPAWN_MULTI_AGENT_AGENTS {
        return Err(ToolError::InvalidInput(format!(
            "SpawnMultiAgent accepts at most {MAX_SPAWN_MULTI_AGENT_AGENTS} agents (got {})",
            input.agents.len()
        )));
    }

    let mut spawned: Vec<Value> = Vec::with_capacity(input.agents.len());
    let mut errors: Vec<Value> = Vec::new();
    let mut name_by_id: BTreeMap<String, String> = BTreeMap::new();
    // Phase 4 verdict channel — source #2: the inverse of `name_by_id`,
    // populated as each member spawns (see below), so a LATER member in the
    // batch whose `__zo_route_judged_agent` smuggle names an EARLIER
    // member can resolve that name to a real agent id. A worker spawned in a
    // later chunk than its reviewer is unresolvable (`None`) by construction —
    // the ordering constraint this map encodes.
    let mut agent_id_by_name: BTreeMap<String, String> = BTreeMap::new();
    let mut completions: Vec<AgentCompletion> = Vec::new();

    // Auto worktree-isolation (tracks 3-3 + 4-3): when the workspace guard is
    // opt-in enabled and this is a real multi-agent fan-out, place each agent in
    // its own git worktree so parallel (and heterogeneous cross-provider) editors
    // never clobber the shared tree, then merge each change-set back after the
    // batch barrier. A git failure leaves the provider unset → the fan-out runs
    // un-isolated (honest fallback), exactly as the workflow engine degrades.
    let auto_isolate =
        fanout_isolation::should_auto_isolate(&input.agents, crate::workspace_guard_enabled());
    let worktree_provider: Option<crate::workflow_tools::worktree::GitWorktreeProvider> =
        auto_isolate
            .then(crate::workflow_tools::worktree::GitWorktreeProvider::new)
            .and_then(Result::ok);
    let heterogeneous = auto_isolate && fanout_isolation::is_heterogeneous(&input.agents);
    let mut isolated_count = 0usize;

    // Permit-before-spawn windowing (BUG-R11/D4): spawn at most `window` agents,
    // then, the instant one finishes, refill its slot with the next pending
    // agent — so a flat fan-out never holds more than `window` live agent
    // threads, yet a slow agent never delays the START of a sibling while a slot
    // is free (no chunk barrier). With the default window (all at once, up to the
    // cap) every agent spawns before the first wait, exactly as before; a smaller
    // `concurrency` opts into tighter bounding without the head-of-line stall.
    let window = effective_spawn_window(input.concurrency, input.agents.len());
    // Ids currently in flight (bounded by `window`); a completed id is removed as
    // its slot is reclaimed. Worktree guards accumulate across the whole run and
    // are merged back once, in spawn order, after every agent has reached a
    // terminal/cancelled state — the flat analog of the per-window teardown,
    // preserving merge ordering.
    let mut in_flight: Vec<String> = Vec::with_capacity(window);
    let mut all_guards: Vec<(
        String,
        Box<dyn crate::workflow_tools::worktree::WorktreeGuard>,
    )> = Vec::new();
    // ONE overall deadline for the whole fan-out, not one per reclaim: a rolling
    // window over N agents reclaims `N - window` times, so a per-call
    // `wait_timeout` would let the total collection budget grow with the agent
    // count. Every reclaim and the final drain share this single deadline, so the
    // public collection budget stays `wait_timeout` regardless of fan-out width.
    let overall_deadline = std::time::Instant::now() + wait_timeout;
    {
        for (idx, agent_val) in input.agents.iter().enumerate() {
            // Never spawn a fresh worker once the overall deadline has elapsed:
            // doing so could push the live-worker count over `window` (a slot
            // freed by a *synthetic* terminal may still back a physically-live
            // worker) and would burn budget past the public collection deadline.
            // Instead record an explicit, ordered timeout error for this input so
            // result cardinality and input order are preserved (aggregation keys
            // both `spawned` and `errors` by `index`).
            if std::time::Instant::now() >= overall_deadline {
                errors.push(serde_json::json!({
                    "index": idx,
                    "error": "fan-out overall deadline elapsed before this agent was spawned",
                }));
                continue;
            }
            // Reclaim a slot before opening the next one: block (up to the shared
            // overall deadline) until at least one in-flight agent reaches a
            // terminal completion, drain those into `completions`, and free their
            // slots. If the deadline passes with nothing terminal, the oldest
            // agent is cancelled+salvaged to a terminal state BEFORE its slot is
            // reused — so a freed slot never leaves a live worker running, and the
            // live worker count is always <= `window`. Only engages once `window`
            // agents are already live, so the first `window` agents all spawn
            // back-to-back with no wait between them.
            if in_flight.len() >= window {
                reclaim_spawn_slots(&mut in_flight, &mut completions, overall_deadline);

                // `reclaim_spawn_slots` may have blocked all the way to the
                // overall deadline before cancelling the oldest agent. Two hazards
                // remain that the top-of-loop check could not see, because they
                // only become true *after* that blocking wait:
                //
                //   1. The deadline may now have elapsed — spawning here would
                //      burn budget past the public collection deadline.
                //   2. The reclaimed slot was freed by a *cooperative* cancel, so
                //      that worker may still be physically live (its worktree is
                //      still owned by a live editor). Spawning a fresh worker now
                //      would push the real live-worker count over `window`.
                //
                // Recheck both against the authoritative physical-liveness signal
                // and, if either still bars a spawn, record an ordered timeout
                // error for this input instead of spawning — preserving input
                // order and result cardinality (aggregation keys by `index`).
                if spawn_barred_after_reclaim(&in_flight, window, overall_deadline) {
                    errors.push(serde_json::json!({
                        "index": idx,
                        "error": "fan-out overall deadline elapsed before a worker slot became free",
                    }));
                    continue;
                }
            }
            let prompt = agent_val
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let subagent_type = agent_val
                .get("subagent_type")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let description = agent_val
                .get("description")
                .and_then(|v| v.as_str())
                .map_or_else(|| format!("parallel-agent-{idx}"), str::to_string);
            let name = agent_val
                .get("name")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            // Per-agent model override (BUG-D5): the model can route each
            // sub-agent explicitly instead of every agent sharing one model.
            // Honored by `resolve_agent_model_selection`; `None` inherits the
            // parent model (CC parity).
            let model = agent_val
                .get("model")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            // Explicit, transcript-visible escape hatch for a cross-provider
            // `model` — reserved for when the user asked for that model (see
            // `AgentInput::allow_cross_provider`).
            let allow_cross_provider = agent_val
                .get("allow_cross_provider")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            // Smart-router WHY, smuggled next to the injected "model" by
            // `apply_smart_models_to_spawn_input` — stamped onto the manifest
            // so the TUI can show the routing decision.
            let route_reason = agent_val
                .get(smart_router::ROUTE_REASON_SMUGGLE_KEY)
                .and_then(|v| v.as_str())
                .map(str::to_string);
            // Resolved smart-route model, smuggled alongside the reason.
            // SECURITY: trusted here ONLY because `apply_smart_models_to_spawn_input`
            // runs first on this same input and scrubs any caller-supplied
            // `__zo_route_model` before (re)inserting a host-computed value —
            // so this is never an untrusted craft. The spawn path then honors it
            // verbatim (already gated to the connected inventory by `route_model`),
            // including a deliberate cross-provider route for a diversity role.
            // `None` inherits the parent model.
            let route_model = agent_val
                .get(smart_router::ROUTE_MODEL_SMUGGLE_KEY)
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let route_fallback_models = agent_val
                .get(smart_router::ROUTE_FALLBACK_MODELS_SMUGGLE_KEY)
                .and_then(|v| v.as_array())
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| value.as_str())
                        .map(str::trim)
                        .filter(|model| !model.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            // Recommended effort tier, smuggled alongside the model/reason.
            // SECURITY: trusted here for the same reason as `route_model` —
            // `apply_smart_models_to_spawn_input` scrubs any caller-supplied
            // `__zo_route_effort` before (re)inserting a host-computed
            // value. `EffortLevel` deserializes from its lowercase wire token
            // (`#[serde(rename_all = "lowercase")]`), so this parses the exact
            // string `apply.rs` serialized. `None` = no recommendation.
            let route_effort = agent_val
                .get(smart_router::ROUTE_EFFORT_SMUGGLE_KEY)
                .cloned()
                .and_then(|value| serde_json::from_value::<api::EffortLevel>(value).ok());
            // P3 v2 route-decision metadata (role/complexity/risk/routeSource),
            // smuggled as one JSON object alongside the other route keys.
            // SECURITY: trusted here for the same reason as `route_model` —
            // `apply_smart_models_to_spawn_input` scrubs any caller-supplied
            // `__zo_route_decision_meta` before (re)inserting a
            // host-computed value. Purely descriptive (stamped onto the
            // manifest / route-outcome record, never re-gates a trust
            // boundary), so a missing/malformed field just stays absent.
            let route_decision_meta = agent_val.get(smart_router::ROUTE_DECISION_META_SMUGGLE_KEY);
            let route_role = route_decision_meta
                .and_then(|value| value.get("role"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let route_complexity = route_decision_meta
                .and_then(|value| value.get("complexity"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let route_risk = route_decision_meta
                .and_then(|value| value.get("risk"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let route_source = route_decision_meta
                .and_then(|value| value.get("routeSource"))
                .and_then(Value::as_str)
                .map(str::to_string);
            // Phase 4 verdict channel — source #2: resolve the planner-bound
            // judged worker's NAME (smuggled by
            // `apply_smart_models_to_spawn_input`) to that worker's real
            // agent id via `agent_id_by_name`, populated as EARLIER members
            // in this same batch spawn (see below). Unresolvable (worker not
            // yet spawned, or the smuggle key is absent) stays `None` —
            // ambiguous binding must never guess.
            let judged_agent = agent_val
                .get(smart_router::ROUTE_JUDGED_AGENT_SMUGGLE_KEY)
                .and_then(Value::as_str)
                .and_then(|worker_name| agent_id_by_name.get(worker_name).cloned());

            if prompt.is_empty() {
                errors.push(json!({"index": idx, "error": "prompt is empty"}));
                continue;
            }

            // Per-agent isolated worktree, when auto-isolation engaged. A failed
            // `create` degrades just this agent to the shared cwd (counted for an
            // honest note), never aborting the fan-out.
            let cwd = worktree_provider.as_ref().and_then(|provider| {
                use crate::workflow_tools::worktree::WorktreeProvider as _;
                provider
                    .create(name.as_deref().unwrap_or("agent"))
                    .ok()
                    .map(|guard| {
                        isolated_count += 1;
                        let path = guard.path().to_path_buf();
                        // The guard is parked until the run's end merge-back; its
                        // id is filled in once the spawn returns an agent id.
                        all_guards.push((String::new(), guard));
                        path
                    })
            });
            let guard_slot = cwd.is_some().then(|| all_guards.len() - 1);
            // Captured before `name` moves into `agent_input` below, so a
            // LATER sibling's judged-agent smuggle can resolve to THIS
            // member's agent id once the spawn below succeeds.
            let name_for_judged_lookup = name.clone();

            let agent_input = agent_tools::AgentInput {
                allow_cross_provider,
                description,
                prompt,
                subagent_type,
                name,
                model,
                cwd,
                schema: None,
                workflow_member,
                // Fan-out members are always collected synchronously; background
                // detach is a single-`Agent`-tool affordance only.
                background: Some(false),
                parent_permission_mode: input.parent_permission_mode,
                parent_session_id: input.parent_session_id.clone(),
                tool_call_id: input.tool_call_id.clone(),
                mcp_passthrough: input.mcp_passthrough.clone(),
                // The per-call `concurrency` now also caps real provider-request
                // concurrency via the adaptive governor, not just OS-thread spawn
                // windowing — so a tighter value genuinely throttles the API.
                api_concurrency: input.concurrency,
                time_budget: agent_time_budget,
                prior_failures: 0,
                route_reason,
                route_model,
                route_fallback_models,
                route_effort,
                route_role,
                route_complexity,
                route_risk,
                route_source,
                judged_agent,
            };
            match agent_tools::execute_agent_with_parent_model_and_hooks(
                agent_input,
                parent_model,
                parent_lsp,
                hook_config,
            ) {
                Ok(output) => {
                    let display_name = output.label.as_deref().unwrap_or(&output.name).to_string();
                    // Tag the parked worktree guard with this agent's id so the
                    // end-of-run merge-back attributes its patch correctly.
                    if let Some(slot) = guard_slot {
                        all_guards[slot].0.clone_from(&output.agent_id);
                    }
                    in_flight.push(output.agent_id.clone());
                    name_by_id.insert(output.agent_id.clone(), display_name.clone());
                    // Phase 4 verdict channel — source #2: make THIS member's
                    // agent id resolvable by name for any LATER sibling's
                    // judged-agent smuggle in the same batch.
                    if let Some(worker_name) = name_for_judged_lookup {
                        agent_id_by_name.insert(worker_name, output.agent_id.clone());
                    }
                    spawned.push(json!({
                        "index": idx,
                        "agentId": output.agent_id,
                        "name": output.name,
                        "label": display_name,
                    }));
                }
                Err(e) => {
                    errors.push(json!({"index": idx, "error": e.to_string()}));
                }
            }
        }
        // Drain the agents still in flight after the last spawn, then guarantee
        // every one of them has reached a terminal/cancelled state *before* any
        // worktree is collected or dropped below. Waiting is bounded by the same
        // shared overall deadline as the reclaims, so the collection budget never
        // grows with the agent count; any agent still live at the deadline is
        // cancelled+salvaged here (not merely observed as `still_running`), so no
        // worktree merge-back/drop can race a live editor and tear a patch.
        drain_or_cancel_remaining_agents(&mut in_flight, &mut completions, overall_deadline);
        // Merge each isolated agent's change-set back into the main tree in spawn
        // order (same 3-way apply the workflow engine uses), then drop the guards
        // to tear down every worktree at once — but ONLY for workers that have
        // physically exited. A cooperative cancel writes a terminal `stopped`
        // manifest immediately while the worker keeps running until it observes
        // the abort (see `cancel_and_salvage_agent`), so the *generation-bound*
        // `agent_worker_generation_is_live` (backed by the still-registered cancel
        // signal for this exact run generation) is the real exit ack. Collecting
        // or dropping a still-live worker's worktree here would race a live editor
        // and remove a live worktree; those are handed to a background owner that
        // tears them down only after that exact worker generation actually exits,
        // and never merges their late change-set back (quarantine, not apply).
        if let Some(provider) = worktree_provider.as_ref() {
            let mut ready: Vec<(String, Box<dyn crate::workflow_tools::worktree::WorktreeGuard>)> =
                Vec::new();
            for (agent_id, guard) in std::mem::take(&mut all_guards) {
                if !agent_id.is_empty()
                    && agent_tools::agent_worker_generation_is_live(
                        &agent_id,
                        agent_tools::AGENT_INITIAL_RUN_GENERATION,
                    )
                {
                    spawn_deferred_worktree_cleanup(
                        agent_id,
                        agent_tools::AGENT_INITIAL_RUN_GENERATION,
                        guard,
                        DEFERRED_WORKTREE_CLEANUP_POLL,
                    );
                } else {
                    ready.push((agent_id, guard));
                }
            }
            merge_back_fanout_worktrees(provider, &ready, &mut errors);
        }
        drop(all_guards);
    }

    // 결과 합치기 — index 순서 유지를 위해 spawn 시 받은 idx 와 매칭.
    let mut completed_count = 0usize;
    let mut failed_count = 0usize;
    let mut stopped_count = 0usize;
    let mut timed_out_count = 0usize;
    // Fair share of the global tool-output envelope per agent, head/tail
    // preserved: the global cap is a blind tail cut on the ASSEMBLED JSON, so
    // without a per-agent budget one verbose early agent evicted every later
    // agent's result mid-JSON. The floor keeps a wide fan-out from squeezing
    // each agent below usefulness; the ceiling matches the background
    // re-injection cap.
    let per_agent_result_budget = (runtime::TruncationConfig::default().default_max_chars
        / spawned.len().max(1))
    .clamp(3_000, 16_000);
    let agents_results: Vec<Value> = spawned
        .iter()
        .map(|s| {
            let agent_id = s
                .get("agentId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            // Only a terminal completion (completed/failed/stopped) is a real
            // result. A `still_running` placeholder means the collection window
            // elapsed before the worker finished — spawned fan-out agents have
            // no wall-clock deadline of their own, so left alone they run to the
            // iteration/tool-call cap while the parent gets nothing back.
            let terminal = completions.iter().find(|c| {
                c.agent_id == agent_id && agent_tools::agent_output_status_is_terminal(&c.status)
            });
            let (status, result, error) = if let Some(c) = terminal {
                match c.status.as_str() {
                    "completed" => completed_count += 1,
                    "failed" => failed_count += 1,
                    _ => stopped_count += 1,
                }
                (
                    c.status.clone(),
                    c.result.as_deref().map(|r| core_types::text::elide_middle(r, per_agent_result_budget)),
                    c.error.clone(),
                )
            } else {
                // Cancel the overran worker (cooperative stop) and salvage
                // whatever it has streamed (`outputTail`) as a partial result,
                // instead of discarding all its work behind a bare
                // `still_running` with no result. Keep the cancel signal
                // registered: a background worktree-cleanup owner may already be
                // gating teardown on this worker's physical exit, and
                // unregistering here would zo a premature exit ack.
                timed_out_count += 1;
                let partial = agent_tools::cancel_and_salvage_agent_keep_worker_registered(
                    &agent_id,
                    "agent exceeded spawn collection timeout",
                );
                ("timed_out".to_string(), partial, None)
            };
            // A timed-out agent carries a live brief (phase / current tool /
            // heartbeat) so the summary says how far it got.
            let live = (status == "timed_out")
                .then(|| agent_tools::agent_live_brief_by_id(&agent_id))
                .flatten();
            json!({
                "index": s.get("index"),
                "agentId": agent_id,
                "name": s.get("name"),
                "status": status,
                "result": result,
                "error": error,
                "live": live,
            })
        })
        .collect();

    let mut summary = json!({
        "status": "completed",
        "spawned_count": spawned.len(),
        "completed": completed_count,
        "failed": failed_count,
        "stopped": stopped_count,
        "timed_out": timed_out_count,
        "error_count": errors.len(),
        "agents": agents_results,
        "errors": errors,
    });
    // Surface auto worktree-isolation (tracks 3-3 + 4-3) so the run is honest
    // about whether parallel editors were sandboxed and merged back. Omitted
    // entirely when isolation did not engage, keeping the common output stable.
    if isolated_count > 0 {
        summary["isolation"] = json!("worktree");
        summary["isolated_agents"] = json!(isolated_count);
        summary["heterogeneous"] = json!(heterogeneous);
    }
    // When the collection window elapsed before an agent finished, we cancelled
    // it and recovered its partial streamed output (if any) into `result`.
    if timed_out_count > 0 {
        summary["note"] = json!(
            "Some agents did not finish within the collection window; they were \
             cancelled and any partial streamed output was recovered into `result`."
        );
    }
    to_pretty_json(summary)
}

fn wait_for_spawned_agent_completions(
    agent_ids: &[String],
    wait_timeout: Duration,
) -> Vec<AgentCompletion> {
    if agent_ids.is_empty() {
        Vec::new()
    } else {
        agent_tools::wait_for_agent_completions(agent_ids, wait_timeout)
    }
}

/// Final drain for the live fan-out scheduler: bring **every** still-in-flight
/// agent to a terminal/cancelled state before the caller collects or drops any
/// worktree.
///
/// First it waits (bounded by the shared overall `deadline`, so the total budget
/// never grows with fan-out width) for agents that finish on their own, draining
/// their terminal completions and clearing them from `in_flight`. Any agent still
/// live when the deadline passes is then cancelled+salvaged via
/// [`reclaim_cancel_and_collect`], which persists a terminal record and drains
/// it. On return `in_flight` is empty and no listed agent has a live editor, so
/// the subsequent per-agent worktree `collect_patch`/`drop` cannot race a writer
/// and tear a patch. Result cardinality/order is unaffected: aggregation iterates
/// the spawned set, and each agent now carries a real terminal completion.
fn drain_or_cancel_remaining_agents(
    in_flight: &mut Vec<String>,
    completions: &mut Vec<AgentCompletion>,
    deadline: std::time::Instant,
) {
    if in_flight.is_empty() {
        return;
    }
    let remaining = deadline
        .saturating_duration_since(std::time::Instant::now())
        .max(Duration::ZERO);
    let observed = wait_for_spawned_agent_completions(in_flight, remaining);
    let terminal: std::collections::HashSet<String> = observed
        .iter()
        .filter(|c| agent_tools::agent_output_status_is_terminal(&c.status))
        .map(|c| c.agent_id.clone())
        .collect();
    if !terminal.is_empty() {
        completions.extend(
            observed
                .into_iter()
                .filter(|c| terminal.contains(&c.agent_id)),
        );
        in_flight.retain(|id| !terminal.contains(id));
    }
    // Anything still live at the deadline is driven terminal here so its worktree
    // is safe to collect. Drain the vector so callers see an empty in-flight set.
    for agent_id in std::mem::take(in_flight) {
        reclaim_cancel_and_collect(&agent_id, completions);
    }
}

/// Poll slice used when waiting for *any one* in-flight agent to finish. Small
/// enough that a slot is reclaimed promptly after a completion lands, large
/// enough not to busy-spin. The overall wait is still bounded by `wait_timeout`.
const SPAWN_SLOT_RECLAIM_POLL: Duration = Duration::from_millis(25);

/// Count the fan-out members that are still *physically* live.
///
/// `in_flight` is the scheduler's slot bookkeeping: an id is removed the moment a
/// slot is reclaimed, even when the reclaim was a *cooperative* cancel whose
/// worker has not yet observed the abort and is still editing its worktree. So
/// `in_flight.len()` undercounts real live workers. This counts the ids whose
/// exact initial-generation worker is still registered (the authoritative
/// physical-liveness signal), which is what the window cap must be measured
/// against before spawning a fresh worker.
fn live_worker_count(in_flight: &[String]) -> usize {
    in_flight
        .iter()
        .filter(|id| {
            agent_tools::agent_worker_generation_is_live(
                id,
                agent_tools::AGENT_INITIAL_RUN_GENERATION,
            )
        })
        .count()
}

/// Whether opening a fresh worker slot is barred right after a reclaim attempt.
///
/// `reclaim_spawn_slots` may block to the overall deadline and then free a slot
/// via a *cooperative* cancel whose worker is still physically live. Spawning
/// then would either burn budget past the public collection deadline or push the
/// real live-worker count over `window` (the reclaimed slot's worker still holds
/// a registered signal). This is the exact decision the execute loop makes after
/// reclaiming; extracted so it can be exercised deterministically. A barred input
/// becomes an ordered timeout error instead of a spawn.
fn spawn_barred_after_reclaim(
    in_flight: &[String],
    window: usize,
    overall_deadline: std::time::Instant,
) -> bool {
    std::time::Instant::now() >= overall_deadline || live_worker_count(in_flight) >= window
}

/// Reclaim at least one in-flight slot for the live fan-out scheduler.
///
/// Blocks until *one or more* of the `in_flight` agents reaches a terminal
/// completion (unlike [`wait_for_spawned_agent_completions`], which drains the
/// whole set): the finished completions are moved into `completions` and their
/// ids are removed from `in_flight`, freeing the slot(s) so the caller can start
/// the next pending agent immediately. This is what turns the old chunk barrier
/// into a rolling window — a slow sibling no longer blocks the next agent's start
/// while a slot is free.
///
/// If the shared `deadline` elapses with nothing terminal, the oldest agent is
/// **cancelled and salvaged to a terminal state before its slot is freed**: the
/// cancel drives that agent to a persisted terminal record, its completion is
/// drained into `completions`, and only then is its id removed from `in_flight`.
/// A freed slot therefore never leaves a live worker running, so the live worker
/// count stays at or below the window, and the reclaimed agent's worktree is
/// safe to collect (its editor is no longer live). `deadline` is the single
/// overall fan-out deadline shared by every reclaim and the final drain, so the
/// total collection budget does not grow with the agent count.
fn reclaim_spawn_slots(
    in_flight: &mut Vec<String>,
    completions: &mut Vec<AgentCompletion>,
    deadline: std::time::Instant,
) {
    if in_flight.is_empty() {
        return;
    }
    loop {
        let slice = SPAWN_SLOT_RECLAIM_POLL.min(
            deadline
                .saturating_duration_since(std::time::Instant::now())
                .max(Duration::ZERO),
        );
        let observed = agent_tools::wait_for_agent_completions(in_flight, slice);
        let terminal: std::collections::HashSet<String> = observed
            .iter()
            .filter(|c| agent_tools::agent_output_status_is_terminal(&c.status))
            .map(|c| c.agent_id.clone())
            .collect();
        if !terminal.is_empty() {
            completions.extend(
                observed
                    .into_iter()
                    .filter(|c| terminal.contains(&c.agent_id)),
            );
            in_flight.retain(|id| !terminal.contains(id));
            return;
        }
        if std::time::Instant::now() >= deadline {
            // Nothing finished within the shared budget; reclaim the oldest slot
            // by driving its agent to a terminal manifest state. The cancel is
            // cooperative, so the physical worker may still be running after this
            // returns; we deliberately keep its cancel signal registered so
            // `agent_worker_is_live` still reports it live and its worktree
            // teardown is deferred until it actually exits. Freeing the scheduler
            // slot here does not breach `live workers <= window`: reclaim only
            // ever runs before the overall deadline, and once the deadline
            // elapses the spawn loop starts no further workers, so the count can
            // only fall.
            let oldest = in_flight.remove(0);
            reclaim_cancel_and_collect(&oldest, completions);
            return;
        }
    }
}

/// Cancel/salvage a single agent whose slot is being reclaimed under deadline
/// pressure, then move its now-terminal completion into `completions`.
///
/// [`agent_tools::cancel_and_salvage_agent_keep_worker_registered`] cooperatively
/// cancels the agent and persists a terminal record (returning real output
/// untouched if the agent had already finished on its own in the race window),
/// but leaves the cancel signal registered so `agent_worker_is_live` keeps
/// reporting the physical worker live until it truly exits — which is what gates
/// deferred worktree teardown. We then drain that terminal completion so the
/// reclaimed agent carries a real terminal status into result aggregation
/// instead of being re-cancelled there. A short bounded wait covers the
/// persist/notify hop; if the record is somehow still not visible we leave the
/// id undrained and aggregation's own straggler path handles it — cardinality
/// and order are preserved either way because aggregation iterates the spawned
/// set, not `completions`.
fn reclaim_cancel_and_collect(agent_id: &str, completions: &mut Vec<AgentCompletion>) {
    let _ = agent_tools::cancel_and_salvage_agent_keep_worker_registered(
        agent_id,
        "fan-out reclaimed this slot before the agent reached a terminal state",
    );
    let drained = agent_tools::wait_for_agent_completions(
        std::slice::from_ref(&agent_id.to_string()),
        SPAWN_SLOT_RECLAIM_POLL,
    );
    if let Some(terminal) = drained
        .into_iter()
        .find(|c| c.agent_id == agent_id && agent_tools::agent_output_status_is_terminal(&c.status))
    {
        completions.push(terminal);
    }
}

/// Pure fan-out schedule shared with the live scheduler above: start agents in
/// input order, holding at most `window` in flight, and the instant a slot frees
/// (`reclaim`) start the next pending agent into it. Extracted so the ordering
/// invariants — never more than `window` concurrent, and a free slot filled
/// without waiting on unrelated siblings — can be exercised deterministically
/// without real provider threads. `spawn(idx)` returns the started id; `reclaim`
/// removes one or more finished ids from the live set and is only called when the
/// window is full.
#[cfg(test)]
fn drive_fanout_schedule(
    agent_count: usize,
    window: usize,
    mut spawn: impl FnMut(usize, &[String]) -> String,
    mut reclaim: impl FnMut(&mut Vec<String>),
) -> Vec<String> {
    let window = window.max(1);
    let mut in_flight: Vec<String> = Vec::with_capacity(window);
    let mut order: Vec<String> = Vec::with_capacity(agent_count);
    for idx in 0..agent_count {
        if in_flight.len() >= window {
            reclaim(&mut in_flight);
        }
        let id = spawn(idx, &in_flight);
        in_flight.push(id.clone());
        order.push(id);
    }
    order
}

/// Poll slice for the background worktree-cleanup owner while it waits for a
/// still-live worker to physically exit. Matches the reclaim poll cadence.
const DEFERRED_WORKTREE_CLEANUP_POLL: Duration = Duration::from_millis(25);

/// A worktree guard parked in the process-global quarantine, tagged with the
/// exact `(agent_id, generation)` whose physical exit must be observed before the
/// guard may be dropped.
type QuarantinedGuard = (
    String,
    u64,
    Box<dyn crate::workflow_tools::worktree::WorktreeGuard>,
);

/// Process-global quarantine for still-live worktree guards whose dedicated
/// cleanup thread could not be spawned.
///
/// The normal path is a per-guard cleanup thread that drops the guard only after
/// its exact worker generation exits. If `thread::Builder::spawn` *fails* (thread
/// limit reached, etc.), letting the closure drop would tear down a **live**
/// worktree — so the caller instead parks the guard here and a single shared
/// quarantine owner drops it only after the same exact-generation exit ack, or
/// safely retains it until process exit. Never applies patches back.
static QUARANTINED_WORKTREE_GUARDS: OnceLock<std::sync::Mutex<Vec<QuarantinedGuard>>> =
    OnceLock::new();

fn quarantined_worktree_guards() -> &'static std::sync::Mutex<Vec<QuarantinedGuard>> {
    QUARANTINED_WORKTREE_GUARDS.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// Test-only seam: force the next `spawn_deferred_worktree_cleanup` thread spawn
/// to fail, exercising the quarantine handoff without exhausting real OS threads.
#[cfg(test)]
static FORCE_DEFERRED_CLEANUP_SPAWN_FAILURE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Guards the single quarantine drainer thread: true while a drainer is running,
/// so concurrent quarantine calls do not spawn a second one.
static QUARANTINE_DRAINER_RUNNING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Park a still-live worktree guard in the process-global quarantine and ensure a
/// single shared owner thread is draining it. Used when the per-guard cleanup
/// thread could not be spawned; the guard is dropped only after its exact
/// `(agent_id, generation)` worker exits, never while still live.
fn quarantine_worktree_guard(
    agent_id: String,
    generation: u64,
    guard: Box<dyn crate::workflow_tools::worktree::WorktreeGuard>,
    poll: Duration,
) {
    use std::sync::atomic::Ordering;

    quarantined_worktree_guards()
        .lock()
        .expect("worktree quarantine lock")
        .push((agent_id, generation, guard));

    // Ensure exactly one drainer runs: win the gate before spawning. If another
    // quarantine call already owns the drainer, this parked guard is picked up by
    // that running drainer's next sweep, so no second thread is needed.
    if QUARANTINE_DRAINER_RUNNING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }

    // Best-effort: start the single shared drainer. If even this spawn fails,
    // release the gate and leave the guard safely parked in the static (never
    // dropped while live) until a later quarantine call starts a drainer or the
    // process exits.
    let spawned = std::thread::Builder::new()
        .name("zo-fanout-worktree-quarantine".to_string())
        .spawn(move || loop {
            {
                let mut parked = quarantined_worktree_guards()
                    .lock()
                    .expect("worktree quarantine lock");
                let mut still_parked: Vec<QuarantinedGuard> = Vec::new();
                for (id, gen, guard) in std::mem::take(&mut *parked) {
                    if agent_tools::agent_worker_generation_is_live(&id, gen) {
                        still_parked.push((id, gen, guard));
                    } else {
                        // Exact-generation exit ack observed: safe to tear down.
                        drop(guard);
                    }
                }
                *parked = still_parked;
                if parked.is_empty() {
                    // Release the gate while still holding the quarantine lock, so
                    // it is atomic with respect to `quarantine_worktree_guard`,
                    // which pushes under the same lock *before* it attempts the
                    // gate CAS. A guard parked after this drainer's last sweep is
                    // therefore either already visible above (kept in `parked`, so
                    // we would not be here) or lands only after the lock is
                    // released — in which case its pusher wins the CAS and starts a
                    // fresh drainer. No guard is ever stranded without a drainer.
                    QUARANTINE_DRAINER_RUNNING.store(false, Ordering::SeqCst);
                    return;
                }
            }
            std::thread::sleep(poll);
        });

    if spawned.is_err() {
        QUARANTINE_DRAINER_RUNNING.store(false, Ordering::SeqCst);
    }
}

/// Tear down a *still-live* fan-out worker's worktree, but **only after that
/// exact worker generation physically exits** — never before, and never on a
/// timeout.
///
/// A cooperative cancel writes a terminal `stopped` manifest immediately, but the
/// worker keeps running (and may keep writing its worktree) until it observes the
/// abort. Dropping the guard runs `git worktree remove`; doing that while the
/// worker is still editing would corrupt or delete a live worktree, so teardown
/// is gated strictly on the real physical-exit ack: `agent_worker_generation_is_live`
/// going false for *this* `(agent_id, generation)`. There is deliberately **no
/// cap**: a worker that never exits keeps its worktree owned here (a bounded,
/// safe quarantine) rather than having a live worktree yanked out from under it —
/// correctness over reclaiming the handle.
///
/// The generation binding defeats the same-id resume ABA: the cancel-signal
/// registry keys each generation separately, so a same-id resume registers a new
/// generation *alongside* the old one. This owner waits on its exact
/// `(agent_id, generation)` and stays independent of the resume — the old
/// generation remains live (and this owner keeps holding the old worktree) until
/// the old worker itself exits and unregisters, never dropped merely because a
/// new generation took the id, and never touching the new generation's worktree.
///
/// This owner **never mutates the shared working tree** (no `apply_patch`). By
/// the time a fan-out worker is cancelled for overrunning the deadline, the tool
/// has already returned its result; merging its late change-set back
/// asynchronously would race later turns and user edits and break the spawn-order
/// merge contract. A timed-out worker's change-set is therefore quarantined in
/// its worktree and dropped, not merged — reported honestly as not-merged rather
/// than applied behind the user's back.
///
/// Deadlock-safety: the owner runs on its own thread and touches only the
/// cancel-signal registry (via `agent_worker_generation_is_live`); it never
/// acquires a per-agent manifest lock, so it cannot invert the manifest →
/// cancel-signal lock order the salvage path uses. Returns the join handle so
/// tests can await deterministic teardown ordering; production callers detach it.
fn spawn_deferred_worktree_cleanup(
    agent_id: String,
    generation: u64,
    guard: Box<dyn crate::workflow_tools::worktree::WorktreeGuard>,
    poll: Duration,
) -> Option<std::thread::JoinHandle<()>> {
    // Share the guard through a handoff cell so that if the cleanup thread cannot
    // be spawned, the caller can recover the *exact same* guard and park it in the
    // quarantine — the guard is owned in exactly one place at all times and is
    // never dropped by a failed `spawn` while its worker is still live.
    let cell: std::sync::Arc<
        std::sync::Mutex<Option<Box<dyn crate::workflow_tools::worktree::WorktreeGuard>>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(Some(guard)));
    let thread_cell = std::sync::Arc::clone(&cell);
    let thread_agent_id = agent_id.clone();

    #[cfg(test)]
    let force_failure = FORCE_DEFERRED_CLEANUP_SPAWN_FAILURE
        .swap(false, std::sync::atomic::Ordering::SeqCst);
    #[cfg(not(test))]
    let force_failure = false;

    let spawn_result = if force_failure {
        Err(std::io::Error::other("forced spawn failure (test seam)"))
    } else {
        std::thread::Builder::new()
            .name("zo-fanout-worktree-cleanup".to_string())
            .spawn(move || {
                while agent_tools::agent_worker_generation_is_live(&thread_agent_id, generation) {
                    std::thread::sleep(poll);
                }
                // Physical-exit ack observed for this exact generation: the
                // worktree is no longer live, so dropping the guard (`git worktree
                // remove`) is safe. No merge-back — the change-set is quarantined,
                // not applied. Take ownership out of the cell so the drop happens
                // exactly once, here.
                let taken = thread_cell.lock().expect("cleanup handoff lock").take();
                drop(taken);
            })
    };

    if let Ok(handle) = spawn_result {
        Some(handle)
    } else {
        // The cleanup thread never ran, so it never took the guard: recover it
        // from the cell and park it in the process-global quarantine, which drops
        // it only after the exact-generation exit ack. Guard ownership moves
        // exactly once, here — never dropped while live.
        if let Some(guard) = cell.lock().expect("cleanup handoff lock").take() {
            quarantine_worktree_guard(agent_id, generation, guard, poll);
        }
        None
    }
}

/// Merge each isolated fan-out agent's change-set back into the main working
/// tree (tracks 3-3 + 4-3), in spawn order, using the same collect-patch →
/// 3-way-apply path the workflow engine uses. Best-effort: a clean worktree
/// contributes nothing, and a patch that fails to collect or apply is recorded
/// as a per-agent error note (left for manual resolution) rather than aborting
/// — sibling change-sets still merge. Guards with no agent id (spawn failed
/// after the worktree was created) are skipped.
fn merge_back_fanout_worktrees(
    provider: &dyn crate::workflow_tools::worktree::WorktreeProvider,
    guards: &[(
        String,
        Box<dyn crate::workflow_tools::worktree::WorktreeGuard>,
    )],
    errors: &mut Vec<Value>,
) {
    for (agent_id, guard) in guards {
        if agent_id.is_empty() {
            continue;
        }
        match guard.collect_patch() {
            Ok(None) => {}
            Ok(Some(patch)) => {
                if let Err(err) = provider.apply_patch(&patch) {
                    errors.push(json!({
                        "agentId": agent_id,
                        "error": format!("worktree merge-back failed: {err}"),
                    }));
                }
            }
            Err(err) => errors.push(json!({
                "agentId": agent_id,
                "error": format!("collecting worktree changes failed: {err}"),
            })),
        }
    }
}
#[cfg(test)]
fn wait_window_label(timeout: Duration) -> String {
    let secs = timeout.as_secs();
    if secs >= 60 && secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

pub(crate) fn run_config(input: ConfigInput) -> Result<String, ToolError> {
    to_pretty_json(config_tools::execute_config(input)?)
}

pub(crate) fn run_enter_plan_mode(input: EnterPlanModeInput) -> Result<String, ToolError> {
    to_pretty_json(config_tools::execute_enter_plan_mode(input)?)
}

pub(crate) fn run_exit_plan_mode(input: ExitPlanModeInput) -> Result<String, ToolError> {
    to_pretty_json(config_tools::execute_exit_plan_mode(input)?)
}

pub(crate) fn run_structured_output(input: StructuredOutputInput) -> Result<String, ToolError> {
    to_pretty_json(execute_structured_output(input)?)
}

pub(crate) fn run_ask_user_question(
    input: AskUserQuestionInput,
    channel: Option<&dyn crate::UserQuestionChannel>,
) -> Result<String, ToolError> {
    run_ask_user_question_with_terminal_state(input, channel, stdin_is_terminal())
}

fn run_ask_user_question_with_terminal_state(
    input: AskUserQuestionInput,
    channel: Option<&dyn crate::UserQuestionChannel>,
    stdin_is_terminal: bool,
) -> Result<String, ToolError> {
    let options = input.options.unwrap_or_default();
    // Multi-select only applies with a fixed choice list; a free-form prompt is
    // always a single answer, matching how the modal degrades the flag.
    let multi_select = input.multi_select && !options.is_empty();
    let raw: Vec<String> = if let Some(ch) = channel {
        ch.ask(
            &input.question,
            input.header.as_deref(),
            &options,
            multi_select,
        )?
    } else if !stdin_is_terminal {
        return to_pretty_json(json!({
            "question": input.question,
            "status": "unanswered",
            "reason": "non-interactive"
        }));
    } else {
        ask_user_question_stdio(&input.question, &options, multi_select)?
    };

    // Map any numeric picks to their labels; the TUI already returns labels, so
    // this only rewrites the stdio "type 2" form and is a no-op otherwise.
    let resolved: Vec<String> = raw
        .iter()
        .map(|answer| resolve_option_choice(answer, &options))
        .collect();

    // Single-select preserves the historical string `answer`; multi-select
    // returns the full list so the model sees every checked option.
    let answer = if multi_select {
        json!(resolved)
    } else {
        json!(resolved.into_iter().next().unwrap_or_default())
    };

    to_pretty_json(json!({
        "question": input.question,
        "answer": answer,
        "status": "answered"
    }))
}

fn stdin_is_terminal() -> bool {
    use std::io::IsTerminal;

    std::io::stdin().is_terminal()
}

pub(crate) fn run_memory_write(
    input: &MemoryWriteInput,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    let output = execute_memory_write(input, ctx)?;
    to_pretty_json(output)
}

fn execute_memory_write(input: &MemoryWriteInput, ctx: &ToolContext) -> Result<Value, ToolError> {
    let slug = normalize_memory_slug(&input.slug)?;
    let summary = sanitize_memory_summary(&input.summary)?;
    let body = input.body.trim();
    if body.is_empty() {
        return Err(ToolError::InvalidInput(
            "`body` must not be empty".to_string(),
        ));
    }
    let body = body_with_hand_written_metadata(body);

    let cwd = ctx
        .cwd
        .clone()
        .map_or_else(std::env::current_dir, Ok)
        .map_err(ToolError::Io)?;
    let outcome = runtime::memory::write_hand_written_memory_entry(
        &cwd,
        input.local,
        &runtime::memory::MemoryWriteRequest {
            slug: slug.clone(),
            summary: summary.clone(),
            body,
        },
    )
    .map_err(|error| ToolError::Io(std::io::Error::other(error)))?;
    let memory_dir = runtime::memory::paths::memory_write_dir(&cwd, input.local);
    let entry_path = memory_dir.join(format!("{slug}.md"));
    let index_path = memory_dir.join("MEMORY.md");
    Ok(json!({
        "status": outcome.as_str(),
        "slug": slug,
        "local": input.local,
        "path": entry_path.display().to_string(),
        "indexPath": index_path.display().to_string()
    }))
}

fn body_with_hand_written_metadata(body: &str) -> String {
    let body = strip_memory_metadata_lines(body);
    let written_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs());
    format!(
        "{}

---
{}",
        body.trim(),
        runtime::memory::hand_written_memory_metadata_line(runtime::memory::MemoryKind::Unknown, written_at)
    )
}

fn sanitize_memory_summary(summary: &str) -> Result<String, ToolError> {
    let summary = summary
        .chars()
        .map(|ch| match ch {
            '[' | ']' | '(' | ')' => ' ',
            ch if ch.is_control() => ' ',
            ch => ch,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if summary.is_empty() {
        return Err(ToolError::InvalidInput(
            "`summary` must not be empty".to_string(),
        ));
    }
    Ok(summary)
}

fn strip_memory_metadata_lines(body: &str) -> String {
    body.lines()
        .filter(|line| !line.trim_start().starts_with("- memory_metadata:"))
        .collect::<Vec<_>>()
        .join("
")
}

fn normalize_memory_slug(input: &str) -> Result<String, ToolError> {
    let stem = input.trim().trim_end_matches(".md");
    let mut slug = String::new();
    let mut previous_dash = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            previous_dash = false;
        } else if (ch.is_ascii_whitespace() || matches!(ch, '-' | '_' | '.' | '/'))
            && !previous_dash
            && !slug.is_empty()
        {
            slug.push('-');
            previous_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    if slug.is_empty() {
        return Err(ToolError::InvalidInput(
            "`slug` must contain at least one ASCII letter or digit".to_string(),
        ));
    }
    Ok(slug)
}

/// Resolve a raw response against an options list (numeric index → label).
/// Anything that is not a valid index passes through as a free-form answer.
fn resolve_option_choice(
    response: &str,
    options: &[runtime::message_stream::QuestionOption],
) -> String {
    let trimmed = response.trim();
    if let Ok(idx) = trimmed.parse::<usize>() {
        if idx >= 1 && idx <= options.len() {
            return options[idx - 1].label.clone();
        }
    }
    trimmed.to_string()
}

/// Fallback: direct stdin/stdout I/O when no channel is configured.
///
/// Returns the raw response tokens; the caller maps any numeric picks to their
/// labels uniformly for both the stdio and channel paths. A multi-select prompt
/// accepts a comma-separated list, so several picks come back as several tokens.
fn ask_user_question_stdio(
    question: &str,
    options: &[runtime::message_stream::QuestionOption],
    multi_select: bool,
) -> Result<Vec<String>, ToolError> {
    use std::io::{self, BufRead, Write};

    let stdout = io::stdout();
    let stdin = io::stdin();
    let mut out = stdout.lock();

    writeln!(out, "\n[Question] {question}")?;

    if options.is_empty() {
        write!(out, "Your answer: ")?;
    } else {
        for (i, option) in options.iter().enumerate() {
            writeln!(out, "  {}. {}", i + 1, option.label)?;
            if let Some(description) = option.description.as_deref() {
                writeln!(out, "     {description}")?;
            }
        }
        if multi_select {
            write!(
                out,
                "Enter choices (1-{}, comma-separated) or free text: ",
                options.len()
            )?;
        } else {
            write!(out, "Enter choice (1-{}) or free text: ", options.len())?;
        }
    }
    out.flush()?;

    let mut response = String::new();
    stdin.lock().read_line(&mut response)?;

    if multi_select && !options.is_empty() {
        Ok(response
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect())
    } else {
        Ok(vec![response.trim().to_string()])
    }
}

pub(crate) fn run_remote_trigger(input: RemoteTriggerInput) -> Result<String, ToolError> {
    crate::http_bridge::run_http(run_remote_trigger_async(input))
}

/// Advisory webhook helper for non-tool callers such as `zo serve`.
///
/// It reuses the `RemoteTrigger` HTTP client and response handling, but fixes the
/// method to POST and serializes `payload` as JSON. Callers should treat errors
/// as best-effort notification failures rather than core turn failures.
pub async fn notify_remote(url: &str, payload: Value) -> Result<String, ToolError> {
    run_remote_trigger_async(RemoteTriggerInput {
        url: url.to_string(),
        method: Some("POST".to_string()),
        headers: Some(json!({ "Content-Type": "application/json" })),
        body: Some(payload.to_string()),
    })
    .await
}

/// Process-wide shared `reqwest::Client` for the `RemoteTrigger` tool.
///
/// Per-request method, headers, and body still vary per call; only the client
/// (and its connection pool / TLS state) is shared. Building one per call
/// re-initialised the TLS backend and discarded the pool every time. Mirrors
/// `web_tools::shared_http_client` and `api::providers::shared_http_client`;
/// `reqwest::Client` is `Arc`-backed so `clone()` is free. The 30s timeout is
/// unchanged.
fn shared_remote_trigger_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new())
        })
        .clone()
}

async fn run_remote_trigger_async(input: RemoteTriggerInput) -> Result<String, ToolError> {
    let method_raw = input.method.clone().unwrap_or_else(|| "GET".to_string());
    let method = reqwest::Method::from_bytes(method_raw.to_uppercase().as_bytes())
        .map_err(|_| ToolError::InvalidInput(format!("unsupported HTTP method: {method_raw}")))?;

    if !matches!(
        method,
        reqwest::Method::GET
            | reqwest::Method::POST
            | reqwest::Method::PUT
            | reqwest::Method::DELETE
            | reqwest::Method::PATCH
            | reqwest::Method::HEAD
    ) {
        return Err(ToolError::InvalidInput(format!(
            "unsupported HTTP method: {method_raw}"
        )));
    }

    let client = shared_remote_trigger_client();
    let mut request = client.request(method.clone(), &input.url);

    if let Some(ref headers) = input.headers {
        if let Some(obj) = headers.as_object() {
            for (key, value) in obj {
                if let Some(val) = value.as_str() {
                    request = request.header(key.as_str(), val);
                }
            }
        }
    }

    if let Some(ref body) = input.body {
        request = request.body(body.clone());
    }

    match request.send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            let truncated_body = if body.len() > 8192 {
                // Clamp to a UTF-8 char boundary at or below 8192 bytes —
                // a raw byte slice panics if the cut lands mid-codepoint,
                // which is common for non-ASCII HTTP response bodies.
                let mut end = 8192;
                while end > 0 && !body.is_char_boundary(end) {
                    end -= 1;
                }
                format!(
                    "{}\n\n[response truncated — {} bytes total]",
                    &body[..end],
                    body.len()
                )
            } else {
                body
            };
            to_pretty_json(json!({
                "url": input.url,
                "method": method.as_str(),
                "status_code": status,
                "body": truncated_body,
                "success": (200..300).contains(&status)
            }))
        }
        Err(e) => to_pretty_json(json!({
            "url": input.url,
            "method": method.as_str(),
            "error": e.to_string(),
            "success": false
        })),
    }
}

pub(crate) fn run_testing_permission(
    input: &TestingPermissionInput,
    enforcer: Option<&PermissionEnforcer>,
) -> Result<String, ToolError> {
    let (permitted, active_mode, message) = match enforcer {
        Some(enf) => {
            let mode = enf.active_mode();
            let allowed = enf.is_allowed(&input.action, "");
            let msg = if allowed {
                format!(
                    "action '{}' is permitted under '{}' mode",
                    input.action,
                    mode.as_str()
                )
            } else {
                format!(
                    "action '{}' is denied under '{}' mode",
                    input.action,
                    mode.as_str()
                )
            };
            (allowed, mode.as_str().to_owned(), msg)
        }
        None => (
            true,
            PermissionMode::DangerFullAccess.as_str().to_owned(),
            format!(
                "no enforcer active; action '{}' assumed permitted",
                input.action
            ),
        ),
    };
    to_pretty_json(json!({
        "action": input.action,
        "permitted": permitted,
        "active_mode": active_mode,
        "message": message
    }))
}

pub(crate) fn run_monitor(input: MonitorInput) -> Result<String, ToolError> {
    let process_id = input.process_id.or(input.command).unwrap_or_default();
    if process_id.trim().is_empty() {
        return Err(ToolError::InvalidInput(
            "either process_id or command must be provided".into(),
        ));
    }
    let max_lines = input.lines;

    // Attempt to read the background process output file written by
    // `run_in_background` in bash_tools.  The convention is that background
    // process output is stored alongside the process id.
    let output_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let output_file = output_dir.join(format!(".zo-bg-{}.log", process_id.trim()));

    let lines: Vec<String> = if output_file.exists() {
        let content = std::fs::read_to_string(&output_file)
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        content
            .lines()
            .rev()
            .take(max_lines)
            .map(String::from)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    } else {
        Vec::new()
    };

    to_pretty_json(json!({
        "process_id": process_id,
        "lines": lines,
        "line_count": lines.len(),
        "source": if output_file.exists() { "file" } else { "not_found" },
        "message": if lines.is_empty() {
            format!("No output found for process '{process_id}'. The process may not have started or has no output yet.")
        } else {
            format!("Returning last {} lines from process '{process_id}'", lines.len())
        }
    }))
}

/// Deliver a message to a spawned agent — the interactive-agents contract:
/// a RUNNING agent receives it mid-turn via its steering queue (drained at the
/// next tool-result boundary), and a TERMINAL agent is resumed in the
/// background with its persisted transcript rehydrated and the message as its
/// next user turn. Both replies ride the background-completion channel, so the
/// interactive host re-invokes the parent model with the result.
pub(crate) fn run_send_message(
    input: &SendMessageInput,
    parent_lsp: Option<&LspRegistry>,
    hook_config: Option<&RuntimeHookConfig>,
    mcp_passthrough: Option<crate::registry::McpPassthrough>,
    parent_permission_mode: Option<runtime::PermissionMode>,
) -> Result<String, ToolError> {
    if input.to.trim().is_empty() {
        return Err(ToolError::InvalidInput("'to' must not be empty".into()));
    }
    if input.message.trim().is_empty() {
        return Err(ToolError::InvalidInput(
            "'message' must not be empty".into(),
        ));
    }

    let Some(mut manifest) = lookup_agent_manifest(&input.to) else {
        return to_pretty_json(json!({
            "to": input.to,
            "delivered": false,
            "agentStatus": "not_found",
            "sentAt": epoch_seconds_now(),
            "info": format!(
                "No spawned agent matches '{}'; check the name/id or spawn a new Agent.",
                input.to
            ),
        }));
    };

    if manifest.status == "running" {
        // Framed so the sub-agent can tell an injected orchestrator message
        // apart from its own task text mid-turn.
        let delivered = agent_tools::steer_agent(
            &manifest.agent_id,
            format!("[message via SendMessage] {}", input.message),
        );
        if delivered || !agent_tools::settle_dead_owner_agent(&manifest) {
            return to_pretty_json(json!({
                "to": input.to,
                "agentId": manifest.agent_id,
                "agentStatus": "running",
                "mode": "steer",
                "delivered": delivered,
                "sentAt": epoch_seconds_now(),
                "info": if delivered {
                    "Delivered into the running agent's turn; it will see the message at its \
                     next tool boundary, and its result is still delivered on completion."
                } else {
                    "Agent is marked running but has no live steering handle in this process \
                     (it may be finishing, or owned by another process). Retry shortly or \
                     wait for its completion."
                },
            }));
        }
        if let Some(settled) = lookup_agent_manifest(&manifest.agent_id) {
            manifest = settled;
        }
    }

    match agent_tools::resume_agent_with_message(
        &manifest,
        &input.message,
        parent_lsp,
        hook_config,
        mcp_passthrough,
        parent_permission_mode,
    ) {
        Ok(resumed) => to_pretty_json(json!({
            "to": input.to,
            "agentId": resumed.agent_id,
            "agentStatus": "running",
            "mode": "resume",
            "delivered": true,
            "sentAt": epoch_seconds_now(),
            "info": "Agent resumed in the background with its prior context intact; its \
                     reply will be delivered to you in a later message. Do NOT poll its \
                     output file.",
        })),
        Err(error) => to_pretty_json(json!({
            "to": input.to,
            "agentId": manifest.agent_id,
            "agentStatus": manifest.status,
            "mode": "resume",
            "delivered": false,
            "sentAt": epoch_seconds_now(),
            "error": error.to_string(),
        })),
    }
}

/// Outcome of a host-initiated agent send ([`send_agent_message`]), shaped for
/// direct display (the Ctrl+G viewer's footer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSendOutcome {
    /// Delivered into the running agent's steering queue; it sees the message
    /// at its next tool boundary.
    Steered { name: String },
    /// The terminal agent was resumed in the background with its prior
    /// context; its reply rides the completion re-injection channel.
    Resumed { name: String },
    /// No spawned agent matches the target.
    NotFound,
    /// Marked running but no live steering handle in this process, with an
    /// owner that is still live or whose liveness could not be resolved.
    Unreachable { name: String },
    /// The resume attempt failed (e.g. no persisted transcript).
    Failed { name: String, error: String },
}

/// Deliver `message` to the agent named/ided `target` on behalf of the USER —
/// the Ctrl+G viewer's message box. Same steer/resume semantics as the
/// model-facing `SendMessage` tool, minus session-context inheritance
/// (LSP/MCP passthrough/hooks), which a user-initiated send has no handle on;
/// the resumed agent's builtin toolset does not depend on it.
/// `parent_permission_mode` is the host session's active mode — a resume
/// re-clamps to it, so the fresh-spawn clamp is not escapable via the viewer.
#[must_use]
pub fn send_agent_message(
    target: &str,
    message: &str,
    parent_permission_mode: Option<runtime::PermissionMode>,
) -> AgentSendOutcome {
    let message = message.trim();
    if target.trim().is_empty() || message.is_empty() {
        return AgentSendOutcome::NotFound;
    }
    let Some(manifest) = lookup_agent_manifest(target) else {
        return AgentSendOutcome::NotFound;
    };
    let name = manifest
        .label
        .clone()
        .unwrap_or_else(|| manifest.name.clone());
    if manifest.status == "running" {
        return if agent_tools::steer_agent(
            &manifest.agent_id,
            format!("[message from the user] {message}"),
        ) {
            AgentSendOutcome::Steered { name }
        } else {
            AgentSendOutcome::Unreachable { name }
        };
    }
    match agent_tools::resume_agent_with_message(
        &manifest,
        message,
        None,
        None,
        None,
        parent_permission_mode,
    ) {
        Ok(_) => AgentSendOutcome::Resumed { name },
        Err(error) => AgentSendOutcome::Failed {
            name,
            error: error.to_string(),
        },
    }
}

/// Drain terminal background-agent completions belonging to `session_id`,
/// clearing their background marks. Hosts WITHOUT an idle re-injection pump
/// (serve — the interactive REPL consumes the completion channel instead)
/// sweep this at the next turn boundary and fold the results into the turn
/// input via [`fold_background_completions_into_input`], so a detached
/// agent's answer is never lost. Agents stamped with a different (or no)
/// session are left marked for their own host to consume.
#[must_use]
pub fn drain_background_completions_for_session(session_id: &str) -> Vec<AgentCompletion> {
    let mut drained = Vec::new();
    for agent_id in agent_tools::background_agent_ids_snapshot() {
        let belongs_to_session = match agent_tools::background_task_session_id(&agent_id) {
            agent_tools::BackgroundTaskSession::Session(task_session_id) => {
                task_session_id == session_id
            }
            agent_tools::BackgroundTaskSession::Unstamped => false,
            agent_tools::BackgroundTaskSession::NotTask => {
                manifest_by_id(&agent_id).is_some_and(|manifest| {
                    manifest.parent_session_id.as_deref() == Some(session_id)
                        && matches!(manifest.status.as_str(), "completed" | "failed")
                })
            }
        };
        if !belongs_to_session {
            continue;
        }
        let Some(completion) = agent_tools::wait_for_agent_completions(
            std::slice::from_ref(&agent_id),
            std::time::Duration::ZERO,
        )
        .into_iter()
        .find(|completion| {
            completion.agent_id == agent_id && completion.status != "still_running"
        }) else {
            continue;
        };
        clear_background_agent(&agent_id);
        drained.push(completion);
    }
    drained
}

/// Render drained background completions plus the user's input as ONE turn
/// input — each completion leads with the same follow-up pointer the REPL
/// re-injection header carries, so the model can `SendMessage` the agent to
/// continue it with context intact. Pure, for testability.
#[must_use]
pub fn fold_background_completions_into_input(
    completions: &[AgentCompletion],
    input: &str,
) -> String {
    if completions.is_empty() {
        return input.to_string();
    }
    let mut sections = Vec::with_capacity(completions.len() + 1);
    for completion in completions {
        let verb = if completion.status == "completed" {
            "finished"
        } else {
            "failed"
        };
        let body = completion
            .result
            .as_deref()
            .or(completion.error.as_deref())
            .unwrap_or("(no output)");
        sections.push(format!(
            "[background agent `{name}` {verb} — its result follows; follow up with \
             SendMessage(to: \"{id}\") to continue it with context intact]\n{body}",
            name = completion.name,
            id = completion.agent_id,
            body = core_types::text::elide_middle(body, 16_000),
        ));
    }
    sections.push(input.to_string());
    sections.join("\n\n---\n\n")
}

/// The manifest for one agent id, read straight from its store path (no
/// directory scan — the id IS the file name).
fn manifest_by_id(agent_id: &str) -> Option<agent_tools::AgentOutput> {
    let store_dir = agent_store_dir().ok()?;
    let text = std::fs::read_to_string(store_dir.join(format!("{agent_id}.json"))).ok()?;
    serde_json::from_str(&text).ok()
}

/// Locate a spawned agent's manifest by exact name/id first, then by
/// name/id substring. Ties go to the most recently created match, so a
/// re-used agent name in a long session addresses the newest incarnation.
fn lookup_agent_manifest(target: &str) -> Option<agent_tools::AgentOutput> {
    let store_dir = agent_store_dir().ok()?;
    let entries = std::fs::read_dir(&store_dir).ok()?;
    let target_lower = target.to_ascii_lowercase();
    let mut exact: Option<agent_tools::AgentOutput> = None;
    let mut fuzzy: Option<agent_tools::AgentOutput> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Non-manifest `.json` siblings (`<id>.resume.json`) fail this parse
        // and are skipped.
        let Ok(manifest) = serde_json::from_str::<agent_tools::AgentOutput>(&content) else {
            continue;
        };
        let slot = if manifest.name == target || manifest.agent_id == target {
            &mut exact
        } else if manifest.name.to_ascii_lowercase().contains(&target_lower)
            || manifest.agent_id.contains(target)
        {
            &mut fuzzy
        } else {
            continue;
        };
        let newer = slot.as_ref().is_none_or(|held| {
            manifest.created_at.parse::<u64>().unwrap_or(0)
                >= held.created_at.parse::<u64>().unwrap_or(0)
        });
        if newer {
            *slot = Some(manifest);
        }
    }
    exact.or(fuzzy)
}

pub(crate) fn run_schedule_wakeup(
    input: &ScheduleWakeupInput,
    session_id: Option<&str>,
) -> Result<String, ToolError> {
    if input.reason.trim().is_empty() {
        return Err(ToolError::InvalidInput("'reason' must not be empty".into()));
    }
    if input.prompt.trim().is_empty() {
        return Err(ToolError::InvalidInput("'prompt' must not be empty".into()));
    }
    if input.delay_seconds < 0.0 {
        return Err(ToolError::InvalidInput(
            "delaySeconds must be non-negative".into(),
        ));
    }

    // Write the wakeup request to a session state file that the main loop checks.
    let mut wakeup = json!({
        "delaySeconds": input.delay_seconds,
        "reason": input.reason,
        "prompt": input.prompt,
        "scheduledAt": epoch_seconds_now(),
    });
    if let Some(session_id) = session_id {
        wakeup["sessionId"] = json!(session_id);
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let state_dir = runtime::zo_state_base(&cwd).join(".zo").join("wakeups");
    let _ = std::fs::create_dir_all(&state_dir);

    let wakeup_id = format!(
        "wakeup-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let wakeup_file = state_dir.join(format!("{wakeup_id}.json"));
    std::fs::write(
        &wakeup_file,
        serde_json::to_string_pretty(&wakeup).unwrap_or_default(),
    )
    .map_err(|e| ToolError::Execution(e.to_string()))?;

    to_pretty_json(json!({
        "wakeupId": wakeup_id,
        "delaySeconds": input.delay_seconds,
        "reason": input.reason,
        "promptLength": input.prompt.len(),
        "scheduledAt": wakeup["scheduledAt"],
        "stateFile": wakeup_file.display().to_string(),
        "message": format!(
            "Wakeup '{}' scheduled in {}s: {}",
            wakeup_id, input.delay_seconds, input.reason
        )
    }))
}

// --- Implementation functions remaining in misc_tools ---

const MAX_SLEEP_DURATION_MS: u64 = 5_000;

#[allow(clippy::unnecessary_wraps)]
fn execute_sleep(input: &SleepInput) -> Result<SleepOutput, String> {
    let clamped = input.duration_ms.min(MAX_SLEEP_DURATION_MS);
    let message = if input.duration_ms > MAX_SLEEP_DURATION_MS {
        format!(
            "Slept for {clamped}ms (clamped from {} to keep tool budget tight)",
            input.duration_ms
        )
    } else {
        format!("Slept for {clamped}ms")
    };
    // live 경로는 런타임이 이미 `tokio::time::sleep` 으로 비차단 대기했으므로
    // 여기서 다시 동기 슬립하면 대기가 2배가 되고 `block_in_place` 로 turn
    // loop 의 render_tick 이 멈춘다. 플래그가 없을 때(비-live)만 슬립한다.
    if !input.already_slept {
        std::thread::sleep(Duration::from_millis(clamped));
    }
    Ok(SleepOutput {
        duration_ms: clamped,
        message,
    })
}

fn execute_structured_output(
    input: StructuredOutputInput,
) -> Result<StructuredOutputResult, String> {
    if input.0.is_empty() {
        return Err(String::from("structured output payload must not be empty"));
    }
    Ok(StructuredOutputResult {
        data: String::from("Structured output provided successfully"),
        structured_output: input.0,
    })
}

// --- ToolSearch ---

fn execute_tool_search(input: &ToolSearchInput, ctx: &ToolContext) -> ToolSearchOutput {
    let registry = GlobalToolRegistry::builtin().with_context(ctx.clone());
    if let Some(passthrough) = ctx.mcp_passthrough() {
        let _ = registry.set_runtime_tools(passthrough.definitions_snapshot());
    }
    registry.search(&input.query, input.max_results.unwrap_or(5), None, None)
}

pub(crate) fn search_tool_specs(
    query: &str,
    max_results: usize,
    specs: &[SearchableToolSpec],
) -> Vec<String> {
    let lowered = query.to_lowercase();
    if let Some(selection) = lowered.strip_prefix("select:") {
        return selection
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .filter_map(|wanted| {
                let wanted = canonical_tool_token(wanted);
                specs
                    .iter()
                    .find(|spec| canonical_tool_token(&spec.name) == wanted)
                    .map(|spec| spec.name.clone())
            })
            .take(max_results)
            .collect();
    }

    let mut required = Vec::new();
    let mut optional = Vec::new();
    for term in lowered.split_whitespace() {
        if let Some(rest) = term.strip_prefix('+') {
            if !rest.is_empty() {
                required.push(rest);
            }
        } else {
            optional.push(term);
        }
    }
    let terms = if required.is_empty() {
        optional.clone()
    } else {
        required.iter().chain(optional.iter()).copied().collect()
    };

    let mut scored = specs
        .iter()
        .filter_map(|spec| {
            let name = spec.name.to_lowercase();
            let canonical_name = canonical_tool_token(&spec.name);
            let normalized_description = normalize_tool_search_query(&spec.description);
            let haystack = format!(
                "{name} {} {canonical_name}",
                spec.description.to_lowercase()
            );
            let normalized_haystack = format!("{canonical_name} {normalized_description}");
            if required.iter().any(|term| !haystack.contains(term)) {
                return None;
            }

            let mut score = 0_i32;
            for term in &terms {
                let canonical_term = canonical_tool_token(term);
                if haystack.contains(term) {
                    score += 2;
                }
                if name == *term {
                    score += 8;
                }
                if name.contains(term) {
                    score += 4;
                }
                if canonical_name == canonical_term {
                    score += 12;
                }
                if normalized_haystack.contains(&canonical_term) {
                    score += 3;
                }
            }

            if score == 0 && !lowered.is_empty() {
                return None;
            }
            Some((score, spec.name.clone()))
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    scored
        .into_iter()
        .map(|(_, name)| name)
        .take(max_results)
        .collect()
}

pub(crate) fn normalize_tool_search_query(query: &str) -> String {
    query
        .trim()
        .split(|ch: char| ch.is_whitespace() || ch == ',')
        .filter(|term| !term.is_empty())
        .map(canonical_tool_token)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn canonical_tool_token(value: &str) -> String {
    let mut canonical = value
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if let Some(stripped) = canonical.strip_suffix("tool") {
        canonical = stripped.to_string();
    }
    canonical
}

#[cfg(test)]
mod tests;
