//! Automatic multi-agent fan-out — host-callable stages 2 (decompose) and 3
//! (run engine).
//!
//! The live-turn seam (`run_live_turn_with_images`) uses these to split a broad
//! request into independent subtasks and run them through the existing
//! `SpawnMultiAgent` engine *without the model emitting a `tool_use`*. They reuse
//! the model-driven primitives (`execute_agent_with_parent_model_and_hooks`,
//! `wait_for_agent_completions`, `run_spawn_multi_agent`) so sub-agents inherit
//! the parent provider and then apply the same smart model routing as the manual
//! path, share the rate-limit semaphore, and flow through the same completion
//! plumbing.
//!
//! **Blocking contract.** Both functions block until their agents finish. The
//! caller MUST invoke them inside `tokio::task::spawn_blocking` — the
//! `is_long_running` routing that keeps the
//! TUI render loop ticking applies only to model-driven tool dispatch, not to a
//! host-initiated call.
//!
//! **Permission contract.** `run_fanout_spawn` runs agents unconditionally.
//! `SpawnMultiAgent` requires `DangerFullAccess`, so the caller MUST gate on
//! that mode first (the seam does). This keeps fan-out fail-closed without
//! threading a `PermissionEnforcer` across the `spawn_blocking` boundary.

use std::fmt::Write as _;
use std::time::Duration;

use api::openai_gpt_model_family;
use serde_json::{json, Value};

use crate::error::ToolError;
use crate::misc_tools::council::{execute_council, CouncilCandidateInput, CouncilInput};
use crate::misc_tools::{
    execute_agent_with_parent_model_and_hooks, run_spawn_multi_agent_with_timeout_and_hooks,
    wait_for_agent_completions, AgentInput, SpawnMultiAgentInput, AGENT_MODEL_ENV,
    SPAWN_MULTI_AGENT_WAIT_TIMEOUT,
};

/// One decomposed subtask: a short role label and the self-contained prompt the
/// parallel sub-agent runs.
#[derive(Debug, Clone)]
pub struct FanoutSubtask {
    pub role: String,
    pub prompt: String,
}

/// Upper bound on decomposed subtasks — keeps fan-out within the
/// `SpawnMultiAgent` agent cap and bounds the token spend of an auto-fan-out.
pub const MAX_FANOUT_SUBTASKS: usize = 4;

/// Number of independent re-answers a self-consistency fan-out spawns before
/// voting them through the Council. Three is the smallest odd count that yields
/// a meaningful majority; capped at [`MAX_FANOUT_SUBTASKS`] at the spawn site so
/// it stays within the `SpawnMultiAgent` agent cap.
pub const SELF_CONSISTENCY_K: usize = 3;

// Self-consistency must stay an odd majority within the spawn cap. Enforced at
// compile time (was a runtime test over constants).
const _: () = assert!(SELF_CONSISTENCY_K <= MAX_FANOUT_SUBTASKS);
const _: () = assert!(SELF_CONSISTENCY_K >= 2);

/// How long the decomposition agent may take before we give up and fall back to
/// a single-agent turn. Short — decomposition is one cheap structured reply.
const DECOMPOSE_TIMEOUT: Duration = Duration::from_secs(60);
/// Result collection window for automatic pre-analysis. This is intentionally
/// shorter than manual `SpawnMultiAgent`: pre-analysis is preliminary context,
/// not a reason to hold the foreground turn for twenty minutes.
pub const AUTO_FANOUT_DECOMPOSE_TIMEOUT: Duration = Duration::from_secs(15);
pub const AUTO_FANOUT_AGENT_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const DIAGNOSE_FANOUT_AGENT_TIMEOUT: Duration = SPAWN_MULTI_AGENT_WAIT_TIMEOUT;

/// Stage 2 — decompose `user_text` into independent parallel subtasks. Runs on a
/// cheap model chosen for the active provider (see [`decompose_model`]). Returns
/// fewer than two subtasks (often empty) when the task does not split cleanly or
/// decomposition fails; the caller then proceeds with a normal single-agent turn.
/// `parent_model` is the active session model, so the decompose call targets the
/// session's provider. BLOCKING — run inside `spawn_blocking`.
#[must_use]
pub fn decompose_for_fanout(user_text: &str, parent_model: Option<&str>) -> Vec<FanoutSubtask> {
    decompose_for_fanout_with_timeout(user_text, parent_model, DECOMPOSE_TIMEOUT)
}

/// Variant used by automatic live pre-analysis, where the caller wants a
/// shorter wall-clock budget than the manual/deep path.
#[must_use]
pub fn decompose_for_fanout_with_timeout(
    user_text: &str,
    parent_model: Option<&str>,
    timeout: Duration,
) -> Vec<FanoutSubtask> {
    decompose_for_fanout_with_timeout_and_hooks(user_text, parent_model, timeout, None, None)
}

/// Variant that also threads configured sub-agent lifecycle/tool hooks into the
/// decomposition agent.
#[must_use]
pub fn decompose_for_fanout_with_timeout_and_hooks(
    user_text: &str,
    parent_model: Option<&str>,
    timeout: Duration,
    hook_config: Option<&runtime::RuntimeHookConfig>,
    parent_session_id: Option<&str>,
) -> Vec<FanoutSubtask> {
    // `resolve_agent_model` ignores the per-agent `model` field and honours the
    // `parent_model` we pass, so the decompose model is whatever `decompose_model`
    // selects for the active provider — never a hardcoded Claude id dialed out to
    // a non-Anthropic one.
    let model = decompose_model(parent_model);
    // StructuredOutput is reliable on Anthropic, but forcing it on some OpenAI
    // models (e.g. gpt-5.5-fast) yields an empty stream ("assistant stream
    // produced no content"). Plain-text sub-agents work on every provider, so for
    // non-Anthropic models we ask for a JSON object in the reply text and parse
    // it instead of forcing the tool.
    //
    // The agent actually runs on `ZO_AGENT_MODEL` when that override is set
    // (`resolve_agent_model` honours it before the parent), so the structured
    // decision must follow the *effective* run model — otherwise an override to a
    // GPT model gets StructuredOutput forced on it and returns empty (BUG-R15).
    let effective_model = non_empty_env(AGENT_MODEL_ENV).unwrap_or_else(|| model.clone());
    let structured = is_anthropic_model_alias(&effective_model);
    let input = AgentInput {
        allow_cross_provider: false,
        description: "fan-out decomposition".to_string(),
        prompt: decompose_prompt(user_text, structured),
        // Micro-prompt harness: this one-shot classification runs on a cheap
        // model outside the parent's cache namespace, so the full system
        // prompt would be pure re-billed weight (see `classifier` in
        // subagent_profile).
        subagent_type: Some("classifier".to_string()),
        name: Some("decompose".to_string()),
        model: None,
        cwd: None,
        schema: structured.then(decompose_schema),
        workflow_member: false,
        background: Some(false),
        // Micro-prompt classifier harness (no workspace tools); the spawn
        // clamp is not threaded through this internal helper path.
        parent_permission_mode: None,
        parent_session_id: parent_session_id.map(str::to_string),
        tool_call_id: None,
        mcp_passthrough: None,
        api_concurrency: None,
        time_budget: Some(timeout),
        prior_failures: 0,
        route_reason: None,
        route_role: None,
        route_complexity: None,
        route_risk: None,
        route_source: None,
        route_model: None,
        route_fallback_models: Vec::new(),
        route_effort: None,
        judged_agent: None,
    };
    let Ok(output) =
        execute_agent_with_parent_model_and_hooks(input, Some(&model), None, hook_config)
    else {
        return Vec::new();
    };
    let Some(completion) = wait_for_agent_completions(&[output.agent_id], timeout)
        .into_iter()
        .next()
    else {
        return Vec::new();
    };
    // Prefer the captured `StructuredOutput` input (Anthropic); otherwise parse a
    // JSON object out of the reply text (non-Anthropic, and a salvage path when a
    // structured reply came back empty).
    completion
        .structured
        .map(|value| parse_subtasks(&value))
        .filter(|subtasks| !subtasks.is_empty())
        .or_else(|| completion.result.as_deref().map(parse_subtasks_from_text))
        .unwrap_or_default()
}

/// The JSON schema the `StructuredOutput` decomposition (Anthropic path) answers
/// in: `{ "subtasks": [{ "role", "prompt" }] }`.
fn decompose_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "2-4 word label for this slice (e.g. 'api layer')"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "self-contained instruction for one agent"
                        }
                    },
                    "required": ["role", "prompt"]
                }
            }
        },
        "required": ["subtasks"]
    })
}

/// The decomposition prompt. `structured` selects the reply contract: the
/// `StructuredOutput` tool (Anthropic) or a raw JSON object in the text reply
/// (every other provider, where forcing the tool yields no content).
fn decompose_prompt(user_text: &str, structured: bool) -> String {
    let reply = if structured {
        "Reply via the StructuredOutput tool."
    } else {
        "Reply with ONLY a single JSON object and NOTHING else — no prose, no markdown fences, \
         and do NOT call any tools. Shape: \
         {\"subtasks\":[{\"role\":\"2-4 word label\",\"prompt\":\"self-contained instruction\"}]}. \
         If it does not split cleanly, reply {\"subtasks\":[]}."
    };
    format!(
        "Split the user's task into 2-{MAX_FANOUT_SUBTASKS} INDEPENDENT subtasks that can run in \
         parallel with no shared state — split by different files, layers, or angles of analysis. \
         Each subtask's `prompt` must be fully self-contained: the agent running it sees ONLY that \
         prompt, not the original task. If the task does not split cleanly into independent parallel \
         work, return an empty list. Write each subtask as bounded analysis: ask for the key \
         findings and `file:line` evidence, and tell the agent to stop tool calls once it has \
         enough evidence instead of exhaustively searching the repository. {reply}\n\nUSER TASK:\n{user_text}"
    )
}

/// Parse subtasks from a free-text reply containing a JSON object (the
/// non-Anthropic path). Extracts the first balanced `{...}` so prose or markdown
/// fences around it do not defeat parsing, then reuses [`parse_subtasks`]. An
/// unparseable reply yields an empty list — the caller's graceful fallback.
fn parse_subtasks_from_text(text: &str) -> Vec<FanoutSubtask> {
    extract_json_object(text)
        .and_then(|object| serde_json::from_str::<Value>(object).ok())
        .map(|value| parse_subtasks(&value))
        .unwrap_or_default()
}

/// The first balanced JSON object substring in `text`, or `None`. String-aware so
/// a `{`/`}` inside a JSON string value does not end the scan early; tolerant of
/// prose or markdown JSON fences around the object.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (offset, byte) in text.bytes().enumerate().skip(start) {
        if escaped {
            escaped = false;
        } else if in_str {
            match byte {
                b'\\' => escaped = true,
                b'"' => in_str = false,
                _ => {}
            }
        } else {
            match byte {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&text[start..=offset]);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The model the decomposition agent runs on. Decomposition is one cheap reply:
/// Anthropic sessions use `haiku`, OpenAI/GPT sessions use an inventory-derived
/// Fast-tier model (see [`fast_tier_triage_model`]), and unknown providers
/// inherit the active model so we never dial a provider with another
/// company's model id.
/// Read an env var, returning `Some(trimmed)` only when it is set and non-blank.
fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn decompose_model(parent_model: Option<&str>) -> String {
    match parent_model
        .map(str::trim)
        .filter(|model| !model.is_empty())
    {
        Some(model) if is_anthropic_model_alias(model) => "haiku".to_string(),
        Some(model) if openai_gpt_model_family(model).is_some() => {
            fast_tier_triage_model(model, api::ProviderKind::OpenAi)
                .unwrap_or_else(|| "gpt-5.3-codex-spark".to_string())
        }
        Some(model) => model.to_string(),
        None => "haiku".to_string(),
    }
}

/// Decomposition/triage's model, derived from the connected catalog instead
/// of a hardcoded per-family literal: the same-provider Fast-tier model with
/// the highest `release_rank` (the newest Fast-tier entry), via the SAME
/// `RouteRole::Fast` AUTO selector the smart router uses elsewhere — so a
/// GPT-5.6 (or any future) parent triages onto its provider's CURRENT fast
/// model instead of a literal frozen at whatever was newest when this file was
/// last edited (the GPT-5.3-era `gpt-5.3-codex-spark` staleness this
/// generalizes away from). Uses `model_inventory_from_authorized_providers`
/// (an explicit provider list), not `connected_model_inventory` (which probes
/// live credentials), so this stays a deterministic function of `parent_model`
/// and `provider` — no ambient environment/credential dependence, and
/// testable without faking connectivity.
///
/// `None` when the provider's catalog has no Fast-tier entry at all (or the
/// resolved route is the main-model fallback, i.e. no genuine Fast pick
/// exists) — the caller falls back to the pre-Phase-2 literal in that case.
fn fast_tier_triage_model(parent_model: &str, provider: api::ProviderKind) -> Option<String> {
    let inventory = runtime::model_inventory_from_authorized_providers(parent_model, &[provider], &[]);
    let request = runtime::RouteRequest::new(runtime::RouteRole::Fast, parent_model);
    let decision = runtime::route_model(&request, &inventory);
    // A genuine AUTO pick is either the normal winner (`AutoSelector`) or
    // Phase 5's deterministic exploration rotation over the SAME rung
    // (`Exploration`) — both are real Fast-tier catalog picks, unlike
    // `MainOnly` (no qualifying candidate at all, which must fall back to the
    // caller's pre-Phase-2 literal). `Exploration` was unreachable here before
    // Phase 5 landed (this fn builds its own single-provider, no-history
    // inventory/context, so a slot could never be set) but is wrong-by-
    // inspection if left unhandled now that the variant exists.
    matches!(
        decision.source,
        runtime::RouteDecisionSource::AutoSelector | runtime::RouteDecisionSource::Exploration
    )
    .then_some(decision.resolved_model)
}

fn is_anthropic_model_alias(model: &str) -> bool {
    let raw = model.trim().to_ascii_lowercase();
    if raw.is_empty() {
        return false;
    }
    if raw == "opus" || raw == "sonnet" || raw == "haiku" || raw.starts_with("claude-") {
        return true;
    }
    api::resolve_registered_model_alias(&raw)
        .to_ascii_lowercase()
        .starts_with("claude-")
}

/// Parse the decomposition agent's structured `{ "subtasks": [{role, prompt}] }`
/// reply, dropping malformed or empty-prompt entries and capping the count.
fn parse_subtasks(structured: &Value) -> Vec<FanoutSubtask> {
    structured
        .get("subtasks")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let role = item.get("role").and_then(Value::as_str)?.trim();
                    let prompt = item.get("prompt").and_then(Value::as_str)?.trim();
                    (!prompt.is_empty()).then(|| FanoutSubtask {
                        role: if role.is_empty() { "subtask" } else { role }.to_string(),
                        prompt: prompt.to_string(),
                    })
                })
                .take(MAX_FANOUT_SUBTASKS)
                .collect()
        })
        .unwrap_or_default()
}

/// Stage 3 — run `subtasks` as parallel sub-agents via the `SpawnMultiAgent`
/// engine, returning the same JSON summary the tool produces (per-agent results
/// inline). Sub-agents inherit the parent provider and smart-route within that
/// family. BLOCKING — run inside
/// `spawn_blocking`. The caller MUST have gated on `DangerFullAccess` (see the
/// module note): this runs agents without a permission check.
pub fn run_fanout_spawn(
    subtasks: &[FanoutSubtask],
    parent_model: Option<&str>,
) -> Result<String, ToolError> {
    run_fanout_spawn_with_timeout(subtasks, parent_model, SPAWN_MULTI_AGENT_WAIT_TIMEOUT, None)
}

/// Run fan-out with a result collection window and optional per-agent wall-clock
/// budget. Stragglers are reported as `still_running` when the collection window
/// elapses; pass a budget only when the agent itself should terminate with an
/// explicit timeout failure.
pub fn run_fanout_spawn_with_timeout(
    subtasks: &[FanoutSubtask],
    parent_model: Option<&str>,
    wait_timeout: Duration,
    agent_time_budget: Option<Duration>,
) -> Result<String, ToolError> {
    run_fanout_spawn_with_timeout_and_hooks(
        subtasks,
        parent_model,
        wait_timeout,
        agent_time_budget,
        None,
        None,
    )
}

/// Run fan-out while applying configured hooks to each spawned sub-agent.
pub fn run_fanout_spawn_with_timeout_and_hooks(
    subtasks: &[FanoutSubtask],
    parent_model: Option<&str>,
    wait_timeout: Duration,
    agent_time_budget: Option<Duration>,
    hook_config: Option<&runtime::RuntimeHookConfig>,
    parent_session_id: Option<&str>,
) -> Result<String, ToolError> {
    let agents: Vec<Value> = subtasks
        .iter()
        .map(|subtask| json!({ "prompt": subtask.prompt, "description": subtask.role }))
        .collect();
    let mut input: SpawnMultiAgentInput = serde_json::from_value(json!({ "agents": agents }))
        .map_err(|error| ToolError::InvalidInput(error.to_string()))?;
    input.parent_session_id = parent_session_id.map(str::to_string);
    run_spawn_multi_agent_with_timeout_and_hooks(
        &input,
        parent_model,
        None,
        wait_timeout,
        agent_time_budget,
        true,
        hook_config,
    )
}

// ===========================================================================
// P2 — intent triage + automatic self-consistency (Council majority) mode.
//
// `clarify_intent` runs one cheap triage agent (same setup as the decompose
// agent) to clarify the user's request and pick a fan-out strategy.
// `run_self_consistency_fanout` spawns k identical answer agents and votes
// their replies through the existing `execute_council` to synthesize one
// answer. Both fail closed to `None`, so the caller falls back to the existing
// decompose path on any failure.
// ===========================================================================

/// Which fan-out strategy best fits a request, as classified by
/// [`clarify_intent`]. The triage LLM picks one from the request's *meaning*, so
/// adding a strategy here is a matter of teaching the prompt what it means — no
/// keyword table, and it generalizes across languages and providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanoutMode {
    /// Analysis/judgement question — spawn k independent re-answers and vote
    /// them through the Council (self-consistency).
    SelfConsistency,
    /// Implementation/breadth task — split into independent subtasks by
    /// file or layer (the existing decompose path).
    Decompose,
    /// Trivial/single-step request — fan-out is not worth it; run a solo turn.
    Solo,
    /// A hard or RECURRING bug where one plausible hypothesis is easily wrong
    /// (timing/concurrency, streaming, a fix that "didn't stick"). Spawn
    /// independent root-cause finders that each try to *refute* their own
    /// hypothesis (simulate / reproduce), so a plausible-but-wrong fix is killed
    /// before it ships — the adversarial diagnosis that a single linear pass
    /// keeps failing. Chosen by the triage LLM from meaning, not keywords.
    Diagnose,
}

/// One cheap triage of the user's request: the clarified one-sentence `intent`,
/// the recommended `mode`, and the model's `confidence` in `0..=1`. Produced by
/// [`clarify_intent`].
#[derive(Debug, Clone)]
pub struct IntentTriage {
    pub intent: String,
    pub mode: FanoutMode,
    pub confidence: f64,
}

/// Wall-clock budget for the one-shot intent-triage agent. Short — triage is a
/// single cheap classification reply, like decomposition.
const TRIAGE_TIMEOUT: Duration = Duration::from_secs(15);

/// Tighter budget for the non-breadth pre-spawn path (ultracode
/// `Pipeline`/`DelegateOne`), where the triage runs serially before the model
/// turn and may choose `diagnose`, `decompose`, `self_consistency`, or `solo`
/// from semantic meaning. A slow/hung triage falls back to the model-led turn
/// fast instead of parking a blocking-pool thread (and the user's first token)
/// for the full 15s.
const TRIAGE_TIMEOUT_NONBREADTH: Duration = Duration::from_secs(8);

/// Run ONE cheap triage agent that clarifies the user's intent in a single
/// sentence and classifies which fan-out [`FanoutMode`] fits. Mirrors
/// [`decompose_for_fanout_with_timeout_and_hooks`]'s agent setup exactly: the
/// cheap [`decompose_model`] for the active provider, `StructuredOutput` on
/// Anthropic and a JSON-object-in-text reply on every other provider (keyed off
/// the *effective* run model), and a short 15s timeout.
///
/// Returns `None` on any failure/timeout/empty reply so the caller falls back
/// to the existing decompose path. A low-confidence (`< 0.4`) triage is still
/// returned verbatim — the caller decides whether to trust the clarified intent
/// or use the raw text. BLOCKING — run inside `spawn_blocking`.
///
/// `breadth` selects the wall-clock budget: a breadth fan-out turn always
/// consumes the triage, so it gets the full [`TRIAGE_TIMEOUT`]; a non-breadth
/// pre-spawn discards it on any non-`diagnose` verdict, so it gets the tighter
/// [`TRIAGE_TIMEOUT_NONBREADTH`] to bound the serial latency before first token.
#[must_use]
pub fn clarify_intent(
    user_text: &str,
    breadth: bool,
    parent_model: Option<&str>,
    hook_config: Option<&runtime::RuntimeHookConfig>,
    parent_session_id: Option<&str>,
) -> Option<IntentTriage> {
    let timeout = if breadth {
        TRIAGE_TIMEOUT
    } else {
        TRIAGE_TIMEOUT_NONBREADTH
    };
    // Same provider-aware cheap model and structured-vs-text branch as
    // `decompose_for_fanout_with_timeout_and_hooks`: honour the effective run
    // model (env override before parent) so a GPT override is not forced onto
    // StructuredOutput (BUG-R15) and returns empty.
    let model = decompose_model(parent_model);
    let effective_model = non_empty_env(AGENT_MODEL_ENV).unwrap_or_else(|| model.clone());
    let structured = is_anthropic_model_alias(&effective_model);
    let input = AgentInput {
        allow_cross_provider: false,
        description: "fan-out intent triage".to_string(),
        prompt: clarify_prompt(user_text, structured),
        // Same micro-prompt harness as decomposition (see above).
        subagent_type: Some("classifier".to_string()),
        name: Some("triage".to_string()),
        model: None,
        cwd: None,
        schema: structured.then(clarify_schema),
        workflow_member: false,
        background: Some(false),
        // Micro-prompt classifier harness (no workspace tools); the spawn
        // clamp is not threaded through this internal helper path.
        parent_permission_mode: None,
        parent_session_id: parent_session_id.map(str::to_string),
        tool_call_id: None,
        mcp_passthrough: None,
        api_concurrency: None,
        time_budget: Some(timeout),
        prior_failures: 0,
        route_reason: None,
        route_role: None,
        route_complexity: None,
        route_risk: None,
        route_source: None,
        route_model: None,
        route_fallback_models: Vec::new(),
        route_effort: None,
        judged_agent: None,
    };
    let output =
        execute_agent_with_parent_model_and_hooks(input, Some(&model), None, hook_config).ok()?;
    let completion = wait_for_agent_completions(&[output.agent_id], timeout)
        .into_iter()
        .next()?;
    // Prefer the captured `StructuredOutput` (Anthropic); otherwise parse a JSON
    // object out of the reply text (non-Anthropic, and a salvage path when a
    // structured reply came back empty).
    completion
        .structured
        .as_ref()
        .and_then(parse_intent_triage_value)
        .or_else(|| completion.result.as_deref().and_then(parse_intent_triage))
}

/// The `StructuredOutput` schema the Anthropic triage answers in:
/// `{ "intent", "mode", "confidence" }`.
fn clarify_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "the user's request clarified in ONE sentence, with no added or reinterpreted scope"
            },
            "mode": {
                "type": "string",
                "enum": ["self_consistency", "decompose", "solo", "diagnose"],
                "description": "which fan-out strategy fits the request"
            },
            "confidence": {
                "type": "number",
                "description": "confidence in the mode classification, between 0 and 1"
            }
        },
        "required": ["intent", "mode", "confidence"]
    })
}

/// The triage prompt. `structured` selects the reply contract: the
/// `StructuredOutput` tool (Anthropic) or a raw JSON object in the text reply
/// (every other provider, where forcing the tool yields no content).
fn clarify_prompt(user_text: &str, structured: bool) -> String {
    let reply = if structured {
        "Reply via the StructuredOutput tool."
    } else {
        "Reply with ONLY a single JSON object and NOTHING else — no prose, no markdown fences, \
         and do NOT call any tools. Shape: \
         {\"intent\":\"one sentence\",\"mode\":\"self_consistency|decompose|solo|diagnose\",\"confidence\":0.0}."
    };
    format!(
        "Clarify the user's request — do NOT reinterpret it, add scope, or solve it. State what the \
         user is actually asking for in ONE sentence as `intent`. Then classify `mode` as exactly \
         one of:\n\
         - self_consistency: analysis, judgement, 'which is better', or review questions where \
         independent re-answers to the same question should be voted on for a more reliable result.\n\
         - decompose: implementation or breadth tasks that split cleanly into independent parallel \
         work by file or layer, including requests in any language that semantically ask for \
         an agent team, collaboration, several perspectives, or parallel investigation. Do \
         not key this off English/Korean keyword lists or a literal phrase; infer it from \
         the user's meaning.\n\
         - diagnose: a HARD or RECURRING bug where one plausible explanation is easily wrong — \
         a fix that didn't stick ('still broken', 'again', 'you keep failing'), a timing / \
         concurrency / streaming / heisenbug, or a symptom whose cause is genuinely uncertain. \
         These need independent root-cause hypotheses that each get REFUTED (simulated / \
         reproduced) before a fix, not one confident guess. Judge this from the MEANING of the \
         request, not specific words.\n\
         - solo: trivial or single-step requests where fanning out is not worth it.\n\
         Also give `confidence` between 0 and 1 in the classification. {reply}\n\nUSER REQUEST:\n{user_text}"
    )
}

/// Parse an [`IntentTriage`] from a free-text reply containing a JSON object
/// (the non-Anthropic path). Extracts the first balanced `{...}` so prose or
/// markdown fences do not defeat parsing, then reuses [`parse_intent_triage_value`].
/// An unparseable or incomplete reply yields `None` — the caller's graceful
/// fallback.
fn parse_intent_triage(text: &str) -> Option<IntentTriage> {
    let object = extract_json_object(text)?;
    let value: Value = serde_json::from_str(object).ok()?;
    parse_intent_triage_value(&value)
}

/// Build an [`IntentTriage`] from an already-parsed JSON object (the Anthropic
/// `StructuredOutput` path, and the inner step of [`parse_intent_triage`]).
/// Requires a non-empty `intent` and a recognized `mode`; an absent `confidence`
/// defaults to `0.0` and any value is clamped to `0..=1`. Returns `None` when a
/// required field is missing or the mode is unknown.
fn parse_intent_triage_value(value: &Value) -> Option<IntentTriage> {
    let intent = value.get("intent").and_then(Value::as_str)?.trim();
    if intent.is_empty() {
        return None;
    }
    let mode = value
        .get("mode")
        .and_then(Value::as_str)
        .and_then(parse_fanout_mode)?;
    let confidence = value
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    Some(IntentTriage {
        intent: intent.to_string(),
        mode,
        confidence,
    })
}

/// Map a `mode` string (the model's classification) to a [`FanoutMode`],
/// tolerant of hyphen/underscore/space spelling. Unknown values yield `None`.
fn parse_fanout_mode(raw: &str) -> Option<FanoutMode> {
    match raw
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_")
        .as_str()
    {
        "self_consistency" | "selfconsistency" => Some(FanoutMode::SelfConsistency),
        "decompose" => Some(FanoutMode::Decompose),
        "solo" => Some(FanoutMode::Solo),
        "diagnose" | "diagnosis" => Some(FanoutMode::Diagnose),
        _ => None,
    }
}

/// Stage 3 (self-consistency variant) — spawn `k` identical answer agents for
/// `intent`, then vote their replies through the existing [`execute_council`]
/// and synthesize one answer string. All k agents run the SAME prompt (the
/// clarified intent), so the Council's self-consistency majority picks the most
/// agreed-upon answer. `k` is capped at [`MAX_FANOUT_SUBTASKS`] to stay within
/// the `SpawnMultiAgent` agent cap.
///
/// Returns `None` when spawning fails or no candidate produced text, so the
/// caller falls back to the existing path. BLOCKING — run inside
/// `spawn_blocking`; the caller MUST have gated on `DangerFullAccess` (see the
/// module note).
#[must_use]
pub fn run_self_consistency_fanout(
    intent: &str,
    parent_model: Option<&str>,
    k: usize,
    hook_config: Option<&runtime::RuntimeHookConfig>,
    parent_session_id: Option<&str>,
) -> Option<String> {
    let intent = intent.trim();
    if intent.is_empty() {
        return None;
    }
    let k = k.clamp(1, MAX_FANOUT_SUBTASKS);
    let prompt = self_consistency_prompt(intent);
    let subtasks: Vec<FanoutSubtask> = (0..k)
        .map(|i| FanoutSubtask {
            role: format!("candidate {i}"),
            prompt: prompt.clone(),
        })
        .collect();
    let summary = run_fanout_spawn_with_timeout_and_hooks(
        &subtasks,
        parent_model,
        SPAWN_MULTI_AGENT_WAIT_TIMEOUT,
        Some(AUTO_FANOUT_AGENT_TIMEOUT),
        hook_config,
        parent_session_id,
    )
    .ok()?;
    synthesize_self_consistency(&summary)
}

/// The self-consistency answer prompt run by every candidate agent. Asks for a
/// decisive, self-contained answer so independent replies can be compared and
/// voted on by the Council.
fn self_consistency_prompt(intent: &str) -> String {
    format!(
        "Answer the following request definitively. Reason it through, then commit to a single, \
         concrete answer and state your conclusion clearly up front so it can be compared against \
         other independent answers to the same request. Be self-contained: you see ONLY this \
         prompt.\n\nREQUEST:\n{intent}"
    )
}

// ===========================================================================
// Diagnose — adversarial root-cause fan-out (FanoutMode::Diagnose).
// ===========================================================================

/// The independent angles an adversarial diagnosis fans out over. Each is a
/// distinct place a bug can hide; one finder per lens gives the *multiple
/// independent perspectives* a single linear pass lacks (the gap that let
/// plausible-but-wrong fixes keep shipping). Deliberately general — not tied to
/// any subsystem — so the same set generalizes across bugs and across models;
/// each finder reasons about meaning under its lens, it is not a keyword match.
const DIAGNOSE_LENSES: &[(&str, &str)] = &[
    (
        "data-flow & state",
        "how a value or piece of state is produced, transformed, cached, and consumed — \
         where it diverges from what the surrounding code assumes",
    ),
    (
        "timing & concurrency",
        "ordering, races, async/await, buffering, lifecycle and event timing — anything \
         that depends on WHEN things happen relative to each other",
    ),
    (
        "boundary & edge cases",
        "empty/full, first/last, zero, overflow, off-by-one, and unusual or malformed \
         inputs the happy path skips",
    ),
    (
        "assumptions & the prior fix",
        "a violated precondition or invariant, or a previous 'fix' that was plausible but \
         wrong (the symptom persists) — treat the most recent change as SUSPECT, not ground truth",
    ),
];

/// Display labels for the diagnose fan-out's finders (one per lens), so the host
/// can render the agent tree before/while they run without duplicating the lens
/// list. Kept in lock-step with [`DIAGNOSE_LENSES`] and the roles
/// [`run_diagnose_fanout`] assigns.
#[must_use]
pub fn diagnose_lens_labels() -> Vec<String> {
    DIAGNOSE_LENSES
        .iter()
        .map(|(lens, _)| format!("diagnose · {lens}"))
        .collect()
}

/// Stage 3 (diagnose variant) — spawn one independent finder per
/// [`DIAGNOSE_LENSES`] angle, each forming a root-cause hypothesis and trying to
/// REFUTE it with a hand-simulation or reproduction before reporting. Their
/// combined findings become the model's pre-analysis, so it synthesizes and
/// cross-checks the competing hypotheses instead of committing to one unverified
/// guess — the adversarial method that cracks bugs a single linear turn keeps
/// surface-fixing. Provider-agnostic: runs on whatever `parent_model` is.
///
/// Returns `None` when spawning fails or no finder produced text, so the caller
/// falls back to the existing path. BLOCKING — run inside `spawn_blocking`; the
/// caller MUST have gated on `DangerFullAccess` (see the module note).
#[must_use]
pub fn run_diagnose_fanout(
    bug: &str,
    parent_model: Option<&str>,
    hook_config: Option<&runtime::RuntimeHookConfig>,
    parent_session_id: Option<&str>,
) -> Option<String> {
    let bug = bug.trim();
    if bug.is_empty() {
        return None;
    }
    let subtasks: Vec<FanoutSubtask> = DIAGNOSE_LENSES
        .iter()
        .map(|(lens, focus)| FanoutSubtask {
            role: format!("diagnose · {lens}"),
            prompt: diagnose_finder_prompt(bug, lens, focus),
        })
        .collect();
    run_fanout_spawn_with_timeout_and_hooks(
        &subtasks,
        parent_model,
        SPAWN_MULTI_AGENT_WAIT_TIMEOUT,
        Some(DIAGNOSE_FANOUT_AGENT_TIMEOUT),
        hook_config,
        parent_session_id,
    )
    .ok()
}

/// The finder prompt for one diagnostic lens. Encodes the discipline that beats a
/// plausible-but-wrong fix: form ONE hypothesis under this lens, then actively
/// try to REFUTE it (simulate the logic, or construct a reproduction), and report
/// it as refuted if it cannot survive. A surviving hypothesis must carry concrete
/// evidence (an exact code location and a reproduction recipe), never a guess.
fn diagnose_finder_prompt(bug: &str, lens: &str, focus: &str) -> String {
    format!(
        "You are one of several INDEPENDENT root-cause finders for a bug. Investigate it ONLY \
         through the lens of {lens} — {focus}. You see only this prompt; other finders cover other \
         lenses.\n\n\
         Method (this is the whole point — do not skip it):\n\
         1. Read the relevant code and form ONE concrete root-cause hypothesis under your lens.\n\
         2. Then try HARD to REFUTE your own hypothesis: simulate the logic by hand with realistic \
         values, or construct a minimal reproduction/scenario. A plausible-sounding cause is worthless \
         until you have tried to break it.\n\
         3. If it does not survive, say so plainly (\"REFUTED: …\") — a confident wrong answer is the \
         exact failure mode we are defeating.\n\
         4. If it survives refutation, report it with concrete evidence: exact file:line and how to \
         reproduce or simulate the failure.\n\n\
         Be terse and decisive. Do NOT propose or apply a fix — only the verified (or refuted) root \
         cause, so the parent can cross-check all lenses before any change.\n\n\
         BUG:\n{bug}"
    )
}

/// Vote the per-candidate replies in a `run_fanout_spawn_*` summary through the
/// Council and synthesize one answer. On a `BestOf` majority, returns the
/// winning candidate's text plus a `(self-consistency: N agreed)` note. On a
/// `Tie`, returns an honest synthesis listing the distinct candidate answers
/// labelled as unreconciled. Returns `None` when no candidate produced text or
/// the Council rejects the input.
fn synthesize_self_consistency(summary: &str) -> Option<String> {
    let candidates = summary_to_council_candidates(summary);
    if candidates.iter().all(|(text, _)| text.trim().is_empty()) {
        return None;
    }
    let council_input = CouncilInput {
        candidates: candidates
            .iter()
            .map(|(text, status)| CouncilCandidateInput {
                text: text.clone(),
                status: status.clone(),
            })
            .collect(),
    };
    let outcome = execute_council(&council_input).ok()?.outcome;
    match outcome {
        runtime::CouncilOutcome::BestOf {
            winner_index,
            supporting_indices,
            ..
        } => {
            let winner = candidates
                .get(winner_index)
                .map(|(text, _)| text.trim())
                .filter(|text| !text.is_empty())?;
            let agreed = supporting_indices.len();
            Some(format!("{winner}\n\n(self-consistency: {agreed} agreed)"))
        }
        runtime::CouncilOutcome::Synthesized { text, .. } => {
            let text = text.trim();
            (!text.is_empty()).then(|| text.to_string())
        }
        runtime::CouncilOutcome::Tie { .. } => synthesize_unreconciled(&candidates),
    }
}

/// Honest no-majority synthesis: list the distinct (case/whitespace-insensitive)
/// non-empty candidate answers, labelled as unreconciled, so the caller never
/// presents a faked winner. `None` when there is no candidate text at all.
fn synthesize_unreconciled(candidates: &[(String, Option<String>)]) -> Option<String> {
    let mut keys: Vec<String> = Vec::new();
    let mut distinct: Vec<&str> = Vec::new();
    for (text, _) in candidates {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = trimmed
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();
        if !keys.contains(&key) {
            keys.push(key);
            distinct.push(trimmed);
        }
    }
    if distinct.is_empty() {
        return None;
    }
    let mut out =
        String::from("(self-consistency: no majority — unreconciled candidate answers below)\n");
    for (i, answer) in distinct.iter().enumerate() {
        let _ = write!(out, "\n{}. {answer}", i + 1);
    }
    Some(out)
}

/// Parse the `agents` array of a `run_fanout_spawn_*` JSON summary into
/// `(text, status)` candidate pairs for [`CouncilInput`]. Mirrors the summary
/// shape `format_fanout_analysis` reads: `agents[].result` (the answer text,
/// empty string when the agent produced none) and `agents[].status`. The
/// candidates keep their array order, so a Council `winner_index` indexes
/// straight back into the returned vec. A summary without a parseable `agents`
/// array yields an empty list. Pure — runs no agents.
fn summary_to_council_candidates(summary: &str) -> Vec<(String, Option<String>)> {
    let Ok(parsed) = serde_json::from_str::<Value>(summary) else {
        return Vec::new();
    };
    let Some(agents) = parsed.get("agents").and_then(Value::as_array) else {
        return Vec::new();
    };
    agents
        .iter()
        .map(|agent| {
            let text = agent
                .get("result")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let status = agent
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_string);
            (text, status)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_fanout_collection_window_is_shorter_than_manual_spawn() {
        assert!(
            AUTO_FANOUT_AGENT_TIMEOUT < SPAWN_MULTI_AGENT_WAIT_TIMEOUT,
            "automatic pre-analysis should not monopolize the foreground turn as long as manual SpawnMultiAgent"
        );
    }

    #[test]
    fn diagnose_fanout_agent_budget_matches_collection_window() {
        assert_eq!(
            DIAGNOSE_FANOUT_AGENT_TIMEOUT, SPAWN_MULTI_AGENT_WAIT_TIMEOUT,
            "diagnose finders need the full collection window; a shorter hard deadline discards terminal failures with no salvage"
        );
        assert!(DIAGNOSE_FANOUT_AGENT_TIMEOUT > AUTO_FANOUT_AGENT_TIMEOUT);
    }

    #[test]
    fn parse_subtasks_keeps_valid_caps_and_skips_blank() {
        let structured = json!({
            "subtasks": [
                {"role": "api layer", "prompt": "analyze api error handling"},
                {"role": "", "prompt": "analyze tui error handling"},
                {"role": "blank", "prompt": "   "},
                {"role": "ok", "prompt": "x"},
                {"role": "fifth", "prompt": "over the cap"}
            ]
        });
        let subtasks = parse_subtasks(&structured);
        // Blank-prompt entry dropped, count capped at MAX_FANOUT_SUBTASKS.
        assert_eq!(subtasks.len(), MAX_FANOUT_SUBTASKS);
        assert_eq!(subtasks[0].role, "api layer");
        // Empty role is backfilled to a placeholder, never empty.
        assert_eq!(subtasks[1].role, "subtask");
        assert!(subtasks.iter().all(|s| !s.prompt.trim().is_empty()));
    }

    #[test]
    fn parse_subtasks_empty_or_missing_yields_none() {
        assert!(parse_subtasks(&json!({ "subtasks": [] })).is_empty());
        assert!(parse_subtasks(&json!({})).is_empty());
        assert!(parse_subtasks(&json!({ "subtasks": "nope" })).is_empty());
    }

    #[test]
    fn decompose_prompt_asks_for_bounded_agent_work() {
        let prompt = decompose_prompt("analyze backend, UI, and ops", false);

        assert!(prompt.contains("bounded analysis"));
        assert!(prompt.contains("stop tool calls once it has enough evidence"));
        assert!(prompt.contains("file:line"));
    }

    #[test]
    fn decompose_model_follows_active_provider_not_hardcoded_haiku() {
        // Anthropic active model → the cheap haiku tier (preserves the optimization).
        assert_eq!(decompose_model(Some("opus")), "haiku");
        assert_eq!(decompose_model(Some("opus-4.6")), "haiku");
        assert_eq!(decompose_model(Some("claude-sonnet-4-6")), "haiku");
        assert_eq!(decompose_model(Some("haiku")), "haiku");
        assert_eq!(
            api::resolve_model_alias(&decompose_model(Some("opus"))),
            "claude-haiku-4-5-20251001"
        );
        // OpenAI/GPT active model → the catalog's current highest-release-rank
        // Fast-tier model — inventory-derived, never a Claude id. Phase 8:
        // `gpt-5.6-luna` (rank 56) is now OpenAI's own declared Fast-and-
        // affordable model (Codex model cache, priority 3) and outranks it,
        // beating both `gpt-5.5-fast` (rank 55; no longer Fast-tier — `-fast`
        // is a priority SERVICE tier of the same-quality gpt-5.5, now declared
        // balanced like its parent) and the GPT-5.3-era `gpt-5.3-codex-spark`
        // (rank 53).
        assert_eq!(decompose_model(Some("gpt-5.5")), "gpt-5.6-luna");
        assert_eq!(decompose_model(Some("gpt-5.5-2026-04-23")), "gpt-5.6-luna");
        // A GPT-5.6 parent triages onto the SAME current fast model — not the
        // stale 5.3-era literal — proving the pick tracks the provider's
        // catalog rather than a per-family hardcode.
        assert_eq!(decompose_model(Some("gpt-5.6-sol")), "gpt-5.6-luna");
        assert_eq!(decompose_model(Some("gpt-5.6-terra")), "gpt-5.6-luna");
        assert_eq!(decompose_model(Some("gpt-5.6-luna")), "gpt-5.6-luna");
        // Unknown/non-GPT providers still inherit the active provider model
        // (xAI's catalog has no Fast-tier entry, so the inventory-derived
        // lookup returns `None` and the caller keeps the verbatim model).
        assert_eq!(decompose_model(Some("grok-3")), "grok-3");
        // No active model (non-live harness) → prior cheap default.
        assert_eq!(decompose_model(None), "haiku");
        assert_eq!(decompose_model(Some("  ")), "haiku");
    }

    #[test]
    fn fast_tier_triage_model_falls_back_to_none_with_no_catalog_fast_entry() {
        // xAI's only catalog entry (grok-3) has no Fast tier, so the caller's
        // literal fallback applies — proving the fallback path still works.
        assert_eq!(
            super::fast_tier_triage_model("grok-3", api::ProviderKind::Xai),
            None
        );
    }

    #[test]
    fn parse_subtasks_from_text_tolerates_prose_fences_and_inner_braces() {
        // Bare JSON object.
        let bare =
            r#"{"subtasks":[{"role":"api","prompt":"do api"},{"role":"ui","prompt":"do ui"}]}"#;
        assert_eq!(parse_subtasks_from_text(bare).len(), 2);
        // Wrapped in prose + markdown fences (a non-Anthropic model often does this).
        let wrapped =
            "Sure:\n```json\n{\"subtasks\":[{\"role\":\"api\",\"prompt\":\"x\"}]}\n```\nDone.";
        let subtasks = parse_subtasks_from_text(wrapped);
        assert_eq!(subtasks.len(), 1);
        assert_eq!(subtasks[0].role, "api");
        // A brace inside a string value must not end the object early.
        let inner = r#"prefix {"subtasks":[{"role":"a","prompt":"use {x} and }y{ here"}]} suffix"#;
        assert_eq!(
            parse_subtasks_from_text(inner)[0].prompt,
            "use {x} and }y{ here"
        );
        // Unparseable reply → empty (the caller's graceful single-agent fallback).
        assert!(parse_subtasks_from_text("no json at all").is_empty());
        assert!(parse_subtasks_from_text("").is_empty());
    }

    // --- P2: intent triage parsing ---

    #[test]
    fn parse_intent_triage_accepts_valid_object() {
        let triage = parse_intent_triage(
            r#"{"intent":"compare the two routing designs","mode":"self_consistency","confidence":0.82}"#,
        )
        .expect("valid triage JSON");
        assert_eq!(triage.intent, "compare the two routing designs");
        assert_eq!(triage.mode, FanoutMode::SelfConsistency);
        assert!((triage.confidence - 0.82).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_intent_triage_tolerates_prose_fences_and_mode_spelling() {
        // Wrapped in prose + markdown fences, with a hyphenated mode spelling.
        let wrapped = "Here:\n```json\n{\"intent\":\"add a flag\",\"mode\":\"Self-Consistency\",\"confidence\":0.5}\n```";
        let triage = parse_intent_triage(wrapped).expect("triage parses through fences");
        assert_eq!(triage.mode, FanoutMode::SelfConsistency);
        assert_eq!(triage.intent, "add a flag");
        // decompose + solo round-trip.
        assert_eq!(
            parse_intent_triage(r#"{"intent":"x","mode":"decompose","confidence":0.9}"#)
                .unwrap()
                .mode,
            FanoutMode::Decompose
        );
        assert_eq!(
            parse_intent_triage(r#"{"intent":"x","mode":"solo","confidence":0.1}"#)
                .unwrap()
                .mode,
            FanoutMode::Solo
        );
        // diagnose round-trips, tolerant of the "diagnosis" spelling — the LLM
        // picks this for a hard/recurring bug from meaning, no keyword.
        assert_eq!(
            parse_intent_triage(r#"{"intent":"x","mode":"diagnose","confidence":0.9}"#)
                .unwrap()
                .mode,
            FanoutMode::Diagnose
        );
        assert_eq!(
            parse_intent_triage(r#"{"intent":"x","mode":"Diagnosis","confidence":0.9}"#)
                .unwrap()
                .mode,
            FanoutMode::Diagnose
        );
    }

    #[test]
    fn clarify_prompt_tells_model_to_infer_collaboration_semantically() {
        let prompt = clarify_prompt(
            "verify and repair with the right collaboration strategy",
            false,
        );

        assert!(prompt.contains("requests in any language"));
        assert!(prompt.contains("semantically ask for"));
        assert!(prompt.contains("Do not key this off English/Korean keyword lists"));
    }

    #[test]
    fn parse_intent_triage_keeps_low_confidence_verbatim() {
        // Low confidence is preserved, not forced — the caller decides.
        let triage =
            parse_intent_triage(r#"{"intent":"unsure ask","mode":"solo","confidence":0.12}"#)
                .expect("low-confidence triage still returned");
        assert!((triage.confidence - 0.12).abs() < f64::EPSILON);
        // Out-of-range confidence is clamped into 0..=1; missing defaults to 0.
        assert!(
            (parse_intent_triage(r#"{"intent":"a","mode":"solo","confidence":9.0}"#)
                .unwrap()
                .confidence
                - 1.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            parse_intent_triage(r#"{"intent":"a","mode":"solo"}"#)
                .unwrap()
                .confidence
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn parse_intent_triage_rejects_invalid_missing_and_unknown_mode() {
        // No JSON at all.
        assert!(parse_intent_triage("not json").is_none());
        assert!(parse_intent_triage("").is_none());
        // Missing required intent.
        assert!(parse_intent_triage(r#"{"mode":"solo","confidence":0.5}"#).is_none());
        // Blank intent is treated as missing.
        assert!(
            parse_intent_triage(r#"{"intent":"   ","mode":"solo","confidence":0.5}"#).is_none()
        );
        // Missing mode.
        assert!(parse_intent_triage(r#"{"intent":"x","confidence":0.5}"#).is_none());
        // Unknown mode value.
        assert!(
            parse_intent_triage(r#"{"intent":"x","mode":"ensemble","confidence":0.5}"#).is_none()
        );
    }

    // --- P2: self-consistency candidate extraction + synthesis ---

    #[test]
    fn summary_to_council_candidates_extracts_result_and_status_in_order() {
        let summary = r#"{
            "status": "completed",
            "agents": [
                {"index": 0, "result": "Answer A", "status": "completed"},
                {"index": 1, "result": "", "status": "failed"},
                {"index": 2, "result": "Answer A", "status": "completed"}
            ]
        }"#;
        let candidates = summary_to_council_candidates(summary);
        assert_eq!(candidates.len(), 3);
        assert_eq!(
            candidates[0],
            ("Answer A".to_string(), Some("completed".to_string()))
        );
        assert_eq!(candidates[1], (String::new(), Some("failed".to_string())));
        assert_eq!(candidates[2].0, "Answer A");
    }

    #[test]
    fn summary_to_council_candidates_handles_missing_and_unparseable() {
        // No `agents` array → empty.
        assert!(summary_to_council_candidates(r#"{"status":"completed"}"#).is_empty());
        // Unparseable JSON → empty (never panics).
        assert!(summary_to_council_candidates("not json").is_empty());
        // Missing `result`/`status` default to empty string / None.
        let candidates = summary_to_council_candidates(r#"{"agents":[{"index":0}]}"#);
        assert_eq!(candidates, vec![(String::new(), None)]);
    }

    #[test]
    fn synthesize_self_consistency_picks_majority_with_agreed_note() {
        let summary = r#"{"agents":[
            {"index":0,"result":"Use ProviderClient routing","status":"completed"},
            {"index":1,"result":"use providerclient routing","status":"completed"},
            {"index":2,"result":"Rewrite the runtime","status":"completed"}
        ]}"#;
        let out = synthesize_self_consistency(summary).expect("majority synthesizes an answer");
        assert!(out.contains("Use ProviderClient routing"));
        assert!(out.contains("(self-consistency: 2 agreed)"));
    }

    #[test]
    fn synthesize_self_consistency_ties_list_distinct_unreconciled() {
        // Three unique answers → no majority → honest unreconciled synthesis.
        let summary = r#"{"agents":[
            {"index":0,"result":"Answer A","status":"completed"},
            {"index":1,"result":"Answer B","status":"completed"},
            {"index":2,"result":"Answer C","status":"completed"}
        ]}"#;
        let out = synthesize_self_consistency(summary).expect("tie still synthesizes honestly");
        assert!(out.contains("no majority"));
        assert!(out.contains("Answer A"));
        assert!(out.contains("Answer B"));
        assert!(out.contains("Answer C"));
        assert!(!out.contains("agreed)"));
    }

    #[test]
    fn synthesize_self_consistency_none_when_no_candidate_text() {
        // All blank / failed → None so the caller falls back.
        let summary = r#"{"agents":[
            {"index":0,"result":"","status":"failed"},
            {"index":1,"result":"   ","status":"failed"}
        ]}"#;
        assert!(synthesize_self_consistency(summary).is_none());
        // No agents at all → None.
        assert!(synthesize_self_consistency(r#"{"status":"completed"}"#).is_none());
    }

}
