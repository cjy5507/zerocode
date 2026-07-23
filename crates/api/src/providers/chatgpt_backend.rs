//! ChatGPT (OpenAI) subscription backend over the Responses API.
//!
//! When a user signs in with `zo login openai`, their ChatGPT
//! `access_token` is sent to `chatgpt.com/backend-api/codex/responses` — the
//! same backend OpenAI's Codex CLI uses — instead of the public
//! `api.openai.com` Chat Completions endpoint, so usage bills against the
//! ChatGPT subscription rather than API credits.
//!
//! The wire format is the OpenAI Responses API: an `input` item list plus a
//! stream of `response.*` SSE events. That differs from Chat Completions, so
//! this module owns its own request builder (here) and SSE translation
//! (`sse` submodule, added alongside the client).

use std::collections::{BTreeSet, VecDeque};
use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

use super::{PromptCacheStrategy, openai_gpt_model_family, shared_http_client};
use crate::error::ApiError;
use core_types::StreamRetryNotice;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    EffortLevel, ImageSource, InputContentBlock, InputMessage, MessageDelta, MessageDeltaEvent,
    MessageRequest, MessageResponse, MessageStartEvent, MessageStopEvent, OutputContentBlock,
    ReasoningRequest,
    StreamEvent, SystemBlock, ToolChoice, ToolDefinition, ToolLedgerView, Usage,
};

/// ChatGPT backend Responses endpoint (Codex variant).
pub const CHATGPT_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
/// `originator` header value identifying the Codex Rust client.
pub const ORIGINATOR: &str = "codex_cli_rs";
/// `OpenAI-Beta` header value opting into the experimental Responses surface.
pub const OPENAI_BETA_RESPONSES: &str = "responses=experimental";
/// `User-Agent` the Codex backend expects. Some models are GATED on this
/// client fingerprint: `gpt-5.6-luna` answers 404 "Model not found" to any
/// request whose user agent does not carry the `codex_cli_rs` product token —
/// verified live (2026-07-13) with the same token, body, and headers, only
/// the UA differing; the version and trailing segments are not checked
/// (`codex_cli_rs` alone passes, `zo/0.1.0` 404s). Mirrors the Claude Max
/// OAuth identity-block fingerprint on the Anthropic path. The `zo` suffix
/// keeps the client honestly identifiable server-side.
pub const USER_AGENT: &str = "codex_cli_rs/0.144.1 (Mac OS; arm64) zo";

/// Build the Responses API request body from zo's provider-agnostic
/// [`MessageRequest`].
///
/// `instructions` carries zo's system prompt — the Responses API takes the
/// system role as a top-level field, not as an `input` item. The conversation
/// history becomes the `input` item list. `store` is forced to `false` (the
/// ChatGPT backend rejects stateful requests) and reasoning encrypted content
/// is requested so multi-turn reasoning continuity is preserved.
///
/// 3-arg convenience wrapper around [`build_responses_request_for_session`]
/// (empty session id) for tests that don't exercise the reasoning-replay
/// cache fallback; every production call site knows its session and calls
/// [`build_responses_request_for_session`] directly.
#[cfg(test)]
pub(crate) fn build_responses_request(
    request: &MessageRequest,
    instructions: &str,
    stream: bool,
) -> Value {
    build_responses_request_for_session(request, instructions, stream, "")
}

/// Like [`build_responses_request`], scoping the reasoning-replay cache
/// fallback (see [`reasoning_for_call`]) to `session_id` — used by the
/// production call sites, which know the client's session. Callers that
/// don't exercise the cache fallback (most existing tests) use the 3-arg
/// [`build_responses_request`], which passes an empty session id.
pub(crate) fn build_responses_request_for_session(
    request: &MessageRequest,
    instructions: &str,
    stream: bool,
    session_id: &str,
) -> Value {
    let mut input = Vec::new();
    for message in &request.messages {
        append_input_items(&mut input, message, session_id);
    }

    let (model_id, fast) = chatgpt_model_and_speed(&request.model);
    // Current GPT families routed here support adaptive default effort and may
    // use the low-latency fast (priority) service tier when the alias carries it.
    let is_current_gpt_family = openai_gpt_model_family(&model_id).is_some();
    // NOTE: The Codex Responses backend (chatgpt.com/backend-api/codex/responses)
    // rejects `max_output_tokens` with `400 {"detail":"Unsupported parameter:
    // max_output_tokens"}` — unlike the public `/v1/responses` API, it accepts no
    // request-side output-token cap at all (confirmed against Codex/Kilocode/
    // LiteLLM reports). Sending it (even from `request.max_tokens`) hard-fails the
    // turn, so it must be omitted. Output length is governed server-side; the
    // response's own `max_output_tokens` incomplete-reason still maps to
    // `StopReason::MaxTokens` in `parse_responses_response`.
    let mut payload = json!({
        "model": model_id,
        "instructions": instructions,
        "input": input,
        "store": false,
        "stream": stream,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": super::prompt_cache_key(&model_id, request, session_id),
    });
    if let Some(retention) =
        PromptCacheStrategy::OpenAiPromptCacheKey.prompt_cache_retention(&model_id)
    {
        payload["prompt_cache_retention"] = json!(retention);
    }

    if let Some(tools) = request.tools.as_ref().filter(|tools| !tools.is_empty()) {
        payload["tools"] = Value::Array(tools.iter().map(responses_tool).collect());
        // Honor an explicit `tool_choice` (e.g. a workflow sub-agent forcing
        // `StructuredOutput`) instead of always sending "auto", which silently
        // weakened the forced tool call (BUG-R16). Default stays "auto".
        payload["tool_choice"] = request
            .tool_choice
            .as_ref()
            .map_or_else(|| json!("auto"), responses_tool_choice);
    }

    // Reasoning effort and fast mode are independent controls. Priority:
    // (1) an explicit provider-neutral `request.effort` the caller set (mapped
    // to GPT's supported wire scale; internal Max/Ultra become xhigh); else
    // (2) a legacy thinking budget selects the tier; else (3) current GPT
    // families scale effort to the task and other families omit reasoning
    // (server-side default).
    let reasoning_request = request.reasoning_request();
    let explicit_top_effort = matches!(
        reasoning_request,
        ReasoningRequest::Effort(
            crate::types::EffortLevel::Xhigh
                | crate::types::EffortLevel::Max
                | crate::types::EffortLevel::Ultra
        )
    );
    let effort = match reasoning_request {
        // Use the user's requested reasoning tier as-is on GPT's scale. `/fast`
        // is a serving-priority signal (service_tier), not a reasoning-effort
        // ceiling, so explicit top-tier requests must remain top-tier.
        //
        // `effort_band_ceiling` marks a DYNAMIC band rather than a static pin
        // (Smart mode): `level` is the band floor (Xhigh), and the shared
        // resolver picks the concrete per-request rung — from this request's
        // own difficulty signals — BEFORE the per-model `gpt_for_model`
        // projection, so the rest of this function (and `explicit_top_effort`
        // above, keyed off the still-Xhigh `reasoning_request`) treats the
        // result exactly as if it had been the named effort all along.
        ReasoningRequest::Effort(level) => {
            let level = match request.effort_band_ceiling {
                Some(ceiling) => super::resolve_effort_band(
                    level,
                    ceiling,
                    &request.model,
                    super::band_difficulty_for_request(request),
                ),
                None => level,
            };
            Some(level.gpt_for_model(&request.model))
        }
        ReasoningRequest::BudgetTokens(budget) => {
            Some(super::effort_level_for_budget(budget).gpt_for_model(&request.model))
        }
        // No explicit budget: scale effort to the task instead of forcing the
        // top tier. A forced `xhigh` front-loads a long, near-silent maximum-
        // reasoning phase on every turn (the "GPT freezes while Claude is fine"
        // report) — `dynamic_effort` keeps trivial asks fast and reserves deep
        // reasoning for heavy work. Top tiers stay reachable via `/effort`.
        ReasoningRequest::Auto if is_current_gpt_family => Some(dynamic_effort(request)),
        ReasoningRequest::Auto => None,
    };
    // Empty-response de-escalation: when the runtime is retrying a turn whose
    // previous attempt produced no visible output (its retry/continuation
    // system reminder is present), an auto/budget-derived maximum-reasoning
    // request would walk the exact same path — the model spends the whole
    // response window reasoning and ends with zero text again, deterministically.
    // Step those requests down so the retry actually changes the outcome. An
    // explicit top-tier `/effort xhigh|max|ultra|smart` selection is a
    // user-selected top-effort contract and must remain top-tier, including on
    // GPT fast; lower explicit efforts keep the existing retry de-escalation
    // behavior.
    let effort = match effort {
        Some(effort) if !explicit_top_effort && empty_response_pressure(request) => {
            Some(de_escalated_effort(effort))
        }
        other => other,
    };
    // gpt-5.6-luna effort contract (user-set, 2026-07-13): luna runs at
    // `xhigh` or above. Applied LAST — after band resolution, dynamic effort,
    // and the empty-retry de-escalation — so nothing steps a luna request
    // back below the floor. `ultra` clamps to `max`: luna's endpoint tops out
    // at `max` (its models-cache entry lists no `ultra` level, unlike
    // sol/terra). The floor deliberately overrides the de-escalation above
    // for luna; the fast-tier model is cheap enough that a top-effort retry
    // beats a below-contract one.
    let effort = if crate::types::model_id_matches_family(&model_id, "gpt-5.6-luna") {
        effort.map(luna_effort_floor)
    } else {
        effort
    };
    if let Some(effort) = effort {
        payload["reasoning"] = json!({ "effort": effort, "summary": "auto" });
    }

    // "/fast on" (currently encoded by the explicit gpt-5.5-fast alias) is a
    // serving-priority signal, orthogonal to reasoning effort: it asks OpenAI's
    // infrastructure to prioritise the request (~1.5x faster serving, higher
    // credit rate) without changing how hard the model thinks. Do not infer
    // priority from arbitrary future `-fast` suffixes (for example Codex Spark)
    // until they are first-class aliases.
    if fast {
        payload["service_tier"] = json!("priority");
    }

    payload
}

/// Stable marker prefixes of the runtime's empty-response retry/continuation
/// system reminders (`crates/runtime/src/conversation/mod.rs`). The api crate
/// cannot depend on the runtime crate, so the literals are duplicated here and
/// pinned by tests on both sides.
const EMPTY_RETRY_REMINDER_MARKER: &str = "[zo:empty-response-retry]";
const EMPTY_CONTINUATION_REMINDER_MARKER: &str = "[zo:empty-response-continuation]";

/// Whether this request carries the runtime's empty-response retry or
/// continuation reminder — i.e. the previous attempt at this same context
/// ended with no visible assistant output.
fn empty_response_pressure(request: &MessageRequest) -> bool {
    request.system.as_ref().is_some_and(|blocks| {
        blocks.iter().any(|block| match block {
            SystemBlock::Text { text, .. } => {
                text.starts_with(EMPTY_RETRY_REMINDER_MARKER)
                    || text.starts_with(EMPTY_CONTINUATION_REMINDER_MARKER)
            }
        })
    })
}

/// One step down the Responses reasoning ladder for an empty-response retry:
/// `ultra` drops to `max`, `max` drops to `high`, `xhigh`/`high` drop to
/// `medium`, and `medium` drops to `low`. Never escalates.
///
/// The `"ultra"` arm closes a wildcard cliff: without it, `"ultra"` fell
/// through to the `_ => "low"` catch-all — a 5-rung drop reachable today via
/// `Auto` + heavy-intent on sol/terra under empty-response retry pressure,
/// and routine once the dynamic ultra band resolves to `"ultra"` on a heavy
/// turn (band picks are still `explicit_top_effort`-protected today, so this
/// arm is a defense-in-depth fix, not currently load-bearing for the band).
fn de_escalated_effort(effort: &'static str) -> &'static str {
    match effort {
        "ultra" => "max",
        "max" => "high",
        "xhigh" | "high" => "medium",
        _ => "low",
    }
}

/// GPT reasoning effort when the user set no explicit thinking budget.
/// Forcing `xhigh` made every default turn front-load a long, near-silent
/// maximum-reasoning phase (the "gpt freezes, Claude is fine" report). Instead
/// we scale effort to the task: trivial asks answer fast at `low`, heavy
/// analysis / debugging / refactoring / large contexts earn the model's
/// heavy-intent ceiling (see [`heavy_intent_effort`]), the rest land at
/// `medium`. `xhigh`/`max`/`ultra` remain reachable via an explicit `/effort`
/// or high enough thinking budget (handled by the caller before this fallback).
fn dynamic_effort(request: &MessageRequest) -> &'static str {
    // Shared with the dynamic ultra band (`super::band_difficulty_for_request`)
    // — this ladder's own keyword table/thresholds were hoisted there so both
    // paths key off exactly one source.
    let difficulty = super::band_difficulty_for_request(request);
    let total_chars = super::total_message_text_chars(request);
    let last_user_len = super::last_user_message_text(request).chars().count();

    if difficulty.heavy_intent || difficulty.large_context || difficulty.long_ask {
        heavy_intent_effort(&request.model)
    } else if total_chars < 2_000 && last_user_len < 200 {
        "low"
    } else {
        "medium"
    }
}

/// Heavy-intent auto-ladder top rung, ceiling-aware via
/// [`super::max_supported_effort`] (the same SSOT the router reads for Deep-tier
/// promotion). GPT-5.6 can select an internal `Max`/`Ultra` rung, which the
/// provider boundary safely projects to `xhigh`; GPT families whose internal
/// ceiling is only `Xhigh` keep the historical `high` auto cap.
fn heavy_intent_effort(model: &str) -> &'static str {
    match super::max_supported_effort(model) {
        EffortLevel::Ultra => EffortLevel::Ultra.gpt_for_model(model),
        EffortLevel::Max => EffortLevel::Max.gpt_for_model(model),
        EffortLevel::Xhigh | EffortLevel::High | EffortLevel::Medium | EffortLevel::Low => "high",
    }
}

/// The `gpt-5.6-luna` reasoning floor: any resolved effort below `xhigh`
/// rises to `xhigh`; `max` is kept (above the floor); `ultra` clamps to `max`
/// because luna's endpoint exposes no `ultra` level. See the call site in
/// [`build_responses_request_for_session`] for why this runs last.
fn luna_effort_floor(effort: &'static str) -> &'static str {
    match effort {
        "max" | "ultra" => "max",
        _ => "xhigh",
    }
}

/// Appended to the system instructions for every model served through this
/// backend (see [`ChatGptBackendClient::instructions`]).
const TOOL_BATCHING_CONTRACT: &str = "\n\n## Tool-call batching contract\n\
Every tool-using response MUST carry every independent tool call you can \
already justify — batch reads, searches, greps, and independent shell checks \
together in one response; three to eight calls at once is normal and \
preferred. A call is independent unless one of its arguments literally \
requires another call's output. Emitting independent calls one per turn is a \
defect, not caution: every extra turn re-sends the entire accumulated \
transcript as billed input. Before ending any tool-using response, ask which \
other calls you already know you need, and add them now.";

fn usage_cached_tokens(usage: Option<&Value>) -> u32 {
    usage
        .and_then(|value| value.pointer("/input_tokens_details/cached_tokens"))
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
        .unwrap_or(0)
}

/// Reasoning items to replay immediately before `call_id`'s `function_call`
/// input item. Priority: (1) the message's own attached `reasoning_replay`
/// field — the root fix, populated on every turn produced after this change
/// and persisted with the session so it survives process restarts and
/// sub-agent fanout; (2) the session-scoped replay cache (`reasoning_for_call`)
/// — a defense-in-depth fallback for history that predates the attached
/// field.
fn reasoning_replay_for_call(message: &InputMessage, call_id: &str, session_id: &str) -> Option<Vec<Value>> {
    reasoning_replay_from_attached(message.reasoning_replay.as_ref(), call_id)
        .or_else(|| reasoning_for_call(session_id, call_id))
}

/// Look up `call_id`'s reasoning items inside a message's attached
/// `reasoning_replay` value, shaped `[{"call_id": ..., "items": [...]}, ...]`.
fn reasoning_replay_from_attached(attached: Option<&Value>, call_id: &str) -> Option<Vec<Value>> {
    attached?
        .as_array()?
        .iter()
        .find(|entry| entry.get("call_id").and_then(Value::as_str) == Some(call_id))
        .and_then(|entry| entry.get("items"))
        .and_then(Value::as_array)
        .cloned()
}

/// Translate one zo message into zero or more Responses `input` items.
///
/// Assistant turns split into `message` items (text) and `function_call` items
/// (tool calls); user turns become `message` items (text/images) and
/// `function_call_output` items (tool results).
///
/// Reasoning replay is *append-only*: every tool call whose reasoning items
/// are still available (message-attached field first, session-scoped cache
/// fallback second) replays them, with no recency window or anchor. Any
/// deterministic bound expressed in this layer must eventually *drop* replay
/// items from an old message as the history grows, and that drop is a
/// mid-history mutation that invalidates the provider's prefix cache for the
/// whole suffix — the previous stride-16 staircase anchor re-billed the last
/// 1–2 strides of transcript on every jump (observed live 07-20: warm sol
/// requests re-sending 60–100k uncached at each anchor advance). Replay
/// volume is bounded upstream instead: the runtime's compaction band rewrites
/// the history — dropping summarized messages and their attached replay with
/// them — at exactly the moments the prefix cache is invalidated anyway.
fn append_input_items(input: &mut Vec<Value>, message: &InputMessage, session_id: &str) {
    if message.role == "assistant" {
        let mut text = String::new();
        for block in &message.content {
            match block {
                InputContentBlock::Text { text: value, .. } => text.push_str(value),
                InputContentBlock::ToolUse { .. } => {
                    let Some(ToolLedgerView::ToolUse {
                        id,
                        name,
                        input: args,
                    }) = ToolLedgerView::from_input_block(block)
                    else {
                        unreachable!("tool use block must project to tool use ledger view");
                    };
                    if !text.is_empty() {
                        input.push(assistant_message(&text));
                        text.clear();
                    }
                    // Reasoning replay (Codex CLI parity): the stateless Codex
                    // backend keeps multi-turn reasoning continuity only when
                    // the client re-sends the reasoning items (encrypted
                    // content included) that preceded this tool call. Without
                    // them gpt-5.5 re-reasons from scratch on every tool
                    // result — slower and weaker than codex desktop.
                    if let Some(items) = reasoning_replay_for_call(message, id, session_id) {
                        input.extend(items);
                    }
                    input.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": args.to_string(),
                    }));
                }
                // Anthropic reasoning blocks are provider-opaque; the Responses
                // backend has its own encrypted-reasoning replay above and never
                // lowers a stored Anthropic thinking block.
                InputContentBlock::ToolResult { .. }
                | InputContentBlock::Image { .. }
                | InputContentBlock::Document { .. }
                | InputContentBlock::Thinking { .. }
                | InputContentBlock::RedactedThinking { .. } => {}
            }
        }
        if !text.is_empty() {
            input.push(assistant_message(&text));
        }
        return;
    }

    let mut pending_message_blocks = Vec::new();
    for block in &message.content {
        match block {
            InputContentBlock::Text { .. } | InputContentBlock::Image { .. } => {
                pending_message_blocks.push(block);
            }
            InputContentBlock::ToolResult { .. } => {
                let Some(ToolLedgerView::ToolResult {
                    tool_use_id,
                    content,
                    ..
                }) = ToolLedgerView::from_input_block(block)
                else {
                    unreachable!("tool result block must project to tool result ledger view");
                };
                flush_responses_user_message(input, &mut pending_message_blocks);
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": super::flatten_tool_result_content(content),
                }));
            }
            InputContentBlock::ToolUse { .. }
            | InputContentBlock::Document { .. }
            | InputContentBlock::Thinking { .. }
            | InputContentBlock::RedactedThinking { .. } => {}
        }
    }
    flush_responses_user_message(input, &mut pending_message_blocks);
}

fn assistant_message(text: &str) -> Value {
    json!({
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": text }],
    })
}

fn flush_responses_user_message(input: &mut Vec<Value>, blocks: &mut Vec<&InputContentBlock>) {
    if blocks.is_empty() {
        return;
    }

    let content = blocks
        .iter()
        .filter_map(|block| match block {
            InputContentBlock::Text { text, .. } => Some(json!({
                "type": "input_text",
                "text": text,
            })),
            InputContentBlock::Image { source, .. } => Some(responses_image_content(source)),
            InputContentBlock::Document { .. }
            | InputContentBlock::ToolUse { .. }
            | InputContentBlock::ToolResult { .. }
            | InputContentBlock::Thinking { .. }
            | InputContentBlock::RedactedThinking { .. } => None,
        })
        .collect::<Vec<_>>();
    blocks.clear();

    if !content.is_empty() {
        input.push(json!({
            "type": "message",
            "role": "user",
            "content": content,
        }));
    }
}

fn responses_image_content(source: &ImageSource) -> Value {
    json!({
        "type": "input_image",
        "image_url": super::image_data_url(source),
        "detail": "auto",
    })
}

/// Flatten a tool-result content list into a single string, matching the
/// Chat Completions path's behaviour (the `function_call_output.output` field
/// is a plain string).
fn responses_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": false,
    })
}

/// Map zo's [`ToolChoice`] onto the **Responses API** shape. Unlike Chat
/// Completions (`{"type":"function","function":{"name":..}}`), the Responses API
/// forces a named function with the flat form `{"type":"function","name":..}` —
/// matching [`responses_tool`] above (verified against the OpenAI function-calling
/// reference). `Any` maps to "required" so a forced `StructuredOutput` is honored.
fn responses_tool_choice(tool_choice: &ToolChoice) -> Value {
    match tool_choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Any => json!("required"),
        ToolChoice::None => json!("none"),
        ToolChoice::Tool { name } => json!({ "type": "function", "name": name }),
    }
}

/// Map zo's extended-thinking budget onto a Responses `reasoning.effort`
/// bucket using the same shared fallback thresholds as the production request
/// builder, in the legacy GPT-5.5 projection used by this test helper. This
/// clamps every budget at or above the `Max` threshold (24k) down to `xhigh`
/// on legacy GPT-5.5, since this budget-only fallback has no `Ultra` bucket
/// because the OpenAI wire enum does not accept literal `max`/`ultra` values.
/// GPT-5.6 uses the same final wire ceiling even when Zo selects one of
/// those higher internal tiers.
#[cfg(test)]
fn reasoning_effort(budget_tokens: u32) -> &'static str {
    super::effort_level_for_budget(budget_tokens).gpt_for_model("gpt-5.5")
}

/// Resolve a zo model alias/id to the supported family id the ChatGPT
/// backend accepts, and detect the low-latency `fast` variant.
///
/// zo ships dated canonical ids (`gpt-5.5-2026-04-23`) and a first-class
/// `gpt-5.5-fast` alias, while the ChatGPT backend wants the short family id
/// (`gpt-5.5`). The `fast` flag is surfaced separately so the caller can
/// request priority serving. Two ways to spell "fast": the legacy bare
/// `gpt-5.5-fast` alias (predates the bracket convention), and the GPT-5.6
/// family's `[fast]` service-tier suffix (`gpt-5.6-terra[fast]`, `/fast`'s
/// toggled id for that family — see `live_cli_commands::toggle_fast`) which is
/// stripped here before the family lookup so it does not reach the wire as
/// part of the model id. Unknown ids pass through unchanged so custom/manual
/// models fail at the provider boundary instead of being rewritten to an
/// invented family; an arbitrary (non-`[fast]`) `-fast` suffix, e.g. an
/// unregistered Codex Spark variant, is deliberately NOT treated as a priority
/// alias.
fn chatgpt_model_and_speed(model: &str) -> (String, bool) {
    let lower = model.to_ascii_lowercase();
    let bracket_stripped = lower.strip_suffix("[fast]");
    let fast = lower == "gpt-5.5-fast" || bracket_stripped.is_some();
    let family_lookup = bracket_stripped.unwrap_or(&lower);
    let Some(family) = openai_gpt_model_family(family_lookup) else {
        return (model.to_string(), fast);
    };
    (family.to_string(), fast)
}

/// How many `(call_id, reasoning items)` pairs one session's replay cache
/// queue retains. Widened from the old process-global 64 now that the cache
/// is partitioned per session (see [`ReasoningReplayStore`]) instead of
/// shared FIFO-evicted across every concurrent sub-agent/session in the
/// process — a single session can afford a deeper queue since it no longer
/// competes with unrelated sessions for the same slots.
const REASONING_REPLAY_CAP: usize = 256;

/// How many distinct sessions [`ReasoningReplayStore`] retains before
/// evicting the oldest session's whole queue. Bounds total process memory
/// across a long-running process that opens many sessions (sub-agent
/// fanout, `--resume`, headless batch runs), the exact scenario that used to
/// evict the *main* conversation's cache entries out of the old
/// process-global FIFO.
const REASONING_REPLAY_MAX_SESSIONS: usize = 32;

/// FIFO entries of one session's reasoning replay cache: `(call_id, reasoning items)`.
type ReasoningReplayEntries = VecDeque<(String, Vec<Value>)>;

/// Session-scoped replay store for Responses reasoning items, keyed by
/// `session_id` then by the `call_id` of the `function_call` that followed
/// the items in the same response.
///
/// The Codex backend is stateless (`store: false`): reasoning continuity
/// across turns exists only if the client *replays* the reasoning items —
/// `encrypted_content` included — in the next request's input, which is what
/// Codex CLI/desktop does. This store is the defense-in-depth fallback for
/// history that predates [`InputMessage::reasoning_replay`] (the root fix —
/// see `append_input_items`); it used to be a single process-wide FIFO queue,
/// which meant a sub-agent fanout's calls could evict the *main*
/// conversation's still-needed entries. Partitioning by session id isolates
/// each session's queue so unrelated sessions can no longer starve each
/// other; `session_order` tracks session insertion order so once more than
/// [`REASONING_REPLAY_MAX_SESSIONS`] sessions have written to the store, the
/// oldest session's entire queue is evicted (simple whole-session FIFO, not
/// per-entry LRU).
#[derive(Default)]
struct ReasoningReplayStore {
    sessions: std::collections::HashMap<String, ReasoningReplayEntries>,
    session_order: VecDeque<String>,
}

impl ReasoningReplayStore {
    fn record(&mut self, session_id: &str, call_id: &str, items: Vec<Value>) {
        if !self.sessions.contains_key(session_id) {
            self.session_order.push_back(session_id.to_string());
            self.sessions.insert(session_id.to_string(), VecDeque::new());
            while self.session_order.len() > REASONING_REPLAY_MAX_SESSIONS {
                let Some(evicted) = self.session_order.pop_front() else {
                    break;
                };
                self.sessions.remove(&evicted);
            }
        }
        // The session just inserted above is always the newest entry in
        // `session_order`, so it cannot have been the one evicted by the
        // FIFO trim; the lookup is still `Option`-guarded rather than
        // indexed/panicking.
        let Some(queue) = self.sessions.get_mut(session_id) else {
            return;
        };
        queue.retain(|(existing, _)| existing != call_id);
        queue.push_back((call_id.to_string(), items));
        while queue.len() > REASONING_REPLAY_CAP {
            queue.pop_front();
        }
    }

    fn lookup(&self, session_id: &str, call_id: &str) -> Option<Vec<Value>> {
        self.sessions
            .get(session_id)?
            .iter()
            .find(|(existing, _)| existing == call_id)
            .map(|(_, items)| items.clone())
    }
}

fn reasoning_replay_store() -> &'static Mutex<ReasoningReplayStore> {
    static STORE: OnceLock<Mutex<ReasoningReplayStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(ReasoningReplayStore::default()))
}

/// Record the reasoning items that preceded `call_id` in a streamed response,
/// scoped to `session_id`. Re-recording the same `call_id` (stream restart)
/// replaces the old entry.
fn cache_reasoning_for_call(session_id: &str, call_id: &str, items: Vec<Value>) {
    if call_id.is_empty() || items.is_empty() {
        return;
    }
    reasoning_replay_store()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .record(session_id, call_id, items);
}

/// The reasoning items to replay immediately before `call_id`'s
/// `function_call` input item, if this process produced them for `session_id`.
/// Legacy fallback — see [`ReasoningReplayStore`] — consulted only when the
/// message itself carries no attached `reasoning_replay` payload.
fn reasoning_for_call(session_id: &str, call_id: &str) -> Option<Vec<Value>> {
    reasoning_replay_store()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .lookup(session_id, call_id)
}

#[cfg(test)]
fn remove_reasoning_for_call(session_id: &str, call_id: &str) {
    let mut store = reasoning_replay_store()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(queue) = store.sessions.get_mut(session_id) {
        queue.retain(|(existing, _)| existing != call_id);
    }
}

/// Incremental parser turning the ChatGPT backend's Responses SSE byte stream
/// into event JSON values. Frames are `data: {json}` blocks separated by blank
/// lines; each payload carries a `"type"` discriminator consumed by
/// [`ResponsesStreamState`].
#[derive(Debug, Default)]
pub(crate) struct ResponsesSseParser {
    buffer: Vec<u8>,
    /// How far into `buffer` the frame-separator search has already looked.
    /// Without this, every `push` rescans the whole accumulated buffer from
    /// byte 0 — O(n²) over a frame that arrives in many chunks. The ChatGPT
    /// backend streams a large `reasoning.encrypted_content` blob as a single
    /// SSE frame split across hundreds of TCP chunks, so the quadratic scan
    /// pegged a CPU core for seconds and froze the TUI event loop (the
    /// "gpt-5.5 stutters/freezes, Claude is fine" report). Resuming the scan
    /// from where the last one stopped makes total work linear in bytes seen.
    scanned: usize,
}

impl ResponsesSseParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk; return every complete event JSON value it now contains.
    ///
    /// Errors (following the Anthropic `SseParser` contract) when the retained
    /// partial-frame buffer would exceed the crate-wide SSE cap, so a stream
    /// that never emits a separator cannot grow memory without limit. The cap is
    /// high enough for the large `reasoning.encrypted_content` frame the Codex
    /// Responses backend streams as one event.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<Value>, ApiError> {
        crate::sse::guard_sse_buffer_push(self.buffer.len(), chunk.len())?;
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(frame) = self.next_frame() {
            if let Some(value) = parse_data_frame(&frame) {
                events.push(value);
            }
        }
        Ok(events)
    }

    /// Pop the next complete `\n\n`- or `\r\n\r\n`-terminated frame, resuming the
    /// separator search from `scanned` rather than the buffer start. A boundary
    /// can straddle the resume point by up to 3 bytes (`\r\n\r\n`), so the scan
    /// starts a few bytes before `scanned`; matched bytes are then drained and
    /// `scanned` rebased to 0.
    fn next_frame(&mut self) -> Option<String> {
        // Back up enough that a separator split across the previous resume
        // point is still found (max separator length is 4 → overlap of 3).
        let start = self.scanned.saturating_sub(3);
        let found = self.buffer[start..]
            .windows(2)
            .position(|window| window == b"\n\n")
            .map(|position| (start + position, 2))
            .or_else(|| {
                self.buffer[start..]
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|position| (start + position, 4))
            });
        let Some((end, sep)) = found else {
            // No frame yet: everything except the trailing 3-byte overlap window
            // has been searched, so the next push resumes from there.
            self.scanned = self.buffer.len().saturating_sub(3);
            return None;
        };
        let frame = String::from_utf8_lossy(&self.buffer[..end]).into_owned();
        self.buffer.drain(..end + sep);
        self.scanned = 0;
        Some(frame)
    }
}

fn parse_data_frame(frame: &str) -> Option<Value> {
    let mut payload = String::new();
    for line in frame.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if !payload.is_empty() {
                payload.push('\n');
            }
            payload.push_str(data.trim_start());
        }
    }
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    serde_json::from_str(&payload).ok()
}

/// Translates Responses events into zo's Anthropic-shaped [`StreamEvent`]s.
/// The Responses `output_index` maps directly onto zo's content-block index,
/// and the state tracks open blocks so `response.completed` can close any text
/// or reasoning item whose explicit `output_item.done` frame was skipped or
/// reordered.
#[derive(Debug)]
pub(crate) struct ResponsesStreamState {
    model: String,
    /// Session id the reasoning-replay cache fallback is scoped to (see
    /// [`ReasoningReplayStore`]). Empty by default; production callers set it
    /// via [`Self::with_session_id`] when they construct the state.
    session_id: String,
    message_started: bool,
    finished: bool,
    started_blocks: BTreeSet<u32>,
    open_blocks: BTreeSet<u32>,
    /// Completed `reasoning` items (full JSON, `encrypted_content` included)
    /// not yet attributed to a tool call. When a `function_call` item
    /// completes, these are cached under its `call_id` for next-turn replay.
    pending_reasoning: Vec<Value>,
    /// Assembled `[{"call_id":..,"items":[..]}]` reasoning-replay entries for
    /// this turn, built once from the authoritative `response.completed` /
    /// `response.incomplete` output array ([`Self::completed_output_deltas`])
    /// so it reflects every `function_call` in the turn regardless of which
    /// individual `output_item.done` streaming frames arrived. `None` when
    /// the turn made no tool calls.
    reasoning_replay: Option<Value>,
    text_delta_indices: BTreeSet<u32>,
    input_delta_indices: BTreeSet<u32>,
    /// Terminal failure reported by the backend (`response.failed` / `error`
    /// SSE events). Held here because `ingest` returns display events only;
    /// the stream loop drains it via [`Self::take_failure`] and surfaces it as
    /// an [`ApiError`] instead of silently ending an empty stream.
    failure: Option<ApiError>,
}

impl ResponsesStreamState {
    pub(crate) fn new(model: String) -> Self {
        Self {
            model,
            session_id: String::new(),
            message_started: false,
            finished: false,
            started_blocks: BTreeSet::new(),
            open_blocks: BTreeSet::new(),
            pending_reasoning: Vec::new(),
            reasoning_replay: None,
            text_delta_indices: BTreeSet::new(),
            input_delta_indices: BTreeSet::new(),
            failure: None,
        }
    }

    /// Scope the reasoning-replay cache fallback to `session_id`. Chainable;
    /// production call sites set it from the client's session immediately
    /// after construction (see [`ChatGptBackendClient::stream_message`]).
    #[must_use]
    pub(crate) fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// Drain a terminal backend failure recorded by `ingest`
    /// (`response.failed` / top-level `error` events).
    pub(crate) fn take_failure(&mut self) -> Option<ApiError> {
        self.failure.take()
    }

    pub(crate) fn ingest(&mut self, event: &Value) -> Vec<StreamEvent> {
        if self.finished {
            return Vec::new();
        }
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.created" => self.start_message(event),
            "response.output_item.added" => self.start_item(event),
            "response.output_text.delta" => self.text_delta(event, "delta", false),
            "response.output_text.done" => self.text_delta(event, "text", true),
            "response.content_part.done" => self.content_part_done(event),
            "response.function_call_arguments.delta" => self.function_args_delta(event, false),
            "response.function_call_arguments.done" => self.function_args_delta(event, true),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => self
                .block_delta(
                    event,
                    Some(OutputContentBlock::Thinking {
                        thinking: String::new(),
                        signature: None,
                    }),
                    |delta| ContentBlockDelta::ThinkingDelta { thinking: delta },
                ),
            "response.reasoning_summary_part.added" => self.reasoning_part_boundary(event),
            "response.output_item.done" => self.block_stop(event),
            "response.completed" => self.finish(event, "end_turn"),
            // An incomplete response is still a terminal close: the model ran
            // out of output budget (typically all spent on reasoning) or was
            // filtered. Close the message with an honest stop_reason instead of
            // ignoring the frame — ignoring it ended the stream with zero
            // events, which the runtime mistook for "the model returned no
            // assistant content" and retried the identical request forever.
            "response.incomplete" => {
                let reason = incomplete_stop_reason(event);
                self.finish(event, reason)
            }
            "response.failed" => {
                self.record_failure(
                    event.pointer("/response/error/code"),
                    event.pointer("/response/error/message"),
                );
                Vec::new()
            }
            "error" => {
                self.record_failure(event.get("code"), event.get("message"));
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Record a terminal backend failure (`response.failed` / `error`).
    /// Server-side faults and throttles are retryable (the pre-commit restart
    /// path re-issues the request); invalid-request class codes are not.
    fn record_failure(&mut self, code: Option<&Value>, message: Option<&Value>) {
        if self.finished || self.failure.is_some() {
            return;
        }
        let code = code.and_then(Value::as_str).unwrap_or_default().to_string();
        let message = message
            .and_then(Value::as_str)
            .unwrap_or("backend reported a terminal stream failure")
            .to_string();
        let retryable = matches!(
            code.as_str(),
            "server_error" | "rate_limit_exceeded" | "overloaded" | "slow_down" | ""
        );
        self.failure = Some(ApiError::StreamApi {
            error_type: (!code.is_empty()).then_some(code),
            message: Some(message.clone()),
            body: message,
            retryable,
        });
    }

    fn start_message(&mut self, event: &Value) -> Vec<StreamEvent> {
        if self.message_started {
            return Vec::new();
        }
        self.message_started = true;
        let id = event
            .pointer("/response/id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        vec![StreamEvent::MessageStart(MessageStartEvent {
            message: MessageResponse {
                id,
                kind: "message".to_string(),
                role: "assistant".to_string(),
                content: Vec::new(),
                model: self.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: zero_usage(),
                request_id: None,
                thought_signature: None,
                reasoning_replay: None,
                context_management: None,
            },
        })]
    }

    fn start_item(&mut self, event: &Value) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        let item = &event["item"];
        let content_block = match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message" => OutputContentBlock::Text {
                text: String::new(),
            },
            "function_call" => OutputContentBlock::ToolUse {
                id: str_field(item, "call_id"),
                name: str_field(item, "name"),
                input: json!({}),
            },
            "reasoning" => OutputContentBlock::Thinking {
                thinking: String::new(),
                signature: None,
            },
            _ => return Vec::new(),
        };
        if !self.started_blocks.insert(index) {
            return Vec::new();
        }
        self.open_blocks.insert(index);
        vec![StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            index,
            content_block,
        })]
    }

    fn text_delta(&mut self, event: &Value, field: &str, final_payload: bool) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        if final_payload && self.text_delta_indices.contains(&index) {
            return Vec::new();
        }
        let Some(text) = event.get(field).and_then(Value::as_str) else {
            return Vec::new();
        };
        self.emit_text_delta(index, text)
    }

    fn content_part_done(&mut self, event: &Value) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        if self.text_delta_indices.contains(&index) {
            return Vec::new();
        }
        let part = &event["part"];
        if part.get("type").and_then(Value::as_str) != Some("text") {
            return Vec::new();
        }
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            return Vec::new();
        };
        self.emit_text_delta(index, text)
    }

    fn emit_text_delta(&mut self, index: u32, text: &str) -> Vec<StreamEvent> {
        if text.is_empty() {
            return Vec::new();
        }
        self.text_delta_indices.insert(index);
        self.emit_delta(
            index,
            Some(OutputContentBlock::Text {
                text: String::new(),
            }),
            ContentBlockDelta::TextDelta {
                text: text.to_string(),
            },
        )
    }

    fn function_args_delta(&mut self, event: &Value, final_payload: bool) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        if final_payload && self.input_delta_indices.contains(&index) {
            return Vec::new();
        }
        let field = if final_payload { "arguments" } else { "delta" };
        let Some(delta) = event.get(field).and_then(Value::as_str) else {
            return Vec::new();
        };
        let fallback_start = final_payload.then(|| OutputContentBlock::ToolUse {
            id: str_field(event, "call_id"),
            name: str_field(event, "name"),
            input: json!({}),
        });
        self.emit_input_delta(index, delta, fallback_start)
    }

    fn emit_input_delta(
        &mut self,
        index: u32,
        delta: &str,
        fallback_start: Option<OutputContentBlock>,
    ) -> Vec<StreamEvent> {
        if delta.is_empty() {
            return Vec::new();
        }
        self.input_delta_indices.insert(index);
        self.emit_delta(
            index,
            fallback_start,
            ContentBlockDelta::InputJsonDelta {
                partial_json: delta.to_string(),
            },
        )
    }

    /// A new reasoning summary part opened. OpenAI streams summary parts as
    /// separate texts on ONE output item with no separator between them, so
    /// concatenation produced a run-on paragraph — and the TUI's thinking
    /// title (first line of the current paragraph) froze on part 0's topic
    /// for the whole reasoning phase. Emit a paragraph break between parts so
    /// they render, and re-title, as paragraphs. `summary_index == 0` (or a
    /// part on a not-yet-started item) needs no separator.
    fn reasoning_part_boundary(&mut self, event: &Value) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        let summary_index = event
            .get("summary_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if summary_index == 0 || !self.started_blocks.contains(&index) {
            return Vec::new();
        }
        self.emit_delta(
            index,
            None,
            ContentBlockDelta::ThinkingDelta {
                thinking: "\n\n".to_string(),
            },
        )
    }

    fn block_delta(
        &mut self,
        event: &Value,
        fallback_start: Option<OutputContentBlock>,
        make: impl FnOnce(String) -> ContentBlockDelta,
    ) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        let Some(delta) = event.get("delta").and_then(Value::as_str) else {
            return Vec::new();
        };
        self.emit_delta(index, fallback_start, make(delta.to_string()))
    }

    fn emit_delta(
        &mut self,
        index: u32,
        fallback_start: Option<OutputContentBlock>,
        delta: ContentBlockDelta,
    ) -> Vec<StreamEvent> {
        self.open_blocks.insert(index);
        let mut events = Vec::new();
        if let Some(content_block) = fallback_start {
            if self.started_blocks.insert(index) {
                events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index,
                    content_block,
                }));
            }
        }
        events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
            index,
            delta,
        }));
        events
    }

    fn block_stop(&mut self, event: &Value) -> Vec<StreamEvent> {
        let Some(index) = output_index(event) else {
            return Vec::new();
        };
        // Reasoning replay capture: `output_item.done` carries the complete
        // item JSON. Hold finished `reasoning` items until the `function_call`
        // they precede completes, then cache them under its `call_id` so the
        // next turn's request can replay them (Codex CLI parity — see
        // `ReasoningReplayStore`). This feeds only the session-scoped cache
        // fallback; the authoritative per-turn `reasoning_replay` payload sent
        // to the runtime is assembled once in `completed_output_deltas`.
        let item = &event["item"];
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "reasoning" => self.pending_reasoning.push(item.clone()),
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                cache_reasoning_for_call(
                    &self.session_id,
                    call_id,
                    std::mem::take(&mut self.pending_reasoning),
                );
            }
            _ => {}
        }
        let mut events = match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message" => self.message_done_delta(index, item),
            "function_call" => self.function_call_done_delta(index, item),
            _ => Vec::new(),
        };
        self.open_blocks.remove(&index);
        self.started_blocks.remove(&index);
        events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
            index,
        }));
        events
    }

    fn message_done_delta(&mut self, index: u32, item: &Value) -> Vec<StreamEvent> {
        if self.text_delta_indices.contains(&index) {
            return Vec::new();
        }
        let Some(parts) = item.get("content").and_then(Value::as_array) else {
            return Vec::new();
        };
        let text = parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<String>();
        self.emit_text_delta(index, &text)
    }

    fn function_call_done_delta(&mut self, index: u32, item: &Value) -> Vec<StreamEvent> {
        if self.input_delta_indices.contains(&index) {
            return Vec::new();
        }
        let Some(arguments) = item.get("arguments").and_then(Value::as_str) else {
            return Vec::new();
        };
        let fallback_start = Some(OutputContentBlock::ToolUse {
            id: str_field(item, "call_id"),
            name: str_field(item, "name"),
            input: json!({}),
        });
        self.emit_input_delta(index, arguments, fallback_start)
    }

    fn finish(&mut self, event: &Value, stop_reason: &str) -> Vec<StreamEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let usage = event.pointer("/response/usage");
        let cache_read_input_tokens = usage_cached_tokens(usage);
        let token_usage = Usage {
            input_tokens: usage_field(usage, "input_tokens")
                .saturating_sub(cache_read_input_tokens),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens,
            output_tokens: usage_field(usage, "output_tokens"),
        };
        let mut events: Vec<StreamEvent> =
            self.completed_output_deltas(event).into_iter().collect();
        events.extend(
            self.open_blocks
                .iter()
                .copied()
                .map(|index| StreamEvent::ContentBlockStop(ContentBlockStopEvent { index })),
        );
        self.open_blocks.clear();
        self.started_blocks.clear();
        self.text_delta_indices.clear();
        self.input_delta_indices.clear();
        events.extend([
            StreamEvent::MessageDelta(MessageDeltaEvent {
                delta: MessageDelta {
                    stop_reason: Some(stop_reason.to_string()),
                    stop_sequence: None,
                    thought_signature: None,
                    reasoning_replay: self.reasoning_replay.take(),
                },
                usage: token_usage,
                context_management: None,
            }),
            StreamEvent::MessageStop(MessageStopEvent {}),
        ]);
        events
    }

    fn completed_output_deltas(&mut self, event: &Value) -> Vec<StreamEvent> {
        let Some(output) = event.pointer("/response/output").and_then(Value::as_array) else {
            return Vec::new();
        };
        let mut events = Vec::new();
        for (index, item) in output.iter().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "message" => events.extend(self.message_done_delta(index, item)),
                "function_call" => events.extend(self.function_call_done_delta(index, item)),
                _ => {}
            }
        }
        // Rebuilt from the authoritative snapshot rather than the incremental
        // `self.pending_reasoning` bucket `block_stop` maintains while
        // streaming: `response.completed`/`response.incomplete` always carries
        // the full `output` array, so re-walking it here also covers a
        // `function_call` whose individual `output_item.done` frame was
        // skipped or reordered (the same guarantee `message_done_delta` /
        // `function_call_done_delta` above already rely on).
        self.reasoning_replay = reasoning_replay_from_output(output, &self.session_id);
        events
    }
}

/// Map a `response.incomplete` close onto zo's Anthropic-shaped stop
/// reasons: an output-token cutoff is `max_tokens`; anything else (e.g. a
/// content filter) still ends the turn cleanly.
fn incomplete_stop_reason(event: &Value) -> &'static str {
    match event
        .pointer("/response/incomplete_details/reason")
        .and_then(Value::as_str)
    {
        Some("max_output_tokens") => "max_tokens",
        _ => "end_turn",
    }
}

const fn zero_usage() -> Usage {
    Usage {
        input_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 0,
    }
}

fn output_index(event: &Value) -> Option<u32> {
    event
        .get("output_index")
        .and_then(Value::as_u64)
        .and_then(|index| u32::try_from(index).ok())
}

fn str_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn usage_field(usage: Option<&Value>, key: &str) -> u32 {
    usage
        .and_then(|value| value.get(key))
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
        .unwrap_or(0)
}

/// Default cap on transparent mid-stream restarts (mirrors the Anthropic
/// connection-retry budget so both backends behave the same under transient
/// faults).
const DEFAULT_STREAM_MAX_RETRIES: u32 = 5;
const DEFAULT_STREAM_INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_millis(500);
const DEFAULT_STREAM_MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);

/// Total wall-clock ceiling over a single pre-commit restart sequence. The
/// per-attempt cap (`max_retries`) does not bound elapsed time, so a silent
/// backend that idle-times-out (`CHATGPT_STREAM_IDLE_TIMEOUT_MS`) and re-opens
/// each time can hold the turn for minutes before exhausting attempts (the
/// observed ~275 s freeze). Once the sequence has been retrying longer than
/// this, the next fault propagates as a retryable error instead of restarting
/// again, so the turn fails fast and the UI is freed. Sized above one idle
/// timeout plus a couple of brisk re-opens, below the multi-minute storm.
const MAX_RESTART_WALLCLOCK: std::time::Duration = std::time::Duration::from_secs(120);

/// Client for the ChatGPT subscription backend (Responses API). Built from a
/// stored ChatGPT OAuth token; the access token is sent as a bearer credential
/// and the `account_id` (from the `id_token` JWT) as the `chatgpt-account-id`
/// header.
#[derive(Debug, Clone)]
pub struct ChatGptBackendClient {
    http: reqwest::Client,
    access_token: String,
    account_id: Option<String>,
    session_id: String,
    /// Stable cache scope for `prompt_cache_key` derivation and the
    /// reasoning-replay fallback cache — the zo session id when the host
    /// provides one. `session_id` above is deliberately NOT used for caching:
    /// it is a random per-client-instance value (Codex-parity wire header),
    /// and clients are rebuilt on every provider-route model swap (each
    /// smart-policy delegation episode) and every OAuth near-expiry rotation,
    /// so keying the provider cache on it rolled the key mid-session and
    /// pinned sol cache reads at the shared ~12k system prefix (observed live
    /// 07-20, hit-rate median 9%). `None` falls back to `session_id` (bare
    /// constructions in tests keep their legacy behavior).
    cache_scope: Option<String>,
    base_url: String,
    max_retries: u32,
    initial_backoff: std::time::Duration,
    max_backoff: std::time::Duration,
}

impl ChatGptBackendClient {
    #[must_use]
    pub fn new(access_token: impl Into<String>, account_id: Option<String>) -> Self {
        Self {
            http: shared_http_client(),
            access_token: access_token.into(),
            account_id,
            session_id: random_session_id(),
            cache_scope: None,
            base_url: CHATGPT_RESPONSES_URL.to_string(),
            max_retries: DEFAULT_STREAM_MAX_RETRIES,
            initial_backoff: DEFAULT_STREAM_INITIAL_BACKOFF,
            max_backoff: DEFAULT_STREAM_MAX_BACKOFF,
        }
    }

    /// Pin the prompt-cache scope (see [`Self::cache_scope`] field docs) to a
    /// host-stable id — the zo session id — so the provider cache key survives
    /// client rebuilds (model swaps, OAuth rotations, 401 recovery).
    #[must_use]
    pub fn with_cache_scope(mut self, scope: impl Into<String>) -> Self {
        let scope = scope.into();
        self.cache_scope = (!scope.is_empty()).then_some(scope);
        self
    }

    /// The active cache scope: the pinned host scope, else the per-instance
    /// wire session id.
    fn cache_scope(&self) -> &str {
        self.cache_scope.as_deref().unwrap_or(&self.session_id)
    }

    /// The pinned host cache scope, if any — used to carry the scope across
    /// client rebuilds (401 recovery reconstructs the client from scratch).
    #[must_use]
    pub fn pinned_cache_scope(&self) -> Option<&str> {
        self.cache_scope.as_deref()
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Tune the transparent mid-stream restart budget. `max_retries` bounds how
    /// many times a stalled/dropped stream is re-issued before any non-replay-safe
    /// output is surfaced; backoff between attempts grows exponentially from
    /// `initial_backoff`, capped at `max_backoff`, with per-thread jitter.
    #[must_use]
    pub fn with_retry_policy(
        mut self,
        max_retries: u32,
        initial_backoff: std::time::Duration,
        max_backoff: std::time::Duration,
    ) -> Self {
        self.max_retries = max_retries;
        self.initial_backoff = initial_backoff;
        self.max_backoff = max_backoff;
        self
    }

    /// Exponential backoff for restart `attempt` (1-based), capped at
    /// `max_backoff`. Mirrors the Anthropic client's schedule; the caller adds
    /// jitter via `retry_backoff::spread_backoff`.
    fn backoff_for_attempt(&self, attempt: u32) -> Result<std::time::Duration, ApiError> {
        super::backoff_for_attempt(attempt, self.initial_backoff, self.max_backoff)
    }

    /// Open a fresh streaming Responses connection for `request` (POST + header
    /// application + success check). Shared by the initial `stream_message` and
    /// every transparent mid-stream restart so they issue an identical request.
    async fn open_stream_response(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let open = self.open_stream_response_unbounded(request);
        let Some((budget, kind)) = stream_open_timeout() else {
            return open.await;
        };
        match tokio::time::timeout(budget, open).await {
            Ok(result) => result,
            Err(_) => Err(match kind {
                StreamOpenTimeoutKind::Idle => ApiError::stream_idle_timeout(budget),
                StreamOpenTimeoutKind::Startup => {
                    ApiError::stream_startup_no_progress(budget, false)
                }
            }),
        }
    }

    /// Raw POST + response-header/body validation. The public stream opener
    /// wraps this whole future in the same idle/startup budget used after SSE
    /// headers arrive; otherwise a backend that accepts the socket but never
    /// answers the HTTP request bypasses every body-level watchdog.
    async fn open_stream_response_unbounded(
        &self,
        request: &MessageRequest,
    ) -> Result<reqwest::Response, ApiError> {
        let instructions = Self::instructions(request);
        let body =
            build_responses_request_for_session(request, &instructions, true, self.cache_scope());
        debug_dump_chatgpt("request", &body.to_string());
        let response = self
            .apply_headers(self.http.post(&self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(ApiError::from)?;
        expect_success(response).await
    }

    /// zo's system blocks become the Responses `instructions` field — the
    /// Responses API takes the system role at the top level, not as an `input`
    /// item.
    fn instructions(request: &MessageRequest) -> String {
        let system = request.system.as_ref().map_or_else(String::new, |blocks| {
            blocks
                .iter()
                .map(|block| match block {
                    SystemBlock::Text { text, .. } => text.as_str(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        });
        if system.is_empty() {
            return system;
        }
        // zo's base prompt is authored for Claude Code and hardcodes a Claude
        // identity ("You are Claude Code…", "Model family: Claude Opus 4.8"). For an
        // OpenAI-served model that text makes it introduce itself as Claude, so
        // prepend an explicit identity override. The tooling / workflow guidance
        // below still applies; only the identity is corrected. Routed through the
        // shared `apply_non_anthropic_identity` so every non-Anthropic backend
        // corrects the identity identically (Gemini / OpenAI-compatible too).
        let (model, _) = chatgpt_model_and_speed(&request.model);
        let identity = super::apply_non_anthropic_identity(
            &system,
            &model,
            super::maker_for_provider(super::ProviderKind::OpenAi),
        );
        // The base prompt's parallel-tools prose alone leaves GPT models
        // averaging ~2.6 tool calls per tool-using message (live sessions
        // 07-19/20) versus ~3.5 for Claude on the same harness — each skipped
        // batch is a whole extra round trip that re-bills the transcript, so
        // the backend restates the contract in the imperative register GPT
        // models follow best, as a trailing section (both prompt ends carry
        // instruction weight; the head must stay identity-first).
        format!("{identity}{TOOL_BATCHING_CONTRACT}")
    }

    fn apply_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("OpenAI-Beta", OPENAI_BETA_RESPONSES)
            .header("originator", ORIGINATOR)
            // Model-gating fingerprint (see [`USER_AGENT`]): without it the
            // backend 404s `gpt-5.6-luna` as "Model not found".
            .header("user-agent", USER_AGENT)
            .header("session_id", &self.session_id)
            .bearer_auth(&self.access_token);
        if let Some(account_id) = &self.account_id {
            builder = builder.header("chatgpt-account-id", account_id);
        }
        builder
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<ChatGptStream, ApiError> {
        let startup_window = startup_no_progress_timeout();
        let startup_started_at = std::time::Instant::now();
        let response = self.open_stream_response(request).await?;
        Ok(ChatGptStream {
            response,
            parser: ResponsesSseParser::new(),
            state: ResponsesStreamState::new(request.model.clone())
                .with_session_id(self.cache_scope().to_string()),
            pending: VecDeque::new(),
            done: false,
            // Retry context: the stream re-issues this exact request through
            // `client` while it is still re-armable (no text/tool args surfaced).
            client: self.clone(),
            request: request.clone(),
            restart_attempts: 0,
            restart_window_start: None,
            committed: false,
            retry_notice: None,
            startup_window,
            startup_deadline: startup_window
                .and_then(|window| startup_started_at.checked_add(window)),
            startup_reasoning_extended: false,
        })
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        let instructions = Self::instructions(request);
        let body =
            build_responses_request_for_session(request, &instructions, false, self.cache_scope());
        let response = self
            .apply_headers(self.http.post(&self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(ApiError::from)?;
        let response = expect_success(response).await?;
        let value = response.json::<Value>().await.map_err(ApiError::from)?;
        Ok(parse_responses_response(
            &value,
            &request.model,
            self.cache_scope(),
        ))
    }
}

/// Streamed Responses turn: pulls chunks, feeds the SSE parser, and yields
/// zo [`StreamEvent`]s as content blocks complete.
///
/// While no non-replay-safe output has been surfaced yet (`committed == false`),
/// a stalled or dropped stream is transparently re-issued via `client` — the
/// common "switched to gpt-5.5, it reasons silently, connection idles out before
/// the first answer/tool token" case recovers without bubbling an error.
/// Provider bookkeeping frames and reasoning summaries are safe to replay; text
/// deltas and tool-call argument deltas are not. The Codex backend forces
/// `store: false`, so there is no server-side cursor to resume from — a restart
/// re-runs the whole turn, which is only safe before the commit point.
pub struct ChatGptStream {
    response: reqwest::Response,
    parser: ResponsesSseParser,
    state: ResponsesStreamState,
    pending: VecDeque<StreamEvent>,
    done: bool,
    /// Client + request used to re-open the stream on a pre-commit fault.
    client: ChatGptBackendClient,
    request: MessageRequest,
    /// Transparent restarts spent so far, bounded by `client.max_retries`.
    restart_attempts: u32,
    /// Wall clock at the first restart of the current pre-commit sequence, so the
    /// whole restart storm is bounded by elapsed time and not only by attempt
    /// count — a silent backend that idle-times-out and re-opens repeatedly would
    /// otherwise hold the turn for minutes (see [`MAX_RESTART_WALLCLOCK`]).
    restart_window_start: Option<std::time::Instant>,
    /// Set once text or tool-call argument bytes are surfaced; locks out further
    /// restarts to avoid duplicate user-visible output or malformed tools.
    committed: bool,
    /// Optional sink invoked just before each transparent restart sleeps, so a
    /// live UI can surface the otherwise-silent multi-second reconnect pause as
    /// "reconnecting" instead of a freeze. `None` (the default) preserves the
    /// old log-only behaviour for non-interactive callers.
    retry_notice: Option<StreamRetryCallback>,
    /// First-action watchdog for keep-alive-only streams. Transport bytes do not
    /// count as progress; one decoded reasoning delta grants one extra window.
    startup_window: Option<std::time::Duration>,
    startup_deadline: Option<std::time::Instant>,
    startup_reasoning_extended: bool,
}

/// Sink for [`ChatGptStream`] mid-stream restart notices. Boxed `Fn` so the
/// runtime can route it to a render channel without `api` depending on any
/// render type.
type StreamRetryCallback = std::sync::Arc<dyn Fn(StreamRetryNotice) + Send + Sync>;

impl std::fmt::Debug for ChatGptStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatGptStream")
            .field("response", &self.response)
            .field("parser", &self.parser)
            .field("state", &self.state)
            .field("pending", &self.pending)
            .field("done", &self.done)
            .field("client", &self.client)
            .field("request", &self.request)
            .field("restart_attempts", &self.restart_attempts)
            .field("restart_window_start", &self.restart_window_start)
            .field("committed", &self.committed)
            .field("has_retry_notice", &self.retry_notice.is_some())
            .field("startup_window", &self.startup_window)
            .field("startup_deadline", &self.startup_deadline)
            .field(
                "startup_reasoning_extended",
                &self.startup_reasoning_extended,
            )
            .finish()
    }
}

const CHATGPT_STREAM_COOPERATIVE_YIELD_EVERY: usize = 16;

/// How long the stream may stay quiet (keep-alive chunks arriving, no decoded
/// event) before the one-shot quiet-reasoning heartbeat notice fires. Kept just
/// under the CLI's 20-second "no output" stall-badge threshold
/// (`STALL_THRESHOLD_SECS`) so that, whenever keep-alives are still arriving,
/// the badge latches to the calm "reasoning · stream alive" *before* it would
/// ever read "no output" — the whole point of the heartbeat. At the old 60s it
/// lost that race by 40 seconds, so a healthy gpt-5.x reasoning pass showed the
/// alarming "no output" for most of a minute (live report: users could not tell
/// a reasoning turn from a hang). Still above any normal first-token gap — a
/// real delta resets `quiet_since`, so fast turns never trip it — and firing the
/// notice only pre-arms the badge; nothing is shown until the 20s threshold, so
/// an earlier fire adds no visual noise. A dead/congested connection sends no
/// chunks at all, so this branch never runs and the idle/startup timeouts own
/// that case instead.
const CHATGPT_QUIET_REASONING_NOTICE_AFTER: std::time::Duration =
    std::time::Duration::from_secs(15);

/// Default idle budget: abort a chunk read that has received no bytes for this
/// long. The ChatGPT/Codex Responses backend keeps the HTTP/2 connection alive
/// while it reasons silently, so `chunk().await` can otherwise block forever.
/// Sized well above any normal inter-chunk gap so legitimate slow reasoning is
/// not cut short.
const CHATGPT_STREAM_IDLE_TIMEOUT_MS: u64 = 90_000;

/// Override for [`CHATGPT_STREAM_IDLE_TIMEOUT_MS`]. A value of `0` disables the
/// idle timeout entirely (restores the unbounded-wait behaviour).
const CHATGPT_STREAM_IDLE_TIMEOUT_ENV: &str = "ZO_CHATGPT_STREAM_IDLE_TIMEOUT_MS";

/// First decoded task action deadline. Unlike the 90-second byte-idle guard,
/// this clock is not reset by transport keep-alives. A reasoning delta grants
/// one equal extension, matching the effort-aware workflow startup watchdog.
const CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_MS: u64 = 240_000;
const CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_ENV: &str =
    "ZO_CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_MS";

/// Resolve the per-chunk idle budget, honouring the env override. `None` means
/// "no timeout" (override set to `0`).
fn stream_idle_timeout() -> Option<std::time::Duration> {
    let millis = std::env::var(CHATGPT_STREAM_IDLE_TIMEOUT_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(CHATGPT_STREAM_IDLE_TIMEOUT_MS);
    (millis > 0).then(|| std::time::Duration::from_millis(millis))
}

fn startup_no_progress_timeout() -> Option<std::time::Duration> {
    let millis = std::env::var(CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_MS);
    (millis > 0).then(|| std::time::Duration::from_millis(millis))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamOpenTimeoutKind {
    Idle,
    Startup,
}

/// Bound the HTTP response-open stage as well as the SSE body. The byte-idle
/// window normally wins (90s vs. the 4m first-action window); disabling it
/// still leaves the startup deadline as a safe upper bound.
fn stream_open_timeout() -> Option<(std::time::Duration, StreamOpenTimeoutKind)> {
    match (stream_idle_timeout(), startup_no_progress_timeout()) {
        (Some(idle), Some(startup)) if idle <= startup => {
            Some((idle, StreamOpenTimeoutKind::Idle))
        }
        (Some(_) | None, Some(startup)) => {
            Some((startup, StreamOpenTimeoutKind::Startup))
        }
        (Some(idle), None) => Some((idle, StreamOpenTimeoutKind::Idle)),
        (None, None) => None,
    }
}

fn extend_startup_deadline_for_reasoning(
    deadline: &mut Option<std::time::Instant>,
    window: Option<std::time::Duration>,
    already_extended: &mut bool,
) {
    if *already_extended {
        return;
    }
    if let (Some(current), Some(extension)) = (*deadline, window) {
        *deadline = current.checked_add(extension);
        *already_extended = true;
    }
}

use super::{crosses_restart_commit_boundary, should_restart_within_budget};

impl ChatGptStream {
    /// Install a sink that fires just before each transparent restart sleeps, so
    /// a live consumer can render the reconnect pause. Chainable; no-op sink by
    /// default.
    #[must_use]
    pub fn with_retry_notice_callback(
        mut self,
        callback: impl Fn(StreamRetryNotice) + Send + Sync + 'static,
    ) -> Self {
        self.retry_notice = Some(std::sync::Arc::new(callback));
        self
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        let idle_timeout = stream_idle_timeout();
        let mut quiet_chunks = 0usize;
        // Quiet-reasoning heartbeat: the connection is delivering keep-alive
        // chunks (so the idle timeout never fires) but the model has emitted
        // no event yet — deep reasoning on a large context can stay silent
        // for minutes, and without a signal the live UI reads "no output Nm"
        // as a hang. Fired once per quiet stretch, through the same sink the
        // reconnect notices ride.
        let mut quiet_since: Option<std::time::Instant> = None;
        let mut quiet_notified = false;
        loop {
            if let Some(event) = self.pending.pop_front() {
                self.observe_startup_event(&event);
                // Reasoning may extend the deadline once, but it must not keep
                // an event-producing stream alive forever. Check after observing
                // the event so the first reasoning delta receives its extension,
                // and before surfacing it so an already-expired stream remains
                // replay-safe when we restart it.
                if self.restart_if_startup_deadline_elapsed().await? {
                    quiet_chunks = 0;
                    quiet_since = None;
                    quiet_notified = false;
                    continue;
                }
                return Ok(Some(event));
            }
            if self.done {
                return Ok(None);
            }
            if self.restart_if_startup_deadline_elapsed().await? {
                quiet_chunks = 0;
                quiet_since = None;
                quiet_notified = false;
                continue;
            }
            // Per-chunk idle timeout: each received chunk resets the budget, so
            // long-but-active streams are never cut, while a truly silent
            // backend (Codex holds the HTTP/2 connection open while it reasons)
            // surfaces a retryable error instead of hanging the turn forever.
            // The startup deadline participates in this wait too: without its
            // own wake-up, a final keepalive just before the deadline could leave
            // `chunk().await` asleep until the longer transport-idle timer.
            let read_budget = self.next_chunk_wait_budget(idle_timeout);
            let read = match read_budget {
                Some(wait) => match tokio::time::timeout(wait, self.response.chunk()).await {
                    Ok(chunk) => chunk.map_err(ApiError::from),
                    Err(_elapsed) => Err(self.startup_no_progress_error().unwrap_or_else(|| {
                        ApiError::stream_idle_timeout(idle_timeout.unwrap_or(wait))
                    })),
                },
                None => self.response.chunk().await.map_err(ApiError::from),
            };
            let chunk = match read {
                Ok(chunk) => chunk,
                Err(error) => {
                    self.recover_or_restart_precommit(error).await?;
                    quiet_chunks = 0;
                    continue;
                }
            };
            match chunk {
                Some(chunk) => {
                    let mut emitted = false;
                    for value in self.parser.push(&chunk)? {
                        let events = self.state.ingest(&value);
                        emitted |= !events.is_empty();
                        self.pending.extend(events);
                    }
                    // A terminal `response.failed` / `error` frame surfaces as a
                    // real error (restartable pre-commit) — never as a silent
                    // zero-event end-of-stream the runtime would misread as an
                    // empty assistant turn.
                    if let Some(failure) = self.state.take_failure() {
                        self.recover_or_restart_precommit(failure).await?;
                        quiet_chunks = 0;
                        continue;
                    }
                    if emitted {
                        quiet_chunks = 0;
                        quiet_since = None;
                        quiet_notified = false;
                    } else {
                        quiet_chunks = quiet_chunks.saturating_add(1);
                        if quiet_chunks >= CHATGPT_STREAM_COOPERATIVE_YIELD_EVERY {
                            quiet_chunks = 0;
                            tokio::task::yield_now().await;
                        }
                        let since = *quiet_since.get_or_insert_with(std::time::Instant::now);
                        if let Some(error) = self.startup_no_progress_error() {
                            if self.can_restart(&error) {
                                self.restart(error).await?;
                                quiet_chunks = 0;
                                quiet_since = None;
                                quiet_notified = false;
                                continue;
                            }
                            return Err(self.wrap_restart_exhaustion(error));
                        }
                        if !quiet_notified
                            && since.elapsed() >= CHATGPT_QUIET_REASONING_NOTICE_AFTER
                        {
                            quiet_notified = true;
                            if let Some(notice) = &self.retry_notice {
                                notice(StreamRetryNotice {
                                    kind: core_types::StreamNoticeKind::QuietReasoning,
                                    label: core_types::QUIET_REASONING_LABEL,
                                    attempt: 0,
                                    max_attempts: 0,
                                    delay: since.elapsed(),
                                });
                            }
                        }
                    }
                }
                None => self.done = true,
            }
        }
    }

    /// Recover a replay-safe stream failure or transparently reopen the stream.
    /// Returning `Ok` means the caller should resume its event loop; an unsafe
    /// or exhausted failure is returned unchanged.
    async fn recover_or_restart_precommit(&mut self, error: ApiError) -> Result<(), ApiError> {
        if self.recover_precommit_failure(&error).await? {
            return Ok(());
        }
        if self.can_restart(&error) {
            self.restart(error).await?;
            return Ok(());
        }
        Err(self.wrap_restart_exhaustion(error))
    }

    fn observe_startup_event(&mut self, event: &StreamEvent) {
        if crosses_restart_commit_boundary(event) {
            self.committed = true;
            self.startup_deadline = None;
            return;
        }
        let reasoning = matches!(
            event,
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::ThinkingDelta { .. },
                ..
            })
        );
        if reasoning && !self.startup_reasoning_extended {
            extend_startup_deadline_for_reasoning(
                &mut self.startup_deadline,
                self.startup_window,
                &mut self.startup_reasoning_extended,
            );
        }
    }

    fn startup_no_progress_error(&self) -> Option<ApiError> {
        let deadline = self.startup_deadline?;
        if std::time::Instant::now() < deadline {
            return None;
        }
        let window = self.startup_window?;
        let budget = if self.startup_reasoning_extended {
            window.checked_add(window).unwrap_or(std::time::Duration::MAX)
        } else {
            window
        };
        Some(ApiError::stream_startup_no_progress(
            budget,
            self.startup_reasoning_extended,
        ))
    }

    fn next_chunk_wait_budget(
        &self,
        idle_timeout: Option<std::time::Duration>,
    ) -> Option<std::time::Duration> {
        let startup_remaining = self
            .startup_deadline
            .map(|deadline| deadline.saturating_duration_since(std::time::Instant::now()));
        match (idle_timeout, startup_remaining) {
            (Some(idle), Some(startup)) => Some(idle.min(startup)),
            (Some(idle), None) => Some(idle),
            (None, Some(startup)) => Some(startup),
            (None, None) => None,
        }
    }

    async fn restart_if_startup_deadline_elapsed(&mut self) -> Result<bool, ApiError> {
        let Some(error) = self.startup_no_progress_error() else {
            return Ok(false);
        };
        if self.can_restart(&error) {
            self.restart(error).await?;
            return Ok(true);
        }
        Err(self.wrap_restart_exhaustion(error))
    }

    fn reset_startup_watchdog(&mut self) {
        self.startup_deadline = self
            .startup_window
            .and_then(|window| std::time::Instant::now().checked_add(window));
        self.startup_reasoning_extended = false;
    }

    /// Last-resort recovery for a pre-commit terminal Responses stream failure.
    ///
    /// Once visible text or tool-call bytes have crossed the commit boundary we
    /// must not replay the request here: duplicate output or partial tool JSON is
    /// worse than a surfaced error. Before that boundary, however, the failed
    /// stream has produced no user-visible assistant content, so a single
    /// non-streaming retry with a lower reasoning tier can turn the common
    /// `xhigh` terminal stream failure into a normal completed response.
    async fn recover_precommit_failure(&mut self, error: &ApiError) -> Result<bool, ApiError> {
        if self.committed || !is_terminal_stream_failure(error) {
            return Ok(false);
        }
        let request = deescalated_recovery_request(&self.request);
        match self.client.send_message(&request).await {
            Ok(response) => {
                self.pending.extend(complete_response_events(response));
                self.done = true;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    fn can_restart(&self, error: &ApiError) -> bool {
        should_restart_within_budget(
            self.committed,
            error.is_retryable(),
            self.restart_attempts,
            self.client.max_retries,
            self.restart_window_start.map(|start| start.elapsed()),
            MAX_RESTART_WALLCLOCK,
        )
    }

    fn wrap_restart_exhaustion(&self, error: ApiError) -> ApiError {
        let attempts_spent = self.restart_attempts >= self.client.max_retries;
        let wallclock_spent = self
            .restart_window_start
            .is_some_and(|start| start.elapsed() >= MAX_RESTART_WALLCLOCK);
        if !self.committed && error.is_retryable() && (attempts_spent || wallclock_spent) {
            ApiError::RetriesExhausted {
                attempts: self.restart_attempts.saturating_add(1),
                last_error: Box::new(error),
            }
        } else {
            error
        }
    }

    /// Re-open the stream after a pre-commit fault: back off (jittered), then
    /// replace the live response and parser/state so the loop resumes from a
    /// clean turn. Any partial bytes buffered in the old parser are discarded
    /// with it — safe because nothing has been surfaced.
    async fn restart(&mut self, last_error: ApiError) -> Result<(), ApiError> {
        // Stamp the start of the restart sequence on the first restart so the
        // wall-clock budget in `can_restart` measures the whole storm.
        self.restart_window_start.get_or_insert_with(std::time::Instant::now);
        self.restart_attempts += 1;
        let base = self.client.backoff_for_attempt(self.restart_attempts)?;
        let delay = super::retry_backoff::spread_backoff(base);
        eprintln!(
            "[zo] gpt stream stalled ({last_error}); restarting in {:.1}s (attempt {}/{})",
            delay.as_secs_f64(),
            self.restart_attempts,
            self.client.max_retries,
        );
        // Surface the otherwise-silent reconnect pause to a live UI so it reads
        // as "reconnecting", not a freeze. The classifier label is shared with
        // the establish-time retry notice so the wording stays in lockstep.
        if let Some(notice) = &self.retry_notice {
            notice(StreamRetryNotice {
                kind: core_types::StreamNoticeKind::Reconnect,
                label: core_types::retry_signal::retry_notice_label(&last_error.to_string()),
                attempt: self.restart_attempts,
                max_attempts: self.client.max_retries,
                delay,
            });
        }
        tokio::time::sleep(delay).await;
        self.response = self.client.open_stream_response(&self.request).await?;
        self.parser = ResponsesSseParser::new();
        self.state = ResponsesStreamState::new(self.request.model.clone())
            .with_session_id(self.client.cache_scope().to_string());
        self.pending.clear();
        self.done = false;
        self.reset_startup_watchdog();
        Ok(())
    }
}

fn is_terminal_stream_failure(error: &ApiError) -> bool {
    let ApiError::StreamApi { error_type, message, body, .. } = error else {
        return false;
    };
    let parts = [error_type.as_deref(), message.as_deref(), Some(body.as_str())];
    parts.iter().flatten().any(|part| {
        let lower = part.to_ascii_lowercase();
        lower.contains("terminal stream failure")
            || lower.contains("response.failed")
            || lower.contains("stream failed")
    })
}

fn deescalated_recovery_request(request: &MessageRequest) -> MessageRequest {
    let mut next = request.clone();
    next.stream = false;
    next.effort = request.effort.map(|effort| match effort {
        EffortLevel::Xhigh | EffortLevel::Max | EffortLevel::Ultra => EffortLevel::High,
        EffortLevel::Low | EffortLevel::Medium | EffortLevel::High => effort,
    });
    // A banded request's floor (Xhigh) just got forced down to a static High
    // above — clear the ceiling too, or the leftover band would let the wire
    // seam's `resolve_effort_band` re-escalate this deliberately-deescalated
    // recovery attempt right back past High on a heavy-intent/large-context
    // signal, undoing the point of this function.
    next.effort_band_ceiling = None;
    next.thinking = None;
    next.output_config = None;
    next
}

fn complete_response_events(response: MessageResponse) -> VecDeque<StreamEvent> {
    let usage = response.usage;
    let stop_reason = response.stop_reason.clone();
    let stop_sequence = response.stop_sequence.clone();
    VecDeque::from(vec![
        StreamEvent::MessageStart(MessageStartEvent { message: response }),
        StreamEvent::MessageDelta(MessageDeltaEvent {
            delta: MessageDelta {
                stop_reason,
                stop_sequence,
                thought_signature: None,
                reasoning_replay: None,
            },
            usage,
            context_management: None,
        }),
        StreamEvent::MessageStop(MessageStopEvent {}),
    ])
}

/// Assemble the aggregate reasoning-replay JSON value from a Responses
/// `output` array — `[{"call_id":..,"items":[..]}]`, one entry per
/// `function_call` whose preceding `reasoning` items (in output order) are
/// attributed to it — and record each entry in the session-scoped replay
/// cache fallback (see [`ReasoningReplayStore`]). Shared by the streaming
/// completed-response path ([`ResponsesStreamState::completed_output_deltas`])
/// and the true non-streaming [`parse_responses_response`].
fn reasoning_replay_from_output(output: &[Value], session_id: &str) -> Option<Value> {
    let mut pending = Vec::new();
    let mut entries = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "reasoning" => pending.push(item.clone()),
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let items = std::mem::take(&mut pending);
                if !call_id.is_empty() && !items.is_empty() {
                    entries.push(json!({
                        "call_id": call_id,
                        "items": items.clone(),
                    }));
                }
                cache_reasoning_for_call(session_id, call_id, items);
            }
            _ => {}
        }
    }
    (!entries.is_empty()).then_some(Value::Array(entries))
}

/// Parse a non-streaming Responses payload into a zo [`MessageResponse`].
fn parse_responses_response(value: &Value, model: &str, session_id: &str) -> MessageResponse {
    let mut content = Vec::new();
    if let Some(output) = value.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "message" => {
                    if let Some(parts) = item.get("content").and_then(Value::as_array) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                content.push(OutputContentBlock::Text {
                                    text: text.to_string(),
                                });
                            }
                        }
                    }
                }
                "function_call" => content.push(OutputContentBlock::ToolUse {
                    id: str_field(item, "call_id"),
                    name: str_field(item, "name"),
                    input: item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|raw| serde_json::from_str(raw).ok())
                        .unwrap_or_else(|| json!({})),
                }),
                _ => {}
            }
        }
    }
    let usage = value.get("usage");
    // Honest stop reason for an incomplete non-streaming response (the model
    // spent its output budget, typically on reasoning): mirrors the streaming
    // path's `response.incomplete` mapping.
    let stop_reason = if value.get("status").and_then(Value::as_str) == Some("incomplete")
        && value
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            == Some("max_output_tokens")
    {
        "max_tokens"
    } else {
        "end_turn"
    };
    MessageResponse {
        id: str_field(value, "id"),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason: Some(stop_reason.to_string()),
        stop_sequence: None,
        usage: {
            let cache_read_input_tokens = usage_cached_tokens(usage);
            Usage {
                input_tokens: usage_field(usage, "input_tokens")
                    .saturating_sub(cache_read_input_tokens),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens,
                output_tokens: usage_field(usage, "output_tokens"),
            }
        },
        request_id: None,
        thought_signature: None,
        reasoning_replay: value
            .get("output")
            .and_then(Value::as_array)
            .and_then(|output| reasoning_replay_from_output(output, session_id)),
        context_management: None,
    }
}

/// Diagnostic dump for the undocumented ChatGPT backend. When
/// `ZO_CHATGPT_DEBUG` is set, writes the request body / error response to
/// `/tmp/zo-chatgpt-<tag>.txt` so the exact wire shape can be compared
/// against codex. No-op (and silent on failure) otherwise — never affects the
/// request itself.
fn debug_dump_chatgpt(tag: &str, content: &str) {
    if std::env::var_os("ZO_CHATGPT_DEBUG").is_some() {
        let _ = std::fs::write(format!("/tmp/zo-chatgpt-{tag}.txt"), content);
    }
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, ApiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(std::time::Duration::from_secs);
    let body = response.text().await.unwrap_or_default();
    debug_dump_chatgpt("error", &format!("status {status}\n{body}"));
    // Parse the OpenAI `{"error":{...}}` envelope so the transcript shows the
    // human-readable message, not the raw JSON body (parity with the API-key
    // `openai_compat` path).
    let parsed = serde_json::from_str::<super::openai_compat::ErrorEnvelope>(&body).ok();
    let error_type = parsed
        .as_ref()
        .and_then(|e| e.error.error_type.clone().or_else(|| e.error.code.clone()));
    let message = parsed.as_ref().and_then(|e| e.error.message.clone());
    Err(ApiError::Api {
        status,
        error_type,
        message,
        body,
        // Match the SDK `shouldRetry`: 408/409/429 plus every server error
        // (>= 500), so 529 overload is retried transparently, not surfaced.
        retryable: matches!(status.as_u16(), 408 | 409 | 429) || status.as_u16() >= 500,
        retry_after,
    })
}

/// Best-effort random session id for the `session_id` header. Falls back to a
/// fixed value if `/dev/urandom` is unavailable.
fn random_session_id() -> String {
    use std::fmt::Write as _;
    use std::io::Read as _;
    let mut bytes = [0u8; 16];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .is_ok()
    {
        bytes.iter().fold(String::new(), |mut acc, byte| {
            let _ = write!(acc, "{byte:02x}");
            acc
        })
    } else {
        "zo-chatgpt-session".to_string()
    }
}

#[cfg(test)]
mod tests;
