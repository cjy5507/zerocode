//! Free-function helpers used inside the conversation loop.
//!
//! These were extracted from `mod.rs` so the file housing
//! `ConversationRuntime` stays focused on the turn state machine. None of
//! the helpers here own state; they are pure (or near-pure) data
//! transformations and async edges.
//!
//! Grouped by concern:
//!
//! - Telemetry & token counting: [`trace_attrs`],
//!   [`estimate_system_prompt_tokens`].
//! - Assistant-event reduction: [`build_assistant_message`],
//!   [`flush_text_block`].
//! - `AskUserQuestion` async edge: [`ask_user_question_async`],
//!   [`resolve_question_choice`].
//! - Hook feedback formatting: [`format_hook_message`],
//!   [`merge_hook_feedback`].

use serde_json::{Map, Value, json};
use tokio::sync::mpsc;

use crate::hooks::HookRunResult;
use crate::message_stream::types::{BlockIdGen, RenderBlock, UserQuestionPrompt};
use crate::session::{ContentBlock, ConversationMessage};
use crate::usage::TokenUsage;

use super::api::{AssistantEvent, PromptCacheEvent};

pub(super) fn trace_attrs(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// Cheap chars/4 token estimate for the system prompt sections, used by
/// `build_request`'s overflow guard so the message-budget calculation
/// accounts for the system text alongside the session messages.
#[must_use]
pub(super) fn estimate_system_prompt_tokens(sections: &[String]) -> u64 {
    sections
        .iter()
        .map(|s| (s.chars().count() / 4 + 1) as u64)
        .sum()
}

/// Outcome of reducing one iteration's [`AssistantEvent`] stream.
///
/// A stream that finishes (sees `MessageStop`) but carries no text or
/// `tool_use` blocks is **not** an error — it happens for a thinking-only
/// response or a transient empty completion. It is surfaced as
/// [`AssistantTurn::Empty`] (still carrying usage/cache telemetry) so the
/// conversation loop can retry or end the turn gracefully instead of
/// discarding a whole turn's worth of work. If text or tool calls already
/// arrived, the caller preserves that work even when the provider omitted
/// the final stop marker; if no content arrived, the stream is treated as an
/// empty turn so the bounded retry path owns the recovery instead of surfacing
/// a transport-shaped runtime error to the user.
#[derive(Debug)]
pub(super) enum AssistantTurn {
    /// A normal assistant message with at least one content block.
    Content {
        message: ConversationMessage,
        usage: Option<TokenUsage>,
        prompt_cache_events: Vec<PromptCacheEvent>,
        /// Provider stop reason (e.g. `"end_turn"`, `"tool_use"`,
        /// `"max_tokens"`/`"length"`). `None` when the provider sent none.
        /// The loop uses it to continue a turn truncated at the output limit
        /// rather than treating the cut-off response as a finished turn.
        stop_reason: Option<String>,
    },
    /// Finished cleanly but produced no renderable content. Carries only the
    /// usage so the caller can keep token telemetry accurate before it retries
    /// or ends the turn; any prompt-cache events for an empty turn are dropped
    /// (there is no message to attribute them to). `stop_reason` lets the
    /// bounded empty-retry tell a thinking-only completion apart from one cut
    /// off at the output-token limit, so the retry reminder can match the cause.
    Empty {
        usage: Option<TokenUsage>,
        stop_reason: Option<String>,
    },
}

impl AssistantTurn {
    /// Provider stop reason for this turn, regardless of whether any content
    /// arrived. Borrows both variants so the loop can peek the reason (e.g. to
    /// detect a `refusal`) before deciding how to consume the turn.
    pub(super) fn stop_reason(&self) -> Option<&str> {
        match self {
            AssistantTurn::Content { stop_reason, .. }
            | AssistantTurn::Empty { stop_reason, .. } => stop_reason.as_deref(),
        }
    }

    /// Token usage captured for this turn, if the provider reported any. Copied
    /// out so the caller can record telemetry for a turn it is about to discard
    /// (e.g. a billed mid-stream refusal partial) before the turn is consumed.
    pub(super) fn usage(&self) -> Option<TokenUsage> {
        match self {
            AssistantTurn::Content { usage, .. } | AssistantTurn::Empty { usage, .. } => *usage,
        }
    }
}

/// Provider streams that end successfully without content are equivalent to a
/// clean empty assistant turn; mark them as stopped so the conversation loop's
/// bounded empty-retry path can handle them.
#[must_use]
pub(super) fn normalize_empty_assistant_stream(
    mut events: Vec<AssistantEvent>,
) -> Vec<AssistantEvent> {
    let has_stop = events
        .iter()
        .any(|event| matches!(event, AssistantEvent::MessageStop));
    let has_content = events.iter().any(|event| match event {
        AssistantEvent::TextDelta(delta) => !delta.is_empty(),
        AssistantEvent::ToolUse { .. } => true,
        // A thinking-only stream is not user-visible content; treat it like an
        // empty turn so the bounded empty-retry path engages (matching the prior
        // behaviour where thinking was dropped entirely).
        AssistantEvent::Thinking { .. }
        | AssistantEvent::RedactedThinking { .. }
        | AssistantEvent::Usage(_)
        | AssistantEvent::PromptCache(_)
        | AssistantEvent::StopReason(_)
        | AssistantEvent::ThoughtSignature(_)
        | AssistantEvent::ProviderState(_)
        | AssistantEvent::ReasoningReplay(_)
        | AssistantEvent::Model(_)
        | AssistantEvent::MessageStop => false,
    });

    if !has_stop && !has_content {
        events.push(AssistantEvent::MessageStop);
    }
    events
}

/// Reduce the stream of [`AssistantEvent`]s emitted during a turn into an
/// [`AssistantTurn`] plus the token-usage and prompt-cache side-effects.
/// An empty stream yields [`AssistantTurn::Empty`] whether or not the provider
/// sent a final stop marker. The conversation loop decides whether to retry or
/// end gracefully; this reducer only preserves content and telemetry.
///
/// After the final text flush, tool-call markup the model leaked as plain
/// text is salvaged into synthetic `tool_use` blocks (see
/// [`super::tool_call_salvage`]) so the dispatch path still executes the
/// intended call instead of silently ending the turn.
pub(super) fn build_assistant_message(events: Vec<AssistantEvent>) -> AssistantTurn {
    let mut text = String::new();
    let mut blocks = Vec::new();
    let mut prompt_cache_events = Vec::new();
    let mut usage = None;
    let mut stop_reason = None;
    let mut thought_signature = None;
    let mut reasoning_replay = None;
    let mut model = None;

    for event in events {
        match event {
            AssistantEvent::TextDelta(delta) => text.push_str(&delta),
            AssistantEvent::ToolUse { id, name, input } => {
                flush_text_block_before_tool_use(&mut text, &mut blocks);
                blocks.push(ContentBlock::ToolUse { id, name, input });
            }
            AssistantEvent::Usage(value) => usage = Some(value),
            AssistantEvent::PromptCache(event) => prompt_cache_events.push(event),
            AssistantEvent::StopReason(reason) => stop_reason = Some(reason),
            AssistantEvent::ThoughtSignature(signature) => thought_signature = Some(signature),
            AssistantEvent::Thinking { thinking, signature } => {
                // Thinking leads an Anthropic assistant turn (before text /
                // tool_use). Flush any pending text first so the stored block
                // order matches arrival order — the exact order the API
                // validates when the block is replayed. `signature` is `None`
                // only for unsigned/legacy data; stored empty, it is dropped at
                // lowering rather than sent unsigned.
                flush_text_block(&mut text, &mut blocks);
                blocks.push(ContentBlock::Thinking {
                    thinking,
                    signature: signature.unwrap_or_default(),
                });
            }
            AssistantEvent::RedactedThinking { data } => {
                flush_text_block(&mut text, &mut blocks);
                blocks.push(ContentBlock::RedactedThinking { data });
            }
            AssistantEvent::ProviderState(state) => {
                if let Some(signature) = state.as_gemini_thought_signature() {
                    thought_signature = Some(signature.to_string());
                }
            }
            AssistantEvent::ReasoningReplay(value) => reasoning_replay = Some(value),
            AssistantEvent::Model(value) => model = Some(value),
            AssistantEvent::MessageStop => {}
        }
    }

    flush_text_block(&mut text, &mut blocks);
    super::tool_call_salvage::salvage_leaked_tool_calls(&mut blocks);

    if blocks.is_empty() {
        return AssistantTurn::Empty { usage, stop_reason };
    }

    AssistantTurn::Content {
        message: ConversationMessage::assistant_with_usage(blocks, usage)
            .with_thought_signature(thought_signature)
            .with_reasoning_replay(reasoning_replay)
            .with_model(model),
        usage,
        prompt_cache_events,
        stop_reason,
    }
}

pub(super) fn flush_text_block(text: &mut String, blocks: &mut Vec<ContentBlock>) {
    if !text.is_empty() {
        blocks.push(ContentBlock::Text {
            text: std::mem::take(text),
        });
    }
}

fn flush_text_block_before_tool_use(text: &mut String, blocks: &mut Vec<ContentBlock>) {
    let _ = core_types::text::strip_trailing_stray_tool_call_marker(text);
    if text.is_empty() {
        return;
    }
    flush_text_block(text, blocks);
}

/// Parsed `AskUserQuestion` tool input.
#[derive(Debug, serde::Deserialize)]
pub(super) struct AskInput {
    question: String,
    #[serde(default)]
    header: Option<String>,
    /// Accepts both bare strings and `{label, description}` objects —
    /// see [`crate::message_stream::QuestionOption`]'s `Deserialize` — and
    /// tolerates the whole array arriving double-encoded as a JSON string
    /// (some models stringify nested arrays; a hard parse failure here
    /// killed the question with no way for the model to correct it).
    #[serde(default, deserialize_with = "lenient_question_options")]
    options: Option<Vec<crate::message_stream::QuestionOption>>,
    /// When `true` the user may check several options and the answer comes back
    /// as an array. Accepts the canonical `camelCase` `multiSelect` and a
    /// `snake_case` alias; missing defaults to single-select.
    #[serde(default, rename = "multiSelect", alias = "multi_select")]
    multi_select: bool,
}

fn lenient_question_options<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<crate::message_stream::QuestionOption>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    use serde::de::Error as _;
    let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    let value = match value {
        serde_json::Value::String(raw) => serde_json::from_str(raw.trim()).map_err(|e| {
            D::Error::custom(format!("`options` was a string but not valid JSON: {e}"))
        })?,
        other => other,
    };
    serde_json::from_value(value).map(Some).map_err(D::Error::custom)
}

pub(super) fn parse_ask_input(input_json: &str) -> Result<AskInput, String> {
    serde_json::from_str(input_json).map_err(|e| format!("Failed to parse AskUserQuestion input: {e}"))
}

/// Handle `AskUserQuestion` as a fully async operation.
///
/// The normal `unblock_tool_execute` path uses `block_in_place`, which
/// blocks the current tokio task. When the conversation loop runs inside
/// a `tokio::select!` alongside the TUI render-tick, blocking the task
/// prevents the TUI from draining the `UserQuestionPrompt` — deadlock.
///
/// This function sends the prompt through `render_tx` and `.await`s the
/// oneshot response, yielding the task so the select loop can process
/// render ticks, key events, and ultimately relay the user's answer.
pub(super) async fn ask_user_question_async(
    input_json: &str,
    render_tx: &mpsc::Sender<RenderBlock>,
    id_gen: &BlockIdGen,
) -> Result<String, String> {
    let parsed = parse_ask_input(input_json)?;
    let options = parsed.options.unwrap_or_default();
    let labels: Vec<String> = options.iter().map(|opt| opt.label.clone()).collect();
    // Multi-select only applies with a fixed choice list; a free-form prompt is
    // always a single answer, matching how the modal degrades the flag.
    let multi_select = parsed.multi_select && !options.is_empty();

    let (responder, response) = tokio::sync::oneshot::channel();
    let prompt = UserQuestionPrompt {
        id: id_gen.next(),
        question: parsed.question.clone(),
        header: parsed.header,
        options,
        multi_select,
        responder,
    };

    render_tx
        .send(RenderBlock::UserQuestionPrompt(prompt))
        .await
        .map_err(|_| "TUI question channel closed".to_string())?;

    let raw_answers = response
        .await
        .map_err(|_| "User question dismissed without answer".to_string())?;

    // Map any numeric picks to their labels (a no-op for the TUI, which already
    // returns labels). Single-select preserves the historical string `answer`;
    // multi-select returns the full array of selected values.
    let resolved: Vec<String> = raw_answers
        .iter()
        .map(|answer| resolve_question_choice(answer, Some(&labels)))
        .collect();
    let answer = if multi_select {
        json!(resolved)
    } else {
        json!(resolved.into_iter().next().unwrap_or_default())
    };

    serde_json::to_string_pretty(&json!({
        "question": parsed.question,
        "answer": answer,
        "status": "answered"
    }))
    .map_err(|e| format!("Failed to format response: {e}"))
}

pub(super) fn resolve_question_choice(response: &str, options: Option<&[String]>) -> String {
    let trimmed = response.trim();
    if let Some(opts) = options {
        if let Ok(idx) = trimmed.parse::<usize>() {
            if idx >= 1 && idx <= opts.len() {
                return opts[idx - 1].clone();
            }
        }
    }
    trimmed.to_string()
}

pub(super) fn format_hook_message(result: &HookRunResult, fallback: &str) -> String {
    if result.messages().is_empty() {
        fallback.to_string()
    } else {
        result.messages().join("\n")
    }
}

pub(super) fn merge_hook_feedback(messages: &[String], output: String, is_error: bool) -> String {
    if messages.is_empty() {
        return output;
    }

    let mut sections = Vec::new();
    if !output.trim().is_empty() {
        sections.push(output);
    }
    let label = if is_error {
        "Hook feedback (error)"
    } else {
        "Hook feedback"
    };
    sections.push(format!("{label}:\n{}", messages.join("\n")));
    sections.join("\n\n")
}

/// Parse a tool's raw output string into a JSON value, falling back to a
/// string value when it isn't JSON (e.g. a plain-text error).
#[cfg(test)]
fn parse_tool_output(raw: &str) -> Value {
    if !might_be_json_value(raw) {
        return Value::String(raw.to_string());
    }
    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
fn might_be_json_value(raw: &str) -> bool {
    matches!(
        raw.trim_start().as_bytes().first().copied(),
        Some(b'{' | b'[' | b'"' | b'-' | b'0'..=b'9' | b't' | b'f' | b'n')
    )
}

/// Choose which output to *render* for a tool result and parse it to JSON.
///
/// A successful result renders the pure tool output so a hook's appended
/// feedback text doesn't get concatenated onto the tool's JSON — which would
/// make `serde_json` fall back to a string and the structured formatters
/// (edit diff, read, …) render a bogus `? · +0 -0`. Error/blocked results
/// render the hook-merged text so the reason stays visible (they fall to a
/// plain `Text` body with no structure to corrupt). The model context always
/// receives the merged output regardless of this choice.
#[cfg(test)]
fn render_value_for_tool_result(pure_output: &str, merged_output: &str, is_error: bool) -> Value {
    let src = if is_error { merged_output } else { pure_output };
    parse_tool_output(src)
}

#[cfg(test)]
mod tests {
    use super::{build_assistant_message, might_be_json_value, render_value_for_tool_result};
    use crate::conversation::api::ProviderStateBlob;
    use crate::conversation::{AssistantEvent, AssistantTurn};
    use serde_json::Value;

    const EDIT_JSON: &str = r#"{"filePath":"/tmp/a.rs","structuredPatch":[]}"#;

    /// The provider-reported model id (from `message_start` / the non-stream
    /// response) is stamped onto the stored assistant message — the per-model
    /// cost-attribution field a smart-routed session ledger needs.
    #[test]
    fn provider_model_event_is_stamped_onto_the_message() {
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::Model("gpt-5.6-sol".to_string()),
            AssistantEvent::TextDelta("ok".to_string()),
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert_eq!(message.model.as_deref(), Some("gpt-5.6-sol"));
    }

    #[test]
    fn legacy_thought_signature_sets_existing_field() {
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::ThoughtSignature("SIG_LEGACY".to_string()),
            AssistantEvent::TextDelta("ok".to_string()),
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert_eq!(message.thought_signature.as_deref(), Some("SIG_LEGACY"));
    }

    /// Streamed thinking events are stored as ordered blocks that LEAD the
    /// assistant turn (before the flushed text and the `tool_use`) — the order the
    /// Anthropic API validates when the blocks are replayed. A `Some(signature)`
    /// event stores the signature; text deltas are still coalesced.
    #[test]
    fn build_assistant_message_stores_thinking_before_text_and_tool_use() {
        use crate::session::ContentBlock;
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::Thinking {
                thinking: "reasoning".to_string(),
                signature: Some("SIG".to_string()),
            },
            AssistantEvent::RedactedThinking {
                data: "BLOB".to_string(),
            },
            AssistantEvent::TextDelta("the ".to_string()),
            AssistantEvent::TextDelta("answer".to_string()),
            AssistantEvent::ToolUse {
                id: "t1".to_string(),
                name: "bash".to_string(),
                input: "{}".to_string(),
            },
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert!(
            matches!(
                &message.blocks[0],
                ContentBlock::Thinking { thinking, signature }
                    if thinking == "reasoning" && signature == "SIG"
            ),
            "{:?}",
            message.blocks
        );
        assert!(matches!(
            &message.blocks[1],
            ContentBlock::RedactedThinking { data } if data == "BLOB"
        ));
        assert!(matches!(
            &message.blocks[2],
            ContentBlock::Text { text } if text == "the answer"
        ));
        assert!(matches!(&message.blocks[3], ContentBlock::ToolUse { .. }));
    }

    /// An unsigned streamed thinking block is still stored (with an empty
    /// signature); it is the lowering seam, not this reducer, that drops it.
    #[test]
    fn build_assistant_message_stores_unsigned_thinking_with_empty_signature() {
        use crate::session::ContentBlock;
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::Thinking {
                thinking: "omitted".to_string(),
                signature: None,
            },
            AssistantEvent::TextDelta("answer".to_string()),
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert!(matches!(
            &message.blocks[0],
            ContentBlock::Thinking { signature, .. } if signature.is_empty()
        ));
    }

    #[test]
    fn provider_state_gemini_thought_signature_sets_existing_field() {
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::ProviderState(ProviderStateBlob::gemini_thought_signature("SIG_NEW")),
            AssistantEvent::TextDelta("ok".to_string()),
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert_eq!(message.thought_signature.as_deref(), Some("SIG_NEW"));
    }

    #[test]
    fn unrelated_provider_state_does_not_set_thought_signature() {
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::ProviderState(ProviderStateBlob::new(
                "openai",
                "openai.encrypted_reasoning",
                "SECRET",
            )),
            AssistantEvent::TextDelta("ok".to_string()),
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert!(message.thought_signature.is_none());
    }

    #[test]
    fn unknown_google_provider_state_kind_does_not_set_thought_signature() {
        let AssistantTurn::Content { message, .. } = build_assistant_message(vec![
            AssistantEvent::ProviderState(ProviderStateBlob::new(
                "google",
                "gemini.unrelated_state",
                "SECRET",
            )),
            AssistantEvent::TextDelta("ok".to_string()),
            AssistantEvent::MessageStop,
        ]) else {
            panic!("expected assistant message");
        };
        assert!(message.thought_signature.is_none());
    }

    #[test]
    fn success_renders_pure_structured_output() {
        // Hook appended text → merged is no longer valid JSON, but a
        // successful result must still render the pure structured output.
        let merged = format!("{EDIT_JSON}\n\nHook feedback:\nformatted");
        let render = render_value_for_tool_result(EDIT_JSON, &merged, false);
        assert!(
            render.get("filePath").is_some(),
            "structured value preserved"
        );
        assert!(!matches!(render, Value::String(_)));
    }

    #[test]
    fn error_renders_merged_reason_string() {
        let merged = format!("{EDIT_JSON}\n\nHook feedback (error):\ndenied by policy");
        let render = render_value_for_tool_result(EDIT_JSON, &merged, true);
        match render {
            Value::String(text) => assert!(text.contains("denied by policy")),
            other => panic!("expected merged string render, got {other:?}"),
        }
    }

    #[test]
    fn clean_output_parses_to_object() {
        let render = render_value_for_tool_result(EDIT_JSON, EDIT_JSON, false);
        assert_eq!(
            render.get("filePath").and_then(Value::as_str),
            Some("/tmp/a.rs")
        );
    }

    #[test]
    fn json_candidate_probe_keeps_structured_outputs_parseable() {
        assert!(might_be_json_value(EDIT_JSON));
        assert!(might_be_json_value("  [1,2,3]"));
        assert!(might_be_json_value("\"json string\""));
        assert!(might_be_json_value("42"));
        assert!(might_be_json_value("false"));
    }

    #[test]
    fn plain_text_result_uses_string_fast_path() {
        let raw = "match line in some source file with a path\n".repeat(10_000);
        assert!(!might_be_json_value(&raw));
        let render = render_value_for_tool_result(&raw, &raw, false);
        assert_eq!(render, Value::String(raw));
    }

    /// Profiling probe (run with `--nocapture`): how long does the *synchronous*
    /// tool-result processing — which runs inline in the streaming turn future
    /// between `.await`s — take for large tool outputs? If it exceeds the 50 ms
    /// render-tick budget it starves the TUI render loop (freeze-then-burst).
    #[test]
    fn profile_large_tool_result_processing() {
        use std::time::Instant;

        // A grep_search result with 50k matches as structured JSON.
        let matches: Vec<String> = (0..50_000)
            .map(|i| format!("crates/x/src/file{i}.rs:{i}:12: pub fn handler_{i}() {{ ok }}"))
            .collect();
        let big_json = serde_json::json!({ "matches": matches }).to_string();
        eprintln!("[PROFILE] grep JSON input = {} bytes", big_json.len());

        let t = Instant::now();
        let value = render_value_for_tool_result(&big_json, &big_json, false);
        eprintln!(
            "[PROFILE] render_value_for_tool_result (parse) = {} ms",
            t.elapsed().as_millis()
        );

        let t = Instant::now();
        let _ = crate::message_stream::anthropic::tools::format_tool_result(
            "grep_search",
            &value,
            false,
        );
        eprintln!(
            "[PROFILE] format_tool_result(grep) = {} ms",
            t.elapsed().as_millis()
        );

        // A large non-JSON output (serde fails → falls back to String).
        let big_text = "match line in some source file with a path\n".repeat(200_000);
        eprintln!("[PROFILE] non-JSON input = {} bytes", big_text.len());
        let t = Instant::now();
        let value = render_value_for_tool_result(&big_text, &big_text, false);
        eprintln!(
            "[PROFILE] render_value(non-json) = {} ms",
            t.elapsed().as_millis()
        );
        let t = Instant::now();
        let _ = crate::message_stream::anthropic::tools::format_tool_result(
            "grep_search",
            &value,
            false,
        );
        eprintln!(
            "[PROFILE] format_tool_result(non-json) = {} ms",
            t.elapsed().as_millis()
        );
    }
}

#[cfg(test)]
mod ask_input_tests {
    use super::parse_ask_input;

    #[test]
    fn options_array_parses_normally() {
        let parsed = parse_ask_input(
            r#"{"question":"진행할까요?","options":[{"label":"네","description":"바로 진행"},"아니오"]}"#,
        )
        .expect("plain array options");
        let options = parsed.options.expect("options present");
        assert_eq!(options.len(), 2);
        assert_eq!(options[0].label, "네");
    }

    #[test]
    fn double_encoded_options_string_is_tolerated() {
        // Some models stringify the nested array; the question must still
        // reach the user instead of dying on a parse error.
        let parsed = parse_ask_input(
            r#"{"question":"제거로 진행?","options":"[{\"label\": \"제거로 진행\"}, {\"label\": \"유지\"}]"}"#,
        )
        .expect("double-encoded options must parse");
        let options = parsed.options.expect("options present");
        assert_eq!(options.len(), 2);
        assert_eq!(options[1].label, "유지");
    }

    #[test]
    fn invalid_options_string_reports_clearly() {
        let error = parse_ask_input(r#"{"question":"?","options":"not json"}"#)
            .expect_err("garbage string must still fail");
        assert!(
            error.contains("`options` was a string"),
            "error should name the double-encoding: {error}"
        );
    }
}
