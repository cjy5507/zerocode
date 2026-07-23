//! Single source of truth for lowering stored [`ConversationMessage`]s into the
//! provider-facing [`InputMessage`] wire form.
//!
//! Both the foreground conversation and sub-agent provider clients resend their
//! full history every iteration, so this conversion is on the hottest seam and
//! had drifted into two near-identical copies. Keeping it here — beside the
//! [`crate::context_compression::wire_tool_output`] it depends on — collapses
//! that drift surface to one place.

use api::{ImageSource, InputContentBlock, InputMessage, ToolResultContentBlock};

use crate::image_guard::{guard_wire_image_base64, oversized_placeholder, WireImageOutcome};
use crate::{ContentBlock, ConversationMessage, MessageRole};

/// Lower stored conversation messages into provider wire messages.
///
/// Maps every block variant, applies the shared model-facing structural
/// compression to tool output (the stored session block keeps the original, so
/// the TUI, persistence, and reversibility are unaffected), carries each turn's
/// Gemini thought signature through, and drops messages that lower to no content.
///
/// Consecutive [`MessageRole::Tool`] messages coalesce into ONE wire `user`
/// message. A parallel tool batch is stored as one result message per tool,
/// but the Anthropic API requires every `tool_use` id of an assistant turn to
/// be answered in *the single next* message — as separate messages the request
/// only survived because the API happens to merge same-role neighbours, and
/// anything ever slotted between two results 400s the whole turn
/// (`tool_use ids were found without tool_result blocks immediately after`).
/// Enforcing the invariant here, at the lowering SSOT, protects every
/// provider and every injection feature above it. Per-block encoders
/// (`openai_compat` emits one `role:"tool"` message per `ToolResult` block)
/// are unaffected by the grouping.
#[must_use]
pub fn convert_messages(messages: &[ConversationMessage]) -> Vec<InputMessage> {
    let mut out: Vec<InputMessage> = Vec::with_capacity(messages.len());
    // Whether the newest message in `out` lowered from `MessageRole::Tool` —
    // messages that lower to no content are invisible on the wire and keep
    // the adjacency alive.
    let mut tail_is_tool_run = false;
    for message in messages {
        let role = match message.role {
            MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
            MessageRole::Assistant => "assistant",
        };
        let content = convert_blocks(message);
        if content.is_empty() {
            continue;
        }
        let is_tool = message.role == MessageRole::Tool;
        if is_tool && tail_is_tool_run {
            if let Some(previous) = out.last_mut() {
                previous.content.extend(content);
                continue;
            }
        }
        out.push(InputMessage {
            role: role.to_string(),
            content,
            // Carry the turn's Gemini thought signature through. Harmless for
            // other providers: their encoders ignore `thought_signature`, and
            // it is `serde(skip)` so it never reaches any wire but Gemini's.
            thought_signature: message.thought_signature.clone(),
            // Carry the turn's ChatGPT reasoning-replay payload through, same
            // isolation as `thought_signature` above: only the ChatGPT
            // encoder reads it.
            reasoning_replay: message.reasoning_replay.clone(),
        });
        tail_is_tool_run = is_tool;
    }
    for message in &mut out {
        enforce_tool_results_lead(message);
    }
    out
}

/// Re-order a lowered `user` message so its `tool_result` blocks lead.
///
/// The Anthropic validator only credits results at the head of the message:
/// any other block ahead of a `tool_result` hides every result behind it and
/// the turn 400s (`tool_use ids were found without tool_result blocks
/// immediately after`, naming exactly the hidden ids). Text can legitimately
/// land amid a coalesced result run — `reconcile_tool_history` rewrites a
/// no-longer-advertised tool's result (e.g. a deferred builtin on a fresh
/// resume) into a text block in place, mid-batch. The partition is stable, so
/// both the result order and the narrative order are preserved.
fn enforce_tool_results_lead(message: &mut InputMessage) {
    if message.role != "user" {
        return;
    }
    let mut seen_other = false;
    let misplaced = message.content.iter().any(|block| {
        let is_result = matches!(block, InputContentBlock::ToolResult { .. });
        let out_of_place = is_result && seen_other;
        seen_other |= !is_result;
        out_of_place
    });
    if !misplaced {
        return;
    }
    let (mut results, rest): (Vec<_>, Vec<_>) = message
        .content
        .drain(..)
        .partition(|block| matches!(block, InputContentBlock::ToolResult { .. }));
    results.extend(rest);
    message.content = results;
}

/// Lower one stored message's blocks into wire content blocks.
///
/// Returns `Option` per block so a stored thinking block that lacks a signature
/// (legacy data, or `display:"omitted"` with no signature) is **dropped** rather
/// than lowered: the Anthropic API 400s on a modified or unsigned thinking block
/// but tolerates omission, so dropping is the safe degradation. Block order is
/// otherwise preserved, so replayed thinking still leads the assistant turn.
fn convert_blocks(message: &ConversationMessage) -> Vec<InputContentBlock> {
    message
        .blocks
        .iter()
        .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(InputContentBlock::Text {
                        text: text.clone(),
                        cache_control: None,
                    }),
                    // Anthropic reasoning replay: re-send the stored block
                    // VERBATIM with its signature so the model keeps interleaved-
                    // thinking continuity across tool calls. Only the Anthropic
                    // wire encoder emits these variants; the OpenAI/Gemini
                    // encoders match and drop them, so thinking never crosses
                    // providers (same isolation as `thought_signature`). An
                    // unsigned block is dropped — sending one unsigned 400s.
                    ContentBlock::Thinking { thinking, signature } => (!signature.is_empty())
                        .then(|| InputContentBlock::Thinking {
                            thinking: thinking.clone(),
                            signature: signature.clone(),
                        }),
                    ContentBlock::RedactedThinking { data } => {
                        Some(InputContentBlock::RedactedThinking { data: data.clone() })
                    }
                    ContentBlock::ToolUse { id, name, input } => Some(InputContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: serde_json::from_str(input)
                            .unwrap_or_else(|_| serde_json::json!({ "raw": input })),
                        cache_control: None,
                    }),
                    ContentBlock::ToolResult {
                        tool_use_id,
                        tool_name,
                        output,
                        is_error,
                        images,
                    } => {
                        // Model-facing structural compression (lossless unwrap
                        // / grouping; outline only for oversized code files).
                        // The session block keeps the original output, so the
                        // TUI, persistence, and reversibility are unaffected.
                        let mut content = vec![ToolResultContentBlock::Text {
                            text: crate::context_compression::wire_tool_output(
                                output, tool_name, *is_error,
                            ),
                        }];
                        // Dimension-guard every stored image on the way to the
                        // wire (see `image_guard`): an oversized screenshot
                        // baked into history otherwise 400s every turn. A drop
                        // degrades to a text placeholder so the model still
                        // learns an image was present.
                        content.extend(images.iter().map(|(media_type, data)| {
                            match guard_image_source(media_type, data) {
                                Ok(source) => ToolResultContentBlock::Image { source },
                                Err(placeholder) => {
                                    ToolResultContentBlock::Text { text: placeholder }
                                }
                            }
                        }));
                        Some(InputContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content,
                            is_error: *is_error,
                            cache_control: None,
                        })
                    }
                    ContentBlock::Image { media_type, data } => {
                        Some(match guard_image_source(media_type, data) {
                            Ok(source) => InputContentBlock::Image {
                                source,
                                cache_control: None,
                            },
                            Err(placeholder) => InputContentBlock::Text {
                                text: placeholder,
                                cache_control: None,
                            },
                        })
                    }
                })
        .collect::<Vec<_>>()
}

/// Dimension-guard one stored `(media_type, base64)` image on the way to the
/// provider wire. Returns an `ImageSource` for a kept or downscaled image, or a
/// text placeholder (`Err`) when a confirmed-oversized image could not be
/// downscaled — in which case the caller substitutes a text block so the model
/// still sees that an image was present. Shared by the `ToolResult` image list
/// and standalone `Image` arms, which lower into different wire enums.
fn guard_image_source(media_type: &str, data: &str) -> Result<ImageSource, String> {
    match guard_wire_image_base64(data) {
        WireImageOutcome::Keep => Ok(ImageSource {
            kind: "base64".to_string(),
            media_type: media_type.to_string(),
            data: data.to_string(),
        }),
        WireImageOutcome::Rescaled {
            media_type,
            data_b64,
        } => Ok(ImageSource {
            kind: "base64".to_string(),
            media_type,
            data: data_b64,
        }),
        WireImageOutcome::Drop { width, height } => Err(oversized_placeholder(width, height)),
    }
}

/// Append harness reminders (recalled memory, todo progress, `TeamInbox` digest,
/// …) to the newest `user`-role wire message as trailing text blocks.
///
/// Message-level injection — rather than a trailing system-prompt section — is
/// what keeps the Anthropic prefix cache alive: the cache hierarchy is
/// tools → system → messages, so a system block that changes (`system_changed`)
/// invalidates every message breakpoint behind it, re-billing the entire
/// history at cache-write price each time recall or a reminder refreshes.
/// Riding the newest user message instead re-bills only that message.
///
/// Each reminder is wrapped in `<system-reminder>` tags (the static prompt
/// tells the model these carry system information) unless the producer already
/// embedded its own tags, e.g. the `UserPromptSubmit` hook reminder — wrapping
/// those again would nest tags.
///
/// Trailing position also satisfies the API rule that `tool_result` blocks
/// lead their user message.
pub fn append_wire_reminders(messages: &mut Vec<InputMessage>, reminders: &[String]) {
    let mut blocks: Vec<InputContentBlock> = reminders
        .iter()
        .filter(|reminder| !reminder.trim().is_empty())
        .map(|reminder| InputContentBlock::Text {
            text: if reminder.contains("<system-reminder>") {
                reminder.clone()
            } else {
                format!("<system-reminder>\n{reminder}\n</system-reminder>")
            },
            cache_control: None,
        })
        .collect();
    if blocks.is_empty() {
        return;
    }
    if let Some(last_user) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == "user")
    {
        last_user.content.append(&mut blocks);
    } else {
        // First request of a session always carries a user message; this arm
        // only guards a pathological empty/assistant-tail history.
        messages.push(InputMessage {
            role: "user".to_string(),
            content: blocks,
            thought_signature: None,
            reasoning_replay: None,
        });
    }
}

/// Place Anthropic prompt-cache breakpoints on the latest conversation prefix.
///
/// System blocks already consume up to two of Anthropic's four cache-control
/// slots. Marking the last two cacheable message blocks gives every multi-turn
/// caller — the foreground turn and sub-agent provider clients alike — a
/// rolling conversation prefix cache while staying within the provider limit.
/// The sub-agent path previously skipped this step entirely, so only its
/// system blocks ever cached and each iteration re-billed the full transcript
/// as uncached input.
///
/// Run this after [`append_wire_reminders`] so the reminder tail sits under
/// the newest breakpoint. Harmless on non-Anthropic wires: the OpenAI and
/// Gemini encoders build their own payloads from these blocks and never
/// serialize `cache_control`.
pub fn mark_conversation_cache_breakpoints(messages: &mut [InputMessage]) {
    mark_breakpoints_with_ttl(messages, &api::CacheControl::ephemeral_1h());
}

/// Sub-agent variant of [`mark_conversation_cache_breakpoints`]: identical
/// rolling breakpoints at the 5-minute TTL. A sub-agent lives minutes and its
/// next iteration lands seconds after the last, so the 1h write premium
/// (2.0x vs 1.25x) buys nothing on its conversation tail — the dominant
/// spawn-side cache-write volume. The shared system prefix keeps its 1h
/// breakpoints (`agent_system_blocks`), so later spawns still reuse it and
/// the request satisfies Anthropic's longer-TTL-before-shorter ordering rule.
pub fn mark_conversation_cache_breakpoints_short_ttl(messages: &mut [InputMessage]) {
    mark_breakpoints_with_ttl(messages, &api::CacheControl::ephemeral());
}

fn mark_breakpoints_with_ttl(messages: &mut [InputMessage], cache_control: &api::CacheControl) {
    const MAX_MESSAGE_CACHE_BREAKPOINTS: usize = 2;

    let mut marked = 0;
    for message in messages.iter_mut().rev() {
        if marked == MAX_MESSAGE_CACHE_BREAKPOINTS {
            break;
        }
        if mark_last_cacheable_block(&mut message.content, cache_control) {
            marked += 1;
        }
    }
}

/// Mark one message's last markable block, returning whether a marker landed.
///
/// Tool blocks are markable: an agentic loop's newest messages are almost
/// always a pure `tool_use`/`tool_result` exchange, and skipping them (the
/// pre-fix shape, when the wire enum had no `cache_control` field on tool
/// variants) pinned the breakpoint at the last *text-bearing* message — every
/// iteration then re-billed the entire tool tail behind it as uncached input.
///
/// Two variants still fall through to the previous block: thinking blocks
/// take no `cache_control` at all, and Anthropic 400s on `cache_control` over
/// an *empty* text block (`cache_control cannot be set for empty text
/// blocks`) — a long session can end a message with a blank text block, e.g.
/// a trailing empty delta.
fn mark_last_cacheable_block(blocks: &mut [InputContentBlock], ttl: &api::CacheControl) -> bool {
    for block in blocks.iter_mut().rev() {
        match block {
            InputContentBlock::Text {
                text,
                cache_control,
            } => {
                if text.trim().is_empty() {
                    continue;
                }
                if cache_control.is_none() {
                    *cache_control = Some(ttl.clone());
                }
                return true;
            }
            InputContentBlock::Image { cache_control, .. }
            | InputContentBlock::Document { cache_control, .. }
            | InputContentBlock::ToolUse { cache_control, .. }
            | InputContentBlock::ToolResult { cache_control, .. } => {
                if cache_control.is_none() {
                    *cache_control = Some(ttl.clone());
                }
                return true;
            }
            InputContentBlock::Thinking { .. } | InputContentBlock::RedactedThinking { .. } => {}
        }
    }
    false
}

#[cfg(test)]
mod cache_marking_tests {
    use super::{convert_messages, mark_conversation_cache_breakpoints, mark_last_cacheable_block};
    use crate::{ContentBlock, ConversationMessage};
    use api::InputContentBlock;

    fn block_cache(block: &InputContentBlock) -> Option<&api::CacheControl> {
        match block {
            InputContentBlock::Text { cache_control, .. }
            | InputContentBlock::Image { cache_control, .. }
            | InputContentBlock::Document { cache_control, .. }
            | InputContentBlock::ToolUse { cache_control, .. }
            | InputContentBlock::ToolResult { cache_control, .. } => cache_control.as_ref(),
            InputContentBlock::Thinking { .. } | InputContentBlock::RedactedThinking { .. } => {
                None
            }
        }
    }

    #[test]
    fn marks_last_two_cacheable_message_blocks_for_prefix_reuse() {
        let messages = vec![
            ConversationMessage::user_text("first prompt"),
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: "second response".to_string(),
            }]),
            ConversationMessage::user_text("third prompt"),
        ];

        let mut converted = convert_messages(&messages);
        mark_conversation_cache_breakpoints(&mut converted);

        assert!(block_cache(&converted[0].content[0]).is_none());
        assert_eq!(
            block_cache(&converted[1].content[0]),
            Some(&api::CacheControl::ephemeral_1h())
        );
        assert_eq!(
            block_cache(&converted[2].content[0]),
            Some(&api::CacheControl::ephemeral_1h())
        );
    }

    /// The agentic-loop shape: the newest messages are a pure
    /// `tool_use`/`tool_result` exchange with no text anywhere near the tail.
    /// Both breakpoints must land inside that tail — on the coalesced result
    /// batch and the `tool_use` turn — NOT fall back to the last text-bearing
    /// message far behind it (the pre-fix behavior that re-billed the whole
    /// tool tail every iteration).
    #[test]
    fn tool_only_tail_carries_the_breakpoints() {
        let tool_use = |id: &str| ContentBlock::ToolUse {
            id: id.to_string(),
            name: "bash".to_string(),
            input: "{}".to_string(),
        };
        let messages = vec![
            ConversationMessage::user_text("prompt"),
            ConversationMessage::assistant(vec![tool_use("tu-1")]),
            ConversationMessage::tool_result("tu-1", "bash", "one", false),
            ConversationMessage::assistant(vec![tool_use("tu-2")]),
            ConversationMessage::tool_result("tu-2", "bash", "two", false),
        ];

        let mut converted = convert_messages(&messages);
        mark_conversation_cache_breakpoints(&mut converted);

        let last = converted.len() - 1;
        assert!(
            matches!(&converted[last].content[0], InputContentBlock::ToolResult { .. }),
            "fixture: tail is a tool_result message"
        );
        assert_eq!(
            block_cache(&converted[last].content[0]),
            Some(&api::CacheControl::ephemeral_1h()),
            "newest tool_result carries a breakpoint"
        );
        assert_eq!(
            block_cache(&converted[last - 1].content[0]),
            Some(&api::CacheControl::ephemeral_1h()),
            "the tool_use turn before it carries the second breakpoint"
        );
        assert!(
            block_cache(&converted[0].content[0]).is_none(),
            "the old text message no longer soaks up a marker"
        );
    }

    /// Thinking blocks take no `cache_control`; a thinking-led assistant turn
    /// must land its marker on the `tool_use` behind the thinking block.
    #[test]
    fn thinking_blocks_fall_through_to_the_tool_use() {
        let message = ConversationMessage::assistant(vec![
            ContentBlock::Thinking {
                thinking: "reasoning".to_string(),
                signature: "SIG".to_string(),
            },
            ContentBlock::ToolUse {
                id: "tu-1".to_string(),
                name: "bash".to_string(),
                input: "{}".to_string(),
            },
        ]);
        let mut converted = convert_messages(&[message]);
        mark_conversation_cache_breakpoints(&mut converted);
        let content = &converted[0].content;
        assert!(matches!(&content[0], InputContentBlock::Thinking { .. }));
        assert_eq!(
            block_cache(&content[1]),
            Some(&api::CacheControl::ephemeral_1h()),
            "marker lands on the tool_use, not the thinking block"
        );
    }

    #[test]
    fn empty_text_block_is_skipped_when_cache_marking() {
        // Anthropic 400s on `cache_control` over an empty text block. When a
        // message ends with a blank text block, the marker must fall through to
        // the previous non-empty block instead of landing on the empty one.
        let mut blocks = vec![
            InputContentBlock::Text {
                text: "real answer".into(),
                cache_control: None,
            },
            InputContentBlock::Text {
                text: "  \n".into(), // whitespace-only → treated as empty
                cache_control: None,
            },
        ];
        assert!(mark_last_cacheable_block(
            &mut blocks,
            &api::CacheControl::ephemeral_1h()
        ));
        let InputContentBlock::Text {
            cache_control: trailing,
            ..
        } = &blocks[1]
        else {
            panic!("expected a Text block");
        };
        assert!(
            trailing.is_none(),
            "blank text block must not be cache-marked"
        );
        let InputContentBlock::Text {
            cache_control: real,
            ..
        } = &blocks[0]
        else {
            panic!("expected a Text block");
        };
        assert!(
            real.is_some(),
            "the previous non-empty block must carry the marker"
        );
    }

    #[test]
    fn all_empty_blocks_mark_nothing() {
        let mut blocks = vec![InputContentBlock::Text {
            text: String::new(),
            cache_control: None,
        }];
        assert!(
            !mark_last_cacheable_block(&mut blocks, &api::CacheControl::ephemeral_1h()),
            "no non-empty cacheable block → nothing marked"
        );
    }

    /// Sub-agent variant: same rolling breakpoints, 5-minute TTL (`ttl: None`
    /// on the wire) — the spawn tail dies within minutes, so it must not pay
    /// the 1h write premium the foreground turn pays.
    #[test]
    fn short_ttl_variant_marks_with_five_minute_ttl() {
        let messages = vec![
            ConversationMessage::user_text("first prompt"),
            ConversationMessage::assistant(vec![ContentBlock::Text {
                text: "second response".to_string(),
            }]),
            ConversationMessage::user_text("third prompt"),
        ];

        let mut converted = convert_messages(&messages);
        super::mark_conversation_cache_breakpoints_short_ttl(&mut converted);

        assert!(block_cache(&converted[0].content[0]).is_none());
        assert_eq!(
            block_cache(&converted[1].content[0]),
            Some(&api::CacheControl::ephemeral())
        );
        assert_eq!(
            block_cache(&converted[2].content[0]),
            Some(&api::CacheControl::ephemeral())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{append_wire_reminders, convert_messages};
    use crate::ConversationMessage;
    use api::{InputContentBlock, ToolResultContentBlock};

    fn wire_text_of(converted: &[api::InputMessage]) -> &str {
        let InputContentBlock::ToolResult { content, .. } = &converted[0].content[0] else {
            panic!("expected a ToolResult");
        };
        let ToolResultContentBlock::Text { text } = &content[0] else {
            panic!("expected a Text block");
        };
        text
    }

    #[test]
    fn tool_result_images_become_wire_image_blocks() {
        // A stored ToolResult carrying an image must convert to an
        // InputContentBlock::ToolResult whose content has a Text block followed
        // by a ToolResultContentBlock::Image — the model actually sees pixels.
        let message = ConversationMessage::tool_result_with_images(
            "tu-1",
            "read_image",
            "staged",
            false,
            vec![("image/png".to_string(), "QUJD".to_string())],
        );
        let converted = convert_messages(&[message]);
        assert_eq!(converted.len(), 1);
        let InputContentBlock::ToolResult { content, .. } = &converted[0].content[0] else {
            panic!("expected an InputContentBlock::ToolResult");
        };
        assert_eq!(content.len(), 2, "text block + one image block");
        assert!(matches!(content[0], ToolResultContentBlock::Text { .. }));
        match &content[1] {
            ToolResultContentBlock::Image { source } => {
                assert_eq!(source.kind, "base64");
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, "QUJD");
            }
            other => panic!("expected an Image block, got {other:?}"),
        }
    }

    #[test]
    fn oversized_stored_tool_result_image_is_downscaled_on_the_wire() {
        // The session-wedge regression: a full-page screenshot taller than
        // 8000px baked into a stored tool_result 400s every turn. Lowering must
        // downscale it (to PNG) so an already-poisoned history un-wedges without
        // surgery. Build a real oversized PNG so the guard actually fires.
        use base64::Engine as _;
        use image::{DynamicImage, ImageFormat, RgbImage};

        let mut buf = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(400, 12000))
            .write_to(&mut buf, ImageFormat::Png)
            .expect("encode oversized test PNG");
        let data = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());

        let message = ConversationMessage::tool_result_with_images(
            "tu-oversized",
            "read_image",
            "staged",
            false,
            vec![("image/png".to_string(), data.clone())],
        );
        let converted = convert_messages(&[message]);
        let InputContentBlock::ToolResult { content, .. } = &converted[0].content[0] else {
            panic!("expected a ToolResult");
        };
        let ToolResultContentBlock::Image { source } = &content[1] else {
            panic!("expected a downscaled image block, got {:?}", content[1]);
        };
        assert_eq!(source.media_type, "image/png", "downscale re-encodes to PNG");
        assert_ne!(source.data, data, "the oversized payload must be replaced");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(source.data.as_bytes())
            .expect("valid base64 out");
        let dims = crate::image_guard::guard_image_bytes(&decoded);
        assert_eq!(
            dims,
            crate::image_guard::ImageGuardOutcome::Keep,
            "the lowered image must now be within the cap"
        );
    }

    #[test]
    fn text_only_tool_result_has_no_image_block() {
        let message = ConversationMessage::tool_result("tu-2", "bash", "ok", false);
        let converted = convert_messages(&[message]);
        let InputContentBlock::ToolResult { content, .. } = &converted[0].content[0] else {
            panic!("expected a ToolResult");
        };
        assert_eq!(content.len(), 1, "text-only → exactly one Text block");
        assert!(matches!(content[0], ToolResultContentBlock::Text { .. }));
    }

    #[test]
    fn read_file_result_is_compressed_on_the_wire_only() {
        // The session block keeps the pretty-JSON envelope; the wire view the
        // model sees is the compact unwrapped form.
        let body = "fn main() {\n    println!(\"hello\");\n}\n".repeat(30);
        let envelope = serde_json::to_string_pretty(&serde_json::json!({
            "type": "text",
            "file": {
                "filePath": "/ws/src/main.rs",
                "content": body,
                "numLines": body.lines().count(),
                "startLine": 1,
                "totalLines": body.lines().count(),
            }
        }))
        .expect("serialize envelope");
        let message = ConversationMessage::tool_result("tu-3", "read_file", &envelope, false);
        let converted = convert_messages(std::slice::from_ref(&message));
        let wire = wire_text_of(&converted);
        assert!(
            wire.starts_with("[file] /ws/src/main.rs"),
            "wire view is unwrapped"
        );
        assert!(
            wire.contains("println!(\"hello\");"),
            "content preserved verbatim"
        );
        assert!(wire.len() < envelope.len(), "wire view is smaller");
        // And the stored session block still carries the original envelope.
        let crate::session::ContentBlock::ToolResult { output, .. } = &message.blocks[0] else {
            panic!("expected a session ToolResult");
        };
        assert_eq!(output, &envelope, "session history is untouched");
    }

    #[test]
    fn error_tool_result_passes_through_verbatim() {
        let error_text = "io error: old_string not found in file";
        let message = ConversationMessage::tool_result("tu-4", "edit_file", error_text, true);
        let converted = convert_messages(&[message]);
        assert_eq!(wire_text_of(&converted), error_text);
    }

    #[test]
    fn unknown_tool_result_passes_through_verbatim() {
        let payload = r#"{"anything": "goes", "even": ["json"]}"#;
        let message = ConversationMessage::tool_result("tu-5", "TodoWrite", payload, false);
        let converted = convert_messages(&[message]);
        assert_eq!(wire_text_of(&converted), payload);
    }

    #[test]
    fn parallel_tool_results_coalesce_into_one_wire_message() {
        // A parallel batch stores one Tool message per result, but the API
        // requires every tool_use id of the assistant turn to be answered in
        // THE single next message. Lowering coalesces the run so the
        // invariant holds in code instead of leaning on the API's
        // same-role-merge leniency (the 400 class: "tool_use ids were found
        // without tool_result blocks immediately after").
        let tool_use = |id: &str| crate::session::ContentBlock::ToolUse {
            id: id.to_string(),
            name: "bash".to_string(),
            input: "{}".to_string(),
        };
        let converted = convert_messages(&[
            ConversationMessage::user_text("prompt"),
            ConversationMessage::assistant(vec![
                tool_use("tu-1"),
                tool_use("tu-2"),
                tool_use("tu-3"),
            ]),
            ConversationMessage::tool_result("tu-1", "bash", "one", false),
            ConversationMessage::tool_result("tu-2", "bash", "two", false),
            ConversationMessage::tool_result("tu-3", "bash", "three", false),
        ]);
        assert_eq!(
            converted.len(),
            3,
            "user, assistant, ONE coalesced result message"
        );
        assert_eq!(converted[2].role, "user");
        let ids: Vec<&str> = converted[2]
            .content
            .iter()
            .map(|block| match block {
                InputContentBlock::ToolResult { tool_use_id, .. } => tool_use_id.as_str(),
                other => panic!("coalesced message must be pure tool results, got {other:?}"),
            })
            .collect();
        assert_eq!(ids, ["tu-1", "tu-2", "tu-3"], "order preserved");
    }

    #[test]
    fn mid_batch_reconcile_text_rides_behind_the_coalesced_results() {
        // session-1783238633488-8 regression: the MIDDLE tool of a parallel
        // batch (TaskStop) is a deferred builtin, so a fresh resume does not
        // advertise it and `reconcile_tool_history` rewrites its result to a
        // text block in place. Coalescing then shipped
        // [result, text, result] and the API refused to credit the result
        // behind the text — a 400 naming exactly the last id. The results
        // must lead the lowered message; the rewrite text rides behind them.
        let tool_use = |id: &str, name: &str| crate::session::ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: "{}".to_string(),
        };
        let stored = std::sync::Arc::new(vec![
            ConversationMessage::user_text("prompt"),
            ConversationMessage::assistant(vec![
                tool_use("tu-todo", "TodoWrite"),
                tool_use("tu-stop", "TaskStop"),
                tool_use("tu-grep", "grep_search"),
            ]),
            ConversationMessage::tool_result("tu-todo", "TodoWrite", "ok", false),
            ConversationMessage::tool_result("tu-stop", "TaskStop", "already terminal", true),
            ConversationMessage::tool_result("tu-grep", "grep_search", "3 matches", false),
            ConversationMessage::user_text("ggo"),
        ]);
        let known: std::collections::BTreeSet<String> = ["TodoWrite", "grep_search"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let reconciled = crate::session::reconcile_tool_history(&stored, &known);
        let converted = convert_messages(&reconciled);

        assert_eq!(converted.len(), 4, "user, assistant, batch, user");
        let batch = &converted[2].content;
        assert_eq!(batch.len(), 3, "2 surviving results + 1 rewrite text");
        // The API-enforced invariant: every surviving tool_use id of the
        // assistant message is answered by the LEADING result run of the next
        // message.
        let leading: Vec<&str> = batch
            .iter()
            .take_while(|block| matches!(block, InputContentBlock::ToolResult { .. }))
            .map(|block| match block {
                InputContentBlock::ToolResult { tool_use_id, .. } => tool_use_id.as_str(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(leading, ["tu-todo", "tu-grep"], "results lead, order kept");
        let InputContentBlock::Text { text, .. } = &batch[2] else {
            panic!("rewritten TaskStop result should trail as text");
        };
        assert!(text.contains("TaskStop"), "rewrite context preserved");
    }

    #[test]
    fn leading_reconcile_text_does_not_hide_the_results_behind_it() {
        // Variant: the FIRST tool of the batch is the unknown one, so the
        // rewrite text starts the coalesced message and pre-fix would hide
        // EVERY result behind it.
        let tool_use = |id: &str, name: &str| crate::session::ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: "{}".to_string(),
        };
        let stored = std::sync::Arc::new(vec![
            ConversationMessage::assistant(vec![
                tool_use("tu-gone", "SendMessage"),
                tool_use("tu-a", "bash"),
                tool_use("tu-b", "read_file"),
            ]),
            ConversationMessage::tool_result("tu-gone", "SendMessage", "sent", false),
            ConversationMessage::tool_result("tu-a", "bash", "ok", false),
            ConversationMessage::tool_result("tu-b", "read_file", "body", false),
        ]);
        let known: std::collections::BTreeSet<String> = ["bash", "read_file"]
            .iter()
            .map(ToString::to_string)
            .collect();
        let reconciled = crate::session::reconcile_tool_history(&stored, &known);
        let converted = convert_messages(&reconciled);
        let batch = &converted[1].content;
        assert!(
            matches!(&batch[0], InputContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "tu-a"),
            "first block must be a result, not the rewrite text"
        );
        assert!(
            matches!(&batch[1], InputContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "tu-b")
        );
        assert!(matches!(&batch[2], InputContentBlock::Text { .. }));
    }

    #[test]
    fn separate_tool_runs_do_not_cross_merge() {
        let converted = convert_messages(&[
            ConversationMessage::tool_result("tu-1", "bash", "one", false),
            ConversationMessage::assistant(vec![crate::session::ContentBlock::Text {
                text: "between".to_string(),
            }]),
            ConversationMessage::tool_result("tu-2", "bash", "two", false),
        ]);
        assert_eq!(converted.len(), 3, "an assistant turn ends the run");
    }

    #[test]
    fn reminders_land_after_the_whole_coalesced_batch() {
        let mut converted = convert_messages(&[
            ConversationMessage::user_text("prompt"),
            ConversationMessage::tool_result("tu-1", "bash", "one", false),
            ConversationMessage::tool_result("tu-2", "bash", "two", false),
        ]);
        append_wire_reminders(&mut converted, &["todo progress".to_string()]);
        let tail = &converted[1].content;
        assert_eq!(tail.len(), 3, "2 results + 1 reminder in one message");
        assert!(matches!(tail[0], InputContentBlock::ToolResult { .. }));
        assert!(matches!(tail[1], InputContentBlock::ToolResult { .. }));
        assert!(
            matches!(tail[2], InputContentBlock::Text { .. }),
            "reminder rides behind the batch, never inside it"
        );
    }

    #[test]
    fn wire_reminders_ride_the_newest_user_message_as_tagged_tail_blocks() {
        let mut converted = convert_messages(&[
            ConversationMessage::user_text("first prompt"),
            ConversationMessage::assistant(vec![crate::session::ContentBlock::Text {
                text: "answer".to_string(),
            }]),
            ConversationMessage::tool_result("tu-1", "bash", "ok", false),
        ]);
        append_wire_reminders(
            &mut converted,
            &["[zo:todo-progress] item 2 in progress".to_string()],
        );

        // Prior messages untouched — the prefix stays byte-identical.
        assert_eq!(converted[0].content.len(), 1);
        assert_eq!(converted[1].content.len(), 1);
        // Reminder is the tail block of the newest user message (after the
        // tool_result, satisfying the tool_result-first rule), tag-wrapped.
        let tail = &converted[2].content;
        assert_eq!(tail.len(), 2);
        assert!(matches!(tail[0], InputContentBlock::ToolResult { .. }));
        let InputContentBlock::Text { text, .. } = &tail[1] else {
            panic!("expected trailing reminder text block");
        };
        assert!(text.starts_with("<system-reminder>\n"));
        assert!(text.contains("[zo:todo-progress] item 2 in progress"));
        assert!(text.ends_with("\n</system-reminder>"));
    }

    #[test]
    fn wire_reminders_skip_empty_and_never_double_wrap() {
        let mut converted = convert_messages(&[ConversationMessage::user_text("prompt")]);
        let pre_tagged = "[zo:hook]\n<system-reminder>\nhook context\n</system-reminder>";
        append_wire_reminders(
            &mut converted,
            &["   ".to_string(), pre_tagged.to_string()],
        );

        let content = &converted[0].content;
        assert_eq!(content.len(), 2, "blank reminder contributes no block");
        let InputContentBlock::Text { text, .. } = &content[1] else {
            panic!("expected reminder text block");
        };
        assert_eq!(text, pre_tagged, "producer-supplied tags are kept as-is");
    }

    #[test]
    fn wire_reminders_no_op_when_empty() {
        let mut converted = convert_messages(&[ConversationMessage::user_text("prompt")]);
        let before = converted.clone();
        append_wire_reminders(&mut converted, &[]);
        assert_eq!(converted, before);
    }

    /// A stored, signed thinking block lowers to the wire VERBATIM (text +
    /// signature) and leads the assistant turn — before text and `tool_use` — which
    /// is the order the Anthropic API validates on replay. The Anthropic path
    /// serializes `InputContentBlock` directly, so the serde shape below is the
    /// exact wire payload.
    #[test]
    fn signed_thinking_replays_verbatim_before_tool_use_on_the_wire() {
        let message = ConversationMessage::assistant(vec![
            crate::session::ContentBlock::Thinking {
                thinking: "let me reason".to_string(),
                signature: "SIG-xyz".to_string(),
            },
            crate::session::ContentBlock::Text {
                text: "answer".to_string(),
            },
            crate::session::ContentBlock::ToolUse {
                id: "tu-1".to_string(),
                name: "bash".to_string(),
                input: "{}".to_string(),
            },
        ]);
        let converted = convert_messages(&[message]);
        assert_eq!(converted.len(), 1);
        let content = &converted[0].content;
        assert!(
            matches!(
                &content[0],
                InputContentBlock::Thinking { thinking, signature }
                    if thinking == "let me reason" && signature == "SIG-xyz"
            ),
            "thinking must lead the turn: {content:?}"
        );
        assert!(matches!(&content[1], InputContentBlock::Text { .. }));
        assert!(matches!(&content[2], InputContentBlock::ToolUse { .. }));

        let wire = serde_json::to_value(&content[0]).expect("serialize thinking to wire");
        assert_eq!(wire["type"], "thinking");
        assert_eq!(wire["thinking"], "let me reason");
        assert_eq!(wire["signature"], "SIG-xyz");
    }

    /// A thinking block with no signature (legacy data, or `display:"omitted"`
    /// without a signature) is DROPPED rather than sent unsigned — the API 400s
    /// on an unsigned thinking block but tolerates omission.
    #[test]
    fn unsigned_thinking_is_dropped_not_sent() {
        let message = ConversationMessage::assistant(vec![
            crate::session::ContentBlock::Thinking {
                thinking: "legacy reasoning".to_string(),
                signature: String::new(),
            },
            crate::session::ContentBlock::Text {
                text: "answer".to_string(),
            },
        ]);
        let converted = convert_messages(&[message]);
        let content = &converted[0].content;
        assert_eq!(content.len(), 1, "unsigned thinking dropped: {content:?}");
        assert!(matches!(&content[0], InputContentBlock::Text { .. }));
        assert!(
            !content
                .iter()
                .any(|block| matches!(block, InputContentBlock::Thinking { .. })),
            "no unsigned thinking may reach the wire"
        );
    }

    /// Determinism pin: the smart-AUTO cache-collapse incident traced back to a
    /// suspicion that a provider-swap/history-lowering pass emitted different
    /// bytes on repeated calls for the same input, defeating Anthropic's prefix
    /// cache. `convert_messages` is the shared SSOT every provider request
    /// lowers through, so it must produce byte-identical (`PartialEq`) output
    /// for the same input every time. The fixture spans every branch that
    /// touches ordering or filtering: a signed thinking block (kept), an
    /// empty-signature thinking block — exactly how GPT-produced reasoning is
    /// stored in session history — (dropped), a parallel tool-use batch, and an
    /// image-bearing `tool_result`.
    #[test]
    fn convert_messages_is_deterministic_across_repeated_calls() {
        let messages = vec![
            ConversationMessage::user_text("prompt"),
            ConversationMessage::assistant(vec![
                crate::session::ContentBlock::Thinking {
                    thinking: "signed reasoning".to_string(),
                    signature: "SIG-1".to_string(),
                },
                crate::session::ContentBlock::Thinking {
                    thinking: "gpt reasoning, no signature".to_string(),
                    signature: String::new(),
                },
                crate::session::ContentBlock::Text {
                    text: "answer".to_string(),
                },
                crate::session::ContentBlock::ToolUse {
                    id: "tu-1".to_string(),
                    name: "bash".to_string(),
                    input: "{}".to_string(),
                },
                crate::session::ContentBlock::ToolUse {
                    id: "tu-2".to_string(),
                    name: "read_file".to_string(),
                    input: "{}".to_string(),
                },
            ]),
            ConversationMessage::tool_result("tu-1", "bash", "one", false),
            ConversationMessage::tool_result_with_images(
                "tu-2",
                "read_file",
                "staged",
                false,
                vec![("image/png".to_string(), "QUJD".to_string())],
            ),
        ];

        let first = convert_messages(&messages);
        let second = convert_messages(&messages);
        assert_eq!(
            first, second,
            "same &[ConversationMessage] must lower to byte-identical output every call"
        );

        // And confirm the empty-signature thinking never survives into either
        // run — it must never reach a provider wire (see anthropic.rs tests for
        // the follow-up trace of what would happen if it ever did).
        for converted in [&first, &second] {
            assert!(
                !converted
                    .iter()
                    .flat_map(|message| &message.content)
                    .any(|block| matches!(
                        block,
                        InputContentBlock::Thinking { signature, .. } if signature.is_empty()
                    )),
                "empty-signature thinking must never reach the wire"
            );
        }
    }

    /// `append_wire_reminders` mutates in place, so pin determinism by running
    /// it twice from independent clones of the same starting state.
    #[test]
    fn append_wire_reminders_is_deterministic() {
        let base = convert_messages(&[
            ConversationMessage::user_text("first"),
            ConversationMessage::assistant(vec![crate::session::ContentBlock::Text {
                text: "answer".to_string(),
            }]),
            ConversationMessage::tool_result("tu-1", "bash", "ok", false),
        ]);
        let reminders = vec![
            "[zo:todo-progress] item 2 in progress".to_string(),
            "   ".to_string(),
            "<system-reminder>\nalready tagged\n</system-reminder>".to_string(),
        ];

        let mut run_a = base.clone();
        append_wire_reminders(&mut run_a, &reminders);
        let mut run_b = base.clone();
        append_wire_reminders(&mut run_b, &reminders);

        assert_eq!(
            run_a, run_b,
            "same starting messages + reminders must append identically every call"
        );
    }

    /// The stronger form of "attaches to the newest user-role message": a
    /// trailing ASSISTANT turn must never receive it, even though it is the
    /// literal last message in the list. Only the last message whose *role* is
    /// `user` may change.
    #[test]
    fn append_wire_reminders_attaches_only_to_the_last_user_role_message_even_behind_an_assistant_tail(
    ) {
        let mut converted = convert_messages(&[
            ConversationMessage::user_text("first"),
            ConversationMessage::assistant(vec![crate::session::ContentBlock::Text {
                text: "a1".to_string(),
            }]),
            ConversationMessage::user_text("second"),
            ConversationMessage::assistant(vec![crate::session::ContentBlock::Text {
                text: "a2".to_string(),
            }]),
        ]);
        let before: Vec<usize> = converted.iter().map(|message| message.content.len()).collect();

        append_wire_reminders(&mut converted, &["reminder".to_string()]);

        assert_eq!(
            converted[0].content.len(),
            before[0],
            "earliest user message untouched"
        );
        assert_eq!(
            converted[1].content.len(),
            before[1],
            "first assistant turn untouched"
        );
        assert_eq!(converted[2].role, "user");
        assert_eq!(
            converted[2].content.len(),
            before[2] + 1,
            "the LAST user-role message gets the reminder"
        );
        assert_eq!(converted[3].role, "assistant");
        assert_eq!(
            converted[3].content.len(),
            before[3],
            "the trailing assistant turn — the actual last message — must NOT receive it"
        );
    }

    /// A `redacted_thinking` block carries no signature but is still replayed
    /// verbatim (its `data` is the encrypted reasoning the API returned).
    #[test]
    fn redacted_thinking_replays_on_the_wire() {
        let message = ConversationMessage::assistant(vec![
            crate::session::ContentBlock::RedactedThinking {
                data: "ENCRYPTED".to_string(),
            },
            crate::session::ContentBlock::Text {
                text: "answer".to_string(),
            },
        ]);
        let converted = convert_messages(&[message]);
        let content = &converted[0].content;
        assert!(matches!(
            &content[0],
            InputContentBlock::RedactedThinking { data } if data == "ENCRYPTED"
        ));
        let wire = serde_json::to_value(&content[0]).expect("serialize redacted to wire");
        assert_eq!(wire["type"], "redacted_thinking");
        assert_eq!(wire["data"], "ENCRYPTED");
    }
}
