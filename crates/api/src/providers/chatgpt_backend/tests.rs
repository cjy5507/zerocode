use super::{
    ResponsesSseParser, ResponsesStreamState, build_responses_request,
    build_responses_request_for_session, cache_reasoning_for_call, crosses_restart_commit_boundary,
    de_escalated_effort, deescalated_recovery_request, dynamic_effort, reasoning_effort,
    reasoning_for_call, remove_reasoning_for_call,
};
use crate::providers::should_restart;
use crate::error::ApiError;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, EffortLevel, ContentBlockStopEvent, ImageSource,
    InputContentBlock, InputMessage, MessageRequest, OutputContentBlock, StreamEvent, SystemBlock,
    ThinkingConfig, ToolChoice, ToolDefinition,
};
use serde_json::json;

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::test_env_lock()
}

fn request(
    messages: Vec<InputMessage>,
    tools: Option<Vec<ToolDefinition>>,
    thinking: Option<ThinkingConfig>,
) -> MessageRequest {
    MessageRequest {
        model: "gpt-5.6-sol".into(),
        max_tokens: 1000,
        messages,
        system: None,
        tools,
        tool_choice: None,
        stream: true,
        thinking,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    }
}

#[test]
fn builds_responses_shape_with_instructions_and_store_false() {
    let body = build_responses_request(
        &request(vec![InputMessage::user_text("hi")], None, None),
        "be helpful",
        true,
    );
    assert_eq!(body["model"], json!("gpt-5.6-sol"));
    assert_eq!(body["instructions"], json!("be helpful"));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    // The Codex backend rejects any request-side output-token cap with a 400
    // `Unsupported parameter: max_output_tokens`, so neither the Responses name
    // nor the Chat Completions name may appear in the payload.
    assert!(
        body.get("max_output_tokens").is_none(),
        "Codex backend rejects max_output_tokens with 400; it must not be sent"
    );
    assert!(
        body.get("max_completion_tokens").is_none(),
        "Codex backend takes no output-token cap; max_completion_tokens is Chat Completions only"
    );
    assert!(
        body.get("max_tokens").is_none(),
        "Codex backend takes no output-token cap; max_tokens must not be sent"
    );
    assert_eq!(body["input"][0]["type"], json!("message"));
    assert_eq!(body["input"][0]["role"], json!("user"));
    assert_eq!(body["input"][0]["content"][0]["type"], json!("input_text"));
    assert_eq!(body["input"][0]["content"][0]["text"], json!("hi"));
    let key = body["prompt_cache_key"]
        .as_str()
        .expect("prompt cache key should be present");
    assert!(key.starts_with("zo-"));
    assert!(key.len() <= 64);
}

#[test]
fn gpt55_responses_request_uses_cache_key_without_unsupported_retention() {
    let body = build_responses_request(&request_with_model("gpt-5.5", None), "i", true);
    assert!(body.get("prompt_cache_key").is_some());
    assert!(body.get("prompt_cache_retention").is_none());
}

#[test]
fn prompt_cache_key_uses_stable_system_prefix_not_dynamic_tail() {
    let mut base = request_with_model("gpt-5.5", None);
    base.system = Some(vec![
        SystemBlock::text("stable identity"),
        SystemBlock::text("dynamic cwd /tmp/a"),
    ]);
    let mut changed_tail = base.clone();
    changed_tail.system = Some(vec![
        SystemBlock::text("stable identity"),
        SystemBlock::text("dynamic cwd /tmp/b"),
    ]);
    let mut changed_prefix = base.clone();
    changed_prefix.system = Some(vec![
        SystemBlock::text("different identity"),
        SystemBlock::text("dynamic cwd /tmp/a"),
    ]);

    let base_key = build_responses_request(&base, "i", true)["prompt_cache_key"].clone();
    let tail_key = build_responses_request(&changed_tail, "i", true)["prompt_cache_key"].clone();
    let prefix_key =
        build_responses_request(&changed_prefix, "i", true)["prompt_cache_key"].clone();

    assert_eq!(
        base_key, tail_key,
        "session-specific system tail must not fragment the GPT prompt cache key"
    );
    assert_ne!(
        base_key, prefix_key,
        "stable system prefix must still participate in the GPT prompt cache key"
    );
}

#[test]
fn prompt_cache_key_is_stable_within_session() {
    let first = request_with_model("gpt-5.6-sol", None);
    let mut next = first.clone();
    next.messages.push(InputMessage::user_text("next turn"));

    let first_key = build_responses_request_for_session(&first, "i", true, "session-a")
        ["prompt_cache_key"]
        .clone();
    let next_key = build_responses_request_for_session(&next, "i", true, "session-a")
        ["prompt_cache_key"]
        .clone();

    assert_eq!(first_key, next_key);
}

#[test]
fn prompt_cache_key_differs_across_sessions() {
    let request = request_with_model("gpt-5.6-sol", None);
    let first = build_responses_request_for_session(&request, "i", true, "session-a")
        ["prompt_cache_key"]
        .clone();
    let second = build_responses_request_for_session(&request, "i", true, "session-b")
        ["prompt_cache_key"]
        .clone();

    assert_ne!(first, second);
}

#[test]
fn empty_session_prompt_cache_key_matches_wrapper_key() {
    let request = request_with_model("gpt-5.6-sol", None);
    let wrapper_key = build_responses_request(&request, "i", true)["prompt_cache_key"].clone();
    let empty_session_key = build_responses_request_for_session(&request, "i", true, "")
        ["prompt_cache_key"]
        .clone();

    assert_eq!(empty_session_key, wrapper_key);
}

/// A rebuilt client (fresh random wire session id) with the same pinned cache
/// scope must resolve the same scope — model swaps and OAuth rotations
/// rebuild the client, and the provider cache key must not roll with them.
#[test]
fn cache_scope_pins_across_client_rebuilds() {
    let first = super::ChatGptBackendClient::new("t", None).with_cache_scope("session-1");
    let rebuilt = super::ChatGptBackendClient::new("t", None).with_cache_scope("session-1");
    assert_eq!(first.cache_scope(), "session-1");
    assert_eq!(first.cache_scope(), rebuilt.cache_scope());
    assert_eq!(first.pinned_cache_scope(), Some("session-1"));

    // Unpinned clients fall back to their per-instance wire session id.
    let bare = super::ChatGptBackendClient::new("t", None);
    assert_eq!(bare.pinned_cache_scope(), None);
    assert_eq!(bare.cache_scope(), bare.session_id);

    // An empty scope is "no scope", not a shared "" bucket.
    let empty = super::ChatGptBackendClient::new("t", None).with_cache_scope("");
    assert_eq!(empty.pinned_cache_scope(), None);
}

/// Two conversation streams sharing one session id (a fanout spawn re-stamps
/// its requests with the parent's session id) must land on distinct cache
/// keys, keyed by their distinct opening user messages — otherwise every
/// concurrent agent competes for one provider cache shard and evicts the
/// others' prefixes (observed live 07-20: sol cache reads pinned at the ~12k
/// shared system prefix across 400+ interleaved spawn requests).
#[test]
fn prompt_cache_key_differs_across_conversation_streams_in_one_session() {
    let main = request_with_model("gpt-5.6-sol", None);
    let mut spawn = main.clone();
    spawn.messages = vec![InputMessage::user_text("spawn task: audit crates/api")];

    let main_key = build_responses_request_for_session(&main, "i", true, "session-a")
        ["prompt_cache_key"]
        .clone();
    let spawn_key = build_responses_request_for_session(&spawn, "i", true, "session-a")
        ["prompt_cache_key"]
        .clone();

    assert_ne!(main_key, spawn_key);
}

#[test]
fn supported_responses_models_skip_unverified_prompt_cache_retention() {
    for model in [
        "gpt-5.5",
        "gpt-5.5-fast",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.3-codex-spark",
    ] {
        let body = build_responses_request(&request_with_model(model, None), "i", true);
        assert!(body.get("max_output_tokens").is_none(), "{model}");
        assert!(body.get("max_completion_tokens").is_none(), "{model}");
        assert!(body.get("prompt_cache_key").is_some(), "{model}");
        assert!(body.get("prompt_cache_retention").is_none(), "{model}");
    }
}

#[test]
fn preserves_user_image_blocks_in_responses_input() {
    let body = build_responses_request(
        &request(
            vec![InputMessage::user_with_images(
                "what is in this image?",
                vec![ImageSource {
                    kind: "base64".into(),
                    media_type: "image/png".into(),
                    data: "abc123".into(),
                }],
            )],
            None,
            None,
        ),
        "i",
        false,
    );

    assert_eq!(body["input"][0]["type"], json!("message"));
    assert_eq!(body["input"][0]["role"], json!("user"));
    assert_eq!(body["input"][0]["content"][0]["type"], json!("input_image"));
    assert_eq!(
        body["input"][0]["content"][0]["image_url"],
        json!("data:image/png;base64,abc123")
    );
    assert_eq!(body["input"][0]["content"][0]["detail"], json!("auto"));
    assert_eq!(body["input"][0]["content"][1]["type"], json!("input_text"));
    assert_eq!(
        body["input"][0]["content"][1]["text"],
        json!("what is in this image?")
    );
}

#[test]
fn translates_tool_use_and_result_to_function_call_items() {
    let assistant = InputMessage {
        role: "assistant".into(),
        content: vec![InputContentBlock::ToolUse {
            id: "call_1".into(),
            name: "read".into(),
            input: json!({ "path": "x" }),
                    cache_control: None,
        }],
        thought_signature: None,
        reasoning_replay: None,
    };
    let user_result = InputMessage {
        role: "user".into(),
        content: vec![InputContentBlock::ToolResult {
            tool_use_id: "call_1".into(),
            content: vec![crate::types::ToolResultContentBlock::Text {
                text: "data".into(),
            }],
            is_error: false,
                    cache_control: None,
        }],
        thought_signature: None,
        reasoning_replay: None,
    };
    let body = build_responses_request(
        &request(vec![assistant, user_result], None, None),
        "i",
        false,
    );
    assert_eq!(body["input"][0]["type"], json!("function_call"));
    assert_eq!(body["input"][0]["call_id"], json!("call_1"));
    assert_eq!(body["input"][0]["name"], json!("read"));
    assert_eq!(body["input"][0]["arguments"], json!("{\"path\":\"x\"}"));
    assert_eq!(body["input"][1]["type"], json!("function_call_output"));
    assert_eq!(body["input"][1]["call_id"], json!("call_1"));
    assert_eq!(body["input"][1]["output"], json!("data"));
}

// end-to-end responses-ordering test; body exceeds the 100-line lint threshold
#[allow(clippy::too_many_lines)]
#[test]
fn mixed_assistant_text_tool_calls_and_outputs_keep_responses_order() {
    cache_reasoning_for_call(
        "",
        "call_order_b",
        vec![json!({
            "type": "reasoning",
            "id": "rs_order_b",
            "encrypted_content": "OPAQUE-ORDER-B"
        })],
    );
    let body = build_responses_request(
        &request(
            vec![
                InputMessage {
                    role: "assistant".into(),
                    content: vec![
                        InputContentBlock::Text {
                            text: "I'll inspect then search.".into(),
                            cache_control: None,
                        },
                        InputContentBlock::ToolUse {
                            id: "call_order_a".into(),
                            name: "read_file".into(),
                            input: json!({"path": "a.rs"}),
                                                    cache_control: None,
                        },
                        InputContentBlock::ToolUse {
                            id: "call_order_b".into(),
                            name: "grep_search".into(),
                            input: json!({"pattern": "TODO"}),
                                                    cache_control: None,
                        },
                    ],
                    thought_signature: None,
                    reasoning_replay: None,
                },
                InputMessage {
                    role: "user".into(),
                    content: vec![InputContentBlock::ToolResult {
                        tool_use_id: "call_order_a".into(),
                        content: vec![crate::types::ToolResultContentBlock::Text {
                            text: "file text".into(),
                        }],
                        is_error: false,
                                            cache_control: None,
                    }],
                    thought_signature: None,
                    reasoning_replay: None,
                },
                InputMessage {
                    role: "user".into(),
                    content: vec![InputContentBlock::ToolResult {
                        tool_use_id: "call_order_b".into(),
                        content: vec![
                            crate::types::ToolResultContentBlock::Json {
                                value: json!({"matches": 2}),
                            },
                            crate::types::ToolResultContentBlock::Image {
                                source: ImageSource {
                                    kind: "base64".into(),
                                    media_type: "image/png".into(),
                                    data: "abc123".into(),
                                },
                            },
                        ],
                        is_error: true,
                                            cache_control: None,
                    }],
                    thought_signature: None,
                    reasoning_replay: None,
                },
            ],
            None,
            None,
        ),
        "i",
        false,
    );

    let input = body["input"].as_array().expect("input array");
    assert_eq!(input[0]["type"], json!("message"));
    assert_eq!(input[0]["role"], json!("assistant"));
    assert_eq!(
        input[0]["content"][0]["text"],
        json!("I'll inspect then search.")
    );
    assert_eq!(input[1]["type"], json!("function_call"));
    assert_eq!(input[1]["call_id"], json!("call_order_a"));
    assert_eq!(input[1]["name"], json!("read_file"));
    assert_eq!(input[1]["arguments"], json!("{\"path\":\"a.rs\"}"));
    assert_eq!(input[2]["type"], json!("reasoning"));
    assert_eq!(input[2]["encrypted_content"], json!("OPAQUE-ORDER-B"));
    assert_eq!(input[3]["type"], json!("function_call"));
    assert_eq!(input[3]["call_id"], json!("call_order_b"));
    assert_eq!(input[3]["arguments"], json!("{\"pattern\":\"TODO\"}"));
    assert_eq!(input[4]["type"], json!("function_call_output"));
    assert_eq!(input[4]["call_id"], json!("call_order_a"));
    assert_eq!(input[4]["output"], json!("file text"));
    assert_eq!(input[5]["type"], json!("function_call_output"));
    assert_eq!(input[5]["call_id"], json!("call_order_b"));
    assert_eq!(
        input[5]["output"],
        json!("{\"matches\":2}\n[image image/png]")
    );

    remove_reasoning_for_call("", "call_order_b");
}

#[test]
fn tools_and_reasoning_attached() {
    let body = build_responses_request(
        &request(
            vec![InputMessage::user_text("hi")],
            Some(vec![ToolDefinition {
                name: "read".into(),
                description: Some("Read a file".into()),
                input_schema: json!({ "type": "object" }),
            }]),
            Some(ThinkingConfig::enabled(8000)),
        ),
        "i",
        true,
    );
    assert_eq!(body["tools"][0]["type"], json!("function"));
    assert_eq!(body["tools"][0]["name"], json!("read"));
    assert_eq!(body["tool_choice"], json!("auto"));
    assert_eq!(body["reasoning"]["effort"], json!("high"));
    assert_eq!(body["reasoning"]["summary"], json!("auto"));
}

#[test]
fn forced_tool_choice_is_honored_in_responses_request() {
    let tools = Some(vec![ToolDefinition {
        name: "StructuredOutput".into(),
        description: Some("emit structured output".into()),
        input_schema: json!({ "type": "object" }),
    }]);
    // A workflow sub-agent forcing a named function must reach the Responses
    // API in its flat `{type,name}` form, not the silently-weakened "auto"
    // (BUG-R16).
    let mut req = request(vec![InputMessage::user_text("hi")], tools.clone(), None);
    req.tool_choice = Some(ToolChoice::Tool {
        name: "StructuredOutput".into(),
    });
    let body = build_responses_request(&req, "i", true);
    assert_eq!(
        body["tool_choice"],
        json!({ "type": "function", "name": "StructuredOutput" })
    );

    // `Any` maps to the Responses "required" mode; absent tool_choice stays "auto".
    let mut req_any = request(vec![InputMessage::user_text("hi")], tools, None);
    req_any.tool_choice = Some(ToolChoice::Any);
    let body_any = build_responses_request(&req_any, "i", true);
    assert_eq!(body_any["tool_choice"], json!("required"));
}

fn request_with_model(model: &str, budget: Option<u32>) -> MessageRequest {
    MessageRequest {
        model: model.into(),
        max_tokens: 1000,
        messages: vec![InputMessage::user_text("hi")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: true,
        thinking: budget.map(ThinkingConfig::enabled),
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    }
}

#[test]
fn explicit_request_effort_drives_responses_reasoning_effort() {
    // zo_gpt's headless wire: build_message_request derives request.effort
    // from ZO_EFFORT and the Responses backend must serialize it as
    // reasoning.effort. Max/Ultra are internal Zo tiers; every GPT family
    // projects them to the provider-supported xhigh wire ceiling. Distinct from
    // the budget path the other tests cover.
    for (model, level, expected) in [
        ("gpt-5.5", crate::types::EffortLevel::Low, "low"),
        ("gpt-5.5", crate::types::EffortLevel::Medium, "medium"),
        ("gpt-5.5", crate::types::EffortLevel::High, "high"),
        ("gpt-5.5", crate::types::EffortLevel::Xhigh, "xhigh"),
        ("gpt-5.5", crate::types::EffortLevel::Max, "xhigh"),
        ("gpt-5.3-codex-spark", crate::types::EffortLevel::Max, "xhigh"),
        ("gpt-5.6-sol", crate::types::EffortLevel::Max, "xhigh"),
        ("gpt-5.6-terra", crate::types::EffortLevel::Max, "xhigh"),
        ("gpt-5.6-luna", crate::types::EffortLevel::Max, "xhigh"),
        ("gpt-5.6-sol", crate::types::EffortLevel::Ultra, "xhigh"),
        ("gpt-5.6-sol-2026-07-09", crate::types::EffortLevel::Ultra, "xhigh"),
        ("gpt-5.6-terra@openai", crate::types::EffortLevel::Ultra, "xhigh"),
        ("gpt-5.6-luna", crate::types::EffortLevel::Ultra, "xhigh"),
        ("gpt-5.5", crate::types::EffortLevel::Ultra, "xhigh"),
    ] {
        let mut req = request_with_model(model, None);
        req.effort = Some(level);
        let body = build_responses_request(&req, "i", true);
        assert_eq!(
            body["reasoning"]["effort"],
            json!(expected),
            "{model} request.effort {level:?} should reach reasoning.effort"
        );
        assert!(body.get("max_output_tokens").is_none(), "{model}");
    }
}

/// The luna effort contract: `gpt-5.6-luna` always ships `xhigh` or above,
/// no matter what the band/dynamic/de-escalation pipeline resolved. Applied
/// last in `build_responses_request_for_session` (see `luna_effort_floor`).
#[test]
fn luna_effort_floors_at_xhigh_regardless_of_resolved_tier() {
    // Explicit low is lifted to the floor.
    let mut req = request_with_model("gpt-5.6-luna", None);
    req.effort = Some(crate::types::EffortLevel::Low);
    let body = build_responses_request(&req, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"), "explicit low");

    // Auto (trivial prompt → dynamic `low` on other families) is lifted too.
    let body = build_responses_request(&request_with_model("gpt-5.6-luna", None), "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"), "auto/dynamic");

    // The empty-retry de-escalation must not step luna below the floor.
    let mut retry = request_with_model("gpt-5.6-luna", None);
    retry.system = Some(vec![crate::types::SystemBlock::text(
        "[zo:empty-response-retry] retry",
    )]);
    let body = build_responses_request(&retry, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"), "de-escalation");

    // Control: the floor is luna-scoped — sol's trivial-prompt dynamic effort
    // stays low.
    let body = build_responses_request(&request_with_model("gpt-5.6-sol", None), "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("low"), "sol control");
}

/// The Codex backend gates some models on the client User-Agent fingerprint:
/// `gpt-5.6-luna` answers 404 "Model not found" unless the UA carries the
/// `codex_cli_rs` product token (verified live 2026-07-13 — same token and
/// body, only the UA differing). Every request must therefore carry
/// [`super::USER_AGENT`].
#[test]
fn requests_carry_the_codex_user_agent_fingerprint() {
    let client = super::ChatGptBackendClient::new("token", Some("acct".to_string()));
    let request = client
        .apply_headers(
            reqwest::Client::new().post("https://chatgpt.com/backend-api/codex/responses"),
        )
        .build()
        .expect("request should build");
    let ua = request
        .headers()
        .get("user-agent")
        .and_then(|value| value.to_str().ok())
        .expect("user-agent header present");
    assert!(
        ua.starts_with("codex_cli_rs/"),
        "the luna model gate requires the codex_cli_rs product token, got: {ua}"
    );
    assert_eq!(ua, super::USER_AGENT);
}

#[test]
fn explicit_ultra_stays_at_the_wire_ceiling_under_empty_response_retry() {
    let mut req = request_with_model("gpt-5.6-sol", None);
    req.effort = Some(crate::types::EffortLevel::Ultra);
    req.system = Some(vec![crate::types::SystemBlock::text(
        "[zo:empty-response-retry] retry",
    )]);
    let body = build_responses_request(&req, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
}

#[test]
fn gpt56_sol_projects_internal_top_efforts_to_the_supported_wire_ceiling() {
    // Max and Ultra are Zo-side tiers. The OpenAI endpoint accepts xhigh as
    // its highest wire value, so neither an explicit tier nor Smart's resolved
    // rung may leak a Zo-only value into the request payload.
    for level in [
        crate::types::EffortLevel::Ultra,
        crate::types::EffortLevel::Max,
    ] {
        let mut req = request_with_model("gpt-5.6-sol", None);
        req.effort = Some(level);
        let body = build_responses_request(&req, "i", true);
        assert_eq!(body["reasoning"]["effort"], json!("xhigh"), "{level:?}");
    }

    let long_ask = "word ".repeat(150);
    let smart = banded_request(
        "gpt-5.6-sol",
        &format!("please refactor this module. {long_ask}"),
    );
    let body = build_responses_request(&smart, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"), "Smart");
}

#[test]
fn model_normalizes_to_family_and_scales_effort_without_budget() {
    let body = build_responses_request(&request_with_model("gpt-5.5", None), "i", true);
    assert_eq!(body["model"], json!("gpt-5.5"));
    // No budget, not fast: effort is scaled to the task (the trivial "hi"
    // probe → `low`) instead of being forced to `xhigh`, and the priority
    // service tier is not requested.
    assert_eq!(body["reasoning"]["effort"], json!("low"));
    assert!(body.get("service_tier").is_none());

    let dated = build_responses_request(&request_with_model("gpt-5.5-2026-04-23", None), "i", true);
    assert_eq!(dated["model"], json!("gpt-5.5"));
}

#[test]
fn fast_variant_scales_effort_and_requests_priority_tier() {
    // gpt-5.5-fast is not a separate model: same gpt-5.5 family with effort
    // scaled to the task (trivial "hi" → `low`), but "/fast on" still
    // requests the priority service tier for ~1.5x faster serving — the two
    // controls are independent.
    let body = build_responses_request(&request_with_model("gpt-5.5-fast", None), "i", true);
    assert_eq!(body["model"], json!("gpt-5.5"));
    assert_eq!(body["reasoning"]["effort"], json!("low"));
    assert_eq!(body["service_tier"], json!("priority"));
}

#[test]
fn fast_mode_is_independent_of_reasoning_effort() {
    // Fast mode adds the priority tier without overriding a configured
    // reasoning budget — the two controls compose.
    let body = build_responses_request(&request_with_model("gpt-5.5-fast", Some(8_000)), "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("high"));
    assert_eq!(body["service_tier"], json!("priority"));
}

#[test]
fn gpt56_bracket_fast_suffix_requests_priority_tier_and_strips_to_bare_family() {
    // GPT-5.6's `/fast` toggle spells "fast" with a `[fast]` service-tier
    // suffix (not the legacy bare `-fast` alias) — this must ALSO reach the
    // wire as `service_tier: "priority"`, with the suffix stripped from the
    // model id sent to the backend (the bracket is a zo-side convention,
    // not something the Codex Responses API understands as part of a model id).
    let body = build_responses_request(&request_with_model("gpt-5.6-terra[fast]", None), "i", true);
    assert_eq!(body["model"], json!("gpt-5.6-terra"));
    assert_eq!(body["service_tier"], json!("priority"));

    let luna = build_responses_request(&request_with_model("gpt-5.6-luna[fast]", None), "i", true);
    assert_eq!(luna["model"], json!("gpt-5.6-luna"));
    assert_eq!(luna["service_tier"], json!("priority"));
}

#[test]
fn unregistered_dash_fast_suffix_is_not_treated_as_priority() {
    // Only the `[fast]` bracket convention (or the legacy bare `gpt-5.5-fast`
    // alias) triggers priority serving — an arbitrary `-fast`-suffixed id that
    // is not a registered alias must not be misread as one (mirrors the
    // existing Codex Spark `-fast` regression pin).
    let body = build_responses_request(&request_with_model("gpt-5.6-terra-fast", None), "i", true);
    assert!(body.get("service_tier").is_none());
}

#[test]
fn fast_variant_keeps_xhigh_and_clamps_max_like_legacy_gpt() {
    // `/fast` maps to priority serving (`service_tier`), not a lower reasoning
    // ceiling. It keeps Xhigh, but legacy GPT still clamps Max -> xhigh.
    for (level, expected) in [
        (crate::types::EffortLevel::Xhigh, "xhigh"),
        (crate::types::EffortLevel::Max, "xhigh"),
    ] {
        let mut req = request_with_model("gpt-5.5-fast", None);
        req.effort = Some(level);
        let body = build_responses_request(&req, "i", true);
        assert_eq!(
            body["reasoning"]["effort"],
            json!(expected),
            "fast variant should project {level:?} to {expected}"
        );
        assert_eq!(body["service_tier"], json!("priority"));
    }

    let body =
        build_responses_request(&request_with_model("gpt-5.5-fast", Some(30_000)), "i", true);
    assert_eq!(
        body["reasoning"]["effort"],
        json!("xhigh"),
        "fast variant must clamp a max-budget to xhigh"
    );

    let mut req = request_with_model("gpt-5.5", None);
    req.effort = Some(crate::types::EffortLevel::Xhigh);
    assert_eq!(
        build_responses_request(&req, "i", true)["reasoning"]["effort"],
        json!("xhigh"),
        "non-fast gpt-5.5 keeps xhigh"
    );
    assert_eq!(
        build_responses_request(&request_with_model("gpt-5.5", Some(30_000)), "i", true)["reasoning"]
            ["effort"],
        json!("xhigh"),
        "non-fast gpt-5.5 budget clamps max to xhigh"
    );
}

#[test]
fn codex_spark_is_not_fast_priority_and_keeps_top_effort() {
    // This is only a Codex regression pin: do not infer a production xhigh
    // ceiling from the `codex` token, and do not apply GPT `/fast` service tier.
    let mut req = request_with_model("gpt-5.3-codex-spark", None);
    req.effort = Some(crate::types::EffortLevel::Xhigh);
    let body = build_responses_request(&req, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert!(body.get("service_tier").is_none());

    let body = build_responses_request(
        &request_with_model("gpt-5.3-codex-spark", Some(30_000)),
        "i",
        true,
    );
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert!(body.get("service_tier").is_none());

    let body = build_responses_request(
        &request_with_model("gpt-5.3-codex-spark-fast", Some(30_000)),
        "i",
        true,
    );
    assert_eq!(body["model"], json!("gpt-5.3-codex-spark"));
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert!(
        body.get("service_tier").is_none(),
        "do not infer priority from an unregistered Codex -fast suffix"
    );
}

#[test]
fn budget_drives_effort_for_non_fast() {
    let body = build_responses_request(&request_with_model("gpt-5.5", Some(8_000)), "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("high"));
}

#[test]
fn reasoning_effort_tiers() {
    assert_eq!(reasoning_effort(1_000), "low");
    assert_eq!(reasoning_effort(4_000), "medium");
    assert_eq!(reasoning_effort(10_000), "high");
    assert_eq!(reasoning_effort(16_000), "xhigh");
    assert_eq!(reasoning_effort(20_000), "xhigh");
    assert_eq!(reasoning_effort(20_001), "xhigh");
    assert_eq!(reasoning_effort(24_000), "xhigh");
    assert_eq!(reasoning_effort(30_000), "xhigh");
    assert_eq!(reasoning_effort(32_000), "xhigh");

    let mut req = request_with_model("gpt-5.6-sol", Some(24_000));
    assert_eq!(build_responses_request(&req, "i", true)["reasoning"]["effort"], json!("xhigh"));
    req.model = "gpt-5.6-terra".to_string();
    assert_eq!(build_responses_request(&req, "i", true)["reasoning"]["effort"], json!("xhigh"));
    req.model = "gpt-5.6-luna".to_string();
    assert_eq!(build_responses_request(&req, "i", true)["reasoning"]["effort"], json!("xhigh"));
}

#[test]
fn terminal_stream_recovery_deliberately_deescalates_ultra() {
    let mut req = request_with_model("gpt-5.6-sol", None);
    req.effort = Some(crate::types::EffortLevel::Ultra);
    let recovered = deescalated_recovery_request(&req);
    assert!(!recovered.stream);
    assert_eq!(recovered.effort, Some(crate::types::EffortLevel::High));
    assert!(recovered.thinking.is_none());
}

#[test]
fn terminal_stream_recovery_clears_the_band_ceiling_so_it_cannot_re_escalate() {
    // A banded (Smart-mode) request's floor gets forced down to a static
    // High above; if `effort_band_ceiling` survived the clone, the wire seam
    // would run `resolve_effort_band` again and could re-escalate a heavy
    // turn right back past the deliberate de-escalation.
    let req = banded_request("gpt-5.6-sol", "please refactor this module");
    let recovered = deescalated_recovery_request(&req);
    assert_eq!(recovered.effort, Some(crate::types::EffortLevel::High));
    assert_eq!(recovered.effort_band_ceiling, None);
    let body = build_responses_request(&recovered, "i", false);
    assert_eq!(body["reasoning"]["effort"], json!("high"));
}

#[test]
fn dynamic_effort_scales_to_task() {
    // Trivial single-line ask → low (the default "hi" probe).
    assert_eq!(dynamic_effort(&request_with_model("gpt-5.5", None)), "low");

    // Heavy-reasoning intent keyword (KO) → high even when short.
    let heavy_ko = MessageRequest {
        messages: vec![InputMessage::user_text("이 코드베이스를 분석해줘")],
        ..request_with_model("gpt-5.5", None)
    };
    assert_eq!(dynamic_effort(&heavy_ko), "high");

    // Heavy-reasoning intent keyword (EN) → high.
    let heavy_en = MessageRequest {
        messages: vec![InputMessage::user_text("please refactor this module")],
        ..request_with_model("gpt-5.5", None)
    };
    assert_eq!(dynamic_effort(&heavy_en), "high");

    // Middling ask: no heavy keyword, 200+ chars, well under the large-
    // context threshold → medium.
    let middling_text = "word ".repeat(60);
    let middling = MessageRequest {
        messages: vec![InputMessage::user_text(middling_text.as_str())],
        ..request_with_model("gpt-5.5", None)
    };
    assert_eq!(dynamic_effort(&middling), "medium");
}

#[test]
fn dynamic_effort_heavy_intent_projects_gpt56_top_tiers_to_xhigh() {
    // Sol/Terra select internal Ultra and Luna selects internal Max for a heavy
    // auto turn. The final OpenAI wire projection safely encodes all three as
    // xhigh; the pure band tests cover the distinct internal rungs.
    let heavy_sol = MessageRequest {
        messages: vec![InputMessage::user_text("please refactor this module")],
        ..request_with_model("gpt-5.6-sol", None)
    };
    assert_eq!(dynamic_effort(&heavy_sol), "xhigh");
    let heavy_terra = MessageRequest {
        messages: vec![InputMessage::user_text("분석해줘")],
        ..request_with_model("gpt-5.6-terra", None)
    };
    assert_eq!(dynamic_effort(&heavy_terra), "xhigh");

    let heavy_luna = MessageRequest {
        messages: vec![InputMessage::user_text("please refactor this module")],
        ..request_with_model("gpt-5.6-luna", None)
    };
    assert_eq!(dynamic_effort(&heavy_luna), "xhigh");

    // Legacy gpt-5.5's ceiling is Xhigh (`gpt_model_accepts_max` is false for
    // it), so a heavy-intent turn keeps the historical `high` cap — byte-
    // identical to before this change.
    let heavy_55 = MessageRequest {
        messages: vec![InputMessage::user_text("please refactor this module")],
        ..request_with_model("gpt-5.5", None)
    };
    assert_eq!(dynamic_effort(&heavy_55), "high");

    // Non-heavy-intent turns are unaffected regardless of model ceiling.
    assert_eq!(dynamic_effort(&request_with_model("gpt-5.6-sol", None)), "low");
}

fn banded_request(model: &str, text: &str) -> MessageRequest {
    MessageRequest {
        model: model.into(),
        max_tokens: 1000,
        messages: vec![InputMessage::user_text(text)],
        system: None,
        tools: None,
        tool_choice: None,
        stream: true,
        thinking: None,
        output_config: None,
        effort: Some(crate::types::EffortLevel::Xhigh),
        effort_band_ceiling: Some(crate::types::EffortLevel::Ultra),
    }
}

#[test]
fn banded_request_never_leaks_internal_top_rungs_after_gpt_projection() {
    // Smart mode: MessageRequest.effort carries the floor (Xhigh) and
    // effort_band_ceiling carries the ceiling (Ultra) — build_responses_request
    // resolves the band BEFORE gpt_for_model. Its internal rung varies by
    // difficulty, while every top rung has the same supported xhigh wire value.
    let trivial = banded_request("gpt-5.6-sol", "hi");
    let body = build_responses_request(&trivial, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));

    let one_signal = banded_request("gpt-5.6-sol", "please refactor this module");
    let body = build_responses_request(&one_signal, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));

    let long_ask = "word ".repeat(150);
    let two_signal = banded_request(
        "gpt-5.6-sol",
        &format!("please refactor this module. {long_ask}"),
    );
    let body = build_responses_request(&two_signal, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));

    // Luna's ceiling is Max, not Ultra — the two-signal rung clamps there.
    let two_signal_luna = banded_request(
        "gpt-5.6-luna",
        &format!("please refactor this module. {long_ask}"),
    );
    let body = build_responses_request(&two_signal_luna, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
}

#[test]
fn banded_request_stays_explicit_top_effort_protected_under_empty_response_pressure() {
    // The floor is Xhigh, so explicit_top_effort still fires for a banded
    // request (it IS a user top-mode contract) — an empty-response retry must
    // not de-escalate a band pick, even though it resolves per-request.
    let mut heavy = banded_request("gpt-5.6-sol", "please refactor this module");
    heavy.system = Some(crate::types::system_from_string(
        "[zo:empty-response-retry] previous attempt produced no output",
    ));
    let body = build_responses_request(&heavy, "i", true);
    assert_eq!(
        body["reasoning"]["effort"],
        json!("xhigh"),
        "band pick must not be de-escalated by empty-response retry pressure"
    );
}

#[test]
fn de_escalated_effort_steps_ultra_down_to_max_not_all_the_way_to_low() {
    // Defensive compatibility for any legacy/raw value reaching the helper.
    // Current GPT request builders project internal Ultra/Max to xhigh before
    // this point, so neither unsupported token is emitted on new requests.
    assert_eq!(de_escalated_effort("ultra"), "max");
    assert_eq!(de_escalated_effort("max"), "high");
    assert_eq!(de_escalated_effort("xhigh"), "medium");
    assert_eq!(de_escalated_effort("high"), "medium");
    assert_eq!(de_escalated_effort("medium"), "low");
}

#[test]
fn sse_parser_extracts_data_frames_and_skips_done() {
    let mut parser = ResponsesSseParser::new();
    let events = parser
        .push(
            b"event: response.created\ndata: {\"type\":\"response.created\"}\n\ndata: [DONE]\n\n",
        )
        .expect("well-formed frames parse");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["type"], json!("response.created"));
}

#[test]
fn ingest_maps_text_item_to_block_events() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    assert!(matches!(
        state.ingest(&json!({"type":"response.created","response":{"id":"resp_1"}}))[0],
        StreamEvent::MessageStart(_)
    ));
    assert!(matches!(
        state.ingest(&json!({
            "type":"response.output_item.added","output_index":0,
            "item":{"type":"message","role":"assistant"}
        }))[0],
        StreamEvent::ContentBlockStart(_)
    ));
    let delta = state.ingest(&json!({
        "type":"response.output_text.delta","output_index":0,"delta":"Hi"
    }));
    match &delta[0] {
        StreamEvent::ContentBlockDelta(event) => assert!(matches!(
            &event.delta,
            ContentBlockDelta::TextDelta { text } if text == "Hi"
        )),
        other => panic!("expected text delta, got {other:?}"),
    }
    assert!(matches!(
        state.ingest(&json!({"type":"response.output_item.done","output_index":0,"item":{}}))[0],
        StreamEvent::ContentBlockStop(_)
    ));
}

/// Codex CLI parity: reasoning items completed by `output_item.done` are
/// cached under the following `function_call`'s `call_id`, and the next
/// request's input replays them (encrypted content intact) immediately
/// before that call's `function_call` item. Without the replay, the
/// stateless Codex backend loses all reasoning continuity across turns —
/// the measured gpt-5.5 quality gap versus codex desktop.
#[test]
fn reasoning_items_round_trip_into_next_request() {
    let session_id = "session-rt1";
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string()).with_session_id(session_id);
    // A reasoning item completes, then the tool call it preceded.
    let _ = state.ingest(&json!({
        "type":"response.output_item.added","output_index":0,
        "item":{"type":"reasoning"}
    }));
    let _ = state.ingest(&json!({
        "type":"response.output_item.done","output_index":0,
        "item":{"type":"reasoning","id":"rs_1",
                "encrypted_content":"OPAQUE-BLOB",
                "summary":[]}
    }));
    let _ = state.ingest(&json!({
        "type":"response.output_item.added","output_index":1,
        "item":{"type":"function_call","call_id":"call_rt1","name":"read"}
    }));
    let _ = state.ingest(&json!({
        "type":"response.output_item.done","output_index":1,
        "item":{"type":"function_call","call_id":"call_rt1","name":"read",
                "arguments":"{}"}
    }));

    // The cache holds the full reasoning item for that call id.
    let cached =
        reasoning_for_call(session_id, "call_rt1").expect("reasoning cached for the call");
    assert_eq!(cached.len(), 1);
    assert_eq!(cached[0]["encrypted_content"], json!("OPAQUE-BLOB"));

    // Rebuilding the request from zo's provider-agnostic history (which
    // only carries the ToolUse) replays the reasoning item right before
    // the function_call input item.
    let request = request(
        vec![
            InputMessage::user_text("do the thing"),
            InputMessage {
                role: "assistant".into(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_rt1".into(),
                    name: "read".into(),
                    input: json!({"path":"x"}),
                                    cache_control: None,
                }],
                thought_signature: None,
                reasoning_replay: None,
            },
            InputMessage {
                role: "user".into(),
                content: vec![InputContentBlock::ToolResult {
                    tool_use_id: "call_rt1".into(),
                    content: vec![],
                    is_error: false,
                                    cache_control: None,
                }],
                thought_signature: None,
                reasoning_replay: None,
            },
        ],
        None,
        None,
    );
    let body = build_responses_request_for_session(&request, "i", true, session_id);
    let input = body["input"].as_array().expect("input array");
    let reasoning_pos = input
        .iter()
        .position(|item| item["type"] == json!("reasoning"))
        .expect("reasoning item replayed into input");
    let call_pos = input
        .iter()
        .position(|item| item["type"] == json!("function_call"))
        .expect("function_call present");
    assert_eq!(
        input[reasoning_pos]["encrypted_content"],
        json!("OPAQUE-BLOB")
    );
    assert!(
        reasoning_pos < call_pos,
        "reasoning must precede its function_call"
    );

    // An unknown call id replays nothing.
    assert!(reasoning_for_call(session_id, "call_unknown").is_none());
    // Re-recording the same id replaces, not duplicates.
    cache_reasoning_for_call(
        session_id,
        "call_rt1",
        vec![json!({"type":"reasoning","id":"rs_2"})],
    );
    let replaced = reasoning_for_call(session_id, "call_rt1").expect("still cached");
    assert_eq!(replaced.len(), 1);
    assert_eq!(replaced[0]["id"], json!("rs_2"));
}

/// Root-fix determinism: a message-attached `reasoning_replay` payload makes
/// `build_responses_request` reproducible regardless of the process-wide
/// cache's state — two calls with the same history produce byte-identical
/// requests, and growing an *unrelated* session's cache by 300 entries in
/// between changes nothing (the cache fallback is never consulted when the
/// attached field is present).
#[test]
fn deterministic_replay_from_attached_field_is_unaffected_by_other_session_cache_growth() {
    let assistant = InputMessage {
        role: "assistant".into(),
        content: vec![InputContentBlock::ToolUse {
            id: "call_det".into(),
            name: "read".into(),
            input: json!({"path": "x"}),
                    cache_control: None,
        }],
        thought_signature: None,
        reasoning_replay: Some(json!([
            {"call_id": "call_det", "items": [
                {"type": "reasoning", "id": "rs_det", "encrypted_content": "OPAQUE-DET"}
            ]}
        ])),
    };
    let req = request(
        vec![InputMessage::user_text("go"), assistant],
        None,
        None,
    );

    let body1 = build_responses_request_for_session(&req, "i", true, "session-det");
    let body2 = build_responses_request_for_session(&req, "i", true, "session-det");
    assert_eq!(body1, body2);
    assert_eq!(
        body1.to_string(),
        body2.to_string(),
        "identical history must produce byte-identical request JSON"
    );

    // Grow a completely unrelated session's cache well past its own cap.
    for i in 0..300 {
        cache_reasoning_for_call(
            "session-other",
            &format!("call_other_{i}"),
            vec![json!({"type": "reasoning", "id": format!("rs_other_{i}")})],
        );
    }

    let body3 = build_responses_request_for_session(&req, "i", true, "session-det");
    assert_eq!(
        body1.to_string(),
        body3.to_string(),
        "unrelated session cache growth must not perturb a request using the attached field"
    );
}

/// Build a history of `n` tool-calling assistant messages (each followed by
/// its tool result), every one carrying an attached `reasoning_replay`
/// payload, and return the wire `input` array.
fn wire_input_with_tool_calls(n: usize) -> Vec<serde_json::Value> {
    let mut messages = vec![InputMessage::user_text("go")];
    for i in 0..n {
        let call_id = format!("call_{i}");
        messages.push(InputMessage {
            role: "assistant".into(),
            content: vec![InputContentBlock::ToolUse {
                id: call_id.clone(),
                name: "read".into(),
                input: json!({}),
                cache_control: None,
            }],
            thought_signature: None,
            reasoning_replay: Some(json!([
                {"call_id": call_id, "items": [
                    {"type": "reasoning", "id": format!("rs_{i}")}
                ]}
            ])),
        });
        messages.push(InputMessage {
            role: "user".into(),
            content: vec![InputContentBlock::ToolResult {
                tool_use_id: call_id,
                content: vec![],
                is_error: false,
                cache_control: None,
            }],
            thought_signature: None,
            reasoning_replay: None,
        });
    }
    let req = request(messages, None, None);
    let body = build_responses_request(&req, "i", true);
    body["input"].as_array().expect("input array").clone()
}

/// Append-stability regression: the 17th tool call must NOT evict the 1st
/// tool call's reasoning from the wire. The original "most recent 16" window
/// slid on every appended tool call, mutating mid-history on every request —
/// which broke the provider prefix cache right after the system prompt and
/// re-billed the whole transcript per call (observed live: cache reads pinned
/// at ~10k while input grew past 200k).
#[test]
fn reasoning_replay_seventeenth_tool_call_keeps_the_first_replayed() {
    let input = wire_input_with_tool_calls(17);
    for i in 0..17 {
        assert!(
            input.iter().any(|item| item["id"] == json!(format!("rs_{i}"))),
            "every reasoning item must be replayed, missing rs_{i}: {input:?}"
        );
    }
}

/// Replay is append-only at any depth: 48 tool calls (past the old stride-16
/// staircase's two anchor jumps) must all keep their reasoning items, each
/// immediately before its `function_call`. The staircase's anchor advance
/// dropped the oldest stride's items in one step — a mid-history mutation
/// that re-billed the whole post-anchor suffix on every jump.
#[test]
fn reasoning_replay_replays_every_tool_call_at_any_depth() {
    let input = wire_input_with_tool_calls(48);
    for i in 0..48 {
        let call_id = json!(format!("call_{i}"));
        let call_pos = input
            .iter()
            .position(|item| item["call_id"] == call_id)
            .unwrap_or_else(|| panic!("call_{i} function_call present"));
        assert_eq!(
            input[call_pos - 1]["type"],
            json!("reasoning"),
            "call_{i} should be preceded by its reasoning item: {input:?}"
        );
        assert_eq!(input[call_pos - 1]["id"], json!(format!("rs_{i}")));
    }
}

/// The prefix-cache safety property stated at wire level: growing the history
/// by one tool call appends input items and changes nothing before them, so
/// the provider's prefix cache stays valid across every request of a
/// conversation. This is exactly the property the stride-16 staircase anchor
/// violated once per stride.
#[test]
fn reasoning_replay_wire_input_is_append_only_as_history_grows() {
    for n in [1, 15, 16, 17, 31, 32, 47, 48] {
        let shorter = wire_input_with_tool_calls(n);
        let longer = wire_input_with_tool_calls(n + 1);
        assert!(
            longer.len() > shorter.len(),
            "growing the history must append items (n={n})"
        );
        assert_eq!(
            longer[..shorter.len()],
            shorter[..],
            "history growth must never mutate earlier wire input (n={n})"
        );
    }
}

/// Session scoping (Stage A defense line): a session's cached reasoning
/// entry survives a different session pushing 300 entries of its own — the
/// old process-wide FIFO cache would have evicted it.
#[test]
fn session_scoped_cache_isolates_unrelated_sessions() {
    cache_reasoning_for_call(
        "session-a",
        "call_a",
        vec![json!({"type": "reasoning", "id": "rs_a"})],
    );

    for i in 0..300 {
        cache_reasoning_for_call(
            "session-b",
            &format!("call_b_{i}"),
            vec![json!({"type": "reasoning", "id": format!("rs_b_{i}")})],
        );
    }

    let cached = reasoning_for_call("session-a", "call_a")
        .expect("session A's entry must survive session B's 300 pushes");
    assert_eq!(cached[0]["id"], json!("rs_a"));
}

#[test]
fn reasoning_summary_parts_get_a_paragraph_separator() {
    use crate::types::{ContentBlockDelta, StreamEvent};

    let thinking_text = |events: &[StreamEvent]| -> Vec<String> {
        events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ContentBlockDelta(delta) => match &delta.delta {
                    ContentBlockDelta::ThinkingDelta { thinking } => Some(thinking.clone()),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    };

    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let _ = state.ingest(&json!({
        "type":"response.output_item.added","output_index":0,
        "item":{"type":"reasoning"}
    }));
    // Part 0 opens: no separator (nothing precedes it).
    let first_part = state.ingest(&json!({
        "type":"response.reasoning_summary_part.added","output_index":0,"summary_index":0
    }));
    assert!(thinking_text(&first_part).is_empty());
    let _ = state.ingest(&json!({
        "type":"response.reasoning_summary_text.delta","output_index":0,
        "delta":"First topic."
    }));
    // Part 1 opens: OpenAI sends no separator between summary parts, so the
    // adapter must inject a paragraph break — otherwise parts render as one
    // run-on paragraph and the TUI's rolling title freezes on part 0.
    let second_part = state.ingest(&json!({
        "type":"response.reasoning_summary_part.added","output_index":0,"summary_index":1
    }));
    assert_eq!(thinking_text(&second_part), vec!["\n\n".to_string()]);
    // A boundary must never phantom-start a reasoning block by itself.
    let mut fresh = ResponsesStreamState::new("gpt-5.5".to_string());
    let orphan = fresh.ingest(&json!({
        "type":"response.reasoning_summary_part.added","output_index":3,"summary_index":1
    }));
    assert!(orphan.is_empty());
}

#[test]
fn ingest_maps_function_call_and_args_delta() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    let added = state.ingest(&json!({
        "type":"response.output_item.added","output_index":1,
        "item":{"type":"function_call","call_id":"call_9","name":"read"}
    }));
    match &added[0] {
        StreamEvent::ContentBlockStart(event) => match &event.content_block {
            OutputContentBlock::ToolUse { id, name, .. } => {
                assert_eq!(id, "call_9");
                assert_eq!(name, "read");
            }
            other => panic!("expected tool use, got {other:?}"),
        },
        other => panic!("expected block start, got {other:?}"),
    }
    let args = state.ingest(&json!({
        "type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"p\":1}"
    }));
    match &args[0] {
        StreamEvent::ContentBlockDelta(event) => assert!(matches!(
            &event.delta,
            ContentBlockDelta::InputJsonDelta { partial_json } if partial_json == "{\"p\":1}"
        )),
        other => panic!("expected json delta, got {other:?}"),
    }
}

#[test]
fn done_events_supply_text_when_deltas_are_absent() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let events = state.ingest(&json!({
        "type":"response.output_text.done",
        "output_index":0,
        "text":"final text only"
    }));
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], StreamEvent::ContentBlockStart(_)));
    match &events[1] {
        StreamEvent::ContentBlockDelta(event) => assert!(matches!(
            &event.delta,
            ContentBlockDelta::TextDelta { text } if text == "final text only"
        )),
        other => panic!("expected text delta, got {other:?}"),
    }

    let duplicate = state.ingest(&json!({
        "type":"response.content_part.done",
        "output_index":0,
        "part":{"type":"text","text":"final text only"}
    }));
    assert!(
        duplicate.is_empty(),
        "done payloads must not duplicate prior text deltas"
    );
}

#[test]
fn function_call_arguments_done_supplies_args_without_delta() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let added = state.ingest(&json!({
        "type":"response.output_item.added","output_index":2,
        "item":{"type":"function_call","call_id":"call_done","name":"read"}
    }));
    assert!(matches!(added[0], StreamEvent::ContentBlockStart(_)));

    let args = state.ingest(&json!({
        "type":"response.function_call_arguments.done",
        "output_index":2,
        "call_id":"call_done",
        "name":"read",
        "arguments":"{\"path\":\"x\"}"
    }));
    assert_eq!(args.len(), 1);
    match &args[0] {
        StreamEvent::ContentBlockDelta(event) => assert!(matches!(
            &event.delta,
            ContentBlockDelta::InputJsonDelta { partial_json } if partial_json == "{\"path\":\"x\"}"
        )),
        other => panic!("expected input json delta, got {other:?}"),
    }

    let duplicate = state.ingest(&json!({
        "type":"response.output_item.done","output_index":2,
        "item":{"type":"function_call","call_id":"call_done","name":"read",
                "arguments":"{\"path\":\"x\"}"}
    }));
    assert_eq!(duplicate.len(), 1);
    assert!(matches!(
        duplicate[0],
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 2 })
    ));
}

#[test]
fn output_item_done_supplies_final_payloads_without_prior_done_events() {
    let mut text_state = ResponsesStreamState::new("gpt-5.5".to_string());
    let text = text_state.ingest(&json!({
        "type":"response.output_item.done","output_index":0,
        "item":{"type":"message","content":[
            {"type":"output_text","text":"hello"},
            {"type":"output_text","text":" world"}
        ]}
    }));
    assert_eq!(text.len(), 3);
    assert!(matches!(text[0], StreamEvent::ContentBlockStart(_)));
    match &text[1] {
        StreamEvent::ContentBlockDelta(event) => assert!(matches!(
            &event.delta,
            ContentBlockDelta::TextDelta { text } if text == "hello world"
        )),
        other => panic!("expected text delta, got {other:?}"),
    }
    assert!(matches!(
        text[2],
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 })
    ));

    let mut tool_state = ResponsesStreamState::new("gpt-5.5".to_string());
    let tool = tool_state.ingest(&json!({
        "type":"response.output_item.done","output_index":1,
        "item":{"type":"function_call","call_id":"call_item","name":"read",
                "arguments":"{\"path\":\"z\"}"}
    }));
    assert_eq!(tool.len(), 3);
    assert!(matches!(tool[0], StreamEvent::ContentBlockStart(_)));
    match &tool[1] {
        StreamEvent::ContentBlockDelta(event) => assert!(matches!(
            &event.delta,
            ContentBlockDelta::InputJsonDelta { partial_json } if partial_json == "{\"path\":\"z\"}"
        )),
        other => panic!("expected input json delta, got {other:?}"),
    }
    assert!(matches!(
        tool[2],
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 1 })
    ));
}

#[test]
fn completed_event_supplies_output_snapshot_without_prior_deltas() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let events = state.ingest(&json!({
        "type":"response.completed",
        "response":{
            "output":[
                {"type":"message","content":[
                    {"type":"output_text","text":"from completed"}
                ]},
                {"type":"function_call","call_id":"call_done","name":"read",
                 "arguments":"{\"path\":\"x\"}"}
            ],
            "usage":{"input_tokens":3,"output_tokens":4}
        }
    }));

    let text: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::TextDelta { text },
                ..
            }) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "from completed");

    let args: String = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::InputJsonDelta { partial_json },
                ..
            }) => Some(partial_json.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(args, "{\"path\":\"x\"}");
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 })
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 1 })
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageStop(_)))
    );
}

#[test]
fn completed_event_does_not_duplicate_prior_text_delta() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let mut events = Vec::new();
    events.extend(state.ingest(&json!({
        "type":"response.output_item.added",
        "output_index":0,
        "item":{"type":"message"}
    })));
    events.extend(state.ingest(&json!({
        "type":"response.output_text.delta",
        "output_index":0,
        "delta":"already streamed"
    })));
    events.extend(state.ingest(&json!({
        "type":"response.output_item.done",
        "output_index":0,
        "item":{"type":"message","content":[
            {"type":"output_text","text":"already streamed"}
        ]}
    })));
    events.extend(state.ingest(&json!({
        "type":"response.completed",
        "response":{
            "output":[{"type":"message","content":[
                {"type":"output_text","text":"already streamed"}
            ]}],
            "usage":{"input_tokens":1,"output_tokens":1}
        }
    })));

    let text_deltas = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                    delta: ContentBlockDelta::TextDelta { .. },
                    ..
                })
            )
        })
        .count();
    assert_eq!(text_deltas, 1);
}

#[test]
fn completed_emits_usage_then_stop() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    let events = state.ingest(&json!({
        "type":"response.completed",
        "response":{"usage":{"input_tokens":12,"output_tokens":7,
            "input_tokens_details":{"cached_tokens":5}}}
    }));
    assert_eq!(events.len(), 2);
    match &events[0] {
        StreamEvent::MessageDelta(event) => {
            assert_eq!(event.usage.input_tokens, 7);
            assert_eq!(event.usage.cache_read_input_tokens, 5);
            assert_eq!(event.usage.output_tokens, 7);
            assert_eq!(event.delta.stop_reason.as_deref(), Some("end_turn"));
        }
        other => panic!("expected message delta, got {other:?}"),
    }
    assert!(matches!(events[1], StreamEvent::MessageStop(_)));
    assert!(
        state
            .ingest(&json!({"type":"response.completed","response":{}}))
            .is_empty()
    );
}

#[test]
fn completed_closes_open_text_item_before_message_stop() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    assert!(matches!(
        state.ingest(&json!({
            "type":"response.output_item.added","output_index":0,
            "item":{"type":"message","role":"assistant"}
        }))[0],
        StreamEvent::ContentBlockStart(_)
    ));
    assert!(matches!(
        state.ingest(&json!({
            "type":"response.output_text.delta","output_index":0,"delta":"tail"
        }))[0],
        StreamEvent::ContentBlockDelta(_)
    ));

    let events = state.ingest(&json!({
        "type":"response.completed",
        "response":{"usage":{"input_tokens":1,"output_tokens":1}}
    }));
    assert_eq!(events.len(), 3);
    assert!(matches!(
        events[0],
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 })
    ));
    assert!(matches!(events[1], StreamEvent::MessageDelta(_)));
    assert!(matches!(events[2], StreamEvent::MessageStop(_)));
}

#[test]
fn completed_closes_text_delta_even_when_item_done_is_missing() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    let delta = state.ingest(&json!({
        "type":"response.output_text.delta","output_index":0,"delta":"early"
    }));
    assert!(matches!(delta[0], StreamEvent::ContentBlockStart(_)));
    assert!(matches!(delta[1], StreamEvent::ContentBlockDelta(_)));

    let events = state.ingest(&json!({
        "type":"response.completed",
        "response":{"usage":{"input_tokens":1,"output_tokens":1}}
    }));
    assert!(matches!(
        events[0],
        StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 })
    ));
    assert!(matches!(events[1], StreamEvent::MessageDelta(_)));
    assert!(matches!(events[2], StreamEvent::MessageStop(_)));
}

#[test]
fn completed_ignores_late_item_done() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    assert_eq!(
        state
            .ingest(&json!({
                "type":"response.output_text.delta","output_index":0,"delta":"tail"
            }))
            .len(),
        2
    );
    assert_eq!(
        state
            .ingest(&json!({"type":"response.completed","response":{}}))
            .len(),
        3
    );
    assert!(
        state
            .ingest(&json!({"type":"response.output_item.done","output_index":0,"item":{}}))
            .is_empty()
    );
}

#[test]
fn text_delta_before_item_added_synthesizes_single_start() {
    let mut state = ResponsesStreamState::new("gpt-5.6-sol".to_string());
    let delta = state.ingest(&json!({
        "type":"response.output_text.delta","output_index":0,"delta":"early"
    }));
    assert!(matches!(delta[0], StreamEvent::ContentBlockStart(_)));
    assert!(matches!(delta[1], StreamEvent::ContentBlockDelta(_)));

    assert!(
        state
            .ingest(&json!({
                "type":"response.output_item.added","output_index":0,
                "item":{"type":"message","role":"assistant"}
            }))
            .is_empty()
    );
}

#[test]
fn client_constructs_and_joins_system_instructions() {
    let _client = super::ChatGptBackendClient::new("token", Some("acc_1".to_string()))
        .with_base_url("http://localhost/responses");
    let mut request = request_with_model("gpt-5.5", None);
    request.system = Some(crate::types::system_from_string("be terse"));
    let instructions = super::ChatGptBackendClient::instructions(&request);
    // The zo system body is preserved …
    assert!(
        instructions.contains("be terse"),
        "system body preserved: {instructions}"
    );
    // … under an explicit OpenAI model identity that overrides zo's
    // hardcoded Claude identity so gpt-5.5 doesn't introduce itself as Claude.
    assert!(
        instructions.contains("gpt-5.5") && instructions.contains("OpenAI"),
        "identity override present: {instructions}"
    );
}

#[test]
fn instructions_empty_system_stays_empty() {
    // No system blocks → no identity preamble, no batching contract
    // (a bare request stays bare).
    let request = request_with_model("gpt-5.5", None);
    assert_eq!(super::ChatGptBackendClient::instructions(&request), "");
}

/// The tool-call batching contract rides at the tail of the composed
/// instructions for every model on this backend: base-prompt prose alone left
/// GPT models at ~2.6 tool calls per tool-using message vs ~3.5 for Claude on
/// the same harness, and each unbatched call is a full extra round trip.
#[test]
fn instructions_append_tool_batching_contract() {
    let mut request = request_with_model("gpt-5.6-sol", None);
    request.system = Some(crate::types::system_from_string("be terse"));
    let instructions = super::ChatGptBackendClient::instructions(&request);
    assert!(
        instructions.ends_with(super::TOOL_BATCHING_CONTRACT),
        "batching contract must close the instructions: {instructions}"
    );
    // The contract must not displace the identity override from the head.
    assert!(
        instructions.starts_with("You are gpt-5.6"),
        "identity stays first: {instructions}"
    );
}

#[test]
fn parses_non_stream_output_to_message_response() {
    let value = json!({
        "id":"resp_9",
        "output":[
            {"type":"message","content":[{"type":"output_text","text":"hi"}]},
            {"type":"function_call","call_id":"c1","name":"read","arguments":"{\"p\":1}"}
        ],
        "usage":{"input_tokens":4,"output_tokens":2,
            "input_tokens_details":{"cached_tokens":3}}
    });
    let response = super::parse_responses_response(&value, "gpt-5.5", "");
    assert_eq!(response.id, "resp_9");
    assert_eq!(response.content.len(), 2);
    assert!(matches!(
        &response.content[0],
        OutputContentBlock::Text { text } if text == "hi"
    ));
    match &response.content[1] {
        OutputContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "c1");
            assert_eq!(name, "read");
            assert_eq!(input, &json!({ "p": 1 }));
        }
        other => panic!("expected tool use, got {other:?}"),
    }
    assert_eq!(response.usage.input_tokens, 1);
    assert_eq!(response.usage.cache_read_input_tokens, 3);
    assert_eq!(response.usage.output_tokens, 2);
}

/// The true non-streaming `send_message` path (`parse_responses_response`)
/// assembles `MessageResponse.reasoning_replay` from the same `output` array
/// walk the streaming completed-response path uses, and records the entry in
/// the session-scoped cache fallback too.
#[test]
fn parses_non_stream_output_populates_reasoning_replay_and_caches_it() {
    let value = json!({
        "id":"resp_10",
        "output":[
            {"type":"reasoning","id":"rs_ns","encrypted_content":"OPAQUE-NS"},
            {"type":"function_call","call_id":"c_ns","name":"read","arguments":"{}"}
        ],
        "usage":{"input_tokens":1,"output_tokens":1}
    });
    let response = super::parse_responses_response(&value, "gpt-5.5", "session-ns");
    let replay = response
        .reasoning_replay
        .expect("non-streaming response must carry the assembled reasoning replay");
    assert_eq!(
        replay,
        json!([{"call_id": "c_ns", "items": [
            {"type": "reasoning", "id": "rs_ns", "encrypted_content": "OPAQUE-NS"}
        ]}])
    );

    // The same entry lands in the session-scoped cache fallback.
    let cached = reasoning_for_call("session-ns", "c_ns").expect("cached under its session");
    assert_eq!(cached[0]["encrypted_content"], json!("OPAQUE-NS"));
}

#[test]
fn stream_idle_timeout_defaults_and_env_override() {
    let _guard = env_lock();
    let key = super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV;
    let restore = std::env::var(key).ok();

    std::env::remove_var(key);
    assert_eq!(
        super::stream_idle_timeout(),
        Some(std::time::Duration::from_millis(
            super::CHATGPT_STREAM_IDLE_TIMEOUT_MS
        )),
        "default budget applies when unset"
    );

    std::env::set_var(key, "1500");
    assert_eq!(
        super::stream_idle_timeout(),
        Some(std::time::Duration::from_millis(1_500)),
        "valid override is honoured"
    );

    std::env::set_var(key, "0");
    assert_eq!(
        super::stream_idle_timeout(),
        None,
        "zero disables the idle timeout"
    );

    std::env::set_var(key, "not-a-number");
    assert_eq!(
        super::stream_idle_timeout(),
        Some(std::time::Duration::from_millis(
            super::CHATGPT_STREAM_IDLE_TIMEOUT_MS
        )),
        "garbage falls back to the default"
    );

    let startup_key = super::CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_ENV;
    let startup_restore = std::env::var(startup_key).ok();
    std::env::remove_var(startup_key);
    assert_eq!(
        super::startup_no_progress_timeout(),
        Some(std::time::Duration::from_millis(
            super::CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_MS
        ))
    );
    std::env::set_var(startup_key, "250");
    assert_eq!(
        super::startup_no_progress_timeout(),
        Some(std::time::Duration::from_millis(250))
    );
    std::env::set_var(startup_key, "0");
    assert_eq!(super::startup_no_progress_timeout(), None);

    match restore {
        Some(value) => std::env::set_var(key, value),
        None => std::env::remove_var(key),
    }
    match startup_restore {
        Some(value) => std::env::set_var(startup_key, value),
        None => std::env::remove_var(startup_key),
    }
}

#[test]
fn startup_reasoning_extends_deadline_exactly_once() {
    let window = std::time::Duration::from_secs(240);
    let initial = std::time::Instant::now()
        .checked_add(window)
        .expect("initial deadline");
    let mut deadline = Some(initial);
    let mut extended = false;

    super::extend_startup_deadline_for_reasoning(
        &mut deadline,
        Some(window),
        &mut extended,
    );
    let once = deadline.expect("extended deadline");
    assert_eq!(once, initial.checked_add(window).unwrap());
    assert!(extended);

    super::extend_startup_deadline_for_reasoning(
        &mut deadline,
        Some(window),
        &mut extended,
    );
    assert_eq!(deadline, Some(once), "later reasoning must not extend again");
}

#[test]
fn terminal_failure_recovery_request_disables_stream_and_deescalates_xhigh() {
    let request = MessageRequest {
        model: "gpt-5.5".into(),
        messages: vec![InputMessage::user_text("hi")],
        effort: Some(EffortLevel::Xhigh),
        thinking: Some(ThinkingConfig::enabled(16_000)),
        stream: true,
        ..request(vec![], None, None)
    };

    let recovered = super::deescalated_recovery_request(&request);

    assert!(!recovered.stream);
    assert_eq!(recovered.effort, Some(EffortLevel::High));
    assert!(recovered.thinking.is_none());
    assert!(recovered.output_config.is_none());
}

#[test]
fn terminal_failure_recovery_request_preserves_auto_effort() {
    let request = MessageRequest {
        model: "gpt-5.5".into(),
        messages: vec![InputMessage::user_text("hi")],
        effort: None,
        stream: true,
        ..request(vec![], None, None)
    };

    let recovered = super::deescalated_recovery_request(&request);

    assert!(!recovered.stream);
    assert_eq!(recovered.effort, None);
    assert!(recovered.thinking.is_none());
    assert!(recovered.output_config.is_none());
}

#[test]
fn restart_is_armed_only_before_commit_and_within_budget() {
    // Pre-commit, retryable, budget available → restart.
    assert!(should_restart(false, true, 0, 5));
    assert!(should_restart(false, true, 4, 5));

    // Committed (output already surfaced) → never restart, even on a
    // retryable fault with budget left. This is the duplicate-output guard.
    assert!(!should_restart(true, true, 0, 5));

    // Non-retryable fault (e.g. auth) → propagate regardless of arming.
    assert!(!should_restart(false, false, 0, 5));

    // Budget exhausted → stop. `attempts == max_retries` is already spent.
    assert!(!should_restart(false, true, 5, 5));
    assert!(!should_restart(false, true, 6, 5));

    // Zero budget → no transparent restart at all (idle timeout still
    // surfaces as a retryable error for an outer caller to handle).
    assert!(!should_restart(false, true, 0, 0));
}

#[test]
fn idle_timeout_error_drives_a_restart_decision() {
    // The exact error the stalled-stream path raises must be classified as
    // restart-eligible while the turn is still re-armable.
    let err = ApiError::stream_idle_timeout(std::time::Duration::from_secs(90));
    assert!(should_restart(false, err.is_retryable(), 0, 5));
    assert!(!should_restart(true, err.is_retryable(), 0, 5));
}

#[test]
fn restart_commit_boundary_ignores_reasoning_prefix() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let reasoning_prefix = [
        json!({"type":"response.created","response":{"id":"r1"}}),
        json!({
            "type":"response.output_item.added",
            "output_index":0,
            "item":{"type":"reasoning"}
        }),
        json!({
            "type":"response.reasoning_summary_text.delta",
            "output_index":0,
            "delta":"Assessing implementation quality"
        }),
    ];

    for value in reasoning_prefix {
        for event in state.ingest(&value) {
            assert!(
                !crosses_restart_commit_boundary(&event),
                "reasoning/bookkeeping event should remain replay-safe: {event:?}"
            );
        }
    }

    let text_events = state.ingest(&json!({
        "type":"response.output_text.delta",
        "output_index":1,
        "delta":"visible"
    }));
    assert!(
        text_events.iter().any(crosses_restart_commit_boundary),
        "visible answer text must lock out transparent restart"
    );
}

#[test]
fn backoff_grows_then_caps() {
    let client = super::ChatGptBackendClient::new("token", None).with_retry_policy(
        5,
        std::time::Duration::from_millis(500),
        std::time::Duration::from_secs(4),
    );
    // 500ms · 2^(n-1): 500, 1000, 2000, then capped at 4s. The method now
    // delegates to the shared `providers::backoff_for_attempt`, which returns a
    // `Result` (Err on shift overflow) — unified with the OpenAI/Anthropic path.
    assert_eq!(
        client.backoff_for_attempt(1).unwrap(),
        std::time::Duration::from_millis(500)
    );
    assert_eq!(
        client.backoff_for_attempt(2).unwrap(),
        std::time::Duration::from_secs(1)
    );
    assert_eq!(
        client.backoff_for_attempt(3).unwrap(),
        std::time::Duration::from_secs(2)
    );
    assert_eq!(
        client.backoff_for_attempt(4).unwrap(),
        std::time::Duration::from_secs(4)
    );
    // A far-out attempt overflows the doubling shift and surfaces a
    // BackoffOverflow error rather than saturating; production caps retries far
    // below this, so the `?` at the call site never trips it in practice.
    assert!(client.backoff_for_attempt(40).is_err());
}

/// A transport that sends only SSE comments is alive at the socket layer but
/// has made no model progress. Those keep-alives must not postpone the startup
/// deadline; the uncommitted request is safe to restart exactly once here.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn response_header_stall_is_bounded_before_stream_construction() {
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut scratch = [0u8; 1024];
        let _ = socket.read(&mut scratch).await;
        // Accept the request but never send an HTTP status line or headers.
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    });

    let idle_key = super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV;
    let startup_key = super::CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_ENV;
    let restore_idle = std::env::var(idle_key).ok();
    let restore_startup = std::env::var(startup_key).ok();
    std::env::set_var(idle_key, "100");
    std::env::set_var(startup_key, "1000");

    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"));
    let started = std::time::Instant::now();
    let error = client
        .stream_message(&request(vec![InputMessage::user_text("hi")], None, None))
        .await
        .expect_err("an unanswered HTTP open must time out");

    match restore_idle {
        Some(value) => std::env::set_var(idle_key, value),
        None => std::env::remove_var(idle_key),
    }
    match restore_startup {
        Some(value) => std::env::set_var(startup_key, value),
        None => std::env::remove_var(startup_key),
    }
    server.abort();

    assert!(
        error.to_string().contains("stream_idle_timeout"),
        "the shorter byte-idle budget should classify the header stall: {error}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_millis(800),
        "header timeout must fire without waiting for the server task"
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn keepalive_only_stream_restarts_at_startup_progress_deadline() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first.flush().await.unwrap();
        let keepalive_writer = tokio::spawn(async move {
            for _ in 0..40 {
                if first.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
                if first.flush().await.is_err() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        });

        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        let body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r2\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,",
            "\"delta\":\"recovered after keepalive stall\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":",
            "{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
        let _ = keepalive_writer.await;
    });

    let startup_key = super::CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_ENV;
    let restore = std::env::var(startup_key).ok();
    std::env::set_var(startup_key, "150");
    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            1,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(20),
        );
    let mut stream = client
        .stream_message(&request(vec![InputMessage::user_text("hi")], None, None))
        .await
        .expect("open stream");

    let mut text = String::new();
    while let Some(event) = stream.next_event().await.expect("restart should recover") {
        if let StreamEvent::ContentBlockDelta(delta) = event {
            if let ContentBlockDelta::TextDelta { text: chunk } = delta.delta {
                text.push_str(&chunk);
            }
        }
    }
    match restore {
        Some(value) => std::env::set_var(startup_key, value),
        None => std::env::remove_var(startup_key),
    }
    server.await.unwrap();

    assert_eq!(text, "recovered after keepalive stall");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// Reasoning is useful startup activity, but it is not a task action. The first
/// delta grants one extension; a backend that then streams reasoning forever
/// must still be restarted when the extended deadline expires. This exercises
/// the real `emitted == true` path rather than the keepalive-only branch above.
// end-to-end reasoning-restart test; body exceeds the 100-line lint threshold
#[allow(clippy::too_many_lines)]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn continuous_reasoning_restarts_after_single_startup_extension() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first
            .write_all(concat!(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
                "\"item\":{\"type\":\"reasoning\"}}\n\n",
            ).as_bytes())
            .await
            .unwrap();
        first.flush().await.unwrap();
        tokio::spawn(async move {
            for _ in 0..100 {
                if first
                    .write_all(concat!(
                        "data: {\"type\":\"response.reasoning_summary_text.delta\",",
                        "\"output_index\":0,\"delta\":\"still thinking\"}\n\n",
                    ).as_bytes())
                    .await
                    .is_err()
                {
                    break;
                }
                if first.flush().await.is_err() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        });

        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        let body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r2\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,",
            "\"delta\":\"recovered after bounded reasoning\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":",
            "{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
    });

    let startup_key = super::CHATGPT_STARTUP_NO_PROGRESS_TIMEOUT_ENV;
    let restore_startup = std::env::var(startup_key).ok();
    std::env::set_var(startup_key, "100");
    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            1,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(20),
        );
    let mut stream = client
        .stream_message(&request(vec![InputMessage::user_text("hi")], None, None))
        .await
        .expect("open stream");

    let outcome = tokio::time::timeout(std::time::Duration::from_millis(900), async {
        let mut text = String::new();
        while let Some(event) = stream.next_event().await? {
            if let StreamEvent::ContentBlockDelta(delta) = event {
                if let ContentBlockDelta::TextDelta { text: chunk } = delta.delta {
                    text.push_str(&chunk);
                }
            }
        }
        Ok::<_, ApiError>(text)
    })
    .await;
    match restore_startup {
        Some(value) => std::env::set_var(startup_key, value),
        None => std::env::remove_var(startup_key),
    }
    if outcome.is_err() {
        server.abort();
    }
    let text = outcome
        .expect("continuous reasoning must not bypass the extended startup deadline")
        .expect("restart should recover");
    server.await.unwrap();

    assert_eq!(text, "recovered after bounded reasoning");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// End-to-end proof that a pre-commit stall recovers over a real socket:
/// the mock server's first connection sends only headers and then hangs
/// (the silent-reasoning case), and the second connection serves a full SSE
/// turn. With a sub-second idle budget the stream must idle out, restart,
/// and yield the recovered text — exactly once, with no error.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn stalled_precommit_stream_restarts_and_recovers() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        // Connection 1: send headers, then go silent forever (until the
        // client gives up and drops us). This is the stalled stream.
        let (mut first, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first.flush().await.unwrap();
        // Hold the first connection open without blocking the listener from
        // accepting the client's restart (real HTTP servers accept both
        // concurrently).
        let first_holder = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            drop(first);
        });

        // Connection 2 (the restart): serve a complete SSE turn.
        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        let body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,",
            "\"delta\":\"recovered\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":",
            "{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        first_holder.abort();
    });

    // Sub-second idle budget so the stalled connection trips quickly.
    std::env::set_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV, "300");
    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            3,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(50),
        );

    let request = MessageRequest {
        messages: vec![InputMessage::user_text("hi")],
        ..request(vec![], None, None)
    };
    let mut stream = client.stream_message(&request).await.expect("open stream");

    let mut text = String::new();
    let mut events = 0;
    while let Some(event) = stream.next_event().await.expect("no error after restart") {
        events += 1;
        if let StreamEvent::ContentBlockDelta(delta) = &event {
            if let ContentBlockDelta::TextDelta { text: chunk } = &delta.delta {
                text.push_str(chunk);
            }
        }
    }
    std::env::remove_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV);
    server.await.unwrap();

    assert_eq!(
        text, "recovered",
        "recovered turn must stream after restart"
    );
    assert!(events > 0, "stream should yield events");
    // The server was hit exactly twice: the stalled attempt + the restart.
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn restart_budget_exhaustion_is_structural() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let mut holders = Vec::new();
        for _ in 0..2 {
            let (mut conn, _) = listener.accept().await.unwrap();
            server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut scratch = [0u8; 1024];
            let _ = conn.read(&mut scratch).await;
            conn.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
                .await
                .unwrap();
            conn.flush().await.unwrap();
            holders.push(tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                drop(conn);
            }));
        }
        for holder in holders {
            holder.await.unwrap();
        }
    });

    std::env::set_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV, "300");
    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            1,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(50),
        );
    let request = MessageRequest {
        messages: vec![InputMessage::user_text("hi")],
        ..request(vec![], None, None)
    };
    let mut stream = client.stream_message(&request).await.expect("open stream");

    let error = stream.next_event().await.expect_err("budget exhausted");
    std::env::remove_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV);
    assert!(
        matches!(
            error,
            ApiError::RetriesExhausted { attempts: 2, last_error }
                if matches!(last_error.as_ref(), ApiError::StreamApi { error_type, .. }
                    if error_type.as_deref() == Some("stream_idle_timeout"))
        ),
        "expected exhausted wrapper around the final idle-timeout"
    );
    server.await.unwrap();
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn terminal_failure_before_commit_falls_back_to_non_stream_response() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 4096];
        let _ = first.read(&mut scratch).await;
        let first_body = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{}}}\n\n";
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            first_body.len()
        );
        first.write_all(head.as_bytes()).await.unwrap();
        first.write_all(first_body.as_bytes()).await.unwrap();
        first.flush().await.unwrap();
        drop(first);

        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        let body = r#"{"id":"r2","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"recovered nonstream"}]}],"usage":{"input_tokens":1,"output_tokens":2}}"#;
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
    });

    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            0,
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(1),
        );
    let request = MessageRequest {
        model: "gpt-5.5".into(),
        messages: vec![InputMessage::user_text("hi")],
        effort: Some(EffortLevel::Xhigh),
        ..request(vec![], None, None)
    };
    let mut stream = client.stream_message(&request).await.expect("open stream");

    let mut recovered = false;
    while let Some(event) = stream.next_event().await.expect("fallback should recover") {
        if let StreamEvent::MessageStart(start) = event {
            recovered = start.message.content.iter().any(|block| {
                matches!(
                    block,
                    OutputContentBlock::Text { text } if text == "recovered nonstream"
                )
            });
        }
    }
    server.await.unwrap();

    assert!(recovered, "non-stream fallback response was not surfaced");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// Same socket-level proof as the empty pre-commit stall, but with the
/// real gpt-5.5 shape that triggered the TUI freeze: Responses emits
/// message/reasoning frames, then goes silent before any visible answer text
/// or tool arguments. Those frames are safe to replay, so the stream must
/// still restart instead of wedging the turn until the user interrupts.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn stalled_stream_after_reasoning_prefix_restarts_and_recovers() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        let first_body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"reasoning\"}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",",
            "\"output_index\":0,\"delta\":\"Assessing implementation quality\"}\n\n",
        );
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first.write_all(first_body.as_bytes()).await.unwrap();
        first.flush().await.unwrap();
        let first_holder = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            drop(first);
        });

        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        let second_body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r2\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,",
            "\"delta\":\"recovered\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":",
            "{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            second_body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(second_body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        first_holder.abort();
    });

    std::env::set_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV, "300");
    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            3,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(50),
        );

    let request = MessageRequest {
        messages: vec![InputMessage::user_text("hi")],
        ..request(vec![], None, None)
    };
    let mut stream = client.stream_message(&request).await.expect("open stream");

    let mut text = String::new();
    let mut reasoning_chunks = 0;
    while let Some(event) = stream.next_event().await.expect("restart should recover") {
        if let StreamEvent::ContentBlockDelta(delta) = &event {
            match &delta.delta {
                ContentBlockDelta::TextDelta { text: chunk } => text.push_str(chunk),
                ContentBlockDelta::ThinkingDelta { .. } => reasoning_chunks += 1,
                _ => {}
            }
        }
    }
    std::env::remove_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV);
    server.await.unwrap();

    assert_eq!(reasoning_chunks, 1, "first attempt surfaced reasoning");
    assert_eq!(
        text, "recovered",
        "second attempt should stream the recovered answer"
    );
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

// The mid-stream restart notice callback fires when a pre-commit stall
// transparently re-opens the upstream connection — the otherwise-silent pause
// a live UI needs to show as "reconnecting" instead of a freeze.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn mid_stream_restart_invokes_retry_notice_callback() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        // First attempt: emit only a reasoning prefix (no commit), then go
        // silent so the per-chunk idle timeout fires and forces a restart.
        let (mut first, _) = listener.accept().await.unwrap();
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        let first_body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"reasoning\"}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",",
            "\"output_index\":0,\"delta\":\"thinking\"}\n\n",
        );
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first.write_all(first_body.as_bytes()).await.unwrap();
        first.flush().await.unwrap();
        let first_holder = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            drop(first);
        });

        // Second attempt: a clean, complete turn.
        let (mut second, _) = listener.accept().await.unwrap();
        let _ = second.read(&mut scratch).await;
        let second_body = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r2\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
            "\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,",
            "\"delta\":\"ok\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":",
            "{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            second_body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(second_body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        first_holder.abort();
    });

    std::env::set_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV, "300");
    let client = super::ChatGptBackendClient::new("token", None)
        .with_base_url(format!("http://{addr}"))
        .with_retry_policy(
            3,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(50),
        );

    let request = MessageRequest {
        messages: vec![InputMessage::user_text("hi")],
        ..request(vec![], None, None)
    };

    let notices = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(u32, u32)>::new()));
    let sink = notices.clone();
    let mut stream = client
        .stream_message(&request)
        .await
        .expect("open stream")
        .with_retry_notice_callback(move |notice| {
            sink.lock()
                .unwrap()
                .push((notice.attempt, notice.max_attempts));
        });

    while let Some(_event) = stream.next_event().await.expect("restart should recover") {}
    std::env::remove_var(super::CHATGPT_STREAM_IDLE_TIMEOUT_ENV);
    server.await.unwrap();

    let seen = notices.lock().unwrap().clone();
    assert!(
        !seen.is_empty(),
        "a mid-stream transparent restart must fire the retry-notice callback"
    );
    assert_eq!(seen[0], (1, 3), "first restart is attempt 1 of max_retries 3");
}

// A large frame split across many small chunks parses exactly once and the
// separator search stays linear (the GPT `encrypted_content` shape that
// used to pin a core and freeze the TUI). See `ResponsesSseParser::scanned`.
#[test]
fn large_frame_split_across_many_chunks_parses_once() {
    let big = "x".repeat(512 * 1024);
    let frame = format!(
        "data: {{\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"{big}\"}}"
    );
    let bytes = frame.as_bytes();
    let mut parser = ResponsesSseParser::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let end = (offset + 1024).min(bytes.len());
        assert!(parser
            .push(&bytes[offset..end])
            .expect("normal large frame stays under the cap")
            .is_empty());
        offset = end;
    }
    let events = parser.push(b"\n\n").expect("terminator completes the frame");
    assert_eq!(
        events.len(),
        1,
        "frame must parse exactly once after terminator"
    );
}

// A stream that never emits a frame separator must be rejected once the retained
// buffer would exceed the crate-wide SSE cap, rather than growing without bound.
#[test]
fn oversized_unterminated_frame_is_rejected() {
    let mut parser = ResponsesSseParser::new();
    let chunk = vec![b'x'; crate::sse::MAX_SSE_BUFFER_BYTES + 1];
    let error = parser
        .push(&chunk)
        .expect_err("a chunk past the cap must be rejected");
    assert!(
        matches!(error, ApiError::InvalidSseFrame(_)),
        "expected invalid sse frame, got {error:?}"
    );
}

// A `\r\n\r\n` separator split byte-by-byte across chunk boundaries (so it
// straddles the scan resume point) must still be detected.
#[test]
fn separator_split_across_chunks_is_found() {
    let mut parser = ResponsesSseParser::new();
    assert!(parser
        .push(
            b"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"hi\"}"
        )
        .expect("partial frame buffers")
        .is_empty());
    assert!(parser.push(b"\r").expect("partial separator buffers").is_empty());
    assert!(parser.push(b"\n").expect("partial separator buffers").is_empty());
    assert!(parser.push(b"\r").expect("partial separator buffers").is_empty());
    let events = parser.push(b"\n").expect("completed separator parses");
    assert_eq!(events.len(), 1, "split CRLFCRLF separator must be detected");
}

/// `response.incomplete` (the model spent its whole output budget, usually
/// on reasoning) must close the message with an honest `max_tokens` stop —
/// ignoring it ended the stream with zero events, which the runtime
/// misread as "no assistant content" and retried the identical request
/// forever (the 2026-06-11 empty-response loop).
#[test]
fn incomplete_close_emits_max_tokens_stop_instead_of_silence() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    assert!(
        !state
            .ingest(&json!({"type":"response.created","response":{"id":"resp_1"}}))
            .is_empty()
    );
    let events = state.ingest(&json!({
        "type": "response.incomplete",
        "response": {
            "incomplete_details": { "reason": "max_output_tokens" },
            "usage": { "input_tokens": 10, "output_tokens": 2048 },
        },
    }));
    let stop_reason = events.iter().find_map(|event| match event {
        StreamEvent::MessageDelta(delta) => delta.delta.stop_reason.clone(),
        _ => None,
    });
    assert_eq!(stop_reason.as_deref(), Some("max_tokens"));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::MessageStop(_))),
        "incomplete must terminate the message"
    );
    // Terminal: later frames are ignored exactly like after `completed`.
    assert!(
        state
            .ingest(&json!({"type":"response.output_text.delta","output_index":0,"delta":"x"}))
            .is_empty()
    );
}

/// `response.failed` / top-level `error` frames surface as a real stream
/// error (retryable for server faults), never as a silent empty stream.
#[test]
fn failed_event_surfaces_as_stream_error_not_empty_stream() {
    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    let events = state.ingest(&json!({
        "type": "response.failed",
        "response": { "error": { "code": "server_error", "message": "boom" } },
    }));
    assert!(events.is_empty(), "failure is not a display event");
    let failure = state.take_failure().expect("failure must be recorded");
    assert!(failure.is_retryable(), "server faults are retryable");
    assert!(state.take_failure().is_none(), "failure drains once");

    let mut state = ResponsesStreamState::new("gpt-5.5".to_string());
    state.ingest(&json!({
        "type": "error",
        "code": "invalid_request_error",
        "message": "bad input item",
    }));
    let failure = state.take_failure().expect("error frame must be recorded");
    assert!(
        !failure.is_retryable(),
        "invalid-request class is not retryable"
    );
}

/// An empty-response retry/continuation reminder in the system blocks must
/// step the reasoning effort down — replaying the identical xhigh request
/// deterministically reproduces the empty turn.
#[test]
fn empty_retry_reminder_de_escalates_reasoning_effort() {
    // Contract pin: the runtime's reminder prefixes
    // (crates/runtime/src/conversation/mod.rs) — if these literals drift,
    // the de-escalation silently stops firing.
    assert_eq!(
        super::EMPTY_RETRY_REMINDER_MARKER,
        "[zo:empty-response-retry]"
    );
    assert_eq!(
        super::EMPTY_CONTINUATION_REMINDER_MARKER,
        "[zo:empty-response-continuation]"
    );

    let reminder_system = Some(vec![
        crate::types::SystemBlock::Text {
            text: "base operating manual".to_string(),
            cache_control: None,
        },
        crate::types::SystemBlock::Text {
            text: "[zo:empty-response-retry] <system-reminder>retry now</system-reminder>"
                .to_string(),
            cache_control: None,
        },
    ]);

    // Legacy GPT projects a 24k Max budget to xhigh, so retry pressure
    // de-escalates one step to medium.
    let mut req = request_with_model("gpt-5.5", Some(24_000));
    req.system.clone_from(&reminder_system);
    let body = build_responses_request(&req, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("medium"));

    // Same request without the reminder keeps the requested tier.
    let clean = build_responses_request(&request_with_model("gpt-5.5", Some(24_000)), "i", true);
    assert_eq!(clean["reasoning"]["effort"], json!("xhigh"));

    // User-selected `/effort xhigh` / `ultracode` is an explicit top-effort
    // contract: retry pressure must not silently lower it, including on GPT fast.
    let mut explicit = request_with_model("gpt-5.5-fast", None);
    explicit.effort = Some(crate::types::EffortLevel::Xhigh);
    explicit.system.clone_from(&reminder_system);
    let body = build_responses_request(&explicit, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert_eq!(body["service_tier"], json!("priority"));

    let mut explicit_max = request_with_model("gpt-5.5-fast", None);
    explicit_max.effort = Some(crate::types::EffortLevel::Max);
    explicit_max.system.clone_from(&reminder_system);
    let body = build_responses_request(&explicit_max, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("xhigh"));
    assert_eq!(body["service_tier"], json!("priority"));

    // Lower explicit efforts keep the existing empty-response retry behavior:
    // they still step down instead of being protected by the top-tier exception.
    let mut explicit_high = request_with_model("gpt-5.5-fast", None);
    explicit_high.effort = Some(crate::types::EffortLevel::High);
    explicit_high.system.clone_from(&reminder_system);
    let body = build_responses_request(&explicit_high, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("medium"));
    assert_eq!(body["service_tier"], json!("priority"));

    let mut explicit_medium = request_with_model("gpt-5.5-fast", None);
    explicit_medium.effort = Some(crate::types::EffortLevel::Medium);
    explicit_medium.system.clone_from(&reminder_system);
    let body = build_responses_request(&explicit_medium, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("low"));
    assert_eq!(body["service_tier"], json!("priority"));

    // The continuation reminder (post-fallback turns) de-escalates budget-derived effort too.
    let mut cont = request_with_model("gpt-5.5", Some(24_000));
    cont.system = Some(vec![crate::types::SystemBlock::Text {
        text: "[zo:empty-response-continuation] <system-reminder>state intact</system-reminder>"
            .to_string(),
        cache_control: None,
    }]);
    let body = build_responses_request(&cont, "i", true);
    assert_eq!(body["reasoning"]["effort"], json!("medium"));
}
