use super::*;
// `flatten_tool_result_content` moved to `providers::mod`, so the production
// module no longer re-exports `ToolResultContentBlock` through `super::*`.
use crate::types::ToolResultContentBlock;
use core_types::{PkceChallengeMethod, PkceCodePair};

const TEST_OAUTH_CLIENT_ID: &str = "test-client.apps.googleusercontent.com";
const TEST_OAUTH_CLIENT_SECRET: &str = "test-client-secret";

#[test]
fn oauth_config_uses_caller_owned_client() {
    let config = oauth_config_for_client(TEST_OAUTH_CLIENT_ID.to_string());
    assert_eq!(config.client_id, TEST_OAUTH_CLIENT_ID);
    assert_eq!(config.token_url, "https://oauth2.googleapis.com/token");
    assert_eq!(config.callback_port, Some(ANTIGRAVITY_CALLBACK_PORT));
    assert!(config.scopes.contains(&CLOUD_PLATFORM_SCOPE.to_string()));
    assert!(config.scopes.contains(&USERINFO_EMAIL_SCOPE.to_string()));
    assert!(config.scopes.contains(&USERINFO_PROFILE_SCOPE.to_string()));
    assert!(config.scopes.contains(&CCLOG_SCOPE.to_string()));
    assert!(config.scopes.contains(&EXPERIMENTS_SCOPE.to_string()));
}

#[test]
fn oauth_credentials_require_both_non_empty_values() {
    let credentials = oauth_client_credentials_from(
        Some(format!(" {TEST_OAUTH_CLIENT_ID} ")),
        Some(format!(" {TEST_OAUTH_CLIENT_SECRET} ")),
    )
    .expect("caller-owned credentials should parse");
    assert_eq!(
        credentials,
        (
            TEST_OAUTH_CLIENT_ID.to_string(),
            TEST_OAUTH_CLIENT_SECRET.to_string()
        )
    );

    let missing_id = oauth_client_credentials_from(None, Some(TEST_OAUTH_CLIENT_SECRET.into()))
        .expect_err("client id is required");
    assert!(missing_id.to_string().contains(GEMINI_CODE_ASSIST_OAUTH_CLIENT_ID_ENV));

    let missing_secret = oauth_client_credentials_from(Some(TEST_OAUTH_CLIENT_ID.into()), None)
        .expect_err("client secret is required");
    assert!(
        missing_secret
            .to_string()
            .contains(GEMINI_CODE_ASSIST_OAUTH_CLIENT_SECRET_ENV)
    );
}

#[test]
fn authorize_url_uses_loopback_offline_pkce() {
    let pkce = PkceCodePair {
        verifier: "verifier".to_string(),
        challenge: "challenge".to_string(),
        challenge_method: PkceChallengeMethod::S256,
    };
    let config = oauth_config_for_client(TEST_OAUTH_CLIENT_ID.to_string());
    let url = authorize_url(&config, &redirect_uri(54545), "state value", &pkce);
    assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth?"));
    assert!(url.contains("client_id=test-client.apps.googleusercontent.com"));
    assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A54545%2Foauth-callback"));
    assert!(url.contains("access_type=offline"));
    assert!(url.contains("prompt=consent"));
    assert!(url.contains("code_challenge=challenge"));
}

#[test]
fn builds_code_assist_generate_content_request() {
    let request = MessageRequest {
        model: "gemini-3-flash-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("hello")],
        system: Some(vec![SystemBlock::text("be concise")]),
        tools: Some(vec![ToolDefinition {
            name: "answer".to_string(),
            description: Some("structured answer".to_string()),
            input_schema: json!({"type":"object"}),
        }]),
        tool_choice: Some(ToolChoice::Auto),
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let body = build_generate_content_request(&request, Some("proj-1"));
    // Flash alias collapses to the bare wire id; effort None defaults the
    // thinkingLevel to the conservative "low".
    assert_eq!(body["model"], "gemini-3-flash");
    assert_eq!(
        body["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
        "low"
    );
    // Thought summaries are requested so the wait shows the model reasoning
    // instead of a blank screen (decoded into reasoning blocks).
    assert_eq!(
        body["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"],
        true
    );
    // Antigravity body wrapper.
    assert_eq!(body["userAgent"], "antigravity");
    assert!(
        body["requestId"]
            .as_str()
            .is_some_and(|id| id.starts_with("zo-")),
        "requestId should be a zo-prefixed id"
    );
    assert_eq!(body["project"], "proj-1");
    assert_eq!(body["request"]["contents"][0]["role"], "user");
    // The Claude-authored system body is preserved, now under the shared
    // non-Anthropic identity override so a Gemini-served model does not
    // introduce itself as Claude (Gemini has no system role, so the override
    // rides in the leading systemInstruction text).
    let system_text = body["request"]["systemInstruction"]["parts"][0]["text"]
        .as_str()
        .expect("systemInstruction text is a string");
    assert!(
        system_text.contains("be concise"),
        "system body preserved: {system_text}"
    );
    assert!(
        system_text.contains("gemini-3-flash-preview")
            && system_text.contains("Google")
            && system_text.contains("do not claim to be Claude"),
        "Gemini identity override present: {system_text}"
    );
    assert_eq!(
        body["request"]["tools"][0]["functionDeclarations"][0]["name"],
        "answer"
    );
}

#[test]
fn refresh_backoff_marks_and_clears() {
    super::clear_refresh_failure();
    assert!(
        !super::refresh_in_backoff(),
        "clean state is not in backoff"
    );
    super::mark_refresh_failure();
    assert!(super::refresh_in_backoff(), "after a failure we back off");
    super::clear_refresh_failure();
    assert!(!super::refresh_in_backoff(), "a success clears the backoff");
}

#[test]
fn gemini_wire_maps_family_and_effort() {
    // Flash: bare wire id, full low|medium|high passthrough (max collapses).
    assert_eq!(
        gemini_wire(
            "gemini-3.5-flash",
            ReasoningRequest::Effort(EffortLevel::Medium)
        ),
        ("gemini-3-flash".to_string(), "medium")
    );
    assert_eq!(
        gemini_wire(
            "gemini-3.5-flash",
            ReasoningRequest::Effort(EffortLevel::Max)
        ),
        ("gemini-3-flash".to_string(), "high")
    );
    assert_eq!(
        gemini_wire(
            "gemini-3.5-flash",
            ReasoningRequest::Effort(EffortLevel::Ultra)
        ),
        ("gemini-3-flash".to_string(), "high")
    );
    assert_eq!(
        gemini_wire("gemini-flash", ReasoningRequest::Auto),
        ("gemini-3-flash".to_string(), "low")
    );
    // Pro: tier baked into the id, only low|high (medium promotes to high).
    assert_eq!(
        gemini_wire("gemini-3.1-pro", ReasoningRequest::Effort(EffortLevel::Low)),
        ("gemini-3-pro-low".to_string(), "low")
    );
    assert_eq!(
        gemini_wire(
            "gemini-3.1-pro-preview",
            ReasoningRequest::Effort(EffortLevel::Medium)
        ),
        ("gemini-3-pro-high".to_string(), "high")
    );
    // Tier-less pro with no effort downgrades to -low.
    assert_eq!(
        gemini_wire("gemini-pro", ReasoningRequest::Auto),
        ("gemini-3-pro-low".to_string(), "low")
    );
}

#[test]
fn gemini_wire_maps_versioned_text_flash_without_overmatching() {
    for model in [
        "gemini-3-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-12.4-flash-preview",
    ] {
        assert_eq!(
            gemini_wire(model, ReasoningRequest::Effort(EffortLevel::High)),
            ("gemini-3-flash".to_string(), "high"),
            "{model}"
        );
    }
    for model in [
        "gemini-omni-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash-image",
        "gemini-3.6-flash-preview-image",
    ] {
        assert_eq!(
            gemini_wire(model, ReasoningRequest::Effort(EffortLevel::High)),
            (model.to_string(), "high"),
            "{model}"
        );
    }
}

#[test]
fn gemini_wire_maps_versioned_text_pro_without_overmatching() {
    for model in [
        "gemini-3-pro",
        "gemini-3.1-pro-preview",
        "gemini-3.6-pro",
        "gemini-12.4-pro-preview-customtools",
    ] {
        assert_eq!(
            gemini_wire(model, ReasoningRequest::Effort(EffortLevel::High)),
            ("gemini-3-pro-high".to_string(), "high"),
            "{model}"
        );
    }
    for model in [
        "gemini-omni-pro",
        "gemini-3.6-pro-image",
        "gemini-3.6-pro-preview-image",
    ] {
        assert_eq!(
            gemini_wire(model, ReasoningRequest::Effort(EffortLevel::High)),
            (model.to_string(), "high"),
            "{model}"
        );
    }
}

#[test]
fn not_found_context_does_not_claim_the_model_is_definitively_unavailable() {
    let error = request_not_found_context(
        ApiError::Api {
            status: reqwest::StatusCode::NOT_FOUND,
            error_type: Some("NOT_FOUND".to_string()),
            message: Some("Requested entity was not found.".to_string()),
            body: r#"{"error":{"status":"NOT_FOUND","message":"Requested entity was not found."}}"#
                .to_string(),
            retryable: false,
            retry_after: None,
        },
        "gemini-3.6-flash",
    );

    let ApiError::Api {
        message,
        body,
        retryable,
        ..
    } = error
    else {
        panic!("expected API error");
    };
    let message = message.unwrap();
    assert!(message.contains("HTTP 404 while requesting model `gemini-3.6-flash`"));
    assert!(message.contains("does not prove that the model ID is unavailable"));
    assert!(message.contains("project, endpoint, and API version"));
    assert!(!message.contains("does not currently expose model"));
    assert!(body.contains("Requested entity was not found"));
    assert!(!retryable);
}

#[test]
fn gemini_wire_flash_budget_uses_existing_effort_buckets() {
    assert_eq!(
        gemini_wire("gemini-flash", ReasoningRequest::BudgetTokens(16_000)),
        ("gemini-3-flash".to_string(), "high")
    );
}

#[test]
fn client_metadata_spoofs_antigravity_ide() {
    let metadata = client_metadata(Some("proj-7"));
    assert_eq!(metadata["ideType"], "ANTIGRAVITY");
    assert_eq!(metadata["pluginType"], "GEMINI");
    assert_eq!(metadata["duetProject"], "proj-7");
    // The header form carries no project.
    assert!(client_metadata(None).get("duetProject").is_none());
    // platform must be a valid ClientMetadata.Platform proto enum, never a
    // free label like "MACOS" (which 400s INVALID_ARGUMENT).
    assert_eq!(metadata["platform"], client_platform());
    assert!(
        [
            "DARWIN_ARM64",
            "DARWIN_AMD64",
            "LINUX_ARM64",
            "LINUX_AMD64",
            "WINDOWS_AMD64",
            "PLATFORM_UNSPECIFIED",
        ]
        .contains(&client_platform())
    );
    assert_ne!(client_platform(), "MACOS");
}

#[test]
fn antigravity_user_agent_is_overridable() {
    assert!(antigravity_user_agent().starts_with("antigravity/"));
    assert!(antigravity_user_agent().ends_with(" darwin/arm64"));
}

#[test]
fn parallel_tool_results_collapse_into_one_user_turn() {
    // Regression: a parallel tool call emits N functionCalls in one model
    // turn, but Zo persists each result as its own message, so the naive
    // 1:1 mapping produced N separate single-functionResponse user contents.
    // Gemini rejects that with "the number of function response parts is
    // equal to the number of function call parts". The two adjacent results
    // must collapse into a single user content carrying both responses.
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![
            InputMessage::user_text("read both files"),
            InputMessage {
                role: "assistant".to_string(),
                content: vec![
                    InputContentBlock::ToolUse {
                        id: "call_a".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "a.rs" }),
                                            cache_control: None,
                    },
                    InputContentBlock::ToolUse {
                        id: "call_b".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "b.rs" }),
                                            cache_control: None,
                    },
                ],
                thought_signature: None,
                reasoning_replay: None,
            },
            InputMessage::user_tool_result("call_a", "contents of a", false),
            InputMessage::user_tool_result("call_b", "contents of b", false),
        ],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let contents = body["request"]["contents"].as_array().unwrap();
    // user question, model(2 functionCalls), user(2 functionResponses)
    assert_eq!(contents.len(), 3, "the two tool results must merge");

    let model_turn = &contents[1];
    assert_eq!(model_turn["role"], "model");
    let calls = model_turn["parts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .count();

    let response_turn = &contents[2];
    assert_eq!(response_turn["role"], "user");
    let responses = response_turn["parts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|part| part.get("functionResponse").is_some())
        .count();

    assert_eq!(calls, 2, "model turn keeps both calls");
    assert_eq!(
        responses, calls,
        "function response parts must equal function call parts"
    );
}

#[test]
fn tool_ledger_parts_preserve_function_call_and_response_payloads() {
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![
            InputMessage {
                role: "assistant".to_string(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_a".to_string(),
                    name: "read_file".to_string(),
                    input: json!({ "path": "a.rs" }),
                                    cache_control: None,
                }],
                thought_signature: None,
                reasoning_replay: None,
            },
            InputMessage::user_tool_result("call_a", "contents of a", true),
        ],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let contents = body["request"]["contents"].as_array().unwrap();
    assert_eq!(
        contents[0]["parts"][0]["functionCall"]["id"],
        json!("call_a")
    );
    assert_eq!(
        contents[0]["parts"][0]["functionCall"]["name"],
        json!("read_file")
    );
    assert_eq!(
        contents[0]["parts"][0]["functionCall"]["args"],
        json!({ "path": "a.rs" })
    );
    // Gemini matches a functionResponse to its functionCall by `name` (the
    // declared function name) and `id` (the call id) — mirroring the official
    // Gemini CLI's `{ id: callId, name: toolName, response }` shape. The name
    // is resolved from the matching functionCall, NOT the opaque tool_use_id.
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["name"],
        json!("read_file")
    );
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["id"],
        json!("call_a")
    );
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["response"]["content"],
        json!("contents of a")
    );
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["response"]["is_error"],
        json!(true)
    );
}

#[test]
fn function_response_name_resolves_from_matching_function_call() {
    // Regression for the Gemini-only runaway loop: a tool result must carry the
    // matching functionCall's declared name (here `read_file`), not the opaque
    // tool_use_id. Gemini pairs response→call by name+id; a name mismatch makes
    // the model believe the result never arrived and re-issue the tool forever
    // until the sub-agent hits its iteration cap.
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![
            InputMessage::user_text("read a.rs"),
            InputMessage {
                role: "assistant".to_string(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_xyz".to_string(),
                    name: "read_file".to_string(),
                    input: json!({ "path": "a.rs" }),
                                    cache_control: None,
                }],
                thought_signature: None,
                reasoning_replay: None,
            },
            InputMessage::user_tool_result("call_xyz", "contents of a", false),
        ],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let contents = body["request"]["contents"].as_array().unwrap();
    let response = &contents[2]["parts"][0]["functionResponse"];
    assert_eq!(response["name"], json!("read_file"));
    assert_eq!(response["id"], json!("call_xyz"));
    assert_eq!(response["response"]["content"], json!("contents of a"));
}

#[test]
fn parallel_function_responses_match_each_call_name_by_id() {
    // Two parallel calls of *different* functions: each response must resolve
    // its own function name through its own id, never collapse to one name.
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![
            InputMessage {
                role: "assistant".to_string(),
                content: vec![
                    InputContentBlock::ToolUse {
                        id: "call_a".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "a.rs" }),
                                            cache_control: None,
                    },
                    InputContentBlock::ToolUse {
                        id: "call_b".to_string(),
                        name: "grep_search".to_string(),
                        input: json!({ "pattern": "TODO" }),
                                            cache_control: None,
                    },
                ],
                thought_signature: None,
                reasoning_replay: None,
            },
            // Results arrive in the opposite order to prove the match is by id,
            // not positional.
            InputMessage::user_tool_result("call_b", "two matches", false),
            InputMessage::user_tool_result("call_a", "file body", false),
        ],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let contents = body["request"]["contents"].as_array().unwrap();
    let response_parts = contents[1]["parts"].as_array().unwrap();
    assert_eq!(response_parts[0]["functionResponse"]["id"], json!("call_b"));
    assert_eq!(
        response_parts[0]["functionResponse"]["name"],
        json!("grep_search")
    );
    assert_eq!(response_parts[1]["functionResponse"]["id"], json!("call_a"));
    assert_eq!(
        response_parts[1]["functionResponse"]["name"],
        json!("read_file")
    );
}

#[test]
fn function_response_falls_back_to_id_when_call_name_unknown() {
    // An orphan tool result with no preceding functionCall (e.g. a replayed or
    // truncated session) cannot resolve a name. Fall back to the id for both
    // fields rather than dropping the part, so the request stays well-formed.
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_tool_result(
            "orphan_call",
            "stranded output",
            false,
        )],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let contents = body["request"]["contents"].as_array().unwrap();
    let response = &contents[0]["parts"][0]["functionResponse"];
    assert_eq!(response["name"], json!("orphan_call"));
    assert_eq!(response["id"], json!("orphan_call"));
    assert_eq!(response["response"]["content"], json!("stranded output"));
}

#[test]
// One line over the threshold from the added `reasoning_replay: None` field
// literal (api-wide addition, unrelated to this test's own logic).
#[allow(clippy::too_many_lines)]
fn mixed_text_parallel_tool_parts_keep_order_payloads_and_signature_slots() {
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![
            InputMessage {
                role: "assistant".to_string(),
                content: vec![
                    InputContentBlock::Text {
                        text: "I'll inspect both.".to_string(),
                        cache_control: None,
                    },
                    InputContentBlock::ToolUse {
                        id: "call_a".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "a.rs" }),
                                            cache_control: None,
                    },
                    InputContentBlock::ToolUse {
                        id: "call_b".to_string(),
                        name: "grep_search".to_string(),
                        input: json!({ "pattern": "TODO" }),
                                            cache_control: None,
                    },
                ],
                thought_signature: Some(r#"[null,"SIG_B"]"#.to_string()),
                reasoning_replay: None,
            },
            InputMessage::user_tool_result("call_a", "contents of a", false),
            InputMessage {
                role: "user".to_string(),
                content: vec![InputContentBlock::ToolResult {
                    tool_use_id: "call_b".to_string(),
                    content: vec![
                        ToolResultContentBlock::Json {
                            value: json!({ "matches": 2 }),
                        },
                        ToolResultContentBlock::Image {
                            source: ImageSource {
                                kind: "base64".to_string(),
                                media_type: "image/png".to_string(),
                                data: "abc123".to_string(),
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
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let contents = body["request"]["contents"].as_array().unwrap();
    assert_eq!(
        contents.len(),
        2,
        "adjacent tool results merge into one user turn"
    );

    let model_parts = contents[0]["parts"].as_array().unwrap();
    assert_eq!(model_parts[0]["text"], json!("I'll inspect both."));
    assert_eq!(model_parts[1]["functionCall"]["id"], json!("call_a"));
    assert_eq!(model_parts[1]["functionCall"]["name"], json!("read_file"));
    assert_eq!(
        model_parts[1]["functionCall"]["args"],
        json!({ "path": "a.rs" })
    );
    // `call_a` carried no real signature (stored as the leading `null` in the
    // `[null,"SIG_B"]` array), so the request-assembly backfill stamps Google's
    // skip sentinel rather than leaving it unsigned — an unsigned functionCall
    // would 400 the multi-turn request.
    assert_eq!(
        model_parts[1]["thoughtSignature"],
        json!("skip_thought_signature_validator")
    );
    assert_eq!(model_parts[2]["functionCall"]["id"], json!("call_b"));
    assert_eq!(model_parts[2]["functionCall"]["name"], json!("grep_search"));
    assert_eq!(
        model_parts[2]["functionCall"]["args"],
        json!({ "pattern": "TODO" })
    );
    // `call_b`'s genuine signature is preserved untouched — the backfill only
    // fills gaps, never overwrites a real signature.
    assert_eq!(model_parts[2]["thoughtSignature"], json!("SIG_B"));

    let response_parts = contents[1]["parts"].as_array().unwrap();
    assert_eq!(
        response_parts[0]["functionResponse"]["name"],
        json!("read_file")
    );
    assert_eq!(response_parts[0]["functionResponse"]["id"], json!("call_a"));
    assert_eq!(
        response_parts[0]["functionResponse"]["response"],
        json!({"content": "contents of a", "is_error": false})
    );
    assert_eq!(
        response_parts[1]["functionResponse"]["name"],
        json!("grep_search")
    );
    assert_eq!(response_parts[1]["functionResponse"]["id"], json!("call_b"));
    assert_eq!(
        response_parts[1]["functionResponse"]["response"],
        json!({"content": "{\"matches\":2}\n[image image/png]", "is_error": true})
    );
}

#[test]
fn captures_thought_signature_from_first_function_call() {
    // Gemini 3 attaches a `thoughtSignature` (sibling of `functionCall` at
    // the part level) to the first tool call. Normalize must surface it on
    // the message so the next request can echo it back — without it the
    // follow-up turn 400s ("functionCall ... is missing a thought_signature").
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("read it")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [{
                    "functionCall": { "name": "read_file", "args": { "path": "a.rs" } },
                    "thoughtSignature": "SIG_ABC123"
                }]}
            }]
        }
    });
    let response = normalize_generate_content_response(&request, &payload).unwrap();
    assert_eq!(response.thought_signature.as_deref(), Some("SIG_ABC123"));
}

#[test]
fn thought_parts_become_reasoning_not_answer_text() {
    // Gemini thought summaries (enabled by `includeThoughts`) arrive as text
    // parts flagged `"thought": true`. They must decode into a reasoning block,
    // never the answer, so the wait shows the model thinking and the final
    // answer is not polluted with its scratch work.
    let request = MessageRequest {
        model: "gemini-3.5-flash".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("solve it")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [
                    { "text": "Let me reason about this.", "thought": true },
                    { "text": "The answer is 42." }
                ]}
            }]
        }
    });
    let response = normalize_generate_content_response(&request, &payload).unwrap();
    assert_eq!(response.content.len(), 2, "one thought + one answer block");
    match &response.content[0] {
        OutputContentBlock::Thinking { thinking, .. } => {
            assert_eq!(thinking, "Let me reason about this.");
        }
        other => panic!("first block must be reasoning, got {other:?}"),
    }
    match &response.content[1] {
        OutputContentBlock::Text { text } => assert_eq!(text, "The answer is 42."),
        other => panic!("second block must be answer text, got {other:?}"),
    }
}

#[test]
fn late_thought_after_answer_does_not_reopen_reasoning_block() {
    // A thought part arriving AFTER the answer text has begun must be dropped,
    // not surface a live reasoning block after visible output. That tail-region
    // reasoning changes suppression/height calculations and can force relayouts
    // while the user is watching streamed prose (Gemini's contract is
    // thoughts-before-answer).
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());

    let _ = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "Thinking about it.", "thought": true }
        ]}}]}
    }));
    let _ = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "The answer is 42." }
        ]}}]}
    }));

    let late = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "(stray late thought)", "thought": true }
        ]}}]}
    }));

    assert!(
        !late.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index: GEMINI_THOUGHT_BLOCK_INDEX,
                ..
            })
        )),
        "late thought must not reopen the reasoning block: {late:?}"
    );
    assert!(
        !late.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::ThinkingDelta { .. },
                ..
            })
        )),
        "late thought must not emit a reasoning delta after the answer started: {late:?}"
    );
}

#[test]
fn late_thought_after_tool_call_does_not_reopen_reasoning_block() {
    // Regression for the flicker path that survived the earlier guard: a
    // functionCall closes the answer text (`text_open = false`), so using
    // `text_open` as the "answer already started" guard allowed a later thought
    // part to surface live reasoning after visible output.
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());

    let _ = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "Thinking before answer.", "thought": true }
        ]}}]}
    }));
    let _ = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "I'll inspect it." }
        ]}}]}
    }));
    let tool = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [{
            "functionCall": {
                "name": "read_file",
                "args": { "path": "a.rs" }
            }
        }]}}]}
    }));
    assert!(
        tool.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: GEMINI_TEXT_BLOCK_INDEX
            })
        )),
        "tool call should close prose before the regression late-thought frame"
    );

    let late = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "(late thought after tool)", "thought": true }
        ]}}]}
    }));

    assert!(
        !late.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index: GEMINI_THOUGHT_BLOCK_INDEX,
                ..
            })
        )),
        "late thought after tool must not reopen the reasoning block: {late:?}"
    );
    assert!(
        !late.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::ThinkingDelta { .. },
                ..
            })
        )),
        "late thought after tool must not emit a reasoning delta: {late:?}"
    );
}

fn fresh_text_block_index(events: &[StreamEvent]) -> u32 {
    events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index,
                content_block: OutputContentBlock::Text { .. },
            }) => Some(*index),
            _ => None,
        })
        .expect("text after a tool is preserved in a fresh text block")
}

fn assert_text_delta(events: &[StreamEvent], index: u32, expected: &str) {
    assert!(
        events.iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                index: delta_index,
                delta: ContentBlockDelta::TextDelta { text },
            }) if *delta_index == index && text == expected
        )),
        "text must emit its delta on the fresh block: {events:?}"
    );
}

fn first_block_start_position(
    events: &[StreamEvent],
    is_expected: fn(&OutputContentBlock) -> bool,
) -> usize {
    events
        .iter()
        .position(|event| matches!(
            event,
            StreamEvent::ContentBlockStart(ContentBlockStartEvent { content_block, .. })
                if is_expected(content_block)
        ))
        .expect("expected content block start event")
}

#[test]
fn text_after_tool_call_uses_fresh_append_only_text_block() {
    // Streaming keeps rows append-only: do not rewrite the settled first prose
    // block, and do not drop model text that arrives after a tool.
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());

    let _ = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "I'll inspect it first." },
            {
                "functionCall": {
                    "name": "read_file",
                    "args": { "path": "a.rs" }
                }
            }
        ]}}]}
    }));

    let late_text = state.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "Continuing after the tool call is preserved." }
        ]}}]}
    }));
    let appended_text_index = fresh_text_block_index(&late_text);

    assert_ne!(appended_text_index, GEMINI_TEXT_BLOCK_INDEX);
    assert_text_delta(
        &late_text,
        appended_text_index,
        "Continuing after the tool call is preserved.",
    );
    assert!(
        state.finish().iter().any(|event| matches!(
            event,
            StreamEvent::ContentBlockStop(ContentBlockStopEvent { index })
                if *index == appended_text_index
        )),
        "finish must close the fresh text block"
    );

    let mut same_frame = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let events = same_frame.ingest(&json!({
        "response": { "candidates": [{ "content": { "parts": [
            {
                "functionCall": {
                    "name": "read_file",
                    "args": { "path": "a.rs" }
                }
            },
            { "text": "Text after a tool in the same frame." }
        ]}}]}
    }));
    let tool_start = first_block_start_position(&events, |block| {
        matches!(block, OutputContentBlock::ToolUse { .. })
    });
    let text_start = first_block_start_position(&events, |block| {
        matches!(block, OutputContentBlock::Text { .. })
    });
    let same_frame_text_index = fresh_text_block_index(&events);

    assert!(tool_start < text_start);
    assert_ne!(same_frame_text_index, GEMINI_TEXT_BLOCK_INDEX);
    assert_text_delta(&events, same_frame_text_index, "Text after a tool in the same frame.");
}

fn first_tool_use_id(events: &[StreamEvent]) -> &str {
    events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                content_block: OutputContentBlock::ToolUse { id, .. },
                ..
            }) => Some(id.as_str()),
            _ => None,
        })
        .expect("tool use start event")
}

#[test]
fn streaming_idless_function_calls_get_unique_fallback_ids_across_streams() {
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [{
                    "functionCall": {
                        "name": "glob_search",
                        "args": { "pattern": "**/*.rs" }
                    }
                }]}
            }]
        }
    });
    let mut first = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let mut second = GeminiStreamState::new("gemini-3.5-flash".to_string());

    let first_id = first_tool_use_id(&first.ingest(&payload)).to_string();
    let second_id = first_tool_use_id(&second.ingest(&payload)).to_string();

    assert!(first_id.starts_with("gemini_tool_call_"));
    assert!(second_id.starts_with("gemini_tool_call_"));
    assert_ne!(
        first_id, second_id,
        "id-less Gemini functionCalls from separate streams must not overwrite prior TUI rows"
    );
}

#[test]
fn streaming_parallel_idless_function_calls_share_scope_but_keep_distinct_indices() {
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [
                    { "functionCall": { "name": "glob_search", "args": { "pattern": "**/*.rs" } } },
                    { "functionCall": { "name": "read_file", "args": { "path": "a.rs" } } }
                ]}
            }]
        }
    });

    let events = state.ingest(&payload);
    let ids = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                content_block: OutputContentBlock::ToolUse { id, .. },
                ..
            }) => Some(id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| id.starts_with("gemini_tool_call_")));
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn streaming_function_call_preserves_provider_id_when_present() {
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [{
                    "functionCall": {
                        "id": "gemini-provider-id",
                        "name": "read_file",
                        "args": { "path": "a.rs" }
                    }
                }]}
            }]
        }
    });

    assert_eq!(first_tool_use_id(&state.ingest(&payload)), "gemini-provider-id");
}

#[test]
fn streaming_function_call_prefills_args_without_duplicate_delta() {
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [{
                    "functionCall": {
                        "name": "read_file",
                        "args": { "path": "a.rs" }
                    }
                }]}
            }]
        }
    });

    let events = state.ingest(&payload);
    assert!(
        events.iter().all(|event| !matches!(
            event,
            StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                delta: ContentBlockDelta::InputJsonDelta { .. },
                ..
            })
        )),
        "Gemini complete args must not be emitted again as input_json_delta"
    );

    let tool_start = events
        .iter()
        .find_map(|event| match event {
            StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                content_block: OutputContentBlock::ToolUse { name, input, .. },
                ..
            }) => Some((name, input)),
            _ => None,
        })
        .expect("tool start event");
    assert_eq!(tool_start.0, "read_file");
    assert_eq!(tool_start.1, &json!({ "path": "a.rs" }));
}

#[test]
fn streaming_function_call_closes_open_text_before_tool() {
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let payload = json!({
        "response": {
            "candidates": [{
                "content": { "parts": [
                    { "text": "I'll inspect it first." },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": { "path": "a.rs" }
                        }
                    }
                ]}
            }]
        }
    });

    let events = state.ingest(&payload);
    let text_stop = events
        .iter()
        .position(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                    index: GEMINI_TEXT_BLOCK_INDEX
                })
            )
        })
        .expect("text block should close before tool call");
    let tool_start = events
        .iter()
        .position(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    content_block: OutputContentBlock::ToolUse { .. },
                    ..
                })
            )
        })
        .expect("tool start event");
    assert!(
        text_stop < tool_start,
        "text must settle before tool row starts"
    );

    let finish_events = state.finish();
    assert!(
        finish_events.iter().all(|event| !matches!(
            event,
            StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: GEMINI_TEXT_BLOCK_INDEX
            })
        )),
        "finish must not emit a second empty text stop after a tool flushed prose"
    );
}

#[test]
fn streaming_answer_text_settles_open_thought_block() {
    // Gemini streams thought summaries ("thought": true) before the answer.
    // When the answer begins, the thought block must settle (ContentBlockStop)
    // so the TUI does not keep a `Reasoning { done: false }` above live prose —
    // which forces a full reasoning+answer re-layout on every token (flicker).
    let mut state = GeminiStreamState::new("gemini-3.5-flash".to_string());
    let thought = json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "Let me reason about this.", "thought": true }
        ]}}]}
    });
    let _ = state.ingest(&thought);

    let answer = json!({
        "response": { "candidates": [{ "content": { "parts": [
            { "text": "Here is the answer." }
        ]}}]}
    });
    let events = state.ingest(&answer);

    let thought_stop = events
        .iter()
        .position(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                    index: GEMINI_THOUGHT_BLOCK_INDEX
                })
            )
        })
        .expect("thought block must settle when the answer begins");
    let text_start = events
        .iter()
        .position(|event| {
            matches!(
                event,
                StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index: GEMINI_TEXT_BLOCK_INDEX,
                    ..
                })
            )
        })
        .expect("answer text block opens");
    assert!(
        thought_stop < text_start,
        "thought settles before the answer text opens"
    );
}

#[test]
fn echoes_thought_signature_back_on_first_function_call() {
    // Round-trip: a stored assistant turn carrying a Gemini signature must
    // re-attach it to its functionCall part on the next request.
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![
            InputMessage::user_text("read it"),
            InputMessage {
                role: "assistant".to_string(),
                content: vec![InputContentBlock::ToolUse {
                    id: "call_a".to_string(),
                    name: "read_file".to_string(),
                    input: json!({ "path": "a.rs" }),
                                    cache_control: None,
                }],
                thought_signature: Some("SIG_ABC123".to_string()),
                reasoning_replay: None,
            },
            InputMessage::user_tool_result("call_a", "contents", false),
        ],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let body = build_generate_content_request(&request, None);
    let model_turn = &body["request"]["contents"][1];
    assert_eq!(model_turn["role"], "model");
    let call_part = model_turn["parts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|part| part.get("functionCall").is_some())
        .expect("functionCall part present");
    assert_eq!(call_part["thoughtSignature"], "SIG_ABC123");
}

#[test]
fn captures_and_echoes_per_part_signatures_for_parallel_calls() {
    // Gemini 3 Flash signs the first one or two parts of a parallel call.
    // Capturing only the first dropped position 2's signature, so the
    // follow-up turn 400'd ("missing a thought_signature ... position 2").
    // Each part's own signature must round-trip back onto its own part.
    let request = MessageRequest {
        model: "gemini-3-flash".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("read both")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "response": { "candidates": [{ "content": { "parts": [
            {
                "functionCall": { "name": "glob_search", "args": { "q": "**/*.md" } },
                "thoughtSignature": "SIG_ONE"
            },
            {
                "functionCall": { "name": "read_file", "args": { "path": "a.rs" } },
                "thoughtSignature": "SIG_TWO"
            }
        ]}}]}
    });
    let response = normalize_generate_content_response(&request, &payload).unwrap();

    // Round-trip the captured signature through a stored assistant turn back
    // into a follow-up request, mirroring how the conversation loop replays it.
    let follow_up = MessageRequest {
        messages: vec![
            InputMessage::user_text("read both"),
            InputMessage {
                role: "assistant".to_string(),
                content: vec![
                    InputContentBlock::ToolUse {
                        id: "call_a".to_string(),
                        name: "glob_search".to_string(),
                        input: json!({ "q": "**/*.md" }),
                                            cache_control: None,
                    },
                    InputContentBlock::ToolUse {
                        id: "call_b".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": "a.rs" }),
                                            cache_control: None,
                    },
                ],
                thought_signature: response.thought_signature.clone(),
                reasoning_replay: None,
            },
            InputMessage::user_tool_result("call_a", "x", false),
            InputMessage::user_tool_result("call_b", "y", false),
        ],
        ..request
    };
    let body = build_generate_content_request(&follow_up, None);
    let parts = body["request"]["contents"][1]["parts"].as_array().unwrap();
    let calls: Vec<&Value> = parts
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["thoughtSignature"], "SIG_ONE");
    assert_eq!(
        calls[1]["thoughtSignature"], "SIG_TWO",
        "position 2 must keep its own signature, not be dropped"
    );
}

#[test]
fn parallel_signature_encoding_skips_unsigned_trailing_parts() {
    // When the model signs only the first of three parallel calls (its known
    // inconsistency), the unsigned parts get no field — never a wrong one.
    let encoded =
        encode_thought_signatures(&[Some("SIG".to_string()), None, None]).expect("encoded");
    let mut parts = vec![
        json!({ "functionCall": { "name": "a" } }),
        json!({ "functionCall": { "name": "b" } }),
        json!({ "functionCall": { "name": "c" } }),
    ];
    distribute_thought_signatures(&encoded, &mut parts);
    assert_eq!(parts[0]["thoughtSignature"], "SIG");
    assert!(parts[1].get("thoughtSignature").is_none());
    assert!(parts[2].get("thoughtSignature").is_none());

    // A lone signature stays a plain string (legacy single-call shape).
    assert_eq!(
        encode_thought_signatures(&[Some("SOLO".to_string())]).as_deref(),
        Some("SOLO")
    );
    assert_eq!(encode_thought_signatures(&[None, None]), None);
}

#[test]
fn parallel_signature_encoding_preserves_unsigned_gaps() {
    // Some Gemini parallel tool turns sign non-contiguous parts. The JSON array
    // encoding must preserve unsigned gaps so later signatures stay attached to
    // their original functionCall position.
    let encoded = encode_thought_signatures(&[
        Some("SIG_ONE".to_string()),
        None,
        Some("SIG_THREE".to_string()),
    ])
    .expect("encoded");
    let mut parts = vec![
        json!({ "functionCall": { "name": "a" } }),
        json!({ "functionCall": { "name": "b" } }),
        json!({ "functionCall": { "name": "c" } }),
    ];

    distribute_thought_signatures(&encoded, &mut parts);

    assert_eq!(parts[0]["thoughtSignature"], "SIG_ONE");
    assert!(parts[1].get("thoughtSignature").is_none());
    assert_eq!(parts[2]["thoughtSignature"], "SIG_THREE");
}

#[test]
fn foreign_model_tool_calls_get_skip_sentinel_not_400() {
    // A mid-conversation swap TO Gemini leaves earlier tool calls that some other
    // model emitted with no `thought_signature` at all. Sending them unsigned
    // 400s ("Function call is missing a thought_signature in functionCall
    // parts"); the request-assembly backfill must stamp Google's skip sentinel on
    // every such call so the turn is accepted.
    let request = MessageRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage {
            role: "assistant".to_string(),
            content: vec![
                InputContentBlock::ToolUse {
                    id: "call_a".to_string(),
                    name: "TodoWrite".to_string(),
                    input: json!({ "todos": [] }),
                                    cache_control: None,
                },
                InputContentBlock::ToolUse {
                    id: "call_b".to_string(),
                    name: "read_file".to_string(),
                    input: json!({ "path": "b.rs" }),
                                    cache_control: None,
                },
            ],
            // No signature: the calls came from a non-Gemini model's turn.
            thought_signature: None,
            reasoning_replay: None,
        }],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    let body = build_generate_content_request(&request, None);
    let parts = body["request"]["contents"][0]["parts"].as_array().unwrap();
    assert_eq!(
        parts[0]["thoughtSignature"],
        json!("skip_thought_signature_validator")
    );
    assert_eq!(
        parts[1]["thoughtSignature"],
        json!("skip_thought_signature_validator")
    );
}

#[test]
fn backfill_never_overwrites_a_real_signature() {
    // The gap-filler must be idempotent w.r.t. genuine signatures: a real one is
    // reasoning continuity Google cryptographically checks, so it stays verbatim
    // while only the unsigned sibling receives the sentinel.
    let mut parts = vec![
        json!({ "functionCall": { "name": "a" }, "thoughtSignature": "REAL" }),
        json!({ "functionCall": { "name": "b" } }),
        json!({ "text": "not a call" }),
    ];
    backfill_missing_thought_signatures(&mut parts);
    assert_eq!(parts[0]["thoughtSignature"], json!("REAL"));
    assert_eq!(
        parts[1]["thoughtSignature"],
        json!("skip_thought_signature_validator")
    );
    // Non-functionCall parts are never stamped.
    assert!(parts[2].get("thoughtSignature").is_none());
}

#[test]
fn backfill_missing_thought_signatures_is_idempotent() {
    // Two applications must equal one: a genuine signature is left alone (as
    // `backfill_never_overwrites_a_real_signature` already pins), and the
    // second pass over an already-stamped or already-signed part finds
    // nothing left to change — `missing` becomes false everywhere.
    let mut parts = vec![
        json!({ "functionCall": { "name": "a" }, "thoughtSignature": "REAL" }),
        json!({ "functionCall": { "name": "b" } }),
        json!({ "functionCall": { "name": "c" }, "thoughtSignature": "" }),
        json!({ "text": "not a call" }),
    ];
    backfill_missing_thought_signatures(&mut parts);
    let once = parts.clone();

    backfill_missing_thought_signatures(&mut parts);
    assert_eq!(
        parts, once,
        "applying the backfill a second time must equal applying it once"
    );
}

#[test]
fn nonstream_idless_function_calls_get_unique_fallback_ids_across_responses() {
    let request = MessageRequest {
        model: "gemini-3-flash".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("use tools")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "response": { "candidates": [{ "content": { "parts": [{
            "functionCall": { "name": "glob_search", "args": { "pattern": "**/*.rs" } }
        }]}}]}
    });

    let first = normalize_generate_content_response(&request, &payload).unwrap();
    let second = normalize_generate_content_response(&request, &payload).unwrap();
    let first_id = match &first.content[0] {
        OutputContentBlock::ToolUse { id, .. } => id,
        other => panic!("expected tool use, got {other:?}"),
    };
    let second_id = match &second.content[0] {
        OutputContentBlock::ToolUse { id, .. } => id,
        other => panic!("expected tool use, got {other:?}"),
    };

    assert!(first_id.starts_with("gemini_tool_call_"));
    assert!(second_id.starts_with("gemini_tool_call_"));
    assert_ne!(
        first_id, second_id,
        "non-stream fallback ids must be unique across separate responses too"
    );
}

#[test]
fn nonstream_function_call_preserves_provider_id_when_present() {
    let request = MessageRequest {
        model: "gemini-3-flash".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("use tool")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "response": { "candidates": [{ "content": { "parts": [{
            "functionCall": {
                "id": "provider-call-id",
                "name": "read_file",
                "args": { "path": "a.rs" }
            }
        }]}}]}
    });

    let response = normalize_generate_content_response(&request, &payload).unwrap();
    let id = match &response.content[0] {
        OutputContentBlock::ToolUse { id, .. } => id,
        other => panic!("expected tool use, got {other:?}"),
    };
    assert_eq!(id, "provider-call-id");
}

#[test]
fn normalizes_code_assist_response() {
    let request = MessageRequest {
        model: "gemini-3-flash-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("hello")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "traceId": "trace-1",
        "response": {
            "modelVersion": "gemini-3-flash-preview",
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 5
            },
            "candidates": [{
                "finishReason": "STOP",
                "content": { "parts": [{ "text": "hi" }] }
            }]
        }
    });
    let response = normalize_generate_content_response(&request, &payload).unwrap();
    assert_eq!(response.id, "trace-1");
    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(response.usage.input_tokens, 3);
    assert_eq!(response.usage.output_tokens, 5);
    assert_eq!(
        response.content,
        vec![OutputContentBlock::Text { text: "hi".into() }]
    );
}

/// Gemini bills reasoning tokens in `thoughtsTokenCount` SEPARATELY from
/// `candidatesTokenCount`, and reports `promptTokenCount` INCLUDING the cached
/// subset `cachedContentTokenCount`. The usage mapping must fold thoughts into
/// output and split cache reads out of input (kept disjoint so `total_tokens`
/// doesn't double-count) — otherwise a thinking turn's cost/context is undercounted.
#[test]
fn usage_counts_thinking_tokens_as_output_and_splits_cached_prompt_tokens() {
    let request = MessageRequest {
        model: "gemini-3-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("hello")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let payload = json!({
        "traceId": "trace-usage",
        "response": {
            "modelVersion": "gemini-3-pro-preview",
            "usageMetadata": {
                "promptTokenCount": 100,
                "cachedContentTokenCount": 30,
                "candidatesTokenCount": 50,
                "thoughtsTokenCount": 200,
                "totalTokenCount": 350
            },
            "candidates": [{
                "finishReason": "STOP",
                "content": { "parts": [{ "text": "answer" }] }
            }]
        }
    });
    let usage = normalize_generate_content_response(&request, &payload)
        .unwrap()
        .usage;
    // Input excludes the cached subset; cache reads are surfaced separately.
    assert_eq!(usage.input_tokens, 70, "input = prompt(100) - cached(30)");
    assert_eq!(
        usage.cache_read_input_tokens, 30,
        "cached prompt tokens must be surfaced, not dropped"
    );
    assert_eq!(usage.cache_creation_input_tokens, 0);
    // Output folds the visible answer + the reasoning/thinking tokens.
    assert_eq!(
        usage.output_tokens, 250,
        "output = candidates(50) + thoughts(200)"
    );
    // Disjoint model: total must not double-count the cached subset.
    assert_eq!(usage.total_tokens(), 350);
}

#[test]
fn rewrites_multi_type_union_into_anyof() {
    // Regression: the Config tool's `value` carries a JSON Schema type union
    // (`["string", "boolean", "number"]`). Gemini's proto Schema rejects a
    // list-valued `type`; it must become an `anyOf` of single-type schemas.
    let schema = json!({
        "type": "object",
        "properties": {
            "setting": { "type": "string" },
            "value": { "type": ["string", "boolean", "number"] }
        },
        "required": ["setting"],
        "additionalProperties": false
    });

    let out = gemini_parameter_schema(&schema);

    assert_eq!(out["type"], "object");
    assert_eq!(out["properties"]["setting"]["type"], "string");
    let value = &out["properties"]["value"];
    assert!(value.get("type").is_none(), "union must not stay a list");
    assert_eq!(
        value["anyOf"],
        json!([{ "type": "string" }, { "type": "boolean" }, { "type": "number" }])
    );
    // Sibling keywords are preserved untouched.
    assert_eq!(out["required"], json!(["setting"]));
    assert_eq!(out["additionalProperties"], false);
}

#[test]
fn rewrites_nullable_union_into_nullable_flag() {
    let schema = json!({ "type": ["string", "null"] });
    let out = gemini_parameter_schema(&schema);
    assert_eq!(out["type"], "string");
    assert_eq!(out["nullable"], true);
    assert!(out.get("anyOf").is_none());
}

#[test]
fn rewrites_multi_type_union_with_null_into_nullable_anyof() {
    let schema = json!({ "type": ["string", "number", "null"] });
    let out = gemini_parameter_schema(&schema);
    assert!(out.get("type").is_none());
    assert_eq!(
        out["anyOf"],
        json!([{ "type": "string" }, { "type": "number" }])
    );
    assert_eq!(out["nullable"], true);
}

#[test]
fn recurses_through_nested_objects_and_arrays() {
    let schema = json!({
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": { "type": ["integer", "null"] }
            }
        }
    });
    let out = gemini_parameter_schema(&schema);
    let item = &out["properties"]["rows"]["items"];
    assert_eq!(item["type"], "integer");
    assert_eq!(item["nullable"], true);
}

#[test]
fn strips_json_schema_metadata_keywords() {
    let schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/tool.schema.json",
        "$defs": { "unused": { "type": "string" } },
        "$ref": "#/$defs/unused",
        "definitions": { "legacy": { "type": "number" } },
        "type": "object",
        "properties": {
            "payload": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "string"
            },
            "$schema": { "type": "string" }
        }
    });

    let out = gemini_parameter_schema(&schema);

    assert!(out.get("$schema").is_none());
    assert!(out.get("$id").is_none());
    assert!(out.get("$defs").is_none());
    assert!(out.get("$ref").is_none());
    assert!(out.get("definitions").is_none());
    assert!(out["properties"]["payload"].get("$schema").is_none());
    assert_eq!(out["properties"]["payload"]["type"], "string");
    // A property literally named "$schema" is an argument name, not a Schema
    // keyword; preserve it while sanitizing its schema value.
    assert_eq!(out["properties"]["$schema"]["type"], "string");
}

#[test]
fn leaves_plain_single_type_schemas_unchanged() {
    let schema = json!({
        "type": "object",
        "properties": {
            "message": { "type": "string", "description": "text" },
            "count": { "type": "integer", "minimum": 1 }
        },
        "required": ["message"]
    });
    assert_eq!(gemini_parameter_schema(&schema), schema);
}

#[test]
fn strips_gemini_unsupported_composition_and_constraint_keywords() {
    // Regression: live MCP tool schemas (playwright `browser_drop.data`,
    // chrome-devtools `list_console_messages.pageSize`) carry `propertyNames`,
    // `patternProperties`, `exclusiveMinimum`, and `not`, all of which Gemini's
    // proto Schema rejects with "Unknown name ... Cannot find field", 400ing the
    // whole tool declaration. They must be sanitized, not passed through.
    let schema = json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "propertyNames": { "type": "string" },
                "patternProperties": { "^x-": { "type": "string" } },
                "additionalProperties": { "type": "string" }
            },
            "pageSize": { "type": "integer", "exclusiveMinimum": 0 },
            "ratio": { "type": "number", "exclusiveMaximum": 1 },
            "flagged": { "not": { "const": "no" } }
        }
    });

    let out = gemini_parameter_schema(&schema);

    let data = &out["properties"]["data"];
    assert!(data.get("propertyNames").is_none());
    assert!(data.get("patternProperties").is_none());
    // `additionalProperties` has a Gemini field and is preserved.
    assert_eq!(data["additionalProperties"]["type"], "string");
    // Numeric exclusive bounds fold into the inclusive form so the constraint
    // survives rather than being silently dropped.
    let page = &out["properties"]["pageSize"];
    assert!(page.get("exclusiveMinimum").is_none());
    assert_eq!(page["minimum"], json!(0));
    let ratio = &out["properties"]["ratio"];
    assert!(ratio.get("exclusiveMaximum").is_none());
    assert_eq!(ratio["maximum"], json!(1));
    // `not` has no Gemini equivalent and is stripped, leaving the field usable.
    assert!(out["properties"]["flagged"].get("not").is_none());
}

#[test]
fn rewrites_string_const_into_single_value_enum() {
    // Regression: atlassian `createJiraIssue.description` uses an anyOf branch
    // with `{ "const": "doc" }`; Gemini has no `const`. A string literal becomes
    // a one-value string enum so the constraint is preserved.
    let schema = json!({
        "type": "object",
        "properties": {
            "kind": {
                "anyOf": [
                    { "type": "string" },
                    { "type": "object", "properties": { "type": { "const": "doc" } } }
                ]
            }
        }
    });

    let out = gemini_parameter_schema(&schema);
    let branch = &out["properties"]["kind"]["anyOf"][1]["properties"]["type"];
    assert!(branch.get("const").is_none());
    assert_eq!(branch["type"], "string");
    assert_eq!(branch["enum"], json!(["doc"]));
}

#[test]
fn rewrites_oneof_into_anyof_and_sanitizes_branches() {
    let schema = json!({
        "oneOf": [
            { "type": "string", "exclusiveMinimum": 0 },
            { "type": ["integer", "null"] }
        ]
    });
    let out = gemini_parameter_schema(&schema);
    assert!(out.get("oneOf").is_none());
    // Branch 0's exclusive bound was folded; branch 1's null union became
    // a nullable flag — proving branches are recursively sanitized.
    assert_eq!(out["anyOf"][0]["type"], "string");
    assert_eq!(out["anyOf"][1]["type"], "integer");
    assert_eq!(out["anyOf"][1]["nullable"], true);
}


// ---------------------------------------------------------------------------
// Mid-stream restart: pre-commit stalls recover; post-commit stalls propagate.
// ---------------------------------------------------------------------------

struct EnvVarGuard {
    key: &'static str,
    original: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: Option<&str>) -> Self {
        let original = std::env::var_os(key);
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn streaming_request() -> MessageRequest {
    MessageRequest {
        model: "gemini-3-flash".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("hi")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: true,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    }
}

async fn open_test_stream(
    client: &GeminiCodeAssistClient,
    url: &str,
    body: Vec<u8>,
) -> GeminiCodeAssistStream {
    let response = client
        .open_stream_response(url, body.clone(), "gemini-3-flash")
        .await
        .expect("open stream");
    GeminiCodeAssistStream::new(
        response,
        client.clone(),
        url.to_string(),
        body,
        "gemini-3-flash".to_string(),
    )
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn gemini_stalled_precommit_stream_restarts_and_recovers() {
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
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;

        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"recovered\"}]},",
            "\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1}}}\n\n"
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    });

    let _idle = EnvVarGuard::set(crate::providers::STREAM_IDLE_TIMEOUT_ENV, Some("300"));
    let client = GeminiCodeAssistClient::new("token").with_retry_policy(
        3,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(50),
    );
    let body = serde_json::to_vec(&build_generate_content_request(&streaming_request(), None)).unwrap();
    let url = format!("http://{addr}/v1internal:streamGenerateContent?alt=sse");
    let mut stream = open_test_stream(&client, &url, body).await;

    let mut text = String::new();
    while let Some(event) = stream.next_event().await.expect("restart should recover") {
        if let StreamEvent::ContentBlockDelta(delta) = &event {
            if let ContentBlockDelta::TextDelta { text: chunk } = &delta.delta {
                text.push_str(chunk);
            }
        }
    }
    server.await.unwrap();

    assert_eq!(text, "recovered");
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn gemini_restart_budget_exhausted_surfaces_error() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut conn, _) = listener.accept().await.unwrap();
            server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut scratch = [0u8; 1024];
            let _ = conn.read(&mut scratch).await;
            conn.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
                .await
                .unwrap();
            conn.flush().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        }
    });

    let _idle = EnvVarGuard::set(crate::providers::STREAM_IDLE_TIMEOUT_ENV, Some("300"));
    let client = GeminiCodeAssistClient::new("token").with_retry_policy(
        1,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(50),
    );
    let body = serde_json::to_vec(&build_generate_content_request(&streaming_request(), None)).unwrap();
    let url = format!("http://{addr}/v1internal:streamGenerateContent?alt=sse");
    let mut stream = open_test_stream(&client, &url, body).await;

    let error = stream.next_event().await.expect_err("budget exhausted");
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

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn gemini_committed_stream_propagates_instead_of_restarting() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let (mut conn, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = conn.read(&mut scratch).await;
        conn.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        conn.write_all(b"data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}}\n\n")
            .await
            .unwrap();
        conn.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    });

    let _idle = EnvVarGuard::set(crate::providers::STREAM_IDLE_TIMEOUT_ENV, Some("300"));
    let client = GeminiCodeAssistClient::new("token").with_retry_policy(
        3,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(50),
    );
    let body = serde_json::to_vec(&build_generate_content_request(&streaming_request(), None)).unwrap();
    let url = format!("http://{addr}/v1internal:streamGenerateContent?alt=sse");
    let mut stream = open_test_stream(&client, &url, body).await;

    let mut saw_partial = false;
    while let Some(event) = stream.next_event().await.expect("events before stall ok") {
        if let StreamEvent::ContentBlockDelta(delta) = event {
            if let ContentBlockDelta::TextDelta { text } = delta.delta {
                assert_eq!(text, "partial");
                saw_partial = true;
                break;
            }
        }
    }
    assert!(saw_partial, "stream must surface the committing text delta");
    let error = stream.next_event().await.expect_err("post-commit error");
    assert!(
        matches!(
            error,
            ApiError::StreamApi { error_type, .. }
                if error_type.as_deref() == Some("stream_idle_timeout")
        ),
        "post-commit idle must propagate"
    );
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    server.abort();
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn gemini_restart_reopen_is_bounded_by_the_wallclock_budget() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        // First stream opens and goes silent pre-commit.
        let (mut first, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(450)).await;

        // Reopen: accept the connection but never answer it. Without the
        // remaining-window timeout around the reopen this would park the
        // stream indefinitely (the shared client has no blanket timeout).
        let (mut second, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = second.read(&mut scratch).await;
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    });

    let _idle = EnvVarGuard::set(crate::providers::STREAM_IDLE_TIMEOUT_ENV, Some("300"));
    let client = GeminiCodeAssistClient::new("token").with_retry_policy(
        3,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(50),
    );
    let body =
        serde_json::to_vec(&build_generate_content_request(&streaming_request(), None)).unwrap();
    let url = format!("http://{addr}/v1internal:streamGenerateContent?alt=sse");
    let mut stream = open_test_stream(&client, &url, body).await;
    // Shrink the restart window so the never-answered reopen fails out in
    // test time instead of the production 120s ceiling.
    stream.max_restart_wallclock = std::time::Duration::from_millis(500);

    let started = std::time::Instant::now();
    let error = stream.next_event().await.expect_err("reopen must time out");
    assert!(
        matches!(
            &error,
            ApiError::RetriesExhausted { attempts: 2, last_error }
                if matches!(last_error.as_ref(), ApiError::StreamApi { error_type, .. }
                    if error_type.as_deref() == Some("stream_restart_timeout"))
        ),
        "expected exhausted wrapper around stream_restart_timeout, got: {error:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "reopen wait must be bounded by the restart window, took {:?}",
        started.elapsed()
    );
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 2);
    server.abort();
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn gemini_restart_fires_the_retry_notice_sink() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        // First stream opens and goes silent pre-commit.
        let (mut first, _) = listener.accept().await.unwrap();
        let mut scratch = [0u8; 1024];
        let _ = first.read(&mut scratch).await;
        first
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        first.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(450)).await;

        // Second stream recovers.
        let (mut second, _) = listener.accept().await.unwrap();
        let _ = second.read(&mut scratch).await;
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"recovered\"}]},",
            "\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1}}}\n\n"
        );
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        second.write_all(head.as_bytes()).await.unwrap();
        second.write_all(body.as_bytes()).await.unwrap();
        second.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    });

    let _idle = EnvVarGuard::set(crate::providers::STREAM_IDLE_TIMEOUT_ENV, Some("300"));
    let client = GeminiCodeAssistClient::new("token").with_retry_policy(
        3,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(50),
    );
    let body =
        serde_json::to_vec(&build_generate_content_request(&streaming_request(), None)).unwrap();
    let url = format!("http://{addr}/v1internal:streamGenerateContent?alt=sse");
    let notices = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = notices.clone();
    let mut stream = open_test_stream(&client, &url, body)
        .await
        .with_retry_notice_callback(move |notice| sink.lock().unwrap().push(notice));

    let mut text = String::new();
    while let Some(event) = stream.next_event().await.expect("restart should recover") {
        if let StreamEvent::ContentBlockDelta(delta) = &event {
            if let ContentBlockDelta::TextDelta { text: chunk } = &delta.delta {
                text.push_str(chunk);
            }
        }
    }
    server.await.unwrap();
    assert_eq!(text, "recovered");

    let notices = notices.lock().unwrap();
    assert_eq!(
        notices.len(),
        1,
        "exactly one restart must fire exactly one notice"
    );
    assert_eq!(notices[0].attempt, 1);
    assert_eq!(notices[0].max_attempts, 3);
    assert_eq!(
        notices[0].label, "transient provider error",
        "idle timeout classifies as a transient (non-capacity) fault"
    );
}

/// A Gemini `functionCall` is surfaced as a single `ContentBlockStart` carrying
/// the COMPLETE arguments (no streamed `InputJsonDelta`, unlike the other
/// backends), so the shared commit boundary never commits it. Without the
/// Gemini-specific rule a retryable fault after the tool call would `restart()`
/// and re-emit it, duplicating a side-effecting tool execution. A surfaced
/// tool-call start must commit (disarm restart); replay-safe framing/reasoning
/// and empty placeholders must not.
#[test]
fn a_surfaced_gemini_tool_call_commits_the_stream_so_a_fault_cannot_duplicate_it() {
    let tool_start = StreamEvent::ContentBlockStart(ContentBlockStartEvent {
        index: 0,
        content_block: OutputContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "write_file".to_string(),
            input: json!({"path": "x", "content": "y"}),
        },
    });
    // Precondition: the shared boundary alone does NOT commit a Gemini tool call
    // (it carries no InputJsonDelta) — exactly why the extra rule is needed.
    assert!(
        !crosses_restart_commit_boundary(&tool_start),
        "shared boundary must not commit a functionCall start on its own"
    );
    assert!(
        gemini_stream_commit_boundary(&tool_start),
        "a surfaced Gemini tool call must commit so a fault cannot replay/duplicate it"
    );

    // A no-arg functionCall (empty seeded args) is just as non-replay-safe.
    let no_arg_start = StreamEvent::ContentBlockStart(ContentBlockStartEvent {
        index: 1,
        content_block: OutputContentBlock::ToolUse {
            id: "call_2".to_string(),
            name: "list_dir".to_string(),
            input: json!({}),
        },
    });
    assert!(gemini_stream_commit_boundary(&no_arg_start));

    // Replay-safe framing, reasoning, and empty text must NOT commit (restart
    // stays armed so a stalled stream can recover without wedging the turn).
    let block_stop = StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: 0 });
    assert!(!gemini_stream_commit_boundary(&block_stop));
    let thinking = StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
        index: 0,
        delta: ContentBlockDelta::ThinkingDelta {
            thinking: "hmm".to_string(),
        },
    });
    assert!(!gemini_stream_commit_boundary(&thinking));
    let empty_text = StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
        index: 0,
        delta: ContentBlockDelta::TextDelta { text: String::new() },
    });
    assert!(!gemini_stream_commit_boundary(&empty_text));

    // Real surfaced text still commits (unchanged shared rule).
    let real_text = StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
        index: 0,
        delta: ContentBlockDelta::TextDelta {
            text: "hi".to_string(),
        },
    });
    assert!(gemini_stream_commit_boundary(&real_text));
}

#[test]
fn restart_body_gets_a_fresh_request_id_and_keeps_everything_else() {
    let original = serde_json::to_vec(&serde_json::json!({
        "model": "gemini-3-flash",
        "userAgent": "antigravity",
        "requestId": "zo-1111",
        "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    }))
    .unwrap();

    let replayed = body_with_fresh_request_id(&original);
    let before: Value = serde_json::from_slice(&original).unwrap();
    let after: Value = serde_json::from_slice(&replayed).unwrap();

    let fresh = after["requestId"].as_str().expect("requestId present");
    assert_ne!(
        fresh, "zo-1111",
        "the replayed attempt must carry its own requestId"
    );
    assert!(
        fresh.starts_with("zo-"),
        "fresh id must keep the zo- trace prefix: {fresh}"
    );
    for key in ["model", "userAgent", "request"] {
        assert_eq!(
            before[key], after[key],
            "field {key:?} must be semantically unchanged on replay"
        );
    }

    // Defensive fallback: bytes that are not a JSON object pass through untouched.
    assert_eq!(body_with_fresh_request_id(b"not json"), b"not json");
    assert_eq!(body_with_fresh_request_id(b"[1,2]"), b"[1,2]");
}

/// Anthropic reasoning is provider-opaque and must never reach a Gemini request:
/// the encoder drops both `thinking` and `redacted_thinking` while keeping the
/// assistant's visible text.
#[test]
fn thinking_blocks_are_dropped_from_gemini_request() {
    let request = MessageRequest {
        model: "gemini-3-flash-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage {
            role: "assistant".to_string(),
            content: vec![
                InputContentBlock::Thinking {
                    thinking: "SECRETREASONING_XYZ".to_string(),
                    signature: "THINKSIG_XYZ".to_string(),
                },
                InputContentBlock::RedactedThinking {
                    data: "REDACTEDBLOB_XYZ".to_string(),
                },
                InputContentBlock::Text {
                    text: "the answer".to_string(),
                    cache_control: None,
                },
            ],
            thought_signature: None,
            reasoning_replay: None,
        }],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let body = build_generate_content_request(&request, None);
    let serialized = serde_json::to_string(&body).expect("serialize");
    assert!(
        !serialized.contains("SECRETREASONING_XYZ"),
        "reasoning text leaked to Gemini: {serialized}"
    );
    assert!(
        !serialized.contains("THINKSIG_XYZ"),
        "thinking signature leaked to Gemini: {serialized}"
    );
    assert!(
        !serialized.contains("REDACTEDBLOB_XYZ"),
        "redacted thinking leaked to Gemini: {serialized}"
    );
    assert!(serialized.contains("the answer"), "{serialized}");
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn gemini_committed_tool_use_stream_propagates_instead_of_restarting() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let _guard = env_lock();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let server_hits = hits.clone();

    let server = tokio::spawn(async move {
        let (mut conn, _) = listener.accept().await.unwrap();
        server_hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut scratch = [0u8; 1024];
        let _ = conn.read(&mut scratch).await;
        conn.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n")
            .await
            .unwrap();
        conn.write_all(
            b"data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"write_file\",\"args\":{\"path\":\"x\",\"content\":\"y\"}}}]}}]}}\n\n"
        )
        .await
        .unwrap();
        conn.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    });

    let _idle = EnvVarGuard::set(crate::providers::STREAM_IDLE_TIMEOUT_ENV, Some("300"));
    let client = GeminiCodeAssistClient::new("token").with_retry_policy(
        3,
        std::time::Duration::from_millis(10),
        std::time::Duration::from_millis(50),
    );
    let body = serde_json::to_vec(&build_generate_content_request(&streaming_request(), None)).unwrap();
    let url = format!("http://{addr}/v1internal:streamGenerateContent?alt=sse");
    let mut stream = open_test_stream(&client, &url, body).await;

    let mut saw_tool_start = false;
    let mut saw_tool_stop = false;
    while let Some(event) = stream.next_event().await.expect("events before stall ok") {
        match event {
            StreamEvent::ContentBlockStart(start) => {
                if let OutputContentBlock::ToolUse { name, .. } = start.content_block {
                    assert_eq!(name, "write_file");
                    saw_tool_start = true;
                }
            }
            StreamEvent::ContentBlockStop(_) => {
                saw_tool_stop = true;
                break;
            }
            _ => {}
        }
    }
    assert!(saw_tool_start, "stream must surface the committing tool use start");
    assert!(saw_tool_stop, "stream must surface the committing tool use stop");
    let error = stream.next_event().await.expect_err("post-commit error");
    assert!(
        matches!(
            error,
            ApiError::StreamApi { error_type, .. }
                if error_type.as_deref() == Some("stream_idle_timeout")
        ),
        "post-commit idle must propagate"
    );
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    server.abort();
}

#[test]
fn usage_clamping_and_missing_thoughts_handling() {
    let request = MessageRequest {
        model: "gemini-3-pro-preview".to_string(),
        max_tokens: 128,
        messages: vec![InputMessage::user_text("hello")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };

    // Case 1: cache_read (40) > prompt (30) -> Clamp cache_read to prompt (30) so input_tokens = 0
    let payload_clamp = json!({
        "traceId": "trace-clamp",
        "response": {
            "modelVersion": "gemini-3-pro-preview",
            "usageMetadata": {
                "promptTokenCount": 30,
                "cachedContentTokenCount": 40,
                "candidatesTokenCount": 20
            },
            "candidates": [{
                "finishReason": "STOP",
                "content": { "parts": [{ "text": "answer" }] }
            }]
        }
    });
    let usage_clamp = normalize_generate_content_response(&request, &payload_clamp)
        .unwrap()
        .usage;
    assert_eq!(usage_clamp.input_tokens, 0);
    assert_eq!(usage_clamp.cache_read_input_tokens, 30); // clamped to prompt (30)
    assert_eq!(usage_clamp.output_tokens, 20);
    assert_eq!(usage_clamp.total_tokens(), 50);

    // Case 2: thoughtsTokenCount is missing, usage metadata should not panic and should fold 0 for thoughts.
    let payload_no_thoughts = json!({
        "traceId": "trace-no-thoughts",
        "response": {
            "modelVersion": "gemini-3-pro-preview",
            "usageMetadata": {
                "promptTokenCount": 50,
                "cachedContentTokenCount": 10,
                "candidatesTokenCount": 20
            },
            "candidates": [{
                "finishReason": "STOP",
                "content": { "parts": [{ "text": "answer" }] }
            }]
        }
    });
    let usage_no_thoughts = normalize_generate_content_response(&request, &payload_no_thoughts)
        .unwrap()
        .usage;
    assert_eq!(usage_no_thoughts.input_tokens, 40);
    assert_eq!(usage_no_thoughts.cache_read_input_tokens, 10);
    assert_eq!(usage_no_thoughts.output_tokens, 20);
    assert_eq!(usage_no_thoughts.total_tokens(), 70);

    // Case 3: usageMetadata is missing entirely.
    let payload_missing_metadata = json!({
        "traceId": "trace-missing-metadata",
        "response": {
            "modelVersion": "gemini-3-pro-preview",
            "candidates": [{
                "finishReason": "STOP",
                "content": { "parts": [{ "text": "answer" }] }
            }]
        }
    });
    let usage_missing = normalize_generate_content_response(&request, &payload_missing_metadata)
        .unwrap()
        .usage;
    assert_eq!(usage_missing.input_tokens, 0);
    assert_eq!(usage_missing.cache_read_input_tokens, 0);
    assert_eq!(usage_missing.output_tokens, 0);
    assert_eq!(usage_missing.total_tokens(), 0);
}
