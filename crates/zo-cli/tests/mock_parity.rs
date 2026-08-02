//! Parity round-trip against the in-process mock Anthropic service.
//!
//! Wires `mock-anthropic-service` — previously a declared-but-unused
//! dev-dependency (ghost infrastructure) — into automated regression. The real
//! `api::AnthropicClient` drives parity scenarios through the mock's HTTP/SSE
//! layer, so request encoding, scenario detection, and response decoding stay
//! covered by `cargo test` instead of bit-rotting.

use api::{AnthropicClient, AuthSource, InputMessage, MessageRequest};
use mock_anthropic_service::{MockAnthropicService, SCENARIO_PREFIX};

/// A non-streaming request whose user message carries the scenario marker the
/// mock's `detect_scenario` looks for.
fn scenario_request(scenario: &str) -> MessageRequest {
    MessageRequest {
        model: "claude-sonnet-4-6".to_string(),
        max_tokens: 256,
        messages: vec![InputMessage::user_text(format!(
            "{SCENARIO_PREFIX}{scenario}"
        ))],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_client_round_trips_mock_scenarios() {
    // These all resolve to a non-streaming message response (a text answer or a
    // tool-use turn), so each must come back with at least one content block.
    const SCENARIOS: [&str; 3] = [
        "token_cost_reporting",
        "read_file_roundtrip",
        "bash_stdout_roundtrip",
    ];

    let mock = MockAnthropicService::spawn()
        .await
        .expect("mock service should bind a local port");
    let client = AnthropicClient::from_auth(AuthSource::None).with_base_url(mock.base_url());

    for scenario in SCENARIOS {
        let response = client
            .send_message(&scenario_request(scenario))
            .await
            .unwrap_or_else(|error| panic!("scenario {scenario} should respond: {error}"));
        assert!(
            !response.content.is_empty(),
            "scenario {scenario} returned no content blocks"
        );
    }

    let captured = mock.captured_requests().await;
    let seen: Vec<&str> = captured
        .iter()
        .map(|request| request.scenario.as_str())
        .collect();
    assert_eq!(seen, SCENARIOS, "mock should record each scenario in order");
    assert!(
        captured
            .iter()
            .all(|request| request.method == "POST" && request.path == "/v1/messages"),
        "every parity request should POST to /v1/messages"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn mock_reports_usage_for_token_cost_scenario() {
    let mock = MockAnthropicService::spawn()
        .await
        .expect("mock service should bind a local port");
    let client = AnthropicClient::from_auth(AuthSource::None).with_base_url(mock.base_url());

    let response = client
        .send_message(&scenario_request("token_cost_reporting"))
        .await
        .expect("token cost scenario should respond");

    // The mock fabricates a non-zero usage record so cost accounting has
    // something to report against.
    assert!(
        response.usage.input_tokens > 0 || response.usage.output_tokens > 0,
        "token_cost_reporting should carry a usage record"
    );
}
