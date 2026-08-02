//! L7c integration smoke tests covering the permission-bridge round
//! trip used by the TUI driver loop.
//!
//! These tests intentionally avoid bringing up a real ratatui terminal
//! or live HTTP transport — those surfaces are exercised by the
//! existing `tui_*` and `mock_parity_harness` test families. This file
//! anchors the L7c contract that the L3 `ChannelPrompter` round-trips
//! cleanly through the TUI render channel into a TUI-style decision
//! and back to the agent loop.
//!
//! ## Living standard
//!
//! 1. Tests are named `<area>_<scenario>` per the L1 convention.
//! 2. No external network or process I/O.
//! 3. No `unwrap` on user-visible error paths — assertions only.

use runtime::message_stream::{
    BlockIdGen, PermissionDecision as RenderPermissionDecision, RenderBlock,
};
use runtime::permission::{
    ChannelPrompter, PermissionChoice as L3PermissionChoice, PermissionDecision as L3Decision,
    PermissionPrompter, PermissionRequest, RiskLevel,
};
use tokio::sync::mpsc;

/// Shared sample request used across the round-trip tests.
fn sample_request() -> PermissionRequest {
    PermissionRequest {
        tool: "bash".to_string(),
        input_summary: "ls".to_string(),
        input_hash: "abc123".to_string(),
        reasoning: "list current directory".to_string(),
        choices: vec![
            L3PermissionChoice {
                key: 'y',
                label: "Allow once".to_string(),
                decision: L3Decision::AllowOnce,
            },
            L3PermissionChoice {
                key: 'a',
                label: "Allow for session".to_string(),
                decision: L3Decision::Allow,
            },
            L3PermissionChoice {
                key: 'n',
                label: "Deny".to_string(),
                decision: L3Decision::Deny,
            },
        ],
        risk_level: RiskLevel::Low,
    }
}

/// Drive a permission request through a hand-rolled bridge that
/// mimics the production [`session::permission_bridge::run_permission_pump`]
/// to keep this test independent of the bin's private modules.
async fn pump_one_request(
    mut request_rx: mpsc::Receiver<(PermissionRequest, runtime::permission::OneshotResponder)>,
    render_tx: mpsc::Sender<RenderBlock>,
    ids: BlockIdGen,
) {
    while let Some((request, responder)) = request_rx.recv().await {
        let (modal_tx, modal_rx) = tokio::sync::oneshot::channel();
        let prompt = runtime::message_stream::PermissionPrompt {
            id: ids.next(),
            tool_call_id: runtime::message_stream::ToolCallId(String::new()),
            tool_name: request.tool.clone(),
            reasoning: request.reasoning.clone(),
            audit_hint: Some(request.audit_hint()),
            choices: request
                .choices
                .iter()
                .map(|c| runtime::message_stream::PermissionChoice {
                    key: c.key,
                    label: c.label.clone(),
                    decision: match c.decision {
                        L3Decision::Allow => RenderPermissionDecision::AllowAlways,
                        L3Decision::AllowOnce => RenderPermissionDecision::AllowOnce,
                        L3Decision::Deny => RenderPermissionDecision::Deny,
                    },
                })
                .collect(),
            responder: modal_tx,
        };
        if render_tx
            .send(RenderBlock::PermissionPrompt(prompt))
            .await
            .is_err()
        {
            return;
        }
        match modal_rx.await {
            Ok(decision) => {
                let l3 = match decision {
                    RenderPermissionDecision::AllowOnce => L3Decision::AllowOnce,
                    RenderPermissionDecision::AllowAlways => L3Decision::Allow,
                    RenderPermissionDecision::Deny | RenderPermissionDecision::DenyAlways => {
                        L3Decision::Deny
                    }
                };
                let _ = responder.respond(l3);
            }
            Err(_) => drop(responder),
        }
    }
}

#[tokio::test]
async fn session_integration_permission_round_trip_allow_always() {
    let (prompter, request_rx) = ChannelPrompter::new(4);
    let (render_tx, mut render_rx) = mpsc::channel::<RenderBlock>(8);
    let pump = tokio::spawn(pump_one_request(
        request_rx,
        render_tx,
        BlockIdGen::default(),
    ));

    let prompter_clone = prompter.clone();
    let decide = tokio::spawn(async move { prompter_clone.decide(sample_request()).await });

    // Simulated TUI: pull the prompt off the render channel and
    // resolve it with `AllowAlways`.
    let block = render_rx.recv().await.expect("prompt arrives");
    let responder = match block {
        RenderBlock::PermissionPrompt(prompt) => prompt.responder,
        other => panic!("unexpected render block: {other:?}"),
    };
    responder
        .send(RenderPermissionDecision::AllowAlways)
        .expect("responder live");

    let decision = decide.await.expect("decide join").expect("decide ok");
    assert_eq!(decision, L3Decision::Allow);

    drop(prompter);
    pump.await.expect("pump join");
}

#[tokio::test]
async fn session_integration_permission_round_trip_deny() {
    let (prompter, request_rx) = ChannelPrompter::new(4);
    let (render_tx, mut render_rx) = mpsc::channel::<RenderBlock>(8);
    let pump = tokio::spawn(pump_one_request(
        request_rx,
        render_tx,
        BlockIdGen::default(),
    ));

    let prompter_clone = prompter.clone();
    let decide = tokio::spawn(async move { prompter_clone.decide(sample_request()).await });

    let block = render_rx.recv().await.expect("prompt arrives");
    let responder = match block {
        RenderBlock::PermissionPrompt(prompt) => prompt.responder,
        other => panic!("unexpected render block: {other:?}"),
    };
    responder
        .send(RenderPermissionDecision::Deny)
        .expect("responder live");

    let decision = decide.await.expect("decide join").expect("decide ok");
    assert_eq!(decision, L3Decision::Deny);

    drop(prompter);
    pump.await.expect("pump join");
}

#[tokio::test]
async fn session_integration_dropping_modal_responder_denies() {
    let (prompter, request_rx) = ChannelPrompter::new(4);
    let (render_tx, mut render_rx) = mpsc::channel::<RenderBlock>(8);
    let pump = tokio::spawn(pump_one_request(
        request_rx,
        render_tx,
        BlockIdGen::default(),
    ));

    let prompter_clone = prompter.clone();
    let decide = tokio::spawn(async move { prompter_clone.decide(sample_request()).await });

    let block = render_rx.recv().await.expect("prompt arrives");
    let responder = match block {
        RenderBlock::PermissionPrompt(prompt) => prompt.responder,
        other => panic!("unexpected render block: {other:?}"),
    };
    drop(responder);

    let decide_result = decide.await.expect("decide join");
    assert!(decide_result.is_err(), "dropped responder is hard deny");

    drop(prompter);
    pump.await.expect("pump join");
}
