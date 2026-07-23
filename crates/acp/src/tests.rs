use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use runtime::message_stream::{BlockId, RenderBlock, ToolCallId, ToolCallStatus, ToolPreview};
use runtime::permission::{
    PermissionChoice, PermissionDecision, PermissionRequest, RiskLevel,
};
use runtime::PermissionMode;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};
use tokio::task::{JoinHandle, LocalSet};
use tokio::time::timeout;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use super::{
    AcpServer, BoxFuture, PermissionRequester, RuntimeError, RuntimeFactory, RuntimeSession,
    TurnCancellation,
};

const IO_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
enum FakeScript {
    TwoTools,
    Permission,
    WaitForCancel,
}

struct FakeFactory {
    script: FakeScript,
    mode: PermissionMode,
    decisions: Arc<Mutex<Vec<PermissionDecision>>>,
    created_cwds: Arc<Mutex<Vec<PathBuf>>>,
}

impl FakeFactory {
    fn new(script: FakeScript, mode: PermissionMode) -> Arc<Self> {
        Arc::new(Self {
            script,
            mode,
            decisions: Arc::new(Mutex::new(Vec::new())),
            created_cwds: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

impl RuntimeFactory for FakeFactory {
    fn create_session(
        &self,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<Arc<dyn RuntimeSession>, RuntimeError>> {
        self.created_cwds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(cwd);
        let runtime: Arc<dyn RuntimeSession> = Arc::new(FakeSession {
            id: "fake-session".to_string(),
            script: self.script,
            mode: self.mode,
            decisions: Arc::clone(&self.decisions),
        });
        Box::pin(async move { Ok(runtime) })
    }
}

struct FakeSession {
    id: String,
    script: FakeScript,
    mode: PermissionMode,
    decisions: Arc<Mutex<Vec<PermissionDecision>>>,
}

impl RuntimeSession for FakeSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn permission_mode(&self) -> PermissionMode {
        self.mode
    }

    fn run_turn(
        &self,
        _prompt: String,
        events: tokio::sync::mpsc::Sender<RenderBlock>,
        permissions: Arc<dyn PermissionRequester>,
        cancellation: TurnCancellation,
    ) -> BoxFuture<'_, Result<(), RuntimeError>> {
        let decisions = Arc::clone(&self.decisions);
        Box::pin(async move {
            match self.script {
                FakeScript::TwoTools => emit_two_tool_turn(&events).await?,
                FakeScript::Permission => {
                    let decision = permissions
                        .request(permission_request())
                        .await
                        .map_err(|error| RuntimeError::new(error.to_string()))?;
                    decisions
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(decision);
                }
                FakeScript::WaitForCancel => {
                    send_event(
                        &events,
                        RenderBlock::TextDelta {
                            id: BlockId(1),
                            text: "waiting".to_string(),
                            done: false,
                        },
                    )
                    .await?;
                    cancellation.cancelled().await;
                    return Err(RuntimeError::new("turn aborted"));
                }
            }
            Ok(())
        })
    }
}

async fn emit_two_tool_turn(
    events: &tokio::sync::mpsc::Sender<RenderBlock>,
) -> Result<(), RuntimeError> {
    send_event(
        events,
        RenderBlock::TextDelta {
            id: BlockId(1),
            text: "hello".to_string(),
            done: false,
        },
    )
    .await?;
    for (block_id, call_id, name, summary) in [
        (2, "call-1", "read_file", "src/lib.rs"),
        (3, "call-2", "bash", "cargo test"),
    ] {
        send_event(
            events,
            tool_call(block_id, call_id, name, summary, ToolCallStatus::Running),
        )
        .await?;
        send_event(
            events,
            tool_call(block_id, call_id, name, summary, ToolCallStatus::Ok),
        )
        .await?;
    }
    Ok(())
}

fn tool_call(
    block_id: u64,
    call_id: &str,
    name: &str,
    summary: &str,
    status: ToolCallStatus,
) -> RenderBlock {
    RenderBlock::ToolCall {
        id: BlockId(block_id),
        tool_call_id: ToolCallId(call_id.to_string()),
        name: name.to_string(),
        summary: summary.to_string(),
        preview: ToolPreview::Generic {
            name: name.to_string(),
            input_summary: summary.to_string(),
        },
        status,
    }
}

async fn send_event(
    events: &tokio::sync::mpsc::Sender<RenderBlock>,
    event: RenderBlock,
) -> Result<(), RuntimeError> {
    events
        .send(event)
        .await
        .map_err(|_| RuntimeError::new("test event receiver closed"))
}

fn permission_request() -> PermissionRequest {
    PermissionRequest {
        tool: "bash".to_string(),
        input_summary: "cargo test".to_string(),
        input_hash: "abc123".to_string(),
        reasoning: "run tests".to_string(),
        choices: vec![
            PermissionChoice {
                key: 'y',
                label: "Allow once".to_string(),
                decision: PermissionDecision::AllowOnce,
            },
            PermissionChoice {
                key: 'n',
                label: "Deny".to_string(),
                decision: PermissionDecision::Deny,
            },
        ],
        risk_level: RiskLevel::Medium,
    }
}

struct WireHarness {
    writer: Option<DuplexStream>,
    reader: BufReader<DuplexStream>,
    server: JoinHandle<Result<(), agent_client_protocol::Error>>,
}

impl WireHarness {
    fn start(factory: Arc<dyn RuntimeFactory>) -> Self {
        let (client_writer, server_reader) = tokio::io::duplex(16 * 1024);
        let (server_writer, client_reader) = tokio::io::duplex(16 * 1024);
        let server = tokio::task::spawn_local(
            AcpServer::new(factory).serve(server_writer.compat_write(), server_reader.compat()),
        );
        Self {
            writer: Some(client_writer),
            reader: BufReader::new(client_reader),
            server,
        }
    }

    async fn send(&mut self, message: &Value) {
        let mut bytes = serde_json::to_vec(message).expect("test JSON must serialize");
        bytes.push(b'\n');
        self.writer
            .as_mut()
            .expect("wire writer must be open")
            .write_all(&bytes)
            .await
            .expect("wire request must write");
    }

    async fn send_raw(&mut self, bytes: &[u8]) {
        self.writer
            .as_mut()
            .expect("wire writer must be open")
            .write_all(bytes)
            .await
            .expect("raw wire request must write");
    }

    async fn recv(&mut self) -> Value {
        let mut line = String::new();
        let read = timeout(IO_TIMEOUT, self.reader.read_line(&mut line))
            .await
            .expect("timed out waiting for ACP response")
            .expect("ACP response must be readable");
        assert_ne!(read, 0, "ACP transport closed before a response arrived");
        serde_json::from_str(&line).expect("ACP response must be valid JSON")
    }

    async fn shutdown(mut self) {
        drop(self.writer.take());
        let result = timeout(IO_TIMEOUT, self.server)
            .await
            .expect("ACP server did not stop on EOF")
            .expect("ACP server task panicked");
        assert!(result.is_ok(), "ACP server failed on EOF: {result:?}");
    }
}

fn initialize_request(id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {"protocolVersion": 1, "clientCapabilities": {}}
    })
}

async fn create_session(wire: &mut WireHarness, cwd: &std::path::Path) {
    wire.send(&json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": {"cwd": cwd, "mcpServers": []}
    }))
    .await;
    let response = wire.recv().await;
    assert_eq!(response["id"], 2);
    assert_eq!(response["result"]["sessionId"], "fake-session");
}

fn prompt_request(id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "session/prompt",
        "params": {
            "sessionId": "fake-session",
            "prompt": [{"type": "text", "text": "do the work"}]
        }
    })
}

#[tokio::test(flavor = "current_thread")]
async fn initialize_advertises_only_v1_baseline_capabilities() {
    LocalSet::new()
        .run_until(async {
            let factory = FakeFactory::new(FakeScript::TwoTools, PermissionMode::Prompt);
            let mut wire = WireHarness::start(factory);
            wire.send(&initialize_request(1)).await;
            let response = wire.recv().await;

            assert_eq!(
                response,
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {
                            "loadSession": false,
                            "promptCapabilities": {
                                "image": false,
                                "audio": false,
                                "embeddedContext": false
                            },
                            "mcpCapabilities": {"http": false, "sse": false, "acp": false},
                            "sessionCapabilities": {},
                            "auth": {}
                        },
                        "authMethods": [],
                        "agentInfo": {"name": "zo", "title": "zo", "version": "0.1.0"}
                    }
                })
            );
            wire.shutdown().await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn streams_text_and_two_tool_lifecycles_in_order() {
    LocalSet::new()
        .run_until(async {
            let factory = FakeFactory::new(FakeScript::TwoTools, PermissionMode::Prompt);
            let temp = tempfile::tempdir().expect("temp cwd must be created");
            let mut wire = WireHarness::start(factory.clone());
            create_session(&mut wire, temp.path()).await;
            assert_eq!(
                *factory
                    .created_cwds
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
                vec![temp.path().to_path_buf()]
            );

            wire.send(&prompt_request(3)).await;
            let mut sequence = Vec::new();
            loop {
                let message = wire.recv().await;
                if message["id"] == 3 {
                    assert_eq!(message["result"]["stopReason"], "end_turn");
                    break;
                }
                assert_eq!(message["method"], "session/update");
                let update = &message["params"]["update"];
                let kind = update["sessionUpdate"].as_str().expect("update kind");
                let item = match kind {
                    "agent_message_chunk" => {
                        format!("text:{}", update["content"]["text"].as_str().expect("text"))
                    }
                    "tool_call" | "tool_call_update" => format!(
                        "{}:{}:{}",
                        kind,
                        update["toolCallId"].as_str().expect("tool id"),
                        update["status"].as_str().expect("tool status")
                    ),
                    other => panic!("unexpected update: {other}"),
                };
                sequence.push(item);
            }
            assert_eq!(
                sequence,
                [
                    "text:hello",
                    "tool_call:call-1:in_progress",
                    "tool_call_update:call-1:completed",
                    "tool_call:call-2:in_progress",
                    "tool_call_update:call-2:completed",
                ]
            );
            wire.shutdown().await;
        })
        .await;
}

async fn exercise_permission(option_id: &str, expected: PermissionDecision) {
    let factory = FakeFactory::new(FakeScript::Permission, PermissionMode::Prompt);
    let temp = tempfile::tempdir().expect("temp cwd must be created");
    let mut wire = WireHarness::start(factory.clone());
    create_session(&mut wire, temp.path()).await;
    wire.send(&prompt_request(3)).await;

    let request = wire.recv().await;
    assert_eq!(request["method"], "session/request_permission");
    assert_eq!(request["params"]["sessionId"], "fake-session");
    assert_eq!(request["params"]["toolCall"]["title"], "bash");
    assert_eq!(request["params"]["options"][0]["optionId"], "y");
    assert_eq!(request["params"]["options"][1]["optionId"], "n");
    wire.send(&json!({
        "jsonrpc": "2.0",
        "id": request["id"].clone(),
        "result": {"outcome": {"outcome": "selected", "optionId": option_id}}
    }))
    .await;

    let response = wire.recv().await;
    assert_eq!(response["id"], 3);
    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "unexpected permission prompt response: {response}"
    );
    assert_eq!(
        *factory
            .decisions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec![expected]
    );
    wire.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn permission_round_trip_maps_approval_and_denial() {
    LocalSet::new()
        .run_until(async {
            exercise_permission("y", PermissionDecision::AllowOnce).await;
            exercise_permission("n", PermissionDecision::Deny).await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn deterministic_permission_modes_never_prompt_the_client() {
    LocalSet::new()
        .run_until(async {
            for (mode, expected) in [
                (PermissionMode::DangerFullAccess, PermissionDecision::AllowOnce),
                (PermissionMode::ReadOnly, PermissionDecision::Deny),
            ] {
                let factory = FakeFactory::new(FakeScript::Permission, mode);
                let temp = tempfile::tempdir().expect("temp cwd must be created");
                let mut wire = WireHarness::start(factory.clone());
                create_session(&mut wire, temp.path()).await;
                wire.send(&prompt_request(3)).await;
                let response = wire.recv().await;
                assert_eq!(response["id"], 3);
                assert_eq!(response["result"]["stopReason"], "end_turn");
                assert_eq!(
                    *factory
                        .decisions
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                    vec![expected]
                );
                wire.shutdown().await;
            }
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn cancel_aborts_an_active_turn_and_returns_cancelled() {
    LocalSet::new()
        .run_until(async {
            let factory = FakeFactory::new(FakeScript::WaitForCancel, PermissionMode::Prompt);
            let temp = tempfile::tempdir().expect("temp cwd must be created");
            let mut wire = WireHarness::start(factory);
            create_session(&mut wire, temp.path()).await;
            wire.send(&prompt_request(3)).await;
            let started = wire.recv().await;
            assert_eq!(started["method"], "session/update");
            assert_eq!(started["params"]["update"]["content"]["text"], "waiting");

            wire.send(&json!({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": "fake-session"}
            }))
            .await;
            let response = wire.recv().await;
            assert_eq!(response["id"], 3);
            assert_eq!(response["result"]["stopReason"], "cancelled");
            wire.shutdown().await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn concurrent_prompt_is_rejected_without_interrupting_active_turn() {
    LocalSet::new()
        .run_until(async {
            let factory = FakeFactory::new(FakeScript::WaitForCancel, PermissionMode::Prompt);
            let temp = tempfile::tempdir().expect("temp cwd must be created");
            let mut wire = WireHarness::start(factory);
            create_session(&mut wire, temp.path()).await;
            wire.send(&prompt_request(3)).await;
            assert_eq!(wire.recv().await["method"], "session/update");

            wire.send(&prompt_request(4)).await;
            let rejected = wire.recv().await;
            assert_eq!(rejected["id"], 4);
            assert_eq!(rejected["error"]["code"], -32600);
            wire.send(&json!({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": "fake-session"}
            }))
            .await;
            assert_eq!(wire.recv().await["result"]["stopReason"], "cancelled");
            wire.shutdown().await;
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn malformed_input_returns_parse_error_and_server_stays_alive() {
    LocalSet::new()
        .run_until(async {
            let factory = FakeFactory::new(FakeScript::TwoTools, PermissionMode::Prompt);
            let mut wire = WireHarness::start(factory);
            wire.send_raw(b"{not-json}\n").await;
            let malformed = wire.recv().await;
            assert_eq!(malformed["id"], Value::Null);
            assert_eq!(malformed["error"]["code"], -32700);

            wire.send(&initialize_request(1)).await;
            let initialized = wire.recv().await;
            assert_eq!(initialized["id"], 1);
            assert_eq!(initialized["result"]["protocolVersion"], 1);
            wire.shutdown().await;
        })
        .await;
}
