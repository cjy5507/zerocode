//! Agent Client Protocol embedding for Zo.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, AuthenticateRequest, AuthenticateResponse, CancelNotification, ContentBlock,
    ContentChunk, Implementation, InitializeRequest, InitializeResponse, MessageId, NewSessionRequest,
    NewSessionResponse, PermissionOption, PermissionOptionKind, PromptRequest, PromptResponse,
    RequestPermissionOutcome, RequestPermissionRequest, SessionId, SessionNotification,
    SessionUpdate, StopReason, TextContent, ToolCall, ToolCallStatus as AcpToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields,
};
use agent_client_protocol::{
    Agent, Channel, Client, ConnectTo, ConnectionTo, Error, RawJsonRpcMessage, Role,
};
use futures_util::StreamExt;
use futures_util::future::{Either, select};
use futures_util::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use runtime::PermissionMode;
use runtime::message_stream::{
    ProjectedRenderBlock, RenderBlock, ToolCallStatus, project_render_block,
};
use runtime::permission::{PermissionDecision, PermissionError, PermissionRequest};
use tokio::sync::{Notify, mpsc};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

const CLEAN_EOF_MESSAGE: &str = "ACP input reached EOF";

/// Boxed future used by the runtime adapter traits.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Error returned by a Zo runtime adapter.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct RuntimeError {
    message: String,
}

impl RuntimeError {
    /// Creates a runtime error with a user-safe message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Factory for one independent Zo runtime per ACP session.
pub trait RuntimeFactory: Send + Sync {
    /// Builds a session rooted at the ACP-supplied working directory.
    fn create_session(
        &self,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<Arc<dyn RuntimeSession>, RuntimeError>>;
}

/// Thin, testable seam over a Zo headless session.
pub trait RuntimeSession: Send + Sync {
    /// Returns the Zo session identifier exposed to the ACP client.
    fn id(&self) -> &str;

    /// Returns the resolved Zo permission mode for this session.
    fn permission_mode(&self) -> PermissionMode;

    /// Runs one turn and emits the same [`RenderBlock`] stream used by Zo serve.
    fn run_turn(
        &self,
        prompt: String,
        events: mpsc::Sender<RenderBlock>,
        permissions: Arc<dyn PermissionRequester>,
        cancellation: TurnCancellation,
    ) -> BoxFuture<'_, Result<(), RuntimeError>>;
}

/// Turn-scoped permission callback supplied to a runtime session.
pub trait PermissionRequester: Send + Sync {
    /// Resolves one runtime permission request.
    fn request(
        &self,
        request: PermissionRequest,
    ) -> BoxFuture<'_, Result<PermissionDecision, PermissionError>>;
}

/// Cancellation signal shared by the ACP state machine and runtime adapter.
#[derive(Clone, Default)]
pub struct TurnCancellation {
    inner: Arc<CancellationInner>,
}

#[derive(Default)]
struct CancellationInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl TurnCancellation {
    /// Signals cancellation. Repeated calls are harmless.
    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::SeqCst) {
            self.inner.notify.notify_waiters();
        }
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Waits until cancellation is requested.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// ACP server state machine parameterized by a Zo runtime factory.
pub struct AcpServer {
    state: Arc<ServerState>,
}

struct ServerState {
    factory: Arc<dyn RuntimeFactory>,
    sessions: Mutex<HashMap<String, Arc<SessionEntry>>>,
    next_permission_id: Arc<AtomicU64>,
}

struct SessionEntry {
    runtime: Arc<dyn RuntimeSession>,
    in_flight: AtomicBool,
    cancellation: Mutex<Option<TurnCancellation>>,
}

impl AcpServer {
    /// Creates an ACP server backed by the supplied runtime factory.
    #[must_use]
    pub fn new(factory: Arc<dyn RuntimeFactory>) -> Self {
        Self {
            state: Arc::new(ServerState {
                factory,
                sessions: Mutex::new(HashMap::new()),
                next_permission_id: Arc::new(AtomicU64::new(1)),
            }),
        }
    }

    /// Serves ACP over newline-delimited standard input and output.
    pub async fn serve_stdio(self) -> Result<(), agent_client_protocol::Error> {
        self.serve(
            tokio::io::stdout().compat_write(),
            tokio::io::stdin().compat(),
        )
        .await
    }

    /// Serves ACP over newline-delimited byte streams until EOF or transport failure.
    pub async fn serve<Output, Input>(
        self,
        output: Output,
        input: Input,
    ) -> Result<(), agent_client_protocol::Error>
    where
        Output: AsyncWrite + Send + Unpin + 'static,
        Input: AsyncRead + Send + Unpin + 'static,
    {
        let new_session_state = Arc::clone(&self.state);
        let prompt_state = Arc::clone(&self.state);
        let cancel_state = Arc::clone(&self.state);

        let result = Agent
            .builder()
            .name("zo-acp")
            .on_receive_request(
                async move |request: InitializeRequest, responder, _connection| {
                    responder.respond(initialize_response(request.protocol_version))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_request: AuthenticateRequest, responder, _connection| {
                    responder.respond(AuthenticateResponse::new())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |request: NewSessionRequest, responder, connection| {
                    let state = Arc::clone(&new_session_state);
                    connection.spawn(async move {
                        match create_session(&state, request).await {
                            Ok(response) => responder.respond(response),
                            Err(error) => responder.respond_with_error(error),
                        }
                    })?;
                    Ok(())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |request: PromptRequest, responder, connection| {
                    let state = Arc::clone(&prompt_state);
                    let prompt_connection = connection.clone();
                    connection.spawn(async move {
                        match run_prompt(&state, request, prompt_connection).await {
                            Ok(response) => responder.respond(response),
                            Err(error) => responder.respond_with_error(error),
                        }
                    })?;
                    Ok(())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_notification(
                async move |notification: CancelNotification, _connection| {
                    cancel_turn(&cancel_state, &notification.session_id);
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_to(CleanByteStreams { output, input })
            .await;
        match result {
            Err(error) if error.message == CLEAN_EOF_MESSAGE => Ok(()),
            result => result,
        }
    }
}

struct CleanByteStreams<Output, Input> {
    output: Output,
    input: Input,
}

impl<Output, Input, R> ConnectTo<R> for CleanByteStreams<Output, Input>
where
    Output: AsyncWrite + Send + Unpin + 'static,
    Input: AsyncRead + Send + Unpin + 'static,
    R: Role,
{
    async fn connect_to(
        self,
        client: impl ConnectTo<R::Counterpart>,
    ) -> Result<(), agent_client_protocol::Error> {
        let (channel, transport) =
            <Self as ConnectTo<R>>::into_channel_and_future(self);
        let client = Box::pin(client.connect_to(channel));
        match select(client, transport).await {
            Either::Left((result, _)) | Either::Right((result, _)) => result,
        }
    }

    fn into_channel_and_future(
        self,
    ) -> (
        Channel,
        BoxFuture<'static, Result<(), agent_client_protocol::Error>>,
    ) {
        let Self { output, input } = self;
        let (caller, transport) = Channel::duplex();
        let Channel { mut rx, tx } = transport;
        let input = async move {
            let mut lines = BufReader::new(input).lines();
            while let Some(line) = lines.next().await {
                let line = line.map_err(io_error)?;
                match serde_json::from_str::<RawJsonRpcMessage>(&line) {
                    Ok(message) => tx.unbounded_send(Ok(message)).map_err(io_error)?,
                    Err(_) => tx
                        .unbounded_send(Err(Error::parse_error().data(serde_json::json!({
                            "line": line
                        }))))
                        .map_err(io_error)?,
                }
            }
            Err(Error::new(-32_099, CLEAN_EOF_MESSAGE))
        };
        let output = async move {
            let mut output = output;
            while let Some(message) = rx.next().await {
                let mut line = serde_json::to_vec(&message?)
                    .map_err(|error| Error::internal_error().data(error.to_string()))?;
                line.push(b'\n');
                output.write_all(&line).await.map_err(io_error)?;
                output.flush().await.map_err(io_error)?;
            }
            Ok(())
        };
        let future = Box::pin(async move {
            match select(Box::pin(input), Box::pin(output)).await {
                Either::Left((result, _)) | Either::Right((result, _)) => result,
            }
        });
        (caller, future)
    }
}

fn io_error(error: impl std::fmt::Display) -> Error {
    Error::internal_error().data(error.to_string())
}

fn initialize_response(_requested: ProtocolVersion) -> InitializeResponse {
    InitializeResponse::new(ProtocolVersion::V1)
        .agent_capabilities(AgentCapabilities::new())
        .agent_info(Implementation::new("zo", env!("CARGO_PKG_VERSION")).title("zo"))
}

async fn create_session(
    state: &Arc<ServerState>,
    request: NewSessionRequest,
) -> Result<NewSessionResponse, Error> {
    if !request.cwd.is_absolute() {
        return Err(Error::invalid_params().data("session cwd must be an absolute path"));
    }
    if !request.cwd.is_dir() {
        return Err(Error::invalid_params().data("session cwd must be an existing directory"));
    }
    if !request.additional_directories.is_empty() {
        return Err(Error::invalid_params().data("additionalDirectories is not supported"));
    }
    if !request.mcp_servers.is_empty() {
        return Err(Error::invalid_params().data("ACP-provided MCP servers are not supported"));
    }

    let runtime = state
        .factory
        .create_session(request.cwd)
        .await
        .map_err(|error| Error::internal_error().data(error.to_string()))?;
    let session_id = runtime.id().to_string();
    let entry = Arc::new(SessionEntry {
        runtime,
        in_flight: AtomicBool::new(false),
        cancellation: Mutex::new(None),
    });
    let replaced = state
        .sessions
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(session_id.clone(), entry);
    if replaced.is_some() {
        return Err(Error::internal_error().data("runtime returned a duplicate session id"));
    }
    Ok(NewSessionResponse::new(session_id))
}

async fn run_prompt(
    state: &Arc<ServerState>,
    request: PromptRequest,
    connection: ConnectionTo<Client>,
) -> Result<PromptResponse, Error> {
    let session_id = request.session_id.0.to_string();
    let entry = state
        .sessions
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&session_id)
        .cloned()
        .ok_or_else(|| Error::resource_not_found(Some(session_id.clone())))?;

    if entry
        .in_flight
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(Error::invalid_request().data("session already has an in-flight prompt"));
    }
    let turn_guard = ActiveTurnGuard::new(Arc::clone(&entry));
    let prompt = prompt_text(request.prompt)?;
    let cancellation = TurnCancellation::default();
    *entry
        .cancellation
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cancellation.clone());

    let permissions: Arc<dyn PermissionRequester> = Arc::new(ClientPermissionRequester {
        connection: connection.clone(),
        session_id: request.session_id.clone(),
        mode: entry.runtime.permission_mode(),
        next_id: Arc::clone(&state.next_permission_id),
    });
    let (event_tx, mut event_rx) = mpsc::channel(64);
    let turn = entry
        .runtime
        .run_turn(prompt, event_tx, permissions, cancellation.clone());
    tokio::pin!(turn);
    let mut tool_state = ToolProjectionState::default();
    let mut events_open = true;

    let turn_result = loop {
        tokio::select! {
            event = event_rx.recv(), if events_open => {
                match event {
                    Some(event) => send_projected_update(
                        &connection,
                        &request.session_id,
                        &event,
                        &mut tool_state,
                    )?,
                    None => events_open = false,
                }
            }
            result = &mut turn => break result,
        }
    };
    while let Ok(event) = event_rx.try_recv() {
        send_projected_update(
            &connection,
            &request.session_id,
            &event,
            &mut tool_state,
        )?;
    }
    drop(turn_guard);

    if cancellation.is_cancelled() {
        return Ok(PromptResponse::new(StopReason::Cancelled));
    }
    match turn_result {
        Ok(()) => Ok(PromptResponse::new(StopReason::EndTurn)),
        Err(error) => Err(Error::internal_error().data(error.to_string())),
    }
}

fn prompt_text(prompt: Vec<ContentBlock>) -> Result<String, Error> {
    let mut parts = Vec::with_capacity(prompt.len());
    for block in prompt {
        match block {
            ContentBlock::Text(text) => parts.push(text.text),
            ContentBlock::ResourceLink(link) => {
                parts.push(format!("[{}]({})", link.name, link.uri));
            }
            _ => {
                return Err(Error::invalid_params()
                    .data("only text and resource_link prompt content is supported"));
            }
        }
    }
    if parts.is_empty() {
        return Err(Error::invalid_params().data("prompt must contain at least one content block"));
    }
    Ok(parts.join("\n\n"))
}

fn cancel_turn(state: &ServerState, session_id: &SessionId) {
    let entry = state
        .sessions
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(session_id.0.as_ref())
        .cloned();
    if let Some(entry) = entry {
        let cancellation = entry
            .cancellation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
        }
    }
}

struct ActiveTurnGuard {
    entry: Arc<SessionEntry>,
}

impl ActiveTurnGuard {
    fn new(entry: Arc<SessionEntry>) -> Self {
        Self { entry }
    }
}

impl Drop for ActiveTurnGuard {
    fn drop(&mut self) {
        *self
            .entry
            .cancellation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        self.entry.in_flight.store(false, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct ToolProjectionState {
    announced: HashSet<String>,
    terminal: HashSet<String>,
}

fn send_projected_update(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    block: &RenderBlock,
    tools: &mut ToolProjectionState,
) -> Result<(), Error> {
    let update = match project_render_block(block) {
        ProjectedRenderBlock::TextDelta { id, text, .. } if !text.is_empty() => {
            Some(SessionUpdate::AgentMessageChunk(
                ContentChunk::new(ContentBlock::Text(TextContent::new(text)))
                    .message_id(MessageId::new(id.to_string())),
            ))
        }
        ProjectedRenderBlock::ToolCall {
            tool_call_id,
            name,
            summary,
            status,
            ..
        } => project_tool_call(tools, tool_call_id, name, summary, status),
        ProjectedRenderBlock::ToolResult {
            tool_call_id,
            is_error,
            ..
        } => project_tool_result(tools, tool_call_id, is_error),
        ProjectedRenderBlock::TextDelta { .. } | ProjectedRenderBlock::Other => None,
    };
    if let Some(update) = update {
        connection.send_notification(SessionNotification::new(session_id.clone(), update))?;
    }
    Ok(())
}

fn project_tool_call(
    tools: &mut ToolProjectionState,
    tool_call_id: &str,
    name: &str,
    summary: &str,
    status: ToolCallStatus,
) -> Option<SessionUpdate> {
    let acp_status = acp_tool_status(status);
    let terminal = is_terminal_tool_status(status);
    if tools.announced.insert(tool_call_id.to_string()) {
        if terminal {
            tools.terminal.insert(tool_call_id.to_string());
        }
        return Some(SessionUpdate::ToolCall(
            ToolCall::new(tool_call_id.to_string(), tool_title(name, summary)).status(acp_status),
        ));
    }
    if terminal && !tools.terminal.insert(tool_call_id.to_string()) {
        return None;
    }
    Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
        tool_call_id.to_string(),
        ToolCallUpdateFields::new()
            .title(tool_title(name, summary))
            .status(acp_status),
    )))
}

fn project_tool_result(
    tools: &mut ToolProjectionState,
    tool_call_id: &str,
    is_error: bool,
) -> Option<SessionUpdate> {
    if !tools.terminal.insert(tool_call_id.to_string()) {
        return None;
    }
    let status = if is_error {
        AcpToolCallStatus::Failed
    } else {
        AcpToolCallStatus::Completed
    };
    Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
        tool_call_id.to_string(),
        ToolCallUpdateFields::new().status(status),
    )))
}

fn tool_title(name: &str, summary: &str) -> String {
    if summary.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {summary}")
    }
}

const fn acp_tool_status(status: ToolCallStatus) -> AcpToolCallStatus {
    match status {
        ToolCallStatus::Pending => AcpToolCallStatus::Pending,
        ToolCallStatus::Running => AcpToolCallStatus::InProgress,
        ToolCallStatus::Ok => AcpToolCallStatus::Completed,
        ToolCallStatus::Errored | ToolCallStatus::Cancelled => AcpToolCallStatus::Failed,
    }
}

const fn is_terminal_tool_status(status: ToolCallStatus) -> bool {
    matches!(
        status,
        ToolCallStatus::Ok | ToolCallStatus::Errored | ToolCallStatus::Cancelled
    )
}

struct ClientPermissionRequester {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    mode: PermissionMode,
    next_id: Arc<AtomicU64>,
}

impl PermissionRequester for ClientPermissionRequester {
    fn request(
        &self,
        request: PermissionRequest,
    ) -> BoxFuture<'_, Result<PermissionDecision, PermissionError>> {
        Box::pin(async move {
            match self.mode {
                PermissionMode::ReadOnly => return Ok(PermissionDecision::Deny),
                PermissionMode::DangerFullAccess | PermissionMode::Allow => {
                    return Ok(PermissionDecision::AllowOnce);
                }
                PermissionMode::WorkspaceWrite | PermissionMode::Prompt => {}
            }
            if request.choices.is_empty() {
                return Ok(PermissionDecision::Deny);
            }

            let permission_id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let tool_call_id = format!("permission-{permission_id}");
            let tool_call = ToolCallUpdate::new(
                tool_call_id,
                ToolCallUpdateFields::new()
                    .title(request.tool.clone())
                    .status(AcpToolCallStatus::Pending),
            );
            let options = request
                .choices
                .iter()
                .map(|choice| {
                    PermissionOption::new(
                        choice.key.to_string(),
                        choice.label.clone(),
                        permission_option_kind(choice.decision),
                    )
                })
                .collect();
            let response = self
                .connection
                .send_request(RequestPermissionRequest::new(
                    self.session_id.clone(),
                    tool_call,
                    options,
                ))
                .block_task()
                .await
                .map_err(|error| PermissionError::Adapter {
                    source_name: "acp",
                    message: error.to_string(),
                })?;
            match response.outcome {
                RequestPermissionOutcome::Selected(selected) => request
                    .choices
                    .iter()
                    .find(|choice| choice.key.to_string() == selected.option_id.0.as_ref())
                    .map_or(Ok(PermissionDecision::Deny), |choice| Ok(choice.decision)),
                _ => Ok(PermissionDecision::Deny),
            }
        })
    }
}

const fn permission_option_kind(decision: PermissionDecision) -> PermissionOptionKind {
    match decision {
        PermissionDecision::Allow => PermissionOptionKind::AllowAlways,
        PermissionDecision::AllowOnce => PermissionOptionKind::AllowOnce,
        PermissionDecision::Deny => PermissionOptionKind::RejectOnce,
    }
}

#[cfg(test)]
mod tests;
