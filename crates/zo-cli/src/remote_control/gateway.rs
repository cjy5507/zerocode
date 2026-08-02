use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::header::{
    self, CACHE_CONTROL, CONTENT_SECURITY_POLICY, CONTENT_TYPE, HeaderName, HeaderValue,
    REFERRER_POLICY, SET_COOKIE,
};
use axum::http::{HeaderMap, Method, Request, Response, StatusCode};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::{get, post, put};
use axum::Router;
use futures_util::{Sink, SinkExt, StreamExt};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder, Rgb, RgbImage};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::sensitive_headers::{
    SetSensitiveRequestHeadersLayer, SetSensitiveResponseHeadersLayer,
};
use tower_http::trace::TraceLayer;

use super::protocol::{
    ClientMessage, ControlRole, MAX_APPROVAL_ID_BYTES, MAX_PROMPT_BYTES, PROTOCOL_VERSION,
    PairStartRequest, PairStartResponse, PairStatusResponse, PromptMode, ServerMessage,
    ToolApprovalDecision, ToolApprovalSource, TurnPhase,
};
use super::push::{
    PushSubscriptionRequest, SubscriptionValidationError, validate_subscription,
};
use super::state::{
    AuthenticatedDevice, CommandDecision, ControllerGrace, PairPoll, PairingError, RemoteEffect,
    RemoteShared, SnapshotPlan, ToolApprovalAttempt,
};

const MAX_HTTP_BODY_BYTES: usize = 64 * 1024;
const MAX_PUSH_BODY_BYTES: usize = 8 * 1024;
const MAX_WS_FRAME_BYTES: usize = 64 * 1024;
const WS_CLOSE_GOING_AWAY: u16 = 1001;
const WS_CLOSE_NO_STATUS: u16 = 1005;
const WS_CLOSE_ABNORMAL: u16 = 1006;
const WS_CLOSE_SESSION_REVOKED: u16 = 4001;
const WS_WRITE_TIMEOUT: Duration = Duration::from_millis(500);
const CONTROLLER_DISCONNECT_GRACE: Duration = Duration::from_secs(30);
const SESSION_COOKIE: &str = "zo_remote_session";

#[derive(Clone)]
pub(crate) struct GatewayState {
    shared: RemoteShared,
    base_path: Arc<str>,
    expected_host: Arc<str>,
    expected_origin: Arc<str>,
    cancellation: CancellationToken,
    debug_logging: bool,
}

impl GatewayState {
    pub(crate) fn new(
        shared: RemoteShared,
        expected_host: String,
        expected_origin: String,
        base_path: String,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            shared,
            base_path: base_path.into(),
            expected_host: expected_host.into(),
            expected_origin: expected_origin.into(),
            cancellation,
            debug_logging: std::env::var("ZO_REMOTE_DEBUG").as_deref() == Ok("1"),
        }
    }
}

pub(crate) async fn serve(
    listener: TcpListener,
    state: GatewayState,
) -> Result<(), std::io::Error> {
    let cancellation = state.cancellation.clone();
    let app = router(state);
    axum::serve(listener, app)
        .with_graceful_shutdown(cancellation.cancelled_owned())
        .await
}

fn router(state: GatewayState) -> Router {
    let debug_logging = state.debug_logging;
    let app = if state.base_path.is_empty() {
        gateway_routes()
    } else {
        let mounted_root = format!("{}/", state.base_path);
        Router::new()
            .merge(gateway_routes())
            .nest(state.base_path.as_ref(), gateway_routes())
            .route(&mounted_root, get(index))
    };
    app
        .fallback(not_found)
        .layer(RequestBodyLimitLayer::new(MAX_HTTP_BODY_BYTES))
        .layer(SetSensitiveRequestHeadersLayer::new([header::COOKIE]))
        .layer(SetSensitiveResponseHeadersLayer::new([SET_COOKIE]))
        .layer(TraceLayer::new_for_http())
        .layer(middleware::from_fn_with_state(
            debug_logging,
            access_log,
        ))
        .with_state(state)
}

async fn access_log(
    State(debug_logging): State<bool>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    if !debug_logging {
        return next.run(request).await;
    }
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let started = Instant::now();
    let response = next.run(request).await;
    eprintln!(
        "{}",
        format_access_log(&method, &path, response.status(), started.elapsed())
    );
    response
}

fn format_access_log(
    method: &Method,
    path: &str,
    status: StatusCode,
    elapsed: Duration,
) -> String {
    format!(
        "[remote] {method} {path} -> {} {}ms",
        status.as_u16(),
        elapsed.as_millis()
    )
}

fn format_ws_log(device_id: &str, event: &str) -> String {
    let short_id = device_id.chars().take(8).collect::<String>();
    format!("[remote] ws {short_id} {event}")
}

fn log_ws(debug_logging: bool, device_id: &str, event: &str) {
    if debug_logging {
        eprintln!("{}", format_ws_log(device_id, event));
    }
}

fn gateway_routes() -> Router<GatewayState> {
    Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/remote-state.js", get(remote_state_js))
        .route("/styles.css", get(styles))
        .route("/manifest.webmanifest", get(manifest))
        .route("/sw.js", get(service_worker))
        .route("/icon.svg", get(icon_svg))
        .route("/apple-touch-icon.png", get(apple_touch_icon))
        .route("/api/pair", post(pair_start))
        .route("/api/pair/{id}", get(pair_status))
        .route("/api/push/config", get(push_config))
        .route(
            "/api/push/subscription",
            put(push_subscription_put)
                .delete(push_subscription_delete)
                .layer(RequestBodyLimitLayer::new(MAX_PUSH_BODY_BYTES)),
        )
        .route("/ws", get(websocket_upgrade))
}

async fn index(State(state): State<GatewayState>, headers: HeaderMap) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/index.html"),
        "text/html; charset=utf-8",
        "no-store",
    )
}

async fn app_js(State(state): State<GatewayState>, headers: HeaderMap) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/app.js"),
        "text/javascript; charset=utf-8",
        "public, max-age=300",
    )
}

async fn remote_state_js(
    State(state): State<GatewayState>,
    headers: HeaderMap,
) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/remote-state.js"),
        "text/javascript; charset=utf-8",
        "public, max-age=300",
    )
}

async fn styles(State(state): State<GatewayState>, headers: HeaderMap) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/styles.css"),
        "text/css; charset=utf-8",
        "public, max-age=300",
    )
}

async fn manifest(State(state): State<GatewayState>, headers: HeaderMap) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/manifest.webmanifest"),
        "application/manifest+json",
        "public, max-age=300",
    )
}

async fn service_worker(State(state): State<GatewayState>, headers: HeaderMap) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/sw.js"),
        "text/javascript; charset=utf-8",
        "no-store",
    )
}

async fn icon_svg(State(state): State<GatewayState>, headers: HeaderMap) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset(
        &state,
        include_str!("../../remote-web/icon.svg"),
        "image/svg+xml",
        "public, max-age=300",
    )
}

async fn apple_touch_icon(
    State(state): State<GatewayState>,
    headers: HeaderMap,
) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    asset_bytes(
        &state,
        apple_touch_icon_png(),
        "image/png",
        "public, max-age=300",
    )
}

async fn push_config(
    State(state): State<GatewayState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(code) = authenticate_push_request(&state, &headers, false) {
        return status(code);
    }
    match state.shared.push_server_key() {
        Some(server_key) => Json(json!({
            "push": { "server_key": server_key }
        }))
        .into_response(),
        None => Json(json!({ "push": null })).into_response(),
    }
}

async fn push_subscription_put(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    request: Result<Json<PushSubscriptionRequest>, JsonRejection>,
) -> Response<Body> {
    let device = match authenticate_push_request(&state, &headers, true) {
        Ok(device) => device,
        Err(code) => return status(code),
    };
    if !state.shared.push_enabled() {
        return status(StatusCode::NOT_FOUND);
    }
    let Ok(Json(request)) = request else {
        return status(StatusCode::BAD_REQUEST);
    };
    let subscription = match validate_subscription(request) {
        Ok(subscription) => subscription,
        Err(SubscriptionValidationError::Malformed) => {
            return status(StatusCode::BAD_REQUEST);
        }
        Err(SubscriptionValidationError::EndpointNotAllowed) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({ "error": "push_endpoint_not_allowed" })),
            )
                .into_response();
        }
    };
    if !state
        .shared
        .replace_push_subscription(&device.id, subscription)
    {
        return status(StatusCode::UNAUTHORIZED);
    }
    status(StatusCode::NO_CONTENT)
}

async fn push_subscription_delete(
    State(state): State<GatewayState>,
    headers: HeaderMap,
) -> Response<Body> {
    let device = match authenticate_push_request(&state, &headers, true) {
        Ok(device) => device,
        Err(code) => return status(code),
    };
    if !state.shared.push_enabled() {
        return status(StatusCode::NOT_FOUND);
    }
    state.shared.remove_push_subscription(&device.id);
    status(StatusCode::NO_CONTENT)
}

async fn pair_start(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(request): Json<PairStartRequest>,
) -> Response<Body> {
    if !valid_host_and_origin(&state, &headers) {
        return json_error(
            StatusCode::FORBIDDEN,
            "invalid_origin",
            "The request origin is not allowed.",
        );
    }
    match state
        .shared
        .begin_pairing(&request.secret, &request.device_name)
    {
        Ok(started) => (
            StatusCode::ACCEPTED,
            Json(PairStartResponse {
                pairing_id: started.id,
                comparison_code: started.comparison_code,
                expires_in_seconds: started.expires_in_seconds,
                poll_expires_in_seconds: started.poll_expires_in_seconds,
            }),
        )
            .into_response(),
        Err(error) => pairing_error(&error),
    }
}

async fn pair_status(
    State(state): State<GatewayState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response<Body> {
    if !valid_host(&state, &headers) {
        return status(StatusCode::MISDIRECTED_REQUEST);
    }
    match state.shared.pairing_status(&id) {
        PairPoll::Pending => Json(PairStatusResponse::Pending).into_response(),
        PairPoll::Denied => Json(PairStatusResponse::Denied).into_response(),
        PairPoll::Expired => (
            StatusCode::GONE,
            Json(PairStatusResponse::Expired),
        )
            .into_response(),
        PairPoll::Approved { token, role } => {
            let mut response = Json(PairStatusResponse::Approved { role }).into_response();
            let cookie_path = if state.base_path.is_empty() {
                "/"
            } else {
                state.base_path.as_ref()
            };
            let cookie = format!("{SESSION_COOKIE}={token}; Path={cookie_path}; Max-Age=28800; Secure; HttpOnly; SameSite=Strict");
            if let Ok(value) = HeaderValue::from_str(&cookie) {
                response.headers_mut().insert(SET_COOKIE, value);
            }
            response
        }
    }
}

async fn websocket_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<GatewayState>,
    headers: HeaderMap,
) -> Response<Body> {
    if !valid_host_and_origin(&state, &headers) {
        return status(StatusCode::FORBIDDEN);
    }
    let device = cookie(&headers, SESSION_COOKIE)
        .and_then(|token| state.shared.authenticate(token));
    let cancellation = state.cancellation.child_token();
    let ws = ws
        .protocols(["zo.remote.v1"])
        .max_frame_size(MAX_WS_FRAME_BYTES)
        .max_message_size(MAX_WS_FRAME_BYTES);
    match device {
        Some(device) => {
            let shared = state.shared.clone();
            let debug_logging = state.debug_logging;
            ws.on_upgrade(move |socket| {
                websocket(socket, shared, device, cancellation, debug_logging)
            })
        }
        None => ws.on_upgrade(move |socket| rejected_websocket(socket, cancellation)),
    }
}

async fn rejected_websocket(mut socket: WebSocket, cancellation: CancellationToken) {
    let _ = send_server_message(&mut socket, &session_revoked(), &cancellation).await;
}

async fn websocket(
    mut socket: WebSocket,
    shared: RemoteShared,
    device: AuthenticatedDevice,
    cancellation: CancellationToken,
    debug_logging: bool,
) {
    if !shared.websocket_connected(&device.id) {
        let _ = send_server_message(&mut socket, &session_revoked(), &cancellation).await;
        return;
    }
    log_ws(debug_logging, &device.id, "connect");
    let (mut writer, mut reader) = socket.split();
    let mut events = shared.events();
    let mut greeted = false;
    let expiry = tokio::time::sleep_until(tokio::time::Instant::from_std(device.expires_at));
    tokio::pin!(expiry);
    let mut close_code = WS_CLOSE_ABNORMAL;

    loop {
        tokio::select! {
            () = cancellation.cancelled() => {
                close_code = WS_CLOSE_GOING_AWAY;
                break;
            },
            () = &mut expiry => {
                let _ = send_server_message(&mut writer, &session_revoked(), &cancellation).await;
                close_code = WS_CLOSE_SESSION_REVOKED;
                break;
            },
            incoming = reader.next() => {
                let Some(Ok(message)) = incoming else { break };
                match message {
                    Message::Text(text) => {
                        let reply = handle_text_message(
                            &shared,
                            &device,
                            &text,
                            &mut greeted,
                            debug_logging,
                        );
                        let terminal_close_code = server_close_code(&reply);
                        match send_server_message(&mut writer, &reply, &cancellation).await {
                            Ok(false) => {}
                            Ok(true) => {
                                if let Some(code) = terminal_close_code {
                                    close_code = code;
                                }
                                break;
                            }
                            Err(()) => break,
                        }
                    }
                    Message::Ping(payload) => {
                        if send_message(&mut writer, Message::Pong(payload), &cancellation)
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Message::Close(frame) => {
                        close_code = frame.map_or(WS_CLOSE_NO_STATUS, |frame| frame.code);
                        break;
                    }
                    Message::Binary(_) | Message::Pong(_) => {}
                }
            }
            event = events.recv(), if greeted => {
                match event {
                    Ok(message) => {
                        let message = authorize_outbound(&shared, &device, message);
                        let terminal_close_code = server_close_code(&message);
                        match send_server_message(&mut writer, &message, &cancellation).await {
                            Ok(false) => {}
                            Ok(true) => {
                                if let Some(code) = terminal_close_code {
                                    close_code = code;
                                }
                                break;
                            }
                            Err(()) => break,
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        let message = authorize_outbound(
                            &shared,
                            &device,
                            ServerMessage::ResyncRequired {
                                next_seq: shared.next_seq(),
                            },
                        );
                        match send_server_message(&mut writer, &message, &cancellation).await {
                            Ok(false) => {}
                            Ok(true) | Err(()) => break,
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    finish_websocket(shared, &device, cancellation, debug_logging, close_code);
}

fn finish_websocket(
    shared: RemoteShared,
    device: &AuthenticatedDevice,
    cancellation: CancellationToken,
    debug_logging: bool,
    close_code: u16,
) {
    if debug_logging {
        log_ws(true, &device.id, &format!("close({close_code})"));
    }
    if let Some(grace) = shared.websocket_disconnected(&device.id) {
        spawn_controller_release(shared, grace, cancellation);
    }
}

fn handle_text_message(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    text: &str,
    greeted: &mut bool,
    debug_logging: bool,
) -> ServerMessage {
    let was_greeted = *greeted;
    let reply = match serde_json::from_str::<ClientMessage>(text) {
        Ok(message) => handle_client_message(shared, device, message, greeted),
        Err(_) => ServerMessage::Error {
            code: "invalid_message",
            message: "The message is not valid Zo Remote JSON.".to_string(),
            recoverable: true,
        },
    };
    if !was_greeted && *greeted {
        log_ws(debug_logging, &device.id, "hello-ok");
    }
    reply
}

fn server_close_code(message: &ServerMessage) -> Option<u16> {
    matches!(
        message,
        ServerMessage::Error {
            code: "session_revoked",
            recoverable: false,
            ..
        }
    )
    .then_some(WS_CLOSE_SESSION_REVOKED)
}

fn spawn_controller_release(
    shared: RemoteShared,
    grace: ControllerGrace,
    cancellation: CancellationToken,
) {
    tokio::spawn(release_controller_after(
        shared,
        grace,
        cancellation,
        CONTROLLER_DISCONNECT_GRACE,
    ));
}

async fn release_controller_after(
    shared: RemoteShared,
    grace: ControllerGrace,
    cancellation: CancellationToken,
    delay: Duration,
) {
    if cancellation.is_cancelled() {
        return;
    }
    tokio::select! {
        biased;
        () = cancellation.cancelled() => {}
        () = tokio::time::sleep(delay) => {
            shared.expire_controller_grace(&grace);
        }
    }
}

fn authorize_outbound(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    message: ServerMessage,
) -> ServerMessage {
    if shared.is_device_active(&device.id) {
        match message {
            ServerMessage::ControlState { .. } => {
                let (controller_exists, role) = shared.control_state_for(&device.id);
                ServerMessage::ControlState {
                    controller_exists,
                    role,
                }
            }
            message => message,
        }
    } else {
        session_revoked()
    }
}

fn is_terminal_message(message: &ServerMessage) -> bool {
    matches!(
        message,
        ServerMessage::Error {
            recoverable: false,
            ..
        }
    )
}

fn session_revoked() -> ServerMessage {
    ServerMessage::Error {
        code: "session_revoked",
        message: "This remote credential is no longer active.".to_string(),
        recoverable: false,
    }
}

fn handle_client_message(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    message: ClientMessage,
    greeted: &mut bool,
) -> ServerMessage {
    if !shared.is_device_active(&device.id) {
        return session_revoked();
    }
    if !*greeted && !matches!(&message, ClientMessage::Hello { .. }) {
        return ServerMessage::Error {
            code: "hello_required",
            message: "Send a protocol hello before remote commands.".to_string(),
            recoverable: false,
        };
    }
    match message {
        ClientMessage::Hello { version, last_seq } => {
            handle_hello(shared, device, version, last_seq, greeted)
        }
        ClientMessage::Ping => ServerMessage::Pong,
        ClientMessage::ControlRequest { command_id } => {
            handle_control_request(shared, device, command_id)
        }
        ClientMessage::PromptSubmit {
            command_id,
            text,
            mode,
        } => handle_prompt_submit(shared, device, command_id, &text, mode),
        ClientMessage::TurnCancel { command_id } => {
            handle_turn_cancel(shared, device, command_id)
        }
        ClientMessage::ToolApprovalRespond {
            command_id,
            request_id,
            decision,
        } => handle_tool_approval_respond(
            shared,
            device,
            command_id,
            &request_id,
            decision,
        ),
    }
}

fn handle_hello(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    version: u16,
    last_seq: u64,
    greeted: &mut bool,
) -> ServerMessage {
    if version != PROTOCOL_VERSION {
        return ServerMessage::Error {
            code: "unsupported_version",
            message: format!("Zo Remote protocol {PROTOCOL_VERSION} is required."),
            recoverable: false,
        };
    }
    *greeted = true;
    let (frames, next_seq, replace) = match shared.snapshot_for(last_seq) {
        SnapshotPlan::Full { frames, next_seq } => (frames, next_seq, true),
        SnapshotPlan::Replay { frames, next_seq } => (frames, next_seq, false),
    };
    ServerMessage::Snapshot {
        version: PROTOCOL_VERSION,
        session: shared.session_info(),
        frames,
        turn: shared.turn(),
        role: shared.role(&device.id),
        replace,
        next_seq,
        approvals: shared.pending_tool_approvals(),
    }
}

fn handle_control_request(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    command_id: String,
) -> ServerMessage {
    let decision = match reserve_command(shared, device, &command_id) {
        Ok(decision) => decision,
        Err(rejection) => return rejection,
    };
    if decision == CommandDecision::Duplicate {
        return accepted(command_id, true);
    }
    if shared.request_control(&device.id) == ControlRole::Controller {
        accepted(command_id, false)
    } else {
        shared.forget_command(&device.id, &command_id);
        rejected(
            command_id,
            "control_unavailable",
            "Another device currently controls this session.",
        )
    }
}

fn handle_prompt_submit(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    command_id: String,
    text: &str,
    mode: PromptMode,
) -> ServerMessage {
    if shared.role(&device.id) != ControlRole::Controller {
        return rejected(
            command_id,
            "observer_only",
            "Request control before sending a command.",
        );
    }
    let text = text.trim().to_string();
    if text.is_empty() || text.len() > MAX_PROMPT_BYTES {
        return rejected(
            command_id,
            "invalid_prompt",
            "Prompts must contain 1 to 32768 UTF-8 bytes.",
        );
    }
    let decision = match reserve_command(shared, device, &command_id) {
        Ok(decision) => decision,
        Err(rejection) => return rejection,
    };
    if decision == CommandDecision::Duplicate {
        return accepted(command_id, true);
    }
    let (turn, turn_generation) = shared.turn_state();
    if (mode == PromptMode::New && turn != TurnPhase::Idle)
        || (mode == PromptMode::Steer && turn != TurnPhase::Running)
    {
        shared.forget_command(&device.id, &command_id);
        let message = match mode {
            PromptMode::New => "A turn is already running; queue or steer instead.",
            PromptMode::Steer => "Steering is only available while a turn is running.",
            PromptMode::Queue => "The requested action is not valid now.",
        };
        return rejected(command_id, "invalid_turn_state", message);
    }
    let turn_generation = (mode == PromptMode::Steer).then_some(turn_generation);
    if shared
        .try_send_effect(RemoteEffect::Prompt {
            text,
            mode,
            turn_generation,
        })
        .is_err()
    {
        shared.forget_command(&device.id, &command_id);
        return rejected(
            command_id,
            "remote_busy",
            "The local Zo input queue is full. Retry shortly.",
        );
    }
    accepted(command_id, false)
}

fn handle_turn_cancel(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    command_id: String,
) -> ServerMessage {
    if shared.role(&device.id) != ControlRole::Controller {
        return rejected(
            command_id,
            "observer_only",
            "Request control before cancelling a turn.",
        );
    }
    let decision = match reserve_command(shared, device, &command_id) {
        Ok(decision) => decision,
        Err(rejection) => return rejection,
    };
    if decision == CommandDecision::Duplicate {
        return accepted(command_id, true);
    }
    let (turn, turn_generation) = shared.turn_state();
    if turn != TurnPhase::Running {
        shared.forget_command(&device.id, &command_id);
        return rejected(
            command_id,
            "invalid_turn_state",
            "No remote turn is running.",
        );
    }
    if shared
        .try_send_effect(RemoteEffect::Cancel { turn_generation })
        .is_err()
    {
        shared.forget_command(&device.id, &command_id);
        return rejected(
            command_id,
            "remote_busy",
            "The local Zo input queue is full. Retry shortly.",
        );
    }
    accepted(command_id, false)
}

fn handle_tool_approval_respond(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    command_id: String,
    request_id: &str,
    decision: ToolApprovalDecision,
) -> ServerMessage {
    if shared.role(&device.id) != ControlRole::Controller {
        return rejected(
            command_id,
            "observer_only",
            "Request control before answering a tool approval.",
        );
    }
    if request_id.is_empty() || request_id.len() > MAX_APPROVAL_ID_BYTES {
        return rejected(
            command_id,
            "invalid_approval_id",
            "Approval request IDs must contain 1 to 128 bytes.",
        );
    }
    let command = match reserve_command(shared, device, &command_id) {
        Ok(command) => command,
        Err(rejection) => return rejection,
    };
    if command == CommandDecision::Duplicate {
        return accepted(command_id, true);
    }
    match shared.resolve_tool_approval(request_id, decision, ToolApprovalSource::Remote) {
        ToolApprovalAttempt::Resolved => accepted(command_id, false),
        ToolApprovalAttempt::AlreadyResolved(_) => accepted(command_id, true),
        ToolApprovalAttempt::InvalidChoice => {
            shared.forget_command(&device.id, &command_id);
            rejected(
                command_id,
                "invalid_approval_choice",
                "That choice is not available for this tool approval.",
            )
        }
        ToolApprovalAttempt::Unknown => {
            shared.forget_command(&device.id, &command_id);
            rejected(
                command_id,
                "approval_unavailable",
                "That tool approval is no longer available.",
            )
        }
    }
}

fn reserve_command(
    shared: &RemoteShared,
    device: &AuthenticatedDevice,
    command_id: &str,
) -> Result<CommandDecision, ServerMessage> {
    shared.begin_command(&device.id, command_id).map_err(|code| {
        rejected(
            command_id.to_string(),
            code,
            "Command IDs must contain 1 to 128 bytes.",
        )
    })
}

fn accepted(command_id: String, duplicate: bool) -> ServerMessage {
    ServerMessage::CommandAccepted {
        command_id,
        duplicate,
    }
}

fn rejected(
    command_id: String,
    code: &'static str,
    message: impl Into<String>,
) -> ServerMessage {
    ServerMessage::CommandRejected {
        command_id,
        code,
        message: message.into(),
    }
}

async fn send_server_message<S>(
    writer: &mut S,
    message: &ServerMessage,
    cancellation: &CancellationToken,
) -> Result<bool, ()>
where
    S: Sink<Message, Error = axum::Error> + Unpin,
{
    send_json(writer, message, cancellation).await?;
    let terminal = is_terminal_message(message);
    if matches!(
        message,
        ServerMessage::Error {
            code: "session_revoked",
            recoverable: false,
            ..
        }
    ) {
        send_message(
            writer,
            Message::Close(Some(CloseFrame {
                code: WS_CLOSE_SESSION_REVOKED,
                reason: "session_revoked".into(),
            })),
            cancellation,
        )
        .await?;
    }
    Ok(terminal)
}

async fn send_json<S>(
    writer: &mut S,
    message: &ServerMessage,
    cancellation: &CancellationToken,
) -> Result<(), ()>
where
    S: Sink<Message, Error = axum::Error> + Unpin,
{
    let json = serde_json::to_string(message).unwrap_or_else(|_| {
        r#"{"type":"error","code":"serialization","message":"Could not serialize the remote event.","recoverable":false}"#.to_string()
    });
    send_message(writer, Message::Text(json.into()), cancellation).await
}

async fn send_message<S>(
    writer: &mut S,
    message: Message,
    cancellation: &CancellationToken,
) -> Result<(), ()>
where
    S: Sink<Message, Error = axum::Error> + Unpin,
{
    tokio::select! {
        () = cancellation.cancelled() => Err(()),
        result = tokio::time::timeout(WS_WRITE_TIMEOUT, writer.send(message)) => {
            match result {
                Ok(Ok(())) => Ok(()),
                Ok(Err(_)) | Err(_) => Err(()),
            }
        }
    }
}

fn valid_host(state: &GatewayState, headers: &HeaderMap) -> bool {
    headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|host| host.eq_ignore_ascii_case(&state.expected_host))
}

fn valid_host_and_origin(state: &GatewayState, headers: &HeaderMap) -> bool {
    valid_host(state, headers)
        && headers
            .get(header::ORIGIN)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|origin| origin == state.expected_origin.as_ref())
}

fn authenticate_push_request(
    state: &GatewayState,
    headers: &HeaderMap,
    mutation: bool,
) -> Result<AuthenticatedDevice, StatusCode> {
    let valid_request = if mutation {
        valid_host_and_origin(state, headers)
    } else {
        valid_host(state, headers)
            && headers
                .get(header::ORIGIN)
                .is_none_or(|value| value.to_str().ok() == Some(state.expected_origin.as_ref()))
    };
    if !valid_request {
        return Err(StatusCode::FORBIDDEN);
    }
    cookie(headers, SESSION_COOKIE)
        .and_then(|token| state.shared.authenticate(token))
        .ok_or(StatusCode::UNAUTHORIZED)
}

fn cookie<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(header::COOKIE)?
        .to_str()
        .ok()?
        .split(';')
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(key, value)| (key == name).then_some(value))
}

fn pairing_error(error: &PairingError) -> Response<Body> {
    let status = match error {
        PairingError::Expired => StatusCode::GONE,
        PairingError::TooManyPending | PairingError::DeviceLimit => StatusCode::TOO_MANY_REQUESTS,
        PairingError::InvalidOffer
        | PairingError::InvalidDeviceName
        | PairingError::UnknownCode(_) => StatusCode::UNAUTHORIZED,
    };
    json_error(status, "pairing_failed", &error.to_string())
}

fn json_error(status: StatusCode, code: &'static str, message: &str) -> Response<Body> {
    (status, Json(json!({ "error": code, "message": message }))).into_response()
}

fn asset(
    state: &GatewayState,
    body: &'static str,
    content_type: &'static str,
    cache: &'static str,
) -> Response<Body> {
    asset_response(state, Body::from(body), content_type, cache)
}

fn asset_bytes(
    state: &GatewayState,
    body: &'static [u8],
    content_type: &'static str,
    cache: &'static str,
) -> Response<Body> {
    asset_response(state, Body::from(body), content_type, cache)
}

fn asset_response(
    state: &GatewayState,
    body: Body,
    content_type: &'static str,
    cache: &'static str,
) -> Response<Body> {
    let mut response = Response::new(body);
    let headers = response.headers_mut();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static(cache));
    let content_security_policy = if state.expected_origin.starts_with("http://localhost:") {
        "default-src 'self'; connect-src 'self' ws:; img-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    } else {
        "default-src 'self'; connect-src 'self' wss:; img-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    };
    headers.insert(
        CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(content_security_policy),
    );
    headers.insert(REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    headers.insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );
    response
}

fn apple_touch_icon_png() -> &'static [u8] {
    static PNG: OnceLock<Vec<u8>> = OnceLock::new();
    PNG.get_or_init(generate_apple_touch_icon).as_slice()
}

fn generate_apple_touch_icon() -> Vec<u8> {
    const SIZE: u32 = 180;
    let background = Rgb([0x1f, 0x1e, 0x1d]);
    let foreground = Rgb([0xfa, 0xf9, 0xf5]);
    let vertices = [
        (90.0, 36.0),
        (137.0, 63.0),
        (137.0, 117.0),
        (90.0, 144.0),
        (43.0, 117.0),
        (43.0, 63.0),
    ];
    let mut image = RgbImage::from_pixel(SIZE, SIZE, background);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if point_in_polygon(f64::from(x) + 0.5, f64::from(y) + 0.5, &vertices) {
            *pixel = foreground;
        }
    }
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(image.as_raw(), SIZE, SIZE, ColorType::Rgb8.into())
        .expect("in-memory Apple touch icon PNG encoding");
    png
}

fn point_in_polygon(x: f64, y: f64, vertices: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let mut previous = vertices.len() - 1;
    for current in 0..vertices.len() {
        let (current_x, current_y) = vertices[current];
        let (previous_x, previous_y) = vertices[previous];
        if (current_y > y) != (previous_y > y)
            && x
                < (previous_x - current_x) * (y - current_y)
                    / (previous_y - current_y)
                    + current_x
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn status(status: StatusCode) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::empty())
        .expect("static status response")
}

async fn not_found(request: Request<Body>) -> Response<Body> {
    let _ = request;
    status(StatusCode::NOT_FOUND)
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::extract::{Path, State};
    use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
    use base64::Engine as _;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::{broadcast, mpsc, oneshot};
    use tokio_util::sync::CancellationToken;

    use super::{GatewayState, handle_client_message, router, valid_host_and_origin};
    use crate::remote_control::protocol::{
        ClientMessage, PromptMode, ServerMessage, ToolApprovalChoice, ToolApprovalDecision,
        ToolApprovalRequest, TurnPhase,
    };
    use crate::remote_control::push::PushService;
    use crate::remote_control::state::{
        AuthenticatedDevice, PairPoll, RemoteEffect, RemoteShared,
    };
    use crate::session::permission_bridge::run_permission_pump_with_remote;
    use runtime::message_stream::{BlockIdGen, RenderBlock};
    use runtime::permission::{
        ChannelPrompter, PermissionChoice, PermissionDecision, PermissionPrompter,
        PermissionRequest, RiskLevel,
    };

    #[derive(Default)]
    struct RecordingSink {
        messages: Vec<super::Message>,
    }

    impl futures_util::Sink<super::Message> for RecordingSink {
        type Error = axum::Error;

        fn poll_ready(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn start_send(
            self: std::pin::Pin<&mut Self>,
            item: super::Message,
        ) -> Result<(), Self::Error> {
            self.get_mut().messages.push(item);
            Ok(())
        }

        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }
    }

    struct PendingSink;

    #[test]
    fn debug_log_formatters_emit_only_request_and_short_device_metadata() {
        assert_eq!(
            super::format_access_log(
                &super::Method::POST,
                "/api/pair",
                super::StatusCode::ACCEPTED,
                std::time::Duration::from_millis(7),
            ),
            "[remote] POST /api/pair -> 202 7ms"
        );
        assert_eq!(
            super::format_ws_log("abcdefghijklmnop", "close(1000)"),
            "[remote] ws abcdefgh close(1000)"
        );
    }

    impl futures_util::Sink<super::Message> for PendingSink {
        type Error = axum::Error;

        fn poll_ready(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }

        fn start_send(
            self: std::pin::Pin<&mut Self>,
            _item: super::Message,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn poll_flush(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Pending
        }

        fn poll_close(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn session_revoked_message_sends_terminal_close_code() {
        let mut sink = RecordingSink::default();
        let cancellation = CancellationToken::new();
        let terminal = super::send_server_message(
            &mut sink,
            &super::session_revoked(),
            &cancellation,
        )
        .await
        .expect("terminal message is written");

        assert!(terminal);
        assert!(matches!(
            sink.messages.first(),
            Some(super::Message::Text(text)) if text.contains("session_revoked")
        ));
        assert!(matches!(
            sink.messages.get(1),
            Some(super::Message::Close(Some(frame)))
                if frame.code == super::WS_CLOSE_SESSION_REVOKED
                    && frame.reason.as_str() == "session_revoked"
        ));
    }

    #[tokio::test]
    async fn websocket_writes_are_time_bounded_and_cancellable() {
        let cancellation = CancellationToken::new();
        let started = std::time::Instant::now();
        assert!(
            super::send_message(
                &mut PendingSink,
                super::Message::Text("blocked".into()),
                &cancellation,
            )
            .await
            .is_err()
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(2));

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let started = std::time::Instant::now();
        assert!(
            super::send_message(
                &mut PendingSink,
                super::Message::Text("cancelled".into()),
                &cancellation,
            )
            .await
            .is_err()
        );
        assert!(started.elapsed() < super::WS_WRITE_TIMEOUT);
    }

    fn state() -> GatewayState {
        state_with_base("")
    }

    fn state_with_base(base_path: &str) -> GatewayState {
        state_with_base_and_push(base_path, PushService::enabled_for_test())
    }

    fn state_with_base_and_push(base_path: &str, push: PushService) -> GatewayState {
        let (prompt_effects, _) = mpsc::channel(8);
        let (control_effects, _) = mpsc::channel(8);
        let (notices, _) = mpsc::channel(8);
        GatewayState::new(
            RemoteShared::new_with_push(
                "session".into(),
                "zo".into(),
                prompt_effects,
                control_effects,
                notices,
                push,
            ),
            "laptop.example.ts.net".into(),
            "https://laptop.example.ts.net".into(),
            base_path.to_string(),
            CancellationToken::new(),
        )
    }

    async fn request_path(path: &str) -> (u16, String) {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind test gateway");
        let address = listener.local_addr().expect("test gateway address");
        let server = tokio::spawn(async move {
            axum::serve(listener, router(state_with_base("/s8790")))
                .await
                .expect("serve test gateway");
        });
        let mut stream = TcpStream::connect(address).await.expect("connect to gateway");
        stream
            .write_all(
                format!(
                    "GET {path} HTTP/1.1\r\nHost: laptop.example.ts.net\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            )
            .await
            .expect("write HTTP request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read HTTP response");
        server.abort();
        let response = String::from_utf8(response).expect("UTF-8 test response");
        let (headers, body) = response
            .split_once("\r\n\r\n")
            .expect("HTTP response separator");
        let status = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|status| status.parse().ok())
            .expect("numeric HTTP status");
        (status, body.to_string())
    }

    #[tokio::test]
    async fn base_path_and_stripped_root_reach_the_same_routes() {
        let root = request_path("/").await;
        let mounted_without_slash = request_path("/s8790").await;
        let mounted = request_path("/s8790/").await;
        let root_asset = request_path("/styles.css").await;
        let mounted_asset = request_path("/s8790/styles.css").await;
        let wrong_mount = request_path("/s8791/").await;

        assert_eq!(root.0, 200);
        assert_eq!(mounted_without_slash.0, 200);
        assert_eq!(mounted.0, 200);
        assert_eq!(root.1, mounted.1);
        assert_eq!(root_asset.0, 200);
        assert_eq!(mounted_asset.0, 200);
        assert_eq!(root_asset.1, mounted_asset.1);
        assert_eq!(wrong_mount.0, 404);
    }

    #[tokio::test]
    async fn session_cookie_is_scoped_to_the_mount_path() {
        let state = state_with_base("/s8790");
        state.shared.refresh_offer("offer");
        let pairing = state
            .shared
            .begin_pairing("offer", "Phone")
            .expect("pairing starts");
        state
            .shared
            .approve(&pairing.comparison_code)
            .expect("pairing is approved");
        let mut headers = HeaderMap::new();
        headers.insert(
            header::HOST,
            HeaderValue::from_static("laptop.example.ts.net"),
        );

        let response = super::pair_status(State(state), Path(pairing.id), headers).await;
        let cookie = response
            .headers()
            .get(header::SET_COOKIE)
            .and_then(|value| value.to_str().ok())
            .expect("session cookie");

        assert!(cookie.contains("Path=/s8790;"));
    }

    fn paired_gateway_state() -> (GatewayState, String, AuthenticatedDevice) {
        let gateway = state();
        gateway.shared.refresh_offer("offer");
        let pairing = gateway
            .shared
            .begin_pairing("offer", "Phone")
            .expect("pairing starts");
        gateway
            .shared
            .approve(&pairing.comparison_code)
            .expect("pairing is approved");
        let token = match gateway.shared.pairing_status(&pairing.id) {
            PairPoll::Approved { token, .. } => token,
            other => panic!("expected approved pairing, got {other:?}"),
        };
        let device = gateway
            .shared
            .authenticate(&token)
            .expect("credential is valid");
        (gateway, token, device)
    }

    fn paired_state() -> (RemoteShared, AuthenticatedDevice) {
        let (gateway, _token, device) = paired_gateway_state();
        (gateway.shared, device)
    }

    fn pair_device(
        shared: &RemoteShared,
        offer: &str,
        name: &str,
    ) -> AuthenticatedDevice {
        shared.refresh_offer(offer);
        let pairing = shared
            .begin_pairing(offer, name)
            .expect("pairing starts");
        shared
            .approve(&pairing.comparison_code)
            .expect("pairing is approved");
        let token = match shared.pairing_status(&pairing.id) {
            PairPoll::Approved { token, .. } => token,
            other => panic!("expected approved pairing, got {other:?}"),
        };
        shared.authenticate(&token).expect("credential is valid")
    }

    async fn next_tool_approval(
        events: &mut broadcast::Receiver<ServerMessage>,
    ) -> ToolApprovalRequest {
        loop {
            if let ServerMessage::ToolApprovalRequest { approval } =
                events.recv().await.expect("tool approval event")
            {
                return approval;
            }
        }
    }

    fn permission_request() -> PermissionRequest {
        PermissionRequest {
            tool: "Bash".to_string(),
            input_summary: "cargo test -p runtime".to_string(),
            input_hash: "abc123".to_string(),
            reasoning: "run runtime tests".to_string(),
            choices: vec![
                PermissionChoice {
                    key: 'o',
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

    #[tokio::test]
    async fn remote_permission_gateway_round_trip_resolves_the_runtime_gate() {
        let (shared, device) = paired_state();
        let mut events = shared.events();
        let (prompter, request_rx) = ChannelPrompter::new(4);
        let (render_tx, mut render_rx) = mpsc::channel(4);
        let (resolution_tx, _resolution_rx) = mpsc::unbounded_channel();
        let pump = tokio::spawn(run_permission_pump_with_remote(
            request_rx,
            render_tx,
            BlockIdGen::default(),
            Some(shared.clone()),
            Some(resolution_tx),
        ));
        let decide_prompter = prompter.clone();
        let decide = tokio::spawn(async move { decide_prompter.decide(permission_request()).await });

        let _prompt = match render_rx.recv().await.expect("TUI prompt") {
            RenderBlock::PermissionPrompt(prompt) => prompt,
            other => panic!("unexpected block: {other:?}"),
        };
        let approval = next_tool_approval(&mut events).await;
        let mut greeted = true;
        let reply = handle_client_message(
            &shared,
            &device,
            ClientMessage::ToolApprovalRespond {
                command_id: "approval-command-1".to_string(),
                request_id: approval.request_id,
                decision: ToolApprovalDecision::AllowOnce,
            },
            &mut greeted,
        );
        assert!(matches!(
            reply,
            ServerMessage::CommandAccepted {
                duplicate: false,
                ..
            }
        ));
        assert_eq!(
            decide.await.expect("decision join").expect("gate decision"),
            PermissionDecision::AllowOnce
        );

        drop(prompter);
        assert!(pump.await.expect("pump join").is_ok());
    }

    #[tokio::test]
    async fn remote_permission_duplicate_and_replayed_answers_are_ignored() {
        let (shared, device) = paired_state();
        let (decision_tx, decision_rx) = oneshot::channel();
        let request_id = shared.publish_tool_approval(
            ToolApprovalRequest {
                request_id: String::new(),
                tool_name: "Bash".to_string(),
                input_summary: "cargo test".to_string(),
                input_hash: "abc123".to_string(),
                choices: vec![ToolApprovalChoice {
                    label: "Allow once".to_string(),
                    decision: ToolApprovalDecision::AllowOnce,
                }],
            },
            decision_tx,
        );
        let command = |command_id: &str| ClientMessage::ToolApprovalRespond {
            command_id: command_id.to_string(),
            request_id: request_id.clone(),
            decision: ToolApprovalDecision::AllowOnce,
        };
        let mut greeted = true;

        assert!(matches!(
            handle_client_message(&shared, &device, command("answer-1"), &mut greeted),
            ServerMessage::CommandAccepted {
                duplicate: false,
                ..
            }
        ));
        assert_eq!(
            decision_rx.await.expect("one gate resolution"),
            ToolApprovalDecision::AllowOnce
        );
        assert!(matches!(
            handle_client_message(&shared, &device, command("answer-1"), &mut greeted),
            ServerMessage::CommandAccepted {
                duplicate: true,
                ..
            }
        ));
        assert!(matches!(
            handle_client_message(&shared, &device, command("answer-2"), &mut greeted),
            ServerMessage::CommandAccepted {
                duplicate: true,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn last_controller_websocket_close_releases_and_broadcasts_control_state() {
        let gateway = state();
        let shared = gateway.shared.clone();
        let controller = pair_device(&shared, "controller-offer", "Phone");
        let observer = pair_device(&shared, "observer-offer", "Tablet");
        let mut events = shared.events();

        assert!(shared.websocket_connected(&controller.id));
        let grace = shared
            .websocket_disconnected(&controller.id)
            .expect("last controller socket starts grace");
        super::release_controller_after(
            shared.clone(),
            grace,
            CancellationToken::new(),
            std::time::Duration::from_millis(1),
        )
        .await;

        let event = events.recv().await.expect("control state broadcast");
        assert!(matches!(
            &event,
            ServerMessage::ControlState {
                controller_exists: false,
                ..
            }
        ));
        let released_for_controller = super::authorize_outbound(&shared, &controller, event);
        assert!(matches!(
            released_for_controller,
            ServerMessage::ControlState {
                controller_exists: false,
                role: super::ControlRole::Observer,
            }
        ));
        assert_eq!(shared.role(&controller.id), super::ControlRole::Observer);
        assert_eq!(shared.request_control(&observer.id), super::ControlRole::Controller);
        let granted = events.recv().await.expect("request-control broadcast");
        let granted_for_observer = super::authorize_outbound(&shared, &observer, granted.clone());
        let granted_for_former_controller =
            super::authorize_outbound(&shared, &controller, granted);
        assert!(matches!(
            granted_for_observer,
            ServerMessage::ControlState {
                controller_exists: true,
                role: super::ControlRole::Controller,
            }
        ));
        assert!(matches!(
            granted_for_former_controller,
            ServerMessage::ControlState {
                controller_exists: true,
                role: super::ControlRole::Observer,
            }
        ));
    }

    #[tokio::test]
    async fn controller_reconnect_within_grace_retains_control_without_release() {
        let (shared, controller) = paired_state();
        let mut events = shared.events();
        assert!(shared.websocket_connected(&controller.id));
        let grace = shared
            .websocket_disconnected(&controller.id)
            .expect("last controller socket starts grace");
        let release = tokio::spawn(super::release_controller_after(
            shared.clone(),
            grace,
            CancellationToken::new(),
            std::time::Duration::from_millis(10),
        ));

        assert!(shared.websocket_connected(&controller.id));
        release.await.expect("release task joins");

        assert_eq!(shared.role(&controller.id), super::ControlRole::Controller);
        assert!(events.try_recv().is_err());
    }

    #[tokio::test]
    async fn gateway_shutdown_cancels_pending_controller_release() {
        let (shared, controller) = paired_state();
        assert!(shared.websocket_connected(&controller.id));
        let grace = shared
            .websocket_disconnected(&controller.id)
            .expect("last controller socket starts grace");
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        super::release_controller_after(
            shared.clone(),
            grace,
            cancellation,
            std::time::Duration::from_secs(1),
        )
        .await;

        assert_eq!(shared.role(&controller.id), super::ControlRole::Controller);
    }

    fn paired_state_with_effects() -> (
        RemoteShared,
        AuthenticatedDevice,
        mpsc::Receiver<RemoteEffect>,
    ) {
        let (prompt_effects, prompt_effect_rx) = mpsc::channel(8);
        let (control_effects, _control_effect_rx) = mpsc::channel(8);
        let (notices, _notice_rx) = mpsc::channel(8);
        let shared = RemoteShared::new(
            "session".into(),
            "zo".into(),
            prompt_effects,
            control_effects,
            notices,
        );
        shared.refresh_offer("offer");
        let pairing = shared
            .begin_pairing("offer", "Phone")
            .expect("pairing starts");
        shared
            .approve(&pairing.comparison_code)
            .expect("pairing is approved");
        let token = match shared.pairing_status(&pairing.id) {
            PairPoll::Approved { token, .. } => token,
            other => panic!("expected approved pairing, got {other:?}"),
        };
        let device = shared.authenticate(&token).expect("credential is valid");
        (shared, device, prompt_effect_rx)
    }

    #[test]
    fn commands_require_protocol_hello() {
        let (shared, device) = paired_state();
        let mut greeted = false;
        let reply = handle_client_message(&shared, &device, ClientMessage::Ping, &mut greeted);
        assert!(super::is_terminal_message(&reply));
        assert!(matches!(
            reply,
            ServerMessage::Error {
                code: "hello_required",
                recoverable: false,
                ..
            }
        ));
    }

    #[test]
    fn inactive_devices_cannot_receive_broadcast_events() {
        let (shared, device) = paired_state();
        shared.revoke_all();
        let message = super::authorize_outbound(
            &shared,
            &device,
            ServerMessage::TurnState {
                turn: TurnPhase::Running,
            },
        );
        assert!(super::is_terminal_message(&message));
        assert!(matches!(
            message,
            ServerMessage::Error {
                code: "session_revoked",
                recoverable: false,
                ..
            }
        ));
    }

    #[test]
    fn recoverable_protocol_errors_keep_the_socket_open() {
        let message = ServerMessage::Error {
            code: "invalid_message",
            message: "invalid".to_string(),
            recoverable: true,
        };
        assert!(!super::is_terminal_message(&message));
    }

    #[test]
    fn duplicate_new_prompt_is_acknowledged_after_turn_starts() {
        let (shared, device, mut effects) = paired_state_with_effects();
        let mut greeted = true;
        let command = || ClientMessage::PromptSubmit {
            command_id: "command-1".to_string(),
            text: "hello".to_string(),
            mode: PromptMode::New,
        };
        assert!(matches!(
            handle_client_message(&shared, &device, command(), &mut greeted),
            ServerMessage::CommandAccepted {
                duplicate: false,
                ..
            }
        ));
        assert!(matches!(
            effects.try_recv(),
            Ok(RemoteEffect::Prompt {
                mode: PromptMode::New,
                ..
            })
        ));

        shared.set_turn(TurnPhase::Running, 1);
        assert!(matches!(
            handle_client_message(&shared, &device, command(), &mut greeted),
            ServerMessage::CommandAccepted {
                duplicate: true,
                ..
            }
        ));
        assert!(effects.try_recv().is_err());
    }

    #[test]
    fn invalid_command_id_is_rejected_instead_of_timing_out() {
        let (shared, device) = paired_state();
        let mut greeted = true;
        let reply = handle_client_message(
            &shared,
            &device,
            ClientMessage::PromptSubmit {
                command_id: String::new(),
                text: "hello".to_string(),
                mode: PromptMode::New,
            },
            &mut greeted,
        );
        assert!(matches!(
            reply,
            ServerMessage::CommandRejected {
                command_id,
                code: "invalid_command_id",
                ..
            } if command_id.is_empty()
        ));
    }

    #[test]
    fn mutation_requires_exact_host_and_origin() {
        let state = state();
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("laptop.example.ts.net"));
        assert!(!valid_host_and_origin(&state, &headers));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://evil.example"),
        );
        assert!(!valid_host_and_origin(&state, &headers));
        headers.insert(
            header::ORIGIN,
            HeaderValue::from_static("https://laptop.example.ts.net"),
        );
        assert!(valid_host_and_origin(&state, &headers));
        headers.insert(
            header::HOST,
            HeaderValue::from_static("laptop.example.ts.net.evil.example"),
        );
        assert!(!valid_host_and_origin(&state, &headers));
    }

    fn push_request_headers(token: Option<&str>, origin: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::HOST,
            HeaderValue::from_static("laptop.example.ts.net"),
        );
        if let Some(origin) = origin {
            headers.insert(
                header::ORIGIN,
                HeaderValue::from_str(origin).expect("origin header value"),
            );
        }
        if let Some(token) = token {
            let cookie = format!("{}={token}", super::SESSION_COOKIE);
            headers.insert(
                header::COOKIE,
                HeaderValue::from_str(&cookie).expect("cookie header value"),
            );
        }
        headers
    }

    fn subscription_request(
        endpoint: &str,
        p256dh: &str,
        auth: &str,
    ) -> super::PushSubscriptionRequest {
        serde_json::from_value(serde_json::json!({
            "endpoint": endpoint,
            "keys": { "p256dh": p256dh, "auth": auth },
        }))
        .expect("subscription request JSON")
    }

    #[tokio::test]
    async fn icon_routes_serve_svg_and_generated_apple_touch_png() {
        let (status, body) = request_path("/s8790/icon.svg").await;
        assert_eq!(status, 200);
        assert!(body.contains("<svg"));

        let response =
            super::apple_touch_icon(State(state()), push_request_headers(None, None)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("image/png")
        );
        assert_eq!(
            response
                .headers()
                .get(header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("public, max-age=300")
        );
        let png =
            image::load_from_memory(super::apple_touch_icon_png()).expect("valid PNG icon");
        assert_eq!(image::GenericImageView::dimensions(&png), (180, 180));
        let corner = image::GenericImageView::get_pixel(&png, 2, 2);
        let center = image::GenericImageView::get_pixel(&png, 90, 90);
        assert_ne!(corner, center, "hexagon glyph must differ from background");
    }

    #[tokio::test]
    async fn push_config_requires_device_auth_and_reports_server_key() {
        let (gateway, token, _device) = paired_gateway_state();

        let anonymous =
            super::push_config(State(gateway.clone()), push_request_headers(None, None)).await;
        assert_eq!(anonymous.status(), StatusCode::UNAUTHORIZED);

        let wrong_origin = super::push_config(
            State(gateway.clone()),
            push_request_headers(Some(&token), Some("https://evil.example")),
        )
        .await;
        assert_eq!(wrong_origin.status(), StatusCode::FORBIDDEN);

        let response =
            super::push_config(State(gateway), push_request_headers(Some(&token), None)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("config body");
        let config: serde_json::Value = serde_json::from_slice(&body).expect("config JSON");
        let server_key = config["push"]["server_key"].as_str().expect("server key");
        let key = URL_SAFE_NO_PAD.decode(server_key).expect("base64url server key");
        assert_eq!(key.len(), 65);
        assert_eq!(key[0], 0x04);
    }

    #[tokio::test]
    async fn push_subscription_routes_validate_payload_and_are_idempotent() {
        let (gateway, token, _device) = paired_gateway_state();
        let origin = "https://laptop.example.ts.net";
        let p256dh = PushService::enabled_for_test()
            .server_key()
            .expect("test server key")
            .to_string();
        let auth = URL_SAFE_NO_PAD.encode([7_u8; 16]);
        let endpoint = "https://fcm.googleapis.com/fcm/send/test-token";

        let missing_origin = super::push_subscription_put(
            State(gateway.clone()),
            push_request_headers(Some(&token), None),
            Ok(super::Json(subscription_request(endpoint, &p256dh, &auth))),
        )
        .await;
        assert_eq!(missing_origin.status(), StatusCode::FORBIDDEN);

        let anonymous = super::push_subscription_put(
            State(gateway.clone()),
            push_request_headers(None, Some(origin)),
            Ok(super::Json(subscription_request(endpoint, &p256dh, &auth))),
        )
        .await;
        assert_eq!(anonymous.status(), StatusCode::UNAUTHORIZED);

        let accepted = super::push_subscription_put(
            State(gateway.clone()),
            push_request_headers(Some(&token), Some(origin)),
            Ok(super::Json(subscription_request(endpoint, &p256dh, &auth))),
        )
        .await;
        assert_eq!(accepted.status(), StatusCode::NO_CONTENT);

        let malformed = super::push_subscription_put(
            State(gateway.clone()),
            push_request_headers(Some(&token), Some(origin)),
            Ok(super::Json(subscription_request(
                endpoint,
                &p256dh,
                &URL_SAFE_NO_PAD.encode([7_u8; 8]),
            ))),
        )
        .await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);

        let not_allowed = super::push_subscription_put(
            State(gateway.clone()),
            push_request_headers(Some(&token), Some(origin)),
            Ok(super::Json(subscription_request(
                "https://evil.example/push",
                &p256dh,
                &auth,
            ))),
        )
        .await;
        assert_eq!(not_allowed.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = to_bytes(not_allowed.into_body(), usize::MAX)
            .await
            .expect("rejection body");
        let error: serde_json::Value = serde_json::from_slice(&body).expect("rejection JSON");
        assert_eq!(error["error"], "push_endpoint_not_allowed");

        let deleted = super::push_subscription_delete(
            State(gateway.clone()),
            push_request_headers(Some(&token), Some(origin)),
        )
        .await;
        assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
        let repeated = super::push_subscription_delete(
            State(gateway),
            push_request_headers(Some(&token), Some(origin)),
        )
        .await;
        assert_eq!(repeated.status(), StatusCode::NO_CONTENT);
    }
}
