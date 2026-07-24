//! `zo attach` — the **rich** TUI client for a running
//! [`zo serve`](crate::serve) session.
//!
//! Where [`crate::attach`] is a plain line client, this reuses the *entire*
//! ratatui [`App`] to render a remote session exactly like the local REPL —
//! transcript cards, reasoning, tool cards, live token/cost ledger, rate-limit
//! gauges — with one substitution: the per-turn driver is a socket round-trip
//! instead of a local `LiveCli`.
//!
//! ## How the reuse works
//!
//! [`App`] is already decoupled from any runtime: `App::new(theme, render_rx,
//! cmd_tx)` takes no `LiveCli`. It renders from a [`RenderBlock`] stream and
//! emits [`AppAction`]s. So the client:
//!
//! 1. boots the terminal + theme with the **same** [`init_terminal`] /
//!    [`boot_theme`] / [`restore_terminal`] the REPL uses;
//! 2. on [`AppAction::Submit`], sends `session.run_turn` and feeds the streamed
//!    render frames straight into the App via [`App::push_block`] (decoded with
//!    [`render_block_from_value`]);
//! 3. hydrates the sidebar once from `session.info`.
//!
//! Frames only arrive in response to a turn request, so the socket is silent at
//! idle — no background reader task is needed; the turn loop reads it directly.
//!
//! ## Pair sessions (track 5)
//!
//! Two connections are opened: the **primary** drives turns and meta RPCs
//! (request/response + its own `run_turn` frames only), and a **spectator**
//! connection `session.subscribe`s and streams fan-out frames from *other*
//! clients' turns straight into the App via `render_tx` — so a second terminal
//! watching the same session sees turns and steering live. The spectator reader
//! is gated off while *this* client drives its own turn (those frames already
//! arrive on the primary stream), so it never double-renders. `/steer <text>`
//! pushes a mid-turn steering message to the session's helm; `/roster` lists the
//! connected peers.
//!
//! ## Scope (proof-of-concept)
//!
//! Model/permission switching, rewind, `$EDITOR`, and clipboard are surfaced as
//! "not available over attach" notices for now; permission prompts stream to the
//! helm. The headline — a full-fidelity TUI rendering a persistent remote
//! session, live-shared across terminals — works today.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};

use runtime::message_stream::{
    BlockIdGen, PermissionDecision as RenderPermissionDecision, RenderBlock, SystemLevel,
};
use serde_json::{json, Value as JsonValue};

use crate::serve_protocol::{is_response_line, RpcRequest, SessionInfo};
use crate::session::tui_loop::{boot_theme, init_terminal, restore_terminal, TuiTerminal};
use crate::session::turn_controller::key_to_permission_decision;
use zo_cli::sinks::permission_decision_tag;
use zo_cli::sinks::render_block_from_value;
use zo_cli::tui::app::AppAction;
use zo_cli::tui::render_schedule::{
    ANIMATION_TICK_INTERVAL, STREAM_FRAME_INTERVAL, StreamFrameGate,
};
use zo_cli::tui::app::{AckVerdict, SpectatorEvent};
use zo_cli::tui::{AgentCommand, App};

/// Bounded render/command channels — App requires a receiver even though the
/// attach client pushes blocks directly. Mirrors the REPL's capacities.
const RENDER_CHANNEL_CAPACITY: usize = 64;
const COMMAND_CHANNEL_CAPACITY: usize = 8;
/// Bound pre-ack tails so a blocked App cannot allocate an unbounded replay.
const MAX_DRAIN_PER_TICK: usize = 256;

/// Errors the attach client surfaces.
#[derive(Debug)]
enum AttachError {
    /// Socket I/O failed.
    Io(std::io::Error),
    /// The server closed the connection unexpectedly.
    Closed,
    /// The server returned a JSON-RPC error.
    Rpc(String),
    /// The server rejected the submission before a turn executed.
    TurnRejected(String),
    /// A terminal/render failure.
    Tui(String),
}

impl std::fmt::Display for AttachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Closed => write!(f, "server closed the connection"),
            Self::Rpc(message) | Self::TurnRejected(message) | Self::Tui(message) => {
                write!(f, "{message}")
            }
        }
    }
}

impl std::error::Error for AttachError {}

/// Validate the JSON-RPC envelope shared by primary request and turn paths.
/// A response is useful only when it has the expected version/id and exactly
/// one payload shape; error payloads additionally need typed code/message.
fn validate_response(value: &JsonValue, expected_id: u64) -> Option<JsonValue> {
    if value.get("jsonrpc").and_then(JsonValue::as_str) != Some("2.0")
        || value.get("id").and_then(JsonValue::as_u64) != Some(expected_id)
    {
        return None;
    }
    match (value.get("result"), value.get("error")) {
        (Some(_), None) => Some(value.clone()),
        (None, Some(error))
            if error.get("code").and_then(JsonValue::as_i64).is_some()
                && error.get("message").and_then(JsonValue::as_str).is_some() =>
        {
            Some(value.clone())
        }
        _ => None,
    }
}

/// The remote session handle: owns the split socket and the request counter.
struct AttachClient {
    write_half: OwnedWriteHalf,
    reader: tokio::io::Lines<BufReader<OwnedReadHalf>>,
    session_id: String,
    next_id: u64,
    /// Server address, retained so a mid-turn Ctrl+C can open a *second*
    /// connection to send `session.cancel_turn` while this connection is busy
    /// streaming the turn (F4).
    bind_addr: String,
    /// Shared secret for a guarded server (see [`crate::serve_auth`]), stamped
    /// onto every request in [`AttachClient::send`]. `None` on the tokenless
    /// loopback default, in which case nothing is added to the wire.
    auth_token: Option<String>,
}

impl AttachClient {
    fn next_request_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Write one request as a `\n`-terminated JSON line, stamping the
    /// shared-secret token (if any) so a guarded server accepts it.
    async fn send(&mut self, request: RpcRequest) -> Result<(), AttachError> {
        let request = request.with_token(self.auth_token.clone());
        let mut line =
            serde_json::to_vec(&request).map_err(|error| AttachError::Io(error.into()))?;
        line.push(b'\n');
        self.write_half
            .write_all(&line)
            .await
            .map_err(AttachError::Io)?;
        self.write_half.flush().await.map_err(AttachError::Io)
    }

    /// Send a non-streaming request and read until its response, returning the
    /// `result` body. Stray render frames (none are expected for these methods)
    /// are skipped.
    async fn request(&mut self, method: &str, params: JsonValue) -> Result<JsonValue, AttachError> {
        let id = self.next_request_id();
        self.send(RpcRequest::new(id, method, params)).await?;
        loop {
            let line = self
                .reader
                .next_line()
                .await
                .map_err(AttachError::Io)?
                .ok_or(AttachError::Closed)?;
            let value: JsonValue = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if !is_response_line(&value) {
                continue;
            }
            let response = validate_response(&value, id).ok_or_else(|| {
                AttachError::Rpc(format!("{method}: invalid JSON-RPC response"))
            })?;
            if let Some(error) = response.get("error") {
                let message = error
                    .get("message")
                    .and_then(JsonValue::as_str)
                    .expect("validated error message");
                return Err(AttachError::Rpc(format!("{method}: {message}")));
            }
            return Ok(response
                .get("result")
                .expect("validated result")
                .clone());
        }
    }
}

/// Entry point for `zo attach` (rich TUI). `session_id` is `None` to create
/// a fresh session on the server and attach to it.
pub(crate) fn run_attach_tui(
    bind_addr: String,
    session_id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // A pure socket + terminal client — no runtime is constructed, so a
    // single-threaded reactor is enough (crossterm's EventStream manages its
    // own input thread; the cmd-drain task runs on this executor).
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(attach_tui_main(bind_addr, session_id))
}

async fn attach_tui_main(
    bind_addr: String,
    session_id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = TcpStream::connect(&bind_addr).await.map_err(|error| {
        format!("zo attach: cannot connect to {bind_addr}: {error}. Is `zo serve` running?")
    })?;
    let (read_half, write_half) = stream.into_split();
    let mut client = AttachClient {
        write_half,
        reader: BufReader::new(read_half).lines(),
        // Resolved below; a placeholder until then.
        session_id: String::new(),
        next_id: 1,
        bind_addr: bind_addr.clone(),
        auth_token: crate::serve_auth::token_from_env(),
    };

    // Resolve the session up front (over the socket, before the TUI boots so a
    // connection error prints cleanly instead of inside the alt-screen).
    client.session_id = if let Some(id) = session_id {
        id
    } else {
        let result = client.request("session.create", JsonValue::Null).await?;
        result
            .get("id")
            .and_then(JsonValue::as_str)
            .ok_or("session.create: server did not return an id")?
            .to_string()
    };
    let info = client
        .request("session.info", json!({ "id": client.session_id }))
        .await
        .ok();

    // Boot the TUI with the same plumbing the local REPL uses.
    let (render_tx, render_rx) = mpsc::channel::<RenderBlock>(RENDER_CHANNEL_CAPACITY);
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<AgentCommand>(COMMAND_CHANNEL_CAPACITY);
    // Track 5: the spectator reader pushes other clients' turn frames onto
    // `render_tx`, so keep our own handle alive (App::run quits if all senders
    // drop) and hand a clone to the reader.
    let _render_tx = render_tx;
    // Drain agent commands so the App never blocks on a full channel (agent
    // spawning over attach is a deferred feature).
    tokio::spawn(async move { while cmd_rx.recv().await.is_some() {} });

    // The primary stream owns local turns; spectator delivery is gated until
    // a marker/HELM_HELD forces one atomic boundary replacement.
    let own_turn = Arc::new(AtomicBool::new(false));

    let (spectator_tx, spectator_rx) = mpsc::channel(RENDER_CHANNEL_CAPACITY);
    let (force_resync_tx, force_resync_rx) = mpsc::channel(1);
    let watermark = Arc::new(AtomicU64::new(0));
    let terminal_mode = zo_cli::tui::TerminalMode::Fullscreen;
    let terminal_background = zo_cli::tui::term::detect_background();
    let (mut terminal, _stderr_guard) = init_terminal(terminal_mode)?;
    let theme = boot_theme(terminal_background);
    let mut app = App::new_with_spectator(theme, render_rx, spectator_rx, cmd_tx);
    let ids = BlockIdGen::default();

    open_spectator_stream(SpectatorContext {
        bind_addr: bind_addr.clone(),
        session_id: client.session_id.clone(),
        subscription_id: Arc::new(AtomicU64::new(0)),
        spectator_tx,
        own_turn: Arc::clone(&own_turn),
        watermark: Arc::clone(&watermark),
        auth_token: crate::serve_auth::token_from_env(),
        force_resync_rx,
    });

    if let Some(ref info) = info {
        apply_session_info(&mut app, info);
    }
    // Same input history + slash/mention frecency as the local REPL — the
    // remote-attach TUI is a fully interactive path (slash commands, @-mentions),
    // so it must load the frecency stores too (project scope = local cwd).
    let attach_cwd = crate::current_cli_cwd().unwrap_or_default();
    crate::session::tui_loop::load_input_frecency(&mut app, &attach_cwd);
    // The supervisor sends the authoritative initial history as a Replace.
    // Do not seed a second, potentially stale session.load snapshot here.
    app.push_block(RenderBlock::System {
        id: ids.next(),
        level: SystemLevel::Info,
        text: format!("attached to {} — Ctrl-D to detach", client.session_id),
    });
    app.enable_input();
    app.draw_frame(&mut terminal)?;

    let result = attach_session_loop(
        &mut client,
        &mut app,
        &mut terminal,
        &ids,
        &own_turn,
        &watermark,
        force_resync_tx,
    )
    .await;

    let restore = restore_terminal(&mut terminal, terminal_mode);
    result?;
    restore?;
    println!(
        "detached — session {} still alive on the server.",
        client.session_id
    );
    Ok(())
}

/// The outer loop: read user actions from the App, drive turns over the socket.
#[allow(clippy::too_many_lines)] // one match arm per AppAction variant
async fn attach_session_loop(
    client: &mut AttachClient,
    app: &mut App,
    terminal: &mut TuiTerminal,
    ids: &BlockIdGen,
    own_turn: &Arc<AtomicBool>,
    watermark: &Arc<AtomicU64>,
    force_resync_tx: mpsc::Sender<()>,
) -> Result<(), AttachError> {
    let mut turn_generation = 0_u64;
    loop {
        app.enable_input();
        let action = app
            .run(terminal)
            .await
            .map_err(|error| AttachError::Tui(error.to_string()))?;
        match action {
            AppAction::Quit => return Ok(()),
            AppAction::Submit(input) => {
                let trimmed = input.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                app.append_history(&trimmed);
                if matches!(trimmed.as_str(), "/exit" | "/quit") {
                    return Ok(());
                }
                // Track 5 pair commands, handled without starting a turn:
                //   /steer <text> — push a mid-turn steering message to the
                //     session's in-flight turn (helm folds it at the next
                //     tool-result boundary; the echo fans out to all watchers).
                //   /roster — list the peers connected to this session.
                if let Some(text) = trimmed.strip_prefix("/steer ") {
                    handle_steer(client, app, terminal, ids, text.trim()).await?;
                    continue;
                }
                if matches!(trimmed.as_str(), "/roster" | "/viewers") {
                    handle_roster(client, app, terminal, ids).await?;
                    continue;
                }
                app.dismiss_startup_screen();
                app.push_block(RenderBlock::UserMessage {
                    id: ids.next(),
                    text: trimmed.clone(),
                });
                let generation = next_attach_turn_generation(&mut turn_generation);
                app.begin_turn_with_generation(generation);
                app.disable_input();
                app.draw_frame(terminal).map_err(tui_err)?;

                // Gate the spectator reader off for the duration of our own turn:
                // those frames arrive on the primary stream, so rendering the
                // fan-out copy too would double them.
                own_turn.store(true, Ordering::Release);
                let outcome = socket_run_turn(
                    client,
                    app,
                    terminal,
                    ids,
                    &trimmed,
                    watermark,
                    &force_resync_tx,
                )
                .await;
                // socket_run_turn stores successful next_seq before this Release.
                own_turn.store(false, Ordering::Release);
                settle_attached_turn(app, &outcome);
                if let Err(error) = outcome {
                    // A HELM_HELD reply means this client speculatively gated
                    // spectator frames, but another helm owned the turn. Force
                    // an atomic boundary replacement rather than stale load.
                    // All terminal errors/cancel/HELM_HELD converge on one
                    // capacity-one boundary resync signal; storms coalesce.
                    let _ = force_resync_tx.try_send(());
                    app.push_block(RenderBlock::System {
                        id: ids.next(),
                        level: SystemLevel::Error,
                        text: format!("turn failed: {error}"),
                    });
                }
                // Refresh the sidebar from the server (model/perm/cwd are static
                // per session, but this also lands any branch change).
                if let Ok(info) = client
                    .request("session.info", json!({ "id": client.session_id }))
                    .await
                {
                    apply_session_info(app, &info);
                }
                app.draw_frame(terminal).map_err(tui_err)?;
            }
            AppAction::ConnectApiKey { provider, api_key } => {
                apply_meta_rpc(
                    client,
                    app,
                    terminal,
                    ids,
                    "session.connect_api_key",
                    json!({ "id": client.session_id, "provider": provider, "api_key": api_key }),
                    false,
                )
                .await?;
            }
            AppAction::ConnectCustomProvider(draft) => {
                apply_meta_rpc(
                    client,
                    app,
                    terminal,
                    ids,
                    "session.connect_custom_provider",
                    json!({
                        "id": client.session_id,
                        "name": draft.name,
                        "base_url": draft.base_url,
                        "auth_env": draft.auth_env,
                        "api_key": draft.api_key,
                        "models": draft.models,
                        "context_window": draft.context_window,
                        "max_output_tokens": draft.max_output_tokens,
                        "include_usage": draft.include_usage,
                    }),
                    false,
                )
                .await?;
            }
            AppAction::SelectModel(model) => {
                apply_meta_rpc(
                    client,
                    app,
                    terminal,
                    ids,
                    "session.set_model",
                    json!({ "id": client.session_id, "model": model.alias }),
                    false,
                )
                .await?;
            }
            AppAction::SelectPermission(mode) => {
                apply_meta_rpc(
                    client,
                    app,
                    terminal,
                    ids,
                    "session.set_permission",
                    json!({ "id": client.session_id, "mode": mode.as_str() }),
                    false,
                )
                .await?;
            }
            AppAction::SelectSession(session_id) => {
                apply_meta_rpc(
                    client,
                    app,
                    terminal,
                    ids,
                    "session.select_session",
                    json!({ "id": client.session_id, "session_path": session_id }),
                    true,
                )
                .await?;
            }
            AppAction::RewindCheckpoint | AppAction::ConfirmRewind => {
                apply_meta_rpc(
                    client,
                    app,
                    terminal,
                    ids,
                    "session.rewind_checkpoint",
                    json!({ "id": client.session_id }),
                    true,
                )
                .await?;
            }
            // Remaining actions are **client-local terminal/OS affordances** the
            // server cannot perform (no terminal, no clipboard) — `$EDITOR`,
            // clipboard copy/paste, mouse-capture toggle — or the richer rewind
            // *viewer* (needs the server snapshot timeline, a follow-up). Surface
            // an honest notice rather than silently dropping.
            other => {
                app.push_block(RenderBlock::System {
                    id: ids.next(),
                    level: SystemLevel::Warn,
                    text: format!(
                        "{} is a local affordance — use the local `zo` REPL",
                        action_label(&other)
                    ),
                });
                app.draw_frame(terminal).map_err(tui_err)?;
            }
        }
    }
}

fn next_attach_turn_generation(turn_generation: &mut u64) -> u64 {
    *turn_generation = turn_generation.saturating_add(1);
    *turn_generation
}

fn settle_attached_turn(app: &mut App, outcome: &Result<(), AttachError>) {
    if matches!(outcome, Err(AttachError::TurnRejected(_))) {
        app.abort_turn();
    } else {
        app.end_turn();
    }
}

/// Gate-throttled streamed-frame repaint for the socket loop: drip + draw when
/// the shared frame budget allows — feeding the measured draw cost back so the
/// cadence adapts to slow emulators — otherwise defer to the next tick via
/// `dirty`.
fn draw_streamed_socket_frame(
    app: &mut App,
    terminal: &mut TuiTerminal,
    frame_gate: &mut StreamFrameGate,
    dirty: &mut bool,
) -> Result<(), AttachError> {
    if frame_gate.on_stream_update(Instant::now()).draws_now() {
        app.drip_stream();
        let draw_started = Instant::now();
        app.draw_frame(terminal).map_err(tui_err)?;
        frame_gate.note_draw_cost(draw_started.elapsed());
        frame_gate.note_stream_draw(Instant::now());
        *dirty = false;
    } else {
        *dirty = true;
    }
    Ok(())
}

/// Drive one turn over the socket: send `session.run_turn`, decode streamed
/// render frames into the App, animate the spinner on the tick, and return when
/// the terminal response arrives.
async fn socket_run_turn(
    client: &mut AttachClient,
    app: &mut App,
    terminal: &mut TuiTerminal,
    ids: &BlockIdGen,
    input: &str,
    watermark: &Arc<AtomicU64>,
    force_resync_tx: &mpsc::Sender<()>,
) -> Result<(), AttachError> {
    use futures_util::StreamExt;

    let id = client.next_request_id();
    // Reuse the request id as the cancellable turn id (F4): the server registers
    // a cancel hook under it so a mid-turn Ctrl+C can interrupt the turn.
    let turn_id = id;
    client
        .send(RpcRequest::new(
            id,
            "session.run_turn",
            json!({ "id": client.session_id, "input": input, "turn_id": turn_id }),
        ))
        .await?;

    let bind_addr = client.bind_addr.clone();
    let mut interval = tokio::time::interval(ANIMATION_TICK_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut frame_gate = StreamFrameGate::new_ready(Instant::now(), STREAM_FRAME_INTERVAL);
    let mut dirty = false;
    // The App is not driving input during a turn, so this is the only consumer
    // of terminal events — used solely to catch Ctrl+C for cancellation.
    let mut events = crossterm::event::EventStream::new();
    let mut cancel_sent = false;
    loop {
        tokio::select! {
            line = client.reader.next_line() => {
                let line = line.map_err(AttachError::Io)?.ok_or(AttachError::Closed)?;
                let value: JsonValue = match serde_json::from_str(&line) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                if is_response_line(&value) {
                    let response = validate_response(&value, id).ok_or_else(|| {
                        AttachError::Rpc("session.run_turn: invalid JSON-RPC response".to_string())
                    })?;
                    // Do not draw one more active-spinner frame after the server
                    // has reported turn completion. The caller immediately calls
                    // `end_turn()` and paints the settled no-spinner frame.
                    if let Some(error) = response.get("error") {
                        let message = error
                            .get("message")
                            .and_then(JsonValue::as_str)
                            .expect("validated error message");
                        let code = error
                            .get("code")
                            .and_then(JsonValue::as_i64)
                            .expect("validated error code");
                        return Err(run_turn_response_error(code, message));
                    }
                    apply_run_turn_boundary(app, &response, watermark, force_resync_tx)?;
                    return Ok(());
                }
                if let Ok(block) = render_block_from_value(&value) {
                    // Mutate immediately, but share the same frame budget as the
                    // local REPL so fast remote streams do not alternate socket-
                    // driven full redraws with tick-driven redraws.
                    app.push_block(block);
                    draw_streamed_socket_frame(app, terminal, &mut frame_gate, &mut dirty)?;
                }
            }
            _ = interval.tick() => {
                app.advance_tick();
                let tick_stream_work = app.turn_activity().is_some() || app.stream_pending();
                let tick_has_work = dirty || tick_stream_work;
                let tick_now = Instant::now();
                let decision = if tick_stream_work {
                    frame_gate.on_stream_tick(tick_now, tick_has_work)
                } else {
                    frame_gate.on_tick(tick_now, tick_has_work)
                };
                if decision.draws_now() {
                    app.draw_frame(terminal).map_err(tui_err)?;
                    if app.turn_activity().is_some() || app.stream_pending() {
                        frame_gate.note_stream_draw(Instant::now());
                    }
                    dirty = false;
                }
            }
            // Terminal input mid-turn: either answer a forwarded permission
            // prompt (F2) or cancel the turn (F4).
            maybe_event = events.next(), if !cancel_sent => {
                if let Some(Ok(crossterm::event::Event::Key(key))) = maybe_event {
                    if let Some(prompt_id) = app.active_prompt().map(|prompt| prompt.id.0) {
                        // F2: a forwarded permission prompt is open. Map the key
                        // to a decision and answer over a fresh connection (the
                        // primary is mid-stream). The reconstructed prompt's
                        // block id *is* the server's routing `prompt_id`.
                        // Non-decision keys leave the modal open.
                        if let Some(decision) = key_to_permission_decision(&key, app) {
                            app.take_active_prompt();
                            send_permission_respond(&bind_addr, prompt_id, decision).await;
                            app.draw_frame(terminal).map_err(tui_err)?;
                        }
                    } else if key.kind == crossterm::event::KeyEventKind::Press
                        && key.code == crossterm::event::KeyCode::Char('c')
                        && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        // F4: cancel the turn. The server closes the render
                        // channel and the turn unwinds, ending this loop with a
                        // cancelled response.
                        cancel_sent = true;
                        send_cancel_turn(&bind_addr, &client.session_id, turn_id).await;
                        app.push_block(RenderBlock::System {
                            id: ids.next(),
                            level: SystemLevel::Info,
                            text: "cancelling…".to_string(),
                        });
                        app.draw_frame(terminal).map_err(tui_err)?;
                    }
                }
            }
        }
    }
}

fn run_turn_response_error(code: i64, message: &str) -> AttachError {
    let error = format!("{code}: {message}");
    if run_turn_rejection_code(code) {
        AttachError::TurnRejected(error)
    } else {
        AttachError::Rpc(error)
    }
}

fn run_turn_rejection_code(code: i64) -> bool {
    matches!(
        code,
        crate::serve_protocol::CODE_INVALID_PARAMS
            | crate::serve_protocol::CODE_NO_SUCH_SESSION
            | crate::serve_protocol::CODE_UNAUTHORIZED
            | crate::serve_protocol::CODE_HELM_HELD
    )
}

fn apply_run_turn_boundary(
    app: &mut App,
    response: &JsonValue,
    watermark: &AtomicU64,
    force_resync_tx: &mpsc::Sender<()>,
) -> Result<(), AttachError> {
    let Some(next_seq) = response
        .get("result")
        .and_then(|result| result.get("next_seq"))
    else {
        // Pre-boundary servers completed the turn successfully but did not
        // return a sequence fence. Treat that response as legacy success and
        // request one coalesced authoritative spectator replacement.
        watermark.store(0, Ordering::Release);
        let _ = force_resync_tx.try_send(());
        return Ok(());
    };
    let next_seq = next_seq
        .as_u64()
        .ok_or_else(|| AttachError::Rpc("run_turn next_seq must be a u64".to_string()))?;
    watermark.store(next_seq, Ordering::Release);
    app.advance_spectator_floor(next_seq);
    Ok(())
}

/// All durable state used by the reconnecting spectator supervisor.
struct SpectatorContext {
    bind_addr: String,
    session_id: String,
    subscription_id: Arc<AtomicU64>,
    spectator_tx: mpsc::Sender<SpectatorEvent>,
    own_turn: Arc<AtomicBool>,
    /// Completed own-turn boundary. It is written before `own_turn` is released.
    watermark: Arc<AtomicU64>,
    auth_token: Option<String>,
    force_resync_rx: mpsc::Receiver<()>,
}

#[derive(Clone)]
struct SubscribeReply {
    history: JsonValue,
    next_seq: u64,
    /// Optional v2 replacement floor. Absent keeps compatibility with older
    /// servers that negotiated subscription identities before this fence.
    floor: Option<u64>,
    /// `None` identifies a pre-v2 server that accepted the request but did not
    /// negotiate the resync subscription capability.
    subscription_id: Option<u64>,
}

/// Start a lifetime supervisor. Only the App ingress closing is terminal; every
/// transport/protocol failure reconnects with capped exponential backoff.
fn open_spectator_stream(context: SpectatorContext) {
    tokio::spawn(async move { spectator_supervisor(context).await; });
}

const INITIAL_SPECTATOR_RECONNECT_DELAY: Duration = Duration::from_secs(1);
const MAX_SPECTATOR_RECONNECT_DELAY: Duration = Duration::from_secs(30);

/// Legacy hydration discards the first subscriber socket after a lock-taking
/// load, then retains the fresh subscriber socket established immediately after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyHydration {
    Initial,
    FreshSubscriber,
}

/// The supervisor needs to distinguish a failed connection attempt from one
/// which completed the authoritative replacement boundary before disconnecting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpectatorConnectionResult {
    /// The App ingress closed, so the detached client must not reconnect.
    AppClosed,
    /// Connect, subscribe, or the initial authoritative replacement failed.
    ReconnectAfterFailure,
    /// The initial authoritative replacement was acknowledged before disconnect.
    ReconnectAfterReady,
    /// A legacy `session.load` replacement was applied; discard the old
    /// subscriber socket and immediately acquire a fresh boundary snapshot.
    ReconnectAfterAckedLegacyLoad,
    /// The App rejected a stale replacement; reconnect for a new boundary lock
    /// rather than retrying an identical socket snapshot in a busy loop.
    ReconnectAfterStale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplaceResult {
    AppClosed,
    LoadFailed,
    Applied,
    AppliedLegacy,
    Stale,
}

async fn spectator_supervisor(mut context: SpectatorContext) {
    let mut delay = INITIAL_SPECTATOR_RECONNECT_DELAY;
    let mut boundary = false;
    let mut legacy_hydration = LegacyHydration::Initial;
    loop {
        let result = spectator_connection(&mut context, boundary, legacy_hydration).await;
        if result == SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad {
            legacy_hydration = LegacyHydration::FreshSubscriber;
        }
        let Some(sleep_delay) = reconnect_delay(&mut delay, result) else {
            // The App has gone away: do not keep reconnecting after detach.
            return;
        };
        boundary = true;
        tokio::time::sleep(sleep_delay).await;
    }
}

/// Return the delay before the next reconnect and maintain the following delay.
/// A completed authoritative replacement resets backoff; app closure has none.
fn reconnect_delay(
    delay: &mut Duration,
    result: SpectatorConnectionResult,
) -> Option<Duration> {
    match result {
        SpectatorConnectionResult::AppClosed => None,
        SpectatorConnectionResult::ReconnectAfterFailure => {
            let sleep_delay = *delay;
            *delay = (*delay * 2).min(MAX_SPECTATOR_RECONNECT_DELAY);
            Some(sleep_delay)
        }
        SpectatorConnectionResult::ReconnectAfterReady => {
            *delay = INITIAL_SPECTATOR_RECONNECT_DELAY * 2;
            Some(INITIAL_SPECTATOR_RECONNECT_DELAY)
        }
        SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad => Some(Duration::ZERO),
        // A stale replacement must yield before acquiring another boundary
        // snapshot. Otherwise an App floor ahead of an idle server's sequence
        // would reconnect at zero delay forever. This is not a successful
        // authoritative ACK, so retain the current backoff unchanged.
        SpectatorConnectionResult::ReconnectAfterStale => Some(*delay),
    }
}

/// Returns [`SpectatorConnectionResult::AppClosed`] exclusively when App ingress
/// closes. A disconnect after the initial Replace ACK is `ReconnectAfterReady`;
/// all earlier transport/protocol failures retain the accumulated backoff.
async fn spectator_connection(
    context: &mut SpectatorContext,
    boundary: bool,
    legacy_hydration: LegacyHydration,
) -> SpectatorConnectionResult {
    let Ok(stream) = TcpStream::connect(&context.bind_addr).await else {
        return SpectatorConnectionResult::ReconnectAfterFailure;
    };
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();
    let mut request_id = 1;
    let Some((reply, pre_ack)) = subscribe_on_stream(
        &mut write_half,
        &mut lines,
        request_id,
        boundary,
        &context.session_id,
        context.auth_token.clone(),
    )
    .await
    else {
        return SpectatorConnectionResult::ReconnectAfterFailure;
    };
    request_id += 1;
    let mut minimum_seq = replacement_floor(&reply);
    context
        .subscription_id
        .store(reply.subscription_id.unwrap_or(0), Ordering::Release);
    match send_replace(context, reply, pre_ack).await {
        ReplaceResult::AppClosed => return SpectatorConnectionResult::AppClosed,
        ReplaceResult::LoadFailed => return SpectatorConnectionResult::ReconnectAfterFailure,
        ReplaceResult::Applied => {}
        ReplaceResult::AppliedLegacy => {
            if legacy_hydration == LegacyHydration::Initial {
                return SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad;
            }
        }
        ReplaceResult::Stale => return SpectatorConnectionResult::ReconnectAfterStale,
    }

    loop {
        tokio::select! {
            Some(()) = context.force_resync_rx.recv() => {
                let Some((reply, pre_ack)) = subscribe_on_stream(
                    &mut write_half, &mut lines, request_id, true,
                    &context.session_id, context.auth_token.clone(),
                ).await else { return SpectatorConnectionResult::ReconnectAfterReady; };
                request_id = request_id.saturating_add(1);
                minimum_seq = replacement_floor(&reply);
                context
                    .subscription_id
                    .store(reply.subscription_id.unwrap_or(0), Ordering::Release);
                match send_replace(context, reply, pre_ack).await {
                    ReplaceResult::AppClosed => return SpectatorConnectionResult::AppClosed,
                    ReplaceResult::LoadFailed => return SpectatorConnectionResult::ReconnectAfterFailure,
                    ReplaceResult::Applied => {}
                    ReplaceResult::AppliedLegacy => {
                        if legacy_hydration == LegacyHydration::Initial {
                            return SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad;
                        }
                    }
                    ReplaceResult::Stale => return SpectatorConnectionResult::ReconnectAfterStale,
                }
            }
            read = lines.next_line() => {
                let Ok(Some(raw)) = read else { return SpectatorConnectionResult::ReconnectAfterReady; };
                let Ok(value) = serde_json::from_str::<JsonValue>(&raw) else { return SpectatorConnectionResult::ReconnectAfterReady; };
                if is_response_line(&value) {
                    // There are no outstanding non-subscribe RPCs here.
                    return SpectatorConnectionResult::ReconnectAfterReady;
                }
                if is_marker(&value, context.subscription_id.load(Ordering::Acquire), minimum_seq) {
                    let Some((reply, pre_ack)) = subscribe_on_stream(
                        &mut write_half, &mut lines, request_id, true,
                        &context.session_id, context.auth_token.clone(),
                    ).await else { return SpectatorConnectionResult::ReconnectAfterReady; };
                    request_id = request_id.saturating_add(1);
                    minimum_seq = replacement_floor(&reply);
                    context
                        .subscription_id
                        .store(reply.subscription_id.unwrap_or(0), Ordering::Release);
                    match send_replace(context, reply, pre_ack).await {
                        ReplaceResult::AppClosed => return SpectatorConnectionResult::AppClosed,
                        ReplaceResult::LoadFailed => return SpectatorConnectionResult::ReconnectAfterFailure,
                        ReplaceResult::Applied => {}
                        ReplaceResult::AppliedLegacy => {
                            if legacy_hydration == LegacyHydration::Initial {
                                return SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad;
                            }
                        }
                        ReplaceResult::Stale => return SpectatorConnectionResult::ReconnectAfterStale,
                    }
                    continue;
                }
                if !process_spectator_value(context, value, minimum_seq).await {
                    return SpectatorConnectionResult::AppClosed;
                }
            }
        }
    }
}

fn is_marker(value: &JsonValue, subscription_id: u64, floor: u64) -> bool {
    match value.get("type").and_then(JsonValue::as_str) {
        Some("resync") => true, // legacy server recovery marker
        Some("marker") => subscription_id != 0
            && value.get("subscription_id").and_then(JsonValue::as_u64) == Some(subscription_id)
            && value.get("next_seq").and_then(JsonValue::as_u64).is_some_and(|next_seq| next_seq >= floor),
        _ => false,
    }
}

/// Send a typed subscribe request and accept only its exact response id/schema.
/// Pre-ACK controls/frames are retained verbatim and hard-capped; overflow is a
/// protocol error that causes reconnect, never a silent tail drop.
async fn subscribe_on_stream(
    writer: &mut OwnedWriteHalf,
    lines: &mut tokio::io::Lines<BufReader<OwnedReadHalf>>,
    request_id: u64,
    boundary: bool,
    session_id: &str,
    token: Option<String>,
) -> Option<(SubscribeReply, VecDeque<JsonValue>)> {
    let request = RpcRequest::new(
        request_id,
        "session.subscribe",
        json!({ "id": session_id, "boundary": boundary, "resync_v2": true }),
    ).with_token(token);
    let mut raw = serde_json::to_vec(&request).ok()?;
    raw.push(b'\n');
    writer.write_all(&raw).await.ok()?;
    writer.flush().await.ok()?;
    let mut pre_ack = VecDeque::new();
    loop {
        let raw = lines.next_line().await.ok()??;
        let value = serde_json::from_str::<JsonValue>(&raw).ok()?;
        if is_response_line(&value) {
            let reply = parse_subscribe_response(&value, request_id, session_id)?;
            return Some((reply, pre_ack));
        }
        if !push_pre_ack(&mut pre_ack, value) {
            return None;
        }
    }
}

/// Retain every pre-ACK item until a replacement can filter it by boundary.
/// `false` is a hard protocol failure: never pop a valid tail to make room.
fn push_pre_ack(queue: &mut VecDeque<JsonValue>, value: JsonValue) -> bool {
    if queue.len() >= MAX_DRAIN_PER_TICK {
        return false;
    }
    queue.push_back(value);
    true
}

fn parse_subscribe_response(
    value: &JsonValue,
    request_id: u64,
    session_id: &str,
) -> Option<SubscribeReply> {
    let response = validate_response(value, request_id)?;
    if response.get("error").is_some() {
        return None;
    }
    let result = response.get("result")?.as_object()?;
    if result.get("id")?.as_str()? != session_id
        || !result.get("history")?.is_array()
        || !matches!(result.get("helm"), Some(JsonValue::Null | JsonValue::String(_)))
    {
        return None;
    }
    Some(SubscribeReply {
        history: result.get("history")?.clone(),
        next_seq: result.get("next_seq")?.as_u64()?,
        floor: match result.get("floor") {
            None => None,
            Some(floor) => Some(floor.as_u64()?),
        },
        // A pre-v2 server omits `subscription_id`; keep that stream alive with
        // legacy `resync` controls. A present zero is invalid, not legacy.
        subscription_id: match result.get("subscription_id") {
            None => None,
            Some(subscription_id) => {
                let subscription_id = subscription_id.as_u64()?;
                if subscription_id == 0 {
                    return None;
                }
                Some(subscription_id)
            }
        },
    })
}

/// Read a legacy server's durable history after it has released an active turn.
/// Unlike a pre-v2 subscribe snapshot, `session.load` takes the session lock,
/// so it cannot describe the history from before an already-streaming turn.
async fn load_legacy_history(context: &SpectatorContext) -> Option<JsonValue> {
    let stream = TcpStream::connect(&context.bind_addr).await.ok()?;
    let (read_half, mut write_half) = stream.into_split();
    let request = RpcRequest::new(
        1,
        "session.load",
        json!({ "id": context.session_id }),
    )
    .with_token(context.auth_token.clone());
    let mut line = serde_json::to_vec(&request).ok()?;
    line.push(b'\n');
    write_half.write_all(&line).await.ok()?;
    write_half.flush().await.ok()?;

    let mut lines = BufReader::new(read_half).lines();
    loop {
        let raw = lines.next_line().await.ok()??;
        let value = serde_json::from_str::<JsonValue>(&raw).ok()?;
        if !is_response_line(&value) {
            continue;
        }
        let response = validate_response(&value, 1)?;
        let result = response.get("result")?.as_object()?;
        if result.get("id")?.as_str()? != context.session_id || !result.get("history")?.is_array() {
            return None;
        }
        return result.get("history").cloned();
    }
}

fn replacement_floor(reply: &SubscribeReply) -> u64 {
    reply.floor.unwrap_or_else(|| {
        if reply.subscription_id.is_none() {
            reply.next_seq
        } else {
            0
        }
    })
}

async fn send_replace(
    context: &mut SpectatorContext,
    reply: SubscribeReply,
    mut pre_ack: VecDeque<JsonValue>,
) -> ReplaceResult {
    let legacy = reply.subscription_id.is_none();
    let next_seq = reply.next_seq;
    // Old servers acknowledge subscribe with a cached pre-turn snapshot. Never
    // make that snapshot authoritative: a lock-taking load waits for the live
    // turn, so its history includes every frame the cached snapshot omitted.
    let history = if legacy {
        let Some(history) = load_legacy_history(context).await else {
            return ReplaceResult::LoadFailed;
        };
        history
    } else {
        reply.history
    };
    let ids = BlockIdGen::default();
    let blocks = history_entries_to_blocks(&json!({ "history": history }), &ids);
    let mut post_boundary = VecDeque::new();
    // Pre-ACK frames belong to the v2 sequence fence. A legacy load already
    // includes its active turn, so replaying its buffered tail would duplicate
    // the authoritative durable history.
    if !legacy {
        while let Some(value) = pre_ack.pop_front() {
            let Some(seq) = value.get("frame_seq").and_then(JsonValue::as_u64) else { continue; };
            if seq >= reply.next_seq {
                if let Ok(block) = render_block_from_value(&value) {
                    post_boundary.push_back(block);
                }
            }
        }
    }
    let (ack, ack_rx) = oneshot::channel();
    if context.spectator_tx.send(SpectatorEvent::Replace { blocks, post_boundary, next_seq, ack }).await.is_err() {
        return ReplaceResult::AppClosed;
    }
    match ack_rx.await {
        Ok(AckVerdict::Applied) if legacy => ReplaceResult::AppliedLegacy,
        Ok(AckVerdict::Applied) => ReplaceResult::Applied,
        Ok(AckVerdict::Stale) => ReplaceResult::Stale,
        Err(_) => ReplaceResult::AppClosed,
    }
}

/// Feed a normal spectator frame through the one ordered ingress. Own-turn
/// duplicate elimination checks the release-published watermark after reading
/// the own-turn gate, so all pre-boundary frames are dropped.
async fn process_spectator_value(
    context: &SpectatorContext,
    value: JsonValue,
    minimum_seq: u64,
) -> bool {
    let Some(seq) = value.get("frame_seq").and_then(JsonValue::as_u64) else {
        return true;
    };
    if !should_forward_spectator_frame(
        context.own_turn.load(Ordering::Acquire),
        seq,
        minimum_seq,
        context.watermark.load(Ordering::Acquire),
    ) {
        return true;
    }
    let Ok(block) = render_block_from_value(&value) else {
        return true;
    };
    enqueue_spectator_frame(
        &context.spectator_tx,
        &context.own_turn,
        &context.watermark,
        seq,
        minimum_seq,
        block,
    )
    .await
}

/// Reserve ingress capacity before revalidating the own-turn fence. A permit
/// makes the final enqueue non-awaiting, closing the full-channel TOCTOU gap.
async fn enqueue_spectator_frame(
    spectator_tx: &mpsc::Sender<SpectatorEvent>,
    own_turn: &AtomicBool,
    watermark: &AtomicU64,
    seq: u64,
    minimum_seq: u64,
    block: RenderBlock,
) -> bool {
    let Ok(permit) = spectator_tx.reserve().await else {
        return false;
    };
    if should_forward_spectator_frame(
        own_turn.load(Ordering::Acquire),
        seq,
        minimum_seq,
        watermark.load(Ordering::Acquire),
    ) {
        permit.send(SpectatorEvent::Frame { frame_seq: seq, block });
    }
    true
}

/// The acquire of `own_turn` pairs with the primary turn's Release after its
/// watermark store. Therefore `own_turn == false` observes the completed fence.
const fn should_forward_spectator_frame(
    own_turn: bool,
    frame_seq: u64,
    minimum_seq: u64,
    watermark: u64,
) -> bool {
    !own_turn && frame_seq >= minimum_seq && frame_seq >= watermark
}

/// `/steer <text>` → push a mid-turn steering message to the session's in-flight
/// turn over the primary connection. A `STEER_DENIED` (no active turn / not the
/// helm) is surfaced as a notice rather than silently dropped.
async fn handle_steer(
    client: &mut AttachClient,
    app: &mut App,
    terminal: &mut TuiTerminal,
    ids: &BlockIdGen,
    text: &str,
) -> Result<(), AttachError> {
    if text.is_empty() {
        return Ok(());
    }
    let (level, message) = match client
        .request(
            "session.steer",
            json!({ "id": client.session_id, "text": text }),
        )
        .await
    {
        Ok(_) => (SystemLevel::Info, format!("steering sent: {text}")),
        Err(error) => (SystemLevel::Warn, format!("{error}")),
    };
    app.push_block(RenderBlock::System {
        id: ids.next(),
        level,
        text: message,
    });
    app.draw_frame(terminal).map_err(tui_err)?;
    Ok(())
}

/// `/roster` → list the peers connected to this session and the current helm.
async fn handle_roster(
    client: &mut AttachClient,
    app: &mut App,
    terminal: &mut TuiTerminal,
    ids: &BlockIdGen,
) -> Result<(), AttachError> {
    match client
        .request("session.roster", json!({ "id": client.session_id }))
        .await
    {
        Ok(roster) => {
            let helm = roster
                .get("helm")
                .and_then(JsonValue::as_str)
                .unwrap_or("(none)");
            let viewers = roster.get("viewers").and_then(JsonValue::as_u64).unwrap_or(0);
            let peers = roster
                .get("peers")
                .and_then(JsonValue::as_array)
                .map(|peers| {
                    peers
                        .iter()
                        .filter_map(|peer| {
                            let label = peer.get("label").and_then(JsonValue::as_str)?;
                            let cap = peer.get("capability").and_then(JsonValue::as_str).unwrap_or("");
                            Some(format!("{label} ({cap})"))
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            app.push_block(RenderBlock::System {
                id: ids.next(),
                level: SystemLevel::Info,
                text: format!("👁 {viewers} watching · helm: {helm} · peers: {peers}"),
            });
        }
        Err(error) => {
            app.push_block(RenderBlock::System {
                id: ids.next(),
                level: SystemLevel::Error,
                text: format!("{error}"),
            });
        }
    }
    app.draw_frame(terminal).map_err(tui_err)?;
    Ok(())
}

/// Fire-and-forget `permission.respond` on a fresh connection (F2). The primary
/// connection is mid-stream awaiting the server's parked responder, so the
/// decision must arrive on a second one. Best-effort: a dropped decision is
/// caught by the server-side prompt timeout, which hard-denies.
async fn send_permission_respond(
    bind_addr: &str,
    prompt_id: u64,
    decision: RenderPermissionDecision,
) {
    let Ok(stream) = TcpStream::connect(bind_addr).await else {
        return;
    };
    let (read_half, mut write_half) = stream.into_split();
    let request = RpcRequest::new(
        1,
        "permission.respond",
        json!({ "prompt_id": prompt_id, "decision": permission_decision_tag(decision) }),
    )
    .with_token(crate::serve_auth::token_from_env());
    let Ok(mut line) = serde_json::to_vec(&request) else {
        return;
    };
    line.push(b'\n');
    if write_half.write_all(&line).await.is_err() {
        return;
    }
    let _ = write_half.flush().await;
    // Drain the single response so the server's write side completes cleanly.
    let mut lines = BufReader::new(read_half).lines();
    let _ = lines.next_line().await;
}

/// Fire-and-forget `session.cancel_turn` on a fresh connection. The primary
/// connection is mid-stream — the server cannot read a cancel there until the
/// turn ends — so a second connection delivers it. Best-effort: a failed cancel
/// just means the turn runs to completion.
async fn send_cancel_turn(bind_addr: &str, session_id: &str, turn_id: u64) {
    let Ok(stream) = TcpStream::connect(bind_addr).await else {
        return;
    };
    let (read_half, mut write_half) = stream.into_split();
    let request = RpcRequest::new(
        1,
        "session.cancel_turn",
        json!({ "turn_id": turn_id, "session_id": session_id }),
    )
    .with_token(crate::serve_auth::token_from_env());
    let Ok(mut line) = serde_json::to_vec(&request) else {
        return;
    };
    line.push(b'\n');
    if write_half.write_all(&line).await.is_err() {
        return;
    }
    let _ = write_half.flush().await;
    // Drain the single response so the server's write side completes cleanly.
    let mut lines = BufReader::new(read_half).lines();
    let _ = lines.next_line().await;
}

/// Seed the transcript with the loaded conversation history so a reattach shows
/// the prior turns.
fn seed_history(app: &mut App, ids: &BlockIdGen, loaded: &JsonValue) {
    for block in history_entries_to_blocks(loaded, ids) {
        app.push_block(block);
    }
}

/// Decode a subscribe hydration result's `history` array into transcript
/// blocks. Meta-operation reseeding also reuses this projection.
fn history_entries_to_blocks(loaded: &JsonValue, ids: &BlockIdGen) -> Vec<RenderBlock> {
    let Some(entries) = loaded.get("history").and_then(JsonValue::as_array) else {
        return Vec::new();
    };
    let mut blocks = Vec::with_capacity(entries.len());
    for entry in entries {
        let role = entry.get("role").and_then(JsonValue::as_str).unwrap_or("");
        let text = entry
            .get("text")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_string();
        let block = match role {
            "user" => RenderBlock::UserMessage {
                id: ids.next(),
                text,
            },
            "assistant" => RenderBlock::TextDelta {
                id: ids.next(),
                text,
                done: true,
            },
            _ => RenderBlock::System {
                id: ids.next(),
                level: SystemLevel::Info,
                text,
            },
        };
        blocks.push(block);
    }
    blocks
}

/// Drive a session-mutating meta RPC (model / permission / session / rewind),
/// then reflect the outcome locally: surface the server's `message`, optionally
/// reseed the transcript (`session.load`, for a session switch or rewind that
/// changes history), and refresh the sidebar from `session.info`. An RPC error
/// is shown rather than silently dropped.
async fn apply_meta_rpc(
    client: &mut AttachClient,
    app: &mut App,
    terminal: &mut TuiTerminal,
    ids: &BlockIdGen,
    method: &str,
    params: JsonValue,
    reseed: bool,
) -> Result<(), AttachError> {
    match client.request(method, params).await {
        Ok(result) => {
            if let Some(message) = result.get("message").and_then(JsonValue::as_str) {
                app.push_block(RenderBlock::System {
                    id: ids.next(),
                    level: SystemLevel::Info,
                    text: message.to_string(),
                });
            }
            if reseed {
                if let Ok(loaded) = client
                    .request("session.load", json!({ "id": client.session_id }))
                    .await
                {
                    app.reset_session_view();
                    seed_history(app, ids, &loaded);
                }
            }
            if let Ok(info) = client
                .request("session.info", json!({ "id": client.session_id }))
                .await
            {
                apply_session_info(app, &info);
            }
        }
        Err(error) => {
            app.push_block(RenderBlock::System {
                id: ids.next(),
                level: SystemLevel::Error,
                text: format!("{error}"),
            });
        }
    }
    app.draw_frame(terminal).map_err(tui_err)?;
    Ok(())
}

/// Hydrate the sidebar from a `session.info` result.
fn apply_session_info(app: &mut App, result: &JsonValue) {
    let Ok(info) = serde_json::from_value::<SessionInfo>(result.clone()) else {
        return;
    };
    let perm = crate::permission_mode::permission_mode_from_label(&info.permission_mode);
    let context_limit = api::context_window_for_model(&info.model);
    app.set_session_meta(
        &info.model,
        context_limit,
        perm,
        PathBuf::from(&info.cwd),
        info.git_branch,
    );
}

/// Short label for an unsupported [`AppAction`], used in the "not available"
/// notice.
fn action_label(action: &AppAction) -> &'static str {
    match action {
        AppAction::SelectModel(_) => "model switching",
        AppAction::ConnectApiKey { .. } => "provider API-key setup",
        AppAction::ConnectCustomProvider(_) => "custom provider setup",
        AppAction::SelectPermission(_) => "permission switching",
        AppAction::SelectSession(_) => "session switching",
        AppAction::Editor => "the external editor",
        AppAction::RewindCheckpoint
        | AppAction::ConfirmRewind
        | AppAction::RewindTo(_)
        | AppAction::OpenRewindViewer => "rewind",
        AppAction::OpenWorkflowViewer => "the workflow viewer",
        AppAction::OpenAgentInViewer(_) => "the agent viewer",
        AppAction::AckTeamInboxUpdate(_)
        | AppAction::IncludeTeamInboxUpdate(_)
        | AppAction::RefreshTeamInboxViewer => "the team inbox viewer",
        AppAction::ToggleTool { .. } => "runtime tool toggles",
        AppAction::SaveSmartSettings(_) => "Smart Router settings",
        AppAction::DeepTier(_) => "deep-tier model settings",
        AppAction::ClipboardCopy(_)
        | AppAction::ClipboardCopyBlock(_)
        | AppAction::ClipboardPaste => "clipboard",
        AppAction::Redraw | AppAction::Submit(_) | AppAction::Quit | AppAction::None => {
            "that action"
        }
    }
}

/// Map a TUI render error into [`AttachError`].
fn tui_err(error: impl std::fmt::Display) -> AttachError {
    AttachError::Tui(error.to_string())
}


#[cfg(test)]
mod spectator_tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use zo_cli::tui::Theme;

    fn subscribe_response(id: u64, result: &JsonValue) -> JsonValue {
        json!({ "jsonrpc": "2.0", "id": id, "result": result })
    }

    #[test]
    fn attach_response_validator_rejects_wrong_jsonrpc_id_and_missing_result_error() {
        let valid = subscribe_response(7, &json!({ "ok": true }));
        assert_eq!(validate_response(&valid, 7), Some(valid.clone()));

        let wrong_version = json!({ "jsonrpc": "1.0", "id": 7, "result": {} });
        assert!(validate_response(&wrong_version, 7).is_none());
        let wrong_id = json!({ "jsonrpc": "2.0", "id": 8, "result": {} });
        assert!(validate_response(&wrong_id, 7).is_none());
        let both = json!({
            "jsonrpc": "2.0", "id": 7, "result": {},
            "error": { "code": -1, "message": "no" },
        });
        assert!(validate_response(&both, 7).is_none());
        let neither = json!({ "jsonrpc": "2.0", "id": 7 });
        assert!(validate_response(&neither, 7).is_none());
        let bad_error = json!({
            "jsonrpc": "2.0", "id": 7,
            "error": { "code": "not-a-number", "message": 9 },
        });
        assert!(validate_response(&bad_error, 7).is_none());
    }

    #[test]
    fn rejected_attached_turn_is_cold_immediately() {
        let theme = Theme::zo();
        let cold_caret = theme.palette.bright;
        let (_blocks, block_rx) = mpsc::channel::<RenderBlock>(1);
        let (cmd_tx, _commands) = mpsc::channel::<AgentCommand>(1);
        let mut app = App::new(theme, block_rx, cmd_tx);
        app.begin_turn_with_generation(1);
        let rejection = Err(AttachError::TurnRejected(
            "-32004: helm held".to_string(),
        ));

        settle_attached_turn(&mut app, &rejection);

        assert!(app.turn_activity().is_none());
        let mut terminal = Terminal::new(TestBackend::new(80, 16)).expect("test terminal");
        app.draw(&mut terminal).expect("draw rejected turn state");
        let buffer = terminal.backend().buffer();
        let caret = (0..16)
            .flat_map(|y| (0..80).map(move |x| (x, y)))
            .map(|position| &buffer[position])
            .find(|cell| cell.symbol() == "❯")
            .expect("input caret is painted");
        assert_eq!(
            caret.fg, cold_caret,
            "a rejected turn must paint Cold, not a false cooling ramp"
        );
        assert!(run_turn_rejection_code(
            crate::serve_protocol::CODE_HELM_HELD
        ));
        assert!(!run_turn_rejection_code(
            crate::serve_protocol::CODE_CANCELLED
        ));
    }

    #[test]
    fn attached_turn_generations_advance_the_zo_verb() {
        let (_blocks, block_rx) = mpsc::channel::<RenderBlock>(1);
        let (cmd_tx, _commands) = mpsc::channel::<AgentCommand>(1);
        let mut app = App::new(Theme::no_color(), block_rx, cmd_tx);
        let mut turn_generation = 0;
        let ids = BlockIdGen::default();

        for (expected_generation, expected_verb) in [(1, "Planning"), (2, "Exploring")] {
            let generation = next_attach_turn_generation(&mut turn_generation);
            assert_eq!(generation, expected_generation);
            app.begin_turn_with_generation(generation);
            app.push_block(RenderBlock::TextDelta {
                id: ids.next(),
                text: "streaming".to_string(),
                done: false,
            });
            assert_eq!(
                app.turn_activity()
                    .expect("attached turn is active")
                    .current_action(),
                expected_verb
            );
            app.abort_turn();
        }
    }

    #[test]
    fn old_server_run_turn_without_next_seq_succeeds_and_requests_boundary_resync() {
        let (force_resync_tx, mut force_resync_rx) = mpsc::channel(1);
        let watermark = AtomicU64::new(99);
        let response = subscribe_response(5, &json!({ "model": "legacy" }));

        let (_blocks, block_rx) = mpsc::channel::<RenderBlock>(1);
        let (cmd_tx, _commands) = mpsc::channel::<AgentCommand>(1);
        let mut app = App::new(Theme::no_color(), block_rx, cmd_tx);
        apply_run_turn_boundary(&mut app, &response, &watermark, &force_resync_tx)
            .expect("legacy terminal result succeeds");
        assert_eq!(watermark.load(Ordering::Acquire), 0);
        assert!(force_resync_rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn legacy_subscribe_loads_completed_history_before_replace() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            let (primary, _) = listener.accept().await.expect("subscribe connection");
            let (primary_read, mut primary_write) = primary.into_split();
            let mut primary_lines = BufReader::new(primary_read).lines();
            let subscribe: JsonValue = serde_json::from_str(
                &primary_lines.next_line().await.expect("subscribe read").expect("subscribe line"),
            )
            .expect("subscribe JSON");
            assert_eq!(subscribe["method"], "session.subscribe");
            primary_write
                .write_all(
                    br#"{"jsonrpc":"2.0","id":1,"result":{"id":"session-a","history":[{"role":"assistant","text":"stale"}],"next_seq":3,"helm":"helm"}}"#,
                )
                .await
                .expect("legacy subscribe response");
            primary_write.write_all(b"\n").await.expect("newline");
            // This pre-turn frame is still buffered on the old subscriber
            // socket while the lock-taking load obtains completed history.
            primary_write
                .write_all(br#"{"type":"text_delta","text":"stale","frame_seq":2}"#)
                .await
                .expect("stale frame");
            primary_write.write_all(b"\n").await.expect("frame newline");
            primary_write.flush().await.expect("flush");

            let (loader, _) = listener.accept().await.expect("load connection");
            let (loader_read, mut loader_write) = loader.into_split();
            let mut loader_lines = BufReader::new(loader_read).lines();
            let load: JsonValue = serde_json::from_str(
                &loader_lines.next_line().await.expect("load read").expect("load line"),
            )
            .expect("load JSON");
            assert_eq!(load["method"], "session.load");
            loader_write
                .write_all(
                    br#"{"jsonrpc":"2.0","id":1,"result":{"id":"session-a","history":[{"role":"assistant","text":"complete"}]}}"#,
                )
                .await
                .expect("load response");
            loader_write.write_all(b"\n").await.expect("newline");
            loader_write.flush().await.expect("flush");
        });

        let (spectator_tx, mut spectator_rx) = mpsc::channel(1);
        let (_resync_tx, force_resync_rx) = mpsc::channel(1);
        let mut context = SpectatorContext {
            bind_addr: addr.to_string(),
            session_id: "session-a".to_string(),
            subscription_id: Arc::new(AtomicU64::new(0)),
            spectator_tx,
            own_turn: Arc::new(AtomicBool::new(false)),
            watermark: Arc::new(AtomicU64::new(0)),
            auth_token: None,
            force_resync_rx,
        };
        let mut connection = Box::pin(spectator_connection(
            &mut context,
            false,
            LegacyHydration::Initial,
        ));
        let event = tokio::select! {
            event = spectator_rx.recv() => event.expect("replacement"),
            result = &mut connection => panic!("connection ended before replacement: {result:?}"),
        };
        let SpectatorEvent::Replace { blocks, post_boundary, ack, .. } = event else {
            panic!("expected transcript replacement");
        };
        assert!(post_boundary.is_empty(), "legacy buffered frames are covered by the load");
        assert!(blocks.iter().any(|block| matches!(block, RenderBlock::TextDelta { text, .. } if text == "complete")));
        assert!(!blocks.iter().any(|block| matches!(block, RenderBlock::TextDelta { text, .. } if text == "stale")));
        ack.send(AckVerdict::Applied).expect("ack replacement");
        assert_eq!(connection.await, SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad);
        assert!(spectator_rx.try_recv().is_err(), "stale socket frame was fenced");
        server.await.expect("legacy server");
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn legacy_supervisor_reconnects_once_then_consumes_live_frames() {
        use std::sync::atomic::AtomicUsize;
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("address");
        let subscribe_count = Arc::new(AtomicUsize::new(0));
        let server_subscribe_count = Arc::clone(&subscribe_count);
        let (send_live_tx, send_live_rx) = oneshot::channel();
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let mut send_live_rx = Some(send_live_rx);
            let mut shutdown_rx = Some(shutdown_rx);
            for attempt in 0..2 {
                let (subscriber, _) = listener.accept().await.expect("subscribe connection");
                let (subscriber_read, mut subscriber_write) = subscriber.into_split();
                let mut subscriber_lines = BufReader::new(subscriber_read).lines();
                let subscribe: JsonValue = serde_json::from_str(
                    &subscriber_lines
                        .next_line()
                        .await
                        .expect("subscribe read")
                        .expect("subscribe line"),
                )
                .expect("subscribe JSON");
                assert_eq!(subscribe["method"], "session.subscribe");
                assert_eq!(subscribe["params"]["boundary"], attempt != 0);
                server_subscribe_count.fetch_add(1, Ordering::SeqCst);
                subscriber_write
                    .write_all(
                        br#"{"jsonrpc":"2.0","id":1,"result":{"id":"session-a","history":[{"role":"assistant","text":"cached"}],"next_seq":3,"helm":null}}"#,
                    )
                    .await
                    .expect("legacy subscribe response");
                subscriber_write.write_all(b"\n").await.expect("newline");
                subscriber_write.flush().await.expect("flush");

                let (loader, _) = listener.accept().await.expect("load connection");
                let (loader_read, mut loader_write) = loader.into_split();
                let mut loader_lines = BufReader::new(loader_read).lines();
                let load: JsonValue = serde_json::from_str(
                    &loader_lines
                        .next_line()
                        .await
                        .expect("load read")
                        .expect("load line"),
                )
                .expect("load JSON");
                assert_eq!(load["method"], "session.load");
                loader_write
                    .write_all(
                        br#"{"jsonrpc":"2.0","id":1,"result":{"id":"session-a","history":[{"role":"assistant","text":"complete"}]}}"#,
                    )
                    .await
                    .expect("load response");
                loader_write.write_all(b"\n").await.expect("newline");
                loader_write.flush().await.expect("flush");

                if attempt == 1 {
                    send_live_rx
                        .take()
                        .expect("live frame receiver")
                        .await
                        .expect("live frame signal");
                    subscriber_write
                        .write_all(br#"{"type":"text_delta","id":3,"text":"live","done":false,"frame_seq":3}"#)
                        .await
                        .expect("live frame");
                    subscriber_write.write_all(b"\n").await.expect("newline");
                    subscriber_write.flush().await.expect("flush");
                    shutdown_rx
                        .take()
                        .expect("shutdown receiver")
                        .await
                        .expect("shutdown signal");
                    subscriber_write
                        .write_all(br#"{"type":"text_delta","id":4,"text":"close","done":false,"frame_seq":4}"#)
                        .await
                        .expect("close frame");
                    subscriber_write.write_all(b"\n").await.expect("newline");
                    subscriber_write.flush().await.expect("flush");
                }
            }
        });

        let (spectator_tx, mut spectator_rx) = mpsc::channel(4);
        let (_resync_tx, force_resync_rx) = mpsc::channel(1);
        let context = SpectatorContext {
            bind_addr: addr.to_string(),
            session_id: "session-a".to_string(),
            subscription_id: Arc::new(AtomicU64::new(0)),
            spectator_tx,
            own_turn: Arc::new(AtomicBool::new(false)),
            watermark: Arc::new(AtomicU64::new(0)),
            auth_token: None,
            force_resync_rx,
        };
        let supervisor = tokio::spawn(spectator_supervisor(context));

        for _ in 0..2 {
            let event = tokio::time::timeout(Duration::from_secs(1), spectator_rx.recv())
                .await
                .expect("legacy replacement arrives")
                .expect("App ingress stays open");
            let SpectatorEvent::Replace { ack, .. } = event else {
                panic!("expected legacy replacement");
            };
            ack.send(AckVerdict::Applied).expect("ack replacement");
        }
        send_live_tx.send(()).expect("send live frame");

        let event = tokio::time::timeout(Duration::from_secs(1), spectator_rx.recv())
            .await
            .expect("fresh legacy socket consumes live frame")
            .expect("live spectator frame");
        assert!(matches!(
            event,
            SpectatorEvent::Frame {
                frame_seq: 3,
                block: RenderBlock::TextDelta { ref text, .. },
            } if text == "live"
        ));
        assert_eq!(subscribe_count.load(Ordering::SeqCst), 2);

        drop(spectator_rx);
        shutdown_tx.send(()).expect("shutdown server");
        tokio::time::timeout(Duration::from_secs(1), supervisor)
            .await
            .expect("supervisor exits after App ingress closes")
            .expect("supervisor task");
        server.await.expect("legacy server");
    }

    #[tokio::test]
    async fn legacy_load_failure_reconnects_instead_of_closing_app_ingress() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("address");
        let server = tokio::spawn(async move {
            let (subscriber, _) = listener.accept().await.expect("subscribe connection");
            let (subscriber_read, mut subscriber_write) = subscriber.into_split();
            let mut subscriber_lines = BufReader::new(subscriber_read).lines();
            let subscribe: JsonValue = serde_json::from_str(
                &subscriber_lines
                    .next_line()
                    .await
                    .expect("subscribe read")
                    .expect("subscribe line"),
            )
            .expect("subscribe JSON");
            assert_eq!(subscribe["method"], "session.subscribe");
            subscriber_write
                .write_all(
                    br#"{"jsonrpc":"2.0","id":1,"result":{"id":"session-a","history":[],"next_seq":3,"helm":null}}"#,
                )
                .await
                .expect("legacy subscribe response");
            subscriber_write.write_all(b"\n").await.expect("newline");
            subscriber_write.flush().await.expect("flush");

            // Accept the legacy loader then close it before responding.
            let (loader, _) = listener.accept().await.expect("load connection");
            drop(loader);
        });

        let (spectator_tx, mut spectator_rx) = mpsc::channel(1);
        let (_resync_tx, force_resync_rx) = mpsc::channel(1);
        let mut context = SpectatorContext {
            bind_addr: addr.to_string(),
            session_id: "session-a".to_string(),
            subscription_id: Arc::new(AtomicU64::new(0)),
            spectator_tx,
            own_turn: Arc::new(AtomicBool::new(false)),
            watermark: Arc::new(AtomicU64::new(0)),
            auth_token: None,
            force_resync_rx,
        };

        assert_eq!(
            spectator_connection(&mut context, false, LegacyHydration::Initial).await,
            SpectatorConnectionResult::ReconnectAfterFailure,
        );
        assert!(spectator_rx.try_recv().is_err(), "no replacement reaches the App");
        server.await.expect("legacy server");
    }

    #[test]
    fn subscribe_response_requires_exact_id_and_complete_typed_schema() {
        let valid = subscribe_response(7, &json!({
            "id": "session-a", "history": [], "next_seq": 4,
            "subscription_id": 11, "helm": null,
        }));
        let reply = parse_subscribe_response(&valid, 7, "session-a").expect("valid response");
        assert_eq!(reply.next_seq, 4);
        assert_eq!(reply.subscription_id, Some(11));
        assert!(parse_subscribe_response(&valid, 8, "session-a").is_none());
        let wrong_version = json!({
            "jsonrpc": "1.0", "id": 7, "result": valid["result"].clone()
        });
        assert!(parse_subscribe_response(&wrong_version, 7, "session-a").is_none());
        assert!(parse_subscribe_response(&subscribe_response(7, &json!({
            "id": "other", "history": [], "next_seq": 4,
            "subscription_id": 11, "helm": null,
        })), 7, "session-a").is_none());
        let legacy = parse_subscribe_response(&subscribe_response(7, &json!({
            "id": "session-a", "history": [], "next_seq": 4, "helm": null,
        })), 7, "session-a").expect("v1 response is accepted");
        assert_eq!(legacy.subscription_id, None);
        assert!(parse_subscribe_response(&subscribe_response(7, &json!({
            "id": "session-a", "history": [], "next_seq": 4,
            "subscription_id": "malformed", "helm": null,
        })), 7, "session-a").is_none());
        assert!(parse_subscribe_response(&subscribe_response(7, &json!({
            "id": "session-a", "history": [], "next_seq": 4,
            "subscription_id": 0, "helm": null,
        })), 7, "session-a").is_none());
    }

    #[test]
    fn pre_ack_cap_errors_without_dropping_retained_frames() {
        let mut queue = VecDeque::new();
        for seq in 0..MAX_DRAIN_PER_TICK {
            assert!(push_pre_ack(&mut queue, json!({ "frame_seq": seq })));
        }
        assert!(!push_pre_ack(&mut queue, json!({ "frame_seq": MAX_DRAIN_PER_TICK })));
        assert_eq!(queue.len(), MAX_DRAIN_PER_TICK);
        assert_eq!(queue.front().unwrap()["frame_seq"], 0);
        assert_eq!(queue.back().unwrap()["frame_seq"], MAX_DRAIN_PER_TICK - 1);
    }

    #[test]
    fn watermark_drops_delayed_own_turn_frame_after_gate_release() {
        // Simulates the reader observing own_turn=false only after the primary
        // has Release-stored watermark=42.
        assert!(!should_forward_spectator_frame(false, 41, 0, 42));
        assert!(should_forward_spectator_frame(false, 42, 0, 42));
        assert!(!should_forward_spectator_frame(true, 99, 0, 42));
    }

    #[tokio::test]
    async fn delayed_frame_is_rechecked_after_ingress_capacity_is_reserved() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.send(SpectatorEvent::Frame {
            frame_seq: 0,
            block: RenderBlock::System {
                id: BlockIdGen::default().next(),
                level: SystemLevel::Info,
                text: "queued first".to_string(),
            },
        })
        .await
        .expect("fill ingress");
        let own_turn = Arc::new(AtomicBool::new(false));
        let watermark = Arc::new(AtomicU64::new(0));
        let task = tokio::spawn({
            let tx = tx.clone();
            let own_turn = Arc::clone(&own_turn);
            let watermark = Arc::clone(&watermark);
            async move {
                enqueue_spectator_frame(
                    &tx,
                    &own_turn,
                    &watermark,
                    1,
                    0,
                    RenderBlock::System {
                        id: BlockIdGen::default().next(),
                        level: SystemLevel::Info,
                        text: "stale frame".to_string(),
                    },
                )
                .await
            }
        });
        tokio::task::yield_now().await;
        own_turn.store(true, Ordering::Release);
        watermark.store(2, Ordering::Release);
        own_turn.store(false, Ordering::Release);
        let _ = rx.recv().await.expect("release capacity");
        assert!(task.await.expect("task"));
        assert!(rx.try_recv().is_err(), "stale frame was not enqueued after the turn");
    }

    #[test]
    fn reconnect_backoff_resets_only_after_authoritative_replace_ack() {
        let mut delay = INITIAL_SPECTATOR_RECONNECT_DELAY;

        // Failures before the initial Replace ACK keep exponentiating.
        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::ReconnectAfterFailure),
            Some(Duration::from_secs(1))
        );
        assert_eq!(delay, Duration::from_secs(2));
        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::ReconnectAfterFailure),
            Some(Duration::from_secs(2))
        );
        assert_eq!(delay, Duration::from_secs(4));

        // This outcome is reachable only after connect + subscribe + Replace ACK.
        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::ReconnectAfterReady),
            Some(Duration::from_secs(1))
        );
        assert_eq!(delay, Duration::from_secs(2));

        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::ReconnectAfterAckedLegacyLoad),
            Some(Duration::ZERO),
            "legacy hydration reconnects immediately without resetting backoff"
        );
        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::ReconnectAfterStale),
            Some(Duration::from_secs(2)),
            "a stale verdict waits before acquiring a fresh boundary lock"
        );
        assert_eq!(delay, Duration::from_secs(2), "stale is not an authoritative ACK");

        // A later failure starts growing from the retained baseline.
        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::ReconnectAfterFailure),
            Some(Duration::from_secs(2))
        );
        assert_eq!(delay, Duration::from_secs(4));

        // Closed App ingress exits the supervisor instead of sleeping/retrying.
        assert_eq!(
            reconnect_delay(&mut delay, SpectatorConnectionResult::AppClosed),
            None
        );
        assert_eq!(delay, Duration::from_secs(4));
    }
    #[test]
    fn force_resync_channel_coalesces_signal_storms() {
        let (tx, mut rx) = mpsc::channel::<()>(1);
        assert!(tx.try_send(()).is_ok());
        assert!(tx.try_send(()).is_err());
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_err());
    }
}
