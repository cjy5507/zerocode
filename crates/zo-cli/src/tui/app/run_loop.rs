use std::time::Instant;

use crossterm::event::{Event, MouseEventKind};
use futures_util::{Stream, StreamExt};
use ratatui::Terminal;
use ratatui::backend::Backend;

use crate::tui::render_schedule::{
    ANIMATION_TICK_INTERVAL, STREAM_FRAME_INTERVAL, StreamFrameGate,
};

use super::{App, AppAction, TuiError};

pub(super) fn track_cooling_boundary(
    was_cooling: &mut bool,
    cooling_active: bool,
    dirty: &mut bool,
) {
    if *was_cooling && !cooling_active {
        *dirty = true;
    }
    *was_cooling = cooling_active;
}

impl App {
    /// Run the main event loop with a temporary terminal event stream.
    ///
    /// Use [`Self::run_with_events`] when an outer future may cancel the input
    /// wait and the terminal reader must survive that cancellation.
    pub async fn run<B: Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<AppAction, TuiError>
    where
        B::Error: std::fmt::Display,
    {
        let mut events = crossterm::event::EventStream::new();
        self.run_with_events(terminal, &mut events).await
    }

    /// Main event loop. Consumes terminal events from the caller-owned stream
    /// and interleaves them with inbound render blocks via `tokio::select!`.
    /// Keeping the stream at session scope preserves the input reader when an
    /// outer wait cancels this future.
    ///
    /// # Errors
    ///
    /// Returns [`TuiError::Io`] if the terminal backend fails, or
    /// [`TuiError::CommandChannelClosed`] if the agent task hung up
    /// before the user could acknowledge.
    #[allow(clippy::too_many_lines)] // the TUI main event loop: select! arms must share one scope
    pub async fn run_with_events<B: Backend, S>(
        &mut self,
        terminal: &mut Terminal<B>,
        events: &mut S,
    ) -> Result<AppAction, TuiError>
    where
        B::Error: std::fmt::Display,
        S: Stream<Item = std::io::Result<Event>> + Unpin,
    {
        use tokio::time;
        // 30 fps tick grid (33 ms), matching the mid-turn loop so spinner and
        // stream-driven redraws land on the same cadence instead of beating
        // against a slower 20 fps idle timer (the "one beat late" feel).
        let mut interval = time::interval(ANIMATION_TICK_INTERVAL);
        // Skip catch-up bursts if a draw or blocking call stalls the loop.
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        // Tracks whether state changed since the last draw. Input / paste /
        // mouse / resize events draw immediately in their own arms; streamed
        // blocks only mutate state and defer the redraw to the next tick, so
        // they flip this flag. At idle (no active turn, nothing changed) the
        // tick arm draws nothing — collapsing the old idle-CPU floor
        // (full widget-tree rebuild + HUD reformat) to zero.
        let mut dirty = false;
        // Shared frame gate: state lands immediately, terminal redraws are
        // coalesced to frame cadence and any skipped stream frame is recovered by
        // the shared animation tick.
        let mut frame_gate = StreamFrameGate::new_ready(Instant::now(), STREAM_FRAME_INTERVAL);
        let mut cooling_active = self.cooling_active_at(Instant::now());

        self.draw(terminal)?;

        let mut workflow_view_snapshot = None;
        let mut agents_rows_snapshot: Option<
            tokio::task::JoinHandle<crate::tui::workflow_progress::AgentRowsSnapshot>,
        > = None;
        let mut workspace_status_snapshot: Option<
            tokio::task::JoinHandle<Option<crate::tui::sidebar::GitStatusSnapshot>>,
        > = None;

        crate::tui::watchdog::set_phase(crate::tui::watchdog::Phase::Idle);
        while !self.should_quit {
            // Liveness heartbeat: the freeze watchdog reads this to tell a real
            // zo-side hang (no beat) from a terminal-side freeze (beats keep
            // coming while the screen is stuck). One relaxed add per frame.
            crate::tui::watchdog::beat();
            tokio::select! {
                // Tick-driven redraws cap streaming updates at 30 fps
                // while keeping spinner animations smooth.
                _ = interval.tick() => {
                    // Land a finished background `@` file scan, if any, and
                    // force a redraw so the picker swaps "scanning…" for the
                    // real list the moment results arrive.
                    if self.poll_file_scan() {
                        dirty = true;
                    }
                    // Same for the inline `@`-mention scan: swap the empty
                    // suggestion list for the real one when the walk lands.
                    if self.poll_workspace_scan() {
                        dirty = true;
                    }
                    if self.refresh_scheduled_wakeup() {
                        dirty = true;
                    }
                    // Live MCP source rows: background discovery finishing
                    // while the prompt sits idle must flip discovering→ready
                    // (or failed/auth-pending) without waiting for the next
                    // user action — the HUD snapshot itself is only rebuilt
                    // at action boundaries.
                    if self.refresh_mcp_status(Instant::now()) {
                        dirty = true;
                    }
                    if workspace_status_snapshot
                        .as_ref()
                        .is_some_and(tokio::task::JoinHandle::is_finished)
                    {
                        if let Some(handle) = workspace_status_snapshot.take() {
                            if let Ok(Some(snapshot)) = handle.await {
                                self.set_changed_files(snapshot.files, snapshot.total);
                                dirty = true;
                            }
                        }
                    }
                    if workspace_status_snapshot.is_none() {
                        workspace_status_snapshot = self.spawn_workspace_status_snapshot();
                    }
                    if workflow_view_snapshot.as_ref().is_some_and(tokio::task::JoinHandle::is_finished) {
                        if let Some(handle) = workflow_view_snapshot.take() {
                            let view = handle.await.unwrap_or(None);
                            self.apply_workflow_viewer_snapshot(view);
                            dirty = true;
                        }
                    }
                    if self.workflow_viewer_refresh_due() && workflow_view_snapshot.is_none() {
                        if let Some((started_after, session_id)) = self.workflow_viewer_snapshot_scope() {
                            workflow_view_snapshot = Some(tokio::task::spawn_blocking(move || {
                                crate::tui::workflow_progress::read_view_refresh_since(
                                    started_after,
                                    session_id.as_deref(),
                                )
                            }));
                        }
                    }
                    if agents_rows_snapshot.as_ref().is_some_and(tokio::task::JoinHandle::is_finished) {
                        if let Some(handle) = agents_rows_snapshot.take() {
                            let snapshot = handle.await.unwrap_or_default();
                            self.apply_agents_viewer_snapshot(snapshot);
                            dirty = true;
                        }
                    }
                    if self.agents_viewer_refresh_due() && agents_rows_snapshot.is_none() {
                        if let Some((started_after, session_id, include_history)) =
                            self.agents_viewer_snapshot_scope()
                        {
                            agents_rows_snapshot = Some(tokio::task::spawn_blocking(move || {
                                crate::tui::workflow_progress::read_agent_rows_since(
                                    started_after,
                                    session_id.as_deref(),
                                    include_history,
                                )
                            }));
                        }
                    }
                    // Live workflow/agents viewers: advance their spinners every
                    // frame; disk-backed snapshots land through the background
                    // tasks above.
                    if self.tick_workflow_viewer() {
                        dirty = true;
                    }
                    if self.tick_agents_viewer() {
                        dirty = true;
                    }
                    let tick_now = Instant::now();
                    let tick_stream_work = self.turn_activity.is_some() || self.stream_pending();
                    let next_cooling_active = self.cooling_active_at(tick_now);
                    track_cooling_boundary(
                        &mut cooling_active,
                        next_cooling_active,
                        &mut dirty,
                    );
                    let tick_has_work = dirty
                        || tick_stream_work
                        || next_cooling_active
                        || self.startup_intro_active();
                    let decision = if tick_stream_work {
                        frame_gate.on_stream_tick(tick_now, tick_has_work)
                    } else {
                        frame_gate.on_tick(tick_now, tick_has_work)
                    };
                    if decision.draws_now() {
                        self.advance_tick();
                        self.draw(terminal)?;
                        if tick_stream_work {
                            frame_gate.note_stream_draw(Instant::now());
                        }
                        dirty = false;
                    }
                }
                // User input events draw immediately for responsiveness.
                maybe_event = events.next() => {
                    match maybe_event {
                        Some(Ok(Event::Key(key))) => {
                            match self.handle_key(key)? {
                                AppAction::Quit => return Ok(AppAction::Quit),
                                AppAction::Submit(text) => return Ok(AppAction::Submit(text)),
                                AppAction::SelectModel(model) => {
                                    return Ok(AppAction::SelectModel(model));
                                }
                                AppAction::ConnectApiKey { provider, api_key } => {
                                    return Ok(AppAction::ConnectApiKey { provider, api_key });
                                }
                                AppAction::ConnectCustomProvider(draft) => {
                                    return Ok(AppAction::ConnectCustomProvider(draft));
                                }
                                AppAction::SelectPermission(mode) => {
                                    return Ok(AppAction::SelectPermission(mode));
                                }
                                AppAction::ClipboardPaste => {
                                    return Ok(AppAction::ClipboardPaste);
                                }
                                AppAction::ClipboardCopy(target) => {
                                    return Ok(AppAction::ClipboardCopy(target));
                                }
                                AppAction::ClipboardCopyBlock(text) => {
                                    return Ok(AppAction::ClipboardCopyBlock(text));
                                }
                                AppAction::SelectSession(id) => {
                                    return Ok(AppAction::SelectSession(id));
                                }
                                AppAction::Editor => {
                                    return Ok(AppAction::Editor);
                                }
                                AppAction::RewindCheckpoint => {
                                    return Ok(AppAction::RewindCheckpoint);
                                }
                                AppAction::ConfirmRewind => {
                                    return Ok(AppAction::ConfirmRewind);
                                }
                                AppAction::OpenRewindViewer => {
                                    return Ok(AppAction::OpenRewindViewer);
                                }
                                AppAction::OpenWorkflowViewer => {
                                    return Ok(AppAction::OpenWorkflowViewer);
                                }
                                AppAction::OpenAgentInViewer(id) => {
                                    return Ok(AppAction::OpenAgentInViewer(id));
                                }
                                AppAction::RewindTo(index) => {
                                    return Ok(AppAction::RewindTo(index));
                                }
                                AppAction::AckTeamInboxUpdate(update_id) => {
                                    return Ok(AppAction::AckTeamInboxUpdate(update_id));
                                }
                                AppAction::IncludeTeamInboxUpdate(text) => {
                                    return Ok(AppAction::IncludeTeamInboxUpdate(text));
                                }
                                AppAction::RefreshTeamInboxViewer => {
                                    return Ok(AppAction::RefreshTeamInboxViewer);
                                }
                                AppAction::ToggleTool { name, enabled } => {
                                    return Ok(AppAction::ToggleTool { name, enabled });
                                }
                                AppAction::SaveSmartSettings(commit) => {
                                    return Ok(AppAction::SaveSmartSettings(commit));
                                }
                                AppAction::DeepTier(action) => {
                                    return Ok(AppAction::DeepTier(action));
                                }
                                AppAction::Redraw | AppAction::None => {}
                            }
                            // Coalesce keystroke repaints to the shared frame
                            // cadence, mirroring the scroll/stream arms. A full
                            // widget-tree redraw on every keystroke floods slower
                            // terminals (Apple Terminal.app) faster than they
                            // paint, which reads as input lag. The first keystroke
                            // still draws immediately (the gate starts ready); a
                            // burst within the frame interval defers to the next
                            // ≤33ms tick — imperceptible to the user but keeps the
                            // terminal fed at a rate it can actually paint.
                            if frame_gate.on_stream_update(Instant::now()).draws_now() {
                                self.draw(terminal)?;
                                dirty = false;
                            } else {
                                dirty = true;
                            }
                        }
                        Some(Ok(Event::Paste(text))) => {
                            self.handle_paste_owned(text);
                            // Same frame coalescing as keystrokes: a paste lands
                            // immediately, but a rapid paste/keystroke burst
                            // shares the frame budget instead of flooding the
                            // terminal with full repaints.
                            if frame_gate.on_stream_update(Instant::now()).draws_now() {
                                self.draw(terminal)?;
                                dirty = false;
                            } else {
                                dirty = true;
                            }
                        }
                        Some(Ok(Event::Mouse(mouse))) => {
                            // Only repaint for scroll events; ignore raw mouse
                            // move/click so a large transcript is not flooded
                            // with full-screen repaints on every motion event
                            // (the "scroll feels a beat behind on big context"
                            // symptom). Mirrors the mid-turn loop's gating.
                            let is_scroll = matches!(
                                mouse.kind,
                                MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
                            );
                            let action = self.handle_mouse(mouse)?;
                            if matches!(action, AppAction::OpenWorkflowViewer) {
                                return Ok(AppAction::OpenWorkflowViewer);
                            }
                            if let AppAction::OpenAgentInViewer(id) = action {
                                return Ok(AppAction::OpenAgentInViewer(id));
                            }
                            if matches!(action, AppAction::ClipboardCopyBlock(_)) {
                                return Ok(action);
                            }
                            if matches!(action, AppAction::Redraw) {
                                self.draw(terminal)?;
                                dirty = false;
                            } else if is_scroll {
                                // Coalesce wheel repaints the same way streamed
                                // blocks are throttled. A macOS trackpad's
                                // inertial scroll fires dozens–hundreds of
                                // ScrollUp/Down events per flick; drawing the
                                // full transcript on every one floods the loop
                                // and the events back up faster than they paint
                                // (the "wheel lags / UI tears on big context"
                                // symptom). The scroll offset is already
                                // accumulated in `handle_mouse`, so repaint at
                                // most once per the shared stream frame interval and
                                // let the tick arm land the final frame via
                                // `dirty`.
                                if frame_gate.on_stream_update(Instant::now()).draws_now() {
                                    self.draw(terminal)?;
                                    dirty = false;
                                } else {
                                    dirty = true;
                                }
                            }
                        }
                        Some(Ok(_)) => {
                            // Focus changes, etc. — defer to the tick to avoid
                            // event-flood repaints.
                            dirty = true;
                        }
                        Some(Err(e)) => return Err(TuiError::Io(e)),
                        None => break,
                    }
                }
                // Streaming blocks update state and repaint at most once per
                // shared stream frame; the tick paints any frame this
                // skipped, so the newest content is never more than ~33 ms late.
                maybe_block = self.rx.recv() => {
                    match maybe_block {
                        Some(block) => {
                            self.push_block(block);
                            self.drain_ready_blocks();
                            // `push_block` lands streamed text in the transcript
                            // at arrival speed, so repaint on the block-driven
                            // path too — throttled by the shared frame gate
                            // so a fast burst can't redraw faster than the frame
                            // grid; otherwise defer to the next tick (`dirty`).
                            if frame_gate.on_stream_update(Instant::now()).draws_now() {
                                self.draw(terminal)?;
                                frame_gate.note_stream_draw(Instant::now());
                                dirty = false;
                            } else {
                                dirty = true;
                            }
                        }
                        None => {
                            self.should_quit = true;
                        }
                    }
                }
                // One FIFO spectator ingress: a Replace clears/replays and
                // ACKs before a later Frame may be processed.
                Some(event) = self.spectator_rx.recv() => {
                    self.process_spectator_event(event);
                    dirty = true;
                }
            }
        }

        Ok(AppAction::Quit)
    }
}
