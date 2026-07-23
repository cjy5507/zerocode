use ratatui::Terminal;
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph};
use unicode_width::UnicodeWidthStr;

use crate::tui::{blocks, glyphs, hud, inline, layout, sidebar, spinner, startup};
use crate::tui::blocks::tool_call::AgentRowSpan;
use crate::tui::modals::ModalPlacement;

use super::{
    App, AppMode, LayoutRegions, TuiError, anchored_modal_rect, centered_modal_rect,
    diff_modal_rect,
    agent_status_is_terminal, draw_effort_rule_badge, draw_mention_hint, draw_slash_hint,
    effort_modal_rect, effort_rule_badge, palette_modal_rect,
};

/// `true` when `ZO_PERF_DEBUG` is set — gates the per-frame draw timing below.
/// Read once and cached, so the hot draw path pays nothing when it is off.
fn perf_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ZO_PERF_DEBUG").is_some())
}

#[derive(Default)]
struct DrawPerf {
    window_frames: u64,
    sum_ms: f64,
    sum_transcript_ms: f64,
    sum_sidebar_ms: f64,
    max_ms: f64,
    slow: u64,
}

/// Record one frame's draw cost (only when `ZO_PERF_DEBUG` is set). Appends a
/// line for any frame over 16 ms (a dropped 30 fps tick — what reads as the
/// "화면 버벅") and a rolling 60-frame summary, to `/tmp/zo-perf.log`. The
/// `streaming` flag marks frames drawn during a live turn, so a slow frame can be
/// attributed to in-turn render cost vs. idle.
///
/// `transcript_ms` / `sidebar_ms` split out the two regions whose cost scales
/// with content so a slow frame is attributable: `chrome` (the remainder) covers
/// the HUD, input, rules, and overlays. This is what tells "transcript draw is the
/// bottleneck" (scales with on-screen blocks / tool output) apart from a slow
/// sidebar or chrome.
fn record_draw_frame(
    ms: f64,
    transcript_ms: f64,
    sidebar_ms: f64,
    blocks: usize,
    streaming: bool,
) {
    use std::fmt::Write as _;
    static ACC: std::sync::OnceLock<std::sync::Mutex<DrawPerf>> = std::sync::OnceLock::new();
    let Ok(mut a) = ACC
        .get_or_init(|| std::sync::Mutex::new(DrawPerf::default()))
        .lock()
    else {
        return;
    };
    a.window_frames += 1;
    a.sum_ms += ms;
    a.sum_transcript_ms += transcript_ms;
    a.sum_sidebar_ms += sidebar_ms;
    a.max_ms = a.max_ms.max(ms);
    if ms > 8.0 {
        a.slow += 1;
    }
    let chrome_ms = (ms - transcript_ms - sidebar_ms).max(0.0);
    let mut out = String::new();
    if ms > 16.0 {
        let _ = writeln!(
            out,
            "[PERF] SLOW frame draw={ms:.1}ms (transcript={transcript_ms:.1} sidebar={sidebar_ms:.1} chrome={chrome_ms:.1}) blocks={blocks} streaming={streaming}"
        );
    }
    if a.window_frames >= 60 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "window frame count is tiny, far below f64's exact-integer range"
        )]
        let frames = a.window_frames as f64;
        let avg = a.sum_ms / frames;
        let t_avg = a.sum_transcript_ms / frames;
        let s_avg = a.sum_sidebar_ms / frames;
        let _ = writeln!(
            out,
            "[PERF] summary: {} frames · draw avg={:.1}ms (transcript avg={:.1} sidebar avg={:.1}) max={:.1}ms · slow(>8ms)={}/{}",
            a.window_frames, avg, t_avg, s_avg, a.max_ms, a.slow, a.window_frames
        );
        *a = DrawPerf::default();
    }
    if !out.is_empty() {
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/zo-perf.log")
        {
            let _ = f.write_all(out.as_bytes());
        }
    }
}

impl App {
    /// Draw one frame, bracketing it in CSI ?2026 synchronized output when the
    /// host terminal is known to handle it (see [`crate::tui::term::TermProfile`]).
    ///
    /// Both `Begin` and `End` go through the backend's own writer (fd 1), never
    /// a second `io::stdout()` handle — that split is what drove xterm.js into
    /// the stuck-synchronized freeze that got the original 2026 wrap reverted,
    /// which is why [`crate::tui::term::TermProfile`] gates this to native
    /// terminals that batch everything between `Begin`/`End` regardless of write
    /// boundaries. `Begin` is flushed *immediately* (not left buffered) so a
    /// panic inside `draw` — before ratatui's own end-of-frame flush — cannot
    /// leave an unflushed `Begin` to be drop-flushed *after* the teardown path's
    /// defensive `End`, which would strand the terminal in synchronized mode.
    /// Best-effort: a failed `Begin`/`End` escape never aborts the frame, and
    /// both `restore_terminal` and `emergency_restore` emit a defensive `End`,
    /// so an error or panic mid-frame still resets synchronized mode.
    pub fn draw_frame<W: std::io::Write>(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<W>>,
    ) -> Result<(), TuiError> {
        use crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
        if self.synchronized_output {
            let _ = crossterm::execute!(terminal.backend_mut(), BeginSynchronizedUpdate);
        }
        let result = self.draw(terminal);
        if self.synchronized_output {
            let _ = crossterm::execute!(terminal.backend_mut(), EndSynchronizedUpdate);
        }
        result
    }

    /// Draw one frame into `terminal`.
    #[allow(clippy::too_many_lines)] // cohesive full-frame TUI render (layout + all panels)
    pub fn draw<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<(), TuiError>
    where
        B::Error: std::fmt::Display,
    {
        if self.terminal_mode.is_inline() {
            // Turn boundaries seal streamed output in `end_turn`; this idle
            // seal also covers immediately-final slash/system output and the
            // submitted user prompt before the host starts the next turn.
            if self.turn_activity.is_none() {
                self.finalize_inline_transcript();
            }
            self.flush_finalized_inline_transcript(terminal)?;
        }
        // Streaming turns never run the loop-top `sync_app_context` HUD rebuild,
        // so a redeploy landing mid-turn stayed invisible for the whole turn —
        // exactly the hour-long grinding turns that most need the /restart cue.
        // The frame path is the one surface guaranteed to run while a turn
        // streams; `check` is throttled to one `stat` per 5s and latches, so
        // per-frame is its designed cadence.
        self.hud_state.stale_binary = crate::tui::stale_binary::check();
        let heat_state = self.heat_state();
        let perf_t0 = perf_debug_enabled().then(std::time::Instant::now);
        // Per-region draw accounting (only meaningful when perf is on). The
        // closure accumulates into these so the post-draw log can attribute a
        // slow frame to the transcript vs. the sidebar vs. chrome.
        let perf_on = perf_t0.is_some();
        let mut transcript_ms = 0.0f64;
        let mut sidebar_ms = 0.0f64;
        let mut rendered_regions: Option<LayoutRegions> = None;
        let mut rendered_agent_panel: Option<Rect> = None;
        let mut rendered_agent_rows: Vec<(Rect, String)> = Vec::new();
        // The rect the transcript body is drawn into this frame (bottom
        // reservations + banner offset applied) — committed to
        // `self.transcript_draw_rect` after the draw so every interaction
        // path clamps against the same viewport the paint used.
        let mut rendered_transcript_rect: Option<Rect> = None;
        let sidebar_visible = !self.terminal_mode.is_inline() && self.sidebar.visible;
        // NOTE: this method never emits CSI ?2026 itself. The synchronized-output
        // bracket lives one level up in `draw_frame`, which queues Begin/End
        // through the *backend* writer (one coalesced frame) and only on
        // terminals `TermProfile` confirms handle 2026 — the earlier blanket
        // wrap here was reverted because emitting begin/end as separate
        // `io::stdout()` writes froze xterm.js mid-stream. The `BufWriter`
        // backend in `init_terminal` already coalesces each frame's cell-diff
        // into a single flush, so this path stays atomic-enough on its own.
        terminal
            .draw(|frame| {
                let area = frame.area();
                // A genuinely unusable terminal renders exactly one centered
                // ASCII notice instead of partial chrome. Wide-but-short frames
                // keep using the compact layout because its one-row HUD remains
                // useful; only a TooSmall frame without spare width takes this
                // bypass. Inline mode keeps its own compact degrade path.
                if !self.terminal_mode.is_inline()
                    && layout::ViewportClass::classify(area, &self.theme).requires_notice(area)
                {
                    Self::draw_too_small_notice(frame, area, &self.theme);
                    rendered_regions = Some(LayoutRegions {
                        sidebar: Rect::new(0, 0, 0, 0),
                        sidebar_width: 0,
                        transcript: Rect::new(area.x, area.y, area.width, 0),
                        rule_top: Rect::new(area.x, area.y, area.width, 0),
                        input: Rect::new(area.x, area.y, area.width, 0),
                        rule_bot: Rect::new(area.x, area.y, area.width, 0),
                        hud: Rect::new(area.x, area.y, area.width, 0),
                    });
                    return;
                }
                // One responsive policy for the whole frame: the sidebar shows
                // only when the user requested it AND the viewport is `Wide`
                // (theme breakpoint), so sidebar visibility and viewport spacing
                // no longer follow unrelated rules. `sidebar_visible` above is
                // the request; the resolved decision every geometry call below
                // reads comes from the one `LayoutPlan`.
                let sidebar_visible =
                    layout::LayoutPlan::compute(area, &self.theme, sidebar_visible).sidebar_visible;
                // Soft-wrap the input against its *real* inner width (content
                // column minus the 1-cell border on each side) so the box grows
                // to the right number of rows before the vertical layout runs.
                let input_inner_w = LayoutRegions::content_width(area.width, sidebar_visible)
                    .saturating_sub(2);
                let desired_input_rows = self.input.desired_rows(input_inner_w, self.mode);
                // App-level blanket Clear stays unnecessary here: widgets
                // that can leave stale cells clear their own local regions,
                // while ratatui still diffs the final frame buffer before
                // writing to the backend.
                // Grant the HUD a dedicated second row for the workflow phase
                // whenever a workflow is active and no other on-screen surface
                // already owns it. The sidebar's workflow section releases the
                // row only when it actually lands on screen (it sits below the
                // session block in a top-anchored unscrollable body, so a short
                // terminal renders the sidebar yet clips the phase).
                // `area.height >= 10` keeps the grant out of heavy vertical
                // degradation territory; below it the single status row carries
                // a truncated badge instead.
                //
                // The phase deliberately does NOT ride the live activity row
                // (rule_top spinner) even though the chat column now fills the
                // full width and is wide enough to host the inline badge: the
                // dedicated row is a stable home that does not appear/disappear
                // with the turn (no per-turn height flap), and it leaves the
                // activity row's budget for the action phrase, model, and
                // context pressure. The activity context suppresses the inline
                // workflow badge to match (see the spinner draw below), so the
                // phase is shown in exactly one place.
                let sidebar_width = layout::resolved_sidebar_width(area.width, sidebar_visible);
                let sidebar_carries_phase = sidebar_width > 0
                    && sidebar::workflow_section_on_screen(
                        Rect::new(0, 0, sidebar_width, area.height),
                        &self.hud_state,
                        &self.theme,
                    );
                let hud_rows = if self.hud_state.workflow.is_some()
                    && !sidebar_carries_phase
                    && area.height >= 10
                {
                    2
                } else {
                    layout::HUD_ROWS
                };
                let regions = LayoutRegions::compute_with_sidebar(
                    area,
                    desired_input_rows,
                    hud_rows,
                    sidebar_visible,
                )
                .unwrap_or_else(|_| LayoutRegions {
                    sidebar: Rect::new(0, 0, 0, 0),
                    sidebar_width: 0,
                    transcript: Rect::new(0, 0, area.width, 0),
                    rule_top: Rect::new(0, 0, area.width, 0),
                    input: Rect::new(0, 0, area.width, 0),
                    rule_bot: Rect::new(0, 0, area.width, 0),
                    hud: Rect::new(0, 0, area.width, area.height.min(1)),
                });
                rendered_regions = Some(regions);

                let rule_style = Style::default().fg(self.theme.palette.faint);
                let rule_char = if self.theme.no_color {
                    glyphs::HORIZONTAL_RULE_NC
                } else {
                    glyphs::HORIZONTAL_RULE
                };

                // Render the sidebar when visible. Draw it down to the very
                // bottom of the terminal (not just the transcript row) so the
                // ledger panel reads as one full-height column: the chat chrome
                // (rules / input / HUD) is narrowed to the content column, so
                // the right `sidebar_width` strip below the transcript would
                // otherwise be an empty corner. Extending the panel fills that
                // corner and bottom-aligns the keybinding footer with the HUD.
                if regions.sidebar_width > 0 {
                    let full_height_sidebar = Rect::new(
                        regions.sidebar.x,
                        regions.sidebar.y,
                        regions.sidebar.width,
                        area.height.saturating_sub(regions.sidebar.y),
                    );
                    let st = perf_on.then(std::time::Instant::now);
                    sidebar::draw(
                        frame,
                        full_height_sidebar,
                        &self.sidebar,
                        &self.hud_state,
                        &self.theme,
                    );
                    if let Some(st) = st {
                        sidebar_ms += st.elapsed().as_secs_f64() * 1000.0;
                    }
                }

                // The transcript uses its full region now; the active-effort
                // badge moved down to the input line (drawn after the input
                // widget below) instead of reserving the transcript's top row.
                let mut transcript_area = regions.transcript;
                // Bottom overlays (search bar, queue preview, Run Dock/todo,
                // agent panel) each want rows above the input. Rather than
                // letting each independently eat transcript height until the
                // conversation reaches zero, one immutable `layout::OverlayPlan`
                // allocates the whole stack: it measures each overlay's demand,
                // then hands out a budget that always leaves the transcript at
                // least `MIN_READABLE_TRANSCRIPT_ROWS`, collapsing in the
                // documented order — the queue yields first, then the Run Dock,
                // then the agent panel, while an active search row and the
                // current running step survive longest. The granted rows drive
                // *both* reservation and paint through `overlay_rect`, so the
                // measure and draw geometry can never disagree. Notice-only
                // frames never reach this path; wide-but-short frames do.
                let full_transcript = regions.transcript;
                let run_dock_executor = self.run_dock_executor_row();
                let search_demand = u16::from(self.mode == AppMode::Search);
                let queue_demand =
                    if !self.queued_messages.is_empty() && regions.rule_top.height > 0 {
                        self.queued_previews_height(full_transcript, search_demand)
                    } else {
                        0
                    };
                // The Run Dock geometry (used again for `run_dock_owns_agents`
                // and paint) is measured once against the full transcript.
                let run_dock_demand_geometry = self.todo_panel_geometry(
                    full_transcript,
                    search_demand + queue_demand,
                    run_dock_executor.is_some(),
                );
                let run_dock_demand =
                    run_dock_demand_geometry.map_or(0, |geometry| geometry.reserved_height);
                // When the dock actually has room to paint a workflow/agent
                // executor row, it owns the compact live-agent surface. If that
                // row is clipped by vertical degradation, keep the existing
                // pinned-panel path eligible instead of suppressing it blindly.
                let run_dock_owns_agents =
                    run_dock_demand_geometry.is_some_and(|geometry| {
                        geometry.shows_executor
                            && run_dock_executor
                                .as_ref()
                                .is_some_and(RunDockExecutorRow::is_inspectable)
                    });
                // Pinned live-agent tree stacks above the queue/todo overlays —
                // but inline-first (Claude-Code style): while the fan-out's own
                // Spawned row is on screen its inline tree IS the live view, so
                // the panel only appears when that row is scrolled away or no
                // host row exists (Workflow placeholder, unattributed spawns).
                // Visibility uses the previous frame's resolved scroll (the
                // reservation runs before this frame's transcript draw); a
                // one-frame lag on scroll is invisible at tick cadence.
                let (agent_panel_lines, agent_row_spans) = {
                    // Judge inline-tree visibility against the height the
                    // transcript was actually drawn at (previous frame — same
                    // one-frame-lag model as the scroll reservation above),
                    // not the full region the overlays shrink.
                    let drawn_h = self
                        .transcript_draw_rect
                        .map_or(full_transcript.height, |r| r.height);
                    if run_dock_owns_agents || self.transcript.live_tree_visible(drawn_h) {
                        (Vec::new(), Vec::new())
                    } else {
                        self.agent_panel_lines()
                    }
                };
                let agent_demand = Self::agent_panel_height(
                    &agent_panel_lines,
                    full_transcript,
                    search_demand + queue_demand + run_dock_demand,
                );
                // Clamp the four demands to the transcript-protecting budget in
                // collapse-priority order (queue first, then Run Dock, then
                // agent; search survives longest).
                let plan = layout::OverlayPlan::allocate(
                    layout::OverlayDemand {
                        search: search_demand,
                        queue: queue_demand,
                        run_dock: run_dock_demand,
                        agent: agent_demand,
                    },
                    full_transcript.height,
                );
                // `overlay_rect(reserved_below, grant)`: the bottom slice of the
                // transcript hosting one overlay, its height capped to the rows
                // already reserved below it plus this overlay's granted rows, so
                // a helper measuring/painting against it bottom-anchors just
                // above the input and can never exceed its grant.
                let overlay_rect = |reserved_below: u16, grant: u16| -> Rect {
                    let mut a = full_transcript;
                    let want = reserved_below.saturating_add(grant).min(a.height);
                    a.y += a.height - want;
                    a.height = want;
                    a
                };
                let mut bottom_reserved = plan.search;
                let queue_area = overlay_rect(bottom_reserved, plan.queue);
                if plan.queue > 0 {
                    bottom_reserved += self.queued_previews_height(queue_area, bottom_reserved);
                }
                // Re-measure the Run Dock against its granted slice so the paint
                // geometry matches the reservation exactly.
                let run_dock_area = overlay_rect(bottom_reserved, plan.run_dock);
                let run_dock_geometry = (plan.run_dock > 0)
                    .then(|| {
                        self.todo_panel_geometry(
                            run_dock_area,
                            bottom_reserved,
                            run_dock_executor.is_some(),
                        )
                    })
                    .flatten();
                let run_dock_reserved =
                    run_dock_geometry.map_or(0, |geometry| geometry.reserved_height);
                bottom_reserved += run_dock_reserved;
                let agent_area = overlay_rect(bottom_reserved, plan.agent);
                let agent_panel_reserved =
                    Self::agent_panel_height(&agent_panel_lines, agent_area, bottom_reserved);
                bottom_reserved += agent_panel_reserved;
                transcript_area.height = transcript_area.height.saturating_sub(bottom_reserved);

                if let Some(startup_screen) = self.startup.screen.as_ref() {
                    // Elapsed since the launchpad appeared. Static launchpads
                    // ignore this; animated variants can use it for a bounded
                    // intro. Reduce-motion renders the settled (`None`) frame
                    // from the first paint.
                    let intro = if crate::tui::term::reduce_motion_enabled() {
                        None
                    } else {
                        self.startup.intro_elapsed()
                    };
                    // Sticky banner behavior matching Claude Code CLI:
                    // the ZO launchpad stays at the top of the transcript
                    // region; newly-arriving messages render in the
                    // remaining space below. On the first frame (empty
                    // transcript) the banner still occupies the whole
                    // region so it looks like the standalone launchpad.
                    let banner_h = startup::preferred_height(transcript_area.width)
                        .min(transcript_area.height);
                    if self.transcript.is_empty() {
                        startup::draw(frame, transcript_area, startup_screen, &self.theme, intro);
                    } else {
                        let banner_rect = Rect::new(
                            transcript_area.x,
                            transcript_area.y,
                            transcript_area.width,
                            banner_h,
                        );
                        let below_h = transcript_area.height.saturating_sub(banner_h);
                        let transcript_rect = Rect::new(
                            transcript_area.x,
                            transcript_area.y + banner_h,
                            transcript_area.width,
                            below_h,
                        );
                        startup::draw(frame, banner_rect, startup_screen, &self.theme, intro);
                        if below_h > 0 {
                            rendered_transcript_rect = Some(transcript_rect);
                            let tt = perf_on.then(std::time::Instant::now);
                            self.transcript.draw_with_hover(
                                frame,
                                transcript_rect,
                                &self.theme,
                                self.tick,
                                self.image_protocol,
                                self.hovered_copy_block_id(),
                            );
                            if let Some(tt) = tt {
                                transcript_ms += tt.elapsed().as_secs_f64() * 1000.0;
                            }
                        }
                    }
                } else {
                    rendered_transcript_rect = Some(transcript_area);
                    let tt = perf_on.then(std::time::Instant::now);
                    self.transcript.draw_with_hover(
                        frame,
                        transcript_area,
                        &self.theme,
                        self.tick,
                        self.image_protocol,
                        self.hovered_copy_block_id(),
                    );
                    if let Some(tt) = tt {
                        transcript_ms += tt.elapsed().as_secs_f64() * 1000.0;
                    }
                }

                if regions.rule_top.height > 0 {
                    let indicator = Rect::new(
                        regions.rule_top.x,
                        regions.rule_top.y,
                        regions.rule_top.width,
                        1,
                    );
                    if let Some(activity) = self.turn_activity.as_ref() {
                        // The live activity row shows the model + context
                        // pressure, but deliberately NOT the workflow phase: the
                        // phase has one stable home on the HUD's dedicated second
                        // row (or the sidebar workflow section), granted for the
                        // whole workflow rather than only while a turn streams.
                        // Surfacing it here too would double-render it now that
                        // the chat column fills the full width — a wide activity
                        // row clears `WORKFLOW_BADGE_MIN_COLS` and would otherwise
                        // paint the same phase the dedicated HUD row already owns.
                        let mut context = spinner::ActivityContext::from_hud(&self.hud_state);
                        context.workflow = None;
                        spinner::draw_with_context(
                            frame,
                            indicator,
                            activity,
                            Some(&context),
                            &self.theme,
                        );
                    } else {
                        let effort_badge = effort_rule_badge(&self.hud_state, &self.theme)
                            .filter(|(_, width)| *width > 0 && *width < indicator.width);
                        let rule_width = effort_badge.as_ref().map_or(indicator.width, |(_, width)| {
                            indicator.width.saturating_sub(*width).saturating_sub(1)
                        });
                        if rule_width > 0 {
                            let rule_area = Rect::new(
                                indicator.x,
                                indicator.y,
                                rule_width,
                                indicator.height,
                            );
                            let width = usize::from(rule_width);
                            let mut spans = Vec::with_capacity(2);
                            if self.theme.no_color {
                                spans.push(Span::styled(rule_char.repeat(width), rule_style));
                            } else {
                                let ember_cells = width.min(3);
                                spans.push(Span::styled(
                                    glyphs::ANVIL_LINE.repeat(ember_cells),
                                    Style::new().fg(self.theme.palette.accent_dim),
                                ));
                                if width > ember_cells {
                                    spans.push(Span::styled(
                                        rule_char.repeat(width - ember_cells),
                                        rule_style,
                                    ));
                                }
                            }
                            let p = Paragraph::new(Line::from(spans));
                            frame.render_widget(p, rule_area);
                        }
                        if let Some((badge, badge_width)) = effort_badge {
                            // Keep one truly empty cell between the hairline and
                            // the right-aligned badge. The draw loop has no
                            // blanket Clear, so erase that separator explicitly.
                            let gap = Rect::new(
                                indicator.x + rule_width,
                                indicator.y,
                                1,
                                indicator.height,
                            );
                            frame.render_widget(Clear, gap);
                            draw_effort_rule_badge(frame, indicator, badge, badge_width);
                        }
                    }
                }

                // Always render the input widget so the user can type
                // while a turn is in progress (queued-message mode).
                self.input.draw_with_heat(
                    frame,
                    regions.input,
                    &self.theme,
                    &self.mode,
                    heat_state,
                );
                if self.input_enabled {
                    if self.slash_hint_active() {
                        let recent = self.command_history.top_recent(10);
                        draw_slash_hint(
                            frame,
                            regions.input,
                            self.terminal_mode.is_inline().then(|| frame.area()),
                            &self.input,
                            &self.prompt_commands,
                            &recent,
                            &self.theme,
                            self.mode,
                            self.hints.slash_cursor,
                        );
                    }
                    // `@`-mention popup (mutually exclusive with slash-hint:
                    // one keys off a leading `/`, the other off an `@token`).
                    if self.mention_hint_active() {
                        let suggestions = self.mention_hint_suggestions();
                        draw_mention_hint(
                            frame,
                            regions.input,
                            self.terminal_mode.is_inline().then(|| frame.area()),
                            &suggestions,
                            &self.theme,
                            self.hints.mention_cursor,
                        );
                    }
                }

                // Queue badge on the rule_top row, right-aligned.
                //
                // The overlay stack starts above the search bar when Search
                // mode owns the transcript's bottom row — the measure pass
                // reserves that row (see bottom_reserved above), so the draw
                // pass must stack from the same base or the queue previews
                // overwrite the search bar.
                // Mirror the reservation base: the granted search-bar row(s).
                let mut bottom_overlay_rows: u16 = plan.search;
                if !self.queued_messages.is_empty() && regions.rule_top.height > 0 {
                    let (badge, badge_width) =
                        queue_badge(self.queued_messages.len(), !self.theme.no_color);
                    if badge_width < regions.rule_top.width {
                        let badge_x =
                            regions.rule_top.x + regions.rule_top.width.saturating_sub(badge_width);
                        let gap_rect = Rect::new(
                            badge_x.saturating_sub(1),
                            regions.rule_top.y,
                            1,
                            1,
                        );
                        let badge_rect = Rect::new(badge_x, regions.rule_top.y, badge_width, 1);
                        let badge_widget = Paragraph::new(Line::from(Span::styled(
                            badge,
                            Style::default().fg(self.theme.palette.warn),
                        )));
                        frame.render_widget(Clear, gap_rect);
                        frame.render_widget(badge_widget, badge_rect);
                    }

                    // Claude Code CLI parity: preview the queued entries as dim
                    // lines just above the input rule so the user sees exactly
                    // what will run next (and in what order), not only a count.
                    bottom_overlay_rows +=
                        self.draw_queued_previews(frame, queue_area, bottom_overlay_rows);
                }

                // Live todo panel (Claude Code parity): while a turn streams,
                // pin the active checklist just above the input, stacked above
                // any queue preview so the two never overwrite each other. The
                // transcript also keeps each settled `Updated Plan` as a bordered
                // history block; this live panel is the current-turn checklist.
                let rendered_run_dock =
                    self.draw_todo_panel(frame, run_dock_geometry, run_dock_executor.as_ref());

                // Pinned live-agent tree above the input (CC parity): always
                // visible while a fan-out runs, stacked above the queue/todo
                // overlays so the running agents read as a moving tree rather
                // than a dead `Delegating · no output` spinner.
                let agent_reserved_below = bottom_overlay_rows + run_dock_reserved;
                if run_dock_owns_agents {
                    // Reuse the existing aggregate agent-panel click target: a
                    // click anywhere on the integrated Run Dock opens Ctrl+O's
                    // workflow/agents surface. Per-agent targets remain available
                    // in the inline tree and Ctrl+G viewer.
                    rendered_agent_panel = rendered_run_dock;
                } else {
                    let (rendered_panel, rendered_rows) = Self::draw_agent_panel(
                        frame,
                        &agent_panel_lines,
                        &agent_row_spans,
                        agent_area,
                        agent_reserved_below,
                    );
                    rendered_agent_panel = rendered_panel;
                    rendered_agent_rows = rendered_rows;
                }

                // `rule_bot` is zero-height by design: whitespace and the HUD
                // surface transition separate the composer without extra chrome.

                if regions.hud.height > 0 {
                    // Avoid top/bottom status duplication: the right ledger owns
                    // session metadata when visible, and the activity row owns
                    // the live turn context while a turn is running. In either
                    // case the bottom HUD should become a quiet input boundary.
                    let details_owned_elsewhere =
                        regions.sidebar_width > 0 || self.turn_activity.is_some();
                    hud::draw_with_heat(
                        frame,
                        regions.hud,
                        &self.hud_state,
                        &self.theme,
                        details_owned_elsewhere,
                        // The pinned agent panel already shows the fleet in
                        // detail — drop the redundant bottom-bar agents chip.
                        rendered_agent_panel.is_some(),
                        heat_state,
                    );
                }

                self.draw_modals(frame, &regions, area);
            })
            .map_err(|error| TuiError::Adapter {
                component: "terminal",
                message: error.to_string(),
            })?;
        self.regions = rendered_regions;
        self.transcript_draw_rect = rendered_transcript_rect;
        self.agent_panel_click_rect = rendered_agent_panel;
        // Reconcile hover against the rows actually painted this frame: if the
        // hovered agent is no longer on screen (finished, scrolled out of the
        // top-6), drop the hover so its underline does not linger on an
        // unrelated row. Cleared without a forced repaint — the next tick paints
        // the settled state.
        if let Some(hovered) = self.hovered_agent.as_deref() {
            if !rendered_agent_rows.iter().any(|(_, id)| id == hovered) {
                self.hovered_agent = None;
            }
        }
        self.agent_row_click_targets = rendered_agent_rows;
        if let Some(t0) = perf_t0 {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            record_draw_frame(
                ms,
                transcript_ms,
                sidebar_ms,
                self.transcript.blocks().len(),
                self.turn_activity.is_some(),
            );
        }
        Ok(())
    }

    /// Emit settled inline transcript chunks without composing a viewport frame.
    ///
    /// Normal draws call this before repainting the live viewport. Shutdown uses
    /// it directly because the next operation clears/restores that viewport; an
    /// extra frame after `insert_before` would unnecessarily cross from the
    /// insertion callback's zero-origin buffer into the viewport's absolute
    /// screen coordinates.
    pub fn flush_finalized_inline_transcript<B: Backend>(
        &mut self,
        terminal: &mut Terminal<B>,
    ) -> Result<(), TuiError>
    where
        B::Error: std::fmt::Display,
    {
        if !self.terminal_mode.is_inline() {
            return Ok(());
        }
        inline::insert_finalized(
            &mut self.finalized_transcript,
            terminal,
            &self.theme,
            self.tick,
            self.image_protocol,
        )
        .map_err(|error| TuiError::Adapter {
            component: "inline transcript",
            message: error.to_string(),
        })
    }

    /// Draw the active modal / overlay for the current [`AppMode`] over the base
    /// frame. Extracted from [`App::draw`] to keep the per-frame render readable;
    /// a pure function of `self` and the computed layout.
    #[allow(clippy::too_many_lines)] // one arm per modal mode; splitting would obscure the dispatch
    fn draw_modals(&self, frame: &mut ratatui::Frame, regions: &LayoutRegions, area: Rect) {
        if self.inline_surface_requires_fullscreen(area) {
            self.draw_inline_fullscreen_notice(frame, regions, area);
            return;
        }
        let modal_area = anchored_modal_rect(self, regions, area);
        // A modal installed on the unified slot renders itself; its placement
        // selects the same rect a legacy arm would use. Migrated modals return
        // here before the legacy per-mode match runs.
        if let Some(modal) = &self.active_modal {
            let rect = match modal.placement() {
                ModalPlacement::Anchored => modal_area,
                ModalPlacement::EffortBanner => effort_modal_rect(regions, area),
                ModalPlacement::Fullscreen => diff_modal_rect(regions, area),
                ModalPlacement::Palette => palette_modal_rect(area),
                ModalPlacement::Centered => {
                    centered_modal_rect(area, modal.desired_size(area, &self.theme))
                }
            };
            // The effort banner is a transient hint strip, not a focus-stealing
            // pane — it gets no frosted backdrop.
            if matches!(modal.placement(), ModalPlacement::EffortBanner) {
                frame.render_widget(Clear, rect);
            } else {
                frost_modal_backdrop(frame, rect, &self.theme);
            }
            modal.draw(frame, rect, &self.theme);
            if let Some(cursor) = modal.cursor(rect) {
                frame.set_cursor_position(cursor);
            }
            return;
        }
        match self.mode {
            AppMode::ModalChoice => {
                if let Some(prompt) = self.active_prompt() {
                    frost_modal_backdrop(frame, modal_area, &self.theme);
                    let selected = self.permission_selected();
                    if self.terminal_mode.is_inline() {
                        blocks::permission::draw_compact(
                            frame,
                            modal_area,
                            prompt,
                            &self.theme,
                            selected,
                        );
                    } else {
                        blocks::permission::draw(
                            frame,
                            modal_area,
                            prompt,
                            &self.theme,
                            true,
                            selected,
                            0,
                        );
                    }
                }
            }
            AppMode::ModalDiff => {
                if let Some(modal) = &self.modals.diff_viewer {
                    let diff_area = diff_modal_rect(regions, area);
                    frost_modal_backdrop(frame, diff_area, &self.theme);
                    modal.draw(frame, diff_area, &self.theme);
                }
            }
            AppMode::ModalRewind => {
                if let Some(modal) = &self.modals.rewind {
                    let rewind_area = diff_modal_rect(regions, area);
                    frost_modal_backdrop(frame, rewind_area, &self.theme);
                    modal.draw(frame, rewind_area, &self.theme);
                }
            }
            AppMode::ModalConfirmRewind => {
                if let Some(lines) = self.rewind_confirm_lines() {
                    frost_modal_backdrop(frame, modal_area, &self.theme);
                    blocks::confirm::draw(frame, modal_area, lines, &self.theme);
                }
            }
            AppMode::ModalWorkflow => {
                if let Some(modal) = &self.modals.workflow {
                    let workflow_area = diff_modal_rect(regions, area);
                    frost_modal_backdrop(frame, workflow_area, &self.theme);
                    modal.draw(frame, workflow_area, &self.theme);
                }
            }
            AppMode::ModalAgents => {
                if let Some(modal) = &self.modals.agents {
                    let agents_area = diff_modal_rect(regions, area);
                    frost_modal_backdrop(frame, agents_area, &self.theme);
                    modal.draw(frame, agents_area, &self.theme);
                }
            }
            AppMode::ModalTeamInbox => {
                if let Some(modal) = &self.modals.team_inbox {
                    let inbox_area = diff_modal_rect(regions, area);
                    frost_modal_backdrop(frame, inbox_area, &self.theme);
                    modal.draw(frame, inbox_area, &self.theme);
                }
            }
            AppMode::ModalUsage => {
                if let Some(modal) = &self.modals.usage_dashboard {
                    let usage_area = diff_modal_rect(regions, area);
                    frost_modal_backdrop(frame, usage_area, &self.theme);
                    modal.draw(frame, usage_area, &self.theme);
                }
            }
            AppMode::Search => self.draw_search_bar(frame, regions),
            AppMode::Pager => self.draw_pager(frame, area),
            // These modal modes are painted by the unified slot early-return
            // above; the branches are unreachable but keep the match exhaustive.
            AppMode::ModalModel
            | AppMode::ModalPermissions
            | AppMode::ModalReport
            | AppMode::ModalArgPick
            | AppMode::ModalSession
            | AppMode::ModalLogin
            | AppMode::ModalEffort
            | AppMode::ModalTools
            | AppMode::ModalHunks
            | AppMode::ModalSmartSettings
            | AppMode::ModalDeepTier
            | AppMode::ModalRemoteOnboarding
            | AppMode::ModalQuestion
            | AppMode::ModalApiKey
            | AppMode::ModalCustomProvider
            | AppMode::ModalFile
            | AppMode::Focus
            | AppMode::Normal => {}
        }
    }

    /// Whether the active surface is intentionally unavailable in the small
    /// inline viewport. Anchored pickers and blocking prompts stay usable;
    /// document-scale viewers and palette/fullscreen modals degrade to notice.
    fn inline_surface_requires_fullscreen(&self, area: Rect) -> bool {
        if !self.terminal_mode.is_inline() {
            return false;
        }
        if let Some(modal) = self.active_modal.as_ref() {
            match modal.placement() {
                ModalPlacement::Fullscreen
                | ModalPlacement::Palette
                | ModalPlacement::EffortBanner => return true,
                // Content-sized surfaces stay usable inline as long as they fit.
                ModalPlacement::Anchored | ModalPlacement::Centered => {
                    let (_, desired_height) = modal.desired_size(area, &self.theme);
                    let available_height = area.height.saturating_sub(4);
                    if desired_height > available_height {
                        return true;
                    }
                }
            }
        }
        matches!(
            self.mode,
            AppMode::ModalDiff
                | AppMode::ModalRewind
                | AppMode::ModalWorkflow
                | AppMode::ModalAgents
                | AppMode::ModalTeamInbox
                | AppMode::ModalUsage
                | AppMode::Pager
                | AppMode::Focus
        )
    }

    fn draw_inline_fullscreen_notice(
        &self,
        frame: &mut ratatui::Frame,
        regions: &LayoutRegions,
        area: Rect,
    ) {
        let host = if regions.transcript.height > 0 {
            regions.transcript
        } else {
            area
        };
        if host.height == 0 || host.width == 0 {
            return;
        }
        let notice = Rect::new(
            host.x,
            host.y + host.height.saturating_sub(1) / 2,
            host.width,
            1,
        );
        frame.render_widget(Clear, notice);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(" {} needs full-screen mode · Esc to close", self.mode),
                Style::new()
                    .fg(self.theme.palette.warn)
                    .add_modifier(Modifier::BOLD),
            ))),
            notice,
        );
    }

    /// Paint the [`layout::TOO_SMALL_MESSAGE`] centered on an otherwise cleared
    /// frame. Called only for a [`layout::ViewportClass::TooSmall`] terminal, so
    /// it deliberately owns the whole `area`: it clears everything first so no
    /// stale cells from a prior larger frame survive, then centers the exact
    /// ASCII notice (truncated only if the terminal is narrower than the
    /// message, which never happens for a classified-usable size).
    fn draw_too_small_notice(frame: &mut ratatui::Frame, area: Rect, theme: &crate::tui::theme::Theme) {
        frame.render_widget(Clear, area);
        if area.width == 0 || area.height == 0 {
            return;
        }
        let msg = layout::TOO_SMALL_MESSAGE;
        let msg_w = u16::try_from(UnicodeWidthStr::width(msg)).unwrap_or(u16::MAX);
        let x = area.x + area.width.saturating_sub(msg_w) / 2;
        let y = area.y + area.height / 2;
        let rect = Rect::new(x, y, area.width.saturating_sub(x - area.x), 1);
        let style = if theme.no_color {
            theme.typography.body
        } else {
            Style::new().fg(theme.palette.warn)
        };
        frame.render_widget(Paragraph::new(Line::from(Span::styled(msg, style))), rect);
    }

    /// Draw the transcript search bar pinned to the bottom row of the transcript.
    fn draw_search_bar(&self, frame: &mut ratatui::Frame, regions: &LayoutRegions) {
        if regions.transcript.height <= 1 {
            return;
        }
        let bar_y = regions.transcript.y + regions.transcript.height.saturating_sub(1);
        let bar_rect = Rect::new(regions.transcript.x, bar_y, regions.transcript.width, 1);
        frame.render_widget(Clear, bar_rect);
        let counter = if self.search.query.is_empty() {
            String::new()
        } else if self.search.matches.is_empty() {
            "  no matches".to_string()
        } else {
            format!(
                "  [{}/{}]  ↵/↓ next · ↑ prev · esc",
                self.search.active_match + 1,
                self.search.matches.len()
            )
        };
        let search_line = Line::from(vec![
            Span::styled(" Search: ", Style::new().fg(self.theme.palette.accent)),
            Span::styled(self.search.query.clone(), self.theme.typography.body),
            Span::styled("_", self.theme.typography.dim),
            Span::styled(counter, self.theme.typography.dim),
        ]);
        frame.render_widget(
            Paragraph::new(search_line).style(Style::new().bg(self.theme.palette.muted)),
            bar_rect,
        );
    }

    /// Draw the full-screen pager overlay for long tool output.
    fn draw_pager(&self, frame: &mut ratatui::Frame, area: Rect) {
        if self.pager_content.is_none() {
            return;
        }
        frame.render_widget(Clear, area);
        let header_rect = Rect::new(area.x, area.y, area.width, 1);
        let body_rect = Rect::new(
            area.x,
            area.y + 1,
            area.width,
            area.height.saturating_sub(2),
        );
        let footer_rect = Rect::new(
            area.x,
            area.y + area.height.saturating_sub(1),
            area.width,
            1,
        );

        let wrap_width = usize::from(body_rect.width.max(1));
        let total_visual_rows = self
            .pager_lines
            .iter()
            .map(|line| line.chars().count().max(1).div_ceil(wrap_width))
            .sum::<usize>();
        let total_lines = u16::try_from(total_visual_rows).unwrap_or(u16::MAX);
        let max_scroll = total_lines.saturating_sub(body_rect.height.min(total_lines));
        let scroll = self.pager_scroll.min(max_scroll);

        let header = Paragraph::new(Line::from(Span::styled(
            " Pager (q/Esc to close)",
            Style::new().fg(self.theme.palette.bright),
        )))
        .style(Style::new().bg(self.theme.palette.muted));
        frame.render_widget(header, header_rect);

        let body_lines: Vec<Line<'_>> = self
            .pager_lines
            .iter()
            .map(|line| Line::from(Span::styled(line.as_str(), self.theme.typography.body)))
            .collect();
        let body_para = Paragraph::new(body_lines)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .scroll((scroll, 0));
        frame.render_widget(body_para, body_rect);

        let footer = Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" Line {}/{}", scroll.saturating_add(1), total_lines),
                Style::new().fg(self.theme.palette.dim),
            ),
            Span::styled(
                " | Up/Down/PgUp/PgDn to scroll",
                Style::new().fg(self.theme.palette.dim),
            ),
        ]))
        .style(Style::new().bg(self.theme.palette.muted));
        frame.render_widget(footer, footer_rect);
    }

    /// Render a compact, dimmed preview of the queued messages anchored to the
    /// bottom of the transcript area (just above the input), so the user can
    /// see exactly what will run next and in what order — matching the Claude
    /// Code CLI experience. Drawn only when the queue is non-empty.
    ///
    /// Returns the number of transcript rows it occupied (preview rows + the
    /// gap row), so the live todo panel stacked above it sits *above* it
    /// instead of overwriting it. `0` when nothing was drawn.
    fn draw_queued_previews(
        &self,
        frame: &mut ratatui::Frame<'_>,
        transcript: Rect,
        reserved_below: u16,
    ) -> u16 {
        // Show at most the last few entries so a long queue never eats the
        // transcript; older ones stay summarized by the count badge.
        const MAX_PREVIEW_ROWS: usize = 4;
        // One blank row between the lowest preview line and the input rule, so
        // the queue does not visually fuse with the input box.
        const GAP_ROWS: u16 = 1;
        // One blank row *above* the previews too, so the last visible transcript
        // line never butts directly against the first queued entry. The preview
        // is a `Clear` overlay on the transcript, so without this top boundary
        // the two surfaces visually fuse.
        const TOP_GAP_ROWS: u16 = 1;
        if self.queued_messages.is_empty() || transcript.width == 0 || transcript.height == 0 {
            return 0;
        }
        let total = self.queued_messages.len();
        let shown = total.min(MAX_PREVIEW_ROWS);
        // Reserve the gap rows above and below the previews; when the transcript
        // is too short for all of them, the previews shrink before either gap is
        // sacrificed.
        let available = transcript
            .height
            .saturating_sub(reserved_below)
            .saturating_sub(GAP_ROWS)
            .saturating_sub(TOP_GAP_ROWS);
        let rows = u16::try_from(shown).unwrap_or(0).min(available);
        if rows == 0 {
            return 0;
        }
        let preview_rect = Rect::new(
            transcript.x,
            transcript.y
                + transcript
                    .height
                    .saturating_sub(rows + GAP_ROWS + reserved_below),
            transcript.width,
            rows,
        );
        let hidden = total.saturating_sub(shown);
        let max_cols = usize::from(transcript.width.saturating_sub(2)).max(8);
        let mut lines: Vec<Line<'_>> = Vec::with_capacity(shown);
        for (i, msg) in self.queued_messages.iter().skip(hidden).enumerate() {
            let ordinal = hidden + i + 1;
            let body = queued_preview_label(msg);
            let clock = glyphs::pick(
                !self.theme.no_color,
                glyphs::QUEUE_CLOCK,
                glyphs::QUEUE_CLOCK_NC,
            );
            let raw = format!("{clock} {ordinal}. {body}");
            let truncated = truncate_to_cells(&raw, max_cols);
            lines.push(Line::from(Span::styled(
                truncated,
                Style::default().fg(self.theme.palette.dim),
            )));
        }
        // Clear the previews plus their top boundary row so the blank line is
        // real (no transcript glyphs bleed through directly above the queue).
        let top_pad = TOP_GAP_ROWS.min(preview_rect.y.saturating_sub(transcript.y));
        let clear_rect = Rect::new(
            preview_rect.x,
            preview_rect.y.saturating_sub(top_pad),
            preview_rect.width,
            preview_rect.height + top_pad,
        );
        frame.render_widget(Clear, clear_rect);
        let para = Paragraph::new(lines);
        frame.render_widget(para, preview_rect);
        // Report the rows consumed *including* the top boundary, so the live
        // todo panel stacked above sits above the blank line, not on it.
        rows + GAP_ROWS + top_pad
    }

    /// Actual, bounded executor rows for the live plan dock. Every value comes
    /// from an already-authoritative in-memory surface (`WorkflowSummary`, live
    /// agent manifests, or the in-flight tool action); no text parsing, disk IO,
    /// or guessed step-to-executor attribution happens in the draw path.
    ///
    /// Prefer a concrete agent, then an actual main-turn tool, and use the
    /// workflow heartbeat only as a last-resort spawn/between-phase fallback.
    fn run_dock_executor_row(&self) -> Option<RunDockExecutorRow> {
        if !self.live_plan_should_pin() {
            return None;
        }

        // Exact step attribution is opt-in: one real current workflow phase,
        // one non-blank step id, and one Todo row with that exact id. Ambiguous
        // or legacy snapshots stay on the existing run-level fallback below.
        let correlated_phase = self.hud_state.workflow.as_ref().and_then(|workflow| {
            let mut phases = workflow
                .phases
                .iter()
                .filter(|phase| phase.id == workflow.current_phase);
            let phase = phases.next()?;
            if phases.next().is_some() {
                return None;
            }
            let step_id = phase.step_id.as_deref()?;
            let base_status = phase.status.split(" · ").next().unwrap_or_default();
            if !matches!(base_status, "running" | "pending") || step_id != phase.id {
                return None;
            }
            unique_todo_step_index(&self.hud_state.todo_items, step_id)?;
            Some((phase, step_id))
        });

        if let Some((phase, step_id)) = correlated_phase {
            let agent_ids_are_valid = !phase.agent_ids.is_empty()
                && phase
                    .agent_ids
                    .iter()
                    .all(|id| !id.is_empty() && id.trim() == id);
            if agent_ids_are_valid {
                let live_agent = self.hud_state.agents.iter().find(|agent| {
                    !agent_status_is_terminal(&agent.status)
                        && phase.agent_ids.iter().any(|id| id == &agent.id)
                });
                let running_agents = u16::try_from(phase.running)
                    .unwrap_or(u16::MAX)
                    .max(u16::from(live_agent.is_some()));
                if running_agents > 0 {
                    return Some(agent_executor_row(
                        running_agents,
                        live_agent,
                        Some(step_id),
                    ));
                }
            }
        }

        let has_agent_rows = !self.hud_state.agents.is_empty();
        let live_agent = self
            .hud_state
            .agents
            .iter()
            .find(|agent| !agent_status_is_terminal(&agent.status));
        let workflow_running = self
            .hud_state
            .workflow
            .as_ref()
            .map_or(0, |workflow| workflow.running_agents);
        // Concrete manifests make `HudState::running_agents` authoritative.
        // Only fall back to workflow progress during the short spawn window
        // before manifest rows arrive; otherwise a stale workflow snapshot can
        // overstate the live count beside a newer manifest row.
        let running_agents = if has_agent_rows {
            self.hud_state.running_agents
        } else {
            self.hud_state
                .running_agents
                .max(u16::try_from(workflow_running).unwrap_or(u16::MAX))
        }
        .max(u16::from(live_agent.is_some()));
        // The HUD/sidebar already owns workflow phase topology. The Run Dock
        // shows the executor instead, avoiding a second "phase X/Y" while still
        // surviving the spawn window before concrete manifests land.
        if running_agents > 0 {
            return Some(agent_executor_row(running_agents, live_agent, None));
        }

        if let Some(action) = self
            .hud_state
            .last_tool
            .as_deref()
            .map(str::trim)
            .filter(|action| !action.is_empty())
        {
            return Some(RunDockExecutorRow {
                kind: RunDockExecutorKind::Main,
                detail: action.to_string(),
                step_id: None,
            });
        }

        self.hud_state
            .workflow
            .as_ref()
            .map(|_| RunDockExecutorRow {
                kind: RunDockExecutorKind::Workflow,
                detail: "active between phases".to_string(),
                step_id: correlated_phase.map(|(_, step_id)| step_id.to_string()),
            })
    }

    /// Whether this turn's plan is eligible for the pinned live surface. Kept as
    /// one predicate so executor formatting, geometry, and paint cannot disagree
    /// about stale carryover plans.
    fn live_plan_should_pin(&self) -> bool {
        if self.turn_activity.is_none() || !todo_panel_should_show(&self.hud_state.todo_items) {
            return false;
        }
        self.todo_touched_this_turn
            || self
                .hud_state
                .todo_items
                .iter()
                .any(|item| item.status == hud::TodoChecklistStatus::InProgress)
    }

    /// Render the live plan as a bottom-anchored Run Dock while a turn is in
    /// flight: plan rows remain stable and actual executor state appears below
    /// them when present. This is the *only* place the full live plan renders —
    /// the transcript's `TodoWrite` result is hidden (height 0), so the dock does
    /// not duplicate it.
    ///
    /// Pure over `self.turn_activity` + `self.hud_state.todo_items` (read-only):
    /// hidden when no turn is active, the list is empty, or every item is
    /// completed. The HUD/sidebar todo section is independent and keeps the
    /// completed snapshot for status accounting.
    #[allow(clippy::too_many_lines)] // one cohesive bordered-overlay paint
    fn draw_todo_panel(
        &self,
        frame: &mut ratatui::Frame<'_>,
        geometry: Option<TodoPanelGeometry>,
        executor_row: Option<&RunDockExecutorRow>,
    ) -> Option<Rect> {
        let geometry = geometry?;
        let items = &self.hud_state.todo_items;
        let executor_row = geometry.shows_executor.then_some(executor_row).flatten();

        let border_style = Style::default().fg(self.theme.palette.dim);
        let title_style = Style::default()
            .fg(self.theme.palette.dim)
            .add_modifier(ratatui::style::Modifier::BOLD);
        let total = self.hud_state.todo_items.len();
        let done = self
            .hud_state
            .todo_items
            .iter()
            .filter(|item| item.status == crate::tui::hud::TodoChecklistStatus::Completed)
            .count();
        let tally_style = if done == total {
            Style::default()
                .fg(self.theme.palette.success)
                .add_modifier(ratatui::style::Modifier::BOLD)
        } else {
            Style::default().fg(self.theme.palette.dim)
        };
        let tally_text = if total == 0 {
            "all done".to_string()
        } else {
            format!("{done}/{total} done")
        };

        let h = if self.theme.no_color { '-' } else { '─' };
        let (tl, tr, bl, br, v) = if self.theme.no_color {
            ('+', '+', '+', '+', '|')
        } else {
            ('╭', '╮', '╰', '╯', '│')
        };

        let mut lines: Vec<Line<'_>> = Vec::with_capacity(usize::from(geometry.rows));

        // 1. Top border. Keep the panel static and cheap: it redraws on the
        // live turn tick, so per-character animation here competes with answer
        // streaming for frame budget.
        let title = if executor_row.is_none() {
            "Updated Plan"
        } else {
            glyphs::pick(!self.theme.no_color, "Updated Plan → Execute", "Updated Plan -> Execute")
        };
        let header_content_width =
            UnicodeWidthStr::width(title) + 5 + UnicodeWidthStr::width(tally_text.as_str());
        let pad_width =
            usize::from(geometry.panel_rect.width).saturating_sub(3 + header_content_width + 1);
        let mut header_spans = vec![
            Span::styled(format!("{tl}{h} "), border_style),
            Span::styled(title.to_string(), title_style),
            Span::styled("  \u{00b7}  ".to_string(), border_style),
            Span::styled(tally_text, tally_style),
        ];
        header_spans.push(Span::styled(h.to_string().repeat(pad_width), border_style));
        header_spans.push(Span::styled(tr.to_string(), border_style));
        lines.push(Line::from(header_spans));

        // 2. Body rows — show EVERY item, completed ones rendered checked +
        // struck through in place (not removed), so finishing a step reads as a
        // checkmark accruing rather than a row silently vanishing. A plan longer
        // than the row budget slides a window that keeps the active frontier
        // (in-progress, else next pending) visible with its just-completed
        // context above it (see `plan_display_range`).
        let item_rows = geometry.plan_rows;
        let inner_width = usize::from(geometry.panel_rect.width).saturating_sub(3);
        let executor_step_index = executor_row
            .and_then(|executor| executor.step_id.as_deref())
            .and_then(|step_id| unique_todo_step_index(items, step_id));
        let window = plan_display_range_with_preferred(items, item_rows, executor_step_index);
        let window_start = window.start;
        let mut executor_rendered = false;
        for (offset, item) in items[window].iter().enumerate() {
            let (marker, marker_style) = todo_panel_marker(item.status, &self.theme);
            let text = if item.status == crate::tui::hud::TodoChecklistStatus::InProgress
                && !item.active_form.trim().is_empty()
            {
                item.active_form.as_str()
            } else {
                item.content.as_str()
            };

            let text_style = match item.status {
                crate::tui::hud::TodoChecklistStatus::Completed => Style::default()
                    .fg(self.theme.palette.dim)
                    .add_modifier(ratatui::style::Modifier::CROSSED_OUT),
                crate::tui::hud::TodoChecklistStatus::InProgress => Style::default()
                    .fg(self.theme.palette.fg)
                    .add_modifier(ratatui::style::Modifier::BOLD),
                crate::tui::hud::TodoChecklistStatus::Pending => {
                    Style::default().fg(self.theme.palette.fg)
                }
            };

            let marker_width = unicode_width::UnicodeWidthStr::width(marker);
            let max_text_width = inner_width.saturating_sub(marker_width + 1);
            let sanitized = crate::util::ansi::sanitize_inline(text);
            let truncated_text = truncate_to_cells(&sanitized, max_text_width);
            let displayed_content_width =
                marker_width + 1 + unicode_width::UnicodeWidthStr::width(truncated_text.as_str());
            let row_pad = inner_width.saturating_sub(displayed_content_width);

            let mut row_spans = vec![
                Span::styled(format!("{v} "), border_style),
                Span::styled(marker.to_string(), marker_style),
                Span::raw(" "),
            ];
            let row_text_style = if item.status == crate::tui::hud::TodoChecklistStatus::InProgress
            {
                // Brightness (not the brand accent) marks the active row —
                // the marker glyph already carries the live signal.
                Style::default()
                    .fg(self.theme.palette.bright)
                    .add_modifier(ratatui::style::Modifier::BOLD)
            } else {
                text_style
            };
            row_spans.push(Span::styled(truncated_text, row_text_style));
            row_spans.push(Span::styled(" ".repeat(row_pad), border_style));
            row_spans.push(Span::styled(v.to_string(), border_style));
            lines.push(Line::from(row_spans));

            if executor_step_index == Some(window_start + offset) {
                if let Some(executor) = executor_row {
                    lines.push(run_dock_executor_line(
                        executor,
                        inner_width,
                        v,
                        border_style,
                        &self.theme,
                    ));
                    executor_rendered = true;
                }
            }
        }

        // 3. Actual execution row. This is deliberately a compact summary,
        // not a second agent tree: the detailed tree remains inline / Ctrl+O,
        // while the dock makes the Plan -> Executor relationship visible at a
        // glance and replaces the duplicate pinned agent panel.
        if let Some(executor) = executor_row.filter(|_| !executor_rendered) {
            lines.push(run_dock_executor_line(
                executor,
                inner_width,
                v,
                border_style,
                &self.theme,
            ));
        }

        // 4. Bottom border. The two-row degraded form spends both rows on the
        // title and active frontier; lower-priority chrome yields first.
        if geometry.shows_bottom_border {
            let h_repeats = usize::from(geometry.panel_rect.width).saturating_sub(2);
            lines.push(Line::from(vec![Span::styled(
                format!("{bl}{}{br}", h.to_string().repeat(h_repeats)),
                border_style,
            )]));
        }

        frame.render_widget(Clear, geometry.clear_rect);
        frame.render_widget(Paragraph::new(lines), geometry.panel_rect);
        Some(geometry.panel_rect)
    }

    /// Measure twin of [`Self::draw_queued_previews`] — `reserved_below` must
    /// carry the same base (the search-bar row) the draw pass stacks from, or
    /// the reservation and the paint disagree by a row.
    fn queued_previews_height(&self, transcript: Rect, reserved_below: u16) -> u16 {
        const MAX_PREVIEW_ROWS: usize = 4;
        const GAP_ROWS: u16 = 1;
        const TOP_GAP_ROWS: u16 = 1;
        if self.queued_messages.is_empty() || transcript.width == 0 || transcript.height == 0 {
            return 0;
        }
        let total = self.queued_messages.len();
        let shown = total.min(MAX_PREVIEW_ROWS);
        let available = transcript
            .height
            .saturating_sub(reserved_below)
            .saturating_sub(GAP_ROWS)
            .saturating_sub(TOP_GAP_ROWS);
        let rows = u16::try_from(shown).unwrap_or(0).min(available);
        if rows == 0 {
            return 0;
        }
        let top_pad = TOP_GAP_ROWS.min(
            transcript
                .height
                .saturating_sub(rows + GAP_ROWS + reserved_below),
        );
        rows + GAP_ROWS + top_pad
    }

    /// Lines for the pinned live-agent panel, or empty when it should not show.
    /// Shown only during an active turn with live (non-terminal) sub-agents: the
    /// Claude-Code-style always-visible fan-out tree pinned above the input, so
    /// `Delegating` is a moving per-agent tree (tool / tokens / elapsed / output
    /// tail) instead of a dead "no output" spinner — even after the host
    /// `ToolCall` row scrolls off, or for fan-out paths that open no host row.
    fn agent_panel_lines(&self) -> (Vec<Line<'static>>, Vec<AgentRowSpan>) {
        if self.turn_activity.is_none() {
            return (Vec::new(), Vec::new());
        }
        if !self.hud_state.agents.is_empty() {
            return crate::tui::blocks::tool_call::live_agent_panel_lines_with_spans(
                &self.hud_state.agents,
                &self.theme,
                self.active_agent_batch_label(),
                self.hovered_agent.as_deref(),
            );
        }
        let Some(workflow) = self
            .hud_state
            .workflow
            .as_ref()
            .filter(|workflow| workflow.running_agents > 0)
        else {
            return (Vec::new(), Vec::new());
        };
        // The workflow placeholder carries no per-agent ids (it is an aggregate
        // phase summary), so it records no click spans — a click still opens the
        // aggregate viewer via `agent_panel_click_rect`.
        (self.workflow_agent_placeholder_lines(workflow), Vec::new())
    }

    fn workflow_agent_placeholder_lines(
        &self,
        workflow: &crate::tui::workflow_progress::WorkflowSummary,
    ) -> Vec<Line<'static>> {
        let color = !self.theme.no_color;
        let agents = workflow.running_agents;
        let mut lines = vec![Line::from(vec![
            Span::styled(
                format!(
                    "{} ",
                    glyphs::pick(color, "●", "*")
                ),
                Style::new().fg(self.theme.palette.teal),
            ),
            Span::styled(
                format!("Running {agents} Workflow agents… (ctrl+g for details)"),
                Style::new()
                    .fg(self.theme.palette.fg)
                    .add_modifier(Modifier::BOLD),
            ),
        ])];

        let mut shown = 0usize;
        for phase in workflow
            .phases
            .iter()
            .filter(|phase| phase.running > 0 || phase.total > 0)
            .take(3)
        {
            shown += 1;
            let terminal = phase.completed.saturating_add(phase.failed);
            let status = if phase.running > 0 {
                format!("{} running", phase.running)
            } else {
                phase.status.clone()
            };
            let fail = if phase.failed > 0 {
                format!(" · {} failed", phase.failed)
            } else {
                String::new()
            };
            lines.push(Line::from(vec![
                Span::styled("  ⎿ ", Style::new().fg(self.theme.palette.dim)),
                Span::styled(
                    phase.id.clone(),
                    Style::new()
                        .fg(self.theme.palette.fg)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" · {terminal}/{} · {status}{fail}", phase.total),
                    Style::new().fg(self.theme.palette.dim),
                ),
            ]));
        }
        let remaining = workflow
            .phases
            .iter()
            .filter(|phase| phase.running > 0 || phase.total > 0)
            .count()
            .saturating_sub(shown);
        if remaining > 0 {
            lines.push(Line::from(vec![
                Span::styled("  ⎿ ", Style::new().fg(self.theme.palette.dim)),
                Span::styled(
                    format!("+{remaining} more phases"),
                    Style::new().fg(self.theme.palette.dim),
                ),
            ]));
        }
        lines
    }

    fn agent_panel_height(
        panel_lines: &[Line<'static>],
        transcript: Rect,
        reserved_below: u16,
    ) -> u16 {
        pinned_agent_panel_geometry(panel_lines.len(), transcript, reserved_below)
            .map_or(0, |geometry| geometry.reserved_height)
    }

    /// Draw the pinned live-agent panel as a bottom-anchored overlay stacked
    /// above the queue preview / todo panel, mirroring their `Clear`-then-paint
    /// stacking so the three never overwrite each other. Returns the painted
    /// panel rect (`None` when the panel is not showing) so the draw loop can
    /// record it as the click target that opens the live agent view.
    ///
    /// `panel_lines` is built ONCE per frame by the draw loop and shared with
    /// [`Self::agent_panel_height`] — building it in both places allocated a
    /// full O(agents) line set that the height check immediately threw away,
    /// every frame of a fan-out.
    fn draw_agent_panel(
        frame: &mut ratatui::Frame<'_>,
        panel_lines: &[Line<'static>],
        row_spans: &[AgentRowSpan],
        transcript: Rect,
        reserved_below: u16,
    ) -> (Option<Rect>, Vec<(Rect, String)>) {
        let Some(geometry) =
            pinned_agent_panel_geometry(panel_lines.len(), transcript, reserved_below)
        else {
            return (None, Vec::new());
        };
        let shown: Vec<Line<'_>> = panel_lines
            .iter()
            .take(usize::from(geometry.rows))
            .cloned()
            .collect();
        frame.render_widget(Clear, geometry.clear_rect);
        frame.render_widget(Paragraph::new(shown), geometry.panel_rect);

        // Map each agent's line span to an absolute screen rect for
        // click-to-inspect. A span that starts at or past the visible cut
        // (`geometry.rows` truncation, matching the `take` above) records no
        // rect, so a click below the fold falls through to the aggregate
        // panel-click. The width matches the panel so the whole row is a target.
        let panel = geometry.panel_rect;
        let mut row_rects: Vec<(Rect, String)> = Vec::with_capacity(row_spans.len());
        for span in row_spans {
            if span.start >= geometry.rows {
                continue;
            }
            let visible_len = span.len.min(geometry.rows - span.start);
            if visible_len == 0 {
                continue;
            }
            let rect = Rect::new(panel.x, panel.y + span.start, panel.width, visible_len);
            row_rects.push((rect, span.id.clone()));
        }
        (Some(panel), row_rects)
    }

    fn todo_panel_geometry(
        &self,
        transcript: Rect,
        reserved_below: u16,
        has_executor: bool,
    ) -> Option<TodoPanelGeometry> {
        if !self.live_plan_should_pin() || transcript.width == 0 || transcript.height == 0 {
            return None;
        }
        let items = &self.hud_state.todo_items;

        // Budget for plan rows plus one compact executor row. This is
        // still lower than the former plan-panel + full pinned-agent-tree stack,
        // and the body calculation below always reserves at least one plan row
        // when vertical degradation clips the dock.
        // The gap below is only needed when this panel is the bottom-most
        // overlay (nothing reserved beneath): a stacked neighbour below already
        // ends with its own one-row top pad, so adding a second blank row here
        // was what spread the overlay stack apart (the "blank band" complaint).
        let gap_below = u16::from(reserved_below == 0) * TODO_PANEL_GAP_ROWS;
        let plan_rows_wanted = items.len().min(TODO_PANEL_MAX_ROWS);
        let executor_rows_wanted = usize::from(has_executor);
        let shown = plan_rows_wanted + executor_rows_wanted + 2;
        let available = transcript
            .height
            .saturating_sub(reserved_below)
            .saturating_sub(gap_below)
            .saturating_sub(TODO_PANEL_TOP_GAP_ROWS);
        let rows = u16::try_from(shown).unwrap_or(0).min(available);
        if rows < 2 {
            return None;
        }
        let shows_bottom_border = rows > 2;
        let chrome_rows = 1 + u16::from(shows_bottom_border);
        let body_rows = usize::from(rows - chrome_rows);
        let shows_executor = has_executor && body_rows > 1;
        let plan_rows = plan_rows_wanted.min(body_rows - usize::from(shows_executor));

        // Always bottom-anchor the live panel just above the input / queue
        // preview. The same geometry drives both transcript reservation and
        // overlay drawing so the border box never paints over transcript rows.
        let panel_top_y = transcript.y
            + transcript
                .height
                .saturating_sub(reserved_below + rows + gap_below);
        let panel_rect = Rect::new(transcript.x, panel_top_y, transcript.width, rows);
        let top_pad = TODO_PANEL_TOP_GAP_ROWS.min(panel_rect.y.saturating_sub(transcript.y));
        let clear_rect = Rect::new(
            panel_rect.x,
            panel_rect.y.saturating_sub(top_pad),
            panel_rect.width,
            panel_rect.height + top_pad,
        );

        Some(TodoPanelGeometry {
            panel_rect,
            clear_rect,
            rows,
            reserved_height: rows + gap_below + top_pad,
            plan_rows,
            shows_executor,
            shows_bottom_border,
        })
    }
}

const TODO_PANEL_MAX_ROWS: usize = 5;
const TODO_PANEL_GAP_ROWS: u16 = 1;
const TODO_PANEL_TOP_GAP_ROWS: u16 = 1;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunDockExecutorKind {
    Workflow,
    Agents,
    Main,
}

struct RunDockExecutorRow {
    kind: RunDockExecutorKind,
    detail: String,
    /// Exact owning Todo step. `None` means this is intentionally the legacy
    /// run-level fallback and must render after the whole visible plan.
    step_id: Option<String>,
}

impl RunDockExecutorRow {
    const fn label(&self) -> &'static str {
        match self.kind {
            RunDockExecutorKind::Workflow => "workflow",
            RunDockExecutorKind::Agents => "agents",
            RunDockExecutorKind::Main => "main",
        }
    }

    const fn is_inspectable(&self) -> bool {
        matches!(
            self.kind,
            RunDockExecutorKind::Workflow | RunDockExecutorKind::Agents
        )
    }
}

fn agent_executor_row(
    running_agents: u16,
    live_agent: Option<&hud::AgentTaskSummary>,
    step_id: Option<&str>,
) -> RunDockExecutorRow {
    let noun = if running_agents == 1 { "agent" } else { "agents" };
    let mut detail = format!("{running_agents} {noun} running");
    if let Some(agent) = live_agent {
        detail.push_str(" · ");
        detail.push_str(&agent.name);
        if let Some(activity) = agent.activity_label() {
            detail.push_str(" -> ");
            detail.push_str(activity);
        }
    }
    RunDockExecutorRow {
        kind: RunDockExecutorKind::Agents,
        detail,
        step_id: step_id.map(str::to_string),
    }
}

fn run_dock_executor_line(
    executor: &RunDockExecutorRow,
    inner_width: usize,
    vertical_border: char,
    border_style: Style,
    theme: &crate::tui::theme::Theme,
) -> Line<'static> {
    let branch = glyphs::pick(!theme.no_color, "└─ ", "`- ");
    let label = executor.label();
    let fixed_width = UnicodeWidthStr::width(branch)
        + UnicodeWidthStr::width(label)
        + UnicodeWidthStr::width(" · ");
    let max_detail_width = inner_width.saturating_sub(fixed_width);
    let sanitized = crate::util::ansi::sanitize_inline(&executor.detail);
    let detail = truncate_to_cells(&sanitized, max_detail_width);
    let content_width = fixed_width + UnicodeWidthStr::width(detail.as_str());
    let row_pad = inner_width.saturating_sub(content_width);
    Line::from(vec![
        Span::styled(format!("{vertical_border} "), border_style),
        Span::styled(branch, Style::new().fg(theme.palette.teal)),
        Span::styled(
            label,
            Style::new()
                .fg(theme.palette.teal)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · ", theme.typography.dim),
        Span::styled(detail, theme.typography.body),
        Span::styled(" ".repeat(row_pad), border_style),
        Span::styled(vertical_border.to_string(), border_style),
    ])
}

#[derive(Clone, Copy)]
struct TodoPanelGeometry {
    panel_rect: Rect,
    clear_rect: Rect,
    rows: u16,
    reserved_height: u16,
    plan_rows: usize,
    shows_executor: bool,
    shows_bottom_border: bool,
}

/// Bottom-anchored geometry for the pinned live-agent panel, computed once and
/// shared by the reservation (`agent_panel_height`) and the paint
/// (`draw_agent_panel`) so the two never disagree. Stacks `reserved_below` the
/// queue preview / todo panel with a one-row top pad; the gap below only
/// exists when the panel is the bottom-most overlay (stacked neighbours
/// already provide the one-row seam via their own top pad).
pub(super) struct PinnedAgentPanelGeometry {
    panel_rect: Rect,
    clear_rect: Rect,
    rows: u16,
    pub(super) reserved_height: u16,
}

pub(super) fn pinned_agent_panel_geometry(
    line_count: usize,
    transcript: Rect,
    reserved_below: u16,
) -> Option<PinnedAgentPanelGeometry> {
    const TOP_GAP_ROWS: u16 = 1;
    if line_count == 0 || transcript.width == 0 || transcript.height == 0 {
        return None;
    }
    // Gap below only when this panel is the bottom-most overlay: a stacked
    // todo/queue neighbour below already ends with its own one-row top pad
    // (see `todo_panel_geometry`), so the shared seam stays exactly one row.
    let gap_below = u16::from(reserved_below == 0);
    let want = u16::try_from(line_count).unwrap_or(u16::MAX);
    let available = transcript
        .height
        .saturating_sub(reserved_below)
        .saturating_sub(gap_below)
        .saturating_sub(TOP_GAP_ROWS);
    let rows = want.min(available);
    if rows == 0 {
        return None;
    }
    let panel_top_y =
        transcript.y + transcript.height.saturating_sub(reserved_below + rows + gap_below);
    let panel_rect = Rect::new(transcript.x, panel_top_y, transcript.width, rows);
    let top_pad = TOP_GAP_ROWS.min(panel_rect.y.saturating_sub(transcript.y));
    let clear_rect = Rect::new(
        panel_rect.x,
        panel_rect.y.saturating_sub(top_pad),
        panel_rect.width,
        panel_rect.height + top_pad,
    );
    Some(PinnedAgentPanelGeometry {
        panel_rect,
        clear_rect,
        rows,
        reserved_height: rows + gap_below + top_pad,
    })
}

fn todo_panel_should_show(items: &[hud::TodoChecklistItem]) -> bool {
    // Lifecycle code clears all-completed snapshots before paint. This local
    // predicate only guards the empty surface; freshness is gated separately by
    // `live_plan_should_pin`, so untouched carryover plans never re-pin.
    !items.is_empty()
}

/// Return the sole Todo row carrying `step_id`. Invalid or duplicated ids are
/// deliberately non-correlatable even though the writer rejects them too: the
/// HUD remains honest when reading a legacy or hand-edited store.
fn unique_todo_step_index(items: &[hud::TodoChecklistItem], step_id: &str) -> Option<usize> {
    if step_id.is_empty() || step_id.trim() != step_id {
        return None;
    }
    let mut matching = items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.step_id.as_deref() == Some(step_id));
    let (index, _) = matching.next()?;
    matching.next().is_none().then_some(index)
}

/// The contiguous slice of plan rows to paint in the height-capped live panel.
///
/// Shows every item when the plan fits the row budget; otherwise slides a window
/// that keeps the active frontier (the in-progress item, else the first pending,
/// else the last item) visible with its recently-completed context above it — so
/// completing a step reads as a checkmark accruing, never a row vanishing.
fn plan_display_range(
    items: &[hud::TodoChecklistItem],
    max_rows: usize,
) -> std::ops::Range<usize> {
    let n = items.len();
    if max_rows == 0 {
        return 0..0;
    }
    if n <= max_rows {
        return 0..n;
    }
    let frontier = items
        .iter()
        .position(|item| item.status == hud::TodoChecklistStatus::InProgress)
        .or_else(|| {
            items
                .iter()
                .position(|item| item.status == hud::TodoChecklistStatus::Pending)
        })
        .unwrap_or(n - 1);
    // Anchor the frontier near the window bottom. Keep one lookahead row when
    // space allows, but a one-row degraded dock must show the frontier itself.
    let lookahead = usize::from(max_rows > 1);
    let end = (frontier + 1 + lookahead).clamp(max_rows, n);
    let start = end - max_rows;
    start..end
}

/// Keep an exactly-correlated executor's owner visible when the plan is taller
/// than the dock. Without a preferred owner this is byte-for-byte the existing
/// active-frontier window, including the one-row short-terminal fallback.
fn plan_display_range_with_preferred(
    items: &[hud::TodoChecklistItem],
    max_rows: usize,
    preferred: Option<usize>,
) -> std::ops::Range<usize> {
    let default = plan_display_range(items, max_rows);
    let Some(preferred) = preferred.filter(|index| *index < items.len()) else {
        return default;
    };
    if max_rows == 0 || default.contains(&preferred) {
        return default;
    }
    let lookahead = usize::from(max_rows > 1);
    let end = (preferred + 1 + lookahead).clamp(max_rows, items.len());
    end - max_rows..end
}

/// Marker glyph + style for one live todo-panel row. Panel-local (does not
/// couple to the sidebar's private `todo_marker`) so the two surfaces can
/// evolve independently; honors `no_color` with ASCII boxes.
fn todo_panel_marker(
    status: crate::tui::hud::TodoChecklistStatus,
    theme: &crate::tui::theme::Theme,
) -> (&'static str, Style) {
    use crate::tui::hud::TodoChecklistStatus;
    if theme.no_color {
        match status {
            TodoChecklistStatus::Pending => ("[ ]", Style::default().fg(theme.palette.dim)),
            TodoChecklistStatus::InProgress => ("[~]", Style::default().fg(theme.palette.warn)),
            TodoChecklistStatus::Completed => ("[x]", Style::default().fg(theme.palette.success)),
        }
    } else {
        match status {
            TodoChecklistStatus::Pending => ("\u{2610}", Style::default().fg(theme.palette.dim)),
            TodoChecklistStatus::InProgress => {
                ("\u{25d0}", Style::default().fg(theme.palette.warn))
            }
            TodoChecklistStatus::Completed => {
                ("\u{2611}", Style::default().fg(theme.palette.success))
            }
        }
    }
}

/// One-line label for a queued message: its text, or an image-count summary
/// when the entry is image-only.
fn queued_preview_label(msg: &super::QueuedMessage) -> String {
    let trimmed = msg.text.trim();
    if !trimmed.is_empty() {
        return trimmed.replace('\n', " ");
    }
    match msg.images.len() {
        0 => String::new(),
        1 => "[image]".to_string(),
        n => format!("[{n} images]"),
    }
}

/// The queued-message rule badge text and its width in DISPLAY CELLS. Sizing by
/// cells (not `badge.len()`) keeps the right-aligned badge on the rule edge: the
/// rich queue clock is three UTF-8 bytes but occupies one display column.
fn queue_badge(count: usize, color: bool) -> (String, u16) {
    let clock = glyphs::pick(color, glyphs::QUEUE_CLOCK, glyphs::QUEUE_CLOCK_NC);
    let badge = format!(" {clock} {count} queued ");
    let width = u16::try_from(unicode_width::UnicodeWidthStr::width(badge.as_str())).unwrap_or(0);
    (badge, width)
}

/// Truncate `text` to at most `max_cells` terminal cells, appending an ellipsis
/// when content is dropped. CJK-aware via `unicode-width`.
/// v3 §10 frosted modal backdrop: replace the opaque `Clear` with a glass
/// pane. Everything *outside* `pane` keeps its glyphs but has its foreground
/// pulled halfway to the surface base ([`Theme::scrim_fg`]) — the terminal
/// equivalent of backdrop blur, so the conversation stays visible-but-muted
/// behind the modal instead of competing with it. The pane itself is cleared
/// and filled with the elevation-2 glass surface so the modal's own content
/// draws on a pane that reads *above* the muted backdrop.
///
/// Cost: one O(visible cells) fg remap, only on frames where a modal is
/// drawn — a modal-less frame pays nothing. On the `NO_COLOR` neutral palette
/// there is no blend, so this degrades to the old opaque `Clear`.
fn frost_modal_backdrop(frame: &mut ratatui::Frame, pane: Rect, theme: &crate::tui::theme::Theme) {
    let area = frame.area();
    let Some(surface) = theme.surface2() else {
        frame.render_widget(Clear, pane);
        return;
    };
    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if pane.contains(ratatui::layout::Position::new(x, y)) {
                continue;
            }
            let cell = &mut buffer[(x, y)];
            let Some(fg) = cell.style().fg else {
                continue;
            };
            let muted = theme.scrim_fg(fg);
            if muted != fg {
                cell.set_style(Style::default().fg(muted));
            }
        }
    }
    for y in pane.top()..pane.bottom() {
        for x in pane.left()..pane.right() {
            let cell = &mut buffer[(x, y)];
            cell.reset();
            cell.set_style(Style::default().bg(surface));
        }
    }
}

fn truncate_to_cells(text: &str, max_cells: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if max_cells == 0 {
        return String::new();
    }
    let total: usize = text
        .chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum();
    if total <= max_cells {
        return text.to_string();
    }
    let budget = max_cells.saturating_sub(1);
    let mut used = 0usize;
    let mut out = String::new();
    for c in text.chars() {
        let w = UnicodeWidthChar::width(c).unwrap_or(0);
        if used + w > budget {
            break;
        }
        used += w;
        out.push(c);
    }
    out.push('\u{2026}');
    out
}

#[cfg(test)]
mod truncate_tests {
    use crate::tui::glyphs;

    use super::truncate_to_cells;
    use unicode_width::UnicodeWidthStr;

    /// Core contract every live-panel / queue-preview row depends on: the
    /// rendered string never occupies more terminal cells than its budget,
    /// across narrow and wide leading glyphs and at the tight boundary widths
    /// where the old `.max(1)` ellipsis budget overflowed a 1-cell slot.
    #[test]
    fn truncate_to_cells_never_exceeds_budget() {
        for &text in &[
            "abc",
            "\u{d55c}\u{ad6d}\u{c5b4}",
            "a\u{d55c}b",
            "ab",
            "\u{c218}\u{c815}\u{d558}\u{ace0}",
        ] {
            for max_cells in [1usize, 2, 3] {
                let out = truncate_to_cells(text, max_cells);
                let width = UnicodeWidthStr::width(out.as_str());
                assert!(
                    width <= max_cells,
                    "truncate_to_cells({text:?}, {max_cells}) = {out:?} width {width} > {max_cells}"
                );
            }
        }
    }

    /// The exact regression: a 1-cell slot must hold only the ellipsis, never
    /// `"a\u{2026}"` (width 2). Fails before the fix, passes after.
    #[test]
    fn truncate_to_cells_one_ascii_cell_is_just_ellipsis() {
        let out = truncate_to_cells("abc", 1);
        assert_eq!(out, "\u{2026}");
        assert_eq!(UnicodeWidthStr::width(out.as_str()), 1);
    }

    #[test]
    fn truncate_to_cells_zero_budget_is_empty() {
        assert_eq!(truncate_to_cells("anything", 0), "");
    }

    /// Fast path: content already within budget is returned verbatim, no
    /// ellipsis appended.
    #[test]
    fn truncate_to_cells_fitting_content_is_unchanged() {
        assert_eq!(truncate_to_cells("abc", 3), "abc");
        assert_eq!(truncate_to_cells("\u{d55c}", 2), "\u{d55c}");
        assert_eq!(truncate_to_cells("", 5), "");
    }

    /// Queue badge must be sized by display cells, not bytes: the rich clock is
    /// three UTF-8 bytes but one display column. Locks sizing to `unicode_width`.
    #[test]
    fn queue_badge_width_is_display_cells_not_bytes() {
        // Exercises the production rich-glyph path; byte length over-counts it.
        let (badge, width) = super::queue_badge(3, true);
        assert!(
            badge.len() > usize::from(width),
            "queue clock is wider in bytes than cells: bytes={} cells={width}",
            badge.len()
        );
        // " ◷ 3 queued " = 1+1+1+1+1+6+1 = 12 display cells.
        assert_eq!(width, 12, "badge {badge:?} display width");
        assert!(badge.contains(glyphs::QUEUE_CLOCK));

        let (plain, plain_width) = super::queue_badge(3, false);
        assert_eq!(plain_width, 12);
        assert!(plain.contains(glyphs::QUEUE_CLOCK_NC));
    }
}

#[cfg(test)]
mod frost_tests {
    use super::{Rect, frost_modal_backdrop};
    use crate::tui::theme::Theme;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::{Color, Style};
    use ratatui::text::Line;
    use ratatui::widgets::Paragraph;

    /// The frosted backdrop mutes fg outside the pane (visible-but-muted, the
    /// terminal stand-in for backdrop blur), while the pane itself is cleared
    /// and filled with the elevation-2 glass surface.
    #[test]
    fn frost_mutes_backdrop_and_fills_pane_with_glass() {
        let theme = Theme::zo();
        let surface = theme.surface2().expect("truecolor theme has surface2");
        let fg = Color::Rgb(220, 220, 220);
        let backend = TestBackend::new(20, 5);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let pane = Rect::new(5, 1, 8, 2);
        terminal
            .draw(|frame| {
                let lines: Vec<Line> = (0..5)
                    .map(|_| Line::styled("abcdefghijklmnopqrst", Style::new().fg(fg)))
                    .collect();
                frame.render_widget(Paragraph::new(lines), frame.area());
                frost_modal_backdrop(frame, pane, &theme);
            })
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();

        let outside = &buffer[(0u16, 0u16)];
        assert_eq!(
            outside.style().fg,
            Some(theme.scrim_fg(fg)),
            "backdrop fg must be pulled toward the surface base"
        );
        assert_eq!(outside.symbol(), "a", "backdrop glyphs stay visible");

        let inside = &buffer[(6u16, 1u16)];
        assert_eq!(inside.symbol(), " ", "the pane is cleared for the modal");
        assert_eq!(
            inside.style().bg,
            Some(surface),
            "the pane carries the elevation-2 glass surface"
        );
    }

    /// Under the `NO_COLOR` neutral palette there is no blend: the pane still
    /// clears (opaque legacy behavior) and the backdrop is left untouched.
    #[test]
    fn frost_degrades_to_opaque_clear_without_color() {
        let theme = Theme::no_color();
        let backend = TestBackend::new(12, 3);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let pane = Rect::new(4, 1, 4, 1);
        terminal
            .draw(|frame| {
                frame.render_widget(
                    Paragraph::new(vec![Line::raw("xxxxxxxxxxxx"); 3]),
                    frame.area(),
                );
                frost_modal_backdrop(frame, pane, &theme);
            })
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        assert_eq!(buffer[(0u16, 0u16)].symbol(), "x", "backdrop untouched");
        assert_eq!(buffer[(5u16, 1u16)].symbol(), " ", "pane cleared");
        assert_eq!(
            buffer[(5u16, 1u16)].style().bg,
            Some(Color::Reset),
            "no glass bg without color — the pane keeps the terminal default"
        );
    }
}
