use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use core_types::{RateLimitSnapshot, RateLimitWindow, TokenUsage, UsageDashboardSnapshot};
use ratatui::Terminal;
use ratatui::backend::{Backend, TestBackend};
use ratatui::layout::{Position, Rect};
use ratatui::{TerminalOptions, Viewport};
use runtime::message_stream::{
    ActiveModel, BlockId, QuestionOption, RenderBlock, SystemLevel, TodoResultStatus, ToolCallId,
    ToolCallStatus, ToolPreview, ToolResultBody, UserQuestionPrompt,
};
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};

use super::{AckVerdict, SpectatorEvent};

use crate::tui::hud::{AgentTaskSummary, PermissionMode};
use crate::tui::HeatState;
use crate::tui::modals::{ApiKeyConnectInfo, RemoteOnboardingView, RemotePendingPair};
use crate::tui::modals::Effort;
use crate::tui::modals::ModalPlacement;
use crate::tui::modals::UsageDashboardModal;
use crate::tui::modals::workflow_viewer::{
    WorkflowAgentRow, WorkflowPhaseRow, WorkflowView, WorkflowViewerModal,
};
use crate::tui::theme::Theme;
use crate::tui::startup::{
    STARTUP_LOGIN_CLAUDE_COMMAND, STARTUP_LOGIN_OPENAI_COMMAND, STARTUP_PERMISSIONS_COMMAND,
    STARTUP_SUMMARIZE_REPO_PROMPT, StartupAuthState, StartupScreen,
};
use crate::tui::workflow_progress::{FleetPhase, WorkflowSummary};

use super::modal_geometry::{
    centered_modal_rect, effort_modal_rect, modal_size_for_mode, point_in_rect,
};
use super::queue::{MAX_PENDING_IMAGES, MAX_QUEUED_MESSAGES};
use super::slash_hint::slash_hint_suggestions;
use super::{
    reasoning_activity_summary, reasoning_title_source, AgentCommand, App, AppAction, AppMode,
    ImageAttachment, ModelPickerEntry, QueueLimitError, ScheduledWakeHud, ToolToggleRow, WakeSource,
};
use crate::tui::layout::LayoutRegions;
use crate::tui::{INLINE_VIEWPORT_HEIGHT, TerminalMode};

#[test]
fn centered_modal_rect_centers_in_80x24() {
    let area = Rect::new(0, 0, 80, 24);
    let rect = centered_modal_rect(area, (40, 10));
    assert_eq!(rect.x, 20);
    assert_eq!(rect.y, 7);
    assert_eq!(rect.width, 40);
    assert_eq!(rect.height, 10);
}

#[test]
fn effort_modal_stays_within_chat_column() {
    // Wide terminal ⇒ sidebar is present; the effort slider must never
    // paint over it (regression: it previously used the full screen
    // width and covered the ZO ledger on the right).
    let area = Rect::new(0, 0, 160, 40);
    let regions = LayoutRegions::compute_with_sidebar(area, 3, 1, true).expect("layout computes");
    assert!(
        regions.sidebar_width > 0,
        "wide layout should have a sidebar"
    );

    let rect = effort_modal_rect(&regions, area);
    let chat = regions.transcript;
    // Right edge of the modal must stay left of the sidebar.
    assert!(
        rect.x + rect.width <= chat.x + chat.width,
        "modal right edge {} exceeds chat column right {}",
        rect.x + rect.width,
        chat.x + chat.width
    );
    assert!(rect.x >= chat.x, "modal starts left of chat column");
    // No corner of the modal lands inside the sidebar rect.
    assert!(!point_in_rect(
        rect.x + rect.width - 1,
        rect.y,
        regions.sidebar
    ));
}

#[test]
fn effort_modal_falls_back_to_area_without_transcript() {
    let area = Rect::new(0, 0, 100, 30);
    let empty = LayoutRegions {
        sidebar: Rect::new(0, 0, 0, 0),
        sidebar_width: 0,
        transcript: Rect::new(0, 0, 0, 0),
        rule_top: Rect::new(0, 0, 0, 0),
        input: Rect::new(0, 0, 0, 0),
        rule_bot: Rect::new(0, 0, 0, 0),
        hud: Rect::new(0, 0, 0, 0),
    };
    let rect = effort_modal_rect(&empty, area);
    assert!(
        rect.width > 0 && rect.height > 0,
        "fallback yields a visible rect"
    );
    assert!(rect.x + rect.width <= area.x + area.width);
}

#[test]
fn centered_modal_rect_clamps_to_area_margin() {
    let area = Rect::new(0, 0, 20, 10);
    let rect = centered_modal_rect(area, (40, 20));
    assert_eq!(rect.width, 16);
    assert_eq!(rect.height, 6);
    assert_eq!(rect.x, 2);
    assert_eq!(rect.y, 2);
}

/// Helper: build a minimal `App` for unit tests.
fn test_app() -> App {
    test_app_with_theme(Theme::no_color())
}

fn test_app_with_theme(theme: Theme) -> App {
    let (_block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
    let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
    App::new(theme, block_rx, cmd_tx)
}

#[tokio::test]
async fn run_accepts_input_after_caller_cancels_pending_wait() {
    let (_block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
    let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
    let mut app = App::new(Theme::no_color(), block_rx, cmd_tx);
    app.enable_input();
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("test terminal");
    let (event_tx, mut event_rx) =
        mpsc::unbounded_channel::<std::io::Result<Event>>();
    let mut events = futures_util::stream::poll_fn(move |cx| event_rx.poll_recv(cx));

    let cancelled = tokio::time::timeout(
        std::time::Duration::from_millis(10),
        app.run_with_events(&mut terminal, &mut events),
    )
    .await;
    assert!(
        cancelled.is_err(),
        "the first idle input wait must remain pending until its caller cancels it"
    );

    event_tx
        .send(Ok(Event::Paste("이거 1011 계속".to_string())))
        .expect("queue Hangul paste after cancellation");
    event_tx
        .send(Ok(Event::Key(press(KeyCode::Enter))))
        .expect("queue submit after cancellation");

    let action = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        app.run_with_events(&mut terminal, &mut events),
    )
    .await
    .expect("resumed input wait must not freeze")
    .expect("resumed input wait must succeed");
    assert_eq!(action, AppAction::Submit("이거 1011 계속".to_string()));
}

#[cfg(unix)]
#[test]
fn history_persistence_failure_surfaces_one_warning() {
    use crate::tui::command_history::CommandHistory;

    let mut app = test_app();
    app.set_command_history(
        CommandHistory::load(PathBuf::from("/dev/null/commands.jsonl"))
            .expect("bind command history to an unwritable path"),
    );
    app.record_command_usage("/first");
    app.record_command_usage("/second");

    let warnings = app
        .transcript
        .blocks()
        .iter()
        .filter(|block| {
            matches!(
                block,
                RenderBlock::System {
                    level: SystemLevel::Warn,
                    text,
                    ..
                } if text.starts_with("History was not saved: ")
            )
        })
        .count();
    assert_eq!(warnings, 1, "history failures should warn once per session");
}

/// The generic report popup: `c` copies the plain-text projection through the
/// host clipboard action while the popup stays open for further reading, and
/// Esc closes back to Normal. Scroll mechanics are pinned at the modal level
/// in `report_popup_modal_scrolls_and_projects_copy_text`.
#[test]
fn report_popup_copies_and_closes() {
    use crate::tui::modals::{ReportTone, ReportViewerBlock};

    let mut app = test_app();
    let body = (1..=60)
        .map(|index| format!("row {index}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.open_report_modal(
        "/doctor".to_string(),
        vec![ReportViewerBlock::Text {
            tone: ReportTone::Info,
            body,
        }],
    );
    assert_eq!(app.mode(), AppMode::ModalReport);

    let copy = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
        .expect("copy key handled");
    match copy {
        AppAction::ClipboardCopyBlock(text) => {
            assert!(text.contains("row 1") && text.contains("row 60"));
        }
        other => panic!("expected clipboard copy action, got {other:?}"),
    }
    assert_eq!(
        app.mode(),
        AppMode::ModalReport,
        "copy must keep the popup open"
    );

    let _ = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert_eq!(app.mode(), AppMode::Normal);
}

/// Modal-level report popup mechanics: key scrolling clamps to the body, the
/// copy payload carries every block (text + card) as plain text, and cards
/// keep their title row so the copied report reads standalone.
#[test]
fn report_popup_modal_scrolls_and_projects_copy_text() {
    use crate::tui::modals::{ReportTone, ReportViewerBlock, ReportViewerModal};
    use core_types::CardModel;

    let body = (1..=30)
        .map(|index| format!("line {index}"))
        .collect::<Vec<_>>()
        .join("\n");
    let card = CardModel::new(" /doctor ").section("Environment");
    let mut modal = ReportViewerModal::new(
        "/doctor",
        vec![
            ReportViewerBlock::Text {
                tone: ReportTone::Info,
                body,
            },
            ReportViewerBlock::Card(card),
        ],
        &Theme::no_color(),
    );

    assert!(modal.copy_text().contains("line 30"));
    assert!(modal.copy_text().contains("Environment"));

    assert_eq!(modal.scroll_offset(), 0);
    let _ = modal.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(modal.scroll_offset(), 0, "top is clamped");
    let _ = modal.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(modal.scroll_offset(), 8);
    let _ = modal.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
    let bottom = modal.scroll_offset();
    let _ = modal.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(modal.scroll_offset(), bottom, "bottom is clamped");
}

/// `refresh_mcp_status`: the idle-tick MCP poll must (a) update the HUD rows
/// and request a redraw when the live statuses change, (b) self-gate to at
/// most one poll per second, and (c) keep the last rows when the poller
/// reports no MCP state. Guards the "sidebar stuck on `discovering` while the
/// prompt sits idle" regression — the full HUD snapshot is otherwise rebuilt
/// only at action boundaries, so background discovery finishing at idle never
/// reached the screen.
#[test]
fn refresh_mcp_status_updates_rows_and_gates_cadence() {
    use crate::tui::hud::McpHudStatus;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    let mut app = test_app();
    let t0 = Instant::now();
    assert!(
        !app.refresh_mcp_status(t0),
        "no poller installed must be a no-op"
    );

    let rows = Arc::new(Mutex::new(Some(vec![
        McpHudStatus::discovering("ctx7").encode(),
    ])));
    let source = Arc::clone(&rows);
    app.set_mcp_status_poller(Box::new(move || source.lock().unwrap().clone()));

    assert!(
        app.refresh_mcp_status(t0),
        "first poll must install the discovering row and request a redraw"
    );
    assert_eq!(
        app.hud_state.mcp_servers,
        vec![McpHudStatus::discovering("ctx7").encode()]
    );

    *rows.lock().unwrap() = Some(vec![McpHudStatus::ready("ctx7").encode()]);
    assert!(
        !app.refresh_mcp_status(t0 + Duration::from_millis(500)),
        "sub-second re-polls must be held by the cadence gate"
    );

    assert!(
        app.refresh_mcp_status(t0 + Duration::from_secs(1)),
        "past the gate the discovering→ready flip must land and redraw"
    );
    assert_eq!(
        app.hud_state.mcp_servers,
        vec![McpHudStatus::ready("ctx7").encode()]
    );

    assert!(
        !app.refresh_mcp_status(t0 + Duration::from_secs(2)),
        "unchanged rows must not request a redraw"
    );

    *rows.lock().unwrap() = None;
    assert!(
        !app.refresh_mcp_status(t0 + Duration::from_secs(3)),
        "a session without MCP state must not clear the rows"
    );
    assert_eq!(
        app.hud_state.mcp_servers,
        vec![McpHudStatus::ready("ctx7").encode()],
        "last known rows survive a None poll"
    );
}

#[test]
fn inline_viewport_emits_settled_turn_and_keeps_live_composer() {
    let mut app = test_app();
    let answer = format!(
        "settled inline answer {}END-OF-INLINE-ANSWER",
        "tail ".repeat(80)
    );
    app.set_terminal_mode(TerminalMode::Inline);
    app.enable_input();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(1),
        text: "inline prompt".to_string(),
    });
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(2),
        text: answer.clone(),
        done: true,
    });
    app.end_turn();

    let backend = TestBackend::new(72, INLINE_VIEWPORT_HEIGHT);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )
    .expect("inline test terminal");
    app.draw(&mut terminal).expect("inline draw");

    let scrollback = terminal.backend().scrollback();
    let scrollback_text = (0..scrollback.area.height).fold(String::new(), |mut out, y| {
        for x in 0..scrollback.area.width {
            out.push_str(scrollback[(x, y)].symbol());
        }
        out.push('\n');
        out
    });
    assert!(scrollback_text.contains("inline prompt"), "{scrollback_text}");
    assert!(
        scrollback_text.contains("settled inline answer")
            && scrollback_text.contains("END-OF-INLINE-ANSWER"),
        "{scrollback_text}"
    );

    let viewport = dump_all(&terminal, 72, INLINE_VIEWPORT_HEIGHT);
    assert!(!viewport.contains("settled inline answer"), "{viewport}");
    let regions = app.regions.expect("inline regions");
    assert_eq!(regions.sidebar_width, 0);
    assert!(regions.input.height > 0, "composer remains visible");
    assert!(regions.hud.height > 0, "HUD remains visible");
}

#[test]
fn inline_repeated_insert_redraw_handles_a_nonzero_viewport_origin() {
    let mut app = test_app();
    app.set_terminal_mode(TerminalMode::Inline);
    app.enable_input();

    let mut backend = TestBackend::new(110, 32);
    backend
        .set_cursor_position(Position::new(0, 10))
        .expect("position inline viewport below existing terminal output");
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )
    .expect("inline test terminal");

    app.draw(&mut terminal).expect("initial inline draw");
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(1),
        text: "short prompt".to_string(),
    });
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(2),
        text: "short answer".to_string(),
        done: true,
    });
    app.end_turn();
    app.draw(&mut terminal).expect("settled turn draw");

    // A late report can cause another ordinary emission/redraw before teardown.
    // Both `insert_before` transitions must preserve the absolute viewport
    // origin. The separate shutdown test verifies teardown does not redraw.
    app.push_block(RenderBlock::System {
        id: BlockId(3),
        level: SystemLevel::Info,
        text: "session ending".to_string(),
    });
    app.finalize_inline_transcript();
    app.draw(&mut terminal).expect("shutdown redraw");
}

#[test]
fn inline_slash_hint_draw_stays_inside_a_nonzero_viewport() {
    let mut app = test_app();
    app.set_terminal_mode(TerminalMode::Inline);
    app.enable_input();
    app.set_input_text("/");

    let mut backend = TestBackend::new(110, 32);
    backend
        .set_cursor_position(Position::new(0, 20))
        .expect("position inline viewport at the bottom of the terminal");
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )
    .expect("inline test terminal");

    let frame_area = terminal.get_frame().area();
    assert_eq!(frame_area, Rect::new(0, 20, 110, INLINE_VIEWPORT_HEIGHT));
    app.draw(&mut terminal)
        .expect("slash hint must stay inside the inline frame buffer");
    let input_area = app.regions.expect("inline layout regions").input;
    let popup_area = app
        .slash_hint_popup_rect()
        .expect("bounded slash hint popup");
    let desired_height = u16::try_from(app.slash_hint_suggestion_count())
        .unwrap_or(u16::MAX)
        .saturating_add(2);
    assert_eq!(popup_area.y, frame_area.y);
    assert!(popup_area.height < desired_height);
    assert!(popup_area.bottom() <= frame_area.bottom());
    assert!(popup_area.bottom() <= input_area.y);
}

#[test]
fn inline_mention_hint_draw_stays_inside_a_nonzero_viewport() {
    let mut app = test_app();
    app.set_terminal_mode(TerminalMode::Inline);
    app.enable_input();
    app.set_input_text("see @");
    app.workspace_files = (0..10).map(|index| format!("src/file-{index}.rs")).collect();

    let mut backend = TestBackend::new(110, 32);
    backend
        .set_cursor_position(Position::new(0, 20))
        .expect("position inline viewport at the bottom of the terminal");
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )
    .expect("inline test terminal");

    let frame_area = terminal.get_frame().area();
    assert_eq!(frame_area, Rect::new(0, 20, 110, INLINE_VIEWPORT_HEIGHT));
    app.draw(&mut terminal)
        .expect("mention hint must stay inside the inline frame buffer");
}

#[test]
fn inline_oversized_anchored_surface_renders_fullscreen_notice() {
    let mut app = test_app();
    app.set_terminal_mode(TerminalMode::Inline);
    app.enable_input();
    app.open_custom_provider_modal();

    let backend = TestBackend::new(72, INLINE_VIEWPORT_HEIGHT);
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
        },
    )
    .expect("inline test terminal");
    app.draw(&mut terminal).expect("inline draw");

    let viewport = dump_all(&terminal, 72, INLINE_VIEWPORT_HEIGHT);
    assert!(viewport.contains("needs full-screen mode"), "{viewport}");
    assert!(!viewport.contains("Base URL"), "oversized form was clipped: {viewport}");
}

fn remote_onboarding_view(running: bool, pending_pairs: Vec<RemotePendingPair>) -> RemoteOnboardingView {
    RemoteOnboardingView {
        running,
        url: running.then(|| "https://zo.tailnet.ts.net".to_string()),
        device_count: 1,
        pending_count: pending_pairs.len(),
        controller: Some("Work phone".to_string()),
        turn_state: "idle".to_string(),
        pending_pairs,
    }
}

#[test]
fn remote_onboarding_stopped_enter_submits_start() {
    let mut app = test_app();
    app.open_remote_onboarding_modal(remote_onboarding_view(false, Vec::new()));

    assert_eq!(app.mode(), AppMode::ModalRemoteOnboarding);
    assert_eq!(
        app.handle_key(press(KeyCode::Enter)).expect("Enter handled"),
        AppAction::Submit("/remote start".to_string())
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn remote_onboarding_running_actions_map_to_existing_commands() {
    let expected = [
        "/remote qr",
        "/remote status",
        "/remote rotate",
        "/remote stop",
    ];
    for (index, command) in expected.into_iter().enumerate() {
        let mut app = test_app();
        app.open_remote_onboarding_modal(remote_onboarding_view(true, Vec::new()));
        for _ in 0..index {
            let _ = app.handle_key(press(KeyCode::Char('j'))).expect("j handled");
        }
        assert_eq!(
            app.handle_key(press(KeyCode::Enter)).expect("Enter handled"),
            AppAction::Submit(command.to_string()),
            "action row {index} must keep the existing text command path"
        );
    }
}

#[test]
fn remote_onboarding_pending_rows_approve_and_deny_the_correct_code() {
    let pairs = vec![
        RemotePendingPair {
            device_name: "Phone".to_string(),
            comparison_code: "AAA-111".to_string(),
        },
        RemotePendingPair {
            device_name: "Tablet".to_string(),
            comparison_code: "BBB-222".to_string(),
        },
    ];
    for (index, command) in [
        (0, "/remote approve AAA-111"),
        (1, "/remote deny AAA-111"),
        (2, "/remote approve BBB-222"),
        (3, "/remote deny BBB-222"),
    ] {
        let mut app = test_app();
        app.open_remote_onboarding_modal(remote_onboarding_view(true, pairs.clone()));
        for _ in 0..index {
            let _ = app.handle_key(press(KeyCode::Down)).expect("Down handled");
        }
        assert_eq!(
            app.handle_key(press(KeyCode::Enter)).expect("Enter handled"),
            AppAction::Submit(command.to_string())
        );
    }
}

#[test]
fn remote_onboarding_escape_closes_without_submit() {
    let mut app = test_app();
    app.open_remote_onboarding_modal(remote_onboarding_view(false, Vec::new()));

    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).expect("Esc handled"),
        AppAction::None
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn remote_onboarding_navigation_supports_arrows_jk_and_wheel() {
    let mut app = test_app();
    app.open_remote_onboarding_modal(remote_onboarding_view(true, Vec::new()));
    let _ = app.handle_key(press(KeyCode::Char('j'))).expect("j handled");
    let _ = app.handle_key(press(KeyCode::Char('k'))).expect("k handled");
    let _ = app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    });

    assert_eq!(
        app.handle_key(press(KeyCode::Enter)).expect("Enter handled"),
        AppAction::Submit("/remote stop".to_string()),
        "the unified modal wheel seam moves the three-row wheel stride"
    );
}

#[test]
fn spectator_replace_is_ordered_before_later_frame_and_acknowledged() {
    use std::collections::VecDeque;

    let mut app = test_app();
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "stale".to_string(),
        done: true,
    });
    let (ack, mut ack_rx) = oneshot::channel();
    app.process_spectator_event(SpectatorEvent::Replace {
        blocks: vec![RenderBlock::TextDelta {
            id: BlockId(2),
            text: "snapshot".to_string(),
            done: true,
        }],
        post_boundary: VecDeque::from([RenderBlock::TextDelta {
            id: BlockId(3),
            text: "tail".to_string(),
            done: true,
        }]),
        next_seq: 4,
        ack,
    });
    assert_eq!(ack_rx.try_recv(), Ok(AckVerdict::Applied), "replace ACK follows transcript clear/replay");
    app.process_spectator_event(SpectatorEvent::Frame {
        frame_seq: 4,
        block: RenderBlock::TextDelta {
            id: BlockId(4),
            text: "later".to_string(),
            done: true,
        },
    });
    assert_eq!(visible_text_deltas(&app), "snapshottaillater");
}

#[test]
fn spectator_frame_floor_rejects_stale_frame_and_keeps_current_frame() {
    let mut app = test_app();
    app.advance_spectator_floor(5);

    app.process_spectator_event(SpectatorEvent::Frame {
        frame_seq: 3,
        block: RenderBlock::TextDelta {
            id: BlockId(1),
            text: "stale".to_string(),
            done: true,
        },
    });
    app.process_spectator_event(SpectatorEvent::Frame {
        frame_seq: 5,
        block: RenderBlock::TextDelta {
            id: BlockId(2),
            text: "current".to_string(),
            done: true,
        },
    });
    assert_eq!(visible_text_deltas(&app), "current");
}

#[test]
fn own_turn_floor_rejects_stale_spectator_replace_without_clearing_transcript() {
    use std::collections::VecDeque;

    let mut app = test_app();
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "own turn".to_string(),
        done: true,
    });
    app.advance_spectator_floor(42);
    let (ack, mut ack_rx) = oneshot::channel();
    app.process_spectator_event(SpectatorEvent::Replace {
        blocks: vec![RenderBlock::TextDelta {
            id: BlockId(2),
            text: "stale snapshot".to_string(),
            done: true,
        }],
        post_boundary: VecDeque::new(),
        next_seq: 41,
        ack,
    });
    assert_eq!(ack_rx.try_recv(), Ok(AckVerdict::Stale));
    assert_eq!(visible_text_deltas(&app), "own turn");
}

#[test]
fn drain_ready_blocks_ingests_large_sse_burst_before_one_draw() {
    // The draw loop is frame-gated elsewhere; this ingest cap must be high
    // enough that a provider burst does not fill the bounded render channel and
    // backpressure the SSE parser after the old 16-block quantum.
    let (block_tx, block_rx) = mpsc::channel::<RenderBlock>(300);
    let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
    let mut app = App::new(Theme::no_color(), block_rx, cmd_tx);

    for i in 0..128u64 {
        block_tx
            .try_send(RenderBlock::TextDelta {
                id: BlockId(i),
                text: format!("token-{i}"),
                done: true,
            })
            .expect("burst fits in test channel");
    }

    let drained = app.drain_ready_blocks();
    assert_eq!(
        drained, 128,
        "a single pre-draw ingest should absorb a realistic burst, not stop at the old 16-block cap"
    );
    assert!(drained > 16);
}

fn startup_screen_for_shortcut() -> StartupScreen {
    StartupScreen {
        version: "0.1.0".to_string(),
        model: "claude-opus-4-8".to_string(),
        permissions: "workspace-write".to_string(),
        branch: "main".to_string(),
        workspace: "zo".to_string(),
        directory: PathBuf::from("/tmp/zo"),
        project_root: Some(PathBuf::from("/tmp/zo")),
        session_id: "session-1234567890".to_string(),
        autosave_path: PathBuf::from("/tmp/session.jsonl"),
        startup_ms: Some(1297),
        memory_mb: Some(42.0),
        auth: StartupAuthState::default(),
        recent_sessions: Vec::new(),
    }
}

fn press_alt_char(app: &mut App, ch: char) -> AppAction {
    app.handle_key(KeyEvent {
        code: KeyCode::Char(ch),
        modifiers: KeyModifiers::ALT,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
    .expect("alt startup shortcut handled")
}

/// A fresh, per-test-isolated prompt history backed by a unique temp path so
/// history-nav tests never see entries leaked by other tests sharing the
/// `test_app` default history file.
fn isolated_history() -> crate::tui::history::History {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!(
        "zo-test-history-{}-{nanos}.jsonl",
        std::process::id()
    ));
    crate::tui::history::History::load(path).expect("isolated history")
}

/// Like [`test_app`] but returns the agent-command receiver so tests can
/// assert on commands the app emits (e.g. mid-turn steering).
fn test_app_with_cmd() -> (App, mpsc::Receiver<AgentCommand>) {
    let (_block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
    let (cmd_tx, cmd_rx) = mpsc::channel::<AgentCommand>(16);
    (App::new(Theme::no_color(), block_rx, cmd_tx), cmd_rx)
}

fn usage_dashboard_modal_for_test() -> UsageDashboardModal {
    UsageDashboardModal::new(UsageDashboardSnapshot::from_session(
        "gpt-5.5",
        TokenUsage {
            input_tokens: 10_000,
            output_tokens: 2_000,
            cache_creation_input_tokens: 500,
            cache_read_input_tokens: 20_000,
        },
        2,
    ))
}

#[test]
fn open_usage_dashboard_enters_modal_usage_and_esc_closes() {
    let mut app = test_app();
    app.open_usage_dashboard_modal(usage_dashboard_modal_for_test());
    assert_eq!(app.mode(), AppMode::ModalUsage);
    assert!(app.modals.usage_dashboard.is_some());

    let action = app.handle_key(press(KeyCode::Esc)).expect("esc handled");
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.modals.usage_dashboard.is_none());
}

fn visible_text_deltas(app: &App) -> String {
    app.transcript
        .blocks()
        .iter()
        .filter_map(|block| match block {
            RenderBlock::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn paced_small_stream_delta_lands_whole_without_typewriter_stutter() {
    // Small provider deltas are already below the perceptual chunk size. They
    // should land whole on arrival; otherwise the TUI feels like a slow
    // typewriter (`1 char → pause → 3 chars → pause`) instead of web-chat
    // streaming.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let full = "안녕하세요 반갑습니다";
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(1), full.to_string(), false);

    assert_eq!(
        visible_text_deltas(&app),
        full,
        "small deltas should reveal whole on the arrival frame"
    );

    // A later small delta for the same block appends in place — no second block
    // and no artificial starter-sized drip.
    let t_end = t0 + std::time::Duration::from_millis(33);
    app.buffer_paced_at(t_end, BlockId(1), " 또 만나요".to_string(), true);
    assert_eq!(visible_text_deltas(&app), "안녕하세요 반갑습니다 또 만나요");
    let text_blocks = app
        .transcript()
        .blocks()
        .iter()
        .filter(|b| matches!(b, RenderBlock::TextDelta { .. }))
        .count();
    assert_eq!(text_blocks, 1, "same-id deltas append into one block");
}

#[test]
fn paced_small_continuation_clump_spreads_across_frames_not_landed_whole() {
    // Regression for the <= IMMEDIATE_CHARS fast path: after a small opening
    // paints immediately, a later tiny Claude token clump for the same block
    // must not be revealed whole on the next frame. Otherwise short bursts still
    // read as clump → pause → clump even though arrivals no longer drip.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(43), "Start ".to_string(), false);
    let opened = visible_text_deltas(&app).chars().count();

    for _ in 0..5u64 {
        app.buffer_paced_at(t0, BlockId(43), "tok ".to_string(), false);
    }
    assert_eq!(
        visible_text_deltas(&app).chars().count(),
        opened,
        "same-instant small continuation burst should accumulate on arrival"
    );

    let frame = std::time::Duration::from_millis(33);
    app.drip_stream_at(t0 + frame, Some(frame));
    let after_one_frame = visible_text_deltas(&app).chars().count();
    let total = opened + 5 * "tok ".chars().count();
    assert!(
        after_one_frame > opened,
        "first frame should reveal part of the small continuation clump"
    );
    assert!(
        after_one_frame < total,
        "first frame must not dump the whole <=24-char continuation clump ({after_one_frame}/{total})"
    );

    app.drip_stream_at(t0 + frame * 2, Some(frame));
    let after_two = visible_text_deltas(&app).chars().count();
    assert!(
        after_two > after_one_frame && after_two <= total,
        "the small clump keeps typing in across frames (was {after_two}/{total})"
    );
    // Fully settled within a few interactive frames — metered out smoothly, not
    // dumped whole on one frame.
    for i in 3..=4 {
        app.drip_stream_at(t0 + frame * i, Some(frame));
    }
    assert_eq!(
        visible_text_deltas(&app).chars().count(),
        total,
        "the small clump settles within a few frames"
    );
}

#[test]
fn paced_token_burst_continuation_spreads_across_frames_not_clumped() {
    // The Claude-only stutter: Claude streams genuine per-token deltas, and the
    // network delivers them in clumps (many tiny tokens in one read, then a
    // pause). Each token is individually below the land-whole threshold, so if
    // every arrival dripped immediately the on-screen reveal mirrored that bursty
    // delivery (clump → pause → clump) — the "안 부드러움" the user saw only with
    // Claude. A coarse-chunk provider (OpenAI-compat / Gemini) never tripped this
    // because its chunks are large enough to engage the pacer's smoothing.
    //
    // Continuation deltas must therefore ACCUMULATE on arrival (only the opening
    // delta paints immediately) and be metered out by the frame-driven drip, so a
    // burst spreads across the following frames at a steady cadence.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let t0 = std::time::Instant::now();
    // Opening delta: a small phrase that lands whole for a low-latency first
    // paint.
    app.buffer_paced_at(t0, BlockId(42), "Let me ".to_string(), false);
    let opened = visible_text_deltas(&app).chars().count();
    assert_eq!(opened, "Let me ".chars().count(), "opening delta paints at once");

    // A burst of 100 tiny tokens (4 chars each = 400 chars) all arrive in the
    // SAME frame instant, mimicking one network read of Claude per-token deltas
    // during a sustained answer.
    for _ in 0..100u64 {
        app.buffer_paced_at(t0, BlockId(42), "tok ".to_string(), false);
    }
    // None of the burst is revealed yet on its arrival instant — it accumulated.
    let after_burst = visible_text_deltas(&app).chars().count();
    assert_eq!(
        after_burst, opened,
        "a same-instant continuation burst must accumulate, not clump onto the arrival frame"
    );

    // The frame-driven drip now meters the backlog out across frames at a steady
    // cadence — each frame reveals a bounded slice, never the whole burst at once.
    let frame = std::time::Duration::from_millis(33);
    let total = opened + 100 * "tok ".chars().count();
    let mut prev = after_burst;
    let mut max_per_frame = 0usize;
    let mut frames_to_drain = 0;
    for i in 1..=8 {
        app.drip_stream_at(t0 + frame * i, Some(frame));
        let now = visible_text_deltas(&app).chars().count();
        max_per_frame = max_per_frame.max(now.saturating_sub(prev));
        prev = now;
        if now >= total {
            frames_to_drain = i;
            break;
        }
    }
    assert_eq!(prev, total, "the burst fully reveals within a few frames");
    assert!(
        frames_to_drain >= 2,
        "the burst is spread across multiple frames, not dumped in one: drained in {frames_to_drain}"
    );
    assert!(
        max_per_frame < (total - opened),
        "no single frame reveals the whole burst (max {max_per_frame} of {})",
        total - opened
    );
}

#[test]
fn measure_real_claude_arrival_cadence_reveals_per_frame() {
    // Reproduce the ACTUAL delivery pattern captured in delta-trace.log: Claude
    // deltas arrive ~16-31 chars each, ~480ms APART (not all in one instant like
    // `paced_token_burst_continuation_spreads_across_frames_not_clumped`). Drive
    // a 30fps frame drip in the gap between arrivals, exactly like the live
    // render tick, and record how many chars become visible on each drawn frame.
    // If a whole 16-31 char delta lands on its first post-arrival frame (instead
    // of spreading across the ~14 frames its 480ms gap affords), that is the
    // "뭉텅이" the user sees. This is a MEASUREMENT (prints), not a hard gate.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    let t0 = std::time::Instant::now();
    let frame = std::time::Duration::from_millis(33);

    // Opening delta paints immediately (low-latency first paint).
    app.buffer_paced_at(t0, BlockId(7), "Let me start. ".to_string(), false);

    // Real captured sizes (chars) and gaps (ms) from delta-trace.log.
    let arrivals = [
        (21usize, 443u64),
        (24, 547),
        (28, 483),
        (31, 500),
        (21, 757),
        (16, 385),
        (30, 478),
    ];

    let mut now = t0;
    let mut prev_visible = visible_text_deltas(&app).chars().count();
    let mut worst_single_frame = 0usize;
    let mut frames_with_reveal = 0usize;
    let mut total_frames = 0usize;

    for (chars, gap_ms) in arrivals {
        // The continuation delta arrives.
        now += std::time::Duration::from_millis(gap_ms);
        app.buffer_paced_at(now, BlockId(7), "y".repeat(chars), false);
        // Then the render tick drips across the whole inter-arrival gap.
        let frames = gap_ms / 33;
        for _ in 0..frames {
            now += frame;
            app.drip_stream_at(now, Some(frame));
            let vis = visible_text_deltas(&app).chars().count();
            let revealed = vis.saturating_sub(prev_visible);
            prev_visible = vis;
            total_frames += 1;
            if revealed > 0 {
                frames_with_reveal += 1;
                worst_single_frame = worst_single_frame.max(revealed);
            }
        }
    }

    eprintln!(
        "[pacer-cadence] worst single-frame reveal={worst_single_frame} chars; \
         frames_with_reveal={frames_with_reveal}/{total_frames}; \
         (a delta revealed whole in one frame == 뭉텅이; spread across frames == smooth)"
    );

    // Regression gate: with the adaptive inter-arrival spreading, a clumpy
    // Claude delivery must NOT land each ~20-char delta whole on one frame.
    // Before the fix, worst_single_frame == the delta size (~17-31) and only ~14
    // frames revealed anything; after it, each delta types across many frames.
    assert!(
        worst_single_frame <= 8,
        "a clumpy 16-31 char delivery must type in across frames, not land whole \
         (worst single-frame reveal was {worst_single_frame} chars — that is 뭉텅이)"
    );
    assert!(
        frames_with_reveal >= total_frames / 3,
        "the reveal must be spread across many frames, not concentrated in a few \
         ({frames_with_reveal}/{total_frames})"
    );
}

#[test]
fn paced_threshold_boundaries_are_fast_and_not_chunky() {
    // Lock the 24-char (phrase) boundary: true word-sized deltas land whole,
    // while anything larger is split by a frame so it types in instead of
    // slamming into the transcript as a provider-sized chunk.
    let frame = std::time::Duration::from_millis(33);
    for (len, expected_first) in [(24usize, 24usize), (25, 24), (48, 24)] {
        let mut app = test_app();
        app.begin_turn_with_generation(0);

        let chunk: String = "b".repeat(len);
        let t0 = std::time::Instant::now();
        app.buffer_paced_at(t0, BlockId(len as u64), chunk.clone(), false);

        let first_chars = visible_text_deltas(&app).chars().count();
        assert_eq!(
            first_chars, expected_first,
            "{len}-char chunk first-frame reveal should match the pacing boundary"
        );

        app.drip_stream_at(t0 + frame, Some(frame));
        let after_one = visible_text_deltas(&app).chars().count();
        if len == expected_first {
            // A delta at the land-whole boundary is already fully revealed on
            // its first paint — nothing left to type in.
            assert_eq!(after_one, len, "{len}-char chunk lands whole at once");
        } else {
            assert!(
                after_one > expected_first && after_one <= len,
                "{len}-char chunk should keep typing in after the first frame \
                 (was {after_one}/{len})"
            );
        }
        // Settles fully within a small, bounded number of interactive frames —
        // a smooth type-in across 2-3 frames, not a one-frame dump (the old
        // cadence) and not a slow trailing reveal.
        for i in 2..=4 {
            app.drip_stream_at(t0 + frame * i, Some(frame));
        }
        assert_eq!(
            visible_text_deltas(&app),
            chunk,
            "{len}-char chunk should be fully visible within a few frames"
        );
    }
}

#[test]
fn paced_mid_sized_stream_delta_is_smoothed_without_added_latency() {
    // Provider chunks around a sentence can feel like a "뭉탱이" when painted in
    // one frame. They should open immediately with a phrase-sized slice, then
    // finish on the next frame instead of being held back like a slow reveal.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let chunk: String = "m".repeat(40);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(2), chunk.clone(), false);

    let first_chars = visible_text_deltas(&app).chars().count();
    assert_eq!(
        first_chars, 24,
        "a 40-char open-stream chunk should not land whole on arrival"
    );

    let frame = std::time::Duration::from_millis(33);
    app.drip_stream_at(t0 + frame, Some(frame));
    let after_one = visible_text_deltas(&app).chars().count();
    assert!(
        after_one > 24 && after_one <= 40,
        "a mid-sized chunk should keep typing in after the first frame rather \
         than dumping whole or stalling (was {after_one}/40)"
    );
    for i in 2..=4 {
        app.drip_stream_at(t0 + frame * i, Some(frame));
    }
    assert_eq!(
        visible_text_deltas(&app),
        chunk,
        "mid-sized chunks should finish within a few frames, not trail behind"
    );
}

#[test]
fn paced_small_done_tail_lands_immediately_without_a_trailing_drip() {
    // A small terminal delta is already below the perceptual chunk threshold.
    // It should still land whole and seal immediately, preserving the fast path
    // for short phrase-sized endings.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let full = "Done: a small ending.";
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(7), full.to_string(), true);

    assert_eq!(
        visible_text_deltas(&app),
        full,
        "a done delta should settle on its arrival frame"
    );
    assert!(
        !app.stream_pending(),
        "the pacer is dropped once the done buffer drains"
    );
    let sealed = app
        .transcript()
        .blocks()
        .iter()
        .any(|b| matches!(b, RenderBlock::TextDelta { done: true, .. }));
    assert!(sealed, "the block is sealed done=true so its caret flips off");
}

#[test]
fn paced_barely_large_done_tail_waits_for_finish_window() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let full: String = "B".repeat(25);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(6), full.clone(), true);

    assert_eq!(
        visible_text_deltas(&app).chars().count(),
        24,
        "IMMEDIATE_CHARS + 1 done chunks should not be promoted to full flush on arrival"
    );
    assert!(app.stream_pending());

    let frame = std::time::Duration::from_millis(33);
    app.drip_stream_at(t0 + frame, Some(frame));
    assert_eq!(visible_text_deltas(&app), full);
    assert!(!app.stream_pending());
}

#[test]
fn paced_large_done_tail_uses_finish_window_instead_of_one_frame_flush() {
    // Some providers can deliver a large final burst together with `done=true`.
    // Do not slam that entire final burst into the transcript on the arrival
    // frame; reveal a phrase immediately, then finish on the next animation
    // frame through the short done window.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let full: String = "D".repeat(500);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(8), full.clone(), true);

    let first_chars = visible_text_deltas(&app).chars().count();
    assert_eq!(
        first_chars, 24,
        "large done chunks should not flush whole on the arrival frame"
    );
    assert!(
        app.stream_pending(),
        "large done chunks keep a short finish-window tail after the first frame"
    );

    let frame = std::time::Duration::from_millis(33);
    app.drip_stream_at(t0 + frame, Some(frame));
    assert_eq!(visible_text_deltas(&app), full);
    assert!(
        !app.stream_pending(),
        "the finish window should settle the final burst by the next frame"
    );
    let sealed = app
        .transcript()
        .blocks()
        .iter()
        .any(|b| matches!(b, RenderBlock::TextDelta { done: true, .. }));
    assert!(sealed, "large done chunk still seals the block promptly");
}

#[test]
fn paced_large_done_tail_preserves_utf8_boundaries_while_finishing() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let source = "한국어 🌟 emoji 混在 テスト 🚀 end ".repeat(8);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(10), source.clone(), true);

    let first = visible_text_deltas(&app);
    assert!(
        source.starts_with(&first),
        "first large-done reveal must be a char-boundary prefix"
    );
    assert!(
        first.chars().count() < source.chars().count(),
        "large UTF-8 done chunks should be smoothed, not flushed whole"
    );

    let frame = std::time::Duration::from_millis(33);
    app.drip_stream_at(t0 + frame, Some(frame));
    assert_eq!(visible_text_deltas(&app), source);
}

#[test]
fn paced_stream_does_not_dump_a_whole_burst_in_one_frame() {
    // The anti-"뭉탱이" invariant: a large burst landing in a single arrival must
    // NOT all paint on the first frame. It is spread across subsequent frames so
    // it reads as a fast type-in instead of a chunk slamming onto the screen.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let burst: String = "x".repeat(500);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(9), burst.clone(), false);

    let first_chars = visible_text_deltas(&app).chars().count();
    assert!(
        (16..=40).contains(&first_chars),
        "a 500-char burst should show a phrase-sized first chunk, not glyph-by-glyph or all at once: showed {first_chars}"
    );
}

#[test]
fn paced_big_content_is_not_rate_capped_like_the_old_reveal() {
    // The old controller capped sustained reveal at 1200 c/s, so big/fast content
    // was throttled below generation speed. The pacer's rate is backlog/window,
    // so a large backlog drains proportionally faster: a 3000-char block is fully
    // revealed within ~3 frames (~99 ms). At the old 1200 c/s cap, 99 ms would
    // surface only ~119 chars — proving there is no ceiling.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let big: String = "y".repeat(3000);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(11), big.clone(), false);

    let frame = std::time::Duration::from_millis(33);
    for i in 1..=3 {
        app.drip_stream_at(t0 + frame * i, Some(frame));
    }
    let revealed = visible_text_deltas(&app).chars().count();
    assert!(
        revealed > 1200,
        "no rate ceiling: a big backlog drains fast (revealed {revealed} in ~99 ms, \
         far above the old 1200 c/s cap)"
    );
}

#[test]
fn paced_large_backlog_finishes_within_interactive_latency() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let big: String = "z".repeat(3000);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(12), big.clone(), false);

    let frame = std::time::Duration::from_millis(33);
    // The bulk drains fast against the ease-out `backlog / WINDOW` rate (no rate
    // ceiling): within ~3 frames the vast majority is already on screen.
    for i in 1..=3 {
        app.drip_stream_at(t0 + frame * i, Some(frame));
    }
    assert!(
        visible_text_deltas(&app).chars().count() > 2600,
        "the bulk of a large backlog drains fast, not throttled (was {}/3000)",
        visible_text_deltas(&app).chars().count()
    );

    // The final phrase-sized tail is then metered out smoothly rather than
    // dumped, so it fully settles within a bounded interactive window (~12
    // frames ≈ 0.4 s) — never a slow trailing typewriter.
    for i in 4..=12 {
        app.drip_stream_at(t0 + frame * i, Some(frame));
    }
    assert_eq!(
        visible_text_deltas(&app).chars().count(),
        big.chars().count(),
        "large provider backlogs settle within interactive latency"
    );
}

#[test]
fn paced_reveal_never_splits_a_multibyte_char() {
    // Every intermediate reveal must land on a UTF-8 char boundary so a CJK glyph
    // or emoji is never torn in half (which would panic on the String slice or
    // paint a replacement box). Drive frame-by-frame and assert each visible
    // snapshot is a valid char-boundary prefix of the source.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let source = "한국어 🌟 emoji 混在 テスト 🚀 end ".repeat(8);
    let t0 = std::time::Instant::now();
    app.buffer_paced_at(t0, BlockId(13), source.clone(), true);

    let frame = std::time::Duration::from_millis(33);
    for i in 0..10 {
        let visible = visible_text_deltas(&app);
        assert!(
            source.starts_with(&visible),
            "frame {i}: visible {visible:?} must be a char-boundary prefix of the source"
        );
        app.drip_stream_at(t0 + frame * (i + 1), Some(frame));
    }
    assert_eq!(visible_text_deltas(&app), source, "full text reveals");
}

/// Measurement harness (not a pass/fail gate): drives the live pacer with three
/// realistic provider delta regimes and prints, per regime, the added tail
/// latency and the largest single-frame reveal ("뭉탱이" size). Run with:
///   cargo test -p zo-cli --lib -- --ignored --nocapture `measure_pacer`
#[test]
#[ignore = "measurement: run with --ignored --nocapture to print the latency table"]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn measure_pacer_latency_profile() {
    use std::time::{Duration, Instant};

    const DRIP_MS: u64 = 16;

    struct Regime {
        label: &'static str,
        gap_ms: u64,
        chars: usize,
        n: usize,
    }

    // (gap, chars, n): the size+cadence of provider `content_block_delta`s.
    let regimes = [
        Regime { label: "tokens (small <=24)", gap_ms: 25, chars: 14, n: 30 },
        Regime { label: "phrase chunks 25-64", gap_ms: 40, chars: 40, n: 12 },
        Regime { label: "sentence bursts", gap_ms: 60, chars: 70, n: 12 },
        Regime { label: "big final dump x1", gap_ms: 0, chars: 2000, n: 1 },
    ];

    // Production drives the drip on the 16 ms stream gate during active
    // streaming (and the 33 ms tick in gaps). 16 ms = the smoothest cadence the
    // gate allows, so this is the faithful active-stream drive.
    println!(
        "\n{:<22} {:>6} {:>14} {:>14} {:>15}",
        "regime", "deltas", "max tail lat", "avg tail lat", "max chars/frame"
    );
    println!("{}", "-".repeat(75));

    for r in regimes {
        let mut app = test_app();
        app.begin_turn_with_generation(0);
        let t0 = Instant::now();

        let arrivals: Vec<(u64, usize, bool)> = (0..r.n)
            .map(|i| (i as u64 * r.gap_ms, r.chars, i + 1 == r.n))
            .collect();
        // Cumulative visible-char target for each delta's *tail*.
        let cum: Vec<usize> = arrivals
            .iter()
            .scan(0usize, |acc, &(_, c, _)| {
                *acc += c;
                Some(*acc)
            })
            .collect();
        let arr_t: Vec<u64> = arrivals.iter().map(|&(t, _, _)| t).collect();
        let mut tail_at: Vec<Option<u64>> = vec![None; r.n];

        let total_ms = r.n as u64 * r.gap_ms + 500;
        let mut next_drip = DRIP_MS;
        let mut ai = 0usize;
        let mut prev_visible = 0usize;
        let mut max_frame = 0usize;

        let mut now_ms = 0u64;
        while now_ms <= total_ms {
            let mut acted = false;
            while ai < arrivals.len() && arrivals[ai].0 == now_ms {
                let (_, c, done) = arrivals[ai];
                app.buffer_paced_at(
                    t0 + Duration::from_millis(now_ms),
                    BlockId(1),
                    "x".repeat(c),
                    done,
                );
                ai += 1;
                acted = true;
            }
            if now_ms == next_drip {
                app.drip_stream_at(t0 + Duration::from_millis(now_ms), None);
                next_drip += DRIP_MS;
                acted = true;
            }
            if acted {
                let vis = visible_text_deltas(&app).chars().count();
                max_frame = max_frame.max(vis.saturating_sub(prev_visible));
                prev_visible = vis;
                for (i, &c) in cum.iter().enumerate() {
                    if tail_at[i].is_none() && vis >= c {
                        tail_at[i] = Some(now_ms);
                    }
                }
            }
            now_ms += 1;
        }
        // Seal and drain any open tail so the last delta's tail time is captured.
        app.finish_stream();
        for k in 1..=30u64 {
            let t = total_ms + k * DRIP_MS;
            app.drip_stream_at(t0 + Duration::from_millis(t), None);
            let vis = visible_text_deltas(&app).chars().count();
            max_frame = max_frame.max(vis.saturating_sub(prev_visible));
            prev_visible = vis;
            for (i, &c) in cum.iter().enumerate() {
                if tail_at[i].is_none() && vis >= c {
                    tail_at[i] = Some(t);
                }
            }
        }

        let lats: Vec<u64> = (0..r.n)
            .map(|i| tail_at[i].unwrap_or(total_ms).saturating_sub(arr_t[i]))
            .collect();
        let max_lat = lats.iter().copied().max().unwrap_or(0);
        let avg_lat = lats.iter().sum::<u64>() / r.n as u64;

        println!(
            "{:<22} {:>6} {:>11} ms {:>11} ms {:>15}",
            r.label, r.n, max_lat, avg_lat, max_frame
        );
    }
    println!();
}

#[test]
fn passthrough_reveal_keeps_arrival_order_with_interleaved_tool_block() {
    // A tool call that arrives mid-stream flushes the paced prose tail first, so
    // the transcript keeps true arrival order (prose, tool, prose) instead of the
    // tool jumping ahead of buffered text or the text being held behind it.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "before tool".to_string(),
        done: false,
    });
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-passthrough".to_string()),
        name: "bash".to_string(),
        summary: r#"{"command":"ls"}"#.to_string(),
        preview: ToolPreview::Bash {
            command: "ls".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "after tool".to_string(),
        done: true,
    });

    let kinds: Vec<&str> = app
        .transcript()
        .blocks()
        .iter()
        .filter_map(|b| match b {
            RenderBlock::TextDelta { text, .. } if !text.is_empty() => Some("text"),
            RenderBlock::ToolCall { .. } => Some("tool"),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["text", "tool", "text"],
        "a non-prose block flushes the paced tail, preserving true arrival order"
    );
    // The first prose block's full text is flushed whole by the tool barrier
    // (not left half-revealed behind the tool).
    let first_text = app
        .transcript()
        .blocks()
        .iter()
        .find_map(|b| match b {
            RenderBlock::TextDelta { text, .. } if !text.is_empty() => Some(text.clone()),
            _ => None,
        })
        .expect("a prose block exists");
    assert_eq!(first_text, "before tool", "the pre-tool tail is flushed whole");
}

#[test]
fn startup_alt_s_prefills_summary_prompt_without_submitting() {
    let mut app = test_app();
    app.enable_input();
    app.set_startup_screen(startup_screen_for_shortcut());

    let action = press_alt_char(&mut app, 's');

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), STARTUP_SUMMARIZE_REPO_PROMPT);
}

#[test]
fn startup_provider_shortcuts_prefill_login_commands_without_submitting() {
    let mut claude = test_app();
    claude.enable_input();
    claude.set_startup_screen(startup_screen_for_shortcut());
    assert_eq!(press_alt_char(&mut claude, 'c'), AppAction::None);
    assert_eq!(claude.input().text(), STARTUP_LOGIN_CLAUDE_COMMAND);

    let mut openai = test_app();
    openai.enable_input();
    openai.set_startup_screen(startup_screen_for_shortcut());
    assert_eq!(press_alt_char(&mut openai, 'o'), AppAction::None);
    assert_eq!(openai.input().text(), STARTUP_LOGIN_OPENAI_COMMAND);
}

#[test]
fn startup_alt_p_prefills_permissions_command_without_submitting() {
    let mut app = test_app();
    app.enable_input();
    app.set_startup_screen(startup_screen_for_shortcut());

    let action = press_alt_char(&mut app, 'p');

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), STARTUP_PERMISSIONS_COMMAND);
}

#[test]
fn startup_prefill_shortcuts_do_not_overwrite_existing_draft() {
    for (key, command) in [
        ('s', STARTUP_SUMMARIZE_REPO_PROMPT),
        ('c', STARTUP_LOGIN_CLAUDE_COMMAND),
        ('o', STARTUP_LOGIN_OPENAI_COMMAND),
        ('p', STARTUP_PERMISSIONS_COMMAND),
    ] {
        let mut app = test_app();
        app.enable_input();
        app.set_startup_screen(startup_screen_for_shortcut());
        app.set_input_text("keep my draft");

        let action = press_alt_char(&mut app, key);

        assert_eq!(action, AppAction::None, "shortcut {command}");
        assert_eq!(app.input().text(), "keep my draft", "shortcut {command}");
    }
}

#[test]
fn startup_prefill_shortcuts_are_noops_when_launchpad_hidden() {
    for key in ['s', 'c', 'o', 'p'] {
        let mut app = test_app();
        app.enable_input();

        let action = press_alt_char(&mut app, key);

        assert_eq!(action, AppAction::None);
        assert_eq!(app.input().text(), "", "shortcut {key} inserted text");
    }
}

#[test]
fn begin_turn_seeds_zo_activity_before_any_block() {
    // Regression: before the first stream block arrives, the live indicator
    // must read as active cognition (the Zo cue `Thinking`), not the bare
    // "Working" fallback. Gemini (Code Assist) computes server-side before its
    // first SSE frame, so without this seed its longer pre-first-block wait read
    // as a frozen turn. Claude/GPT stream reasoning at once and overwrite it.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    let action = app
        .turn_activity()
        .expect("turn is active")
        .current_action();
    assert_eq!(
        action,
        crate::tui::blocks::reasoning::ZO_REVEAL_VERBS[0]
    );
    assert_eq!(action, "Thinking");
}

#[test]
fn first_block_replaces_the_seeded_thinking_activity() {
    // The seed is only a placeholder for the pre-first-block wait: the first
    // real block (here a running tool call) must take over the activity line.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call-seed".to_string()),
        name: "bash".to_string(),
        summary: r#"{"command":"echo hi"}"#.to_string(),
        preview: ToolPreview::Bash {
            command: "echo hi".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    let action = app
        .turn_activity()
        .expect("turn is active")
        .current_action();
    assert_ne!(
        action,
        crate::tui::blocks::reasoning::ZO_REVEAL_VERBS[0],
        "the first real block must take over from the seeded zo cue"
    );
    assert_eq!(action, "Running command: echo hi");
}

#[test]
fn completed_text_block_replaces_seeded_thinking_activity() {
    let mut app = test_app();
    app.begin_turn_with_generation(1);

    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "final answer".to_string(),
        done: true,
    });

    let action = app
        .turn_activity()
        .expect("turn is active")
        .current_action();
    assert_eq!(action, "Planning");
}

#[test]
fn turn_lifecycle_shows_live_activity_then_clears_it_on_settle() {
    // v3 cross-surface contract (streaming-style-v3 §4): the transcript renders
    // a live turn and a settled turn identically — the live-vs-done distinction
    // lives ONLY in the bottom activity line. So the full lifecycle must show a
    // live activity while streaming and clear it on settle: begin_turn seeds it,
    // a streaming (not-done) delta drives it to the turn's Zo verb, and
    // end_turn removes it so a settled turn shows no live line.
    let mut app = test_app_with_theme(Theme::zo());

    app.begin_turn_with_generation(0);
    assert!(
        app.turn_activity().is_some(),
        "begin_turn seeds a live activity line"
    );

    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "streaming answer".to_string(),
        done: false,
    });
    assert_eq!(
        app.turn_activity()
            .expect("the turn is still live while streaming")
            .current_action(),
        "Thinking",
        "a live streaming delta drives the activity line"
    );

    app.end_turn();
    assert!(
        app.turn_activity().is_none(),
        "end_turn clears the live activity — a settled turn shows no live line"
    );
    assert!(
        matches!(app.heat_state(), HeatState::Cooling { .. }),
        "the existing activity teardown stamps the cooling window"
    );

    app.begin_turn_with_generation(0);
    assert_eq!(app.heat_state(), HeatState::Hot);
    assert!(
        app.cooled_since.is_none(),
        "the next canonical turn start clears the prior cooling timestamp"
    );
}

#[test]
fn abort_turn_clears_activity_without_starting_cooling() {
    let mut app = test_app_with_theme(Theme::zo());
    app.begin_turn_with_generation(1);

    app.abort_turn();

    assert!(app.turn_activity().is_none());
    assert!(app.cooled_since.is_none());
    assert_eq!(app.heat_state(), HeatState::Cold);
}

#[test]
fn no_color_end_turn_does_not_schedule_cooling() {
    let mut app = test_app();
    app.begin_turn_with_generation(1);

    app.end_turn();

    assert!(app.turn_activity().is_none());
    assert!(app.cooled_since.is_none());
    assert!(!app.cooling_active_at(std::time::Instant::now()));
}

#[test]
fn cooling_animation_request_stops_at_cold_boundary() {
    use std::time::{Duration, Instant};

    let now = Instant::now();
    let mut app = test_app_with_theme(Theme::zo());
    app.mode = AppMode::Focus;
    app.cooled_since = Some(
        now.checked_sub(Duration::from_millis(2_999))
            .expect("test instant has 2999ms of history"),
    );
    let mut was_cooling = app.cooling_active_at(now);
    assert!(was_cooling);

    app.cooled_since = Some(
        now.checked_sub(Duration::from_secs(3))
            .expect("test instant has three seconds of history"),
    );
    let cooling_active = app.cooling_active_at(now);
    assert!(!cooling_active);
    assert_eq!(app.heat_state_at(now), HeatState::Cold);

    let mut dirty = false;
    super::run_loop::track_cooling_boundary(
        &mut was_cooling,
        cooling_active,
        &mut dirty,
    );
    assert!(dirty, "the first Cold tick must request one final frame");

    let mut terminal = Terminal::new(TestBackend::new(80, 16)).expect("test terminal");
    app.draw(&mut terminal).expect("draw final Cold frame");
    let input = app.regions.expect("draw computes input region").input;
    let buffer = terminal.backend().buffer();
    let caret = &buffer[(input.x + 1, input.y + 1)];
    let rail = &buffer[(input.x, input.y + input.height - 1)];
    assert_eq!(caret.symbol(), "❯");
    assert_eq!(caret.fg, app.theme.palette.bright);
    assert_eq!(rail.symbol(), "┃");
    assert_eq!(rail.fg, app.theme.palette.faint);

    dirty = false;
    super::run_loop::track_cooling_boundary(&mut was_cooling, false, &mut dirty);
    assert!(!dirty, "later Cold ticks must stop requesting frames");
}

/// 라이브 리포트: 침묵 리즈닝 하트비트가 트랜스크립트 한 줄로만 흐르고
/// 스피너 배지는 계속 "no output"이라 유저가 행으로 오인, Esc로 턴을 죽였다.
/// 하트비트 System 행이 App을 통과하면 quiet 래치가 켜져 배지가
/// "reasoning · stream alive"로 바뀌고, 다음 실제 스트림 진행이 래치를 푼다.
#[test]
fn quiet_reasoning_notice_row_latches_calm_badge_state() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::System {
        id: BlockId(1),
        level: SystemLevel::Info,
        text: format!(
            "{} (61s+ without visible output)",
            core_types::QUIET_REASONING_LABEL
        ),
    });
    assert!(
        app.turn_activity()
            .expect("turn is active")
            .stream_alive_quiet(),
        "the heartbeat row must latch the calm badge state"
    );
    // An unrelated System row must NOT latch.
    let mut other = test_app();
    other.begin_turn_with_generation(0);
    other.push_block(RenderBlock::System {
        id: BlockId(2),
        level: SystemLevel::Info,
        text: "Agent 'explorer' finished".to_string(),
    });
    assert!(!other
        .turn_activity()
        .expect("turn is active")
        .stream_alive_quiet());
    // Real streamed progress clears the latch, so a later genuine freeze
    // surfaces as "no output" again.
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(3),
        text: "first visible token".to_string(),
        done: false,
    });
    assert!(!app
        .turn_activity()
        .expect("turn is active")
        .stream_alive_quiet());
}

#[test]
fn turn_start_clears_fallback_models_and_hud_rebuilds_preserve_live_notices() {
    let mut app = test_app();
    app.hud_state.turn_fallback_model = Some("opus".to_string());
    app.hud_state.quota_fallback_model = Some("openai:gpt-5.6-sol".to_string());

    let mut rebuild = app.hud_state.clone();
    rebuild.turn_fallback_model = None;
    rebuild.quota_fallback_model = None;
    app.set_hud_state(rebuild);
    assert_eq!(app.hud_state.turn_fallback_model.as_deref(), Some("opus"));
    assert_eq!(
        app.hud_state.quota_fallback_model.as_deref(),
        Some("openai:gpt-5.6-sol")
    );

    app.begin_turn_with_generation(1);
    assert_eq!(app.hud_state.turn_fallback_model, None);
    assert_eq!(app.hud_state.quota_fallback_model, None);
}

#[test]
fn fallback_notices_update_hud_without_mislabeling_quota_swap_as_hold() {
    let mut refusal = test_app();
    refusal.begin_turn_with_generation(0);
    refusal.push_block(RenderBlock::System {
        id: BlockId(10),
        level: SystemLevel::Warn,
        text: core_types::REFUSAL_FALLBACK_WARN.to_string(),
    });
    assert_eq!(
        refusal.hud_state.turn_fallback_model.as_deref(),
        Some("opus")
    );

    let model = "openai:gpt-5.6-sol";
    for (id, detail) in [
        (11, "the main model is rate-limited (quota exhausted)"),
        (12, "the main model is still cooling down from a rate limit"),
    ] {
        let mut app = test_app();
        app.begin_turn_with_generation(0);
        app.push_block(RenderBlock::System {
            id: BlockId(id),
            level: SystemLevel::Warn,
            text: format!(
                "{}{model}; {detail}",
                core_types::QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX
            ),
        });
        assert_eq!(app.hud_state.quota_fallback_model.as_deref(), Some(model));
        assert!(
            !app.turn_activity().expect("turn is active").quota_hold(),
            "quota fallback notice must not latch the same-model hold state"
        );
    }

    let mut hold = test_app();
    hold.begin_turn_with_generation(0);
    hold.push_block(RenderBlock::System {
        id: BlockId(13),
        level: SystemLevel::Warn,
        text: format!(
            "{} (claude-fable-5); holding this turn",
            core_types::QUOTA_HOLD_NOTICE_PREFIX
        ),
    });
    assert_eq!(hold.hud_state.quota_fallback_model, None);
    assert!(hold.turn_activity().expect("turn is active").quota_hold());
}

#[test]
fn reasoning_deltas_echo_accumulated_first_line_not_a_stray_fragment() {
    // Regression: the reasoning activity used to read only the latest DELTA
    // (`ThinkingDelta` is emitted per-event, NOT accumulated), so a later delta
    // like "먼저 살펴보자" would replace the full first line and the status would
    // jump/flicker to a mid-stream fragment. The fix reconstructs the accumulated
    // reasoning (transcript's prior block of the same id + current delta) and
    // echoes its *first line*, which grows monotonically as deltas arrive.
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    let id = BlockId(700);
    let mut seen = Vec::new();
    for delta in [
        "인증 ",
        "흐름을 ",
        "먼저 살펴보자\n다음으로 토큰 갱신을 확인",
    ] {
        app.push_block(RenderBlock::Reasoning {
            id,
            text: delta.to_string(),
            signature: None,
            done: false,
        });
        seen.push(
            app.turn_activity()
                .expect("turn is active")
                .current_action()
                .to_string(),
        );
    }

    // Each step echoes the accumulated first line so far — never a bare later
    // fragment ("먼저 살펴보자") and never the "Thinking…" placeholder once text
    // has arrived.
    assert_eq!(seen[0], "인증");
    assert_eq!(seen[1], "인증 흐름을");
    assert_eq!(seen[2], "인증 흐름을 먼저 살펴보자");
    // Monotonic: every step is a prefix of the next (the first line only grows).
    assert!(
        seen[1].starts_with(&seen[0]) && seen[2].starts_with(&seen[1]),
        "reasoning activity must grow monotonically, got {seen:?}"
    );
}

#[test]
fn tool_call_updates_live_turn_activity_summary() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    app.push_block(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call-1".to_string()),
        name: "bash".to_string(),
        summary: r#"{"command":"cargo test -p zo-cli"}"#.to_string(),
        preview: ToolPreview::Bash {
            command: "cargo test -p zo-cli".to_string(),
        },
        status: ToolCallStatus::Running,
    });

    let action = app
        .turn_activity()
        .expect("turn is active")
        .current_action();
    assert_eq!(action, "Running command: cargo test -p zo-cli");
}

#[test]
fn tool_result_updates_live_turn_activity_as_decision_step() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    app.push_block(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });

    let action = app
        .turn_activity()
        .expect("turn is active")
        .current_action();
    assert_eq!(action, "Reading tool output; choosing next step");
}

#[test]
fn todo_write_result_updates_hud_checklist_immediately() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    // A TodoWrite result body carries `new_todos`; the HUD must reflect it the
    // instant the block lands, without waiting for the file-poll snapshot.
    let payload = serde_json::json!({
        "old_todos": [],
        "new_todos": [
            {"stepId": "write-parser", "content": "Write parser", "activeForm": "Writing parser", "status": "in_progress"},
            {"content": "Add tests", "activeForm": "Adding tests", "status": "pending"}
        ]
    })
    .to_string();
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(3),
        tool_call_id: ToolCallId("todo-1".to_string()),
        is_error: false,
        body: ToolResultBody::Generic {
            name: "TodoWrite".to_string(),
            content: payload,
            truncated: false,
        },
    });

    assert_eq!(
        app.hud_state.todo_items.len(),
        2,
        "todos must appear at once"
    );
    assert_eq!(app.hud_state.todo_items[0].content, "Write parser");
    assert_eq!(
        app.hud_state.todo_items[0].step_id.as_deref(),
        Some("write-parser")
    );
    assert_eq!(app.hud_state.todo_items[1].step_id, None);
    assert_eq!(
        app.hud_state.todo_items[0].status,
        crate::tui::hud::TodoChecklistStatus::InProgress
    );
    assert_eq!(
        app.hud_state.todo_summary.as_deref(),
        Some("2 todos active")
    );
}

#[test]
fn typed_todos_result_body_updates_hud_checklist_immediately() {
    // Production shape: the runtime formats a TodoWrite result into the typed
    // `Todos` body (not raw JSON). `apply_todo_tool_result` must map it into the
    // HUD checklist the instant it lands, keeping the sidebar in lockstep with
    // the transcript block (no wait for the ~330ms disk poll).
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(7),
        tool_call_id: ToolCallId("todo-typed-1".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Write parser".to_string(),
                active_form: "Writing parser".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Add tests".to_string(),
                active_form: "Adding tests".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });

    assert_eq!(
        app.hud_state.todo_items.len(),
        2,
        "typed todos appear at once"
    );
    assert_eq!(app.hud_state.todo_items[0].content, "Write parser");
    assert_eq!(
        app.hud_state.todo_items[0].step_id, None,
        "provider-neutral typed results wait for the store poll to restore Zo ids"
    );
    assert_eq!(
        app.hud_state.todo_items[0].status,
        crate::tui::hud::TodoChecklistStatus::InProgress
    );
    assert_eq!(
        app.hud_state.todo_summary.as_deref(),
        Some("2 todos active")
    );
}

#[test]
fn typed_todos_result_body_all_completed_clears_hud_checklist() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(8),
        tool_call_id: ToolCallId("todo-typed-2".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![runtime::message_stream::TodoResultItem {
            content: "Ship it".to_string(),
            active_form: "Shipping it".to_string(),
            status: runtime::message_stream::TodoResultStatus::Completed,
        }]),
    });
    assert!(
        app.hud_state.todo_items.is_empty(),
        "an all-completed typed list should clear from the HUD immediately"
    );
    assert_eq!(
        app.hud_state.todo_summary, None,
        "completed-only lists must not claim active work remains"
    );
}

// --- set_hud_state contract (P7 pin) ---------------------------------------
// The turn-boundary rebuild can momentarily report LOWER tool counts and a
// zero ctx/cost (a tool-only iteration, or before the UsageTracker has billed
// the first turn). set_hud_state must merge, not clobber: user-visible totals
// never regress. Later phases (P8 helpers) assemble against this pinned merge.

#[test]
fn set_hud_state_never_regresses_counters_or_ctx_cost() {
    let mut app = test_app();
    app.hud_state.bash_count = 5;
    app.hud_state.read_count = 7;
    app.hud_state.edit_count = 3;
    app.hud_state.ctx_used = 100;
    app.hud_state.cost_usd = 1.25;

    let mut rebuild = app.hud_state.clone();
    rebuild.bash_count = 2;
    rebuild.read_count = 1;
    rebuild.edit_count = 0;
    rebuild.ctx_used = 0;
    rebuild.cost_usd = 0.0;
    app.set_hud_state(rebuild);

    assert_eq!(app.hud_state.bash_count, 5, "bash count must not regress");
    assert_eq!(app.hud_state.read_count, 7, "read count must not regress");
    assert_eq!(app.hud_state.edit_count, 3, "edit count must not regress");
    assert_eq!(
        app.hud_state.ctx_used, 100,
        "a zero-ctx rebuild must floor to the live value, not reset"
    );
    assert!(
        (app.hud_state.cost_usd - 1.25).abs() < f64::EPSILON,
        "a zero-cost rebuild must floor to the live value, not reset"
    );
}

#[test]
fn set_hud_state_raises_counters_and_usage_when_higher() {
    let mut app = test_app();
    app.hud_state.bash_count = 5;
    app.hud_state.ctx_used = 100;
    app.hud_state.cost_usd = 1.0;

    let mut rebuild = app.hud_state.clone();
    rebuild.bash_count = 9;
    rebuild.ctx_used = 250;
    rebuild.cost_usd = 2.5;
    app.set_hud_state(rebuild);

    assert_eq!(
        app.hud_state.bash_count, 9,
        "a higher count is authoritative and must be applied"
    );
    assert_eq!(
        app.hud_state.ctx_used, 250,
        "a non-zero rebuild ctx must raise the live value"
    );
    assert!(
        (app.hud_state.cost_usd - 2.5).abs() < f64::EPSILON,
        "a positive rebuild cost must raise the live value"
    );
}

#[test]
fn todo_write_result_all_completed_clears_hud_checklist() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    // Seed a live list first.
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(4),
        tool_call_id: ToolCallId("todo-2".to_string()),
        is_error: false,
        body: ToolResultBody::Generic {
            name: "TodoWrite".to_string(),
            content: serde_json::json!({
                "new_todos": [
                    {"content": "A", "activeForm": "A", "status": "in_progress"}
                ]
            })
            .to_string(),
            truncated: false,
        },
    });
    assert_eq!(app.hud_state.todo_items.len(), 1);

    // Now every item is completed → clear the HUD/live panel immediately so it
    // does not linger as a stale completed checklist.
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(5),
        tool_call_id: ToolCallId("todo-3".to_string()),
        is_error: false,
        body: ToolResultBody::Generic {
            name: "TodoWrite".to_string(),
            content: serde_json::json!({
                "new_todos": [
                    {"content": "A", "activeForm": "A", "status": "completed"}
                ]
            })
            .to_string(),
            truncated: false,
        },
    });
    assert!(
        app.hud_state.todo_items.is_empty(),
        "all-completed checklist must clear from the HUD immediately"
    );
    assert_eq!(app.hud_state.todo_summary, None);
}

#[test]
fn completed_todo_clears_live_panel_and_hud_state() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(9),
        tool_call_id: ToolCallId("todo-complete-panel".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![runtime::message_stream::TodoResultItem {
            content: "Ship it".to_string(),
            active_form: "Shipping it".to_string(),
            status: runtime::message_stream::TodoResultStatus::Completed,
        }]),
    });

    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let during = dump_all(&terminal, 80, 16);

    // A plan written/updated this turn whose items are now complete should not
    // linger as a checked-off live panel or sidebar checklist.
    assert!(
        !during.contains("Updated Plan"),
        "a completed plan touched this turn should clear from the live panel: {during}"
    );
    assert!(
        !during.contains("Ship it"),
        "the completed item row should not linger in the live panel: {during}"
    );
    assert!(
        app.hud_state.todo_items.is_empty(),
        "completed snapshot clears HUD/sidebar state immediately"
    );
}

#[test]
fn live_todo_panel_stays_bottom_anchored_on_short_transcript() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(10),
        text: "short log".to_string(),
        done: true,
    });
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(11),
        tool_call_id: ToolCallId("todo-anchor".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "A".to_string(),
                active_form: "Doing A".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "B".to_string(),
                active_form: "Doing B".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });

    let width = 80;
    let height = 20;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("regions after draw");
    let title_y = (0..height)
        .find(|&y| buffer_row(&terminal, width, y).contains("Updated Plan"))
        .expect("live todo panel title row should render");

    // Two items + one title row + one bottom border row, plus one gap below the panel.
    // With no queue preview reserved, the title row should be exactly in the
    // bottom-anchored slot above the input rule, not up by the short transcript content.
    let expected_y = regions
        .transcript
        .y
        .saturating_add(regions.transcript.height.saturating_sub(4 + 1));
    assert_eq!(
        title_y, expected_y,
        "todo panel must be bottom-anchored above the input, not content-anchored"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn live_todo_panel_border_box_and_overlap_prevention() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(10),
        text: "short log".to_string(),
        done: true,
    });
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(11),
        tool_call_id: ToolCallId("todo-anchor".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "A".to_string(),
                active_form: "Doing A".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "B".to_string(),
                active_form: "Doing B".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });

    let width = 80;
    let height = 20;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let regions = app.regions.expect("regions after draw");
    let transcript_y = regions.transcript.y;
    let transcript_h = regions.transcript.height;
    let transcript_w = regions.transcript.width as usize;

    // The todo panel height is:
    // 2 items + 1 top border + 1 bottom border = 4 rows.
    // Plus 1 gap row below + 1 top gap row = 6 rows total.
    // The panel's top (title) row is at transcript_y + transcript_h - (4 + 1) = transcript_y + transcript_h - 5.
    let title_y = transcript_y + transcript_h - 5;

    // Check top border row: contains "+ Updated Plan · 0/2 done" and ends with "+"
    let top_row_full = buffer_row(&terminal, width, title_y);
    let top_row_chars: Vec<char> = top_row_full.chars().collect();
    let top_row = top_row_chars[0..transcript_w].iter().collect::<String>();
    assert!(
        top_row.contains("+- Updated Plan"),
        "top border title missing: {top_row}"
    );
    assert!(
        top_row.contains("0/2 done"),
        "top border tally missing: {top_row}"
    );
    assert!(
        top_row.ends_with('+'),
        "top border corner missing: {top_row}"
    );
    assert!(
        top_row.matches('-').count() > 10,
        "top border must draw a horizontal rule, not only a title row: {top_row}"
    );

    // Check item rows
    let item1_row_full = buffer_row(&terminal, width, title_y + 1);
    let item1_row_chars: Vec<char> = item1_row_full.chars().collect();
    let item1_row = item1_row_chars[0..transcript_w].iter().collect::<String>();
    assert!(
        item1_row.contains("| [~] Doing A"),
        "item 1 content/marker incorrect: {item1_row}"
    );
    assert!(
        item1_row.ends_with('|'),
        "item 1 border missing: {item1_row}"
    );

    let item2_row_full = buffer_row(&terminal, width, title_y + 2);
    let item2_row_chars: Vec<char> = item2_row_full.chars().collect();
    let item2_row = item2_row_chars[0..transcript_w].iter().collect::<String>();
    assert!(
        item2_row.contains("| [ ] B"),
        "item 2 content/marker incorrect: {item2_row}"
    );
    assert!(
        item2_row.ends_with('|'),
        "item 2 border missing: {item2_row}"
    );

    // Check bottom border row: starts with "+" and ends with "+"
    let bot_row_full = buffer_row(&terminal, width, title_y + 3);
    let bot_row_chars: Vec<char> = bot_row_full.chars().collect();
    let bot_row = bot_row_chars[0..transcript_w].iter().collect::<String>();
    assert!(
        bot_row.starts_with('+'),
        "bottom border start missing: {bot_row}"
    );
    assert!(
        bot_row.ends_with('+'),
        "bottom border end missing: {bot_row}"
    );

    // Check gap row above the panel (title_y - 1) is empty/cleared
    let top_gap_full = buffer_row(&terminal, width, title_y - 1);
    let top_gap_chars: Vec<char> = top_gap_full.chars().collect();
    let top_gap = top_gap_chars[0..transcript_w].iter().collect::<String>();
    assert!(
        top_gap.trim().is_empty(),
        "top gap row must be empty: {top_gap}"
    );

    // Check gap row below the panel (title_y + 4) is empty/cleared
    let bot_gap_full = buffer_row(&terminal, width, title_y + 4);
    let bot_gap_chars: Vec<char> = bot_gap_full.chars().collect();
    let bot_gap = bot_gap_chars[0..transcript_w].iter().collect::<String>();
    assert!(
        bot_gap.trim().is_empty(),
        "bottom gap row must be empty: {bot_gap}"
    );

    // Check transcript text "short log" is rendered above the top gap (so y < title_y - 1)
    let found_log_y = (0..title_y - 1).find(|&y| {
        let r_full = buffer_row(&terminal, width, y);
        let r_chars: Vec<char> = r_full.chars().collect();
        let r = r_chars[0..transcript_w].iter().collect::<String>();
        r.contains("short log")
    });
    assert!(
        found_log_y.is_some(),
        "transcript content must be rendered above the todo panel to prevent overlap"
    );
}

#[test]
fn live_plan_and_workflow_share_one_plan_to_executor_dock() {
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(12),
        tool_call_id: ToolCallId("todo-run-dock".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Inspect the TUI".to_string(),
                active_form: "Inspecting the TUI".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Verify the result".to_string(),
                active_form: "Verifying the result".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "tui-run-dock".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "inspect".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 2,
        progress_percent: 33,
        completed_phases: 0,
        next_phase: Some("verify".to_string()),
        total_agents: 2,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 2,
        phases: Vec::new(),
    });
    let mut agent = running_agent_summary("researcher");
    agent.current_tool = Some("read_file".to_string());
    app.hud_state.running_agents = 2;
    app.hud_state.agents = vec![agent];

    let (width, height) = (100, 24);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw integrated run dock");
    let rendered = dump_all(&terminal, width, height);

    assert!(
        rendered.contains("Updated Plan -> Execute"),
        "the live surface must expose the Plan -> Executor transition: {rendered}"
    );
    assert!(
        rendered.contains("Inspecting the TUI") && rendered.contains("Verify the result"),
        "plan rows stay visible in the integrated dock: {rendered}"
    );
    assert!(
        rendered.contains("agents")
            && rendered.contains("2 agents running")
            && rendered.contains("researcher -> read_file"),
        "the dock must use the actual live executor snapshot: {rendered}"
    );
    assert_eq!(
        rendered.matches("phase 1/2 inspect").count(),
        1,
        "workflow topology remains owned by the HUD instead of duplicating inside the dock: {rendered}"
    );
    assert!(
        !rendered.contains("Running 2 Workflow agents"),
        "the old pinned workflow placeholder must not duplicate the integrated dock: {rendered}"
    );
    assert_eq!(
        rendered.matches("researcher").count(),
        1,
        "the old pinned agent tree must not survive beside the integrated dock: {rendered}"
    );

    let dock = app
        .agent_panel_click_rect
        .expect("the inspectable run dock is an aggregate workflow click target");
    let action = app
        .handle_mouse(left_click(dock.x + 2, dock.y + 1))
        .expect("run dock click handled");
    assert_eq!(action, AppAction::OpenWorkflowViewer);
}

#[test]
fn exact_workflow_ids_place_only_the_phase_agent_under_its_plan_step() {
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(18),
        tool_call_id: ToolCallId("todo-exact-run-dock".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Inspect exact ownership".to_string(),
                active_form: "Inspecting exact ownership".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Validate fallback behavior".to_string(),
                active_form: "Validating fallback behavior".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });
    app.hud_state.todo_items[0].step_id = Some("inspect-step".to_string());
    app.hud_state.todo_items[1].step_id = Some("verify-step".to_string());
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "exact-run-dock".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "inspect-step".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 2,
        progress_percent: 20,
        completed_phases: 0,
        next_phase: Some("verify-step".to_string()),
        total_agents: 1,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 1,
        phases: vec![FleetPhase {
            id: "inspect-step".to_string(),
            step_id: Some("inspect-step".to_string()),
            agent_ids: vec!["target-agent".to_string()],
            status: "running".to_string(),
            total: 1,
            completed: 0,
            failed: 0,
            running: 1,
        }],
    });

    // Same display name, deliberately wrong global order: only manifest ids
    // can distinguish the phase executor from the unrelated live agent.
    let mut decoy = running_agent_summary("worker");
    decoy.id = "other-agent".to_string();
    decoy.current_tool = Some("wrong_tool".to_string());
    let mut target = running_agent_summary("worker");
    target.id = "target-agent".to_string();
    target.current_tool = Some("target_tool".to_string());
    app.hud_state.running_agents = 2;
    app.hud_state.agents = vec![decoy, target];

    let (width, height) = (100, 22);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw exactly correlated run dock");
    let rendered = dump_all(&terminal, width, height);
    let rows = rendered.lines().collect::<Vec<_>>();
    let owner_y = rows
        .iter()
        .position(|row| row.contains("Inspecting exact ownership"))
        .expect("owning plan row");
    let executor_y = rows
        .iter()
        .position(|row| row.contains("target_tool"))
        .expect("exact executor row");
    let next_y = rows
        .iter()
        .position(|row| row.contains("Validate fallback behavior"))
        .expect("next plan row");

    assert_eq!(
        executor_y,
        owner_y + 1,
        "the executor must sit directly under its owning step: {rendered}"
    );
    assert!(
        executor_y < next_y,
        "the exact executor belongs before the next plan step: {rendered}"
    );
    assert!(
        rows[executor_y].contains("1 agent running")
            && !rows[executor_y].contains("2 agents running"),
        "the row must use the phase tally, not the global live count: {rendered}"
    );
    assert!(
        !rendered.contains("wrong_tool"),
        "an unrelated first live agent must never be attributed to the step: {rendered}"
    );

    let dock = app
        .agent_panel_click_rect
        .expect("the correlated executor remains inspectable");
    assert_eq!(
        app.handle_mouse(left_click(dock.x + 2, dock.y + 2))
            .expect("correlated run dock click handled"),
        AppAction::OpenWorkflowViewer
    );
}

#[test]
fn exact_executor_owner_stays_visible_outside_the_default_plan_window() {
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};

    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    let todo = |step_id: &str, content: &str, status| TodoChecklistItem {
        step_id: Some(step_id.to_string()),
        content: content.to_string(),
        active_form: content.to_string(),
        status,
    };
    app.hud_state.todo_items = vec![
        todo(
            "active-frontier",
            "Unrelated active frontier",
            TodoChecklistStatus::InProgress,
        ),
        todo("queued-1", "Queued one", TodoChecklistStatus::Pending),
        todo("queued-2", "Queued two", TodoChecklistStatus::Pending),
        todo("queued-3", "Queued three", TodoChecklistStatus::Pending),
        todo("queued-4", "Queued four", TodoChecklistStatus::Pending),
        todo(
            "phase-six",
            "Correlated sixth step",
            TodoChecklistStatus::Pending,
        ),
    ];
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "preferred-owner".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "phase-six".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 1,
        progress_percent: 10,
        completed_phases: 0,
        next_phase: None,
        total_agents: 1,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 1,
        phases: vec![FleetPhase {
            id: "phase-six".to_string(),
            step_id: Some("phase-six".to_string()),
            agent_ids: vec!["phase-six-agent".to_string()],
            status: "running".to_string(),
            total: 1,
            completed: 0,
            failed: 0,
            running: 1,
        }],
    });
    let mut agent = running_agent_summary("sixth-owner");
    agent.id = "phase-six-agent".to_string();
    agent.current_tool = Some("verify_owner".to_string());
    app.hud_state.running_agents = 1;
    app.hud_state.agents = vec![agent];

    let (width, height) = (100, 24);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal)
        .expect("draw preferred exact executor owner");
    let rendered = dump_all(&terminal, width, height);
    let rows = rendered.lines().collect::<Vec<_>>();
    let owner_y = rows
        .iter()
        .position(|row| row.contains("Correlated sixth step"))
        .expect("correlated owner is forced into the capped plan window");
    let executor_y = rows
        .iter()
        .position(|row| row.contains("sixth-owner -> verify_owner"))
        .expect("correlated executor row");

    assert_eq!(executor_y, owner_y + 1, "{rendered}");
    assert!(
        !rendered.contains("Unrelated active frontier"),
        "the capped window should prioritize the exact owner over an unrelated frontier: {rendered}"
    );
}

#[test]
fn height_constrained_plan_dock_keeps_a_plan_row_and_drops_executor_chrome_safely() {
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(15),
        tool_call_id: ToolCallId("todo-short-run-dock".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Keep the plan visible".to_string(),
                active_form: "Keeping the plan visible".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Follow-up validation".to_string(),
                active_form: "Validating afterwards".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });
    app.hud_state.todo_items[0].step_id = Some("execute".to_string());
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "short".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "execute".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 1,
        progress_percent: 10,
        completed_phases: 0,
        next_phase: None,
        total_agents: 1,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 1,
        phases: vec![FleetPhase {
            id: "execute".to_string(),
            step_id: Some("execute".to_string()),
            agent_ids: vec!["starting-agent".to_string()],
            status: "running".to_string(),
            total: 1,
            completed: 0,
            failed: 0,
            running: 1,
        }],
    });

    let (width, height) = (80, 11);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("short dock must not panic");
    let rendered = dump_all(&terminal, width, height);

    assert!(
        rendered.contains("Updated Plan") && rendered.contains("Keeping the plan visible"),
        "vertical degradation must preserve the active frontier row: {rendered}"
    );
    assert!(
        !rendered.contains("Follow-up validation"),
        "the single-row window should not displace the active frontier with lookahead: {rendered}"
    );
    assert!(
        !rendered.contains("Updated Plan -> Execute") && !rendered.contains("agents ·"),
        "executor chrome should disappear cleanly when no body row fits: {rendered}"
    );
    assert!(
        app.agent_panel_click_rect.is_none(),
        "a clipped executor row must not leave a stale aggregate click target"
    );
}

#[test]
fn live_plan_shows_the_actual_main_tool_without_creating_an_agent_target() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(13),
        tool_call_id: ToolCallId("todo-main-executor".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![runtime::message_stream::TodoResultItem {
            content: "Run verification".to_string(),
            active_form: "Running verification".to_string(),
            status: runtime::message_stream::TodoResultStatus::InProgress,
        }]),
    });
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "between-phases".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "verify".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 2,
        progress_percent: 50,
        completed_phases: 1,
        next_phase: None,
        total_agents: 0,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 0,
        phases: Vec::new(),
    });
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(14),
        tool_call_id: ToolCallId("bash-main-executor".to_string()),
        name: "bash".to_string(),
        summary: "cargo test -p zo-cli".to_string(),
        preview: ToolPreview::Generic {
            name: "bash".to_string(),
            input_summary: "cargo test -p zo-cli".to_string(),
        },
        status: ToolCallStatus::Running,
    });

    let (width, height) = (100, 20);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw main executor row");
    let rendered = dump_all(&terminal, width, height);

    assert!(
        rendered.contains("Updated Plan -> Execute")
            && rendered.contains("main")
            && rendered.contains("cargo test -p zo-cli"),
        "the current tool action must appear under the plan: {rendered}"
    );
    assert!(
        !rendered.contains("active between phases"),
        "a generic workflow fallback must not hide a concrete main-turn tool: {rendered}"
    );
    assert!(
        app.agent_panel_click_rect.is_none(),
        "a main-tool-only row must not pretend that an agent/workflow viewer exists"
    );
}

#[test]
fn multiple_in_progress_steps_share_one_run_level_executor_row() {
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(16),
        tool_call_id: ToolCallId("todo-parallel-run-dock".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Inspect layout".to_string(),
                active_form: "Inspecting layout".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Audit state".to_string(),
                active_form: "Auditing state".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
        ]),
    });
    let mut agent = running_agent_summary("builder");
    agent.status = "still_running".to_string();
    agent.current_tool = Some("bash".to_string());
    app.hud_state.running_agents = 1;
    app.hud_state.agents = vec![agent];

    let (width, height) = (100, 20);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw shared executor row");
    let rendered = dump_all(&terminal, width, height);
    let rows = rendered.lines().collect::<Vec<_>>();
    let first_step_y = rows
        .iter()
        .position(|row| row.contains("Inspecting layout"))
        .expect("first plan row");
    let second_step_y = rows
        .iter()
        .position(|row| row.contains("Auditing state"))
        .expect("second plan row");
    let executor_y = rows
        .iter()
        .position(|row| row.contains("builder -> bash"))
        .expect("legacy executor row");

    assert!(
        rendered.contains("Inspecting layout") && rendered.contains("Auditing state"),
        "parallel plan steps must remain separate plan rows: {rendered}"
    );
    assert_eq!(
        rendered.matches("1 agent running").count(),
        1,
        "executor state is run-level and must not be guessed once per plan step: {rendered}"
    );
    assert!(
        rendered.contains("builder -> bash"),
        "non-terminal manifest states must still expose concrete executor activity: {rendered}"
    );
    assert!(
        first_step_y < second_step_y && second_step_y < executor_y,
        "without exact ids the single run-level executor must remain below the plan: {rendered}"
    );
}

#[test]
fn synthetic_agents_phase_never_claims_a_same_named_plan_step() {
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(19),
        tool_call_id: ToolCallId("todo-synthetic-run-dock".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Run agents step".to_string(),
                active_form: "Running agents step".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Review agent output".to_string(),
                active_form: "Reviewing agent output".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });
    app.hud_state.todo_items[0].step_id = Some("agents".to_string());
    app.hud_state.todo_items[1].step_id = Some("review".to_string());
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "agents".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "agents".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 1,
        progress_percent: 10,
        completed_phases: 0,
        next_phase: None,
        total_agents: 1,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 1,
        phases: vec![FleetPhase {
            id: "agents".to_string(),
            step_id: None,
            agent_ids: vec!["synthetic-agent".to_string()],
            status: "running".to_string(),
            total: 1,
            completed: 0,
            failed: 0,
            running: 1,
        }],
    });
    let mut agent = running_agent_summary("synthetic-worker");
    agent.id = "synthetic-agent".to_string();
    agent.current_tool = Some("read_file".to_string());
    app.hud_state.running_agents = 1;
    app.hud_state.agents = vec![agent];

    let (width, height) = (100, 20);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw synthetic fallback");
    let rendered = dump_all(&terminal, width, height);
    let rows = rendered.lines().collect::<Vec<_>>();
    let owner_y = rows
        .iter()
        .position(|row| row.contains("Running agents step"))
        .expect("same-named plan row");
    let next_y = rows
        .iter()
        .position(|row| row.contains("Review agent output"))
        .expect("next plan row");
    let executor_y = rows
        .iter()
        .position(|row| row.contains("synthetic-worker -> read_file"))
        .expect("run-level executor row");

    assert!(
        owner_y < next_y && next_y < executor_y,
        "a synthetic phase id is not a real step id and must stay run-level: {rendered}"
    );
}

#[test]
fn terminal_or_inconsistent_workflow_phase_stays_run_level() {
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};

    for (case, phase_id, step_id, phase_status) in [
        ("terminal", "guard-step", "guard-step", "done"),
        ("mismatched-id", "other-phase", "guard-step", "running"),
    ] {
        let mut app = test_app();
        app.sidebar.visible = false;
        app.begin_turn_with_generation(0);
        app.hud_state.todo_items = vec![
            TodoChecklistItem {
                step_id: Some("guard-step".to_string()),
                content: "Completed workflow step".to_string(),
                active_form: "Completing workflow step".to_string(),
                status: TodoChecklistStatus::Completed,
            },
            TodoChecklistItem {
                step_id: Some("synthesize".to_string()),
                content: "Synthesize final result".to_string(),
                active_form: "Synthesizing final result".to_string(),
                status: TodoChecklistStatus::InProgress,
            },
        ];
        app.hud_state.workflow = Some(WorkflowSummary {
            name: format!("guard-{case}"),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: phase_id.to_string(),
            current_phase_status: phase_status.to_string(),
            current_phase_index: 1,
            total_phases: 1,
            progress_percent: 90,
            completed_phases: usize::from(phase_status == "done"),
            next_phase: None,
            total_agents: 1,
            completed_agents: 1,
            failed_agents: 0,
            running_agents: 0,
            phases: vec![FleetPhase {
                id: phase_id.to_string(),
                step_id: Some(step_id.to_string()),
                agent_ids: vec!["finished-agent".to_string()],
                status: phase_status.to_string(),
                total: 1,
                completed: 1,
                failed: 0,
                running: 0,
            }],
        });

        let (width, height) = (100, 20);
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        app.draw(&mut terminal)
            .expect("draw guarded workflow fallback");
        let rendered = dump_all(&terminal, width, height);
        let rows = rendered.lines().collect::<Vec<_>>();
        let completed_y = rows
            .iter()
            .position(|row| row.contains("Completed workflow step"))
            .expect("completed plan row");
        let active_y = rows
            .iter()
            .position(|row| row.contains("Synthesizing final result"))
            .expect("active synthesis row");
        let executor_y = rows
            .iter()
            .position(|row| row.contains("active between phases"))
            .expect("run-level workflow fallback");

        assert!(
            completed_y < active_y && active_y < executor_y,
            "{case} phase metadata must not claim the completed step: {rendered}"
        );
    }
}

#[test]
fn invalid_agent_ids_cannot_exactly_claim_a_plan_step() {
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};

    for (case, agent_ids, agent_id, phase_running) in [
        ("empty", vec![String::new()], "", 1),
        (
            "untrimmed",
            vec![" untrimmed-agent ".to_string()],
            " untrimmed-agent ",
            1,
        ),
    ] {
        let mut app = test_app();
        app.sidebar.visible = false;
        app.begin_turn_with_generation(0);
        app.hud_state.todo_items = vec![
            TodoChecklistItem {
                step_id: Some("guard-step".to_string()),
                content: "Guard exact agent ids".to_string(),
                active_form: "Guarding exact agent ids".to_string(),
                status: TodoChecklistStatus::InProgress,
            },
            TodoChecklistItem {
                step_id: Some("next-step".to_string()),
                content: "Continue safely".to_string(),
                active_form: "Continuing safely".to_string(),
                status: TodoChecklistStatus::Pending,
            },
        ];
        app.hud_state.workflow = Some(WorkflowSummary {
            name: format!("invalid-agent-id-{case}"),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "guard-step".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: 1,
            progress_percent: 10,
            completed_phases: 0,
            next_phase: None,
            total_agents: agent_ids.len(),
            completed_agents: 0,
            failed_agents: 0,
            running_agents: phase_running,
            phases: vec![FleetPhase {
                id: "guard-step".to_string(),
                step_id: Some("guard-step".to_string()),
                total: agent_ids.len(),
                agent_ids,
                status: "running".to_string(),
                completed: 0,
                failed: 0,
                running: phase_running,
            }],
        });
        let mut agent = running_agent_summary("invalid-id-worker");
        agent.id = agent_id.to_string();
        agent.current_tool = Some("invalid_id_tool".to_string());
        app.hud_state.running_agents = 1;
        app.hud_state.agents = vec![agent];

        let (width, height) = (100, 20);
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        app.draw(&mut terminal).expect("draw invalid id fallback");
        let rendered = dump_all(&terminal, width, height);
        let rows = rendered.lines().collect::<Vec<_>>();
        let next_y = rows
            .iter()
            .position(|row| row.contains("Continue safely"))
            .expect("next plan row");
        let executor_y = rows
            .iter()
            .position(|row| row.contains("invalid-id-worker -> invalid_id_tool"))
            .expect("run-level agent fallback");

        assert!(
            next_y < executor_y && rows[executor_y].contains("1 agent running"),
            "{case} ids must fall back below the plan without inflating the exact phase count: {rendered}"
        );
    }
}

#[test]
fn ambiguous_todo_or_current_phase_ids_stay_run_level() {
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};

    for case in ["duplicate-todo", "duplicate-phase"] {
        let mut app = test_app();
        app.sidebar.visible = false;
        app.begin_turn_with_generation(0);
        app.hud_state.todo_items = vec![
            TodoChecklistItem {
                step_id: Some("guard-step".to_string()),
                content: "Own guarded work".to_string(),
                active_form: "Owning guarded work".to_string(),
                status: TodoChecklistStatus::InProgress,
            },
            TodoChecklistItem {
                step_id: Some(if case == "duplicate-todo" {
                    "guard-step".to_string()
                } else {
                    "next-step".to_string()
                }),
                content: "Keep ambiguous work separate".to_string(),
                active_form: "Keeping ambiguous work separate".to_string(),
                status: TodoChecklistStatus::Pending,
            },
        ];
        let phase = FleetPhase {
            id: "guard-step".to_string(),
            step_id: Some("guard-step".to_string()),
            agent_ids: vec!["guard-agent".to_string()],
            status: "running".to_string(),
            total: 1,
            completed: 0,
            failed: 0,
            running: 1,
        };
        let phases = if case == "duplicate-phase" {
            vec![phase.clone(), phase]
        } else {
            vec![phase]
        };
        app.hud_state.workflow = Some(WorkflowSummary {
            name: format!("ambiguous-{case}"),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "guard-step".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: phases.len(),
            progress_percent: 10,
            completed_phases: 0,
            next_phase: None,
            total_agents: 1,
            completed_agents: 0,
            failed_agents: 0,
            running_agents: 1,
            phases,
        });
        let mut agent = running_agent_summary("guard-worker");
        agent.id = "guard-agent".to_string();
        agent.current_tool = Some("guard_tool".to_string());
        app.hud_state.running_agents = 1;
        app.hud_state.agents = vec![agent];

        let (width, height) = (100, 20);
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        app.draw(&mut terminal).expect("draw ambiguous id fallback");
        let rendered = dump_all(&terminal, width, height);
        let rows = rendered.lines().collect::<Vec<_>>();
        let next_y = rows
            .iter()
            .position(|row| row.contains("Keep ambiguous work separate"))
            .expect("second plan row");
        let executor_y = rows
            .iter()
            .position(|row| row.contains("guard-worker -> guard_tool"))
            .expect("run-level fallback row");

        assert!(
            next_y < executor_y,
            "{case} metadata must not produce an exact step attribution: {rendered}"
        );
    }
}

#[test]
fn run_dock_stacks_above_queued_preview_without_overlapping() {
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(17),
        tool_call_id: ToolCallId("todo-queued-run-dock".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![runtime::message_stream::TodoResultItem {
            content: "Inspect overlay stack".to_string(),
            active_form: "Inspecting overlay stack".to_string(),
            status: runtime::message_stream::TodoResultStatus::InProgress,
        }]),
    });
    let mut agent = running_agent_summary("stack-auditor");
    agent.current_tool = Some("read_file".to_string());
    app.hud_state.running_agents = 1;
    app.hud_state.agents = vec![agent];
    app.queue_message("QUEUED_AFTER_RUN_DOCK")
        .expect("queue message");

    let (width, height) = (100, 24);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw dock and queue stack");
    let rendered = dump_all(&terminal, width, height);
    let rows = rendered.lines().collect::<Vec<_>>();
    let plan_y = rows
        .iter()
        .position(|row| row.contains("Updated Plan -> Execute"))
        .expect("plan header row");
    let executor_y = rows
        .iter()
        .position(|row| row.contains("stack-auditor -> read_file"))
        .expect("executor row");
    let queue_y = rows
        .iter()
        .position(|row| row.contains("QUEUED_AFTER_RUN_DOCK"))
        .expect("queued preview row");

    assert!(
        plan_y < executor_y && executor_y < queue_y,
        "the dock must remain intact above the queued preview: {rendered}"
    );
    let dock = app
        .agent_panel_click_rect
        .expect("integrated dock click rect");
    assert!(
        usize::from(dock.y + dock.height) <= queue_y,
        "dock geometry must end before the queued preview begins: dock={dock:?}, queue_y={queue_y}"
    );
}

#[test]
fn non_todo_tool_result_leaves_hud_checklist_untouched() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(6),
        tool_call_id: ToolCallId("bash-1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "some normal output without the keyword".to_string(),
            truncated: false,
        },
    });
    assert!(app.hud_state.todo_items.is_empty());
}

#[test]
fn todowrite_plan_renders_once_during_turn_then_restores_as_history() {
    // Regression (the screenshot's double `Updated Plan`): while a turn streams,
    // the plan must render exactly once — in the live pinned bottom panel — and
    // the transcript's settled `Updated Plan` Todos block must be suppressed so
    // it does not also scroll in. Once the turn settles, that transcript block
    // reappears as a single settled history entry.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(80),
        tool_call_id: ToolCallId("todo-once-1".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Extract registry_io".to_string(),
                active_form: "Extracting registry_io".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Dedupe reporter if-else".to_string(),
                active_form: "Deduping reporter".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });
    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let during = dump_all(&terminal, 80, 16);

    // Exactly one `Updated Plan` while the turn is active: the live panel only.
    assert_eq!(
        during.matches("Updated Plan").count(),
        1,
        "plan must render once (live panel) during a turn, not duplicated: {during}"
    );
    // The in-progress active-form renders in that single live surface.
    assert!(
        during.contains("Extracting registry_io"),
        "in-progress active-form must render in the live panel: {during}"
    );
    // The HUD sidebar todo path stays in lockstep — the typed Todos body feeds
    // it the instant the result lands (no wait for the disk-poll).
    assert_eq!(
        app.hud_state.todo_items.len(),
        2,
        "sidebar/HUD todo state remains populated alongside the live panel"
    );

    // Turn settles: the live panel disappears and the transcript's settled
    // `Updated Plan` history block reappears — still exactly one on screen.
    app.end_turn();
    app.draw(&mut terminal).expect("draw");
    let after = dump_all(&terminal, 80, 16);
    assert_eq!(
        after.matches("Updated Plan").count(),
        1,
        "after the turn the plan shows once as settled transcript history: {after}"
    );
    // The settled block is the bordered history entry (top/bottom boundary).
    assert!(
        after.contains('+') || after.contains('╭'),
        "settled plan is a bordered block after the turn: {after}"
    );
}

#[test]
fn completed_todowrite_snapshot_clears_live_history_and_hud() {
    let _lock = todo_store_env_lock();
    let store = ScopedTodoStore::new("completed-history");
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(82),
        tool_call_id: ToolCallId("todo-complete-history-1".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Implement fix".to_string(),
                active_form: "Implementing fix".to_string(),
                status: runtime::message_stream::TodoResultStatus::InProgress,
            },
            runtime::message_stream::TodoResultItem {
                content: "Verify fix".to_string(),
                active_form: "Verifying fix".to_string(),
                status: runtime::message_stream::TodoResultStatus::Pending,
            },
        ]),
    });

    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let active = dump_all(&terminal, 80, 16);
    assert_eq!(
        active.matches("Updated Plan").count(),
        1,
        "active incomplete plan renders once in the live panel: {active}"
    );

    app.push_block(RenderBlock::ToolResult {
        id: BlockId(83),
        tool_call_id: ToolCallId("todo-complete-history-2".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![
            runtime::message_stream::TodoResultItem {
                content: "Implement fix".to_string(),
                active_form: "Implementing fix".to_string(),
                status: runtime::message_stream::TodoResultStatus::Completed,
            },
            runtime::message_stream::TodoResultItem {
                content: "Verify fix".to_string(),
                active_form: "Verifying fix".to_string(),
                status: runtime::message_stream::TodoResultStatus::Completed,
            },
        ]),
    });
    app.draw(&mut terminal).expect("draw completed snapshot");
    let completed_live = dump_all(&terminal, 80, 16);
    assert_eq!(
        completed_live.matches("Updated Plan").count(),
        0,
        "an all-completed plan should clear from the live panel immediately: {completed_live}"
    );

    app.end_turn();
    app.draw(&mut terminal).expect("draw after end_turn");
    let after = dump_all(&terminal, 80, 16);
    // Once the turn settles, neither the live panel nor a completed `Updated
    // Plan` history card should remain.
    assert_eq!(
        after.matches("Updated Plan").count(),
        0,
        "completed snapshot should not remain as chat history after the turn ends: {after}"
    );
    // The HUD/sidebar store, by contrast, IS cleared: a finished plan must not
    // ghost in the sidebar or reappear on the next, unrelated turn.
    assert!(
        app.hud_state.todo_items.is_empty(),
        "a completed plan is cleared from the sidebar/HUD when the turn settles"
    );
    let persisted = std::fs::read_to_string(store.path()).unwrap_or_default();
    assert!(
        persisted.trim().is_empty() || persisted.trim() == "[]",
        "the session todo store is emptied when the plan completes; persisted={persisted:?}"
    );
}

/// Serializes the lib tests that set `ZO_TODO_STORE` (a process-global env
/// var) — via the crate-wide env lock, because turn/compaction paths driven by
/// OTHER test modules read the same variable mid-test, and a module-local
/// mutex excludes none of them.
fn todo_store_env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::test_env_lock()
}

/// Points `ZO_TODO_STORE` at a unique temp file for a test and restores the
/// previous value on drop. Hold [`todo_store_env_lock`] for the test's lifetime.
struct ScopedTodoStore {
    prior: Option<std::ffi::OsString>,
    path: PathBuf,
}

impl ScopedTodoStore {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "zo-test-todo-{tag}-{}-{nanos}.json",
            std::process::id()
        ));
        let prior = std::env::var_os("ZO_TODO_STORE");
        std::env::set_var("ZO_TODO_STORE", &path);
        Self { prior, path }
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ScopedTodoStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        match self.prior.take() {
            Some(value) => std::env::set_var("ZO_TODO_STORE", value),
            None => std::env::remove_var("ZO_TODO_STORE"),
        }
    }
}

#[test]
fn stale_carryover_plan_does_not_pin_live_panel_until_touched() {
    // The "ghost plan" residue: an all-pending plan left in the store by an
    // earlier turn is re-loaded by the HUD poll. A later, unrelated turn must
    // NOT re-pin it above the input — only a plan the model writes/updates this
    // turn (or one with an in-progress item) owns the live panel.
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};

    let mut app = test_app();
    app.begin_turn_with_generation(0);
    // Simulate the live-snapshot poll loading a stale, all-pending checklist
    // from the store — directly into the HUD, the way `update_hud_live_snapshot`
    // does (no `TodoWrite` this turn, so `todo_touched_this_turn` stays false).
    app.hud_state.todo_items = vec![TodoChecklistItem {
        step_id: None,
        content: "old ghost plan".to_string(),
        status: TodoChecklistStatus::Pending,
        active_form: "old ghost plan".to_string(),
    }];

    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw stale carryover");
    let stale = dump_all(&terminal, 80, 16);
    // The live pinned panel (the bordered `Updated Plan` box above the input)
    // must be absent. The sidebar's `todo` section legitimately still lists the
    // item as outstanding work — that surface is the persistent status, not the
    // active-turn pin we are guarding here.
    assert!(
        !stale.contains("Updated Plan"),
        "a stale carryover plan untouched this turn must not pin the live panel: {stale}"
    );

    // The model now writes a plan this turn → it becomes the live, pinned plan.
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(91),
        tool_call_id: ToolCallId("todo-touch-1".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![runtime::message_stream::TodoResultItem {
            content: "Real work this turn".to_string(),
            active_form: "Doing real work".to_string(),
            status: runtime::message_stream::TodoResultStatus::InProgress,
        }]),
    });
    app.draw(&mut terminal).expect("draw touched plan");
    let touched = dump_all(&terminal, 80, 16);
    assert_eq!(
        touched.matches("Updated Plan").count(),
        1,
        "a plan written this turn pins the live panel exactly once: {touched}"
    );
    assert!(
        touched.contains("Doing real work"),
        "the freshly written in-progress item renders in the live panel: {touched}"
    );
}

fn dump_all(terminal: &Terminal<TestBackend>, width: u16, height: u16) -> String {
    (0..height).fold(String::new(), |mut out, y| {
        out.push_str(&buffer_row(terminal, width, y));
        out.push('\n');
        out
    })
}

#[test]
fn empty_open_text_delta_does_not_create_phantom_block() {
    // Some providers emit empty keepalive-ish text deltas. An empty non-final
    // delta carries no content, so `push_block` must skip it rather than paint a
    // phantom assistant prose block; the first non-empty delta opens the block.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(505),
        text: String::new(),
        done: false,
    });

    assert!(
        app.transcript().blocks().iter().all(|block| !matches!(block, RenderBlock::TextDelta { id, .. } if *id == BlockId(505))),
        "empty non-done text must not create a phantom prose block"
    );
}

#[test]
fn queued_preview_keeps_a_blank_boundary_above_the_first_entry() {
    // Regression (same class as the todo panel): the queued-message preview is
    // a `Clear` overlay pinned to the transcript bottom. Without a top boundary
    // row, the last visible transcript line butts directly against the first
    // queued entry. The transcript row directly above the first preview entry
    // must be blank.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    for i in 0..40 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(2000 + i),
            text: format!("DENSE_TRANSCRIPT_LINE_{i:02}_YYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"),
            done: true,
        });
    }
    app.queue_message("QUEUEDENTRY_ONE").expect("queue message");
    let width = 80;
    let height = 12;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let transcript = app.regions.expect("regions after draw").transcript;
    let col_lo = transcript.x;
    let col_hi = transcript.x + transcript.width;
    let transcript_slice = |y: u16| -> String {
        let buffer = terminal.backend().buffer();
        (col_lo..col_hi)
            .map(|x| buffer[(x, y)].symbol().to_string())
            .collect()
    };

    let preview_row = (0..height)
        .find(|&y| transcript_slice(y).contains("QUEUEDENTRY_ONE"))
        .expect("the queued-message preview must render its first entry");
    assert!(
        preview_row > 0,
        "preview cannot sit on the very first row here"
    );
    let boundary = transcript_slice(preview_row - 1);
    assert!(
        boundary.trim().is_empty(),
        "the transcript row directly above the first queued entry must be a blank \
         boundary, not fused transcript content: {boundary:?}"
    );
}

#[test]
fn empty_settled_prose_block_renders_no_author_mark() {
    // Phase-2 regression (the empty settled block): a settled TextDelta with
    // no body must be suppressed entirely — no author bullet on a row with no
    // answer — instead of painting a bare `◆` over an answer that does not
    // exist.
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(90),
        text: "hello".to_string(),
    });
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(91),
        text: String::new(),
        done: true,
    });

    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let height = 12u16;
    let backend = terminal.backend();
    let buffer = backend.buffer();
    for y in 0..height {
        let row: String = (0..80u16)
            .map(|x| buffer[(x, y)].symbol().to_string())
            .collect();
        let bare_mark = ['\u{25c6}', '*'].iter().any(|&mark| {
            row.starts_with(mark) && row.trim_start_matches(mark).trim().is_empty()
        });
        assert!(
            !bare_mark,
            "an empty settled prose block must not paint a bare author bullet: {row:?}"
        );
    }
}

#[test]
fn nonempty_prose_after_empty_phantom_keeps_author_mark() {
    // The phantom-suppression must be transparent to authorship: a real answer
    // following an empty settled block still earns its `◆` author bullet
    // instead of mis-reading the phantom as a prior prose block and dropping
    // to an indent-only continuation.
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(95),
        text: "question".to_string(),
    });
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(96),
        text: String::new(),
        done: true,
    });
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(97),
        text: "real answer body".to_string(),
        done: true,
    });

    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let dumped = dump_all(&terminal, 80, 12);
    assert!(
        dumped.contains("real answer body"),
        "the real answer must render: {dumped}"
    );
    assert!(
        dumped.contains("\u{25c6}  real answer body") || dumped.contains("*  real answer body"),
        "the real answer after a phantom must keep its author bullet (◆, or `*` under no-color): {dumped}"
    );
}

#[test]
fn steering_delivery_echo_clears_queued_entry() {
    // The runtime's `⤷ steering:` echo marks the moment a queued message was
    // folded into the live turn — its pending entry must leave the queue so
    // it does not also run as a separate turn afterwards.
    let (mut app, mut cmd_rx) = test_app_with_cmd();
    app.begin_turn_with_generation(0);
    app.disable_input();
    for ch in "focus on tests".chars() {
        app.handle_key(press(KeyCode::Char(ch))).expect("typed");
    }
    app.handle_key(press(KeyCode::Enter)).expect("enter");
    assert_eq!(app.queued_message_count(), 1, "entry shows as queued");
    let _ = cmd_rx.try_recv();

    // Delivery echo (as the runtime emits when folding) clears the entry.
    app.push_block(RenderBlock::System {
        id: BlockId(90),
        level: SystemLevel::Info,
        text: format!("{}focus on tests", runtime::STEERING_ECHO_PREFIX),
    });
    assert_eq!(
        app.queued_message_count(),
        0,
        "steered message must leave the queue once delivered"
    );
}

#[test]
fn mid_turn_slash_enter_queues_without_steering() {
    // Slash commands wait for their own turn — they never steer.
    let (mut app, mut cmd_rx) = test_app_with_cmd();
    app.begin_turn_with_generation(0);
    app.disable_input();
    for ch in "/status".chars() {
        app.handle_key(press(KeyCode::Char(ch))).expect("typed");
    }
    app.handle_key(press(KeyCode::Enter)).expect("enter");

    assert_eq!(app.queued_message_count(), 1);
    assert!(
        cmd_rx.try_recv().is_err(),
        "slash entries must not ride the steering channel"
    );
}

#[test]
fn replay_push_without_active_turn_lands_immediately() {
    // Rehydrate / resume reseeding pushes blocks with no live turn — those
    // must not typewriter-replay.
    let mut app = test_app();
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(73),
        text: "replayed history line REPLAYMARK".to_string(),
        done: true,
    });
    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let dumped = dump_all(&terminal, 80, 12);
    assert!(
        dumped.contains("REPLAYMARK"),
        "replay text lands whole in one frame: {dumped}"
    );
}

#[test]
fn host_prelude_can_set_live_turn_activity_without_render_block() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    app.set_turn_activity("Smart: preparing parallel pre-analysis");

    let action = app
        .turn_activity()
        .expect("turn is active")
        .current_action();
    assert_eq!(
        action,
        "Smart: preparing parallel pre-analysis"
    );
}

#[test]
fn plain_click_on_transcript_block_does_not_copy() {
    // Press and release without a drag stays inert. Copy remains available via
    // the explicit hover button, drag selection, and keyboard bindings.
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(42),
        text: "copy this block".to_string(),
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;

    let press = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: transcript.x.saturating_add(1),
            row: transcript.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("mouse handled");
    assert_eq!(press, AppAction::None, "press alone must not copy");
    assert!(
        !app.transcript.has_char_selection(),
        "a press without cell movement is not a drag selection"
    );

    let release = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: transcript.x.saturating_add(1),
            row: transcript.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("mouse handled");
    assert_eq!(
        release,
        AppAction::None,
        "plain click must not trigger a clipboard write"
    );
    assert!(!app.transcript.has_char_selection());
}

#[test]
fn wheel_mid_drag_extends_selection_and_copies_past_the_viewport() {
    // 드래그 홀드 중 휠 노치는 선택을 지우지 않는다: 스크롤한 뒤 포인터 밑
    // 셀로 head 를 다시 늘여 새로 드러난 행들이 제스처에 합류하고, 릴리즈는
    // 화면 밖으로 흘러나간 행까지 포함해 복사한다 — "휠로 더 많은 영역을
    // 복사하려면 드래그가 풀리던" 회귀의 앱-레벨 계약.
    let mut app = test_app();
    let words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima",
    ];
    for (idx, word) in words.iter().enumerate() {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx as u64 + 1),
            text: format!("row {word}"),
            done: true,
        });
    }
    let backend = TestBackend::new(120, 12);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    app.transcript.scroll_to_top();
    app.draw(&mut terminal).expect("redraw at top");
    let transcript = app.regions.expect("layout regions after draw").transcript;
    let (col, top) = (transcript.x.saturating_add(1), transcript.y);

    let press = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: col,
            row: top,
            modifiers: KeyModifiers::NONE,
        })
        .expect("press handled");
    assert_eq!(press, AppAction::None);
    let drag = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: col.saturating_add(20),
            row: top.saturating_add(1),
            modifiers: KeyModifiers::NONE,
        })
        .expect("drag handled");
    assert_eq!(drag, AppAction::Redraw, "a real drag repaints (and mines)");
    app.draw(&mut terminal).expect("mine dragged rows");

    // Two wheel notches while the button is held: each must keep the
    // gesture, scroll, and request the immediate repaint that mines the
    // newly revealed rows.
    for _ in 0..2 {
        let wheel = app
            .handle_mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: col.saturating_add(20),
                row: top.saturating_add(1),
                modifiers: KeyModifiers::NONE,
            })
            .expect("wheel handled");
        assert_eq!(
            wheel,
            AppAction::Redraw,
            "mid-drag wheel must repaint immediately, not coalesce"
        );
        assert!(
            app.transcript.has_char_selection(),
            "mid-drag wheel must keep the selection"
        );
        app.draw(&mut terminal).expect("mine revealed rows");
    }
    assert!(
        app.transcript.scroll() > 0,
        "the wheel really scrolled while dragging"
    );

    let release = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: col.saturating_add(20),
            row: top.saturating_add(1),
            modifiers: KeyModifiers::NONE,
        })
        .expect("release handled");
    match release {
        AppAction::ClipboardCopyBlock(text) => {
            assert!(
                text.contains("row alpha"),
                "rows scrolled out of the viewport must stay in the copy: {text:?}"
            );
            assert!(
                text.contains("row charlie"),
                "rows revealed by the mid-drag wheel must join the copy: {text:?}"
            );
        }
        other => panic!("wheel-extended drag must copy on release, got {other:?}"),
    }
}

#[test]
fn click_on_tool_call_expands_matching_result() {
    let mut app = test_app();
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call_1".to_string()),
        name: "Bash".to_string(),
        summary: "cargo test".to_string(),
        preview: ToolPreview::Bash {
            command: "cargo test".to_string(),
        },
        status: ToolCallStatus::Ok,
    });
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call_1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "many\nlines\nof\noutput".to_string(),
            truncated: true,
        },
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;
    let (column, row) = (transcript.x.saturating_add(1), transcript.y);

    assert!(!app.transcript.is_expanded(1));
    assert_eq!(
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("press handled"),
        AppAction::None
    );
    assert_eq!(
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("release handled"),
        AppAction::Redraw
    );
    assert!(app.transcript.is_expanded(1));
}

#[test]
fn hover_reveals_copy_button_and_clicking_button_copies_block() {
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(42),
        text: "copy this block".to_string(),
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;

    let hover = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Moved,
            column: transcript.x.saturating_add(1),
            row: transcript.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("hover handled");
    assert_eq!(hover, AppAction::Redraw);

    app.draw(&mut terminal).expect("redraw shows copy button");
    let button = app
        .transcript_view
        .hovered_copy_button
        .expect("hover target recorded")
        .button;
    assert!(
        buffer_row(&terminal, 120, button.y).contains("⧉"),
        "hover should reveal block-level copy affordance"
    );

    let action = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: button.x,
            row: button.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("copy button click handled");

    assert_eq!(
        action,
        AppAction::ClipboardCopyBlock("copy this block".to_string())
    );
}

#[test]
fn copy_button_click_uses_hovered_block_id_not_click_row() {
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(1),
        text: "first block".to_string(),
    });
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(2),
        text: "second block".to_string(),
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Moved,
        column: transcript.x.saturating_add(1),
        row: transcript.y,
        modifiers: KeyModifiers::NONE,
    })
    .expect("hover handled");
    let button = app
        .transcript_view
        .hovered_copy_button
        .expect("hover target recorded")
        .button;

    app.transcript_view.hovered_copy_button = Some(super::HoveredCopyButton {
        block_id: BlockId(2),
        button,
    });

    let action = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: button.x,
            row: button.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("copy button click handled");

    assert_eq!(
        action,
        AppAction::ClipboardCopyBlock("second block".to_string())
    );
}

#[test]
fn hover_omits_copy_button_for_copy_hostile_tool_call_metadata() {
    let mut app = test_app();
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(42),
        tool_call_id: ToolCallId("call-secret".to_string()),
        name: "bash".to_string(),
        summary: "SECRET_INPUT_METADATA".to_string(),
        preview: ToolPreview::Generic {
            name: "bash".to_string(),
            input_summary: "SECRET_PREVIEW_METADATA".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;

    let hover = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Moved,
            column: transcript.x.saturating_add(1),
            row: transcript.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("hover handled");

    assert_eq!(hover, AppAction::None);
    assert_eq!(app.transcript_view.hovered_copy_button, None);
}

#[test]
fn copy_range_skips_tool_call_and_image_and_copies_only_text_bodies() {
    let mut app = test_app();
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(10),
        text: "assistant body".to_string(),
        done: true,
    });
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(11),
        tool_call_id: ToolCallId("call-poison".to_string()),
        name: "bash".to_string(),
        summary: "SECRET_INPUT_METADATA".to_string(),
        preview: ToolPreview::Generic {
            name: "bash".to_string(),
            input_summary: "SECRET_PREVIEW_METADATA".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    app.push_block(RenderBlock::Image {
        id: BlockId(12),
        data: vec![1, 2, 3],
        media_type: "image/png SECRET_IMAGE_METADATA".to_string(),
    });
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(13),
        tool_call_id: ToolCallId("call-text".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "tool text body".to_string(),
            truncated: false,
        },
    });
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(14),
        tool_call_id: ToolCallId("call-read".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "secret/path.rs".to_string(),
            content: "read body only".to_string(),
            language: Some("rust".to_string()),
            truncated: false,
        },
    });
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(15),
        tool_call_id: ToolCallId("call-todo".to_string()),
        is_error: false,
        body: ToolResultBody::Todos(vec![runtime::message_stream::TodoResultItem {
            content: "Finish task".to_string(),
            active_form: "Finishing task".to_string(),
            status: TodoResultStatus::InProgress,
        }]),
    });

    let copied = app
        .transcript
        .copy_text_for_block_range(BlockId(10), BlockId(15))
        .expect("range has textual payloads");

    assert_eq!(
        copied,
        "assistant body\n\ntool text body\n\nread body only\n\n[~] Finishing task"
    );
    assert!(!copied.contains("SECRET_INPUT_METADATA"));
    assert!(!copied.contains("SECRET_PREVIEW_METADATA"));
    assert!(!copied.contains("SECRET_IMAGE_METADATA"));
    assert!(!copied.contains("secret/path.rs"));
}

#[test]
fn left_drag_copies_visible_character_selection_on_release() {
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(42),
        text: "copy this visible text".to_string(),
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.transcript_view_rect().expect("transcript viewport");
    let row = (transcript.y..transcript.y + transcript.height)
        .find(|row| buffer_row(&terminal, 120, *row).contains("copy this visible text"))
        .expect("text row is visible");
    let col_start = transcript.x;
    let col_end = transcript.x + transcript.width - 1;
    let expected = {
        let buffer = terminal.backend().buffer();
        (col_start..=col_end).fold(String::new(), |mut text, col| {
            text.push_str(buffer[(col, row)].symbol());
            text
        })
    };
    let expected = expected.trim_end().to_string();

    let down = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: col_start,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("press handled");
    assert_eq!(down, AppAction::None, "a press alone never copies");

    let drag = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: col_end,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("drag handled");
    assert_eq!(drag, AppAction::Redraw);
    app.draw(&mut terminal)
        .expect("draw mines selected cells from the frame");

    let up = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: col_end,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("release handled");
    let AppAction::ClipboardCopyBlock(copied) = up else {
        panic!("drag release should copy selected cells, got {up:?}");
    };
    assert!(!copied.is_empty());
    assert!(copied.contains("copy this visible text"), "copied: {copied:?}");
    assert_eq!(copied, expected, "clipboard text matches the visible row");
}

#[test]
fn wheel_scroll_keeps_settled_selection_highlight() {
    // 릴리즈로 확정된 하이라이트는 콘텐츠 행에 앵커되어 스크롤을 그대로
    // 따라간다: 드래그 밖의 휠은 예전처럼 일반 스크롤(None, 코얼레스 경로)로
    // 흐르되 더 이상 하이라이트를 지우지 않는다 (화면 고정 좌표이던 시절의
    // 클리어 계약을 대체).
    let (mut app, mut terminal) = app_with_overflowing_transcript();
    let transcript = app.transcript_view_rect().expect("transcript viewport");
    let column = transcript.x + 1;
    let row = transcript.y + 1;

    // Sweep three rows: the block grid alternates content and gap rows, so a
    // single-row gesture could land on a blank row and mine nothing.
    for (kind, event_row) in [
        (MouseEventKind::Down(MouseButton::Left), row),
        (MouseEventKind::Drag(MouseButton::Left), row + 2),
    ] {
        app.handle_mouse(MouseEvent {
            kind,
            column,
            row: event_row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("selection gesture handled");
    }
    assert!(app.transcript.has_char_selection());
    app.draw(&mut terminal).expect("mine the dragged cells");
    let release = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column,
            row: row + 2,
            modifiers: KeyModifiers::NONE,
        })
        .expect("release handled");
    assert!(
        matches!(release, AppAction::ClipboardCopyBlock(_)),
        "the drag release copies, got {release:?}"
    );
    assert!(app.transcript.has_char_selection(), "highlight persists");

    let wheel = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("wheel handled");
    assert_eq!(
        wheel,
        AppAction::None,
        "outside a drag the wheel stays on the coalesced scroll path"
    );
    assert!(
        app.transcript.has_char_selection(),
        "a settled highlight survives scrolling (content-row anchored)"
    );
}

#[test]
fn next_draw_keeps_character_selection_after_scroll_offset_changes() {
    // 선택은 콘텐츠 행에 앵커된다: 스크롤 오프셋만 바뀐 다음 draw 는
    // 하이라이트를 유지한 채 새 위치에 다시 칠한다 (화면 고정 + 스크롤 핀
    // 시절에는 여기서 드롭됐다).
    let (mut app, mut terminal) = app_with_overflowing_transcript();
    let transcript = app.transcript_view_rect().expect("transcript viewport");
    let column = transcript.x + 1;
    let row = transcript.y + 1;

    for (kind, col) in [
        (MouseEventKind::Down(MouseButton::Left), column),
        (MouseEventKind::Drag(MouseButton::Left), column + 1),
    ] {
        app.handle_mouse(MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("selection gesture handled");
    }
    assert!(app.transcript.has_char_selection());

    let scroll_before = app.transcript.scroll();
    app.transcript.scroll_up(1);
    assert_ne!(app.transcript.scroll(), scroll_before);
    app.draw(&mut terminal).expect("draw after direct scroll change");
    assert!(
        app.transcript.has_char_selection(),
        "scrolling alone must not drop a content-anchored selection"
    );
}

#[test]
fn new_press_clears_persisted_character_selection_highlight() {
    let mut app = test_app();
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(42),
        text: "persisted highlight".to_string(),
    });
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.transcript_view_rect().expect("transcript viewport");
    let row = (transcript.y..transcript.y + transcript.height)
        .find(|row| buffer_row(&terminal, 120, *row).contains("persisted highlight"))
        .expect("text row is visible");
    let col_start = transcript.x;
    let col_end = transcript.x + transcript.width - 1;

    for (kind, col) in [
        (MouseEventKind::Down(MouseButton::Left), col_start),
        (MouseEventKind::Drag(MouseButton::Left), col_end),
    ] {
        app.handle_mouse(MouseEvent {
            kind,
            column: col,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("selection gesture handled");
    }
    app.draw(&mut terminal).expect("draw selection highlight");
    let copied = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: col_end,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("release handled");
    assert!(matches!(copied, AppAction::ClipboardCopyBlock(_)));
    assert!(app.transcript.has_char_selection());

    let next_press = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: col_start,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("next press handled");
    assert_eq!(next_press, AppAction::Redraw);
    assert!(!app.transcript.has_char_selection());
}

#[test]
fn left_click_on_hud_toggles_sidebar() {
    let mut app = test_app();
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let hud = app.regions.expect("layout regions after draw").hud;
    let initially_visible = app.sidebar.visible;

    let action = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: hud.x.saturating_add(1),
            row: hud.y,
            modifiers: KeyModifiers::NONE,
        })
        .expect("mouse handled");

    assert_eq!(action, AppAction::Redraw);
    assert_eq!(app.sidebar.visible, !initially_visible);
}

#[test]
fn tab_focuses_block_enter_expands_esc_clears_focus() {
    // Pins the keybinding-overlay promise: Tab focuses the next block, Enter
    // expands/collapses the focused block, Esc drops the focus — and the
    // composer-safety gates around them.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });
    // Settle the turn so Esc reaches the focus-clear branch rather than the
    // (higher-priority) running-turn interrupt, and the block is flushed into
    // the transcript so it is focusable.
    app.end_turn();

    assert_eq!(app.transcript.focused_idx(), None, "no focus initially");

    // Tab on an empty composer focuses the (only) interactable block.
    let action = app.handle_key(press(KeyCode::Tab)).expect("tab");
    let focused = app
        .transcript
        .focused_idx()
        .expect("Tab focuses a block on an empty composer");
    assert!(matches!(action, AppAction::Redraw), "focus repaint");
    let before = app.transcript.is_expanded(focused);

    // Enter while a block is focused toggles its expansion (not submit).
    let action = app.handle_key(press(KeyCode::Enter)).expect("enter");
    assert!(matches!(action, AppAction::Redraw), "expand repaint");
    assert_ne!(
        app.transcript.is_expanded(focused),
        before,
        "Enter toggled the focused block's expansion"
    );

    // Esc clears the focus, restoring composer Enter/Tab semantics.
    let action = app.handle_key(press(KeyCode::Esc)).expect("esc");
    assert!(matches!(action, AppAction::Redraw), "clear-focus repaint");
    assert_eq!(app.transcript.focused_idx(), None, "Esc cleared focus");
}

#[test]
fn tab_does_not_steal_focus_while_composer_has_text() {
    // Regression guard: with a non-empty composer Tab must still fall through
    // to slash-completion / the input widget, never stealing block focus.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });

    let _ = app.handle_key(press(KeyCode::Char('h')));
    assert!(!app.input.text().is_empty(), "composer holds text");

    let _ = app.handle_key(press(KeyCode::Tab));
    assert_eq!(
        app.transcript.focused_idx(),
        None,
        "Tab must not focus a block while the composer holds text"
    );
}

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn press_with(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

/// Shift+Tab must cycle the permission mode under BOTH encodings: legacy
/// terminals deliver `BackTab`, while the Kitty keyboard protocol
/// (`DISAMBIGUATE_ESCAPE_CODES`, which `init_terminal` pushes when supported)
/// delivers `Tab` + SHIFT. Before the fix, only `BackTab` was handled, so the
/// cycle silently did nothing on kitty/foot/WezTerm/Ghostty/`iTerm2`.
#[test]
fn shift_tab_cycles_permission_under_both_key_encodings() {
    let legacy = test_app()
        .handle_key(press(KeyCode::BackTab))
        .expect("backtab handled");
    let kitty = test_app()
        .handle_key(press_with(KeyCode::Tab, KeyModifiers::SHIFT))
        .expect("tab+shift handled");

    assert!(
        matches!(legacy, AppAction::SelectPermission(_)),
        "legacy BackTab must cycle permission, got {legacy:?}"
    );
    assert_eq!(
        legacy, kitty,
        "Kitty-protocol Tab+SHIFT must cycle identically to legacy BackTab"
    );

    // Plain Tab (no shift) must NOT be a permission cycle — it stays available
    // for slash/mention completion.
    let plain = test_app()
        .handle_key(press(KeyCode::Tab))
        .expect("plain tab handled");
    assert!(
        !matches!(plain, AppAction::SelectPermission(_)),
        "plain Tab must not cycle permission, got {plain:?}"
    );
}

/// Shift+Tab cycles `ReadOnly → Plan → Workspace → All → ReadOnly`. `Plan` is
/// the read-only-backed planning stop: the key emits a runtime `SelectPermission`
/// and toggles `plan_mode_active`, and feeding the runtime mode back through
/// `set_session_meta` (as the host loop does) resolves the visible HUD badge.
/// Before plan mode was surfaced the cycle was the 3-arm
/// `ReadOnly → Workspace → All`, so the first step landed on Workspace — this
/// test fails on that older cycle.
#[test]
fn shift_tab_cycle_visits_plan_between_read_only_and_workspace() {
    fn step(app: &mut App, cwd: &std::path::Path) -> PermissionMode {
        let action = app
            .handle_key(press(KeyCode::BackTab))
            .expect("Shift+Tab handled");
        let AppAction::SelectPermission(mode) = action else {
            panic!("expected SelectPermission, got {action:?}");
        };
        app.set_session_meta("gpt-5.5", 258_000, mode, cwd.to_path_buf(), None);
        app.perm_mode()
    }

    let mut app = test_app();
    let cwd = std::path::PathBuf::from("/tmp");
    // Normalize the starting badge to ReadOnly regardless of the test default.
    app.set_session_meta(
        "gpt-5.5",
        258_000,
        runtime::PermissionMode::ReadOnly,
        cwd.clone(),
        None,
    );
    assert_eq!(app.perm_mode(), PermissionMode::ReadOnly);
    assert_eq!(step(&mut app, &cwd), PermissionMode::Plan);
    assert_eq!(step(&mut app, &cwd), PermissionMode::Workspace);
    assert_eq!(step(&mut app, &cwd), PermissionMode::All);
    assert_eq!(step(&mut app, &cwd), PermissionMode::ReadOnly);
}

#[test]
fn test_hud_state_sync_preserves_plan_mode() {
    let mut app = test_app();
    let cwd = std::path::PathBuf::from("/tmp");

    // Start with app HUD ReadOnly
    app.set_session_meta(
        "gpt-5.5",
        258_000,
        runtime::PermissionMode::ReadOnly,
        cwd.clone(),
        None,
    );
    assert_eq!(app.perm_mode(), PermissionMode::ReadOnly);

    // Set plan active directly
    app.set_plan_mode_active(true);
    app.hud_state.perm_mode = PermissionMode::Plan;
    assert_eq!(app.perm_mode(), PermissionMode::Plan);

    // Prepare incoming HudState with ReadOnly
    let mut incoming_hud = app.hud_state.clone();
    incoming_hud.perm_mode = PermissionMode::ReadOnly;

    // Call set_hud_state with incoming HUD ReadOnly
    app.set_hud_state(incoming_hud);

    // Assert HUD remains Plan
    assert_eq!(app.perm_mode(), PermissionMode::Plan);
    assert!(app.plan_mode_active());

    // Call handle_key with BackTab to verify it cycles to WorkspaceWrite
    let action = app
        .handle_key(press(KeyCode::BackTab))
        .expect("BackTab handled");
    assert_eq!(
        action,
        AppAction::SelectPermission(super::RuntimePermissionMode::WorkspaceWrite)
    );

    // Call set_hud_state with Workspace
    let mut incoming_hud_workspace = app.hud_state.clone();
    incoming_hud_workspace.perm_mode = PermissionMode::Workspace;
    app.set_hud_state(incoming_hud_workspace);

    // Assert plan flag is cleared and HUD reflects Workspace
    assert_eq!(app.perm_mode(), PermissionMode::Workspace);
    assert!(!app.plan_mode_active());
}

/// `/plan on` remembers the mode being left and `/plan off` restores exactly it
/// (never a blind `ReadOnly` downgrade). Exercises the `App::enter_plan_mode` /
/// `exit_plan_mode` state the `session_cmds` `/plan` handler drives. A naive
/// restore-to-`ReadOnly` implementation fails the `restored == All` assertion.
#[test]
fn plan_mode_enter_remembers_prior_and_exit_restores_it() {
    let mut app = test_app();
    let cwd = std::path::PathBuf::from("/tmp");
    app.set_session_meta(
        "gpt-5.5",
        258_000,
        runtime::PermissionMode::DangerFullAccess,
        cwd,
        None,
    );
    assert_eq!(app.perm_mode(), PermissionMode::All);

    app.enter_plan_mode();
    assert_eq!(app.perm_mode(), PermissionMode::Plan);
    assert!(app.plan_mode_active());

    // A re-entrant `/plan on` keeps the ORIGINAL prior (All), not `Plan`.
    app.enter_plan_mode();

    let restored = app.exit_plan_mode();
    assert_eq!(
        restored,
        PermissionMode::All,
        "exit restores the pre-plan mode, not ReadOnly"
    );
    assert_eq!(app.perm_mode(), PermissionMode::All);
    assert!(!app.plan_mode_active());
}

/// A plan transition mutates the App before the runtime permission change is
/// applied. If that change fails, the transition must be rolled back so the UI
/// Plan flag never diverges from the runtime. This proves the pure snapshot /
/// restore contract the `/plan` handler and Shift+Tab cycle rely on: a snapshot
/// taken before `enter_plan_mode` restores the exact prior `plan_mode_active`,
/// remembered prior mode, and HUD badge.
#[test]
fn plan_mode_snapshot_restore_rolls_back_a_failed_transition_exactly() {
    let mut app = test_app();
    let cwd = std::path::PathBuf::from("/tmp");
    app.set_session_meta(
        "gpt-5.5",
        258_000,
        runtime::PermissionMode::WorkspaceWrite,
        cwd,
        None,
    );
    assert_eq!(app.perm_mode(), PermissionMode::Workspace);
    assert!(!app.plan_mode_active());

    // Snapshot the pre-transition state, then optimistically enter Plan as the
    // host loop does before calling the runtime.
    let snapshot = app.plan_mode_snapshot();
    app.enter_plan_mode();
    assert_eq!(app.perm_mode(), PermissionMode::Plan);
    assert!(app.plan_mode_active());

    // Runtime permission change "fails": roll back to the snapshot.
    app.restore_plan_mode_snapshot(snapshot);
    assert_eq!(
        app.perm_mode(),
        PermissionMode::Workspace,
        "failed transition restores the exact prior HUD badge"
    );
    assert!(
        !app.plan_mode_active(),
        "failed transition leaves the plan flag unchanged"
    );

    // A subsequent clean enter/exit still behaves normally after a rollback:
    // the remembered prior mode was not corrupted by the aborted attempt.
    app.enter_plan_mode();
    let restored = app.exit_plan_mode();
    assert_eq!(restored, PermissionMode::Workspace);
    assert!(!app.plan_mode_active());
}

/// The Shift+Tab cycle arms a rollback snapshot before it mutates the plan-gate
/// (the host loop restores it if the runtime change fails and takes it on
/// success). This proves the arm/take handshake is exact: arming captures the
/// pre-mutation state, and taking it twice yields the snapshot once then `None`.
#[test]
fn plan_cycle_rollback_arms_and_is_taken_once() {
    let mut app = test_app();
    let cwd = std::path::PathBuf::from("/tmp");
    app.set_session_meta(
        "gpt-5.5",
        258_000,
        runtime::PermissionMode::ReadOnly,
        cwd,
        None,
    );

    // No cycle in flight: nothing to take.
    assert!(app.take_plan_cycle_rollback().is_none());

    // Arm, then mutate into Plan (as keys.rs does on Shift+Tab).
    app.arm_plan_cycle_rollback();
    app.enter_plan_mode();
    assert!(app.plan_mode_active());

    // The host loop takes the snapshot exactly once; a second take is empty.
    let snapshot = app
        .take_plan_cycle_rollback()
        .expect("armed rollback is available");
    assert!(app.take_plan_cycle_rollback().is_none());

    // Restoring it undoes the cycle's mutation.
    app.restore_plan_mode_snapshot(snapshot);
    assert!(!app.plan_mode_active());
    assert_eq!(app.perm_mode(), PermissionMode::ReadOnly);
}

fn buffer_row(terminal: &Terminal<TestBackend>, width: u16, y: u16) -> String {
    let buffer = terminal.backend().buffer();
    (0..width).fold(String::new(), |mut out, x| {
        out.push_str(buffer[(x, y)].symbol());
        out
    })
}

#[test]
fn wide_terminal_fills_chat_column_up_to_the_sidebar_no_dead_gutter() {
    // Regression (this fix): on a wide terminal the chat content column
    // (transcript / rules / input / HUD) must fill every cell from the left
    // edge up to the sidebar's left edge — no dead gutter stranded between the
    // chat and the HUD ledger. Before the column cap was removed, the chat
    // stopped at 100 cols and the leftover span (here ~64 cells) was quiet
    // void — the "빈 공백" the user reported.
    let mut app = test_app();
    app.sidebar.visible = true;

    let width = 200;
    let backend = TestBackend::new(width, 40);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert!(regions.sidebar_width > 0, "200 cols hosts a sidebar");
    // The chat column starts at the left edge and runs right up to the sidebar.
    assert_eq!(regions.hud.x, 0, "chat column stays left-anchored");
    assert_eq!(
        regions.hud.x + regions.hud.width,
        regions.sidebar.x,
        "the chat column fills up to the sidebar's left edge — no dead gutter"
    );
    // The transcript shares the exact same span (one aligned document).
    assert_eq!(regions.transcript.width, regions.hud.width);
    assert_eq!(
        regions.transcript.width + regions.sidebar_width,
        width,
        "chat column + sidebar tile the full terminal width with no gap"
    );
    // The column is now much wider than the old 100-col cap.
    assert!(
        regions.transcript.width > 100,
        "wide terminal must exceed the retired 100-col cap: {}",
        regions.transcript.width
    );
}

#[test]
fn workflow_hud_height_is_stable_across_turn_boundaries() {
    // Regression (this fix): the dedicated workflow HUD row is granted for the
    // whole active workflow, independent of whether a turn is currently
    // streaming. Removing the column cap made the wide activity row able to
    // host the inline phase badge, so a turn-aware grant would flap the HUD
    // height 1↔2 on every turn boundary. Guard that the height is identical
    // whether or not a turn is active.
    let make_workflow = || WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    };
    let width = 160;

    // Idle (no turn) with an active workflow.
    let mut idle = test_app();
    idle.sidebar.visible = false;
    idle.hud_state.workflow = Some(make_workflow());
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    idle.draw(&mut terminal).expect("draw idle");
    let idle_hud_h = idle.regions.expect("idle regions").hud.height;

    // Same state, but a turn is streaming.
    let mut active = test_app();
    active.sidebar.visible = false;
    active.hud_state.workflow = Some(make_workflow());
    active.begin_turn_with_generation(0);
    active.set_turn_activity("Running command: cargo test");
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    active.draw(&mut terminal).expect("draw active");
    let active_hud_h = active.regions.expect("active regions").hud.height;

    assert_eq!(
        idle_hud_h, active_hud_h,
        "HUD height must not flap between idle and active turns: idle={idle_hud_h} active={active_hud_h}"
    );
    assert_eq!(idle_hud_h, 2, "an active workflow keeps the dedicated row");
}

#[test]
fn live_activity_row_shows_model_and_workflow_context() {
    let mut app = test_app();
    // No sidebar: the dedicated HUD workflow row is only granted when no
    // other surface (sidebar workflow section) owns the phase.
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.update_turn_tokens(1_100, 240);
    app.hud_state.model = ActiveModel {
        provider: "openai",
        alias: "gpt".to_string(),
        display_name: "gpt-5.5-fast".to_string(),
        context_limit: 258_000,
    };
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });
    app.push_block(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call-1".to_string()),
        name: "bash".to_string(),
        summary: r#"{"command":"cargo test"}"#.to_string(),
        preview: ToolPreview::Bash {
            command: "cargo test".to_string(),
        },
        status: ToolCallStatus::Running,
    });

    let width = 160;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");
    let row = buffer_row(&terminal, width, regions.rule_top.y);

    assert!(
        row.contains("Running command"),
        "activity action should stay visible: {row:?}"
    );
    assert!(
        row.contains("gpt-5.5-fast"),
        "resolved model context missing from live activity row: {row:?}"
    );
    // The workflow phase has its stable home on the HUD's dedicated workflow
    // row (granted whenever a workflow is active, no sidebar owns it, and the
    // terminal is tall enough), not the live activity row — surfacing it in
    // both places would double-render it now that the chat column fills the
    // full width.
    assert!(
        !row.contains("phase 2/4"),
        "activity row leaves the phase to the dedicated HUD row: {row:?}"
    );
    assert_eq!(
        regions.hud.height, 2,
        "an active workflow grants the two-row HUD"
    );
    let workflow_row = buffer_row(&terminal, width, regions.hud.y);
    assert!(
        workflow_row.contains("phase 2/4 read-code"),
        "workflow phase context missing from the dedicated HUD row: {workflow_row:?}"
    );
    assert!(
        workflow_row.contains("25%"),
        "workflow completion percentage missing from the dedicated HUD row: {workflow_row:?}"
    );
}

#[test]
fn set_session_meta_resolves_model_provider_alias_for_attach_sessions() {
    let mut app = test_app();
    // gpt-5.5 퇴역 후 bare 세대 별칭의 대표는 gpt-5.6(→terra)이다.
    app.set_session_meta(
        "gpt-5.6",
        258_000,
        runtime::PermissionMode::WorkspaceWrite,
        PathBuf::from("/tmp"),
        Some("main".to_string()),
    );

    assert_eq!(app.hud_state.model.provider, "openai");
    assert_eq!(app.hud_state.model.alias, "gpt-5.6");
    assert_eq!(
        app.hud_state.model.display_name,
        "gpt-5.6-terra".to_string()
    );
    assert_eq!(app.hud_state.perm_mode, PermissionMode::Workspace);
    assert_eq!(app.hud_state.ctx_limit, 258_000);
    assert_eq!(app.hud_state.cwd, PathBuf::from("/tmp"));
    assert_eq!(app.hud_state.git_branch, Some("main".to_string()));
}

#[test]
fn live_activity_context_collapses_bottom_hud_duplicates() {
    let mut app = test_app();
    app.sidebar.visible = true;
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Running command: cargo test");
    app.update_turn_tokens(1_100, 240);
    app.hud_state.model = ActiveModel {
        provider: "openai",
        alias: "gpt".to_string(),
        display_name: "gpt-5.5-fast".to_string(),
        context_limit: 258_000,
    };
    app.hud_state.ctx_used = 82_560;
    app.hud_state.ctx_limit = 258_000;
    app.hud_state.compact_threshold = 206_400;
    app.hud_state.cost_usd = 6.20;
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });

    let width = 160;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");
    assert!(regions.sidebar_width > 0, "sidebar must participate in the three-way check");
    let screen = (0..24)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");
    let activity = buffer_row(&terminal, width, regions.rule_top.y);
    let hud = buffer_row(&terminal, width, regions.hud.y);

    assert!(
        activity.contains("Running command") && activity.contains("gpt-5.5-fast"),
        "top activity row should own live context: {activity:?}"
    );
    assert!(
        activity.contains("40% ctx"),
        "activity row uses auto-compaction pressure: {activity:?}"
    );
    assert!(
        screen.contains("2/4 read-code"),
        "visible sidebar owns workflow phase context: {screen:?}"
    );
    assert!(
        hud.contains("gpt-5.5-fast"),
        "session HUD keeps current model identity after the heat marker: {hud:?}"
    );
    assert!(
        hud.contains("ctx 40%"),
        "bottom HUD uses the same auto-compaction pressure: {hud:?}"
    );
    assert!(
        screen.contains("use  40%"),
        "sidebar uses the same auto-compaction pressure: {screen:?}"
    );
    // Entries owned by the activity/workflow surfaces stay absent from the
    // session HUD even though both context labels now share one calculation.
    for duplicate in [
        "phase 2/4",
        "danger-full-access",
        "workspace-write",
        "$6.20",
        "tokens",
    ] {
        assert!(
            !hud.contains(duplicate),
            "bottom HUD must not duplicate top activity context {duplicate:?}: {hud:?}"
        );
    }
}

#[test]
fn workflow_phase_survives_short_wide_terminals() {
    // Regression (short-wide terminal): on a wide-but-short terminal (below
    // the old height≥20 comfort gate) the activity row deliberately leaves the
    // phase to the HUD, so the dedicated HUD row is the phase's only home — it
    // must still be granted.
    let mut app = test_app();
    app.sidebar.visible = false;
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });

    let width = 160;
    let backend = TestBackend::new(width, 18);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert_eq!(
        regions.hud.height, 2,
        "short-but-wide terminals still get the dedicated workflow row"
    );
    let workflow_row = buffer_row(&terminal, width, regions.hud.y);
    assert!(
        workflow_row.contains("phase 2/4"),
        "workflow phase must never disappear from the screen: {workflow_row:?}"
    );
}

#[test]
fn tiny_terminal_shows_truncated_workflow_badge_on_single_row_hud() {
    // Below the two-row grant floor (height < 10) the single status row is
    // the phase's last home: the badge now truncates to the available cells
    // instead of vanishing when the full label doesn't fit the status row
    // (adversarial review: the height 7..9 wide band used to lose the phase).
    let mut app = test_app();
    app.sidebar.visible = false;
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });

    let width = 160;
    let backend = TestBackend::new(width, 8);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert_eq!(
        regions.hud.height, 1,
        "below the grant floor the HUD stays single-row"
    );
    let hud = buffer_row(&terminal, width, regions.hud.y);
    assert!(
        hud.contains("phase 2/4"),
        "single-row HUD carries a truncated phase badge: {hud:?}"
    );
}

#[test]
fn wheel_scroll_reaches_the_tail_behind_pinned_overlays() {
    // Regression (wheel-clip): the scroll clamp used the FULL transcript
    // region while the body drew into the overlay-reduced rect, so the last
    // `bottom_reserved` rows could never be wheeled into view behind the
    // pinned live-plan panel. `transcript_draw_rect` keeps interaction and
    // paint on one viewport.
    let mut app = test_app();
    app.sidebar.visible = false;
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Running plan");
    app.hud_state.todo_items = (0..6)
        .map(|i| crate::tui::hud::TodoChecklistItem {
            step_id: None,
            content: format!("task {i}"),
            status: if i == 0 {
                crate::tui::hud::TodoChecklistStatus::InProgress
            } else {
                crate::tui::hud::TodoChecklistStatus::Pending
            },
            active_form: format!("doing task {i}"),
        })
        .collect();
    for i in 0..30 {
        app.push_block(RenderBlock::UserMessage {
            id: BlockId(5000 + i),
            text: format!("filler row {i}"),
        });
    }
    app.push_block(RenderBlock::UserMessage {
        id: BlockId(5990),
        text: "TAIL_MARKER_LAST".to_string(),
    });

    let (width, height) = (80u16, 20u16);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let marker_visible = |terminal: &Terminal<TestBackend>| {
        (0..height).any(|y| buffer_row(terminal, width, y).contains("TAIL_MARKER_LAST"))
    };
    assert!(
        marker_visible(&terminal),
        "follow-tail starts with the last row on screen"
    );

    let wheel = |app: &mut App, kind: crossterm::event::MouseEventKind| {
        let _ = app.handle_mouse(crossterm::event::MouseEvent {
            kind,
            column: 10,
            row: 5,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
    };
    wheel(&mut app, crossterm::event::MouseEventKind::ScrollUp);
    app.draw(&mut terminal).expect("draw after wheel up");
    for _ in 0..40 {
        wheel(&mut app, crossterm::event::MouseEventKind::ScrollDown);
        app.draw(&mut terminal).expect("draw after wheel down");
    }
    assert!(
        marker_visible(&terminal),
        "wheel-down must reach the last transcript row behind the pinned plan panel"
    );
}

#[test]
fn sidebar_owns_workflow_detail_so_hud_stays_single_row() {
    // The rendered sidebar carries the full workflow section (second from its
    // top) — a dedicated HUD workflow row would spend a transcript row
    // repeating information that is already on screen.
    let mut app = test_app();
    app.sidebar.visible = true;
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });

    let width = 160;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert!(regions.sidebar_width > 0, "sidebar renders at 160 cols");
    assert_eq!(
        regions.hud.height, 1,
        "sidebar owns the workflow detail — no dedicated HUD row"
    );
}

#[test]
fn short_sidebar_cannot_carry_the_phase_so_hud_keeps_workflow_row() {
    // Regression (Phase C adversarial review, D3): with the sidebar visible on
    // a short terminal, the top-anchored unscrollable body clips the workflow
    // section (header + session block sit above it) — suppressing the
    // dedicated HUD row there would lose the phase entirely.
    let mut app = test_app();
    app.sidebar.visible = true;
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });

    let width = 160;
    let backend = TestBackend::new(width, 12);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert!(regions.sidebar_width > 0, "sidebar renders at 160 cols");
    assert_eq!(
        regions.hud.height, 2,
        "clipped sidebar cannot carry the phase — HUD keeps the dedicated row"
    );
    let workflow_row = buffer_row(&terminal, width, regions.hud.y);
    assert!(
        workflow_row.contains("phase 2/4"),
        "workflow phase must never disappear from the screen: {workflow_row:?}"
    );
}

#[test]
fn wrapped_sidebar_session_lines_keep_the_hud_workflow_row() {
    // Regression (Phase C adversarial review, D3b): the sidebar body renders
    // with Wrap and clamps by *wrapped row count* — a rate-limit gauge row
    // (indent + label + bar + percent + reset) soft-wraps in the narrow panel
    // and pushes the workflow section off screen even when the plain line
    // count says it fits. The on-screen probe must count wrapped rows, or
    // the phase disappears from both surfaces. (Branch names used to be a
    // second wrap vector; the header now truncates them, so the gauge rows
    // carry this regression.)
    let mut app = test_app();
    app.sidebar.visible = true;
    app.hud_state.rate_limit = Some(RateLimitSnapshot {
        five_hour: Some(RateLimitWindow {
            utilization: 0.42,
            resets_at_unix: Some(1_800_000_000),
        }),
        seven_day: Some(RateLimitWindow {
            utilization: 0.63,
            resets_at_unix: Some(1_800_500_000),
        }),
        representative: None,
    });
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "read-code".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 4,
        next_phase: Some("verify".to_string()),
        total_agents: 8,
        progress_percent: 25,
        completed_phases: 1,
        completed_agents: 2,
        failed_agents: 0,
        running_agents: 6,
        phases: Vec::new(),
    });

    let width = 160;
    // The compact reset labels and two-row footer let this state fit at height
    // 16; height 15 keeps the rate-limit boundary where the HUD owns the phase.
    let backend = TestBackend::new(width, 15);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert!(regions.sidebar_width > 0, "sidebar renders at 160 cols");
    assert_eq!(
        regions.hud.height, 2,
        "wrap-displaced sidebar cannot carry the phase — HUD keeps the row"
    );
    let workflow_row = buffer_row(&terminal, width, regions.hud.y);
    assert!(
        workflow_row.contains("phase 2/4"),
        "workflow phase must never disappear from the screen: {workflow_row:?}"
    );
}

#[test]
fn two_row_hud_masks_the_inline_workflow_badge() {
    // Regression (two-row HUD, D2): the dedicated first row owns the workflow
    // phase, while the quiet session row below must suppress its inline badge.
    let mut app = test_app();
    app.sidebar.visible = false;
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "wf".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "fix".to_string(),
        current_phase_status: String::new(),
        current_phase_index: 1,
        total_phases: 2,
        next_phase: None,
        total_agents: 1,
        progress_percent: 5,
        completed_phases: 0,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 1,
        phases: Vec::new(),
    });

    let width = 120;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let regions = app.regions.expect("layout regions after draw");

    assert_eq!(regions.hud.height, 2, "workflow grants the two-row HUD");
    let workflow_row = buffer_row(&terminal, width, regions.hud.y);
    let session_row = buffer_row(&terminal, width, regions.hud.y + 1);
    assert!(
        workflow_row.contains("phase 1/2"),
        "dedicated row carries the phase: {workflow_row:?}"
    );
    assert!(
        !session_row.contains("phase 1/2"),
        "session row must not duplicate the dedicated workflow row: {session_row:?}"
    );
}

#[test]
fn live_activity_row_describes_pending_tokens_instead_of_zero() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Drafting response");

    let width = 120;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let rule_top_y = app.regions.expect("layout regions after draw").rule_top.y;
    let row = buffer_row(&terminal, width, rule_top_y);

    assert!(
        row.contains("tokens pending"),
        "zero-token streaming state should read as pending, not broken: {row:?}"
    );
    assert!(
        !row.contains("~0 tokens"),
        "live activity row must not surface a misleading zero-token counter: {row:?}"
    );
}

#[test]
fn live_activity_row_distinguishes_input_from_pending_output() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Drafting response");
    app.update_turn_tokens(1_200, 0);

    let width = 132;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let rule_top_y = app.regions.expect("layout regions after draw").rule_top.y;
    let row = buffer_row(&terminal, width, rule_top_y);

    assert!(
        row.contains("↑ 1.2k input"),
        "input usage should be visible before first output token: {row:?}"
    );
    assert!(
        row.contains("output pending"),
        "zero output should read as a pending stream state, not a dead counter: {row:?}"
    );
    assert!(
        !row.contains("~0 tokens"),
        "live activity row must not show a misleading zero-token counter: {row:?}"
    );
}

#[test]
fn live_activity_row_keeps_fanout_progress_without_workflow_context() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity(
        "Smart: 2/4 complete · 50% · 50% left · 2 pre-analysis agents active (2 running)",
    );

    let width = 120;
    let backend = TestBackend::new(width, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    let rule_top_y = app.regions.expect("layout regions after draw").rule_top.y;
    let row = buffer_row(&terminal, width, rule_top_y);

    assert!(
        row.contains("2/4 complete"),
        "fan-out completion fraction should remain visible without a workflow context badge: {row:?}"
    );
    assert!(
        row.contains("50%"),
        "fan-out percentage should remain visible without a workflow context badge: {row:?}"
    );
    assert!(
        row.contains("50% left"),
        "fan-out remaining percentage should remain visible without a workflow context badge: {row:?}"
    );
}

#[test]
fn reset_session_view_clears_workflow_without_changing_agent_scope() {
    let mut app = test_app();
    app.set_agent_manifest_started_after(123);
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "old-flow".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "old-phase".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 1,
        total_phases: 2,
        next_phase: Some("next".to_string()),
        total_agents: 3,
        progress_percent: 0,
        completed_phases: 0,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 3,
        phases: Vec::new(),
    });
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));
    assert!(app.workflow_viewer_open(), "precondition: modal is open");

    app.reset_session_view();

    assert!(
        app.hud_state.workflow.is_none(),
        "old workflow summary must not survive a visible session reset"
    );
    assert!(
        !app.workflow_viewer_open(),
        "old workflow modal must close on visible session reset"
    );
    assert_eq!(
        app.agent_manifest_started_after(),
        123,
        "plain view reset must not invent a new agent scope"
    );
}

#[test]
fn idle_rule_embers_are_static_across_two_byte_identical_draws() {
    let mut app = test_app_with_theme(Theme::zo());
    app.sidebar.visible = false;
    let width = 80;
    let mut terminal = Terminal::new(TestBackend::new(width, 20)).expect("test terminal");

    app.draw(&mut terminal).expect("first idle draw");
    let regions = app.regions.expect("layout regions after first draw");
    let first = terminal.backend().buffer().clone();
    for offset in 0..3 {
        let cell = &first[(regions.rule_top.x + offset, regions.rule_top.y)];
        assert_eq!(cell.symbol(), crate::tui::glyphs::ANVIL_LINE);
        assert_eq!(cell.fg, app.theme.palette.accent_dim);
    }
    let cold_rule = &first[(regions.rule_top.x + 3, regions.rule_top.y)];
    assert_eq!(cold_rule.symbol(), crate::tui::glyphs::HORIZONTAL_RULE);
    assert_eq!(cold_rule.fg, app.theme.palette.faint);

    app.draw(&mut terminal).expect("second idle draw");
    assert_eq!(
        terminal.backend().buffer(),
        &first,
        "static idle embers must not introduce an animation frame"
    );
}

#[test]
fn no_color_idle_rule_keeps_persistent_heat_off() {
    let mut app = test_app();
    app.sidebar.visible = false;
    let mut terminal = Terminal::new(TestBackend::new(80, 20)).expect("test terminal");

    app.draw(&mut terminal).expect("idle no-color draw");
    let regions = app.regions.expect("layout regions after draw");
    let first = &terminal.backend().buffer()[(regions.rule_top.x, regions.rule_top.y)];
    assert_eq!(first.symbol(), crate::tui::glyphs::HORIZONTAL_RULE_NC);
    assert_eq!(first.fg, ratatui::style::Color::Reset);
}

#[test]
fn effort_badge_renders_on_rule_line_not_transcript_top() {
    let mut app = test_app();
    app.hud_state.effort = Some(Effort::Smart);
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(1),
        text: "hello from transcript".to_string(),
        done: true,
    });

    let width = 80;
    let height = 20;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    // The badge no longer reserves the transcript's top row: the transcript now
    // uses its full height, so its first line sits at row 0.
    let top = buffer_row(&terminal, width, 0);
    assert!(
        !top.contains("smart"),
        "effort badge should no longer float at the transcript top: {top:?}"
    );

    // The badge moved down to the rule line (the input's top edge) near the
    // bottom of the screen.
    let top_half = (0..height / 2)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");
    let bottom_half = (height / 2..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !top_half.contains("smart"),
        "effort badge should not be in the top half:\n{top_half}"
    );
    assert!(
        bottom_half.contains("smart"),
        "active effort badge should render on the rule line above input (bottom half):\n{bottom_half}"
    );
    let badge_row = (height / 2..height)
        .map(|y| buffer_row(&terminal, width, y))
        .find(|row| row.contains("smart"))
        .expect("effort badge row");
    let label_col = badge_row
        .chars()
        .position(|ch| ch == 's')
        .expect("smart label position");
    let badge_cells = badge_row.chars().collect::<Vec<_>>();
    assert_eq!(
        &badge_cells[label_col - 3..label_col],
        &['-', ' ', ' '],
        "idle hairline must stop for one clear cell before the padded badge: {badge_row:?}"
    );
    let full = format!("{top_half}\n{bottom_half}");
    assert!(
        full.contains("hello from transcript"),
        "transcript text should remain visible:\n{full}"
    );
}

#[test]
fn queue_badge_keeps_one_clear_cell_before_right_aligned_badge() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.queue_message("queued work").expect("queue message");

    let width = 80;
    let height = 20;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let badge_row = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .find(|row| row.contains("1 queued"))
        .expect("queue badge row");
    let badge_cells = badge_row.chars().collect::<Vec<_>>();
    let icon_col = badge_cells
        .iter()
        .rposition(|ch| *ch == 'o')
        .expect("queue badge icon position");
    assert_eq!(
        badge_cells[icon_col - 2],
        ' ',
        "queue badge must have one clear cell before its own leading padding: {badge_row:?}"
    );
    assert_eq!(
        badge_cells[icon_col - 1],
        ' ',
        "queue badge keeps its right-aligned leading padding: {badge_row:?}"
    );
}

#[test]
fn fanout_progress_block_keeps_chat_pane_nonblank_during_delegation() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity(
        "Smart: 0/4 complete · 0% · 100% left · 4 pre-analysis agents launching",
    );
    app.upsert_system_block(
        BlockId(42),
        SystemLevel::Info,
        "Smart pre-analysis: running · phase 2/3 · 0% complete · 100% left\n- agents: 0/4 terminal, 4 active (4 running), 0 failed, 0 stopped\n- remaining: waiting for 4 agent results\n- models: gpt-5.5 x4\n- tokens: agent output pending (waiting for usage)".to_string(),
    );

    let width = 120;
    let height = 24;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let dump = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        dump.contains("Smart pre-analysis: running") && dump.contains("0% complete"),
        "central chat pane should show live fan-out progress, not a blank screen:\n{dump}"
    );
    assert!(
        dump.contains("100% left"),
        "central transcript should say how much progress remains:\n{dump}"
    );
    assert!(
        dump.contains("0/4 terminal, 4 active"),
        "agent progress detail should be visible in the transcript:\n{dump}"
    );
    assert!(
        dump.contains("waiting for 4 agent results"),
        "central transcript should say how much work remains:\n{dump}"
    );
    assert!(
        dump.contains("models: gpt-5.5 x4"),
        "central transcript should show the actual bound agent model summary:\n{dump}"
    );
    assert!(
        dump.contains("agent output pending"),
        "central transcript should explain token accounting before usage arrives:\n{dump}"
    );
    assert!(
        dump.contains("Smart"),
        "activity rule should still show the current phase:\n{dump}"
    );
    assert!(
        dump.contains("0/4 complete") && dump.contains("0%") && dump.contains("100% left"),
        "activity rule should show immediate fan-out progress, not just a generic spinner:\n{dump}"
    );
}

#[test]
fn full_frame_keeps_sidebar_chat_and_activity_in_sync_for_agents() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Smart: 2 pre-analysis agents running");
    app.hud_state.ctx_used = 12_400;
    app.hud_state.ctx_limit = 1_000_000;
    app.hud_state.ctx_new_input = 2_000;
    app.hud_state.ctx_cached = 10_400;
    app.hud_state.cost_usd = 0.43;
    app.hud_state.running_agents = 2;
    app.hud_state.agents = vec![
        AgentTaskSummary {
            name: "runtime-streaming".to_string(),
            status: "running".to_string(),
            model: "openai/gpt-5.5-fast".to_string(),
            elapsed_secs: 73,
            token_history: vec![120, 80],
            current_tool: Some("bash".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "cli-ui".to_string(),
            status: "running".to_string(),
            model: "openai/gpt-5.5-fast".to_string(),
            elapsed_secs: 51,
            token_history: vec![90],
            current_tool: Some("read_file".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
    ];
    app.hud_state.workflow = Some(WorkflowSummary {
        name: "ui-polish".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        current_phase: "inspect".to_string(),
        current_phase_status: "running".to_string(),
        current_phase_index: 2,
        total_phases: 3,
        next_phase: Some("verify".to_string()),
        total_agents: 2,
        progress_percent: 33,
        completed_phases: 1,
        completed_agents: 0,
        failed_agents: 0,
        running_agents: 2,
        phases: Vec::new(),
    });
    app.upsert_system_block(
        BlockId(43),
        SystemLevel::Info,
        "Smart pre-analysis: running\n- agents: 2 running, 0 terminal, 0 failed, 0 stopped\n- tokens: ~290 agent output tokens".to_string(),
    );

    let width = 180;
    let height = 32;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let dump = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        dump.contains("Smart pre-analysis: running"),
        "central progress block missing:\n{dump}"
    );
    assert!(
        dump.contains("ctx 12.4k / 1.0M"),
        "sidebar should show non-zero live context, not ctx 0:\n{dump}"
    );
    assert!(
        dump.contains("cost $0.43"),
        "sidebar should show live cost:\n{dump}"
    );
    assert!(
        dump.contains("2 agents"),
        "agent count should match progress/sidebar:\n{dump}"
    );
    assert!(
        dump.contains("runtime-strea") && dump.contains("cli-ui"),
        "expanded agent rows should be visible:\n{dump}"
    );
    assert!(
        dump.contains("workflow") && dump.contains("inspect"),
        "workflow summary should be visible beside agents:\n{dump}"
    );
    assert!(
        dump.contains("Smart"),
        "activity rule should stay visible while transcript carries details:\n{dump}"
    );
}

#[test]
fn live_snapshot_uses_visible_sessions_background_count() {
    let mut app = test_app();
    let registry = runtime::task_registry::TaskRegistry::new_in_memory();
    let task = registry.create_background_process(
        "serve local preview",
        None,
        Some("session-a"),
    );

    app.set_background_process_count(
        registry.live_background_process_count(Some("session-a")),
    );
    app.update_hud_live_snapshot(0, Vec::new(), Vec::new(), None);
    assert_eq!(app.hud_state.background_tasks, 1);

    app.set_background_process_count(
        registry.live_background_process_count(Some("session-b")),
    );
    app.update_hud_live_snapshot(0, Vec::new(), Vec::new(), None);
    assert_eq!(
        app.hud_state.background_tasks, 0,
        "switching the visible session must replace rather than reuse the count"
    );

    registry
        .set_status(
            &task.task_id,
            runtime::task_registry::TaskStatus::Completed,
        )
        .expect("background task should finish");
    app.set_background_process_count(
        registry.live_background_process_count(Some("session-a")),
    );
    assert_eq!(app.hud_state.background_tasks, 0);
}

#[test]
fn scheduled_wake_snapshot_picks_nearest_source_and_clears() {
    let scans = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let poll_scans = std::sync::Arc::clone(&scans);
    let dirty = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let poll_dirty = std::sync::Arc::clone(&dirty);
    let mut app = test_app();
    app.set_scheduled_wakeup_poller(Box::new(move || {
        if !poll_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return None;
        }
        poll_scans.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(Some(ScheduledWakeHud {
            due_at_epoch: 200,
            reason: "file wakeup".to_string(),
            source: WakeSource::Wakeup,
        }))
    }));

    assert!(app.refresh_scheduled_wakeup());
    assert!(!app.refresh_scheduled_wakeup(), "second scan is not due");
    assert_eq!(scans.load(std::sync::atomic::Ordering::Relaxed), 1);

    app.set_scheduled_loop_wake(Some(ScheduledWakeHud {
        due_at_epoch: 150,
        reason: "loop check".to_string(),
        source: WakeSource::Loop,
    }));
    assert_eq!(
        app.hud_state.scheduled_wake.as_ref().map(|wake| wake.source),
        Some(WakeSource::Loop)
    );

    app.set_scheduled_loop_wake(Some(ScheduledWakeHud {
        due_at_epoch: 250,
        reason: "later loop".to_string(),
        source: WakeSource::Loop,
    }));
    assert_eq!(
        app.hud_state.scheduled_wake.as_ref().map(|wake| wake.source),
        Some(WakeSource::Wakeup)
    );

    app.clear_scheduled_file_wake();
    assert_eq!(
        app.hud_state.scheduled_wake.as_ref().map(|wake| wake.source),
        Some(WakeSource::Loop)
    );
    app.set_scheduled_loop_wake(None);
    assert!(app.hud_state.scheduled_wake.is_none());
}

#[test]
fn real_background_bash_watcher_drives_live_snapshot_from_one_to_zero() {
    struct TempDir(std::path::PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let root = std::env::temp_dir().join(format!(
        "zo-tui-background-bash-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("create background bash test dir");
    let _temp = TempDir(root.clone());
    let started = root.join("started");
    let release = root.join("release");
    let command = format!(
        "printf started > '{}'; i=0; while [ ! -f '{}' ] && [ \"$i\" -lt 5 ]; do sleep 1; i=$((i + 1)); done",
        started.display(),
        release.display()
    );
    let registry = runtime::task_registry::TaskRegistry::new_in_memory();
    let mut app = test_app();
    app.set_background_process_count(
        registry.live_background_process_count(Some("visible-session")),
    );

    let output = runtime::execute_bash_with_tasks(
        runtime::BashCommandInput {
            command,
            timeout: None,
            description: None,
            run_in_background: Some(true),
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: None,
            isolate_network: None,
            filesystem_mode: None,
            allowed_mounts: None,
            cwd: Some(root.clone()),
        },
        Some(&registry),
        Some("visible-session"),
    )
    .expect("real background Bash API should spawn");
    let task_id = output.background_task_id.expect("background task id");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !started.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(started.exists(), "background shell should reach its synchronization point");
    app.update_hud_live_snapshot(0, Vec::new(), Vec::new(), None);
    assert_eq!(app.hud_state.background_tasks, 1);

    std::fs::write(&release, b"release").expect("release background shell");
    let finish_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while registry
        .get(&task_id)
        .is_some_and(|task| {
            !matches!(
                task.status,
                runtime::task_registry::TaskStatus::Completed
                    | runtime::task_registry::TaskStatus::Failed
                    | runtime::task_registry::TaskStatus::Stopped
            )
        })
        && std::time::Instant::now() < finish_deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(
        registry.get(&task_id).is_some_and(|task| {
            matches!(
                task.status,
                runtime::task_registry::TaskStatus::Completed
                    | runtime::task_registry::TaskStatus::Failed
                    | runtime::task_registry::TaskStatus::Stopped
            )
        }),
        "actual command watcher should publish a terminal status before timeout"
    );

    app.update_hud_live_snapshot(0, Vec::new(), Vec::new(), None);
    assert_eq!(app.hud_state.background_tasks, 0);
}

#[test]
fn generic_running_task_cannot_drive_background_hud_badge() {
    let registry = runtime::task_registry::TaskRegistry::new_in_memory();
    let task = registry.create("ordinary task", None);
    registry
        .set_status(&task.task_id, runtime::task_registry::TaskStatus::Running)
        .expect("generic task should run");
    let mut app = test_app();
    app.set_background_process_count(
        registry.live_background_process_count(Some("visible-session")),
    );

    app.update_hud_live_snapshot(0, Vec::new(), Vec::new(), None);

    assert_eq!(app.hud_state.background_tasks, 0);
}

#[test]
fn restored_stale_background_record_cannot_drive_background_hud_badge() {
    struct TempDir(std::path::PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let root = std::env::temp_dir().join(format!(
        "zo-tui-stale-background-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("create stale task test dir");
    let _temp = TempDir(root.clone());
    let path = root.join("tasks.json");
    let registry = runtime::task_registry::TaskRegistry::with_persistence_path(Some(path.clone()));
    let task = registry.create_background_process(
        "stale process record",
        None,
        Some("visible-session"),
    );
    registry
        .set_status(&task.task_id, runtime::task_registry::TaskStatus::Running)
        .expect("persist stale running record");
    drop(registry);

    let restored = runtime::task_registry::TaskRegistry::with_persistence_path(Some(path));
    let mut app = test_app();
    app.set_background_process_count(
        restored.live_background_process_count(Some("visible-session")),
    );
    app.update_hud_live_snapshot(0, Vec::new(), Vec::new(), None);

    assert_eq!(
        app.hud_state.background_tasks, 0,
        "a fresh process-owned tracker must not resurrect persisted live state"
    );
}

#[test]
fn mixed_agent_snapshot_counts_live_rows_but_shows_finished_siblings() {
    let mut app = test_app();
    app.update_hud_live_snapshot(
        1,
        Vec::new(),
        vec![
            AgentTaskSummary {
                name: "live-runner".to_string(),
                status: "running".to_string(),
                model: "openai/gpt-5.5-fast".to_string(),
                elapsed_secs: 31,
                token_history: vec![64],
                current_tool: Some("bash".to_string()),
                current_phase: None,
                last_activity_at: None,
                ..Default::default()
            },
            AgentTaskSummary {
                name: "finished-reviewer".to_string(),
                status: "stopped".to_string(),
                model: "openai/gpt-5.5-fast".to_string(),
                elapsed_secs: 29,
                token_history: vec![12],
                current_tool: Some("read_file".to_string()),
                current_phase: None,
                last_activity_at: None,
                ..Default::default()
            },
        ],
        None,
    );

    assert_eq!(app.hud_state.running_agents, 1, "count is live rows only");
    assert_eq!(
        app.hud_state.agents.len(),
        2,
        "a just-finished sibling stays in panel data while others run, so \
         its completed/failed flip is visible live"
    );

    let width = 180;
    let height = 28;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let dump = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        dump.contains("1 agents"),
        "compact HUD/sidebar count should match live rows:\n{dump}"
    );
    assert!(
        dump.contains("live-runner"),
        "running row should remain visible:\n{dump}"
    );
    assert!(
        dump.contains("finished-reviewer"),
        "a finished sibling shows its terminal state (grace window) instead \
         of vanishing one frame after completion:\n{dump}"
    );
}

#[test]
fn workflow_summary_pins_agent_fallback_before_manifest_rows_arrive() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Running Workflow");
    app.update_hud_live_snapshot(
        0,
        Vec::new(),
        Vec::new(),
        Some(WorkflowSummary {
            name: "review-changes".to_string(),
            status: "running".to_string(),
            mode: "phases".to_string(),
            current_phase: "work".to_string(),
            current_phase_status: "running".to_string(),
            current_phase_index: 1,
            total_phases: 2,
            progress_percent: 10,
            completed_phases: 0,
            next_phase: Some("verify".to_string()),
            total_agents: 3,
            completed_agents: 0,
            failed_agents: 0,
            running_agents: 3,
            phases: vec![FleetPhase {
                id: "work".to_string(),
                step_id: None,
                agent_ids: Vec::new(),
                status: "running".to_string(),
                total: 3,
                completed: 0,
                failed: 0,
                running: 3,
            }],
        }),
    );

    let width = 180;
    let height = 30;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let dump = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        dump.contains("Running 3 Workflow agents"),
        "workflow fallback header should confirm sub-agents before manifest rows land:\n{dump}"
    );
    assert!(
        dump.contains("work") && dump.contains("3 running"),
        "workflow fallback should show active phase tally:\n{dump}"
    );
}

#[test]
fn live_snapshot_poll_clears_all_completed_todo_list() {
    // The periodic disk-poll path must honor the same all-completed → clear
    // convention as the push path. Otherwise a settled plan lingered as a
    // fully-checked list and got re-seeded every 500ms, so the panel/sidebar
    // never cleared after the last item finished (the "todo stuck completed"
    // bug). A mixed (not-all-done) list is kept as-is.
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};
    let mut app = test_app();

    // First a live, in-progress plan lands via the poll → tracked.
    app.update_hud_live_snapshot(
        0,
        vec![
            TodoChecklistItem {
                step_id: None,
                content: "Wire the parser".to_string(),
                active_form: "Wiring the parser".to_string(),
                status: TodoChecklistStatus::Completed,
            },
            TodoChecklistItem {
                step_id: None,
                content: "Render the block".to_string(),
                active_form: "Rendering the block".to_string(),
                status: TodoChecklistStatus::InProgress,
            },
        ],
        Vec::new(),
        None,
    );
    assert_eq!(app.hud_state.todo_items.len(), 2, "mixed plan is kept");
    assert_eq!(
        app.hud_state.todo_summary.as_deref(),
        Some("1 todos active")
    );

    // Then every item finishes — the poll must clear the fully checked list so
    // it does not re-seed the panel/sidebar every tick.
    app.update_hud_live_snapshot(
        0,
        vec![
            TodoChecklistItem {
                step_id: None,
                content: "Wire the parser".to_string(),
                active_form: "Wiring the parser".to_string(),
                status: TodoChecklistStatus::Completed,
            },
            TodoChecklistItem {
                step_id: None,
                content: "Render the block".to_string(),
                active_form: "Rendering the block".to_string(),
                status: TodoChecklistStatus::Completed,
            },
        ],
        Vec::new(),
        None,
    );
    assert!(
        app.hud_state.todo_items.is_empty(),
        "an all-completed plan from the poll must clear from the HUD"
    );
    assert_eq!(app.hud_state.todo_summary, None);
}

#[test]
#[allow(clippy::too_many_lines)]
fn zero_running_agent_snapshot_clears_stale_agent_rows_from_frame() {
    let mut app = test_app();
    app.hud_state.running_agents = 2;
    app.hud_state.agents = vec![
        AgentTaskSummary {
            name: "cli-ui".to_string(),
            status: "running".to_string(),
            model: "openai/gpt-5.5-fast".to_string(),
            elapsed_secs: 12,
            token_history: vec![42],
            current_tool: Some("bash".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "runtime".to_string(),
            status: "running".to_string(),
            model: "openai/gpt-5.5-fast".to_string(),
            elapsed_secs: 9,
            token_history: vec![7],
            current_tool: Some("read_file".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
    ];

    app.update_hud_live_snapshot(
        0,
        Vec::new(),
        vec![
            AgentTaskSummary {
                name: "cli-ui".to_string(),
                status: "stopped".to_string(),
                model: "openai/gpt-5.5-fast".to_string(),
                elapsed_secs: 720,
                token_history: vec![42],
                current_tool: Some("bash".to_string()),
                current_phase: None,
                last_activity_at: None,
                ..Default::default()
            },
            AgentTaskSummary {
                name: "runtime".to_string(),
                status: "stopped".to_string(),
                model: "openai/gpt-5.5-fast".to_string(),
                elapsed_secs: 719,
                token_history: vec![7],
                current_tool: Some("read_file".to_string()),
                current_phase: None,
                last_activity_at: None,
                ..Default::default()
            },
        ],
        Some(WorkflowSummary {
            name: "stale-flow".to_string(),
            status: "completed".to_string(),
            mode: "phases".to_string(),
            current_phase: "cleanup".to_string(),
            current_phase_status: "done".to_string(),
            current_phase_index: 2,
            total_phases: 2,
            next_phase: None,
            total_agents: 2,
            progress_percent: 100,
            completed_phases: 2,
            completed_agents: 0,
            failed_agents: 0,
            running_agents: 0,
            phases: Vec::new(),
        }),
    );

    assert_eq!(app.hud_state.running_agents, 0);
    assert!(
        app.hud_state.agents.is_empty(),
        "terminal-only snapshots must not leave stale agent rows in HUD state"
    );
    assert!(
        app.hud_state.workflow.is_none(),
        "terminal workflow summaries must not remain pinned in compact HUD state"
    );

    let width = 180;
    let height = 28;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let dump = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !dump.contains("agents"),
        "stopped agents should disappear from sidebar and bottom HUD on the next frame:\n{dump}"
    );
    assert!(
        !dump.contains("cli-ui") && !dump.contains("runtime"),
        "stale agent rows should not remain visible after all agents stop:\n{dump}"
    );
    assert!(
        !dump.contains("stale-flow") && !dump.contains("cleanup"),
        "terminal workflow summaries should disappear from sidebar and bottom HUD:\n{dump}"
    );
}

#[test]
fn slash_command_while_input_disabled_queues() {
    let mut app = test_app();
    // Simulate a turn in progress: input is disabled. Every composed entry
    // typed mid-turn — slash commands and plain text alike — queues to run as
    // its own step once the turn ends (Claude Code CLI parity).
    app.disable_input();
    assert!(!app.input_enabled());

    for ch in "/help".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(app.input().text(), "/help");

    // Press Enter — should queue the command, not submit.
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.queued_message_count(), 1);
    // Input should be cleared after queuing.
    assert!(app.input().text().is_empty());

    // Type and queue a second command.
    for ch in "/status".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let _ = app.handle_key(press(KeyCode::Enter));
    assert_eq!(app.queued_message_count(), 2);

    // Drain the queue.
    let messages = app.take_queued_messages();
    let texts: Vec<String> = messages.iter().map(|m| m.text.clone()).collect();
    assert_eq!(texts, vec!["/help".to_string(), "/status".to_string()]);
    assert_eq!(app.queued_message_count(), 0);
}

#[test]
fn queue_message_adds_next_turn_prompt() {
    let mut app = test_app();

    app.queue_message("review prompt").unwrap();

    assert_eq!(app.queued_message_count(), 1);
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text).as_deref(),
        Some("review prompt")
    );
    assert_eq!(app.queued_message_count(), 0);
}

/// 큐에 섞여 있는 agent-result 재주입만 순서대로 뽑아내고, 사용자가 타이핑한
/// 메시지는 FIFO 슬롯을 유지한다 — REPL이 완료 배치를 한 턴으로 fold할 때
/// 사용자 입력을 밀어내거나 삼키면 안 된다.
#[test]
fn drain_queued_agent_results_extracts_only_agent_entries_in_order() {
    use runtime::message_stream::AgentResultStatus;

    let mut app = test_app();
    let meta = |status| super::AgentResultMeta {
        label: "background bash".to_string(),
        status,
    };
    app.queue_agent_result_message("[bg a] done", meta(AgentResultStatus::Completed))
        .unwrap();
    app.queue_message("user typed").unwrap();
    app.queue_agent_result_message("[bg b] boom", meta(AgentResultStatus::Failed))
        .unwrap();

    let head = app.pop_next_queued_message().expect("agent head pops first");
    assert!(head.agent_result.is_some());
    assert_eq!(head.text, "[bg a] done");

    let rest = app.drain_queued_agent_results();
    assert_eq!(rest.len(), 1);
    assert_eq!(rest[0].text, "[bg b] boom");

    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text).as_deref(),
        Some("user typed")
    );
    assert_eq!(app.queued_message_count(), 0);
}

#[test]
fn transcript_view_request_roundtrips_and_drains_once() {
    // `/dump` records the request; the host loop drains it exactly once —
    // a second take must be None or the loop would relaunch the viewer on
    // every subsequent slash command.
    let mut app = test_app();
    assert_eq!(app.take_pending_transcript_view(), None);

    app.request_transcript_view(PathBuf::from("/tmp/zo-transcript-x.txt"), true);
    let view = app.take_pending_transcript_view().expect("recorded request");
    assert_eq!(view.path, PathBuf::from("/tmp/zo-transcript-x.txt"));
    assert!(view.edit);
    assert_eq!(app.take_pending_transcript_view(), None);
}

#[test]
fn queued_message_count_is_bounded() {
    let mut app = test_app();

    for i in 0..MAX_QUEUED_MESSAGES {
        app.queue_message(format!("prompt {i}")).unwrap();
    }

    assert_eq!(app.queued_message_count(), MAX_QUEUED_MESSAGES);
    assert_eq!(
        app.queue_message("overflow").unwrap_err(),
        QueueLimitError::QueuedMessagesFull {
            limit: MAX_QUEUED_MESSAGES
        }
    );
    assert_eq!(app.queued_message_count(), MAX_QUEUED_MESSAGES);
}

#[test]
fn pending_clipboard_images_are_bounded() {
    let mut app = test_app();

    for i in 0..MAX_PENDING_IMAGES {
        app.push_clipboard_image("image/png".to_string(), format!("data-{i}"))
            .unwrap();
    }

    assert_eq!(app.pending_images.len(), MAX_PENDING_IMAGES);
    assert_eq!(app.input().image_count(), MAX_PENDING_IMAGES);
    assert_eq!(
        app.push_clipboard_image("image/png".to_string(), "overflow".to_string())
            .unwrap_err(),
        QueueLimitError::PendingImagesFull {
            limit: MAX_PENDING_IMAGES
        }
    );
    assert_eq!(app.pending_images.len(), MAX_PENDING_IMAGES);
    assert_eq!(app.input().image_count(), MAX_PENDING_IMAGES);
}

#[test]
fn queued_images_stage_for_submit_without_reentering_paste_admission() {
    let mut app = test_app();
    let images = vec![ImageAttachment {
        media_type: "image/png".to_string(),
        data: "queued-image".to_string(),
    }];

    app.stage_queued_images_for_submit(images);

    assert_eq!(app.pending_images.len(), 1);
    assert_eq!(app.pending_images[0].data, "queued-image");
    assert_eq!(app.input().image_count(), 1);
}

#[test]
fn full_mid_turn_queue_keeps_draft_and_pending_images() {
    let mut app = test_app();
    for i in 0..MAX_QUEUED_MESSAGES {
        app.queue_message(format!("prompt {i}")).unwrap();
    }
    app.disable_input();
    app.push_clipboard_image("image/png".to_string(), "image".to_string())
        .unwrap();
    for ch in "overflow".chars() {
        app.handle_key(press(KeyCode::Char(ch))).expect("typed");
    }

    let action = app.handle_key(press(KeyCode::Enter)).expect("enter");

    assert_eq!(action, AppAction::None);
    assert_eq!(app.queued_message_count(), MAX_QUEUED_MESSAGES);
    assert_eq!(app.input().text(), "overflow");
    assert_eq!(app.pending_images.len(), 1);
    assert_eq!(app.input().image_count(), 1);
}

#[test]
fn slash_hint_enter_accepts_prompt_command() {
    let mut app = test_app();
    app.enable_input();
    app.set_prompt_commands(vec![commands::PromptCommandDef {
        name: "review-local".to_string(),
        description: Some("Review local changes".to_string()),
        argument_hint: Some("<scope>".to_string()),
        model: None,
        effort: None,
        body: "Review $ARGUMENTS".to_string(),
        allowed_tools: Vec::new(),
        path: PathBuf::from(".zo/commands/review-local.md"),
    }]);
    app.set_input_text("/review-local");

    let prompt_index =
        slash_hint_suggestions(app.input().text().as_str(), &app.prompt_commands, &[], 10)
            .iter()
            .position(|suggestion| suggestion.command == "/review-local")
            .expect("prompt command hint");
    app.hints.slash_cursor = Some(prompt_index);

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "/review-local ");
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn slash_completion_tab_accepts_prompt_command_prefix() {
    let mut app = test_app();
    app.enable_input();
    app.set_prompt_commands(vec![commands::PromptCommandDef {
        name: "review-local".to_string(),
        description: Some("Review local changes".to_string()),
        argument_hint: Some("<scope>".to_string()),
        model: None,
        effort: None,
        body: "Review $ARGUMENTS".to_string(),
        allowed_tools: Vec::new(),
        path: PathBuf::from(".zo/commands/review-local.md"),
    }]);
    app.set_input_text("/review-l");

    let action = app.handle_key(press(KeyCode::Tab)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "/review-local ");
    assert!(
        !app.slash_hint_active(),
        "Tab completion should hide the slash hint for the accepted text"
    );
}

#[test]
fn slash_hint_esc_hides_until_input_changes() {
    let mut app = test_app();
    app.enable_input();
    app.set_input_text("/help");
    assert!(app.slash_hint_active(), "slash hint starts visible");

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "/help");
    assert!(
        !app.slash_hint_active(),
        "Esc should actually close the slash hint"
    );

    let _ = app.handle_key(press(KeyCode::Char('x')));
    assert!(
        app.slash_hint_active(),
        "editing the slash buffer re-enables hinting"
    );
}

#[test]
fn mention_hint_esc_hides_until_input_changes() {
    let mut app = test_app();
    app.enable_input();
    app.workspace_files = vec![
        "src/convert.rs".to_string(),
        "src/config.rs".to_string(),
        "README.md".to_string(),
    ];
    app.set_input_text("see @con");
    assert!(app.mention_hint_active(), "mention hint starts visible");

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "see @con");
    assert!(
        !app.mention_hint_active(),
        "Esc should actually close the mention hint"
    );

    let _ = app.handle_key(press(KeyCode::Char('v')));
    assert!(
        app.mention_hint_active(),
        "editing the mention buffer re-enables hinting"
    );
}

#[test]
fn up_arrow_scrolls_transcript_when_history_is_empty() {
    // With no prompt history to recall, Up still falls through to transcript
    // scrolling (the history nav handler returns None on an empty history).
    let mut app = test_app();
    app.enable_input();
    app.set_history(isolated_history());
    for i in 0..40 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("line {i}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets scroll state");
    let before_up = app.transcript.scroll();
    assert!(before_up > 0, "fixture should overflow the transcript");

    let action = app.handle_key(press(KeyCode::Up)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "");
    assert_eq!(app.history_cursor, None);
    assert!(app.transcript.scroll() < before_up);
}

#[test]
fn slow_up_presses_still_browse_history() {
    // Human-speed Up presses keep recalling prompt history normally.
    let mut app = test_app();
    app.enable_input();
    app.set_history(isolated_history());
    app.append_history("older prompt");
    app.append_history("newer prompt");

    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "newer prompt");
    std::thread::sleep(std::time::Duration::from_millis(20));
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "older prompt");
    assert_eq!(app.history_cursor, Some(1));
}

#[test]
fn fast_up_presses_advance_history_not_toggle() {
    // Rapid keyboard repeats are still ordinary history navigation.
    let mut app = test_app();
    app.enable_input();
    app.set_history(isolated_history());
    app.append_history("older prompt");
    app.append_history("newer prompt");

    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "newer prompt");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(
        app.input().text(),
        "older prompt",
        "the second Up must keep browsing, not toggle back to the blank draft"
    );
    assert_eq!(app.history_cursor, Some(1));
}

#[test]
fn up_down_browse_prompt_history_claude_code_parity() {
    // Claude Code CLI parity: with the cursor on the first line, Up recalls the
    // previous (older) prompt and keeps walking back; Down walks toward the more
    // recent ones and finally restores the in-progress draft.
    let mut app = test_app();
    app.enable_input();
    app.set_history(isolated_history());
    app.append_history("first prompt");
    app.append_history("second prompt");
    app.append_history("third prompt");

    // Type a draft that must be stashed when history browsing starts.
    for ch in "draft".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }

    // Up → most recent ("third"), then older, then oldest.
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "third prompt");
    assert_eq!(app.history_cursor, Some(0));

    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "second prompt");

    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "first prompt");
    assert_eq!(app.history_cursor, Some(2));

    // At the oldest entry, a further Up no longer changes the input (falls
    // through to transcript scrolling) and keeps the oldest prompt visible.
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "first prompt");
    assert_eq!(app.history_cursor, Some(2));

    // Down walks back toward recent prompts.
    let _ = app.handle_key(press(KeyCode::Down)).unwrap();
    assert_eq!(app.input().text(), "second prompt");
    let _ = app.handle_key(press(KeyCode::Down)).unwrap();
    assert_eq!(app.input().text(), "third prompt");

    // One more Down past the newest entry restores the stashed draft and exits
    // history browsing.
    let _ = app.handle_key(press(KeyCode::Down)).unwrap();
    assert_eq!(app.input().text(), "draft");
    assert_eq!(app.history_cursor, None);
}

#[test]
fn arrows_move_between_lines_inside_multiline_draft() {
    // In a multi-line draft the arrows move the cursor between lines; history
    // is recalled only at the edges (first line Up / last line Down).
    let mut app = test_app();
    app.enable_input();
    app.set_history(isolated_history());
    app.append_history("old prompt");

    // Compose "a\nb\nc" via Shift+Enter newlines, leaving the cursor on the
    // last line.
    let shift_enter = KeyEvent {
        code: KeyCode::Enter,
        modifiers: KeyModifiers::SHIFT,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    };
    let _ = app.handle_key(press(KeyCode::Char('a')));
    let _ = app.handle_key(shift_enter);
    let _ = app.handle_key(press(KeyCode::Char('b')));
    let _ = app.handle_key(shift_enter);
    let _ = app.handle_key(press(KeyCode::Char('c')));
    assert_eq!(app.input().cursor().0, 2, "cursor starts on the last line");

    // Up from the last line moves to the middle line — NOT history.
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().cursor().0, 1);
    assert_eq!(app.input().text(), "a\nb\nc", "draft is untouched");
    assert_eq!(app.history_cursor, None);

    // Up again reaches the first line — still navigating the draft, not history.
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().cursor().0, 0);
    assert_eq!(app.input().text(), "a\nb\nc");
    assert_eq!(app.history_cursor, None);

    // Now on the first line, one more Up recalls prompt history.
    let _ = app.handle_key(press(KeyCode::Up)).unwrap();
    assert_eq!(app.input().text(), "old prompt");
    assert_eq!(app.history_cursor, Some(0));
}

#[test]
fn arrow_keys_move_slash_hint_selection_without_scrolling_transcript() {
    let mut app = test_app();
    app.enable_input();
    app.set_input_text("/");
    app.transcript.scroll_to_top();
    let before_scroll = app.transcript.scroll();
    assert!(
        app.slash_hint_suggestion_count() > 1,
        "fixture should expose multiple slash hints"
    );

    let action = app.handle_key(press(KeyCode::Down)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "/");
    assert_eq!(app.hints.slash_cursor, Some(1));
    assert_eq!(app.transcript.scroll(), before_scroll);

    let action = app.handle_key(press(KeyCode::Up)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "/");
    assert_eq!(app.hints.slash_cursor, Some(0));
    assert_eq!(app.transcript.scroll(), before_scroll);
}

#[test]
fn plain_text_mid_turn_queues_and_steers() {
    // Claude Code CLI parity ("type to steer"): plain text typed mid-turn
    // shows as queued *and* rides the steering channel so the live turn folds
    // it at its next boundary. If the turn never folds it, the queued entry
    // still auto-submits as its own turn afterwards.
    let (mut app, mut cmd_rx) = test_app_with_cmd();
    app.disable_input();

    for ch in "use X instead".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.queued_message_count(), 1);
    assert!(app.input().text().is_empty(), "input cleared after queuing");
    match cmd_rx.try_recv() {
        Ok(AgentCommand::Steer(text)) => assert_eq!(text, "use X instead"),
        other => panic!("plain text must also steer the live turn: {other:?}"),
    }
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text),
        Some("use X instead".to_string())
    );
}

/// FIFO parity with Claude Code: a plain-text message typed mid-turn must NOT
/// steer (cut into the live turn) when an earlier entry is already queued —
/// otherwise it would reach the model before that earlier entry, inverting the
/// order the user typed. It only steers when it is at the front of the queue.
#[test]
fn plain_text_does_not_steer_ahead_of_an_earlier_queued_entry() {
    let (mut app, mut cmd_rx) = test_app_with_cmd();
    app.disable_input();

    // First entry: a slash command, which never steers — it waits its turn.
    for ch in "/compact".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let _ = app.handle_key(press(KeyCode::Enter));
    assert!(cmd_rx.try_recv().is_err(), "slash command must not steer");

    // Second entry: plain text. It is now BEHIND /compact, so it must queue
    // without steering — cutting in would deliver it before /compact.
    for ch in "then refactor".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let _ = app.handle_key(press(KeyCode::Enter));
    assert!(
        cmd_rx.try_recv().is_err(),
        "plain text behind an earlier queued entry must wait, not steer ahead"
    );

    // Both drain in the order typed.
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text),
        Some("/compact".to_string())
    );
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text),
        Some("then refactor".to_string())
    );

    // And a lone plain-text entry (empty queue) still steers — the kept reflex.
    for ch in "go now".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let _ = app.handle_key(press(KeyCode::Enter));
    match cmd_rx.try_recv() {
        Ok(AgentCommand::Steer(text)) => assert_eq!(text, "go now"),
        other => panic!("a front-of-queue plain message must still steer: {other:?}"),
    }
}

#[test]
fn workflow_modal_mid_turn_queues_and_steers_plain_text() {
    let (mut app, mut cmd_rx) = test_app_with_cmd();
    app.disable_input();
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));

    for ch in "check agents".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();

    assert_eq!(action, AppAction::None);
    assert_eq!(app.input().text(), "");
    match cmd_rx.try_recv() {
        Ok(AgentCommand::Steer(text)) => assert_eq!(text, "check agents"),
        other => panic!("workflow-modal text must also steer: {other:?}"),
    }
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text),
        Some("check agents".to_string())
    );
}

#[test]
fn workflow_modal_post_turn_submits_and_closes_viewer() {
    // Regression: after a Workflow turn ends, the viewer stays open
    // (`AppMode::ModalWorkflow`) and input is re-enabled. A plain Enter must
    // submit the composed message — previously it was swallowed because no
    // handler claimed it (`handle_input_key` only ran in `Normal`), so the
    // composer could neither submit nor clear. Submitting must also close the
    // read-only viewer so the new turn is not hidden behind its `Clear` overlay.
    let (mut app, _cmd_rx) = test_app_with_cmd();
    app.enable_input();
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));
    assert!(app.workflow_viewer_open(), "precondition: viewer open");

    for ch in "next steps".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(
        app.input().text(),
        "next steps",
        "chars must compose into the composer while the viewer is open"
    );

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();

    assert_eq!(action, AppAction::Submit("next steps".to_string()));
    assert_eq!(app.input().text(), "", "composer must clear on submit");
    assert!(
        !app.workflow_viewer_open(),
        "submitting a fresh message must close the read-only viewer"
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn mouse_wheel_over_input_scrolls_transcript_without_moving_composer() {
    let mut app = test_app();
    app.enable_input();
    app.set_history(isolated_history());
    app.append_history("previous prompt");
    app.set_input_text("draft stays put");
    for i in 0..40 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("line {i}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let input = app.regions.expect("layout regions after draw").input;
    let draft = app.input().text();
    let before = app.transcript.scroll();
    assert!(before > 0, "fixture should overflow the transcript");
    // Wheel-up while the pointer rests over the composer must scroll the
    // transcript up (the regression: the input box used to swallow the event).
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: input.x.saturating_add(1),
        row: input.y,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse handled");

    let after_up = app.transcript.scroll();
    assert!(
        after_up < before,
        "wheel-up over the input must scroll the transcript up ({after_up} < {before})"
    );
    assert!(
        !app.transcript_view.follow_output,
        "scrolling up over the input must drop auto-follow"
    );
    assert_eq!(
        app.input().text(),
        draft,
        "wheel-up must not recall prompt history into the composer"
    );
    app.draw(&mut terminal).expect("draw after wheel-up");
    assert_eq!(
        app.regions.expect("layout regions after wheel-up").input,
        input,
        "wheel-up must not move or resize the composer"
    );

    // And wheel-down over the input scrolls back toward the tail.
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: input.x.saturating_add(1),
        row: input.y,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse handled");

    assert!(
        app.transcript.scroll() > after_up,
        "wheel-down over the input must scroll the transcript back down"
    );
    assert_eq!(
        app.input().text(),
        draft,
        "wheel-down must not recall prompt history into the composer"
    );
    app.draw(&mut terminal).expect("draw after wheel-down");
    assert_eq!(
        app.regions.expect("layout regions after wheel-down").input,
        input,
        "wheel-down must not move or resize the composer"
    );
}

#[test]
fn mouse_wheel_over_slash_hint_moves_hint_not_transcript() {
    let mut app = test_app();
    app.enable_input();
    app.set_input_text("/");
    for i in 0..40 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("line {i}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let popup = app
        .slash_hint_popup_rect()
        .expect("slash popup rect after draw");
    assert!(
        app.slash_hint_suggestion_count() > 1,
        "fixture should expose multiple slash hints"
    );
    let before = app.transcript.scroll();
    let follow_before = app.transcript_view.follow_output;
    assert!(before > 0, "fixture should overflow the transcript");

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: popup.x.saturating_add(1),
        row: popup.y.saturating_add(1),
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse handled");

    assert_eq!(
        app.transcript.scroll(),
        before,
        "wheel over slash hint must not scroll transcript"
    );
    assert_eq!(
        app.transcript_view.follow_output, follow_before,
        "slash hint wheel must not change auto-follow"
    );
    assert_eq!(app.hints.slash_cursor, Some(1));

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: popup.x.saturating_add(1),
        row: popup.y.saturating_add(1),
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse handled");

    assert_eq!(app.transcript.scroll(), before);
    assert_eq!(app.hints.slash_cursor, Some(0));
}

#[test]
fn mouse_wheel_over_sidebar_scrolls_transcript_not_sidebar() {
    let mut app = test_app();
    app.sidebar.visible = true;

    app.sidebar.set_changed_files(
        vec![crate::tui::sidebar::ChangedFile {
            path: "src/main.rs".to_string(),
            status: crate::tui::sidebar::FileStatus::Modified,
            adds: 0,
            rems: 0,
        }],
        1,
    );
    for i in 0..40 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("line {i}"),
            done: true,
        });
    }

    let backend = TestBackend::new(160, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let sidebar = app.regions.expect("layout regions after draw").sidebar;
    assert!(sidebar.width > 0, "wide layout should show a sidebar");

    app.transcript.scroll_to_top();
    let before_transcript = app.transcript.scroll();
    let before_sidebar = app.sidebar.scroll;

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: sidebar.x,
        row: sidebar.y,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse handled");

    assert_eq!(
        app.sidebar.scroll, before_sidebar,
        "normal-mode wheel events should not scroll the sidebar"
    );
    assert!(
        app.transcript.scroll() > before_transcript,
        "wheel-down over the sidebar should move the transcript"
    );
}

#[test]
fn enter_on_empty_input_does_not_queue() {
    let mut app = test_app();
    app.disable_input();

    // Press Enter with empty input — nothing should be queued.
    let _ = app.handle_key(press(KeyCode::Enter));
    assert_eq!(app.queued_message_count(), 0);

    // Whitespace-only input should also not queue.
    let _ = app.handle_key(press(KeyCode::Char(' ')));
    let _ = app.handle_key(press(KeyCode::Enter));
    assert_eq!(app.queued_message_count(), 0);
}

#[test]
fn shift_enter_mid_turn_inserts_newline_not_a_second_queue_entry() {
    // Regression: while a turn is in flight the composer is in queued-message
    // mode. Shift+Enter must insert a newline so a multi-line message stays a
    // single queued entry — previously the queue path treated *any* Enter
    // (modifiers ignored) as a commit, so each Shift+Enter split one message
    // into several separate queued turns.
    let mut app = test_app();
    app.disable_input();

    for ch in "line1".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let shift_enter = KeyEvent {
        code: KeyCode::Enter,
        modifiers: KeyModifiers::SHIFT,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    };
    let _ = app.handle_key(shift_enter);
    for ch in "line2".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    // Mid-composition nothing is queued yet — the newline stayed in the buffer.
    assert_eq!(app.queued_message_count(), 0);

    // A plain Enter commits the whole multi-line message as one queued entry.
    let _ = app.handle_key(press(KeyCode::Enter));
    assert_eq!(app.queued_message_count(), 1);
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text),
        Some("line1\nline2".to_string())
    );
}

#[test]
fn question_mark_mid_turn_types_into_queue_not_help_overlay() {
    // Regression: `?` on an empty prompt opens the keybinding help pager, but
    // only between turns. Mid-turn (input disabled) the composer is queuing,
    // so `?` must land in the input buffer instead of stealing focus into the
    // pager and dropping whatever the user was queuing.
    let mut app = test_app();
    app.disable_input();

    let _ = app.handle_key(press(KeyCode::Char('?')));
    for ch in "fix it".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(app.input().text(), "?fix it");

    let _ = app.handle_key(press(KeyCode::Enter));
    assert_eq!(app.queued_message_count(), 1);
    assert_eq!(
        app.pop_next_queued_message().map(|m| m.text),
        Some("?fix it".to_string())
    );
}

#[test]
fn queue_not_drained_by_enable_input() {
    let mut app = test_app();
    app.disable_input();

    // A slash command queues (plain text now also queues).
    for ch in "/compact".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let _ = app.handle_key(press(KeyCode::Enter));
    assert_eq!(app.queued_message_count(), 1);

    // Re-enabling input should NOT clear queued messages.
    app.enable_input();
    assert_eq!(app.queued_message_count(), 1);

    let messages = app.take_queued_messages();
    let texts: Vec<String> = messages.iter().map(|m| m.text.clone()).collect();
    assert_eq!(texts, vec!["/compact".to_string()]);
}

#[test]
fn input_enabled_enter_submits_not_queues() {
    let mut app = test_app();
    app.enable_input();

    for ch in "submit me".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    // When input is enabled, Enter should produce a Submit action.
    assert!(matches!(action, AppAction::Submit(ref t) if t == "submit me"));
    // Nothing should be queued.
    assert_eq!(app.queued_message_count(), 0);
}

#[test]
fn image_only_paste_mid_turn_queues_with_image() {
    // Pasting an image while a turn is in flight, then pressing Enter, must
    // queue the image (text empty) rather than silently dropping it.
    let mut app = test_app();
    app.disable_input();

    app.handle_paste("data:image/png;base64,aGVsbG8=");
    assert_eq!(app.input().image_count(), 1);

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.queued_message_count(), 1);
    // The pending image moved into the queued entry, not the live composer.
    assert!(!app.has_pending_images());

    let queued = app.pop_next_queued_message().expect("queued entry");
    assert!(queued.text.trim().is_empty());
    assert_eq!(queued.images.len(), 1);
    assert_eq!(queued.images[0].media_type, "image/png");
    assert_eq!(queued.images[0].data, "aGVsbG8=");
}

#[test]
fn text_plus_image_mid_turn_queues_both() {
    // Text plus a pasted image, queued together as one entry mid-turn.
    let mut app = test_app();
    app.disable_input();

    app.handle_paste("data:image/png;base64,aGVsbG8=");
    for ch in "look at this".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    let _ = app.handle_key(press(KeyCode::Enter));

    let queued = app.pop_next_queued_message().expect("queued entry");
    assert_eq!(queued.text, "look at this");
    assert_eq!(queued.images.len(), 1);
}

#[test]
fn paste_image_data_url_stages_image_attachment() {
    let mut app = test_app();
    app.enable_input();

    app.handle_paste("data:image/png;base64,aGVsbG8=");

    assert!(
        app.input().text().is_empty(),
        "data URL must not enter text"
    );
    assert_eq!(app.input().image_count(), 1);
    assert!(app.has_pending_images());

    let images = app.take_pending_images();
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].media_type, "image/png");
    assert_eq!(images[0].data, "aGVsbG8=");
}

#[test]
fn paste_regular_text_still_enters_input() {
    let mut app = test_app();
    app.enable_input();

    app.handle_paste("data:application/json;base64,e30=");

    assert_eq!(app.input().text(), "data:application/json;base64,e30=");
    assert!(!app.has_pending_images());
}

#[test]
fn paste_large_text_preserves_existing_input_and_stays_collapsed() {
    let mut app = test_app();
    app.enable_input();
    app.input_mut().insert_text("before ");
    let pasted = (1..=12)
        .map(|n| format!("line {n}"))
        .collect::<Vec<_>>()
        .join("\n");

    let expected = format!("before {pasted} after");
    app.handle_paste_owned(pasted);
    for ch in " after".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }

    assert_eq!(app.input().text(), expected);
    assert_eq!(app.input().lines().len(), 1);
    assert!(app.input().lines()[0].starts_with("before ("));
    assert!(app.input().lines()[0].contains("pasted"));
    assert!(app.input().lines()[0].ends_with(" after"));
}

#[test]
fn paste_reaches_composer_while_workflow_viewer_open_mid_turn() {
    // Live bug (push session): with the workflow live monitor up, printable
    // keys fall through to the composer so the user can steer while watching —
    // but IME-committed Hangul arrives as a *paste*, which `handle_paste`
    // dropped for `ModalWorkflow`. Korean steering was silently dead exactly
    // while a workflow ran; the composer looked broken ("input창에만 안 써짐").
    let mut app = test_app();
    // Mid-turn: input stays disabled (queued-message mode).
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));
    assert!(app.workflow_viewer_open(), "precondition: viewer open");

    app.handle_paste("한글 조향");

    assert_eq!(
        app.input().text(),
        "한글 조향",
        "IME paste must reach the composer while the workflow viewer is open"
    );
}

#[test]
fn paste_extends_search_query_and_strips_control_chars() {
    // Search built its query only from per-character key events, so a Hangul
    // query (IME → paste) was impossible. Pastes now extend the query with
    // control characters stripped — a query is a single line by construction.
    let mut app = test_app();
    app.enter_search();

    app.handle_paste("한글");
    assert_eq!(app.search_query(), "한글");

    app.handle_paste("검\n색");
    assert_eq!(
        app.search_query(),
        "한글검색",
        "multi-line paste must fold into a single-line query"
    );
}

#[test]
fn at_opens_file_picker_modal() {
    let mut app = test_app();
    app.enable_input();

    // `@` in Normal/input-enabled mode should open the file picker and
    // NOT land a literal `@` in the buffer.
    let action = app.handle_key(press(KeyCode::Char('@'))).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::ModalFile);
    assert!(app.input().text().is_empty(), "@ must not enter the buffer");
}

#[test]
fn workspace_file_scan_returns_empty_when_cancelled_before_start() {
    let cancel = super::new_scan_cancel_token();
    cancel.store(true, std::sync::atomic::Ordering::Relaxed);

    assert!(super::collect_workspace_files(&cancel).is_empty());
}

#[test]
fn at_completion_omits_gitignored_paths() {
    // The `@`-picker scan must honour `.gitignore` so build/coverage output
    // (here `target/`) never pollutes file completion — only tracked sources
    // appear. Regression: the old hand-rolled walk ignored `.gitignore`.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let root = std::env::temp_dir().join(format!(
        "zo-at-ignore-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(root.join("target")).expect("target dir");
    // A `.git` dir makes this a repo so `ignore` honours `.gitignore`
    // (its default `require_git` semantics), mirroring a real checkout.
    std::fs::create_dir_all(root.join(".git")).expect("git dir");
    std::fs::write(root.join(".gitignore"), "target/\n**/.zo/cache/\n").expect("gitignore");
    std::fs::write(root.join("src.rs"), "fn main() {}\n").expect("tracked file");
    std::fs::write(root.join("target").join("build.o"), "junk\n").expect("ignored file");
    let nested_cache = root.join("nested").join(".zo").join("cache").join("prompt-cache");
    std::fs::create_dir_all(&nested_cache).expect("nested prompt cache dir");
    std::fs::write(nested_cache.join("entry.json"), "cached prompt\n").expect("cache entry");
    // A cache file directly under `.zo/cache/` (not the prompt-cache subdir):
    // the recursive `**/.zo/cache/` rule must ignore the whole cache tree, not
    // just the prompt-cache leaf.
    let other_cache = root.join("nested").join(".zo").join("cache");
    std::fs::write(other_cache.join("other.bin"), "junk\n").expect("other cache entry");

    let cancel = super::new_scan_cancel_token();
    let files = super::collect_workspace_files_in(&root, &cancel);

    assert!(
        files.iter().any(|f| f == "src.rs"),
        "tracked source must appear in @ completion, got {files:?}"
    );
    assert!(
        !files.iter().any(|f| f.starts_with("target/")),
        "gitignored target/ output must be absent from @ completion, got {files:?}"
    );
    assert!(
        !files
            .iter()
            .any(|f| f == "nested/.zo/cache/prompt-cache/entry.json"),
        "prompt-cache entries must be ignored at any depth, got {files:?}"
    );
    assert!(
        !files.iter().any(|f| f == "nested/.zo/cache/other.bin"),
        "the whole .zo/cache/ tree must be ignored, not just prompt-cache/, got {files:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn wheel_scrolls_active_picker() {
    // A mouse wheel notch over an open selection-list picker must move its
    // highlight, matching the arrow keys. Regression: modal modes hit the
    // `_ => {}` catch-all in `handle_mouse`, so the wheel was dropped. The arg
    // picker now lives on the unified modal slot, so the highlight is observed
    // through the public submit path (Enter re-submits the highlighted label)
    // rather than a concrete field.
    let rows = App::MOUSE_SCROLL_ROWS as usize;
    // Enough rows that a wheel notch lands mid-list (not clamped to the ends),
    // so the scroll distance — not just "moved at all" — is observed.
    let options = || (0..(rows + 3)).map(|n| format!("row{n}")).collect::<Vec<_>>();

    // Wheel-down advances the highlight by exactly one notch (`rows`).
    let mut app = test_app();
    app.open_arg_picker("theme", "/theme", options());
    assert_eq!(app.mode(), AppMode::ModalArgPick);
    let action = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 10,
            row: 10,
            modifiers: KeyModifiers::NONE,
        })
        .expect("wheel handled");
    // The wheel only scrolls — it never confirms a choice, so the modal stays open.
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::ModalArgPick);
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    let expected = format!("/theme row{rows}");
    assert!(
        matches!(action, AppAction::Submit(ref t) if *t == expected),
        "wheel-down must advance the highlight by one notch, got {action:?}"
    );

    // Wheel-up cancels the wheel-down, returning the highlight to the top.
    let mut app = test_app();
    app.open_arg_picker("theme", "/theme", options());
    for kind in [MouseEventKind::ScrollDown, MouseEventKind::ScrollUp] {
        let action = app
            .handle_mouse(MouseEvent {
                kind,
                column: 10,
                row: 10,
                modifiers: KeyModifiers::NONE,
            })
            .expect("wheel handled");
        assert_eq!(action, AppAction::None);
    }
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::Submit(ref t) if t == "/theme row0"),
        "wheel-up must move the highlight back to the first row, got {action:?}"
    );
}

#[test]
fn exit_modal_marks_pending_file_scan_cancelled() {
    let mut app = test_app();
    let cancel = super::new_scan_cancel_token();
    app.scans.file_cancel = Some(std::sync::Arc::clone(&cancel));

    app.exit_modal();

    assert!(cancel.load(std::sync::atomic::Ordering::Relaxed));
    assert!(app.scans.file_cancel.is_none());
}

#[test]
fn file_picker_selection_inserts_reference_token() {
    let mut app = test_app();
    app.enable_input();

    // Seed a deterministic list so the test does not depend on the cwd.
    app.open_file_picker_for_test(vec!["src/main.rs".to_string(), "Cargo.toml".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalFile);

    // Move to the second entry and select it.
    let _ = app.handle_key(press(KeyCode::Down));
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);

    // Modal closes and the chosen path is spliced in as an `@path ` token.
    assert_eq!(app.mode(), AppMode::Normal);
    assert_eq!(app.input().text(), "@Cargo.toml ");
}

#[test]
fn arg_picker_resubmits_selected_choice_as_slash_command() {
    let mut app = test_app();
    // `/theme` with no argument opens the generic fixed-choice picker.
    app.open_arg_picker(
        "theme",
        "/theme",
        vec!["zo".to_string(), "dark".to_string(), "light".to_string()],
    );
    assert_eq!(app.mode(), AppMode::ModalArgPick);

    // Move to the second entry ("dark") and confirm.
    let _ = app.handle_key(press(KeyCode::Down));
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();

    // The choice re-enters the text path as `/theme dark`, so the command's
    // existing handler applies it — no duplicated apply logic in the modal
    // (mirrors the `/login` and `/effort` pickers).
    assert!(
        matches!(action, AppAction::Submit(ref t) if t == "/theme dark"),
        "expected `/theme dark` submit, got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn arg_picker_esc_cancels_without_submitting() {
    let mut app = test_app();
    app.open_arg_picker("plan", "/plan", vec!["on".to_string(), "off".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalArgPick);

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn arg_picker_occupies_unified_slot() {
    // The migrated arg picker lives on the single `active_modal` slot (not a
    // per-modal field); Esc clears the slot back to Normal in one place.
    let mut app = test_app();
    app.open_arg_picker("theme", "/theme", vec!["zo".to_string(), "dark".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalArgPick);
    assert!(app.active_modal.is_some(), "arg picker must occupy the unified slot");

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none(), "Esc must clear the slot");
}

#[test]
fn async_question_supersedes_open_slot_modal() {
    // A question is ingested from the async agent loop regardless of mode, so it
    // can arrive while another slot modal (the arg picker) is open. It takes over
    // the single slot rather than being shadowed by the stale modal.
    let mut app = test_app();
    app.open_arg_picker("theme", "/theme", vec!["zo".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalArgPick);

    let (responder, _rx) = tokio::sync::oneshot::channel();
    app.open_user_question_modal(UserQuestionPrompt {
        id: BlockId(1),
        question: "continue?".to_string(),
        header: None,
        options: vec![QuestionOption::plain("yes")],
        multi_select: false,
        responder,
    });

    assert_eq!(app.mode(), AppMode::ModalQuestion, "the question takes the slot");
    assert!(app.active_modal.is_some(), "the question modal occupies the slot");
    assert!(
        app.active_user_question.is_some(),
        "the question's responder is held until it resolves"
    );
}

#[test]
fn user_question_answer_routes_through_responder() {
    // Confirming the migrated question modal sends the answer on its oneshot and
    // releases the held prompt, then closes the slot.
    let (responder, mut rx) = tokio::sync::oneshot::channel();
    let mut app = test_app();
    app.open_user_question_modal(UserQuestionPrompt {
        id: BlockId(2),
        question: "pick".to_string(),
        header: None,
        options: vec![QuestionOption::plain("alpha"), QuestionOption::plain("beta")],
        multi_select: false,
        responder,
    });
    assert_eq!(app.mode(), AppMode::ModalQuestion);

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none(), "slot clears after answering");
    assert!(
        app.active_user_question.is_none(),
        "the held prompt is released after answering"
    );
    let answer = rx.try_recv().expect("an answer was sent on the responder");
    // Single-select resolves to a one-element list — the cursor starts on the
    // first option, so Enter picks "alpha".
    assert_eq!(answer, vec!["alpha".to_string()]);
}

#[test]
fn session_picker_selects_session_by_index() {
    // The migrated session picker lives on the unified slot; selecting a row
    // must resolve to the parallel `session_ids[index]` via SelectSession.
    let mut app = test_app();
    app.open_session_modal(
        vec!["most recent".to_string(), "yesterday".to_string()],
        vec!["id-recent".to_string(), "id-yesterday".to_string()],
    );
    assert_eq!(app.mode(), AppMode::ModalSession);
    assert!(app.active_modal.is_some(), "session picker occupies the slot");

    // Move to the second row and confirm.
    let _ = app.handle_key(press(KeyCode::Down));
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::SelectSession("id-yesterday".to_string()));
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none(), "slot clears after selection");
}

#[test]
fn session_picker_esc_cancels() {
    let mut app = test_app();
    app.open_session_modal(vec!["only".to_string()], vec!["id-only".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalSession);

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none());
}

#[test]
fn login_picker_resubmits_provider_command() {
    // A non-`connect-key` token re-enters the text path as `/<command> <provider>`
    // so the provider-specific slash handler runs unchanged.
    let mut app = test_app();
    app.open_login_modal(
        "/login",
        vec!["Claude".to_string()],
        vec!["login:claude".to_string()],
    );
    assert_eq!(app.mode(), AppMode::ModalLogin);

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::Submit(ref t) if t == "/login claude"),
        "expected `/login claude` submit, got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn login_picker_connect_key_opens_api_key_modal() {
    // A `connect-key:<provider>` token with a known preset opens the API-key
    // setup modal instead of submitting, and yields no immediate action.
    let mut app = test_app();
    app.open_login_modal(
        "/connect",
        vec!["DeepSeek".to_string()],
        vec!["connect-key:deepseek".to_string()],
    );
    assert_eq!(app.mode(), AppMode::ModalLogin);

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(
        app.mode(),
        AppMode::ModalApiKey,
        "connect-key must chain into the API-key modal"
    );
    assert!(
        app.active_modal.is_some(),
        "the api-key modal is installed on the unified slot"
    );
}

#[test]
fn login_picker_connect_custom_opens_custom_provider_modal() {
    // A `connect-custom:*` token opens the guided custom provider wizard.
    let mut app = test_app();
    app.open_login_modal(
        "/connect",
        vec!["Custom".to_string()],
        vec!["connect-custom:openai-compatible".to_string()],
    );
    assert_eq!(app.mode(), AppMode::ModalLogin);

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(
        app.mode(),
        AppMode::ModalCustomProvider,
        "connect-custom must chain into the custom provider wizard"
    );
    assert!(
        app.active_modal.is_some(),
        "the custom provider modal is installed on the unified slot"
    );
}

fn sample_model_entry(alias: &str) -> ModelPickerEntry {
    ModelPickerEntry {
        provider: "anthropic".to_string(),
        model: ActiveModel {
            provider: "anthropic",
            alias: alias.to_string(),
            display_name: format!("Claude {alias}"),
            context_limit: 200_000,
        },
    }
}

#[test]
fn model_picker_selects_model_on_the_slot() {
    // The migrated model picker lives on the unified slot; Enter resolves to
    // SelectModel with the highlighted entry's ActiveModel.
    let mut app = test_app();
    app.open_model_modal(vec![sample_model_entry("opus")]);
    assert_eq!(app.mode(), AppMode::ModalModel);
    assert!(app.active_modal.is_some(), "model picker occupies the slot");

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::SelectModel(ref m) if m.alias == "opus" && m.provider == "anthropic"),
        "expected SelectModel(opus), got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none(), "slot clears after selection");
}

#[test]
fn model_picker_keeps_wider_geometry_than_list_pickers() {
    // The model picker keeps its wider width clamp (40..72) on the slot, vs the
    // 36..64 list-picker clamp; a regression here truncates grouped provider
    // labels. Byte-parity with the removed `modal_size_for_mode` ModalModel arm.
    let area = Rect::new(0, 0, 200, 40);

    let mut model_app = test_app();
    model_app.open_model_modal(vec![sample_model_entry("opus")]);
    let (model_width, _) = modal_size_for_mode(&model_app, area);
    assert_eq!(model_width, 72, "model picker clamps width to 72 on a wide screen");

    let mut list_app = test_app();
    list_app.open_arg_picker("theme", "/theme", vec!["zo".to_string()]);
    let (list_width, _) = modal_size_for_mode(&list_app, area);
    assert_eq!(list_width, 64, "list picker clamps width to 64");
}

#[test]
fn permission_picker_selects_and_scrolls_on_the_slot() {
    // Enter confirms the pre-selected mode on the unified slot.
    let mut app = test_app();
    app.open_permission_picker_modal(runtime::PermissionMode::WorkspaceWrite);
    assert_eq!(app.mode(), AppMode::ModalPermissions);
    assert!(app.active_modal.is_some(), "permission picker occupies the slot");
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::SelectPermission(m) if m == runtime::PermissionMode::WorkspaceWrite),
        "expected SelectPermission(WorkspaceWrite), got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none());

    // A wheel notch moves the highlight before Enter (the picker has no
    // dedicated scroll method, so `Modal::scroll` synthesizes an arrow key).
    let mut app = test_app();
    app.open_permission_picker_modal(runtime::PermissionMode::ReadOnly);
    let _ = app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 10,
        row: 10,
        modifiers: KeyModifiers::NONE,
    });
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::SelectPermission(m) if m != runtime::PermissionMode::ReadOnly),
        "wheel-down must move the highlight off ReadOnly, got {action:?}"
    );
}

#[test]
fn api_key_paste_and_submit_on_the_slot() {
    // The API-key modal is on the slot but the host owns the clipboard: the
    // paste chord is intercepted to a ClipboardPaste action, then handle_paste
    // routes the text into the slot modal via the `active_modal_as` downcast.
    let mut app = test_app();
    app.open_connect_api_key_modal(ApiKeyConnectInfo {
        provider: "deepseek".to_string(),
        label: "DeepSeek".to_string(),
        auth_env: "DEEPSEEK_API_KEY".to_string(),
        models: vec!["deepseek-chat".to_string()],
    });
    assert_eq!(app.mode(), AppMode::ModalApiKey);
    assert!(app.active_modal.is_some(), "api-key modal occupies the slot");

    let paste = app
        .handle_key(KeyEvent {
            code: KeyCode::Char('v'),
            modifiers: KeyModifiers::CONTROL,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
        .unwrap();
    assert_eq!(paste, AppAction::ClipboardPaste, "Ctrl+V asks the host to paste");

    // The host reads the clipboard and routes it into the slot modal.
    app.handle_paste("sk-secret-123");

    // Enter submits the preset provider + the pasted key.
    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::ConnectApiKey { ref provider, ref api_key }
            if provider == "deepseek" && api_key == "sk-secret-123"),
        "expected ConnectApiKey(deepseek, pasted key), got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none());
}

#[test]
fn tool_toggle_stays_open_and_emits_toggle_on_the_slot() {
    // The `/tools` toggle is a Fullscreen slot modal that — unlike the other
    // pickers — stays open after a selection so several tools can be flipped.
    let mut app = test_app();
    app.open_tool_toggle_modal(vec![ToolToggleRow {
        name: "bash".to_string(),
        description: Some("run shell".to_string()),
        source: "builtin".to_string(),
        enabled: true,
    }]);
    assert_eq!(app.mode(), AppMode::ModalTools);
    assert!(app.active_modal.is_some(), "tool toggle occupies the slot");
    assert_eq!(
        app.active_modal.as_ref().unwrap().placement(),
        ModalPlacement::Fullscreen
    );

    // Space toggles the highlighted tool and keeps the modal on the slot.
    let action = app.handle_key(press(KeyCode::Char(' '))).unwrap();
    assert!(
        matches!(action, AppAction::ToggleTool { ref name, enabled } if name == "bash" && !enabled),
        "expected ToggleTool(bash,false), got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::ModalTools, "modal stays open after a toggle");
    assert!(app.active_modal.is_some(), "slot still holds the tool toggle");

    // Esc closes it.
    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none());
}

#[test]
fn effort_picker_submits_on_the_slot() {
    // The migrated effort slider lives on the unified slot; Enter re-enters the
    // text path as `/effort <budget>` (mirrors the legacy arm — one apply path).
    let mut app = test_app();
    app.open_effort_modal(None);
    assert_eq!(app.mode(), AppMode::ModalEffort);
    assert!(app.active_modal.is_some(), "effort picker occupies the slot");

    let action = app.handle_key(press(KeyCode::Enter)).unwrap();
    assert!(
        matches!(action, AppAction::Submit(ref t) if t.starts_with("/effort ")),
        "expected an /effort submit, got {action:?}"
    );
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.active_modal.is_none());
}

#[test]
fn effort_picker_declares_banner_placement() {
    // Effort is NOT anchored above the input: it declares `EffortBanner`, so
    // `draw_modals` positions it via `effort_modal_rect` (transcript column) and
    // `modal_size_for_mode` does not early-return its `desired_size`.
    let mut app = test_app();
    app.open_effort_modal(None);
    let placement = app.active_modal.as_ref().unwrap().placement();
    assert_eq!(placement, ModalPlacement::EffortBanner);

    // Because it is non-anchored, the anchored geometry path falls through to
    // the legacy default rather than the slider's own size.
    let (width, _) = modal_size_for_mode(&app, Rect::new(0, 0, 200, 40));
    assert!(width <= 64, "effort must not use the anchored desired_size path");
}

#[test]
fn alt_1_opens_model_picker_via_submit() {
    let mut app = test_app();
    app.enable_input();

    // Alt+1 is routed as a `/model` Submit so the existing dispatch builds the
    // provider-grouped entries (which need the live `cli`, unavailable in `App`).
    let action = app
        .handle_key(KeyEvent {
            code: KeyCode::Char('1'),
            modifiers: KeyModifiers::ALT,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
        .unwrap();
    assert!(
        matches!(action, AppAction::Submit(ref t) if t == "/model"),
        "expected `/model` submit, got {action:?}"
    );
}

#[test]
fn arg_picker_renders_command_caption_and_options() {
    let mut app = test_app();
    app.open_arg_picker(
        "theme",
        "/theme",
        vec!["zo".to_string(), "dark".to_string(), "light".to_string()],
    );

    let width = 80;
    let height = 24;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");

    let dump = (0..height)
        .map(|y| buffer_row(&terminal, width, y))
        .collect::<Vec<_>>()
        .join("\n");

    // The modal must route through render.rs's unified slot arm and paint the
    // command caption plus every option label (end-to-end: open → size → draw).
    assert!(dump.contains("/theme"), "modal caption missing:\n{dump}");
    assert!(dump.contains("zo"), "option `zo` missing:\n{dump}");
    assert!(dump.contains("dark"), "option `dark` missing:\n{dump}");
    assert!(dump.contains("light"), "option `light` missing:\n{dump}");
}

#[test]
fn file_picker_inserts_with_leading_space_after_text() {
    let mut app = test_app();
    app.enable_input();

    // Existing buffer text ending in a non-space char.
    for ch in "review".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }

    app.open_file_picker_for_test(vec!["src/lib.rs".to_string()]);
    let _ = app.handle_key(press(KeyCode::Enter));

    // A separating space is inserted before the mention so it stays a
    // distinct token.
    assert_eq!(app.input().text(), "review @src/lib.rs ");
}

#[test]
fn file_picker_esc_cancels_without_inserting() {
    let mut app = test_app();
    app.enable_input();

    app.open_file_picker_for_test(vec!["src/main.rs".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalFile);

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(
        app.input().text().is_empty(),
        "cancel must not insert a token"
    );
}

#[test]
fn single_esc_in_normal_is_a_noop() {
    // A lone Esc in Normal mode must not emit any externally-visible
    // action — it only arms the double-tap window. This guards the
    // "do not break single-Esc semantics" requirement.
    let mut app = test_app();
    app.enable_input();

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn esc_esc_in_normal_emits_rewind_checkpoint() {
    // Two Esc presses within the double-tap window emit the combined
    // conversation+code rewind action.
    let mut app = test_app();
    app.enable_input();

    let first = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(first, AppAction::None, "first Esc only arms the window");

    let second = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(second, AppAction::RewindCheckpoint);
}

#[test]
fn esc_esc_resets_after_firing() {
    // After firing, the window resets so a third lone Esc does not
    // re-fire — it takes a fresh pair.
    let mut app = test_app();
    app.enable_input();

    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::None
    );
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::RewindCheckpoint
    );
    // Third Esc must only re-arm (not fire) since the pair was consumed.
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::None
    );
    // Fourth completes a fresh pair.
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::RewindCheckpoint
    );
}

#[test]
fn rewind_confirm_modal_y_confirms() {
    // The Esc-Esc rewind is destructive, so the outer loop opens a
    // confirmation card (`open_rewind_confirm`) instead of rewinding
    // immediately. `y` confirms, returning `ConfirmRewind` and Normal mode.
    let mut app = test_app();
    app.enable_input();

    app.open_rewind_confirm(vec!["Rewind the latest turn?".to_string()]);
    assert_eq!(app.mode(), AppMode::ModalConfirmRewind);
    assert!(app.rewind_confirm_lines().is_some());

    let confirm = app.handle_key(press(KeyCode::Char('y'))).unwrap();
    assert_eq!(confirm, AppAction::ConfirmRewind);
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(app.rewind_confirm_lines().is_none());
}

#[test]
fn rewind_confirm_modal_cancels_on_n_and_esc() {
    // A reflexive Esc burst (e.g. after denying a permission prompt) must
    // never confirm: `n`, Esc, and any non-`y` key all cancel with no rewind.
    for cancel in [KeyCode::Char('n'), KeyCode::Esc] {
        let mut app = test_app();
        app.enable_input();
        app.open_rewind_confirm(vec!["Rewind?".to_string()]);

        let action = app.handle_key(press(cancel)).unwrap();
        assert_eq!(action, AppAction::None, "cancel key must not confirm");
        assert_eq!(app.mode(), AppMode::Normal);
        assert!(app.rewind_confirm_lines().is_none());
    }
}

#[test]
fn esc_interrupts_inflight_turn_like_claude_code() {
    // CC parity: the spinner advertises `esc to interrupt`, so Esc during a
    // live turn must actually send CancelTurn (it used to be Ctrl+C only).
    let (mut app, mut cmd_rx) = test_app_with_cmd();
    app.enable_input();
    app.begin_turn_with_generation(0);

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert!(
        matches!(cmd_rx.try_recv(), Ok(AgentCommand::CancelTurn)),
        "Esc during a turn must request a turn cancel"
    );

    // A second Esc while the turn is still live re-sends the cancel —
    // it must never fall through to the rewind double-tap.
    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert!(matches!(cmd_rx.try_recv(), Ok(AgentCommand::CancelTurn)));
}

#[test]
fn esc_clears_input_draft_when_idle() {
    // CC parity: with no turn running, Esc clears a non-empty draft; the
    // rewind double-tap only arms once the composer is empty.
    let mut app = test_app();
    app.enable_input();
    for ch in "draft text".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(app.input().text(), "draft text");

    let action = app.handle_key(press(KeyCode::Esc)).unwrap();
    assert_eq!(action, AppAction::None);
    assert!(
        app.input().text().is_empty(),
        "Esc must clear the idle draft"
    );

    // The clearing Esc must not arm the rewind window: the next Esc only
    // arms, and the one after fires.
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::None
    );
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::RewindCheckpoint
    );
}

#[test]
fn leaked_sgr_mouse_bytes_do_not_land_in_prompt() {
    let mut app = test_app();
    app.enable_input();

    for ch in "[<35;24;26M[<35;36;25M".chars() {
        let _ = app.handle_key(press(KeyCode::Char(ch)));
    }

    assert_eq!(app.input().text(), "");
}

#[test]
fn esc_does_not_fire_rewind_while_modal_open() {
    // Esc inside a modal cancels the modal (handled in the mode dispatch)
    // and must never arm or fire the Normal-mode rewind double-tap.
    let mut app = test_app();
    app.enable_input();
    app.open_file_picker_for_test(vec!["src/main.rs".to_string()]);

    // Esc closes the modal → Normal mode, no rewind.
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::None
    );
    assert_eq!(app.mode(), AppMode::Normal);
    // The next single Esc (now in Normal) must only arm, not fire — proving
    // the modal-cancel Esc did not leak into the double-tap state.
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).unwrap(),
        AppAction::None
    );
}

#[test]
fn at_after_text_opens_inline_mention_not_modal() {
    let mut app = test_app();
    // A bare buffer: `@` opens the full picker modal, not an inline popup.
    app.set_input_text("");
    assert!(!app.mention_opens_inline(), "empty buffer → modal");
    // After text: `@` begins an inline mention instead.
    app.set_input_text("fix ");
    assert!(app.mention_opens_inline(), "text before cursor → inline");
}

#[test]
fn mention_hint_active_and_ranks_workspace_files() {
    let mut app = test_app();
    app.enable_input();
    app.set_input_text("see @con");
    app.workspace_files = vec![
        "src/convert.rs".to_string(),
        "src/config.rs".to_string(),
        "README.md".to_string(),
    ];
    assert!(app.mention_hint_active(), "@token is active");
    let sugg = app.mention_hint_suggestions();
    assert!(!sugg.is_empty(), "ranked suggestions present");
    assert!(
        sugg.iter().all(|s| s.to_lowercase().contains("con")),
        "every suggestion matches 'con': {sugg:?}"
    );
    // A slash buffer is the slash-hint's domain, never the mention popup.
    app.set_input_text("/help");
    assert!(!app.mention_hint_active(), "slash buffer is not a mention");
}

#[test]
fn mention_enter_replaces_token_and_records_frecency() {
    let mut app = test_app();
    app.enable_input();
    app.set_input_text("see @conf");
    app.workspace_files = vec!["src/config.rs".to_string()];
    // Tab accepts the first suggestion; arrows are reserved for transcript
    // scrolling while the composer stays focused.
    let _ = app.handle_key(press(KeyCode::Tab));
    assert_eq!(app.input.text(), "see @src/config.rs ");
    // The pick is recorded so it ranks higher on the next mention.
    assert!(
        app.mention_history
            .frecency_scores()
            .contains_key("src/config.rs"),
        "selected mention recorded for frecency"
    );
}

#[test]
fn usage_split_breaks_down_current_window_not_session_total() {
    // Regression: the `⤷ N new · M cached` ledger line must break down the
    // *current* context window, not session-cumulative totals. The bug fed a
    // huge cumulative cache-read (8.6M) into the split, so it rendered
    // `8.6M cached` under a `191.9k / 1.0M` window — a number larger than the
    // whole limit. The split must come from the latest request (`current`).
    let mut app = test_app();
    let cumulative = runtime::TokenUsage {
        input_tokens: 164,
        output_tokens: 5_000,
        cache_creation_input_tokens: 1_000,
        cache_read_input_tokens: 8_600_000,
    };
    let current = runtime::TokenUsage {
        input_tokens: 164,
        output_tokens: 120,
        cache_creation_input_tokens: 400,
        cache_read_input_tokens: 191_500,
    };
    app.push_block(RenderBlock::Usage {
        ctx_tokens: 191_900,
        cumulative,
        current,
    });

    assert_eq!(app.hud_state.ctx_new_input, 164, "new = current input");
    assert_eq!(
        app.hud_state.ctx_cached, 191_500,
        "cached = current-turn cache read, not the 8.6M session total"
    );
    assert!(
        app.hud_state.ctx_new_input + app.hud_state.ctx_cached <= app.hud_state.ctx_used,
        "the split must fit within the ctx window it describes ({} + {} > {})",
        app.hud_state.ctx_new_input,
        app.hud_state.ctx_cached,
        app.hud_state.ctx_used
    );
}

#[test]
fn usage_ctx_only_snapshot_preserves_prior_split() {
    // A `message_start` ctx-only snapshot (empty cumulative) advances ctx but
    // must not zero the split — it keeps the last real breakdown until the
    // response completes, so the ledger never blinks `0 new · 0 cached`.
    let mut app = test_app();
    app.push_block(RenderBlock::Usage {
        ctx_tokens: 191_900,
        cumulative: runtime::TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 191_000,
        },
        current: runtime::TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 191_000,
        },
    });
    assert_eq!(app.hud_state.ctx_cached, 191_000);

    // Next request lands: ctx known, cumulative still empty (cost unknown).
    app.push_block(RenderBlock::Usage {
        ctx_tokens: 205_000,
        cumulative: runtime::TokenUsage::default(),
        current: runtime::TokenUsage::default(),
    });
    assert_eq!(app.hud_state.ctx_used, 205_000, "ctx advances immediately");
    assert_eq!(
        app.hud_state.ctx_cached, 191_000,
        "prior split is preserved across an empty-cumulative ctx snapshot"
    );
}

#[test]
fn slash_on_empty_enabled_input_opens_inline_hint() {
    let mut app = test_app();
    app.enable_input();
    assert_eq!(app.mode(), AppMode::Normal);

    let _ = app.handle_key(press(KeyCode::Char('/')));

    // A leading "/" is an ordinary character now: it lands in the buffer and
    // surfaces the inline command hint, with no separate palette mode.
    assert_eq!(app.mode(), AppMode::Normal);
    assert_eq!(app.input().text(), "/");
}

#[test]
fn slash_mid_text_inserts_as_a_literal() {
    let mut app = test_app();
    app.enable_input();
    let _ = app.handle_key(press(KeyCode::Char('a')));
    let _ = app.handle_key(press(KeyCode::Char('/')));

    // "/" mid-text is an ordinary character, exactly like a leading slash.
    assert_eq!(app.mode(), AppMode::Normal);
    assert_eq!(app.input().text(), "a/");
}

// ---------------------------------------------------------------------------
// Auto-follow vs. manual scroll (the "scroll up, then it jumps back down" bug)
// ---------------------------------------------------------------------------

/// Fill the transcript with enough finished blocks to overflow a small
/// viewport, then draw once so `regions` and the layout cache are populated.
fn app_with_overflowing_transcript() -> (App, ratatui::Terminal<TestBackend>) {
    let mut app = test_app();
    for i in 0..60 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("line {i}"),
            done: true,
        });
    }
    let backend = TestBackend::new(80, 20);
    let mut terminal = ratatui::Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw populates regions");
    (app, terminal)
}

#[test]
fn end_turn_preserves_reader_position_after_scrolling_up_mid_stream() {
    let (mut app, _terminal) = app_with_overflowing_transcript();

    // User scrolls up to read while the response is still streaming: this turns
    // auto-follow off and leaves the viewport above the tail.
    app.begin_turn_with_generation(0);
    let transcript = app.regions.expect("regions after draw").transcript;
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: transcript.x + 1,
        row: transcript.y + 1,
        modifiers: KeyModifiers::NONE,
    })
    .expect("wheel handled");
    assert!(!app.transcript_view.follow_output, "scrolling up disables auto-follow");

    // The turn settles. CC parity: a reader who scrolled up keeps their place —
    // the old unconditional snap ("always land on the answer"), combined with
    // auto-started follow-up turns, yanked the viewport at every turn boundary
    // and read as broken mouse scrolling during long agentic sessions.
    let scrolled_up = app.transcript.scroll();
    app.end_turn();
    assert!(
        !app.transcript_view.follow_output,
        "end_turn must not re-arm auto-follow over a reader's scroll position"
    );
    assert_eq!(
        app.transcript.scroll(),
        scrolled_up,
        "end_turn must leave the scrolled-up viewport where the reader put it"
    );
}

#[test]
fn follow_latest_reenables_auto_follow_for_command_confirmation() {
    let (mut app, _terminal) = app_with_overflowing_transcript();

    // User has scrolled up to read earlier output, turning auto-follow off.
    let transcript = app.regions.expect("regions after draw").transcript;
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: transcript.x + 1,
        row: transcript.y + 1,
        modifiers: KeyModifiers::NONE,
    })
    .expect("wheel handled");
    assert!(!app.transcript_view.follow_output, "scrolling up disables auto-follow");
    let scrolled_up = app.transcript.scroll();

    // Running a slash command (e.g. /goal) calls follow_latest so its
    // confirmation is visible instead of landing off-screen — the "/goal looked
    // like a no-op" fix.
    app.follow_latest();
    assert!(
        app.transcript_view.follow_output,
        "follow_latest must re-enable auto-follow for the command confirmation"
    );
    assert_ne!(
        app.transcript.scroll(),
        scrolled_up,
        "follow_latest must scroll back to the tail, not leave the reader scrolled up"
    );
}

#[test]
fn end_turn_snaps_to_tail_when_following() {
    let (mut app, _terminal) = app_with_overflowing_transcript();
    app.begin_turn_with_generation(0);
    // User did NOT scroll: following is still armed.
    assert!(app.transcript_view.follow_output);

    app.end_turn();
    assert!(
        app.transcript_view.follow_output,
        "following stays on across a settled turn"
    );
    // The tail sentinel is set so the next draw pins to the bottom.
    assert_eq!(
        app.transcript.scroll(),
        u16::MAX,
        "a following turn snaps to the tail on settle"
    );
}

#[test]
fn begin_turn_preserves_reader_position_when_not_following() {
    let (mut app, _terminal) = app_with_overflowing_transcript();

    // User scrolled up during/after a previous turn — following is off.
    let transcript = app.regions.expect("regions after draw").transcript;
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: transcript.x + 1,
        row: transcript.y + 1,
        modifiers: KeyModifiers::NONE,
    })
    .expect("wheel handled");
    assert!(!app.transcript_view.follow_output);
    let scrolled_up = app.transcript.scroll();

    // An auto-started turn (queued drain, loop iteration, agent-result
    // re-injection) must not yank the reader to the tail. Turns the user
    // submits themselves still snap: the composer's submit path re-arms
    // follow before begin_turn runs.
    app.begin_turn_with_generation(0);
    assert!(
        !app.transcript_view.follow_output,
        "begin_turn must not re-arm auto-follow over a reader's scroll position"
    );
    assert_eq!(
        app.transcript.scroll(),
        scrolled_up,
        "begin_turn must leave the scrolled-up viewport where the reader put it"
    );
}

/// The streaming reasoning-title label must be byte-identical whether it is
/// computed from the whole accumulated thought or from the O(delta)
/// `reasoning_title_source` splice — for every split point. This pins the
/// O(n²)→O(delta) optimization to behavior-preserving: the visible label can
/// never change because the rebuild got cheaper.
#[test]
fn reasoning_title_source_matches_full_accumulation_at_every_split() {
    // Mix of leading blank lines, a long first title line, and a long body so
    // the cheap path (borrowed `prior`) and the open-first-line path (spliced
    // tail) are both exercised across the splits.
    let whole = "\n\n  Analyzing the nationality corridor filter and its edge cases\n\
                 Now I will inspect applyLane and the merge queue wiring in detail.\n\
                 Then verify the TypeScript build stays green.";
    let expected = reasoning_activity_summary(whole);
    // The title must actually be the first non-empty line, not the fallback.
    assert!(
        expected.starts_with("Analyzing the nationality corridor"),
        "fixture should yield a real title, got {expected:?}"
    );

    for split in 0..=whole.len() {
        if !whole.is_char_boundary(split) {
            continue;
        }
        let (prior, delta) = whole.split_at(split);
        let source = reasoning_title_source(prior, delta);
        assert_eq!(
            reasoning_activity_summary(&source),
            expected,
            "title diverged at split {split} (prior={prior:?}, delta={delta:?})"
        );
    }
}

/// Before any non-empty line has arrived, the label falls back to the Zo cue
/// — and the cheap source path must preserve that too (no panic on all-blank
/// prior, correct fallback).
#[test]
fn reasoning_title_source_handles_all_blank_prefix() {
    let source = reasoning_title_source("\n  \n", "");
    let label = reasoning_activity_summary(&source);
    assert_eq!(
        label,
        format!("{}…", crate::tui::blocks::reasoning::ZO_REVEAL_VERBS[0]),
        "an all-blank reasoning prefix keeps the Thinking… fallback"
    );
}

/// Once `prior` already holds a terminated non-empty first line, the splice is a
/// zero-copy borrow of `prior` (the delta cannot change a settled first line).
#[test]
fn reasoning_title_source_bounds_unterminated_first_line_source() {
    let prior = "a".repeat(10_000);
    let delta = "b".repeat(1_000);
    let source = reasoning_title_source(&prior, &delta);
    let owned = match source {
        std::borrow::Cow::Owned(owned) => owned,
        std::borrow::Cow::Borrowed(_) => panic!("unterminated first line must be owned"),
    };
    assert!(
        owned.len() <= 512,
        "unterminated no-newline source must stay bounded, got {} bytes",
        owned.len()
    );
    assert_eq!(reasoning_activity_summary(&owned), "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa…");
}

#[test]
fn reasoning_activity_title_caches_clear_across_turn_boundaries() {
    let mut app = test_app();

    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(612),
        text: "Old finalized title
body".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.reasoning_activity_titles.get(&BlockId(612)).map(String::as_str),
        Some("Old finalized title")
    );
    app.end_turn();
    assert!(app.reasoning_activity_titles.is_empty());
    assert!(app.reasoning_activity_open_titles.is_empty());

    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(612),
        text: "New ".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "New",
        "same BlockId in a later turn must not reuse finalized stale title"
    );
    app.end_turn();
    assert!(app.reasoning_activity_titles.is_empty());
    assert!(app.reasoning_activity_open_titles.is_empty());

    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(612),
        text: "Open stale ".to_string(),
        signature: None,
        done: false,
    });
    assert!(app.reasoning_activity_open_titles.contains_key(&BlockId(612)));
    app.end_turn();
    assert!(app.reasoning_activity_titles.is_empty());
    assert!(app.reasoning_activity_open_titles.is_empty());

    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(612),
        text: "Fresh title
".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Fresh title",
        "same BlockId in a later turn must not append stale open title"
    );
}

#[test]
fn reasoning_activity_title_rolls_to_the_newest_paragraph() {
    // GPT streams its reasoning summary as paragraphs on ONE block; the
    // spinner title must follow the newest paragraph instead of freezing on
    // the first one for the whole reasoning phase.
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(701),
        text: "Scanning the router policy\nlooking at scoring".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Scanning the router policy"
    );

    // Paragraph break + new topic in one delta → the title re-rolls.
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(701),
        text: " more.\n\nNow checking the quota ladder\ndetails".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Now checking the quota ladder",
        "title must follow the newest paragraph, not freeze on the first"
    );

    // A boundary landing at the very end of a delta re-arms titling: the
    // next fragment starts the new title.
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(701),
        text: " tail.\n\n".to_string(),
        signature: None,
        done: false,
    });
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(701),
        text: "Final synthesis pass\n".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Final synthesis pass"
    );
}

#[test]
fn reasoning_activity_open_title_cache_skips_blank_lines_before_split_title() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(614),
        text: concat!("\n", "   \n", "Actual ").to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual"
    );
    assert_eq!(
        app.reasoning_activity_open_titles
            .get(&BlockId(614))
            .map(String::as_str),
        Some("Actual "),
        "blank lines before the first title fragment must not enter the open cache"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(614),
        text: "title\nbody".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual title"
    );
    assert_eq!(
        app.reasoning_activity_titles
            .get(&BlockId(614))
            .map(String::as_str),
        Some("Actual title")
    );
    assert!(
        !app.reasoning_activity_open_titles.contains_key(&BlockId(614)),
        "newline-terminated first title line is promoted out of the open cache"
    );
}

#[test]
fn reasoning_activity_open_title_cache_does_not_spend_budget_on_leading_blank() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(613),
        text: format!("{}Actual ", " ".repeat(511)),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual"
    );
    assert_eq!(
        app.reasoning_activity_open_titles.get(&BlockId(613)).map(String::as_str),
        Some("Actual "),
        "open title cache stores the title fragment, not the leading blank prefix"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(613),
        text: "title
".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual title"
    );
    assert_eq!(
        app.reasoning_activity_titles.get(&BlockId(613)).map(String::as_str),
        Some("Actual title")
    );
}

#[test]
fn reasoning_activity_open_title_cache_preserves_split_title_after_long_blank_prefix() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(611),
        text: " ".repeat(10_000),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Thinking…"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(611),
        text: "Actual ".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual"
    );
    assert!(
        app.reasoning_activity_open_titles.contains_key(&BlockId(611)),
        "unterminated title fragment stays in the bounded open-title cache"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(611),
        text: "title
body".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual title"
    );
    assert_eq!(
        app.reasoning_activity_titles.get(&BlockId(611)).map(String::as_str),
        Some("Actual title")
    );
    assert!(
        !app.reasoning_activity_open_titles.contains_key(&BlockId(611)),
        "newline-terminated title moves from open cache to finalized cache"
    );
}

#[test]
fn reasoning_activity_cache_preserves_title_after_long_blank_prefix() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.push_block(RenderBlock::Reasoning {
        id: BlockId(610),
        text: " ".repeat(10_000),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Thinking…"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(610),
        text: "Actual title after blank prefix
".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual title after blank prefix"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(610),
        text: "body continuation should not replace title".to_string(),
        signature: None,
        done: false,
    });
    assert_eq!(
        app.turn_activity().expect("turn active").current_action(),
        "Actual title after blank prefix",
        "once discovered, the first non-empty reasoning title stays stable"
    );

    app.push_block(RenderBlock::Reasoning {
        id: BlockId(610),
        text: String::new(),
        signature: None,
        done: true,
    });
    assert!(
        !app.reasoning_activity_titles.contains_key(&BlockId(610)),
        "done reasoning blocks release their cached spinner title"
    );
}

#[test]
fn reasoning_title_source_does_not_let_long_blank_prefix_hide_delta_title() {
    let prior = " ".repeat(10_000);
    let delta = "Actual title after blank prefix";
    let source = reasoning_title_source(&prior, delta);
    assert!(
        matches!(source, std::borrow::Cow::Owned(_)),
        "unterminated blank prefix plus delta is synthesized"
    );
    assert_eq!(reasoning_activity_summary(&source), delta);
}

#[test]
fn reasoning_title_source_borrows_once_first_line_settled() {
    let source = reasoning_title_source("Settled title line\nmore body", " and more delta");
    assert!(
        matches!(source, std::borrow::Cow::Borrowed(_)),
        "a settled first line must not reallocate per delta"
    );
}

#[test]
fn provider_agnostic_stream_and_turn_ticks_share_frame_budget() {
    use crate::tui::render_schedule::{STREAM_FRAME_INTERVAL, StreamFrameGate};
    use std::time::{Duration, Instant};

    struct Regime {
        label: &'static str,
        gap_ms: u64,
        chars: usize,
        n: usize,
    }

    // This is a pass/fail reproduction of the provider-common stutter class:
    // provider-arrival draws and turn-tick draws used to spend separate budgets,
    // so a stream draw at T could be immediately overpainted by the active-turn
    // tick at T+1ms. That floods slower terminals, which users perceive as
    // "fast output → pause → resume" for both Claude token deltas and GPT-style
    // larger chunks. A manual terminal paint-speed repro is noisy; the objective
    // invariant is that both redraw drivers never produce sub-frame full draws.
    let regimes = [
        Regime { label: "claude-like tokens", gap_ms: 25, chars: 14, n: 40 },
        Regime { label: "gpt-like chunks", gap_ms: 45, chars: 45, n: 28 },
    ];

    for r in regimes {
        let mut app = test_app();
        app.begin_turn_with_generation(0);
        let t0 = Instant::now();
        let mut gate = StreamFrameGate::new_ready(t0, STREAM_FRAME_INTERVAL);
        let mut next_tick_ms = 1u64; // deliberately races right after the first provider draw.
        let mut next_arrival_idx = 0usize;
        let total_ms = r.n as u64 * r.gap_ms + 300;
        let mut last_draw_ms: Option<u64> = None;
        let mut min_draw_gap_ms = u64::MAX;
        let mut draws = 0usize;

        for now_ms in 0..=total_ms {
            let now = t0 + Duration::from_millis(now_ms);

            if next_arrival_idx < r.n && now_ms == next_arrival_idx as u64 * r.gap_ms {
                let done = next_arrival_idx + 1 == r.n;
                app.buffer_paced_at(now, BlockId(1), "x".repeat(r.chars), done);
                next_arrival_idx += 1;
                if gate.on_stream_update(now).draws_now() {
                    app.drip_stream_at(now, None);
                    if let Some(prev) = last_draw_ms {
                        min_draw_gap_ms = min_draw_gap_ms.min(now_ms.saturating_sub(prev));
                    }
                    last_draw_ms = Some(now_ms);
                    draws += 1;
                    gate.note_stream_draw(now);
                }
            }

            if now_ms == next_tick_ms {
                next_tick_ms += 33;
                app.drip_stream_at(now, None);
                if gate.on_stream_tick(now, true).draws_now() {
                    if let Some(prev) = last_draw_ms {
                        min_draw_gap_ms = min_draw_gap_ms.min(now_ms.saturating_sub(prev));
                    }
                    last_draw_ms = Some(now_ms);
                    draws += 1;
                    gate.note_stream_draw(now);
                }
            }
        }

        assert!(draws > 2, "{label}: simulation should exercise both redraw drivers", label = r.label);
        let min_allowed_ms = u64::try_from(STREAM_FRAME_INTERVAL.as_millis())
            .expect("stream frame interval fits in u64 milliseconds");
        assert!(
            min_draw_gap_ms >= min_allowed_ms,
            "{label}: stream/tick redraws must share one frame budget; saw a {min_draw_gap_ms}ms full-redraw gap",
            label = r.label
        );
    }
}

/// Regression: the drip clock must stay fresh across an inter-delta idle gap.
///
/// Before the fix, `last_drip` froze at the last actual reveal while the pacer
/// idled empty, so the first drip after a clumpy provider's ~470ms gap
/// measured dt ≈ the whole gap, earned ~a delta's worth of characters, and
/// dumped the freshly arrived backlog in one frame — the clump→pause stutter.
/// With the clock pinned to the tick grid the same drip sees dt ≈ one frame
/// and meters the delta smoothly.
#[test]
fn idle_gap_keeps_continuation_metered() {
    use std::time::{Duration, Instant};

    let mut app = test_app();
    app.begin_turn_with_generation(0);
    let t0 = Instant::now();
    app.buffer_paced_at(t0, BlockId(1), "x".repeat(40), false);

    // Run the 33ms tick grid like the live loop: the opening drains in the
    // first frames, and the trailing ticks idle on the empty pacer through a
    // ~470ms silent gap (44 ticks ≈ 1.45s total).
    for i in 1..=44u64 {
        app.drip_stream_at(t0 + Duration::from_millis(33 * i), None);
    }
    let drained = visible_text_deltas(&app).chars().count();
    assert_eq!(drained, 40, "opening must fully drain before the idle gap");

    // Clumpy-provider continuation lands right after the last idle tick; the
    // next tick frame must meter it, not dump it whole.
    let arrive = t0 + Duration::from_millis(33 * 44 + 10);
    app.buffer_paced_at(arrive, BlockId(1), "y".repeat(20), false);
    app.drip_stream_at(t0 + Duration::from_millis(33 * 45), None);

    let first_frame = visible_text_deltas(&app)
        .chars()
        .count()
        .saturating_sub(drained);
    assert!(
        (1..10).contains(&first_frame),
        "continuation after an idle gap must be metered across frames, \
         got {first_frame} chars on the first frame"
    );
}

// Production-faithful pacer measurement (fact-finding, not a pass/fail gate).
//
// Unlike measure_pacer_latency_profile (which drips on a perfect 16 ms grid),
// this reproduces the REAL turn-render select loop in turn_controller:
// a token-arrival driver (Claude per-token deltas) plus an independent 33 ms
// render_tick driver, both gated through a StreamFrameGate(16 ms). A reveal only
// becomes visible on a frame the gate actually let DRAW. This measures the
// cadence the USER sees (gap between drawn frames + chars revealed per drawn
// frame), which is where Claude's "뚝뚝" lives.
//
//   cargo test -p zo-cli --lib -- --ignored --nocapture measure_production_pacer_cadence
#[test]
#[ignore = "measurement: run with --ignored --nocapture"]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn measure_production_pacer_cadence() {
    use crate::tui::render_schedule::{StreamFrameGate, STREAM_FRAME_INTERVAL};
    use std::time::{Duration, Instant};

    struct Regime {
        label: &'static str,
        gap_ms: u64,
        chars: usize,
        n: usize,
    }
    const RENDER_TICK_MS: u64 = 33;
    let regimes = [
        Regime { label: "claude tokens (14c/25ms)", gap_ms: 25, chars: 14, n: 60 },
        Regime { label: "claude fast (8c/12ms)", gap_ms: 12, chars: 8, n: 80 },
        Regime { label: "gpt chunks (45c/45ms)", gap_ms: 45, chars: 45, n: 30 },
        Regime { label: "gpt w/ hitch (45c/80ms)", gap_ms: 80, chars: 45, n: 20 },
        // Claude-over-OAuth sentence clumps measured in the wild
        // (~/.zo/logs/delta-trace.log): ~20 chars every ~470ms. The regime
        // that exposed the frozen-drip-clock stutter (last_drip stuck at the
        // previous reveal → first drip after the gap dumped the delta whole).
        Regime { label: "claude clumps (20c/470ms)", gap_ms: 470, chars: 20, n: 12 },
    ];

    println!(
        "\n{:<26} {:>7} {:>13} {:>14} {:>15}",
        "regime", "draws", "max gap (ms)", "avg gap (ms)", "max char/draw"
    );
    println!("{}", "-".repeat(80));

    for r in &regimes {
        let mut app = test_app();
        app.begin_turn_with_generation(0);
        let t0 = Instant::now();
        let mut gate = StreamFrameGate::new_ready(t0, STREAM_FRAME_INTERVAL);

        let total_ms = r.n as u64 * r.gap_ms + 600;
        let mut next_tick = RENDER_TICK_MS;
        let mut next_arrival_idx = 0usize;

        let mut prev_visible = 0usize;
        let mut last_draw_ms: Option<u64> = None;
        let mut max_gap = 0u64;
        let mut gap_sum = 0u64;
        let mut draws = 0u64;
        let mut max_char_per_draw = 0usize;

        // Simulate millisecond-by-millisecond; at each ms, fire whichever
        // driver(s) are due, exactly like the real `tokio::select!`.
        for now_ms in 0..=total_ms {
            let now = t0 + Duration::from_millis(now_ms);

            // Driver 1: token arrival -> buffer into pacer, then gate a draw.
            let mut drew_this_ms = false;
            if next_arrival_idx < r.n && now_ms == next_arrival_idx as u64 * r.gap_ms {
                let done = next_arrival_idx + 1 == r.n;
                app.buffer_paced_at(now, BlockId(1), "x".repeat(r.chars), done);
                next_arrival_idx += 1;
                if gate.on_stream_update(now).draws_now() {
                    app.drip_stream_at(now, None);
                    drew_this_ms = true;
                }
            }

            // Driver 2: render_tick (advance_tick drips, then gate.on_tick).
            if now_ms == next_tick {
                next_tick += RENDER_TICK_MS;
                app.drip_stream_at(now, None); // mirrors advance_tick's drip
                // turn is active, so tick always has stream/turn work and must
                // share the provider-arrival frame budget.
                if gate.on_stream_tick(now, true).draws_now() {
                    drew_this_ms = true;
                    gate.note_stream_draw(now);
                }
            }

            if drew_this_ms {
                draws += 1;
                let vis = visible_text_deltas(&app).chars().count();
                max_char_per_draw = max_char_per_draw.max(vis.saturating_sub(prev_visible));
                prev_visible = vis;
                if let Some(prev) = last_draw_ms {
                    let gap = now_ms - prev;
                    max_gap = max_gap.max(gap);
                    gap_sum += gap;
                }
                last_draw_ms = Some(now_ms);
            }
        }

        let avg_gap = if draws > 1 { gap_sum / (draws - 1) } else { 0 };
        println!(
            "{:<26} {:>7} {:>13} {:>14} {:>15}",
            r.label, draws, max_gap, avg_gap, max_char_per_draw
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Live agent view: mouse click targets + Ctrl+O / Ctrl+G surface parity.
//
// The user-visible contract: while agents run, the pinned panel and the
// spawn-family tool card are click targets that open the live agent view, and
// the keys that open the agent surfaces keep working (toggle / switch) while
// one of those surfaces is already up — they were dead keys there before.
// ---------------------------------------------------------------------------

fn running_agent_summary(name: &str) -> AgentTaskSummary {
    AgentTaskSummary {
        id: format!("agent-{name}"),
        name: name.to_string(),
        status: "running".to_string(),
        model: "claude-opus-4-8".to_string(),
        ..AgentTaskSummary::default()
    }
}

fn workflow_view_for_plan_label_test() -> WorkflowView {
    WorkflowView {
        run_id: "plan-viewer-run".to_string(),
        name: "plan-viewer".to_string(),
        description: "Plan-to-Executor viewer".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        synthesizing: false,
        phases: vec![WorkflowPhaseRow {
            step_id: Some("build-viewer".to_string()),
            plan_step: None,
            id: "build-viewer".to_string(),
            kind: "fanout".to_string(),
            status: "running".to_string(),
            round: 1,
            completed: 0,
            failed: 0,
            still_running: 1,
            total: 1,
            agents: vec![WorkflowAgentRow {
                id: "viewer-agent".to_string(),
                name: "viewer-builder".to_string(),
                status: "running".to_string(),
                current_tool: Some("edit_file".to_string()),
                ..WorkflowAgentRow::default()
            }],
        }],
    }
}

fn spawn_tool_call_block(id: u64, name: &str) -> RenderBlock {
    RenderBlock::ToolCall {
        id: BlockId(id),
        tool_call_id: ToolCallId(format!("call-{id}")),
        name: name.to_string(),
        summary: "3 agents".to_string(),
        preview: ToolPreview::Generic {
            name: name.to_string(),
            input_summary: "spawn agents".to_string(),
        },
        status: ToolCallStatus::Running,
    }
}

fn left_click(column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

fn left_release(column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind: MouseEventKind::Up(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

#[test]
fn agent_panel_click_opens_live_agent_view() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    app.hud_state.agents = vec![running_agent_summary("researcher")];
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw records panel rect");
    let panel = app
        .agent_panel_click_rect
        .expect("pinned live-agent panel must be painted while agents run");

    let action = app
        .handle_mouse(left_click(panel.x + panel.width / 2, panel.y))
        .expect("panel click handled");

    assert_eq!(
        action,
        AppAction::OpenWorkflowViewer,
        "clicking the pinned agent panel must open the live agent view"
    );
}

#[test]
fn agent_panel_row_click_opens_that_agents_view() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    // Two agents; the panel sorts by (created_at, name) so `alpha` renders on
    // the first agent row (panel line 1), `beta` on line 2.
    app.hud_state.agents = vec![
        running_agent_summary("beta"),
        running_agent_summary("alpha"),
    ];
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw records row targets");
    let panel = app.agent_panel_click_rect.expect("panel painted");

    // Line 0 is the header; the first agent row is one line below it.
    let action = app
        .handle_mouse(left_click(panel.x + 2, panel.y + 1))
        .expect("row click handled");
    assert_eq!(
        action,
        AppAction::OpenAgentInViewer("agent-alpha".to_string()),
        "clicking an agent row must open the viewer focused on THAT agent"
    );

    // The second agent row opens its own id.
    let action = app
        .handle_mouse(left_click(panel.x + 2, panel.y + 2))
        .expect("second row click handled");
    assert_eq!(action, AppAction::OpenAgentInViewer("agent-beta".to_string()));
}

#[test]
fn agent_panel_header_click_opens_aggregate_view() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    app.hud_state.agents = vec![
        running_agent_summary("alpha"),
        running_agent_summary("beta"),
    ];
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw records panel rect");
    let panel = app.agent_panel_click_rect.expect("panel painted");

    // The header row (line 0) is not an agent row → aggregate view.
    let action = app
        .handle_mouse(left_click(panel.x + 2, panel.y))
        .expect("header click handled");
    assert_eq!(
        action,
        AppAction::OpenWorkflowViewer,
        "clicking the header (not a specific agent) opens the aggregate view"
    );
}

#[test]
fn agent_panel_row_hover_underlines_and_repaints_only_on_change() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    app.hud_state.agents = vec![running_agent_summary("alpha")];
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw records row targets");
    let panel = app.agent_panel_click_rect.expect("panel painted");

    let moved = |column: u16, row: u16| MouseEvent {
        kind: MouseEventKind::Moved,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    };

    // First move onto the agent row: hover changes → repaint requested.
    let action = app
        .handle_mouse(moved(panel.x + 2, panel.y + 1))
        .expect("hover handled");
    assert_eq!(action, AppAction::Redraw, "entering a row repaints once");
    assert_eq!(app.hovered_agent.as_deref(), Some("agent-alpha"));

    // A second move within the same row: no target change → no repaint.
    let action = app
        .handle_mouse(moved(panel.x + 5, panel.y + 1))
        .expect("hover handled");
    assert_eq!(
        action,
        AppAction::None,
        "staying on the same row must not force a repaint (no motion flood)"
    );

    // Moving off the panel clears the hover and repaints once.
    let action = app
        .handle_mouse(moved(panel.x + 2, panel.y.saturating_sub(3)))
        .expect("hover handled");
    assert_eq!(action, AppAction::Redraw, "leaving a row repaints once");
    assert_eq!(app.hovered_agent, None);
}

#[test]
fn agent_panel_click_rect_clears_when_panel_hidden() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    app.hud_state.agents = vec![running_agent_summary("researcher")];
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw records panel rect");
    let panel = app.agent_panel_click_rect.expect("panel painted");

    // Turn ends → the panel disappears on the next frame; the stale rect must
    // not keep hijacking clicks on ordinary transcript rows.
    app.end_turn();
    app.hud_state.agents.clear();
    app.draw(&mut terminal).expect("redraw clears panel rect");
    assert!(
        app.agent_panel_click_rect.is_none(),
        "hidden panel must clear its click target"
    );

    let action = app
        .handle_mouse(left_click(panel.x + 1, panel.y))
        .expect("click handled");
    assert_ne!(
        action,
        AppAction::OpenWorkflowViewer,
        "a click where the panel used to be must not open the viewer"
    );
}

#[test]
fn spawn_tool_card_plain_click_opens_live_agent_view() {
    let mut app = test_app();
    app.push_block(spawn_tool_call_block(7, "Task"));
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;
    let (col, row) = (transcript.x + 2, transcript.y);

    let down = app.handle_mouse(left_click(col, row)).expect("press handled");
    assert_eq!(down, AppAction::None, "plain press only anchors");
    let up = app
        .handle_mouse(left_release(col, row))
        .expect("release handled");

    assert_eq!(
        up,
        AppAction::OpenWorkflowViewer,
        "a plain click on a Task/Agent/Workflow tool card must open the live agent view"
    );
}

#[test]
fn running_bash_tool_card_plain_click_toggles_live_tail() {
    let mut app = test_app();
    app.push_block(spawn_tool_call_block(7, "bash"));
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;
    let (col, row) = (transcript.x + 2, transcript.y);

    let _ = app.handle_mouse(left_click(col, row)).expect("press handled");
    let up = app
        .handle_mouse(left_release(col, row))
        .expect("release handled");

    assert_eq!(up, AppAction::Redraw);
    assert!(
        app.transcript.is_expanded(0),
        "clicking a running Bash row opens its live tail"
    );
}

#[test]
fn spawn_tool_card_drag_does_not_open_agent_view() {
    let mut app = test_app();
    app.push_block(spawn_tool_call_block(7, "Task"));
    let backend = TestBackend::new(120, 24);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw sets layout regions");
    let transcript = app.regions.expect("layout regions after draw").transcript;
    let (col, row) = (transcript.x + 2, transcript.y);

    let _ = app.handle_mouse(left_click(col, row)).expect("press handled");
    let _ = app
        .handle_mouse(MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: col + 4,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("drag handled");
    let up = app
        .handle_mouse(left_release(col + 4, row))
        .expect("release handled");

    assert_ne!(
        up,
        AppAction::OpenWorkflowViewer,
        "a drag gesture (text selection) must not open the agent view"
    );
}

#[test]
fn agents_viewer_ctrl_o_switches_to_workflow_viewer() {
    let mut app = test_app();
    app.open_agents_viewer();
    assert_eq!(
        app.mode(),
        AppMode::ModalAgents,
        "precondition: agents viewer open"
    );

    let action = app
        .handle_key(press_with(KeyCode::Char('o'), KeyModifiers::CONTROL))
        .expect("ctrl+o handled");

    assert_eq!(
        action,
        AppAction::OpenWorkflowViewer,
        "Ctrl+O in the agents viewer must route to the live workflow viewer"
    );
    assert_eq!(app.mode(), AppMode::Normal, "viewer closed before the switch");
}

#[test]
fn agents_viewer_ctrl_g_toggles_closed() {
    let mut app = test_app();
    app.open_agents_viewer();
    assert_eq!(
        app.mode(),
        AppMode::ModalAgents,
        "precondition: agents viewer open"
    );

    let action = app
        .handle_key(press_with(KeyCode::Char('g'), KeyModifiers::CONTROL))
        .expect("ctrl+g handled");

    assert_eq!(action, AppAction::None);
    assert_eq!(
        app.mode(),
        AppMode::Normal,
        "Ctrl+G must toggle the agents viewer closed, mirroring the key that opened it"
    );
}

/// 일반 pager(긴 도구 출력·help)에서 Ctrl+G 는 pager 를 닫고 에이전트 뷰어로
/// 넘어간다 — 옛 "agents pager 토글" 특수분기의 회귀 방지.
#[test]
fn pager_ctrl_g_switches_to_agents_viewer() {
    let mut app = test_app();
    app.open_pager("long tool output\n".repeat(200));
    assert_eq!(app.mode(), AppMode::Pager, "precondition: generic pager open");

    let action = app
        .handle_key(press_with(KeyCode::Char('g'), KeyModifiers::CONTROL))
        .expect("ctrl+g handled");

    assert_eq!(action, AppAction::None);
    assert_eq!(
        app.mode(),
        AppMode::ModalAgents,
        "Ctrl+G over a generic pager must land on the agents viewer"
    );
}

#[test]
fn workflow_viewer_open_and_refresh_join_the_live_hud_plan_label() {
    use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};

    let mut app = test_app();
    app.sidebar.visible = false;
    app.hud_state.todo_items = vec![TodoChecklistItem {
        step_id: Some("build-viewer".to_string()),
        content: "Build workflow viewer".to_string(),
        status: TodoChecklistStatus::InProgress,
        active_form: "Building workflow viewer".to_string(),
    }];
    app.open_workflow_viewer(WorkflowViewerModal::new(
        workflow_view_for_plan_label_test(),
    ));

    let (width, height) = (140, 30);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw joined Plan label");
    let opened = dump_all(&terminal, width, height);
    assert!(opened.contains("Building workflow viewer"), "{opened}");
    assert!(opened.contains("PLAN") && opened.contains("Executors"), "{opened}");

    app.hud_state.todo_items[0].content = "Verify workflow viewer".to_string();
    app.hud_state.todo_items[0].active_form = "Verifying workflow viewer".to_string();
    app.refresh_workflow_viewer(workflow_view_for_plan_label_test());
    app.draw(&mut terminal).expect("draw refreshed Plan label");
    let refreshed = dump_all(&terminal, width, height);
    assert!(refreshed.contains("Verify workflow viewer"), "{refreshed}");
    assert!(refreshed.contains("Verifying workflow view"), "{refreshed}");
    assert!(!refreshed.contains("Building workflow viewer"), "{refreshed}");
}

#[test]
fn workflow_viewer_ctrl_e_opens_events_while_plain_e_stays_in_composer() {
    let mut app = test_app();
    app.open_workflow_viewer(WorkflowViewerModal::new(
        workflow_view_for_plan_label_test(),
    ));

    let action = app
        .handle_key(press_with(KeyCode::Char('e'), KeyModifiers::CONTROL))
        .expect("ctrl+e handled");
    assert_eq!(action, AppAction::None);
    let (width, height) = (120, 24);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw event inspector");
    assert!(dump_all(&terminal, width, height).contains("event log"));

    app.handle_key(press_with(KeyCode::Char('e'), KeyModifiers::CONTROL))
        .expect("ctrl+e returns to workflow");
    app.handle_key(press(KeyCode::Char('e')))
        .expect("plain e reaches composer");
    assert_eq!(app.input().text(), "e");
    assert!(app.workflow_viewer_open());
}

#[test]
fn workflow_viewer_ctrl_o_toggles_closed() {
    let mut app = test_app();
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));
    assert!(app.workflow_viewer_open(), "precondition: viewer open");

    let action = app
        .handle_key(press_with(KeyCode::Char('o'), KeyModifiers::CONTROL))
        .expect("ctrl+o handled");

    assert_eq!(action, AppAction::None);
    assert!(
        !app.workflow_viewer_open(),
        "Ctrl+O must toggle the live viewer closed, Claude-Code style"
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[test]
fn workflow_viewer_ctrl_g_switches_to_agents_viewer() {
    let mut app = test_app();
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));
    assert!(app.workflow_viewer_open(), "precondition: viewer open");

    let action = app
        .handle_key(press_with(KeyCode::Char('g'), KeyModifiers::CONTROL))
        .expect("ctrl+g handled");

    assert_eq!(action, AppAction::None);
    assert!(!app.workflow_viewer_open(), "viewer closed on switch");
    assert_eq!(
        app.mode(),
        AppMode::ModalAgents,
        "Ctrl+G must land on the agents viewer"
    );
}

#[test]
fn workflow_viewer_survives_one_empty_refresh() {
    let mut app = test_app();
    app.open_workflow_viewer(WorkflowViewerModal::new(WorkflowView::default()));

    // One empty snapshot can be a torn manifest / progress-doc swap mid-write:
    // the freshly opened viewer must not vanish under the user.
    app.apply_workflow_viewer_snapshot(None);
    assert!(
        app.workflow_viewer_open(),
        "a single empty refresh must not close the viewer"
    );

    // A real snapshot resets the tolerance…
    app.apply_workflow_viewer_snapshot(Some(WorkflowView::default()));
    app.apply_workflow_viewer_snapshot(None);
    assert!(
        app.workflow_viewer_open(),
        "the empty-refresh tolerance must reset after a live snapshot"
    );

    // …and only two consecutive empties mean the run is genuinely gone.
    app.apply_workflow_viewer_snapshot(None);
    assert!(
        !app.workflow_viewer_open(),
        "two consecutive empty refreshes close the viewer"
    );
}

#[test]
fn agents_viewer_falls_back_to_hud_snapshot_during_spawn_window() {
    let mut app = test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    app.hud_state.agents = vec![running_agent_summary("researcher")];

    app.open_agents_viewer();

    // Whether or not this machine has manifests on disk, the viewer must show
    // per-agent content — never an empty dead-end while the HUD already knows
    // the fleet. (With manifests present the disk rows win; either way the
    // list is non-empty.)
    let has_rows = app
        .modals
        .agents
        .as_ref()
        .is_some_and(|modal| !modal.is_empty());
    assert!(
        has_rows,
        "spawn-window fallback must surface the HUD fleet in the viewer"
    );
}

/// 오버레이 스택 gap 공유: 핀 패널의 아래 gap 은 스택 최하단(`reserved_below
/// == 0`)일 때만 1행 — 그 위에 쌓일 때(todo/queue 위)는 아래 이웃의 top pad 가
/// 이미 1행 seam 을 제공하므로 0. 각자 gap+pad 를 이중 예약해 인접 오버레이
/// 사이가 2행씩 벌어지던 "빈 띠"의 회귀 방지.
#[test]
fn pinned_agent_panel_gap_below_only_when_bottom_most() {
    let transcript = Rect::new(0, 0, 80, 30);
    let bottom_most = super::render::pinned_agent_panel_geometry(3, transcript, 0)
        .expect("panel fits");
    assert_eq!(
        bottom_most.reserved_height,
        3 + 1 /* gap to input */ + 1, /* top pad */
        "bottom-most panel keeps its one-row gap to the input"
    );

    let stacked = super::render::pinned_agent_panel_geometry(3, transcript, 8)
        .expect("stacked panel fits");
    assert_eq!(
        stacked.reserved_height,
        3 + 1, /* top pad only — the neighbour below owns the seam */
        "stacked panel shares the seam instead of adding a second blank row"
    );
}

/// Concatenate every cell of a `TestBackend` frame into one string so a
/// centered notice can be located regardless of its row/column.
fn buffer_text(terminal: &Terminal<TestBackend>, width: u16, height: u16) -> String {
    (0..height).fold(String::new(), |mut out, y| {
        out.push_str(&buffer_row(terminal, width, y));
        out.push('\n');
        out
    })
}

/// A genuinely unusable fullscreen terminal renders the exact ASCII notice
/// instead of degrading to HUD-only / empty chrome. The message must be present
/// and no input/HUD region may claim height.
#[test]
fn too_small_terminal_shows_exact_ascii_notice() {
    let mut app = test_app();
    assert!(!app.terminal_mode().is_inline(), "default app is fullscreen");

    let (width, height) = (40u16, 10u16);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw too-small frame");

    let text = buffer_text(&terminal, width, height);
    assert!(
        text.contains(crate::tui::layout::TOO_SMALL_MESSAGE),
        "40x10 must render the exact TooSmall notice, got:\n{text}"
    );
    let regions = app.regions.expect("regions recorded even when too small");
    assert_eq!(regions.input.height, 0, "no composer region when too small");
    assert_eq!(regions.hud.height, 0, "no HUD region when too small");
    assert_eq!(regions.sidebar_width, 0, "no sidebar when too small");
}

/// The three required usable sizes render the composer + HUD and never the
/// `TooSmall` notice, and keep a readable transcript.
#[test]
fn required_sizes_render_usable_chrome_not_the_too_small_notice() {
    for (width, height) in [(80u16, 24u16), (120, 40), (200, 60)] {
        let mut app = test_app();
        app.sidebar.visible = true;
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        app.draw(&mut terminal).expect("draw usable frame");

        let text = buffer_text(&terminal, width, height);
        assert!(
            !text.contains(crate::tui::layout::TOO_SMALL_MESSAGE),
            "{width}x{height} is usable and must not show the TooSmall notice"
        );
        let regions = app.regions.expect("regions after draw");
        assert!(regions.input.height > 0, "{width}x{height}: composer visible");
        assert!(regions.hud.height > 0, "{width}x{height}: HUD visible");
        assert!(
            regions.transcript.height >= crate::tui::layout::MIN_READABLE_TRANSCRIPT_ROWS,
            "{width}x{height}: transcript below readable minimum"
        );
        // The wide sizes keep the optional sidebar; 80x24 (Compact) suppresses
        // it, matching the unified LayoutPlan policy.
        if width >= 120 {
            assert!(regions.sidebar_width > 0, "{width}x{height}: wide sidebar preserved");
        } else {
            assert_eq!(regions.sidebar_width, 0, "80x24 is not wide ⇒ no sidebar");
        }
    }
}

/// Resize/reflow: 200x60 → 120x40 → 80x24 → 40x10 → 120x40 must preserve input
/// text and transcript scroll state semantically (geometry reflows; state does
/// not reset), and the 40x10 step shows the `TooSmall` notice while later steps
/// recover to usable chrome.
#[test]
fn resize_sequence_preserves_input_and_scroll_state() {
    let mut app = test_app();
    app.sidebar.visible = true;

    // Give the transcript enough content that scrolling up is meaningful.
    for i in 0..40 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("transcript line {i}"),
            done: true,
        });
    }
    // Type some composer text through the real key path.
    for ch in "keep me".chars() {
        let code = if ch == ' ' {
            KeyCode::Char(' ')
        } else {
            KeyCode::Char(ch)
        };
        let _ = app.handle_key(press(code));
    }
    let typed = app.input.text();
    assert_eq!(typed, "keep me", "precondition: composer holds the typed text");

    // Draw once at the largest size, then scroll up so we have an explicit,
    // non-tail scroll offset to preserve across the reflow.
    let draw_at = |app: &mut App, w: u16, h: u16| -> Terminal<TestBackend> {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        app.draw(&mut terminal).expect("draw frame");
        terminal
    };

    let _ = draw_at(&mut app, 200, 60);
    app.transcript.scroll_up(3);
    assert!(
        app.transcript.scroll() > 0,
        "precondition: an explicit scroll offset exists"
    );

    let sequence = [(200u16, 60u16), (120, 40), (80, 24), (40, 10), (120, 40)];
    for (idx, (w, h)) in sequence.into_iter().enumerate() {
        let terminal = draw_at(&mut app, w, h);
        let text = buffer_text(&terminal, w, h);

        // State is preserved across every reflow: the composer text never
        // resets, regardless of whether the frame was usable or TooSmall.
        assert_eq!(
            app.input.text(),
            "keep me",
            "step {idx} ({w}x{h}): composer text must survive the resize"
        );

        if (w, h) == (40, 10) {
            assert!(
                text.contains(crate::tui::layout::TOO_SMALL_MESSAGE),
                "step {idx}: 40x10 must show the TooSmall notice"
            );
        } else {
            assert!(
                !text.contains(crate::tui::layout::TOO_SMALL_MESSAGE),
                "step {idx} ({w}x{h}): usable size must not show the TooSmall notice"
            );
            let regions = app.regions.expect("regions after usable draw");
            assert!(
                regions.transcript.height >= crate::tui::layout::MIN_READABLE_TRANSCRIPT_ROWS,
                "step {idx} ({w}x{h}): transcript kept its readable minimum"
            );
        }
    }

    // After the sequence the scroll offset is still a live, explicit value
    // (not reset to tail by any reflow) — the state machine survived the
    // TooSmall frame in the middle. Asserted semantically (offset preserved,
    // not snapped back to the tail) rather than bit-exact, since a smaller
    // viewport may legitimately clamp the resolved offset without resetting it.
    assert!(
        app.transcript.scroll() > 0,
        "explicit transcript scroll offset must survive the resize sequence, not reset to tail"
    );
}
