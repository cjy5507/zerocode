//! Integration tests for Lane L2 (`tui/` foundation).
//!
//! Living-standard naming (L1): tests live at
//! `crates/zo-cli/tests/tui_<scope>.rs` with the convention
//! `<area>_<scenario>`. Every test exercises the library target via
//! `ratatui::backend::TestBackend` — no real terminal, no blocking on
//! `crossterm::event::EventStream`.

use std::path::PathBuf;

use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MouseEvent, MouseEventKind,
};
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::Terminal;
use runtime::message_stream::{ActiveModel, BlockId, RenderBlock, SystemLevel, UserQuestionPrompt};
use runtime::PermissionMode as RuntimePermissionMode;
use zo_cli::tui::{
    app::{AgentCommand, App, AppAction, AppMode, ClipboardCopyTarget},
    layout::{LayoutRegions, HUD_ROWS, INPUT_MAX_ROWS, INPUT_MIN_ROWS},
    modals::{ModelPickerEntry, ToolToggleRow},
    theme::{Breakpoint, Theme},
    ChangedFile, FileStatus, HudState, LspStatusItem, PermissionMode as HudPermissionMode,
    SecurityPosture, TodoChecklistItem, TodoChecklistStatus,
};
use tokio::sync::mpsc;
use tokio::sync::oneshot;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tokens_path() -> PathBuf {
    // Vendored fixture — keeps the test hermetic regardless of whether
    // the repo's untracked `.zo/design/tokens.json` is materialised
    // in the current worktree.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokens.json")
}

fn sample_system_block() -> RenderBlock {
    RenderBlock::System {
        id: BlockId(42),
        level: SystemLevel::Info,
        text: "hello from test".to_string(),
    }
}

fn user_question_block(
    question: &str,
    options: Vec<String>,
) -> (RenderBlock, oneshot::Receiver<Vec<String>>) {
    let (responder, response) = oneshot::channel();
    (
        RenderBlock::UserQuestionPrompt(UserQuestionPrompt {
            id: BlockId(99),
            question: question.to_string(),
            header: None,
            options: options
                .into_iter()
                .map(runtime::message_stream::QuestionOption::plain)
                .collect(),
            multi_select: false,
            responder,
        }),
        response,
    )
}

fn new_app(theme: Theme) -> (App, mpsc::Sender<RenderBlock>, mpsc::Receiver<AgentCommand>) {
    let (block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
    let (cmd_tx, cmd_rx) = mpsc::channel::<AgentCommand>(16);
    (App::new(theme, block_rx, cmd_tx), block_tx, cmd_rx)
}

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn press_mods(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn dump_terminal(terminal: &Terminal<TestBackend>) -> String {
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect()
}

fn sample_hud_state() -> HudState {
    HudState {
        session_identity: None,
        model: ActiveModel {
            provider: "claude",
            alias: "opus".to_string(),
            display_name: "Claude Opus".to_string(),
            context_limit: 200_000,
        },
        turn_fallback_model: None,
        quota_fallback_model: None,
        ctx_used: 40_146,
        ctx_limit: 200_000,
        ctx_new_input: 0,
        ctx_cached: 0,
        compact_threshold: 0,
        cost_usd: 0.37,
        cost_approx: false,
        cwd: PathBuf::from("/Users/joe/2026/zo"),
        git_branch: Some("main".to_string()),
        perm_mode: HudPermissionMode::Workspace,
        security_posture: SecurityPosture::SandboxActive,
        effort: None,
        architect_impl: None,
        mcp_servers: vec!["almanac".to_string(), "context7".to_string()],
        bash_count: 1,
        read_count: 2,
        edit_count: 3,
        changed_files: 0,
        todo_summary: Some("4 todos active".to_string()),
        todo_items: vec![TodoChecklistItem {
            step_id: None,
            content: "Render real Todo checklist".to_string(),
            status: TodoChecklistStatus::InProgress,
            active_form: "Rendering real Todo checklist".to_string(),
        }],
        automation_lines: Vec::new(),
        lsp_servers: vec![LspStatusItem {
            language: "rust".to_string(),
            status: "connected".to_string(),
        }],
        running_agents: 0,
        agents: Vec::new(),
        workflow: None,
        last_tool: None,
        rate_limit: None,
        provider_quotas: Vec::new(),
        auth_origin: None,
        status_line: None,
        team_inbox_unread: 0,
        stale_binary: None,
        background_tasks: 0,
        scheduled_wake: None,
    }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

#[tokio::test]
async fn theme_loads_full_palette_from_tokens_json() {
    let theme = Theme::load(&tokens_path()).expect("theme loads");
    // Spacing should come straight from the tokens file — not the
    // `fallback()` defaults.
    assert_eq!(theme.spacing.row_gap, 1);
    assert_eq!(theme.spacing.card_padding_x, 1);
    // Breakpoint thresholds come from the tokens file.
    assert_eq!(theme.narrow_max, 59);
    assert_eq!(theme.wide_min, 100);
    assert!(!theme.no_color || std::env::var_os("NO_COLOR").is_some());
}

#[tokio::test]
async fn theme_no_color_variant_has_neutral_palette() {
    let theme = Theme::no_color();
    assert!(theme.no_color);
    // The typography styles still exist; only colours are neutralised.
    assert_eq!(theme.palette.accent, ratatui::style::Color::Reset);
    assert_eq!(theme.palette.fg, ratatui::style::Color::Reset);
    // Default breakpoint thresholds apply when no file was read.
    assert_eq!(theme.narrow_max, Theme::DEFAULT_NARROW_MAX);
    assert_eq!(theme.wide_min, Theme::DEFAULT_WIDE_MIN);
}

#[tokio::test]
async fn theme_for_width_classifies_breakpoints() {
    let theme = Theme::no_color();
    assert_eq!(theme.for_width(40), Breakpoint::Narrow);
    assert_eq!(theme.for_width(80), Breakpoint::Compact);
    assert_eq!(theme.for_width(120), Breakpoint::Wide);
    // Boundary conditions.
    assert_eq!(
        theme.for_width(Theme::DEFAULT_NARROW_MAX),
        Breakpoint::Narrow
    );
    assert_eq!(theme.for_width(Theme::DEFAULT_WIDE_MIN), Breakpoint::Wide);
}

#[tokio::test]
async fn theme_builtin_opencode_is_available() {
    let theme = Theme::builtin("opencode").expect("opencode theme should exist");
    assert_eq!(theme.name, "opencode");
    assert!(!theme.no_color);
    assert!(Theme::builtin_names().contains(&"opencode"));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn layout_tiles_small_medium_and_wide_terminals() {
    for (w, h) in [(60_u16, 20_u16), (100, 40), (200, 60)] {
        let area = Rect::new(0, 0, w, h);
        let regions = LayoutRegions::compute(area, 3).expect("layout computes");

        assert!(
            regions.tiles(area),
            "regions must tile exactly at {w}x{h}: {regions:?}"
        );
        assert_eq!(regions.hud.height, HUD_ROWS);
        assert!(regions.input.height >= INPUT_MIN_ROWS);
        assert!(regions.input.height <= INPUT_MAX_ROWS);
        assert!(regions.transcript.height >= 1);
    }
}

#[tokio::test]
async fn layout_clamps_input_rows_into_allowed_range() {
    let area = Rect::new(0, 0, 100, 40);
    let tiny = LayoutRegions::compute(area, 1).unwrap();
    let huge = LayoutRegions::compute(area, 99).unwrap();
    assert_eq!(tiny.input.height, INPUT_MIN_ROWS);
    assert_eq!(huge.input.height, INPUT_MAX_ROWS);
}

#[tokio::test]
async fn layout_sidebar_visible_uses_right_panel() {
    let area = Rect::new(0, 0, 120, 40);
    let regions = LayoutRegions::compute_with_sidebar(area, 3, HUD_ROWS, true).expect("layout");

    assert!(regions.tiles(area));
    assert!(regions.sidebar_width > 0);
    assert_eq!(regions.transcript.x, area.x);
    assert_eq!(
        regions.sidebar.x,
        regions.transcript.x + regions.transcript.width
    );
    assert_eq!(regions.sidebar.height, regions.transcript.height);
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn app_starts_in_normal_mode_and_can_draw() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    assert_eq!(app.mode(), AppMode::Normal);

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");

    // No blocks have arrived yet.
    assert_eq!(app.blocks_drained(), 0);
}

#[tokio::test]
async fn app_draws_codex_like_right_sidebar_metadata() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.set_hud_state(sample_hud_state());
    app.set_changed_files(
        vec![ChangedFile {
            path: "crates/runtime/src/conversation.rs".to_string(),
            status: FileStatus::Modified,
            adds: 0,
            rems: 0,
        }],
        1,
    );

    let backend = TestBackend::new(140, 36);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");

    let dumped = dump_terminal(&terminal);
    for expected in [
        "write",
        "zo",
        "session",
        "sources 2",
        "almanac",
        "context7",
        "lsp 1",
        "rust",
        "connected",
        "todo 1",
        "[-]",
        "Rendering real",
        "changes 1",
        "conversation.rs",
    ] {
        assert!(dumped.contains(expected), "missing {expected}: {dumped}");
    }
}

#[tokio::test]
async fn app_when_input_is_disabled_renders_input_widget_for_queuing() {
    // With the queued-message feature, the input widget is always
    // rendered — even when input_enabled is false — so the user can
    // type ahead while a turn is in progress.
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");

    let dumped = dump_terminal(&terminal);
    // The editable input placeholder should be visible even when
    // input is disabled, since the user can now type queued messages.
    assert!(
        dumped.contains("Message Zo"),
        "input widget should render when disabled for queue-ahead: {dumped}"
    );
}

#[tokio::test]
async fn app_ctrl_v_requests_clipboard_even_when_input_is_disabled() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    assert!(!app.input_enabled());
    assert_eq!(
        app.handle_key(press_mods(KeyCode::Char('v'), KeyModifiers::CONTROL))
            .expect("ctrl-v"),
        AppAction::ClipboardPaste
    );
}

#[tokio::test]
async fn app_ctrl_y_requests_last_message_copy_without_stealing_ctrl_c() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    assert_eq!(
        app.handle_key(press_mods(KeyCode::Char('y'), KeyModifiers::CONTROL))
            .expect("ctrl-y"),
        AppAction::ClipboardCopy(ClipboardCopyTarget::Last)
    );

    assert_ne!(
        app.handle_key(press_mods(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .expect("ctrl-c"),
        AppAction::ClipboardCopy(ClipboardCopyTarget::Last)
    );
}

#[tokio::test]
async fn app_down_scrolls_transcript_when_queue_input_is_empty() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..12 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    assert!(!app.input_enabled());
    app.transcript_mut().scroll_to_top();
    assert_eq!(app.transcript_mut().scroll(), 0);
    assert_eq!(
        app.handle_key(press(KeyCode::Down)).expect("down"),
        AppAction::None
    );
    assert!(app.transcript_mut().scroll() > 0);
}

#[tokio::test]
async fn app_ctrl_u_scrolls_transcript_when_input_is_empty() {
    // [C6] With an empty input buffer, Ctrl+U keeps its transcript
    // half-page-up binding rather than reaching the input widget.
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..30 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");
    app.transcript_mut().scroll_to_bottom();
    assert_eq!(app.transcript_mut().scroll(), u16::MAX);
    assert!(app.input_mut().is_empty());

    assert_eq!(
        app.handle_key(press_mods(KeyCode::Char('u'), KeyModifiers::CONTROL))
            .expect("ctrl-u"),
        AppAction::None
    );
    // Routed to the scroll path: the tail sentinel was normalized and the
    // view moved up, and the input buffer was never touched.
    assert_ne!(app.transcript_mut().scroll(), u16::MAX);
    assert!(app.input_mut().is_empty());
}

#[tokio::test]
async fn app_ctrl_a_moves_cursor_to_line_start_when_input_has_text() {
    // [C6] With a non-empty input buffer, Ctrl+A falls through to the
    // input widget's readline binding (move to line start) instead of
    // toggling the agents sidebar.
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();
    app.input_mut().insert_text("hello world");
    assert_eq!(app.input_mut().cursor(), (0, 11));

    assert_eq!(
        app.handle_key(press_mods(KeyCode::Char('a'), KeyModifiers::CONTROL))
            .expect("ctrl-a"),
        AppAction::None
    );
    assert_eq!(app.input_mut().cursor(), (0, 0));
    assert_eq!(app.input_mut().text(), "hello world");
}

#[tokio::test]
async fn app_ctrl_u_kills_line_when_input_has_text() {
    // [C6] With a non-empty input buffer, Ctrl+U falls through to the
    // input widget and kills the whole line rather than scrolling.
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();
    app.input_mut().insert_text("delete me");
    assert!(!app.input_mut().is_empty());

    assert_eq!(
        app.handle_key(press_mods(KeyCode::Char('u'), KeyModifiers::CONTROL))
            .expect("ctrl-u"),
        AppAction::None
    );
    assert!(app.input_mut().is_empty());
}

#[tokio::test]
async fn app_down_from_tail_sentinel_normalizes_scroll_state() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..30 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");

    // New output while following the tail uses the internal sentinel.
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(100),
        text: "new tail".to_string(),
        done: true,
    });
    assert_eq!(app.transcript_mut().scroll(), u16::MAX);

    assert_eq!(
        app.handle_key(press(KeyCode::Down)).expect("down"),
        AppAction::None
    );
    assert_ne!(
        app.transcript_mut().scroll(),
        u16::MAX,
        "user scroll should normalize the tail sentinel before applying Down"
    );
}

#[tokio::test]
async fn app_mode_transitions_between_modals_and_back() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    app.enter_mode(AppMode::ModalModel);
    assert_eq!(app.mode(), AppMode::ModalModel);
    assert!(app.mode().is_modal());

    app.enter_mode(AppMode::ModalPermissions);
    assert_eq!(app.mode(), AppMode::ModalPermissions);

    app.enter_mode(AppMode::ModalChoice);
    assert_eq!(app.mode(), AppMode::ModalChoice);

    app.exit_modal();
    assert_eq!(app.mode(), AppMode::Normal);
    assert!(!app.mode().is_modal());
}

#[tokio::test]
async fn app_drains_render_blocks_without_panicking() {
    let theme = Theme::no_color();
    let (mut app, block_tx, _cmd_rx) = new_app(theme);

    // Feed three canned system blocks (L5 will turn these into real
    // widgets — L2 just has to not panic on any variant).
    for _ in 0..3 {
        block_tx
            .send(sample_system_block())
            .await
            .expect("send block");
    }

    let drained = app.drain_ready_blocks();
    assert_eq!(drained, 3);
    assert_eq!(app.blocks_drained(), 3);
}

#[tokio::test]
async fn app_drain_with_first_respects_frame_cap() {
    // Assert against the real per-frame cap, not a hardcoded literal: a fixed
    // number silently rots whenever `MAX_DRAIN_PER_TICK` is retuned.
    let cap = App::max_drain_per_tick();
    let theme = Theme::no_color();
    // The shared `new_app` channel is bounded at 16, too small to hold a full
    // cap-sized burst; build a dedicated wider channel so queuing `cap` blocks
    // never backpressures `send().await` against the unpolled receiver.
    let (block_tx, block_rx) = mpsc::channel::<RenderBlock>(cap + 1);
    let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
    let mut app = App::new(theme, block_rx, cmd_tx);

    // Queue exactly `cap` blocks; the drip's `first` block counts toward the
    // same per-frame cap, so the drain fills to `cap` and the surplus block
    // stays queued for the next frame.
    for _ in 0..cap {
        block_tx
            .send(sample_system_block())
            .await
            .expect("send block");
    }

    let drained = app.drain_ready_blocks_with_first(sample_system_block());
    assert_eq!(drained, cap, "first block counts toward the frame cap");
    assert_eq!(app.blocks_drained(), cap);

    let next_frame = app.drain_ready_blocks();
    assert_eq!(next_frame, 1, "overflow stays queued for the next frame");
    assert_eq!(app.blocks_drained(), cap + 1);
}

#[tokio::test]
async fn app_ctrl_c_double_tap_exits_and_sends_quit() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, mut cmd_rx) = new_app(theme);

    let ctrl_c = KeyEvent {
        code: KeyCode::Char('c'),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    // First Ctrl-C → cancel, not exit.
    let first = app.handle_key(ctrl_c).expect("first ctrl-c ok");
    assert_eq!(
        first,
        AppAction::None,
        "first Ctrl-C must not exit the loop"
    );
    let cmd1 = cmd_rx.recv().await.expect("first command arrives");
    assert_eq!(cmd1, AgentCommand::CancelTurn);

    // Second Ctrl-C within the double-tap window → exit + Quit.
    let second = app.handle_key(ctrl_c).expect("second ctrl-c ok");
    assert_eq!(second, AppAction::Quit, "second Ctrl-C must exit the loop");
    let cmd2 = cmd_rx.recv().await.expect("quit command arrives");
    assert_eq!(cmd2, AgentCommand::Quit);
}

#[tokio::test]
async fn app_when_input_enabled_enter_submits_buffer() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();

    let key_h = KeyEvent {
        code: KeyCode::Char('h'),
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };
    let key_i = KeyEvent {
        code: KeyCode::Char('i'),
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };
    let enter = KeyEvent {
        code: KeyCode::Enter,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };

    assert_eq!(app.handle_key(key_h).expect("h"), AppAction::None);
    assert_eq!(app.handle_key(key_i).expect("i"), AppAction::None);
    assert_eq!(
        app.handle_key(enter).expect("submit"),
        AppAction::Submit("hi".to_string())
    );
}

#[tokio::test]
async fn app_mouse_wheel_scrolls_transcript_inside_transcript_region() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..12 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 2,
        row: 1,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse scroll");
    assert!(app.transcript_mut().scroll() > 0);
}

#[tokio::test]
async fn app_mouse_wheel_over_input_scrolls_transcript() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..12 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");
    app.transcript_mut().scroll_to_top();
    // The wheel must scroll the chat even when the pointer rests over the
    // input box (idle or while typing) — only the slash-hint popup consumes it.
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 2,
        row: 21,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse scroll over input");
    assert!(
        app.transcript_mut().scroll() > 0,
        "scrolling over the input box should still move the transcript"
    );
}

#[tokio::test]
async fn app_mouse_wheel_over_hud_or_outside_scrolls_transcript() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..12 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");
    app.transcript_mut().scroll_to_top();

    for (column, row) in [(2, 23), (100, 100)] {
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
        .expect("mouse scroll outside transcript");
    }

    assert_eq!(
        app.transcript_mut().scroll(),
        6,
        "normal-mode wheel events should move the transcript regardless of pointer row"
    );
}

#[tokio::test]
async fn app_mouse_wheel_over_sidebar_scrolls_transcript_not_sidebar() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.set_hud_state(sample_hud_state());
    app.set_changed_files(
        (0..30)
            .map(|idx| ChangedFile {
                path: format!("crates/runtime/src/file_{idx}.rs"),
                status: FileStatus::Modified,
                adds: 0,
                rems: 0,
            })
            .collect(),
        30,
    );

    for idx in 0..12 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(140, 36);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");
    app.transcript_mut().scroll_to_top();

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 130,
        row: 2,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse scroll over sidebar");

    assert!(
        app.transcript_mut().scroll() > 0,
        "scrolling over the sidebar should move the transcript"
    );
    assert_eq!(
        app.sidebar().scroll,
        0,
        "normal-mode wheel events should not move the sidebar"
    );
}

#[tokio::test]
async fn app_manual_scroll_up_disables_follow_output_until_bottom_is_restored() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..20 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: 2,
        row: 1,
        modifiers: KeyModifiers::NONE,
    })
    .expect("mouse scroll up");
    let scrolled_up = app.transcript_mut().scroll();

    app.push_block(RenderBlock::TextDelta {
        id: BlockId(100),
        text: "new tail".to_string(),
        done: true,
    });
    assert_eq!(
        app.transcript_mut().scroll(),
        scrolled_up,
        "new output should not yank the transcript back to the tail while manually scrolled up"
    );

    app.handle_key(press(KeyCode::End)).expect("end");
    app.push_block(RenderBlock::TextDelta {
        id: BlockId(101),
        text: "latest tail".to_string(),
        done: true,
    });
    assert!(
        app.transcript_mut().scroll() >= scrolled_up,
        "restoring tail follow should allow new output to keep the viewport at the bottom"
    );
}

#[tokio::test]
async fn app_shift_up_scrolls_without_requiring_mouse_support() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    for idx in 0..20 {
        app.push_block(RenderBlock::TextDelta {
            id: BlockId(idx),
            text: format!("line {idx}"),
            done: true,
        });
    }

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    app.draw(&mut terminal).expect("draw without error");
    app.handle_key(KeyEvent {
        code: KeyCode::Up,
        modifiers: KeyModifiers::SHIFT,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
    .expect("shift+up");
    assert!(app.transcript_mut().scroll() > 0);
}

#[tokio::test]
async fn app_model_modal_returns_selected_model_action() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.open_model_modal(vec![
        ModelPickerEntry {
            provider: "anthropic".to_string(),
            model: ActiveModel {
                provider: "anthropic",
                alias: "opus".to_string(),
                display_name: "Claude Opus".to_string(),
                context_limit: 200_000,
            },
        },
        ModelPickerEntry {
            provider: "anthropic".to_string(),
            model: ActiveModel {
                provider: "anthropic",
                alias: "sonnet".to_string(),
                display_name: "Claude Sonnet".to_string(),
                context_limit: 200_000,
            },
        },
    ]);

    assert_eq!(app.mode(), AppMode::ModalModel);
    assert_eq!(
        app.handle_key(press(KeyCode::Down)).expect("down"),
        AppAction::None
    );
    match app.handle_key(press(KeyCode::Enter)).expect("enter") {
        AppAction::SelectModel(model) => assert_eq!(model.alias, "sonnet"),
        other => panic!("expected SelectModel, got {other:?}"),
    }
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn app_permission_modal_returns_selected_permission_action() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.open_permission_picker_modal(RuntimePermissionMode::ReadOnly);

    assert_eq!(app.mode(), AppMode::ModalPermissions);
    assert_eq!(
        app.handle_key(press(KeyCode::Down)).expect("down"),
        AppAction::None
    );
    match app.handle_key(press(KeyCode::Enter)).expect("enter") {
        AppAction::SelectPermission(mode) => {
            assert_ne!(mode, RuntimePermissionMode::ReadOnly);
        }
        other => panic!("expected SelectPermission, got {other:?}"),
    }
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn app_tool_toggle_modal_returns_toggle_action_without_closing() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.open_tool_toggle_modal(vec![ToolToggleRow {
        name: "WebSearch".to_string(),
        description: Some("search the web".to_string()),
        source: "builtin".to_string(),
        enabled: true,
    }]);

    assert_eq!(app.mode(), AppMode::ModalTools);
    match app.handle_key(press(KeyCode::Enter)).expect("enter") {
        AppAction::ToggleTool { name, enabled } => {
            assert_eq!(name, "WebSearch");
            assert!(!enabled);
        }
        other => panic!("expected ToggleTool, got {other:?}"),
    }
    assert_eq!(app.mode(), AppMode::ModalTools);
    assert_eq!(
        app.handle_key(press(KeyCode::Esc)).expect("esc"),
        AppAction::None
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn app_user_question_prompt_returns_selected_option() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    let (block, response) =
        user_question_block("Pick one", vec!["alpha".to_string(), "beta".to_string()]);

    app.push_block(block);

    assert_eq!(app.mode(), AppMode::ModalQuestion);
    assert_eq!(
        app.handle_key(press(KeyCode::Down)).expect("down"),
        AppAction::None
    );
    assert_eq!(
        app.handle_key(press(KeyCode::Enter)).expect("enter"),
        AppAction::None
    );
    assert_eq!(
        response.await.expect("question response"),
        vec!["beta".to_string()]
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn app_user_question_prompt_returns_freeform_answer() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    let (block, response) = user_question_block("Name?", Vec::new());

    app.push_block(block);

    assert_eq!(app.mode(), AppMode::ModalQuestion);
    assert_eq!(
        app.handle_key(press(KeyCode::Char('o'))).expect("o"),
        AppAction::None
    );
    assert_eq!(
        app.handle_key(press(KeyCode::Char('k'))).expect("k"),
        AppAction::None
    );
    assert_eq!(
        app.handle_key(press(KeyCode::Enter)).expect("enter"),
        AppAction::None
    );
    assert_eq!(
        response.await.expect("question response"),
        vec!["ok".to_string()]
    );
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn app_shift_tab_cycles_permission_via_session_action() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    #[allow(clippy::items_after_statements)]
    fn hud_with_mode(mode: HudPermissionMode) -> HudState {
        HudState {
            session_identity: None,
            model: ActiveModel {
                provider: "claude",
                alias: "opus".to_string(),
                display_name: "Claude Opus".to_string(),
                context_limit: 200_000,
            },
            turn_fallback_model: None,
            quota_fallback_model: None,
            ctx_used: 0,
            ctx_limit: 200_000,
            ctx_new_input: 0,
            ctx_cached: 0,
            compact_threshold: 0,
            cost_usd: 0.0,
            cost_approx: false,
            cwd: PathBuf::from("."),
            git_branch: None,
            perm_mode: mode,
            security_posture: SecurityPosture::SandboxActive,
            effort: None,
            architect_impl: None,
            mcp_servers: Vec::new(),
            bash_count: 0,
            read_count: 0,
            edit_count: 0,
            changed_files: 0,
            todo_summary: None,
            todo_items: Vec::new(),
            automation_lines: Vec::new(),
            lsp_servers: Vec::new(),
            running_agents: 0,
            agents: Vec::new(),
            workflow: None,
            last_tool: None,
            rate_limit: None,
            provider_quotas: Vec::new(),
            auth_origin: None,
            status_line: None,
            team_inbox_unread: 0,
            stale_binary: None,
            background_tasks: 0,
            scheduled_wake: None,
        }
    }

    let back_tab = KeyEvent {
        code: KeyCode::BackTab,
        modifiers: KeyModifiers::SHIFT,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    };

    // The HUD is synced from the session's real PermissionMode by the
    // outer loop; Shift+Tab must emit SelectPermission so the session —
    // not local UI state — is the single source of truth.
    // ReadOnly → Plan: Plan is the read-only-backed planning stop, so the key
    // emits the read-only runtime mode (the host loop resolves the Plan badge
    // from `plan_mode_active`).
    app.set_hud_state(hud_with_mode(HudPermissionMode::ReadOnly));
    match app.handle_key(back_tab).expect("backtab") {
        AppAction::SelectPermission(mode) => {
            assert_eq!(mode, RuntimePermissionMode::ReadOnly);
        }
        other => panic!("expected SelectPermission, got {other:?}"),
    }

    // Plan → Workspace.
    app.set_hud_state(hud_with_mode(HudPermissionMode::Plan));
    match app.handle_key(back_tab).expect("backtab") {
        AppAction::SelectPermission(mode) => {
            assert_eq!(mode, RuntimePermissionMode::WorkspaceWrite);
        }
        other => panic!("expected SelectPermission, got {other:?}"),
    }

    app.set_hud_state(hud_with_mode(HudPermissionMode::Workspace));
    match app.handle_key(back_tab).expect("backtab") {
        AppAction::SelectPermission(mode) => {
            assert_eq!(mode, RuntimePermissionMode::DangerFullAccess);
        }
        other => panic!("expected SelectPermission, got {other:?}"),
    }

    app.set_hud_state(hud_with_mode(HudPermissionMode::All));
    match app.handle_key(back_tab).expect("backtab") {
        AppAction::SelectPermission(mode) => {
            assert_eq!(mode, RuntimePermissionMode::ReadOnly);
        }
        other => panic!("expected SelectPermission, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Search mode (Enhancement 3)
// ---------------------------------------------------------------------------

fn ctrl(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

#[tokio::test]
async fn ctrl_f_enters_search_mode() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();

    let action = app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");
    assert_eq!(action, AppAction::None);
    assert_eq!(app.mode(), AppMode::Search);
}

#[tokio::test]
async fn search_mode_accumulates_query_and_exits_on_esc() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();
    app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");

    app.handle_key(press(KeyCode::Char('h'))).expect("h");
    app.handle_key(press(KeyCode::Char('i'))).expect("i");
    assert_eq!(app.search_query(), "hi");

    app.handle_key(press(KeyCode::Esc)).expect("esc");
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn search_mode_backspace_removes_last_char() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();
    app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");

    app.handle_key(press(KeyCode::Char('a'))).expect("a");
    app.handle_key(press(KeyCode::Char('b'))).expect("b");
    app.handle_key(press(KeyCode::Backspace)).expect("bs");
    assert_eq!(app.search_query(), "a");
}

#[tokio::test]
async fn search_is_incremental_and_enter_keeps_mode_open() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();

    app.push_block(RenderBlock::System {
        id: BlockId(1),
        level: SystemLevel::Info,
        text: "find the target here".to_string(),
    });

    app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");
    app.handle_key(press(KeyCode::Char('t'))).expect("t");
    app.handle_key(press(KeyCode::Char('a'))).expect("a");
    app.handle_key(press(KeyCode::Char('r'))).expect("r");

    // Incremental search found the match while typing.
    assert_eq!(app.search_match_count(), 1);
    assert_eq!(app.search_active_position(), 1);

    // Enter cycles (wraps to the same single match) and stays in search.
    app.handle_key(press(KeyCode::Enter)).expect("enter");
    assert_eq!(app.mode(), AppMode::Search);
    assert_eq!(app.search_active_position(), 1);

    // Esc returns to Normal.
    app.handle_key(press(KeyCode::Esc)).expect("esc");
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn search_cycles_across_multiple_matches() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();

    for i in 0..3u64 {
        app.push_block(RenderBlock::System {
            id: BlockId(i + 1),
            level: SystemLevel::Info,
            text: format!("hit number {i}"),
        });
    }

    app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");
    for ch in "hit".chars() {
        app.handle_key(press(KeyCode::Char(ch))).expect("char");
    }
    assert_eq!(app.search_match_count(), 3);
    assert_eq!(app.search_active_position(), 1);

    app.handle_key(press(KeyCode::Enter)).expect("next"); // -> 2
    assert_eq!(app.search_active_position(), 2);
    app.handle_key(press(KeyCode::Down)).expect("next"); // -> 3
    assert_eq!(app.search_active_position(), 3);
    app.handle_key(press(KeyCode::Enter)).expect("wrap"); // -> 1 (wrap)
    assert_eq!(app.search_active_position(), 1);
    app.handle_key(press(KeyCode::Up)).expect("prev"); // -> 3 (wrap back)
    assert_eq!(app.search_active_position(), 3);
}

#[tokio::test]
async fn search_reports_no_matches_and_nav_is_noop() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();

    app.push_block(RenderBlock::System {
        id: BlockId(1),
        level: SystemLevel::Info,
        text: "alpha".to_string(),
    });

    app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");
    for ch in "zzz".chars() {
        app.handle_key(press(KeyCode::Char(ch))).expect("char");
    }
    assert_eq!(app.search_match_count(), 0);
    assert_eq!(app.search_active_position(), 0);

    // Enter/Up on an empty match set must not panic or move.
    app.handle_key(press(KeyCode::Enter)).expect("enter");
    app.handle_key(press(KeyCode::Up)).expect("up");
    assert_eq!(app.search_active_position(), 0);
}

// ---------------------------------------------------------------------------
// Pager mode (Enhancement 4)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pager_opens_for_long_content() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let long_content = (0..100)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let opened = app.show_in_pager(long_content, 24);
    assert!(opened, "pager should open for content exceeding viewport");
    assert_eq!(app.mode(), AppMode::Pager);
}

#[tokio::test]
async fn pager_does_not_open_for_short_content() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let short_content = "line 1\nline 2\nline 3".to_string();
    let opened = app.show_in_pager(short_content, 24);
    assert!(!opened, "pager should not open for short content");
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn pager_exits_on_q() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let long_content = (0..100)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.show_in_pager(long_content, 24);
    assert_eq!(app.mode(), AppMode::Pager);

    app.handle_key(press(KeyCode::Char('q'))).expect("q");
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn pager_exits_on_esc() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let long_content = (0..100)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.show_in_pager(long_content, 24);

    app.handle_key(press(KeyCode::Esc)).expect("esc");
    assert_eq!(app.mode(), AppMode::Normal);
}

#[tokio::test]
async fn pager_scrolls_with_arrow_keys() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let long_content = (0..100)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.show_in_pager(long_content, 24);

    app.handle_key(press(KeyCode::Down)).expect("down");
    app.handle_key(press(KeyCode::Down)).expect("down");
    // Just assert it doesn't panic and stays in pager mode.
    assert_eq!(app.mode(), AppMode::Pager);

    app.handle_key(press(KeyCode::Up)).expect("up");
    assert_eq!(app.mode(), AppMode::Pager);
}

#[tokio::test]
async fn search_bar_renders_in_transcript_area() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);
    app.enable_input();
    app.handle_key(ctrl(KeyCode::Char('f'))).expect("ctrl+f");
    app.handle_key(press(KeyCode::Char('x'))).expect("x");

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("test backend");
    app.draw(&mut terminal).expect("draw");

    let rendered = dump_terminal(&terminal);
    assert!(
        rendered.contains("Search:"),
        "search bar should be visible: {rendered}"
    );
}

#[tokio::test]
async fn pager_renders_header_and_footer() {
    let theme = Theme::no_color();
    let (mut app, _block_tx, _cmd_rx) = new_app(theme);

    let long_content = (0..100)
        .map(|i| format!("pager-line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.show_in_pager(long_content, 24);

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).expect("test backend");
    app.draw(&mut terminal).expect("draw");

    let rendered = dump_terminal(&terminal);
    assert!(
        rendered.contains("Pager"),
        "pager header should be visible: {rendered}"
    );
    assert!(
        rendered.contains("Line"),
        "pager footer should show line info: {rendered}"
    );
}
