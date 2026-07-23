//! Integration tests for `tui::input::InputWidget` (Phase 3, Lane L6).

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use zo_cli::tui::app::AppMode;
use zo_cli::tui::HeatState;
use zo_cli::tui::input::{InputCommand, InputWidget, PLACEHOLDER};
use zo_cli::tui::theme::Theme;
use std::time::{Duration, Instant};

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    }
}

fn press_mods(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: mods,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    }
}

#[test]
fn input_inserts_plain_chars() {
    let mut input = InputWidget::new();
    for ch in "hello".chars() {
        assert!(input.handle_key(press(KeyCode::Char(ch))).is_none());
    }
    assert_eq!(input.text(), "hello");
    assert_eq!(input.cursor(), (0, 5));
}

#[test]
fn input_enter_submits_and_clears_buffer() {
    let mut input = InputWidget::new();
    for ch in "hi".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    let result = input.handle_key(press(KeyCode::Enter));
    assert_eq!(result, Some(InputCommand::Submit("hi".to_string())));
    assert!(input.is_empty());
}

#[test]
fn input_shift_enter_inserts_newline_without_submit() {
    let mut input = InputWidget::new();
    for ch in "ab".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    let r = input.handle_key(press_mods(KeyCode::Enter, KeyModifiers::SHIFT));
    assert!(r.is_none());
    input.handle_key(press(KeyCode::Char('c')));
    assert_eq!(input.lines().len(), 2);
    assert_eq!(input.text(), "ab\nc");
}

#[test]
fn input_backspace_deletes_across_newline() {
    let mut input = InputWidget::new();
    input.handle_key(press(KeyCode::Char('a')));
    input.handle_key(press_mods(KeyCode::Enter, KeyModifiers::SHIFT));
    input.handle_key(press(KeyCode::Char('b')));
    assert_eq!(input.text(), "a\nb");
    // cursor is at (1, 1). Backspace removes 'b'.
    input.handle_key(press(KeyCode::Backspace));
    assert_eq!(input.text(), "a\n");
    // Next backspace merges lines.
    input.handle_key(press(KeyCode::Backspace));
    assert_eq!(input.text(), "a");
    assert_eq!(input.cursor(), (0, 1));
}

#[test]
fn input_cursor_movement_left_right() {
    let mut input = InputWidget::new();
    for ch in "abc".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(input.cursor(), (0, 3));
    input.handle_key(press(KeyCode::Left));
    assert_eq!(input.cursor(), (0, 2));
    input.handle_key(press(KeyCode::Left));
    input.handle_key(press(KeyCode::Left));
    input.handle_key(press(KeyCode::Left)); // clamped — no underflow
    assert_eq!(input.cursor(), (0, 0));
    input.handle_key(press(KeyCode::Right));
    assert_eq!(input.cursor(), (0, 1));
}

#[test]
fn input_placeholder_rendered_when_empty() {
    let input = InputWidget::new();
    let theme = Theme::no_color();
    // 80 cols accommodates the full placeholder inside the rounded border.
    let backend = TestBackend::new(80, 6);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw(frame, area, &theme, &AppMode::Normal);
        })
        .unwrap();
    let buffer = terminal.backend().buffer().clone();
    let dumped: String = buffer
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect();
    let trimmed: String = PLACEHOLDER.trim().to_string();
    assert!(
        dumped.contains(&trimmed),
        "placeholder substring missing: {dumped}"
    );
    assert!(
        !dumped.contains("\u{2191} history"),
        "history hint must stay hidden while no history exists: {dumped}"
    );
}

#[test]
fn input_placeholder_advertises_history_recall_when_available() {
    // CC parity: once prompt history exists, the empty-buffer placeholder
    // advertises Up-arrow recall so the feature is discoverable.
    let mut input = InputWidget::new();
    input.set_history_hint(true);
    let theme = Theme::no_color();
    let backend = TestBackend::new(80, 6);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw(frame, area, &theme, &AppMode::Normal);
        })
        .unwrap();
    let buffer = terminal.backend().buffer().clone();
    let dumped: String = buffer
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect();
    assert!(
        dumped.contains("\u{2191} history"),
        "history hint missing from placeholder: {dumped}"
    );
}

#[test]
fn input_clears_stale_glyphs_when_text_shrinks() {
    // Regression: App::draw disables the per-frame full-screen Clear for the
    // diff-buffer perf win, so InputWidget::draw must wipe its own inner
    // region — else a long line's right-hand glyphs linger when the buffer
    // shrinks to a shorter string (the ghosting the user reported).
    let theme = Theme::no_color();
    let backend = TestBackend::new(40, 3);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut input = InputWidget::new();

    // Frame 1: a long single line that fills most of the inner width.
    for ch in "abcdefghijklmnopqrstuvwxyz0123".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw(frame, area, &theme, &AppMode::Normal);
        })
        .unwrap();

    // Frame 2: shrink to a short line.
    input.clear();
    for ch in "hi".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw(frame, area, &theme, &AppMode::Normal);
        })
        .unwrap();

    // The inner content row (inside the top border) must show "hi" with no
    // trailing ghost from the long line.
    let buffer = terminal.backend().buffer();
    let row1: String = (0..40)
        .map(|x| {
            buffer
                .cell((x, 1))
                .map_or(" ", ratatui::buffer::Cell::symbol)
                .to_string()
        })
        .collect();
    assert!(row1.contains("hi"), "short text should render: {row1:?}");
    assert!(
        !row1.contains("xyz") && !row1.contains("0123"),
        "stale long-line glyphs must be cleared: {row1:?}"
    );
}

#[test]
fn input_no_color_theme_draws_without_panic() {
    let mut input = InputWidget::new();
    for ch in "line1".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    input.handle_key(press_mods(KeyCode::Enter, KeyModifiers::SHIFT));
    for ch in "line2".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    let theme = Theme::no_color();
    assert!(theme.no_color);
    let backend = TestBackend::new(30, 6);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw(frame, area, &theme, &AppMode::ModalModel);
        })
        .unwrap();
    let rail = &terminal.backend().buffer()[(0, 0)];
    assert_eq!(rail.symbol(), "|");
    assert_eq!(rail.fg, ratatui::style::Color::Reset);
}

#[test]
fn input_hot_rail_fades_upward_from_the_bottom_cell() {
    let input = InputWidget::new();
    let theme = Theme::default_dark();
    let mut terminal = Terminal::new(TestBackend::new(30, 6)).unwrap();

    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw_with_heat(
                frame,
                area,
                &theme,
                &AppMode::Normal,
                HeatState::Hot,
            );
        })
        .unwrap();

    let buffer = terminal.backend().buffer();
    assert_eq!(buffer[(0, 5)].fg, theme.heat().rail_fade[0]);
    assert_eq!(buffer[(0, 5)].fg, theme.heat().molten);
    assert_eq!(buffer[(0, 4)].fg, theme.heat().rail_fade[1]);
    assert_ne!(
        buffer[(0, 5)].fg, buffer[(0, 0)].fg,
        "the Hot rail must be a real bottom-up fade"
    );
}

#[test]
fn input_cold_focused_rail_uses_steel() {
    let input = InputWidget::new();
    let theme = Theme::default_dark();
    let mut terminal = Terminal::new(TestBackend::new(30, 4)).unwrap();

    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw_with_heat(
                frame,
                area,
                &theme,
                &AppMode::Normal,
                HeatState::Cold,
            );
        })
        .unwrap();

    assert_eq!(terminal.backend().buffer()[(0, 0)].fg, theme.heat().steel);
}

#[test]
fn input_completed_cooling_settles_to_identical_buffers() {
    let input = InputWidget::new();
    let theme = Theme::default_dark();
    let mut terminal = Terminal::new(TestBackend::new(30, 4)).unwrap();
    let now = Instant::now();
    let cooled_since = now
        .checked_sub(Duration::from_secs(3))
        .expect("test instant has three seconds of history");
    let heat_state = HeatState::derive(false, Some(cooled_since), now);
    assert_eq!(heat_state, HeatState::Cold);

    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw_with_heat(
                frame,
                area,
                &theme,
                &AppMode::Normal,
                heat_state,
            );
        })
        .unwrap();
    let first = terminal.backend().buffer().clone();
    terminal
        .draw(|frame| {
            let area = frame.area();
            input.draw_with_heat(
                frame,
                area,
                &theme,
                &AppMode::Normal,
                heat_state,
            );
        })
        .unwrap();

    assert_eq!(terminal.backend().buffer(), &first);
}

#[test]
fn input_ctrl_c_emits_cancel_and_clears() {
    let mut input = InputWidget::new();
    for ch in "draft".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    let r = input.handle_key(press_mods(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert_eq!(r, Some(InputCommand::Cancel));
    assert!(input.is_empty());
}

#[test]
fn input_multi_line_buffer_yields_single_submit_string() {
    let mut input = InputWidget::new();
    for ch in "one".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    input.handle_key(press_mods(KeyCode::Enter, KeyModifiers::SHIFT));
    for ch in "two".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    input.handle_key(press_mods(KeyCode::Enter, KeyModifiers::SHIFT));
    for ch in "three".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    let r = input.handle_key(press(KeyCode::Enter));
    assert_eq!(r, Some(InputCommand::Submit("one\ntwo\nthree".to_string())));
}

#[test]
fn input_insert_text_single_line() {
    let mut input = InputWidget::new();
    input.insert_text("hello world");
    assert_eq!(input.text(), "hello world");
    assert_eq!(input.cursor(), (0, 11));
}

#[test]
fn input_insert_text_multiline() {
    let mut input = InputWidget::new();
    input.insert_text("line one\nline two\nline three");
    assert_eq!(input.text(), "line one\nline two\nline three");
    assert_eq!(input.lines().len(), 3);
    assert_eq!(input.cursor(), (2, 10));
}

/// A large paste collapses into a summary chip. Typing afterwards must keep
/// that chip collapsed while the submitted text carries both the hidden paste
/// body and the visible suffix.
#[test]
fn input_typing_after_collapsed_paste_stays_collapsed_and_submits_suffix() {
    let mut input = InputWidget::new();
    let pasted = (1..=12)
        .map(|n| format!("line {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    input.insert_text(&pasted);
    // Collapsed: the visible buffer is a one-line summary, not the raw paste.
    assert_eq!(input.lines().len(), 1);
    assert!(input.lines()[0].contains("pasted"));

    for ch in " tail".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }

    // The paste remains collapsed in the composer, but the submit payload is
    // the hidden paste body plus the newly typed suffix.
    assert_eq!(input.text(), format!("{pasted} tail"));
    assert_eq!(input.lines().len(), 1);
    assert!(input.lines()[0].contains("pasted"));
    assert!(input.lines()[0].ends_with(" tail"));
}

#[test]
fn input_collapsed_paste_preserves_prefix_and_suffix() {
    let mut input = InputWidget::new();
    let pasted = (1..=12)
        .map(|n| format!("row {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    input.insert_text("abc");
    input.handle_key(press(KeyCode::Left)); // ab|c
    input.insert_text(&pasted);
    input.handle_key(press(KeyCode::Char('X')));

    assert_eq!(input.text(), format!("ab{pasted}Xc"));
    assert_eq!(input.lines().len(), 1);
    assert!(input.lines()[0].starts_with("ab("));
    assert!(input.lines()[0].contains("pasted"));
    assert!(input.lines()[0].ends_with("Xc"));
}

/// Backspace at a collapsed paste boundary removes the paste chip as one atom;
/// it must never materialise the hidden paste body into visible input rows.
#[test]
fn input_backspace_after_collapsed_paste_deletes_chip_without_expanding() {
    let mut input = InputWidget::new();
    let pasted = (1..=11)
        .map(|n| format!("row{n}"))
        .collect::<Vec<_>>()
        .join("\n");
    input.insert_text(&pasted);
    assert!(input.lines()[0].contains("pasted"));

    input.handle_key(press(KeyCode::Backspace));

    assert!(input.is_empty());
    assert_eq!(input.text(), "");
    assert_eq!(input.lines(), &[String::new()]);
}

#[test]
fn input_insert_text_at_cursor_mid_buffer() {
    let mut input = InputWidget::new();
    for ch in "abc".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    input.handle_key(press(KeyCode::Left)); // cursor at (0, 2)
    input.insert_text("XY");
    assert_eq!(input.text(), "abXYc");
    assert_eq!(input.cursor(), (0, 4));
}

#[test]
fn input_insert_text_empty_string_is_noop() {
    let mut input = InputWidget::new();
    input.insert_text("hi");
    input.insert_text("");
    assert_eq!(input.text(), "hi");
    assert_eq!(input.cursor(), (0, 2));
}

#[test]
fn input_insert_text_with_trailing_newline() {
    let mut input = InputWidget::new();
    input.insert_text("first\n");
    assert_eq!(input.lines().len(), 2);
    assert_eq!(input.text(), "first\n");
    assert_eq!(input.cursor(), (1, 0));
}

// ── Readline-style editing: word/line kills, motion, undo/redo ──────

#[test]
fn input_ctrl_w_deletes_previous_word() {
    let mut input = InputWidget::new();
    for ch in "hello world".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    // Cursor at end (0, 11). Ctrl-W removes "world".
    let r = input.handle_key(press_mods(KeyCode::Char('w'), KeyModifiers::CONTROL));
    assert!(r.is_none());
    assert_eq!(input.text(), "hello ");
    assert_eq!(input.cursor(), (0, 6));
}

#[test]
fn input_alt_backspace_deletes_previous_word() {
    let mut input = InputWidget::new();
    input.insert_text("foo bar");
    let r = input.handle_key(press_mods(KeyCode::Backspace, KeyModifiers::ALT));
    assert!(r.is_none());
    assert_eq!(input.text(), "foo ");
}

#[test]
fn input_alt_d_deletes_next_word() {
    let mut input = InputWidget::new();
    input.insert_text("foo bar baz");
    for _ in 0..11 {
        input.handle_key(press(KeyCode::Left));
    }
    assert_eq!(input.cursor(), (0, 0));
    input.handle_key(press_mods(KeyCode::Char('d'), KeyModifiers::ALT));
    assert_eq!(input.text(), " bar baz");
    assert_eq!(input.cursor(), (0, 0));
}

#[test]
fn input_ctrl_k_kills_to_line_end() {
    let mut input = InputWidget::new();
    input.insert_text("keep this");
    for _ in 0..4 {
        input.handle_key(press(KeyCode::Left));
    }
    assert_eq!(input.cursor(), (0, 5));
    input.handle_key(press_mods(KeyCode::Char('k'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "keep ");
}

#[test]
fn input_ctrl_a_moves_to_line_start() {
    let mut input = InputWidget::new();
    input.insert_text("hello world");
    assert_eq!(input.cursor(), (0, 11));
    let r = input.handle_key(press_mods(KeyCode::Char('a'), KeyModifiers::CONTROL));
    assert!(r.is_none());
    assert_eq!(input.cursor(), (0, 0));
    // Text is untouched — Ctrl-A is pure motion.
    assert_eq!(input.text(), "hello world");
}

#[test]
fn input_ctrl_e_moves_to_line_end() {
    let mut input = InputWidget::new();
    input.insert_text("hello world");
    // Park the cursor at the line start first.
    input.handle_key(press_mods(KeyCode::Char('a'), KeyModifiers::CONTROL));
    assert_eq!(input.cursor(), (0, 0));
    let r = input.handle_key(press_mods(KeyCode::Char('e'), KeyModifiers::CONTROL));
    assert!(r.is_none());
    assert_eq!(input.cursor(), (0, 11));
    assert_eq!(input.text(), "hello world");
}

#[test]
fn input_ctrl_u_kills_whole_line_and_is_undoable() {
    let mut input = InputWidget::new();
    input.insert_text("delete me");
    let r = input.handle_key(press_mods(KeyCode::Char('u'), KeyModifiers::CONTROL));
    assert!(r.is_none());
    assert!(input.is_empty());
    assert_eq!(input.cursor(), (0, 0));
    // The kill is checkpointed, so a single undo restores the line.
    assert!(input.undo());
    assert_eq!(input.text(), "delete me");
}

#[test]
fn input_alt_b_and_alt_f_move_by_word() {
    let mut input = InputWidget::new();
    input.insert_text("alpha beta gamma");
    input.handle_key(press_mods(KeyCode::Char('b'), KeyModifiers::ALT));
    assert_eq!(input.cursor(), (0, 11)); // start of "gamma"
    input.handle_key(press_mods(KeyCode::Char('b'), KeyModifiers::ALT));
    assert_eq!(input.cursor(), (0, 6)); // start of "beta"
    input.handle_key(press_mods(KeyCode::Char('f'), KeyModifiers::ALT));
    assert_eq!(input.cursor(), (0, 11)); // forward to "gamma"
}

#[test]
fn input_ctrl_arrows_move_by_word() {
    let mut input = InputWidget::new();
    input.insert_text("one two");
    input.handle_key(press_mods(KeyCode::Left, KeyModifiers::CONTROL));
    assert_eq!(input.cursor(), (0, 4)); // start of "two"
    input.handle_key(press_mods(KeyCode::Right, KeyModifiers::CONTROL));
    assert_eq!(input.cursor(), (0, 7));
}

#[test]
fn input_undo_and_redo_round_trip() {
    let mut input = InputWidget::new();
    for ch in "abc".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(input.text(), "abc");
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "");
    // Redo lives on Alt-Z (pairs with Ctrl-Z undo); Ctrl-Y is the readline
    // yank now.
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::ALT));
    assert_eq!(input.text(), "abc");
    assert_eq!(input.cursor(), (0, 3));
}

#[test]
fn input_undo_is_word_granular() {
    let mut input = InputWidget::new();
    for ch in "hello world".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    assert_eq!(input.text(), "hello world");
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "hello ");
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "hello");
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "");
}

#[test]
fn input_undo_restores_word_deletion() {
    let mut input = InputWidget::new();
    input.insert_text("keep gone");
    input.handle_key(press_mods(KeyCode::Char('w'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "keep ");
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "keep gone");
    assert_eq!(input.cursor(), (0, 9));
}

#[test]
fn input_cursor_move_breaks_undo_coalescing() {
    let mut input = InputWidget::new();
    for ch in "ab".chars() {
        input.handle_key(press(KeyCode::Char(ch)));
    }
    input.handle_key(press(KeyCode::Left)); // breaks the typing run
    input.handle_key(press(KeyCode::Char('X'))); // -> "aXb"
    assert_eq!(input.text(), "aXb");
    input.handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "ab"); // only X is undone
}

#[test]
fn input_ctrl_w_respects_cjk_word_boundary() {
    let mut input = InputWidget::new();
    input.insert_text("한국어 코드");
    assert_eq!(input.cursor(), (0, 6));
    input.handle_key(press_mods(KeyCode::Char('w'), KeyModifiers::CONTROL));
    assert_eq!(input.text(), "한국어 ");
    assert_eq!(input.cursor(), (0, 4));
}

#[test]
fn input_undo_on_empty_buffer_is_harmless() {
    let mut input = InputWidget::new();
    // Nothing to undo/redo/yank — must not panic or emit a command.
    assert!(input
        .handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::CONTROL))
        .is_none());
    assert!(input
        .handle_key(press_mods(KeyCode::Char('z'), KeyModifiers::ALT))
        .is_none());
    assert!(input
        .handle_key(press_mods(KeyCode::Char('y'), KeyModifiers::CONTROL))
        .is_none());
    assert!(input.is_empty());
}
