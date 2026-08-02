//! Integration tests for `tui::modals` (Phase 3, Lane L6).

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::Terminal;
use runtime::message_stream::ActiveModel;
use runtime::PermissionMode;
use zo_cli::tui::modals::{
    ChoicePickerModal, ModalResult, ModalSelection, ModelPickerEntry, ModelPickerModal,
    PermissionPickerModal, ToolToggleModal, ToolToggleRow,
};
use zo_cli::tui::theme::Theme;

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    }
}

fn model(provider: &'static str, alias: &str) -> ActiveModel {
    ActiveModel {
        provider,
        alias: alias.to_string(),
        display_name: format!("{provider}:{alias}"),
        context_limit: 200_000,
    }
}

fn entry(provider: &'static str, alias: &str) -> ModelPickerEntry {
    ModelPickerEntry {
        provider: provider.to_string(),
        model: model(provider, alias),
    }
}

fn grouped_registry() -> Vec<ModelPickerEntry> {
    vec![
        entry("anthropic", "opus"),
        entry("anthropic", "sonnet"),
        entry("anthropic", "haiku"),
        entry("codex", "gpt-5"),
        entry("codex", "gpt-5-mini"),
    ]
}

// ---------------------------------------------------------------------------
// ModelPickerModal
// ---------------------------------------------------------------------------

#[test]
fn model_picker_cursor_moves_down_within_group() {
    let mut picker = ModelPickerModal::new(grouped_registry());
    picker.handle_key(press(KeyCode::Down));
    let result = picker.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::Model(m))) => {
            assert_eq!(m.alias, "sonnet");
            assert_eq!(m.provider, "anthropic");
        }
        other => panic!("expected anthropic:sonnet, got {other:?}"),
    }
}

#[test]
fn model_picker_right_arrow_jumps_to_next_provider_group() {
    let mut picker = ModelPickerModal::new(grouped_registry());
    picker.handle_key(press(KeyCode::Right));
    let result = picker.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::Model(m))) => {
            assert_eq!(m.provider, "codex");
        }
        other => panic!("expected codex group, got {other:?}"),
    }
}

#[test]
fn model_picker_left_arrow_jumps_back_to_previous_group() {
    let mut picker = ModelPickerModal::new(grouped_registry());
    picker.handle_key(press(KeyCode::Right));
    picker.handle_key(press(KeyCode::Left));
    let result = picker.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::Model(m))) => {
            assert_eq!(m.provider, "anthropic");
            assert_eq!(m.alias, "opus");
        }
        other => panic!("expected first anthropic entry, got {other:?}"),
    }
}

#[test]
fn model_picker_esc_cancels() {
    let mut picker = ModelPickerModal::new(grouped_registry());
    let result = picker.handle_key(press(KeyCode::Esc));
    assert!(matches!(result, Some(ModalResult::Cancelled)));
}

#[test]
fn model_picker_renders_provider_group_headers() {
    let theme = Theme::no_color();
    let picker = ModelPickerModal::new(grouped_registry());
    let lines = picker.render_lines(&theme);
    let joined: String = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.clone()))
        .collect::<Vec<_>>()
        .join("");
    assert!(
        joined.contains("anthropic") && joined.contains("codex"),
        "expected both provider groups in render output, got:\n{joined}"
    );
}

#[test]
fn model_picker_draws_into_test_backend_without_panicking() {
    let theme = Theme::no_color();
    let picker = ModelPickerModal::new(grouped_registry());
    let backend = TestBackend::new(60, 20);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 20);
            picker.draw(frame, area, &theme);
        })
        .expect("draw");
}

// ---------------------------------------------------------------------------
// PermissionPickerModal
// ---------------------------------------------------------------------------

#[test]
fn permission_picker_enter_returns_current_mode() {
    let mut picker = PermissionPickerModal::with_selected(PermissionMode::ReadOnly);
    let result = picker.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::Permission(m))) => {
            assert_eq!(m, PermissionMode::ReadOnly);
        }
        other => panic!("expected ReadOnly selection, got {other:?}"),
    }
}

#[test]
fn permission_picker_down_arrow_advances_selection() {
    let mut picker = PermissionPickerModal::with_selected(PermissionMode::ReadOnly);
    picker.handle_key(press(KeyCode::Down));
    let next = picker.current();
    assert_ne!(next, PermissionMode::ReadOnly);
}

#[test]
fn permission_picker_esc_cancels() {
    let mut picker = PermissionPickerModal::with_selected(PermissionMode::ReadOnly);
    let result = picker.handle_key(press(KeyCode::Esc));
    assert!(matches!(result, Some(ModalResult::Cancelled)));
}

#[test]
fn permission_picker_draws_in_no_color_theme() {
    let theme = Theme::no_color();
    let picker = PermissionPickerModal::with_selected(PermissionMode::WorkspaceWrite);
    let backend = TestBackend::new(60, 20);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 20);
            picker.draw(frame, area, &theme);
        })
        .expect("draw");
}

// ---------------------------------------------------------------------------
// ChoicePickerModal
// ---------------------------------------------------------------------------

#[test]
fn choice_picker_enter_returns_selected_index_and_label() {
    let mut picker = ChoicePickerModal::new(
        "Pick one",
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
    );
    picker.handle_key(press(KeyCode::Down));
    picker.handle_key(press(KeyCode::Down));
    let result = picker.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::Choice { index, label })) => {
            assert_eq!(index, 2);
            assert_eq!(label, "gamma");
        }
        other => panic!("expected gamma selection, got {other:?}"),
    }
}

#[test]
fn choice_picker_esc_cancels() {
    let mut picker = ChoicePickerModal::new("Pick", vec!["a".to_string(), "b".to_string()]);
    let result = picker.handle_key(press(KeyCode::Esc));
    assert!(matches!(result, Some(ModalResult::Cancelled)));
}

#[test]
fn choice_picker_up_does_not_underflow_at_top() {
    let mut picker = ChoicePickerModal::new("Pick", vec!["only".to_string()]);
    picker.handle_key(press(KeyCode::Up));
    picker.handle_key(press(KeyCode::Up));
    let result = picker.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::Choice { index, .. })) => {
            assert_eq!(index, 0);
        }
        other => panic!("expected index 0, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// ToolToggleModal
// ---------------------------------------------------------------------------

fn tool_row(name: &str, source: &str, enabled: bool) -> ToolToggleRow {
    ToolToggleRow {
        name: name.to_string(),
        description: Some(format!("{name} description")),
        source: source.to_string(),
        enabled,
    }
}

#[test]
fn tool_toggle_modal_enter_toggles_current_tool() {
    let mut modal = ToolToggleModal::new(vec![
        tool_row("WebSearch", "builtin", true),
        tool_row("mcp__demo__echo", "mcp", false),
    ]);
    let result = modal.handle_key(press(KeyCode::Enter));
    match result {
        Some(ModalResult::Selected(ModalSelection::ToolToggle { name, enabled })) => {
            assert_eq!(name, "WebSearch");
            assert!(!enabled);
        }
        other => panic!("expected WebSearch toggle, got {other:?}"),
    }
    assert!(!modal.rows()[modal.cursor()].enabled);
}

#[test]
fn tool_toggle_modal_renders_enabled_and_disabled_state() {
    let theme = Theme::no_color();
    let modal = ToolToggleModal::new(vec![
        tool_row("read_file", "builtin", true),
        tool_row("mcp__demo__echo", "mcp", false),
    ]);
    let joined = modal
        .render_lines(&theme, 8, 80)
        .into_iter()
        .flat_map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    // Pi select-list idiom: the enabled/disabled state rides a check marker,
    // not a `[x]`/`[ ]` checkbox. Under NO_COLOR the glyphs degrade to `*`/`-`.
    assert!(joined.contains("* "), "enabled row marker missing: {joined}");
    assert!(joined.contains("- "), "disabled row marker missing: {joined}");
    assert!(joined.contains("[builtin]"));
    assert!(joined.contains("mcp__demo__echo"));
}

#[test]
fn tool_toggle_modal_draws_in_no_color_theme() {
    let theme = Theme::no_color();
    let modal = ToolToggleModal::new(vec![tool_row("read_file", "builtin", true)]);
    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 80, 20);
            modal.draw(frame, area, &theme);
        })
        .expect("draw");
}

// ---------------------------------------------------------------------------
// ProviderManagerModal (/providers) + AskUserQuestion option previews
//
// These paint through the real `draw` path — geometry, borders, the two-pane
// split — which the per-modal line tests deliberately do not exercise.
// ---------------------------------------------------------------------------

/// Flatten a `TestBackend` frame into one string per row.
fn frame_rows(terminal: &Terminal<TestBackend>) -> Vec<String> {
    let buffer = terminal.backend().buffer();
    let area = buffer.area();
    (0..area.height)
        .map(|y| {
            (0..area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn draw_modal<F>(width: u16, height: u16, draw: F) -> Vec<String>
where
    F: FnOnce(&mut ratatui::Frame<'_>, Rect, &Theme),
{
    let theme = Theme::default_dark();
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, width, height);
            draw(frame, area, &theme);
        })
        .expect("draw");
    frame_rows(&terminal)
}

#[test]
fn provider_manager_paints_the_tree_with_models_and_the_add_row() {
    use zo_cli::tui::modals::{
        ProviderAccountRow, ProviderKeyState, ProviderManagerModal, ProviderManagerRow,
        ProviderOrigin,
    };

    let accounts = vec![ProviderAccountRow {
        id: "claude".to_string(),
        label: "Claude".to_string(),
        detail: "saved Anthropic OAuth".to_string(),
        connected: true,
        disconnectable: true,
    }];
    let rows = vec![
        ProviderManagerRow {
            name: "deepseek".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            auth_env: Some("DEEPSEEK_API_KEY".to_string()),
            models: vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()],
            key_state: ProviderKeyState::Stored,
            origin: ProviderOrigin::GlobalSettings,
            key_shared: false,
        },
        ProviderManagerRow {
            name: "ollama".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
            auth_env: None,
            models: vec!["llama3.1".to_string()],
            key_state: ProviderKeyState::Keyless,
            origin: ProviderOrigin::GlobalSettings,
            key_shared: false,
        },
    ];
    let modal = ProviderManagerModal::new(accounts, rows, "/home/u/.zo/settings.json");
    let painted = draw_modal(90, 24, |frame, area, theme| {
        modal.draw(frame, area, theme);
    })
    .join("\n");

    assert!(painted.contains("Providers"), "{painted}");
    assert!(painted.contains("Accounts"), "{painted}");
    assert!(painted.contains("Claude"), "{painted}");
    assert!(painted.contains("Registered providers"), "{painted}");
    assert!(painted.contains("deepseek"), "{painted}");
    assert!(
        painted.contains("deepseek-chat"),
        "the first provider paints expanded: {painted}"
    );
    assert!(painted.contains("ollama"), "{painted}");
    assert!(
        painted.contains("/home/u/.zo/settings.json"),
        "the global scope is visible on screen: {painted}"
    );
    assert!(
        painted.contains("Add a provider"),
        "registration is reachable from the list: {painted}"
    );
    // The footer tracks the highlighted row, which starts on the account.
    assert!(
        painted.contains("disconnect"),
        "account key hints paint: {painted}"
    );
}

#[test]
fn user_question_with_previews_paints_options_beside_the_mockup() {
    use runtime::message_stream::{BlockId, QuestionOption, UserQuestionPrompt};
    use zo_cli::tui::modals::UserQuestionModal;

    let (responder, _rx) = tokio::sync::oneshot::channel();
    let prompt = UserQuestionPrompt {
        id: BlockId(1),
        question: "Which layout?".to_string(),
        header: Some("Layout".to_string()),
        options: vec![
            QuestionOption::plain("Sidebar")
                .with_preview("+-------+------+\n| nav   | body |\n+-------+------+"),
            QuestionOption::plain("Topbar")
                .with_preview("+--------------+\n| topbar       |\n+--------------+"),
        ],
        multi_select: false,
        responder,
    };
    let modal = UserQuestionModal::from_prompt(&prompt);
    let painted = draw_modal(110, 20, |frame, area, theme| {
        modal.draw(frame, area, theme);
    });
    let joined = painted.join("\n");

    assert!(joined.contains("Which layout?"), "{joined}");
    assert!(joined.contains("1. Sidebar"), "{joined}");
    assert!(joined.contains("preview"), "the pane is labelled: {joined}");
    assert!(
        joined.contains("| nav   | body |"),
        "the focused option's mockup paints verbatim: {joined}"
    );
    // Side-by-side means an option row and the mockup share a screen row.
    assert!(
        painted
            .iter()
            .any(|row| row.contains("Sidebar") && row.contains("+---")),
        "options and preview must sit on the same rows: {joined}"
    );
    // The narrow list column must not fold its hints onto a second line.
    assert!(
        painted
            .iter()
            .any(|row| row.contains("Esc") && row.contains("move")),
        "the footer stays on one line beside the preview: {joined}"
    );
}

#[test]
fn a_narrow_terminal_falls_back_to_the_single_column_question_layout() {
    use runtime::message_stream::{BlockId, QuestionOption, UserQuestionPrompt};
    use zo_cli::tui::modals::UserQuestionModal;

    let (responder, _rx) = tokio::sync::oneshot::channel();
    let prompt = UserQuestionPrompt {
        id: BlockId(2),
        question: "Which layout?".to_string(),
        header: None,
        options: vec![QuestionOption::plain("Sidebar").with_preview("| nav | body |")],
        multi_select: false,
        responder,
    };
    let modal = UserQuestionModal::from_prompt(&prompt);
    let painted = draw_modal(48, 14, |frame, area, theme| {
        modal.draw(frame, area, theme);
    })
    .join("\n");

    assert!(painted.contains("1. Sidebar"), "{painted}");
    assert!(
        !painted.contains("preview"),
        "a narrow terminal keeps one column instead of clipping the mockup: {painted}"
    );
}

// ---------------------------------------------------------------------------
// Non-clipping bodies
// ---------------------------------------------------------------------------

/// A `Paragraph` without `.wrap(...)` composes its rows through ratatui's
/// `LineTruncator`, which stops at the rect edge and appends nothing — so an
/// over-wide row loses its tail *silently*, and on a CJK boundary it loses it
/// mid-glyph. Every modal body now fits its rows to the rect first
/// (`modals::fit_body_rows`), which cuts with a visible `…`.
///
/// The assertion is on the marker, not on the absence of the tail: a row that
/// merely happens to end at the rect edge is indistinguishable from one that was
/// cut, and that ambiguity is exactly the bug.
#[test]
fn an_over_wide_tool_row_is_cut_with_a_visible_ellipsis() {
    let rows = vec![ToolToggleRow {
        name: "a_tool_whose_name_runs_far_past_any_reasonable_modal_width".to_string(),
        description: None,
        source: "builtin".to_string(),
        enabled: true,
    }];
    let modal = ToolToggleModal::new(rows);
    let painted = draw_modal(44, 10, |frame, area, theme| {
        modal.draw(frame, area, theme);
    });

    let row = painted
        .iter()
        .find(|row| row.contains("a_tool_whose_name"))
        .expect("the tool row paints");
    assert!(
        row.ends_with('\u{2026}'),
        "an over-wide row must announce its cut: {row:?}"
    );
    assert!(
        !row.contains("reasonable_modal_width"),
        "the tail is gone, as it must be at this width: {row:?}"
    );
}

/// Hangul is two cells per syllable, so a `chars().count()` budget would let a
/// truncated row overrun the rect by one column per syllable — which is what
/// paints a half-glyph at the edge. The cut is measured in cells.
#[test]
fn a_hangul_row_is_cut_on_a_cell_boundary_not_a_char_boundary() {
    let rows = vec![ToolToggleRow {
        name: "가나다라마바사아자차카타파하가나다라마바사아자차카타파하".to_string(),
        description: None,
        source: "한글".to_string(),
        enabled: true,
    }];
    let modal = ToolToggleModal::new(rows);
    let width = 30u16;
    let painted = draw_modal(width, 8, |frame, area, theme| {
        modal.draw(frame, area, theme);
    });

    let row = painted
        .iter()
        .find(|row| row.contains('\u{ac00}'))
        .unwrap_or_else(|| panic!("the Hangul row paints: {painted:#?}"));
    assert!(
        row.ends_with('\u{2026}'),
        "the Hangul row must announce its cut: {row:?}"
    );
    // One dumped `char` per buffer cell: a wide glyph occupies its own cell and
    // ratatui blanks the continuation cell, so counting chars here counts columns
    // (which is why the dump reads `가 나 다` — an artifact, not a gap).
    assert!(
        row.chars().count() <= usize::from(width),
        "the cut row must fit the rect in columns: {row:?}"
    );
}

/// `truncate_line_to_cells` rebuilds the row from the spans it kept, so it has
/// to carry the row-level style across: a selected list row paints its highlight
/// band through `Line::style`, and dropping it would strip the highlight off
/// precisely the rows long enough to need cutting.
#[test]
fn truncation_keeps_the_row_level_style_that_paints_a_selection_band() {
    use ratatui::style::{Color, Style};
    use ratatui::text::Line;
    use zo_cli::tui::text_metrics::truncate_line_to_cells;

    let band = Style::new().bg(Color::Indexed(238));
    let line = Line::from("a label far wider than the budget").style(band);
    let cut = truncate_line_to_cells(line, 12);

    assert_eq!(cut.style, band, "the selection band survives the cut");
    assert_eq!(
        cut.spans
            .iter()
            .map(|span| zo_cli::tui::text_metrics::display_width(span.content.as_ref()))
            .sum::<usize>(),
        12,
        "the cut row fills the budget exactly, ellipsis included"
    );
}

/// ratatui's `WordWrapper` decides a row is full *before* adding the grapheme it
/// is holding, so a double-width glyph on the boundary pushes the row one cell
/// past the limit and the paint — which cannot draw half a glyph — loses it. On
/// `ratatui-widgets` 0.3.0, wrapping this sentence into 13 cells paints a 14-cell
/// row and `한` disappears from the middle of it.
///
/// Every modal body that can hold CJK wraps through `wrap_line_to_cells` instead.
/// The contract it owes them is exactly this: nothing lost, nothing over-wide.
#[test]
fn wrapping_hangul_loses_no_syllable_at_any_width() {
    use ratatui::text::Line;
    use zo_cli::tui::text_metrics::{display_width, wrap_line_to_cells};

    let corpus = [
        "말줄임표로 잘라낸다 (권장) 아주 긴 한글 문장을 좁은 폭에서 접어봅니다",
        "모달 본문 잘림을 어떤 방식으로 고칠까요? mixed ASCII and 한글 in one paragraph",
        "supercalifragilisticexpialidocious 그리고 아주아주긴한글단어하나가섞여있는경우",
        "a b c d e f g h i j k l m n o p",
    ];
    for source in corpus {
        let expected: String = source.chars().filter(|c| !c.is_whitespace()).collect();
        for width in 2usize..=60 {
            let rows = wrap_line_to_cells(&Line::from(source), width, true);
            for row in &rows {
                let cells: usize = row
                    .spans
                    .iter()
                    .map(|span| display_width(span.content.as_ref()))
                    .sum();
                assert!(
                    cells <= width,
                    "a wrapped row must fit the width it was given, got {cells} in {width}: {row:?}"
                );
            }
            let painted: String = rows
                .iter()
                .flat_map(|row| row.spans.iter())
                .flat_map(|span| span.content.chars())
                .filter(|c| !c.is_whitespace())
                .collect();
            assert_eq!(
                expected, painted,
                "wrapping at width {width} dropped characters from {source:?}"
            );
        }
    }
}

/// The end of the same path: a Korean question, painted. Every syllable the user
/// typed has to reach the screen at every width the modal is willing to draw at.
#[test]
fn a_korean_question_paints_every_syllable_at_narrow_widths() {
    use runtime::message_stream::{BlockId, QuestionOption, UserQuestionPrompt};
    use zo_cli::tui::modals::UserQuestionModal;

    let question = "모달 본문 잘림을 어떤 방식으로 고칠까요? 아주 긴 질문 문장을 넣어 확인합니다";
    for width in [30u16, 34, 38, 41, 45, 46, 47, 52, 61, 70] {
        let (responder, _rx) = tokio::sync::oneshot::channel();
        let prompt = UserQuestionPrompt {
            id: BlockId(1),
            question: question.to_string(),
            header: None,
            options: vec![QuestionOption::plain("예")],
            multi_select: false,
            responder,
        };
        let modal = UserQuestionModal::from_prompt(&prompt);
        let painted = draw_modal(width, 30, |frame, area, theme| {
            modal.draw(frame, area, theme);
        })
        .join("");
        for syllable in question.chars().filter(|c| !c.is_whitespace()) {
            assert!(
                painted.contains(syllable),
                "width {width} lost {syllable:?} out of the question"
            );
        }
    }
}
