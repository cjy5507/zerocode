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
    assert!(joined.contains("[x]"));
    assert!(joined.contains("[ ]"));
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
