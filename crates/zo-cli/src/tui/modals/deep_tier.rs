//! Interactive `/tier` picker for the ordered Architect PLAN/VERIFY model pool.
//!
//! The modal owns only selection, input, confirmation, and rendering state.
//! Settings reads and writes stay in the session host, which feeds a fresh
//! [`DeepTierView`] back after every [`DeepTierAction`].

use commands::DeepTierAction;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};
use unicode_width::UnicodeWidthStr;

use super::super::cards::{CardFrame, SurfaceKind};
use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, draw_scrollbar, key_hint_footer,
    selected_style,
};

/// Active ordered pool plus the source that supplied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepTierView {
    /// Active models in preference order.
    pub models: Vec<String>,
    /// Whether the merged pool came from explicit configuration.
    pub configured: bool,
}

/// List picker and inline editor for [`DeepTierView`].
#[derive(Debug, Clone)]
pub struct DeepTierModal {
    view: DeepTierView,
    selected: usize,
    input: Option<String>,
    confirming_reset: bool,
    feedback: Option<(String, bool)>,
}

impl DeepTierModal {
    #[must_use]
    pub fn new(view: DeepTierView) -> Self {
        Self {
            view,
            selected: 0,
            input: None,
            confirming_reset: false,
            feedback: None,
        }
    }

    /// Land the authoritative post-action snapshot and its existing text-command result.
    pub fn apply_update(&mut self, view: Option<DeepTierView>, result: Result<String, String>) {
        if let Some(view) = view {
            let selected_model = self.view.models.get(self.selected).cloned();
            self.view = view;
            self.selected = selected_model
                .as_deref()
                .and_then(|model| self.view.models.iter().position(|candidate| candidate == model))
                .unwrap_or_else(|| self.selected.min(self.view.models.len().saturating_sub(1)));
        }
        self.input = None;
        self.confirming_reset = false;
        self.feedback = Some(match result {
            Ok(message) => (single_line(&message), false),
            Err(error) => (single_line(&error), true),
        });
    }

    pub fn paste_text(&mut self, text: &str) {
        if let Some(input) = self.input.as_mut() {
            input.extend(text.chars().filter(|ch| !ch.is_control()));
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if self.input.is_some() {
            return self.handle_input_key(key);
        }
        if self.confirming_reset {
            return self.handle_reset_confirmation(key);
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(ModalResult::Cancelled),
            KeyCode::Up | KeyCode::Char('k') => {
                self.select_up(1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.select_down(1);
                None
            }
            KeyCode::Home => {
                self.selected = 0;
                None
            }
            KeyCode::End => {
                self.selected = self.view.models.len().saturating_sub(1);
                None
            }
            KeyCode::Char('a') if key.modifiers.is_empty() => {
                self.input = Some(String::new());
                self.feedback = None;
                None
            }
            KeyCode::Char('d')
                if key.modifiers.is_empty() && !self.view.models.is_empty() =>
            {
                self.feedback = None;
                Some(Self::selection(DeepTierAction::Remove {
                    target: (self.selected + 1).to_string(),
                }))
            }
            KeyCode::Delete if !self.view.models.is_empty() => {
                self.feedback = None;
                Some(Self::selection(DeepTierAction::Remove {
                    target: (self.selected + 1).to_string(),
                }))
            }
            KeyCode::Char('K') if self.selected > 0 => {
                self.feedback = None;
                Some(Self::selection(DeepTierAction::Move {
                    from: self.selected + 1,
                    to: self.selected,
                }))
            }
            KeyCode::Char('J') if self.selected + 1 < self.view.models.len() => {
                self.feedback = None;
                Some(Self::selection(DeepTierAction::Move {
                    from: self.selected + 1,
                    to: self.selected + 2,
                }))
            }
            KeyCode::Char('r') if key.modifiers.is_empty() => {
                self.confirming_reset = true;
                self.feedback = None;
                None
            }
            _ => None,
        }
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        let input = self.input.as_mut()?;
        match key.code {
            KeyCode::Esc => {
                self.input = None;
                None
            }
            KeyCode::Enter => {
                let model = input.trim().to_string();
                if model.is_empty() {
                    return None;
                }
                self.input = None;
                Some(Self::selection(DeepTierAction::Add { model }))
            }
            KeyCode::Backspace => {
                input.pop();
                None
            }
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                input.push(ch);
                None
            }
            _ => None,
        }
    }

    fn handle_reset_confirmation(&mut self, key: KeyEvent) -> Option<ModalResult> {
        match key.code {
            KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                self.confirming_reset = false;
                Some(Self::selection(DeepTierAction::Reset))
            }
            KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                self.confirming_reset = false;
                None
            }
            KeyCode::Char('q') => Some(ModalResult::Cancelled),
            _ => None,
        }
    }

    fn selection(action: DeepTierAction) -> ModalResult {
        ModalResult::Selected(ModalSelection::DeepTier(action))
    }

    fn select_up(&mut self, rows: usize) {
        self.selected = self.selected.saturating_sub(rows);
    }

    fn select_down(&mut self, rows: usize) {
        self.selected = self
            .selected
            .saturating_add(rows)
            .min(self.view.models.len().saturating_sub(1));
    }

    fn list_offset(&self, height: u16) -> u16 {
        let len = u16::try_from(self.view.models.len()).unwrap_or(u16::MAX);
        let max_offset = len.saturating_sub(height);
        let selected = u16::try_from(self.selected).unwrap_or(u16::MAX);
        selected
            .saturating_sub(height.saturating_sub(1))
            .min(max_offset)
    }

    #[must_use]
    fn content_rows(&self) -> usize {
        self.view.models.len().max(1).saturating_add(5)
    }

    #[must_use]
    pub fn desired_size(&self, area: Rect, theme: &Theme) -> (u16, u16) {
        let source = self.source_label();
        let row_width = self
            .view
            .models
            .iter()
            .enumerate()
            .map(|(index, model)| {
                let marker = cursor_marker(!theme.no_color);
                format!("{marker}{}. {model} ({source})", index + 1).width()
            })
            .max()
            .unwrap_or_default();
        let footer_width = normal_footer_lines(theme)
            .iter()
            .map(line_width)
            .max()
            .unwrap_or_default();
        let content_width = row_width
            .max(footer_width)
            .max("Architect PLAN/VERIFY pool · first entry is preferred".width());
        let width = u16::try_from(content_width.saturating_add(4))
            .unwrap_or(u16::MAX)
            .clamp(64, 104)
            .min(area.width.saturating_sub(4).max(24));
        let content = u16::try_from(self.content_rows())
            .unwrap_or(u16::MAX)
            .saturating_add(2);
        let height = content
            .clamp(9, 24)
            .min(area.height.saturating_sub(2).max(6));
        (width, height)
    }

    pub fn scroll(&mut self, up: bool, rows: usize) {
        if up {
            self.select_up(rows);
        } else {
            self.select_down(rows);
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = CardFrame::new(SurfaceKind::Modal, theme)
            .title(Line::styled(" Deep-tier models ", theme.typography.heading_1))
            .padding(Padding::symmetric(1, 0))
            .render(frame, area);
        if inner.width == 0 || inner.height < 5 {
            return;
        }
        let [header, list, action, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .areas(inner);

        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "Architect PLAN/VERIFY pool · first entry is preferred",
                theme.typography.dim,
            ))),
            header,
        );

        let rows = if self.view.models.is_empty() {
            vec![Line::from(Span::styled("no active models", theme.typography.dim))]
        } else {
            self.view
                .models
                .iter()
                .enumerate()
                .map(|(index, model)| self.row_line(index, model, theme))
                .collect()
        };
        let offset = self.list_offset(list.height);
        frame.render_widget(Paragraph::new(rows).scroll((offset, 0)), list);
        draw_scrollbar(frame, list, offset, self.view.models.len(), theme);

        frame.render_widget(Paragraph::new(self.action_line(theme)), action);
        frame.render_widget(Paragraph::new(self.footer_lines(theme)), footer);
    }

    fn row_line(&self, index: usize, model: &str, theme: &Theme) -> Line<'static> {
        let selected = index == self.selected;
        let marker = if selected {
            cursor_marker(!theme.no_color)
        } else {
            blank_marker()
        };
        let style = if selected { selected_style(theme) } else { theme.typography.body };
        Line::from(Span::styled(
            format!("{marker}{}. {model} ({})", index + 1, self.source_label()),
            style,
        ))
    }

    fn action_line(&self, theme: &Theme) -> Line<'static> {
        if let Some(input) = self.input.as_ref() {
            return Line::from(vec![
                Span::styled("add model ❯ ", Style::new().fg(theme.palette.accent)),
                Span::styled(input.clone(), theme.typography.body),
                Span::styled("▌", Style::new().fg(theme.palette.accent)),
            ]);
        }
        if self.confirming_reset {
            return Line::from(Span::styled(
                "Reset to the built-in default?",
                Style::new().fg(theme.palette.warn),
            ));
        }
        if let Some((message, is_error)) = self.feedback.as_ref() {
            let color = if *is_error { theme.palette.warn } else { theme.palette.accent };
            return Line::from(Span::styled(message.clone(), Style::new().fg(color)));
        }
        Line::from(Span::styled(
            format!("{} active models · {}", self.view.models.len(), self.source_label()),
            theme.typography.dim,
        ))
    }

    fn footer_lines(&self, theme: &Theme) -> Vec<Line<'static>> {
        if self.input.is_some() {
            return vec![
                next_turn_line(theme),
                Line::default(),
                key_hint_footer(theme, &[("Enter", "add"), ("Esc", "cancel input")]),
            ];
        }
        if self.confirming_reset {
            return vec![
                next_turn_line(theme),
                Line::default(),
                key_hint_footer(theme, &[("y/Enter", "confirm"), ("n/Esc", "cancel")]),
            ];
        }
        normal_footer_lines(theme)
    }

    fn source_label(&self) -> &'static str {
        if self.view.configured { "configured" } else { "built-in default" }
    }
}

fn normal_footer_lines(theme: &Theme) -> Vec<Line<'static>> {
    vec![
        next_turn_line(theme),
        Line::default(),
        key_hint_footer(
            theme,
            &[
                ("↑↓/j k", "select"),
                ("a", "add"),
                ("d/Del", "remove"),
                ("K/J", "move"),
                ("r", "reset"),
                ("Esc/q", "close"),
            ],
        ),
    ]
}

fn next_turn_line(theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        "changes apply from the next turn",
        theme.typography.dim,
    ))
}

fn single_line(value: &str) -> String {
    value.replace(['\r', '\n'], "  ·  ")
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| span.content.as_ref().width())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn view(models: &[&str], configured: bool) -> DeepTierView {
        DeepTierView {
            models: models.iter().map(|model| (*model).to_string()).collect(),
            configured,
        }
    }

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn shifted(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::SHIFT)
    }

    fn dump(modal: &DeepTierModal, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| modal.draw(frame, frame.area(), &Theme::zo()))
            .expect("draw");
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|row| {
                (0..width)
                    .map(|column| buffer[(column, row)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn selection_clamps_at_pool_bounds() {
        let mut modal = DeepTierModal::new(view(&["architect-a", "architect-b"], true));
        modal.handle_key(press(KeyCode::Up));
        assert_eq!(modal.selected, 0);
        modal.handle_key(press(KeyCode::Char('j')));
        modal.handle_key(press(KeyCode::Down));
        assert_eq!(modal.selected, 1);
        modal.handle_key(press(KeyCode::Char('k')));
        assert_eq!(modal.selected, 0);
    }

    #[test]
    fn add_input_commits_and_escape_cancels_only_the_input() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        modal.handle_key(press(KeyCode::Char('a')));
        modal.handle_key(press(KeyCode::Char('x')));
        assert!(modal.input.is_some());
        assert!(modal.handle_key(press(KeyCode::Esc)).is_none());
        assert!(modal.input.is_none());

        modal.handle_key(press(KeyCode::Char('a')));
        for ch in "new-model".chars() {
            modal.handle_key(press(KeyCode::Char(ch)));
        }
        assert!(matches!(
            modal.handle_key(press(KeyCode::Enter)),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Add { model }
            ))) if model == "new-model"
        ));
    }

    #[test]
    fn remove_last_refusal_is_surfaced_inline() {
        let pool = view(&["only-architect"], true);
        let mut modal = DeepTierModal::new(pool.clone());
        assert!(matches!(
            modal.handle_key(press(KeyCode::Char('d'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Remove { target }
            ))) if target == "1"
        ));
        modal.apply_update(
            Some(pool),
            Err("Cannot remove the last deep-tier model".to_string()),
        );
        let rendered = dump(&modal, 96, 12);
        assert!(rendered.contains("Cannot remove the last deep-tier model"), "{rendered}");
    }

    #[test]
    fn uppercase_jk_emit_ordered_move_actions() {
        let mut modal = DeepTierModal::new(view(&["a", "b", "c"], true));
        modal.handle_key(press(KeyCode::Down));
        assert!(matches!(
            modal.handle_key(shifted(KeyCode::Char('K'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Move { from: 2, to: 1 }
            )))
        ));
        assert!(matches!(
            modal.handle_key(shifted(KeyCode::Char('J'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Move { from: 2, to: 3 }
            )))
        ));
    }

    #[test]
    fn reset_requires_inline_confirmation() {
        let mut modal = DeepTierModal::new(view(&["architect-a"], true));
        assert!(modal.handle_key(press(KeyCode::Char('r'))).is_none());
        assert!(modal.confirming_reset);
        assert!(modal.handle_key(press(KeyCode::Char('n'))).is_none());
        assert!(!modal.confirming_reset);

        modal.handle_key(press(KeyCode::Char('r')));
        assert!(matches!(
            modal.handle_key(press(KeyCode::Char('y'))),
            Some(ModalResult::Selected(ModalSelection::DeepTier(
                DeepTierAction::Reset
            )))
        ));
    }

    #[test]
    fn render_dump_shows_named_pool_source_selection_and_keys() {
        let modal = DeepTierModal::new(view(
            &["claude-architect", "gpt-verifier", "gemini-reviewer"],
            true,
        ));
        let rendered = dump(&modal, 104, 14);
        for expected in [
            "Deep-tier models",
            "1. claude-architect (configured)",
            "2. gpt-verifier (configured)",
            "3. gemini-reviewer (configured)",
            "↑↓/j k",
            "d/Del",
            "K/J",
            "changes apply from the next turn",
        ] {
            assert!(rendered.contains(expected), "missing {expected:?} in:\n{rendered}");
        }
    }
}
