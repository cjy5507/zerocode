//! Generic single-select choice modal (Phase 3, Lane L6).
//!
//! Used for arbitrary in-app yes/no or multi-choice prompts. See
//! `.zo/design/components.md` §6 for the visual language.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, key_hint_footer, selected_style,
};

/// Cursor rows a PageUp/PageDown jumps through. Matches the page stride used by
/// the other selection-list modals (see `tool_toggle::page_down`).
const PAGE_STRIDE: usize = 8;

/// Generic single-select list modal.
#[derive(Debug, Clone)]
pub struct ChoicePickerModal {
    title: String,
    options: Vec<String>,
    cursor: usize,
}

impl ChoicePickerModal {
    /// Construct a modal with `title` and a list of option labels.
    #[must_use]
    pub fn new(title: impl Into<String>, options: Vec<String>) -> Self {
        Self {
            title: title.into(),
            options,
            cursor: 0,
        }
    }

    /// Title displayed in the modal border.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Current cursor index.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Number of options.
    #[must_use]
    pub fn len(&self) -> usize {
        self.options.len()
    }

    /// `true` if there are no options.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// Move cursor down by one, wrapping.
    pub fn move_down(&mut self) {
        if self.options.is_empty() {
            return;
        }
        self.cursor = (self.cursor + 1) % self.options.len();
    }

    /// Move cursor up by one, wrapping.
    pub fn move_up(&mut self) {
        if self.options.is_empty() {
            return;
        }
        if self.cursor == 0 {
            self.cursor = self.options.len() - 1;
        } else {
            self.cursor -= 1;
        }
    }

    /// Move the cursor down by a page, clamping at the last option.
    pub fn page_down(&mut self) {
        if self.options.is_empty() {
            return;
        }
        self.cursor = (self.cursor + PAGE_STRIDE).min(self.options.len() - 1);
    }

    /// Move the cursor up by a page, clamping at the first option.
    pub fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(PAGE_STRIDE);
    }

    /// Jump the cursor to the first option.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Jump the cursor to the last option.
    pub fn move_end(&mut self) {
        self.cursor = self.options.len().saturating_sub(1);
    }

    /// Move the cursor down by `count` rows, clamping at the end. Used by the
    /// host's mouse-wheel routing (which owns the app-level dispatch).
    pub fn scroll_down(&mut self, count: usize) {
        if self.options.is_empty() {
            return;
        }
        self.cursor = (self.cursor + count).min(self.options.len() - 1);
    }

    /// Move the cursor up by `count` rows, clamping at the top. Used by the
    /// host's mouse-wheel routing.
    pub fn scroll_up(&mut self, count: usize) {
        self.cursor = self.cursor.saturating_sub(count);
    }

    /// Handle a single key event.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc => Some(ModalResult::Cancelled),
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
                None
            }
            KeyCode::PageUp => {
                self.page_up();
                None
            }
            KeyCode::PageDown => {
                self.page_down();
                None
            }
            KeyCode::Home => {
                self.move_home();
                None
            }
            KeyCode::End => {
                self.move_end();
                None
            }
            KeyCode::Enter => {
                if self.options.is_empty() {
                    return Some(ModalResult::Cancelled);
                }
                let label = self.options[self.cursor].clone();
                Some(ModalResult::Selected(ModalSelection::Choice {
                    index: self.cursor,
                    label,
                }))
            }
            _ => None,
        }
    }

    /// Build the rendered lines used by both [`Self::draw`] and tests.
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme) -> Vec<Line<'a>> {
        let mut lines: Vec<Line<'a>> = self
            .options
            .iter()
            .enumerate()
            .map(|(idx, label)| {
                let selected = idx == self.cursor;
                let marker = if selected {
                    cursor_marker(!theme.no_color)
                } else {
                    blank_marker()
                };
                let text = format!("{marker}{label}");
                let style = if selected {
                    selected_style(theme)
                } else {
                    theme.typography.body
                };
                Line::from(Span::styled(text, style))
            })
            .collect();
        lines.push(Line::from(""));
        lines.push(key_hint_footer(
            theme,
            &[("↑↓", "move"), ("Enter", "confirm"), ("Esc", "cancel")],
        ));
        lines
    }

    /// Draw the modal into `area` using `theme`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = super::modal_frame(frame, area, self.title.clone(), theme);
        let lines = self.render_lines(theme);
        let paragraph = Paragraph::new(lines).style(theme.typography.body);
        frame.render_widget(paragraph, inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEventState, KeyModifiers};

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn modal_with(count: usize) -> ChoicePickerModal {
        let options: Vec<String> = (0..count).map(|i| format!("option {i}")).collect();
        ChoicePickerModal::new("pick", options)
    }

    #[test]
    fn page_down_advances_by_a_page_and_clamps() {
        let mut modal = modal_with(20);
        assert_eq!(modal.cursor(), 0);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), PAGE_STRIDE);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), PAGE_STRIDE * 2);
        modal.handle_key(press(KeyCode::PageDown));
        assert_eq!(modal.cursor(), 19, "PageDown clamps at the last option");
    }

    #[test]
    fn home_and_end_jump_to_bounds() {
        let mut modal = modal_with(10);
        modal.handle_key(press(KeyCode::End));
        assert_eq!(modal.cursor(), 9);
        modal.handle_key(press(KeyCode::PageUp));
        assert_eq!(modal.cursor(), 9 - PAGE_STRIDE);
        modal.handle_key(press(KeyCode::Home));
        assert_eq!(modal.cursor(), 0);
    }

    /// Flatten the modal's rendered rows into one string for glyph inspection.
    fn rendered_text(modal: &ChoicePickerModal, theme: &Theme) -> String {
        modal
            .render_lines(theme)
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect()
    }

    /// Normal/color mode keeps the Unicode selection chevron `❯`; the ASCII
    /// fallback never leaks into the rich render.
    #[test]
    fn selection_cursor_keeps_unicode_in_color_mode() {
        let modal = modal_with(3);
        let text = rendered_text(&modal, &Theme::zo());
        assert!(text.contains('\u{276f}'), "color mode keeps the ❯ cursor: {text:?}");
    }

    /// Plain/`NO_COLOR` mode swaps the chevron for a one-cell ASCII `>` and
    /// never emits the Unicode chevron.
    #[test]
    fn selection_cursor_uses_ascii_fallback_under_no_color() {
        let modal = modal_with(3);
        let text = rendered_text(&modal, &Theme::no_color());
        assert!(text.contains('>'), "plain cursor is one-cell '>': {text:?}");
        assert!(
            !text.contains('\u{276f}'),
            "no Unicode chevron under NO_COLOR: {text:?}"
        );
    }
}
