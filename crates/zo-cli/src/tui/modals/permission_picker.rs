//! `/permissions` picker modal (Phase 3, Lane L6).
//!
//! Mirrors the CLI's `session.rs::prompt_permissions_picker` semantics
//! as an in-app ratatui modal per `.zo/design/components.md` §6.2.
//! Returns a [`runtime::PermissionMode`] via [`ModalSelection::Permission`].

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use runtime::PermissionMode;

use super::super::theme::Theme;
use super::{
    ModalResult, ModalSelection, blank_marker, cursor_marker, key_hint_footer_fitted, selected_style,
};

/// Ordered list of permission modes offered by the modal.
///
/// Follows the same order the CLI's `session.rs` picker uses today.
pub const PERMISSION_ORDER: [PermissionMode; 5] = [
    PermissionMode::ReadOnly,
    PermissionMode::WorkspaceWrite,
    PermissionMode::Prompt,
    PermissionMode::Allow,
    PermissionMode::DangerFullAccess,
];

/// In-app permission picker modal.
#[derive(Debug, Clone)]
pub struct PermissionPickerModal {
    cursor: usize,
}

impl Default for PermissionPickerModal {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionPickerModal {
    /// Construct a modal pre-positioned on the first entry.
    #[must_use]
    pub const fn new() -> Self {
        Self { cursor: 0 }
    }

    /// Build a modal pre-positioned on `mode` if it appears in the
    /// canonical order; otherwise position on the first entry.
    #[must_use]
    pub fn with_selected(mode: PermissionMode) -> Self {
        let cursor = PERMISSION_ORDER
            .iter()
            .position(|m| *m == mode)
            .unwrap_or(0);
        Self { cursor }
    }

    /// Current cursor index.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Currently highlighted mode.
    #[must_use]
    pub fn current(&self) -> PermissionMode {
        PERMISSION_ORDER[self.cursor]
    }

    /// Move cursor down by one, wrapping.
    pub fn move_down(&mut self) {
        self.cursor = (self.cursor + 1) % PERMISSION_ORDER.len();
    }

    /// Move cursor up by one, wrapping.
    pub fn move_up(&mut self) {
        if self.cursor == 0 {
            self.cursor = PERMISSION_ORDER.len() - 1;
        } else {
            self.cursor -= 1;
        }
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
            KeyCode::Enter => Some(ModalResult::Selected(ModalSelection::Permission(
                self.current(),
            ))),
            _ => None,
        }
    }

    /// Build the rendered lines used by both [`Self::draw`] and tests.
    ///
    /// `width` is the content width the rows will be painted into; the key-hint
    /// footer is fitted to it because `draw` renders without `Wrap` and would
    /// otherwise have the row cut mid-word at the rect edge. That matters more
    /// here than elsewhere: these labels are Hangul, so each is two columns wide.
    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme, width: u16) -> Vec<Line<'a>> {
        let mut lines: Vec<Line<'a>> = PERMISSION_ORDER
            .iter()
            .enumerate()
            .map(|(idx, mode)| {
                let selected = idx == self.cursor;
                let marker = if selected {
                    cursor_marker(!theme.no_color)
                } else {
                    blank_marker()
                };
                let label = format!("{marker}{}", mode.as_str());
                let style = if selected {
                    selected_style(theme)
                } else {
                    theme.typography.body
                };
                Line::from(Span::styled(label, style))
            })
            .collect();
        lines.push(Line::from(""));
        lines.push(key_hint_footer_fitted(
            theme,
            &[("↑↓", "이동"), ("Enter", "확정"), ("Esc", "취소")],
            width,
        ));
        lines
    }

    /// Draw the modal into `area` using `theme`.
    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = super::modal_frame(frame, area, "/permissions", theme);
        // The cursor row's wash runs to the column edge (Pi select-list idiom).
        let lines = self
            .render_lines(theme, inner.width)
            .into_iter()
            .enumerate()
            .map(|(idx, line)| {
                if idx == self.cursor {
                    super::wash_row(line, inner.width, theme)
                } else {
                    line
                }
            })
            .collect::<Vec<_>>();
        // Fitted: this body renders without `wrap`, so an over-wide row would be
        // cut mid-glyph by `LineTruncator` instead of losing its tail visibly.
        let paragraph = Paragraph::new(super::fit_body_rows(lines, inner.width))
            .style(theme.typography.body);
        frame.render_widget(paragraph, inner);
    }
}
