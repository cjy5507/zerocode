//! Runtime tool toggle modal for `/tools`.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

use super::super::theme::Theme;
use super::{ModalResult, ModalSelection};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolToggleRow {
    pub name: String,
    pub description: Option<String>,
    pub source: String,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ToolToggleModal {
    rows: Vec<ToolToggleRow>,
    cursor: usize,
    scroll: usize,
}

impl ToolToggleModal {
    #[must_use]
    pub fn new(mut rows: Vec<ToolToggleRow>) -> Self {
        rows.sort_by(|left, right| {
            left.source
                .cmp(&right.source)
                .then_with(|| left.name.cmp(&right.name))
        });
        Self {
            rows,
            cursor: 0,
            scroll: 0,
        }
    }

    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    #[must_use]
    pub fn rows(&self) -> &[ToolToggleRow] {
        &self.rows
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(ModalResult::Cancelled),
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
            KeyCode::Enter | KeyCode::Char(' ') => self.toggle_current(),
            _ => None,
        }
    }

    fn move_up(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
        self.ensure_visible(1);
    }

    fn move_down(&mut self) {
        if self.cursor + 1 < self.rows.len() {
            self.cursor += 1;
        }
        self.ensure_visible(1);
    }

    fn page_up(&mut self) {
        self.cursor = self.cursor.saturating_sub(8);
        self.ensure_visible(1);
    }

    fn page_down(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        self.cursor = (self.cursor + 8).min(self.rows.len() - 1);
        self.ensure_visible(1);
    }

    fn toggle_current(&mut self) -> Option<ModalResult> {
        let row = self.rows.get_mut(self.cursor)?;
        row.enabled = !row.enabled;
        Some(ModalResult::Selected(ModalSelection::ToolToggle {
            name: row.name.clone(),
            enabled: row.enabled,
        }))
    }

    fn ensure_visible(&mut self, visible_rows: usize) {
        if self.cursor < self.scroll {
            self.scroll = self.cursor;
        } else if visible_rows > 0 && self.cursor >= self.scroll + visible_rows {
            self.scroll = self.cursor + 1 - visible_rows;
        }
    }

    #[must_use]
    pub fn render_lines<'a>(&'a self, theme: &Theme, height: usize, width: usize) -> Vec<Line<'a>> {
        if self.rows.is_empty() {
            return vec![Line::from(Span::styled(
                "No toggleable tools are currently registered.",
                theme.typography.dim,
            ))];
        }

        let source_width = self
            .rows
            .iter()
            .map(|row| row.source.width())
            .max()
            .unwrap_or(3)
            .min(12);
        let visible_rows = height.max(1);
        let scroll = self.scroll.min(self.rows.len().saturating_sub(1));
        let end = (scroll + visible_rows).min(self.rows.len());
        self.rows[scroll..end]
            .iter()
            .enumerate()
            .map(|(offset, row)| {
                let index = scroll + offset;
                let selected = index == self.cursor;
                // Pi select-list row: `→ ✓ [source] name  description`.
                let wash = if selected {
                    super::selection_wash(theme)
                } else {
                    Style::new()
                };
                let marker = if selected {
                    super::cursor_marker(!theme.no_color)
                } else {
                    super::blank_marker()
                };
                let state = if row.enabled {
                    Span::styled(
                        format!("{} ", super::current_marker(!theme.no_color)),
                        super::current_style(theme).patch(wash),
                    )
                } else {
                    Span::styled(
                        format!("{} ", if theme.no_color { "-" } else { "\u{00b7}" }),
                        theme.typography.dim.patch(wash),
                    )
                };
                let name_style = if selected {
                    super::selected_style(theme)
                } else if row.enabled {
                    theme.typography.body
                } else {
                    theme.typography.dim
                };
                let source = fit_cell_width(&row.source, source_width);
                let mut spans = vec![
                    Span::styled(
                        marker.to_string(),
                        if selected {
                            super::selected_style(theme)
                        } else {
                            theme.typography.dim
                        },
                    ),
                    state,
                    Span::styled(
                        format!("[{source}] "),
                        super::row_detail_style(theme, selected),
                    ),
                    Span::styled(row.name.clone(), name_style),
                ];
                let used: usize = spans.iter().map(|span| span.content.width()).sum();
                let available = width.saturating_sub(used + 3);
                if available > 8 {
                    if let Some(description) = &row.description {
                        if !description.trim().is_empty() {
                            spans.push(Span::styled(
                                format!("  {}", fit_cell_width(description, available)),
                                super::row_detail_style(theme, selected),
                            ));
                        }
                    }
                }
                Line::from(spans)
            })
            .collect()
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let enabled = self.rows.iter().filter(|row| row.enabled).count();
        let title = format!("/tools {enabled}/{}", self.rows.len());
        let inner = super::modal_frame(frame, area, title, theme);
        let height = usize::from(inner.height.saturating_sub(2));
        let width = usize::from(inner.width);
        let selected_row = self.cursor.saturating_sub(self.scroll);
        let mut lines = self
            .render_lines(theme, height, width)
            .into_iter()
            .enumerate()
            .map(|(idx, line)| {
                if idx == selected_row && !self.rows.is_empty() {
                    super::wash_row(line, inner.width, theme)
                } else {
                    line
                }
            })
            .collect::<Vec<_>>();
        if inner.height > 1 {
            lines.push(Line::default());
        }
        if inner.height > 0 {
            // Fitted to `inner.width`: this `Paragraph` has no `Wrap`, so an
            // over-wide footer would be cut mid-word at the rect edge rather than
            // losing a whole hint (see `super::modal_footer_fitted`).
            lines.push(super::key_hint_footer_fitted(
                theme,
                &[("Space/Enter", "토글"), ("Esc", "닫기")],
                inner.width,
            ));
        }
        // Fitted: this body renders without `wrap`, so an over-wide row would be
        // cut mid-glyph by `LineTruncator` instead of losing its tail visibly.
        frame.render_widget(
            Paragraph::new(super::fit_body_rows(lines, inner.width))
                .style(theme.typography.body),
            inner,
        );
    }
}

fn fit_cell_width(value: &str, width: usize) -> String {
    if value.width() <= width {
        return value.to_string();
    }
    if width == 0 {
        return String::new();
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let mut out = String::new();
    let mut used = 0;
    for ch in value.chars() {
        let char_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + char_width + 3 > width {
            break;
        }
        out.push(ch);
        used += char_width;
    }
    out.push_str("...");
    out
}
