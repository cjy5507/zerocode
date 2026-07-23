//! Generic slash-command report popup.
//!
//! One scrollable frame for every read-only report command (`/mcp`,
//! `/doctor`, `/status`, …): the dispatcher hands over the same block-shaped
//! content it used to print into the transcript, and this modal renders it
//! centered with a copy key instead. Nothing is recorded in the transcript —
//! a report is read-only and re-derivable by re-running its command — so the
//! conversation stays clean while every report command shares one popup
//! frame, one key map, and one scroll model.

use core_types::CardModel;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthStr;

use super::super::cards;
use super::super::theme::Theme;
use super::{ModalResult, ModalSelection};

/// Severity accent for a plain-text report block. Mirrors the dispatcher's
/// `SystemLevel` without importing the runtime type into the modal layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportTone {
    Info,
    Warn,
    Error,
}

/// One renderable unit of popup content — the popup-side mirror of the
/// dispatcher's `OutputBlock`, so report handlers keep producing the exact
/// content model they already build for the transcript path.
#[derive(Debug, Clone)]
pub enum ReportViewerBlock {
    /// Plain multi-line report text at the given severity.
    Text {
        /// Severity accent for the whole block.
        tone: ReportTone,
        /// Raw report body (newline-separated).
        body: String,
    },
    /// A structured rich card, rendered through the shared card renderer so
    /// the popup and the transcript can never drift in card styling.
    Card(CardModel),
}

/// Width used to pre-render cards for the copy payload and line count.
/// Cards truncate (never wrap), so row count is width-independent; a wide
/// canonical width just keeps the copied text untruncated.
const CANONICAL_RENDER_WIDTH: u16 = 400;

/// Rows consumed by the frame chrome (border + footer hint line).
const CHROME_ROWS: u16 = 3;

#[derive(Debug, Clone)]
pub struct ReportViewerModal {
    title: String,
    blocks: Vec<ReportViewerBlock>,
    scroll: usize,
    /// Total body lines, measured once at construction (card rows are
    /// width-independent), so key handling can clamp without a layout pass.
    line_count: usize,
    /// Widest body line in display cells, for content-sized centering.
    max_line_width: usize,
    /// Plain-text projection of the whole report for the copy key.
    copy_text: String,
}

impl ReportViewerModal {
    #[must_use]
    pub fn new(title: impl Into<String>, blocks: Vec<ReportViewerBlock>, theme: &Theme) -> Self {
        let title = title.into();
        let lines = compose_lines(&blocks, theme, CANONICAL_RENDER_WIDTH);
        let line_count = lines.len();
        let max_line_width = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.width())
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        let copy_text = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        Self {
            title,
            blocks,
            scroll: 0,
            line_count,
            max_line_width,
            copy_text,
        }
    }

    /// Popup title (also the copy toast label), exposed for tests.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Current scroll offset, exposed for tests.
    #[must_use]
    pub const fn scroll_offset(&self) -> usize {
        self.scroll
    }

    /// Plain-text projection handed to the clipboard on `c`.
    #[must_use]
    pub fn copy_text(&self) -> &str {
        &self.copy_text
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Option<ModalResult> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => Some(ModalResult::Cancelled),
            KeyCode::Char('c') => Some(ModalResult::Selected(ModalSelection::CopyText(
                self.copy_text.clone(),
            ))),
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll_by(-1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll_by(1);
                None
            }
            KeyCode::PageUp => {
                self.scroll_by(-8);
                None
            }
            KeyCode::PageDown => {
                self.scroll_by(8);
                None
            }
            KeyCode::Home | KeyCode::Char('g') => {
                self.scroll = 0;
                None
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.scroll = self.max_scroll();
                None
            }
            _ => None,
        }
    }

    pub fn scroll_wheel(&mut self, up: bool, rows: usize) {
        let delta = i64::try_from(rows.max(1)).unwrap_or(1);
        self.scroll_by(if up { -delta } else { delta });
    }

    fn scroll_by(&mut self, delta: i64) {
        let next = i64::try_from(self.scroll).unwrap_or(i64::MAX).saturating_add(delta);
        let clamped = next.clamp(0, i64::try_from(self.max_scroll()).unwrap_or(i64::MAX));
        self.scroll = usize::try_from(clamped).unwrap_or(0);
    }

    /// Upper scroll bound: keep at least a screenful's tail visible. The view
    /// height is not known at key time, so clamp against a conservative
    /// minimum body height; draw-time slicing tolerates any residual overshoot
    /// by rendering blank tail rows.
    fn max_scroll(&self) -> usize {
        self.line_count.saturating_sub(4)
    }

    /// Content-sized popup: wide enough for the widest line, tall enough for
    /// every body line plus chrome, clamped by the caller to the screen.
    #[must_use]
    pub fn desired_size(&self) -> (u16, u16) {
        let width = u16::try_from(self.max_line_width.saturating_add(4)).unwrap_or(u16::MAX);
        let height =
            u16::try_from(self.line_count).unwrap_or(u16::MAX).saturating_add(CHROME_ROWS);
        (width.clamp(44, 110), height.clamp(8, 40))
    }

    pub fn draw(&self, frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
        let inner = super::modal_frame(frame, area, format!(" {} ", self.title), theme);
        if inner.height == 0 || inner.width == 0 {
            return;
        }
        let body_height = usize::from(inner.height.saturating_sub(1));
        let lines = compose_lines(&self.blocks, theme, inner.width);
        let scroll = self.scroll.min(lines.len().saturating_sub(1));
        let mut visible: Vec<Line<'static>> =
            lines.into_iter().skip(scroll).take(body_height).collect();
        while visible.len() < body_height {
            visible.push(Line::default());
        }
        visible.push(super::key_hint_footer(
            theme,
            &[("↑↓", "scroll"), ("c", "copy"), ("Esc", "close")],
        ));
        frame.render_widget(Paragraph::new(visible).style(theme.typography.body), inner);
    }
}

/// Render every block into display lines at `width`. Cards go through the
/// shared transcript card renderer; text blocks split on newlines with a
/// severity accent. Blocks are separated by one blank line.
fn compose_lines(
    blocks: &[ReportViewerBlock],
    theme: &Theme,
    width: u16,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if index > 0 {
            out.push(Line::default());
        }
        match block {
            ReportViewerBlock::Text { tone, body } => {
                let style = tone_style(*tone, theme);
                for raw in body.lines() {
                    out.push(Line::from(Span::styled(raw.to_string(), style)));
                }
                if body.is_empty() {
                    out.push(Line::default());
                }
            }
            ReportViewerBlock::Card(card) => {
                out.extend(cards::render_lines(card, theme, width));
            }
        }
    }
    out
}

fn tone_style(tone: ReportTone, theme: &Theme) -> Style {
    match tone {
        ReportTone::Info => theme.typography.body,
        ReportTone::Warn => Style::new().fg(theme.palette.warn),
        ReportTone::Error => Style::new().fg(theme.palette.error),
    }
}
