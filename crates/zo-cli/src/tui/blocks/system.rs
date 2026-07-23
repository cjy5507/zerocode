//! `RenderBlock::System` widget — borderless muted single-line notice.
//!
//! Rendered as a left-aligned CLI status row so it reads like the
//! reference console transcript rather than a centered banner.
//!
//! See `code-rules.md` R2 (no ANSI), R9 (`&Theme` styling).

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use runtime::message_stream::SystemLevel;

use crate::tui::glyphs;
use crate::tui::markdown;
use crate::tui::theme::Theme;

use super::wrapped_rows;

/// Render a system banner.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    level: SystemLevel,
    text: &str,
    theme: &Theme,
    scroll_offset: u16,
) {
    frame.render_widget(
        Paragraph::new(rendered_lines(level, text, theme, area.width))
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0)),
        area,
    );
}

pub(crate) fn estimate_rows(level: SystemLevel, text: &str, theme: &Theme, width: u16) -> u16 {
    wrapped_rows(&rendered_lines(level, text, theme, width), width)
}

fn has_any_markdown_signal(text: &str) -> bool {
    if markdown::has_strong_markdown_signal(text) {
        return true;
    }
    for raw in text.lines() {
        let line = raw.trim_start();
        if line.starts_with("- ") || line.starts_with("* ") || line.starts_with("+ ") {
            return true;
        }
        if line.contains("**") || line.contains('`') {
            return true;
        }
    }
    false
}

pub(crate) fn rendered_lines(
    level: SystemLevel,
    text: &str,
    theme: &Theme,
    width: u16,
) -> Vec<Line<'static>> {
    let nc = theme.no_color;
    let (glyph_color, glyph, text_style) = match level {
        SystemLevel::Info => (
            theme.palette.info,
            if nc {
                glyphs::INFO_CIRCLE_NC
            } else {
                glyphs::INFO_CIRCLE
            },
            Style::new().fg(theme.palette.dim),
        ),
        SystemLevel::Warn => (
            theme.palette.warn,
            if nc {
                glyphs::WARN_TRIANGLE_NC
            } else {
                glyphs::WARN_TRIANGLE
            },
            Style::new()
                .fg(theme.palette.warn)
                .add_modifier(Modifier::BOLD),
        ),
        SystemLevel::Error => (
            theme.palette.error,
            if nc { glyphs::CROSS_NC } else { glyphs::CROSS },
            Style::new()
                .fg(theme.palette.error)
                .add_modifier(Modifier::BOLD),
        ),
    };
    let glyph_style = Style::new().fg(glyph_color);
    let mut lines = Vec::new();

    if has_any_markdown_signal(text) {
        // The level glyph takes 2 cells: e.g. "⚠ " (or "  " on continuation lines).
        let width_for_markdown = width.saturating_sub(2).max(10);
        let md_lines = markdown::rendered_lines_for_width(text, theme, width_for_markdown);

        for (idx, md_line) in md_lines.into_iter().enumerate() {
            let mut spans = Vec::with_capacity(2 + md_line.spans.len());
            if idx == 0 {
                spans.push(Span::styled(format!("{glyph} "), glyph_style));
            } else {
                spans.push(Span::raw("  "));
            }

            // Inherit markdown spans but merge/patch fallback system style
            for mut span in md_line.spans {
                if level == SystemLevel::Warn {
                    span.style = span.style.patch(
                        Style::new()
                            .fg(theme.palette.warn)
                            .add_modifier(Modifier::BOLD),
                    );
                } else if level == SystemLevel::Error {
                    span.style = span.style.patch(
                        Style::new()
                            .fg(theme.palette.error)
                            .add_modifier(Modifier::BOLD),
                    );
                } else if span.style == Style::default() {
                    span.style = text_style;
                }
                spans.push(span);
            }
            lines.push(Line::from(spans));
        }
    } else {
        // Simple plain-text split (preserves exact manual layout / spaces)
        let source_lines: Vec<&str> = if text.is_empty() {
            vec![""]
        } else {
            text.lines().collect()
        };

        for (idx, body) in source_lines.into_iter().enumerate() {
            let mut spans = Vec::with_capacity(2);
            if idx == 0 {
                spans.push(Span::styled(format!("{glyph} "), glyph_style));
            } else {
                spans.push(Span::raw("  "));
            }
            spans.push(Span::styled(body.to_string(), text_style));
            lines.push(Line::from(spans));
        }
    }

    lines
}
