//! `RenderBlock::UserNotice` widget — a "to you" panel for a `send_to_user`
//! push.
//!
//! Framed distinctly from a muted [`RenderBlock::System`] status line: this is
//! verbatim content the model wanted the user to read mid-run, so it carries a
//! `✦  to you` header and an info-tinted left rail (`┃`) down the body. The
//! body flows through the same markdown engine as prose/system blocks, so a
//! pushed diff or list renders — never truncated (any size cap is applied at
//! the tool boundary).
//!
//! See `code-rules.md` R2 (no ANSI), R9 (`&Theme` styling).

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::tui::glyphs;
use crate::tui::theme::Theme;

use super::{wrapped_rows, ROLE_RAIL_WIDTH};

/// Render a `send_to_user` notice panel.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    message: &str,
    theme: &Theme,
    scroll_offset: u16,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let para = Paragraph::new(rendered_lines(message, theme, area.width))
        .style(theme.typography.body)
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));
    frame.render_widget(para, area);
}

pub(crate) fn estimate_rows(message: &str, theme: &Theme, width: u16) -> u16 {
    wrapped_rows(&rendered_lines(message, theme, width), width)
}

/// Header + rail-prefixed markdown body. Mirrors the user-message rail layout
/// but tints the rail/header with the info palette so the panel reads as a
/// system-authored push *to* the user, not an amber user paste.
fn rendered_lines(message: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    let inner_width = width.saturating_sub(ROLE_RAIL_WIDTH);
    let body = if message.is_empty() {
        vec![Line::from("")]
    } else {
        crate::tui::markdown::rendered_lines_for_width(message, theme, inner_width)
    };

    let rail = format!("{}  ", rail_glyph(theme));
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(body.len() + 1);
    lines.push(header_line(theme));
    for line in body {
        let mut spans = Vec::with_capacity(line.spans.len() + 1);
        spans.push(Span::styled(rail.clone(), rail_style(theme)));
        spans.extend(line.spans);
        lines.push(Line::from(spans));
    }
    lines
}

/// `✦  to you` — the panel header.
fn header_line(theme: &Theme) -> Line<'static> {
    Line::from(Span::styled(
        format!("{}  to you", header_glyph(theme)),
        rail_style(theme),
    ))
}

/// Header spark glyph (`✦`, or `+` under `NO_COLOR`).
fn header_glyph(theme: &Theme) -> &'static str {
    if theme.no_color {
        glyphs::ZO_SPARK_NC
    } else {
        glyphs::ZO_SPARK
    }
}

/// Body rail glyph (`┃`, or `|` under `NO_COLOR`).
fn rail_glyph(theme: &Theme) -> &'static str {
    if theme.no_color {
        glyphs::ZO_RAIL_NC
    } else {
        glyphs::ZO_RAIL
    }
}

/// Info-tinted bold style for the rail and header.
fn rail_style(theme: &Theme) -> Style {
    Style::new()
        .fg(theme.palette.info)
        .add_modifier(Modifier::BOLD)
}
