//! `ToolResultBody::Read` variant — file-read result cards: a path header
//! plus line-numbered content.
//!
//! P10-B registry shape: every Read-specific rendering concern lives here;
//! `mod.rs` keeps only the exhaustive dispatch arms.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::theme::Theme;

use super::super::compact_path_label;
use super::{
    body_line_cap, count_display_lines, display_text_for_result, push_clip_notice, trunc_suffix,
};

/// Summary-line text: the compact path, the display-line count, and a
/// `(clipped)` suffix when the read was cut. Shares [`trunc_suffix`] so the whole
/// card speaks one hard-clip vocabulary instead of a private "(truncated)".
pub(super) fn summary(path: &str, content: &str, truncated: bool) -> String {
    let path = compact_path_label(path);
    format!(
        "{} · {} lines{}",
        path,
        count_display_lines(content),
        trunc_suffix(truncated)
    )
}

/// Collapsed-group digest: the display-line count as `N ln`.
pub(super) fn digest(content: &str) -> String {
    format!("{} ln", count_display_lines(content))
}

/// Expanded body: a cyan path header (with an optional language tag) then the
/// content with 1-based line numbers, capped with the shared clip notice.
pub(super) fn body_lines<'a>(
    path: &'a str,
    content: &'a str,
    language: Option<&'a str>,
    theme: &Theme,
    expanded: bool,
) -> Vec<Line<'a>> {
    let mut lines: Vec<Line<'_>> = Vec::new();
    let path = compact_path_label(path);
    let mut header = vec![Span::styled(
        path,
        Style::new()
            .fg(theme.palette.cyan)
            .add_modifier(Modifier::BOLD),
    )];
    if let Some(language) = language {
        header.push(Span::raw("  "));
        header.push(Span::styled(
            language.to_string(),
            Style::new().fg(theme.palette.dim),
        ));
    }
    lines.push(Line::from(header));
    let display = display_text_for_result(content);
    let cap = body_line_cap(expanded, 24);
    let total = display.lines().count();
    for (idx, line) in display.lines().take(cap).enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:>4} ", idx + 1),
                Style::new().fg(theme.palette.dim),
            ),
            Span::styled(line.to_string(), Style::new().fg(theme.palette.fg)),
        ]));
    }
    push_clip_notice(&mut lines, expanded, total, cap, theme);
    lines
}
