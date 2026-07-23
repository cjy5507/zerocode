//! `ToolResultBody::Listing` variant — `glob` / `grep` path-or-match lists.
//!
//! P10-B registry shape: every Listing-specific rendering concern lives
//! here; `mod.rs` keeps only the exhaustive dispatch arms.

use ratatui::style::Style;
use ratatui::text::Line;

use crate::tui::theme::Theme;

use super::super::sanitize_inline;
use super::{body_line_cap, push_clip_notice, trunc_suffix};

/// Summary-line text: entry tally plus the first entry as a taste
/// (`3 entries · src/main.rs`, `1+ entries · …`).
pub(super) fn summary(entries: &[String], truncated: bool) -> String {
    let count = entries.len();
    let count_label = match (count, truncated) {
        (0, _) => format!("0 entries{}", trunc_suffix(truncated)),
        (1, false) => "1 entry".to_string(),
        (1, true) => "1+ entries".to_string(),
        (_, false) => format!("{count} entries"),
        (_, true) => format!("{count}+ entries"),
    };
    entries.first().map_or(count_label.clone(), |entry| {
        format!("{count_label} · {}", sanitize_inline(entry))
    })
}

/// Collapsed-group digest: count with a tool-appropriate unit
/// (`12 hits` for grep, `4 files` for glob, `N items` otherwise; `+` marks
/// a truncated listing).
pub(super) fn digest(name: &str, entries: &[String], truncated: bool) -> String {
    let unit = match name {
        "grep_search" | "Grep" => "hits",
        "glob_search" | "Glob" => "files",
        _ => "items",
    };
    let plus = if truncated { "+" } else { "" };
    format!("{}{plus} {unit}", entries.len())
}

/// Expanded body: one row per entry up to the cap, then the clip notice.
pub(super) fn body_lines<'a>(
    entries: &'a [String],
    theme: &Theme,
    expanded: bool,
) -> Vec<Line<'a>> {
    let cap = body_line_cap(expanded, 24);
    let mut lines: Vec<Line<'a>> = entries
        .iter()
        .take(cap)
        .map(|e| Line::styled(sanitize_inline(e), Style::new().fg(theme.palette.fg)))
        .collect();
    push_clip_notice(&mut lines, expanded, entries.len(), cap, theme);
    lines
}
