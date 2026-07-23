//! `ToolResultBody::Bash` variant — `bash` command stdout / stderr result.
//!
//! P10-B registry shape: every Bash-specific rendering concern lives here;
//! `mod.rs` keeps only the exhaustive dispatch arms.

use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use runtime::message_stream::BashResult;

use crate::tui::theme::Theme;

use super::super::sanitize_inline;
use super::{body_line_cap, count_lines, display_text_for_result, push_clip_notice, trunc_suffix};

/// Summary-line text: the first non-empty stdout line, else `stderr: …`,
/// else `no output` (each with the truncation suffix).
pub(super) fn summary(result: &BashResult) -> String {
    let stdout = display_text_for_result(&result.stdout);
    let stderr = display_text_for_result(&result.stderr);
    let stdout_preview = stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(sanitize_inline);
    let stderr_preview = stderr
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(sanitize_inline);

    match (stdout_preview, stderr_preview) {
        (Some(stdout), _) => format!("{stdout}{}", trunc_suffix(result.truncated)),
        (None, Some(stderr)) => format!("stderr: {stderr}{}", trunc_suffix(result.truncated)),
        (None, None) => format!("no output{}", trunc_suffix(result.truncated)),
    }
}

/// Collapsed-group digest: `exit N` for a non-zero exit, quiet on success
/// (the call row's marker already signals success).
pub(super) fn digest(exit_code: i32) -> String {
    if exit_code != 0 {
        format!("exit {exit_code}")
    } else {
        String::new()
    }
}

/// Expanded body: labelled `stdout (N)` / `stderr (N)` sections, each capped
/// with the shared clip notice; `no output` when both streams are empty.
pub(super) fn body_lines<'a>(b: &'a BashResult, theme: &Theme, expanded: bool) -> Vec<Line<'a>> {
    let mut lines: Vec<Line<'_>> = Vec::new();
    let stdout = display_text_for_result(&b.stdout);
    let stderr = display_text_for_result(&b.stderr);
    push_stream_lines(&mut lines, "stdout", stdout.as_ref(), theme, false, expanded);
    push_stream_lines(&mut lines, "stderr", stderr.as_ref(), theme, true, expanded);
    if lines.is_empty() {
        lines.push(Line::styled(
            "no output",
            Style::new().fg(theme.palette.dim),
        ));
    }
    lines
}

fn push_stream_lines(
    lines: &mut Vec<Line<'_>>,
    label: &str,
    text: &str,
    theme: &Theme,
    is_error: bool,
    expanded: bool,
) {
    let count = count_lines(text);
    if count == 0 {
        return;
    }
    lines.push(Line::styled(
        format!("{label} ({count})"),
        Style::new()
            .fg(theme.palette.dim)
            .add_modifier(Modifier::BOLD),
    ));
    let style = if is_error {
        Style::new().fg(theme.palette.error)
    } else {
        Style::new().fg(theme.palette.fg)
    };
    let cap = body_line_cap(expanded, 12);
    for line in text.lines().take(cap) {
        lines.push(Line::styled(format!("  {line}"), style));
    }
    push_clip_notice(lines, expanded, count, cap, theme);
}
