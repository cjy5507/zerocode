//! `ToolResultBody::Text` and plain-`Generic` variants — free-form text and
//! JSON-envelope result cards, plus the `AskUserQuestion` result summary.
//!
//! P10-B registry shape: every Text / plain-Generic / question rendering
//! concern lives here; `mod.rs` keeps only the exhaustive dispatch arms. The
//! `web_fetch` / `web_search` Generic concerns stay in `mod.rs` for `web.rs`.

use ratatui::style::Style;
use ratatui::text::Line;

use crate::tui::theme::Theme;

use super::super::sanitize_inline;
use super::{body_line_cap, display_text_for_result, push_clip_notice, summarize_json, trunc_suffix};

/// Summary-line text for a Text / plain-Generic body: a recognised JSON
/// envelope digest, else the first non-empty line (`structured output` for a
/// bare or truncated brace).
pub(super) fn summary(content: &str, truncated: bool) -> String {
    let trimmed_original = content.trim();

    // Recognize common JSON envelopes so users see "ok" / status names /
    // key counts instead of raw "{ \"ok\": true, ... }" fragments
    // bleeding into the summary line (UX regression observed at L8).
    if !trimmed_original.is_empty()
        && (trimmed_original.starts_with('{') || trimmed_original.starts_with('['))
    {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed_original) {
            return format!("{}{}", summarize_json(&value), trunc_suffix(truncated));
        }
    }

    let display = display_text_for_result(content);
    let preview = display
        .lines()
        .find(|line| !line.trim().is_empty())
        .map_or_else(|| "(empty)".to_string(), sanitize_inline);
    let preview_trim = preview.trim_start();

    // A first-line of just `{` or `[` (the opening brace of a pretty-printed
    // JSON value that failed the parse above, e.g. mid-stream truncation)
    // is even worse to surface than "structured output".
    if preview_trim == "{" || preview_trim == "[" {
        return format!("structured output{}", trunc_suffix(truncated));
    }
    if truncated && (preview_trim.starts_with('{') || preview_trim.starts_with('[')) {
        return format!("structured output{}", trunc_suffix(truncated));
    }
    format!("{preview}{}", trunc_suffix(truncated))
}

/// Summary-line text for an `AskUserQuestion` result: `answered · …` /
/// `not answered · reason` / a dismissal phrase, parsed from the JSON status
/// (falling back to a plain preview).
pub(super) fn question_summary(content: &str, truncated: bool) -> String {
    let display = display_text_for_result(content);
    let trimmed = display.trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if value
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("answered"))
        {
            let answer = value
                .get("answer")
                .and_then(serde_json::Value::as_str)
                .map(sanitize_inline)
                .unwrap_or_default();
            return if answer.is_empty() {
                format!("answered{}", trunc_suffix(truncated))
            } else {
                format!("answered · {answer}{}", trunc_suffix(truncated))
            };
        }

        if value
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("unanswered"))
        {
            let reason = value
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .map(|reason| format!(" · {}", sanitize_inline(reason)))
                .unwrap_or_default();
            return format!("not answered{reason}{}", trunc_suffix(truncated));
        }
    }

    let lower = trimmed.to_ascii_lowercase();
    let summary = if lower.contains("dismissed") || lower.contains("without an answer") {
        "dismissed before answer".to_string()
    } else if lower.contains("channel closed") {
        "question channel closed".to_string()
    } else if lower.contains("parse") {
        "invalid question payload".to_string()
    } else if trimmed.is_empty() {
        "not answered".to_string()
    } else {
        summary(trimmed, false)
    };
    format!("{summary}{}", trunc_suffix(truncated))
}

/// Render free-form text result content. Strips ANSI escape sequences
/// so SGR bytes never reach the terminal and pretty-prints oneline JSON
/// payloads (e.g. `{"ok":true,...}`) so they line-wrap legibly instead
/// of bleeding into one long row.
pub(super) fn body_lines(content: &str, theme: &Theme, expanded: bool) -> Vec<Line<'static>> {
    let pretty_original = prettify_if_json(content);
    let display = display_text_for_result(content);
    let pretty_display = pretty_original
        .as_deref()
        .is_none()
        .then(|| prettify_if_json(display.as_ref()))
        .flatten();
    let source: &str = pretty_original
        .as_deref()
        .or(pretty_display.as_deref())
        .unwrap_or(display.as_ref());
    let style = Style::new().fg(theme.palette.fg);
    let cap = body_line_cap(expanded, 24);
    let total = source.lines().count();
    let mut lines: Vec<Line<'static>> = source
        .lines()
        .take(cap)
        .map(|l| Line::styled(sanitize_inline(l), style))
        .collect();
    push_clip_notice(&mut lines, expanded, total, cap, theme);
    lines
}

/// If `content` is a syntactically valid JSON object or array on a
/// single line (heuristic: starts with `{`/`[` and contains no newline),
/// return a pretty-printed version. Returns `None` otherwise so the
/// caller can render the original content as-is.
fn prettify_if_json(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
        return None;
    }
    if content.contains('\n') {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    serde_json::to_string_pretty(&value).ok()
}
