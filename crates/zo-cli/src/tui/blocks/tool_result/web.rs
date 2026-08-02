//! `web_fetch` / `web_search` Generic results — tailored summaries, the
//! collapsed-group digest, and the inline markdown page body.
//!
//! P10-B registry shape: every web-tool rendering concern lives here; `mod.rs`
//! keeps only the exhaustive dispatch arms (predicate guards + one-line
//! delegations).

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::tui::theme::Theme;

use super::super::sanitize_inline;
use super::{body_line_cap, display_text_for_result, push_clip_notice, summarize_json, trunc_suffix};

/// Web-search tools render through the generic envelope (see
/// `format_tool_result`'s catch-all), so without a tailored summary the line
/// degrades to a raw URL list or "N fields (…)". grep/glob "search" tools
/// format as `Listing`, never `Generic`, so a bare "search" substring reaching
/// the generic arm always means a web / semantic search.
pub(super) fn is_search_tool(name: &str) -> bool {
    name.to_ascii_lowercase().contains("search")
}

/// File reads/opens format as `Read`, never `Generic`, so "fetch" reaching the
/// generic arm always means a web / MCP fetch.
pub(super) fn is_fetch_tool(name: &str) -> bool {
    name.to_ascii_lowercase().contains("fetch")
}

/// `N results` for a web search, recognising both the builtin Markdown bullet
/// list and an MCP JSON `{results: [...]}` envelope.
pub(super) fn search_summary(content: &str, truncated: bool) -> String {
    let trimmed = content.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(n) = json_hit_count(&value) {
                return format!("{n} result{}{}", plural_s(n), trunc_suffix(truncated));
            }
            return format!("{}{}", summarize_json(&value), trunc_suffix(truncated));
        }
    }
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.is_empty() || lower.contains("no web search results") || lower.contains("no results")
    {
        return format!("0 results{}", trunc_suffix(truncated));
    }
    let hits = trimmed
        .lines()
        .filter(|line| line.trim_start().starts_with("- "))
        .count();
    if hits > 0 {
        return format!("{hits} result{}{}", plural_s(hits), trunc_suffix(truncated));
    }
    super::text::summary(content, truncated)
}

/// `fetched host · 12KB · title found` for a web fetch, recognising both the
/// builtin `Fetched {url}\nTitle: …` text and an MCP JSON `{bytes, title, …}`
/// envelope. Replaces the cryptic `N fields (bytes, …)` the generic path emits.
pub(super) fn fetch_summary(content: &str, truncated: bool) -> String {
    let trimmed = content.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let serde_json::Value::Object(map) = &value {
                let mut parts = Vec::new();
                if let Some(bytes) = ["bytes", "byteLength", "size", "contentLength"]
                    .iter()
                    .find_map(|key| map.get(*key).and_then(serde_json::Value::as_u64))
                {
                    parts.push(human_size(bytes));
                }
                if map
                    .get("title")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|title| !title.trim().is_empty())
                {
                    parts.push("title found".to_string());
                }
                if !parts.is_empty() {
                    return format!("{}{}", parts.join(" \u{00b7} "), trunc_suffix(truncated));
                }
            }
            return format!("{}{}", summarize_json(&value), trunc_suffix(truncated));
        }
    }
    if let Some(rest) = trimmed.strip_prefix("Fetched ") {
        let host = url_host(rest.lines().next().unwrap_or("").trim());
        let title = if trimmed.contains("Title:") {
            " \u{00b7} title found"
        } else {
            ""
        };
        return if host.is_empty() {
            format!("fetched{title}{}", trunc_suffix(truncated))
        } else {
            format!("fetched {host}{title}{}", trunc_suffix(truncated))
        };
    }
    super::text::summary(content, truncated)
}

/// Collapsed-group digest for a web search: `N results` / `N+ results` from the
/// JSON hit count, or empty when the payload carries no recognisable count.
pub(super) fn digest(content: &str, truncated: bool) -> String {
    serde_json::from_str::<serde_json::Value>(content.trim())
        .ok()
        .and_then(|v| json_hit_count(&v))
        .map(|n| format!("{n}{} results", if truncated { "+" } else { "" }))
        .unwrap_or_default()
}

/// Render a `web_fetch` / `web_search` result body: leading metadata lines
/// become cyan labels and the remaining page content flows through the
/// markdown engine (headings, emphasis, links, lists) at `inner_width`, so it
/// reads like the source page rather than one flat monochrome block. Capped
/// like any other body so a large page cannot flood the transcript.
pub(super) fn body_lines(
    content: &str,
    theme: &Theme,
    inner_width: u16,
    expanded: bool,
) -> Vec<Line<'static>> {
    let display = display_text_for_result(content);
    let text = display.as_ref();

    let label_style = Style::new()
        .fg(theme.palette.cyan)
        .add_modifier(Modifier::BOLD);
    let value_style = Style::new().fg(theme.palette.fg);

    // Peel leading `Label: value` metadata lines; markdown begins at the first
    // blank or non-metadata line.
    let mut meta: Vec<Line<'static>> = Vec::new();
    let mut rest_start = 0usize;
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            rest_start += 1;
            break;
        }
        if let Some((label, value)) = split_meta_label(line) {
            meta.push(Line::from(vec![
                Span::styled(format!("{label}: "), label_style),
                Span::styled(sanitize_inline(value), value_style),
            ]));
            rest_start += 1;
        } else {
            break;
        }
    }

    let rest = text.lines().skip(rest_start).collect::<Vec<_>>().join("\n");

    let mut lines = meta;
    if !rest.trim().is_empty() {
        lines.extend(crate::tui::markdown::rendered_lines_for_width(
            &rest,
            theme,
            inner_width,
        ));
    }

    let cap = body_line_cap(expanded, 24);
    let total = lines.len();
    lines.truncate(cap);
    push_clip_notice(&mut lines, expanded, total, cap, theme);
    lines
}

const fn plural_s(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// Hit count from a JSON search envelope: the first array among the common
/// result keys, else a top-level array.
fn json_hit_count(value: &serde_json::Value) -> Option<usize> {
    match value {
        serde_json::Value::Array(arr) => Some(arr.len()),
        serde_json::Value::Object(map) => ["results", "hits", "items", "matches", "data"]
            .iter()
            .find_map(|key| map.get(*key).and_then(serde_json::Value::as_array))
            .map(Vec::len),
        _ => None,
    }
}

/// `812B` / `12KB` / `3MB` — a compact, float-free byte figure for fetch lines.
fn human_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}MB", bytes / (1024 * 1024))
    }
}

/// Host of a URL (scheme + leading `www.` trimmed, path dropped) for compact
/// fetch summaries.
fn url_host(url: &str) -> String {
    let without_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    without_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .trim_start_matches("www.")
        .to_string()
}

/// Detect a leading `Label: value` metadata line (Title / Published / URL /
/// Author / Site) on a fetched web result. The label half never contains a
/// colon, so splitting on the first `:` leaves a URL value (with its own
/// `://`) intact.
fn split_meta_label(line: &str) -> Option<(&str, &str)> {
    const LABELS: [&str; 5] = ["Title", "Published", "URL", "Author", "Site"];
    let (label, value) = line.split_once(':')?;
    LABELS
        .contains(&label.trim())
        .then(|| (label.trim(), value.trim_start()))
}
