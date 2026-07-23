//! `RenderBlock::ToolResult` widget — collapsible result card.
//!
//! Per `.zo/design/components.md` §5.3. Dispatches body rendering
//! to [`crate::tui::blocks::diff`] for [`ToolResultBody::Diff`] and
//! handles the other variants inline (Text / Bash / Read / Listing /
//! Generic) with a compact summary.
//!
//! See `code-rules.md` R1, R2, R9.

#![allow(clippy::doc_markdown)]

use std::borrow::Cow;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use runtime::message_stream::{BashResult, ToolResultBody};

use crate::tui::glyphs;
use crate::tui::theme::Theme;

use super::diff;
use super::{compact_path_label, sanitize_inline, wrapped_rows};

mod bash;
mod diff_card;
mod listing;
mod read;
mod text;
mod todos;
mod web;

/// Chevron used on the summary line when collapsed.
pub const CHEVRON_COLLAPSED: &str = "\u{25b8}";
/// Chevron used on the summary line when expanded.
pub const CHEVRON_EXPANDED: &str = "\u{25be}";

/// Tool results with more than this many content lines are
/// auto-collapsed and show only a preview + expand hint.
const COLLAPSE_LINE_THRESHOLD: usize = 20;

/// Maximum hunk lines an inline `Diff` body renders before it is capped with a
/// `+N lines` hint. Claude Code shows edit diffs inline by default; this keeps a
/// large refactor from flooding the transcript while still surfacing the change
/// in place (the rest stays one keystroke away via expand / `/diff`). Larger
/// than [`COLLAPSE_LINE_THRESHOLD`] so ordinary edits show in full.
const DIFF_INLINE_LINE_CAP: usize = 32;

/// Upper bound on rows an *expanded* tool body renders. Expanding must show the
/// whole result (the roadmap's "펼쳤는데 잘렸다" trust fix), but an unbounded
/// render would make the transcript's per-block height prefix-sum (and the
/// hanging-line wrap that feeds it) scale with the full line count — a multi-
/// thousand-line file would stall layout. Cap at a generous ceiling and append a
/// `+N more  clipped` notice when clipped.
const EXPANDED_HARD_CAP: usize = 2000;

/// Hard ceiling on the wrapped rows a single tool body may emit. The *source
/// line count* is already capped at [`EXPANDED_HARD_CAP`], but that does nothing
/// for a few **very wide** lines — a minified blob, a base64 dump, or a mysql
/// row with no newlines can each wrap into hundreds of thousands of rows.
/// `wrap_hanging_line` would then allocate a `Line`/`Span` per row and stall
/// layout for tens of seconds (the freeze the watchdog catches as a
/// `drive_turn render loop` hang). Bounding the *output* rows keeps the wrap
/// O(ceiling·width) no matter how pathological the input is; the clipped tail is
/// never on screen anyway (the viewport only ever shows a small window). Far
/// above any legitimate body, so normal output is untouched.
const WRAP_ROW_CEILING: usize = 20_000;

/// Row cap for one tool-body section: the full [`EXPANDED_HARD_CAP`] when the
/// block is expanded, else the compact collapsed cap. Single source so the
/// expand/collapse split is identical at every body renderer.
pub(super) fn body_line_cap(expanded: bool, collapsed_cap: usize) -> usize {
    if expanded {
        EXPANDED_HARD_CAP
    } else {
        collapsed_cap
    }
}

/// Whether a "+N more" footer marks an openable preview or a hard limit.
///
/// One vocabulary for every tool-result card: the shape is always
/// `+N more  <tail>`, and only the tail word differs — `⏎ expand` for content a
/// keystroke reveals, `clipped` for content no keystroke can (a display cap or a
/// render ceiling). This replaces the four ad-hoc phrasings that used to share a
/// single card ("(truncated)", "more: +N ⏎", "showing first N of M", "output
/// clipped …").
#[derive(Clone, Copy)]
pub(super) enum MoreKind {
    /// Collapsed preview — Enter expands to reveal the hidden lines.
    Expand,
    /// Hard clip — a display cap or render ceiling Enter cannot lift.
    Clipped,
}

/// The single "+N more" footer token. Uses only one-cell glyphs (`+`, digits,
/// `⏎`/`enter`) so it never widens a card row under a `ko_KR` wide-ambiguous
/// tmux — unlike the `…` (U+2026) it replaces, which is East-Asian-Ambiguous.
pub(super) fn more_text(hidden: usize, kind: MoreKind, theme: &Theme) -> String {
    match kind {
        MoreKind::Expand => {
            let enter = glyphs::pick(!theme.no_color, glyphs::KEY_ENTER, glyphs::KEY_ENTER_NC);
            format!("+{hidden} more  {enter} expand")
        }
        MoreKind::Clipped => format!("+{hidden} more  clipped"),
    }
}

/// Append a dim `+N more  clipped` notice when an *expanded* body was still
/// clipped at its cap. Collapsed bodies get their own `+N more  ⏎ expand` hint
/// from `rendered_lines`, so this only fires for expanded views. No-op otherwise,
/// so the row count stays exactly `min(total, cap)` (+1 only when truly clipped),
/// which both the draw and height-estimate paths compute identically.
pub(super) fn push_clip_notice(
    lines: &mut Vec<Line<'_>>,
    expanded: bool,
    total: usize,
    cap: usize,
    theme: &Theme,
) {
    if expanded && total > cap {
        lines.push(Line::styled(
            more_text(total - cap, MoreKind::Clipped, theme),
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::ITALIC),
        ));
    }
}

/// Render a ToolResult row in Codex style:
/// `└ summary  [badge]` — borderless, attached to the preceding call.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    is_error: bool,
    body: &ToolResultBody,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    scroll_offset: u16,
) {
    let para = Paragraph::new(rendered_lines_for_width(
        is_error, body, theme, focused, expanded, area.width,
    ))
    .style(Style::new().fg(theme.palette.fg))
    .wrap(Wrap { trim: false })
    .scroll((scroll_offset, 0));
    frame.render_widget(para, area);
}

pub(crate) fn estimate_rows(
    is_error: bool,
    body: &ToolResultBody,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    width: u16,
) -> u16 {
    wrapped_rows(
        &rendered_lines_for_width(is_error, body, theme, focused, expanded, width),
        width,
    )
}

#[allow(clippy::too_many_lines)]
pub(crate) fn rendered_lines<'a>(
    is_error: bool,
    body: &'a ToolResultBody,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    width: u16,
) -> Vec<Line<'a>> {
    if let ToolResultBody::Todos(items) = body {
        return todos::block_lines(items, theme);
    }

    let (title_label, badge_label, badge_style) = header_pieces(is_error, body, theme);
    let title_style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::BOLD);
    let leader_style = Style::new().fg(theme.palette.dim);
    let summary_style = if is_error {
        Style::new().fg(theme.palette.error)
    } else if focused {
        Style::new().fg(theme.palette.info)
    } else {
        Style::new().fg(theme.palette.fg)
    };
    // Keep successful payload rails quiet so the result reads as a child of the
    // call rather than a second status card. Errors retain the semantic accent.
    let rail_style = if is_error && !theme.no_color {
        Style::new().fg(theme.palette.error)
    } else {
        Style::new().fg(theme.palette.dim)
    };

    let content_line_count = body_content_line_count(body);
    let is_long = content_line_count > COLLAPSE_LINE_THRESHOLD;
    // web_fetch / web_search results carry the page itself as their body. Show
    // it inline by default (toggleable) and render it as markdown below, so it
    // reads like the source page instead of one flat monochrome block — pi
    // parity for the most content-rich tool results.
    let is_web_result = matches!(
        body,
        ToolResultBody::Generic { name, .. } if web::is_fetch_tool(name) || web::is_search_tool(name)
    );
    let toggleable = matches!(body, ToolResultBody::Read { .. } | ToolResultBody::Diff(_))
        || is_long
        || is_web_result;
    let titled_payload = matches!(
        body,
        ToolResultBody::Text { .. } | ToolResultBody::Generic { .. }
    ) && title_label != "Result";
    // Claude Code parity: an edit `Diff` is shown inline by default — never
    // hidden behind an expand gate the way other long results are. A diff that
    // fits the inline cap renders in full; a larger one renders a capped slice
    // plus a `+N lines` hint (and `/diff` / expand still shows the rest). This
    // is provider-neutral: it keys only on the `Diff` body variant, so every
    // model's edit lands the same inline view.
    let is_diff = matches!(body, ToolResultBody::Diff(_));
    let diff_overflows_inline_cap = is_diff && content_line_count > DIFF_INLINE_LINE_CAP;
    // Show the full body when explicitly expanded, for short toggleable results,
    // or for a diff that fits inline. A long diff shows a capped inline preview.
    let show_body = expanded
        || (toggleable && !is_long && !diff_overflows_inline_cap)
        || (is_diff && !diff_overflows_inline_cap);
    // A *successful* Bash result collapses to just its summary line: the full
    // stdout is noise on the happy path. The chevron still marks it toggleable,
    // so expand restores the body one keystroke away. A failure (is_error or a
    // non-zero exit) keeps its preview — the stderr / exit code is the signal.
    let bash_ok = !is_error && matches!(body, ToolResultBody::Bash(bash) if bash.exit_code == 0);
    // Keep successful long payloads at one summary row by default. Their full
    // content is still one expand action away. Failures retain a capped preview,
    // and diffs keep their existing inline-preview contract.
    let failed_bash = matches!(body, ToolResultBody::Bash(bash) if bash.exit_code != 0);
    let show_collapsed_preview = !show_body
        && !bash_ok
        && ((is_long && !expanded && (is_error || failed_bash)) || diff_overflows_inline_cap);

    let chevron = if show_body {
        CHEVRON_EXPANDED
    } else {
        CHEVRON_COLLAPSED
    };

    let mut lines: Vec<Line<'_>> = Vec::new();
    let mut summary_spans: Vec<Span<'_>> = Vec::new();
    let summary_marker = summary_marker(theme);
    summary_spans.push(Span::styled(summary_marker.to_string(), leader_style));
    if toggleable && !is_diff {
        summary_spans.push(Span::styled(format!("{chevron} "), title_style));
        summary_spans.push(Span::styled(format!("{title_label}: "), title_style));
    } else if titled_payload {
        summary_spans.push(Span::styled(format!("{title_label}: "), title_style));
    }

    if let ToolResultBody::Diff(view) = body {
        summary_spans.extend(diff_card::summary_spans(view, theme, summary_style));
    } else {
        let summary_text = sanitize_inline(&body_summary(body));
        summary_spans.push(Span::styled(summary_text.clone(), summary_style));
    }

    // The badge is now empty on success (✓ XOR done — the call row's marker is the
    // sole success signal). It only carries information on failure (`exit 1`,
    // `x error`, `not answered`). Skip an empty badge, and also suppress a badge
    // the summary text already repeats verbatim.
    let summary_text_for_badge = body_summary(body);
    let badge_is_redundant = summary_text_for_badge
        .trim()
        .eq_ignore_ascii_case(badge_label.trim());
    if !badge_label.trim().is_empty() && !badge_is_redundant {
        summary_spans.push(Span::styled(" · ", Style::new().fg(theme.palette.dim)));
        summary_spans.push(Span::styled(badge_label, badge_style));
    }
    lines.push(Line::from(summary_spans));

    // Body rows sit one level under the summary. Use a continuation rail
    // instead of another root corner so dense read previews read as payload,
    // not as sibling transcript events.
    let body_indent = body_indent(theme);
    if show_body {
        let indent_w = u16::try_from(display_width(body_indent)).unwrap_or(0);
        let inner_width = width.saturating_sub(indent_w).max(1);
        let body_lines = match body {
            ToolResultBody::Generic { content, .. } if is_web_result => {
                web::body_lines(content, theme, inner_width, expanded)
            }
            _ => render_body(body, theme, expanded),
        };
        for l in body_lines {
            let mut spans = vec![Span::styled(body_indent, rail_style)];
            spans.extend(l.spans);
            lines.push(Line::from(spans));
        }
    } else if show_collapsed_preview {
        let preview_content_line_count = collapsed_preview_content_line_count(body);
        let preview_lines = collapsed_preview_line_count(body).min(preview_content_line_count);
        for l in collapsed_preview_lines(body, theme)
            .into_iter()
            .take(preview_lines)
        {
            let mut spans = vec![Span::styled(body_indent, rail_style)];
            spans.extend(l.spans);
            lines.push(Line::from(spans));
        }
        let hidden = preview_content_line_count.saturating_sub(preview_lines);
        let hint_style = Style::new()
            .fg(theme.palette.dim)
            .add_modifier(Modifier::ITALIC);
        lines.push(Line::from(vec![
            Span::styled(body_indent, rail_style),
            Span::styled(more_text(hidden, MoreKind::Expand, theme), hint_style),
        ]));
    }
    lines
}

pub(crate) fn rendered_lines_for_width<'a>(
    is_error: bool,
    body: &'a ToolResultBody,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    width: u16,
) -> Vec<Line<'a>> {
    let lines = rendered_lines(is_error, body, theme, focused, expanded, width);
    let width = usize::from(width.max(1));
    let mut lines = wrap_hanging_lines(lines, width, theme);
    if matches!(body, ToolResultBody::Diff(_)) {
        diff::pad_changed_rows(&mut lines, width);
    }
    lines
}

/// Render a `TodoWrite` / `TaskList` result as a grouped transcript block with
/// explicit top/bottom boundaries. The live todo panel can still show the
/// active checklist, but the transcript keeps a settled `Updated Plan` history
/// entry instead of dropping the result on the floor.
fn wrap_hanging_lines<'a>(lines: Vec<Line<'a>>, width: usize, theme: &Theme) -> Vec<Line<'a>> {
    let mut out: Vec<Line<'a>> = Vec::new();
    let mut clipped = false;
    let mut hidden = 0usize; // whole source lines dropped past the render ceiling
    let total = lines.len();
    for (idx, line) in lines.into_iter().enumerate() {
        if out.len() >= WRAP_ROW_CEILING {
            hidden = total - idx; // this line + all following never rendered
            clipped = true;
            break;
        }
        // Give each line only the rows left under the ceiling, so a single
        // pathological wide line cannot blow the whole budget on its own.
        let budget = WRAP_ROW_CEILING - out.len();
        let (rows, line_clipped) = wrap_hanging_line(line, width, theme, budget);
        out.extend(rows);
        clipped |= line_clipped;
    }
    if clipped {
        // Same `+N more  clipped` vocabulary as the cap/preview notices, plus the
        // reason. When only a single over-long line was cut mid-row (no whole
        // line dropped) the count is unknown, so fall back to a bare note.
        let text = if hidden > 0 {
            format!("{} (too large)", more_text(hidden, MoreKind::Clipped, theme))
        } else {
            "clipped (output too large to render in full)".to_string()
        };
        out.push(Line::styled(
            text,
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::ITALIC),
        ));
    }
    out
}

// `row = continuation.clone()` re-initializes a var moved into the pushed Line;
// `clone_from` needs an initialized target, so `assigning_clones` is a false positive here.
#[allow(clippy::assigning_clones)]
fn wrap_hanging_line<'a>(
    line: Line<'a>,
    width: usize,
    theme: &Theme,
    budget: usize,
) -> (Vec<Line<'a>>, bool) {
    if line_width(&line) <= width {
        return (vec![line], false);
    }

    let continuation = continuation_prefix(&line, theme);
    let continuation_width = spans_width(&continuation);
    let continuation_limit = width.saturating_sub(continuation_width).max(1);

    let line_style = line.style;
    let alignment = line.alignment;
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut row_width = 0usize;
    let mut limit = width;
    let mut clipped = false;

    'wrap: for span in line.spans {
        let style = span.style;
        let text = span.content.into_owned();
        let mut remaining = text.as_str();
        while !remaining.is_empty() {
            // Stop before emitting a row past the caller's budget: one very wide
            // line (minified/base64/no-newline dump) would otherwise wrap into
            // hundreds of thousands of `Line`/`Span` allocations and stall layout
            // for tens of seconds. The clipped tail is offscreen anyway, and this
            // keeps the work O(budget·width) regardless of the input's width.
            if rows.len() >= budget {
                clipped = true;
                break 'wrap;
            }
            if !continuation.is_empty() && row_width == continuation_width {
                remaining = remaining.trim_start_matches(' ');
                if remaining.is_empty() {
                    break;
                }
            }
            let available = limit.saturating_sub(row_width);
            if available == 0 {
                rows.push(Line {
                    spans: row,
                    style: line_style,
                    alignment,
                });
                row = continuation.clone();
                row_width = continuation_width;
                limit = continuation_width.saturating_add(continuation_limit);
                continue;
            }

            let (head, tail) = crate::tui::markdown::split_at_display_width(remaining, available);
            if !head.is_empty() {
                row_width = row_width.saturating_add(display_width(head));
                row.push(Span::styled(head.to_string(), style));
            }
            if tail.is_empty() {
                break;
            }

            rows.push(Line {
                spans: row,
                style: line_style,
                alignment,
            });
            row = continuation.clone();
            row_width = continuation_width;
            limit = continuation_width.saturating_add(continuation_limit);
            remaining = if continuation.is_empty() {
                tail
            } else {
                tail.trim_start_matches(' ')
            };
        }
    }

    rows.push(Line {
        spans: row,
        style: line_style,
        alignment,
    });
    (rows, clipped)
}

fn continuation_prefix<'a>(line: &Line<'a>, theme: &Theme) -> Vec<Span<'a>> {
    let leader = line.spans.first().map(|span| span.content.as_ref());
    let leader_style = line
        .spans
        .first()
        .map_or(Style::default(), |span| span.style);
    let body_indent = body_indent(theme);

    if leader == Some(body_indent) {
        let mut spans = vec![Span::styled(body_indent.to_string(), leader_style)];
        if let Some((width, style)) = diff_code_prefix_width(line) {
            spans.push(Span::styled(" ".repeat(width), style));
            return spans;
        }
        if let Some(number) = line.spans.get(1).filter(|span| is_source_line_number(span)) {
            spans.push(Span::styled(
                " ".repeat(display_width(number.content.as_ref())),
                number.style,
            ));
            return spans;
        }
        spans.push(Span::styled("  ".to_string(), leader_style));
        return spans;
    }

    if leader == Some(summary_marker(theme)) {
        return vec![Span::styled("    ".to_string(), leader_style)];
    }

    Vec::new()
}

fn body_indent(theme: &Theme) -> &'static str {
    if theme.no_color {
        "  | "
    } else {
        "  \u{2502} "
    }
}

fn summary_marker(theme: &Theme) -> &'static str {
    if theme.no_color {
        "  ` "
    } else {
        "  \u{2514} "
    }
}

fn diff_code_prefix_width(line: &Line<'_>) -> Option<(usize, Style)> {
    // Span 0 is ToolResult's body indent. Diff code rows then use:
    // line-number, space, op, separator, code...
    // Wrapped continuations should resume at the code column.
    let number = line.spans.get(1)?;
    let number_op_space = line.spans.get(2)?;
    let op = line.spans.get(3)?;
    let separator = line.spans.get(4)?;
    line.spans.get(5)?;

    let op_text = op.content.as_ref();
    let separator_text = separator.content.as_ref();
    if number_op_space.content.as_ref() != " "
        || !matches!(op_text, "+" | "-" | " ")
        || separator_text != " "
    {
        return None;
    }

    let width = display_width(number.content.as_ref())
        + display_width(number_op_space.content.as_ref())
        + display_width(op_text)
        + display_width(separator_text);
    Some((width, number.style))
}

fn is_source_line_number(span: &Span<'_>) -> bool {
    let text = span.content.as_ref();
    text.len() == 5 && text.ends_with(' ') && text.trim().chars().all(|ch| ch.is_ascii_digit())
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

use crate::tui::text_metrics::display_width;

fn collapsed_preview_line_count(body: &ToolResultBody) -> usize {
    match body {
        // A capped inline diff shows up to the inline cap before the `+N` hint,
        // so an overflowing edit still surfaces a substantial slice in place.
        ToolResultBody::Diff(_) => DIFF_INLINE_LINE_CAP,
        ToolResultBody::Bash(_) => 1,
        ToolResultBody::Read { .. } => 2,
        ToolResultBody::Text { .. }
        | ToolResultBody::Generic { .. }
        | ToolResultBody::Listing { .. }
        | ToolResultBody::Todos(_) => 3,
    }
}

fn collapsed_preview_content_line_count(body: &ToolResultBody) -> usize {
    match body {
        ToolResultBody::Diff(view) => {
            diff::rendered_line_count(view).saturating_sub(diff::header_rows())
        }
        _ => body_content_line_count(body),
    }
}

fn header_pieces(
    is_error: bool,
    body: &ToolResultBody,
    theme: &Theme,
) -> (&'static str, String, Style) {
    let title = match body {
        ToolResultBody::Generic { name, .. } if is_ask_user_question(name) => "Question",
        ToolResultBody::Text { content, .. } | ToolResultBody::Generic { content, .. } => {
            text_payload_kind(content).map_or("Result", FilePayloadKind::title)
        }
        ToolResultBody::Bash(_) => "Bash",
        ToolResultBody::Read { .. } => "Read",
        ToolResultBody::Diff(_) => "Edit",
        ToolResultBody::Listing { .. } => "List",
        ToolResultBody::Todos(_) => "Plan",
    };
    let (badge, style) = if is_error && is_ask_user_question_result(body) {
        (
            "not answered".to_string(),
            Style::new()
                .fg(theme.palette.warn)
                .add_modifier(Modifier::BOLD),
        )
    } else if is_error {
        (
            "x error".to_string(),
            Style::new()
                .fg(theme.palette.error)
                .add_modifier(Modifier::BOLD),
        )
    } else if let ToolResultBody::Bash(BashResult { exit_code, .. }) = body {
        (
            // Success is already signalled by the call row's `✓` marker, so an
            // extra "done" word is redundant chrome — emit nothing. Only a
            // non-zero exit carries information the marker does not.
            if *exit_code == 0 {
                String::new()
            } else {
                format!("exit {exit_code}")
            },
            Style::new()
                .fg(theme.palette.error)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        // Plain success: the `✓` marker says it; no "ok" badge.
        (String::new(), Style::new().fg(theme.palette.success))
    };
    (title, badge, style)
}

fn body_summary(body: &ToolResultBody) -> String {
    match body {
        ToolResultBody::Generic {
            name,
            content,
            truncated,
        } if is_ask_user_question(name) => text::question_summary(content, *truncated),
        ToolResultBody::Generic {
            name,
            content,
            truncated,
        } if web::is_search_tool(name) => web::search_summary(content, *truncated),
        ToolResultBody::Generic {
            name,
            content,
            truncated,
        } if web::is_fetch_tool(name) => web::fetch_summary(content, *truncated),
        ToolResultBody::Text { content, truncated }
        | ToolResultBody::Generic {
            content, truncated, ..
        } => text::summary(content, *truncated),
        ToolResultBody::Bash(b) => bash::summary(b),
        ToolResultBody::Read {
            path,
            content,
            truncated,
            ..
        } => read::summary(path, content, *truncated),
        ToolResultBody::Diff(view) => diff_card::summary(view),
        ToolResultBody::Listing { entries, truncated } => listing::summary(entries, *truncated),
        ToolResultBody::Todos(items) => todos::summary(items),
    }
}

fn is_ask_user_question_result(body: &ToolResultBody) -> bool {
    matches!(body, ToolResultBody::Generic { name, .. } if is_ask_user_question(name))
}

fn is_ask_user_question(name: &str) -> bool {
    name.eq_ignore_ascii_case("AskUserQuestion")
}

/// One ultra-compact result digest for a *collapsed tool-group* detail row, so a
/// settled group of 2+ tools no longer hides each result (grep hit count, bash
/// exit, read line count, web result count) the way the merged view did. Reuses
/// the same primitives the single-tool summary uses. Fail-open: a tool without a
/// meaningful one-token signal returns "" and the row shows verb+target only.
///
/// Failure surfaces the signal instead of the count: bash always shows a non-zero
/// `exit N`, and any other errored tool shows its first error code-word
/// (`ENOENT`/`timeout`/…) — the caller tints that red.
pub(crate) fn collapsed_group_digest(name: &str, body: &ToolResultBody, is_error: bool) -> String {
    // Bash carries its signal in the exit code, error or not (exit 0 stays quiet).
    if let ToolResultBody::Bash(b) = body {
        return bash::digest(b.exit_code);
    }
    // Any other failed tool: surface the first error code-word, not a count.
    if is_error {
        return first_error_token(body_digest_text(body));
    }
    match body {
        ToolResultBody::Read { content, .. } => read::digest(content),
        ToolResultBody::Listing { entries, truncated } => {
            listing::digest(name, entries, *truncated)
        }
        ToolResultBody::Generic {
            content, truncated, ..
        } if web::is_search_tool(name) => web::digest(content, *truncated),
        _ => String::new(),
    }
}

/// Text mined for a one-word error code when a non-bash tool failed.
fn body_digest_text(body: &ToolResultBody) -> &str {
    match body {
        ToolResultBody::Text { content, .. }
        | ToolResultBody::Generic { content, .. }
        | ToolResultBody::Read { content, .. } => content,
        _ => "",
    }
}

/// First whitespace/`:`/`,`-delimited token of the first non-empty line, clamped
/// short — "ENOENT: no such file" → "ENOENT", "timeout after 30s" → "timeout".
/// Falls back to "error" so a failed row never renders an empty digest.
fn first_error_token(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let token: String = line
        .split(|c: char| c.is_whitespace() || c == ':' || c == ',')
        .find(|t| !t.is_empty())
        .unwrap_or("")
        .chars()
        .take(16)
        .collect();
    if token.is_empty() {
        "error".to_string()
    } else {
        token
    }
}

/// One-line summary for a parsed JSON value used as a tool result.
///
/// Recognises three common envelope shapes — `{ "ok": bool }`,
/// `{ "status": "..." }`, and `{ "error": "..." }` — and falls back to a
/// generic shape descriptor (field count, item count, scalar value)
/// otherwise. Never returns a multi-line string.
pub(super) fn summarize_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(summary) = summarize_file_payload(map) {
                return summary;
            }
            if let Some(err) = map.get("error").and_then(serde_json::Value::as_str) {
                return format!("error: {}", truncate_inline(err, 60));
            }
            if let Some(status) = map.get("status").and_then(serde_json::Value::as_str) {
                return status.to_string();
            }
            if let Some(ok) = map.get("ok").and_then(serde_json::Value::as_bool) {
                return if ok {
                    "ok".to_string()
                } else {
                    "failed".to_string()
                };
            }
            match map.len() {
                0 => "empty object".to_string(),
                1 => format!("{}: …", map.keys().next().expect("len == 1")),
                n => {
                    let first = map.keys().next().expect("len > 0");
                    format!("{n} fields ({first}, …)")
                }
            }
        }
        serde_json::Value::Array(arr) => match arr.len() {
            0 => "empty array".to_string(),
            n => format!("{n} items"),
        },
        serde_json::Value::String(s) => truncate_inline(s, 80),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

fn summarize_file_payload(map: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let path = map
        .get("filePath")
        .and_then(serde_json::Value::as_str)
        .or_else(|| map.get("path").and_then(serde_json::Value::as_str))?;
    let path = compact_path_label(path);
    let kind = file_payload_kind(map)?;
    Some(format!("{path} · {}", kind.summary_label()))
}

#[derive(Clone, Copy)]
enum FilePayloadKind {
    Edit,
    Diff,
}

impl FilePayloadKind {
    const fn title(self) -> &'static str {
        match self {
            Self::Edit => "Edit",
            Self::Diff => "Diff",
        }
    }

    const fn summary_label(self) -> &'static str {
        match self {
            Self::Edit => "edit",
            Self::Diff => "diff",
        }
    }
}

fn file_payload_kind(map: &serde_json::Map<String, serde_json::Value>) -> Option<FilePayloadKind> {
    if map.contains_key("newString") || map.contains_key("oldString") {
        return Some(FilePayloadKind::Edit);
    }
    if map.contains_key("gitDiff") {
        return Some(FilePayloadKind::Diff);
    }
    None
}

fn text_payload_kind(content: &str) -> Option<FilePayloadKind> {
    let value: serde_json::Value = serde_json::from_str(content.trim()).ok()?;
    let serde_json::Value::Object(map) = value else {
        return None;
    };
    if !map.contains_key("filePath") && !map.contains_key("path") {
        return None;
    }
    file_payload_kind(&map)
}

fn truncate_inline(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let end = text
        .char_indices()
        .nth(max_chars)
        .map_or(text.len(), |(idx, _)| idx);
    format!("{}…", &text[..end])
}

/// Summary-line suffix when the source itself was cut upstream. Uses the card's
/// single hard-clip word ("clipped") so a summary and its `+N more  clipped`
/// footer speak the same vocabulary, rather than the old "(truncated)".
pub(super) fn trunc_suffix(truncated: bool) -> &'static str {
    if truncated { " (clipped)" } else { "" }
}

fn render_body<'a>(body: &'a ToolResultBody, theme: &Theme, expanded: bool) -> Vec<Line<'a>> {
    match body {
        ToolResultBody::Diff(view) => diff::body_lines(view, theme),
        ToolResultBody::Bash(b) => bash::body_lines(b, theme, expanded),
        ToolResultBody::Read {
            path,
            content,
            language,
            ..
        } => read::body_lines(path, content, language.as_deref(), theme, expanded),
        ToolResultBody::Text { content, .. } | ToolResultBody::Generic { content, .. } => {
            text::body_lines(content, theme, expanded)
        }
        ToolResultBody::Listing { entries, .. } => {
            listing::body_lines(entries, theme, expanded)
        }
        ToolResultBody::Todos(items) => todos::block_lines(items, theme),
    }
}

/// Collapsed preview always renders with the compact caps (`expanded = false`),
/// regardless of the block's expand state — the collapsed preview is a teaser
/// capped further by `rendered_lines` with its own `+N more  ⏎ expand` hint.
fn collapsed_preview_lines<'a>(body: &'a ToolResultBody, theme: &Theme) -> Vec<Line<'a>> {
    render_body(body, theme, false)
}

pub(super) fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.lines().count()
    }
}

/// Count the total content lines for a tool result body. Used to
/// decide whether the result should be auto-collapsed.
fn body_content_line_count(body: &ToolResultBody) -> usize {
    match body {
        ToolResultBody::Text { content, .. } | ToolResultBody::Generic { content, .. } => {
            count_display_lines(content)
        }
        ToolResultBody::Bash(b) => count_display_lines(&b.stdout) + count_display_lines(&b.stderr),
        ToolResultBody::Read { content, .. } => count_display_lines(content),
        ToolResultBody::Diff(view) => {
            diff::rendered_line_count(view).saturating_sub(diff::header_rows())
        }
        ToolResultBody::Listing { entries, .. } => entries.len(),
        ToolResultBody::Todos(items) => items.len().saturating_add(3),
    }
}

pub(super) fn count_display_lines(text: &str) -> usize {
    let display = display_text_for_result(text);
    count_lines(display.as_ref())
}

pub(super) fn display_text_for_result(text: &str) -> Cow<'_, str> {
    decode_escaped_multiline_for_display(text)
        .map(Cow::Owned)
        .unwrap_or(Cow::Borrowed(text))
}

fn decode_escaped_multiline_for_display(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if (trimmed.starts_with('{') || trimmed.starts_with('['))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        return None;
    }

    let escaped_newlines = text.matches("\\n").count();
    if escaped_newlines < 2 || text.matches('\n').count() > escaped_newlines {
        return None;
    }

    let decoded = text
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\\"", "\"");
    (decoded != text).then_some(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::Theme;
    use runtime::message_stream::{TodoResultItem, TodoResultStatus};

    fn bash(exit_code: i32) -> ToolResultBody {
        ToolResultBody::Bash(BashResult {
            exit_code,
            stdout: "ran\n".to_string(),
            stderr: String::new(),
            truncated: false,
        })
    }

    fn flatten(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect()
    }

    #[test]
    fn expanded_body_uncaps_read_and_bash_collapsed_keeps_preview_caps() {
        // Roadmap ④: take(24)/take(12) used to clip even an expanded block. Now
        // expanded shows everything (up to EXPANDED_HARD_CAP); collapsed keeps the
        // compact caps. Height stays coupled because both draw and measure go
        // through the same `render_body(.., expanded)`.
        let theme = Theme::default_dark();
        let content = (1..=40).map(|i| format!("ln{i}")).collect::<Vec<_>>().join("\n");
        let read = ToolResultBody::Read {
            path: "a.rs".to_string(),
            content,
            language: None,
            truncated: false,
        };
        assert_eq!(
            render_body(&read, &theme, false).len(),
            1 + 24,
            "collapsed read caps body at 24 (+1 header)"
        );
        assert_eq!(
            render_body(&read, &theme, true).len(),
            1 + 40,
            "expanded read shows all 40 lines (+1 header)"
        );
        let stdout = (1..=20).map(|i| format!("o{i}")).collect::<Vec<_>>().join("\n");
        let bash = ToolResultBody::Bash(BashResult {
            exit_code: 0,
            stdout,
            stderr: String::new(),
            truncated: false,
        });
        assert_eq!(
            render_body(&bash, &theme, false).len(),
            1 + 12,
            "collapsed bash caps stdout at 12 (+1 stream header)"
        );
        assert_eq!(
            render_body(&bash, &theme, true).len(),
            1 + 20,
            "expanded bash shows all 20 stdout lines (+1 stream header)"
        );
    }

    #[test]
    fn clip_notice_only_fires_when_expanded_and_over_cap() {
        let theme = Theme::default_dark();
        let mut over = Vec::new();
        push_clip_notice(&mut over, true, 5000, 2000, &theme);
        assert_eq!(over.len(), 1);
        // Unified vocabulary: `+N more  clipped` (N = total − cap), not the old
        // "showing first N of M".
        assert!(flatten(&over).contains("+3000 more  clipped"));
        let mut collapsed = Vec::new();
        push_clip_notice(&mut collapsed, false, 5000, 2000, &theme);
        assert!(
            collapsed.is_empty(),
            "collapsed bodies use the '+N more  ⏎ expand' hint, not this notice"
        );
        let mut under = Vec::new();
        push_clip_notice(&mut under, true, 100, 2000, &theme);
        assert!(under.is_empty(), "no notice when nothing was clipped");
    }

    #[test]
    fn group_digest_extracts_per_tool_signal_fail_open() {
        // Roadmap ⑤ extractor: reuse single-tool primitives, fail-open on unknowns.
        let grep = ToolResultBody::Listing {
            entries: vec!["m".to_string(); 12],
            truncated: false,
        };
        assert_eq!(collapsed_group_digest("grep_search", &grep, false), "12 hits");
        let glob = ToolResultBody::Listing {
            entries: vec!["f".to_string(); 3],
            truncated: false,
        };
        assert_eq!(collapsed_group_digest("glob_search", &glob, false), "3 files");
        assert_eq!(collapsed_group_digest("bash", &bash(2), false), "exit 2");
        assert_eq!(
            collapsed_group_digest("bash", &bash(0), false),
            "",
            "successful bash stays quiet"
        );
        let read = ToolResultBody::Read {
            path: "a".to_string(),
            content: "l1\nl2\nl3".to_string(),
            language: None,
            truncated: false,
        };
        assert_eq!(collapsed_group_digest("read_file", &read, false), "3 ln");
        let read_err = ToolResultBody::Read {
            path: "a".to_string(),
            content: "ENOENT: no such file".to_string(),
            language: None,
            truncated: false,
        };
        assert_eq!(
            collapsed_group_digest("read_file", &read_err, true),
            "ENOENT",
            "failed non-bash tool surfaces the first error code-word"
        );
        let other = ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        };
        assert_eq!(
            collapsed_group_digest("write_file", &other, false),
            "",
            "unsupported tool/body → empty digest (row shows verb+target only)"
        );
    }

    #[test]
    fn success_omits_the_done_badge_failure_keeps_exit_code() {
        // ✓ XOR done: the call row's marker already signals success, so the result
        // row drops the redundant "done"/"ok". Only a non-zero exit is informative.
        let ok = flatten(&rendered_lines(false, &bash(0), &Theme::no_color(), false, false, 80));
        assert!(!ok.contains("done"), "success row must not repeat 'done': {ok}");

        let failed = flatten(&rendered_lines(false, &bash(1), &Theme::no_color(), false, false, 80));
        assert!(
            failed.contains("exit 1"),
            "failure keeps the informative exit badge: {failed}"
        );
    }

    #[test]
    fn tool_result_body_uses_continuation_rail_under_summary() {
        let body = ToolResultBody::Text {
            content: "first\nsecond".to_string(),
            truncated: false,
        };

        // Summary owns a child marker; body rows use a continuation rail so
        // dense tool clusters do not look like several sibling roots.
        let plain = rendered_lines(false, &body, &Theme::no_color(), false, true, 80);
        assert!(plain.len() >= 3, "summary + two body rows: {}", plain.len());
        assert_eq!(plain[0].spans[0].content.as_ref(), "  ` ");
        assert_eq!(
            plain[1].spans[0].content.as_ref(),
            "  | ",
            "first body row uses an ASCII continuation rail"
        );
        assert_eq!(
            plain[2].spans[0].content.as_ref(),
            "  | ",
            "continuation rows align with the first body row"
        );

        // A true-color palette keeps the same rail width but uses a quiet
        // box-rule glyph so payload rows remain visually grouped.
        let colored = rendered_lines(false, &body, &Theme::zo(), false, true, 80);
        assert_eq!(colored[0].spans[0].content.as_ref(), "  \u{2514} ");
        assert_eq!(colored[1].spans[0].content.as_ref(), "  \u{2502} ");
    }

    #[test]
    fn web_fetch_body_renders_metadata_labels_and_markdown() {
        // pi parity: a fetched page shows cyan Title:/Published: labels and its
        // content as rendered markdown (links underlined), not flat monochrome
        // text — and web results show their body inline (toggleable) by default.
        let theme = Theme::zo();
        let body = generic(
            "web_fetch",
            "Title: Pi Coding Agent\nPublished: Jun 19, 2026\n\n# Pi 0.79.8\n\nGet it from [npm](https://npm.im/pi).",
        );
        let lines = rendered_lines(false, &body, &theme, false, true, 80);

        let title = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.as_ref() == "Title: ")
            .expect("Title: metadata label is rendered");
        assert_eq!(
            title.style.fg,
            Some(theme.palette.cyan),
            "Title label is cyan"
        );

        let link = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.as_ref() == "npm")
            .expect("markdown link text is rendered (engine ran on the body)");
        assert_eq!(link.style.fg, Some(theme.palette.cyan), "link is cyan");
        assert!(
            link.style.add_modifier.contains(Modifier::UNDERLINED),
            "markdown engine rendered the link as underlined"
        );
    }

    #[test]
    fn result_body_rail_is_quiet_except_on_error() {
        // Clean results stay neutral; only a failure spends semantic color on
        // the continuation rail. Glyph/content is unchanged.
        let theme = Theme::zo();
        let body = ToolResultBody::Text {
            content: "a\nb".to_string(),
            truncated: false,
        };

        let ok = rendered_lines(false, &body, &theme, false, true, 80);
        let ok_rail = &ok[1].spans[0];
        assert_eq!(ok_rail.content.as_ref(), "  \u{2502} ", "rail glyph unchanged");
        assert_eq!(
            ok_rail.style.fg,
            Some(theme.palette.dim),
            "clean result rail stays neutral"
        );

        let err = rendered_lines(true, &body, &theme, false, true, 80);
        assert_eq!(
            err[1].spans[0].style.fg,
            Some(theme.palette.error),
            "failed result rail is error-tinted"
        );

        // NO_COLOR stays neutral.
        let plain = rendered_lines(false, &body, &Theme::no_color(), false, true, 80);
        assert_eq!(
            plain[1].spans[0].style.fg,
            Some(Theme::no_color().palette.dim),
            "NO_COLOR rail stays neutral dim"
        );
    }

    fn generic(name: &str, content: &str) -> ToolResultBody {
        ToolResultBody::Generic {
            name: name.to_string(),
            content: content.to_string(),
            truncated: false,
        }
    }

    #[test]
    fn web_search_summary_counts_hits_not_json_fields() {
        // Builtin DDG returns a Markdown bullet list.
        let md = "- [GPT-5.5](https://openai.com/a)\n- [GPT-5.6-Sol](https://openai.com/b)";
        assert_eq!(body_summary(&generic("web_search", md)), "2 results");

        // Empty search.
        let none = "No web search results matched the query \"x\".";
        assert_eq!(body_summary(&generic("WebSearch", none)), "0 results");

        // MCP JSON envelope — count the results array, not the field count.
        let json = r#"{"results":[{"t":1},{"t":2},{"t":3}]}"#;
        assert_eq!(body_summary(&generic("mcp__x__search", json)), "3 results");
    }

    #[test]
    fn web_fetch_summary_replaces_cryptic_field_count() {
        // The exact regression: a JSON fetch result degraded to
        // "6 fields (bytes, …)" via the generic JSON summarizer.
        let json = r#"{"bytes":12288,"title":"Introducing GPT-5.5","url":"https://openai.com","a":1,"b":2,"c":3}"#;
        let summary = body_summary(&generic("fetch", json));
        assert!(summary.contains("12KB"), "expected size, got {summary}");
        assert!(summary.contains("title found"), "expected title, got {summary}");
        assert!(
            !summary.contains("fields"),
            "must not show cryptic field count: {summary}"
        );

        // Builtin DDG fetch text.
        let text = "Fetched https://www.openai.com/index/introducing-gpt-5-5/\nTitle: Introducing GPT-5.5";
        let summary = body_summary(&generic("web_fetch", text));
        assert!(summary.contains("openai.com"), "expected host, got {summary}");
        assert!(summary.contains("title found"), "expected title, got {summary}");
    }

    fn todo(content: &str, active: &str, status: TodoResultStatus) -> TodoResultItem {
        TodoResultItem {
            content: content.to_string(),
            active_form: active.to_string(),
            status,
        }
    }

    /// The transcript ToolResult for a TodoWrite now owns a settled bordered
    /// `Updated Plan` history block, with explicit top and bottom boundaries.
    #[test]
    fn todos_result_renders_bordered_plan_block_in_transcript() {
        let body = ToolResultBody::Todos(vec![
            todo(
                "Render the block",
                "Rendering the block",
                TodoResultStatus::InProgress,
            ),
            todo("Add tests", "Adding tests", TodoResultStatus::Pending),
        ]);
        let lines = rendered_lines(false, &body, &Theme::no_color(), false, true, 80);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();

        assert_eq!(lines.len(), 4, "top + 2 items + bottom: {text}");
        assert!(text.contains("Updated Plan"), "title visible: {text}");
        assert!(
            text.contains("Rendering the block"),
            "active form visible: {text}"
        );
        assert!(
            lines.first().is_some_and(|line| line
                .spans
                .first()
                .is_some_and(|span| span.content.starts_with('+'))),
            "top border is visible: {text}"
        );
        assert!(
            lines.last().is_some_and(|line| line
                .spans
                .first()
                .is_some_and(|span| span.content.starts_with('+'))),
            "bottom border is visible: {text}"
        );
    }

}
