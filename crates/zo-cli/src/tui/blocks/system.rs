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
use ratatui::widgets::{Paragraph};
use runtime::message_stream::SystemLevel;

use crate::tui::glyphs;
use crate::tui::markdown;
use crate::tui::text_metrics::{char_width, display_width, wrap_line_to_cells};
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
        Paragraph::new(super::wrap_rows(
            &rendered_lines(level, text, theme, area.width),
            area.width,
        ))
        .scroll((scroll_offset, 0)),
        area,
    );
}

pub(crate) fn estimate_rows(level: SystemLevel, text: &str, theme: &Theme, width: u16) -> u16 {
    wrapped_rows(&rendered_lines(level, text, theme, width), width)
}

/// Visible marker of a plan-step "chapter" notice — the one-line transcript
/// heading the App pushes when the active plan step changes. Carried in the text
/// (not a new `RenderBlock` variant) so the runtime enum, session
/// serialization, scroll/copy, and the attach protocol stay untouched.
pub(crate) const PLAN_CHAPTER_PREFIX: &str = "▸ ";

/// Chapter rows: accent marker, bold sentence. Deliberately bypasses the
/// markdown/report routing above — a chapter sentence carrying backticks is a
/// heading, not a document.
fn rendered_chapter_lines(text: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    let (marker, body) = text.split_at(PLAN_CHAPTER_PREFIX.len());
    let marker_style = Style::new().fg(theme.palette.accent);
    let body_style = Style::new()
        .fg(theme.palette.fg)
        .add_modifier(Modifier::BOLD);
    let body_width = usize::from(width.saturating_sub(2)).max(1);
    let body_line = Line::from(vec![Span::styled(body.to_string(), body_style)]);
    wrap_line_to_cells(&body_line, body_width, false)
        .into_iter()
        .enumerate()
        .map(|(wrapped_row, row)| {
            let mut spans = Vec::with_capacity(row.spans.len() + 1);
            if wrapped_row == 0 {
                spans.push(Span::styled(marker.to_string(), marker_style));
            } else {
                spans.push(Span::raw("  "));
            }
            spans.extend(row.spans);
            Line::from(spans)
        })
        .collect()
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

fn has_markdown_block_signal(text: &str) -> bool {
    text.lines().any(|raw| {
        let line = raw.trim_start();
        if line.starts_with("```") || line.starts_with("~~~") {
            return true;
        }
        if line.starts_with("> ") || line == ">" {
            return true;
        }
        for marker in ["- [ ] ", "- [x] ", "- [X] ", "- ", "* ", "+ "] {
            if line.starts_with(marker) {
                return true;
            }
        }
        let digits = line.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0
            && line.as_bytes().get(digits) == Some(&b'.')
            && line.as_bytes().get(digits + 1) == Some(&b' ')
        {
            return true;
        }
        let hashes = line.bytes().take_while(|byte| *byte == b'#').count();
        (1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ')
    })
}

/// Return the shared value column of a hand-aligned report. Report rows start
/// with two spaces and separate their label from their value with at least two
/// more spaces. Once one such row establishes the column, longer labels with a
/// one-space separator (for example `Working directory`) use the same column.
fn report_value_column(text: &str) -> Option<usize> {
    if text.lines().count() < 2 || has_markdown_block_signal(text) {
        return None;
    }
    text.lines().skip(1).find_map(|line| {
        if !line.starts_with("  ") {
            return None;
        }
        let bytes = line.as_bytes();
        let mut cursor = 2;
        while cursor < bytes.len() {
            if bytes[cursor] != b' ' {
                cursor += 1;
                continue;
            }
            let separator_start = cursor;
            while cursor < bytes.len() && bytes[cursor] == b' ' {
                cursor += 1;
            }
            if separator_start > 2 && cursor - separator_start >= 2 && cursor < bytes.len() {
                return Some(display_width(&line[..cursor]));
            }
        }
        None
    })
}

fn split_at_display_column(line: &str, column: usize) -> Option<(&str, &str)> {
    if !line.starts_with("  ") || display_width(line) <= column {
        return None;
    }
    let mut used = 0;
    for (byte_index, ch) in line.char_indices() {
        if used == column {
            let (prefix, value) = line.split_at(byte_index);
            return (prefix.ends_with(' ') && !prefix.trim().is_empty() && !value.is_empty())
                .then_some((prefix, value));
        }
        used += char_width(ch);
        if used > column {
            return None;
        }
    }
    None
}

fn balanced_marker(text: &str, marker: &str) -> Option<(usize, usize)> {
    let start = text.find(marker)?;
    let content_start = start + marker.len();
    let end = content_start + text[content_start..].find(marker)?;
    Some((start, end))
}

/// Render the two inline constructs used by reports without handing the whole
/// row to CommonMark, which intentionally collapses soft breaks and spaces.
fn report_inline_spans(
    text: &str,
    text_style: Style,
    code_style: Style,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let code = balanced_marker(rest, "`").map(|(start, end)| (start, end, "`"));
        let bold = balanced_marker(rest, "**").map(|(start, end)| (start, end, "**"));
        let next = match (code, bold) {
            (Some(code), Some(bold)) => Some(if code.0 <= bold.0 { code } else { bold }),
            (Some(code), None) => Some(code),
            (None, Some(bold)) => Some(bold),
            (None, None) => None,
        };
        let Some((start, end, marker)) = next else {
            spans.push(Span::styled(rest.to_string(), text_style));
            break;
        };
        if start > 0 {
            spans.push(Span::styled(rest[..start].to_string(), text_style));
        }
        let content_start = start + marker.len();
        let style = if marker == "**" {
            text_style.add_modifier(Modifier::BOLD)
        } else {
            code_style
        };
        spans.push(Span::styled(rest[content_start..end].to_string(), style));
        rest = &rest[end + marker.len()..];
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), text_style));
    }
    spans
}

fn report_prefix(
    source_row: usize,
    wrapped_row: usize,
    glyph: &str,
    glyph_style: Style,
) -> Span<'static> {
    if source_row == 0 && wrapped_row == 0 {
        Span::styled(format!("{glyph} "), glyph_style)
    } else {
        Span::raw("  ")
    }
}

fn render_report_lines(
    text: &str,
    value_column: usize,
    width: u16,
    glyph: &str,
    glyph_style: Style,
    text_style: Style,
    code_style: Style,
) -> Vec<Line<'static>> {
    let body_width = usize::from(width.saturating_sub(2)).max(1);
    let source_lines: Vec<&str> = if text.is_empty() {
        vec![""]
    } else {
        text.lines().collect()
    };
    let mut lines = Vec::new();

    for (source_row, body) in source_lines.into_iter().enumerate() {
        if value_column < body_width {
            if let Some((field, value)) = split_at_display_column(body, value_column) {
                let value_line = Line::from(report_inline_spans(value, text_style, code_style));
                let value_rows = wrap_line_to_cells(&value_line, body_width - value_column, true);
                for (wrapped_row, value_row) in value_rows.into_iter().enumerate() {
                    let mut spans = Vec::with_capacity(value_row.spans.len() + 2);
                    spans.push(report_prefix(source_row, wrapped_row, glyph, glyph_style));
                    if wrapped_row == 0 {
                        spans.push(Span::styled(field.to_string(), text_style));
                    } else {
                        spans.push(Span::raw(" ".repeat(value_column)));
                    }
                    spans.extend(value_row.spans);
                    lines.push(Line::from(spans));
                }
                continue;
            }
        }

        let body_line = Line::from(report_inline_spans(body, text_style, code_style));
        let body_rows = wrap_line_to_cells(&body_line, body_width, false);
        for (wrapped_row, body_row) in body_rows.into_iter().enumerate() {
            let mut spans = Vec::with_capacity(body_row.spans.len() + 1);
            spans.push(report_prefix(source_row, wrapped_row, glyph, glyph_style));
            spans.extend(body_row.spans);
            lines.push(Line::from(spans));
        }
    }

    lines
}

fn report_code_style(level: SystemLevel, text_style: Style, theme: &Theme) -> Style {
    if matches!(level, SystemLevel::Warn | SystemLevel::Error) {
        text_style
    } else {
        text_style.patch(Style::new().fg(theme.palette.cyan))
    }
}

pub(crate) fn rendered_lines(
    level: SystemLevel,
    text: &str,
    theme: &Theme,
    width: u16,
) -> Vec<Line<'static>> {
    if level == SystemLevel::Info && text.starts_with(PLAN_CHAPTER_PREFIX) {
        return rendered_chapter_lines(text, theme, width);
    }
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
        // The check glyph carries the whole signal, so the body stays
        // default-weight instead of tinting every line: unlike Warn/Error below,
        // a success is not something the eye must be pulled back to.
        SystemLevel::Success => (
            theme.palette.success,
            if nc { glyphs::CHECK_NC } else { glyphs::CHECK },
            Style::new().fg(theme.palette.fg),
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

    if let Some(value_column) = report_value_column(text) {
        let code_style = report_code_style(level, text_style, theme);
        return render_report_lines(
            text,
            value_column,
            width,
            glyph,
            glyph_style,
            text_style,
            code_style,
        );
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Flatten each rendered row back to plain text, one `String` per row.
    fn rows(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    /// The `/model` switch notice exactly as the session builds it.
    const MODEL_SWITCH_REPORT: &str = "Model updated\n  \
        Previous         gpt-5.6-sol\n  \
        Current          claude-fable-5\n  \
        Preserved msgs   0\n  \
        Delegation       smart-routed per role (this pin binds the main turn only)";

    /// Leading spaces on a rendered row.
    fn indent_of(row: &str) -> usize {
        row.len() - row.trim_start_matches(' ').len()
    }

    /// The live regression: a single backtick in one row used to route the whole
    /// report through the markdown paragraph renderer, whose soft breaks join
    /// every row into one run-on line. A report is laid out by hand, so one
    /// source row must stay one rendered row.
    #[test]
    fn report_rows_are_not_reflowed_into_one_paragraph() {
        let theme = Theme::default_dark();
        let report = format!(
            "{MODEL_SWITCH_REPORT}\n  Override         `/smart off` or `/smart pin <role> <model>`"
        );
        let rendered = rows(&rendered_lines(SystemLevel::Info, &report, &theme, 200));

        assert_eq!(
            rendered.len(),
            6,
            "one rendered row per source row: {rendered:?}"
        );
        assert!(
            rendered[1].contains("Previous         gpt-5.6-sol"),
            "{rendered:?}"
        );
        assert!(
            !rendered[1].contains("Current"),
            "rows must not merge into one paragraph: {rendered:?}"
        );
        assert!(
            rendered[5].contains("/smart off") && !rendered[5].contains('`'),
            "an inline-code hint keeps its text and drops its delimiters: {rendered:?}"
        );
    }

    /// A value that outruns the terminal wraps under the value column instead of
    /// falling back to column 0 — the second misalignment in the report.
    #[test]
    fn a_wrapped_value_hangs_under_the_value_column() {
        let theme = Theme::default_dark();
        let width = 46;
        let rendered = rows(&rendered_lines(
            SystemLevel::Info,
            MODEL_SWITCH_REPORT,
            &theme,
            width,
        ));

        let delegation = rendered
            .iter()
            .position(|row| row.contains("Delegation"))
            .expect("delegation row is rendered");
        let continuations = &rendered[delegation + 1..];
        assert!(
            !continuations.is_empty(),
            "the delegation value must wrap at width {width}: {rendered:?}"
        );
        // Glyph column (2) + `  Delegation       ` (19) = the value column.
        for row in continuations {
            assert!(
                indent_of(row) >= 21,
                "continuation must hang under the value column: {row:?}"
            );
        }
    }

    /// The row layout pre-wraps to the same width the painter wraps to, so the
    /// painter's wrap is a no-op. Otherwise the hanging indent would be broken
    /// again downstream and the height estimate would drift from the paint.
    #[test]
    fn pre_wrapped_report_rows_are_not_wrapped_again() {
        let theme = Theme::default_dark();
        let width = 46;
        let lines = rendered_lines(SystemLevel::Info, MODEL_SWITCH_REPORT, &theme, width);
        assert_eq!(
            crate::tui::blocks::wrap_rows(&lines, width).len(),
            lines.len(),
            "pre-wrapped rows must already fit the paint width"
        );
    }

    /// Genuine markdown keeps the markdown renderer: a bulleted list is not a
    /// column-aligned report and must not be routed to the row layout.
    #[test]
    fn markdown_notices_still_render_as_markdown() {
        let theme = Theme::default_dark();
        let text = "Available modes\n\n- read-only\n- workspace-write";
        let rendered = rows(&rendered_lines(SystemLevel::Info, text, &theme, 80)).join("\n");
        assert!(
            !rendered.contains("- read-only"),
            "the list marker should be rendered, not left literal: {rendered}"
        );
    }

    /// Authored markdown wins even when one indented line happens to resemble a
    /// report field. Otherwise the report detector would expose heading markers.
    #[test]
    fn authored_markdown_with_a_report_like_row_stays_markdown() {
        let theme = Theme::default_dark();
        let text = "# Status\n\n  Result           ready";
        let rendered = rows(&rendered_lines(SystemLevel::Info, text, &theme, 80)).join("\n");
        assert!(
            !rendered.contains("# Status"),
            "the heading marker should be rendered, not left literal: {rendered}"
        );
    }

    /// Ordered lists and blockquotes are authored markdown too. A report-like
    /// row elsewhere in the notice must not make the report router claim them.
    #[test]
    fn ordered_lists_and_blockquotes_do_not_use_report_layout() {
        for text in [
            "Steps\n\n1. first\n\n  Result           ready",
            "Context\n\n> note\n\n  Result           ready",
        ] {
            assert_eq!(report_value_column(text), None, "{text:?}");
        }
    }

    /// Paint through Ratatui's backend so the final terminal cell grid, not only
    /// the intermediate `Line`s, locks the report columns and measured height.
    #[test]
    fn report_draw_matches_estimate_and_keeps_value_columns_aligned() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let theme = Theme::default_dark();
        let width = 100;
        let report = format!(
            "{MODEL_SWITCH_REPORT}\n  Override         /smart off or /smart pin <role> <model>"
        );
        let height = estimate_rows(SystemLevel::Info, &report, &theme, width);
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| {
                draw(
                    frame,
                    Rect::new(0, 0, width, height),
                    SystemLevel::Info,
                    &report,
                    &theme,
                    0,
                );
            })
            .expect("draw report");

        let buffer = terminal.backend().buffer();
        let painted = (0..height)
            .map(|row| {
                (0..width)
                    .map(|column| buffer[(column, row)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>();
        assert_eq!(painted.len(), 6, "{painted:#?}");
        let value_columns = ["gpt-5.6-sol", "claude-fable-5", "0", "smart-routed", "/smart off"]
            .map(|value| {
                painted
                    .iter()
                    .find_map(|row| row.find(value))
                    .unwrap_or_else(|| panic!("missing {value:?} in {painted:#?}"))
            });
        assert!(
            value_columns.windows(2).all(|pair| pair[0] == pair[1]),
            "report values must share one painted column: {painted:#?}"
        );
    }

    /// A plan-step chapter renders as one accent-marked, bold heading row.
    #[test]
    fn a_plan_chapter_renders_one_bold_marked_row() {
        let theme = Theme::default_dark();
        let text = format!("{PLAN_CHAPTER_PREFIX}2/5 구현 결과를 검증하는 중");
        let lines = rendered_lines(SystemLevel::Info, &text, &theme, 120);
        let rendered = rows(&lines);
        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert_eq!(rendered[0], text, "{rendered:?}");
        assert_eq!(
            lines[0].spans[0].content.as_ref(),
            PLAN_CHAPTER_PREFIX,
            "the marker owns its own accent span: {:?}",
            lines[0].spans
        );
        assert_eq!(
            lines[0].spans[0].style,
            Style::new().fg(theme.palette.accent)
        );
        assert!(
            lines[0].spans[1..]
                .iter()
                .all(|span| span.style.add_modifier.contains(Modifier::BOLD)),
            "the sentence is bold: {:?}",
            lines[0].spans
        );
    }

    /// The chapter branch stays measure==paint, and a chapter sentence carrying
    /// backticks must not be re-routed through the markdown renderer.
    #[test]
    fn a_plan_chapter_measures_what_it_paints() {
        let theme = Theme::default_dark();
        for (text, width) in [
            (format!("{PLAN_CHAPTER_PREFIX}1/3 Wiring `hud::now_step` in"), 120),
            (format!("{PLAN_CHAPTER_PREFIX}1/3 Wiring `hud::now_step` in"), 24),
            (
                format!("{PLAN_CHAPTER_PREFIX}3/7 아주 긴 한국어 단계 문장을 좁은 폭에서 접는 경우"),
                28,
            ),
        ] {
            let lines = rendered_lines(SystemLevel::Info, &text, &theme, width);
            assert_eq!(
                u16::try_from(lines.len()).unwrap_or(u16::MAX),
                estimate_rows(SystemLevel::Info, &text, &theme, width),
                "estimate must equal the painted row count: {text:?} @ {width}"
            );
            let rendered = rows(&lines).join("\n");
            assert_eq!(
                rendered.contains('`'),
                text.contains('`'),
                "a chapter keeps its literal text, unrouted through markdown: {rendered:?}"
            );
        }
    }

    /// An ordinary Info notice is untouched by the chapter branch.
    #[test]
    fn a_non_chapter_info_notice_is_unchanged() {
        let theme = Theme::default_dark();
        let rendered = rows(&rendered_lines(
            SystemLevel::Info,
            "Model updated to claude-fable-5",
            &theme,
            80,
        ));
        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert!(rendered[0].starts_with(glyphs::INFO_CIRCLE), "{rendered:?}");
    }

    /// A one-line notice is unchanged: level glyph, one row.
    #[test]
    fn a_one_line_notice_keeps_its_level_glyph() {
        let theme = Theme::default_dark();
        let rendered = rows(&rendered_lines(
            SystemLevel::Warn,
            "Compacted conversation · 8 messages summarized",
            &theme,
            80,
        ));
        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert!(rendered[0].starts_with(glyphs::WARN_TRIANGLE), "{rendered:?}");
    }
}
