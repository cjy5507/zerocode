//! `ToolResultBody::Todos` variant — the bordered `Updated Plan` checklist
//! card for `TodoWrite` / `TaskList` results.
//!
//! P10-B registry shape: every Todos-specific rendering concern lives here;
//! `mod.rs` keeps only the exhaustive dispatch arms (one-line delegation).

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use runtime::message_stream::{TodoResultItem, TodoResultStatus};

use crate::tui::text_metrics::display_width;
use crate::tui::theme::Theme;

use super::super::sanitize_inline;

/// Summary-line text: the completion tally (`2/3 done`, `all done`).
pub(super) fn summary(items: &[TodoResultItem]) -> String {
    let total = items.len();
    let done = items
        .iter()
        .filter(|item| item.status == TodoResultStatus::Completed)
        .count();
    if total == 0 {
        "all done".to_string()
    } else {
        format!("{done}/{total} done")
    }
}

/// The full checklist card: `╭ Updated Plan · N/M done ╮` header, one
/// bordered row per item, closing border. Replaces the generic summary/body
/// frame entirely (early return in `rendered_lines`).
pub(super) fn block_lines<'a>(items: &'a [TodoResultItem], theme: &Theme) -> Vec<Line<'a>> {
    let border_style = Style::new().fg(theme.palette.dim);
    let title_style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::BOLD);
    let total = items.len();
    let done = items
        .iter()
        .filter(|item| item.status == TodoResultStatus::Completed)
        .count();
    let tally_style = if done == total {
        Style::new()
            .fg(theme.palette.success)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(theme.palette.dim)
    };
    // Same text as the collapsed summary line — one tally, two surfaces.
    let tally_text = summary(items);
    let title = "Updated Plan";
    let header_width = display_width(title) + display_width("  ·  ") + display_width(&tally_text);
    let row_width = items
        .iter()
        .map(|item| {
            let (marker, _) = marker(item.status, theme);
            let text = row_text(item);
            display_width(marker) + 1 + display_width(&sanitize_inline(text))
        })
        .max()
        .unwrap_or_else(|| display_width("all done"));
    let inner_width = header_width
        .max(row_width)
        .max(display_width("Updated Plan"));
    let mut lines: Vec<Line<'_>> = Vec::with_capacity(items.len() + 3);

    let h = if theme.no_color { '-' } else { '─' };
    let (tl, tr, bl, br, v) = if theme.no_color {
        ('+', '+', '+', '+', '|')
    } else {
        ('╭', '╮', '╰', '╯', '│')
    };

    let header_pad = inner_width.saturating_sub(header_width);
    lines.push(Line::from(vec![
        Span::styled(format!("{tl} "), border_style),
        Span::styled(title.to_string(), title_style),
        Span::styled("  ·  ".to_string(), border_style),
        Span::styled(tally_text, tally_style),
        Span::styled(" ".repeat(header_pad + 1), border_style),
        Span::styled(tr.to_string(), border_style),
    ]));

    if items.is_empty() {
        lines.push(bordered_row(
            v,
            vec![Span::styled("all done".to_string(), tally_style)],
            display_width("all done"),
            inner_width,
            border_style,
        ));
    } else {
        for item in items {
            let (marker, marker_style) = marker(item.status, theme);
            let text = sanitize_inline(row_text(item));
            let text_style = match item.status {
                TodoResultStatus::Completed => Style::new()
                    .fg(theme.palette.dim)
                    .add_modifier(Modifier::CROSSED_OUT),
                TodoResultStatus::InProgress => Style::new()
                    .fg(theme.palette.fg)
                    .add_modifier(Modifier::BOLD),
                TodoResultStatus::Pending => Style::new().fg(theme.palette.fg),
            };
            let content_width = display_width(marker) + 1 + display_width(&text);
            lines.push(bordered_row(
                v,
                vec![
                    Span::styled(marker.to_string(), marker_style),
                    Span::raw(" "),
                    Span::styled(text, text_style),
                ],
                content_width,
                inner_width,
                border_style,
            ));
        }
    }

    lines.push(Line::styled(
        format!("{bl}{}{br}", h.to_string().repeat(inner_width + 2)),
        border_style,
    ));
    lines
}

/// The text shown for one row: the present-progressive `active_form` while
/// in progress (when non-empty), the imperative `content` otherwise.
fn row_text(item: &TodoResultItem) -> &str {
    if item.status == TodoResultStatus::InProgress && !item.active_form.trim().is_empty() {
        item.active_form.as_str()
    } else {
        item.content.as_str()
    }
}

fn bordered_row(
    border: char,
    mut content: Vec<Span<'_>>,
    content_width: usize,
    inner_width: usize,
    border_style: Style,
) -> Line<'_> {
    let mut spans = vec![Span::styled(format!("{border} "), border_style)];
    spans.append(&mut content);
    spans.push(Span::styled(
        " ".repeat(inner_width.saturating_sub(content_width) + 1),
        border_style,
    ));
    spans.push(Span::styled(border.to_string(), border_style));
    Line::from(spans)
}

/// Marker glyph + style for one todo checklist row. ASCII boxes under
/// `no_color`, filled glyphs otherwise; in-progress is warn-colored so the
/// active step stands out, completed is success-colored.
fn marker(status: TodoResultStatus, theme: &Theme) -> (&'static str, Style) {
    if theme.no_color {
        match status {
            TodoResultStatus::Pending => ("[ ]", Style::new().fg(theme.palette.dim)),
            TodoResultStatus::InProgress => ("[~]", Style::new().fg(theme.palette.warn)),
            TodoResultStatus::Completed => ("[x]", Style::new().fg(theme.palette.success)),
        }
    } else {
        match status {
            TodoResultStatus::Pending => ("\u{2610}", Style::new().fg(theme.palette.dim)),
            TodoResultStatus::InProgress => ("\u{25d0}", Style::new().fg(theme.palette.warn)),
            TodoResultStatus::Completed => ("\u{2611}", Style::new().fg(theme.palette.success)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn todo(content: &str, active: &str, status: TodoResultStatus) -> TodoResultItem {
        TodoResultItem {
            content: content.to_string(),
            active_form: active.to_string(),
            status,
        }
    }

    /// The block renderer produces a bordered `Updated Plan · N/M done` header
    /// in-progress via its active form, never raw JSON.
    #[test]
    fn todos_block_lines_renders_titled_checklist() {
        let items = vec![
            todo(
                "Wire the parser",
                "Wiring the parser",
                TodoResultStatus::Completed,
            ),
            todo(
                "Render the block",
                "Rendering the block",
                TodoResultStatus::InProgress,
            ),
            todo("Add tests", "Adding tests", TodoResultStatus::Pending),
        ];
        let lines = block_lines(&items, &Theme::no_color());
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();

        assert!(text.contains("Updated Plan"), "titled header: {text}");
        assert!(text.contains("1/3 done"), "done tally: {text}");
        assert!(text.contains("Rendering the block"), "active form: {text}");
        assert!(
            text.contains("Wire the parser"),
            "completed content: {text}"
        );
        assert!(text.contains("Add tests"), "pending content: {text}");
        assert!(text.contains("[x]") && text.contains("[~]") && text.contains("[ ]"));
        assert!(
            !text.contains('{') && !text.contains("newTodos"),
            "no raw JSON: {text}"
        );
    }

    /// An all-completed list still reads with a `N/N done` tally in the block.
    #[test]
    fn todos_block_lines_all_completed_shows_full_tally() {
        let items = vec![todo("Ship it", "Shipping it", TodoResultStatus::Completed)];
        let lines = block_lines(&items, &Theme::no_color());
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(text.contains("1/1 done"), "tally for all-complete: {text}");
    }
}
