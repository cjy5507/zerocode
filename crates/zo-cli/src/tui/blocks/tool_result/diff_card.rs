//! `ToolResultBody::Diff` variant — the edit/diff summary line. The hunk body
//! itself renders through [`crate::tui::blocks::diff`]; only the tool-result
//! summary concerns live here.
//!
//! P10-B registry shape: the Diff-specific summary text/spans live here;
//! `mod.rs` keeps the exhaustive dispatch arms and the `diff::body_lines`
//! body delegation.

use ratatui::style::{Modifier, Style};
use ratatui::text::Span;

use runtime::message_stream::DiffView;

use crate::tui::theme::Theme;

use super::super::diff;

/// Summary-line text: `path (+adds -rems)`.
pub(super) fn summary(view: &DiffView) -> String {
    let (adds, rems) = diff::tally(view);
    format!("{} (+{adds} -{rems})", diff::path_label(view))
}

/// Summary-line spans mirror editor review headers: `path (+adds -rems)`.
pub(super) fn summary_spans(
    view: &DiffView,
    theme: &Theme,
    summary_style: Style,
) -> Vec<Span<'static>> {
    let (adds, rems) = diff::tally(view);
    let path_label = diff::path_label(view);
    let mut spans = vec![Span::styled(format!("{path_label} ("), summary_style)];

    let add_style = if theme.no_color {
        summary_style
    } else {
        Style::new()
            .fg(theme.palette.success)
            .add_modifier(Modifier::BOLD)
    };
    let rem_style = if theme.no_color {
        summary_style
    } else {
        Style::new()
            .fg(theme.palette.error)
            .add_modifier(Modifier::BOLD)
    };

    spans.push(Span::styled(format!("+{adds}"), add_style));
    spans.push(Span::styled(" ", summary_style));
    spans.push(Span::styled(format!("-{rems}"), rem_style));
    spans.push(Span::styled(")", summary_style));
    spans
}
