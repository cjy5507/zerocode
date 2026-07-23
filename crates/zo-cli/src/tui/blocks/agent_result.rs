//! Renderer for [`RenderBlock::AgentResult`] — a finished background sub-agent's
//! result, shown as a bordered, collapsible "agent result" card instead of an
//! amber `You` user message.
//!
//! ## Why a card, not a user message
//!
//! A re-injected agent completion is a user-role turn *under the hood* (the main
//! model must read the body to continue), but its visual author is the agent.
//! Rendering it as a `┃  You` message was doubly wrong: it mislabels the author,
//! and it dumps the full markdown body with no collapse — a 200-line agent
//! result becomes an unbroken wall. This card gives the result its own identity
//! (a teal `⎔ agent` rail glyph) and collapses by default: header + one preview
//! line, body one keystroke away, mirroring the [`super::tool_result`] card.
//!
//! ## Structure (clean-code)
//!
//! Rendering is a pipeline of small single-purpose helpers over one
//! [`AgentCardView`] parameter object (so no renderer takes a long positional
//! argument list): [`header_line`] → [`preview_line`] / [`body_lines`], composed
//! by [`rendered_lines_for_width`]. Layout thresholds are named constants, never
//! inline magic numbers. The public [`draw`] / [`estimate_rows`] pair matches the
//! `tool_result` renderer so this block flows through the transcript's cached
//! draw + height-prefix paths with identical semantics.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Padding, Paragraph, Wrap};
use runtime::message_stream::AgentResultStatus;

use crate::tui::cards::{CardFrame, SurfaceKind};
use crate::tui::glyphs;
use crate::tui::theme::Theme;

/// Agent results with more than this many body lines collapse by default,
/// showing only the header + a one-line preview. Matches the spirit of
/// [`super::tool_result`]'s collapse threshold: short results render in full,
/// long ones fold. A background scout report is almost always well over this,
/// which is exactly the wall we are taming.
const COLLAPSE_LINE_THRESHOLD: usize = 8;

/// Upper bound on body lines an *expanded* card renders. Expanding must show the
/// whole result, but an unbounded body would make the transcript's per-block
/// height prefix-sum scale with the full line count (the same stall the tool
/// card's `EXPANDED_HARD_CAP` prevents). Clip at a generous ceiling and append a
/// "showing first N of M" notice.
const EXPANDED_HARD_CAP: usize = 2000;

/// Horizontal padding inside the card border, in cells per side. One column of
/// breathing room so the body never touches the border rule.
const CARD_HORIZONTAL_PADDING: u16 = 1;

/// Cells the border + padding consume horizontally (`│` + pad on each side), so
/// the body wraps to the true inner width and the height estimate matches the
/// draw exactly.
const CARD_HORIZONTAL_CHROME: u16 = 2 * (1 + CARD_HORIZONTAL_PADDING);

/// Chevron shown on the header when the card is collapsed (body hidden).
const CHEVRON_COLLAPSED: &str = "\u{25b8}";
/// Chevron shown on the header when the card is expanded (body visible).
const CHEVRON_EXPANDED: &str = "\u{25be}";

/// Everything the card renderer needs about one finished agent result, gathered
/// into a single parameter object so the helper functions take one argument
/// instead of a long positional list (clean-code: few arguments, high cohesion).
pub(crate) struct AgentCardView<'a> {
    /// Sub-agent display label, e.g. `runtime-scout`.
    pub label: &'a str,
    /// Completion status driving the header glyph / tint.
    pub status: AgentResultStatus,
    /// The agent's raw result markdown.
    pub body: &'a str,
    /// Whether the user has expanded this card (body shown).
    pub expanded: bool,
    /// Whether this card currently holds keyboard focus (drives the accent
    /// border so the arrow-key/Enter target is obvious).
    pub focused: bool,
}

impl AgentCardView<'_> {
    /// Number of non-empty-trimmed body lines. Drives both the header count and
    /// the collapse decision. Computed from the raw body (not the wrapped
    /// render) so it is stable across terminal widths.
    fn body_line_count(&self) -> usize {
        self.body.lines().count()
    }

    /// A collapsed card shows only the header + preview; an expanded one shows
    /// the full (capped) body. Short results never collapse — folding a
    /// three-line result would add friction with no payoff.
    fn is_collapsed(&self) -> bool {
        !self.expanded && self.body_line_count() > COLLAPSE_LINE_THRESHOLD
    }
}

/// Glyph + tint for a completion status: `✓` success (teal) or `✕` failure
/// (error red). Degrades to `[OK]` / `[FAIL]` under NO_COLOR so the outcome
/// survives on a monochrome terminal (text, not hue).
fn status_badge(status: AgentResultStatus, theme: &Theme) -> (String, Style) {
    let color = !theme.no_color;
    match status {
        AgentResultStatus::Completed => (
            glyphs::pick(color, "\u{2713}", "[OK]").to_string(),
            Style::new().fg(theme.palette.success),
        ),
        AgentResultStatus::Failed => (
            glyphs::pick(color, "\u{2717}", "[FAIL]").to_string(),
            Style::new().fg(theme.palette.error),
        ),
    }
}

/// The header line: `⎔ agent · <label> ✓ · 214 lines   ▸`.
///
/// The `⎔` diamond + teal `agent` word is the author identity that replaces the
/// amber `You` rail. The trailing chevron mirrors the tool card's expand
/// affordance. Under NO_COLOR the diamond degrades to `*` and the styles fall
/// back to plain attributes.
fn header_line(view: &AgentCardView<'_>, theme: &Theme) -> Line<'static> {
    let color = !theme.no_color;
    let (badge, badge_style) = status_badge(view.status, theme);
    let author_style = Style::new()
        .fg(theme.palette.teal)
        .add_modifier(Modifier::BOLD);
    let dim = Style::new().fg(theme.palette.dim);
    let line_count = view.body_line_count();
    let chevron = if view.is_collapsed() {
        CHEVRON_COLLAPSED
    } else {
        CHEVRON_EXPANDED
    };

    Line::from(vec![
        Span::styled(
            format!("{} agent", glyphs::pick(color, "\u{2394}", "*")),
            author_style,
        ),
        Span::styled(" · ", dim),
        Span::styled(
            view.label.to_string(),
            Style::new().fg(theme.palette.fg).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" ", dim),
        Span::styled(badge, badge_style),
        Span::styled(format!("  ·  {line_count} lines  "), dim),
        Span::styled(chevron.to_string(), dim),
    ])
}

/// First non-blank body line, trimmed to a single wrapped row, shown under a
/// collapsed header so the user can scan the result without expanding. A blank
/// body yields a dim `(no output)` marker rather than an empty preview.
fn preview_line(view: &AgentCardView<'_>, theme: &Theme) -> Line<'static> {
    let first = view
        .body
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("(no output)");
    Line::from(Span::styled(
        first.to_string(),
        Style::new().fg(theme.palette.dim),
    ))
}

/// The expanded body, rendered through the shared markdown engine at
/// `inner_width` (headings, emphasis, links, lists) and capped at
/// [`EXPANDED_HARD_CAP`] source lines. A clip notice is appended only when the
/// body actually exceeds the cap, so the row count the draw produces matches the
/// height estimate exactly.
fn body_lines(view: &AgentCardView<'_>, theme: &Theme, inner_width: u16) -> Vec<Line<'static>> {
    let total = view.body_line_count();
    let capped: String = if total > EXPANDED_HARD_CAP {
        view.body
            .lines()
            .take(EXPANDED_HARD_CAP)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        view.body.to_string()
    };

    let mut lines = crate::tui::markdown::rendered_lines_for_width(&capped, theme, inner_width);
    if total > EXPANDED_HARD_CAP {
        lines.push(Line::styled(
            format!("… showing first {EXPANDED_HARD_CAP} of {total}"),
            Style::new()
                .fg(theme.palette.dim)
                .add_modifier(Modifier::ITALIC),
        ));
    }
    lines
}

/// Compose the card's inner content (header + preview or body) for a given
/// content width. This is the single source both [`draw`] and [`estimate_rows`]
/// consume, so the painted rows and the measured height can never diverge.
fn rendered_lines_for_width(view: &AgentCardView<'_>, theme: &Theme, inner_width: u16) -> Vec<Line<'static>> {
    let mut lines = vec![header_line(view, theme)];
    if view.is_collapsed() {
        lines.push(preview_line(view, theme));
    } else {
        lines.extend(body_lines(view, theme, inner_width));
    }
    lines
}

/// The bordered container. Rounded when focused-or-not is conveyed by tint:
/// accent border on focus (the arrow-key/Enter target), dim otherwise. Uses
/// ratatui's [`Block`] inner-area model so the body is inset by border+padding
/// rather than hand-prefixed with a rail.
fn card_block(focused: bool, theme: &Theme) -> Block<'_> {
    let border_style = if focused {
        Style::new().fg(theme.palette.accent)
    } else {
        Style::new().fg(theme.palette.dim)
    };
    CardFrame::new(SurfaceKind::Card, theme)
        .border_style(border_style)
        .padding(Padding::horizontal(CARD_HORIZONTAL_PADDING))
        .block()
}

/// Draw a finished agent-result card into `area`.
pub(crate) fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    view: &AgentCardView<'_>,
    theme: &Theme,
    scroll_offset: u16,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let inner_width = area.width.saturating_sub(CARD_HORIZONTAL_CHROME);
    let lines = rendered_lines_for_width(view, theme, inner_width);
    let para = Paragraph::new(lines)
        .block(card_block(view.focused, theme))
        .style(theme.typography.body)
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));
    frame.render_widget(para, area);
}

/// Rows this card occupies at `width`, including the top+bottom border (2). Must
/// match [`draw`]'s output exactly so the transcript height prefix-sum stays
/// correct; both go through [`rendered_lines_for_width`] at the same inner width.
pub(crate) fn estimate_rows(view: &AgentCardView<'_>, theme: &Theme, width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let inner_width = width.saturating_sub(CARD_HORIZONTAL_CHROME).max(1);
    let content = rendered_lines_for_width(view, theme, inner_width);
    let body_rows = super::wrapped_rows(&content, inner_width);
    // +2 for the top and bottom border rules drawn by `card_block`.
    body_rows.saturating_add(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::Theme;

    /// Flatten a rendered line's spans into plain text for assertions.
    fn flatten(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn view(body: &str, expanded: bool) -> AgentCardView<'_> {
        AgentCardView {
            label: "runtime-scout",
            status: AgentResultStatus::Completed,
            body,
            expanded,
            focused: false,
        }
    }

    #[test]
    fn long_result_collapses_to_header_plus_preview() {
        let theme = Theme::default_dark();
        let body = (1..=40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let collapsed = rendered_lines_for_width(&view(&body, false), &theme, 60);
        // Header + exactly one preview line — the 40-line wall is folded away.
        assert_eq!(collapsed.len(), 2, "collapsed = header + preview only");
        let text = flatten(&collapsed);
        assert!(text.contains("agent"), "header shows agent author: {text}");
        assert!(text.contains("runtime-scout"), "header shows label: {text}");
        assert!(text.contains("40 lines"), "header shows line count: {text}");
        assert!(text.contains("line 1"), "preview shows first line: {text}");
        assert!(!text.contains("line 2"), "collapsed hides the rest: {text}");
    }

    #[test]
    fn expanding_reveals_the_full_body() {
        let theme = Theme::default_dark();
        let body = (1..=40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let expanded = rendered_lines_for_width(&view(&body, true), &theme, 60);
        let text = flatten(&expanded);
        assert!(text.contains("line 1") && text.contains("line 40"), "expanded shows all lines");
        // Expanded is much taller than the 2-line collapsed form.
        let collapsed = rendered_lines_for_width(&view(&body, false), &theme, 60);
        assert!(
            expanded.len() > collapsed.len(),
            "expanded ({}) must be taller than collapsed ({})",
            expanded.len(),
            collapsed.len()
        );
    }

    #[test]
    fn short_result_never_collapses() {
        let theme = Theme::default_dark();
        // Below COLLAPSE_LINE_THRESHOLD → shown in full even when not expanded.
        let body = "one\ntwo\nthree";
        assert!(!view(body, false).is_collapsed());
        let lines = rendered_lines_for_width(&view(body, false), &theme, 60);
        assert!(flatten(&lines).contains("three"), "short result shown whole");
    }

    #[test]
    fn no_color_degrades_glyphs_to_text() {
        let theme = Theme::no_color();
        let body = (1..=20).map(|i| i.to_string()).collect::<Vec<_>>().join("\n");
        let header = flatten(&[header_line(&view(&body, false), &theme)]);
        // The diamond author glyph and the ✓ badge degrade to ASCII.
        assert!(header.contains("* agent"), "author glyph → '*': {header}");
        assert!(header.contains("[OK]"), "status badge → [OK]: {header}");
        assert!(!header.contains('\u{2394}') && !header.contains('\u{2713}'));
    }

    #[test]
    fn failed_status_shows_failure_badge() {
        let theme = Theme::default_dark();
        let v = AgentCardView {
            label: "api-scout",
            status: AgentResultStatus::Failed,
            body: "boom",
            expanded: false,
            focused: false,
        };
        let header = flatten(&[header_line(&v, &theme)]);
        assert!(header.contains('\u{2717}'), "failed → ✕ badge: {header}");
    }

    #[test]
    fn estimate_rows_matches_rendered_line_count_plus_border() {
        let theme = Theme::default_dark();
        // Short body, no wrapping at width 60 → rows == content lines + 2 border.
        let body = "one\ntwo\nthree";
        let v = view(body, false);
        let inner = 60u16.saturating_sub(CARD_HORIZONTAL_CHROME).max(1);
        let content = rendered_lines_for_width(&v, &theme, inner);
        assert_eq!(
            estimate_rows(&v, &theme, 60),
            u16::try_from(content.len()).unwrap() + 2,
            "height = content rows + top/bottom border"
        );
    }
}
