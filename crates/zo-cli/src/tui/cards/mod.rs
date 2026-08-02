//! Rich command-output **card** renderer.
//!
//! Turns a provider-neutral [`core_types::CardModel`] into a bordered,
//! gauged `ratatui` panel — the reusable widget behind every redesigned
//! slash command (`/status`, `/cost`, `/context`, …), giving them the
//! polished, `/effort`-grade look from one place.
//!
//! Single responsibility: `CardModel` (data) → styled `Line`s (view). All
//! color goes through `&Theme` (code-rules R9); semantic [`CardTone`] and
//! gauge ratios resolve to theme colors so `NO_COLOR` and every palette
//! degrade consistently (R10). Gauge ratios are clamped, so an
//! over-budget value can never panic.

use core_types::{CardElement, CardModel, CardTone};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph};
use unicode_width::UnicodeWidthStr;

mod frame;
pub use frame::{CardFrame, SurfaceKind};

use super::glyphs;
use super::theme::Theme;

/// Horizontal padding inside the card border (each side).
const PAD_X: u16 = 1;
/// Border rows (top + bottom).
const BORDER_ROWS: u16 = 2;
/// Minimum inner content width before we stop trying to right-align.
const MIN_INNER: u16 = 8;

/// Draw `card` into `area` as a bordered panel.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    card: &CardModel,
    theme: &Theme,
    scroll_offset: u16,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let inner = CardFrame::new(SurfaceKind::Card, theme)
        .title(Line::styled(card.title.clone(), theme.typography.heading_1))
        .padding(Padding::horizontal(PAD_X))
        .render(frame, area);
    let lines = render_lines(card, theme, inner.width);
    let para = Paragraph::new(lines)
        .style(theme.typography.body)
        .scroll((scroll_offset, 0));
    frame.render_widget(para, inner);
}

/// Estimate the rendered height (rows) of `card` at total block `width`.
///
/// Mirrors [`draw`] exactly so the transcript reserves the right height:
/// inner content lines + the two border rows.
#[must_use]
pub fn estimate_rows(card: &CardModel, theme: &Theme, width: u16) -> u16 {
    let inner_width = width.saturating_sub(BORDER_ROWS + PAD_X * 2);
    let lines = render_lines(card, theme, inner_width);
    u16::try_from(lines.len())
        .unwrap_or(u16::MAX)
        .saturating_add(BORDER_ROWS)
        .max(BORDER_ROWS + 1)
}

/// Render the card body into styled lines laid out against `width`
/// (the padded inner width).
#[must_use]
pub fn render_lines(card: &CardModel, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if let Some(subtitle) = &card.subtitle {
        lines.push(Line::from(Span::styled(
            subtitle.clone(),
            theme.typography.dim,
        )));
        lines.push(Line::from(""));
    }
    for element in &card.elements {
        push_element(&mut lines, element, theme, width);
    }
    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

fn push_element(lines: &mut Vec<Line<'static>>, el: &CardElement, theme: &Theme, width: u16) {
    match el {
        CardElement::Section { label } => {
            if !lines.last().is_some_and(|l| l.spans.is_empty()) && !lines.is_empty() {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(Span::styled(
                label.clone(),
                theme.typography.heading_2,
            )));
        }
        CardElement::Metric { label, value, tone } => {
            lines.push(metric_line(label, value, *tone, theme, width));
        }
        CardElement::Gauge {
            label,
            ratio,
            caption,
        } => {
            lines.push(gauge_line(label, *ratio, caption, theme, width));
        }
        CardElement::Table { header, rows } => {
            push_table(lines, header, rows, theme, width);
        }
        CardElement::Badge { ok, text } => {
            lines.push(badge_line(*ok, text, theme));
        }
        CardElement::KeyValue { key, value } => {
            lines.push(Line::from(vec![
                Span::styled(format!("{key}  "), theme.typography.dim),
                Span::styled(value.clone(), theme.typography.body),
            ]));
        }
        CardElement::Text { text, tone } => {
            lines.push(Line::from(Span::styled(
                text.clone(),
                Style::new().fg(tone_color(theme, *tone)),
            )));
        }
        CardElement::Spacer => lines.push(Line::from("")),
    }
}

/// `label ……… value` with the value right-aligned to `width`.
fn metric_line(
    label: &str,
    value: &str,
    tone: CardTone,
    theme: &Theme,
    width: u16,
) -> Line<'static> {
    let w = usize::from(width.max(MIN_INNER));
    let label_w = UnicodeWidthStr::width(label);
    let value_w = UnicodeWidthStr::width(value);
    let gap = w.saturating_sub(label_w + value_w).max(1);
    Line::from(vec![
        Span::styled(label.to_string(), theme.typography.dim),
        Span::raw(" ".repeat(gap)),
        Span::styled(value.to_string(), Style::new().fg(tone_color(theme, tone))),
    ])
}

/// `label  [████░░░░]  caption` — a single-line gauge tinted by ratio.
fn gauge_line(label: &str, ratio: f64, caption: &str, theme: &Theme, width: u16) -> Line<'static> {
    let w = usize::from(width.max(MIN_INNER));
    // Reserve label + caption + the two brackets + spacing, give the rest
    // to the bar (min 4 cells).
    let label_w = UnicodeWidthStr::width(label);
    let caption_w = UnicodeWidthStr::width(caption);
    let overhead = label_w + caption_w + 6; // two spaces + "[]" + a space
    let bar_w = w.saturating_sub(overhead).max(4);
    let clamped = ratio.clamp(0.0, 1.0);
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let filled = (clamped * bar_w as f64).round() as usize;
    let filled = filled.min(bar_w);
    let empty = bar_w - filled;
    let fill_color = theme.metric_color(clamped);
    let color = !theme.no_color;
    Line::from(vec![
        Span::styled(format!("{label}  "), theme.typography.dim),
        Span::styled(
            glyphs::card_gauge_fill(color).repeat(filled),
            Style::new().fg(fill_color),
        ),
        Span::styled(
            glyphs::card_gauge_empty(color).repeat(empty),
            Style::new().fg(theme.palette.faint),
        ),
        Span::styled(format!("  {caption}"), theme.typography.dim),
    ])
}

fn badge_line(ok: bool, text: &str, theme: &Theme) -> Line<'static> {
    let (glyph, color) = if ok {
        (
            if theme.no_color { "[ok]" } else { "✓" },
            theme.palette.success,
        )
    } else {
        (
            if theme.no_color { "[x]" } else { "✗" },
            theme.palette.error,
        )
    };
    Line::from(vec![
        Span::styled(
            format!("{glyph} "),
            Style::new().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(text.to_string(), theme.typography.body),
    ])
}

/// Render a header + body table, columns sized to content and fit to `width`.
fn push_table(
    lines: &mut Vec<Line<'static>>,
    header: &[String],
    rows: &[Vec<String>],
    theme: &Theme,
    width: u16,
) {
    let cols = header
        .len()
        .max(rows.iter().map(Vec::len).max().unwrap_or(0));
    if cols == 0 {
        return;
    }
    let mut widths = vec![0usize; cols];
    let mut consider = |row: &[String]| {
        for (i, cell) in row.iter().enumerate() {
            if i < cols {
                widths[i] = widths[i].max(UnicodeWidthStr::width(cell.as_str()));
            }
        }
    };
    consider(header);
    for row in rows {
        consider(row);
    }
    // Fit: shrink the widest column while the total (cols separated by 2
    // spaces) exceeds the available width.
    let sep = 2usize;
    let avail = usize::from(width.max(MIN_INNER));
    while widths.iter().sum::<usize>() + sep * cols.saturating_sub(1) > avail {
        let Some((idx, _)) = widths.iter().enumerate().max_by_key(|(_, w)| **w) else {
            break;
        };
        if widths[idx] <= 4 {
            break;
        }
        widths[idx] -= 1;
    }
    // A header row is only drawn when it actually carries labels. Cards
    // that use a table purely for *aligned* rows (e.g. `/help`) pass an
    // empty header to skip the heading line entirely.
    if header.iter().any(|cell| !cell.is_empty()) {
        lines.push(table_row(
            header,
            &widths,
            theme.table_header_style(),
            theme,
            true,
        ));
    }
    for row in rows {
        lines.push(table_row(row, &widths, theme.typography.body, theme, false));
    }
}

fn table_row(
    cells: &[String],
    widths: &[usize],
    style: Style,
    theme: &Theme,
    _header: bool,
) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(widths.len() * 2);
    for (i, w) in widths.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("  "));
        }
        let raw = cells.get(i).map_or("", String::as_str);
        let cell = truncate_to_width(raw, *w);
        let pad = w.saturating_sub(UnicodeWidthStr::width(cell.as_str()));
        spans.push(Span::styled(cell, style));
        if pad > 0 {
            spans.push(Span::styled(" ".repeat(pad), theme.typography.body));
        }
    }
    Line::from(spans)
}

/// Truncate `s` to at most `width` display cells, adding `…` if cut.
/// Truncate `s` to at most `width` display cells (Unicode-width aware), adding a
/// trailing `…` when it had to cut. `width == 0` yields an empty string. Shared by
/// the card grid and the collapsed tool-group rows so neither hard-clips a long
/// path at the terminal edge.
pub(crate) fn truncate_to_width(s: &str, width: usize) -> String {
    if UnicodeWidthStr::width(s) <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut acc = 0usize;
    let mut out = String::new();
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if acc + cw > width.saturating_sub(1) {
            break;
        }
        out.push(ch);
        acc += cw;
    }
    out.push('\u{2026}');
    out
}

fn tone_color(theme: &Theme, tone: CardTone) -> ratatui::style::Color {
    match tone {
        CardTone::Default => theme.palette.fg,
        CardTone::Accent => theme.palette.accent,
        CardTone::Ok => theme.palette.success,
        CardTone::Warn => theme.palette.warn,
        CardTone::Crit => theme.palette.error,
        CardTone::Muted => theme.palette.dim,
    }
}

#[cfg(test)]
mod tests {
    use super::{draw, estimate_rows, render_lines};
    use crate::tui::theme::Theme;
    use core_types::{CardModel, CardTone};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    fn sample() -> CardModel {
        CardModel::new(" /status ")
            .section("Session")
            .key_value("model", "Opus 4.8")
            .gauge("context", 0.6, "120k / 200k · 60%")
            .metric("cost", "$1.23", CardTone::Warn)
            .table(
                vec!["server".into(), "status".into()],
                vec![vec!["fs".into(), "ok".into()]],
            )
            .badge(true, "API key set")
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn estimate_matches_drawn_line_count() {
        let theme = Theme::zo();
        let card = sample();
        // estimate = inner lines + 2 borders.
        let inner = render_lines(&card, &theme, 60 - 4);
        assert_eq!(estimate_rows(&card, &theme, 60), inner.len() as u16 + 2);
    }

    #[test]
    fn gauge_ratio_over_one_does_not_panic_and_clamps() {
        let theme = Theme::zo();
        let card = CardModel::new("t").gauge("ctx", 5.0, "over");
        let lines = render_lines(&card, &theme, 50);
        // crit color for an over-budget (clamped to 1.0) gauge.
        let bar = lines
            .iter()
            .flat_map(|l| &l.spans)
            .find(|s| s.content.contains('█'))
            .expect("gauge bar");
        assert_eq!(bar.style.fg, Some(theme.palette.error));
    }

    #[test]
    fn draws_without_panic_under_no_color_and_narrow_width() {
        let theme = Theme::no_color();
        let card = sample();
        for w in [10u16, 24, 60] {
            let backend = TestBackend::new(w, 20);
            let mut term = Terminal::new(backend).expect("backend");
            term.draw(|f| draw(f, Rect::new(0, 0, w, 20), &card, &theme, 0))
                .expect("draw");
        }
    }

    #[test]
    fn metric_value_is_right_aligned() {
        let theme = Theme::zo();
        let card = CardModel::new("t").metric("label", "VAL", CardTone::Default);
        let lines = render_lines(&card, &theme, 40);
        let joined: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(joined.starts_with("label"));
        assert!(joined.trim_end().ends_with("VAL"));
        assert_eq!(joined.chars().count(), 40, "row padded to inner width");
    }

    /// Flatten a rendered card into one string for glyph inspection.
    fn card_text(card: &CardModel, theme: &Theme, width: u16) -> String {
        render_lines(card, theme, width)
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect()
    }

    /// Normal/color mode keeps the Unicode block gauge (`█`/`░`); the ASCII
    /// fallbacks never leak into the rich render.
    #[test]
    fn gauge_keeps_unicode_blocks_in_color_mode() {
        let theme = Theme::zo();
        let card = CardModel::new("t").gauge("ctx", 0.6, "60%");
        let text = card_text(&card, &theme, 40);
        assert!(text.contains('█'), "color gauge keeps the filled block");
        assert!(text.contains('░'), "color gauge keeps the empty block");
        // The ASCII fallbacks must not appear as gauge cells in rich mode.
        assert!(!text.contains('#'), "no ASCII fill leak in color mode: {text:?}");
    }

    /// Plain/`NO_COLOR` mode swaps the block gauge for one-cell ASCII: filled
    /// `#`, empty `-`, and never the Unicode blocks.
    #[test]
    fn gauge_uses_ascii_fallback_under_no_color() {
        let theme = Theme::no_color();
        let card = CardModel::new("t").gauge("ctx", 0.6, "60%");
        let text = card_text(&card, &theme, 40);
        assert!(text.contains('#'), "plain gauge filled cell is '#': {text:?}");
        assert!(text.contains('-'), "plain gauge empty cell is '-': {text:?}");
        assert!(!text.contains('█'), "no Unicode fill under NO_COLOR");
        assert!(!text.contains('░'), "no Unicode empty under NO_COLOR");
    }
}
