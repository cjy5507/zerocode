//! Chart primitives for the modal dashboards.
//!
//! These return `Vec<Line<'static>>` instead of drawing into a `Rect`. The
//! dashboards that consume them already compose a `Paragraph` out of styled
//! rows, so handing back rows keeps the widget tree shallow — the same reason
//! `sidebar::inline_sparkline` embeds glyphs rather than reaching for
//! `ratatui::widgets::Sparkline`.
//!
//! **Glyph budget.** Every character emitted here is East-Asian *Neutral*
//! (`width_cjk() == 1`). Braille (U+2800–U+28FF) qualifies and carries 2×4 dots
//! per cell, which is eight vertical levels out of a single text row — that
//! resolution is why the area chart is drawn with it. The familiar ramp
//! `▁▂▃▄▅▆▇█` (U+2580–U+259F) is *Ambiguous*: under a `ko_KR` wide-ambiguous
//! tmux every glyph paints two columns and the chart silently doubles its own
//! width. `glyphs.rs` documents the same rule for the gauge bars, and
//! `sidebar::inline_sparkline` is deliberately NOT reused here because it is
//! built on that banned ramp.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::tui::glyphs;
use crate::tui::theme::Theme;

/// Buckets needed before a trend is a shape rather than a single level.
const MIN_TREND_POINTS: usize = 2;

/// Dot bit for `[row][column]` within one braille cell. The Unicode braille
/// block numbers the top three rows down each column first and appends the
/// fourth row as the high bits, which is why the last row is not `0x08`-aligned
/// with the rest.
const BRAILLE_DOTS: [[u8; 2]; 4] = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
];

/// Filled-area chart of `series` over `width` × `height` character cells.
///
/// `max` is supplied rather than derived so a caller can share one denominator
/// with the table beneath the chart; deriving it here would let the band and
/// the rows disagree about which bucket is the tall one.
///
/// Returns no rows when there is nothing to draw, and a plain sentence when
/// every bucket is zero — a flat baseline would read as "a little usage", which
/// is a shape the data does not have.
#[must_use]
pub fn braille_area_chart(
    series: &[u64],
    max: u64,
    width: u16,
    height: u16,
    color: Color,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let cells_w = usize::from(width);
    let cells_h = usize::from(height);
    if series.is_empty() || cells_w == 0 || cells_h == 0 {
        return Vec::new();
    }
    if max == 0 {
        return vec![Line::from(Span::styled(
            "no usage in this range",
            theme.typography.placeholder,
        ))];
    }
    // One bucket has no trend to draw. Stretched across the pane it normalizes
    // to its own maximum and fills every cell, so a first-day user opened the
    // dashboard to a solid rectangle that looked like a rendering fault. Say
    // there is not enough history instead; the caption still carries the value.
    if series.len() < MIN_TREND_POINTS {
        return vec![Line::from(Span::styled(
            "one bucket so far — no trend to plot yet",
            theme.typography.placeholder,
        ))];
    }
    if theme.no_color {
        return ascii_area_chart(series, max, cells_w, cells_h);
    }

    let dot_cols = cells_w * 2;
    let dot_rows = cells_h * 4;
    let mut cells = vec![0u8; cells_w * cells_h];
    for dx in 0..dot_cols {
        let filled = fill_levels(sample(series, dx, dot_cols), max, dot_rows);
        for level in 0..filled {
            let dy = dot_rows - 1 - level;
            cells[(dy / 4) * cells_w + dx / 2] |= BRAILLE_DOTS[dy % 4][dx % 2];
        }
    }
    (0..cells_h)
        .map(|row| {
            let text: String = cells[row * cells_w..(row + 1) * cells_w]
                .iter()
                .map(|bits| glyphs::braille_cell(*bits))
                .collect();
            Line::from(Span::styled(text, Style::new().fg(color)))
        })
        .collect()
}

/// One horizontal bar whose runs are proportional to `segments`, plus a legend
/// row naming each run and its share.
///
/// Each segment carries its own colour because the legend has to pair a name
/// with the run the eye just saw; a shared ramp would make two adjacent
/// segments indistinguishable.
#[must_use]
pub fn stacked_composition_bar(
    segments: &[(String, u64, Color)],
    width: u16,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let cells = usize::from(width);
    let total: u64 = segments.iter().map(|(_, value, _)| *value).sum();
    if cells == 0 || total == 0 {
        return Vec::new();
    }
    let fill = glyphs::pick(!theme.no_color, glyphs::GAUGE_FILL, glyphs::GAUGE_FILL_NC);
    let nonzero: Vec<usize> = segments
        .iter()
        .enumerate()
        .filter(|(_, (_, value, _))| *value > 0)
        .map(|(idx, _)| idx)
        .collect();
    if nonzero.is_empty() {
        return Vec::new();
    }

    // Every non-zero segment is handed one cell before anything is shared out.
    // Allocating greedily instead let the big runs spend the whole width and
    // dropped the small ones entirely, which reports a mix the data does not
    // have. The rest goes by largest remainder, so runs stay proportional and
    // the runs still sum to exactly `cells`.
    let mut runs = vec![0usize; segments.len()];
    if nonzero.len() >= cells {
        for idx in nonzero.iter().take(cells) {
            runs[*idx] = 1;
        }
    } else {
        let budget = cells - nonzero.len();
        let mut remainders: Vec<(u128, usize)> = Vec::with_capacity(nonzero.len());
        let mut handed = 0usize;
        for &idx in &nonzero {
            let exact = u128::from(segments[idx].1) * budget as u128;
            let share = usize::try_from(exact / u128::from(total)).unwrap_or(0);
            runs[idx] = 1 + share;
            handed += share;
            remainders.push((exact % u128::from(total), idx));
        }
        remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut leftover = budget.saturating_sub(handed);
        for (_, idx) in remainders {
            if leftover == 0 {
                break;
            }
            runs[idx] += 1;
            leftover -= 1;
        }
    }

    let mut spans = Vec::with_capacity(nonzero.len());
    for (idx, (_, _, color)) in segments.iter().enumerate() {
        if runs[idx] == 0 {
            continue;
        }
        spans.push(Span::styled(
            fill.repeat(runs[idx]),
            Style::new().fg(*color),
        ));
    }

    let mut legend = Vec::with_capacity(segments.len() * 3);
    for (idx, (label, value, color)) in segments.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        if idx > 0 && !legend.is_empty() {
            legend.push(Span::styled("  ", theme.typography.dim));
        }
        legend.push(Span::styled(format!("{fill} "), Style::new().fg(*color)));
        legend.push(Span::styled(
            format!("{label} {}%", percent(*value, total)),
            theme.typography.dim,
        ));
    }
    vec![Line::from(spans), Line::from(legend)]
}

/// Value under horizontal position `at` of `steps` sample points.
fn sample(series: &[u64], at: usize, steps: usize) -> u64 {
    let last = series.len() - 1;
    if last == 0 || steps <= 1 {
        return series[0];
    }
    series[(at * last / (steps - 1)).min(last)]
}

/// Levels to fill for `value` out of `levels`. A non-zero bucket always claims
/// at least one level so a quiet day stays visible instead of vanishing into
/// the baseline and reading as no usage at all.
fn fill_levels(value: u64, max: u64, levels: usize) -> usize {
    if value == 0 || max == 0 {
        return 0;
    }
    scaled_len(value, max, levels).clamp(1, levels)
}

/// `value / total` of `cells`, in integer arithmetic.
fn scaled_len(value: u64, total: u64, cells: usize) -> usize {
    if total == 0 {
        return 0;
    }
    let scaled = u128::from(value) * cells as u128 / u128::from(total);
    usize::try_from(scaled).unwrap_or(cells).min(cells)
}

/// Whole-percent share, rounded.
fn percent(value: u64, total: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    u64::try_from((u128::from(value) * 200 / u128::from(total)).div_ceil(2)).unwrap_or(100)
}

/// `NO_COLOR` fallback: one cell per column, filled up to the column's height.
/// Coarser than braille by a factor of eight, but the shape still reads without
/// a single colour — the chart degrades rather than disappearing.
fn ascii_area_chart(
    series: &[u64],
    max: u64,
    cells_w: usize,
    cells_h: usize,
) -> Vec<Line<'static>> {
    (0..cells_h)
        .map(|row| {
            let level_from_bottom = cells_h - row;
            let text: String = (0..cells_w)
                .map(|col| {
                    let filled = fill_levels(sample(series, col, cells_w), max, cells_h);
                    if filled >= level_from_bottom {
                        glyphs::BRAILLE_FILL_NC
                    } else {
                        glyphs::BRAILLE_EMPTY_NC
                    }
                })
                .collect();
            Line::from(text)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_width::UnicodeWidthStr;

    fn text_of(lines: &[Line<'static>]) -> Vec<String> {
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

    /// The whole reason this module exists: a `ko_KR` wide-ambiguous tmux must
    /// still paint one column per cell, or the chart doubles its own width and
    /// overruns the pane. `width_cjk` is the proxy for that locale.
    #[test]
    fn every_chart_glyph_is_one_cell_under_wide_ambiguous() {
        let theme = Theme::zo();
        let lines = braille_area_chart(&[1, 5, 3, 9, 2], 9, 12, 4, theme.palette.bright, &theme);
        assert_eq!(lines.len(), 4);
        for row in text_of(&lines) {
            assert_eq!(
                UnicodeWidthStr::width_cjk(row.as_str()),
                12,
                "row must stay 12 columns wide: {row:?}"
            );
            assert!(
                !row.contains('\u{2588}') && !row.contains('\u{2581}'),
                "the Ambiguous block ramp must never reach a chart: {row:?}"
            );
        }
    }

    /// The tallest bucket has to reach the top row and the shortest must not,
    /// otherwise the band is decoration rather than a reading of the data.
    #[test]
    fn the_peak_reaches_the_top_row_and_the_trough_does_not() {
        let theme = Theme::zo();
        let lines = braille_area_chart(&[0, 100], 100, 8, 4, theme.palette.bright, &theme);
        let rows = text_of(&lines);

        let top_ink = rows[0].chars().any(|ch| ch != glyphs::BRAILLE_BLANK);
        assert!(top_ink, "peak must reach the top row: {rows:?}");
        let first_col_top = rows[0].chars().next().expect("a first cell");
        assert_eq!(
            first_col_top,
            glyphs::BRAILLE_BLANK,
            "a zero bucket must leave the top row blank: {rows:?}"
        );
    }

    /// A quiet-but-nonzero bucket must still show. Rounding it to nothing would
    /// claim there was no usage, which is a different fact.
    #[test]
    fn a_tiny_nonzero_bucket_still_draws_ink() {
        assert_eq!(fill_levels(1, 10_000, 32), 1);
        assert_eq!(fill_levels(0, 10_000, 32), 0);
    }

    /// Degenerate inputs are ordinary here: a fresh install has no usage at all.
    #[test]
    fn degenerate_series_never_panic() {
        let theme = Theme::zo();
        assert!(braille_area_chart(&[], 0, 20, 4, theme.palette.bright, &theme).is_empty());
        assert!(braille_area_chart(&[5], 5, 0, 4, theme.palette.bright, &theme).is_empty());
        assert!(braille_area_chart(&[5], 5, 20, 0, theme.palette.bright, &theme).is_empty());

        // All-zero says so in words rather than drawing a floor that reads as
        // a small amount of usage.
        let flat = braille_area_chart(&[0, 0, 0], 0, 20, 4, theme.palette.bright, &theme);
        assert_eq!(text_of(&flat), vec!["no usage in this range".to_string()]);

        // A single sample is a legitimate first day, and it must not become a
        // solid rectangle: normalized against itself every cell would fill.
        let single = text_of(&braille_area_chart(
            &[7],
            7,
            6,
            2,
            theme.palette.bright,
            &theme,
        ));
        assert_eq!(
            single,
            vec!["one bucket so far — no trend to plot yet".to_string()],
            "one bucket states the fact instead of painting a full block"
        );
    }

    /// Without colour the chart must degrade, not vanish — and it must not leak
    /// the Unicode it can no longer differentiate.
    #[test]
    fn no_color_degrades_to_ascii_without_leaking_braille() {
        let theme = Theme::no_color();
        let rows = text_of(&braille_area_chart(
            &[1, 4, 2],
            4,
            9,
            3,
            theme.palette.bright,
            &theme,
        ));
        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter().any(|row| row.contains(glyphs::BRAILLE_FILL_NC)),
            "the shape still has to read: {rows:?}"
        );
        for row in &rows {
            assert!(row.is_ascii(), "no braille under NO_COLOR: {row:?}");
        }
    }

    /// The bar must spend its whole width, and every non-zero segment must be
    /// visible — a 1% slice rounding away would misreport the mix.
    #[test]
    fn a_composition_bar_fills_its_width_and_shows_every_segment() {
        let theme = Theme::zo();
        let segments = vec![
            ("spent".to_string(), 990_u64, theme.palette.violet),
            ("cache".to_string(), 5, theme.palette.success),
            ("mix".to_string(), 5, theme.palette.teal),
        ];
        let lines = stacked_composition_bar(&segments, 30, &theme);
        let rows = text_of(&lines);
        assert_eq!(rows.len(), 2, "bar plus legend");
        assert_eq!(
            UnicodeWidthStr::width_cjk(rows[0].as_str()),
            30,
            "the bar spends its full width: {rows:?}"
        );
        assert_eq!(
            lines[0].spans.len(),
            3,
            "each non-zero segment keeps a run of its own: {rows:?}"
        );
        assert!(rows[1].contains("spent"), "legend names the runs: {rows:?}");

        assert!(stacked_composition_bar(&segments, 0, &theme).is_empty());
        let zeroed = vec![("none".to_string(), 0_u64, theme.palette.violet)];
        assert!(stacked_composition_bar(&zeroed, 20, &theme).is_empty());
    }
}
