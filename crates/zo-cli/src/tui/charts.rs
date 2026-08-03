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
use crate::tui::text_metrics::display_width;
use crate::tui::theme::Theme;

/// Buckets needed before a trend is a shape rather than a single level.
const MIN_TREND_POINTS: usize = 2;

/// Marks that stand in for hue when a run has none. See [`segment_mark`].
const PLAIN_SEGMENT_MARKS: [&str; 5] = ["#", "=", "+", "~", ":"];

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

/// Stroked line chart of `series` over `width` × `height` character cells.
///
/// A **stroke, not a fill.** Filling the area under the curve spends every cell
/// below the value, so on a wide pane the chart reads as a block of colour and
/// the eye has nothing to follow; the shape is carried entirely by the top
/// edge, which is the only part worth drawing. Vertical runs are bridged
/// between adjacent columns so a steep climb stays one continuous line rather
/// than a dotted stair.
///
/// `max` is supplied rather than derived so a caller can share one denominator
/// with the table beneath the chart; deriving it here would let the band and
/// the rows disagree about which bucket is the tall one.
///
/// Returns no rows when there is nothing to draw, and a plain sentence when
/// every bucket is zero — a flat baseline would read as "a little usage", which
/// is a shape the data does not have.
#[must_use]
pub fn braille_line_chart(
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
        return ascii_line_chart(series, max, cells_w, cells_h);
    }

    let dot_cols = cells_w * 2;
    let dot_rows = cells_h * 4;
    let mut cells = vec![0u8; cells_w * cells_h];
    let mut previous: Option<usize> = None;
    for dx in 0..dot_cols {
        let filled = fill_levels(sample(series, dx, dot_cols), max, dot_rows);
        let dy = dot_rows - filled.max(1);
        // Bridge the gap to the previous column so a jump between buckets draws
        // as one line. Without it a steep step leaves two disconnected dots and
        // the series reads as scattered points rather than a path.
        let (top, bottom) = match previous {
            Some(prior) if prior < dy => (prior + 1, dy),
            Some(prior) if prior > dy => (dy, prior - 1),
            _ => (dy, dy),
        };
        for row in top..=bottom {
            cells[(row / 4) * cells_w + dx / 2] |= BRAILLE_DOTS[row % 4][dx % 2];
        }
        previous = Some(dy);
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
            segment_mark(idx, fill, theme).repeat(runs[idx]),
            Style::new().fg(*color),
        ));
    }

    // The legend is dropped whole entry by whole entry once the row is spent —
    // cutting instead leaves half a model name beside a run and pairs the eye
    // with the wrong thing. The `+N more` tail is text on that same row, so it
    // is budgeted like an entry: appended blind it overran the row and the
    // renderer cut it to `+2 mo`, the exact fragment the tail exists to
    // prevent. Reserving the tail can push one more entry out and lengthen the
    // tail in turn, so the fit is iterated to a fixed point — it terminates
    // because the fitted count only ever falls.
    let entries: Vec<(usize, String)> = segments
        .iter()
        .enumerate()
        .filter(|(_, (_, value, _))| *value > 0)
        .map(|(idx, (label, value, _))| (idx, format!("{label} {}%", percent(*value, total))))
        .collect();
    let mark_width = display_width(fill);
    let mut shown = entries.len();
    loop {
        let reserve = match entries.len() - shown {
            0 => 0,
            hidden => display_width(&format!("  +{hidden} more")),
        };
        let fits = legend_entries_that_fit(&entries, cells, reserve, mark_width);
        if fits >= shown {
            break;
        }
        shown = fits;
    }
    let hidden = entries.len() - shown;

    let mut legend: Vec<Span<'static>> = Vec::with_capacity(shown * 3 + 1);
    for (position, (idx, text)) in entries.iter().take(shown).enumerate() {
        if position > 0 {
            legend.push(Span::styled("  ", theme.typography.dim));
        }
        legend.push(Span::styled(
            format!("{} ", segment_mark(*idx, fill, theme)),
            Style::new().fg(segments[*idx].2),
        ));
        legend.push(Span::styled(text.clone(), theme.typography.dim));
    }
    // A row too narrow for even one name gets no legend at all. A bare
    // "+3 more" names nothing, and a truncated first entry is the fragment
    // this whole path is avoiding; the bar still carries the proportions.
    if hidden > 0 && shown > 0 {
        legend.push(Span::styled(
            format!("  +{hidden} more"),
            theme.typography.dim,
        ));
    }
    vec![Line::from(spans), Line::from(legend)]
}

/// Legend entries that fit in `cells` while holding `reserve` cells back for
/// the `+N more` tail. Each entry costs its two-space separator (after the
/// first), its mark, the space behind it, and its own text.
fn legend_entries_that_fit(
    entries: &[(usize, String)],
    cells: usize,
    reserve: usize,
    mark_width: usize,
) -> usize {
    let mut used = 0usize;
    for (position, (_, text)) in entries.iter().enumerate() {
        let gap = if position > 0 { 2 } else { 0 };
        let entry = gap + mark_width + 1 + display_width(text);
        if used + entry + reserve > cells {
            return position;
        }
        used += entry;
    }
    entries.len()
}

/// The glyph one run is drawn with.
///
/// Colour carries the distinction when there is colour. Under `NO_COLOR` every
/// palette slot resolves to `Color::Reset`, so a single glyph paints the whole
/// bar as one undivided run and the legend's swatches name nothing — the
/// composition becomes unreadable exactly where colour cannot rescue it. ASCII
/// marks are EAW-Neutral by construction, so the width contract still holds.
fn segment_mark(index: usize, fill: &'static str, theme: &Theme) -> &'static str {
    if theme.no_color {
        PLAIN_SEGMENT_MARKS[index % PLAIN_SEGMENT_MARKS.len()]
    } else {
        fill
    }
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

/// `NO_COLOR` fallback: the same stroke at one level per cell instead of eight.
/// Coarser than braille by a factor of four vertically, but it is the same
/// picture — a traced top edge, not a filled block — so the chart degrades
/// rather than changing shape.
fn ascii_line_chart(
    series: &[u64],
    max: u64,
    cells_w: usize,
    cells_h: usize,
) -> Vec<Line<'static>> {
    let levels: Vec<usize> = (0..cells_w)
        .map(|col| fill_levels(sample(series, col, cells_w), max, cells_h).max(1))
        .collect();
    (0..cells_h)
        .map(|row| {
            let level_from_bottom = cells_h - row;
            let text: String = levels
                .iter()
                .map(|level| {
                    if *level == level_from_bottom {
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
        let lines = braille_line_chart(&[1, 5, 3, 9, 2], 9, 12, 4, theme.palette.bright, &theme);
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
        let lines = braille_line_chart(&[0, 100], 100, 8, 4, theme.palette.bright, &theme);
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
        assert!(braille_line_chart(&[], 0, 20, 4, theme.palette.bright, &theme).is_empty());
        assert!(braille_line_chart(&[5], 5, 0, 4, theme.palette.bright, &theme).is_empty());
        assert!(braille_line_chart(&[5], 5, 20, 0, theme.palette.bright, &theme).is_empty());

        // All-zero says so in words rather than drawing a floor that reads as
        // a small amount of usage.
        let flat = braille_line_chart(&[0, 0, 0], 0, 20, 4, theme.palette.bright, &theme);
        assert_eq!(text_of(&flat), vec!["no usage in this range".to_string()]);

        // A single sample is a legitimate first day, and it must not become a
        // solid rectangle: normalized against itself every cell would fill.
        let single = text_of(&braille_line_chart(
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
        let rows = text_of(&braille_line_chart(
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

    /// Without colour every run is the same glyph in the same `Color::Reset`,
    /// so the composition reads as one undivided bar and the legend's marks
    /// pair with nothing. Each run needs its own mark to survive `NO_COLOR`.
    #[test]
    fn plain_mode_tells_the_runs_apart_without_colour() {
        let theme = Theme::no_color();
        let segments = vec![
            ("alpha".to_string(), 40_u64, theme.palette.violet),
            ("beta".to_string(), 30, theme.palette.success),
            ("gamma".to_string(), 30, theme.palette.teal),
        ];
        // Wide enough to seat all three legend entries: the pairing between a
        // run and its name is only defined for entries the legend shows.
        let rows = text_of(&stacked_composition_bar(&segments, 44, &theme));

        let marks: std::collections::BTreeSet<char> = rows[0].chars().collect();
        assert!(
            marks.len() >= 3,
            "each run carries its own mark without colour: {rows:?}"
        );
        for mark in marks {
            assert!(
                rows[1].contains(mark),
                "the legend repeats the mark it names: {rows:?}"
            );
        }
    }

    /// The `+N more` tail is itself text on the row, so it has to be budgeted
    /// like an entry. Appended blind it overran the row and the renderer cut it
    /// to `+2 mo` — the same unreadable fragment the tail exists to prevent.
    #[test]
    fn the_legend_never_overruns_its_row_at_any_width() {
        let theme = Theme::zo();
        let segments = vec![
            ("alpha-model".to_string(), 40_u64, theme.palette.violet),
            ("beta-model".to_string(), 30, theme.palette.success),
            ("gamma-model".to_string(), 30, theme.palette.teal),
        ];

        for width in 8..=60u16 {
            let rows = text_of(&stacked_composition_bar(&segments, width, &theme));
            assert!(
                UnicodeWidthStr::width_cjk(rows[1].as_str()) <= usize::from(width),
                "legend overruns at width {width}: {rows:?}"
            );
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
