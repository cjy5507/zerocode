//! Single source of truth for terminal-cell text measurement.
//!
//! Every width calculation in `tui/` funnels through here so the CJK /
//! ambiguous-width policy lives in exactly one place (see `styles.md`). All
//! measurement uses `unicode-width` with the default (non-CJK, ambiguous =
//! narrow) tables — the same tables ratatui's `Line::width` uses, so height
//! measurement and paint agree.

use ratatui::layout::Alignment;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Display width of `text` in terminal cells.
#[must_use]
pub fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Display width of a single `char` in terminal cells. Control and
/// zero-width characters measure as `0`.
#[must_use]
pub fn char_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

/// Truncate `text` to `budget` terminal cells, appending `…` when it had to cut.
///
/// The cell-level counterpart to [`truncate_line_to_cells`], for the places that
/// compose a row *around* a label — a section rule, a two-column field — and so
/// need the label's width before it becomes a `Span`.
#[must_use]
pub fn truncate_to_cells(text: &str, budget: usize) -> String {
    if display_width(text) <= budget {
        return text.to_string();
    }
    if budget == 0 {
        return String::new();
    }
    // One cell is reserved for the ellipsis that marks the cut.
    let keep = budget - 1;
    let mut out = String::new();
    let mut used = 0usize;
    for ch in text.chars() {
        let width = char_width(ch);
        if used + width > keep {
            break;
        }
        used += width;
        out.push(ch);
    }
    out.push('\u{2026}');
    out
}

/// Truncate a styled row to `budget` terminal cells, appending `…` when it had to
/// cut. Span styles are preserved up to the cut so a truncated row keeps its
/// colour coding.
///
/// Shared by the ledger rail and the modal bodies. Both render through a
/// `Paragraph` without `.wrap(...)`, where ratatui composes lines via
/// `LineTruncator` — which stops at the rect edge and adds no ellipsis, so an
/// over-wide row silently loses its tail mid-glyph. Cutting here instead keeps the
/// loss *visible*, which is the whole difference between a shortened label and a
/// corrupted one.
///
/// Measured in cells, not chars: the rail carries Hangul labels, and a
/// `chars().count()` budget would let those rows overrun the panel by one column
/// per syllable.
#[must_use]
pub fn truncate_line_to_cells(line: Line<'_>, budget: usize) -> Line<'_> {
    if budget == 0 {
        return Line::from(Vec::new());
    }
    let total: usize = line
        .spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum();
    if total <= budget {
        return line;
    }
    // One cell is reserved for the ellipsis that marks the cut.
    let keep = budget.saturating_sub(1);
    // Row-level style and alignment are carried over: a selected row paints its
    // wash through `Line::style`, so rebuilding the line without them would strip
    // the highlight off exactly the rows long enough to need cutting.
    let row_style = line.style;
    let alignment = line.alignment;
    let mut out: Vec<Span<'_>> = Vec::with_capacity(line.spans.len() + 1);
    let mut used = 0usize;
    for span in line.spans {
        let span_width = display_width(span.content.as_ref());
        if used + span_width <= keep {
            used += span_width;
            out.push(span);
            continue;
        }
        let style = span.style;
        let mut cut = String::new();
        for ch in span.content.chars() {
            let w = char_width(ch);
            if used + w > keep {
                break;
            }
            used += w;
            cut.push(ch);
        }
        if !cut.is_empty() {
            out.push(Span::styled(cut, style));
        }
        out.push(Span::styled("\u{2026}".to_string(), style));
        return Line {
            style: row_style,
            alignment,
            spans: out,
        };
    }
    Line {
        style: row_style,
        alignment,
        spans: out,
    }
}

/// Word-wrap `line` to `width` terminal cells, preserving every non-whitespace
/// character and every span style.
///
/// # Why this exists instead of `Paragraph::wrap`
///
/// ratatui's `WordWrapper` decides a row is full by comparing the width it has
/// *already* accumulated against the limit (`reflow.rs`:
/// `line_width + whitespace_width + word_width >= max_line_width`), and only then
/// appends the current grapheme. With single-cell text the comparison is exact;
/// with a double-width grapheme the row overshoots the limit by one cell, and the
/// paint — which cannot draw half a glyph — drops that grapheme outright.
///
/// Reproduced on ratatui 0.30 (`ratatui-widgets` 0.3.0): wrapping
/// `"말줄임표로 잘라낸다 (권장) 아주 긴 한글 문장을 좁은 폭에서 접어봅니다"` into 13 cells
/// paints one row 14 cells wide and loses `한` entirely — a silently corrupted
/// sentence, not a shortened one. Korean prose hits it at roughly one width in
/// seven, which is exactly the "글자가 사라진다" class of render corruption.
///
/// # Contract
///
/// * no returned row measures more than `width` cells
/// * every non-whitespace character of the input appears exactly once, in order
/// * a word longer than `width` breaks mid-word on a cell boundary
/// * `trim` drops the whitespace that would otherwise lead a continuation row
///
/// Rows come back owned so a caller can render them through a `Paragraph`
/// *without* `wrap` — one input row in, N painted rows out, which also makes
/// height measurement a `len()` instead of a second guess at the wrapper.
#[must_use]
pub fn wrap_line_to_cells(line: &Line<'_>, width: usize, trim: bool) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let mut wrapper = Wrapper::new(line, width, trim);
    for span in &line.spans {
        for ch in span.content.chars() {
            wrapper.push(ch, span.style);
        }
    }
    wrapper.finish()
}

/// Accumulator for [`wrap_line_to_cells`]: a committed row, the word being read,
/// and the whitespace between them — the three buffers a word wrapper needs.
struct Wrapper {
    width: usize,
    trim: bool,
    row_style: Style,
    alignment: Option<Alignment>,
    rows: Vec<Line<'static>>,
    row: Vec<(char, Style)>,
    row_width: usize,
    space: Vec<(char, Style)>,
    space_width: usize,
    word: Vec<(char, Style)>,
    word_width: usize,
}

impl Wrapper {
    fn new(line: &Line<'_>, width: usize, trim: bool) -> Self {
        Self {
            width,
            trim,
            row_style: line.style,
            alignment: line.alignment,
            rows: Vec::new(),
            row: Vec::new(),
            row_width: 0,
            space: Vec::new(),
            space_width: 0,
            word: Vec::new(),
            word_width: 0,
        }
    }

    fn push(&mut self, ch: char, style: Style) {
        let cells = char_width(ch);
        if ch.is_whitespace() {
            self.commit_word();
            // Whitespace past the row's edge is the break itself: it is dropped
            // rather than carried, which is what keeps a wrapped paragraph from
            // starting a row on a space.
            if self.row_width + self.space_width + cells <= self.width {
                self.space.push((ch, style));
                self.space_width += cells;
            }
            return;
        }
        if cells > self.width {
            // Wider than any row could ever hold; nothing to do but skip it,
            // exactly as ratatui does — carrying it would loop forever.
            return;
        }
        // A word longer than a whole row breaks mid-word, on a cell boundary.
        if self.word_width + cells > self.width {
            self.commit_word();
            self.break_row();
        }
        self.word.push((ch, style));
        self.word_width += cells;
    }

    /// Move the pending word onto the current row, breaking the row first when it
    /// no longer fits. This is the only place a row's width grows.
    fn commit_word(&mut self) {
        if self.word.is_empty() {
            return;
        }
        if self.row_width + self.space_width + self.word_width > self.width {
            self.break_row();
        }
        if !self.row.is_empty() || !self.trim {
            self.row.append(&mut self.space);
            self.row_width += self.space_width;
        } else {
            self.space.clear();
        }
        self.space_width = 0;
        self.row.append(&mut self.word);
        self.row_width += self.word_width;
        self.word_width = 0;
    }

    /// Emit the current row (even empty, so a blank input row stays a blank
    /// output row) and drop the whitespace that straddled the break.
    fn break_row(&mut self) {
        let row = std::mem::take(&mut self.row);
        self.rows.push(coalesce(row, self.row_style, self.alignment));
        self.row_width = 0;
        self.space.clear();
        self.space_width = 0;
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.commit_word();
        if !self.row.is_empty() || self.rows.is_empty() {
            let row = std::mem::take(&mut self.row);
            self.rows.push(coalesce(row, self.row_style, self.alignment));
        }
        self.rows
    }
}

/// Rebuild a row of `(char, Style)` into spans, merging neighbours that share a
/// style so the output has no more spans than the input did.
fn coalesce(
    chars: Vec<(char, Style)>,
    row_style: Style,
    alignment: Option<Alignment>,
) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (ch, style) in chars {
        match spans.last_mut() {
            Some(last) if last.style == style => last.content.to_mut().push(ch),
            _ => spans.push(Span::styled(ch.to_string(), style)),
        }
    }
    Line {
        style: row_style,
        alignment,
        spans,
    }
}
