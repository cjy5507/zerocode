//! Single source of truth for terminal-cell text measurement.
//!
//! Every width calculation in `tui/` funnels through here so the CJK /
//! ambiguous-width policy lives in exactly one place (see `styles.md`). All
//! measurement uses `unicode-width` with the default (non-CJK, ambiguous =
//! narrow) tables — the same tables ratatui's `Line::width` uses, so height
//! measurement and paint agree.

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
