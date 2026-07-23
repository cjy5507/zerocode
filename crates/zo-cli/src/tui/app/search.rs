//! Transcript search (Ctrl+F) behavior for [`App`].
//!
//! Split out of `app/mod.rs` as a focused `impl App` block: the search
//! query, match set, and active-match cursor form one cohesive
//! responsibility (incremental find + highlight + wrap-around navigation).
//! Keeping it here trims the `app/mod.rs` god-object without touching the
//! `App` struct, its fields, or the public API — Rust permits multiple
//! `impl` blocks for the same type across files in a crate.

use super::{App, AppMode};

impl App {
    // ── Search (Ctrl+F) ────────────────────────────────────────────

    /// Enter search mode, clearing any previous query and highlight.
    pub fn enter_search(&mut self) {
        self.search.query.clear();
        self.search.matches.clear();
        self.search.active_match = 0;
        self.transcript.set_search_highlight(None);
        self.mode = AppMode::Search;
    }

    /// Exit search mode and return to Normal, clearing the highlight.
    pub fn exit_search(&mut self) {
        self.transcript.set_search_highlight(None);
        self.mode = AppMode::Normal;
    }

    /// Current search query (only meaningful in [`AppMode::Search`]).
    #[must_use]
    pub fn search_query(&self) -> &str {
        &self.search.query
    }

    /// Number of blocks matching the current query.
    #[must_use]
    pub fn search_match_count(&self) -> usize {
        self.search.matches.len()
    }

    /// 1-based position of the active match, or `0` when there are none.
    #[must_use]
    pub fn search_active_position(&self) -> usize {
        if self.search.matches.is_empty() {
            0
        } else {
            self.search.active_match + 1
        }
    }

    /// Recompute the match set for the current query (incremental search)
    /// and jump to the first hit. Called on every query edit.
    pub fn refresh_search(&mut self) {
        let query = self.search.query.to_lowercase();
        self.search.matches = if query.is_empty() {
            Vec::new()
        } else {
            self.transcript.find_all_blocks_containing(&query)
        };
        self.search.active_match = 0;
        self.focus_active_match();
    }

    /// Scroll to and highlight the active match, or clear the highlight
    /// when there are no matches.
    fn focus_active_match(&mut self) {
        if let Some(&idx) = self.search.matches.get(self.search.active_match) {
            self.transcript.scroll_to_block(idx);
            self.transcript.set_search_highlight(Some(idx));
            self.transcript_view.follow_output = false;
        } else {
            self.transcript.set_search_highlight(None);
        }
    }

    /// Advance to the next match, wrapping around.
    pub fn search_next(&mut self) {
        let count = self.search.matches.len();
        if count == 0 {
            return;
        }
        self.search.active_match = (self.search.active_match + 1) % count;
        self.focus_active_match();
    }

    /// Step to the previous match, wrapping around.
    pub fn search_prev(&mut self) {
        let count = self.search.matches.len();
        if count == 0 {
            return;
        }
        self.search.active_match = (self.search.active_match + count - 1) % count;
        self.focus_active_match();
    }
}
