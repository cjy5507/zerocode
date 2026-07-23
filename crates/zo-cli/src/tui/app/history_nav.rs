//! 입력 히스토리 탐색 — 명령 히스토리(빈도 기반)와 줄 히스토리(↑/↓)를 App
//! 입력 버퍼에 얹는 책임. App 의 `history`/`command_history` 상태만 다룬다.

use super::{App, SystemLevel};
use crate::tui::command_history::CommandHistory;
use crate::tui::history::{History, HistoryRecord};

impl App {
    /// Attach a loaded history instance. Called after construction
    /// by the session loop once the data directory is known.
    pub fn set_history(&mut self, history: History) {
        self.history = history;
        self.input.set_history_hint(!self.history.is_empty());
    }

    /// Attach a loaded command usage history.
    pub fn set_command_history(&mut self, history: CommandHistory) {
        self.command_history = history;
    }

    fn warn_history_persistence(&mut self, error: impl std::fmt::Display) {
        if self.history_persistence_warning_shown {
            return;
        }
        self.history_persistence_warning_shown = true;
        self.push_diff_note(SystemLevel::Warn, format!("History was not saved: {error}"));
    }

    /// Record a slash command usage for frecency tracking.
    pub fn record_command_usage(&mut self, command: &str) {
        if let Err(error) = self.command_history.record(command) {
            self.warn_history_persistence(error);
        }
    }

    /// Record a submitted prompt in the input history.
    pub fn append_history(&mut self, text: &str) {
        let record = HistoryRecord::now(text.to_string(), "normal");
        if let Err(error) = self.history.append(record) {
            self.warn_history_persistence(error);
        }
        self.history_cursor = None;
        self.history_stash.clear();
        // First-ever entry: the placeholder hint becomes truthful now.
        self.input.set_history_hint(true);
    }

    /// Load a history entry into the input by its [`Self::history_cursor`]
    /// offset (0 = most recent, 1 = the one before, …). The text replaces the
    /// whole buffer with the cursor parked at the end, mirroring `set_input_text`
    /// but without disturbing slash/mention hint state machinery.
    fn load_history_entry(&mut self, cursor: usize) {
        let entries = self.history.entries();
        let len = entries.len();
        if cursor >= len {
            return;
        }
        let text = entries[len - 1 - cursor].text.clone();
        self.input.clear();
        for ch in text.chars() {
            self.input.insert_char(ch);
        }
    }

    /// Browse to the previous (older) prompt in history — the Up-arrow action
    /// when the input cursor sits on the first line. On first entry the current
    /// draft is stashed so a later Down restores it. Returns `true` when a
    /// history entry was loaded, `false` when there is nothing older to show
    /// (so the caller can fall back to transcript scrolling).
    pub fn history_prev(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        let next_cursor = match self.history_cursor {
            None => {
                // Entering history browse: stash the in-progress draft.
                self.history_stash = self.input.text();
                0
            }
            Some(c) => c + 1,
        };
        if next_cursor >= self.history.len() {
            // Already at the oldest entry — nothing further back.
            return false;
        }
        self.history_cursor = Some(next_cursor);
        self.load_history_entry(next_cursor);
        true
    }

    /// Browse toward more-recent prompts — the Down-arrow action when the input
    /// cursor sits on the last line. Stepping past the most recent entry
    /// restores the stashed draft and exits history browsing. Returns `true`
    /// when input was updated, `false` when not currently browsing history (so
    /// the caller can fall back to transcript scrolling).
    pub fn history_next(&mut self) -> bool {
        match self.history_cursor {
            None => false,
            Some(0) => {
                // Step past the newest entry: restore the stashed draft.
                let stash = std::mem::take(&mut self.history_stash);
                self.input.clear();
                for ch in stash.chars() {
                    self.input.insert_char(ch);
                }
                self.history_cursor = None;
                true
            }
            Some(c) => {
                let next_cursor = c - 1;
                self.history_cursor = Some(next_cursor);
                self.load_history_entry(next_cursor);
                true
            }
        }
    }
}
