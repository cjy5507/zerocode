//! Multi-line input widget (Phase 3, Lane L6).
//!
//! Hand-rolled minimal textarea used by the Phase-3 TUI. The lane spec
//! (`.zo/tasks/L6-tui-input-modals.md`) permits falling back to a
//! hand-rolled `TextArea`-lite rather than pulling in
//! `tui-textarea = "0.7"`; the decision and rationale are captured in
//! the L6 handoff. Keeping the implementation in-tree avoids a new
//! dependency, keeps every key binding explicit, and lets us round-trip
//! the widget in unit tests without a real terminal backend.
//!
//! ## Living standard (mirrors L1)
//!
//! 1. Module layout: one file per widget under `tui/`.
//! 2. Errors: this widget has no fallible surface and therefore no
//!    dedicated `thiserror` enum — errors only appear when it escapes
//!    through [`super::TuiError`] from the app loop.
//! 3. No async traits live here.
//! 4. Tests live at `crates/zo-cli/tests/tui_input.rs` and
//!    follow the `<area>_<scenario>` naming convention.
//! 5. Every `pub` item carries a `///` doc comment.
//!
//! Code-rules: R1 (neutral vocabulary), R2 (no ANSI), R9 (`&Theme`
//! drives every style decision).

use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Padding, Paragraph};

use unicode_width::UnicodeWidthChar;

use super::app::AppMode;
use super::glyphs;
use super::heat::HeatState;
use super::layout::{INPUT_MAX_ROWS, INPUT_MIN_ROWS};
use super::theme::Theme;

/// Shell-style prompt rendered at the start of the first input row.
pub const PROMPT: &str = "\u{276f} ";
/// Visible character width of [`PROMPT`].
const PROMPT_WIDTH: u16 = 2;

/// Command emitted by the input widget after a key press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputCommand {
    /// User pressed plain `Enter` — submit the buffered text.
    Submit(String),
    /// User pressed `Ctrl-C` — cancel/clear the buffer.
    Cancel,
}

/// Placeholder shown when the buffer is empty.
pub const PLACEHOLDER: &str = "Message Zo, / for commands";

/// Multi-line input widget.
///
/// Owns the text buffer (a `Vec<String>` of lines) and a `(row, col)`
/// cursor. Designed to be cheap to drive from unit tests without any
/// terminal backend.
#[derive(Debug, Clone)]
pub struct InputWidget {
    /// Logical editable lines. Large paste blocks are stored as one private-use
    /// marker character here and expanded only by [`Self::text`].
    lines: Vec<String>,
    /// Display projection of [`Self::lines`]: paste markers are replaced with
    /// their compact `(N lines, M chars pasted)` summary. Kept in sync after
    /// every edit so callers that inspect `lines()` never see marker glyphs.
    display_lines: Vec<String>,
    /// Monotonic token for cheap change detection without expanding collapsed
    /// paste payloads into temporary `String`s on every key event.
    content_revision: u64,
    row: usize,
    col: usize,
    /// Number of clipboard images awaiting submission.
    image_count: usize,
    /// Hidden payloads for collapsed paste chips. Bodies are shared through
    /// `Arc<String>` so undo/redo snapshots never duplicate multi-KB/MB pastes,
    /// while owned paste events can move their existing allocation in place.
    paste_blocks: Vec<PasteBlock>,
    next_paste_id: u32,
    /// Pre-edit snapshots for undo (most recent last). Capped at
    /// [`MAX_UNDO`]; consecutive same-kind character edits coalesce
    /// into one entry (see [`InputWidget::checkpoint`]).
    undo_stack: Vec<EditSnapshot>,
    /// States popped by undo, available for redo until the next edit.
    redo_stack: Vec<EditSnapshot>,
    /// Kind of the previous buffer edit, used to coalesce undo steps.
    last_edit: Option<EditKind>,
    /// Whether prompt history exists to recall — appends a discoverability
    /// hint (`↑ history`) to the empty-buffer placeholder, CC-style.
    history_hint: bool,
    /// Recently killed text (readline kill ring), most recent last. Every
    /// word/line kill (`Ctrl-W`/`Alt-D`/`Ctrl-K`/`Ctrl-U`) lands here and
    /// `Ctrl-Y` yanks the most recent entry back at the cursor. Bounded by
    /// [`MAX_KILL_RING`].
    kill_ring: Vec<String>,
}

/// Pastes longer than this many lines are collapsed in the input
/// display and expanded only at submit time.
const COLLAPSE_THRESHOLD: usize = 10;

/// Pastes longer than this many characters are collapsed even when they
/// span few/no lines. Image/base64/binary clipboard payloads arrive as one
/// huge single line, so the line-count gate alone would send them through the
/// inline insert path and freeze the UI; collapsing on size keeps large pastes
/// off the O(N) per-line insert entirely.
const COLLAPSE_CHAR_THRESHOLD: usize = 2000;

/// Maximum number of undo checkpoints retained. Older entries are
/// dropped once the stack exceeds this depth, bounding memory.
const MAX_UNDO: usize = 256;

/// Maximum kill-ring entries retained (readline keeps a short ring too;
/// only the most recent entry is yankable today).
const MAX_KILL_RING: usize = 10;

/// Start of Unicode Plane 15 Private Use Area. Each large paste gets one
/// private marker char in the logical buffer; render/text paths resolve it via
/// `paste_blocks`, so the payload never has to be copied into editable lines.
const PASTE_MARKER_BASE: u32 = 0xF0000;
const PASTE_MARKER_END: u32 = 0xFFFFD;

/// A hidden large-paste payload rendered as a compact chip in the composer.
#[derive(Debug, Clone)]
struct PasteBlock {
    id: u32,
    body: Arc<String>,
    summary: String,
}

impl PasteBlock {
    fn summary(&self) -> &str {
        &self.summary
    }
}

/// A reversible snapshot of the editor: logical lines, hidden paste payloads,
/// and cursor. Paste bodies are reference-counted, so cloning snapshots shares
/// their `String` allocations.
#[derive(Debug, Clone)]
struct EditSnapshot {
    lines: Vec<String>,
    paste_blocks: Vec<PasteBlock>,
    next_paste_id: u32,
    row: usize,
    col: usize,
}

/// Classifies an edit so consecutive same-kind character edits coalesce
/// into a single undo step, while structural edits stay discrete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditKind {
    /// Single-character insertion.
    Insert,
    /// Single-character deletion (backspace).
    Delete,
    /// A multi-character structural edit (word/line kill, newline,
    /// paste) — never coalesces with its neighbours.
    Chunk,
}

impl Default for InputWidget {
    fn default() -> Self {
        Self::new()
    }
}

impl InputWidget {
    /// Construct an empty widget.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            display_lines: vec![String::new()],
            content_revision: 0,
            row: 0,
            col: 0,
            image_count: 0,
            paste_blocks: Vec::new(),
            next_paste_id: 0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            last_edit: None,
            history_hint: false,
            kill_ring: Vec::new(),
        }
    }

    /// Toggle the placeholder's `↑ history` hint (set once prompt history
    /// is known to be non-empty).
    pub fn set_history_hint(&mut self, available: bool) {
        self.history_hint = available;
    }

    /// Clamp `self.row` so it is always within `self.lines` bounds.
    ///
    /// Called defensively before every direct `self.lines[self.row]`
    /// access to prevent panics if an edge case breaks the invariant.
    #[inline]
    fn clamp_row(&mut self) {
        if self.row >= self.lines.len() {
            self.row = self.lines.len().saturating_sub(1);
        }
    }

    fn paste_block(&self, id: u32) -> Option<&PasteBlock> {
        self.paste_blocks.iter().find(|block| block.id == id)
    }

    fn display_line_for(&self, line: &str) -> String {
        let mut out = String::new();
        for ch in line.chars() {
            if let Some(id) = paste_id_from_marker(ch) {
                if let Some(block) = self.paste_block(id) {
                    out.push_str(block.summary());
                    continue;
                }
            }
            out.push(ch);
        }
        out
    }

    fn refresh_display_lines(&mut self) {
        self.retain_referenced_paste_blocks();
        self.display_lines = self
            .lines
            .iter()
            .map(|line| self.display_line_for(line))
            .collect();
        if self.display_lines.is_empty() {
            self.display_lines.push(String::new());
        }
        self.content_revision = self.content_revision.wrapping_add(1);
    }

    fn retain_referenced_paste_blocks(&mut self) {
        let mut referenced = Vec::new();
        for line in &self.lines {
            for ch in line.chars() {
                if let Some(id) = paste_id_from_marker(ch) {
                    if !referenced.contains(&id) {
                        referenced.push(id);
                    }
                }
            }
        }
        self.paste_blocks
            .retain(|block| referenced.contains(&block.id));
    }

    fn insert_char_raw(&mut self, ch: char) {
        self.clamp_row();
        let line = &mut self.lines[self.row];
        let byte_idx = char_byte_index(line, self.col);
        line.insert(byte_idx, ch);
        self.col += 1;
    }

    fn insert_newline_raw(&mut self) {
        self.clamp_row();
        let byte_idx = char_byte_index(&self.lines[self.row], self.col);
        let right = self.lines[self.row].split_off(byte_idx);
        self.lines.insert(self.row + 1, right);
        self.row += 1;
        self.col = 0;
    }

    fn insert_collapsed_paste(
        &mut self,
        body: Arc<String>,
        line_count: usize,
        char_count: usize,
    ) -> bool {
        let id = self.next_paste_id;
        let Some(marker) = paste_marker_for_id(id) else {
            return false;
        };
        self.next_paste_id = self.next_paste_id.saturating_add(1);
        self.paste_blocks.push(PasteBlock {
            id,
            body,
            summary: format!("({line_count} lines, {char_count} chars pasted)"),
        });
        self.insert_char_raw(marker);
        true
    }

    fn push_text_payload_for_line(&self, line: &str, out: &mut String) {
        for ch in line.chars() {
            if let Some(id) = paste_id_from_marker(ch) {
                if let Some(block) = self.paste_block(id) {
                    out.push_str(&block.body);
                    continue;
                }
            }
            out.push(ch);
        }
    }

    fn payload_capacity_for_line(&self, line: &str) -> usize {
        line.chars()
            .map(|ch| {
                paste_id_from_marker(ch)
                    .and_then(|id| self.paste_block(id))
                    .map_or_else(|| ch.len_utf8(), |block| block.body.len())
            })
            .sum()
    }

    fn paste_display_ranges_for_line(&self, line: &str) -> Vec<(usize, usize)> {
        let mut ranges = Vec::new();
        let mut display_col = 0usize;
        for ch in line.chars() {
            if let Some(id) = paste_id_from_marker(ch) {
                if let Some(block) = self.paste_block(id) {
                    let len = block.summary().chars().count();
                    ranges.push((display_col, display_col + len));
                    display_col += len;
                    continue;
                }
            }
            display_col += 1;
        }
        ranges
    }

    fn display_width_until_logical_col(&self, line: &str, col: usize) -> u16 {
        let mut width = 0u16;
        for ch in line.chars().take(col) {
            if let Some(id) = paste_id_from_marker(ch) {
                if let Some(block) = self.paste_block(id) {
                    width = width.saturating_add(display_width(block.summary()));
                    continue;
                }
            }
            width = width.saturating_add(char_display_width(ch));
        }
        width
    }

    /// `true` if the text buffer contains no characters. Pending images are
    /// deliberately ignored, matching `text().is_empty()` without allocation.
    #[must_use]
    pub fn is_text_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines.first().is_none_or(String::is_empty)
    }

    /// `true` if the buffer contains no characters and no images.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_text_empty() && self.image_count == 0
    }

    /// Monotonic content-change token that never expands hidden paste bodies.
    #[must_use]
    pub const fn content_revision(&self) -> u64 {
        self.content_revision
    }

    /// `true` when at least one large paste is represented by a compact chip.
    #[must_use]
    pub fn has_collapsed_paste(&self) -> bool {
        !self.paste_blocks.is_empty()
    }

    /// Join all lines into a single submit string using `\n` separators.
    /// Collapsed paste chips are expanded only in this output path.
    #[must_use]
    pub fn text(&self) -> String {
        let capacity = self
            .lines
            .iter()
            .map(|line| self.payload_capacity_for_line(line))
            .sum::<usize>()
            .saturating_add(self.lines.len().saturating_sub(1));
        let mut out = String::with_capacity(capacity);
        for (idx, line) in self.lines.iter().enumerate() {
            if idx > 0 {
                out.push('\n');
            }
            self.push_text_payload_for_line(line, &mut out);
        }
        out
    }

    /// Increment the pending image counter (called after clipboard image read).
    pub fn add_image(&mut self) {
        self.image_count += 1;
    }

    /// Current pending image count.
    #[must_use]
    pub const fn image_count(&self) -> usize {
        self.image_count
    }

    /// Remove the last pending image. Returns `true` if one was removed.
    pub fn remove_last_image(&mut self) -> bool {
        if self.image_count > 0 {
            self.image_count -= 1;
            true
        } else {
            false
        }
    }

    /// Current `(row, col)` cursor position in character units.
    #[must_use]
    pub const fn cursor(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    /// Read-only view of the display line buffer. Large hidden paste payloads
    /// appear here as compact summary chips, never as private marker glyphs.
    #[must_use]
    #[allow(
        clippy::misnamed_getters,
        reason = "public API returns the display-line view, not the raw `lines` field"
    )]
    pub fn lines(&self) -> &[String] {
        &self.display_lines
    }

    /// Clear the buffer and reset the cursor to the origin.
    pub fn clear(&mut self) {
        self.lines.clear();
        self.lines.push(String::new());
        self.display_lines.clear();
        self.display_lines.push(String::new());
        self.row = 0;
        self.col = 0;
        self.image_count = 0;
        self.paste_blocks.clear();
        self.next_paste_id = 0;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.last_edit = None;
        self.content_revision = self.content_revision.wrapping_add(1);
    }

    /// Remove leaked SGR mouse event bytes from the editable text.
    ///
    /// When the terminal event loop is starved, crossterm can occasionally surface
    /// the tail of a mouse event as printable chars (`[<35;24;26M`). Keep the
    /// cleanup narrow so ordinary typed text is left alone.
    pub fn strip_sgr_mouse_sequences(&mut self) -> bool {
        let Some(cleaned) = strip_sgr_mouse_sequences_from_text(&self.display_lines.join("\n"))
        else {
            return false;
        };
        let image_count = self.image_count;
        self.lines = cleaned.split('\n').map(ToOwned::to_owned).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.row = self.row.min(self.lines.len().saturating_sub(1));
        self.col = self.col.min(self.lines[self.row].chars().count());
        self.image_count = image_count;
        self.paste_blocks.clear();
        self.next_paste_id = 0;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.last_edit = None;
        self.refresh_display_lines();
        true
    }

    /// Insert a single character at the cursor.
    pub fn insert_char(&mut self, ch: char) {
        self.insert_char_raw(ch);
        self.refresh_display_lines();
    }

    /// Insert a borrowed block of text at the cursor (e.g. from a modal).
    ///
    /// Production paste events should prefer [`Self::insert_text_owned`] so a
    /// collapsed multi-MB payload can retain the event's existing allocation.
    pub fn insert_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.checkpoint(EditKind::Chunk);
        let (line_count, char_count) = paste_shape(text);
        let should_collapse =
            line_count > COLLAPSE_THRESHOLD || char_count > COLLAPSE_CHAR_THRESHOLD;
        if should_collapse && paste_marker_for_id(self.next_paste_id).is_some() {
            let inserted = self.insert_collapsed_paste(
                Arc::new(text.to_owned()),
                line_count,
                char_count,
            );
            debug_assert!(inserted);
            self.refresh_display_lines();
            return;
        }
        self.insert_inline_text(text);
        self.refresh_display_lines();
    }

    /// Insert an owned paste payload, moving its allocation into a collapsed
    /// paste block instead of copying it while the event buffer is still live.
    pub fn insert_text_owned(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        self.checkpoint(EditKind::Chunk);
        let (line_count, char_count) = paste_shape(&text);
        let should_collapse =
            line_count > COLLAPSE_THRESHOLD || char_count > COLLAPSE_CHAR_THRESHOLD;
        if should_collapse && paste_marker_for_id(self.next_paste_id).is_some() {
            let inserted =
                self.insert_collapsed_paste(Arc::new(text), line_count, char_count);
            debug_assert!(inserted);
            self.refresh_display_lines();
            return;
        }
        self.insert_inline_text(&text);
        self.refresh_display_lines();
    }

    /// Insert a sub-threshold paste in bulk so it stays O(N) rather than the
    /// O(N²) cost of repeated single-character inserts.
    fn insert_inline_text(&mut self, text: &str) {
        for (i, chunk) in text.split('\n').enumerate() {
            if i > 0 {
                self.insert_newline_raw();
            }
            self.clamp_row();
            let byte_idx = char_byte_index(&self.lines[self.row], self.col);
            self.lines[self.row].insert_str(byte_idx, chunk);
            self.col += chunk.chars().count();
        }
    }

    /// Insert a newline at the cursor, splitting the current line.
    pub fn insert_newline(&mut self) {
        self.insert_newline_raw();
        self.refresh_display_lines();
    }

    /// Backspace at the cursor.
    pub fn backspace(&mut self) {
        self.clamp_row();
        if self.col > 0 {
            let line = &mut self.lines[self.row];
            let prev_byte = char_byte_index(line, self.col - 1);
            let curr_byte = char_byte_index(line, self.col);
            line.replace_range(prev_byte..curr_byte, "");
            self.col -= 1;
        } else if self.row > 0 {
            let current = self.lines.remove(self.row);
            self.row -= 1;
            let prev_len = self.lines[self.row].chars().count();
            self.lines[self.row].push_str(&current);
            self.col = prev_len;
        }
        self.refresh_display_lines();
    }

    /// Move cursor left by one character.
    pub fn move_left(&mut self) {
        self.last_edit = None;
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = self.lines[self.row].chars().count();
        }
    }

    /// Move cursor right by one character.
    pub fn move_right(&mut self) {
        self.last_edit = None;
        let line_len = self.lines[self.row].chars().count();
        if self.col < line_len {
            self.col += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = 0;
        }
    }

    /// Move cursor up by one line, clamping column.
    pub fn move_up(&mut self) {
        self.last_edit = None;
        if self.row > 0 {
            self.row -= 1;
            let len = self.lines[self.row].chars().count();
            if self.col > len {
                self.col = len;
            }
        }
    }

    /// Move cursor down by one line, clamping column.
    pub fn move_down(&mut self) {
        self.last_edit = None;
        if self.row + 1 < self.lines.len() {
            self.row += 1;
            let len = self.lines[self.row].chars().count();
            if self.col > len {
                self.col = len;
            }
        }
    }

    /// Move cursor to the start of the next word (vim `w`).
    pub fn move_word_forward(&mut self) {
        self.last_edit = None;
        self.clamp_row();
        let line: Vec<char> = self.lines[self.row].chars().collect();
        let len = line.len();
        if self.col >= len {
            // Move to next line if possible.
            if self.row + 1 < self.lines.len() {
                self.row += 1;
                self.col = 0;
            }
            return;
        }
        let mut pos = self.col;
        // Skip current word characters.
        while pos < len && !line[pos].is_whitespace() {
            pos += 1;
        }
        // Skip whitespace.
        while pos < len && line[pos].is_whitespace() {
            pos += 1;
        }
        if pos <= len {
            self.col = pos;
        }
    }

    /// Move cursor to the start of the previous word (vim `b`).
    pub fn move_word_backward(&mut self) {
        self.last_edit = None;
        self.clamp_row();
        if self.col == 0 {
            if self.row > 0 {
                self.row -= 1;
                self.col = self.lines[self.row].chars().count();
            }
            return;
        }
        let line: Vec<char> = self.lines[self.row].chars().collect();
        let mut pos = self.col;
        // Skip whitespace backward.
        while pos > 0 && line[pos - 1].is_whitespace() {
            pos -= 1;
        }
        // Skip word characters backward.
        while pos > 0 && !line[pos - 1].is_whitespace() {
            pos -= 1;
        }
        self.col = pos;
    }

    /// Move cursor to start of line (vim `0`).
    pub fn move_to_line_start(&mut self) {
        self.col = 0;
    }

    /// Move cursor to end of line (vim `$`).
    pub fn move_to_line_end(&mut self) {
        self.clamp_row();
        let len = self.lines[self.row].chars().count();
        self.col = len;
    }

    // ── Word / line deletions (readline-style) ──────────────────────

    /// Delete from the cursor back to the start of the previous word
    /// (readline `Ctrl-W` / `Alt-Backspace`). At column 0 it joins with
    /// the previous line, like a plain backspace.
    pub fn delete_word_backward(&mut self) {
        self.clamp_row();
        if self.col == 0 {
            self.backspace();
            return;
        }
        let line: Vec<char> = self.lines[self.row].chars().collect();
        let mut pos = self.col;
        // Skip whitespace immediately left of the cursor, then the word.
        while pos > 0 && line[pos - 1].is_whitespace() {
            pos -= 1;
        }
        while pos > 0 && !line[pos - 1].is_whitespace() {
            pos -= 1;
        }
        let start = char_byte_index(&self.lines[self.row], pos);
        let end = char_byte_index(&self.lines[self.row], self.col);
        self.record_kill(self.lines[self.row][start..end].to_string());
        self.lines[self.row].replace_range(start..end, "");
        self.col = pos;
        self.refresh_display_lines();
    }

    /// Delete from the cursor forward to the end of the next word
    /// (readline `Alt-D`). At end of line it pulls the next line up.
    pub fn delete_word_forward(&mut self) {
        self.clamp_row();
        let line: Vec<char> = self.lines[self.row].chars().collect();
        let len = line.len();
        if self.col >= len {
            if self.row + 1 < self.lines.len() {
                let next = self.lines.remove(self.row + 1);
                self.lines[self.row].push_str(&next);
                self.refresh_display_lines();
            }
            return;
        }
        let mut pos = self.col;
        // Skip whitespace under/after the cursor, then the word.
        while pos < len && line[pos].is_whitespace() {
            pos += 1;
        }
        while pos < len && !line[pos].is_whitespace() {
            pos += 1;
        }
        let start = char_byte_index(&self.lines[self.row], self.col);
        let end = char_byte_index(&self.lines[self.row], pos);
        self.record_kill(self.lines[self.row][start..end].to_string());
        self.lines[self.row].replace_range(start..end, "");
        self.refresh_display_lines();
    }

    /// Delete from the cursor to the end of the line (readline `Ctrl-K`).
    /// When already at the end, it pulls the next line up (joins).
    pub fn kill_to_line_end(&mut self) {
        self.clamp_row();
        let len = self.lines[self.row].chars().count();
        if self.col < len {
            let start = char_byte_index(&self.lines[self.row], self.col);
            self.record_kill(self.lines[self.row][start..].to_string());
            self.lines[self.row].truncate(start);
        } else if self.row + 1 < self.lines.len() {
            let next = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&next);
        }
        self.refresh_display_lines();
    }

    /// Kill the entire current line, leaving an empty line in its place and
    /// parking the cursor at column 0 (readline `Ctrl-U`, whole-line variant).
    /// The line itself is preserved (cleared) rather than removed so a single
    /// `Ctrl-U` never collapses a multi-line buffer's row count.
    pub fn kill_whole_line(&mut self) {
        self.clamp_row();
        let killed = std::mem::take(&mut self.lines[self.row]);
        self.record_kill(killed);
        self.col = 0;
        self.refresh_display_lines();
    }

    // ── Kill ring (readline yank) ───────────────────────────────────

    /// Save killed text for a later yank. Empty kills are ignored so a
    /// `Ctrl-K` at end of line never clobbers the last real kill.
    fn record_kill(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        self.kill_ring.push(text);
        if self.kill_ring.len() > MAX_KILL_RING {
            self.kill_ring.remove(0);
        }
    }

    /// Insert the most recent kill at the cursor (readline `Ctrl-Y`).
    /// Returns `false` when nothing has been killed yet.
    pub fn yank(&mut self) -> bool {
        let Some(text) = self.kill_ring.last().cloned() else {
            return false;
        };
        self.checkpoint(EditKind::Chunk);
        for ch in text.chars() {
            self.insert_char(ch);
        }
        true
    }

    // ── Undo / redo ─────────────────────────────────────────────────

    /// Undo the most recent edit group. Returns `true` if a state was
    /// restored.
    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.undo_stack.pop() {
            self.redo_stack.push(self.snapshot());
            self.restore(prev);
            self.last_edit = None;
            true
        } else {
            false
        }
    }

    /// Redo the most recently undone edit. Returns `true` if a state was
    /// restored.
    pub fn redo(&mut self) -> bool {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.snapshot());
            self.restore(next);
            self.last_edit = None;
            true
        } else {
            false
        }
    }

    /// Capture the current buffer + cursor as a snapshot.
    fn snapshot(&self) -> EditSnapshot {
        EditSnapshot {
            lines: self.lines.clone(),
            paste_blocks: self.paste_blocks.clone(),
            next_paste_id: self.next_paste_id,
            row: self.row,
            col: self.col,
        }
    }

    /// Restore a previously captured snapshot.
    fn restore(&mut self, snap: EditSnapshot) {
        self.lines = snap.lines;
        self.paste_blocks = snap.paste_blocks;
        self.next_paste_id = snap.next_paste_id;
        self.row = snap.row;
        self.col = snap.col;
        self.clamp_row();
        self.refresh_display_lines();
    }

    /// Record a pre-edit checkpoint for undo. Consecutive edits of the
    /// same character kind ([`EditKind::Insert`] / [`EditKind::Delete`])
    /// coalesce into one entry; [`EditKind::Chunk`] always starts a new
    /// one. Call *before* mutating the buffer. Clears the redo stack.
    fn checkpoint(&mut self, kind: EditKind) {
        let coalesce = kind != EditKind::Chunk && self.last_edit == Some(kind);
        if !coalesce {
            self.undo_stack.push(self.snapshot());
            if self.undo_stack.len() > MAX_UNDO {
                self.undo_stack.remove(0);
            }
        }
        self.redo_stack.clear();
        self.last_edit = Some(kind);
    }

    /// Handle a single key event.
    ///
    /// Returns a command when the event should propagate to the app
    /// loop (submit / cancel); otherwise `None`.
    ///
    /// Key bindings:
    /// * `Enter` — submit
    /// * `Shift+Enter` or `Alt+Enter` — insert newline
    /// * `Ctrl-C` — cancel (clears buffer)
    /// * `Ctrl-Z` / `Ctrl-Y` — undo / redo
    /// * `Ctrl-W` / `Alt-Backspace` — delete previous word
    /// * `Alt-D` — delete next word; `Ctrl-K` — kill to end of line
    /// * `Alt-B`/`Alt-F` or `Ctrl`/`Alt`+`←`/`→` — word-wise motion
    /// * `Ctrl-A`/`Ctrl-E` — move to line start / end
    /// * `Ctrl-U` — kill the whole line
    /// * `Backspace` / arrow keys / printable chars — edit
    ///
    /// Note: `Ctrl-A/E/U` and `Home`/`End` double as transcript-navigation
    /// bindings in the app loop. The app loop only forwards them here once
    /// the input buffer is non-empty (see `app::handle_key`); on an empty
    /// buffer they scroll / toggle the sidebar / open the editor instead.
    pub fn handle_key(&mut self, key: KeyEvent) -> Option<InputCommand> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);

        // Readline-style editing chords are resolved first; they never
        // submit or cancel, so a handled chord short-circuits to `None`.
        if (ctrl || alt) && self.handle_editing_chord(key.code, ctrl, alt) {
            return None;
        }

        match key.code {
            KeyCode::Char('c') if ctrl => {
                self.clear();
                Some(InputCommand::Cancel)
            }
            KeyCode::Enter if shift || alt => {
                self.checkpoint(EditKind::Chunk);
                self.insert_newline();
                None
            }
            KeyCode::Enter => {
                if self.is_empty() {
                    None
                } else {
                    let text = self.text();
                    self.clear();
                    Some(InputCommand::Submit(text))
                }
            }
            KeyCode::Backspace => {
                self.checkpoint(EditKind::Delete);
                self.backspace();
                None
            }
            KeyCode::Left => {
                self.move_left();
                None
            }
            KeyCode::Right => {
                self.move_right();
                None
            }
            KeyCode::Up => {
                self.move_up();
                None
            }
            KeyCode::Down => {
                self.move_down();
                None
            }
            KeyCode::Char(ch) if !ctrl => {
                // Whitespace starts a fresh undo group so that undo is
                // word-granular rather than wiping a whole typing burst.
                let kind = if ch.is_whitespace() {
                    EditKind::Chunk
                } else {
                    EditKind::Insert
                };
                self.checkpoint(kind);
                self.insert_char(ch);
                None
            }
            _ => None,
        }
    }

    /// Resolve a `Ctrl`/`Alt` editing chord (word/line kills, word-wise
    /// motion, undo/redo). Returns `true` when the chord was recognised
    /// and applied; `false` lets the caller fall through to plain key
    /// handling (e.g. so `Ctrl-C` can still cancel).
    fn handle_editing_chord(&mut self, code: KeyCode, ctrl: bool, alt: bool) -> bool {
        match code {
            KeyCode::Char('z') if ctrl => {
                self.undo();
                true
            }
            // Redo on Alt-Z (pairs with Ctrl-Z undo). Ctrl-Y is the readline
            // yank below — its old redo binding was unreachable anyway (the
            // app layer consumed Ctrl-Y globally before the composer saw it).
            KeyCode::Char('z') if alt => {
                self.redo();
                true
            }
            // Yank the most recent kill (readline `Ctrl-Y`, completing the
            // Ctrl-K/Ctrl-U/Ctrl-W kill set). Only reachable while the
            // composer is non-empty — an empty buffer keeps Ctrl-Y as the
            // app-level "copy last message" binding (same split as Ctrl-A/E).
            KeyCode::Char('y') if ctrl => {
                self.yank();
                true
            }
            // Delete the previous word: Ctrl-W, or Ctrl/Alt-Backspace.
            KeyCode::Char('w') if ctrl => {
                self.checkpoint(EditKind::Chunk);
                self.delete_word_backward();
                true
            }
            KeyCode::Backspace if ctrl || alt => {
                self.checkpoint(EditKind::Chunk);
                self.delete_word_backward();
                true
            }
            // Delete the next word (Alt-D).
            KeyCode::Char('d') if alt => {
                self.checkpoint(EditKind::Chunk);
                self.delete_word_forward();
                true
            }
            // Kill to end of line (Ctrl-K).
            KeyCode::Char('k') if ctrl => {
                self.checkpoint(EditKind::Chunk);
                self.kill_to_line_end();
                true
            }
            // Kill the whole line (Ctrl-U). Checkpointed so it is undoable.
            KeyCode::Char('u') if ctrl => {
                self.checkpoint(EditKind::Chunk);
                self.kill_whole_line();
                true
            }
            // Line-start / line-end motion (readline Ctrl-A / Ctrl-E).
            // Live insert path: the app loop only forwards these once the
            // input buffer is non-empty (empty buffer keeps them as
            // transcript scroll / sidebar / editor bindings).
            KeyCode::Char('a') if ctrl => {
                self.move_to_line_start();
                true
            }
            KeyCode::Char('e') if ctrl => {
                self.move_to_line_end();
                true
            }
            // Word-wise motion: Alt-B/F or Ctrl/Alt + arrow.
            KeyCode::Char('b') if alt => {
                self.move_word_backward();
                true
            }
            KeyCode::Char('f') if alt => {
                self.move_word_forward();
                true
            }
            KeyCode::Left if ctrl || alt => {
                self.move_word_backward();
                true
            }
            KeyCode::Right if ctrl || alt => {
                self.move_word_forward();
                true
            }
            _ => false,
        }
    }

    /// Width of the leading gutter (prompt + mode tag) shown on
    /// the first row and mirrored as indentation on wrapped/continuation rows.
    /// Shared by [`Self::draw`] and [`Self::desired_rows`] so the soft-wrap row
    /// count always matches what is actually drawn.
    fn gutter(mode: AppMode) -> u16 {
        PROMPT_WIDTH + display_width(mode_tag_for(mode))
    }

    /// Pending-image badge text shown after the prompt on the first row, if any.
    fn badge_text(&self) -> Option<String> {
        match self.image_count {
            0 => None,
            1 => Some("[image: image/png] ".to_string()),
            n => Some(format!("[{n} images] ")),
        }
    }

    /// Desired number of rows for this input given its current buffer, soft-wrapped
    /// to `inner_width`. Includes 2 rows for the top/bottom border and is clamped
    /// to `INPUT_MIN_ROWS..=INPUT_MAX_ROWS`; content beyond the cap scrolls inside
    /// the box (see [`Self::draw`]). The caller feeds this into
    /// [`super::layout::LayoutRegions::compute_with_sidebar`] so the box auto-grows.
    #[must_use]
    pub fn desired_rows(&self, inner_width: u16, mode: AppMode) -> u16 {
        let gutter = Self::gutter(mode);
        let badge_w = self.badge_text().as_deref().map_or(0, display_width);
        let text_w = inner_width.saturating_sub(gutter).max(1);
        let mut rows: usize = 0;
        for (i, line) in self.display_lines.iter().enumerate() {
            let first_w = if i == 0 {
                text_w.saturating_sub(badge_w).max(1)
            } else {
                text_w
            };
            rows += wrap_segments(line, first_w, text_w).len();
        }
        let n = u16::try_from(rows)
            .unwrap_or(INPUT_MAX_ROWS)
            .saturating_add(2);
        n.clamp(INPUT_MIN_ROWS, INPUT_MAX_ROWS)
    }

    /// Render the widget into `area` using the current theme.
    ///
    /// A heavy left rail is the composer's single focused structural cue. A
    /// placeholder hint is shown when the buffer is empty. The terminal's
    /// hardware cursor is placed via
    /// `frame.set_cursor_position` inside the input area.
    pub fn draw(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        theme: &Theme,
        mode: &AppMode,
    ) {
        self.draw_with_heat(frame, area, theme, mode, HeatState::Cold);
    }

    /// Render the widget with draw-time chrome temperature styling.
    #[allow(clippy::too_many_lines)] // cohesive input panel render
    pub fn draw_with_heat(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        theme: &Theme,
        mode: &AppMode,
        heat_state: HeatState,
    ) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        // Hand-painted input rail: removing `Borders::LEFT` lets every row carry
        // its own heat color. Moving that one cell into left padding preserves
        // the exact H1 inner rect (x + 1, width - 2, y + 1, height - 2).
        let focused = matches!(mode, AppMode::Normal);
        let block = input_block();

        // Compute the inner area (content area inside the border).
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let rail_glyph = glyphs::pick(
            !theme.no_color,
            glyphs::ZO_RAIL,
            glyphs::ZO_RAIL_NC,
        );
        let rail = (0..area.height)
            .map(|row| {
                let distance_from_bottom = usize::from(area.height - 1 - row);
                Line::from(Span::styled(
                    rail_glyph,
                    input_rail_style(theme, heat_state, focused, distance_from_bottom),
                ))
            })
            .collect::<Vec<_>>();
        frame.render_widget(
            Paragraph::new(rail),
            Rect::new(area.x, area.y, 1, area.height),
        );

        if inner.width == 0 || inner.height == 0 {
            return;
        }

        // Wipe stale glyphs in this widget's inner region before repainting.
        // The per-frame full-screen Clear is disabled (see `App::draw`) for the
        // diff-buffer perf win, but `Paragraph` only writes cells that carry
        // text — when the content shrinks (long line → short text/placeholder,
        // or an image badge is removed) the old trailing glyphs linger as a
        // ghost. Clearing just this rect fixes it without the whole-frame cost.
        frame.render_widget(Clear, inner);

        // --- Styles ---------------------------------------------------------
        let body_style = theme.typography.body;
        let heat = theme.heat();
        let prompt_color = match heat_state {
            HeatState::Cold => theme.palette.bright,
            HeatState::Hot => heat.ember,
            HeatState::Cooling { ramp_idx } => heat.ramp[ramp_idx.min(heat.ramp.len() - 1)],
        };
        let prompt_style = Style::new()
            .fg(prompt_color)
            .add_modifier(Modifier::BOLD);

        let mode_tag = mode_tag_for(*mode);
        let gutter = Self::gutter(*mode);
        if inner.width <= gutter {
            return;
        }

        // Style for image badges — cyan+bold so they stand out.
        let image_style = theme.typography.heading_2;
        // Style for collapsed paste summary — dim but readable.
        let paste_style = theme.typography.dim;

        let image_badge = self.badge_text();
        let badge_w = image_badge.as_deref().map_or(0, display_width);
        let has_text = !(self.lines.len() == 1 && self.lines.first().is_none_or(String::is_empty));
        let text_w = inner.width.saturating_sub(gutter).max(1);

        // Build display rows by soft-wrapping each logical line, recording the
        // cursor's display position `(row index, x cells from inner.x)` so the
        // hardware cursor follows the wrapped text instead of clipping off-box.
        let mut lines: Vec<Line<'_>> = Vec::new();
        let mut cursor_disp: Option<(usize, u16)> = None;

        if !has_text && image_badge.is_none() {
            // Empty buffer — show placeholder hint.
            let mut spans = Vec::new();
            spans.push(Span::styled(PROMPT, prompt_style));
            spans.push(Span::styled(mode_tag, prompt_style));
            spans.push(Span::styled(
                PLACEHOLDER.trim(),
                theme.typography.placeholder,
            ));
            if self.history_hint {
                spans.push(Span::styled(
                    " \u{00b7} \u{2191} history",
                    theme.typography.placeholder,
                ));
            }
            lines.push(Line::from(spans));
            cursor_disp = Some((0, gutter));
        } else {
            for (lrow, text) in self.display_lines.iter().enumerate() {
                let first_w = if lrow == 0 {
                    text_w.saturating_sub(badge_w).max(1)
                } else {
                    text_w
                };
                let segments = wrap_segments(text, first_w, text_w);
                let seg_count = segments.len();
                let logical_line = self.lines.get(lrow).map_or("", String::as_str);
                let cursor_cells = if lrow == self.row {
                    self.display_width_until_logical_col(logical_line, self.col)
                } else {
                    0
                };
                let paste_ranges = self.paste_display_ranges_for_line(logical_line);
                for (seg_i, (start_char, seg_text)) in segments.into_iter().enumerate() {
                    let is_first_overall = lrow == 0 && seg_i == 0;
                    let (lead, lead_w) = if is_first_overall {
                        let mut v = Vec::new();
                        v.push(Span::styled(PROMPT, prompt_style));
                        v.push(Span::styled(mode_tag, prompt_style));
                        if let Some(ref badge) = image_badge {
                            v.push(Span::styled(badge.clone(), image_style));
                        }
                        (v, gutter + badge_w)
                    } else {
                        (vec![Span::raw(" ".repeat(usize::from(gutter)))], gutter)
                    };
                    // Cursor mapping: the logical cursor counts a collapsed paste
                    // as one atom, while the display row expands it to its summary
                    // chip. Compare terminal cell widths, not raw char indices.
                    if lrow == self.row {
                        let seg_start_cells = display_width(char_prefix(text, start_char));
                        let seg_end_cells =
                            seg_start_cells.saturating_add(display_width(&seg_text));
                        let is_last = seg_i + 1 == seg_count;
                        if cursor_cells >= seg_start_cells
                            && (cursor_cells < seg_end_cells || is_last)
                        {
                            let x = lead_w + cursor_cells.saturating_sub(seg_start_cells);
                            cursor_disp = Some((lines.len(), x));
                        }
                    }
                    let mut spans = lead;
                    push_styled_display_segment(
                        &mut spans,
                        &seg_text,
                        start_char,
                        &paste_ranges,
                        body_style,
                        paste_style,
                    );
                    lines.push(Line::from(spans));
                }
            }
        }

        // Vertical scroll so the cursor row stays inside the visible inner box
        // even when the wrapped content exceeds the box height.
        let view_h = inner.height.max(1);
        let total = u16::try_from(lines.len()).unwrap_or(u16::MAX);
        let cursor_row = cursor_disp.map_or(0, |(r, _)| u16::try_from(r).unwrap_or(0));
        let scroll = if total > view_h && cursor_row >= view_h {
            cursor_row + 1 - view_h
        } else {
            0
        };

        let paragraph = Paragraph::new(lines).style(body_style).scroll((scroll, 0));
        frame.render_widget(paragraph, inner);

        // Place the hardware cursor at its wrapped display position.
        if let Some((drow, x)) = cursor_disp {
            let drow = u16::try_from(drow).unwrap_or(0);
            if drow >= scroll && drow < scroll.saturating_add(view_h) {
                let cy = inner.y + (drow - scroll);
                let cx = (inner.x + x).min(inner.x + inner.width.saturating_sub(1));
                frame.set_cursor_position(Position { x: cx, y: cy });
            }
        }
    }
}

fn input_block() -> Block<'static> {
    Block::default().padding(Padding::new(1, 1, 1, 1))
}

fn input_rail_style(
    theme: &Theme,
    heat_state: HeatState,
    focused: bool,
    distance_from_bottom: usize,
) -> Style {
    let heat = theme.heat();
    let color = match heat_state {
        HeatState::Hot => heat.rail_fade[distance_from_bottom.min(heat.rail_fade.len() - 1)],
        HeatState::Cold if focused => heat.steel,
        HeatState::Cold => theme.palette.faint,
        HeatState::Cooling { ramp_idx } => heat.ramp[ramp_idx.min(heat.ramp.len() - 1)],
    };
    if theme.no_color || color == Color::Reset {
        Style::default()
    } else {
        Style::default().fg(color)
    }
}

/// Mode marker appended to the prompt (e.g. `❯ /model `). Shared by
/// [`InputWidget::draw`] and [`InputWidget::gutter`].
fn mode_tag_for(mode: AppMode) -> &'static str {
    match mode {
        // Normal/overlay modes, the generic argument picker, and the report
        // popup show no input-row tag — their border captions already name
        // the command (e.g. `/theme`, `/mcp`).
        AppMode::Normal
        | AppMode::Pager
        | AppMode::Focus
        | AppMode::ModalArgPick
        | AppMode::ModalReport => "",
        AppMode::ModalModel => "/model ",
        AppMode::ModalPermissions => "/perm ",
        AppMode::ModalChoice => "/choose ",
        AppMode::ModalQuestion => "/question ",
        AppMode::ModalSession => "/resume ",
        AppMode::ModalLogin => "/login ",
        AppMode::ModalApiKey | AppMode::ModalCustomProvider => "/connect ",
        AppMode::ModalEffort => "/effort ",
        AppMode::ModalDiff => "/diff ",
        AppMode::ModalHunks => "/hunks ",
        AppMode::ModalRewind => "rewind ",
        AppMode::ModalConfirmRewind => "rewind? ",
        AppMode::ModalWorkflow => "workflow ",
        AppMode::ModalAgents => "agents ",
        AppMode::ModalTeamInbox => "inbox ",
        AppMode::ModalTools => "/tools ",
        AppMode::ModalUsage => "/usage ",
        AppMode::ModalSmartSettings => "/smart ",
        AppMode::ModalDeepTier => "/tier ",
        AppMode::ModalRemoteOnboarding => "/remote ",
        AppMode::ModalFile => "@file ",
        AppMode::Search => "/search ",
    }
}

/// Soft-wrap one logical line into display segments that each fit within the
/// given cell widths (`first_width` for the first segment, `cont_width` for
/// wrapped continuations). Returns `(start_char_index, segment_text)` per
/// display row, wrapping on character boundaries using terminal cell width so
/// CJK / full-width glyphs count as two columns. An empty line yields a single
/// empty segment so the cursor still has a row to land on.
fn wrap_segments(line: &str, first_width: u16, cont_width: u16) -> Vec<(usize, String)> {
    if line.is_empty() {
        return vec![(0, String::new())];
    }
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut seg_start = 0usize;
    let mut cur = String::new();
    let mut cur_w: u16 = 0;
    for (idx, ch) in line.chars().enumerate() {
        let limit = if out.is_empty() {
            first_width
        } else {
            cont_width
        }
        .max(1);
        let w = u16::try_from(UnicodeWidthChar::width(ch).unwrap_or(0)).unwrap_or(0);
        if cur_w + w > limit && !cur.is_empty() {
            out.push((seg_start, std::mem::take(&mut cur)));
            seg_start = idx;
            cur_w = 0;
        }
        cur.push(ch);
        cur_w = cur_w.saturating_add(w);
    }
    out.push((seg_start, cur));
    out
}

fn push_styled_display_segment(
    spans: &mut Vec<Span<'static>>,
    segment: &str,
    segment_start_char: usize,
    paste_ranges: &[(usize, usize)],
    body_style: Style,
    paste_style: Style,
) {
    if segment.is_empty() {
        spans.push(Span::styled(String::new(), body_style));
        return;
    }

    let segment_len = segment.chars().count();
    let segment_end = segment_start_char + segment_len;
    let mut cursor = segment_start_char;
    while cursor < segment_end {
        let in_paste = paste_ranges
            .iter()
            .any(|(start, end)| cursor >= *start && cursor < *end);
        let mut next = segment_end;
        for (start, end) in paste_ranges {
            if cursor < *start {
                next = next.min(*start);
            } else if cursor < *end {
                next = next.min(*end);
            }
        }
        next = next.max(cursor + 1).min(segment_end);
        let local_start = cursor - segment_start_char;
        let local_end = next - segment_start_char;
        let piece = char_range_owned(segment, local_start, local_end);
        spans.push(Span::styled(
            piece,
            if in_paste { paste_style } else { body_style },
        ));
        cursor = next;
    }
}

fn char_range_owned(s: &str, start: usize, end: usize) -> String {
    s.chars().skip(start).take(end - start).collect()
}

fn paste_shape(text: &str) -> (usize, usize) {
    let mut line_count = usize::from(!text.ends_with('\n'));
    let mut char_count = 0usize;
    for ch in text.chars() {
        char_count += 1;
        if ch == '\n' {
            line_count += 1;
        }
    }
    (line_count.max(1), char_count)
}

fn paste_marker_for_id(id: u32) -> Option<char> {
    let codepoint = PASTE_MARKER_BASE.checked_add(id)?;
    if codepoint > PASTE_MARKER_END {
        return None;
    }
    char::from_u32(codepoint)
}

fn paste_id_from_marker(ch: char) -> Option<u32> {
    let codepoint = u32::from(ch);
    if (PASTE_MARKER_BASE..=PASTE_MARKER_END).contains(&codepoint) {
        Some(codepoint - PASTE_MARKER_BASE)
    } else {
        None
    }
}

fn char_display_width(ch: char) -> u16 {
    u16::try_from(UnicodeWidthChar::width(ch).unwrap_or(0)).unwrap_or(0)
}

/// Convert a character index to a byte offset inside `s`.
fn char_byte_index(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map_or_else(|| s.len(), |(b, _)| b)
}

fn strip_sgr_mouse_sequences_from_text(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut idx = 0;
    let mut changed = false;
    let mut out = String::with_capacity(text.len());

    while idx < bytes.len() {
        if let Some(len) = sgr_mouse_sequence_len(bytes, idx) {
            idx += len;
            changed = true;
            continue;
        }
        let ch = text[idx..].chars().next().expect("idx on char boundary");
        out.push(ch);
        idx += ch.len_utf8();
    }

    changed.then_some(out)
}

fn sgr_mouse_sequence_len(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start..start + 2) != Some(b"[<") {
        return None;
    }
    let mut idx = start + 2;
    idx = parse_digits(bytes, idx)?;
    if bytes.get(idx) != Some(&b';') {
        return None;
    }
    idx = parse_digits(bytes, idx + 1)?;
    if bytes.get(idx) != Some(&b';') {
        return None;
    }
    idx = parse_digits(bytes, idx + 1)?;
    match bytes.get(idx) {
        Some(b'M' | b'm') => Some(idx + 1 - start),
        _ => None,
    }
}

fn parse_digits(bytes: &[u8], mut idx: usize) -> Option<usize> {
    let start = idx;
    while matches!(bytes.get(idx), Some(b'0'..=b'9')) {
        idx += 1;
    }
    (idx > start).then_some(idx)
}

fn char_prefix(s: &str, char_idx: usize) -> &str {
    let byte_idx = char_byte_index(s, char_idx);
    &s[..byte_idx]
}

fn display_width(s: &str) -> u16 {
    let width = Line::from(s.to_string()).width();
    u16::try_from(width).unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        HeatState, InputWidget, input_block, strip_sgr_mouse_sequences_from_text, wrap_segments,
    };
    use crate::tui::app::AppMode;

    fn type_text(input: &mut InputWidget, text: &str) {
        for ch in text.chars() {
            input.insert_char(ch);
        }
    }

    #[test]
    fn hand_painted_rail_preserves_h1_inner_geometry() {
        use ratatui::layout::Rect;
        use ratatui::widgets::{Block, BorderType, Borders, Padding};

        let area = Rect::new(7, 11, 30, 6);
        let h1 = Block::default()
            .borders(Borders::LEFT)
            .border_type(BorderType::Thick)
            .padding(Padding::new(0, 1, 1, 1));

        assert_eq!(h1.inner(area), Rect::new(8, 12, 28, 4));
        assert_eq!(input_block().inner(area), h1.inner(area));
    }

    #[test]
    fn prompt_and_mode_tag_follow_heat_state() {
        use crate::tui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::layout::Rect;
        use ratatui::style::Modifier;

        let theme = Theme::zo();
        let input = InputWidget::new();
        let cell_at = |heat_state, mode, x| {
            let mut terminal =
                Terminal::new(TestBackend::new(30, 4)).expect("test backend initializes");
            terminal
                .draw(|frame| {
                    input.draw_with_heat(
                        frame,
                        Rect::new(0, 0, 30, 4),
                        &theme,
                        &mode,
                        heat_state,
                    );
                })
                .expect("input draws");
            terminal.backend().buffer()[(x, 1)].clone()
        };

        let cold = cell_at(HeatState::Cold, AppMode::Normal, 1);
        let hot = cell_at(HeatState::Hot, AppMode::Normal, 1);
        assert_eq!(cold.symbol(), "❯");
        assert_eq!(cold.fg, theme.palette.bright);
        assert!(cold.modifier.contains(Modifier::BOLD));
        assert_eq!(hot.symbol(), "❯");
        assert_eq!(hot.fg, theme.heat().ember);
        assert!(hot.modifier.contains(Modifier::BOLD));

        let mode_tag = cell_at(HeatState::Hot, AppMode::ModalModel, 3);
        assert_eq!(mode_tag.symbol(), "/");
        assert_eq!(mode_tag.fg, theme.heat().ember);
    }

    #[test]
    fn kill_ring_yanks_the_most_recent_kill() {
        let mut input = InputWidget::new();
        type_text(&mut input, "alpha beta");
        input.delete_word_backward(); // kills "beta"
        assert_eq!(input.text(), "alpha ");
        assert!(input.yank(), "a kill must be yankable");
        assert_eq!(input.text(), "alpha beta");

        // Ctrl-K tail kill becomes the newer entry and wins the next yank.
        input.move_to_line_start();
        input.kill_to_line_end(); // kills "alpha beta"
        assert_eq!(input.text(), "");
        assert!(input.yank());
        assert_eq!(input.text(), "alpha beta");
    }

    #[test]
    fn kill_whole_line_is_yankable_and_empty_kills_are_ignored() {
        let mut input = InputWidget::new();
        assert!(!input.yank(), "nothing killed yet");
        type_text(&mut input, "keep me");
        input.kill_whole_line();
        assert_eq!(input.text(), "");
        // A Ctrl-K at end of an empty line kills nothing and must not
        // clobber the last real kill.
        input.kill_to_line_end();
        assert!(input.yank());
        assert_eq!(input.text(), "keep me");
    }

    #[test]
    fn yank_is_undoable() {
        let mut input = InputWidget::new();
        type_text(&mut input, "word");
        input.delete_word_backward();
        assert_eq!(input.text(), "");
        assert!(input.yank());
        assert_eq!(input.text(), "word");
        assert!(input.undo());
        assert_eq!(input.text(), "", "yank must undo as one chunk");
    }

    #[test]
    fn wrap_segments_splits_ascii_at_width() {
        let segs = wrap_segments("abcdefghij", 4, 4);
        let texts: Vec<&str> = segs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["abcd", "efgh", "ij"]);
        // start-char offsets line up with the wrap points.
        assert_eq!(
            segs.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            vec![0, 4, 8]
        );
    }

    #[test]
    fn wrap_segments_counts_cjk_as_two_cells() {
        // Each Hangul syllable is 2 cells wide → only 2 fit per 4-cell row.
        let segs = wrap_segments("가나다라", 4, 4);
        let texts: Vec<&str> = segs.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(texts, vec!["가나", "다라"]);
    }

    #[test]
    fn wrap_segments_empty_line_yields_one_empty_segment() {
        assert_eq!(wrap_segments("", 10, 10), vec![(0, String::new())]);
    }

    #[test]
    fn desired_rows_grows_with_soft_wrapped_long_line() {
        let mut input = InputWidget::new();
        for ch in "abcdefghijklmnopqrstuvwxyz".chars() {
            input.insert_char(ch);
        }
        // Inner width 12 → text width after the 2-cell prompt gutter is 10,
        // so 26 chars wrap onto 3 display rows → 3 + 2 border = 5.
        let rows = input.desired_rows(12, AppMode::Normal);
        assert_eq!(rows, 5);
    }

    #[test]
    fn desired_rows_clamps_to_max() {
        let mut input = InputWidget::new();
        for ch in std::iter::repeat_n('x', 500) {
            input.insert_char(ch);
        }
        let rows = input.desired_rows(8, AppMode::Normal);
        assert_eq!(rows, super::INPUT_MAX_ROWS);
    }

    #[test]
    fn strips_leaked_sgr_mouse_sequences() {
        assert_eq!(
            strip_sgr_mouse_sequences_from_text("hi[<35;24;26M[<35;36;25m there"),
            Some("hi there".to_string())
        );
        assert_eq!(strip_sgr_mouse_sequences_from_text("[<35;24;26"), None);
    }

    #[test]
    fn input_card_uses_left_rail_not_full_box() {
        use crate::tui::theme::Theme;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::layout::Rect;

        let theme = Theme::default_dark();
        let mut input = InputWidget::new();
        for c in "hello".chars() {
            input.insert_char(c);
        }
        let backend = TestBackend::new(30, 4);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| input.draw(f, Rect::new(0, 0, 30, 4), &theme, &AppMode::Normal))
            .expect("draw");
        let buf = term.backend().buffer().clone();
        let mut dump = String::new();
        for y in 0..4 {
            for x in 0..30 {
                dump.push_str(buf[(x, y)].symbol());
            }
            dump.push('\n');
        }
        // Input focus cue: match the chat rail with only a heavy left line.
        assert!(
            dump.contains('\u{2503}'),
            "input must use ┃ left rail:\n{dump}"
        );
        // 라운드 풀박스의 상/하 가로 rule(─)·코너(╭╰)는 더 이상 없다.
        assert!(
            !dump.contains('\u{2500}'),
            "no top/bottom box rule ─:\n{dump}"
        );
        assert!(!dump.contains('\u{256d}'), "no rounded corner ╭:\n{dump}");
        assert!(!dump.contains('\u{2570}'), "no rounded corner ╰:\n{dump}");
        assert!(dump.contains("hello"), "input text still rendered:\n{dump}");
    }

    #[test]
    fn large_single_line_paste_collapses_and_roundtrips() {
        // A single-line clipboard payload (image/base64/binary) with no
        // newlines: gated only on line count the old path did O(N²) char
        // inserts and froze. It must collapse on the char threshold instead.
        let big = "a".repeat(100_000);
        let mut input = InputWidget::new();
        input.insert_text(&big);

        // Collapsed into exactly one hidden paste chip.
        assert_eq!(input.paste_blocks.len(), 1);

        // Submit expansion round-trips the full payload verbatim.
        assert!(input.text().contains(&big));
        assert_eq!(input.text(), big);
    }

    #[test]
    fn owned_large_paste_reuses_source_allocation() {
        let big = "x".repeat(4 * 1024 * 1024);
        let source_ptr = big.as_ptr();
        let source_len = big.len();
        let mut input = InputWidget::new();

        input.insert_text_owned(big);

        assert_eq!(input.paste_blocks.len(), 1);
        assert_eq!(input.paste_blocks[0].body.as_str().as_ptr(), source_ptr);
        assert_eq!(input.paste_blocks[0].body.len(), source_len);
        assert_eq!(input.lines[0].chars().count(), 1);
        assert!(input.display_lines[0].len() < 80);
    }

    #[test]
    fn content_revision_detects_edits_without_tracking_cursor_motion() {
        let mut input = InputWidget::new();
        let initial = input.content_revision();

        input.move_left();
        assert_eq!(input.content_revision(), initial);
        input.insert_char('x');
        assert_ne!(input.content_revision(), initial);
    }

    #[test]
    fn short_single_line_paste_stays_inline() {
        // Well under COLLAPSE_CHAR_THRESHOLD (2000) and one line: normal
        // input must stay editable inline, never collapse into a chip.
        let small = "a".repeat(50);
        let mut input = InputWidget::new();
        input.insert_text(&small);

        assert!(input.paste_blocks.is_empty());
        assert_eq!(input.text(), small);
    }
}
