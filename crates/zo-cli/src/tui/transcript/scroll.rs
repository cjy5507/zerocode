//! Scroll, focus, and search navigation over the cached transcript layout.
//!
//! Owns viewport position, the focused-block cursor, the expanded-block
//! set, and block text search. Heights and y-offsets come from the layout
//! cache maintained in [`super::layout`]; this module never re-measures.

#![allow(clippy::doc_markdown)]

use ratatui::layout::Rect;
use runtime::message_stream::{BlockId, RenderBlock, ToolCallStatus};

use crate::tui::blocks::is_interactable;
use crate::tui::image_protocol::ImageProtocol;
use crate::tui::theme::Theme;

use super::layout::block_id;
use super::tool_groups::ToolGroupState;
use super::{COPY_BUTTON_WIDTH, CopyButtonHit};
use super::{CharSelection, DEFAULT_BLOCK_ROWS, ScrollPos, Transcript};

impl Transcript {
    /// Current resolved scroll offset in rows. Between a `scroll_to_bottom` and
    /// the next draw this is the `u16::MAX` tail sentinel; after any draw it is
    /// the real, content-clamped offset.
    #[must_use]
    pub fn scroll(&self) -> u16 {
        self.resolved_scroll
    }

    /// Move to an explicit row offset: record the intent as [`ScrollPos::Rows`]
    /// and mirror it into the resolved cache so reads before the next draw stay
    /// consistent. `draw` re-clamps the resolved value against the live content
    /// height, so callers may pass an un-clamped target (e.g. a saturating add).
    pub(super) fn set_explicit_scroll(&mut self, rows: u16) {
        self.scroll = ScrollPos::Rows(rows);
        self.resolved_scroll = rows;
    }

    /// Current focused block index (within [`Self::blocks`]), if any.
    #[must_use]
    pub fn focused_idx(&self) -> Option<usize> {
        self.focused_idx
    }

    /// Scroll down by `rows` rows (saturating). Moves from where the tail
    /// currently sits and drops follow-tail (becomes an explicit offset).
    pub fn scroll_down(&mut self, rows: u16) {
        self.set_explicit_scroll(self.resolved_scroll.saturating_add(rows));
    }

    /// Scroll up by `rows` rows (saturating). Moves from where the tail
    /// currently sits and drops follow-tail (becomes an explicit offset).
    pub fn scroll_up(&mut self, rows: u16) {
        self.set_explicit_scroll(self.resolved_scroll.saturating_sub(rows));
    }

    /// Snap scroll to the bottom (follow the tail).
    pub fn scroll_to_bottom(&mut self) {
        self.scroll = ScrollPos::Bottom;
        // Transient sentinel until the next draw resolves it to the real max;
        // keeps `scroll()` reporting the tail for callers reading before a draw.
        self.resolved_scroll = u16::MAX;
    }

    /// Total content height including the bottom breathing-room pad (2 rows
    /// between the last block and the input box). Single source of truth so
    /// `draw`, `clamp_scroll_to_content`, and `is_at_bottom` agree on the max
    /// scroll offset — a mismatch made the view jump on the first manual
    /// scroll and trapped auto-follow (couldn't leave the tail).
    pub(super) fn content_total(&self) -> u16 {
        let base = self
            .cached_layout
            .last()
            .map_or(0, |r| r.1.saturating_add(r.2));
        if self.cached_layout.is_empty() {
            base
        } else {
            base.saturating_add(2)
        }
    }

    /// Clamp the `u16::MAX` "tail" sentinel (set by
    /// [`Self::scroll_to_bottom`]) to the actual maximum scroll offset
    /// for the given viewport. Must be called before any user-initiated
    /// `scroll_up` so that the first upward step leaves the tail.
    pub fn clamp_scroll_to_content(
        &mut self,
        viewport_h: u16,
        theme: &Theme,
        width: u16,
        image_protocol: ImageProtocol,
    ) {
        if width == 0 {
            return;
        }
        // Reuse the width the last draw laid out with. The draw pass narrows
        // the content column by 1 for the scrollbar gutter, so the caller's
        // full region width usually differs from `cached_layout_width`;
        // passing it through forced a full O(n) re-layout here AND a second
        // one in the next draw — and because every per-block render cache is
        // width-keyed, each scroll event invalidated *every* cache entry
        // (markdown + syntect re-render of the whole transcript). That width
        // thrash was the "scrolling lags on long transcripts" bug. Scroll
        // clamping only needs the existing layout's totals; `draw` re-clamps
        // against the authoritative layout anyway.
        let layout_width = self.layout_width_or(width);
        self.ensure_layout(layout_width, theme, image_protocol);
        let content_total = self.content_total();
        let max_scroll = content_total.saturating_sub(viewport_h);
        // Materialize the intent into a real offset so a following `scroll_up`
        // moves from where the tail actually sits: `Bottom` (and any stale
        // over-scroll) becomes an explicit `Rows(max_scroll)`; an in-range
        // explicit offset is left where it is.
        let resolved = match self.scroll {
            ScrollPos::Bottom => max_scroll,
            ScrollPos::Rows(offset) => offset.min(max_scroll),
        };
        self.set_explicit_scroll(resolved);
    }

    /// Width to lay out with for scroll bookkeeping: the cached layout's
    /// width when one exists (matching what the last draw used, scrollbar
    /// gutter included), else the caller-provided region width.
    fn layout_width_or(&self, width: u16) -> u16 {
        if self.cached_layout.is_empty() || self.cached_layout_width == 0 {
            width
        } else {
            self.cached_layout_width
        }
    }

    /// Snap scroll to the top.
    pub fn scroll_to_top(&mut self) {
        self.set_explicit_scroll(0);
    }

    /// Return the plain-text payload of the visible block under a transcript
    /// viewport row. Used by the always-on click/drag copy interaction.
    pub fn copy_text_at_viewport_row(
        &mut self,
        row_in_viewport: u16,
        theme: &Theme,
        width: u16,
        image_protocol: ImageProtocol,
    ) -> Option<String> {
        if width == 0 {
            return None;
        }
        let layout_width = self.layout_width_or(width);
        self.ensure_layout(layout_width, theme, image_protocol);
        let content_row = self.resolved_scroll.saturating_add(row_in_viewport);
        let (idx, _, _) = self
            .cached_layout
            .iter()
            .copied()
            .find(|(idx, top, height)| {
                content_row >= *top
                    && content_row < top.saturating_add(*height)
                    && !matches!(self.tool_groups.get(*idx), Some(ToolGroupState::Hidden))
            })?;
        let text = block_copy_text_content(self.blocks.get(idx)?);
        (!text.trim().is_empty()).then_some(text)
    }

    /// Return the copy affordance for the visible block under a transcript
    /// viewport row. This deliberately avoids materializing the block text;
    /// hover can fire very often, and the actual copy payload is only needed
    /// after the user clicks the small button.
    pub(crate) fn copy_affordance_at_viewport_row(
        &mut self,
        row_in_viewport: u16,
        area: Rect,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) -> Option<CopyButtonHit> {
        if area.width <= COPY_BUTTON_WIDTH || area.height == 0 {
            return None;
        }
        self.copy_hit_for_content_row(
            self.resolved_scroll.saturating_add(row_in_viewport),
            area,
            theme,
            image_protocol,
        )
    }

    /// Return a block's copy payload by stable block id. Called only after the
    /// user clicks the hover button, so building the owned string here is fine.
    pub fn copy_text_for_block_id(&self, target: BlockId) -> Option<String> {
        let text = self
            .blocks
            .iter()
            .find(|block| block_id(block) == target)
            .map(block_copy_text_content)?;
        (!text.trim().is_empty()).then_some(text)
    }

    /// Anchor a char-selection at a viewport cell (left-button press on the
    /// transcript). The row is stored as a content row (viewport row + the
    /// current resolved scroll), so the gesture survives any scroll — a wheel
    /// notch mid-drag extends the selection instead of invalidating it. No
    /// highlight yet — `dragged` flips only once the head leaves the anchor
    /// cell — so a plain click never flashes a selection.
    pub fn begin_char_selection(&mut self, col: u16, row_in_viewport: u16) {
        let row = self.resolved_scroll.saturating_add(row_in_viewport);
        self.char_selection = Some(CharSelection {
            anchor: (col, row),
            head: (col, row),
            dragged: false,
        });
        self.char_selection_mined.clear();
    }

    /// Move the in-progress char-selection head to the viewport cell under
    /// the pointer (converted to a content row against the current scroll),
    /// marking the gesture dragged once it leaves the anchor cell. Returns
    /// whether the highlight changed (so the caller repaints only on real
    /// movement — and the repaint is what mines the selection text).
    pub fn extend_char_selection(&mut self, col: u16, row_in_viewport: u16) -> bool {
        let row = self.resolved_scroll.saturating_add(row_in_viewport);
        let Some(selection) = self.char_selection.as_mut() else {
            return false;
        };
        let head = (col, row);
        let newly_dragged = !selection.dragged && head != selection.anchor;
        let moved = selection.head != head;
        if newly_dragged {
            selection.dragged = true;
        }
        selection.head = head;
        newly_dragged || (selection.dragged && moved)
    }

    /// End the gesture. A dragged selection joins its mined rows over the
    /// selected content-row range (rows scrolled offscreen kept their last
    /// mined text, so a wheel-extended drag copies past one screenful) and
    /// keeps its highlight until the next press. A plain click, or a drag
    /// whose selected cells were blank, clears the selection and yields
    /// nothing.
    pub fn finish_char_selection(&mut self) -> Option<String> {
        let selection = self.char_selection?;
        if !selection.dragged {
            self.clear_char_selection();
            return None;
        }
        let (start_row, end_row) = {
            let (a, b) = (selection.anchor.1, selection.head.1);
            (a.min(b), a.max(b))
        };
        let lines: Vec<String> = (start_row..=end_row)
            .map(|row| {
                self.char_selection_mined
                    .get(&row)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        let text = super::join_selection_lines(&lines);
        if text.is_empty() {
            self.clear_char_selection();
            return None;
        }
        Some(text)
    }

    /// Drop any char-selection highlight or in-progress gesture. Returns
    /// whether a highlight was actually showing, so the caller repaints only
    /// when the screen changed.
    pub fn clear_char_selection(&mut self) -> bool {
        let had_highlight = self.has_char_selection();
        self.char_selection = None;
        self.char_selection_mined.clear();
        had_highlight
    }

    /// Whether a dragged char-selection highlight is currently on screen.
    #[must_use]
    pub fn has_char_selection(&self) -> bool {
        self.char_selection.is_some_and(|selection| selection.dragged)
    }

    /// Whether a char-selection gesture exists at all (pressed, dragged, or
    /// kept highlighted after release). The wheel handler uses this to route
    /// a notch into drag-extension instead of clearing the gesture.
    #[must_use]
    pub fn char_selection_active(&self) -> bool {
        self.char_selection.is_some()
    }

    /// Drop the char-selection when the pending layout pass would shift the
    /// rows it is anchored to. Content-row anchors survive scrolling and tail
    /// appends (earlier rows keep their offsets) but not a re-wrap (width
    /// change) or an in-place mutation at/above the selection — the dirtied
    /// block's height may change and shove every row below it, so the wash
    /// and the mined copy would land on shifted text. Called by
    /// `ensure_layout` before it consumes the dirty marks, while
    /// `cached_layout` still holds the offsets the selection was anchored
    /// against.
    pub(super) fn drop_char_selection_on_layout_shift(&mut self, width: u16) {
        let Some(selection) = self.char_selection else {
            return;
        };
        if self.cached_layout_width != 0 && self.cached_layout_width != width {
            self.clear_char_selection();
            return;
        }
        let Some(dirty_from) = self.layout_dirty_from else {
            return;
        };
        // A brand-new tail block has no layout entry yet (`get` misses) and
        // never moves earlier rows; only a dirty block that *starts* at or
        // above the selection's last row can shift it.
        let sel_last_row = selection.anchor.1.max(selection.head.1);
        let pos = self
            .cached_layout
            .partition_point(|&(idx, _, _)| idx < dirty_from);
        let dirty_top = self.cached_layout.get(pos).map(|&(_, top, _)| top);
        if dirty_top.is_some_and(|top| top <= sel_last_row) {
            self.clear_char_selection();
        }
    }

    /// Stable id of the visible block under a transcript viewport row, or `None`
    /// over an empty row or a hidden tool-group member. Mirrors
    /// [`Self::copy_text_at_viewport_row`]'s lookup but returns the id so a
    /// transcript press can preserve block-level click behavior on release.
    pub fn block_id_at_viewport_row(
        &mut self,
        row_in_viewport: u16,
        theme: &Theme,
        width: u16,
        image_protocol: ImageProtocol,
    ) -> Option<BlockId> {
        if width == 0 {
            return None;
        }
        let layout_width = self.layout_width_or(width);
        self.ensure_layout(layout_width, theme, image_protocol);
        let content_row = self.resolved_scroll.saturating_add(row_in_viewport);
        let (idx, _, _) = self
            .cached_layout
            .iter()
            .copied()
            .find(|(idx, top, height)| {
                content_row >= *top
                    && content_row < top.saturating_add(*height)
                    && !matches!(self.tool_groups.get(*idx), Some(ToolGroupState::Hidden))
            })?;
        Some(block_id(self.blocks.get(idx)?))
    }

    /// Concatenated plain text of the blocks spanned by the inclusive id range
    /// `from..=to`, in display order regardless of which id the drag began from.
    /// Hidden tool-group members and empty blocks are skipped; the rest join
    /// with a blank line so the copy reads like the transcript without the
    /// gutter chrome (`│ ◆ ⎿`). `None` when the range yields no copyable text.
    #[must_use]
    pub fn copy_text_for_block_range(&self, from: BlockId, to: BlockId) -> Option<String> {
        let ia = self.blocks.iter().position(|blk| block_id(blk) == from)?;
        let ib = self.blocks.iter().position(|blk| block_id(blk) == to)?;
        let (lo, hi) = (ia.min(ib), ia.max(ib));
        let mut sections = Vec::new();
        for idx in lo..=hi {
            if matches!(self.tool_groups.get(idx), Some(ToolGroupState::Hidden)) {
                continue;
            }
            let Some(block) = self.blocks.get(idx) else {
                continue;
            };
            let text = block_copy_text_content(block);
            if !text.trim().is_empty() {
                sections.push(text);
            }
        }
        (!sections.is_empty()).then(|| sections.join("\n\n"))
    }

    pub(crate) fn copy_button_for_block(
        &mut self,
        target: BlockId,
        area: Rect,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) -> Option<CopyButtonHit> {
        if area.width <= COPY_BUTTON_WIDTH || area.height == 0 {
            return None;
        }
        let layout_width = self.layout_width_or(area.width);
        self.ensure_layout(layout_width, theme, image_protocol);
        let (_, top, _) = self.cached_layout.iter().copied().find(|(idx, _, _)| {
            self.blocks
                .get(*idx)
                .is_some_and(|block| block_id(block) == target)
        })?;
        self.copy_hit_for_content_row(top.max(self.resolved_scroll), area, theme, image_protocol)
            .filter(|hit| hit.block_id == target)
    }

    /// Set the scroll offset from a scrollbar click/drag at `row_in_viewport`
    /// (0 = the transcript's top row). The drawn scrollbar puts ▲ on the first
    /// row and ▼ on the last, so the thumb travels the rows between; map a click
    /// within that travel proportionally onto `[0, max_scroll]`. Uses the same
    /// content height (`content_total` minus the viewport) as the wheel/clamp
    /// path so the thumb and the text never disagree.
    pub fn scroll_to_viewport_row(
        &mut self,
        row_in_viewport: u16,
        viewport_h: u16,
        theme: &Theme,
        width: u16,
        image_protocol: ImageProtocol,
    ) {
        if width == 0 || viewport_h == 0 {
            return;
        }
        let layout_width = self.layout_width_or(width);
        self.ensure_layout(layout_width, theme, image_protocol);
        let max_scroll = self.content_total().saturating_sub(viewport_h);
        if max_scroll == 0 {
            return;
        }
        let track = viewport_h.saturating_sub(2).max(1);
        let pos = row_in_viewport.saturating_sub(1).min(track);
        let scroll = u32::from(pos) * u32::from(max_scroll) / u32::from(track);
        self.set_explicit_scroll(u16::try_from(scroll).unwrap_or(max_scroll).min(max_scroll));
    }

    /// `true` if the viewport is currently showing the tail of the content.
    #[must_use]
    pub fn is_at_bottom(
        &self,
        viewport_h: u16,
        _theme: &Theme,
        width: u16,
        _image_protocol: ImageProtocol,
    ) -> bool {
        if self.blocks.is_empty() {
            return true;
        }
        // Use cached layout when valid to avoid O(n) recalculation. The draw
        // pass may have laid out at `width` or `width - 1` (scrollbar gutter);
        // both describe the same content, so accept either rather than
        // falling through to the stale "assume at bottom" default.
        if !self.cached_layout.is_empty()
            && (self.cached_layout_width == width
                || self.cached_layout_width == width.saturating_sub(1))
        {
            let content_total = self.content_total();
            return self.resolved_scroll >= content_total.saturating_sub(viewport_h);
        }
        // No valid cache — assume at bottom (safe default for auto-scroll).
        true
    }

    /// Move focus to the next interactable block (forward).
    ///
    /// Wraps at the end. Returns `true` if focus moved.
    pub fn focus_next(&mut self) -> bool {
        let start = self
            .focused_idx
            .map_or(0, |i| i.saturating_add(1).min(self.blocks.len()));
        for (idx, block) in self.blocks.iter().enumerate().skip(start) {
            if is_interactable(block) {
                self.focused_idx = Some(idx);
                return true;
            }
        }
        // Wrap from the start.
        for (idx, block) in self.blocks.iter().enumerate().take(start) {
            if is_interactable(block) {
                self.focused_idx = Some(idx);
                return true;
            }
        }
        false
    }

    /// Move focus to the previous interactable block (backward).
    pub fn focus_prev(&mut self) -> bool {
        let end = self.focused_idx.unwrap_or(self.blocks.len());
        for idx in (0..end).rev() {
            if is_interactable(&self.blocks[idx]) {
                self.focused_idx = Some(idx);
                return true;
            }
        }
        for idx in (end..self.blocks.len()).rev() {
            if is_interactable(&self.blocks[idx]) {
                self.focused_idx = Some(idx);
                return true;
            }
        }
        false
    }

    /// Toggle the expanded state of the currently focused block.
    ///
    /// Returns `true` if a block was toggled.
    pub fn toggle_expanded(&mut self) -> bool {
        let Some(idx) = self.focused_idx else {
            return false;
        };
        let Some(block) = self.blocks.get(idx) else {
            return false;
        };
        let id = block_id(block);
        if !self.expanded.insert(id.0) {
            self.expanded.remove(&id.0);
        }
        self.invalidate_layout_cache();
        true
    }

    /// Route a plain mouse click on block `id` to an expand/collapse action
    /// (Claude-Code parity: clicking a collapsed row opens it in place).
    ///
    /// * Collapsed tool-group `Summary` leader → reveal the group's individual
    ///   rows; clicking the (revealed) leader again re-collapses it.
    /// * Plain `ToolCall` row → toggle its matching `ToolResult`'s expansion
    ///   (the clipped `▸ Result … expand` body). No result yet → not consumed.
    /// * `ToolResult` / `Reasoning` / `AgentResult` → toggle own expansion.
    ///
    /// Returns `true` when the click was consumed (layout invalidated); `false`
    /// leaves non-expandable prose unchanged.
    pub fn toggle_expand_for_click(&mut self, id: BlockId) -> bool {
        enum ClickExpand {
            Group,
            OwnBlock(u64),
        }
        let Some(idx) = self
            .blocks
            .iter()
            .position(|block| block_id(block) == id)
        else {
            return false;
        };
        let target = match &self.blocks[idx] {
            RenderBlock::ToolCall {
                id,
                tool_call_id,
                name,
                status,
                ..
            } => {
                if matches!(
                    self.tool_groups.get(idx),
                    Some(ToolGroupState::Summary { .. })
                ) || self.revealed_groups.contains(&id.0)
                {
                    ClickExpand::Group
                } else if *status == ToolCallStatus::Running
                    && crate::tui::blocks::tool_call::is_bash(name)
                {
                    ClickExpand::OwnBlock(id.0)
                } else {
                    // Clicking the call row answers with its result body — the
                    // pair reads as one card, and the call row is the natural
                    // click target (CC behavior).
                    let call_id = tool_call_id;
                    match self.blocks.iter().find_map(|block| match block {
                        RenderBlock::ToolResult {
                            id, tool_call_id, ..
                        } if tool_call_id == call_id => Some(id.0),
                        _ => None,
                    }) {
                        Some(result_id) => ClickExpand::OwnBlock(result_id),
                        None => return false,
                    }
                }
            }
            RenderBlock::ToolResult { .. }
            | RenderBlock::Reasoning { .. }
            | RenderBlock::AgentResult { .. } => ClickExpand::OwnBlock(id.0),
            _ => return false,
        };
        match target {
            ClickExpand::Group => {
                if !self.revealed_groups.insert(id.0) {
                    self.revealed_groups.remove(&id.0);
                }
            }
            ClickExpand::OwnBlock(raw_id) => {
                if !self.expanded.insert(raw_id) {
                    self.expanded.remove(&raw_id);
                }
            }
        }
        self.invalidate_layout_cache();
        true
    }

    /// Drop any block focus, returning the transcript to its default
    /// (composer-driven) key semantics. Returns `true` if a block had been
    /// focused, so the caller can decide whether the key was consumed.
    pub fn clear_focus(&mut self) -> bool {
        if self.focused_idx.is_some() {
            self.focused_idx = None;
            true
        } else {
            false
        }
    }

    /// `true` if the block at index `idx` is currently expanded.
    #[must_use]
    pub fn is_expanded(&self, idx: usize) -> bool {
        self.blocks
            .get(idx)
            .is_some_and(|block| self.expanded.contains(&block_id(block).0))
    }

    /// Find the first block whose text content contains `query`
    /// (case-insensitive, caller should pass a lowercased query).
    /// Returns the block index if found.
    #[must_use]
    pub fn find_block_containing(&self, query: &str) -> Option<usize> {
        for (idx, block) in self.blocks.iter().enumerate() {
            let text = block_text_content(block);
            if text.to_lowercase().contains(query) {
                return Some(idx);
            }
        }
        None
    }

    /// Find every block whose text content contains `query`
    /// (case-insensitive; caller passes a lowercased query), in
    /// document order. Backs incremental search with n/N navigation.
    #[must_use]
    pub fn find_all_blocks_containing(&self, query: &str) -> Vec<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block_text_content(block).to_lowercase().contains(query))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Set (or clear) the block index drawn with the search-match accent
    /// marker. Driven by the app's search state.
    pub fn set_search_highlight(&mut self, idx: Option<usize>) {
        self.search_highlight = idx;
    }

    /// Scroll the viewport so that block at `idx` is visible near the
    /// top. Uses the cached layout when available for accuracy.
    pub fn scroll_to_block(&mut self, idx: usize) {
        if idx >= self.blocks.len() {
            return;
        }
        // Use cached layout for an accurate position when available.
        if !self.cached_layout.is_empty() {
            for &(block_idx, top, _height) in &self.cached_layout {
                if block_idx == idx {
                    self.set_explicit_scroll(top);
                    return;
                }
            }
        }
        // Fallback: approximate position when no cache is available.
        // Estimate using the per-block reserved height plus one row for
        // the inter-block separator the layout inserts between blocks.
        let est_rows = u32::from(DEFAULT_BLOCK_ROWS) + 1;
        let approx = u32::try_from(idx)
            .unwrap_or(u32::MAX)
            .saturating_mul(est_rows);
        self.set_explicit_scroll(u16::try_from(approx).unwrap_or(u16::MAX));
    }

    fn copy_hit_for_content_row(
        &mut self,
        content_row: u16,
        area: Rect,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) -> Option<CopyButtonHit> {
        let layout_width = self.layout_width_or(area.width);
        if layout_width <= COPY_BUTTON_WIDTH {
            return None;
        }
        self.ensure_layout(layout_width, theme, image_protocol);
        let (idx, top, _) = self
            .cached_layout
            .iter()
            .copied()
            .find(|(idx, top, height)| {
                content_row >= *top
                    && content_row < top.saturating_add(*height)
                    && !matches!(self.tool_groups.get(*idx), Some(ToolGroupState::Hidden))
            })?;
        let block = self.blocks.get(idx)?;
        if !block_has_copyable_text(block) {
            return None;
        }

        let draw_y = area.y.saturating_add(top.saturating_sub(self.resolved_scroll));
        if draw_y >= area.y.saturating_add(area.height) {
            return None;
        }
        let button = Rect::new(
            area.x + layout_width.saturating_sub(COPY_BUTTON_WIDTH),
            draw_y,
            COPY_BUTTON_WIDTH,
            1,
        );
        Some(CopyButtonHit {
            block_id: block_id(block),
            button,
        })
    }
}

fn block_has_copyable_text(block: &RenderBlock) -> bool {
    block_copy_payload(block).has_text()
}

enum BlockCopyPayload<'a> {
    Empty,
    Borrowed(&'a str),
    Bash(&'a runtime::message_stream::BashResult),
    Diff(&'a runtime::message_stream::DiffView),
    Listing(&'a [String]),
    Todos(&'a [runtime::message_stream::TodoResultItem]),
    Owned(String),
}

impl BlockCopyPayload<'_> {
    fn has_text(&self) -> bool {
        match self {
            Self::Empty => false,
            Self::Borrowed(text) => !text.trim().is_empty(),
            Self::Owned(text) => !text.trim().is_empty(),
            Self::Bash(result) => [result.stdout.as_str(), result.stderr.as_str()]
                .into_iter()
                .any(|part| !part.trim().is_empty()),
            Self::Diff(view) => {
                view.old_path
                    .as_deref()
                    .is_some_and(|path| !path.trim().is_empty())
                    || view
                        .new_path
                        .as_deref()
                        .is_some_and(|path| !path.trim().is_empty())
                    || view.hunks.iter().any(|hunk| !hunk.lines.is_empty())
            }
            Self::Listing(entries) => entries.iter().any(|entry| !entry.trim().is_empty()),
            Self::Todos(items) => items.iter().any(|item| !todo_copy_text(item).trim().is_empty()),
        }
    }

    fn into_text(self) -> String {
        match self {
            Self::Empty => String::new(),
            Self::Borrowed(text) => text.to_string(),
            Self::Owned(text) => text,
            Self::Bash(result) => [result.stdout.as_str(), result.stderr.as_str()]
                .into_iter()
                .filter(|part| !part.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            Self::Diff(view) => unified_diff_text(view),
            Self::Listing(entries) => entries.join("\n"),
            Self::Todos(items) => items
                .iter()
                .map(|item| {
                    let mark = todo_status_mark(item.status);
                    let text = todo_copy_text(item);
                    format!("{mark} {text}")
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

fn todo_status_mark(status: runtime::message_stream::TodoResultStatus) -> &'static str {
    use runtime::message_stream::TodoResultStatus;
    match status {
        TodoResultStatus::Pending => "[ ]",
        TodoResultStatus::InProgress => "[~]",
        TodoResultStatus::Completed => "[x]",
    }
}

fn todo_copy_text(item: &runtime::message_stream::TodoResultItem) -> &str {
    if item.status == runtime::message_stream::TodoResultStatus::InProgress
        && !item.active_form.trim().is_empty()
    {
        item.active_form.as_str()
    } else {
        item.content.as_str()
    }
}

fn block_copy_payload(block: &RenderBlock) -> BlockCopyPayload<'_> {
    match block {
        RenderBlock::TextDelta { text, .. }
        | RenderBlock::Reasoning { text, .. }
        | RenderBlock::UserMessage { text, .. }
        | RenderBlock::System { text, .. } => BlockCopyPayload::Borrowed(text),
        RenderBlock::AgentResult { body, .. } => BlockCopyPayload::Borrowed(body),
        // A pushed `send_to_user` notice is verbatim content, so its text must
        // be copyable out of the transcript like any other prose.
        RenderBlock::UserNotice { message, .. } => BlockCopyPayload::Borrowed(message),
        RenderBlock::ToolCall { .. }
        | RenderBlock::Image { .. }
        | RenderBlock::PermissionPrompt(_)
        | RenderBlock::UserQuestionPrompt(_)
        | RenderBlock::Usage { .. }
        | RenderBlock::CompactionProgress { .. }
        | RenderBlock::RateLimit(_) => BlockCopyPayload::Empty,
        RenderBlock::ToolResult { body, .. } => {
            use runtime::message_stream::ToolResultBody;
            match body {
                ToolResultBody::Text { content, .. } | ToolResultBody::Generic { content, .. } => {
                    BlockCopyPayload::Borrowed(content)
                }
                ToolResultBody::Bash(result) => BlockCopyPayload::Bash(result),
                ToolResultBody::Read { content, .. } => BlockCopyPayload::Borrowed(content),
                ToolResultBody::Diff(view) => BlockCopyPayload::Diff(view),
                ToolResultBody::Listing { entries, .. } => BlockCopyPayload::Listing(entries),
                ToolResultBody::Todos(items) => BlockCopyPayload::Todos(items),
            }
        }
        RenderBlock::Card { card, .. } => BlockCopyPayload::Owned(card.plain_text()),
    }
}

/// Serialize a structured [`DiffView`] back into standard unified-diff text
/// (`--- a/… / +++ b/… / @@ … @@` plus `+`/`-`/space-prefixed body lines), so
/// copying a diff tool-result block yields a patch the user can re-apply rather
/// than just the file path. New files map the old side to `/dev/null`, deletions
/// map the new side to `/dev/null`.
fn unified_diff_text(view: &runtime::message_stream::DiffView) -> String {
    use std::fmt::Write as _;

    use runtime::message_stream::DiffLineKind;

    let old_header = view
        .old_path
        .as_deref()
        .map_or_else(|| "/dev/null".to_string(), |path| format!("a/{path}"));
    let new_header = view
        .new_path
        .as_deref()
        .map_or_else(|| "/dev/null".to_string(), |path| format!("b/{path}"));

    let mut out = format!("--- {old_header}\n+++ {new_header}\n");
    for hunk in &view.hunks {
        let _ = writeln!(
            out,
            "@@ -{},{} +{},{} @@",
            hunk.old_start, hunk.old_lines, hunk.new_start, hunk.new_lines
        );
        for line in &hunk.lines {
            let marker = match line.kind {
                DiffLineKind::Added => '+',
                DiffLineKind::Removed => '-',
                DiffLineKind::Context => ' ',
            };
            out.push(marker);
            out.push_str(&line.text);
            out.push('\n');
        }
    }
    out
}

/// Extract the clean user-facing payload copied from a render block. Unlike
/// [`block_text_content`], this excludes copy-hostile UI/meta cards such as
/// tool-call status/input summaries and image placeholders while keeping real
/// prose/result bodies.
fn block_copy_text_content(block: &RenderBlock) -> String {
    block_copy_payload(block).into_text()
}


/// Extract a searchable text representation from a render block.
fn block_text_content(block: &RenderBlock) -> String {
    match block {
        RenderBlock::TextDelta { text, .. }
        | RenderBlock::Reasoning { text, .. }
        | RenderBlock::UserMessage { text, .. }
        | RenderBlock::System { text, .. } => text.clone(),
        RenderBlock::AgentResult { label, body, .. } => format!("{label}\n{body}"),
        RenderBlock::UserNotice { message, .. } => message.clone(),
        RenderBlock::ToolCall { name, summary, .. } => format!("{name} {summary}"),
        RenderBlock::ToolResult { body, .. } => {
            use runtime::message_stream::ToolResultBody;
            match body {
                ToolResultBody::Text { content, .. } | ToolResultBody::Generic { content, .. } => {
                    content.clone()
                }
                ToolResultBody::Bash(b) => format!("{}\n{}", b.stdout, b.stderr),
                ToolResultBody::Read { path, content, .. } => format!("{path}\n{content}"),
                ToolResultBody::Diff(view) => unified_diff_text(view),
                ToolResultBody::Listing { entries, .. } => entries.join("\n"),
                ToolResultBody::Todos(items) => items
                    .iter()
                    .map(|item| {
                        use runtime::message_stream::TodoResultStatus;
                        let mark = match item.status {
                            TodoResultStatus::Pending => "[ ]",
                            TodoResultStatus::InProgress => "[~]",
                            TodoResultStatus::Completed => "[x]",
                        };
                        format!("{mark} {}", item.content)
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            }
        }
        RenderBlock::PermissionPrompt(p) => p.audit_hint.as_ref().map_or_else(
            || p.tool_name.clone(),
            |audit_hint| format!("{}\n{}\n{}", p.tool_name, p.reasoning, audit_hint),
        ),
        RenderBlock::UserQuestionPrompt(p) => p.question.clone(),
        RenderBlock::Image { media_type, .. } => format!("[image: {media_type}]"),
        RenderBlock::Card { card, .. } => card.plain_text(),
        RenderBlock::Usage { .. }
        | RenderBlock::CompactionProgress { .. }
        | RenderBlock::RateLimit(_) => String::new(),
    }
}
