//! Layout cache + per-block height measurement — the memoization layer
//! behind the O(1) render hotpath (cached `(idx, top, height)` tuples,
//! per-block rendered-line caches, and inter-block gap rules).

#![allow(clippy::doc_markdown)]

use runtime::message_stream::{BlockId, RenderBlock};

use crate::tui::blocks;
use crate::tui::image_protocol::ImageProtocol;
use crate::tui::theme::Theme;

use super::tool_groups::{
    ToolGroupState, apply_group_reveals, collapsed_summary_height, collapsed_tool_detail_lines,
    compute_tool_groups, group_span_end, recompute_tool_groups_tail, span_is_live,
    tool_group_recompute_start,
};
use super::{
    DEFAULT_BLOCK_ROWS, RenderCache, Transcript, assistant_prose_style, is_response_block,
    is_tool_block, lines_to_static,
};

impl Transcript {
    /// Compute block height, using the rendered-lines cache for
    /// `TextDelta` blocks when available. Updates the cache on miss.
    /// Height (rows) for a markdown-bearing block (`TextDelta` / `UserMessage`),
    /// populating [`RenderCache::Text`] as a side effect. Returns `None` for any
    /// other block so the caller falls through to the next cache policy.
    ///
    /// Streaming (`done == false`) renders incrementally: completed markdown
    /// blocks are styled once and the open tail re-rendered per frame (see
    /// [`crate::tui::blocks::text::streaming_incremental`]); normal-sized
    /// completions run one authoritative full pass on `done`, while large
    /// completions keep the incremental cache to protect input responsiveness.
    // Cohesive cache-policy branch (hit → O(1) height; miss → done/streaming
    // render + cache fill). Splitting it would thread the same dozen locals
    // through a helper for no clarity gain.
    #[allow(clippy::too_many_lines)]
    fn text_block_height(&mut self, idx: usize, width: u16, theme: &Theme) -> Option<u16> {
        let (text, done, is_user_msg) = match &self.blocks[idx] {
            RenderBlock::TextDelta { text, done, .. } => (text.as_str(), *done, false),
            RenderBlock::UserMessage { text, .. } => (text.as_str(), true, true),
            _ => return None,
        };
        // Height measurement must match draw width. User messages reserve a
        // 3-cell role rail (`┃  ` / `|  `) in `draw_user_message_from_cache`,
        // and marked assistant prose (bullet *or* indent continuation) reserves
        // the same-width mark column, so their cached markdown body must be
        // wrapped to the remaining body width. Otherwise long prompts gain
        // extra visual rows at draw-time and the next block can appear glued.
        let prose_style = assistant_prose_style(&self.blocks, idx);
        // Marked prose (and user messages) additionally cap their measure so
        // ultra-wide terminals retain a right-side breathing margin (v3
        // readability; see `prose_wrap_cap`).
        let body_width = if is_user_msg || prose_style.has_indent() {
            let available_width = width.saturating_sub(blocks::ROLE_RAIL_WIDTH);
            available_width.clamp(1, blocks::prose_wrap_cap(available_width))
        } else {
            width.max(1)
        };
        let cache_width = body_width;
        let content_version = self.render_version(idx);
        if self.rendered_cache.len() <= idx {
            self.rendered_cache.resize_with(self.blocks.len(), || None);
        }
        // User messages carry a `You` header row. Assistant prose has no
        // header in the bullet grammar — the `◆` bullet rides the first body
        // row, so it adds no height.
        let header_row: u16 = u16::from(is_user_msg);
        let theme_fp = self.render_cache_theme_fp;
        // O(1) hit: the cached row prefix already carries the wrapped height —
        // re-running `wrapped_rows` over every line here made each layout
        // refresh O(total message) while streaming. Computed into a local so
        // the hit tally can be recorded after the `rendered_cache` borrow ends.
        let cached_body = match &self.rendered_cache[idx] {
            Some(RenderCache::Text {
                content_version: cached_version,
                theme_fp: cached_theme_fp,
                width: cached_w,
                done: cached_done,
                preserves,
                lines,
                row_prefix,
                ..
            }) if *cached_version == content_version
                && *cached_theme_fp == theme_fp
                && *cached_w == cache_width
                && *cached_done == done =>
            {
                Some(if *preserves {
                    u16::try_from(lines.len()).unwrap_or(u16::MAX).max(1)
                } else {
                    u16::try_from(row_prefix.last().copied().unwrap_or(0))
                        .unwrap_or(u16::MAX)
                        .max(1)
                })
            }
            _ => None,
        };
        if let Some(body) = cached_body {
            self.render_cache_hits = self.render_cache_hits.saturating_add(1);
            return Some(body.saturating_add(header_row));
        }
        self.render_cache_misses = self.render_cache_misses.saturating_add(1);
        // Cache miss. While streaming, reuse the previously-styled stable prefix
        // (moved out — zero copy) so completed blocks are styled exactly once;
        // only the small open tail is re-rendered per frame. On completion, very
        // large blocks keep that incremental cache instead of forcing a full
        // markdown + syntect pass on the input/render loop.
        let previous = self.rendered_cache[idx].take();
        let mut done_incremental_seed = None;
        let (prev_lines, prev_rows, prev_stable_len, prev_stable_count, prev_scan) = match previous {
            Some(RenderCache::Text {
                theme_fp: prev_theme_fp,
                width: prev_w,
                done: prev_done,
                lines: prev_lines,
                row_prefix: prev_rows,
                stable_len,
                stable_line_count,
                scan,
                ..
            }) if prev_theme_fp == theme_fp && prev_w == cache_width && !prev_done => {
                if done && should_keep_incremental_done_cache(text) {
                    done_incremental_seed =
                        Some((prev_lines, prev_rows, stable_len, stable_line_count));
                    (Vec::new(), Vec::new(), 0, 0, None)
                } else if !done {
                    (prev_lines, prev_rows, stable_len, stable_line_count, scan)
                } else {
                    (Vec::new(), Vec::new(), 0, 0, None)
                }
            }
            _ => (Vec::new(), Vec::new(), 0, 0, None),
        };
        let (lines, row_prefix, preserves, stable_len, stable_line_count, scan) = if done {
            if let Some((seed_lines, seed_rows, seed_stable_len, seed_stable_count)) =
                done_incremental_seed
            {
                let (lines, row_prefix, _, _) = blocks::text::streaming_incremental(
                    text,
                    theme,
                    cache_width,
                    seed_lines,
                    seed_rows,
                    seed_stable_len,
                    seed_stable_count,
                );
                (lines, row_prefix, false, 0, 0, None)
            } else {
                let lines =
                    blocks::text::rendered_lines_for_width(text, true, theme, 0, cache_width);
                let preserves = super::blocks::text::preserves_layout_pub(text);
                let row_prefix = if preserves {
                    Vec::new()
                } else {
                    super::blocks::wrapped_row_prefix(&lines, cache_width)
                };
                (lines, row_prefix, preserves, 0, 0, None)
            }
        } else {
            // Streaming: incrementally styled. Never layout-preserving — always
            // let Paragraph wrap; the markdown layout policy (tables / ASCII
            // trees) applies on the `done` pass for normal-sized blocks. The
            // resumed scan cursor makes each frame's stable-prefix scan O(new
            // suffix) instead of O(whole text) (the streaming-freeze fix).
            let (lines, row_prefix, stable_len, stable_count, scan) =
                blocks::text::streaming_incremental_resumed(
                    text,
                    theme,
                    cache_width,
                    prev_lines,
                    prev_rows,
                    prev_stable_len,
                    prev_stable_count,
                    prev_scan,
                );
            (lines, row_prefix, false, stable_len, stable_count, scan)
        };
        let height = if preserves {
            u16::try_from(lines.len()).unwrap_or(u16::MAX).max(1)
        } else {
            u16::try_from(row_prefix.last().copied().unwrap_or(0))
                .unwrap_or(u16::MAX)
                .max(1)
        };
        self.rendered_cache[idx] = Some(RenderCache::Text {
            content_version,
            theme_fp,
            width: cache_width,
            done,
            preserves,
            lines,
            row_prefix,
            stable_len,
            stable_line_count,
            scan,
        });
        Some(height.saturating_add(header_row))
    }

    /// Height of a collapsed tool group's summary, cached for settled groups.
    ///
    /// A settled summary is re-measured on every suffix rebuild that covers its
    /// leader and re-styled on every visible frame, yet its rows only change
    /// when the span itself changes — so build the styled lines once, keep them
    /// with the height in the leader's [`RenderCache::Group`] slot, and let the
    /// draw site reuse them. Live groups (an in-flight member) mutate as
    /// results land and animate per-tool status markers, so they stay on the
    /// direct measure path and are never cached.
    fn collapsed_group_height(
        &mut self,
        idx: usize,
        err_count: u16,
        width: u16,
        theme: &Theme,
    ) -> u16 {
        if span_is_live(&self.blocks, &self.tool_groups, idx) {
            // A settled run that a chained parallel batch re-leads as LIVE
            // keeps the same leader block — drop its stale settled entry here,
            // or the draw's (id, width) match would keep painting the old
            // settled summary (no spinner, new tools invisible) against the
            // fresh live height until the whole batch finishes. The re-lead
            // always re-measures the leader (the group recompute walks back to
            // the run start), so this bypass is guaranteed to run on that
            // transition frame.
            if let Some(slot) = self.rendered_cache.get_mut(idx) {
                if matches!(slot, Some(RenderCache::Group { .. })) {
                    *slot = None;
                }
            }
            return collapsed_summary_height(&self.blocks, &self.tool_groups, idx);
        }
        let leader_block_id = block_id(&self.blocks[idx]).0;
        let span_end = group_span_end(&self.blocks, &self.tool_groups, idx);
        let span_len = span_end.saturating_sub(idx);
        // Wrapping sum of the span members' render versions: an in-place
        // member rewrite bumps its version, so the key sees it even when the
        // span shape (a revealed diff pair, the Ctrl+X window) hides the
        // leader from any state walk.
        let span_versions = self
            .render_versions
            .get(idx..span_end.min(self.render_versions.len()))
            .map_or(0u64, |versions| {
                versions.iter().fold(0u64, |acc, v| acc.wrapping_add(*v))
            });
        if self.rendered_cache.len() <= idx {
            self.rendered_cache.resize_with(self.blocks.len(), || None);
        }
        let theme_fp = self.render_cache_theme_fp;
        let cached_height = match &self.rendered_cache[idx] {
            Some(RenderCache::Group {
                leader_block_id: cached_id,
                theme_fp: cached_theme_fp,
                width: cached_width,
                span_len: cached_span,
                err_count: cached_err,
                span_versions: cached_versions,
                height,
                ..
            }) if *cached_id == leader_block_id
                && *cached_theme_fp == theme_fp
                && *cached_width == width
                && *cached_span == span_len
                && *cached_err == err_count
                && *cached_versions == span_versions =>
            {
                Some(*height)
            }
            _ => None,
        };
        if let Some(height) = cached_height {
            self.render_cache_hits = self.render_cache_hits.saturating_add(1);
            return height;
        }
        self.render_cache_misses = self.render_cache_misses.saturating_add(1);
        let height = collapsed_summary_height(&self.blocks, &self.tool_groups, idx);
        let lines =
            collapsed_tool_detail_lines(&self.blocks, &self.tool_groups, idx, err_count, theme, width);
        self.rendered_cache[idx] = Some(RenderCache::Group {
            leader_block_id,
            theme_fp,
            width,
            span_len,
            err_count,
            span_versions,
            height,
            lines,
        });
        height
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn cached_block_height(
        &mut self,
        idx: usize,
        width: u16,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) -> u16 {
        // Guard against out-of-bounds access from stale layout indices.
        if idx >= self.blocks.len() {
            return 1;
        }
        // Tool group collapse: hidden blocks get height 0; a summary leader
        // (live or settled) gets one row per member tool (capped).
        if let Some(state) = self.tool_groups.get(idx) {
            match state {
                ToolGroupState::Hidden => return 0,
                ToolGroupState::Summary { err_count, .. } => {
                    let err_count = *err_count;
                    return self.collapsed_group_height(idx, err_count, width, theme);
                }
                ToolGroupState::Normal => {}
            }
        }
        // While a turn streams, a Todos result contributes no transcript rows —
        // the live pinned panel owns the plan; it reappears as settled `Updated
        // Plan` history once the turn ends. `set_turn_active` invalidates the
        // layout on the turn edge so this height flips in lockstep with the
        // draw-site suppression (both via `todos_suppressed_during_turn`).
        if self.todos_suppressed_during_turn(idx) {
            return 0;
        }
        // A settled assistant prose block with no visible body is a phantom
        // (provider closed an empty text part). Contribute no rows so the
        // transcript never reserves the `Zo` author header for an answer
        // that does not exist — the empty `✓ Zo · done` block bug. Kept in
        // lockstep with the draw-site skip via `is_empty_prose_suppressed`.
        if self.is_empty_prose_suppressed(idx) {
            return 0;
        }
        let focused = self.focused_idx == Some(idx);
        let expanded = self.is_expanded(idx);

        // Cache markdown-bearing blocks: TextDelta (assistant streams) and
        // UserMessage (user pastes) share the same markdown engine + cache.
        if let Some(height) = self.text_block_height(idx, width, theme) {
            return height;
        }

        // ToolResult / Diff height cache — `block_id` 가 stable key 라 매
        // frame syntect/diff highlight 재계산이 첫 호출 후 0 으로 수렴한다.
        // 변형 인자 (focused/expanded) 가 바뀌면 자연 cache miss.
        // draw site 의 `rendered_lines` 추가 호출은 별도 chunk (A.3c-2) 로
        // 분리 — 본 chunk 는 layout 계산 경로만 해소.
        if let RenderBlock::ToolResult {
            id, is_error, body, ..
        } = &self.blocks[idx]
        {
            let block_id_val = id.0;
            let is_error_val = *is_error;
            // ToolResult now renders at full block width; keep measurement and
            // draw in lockstep so tail scrolling does not drift.
            let measure_width = width.max(1);
            if self.rendered_cache.len() <= idx {
                self.rendered_cache.resize_with(self.blocks.len(), || None);
            }
            let theme_fp = self.render_cache_theme_fp;
            // O(1) from the cached prefix-sum instead of re-wrapping the whole
            // tool body every frame (`row_prefix.last()` == the old
            // `wrapped_rows`; see `wrapped_row_prefix_matches_wrapped_rows`).
            // Computed into a local so the hit tally records after the borrow.
            let cached_height = match &self.rendered_cache[idx] {
                Some(RenderCache::Tool {
                    block_id,
                    theme_fp: cached_theme_fp,
                    width: cached_w,
                    focused: cached_focused,
                    expanded: cached_expanded,
                    row_prefix,
                    ..
                }) if *block_id == block_id_val
                    && *cached_theme_fp == theme_fp
                    && *cached_w == width
                    && *cached_focused == focused
                    && *cached_expanded == expanded =>
                {
                    Some(u16::try_from(row_prefix.last().copied().unwrap_or(0)).unwrap_or(u16::MAX))
                }
                _ => None,
            };
            if let Some(height) = cached_height {
                self.render_cache_hits = self.render_cache_hits.saturating_add(1);
                return height;
            }
            self.render_cache_misses = self.render_cache_misses.saturating_add(1);
            let rendered = blocks::tool_result::rendered_lines_for_width(
                is_error_val,
                body,
                theme,
                focused,
                expanded,
                width,
            );
            let owned = lines_to_static(rendered);
            // Wrap-row prefix-sum, measured once on the cache miss with the same
            // engine as `wrapped_rows`, so both the O(1) height read above and the
            // windowed draw share one authoritative layout.
            let row_prefix = super::blocks::wrapped_row_prefix(&owned, measure_width);
            let height =
                u16::try_from(row_prefix.last().copied().unwrap_or(0)).unwrap_or(u16::MAX);
            self.rendered_cache[idx] = Some(RenderCache::Tool {
                block_id: block_id_val,
                theme_fp,
                width,
                focused,
                expanded,
                lines: owned,
                row_prefix,
            });
            return height;
        }

        // Reasoning 높이는 elapsed 접미("· N.Ns")가 줄 폭에 영향을 줄 수 있어
        // draw 와 동일한 elapsed 로 측정해야 정합한다. done 전환 프레임에서
        // layout 이 dirty 로 재측정되므로 동결값과 draw 가 같은 폭을 본다.
        if let RenderBlock::Reasoning { id, text, done, .. } = &self.blocks[idx] {
            if self.is_reasoning_visually_suppressed(idx) {
                return 0;
            }
            let elapsed = self.reasoning_display_elapsed(id.0);
            return blocks::reasoning::estimate_rows(
                text, *done, theme, focused, expanded, width, elapsed, id.0,
            );
        }

        // ToolCall 높이는 에이전트 트리(사이드테이블) 행 수를 포함해야 draw 와
        // 일치한다 — 트리 갱신은 `set_agent_tree` 가 dirty 마킹으로 재측정시킨다.
        if let RenderBlock::ToolCall {
            id,
            tool_call_id,
            name,
            summary,
            preview,
            status,
            ..
        } = &self.blocks[idx]
        {
            return blocks::tool_call::estimate_rows(
                Some(&tool_call_id.0),
                name,
                summary,
                preview,
                *status,
                theme,
                width,
                self.agent_trees.get(&tool_call_id.0),
                self.expanded.contains(&id.0)
                    && *status == runtime::message_stream::ToolCallStatus::Running
                    && blocks::tool_call::is_bash(name),
            );
        }

        block_height(
            &self.blocks[idx],
            focused,
            expanded,
            width,
            theme,
            image_protocol,
        )
    }

    /// Invalidate the cached layout so the next `calculate_layout`
    /// recomputes from scratch.
    pub(super) fn invalidate_layout_cache(&mut self) {
        self.cached_layout.clear();
        self.cached_layout_width = 0;
        self.cached_layout_block_count = 0;
        self.cached_layout_dirty = false;
        self.layout_dirty_from = None;
    }

    /// Ensure `self.cached_layout` is up-to-date for the given width.
    ///
    /// Three cases, cheapest first:
    /// 1. **Hit** — nothing changed, return.
    /// 2. **Incremental** — same width with a live cache: an in-place mutation
    ///    (same count, dirty) or an append (count grew). Every branch rebuilds
    ///    only a suffix; the internal ladder is ordered cheapest-first and the
    ///    first matching branch returns.
    /// 3. **Full** — width changed, blocks removed, cold cache, or the group
    ///    shape itself changed.
    // One cohesive cache-state dispatch; splitting branches into helpers would
    // scatter the shared guards and obscure the precedence documented inline.
    #[allow(clippy::too_many_lines)]
    pub(super) fn ensure_layout(
        &mut self,
        width: u16,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) {
        // Before the dirty marks are consumed: a re-wrap or an in-place
        // mutation above the char-selection shifts the content rows it is
        // anchored to, so the gesture must drop rather than wash shifted text.
        self.drop_char_selection_on_layout_shift(width);
        // Theme identity is part of every per-block render-cache key. Refresh
        // the tracked fingerprint here (once per pass, not per block); a change
        // — a live `/theme` switch — drops the layout cache so the pass below
        // takes the full-rebuild path and each per-block key then misses on the
        // new fingerprint and re-renders under the new palette. Placed after
        // the char-selection check so a palette-only switch (same width) does
        // not read as a width shift. Heights are palette-independent, so the
        // rebuilt geometry is identical.
        let theme_fp = theme.render_cache_fingerprint();
        if self.render_cache_theme_fp != theme_fp {
            self.render_cache_theme_fp = theme_fp;
            self.invalidate_layout_cache();
        }
        let same_width = self.cached_layout_width == width;
        let same_count = self.cached_layout_block_count == self.blocks.len();
        let has_cache = !self.cached_layout.is_empty();
        // Lowest in-place-mutated index since the last pass. The tail fast
        // paths below are only valid when nothing *before* the tail moved —
        // a mid-list `upsert_system` or an out-of-order tool-status reconcile
        // must fall through to a suffix rebuild from that index.
        // `mark_layout_dirty_from` is the only dirty setter and always records
        // the index, so `cached_layout_dirty` implies a real `dirty_from`.
        let dirty_from = self.layout_dirty_from.unwrap_or(usize::MAX);
        let only_tail_dirty = dirty_from >= self.blocks.len().saturating_sub(1);
        let groups_aligned = self.tool_groups.len() == self.blocks.len();

        // Case 1: Exact cache hit — nothing changed.
        if has_cache && same_width && same_count && !self.cached_layout_dirty {
            return;
        }

        // Case 2: Incremental — same width, live cache.
        if has_cache && same_width {
            if same_count && self.cached_layout_dirty {
                // 2a: Streaming text/reasoning merged into the tail. Tool
                // groups cannot change, so re-measure the last entry in place —
                // the O(1) per-token path; everything below it is per-event.
                if only_tail_dirty
                    && groups_aligned
                    && matches!(
                        self.blocks.last(),
                        Some(RenderBlock::TextDelta { .. } | RenderBlock::Reasoning { .. })
                    )
                {
                    let last_idx = self.cached_layout.last().map(|e| e.0);
                    if let Some(idx) = last_idx {
                        let h = self.cached_block_height(idx, width, theme, image_protocol);
                        if let Some(entry) = self.cached_layout.last_mut() {
                            entry.2 = h;
                        }
                    }
                    self.cached_layout_dirty = false;
                    self.layout_dirty_from = None;
                    return;
                }

                // 2b: a ToolCall status flip — tail or mid-list (a parallel
                // tool settling out of order). A status change is
                // structural-preserving: it can only alter the collapse counts
                // of the contiguous tool run that contains it, never the
                // grouping of any earlier run. Recompute just that run (from
                // its boundary) instead of paying the O(all blocks)
                // `compute_tool_groups` scan on *every* status event — the
                // cost that made tool/agent-heavy turns lag as context filled.
                // `dirty_from` is the lowest mutated index, so a run recompute
                // from its boundary to the end covers every dirty block. (A
                // tail flip is the same operation: its run boundary and the
                // rebuild start land on the same indices.)
                if groups_aligned
                    && dirty_from < self.blocks.len()
                    && matches!(
                        self.blocks.get(dirty_from),
                        Some(RenderBlock::ToolCall { .. })
                    )
                {
                    let recompute_from = tool_group_recompute_start(&self.blocks, dirty_from + 1);
                    recompute_tool_groups_tail(&self.blocks, &mut self.tool_groups, recompute_from);
                    apply_group_reveals(&self.blocks, &mut self.tool_groups, &self.revealed_groups);
                    self.rebuild_layout_suffix(
                        recompute_from.min(dirty_from),
                        width,
                        theme,
                        image_protocol,
                    );
                    return;
                }

                // 2c: streaming prose started after an unfinished Reasoning
                // block. `Transcript::push` marks the Reasoning index dirty so
                // its transient placeholder height can collapse once prose
                // appears. Tool grouping cannot change for a
                // Reasoning/TextDelta suffix, so rebuild just that two-block
                // suffix instead of scanning every block per streamed token.
                if groups_aligned
                    && dirty_from.saturating_add(1) == self.blocks.len().saturating_sub(1)
                    && matches!(
                        self.blocks.get(dirty_from),
                        Some(RenderBlock::Reasoning { done: false, .. })
                    )
                    && matches!(
                        self.blocks.last(),
                        Some(RenderBlock::TextDelta { text, .. }) if !text.is_empty()
                    )
                {
                    self.rebuild_layout_suffix(dirty_from, width, theme, image_protocol);
                    return;
                }

                // 2d: any other in-place mutation. If the group shape survived
                // it, rebuild from the lowest mutated index; a shape change
                // falls through to the full rebuild. Reveals are applied to the
                // fresh compute BEFORE the compare — `self.tool_groups` already
                // carries them, so a raw compute would spuriously mismatch
                // whenever any group is user-revealed.
                let mut next_tool_groups = compute_tool_groups(&self.blocks);
                apply_group_reveals(&self.blocks, &mut next_tool_groups, &self.revealed_groups);
                if self.tool_groups == next_tool_groups {
                    let start = dirty_from.min(self.blocks.len().saturating_sub(1));
                    self.rebuild_layout_suffix(start, width, theme, image_protocol);
                    return;
                }
            } else if self.cached_layout_block_count < self.blocks.len()
                && self.cached_layout_block_count > 0
                && self.tool_groups.len() >= self.cached_layout_block_count
            {
                // 2e: append-only — new blocks after the cached prefix. A new
                // block can only alter collapse state inside the trailing tool
                // run that touches the append point — but an in-place mutation
                // may have landed *earlier in the same drain batch* (the
                // classic case: a streaming text block's final delta + `done`
                // merge immediately followed by the next ToolCall append), so
                // the rebuild must start no later than that index or the
                // merged block keeps its truncated streaming render forever.
                let prev_count = self.cached_layout_block_count;
                let recompute_from = tool_group_recompute_start(&self.blocks, prev_count);
                recompute_tool_groups_tail(&self.blocks, &mut self.tool_groups, recompute_from);
                apply_group_reveals(&self.blocks, &mut self.tool_groups, &self.revealed_groups);
                self.rebuild_layout_suffix(
                    recompute_from.min(dirty_from),
                    width,
                    theme,
                    image_protocol,
                );
                return;
            }
        }

        // Case 3: Full recalculation (width changed, blocks removed,
        // cold cache, or the group shape changed under an in-place mutation).
        let mut next_tool_groups = compute_tool_groups(&self.blocks);
        apply_group_reveals(&self.blocks, &mut next_tool_groups, &self.revealed_groups);
        self.tool_groups = next_tool_groups;
        self.cached_layout.clear();
        let mut cursor: u16 = 0;
        for idx in 0..self.blocks.len() {
            let height = self.cached_block_height(idx, width, theme, image_protocol);
            self.cached_layout.push((idx, cursor, height));
            cursor = cursor.saturating_add(height);
            if idx + 1 < self.blocks.len() {
                cursor = cursor.saturating_add(self.visual_block_gap(idx, idx + 1, theme, width));
            }
        }
        self.cached_layout_width = width;
        self.cached_layout_block_count = self.blocks.len();
        self.cached_layout_dirty = false;
        self.layout_dirty_from = None;
    }

    fn rebuild_layout_suffix(
        &mut self,
        start_idx: usize,
        width: u16,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) {
        let start_idx = start_idx.min(self.blocks.len());
        self.cached_layout.truncate(start_idx);

        let mut cursor = if start_idx == 0 {
            0
        } else {
            let previous_idx = start_idx - 1;
            self.cached_layout
                .last()
                .map_or(0, |(_, top, height)| top.saturating_add(*height))
                .saturating_add(self.visual_block_gap(previous_idx, start_idx, theme, width))
        };

        for idx in start_idx..self.blocks.len() {
            let height = self.cached_block_height(idx, width, theme, image_protocol);
            self.cached_layout.push((idx, cursor, height));
            cursor = cursor.saturating_add(height);
            if idx + 1 < self.blocks.len() {
                cursor = cursor.saturating_add(self.visual_block_gap(idx, idx + 1, theme, width));
            }
        }
        self.cached_layout_width = width;
        self.cached_layout_block_count = self.blocks.len();
        self.cached_layout_dirty = false;
        self.layout_dirty_from = None;
    }

    fn visual_block_gap(
        &self,
        current_idx: usize,
        next_idx: usize,
        theme: &Theme,
        width: u16,
    ) -> u16 {
        let current_state = self.tool_groups.get(current_idx);
        let next_state = self.tool_groups.get(next_idx);
        if matches!(next_state, Some(ToolGroupState::Hidden)) {
            return 0;
        }

        let current_reasoning_suppressed = self.is_reasoning_visually_suppressed(current_idx);
        let next_reasoning_suppressed = self.is_reasoning_visually_suppressed(next_idx);
        // A phantom (empty, settled) prose block contributes no rows, so any gap
        // to/from it must collapse exactly like a hidden reasoning row — defer
        // the boundary to the next visible edge so User→assistant separators
        // are not doubled or stranded around a 0-row block.
        let current_empty_prose = self.is_empty_prose_suppressed(current_idx);
        let next_empty_prose = self.is_empty_prose_suppressed(next_idx);
        if next_reasoning_suppressed || next_empty_prose {
            // Defer the gap across a hidden reasoning block to the next visible
            // edge. This preserves User→assistant separators when a transient
            // Thinking row sits between them and later collapses to height 0.
            return 0;
        }

        // Consecutive tool clusters (a settled summary followed by the next
        // batch) read as one continuous operation log — no meaningless blank row
        // between them, matching the zero gap *within* a cluster. The wide
        // breathing room belongs between prose and tools, not between tool rows.
        if self.visible_anchor_is_tool_summary(current_idx)
            && self.blocks.get(next_idx).is_some_and(is_tool_block)
        {
            return 0;
        }

        let current = if matches!(current_state, Some(ToolGroupState::Hidden))
            || current_reasoning_suppressed
            || current_empty_prose
        {
            self.visible_gap_anchor_before(current_idx)
        } else {
            self.blocks.get(current_idx)
        };

        match (current, self.blocks.get(next_idx)) {
            (Some(current), Some(next)) => block_gap(current, next, theme, width),
            _ => 0,
        }
    }

    fn visible_gap_anchor_before(&self, idx: usize) -> Option<&RenderBlock> {
        (0..=idx).rev().find_map(|candidate| {
            if matches!(
                self.tool_groups.get(candidate),
                Some(ToolGroupState::Hidden)
            ) || self.is_reasoning_visually_suppressed(candidate)
                || self.is_empty_prose_suppressed(candidate)
            {
                None
            } else {
                self.blocks.get(candidate)
            }
        })
    }

    fn visible_anchor_is_tool_summary(&self, idx: usize) -> bool {
        for candidate in (0..=idx).rev() {
            if self.is_reasoning_visually_suppressed(candidate)
                || self.is_empty_prose_suppressed(candidate)
            {
                continue;
            }
            match self.tool_groups.get(candidate) {
                Some(ToolGroupState::Hidden) => {}
                Some(ToolGroupState::Summary { .. }) => return true,
                _ => return false,
            }
        }
        false
    }
}

/// Conversation turn 경계 gap (components.md §2) — 폭 breakpoint 에 비례.
///
/// A new user turn gets the wider 3-row author boundary. User→assistant response
/// uses a tighter 2-row boundary because the assistant's labeled prose already
/// reserves its own breathing row; using 3 rows there double-counts the spacer and
/// creates the visible "one blank row too many" gap before `Zo`.
pub(super) fn turn_boundary_gap(_theme: &Theme, _width: u16) -> u16 {
    3
}

pub(super) fn response_boundary_gap(_theme: &Theme, _width: u16) -> u16 {
    2
}

/// Tool/prose transitions happen inside one assistant turn. Keep one blank row
/// of air, then let the explicit `Zo` label (for prose) or tool marker carry
/// the boundary. Larger gaps made a single workflow look broken into chunks.
pub(super) fn prose_tool_boundary_gap(_theme: &Theme, _width: u16) -> u16 {
    1
}

fn block_gap(current: &RenderBlock, next: &RenderBlock, theme: &Theme, width: u16) -> u16 {
    // ToolCall → its matching ToolResult: zero gap (tight pair).
    if let (
        RenderBlock::ToolCall {
            tool_call_id: current_tool_call_id,
            ..
        },
        RenderBlock::ToolResult {
            tool_call_id: next_tool_call_id,
            ..
        },
    ) = (current, next)
    {
        if current_tool_call_id == next_tool_call_id {
            return 0;
        }
    }

    // Tool clusters should read as one compact operation log. The wide
    // breathing room belongs between prose and tools, not between sibling tool
    // rows/results in the same cluster.
    if is_tool_block(current) && is_tool_block(next) {
        return 0;
    }

    // Turn boundary: any non-UserMessage → UserMessage gets a wide
    // gap so conversation turns are visually separated.
    if matches!(next, RenderBlock::UserMessage { .. })
        && !matches!(current, RenderBlock::UserMessage { .. })
    {
        return turn_boundary_gap(theme, width);
    }

    // Response boundary: a user prompt followed by assistant-side output needs
    // the same visual break. This is especially important when the model is
    // quiet for a while and the first visible output is a tool call; otherwise
    // the tool row reads as if it belongs to the user's message.
    if matches!(current, RenderBlock::UserMessage { .. }) && is_response_block(next) {
        return response_boundary_gap(theme, width);
    }

    // Prose ↔ tool boundary gets its own slightly larger gap so authorship
    // stays legible. A *visible* reasoning block counts as prose here: it is the
    // model's voice immediately before/after a tool, so the same boundary
    // applies (and stays correct if the base and prose-tool gaps ever diverge).
    // This is every visible Reasoning, not only the streaming `Thinking…` line:
    // a *long settled* thought now also leaves a one-line trace (roadmap ③) and
    // must read as prose too. Suppressed reasoning never reaches this point —
    // `visual_block_gap` swaps a hidden `current` for its visible anchor and
    // early-returns on a hidden `next` — so matching all Reasoning is safe.
    let current_is_prose = matches!(current, RenderBlock::TextDelta { .. })
        || matches!(current, RenderBlock::Reasoning { .. });
    let next_is_prose = matches!(next, RenderBlock::TextDelta { .. })
        || matches!(next, RenderBlock::Reasoning { .. });
    if (is_tool_block(current) && next_is_prose) || (current_is_prose && is_tool_block(next)) {
        return prose_tool_boundary_gap(theme, width);
    }

    // Every other adjacency keeps the base rhythm. Sibling tool rows and
    // matching call/result pairs already collapse to zero above; prose/tool
    // boundaries get their own slightly larger gap so authorship stays legible.
    theme.spacing.block_gap
}

pub(super) fn separator_y(layout: &[(usize, u16, u16)], layout_idx: usize) -> Option<u16> {
    if layout_idx == 0 {
        return None;
    }
    let (_, prev_top, prev_height) = *layout.get(layout_idx - 1)?;
    let (_, block_top, _) = *layout.get(layout_idx)?;
    let prev_bottom = prev_top.saturating_add(prev_height);
    let gap = block_top.saturating_sub(prev_bottom);
    (gap > 0).then_some(prev_bottom.saturating_add(gap / 2))
}

fn block_height(
    block: &RenderBlock,
    focused: bool,
    expanded: bool,
    width: u16,
    theme: &Theme,
    image_protocol: ImageProtocol,
) -> u16 {
    match block {
        RenderBlock::TextDelta { text, done, .. } => {
            blocks::text::estimate_rows(text, *done, theme, width)
        }
        RenderBlock::Reasoning { id, text, done, .. } => {
            // Reasoning 은 cached_block_height 가 elapsed 포함으로 선처리하므로
            // 이 자유함수 경로는 폴백(elapsed 없음). seed 는 블록 id 로 안정.
            blocks::reasoning::estimate_rows(
                text, *done, theme, focused, expanded, width, None, id.0,
            )
        }
        RenderBlock::ToolCall {
            tool_call_id,
            name,
            summary,
            preview,
            status,
            ..
        } => blocks::tool_call::estimate_rows(
            Some(&tool_call_id.0),
            name,
            summary,
            preview,
            *status,
            theme,
            width,
            None,
            expanded
                && *status == runtime::message_stream::ToolCallStatus::Running
                && blocks::tool_call::is_bash(name),
        ),
        RenderBlock::ToolResult { is_error, body, .. } => {
            blocks::tool_result::estimate_rows(*is_error, body, theme, focused, expanded, width)
        }
        RenderBlock::PermissionPrompt(prompt) => {
            let selected = blocks::permission::default_selected_index(prompt);
            blocks::permission::estimate_rows(prompt, theme, selected, width)
        }
        RenderBlock::UserQuestionPrompt(_) => DEFAULT_BLOCK_ROWS,
        RenderBlock::Image {
            data, media_type, ..
        } => blocks::image::estimate_rows(data, media_type, image_protocol, theme, width),
        RenderBlock::UserMessage { text, .. } => {
            #[allow(clippy::cast_possible_truncation)]
            {
                text.lines().count().max(1) as u16
            }
        }
        RenderBlock::System { level, text, .. } => {
            blocks::system::estimate_rows(*level, text, theme, width)
        }
        RenderBlock::UserNotice { message, .. } => {
            blocks::user_notice::estimate_rows(message, theme, width)
        }
        RenderBlock::Card { card, .. } => crate::tui::cards::estimate_rows(card, theme, width),
        RenderBlock::AgentResult {
            label,
            status,
            body,
            ..
        } => {
            let view = blocks::agent_result::AgentCardView {
                label,
                status: *status,
                body,
                expanded,
                focused,
            };
            blocks::agent_result::estimate_rows(&view, theme, width)
        }
        // Usage/RateLimit are live-ledger events, never transcript blocks.
        RenderBlock::Usage { .. }
        | RenderBlock::CompactionProgress { .. }
        | RenderBlock::RateLimit(_) => 0,
    }
}

pub(super) fn block_id(block: &RenderBlock) -> BlockId {
    match block {
        RenderBlock::TextDelta { id, .. }
        | RenderBlock::Reasoning { id, .. }
        | RenderBlock::ToolCall { id, .. }
        | RenderBlock::ToolResult { id, .. }
        | RenderBlock::Image { id, .. }
        | RenderBlock::UserMessage { id, .. }
        | RenderBlock::UserNotice { id, .. }
        | RenderBlock::AgentResult { id, .. }
        | RenderBlock::System { id, .. }
        | RenderBlock::Card { id, .. } => *id,
        RenderBlock::PermissionPrompt(p) => p.id,
        RenderBlock::UserQuestionPrompt(p) => p.id,
        // Usage/RateLimit never enter the transcript; sentinel id keeps totals.
        RenderBlock::Usage { .. }
        | RenderBlock::CompactionProgress { .. }
        | RenderBlock::RateLimit(_) => BlockId(0),
    }
}

/// Assistant prose larger than this keeps the incremental render cache when the
/// stream flips to `done`. A synchronous full markdown/syntect pass over a block
/// this large is visible as an input freeze because it runs on the TUI loop.
const FINAL_DONE_FULL_RENDER_LIMIT: usize = 96 * 1024;

/// Markdown-heavy prose also becomes expensive before the generic prose cap:
/// the final pass reparses and rewraps the whole block on the TUI loop. Keep the
/// incremental no-syntect cache once a long answer clearly contains markdown.
const FINAL_DONE_MARKDOWN_RENDER_LIMIT: usize = 32 * 1024;

/// Code-heavy answers become expensive earlier than prose because the normal
/// final render syntax-highlights fenced blocks with syntect.
const FINAL_DONE_CODE_RENDER_LIMIT: usize = 24 * 1024;

fn should_keep_incremental_done_cache(text: &str) -> bool {
    text.len() > FINAL_DONE_FULL_RENDER_LIMIT
        || (text.len() > FINAL_DONE_MARKDOWN_RENDER_LIMIT
            && crate::tui::markdown::has_strong_markdown_signal(text))
        || (text.len() > FINAL_DONE_CODE_RENDER_LIMIT
            && (text.contains("```") || text.contains("~~~")))
}

// Text render cache keys use `Transcript::render_version`: O(1) to compare and
// bumped only at mutation sites. Do not replace it with a full-content hash on
// the streaming hot path; hashing the accumulated answer every frame makes
// long outputs pause and then catch up.
