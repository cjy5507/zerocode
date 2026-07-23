//! Transcript viewport — scrollable stack of [`RenderBlock`] widgets.
//!
//! Per `.zo/design/components.md` §1.1 (layout) + §7 (scroll
//! indicator) + §9.2 (focused-block cursor, space expands).
//!
//! Lane boundary: this module owns **viewport** state only. The
//! event loop (L2 + L6) decides when to call [`Transcript::push`],
//! [`Transcript::scroll_up`], [`Transcript::scroll_down`],
//! [`Transcript::focus_next`], [`Transcript::focus_prev`], and
//! [`Transcript::toggle_expanded`]. See `code-rules.md` R1 — the
//! transcript consumes `RenderBlock` only.

#![allow(clippy::doc_markdown)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use unicode_width::UnicodeWidthStr;
use runtime::message_stream::{
    BlockId, RenderBlock, TodoResultItem, TodoResultStatus, ToolCallId, ToolCallStatus,
    ToolPreview, ToolResultBody,
};

use super::blocks;
use super::glyphs;
use super::image_protocol::ImageProtocol;
use super::theme::Theme;

mod layout;
mod scroll;
mod tool_groups;

use layout::{block_id, separator_y};
use tool_groups::{ToolGroupState, collapsed_summary_height, collapsed_tool_detail_lines};

/// Cached `(render_version, width, preserves_layout, rendered_lines)` entry
/// stored per markdown-bearing block (`TextDelta` for assistant streams,
/// `UserMessage` for user pastes). Caret-free; the streaming caret is
/// applied at draw time so blink ticks don't invalidate the cache.
///
/// For `UserMessage` the cached lines hold the markdown body **without**
/// the amber bar prefix — the prefix is prepended at draw time by
/// [`crate::tui::blocks::draw_user_message_from_cache`].
/// Per-block render cache — block 유형별로 다른 cache key 정책을 표현한
/// enum. 같은 슬롯에 어떤 variant 든 들어갈 수 있게 만들어 `Vec<Option<_>>`
/// 캐시 슬롯 1개로 모든 block 타입을 다룬다.
///
/// 캐시 키 설계:
/// * `Text` — content 가 streaming 으로 변하므로 transcript 의 per-block
///   render version 을 key 로 쓴다. 매 프레임 전체 본문을 해시하지 않고,
///   실제 mutate 지점에서만 O(1) 로 버전을 올려 같은 길이 mutation 의 stale
///   hit 도 막는다.
/// * `Tool` / `Diff` — `push` 후 본문이 immutable (try_merge 도 invalidate
///   처리됨) 이므로 `block_id` 만으로 stable. focused/expanded 같은 외부
///   상태는 별도 key 로 분리.
///
/// 모든 variant 가 `Vec<Line<'static>>` 보관 — body lifetime 에서 풀어
/// 슬롯 재사용 시 dangling 위험 0.
#[derive(Debug)]
#[allow(dead_code)] // Diff/Reasoning variants: 캐시 정책 선설계, 호출자 후속 PR 에서 연결
enum RenderCache {
    Text {
        content_version: u64,
        width: u16,
        /// Whether the stream that produced `lines` had finished. While a
        /// turn streams (`done == false`) `lines` hold the incrementally
        /// styled output (completed blocks styled once + the open tail
        /// re-rendered each frame); normal-sized blocks run one final
        /// authoritative markdown + syntect pass once `done` flips true, while
        /// large blocks may keep the incremental cache to avoid blocking input.
        /// Part of the key so that transition re-renders even when the final
        /// delta carries no new text.
        done: bool,
        preserves: bool,
        lines: Vec<Line<'static>>,
        /// 줄별 wrap-행 prefix-sum (`len == lines.len() + 1`, 폭 = `width`).
        /// 높이 질의를 O(전체 라인 재-wrap) 대신 O(1)(`last()`)로 만들고,
        /// draw 가 viewport 에 보이는 줄 구간만 binary-search 로 잘라 그리게
        /// 한다 — 긴 답변에서 프레임 비용이 본문 길이에 비례하던 두 지점의
        /// 공통 해법. `preserves == true`(wrap 없음, 줄==행)면 빈 Vec.
        row_prefix: Vec<u32>,
        /// Streaming incremental cursor: byte length of `text` whose styled
        /// output occupies `lines[..stable_line_count]`. Those leading lines
        /// are final (their markdown blocks completed) and are never
        /// re-rendered — only the open tail after them is. Both `0` when
        /// `done` (the final pass renders everything at once). See
        /// [`crate::tui::blocks::text::streaming_incremental`].
        stable_len: usize,
        stable_line_count: usize,
        /// Resumable markdown scan cursor at `stable_len`, so the next streaming
        /// frame continues the stable-prefix scan from the last boundary instead
        /// of re-scanning the whole accumulated text (O(total)→O(suffix) — what
        /// keeps a long streamed answer from freezing the loop). `None` on the
        /// final/`done` pass, which renders everything at once.
        scan: Option<crate::tui::markdown::StreamScanState>,
    },
    Tool {
        block_id: u64,
        width: u16,
        focused: bool,
        expanded: bool,
        lines: Vec<Line<'static>>,
        /// 줄별 wrap-행 prefix-sum (`len == lines.len() + 1`, 폭 = `width`).
        /// `Text` 와 같은 정책: 높이 질의를 O(전체 라인 재-wrap) 대신 O(1)
        /// (`last()`)로, draw 가 viewport 에 보이는 줄 구간만 binary-search
        /// 로 잘라 그리게 한다. 없었을 때는 큰 tool 출력이 화면에 보이는 매
        /// 프레임마다 본문 전체를 재-wrap 했다(긴 read/bash 출력에서 "글 많으면
        /// 렉"·스크롤 렉의 한 축).
        row_prefix: Vec<u32>,
    },
    /// A *settled* collapsed tool group's summary, cached in the **leader's**
    /// slot and covering the whole span. Without it a settled summary rebuilt
    /// all member detail rows on every suffix re-measure *and* re-styled them
    /// on every visible frame. Live groups (an in-flight member) stay on the
    /// direct path so their per-tool status markers keep moving.
    ///
    /// Key: `span_len`/`err_count` re-key the entry when membership grows (a
    /// settled call/result appends to the run) or an error flips — both change
    /// without any leader-slot mutation. `span_versions` (the wrapping sum of
    /// the span members' render versions) re-keys the one remaining
    /// key-invisible mutation: an in-place member rewrite
    /// (`update_existing_tool_call` bumps the member's version), regardless of
    /// the span's collapse shape (revealed diff pairs and the Ctrl+X window
    /// leave `Normal` states mid-span, so a state walk cannot find the leader
    /// reliably — the version sum can). The live bypass in
    /// `collapsed_group_height` drops a stale entry when a chained batch
    /// re-leads the settled run as live.
    Group {
        leader_block_id: u64,
        width: u16,
        span_len: usize,
        err_count: u16,
        span_versions: u64,
        /// Rows the collapsed summary occupies (== `collapsed_summary_height`).
        height: u16,
        lines: Vec<Line<'static>>,
    },
    Diff {
        block_id: u64,
        width: u16,
        expanded: bool,
        lines: Vec<Line<'static>>,
    },
    Reasoning {
        text_hash: u64,
        width: u16,
        expanded: bool,
        lines: Vec<Line<'static>>,
    },
}

/// Estimated display height (in rows) reserved for a single block
/// widget before expansion. The transcript layer lays blocks out
/// linearly using this value.
pub const DEFAULT_BLOCK_ROWS: u16 = 3;

/// Compact copy affordance shown only for the hovered transcript block.
///
/// Use a monochrome duplicate-square glyph instead of an emoji: emoji clipboard
/// glyphs often render as double-width/color cells in terminals, while `⧉` is
/// stable enough to keep hit-testing and redraws predictable. Keep one cell of
/// padding on both sides so the mouse target is still usable.
pub(crate) const COPY_BUTTON_LABEL: &str = " ⧉ ";
pub(crate) const COPY_BUTTON_WIDTH: u16 = 3;

/// Absolute terminal rect for the block-level copy affordance currently under
/// the pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyButtonHit {
    pub block_id: BlockId,
    pub button: Rect,
}

#[cfg(test)]
const MAX_TRANSCRIPT_BLOCKS: usize = 2_048;
#[cfg(not(test))]
const MAX_TRANSCRIPT_BLOCKS: usize = 10_000;

#[cfg(test)]
const TRANSCRIPT_PRUNE_CHUNK: usize = 64;
#[cfg(not(test))]
const TRANSCRIPT_PRUNE_CHUNK: usize = 512;

fn first_visible_layout_entry(layout: &[(usize, u16, u16)], scroll: u16) -> usize {
    layout.partition_point(|&(_, block_top, height)| block_top.saturating_add(height) <= scroll)
}

/// The transcript's scroll **intent** — what the viewport wants to show,
/// independent of the concrete row offset a draw resolves it to.
///
/// This is the single-responsibility split of the old `scroll: u16` field,
/// which overloaded a `u16::MAX` "follow the tail" sentinel onto the very same
/// integer that also carried the concrete, content-clamped offset used for
/// hit-testing after a draw. [`Transcript`] now keeps this intent alongside a
/// separate `resolved_scroll` cache: `draw` resolves the intent against the
/// live content height every frame and writes the cache, and every geometry
/// read (hit-test, copy affordance, [`Transcript::is_at_bottom`]) reads the
/// cache — never the intent.
///
/// [`Self::Bottom`] follows the tail: `draw` resolves it to the maximum offset
/// each frame, so appended content stays pinned without the caller re-asserting
/// an offset. [`Self::Rows`] is an explicit offset (still clamped to content at
/// draw time). A future `Anchored` variant (pin a specific block) is deferred
/// until there is a concrete need for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollPos {
    /// Follow the tail: resolve to the maximum scroll offset on every draw.
    Bottom,
    /// An explicit row offset from the top (clamped to content at draw time).
    Rows(u16),
}

impl Default for ScrollPos {
    fn default() -> Self {
        // A fresh transcript sits at the top (offset 0), matching the historical
        // `scroll: u16 = 0` default. Following the tail is engaged explicitly by
        // the first `scroll_to_bottom` once content arrives.
        ScrollPos::Rows(0)
    }
}

/// An in-progress mouse character-selection over the transcript. Columns are
/// screen-absolute cells (the transcript never scrolls horizontally); rows
/// are *content* rows — the viewport row plus the resolved scroll at event
/// time — so the gesture survives scrolling mid-drag. A wheel notch while the
/// button is down extends the reachable range instead of invalidating the
/// anchor (the editor/browser drag-scroll idiom the old screen-pinned
/// selection could not express), and a settled highlight tracks its text when
/// the view moves afterwards. Anchored where the left button went down;
/// `head` follows the pointer while dragging. `dragged` flips only once the
/// head leaves the anchor cell, so a plain click never selects (and never
/// copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CharSelection {
    pub anchor: (u16, u16),
    pub head: (u16, u16),
    pub dragged: bool,
}

/// Scrollable transcript.
// The bool fields are independent flags — two cache-dirty bits plus two
// live-presentation gates (turn streaming, pinned agent panel on screen) — not
// a state-machine mode, so an enum would obscure rather than clarify.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Default)]
pub struct Transcript {
    /// The ordered list of blocks currently in the transcript.
    blocks: Vec<RenderBlock>,
    /// The transcript's scroll intent (follow-tail vs an explicit offset).
    /// Split from the resolved offset below so the old `u16::MAX` "follow"
    /// sentinel is a named [`ScrollPos::Bottom`] variant rather than a magic
    /// number overloaded onto the geometry value.
    scroll: ScrollPos,
    /// The resolved, content-clamped scroll offset in rows. `draw` computes it
    /// from [`Self::scroll`] each frame; every post-draw geometry read
    /// (hit-test, copy affordance, [`Self::is_at_bottom`],
    /// [`Self::content_rows_in_viewport`]) reuses it. Holds the transient
    /// `u16::MAX` sentinel between a `scroll_to_bottom` and the next draw that
    /// resolves it, so callers reading `scroll()` before a draw still see the
    /// tail.
    resolved_scroll: u16,
    /// Index of the currently focused interactable block, if any.
    focused_idx: Option<usize>,
    /// In-progress mouse character-selection (left-drag over the transcript),
    /// content-row anchored — the terminal-native selection feel (drag any
    /// run of text, release copies it) reproduced inside the mouse-capture
    /// TUI, plus wheel-extension past the viewport. `None` when nothing is
    /// selected.
    char_selection: Option<CharSelection>,
    /// Clean text under the selection, mined row-by-row from the frame buffer
    /// by the draw pass (the only place rendered cells exist) and keyed by
    /// content row. A visible row is re-mined every frame; a row scrolled out
    /// of the viewport keeps its last mined text — that persistence is what
    /// lets a wheel-extended selection copy more than one screenful. Release
    /// joins the selected row range from this store.
    char_selection_mined: BTreeMap<u16, String>,
    /// Set of block ids that are currently expanded.
    expanded: HashSet<u64>,
    /// `ToolCall` block ids of collapsed tool-group *leaders* the user revealed
    /// (mouse click, CC parity): the group's Hidden members render as normal
    /// rows until clicked again. Applied as a post-pass over every tool-group
    /// recompute (`apply_group_reveals`), so appends/live regrouping cannot
    /// silently re-collapse a user-opened group.
    revealed_groups: HashSet<u64>,
    /// Cached layout: `(block_idx, top_y, height)` tuples.
    /// Invalidated when blocks change or width changes.
    cached_layout: Vec<(usize, u16, u16)>,
    /// Width used for the cached layout. `0` means cache is invalid.
    cached_layout_width: u16,
    /// Number of blocks when the cache was last computed.
    cached_layout_block_count: usize,
    /// Whether the last block was a streaming delta (needs re-measure).
    cached_layout_dirty: bool,
    /// Lowest block index whose height/render may have changed since the last
    /// layout pass (`None` = clean). Set by every in-place mutation — tail
    /// merges, mid-list [`Self::upsert_system`], out-of-order tool-status
    /// reconciles — so [`Self::ensure_layout`] recomputes from this point
    /// instead of assuming only the tail moved. Assuming tail-only was the
    /// "streamed text stays truncated" bug: a final delta merged in the same
    /// drain batch as the next ToolCall append left the text block's stale
    /// height/render cache behind the suffix recompute start, and the draw
    /// path trusts that cache verbatim.
    layout_dirty_from: Option<usize>,
    /// Per-block rendered lines cache. See [`RenderCache`] for the per-
    /// variant key policy. `None` means cache-miss for that slot.
    rendered_cache: Vec<Option<RenderCache>>,
    /// Per-block content/render version. Incremented at the exact mutation
    /// sites that can make a content-keyed render cache stale, so cache lookup
    /// stays O(1) even for very long streamed text. Same length as `blocks`.
    render_versions: Vec<u64>,
    /// Per-block tool group collapse state. Same length as `blocks`.
    tool_groups: Vec<ToolGroupState>,
    /// `BlockId.0` → 그 도구 호출이 transcript 에 처음 추가된 시각.
    /// `tool_call.rs` 가 `status: Running` 일 때 elapsed seconds 를 그려
    /// 사용자에게 작업 진행감을 준다. Cleared by [`Self::clear`].
    tool_call_started_at: HashMap<u64, Instant>,
    /// Expanded foreground Bash row plus its last measured live-tail height.
    /// Same-height output updates repaint without invalidating layout, so an
    /// in-progress drag selection is not cancelled on every render tick.
    live_tail_layout: Option<(u64, u16)>,
    /// Transcript block index to visually mark as the active search
    /// match (a left accent bar). `None` outside [`crate::tui::app`]
    /// search mode. Set via [`Self::set_search_highlight`].
    search_highlight: Option<usize>,
    /// `BlockId.0` → reasoning 블록이 처음 등장한 시각. 스트리밍 동안
    /// live elapsed("· 1.2s")를 그려 스피너 없이 진행감을 준다.
    reasoning_started_at: HashMap<u64, Instant>,
    /// `BlockId.0` → reasoning 이 `done` 으로 전환되는 순간 동결한 경과.
    /// 완료 후에도 "· 2.7s" 가 영구히 남아 OpenCode 식 사고 시간 기록이 된다.
    /// Cleared by [`Self::clear`].
    reasoning_elapsed: HashMap<u64, Duration>,
    /// `ToolCallId.0` → live/finished agent data. The transcript renders its
    /// compact aggregate while the dedicated agent surfaces retain full rows.
    /// Side table so the boundary [`RenderBlock`] enum stays untouched. Cleared
    /// by [`Self::clear`]; updated via [`Self::set_agent_tree`].
    agent_trees: HashMap<String, blocks::tool_call::AgentTree>,
    /// Cached index of the highest-indexed `ToolCall` block whose status is
    /// `Pending` or `Running`. `None` when no active tool call exists.
    ///
    /// Replaces the per-frame O(n) reverse scan in `draw()` that previously
    /// scanned the *entire* block list every frame when no active tool calls
    /// existed (the common post-turn scroll case). The cache is updated
    /// incrementally at the exact moments tool-call status can change:
    /// `push` (new block appended or ToolCall merged), `reconcile_tool_call_status`
    /// (ToolResult arrives and flips a call to Ok/Errored), and `clear`.
    /// Between those events the `draw` path reads this in O(1).
    ///
    /// `tail_active_dirty` is set whenever the cache may be stale; the next
    /// `draw` call rescans and updates it once.
    cached_tail_active_idx: Option<usize>,
    /// When `true` the cached value is stale and `draw` must rescan once.
    tail_active_dirty: bool,
    /// Whether a model turn is currently streaming.
    ///
    /// While `true`, the live bottom todo panel (`app::render::draw_todo_panel`)
    /// owns the current turn's plan, so `ToolResult::Todos` blocks appended
    /// after the turn started are suppressed here (height 0, not drawn) to avoid
    /// rendering twice — once pinned above the input and once scrolled into the
    /// transcript. Todo history from previous turns remains visible so starting
    /// a new turn does not make old plans disappear and then pop back in.
    /// Toggled via [`Self::set_turn_active`]; the sidebar `todo` section is
    /// independent of this and always renders.
    turn_active: bool,
    /// Block index captured when the current turn begins. During the turn only
    /// `ToolResult::Todos` blocks at/after this index are hidden; earlier todo
    /// blocks are settled history and must stay visible.
    turn_start_block_idx: Option<usize>,
    /// `ToolResult::Todos` block ids superseded by an all-completed snapshot in
    /// the same turn. They stay in the transcript model for export/history, but
    /// render at height 0 so a finished plan is deleted from the visible TUI
    /// instead of leaving an old `Updated Plan` block behind.
    superseded_todo_block_ids: HashSet<u64>,
}

fn todo_items_all_completed(items: &[TodoResultItem]) -> bool {
    items
        .iter()
        .all(|item| item.status == TodoResultStatus::Completed)
}

impl Transcript {
    /// Construct an empty transcript.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remove all blocks and reset viewport state.
    pub fn clear(&mut self) {
        self.clear_entries();
        self.set_explicit_scroll(0);
        self.focused_idx = None;
        self.expanded.clear();
        self.revealed_groups.clear();
        self.invalidate_layout_cache();
        self.tool_call_started_at.clear();
        self.live_tail_layout = None;
        self.reasoning_started_at.clear();
        self.reasoning_elapsed.clear();
        self.search_highlight = None;
        self.agent_trees.clear();
        self.cached_tail_active_idx = None;
        self.tail_active_dirty = false;
        self.turn_active = false;
        self.turn_start_block_idx = None;
        self.superseded_todo_block_ids.clear();
    }

    /// Mark whether a model turn is currently streaming.
    ///
    /// Set `true` from `App::begin_turn` and `false` from `App::end_turn`. While
    /// active, a `ToolResult::Todos` block is suppressed in the transcript so the
    /// plan shows only in the live pinned panel (no double render); on the
    /// `true → false` settle, incomplete plans reappear as history while
    /// completed/superseded plans stay hidden. Only a *change* invalidates the
    /// layout cache (so the suppressed Todos block's height is re-measured); an
    /// idempotent set is a cheap no-op.
    pub fn set_turn_active(&mut self, active: bool) {
        if self.turn_active == active {
            return;
        }

        let previous_start_idx = self.turn_start_block_idx;
        self.turn_active = active;

        if active {
            // A fresh turn has no current-turn Todo blocks yet. Older Todo
            // history remains visible, so there is nothing to remeasure here.
            // Avoid dropping the whole render cache on every turn start: long
            // transcripts would otherwise restyle/re-measure all markdown and
            // tool results before the first streamed token can paint.
            self.turn_start_block_idx = Some(self.blocks.len());
            return;
        }

        self.turn_start_block_idx = None;

        // The turn settled: incomplete Todos appended after `previous_start_idx`
        // flip from height 0 (live panel owned them) to their normal transcript
        // height; completed/superseded Todos remain height 0. Mark that suffix
        // dirty and clear only those Todo render-cache slots; older history
        // caches stay hot.
        let start_idx = previous_start_idx.unwrap_or(0).min(self.blocks.len());
        let first_revealed_todo = (start_idx..self.blocks.len()).find(|&idx| {
            matches!(
                self.blocks.get(idx),
                Some(RenderBlock::ToolResult {
                    body: ToolResultBody::Todos(_),
                    ..
                })
            )
        });
        if let Some(first_idx) = first_revealed_todo {
            for idx in first_idx..self.blocks.len() {
                if matches!(
                    self.blocks.get(idx),
                    Some(RenderBlock::ToolResult {
                        body: ToolResultBody::Todos(_),
                        ..
                    })
                ) {
                    if let Some(slot) = self.rendered_cache.get_mut(idx) {
                        *slot = None;
                    }
                }
            }
            self.mark_layout_dirty_from(first_idx);
        }
    }

    /// Whether the block at `idx` is a `ToolResult::Todos` that must be hidden
    /// from the visible transcript. This is the single source of truth for both
    /// layout height (0 rows) and draw skip, so measure and paint never
    /// disagree.
    fn todos_suppressed(&self, idx: usize) -> bool {
        let Some(RenderBlock::ToolResult { id, body, .. }) = self.blocks.get(idx) else {
            return false;
        };
        let ToolResultBody::Todos(items) = body else {
            return false;
        };

        // A completed plan is not useful chat history once the HUD/store have
        // acknowledged completion; keep incomplete `Updated Plan` history, but
        // hide all-done snapshots so they do not linger in the chat window.
        todo_items_all_completed(items)
            || self.superseded_todo_block_ids.contains(&id.0)
            || self.current_turn_todo_suppressed(idx)
    }

    /// Back-compat name for sibling modules/tests: completed/superseded todo
    /// results are suppressed too, not only active-turn blocks.
    fn todos_suppressed_during_turn(&self, idx: usize) -> bool {
        self.todos_suppressed(idx)
    }

    /// Whether the block at `idx` is a current-turn `ToolResult::Todos` hidden
    /// because a turn is streaming (the live pinned panel owns the active plan).
    /// Older incomplete todo history stays visible while a new turn streams.
    fn current_turn_todo_suppressed(&self, idx: usize) -> bool {
        if !self.turn_active {
            return false;
        }
        if let Some(start_idx) = self.turn_start_block_idx {
            if idx < start_idx {
                return false;
            }
        }
        matches!(
            self.blocks.get(idx),
            Some(RenderBlock::ToolResult {
                body: ToolResultBody::Todos(_),
                ..
            })
        )
    }

    /// Install/refresh the live agent tree under the Spawn-family tool call
    /// `tool_call_id`. No-op when unchanged or when the owning block is absent;
    /// otherwise the owning block's height is re-measured from its index (the
    /// tree adds rows).
    pub fn set_agent_tree(&mut self, tool_call_id: &str, tree: blocks::tool_call::AgentTree) {
        let Some(idx) = self.blocks.iter().rposition(|block| {
            matches!(
                block,
                RenderBlock::ToolCall { tool_call_id: existing, .. } if existing.0 == tool_call_id
            )
        }) else {
            self.agent_trees.remove(tool_call_id);
            return;
        };
        if self.agent_trees.get(tool_call_id) == Some(&tree) {
            return;
        }
        self.agent_trees.insert(tool_call_id.to_string(), tree);
        self.mark_layout_dirty_from(idx);
    }

    /// The agent tree currently attached to `tool_call_id`, if any.
    #[must_use]
    pub fn agent_tree(&self, tool_call_id: &str) -> Option<&blocks::tool_call::AgentTree> {
        self.agent_trees.get(tool_call_id)
    }

    /// `true` when any ToolCall block carrying a **live** agent batch
    /// intersects the viewport, as of the last draw's resolved scroll and the
    /// given viewport height. The App gates the pinned live-agent panel on this:
    /// the compact batch summary renders under its host Spawned row, and the
    /// detailed bottom panel appears when that row scrolls off screen (or no
    /// host row exists), avoiding two simultaneous agent surfaces.
    ///
    /// Uses the previous frame's `resolved_scroll` (the panel height is
    /// reserved before this frame's transcript draw), so the panel can lag a
    /// scroll by one frame — invisible in practice at tick cadence.
    #[must_use]
    pub fn live_tree_visible(&self, viewport_h: u16) -> bool {
        if viewport_h == 0
            || self.cached_layout.is_empty()
            || !self.agent_trees.values().any(blocks::tool_call::AgentTree::is_live)
        {
            return false;
        }
        let scroll = self.resolved_scroll;
        let end = scroll.saturating_add(viewport_h);
        let start = first_visible_layout_entry(&self.cached_layout, scroll);
        for &(idx, block_top, _height) in &self.cached_layout[start..] {
            if block_top >= end {
                break;
            }
            let Some(RenderBlock::ToolCall { tool_call_id, .. }) = self.blocks.get(idx) else {
                continue;
            };
            if self
                .agent_trees
                .get(&tool_call_id.0)
                .is_some_and(blocks::tool_call::AgentTree::is_live)
            {
                return true;
            }
        }
        false
    }

    /// Drop every per-block rendered-lines cache entry while leaving the
    /// blocks (and their parallel cache slots) intact.
    ///
    /// The render cache is keyed by `(render version, width, done)` — none of
    /// which change when only the *palette* changes — so a live `/theme`
    /// switch would otherwise keep serving lines whose `Style`s bake in the
    /// previous theme's colors until each block's content or the width
    /// changes. Resetting each slot to `None` (rather than clearing the
    /// `Vec`, which would desync it from `blocks`) forces the next draw to
    /// re-render every block with the new palette. Block heights are
    /// palette-independent, so the layout cache stays valid.
    pub fn invalidate_render_cache(&mut self) {
        for slot in &mut self.rendered_cache {
            *slot = None;
        }
    }

    /// Append a block to the transcript tail.
    pub fn push(&mut self, block: RenderBlock) {
        // Reasoning block this push extended in place mid-list (see
        // `try_rejoin_reasoning`); the timing update below must look at that
        // block instead of the tail.
        let mut rejoined_idx: Option<usize> = None;
        if let Some(block) = self.try_merge_block(block) {
            match self.try_rejoin_reasoning(block) {
                Ok(idx) => rejoined_idx = Some(idx),
                Err(block) => {
                    // Phase 5 — tool call 의 시작 시각 기록 (elapsed seconds 표시용).
                    // `insert`(재스탬프)여야 한다: block id 는 턴마다 0부터 재시작하고
                    // 지난 턴의 블록이 transcript 에 남아 GC(retain)가 그 엔트리를
                    // 계속 live 로 보므로, `or_insert` 는 새 턴의 같은 id 툴에게 지난
                    // 턴의 시작 시각을 물려줘 "30초 툴에 · 9m 40s · still waiting" 류
                    // 유령 경과를 그렸다. 같은 id 의 상태 갱신은 merge 경로로 빠져
                    // 여기 오지 않으니, 이 push 는 항상 "새 툴 등장"이다.
                    if let RenderBlock::ToolCall { id, .. } = &block {
                        self.tool_call_started_at.insert(id.0, Instant::now());
                    }
                    // A new ToolCall append may create or replace the active tail call.
                    if matches!(&block, RenderBlock::ToolCall { .. }) {
                        self.tail_active_dirty = true;
                    }
                    self.push_entry(block);
                }
            }
        } else {
            // Block was merged into the last entry. Whether its render cache
            // must drop depends on the cache key: `RenderCache::Text` is keyed
            // by a per-block render version, so a text/reasoning tail merge is
            // detected as a natural cache miss *and* the preserved entry carries the
            // incrementally-styled stable prefix that
            // [`crate::tui::blocks::text::streaming_incremental`] reuses.
            // Dropping it here forced every token frame to restyle the whole
            // accumulated message from scratch (syntect included) — the frame
            // cost grew linearly with answer length and long streams stuttered.
            // Tool merges mutate fields (status) that their cache key does NOT
            // track, so those still invalidate explicitly.
            let content_keyed = matches!(
                self.blocks.last(),
                Some(RenderBlock::TextDelta { .. } | RenderBlock::Reasoning { .. })
            );
            if !content_keyed {
                if let Some(last) = self.rendered_cache.last_mut() {
                    *last = None;
                }
                // A ToolCall status merge (e.g. Pending→Running) may change
                // the active index — mark dirty so draw refreshes the cache.
                if matches!(self.blocks.last(), Some(RenderBlock::ToolCall { .. })) {
                    self.tail_active_dirty = true;
                }
            }
        }
        // Reasoning 타이밍: 이번 push 가 만진 블록을 본다 — tail merge/append 는
        // last, 미드스트림 재접합은 rejoined_idx. 재접합된 done=true 델타가
        // last(끼어든 공지)만 보던 갱신을 비껴가면 동결값이 안 박혀 정착 후에도
        // elapsed 가 계속 오른다. 첫 등장에 시작 시각을 박고, done 으로 넘어가는
        // 순간 경과를 동결한다 (이후 프레임에서도 같은 값 → "· N.Ns" 영구).
        // 동결값이 생긴 뒤에는 or_insert 가 no-op 이라 한 번만 박힌다.
        let touched_idx = rejoined_idx.unwrap_or_else(|| self.blocks.len().saturating_sub(1));
        if let Some(RenderBlock::Reasoning { id, done, .. }) = self.blocks.get(touched_idx) {
            let (rid, rdone) = (id.0, *done);
            let start = *self
                .reasoning_started_at
                .entry(rid)
                .or_insert_with(Instant::now);
            if rdone {
                self.reasoning_elapsed
                    .entry(rid)
                    .or_insert_with(|| start.elapsed());
            }
        }
        let current_todo_dirty_from = self.suppress_current_turn_todo_snapshots_before_new_one();
        let completed_todo_dirty_from = self.suppress_superseded_todos_after_completed_snapshot();
        let dirty_from = if self.enforce_block_limit() {
            0
        } else {
            self.streaming_reasoning_hidden_by_tail_prose_idx()
                .unwrap_or_else(|| self.blocks.len().saturating_sub(1))
        };
        let dirty_from = current_todo_dirty_from.map_or(dirty_from, |idx| dirty_from.min(idx));
        let dirty_from = completed_todo_dirty_from.map_or(dirty_from, |idx| dirty_from.min(idx));
        self.mark_layout_dirty_from(dirty_from);
        self.gc_timing_maps();
    }

    fn suppress_current_turn_todo_snapshots_before_new_one(&mut self) -> Option<usize> {
        if !self.turn_active {
            return None;
        }
        let current_idx = self.blocks.len().checked_sub(1)?;
        if !matches!(
            self.blocks.get(current_idx),
            Some(RenderBlock::ToolResult {
                body: ToolResultBody::Todos(_),
                ..
            })
        ) {
            return None;
        }

        let start_idx = self.turn_start_block_idx.unwrap_or(0).min(current_idx);
        let mut first_suppressed = None;
        // TodoWrite emits a full snapshot each time. While a turn is active,
        // only the latest current-turn snapshot should survive as the eventual
        // `Updated Plan` history card; otherwise several incomplete snapshots
        // hidden behind the live panel all reappear together when the turn ends.
        for idx in start_idx..current_idx {
            if let Some(RenderBlock::ToolResult {
                id,
                body: ToolResultBody::Todos(_),
                ..
            }) = self.blocks.get(idx)
            {
                self.superseded_todo_block_ids.insert(id.0);
                first_suppressed.get_or_insert(idx);
                if let Some(slot) = self.rendered_cache.get_mut(idx) {
                    *slot = None;
                }
            }
        }
        first_suppressed
    }

    fn suppress_superseded_todos_after_completed_snapshot(&mut self) -> Option<usize> {
        let completed_idx = self.blocks.len().checked_sub(1)?;
        let Some(RenderBlock::ToolResult {
            body: ToolResultBody::Todos(items),
            ..
        }) = self.blocks.get(completed_idx)
        else {
            return None;
        };
        if !todo_items_all_completed(items) {
            return None;
        }

        let start_idx = if self.turn_active {
            self.turn_start_block_idx.unwrap_or(0).min(completed_idx)
        } else {
            completed_idx
        };
        let mut first_suppressed = None;
        // Mark the *earlier* todo blocks this completed snapshot supersedes, but
        // NOT the completed snapshot itself (`..completed_idx`, exclusive): it is
        // the superseding plan, not a superseded one. Leaving it out of the set
        // lets it reappear as settled `Updated Plan · N/N done` history once the
        // turn ends (during the turn it is still hidden by
        // `current_turn_todo_suppressed`).
        for idx in start_idx..completed_idx {
            if let Some(RenderBlock::ToolResult {
                id,
                body: ToolResultBody::Todos(_),
                ..
            }) = self.blocks.get(idx)
            {
                self.superseded_todo_block_ids.insert(id.0);
                first_suppressed.get_or_insert(idx);
                if let Some(slot) = self.rendered_cache.get_mut(idx) {
                    *slot = None;
                }
            }
        }
        first_suppressed
    }

    /// The transient animated `Thinking...` row is only useful before the model
    /// has produced visible assistant-side output. Once prose or a tool block
    /// follows it in the same turn, hide the whole streaming reasoning cue (even
    /// if the provider sent partial reasoning text) so users do not see stale
    /// motion above real work. Completed reasoning remains governed by
    /// `suppress_collapsed`.
    fn is_reasoning_visually_suppressed(&self, idx: usize) -> bool {
        let Some(RenderBlock::Reasoning { id, text, done, .. }) = self.blocks.get(idx) else {
            return false;
        };
        let expanded = self.is_expanded(idx);
        // Same elapsed the layout/draw paths feed `estimate_rows`/`draw`, so the
        // suppress decision (and thus the 0-vs-1 row height) is identical at all
        // four sites — measure and paint never disagree.
        let elapsed = self.reasoning_display_elapsed(id.0);
        blocks::reasoning::suppress_collapsed(text, *done, expanded, elapsed)
            || (!*done && self.reasoning_followed_by_assistant_output(idx))
    }

    fn reasoning_followed_by_assistant_output(&self, idx: usize) -> bool {
        self.blocks
            .iter()
            .enumerate()
            .skip(idx.saturating_add(1))
            .take_while(|(_, block)| !matches!(block, RenderBlock::UserMessage { .. }))
            .any(|(block_idx, block)| match block {
                RenderBlock::TextDelta { text, .. } => !text.is_empty(),
                RenderBlock::ToolCall { .. } => true,
                RenderBlock::ToolResult { .. } => !self.todos_suppressed(block_idx),
                _ => false,
            })
    }

    /// Whether an assistant prose block carries no visible body and must be
    /// hidden entirely (height 0, not drawn, no surrounding gap).
    ///
    /// A `TextDelta` whose accumulated text is empty (or whitespace-only) once
    /// it has settled (`done == true`) is a phantom block: the provider closed
    /// a text part with no content — common when a model ends a turn right after
    /// a tool call, or when a reveal block is opened and then settled empty. The
    /// transcript would otherwise reserve the `Zo` author header rows for it
    /// and paint a bare `✓ Zo · done` line above nothing (the reported
    /// empty-block bug). Streaming (`done == false`) empties are NOT suppressed:
    /// a freshly-opened block legitimately has no text yet for a frame or two,
    /// and the live caret/header is the cue that an answer is arriving.
    ///
    /// This is the single source of truth for both the layout height (0 rows)
    /// and the draw skip, mirroring `todos_suppressed` and
    /// `is_reasoning_visually_suppressed`, so measure and paint never disagree.
    fn is_empty_prose_suppressed(&self, idx: usize) -> bool {
        matches!(
            self.blocks.get(idx),
            Some(RenderBlock::TextDelta { text, done: true, .. }) if text.trim().is_empty()
        )
    }

    fn is_turn_boundary_visually(&self, idx: usize) -> bool {
        if !matches!(self.blocks.get(idx), Some(RenderBlock::UserMessage { .. })) {
            return false;
        }
        self.previous_visible_block(idx)
            .is_some_and(|previous| !matches!(previous, RenderBlock::UserMessage { .. }))
    }

    /// 1-based turn ordinal for the user message at `idx` — the count of
    /// `UserMessage` blocks up to and including it. Cheap (only computed for the
    /// few turn-boundary separators actually on screen).
    fn turn_ordinal(&self, idx: usize) -> usize {
        let end = (idx + 1).min(self.blocks.len());
        self.blocks[..end]
            .iter()
            .filter(|block| matches!(block, RenderBlock::UserMessage { .. }))
            .count()
    }

    fn is_response_boundary_visually(&self, idx: usize) -> bool {
        if !self.blocks.get(idx).is_some_and(is_response_block) {
            return false;
        }
        self.previous_visible_block(idx)
            .is_some_and(|previous| matches!(previous, RenderBlock::UserMessage { .. }))
    }

    fn previous_visible_block(&self, idx: usize) -> Option<&RenderBlock> {
        (0..idx).rev().find_map(|candidate| {
            if self.is_reasoning_visually_suppressed(candidate)
                || self.is_empty_prose_suppressed(candidate)
                || matches!(
                    self.tool_groups.get(candidate),
                    Some(ToolGroupState::Hidden)
                )
            {
                None
            } else {
                self.blocks.get(candidate)
            }
        })
    }

    fn streaming_reasoning_hidden_by_tail_prose_idx(&self) -> Option<usize> {
        let Some(RenderBlock::TextDelta { text, .. }) = self.blocks.last() else {
            return None;
        };
        if text.is_empty() {
            return None;
        }
        let tail_idx = self.blocks.len().saturating_sub(1);
        for idx in (0..tail_idx).rev() {
            match &self.blocks[idx] {
                RenderBlock::UserMessage { .. } => return None,
                RenderBlock::TextDelta { text, .. } if !text.is_empty() => return None,
                RenderBlock::Reasoning { done: false, .. } => {
                    return Some(idx);
                }
                _ => {}
            }
        }
        None
    }

    /// Record that block `idx` (and therefore every later y-offset) needs a
    /// layout re-measure on the next [`Self::ensure_layout`] pass.
    fn mark_layout_dirty_from(&mut self, idx: usize) {
        self.cached_layout_dirty = true;
        self.layout_dirty_from = Some(self.layout_dirty_from.map_or(idx, |cur| cur.min(idx)));
    }

    /// Toggle the verbose "show every tool row" transcript view (Ctrl+X,
    /// Claude Code's ctrl+o parity). Returns the new state (`true` =
    /// grouping disabled, every tool call/result rendered individually).
    /// Flips the process-global classification gate, then invalidates the
    /// whole layout AND every per-block render cache — collapse summaries are
    /// baked into cached lines, so a version bump is required or a cache hit
    /// would repaint the old grouped view at the new heights.
    pub(crate) fn toggle_tool_groups_disabled(&mut self) -> bool {
        let next = !tool_groups::tool_groups_disabled();
        tool_groups::set_tool_groups_disabled(next);
        for idx in 0..self.blocks.len() {
            self.bump_render_version(idx);
        }
        self.mark_layout_dirty_from(0);
        next
    }

    // --- parallel per-block tables (P9 Step 3-lite) -------------------------
    // `blocks`, `render_versions`, and `tool_groups` move in lockstep, and
    // `rendered_cache` never exceeds them (it may lag and is grown lazily by
    // the layout pass). Every structural mutation funnels through these four
    // helpers so the tables cannot drift apart — the desync risk lived
    // entirely at these clusters. The full single-Vec BlockEntry fold was
    // measured at ~217 read sites, ~12 parallel-slice fn signatures in
    // tool_groups, and 3 simultaneous split borrows (`&self.blocks` +
    // `&mut self.tool_groups`), and deliberately deferred as a big-bang.

    /// Append one block's slots to every parallel table (the single push point).
    fn push_entry(&mut self, block: RenderBlock) {
        self.blocks.push(block);
        self.rendered_cache.push(None);
        self.render_versions.push(0);
        self.tool_groups.push(ToolGroupState::Normal);
        self.debug_assert_entries_aligned();
    }

    /// Clear every parallel table together.
    fn clear_entries(&mut self) {
        self.blocks.clear();
        self.rendered_cache.clear();
        self.render_versions.clear();
        self.tool_groups.clear();
        self.clear_char_selection();
        self.debug_assert_entries_aligned();
    }

    /// Front-drain `prune_count` slots from every parallel table together.
    fn drain_entries_front(&mut self, prune_count: usize) {
        self.blocks.drain(0..prune_count);
        self.rendered_cache
            .drain(0..prune_count.min(self.rendered_cache.len()));
        self.render_versions
            .drain(0..prune_count.min(self.render_versions.len()));
        self.tool_groups
            .drain(0..prune_count.min(self.tool_groups.len()));
        // Every retained row shifts up: a content-row-anchored char-selection
        // (and its mined rows) would wash and copy the wrong text.
        self.clear_char_selection();
        self.debug_assert_entries_aligned();
    }

    /// Remove the sibling-table slots for a block the caller already removed
    /// from `blocks` (the caller keeps the removed block for its own
    /// bookkeeping, e.g. dropping its timing entries).
    fn remove_entry_slots(&mut self, idx: usize) {
        // Mid-list removal shifts every later row; see `drain_entries_front`.
        self.clear_char_selection();
        if idx < self.rendered_cache.len() {
            self.rendered_cache.remove(idx);
        }
        if idx < self.render_versions.len() {
            self.render_versions.remove(idx);
        }
        if idx < self.tool_groups.len() {
            self.tool_groups.remove(idx);
        }
        self.debug_assert_entries_aligned();
    }

    /// The lockstep invariant the helpers above preserve.
    fn debug_assert_entries_aligned(&self) {
        debug_assert_eq!(self.blocks.len(), self.render_versions.len());
        debug_assert_eq!(self.blocks.len(), self.tool_groups.len());
        debug_assert!(self.rendered_cache.len() <= self.blocks.len());
    }

    fn bump_render_version(&mut self, idx: usize) {
        if let Some(version) = self.render_versions.get_mut(idx) {
            *version = version.wrapping_add(1);
        }
    }

    pub(super) fn render_version(&self, idx: usize) -> u64 {
        self.render_versions.get(idx).copied().unwrap_or(0)
    }

    /// Drop timing entries whose blocks are no longer present.
    ///
    /// `tool_call_started_at` / `reasoning_started_at` / `reasoning_elapsed`
    /// are keyed by `BlockId` and only ever inserted into during `push`. In a
    /// long streaming session that grows without bound — a slow memory leak
    /// that tracks the lifetime block count, not the live transcript. Blocks
    /// themselves are the transcript's real content (kept by design), but these
    /// side maps can be pruned to the ids still on screen. The scan is O(blocks)
    /// so it only runs once a map has grown well past the block count (amortized
    /// O(1) per push), and never during the hot per-token merge path.
    fn gc_timing_maps(&mut self) {
        // Headroom before a pruning scan: keep up to `slack` stale entries plus
        // a small constant so short sessions never scan, and per-token merges
        // (which never add ids) stay free.
        const SLACK_FACTOR: usize = 2;
        const SLACK_FLOOR: usize = 64;

        let tracked = self.tool_call_started_at.len()
            + self.reasoning_started_at.len()
            + self.reasoning_elapsed.len();
        let budget = self.blocks.len() * SLACK_FACTOR + SLACK_FLOOR;
        if tracked <= budget {
            return;
        }

        self.prune_side_tables_to_live_blocks();
    }

    fn enforce_block_limit(&mut self) -> bool {
        let prune_trigger = MAX_TRANSCRIPT_BLOCKS.saturating_add(TRANSCRIPT_PRUNE_CHUNK);
        if self.blocks.len() <= prune_trigger {
            return false;
        }

        // Front-draining a Vec shifts every retained block/cache slot. Keep a
        // small headroom above the target cap and prune back to the target in
        // one batch so long sessions pay that shift once per chunk, not once
        // per appended block after crossing the cap.
        let prune_count = self.blocks.len().saturating_sub(MAX_TRANSCRIPT_BLOCKS);
        self.drain_entries_front(prune_count);
        // Keep the viewport visually anchored across the front-drain by shifting
        // an explicit offset up by the pruned height. `Bottom` intent needs no
        // adjustment — it keeps following the now-shorter tail and the next draw
        // resolves it to the new max.
        if let ScrollPos::Rows(offset) = self.scroll {
            let pruned_rows = u16::try_from(
                prune_count
                    .saturating_mul(usize::from(DEFAULT_BLOCK_ROWS))
                    .min(usize::from(u16::MAX)),
            )
            .unwrap_or(u16::MAX);
            self.set_explicit_scroll(offset.saturating_sub(pruned_rows));
        }
        self.focused_idx = self
            .focused_idx
            .and_then(|idx| idx.checked_sub(prune_count));
        self.search_highlight = self
            .search_highlight
            .and_then(|idx| idx.checked_sub(prune_count));
        self.turn_start_block_idx = self.turn_start_block_idx.map(|idx| {
            // If the turn boundary itself was pruned, every retained block
            // belongs to the active turn relative to that lost boundary.
            idx.saturating_sub(prune_count)
        });
        self.invalidate_layout_cache();
        self.cached_tail_active_idx = None;
        self.tail_active_dirty = true;
        self.prune_side_tables_to_live_blocks();
        true
    }

    fn prune_side_tables_to_live_blocks(&mut self) {
        let live: HashSet<u64> = self.blocks.iter().map(|block| block_id(block).0).collect();
        self.expanded.retain(|id| live.contains(id));
        self.tool_call_started_at.retain(|id, _| live.contains(id));
        self.reasoning_started_at.retain(|id, _| live.contains(id));
        self.reasoning_elapsed.retain(|id, _| live.contains(id));
        self.superseded_todo_block_ids
            .retain(|id| live.contains(id));

        let live_tool_calls: HashSet<&str> = self
            .blocks
            .iter()
            .filter_map(|block| match block {
                RenderBlock::ToolCall { tool_call_id, .. } => Some(tool_call_id.0.as_str()),
                _ => None,
            })
            .collect();
        self.agent_trees
            .retain(|tool_call_id, _| live_tool_calls.contains(tool_call_id.as_str()));
    }

    /// Return the index of the most recent `Pending | Running` ToolCall, or
    /// `None` when no active tool call exists in the transcript.
    ///
    /// O(1) in the steady state (idle / between tool events). O(n) only at
    /// tool-event boundaries (push/merge of a ToolCall, or a status reconcile)
    /// when `tail_active_dirty` is `true` — the cache is refreshed once and
    /// then remains valid until the next mutation.
    ///
    /// This replaces the per-frame `iter().rev().find_map(...)` scan that was
    /// O(n) for every draw call when no active tool existed — the common
    /// post-turn scroll case where the user sees lag growing with transcript
    /// length.
    fn tail_active_idx(&mut self) -> Option<usize> {
        if self.tail_active_dirty {
            self.cached_tail_active_idx =
                self.blocks.iter().enumerate().rev().find_map(|(i, b)| {
                    matches!(
                        b,
                        RenderBlock::ToolCall {
                            status: ToolCallStatus::Pending | ToolCallStatus::Running,
                            ..
                        }
                    )
                    .then_some(i)
                });
            self.tail_active_dirty = false;
        }
        self.cached_tail_active_idx
    }

    /// reasoning 블록(`id`)의 화면 표기용 경과 — 완료됐으면 동결값, 아니면
    /// 시작 시각에서 계산한 live 경과. 측정 기록이 없으면 `None`.
    fn reasoning_display_elapsed(&self, id: u64) -> Option<Duration> {
        self.reasoning_elapsed
            .get(&id)
            .copied()
            .or_else(|| self.reasoning_started_at.get(&id).map(Instant::elapsed))
    }

    /// Number of blocks currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Replace a synthetic system status block in place, or append it when this
    /// is the first update. Used for host-driven live progress surfaces that do
    /// not stream through normal model deltas.
    pub fn upsert_system(
        &mut self,
        id: BlockId,
        level: runtime::message_stream::SystemLevel,
        text: String,
    ) {
        let found = self.blocks.iter().rposition(|block| {
            matches!(block, RenderBlock::System { id: existing_id, .. } if *existing_id == id)
        });
        if let Some(idx) = found {
            if let Some(RenderBlock::System {
                level: existing_level,
                text: existing_text,
                ..
            }) = self.blocks.get_mut(idx)
            {
                *existing_level = level;
                *existing_text = text;
            }
            if let Some(cache) = self.rendered_cache.get_mut(idx) {
                *cache = None;
            }
            // Mid-list replacement: the new text may wrap to a different
            // height, so the layout must re-measure from here, not just the
            // tail (live fan-out progress grows over time).
            self.mark_layout_dirty_from(idx);
            return;
        }
        self.push(RenderBlock::System { id, level, text });
    }

    /// Number of per-block rendered cache slots.
    ///
    /// Exposed for integration tests that verify transcript block/cache
    /// invariants after compaction or session reseeding.
    #[must_use]
    pub fn rendered_cache_len(&self) -> usize {
        self.rendered_cache.len()
    }

    /// `true` if the transcript is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Read-only access to the live block list. Primarily for tests and
    /// host-side introspection (e.g. asserting a resumed transcript rebuilt
    /// rich `ToolCall`/`ToolResult` blocks rather than flat `System` notices).
    #[must_use]
    pub fn blocks(&self) -> &[RenderBlock] {
        &self.blocks
    }

    /// Height of settled transcript content rendered at `width`.
    ///
    /// This excludes the viewport-only two-row breathing pad. Inline mode uses
    /// it to size an off-screen render before copying the styled cells into
    /// native terminal scrollback.
    pub(crate) fn scrollback_height(
        &mut self,
        width: u16,
        theme: &Theme,
        image_protocol: ImageProtocol,
    ) -> u16 {
        if width == 0 || self.blocks.is_empty() {
            return 0;
        }
        self.ensure_layout(width, theme, image_protocol);
        self.cached_layout
            .last()
            .map_or(0, |(_, top, height)| top.saturating_add(*height))
    }

    /// Rows (within a `viewport_h`-tall viewport) the transcript content
    /// actually occupies, given the current scroll offset — i.e. the relative
    /// row just past the last rendered block, excluding the bottom
    /// breathing-room pad. Returns `viewport_h` when the content fills or
    /// overflows the viewport (scrolled, or simply long), so a bottom-anchored
    /// overlay can sit flush under short content yet stay pinned to the input
    /// edge once the log is long enough to reach it.
    ///
    /// Must be read *after* a draw pass: it reuses the clamped
    /// `self.resolved_scroll` and the layout cache that `draw` refreshed for the
    /// current width.
    #[must_use]
    pub fn content_rows_in_viewport(&self, viewport_h: u16) -> u16 {
        if viewport_h == 0 {
            return 0;
        }
        // `content_total` adds a 2-row bottom pad; the real content bottom is
        // the last block's end, so drop that pad before measuring.
        let content_bottom = self.content_total().saturating_sub(2);
        let visible = content_bottom.saturating_sub(self.resolved_scroll);
        visible.min(viewport_h)
    }

    /// Draw the transcript into `area`.
    #[allow(clippy::too_many_lines)] // cohesive transcript frame render
    pub fn draw(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        theme: &Theme,
        tick: u64,
        image_protocol: ImageProtocol,
    ) {
        self.draw_with_hover(frame, area, theme, tick, image_protocol, None);
    }

    #[allow(clippy::too_many_lines)]
    pub fn draw_with_hover(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        theme: &Theme,
        tick: u64,
        image_protocol: ImageProtocol,
        hovered_copy_block: Option<BlockId>,
    ) {
        if area.height == 0 || area.width == 0 {
            return;
        }

        // Expanded live tails can gain/lose rows without a transcript event.
        // Same-height repaints need no layout invalidation (and therefore keep
        // drag selection intact); only a row-count transition remeasures.
        let live_tail = self.blocks.iter().enumerate().find_map(|(idx, block)| match block {
            RenderBlock::ToolCall {
                id,
                tool_call_id,
                name,
                status: ToolCallStatus::Running,
                ..
            } if self.expanded.contains(&id.0) && blocks::tool_call::is_bash(name) => {
                Some((idx, id.0, tool_call_id.0.as_str()))
            }
            _ => None,
        });
        if let Some((idx, block_id, tool_call_id)) = live_tail {
            let rows = blocks::tool_call::live_tail_row_count(tool_call_id);
            if self.live_tail_layout != Some((block_id, rows)) {
                self.live_tail_layout = Some((block_id, rows));
                self.mark_layout_dirty_from(idx);
            }
        } else {
            self.live_tail_layout = None;
        }

        // Reset this widget's own region before repainting. This prevents
        // stale cells when the transcript shrinks, reflows, or is overpainted
        // after a previous frame. Ratatui still diffs the final frame buffer,
        // so this does not force terminal I/O for cells that end unchanged.
        frame.render_widget(Clear, area);

        let viewport_h = area.height;

        // Index of the most recent in-flight tool call (Pending or
        // Running). Computed once per frame so each draw_block call can
        // cheaply decide whether to attach the "active" wave indicator.
        // O(1): the result is cached and only refreshed when a ToolCall
        // push/merge or status reconcile marks `tail_active_dirty`. Between
        // those events — including all idle scroll frames after a turn ends —
        // the cached value is returned directly. Previously this was an
        // unconditional O(n) reverse scan every frame, which caused lag that
        // scaled with transcript length when no active tool calls existed.
        let tail_active_idx = self.tail_active_idx();

        // Determine content width. If the previous frame needed a
        // scrollbar (cached width < area width), skip the full-width
        // probe entirely — avoids a redundant O(n) layout pass.
        let prev_needed_scrollbar =
            !self.cached_layout.is_empty() && self.cached_layout_width < area.width;
        let mut content_width = if prev_needed_scrollbar && area.width > 1 {
            area.width.saturating_sub(1)
        } else {
            area.width
        };
        self.ensure_layout(content_width, theme, image_protocol);
        // content_total() includes the bottom breathing-room pad, and is the
        // same value clamp_scroll_to_content / is_at_bottom use — keeping the
        // max scroll offset consistent across all three.
        let mut content_total = self.content_total();

        if content_total > viewport_h && content_width > 1 && !prev_needed_scrollbar {
            content_width = area.width.saturating_sub(1);
            self.ensure_layout(content_width, theme, image_protocol);
            content_total = self.content_total();
        }

        // Resolve the scroll intent against the live content height. `Bottom`
        // follows the tail (max offset); an explicit `Rows` offset is clamped
        // into range. `Bottom` is left durable so following persists across
        // frames without the caller re-asserting it — the one intended
        // behavior change from the old `u16::MAX` sentinel, which every draw
        // collapsed back into the geometry value. An explicit offset is instead
        // normalized to its clamped value (matching the old write-back), so a
        // stale over-scroll can never linger in the stored intent and the
        // `Rows` path stays byte-identical to the pre-split field.
        let max_scroll = content_total.saturating_sub(viewport_h.min(content_total));
        let scroll = match self.scroll {
            ScrollPos::Bottom => max_scroll,
            ScrollPos::Rows(offset) => offset.min(max_scroll),
        };
        self.resolved_scroll = scroll;
        if matches!(self.scroll, ScrollPos::Rows(_)) {
            self.scroll = ScrollPos::Rows(scroll);
        }

        // Screen rect (y, height) of the active search match, captured
        // during the draw pass and overlaid with a marker afterwards.
        let mut highlight_hit: Option<(u16, u16)> = None;
        let block_count = self.blocks.len();
        let visible_start = first_visible_layout_entry(&self.cached_layout, scroll);
        let visible_end = visible_start
            + self.cached_layout[visible_start..].partition_point(|&(_, block_top, _)| {
                block_top < scroll.saturating_add(viewport_h)
            });
        for layout_idx in visible_start..visible_end {
            let (idx, block_top, height) = self.cached_layout[layout_idx];
            // Guard against stale layout indices referencing removed blocks.
            if idx >= block_count {
                continue;
            }

            // Boundary separator: draw a thin rule in the gap before a new user
            // turn, and also after a user prompt when the first assistant-side
            // output is a tool/reasoning/system block. Without the latter, a
            // delayed tool call can appear glued to the user's own text.
            let is_turn = self.is_turn_boundary_visually(idx);
            if is_turn || self.is_response_boundary_visually(idx) {
                // Label only a real new-turn boundary with its ordinal; a mid-turn
                // response boundary keeps the plain rule.
                let turn_number = is_turn.then(|| self.turn_ordinal(idx));
                if let Some(sep_y) = separator_y(&self.cached_layout, layout_idx) {
                    if sep_y >= scroll {
                        let rel_sep = sep_y.saturating_sub(scroll);
                        let draw_sep_y = area.y.saturating_add(rel_sep);
                        if draw_sep_y < area.y.saturating_add(viewport_h) {
                            draw_turn_separator(
                                frame,
                                Rect::new(area.x, draw_sep_y, content_width, 1),
                                theme,
                                turn_number,
                            );
                        }
                    }
                }
            }

            // Tool group: skip hidden blocks, draw the multi-row group for
            // leaders. Live and settled groups share one renderer
            // (`collapsed_tool_detail_lines`): live rows animate per-tool
            // status markers, settled rows read as calm history.
            if let Some(group_state) = self.tool_groups.get(idx) {
                match group_state {
                    ToolGroupState::Hidden => continue,
                    ToolGroupState::Summary { err_count, .. } => {
                        let err_count = *err_count;
                        // Settled groups reuse the summary the layout pass built
                        // and keyed (`collapsed_group_height`); the id + width
                        // check suffices here because `ensure_layout` above
                        // already re-keyed the entry against the current
                        // span/err_count, and every member mutation marks the
                        // layout dirty. A miss (live group, or no measure yet)
                        // styles per frame as before — live rows keep their
                        // per-tool status markers moving.
                        let cached: Option<(u16, &Vec<Line<'static>>)> =
                            match self.rendered_cache.get(idx) {
                                Some(Some(RenderCache::Group {
                                    leader_block_id,
                                    width,
                                    height,
                                    lines,
                                    ..
                                })) if *leader_block_id
                                    == layout::block_id(&self.blocks[idx]).0
                                    && *width == content_width =>
                                {
                                    Some((*height, lines))
                                }
                                _ => None,
                            };
                        let total_lines = cached.as_ref().map_or_else(
                            || collapsed_summary_height(&self.blocks, &self.tool_groups, idx),
                            |(height, _)| *height,
                        );
                        // Scroll-aware like a normal block: a group can be several
                        // rows, so honor `block_scroll` and clip to the viewport.
                        let block_scroll = scroll.saturating_sub(block_top);
                        let visible_height = total_lines.saturating_sub(block_scroll);
                        let rel_top = block_top.saturating_sub(scroll);
                        let draw_y = area.y.saturating_add(rel_top);
                        let remaining =
                            area.y.saturating_add(viewport_h).saturating_sub(draw_y);
                        let draw_h = remaining.min(visible_height);
                        if draw_h > 0 {
                            let rect = Rect::new(area.x, draw_y, content_width, draw_h);
                            // One `verb  target` line per tool; live rows carry
                            // their own spinner/✓/× marker per tool.
                            let lines = match cached {
                                // 캐시 히트 — String 복제 없이 얕게 빌려 그린다.
                                Some((_, lines)) => blocks::borrow_lines(lines),
                                None => collapsed_tool_detail_lines(
                                    &self.blocks,
                                    &self.tool_groups,
                                    idx,
                                    err_count,
                                    theme,
                                    rect.width,
                                ),
                            };
                            frame.render_widget(
                                Paragraph::new(lines).scroll((block_scroll, 0)),
                                rect,
                            );
                        }
                        continue;
                    }
                    ToolGroupState::Normal => {}
                }
            }

            let block_scroll = scroll.saturating_sub(block_top);
            let visible_height = height.saturating_sub(block_scroll);

            let rel_top = block_top.saturating_sub(scroll);
            let draw_y = area.y.saturating_add(rel_top);
            let remaining = area.y.saturating_add(viewport_h).saturating_sub(draw_y);
            let draw_h = remaining.min(visible_height);

            if draw_h == 0 {
                continue;
            }

            let rect = Rect::new(area.x, draw_y, content_width, draw_h);
            if self.search_highlight == Some(idx) {
                highlight_hit = Some((draw_y, draw_h));
            }

            // Codex-style transcript: blocks own their first column and tool
            // relationships are expressed by event/result text, not a global
            // colored rail.
            let ctx = blocks::BlockDrawCtx {
                theme,
                focused: self.focused_idx == Some(idx),
                expanded: self.expanded.contains(&block_id(&self.blocks[idx]).0),
                tick,
                scroll_offset: block_scroll,
                image_protocol,
                is_tail_active: tail_active_idx == Some(idx),
            };
            self.draw_visible_block(frame, rect, idx, &ctx);
        }

        // Mouse char-selection: the gesture lives in content rows, so map the
        // visible slice back onto screen rows, mine each row's clean text from
        // the freshly painted buffer into the content-row-keyed store (rows
        // scrolled offscreen keep their last mined text — that is what lets a
        // wheel-extended drag copy more than one screenful; release joins the
        // store), then wash the visible cells terminal-selection style.
        // Patching only `bg` keeps every glyph and foreground color intact; on
        // neutral palettes (no blend) fall back to a reversed wash so the
        // selection is still visible. Clipped to the content column, so the
        // scrollbar gutter is never highlighted nor copied — the exact
        // artifact native Shift+drag has.
        if let Some(selection) = self.char_selection.filter(|sel| sel.dragged) {
            let clip = Rect::new(area.x, area.y, content_width, viewport_h);
            let selection_style = theme.selection_bg().map_or_else(
                || Style::new().add_modifier(Modifier::REVERSED),
                |bg| Style::new().bg(bg),
            );
            let buffer = frame.buffer_mut();
            for (content_row, col_start, col_end) in
                char_selection_rows(selection.anchor, selection.head, clip, scroll)
            {
                // In-band by construction (`char_selection_rows` clips to the
                // visible content rows), so the subtraction cannot wrap.
                let row = area.y + (content_row - scroll);
                let text = buffer_row_text(buffer, row, col_start, col_end);
                self.char_selection_mined.insert(content_row, text);
                buffer.set_style(
                    Rect::new(col_start, row, col_end - col_start + 1, 1),
                    selection_style,
                );
            }
        }

        // Search-match marker: a thin accent bar on the left edge of the
        // active match's rows, drawn after block content so it sits on top.
        if let Some((hy, hh)) = highlight_hit {
            let accent = Style::new()
                .fg(theme.palette.accent)
                .add_modifier(Modifier::BOLD);
            let bar: Vec<Line> = (0..hh)
                .map(|_| Line::from(Span::styled("▎", accent)))
                .collect();
            frame.render_widget(Paragraph::new(bar), Rect::new(area.x, hy, 1, hh));
        }

        if content_total > viewport_h {
            draw_scroll_indicator(frame, area, scroll, content_total, viewport_h, theme);
        }

        if let Some(block_id) = hovered_copy_block {
            if let Some(hit) = self.copy_button_for_block(block_id, area, theme, image_protocol) {
                draw_copy_button(frame, hit.button, theme);
            }
        }
    }

    /// Draw a single visible transcript block, choosing the cached fast path
    /// (TextDelta / UserMessage / ToolResult) or the per-type renderer
    /// (ToolCall / Reasoning / TextDelta / generic). Extracted from
    /// `draw_with_hover`'s visible loop; reuses the loop's [`blocks::BlockDrawCtx`].
    #[allow(clippy::too_many_lines)]
    fn draw_visible_block(
        &self,
        frame: &mut Frame<'_>,
        widget_rect: Rect,
        idx: usize,
        ctx: &blocks::BlockDrawCtx,
    ) {
        // Hot path: reuse cached rendered lines to skip per-frame
        // pulldown-cmark + syntect re-parse. Both assistant TextDelta
        // and user UserMessage blocks share this cache because both
        // go through the same `markdown::rendered_lines_for_width`.
        // ToolResult 도 같은 cache 슬롯의 `RenderCache::Tool` variant
        // 로 처리해 매 frame sanitize/syntect 호출을 0 으로 만든다.
        match &self.blocks[idx] {
            RenderBlock::TextDelta { .. } => {
                // A settled empty prose block is a phantom (height 0): draw
                // nothing even on a stale cache, so an author bullet is never
                // painted over an answer that has no body. Mirrors the
                // height-0 suppression in `cached_block_height`.
                if self.is_empty_prose_suppressed(idx) {
                    return;
                }
                let prose_style = assistant_prose_style(&self.blocks, idx);
                if let Some(Some(RenderCache::Text {
                    preserves,
                    lines: cached_lines,
                    row_prefix,
                    ..
                })) = self.rendered_cache.get(idx)
                {
                    blocks::text::draw_cached(
                        frame,
                        widget_rect,
                        cached_lines,
                        row_prefix,
                        *preserves,
                        ctx.theme,
                        ctx.scroll_offset,
                        prose_style,
                    );
                    return;
                }
            }
            RenderBlock::UserMessage { .. } => {
                if let Some(Some(RenderCache::Text {
                    lines: cached_lines,
                    row_prefix,
                    preserves,
                    ..
                })) = self.rendered_cache.get(idx)
                {
                    blocks::draw_user_message_from_cache(
                        frame,
                        widget_rect,
                        cached_lines,
                        row_prefix,
                        *preserves,
                        ctx.theme,
                        ctx.scroll_offset,
                    );
                    return;
                }
            }
            RenderBlock::ToolResult { .. } => {
                // While a turn streams, a Todos result is suppressed (height 0,
                // owned by the live panel); draw nothing even on a stale cache.
                if self.todos_suppressed(idx) {
                    return;
                }
                if let Some(Some(RenderCache::Tool {
                    lines: cached_lines,
                    row_prefix,
                    ..
                })) = self.rendered_cache.get(idx)
                {
                    // Window to the visible rows like the text path, then wrap
                    // only that slice. Re-wrapping the whole (possibly large) tool
                    // body every frame just to apply `scroll` made a visible tool
                    // result's draw cost scale with its full length — the "big
                    // output on screen / scroll lags" axis. `visible_line_window`
                    // is proven pixel-identical to the full scroll by
                    // `visible_line_window_paints_identically_to_full_scroll`.
                    let (window, line_scroll, _) = blocks::visible_line_window(
                        cached_lines,
                        row_prefix,
                        ctx.scroll_offset,
                        widget_rect.height,
                    );
                    // 캐시 히트 — String 복제 없이 얕게 빌려 그린다.
                    let para = ratatui::widgets::Paragraph::new(blocks::borrow_lines(window))
                        .wrap(ratatui::widgets::Wrap { trim: false })
                        .scroll((line_scroll, 0));
                    frame.render_widget(para, widget_rect);
                    return;
                }
            }
            _ => {}
        }

        // Phase 5 — ToolCall 은 transcript 가 보관하는 시작 시각으로
        // elapsed 를 계산해서 직접 그린다. dispatch fallback 으로 가면
        // elapsed=None 이 되어 작업 진행감이 안 보임.
        if let RenderBlock::ToolCall {
            id,
            tool_call_id,
            name,
            summary,
            preview,
            status,
        } = &self.blocks[idx]
        {
            let elapsed = self.tool_call_started_at.get(&id.0).map(Instant::elapsed);
            blocks::tool_call::draw(
                frame,
                widget_rect,
                &tool_call_id.0,
                name,
                summary,
                preview,
                *status,
                ctx.theme,
                ctx.tick,
                ctx.scroll_offset,
                ctx.is_tail_active,
                elapsed,
                self.agent_trees.get(&tool_call_id.0),
                ctx.expanded,
            );
            return;
        }

        // Reasoning 은 transcript 가 보관하는 시작/동결 시각으로 elapsed 를
        // 계산해 직접 그린다 — "· N.Ns" 사고 시간을 완료 후에도 유지한다.
        if let RenderBlock::Reasoning { id, text, done, .. } = &self.blocks[idx] {
            if self.is_reasoning_visually_suppressed(idx) {
                return;
            }
            let elapsed = self.reasoning_display_elapsed(id.0);
            blocks::reasoning::draw(
                frame,
                widget_rect,
                text,
                *done,
                ctx.theme,
                ctx.focused,
                ctx.expanded,
                ctx.tick,
                ctx.scroll_offset,
                elapsed,
                id.0,
            );
            return;
        }

        if let RenderBlock::TextDelta { text, done, .. } = &self.blocks[idx] {
            blocks::text::draw_with_mark(
                frame,
                widget_rect,
                text,
                *done,
                ctx.theme,
                ctx.tick,
                ctx.scroll_offset,
                assistant_prose_style(&self.blocks, idx),
            );
            return;
        }

        blocks::draw_block(frame, widget_rect, &self.blocks[idx], ctx);
    }
}

fn draw_copy_button(frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let style = Style::new()
        .fg(theme.palette.accent)
        .add_modifier(Modifier::BOLD | Modifier::REVERSED);
    let line = Line::from(Span::styled(COPY_BUTTON_LABEL, style));
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(line), area);
}

impl Transcript {
    /// Re-join a streaming reasoning delta whose tail merge was broken by a
    /// mid-stream append: find the current turn's same-id `Reasoning` block
    /// and merge in place, returning its index. `Err` hands the block back
    /// for the normal append path.
    ///
    /// [`Self::try_merge_block`] only merges into the *tail*, so any block
    /// appended between two deltas of one segment — e.g. the clipboard-copy
    /// notice a mouse copy pushes mid-stream — orphaned the segment: the next
    /// delta opened a duplicate same-id block below the intruder, and the
    /// original stayed `done == false` forever, painting the live
    /// `✦ Thinking…` cue (with the id-shared frozen elapsed) as a permanent
    /// transcript row.
    ///
    /// Block ids restart at 0 every turn, so the scan is bounded by
    /// `turn_start_block_idx` and additionally stops at a `UserMessage`
    /// (mid-turn steering) — joining across either boundary would splice a
    /// new segment onto an unrelated old block.
    // `Err` hands the unmerged block back to the caller's append path — a
    // move the caller performs anyway — so boxing it would trade one stack
    // copy on the per-delta hot path for an allocation.
    #[allow(clippy::result_large_err)]
    fn try_rejoin_reasoning(&mut self, block: RenderBlock) -> Result<usize, RenderBlock> {
        let target = match &block {
            RenderBlock::Reasoning { id, .. } => self.reasoning_rejoin_target(id.0),
            _ => None,
        };
        let Some(idx) = target else {
            return Err(block);
        };
        let RenderBlock::Reasoning {
            text,
            signature,
            done,
            ..
        } = block
        else {
            // Unreachable by construction (`target` is only `Some` for a
            // reasoning block); keep the safe fallback over a panic.
            return Err(block);
        };
        if let Some(RenderBlock::Reasoning {
            text: existing_text,
            signature: existing_signature,
            done: existing_done,
            ..
        }) = self.blocks.get_mut(idx)
        {
            existing_text.push_str(&text);
            if signature.is_some() {
                *existing_signature = signature;
            }
            *existing_done = done;
        }
        // The reasoning render cache is keyed by the per-block render version
        // (same contract as the tail-merge path), and the block sits mid-list,
        // so its height change must dirty the layout from its own y-offset.
        self.bump_render_version(idx);
        self.mark_layout_dirty_from(idx);
        Ok(idx)
    }

    /// Index of the current turn's `Reasoning` block carrying `id`, or `None`
    /// when no turn is active or the id has no block yet. Scans backwards and
    /// never crosses the turn start or a mid-turn `UserMessage`.
    fn reasoning_rejoin_target(&self, id: u64) -> Option<usize> {
        let turn_start = self.turn_start_block_idx?;
        for idx in (turn_start..self.blocks.len()).rev() {
            match self.blocks.get(idx) {
                Some(RenderBlock::UserMessage { .. }) => return None,
                Some(RenderBlock::Reasoning { id: existing, .. }) if existing.0 == id => {
                    return Some(idx);
                }
                _ => {}
            }
        }
        None
    }

    fn try_merge_block(&mut self, block: RenderBlock) -> Option<RenderBlock> {
        if let RenderBlock::ToolResult {
            tool_call_id,
            is_error,
            ..
        } = &block
        {
            self.reconcile_tool_call_status(tool_call_id, *is_error);
        }

        if let RenderBlock::ToolCall {
            id,
            tool_call_id,
            name,
            summary,
            preview,
            status,
        } = block
        {
            self.remove_trailing_stray_tool_call_marker();
            if let Some(idx) = self.find_tool_call_index(&tool_call_id) {
                self.update_existing_tool_call(idx, &tool_call_id, name, summary, preview, status);
                return None;
            }
            return Some(RenderBlock::ToolCall {
                id,
                tool_call_id,
                name,
                summary,
                preview,
                status,
            });
        }

        let tail_idx = self.blocks.len().saturating_sub(1);
        let Some(last) = self.blocks.last_mut() else {
            return Some(block);
        };

        let mut bumped_tail_version = false;
        let merged = match (last, block) {
            (
                RenderBlock::TextDelta {
                    id: last_id,
                    text: last_text,
                    done: last_done,
                },
                RenderBlock::TextDelta { id, text, done },
            ) if *last_id == id => {
                last_text.push_str(&text);
                *last_done = done;
                bumped_tail_version = true;
                None
            }
            (
                RenderBlock::Reasoning {
                    id: last_id,
                    text: last_text,
                    signature: last_signature,
                    done: last_done,
                },
                RenderBlock::Reasoning {
                    id,
                    text,
                    signature,
                    done,
                },
            ) if *last_id == id => {
                last_text.push_str(&text);
                if signature.is_some() {
                    *last_signature = signature;
                }
                *last_done = done;
                bumped_tail_version = true;
                None
            }
            (_, block) => Some(block),
        };
        if bumped_tail_version {
            self.bump_render_version(tail_idx);
        }
        merged
    }

    fn find_tool_call_index(&self, tool_call_id: &ToolCallId) -> Option<usize> {
        self.blocks.iter().position(|block| {
            matches!(
                block,
                RenderBlock::ToolCall { tool_call_id: existing, .. } if existing == tool_call_id
            )
        })
    }

    fn remove_trailing_stray_tool_call_marker(&mut self) {
        let Some(idx) = self.blocks.len().checked_sub(1) else {
            return;
        };
        let Some(RenderBlock::TextDelta { text, .. }) = self.blocks.get_mut(idx) else {
            return;
        };
        if !core_types::text::strip_trailing_stray_tool_call_marker(text) {
            return;
        }

        if text.is_empty() {
            self.remove_block_at(idx);
        } else {
            self.bump_render_version(idx);
            if let Some(slot) = self.rendered_cache.get_mut(idx) {
                *slot = None;
            }
        }
        self.mark_layout_dirty_from(idx);
        self.tail_active_dirty = true;    }

    fn update_existing_tool_call(
        &mut self,
        idx: usize,
        tool_call_id: &ToolCallId,
        name: String,
        summary: String,
        preview: ToolPreview,
        status: ToolCallStatus,
    ) {
        if let Some(RenderBlock::ToolCall {
            id,
            name: existing_name,
            summary: existing_summary,
            preview: existing_preview,
            status: existing_status,
            ..
        }) = self.blocks.get_mut(idx)
        {
            *existing_name = name;
            *existing_summary = summary;
            *existing_preview = preview;
            *existing_status = status;
            if !matches!(status, ToolCallStatus::Pending | ToolCallStatus::Running) {
                self.expanded.remove(&id.0);
            }
        }

        if let Some(slot) = self.rendered_cache.get_mut(idx) {
            *slot = None;
        }
        // An in-place member rewrite (same call id, new name/summary/preview)
        // changes rendered content without touching its collapsed group's span
        // or err count. Bump this member's render version so the group's
        // cached summary — keyed on the span's version sum in the LEADER's
        // slot, which the member-level clear above does not touch — re-keys
        // and rebuilds, regardless of the span's collapse shape.
        self.bump_render_version(idx);

        // Older sessions could already have duplicate ToolCall rows for the
        // same provider call id. Compact those while handling the next update
        // so they stop inflating tool summaries and stale active state.
        let duplicate_indices: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(candidate_idx, block)| {
                if candidate_idx == idx {
                    return None;
                }
                match block {
                    RenderBlock::ToolCall {
                        tool_call_id: existing,
                        ..
                    } if existing == tool_call_id => Some(candidate_idx),
                    _ => None,
                }
            })
            .collect();
        for duplicate_idx in duplicate_indices.into_iter().rev() {
            self.remove_block_at(duplicate_idx);
        }

        self.mark_layout_dirty_from(idx);
        self.tail_active_dirty = true;    }

    fn remove_block_at(&mut self, idx: usize) {
        if idx >= self.blocks.len() {
            return;
        }
        let removed = self.blocks.remove(idx);
        let removed_id = block_id(&removed).0;
        self.superseded_todo_block_ids.remove(&removed_id);
        if let RenderBlock::ToolCall { id, .. } = removed {
            self.tool_call_started_at.remove(&id.0);
        }
        self.remove_entry_slots(idx);
        if let Some(focused_idx) = self.focused_idx {
            self.focused_idx = match focused_idx.cmp(&idx) {
                std::cmp::Ordering::Equal => idx.checked_sub(1),
                std::cmp::Ordering::Greater => Some(focused_idx - 1),
                std::cmp::Ordering::Less => Some(focused_idx),
            };
        }
    }

    fn reconcile_tool_call_status(
        &mut self,
        tool_call_id: &runtime::message_stream::ToolCallId,
        is_error: bool,
    ) {
        let next_status = if is_error {
            ToolCallStatus::Errored
        } else {
            ToolCallStatus::Ok
        };
        let matching_indices: Vec<usize> = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(idx, block)| {
                matches!(
                    block,
                    RenderBlock::ToolCall { tool_call_id: existing, .. } if existing == tool_call_id
                )
                .then_some(idx)
            })
            .collect();
        let Some(&first_idx) = matching_indices.first() else {
            return;
        };

        let settled_ids: Vec<u64> = matching_indices
            .iter()
            .filter_map(|&idx| match self.blocks.get(idx) {
                Some(RenderBlock::ToolCall { id, .. }) => Some(id.0),
                _ => None,
            })
            .collect();
        for id in settled_ids {
            self.expanded.remove(&id);
        }

        for &idx in &matching_indices {
            if let Some(RenderBlock::ToolCall { status, .. }) = self.blocks.get_mut(idx) {
                *status = next_status;
            }
            if let Some(slot) = self.rendered_cache.get_mut(idx) {
                *slot = None;
            }
        }
        for &duplicate_idx in matching_indices.iter().skip(1).rev() {
            self.remove_block_at(duplicate_idx);
        }

        // Parallel tools complete out of order: the flipped call may sit
        // well before the tail, and a status flip can re-group its tool
        // run (collapse state → heights). Re-measure from there.
        self.mark_layout_dirty_from(first_idx);
        // A status flip from Pending/Running → Ok/Errored removes that call
        // from the active set. Mark dirty so the next draw refreshes the
        // cached tail-active index.
        self.tail_active_dirty = true;    }
}

/// Render a thin horizontal separator at a conversation turn boundary.
///
/// `turn_number` labels a genuine new-turn boundary as `── turn N ─────` so a
/// long transcript reads as discrete turns at a glance (macro rhythm the rail
/// alone is too subtle to carry); a mid-turn response boundary passes `None` and
/// gets the plain rule.
fn draw_turn_separator(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: &Theme,
    turn_number: Option<usize>,
) {
    let rule_char = if theme.no_color { "-" } else { "\u{2500}" };
    let width = usize::from(area.width);
    // Faint hue, but no DIM modifier: the turn rule should be a quiet but
    // legible divider. With the gap tightened to 2 it carries the turn
    // separation on its own, so the previous DIM (near-invisible on dark
    // themes) defeated the purpose.
    let style = Style::new().fg(theme.palette.faint);
    // A labeled band only when there is room for the label plus a little rule on
    // each side; otherwise fall back to the plain full-width rule.
    let line = match turn_number {
        Some(n) if width >= 12 => {
            let label = format!(" turn {n} ");
            let lead = 3usize;
            let trail = width.saturating_sub(lead + label.chars().count());
            Line::from(vec![
                Span::styled(rule_char.repeat(lead), style),
                Span::styled(label, style),
                Span::styled(rule_char.repeat(trail), style),
            ])
        }
        _ => Line::from(Span::styled(rule_char.repeat(width), style)),
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn is_tool_block(block: &RenderBlock) -> bool {
    matches!(
        block,
        RenderBlock::ToolCall { .. }
            | RenderBlock::ToolResult { .. }
            | RenderBlock::PermissionPrompt(_)
    )
}

fn is_response_block(block: &RenderBlock) -> bool {
    matches!(
        block,
        RenderBlock::TextDelta { .. }
            | RenderBlock::Reasoning { .. }
            | RenderBlock::ToolCall { .. }
            | RenderBlock::ToolResult { .. }
            | RenderBlock::PermissionPrompt(_)
            | RenderBlock::UserQuestionPrompt(_)
            | RenderBlock::System { .. }
            | RenderBlock::UserNotice { .. }
            | RenderBlock::Card { .. }
    )
}

/// Author-mark style for an assistant prose block: a `◆`-bulleted head, an
/// indent-only continuation (prose directly after prose), or bare (non-prose).
/// The continuation case keeps the indent so its text stays in the same left
/// column as the block it continues, rather than jumping to col 0.
fn assistant_prose_style(blocks: &[RenderBlock], idx: usize) -> blocks::ProseMark {
    use blocks::ProseMark;
    if !matches!(blocks.get(idx), Some(RenderBlock::TextDelta { .. })) {
        return ProseMark::Bare;
    }

    for previous in blocks[..idx].iter().rev() {
        // Reasoning blocks are transparent to prose authorship — a thinking
        // block (whether the live `Thinking…` line or a settled, collapsed
        // step) is not the prose "author", so the mark style of the answer must
        // look *past* it. Skipping only the settled (suppressed) case made the
        // mark flip Bullet↔Indent (or ↔Bare) on the `done` transition
        // mid-answer, changing the block height and jumping the layout. Skip
        // every Reasoning variant so the answer's mark is stable across the
        // flip.
        if matches!(previous, RenderBlock::Reasoning { .. }) {
            continue;
        }
        // A settled empty prose block is suppressed (height 0, not drawn — see
        // `is_empty_prose_suppressed`), so it is not the visible author either.
        // Look past it so the next real answer still gets its `◆` bullet
        // instead of mis-reading the phantom as a prior prose block and
        // dropping to an indent-only continuation.
        if matches!(
            previous,
            RenderBlock::TextDelta { text, done: true, .. } if text.trim().is_empty()
        ) {
            continue;
        }
        return match previous {
            RenderBlock::TextDelta { .. } => ProseMark::Indent,
            _ => ProseMark::Bullet,
        };
    }

    // Nothing but reasoning (or nothing at all) precedes this prose: it is the
    // turn's first *spoken* answer either way, so it carries the author bullet.
    // The bullet is the only author cue in the bullet grammar (no header, no
    // rail), so the very first answer of a session gets one too.
    ProseMark::Bullet
}

/// Draw the vertical scrollbar in the 1-column gutter the layout reserves on
/// the right edge of `area` when content overflows. Uses ratatui's stateful
/// [`Scrollbar`] widget (rendered as a second pass over the same `area`) so the
/// thumb sizing/position is handled by the framework; theme colors mirror the
/// previous hand-rolled indicator (dim arrows, accent thumb, muted track).
fn draw_scroll_indicator(
    frame: &mut Frame<'_>,
    area: Rect,
    scroll: u16,
    content_total: u16,
    viewport_h: u16,
    theme: &Theme,
) {
    let arrow_style = Style::new()
        .fg(theme.palette.dim)
        .add_modifier(Modifier::DIM);
    // ratatui's `content_length` is the number of *scroll positions*, and it puts
    // the thumb at the bottom only when `position == content_length - 1`. Our
    // `scroll` maxes out at `content_total - viewport_h` (we never scroll the last
    // line past the bottom edge), so passing the full `content_total` left the
    // thumb stuck mid-track at the real bottom. Pass the scroll-position count
    // (`max_scroll + 1`) instead, so the thumb reaches the bottom when the text
    // does — and its size still reflects viewport/content (max_viewport_position
    // works out to `content_total`).
    let scroll_positions = content_total.saturating_sub(viewport_h).saturating_add(1);
    let mut state = ScrollbarState::new(usize::from(scroll_positions))
        .position(usize::from(scroll))
        .viewport_content_length(usize::from(viewport_h));
    // Glyphs flow through `glyphs::SCROLL_*` (the single source of truth per
    // components.md §7) and degrade to their 1-cell ASCII siblings (`^ v # .`)
    // under `NO_COLOR`/`TERM=dumb`, so the gutter never paints a rich glyph
    // a dumb terminal can't show (R10).
    let color = !theme.no_color;
    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(Some(glyphs::pick(
            color,
            glyphs::SCROLL_UP,
            glyphs::SCROLL_UP_NC,
        )))
        .end_symbol(Some(glyphs::pick(
            color,
            glyphs::SCROLL_DOWN,
            glyphs::SCROLL_DOWN_NC,
        )))
        .track_symbol(Some(glyphs::pick(
            color,
            glyphs::SCROLL_TRACK,
            glyphs::SCROLL_TRACK_NC,
        )))
        .thumb_symbol(glyphs::pick(
            color,
            glyphs::SCROLL_THUMB,
            glyphs::SCROLL_THUMB_NC,
        ))
        .begin_style(arrow_style)
        .end_style(arrow_style)
        .track_style(Style::new().fg(theme.palette.muted))
        .thumb_style(Style::new().fg(theme.palette.dim));
    frame.render_stateful_widget(scrollbar, area, &mut state);
}

/// `Vec<Line<'a>>` → `Vec<Line<'static>>` 로 owned 변환.
///
/// `tool_result::rendered_lines` / `diff::lines` 등의 반환값은 body lifetime
/// 에 묶여 있어 [`RenderCache`] 슬롯에 그대로 저장할 수 없다. Span 의
/// `Cow<'a, str>` 을 `Cow::Owned(String)` 으로 승격해 lifetime 을 끊는다.
/// 비용: span 당 String alloc + memcpy. cache miss 한 번만 부담하고,
/// 이후 매 frame cache hit 시 0.
fn lines_to_static(lines: Vec<Line<'_>>) -> Vec<Line<'static>> {
    lines.into_iter().map(line_to_static).collect()
}

fn line_to_static(line: Line<'_>) -> Line<'static> {
    let spans = line
        .spans
        .into_iter()
        .map(|span| Span {
            style: span.style,
            content: std::borrow::Cow::Owned(span.content.into_owned()),
        })
        .collect();
    Line {
        style: line.style,
        alignment: line.alignment,
        spans,
    }
}

/// Resolve a char-selection gesture — `anchor`/`head` in `(screen col,
/// content row)` coordinates — to per-row inclusive column spans
/// `(content_row, col_start, col_end)` for the rows currently visible.
/// `scroll` is the first visible content row; `clip` is the on-screen
/// content rect (scrollbar gutter excluded) that bounds columns and the
/// visible row count. The first/middle/last shape is decided over the *full*
/// selection before clipping, so a selection taller than the screen washes
/// the correct slice: an interior row revealed by scrolling spans the full
/// width even though the selection's endpoints are offscreen.
/// Terminal-selection shape: a single row spans between the two columns;
/// across rows, the first row runs from its start column to the right edge,
/// middle rows span the full width, and the last row runs from the left edge
/// to its end column. Order-agnostic — an upward/leftward drag yields the
/// same spans.
fn char_selection_rows(
    anchor: (u16, u16),
    head: (u16, u16),
    clip: Rect,
    scroll: u16,
) -> Vec<(u16, u16, u16)> {
    if clip.width == 0 || clip.height == 0 {
        return Vec::new();
    }
    let left = clip.x;
    let right = clip.x + clip.width - 1;
    let clamp_col = |col: u16| col.clamp(left, right);
    let a = (anchor.1, clamp_col(anchor.0));
    let b = (head.1, clamp_col(head.0));
    let (start, end) = if a <= b { (a, b) } else { (b, a) };
    // Clip to the visible content-row band. `(row, col)` tuple order sorted
    // the endpoints, so on a single-row selection start.1 <= end.1 already.
    let first = start.0.max(scroll);
    let last = end.0.min(scroll.saturating_add(clip.height - 1));
    if first > last {
        return Vec::new();
    }
    let mut rows = Vec::with_capacity(usize::from(last - first) + 1);
    for row in first..=last {
        let col_start = if row == start.0 { start.1 } else { left };
        let col_end = if row == end.0 { end.1 } else { right };
        rows.push((row, col_start, col_end));
    }
    rows
}

/// Concatenate one buffer row's cell symbols over the inclusive column span.
/// Wide graphemes (CJK) occupy multiple cells but store their symbol only in
/// the head cell; advancing by the symbol's display width skips the
/// continuation cells, so 한글 copies as `안녕` — not `안 녕 ` like a raw
/// screen scrape (the exact artifact the render-dump tests document).
fn buffer_row_text(buffer: &Buffer, row: u16, col_start: u16, col_end: u16) -> String {
    let mut text = String::new();
    let mut col = col_start;
    while col <= col_end {
        let symbol = buffer[(col, row)].symbol();
        text.push_str(symbol);
        let width = u16::try_from(UnicodeWidthStr::width(symbol).max(1)).unwrap_or(1);
        let Some(next) = col.checked_add(width) else {
            break;
        };
        col = next;
    }
    text
}

/// Join mined selection rows into the clipboard payload: trailing whitespace
/// per row is rendering slack (padding cells), not content, so strip it; keep
/// interior blank rows so multi-paragraph copies read like the transcript.
/// Empty overall → empty string (the caller treats that as "nothing to copy").
fn join_selection_lines(lines: &[String]) -> String {
    let joined = lines
        .iter()
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.trim().is_empty() {
        String::new()
    } else {
        joined
    }
}

#[cfg(test)]
mod tests;
