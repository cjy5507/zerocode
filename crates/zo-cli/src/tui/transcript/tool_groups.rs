//! Tool-group collapse domain (components.md §5.3) — grouping state over
//! consecutive tool calls, per-group tool-mix counting, and the collapsed
//! summary line renderer.

#![allow(clippy::doc_markdown)]

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use runtime::message_stream::{RenderBlock, ToolCallStatus, ToolPreview, ToolResultBody};
use unicode_width::UnicodeWidthStr;

use crate::tui::theme::Theme;

/// Per-block tool group collapse state (components.md §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolGroupState {
    /// Not part of a collapsed group — render normally.
    Normal,
    /// First block in a collapsed group — render the summary line.
    Summary {
        total: u16,
        ok_count: u16,
        err_count: u16,
        running_count: u16,
        pending_count: u16,
        read_count: u16,
        search_count: u16,
        /// Web-research breakdown (a subset overlap: a web search counts in both
        /// `search_count` and `web_search_count`). Drives the live "Web research
        /// · N searches · M fetches" leader while the batch is in flight.
        web_search_count: u16,
        fetch_count: u16,
        exec_count: u16,
    },
    /// Hidden member of a collapsed group — height 0.
    Hidden,
}

#[cfg(test)]
thread_local! {
    static COMPUTE_TOOL_GROUPS_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Global "show every tool row" toggle (Ctrl+X — Claude Code's ctrl+o
/// verbose-transcript parity): when set, grouping is disabled at the
/// classification point, so BOTH the full recompute and every incremental
/// tail path classify each block `Normal` and the transcript renders each
/// tool call/result individually. Process-global on purpose: it is a pure
/// view preference read at classification time, so no layout-cache seam can
/// disagree with another.
static TOOL_GROUPS_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn tool_groups_disabled() -> bool {
    TOOL_GROUPS_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn set_tool_groups_disabled(disabled: bool) {
    TOOL_GROUPS_DISABLED.store(disabled, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn reset_compute_tool_groups_call_count() {
    COMPUTE_TOOL_GROUPS_CALLS.set(0);
}

#[cfg(test)]
pub(super) fn compute_tool_groups_call_count() -> usize {
    COMPUTE_TOOL_GROUPS_CALLS.get()
}

/// Recompute tool group collapse state for all blocks.
///
/// A "tool group" is a run of 2+ ToolCall/ToolResult pairs. Settled runs
/// collapse into one compact `verb target` row per tool. In-flight clusters
/// (`Pending`/`Running`) form a **live group** (Claude Code parity): the same
/// per-tool rows, but each row carries its own status marker — a spinner while
/// that tool runs, `○` while queued, `✓`/`×` the moment it settles — so a
/// parallel batch is visible and animating *while it executes* instead of
/// popping in fully-formed after the fact. When a new in-flight suffix is
/// appended behind an already-settled summary, only that suffix becomes the
/// live group so the stable completed summary does not flicker.
#[allow(clippy::too_many_lines)] // one forward sweep of the grouping state machine
pub(super) fn compute_tool_groups(blocks: &[RenderBlock]) -> Vec<ToolGroupState> {
    #[cfg(test)]
    COMPUTE_TOOL_GROUPS_CALLS.set(COMPUTE_TOOL_GROUPS_CALLS.get().saturating_add(1));

    // Verbose-transcript toggle: grouping off means every block is `Normal`.
    // This early return is the single gate for the incremental paths too —
    // `recompute_tool_groups_tail` delegates here.
    if tool_groups_disabled() {
        return vec![ToolGroupState::Normal; blocks.len()];
    }

    let mut states = vec![ToolGroupState::Normal; blocks.len()];
    let len = blocks.len();
    let mut i = 0;

    while i < len {
        // Try to start a tool group at position i.
        if !matches!(
            blocks[i],
            RenderBlock::ToolCall {
                status: ToolCallStatus::Ok | ToolCallStatus::Cancelled,
                ..
            }
        ) {
            i += 1;
            continue;
        }

        let group_start = i;

        // Parallel tool dispatch often arrives as a batch of completed calls
        // followed by a batch of their results:
        //
        //   ToolCall(a), ToolCall(b), ToolResult(a), ToolResult(b)
        //
        // The interleaved-pair logic below only sees the completed calls and
        // leaves the result batch visible, producing a confusing "N tools done"
        // summary immediately followed by N individual results. Treat this
        // call-run + matching-result-run as one collapsed operation group.
        let mut call_ids = Vec::new();
        let mut call_run_end = group_start;
        while call_run_end < len {
            let RenderBlock::ToolCall {
                tool_call_id,
                status: ToolCallStatus::Ok | ToolCallStatus::Cancelled,
                ..
            } = &blocks[call_run_end]
            else {
                break;
            };
            call_ids.push(tool_call_id);
            call_run_end += 1;
        }

        if call_ids.len() >= 2 {
            let mut result_run_end = call_run_end;
            let mut result_count: u16 = 0;
            let mut err_count: u16 = 0;
            while result_run_end < len {
                let RenderBlock::ToolResult {
                    tool_call_id,
                    is_error,
                    ..
                } = &blocks[result_run_end]
                else {
                    break;
                };
                if !call_ids.contains(&tool_call_id) {
                    break;
                }
                result_count = result_count.saturating_add(1);
                if *is_error {
                    err_count = err_count.saturating_add(1);
                }
                result_run_end += 1;
            }

            if result_count > 0 {
                let total = u16::try_from(call_ids.len()).unwrap_or(u16::MAX);
                states[group_start] = ToolGroupState::Summary {
                    total,
                    ok_count: total.saturating_sub(err_count),
                    err_count,
                    running_count: 0,
                    pending_count: 0,
                    read_count: 0,
                    search_count: 0,
                    web_search_count: 0,
                    fetch_count: 0,
                    exec_count: 0,
                };
                for state in &mut states[(group_start + 1)..result_run_end] {
                    *state = ToolGroupState::Hidden;
                }
                i = result_run_end;
                continue;
            }
        }

        let mut pair_count: u16 = 0;
        let mut ok_count: u16 = 0;
        let mut err_count: u16 = 0;
        let mut j = i;

        while j < len {
            // Expect a completed ToolCall.
            let is_completed_call = matches!(
                &blocks[j],
                RenderBlock::ToolCall {
                    status: ToolCallStatus::Ok | ToolCallStatus::Cancelled,
                    ..
                }
            );
            if !is_completed_call {
                break;
            }

            // The completed-call guard above only admits `Ok`/`Cancelled`
            // calls, so errors here are signalled solely by the result block.
            // Check if next block is the matching ToolResult.
            if j + 1 < len && matches!(&blocks[j + 1], RenderBlock::ToolResult { .. }) {
                pair_count += 1;
                let is_error_result = matches!(
                    &blocks[j + 1],
                    RenderBlock::ToolResult { is_error: true, .. }
                );
                if is_error_result {
                    err_count += 1;
                } else {
                    ok_count += 1;
                }
                j += 2;
            } else {
                // Solo ToolCall without matching ToolResult — still count it.
                pair_count += 1;
                ok_count += 1;
                j += 1;
            }
        }

        // Only collapse groups of 2+ pairs.
        if pair_count >= 2 {
            states[group_start] = ToolGroupState::Summary {
                total: pair_count,
                ok_count,
                err_count,
                running_count: 0,
                pending_count: 0,
                read_count: 0,
                search_count: 0,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            };
            for state in &mut states[(group_start + 1)..j] {
                *state = ToolGroupState::Hidden;
            }
        }

        i = j;
    }

    coalesce_split_tool_groups(blocks, &mut states);
    mark_live_tool_groups(blocks, &mut states);
    reveal_diff_pairs(blocks, &mut states);
    recount_summaries_excluding_diffs(blocks, &mut states);
    states
}

/// Claude Code parity: an `Edit`/`Write` diff is shown **inline**, never folded
/// into a `N tools done` summary. After the grouping passes have decided
/// collapse state, re-reveal every `Diff` `ToolResult` and its matching
/// `ToolCall` (flip `Hidden` → `Normal`) and drop them from the summary leader's
/// counts, so the leader truthfully reports only the tools that stay collapsed.
///
/// Provider-neutral: this keys solely on the boundary `RenderBlock::ToolResult`
/// carrying a [`ToolResultBody::Diff`], so every adapter (Anthropic, OpenAI,
/// future backends) that lowers an edit into that variant gets the same inline
/// diff with no model-specific branch. A diff is *settled by definition* (the
/// result exists), so this also fires inside a still-live group: the moment an
/// edit lands mid-batch its diff pops out of the live rows and renders inline,
/// while the remaining in-flight tools keep animating in the live group
/// ([`mark_live_tool_groups`] runs first).
fn reveal_diff_pairs(blocks: &[RenderBlock], states: &mut [ToolGroupState]) {
    let len = blocks.len().min(states.len());

    // Collect the call ids whose result is a diff, so the paired ToolCall row is
    // revealed too (a lone collapsed call above an inline diff reads as orphaned).
    let diff_call_ids: Vec<&str> = (0..len)
        .filter_map(|idx| match &blocks[idx] {
            RenderBlock::ToolResult {
                tool_call_id,
                body: ToolResultBody::Diff(_),
                ..
            } => Some(tool_call_id.0.as_str()),
            _ => None,
        })
        .collect();
    if diff_call_ids.is_empty() {
        return;
    }

    for idx in 0..len {
        let is_diff_pair = match &blocks[idx] {
            RenderBlock::ToolResult {
                body: ToolResultBody::Diff(_),
                ..
            } => true,
            RenderBlock::ToolCall { tool_call_id, .. } => {
                diff_call_ids.contains(&tool_call_id.0.as_str())
            }
            _ => false,
        };
        if !is_diff_pair {
            continue;
        }
        // A diff pair that was hidden inside a collapsed group becomes its own
        // inline block again. A `Summary` leader that happens to be a diff pair
        // keeps leading the run, but its counts are corrected below.
        if matches!(states[idx], ToolGroupState::Hidden) {
            states[idx] = ToolGroupState::Normal;
        }
    }
}

/// Recompute each `Summary` leader's counts over only the rows that remain
/// `Hidden` under it (diff pairs were just revealed to `Normal`, and a live
/// suffix owns its own leader). A leader left with fewer than two
/// still-collapsed calls is demoted to `Normal` so a lone tool is shown as a
/// normal row rather than a one-row group.
fn recount_summaries_excluding_diffs(blocks: &[RenderBlock], states: &mut [ToolGroupState]) {
    let len = blocks.len().min(states.len());
    let mut i = 0;
    while i < len {
        if !matches!(states[i], ToolGroupState::Summary { .. }) {
            i += 1;
            continue;
        }
        let leader = i;
        // Span this leader's own group: the contiguous tool rows up to the
        // next `Summary` leader (a live suffix behind a settled prefix owns
        // its own leader — its rows must not leak into the prefix counts) or
        // the first non-tool block. `Normal` rows inside the span (revealed
        // diff pairs, exempt delegation hosts) are excluded by the collapsed
        // predicate in [`count_collapsed_span`].
        let run_end = group_span_end(blocks, states, leader);

        let counts = count_collapsed_span(blocks, states, leader, run_end);
        if counts.total >= 2 {
            states[leader] = counts.into_summary();
        } else {
            // Not enough collapsed tools to justify a group — show the leader
            // and its (now mostly revealed) run as normal rows.
            for state in &mut states[leader..run_end] {
                if matches!(
                    state,
                    ToolGroupState::Summary { .. } | ToolGroupState::Hidden
                ) {
                    *state = ToolGroupState::Normal;
                }
            }
        }
        i = run_end.max(leader + 1);
    }
}

/// End (exclusive) of the group led by `leader`: the contiguous tool-block
/// span up to — but not including — the next `Summary` leader. Shared by the
/// recount pass, the detail-row builder, and the height measurement so all
/// three agree on which rows belong to a leader.
pub(super) fn group_span_end(
    blocks: &[RenderBlock],
    states: &[ToolGroupState],
    leader: usize,
) -> usize {
    let len = blocks.len().min(states.len());
    let mut end = leader + 1;
    while end < len
        && is_tool_group_block(&blocks[end])
        && !matches!(states[end], ToolGroupState::Summary { .. })
    {
        end += 1;
    }
    end
}

/// Reveal user-opened collapsed groups (mouse click on the summary leader —
/// CC parity): every `Summary` leader whose block id is in `revealed` flips to
/// `Normal`, along with its `Hidden` span members, so the individual tool rows
/// render. Runs as a post-pass after EVERY tool-group recompute (full and
/// tail), so an append/live regroup cannot silently re-collapse a group the
/// user opened; a revealed id whose leader no longer exists (regrouped away)
/// simply no-ops.
pub(super) fn apply_group_reveals(
    blocks: &[RenderBlock],
    states: &mut [ToolGroupState],
    revealed: &std::collections::HashSet<u64>,
) {
    if revealed.is_empty() {
        return;
    }
    let len = blocks.len().min(states.len());
    let mut idx = 0;
    while idx < len {
        if matches!(states[idx], ToolGroupState::Summary { .. })
            && revealed.contains(&super::layout::block_id(&blocks[idx]).0)
        {
            let end = group_span_end(blocks, states, idx);
            states[idx] = ToolGroupState::Normal;
            for state in &mut states[idx + 1..end] {
                if matches!(state, ToolGroupState::Hidden) {
                    *state = ToolGroupState::Normal;
                }
            }
            idx = end;
        } else {
            idx += 1;
        }
    }
}

/// Tallies for one collapsed group span. Built by [`count_collapsed_span`];
/// converted into the leader state via [`Self::into_summary`].
#[derive(Debug, Default, Clone, Copy)]
struct GroupCounts {
    total: u16,
    err_count: u16,
    running_count: u16,
    pending_count: u16,
    read_count: u16,
    search_count: u16,
    web_search_count: u16,
    fetch_count: u16,
    exec_count: u16,
}

impl GroupCounts {
    fn into_summary(self) -> ToolGroupState {
        ToolGroupState::Summary {
            total: self.total,
            ok_count: self
                .total
                .saturating_sub(self.err_count)
                .saturating_sub(self.running_count)
                .saturating_sub(self.pending_count),
            err_count: self.err_count,
            running_count: self.running_count,
            pending_count: self.pending_count,
            read_count: self.read_count,
            search_count: self.search_count,
            web_search_count: self.web_search_count,
            fetch_count: self.fetch_count,
            exec_count: self.exec_count,
        }
    }
}

/// Count the rows still collapsed under `leader` within `[leader, end)`: the
/// leader itself plus `Hidden` rows. `Normal` rows (revealed diff pairs,
/// exempt delegation hosts, demoted members) are excluded. In-flight calls
/// count toward `total` *and* their running/pending tallies, so a live group's
/// leader reports an accurate `N tools` figure while animating.
fn count_collapsed_span(
    blocks: &[RenderBlock],
    states: &[ToolGroupState],
    leader: usize,
    end: usize,
) -> GroupCounts {
    let mut counts = GroupCounts::default();
    let end = end.min(blocks.len()).min(states.len());
    for idx in leader..end {
        let collapsed = idx == leader || matches!(states[idx], ToolGroupState::Hidden);
        if !collapsed {
            continue;
        }
        match &blocks[idx] {
            RenderBlock::ToolCall { name, status, .. } => {
                counts.total = counts.total.saturating_add(1);
                match status {
                    ToolCallStatus::Running => {
                        counts.running_count = counts.running_count.saturating_add(1);
                    }
                    ToolCallStatus::Pending => {
                        counts.pending_count = counts.pending_count.saturating_add(1);
                    }
                    _ => {}
                }
                match name.as_str() {
                    "read_file" | "Read" => {
                        counts.read_count = counts.read_count.saturating_add(1);
                    }
                    "web_search" | "WebSearch" => {
                        // A web search is both a "search" (generic breakdown)
                        // and web research.
                        counts.search_count = counts.search_count.saturating_add(1);
                        counts.web_search_count = counts.web_search_count.saturating_add(1);
                    }
                    "web_fetch" | "WebFetch" => {
                        counts.fetch_count = counts.fetch_count.saturating_add(1);
                    }
                    "grep_search" | "Grep" | "glob_search" | "Glob" => {
                        counts.search_count = counts.search_count.saturating_add(1);
                    }
                    "bash" | "Bash" => {
                        counts.exec_count = counts.exec_count.saturating_add(1);
                    }
                    _ => {}
                }
            }
            RenderBlock::ToolResult { is_error: true, .. } => {
                counts.err_count = counts.err_count.saturating_add(1);
            }
            _ => {}
        }
    }
    counts
}

/// In-flight tools are live progress — show them as a *live group*, not
/// hidden rows.
///
/// Preserve any already-collapsed settled prefix in a contiguous tool run,
/// then turn the in-flight suffix into a live group: a `Summary` leader whose
/// detail rows animate per-tool status markers (see
/// [`collapsed_tool_detail_lines`]). This keeps stable history such as
/// `✓ glob/read … +N more` visible while a new tool iteration starts, and —
/// unlike the old hide-everything pass — makes a parallel batch visible row by
/// row *while it executes* (Claude Code parity). If the run has no settled
/// summary prefix, the whole run becomes one live group.
fn mark_live_tool_groups(blocks: &[RenderBlock], states: &mut [ToolGroupState]) {
    let len = blocks.len().min(states.len());
    let mut i = 0;
    while i < len {
        if !is_tool_group_block(&blocks[i]) {
            i += 1;
            continue;
        }

        let run_start = i;
        let mut run_end = i;
        let mut first_inflight = None;
        while run_end < len && is_tool_group_block(&blocks[run_end]) {
            if first_inflight.is_none() && is_inflight_call(&blocks[run_end]) {
                first_inflight = Some(run_end);
            }
            run_end += 1;
        }

        if let Some(first_inflight) = first_inflight {
            let has_settled_summary_prefix = states[run_start..first_inflight]
                .iter()
                .any(|state| matches!(state, ToolGroupState::Summary { .. }));
            let prefix_has_results = blocks[run_start..first_inflight]
                .iter()
                .any(|block| matches!(block, RenderBlock::ToolResult { .. }));
            let suffix_already_owns_collapsed_rows = states[first_inflight..run_end]
                .iter()
                .any(|state| {
                    matches!(state, ToolGroupState::Summary { .. } | ToolGroupState::Hidden)
                });
            let live_start = if has_settled_summary_prefix
                && prefix_has_results
                && !suffix_already_owns_collapsed_rows
            {
                // Pull contiguous not-yet-grouped rows (`Normal`) immediately
                // before the first in-flight call into the live group: a
                // just-settled call whose sibling is still running (e.g.
                // `c10 Ok · r10 · c11 Pending`) belongs to the live batch —
                // its row flips to ✓ in place — instead of dropping out as a
                // stray solo row plus a full result body that re-collapses a
                // frame later.
                let mut start = first_inflight;
                while start > run_start
                    && matches!(states[start - 1], ToolGroupState::Normal)
                {
                    start -= 1;
                }
                start
            } else {
                run_start
            };
            mark_live_group(blocks, states, live_start, run_end);
        }

        i = run_end;
    }
}

/// Whether this block is a still-in-flight tool call.
fn is_inflight_call(block: &RenderBlock) -> bool {
    matches!(
        block,
        RenderBlock::ToolCall {
            status: ToolCallStatus::Pending | ToolCallStatus::Running,
            ..
        }
    )
}

/// An in-flight delegation host (Spawn family / `Workflow`) renders its own
/// live per-agent tree under its row, so it is exempt from live-group folding —
/// folding it would reduce the visible tree to one `agent …` line. Settled
/// spawn pairs keep folding into settled summaries exactly as before.
fn is_live_exempt_call(block: &RenderBlock) -> bool {
    matches!(
        block,
        RenderBlock::ToolCall { name, status: ToolCallStatus::Pending | ToolCallStatus::Running, .. }
            if crate::tui::blocks::tool_call::opens_agent_batch(name)
    )
}

/// Turn `[start, end)` into one live group: a `Summary` leader on the first
/// foldable call plus `Hidden` members. Spans holding fewer than two foldable
/// calls stay `Normal` instead — a solo running tool reads better as a normal
/// live event row (with its own elapsed/marker rendering) than a one-row
/// group. In-flight delegation hosts stay `Normal` so their live agent tree
/// keeps rendering under their own row.
fn mark_live_group(
    blocks: &[RenderBlock],
    states: &mut [ToolGroupState],
    start: usize,
    end: usize,
) {
    let exempt_ids: Vec<&str> = blocks[start..end]
        .iter()
        .filter_map(|block| match block {
            RenderBlock::ToolCall { tool_call_id, .. } if is_live_exempt_call(block) => {
                Some(tool_call_id.0.as_str())
            }
            _ => None,
        })
        .collect();
    let is_exempt = |block: &RenderBlock| match block {
        RenderBlock::ToolCall { tool_call_id, .. }
        | RenderBlock::ToolResult { tool_call_id, .. } => {
            exempt_ids.contains(&tool_call_id.0.as_str())
        }
        _ => false,
    };

    let member_calls = blocks[start..end]
        .iter()
        .filter(|block| matches!(block, RenderBlock::ToolCall { .. }) && !is_exempt(block))
        .count();
    if member_calls < 2 {
        // Solo in-flight tool (or delegation hosts only): normal live rows.
        for state in &mut states[start..end] {
            *state = ToolGroupState::Normal;
        }
        return;
    }

    let mut leader: Option<usize> = None;
    // Prefer a leader that is not a settled diff pair: `reveal_diff_pairs`
    // will pop diff pairs out of the group as inline blocks, and a leader
    // cannot be revealed (only `Hidden` rows flip), which would strand the
    // diff inside the group.
    let diff_ids: Vec<&str> = blocks[start..end]
        .iter()
        .filter_map(|block| match block {
            RenderBlock::ToolResult {
                tool_call_id,
                body: ToolResultBody::Diff(_),
                ..
            } => Some(tool_call_id.0.as_str()),
            _ => None,
        })
        .collect();
    let mut fallback_leader: Option<usize> = None;
    for idx in start..end {
        let block = &blocks[idx];
        if is_exempt(block) {
            states[idx] = ToolGroupState::Normal;
            continue;
        }
        if leader.is_none() {
            if let RenderBlock::ToolCall { tool_call_id, .. } = block {
                if !diff_ids.contains(&tool_call_id.0.as_str()) {
                    leader = Some(idx);
                    continue;
                }
                if fallback_leader.is_none() {
                    fallback_leader = Some(idx);
                }
            }
        }
        states[idx] = ToolGroupState::Hidden;
    }
    let leader = leader.or(fallback_leader);
    if let Some(leader) = leader {
        states[leader] = count_collapsed_span(blocks, states, leader, end).into_summary();
    }
}

/// Collapse a contiguous settled tool run that fragmented into **multiple**
/// summary leaders back into a single summary.
///
/// Parallel tool results can arrive at the tail in a different order than the
/// original calls. The completed-pair scanner may then produce more than one
/// `✓ done` leader for a single contiguous tool cluster. When a run owns 2+
/// summary leaders, recompute it as one group so settled tool history stays to
/// one compact row. In-flight runs become live groups via
/// [`mark_live_tool_groups`] after this pass.
fn coalesce_split_tool_groups(blocks: &[RenderBlock], states: &mut [ToolGroupState]) {
    let len = blocks.len().min(states.len());
    let mut i = 0;
    while i < len {
        if !is_tool_group_block(&blocks[i]) {
            i += 1;
            continue;
        }
        let run_start = i;
        let mut run_end = i;
        let mut has_inflight = false;
        while run_end < len && is_tool_group_block(&blocks[run_end]) {
            if matches!(
                blocks[run_end],
                RenderBlock::ToolCall {
                    status: ToolCallStatus::Pending | ToolCallStatus::Running,
                    ..
                }
            ) {
                has_inflight = true;
            }
            run_end += 1;
        }
        if has_inflight {
            i = run_end;
            continue;
        }

        let leader_count = states[run_start..run_end]
            .iter()
            .filter(|state| matches!(state, ToolGroupState::Summary { .. }))
            .count();
        if leader_count >= 2 {
            let mut total: u16 = 0;
            let mut running_count: u16 = 0;
            let mut pending_count: u16 = 0;
            let mut err_count: u16 = 0;
            for block in &blocks[run_start..run_end] {
                match block {
                    RenderBlock::ToolCall { status, .. } => {
                        total = total.saturating_add(1);
                        match status {
                            ToolCallStatus::Running => {
                                running_count = running_count.saturating_add(1);
                            }
                            ToolCallStatus::Pending => {
                                pending_count = pending_count.saturating_add(1);
                            }
                            _ => {}
                        }
                    }
                    RenderBlock::ToolResult { is_error: true, .. } => {
                        err_count = err_count.saturating_add(1);
                    }
                    _ => {}
                }
            }
            states[run_start] = ToolGroupState::Summary {
                total,
                ok_count: total.saturating_sub(err_count),
                err_count,
                running_count,
                pending_count,
                read_count: 0,
                search_count: 0,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            };
            for state in &mut states[(run_start + 1)..run_end] {
                *state = ToolGroupState::Hidden;
            }
        }

        i = run_end;
    }
}

pub(super) fn tool_group_recompute_start(blocks: &[RenderBlock], append_at: usize) -> usize {
    let mut start = append_at.min(blocks.len());
    while start > 0 && is_tool_group_block(&blocks[start - 1]) {
        start -= 1;
    }
    start
}

pub(super) fn recompute_tool_groups_tail(
    blocks: &[RenderBlock],
    states: &mut Vec<ToolGroupState>,
    start: usize,
) {
    let start = start.min(blocks.len());
    states.resize(blocks.len(), ToolGroupState::Normal);
    for state in &mut states[start..] {
        *state = ToolGroupState::Normal;
    }
    for (offset, state) in compute_tool_groups(&blocks[start..])
        .into_iter()
        .enumerate()
    {
        states[start + offset] = state;
    }
}

fn is_tool_group_block(block: &RenderBlock) -> bool {
    matches!(
        block,
        RenderBlock::ToolCall { .. } | RenderBlock::ToolResult { .. }
    )
}

/// Cap on per-tool detail rows a collapsed group renders before it folds the
/// rest into a `… +N more` line, so a big batch never floods the transcript
/// Live/error/small heterogeneous groups still show concrete rows. Repetitive
/// or larger successful history compresses to one action digest.
const TOOL_DETAIL_MAX_ROWS: u16 = 5;

/// Whether the group led by `leader` still has an in-flight member call.
/// SSOT for the live-vs-settled branch shared by the height measurement and
/// the line renderer, so both always agree (deriving it from the possibly
/// stale `Summary` counts could briefly disagree with the freshly scanned
/// rows).
pub(super) fn span_is_live(
    blocks: &[RenderBlock],
    states: &[ToolGroupState],
    leader: usize,
) -> bool {
    let end = group_span_end(blocks, states, leader);
    (leader..end).any(|idx| {
        (idx == leader || matches!(states[idx], ToolGroupState::Hidden))
            && is_inflight_call(&blocks[idx])
    })
}

/// Rows a collapsed group occupies. Live/error groups keep per-tool detail;
/// repetitive or large successful settled groups compress to one action digest.
/// Must agree with [`collapsed_tool_detail_lines`] so scroll math stays correct.
pub(super) fn collapsed_summary_height(
    blocks: &[RenderBlock],
    states: &[ToolGroupState],
    leader: usize,
) -> u16 {
    let live = span_is_live(blocks, states, leader);
    let detail_rows = collapsed_tool_detail_rows(blocks, states, leader, live);
    let line_count = if !live && compact_settled_group(&detail_rows) {
        1
    } else {
        detail_rows
            .len()
            .clamp(1, usize::from(TOOL_DETAIL_MAX_ROWS))
    };
    u16::try_from(line_count)
        .unwrap_or(TOOL_DETAIL_MAX_ROWS)
}

/// Short action verb for a tool, aligned into a column so targets line up.
fn tool_verb(name: &str) -> String {
    let known = match name {
        "read_file" | "Read" | "read" => Some("read"),
        "write_file" | "Write" | "write" => Some("write"),
        "edit_file" | "Edit" | "edit" => Some("edit"),
        "grep_search" | "Grep" => Some("grep"),
        "glob_search" | "Glob" => Some("glob"),
        "web_search" | "WebSearch" => Some("search"),
        "web_fetch" | "WebFetch" => Some("fetch"),
        "bash" | "Bash" => Some("bash"),
        "list_dir" | "LS" | "ls" => Some("ls"),
        "SpawnMultiAgent" | "Task" | "Agent" => Some("agent"),
        _ => None,
    };
    if let Some(verb) = known {
        return verb.to_string();
    }
    // Unknown / MCP tool (`mcp__server__tool`): trailing segment, else the bare
    // name — lowercased and clipped so it stays inside the verb column.
    let tail = name.rsplit("__").next().unwrap_or(name);
    let mut verb: String = tail.chars().take(8).collect();
    verb.make_ascii_lowercase();
    verb
}

/// Render a *settled* collapsed group as one compact `verb  target` line per
/// tool (capped, with a `… +N more` fold), instead of the merged
/// `N tools: +N read` one-liner. The leader line carries the group ✓/× marker;
/// a row whose result failed is tinted. Pure (returns lines) so it is testable
/// and so the draw site can render it through one scroll-aware `Paragraph`.
/// Append a right-aligned dim digest to an existing detail row's spans. Rides on
/// the SAME line (no new row, so the height contract with `collapsed_summary_height`
/// holds); dropped entirely when it would collide with the target on a narrow
/// terminal — the target wins. Failed-row digests are tinted with the error color.
fn push_digest_span(
    spans: &mut Vec<Span<'static>>,
    digest: &str,
    is_err: bool,
    width: u16,
    dim: Style,
    err_row_style: Style,
) {
    if digest.is_empty() {
        return;
    }
    let left_w: usize = spans
        .iter()
        .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
        .sum();
    let digest_w = UnicodeWidthStr::width(digest);
    let avail = usize::from(width).saturating_sub(left_w);
    if avail > digest_w {
        let digest_style = if is_err { err_row_style } else { dim };
        spans.push(Span::styled(" ".repeat(avail - digest_w), dim));
        spans.push(Span::styled(digest.to_string(), digest_style));
    }
}

#[derive(Clone, Debug)]
struct CollapsedToolDetailRow {
    verb: String,
    target: String,
    is_err: bool,
    digest: String,
    /// Per-row lifecycle for the live-group markers: the call's status, with
    /// "has a result" folded in (a call whose result landed is settled even if
    /// a stale status update lags behind).
    status: ToolCallStatus,
    has_result: bool,
}

impl CollapsedToolDetailRow {
    /// Whether this member tool is still in flight (no result yet).
    fn is_inflight(&self) -> bool {
        !self.has_result
            && matches!(
                self.status,
                ToolCallStatus::Pending | ToolCallStatus::Running
            )
    }
}

/// Small heterogeneous groups keep their concrete targets. Repetitive groups
/// and larger successful batches collapse to one digest; click-to-expand still
/// reveals every original call/result block. Errors never collapse this far.
fn compact_settled_group(rows: &[CollapsedToolDetailRow]) -> bool {
    if rows.iter().any(|row| row.is_err) {
        return false;
    }
    let same_verb = rows
        .first()
        .is_some_and(|first| rows.iter().all(|row| row.verb == first.verb));
    rows.len() >= 4 || (rows.len() >= 3 && same_verb)
}

/// `✓ 7 tools · read ×7` — a calm settled-history row. Targets and per-tool
/// digests remain available when the group is expanded.
fn settled_group_summary_line(
    rows: &[CollapsedToolDetailRow],
    theme: &Theme,
    width: u16,
) -> Line<'static> {
    let mut verbs: Vec<(String, usize)> = Vec::new();
    for row in rows {
        if let Some((_, count)) = verbs.iter_mut().find(|(verb, _)| verb == &row.verb) {
            *count += 1;
        } else {
            verbs.push((row.verb.clone(), 1));
        }
    }

    let multiplier = if theme.no_color { "x" } else { "×" };
    let actions = verbs
        .into_iter()
        .map(|(verb, count)| {
            if count == 1 {
                verb
            } else {
                format!("{verb} {multiplier}{count}")
            }
        })
        .collect::<Vec<_>>()
        .join(" · ");
    let label = format!("{} tools", rows.len());
    let fixed = 2 + UnicodeWidthStr::width(label.as_str()) + 3;
    let action_width = usize::from(width).saturating_sub(fixed);
    let actions = if action_width == 0 {
        String::new()
    } else {
        crate::tui::cards::truncate_to_width(&actions, action_width)
    };
    let dim = Style::new().fg(theme.palette.dim);
    let mut spans = vec![
        Span::styled(
            format!("{} ", if theme.no_color { "v" } else { "\u{2713}" }),
            Style::new().fg(theme.palette.success),
        ),
        Span::styled(
            label,
            Style::new()
                .fg(theme.palette.fg)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if !actions.is_empty() {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(actions, dim));
    }
    Line::from(spans)
}

/// Member rows for the group led by `leader`. `live` keeps *every* member row
/// (a still-running tool has no target/digest yet, but its animating marker is
/// the point), while a settled group drops non-informative rows so bare
/// `git`/`tool` verbs do not occupy history space.
fn collapsed_tool_detail_rows(
    blocks: &[RenderBlock],
    states: &[ToolGroupState],
    leader: usize,
    live: bool,
) -> Vec<CollapsedToolDetailRow> {
    let run_end = group_span_end(blocks, states, leader);

    let mut failed: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut bodies: std::collections::HashMap<&str, &ToolResultBody> =
        std::collections::HashMap::new();
    for block in &blocks[leader..run_end] {
        if let RenderBlock::ToolResult {
            tool_call_id,
            is_error,
            body,
            ..
        } = block
        {
            if *is_error {
                failed.insert(tool_call_id.0.as_str());
            }
            bodies.insert(tool_call_id.0.as_str(), body);
        }
    }

    let mut rows = Vec::new();
    for (offset, block) in blocks[leader..run_end].iter().enumerate() {
        let idx = leader + offset;
        let collapsed = idx == leader || matches!(states.get(idx), Some(ToolGroupState::Hidden));
        if !collapsed {
            continue;
        }
        let RenderBlock::ToolCall {
            name,
            summary,
            preview,
            tool_call_id,
            status,
            ..
        } = block
        else {
            continue;
        };

        let verb = tool_verb(name);
        let mut target = crate::tui::blocks::tool_call::display_tool_summary(name, summary);
        // Surface the read window (`path:start-end`) so two distinct windows
        // of one file no longer collapse to identical `read App.tsx` rows.
        if let ToolPreview::Read {
            range: Some((start, end)),
            ..
        } = preview
        {
            if !target.is_empty() && !target.contains(':') {
                target = format!("{target}:{start}-{end}");
            }
        }
        let id = tool_call_id.0.as_str();
        let is_err = failed.contains(id);
        let has_result = bodies.contains_key(id);
        let digest = bodies.get(id).map_or_else(String::new, |body| {
            crate::tui::blocks::tool_result::collapsed_group_digest(name, body, is_err)
        });

        // A successful generic tool with no target and no digest carries no
        // user-visible information once settled. While the group is live the
        // row stays: its animating status marker *is* the information, and
        // keeping it stabilizes the group height as results land.
        if !live && target.trim().is_empty() && digest.trim().is_empty() && !is_err {
            continue;
        }

        rows.push(CollapsedToolDetailRow {
            verb,
            target,
            is_err,
            digest,
            status: *status,
            has_result,
        });
    }
    rows
}

#[allow(clippy::too_many_lines)]
pub(super) fn collapsed_tool_detail_lines(
    blocks: &[RenderBlock],
    states: &[ToolGroupState],
    leader: usize,
    err_count: u16,
    theme: &Theme,
    width: u16,
) -> Vec<Line<'static>> {
    // `live` MUST come from the same SSOT as the height measurement
    // (`span_is_live`) so painted rows always match reserved rows.
    let live = span_is_live(blocks, states, leader);
    let rows = collapsed_tool_detail_rows(blocks, states, leader, live);

    if !live && compact_settled_group(&rows) {
        return vec![settled_group_summary_line(&rows, theme, width)];
    }

    let dim = Style::new().fg(theme.palette.dim);
    let target_style = Style::new().fg(theme.palette.fg);
    let err_row_style = Style::new().fg(theme.palette.error);
    let leader_style = if err_count > 0 {
        Style::new()
            .fg(theme.palette.error)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::new()
            .fg(theme.palette.success)
            .add_modifier(Modifier::BOLD)
    };
    let leader_marker = if err_count > 0 {
        if theme.no_color { "x" } else { "\u{00d7}" }
    } else if theme.no_color {
        "v"
    } else {
        "\u{2713}"
    };

    let verb_w = rows
        .iter()
        .map(|row| row.verb.chars().count())
        .max()
        .unwrap_or(4)
        .min(8);
    let max = TOOL_DETAIL_MAX_ROWS as usize;
    // Overflow: keep max-1 rows and spend the last line on the fold so the whole
    // group never exceeds TOOL_DETAIL_MAX_ROWS (matches `collapsed_summary_height`).
    let shown = if rows.len() > max { max - 1 } else { rows.len() };

    // Live-group marker styles. The marker glyph is time-based
    // (`spinner::marker_glyph`), so every running row pulses ✦/✧ in unison —
    // many parallel rows read as one calm spark heartbeat, not a per-row
    // rotation — while the transcript redraws each frame during a turn.
    let running_style = Style::new()
        .fg(theme.heat().spark)
        .add_modifier(Modifier::BOLD);
    let ok_style = Style::new().fg(theme.palette.success);
    let err_marker_style = Style::new()
        .fg(theme.palette.error)
        .add_modifier(Modifier::BOLD);

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(shown + 1);
    for (i, row) in rows.iter().take(shown).enumerate() {
        // Marker policy: a live group animates *per row* — spinner while that
        // tool runs, `○` while queued, flipping to a static `✓`/`×` in place
        // the moment its result lands (the CC "parallel batch settling row by
        // row" feel). A settled group keeps the single leader ✓/× so history
        // stays calm.
        let (marker, marker_style): (&str, Style) = if live {
            if row.is_inflight() {
                match row.status {
                    ToolCallStatus::Running => (
                        if theme.no_color {
                            "*"
                        } else {
                            crate::tui::spinner::marker_glyph()
                        },
                        running_style,
                    ),
                    _ => (if theme.no_color { "o" } else { "\u{25cb}" }, dim),
                }
            } else if row.is_err {
                (if theme.no_color { "x" } else { "\u{00d7}" }, err_marker_style)
            } else {
                (if theme.no_color { "v" } else { "\u{2713}" }, ok_style)
            }
        } else if i == 0 {
            (leader_marker, leader_style)
        } else {
            (" ", dim)
        };
        let pad = verb_w.saturating_sub(row.verb.chars().count());
        let verb_cell = format!("{}{}  ", row.verb, " ".repeat(pad));
        let mut spans = vec![
            Span::styled(format!("{marker} "), marker_style),
            Span::styled(verb_cell, dim),
        ];
        if !row.target.is_empty() {
            let row_style = if row.is_err { err_row_style } else { target_style };
            // Elide an over-long target (e.g. a deep absolute path) so the row stays
            // inside `width`. Without this it overflowed and the renderer hard-clipped
            // the tail — the transcript Paragraph has no `.wrap()` — swallowing the
            // path end AND the right-aligned `N ln` digest. The fixed cost is the
            // marker (`"{m} "` = 2), the verb cell (`verb_w` + 2 trailing) and the
            // digest with a 1-cell gap; we only truncate when (a) that cost still
            // fits the row, so eliding actually buys digest space, and (b) the target
            // is what overflows. Otherwise (hopelessly narrow row) we leave it as-is
            // and the existing drop-digest/clip fallback applies.
            let digest_w = if row.digest.is_empty() {
                0
            } else {
                UnicodeWidthStr::width(row.digest.as_str()) + 1
            };
            let fixed = 2 + verb_w + 2 + digest_w;
            let width_cells = usize::from(width);
            let target_w = UnicodeWidthStr::width(row.target.as_str());
            let target_fit = if fixed < width_cells && target_w > width_cells - fixed {
                crate::tui::cards::truncate_to_width(&row.target, width_cells - fixed)
            } else {
                row.target.clone()
            };
            spans.push(Span::styled(target_fit, row_style));
        }
        push_digest_span(&mut spans, &row.digest, row.is_err, width, dim, err_row_style);
        lines.push(Line::from(spans));
    }
    if rows.len() > shown {
        let more = rows.len() - shown;
        // Keep the fold informative while live: say how many of the folded
        // tools are still in flight so a big batch's tail progress is visible.
        let inflight_folded = rows
            .iter()
            .skip(shown)
            .filter(|row| row.is_inflight())
            .count();
        let fold = if live && inflight_folded > 0 {
            format!("  … +{more} more ({inflight_folded} running)")
        } else {
            format!("  … +{more} more")
        };
        lines.push(Line::from(Span::styled(fold, dim)));
    }
    if lines.is_empty() {
        // Defensive: a leader with no extractable rows still shows one line.
        lines.push(Line::from(Span::styled(
            format!("{leader_marker} {} tools", rows.len().max(1)),
            leader_style,
        )));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    use runtime::message_stream::{
        BlockId, DiffHunk, DiffLine, DiffLineKind, DiffView, ToolCallId, ToolPreview,
    };

    fn call(id: &str, name: &str) -> RenderBlock {
        RenderBlock::ToolCall {
            id: BlockId(0),
            tool_call_id: ToolCallId(id.to_string()),
            name: name.to_string(),
            summary: name.to_string(),
            preview: ToolPreview::Generic {
                name: name.to_string(),
                input_summary: String::new(),
            },
            status: ToolCallStatus::Ok,
        }
    }

    fn ok_result(id: &str) -> RenderBlock {
        RenderBlock::ToolResult {
            id: BlockId(0),
            tool_call_id: ToolCallId(id.to_string()),
            is_error: false,
            body: ToolResultBody::Bash(runtime::message_stream::BashResult {
                exit_code: 0,
                stdout: "ok".to_string(),
                stderr: String::new(),
                truncated: false,
            }),
        }
    }

    fn diff_result(id: &str) -> RenderBlock {
        RenderBlock::ToolResult {
            id: BlockId(0),
            tool_call_id: ToolCallId(id.to_string()),
            is_error: false,
            body: ToolResultBody::Diff(DiffView {
                old_path: Some("f.rs".to_string()),
                new_path: Some("f.rs".to_string()),
                language: Some("rust".to_string()),
                hunks: vec![DiffHunk {
                    old_start: 1,
                    old_lines: 1,
                    new_start: 1,
                    new_lines: 1,
                    lines: vec![DiffLine {
                        kind: DiffLineKind::Added,
                        text: "x".to_string(),
                    }],
                }],
            }),
        }
    }

    #[test]
    fn diff_pair_is_revealed_inline_and_excluded_from_summary() {
        // A batch of read/bash tools plus one edit (diff). Claude Code parity:
        // the diff call+result render inline (Normal), the other tools collapse
        // into a summary that counts only themselves — not the diff.
        let blocks = vec![
            call("a", "Read"),
            call("b", "Read"),
            call("c", "Edit"),
            ok_result("a"),
            ok_result("b"),
            diff_result("c"),
        ];
        let states = compute_tool_groups(&blocks);

        // The Edit call and its diff result must be visible inline.
        assert!(
            matches!(states[2], ToolGroupState::Normal),
            "edit call should be inline, got {:?}",
            states[2]
        );
        assert!(
            matches!(states[5], ToolGroupState::Normal),
            "diff result should be inline, got {:?}",
            states[5]
        );

        // Exactly one summary leader, counting only the two non-diff calls.
        let summary = states
            .iter()
            .find_map(|s| match s {
                ToolGroupState::Summary { total, .. } => Some(*total),
                _ => None,
            })
            .expect("a summary leader for the non-diff tools");
        assert_eq!(
            summary, 2,
            "summary must exclude the diff pair from its count"
        );
    }

    #[test]
    fn verbose_toggle_disables_grouping_globally() {
        struct RestoreGrouping;
        impl Drop for RestoreGrouping {
            fn drop(&mut self) {
                super::set_tool_groups_disabled(false);
            }
        }
        // The toggle is a process-global; hold the crate-wide test lock so
        // this cannot stomp parallel grouping tests (the same cross-module
        // discipline as env mutation), and restore on every exit path.
        let _lock = crate::test_env_lock();
        let _restore = RestoreGrouping;

        let blocks = vec![
            call("a", "Read"),
            call("b", "Grep"),
            ok_result("a"),
            ok_result("b"),
        ];
        assert!(
            compute_tool_groups(&blocks)
                .iter()
                .any(|s| matches!(s, ToolGroupState::Summary { .. })),
            "precondition: this run groups by default"
        );

        super::set_tool_groups_disabled(true);
        assert!(
            compute_tool_groups(&blocks)
                .iter()
                .all(|s| matches!(s, ToolGroupState::Normal)),
            "verbose mode must classify every block Normal"
        );

        // The incremental tail path must agree with the full recompute.
        let mut states = Vec::new();
        recompute_tool_groups_tail(&blocks, &mut states, 0);
        assert!(states.iter().all(|s| matches!(s, ToolGroupState::Normal)));
    }

    #[test]
    fn lone_non_diff_tool_beside_diff_is_not_summarized() {
        // One read + one edit: after the diff pair is revealed, only a single
        // non-diff tool remains — too few to summarize, so it shows inline too.
        let blocks = vec![
            call("a", "Read"),
            call("b", "Edit"),
            ok_result("a"),
            diff_result("b"),
        ];
        let states = compute_tool_groups(&blocks);
        assert!(
            states.iter().all(|s| matches!(s, ToolGroupState::Normal)),
            "no summary should form around a single non-diff tool: {states:?}"
        );
    }

    #[test]
    fn diffs_without_other_tools_all_render_inline() {
        // Two consecutive edits: both diffs inline, no summary at all.
        let blocks = vec![
            call("a", "Edit"),
            call("b", "Edit"),
            diff_result("a"),
            diff_result("b"),
        ];
        let states = compute_tool_groups(&blocks);
        assert!(
            states.iter().all(|s| matches!(s, ToolGroupState::Normal)),
            "pure-diff runs must never collapse: {states:?}"
        );
    }

    #[test]
    fn tool_breakdown_counts_in_summary() {
        let blocks = vec![
            call("a", "read_file"),
            call("b", "grep_search"),
            call("c", "bash"),
            call("d", "glob_search"),
            ok_result("a"),
            ok_result("b"),
            ok_result("c"),
            ok_result("d"),
        ];
        let states = compute_tool_groups(&blocks);

        // Find the summary and check breakdown counts
        let summary = states
            .iter()
            .find(|s| matches!(s, ToolGroupState::Summary { .. }))
            .expect("should find summary");

        if let ToolGroupState::Summary {
            total,
            read_count,
            search_count,
            exec_count,
            ..
        } = summary
        {
            assert_eq!(*total, 4);
            assert_eq!(*read_count, 1);
            assert_eq!(*search_count, 2); // grep_search + glob_search
            assert_eq!(*exec_count, 1); // bash
        } else {
            panic!("Expected a summary state");
        }
    }

    fn call_block(id: u64, name: &str, summary: &str) -> RenderBlock {
        RenderBlock::ToolCall {
            id: BlockId(id),
            tool_call_id: ToolCallId(format!("c{id}")),
            name: name.to_string(),
            summary: summary.to_string(),
            preview: ToolPreview::Generic {
                name: name.to_string(),
                input_summary: summary.to_string(),
            },
            status: ToolCallStatus::Ok,
        }
    }

    fn result_block(id: u64, call_id: u64, is_error: bool) -> RenderBlock {
        RenderBlock::ToolResult {
            id: BlockId(id),
            tool_call_id: ToolCallId(format!("c{call_id}")),
            is_error,
            body: ToolResultBody::Text {
                content: String::new(),
                truncated: false,
            },
        }
    }

    fn result_block_with_body(
        id: u64,
        call_id: u64,
        is_error: bool,
        body: ToolResultBody,
    ) -> RenderBlock {
        RenderBlock::ToolResult {
            id: BlockId(id),
            tool_call_id: ToolCallId(format!("c{call_id}")),
            is_error,
            body,
        }
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.to_string()).collect()
    }

    #[test]
    fn collapsed_group_omits_successful_rows_with_no_visible_detail() {
        // Regression: generic tools with no call summary and empty successful
        // results rendered as bare verbs (`git`, `tool`, …), producing the
        // empty-looking history rows in collapsed tool batches.
        let theme = Theme::default_dark();
        let blocks = vec![
            call_block(0, "git", "diff --stat"),
            result_block(1, 0, false),
            call_block(2, "git", ""),
            result_block(3, 2, false),
            call_block(4, "git", "{}"),
            result_block(5, 4, false),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 3,
                ok_count: 3,
                err_count: 0,
                running_count: 0,
                pending_count: 0,
                read_count: 0,
                search_count: 0,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            },
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
        ];

        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(text.len(), 1, "only the informative git row remains: {text:?}");
        assert!(
            text[0].contains("action") || text[0].contains("diff"),
            "the retained row still shows useful detail: {text:?}"
        );
        assert_eq!(
            collapsed_summary_height(&blocks, &states, 0),
            u16::try_from(lines.len()).unwrap(),
            "layout height matches hidden empty rows"
        );
    }

    #[test]
    fn collapsed_group_detail_appends_digest_and_keeps_row_count() {
        // Roadmap ⑤: a 2-tool settled group used to hide each result. The detail
        // rows now carry a right-aligned digest (grep hits / bash exit) WITHOUT
        // adding rows — the height contract with `collapsed_summary_height` holds.
        let theme = Theme::default_dark();
        let blocks = vec![
            call_block(0, "grep_search", r#"{"pattern":"x"}"#),
            result_block_with_body(
                1,
                0,
                false,
                ToolResultBody::Listing {
                    entries: vec!["m".to_string(); 12],
                    truncated: false,
                },
            ),
            call_block(2, "bash", r#"{"command":"make"}"#),
            result_block_with_body(
                3,
                2,
                true,
                ToolResultBody::Bash(runtime::message_stream::BashResult {
                    exit_code: 1,
                    stdout: String::new(),
                    stderr: "boom".to_string(),
                    truncated: false,
                }),
            ),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 2,
                ok_count: 1,
                err_count: 1,
                running_count: 0,
                pending_count: 0,
                read_count: 0,
                search_count: 1,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 1,
            },
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
        ];
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 1, &theme, 80);
        assert_eq!(
            lines.len(),
            2,
            "digest rides existing rows; row count is unchanged"
        );
        let text: String = lines
            .iter()
            .map(|l| line_text(l))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("12 hits"), "grep row shows hit count: {text}");
        assert!(
            text.contains("exit 1"),
            "failed bash row shows exit code: {text}"
        );
    }

    #[test]
    fn collapsed_group_digest_dropped_on_narrow_terminal_target_wins() {
        // Narrow width: digest is dropped (target stays). Row count still unchanged.
        let theme = Theme::default_dark();
        let blocks = vec![
            call_block(0, "grep_search", r#"{"pattern":"a-very-long-search-pattern-here"}"#),
            result_block_with_body(
                1,
                0,
                false,
                ToolResultBody::Listing {
                    entries: vec!["m".to_string(); 99],
                    truncated: false,
                },
            ),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 1,
                ok_count: 1,
                err_count: 0,
                running_count: 0,
                pending_count: 0,
                read_count: 0,
                search_count: 1,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            },
            ToolGroupState::Hidden,
        ];
        // A single-tool "group" of one is still rendered as one detail row.
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 12);
        assert_eq!(lines.len(), 1, "row count unchanged on narrow width");
        let text = line_text(&lines[0]);
        assert!(
            !text.contains("99 hits"),
            "digest dropped when it would collide with target: {text}"
        );
    }

    #[test]
    fn long_path_is_elided_so_the_digest_survives_within_width() {
        // Regression ("뒷부분이 짤린다"): a deep absolute path overflowed the row and
        // the renderer (no `.wrap()`) hard-clipped the tail, dropping the path end
        // AND the right-aligned `N ln` digest. Now the path is elided with `…`, the
        // digest stays visible, and the whole row fits inside `width`.
        let theme = Theme::default_dark();
        let long_path =
            "/Users/joe/2026/zo/crates/zo-cli/src/tui/transcript/tool_groups.rs";
        let blocks = vec![
            call_block(0, "read_file", &format!(r#"{{"file_path":"{long_path}"}}"#)),
            result_block_with_body(
                1,
                0,
                false,
                ToolResultBody::Read {
                    path: long_path.to_string(),
                    content: "x\n".repeat(64),
                    language: None,
                    truncated: false,
                },
            ),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 1,
                ok_count: 1,
                err_count: 0,
                running_count: 0,
                pending_count: 0,
                read_count: 1,
                search_count: 0,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            },
            ToolGroupState::Hidden,
        ];
        // `display_tool_summary` already compacts the absolute path to a repo
        // relative one (~57 cells); pick a width where even that overflows so the
        // row genuinely needs eliding (the real bug fires on deep paths / narrow
        // panes).
        let width: u16 = 50;
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, width);
        assert_eq!(lines.len(), 1);
        let text = line_text(&lines[0]);
        assert!(text.contains('\u{2026}'), "long path is elided: {text}");
        assert!(text.contains(" ln"), "digest survives the elision: {text}");
        assert!(
            UnicodeWidthStr::width(text.as_str()) <= usize::from(width),
            "row fits inside width ({} > {width}): {text}",
            UnicodeWidthStr::width(text.as_str())
        );
    }

    fn read_call_block(id: u64, path: &str, range: Option<(u64, u64)>) -> RenderBlock {
        RenderBlock::ToolCall {
            id: BlockId(id),
            tool_call_id: ToolCallId(format!("c{id}")),
            name: "read_file".to_string(),
            summary: format!(r#"{{"path":"{path}"}}"#),
            preview: ToolPreview::Read {
                path: path.to_string(),
                range,
            },
            status: ToolCallStatus::Ok,
        }
    }

    #[test]
    fn collapsed_group_distinguishes_windowed_reads_of_one_file() {
        // Regression: two distinct *windows* of one file (a model paging an 80-line
        // file) collapsed to identical `read App.tsx` rows — the "read the same
        // file 4 times" illusion in the screenshot. The collapsed row now appends
        // the preview's `:start-end` range, so each window is a distinct row.
        let theme = Theme::default_dark();
        let blocks = vec![
            read_call_block(0, "src/App.tsx", Some((1, 80))),
            result_block(1, 0, false),
            read_call_block(2, "src/App.tsx", Some((81, 160))),
            result_block(3, 2, false),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 2,
                ok_count: 2,
                err_count: 0,
                running_count: 0,
                pending_count: 0,
                read_count: 2,
                search_count: 0,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            },
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
        ];
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(text.len(), 2, "one detail row per windowed read: {text:?}");
        assert!(
            text[0].contains("App.tsx:1-80"),
            "first window carries its line range: {text:?}"
        );
        assert!(
            text[1].contains("App.tsx:81-160"),
            "second window carries its distinct range: {text:?}"
        );
        assert_ne!(
            text[0].trim_start_matches(['\u{2713}', ' ']),
            text[1].trim_start_matches(['\u{2713}', ' ']),
            "the two windowed reads no longer render as identical rows: {text:?}"
        );
    }

    #[test]
    fn collapsed_group_rangeless_read_keeps_bare_path() {
        // A whole-file read (no window) carries no range, so the row stays the bare
        // `read <path>` — the range suffix is only added when a window exists.
        let theme = Theme::default_dark();
        let blocks = vec![
            read_call_block(0, "src/lib.rs", None),
            result_block(1, 0, false),
            read_call_block(2, "src/main.rs", None),
            result_block(3, 2, false),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 2,
                ok_count: 2,
                err_count: 0,
                running_count: 0,
                pending_count: 0,
                read_count: 2,
                search_count: 0,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            },
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
        ];
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert!(
            text.iter().any(|l| l.contains("lib.rs") && !l.contains(':')),
            "a rangeless read keeps the bare path (no stray `:`): {text:?}"
        );
    }

    #[test]
    fn settled_group_renders_one_detail_row_per_tool_not_a_merged_count() {
        let theme = Theme::default_dark();
        // read app.rs + grep "reveal" — a 2-tool settled cluster.
        let blocks = vec![
            call_block(0, "read_file", r#"{"file_path":"src/app.rs"}"#),
            result_block(1, 0, false),
            call_block(2, "grep_search", r#"{"pattern":"reveal"}"#),
            result_block(3, 2, false),
        ];
        let states = vec![
            ToolGroupState::Summary {
                total: 2,
                ok_count: 2,
                err_count: 0,
                running_count: 0,
                pending_count: 0,
                read_count: 1,
                search_count: 1,
                web_search_count: 0,
                fetch_count: 0,
                exec_count: 0,
            },
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
            ToolGroupState::Hidden,
        ];
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        let text: Vec<String> = lines.iter().map(line_text).collect();

        assert_eq!(
            text.len(),
            2,
            "one detail row per tool, not a single merged count: {text:?}"
        );
        assert!(text[0].contains("read"), "row 1 shows the verb: {text:?}");
        assert!(text[1].contains("grep"), "row 2 shows the verb: {text:?}");
        assert!(
            text.iter().any(|l| l.contains("app.rs")),
            "the actual read target is surfaced, not hidden behind a count: {text:?}"
        );
        assert!(
            text.iter().any(|l| l.contains("reveal")),
            "the actual search target is surfaced: {text:?}"
        );
        assert!(
            text[0].starts_with('\u{2713}'),
            "the leader row carries the group done marker: {text:?}"
        );
    }

    #[test]
    fn large_settled_group_compacts_to_one_action_digest() {
        let theme = Theme::default_dark();
        // Seven successful reads are still fully expandable, but settled
        // history should not spend five permanent transcript rows on them.
        let mut blocks = Vec::new();
        let mut states = Vec::new();
        for i in 0..7u64 {
            blocks.push(call_block(
                i * 2,
                "read_file",
                &format!(r#"{{"file_path":"f{i}.rs"}}"#),
            ));
            blocks.push(result_block(i * 2 + 1, i * 2, false));
        }
        states.push(ToolGroupState::Summary {
            total: 7,
            ok_count: 7,
            err_count: 0,
            running_count: 0,
            pending_count: 0,
            read_count: 7,
            search_count: 0,
            web_search_count: 0,
            fetch_count: 0,
            exec_count: 0,
        });
        for _ in 1..blocks.len() {
            states.push(ToolGroupState::Hidden);
        }

        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        assert_eq!(lines.len(), 1, "a large successful batch uses one line");
        let summary = line_text(&lines[0]);
        assert!(
            summary.contains("7 tools") && summary.contains("read ×7"),
            "the compact row keeps count and action: {summary:?}"
        );

        // Height accounting must agree with the rendered line count.
        assert_eq!(
            collapsed_summary_height(&blocks, &states, 0),
            1,
            "layout height matches the compact digest"
        );
    }

    fn call_running(id: &str, name: &str, status: ToolCallStatus) -> RenderBlock {
        RenderBlock::ToolCall {
            id: BlockId(0),
            tool_call_id: ToolCallId(id.to_string()),
            name: name.to_string(),
            summary: name.to_string(),
            preview: ToolPreview::Generic {
                name: name.to_string(),
                input_summary: String::new(),
            },
            status,
        }
    }

    #[test]
    fn live_group_rows_carry_per_tool_status_markers() {
        // CC parity: while a parallel batch runs, each member row shows its own
        // lifecycle — settled ✓, running spinner, queued ○ — instead of one
        // merged "N tools active" line.
        let theme = Theme::default_dark();
        let blocks = vec![
            call("a", "read_file"),
            call_running("b", "grep_search", ToolCallStatus::Running),
            call_running("c", "bash", ToolCallStatus::Pending),
            ok_result("a"),
        ];
        let states = compute_tool_groups(&blocks);
        assert!(
            matches!(
                states[0],
                ToolGroupState::Summary {
                    total: 3,
                    running_count: 1,
                    pending_count: 1,
                    ..
                }
            ),
            "the live batch forms one leader: {states:?}"
        );

        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert_eq!(text.len(), 3, "one row per member tool: {text:?}");
        assert!(
            text[0].starts_with('\u{2713}'),
            "the settled member flips to ✓ in place: {text:?}"
        );
        assert!(
            text[1].starts_with(crate::tui::glyphs::ZO_SPARK)
                || text[1].starts_with(crate::tui::glyphs::ZO_SPARK_HOLLOW),
            "the running member pulses the spark marker: {text:?}"
        );
        assert!(
            text[2].starts_with('\u{25cb}'),
            "the queued member shows the pending ring: {text:?}"
        );

        // Height contract: reserved rows equal painted rows for live groups.
        assert_eq!(
            collapsed_summary_height(&blocks, &states, 0),
            u16::try_from(lines.len()).unwrap(),
            "live-group layout height matches its rendered rows"
        );
    }

    #[test]
    fn live_group_keeps_rows_without_targets_for_stable_height() {
        // A running generic tool has no target/digest yet. While the group is
        // live its row must still be reserved (the marker is the signal) so the
        // group height does not thrash as results land.
        let theme = Theme::default_dark();
        let blocks = vec![
            call_running("a", "git", ToolCallStatus::Running),
            call_running("b", "tool", ToolCallStatus::Running),
        ];
        let states = compute_tool_groups(&blocks);
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        assert_eq!(
            lines.len(),
            2,
            "live rows without visible detail still occupy their row"
        );
        assert_eq!(
            collapsed_summary_height(&blocks, &states, 0),
            2,
            "height matches while live"
        );
    }

    #[test]
    fn mid_batch_diff_pops_inline_while_group_stays_live() {
        // An edit lands mid-batch while siblings still run: its diff renders
        // inline immediately (CC parity — the diff is settled the moment it
        // exists), and the remaining in-flight tools keep animating as a live
        // group whose counts exclude the diff pair.
        let blocks = vec![
            call("e", "Edit"),
            call_running("r1", "read_file", ToolCallStatus::Running),
            call_running("r2", "grep_search", ToolCallStatus::Running),
            diff_result("e"),
        ];
        let states = compute_tool_groups(&blocks);
        assert!(
            matches!(states[0], ToolGroupState::Normal),
            "the settled edit call renders inline: {states:?}"
        );
        assert!(
            matches!(states[3], ToolGroupState::Normal),
            "its diff result renders inline mid-batch: {states:?}"
        );
        let leader = states
            .iter()
            .find_map(|state| match state {
                ToolGroupState::Summary {
                    total,
                    running_count,
                    ..
                } => Some((*total, *running_count)),
                _ => None,
            })
            .expect("the in-flight members still form a live group");
        assert_eq!(
            leader,
            (2, 2),
            "the live leader counts only the non-diff in-flight tools: {states:?}"
        );
    }

    #[test]
    fn inflight_spawn_call_is_exempt_from_live_grouping() {
        // A delegation host renders its own live agent tree under its row;
        // folding it into the live group would hide that tree. Sibling plain
        // tools still group.
        let blocks = vec![
            call_running("s", "SpawnMultiAgent", ToolCallStatus::Running),
            call_running("r1", "read_file", ToolCallStatus::Running),
            call_running("r2", "grep_search", ToolCallStatus::Running),
        ];
        let states = compute_tool_groups(&blocks);
        assert!(
            matches!(states[0], ToolGroupState::Normal),
            "the in-flight spawn host stays a normal row (live agent tree): {states:?}"
        );
        assert!(
            matches!(
                states[1],
                ToolGroupState::Summary {
                    total: 2,
                    running_count: 2,
                    ..
                }
            ),
            "the plain in-flight siblings still form a live group: {states:?}"
        );
        assert!(
            matches!(states[2], ToolGroupState::Hidden),
            "the second sibling collapses under the live leader: {states:?}"
        );
    }

    #[test]
    fn live_group_fold_reports_folded_running_tools() {
        // 7-wide live batch: the fold line names how many of the folded tools
        // are still running so tail progress stays visible.
        let theme = Theme::default_dark();
        let blocks: Vec<RenderBlock> = (0..7)
            .map(|i| {
                call_running(
                    &format!("t{i}"),
                    "read_file",
                    ToolCallStatus::Running,
                )
            })
            .collect();
        let states = compute_tool_groups(&blocks);
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        assert_eq!(
            u16::try_from(lines.len()).unwrap(),
            TOOL_DETAIL_MAX_ROWS,
            "a live batch is capped at TOOL_DETAIL_MAX_ROWS lines"
        );
        let last = line_text(&lines[lines.len() - 1]);
        assert!(
            last.contains("+3 more") && last.contains("running"),
            "the live fold names the folded running tools: {last:?}"
        );
        assert_eq!(
            collapsed_summary_height(&blocks, &states, 0),
            TOOL_DETAIL_MAX_ROWS,
            "live height contract holds at the cap"
        );
    }

    #[test]
    fn settled_group_after_live_batch_drops_live_markers() {
        // Once every member settles, the same rows re-render as calm history:
        // leader ✓ plus indented rows, no per-row spinners.
        let theme = Theme::default_dark();
        let blocks = vec![
            call("a", "read_file"),
            call("b", "grep_search"),
            ok_result("a"),
            ok_result("b"),
        ];
        let states = compute_tool_groups(&blocks);
        let lines = collapsed_tool_detail_lines(&blocks, &states, 0, 0, &theme, 80);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        assert!(
            text[0].starts_with('\u{2713}'),
            "settled leader carries the group ✓: {text:?}"
        );
        assert!(
            text[1].starts_with(' '),
            "settled follower rows stay unmarked: {text:?}"
        );
        for line in &text {
            assert!(
                !line.contains(crate::tui::glyphs::ZO_SPARK)
                    && !line.contains(crate::tui::glyphs::ZO_SPARK_HOLLOW),
                "no pulse marker survives in settled history: {line:?}"
            );
        }
    }
}
