//! Live agent-batch state for Spawn-family tool calls (Claude Code parity).
//!
//! `SpawnMultiAgent`/`Task`/`Agent` calls render a per-agent tree under their
//! transcript row: spawn-order rows, each flipping to `⎿ Done` the moment that
//! agent's completion lands (completion order), and a
//! `N agents finished (ctrl+g for details)` header once the call returns.
//!
//! Data flow: manifests (polled into [`AgentTaskSummary`]) fill the live rows;
//! the completion channel flips terminal states instantly; `push_block` opens
//! the batch on the `ToolCall` and seals it on the `ToolResult`. The tree
//! itself is stored in the transcript's side table
//! (`Transcript::set_agent_tree`), so this module only owns the merge state.

use super::App;
use crate::tui::blocks::tool_call::{AgentTree, AgentTreeRow};
use crate::tui::hud::AgentTaskSummary;

/// Merge state for the batch owned by one Spawn-family tool call.
#[derive(Debug, Default)]
pub(crate) struct ActiveAgentBatch {
    /// `ToolCallId.0` of the owning Spawn-family call.
    tool_call_id: String,
    /// Spawn-order rows (see [`AgentTreeRow::created_at`] for the sort key).
    rows: Vec<AgentTreeRow>,
    /// Optional provenance label shown in the tree header (for host Smart
    /// prelude batches); model-invoked batches leave it unset.
    batch_label: Option<String>,
    /// Next completion sequence number (1-based).
    done_seq: u32,
    /// True once the owning tool call returned its result.
    finished: bool,
}

fn is_terminal(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "stopped")
}

fn matches_agent_identity(row: &AgentTreeRow, agent_id: &str, name: &str) -> bool {
    if !agent_id.is_empty() {
        return row.agent_id == agent_id;
    }
    !name.is_empty() && row.name == name
}

impl ActiveAgentBatch {
    fn new(tool_call_id: &str, batch_label: Option<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.to_string(),
            batch_label,
            ..Self::default()
        }
    }

    fn tree(&self) -> AgentTree {
        AgentTree {
            rows: self.rows.clone(),
            batch_label: self.batch_label.clone(),
            finished: self.finished,
        }
    }

    fn next_done_order(&mut self) -> u32 {
        self.done_seq = self.done_seq.saturating_add(1);
        self.done_seq
    }

    /// Merge one manifest summary into the batch rows. Returns `true` when
    /// anything changed (drives the cheap no-op detection in the transcript).
    fn merge_summary(&mut self, summary: &AgentTaskSummary, allow_new: bool) -> bool {
        if summary.id.is_empty() {
            return false;
        }
        let idx = self.rows.iter().position(|row| row.agent_id == summary.id);
        let Some(idx) = idx else {
            // New agents join only when the caller resolved this batch as the
            // right home: the collecting-batch fallback only ever passes an
            // unfinished batch, and a `toolCallId`-stamped manifest may join
            // its own batch even after it sealed (the stamp proves ownership —
            // a fast fan-out's manifests can land after the call returned).
            if !allow_new {
                return false;
            }
            self.rows.push(AgentTreeRow {
                agent_id: summary.id.clone(),
                name: summary.name.clone(),
                model: summary.model.clone(),
                status: summary.status.clone(),
                subagent_type: summary.subagent_type.clone(),
                tool_calls: summary.tool_calls,
                tokens: summary.tokens,
                elapsed_secs: summary.elapsed_secs,
                activity: summary.activity_label().map(str::to_string),
                output_tail: summary.output_tail.clone(),
                done_order: None,
                created_at: summary.created_at,
                route_reason: summary.route_reason.clone(),
            });
            // Manifest listings arrive newest-first; the tree shows spawn order.
            self.rows
                .sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.name.cmp(&b.name)));
            return true;
        };
        let row = &mut self.rows[idx];
        let mut changed = false;
        if summary.tool_calls.is_some() && row.tool_calls != summary.tool_calls {
            row.tool_calls = summary.tool_calls;
            changed = true;
        }
        if summary.tokens > row.tokens {
            row.tokens = summary.tokens;
            changed = true;
        }
        // Unconditional (not gated on running/terminal): the route reason is a
        // spawn-time fact that may arrive a poll or two after the row is first
        // created, and should stay visible once the agent finishes rather than
        // being cleared like the live-only fields below.
        if summary.route_reason.is_some() && row.route_reason != summary.route_reason {
            row.route_reason.clone_from(&summary.route_reason);
            changed = true;
        }
        let was_terminal = is_terminal(&row.status);
        if !was_terminal && row.status != summary.status {
            row.status.clone_from(&summary.status);
            if is_terminal(&summary.status) {
                self.rows[idx].activity = None;
                let order = self.next_done_order();
                self.rows[idx].done_order = Some(order);
                return true;
            }
            changed = true;
        }
        let row = &mut self.rows[idx];
        if !is_terminal(&row.status) {
            if row.elapsed_secs != summary.elapsed_secs {
                row.elapsed_secs = summary.elapsed_secs;
                changed = true;
            }
            let activity = summary.activity_label().map(str::to_string);
            if row.activity != activity {
                row.activity = activity;
                changed = true;
            }
            if row.output_tail != summary.output_tail {
                row.output_tail.clone_from(&summary.output_tail);
                changed = true;
            }
        }
        changed
    }
    fn contains_agent(&self, agent_id: &str, name: &str) -> bool {
        self.rows
            .iter()
            .any(|row| matches_agent_identity(row, agent_id, name))
    }

    fn note_completion(
        &mut self,
        agent_id: &str,
        name: &str,
        status: &str,
        output_tokens: u64,
        allow_seed: bool,
    ) -> bool {
        let idx = self
            .rows
            .iter()
            .position(|row| matches_agent_identity(row, agent_id, name));
        let Some(idx) = idx else {
            if !allow_seed || self.finished || !is_terminal(status) {
                return false;
            }
            // Completion can outrun the first manifest poll. Seed a terminal
            // row while this batch is the current open collection window so
            // fast fan-out agents remain visible; sealed/non-collecting batches
            // must not absorb unrelated late events.
            let order = self.next_done_order();
            self.rows.push(AgentTreeRow {
                agent_id: agent_id.to_string(),
                name: if name.is_empty() {
                    agent_id.to_string()
                } else {
                    name.to_string()
                },
                status: status.to_string(),
                tokens: output_tokens,
                done_order: Some(order),
                ..AgentTreeRow::default()
            });
            return true;
        };
        let row = &mut self.rows[idx];
        if is_terminal(&row.status) {
            return true;
        }
        if !is_terminal(status) {
            return false;
        }
        row.status = status.to_string();
        if output_tokens > row.tokens {
            row.tokens = output_tokens;
        }
        row.activity = None;
        let order = self.next_done_order();
        self.rows[idx].done_order = Some(order);
        true
    }
}

impl App {
    /// Open (or keep) the live batch for a Spawn-family tool call that just
    /// started.
    pub fn begin_agent_batch(&mut self, tool_call_id: &str) {
        self.begin_agent_batch_with_label(tool_call_id, None);
    }

    /// Open a live batch with an optional provenance label in its rendered
    /// header. Use this for host-created Smart preludes; plain model-invoked
    /// spawn-family calls should keep using [`Self::begin_agent_batch`].
    pub fn begin_agent_batch_with_label(&mut self, tool_call_id: &str, batch_label: Option<&str>) {
        if let Some(index) = self.batch_index(tool_call_id) {
            let batch = &mut self.agent_batches[index];
            if let Some(label) = batch_label.filter(|label| !label.is_empty()) {
                if batch.batch_label.as_deref() != Some(label) {
                    batch.batch_label = Some(label.to_string());
                    self.sync_agent_batch(index);
                }
            }
            return;
        }
        self.agent_batches.push(ActiveAgentBatch::new(
            tool_call_id,
            batch_label.map(str::to_string),
        ));
    }

    /// Label for the most recent active agent batch, if it has one.
    pub(crate) fn active_agent_batch_label(&self) -> Option<&str> {
        self.agent_batches
            .iter()
            .rev()
            .find_map(|batch| batch.batch_label.as_deref())
    }

    /// Seal the batch when its owning tool call returns: the transcript row
    /// flips to the `N agents finished` header. Late completions still update
    /// row states (stragglers from a closed collection window).
    pub fn finish_agent_batch(&mut self, tool_call_id: &str) {
        let Some(index) = self.batch_index(tool_call_id) else {
            return;
        };
        let batch = &mut self.agent_batches[index];
        if batch.finished {
            return;
        }
        batch.finished = true;
        self.sync_agent_batch(index);
    }

    /// Fold the latest manifest poll into the live tree. Called wherever the
    /// sidebar's agent list is refreshed, so the transcript tree and sidebar
    /// always read from the same scan.
    pub(crate) fn refresh_agent_batch(&mut self, agents: &[AgentTaskSummary]) {
        if self.agent_batches.is_empty() {
            return;
        }
        let mut changed = Vec::new();
        for summary in agents {
            let mut matched_existing = false;
            for index in 0..self.agent_batches.len() {
                if self.agent_batches[index].contains_agent(&summary.id, &summary.name) {
                    matched_existing = true;
                    if self.agent_batches[index].merge_summary(summary, false) {
                        changed.push(index);
                    }
                }
            }
            if !matched_existing {
                // A `toolCallId`-stamped manifest routes straight to the batch
                // of the delegation call that spawned it, and NEVER leaks
                // elsewhere: with two concurrent Spawn-family calls open, the
                // collecting-batch fallback used to dump every new agent into
                // the first unfinished batch. A stamp whose batch is gone
                // (another turn's call) drops the summary instead of polluting
                // whichever batch happens to be collecting now. Unstamped
                // (legacy/host-spawned) manifests keep the fallback.
                let index = match summary.tool_call_id.as_deref() {
                    Some(id) => match self.batch_index(id) {
                        Some(index) => index,
                        None => continue,
                    },
                    None => match self.current_collecting_batch_index() {
                        Some(index) => index,
                        None => continue,
                    },
                };
                if self.agent_batches[index].merge_summary(summary, true) {
                    changed.push(index);
                }
            }
        }
        changed.sort_unstable();
        changed.dedup();
        for index in changed {
            self.sync_agent_batch(index);
        }
    }

    /// Flip one agent terminal the instant its completion event lands
    /// (completion order — this is what makes `⎿ Done` appear per agent while
    /// the batch is still collecting). Returns `true` when the event was
    /// absorbed by the tree, so callers can skip the redundant system line.
    pub fn note_agent_completion_display(
        &mut self,
        agent_id: &str,
        name: &str,
        status: &str,
        output_tokens: u64,
    ) -> bool {
        for index in 0..self.agent_batches.len() {
            if self.agent_batches[index].contains_agent(agent_id, name) {
                let absorbed = self.agent_batches[index].note_completion(
                    agent_id,
                    name,
                    status,
                    output_tokens,
                    false,
                );
                if absorbed {
                    self.sync_agent_batch(index);
                }
                return absorbed;
            }
        }
        let Some(index) = self.current_collecting_batch_index() else {
            return false;
        };
        let absorbed = self.agent_batches[index].note_completion(
            agent_id,
            name,
            status,
            output_tokens,
            true,
        );
        if absorbed {
            self.sync_agent_batch(index);
        }
        absorbed
    }

    fn batch_index(&self, tool_call_id: &str) -> Option<usize> {
        self.agent_batches
            .iter()
            .position(|batch| batch.tool_call_id == tool_call_id)
    }

    fn current_collecting_batch_index(&self) -> Option<usize> {
        self.agent_batches.iter().position(|batch| !batch.finished)
    }

    /// Push the selected batch state into the transcript side table.
    fn sync_agent_batch(&mut self, index: usize) {
        let Some(batch) = self.agent_batches.get(index) else {
            return;
        };
        let tree = batch.tree();
        let tool_call_id = batch.tool_call_id.clone();
        self.transcript.set_agent_tree(&tool_call_id, tree);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::AgentCommand;
    use crate::tui::theme::Theme;
    use runtime::message_stream::{
        BlockIdGen, RenderBlock, ToolCallId, ToolCallStatus, ToolPreview,
    };
    use tokio::sync::mpsc;

    fn test_app() -> App {
        let (_block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
        let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
        App::new(Theme::no_color(), block_rx, cmd_tx)
    }

    fn summary(id: &str, status: &str, created_at: u64) -> AgentTaskSummary {
        AgentTaskSummary {
            id: id.to_string(),
            name: id.to_string(),
            status: status.to_string(),
            model: "haiku".to_string(),
            tool_calls: Some(3),
            tokens: 1_000,
            created_at: Some(created_at),
            ..AgentTaskSummary::default()
        }
    }

    fn summary_named(id: &str, name: &str, status: &str, created_at: u64) -> AgentTaskSummary {
        AgentTaskSummary {
            name: name.to_string(),
            ..summary(id, status, created_at)
        }
    }

    fn spawn_tool_call(app: &mut App, ids: &BlockIdGen, call: &str) {
        app.push_block(RenderBlock::ToolCall {
            id: ids.next(),
            tool_call_id: ToolCallId(call.to_string()),
            name: "SpawnMultiAgent".to_string(),
            summary: String::new(),
            preview: ToolPreview::Generic {
                name: "SpawnMultiAgent".to_string(),
                input_summary: "delegating".to_string(),
            },
            status: ToolCallStatus::Running,
        });
    }

    #[test]
    fn begin_agent_batch_with_label_updates_existing_batch_opened_by_tool_call() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-smart");

        // `push_block` opens the batch first via `begin_agent_batch(..., None)`.
        // Smart prelude provenance arrives immediately after and must update the
        // active batch instead of early-returning with `batch_label=None`.
        app.begin_agent_batch_with_label("call-smart", Some("Smart"));
        assert_eq!(app.active_agent_batch_label(), Some("Smart"));

        app.refresh_agent_batch(&[summary("a", "running", 100)]);
        let tree = app
            .transcript_mut()
            .agent_tree("call-smart")
            .cloned()
            .expect("tree");
        assert_eq!(tree.batch_label.as_deref(), Some("Smart"));
    }

    /// 동시 멀티위임: `toolCallId` 스탬프가 있는 manifest 는 첫 수집 배치가
    /// 아니라 자신을 스폰한 호출의 배치로 귀속되고, 스탬프가 가리키는 배치가
    /// 사라졌으면 어느 배치에도 새지 않는다.
    #[test]
    fn stamped_summary_routes_to_owning_batch() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-1");
        spawn_tool_call(&mut app, &ids, "call-2");

        let mut owned = summary("agent-2", "running", 200);
        owned.tool_call_id = Some("call-2".to_string());
        let mut orphan = summary("agent-x", "running", 300);
        orphan.tool_call_id = Some("call-gone".to_string());
        app.refresh_agent_batch(&[owned, orphan]);

        let tree_2 = app
            .transcript_mut()
            .agent_tree("call-2")
            .cloned()
            .expect("tree");
        assert_eq!(tree_2.rows.len(), 1, "stamped agent joins its own batch");
        assert_eq!(tree_2.rows[0].agent_id, "agent-2");
        // The first (collecting-fallback) batch stays empty: neither the
        // stamped agent nor the orphan of a vanished call leaks into it.
        assert!(
            app.transcript_mut()
                .agent_tree("call-1")
                .is_none_or(|tree| tree.rows.is_empty()),
            "no leak into the first collecting batch"
        );
    }

    /// 빠른 fan-out: 호출이 이미 반환(finished)된 뒤 첫 manifest 폴이 도착해도
    /// `toolCallId` 스탬프가 소유를 증명하므로 자신의 배치에 합류한다.
    #[test]
    fn stamped_summary_joins_its_finished_batch() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-1");
        app.finish_agent_batch("call-1");

        let mut owned = summary("agent-late", "completed", 100);
        owned.tool_call_id = Some("call-1".to_string());
        app.refresh_agent_batch(&[owned]);

        let tree = app
            .transcript_mut()
            .agent_tree("call-1")
            .cloned()
            .expect("tree");
        assert_eq!(tree.rows.len(), 1, "stamp proves ownership past finish");
        assert_eq!(tree.rows[0].agent_id, "agent-late");
    }

    /// 끝나는 순서대로: 채널 완료 이벤트가 manifest 폴링보다 먼저/뒤 어느 쪽이든
    /// `done_order` 는 도착 순서를 기록하고, 스폰 순서(`created_at`)는 행 순서를
    /// 유지한다.
    #[test]
    fn batch_records_completion_order_independent_of_spawn_order() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-1");

        // manifest 폴링은 newest-first 로 도착한다 — 트리는 스폰 순서.
        app.refresh_agent_batch(&[summary("b", "running", 200), summary("a", "running", 100)]);
        let tree = app
            .transcript_mut()
            .agent_tree("call-1")
            .cloned()
            .expect("tree");
        assert_eq!(tree.rows[0].agent_id, "a", "spawn order by created_at");
        assert_eq!(tree.rows[1].agent_id, "b");

        // b 가 먼저 끝난다 (완료 순서 ≠ 스폰 순서).
        assert!(app.note_agent_completion_display("b", "b", "completed", 2_000));
        assert!(app.note_agent_completion_display("a", "a", "failed", 0));
        let tree = app
            .transcript_mut()
            .agent_tree("call-1")
            .cloned()
            .expect("tree");
        assert_eq!(tree.rows[0].agent_id, "a");
        assert_eq!(tree.rows[0].done_order, Some(2));
        assert_eq!(tree.rows[0].status, "failed");
        assert_eq!(tree.rows[1].done_order, Some(1), "b finished first");
        assert_eq!(tree.rows[1].tokens, 2_000, "completion tokens win");

        // ToolResult 가 배치를 봉인 → finished 헤더 상태.
        app.push_block(RenderBlock::ToolResult {
            id: ids.next(),
            tool_call_id: ToolCallId("call-1".to_string()),
            is_error: false,
            body: runtime::message_stream::ToolResultBody::Text {
                content: "ok".to_string(),
                truncated: false,
            },
        });
        let tree = app
            .transcript_mut()
            .agent_tree("call-1")
            .cloned()
            .expect("tree");
        assert!(tree.finished);
    }

    #[test]
    fn completion_before_manifest_poll_seeds_visible_row_while_batch_is_open() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-3");

        assert!(app.note_agent_completion_display(
            "agent-fast",
            "fast reviewer",
            "completed",
            42,
        ));
        let tree = app
            .transcript_mut()
            .agent_tree("call-3")
            .cloned()
            .expect("tree");
        assert!(!tree.finished);
        assert_eq!(tree.rows.len(), 1);
        assert_eq!(tree.rows[0].agent_id, "agent-fast");
        assert_eq!(tree.rows[0].name, "fast reviewer");
        assert_eq!(tree.rows[0].status, "completed");
        assert_eq!(tree.rows[0].done_order, Some(1));
    }

    #[test]
    fn same_turn_multiple_delegation_calls_keep_separate_inline_trees() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-a");
        spawn_tool_call(&mut app, &ids, "call-b");

        // Runtime can emit Running blocks for both delegation tool calls before
        // either executes. The first manifest poll belongs to the first serially
        // executing delegation call, so it must attach to call-a, not the later
        // call-b row.
        app.refresh_agent_batch(&[summary("agent-a", "running", 100)]);
        app.finish_agent_batch("call-a");
        let tree_a = app
            .transcript_mut()
            .agent_tree("call-a")
            .cloned()
            .expect("call-a tree");
        assert!(tree_a.finished);
        assert_eq!(tree_a.rows.len(), 1);
        assert_eq!(tree_a.rows[0].agent_id, "agent-a");
        assert!(
            app.transcript_mut()
                .agent_tree("call-b")
                .is_none_or(|tree| tree.rows.is_empty()),
            "call-b must not absorb call-a's agents"
        );

        // After call-a returns, call-b becomes the collecting batch.
        app.refresh_agent_batch(&[summary("agent-b", "running", 200)]);
        app.finish_agent_batch("call-b");
        let tree_b = app
            .transcript_mut()
            .agent_tree("call-b")
            .cloned()
            .expect("call-b tree");
        assert!(tree_b.finished);
        assert_eq!(tree_b.rows.len(), 1);
        assert_eq!(tree_b.rows[0].agent_id, "agent-b");
    }

    #[test]
    fn same_turn_manifest_name_collision_routes_by_agent_id_not_display_name() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-a");
        spawn_tool_call(&mut app, &ids, "call-b");

        app.refresh_agent_batch(&[summary_named("agent-a", "same-name", "running", 100)]);
        app.finish_agent_batch("call-a");
        app.refresh_agent_batch(&[summary_named("agent-b", "same-name", "running", 200)]);
        app.finish_agent_batch("call-b");

        let tree_a = app
            .transcript_mut()
            .agent_tree("call-a")
            .cloned()
            .expect("call-a tree");
        let tree_b = app
            .transcript_mut()
            .agent_tree("call-b")
            .cloned()
            .expect("call-b tree");
        assert_eq!(tree_a.rows.len(), 1);
        assert_eq!(tree_a.rows[0].agent_id, "agent-a");
        assert_eq!(tree_a.rows[0].name, "same-name");
        assert_eq!(tree_b.rows.len(), 1);
        assert_eq!(tree_b.rows[0].agent_id, "agent-b");
        assert_eq!(tree_b.rows[0].name, "same-name");
    }

    #[test]
    fn same_turn_completion_name_collision_routes_by_agent_id_not_display_name() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-a");
        spawn_tool_call(&mut app, &ids, "call-b");

        assert!(app.note_agent_completion_display("agent-a", "same-name", "completed", 111));
        app.finish_agent_batch("call-a");
        assert!(app.note_agent_completion_display("agent-b", "same-name", "completed", 222));
        app.finish_agent_batch("call-b");

        let tree_a = app
            .transcript_mut()
            .agent_tree("call-a")
            .cloned()
            .expect("call-a tree");
        let tree_b = app
            .transcript_mut()
            .agent_tree("call-b")
            .cloned()
            .expect("call-b tree");
        assert_eq!(tree_a.rows.len(), 1);
        assert_eq!(tree_a.rows[0].agent_id, "agent-a");
        assert_eq!(tree_a.rows[0].name, "same-name");
        assert_eq!(tree_a.rows[0].tokens, 111);
        assert_eq!(tree_b.rows.len(), 1);
        assert_eq!(tree_b.rows[0].agent_id, "agent-b");
        assert_eq!(tree_b.rows[0].name, "same-name");
        assert_eq!(tree_b.rows[0].tokens, 222);
    }

    #[test]
    fn same_turn_multiple_delegation_completions_seed_the_collecting_batch() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-a");
        spawn_tool_call(&mut app, &ids, "call-b");

        assert!(app.note_agent_completion_display("agent-a", "Agent A", "completed", 111));
        app.finish_agent_batch("call-a");
        let tree_a = app
            .transcript_mut()
            .agent_tree("call-a")
            .cloned()
            .expect("call-a tree");
        assert_eq!(tree_a.rows.len(), 1);
        assert_eq!(tree_a.rows[0].agent_id, "agent-a");
        assert_eq!(tree_a.rows[0].done_order, Some(1));
        assert!(
            app.transcript_mut()
                .agent_tree("call-b")
                .is_none_or(|tree| tree.rows.is_empty()),
            "call-b must not seed call-a's pre-manifest completion"
        );

        assert!(app.note_agent_completion_display("agent-b", "Agent B", "completed", 222));
        app.finish_agent_batch("call-b");
        let tree_b = app
            .transcript_mut()
            .agent_tree("call-b")
            .cloned()
            .expect("call-b tree");
        assert_eq!(tree_b.rows.len(), 1);
        assert_eq!(tree_b.rows[0].agent_id, "agent-b");
        assert_eq!(tree_b.rows[0].done_order, Some(1));
    }

    #[test]
    fn late_completion_after_finished_batch_does_not_seed_unrelated_row() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-4");
        app.finish_agent_batch("call-4");

        assert!(!app.note_agent_completion_display(
            "agent-late",
            "late reviewer",
            "completed",
            42,
        ));
        let tree = app
            .transcript_mut()
            .agent_tree("call-4")
            .cloned()
            .expect("tree");
        assert!(tree.finished);
        assert!(tree.rows.is_empty());
    }

    /// 종결 상태는 이후 manifest 폴링의 `running` 으로 강등되지 않는다.
    #[test]
    fn terminal_rows_never_regress_from_later_manifest_polls() {
        let mut app = test_app();
        let ids = BlockIdGen::default();
        spawn_tool_call(&mut app, &ids, "call-2");
        app.refresh_agent_batch(&[summary("a", "running", 100)]);
        assert!(app.note_agent_completion_display("a", "a", "completed", 500));
        // 매니페스트가 한 박자 늦게 running 을 보고해도 무시.
        app.refresh_agent_batch(&[summary("a", "running", 100)]);
        let tree = app
            .transcript_mut()
            .agent_tree("call-2")
            .cloned()
            .expect("tree");
        assert_eq!(tree.rows[0].status, "completed");
        assert_eq!(tree.rows[0].done_order, Some(1));
    }
}
