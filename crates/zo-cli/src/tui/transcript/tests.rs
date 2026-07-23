use super::first_visible_layout_entry;
use super::layout::{prose_tool_boundary_gap, turn_boundary_gap};
use super::tool_groups::{
    ToolGroupState, compute_tool_groups, compute_tool_groups_call_count,
    recompute_tool_groups_tail, reset_compute_tool_groups_call_count, tool_group_recompute_start,
};
use crate::tui::theme::Theme;

use super::RenderCache;
use super::Transcript;
use super::{buffer_row_text, char_selection_rows, join_selection_lines};
use crate::tui::blocks::tool_call::{AgentTree, AgentTreeRow};
use crate::tui::image_protocol::ImageProtocol;
use ratatui::Terminal;
use ratatui::backend::{Backend, ClearType, TestBackend, WindowSize};
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::{Position, Rect, Size};
use ratatui::style::{Color, Style};
use runtime::message_stream::{
    BashResult, BlockId, DiffHunk, DiffLine, DiffLineKind, DiffView, RenderBlock, SystemLevel,
    TodoResultItem, TodoResultStatus, ToolCallId, ToolCallStatus, ToolPreview, ToolResultBody,
};

fn id() -> BlockId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    BlockId(N.fetch_add(1, Ordering::Relaxed))
}

#[test]
#[ignore = "perf measurement, run manually with --ignored --nocapture"]
fn perf_search_large_transcript() {
    use std::time::Instant;
    let mut t = Transcript::new();
    for i in 0..3000u64 {
        t.push(RenderBlock::TextDelta {
            id: BlockId(i),
            text: format!("블록 {i}: 이것은 검색 대상 내용 텍스트 with english words here"),
            done: true,
        });
    }
    eprintln!("[perf] blocks = {}", t.blocks().len());

    let s = Instant::now();
    let m = t.find_all_blocks_containing("내용");
    eprintln!(
        "[perf] find_all_blocks_containing: {:?} ({} matches)",
        s.elapsed(),
        m.len()
    );

    // 검색 입력 연타(10타) 시뮬레이션 — 매 키입력마다 전체 재스캔.
    let s = Instant::now();
    for q in [
        "검색",
        "내용",
        "텍스트",
        "english",
        "words",
        "블록",
        "xq",
        "yz",
        "abc",
        "내용",
    ] {
        let _ = t.find_all_blocks_containing(q);
    }
    eprintln!("[perf] find_all x10 (검색 연타): {:?}", s.elapsed());
}

fn sample_agent_tree() -> AgentTree {
    AgentTree {
        rows: vec![AgentTreeRow {
            agent_id: "agent-1".to_string(),
            name: "explorer".to_string(),
            model: "gpt-5.5".to_string(),
            status: "running".to_string(),
            ..AgentTreeRow::default()
        }],
        batch_label: None,
        finished: false,
    }
}

fn todo_block(label: &str) -> RenderBlock {
    todo_block_with_status(label, TodoResultStatus::Pending)
}

fn todo_block_with_status(label: &str, status: TodoResultStatus) -> RenderBlock {
    RenderBlock::ToolResult {
        id: id(),
        tool_call_id: ToolCallId(format!("todo-{label}")),
        is_error: false,
        body: ToolResultBody::Todos(vec![TodoResultItem {
            content: label.to_string(),
            active_form: label.to_string(),
            status,
        }]),
    }
}

fn generic_tool_call(label: &str) -> RenderBlock {
    RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId(format!("call-{label}")),
        name: "Cargo".to_string(),
        summary: "test".to_string(),
        preview: ToolPreview::Generic {
            name: "Cargo".to_string(),
            input_summary: "test".to_string(),
        },
        status: ToolCallStatus::Running,
    }
}

#[test]
fn transcript_drops_stray_call_text_before_tool_call() {
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::TextDelta {
        id: id(),
        text: "call\n\ncall".to_string(),
        done: true,
    });
    transcript.push(generic_tool_call("stray"));

    assert_eq!(transcript.blocks().len(), 1);
    assert!(matches!(
        transcript.blocks()[0],
        RenderBlock::ToolCall { .. }
    ));
    assert_eq!(transcript.rendered_cache_len(), transcript.len());
}

#[test]
fn transcript_strips_trailing_call_marker_after_real_text_before_tool_call() {
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::TextDelta {
        id: id(),
        text: "먼저 진짜 버그를 고치겠습니다.\n\ncall\n\ncall\n".to_string(),
        done: true,
    });
    transcript.push(generic_tool_call("stray-suffix"));

    assert_eq!(transcript.blocks().len(), 2);
    assert!(matches!(
        &transcript.blocks()[0],
        RenderBlock::TextDelta { text, .. } if text == "먼저 진짜 버그를 고치겠습니다."
    ));
    assert!(matches!(
        transcript.blocks()[1],
        RenderBlock::ToolCall { .. }
    ));
    assert_eq!(transcript.rendered_cache_len(), transcript.len());
}

#[test]
fn transcript_preserves_real_call_text_before_tool_call() {
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::TextDelta {
        id: id(),
        text: "I'll call cargo now.".to_string(),
        done: true,
    });
    transcript.push(generic_tool_call("real-text"));

    assert_eq!(transcript.blocks().len(), 2);
    assert!(matches!(
        &transcript.blocks()[0],
        RenderBlock::TextDelta { text, .. } if text == "I'll call cargo now."
    ));
    assert!(matches!(
        transcript.blocks()[1],
        RenderBlock::ToolCall { .. }
    ));
}

#[test]
fn active_turn_suppresses_only_current_turn_todos_not_history() {
    let mut transcript = Transcript::new();
    transcript.push(todo_block("old plan"));
    transcript.set_turn_active(true);
    transcript.push(todo_block("new plan"));

    assert!(
        !transcript.todos_suppressed_during_turn(0),
        "todo history from previous turns must stay visible while a new turn streams"
    );
    assert!(
        transcript.todos_suppressed_during_turn(1),
        "only the current turn's todo block is hidden behind the live panel"
    );

    transcript.set_turn_active(false);
    assert!(
        !transcript.todos_suppressed_during_turn(1),
        "the current-turn todo reappears as settled history once the turn ends"
    );
}

#[test]
fn repeated_incomplete_todo_snapshots_in_same_turn_leave_only_latest_history() {
    let mut transcript = Transcript::new();
    transcript.push(todo_block("old history"));
    transcript.set_turn_active(true);
    transcript.push(todo_block("first current snapshot"));
    transcript.push(todo_block("second current snapshot"));
    transcript.push(todo_block("latest current snapshot"));

    assert!(
        !transcript.todos_suppressed_during_turn(0),
        "previous-turn todo history must stay visible"
    );
    assert!(
        transcript.todos_suppressed_during_turn(1),
        "the first current-turn snapshot is superseded"
    );
    assert!(
        transcript.todos_suppressed_during_turn(2),
        "the second current-turn snapshot is superseded"
    );
    assert!(
        transcript.todos_suppressed_during_turn(3),
        "the live panel owns the latest snapshot while the turn streams"
    );

    transcript.set_turn_active(false);
    assert!(
        !transcript.todos_suppressed_during_turn(0),
        "old history remains visible after settle"
    );
    assert!(
        transcript.todos_suppressed_during_turn(1),
        "superseded snapshot must not reappear after settle"
    );
    assert!(
        transcript.todos_suppressed_during_turn(2),
        "superseded snapshot must not reappear after settle"
    );
    assert!(
        !transcript.todos_suppressed_during_turn(3),
        "only the latest incomplete snapshot becomes settled Updated Plan history"
    );
}

#[test]
fn completed_todo_snapshot_stays_hidden_as_chat_history() {
    let mut transcript = Transcript::new();
    transcript.push(todo_block_with_status(
        "completed plan",
        TodoResultStatus::Completed,
    ));

    // All-completed snapshots are acknowledgements, not durable chat history.
    // They should not linger as `Updated Plan · N/N done` cards.
    assert!(
        transcript.todos_suppressed_during_turn(0),
        "a completed todo snapshot must not remain as chat history"
    );
}

#[test]
fn completed_plan_in_a_turn_stays_hidden_after_settle() {
    // The desired behavior: a completed plan should not linger. It is hidden
    // while the turn streams and stays hidden after settle.
    let mut transcript = Transcript::new();
    transcript.set_turn_active(true);
    transcript.push(todo_block_with_status(
        "done plan",
        TodoResultStatus::Completed,
    ));
    assert!(
        transcript.todos_suppressed_during_turn(0),
        "the live panel owns the plan while the turn streams, so the snapshot is hidden"
    );

    transcript.set_turn_active(false);
    assert!(
        transcript.todos_suppressed_during_turn(0),
        "after the turn settles the completed plan should not reappear as a history card"
    );
}

#[test]
fn completed_snapshot_supersedes_current_turn_todo_history_only() {
    let mut transcript = Transcript::new();
    transcript.push(todo_block("old incomplete history"));
    transcript.set_turn_active(true);
    transcript.push(todo_block("current incomplete"));
    transcript.push(todo_block_with_status(
        "current completed",
        TodoResultStatus::Completed,
    ));

    assert!(
        !transcript.todos_suppressed_during_turn(0),
        "previous-turn incomplete history must remain visible"
    );
    assert!(
        transcript.todos_suppressed_during_turn(1),
        "completed snapshot should supersede the current-turn incomplete plan"
    );
    assert!(
        transcript.todos_suppressed_during_turn(2),
        "completed snapshot itself must stay hidden"
    );

    transcript.set_turn_active(false);
    assert!(
        !transcript.todos_suppressed_during_turn(0),
        "settling the turn must not hide older incomplete history"
    );
    assert!(
        transcript.todos_suppressed_during_turn(1),
        "superseded current-turn plan must not reappear as old history"
    );
    assert!(
        transcript.todos_suppressed_during_turn(2),
        "the completed snapshot must stay hidden after the turn settles"
    );
}

#[test]
fn active_turn_start_preserves_history_render_cache() {
    let theme = Theme::default_dark();
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::TextDelta {
        id: id(),
        text: "cached **history** before the next turn".to_string(),
        done: true,
    });
    transcript.push(todo_block("old plan"));

    let backend = TestBackend::new(72, 10);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|frame| {
            transcript.draw(
                frame,
                Rect::new(0, 0, 72, 10),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw");

    let cached_before = transcript
        .rendered_cache
        .iter()
        .filter(|slot| slot.is_some())
        .count();
    assert!(
        cached_before > 0,
        "initial draw should populate transcript render caches"
    );

    transcript.set_turn_active(true);

    let cached_after = transcript
        .rendered_cache
        .iter()
        .filter(|slot| slot.is_some())
        .count();
    assert_eq!(
        cached_after, cached_before,
        "starting a new turn must not drop cached history just because old Todo history exists"
    );
}

#[test]
fn transcript_prunes_overflow_in_chunks_and_keeps_parallel_caches_aligned() {
    let mut transcript = Transcript::new();
    let prune_trigger = super::MAX_TRANSCRIPT_BLOCKS + super::TRANSCRIPT_PRUNE_CHUNK;
    for index in 0..prune_trigger {
        transcript.push(RenderBlock::System {
            id: BlockId(index as u64),
            level: SystemLevel::Info,
            text: format!("system {index}"),
        });
    }

    assert_eq!(transcript.len(), prune_trigger);
    assert_eq!(transcript.rendered_cache_len(), transcript.len());
    let RenderBlock::System { text, .. } = &transcript.blocks()[0] else {
        panic!("expected system block");
    };
    assert_eq!(text, "system 0");

    transcript.push(RenderBlock::System {
        id: BlockId(prune_trigger as u64),
        level: SystemLevel::Info,
        text: format!("system {prune_trigger}"),
    });

    assert_eq!(transcript.len(), super::MAX_TRANSCRIPT_BLOCKS);
    assert_eq!(transcript.rendered_cache_len(), transcript.len());
    let RenderBlock::System { text, .. } = &transcript.blocks()[0] else {
        panic!("expected system block");
    };
    assert_eq!(
        text,
        &format!("system {}", super::TRANSCRIPT_PRUNE_CHUNK + 1)
    );
}

#[test]
fn set_agent_tree_ignores_orphan_updates() {
    let mut transcript = Transcript::new();
    let tree = sample_agent_tree();

    transcript.set_agent_tree("missing_call", tree.clone());
    assert!(
        transcript.agent_tree("missing_call").is_none(),
        "agent tree without an owning ToolCall must not be retained"
    );

    transcript
        .agent_trees
        .insert("missing_call".to_string(), tree.clone());
    transcript.set_agent_tree("missing_call", tree);
    assert!(
        transcript.agent_tree("missing_call").is_none(),
        "orphan refresh should drop any stale side-table entry"
    );
}

#[test]
fn set_agent_tree_attaches_to_existing_tool_call() {
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("call_7".to_string()),
        name: "Bash".to_string(),
        summary: "echo".to_string(),
        preview: ToolPreview::Generic {
            name: "Bash".to_string(),
            input_summary: "echo".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    let tree = sample_agent_tree();

    transcript.set_agent_tree("call_7", tree.clone());

    assert_eq!(transcript.agent_tree("call_7"), Some(&tree));
}

/// 라이브 배치는 host 행 아래에서 고정 높이 요약으로 렌더되며, 기존
/// `live_tree_visible` viewport 판정은 그대로 하단 핀 패널을 게이트한다.
#[test]
fn live_agent_batch_stays_compact_and_keeps_the_pinned_panel_visibility_gate() {
    let theme = Theme::default_dark();
    let width = 80;
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("call_fan".to_string()),
        name: "Task".to_string(),
        summary: "spawn".to_string(),
        preview: ToolPreview::Generic {
            name: "Task".to_string(),
            input_summary: "spawn".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    let bare = transcript.cached_block_height(0, width, &theme, ImageProtocol::None);

    // A live batch: three still-running agents (none terminal, not finished).
    let live = AgentTree {
        rows: (0..3)
            .map(|i| AgentTreeRow {
                agent_id: format!("agent-{i}"),
                name: format!("explorer-{i}"),
                model: "gpt-5.5".to_string(),
                status: "running".to_string(),
                ..AgentTreeRow::default()
            })
            .collect(),
        batch_label: None,
        finished: false,
    };
    transcript.set_agent_tree("call_fan", live);
    let with_tree = transcript.cached_block_height(0, width, &theme, ImageProtocol::None);
    assert_eq!(
        with_tree, bare,
        "agent fan-out must replace the placeholder with a compact aggregate: bare {bare} rows, with tree {with_tree} rows"
    );
    assert_eq!(with_tree, 2, "header + one aggregate row");

    // Visibility gate: with the block in the viewport (layout starts at the
    // top, resolved scroll 0) the panel must stay hidden…
    let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, 12))
        .expect("test terminal");
    term.draw(|frame| {
        let area = frame.area();
        transcript.draw(frame, area, &theme, 0, ImageProtocol::None);
    })
    .expect("draw");
    assert!(
        transcript.live_tree_visible(12),
        "a live-tree block inside the viewport suppresses the pinned panel"
    );
    // …and a zero-height viewport (nothing visible) re-admits it.
    assert!(
        !transcript.live_tree_visible(0),
        "no viewport, no visible tree — the panel may show"
    );
}

#[test]
fn upsert_system_replaces_existing_progress_block() {
    let mut transcript = Transcript::new();
    let progress_id = id();

    transcript.upsert_system(
        progress_id,
        SystemLevel::Info,
        "Smart pre-analysis: running".to_string(),
    );
    transcript.upsert_system(
        progress_id,
        SystemLevel::Info,
        "Smart pre-analysis: stopped".to_string(),
    );

    assert_eq!(transcript.len(), 1);
    let RenderBlock::System { text, .. } = &transcript.blocks[0] else {
        panic!("expected system progress block");
    };
    assert_eq!(text, "Smart pre-analysis: stopped");
}

struct CountingBackend {
    inner: TestBackend,
    draw_counts: Vec<usize>,
}

impl CountingBackend {
    fn new(width: u16, height: u16) -> Self {
        Self {
            inner: TestBackend::new(width, height),
            draw_counts: Vec::new(),
        }
    }
}

impl Backend for CountingBackend {
    type Error = core::convert::Infallible;

    fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        let mut count = 0;
        self.inner.draw(content.inspect(|_| count += 1))?;
        self.draw_counts.push(count);
        Ok(())
    }

    fn hide_cursor(&mut self) -> Result<(), Self::Error> {
        self.inner.hide_cursor()
    }

    fn show_cursor(&mut self) -> Result<(), Self::Error> {
        self.inner.show_cursor()
    }

    fn get_cursor_position(&mut self) -> Result<Position, Self::Error> {
        self.inner.get_cursor_position()
    }

    fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> Result<(), Self::Error> {
        self.inner.set_cursor_position(position)
    }

    fn clear(&mut self) -> Result<(), Self::Error> {
        self.inner.clear()
    }

    fn clear_region(&mut self, clear_type: ClearType) -> Result<(), Self::Error> {
        self.inner.clear_region(clear_type)
    }

    fn size(&self) -> Result<Size, Self::Error> {
        self.inner.size()
    }

    fn window_size(&mut self) -> Result<WindowSize, Self::Error> {
        self.inner.window_size()
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }
}

/// Render a fixed sample conversation to a `TestBackend` and return the
/// buffer as newline-separated rows — a human-readable "screenshot" that
/// lets the role-rail / header layout be eyeballed in test output.
fn render_sample(width: u16, height: u16) -> String {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "아키텍처 그려줘".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "의존성을 모두 파악했습니다. 계층형으로 정리하겠습니다.".to_string(),
        done: true,
    });
    t.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("call_1".to_string()),
        name: "Bash".to_string(),
        summary: "cargo metadata".to_string(),
        preview: ToolPreview::Generic {
            name: "Bash".to_string(),
            input_summary: "cargo metadata".to_string(),
        },
        status: ToolCallStatus::Ok,
    });
    t.push(RenderBlock::ToolResult {
        id: id(),
        tool_call_id: ToolCallId("call_1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "12 crates · 그래프 생성".to_string(),
            truncated: false,
        },
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "핵심은 위에서 아래로 흐르는 계층 구조이며 순환 의존성이 없습니다.".to_string(),
        done: true,
    });
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| {
            t.draw(
                f,
                Rect::new(0, 0, width, height),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw");
    dump_terminal(&terminal, width, height)
}

#[test]
fn turn_boundary_separator_is_labeled_with_turn_ordinal() {
    // A new user message after assistant output is a turn boundary; its rule must
    // carry the 1-based turn ordinal so a long transcript reads as discrete turns.
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "first question".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "first answer".to_string(),
        done: true,
    });
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "second question".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "second answer".to_string(),
        done: true,
    });

    let (width, height) = (60u16, 16u16);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| {
            t.draw(
                f,
                Rect::new(0, 0, width, height),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw");
    let dump = dump_terminal(&terminal, width, height);
    assert!(
        dump.contains("turn 2"),
        "the turn-2 boundary separator must carry its ordinal label:\n{dump}"
    );
}

#[test]
fn tool_block_does_not_paint_state_tinted_card_background() {
    // Routine ToolCall + ToolResult history is plain layout. Status remains in
    // glyph/foreground color instead of becoming a card-shaped background wash.
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("c1".to_string()),
        name: "Bash".to_string(),
        summary: "ls".to_string(),
        preview: ToolPreview::Generic {
            name: "Bash".to_string(),
            input_summary: "ls".to_string(),
        },
        status: ToolCallStatus::Ok,
    });
    t.push(RenderBlock::ToolResult {
        id: id(),
        tool_call_id: ToolCallId("c1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });

    let (w, h) = (60u16, 10u16);
    let mut any_cell_has_bg = |theme: &Theme, target: ratatui::style::Color| -> bool {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("backend");
        terminal
            .draw(|f| t.draw(f, Rect::new(0, 0, w, h), theme, 0, ImageProtocol::None))
            .expect("draw");
        let buf = terminal.backend().buffer();
        (0..w).any(|x| (0..h).any(|y| buf.cell((x, y)).is_some_and(|c| c.bg == target)))
    };

    let zo = Theme::zo();
    let success_bg = zo
        .tool_card_bg(zo.palette.success)
        .expect("zo palette can derive the old card tint sentinel");
    assert!(
        !any_cell_has_bg(&zo, success_bg),
        "an Ok tool block must remain background-free"
    );
    assert!(
        !any_cell_has_bg(&Theme::no_color(), success_bg),
        "NO_COLOR also remains background-free"
    );
}

/// Regression: diff rows own their own add/remove/context backgrounds. The
/// generic success/error tool-card wash must not repaint a Diff result after the
/// diff renderer has assigned row-local styling; otherwise unchanged/header rows
/// look highlighted and changed rows lose their add/delete colors.
#[test]
fn diff_tool_result_preserves_diff_backgrounds_without_card_wash() {
    use runtime::message_stream::{DiffHunk, DiffLine, DiffLineKind, DiffView};

    let mut t = Transcript::new();
    t.push(RenderBlock::ToolResult {
        id: id(),
        tool_call_id: ToolCallId("diff-1".to_string()),
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("src/lib.rs".to_string()),
            new_path: Some("src/lib.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 2,
                new_start: 1,
                new_lines: 2,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Context,
                        text: "let unchanged = true;".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "let value = old_value;".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "let value = new_value;".to_string(),
                    },
                ],
            }],
        }),
    });

    let (w, h) = (96u16, 14u16);
    let zo = Theme::zo();
    let success_bg = zo
        .tool_card_bg(zo.palette.success)
        .expect("zo truecolor palette tints normal tool cards");
    let add_bg = zo.diff_add_bg().expect("zo palette has diff add bg");
    let del_bg = zo.diff_del_bg().expect("zo palette has diff del bg");

    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, w, h), &zo, 0, ImageProtocol::None))
        .expect("draw");
    let buf = terminal.backend().buffer();

    let count_bg = |target| -> usize {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| buf.cell((x, y)).is_some_and(|c| c.bg == target))
            .count()
    };

    assert_eq!(
        count_bg(success_bg),
        0,
        "Diff ToolResult must not receive the generic success card wash"
    );
    assert!(
        count_bg(add_bg) > 0,
        "added diff rows must keep their add background"
    );
    assert!(
        count_bg(del_bg) > 0,
        "removed diff rows must keep their delete background"
    );

    let row_text = |y: u16| -> String { (0..w).map(|x| buf[(x, y)].symbol()).collect() };
    let context_y = (0..h)
        .find(|&y| row_text(y).contains("unchanged"))
        .expect("context row should render");
    assert!(
        (0..w).all(|x| buf
            .cell((x, context_y))
            .is_none_or(|c| c.bg != success_bg && c.bg != add_bg && c.bg != del_bg)),
        "unchanged context row must remain unwashed"
    );
}

fn dump_terminal(terminal: &Terminal<TestBackend>, width: u16, height: u16) -> String {
    let buf = terminal.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..height {
        for x in 0..width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

// --- P9 styled-golden safety net -------------------------------------------
// `dump_terminal` above captures glyphs ONLY, so a restructure that recolored
// every rail / wash / highlight would leave all glyph-based tests green. The
// P9 transcript re-architecture (ScrollPos / BlockEntry / RenderCache::Group /
// ensure_layout consolidation) must be proven render-identical down to the
// per-cell STYLE — that is what these helpers guard.

/// Non-default style of one cell as a compact, stable token (empty when the
/// cell carries the reset style, so unstyled runs add no noise).
fn cell_style_token(cell: &Cell) -> String {
    let mut parts: Vec<String> = Vec::new();
    if cell.fg != Color::Reset {
        parts.push(format!("fg={:?}", cell.fg));
    }
    if cell.bg != Color::Reset {
        parts.push(format!("bg={:?}", cell.bg));
    }
    if !cell.modifier.is_empty() {
        parts.push(format!("mod={:?}", cell.modifier));
    }
    parts.join(",")
}

/// Render `t` to a fixed `width`x`height` `TestBackend` and serialize every
/// cell's symbol AND style. Each row emits its fenced glyph line (catching any
/// symbol/wrap change) followed by a run-length list of non-default styles
/// (catching any recolor: a style change splits, shifts, adds, or drops a run).
fn snap_styled(width: u16, height: u16, t: &mut Transcript) -> String {
    let theme = Theme::default_dark();
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, width, height), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let buf = terminal.backend().buffer();

    let mut out = String::new();
    for y in 0..height {
        out.push('|');
        for x in 0..width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push_str("|\n");

        let mut runs: Vec<String> = Vec::new();
        let mut x = 0u16;
        while x < width {
            let tok = cell_style_token(&buf[(x, y)]);
            if tok.is_empty() {
                x += 1;
                continue;
            }
            let start = x;
            x += 1;
            while x < width && cell_style_token(&buf[(x, y)]) == tok {
                x += 1;
            }
            runs.push(format!("{start}-{}[{tok}]", x - 1));
        }
        if !runs.is_empty() {
            out.push_str("  ");
            out.push_str(&runs.join(" "));
            out.push('\n');
        }
    }
    out
}

/// Byte-compare `snap_styled(t)` at each width against a committed fixture in
/// `src/tui/transcript/golden/<name>.snap`. Set `ZO_UPDATE_GOLDEN=1` to
/// (re)generate the fixture after an *intentional* rendering change.
fn golden(name: &str, height: u16, widths: &[u16], build: impl Fn() -> Transcript) {
    use std::fmt::Write as _;
    let mut actual = String::new();
    for &w in widths {
        let _ = writeln!(actual, "=== width {w} height {height} ===");
        // A fresh transcript per width, so no width-keyed layout/render-cache
        // state carries over between renders — each cell is an independent
        // per-width golden.
        let mut t = build();
        actual.push_str(&snap_styled(w, height, &mut t));
        actual.push('\n');
    }
    assert_or_write_golden(name, &actual);
}

/// Shared read/write/assert tail for [`golden`] and [`golden_timeline`]:
/// `ZO_UPDATE_GOLDEN=1` (re)writes the fixture at
/// `src/tui/transcript/golden/<name>.snap`, else byte-compares against it.
fn assert_or_write_golden(name: &str, actual: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/tui/transcript/golden")
        .join(format!("{name}.snap"));
    if std::env::var_os("ZO_UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().expect("golden dir has a parent"))
            .expect("create golden dir");
        std::fs::write(&path, actual).expect("write golden fixture");
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing golden `{name}` at {}; run ZO_UPDATE_GOLDEN=1 cargo test to create it",
            path.display()
        )
    });
    assert_eq!(
        actual, expected,
        "golden `{name}` changed — if the new rendering is intentional, regenerate with ZO_UPDATE_GOLDEN=1"
    );
}

/// Time-axis sibling of [`golden`]: freeze the per-stage streaming render of a
/// single growing transcript at one width. Builds ONE [`Transcript`]; for each
/// `(label, stage)` it applies the stage (typically a same-id `TextDelta`
/// append — the streaming mechanism, see `Transcript::try_merge_block`) then
/// snapshots, concatenating `=== stage: <label> ===` sections into one fixture.
/// Locks the mid-stream markdown approximations (inline-marker repair, whole-
/// tail holds, the >8KB lightweight path) that later markdown churn must not
/// silently move.
// clippy::type_complexity: the labeled-stage slice `&[(&str, &dyn Fn(&mut
// Transcript))]` is the intended harness shape (a stage label + its mutation);
// a type alias would only rename it.
#[allow(clippy::type_complexity)]
fn golden_timeline(
    name: &str,
    height: u16,
    width: u16,
    stages: &[(&str, &dyn Fn(&mut Transcript))],
) {
    use std::fmt::Write as _;
    let mut actual = String::new();
    let mut t = Transcript::new();
    for (label, stage) in stages {
        stage(&mut t);
        let _ = writeln!(actual, "=== stage: {label} ===");
        actual.push_str(&snap_styled(width, height, &mut t));
        actual.push('\n');
    }
    assert_or_write_golden(name, &actual);
}

/// Push one streaming `TextDelta` chunk under a fixed id. Same-id pushes append
/// (`Transcript::try_merge_block`), so a sequence streams one growing assistant
/// message; a final `done: true` chunk settles it.
fn tl_delta(t: &mut Transcript, text: &str, done: bool) {
    t.push(RenderBlock::TextDelta {
        id: BlockId(1),
        text: text.to_string(),
        done,
    });
}

#[test]
fn golden_tl_prose_emphasis() {
    // Freezes the mid-stream inline-marker repair shim: bold / code / italic
    // markers are split ACROSS deltas (`**bo`|`ld**`, `` `co ``|`` de` ``,
    // `*ital`|`ic*`), so each stage locks the streaming approximation and the
    // settle locks the authoritative render.
    golden_timeline(
        "tl_prose_emphasis",
        24,
        80,
        &[
            ("prefix", &|t| tl_delta(t, "The value ", false)),
            ("open-bold", &|t| tl_delta(t, "**bo", false)),
            ("close-bold-open-code", &|t| tl_delta(t, "ld** and `co", false)),
            ("close-code-open-ital", &|t| tl_delta(t, "de` plus *ital", false)),
            ("close-ital-tail", &|t| tl_delta(t, "ic* tail.", false)),
            ("settle", &|t| tl_delta(t, "", true)),
        ],
    );
}

#[test]
fn golden_tl_code_fence() {
    // Freezes the fence open -> interior -> close -> settle transitions when the
    // ```rust``` fence and its body arrive across delta boundaries.
    golden_timeline(
        "tl_code_fence",
        24,
        80,
        &[
            ("open-fence", &|t| tl_delta(t, "Setting up:\n\n```ru", false)),
            ("lang-and-sig", &|t| tl_delta(t, "st\nfn main() {", false)),
            ("interior", &|t| tl_delta(t, "\n    let x = 1;\n", false)),
            ("close-fence", &|t| tl_delta(t, "}\n```\ndone.", false)),
            ("settle", &|t| tl_delta(t, "", true)),
        ],
    );
}

#[test]
fn golden_tl_list() {
    // Freezes the current whole-tail hold for a streaming unordered list (item
    // "bra"|"vo" split across the delta boundary).
    golden_timeline(
        "tl_list",
        24,
        80,
        &[
            ("first-items", &|t| tl_delta(t, "Steps:\n\n- alpha\n- bra", false)),
            ("rest", &|t| tl_delta(t, "vo\n- charlie\n- delta", false)),
            ("settle", &|t| tl_delta(t, "", true)),
        ],
    );
}

#[test]
fn golden_tl_table() {
    // Freezes the streaming table lifecycle: an incomplete header alone (the
    // pre-box state), the transition to box-drawing once the separator + first
    // row make it well-formed, the row append, and the settle. The pre-box ->
    // box step is exactly what later markdown churn will touch.
    golden_timeline(
        "tl_table",
        24,
        80,
        &[
            ("header-only", &|t| tl_delta(t, "| name | qty |", false)),
            ("well-formed-box", &|t| {
                tl_delta(t, "\n|---|---|\n| apple | 1 |", false);
            }),
            ("row2", &|t| tl_delta(t, "\n| pear | 2 |", false)),
            ("settle", &|t| tl_delta(t, "", true)),
        ],
    );
}

#[test]
fn golden_tl_diff_fence() {
    // Freezes the gutterless diff-fence interior fallback (the ```diff``` path
    // in streaming markdown).
    golden_timeline(
        "tl_diff_fence",
        24,
        80,
        &[
            ("open-diff", &|t| {
                tl_delta(t, "```diff\n+ added line\n- removed line", false);
            }),
            ("hunk-context", &|t| {
                tl_delta(t, "\n@@ -1,2 +1,2 @@\n context", false);
            }),
            ("close-fence", &|t| tl_delta(t, "\n```", false)),
            ("settle", &|t| tl_delta(t, "", true)),
        ],
    );
}

/// Deterministic diff-fence body rows for [`golden_tl_promoted_diff_fence`]:
/// a repeating +/-/@@/context mix, ~45 bytes per line.
fn promoted_diff_rows(from: u32, to: u32) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    for n in from..=to {
        // Infallible writes into a String; the results are intentionally ignored.
        let _ = match n % 4 {
            0 => writeln!(s, "@@ -{n},4 +{n},4 @@ hunk marker row"),
            1 => writeln!(s, "+ added widget {n} with extra payload text"),
            2 => writeln!(s, "- removed widget {n} with extra payload text"),
            _ => writeln!(s, "  context row {n} keeps surrounding code visible"),
        };
    }
    s
}

/// Freezes the PROMOTED diff-fence interior — `diff_fence_interior_lines`
/// (formerly the naive parser's diff branch). `tl_diff_fence` stays <8KB and
/// renders through the pulldown tail, so THIS cell is the only golden that
/// exercises the promoted gutterless interior: the fence buffer must exceed
/// `LARGE_OPEN_FENCE_LIMIT` (16KB) for the boundary scanner to promote the
/// open fence, routing interior rows through `rendered_fence_interior_lines`'
/// diff branch. Stage 1 stays small (<8KB → pulldown tail, identical pre/post
/// D-3); stage 2 crosses the promotion threshold; settle renders the
/// authoritative gutter card. The committed snap was generated against
/// pre-D-3 HEAD (the naive branch) and passes byte-identically against the
/// D-3 port — the cell doubles as the port's equivalence proof.
#[test]
fn golden_tl_promoted_diff_fence() {
    golden_timeline(
        "tl_promoted_diff_fence",
        26,
        80,
        &[
            ("small-open", &|t| {
                tl_delta(t, &format!("```diff\n{}", promoted_diff_rows(1, 40)), false);
            }),
            ("promoted", &|t| {
                tl_delta(t, &promoted_diff_rows(41, 440), false);
            }),
            ("settle", &|t| tl_delta(t, "\n```\n", true)),
        ],
    );
}

#[test]
fn golden_tl_large_tail() {
    // Post-D-3 convergence guard for a large streaming tail. A blank-line-free
    // PARAGRAPH (NOT a list: list/blockquote markers advance the stable boundary
    // per line — markdown.rs `is_list_item_or_blockquote_marker` — shrinking the
    // open tail) is one open segment, so the whole tail routes to the streaming
    // tail renderer. D-3 deleted the naive third parser, so that renderer is now
    // the SAME pulldown path as the settle pass: streaming and `done` agree. The
    // spaced `2 * 3 and 4 * 5` line — which the deleted naive parser flipped
    // italic (no CommonMark flanking check) — now renders LITERALLY (stars kept,
    // no ITALIC run) in EVERY stage, matching settle; the old streaming-vs-settle
    // divergence is gone. `snake_case_word` (line 12) freezes intraword-`_`
    // parity; a balanced **bold** (line 1) exercises emphasis. The tail (~21KB)
    // stays under layout.rs `FINAL_DONE_MARKDOWN_RENDER_LIMIT` (32KB) so the done
    // pass re-renders via pulldown. Viewport is top-anchored (scroll 0), so the
    // special lines sit early (items 1/5/12) to stay visible.
    let mut items: Vec<String> = Vec::with_capacity(340);
    for n in 1..=340u32 {
        let line = if n == 1 {
            // Lines must NOT start with a list/blockquote marker: those advance
            // the stable boundary per line, shrinking the open tail — a blank-
            // line-free PARAGRAPH stays one open segment so the whole tail is
            // rendered by the streaming tail renderer (now the pulldown path,
            // post-D-3). A balanced **bold** exercises emphasis across the
            // streaming→settle boundary.
            "prose opener carries a **bold** run to route the tail".to_string()
        } else if n == 5 {
            // Spaces around the asterisks: CommonMark flanking makes pulldown
            // render these literally (stars kept, no italic). Post-D-3 the
            // streaming stages use that same pulldown path, so this line reads
            // identically while streaming and at settle — the old naive-parser
            // divergence (which flipped italic here) is gone.
            "prose 5 holds 2 * 3 and 4 * 5 as spaced stars".to_string()
        } else if n == 12 {
            "prose 12 holds snake_case_word intraword underscores".to_string()
        } else {
            format!("prose line {n} keeps the paragraph growing without blank lines")
        };
        items.push(line);
    }
    let full = items.join("\n");
    // Split at the 170-item midpoint: chunk 1 (~10KB) and the full (~21KB) both
    // stay under the 64KB pulldown-tail bound, so every streaming snapshot and
    // the settle render through the same pulldown path — proving convergence.
    let mid = items[..170].join("\n").len();
    let chunk1 = full[..mid].to_string();
    let chunk2 = full[mid..].to_string();
    golden_timeline(
        "tl_large_tail",
        30,
        80,
        &[
            ("half-1", &|t| tl_delta(t, &chunk1, false)),
            ("half-2", &|t| tl_delta(t, &chunk2, false)),
            ("settle", &|t| tl_delta(t, "", true)),
        ],
    );
}

#[test]
fn golden_plain_prose() {
    // Lowest-risk cell: settled user prompt + two settled prose deltas. Frozen
    // at three widths so the role rail, wrap points, and prose style are all
    // locked before any P9 restructuring touches layout or the render cache.
    golden("plain_prose", 20, &[60, 80, 120], || {
        let mut t = Transcript::new();
        t.push(RenderBlock::UserMessage {
            id: BlockId(1),
            text: "draw the zo architecture".to_string(),
        });
        t.push(RenderBlock::TextDelta {
            id: BlockId(2),
            text: "Mapped every dependency. Organizing it into layers now.".to_string(),
            done: true,
        });
        t.push(RenderBlock::TextDelta {
            id: BlockId(3),
            text: "The core flows top-to-bottom with no dependency cycles.".to_string(),
            done: true,
        });
        t
    });
}

#[test]
fn golden_code_fence() {
    // Locks the code-card frame + syntect styling that P9's RenderCache and
    // ensure_layout steps must render byte-identically.
    golden("code_fence", 20, &[60, 80], || {
        let mut t = Transcript::new();
        t.push(RenderBlock::UserMessage {
            id: BlockId(1),
            text: "show a hello world".to_string(),
        });
        t.push(RenderBlock::TextDelta {
            id: BlockId(2),
            text: "Here it is:\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n"
                .to_string(),
            done: true,
        });
        t
    });
}

#[test]
fn golden_tool_cards() {
    // Locks the tool-card rails/status styling that RenderCache::Group (Step 2)
    // and the ensure_layout consolidation (Step 4) touch.
    golden("tool_cards", 20, &[60, 80], || {
        let mut t = Transcript::new();
        t.push(RenderBlock::ToolCall {
            id: BlockId(1),
            tool_call_id: ToolCallId("call_1".to_string()),
            name: "Bash".to_string(),
            summary: "cargo build".to_string(),
            preview: ToolPreview::Bash {
                command: "cargo build".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(2),
            tool_call_id: ToolCallId("call_1".to_string()),
            is_error: false,
            body: ToolResultBody::Text {
                content: "Finished in 2.3s".to_string(),
                truncated: false,
            },
        });
        t.push(RenderBlock::ToolCall {
            id: BlockId(3),
            tool_call_id: ToolCallId("call_2".to_string()),
            name: "Read".to_string(),
            summary: "src/main.rs".to_string(),
            preview: ToolPreview::Generic {
                name: "Read".to_string(),
                input_summary: "src/main.rs".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(4),
            tool_call_id: ToolCallId("call_2".to_string()),
            is_error: false,
            body: ToolResultBody::Text {
                content: "fn main() {}".to_string(),
                truncated: false,
            },
        });
        t
    });
}

/// P10-B Step 0: freezes the per-variant result cards the `tool_result.rs`
/// decomposition will move — an inline diff (small enough to skip the
/// overflow cap), a todos checklist, and a grep listing. Call/result pairs
/// are interleaved (like `tool_cards`) so no collapsed group forms and each
/// variant's own card is what the snap locks.
#[test]
#[allow(clippy::too_many_lines)] // golden fixture: verbose typed literals, zero logic
fn golden_result_variant_cards() {
    golden("result_variant_cards", 30, &[60, 80], || {
        let mut t = Transcript::new();
        t.push(RenderBlock::ToolCall {
            id: BlockId(1),
            tool_call_id: ToolCallId("call_1".to_string()),
            name: "Edit".to_string(),
            summary: "src/lib.rs".to_string(),
            preview: ToolPreview::Generic {
                name: "Edit".to_string(),
                input_summary: "src/lib.rs".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(2),
            tool_call_id: ToolCallId("call_1".to_string()),
            is_error: false,
            body: ToolResultBody::Diff(DiffView {
                old_path: Some("src/lib.rs".to_string()),
                new_path: Some("src/lib.rs".to_string()),
                language: Some("rust".to_string()),
                hunks: vec![DiffHunk {
                    old_start: 1,
                    old_lines: 3,
                    new_start: 1,
                    new_lines: 3,
                    lines: vec![
                        DiffLine {
                            kind: DiffLineKind::Context,
                            text: "fn main() {".to_string(),
                        },
                        DiffLine {
                            kind: DiffLineKind::Removed,
                            text: "    println!(\"old\");".to_string(),
                        },
                        DiffLine {
                            kind: DiffLineKind::Added,
                            text: "    println!(\"new\");".to_string(),
                        },
                        DiffLine {
                            kind: DiffLineKind::Context,
                            text: "}".to_string(),
                        },
                    ],
                }],
            }),
        });
        // A prose block between pairs breaks tool-run adjacency: ≥2 settled
        // pairs would otherwise absorb into a collapsed group and replace the
        // per-variant cards (the very thing this cell freezes) with digests.
        t.push(RenderBlock::TextDelta {
            id: BlockId(30),
            text: "Edit applied.".to_string(),
            done: true,
        });
        t.push(RenderBlock::ToolCall {
            id: BlockId(3),
            tool_call_id: ToolCallId("call_2".to_string()),
            name: "TodoWrite".to_string(),
            summary: "3 todos".to_string(),
            preview: ToolPreview::Generic {
                name: "TodoWrite".to_string(),
                input_summary: "3 todos".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(4),
            tool_call_id: ToolCallId("call_2".to_string()),
            is_error: false,
            body: ToolResultBody::Todos(vec![
                TodoResultItem {
                    content: "Write the parser".to_string(),
                    active_form: "Writing the parser".to_string(),
                    status: TodoResultStatus::Completed,
                },
                TodoResultItem {
                    content: "Wire the renderer".to_string(),
                    active_form: "Wiring the renderer".to_string(),
                    status: TodoResultStatus::InProgress,
                },
                TodoResultItem {
                    content: "Add golden tests".to_string(),
                    active_form: "Adding golden tests".to_string(),
                    status: TodoResultStatus::Pending,
                },
            ]),
        });
        t.push(RenderBlock::TextDelta {
            id: BlockId(31),
            text: "Checklist updated.".to_string(),
            done: true,
        });
        t.push(RenderBlock::ToolCall {
            id: BlockId(5),
            tool_call_id: ToolCallId("call_3".to_string()),
            name: "Grep".to_string(),
            summary: "fn main".to_string(),
            preview: ToolPreview::Generic {
                name: "Grep".to_string(),
                input_summary: "fn main".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(6),
            tool_call_id: ToolCallId("call_3".to_string()),
            is_error: false,
            body: ToolResultBody::Listing {
                entries: vec![
                    "src/main.rs".to_string(),
                    "src/lib.rs".to_string(),
                    "tests/cli.rs".to_string(),
                ],
                truncated: false,
            },
        });
        t
    });
}

/// P10-B Step 0: freezes the collapse/preview policies — a failing bash keeps
/// its stderr preview (success collapses to the summary line), a >20-line
/// read collapses to a capped preview plus the `+N more` expand hint, and a
/// web_fetch result renders its body as markdown.
#[test]
fn golden_result_edge_previews() {
    golden("result_edge_previews", 30, &[60, 80], || {
        let mut t = Transcript::new();
        t.push(RenderBlock::ToolCall {
            id: BlockId(1),
            tool_call_id: ToolCallId("call_1".to_string()),
            name: "Bash".to_string(),
            summary: "cargo test".to_string(),
            preview: ToolPreview::Bash {
                command: "cargo test".to_string(),
            },
            status: ToolCallStatus::Errored,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(2),
            tool_call_id: ToolCallId("call_1".to_string()),
            is_error: true,
            body: ToolResultBody::Bash(BashResult {
                exit_code: 101,
                stdout: "running 3 tests".to_string(),
                stderr: "error: expected `;`, found `}`".to_string(),
                truncated: false,
            }),
        });
        // Same run-breaking prose separators as `result_variant_cards`.
        t.push(RenderBlock::TextDelta {
            id: BlockId(30),
            text: "Tests failed; reading the file.".to_string(),
            done: true,
        });
        t.push(RenderBlock::ToolCall {
            id: BlockId(3),
            tool_call_id: ToolCallId("call_2".to_string()),
            name: "Read".to_string(),
            summary: "src/big.rs".to_string(),
            preview: ToolPreview::Generic {
                name: "Read".to_string(),
                input_summary: "src/big.rs".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        let long = (1..=25)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        t.push(RenderBlock::ToolResult {
            id: BlockId(4),
            tool_call_id: ToolCallId("call_2".to_string()),
            is_error: false,
            body: ToolResultBody::Read {
                path: "src/big.rs".to_string(),
                content: long,
                language: Some("rust".to_string()),
                truncated: false,
            },
        });
        t.push(RenderBlock::TextDelta {
            id: BlockId(31),
            text: "Fetching the reference page.".to_string(),
            done: true,
        });
        t.push(RenderBlock::ToolCall {
            id: BlockId(5),
            tool_call_id: ToolCallId("call_3".to_string()),
            name: "web_fetch".to_string(),
            summary: "https://example.com".to_string(),
            preview: ToolPreview::Generic {
                name: "web_fetch".to_string(),
                input_summary: "https://example.com".to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(6),
            tool_call_id: ToolCallId("call_3".to_string()),
            is_error: false,
            body: ToolResultBody::Generic {
                name: "web_fetch".to_string(),
                content: "# Example Domain\n\nThis domain is for **illustrative** use."
                    .to_string(),
                truncated: false,
            },
        });
        t
    });
}

/// Locks a COLLAPSED tool group (Summary leader + Hidden members) — the exact
/// surface `RenderCache::Group` (Step 2) will cache. A batch of ≥2 completed
/// tool calls followed by their results collapses into one multi-row summary,
/// so this freezes the per-tool digest rows/rails/status glyphs before the
/// caching change, proving the cache renders byte-identically.
/// The settled three-tool batch the `collapsed_group` golden freezes: a run of
/// completed calls then their results — the shape `compute_tool_groups`
/// collapses into a `Summary` group. Shared with the `RenderCache::Group`
/// equivalence tests so they exercise the exact frozen scenario.
fn settled_group_transcript() -> Transcript {
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: BlockId(1),
        text: "audit the build".to_string(),
    });
    for (n, (name, summary)) in [
        ("Bash", "cargo build"),
        ("Read", "src/main.rs"),
        ("Bash", "cargo test"),
    ]
    .into_iter()
    .enumerate()
    {
        t.push(RenderBlock::ToolCall {
            id: BlockId(10 + n as u64),
            tool_call_id: ToolCallId(format!("call_{n}")),
            name: name.to_string(),
            summary: summary.to_string(),
            preview: ToolPreview::Generic {
                name: name.to_string(),
                input_summary: summary.to_string(),
            },
            status: ToolCallStatus::Ok,
        });
    }
    for n in 0..3u64 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(20 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: format!("result {n}"),
                truncated: false,
            },
        });
    }
    t
}

#[test]
fn golden_collapsed_group() {
    golden("collapsed_group", 20, &[60, 80], settled_group_transcript);
}

/// Index of the collapsed group's Summary leader in
/// [`settled_group_transcript`] (block 0 is the user message; the first tool
/// call leads the run).
const SETTLED_GROUP_LEADER: usize = 1;

#[test]
fn settled_group_cache_engages_and_hit_renders_byte_identical() {
    let mut t = settled_group_transcript();
    let first = snap_styled(60, 20, &mut t);
    assert!(
        matches!(
            t.rendered_cache.get(SETTLED_GROUP_LEADER),
            Some(Some(RenderCache::Group { .. }))
        ),
        "a settled Summary leader must hold a Group cache entry after a draw"
    );
    let second = snap_styled(60, 20, &mut t);
    assert_eq!(
        first, second,
        "a Group cache hit must render byte-identically to the miss that filled it"
    );
}

#[test]
fn live_group_summary_is_never_cached() {
    // Same batch shape, but the last call is still running — the group is
    // live, so its per-tool status markers must keep re-styling per frame.
    let mut t = Transcript::new();
    for n in 0..3u64 {
        let status = if n == 2 {
            ToolCallStatus::Running
        } else {
            ToolCallStatus::Ok
        };
        t.push(RenderBlock::ToolCall {
            id: BlockId(10 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            name: "Bash".to_string(),
            summary: format!("step {n}"),
            preview: ToolPreview::Bash {
                command: format!("step {n}"),
            },
            status,
        });
    }
    for n in 0..2u64 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(20 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: "ok".to_string(),
                truncated: false,
            },
        });
    }
    let _ = snap_styled(60, 20, &mut t);
    assert!(
        !matches!(
            t.rendered_cache.first(),
            Some(Some(RenderCache::Group { .. }))
        ),
        "a live group must not cache its summary (its status markers animate)"
    );
}

#[test]
fn member_rewrite_refreshes_the_cached_group_summary() {
    // A re-upsert of an existing call id rewrites the member in place — the
    // one mutation the Group key (span/err) cannot see. The leader-slot clear
    // in update_existing_tool_call must force a rebuild, or the stale target
    // would keep rendering from the cache.
    let mut t = settled_group_transcript();
    let before = snap_styled(60, 20, &mut t);
    assert!(before.contains("src/main.rs"), "precondition: original target");
    t.push(RenderBlock::ToolCall {
        id: BlockId(90),
        tool_call_id: ToolCallId("call_1".to_string()),
        name: "Read".to_string(),
        summary: "src/lib.rs".to_string(),
        preview: ToolPreview::Generic {
            name: "Read".to_string(),
            input_summary: "src/lib.rs".to_string(),
        },
        status: ToolCallStatus::Ok,
    });
    let after = snap_styled(60, 20, &mut t);
    assert!(
        after.contains("src/lib.rs"),
        "the rewritten member's new target must render"
    );
    assert!(
        !after.contains("src/main.rs"),
        "the stale cached summary must not survive a member rewrite"
    );
}

#[test]
fn relead_live_group_does_not_serve_stale_settled_summary() {
    // Settled -> live re-lead: a parallel batch appended to a settled run and
    // settling OUT OF ORDER re-leads the WHOLE run at the old leader (frozen
    // by mid_settle_parallel_batch_stays_one_live_group_until_all_tools_
    // finish). The layout's live bypass must drop the leader's stale settled
    // Group entry — otherwise the draw's (id, width) match keeps painting the
    // old two-row summary (no spinner, new tools invisible) against the fresh
    // live height for the whole tail of the batch.
    let mut t = Transcript::new();
    for n in 0..2u64 {
        t.push(RenderBlock::ToolCall {
            id: BlockId(10 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            name: "Bash".to_string(),
            summary: format!("step {n}"),
            preview: ToolPreview::Bash {
                command: format!("step {n}"),
            },
            status: ToolCallStatus::Ok,
        });
    }
    for n in 0..2u64 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(20 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: "ok".to_string(),
                truncated: false,
            },
        });
    }
    // Settled draw fills the leader's Group entry.
    let _ = snap_styled(70, 20, &mut t);
    assert!(
        matches!(t.rendered_cache.first(), Some(Some(RenderCache::Group { .. }))),
        "precondition: the settled summary is cached at the leader"
    );

    // A three-wide parallel batch chains onto the same run…
    for n in 2..5u64 {
        t.push(RenderBlock::ToolCall {
            id: BlockId(10 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            name: "Bash".to_string(),
            summary: format!("step {n}"),
            preview: ToolPreview::Bash {
                command: format!("step {n}"),
            },
            status: ToolCallStatus::Running,
        });
    }
    // …and settles out of order: call_3 / call_4 finish while call_2 runs.
    for n in 3..5u64 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(20 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: "ok".to_string(),
                truncated: false,
            },
        });
    }
    let live = snap_styled(70, 20, &mut t);
    assert!(
        !matches!(t.rendered_cache.first(), Some(Some(RenderCache::Group { .. }))),
        "the live bypass must drop the stale settled entry at the re-led leader"
    );
    assert!(
        live.contains("step 4"),
        "the re-led live summary must show the batch's new tools, not the stale two-row settled summary"
    );
}

#[test]
fn settled_group_rekeys_when_settled_calls_are_absorbed() {
    // Membership growth with no status flip anywhere: TWO settled call+result
    // pairs appended to the run form their own summary and are coalesced into
    // the original leader's span (a single appended pair stays a standalone
    // Normal row and never touches the cached summary). The span_len key
    // component must re-key the entry, or the cached three-row summary would
    // keep hiding the absorbed tools.
    let mut t = settled_group_transcript();
    let before = snap_styled(60, 20, &mut t);
    for (n, target) in [(3u64, "cargo doc"), (4u64, "cargo bench")] {
        t.push(RenderBlock::ToolCall {
            id: BlockId(10 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            name: "Bash".to_string(),
            summary: target.to_string(),
            preview: ToolPreview::Bash {
                command: target.to_string(),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(20 + n),
            tool_call_id: ToolCallId(format!("call_{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: "ok".to_string(),
                truncated: false,
            },
        });
    }
    let after = snap_styled(60, 20, &mut t);
    assert_ne!(
        after, before,
        "absorbing settled calls must invalidate the cached group summary"
    );
    assert!(
        after.contains("5 tools") && after.contains("bash ×4") && after.contains("read"),
        "the re-keyed compact digest must account for every absorbed tool"
    );
}

/// A transcript taller than a typical golden viewport, so scroll position is
/// meaningful. Fixed BlockIds keep it order-independent.
fn tall_transcript() -> Transcript {
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: BlockId(1),
        text: "list the dependency layers".to_string(),
    });
    for i in 0..14u64 {
        t.push(RenderBlock::TextDelta {
            id: BlockId(2 + i),
            text: format!("Layer {i:02}: a dependency-graph entry occupying one row."),
            done: true,
        });
    }
    t
}

#[test]
fn golden_scroll_bottom() {
    // Follow-tail (the u16::MAX "Bottom" sentinel today; `ScrollPos::Bottom`
    // after Step 1). Frozen so the ScrollPos refactor is proven identical at
    // the tail.
    golden("scroll_bottom", 12, &[70], || {
        let mut t = tall_transcript();
        t.scroll_to_bottom();
        t
    });
}

#[test]
fn golden_scroll_top() {
    // Pinned to the top (offset 0; `ScrollPos::Rows(0)` after Step 1). Must
    // differ from the bottom cell, proving scroll actually moved the viewport.
    golden("scroll_top", 12, &[70], || {
        let mut t = tall_transcript();
        t.scroll_to_top();
        t
    });
}

/// Time-axis follow semantics that the single-frame style goldens cannot
/// cover: `ScrollPos::Bottom` must keep following the tail across frames as
/// content grows *without* any caller re-asserting `scroll_to_bottom`, while an
/// explicit `ScrollPos::Rows` offset must stay parked. This is the one intended
/// behavior change of the ScrollPos split — the old `u16::MAX` field was
/// collapsed to a concrete offset on every draw, so it only appeared to follow
/// because the app re-asserted the sentinel on each content push. Pinning it
/// here is exactly the "static goldens miss the follow axis" guard the P9 plan
/// calls out.
#[test]
fn bottom_intent_follows_growing_content_but_explicit_offset_stays_parked() {
    let theme = Theme::no_color();
    let (w, h) = (70u16, 8u16);
    let draw = |t: &mut Transcript| {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| t.draw(f, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None))
            .expect("draw");
    };
    // Append `n` tail rows with ids well clear of `tall_transcript`'s 1..=15 so
    // no push merges into an existing block.
    let grow = |t: &mut Transcript, n: u64| {
        for i in 0..n {
            t.push(RenderBlock::TextDelta {
                id: BlockId(100 + i),
                text: format!("appended tail line {i} with body text to fill a row"),
                done: true,
            });
        }
    };

    // Following the tail: the first draw resolves `Bottom` to its max offset.
    let mut following = tall_transcript();
    following.scroll_to_bottom();
    draw(&mut following);
    let at_tail = following.scroll();
    assert!(
        following.is_at_bottom(h, &theme, w, ImageProtocol::None),
        "precondition: scrolled to the tail"
    );

    // Grow the content and redraw WITHOUT re-asserting scroll_to_bottom. The
    // `Bottom` intent must have persisted and followed to the new, larger max.
    grow(&mut following, 6);
    draw(&mut following);
    assert!(
        following.scroll() > at_tail,
        "Bottom intent must follow growing content across frames (was {at_tail}, now {})",
        following.scroll()
    );
    assert!(
        following.is_at_bottom(h, &theme, w, ImageProtocol::None),
        "Bottom intent stays pinned to the tail after growth"
    );

    // Contrast: an explicit offset must NOT follow growth. Pin to the top, grow,
    // and confirm the viewport stays exactly where the user left it.
    let mut pinned = tall_transcript();
    pinned.scroll_to_top();
    draw(&mut pinned);
    assert_eq!(pinned.scroll(), 0, "precondition: pinned to the top");
    grow(&mut pinned, 6);
    draw(&mut pinned);
    assert_eq!(
        pinned.scroll(),
        0,
        "an explicit Rows offset must not follow the growing tail"
    );

    // And once the user leaves the tail (`Bottom` -> `Rows` via scroll_up),
    // later growth must not be followed until they re-engage.
    let mut left_tail = tall_transcript();
    left_tail.scroll_to_bottom();
    draw(&mut left_tail);
    left_tail.scroll_up(2);
    draw(&mut left_tail);
    let parked = left_tail.scroll();
    assert!(
        !left_tail.is_at_bottom(h, &theme, w, ImageProtocol::None),
        "scroll_up leaves the tail"
    );
    grow(&mut left_tail, 6);
    draw(&mut left_tail);
    assert_eq!(
        left_tail.scroll(),
        parked,
        "after leaving the tail, growth is not followed (offset stays parked)"
    );
}

/// Resize/reflow follow — the cleanest trigger for the one intended behavior
/// change, and one neither the fixed-width goldens nor the no-draw
/// `scroll()==u16::MAX` asserts can catch. Narrowing the viewport wraps the same
/// blocks into more rows, growing the max offset with NO content push and NO
/// `scroll_to_bottom` re-assert (the resize path only marks the frame dirty).
/// `Bottom` must re-resolve to the new max and stay pinned. The pre-split field
/// froze at the last-drawn offset here (and could then latch auto-follow off on
/// the next `is_at_bottom`), so this pins the fix, not just parity.
#[test]
fn bottom_intent_stays_pinned_across_a_reflowing_resize() {
    let theme = Theme::no_color();
    let h = 8u16;
    let draw = |t: &mut Transcript, w: u16| {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| t.draw(f, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None))
            .expect("draw");
    };

    let mut t = tall_transcript();
    t.scroll_to_bottom();
    draw(&mut t, 70);
    assert!(
        t.is_at_bottom(h, &theme, 70, ImageProtocol::None),
        "precondition: at the tail while wide"
    );
    let wide_max = t.scroll();

    // Narrow the viewport: same blocks wrap into more rows, so the max offset
    // grows. Only the width changed — no content, no re-assert.
    draw(&mut t, 24);
    assert!(
        t.scroll() > wide_max,
        "narrower reflow must grow the max offset (was {wide_max}, now {})",
        t.scroll()
    );
    assert!(
        t.is_at_bottom(h, &theme, 24, ImageProtocol::None),
        "Bottom stays pinned to the tail across a reflowing resize, not frozen at the old offset"
    );
}

/// The mirror of the above for an EXPLICIT (non-following) offset: it must stay
/// byte-identical to the pre-split field. `draw` normalizes a `Rows` intent to
/// the clamped value each frame (exactly as the old field's write-back did), so
/// a transient content shrink permanently pins the offset — a later regrow must
/// NOT re-anchor to the pre-shrink position. This locks the deliberate choice to
/// keep the follow-OFF path identical while only `Bottom` gains the durable
/// follow. A wider viewport (fewer wrapped rows) provides the shrink.
#[test]
fn explicit_offset_is_clamped_and_does_not_re_anchor_after_a_shrink() {
    let theme = Theme::no_color();
    let h = 8u16;
    let draw = |t: &mut Transcript, w: u16| {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| t.draw(f, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None))
            .expect("draw");
    };

    // Explicit offset near the tail while narrow (large max).
    let mut t = tall_transcript();
    draw(&mut t, 24);
    t.scroll_to_bottom();
    draw(&mut t, 24);
    t.scroll_up(1); // leave the tail -> explicit Rows offset near the bottom
    draw(&mut t, 24);
    let narrow_offset = t.scroll();
    assert!(narrow_offset > 0, "precondition: parked at a non-zero offset");

    // Shrink by widening (fewer wrapped rows -> smaller max). The offset clamps
    // down and the intent is normalized to that clamped value.
    draw(&mut t, 70);
    let shrunk_offset = t.scroll();
    assert!(
        shrunk_offset < narrow_offset,
        "widening shrinks content and clamps the offset ({shrunk_offset} < {narrow_offset})"
    );

    // Regrow by narrowing again: the offset must NOT jump back to narrow_offset
    // (that would be the follow-off improvement we deliberately did not ship).
    draw(&mut t, 24);
    assert_eq!(
        t.scroll(),
        shrunk_offset,
        "an explicit offset stays clamped after a shrink and does not re-anchor on regrow"
    );
}

#[test]
fn parallel_block_tables_stay_aligned_through_every_mutation() {
    // P9 Step 3-lite: the four per-block tables (blocks / rendered_cache /
    // render_versions / tool_groups) must move in lockstep through every
    // structural mutation — push, mid-list removal, clear, and the front-drain
    // cap prune. A drift here silently mis-attributes caches and group states
    // to the wrong blocks.
    let aligned = |t: &Transcript| {
        t.blocks.len() == t.render_versions.len()
            && t.blocks.len() == t.tool_groups.len()
            && t.rendered_cache.len() <= t.blocks.len()
    };

    let mut t = Transcript::new();
    for i in 0..10u64 {
        t.push(text_block(i + 1));
        assert!(aligned(&t), "push must keep the tables aligned");
    }
    assert_eq!(t.blocks.len(), 10);

    t.remove_block_at(3);
    assert!(aligned(&t), "mid-list removal must keep the tables aligned");
    assert_eq!(t.blocks.len(), 9);

    t.clear();
    assert!(aligned(&t), "clear must reset every table together");
    assert_eq!(t.blocks.len(), 0);

    // Push far enough past the (test-profile) block cap to trigger the
    // front-drain prune at least once.
    for i in 0..2_200u64 {
        t.push(text_block(10_000 + i));
    }
    assert!(aligned(&t), "the cap prune must front-drain every table together");
    assert!(
        t.blocks.len() <= 2_048 + 64,
        "the cap must actually have pruned (len {})",
        t.blocks.len()
    );
}

#[test]
fn reasoning_delta_rejoins_segment_split_by_midstream_notice() {
    // 스트리밍 리즈닝과 다음 델타 사이에 시스템 공지가 append 되면(마우스
    // 복사의 "Copied block to clipboard…" 토스트가 실제 침입자였다) tail-only
    // 병합이 끊겨 같은 id 의 중복 블록이 생기고, 원본은 done=false 고아로
    // 남아 "✦ Thinking… · N.Ns" 를 영구히 그렸다. 델타는 침입자를 건너 원본
    // 블록으로 재접합되어야 한다.
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.set_turn_active(true);
    let reasoning_id = id();
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "planning the fix".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::System {
        id: id(),
        level: SystemLevel::Info,
        text: "Copied block to clipboard via pbcopy (99 chars)".to_string(),
    });
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: " and shipping it".to_string(),
        signature: Some("sig".to_string()),
        done: true,
    });

    let reasonings: Vec<_> = t
        .blocks()
        .iter()
        .filter_map(|block| match block {
            RenderBlock::Reasoning {
                text,
                signature,
                done,
                ..
            } => Some((text.clone(), signature.clone(), *done)),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasonings,
        vec![(
            "planning the fix and shipping it".to_string(),
            Some("sig".to_string()),
            true
        )],
        "the split segment must re-join into the single original block"
    );

    // 정착 후 어떤 행에도 라이브 Thinking 큐가 남아 있으면 안 된다.
    let backend = TestBackend::new(64, 10);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 64, 10), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 64, 10);
    assert!(
        crate::tui::blocks::reasoning::ZO_REVEAL_VERBS
            .iter()
            .all(|verb| !dumped.contains(verb)),
        "no orphaned live zo cue may survive the settle: {dumped}"
    );
}

#[test]
fn reasoning_rejoin_never_crosses_turn_or_steering_boundary() {
    // 블록 id 는 턴마다 0부터 재시작한다: 지난 턴/스티어링 이전의 동일-id
    // 리즈닝에 이어 붙이면 무관한 과거 블록이 오염된다. 경계 밖에서는
    // 재접합하지 않고 새 블록을 append 해야 한다.
    let mut t = Transcript::new();
    let reasoning_id = id();

    // 턴 비활성: 재접합 자체가 꺼진다.
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "old turn".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::System {
        id: id(),
        level: SystemLevel::Info,
        text: "notice".to_string(),
    });
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "new segment".to_string(),
        signature: None,
        done: false,
    });
    let count = t
        .blocks()
        .iter()
        .filter(|block| matches!(block, RenderBlock::Reasoning { .. }))
        .count();
    assert_eq!(count, 2, "no turn active → plain append, no rejoin");

    // 미드턴 스티어링(UserMessage) 뒤의 같은 id 도 경계를 넘지 않는다.
    let mut t = Transcript::new();
    t.set_turn_active(true);
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "before steering".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "steer!".to_string(),
    });
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "after steering".to_string(),
        signature: None,
        done: false,
    });
    let texts: Vec<_> = t
        .blocks()
        .iter()
        .filter_map(|block| match block {
            RenderBlock::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        vec!["before steering", "after steering"],
        "a UserMessage is a hard rejoin boundary"
    );
}

#[test]
fn streaming_thinking_placeholder_hides_after_visible_prose() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: String::new(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "final answer".to_string(),
        done: true,
    });

    let backend = TestBackend::new(56, 8);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 56, 8), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 56, 8);

    assert!(
        dumped.contains("final answer"),
        "prose should render: {dumped}"
    );
    assert!(
        !dumped.contains("Thinking")
            && crate::tui::blocks::reasoning::ZO_REVEAL_VERBS
                .iter()
                .all(|verb| !dumped.contains(verb)),
        "the streaming zo/thinking placeholder should disappear after prose: {dumped}"
    );
}

#[test]
fn non_empty_streaming_reasoning_disappears_after_visible_prose() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: "I am checking the files".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "final answer".to_string(),
        done: true,
    });

    let backend = TestBackend::new(64, 10);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 64, 10), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 64, 10);

    assert!(
        !dumped.contains("Thinking")
            && crate::tui::blocks::reasoning::ZO_REVEAL_VERBS
                .iter()
                .all(|verb| !dumped.contains(verb)),
        "streaming zo/thinking cue should disappear after prose starts: {dumped}"
    );
    assert!(
        dumped.contains("final answer"),
        "prose should render: {dumped}"
    );
}

#[test]
fn reasoning_tail_prose_merge_keeps_layout_on_suffix_fast_path() {
    // Long-session hot path: once visible prose follows an unfinished Reasoning
    // block, every subsequent token merge marks the Reasoning index dirty so
    // its placeholder stays collapsed. That must rebuild only the
    // Reasoning/TextDelta suffix, not rescan all blocks for tool groups.
    let theme = Theme::default_dark();
    let (w, h) = (80u16, 24u16);
    let draw = |t: &mut Transcript| {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("backend");
        terminal
            .draw(|f| t.draw(f, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None))
            .expect("draw");
    };

    let mut t = Transcript::new();
    for i in 0..300u64 {
        t.push(RenderBlock::TextDelta {
            id: BlockId(10_000 + i),
            text: format!("old transcript block {i}"),
            done: true,
        });
    }
    t.push(RenderBlock::Reasoning {
        id: BlockId(20_000),
        text: "thinking before prose".to_string(),
        signature: None,
        done: false,
    });

    // Warm the full cache, then append the first prose block so the reasoning
    // cue is already hidden before the token-merge path under test.
    draw(&mut t);
    t.push(RenderBlock::TextDelta {
        id: BlockId(20_001),
        text: "a".to_string(),
        done: false,
    });
    draw(&mut t);

    reset_compute_tool_groups_call_count();
    t.push(RenderBlock::TextDelta {
        id: BlockId(20_001),
        text: "b".to_string(),
        done: false,
    });
    draw(&mut t);

    assert_eq!(
        compute_tool_groups_call_count(),
        0,
        "reasoning+tail token merge should not run full compute_tool_groups"
    );
    assert_eq!(
        t.tool_groups,
        compute_tool_groups(&t.blocks),
        "suffix fast path must preserve authoritative tool-group state"
    );
}

#[test]
fn streaming_reasoning_disappears_after_visible_tool_call() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: "I am about to inspect a file".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("call-after-reasoning".to_string()),
        name: "Cargo".to_string(),
        summary: "test".to_string(),
        preview: ToolPreview::Generic {
            name: "Cargo".to_string(),
            input_summary: "test".to_string(),
        },
        status: ToolCallStatus::Ok,
    });

    let backend = TestBackend::new(72, 10);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 72, 10), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 72, 10);

    assert!(dumped.contains("Cargo"), "tool call should render: {dumped}");
    assert!(
        !dumped.contains("Thinking")
            && crate::tui::blocks::reasoning::ZO_REVEAL_VERBS
                .iter()
                .all(|verb| !dumped.contains(verb)),
        "streaming zo/thinking cue should disappear after tool work starts: {dumped}"
    );
}

#[test]
fn streaming_reasoning_disappears_after_visible_tool_result() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: "I am reading the command output".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::ToolResult {
        id: id(),
        tool_call_id: ToolCallId("result-after-reasoning".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "tool output is ready".to_string(),
            truncated: false,
        },
    });

    let backend = TestBackend::new(72, 10);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 72, 10), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 72, 10);

    assert!(
        dumped.contains("tool output is ready"),
        "tool result should render: {dumped}"
    );
    assert!(
        !dumped.contains("Thinking")
            && crate::tui::blocks::reasoning::ZO_REVEAL_VERBS
                .iter()
                .all(|verb| !dumped.contains(verb)),
        "streaming zo/thinking cue should disappear after tool output starts: {dumped}"
    );
}

#[test]
fn completed_reasoning_renders_when_expanded() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: "checked the cache predicate".to_string(),
        signature: None,
        done: true,
    });

    assert!(t.focus_next(), "reasoning blocks must be focusable");
    assert!(t.toggle_expanded(), "focused reasoning block should expand");

    let backend = TestBackend::new(72, 8);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 72, 8), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 72, 8);

    assert!(
        dumped.contains("work steps"),
        "expanded completed reasoning should render its header: {dumped}"
    );
    assert!(
        dumped.contains("checked the cache predicate"),
        "expanded completed reasoning should render its body: {dumped}"
    );
}

#[test]
fn streaming_reasoning_after_prose_stays_hidden_even_when_expanded() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: "stale transient thought".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "final answer".to_string(),
        done: true,
    });

    assert!(t.focus_next(), "reasoning blocks must be focusable");
    assert!(
        t.toggle_expanded(),
        "expanding hidden live reasoning should not reveal it"
    );

    let backend = TestBackend::new(72, 10);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 72, 10), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 72, 10);

    assert!(
        dumped.contains("final answer"),
        "prose should still render: {dumped}"
    );
    assert!(
        !dumped.contains("stale transient thought") && !dumped.contains("work steps"),
        "expanded stale streaming reasoning must remain hidden after prose starts: {dumped}"
    );
}

#[test]
fn hidden_streaming_reasoning_preserves_user_to_answer_separator() {
    let theme = Theme::no_color();
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "prompt".to_string(),
    });
    t.push(RenderBlock::Reasoning {
        id: id(),
        text: "transient thought".to_string(),
        signature: None,
        done: false,
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "final answer".to_string(),
        done: true,
    });

    let backend = TestBackend::new(72, 14);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 72, 14), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 72, 14);

    assert!(
        dumped.contains("prompt") && dumped.contains("final answer"),
        "prompt and answer should both render: {dumped}"
    );
    assert!(
        dumped.contains("----------------"),
        "hidden streaming reasoning must not swallow the user→answer separator: {dumped}"
    );
}

/// A live `/theme` switch must drop the per-block render cache: the
/// cached lines bake in the old palette but are keyed only by content +
/// width, so without invalidation they would keep showing old colors.
#[test]
fn invalidate_render_cache_drops_entries_without_desync() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "**bold** body text".to_string(),
        done: true,
    });
    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 40, 8), &theme, 0, ImageProtocol::None))
        .expect("draw");

    // The draw populated the per-block render cache.
    assert!(
        t.rendered_cache.iter().any(Option::is_some),
        "draw should populate the render cache"
    );

    t.invalidate_render_cache();

    // Every slot is dropped, and the cache stays parallel to `blocks`
    // (resetting slots to `None`, never clearing the Vec) so indexed
    // access on the next draw cannot panic.
    assert_eq!(t.rendered_cache.len(), t.blocks.len());
    assert!(
        t.rendered_cache.iter().all(Option::is_none),
        "theme invalidation must drop every cached entry"
    );
}

#[test]
fn transcript_clears_preexisting_cells_in_its_region() {
    let theme = Theme::default_dark();
    let backend = TestBackend::new(56, 8);
    let mut terminal = Terminal::new(backend).expect("backend");
    let mut transcript = Transcript::new();
    let area = Rect::new(0, 0, 56, 8);

    terminal
        .draw(|frame| {
            frame.render_widget(ratatui::widgets::Paragraph::new("stale long tail"), area);
            transcript.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw frame");

    let buffer = terminal.backend().buffer();
    let rendered = (0..8)
        .map(|y| {
            (0..56)
                .map(|x| {
                    buffer
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !rendered.contains("stale long tail"),
        "transcript must clear its own area before drawing: {rendered:?}"
    );
}

#[test]
fn transcript_local_clear_does_not_force_full_backend_redraw() {
    let theme = Theme::default_dark();
    let backend = CountingBackend::new(56, 8);
    let mut terminal = Terminal::new(backend).expect("backend");
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::TextDelta {
        id: id(),
        text: "stable assistant text".to_string(),
        done: true,
    });

    for _ in 0..3 {
        terminal
            .draw(|frame| {
                transcript.draw(
                    frame,
                    Rect::new(0, 0, 56, 8),
                    &theme,
                    0,
                    ImageProtocol::None,
                );
            })
            .expect("draw frame");
    }

    let draw_counts = &terminal.backend().draw_counts;
    assert_eq!(draw_counts.len(), 3);
    assert!(
        draw_counts[2] < 56 * 8 / 4,
        "local Clear should not invalidate the final frame diff: {draw_counts:?}"
    );
}

#[test]
fn visible_layout_lookup_jumps_to_tail_without_front_scan() {
    let layout = vec![(0, 0, 3), (1, 4, 3), (2, 8, 3), (3, 12, 3)];

    assert_eq!(first_visible_layout_entry(&layout, 0), 0);
    assert_eq!(first_visible_layout_entry(&layout, 3), 1);
    assert_eq!(first_visible_layout_entry(&layout, 9), 2);
    assert_eq!(first_visible_layout_entry(&layout, 15), 4);
}

#[test]
fn tool_group_tail_recompute_matches_full_recompute() {
    fn call(n: usize) -> RenderBlock {
        RenderBlock::ToolCall {
            id: id(),
            tool_call_id: ToolCallId(format!("call_{n}")),
            name: "Bash".to_string(),
            summary: "echo".to_string(),
            preview: ToolPreview::Generic {
                name: "Bash".to_string(),
                input_summary: "echo".to_string(),
            },
            status: ToolCallStatus::Ok,
        }
    }

    fn result(n: usize) -> RenderBlock {
        RenderBlock::ToolResult {
            id: id(),
            tool_call_id: ToolCallId(format!("call_{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: "ok".to_string(),
                truncated: false,
            },
        }
    }

    let mut blocks = (0..128)
        .map(|i| RenderBlock::TextDelta {
            id: id(),
            text: format!("prefix {i}"),
            done: true,
        })
        .collect::<Vec<_>>();
    blocks.push(call(1));
    blocks.push(result(1));
    let append_at = blocks.len();

    let mut states = compute_tool_groups(&blocks);
    blocks.push(call(2));
    blocks.push(result(2));

    let recompute_from = tool_group_recompute_start(&blocks, append_at);
    assert_eq!(
        recompute_from, 128,
        "long non-tool prefixes should not be rescanned"
    );
    recompute_tool_groups_tail(&blocks, &mut states, recompute_from);

    assert_eq!(states, compute_tool_groups(&blocks));
}

#[test]
fn mid_list_tool_status_flip_matches_full_recompute() {
    // ensure_layout Case 2c: a parallel tool settling out of order is a mid-list
    // (non-tail) ToolCall mutation. The localized per-run recompute must produce
    // the exact same tool-group states as a from-scratch `compute_tool_groups`,
    // so the O(run) fast path can never drift from the authoritative grouping.
    let theme = Theme::default_dark();
    let (w, h) = (80u16, 24u16);
    let draw = |t: &mut Transcript| {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("backend");
        terminal
            .draw(|f| t.draw(f, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None))
            .expect("draw");
    };

    let mut t = Transcript::new();
    // Leading non-tool block so the first tool run does not start at index 0 —
    // the localized recompute must leave that prefix untouched.
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "intro".to_string(),
        done: true,
    });
    // Run A: a 2-wide parallel batch still in flight (Running, no results yet →
    // the whole cluster is Hidden). Prose, then a settled Run B (call + result →
    // a real Summary leader that the localized recompute must preserve), then a
    // trailing block so Run A's calls are mid-list, not the tail.
    t.push(bash_call(1, ToolCallStatus::Running));
    t.push(bash_call(2, ToolCallStatus::Running));
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "between runs".to_string(),
        done: true,
    });
    t.push(bash_call(3, ToolCallStatus::Ok));
    t.push(bash_result(3));
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "final".to_string(),
        done: true,
    });

    // Establish a clean layout cache (Run A hidden while in flight).
    draw(&mut t);
    let groups_before = t.tool_groups.clone();

    // Run A settles out of order: both calls flip Running -> Ok. Each is an
    // in-place upsert (neither is the tail), marking the layout dirty mid-list
    // without changing the block count.
    t.push(bash_call(1, ToolCallStatus::Ok));
    t.push(bash_call(2, ToolCallStatus::Ok));
    // Next draw routes through Case 2c (lowest dirty index is a ToolCall).
    draw(&mut t);

    // Core safety property: the localized run recompute never drifts from the
    // authoritative full grouping.
    assert_eq!(
        t.tool_groups,
        compute_tool_groups(&t.blocks),
        "mid-list tool status flip must yield the same groups as a full recompute"
    );
    // The flip is a genuine state transition (in-flight → settled), so the test
    // exercises a real regrouping rather than a parity-preserving no-op.
    assert_ne!(
        t.tool_groups, groups_before,
        "settling a mid-list in-flight run must change its group state"
    );
    // And the upsert landed mid-list: both calls now read Ok.
    let run_a_statuses: Vec<_> = t
        .blocks
        .iter()
        .filter_map(|b| match b {
            RenderBlock::ToolCall {
                tool_call_id,
                status,
                ..
            } if matches!(tool_call_id.0.as_str(), "call_1" | "call_2") => Some(*status),
            _ => None,
        })
        .collect();
    assert_eq!(run_a_statuses, vec![ToolCallStatus::Ok, ToolCallStatus::Ok]);
}

#[test]
#[ignore = "wall-clock perf demonstration; run with `--ignored --nocapture`"]
fn expanded_tool_result_draw_cost_is_windowed_not_proportional_to_size() {
    // Direct proof of the ToolResult windowing fix: a steady-state (warm-cache)
    // draw of a huge expanded tool body must cost ~the same as a tiny one, because
    // only the visible viewport rows are wrapped. Before the fix the whole body
    // was re-wrapped every frame, so cost scaled with body size (the "big output
    // on screen / scroll lags" symptom).
    use std::time::Instant;

    fn build(lines: usize) -> Transcript {
        let mut t = Transcript::new();
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: "header".to_string(),
            done: true,
        });
        let content = (0..lines)
            .map(|i| format!("output line {i}: lorem ipsum dolor sit amet consectetur adipiscing"))
            .collect::<Vec<_>>()
            .join("\n");
        t.push(RenderBlock::ToolResult {
            id: id(),
            tool_call_id: ToolCallId("call_big".to_string()),
            is_error: false,
            body: ToolResultBody::Text {
                content,
                truncated: false,
            },
        });
        // Focus the tool result and expand it so the full (capped) body renders,
        // then park the viewport at the bottom so the window must skip far into
        // the body — the case the old full re-wrap paid the most for.
        while t.focus_next() {
            let on_result = t
                .focused_idx()
                .and_then(|i| t.blocks.get(i))
                .is_some_and(|b| matches!(b, RenderBlock::ToolResult { .. }));
            if on_result {
                break;
            }
        }
        assert!(t.toggle_expanded(), "tool result should expand");
        t.scroll_to_bottom();
        t
    }

    fn avg_draw_ms(t: &mut Transcript, theme: &Theme, w: u16, h: u16, iters: u32) -> f64 {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("backend");
        // Warm the cache (first draw fills `row_prefix`).
        term.draw(|f| t.draw(f, Rect::new(0, 0, w, h), theme, 0, ImageProtocol::None))
            .expect("draw");
        let start = Instant::now();
        for tick in 0..iters {
            term.draw(|f| {
                t.draw(f, Rect::new(0, 0, w, h), theme, u64::from(tick), ImageProtocol::None);
            })
            .expect("draw");
        }
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters)
    }

    let theme = Theme::default_dark();
    let (w, h) = (100u16, 40u16);
    let mut small = build(40);
    // 4000 source lines → clamped to EXPANDED_HARD_CAP (2000) rendered lines.
    let mut big = build(4000);

    let small_ms = avg_draw_ms(&mut small, &theme, w, h, 300);
    let big_ms = avg_draw_ms(&mut big, &theme, w, h, 300);
    let ratio = big_ms / small_ms.max(1e-6);
    // Guard against a trivial pass: the big body must genuinely be huge (far past
    // the viewport), so the windowed draw really is skipping most of it.
    assert!(
        big.content_total() > 500,
        "big tool result must render a large body for this test to mean anything (got {} rows)",
        big.content_total()
    );
    eprintln!(
        "[DIRECT-TEST] expanded tool-result steady-state draw: small(40 lines)={small_ms:.4}ms  big(~2000 lines)={big_ms:.4}ms  ratio={ratio:.2}x"
    );

    // Windowed → ~flat in body size. A non-windowed (full re-wrap) draw would be
    // tens of times slower for the big body; allow generous headroom for noise
    // and the fixed per-frame Clear/diff cost so the assert is robust, not flaky.
    assert!(
        big_ms < small_ms * 5.0 + 0.5,
        "windowed tool-result draw must not scale with body size: big={big_ms:.4}ms small={small_ms:.4}ms ratio={ratio:.2}x"
    );
}

fn bash_call(n: usize, status: ToolCallStatus) -> RenderBlock {
    RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId(format!("call_{n}")),
        name: "Bash".to_string(),
        summary: "echo".to_string(),
        preview: ToolPreview::Bash {
            command: "echo".to_string(),
        },
        status,
    }
}

fn bash_result(n: usize) -> RenderBlock {
    RenderBlock::ToolResult {
        id: id(),
        tool_call_id: ToolCallId(format!("call_{n}")),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    }
}

fn summary_leaders(states: &[ToolGroupState]) -> Vec<&ToolGroupState> {
    states
        .iter()
        .filter(|state| matches!(state, ToolGroupState::Summary { .. }))
        .collect()
}

#[test]
fn tool_call_updates_upsert_by_call_id_even_when_not_tail() {
    let mut t = Transcript::new();
    t.push(bash_call(1, ToolCallStatus::Pending));
    t.push(bash_call(2, ToolCallStatus::Pending));

    // Runtime can emit a second Running update for call_1 while call_2 is the
    // transcript tail. This must update call_1 in place, not append a duplicate
    // active row that keeps `tools active` stale after results arrive.
    t.push(bash_call(1, ToolCallStatus::Running));

    let tool_calls: Vec<_> = t
        .blocks
        .iter()
        .filter_map(|block| match block {
            RenderBlock::ToolCall {
                tool_call_id,
                status,
                ..
            } => Some((tool_call_id.0.as_str(), *status)),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_calls,
        vec![
            ("call_1", ToolCallStatus::Running),
            ("call_2", ToolCallStatus::Pending),
        ],
        "non-tail tool updates should be upserts, not appended duplicates"
    );

    t.push(bash_result(1));
    let call_1_statuses: Vec<_> = t
        .blocks
        .iter()
        .filter_map(|block| match block {
            RenderBlock::ToolCall {
                tool_call_id,
                status,
                ..
            } if tool_call_id.0 == "call_1" => Some(*status),
            _ => None,
        })
        .collect();
    assert_eq!(
        call_1_statuses,
        vec![ToolCallStatus::Ok],
        "result reconciliation should not leave duplicate running call_1 rows"
    );
}

#[test]
fn in_flight_tool_clusters_form_a_live_group() {
    // CC parity: a parallel in-flight batch is visible in the transcript as a
    // live group — one Summary leader whose rows animate per-tool status —
    // instead of being hidden until everything settles.
    let blocks = vec![
        bash_call(1, ToolCallStatus::Running),
        bash_call(2, ToolCallStatus::Pending),
        bash_call(3, ToolCallStatus::Running),
    ];

    let states = compute_tool_groups(&blocks);
    assert!(
        matches!(
            states[0],
            ToolGroupState::Summary {
                total: 3,
                running_count: 2,
                pending_count: 1,
                ..
            }
        ),
        "an in-flight batch should form a live group leader carrying its running/pending tallies: {states:?}"
    );
    assert!(
        states[1..]
            .iter()
            .all(|state| matches!(state, ToolGroupState::Hidden)),
        "live group members collapse under the leader: {states:?}"
    );
}

#[test]
fn mid_settle_parallel_batch_stays_one_live_group_until_all_tools_finish() {
    // A 4-wide parallel batch settling unevenly: two calls finished and two
    // are still Running. The whole contiguous cluster stays ONE live group —
    // settled members flip their row marker to ✓ in place while the running
    // members keep animating — so the batch neither splits nor flickers.
    let blocks = vec![
        bash_call(1, ToolCallStatus::Ok),
        bash_call(2, ToolCallStatus::Ok),
        bash_call(3, ToolCallStatus::Running),
        bash_call(4, ToolCallStatus::Running),
        bash_result(1),
        bash_result(2),
    ];

    let states = compute_tool_groups(&blocks);
    let leaders = summary_leaders(&states);
    assert_eq!(
        leaders.len(),
        1,
        "a mid-settle batch is one live group, not fragments: {states:?}"
    );
    assert!(
        matches!(
            states[0],
            ToolGroupState::Summary {
                total: 4,
                running_count: 2,
                ..
            }
        ),
        "the live leader counts every member and the in-flight tally: {states:?}"
    );
    assert!(
        states[1..]
            .iter()
            .all(|state| matches!(state, ToolGroupState::Hidden)),
        "members stay collapsed under the live leader: {states:?}"
    );
}

#[test]
fn settled_batch_then_lone_pending_keeps_settled_summary_visible() {
    // A new tool iteration must not erase the stable completed-tool summary
    // immediately before it. Gemini often emits quick tool loops without prose
    // between them; regrouping the whole contiguous run makes the settled
    // `glob/read … +N more` rows disappear and reappear.
    let blocks = vec![
        bash_call(1, ToolCallStatus::Ok),
        bash_result(1),
        bash_call(2, ToolCallStatus::Ok),
        bash_result(2),
        bash_call(3, ToolCallStatus::Ok),
        bash_result(3),
        bash_call(99, ToolCallStatus::Pending),
    ];

    let states = compute_tool_groups(&blocks);
    let leaders = summary_leaders(&states);
    assert_eq!(
        leaders.len(),
        1,
        "settled prefix should remain visible while the new suffix is pending: {states:?}"
    );
    assert!(
        matches!(states[0], ToolGroupState::Summary { total: 3, .. }),
        "completed prefix should keep its three-tool summary: {states:?}"
    );
    assert!(
        states[1..=5]
            .iter()
            .all(|state| matches!(state, ToolGroupState::Hidden)),
        "completed prefix members should stay collapsed under the visible leader: {states:?}"
    );
    assert!(
        matches!(states[6], ToolGroupState::Normal),
        "a lone in-flight suffix renders as its own live event row: {states:?}"
    );
}

#[test]
fn inflight_suffix_batch_behind_settled_summary_forms_its_own_live_group() {
    // Settled 2-tool summary, then a NEW 2-wide parallel batch starts in the
    // same contiguous run. The settled leader must keep its counts while the
    // suffix animates as its own live group with its own leader.
    let blocks = vec![
        bash_call(1, ToolCallStatus::Ok),
        bash_result(1),
        bash_call(2, ToolCallStatus::Ok),
        bash_result(2),
        bash_call(10, ToolCallStatus::Running),
        bash_call(11, ToolCallStatus::Pending),
    ];

    let states = compute_tool_groups(&blocks);
    assert!(
        matches!(states[0], ToolGroupState::Summary { total: 2, running_count: 0, .. }),
        "settled prefix keeps its own leader and counts: {states:?}"
    );
    assert!(
        matches!(
            states[4],
            ToolGroupState::Summary {
                total: 2,
                running_count: 1,
                pending_count: 1,
                ..
            }
        ),
        "the in-flight suffix owns a separate live leader: {states:?}"
    );
    assert!(
        matches!(states[5], ToolGroupState::Hidden),
        "the live group's second member collapses under its leader: {states:?}"
    );
}

#[test]
fn final_text_merge_in_same_batch_as_append_refreshes_render() {
    // Regression: the final streaming delta (text grows + `done`) merging into
    // the tail block *in the same drain batch* as the next ToolCall append
    // used to leave the text block's truncated streaming render cached forever
    // — the suffix rebuild started at the appended block, the merged block's
    // height/render were never re-measured, and the draw path trusts the
    // cache verbatim ("…파악하고 팬"에서 영구 잘림).
    let theme = Theme::default_dark();
    let text_id = id();
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "start".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: text_id,
        text: "alpha beta gam".to_string(),
        done: false,
    });

    // Frame 1: caches the open streaming render ("…gam").
    let backend = TestBackend::new(60, 12);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 60, 12), &theme, 0, ImageProtocol::None))
        .expect("draw");

    // Same batch, no draw in between: the closing delta merges into the tail,
    // then the model's first tool call appends right behind it.
    t.push(RenderBlock::TextDelta {
        id: text_id,
        text: "ma done.".to_string(),
        done: true,
    });
    t.push(bash_call(900, ToolCallStatus::Running));

    // Frame 2 must show the completed sentence, not the stale streaming tail.
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 60, 12), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 60, 12);
    assert!(
        dumped.contains("alpha beta gamma done."),
        "merged final text must be re-rendered after a same-batch append: {dumped}"
    );
}

#[test]
fn mid_list_system_upsert_remeasures_following_layout() {
    // Regression: `upsert_system` replaces a mid-list block in place. When the
    // replacement wraps to *more* rows (live fan-out progress grows), the
    // layout must re-measure from that index — the tail-only fast path left
    // every later block at a stale offset, overdrawing the grown block.
    let theme = Theme::default_dark();
    let progress_id = id();
    let mut t = Transcript::new();
    t.push(RenderBlock::System {
        id: progress_id,
        level: SystemLevel::Info,
        text: "fanout: starting".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "tail prose".to_string(),
        done: true,
    });

    let backend = TestBackend::new(40, 12);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 40, 12), &theme, 0, ImageProtocol::None))
        .expect("draw");

    // Grow the progress block by two wrapped lines.
    t.upsert_system(
        progress_id,
        SystemLevel::Info,
        "fanout: running\nagent-1 reading\nagent-2 testing".to_string(),
    );
    terminal
        .draw(|f| t.draw(f, Rect::new(0, 0, 40, 12), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 40, 12);
    for expected in ["agent-2 testing", "tail prose"] {
        assert!(
            dumped.contains(expected),
            "grown progress block and the block after it must both be visible: {dumped}"
        );
    }
}

/// Structural regression guard for the low-chrome transcript. Also prints the
/// rendered sample under `--nocapture` so the layout can be eyeballed.
#[test]
fn sample_conversation_shows_codex_style_events_without_global_tool_rail() {
    let dump = render_sample(64, 22);
    println!(
        "\n┌─ transcript sample ─────────────────────────────\n{dump}└──────────────────────────────────────────────────"
    );

    // The user header and assistant author bullets attribute adjacent
    // prompt/answer rows so a user request and the model's first sentence do
    // not read as one author.
    assert!(dump.contains("You"), "user role header missing:\n{dump}");
    // Role marks remain local to the attributed user/assistant prose blocks;
    // the old global tool/transcript rail must not reappear.
    assert!(dump.contains('\u{2503}'), "user bar ┃ missing:\n{dump}");
    assert_eq!(
        dump.matches('\u{25c6}').count(),
        2,
        "assistant prose should carry a ◆ bullet after the user prompt and after the tool result:\n{dump}"
    );
    // The bullet rides the first body row; wrapped continuation rows keep the
    // same body column via the indent (no rail glyph, no header row).
    assert!(
        dump.contains("\u{25c6}  의") && dump.contains("\u{25c6}  핵") && dump.contains("   없"),
        "assistant body must start on the bullet row and wrapped rows keep the indent column:\n{dump}"
    );
    assert!(
        !dump.contains("zo") && !dump.contains('\u{2502}'),
        "the retired Zo header / │ prose rail resurfaced:\n{dump}"
    );
    assert!(
        !dump.contains('\u{251C}'),
        "├ tool branch resurfaced:\n{dump}"
    );
    assert!(
        dump.contains("Ran cargo metadata"),
        "tool event missing:\n{dump}"
    );
    assert!(dump.contains("└ 12 crates"), "tool result child missing:\n{dump}");
    assert!(
        !dump.contains("• 12 crates"),
        "tool result should not compete with its call as another root:\n{dump}"
    );
    // The retired decorations must not reappear: the thin ▏ gutter, the
    // ╰─► result leader, and the › tool-call leader.
    assert!(
        !dump.contains('\u{258F}'),
        "old ▏ gutter resurfaced:\n{dump}"
    );
    assert!(
        !dump.contains('\u{2570}'),
        "old ╰─► result leader resurfaced:\n{dump}"
    );
    assert!(
        !dump.contains('\u{203A}'),
        "old › tool leader resurfaced:\n{dump}"
    );
}

#[test]
fn orphan_assistant_output_uses_low_chrome_event_rows() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "Recovered assistant output after resume.".to_string(),
        done: true,
    });
    t.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("orphan_bash".to_string()),
        name: "Bash".to_string(),
        summary: "echo ok".to_string(),
        preview: ToolPreview::Generic {
            name: "Bash".to_string(),
            input_summary: "echo ok".to_string(),
        },
        status: ToolCallStatus::Ok,
    });

    let backend = TestBackend::new(64, 8);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| {
            t.draw(
                frame,
                Rect::new(0, 0, 64, 8),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw transcript");

    let dump = dump_terminal(&terminal, 64, 8);
    assert!(
        dump.contains("Recovered assistant output after resume."),
        "orphan assistant text must remain visible:\n{dump}"
    );
    assert!(
        dump.contains("Ran echo ok"),
        "orphan tool call must render as a Codex event row:\n{dump}"
    );
    assert!(
        !dump.contains('\u{2502}') && !dump.contains('\u{251C}'),
        "orphan assistant/tool output must not restore the global rail:\n{dump}"
    );
}

/// `scroll_to_block` falls back to a constant-derived estimate when no
/// layout cache exists yet (e.g. a search jump before the first draw).
/// The estimate must track [`DEFAULT_BLOCK_ROWS`] (+1 for the gap) rather
/// than a bare magic number, so search jumps land near the target.
#[test]
fn scroll_to_block_fallback_uses_default_block_rows_estimate() {
    use super::DEFAULT_BLOCK_ROWS;
    let mut t = Transcript::new();
    for i in 0..10 {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("line {i}"),
            done: true,
        });
    }
    // No draw() has run, so the layout cache is cold and the fallback
    // branch is exercised.
    let est_rows = DEFAULT_BLOCK_ROWS + 1;
    t.scroll_to_block(4);
    assert_eq!(t.scroll(), 4 * est_rows);

    // Out-of-range index is a no-op (scroll unchanged).
    let before = t.scroll();
    t.scroll_to_block(999);
    assert_eq!(t.scroll(), before);
}

/// Regression: `draw` clamped scroll with `content_total` INCLUDING the
/// tail pad, but `clamp_scroll_to_content` / `is_at_bottom` EXCLUDED it —
/// the first user scroll jumped 2 rows and auto-follow couldn't be left.
/// All three now share `content_total()`.
#[test]
fn scroll_clamp_agrees_with_draw_no_tail_pad_jump() {
    let theme = Theme::no_color();
    let mut t = Transcript::new();
    for i in 0..40 {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("line {i}"),
            done: true,
        });
    }
    let width = 40u16;
    let height = 8u16;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");

    // Auto-follow tail: draw clamps the sentinel to its max offset.
    t.scroll_to_bottom();
    terminal
        .draw(|f| {
            t.draw(
                f,
                Rect::new(0, 0, width, height),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw");
    let after_draw = t.scroll();

    // Preparing a user scroll must land on the SAME offset (no 2-row jump).
    t.clamp_scroll_to_content(height, &theme, width, ImageProtocol::None);
    assert_eq!(
        t.scroll(),
        after_draw,
        "clamp_scroll_to_content must agree with draw's content_total"
    );

    // At the clamped bottom we report at-bottom; one row up we must not.
    assert!(t.is_at_bottom(height, &theme, width, ImageProtocol::None));
    t.scroll_up(1);
    assert!(
        !t.is_at_bottom(height, &theme, width, ImageProtocol::None),
        "one row above bottom must not report at-bottom"
    );
}

/// Regression: scroll bookkeeping (`clamp_scroll_to_content`) receives the
/// FULL transcript region width, but `draw` lays out at `width - 1` when a
/// scrollbar gutter is reserved. Passing the full width through forced a
/// complete O(n) re-layout per scroll event AND — because every per-block
/// render cache is width-keyed — invalidated every cached markdown/syntect
/// render, twice per wheel tick (once here, once when the next draw flipped
/// the width back). On long transcripts that was the scroll lag. The scroll
/// path must reuse the layout width the last draw established.
#[test]
fn scroll_clamp_reuses_draw_layout_width_no_relayout_thrash() {
    let theme = Theme::no_color();
    let mut t = Transcript::new();
    for i in 0..60 {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("paragraph {i} with enough text to be a real block"),
            done: true,
        });
    }
    let width = 40u16;
    let height = 8u16;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|f| {
            t.draw(
                f,
                Rect::new(0, 0, width, height),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw");
    // Content overflows → draw laid out at the gutter-narrowed width.
    let layout_width_after_draw = t.cached_layout_width;
    assert_eq!(
        layout_width_after_draw,
        width - 1,
        "precondition: overflowing content reserves the scrollbar gutter"
    );

    // A wheel-scroll preamble passes the FULL region width; the layout the
    // draw built must survive untouched (same width → no O(n) re-layout, no
    // width-keyed render-cache invalidation).
    t.clamp_scroll_to_content(height, &theme, width, ImageProtocol::None);
    assert_eq!(
        t.cached_layout_width, layout_width_after_draw,
        "scroll bookkeeping must reuse the draw's layout width"
    );

    // And `is_at_bottom` must answer from that same cache (not the stale
    // "assume bottom" fallback) even though the caller passes full width.
    t.scroll_to_bottom();
    t.clamp_scroll_to_content(height, &theme, width, ImageProtocol::None);
    assert!(t.is_at_bottom(height, &theme, width, ImageProtocol::None));
    t.scroll_up(1);
    assert!(
        !t.is_at_bottom(height, &theme, width, ImageProtocol::None),
        "is_at_bottom must read the gutter-width cache, not fall back"
    );
}

#[test]
fn scrollbar_thumb_reaches_bottom_when_text_does() {
    let theme = Theme::no_color();
    let mut t = Transcript::new();
    for i in 0..40 {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("line {i}"),
            done: true,
        });
    }
    let (w, h) = (40u16, 10u16);
    let right_col = |t: &mut Transcript| -> Vec<String> {
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| t.draw(f, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None))
            .expect("draw");
        let buf = term.backend().buffer();
        (0..h)
            .map(|y| buf[(w - 1, y)].symbol().to_string())
            .collect()
    };

    // Scrolled to the top: the thumb sits at the first track row (just below
    // the top arrow at row 0). This theme is `no_color()`, so the gutter
    // degrades to ASCII siblings — the thumb is `#`, not `█` (R10).
    t.scroll_to_top();
    let top = right_col(&mut t);
    assert_eq!(top[1], "#", "thumb is at the top when scrolled to the top");

    // Scrolled to the bottom: the thumb must reach the last track row (just
    // above the bottom arrow at row h-1) — the bug was it stalling mid-track
    // while the text was already at the end.
    t.scroll_to_bottom();
    let bottom = right_col(&mut t);
    assert_eq!(
        bottom[(h - 2) as usize],
        "#",
        "thumb reaches the bottom when the text does"
    );
    assert_ne!(
        top, bottom,
        "the thumb actually moved between top and bottom"
    );
}

/// The scrollbar gutter paints the prototype's rich glyphs (`▲ █ ░ ▼`,
/// components.md §7) under a color theme and degrades every one to its
/// 1-cell ASCII sibling (`^ # . v`) under `NO_COLOR` — zero rich glyphs
/// survive the mono render (R10).
#[test]
fn scrollbar_glyphs_match_prototype_and_degrade_under_no_color() {
    let (w, h) = (40u16, 10u16);
    let gutter = |theme: &Theme| -> Vec<String> {
        let mut t = Transcript::new();
        for i in 0..40 {
            t.push(RenderBlock::TextDelta {
                id: id(),
                text: format!("line {i}"),
                done: true,
            });
        }
        t.scroll_to_top();
        let backend = TestBackend::new(w, h);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| t.draw(f, Rect::new(0, 0, w, h), theme, 0, ImageProtocol::None))
            .expect("draw");
        let buf = term.backend().buffer();
        (0..h)
            .map(|y| buf[(w - 1, y)].symbol().to_string())
            .collect()
    };

    // Color theme: rich glyphs — ▲ top arrow, ▼ bottom arrow, █ thumb, ░ track.
    let rich = gutter(&Theme::default_dark());
    assert_eq!(rich[0], "▲", "top arrow is ▲ under color: {rich:?}");
    assert_eq!(
        rich[(h - 1) as usize],
        "▼",
        "bottom arrow is ▼ under color: {rich:?}"
    );
    assert!(
        rich.iter().any(|g| g == "█"),
        "thumb is █ under color: {rich:?}"
    );
    assert!(
        rich.iter().any(|g| g == "░"),
        "track is ░ under color: {rich:?}"
    );

    // NO_COLOR: every glyph is its ASCII sibling, and nothing rich survives.
    let plain = gutter(&Theme::no_color());
    assert_eq!(plain[0], "^", "top arrow degrades to ^: {plain:?}");
    assert_eq!(
        plain[(h - 1) as usize],
        "v",
        "bottom arrow degrades to v: {plain:?}"
    );
    assert!(
        plain.iter().any(|g| g == "#"),
        "thumb degrades to #: {plain:?}"
    );
    assert!(
        plain.iter().any(|g| g == "."),
        "track degrades to .: {plain:?}"
    );
    for g in &plain {
        assert!(
            g.is_ascii(),
            "NO_COLOR scrollbar glyph must be ASCII: {g:?}"
        );
    }
}

#[test]
fn scrollbar_drag_maps_row_to_scroll_offset() {
    let theme = Theme::no_color();
    let mut t = Transcript::new();
    for i in 0..40 {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("line {i}"),
            done: true,
        });
    }
    let width = 40u16;
    let height = 8u16;
    // The max offset = the tail: snap to bottom, then clamp the sentinel.
    t.scroll_to_bottom();
    t.clamp_scroll_to_content(height, &theme, width, ImageProtocol::None);
    let max_scroll = t.scroll();
    assert!(
        max_scroll > 0,
        "content overflows so there is room to scroll"
    );

    // Drag to the very top of the track → offset 0.
    t.scroll_to_viewport_row(0, height, &theme, width, ImageProtocol::None);
    assert_eq!(t.scroll(), 0, "top of scrollbar maps to the top");

    // Drag to the bottom of the track → the max offset (same as the tail).
    t.scroll_to_viewport_row(height - 1, height, &theme, width, ImageProtocol::None);
    assert_eq!(
        t.scroll(),
        max_scroll,
        "bottom of scrollbar maps to the end"
    );

    // A middle row lands strictly between the two ends (monotonic mapping).
    t.scroll_to_viewport_row(height / 2, height, &theme, width, ImageProtocol::None);
    let mid = t.scroll();
    assert!(
        mid > 0 && mid < max_scroll,
        "mid drag lands in the interior"
    );
}

/// [B3] 회귀 가드: TextDelta 의 높이 측정 폭이 실제 draw 폭과 일치해야 한다.
///
/// 버그: 렌더 폭과 측정 폭이 다르면 임계 길이의 줄이 1행 과소/과대측정되어
/// 블록 끝줄 클립/스크롤 최하단 어긋남이 생긴다.
#[test]
fn text_delta_height_matches_draw_width() {
    let theme = Theme::no_color();
    let width = 40u16;
    let draw_width = width;

    // 공백 없는 41-cell 토큰: draw width(40)에선 2행으로 wrap 된다.
    let token: String = std::iter::repeat_n('x', 41).collect();

    let mut t = Transcript::new();
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: token.clone(),
        done: true,
    });

    // 측정 높이 (layout 경로).
    let measured = t.cached_block_height(0, width, &theme, ImageProtocol::None);

    // Ground truth: 실제 draw 폭으로 렌더 + wrap 한 본문 행수. Assistant
    // TextDelta 는 low-chrome 모드에서 별도 헤더 행을 예약하지 않는다.
    // 높이 경로와 동일하게
    // blocks::text 의 done 렌더(끝 빈 줄 trim 포함)를 ground truth 로 쓴다 —
    // markdown 직접 호출은 문단 trailing blank 가 남아 height 와 어긋난다.
    let lines =
        crate::tui::blocks::text::rendered_lines_for_width(&token, true, &theme, 0, draw_width);
    let body_rows = super::super::blocks::wrapped_rows(&lines, draw_width);
    let header_row = 0u16;
    let expected = body_rows + header_row;

    assert_eq!(
        measured, expected,
        "TextDelta 높이는 draw 폭({draw_width})으로 측정해야 한다: \
             measured={measured}, expected={expected}"
    );

    assert!(
        body_rows > 1,
        "테스트 토큰이 wrap 되어야 함: draw={body_rows}"
    );
}

#[test]
fn streaming_single_trailing_newline_layout_height_is_cache_stable() {
    let theme = Theme::default_dark();
    let width = 80u16;
    let mut t = Transcript::new();
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: "단일 줄\n".to_string(),
        done: false,
    });

    let first_height = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    let cached_height = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    assert_eq!(
        first_height, cached_height,
        "cache hit must not change one-line streaming block height"
    );
    match &t.rendered_cache[0] {
        Some(super::RenderCache::Text {
            lines, row_prefix, ..
        }) => {
            assert_eq!(lines.len(), 1, "no phantom blank body line in cache");
            assert_eq!(row_prefix.last().copied(), Some(1), "one body row");
        }
        other => panic!("expected text render cache, got {other:?}"),
    }

    t.invalidate_render_cache();
    let refreshed_height = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    assert_eq!(
        first_height, refreshed_height,
        "cache miss/rebuild must keep one-line streaming block height stable"
    );
    match &t.rendered_cache[0] {
        Some(super::RenderCache::Text {
            lines, row_prefix, ..
        }) => {
            assert_eq!(lines.len(), 1, "rebuilt cache must not add a blank line");
            assert_eq!(
                row_prefix.last().copied(),
                Some(1),
                "rebuilt body row count"
            );
        }
        other => panic!("expected rebuilt text render cache, got {other:?}"),
    }
}

#[test]
fn streaming_text_cache_key_tracks_tail_mutation_version() {
    let theme = Theme::no_color();
    let width = 80u16;
    let mut t = Transcript::new();
    let block_id = id();
    t.push(RenderBlock::TextDelta {
        id: block_id,
        text: "alpha".to_string(),
        done: false,
    });

    let _ = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    match &t.rendered_cache[0] {
        Some(super::RenderCache::Text {
            content_version, ..
        }) => assert_eq!(*content_version, 0),
        other => panic!("expected initial text cache, got {other:?}"),
    }

    t.push(RenderBlock::TextDelta {
        id: block_id,
        text: " beta".to_string(),
        done: false,
    });
    assert_eq!(t.render_version(0), 1);
    assert!(
        matches!(t.rendered_cache[0], Some(super::RenderCache::Text { .. })),
        "tail merge must keep the previous cache as the incremental seed"
    );

    let _ = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    match &t.rendered_cache[0] {
        Some(super::RenderCache::Text {
            content_version, ..
        }) => assert_eq!(*content_version, 1),
        other => panic!("expected refreshed text cache, got {other:?}"),
    }
}

#[test]
fn large_done_transition_keeps_incremental_cache_to_avoid_input_stall() {
    use std::fmt::Write as _;
    let theme = Theme::no_color();
    let width = 80u16;
    let mut table = String::from("| col | value |\n| --- | --- |\n");
    for i in 0..7000 {
        let _ = writeln!(table, "| row {i} | generated value {i} |");
    }
    assert!(
        table.len() > 96 * 1024,
        "test table must cross the large-final-render threshold"
    );
    assert!(
        crate::tui::blocks::text::preserves_layout_pub(&table),
        "a full done render would switch table markdown into layout-preserving output"
    );

    let mut t = Transcript::new();
    let block_id = id();
    t.push(RenderBlock::TextDelta {
        id: block_id,
        text: table,
        done: false,
    });
    let streaming_height = t.cached_block_height(0, width, &theme, ImageProtocol::None);

    t.push(RenderBlock::TextDelta {
        id: block_id,
        text: String::new(),
        done: true,
    });
    let done_height = t.cached_block_height(0, width, &theme, ImageProtocol::None);

    match &t.rendered_cache[0] {
        Some(super::RenderCache::Text {
            done, preserves, ..
        }) => {
            assert!(*done, "cache must key the block as completed");
            assert!(
                !*preserves,
                "large done transition should keep the incremental cache instead of synchronously \
                 rebuilding the full layout-preserving table render"
            );
        }
        other => panic!("expected text render cache, got {other:?}"),
    }
    assert_eq!(
        streaming_height, done_height,
        "empty done sentinel should not force a new expensive layout for huge blocks"
    );
}

/// turn 경계 gap 은 separator 가 직전/다음 블록을 덮지 않는 최소 2행을
/// 유지한다. Zo/You label 자체가 본문 spacer 를 갖기 때문에 폭이 넓어도
/// 여기서 추가 blank 를 더하지 않는다.
#[test]
fn turn_boundary_gap_uses_minimum_separator_space() {
    let theme = Theme::default_dark();
    let narrow = turn_boundary_gap(&theme, 40); // ≤ narrow_max(59)
    let compact = turn_boundary_gap(&theme, 80); // 60..=99
    let wide = turn_boundary_gap(&theme, 120); // ≥ wide_min(100)
    assert_eq!(narrow, 3, "narrow separator gap: {narrow}");
    assert_eq!(compact, 3, "compact separator gap: {compact}");
    assert_eq!(wide, 3, "wide separator gap: {wide}");
}

#[test]
fn prose_tool_boundary_gap_keeps_workflow_compact() {
    let theme = Theme::default_dark();
    assert_eq!(prose_tool_boundary_gap(&theme, 40), 1);
    assert_eq!(prose_tool_boundary_gap(&theme, 80), 1);
    assert_eq!(prose_tool_boundary_gap(&theme, 120), 1);
}

/// Memory-leak guard: streaming many finished reasoning blocks must not grow the
/// timing side maps without bound. They are keyed by `BlockId` and inserted into
/// on every `push`; `gc_timing_maps` prunes entries whose blocks are gone, so the
/// maps stay bounded by the live block count rather than the lifetime push count.
#[test]
fn timing_maps_stay_bounded_under_long_streaming() {
    let mut t = Transcript::new();
    // Each finished reasoning block gets a fresh id and a frozen-elapsed entry.
    for _ in 0..5_000 {
        t.push(RenderBlock::Reasoning {
            id: id(),
            text: "thinking".to_string(),
            signature: None,
            done: true,
        });
    }
    let tracked =
        t.tool_call_started_at.len() + t.reasoning_started_at.len() + t.reasoning_elapsed.len();
    assert!(
        tracked <= t.blocks.len() * 2 + 64,
        "timing maps must stay bounded by live blocks, got {tracked} for {} blocks",
        t.blocks.len()
    );
}

// ===========================================================================
// Manual perf probes — excluded from normal runs (#[ignore]). Run with:
//   cargo test -p zo-cli --lib --release -- --ignored perf_probe --nocapture
// Numbers are printed, not asserted: wall-clock thresholds would be flaky
// across machines; the probes exist to quantify frame cost before/after
// rendering changes.
// ===========================================================================

/// Regression: `draw()` used to do a full O(n) reverse scan of all blocks every
/// frame to find the most recent `Pending | Running` ToolCall. After a turn ends
/// (all tools settled to `Ok`/`Errored`) the scan found nothing but still touched
/// every block — exactly the "lag grows with content" symptom on mouse-wheel
/// scroll. The fix caches the result and only rescans at tool-event boundaries.
///
/// This test builds a transcript with 2000 completed tool-pair blocks (no
/// active tools), draws one frame to warm the cache, then performs 50 scroll
/// up/down + draw cycles and asserts that `tail_active_dirty` remains `false`
/// throughout — proving the O(1) path is taken, not the O(n) rescan.
#[test]
fn scroll_on_settled_transcript_does_not_rescan_tail_active_idx() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();

    // Build 2000 completed tool pairs (no Pending/Running).
    for n in 0..500usize {
        t.push(bash_call(n, ToolCallStatus::Ok));
        t.push(bash_result(n));
    }
    assert_eq!(t.blocks.len(), 1000);

    let (w, h) = (80u16, 30u16);
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("backend");
    let area = Rect::new(0, 0, w, h);

    // Warm-up draw: builds the layout cache and refreshes the tail-active
    // cache once (all settled → cached_tail_active_idx = None, dirty = false).
    terminal
        .draw(|f| t.draw(f, area, &theme, 0, ImageProtocol::None))
        .expect("warm-up draw");

    // After the first draw, `tail_active_dirty` must be false (cache is warm).
    assert!(
        !t.tail_active_dirty,
        "tail_active_dirty must be false after the first draw on a fully-settled transcript"
    );
    assert_eq!(
        t.cached_tail_active_idx, None,
        "no active tool calls → cached_tail_active_idx must be None"
    );

    // Simulate 50 scroll+draw cycles (like rapid mouse-wheel scroll).
    for tick in 0..50u64 {
        t.scroll_up(3);
        terminal
            .draw(|f| t.draw(f, area, &theme, tick + 1, ImageProtocol::None))
            .expect("scroll-up draw");

        // The draw must NOT have set tail_active_dirty (no ToolCall mutations).
        assert!(
            !t.tail_active_dirty,
            "tail_active_dirty must stay false during scroll-up (tick {tick})"
        );

        t.scroll_down(3);
        terminal
            .draw(|f| t.draw(f, area, &theme, tick + 51, ImageProtocol::None))
            .expect("scroll-down draw");

        assert!(
            !t.tail_active_dirty,
            "tail_active_dirty must stay false during scroll-down (tick {tick})"
        );

        // cached_tail_active_idx must remain None: no active tool calls were
        // pushed, so the cache never needs updating.
        assert_eq!(
            t.cached_tail_active_idx, None,
            "cached_tail_active_idx must stay None during scroll (tick {tick})"
        );
    }
}

/// Counterpart to the above: a new Pending tool call must mark dirty, and the
/// first subsequent draw must rescan and cache the new active index.
#[test]
fn pending_tool_call_marks_tail_active_dirty_and_draw_refreshes_cache() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();

    // Start with settled blocks.
    for n in 0..50usize {
        t.push(bash_call(n, ToolCallStatus::Ok));
        t.push(bash_result(n));
    }

    let (w, h) = (80u16, 20u16);
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("backend");
    let area = Rect::new(0, 0, w, h);

    terminal
        .draw(|f| t.draw(f, area, &theme, 0, ImageProtocol::None))
        .expect("warm-up draw");
    assert!(!t.tail_active_dirty, "settled → not dirty after draw");
    assert_eq!(t.cached_tail_active_idx, None, "settled → no active call");

    // Push a new Pending tool call — must mark dirty.
    t.push(bash_call(999, ToolCallStatus::Pending));
    assert!(
        t.tail_active_dirty,
        "pushing a Pending ToolCall must mark tail_active_dirty"
    );

    // The next draw must refresh the cache and clear dirty.
    terminal
        .draw(|f| t.draw(f, area, &theme, 1, ImageProtocol::None))
        .expect("post-pending draw");

    assert!(
        !t.tail_active_dirty,
        "draw must clear tail_active_dirty after refresh"
    );
    let expected_idx = t.blocks.len() - 1;
    assert_eq!(
        t.cached_tail_active_idx,
        Some(expected_idx),
        "cached_tail_active_idx must point to the new Pending call at idx {expected_idx}"
    );

    // After the tool settles, reconcile_tool_call_status marks dirty again.
    t.push(bash_result(999));
    assert!(
        t.tail_active_dirty,
        "reconcile_tool_call_status (ToolResult push) must mark tail_active_dirty"
    );

    terminal
        .draw(|f| t.draw(f, area, &theme, 2, ImageProtocol::None))
        .expect("post-settle draw");

    assert!(!t.tail_active_dirty, "draw must clear dirty after settle");
    assert_eq!(
        t.cached_tail_active_idx, None,
        "no active calls after settle → None"
    );
}

/// Realistic mixed markdown (Korean+English prose, lists, fenced code) of at
/// least `target_bytes`, mimicking a long streamed assistant answer.
fn synthetic_markdown(target_bytes: usize) -> String {
    let section = concat!(
        "## 섹션 제목과 설명\n\n",
        "이 문단은 스트리밍 응답의 전형적인 prose 텍스트를 모사한다. ",
        "`inline code`, **bold**, 그리고 [링크](https://example.com)를 포함하며 ",
        "한국어와 English mixed width 계산이 필요한 CJK 문자가 들어있다.\n\n",
        "- 첫 번째 항목: 설명 텍스트가 이어진다\n",
        "- 두 번째 항목: `code` 와 **강조**\n",
        "  - 중첩 항목 하나\n\n",
        "```rust\n",
        "fn example(input: &str) -> Result<usize, Error> {\n",
        "    let parsed = input.trim().parse::<usize>()?;\n",
        "    Ok(parsed.saturating_mul(2))\n",
        "}\n",
        "```\n\n",
    );
    let mut out = String::with_capacity(target_bytes + section.len());
    while out.len() < target_bytes {
        out.push_str(section);
    }
    out
}

/// Draw `frames` frames and print the mean per-frame cost. One warm-up frame
/// builds layout/render caches so the steady state is what gets measured.
fn probe_frames(t: &mut Transcript, theme: &Theme, frames: u32, label: &str) {
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("terminal");
    let area = Rect::new(0, 0, 120, 40);
    terminal
        .draw(|f| t.draw(f, area, theme, 0, ImageProtocol::None))
        .expect("warm-up draw");
    let start = std::time::Instant::now();
    for tick in 0..frames {
        terminal
            .draw(|f| t.draw(f, area, theme, u64::from(tick), ImageProtocol::None))
            .expect("draw");
    }
    let total = start.elapsed();
    eprintln!(
        "[perf_probe] {label}: {:.3} ms/frame ({frames} frames, total {total:?})",
        total.as_secs_f64() * 1000.0 / f64::from(frames),
    );
}

#[test]
#[ignore = "manual perf probe — release-only, prints timings"]
fn perf_probe_large_block_scroll() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: id(),
        text: "아키텍처를 자세히 설명해줘".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: id(),
        text: synthetic_markdown(160 * 1024),
        done: true,
    });
    t.push(RenderBlock::System {
        id: id(),
        level: SystemLevel::Info,
        text: "tail marker".to_string(),
    });

    t.scroll_to_top();
    probe_frames(&mut t, &theme, 60, "large-block scroll=top");

    t.scroll_to_bottom();
    probe_frames(&mut t, &theme, 60, "large-block scroll=bottom");

    // Mid-block: clamp to bottom first (draw sets the real offset), then back
    // up to roughly half the content so the viewport sits inside the block.
    let bottom = t.scroll();
    t.scroll_up(bottom / 2);
    probe_frames(&mut t, &theme, 60, "large-block scroll=mid");
}

#[test]
#[ignore = "manual perf probe — release-only, prints timings"]
fn perf_probe_streaming_tail() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    let block = id();
    let md = synthetic_markdown(120 * 1024);
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("terminal");
    let area = Rect::new(0, 0, 120, 40);

    let chunk = 1024usize;
    let mut pushed = 0usize;
    let mut frames_in_quarter = 0u32;
    let mut quarter_start = std::time::Instant::now();
    let mut next_quarter = md.len() / 4;
    let mut quarter_no = 1u8;
    let mut tick = 0u64;
    while pushed < md.len() {
        let mut end = (pushed + chunk).min(md.len());
        while !md.is_char_boundary(end) {
            end += 1;
        }
        t.push(RenderBlock::TextDelta {
            id: block,
            text: md[pushed..end].to_string(),
            done: false,
        });
        pushed = end;
        terminal
            .draw(|f| t.draw(f, area, &theme, tick, ImageProtocol::None))
            .expect("draw");
        tick += 1;
        frames_in_quarter += 1;
        if pushed >= next_quarter {
            let elapsed = quarter_start.elapsed();
            eprintln!(
                "[perf_probe] streaming q{quarter_no} (~{} KiB cum): {:.3} ms/frame ({} frames)",
                pushed / 1024,
                elapsed.as_secs_f64() * 1000.0 / f64::from(frames_in_quarter.max(1)),
                frames_in_quarter,
            );
            quarter_no += 1;
            next_quarter = (usize::from(quarter_no) * md.len()) / 4;
            frames_in_quarter = 0;
            quarter_start = std::time::Instant::now();
        }
    }
    t.push(RenderBlock::TextDelta {
        id: block,
        text: String::new(),
        done: true,
    });
    let done_start = std::time::Instant::now();
    terminal
        .draw(|f| t.draw(f, area, &theme, tick, ImageProtocol::None))
        .expect("draw");
    eprintln!(
        "[perf_probe] streaming done-frame (authoritative pass): {:?}",
        done_start.elapsed()
    );
}

#[test]
#[ignore = "manual perf probe — release-only, prints timings"]
fn perf_probe_many_tool_blocks() {
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    for n in 0..400u64 {
        let call = id();
        t.push(RenderBlock::ToolCall {
            id: call,
            tool_call_id: ToolCallId(format!("tc-{n}")),
            name: "Bash".to_string(),
            summary: format!("cargo test --workspace --case-{n}"),
            preview: ToolPreview::Bash {
                command: format!("cargo test --workspace --case-{n}"),
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: id(),
            tool_call_id: ToolCallId(format!("tc-{n}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: format!("test result: ok. {n} passed; 0 failed\nfinished in 0.{n}s"),
                truncated: false,
            },
        });
    }
    t.scroll_to_bottom();
    probe_frames(&mut t, &theme, 60, "400-tool-pairs scroll=bottom");
    t.scroll_to_top();
    probe_frames(&mut t, &theme, 60, "400-tool-pairs scroll=top");
}

#[test]
fn copy_text_for_block_range_joins_spanned_blocks_in_display_order() {
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: BlockId(1),
        text: "first".to_string(),
    });
    t.push(RenderBlock::UserMessage {
        id: BlockId(2),
        text: "second".to_string(),
    });
    t.push(RenderBlock::UserMessage {
        id: BlockId(3),
        text: "third".to_string(),
    });

    // A range copies the spanned blocks' clean text, blank-line joined, with the
    // out-of-range block excluded.
    assert_eq!(
        t.copy_text_for_block_range(BlockId(1), BlockId(2))
            .as_deref(),
        Some("first\n\nsecond"),
        "the third block is outside the range and excluded",
    );
    // Order-agnostic: an upward drag (head before anchor) still yields the text
    // in display order.
    assert_eq!(
        t.copy_text_for_block_range(BlockId(3), BlockId(1))
            .as_deref(),
        Some("first\n\nsecond\n\nthird"),
    );
    // A single-block range copies just that block.
    assert_eq!(
        t.copy_text_for_block_range(BlockId(2), BlockId(2))
            .as_deref(),
        Some("second"),
    );
}

#[test]
fn copy_diff_tool_result_block_serializes_to_unified_diff() {
    use runtime::message_stream::{DiffHunk, DiffLine, DiffLineKind, DiffView};

    let mut t = Transcript::new();
    t.push(RenderBlock::ToolResult {
        id: BlockId(1),
        tool_call_id: ToolCallId("call_1".to_string()),
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("src/old.rs".to_string()),
            new_path: Some("src/new.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 10,
                old_lines: 2,
                new_start: 10,
                new_lines: 3,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Context,
                        text: "fn main() {".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "    old_code();".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "    new_code();".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "    more_new_code();".to_string(),
                    },
                ],
            }],
        }),
    });

    let copied = t.copy_text_for_block_id(BlockId(1));
    assert!(copied.is_some());
    let copied_str = copied.unwrap();

    let expected = "\
--- a/src/old.rs
+++ b/src/new.rs
@@ -10,2 +10,3 @@
 fn main() {
-    old_code();
+    new_code();
+    more_new_code();
";
    assert_eq!(copied_str, expected);
}

/// Jank regression: a prose answer that follows a reasoning block must keep the
/// SAME author-mark style whether the reasoning is still streaming (`done=false`,
/// the live `Thinking…` line) or has just settled (`done=true`, collapsed). The
/// old `is_suppressed_reasoning`-only skip flipped the style Bullet↔Indent
/// on the `done` transition, changing the block height and jumping the layout
/// mid-answer. `assistant_prose_style` now treats every Reasoning block as
/// transparent to prose authorship, so the mark is stable across the flip.
#[test]
fn prose_mark_is_stable_across_reasoning_done_transition() {
    use crate::tui::blocks::ProseMark;

    // First prose of a turn, preceded only by a reasoning block: must carry
    // the author bullet in BOTH reasoning states (no indent flip).
    for done in [false, true] {
        let blocks = vec![
            RenderBlock::Reasoning {
                id: BlockId(1),
                text: "weighing options".to_string(),
                signature: None,
                done,
            },
            RenderBlock::TextDelta {
                id: BlockId(2),
                text: "the answer".to_string(),
                done: false,
            },
        ];
        assert_eq!(
            super::assistant_prose_style(&blocks, 1),
            ProseMark::Bullet,
            "prose after a reasoning block stays a Bullet regardless of done={done}"
        );
    }

    // Prose → reasoning → prose: the second prose continues the first
    // (Indent) and must NOT flip when the intervening reasoning settles.
    for done in [false, true] {
        let blocks = vec![
            RenderBlock::TextDelta {
                id: BlockId(1),
                text: "first".to_string(),
                done: true,
            },
            RenderBlock::Reasoning {
                id: BlockId(2),
                text: "mid-answer thought".to_string(),
                signature: None,
                done,
            },
            RenderBlock::TextDelta {
                id: BlockId(3),
                text: "second".to_string(),
                done: false,
            },
        ];
        assert_eq!(
            super::assistant_prose_style(&blocks, 2),
            ProseMark::Indent,
            "prose stays an Indent across the reasoning done={done} transition"
        );
    }
}


#[test]
fn workflow_selective_state_text_remains_visible_in_transcript() {
    let mut transcript = Transcript::new();
    transcript.push(RenderBlock::TextDelta {
        id: id(),
        text: "workflow verify: running · 5 selective · 1 findings · 1 blocked".to_string(),
        done: true,
    });
    let backend = TestBackend::new(96, 8);
    let mut terminal = Terminal::new(backend).expect("backend");
    let theme = Theme::default_dark();
    terminal
        .draw(|frame| transcript.draw(frame, frame.area(), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let dumped = dump_terminal(&terminal, 96, 8);
    assert!(dumped.contains("selective"), "selective state should render: {dumped}");
    assert!(dumped.contains("findings"), "finding state should render: {dumped}");
    assert!(dumped.contains("blocked"), "blocked state should render: {dumped}");
}

fn text_block(id: u64) -> RenderBlock {
    RenderBlock::TextDelta {
        id: BlockId(id),
        text: format!("block {id}"),
        done: true,
    }
}

#[test]
fn char_selection_rows_match_terminal_selection_shape() {
    // Columns are screen cells clamped to the clip; rows are content rows
    // against `scroll` (= the first visible content row).
    let clip = Rect::new(10, 20, 5, 4);
    let scroll = 100;

    assert_eq!(
        char_selection_rows((11, 101), (13, 101), clip, scroll),
        vec![(101, 11, 13)]
    );
    assert_eq!(
        char_selection_rows((13, 101), (11, 101), clip, scroll),
        vec![(101, 11, 13)],
        "single-row selection is column-order agnostic"
    );

    let downward = vec![(100, 12, 14), (101, 10, 14), (102, 10, 13)];
    assert_eq!(
        char_selection_rows((12, 100), (13, 102), clip, scroll),
        downward
    );
    assert_eq!(
        char_selection_rows((13, 102), (12, 100), clip, scroll),
        downward,
        "upward/leftward drag yields the same spans"
    );

    assert_eq!(
        char_selection_rows((0, 90), (u16::MAX, u16::MAX), clip, scroll),
        vec![
            (100, 10, 14),
            (101, 10, 14),
            (102, 10, 14),
            (103, 10, 14),
        ],
        "columns clamp to the clip; rows clip to the visible band"
    );
    assert!(char_selection_rows((1, 1), (2, 2), Rect::new(0, 0, 0, 4), 0).is_empty());
    assert!(char_selection_rows((1, 1), (2, 2), Rect::new(0, 0, 4, 0), 0).is_empty());
}

#[test]
fn char_selection_rows_keep_shape_for_offscreen_endpoints() {
    // A selection taller than the screen: rows outside the visible band drop,
    // and an interior row at the band edge spans the full width even though
    // both endpoints are offscreen — the first/middle/last shape is decided
    // over the full selection before clipping.
    let clip = Rect::new(0, 0, 10, 3); // visible band = content rows 50..=52
    let scroll = 50;
    assert_eq!(
        char_selection_rows((4, 40), (6, 60), clip, scroll),
        vec![(50, 0, 9), (51, 0, 9), (52, 0, 9)]
    );
    // The band's top row is the selection's *last* row: it takes the end
    // column, not the full width.
    assert_eq!(
        char_selection_rows((4, 40), (6, 50), clip, scroll),
        vec![(50, 0, 6)]
    );
    // Entirely above / below the band → nothing to wash.
    assert!(char_selection_rows((4, 10), (6, 20), clip, scroll).is_empty());
    assert!(char_selection_rows((4, 90), (6, 95), clip, scroll).is_empty());
}

#[test]
fn wheel_extended_char_selection_copies_past_the_viewport() {
    // The regression the content-row anchor exists for: drag inside a short
    // viewport, wheel-scroll further content into view mid-drag, release —
    // the copy must include the rows that scrolled *out* of the viewport
    // (their last mined text persists in the content-row store).
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    for word in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"] {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("row {word}"),
            done: true,
        });
    }
    let area = Rect::new(0, 0, 24, 3);
    let backend = TestBackend::new(24, 3);
    let mut terminal = Terminal::new(backend).expect("backend");
    let mut draw = |t: &mut Transcript| {
        terminal
            .draw(|f| t.draw(f, area, &theme, 0, ImageProtocol::None))
            .expect("draw");
    };

    t.scroll_to_top();
    draw(&mut t);
    assert_eq!(t.scroll(), 0);

    // Press on the very first cell, drag to the viewport's last row.
    t.begin_char_selection(0, 0);
    t.extend_char_selection(23, 2);
    draw(&mut t);

    // Wheel down (2 rows per notch), re-extending to the pointer's cell at
    // the viewport bottom after every notch — the app wheel handler's exact
    // sequence — and repaint so the newly revealed rows are mined. Blocks
    // render as content + one gap row, so 4 notches sweep all six blocks.
    for _ in 0..4 {
        t.scroll_down(2);
        t.extend_char_selection(23, 2);
        draw(&mut t);
    }

    let copied = t.finish_char_selection().expect("selection should copy");
    assert!(
        copied.contains("row alpha"),
        "rows scrolled out of the viewport must stay in the copy: {copied}"
    );
    let tail_words = ["bravo", "charlie", "delta", "echo", "foxtrot"];
    for word in tail_words {
        assert!(
            copied.contains(&format!("row {word}")),
            "every row swept by the drag belongs to the copy ({word}): {copied}"
        );
    }
    assert!(
        t.has_char_selection(),
        "a successful copy keeps its highlight until the next press"
    );

    // A follow-up scroll must NOT drop the settled highlight anymore — the
    // content-row anchor tracks the text (the old screen-pinned selection
    // cleared here).
    t.scroll_up(2);
    draw(&mut t);
    assert!(
        t.has_char_selection(),
        "scrolling after release keeps the content-anchored highlight"
    );
}

#[test]
fn char_selection_drops_when_rows_above_it_shift() {
    // Content-row anchors survive appends below, but an in-place mutation
    // above the selection shifts its rows — the layout hook must drop the
    // gesture instead of washing (and copying) shifted text.
    let theme = Theme::default_dark();
    let mut t = Transcript::new();
    t.set_turn_active(true);
    let reasoning_id = id();
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "thinking".to_string(),
        signature: None,
        done: false,
    });
    for word in ["alpha", "bravo", "charlie"] {
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!("row {word}"),
            done: true,
        });
    }
    let area = Rect::new(0, 0, 24, 6);
    let backend = TestBackend::new(24, 6);
    let mut terminal = Terminal::new(backend).expect("backend");
    let mut draw = |t: &mut Transcript| {
        terminal
            .draw(|f| t.draw(f, area, &theme, 0, ImageProtocol::None))
            .expect("draw");
    };
    t.scroll_to_top();
    draw(&mut t);

    // Select the prose rows below the streaming reasoning block.
    t.begin_char_selection(0, 2);
    t.extend_char_selection(23, 4);
    draw(&mut t);
    assert!(t.has_char_selection());

    // Mid-list mutation above the selection: the reasoning block grows
    // (multi-line → taller), shoving every row below it.
    t.push(RenderBlock::Reasoning {
        id: reasoning_id,
        text: "\nmore\nlines\nof\nthought".to_string(),
        signature: None,
        done: false,
    });
    draw(&mut t);
    assert!(
        !t.has_char_selection(),
        "a height change above the selection must drop it"
    );
}

#[test]
fn buffer_row_text_copies_ascii_and_skips_cjk_continuation_cells() {
    let mut buffer = Buffer::empty(Rect::new(0, 0, 12, 1));
    buffer.set_string(0, 0, "abcdef", Style::default());
    assert_eq!(buffer_row_text(&buffer, 0, 1, 3), "bcd");

    buffer.reset();
    buffer.set_string(0, 0, "안녕 hi", Style::default());
    assert_eq!(buffer_row_text(&buffer, 0, 0, 6), "안녕 hi");
}

#[test]
fn join_selection_lines_trims_row_ends_and_keeps_interior_blanks() {
    let lines = vec![
        "alpha   ".to_string(),
        "   ".to_string(),
        "omega\t".to_string(),
    ];
    assert_eq!(join_selection_lines(&lines), "alpha\n\nomega");

    let blank = vec!["   ".to_string(), "\t".to_string()];
    assert_eq!(join_selection_lines(&blank), "");
}

/// Isolated draw-cost measurement for a long, many-turn transcript — the
/// "context fills up → frames get heavy" scenario. Renders the same visible
/// window many times (idle repaint cadence) and reports per-frame ms so a
/// hotpath regression/optimization is attributable. Run manually:
///   cargo test -p zo-cli --lib tui::transcript::tests::perf_draw_long_many_turn_transcript -- --ignored --nocapture
#[test]
#[ignore = "perf measurement, run manually with --ignored --nocapture"]
fn perf_draw_long_many_turn_transcript() {
    use std::time::Instant;
    let theme = Theme::default_dark();
    let mut t = Transcript::new();

    // 200 turns, each: a user message + an assistant prose block + a tool
    // call/result pair. ~800 blocks with many UserMessage turn boundaries — the
    // exact shape that makes `turn_ordinal`'s prefix scan add up per frame.
    let turns = 200u64;
    for turn in 0..turns {
        t.push(RenderBlock::UserMessage {
            id: id(),
            text: format!("turn {turn}: 이 기능을 구현해줘 with some english words too"),
        });
        t.push(RenderBlock::TextDelta {
            id: id(),
            text: format!(
                "turn {turn} 답변: 의존성을 파악하고 계층형으로 정리했습니다. \
                 Here is a longer paragraph so the block spans multiple wrapped rows \
                 and the layout cache has real height to measure across the window."
            ),
            done: true,
        });
        t.push(generic_tool_call(&format!("t{turn}")));
        t.push(todo_block(&format!("plan {turn}")));
    }
    let width = 100u16;
    let height = 40u16;
    eprintln!("[perf] blocks = {}", t.blocks().len());

    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");

    // Warm the layout cache once (cold full Case-4 build).
    let warm = Instant::now();
    terminal
        .draw(|frame| {
            t.draw(frame, Rect::new(0, 0, width, height), &theme, 0, ImageProtocol::None);
        })
        .expect("draw");
    eprintln!("[perf] cold first draw (Case-4 build): {:?}", warm.elapsed());

    // Idle repaint loop: scroll to the top (so the visible window sits over
    // high block indices' turn boundaries) and redraw repeatedly. The layout
    // cache is warm and unchanged, so this isolates the *per-frame* draw cost
    // (the visible loop + turn_ordinal/boundary scans), not layout rebuilds.
    t.scroll_to_top();
    let frames = 600u32;
    let mut max_ms = 0.0f64;
    let loop_start = Instant::now();
    for f in 0..frames {
        // Nudge scroll a little each frame so we sweep different turn-boundary
        // separators into view (mimics wheel scrolling through history).
        if f % 2 == 0 {
            t.scroll_down(1);
        } else {
            t.scroll_up(1);
        }
        let s = Instant::now();
        terminal
            .draw(|frame| {
                t.draw(frame, Rect::new(0, 0, width, height), &theme, u64::from(f), ImageProtocol::None);
            })
            .expect("draw");
        let ms = s.elapsed().as_secs_f64() * 1000.0;
        max_ms = max_ms.max(ms);
    }
    let total = loop_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[perf] {frames} idle repaints: total={total:.1}ms avg={:.3}ms max={max_ms:.3}ms",
        total / f64::from(frames)
    );
}

/// A running spawn batch stays compact at narrow width. Per-agent route/tail
/// detail lives in the pinned/Ctrl+G surfaces, and routine transcript rows have
/// no card background wash.
#[test]
fn narrow_agent_batch_stays_compact_and_background_free() {
    let theme = Theme::default_dark();
    let width = 60u16;
    let height = 16u16;

    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: id(),
        tool_call_id: ToolCallId("call_fan".to_string()),
        name: "Task".to_string(),
        summary: "spawn".to_string(),
        preview: ToolPreview::Generic {
            name: "Task".to_string(),
            input_summary: "spawn".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    let tree = AgentTree {
        rows: vec![
            AgentTreeRow {
                agent_id: "agent-0".to_string(),
                name: "correctness-audit".to_string(),
                model: "gpt-5.5".to_string(),
                status: "running".to_string(),
                subagent_type: Some("Reviewer".to_string()),
                tool_calls: Some(18),
                elapsed_secs: 114,
                activity: Some("bash".to_string()),
                output_tail: Some(
                    "scanning crates/runtime for a very long streamed tail line here".to_string(),
                ),
                route_reason: Some(
                    "Reviewer·Medium — auto role selector with a long overflowing tail".to_string(),
                ),
                ..AgentTreeRow::default()
            },
            AgentTreeRow {
                agent_id: "agent-1".to_string(),
                name: "impl-width".to_string(),
                model: "gpt-5.5".to_string(),
                status: "running".to_string(),
                subagent_type: Some("Reviewer".to_string()),
                tool_calls: Some(9),
                elapsed_secs: 60,
                activity: Some("edit".to_string()),
                route_reason: Some(
                    "Coder·High — dynamic band ceiling with another long overflowing reason"
                        .to_string(),
                ),
                ..AgentTreeRow::default()
            },
        ],
        batch_label: None,
        finished: false,
    };
    t.set_agent_tree("call_fan", tree);

    let backend = TestBackend::new(width, height);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| t.draw(f, Rect::new(0, 0, width, height), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let buf = term.backend().buffer().clone();

    let block_height = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    assert_eq!(block_height, 2, "header + one aggregate progress row");

    let row_text = |y: u16| -> String { (0..width).map(|x| buf[(x, y)].symbol()).collect() };

    // The child row is indented and cannot wrap an orphan fragment to column 0.
    for y in 1..block_height {
        assert_eq!(
            buf[(0, y)].symbol(),
            " ",
            "row {y} ({:?}) must start with the child indent",
            row_text(y)
        );
    }
    let rendered = (0..block_height)
        .map(row_text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("Running 2 Reviewer agents"), "{rendered}");
    assert!(rendered.contains("2 running · 27 tool uses"), "{rendered}");
    assert!(
        !rendered.contains("correctness-audit")
            && !rendered.contains("routed:")
            && !rendered.contains("scanning crates"),
        "per-agent detail must not leak into the compact transcript: {rendered}"
    );

    // The old card tint may still be derivable by the theme, but no transcript
    // cell receives it.
    let card_bg = theme
        .tool_card_bg(theme.palette.accent)
        .expect("dark theme derives the old card tint sentinel");
    assert!(
        (0..height).all(|y| (0..width).all(|x| buf[(x, y)].bg != card_bg)),
        "routine transcript rows must remain background-free"
    );
}

/// Responsive prose uses 90% of the available 200-cell body width, and layout
/// measurement must paint exactly the same rows at that 180-cell wrap width.
#[test]
fn ultrawide_prose_paints_every_measured_row() {
    let theme = Theme::no_color();
    let available_width = 200u16;
    let width = available_width + crate::tui::blocks::ROLE_RAIL_WIDTH;

    let mut t = Transcript::new();
    let text = "x".repeat(360);
    t.push(RenderBlock::TextDelta {
        id: id(),
        text,
        done: true,
    });

    let height = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    assert_eq!(height, 2, "360 cells must measure as two 180-cell rows");
    match &t.rendered_cache[0] {
        Some(RenderCache::Text { width, .. }) => assert_eq!(*width, 180),
        other => panic!("expected responsive prose render cache, got {other:?}"),
    }

    let view_h = height.saturating_add(4);
    let backend = TestBackend::new(width, view_h);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| t.draw(f, Rect::new(0, 0, width, view_h), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let buf = term.backend().buffer().clone();

    let row_text = |y: u16| -> String { (0..width).map(|x| buf[(x, y)].symbol()).collect() };
    let dump: Vec<String> = (0..view_h).map(row_text).collect();
    let painted_rows = dump.iter().filter(|row| row.contains('x')).count();
    assert_eq!(
        painted_rows,
        usize::from(height),
        "painted rows must equal measured rows with no phantom tail:\n{dump:#?}"
    );
    let cap_right = crate::tui::blocks::ROLE_RAIL_WIDTH + 180;
    for y in 0..height {
        let beyond: String = (cap_right..width).map(|x| buf[(x, y)].symbol()).collect();
        assert!(
            beyond.trim().is_empty(),
            "row {y} painted beyond the responsive cap ({cap_right}+): {:?}",
            dump[usize::from(y)]
        );
        assert_eq!(
            dump[usize::from(y)].matches('x').count(),
            180,
            "row {y} must fill the 180-cell measured body width:\n{dump:#?}"
        );
    }
}

/// User prose shares the same responsive measure/paint width and keeps its rail
/// on every wrapped body row.
#[test]
fn ultrawide_user_message_paints_every_measured_row() {
    let theme = Theme::no_color();
    let available_width = 200u16;
    let width = available_width + crate::tui::blocks::ROLE_RAIL_WIDTH;

    let mut t = Transcript::new();
    let text = "x".repeat(360);
    t.push(RenderBlock::UserMessage { id: id(), text });

    let height = t.cached_block_height(0, width, &theme, ImageProtocol::None);
    assert_eq!(height, 3, "header plus two 180-cell body rows");
    match &t.rendered_cache[0] {
        Some(RenderCache::Text { width, .. }) => assert_eq!(*width, 180),
        other => panic!("expected responsive user-message cache, got {other:?}"),
    }

    let view_h = height.saturating_add(4);
    let backend = TestBackend::new(width, view_h);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| t.draw(f, Rect::new(0, 0, width, view_h), &theme, 0, ImageProtocol::None))
        .expect("draw");
    let buf = term.backend().buffer().clone();

    let row_text = |y: u16| -> String { (0..width).map(|x| buf[(x, y)].symbol()).collect() };
    let dump: Vec<String> = (0..view_h).map(row_text).collect();
    let painted_body_rows = dump.iter().filter(|row| row.contains('x')).count();
    assert_eq!(painted_body_rows, usize::from(height - 1), "{dump:#?}");
    for y in 1..height {
        assert_eq!(
            buf[(0, y)].symbol(),
            "|",
            "body row {y} must keep the user rail: {:?}",
            dump[usize::from(y)]
        );
        assert_eq!(dump[usize::from(y)].matches('x').count(), 180);
    }
}

/// 클릭-투-익스팬드 (CC parity): 접힌 툴 그룹 요약 리더를 클릭하면 그룹이
/// 펼쳐져 개별 행이 보이고, 다시 클릭하면 재접힌다.
#[test]
fn click_toggles_collapsed_tool_group_open_and_closed() {
    let theme = Theme::no_color();
    let width = 80u16;
    let mk_call = |bid: u64, cid: &str, name: &str| RenderBlock::ToolCall {
        id: BlockId(bid),
        tool_call_id: ToolCallId(cid.to_string()),
        name: name.to_string(),
        summary: name.to_string(),
        preview: ToolPreview::Generic {
            name: name.to_string(),
            input_summary: String::new(),
        },
        status: ToolCallStatus::Ok,
    };
    let mk_result = |bid: u64, cid: &str| RenderBlock::ToolResult {
        id: BlockId(bid),
        tool_call_id: ToolCallId(cid.to_string()),
        is_error: false,
        body: ToolResultBody::Bash(BashResult {
            exit_code: 0,
            stdout: "ok".to_string(),
            stderr: String::new(),
            truncated: false,
        }),
    };

    let mut t = Transcript::new();
    t.push(mk_call(1, "g1", "Read"));
    t.push(mk_call(2, "g2", "Grep"));
    t.push(mk_result(3, "g1"));
    t.push(mk_result(4, "g2"));

    let draw = |t: &mut Transcript| {
        let backend = TestBackend::new(width, 24);
        let mut term = Terminal::new(backend).expect("backend");
        term.draw(|f| t.draw(f, Rect::new(0, 0, width, 24), &theme, 0, ImageProtocol::None))
            .expect("draw");
    };
    draw(&mut t);
    assert!(
        matches!(t.tool_groups[0], ToolGroupState::Summary { .. }),
        "precondition: 이 배치는 요약으로 접힌다, got {:?}",
        t.tool_groups
    );

    // 리더 클릭 → 그룹 펼침.
    assert!(t.toggle_expand_for_click(BlockId(1)), "leader click consumed");
    draw(&mut t);
    assert!(
        t.tool_groups
            .iter()
            .all(|s| matches!(s, ToolGroupState::Normal)),
        "펼친 그룹의 모든 행이 Normal 로 렌더된다, got {:?}",
        t.tool_groups
    );

    // 리더 재클릭 → 재접힘.
    assert!(t.toggle_expand_for_click(BlockId(1)), "re-click consumed");
    draw(&mut t);
    assert!(
        matches!(t.tool_groups[0], ToolGroupState::Summary { .. }),
        "재클릭으로 그룹이 다시 접힌다, got {:?}",
        t.tool_groups
    );
}

/// 클릭-투-익스팬드: 일반 툴 행 클릭은 그 결과 본문의 expand 를 토글하고,
/// 결과 블록 클릭도 같은 토글이다. 산문 블록 클릭은 소비하지 않는다
/// (클릭-복사 기본 동작 유지).
#[test]
fn click_on_tool_call_toggles_matching_result_expansion() {
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call_1".to_string()),
        name: "Bash".to_string(),
        summary: "cargo test".to_string(),
        preview: ToolPreview::Bash {
            command: "cargo test".to_string(),
        },
        status: ToolCallStatus::Ok,
    });
    t.push(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call_1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "many\nlines\nof\noutput".to_string(),
            truncated: true,
        },
    });
    t.push(RenderBlock::TextDelta {
        id: BlockId(3),
        text: "산문은 복사 대상".to_string(),
        done: true,
    });

    // 호출 행 클릭 → 짝 결과 expand.
    assert!(t.toggle_expand_for_click(BlockId(1)));
    assert!(t.is_expanded(1), "결과 블록(idx 1)이 펼쳐진다");
    // 결과 행 클릭 → 다시 접힘.
    assert!(t.toggle_expand_for_click(BlockId(2)));
    assert!(!t.is_expanded(1), "결과 블록이 다시 접힌다");
    // 산문 클릭은 expand 라우트가 소비하지 않는다 → 클릭-복사 폴백.
    assert!(!t.toggle_expand_for_click(BlockId(3)));
}

#[test]
fn click_on_running_bash_call_toggles_live_tail_expansion() {
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: BlockId(11),
        tool_call_id: ToolCallId("call_live_bash".to_string()),
        name: "bash".to_string(),
        summary: "gh run watch 123".to_string(),
        preview: ToolPreview::Bash {
            command: "gh run watch 123".to_string(),
        },
        status: ToolCallStatus::Running,
    });

    assert!(t.toggle_expand_for_click(BlockId(11)));
    assert!(t.is_expanded(0), "first click opens the live tail");
    assert!(t.toggle_expand_for_click(BlockId(11)));
    assert!(!t.is_expanded(0), "second click closes the live tail");
    assert!(t.toggle_expand_for_click(BlockId(11)));
    t.push(RenderBlock::ToolResult {
        id: BlockId(12),
        tool_call_id: ToolCallId("call_live_bash".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "done".to_string(),
            truncated: false,
        },
    });
    assert!(!t.is_expanded(0), "completion drops the running-row state");
}
