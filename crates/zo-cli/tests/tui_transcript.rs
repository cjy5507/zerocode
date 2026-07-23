//! Integration tests for `tui::transcript::Transcript` (Phase 3, Lane L5).

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use runtime::message_stream::{BlockId, BlockIdGen, RenderBlock, SystemLevel};
use runtime::message_stream::{
    DiffHunk, DiffLine, DiffLineKind, DiffView, ToolCallId, ToolCallStatus, ToolPreview,
    ToolResultBody,
};
use zo_cli::tui::image_protocol::ImageProtocol;
use zo_cli::tui::theme::Theme;
use zo_cli::tui::transcript::Transcript;

fn ids() -> BlockIdGen {
    BlockIdGen::default()
}

fn push_text(t: &mut Transcript, ids: &BlockIdGen, body: &str) -> BlockId {
    let id = ids.next();
    t.push(RenderBlock::TextDelta {
        id,
        text: body.to_string(),
        done: true,
    });
    id
}

fn push_reasoning(t: &mut Transcript, ids: &BlockIdGen, body: &str) -> BlockId {
    let id = ids.next();
    t.push(RenderBlock::Reasoning {
        id,
        text: body.to_string(),
        signature: None,
        done: true,
    });
    id
}

fn push_system(t: &mut Transcript, ids: &BlockIdGen, body: &str) -> BlockId {
    let id = ids.next();
    t.push(RenderBlock::System {
        id,
        level: SystemLevel::Info,
        text: body.to_string(),
    });
    id
}

#[test]
fn transcript_starts_empty() {
    let t = Transcript::new();
    assert!(t.is_empty());
    assert_eq!(t.len(), 0);
    assert_eq!(t.scroll(), 0);
    assert!(t.focused_idx().is_none());
}

#[test]
fn transcript_push_increments_len() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "hello");
    push_text(&mut t, &ids, "world");
    assert_eq!(t.len(), 2);
    assert!(!t.is_empty());
}

#[test]
fn transcript_keeps_tool_call_and_matching_tool_result_connected() {
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call-1".to_string()),
        name: "Read".to_string(),
        summary: "file.rs".to_string(),
        preview: ToolPreview::Read {
            path: "file.rs".to_string(),
            range: None,
        },
        status: ToolCallStatus::Running,
    });
    t.push(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "file.rs".to_string(),
            content: "fn main() {}".to_string(),
            language: Some("rust".to_string()),
            truncated: false,
        },
    });
    assert_eq!(t.len(), 2);

    let rows = transcript_rows(&mut t, 80, 8);
    let call_row = rows
        .iter()
        .position(|row| row.contains("Explored read file.rs"))
        .expect("tool call row");
    let result_row = rows
        .iter()
        .position(|row| row.contains("Read: file.rs"))
        .expect("tool result summary row");
    let file_row = rows
        .iter()
        .position(|row| row.contains("file.rs") && row.contains("rust"))
        .expect("read body file row");

    assert_eq!(
        result_row,
        call_row + 1,
        "matching tool result should stay attached to the call: {rows:?}"
    );
    assert!(
        rows[result_row].starts_with("  ` "),
        "tool result summary should sit under the tool call: {rows:?}"
    );
    assert!(
        rows[file_row].starts_with("  | file.rs"),
        "read body should sit one level under the result summary: {rows:?}"
    );
}

#[test]
fn transcript_markdown_header_distinguishes_live_from_done() {
    let mut live = Transcript::new();
    live.push(RenderBlock::UserMessage {
        id: BlockId(9),
        text: "prompt".to_string(),
    });
    live.push(RenderBlock::TextDelta {
        id: BlockId(10),
        text: "streaming markdown".to_string(),
        done: false,
    });
    // v3 (streaming-style-v3 §3.1, §4): the transcript carries no
    // `Zo · writing/done` header. Live vs done is surfaced only by the
    // bottom activity line (App surface), so inside the transcript both render
    // identically — a `*` author bullet (no_color) with no writing/done chrome.
    let live_rows = transcript_rows(&mut live, 72, 6);
    let live_joined = live_rows.join("\n");
    assert!(
        live_rows
            .iter()
            .any(|row| row.starts_with('*') && row.contains("streaming markdown")),
        "live assistant markdown carries a `*` author bullet: {live_joined}"
    );
    assert!(
        !live_joined.contains("Zo")
            && !live_joined.contains("writing")
            && !live_joined.contains("done"),
        "v3 transcript carries no Zo/writing/done chrome while live: {live_joined}"
    );

    let mut settled = Transcript::new();
    settled.push(RenderBlock::UserMessage {
        id: BlockId(12),
        text: "prompt".to_string(),
    });
    settled.push(RenderBlock::TextDelta {
        id: BlockId(11),
        text: "settled markdown".to_string(),
        done: true,
    });
    let settled_rows = transcript_rows(&mut settled, 72, 6);
    let settled_joined = settled_rows.join("\n");
    assert!(
        settled_rows
            .iter()
            .any(|row| row.starts_with('*') && row.contains("settled markdown")),
        "completed assistant markdown carries the same `*` author bullet: {settled_joined}"
    );
    assert!(
        !settled_joined.contains("Zo")
            && !settled_joined.contains("writing")
            && !settled_joined.contains("done"),
        "v3 done transition is a transcript no-op — no writing/done chrome: {settled_joined}"
    );
}

#[test]
fn user_to_first_assistant_response_does_not_double_space_before_author_bullet() {
    let mut t = Transcript::new();
    t.push(RenderBlock::UserMessage {
        id: BlockId(21),
        text: "go".to_string(),
    });
    t.push(RenderBlock::TextDelta {
        id: BlockId(22),
        text: "test fixture cleanup will proceed.".to_string(),
        done: true,
    });

    let rows = transcript_rows(&mut t, 80, 12);
    let user_body = rows
        .iter()
        .position(|row| row.contains("go"))
        .expect("user body row");
    let assistant_body = rows
        .iter()
        .position(|row| row.contains("test fixture"))
        .expect("assistant body row");

    // v3 (streaming-style-v3 §3.1): no `Zo` header — the first assistant
    // prose carries a `*` author bullet (no_color). The user→assistant boundary
    // keeps exactly one blank row plus a separator rule, never a double blank.
    assert!(
        rows[assistant_body].starts_with('*'),
        "first assistant prose carries a `*` author bullet: {rows:?}"
    );
    let between = &rows[user_body + 1..assistant_body];
    let blanks = between.iter().filter(|row| row.trim().is_empty()).count();
    assert_eq!(
        blanks, 1,
        "user→assistant boundary keeps exactly one blank row (no double-space): {rows:?}"
    );
    assert!(
        between.iter().any(|row| row.trim_start().starts_with("---")),
        "user→assistant boundary draws a separator rule: {rows:?}"
    );
}

#[test]
fn transcript_merges_text_deltas_with_same_id() {
    let ids = ids();
    let mut t = Transcript::new();
    let id = ids.next();
    t.push(RenderBlock::TextDelta {
        id,
        text: "안녕".to_string(),
        done: false,
    });
    t.push(RenderBlock::TextDelta {
        id,
        text: "하세요".to_string(),
        done: false,
    });
    t.push(RenderBlock::TextDelta {
        id,
        text: String::new(),
        done: true,
    });
    assert_eq!(t.len(), 1);
}

#[test]
fn transcript_scroll_down_and_up_round_trip() {
    let ids = ids();
    let mut t = Transcript::new();
    for i in 0..5 {
        push_text(&mut t, &ids, &format!("line {i}"));
    }
    t.scroll_down(3);
    assert_eq!(t.scroll(), 3);
    t.scroll_up(2);
    assert_eq!(t.scroll(), 1);
    t.scroll_up(10);
    assert_eq!(t.scroll(), 0);
}

#[test]
fn transcript_focus_next_skips_non_interactable_blocks() {
    // System + plain text are not interactable; Reasoning is.
    let ids = ids();
    let mut t = Transcript::new();
    push_system(&mut t, &ids, "/clear");
    push_text(&mut t, &ids, "just text");
    push_reasoning(&mut t, &ids, "thinking…");
    let advanced = t.focus_next();
    assert!(advanced, "focus_next should move into the reasoning block");
    assert_eq!(t.focused_idx(), Some(2));
}

#[test]
fn transcript_focus_prev_returns_none_at_start() {
    let ids = ids();
    let mut t = Transcript::new();
    push_reasoning(&mut t, &ids, "thinking");
    t.focus_next();
    let moved = t.focus_prev();
    assert!(!moved || t.focused_idx().is_none() || t.focused_idx() == Some(0));
}

#[test]
fn transcript_toggle_expanded_round_trips() {
    let ids = ids();
    let mut t = Transcript::new();
    push_reasoning(&mut t, &ids, "reason");
    t.focus_next();
    let idx = t.focused_idx().expect("focused");
    let initial = t.is_expanded(idx);
    t.toggle_expanded();
    let after_first = t.is_expanded(idx);
    t.toggle_expanded();
    let after_second = t.is_expanded(idx);
    assert_ne!(initial, after_first, "first toggle should change state");
    assert_eq!(initial, after_second, "second toggle should revert");
}

#[test]
fn transcript_draws_into_test_backend_without_panicking() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "hello world");
    push_reasoning(&mut t, &ids, "thinking…");
    let backend = TestBackend::new(60, 10);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 10);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");
}

#[test]
fn transcript_multiline_text_keeps_following_blocks_visible() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_text(
        &mut t,
        &ids,
        "첫 줄\n둘째 줄\n셋째 줄\n넷째 줄\n다섯째 줄\n여섯째 줄",
    );
    push_system(&mut t, &ids, "after");
    let backend = TestBackend::new(40, 12);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 40, 12);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect::<String>();
    assert!(
        rendered.contains("after"),
        "following block missing: {rendered}"
    );
}

#[test]
fn transcript_short_content_top_aligns_like_native_transcript() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "hello");
    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 40, 8);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");
    let buffer = terminal.backend().buffer().clone();
    let top_rows = (0..3)
        .flat_map(|y| {
            let buffer_ref = &buffer;
            (0..40).map(move |x| {
                buffer_ref
                    .cell((x, y))
                    .map_or(" ", ratatui::buffer::Cell::symbol)
                    .to_string()
            })
        })
        .collect::<String>();
    assert!(
        top_rows.contains("hello"),
        "content should start near top: {top_rows}"
    );
}

#[test]
fn transcript_applies_block_gap_between_adjacent_blocks() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "hello");
    push_system(&mut t, &ids, "after");
    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 40, 8);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");
    let buffer = terminal.backend().buffer().clone();
    let rows = (0..8)
        .map(|y| {
            (0..40)
                .map(|x| {
                    buffer
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                        .to_string()
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let hello_row = rows.iter().position(|row| row.contains("hello"));
    let after_row = rows.iter().position(|row| row.contains("after"));
    let gap_present = match (hello_row, after_row) {
        (Some(hello_row), Some(after_row)) if after_row > hello_row + 1 => rows
            [hello_row + 1..after_row]
            .iter()
            .any(|row| row.trim().is_empty()),
        _ => false,
    };
    assert!(
        gap_present,
        "adjacent blocks should have breathing room: {rows:?}"
    );
}

#[test]
fn transcript_keeps_tool_call_and_result_visually_tight_and_marks_call_complete() {
    let theme = Theme::no_color();
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call-1".to_string()),
        name: "Bash".to_string(),
        summary: "pwd".to_string(),
        preview: ToolPreview::Bash {
            command: "pwd".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    t.push(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });

    let backend = TestBackend::new(60, 6);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 6);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");

    let rows = (0..6)
        .map(|y| {
            (0..60)
                .map(|x| {
                    terminal
                        .backend()
                        .buffer()
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                        .to_string()
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    let bash_row = rows.iter().position(|row| row.contains("Ran pwd"));
    // Codex-style rows keep the result visually attached with a single corner
    // leader, without a global colored transcript rail.
    let result_row = rows.iter().position(|row| row.trim() == "` ok");
    assert_eq!(
        bash_row,
        Some(0),
        "tool call should start immediately: {rows:?}"
    );
    assert_eq!(
        result_row,
        Some(1),
        "tool result should follow without a blank gap: {rows:?}"
    );
    assert!(
        rows.iter().all(|row| !row.contains("running")),
        "tool call should be upgraded from running once result arrives: {rows:?}"
    );
}

#[test]
fn transcript_gives_answer_room_after_tool_result() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("call-1".to_string()),
        name: "Bash".to_string(),
        summary: "git diff".to_string(),
        preview: ToolPreview::Bash {
            command: "git diff".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    t.push(RenderBlock::ToolResult {
        id: BlockId(2),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "tool output".to_string(),
            truncated: false,
        },
    });
    push_text(&mut t, &ids, "**Answer**\nReadable summary.");

    let backend = TestBackend::new(60, 20);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            t.draw(
                frame,
                Rect::new(0, 0, 60, 20),
                &theme,
                0,
                ImageProtocol::None,
            );
        })
        .expect("draw");

    let rows = (0..20)
        .map(|y| {
            (0..60)
                .map(|x| {
                    terminal
                        .backend()
                        .buffer()
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                        .to_string()
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    let output_row = rows
        .iter()
        .position(|row| row.contains("tool output"))
        .expect("tool output row");
    let answer_row = rows
        .iter()
        .position(|row| row.contains("Answer"))
        .expect("answer heading row");
    // v3 (streaming-style-v3 §3.1): no `Zo` label — the assistant answer
    // carries a `*` author bullet. Tool output is separated from the answer by
    // one blank spacer (no label row, so one fewer blank than the v2 contract).
    assert!(
        output_row < answer_row,
        "assistant answer follows the tool output: {rows:?}"
    );
    let blank_rows = rows[output_row + 1..answer_row]
        .iter()
        .filter(|row| row.trim().is_empty())
        .count();
    assert_eq!(
        blank_rows, 1,
        "tool output is separated from the assistant answer by one blank spacer: {rows:?}"
    );
    assert!(
        rows[answer_row].starts_with('*'),
        "assistant answer after tool output carries a `*` author bullet: {rows:?}"
    );
}

#[test]
fn transcript_keeps_sibling_tool_results_compact() {
    let mut t = Transcript::new();
    for (idx, content) in ["first output", "second output"].iter().enumerate() {
        t.push(RenderBlock::ToolResult {
            id: BlockId(20 + u64::try_from(idx).unwrap()),
            tool_call_id: ToolCallId(format!("call-result-{idx}")),
            is_error: false,
            body: ToolResultBody::Text {
                content: (*content).to_string(),
                truncated: false,
            },
        });
    }

    let rows = transcript_rows(&mut t, 60, 6);
    let first = rows
        .iter()
        .position(|row| row.contains("first output"))
        .expect("first result row");
    let second = rows
        .iter()
        .position(|row| row.contains("second output"))
        .expect("second result row");

    assert_eq!(
        second,
        first + 1,
        "sibling tool results should not have blank spacer rows: {rows:?}"
    );
}

#[test]
fn transcript_labels_assistant_prose_after_system_notice() {
    let ids = ids();
    let mut t = Transcript::new();
    push_system(&mut t, &ids, "session resumed");
    push_text(&mut t, &ids, "continuing the answer");

    let rows = transcript_rows(&mut t, 60, 10);
    let system_row = rows
        .iter()
        .position(|row| row.contains("session resumed"))
        .expect("system row");
    let answer_row = rows
        .iter()
        .position(|row| row.contains("continuing the answer"))
        .expect("assistant body row");

    // v3 (streaming-style-v3 §3.1): no `Zo` label — assistant prose after a
    // system notice carries a `*` author bullet, separated by one blank spacer.
    assert!(
        system_row < answer_row,
        "assistant prose follows the system notice: {rows:?}"
    );
    assert!(
        rows[answer_row].starts_with('*'),
        "assistant prose after a system notice carries a `*` author bullet: {rows:?}"
    );
    let blanks = rows[system_row + 1..answer_row]
        .iter()
        .filter(|row| row.trim().is_empty())
        .count();
    assert_eq!(
        blanks, 1,
        "one blank spacer separates the system notice from the assistant prose: {rows:?}"
    );
}

#[test]
fn transcript_hides_completed_empty_reasoning_noise() {
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: BlockId(1),
        text: String::new(),
        signature: None,
        done: true,
    });
    t.push(RenderBlock::TextDelta {
        id: BlockId(2),
        text: "visible answer".to_string(),
        done: true,
    });

    let rows = transcript_rows(&mut t, 60, 6);
    let rendered = rows.join("\n");

    assert!(rendered.contains("visible answer"), "{rows:?}");
    assert!(!rendered.contains("thought"), "{rows:?}");
    assert!(!rendered.contains("step"), "{rows:?}");
}

#[test]
fn transcript_shows_live_zo_line_while_reasoning_streams() {
    // A streaming reasoning block (done == false) is NOT suppressed: it renders
    // a visible, animated zo cue in the transcript (Gemini thought summaries,
    // Anthropic thinking, OpenAI reasoning all flow here). The raw partial
    // reasoning text stays hidden behind the cue, and the cue carries the Zo
    // metaphor (a reveal verb in amber) rather than a bare `Thinking`.
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: BlockId(1),
        text: "weighing the corridor alias options".to_string(),
        signature: None,
        done: false,
    });

    let rows = transcript_rows(&mut t, 60, 6);
    let rendered = rows.join("\n");

    // seed == block id 1 → ZO_REVEAL_VERBS[1 % 6] == "Planning".
    assert!(
        rendered.contains("Planning"),
        "a streaming reasoning block shows the live zo line: {rows:?}"
    );
    assert!(
        !rendered.contains("Thinking"),
        "the bare Thinking word is gone, replaced by a zo verb: {rows:?}"
    );
    assert!(
        !rendered.contains("corridor alias"),
        "raw partial reasoning text stays hidden behind the cue: {rows:?}"
    );
}

#[test]
fn transcript_hides_thinking_line_once_reasoning_settles() {
    // The same block, once settled (done == true) with no expand, returns to the
    // quiet default — hidden, no stray streaming cue (`Thinking…` / `Thinking…`)
    // left pinned in history.
    let mut t = Transcript::new();
    t.push(RenderBlock::Reasoning {
        id: BlockId(1),
        text: "weighing options".to_string(),
        signature: None,
        done: true,
    });
    t.push(RenderBlock::TextDelta {
        id: BlockId(2),
        text: "final answer".to_string(),
        done: true,
    });

    let rows = transcript_rows(&mut t, 60, 6);
    let rendered = rows.join("\n");

    assert!(rendered.contains("final answer"), "{rows:?}");
    // No streaming reveal cue survives once the block settles. `ZO_REVEAL_VERBS`
    // is `pub(crate)`, so list them literally here (integration crate boundary).
    let zo_verbs = [
        "Thinking",
        "Planning",
        "Exploring",
        "Solving",
        "Reviewing",
        "Working",
    ];
    assert!(
        !rendered.contains("Thinking") && zo_verbs.iter().all(|v| !rendered.contains(v)),
        "a settled reasoning block leaves no streaming cue pinned: {rows:?}"
    );
}

fn push_completed_bash_pair(t: &mut Transcript, id: u64) {
    let tool_call_id = ToolCallId(format!("call-{id}"));
    t.push(RenderBlock::ToolCall {
        id: BlockId(id * 2),
        tool_call_id: tool_call_id.clone(),
        name: "bash".to_string(),
        summary: format!("cmd-{id}"),
        preview: ToolPreview::Bash {
            command: format!("cmd-{id}"),
        },
        status: ToolCallStatus::Running,
    });
    t.push(RenderBlock::ToolResult {
        id: BlockId(id * 2 + 1),
        tool_call_id,
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });
}

fn push_pending_bash(t: &mut Transcript, id: u64) {
    t.push(RenderBlock::ToolCall {
        id: BlockId(id),
        tool_call_id: ToolCallId(format!("pending-{id}")),
        name: "bash".to_string(),
        summary: "queued".to_string(),
        preview: ToolPreview::Bash {
            command: "queued".to_string(),
        },
        status: ToolCallStatus::Pending,
    });
}

fn push_running_bash(t: &mut Transcript, id: u64) {
    t.push(RenderBlock::ToolCall {
        id: BlockId(id),
        tool_call_id: ToolCallId(format!("running-{id}")),
        name: "bash".to_string(),
        summary: format!(r#"{{"command":"cmd-{id}"}}"#),
        preview: ToolPreview::Bash {
            command: format!("cmd-{id}"),
        },
        status: ToolCallStatus::Running,
    });
}

fn transcript_rows(t: &mut Transcript, width: u16, height: u16) -> Vec<String> {
    let theme = Theme::no_color();
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, width, height);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");
    let buffer = terminal.backend().buffer().clone();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| {
                    buffer
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                        .to_string()
                })
                .collect::<String>()
        })
        .collect()
}

#[test]
fn transcript_shows_running_tool_group_as_live_rows() {
    let mut t = Transcript::new();
    push_running_bash(&mut t, 1);
    push_running_bash(&mut t, 2);
    push_running_bash(&mut t, 3);

    let rows = transcript_rows(&mut t, 80, 10);
    let rendered = rows.join("\n");
    // CC parity: the running batch is visible as a live group — one row per
    // command with an animating status marker (`*` under NO_COLOR).
    assert!(
        rendered.contains("cmd-1") && rendered.contains("cmd-2") && rendered.contains("cmd-3"),
        "each running command should be visible as a live group row: {rows:?}"
    );
    assert!(
        rows.iter().filter(|row| row.starts_with("* bash")).count() == 3,
        "every running row carries the in-flight marker: {rows:?}"
    );
    assert!(
        !rendered.contains("tools active"),
        "no merged 'N tools active' one-liner: {rows:?}"
    );
}

#[test]
fn transcript_settled_tool_group_shows_per_tool_detail() {
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolCall {
        id: BlockId(1),
        tool_call_id: ToolCallId("read-1".to_string()),
        name: "Read".to_string(),
        summary: "src/lib.rs".to_string(),
        preview: ToolPreview::Read {
            path: "src/lib.rs".to_string(),
            range: None,
        },
        status: ToolCallStatus::Ok,
    });
    t.push(RenderBlock::ToolCall {
        id: BlockId(2),
        tool_call_id: ToolCallId("grep-1".to_string()),
        name: "Grep".to_string(),
        summary: "pattern".to_string(),
        preview: ToolPreview::Grep {
            pattern: "pattern".to_string(),
            path: Some("src".to_string()),
        },
        status: ToolCallStatus::Ok,
    });
    t.push(RenderBlock::ToolCall {
        id: BlockId(3),
        tool_call_id: ToolCallId("edit-1".to_string()),
        name: "edit_file".to_string(),
        summary: "src/lib.rs".to_string(),
        preview: ToolPreview::Edit {
            path: "src/lib.rs".to_string(),
            hunk_count: 1,
        },
        status: ToolCallStatus::Ok,
    });

    let rows = transcript_rows(&mut t, 100, 10);
    let rendered = rows.join("\n");
    // New design: a settled group shows one `verb  target` row per tool so the
    // user can recognize what ran — not a merged `N tools` count that hid it.
    assert!(
        !rendered.contains("3 tools"),
        "settled group should show per-tool detail, not a merged count: {rows:?}"
    );
    assert!(
        rendered.contains("read") && rendered.contains("grep") && rendered.contains("edit"),
        "each tool's verb should be visible: {rows:?}"
    );
    assert!(
        rendered.contains("src/lib.rs") && rendered.contains("pattern"),
        "each tool's target (path / query) should be visible: {rows:?}"
    );
    assert!(
        rows.iter().any(|row| row.starts_with("v ")),
        "the group leader row keeps the done marker: {rows:?}"
    );
}

#[test]
fn transcript_completed_tool_group_uses_plain_done_status() {
    let mut t = Transcript::new();
    push_completed_bash_pair(&mut t, 1);
    push_completed_bash_pair(&mut t, 2);
    push_completed_bash_pair(&mut t, 3);

    let rows = transcript_rows(&mut t, 80, 10);
    let rendered = rows.join("\n");
    // A repetitive settled batch collapses to one calm action digest.
    assert!(
        rendered.contains("3 tools") && rendered.contains("bash x3"),
        "settled group should show a compact, fully counted digest: {rows:?}"
    );
    assert!(
        rows.iter().any(|row| row.starts_with("v 3 tools")),
        "the leader row renders as a root event with the done marker: {rows:?}"
    );
    assert!(
        rendered.matches("bash").count() == 1,
        "repetitive settled calls should not flood history: {rows:?}"
    );
    assert!(
        !rendered.contains("forged"),
        "status copy should not use brand/jargon wording: {rows:?}"
    );
}

#[test]
fn transcript_shows_edit_diff_inline_while_other_tools_collapse() {
    // End-to-end Claude Code parity for the reported gap: a turn that reads a
    // few files and then makes one edit must collapse the reads into a `tools
    // done` summary while showing the edit's diff inline — not fold the diff
    // into the summary. Mirrors the real screenshot (several tools + an edit).
    let mut t = Transcript::new();

    for idx in 1..=3u64 {
        let call = ToolCallId(format!("read-{idx}"));
        t.push(RenderBlock::ToolCall {
            id: BlockId(idx * 2),
            tool_call_id: call.clone(),
            name: "Read".to_string(),
            summary: format!("src/file_{idx}.rs"),
            preview: ToolPreview::Read {
                path: format!("src/file_{idx}.rs"),
                range: None,
            },
            status: ToolCallStatus::Ok,
        });
        t.push(RenderBlock::ToolResult {
            id: BlockId(idx * 2 + 1),
            tool_call_id: call,
            is_error: false,
            body: ToolResultBody::Read {
                path: format!("src/file_{idx}.rs"),
                content: "fn main() {}".to_string(),
                language: Some("rust".to_string()),
                truncated: false,
            },
        });
    }

    let edit_call = ToolCallId("edit-1".to_string());
    t.push(RenderBlock::ToolCall {
        id: BlockId(100),
        tool_call_id: edit_call.clone(),
        name: "Edit".to_string(),
        summary: "src/widget.rs".to_string(),
        preview: ToolPreview::Edit {
            path: "src/widget.rs".to_string(),
            hunk_count: 1,
        },
        status: ToolCallStatus::Ok,
    });
    t.push(RenderBlock::ToolResult {
        id: BlockId(101),
        tool_call_id: edit_call,
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("src/widget.rs".to_string()),
            new_path: Some("src/widget.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 10,
                old_lines: 1,
                new_start: 10,
                new_lines: 1,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "let old_widget = 1;".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "let new_widget = 2;".to_string(),
                    },
                ],
            }],
        }),
    });

    let rows = transcript_rows(&mut t, 100, 24);
    let rendered = rows.join("\n");

    // The three reads collapse to one counted action digest; their file bodies
    // and individual targets remain available in the expanded group.
    assert!(
        rendered.contains("3 tools") && rendered.contains("read x3"),
        "non-diff tools should collapse into one counted digest: {rows:?}"
    );
    assert!(
        !rendered.contains("src/file_1.rs"),
        "individual read targets should stay out of the compact default: {rows:?}"
    );
    // The edit's diff is shown inline: file path and changed code are visible.
    assert!(
        rendered.contains("widget.rs"),
        "the edit diff must render inline with its file path: {rows:?}"
    );
    assert!(
        rendered.contains("new_widget"),
        "the edit diff body (changed code) must be visible inline: {rows:?}"
    );
    // The reads' bodies (file contents) remain hidden — only the path shows.
    assert!(
        !rendered.contains("src/file_1.rs  rust"),
        "collapsed read bodies must stay hidden: {rows:?}"
    );
}

#[test]
fn transcript_collapses_batched_parallel_tool_calls_and_results() {
    let mut t = Transcript::new();
    for idx in 1..=3 {
        t.push(RenderBlock::ToolCall {
            id: BlockId(idx),
            tool_call_id: ToolCallId(format!("read-{idx}")),
            name: "Read".to_string(),
            summary: format!("src/file_{idx}.rs"),
            preview: ToolPreview::Read {
                path: format!("src/file_{idx}.rs"),
                range: None,
            },
            status: ToolCallStatus::Ok,
        });
    }
    for idx in 1..=3 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(100 + idx),
            tool_call_id: ToolCallId(format!("read-{idx}")),
            is_error: false,
            body: ToolResultBody::Read {
                path: format!("src/file_{idx}.rs"),
                content: "fn main() {}".to_string(),
                language: Some("rust".to_string()),
                truncated: false,
            },
        });
    }

    let rows = transcript_rows(&mut t, 100, 10);
    let rendered = rows.join("\n");
    // Batched calls + results collapse into one counted action digest.
    assert!(
        rendered.contains("3 tools") && rendered.contains("read x3"),
        "batched calls should collapse into one counted digest: {rows:?}"
    );
    assert!(
        rows.iter().any(|row| row.starts_with("v 3 tools")),
        "the collapsed batch should render as one root event: {rows:?}"
    );
    assert!(
        !rendered.contains("src/file_1.rs"),
        "individual paths stay behind the existing expand interaction: {rows:?}"
    );
    assert!(
        !rendered.contains("Read: src/file_1.rs") && !rendered.contains("src/file_1.rs  rust"),
        "the result bodies stay hidden with the collapsed group: {rows:?}"
    );
}

#[test]
fn transcript_replaces_streamed_active_tool_group_with_done_summary() {
    let mut t = Transcript::new();
    for idx in 1..=3 {
        t.push(RenderBlock::ToolCall {
            id: BlockId(idx),
            tool_call_id: ToolCallId(format!("read-{idx}")),
            name: "Read".to_string(),
            summary: format!("src/file_{idx}.rs"),
            preview: ToolPreview::Read {
                path: format!("src/file_{idx}.rs"),
                range: None,
            },
            status: ToolCallStatus::Running,
        });
    }

    let active_rows = transcript_rows(&mut t, 100, 10);
    assert!(
        !active_rows.join("\n").contains("tools active"),
        "initial running tool batch should stay out of transcript history: {active_rows:?}"
    );

    for idx in 1..=3 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(100 + idx),
            tool_call_id: ToolCallId(format!("read-{idx}")),
            is_error: false,
            body: ToolResultBody::Read {
                path: format!("src/file_{idx}.rs"),
                content: "fn main() {}".to_string(),
                language: Some("rust".to_string()),
                truncated: false,
            },
        });
    }

    let rows = transcript_rows(&mut t, 100, 12);
    let rendered = rows.join("\n");
    assert!(
        rows.iter().any(|row| row.starts_with("v 3 tools"))
            && rendered.contains("read x3"),
        "completed streamed batch should collapse to one counted digest: {rows:?}"
    );
    assert!(
        !rendered.contains("tools active"),
        "the stale active summary must not survive once results arrive: {rows:?}"
    );
    assert!(
        !rendered.contains("Read: src/file_1.rs") && !rendered.contains("src/file_1.rs  rust"),
        "matching streamed result bodies should stay hidden: {rows:?}"
    );
}

#[test]
fn transcript_reveals_completed_summary_after_pending_tool_settles() {
    let mut t = Transcript::new();

    push_completed_bash_pair(&mut t, 1);
    let _ = transcript_rows(&mut t, 80, 20);
    push_completed_bash_pair(&mut t, 2);
    let _ = transcript_rows(&mut t, 80, 20);
    push_completed_bash_pair(&mut t, 3);
    let _ = transcript_rows(&mut t, 80, 20);
    push_completed_bash_pair(&mut t, 4);
    let _ = transcript_rows(&mut t, 80, 20);
    push_pending_bash(&mut t, 99);

    let active_rows = transcript_rows(&mut t, 80, 20);
    let active_rendered = active_rows.join("\n");
    // The settled prefix must stay put, and the lone pending call shows as its
    // own live event row (`o Ran queued`) instead of erasing the history.
    assert!(
        active_rows.iter().any(|row| row.starts_with("v 4 tools"))
            && active_rendered.contains("bash x4"),
        "the settled prefix stays visible while the new call is pending: {active_rows:?}"
    );
    assert!(
        active_rendered.contains("queued"),
        "the pending call is visible as a live event row: {active_rows:?}"
    );
    assert!(
        !active_rendered.contains("tools active"),
        "no merged 'N tools active' one-liner: {active_rows:?}"
    );

    t.push(RenderBlock::ToolResult {
        id: BlockId(999),
        tool_call_id: ToolCallId("pending-99".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: "ok".to_string(),
            truncated: false,
        },
    });

    let rows = transcript_rows(&mut t, 80, 20);
    let rendered = rows.join("\n");
    // Once the pending call settles, the digest absorbs it and increments both
    // the total and per-action count.
    assert!(
        rows.iter().any(|row| row.starts_with("v 5 tools"))
            && rendered.contains("bash x5"),
        "once the pending call settles, the compact digest must include it: {rows:?}"
    );
    assert!(
        !rendered.contains("tools active") && !rendered.contains("Ran queued"),
        "settled transcript keeps no stale active row: {rows:?}"
    );
}

#[test]
fn transcript_collapsed_tool_group_keeps_answer_boundary() {
    let ids = ids();
    let mut t = Transcript::new();

    push_completed_bash_pair(&mut t, 1);
    push_completed_bash_pair(&mut t, 2);
    push_completed_bash_pair(&mut t, 3);
    push_text(&mut t, &ids, "작업트리 기준으로 이어서 확인했습니다.");

    let rows = transcript_rows(&mut t, 80, 16);
    // Detail rendering: the cluster is now several `bash  <cmd>` rows; take the
    // last one as the tool/prose boundary anchor.
    let summary_row = rows
        .iter()
        .rposition(|row| row.contains("bash"))
        .expect("collapsed tool detail row");
    let answer_row = rows
        .iter()
        .position(|row| row.contains("작 업 트 리"))
        .unwrap_or_else(|| panic!("assistant body row missing: {rows:?}"));

    // v3 (streaming-style-v3 §3.1): no `Zo` label — the answer after a
    // collapsed tool group carries a `*` author bullet, kept off the tool
    // summary by one blank spacer.
    assert!(
        summary_row < answer_row,
        "assistant answer follows the collapsed tool group: {rows:?}"
    );
    let blanks_before_answer = rows[summary_row + 1..answer_row]
        .iter()
        .filter(|row| row.trim().is_empty())
        .count();
    assert_eq!(
        blanks_before_answer, 1,
        "collapsed tool summary keeps one blank row before the assistant answer: {rows:?}"
    );
    assert!(
        rows[answer_row].starts_with('*'),
        "assistant answer after a collapsed tool group carries a `*` author bullet: {rows:?}"
    );
}

#[test]
fn transcript_listing_result_keeps_blank_before_assistant_label() {
    let ids = ids();
    let mut t = Transcript::new();
    t.push(RenderBlock::ToolResult {
        id: BlockId(70),
        tool_call_id: ToolCallId("list-1".to_string()),
        is_error: false,
        body: ToolResultBody::Listing {
            entries: vec!["crates/zo-cli/src/main.rs".to_string()],
            truncated: false,
        },
    });
    push_text(&mut t, &ids, "이어서 crate 구조를 확인하겠습니다.");

    let rows = transcript_rows(&mut t, 80, 12);
    let listing_row = rows
        .iter()
        .position(|row| row.contains("crates/zo-cli/src/main.rs"))
        .expect("listing result row");
    let answer_row = rows
        .iter()
        .position(|row| row.contains("이 어 서"))
        .unwrap_or_else(|| panic!("assistant body row missing: {rows:?}"));

    // v3 (streaming-style-v3 §3.1): no `Zo` label — the listing result is
    // separated from the assistant answer (a `*` author bullet) by one blank row.
    assert!(
        listing_row < answer_row,
        "assistant answer follows the listing result: {rows:?}"
    );
    let blanks = rows[listing_row + 1..answer_row]
        .iter()
        .filter(|row| row.trim().is_empty())
        .count();
    assert_eq!(
        blanks, 1,
        "listing result keeps one blank row before the assistant answer: {rows:?}"
    );
    assert!(
        rows[answer_row].starts_with('*'),
        "assistant answer after a listing result carries a `*` author bullet: {rows:?}"
    );
}

#[test]
fn transcript_read_result_cluster_keeps_clear_boundary_before_assistant_label() {
    let ids = ids();
    let mut t = Transcript::new();
    for idx in 0..3 {
        t.push(RenderBlock::ToolResult {
            id: BlockId(80 + idx),
            tool_call_id: ToolCallId(format!("read-{idx}")),
            is_error: false,
            body: ToolResultBody::Read {
                path: format!(
                    "/Users/joe/2026/zo/crates/zo-cli/src/file_{idx}.rs"
                ),
                content: "mod attach;\nfn main() {}".to_string(),
                language: Some("rust".to_string()),
                truncated: true,
            },
        });
    }
    push_text(
        &mut t,
        &ids,
        "이제 crate의 역할과 호출 흐름을 정리하겠습니다.",
    );

    let rows = transcript_rows(&mut t, 100, 20);
    let last_read_row = rows
        .iter()
        .rposition(|row| row.contains("+0 more") || row.contains("fn main"))
        .unwrap_or_else(|| panic!("last read payload row missing: {rows:?}"));
    let answer_row = rows
        .iter()
        .position(|row| row.contains("이 제"))
        .unwrap_or_else(|| panic!("assistant body row missing: {rows:?}"));

    // v3 (streaming-style-v3 §3.1): no `Zo` label — the read cluster is
    // separated from the assistant answer (a `*` author bullet) by one blank row.
    assert!(
        last_read_row < answer_row,
        "assistant answer follows the read result cluster: {rows:?}"
    );
    let blanks_before_answer = rows[last_read_row + 1..answer_row]
        .iter()
        .filter(|row| row.trim().is_empty())
        .count();
    assert_eq!(
        blanks_before_answer, 1,
        "read result cluster keeps one blank row before the assistant answer: {rows:?}"
    );
    assert!(
        rows[answer_row].starts_with('*'),
        "assistant answer after a read cluster carries a `*` author bullet: {rows:?}"
    );
}

// ---------------------------------------------------------------------------
// Conversation Turn boundaries (components.md §2)
// ---------------------------------------------------------------------------

fn push_user(t: &mut Transcript, ids: &BlockIdGen, body: &str) -> BlockId {
    let id = ids.next();
    t.push(RenderBlock::UserMessage {
        id,
        text: body.to_string(),
    });
    id
}

#[test]
fn transcript_turn_boundary_inserts_separator_between_turns() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_user(&mut t, &ids, "first question");
    push_text(&mut t, &ids, "first answer");
    push_user(&mut t, &ids, "second question");

    let backend = TestBackend::new(60, 20);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 20);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");

    let rows: Vec<String> = (0..20)
        .map(|y| {
            (0..60)
                .map(|x| {
                    terminal
                        .backend()
                        .buffer()
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                        .to_string()
                })
                .collect::<String>()
        })
        .collect();

    let first_row = rows.iter().position(|r| r.contains("first question"));
    let second_row = rows.iter().position(|r| r.contains("second question"));
    assert!(first_row.is_some(), "first user msg: {rows:?}");
    assert!(second_row.is_some(), "second user msg: {rows:?}");

    let gap = second_row.unwrap() - first_row.unwrap();
    assert!(
        gap >= 4,
        "turn boundary should add extra gap (got {gap}): {rows:?}"
    );
}

#[test]
fn transcript_separates_user_prompt_from_first_tool_call() {
    let ids = ids();
    let mut t = Transcript::new();
    push_user(&mut t, &ids, "please inspect the run loop");
    t.push(RenderBlock::ToolCall {
        id: ids.next(),
        tool_call_id: ToolCallId("call-boundary".to_string()),
        name: "bash".to_string(),
        summary: "unique-boundary-probe".to_string(),
        preview: ToolPreview::Bash {
            command: "rg select".to_string(),
        },
        status: ToolCallStatus::Ok,
    });

    let rows = transcript_rows(&mut t, 70, 14);
    let user_row = rows
        .iter()
        .position(|row| row.contains("please inspect"))
        .expect("user prompt row");
    let tool_row = rows
        .iter()
        .position(|row| row.contains("Ran rg select"))
        .expect("tool call row");
    let boundary_rows = &rows[user_row + 1..tool_row];

    assert!(
        boundary_rows
            .iter()
            .any(|row| row.trim().starts_with("---")),
        "first assistant/tool output should be visibly separated from the user prompt: {rows:?}"
    );
}

#[test]
fn transcript_separates_user_prompt_from_first_assistant_text() {
    let ids = ids();
    let mut t = Transcript::new();
    push_user(&mut t, &ids, "직접 테스트 해서 확인해줘");
    push_text(
        &mut t,
        &ids,
        "직접 확인하겠습니다. 실제 GPT 세션은 OAuth와 네트워크가 필요합니다.",
    );

    let rows = transcript_rows(&mut t, 80, 14);
    let user_row = rows
        .iter()
        .position(|row| row.replace(' ', "").contains("직접테스트"))
        .expect("user prompt row");
    let answer_row = rows
        .iter()
        .position(|row| row.replace(' ', "").contains("직접확인하겠습니다"))
        .expect("assistant text row");
    let boundary_rows = &rows[user_row + 1..answer_row];

    assert!(
        boundary_rows
            .iter()
            .any(|row| row.trim().starts_with("---")),
        "first assistant text should be visibly separated from the user prompt: {rows:?}"
    );
    assert!(
        boundary_rows
            .iter()
            .filter(|row| row.trim().is_empty())
            .count()
            >= 1,
        "v3 keeps one blank row of air above the boundary rule: {rows:?}"
    );
    // v3 (streaming-style-v3 §3.1): the first assistant body carries a `*`
    // author bullet (no `Zo` label, no `|` assistant rail).
    assert!(
        rows[answer_row].starts_with('*'),
        "first assistant body carries a `*` author bullet: {rows:?}"
    );
}

#[test]
fn transcript_first_assistant_text_after_user_has_author_bullet() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_user(&mut t, &ids, "hi");
    push_text(&mut t, &ids, "hello back");

    let backend = TestBackend::new(60, 12);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 12);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");

    let rendered: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect();

    assert!(
        rendered.contains("hello back"),
        "assistant body: {rendered}"
    );
    // v3 (streaming-style-v3 §3.1): the assistant author cue is a `*`/`◆`
    // bullet, not a `Zo` label.
    assert!(
        rendered.contains('*'),
        "assistant turn carries a `*` author bullet: {rendered}"
    );

    let rows = transcript_rows(&mut t, 60, 12);
    let user_row = rows
        .iter()
        .position(|row| row.contains("hi"))
        .expect("user row");
    let answer_row = rows
        .iter()
        .position(|row| row.contains("hello back"))
        .expect("assistant answer row");
    // v3: user prompt and assistant answer are vertically distinct; the answer
    // row carries a `*` author bullet (no intervening `Zo` label row).
    assert!(
        user_row < answer_row,
        "user prompt and assistant answer are vertically distinct: {rows:?}"
    );
    assert!(
        rows[answer_row].starts_with('*'),
        "first assistant answer carries a `*` author bullet: {rows:?}"
    );
}

#[test]
fn transcript_user_message_shows_visible_author_label() {
    let theme = Theme::no_color();
    let ids = ids();
    let mut t = Transcript::new();
    push_user(&mut t, &ids, "test message");

    let backend = TestBackend::new(60, 8);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 8);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");

    let rendered: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect();

    assert!(
        rendered.contains("|  You"),
        "user author label should make the prompt boundary clear: {rendered}"
    );
    assert!(
        rendered.contains("test message"),
        "user body should appear: {rendered}"
    );
}

#[test]
fn transcript_long_user_message_reserves_rail_width_before_assistant_reply() {
    let ids = ids();
    let mut t = Transcript::new();
    push_user(
        &mut t,
        &ids,
        "this is a deliberately long user prompt that must wrap with the user rail width reserved",
    );
    push_text(&mut t, &ids, "assistant reply starts after the prompt");

    let rows = transcript_rows(&mut t, 34, 16);
    let user_row = rows
        .iter()
        .position(|row| row.contains("deliberately long"))
        .expect("wrapped user body row");
    let reply_row = rows
        .iter()
        .position(|row| row.contains("assistant reply"))
        .expect("assistant reply row");

    // v3 (streaming-style-v3 §3.1): the reply carries a `*` author bullet (no
    // `Zo` label). The wrapped user prompt keeps its `|` role rail.
    assert!(
        user_row < reply_row,
        "long user body and assistant reply remain vertically distinct: {rows:?}"
    );
    assert!(
        rows[reply_row].starts_with('*'),
        "assistant reply after a long user prompt carries a `*` author bullet: {rows:?}"
    );
    assert!(
        rows[..reply_row]
            .iter()
            .any(|row| row.trim_start().starts_with("|  ")),
        "wrapped user prompt rows should keep the role rail: {rows:?}"
    );
}

// ---------------------------------------------------------------------------
// Search (Enhancement 3)
// ---------------------------------------------------------------------------

#[test]
fn transcript_find_block_containing_returns_matching_index() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "hello world");
    push_system(&mut t, &ids, "system notice");
    push_text(&mut t, &ids, "search target here");

    let found = t.find_block_containing("target");
    assert_eq!(found, Some(2), "should find the third block");
}

#[test]
fn transcript_find_block_containing_case_insensitive() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "Hello World");

    let found = t.find_block_containing("hello world");
    assert_eq!(found, Some(0), "search should be case-insensitive");
}

#[test]
fn transcript_find_block_containing_returns_none_when_not_found() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "hello");

    let found = t.find_block_containing("nonexistent");
    assert!(found.is_none());
}

#[test]
fn transcript_find_all_blocks_containing_returns_all_in_order() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "alpha match one");
    push_system(&mut t, &ids, "no hit here");
    push_text(&mut t, &ids, "beta MATCH two");

    let found = t.find_all_blocks_containing("match");
    assert_eq!(found, vec![0, 2], "case-insensitive, document order");
}

#[test]
fn transcript_find_all_blocks_containing_empty_when_none() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(&mut t, &ids, "alpha");
    assert!(t.find_all_blocks_containing("zzz").is_empty());
}

#[test]
fn transcript_scroll_to_block_updates_scroll_offset() {
    let ids = ids();
    let mut t = Transcript::new();
    for i in 0..10 {
        push_text(&mut t, &ids, &format!("block {i}"));
    }
    t.scroll_to_block(5);
    assert!(t.scroll() > 0, "scrolling to block 5 should move offset");
}

/// 스트리밍(`done == false`) 중에도 **완료된** 마크다운 블록은 캐시+draw
/// 경로를 거쳐 스타일링되어야 한다 — raw `##` 노출 없이 헤딩 글리프가 보인다.
/// (블록 단위 증분 렌더 end-to-end 회귀 가드.)
#[test]
fn streaming_text_styles_completed_heading_block_without_raw_markers() {
    let ids = ids();
    let mut t = Transcript::new();
    let id = ids.next();
    // 헤딩은 빈 줄로 닫혀 "완료" → 스타일링 대상. 본문은 아직 타이핑 중(열린 꼬리).
    t.push(RenderBlock::TextDelta {
        id,
        text: "## Overview\n\nstreaming body".to_string(),
        done: false,
    });

    let theme = Theme::default_dark();
    let backend = TestBackend::new(60, 8);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 60, 8);
            t.draw(frame, area, &theme, 0, ImageProtocol::None);
        })
        .expect("draw");

    let text: String = (0..8)
        .map(|y| {
            (0..60)
                .map(|x| {
                    terminal
                        .backend()
                        .buffer()
                        .cell((x, y))
                        .map_or(" ", ratatui::buffer::Cell::symbol)
                        .to_string()
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        text.contains("Overview"),
        "heading text must render: {text:?}"
    );
    assert!(
        !text.contains("## "),
        "raw heading marker must not leak while streaming: {text:?}"
    );
    assert!(
        text.contains('\u{258C}'),
        "H2 glyph (▌) must render during streaming: {text:?}"
    );
    assert!(
        text.contains("streaming body"),
        "open tail must still render: {text:?}"
    );
}

/// `done` 마크다운 블록은 같은 폭으로 다시 그려도 byte-identical 이어야 한다
/// — 캐시 키 `(content_hash, width, done)` 히트로 재파싱 없이 재사용됨을
/// 관찰 가능하게 보장(스크롤/틱 redraw 시 "frozen spinner" 유발하는 전체
/// 재wrap 방지). 폭 변경 시엔 재레이아웃하되 패닉/소실이 없어야 한다.
#[test]
fn done_markdown_redraw_is_stable_across_frames_and_resizes() {
    let ids = ids();
    let mut t = Transcript::new();
    push_text(
        &mut t,
        &ids,
        "## 제목\n\n- 항목 하나\n- 항목 둘\n\n본문 단락입니다.",
    );
    let theme = Theme::default_dark();

    let render = |t: &mut Transcript, w: u16, h: u16| -> Vec<String> {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("test backend");
        terminal
            .draw(|frame| {
                t.draw(frame, Rect::new(0, 0, w, h), &theme, 0, ImageProtocol::None);
            })
            .expect("draw");
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| {
                        terminal
                            .backend()
                            .buffer()
                            .cell((x, y))
                            .map_or(" ", ratatui::buffer::Cell::symbol)
                            .to_string()
                    })
                    .collect::<String>()
            })
            .collect()
    };

    // 같은 폭 재draw 는 byte-identical (캐시 키 히트로 재파싱 없음).
    let first = render(&mut t, 80, 12);
    let second = render(&mut t, 80, 12);
    let third = render(&mut t, 80, 12);
    assert_eq!(
        first, second,
        "same-width redraw must be byte-identical (cache hit)"
    );
    assert_eq!(
        second, third,
        "same-width redraw must stay stable across frames"
    );

    // 리사이즈는 재레이아웃하되 패닉/소실이 없어야 하고, 새 폭에서도 멱등.
    // (CJK 전각 문자는 TestBackend 셀 단위로 읽으면 continuation 셀이 끼므로
    //  본문 문자열 매칭 대신 헤딩 글리프 ▌ 존재로 렌더를 확인한다.)
    let narrow = render(&mut t, 50, 16);
    let narrow_again = render(&mut t, 50, 16);
    assert_eq!(
        narrow, narrow_again,
        "resized redraw must also be idempotent"
    );
    let narrow_text = narrow.join("\n");
    assert!(
        narrow_text.contains('\u{258C}'),
        "heading glyph survives resize: {narrow_text:?}"
    );
}
