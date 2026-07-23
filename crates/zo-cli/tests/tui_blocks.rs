//! Integration tests for Lane L5 block widgets.
//!
//! Living-standard naming (L1): `<area>_<scenario>`. Every test
//! renders into a `ratatui::backend::TestBackend` and asserts
//! substrings of the rendered buffer — this keeps assertions
//! resilient to minor layout tweaks while still locking in the
//! visible contract that `.zo/design/components.md` §5 promises.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use runtime::message_stream::{
    BashResult, BlockId, DiffHunk, DiffLine, DiffLineKind, DiffView, PermissionChoice,
    PermissionDecision, PermissionPrompt, RenderBlock, SystemLevel, ToolCallId, ToolCallStatus,
    ToolPreview, ToolResultBody,
};
use zo_cli::tui::blocks::text;
use zo_cli::tui::blocks::{BlockDrawCtx, draw_block};
use zo_cli::tui::image_protocol::ImageProtocol;
use zo_cli::tui::theme::Theme;
use tokio::sync::oneshot;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn render(block: &RenderBlock, width: u16, height: u16, theme: &Theme) -> Buffer {
    render_with_tick(block, width, height, theme, false, false, 0)
}

fn render_with(
    block: &RenderBlock,
    width: u16,
    height: u16,
    theme: &Theme,
    focused: bool,
    expanded: bool,
) -> Buffer {
    render_with_tick(block, width, height, theme, focused, expanded, 0)
}

fn render_with_tick(
    block: &RenderBlock,
    width: u16,
    height: u16,
    theme: &Theme,
    focused: bool,
    expanded: bool,
    tick: u64,
) -> Buffer {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|f| {
            let rect = Rect::new(0, 0, width, height);
            draw_block(
                f,
                rect,
                block,
                &BlockDrawCtx {
                    theme,
                    focused,
                    expanded,
                    tick,
                    scroll_offset: 0,
                    image_protocol: ImageProtocol::None,
                    is_tail_active: false,
                },
            );
        })
        .expect("draw");
    terminal.backend().buffer().clone()
}

fn buffer_contains(buf: &Buffer, needle: &str) -> bool {
    for y in 0..buf.area.height {
        let mut row = String::new();
        for x in 0..buf.area.width {
            row.push_str(buf.cell((x, y)).map_or(" ", ratatui::buffer::Cell::symbol));
        }
        if row.contains(needle) {
            return true;
        }
    }
    false
}

fn buffer_rows(buf: &Buffer) -> Vec<String> {
    (0..buf.area.height)
        .map(|y| {
            let mut row = String::new();
            for x in 0..buf.area.width {
                row.push_str(buf.cell((x, y)).map_or(" ", ratatui::buffer::Cell::symbol));
            }
            row
        })
        .collect()
}

fn text_block(text: &str, done: bool) -> RenderBlock {
    RenderBlock::TextDelta {
        id: BlockId(1),
        text: text.to_string(),
        done,
    }
}

fn reasoning_block(text: &str, done: bool) -> RenderBlock {
    RenderBlock::Reasoning {
        id: BlockId(2),
        text: text.to_string(),
        signature: None,
        done,
    }
}

fn tool_call_block(status: ToolCallStatus) -> RenderBlock {
    RenderBlock::ToolCall {
        id: BlockId(3),
        tool_call_id: ToolCallId("call-1".to_string()),
        name: "Read".to_string(),
        summary: "main.rs:1-20".to_string(),
        preview: ToolPreview::Read {
            path: "main.rs".to_string(),
            range: Some((1, 20)),
        },
        status,
    }
}

fn tool_result_diff() -> RenderBlock {
    RenderBlock::ToolResult {
        id: BlockId(4),
        tool_call_id: ToolCallId("call-1".to_string()),
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("parser.rs".to_string()),
            new_path: Some("parser.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 41,
                old_lines: 1,
                new_start: 41,
                new_lines: 2,
                lines: vec![
                    DiffLine {
                        kind: DiffLineKind::Context,
                        text: "ctx".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Added,
                        text: "added".to_string(),
                    },
                    DiffLine {
                        kind: DiffLineKind::Removed,
                        text: "removed".to_string(),
                    },
                ],
            }],
        }),
    }
}

fn tool_result_long_diff() -> RenderBlock {
    let lines = (0..24)
        .map(|idx| DiffLine {
            kind: if idx % 2 == 0 {
                DiffLineKind::Added
            } else {
                DiffLineKind::Removed
            },
            text: format!("line-{idx}"),
        })
        .collect();

    RenderBlock::ToolResult {
        id: BlockId(8),
        tool_call_id: ToolCallId("call-long-diff".to_string()),
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("long.rs".to_string()),
            new_path: Some("long.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 12,
                new_start: 1,
                new_lines: 12,
                lines,
            }],
        }),
    }
}

fn tool_result_bash() -> RenderBlock {
    RenderBlock::ToolResult {
        id: BlockId(5),
        tool_call_id: ToolCallId("call-2".to_string()),
        is_error: false,
        body: ToolResultBody::Bash(BashResult {
            exit_code: 0,
            stdout: "running tests\nok 1 passed".to_string(),
            stderr: String::new(),
            truncated: false,
        }),
    }
}

fn permission_block() -> RenderBlock {
    let (tx, _rx) = oneshot::channel();
    RenderBlock::PermissionPrompt(PermissionPrompt {
        id: BlockId(6),
        tool_call_id: ToolCallId("call-p".to_string()),
        tool_name: "Bash".to_string(),
        reasoning: "delete build artifacts".to_string(),
        audit_hint: None,
        choices: vec![
            PermissionChoice {
                key: 'y',
                label: "yes".to_string(),
                decision: PermissionDecision::AllowOnce,
            },
            PermissionChoice {
                key: 'n',
                label: "no".to_string(),
                decision: PermissionDecision::Deny,
            },
        ],
        responder: tx,
    })
}

fn permission_block_with_audit() -> RenderBlock {
    let (tx, _rx) = oneshot::channel();
    RenderBlock::PermissionPrompt(PermissionPrompt {
        id: BlockId(7),
        tool_call_id: ToolCallId("call-audit".to_string()),
        tool_name: "PowerShell".to_string(),
        reasoning: "requires approval to escalate from read-only".to_string(),
        audit_hint: Some("risk: high; explicitly unblock with [y] Allow".to_string()),
        choices: vec![
            PermissionChoice {
                key: 'y',
                label: "allow".to_string(),
                decision: PermissionDecision::AllowOnce,
            },
            PermissionChoice {
                key: 'n',
                label: "deny".to_string(),
                decision: PermissionDecision::Deny,
            },
        ],
        responder: tx,
    })
}

fn system_block(level: SystemLevel) -> RenderBlock {
    RenderBlock::System {
        id: BlockId(7),
        level,
        text: "transcript cleared".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Text widget
// ---------------------------------------------------------------------------

#[test]
fn text_block_renders_assistant_prefix_and_body() {
    let theme = Theme::no_color();
    let buf = render(&text_block("hello world", true), 40, 2, &theme);
    assert!(buffer_contains(&buf, "hello"));
    // Assistant text is prefixless in the OpenCode-style transcript.
    assert!(!buffer_contains(&buf, "⏺"));
}

#[test]
fn text_block_bullet_mark_carries_author_diamond_without_header() {
    // v3 bullet grammar: the first spoken answer carries the author bullet
    // (`◆`, `*` under NO_COLOR) on its first body row. The old `Zo` header
    // row and its boundary rule are retired — the turn separator owns
    // boundaries now.
    let theme = Theme::no_color();
    let backend = TestBackend::new(50, 3);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|f| {
            text::draw_with_mark(
                f,
                Rect::new(0, 0, 50, 3),
                "hello world",
                true,
                &theme,
                0,
                0,
                zo_cli::tui::blocks::ProseMark::Bullet,
            );
        })
        .expect("draw");
    let buf = terminal.backend().buffer().clone();

    let rows = buffer_rows(&buf);
    assert!(
        rows[0].starts_with("*  hello world"),
        "bullet mark rides the first body row: {rows:?}"
    );
    assert!(
        !buffer_contains(&buf, "Zo") && !buffer_contains(&buf, "--------"),
        "the retired Zo header/boundary rule must not resurface: {rows:?}"
    );
}

#[test]
fn text_block_indent_continuation_keeps_body_column_without_marks() {
    // v3: an `Indent` continuation repeats no bullet and carries no live/done
    // rail mark (the caret was retired) — the body simply stays in the shared
    // col-3 mark column, identical while streaming and once settled.
    let theme = Theme::no_color();

    let render_continuation = |done: bool| {
        let backend = TestBackend::new(48, 2);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|f| {
                text::draw_with_mark(
                    f,
                    Rect::new(0, 0, 48, 2),
                    if done {
                        "settled continuation"
                    } else {
                        "live continuation"
                    },
                    done,
                    &theme,
                    0,
                    0,
                    zo_cli::tui::blocks::ProseMark::Indent,
                );
            })
            .expect("draw");
        buffer_rows(&terminal.backend().buffer().clone())
    };

    let live = render_continuation(false);
    let done = render_continuation(true);

    assert!(
        live[0].starts_with("   live continuation"),
        "live continuation keeps the bare body column (no caret/rail): {live:?}"
    );
    assert!(
        done[0].starts_with("   settled continuation"),
        "settled continuation keeps the bare body column: {done:?}"
    );
}

#[test]
fn text_block_verifier_json_renders_summary_not_raw_payload() {
    let theme = Theme::no_color();
    let buf = render(
        &text_block(
            r#"{"accepted":false,"issues":["A leaked untracked local settings file remains in the worktree."]}"#,
            true,
        ),
        90,
        4,
        &theme,
    );

    assert!(
        buffer_contains(&buf, "Verification rejected"),
        "verifier verdict should be summarized"
    );
    assert!(
        buffer_contains(&buf, "leaked untracked local settings"),
        "issue text should remain visible"
    );
    assert!(
        !buffer_contains(&buf, "\"accepted\"") && !buffer_contains(&buf, "\"issues\""),
        "raw verifier JSON keys should not be shown"
    );
}

#[test]
fn text_block_verifier_accept_json_renders_success_summary() {
    let theme = Theme::no_color();
    let buf = render(
        &text_block(r#"{"accepted":true,"issues":[]}"#, true),
        60,
        2,
        &theme,
    );

    assert!(buffer_contains(&buf, "Verification accepted"));
    assert!(
        !buffer_contains(&buf, "\"accepted\""),
        "raw verifier JSON key should not be shown"
    );
}

#[test]
fn text_block_streaming_body_carries_no_caret() {
    // v3 §4: the streaming caret was retired — the body carries no progress
    // decoration at any tick; liveness belongs to the bottom activity line.
    let theme = Theme::no_color();
    for tick in [0, 16] {
        let buf = render_with_tick(&text_block("partial", false), 40, 2, &theme, false, false, tick);
        assert!(
            buffer_contains(&buf, "partial"),
            "streaming body visible at tick {tick}"
        );
        assert!(
            !buffer_contains(&buf, "_"),
            "no caret decoration in the streaming body at tick {tick}"
        );
    }
}

#[test]
fn text_block_preserves_markdown_table_shape() {
    let theme = Theme::no_color();
    // Closed rounded box: ╭┬╮ top, header row, ├┼┤ separator, body rows, ╰┴╯
    // bottom — 6 rows total for a 2-data-row table. Give the viewport enough
    // height so the bottom border is visible too.
    let buf = render(
        &text_block(
            "| Name | Value |\n| ---- | ----- |\n| alpha | 1 |\n| beta | 22 |",
            true,
        ),
        40,
        7,
        &theme,
    );
    assert!(buffer_contains(&buf, "╭───────┬───────╮"));
    assert!(buffer_contains(&buf, "│ Name  │ Value │"));
    assert!(buffer_contains(&buf, "├───────┼───────┤"));
    assert!(buffer_contains(&buf, "│ alpha │ 1     │"));
    assert!(buffer_contains(&buf, "╰───────┴───────╯"));
}

// ---------------------------------------------------------------------------
// Reasoning widget (R6 — Reasoning, not Thinking)
// ---------------------------------------------------------------------------

#[test]
fn reasoning_block_shows_rail_and_italic_body() {
    let theme = Theme::no_color();
    // Expanded mode shows the full body. no_color -> rail becomes `:` and prefix `[step]`.
    let buf = render_with(&reasoning_block("step 1", true), 40, 4, &theme, false, true);
    assert!(buffer_contains(&buf, ":"));
    assert!(buffer_contains(&buf, "[step]"));
    assert!(buffer_contains(&buf, "step 1"));
}

#[test]
fn reasoning_block_collapsed_by_default() {
    let theme = Theme::no_color();
    // Default state (not expanded) keeps work steps out of the transcript.
    let buf = render_with(
        &reasoning_block("a\nb\nc", true),
        50,
        3,
        &theme,
        /*focused=*/ false,
        /*expanded=*/ false,
    );
    assert!(!buffer_contains(&buf, "step · a"));
    assert!(!buffer_contains(&buf, "3 lines"));
}

#[test]
fn reasoning_block_focused_collapsed_stays_hidden() {
    let theme = Theme::no_color();
    let buf = render_with(
        &reasoning_block("a\nb\nc", true),
        50,
        3,
        &theme,
        /*focused=*/ true,
        /*expanded=*/ false,
    );
    assert!(!buffer_contains(&buf, "+ step · a"));
    assert!(!buffer_contains(&buf, "🧠"));
}

#[test]
fn reasoning_block_streaming_caret_blinks_off_on_later_tick() {
    let theme = Theme::no_color();
    // Reasoning shares text.rs's caret cadence (visible 0..16, off 16..32),
    // so the off-phase tick matches `text_block_streaming_caret_blinks_off_…`.
    let buf = render_with_tick(
        &reasoning_block("step 1", false),
        40,
        4,
        &theme,
        false,
        true,
        16,
    );
    assert!(!buffer_contains(&buf, "_"));
}

// ---------------------------------------------------------------------------
// Tool call widget
// ---------------------------------------------------------------------------

#[test]
fn tool_call_block_renders_codex_event_row() {
    let theme = Theme::no_color();
    let buf = render(&tool_call_block(ToolCallStatus::Running), 60, 3, &theme);
    assert!(buffer_contains(&buf, "Explored read main.rs:1-20"));
    assert!(!buffer_contains(&buf, "running"));
    assert!(!buffer_contains(&buf, "✦"));
}

#[test]
fn tool_call_block_running_tick_does_not_expose_spinner() {
    let theme = Theme::no_color();
    let initial = render(&tool_call_block(ToolCallStatus::Running), 60, 3, &theme);
    let advanced = render_with_tick(
        &tool_call_block(ToolCallStatus::Running),
        60,
        3,
        &theme,
        false,
        false,
        3,
    );
    assert!(buffer_contains(&initial, "Explored read main.rs:1-20"));
    assert!(buffer_contains(&advanced, "Explored read main.rs:1-20"));
    assert!(!buffer_contains(&initial, "|"));
    assert!(!buffer_contains(&advanced, "/"));
}

#[test]
fn tool_call_block_renders_cancelled_badge() {
    let theme = Theme::no_color();
    let buf = render(&tool_call_block(ToolCallStatus::Cancelled), 60, 3, &theme);
    assert!(buffer_contains(&buf, "cancelled"));
}

// ---------------------------------------------------------------------------
// Tool result widget
// ---------------------------------------------------------------------------

#[test]
fn tool_result_diff_default_view_matches_editor_review_shape() {
    let theme = Theme::no_color();
    let buf = render_with(&tool_result_diff(), 70, 10, &theme, false, false);
    let rendered = buffer_rows(&buf).join("\n");
    assert!(buffer_contains(&buf, "parser.rs (+1 -1)"));
    assert!(buffer_contains(&buf, "added"));
    assert!(buffer_contains(&buf, "removed"));
    assert!(
        !rendered.contains("Diff ·"),
        "ToolResult summary already carries diff metadata; body should start at hunks: {rendered:?}"
    );
    assert!(
        !rendered.contains("Hunk 1") && !rendered.contains("old  new +/- code"),
        "ToolResult diff should use a compact editor-style gutter: {rendered:?}"
    );
}

#[test]
fn tool_result_diff_expanded_keeps_editor_review_shape() {
    let theme = Theme::no_color();
    let buf = render_with(&tool_result_diff(), 70, 10, &theme, true, true);
    assert!(buffer_contains(&buf, "parser.rs (+1 -1)"));
    assert!(!buffer_contains(&buf, "Hunk 1"));
}

#[test]
fn tool_result_diff_wrapped_code_aligns_under_code_column() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(59),
        tool_call_id: ToolCallId("call-diff-wrap".to_string()),
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("src/lib.rs".to_string()),
            new_path: Some("src/lib.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 10,
                old_lines: 1,
                new_start: 10,
                new_lines: 1,
                lines: vec![DiffLine {
                    kind: DiffLineKind::Added,
                    text: "let very_long_variable_name = compute_something_with_many_arguments(alpha, beta, gamma);"
                        .to_string(),
                }],
            }],
        }),
    };

    let buf = render_with(&block, 40, 12, &theme, true, true);
    let rows = buffer_rows(&buf);
    let first_code_row = rows
        .iter()
        .position(|row| row.contains("+ let very_long_variable"))
        .expect("first wrapped diff row");
    let code_col = rows[first_code_row]
        .find("let ")
        .expect("code column in first row");
    let continuation_row = first_code_row + 1;
    let continuation_col = rows[continuation_row]
        .char_indices()
        .find(|(idx, ch)| *idx >= code_col && !ch.is_whitespace())
        .map(|(idx, _)| idx)
        .expect("code column in continuation row");

    assert_eq!(
        continuation_col, code_col,
        "wrapped diff continuation should align under the code column, not the rail/gutter: {rows:?}"
    );
}

#[test]
fn tool_result_diff_under_cap_shows_inline_in_full_without_collapse() {
    // Claude Code parity: an edit diff is shown inline by default. This 24-line
    // diff sits above the generic collapse threshold but under the diff inline
    // cap, so it renders in full with no expand chevron and no `more` hint — the
    // change is visible in place without a keystroke. (Task: do not fold edit
    // diffs behind a `tools`/expand gate.)
    let theme = Theme::no_color();
    let buf = render_with(&tool_result_long_diff(), 80, 40, &theme, false, false);
    assert!(buffer_contains(&buf, "long.rs"));
    assert!(buffer_contains(&buf, "long.rs (+12 -12)"));
    // First and last hunk lines both visible → full inline body.
    assert!(buffer_contains(&buf, "line-0"));
    assert!(buffer_contains(&buf, "line-23"));
    // No collapsed-diff affordances.
    assert!(
        !buffer_contains(&buf, "▸"),
        "an inline diff under the cap must not show the collapsed chevron"
    );
    assert!(
        !buffer_contains(&buf, "more"),
        "a fully-inline diff must not show a `more` hint"
    );
}

#[test]
fn tool_result_diff_over_cap_shows_capped_inline_preview_with_more_hint() {
    // A diff larger than the inline cap still renders inline (no expand gate),
    // but capped to a substantial slice plus a `+N lines` / `more` hint so a
    // huge refactor does not flood the transcript. The rest stays available via
    // expand / `/diff`.
    let theme = Theme::no_color();
    let lines = (0..120)
        .map(|idx| DiffLine {
            kind: if idx % 2 == 0 {
                DiffLineKind::Added
            } else {
                DiffLineKind::Removed
            },
            text: format!("line-{idx}"),
        })
        .collect();
    let block = RenderBlock::ToolResult {
        id: BlockId(8),
        tool_call_id: ToolCallId("call-huge-diff".to_string()),
        is_error: false,
        body: ToolResultBody::Diff(DiffView {
            old_path: Some("huge.rs".to_string()),
            new_path: Some("huge.rs".to_string()),
            language: Some("rust".to_string()),
            hunks: vec![DiffHunk {
                old_start: 1,
                old_lines: 60,
                new_start: 1,
                new_lines: 60,
                lines,
            }],
        }),
    };

    // Collapsed (not expanded): inline preview shows the head of the diff and a
    // `more` hint, but not the tail.
    let buf = render_with(&block, 80, 40, &theme, false, false);
    assert!(buffer_contains(&buf, "huge.rs"));
    assert!(buffer_contains(&buf, "line-0"));
    assert!(
        buffer_contains(&buf, "more"),
        "an over-cap diff must surface a `more` hint for the hidden tail"
    );
    assert!(
        !buffer_contains(&buf, "line-119"),
        "an over-cap diff must not render its full tail inline"
    );
}

#[test]
fn tool_result_successful_bash_suppresses_body_until_expanded() {
    let theme = Theme::no_color();
    let stdout = (0..30)
        .map(|idx| format!("line-{idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let block = RenderBlock::ToolResult {
        id: BlockId(9),
        tool_call_id: ToolCallId("call-long-bash".to_string()),
        is_error: false,
        body: ToolResultBody::Bash(BashResult {
            exit_code: 0,
            stdout,
            stderr: String::new(),
            truncated: false,
        }),
    };

    // A successful bash collapses to just its summary line (which may carry the
    // first stdout line as its one-line essence): the chevron still marks it
    // expandable, but the stdout *body dump* — the bold "stdout (N)" header and
    // the "+N more" overflow hint — is gone. The full output is noise here.
    let buf = render_with(&block, 80, 5, &theme, false, false);
    assert!(buffer_contains(&buf, "▸"), "toggleable chevron survives");
    assert!(
        !buffer_contains(&buf, "stdout (30)"),
        "no stdout body header on a successful bash"
    );
    assert!(
        !buffer_contains(&buf, "more"),
        "no overflow hint on a quiet success"
    );

    // Expanding restores the full body for auditability.
    let expanded = render_with(&block, 80, 40, &theme, false, true);
    assert!(buffer_contains(&expanded, "stdout (30)"));
}

#[test]
fn tool_result_bash_success_is_quiet_no_done_no_exit_zero() {
    let theme = Theme::no_color();
    let buf = render_with(&tool_result_bash(), 60, 4, &theme, false, false);
    // ✓ XOR done: the call row's marker is the sole success signal, so a clean
    // exit-0 bash carries neither a redundant "done" badge nor an "exit 0".
    // Its one informative essence — the first stdout line — is the summary.
    assert!(buffer_contains(&buf, "running tests"));
    assert!(!buffer_contains(&buf, "done"));
    assert!(!buffer_contains(&buf, "exit 0"));
    // A short (non-long, non-Read/Diff) bash is not toggleable, so no chevron.
    assert!(!buffer_contains(&buf, "▸"));
    assert!(!buffer_contains(&buf, "Bash:"));
    assert!(!buffer_contains(&buf, "stderr (0)"));
}

#[test]
fn tool_result_bash_stderr_only_hides_empty_stdout() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(50),
        tool_call_id: ToolCallId("call-stderr".to_string()),
        is_error: true,
        body: ToolResultBody::Bash(BashResult {
            exit_code: 2,
            stdout: String::new(),
            stderr: "permission denied\nmissing file".to_string(),
            truncated: false,
        }),
    };

    let buf = render_with(&block, 70, 5, &theme, true, true);
    assert!(buffer_contains(&buf, "stderr (2)"));
    assert!(buffer_contains(&buf, "permission denied"));
    assert!(
        !buffer_contains(&buf, "stdout (0)"),
        "empty stdout section should stay hidden"
    );
}

#[test]
fn tool_result_bash_empty_output_is_one_quiet_line() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(51),
        tool_call_id: ToolCallId("call-empty".to_string()),
        is_error: false,
        body: ToolResultBody::Bash(BashResult {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
            truncated: false,
        }),
    };

    let buf = render_with(&block, 70, 4, &theme, true, true);
    assert!(buffer_contains(&buf, "no output"));
    assert!(!buffer_contains(&buf, "stdout (0)"));
    assert!(!buffer_contains(&buf, "stderr (0)"));
}

#[test]
fn tool_result_listing_summary_shows_count_and_first_entry() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(52),
        tool_call_id: ToolCallId("call-list".to_string()),
        is_error: false,
        body: ToolResultBody::Listing {
            entries: vec![
                "src/main.rs".to_string(),
                "src/lib.rs".to_string(),
                "README.md".to_string(),
            ],
            truncated: false,
        },
    };

    let buf = render(&block, 80, 3, &theme);
    assert!(buffer_contains(&buf, "3 entries"));
    assert!(buffer_contains(&buf, "src/main.rs"));
}

#[test]
fn tool_result_listing_truncated_summary_shows_more_entries() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(53),
        tool_call_id: ToolCallId("call-list-truncated".to_string()),
        is_error: false,
        body: ToolResultBody::Listing {
            entries: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            truncated: true,
        },
    };

    let buf = render(&block, 80, 3, &theme);
    assert!(buffer_contains(&buf, "2+ entries"));
    assert!(buffer_contains(&buf, "src/main.rs"));
}

#[test]
fn tool_result_ask_user_question_error_reads_like_question_state() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(55),
        tool_call_id: ToolCallId("call-question".to_string()),
        is_error: true,
        body: ToolResultBody::Generic {
            name: "AskUserQuestion".to_string(),
            content: "User question dismissed without answer".to_string(),
            truncated: false,
        },
    };

    let buf = render_with(&block, 80, 4, &theme, false, false);
    assert!(buffer_contains(&buf, "Question:"));
    assert!(buffer_contains(&buf, "dismissed before answer"));
    assert!(buffer_contains(&buf, "not answered"));
    assert!(
        !buffer_contains(&buf, "Result:"),
        "question result should not look like a generic tool result"
    );
    assert!(
        !buffer_contains(&buf, "x error"),
        "dismissed questions should not show the low-level error badge"
    );
}

#[test]
fn tool_result_read_summary_counts_decoded_lines() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(54),
        tool_call_id: ToolCallId("call-read-escaped".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "src/lib.rs".to_string(),
            content: "first\\nsecond\\nthird".to_string(),
            language: Some("rust".to_string()),
            truncated: false,
        },
    };

    let buf = render_with(&block, 80, 5, &theme, false, false);
    assert!(buffer_contains(&buf, "src/lib.rs · 3 lines"));
    assert!(!buffer_contains(&buf, "1 lines"));
}

#[test]
fn tool_result_read_compacts_absolute_paths_in_summary_and_body() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(57),
        tool_call_id: ToolCallId("call-read-absolute".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "/Users/joe/2026/zo/crates/zo-cli/src/lib.rs".to_string(),
            content: "first\nsecond".to_string(),
            language: Some("rust".to_string()),
            truncated: false,
        },
    };

    let buf = render_with(&block, 140, 6, &theme, false, false);
    let rendered = buffer_rows(&buf).join("\n");
    assert!(
        rendered.contains("crates/zo-cli/src/lib.rs · 2 lines"),
        "summary should use a compact workspace path: {rendered:?}"
    );
    assert!(
        rendered.contains("crates/zo-cli/src/lib.rs  rust"),
        "body header should use the same compact path: {rendered:?}"
    );
    assert!(
        !rendered.contains("/Users/joe/2026/zo"),
        "absolute workspace prefix should not leak into read results: {rendered:?}"
    );
}

#[test]
fn tool_result_read_defaults_to_summary_and_expands_with_body_rail() {
    let theme = Theme::no_color();
    let content = (0..30)
        .map(|idx| format!("line_{idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let block = RenderBlock::ToolResult {
        id: BlockId(56),
        tool_call_id: ToolCallId("call-read-long".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "src/lib.rs".to_string(),
            content,
            language: Some("rust".to_string()),
            truncated: true,
        },
    };

    let collapsed = render_with(&block, 80, 6, &theme, false, false);
    let collapsed_rows = buffer_rows(&collapsed);
    assert!(
        collapsed_rows.iter().any(|row| row.contains("src/lib.rs")),
        "collapsed summary keeps the target: {collapsed_rows:?}"
    );
    assert!(
        !collapsed_rows.iter().any(|row| row.contains("line_0")),
        "successful long content stays behind expand: {collapsed_rows:?}"
    );

    let expanded = render_with(&block, 80, 6, &theme, false, true);
    let rows = buffer_rows(&expanded);
    let file_row = rows
        .iter()
        .find(|row| row.contains("src/lib.rs") && row.contains("rust"))
        .expect("read preview file row");

    assert!(
        file_row.starts_with("  | src/lib.rs"),
        "file row should sit under the summary with a continuation rail: {rows:?}"
    );
    assert!(
        !file_row.trim_start().starts_with('└'),
        "nested body rows should not look like sibling root branches: {rows:?}"
    );
}

#[test]
fn tool_result_read_color_theme_expands_with_payload_rail() {
    let theme = Theme::default_dark();
    let content = (0..30)
        .map(|idx| format!("line_{idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let block = RenderBlock::ToolResult {
        id: BlockId(57),
        tool_call_id: ToolCallId("call-read-long-color".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "src/lib.rs".to_string(),
            content,
            language: Some("rust".to_string()),
            truncated: true,
        },
    };

    let collapsed = render_with(&block, 80, 6, &theme, false, false);
    let collapsed_rows = buffer_rows(&collapsed);
    assert!(
        collapsed_rows.iter().any(|row| row.contains("src/lib.rs")),
        "collapsed summary keeps the target: {collapsed_rows:?}"
    );
    assert!(
        !collapsed_rows.iter().any(|row| row.contains("line_0")),
        "successful long content stays behind expand: {collapsed_rows:?}"
    );

    let expanded = render_with(&block, 80, 6, &theme, false, true);
    let rows = buffer_rows(&expanded);
    let file_row = rows
        .iter()
        .find(|row| row.contains("src/lib.rs") && row.contains("rust"))
        .expect("read preview file row");

    assert!(
        file_row.starts_with("  │ src/lib.rs"),
        "colored body rows should stay grouped under a quiet payload rail: {rows:?}"
    );
    assert!(
        !file_row.trim_start().starts_with('└'),
        "colored payload rows should not look like sibling root branches: {rows:?}"
    );
}

#[test]
fn tool_result_read_wrapped_path_keeps_body_rail() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(58),
        tool_call_id: ToolCallId("call-read-wrapped-path".to_string()),
        is_error: false,
        body: ToolResultBody::Read {
            path: "/Users/joe/2026/zo/crates/zo-cli/src/session/live_cli.rs"
                .to_string(),
            content: "fn main() {}".to_string(),
            language: Some("rust".to_string()),
            truncated: false,
        },
    };

    let buf = render_with(&block, 32, 12, &theme, false, false);
    let rows = buffer_rows(&buf);
    let first_body_row = rows
        .iter()
        .position(|row| row.starts_with("  | ") && row.contains("crates/"))
        .expect("first wrapped read header row");
    let source_row = rows
        .iter()
        .position(|row| row.contains("fn main"))
        .expect("source preview row");

    assert!(
        source_row > first_body_row + 1,
        "narrow path should wrap before source rows so the test exercises hanging indent: {rows:?}"
    );
    for row in &rows[first_body_row..source_row] {
        assert!(
            row.trim().is_empty() || row.starts_with("  | "),
            "every wrapped read header row must remain under the body rail: {rows:?}"
        );
    }
    for row in &rows[first_body_row + 1..source_row] {
        assert!(
            row.trim().is_empty() || row.starts_with("  |   "),
            "wrapped read header continuations should hang under the payload, not flush against the rail: {rows:?}"
        );
    }
}

#[test]
fn tool_result_focused_uses_accent_border_not_error() {
    let theme = Theme::no_color();
    // Just assert it renders in both focused and unfocused modes.
    let buf_unfocused = render_with(&tool_result_bash(), 60, 4, &theme, false, false);
    let buf_focused = render_with(&tool_result_bash(), 60, 4, &theme, true, false);
    assert!(buffer_contains(&buf_unfocused, "running tests"));
    assert!(buffer_contains(&buf_focused, "running tests"));
}

// ---------------------------------------------------------------------------
// Permission widget
// ---------------------------------------------------------------------------

#[test]
fn permission_block_renders_choices() {
    let theme = Theme::no_color();
    // Height generous enough for the arrow-nav layout (one navigable row per
    // choice + the move/confirm/deny footer); production sizes the modal via
    // `permission::estimate_rows`, so a fixed too-short viewport would clip the
    // last choice off the bottom.
    let buf = render(&permission_block(), 70, 12, &theme);
    assert!(buffer_contains(&buf, "Permission required"));
    assert!(buffer_contains(&buf, "Bash"));
    assert!(buffer_contains(&buf, "[ y ]"));
    assert!(buffer_contains(&buf, "[ n ]"));
}

#[test]
fn permission_block_renders_audit_hint() {
    let theme = Theme::no_color();
    let buf = render(&permission_block_with_audit(), 80, 8, &theme);
    assert!(buffer_contains(&buf, "PowerShell"));
    assert!(buffer_contains(&buf, "Audit"));
    assert!(buffer_contains(&buf, "explicitly unblock"));
    assert!(buffer_contains(&buf, "[ y ]"));
}

#[test]
fn permission_block_focused_highlights_border() {
    let theme = Theme::no_color();
    let buf_a = render_with(&permission_block(), 70, 6, &theme, false, false);
    let buf_b = render_with(&permission_block(), 70, 6, &theme, true, false);
    assert!(buffer_contains(&buf_a, "Permission"));
    assert!(buffer_contains(&buf_b, "Permission"));
}

// ---------------------------------------------------------------------------
// System widget
// ---------------------------------------------------------------------------

#[test]
fn system_block_info_renders_centered_notice() {
    let theme = Theme::no_color();
    let buf = render(&system_block(SystemLevel::Info), 50, 2, &theme);
    assert!(buffer_contains(&buf, "transcript cleared"));
    assert!(!buffer_contains(&buf, "|"));
    assert!(buffer_contains(&buf, "i"));
}

#[test]
fn system_block_warn_and_error_render() {
    let theme = Theme::no_color();
    let warn = render(&system_block(SystemLevel::Warn), 50, 2, &theme);
    let err = render(&system_block(SystemLevel::Error), 50, 2, &theme);
    assert!(buffer_contains(&warn, "!"));
    assert!(buffer_contains(&err, "x"));
}

// ---------------------------------------------------------------------------
// UserNotice widget (send_to_user push)
// ---------------------------------------------------------------------------

#[test]
fn user_notice_renders_to_you_header_and_verbatim_body() {
    let theme = Theme::no_color();
    let block = RenderBlock::UserNotice {
        id: BlockId(9),
        message: "verbatim finding".to_string(),
    };
    let buf = render(&block, 50, 3, &theme);
    // The distinct "to you" header frames it apart from a muted system line,
    // and the pushed content renders verbatim behind the NO_COLOR rail.
    assert!(buffer_contains(&buf, "to you"), "header present");
    assert!(buffer_contains(&buf, "verbatim finding"), "body present");
}

// ---------------------------------------------------------------------------
// Collapsible tool results (Enhancement 2)
// ---------------------------------------------------------------------------

fn tool_result_long_text() -> RenderBlock {
    let content = (0..30)
        .map(|i| format!("output line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    RenderBlock::ToolResult {
        id: BlockId(10),
        tool_call_id: ToolCallId("call-long".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content,
            truncated: false,
        },
    }
}

#[test]
fn tool_result_long_text_defaults_to_one_line_summary() {
    let theme = Theme::no_color();
    // Not focused, not expanded — retain the useful first line without a body
    // preview or a second "more" row.
    let buf = render_with(&tool_result_long_text(), 80, 12, &theme, false, false);
    assert!(
        buffer_contains(&buf, "output line 0"),
        "the summary should retain the first meaningful line"
    );
    assert!(
        buffer_contains(&buf, CHEVRON_COLLAPSED),
        "collapsed chevron should appear"
    );
    assert!(
        !buffer_contains(&buf, "output line 1"),
        "successful payload rows stay hidden until expanded"
    );
    assert!(
        !buffer_contains(&buf, "more"),
        "the compact default should not spend another row on a collapse hint"
    );
}

#[test]
fn tool_result_json_edit_payload_summarizes_file_instead_of_raw_fields() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(11),
        tool_call_id: ToolCallId("call-edit-json".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: r#"{"filePath":"/Users/joe/2026/zo/crates/zo-cli/src/init.rs","gitDiff":null,"oldString":"const GITIGNORE_ENTRIES: [&str; 1] = [];","newString":"const GITIGNORE_ENTRIES: [&str; 4] = [\".zo/settings.local.json\"];"}"#.to_string(),
            truncated: false,
        },
    };

    let buf = render_with(&block, 90, 4, &theme, false, false);
    assert!(
        buffer_contains(&buf, "Edit:"),
        "file edit payloads should be labeled as edits, not generic results"
    );
    assert!(
        buffer_contains(&buf, "zo-cli/src/init.rs"),
        "summary should name the edited file"
    );
    assert!(
        buffer_contains(&buf, "edit"),
        "summary should describe the payload shape"
    );
    assert!(
        !buffer_contains(&buf, "newString"),
        "raw JSON field names should not crowd the summary"
    );
    assert!(
        !buffer_contains(&buf, "GITIGNORE_ENTRIES"),
        "large edit content should not leak into the summary"
    );
}

#[test]
fn tool_result_json_diff_payload_summarizes_file_instead_of_raw_fields() {
    let theme = Theme::no_color();
    let block = RenderBlock::ToolResult {
        id: BlockId(12),
        tool_call_id: ToolCallId("call-diff-json".to_string()),
        is_error: false,
        body: ToolResultBody::Generic {
            name: "Edit".to_string(),
            content: r#"{"filePath":"/Users/joe/2026/zo/crates/zo-cli/src/tui/blocks/tool_result.rs","gitDiff":"diff --git a/tool_result.rs b/tool_result.rs\n+new line"}"#.to_string(),
            truncated: false,
        },
    };

    let buf = render_with(&block, 90, 4, &theme, false, false);
    assert!(
        buffer_contains(&buf, "Diff:"),
        "git diff payloads should be labeled as diffs, not generic results"
    );
    assert!(
        buffer_contains(&buf, "blocks/tool_result.rs"),
        "summary should name the changed file"
    );
    assert!(
        buffer_contains(&buf, "diff"),
        "summary should describe the payload shape"
    );
    assert!(
        !buffer_contains(&buf, "gitDiff"),
        "raw JSON field names should not crowd the summary"
    );
}

#[test]
fn tool_result_escaped_multiline_text_uses_a_decoded_one_line_summary() {
    let theme = Theme::no_color();
    let escaped = (0..30)
        .map(|idx| format!("fn line_{idx}() {{ println!(\\\"ok\\\"); }}"))
        .collect::<Vec<_>>()
        .join("\\n");
    let block = RenderBlock::ToolResult {
        id: BlockId(11),
        tool_call_id: ToolCallId("call-escaped".to_string()),
        is_error: false,
        body: ToolResultBody::Text {
            content: escaped,
            truncated: false,
        },
    };

    let buf = render_with(&block, 80, 8, &theme, false, false);
    assert!(
        buffer_contains(&buf, CHEVRON_COLLAPSED),
        "escaped multiline payload should auto-collapse"
    );
    assert!(
        buffer_contains(&buf, "fn line_0()"),
        "summary should show the decoded first line"
    );
    assert!(
        !buffer_contains(&buf, "\\nfn line_1"),
        "literal escaped newlines must not leak into the rendered preview"
    );
    assert!(
        !buffer_contains(&buf, "line_1()"),
        "the remaining decoded payload should stay hidden until expanded"
    );
    assert!(
        !buffer_contains(&buf, "more"),
        "the one-line summary should not add a redundant hint row"
    );
}

#[test]
fn tool_result_long_text_expands_when_expanded_flag_set() {
    let theme = Theme::no_color();
    // Focused and expanded — should show full body.
    let buf = render_with(&tool_result_long_text(), 80, 30, &theme, true, true);
    assert!(
        buffer_contains(&buf, CHEVRON_EXPANDED),
        "expanded chevron should appear"
    );
    // Should show content beyond the preview lines.
    assert!(
        buffer_contains(&buf, "output line 5"),
        "expanded body should include lines beyond preview"
    );
}

use zo_cli::tui::blocks::tool_result::{CHEVRON_COLLAPSED, CHEVRON_EXPANDED};
