use super::*;
use crate::tui::theme::Theme;

fn dark() -> Theme {
    Theme::default_dark()
}

fn no_color() -> Theme {
    Theme::no_color()
}

fn collect_text(lines: &[Line<'static>]) -> String {
    lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
        .collect()
}

#[test]
fn running_marker_reduce_motion_holds_first_frame_else_cycles() {
    let mut saw_hollow = false;
    for tick in 0..40u64 {
        // Default color path: the ✦/✧ spark heartbeat shared with the
        // activity line (~1Hz on the 33ms tick clock).
        let marker = running_marker(tick, true, false);
        assert!(
            marker == crate::tui::glyphs::ZO_SPARK
                || marker == crate::tui::glyphs::ZO_SPARK_HOLLOW,
            "color marker must be a spark pulse frame: {marker:?}"
        );
        saw_hollow |= marker == crate::tui::glyphs::ZO_SPARK_HOLLOW;
        // NO_COLOR keeps the classic ASCII rotation for liveness.
        assert_eq!(
            running_marker(tick, false, false),
            ["-", "\\", "|", "/"][usize::try_from(tick / 4).unwrap_or(0) % 4]
        );
        // Reduce-motion: settled first frame for every tick, both paths.
        assert_eq!(
            running_marker(tick, true, true),
            crate::tui::glyphs::ZO_SPARK
        );
        assert_eq!(running_marker(tick, false, true), "-");
    }
    assert!(saw_hollow, "the pulse must actually alternate within 40 ticks");
}

/// Per-agent-row status-dot foreground colors, in row order. Scopes to lines
/// bearing a tree branch (`├`/`└`) so the finished-batch header — which also
/// opens with a `✓` glyph — is never miscounted as a row dot. Used to assert the
/// live color band (one colored `●`/`✓`/`×`/`⊘` per agent row).
fn agent_row_dot_colors(lines: &[Line<'static>]) -> Vec<Option<ratatui::style::Color>> {
    let is_branch =
        |s: &str| s.contains('\u{251c}') || s.contains('\u{2514}');
    let is_dot = |s: &str| {
        matches!(
            s.trim(),
            "\u{25cf}" | "\u{2713}" | "\u{00d7}" | "\u{2298}"
        )
    };
    lines
        .iter()
        .filter(|l| l.spans.iter().any(|s| is_branch(&s.content)))
        .flat_map(|l| l.spans.iter())
        .filter(|s| is_dot(&s.content))
        .map(|s| s.style.fg)
        .collect()
}

fn bash_preview(command: &str) -> ToolPreview {
    ToolPreview::Bash {
        command: command.to_string(),
    }
}

fn spawn_preview() -> ToolPreview {
    ToolPreview::Generic {
        name: "SpawnMultiAgent".to_string(),
        input_summary: "delegating".to_string(),
    }
}

fn tree_row(name: &str, status: &str, tools: usize, tokens: u64) -> AgentTreeRow {
    AgentTreeRow {
        agent_id: format!("agent-{name}"),
        name: name.to_string(),
        status: status.to_string(),
        subagent_type: Some("Explore".to_string()),
        tool_calls: Some(tools),
        tokens,
        elapsed_secs: 50,
        ..AgentTreeRow::default()
    }
}

/// 진행 중 배치는 헤더 + 집계 한 줄만 transcript에 남긴다. 개별 에이전트
/// 행은 pinned panel/sidebar/Ctrl+G가 계속 제공한다.
#[test]
fn running_agent_batch_renders_compact_progress() {
    let tree = AgentTree {
        rows: vec![
            tree_row("wiring", "completed", 53, 95_100),
            tree_row("deep-gate", "running", 31, 9_000),
        ],
        batch_label: None,
        finished: false,
    };
    let lines = rendered_lines_with_tree(
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Running,
        &dark(),
        0,
        false,
        None,
        Some(&tree),
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("Running 2 Explore agents\u{2026} (ctrl+g for details)"),
        "live header names count + type: {text}"
    );
    assert!(!text.contains("Spawned"), "generic verb replaced by header: {text}");
    assert!(
        text.contains("└ 1 done · 1 running · 84 tool uses · 104.1k tokens"),
        "compact child line keeps aggregate progress: {text}"
    );
    assert_eq!(lines.len(), 2, "header + one aggregate child: {text}");
    assert!(
        !text.contains("wiring") && !text.contains("deep-gate") && !text.contains("⎿"),
        "per-agent detail must not be duplicated in transcript: {text}"
    );
}

#[test]
fn smart_agent_tree_header_marks_host_prelude_only() {
    let mut smart_tree = AgentTree {
        rows: vec![tree_row("wiring", "running", 3, 1_200)],
        batch_label: Some("Smart".to_string()),
        finished: false,
    };
    let smart_live = running_header(&smart_tree, true);
    assert_eq!(
        smart_live,
        format!(
            "{} Smart running 1 Explore agent\u{2026} (ctrl+g for details)",
            glyphs::SMART_AUTO
        )
    );

    assert_eq!(
        running_header(&smart_tree, false),
        "S Smart running 1 Explore agent\u{2026} (ctrl+g for details)"
    );

    smart_tree.rows[0].status = "completed".to_string();
    smart_tree.finished = true;
    assert_eq!(
        finished_header(&smart_tree, true),
        format!(
            "{} Smart 1 Explore agent finished (ctrl+g for details)",
            glyphs::SMART_AUTO
        )
    );
    assert_eq!(
        finished_header(&smart_tree, false),
        "S Smart 1 Explore agent finished (ctrl+g for details)"
    );

    let plain_tree = AgentTree {
        rows: smart_tree.rows.clone(),
        batch_label: None,
        finished: true,
    };
    assert_eq!(
        finished_header(&plain_tree, true),
        "1 Explore agent finished (ctrl+g for details)"
    );
}

#[test]
fn non_smart_batch_label_is_not_decorated_with_smart_glyph() {
    let tree = AgentTree {
        rows: vec![tree_row("wiring", "running", 3, 1_200)],
        batch_label: Some("Host".to_string()),
        finished: false,
    };

    assert_eq!(
        running_header(&tree, true),
        "Host running 1 Explore agent\u{2026} (ctrl+g for details)"
    );
    assert_eq!(batch_label_with_glyph("Host", true), "Host");
    assert_eq!(batch_label_with_glyph("Host", false), "Host");
}

fn agent_summary(name: &str, status: &str, created_at: u64) -> AgentTaskSummary {
    AgentTaskSummary {
        id: format!("id-{name}"),
        name: name.to_string(),
        status: status.to_string(),
        tool_calls: Some(3),
        tokens: 1_000,
        elapsed_secs: 12,
        current_tool: Some("Grep".to_string()),
        created_at: Some(created_at),
        ..AgentTaskSummary::default()
    }
}

/// 진행 중 행에는 마지막으로 흘려보낸 `outputTail` 한 줄이 `⤷` 리드와 함께
/// dim 서브라인으로 붙는다 (CC 처럼 "무엇을 말하는 중"인지 보이게).
#[test]
fn running_row_shows_last_output_tail_line() {
    let mut row = tree_row("scout", "running", 4, 1_200);
    row.output_tail = Some("first chunk\nscanning crates/runtime for clone()".to_string());
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let text = collect_text(&agent_tree_lines(&tree, &dark()));
    assert!(text.contains('\u{2937}'), "output-tail lead present: {text}");
    assert!(
        text.contains("scanning crates/runtime for clone()"),
        "last tail line shown: {text}"
    );
    assert!(
        !text.contains("first chunk"),
        "only the LAST tail line shows: {text}"
    );
}

#[test]
fn agent_tail_preview_sanitizes_collapses_and_caps_bursty_snapshots() {
    let preview = agent_tail_preview(
        "old\n    streaming      a very very very very very very very very very very long tail line",
    )
    .expect("preview");

    assert!(
        !preview.contains("  "),
        "preview collapses noisy whitespace: {preview:?}"
    );
    assert!(preview.ends_with('\u{2026}'), "long preview is capped: {preview}");
    assert!(
        display_width(&preview) <= AGENT_TAIL_PREVIEW_CELLS,
        "preview must fit the display cap: {preview}"
    );
    assert!(
        !preview.contains("old"),
        "preview uses only the latest nonblank line: {preview}"
    );
}

/// 종결 행은 outputTail 대신 `⎿ Done` 만 보여준다 (꼬리는 running 한정).
#[test]
fn terminal_row_hides_output_tail() {
    let mut row = tree_row("scout", "completed", 4, 1_200);
    row.output_tail = Some("leftover partial".to_string());
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let text = collect_text(&agent_tree_lines(&tree, &dark()));
    assert!(
        !text.contains("leftover partial"),
        "terminal row shows Done, not a tail: {text}"
    );
    assert!(text.contains("Done"), "{text}");
}

/// Smart 라우팅 사유(manifest `routeReason`)가 있으면 `routed: …` dim meta
/// 세그먼트로 행에 붙는다 — P7 관측성: 인라인 트리에서도 auto 라우팅 근거가
/// 보여야 `/smart doctor`/Ctrl+G 없이도 설명 가능하다.
#[test]
fn agent_row_shows_route_reason_when_present() {
    let mut row = tree_row("scout", "running", 4, 1_200);
    row.route_reason = Some("auto:coding tier=strong seed+12".to_string());
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let text = collect_text(&agent_tree_lines(&tree, &dark()));
    assert!(
        text.contains("routed: auto:coding tier=strong seed+12"),
        "route reason meta segment present: {text}"
    );
}

/// 명시 모델 / 라우팅 꺼짐(오늘의 기본값) 케이스는 route_reason 이 `None` 이라
/// 세그먼트 자체가 없다 — 기존 사용자에게는 순수 additive, 노이즈 없음.
#[test]
fn agent_row_omits_route_reason_segment_when_absent() {
    let row = tree_row("scout", "running", 4, 1_200);
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let text = collect_text(&agent_tree_lines(&tree, &dark()));
    assert!(
        !text.contains("routed:"),
        "no segment without a route reason: {text}"
    );
}

/// route_reason 은 스폰 시점에 결정되는 사실이라 종결 행에서도 유지된다
/// (elapsed/activity 처럼 종결과 함께 사라지지 않음).
#[test]
fn terminal_row_still_shows_route_reason() {
    let mut row = tree_row("scout", "completed", 4, 1_200);
    row.route_reason = Some("pin:coding".to_string());
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let text = collect_text(&agent_tree_lines(&tree, &dark()));
    assert!(
        text.contains("routed: pin:coding"),
        "route reason persists past terminal: {text}"
    );
}

/// 핀 라이브 패널: HUD 의 [`AgentTaskSummary`] 스캔에서 `Running N agents…`
/// 헤더 + per-agent 트리를 만든다 (사이드바와 같은 데이터, host ToolCall 불필요).
#[test]
fn live_agent_panel_builds_header_and_rows_from_summaries() {
    let agents = vec![
        agent_summary("beta", "running", 200),
        agent_summary("alpha", "running", 100),
    ];
    let text = collect_text(&live_agent_panel_lines_with_spans(&agents, &dark(), None, None).0);
    assert!(
        text.contains("Running 2 agents\u{2026} (ctrl+g for details)"),
        "header names the live count: {text}"
    );
    // created_at 스폰 순서: alpha(100) 가 beta(200) 보다 먼저.
    let a = text.find("alpha").expect("alpha present");
    let b = text.find("beta").expect("beta present");
    assert!(a < b, "rows sorted by spawn order: {text}");
    assert!(text.contains("Grep"), "live current tool surfaces in the row: {text}");
}

#[test]
fn live_agent_panel_can_render_smart_batch_label() {
    let agents = vec![agent_summary("alpha", "running", 100)];
    let text = collect_text(
        &live_agent_panel_lines_with_spans(&agents, &dark(), Some("Smart"), None).0,
    );
    assert!(
        text.contains(&format!(
            "{} Smart running 1 agent\u{2026} (ctrl+g for details)",
            glyphs::SMART_AUTO
        )),
        "Smart provenance should reach the pinned live panel: {text}"
    );
}

#[test]
fn live_agent_panel_degrades_smart_batch_glyph_under_no_color() {
    let agents = vec![agent_summary("alpha", "running", 100)];
    let text = collect_text(
        &live_agent_panel_lines_with_spans(&agents, &no_color(), Some("Smart"), None).0,
    );
    assert!(
        text.contains("S Smart running 1 agent\u{2026} (ctrl+g for details)"),
        "Smart provenance should use ASCII fallback under no-color: {text}"
    );
    assert!(
        !text.contains(glyphs::SMART_AUTO),
        "rich Smart glyph must not render under no-color: {text}"
    );
}

/// 패널은 행을 상한까지만 보여주고 나머지는 `… +N more` 로 요약한다.
#[test]
fn live_agent_panel_caps_and_summarizes_overflow() {
    let agents: Vec<AgentTaskSummary> = (0..LIVE_PANEL_MAX_AGENTS + 2)
        .map(|i| agent_summary(&format!("a{i}"), "running", u64::try_from(i).unwrap_or(0)))
        .collect();
    let text = collect_text(&live_agent_panel_lines_with_spans(&agents, &dark(), None, None).0);
    assert!(text.contains("+2 more"), "overflow summarized: {text}");
}

#[test]
fn live_agent_panel_is_empty_without_agents() {
    assert!(live_agent_panel_lines_with_spans(&[], &dark(), None, None).0.is_empty());
}

/// 전 에이전트 종결 시 헤더 + 완료 집계만 transcript에 남는다.
#[test]
fn finished_agent_batch_swaps_header_and_compacts_details() {
    let tree = AgentTree {
        rows: vec![
            tree_row("wiring", "completed", 53, 95_100),
            tree_row("audit", "completed", 59, 75_700),
        ],
        batch_label: None,
        finished: true,
    };
    let lines = rendered_lines_with_tree(
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Ok,
        &dark(),
        0,
        false,
        None,
        Some(&tree),
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("2 Explore agents finished (ctrl+g for details)"),
        "{text}"
    );
    assert!(!text.contains("Spawned"), "verb line replaced: {text}");
    assert!(
        text.contains("└ 2 done · 112 tool uses · 170.8k tokens"),
        "aggregate outcome remains visible: {text}"
    );
    assert_eq!(lines.len(), 2, "header + one aggregate child: {text}");
    assert!(
        !text.contains("wiring") && !text.contains("audit") && !text.contains("⎿"),
        "per-agent rows remain in Ctrl+G, not transcript: {text}"
    );
}

/// 라이브 spawn 배치가 아직 매니페스트(행)를 못 낸 구간엔 `⎿ spawning…`
/// placeholder 가 떠서, 맨 `● Spawned` verb 줄로만 멈춰 있지 않는다.
#[test]
fn live_spawn_with_no_rows_yet_shows_spawning_placeholder() {
    // tree 가 아직 없는(None) stuck 케이스.
    let lines = rendered_lines_with_tree(
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Running,
        &dark(),
        0,
        false,
        None,
        None,
    );
    assert!(
        collect_text(&lines).contains("spawning\u{2026}"),
        "placeholder fills the gap before the first manifest"
    );

    // 빈 트리(Some, rows 0)도 동일하게 placeholder.
    let empty = AgentTree {
        rows: vec![],
        batch_label: None,
        finished: false,
    };
    let lines = rendered_lines_with_tree(
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Running,
        &dark(),
        0,
        false,
        None,
        Some(&empty),
    );
    assert!(collect_text(&lines).contains("spawning\u{2026}"));
}

/// placeholder 는 spawn-family / Workflow 전용 — 일반 도구(bash)엔 안 붙는다.
#[test]
fn non_spawn_tool_never_gets_spawning_placeholder() {
    let lines = rendered_lines_with_tree(
        "bash",
        "",
        &bash_preview("npm test"),
        ToolCallStatus::Running,
        &dark(),
        0,
        false,
        None,
        None,
    );
    assert!(
        !collect_text(&lines).contains("spawning"),
        "bash is not a spawn batch"
    );
}

#[test]
fn spawn_family_matcher_covers_task_and_agent() {
    assert!(is_spawn_family("SpawnMultiAgent"));
    assert!(is_spawn_family("Task"));
    assert!(is_spawn_family("Agent"));
    assert!(!is_spawn_family("bash"));
}

fn generic_preview(name: &str) -> ToolPreview {
    ToolPreview::Generic {
        name: name.to_string(),
        input_summary: String::new(),
    }
}

#[test]
fn plan_update_call_row_renders_nothing_result_block_owns_the_plan() {
    let theme = dark();
    let lines = rendered_lines(
        "TodoWrite",
        "",
        &generic_preview("TodoWrite"),
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );

    // A TodoWrite/TaskList call row renders nothing: the ToolResult `Todos`
    // block (`• Updated Plan · N/M done` + checklist) is the complete plan
    // display. Rendering a head here too produced a duplicate `Updated Plan`
    // line above that block (the reported duplication).
    assert!(
        lines.is_empty(),
        "plan-update call row must be empty so the result block alone shows the plan: {:?}",
        collect_text(&lines)
    );
}

#[test]
fn namespaced_plan_update_call_row_renders_nothing() {
    let theme = dark();
    let lines = rendered_lines(
        "functions.TodoWrite",
        r#"[{"activeForm":"Updating plan","content":"Update plan","status":"in_progress"}]"#,
        &generic_preview("functions.TodoWrite"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );

    assert!(
        lines.is_empty(),
        "namespaced TodoWrite must not leak raw plan JSON as a call row: {:?}",
        collect_text(&lines)
    );
}

#[test]
fn task_list_call_row_also_renders_nothing() {
    let theme = dark();
    let lines = rendered_lines(
        "TaskList",
        "",
        &generic_preview("TaskList"),
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );
    assert!(
        lines.is_empty(),
        "TaskList call row is owned by its result block too: {:?}",
        collect_text(&lines)
    );
}

#[test]
fn non_plan_rows_do_not_get_a_filled_rule() {
    let theme = dark();
    let lines = rendered_lines(
        "bash",
        "",
        &bash_preview("pwd"),
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );

    assert_eq!(lines.len(), 1, "ordinary tool rows should stay compact");
    assert!(
        !collect_text(&lines).contains("────────"),
        "non-plan rows must not gain a divider rule"
    );
}

#[test]
fn format_elapsed_sub_minute() {
    assert_eq!(format_elapsed(Duration::from_secs(1)), "1s");
    assert_eq!(format_elapsed(Duration::from_secs(26)), "26s");
    assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
}

#[test]
fn format_elapsed_minutes() {
    assert_eq!(format_elapsed(Duration::from_secs(60)), "1m");
    assert_eq!(format_elapsed(Duration::from_secs(92)), "1m 32s");
    assert_eq!(format_elapsed(Duration::from_secs(3599)), "59m 59s");
}

#[test]
fn format_elapsed_hours() {
    assert_eq!(format_elapsed(Duration::from_secs(3600)), "1h 0m");
    assert_eq!(format_elapsed(Duration::from_secs(7320)), "2h 2m");
}

#[test]
fn running_tool_with_elapsed_shows_seconds_label() {
    let theme = dark();
    let lines = rendered_lines(
        "bash",
        "",
        &bash_preview(""),
        ToolCallStatus::Running,
        &theme,
        0,
        true,
        Some(Duration::from_secs(26)),
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("· 26s"),
        "must include elapsed label: {text:?}"
    );
}

#[test]
fn running_tool_below_one_second_omits_label() {
    let theme = dark();
    let lines = rendered_lines(
        "bash",
        "",
        &bash_preview(""),
        ToolCallStatus::Running,
        &theme,
        0,
        true,
        Some(Duration::from_millis(800)),
    );
    let text = collect_text(&lines);
    assert!(
        !text.contains("(0s)") && !text.contains("(1s)"),
        "sub-second elapsed must not render: {text:?}"
    );
}

#[test]
fn ok_tool_with_elapsed_omits_label() {
    let theme = dark();
    let lines = rendered_lines(
        "bash",
        "",
        &bash_preview(""),
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        Some(Duration::from_secs(10)),
    );
    let text = collect_text(&lines);
    assert!(
        !text.contains("(10s)"),
        "completed tools must not show running elapsed: {text:?}"
    );
}

#[test]
fn completed_tool_renders_codex_event_row_not_spinner() {
    let theme = dark();
    let preview = bash_preview("pwd");
    let lines = rendered_lines(
        "bash",
        "",
        &preview,
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(text.starts_with("✓ Ran pwd"), "event row: {text:?}");
    for frame in SPINNER_FRAMES {
        assert!(
            !text.contains(frame),
            "completed row must not contain spinner frame {frame:?}: {text:?}"
        );
    }
}

#[test]
fn distinct_tools_render_distinct_event_text() {
    let theme = dark();
    let bash_preview = bash_preview("pwd");
    let read_preview = ToolPreview::Read {
        path: "file.rs".to_string(),
        range: None,
    };
    let bash = collect_text(&rendered_lines(
        "bash",
        "",
        &bash_preview,
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    ));
    let read = collect_text(&rendered_lines(
        "Read",
        "",
        &read_preview,
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    ));
    assert!(bash.contains("Ran pwd"), "bash event text: {bash:?}");
    assert!(
        read.contains("Explored read file.rs"),
        "read event text: {read:?}"
    );
}

#[test]
fn running_tool_row_keeps_event_text_with_animated_marker() {
    // 2026-07 revision of the old "no spinner frames" rule: a static `●` on a
    // silent in-flight call read as frozen (a hung MCP request ran 52 minutes
    // indistinguishable from progress, so users cancelled healthy turns).
    // In-flight rows now carry the shared activity-spinner glyph; the event
    // text stays Codex-style and settled rows stay static (see
    // `settled_tool_marker_stays_static`). The legacy 10-frame set stays gone.
    let theme = dark();
    let preview = bash_preview("cargo test");
    let lines = rendered_lines(
        "bash",
        "",
        &preview,
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("Ran cargo test"),
        "event text must survive the marker change: {text:?}"
    );
    let marker = lines
        .first()
        .and_then(|l| l.spans.first())
        .map(|s| s.content.trim().to_string())
        .unwrap_or_default();
    assert!(
        marker == crate::tui::glyphs::ZO_SPARK
            || marker == crate::tui::glyphs::ZO_SPARK_HOLLOW,
        "in-flight marker must be a spark pulse frame: {marker:?}"
    );
    assert!(
        SPINNER_FRAMES.iter().all(|f| !text.contains(f)),
        "the legacy 10-frame spinner set stays retired: {text:?}"
    );
}

#[test]
fn expanded_running_bash_draws_sanitized_live_tail() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let live = tools::live_output::register(Some("call-live-render"));
    live.writer().append_stdout(
        b"\x1b]0;watch\x07old progress\r\x1b[32mnew progress\x1b[0m\nsecond line\n",
    );
    let (width, height) = (80u16, 5u16);
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("backend");
    terminal
        .draw(|frame| {
            draw(
                frame,
                Rect::new(0, 0, width, height),
                "call-live-render",
                "bash",
                "",
                &bash_preview("gh run watch 123"),
                ToolCallStatus::Running,
                &no_color(),
                0,
                0,
                true,
                Some(Duration::from_secs(65)),
                None,
                true,
            );
        })
        .expect("draw");
    let buffer = terminal.backend().buffer();
    let dump = (0..height).fold(String::new(), |mut dump, row| {
        for col in 0..width {
            dump.push_str(buffer[(col, row)].symbol());
        }
        dump.push('\n');
        dump
    });

    assert!(dump.contains("live output · 0s ago"), "{dump}");
    assert!(dump.contains("new progress"), "{dump}");
    assert!(dump.contains("second line"), "{dump}");
    assert!(!dump.contains("old progress") && !dump.contains("[32m"), "{dump}");
}

#[test]
fn pending_error_and_cancelled_tools_have_distinct_markers() {
    let theme = dark();
    let preview = bash_preview("cargo test");
    let cases = [
        (ToolCallStatus::Pending, "○ Ran cargo test"),
        (ToolCallStatus::Errored, "× Ran cargo test"),
        (ToolCallStatus::Cancelled, "⊘ Ran cargo test"),
    ];
    for (status, expected) in cases {
        let lines = rendered_lines("bash", "", &preview, status, &theme, 0, false, None);
        let text = collect_text(&lines);
        assert!(
            text.starts_with(expected),
            "{status:?} marker mismatch: {text:?}"
        );
    }
}

/// pending_action 은 대소문자 무관하게 액션 문구를 찾는다.
#[test]
fn pending_action_is_case_insensitive() {
    assert_eq!(pending_action("grep"), pending_action("Grep"));
    assert_eq!(pending_action("Grep"), Some("Searching content\u{2026}"));
    assert_eq!(pending_action("bash"), Some("Writing command\u{2026}"));
    assert!(pending_action("DefinitelyNotARealTool").is_none());
}

/// 인자가 아직 없는 in-flight(Pending/Running) 도구는 액션 명사형
/// 문구를 summary 로 표시한다 — opencode 의 pending 라벨 패턴.
#[test]
fn in_flight_tool_without_args_shows_action_phrase() {
    let theme = dark();
    let preview = ToolPreview::Grep {
        pattern: "TODO".to_string(),
        path: None,
    };
    for status in [ToolCallStatus::Pending, ToolCallStatus::Running] {
        let lines = rendered_lines("Grep", "", &preview, status, &theme, 0, false, None);
        let text = collect_text(&lines);
        assert!(
            text.contains("Explored grep TODO"),
            "in-flight grep must show compact event text ({status:?}): {text:?}"
        );
    }
}

#[test]
fn pending_generic_tool_without_args_uses_starting_phrase() {
    let theme = dark();
    let preview = generic_preview("bash");
    let lines = rendered_lines(
        "bash",
        "",
        &preview,
        ToolCallStatus::Pending,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("Starting Writing command"),
        "pending generic row should say what is starting: {text:?}"
    );
    assert!(
        !text.contains("Called bash()"),
        "empty call parentheses should not leak into pending rows: {text:?}"
    );
}

#[test]
fn completed_generic_tool_without_args_keeps_event_shape() {
    let theme = dark();
    let preview = generic_preview("bash");
    let lines = rendered_lines(
        "bash",
        "",
        &preview,
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("Called bash({})"),
        "completed generic rows keep their historical event shape: {text:?}"
    );
}

/// 구체적 인자가 도착하면 액션 문구 대신 인자를 표시한다(인자 우선).
#[test]
fn concrete_args_win_over_action_phrase() {
    let theme = dark();
    let summary = r#"bash({"command":"ls -la"})"#;
    let lines = rendered_lines(
        "bash",
        summary,
        &bash_preview("ls -la"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("ls -la"),
        "concrete command must render: {text:?}"
    );
    assert!(
        !text.contains("$ "),
        "Codex-style command summaries omit shell prompt chrome: {text:?}"
    );
    assert!(
        !text.contains("Writing command"),
        "action phrase must not appear once args arrived: {text:?}"
    );
}

#[test]
fn non_tail_running_tool_marker_style_is_stable_across_ticks() {
    let theme = dark();
    let preview = bash_preview("cargo test");
    let tick_a = rendered_lines(
        "bash",
        r#"{"command":"cargo test"}"#,
        &preview,
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let tick_b = rendered_lines(
        "bash",
        r#"{"command":"cargo test"}"#,
        &preview,
        ToolCallStatus::Running,
        &theme,
        8,
        false,
        None,
    );
    assert_eq!(
        tick_a[0].spans[0].style, tick_b[0].spans[0].style,
        "completed/off-tail rows must not keep blinking when a later tool is active"
    );

    let active_a = rendered_lines(
        "bash",
        r#"{"command":"cargo test"}"#,
        &preview,
        ToolCallStatus::Running,
        &theme,
        0,
        true,
        None,
    );
    let active_b = rendered_lines(
        "bash",
        r#"{"command":"cargo test"}"#,
        &preview,
        ToolCallStatus::Running,
        &theme,
        8,
        true,
        None,
    );
    assert_ne!(
        active_a[0].spans[0].style, active_b[0].spans[0].style,
        "the actual tail-active running tool should still pulse"
    );
}

#[test]
fn bash_read_command_renders_codex_summary() {
    let theme = dark();
    let summary =
        r#"{"command":"nl -ba crates/zo-cli/src/tui/transcript.rs | sed -n '1,80p'"}"#;
    let lines = rendered_lines(
        "bash",
        summary,
        &bash_preview("nl -ba crates/zo-cli/src/tui/transcript.rs | sed -n '1,80p'"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("read crates/zo-cli/src/tui/transcript.rs"),
        "read command should be summarized semantically: {text:?}"
    );
    assert!(
        !text.contains("nl -ba") && !text.contains("sed -n"),
        "raw shell read pipeline should stay hidden: {text:?}"
    );
}

#[test]
fn file_tool_rows_compact_absolute_paths() {
    let theme = dark();
    let abs = "/Users/joe/2026/zo/crates/zo-cli/src/tui/transcript.rs";
    let lines = rendered_lines(
        "Read",
        "",
        &ToolPreview::Read {
            path: abs.to_string(),
            range: Some((1, 80)),
        },
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("read crates/zo-cli/src/tui/transcript.rs:1-80"),
        "absolute read path should render repo-relative: {text:?}"
    );
    assert!(
        !text.contains("/Users/joe"),
        "absolute user path should not crowd the call row: {text:?}"
    );
}

#[test]
fn write_and_edit_json_summaries_compact_absolute_paths() {
    let theme = dark();
    let write = r#"write_file({"file_path":"/Users/joe/2026/zo/crates/api/src/lib.rs","content":"hello"})"#;
    let write_text = collect_text(&rendered_lines(
        "write_file",
        write,
        &generic_preview("write_file"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    ));
    assert!(
        write_text.contains("crates/api/src/lib.rs · 5 bytes"),
        "write path should compact: {write_text:?}"
    );
    assert!(
        !write_text.contains("/Users/joe"),
        "write row should hide absolute prefix: {write_text:?}"
    );

    let edit = r#"edit_file({"file_path":"/Users/joe/2026/zo/crates/api/src/lib.rs","old_string":"a","new_string":"b"})"#;
    let edit_text = collect_text(&rendered_lines(
        "edit_file",
        edit,
        &generic_preview("edit_file"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    ));
    assert!(
        edit_text.contains("crates/api/src/lib.rs · edit"),
        "edit path should compact: {edit_text:?}"
    );
    assert!(
        !edit_text.contains("/Users/joe"),
        "edit row should hide absolute prefix: {edit_text:?}"
    );
}

#[test]
fn activity_summary_for_bash_hides_raw_json() {
    let summary = r#"{"command":"cargo test -p zo-cli"}"#;
    let text = activity_summary("bash", summary);
    assert_eq!(text, "Running command: cargo test -p zo-cli");
    assert!(
        !text.contains('{') && !text.contains("\"command\""),
        "activity line must not leak raw JSON: {text:?}"
    );
}

#[test]
fn activity_summary_for_bash_read_uses_file_verb() {
    let summary = r#"{"command":"nl -ba crates/foo.rs | sed -n '1,80p'"}"#;
    let text = activity_summary("bash", summary);
    assert_eq!(text, "Reading file: crates/foo.rs");
}

#[test]
fn context7_call_renders_as_docs_work_not_raw_tool_payload() {
    let theme = dark();
    let summary = r#"{"libraryId":"/websites/developers_openai_api","query":"Chat Completions API providers"}"#;
    let lines = rendered_lines(
        "mcp__context7__query-docs",
        summary,
        &generic_preview("query-docs"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("Checked docs Chat Completions API providers"),
        "MCP docs call should explain the work: {text:?}"
    );
    assert!(
        text.contains("context7"),
        "source label should remain: {text:?}"
    );
    assert!(
        !text.contains("Called") && !text.contains("libraryId") && !text.contains('{'),
        "raw MCP payload should stay hidden: {text:?}"
    );
}

#[test]
fn context7_activity_summary_explains_current_work() {
    let summary =
        r#"{"libraryId":"/websites/developers_openai_api","query":"Responses API streaming"}"#;
    let text = activity_summary("mcp__context7__query-docs", summary);

    assert_eq!(text, "Checking docs: Responses API streaming");
}

#[test]
fn activity_summary_hides_delegating_detail_duplicate() {
    let text = activity_summary("SpawnMultiAgent", "delegating");
    assert_eq!(text, "Delegating");
}

#[test]
fn activity_summary_falls_back_when_payload_is_partial() {
    let text = activity_summary("bash", r#"{"command":"cargo test""#);
    assert_eq!(text, "Writing command");
    assert!(
        !text.contains('{') && !text.contains("\"command\""),
        "partial payload must not leak raw JSON: {text:?}"
    );
}

#[test]
fn display_summary_handles_escaped_json_payload() {
    let theme = dark();
    let summary = r#""{\"command\":\"cargo check\"}""#;
    let lines = rendered_lines(
        "bash",
        summary,
        &generic_preview("bash"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("cargo check") && !text.contains("$ "),
        "escaped JSON payload should still render compact command: {text:?}"
    );
}

#[test]
fn edit_tool_summary_hides_replacement_payload() {
    let theme = dark();
    let summary = r#"edit_file({"file_path":"src/lib.rs","old_string":"fn noisy_old() {\n    println!(\"old\");\n}","new_string":"fn noisy_new() {\n    println!(\"new\");\n}","replace_all":false})"#;
    let lines = rendered_lines(
        "edit_file",
        summary,
        &generic_preview("edit_file"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("src/lib.rs · edit"),
        "edit row must keep a compact path summary: {text:?}"
    );
    assert!(
        !text.contains("old_string")
            && !text.contains("new_string")
            && !text.contains("noisy_old")
            && !text.contains("noisy_new"),
        "edit row must not render replacement payload: {text:?}"
    );
}

#[test]
fn write_tool_summary_hides_file_content_payload() {
    let theme = dark();
    let summary = r#"write_file({"file_path":"src/lib.rs","content":"line 1\nline 2\nline 3"})"#;
    let lines = rendered_lines(
        "write_file",
        summary,
        &generic_preview("write_file"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("src/lib.rs · 20 bytes"),
        "write row must show path and byte count: {text:?}"
    );
    assert!(
        !text.contains("content") && !text.contains("line 1"),
        "write row must not render file content payload: {text:?}"
    );
}

#[test]
fn malformed_grep_summary_does_not_leak_parse_error_or_commentary() {
    let theme = dark();
    let summary = r#"grep_search(/providerktcloud|providerncp|func \(runner localScanRunner\)|case "} Wait tool JSON maybe okay. io error: regex parse error: x error)"#;
    let lines = rendered_lines(
        "grep_search",
        summary,
        &generic_preview("grep_search"),
        ToolCallStatus::Errored,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("Explored grep providerktcloud"),
        "grep summary should stay useful: {text:?}"
    );
    assert!(!text.contains("grep_search("), "raw call leaked: {text:?}");
    assert!(
        !text.contains("Wait tool JSON"),
        "commentary leaked: {text:?}"
    );
    assert!(
        !text.contains("regex parse error"),
        "parse error leaked into call row: {text:?}"
    );
}

#[test]
fn grep_summary_compacts_long_alternation_patterns() {
    let theme = dark();
    let lines = rendered_lines(
        "Grep",
        "",
        &ToolPreview::Grep {
            pattern: "load_session_preferences|save_session_preferences|DEFAULT_EFFORT|effort_from_preferences|fn set_model".to_string(),
            path: Some("crates/zo-cli/src/session/live_cli.rs".to_string()),
        },
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("grep load_session_preferences +4 more"),
        "grep alternation should compact instead of stretching the row: {text:?}"
    );
    assert!(
        !text.contains("save_session_preferences|DEFAULT_EFFORT"),
        "raw alternation should stay hidden: {text:?}"
    );
}

#[test]
fn grep_summary_compacts_absolute_scope_path() {
    let theme = dark();
    let lines = rendered_lines(
        "Grep",
        "",
        &ToolPreview::Grep {
            pattern: "ToolCallStatus".to_string(),
            path: Some(
                "/Users/joe/2026/zo/crates/zo-cli/src/tui/blocks/tool_call.rs"
                    .to_string(),
            ),
        },
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);

    assert!(
        text.contains("in crates/zo-cli/src/tui/blocks"),
        "grep scope should keep useful repo path context: {text:?}"
    );
    assert!(
        !text.contains("/Users/joe"),
        "grep row should hide absolute prefix: {text:?}"
    );
}

#[test]
fn activity_truncation_counts_korean_by_cell_width() {
    let text = truncate_activity("한글파일이름이길다", 7);

    assert!(
        text.ends_with('\u{2026}'),
        "truncated text keeps ellipsis: {text}"
    );
    let width = text
        .chars()
        .map(|ch| UnicodeWidthChar::width(ch).unwrap_or(0))
        .sum::<usize>();
    assert!(width <= 7, "truncated text must fit cell width: {text}");
}

#[test]
fn smart_command_truncation_handles_korean_arguments() {
    let command = concat!(
        "printf ",
        "살펴보기|도움받기|자동맡기기|보기|옆에서|맡김 ",
        "another-super-long-argument-that-forces-overflow"
    );
    let text = smart_truncate_command(command, BASH_LABEL_CAP);

    assert!(
        text.contains("..."),
        "long arguments should be shortened: {text}"
    );
    assert!(
        text.contains("맡김"),
        "truncation should keep the tail: {text}"
    );
    assert!(
        display_width(&text) <= BASH_LABEL_CAP,
        "command summary must fit the display cap: {text}"
    );
}

/// 완료된(Ok) 도구는 인자가 없어도 액션 문구를 표시하지 않는다 —
/// 진행 중이 아니므로 noise 가 되면 안 된다.
#[test]
fn settled_tool_without_args_omits_action_phrase() {
    let theme = dark();
    let preview = generic_preview("Grep");
    let lines = rendered_lines(
        "Grep",
        "",
        &preview,
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
    );
    let text = collect_text(&lines);
    assert!(
        !text.contains("Searching content"),
        "settled tool must not show in-flight action phrase: {text:?}"
    );
}

/// `NO_COLOR`/`TERM=dumb`: the status markers (`○●✓×⊘`) degrade to their
/// 1-cell ASCII siblings via `glyphs::pick`, so no Unicode marker survives to
/// render as tofu on a non-Nerd-Font terminal.
#[test]
fn no_color_status_markers_degrade_to_ascii_siblings() {
    let theme = no_color();
    let preview = bash_preview("cargo test");
    let cases = [
        (ToolCallStatus::Pending, "o Ran cargo test", '\u{25cb}'),
        // Running rows animate; at tick 0 the ASCII spinner shows `-` (the
        // braille frames stay forbidden under NO_COLOR like every rich glyph).
        (ToolCallStatus::Running, "- Ran cargo test", '\u{25cf}'),
        (ToolCallStatus::Ok, "v Ran cargo test", '\u{2713}'),
        (ToolCallStatus::Errored, "x Ran cargo test", '\u{00d7}'),
        (ToolCallStatus::Cancelled, "/ Ran cargo test", '\u{2298}'),
    ];
    for (status, expected, rich) in cases {
        let lines = rendered_lines("bash", "", &preview, status, &theme, 0, false, None);
        let text = collect_text(&lines);
        assert!(
            text.starts_with(expected),
            "{status:?} should use ASCII marker under NO_COLOR: {text:?}"
        );
        assert!(
            !text.contains(rich),
            "{status:?} must not paint rich marker {rich:?} under NO_COLOR: {text:?}"
        );
    }
}

/// `NO_COLOR`: the agent-tree box glyphs (`├└│⎿`) degrade to ASCII
/// (`+`/`` ` ``/`|`/`+`), and the finished-batch header's `✓` becomes `v`.
#[test]
fn no_color_agent_tree_glyphs_degrade_to_ascii() {
    let theme = no_color();
    let tree = AgentTree {
        rows: vec![
            tree_row("wiring", "completed", 53, 95_100),
            tree_row("deep-gate", "running", 31, 9_000),
        ],
        batch_label: None,
        finished: false,
    };
    // Dedicated detail surfaces still render the complete tree and preserve
    // their ASCII fallbacks.
    let lines = agent_tree_lines(&tree, &theme);
    let text = collect_text(&lines);
    // Branch (├), elbow (└), stem (│), and the ⎿ completion hook are ASCII.
    // Each row's status dot also degrades: ✓ (completed) → 'v', ● (running) → '*'.
    assert!(text.contains("+ v wiring"), "branch '+' + completed dot 'v': {text:?}");
    assert!(text.contains("` * deep-gate"), "elbow '`' + running dot '*': {text:?}");
    assert!(text.contains("| +  Done"), "stem '|' + hook '+': {text:?}");
    for rich in ['\u{251c}', '\u{2514}', '\u{2502}', '\u{23bf}'] {
        assert!(
            !text.contains(rich),
            "no rich box glyph {rich:?} may survive NO_COLOR: {text:?}"
        );
    }

    // A fully finished batch swaps in the `✓` header; under NO_COLOR it is `v`.
    let finished = AgentTree {
        rows: vec![tree_row("wiring", "completed", 53, 95_100)],
        batch_label: None,
        finished: true,
    };
    let finished_text = collect_text(&rendered_lines_with_tree(
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        None,
        Some(&finished),
    ));
    assert!(
        finished_text.starts_with("v "),
        "finished header ✓ → 'v' under NO_COLOR: {finished_text:?}"
    );
    assert!(
        !finished_text.contains('\u{2713}'),
        "no rich ✓ in NO_COLOR finished header: {finished_text:?}"
    );
}

#[test]
fn verb_color_uses_brightness_steps_not_category_hues() {
    let theme = dark();
    let p = &theme.palette;
    // The consequential mutate band pops via brightness, not a hue.
    assert_eq!(verb_color("Wrote", p), p.bright);
    assert_eq!(verb_color("Edited", p), p.bright);
    // Plan / state bookkeeping recedes.
    assert_eq!(verb_color("Updated", p), p.dim);
    // Read / run / search / delegate / unknown all sit at the body fg — the
    // verb word, not a color band, distinguishes them (noise reduction).
    for verb in ["Ran", "Explored", "Searched", "Spawned", "Called", "Whatever"] {
        assert_eq!(verb_color(verb, p), p.fg, "{verb} recedes to fg");
    }
    // Chromatic ink is reserved for semantic status: no verb may spend a hue.
    for verb in [
        "Ran", "Explored", "Searched", "Wrote", "Edited", "Spawned", "Updated", "Called",
    ] {
        let c = verb_color(verb, p);
        for hue in [p.cyan, p.violet, p.teal, p.info, p.warn, p.error, p.success] {
            assert_ne!(c, hue, "{verb} must not use a chromatic status hue");
        }
    }
}

/// 가독성: per-agent 행에서 (1) 에이전트 이름은 주변 dim 메타와 구분되도록
/// BOLD 로 강조되고, (2) 종결 행의 라벨(`Done`/`Failed`/`Stopped`)은 BOLD +
/// 결과 색상이며 (3) `⎿` 훅이 라벨과 같은 색 밴드를 공유한다. 평평한 한 색
/// 벽이었던 트리에서 이름과 결과가 한눈에 스캔되도록 하는 회귀 방지 테스트.
#[test]
fn agent_tree_emphasizes_name_and_color_bands_the_outcome() {
    let theme = dark();
    let p = &theme.palette;
    let tree = AgentTree {
        rows: vec![
            tree_row("wiring", "completed", 53, 95_100),
            tree_row("gate", "failed", 12, 4_000),
            tree_row("probe", "stopped", 3, 100),
            tree_row("live", "running", 7, 2_000),
        ],
        batch_label: None,
        finished: true,
    };
    // This is the dedicated-detail renderer used by the pinned panel/sidebar;
    // the primary transcript intentionally shows only an aggregate line.
    let lines = agent_tree_lines(&tree, &theme);

    // Helper: find the first span whose trimmed content equals `needle`.
    let span_for = |needle: &str| {
        lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .find(|s| s.content.trim() == needle)
            .unwrap_or_else(|| panic!("span {needle:?} not found"))
            .clone()
    };

    // (0) Every row — running ones too, not just the terminal `⎿` line — leads
    // with a colored status dot, so an in-flight batch shows a live color band
    // (teal running / green done / red failed / amber stopped) instead of the
    // old flat-grey-until-finished wall. `agent_row_dot_colors` scopes the count
    // to the per-agent rows so the finished-batch header (also `✓`) is excluded.
    let dot_colors = agent_row_dot_colors(&lines);
    assert_eq!(dot_colors.len(), 4, "one status dot per agent row: {dot_colors:?}");
    assert!(
        dot_colors.contains(&Some(p.teal)),
        "the running row's dot is teal — the live state now carries color: {dot_colors:?}"
    );
    assert!(
        [p.success, p.error, p.warn, p.teal]
            .iter()
            .all(|hue| dot_colors.contains(&Some(*hue))),
        "running/done/failed/stopped each get a distinct status hue: {dot_colors:?}"
    );

    // (1) Each agent name carries BOLD weight (survives NO_COLOR).
    for name in ["wiring", "gate", "probe"] {
        let span = span_for(name);
        assert!(
            span.style.add_modifier.contains(Modifier::BOLD),
            "agent name {name:?} must be bold: {:?}",
            span.style
        );
    }

    // (2) Each outcome label is BOLD and tinted to its result hue.
    for (label, hue) in [
        ("Done", p.success),
        ("Failed", p.error),
        ("Stopped", p.warn),
    ] {
        let span = span_for(label);
        assert!(
            span.style.add_modifier.contains(Modifier::BOLD),
            "outcome {label:?} must be bold: {:?}",
            span.style
        );
        assert_eq!(
            span.style.fg,
            Some(hue),
            "outcome {label:?} must carry its result hue"
        );
    }

    // (3) The `⎿` hook shares the outcome's color band (not flat dim). There is
    // one hook per terminal row; every hook must be tinted to a result hue, and
    // at least one must be the failed-red so we know it is not all `success`.
    let hook_colors: Vec<_> = lines
        .iter()
        .flat_map(|l| l.spans.iter())
        .filter(|s| s.content.trim_start().starts_with('\u{23bf}'))
        .map(|s| s.style.fg)
        .collect();
    assert_eq!(hook_colors.len(), 3, "one hook per terminal row");
    assert!(
        hook_colors.contains(&Some(p.error)),
        "a failed row's hook must be red, not flat dim: {hook_colors:?}"
    );
    assert!(
        hook_colors.iter().all(|c| matches!(
            *c,
            Some(c) if c == p.success || c == p.error || c == p.warn
        )),
        "every hook shares its outcome color band: {hook_colors:?}"
    );
}

// ---------------------------------------------------------------------------
// In-flight liveness: the running marker spins and a long-silent call names
// its wait. Motivating case: a hung MCP request ran 52 minutes behind a
// static `●` and users could not tell "working" from "dead", so they
// cancelled healthy turns.
// ---------------------------------------------------------------------------

fn first_marker(lines: &[Line<'static>]) -> String {
    lines
        .first()
        .and_then(|l| l.spans.first())
        .map(|s| s.content.trim().to_string())
        .unwrap_or_default()
}

#[test]
fn running_tool_marker_animates_across_ticks() {
    let theme = dark();
    let render = |tick: u64| {
        rendered_lines(
            "mcp__atlassian__search",
            "query",
            &generic_preview("mcp__atlassian__search"),
            ToolCallStatus::Running,
            &theme,
            tick,
            false,
            Some(std::time::Duration::from_secs(5)),
        )
    };

    // The spark heartbeat holds each phase for ~500ms (≈15 ticks on the 33ms
    // clock), so sample across a phase boundary instead of adjacent ticks.
    let frame_a = first_marker(&render(0));
    let frame_b = first_marker(&render(16));

    assert_ne!(
        frame_a, frame_b,
        "the in-flight marker must visibly move between animation ticks"
    );
}

#[test]
fn settled_tool_marker_stays_static() {
    let theme = dark();
    let render = |tick: u64| {
        rendered_lines(
            "bash",
            "cargo test",
            &generic_preview("bash"),
            ToolCallStatus::Ok,
            &theme,
            tick,
            false,
            None,
        )
    };

    assert_eq!(
        first_marker(&render(0)),
        first_marker(&render(4)),
        "settled rows must not strobe"
    );
}

#[test]
fn long_running_mcp_call_names_its_wait_and_the_interrupt_key() {
    let theme = dark();
    let lines = rendered_lines(
        "mcp__atlassian__search",
        "query",
        &generic_preview("mcp__atlassian__search"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        Some(std::time::Duration::from_secs(LONG_WAIT_HINT_SECS + 1)),
    );

    let text = collect_text(&lines);
    assert!(
        text.contains("waiting on atlassian"),
        "a silent MCP call must name the server it waits on: {text:?}"
    );
    assert!(
        text.contains("esc to interrupt"),
        "the wait hint must carry the interrupt affordance: {text:?}"
    );
}

#[test]
fn long_running_local_tool_shows_generic_wait_hint() {
    let theme = dark();
    let lines = rendered_lines(
        "bash",
        "cargo build",
        &generic_preview("bash"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        Some(std::time::Duration::from_secs(LONG_WAIT_HINT_SECS)),
    );

    let text = collect_text(&lines);
    assert!(
        text.contains("still waiting") && text.contains("esc to interrupt"),
        "a long local call still explains itself: {text:?}"
    );
}

#[test]
fn short_running_tool_has_no_wait_hint() {
    let theme = dark();
    let lines = rendered_lines(
        "mcp__atlassian__search",
        "query",
        &generic_preview("mcp__atlassian__search"),
        ToolCallStatus::Running,
        &theme,
        0,
        false,
        Some(std::time::Duration::from_secs(5)),
    );

    let text = collect_text(&lines);
    assert!(
        !text.contains("waiting"),
        "ordinary short calls must not nag: {text:?}"
    );
}

#[test]
fn settled_tool_has_no_wait_hint_even_with_long_elapsed() {
    let theme = dark();
    let lines = rendered_lines(
        "mcp__atlassian__search",
        "query",
        &generic_preview("mcp__atlassian__search"),
        ToolCallStatus::Ok,
        &theme,
        0,
        false,
        Some(std::time::Duration::from_secs(600)),
    );

    let text = collect_text(&lines);
    assert!(
        !text.contains("waiting"),
        "a completed call must drop the wait hint: {text:?}"
    );
}

/// 인라인 우선: 라이브 트리를 가진 spawn 행은 언제나 진짜 트리를 그린다 —
/// `spawning…` placeholder 나 `still waiting` 정지 오경보가 절대 붙지 않는다.
/// 47 tool uses 로 8분째 도는 배치가 "spawning… · still waiting" 으로 멈춰
/// 보이던 모순의 회귀 방지 (지금은 running 헤더 분기가 힌트 경로에 선행한다).
#[test]
fn live_tree_with_rows_never_reads_as_spawning_or_stalled() {
    let tree = AgentTree {
        rows: vec![tree_row("a1-fix-round2", "running", 47, 9_000)],
        batch_label: None,
        finished: false,
    };
    let lines = rendered_lines_with_tree(
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Running,
        &dark(),
        0,
        false,
        Some(Duration::from_secs(8 * 60)),
        Some(&tree),
    );
    let text = collect_text(&lines);
    assert!(
        text.contains("Running 1 Explore agent\u{2026}"),
        "the real live header renders inline: {text}"
    );
    assert!(
        text.contains("1 running · 47 tool uses · 9.0k tokens"),
        "aggregate progress renders inline: {text}"
    );
    assert!(
        !text.contains("a1-fix-round2"),
        "agent name stays in the pinned/Ctrl+G detail surfaces: {text}"
    );
    assert!(
        !text.contains("spawning"),
        "a batch with rows must never read as spawning: {text}"
    );
    assert!(
        !text.contains("still waiting"),
        "a live delegation must never read as a stalled call: {text}"
    );
}

// --- Width-aware clip (agent-tree row wrap / orphan-fragment fix) -----------
// The ToolCall block's lines are single-line-by-design; without a clip pass an
// over-long agent-tree row (long `routed: …` reason, `⤷` output tail) wrapped to
// column 0 under `Paragraph::wrap`, scattering orphan meta fragments and
// breaking the tree's left structure. `clip_lines_to_width` truncates from the
// RIGHT so the left tree glyphs survive and line-count stays == row-count.

/// Per-line display width in cells (spans summed) — the quantity ratatui's wrap
/// consults, so `<= width` proves the line does not wrap.
fn line_cells(line: &Line<'static>) -> usize {
    line.spans
        .iter()
        .map(|s| display_width(s.content.as_ref()))
        .sum()
}

#[test]
fn clip_line_leaves_fitting_line_untouched() {
    let line = Line::from(vec![
        Span::raw("   "),
        Span::raw("├ "),
        Span::raw("● scout"),
    ]);
    let before = line.clone();
    let clipped = clip_line_to_width(line, 40);
    assert_eq!(clipped.spans, before.spans, "a fitting line is byte-identical");
    assert!(
        !collect_text(std::slice::from_ref(&clipped)).contains('\u{2026}'),
        "no ellipsis is added to a fitting line"
    );
}

#[test]
fn clip_line_truncates_from_right_with_ellipsis_and_keeps_left() {
    // A tree row: left structure + a long trailing meta that overflows.
    let line = Line::from(vec![
        Span::raw("   ├ "),
        Span::raw("● correctness-audit"),
        Span::raw(" · routed: Reviewer·Medium — auto role selector with a long tail"),
    ]);
    let clipped = clip_line_to_width(line, 40);
    assert!(line_cells(&clipped) <= 40, "clipped line fits the width");
    let text = collect_text(std::slice::from_ref(&clipped));
    assert!(text.ends_with('\u{2026}'), "truncation is marked with an ellipsis: {text:?}");
    // Left tree structure and the agent name survive; only trailing meta is lost.
    assert!(text.starts_with("   ├ ● correctness-audit"), "left structure preserved: {text:?}");
    assert!(!text.contains("long tail"), "trailing meta is dropped: {text:?}");
}

#[test]
fn clip_line_respects_double_width_cells() {
    // Double-width CJK meta that overflows: the clip must count display cells,
    // never char count, or the clipped line would still exceed the width and wrap.
    let line = Line::from(vec![
        Span::raw("   └ ● 에이전트 "),
        Span::raw("· 한국어 텍스트가 좁은 폭에서 넘치는 아주 긴 메타 세그먼트"),
    ]);
    for width in [12usize, 20, 33, 41] {
        let clipped = clip_line_to_width(line.clone(), width);
        assert!(
            line_cells(&clipped) <= width,
            "CJK clip must fit width {width} in cells, got {}",
            line_cells(&clipped)
        );
    }
}

#[test]
fn agent_tree_rows_clip_to_one_row_each_at_narrow_width() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::widgets::{Paragraph, Wrap};

    // Reproduces the live screenshot: a running agent row whose `routed:` reason
    // (already 40-char truncated) still overflows a 60-col terminal.
    let mut row = tree_row("correctness-audit", "running", 18, 0);
    row.elapsed_secs = 114; // 1m 54s
    row.activity = Some("bash".to_string());
    row.route_reason =
        Some("Reviewer·Medium — auto role selector with a long overflowing tail".to_string());
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let width = 60u16;
    let lines = clip_lines_to_width(
        rendered_lines_with_tree(
            "SpawnMultiAgent",
            "",
            &spawn_preview(),
            ToolCallStatus::Running,
            &dark(),
            0,
            false,
            None,
            Some(&tree),
        ),
        width,
    );
    for (i, line) in lines.iter().enumerate() {
        assert!(
            line_cells(line) <= usize::from(width),
            "clipped design line {i} must fit width {width}, got {}: {line:?}",
            line_cells(line)
        );
    }
    // line-count == row-count: nothing wraps to a phantom extra row.
    assert_eq!(
        usize::from(wrapped_rows(&lines, width)),
        lines.len(),
        "no design line wraps: {lines:#?}"
    );

    // Render and prove no orphan meta fragment lands at column 0 on a tree row:
    // the header owns row 0, every subsequent tree row begins with the 3-space
    // indent, so col 0 for rows >= 1 must be blank.
    let height = u16::try_from(lines.len()).unwrap_or(u16::MAX).max(1);
    let backend = TestBackend::new(width, height);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| {
        f.render_widget(
            Paragraph::new(lines.clone()).wrap(Wrap { trim: false }),
            Rect::new(0, 0, width, height),
        );
    })
    .expect("draw");
    let buf = term.backend().buffer().clone();
    for y in 1..height {
        assert_eq!(
            buf[(0, y)].symbol(),
            " ",
            "row {y} must start with the tree indent, not an orphan fragment"
        );
    }
}

#[test]
fn estimate_rows_counts_one_row_per_clipped_design_line() {
    let mut row = tree_row("audit", "running", 18, 0);
    row.route_reason = Some("auto role selector ".repeat(6));
    let tree = AgentTree {
        rows: vec![row],
        batch_label: None,
        finished: false,
    };
    let width = 60u16;
    let rows = estimate_rows(
        None,
        "SpawnMultiAgent",
        "",
        &spawn_preview(),
        ToolCallStatus::Running,
        &dark(),
        width,
        Some(&tree),
        false,
    );
    let clipped = clip_lines_to_width(
        rendered_lines_with_tree(
            "SpawnMultiAgent",
            "",
            &spawn_preview(),
            ToolCallStatus::Running,
            &dark(),
            0,
            false,
            None,
            Some(&tree),
        ),
        width,
    );
    assert_eq!(
        usize::from(rows),
        clipped.len(),
        "measured height must equal the clipped design-line count"
    );
}
