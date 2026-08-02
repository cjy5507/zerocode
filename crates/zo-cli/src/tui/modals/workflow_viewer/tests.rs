use super::*;
use crate::tui::hud::{TodoChecklistItem, TodoChecklistStatus};
use crate::tui::theme::Theme;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;

fn press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    }
}

fn ctrl_press(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    }
}

fn agent(name: &str, status: &str, tool: Option<&str>) -> WorkflowAgentRow {
    WorkflowAgentRow {
        id: format!("{name}-id"),
        name: name.to_string(),
        description: format!("Inspect {name} and report findings"),
        subagent_type: Some("analysis".to_string()),
        status: status.to_string(),
        current_tool: tool.map(str::to_string),
        model: "openai:gpt-5.5-fast".to_string(),
        tool_calls: Some(22),
        tokens: 103_000,
        elapsed_secs: 175,
        output_file: Some(format!("/tmp/{name}.md")),
        last_event: Some(format!("lane.finished: {name} done")),
        ..WorkflowAgentRow::default()
    }
}

fn plan_step(
    step_id: &str,
    content: &str,
    active_form: &str,
    status: TodoChecklistStatus,
) -> TodoChecklistItem {
    TodoChecklistItem {
        step_id: Some(step_id.to_string()),
        content: content.to_string(),
        status,
        active_form: active_form.to_string(),
    }
}

fn sample() -> WorkflowView {
    WorkflowView {
        run_id: "test-run".to_string(),
        name: "zo-workflow-live-viz-analysis".to_string(),
        description: "동적 워크플로우 라이브 시각화".to_string(),
        status: "running".to_string(),
        mode: "phases".to_string(),
        synthesizing: false,
        phases: vec![
            WorkflowPhaseRow {
                step_id: Some("read".to_string()),
                plan_step: Some(plan_step(
                    "read",
                    "Inspect code paths",
                    "Inspecting code paths",
                    TodoChecklistStatus::InProgress,
                )),
                id: "read".to_string(),
                kind: "fanout".to_string(),
                status: "running".to_string(),
                round: 1,
                completed: 5,
                failed: 0,
                still_running: 1,
                total: 6,
                agents: vec![
                    agent("read:engine", "completed", None),
                    agent("read:dispatch", "running", Some("read_file")),
                ],
            },
            WorkflowPhaseRow {
                step_id: Some("synthesize".to_string()),
                plan_step: Some(plan_step(
                    "synthesize",
                    "Synthesize findings",
                    "Synthesizing findings",
                    TodoChecklistStatus::Pending,
                )),
                id: "synthesize".to_string(),
                kind: "single".to_string(),
                status: "pending".to_string(),
                round: 0,
                completed: 0,
                failed: 0,
                still_running: 0,
                total: 0,
                agents: vec![],
            },
        ],
    }
}

#[test]
fn empty_view_reports_empty() {
    assert!(WorkflowViewerModal::new(WorkflowView::default()).is_empty());
    assert!(!WorkflowViewerModal::new(sample()).is_empty());
}

/// **클릭 회귀 핀** — Ctrl+O 뷰어는 `Down(Left)` 를 통째로 버려서 클릭이
/// 100% 무시됐다(사용자 불만: "스트리밍창 접었다 펼치기가 클릭으로 안 됨").
/// 좌표는 **렌더 버퍼에서 찾아온다**: 하드코딩하면 레이아웃과 히트테스트가
/// 갈라져도 테스트가 통과해버린다.
mod clicks {
    use super::*;

    /// `(y, x)` of the first cell of `needle` on screen.
    fn at(buf: &ratatui::buffer::Buffer, needle: &str) -> (u16, u16) {
        find_row(buf, needle)
    }

    fn dump_of(buf: &ratatui::buffer::Buffer) -> String {
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn clicking_an_executor_row_selects_it() {
        let theme = Theme::zo();
        let area = Rect::new(0, 0, 160, 40);
        let mut modal = WorkflowViewerModal::new(sample());
        assert_eq!(modal.selected_agent(), 1, "precondition: live row focused");

        let buf = buffer_dump(&modal, &theme, area.width, area.height);
        let (y, x) = at(&buf, "read:engine");
        assert!(
            modal.handle_click(x, y, area, &theme),
            "a click on an executor row must land"
        );
        assert_eq!(modal.selected_agent(), 0, "the clicked row is selected");
        assert!(
            !modal.handle_click(x, y, area, &theme),
            "re-clicking the selected row changes nothing (no spurious redraw)"
        );
    }

    #[test]
    fn clicking_a_plan_step_selects_it() {
        let theme = Theme::zo();
        let area = Rect::new(0, 0, 160, 40);
        let mut modal = WorkflowViewerModal::new(sample());
        let buf = buffer_dump(&modal, &theme, area.width, area.height);
        // The rail row, not the header's `next Synthesize findings` echo.
        let (y, x) = at(&buf, "\u{25cb} Synthesize findings");
        assert!(modal.handle_click(x, y, area, &theme));
        assert_eq!(modal.selected_phase(), 1);
    }

    /// 섹션 헤딩 클릭 = 접기/펼치기, 접힌 섹션은 한 줄 요약으로 축약된다.
    #[test]
    fn clicking_a_section_heading_folds_and_summarizes_it() {
        let theme = Theme::zo();
        let area = Rect::new(0, 0, 160, 40);
        let mut view = sample();
        view.phases[0].agents[1].recent_tools = vec![
            "read_file · src/main.rs".to_string(),
            "bash · cargo test".to_string(),
        ];
        let mut modal = WorkflowViewerModal::new(view);

        let buf = buffer_dump(&modal, &theme, area.width, area.height);
        assert!(dump_of(&buf).contains("cargo test"), "precondition: feed open");
        let (y, x) = at(&buf, "ACTIVITY");
        assert!(
            modal.handle_click(x, y, area, &theme),
            "a click on a section heading must land"
        );

        let folded = buffer_dump(&modal, &theme, area.width, area.height);
        let dump = dump_of(&folded);
        assert!(
            dump.contains("ACTIVITY \u{00b7} 2 lines"),
            "a folded section summarizes what it hides: {dump}"
        );
        assert!(!dump.contains("cargo test"), "feed rows are gone: {dump}");

        // The fold is modal state: it survives moving to another executor.
        modal.handle_key(press(KeyCode::Up));
        let other = dump_of(&buffer_dump(&modal, &theme, area.width, area.height));
        assert!(
            !other.contains("\u{25be} ACTIVITY"),
            "the fold follows the reader, not the agent: {other}"
        );

        modal.handle_key(press(KeyCode::Down));
        let (fy, fx) = at(
            &buffer_dump(&modal, &theme, area.width, area.height),
            "ACTIVITY",
        );
        assert!(modal.handle_click(fx, fy, area, &theme));
        let reopened = dump_of(&buffer_dump(&modal, &theme, area.width, area.height));
        assert!(reopened.contains("cargo test"), "{reopened}");
    }

    /// 패널 바깥(헤더·푸터)은 no-op — 클릭이 엉뚱한 선택을 만들지 않는다.
    #[test]
    fn clicks_outside_the_panes_do_nothing() {
        let theme = Theme::zo();
        let area = Rect::new(0, 0, 160, 40);
        let mut modal = WorkflowViewerModal::new(sample());
        assert!(!modal.handle_click(0, 0, area, &theme), "the title rule");
        assert!(
            !modal.handle_click(area.width / 2, area.height - 1, area, &theme),
            "the footer"
        );
        assert_eq!(modal.selected_agent(), 1);
        assert_eq!(modal.selected_phase(), 0);
    }
}

#[test]
fn plan_items_join_only_by_one_exact_real_workflow_step_id() {
    let mut view = sample();
    for phase in &mut view.phases {
        phase.plan_step = None;
    }
    let mut modal = WorkflowViewerModal::new(view);
    let replacement = plan_step(
        "read",
        "Read exact workflow state",
        "Reading exact workflow state",
        TodoChecklistStatus::InProgress,
    );
    modal.attach_plan_items(std::slice::from_ref(&replacement));
    assert_eq!(
        modal.view.phases[0]
            .plan_step
            .as_ref()
            .map(|step| step.content.as_str()),
        Some("Read exact workflow state")
    );

    modal.attach_plan_items(&[replacement.clone(), replacement]);
    assert!(
        modal.view.phases[0].plan_step.is_none(),
        "duplicate Todo ids must remain unlinked"
    );
    let rendered = dump(&modal, 140, 30);
    assert!(rendered.contains("Plan link unavailable"), "{rendered}");
    assert!(!rendered.contains("Plan linked"), "{rendered}");

    let mut duplicated_phase = sample();
    duplicated_phase.phases.push(duplicated_phase.phases[0].clone());
    for phase in &mut duplicated_phase.phases {
        phase.plan_step = None;
    }
    let mut duplicated_modal = WorkflowViewerModal::new(duplicated_phase);
    duplicated_modal.attach_plan_items(&[plan_step(
        "read",
        "Ambiguous phase",
        "Reading ambiguous phase",
        TodoChecklistStatus::InProgress,
    )]);
    assert!(
        duplicated_modal
            .view
            .phases
            .iter()
            .filter(|phase| phase.step_id.as_deref() == Some("read"))
            .all(|phase| phase.plan_step.is_none()),
        "duplicate phase ids must remain unlinked"
    );
}

#[test]
fn partial_plan_join_reports_global_and_selected_scope_consistently() {
    let mut view = sample();
    view.phases[1].plan_step = None;
    let mut modal = WorkflowViewerModal::new(view);

    let linked = dump(&modal, 140, 30);
    assert!(linked.contains("1/2 phases Plan linked"), "{linked}");
    assert!(
        linked.contains("Inspecting code paths"),
        "the Plan-linked phase keeps its Todo label in the tree: {linked}"
    );

    modal.handle_key(press(KeyCode::Right));
    let unlinked = dump(&modal, 140, 30);
    // The group that failed to join is marked where the reader is looking —
    // beside its own label — not only in the run-level tally above.
    let group = unlinked
        .lines()
        .find(|line| line.contains("synthesize"))
        .unwrap_or_else(|| panic!("the unlinked phase heads its own group:\n{unlinked}"));
    assert!(
        group.contains("unlinked"),
        "a Plan-unlinked group says so on its own row: {group:?}"
    );
    // It has started no executor, so the inspector has nothing to inspect and
    // says so instead of framing an empty card.
    assert!(unlinked.contains("select an executor"), "{unlinked}");
}

#[test]
fn synthetic_fanout_is_visibly_unlinked_even_with_a_same_named_todo() {
    let mut view = sample();
    view.run_id.clear();
    view.name = "agents".to_string();
    view.description = "2 spawned agents".to_string();
    view.phases.truncate(1);
    view.phases[0].id = "agents".to_string();
    view.phases[0].step_id = None;
    view.phases[0].plan_step = None;
    let mut modal = WorkflowViewerModal::new(view);
    modal.attach_plan_items(&[plan_step(
        "agents",
        "This must not be claimed",
        "Claiming the wrong step",
        TodoChecklistStatus::InProgress,
    )]);

    assert!(modal.view.phases[0].plan_step.is_none());
    let rendered = dump(&modal, 140, 30);
    assert!(rendered.contains("not linked to a Plan step"), "{rendered}");
    // A single-phase run folds its rail into the summary row (defect D), so the
    // ` Run scope · unlinked ` pane title is gone; the scope is stated by the
    // folded header and by the header's own warning instead.
    assert!(rendered.contains("Run-level fan-out"), "{rendered}");
    assert!(rendered.contains("not linked to a Plan step"), "{rendered}");
    assert!(!rendered.contains("This must not be claimed"), "{rendered}");
    assert!(!rendered.contains("^E events"), "{rendered}");
    modal.handle_key(ctrl_press(KeyCode::Char('e')));
    assert!(
        !modal.events_mode,
        "synthetic fan-out has no run event log to open"
    );
}

#[test]
fn same_run_refresh_keeps_joined_plan_labels_after_todos_clear() {
    let mut modal = WorkflowViewerModal::new(sample());
    let mut refreshed = terminal_sample();
    for phase in &mut refreshed.phases {
        phase.plan_step = None;
    }

    modal.refresh(refreshed, &[]);

    assert_eq!(
        modal.view.phases[0]
            .plan_step
            .as_ref()
            .map(|step| step.content.as_str()),
        Some("Inspect code paths")
    );
    assert_eq!(
        modal.view.phases[0]
            .plan_step
            .as_ref()
            .map(|step| step.status),
        Some(TodoChecklistStatus::Completed),
        "terminal preservation must stop showing the active-form label"
    );
    let rendered = dump(&modal, 140, 30);
    assert!(rendered.contains("Inspect code paths"), "{rendered}");
    assert!(!rendered.contains("Inspecting code paths"), "{rendered}");
}

#[test]
fn same_run_refresh_preserves_only_omitted_unique_plan_steps() {
    let mut modal = WorkflowViewerModal::new(sample());
    let mut refreshed = terminal_sample();
    for phase in &mut refreshed.phases {
        phase.plan_step = None;
    }
    let remaining = plan_step(
        "synthesize",
        "Synthesize revised findings",
        "Synthesizing revised findings",
        TodoChecklistStatus::Completed,
    );
    modal.refresh(refreshed, std::slice::from_ref(&remaining));

    assert_eq!(
        modal.view.phases[0]
            .plan_step
            .as_ref()
            .map(|step| step.content.as_str()),
        Some("Inspect code paths"),
        "a dropped completed row keeps its last exact human label"
    );
    assert_eq!(
        modal.view.phases[1]
            .plan_step
            .as_ref()
            .map(|step| step.content.as_str()),
        Some("Synthesize revised findings"),
        "the current exact Todo snapshot remains authoritative"
    );
}

#[test]
fn refresh_does_not_preserve_through_new_phase_or_todo_ambiguity() {
    let mut duplicate_phase_modal = WorkflowViewerModal::new(sample());
    let mut duplicate_phase_view = sample();
    duplicate_phase_view
        .phases
        .push(duplicate_phase_view.phases[0].clone());
    for phase in &mut duplicate_phase_view.phases {
        phase.plan_step = None;
    }
    duplicate_phase_modal.refresh(duplicate_phase_view, &[]);
    assert!(
        duplicate_phase_modal
            .view
            .phases
            .iter()
            .filter(|phase| phase.step_id.as_deref() == Some("read"))
            .all(|phase| phase.plan_step.is_none()),
        "a new duplicate phase id must not inherit the old exact label"
    );

    let mut duplicate_todo_modal = WorkflowViewerModal::new(sample());
    let mut refreshed = sample();
    for phase in &mut refreshed.phases {
        phase.plan_step = None;
    }
    let duplicate = plan_step(
        "read",
        "Ambiguous",
        "Reading ambiguous",
        TodoChecklistStatus::InProgress,
    );
    duplicate_todo_modal.refresh(refreshed, &[duplicate.clone(), duplicate]);
    assert!(
        duplicate_todo_modal.view.phases[0].plan_step.is_none(),
        "a duplicate current Todo id must not fall back to a stale label"
    );
}

#[test]
fn select_agent_by_id_focuses_the_matching_row_or_leaves_default() {
    // Hit: the second agent of phase 0 → cursor moves to (phase 0, agent 1).
    let mut modal = WorkflowViewerModal::new(sample());
    assert!(modal.select_agent_by_id("read:dispatch-id"));
    assert_eq!(modal.selected_phase(), 0);
    assert_eq!(modal.selected_agent(), 1);

    // Miss: an unknown id leaves the initial active-executor selection intact
    // and reports false rather than focusing a wrong row.
    let mut modal = WorkflowViewerModal::new(sample());
    assert!(!modal.select_agent_by_id("does-not-exist"));
    assert_eq!(modal.selected_phase(), 0);
    assert_eq!(modal.selected_agent(), 1);
}

#[test]
fn details_render_recent_tool_activity_feed() {
    let mut view = sample();
    view.phases[0].agents[0].recent_tools = vec![
        "read_file \u{00b7} src/main.rs".to_string(),
        "bash \u{00b7} cargo test -p tools".to_string(),
    ];
    let mut modal = WorkflowViewerModal::new(view);
    assert!(modal.select_agent_by_id("read:engine-id"));
    let dumped = dump(&modal, 160, 40);
    // Section headings are uppercase inlays (`─ ACTIVITY ────`), the sidebar's
    // `section_heading_span` idiom — case is what tells a *section* of the card
    // apart from a lowercase field label inside it.
    assert!(dumped.contains("ACTIVITY"), "activity section: {dumped}");
    assert!(
        dumped.contains("read_file") && dumped.contains("cargo test -p tools"),
        "feed entries must render: {dumped}"
    );
}

/// P7 관측성: `route_reason` (manifest `routeReason`) 은 detail 카드의
/// `model` 행 바로 아래 `route` 라벨로 렌더된다 — `TestBackend` ASCII
/// 렌더덤프로 실제 화면 버퍼를 검증한다(프로젝트 관례).
#[test]
fn details_render_route_reason() {
    let mut view = sample();
    view.phases[0].agents[0].route_reason =
        Some("auto:coding tier=strong · learned-shadow-differs:gpt-5.6-sol".to_string());
    let mut modal = WorkflowViewerModal::new(view);
    assert!(modal.select_agent_by_id("read:engine-id"));
    let dumped = dump(&modal, 160, 40);
    assert!(dumped.contains("route"), "route label rendered: {dumped}");
    // Detail values are now clipped to their column (defect B) — a 60-cell
    // route reason is wider than any detail column the modal ever gets, so the
    // assertion pins the part that identifies the decision, not the tail.
    assert!(
        dumped.contains("auto:coding tier=strong · learned-shadow"),
        "route reason text rendered: {dumped}"
    );
}

/// 라우팅 사유가 없는(기본값·명시 모델) 에이전트는 `route` 행 자체가 없다.
#[test]
fn details_omit_route_line_when_no_route_reason() {
    let modal = WorkflowViewerModal::new(sample());
    let dumped = dump(&modal, 160, 40);
    assert!(
        !dumped.contains("route "),
        "no route line without a route reason: {dumped}"
    );
}

#[test]
fn details_render_output_tail_once_file_lands() {
    let dir = std::env::temp_dir().join(format!(
        "zo-viewer-tail-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out_path = dir.join("agent.md");
    std::fs::write(&out_path, "# Result\nfound 3 issues\nTAILLINE end\n").expect("write output");

    let mut view = sample();
    view.phases[0].agents[0].output_file = Some(out_path.display().to_string());
    let mut modal = WorkflowViewerModal::new(view);
    assert!(modal.select_agent_by_id("read:engine-id"));
    modal.refresh_output_tail();
    let dumped = dump(&modal, 160, 40);
    assert!(
        dumped.contains("OUTPUT TAIL") && dumped.contains("TAILLINE end"),
        "output tail must render once the file exists: {dumped}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn phase_navigation_uses_left_right() {
    let mut modal = WorkflowViewerModal::new(sample());
    assert_eq!(modal.selected_phase(), 0);
    modal.handle_key(press(KeyCode::Left)); // already first
    assert_eq!(modal.selected_phase(), 0);
    modal.handle_key(press(KeyCode::Right));
    assert_eq!(modal.selected_phase(), 1);
    modal.handle_key(press(KeyCode::Right)); // clamp at last
    assert_eq!(modal.selected_phase(), 1);
}

#[test]
fn initial_focus_tracks_the_active_phase_and_running_executor() {
    let mut view = sample();
    let mut executors = view.phases[0].agents.clone();
    for executor in &mut executors {
        executor.status = "completed".to_string();
        executor.current_tool = None;
    }
    executors[1].status = "running".to_string();
    executors[1].current_tool = Some("cargo_test".to_string());
    view.phases[0].status = "done".to_string();
    view.phases[1].status = "running".to_string();
    view.phases[1].agents = executors;
    view.phases[1].total = 2;

    let modal = WorkflowViewerModal::new(view);
    assert_eq!(modal.selected_phase(), 1);
    assert_eq!(modal.selected_agent(), 1);
    let rendered = dump(&modal, 140, 30);
    assert!(rendered.contains("Synthesize findings"), "{rendered}");
    assert!(rendered.contains("cargo_test"), "{rendered}");
}

#[test]
fn agent_navigation_uses_up_down_within_phase() {
    let mut modal = WorkflowViewerModal::new(sample());
    assert_eq!(modal.selected_agent(), 1, "initial focus follows live work");
    modal.handle_key(press(KeyCode::Up));
    assert_eq!(modal.selected_agent(), 0);
    modal.handle_key(press(KeyCode::Down));
    assert_eq!(modal.selected_agent(), 1);
    modal.handle_key(press(KeyCode::Down)); // clamp at last agent
    assert_eq!(modal.selected_agent(), 1);
}

#[test]
fn refresh_preserves_and_clamps_selection() {
    let mut modal = WorkflowViewerModal::new(sample());
    modal.handle_key(press(KeyCode::Right));
    assert_eq!(modal.selected_phase(), 1);
    // A refresh that drops to a single phase must clamp the cursor.
    let mut shrunk = sample();
    shrunk.phases.truncate(1);
    modal.refresh(shrunk, &[]);
    assert_eq!(modal.selected_phase(), 0);
}

#[test]
fn esc_and_q_and_ctrl_c_close() {
    let mut modal = WorkflowViewerModal::new(sample());
    assert_eq!(
        modal.handle_key(press(KeyCode::Esc)),
        Some(WorkflowViewerAction::Close)
    );
    assert_eq!(
        modal.handle_key(press(KeyCode::Char('q'))),
        Some(WorkflowViewerAction::Close)
    );
    let ctrl_c = KeyEvent {
        code: KeyCode::Char('c'),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: crossterm::event::KeyEventState::NONE,
    };
    assert_eq!(modal.handle_key(ctrl_c), Some(WorkflowViewerAction::Close));
}

#[test]
fn dump_shows_phases_agents_and_metrics() {
    let theme = Theme::zo();
    let backend = TestBackend::new(140, 30);
    let mut term = Terminal::new(backend).expect("backend");
    let modal = WorkflowViewerModal::new(sample());
    term.draw(|f| modal.draw(f, Rect::new(0, 0, 140, 30), &theme))
        .expect("draw");

    let buf = term.backend().buffer();
    let mut dump = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            dump.push_str(buf[(x, y)].symbol());
        }
        dump.push('\n');
    }
    assert!(dump.contains("FLEET"), "fleet tree title present");
    assert!(
        !dump.contains("› 1/2"),
        "the rail must not present workflow-phase order as a Todo ordinal: {dump}"
    );
    assert!(
        dump.contains("read:engine") && dump.contains("read:dispatch"),
        "both executors are listed under their group: {dump}"
    );
    assert!(
        dump.contains("Inspecting code paths"),
        "the active Todo label heads its group: {dump}"
    );
    assert!(
        !dump.contains("PLAN 1/2"),
        "workflow phase order must not impersonate Todo plan ordinals: {dump}"
    );
    assert!(dump.contains("10%"), "overall progress percent present");
    assert!(
        !dump.contains("% left"),
        "the redundant '% left' half must no longer be shown: {dump}"
    );
    assert!(
        dump.contains("1/6"),
        "the group header carries its own done/total tally: {dump}"
    );
    assert!(dump.contains("read"), "phase id present");
    assert!(dump.contains("read:engine"), "agent name present");
    assert!(
        dump.contains("Executor · read:dispatch"),
        "active executor detail pane title present"
    );
    // The Plan-to-Executor relationship is structural now: the executor is a
    // child row of the group its Plan step heads, so the inspector no longer
    // spells it out in prose.
    // The header NOW line carries the same sentence, so anchor on the tree's
    // group header row via its done/total tally.
    let group_row = dump
        .lines()
        .position(|line| line.contains("Inspecting code paths") && line.contains("1/6"))
        .expect("the Plan step heads a group");
    assert!(
        dump.lines().nth(group_row + 2).is_some_and(|line| line.contains("read:dispatch")),
        "the selected executor hangs off its Plan step's group: {dump}"
    );
    assert!(dump.contains("output"), "agent output path label present");
    assert!(dump.contains("gpt-5.5-fast"), "actual model visible");
    assert!(
        !dump.contains("openai:gpt-5.5-fast"),
        "provider prefix should not crowd model labels"
    );
    assert!(dump.contains("tools"), "agent metrics present");
    assert!(dump.contains("running"), "run status present");
}

#[test]
fn failed_executor_is_counted_and_its_error_stays_in_detail() {
    let mut view = sample();
    view.phases[0].agents[0].status = "failed".to_string();
    view.phases[0].agents[0].error = Some("provider exhausted retries".to_string());
    let mut modal = WorkflowViewerModal::new(view);
    assert!(modal.select_agent_by_id("read:engine-id"));
    let rendered = dump(&modal, 140, 30);

    assert!(rendered.contains("1 failed"), "failure tally: {rendered}");
    assert!(rendered.contains("provider exhausted retries"), "{rendered}");
    assert!(
        rendered.contains("Inspecting code paths"),
        "the tree still names the group the failed executor belongs to: {rendered}"
    );
    for width in [87, 88] {
        let narrow = dump(&modal, width, 24);
        assert!(
            // The label column is 8 cells wide (the longest label the pane
            // emits is `activity`), so `error` is padded to 8 + one gap.
            narrow.contains("error    provider exhausted retries"),
            "width {width} must keep the failure cause visible: {narrow}"
        );
    }
}

#[test]
fn low_height_fallback_keeps_plan_executor_and_activity_visible() {
    let rendered = dump(&WorkflowViewerModal::new(sample()), 48, 8);
    assert!(rendered.contains("Plan"), "compact Plan row: {rendered}");
    assert!(rendered.contains("Executor"), "compact executor row: {rendered}");
    assert!(
        rendered.contains("read:dispatch"),
        "active executor is selected: {rendered}"
    );
}

#[test]
fn low_height_boundary_avoids_empty_detail_cards() {
    for height in [12, 13, 14] {
        let rendered = dump(&WorkflowViewerModal::new(sample()), 72, height);
        assert!(rendered.contains("Activity"), "height {height}: {rendered}");
        assert!(rendered.contains("read_file"), "height {height}: {rendered}");
        assert!(
            !rendered.contains("╭ Executor ·"),
            "height {height} should use the compact body: {rendered}"
        );
    }
}

#[test]
fn responsive_breakpoints_keep_live_tool_visible() {
    // These straddle the vertical→medium and medium→wide body thresholds after
    // the outer frame's four columns of border/padding are removed.
    for width in [87, 88, 131, 132] {
        let rendered = dump(&WorkflowViewerModal::new(sample()), width, 24);
        assert!(
            rendered.contains("tool     read_file"),
            "width {width} must keep primary executor activity visible: {rendered}"
        );
        assert!(
            rendered.contains("Inspecting code paths"),
            "width {width} must retain Plan context: {rendered}"
        );
    }
}

#[test]
fn long_plan_rail_scrolls_to_keep_the_selected_step_visible() {
    let template = sample().phases.remove(0);
    let mut view = sample();
    view.phases = (0..14)
        .map(|idx| {
            let mut phase = template.clone();
            phase.id = format!("step-{idx}");
            phase.step_id = Some(phase.id.clone());
            phase.plan_step = Some(plan_step(
                &phase.id,
                &format!("Plan step {idx}"),
                &format!("Running plan step {idx}"),
                TodoChecklistStatus::Pending,
            ));
            phase.agents = vec![agent(&format!("executor-{idx}"), "pending", None)];
            phase
        })
        .collect();
    let mut modal = WorkflowViewerModal::new(view);
    for _ in 0..13 {
        modal.handle_key(press(KeyCode::Right));
    }

    let rendered = dump(&modal, 90, 14);
    assert!(rendered.contains("Plan step 13"), "selected Plan step: {rendered}");
    assert!(rendered.contains("executor-13"), "selected executor: {rendered}");
}

#[test]
fn event_inspector_toggles_renders_and_scrolls() {
    let mut modal = WorkflowViewerModal::new(sample());
    assert!(!modal.events_mode);

    // Ctrl+E opens the inspector; the timeline header renders even with no log on
    // disk (the sample run has none → the empty-state line).
    assert_eq!(modal.handle_key(ctrl_press(KeyCode::Char('e'))), None);
    assert!(modal.events_mode);
    let rendered = dump(&modal, 120, 24);
    assert!(
        rendered.contains("event log"),
        "inspector header: {rendered}"
    );
    let event_footer = footer_line(&Theme::zo(), 80, true, true);
    let event_footer_text: String = event_footer
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    assert!(event_footer_text.contains("^E back"), "{event_footer_text}");

    // In events mode ↓ scrolls the timeline, never the (frozen) agent cursor.
    let agent_before = modal.selected_agent();
    modal.handle_key(press(KeyCode::Down));
    assert_eq!(
        modal.selected_agent(),
        agent_before,
        "agent cursor is frozen while the inspector is open"
    );
    modal.events = (0..20).map(|idx| format!("event {idx}")).collect();
    modal.scroll_agents_down(3);
    assert_eq!(modal.events_scroll, 3, "mouse wheel scrolls events mode");

    // Ctrl+E returns to the Plan/Executor view; `q` still closes the modal.
    modal.handle_key(ctrl_press(KeyCode::Char('e')));
    assert!(!modal.events_mode);
    modal.handle_key(press(KeyCode::Char('e')));
    assert!(!modal.events_mode, "printable e remains a composer key");
    assert_eq!(
        modal.handle_key(press(KeyCode::Char('q'))),
        Some(WorkflowViewerAction::Close)
    );
}

/// A view already reconciled to a terminal state by the reader (Phase-5: the
/// status reconciliation now lives in `workflow_progress`, not the viewer).
fn terminal_sample() -> WorkflowView {
    let mut view = sample();
    view.status = "completed".to_string();
    view.phases[0].status = "done".to_string();
    view.phases[0].still_running = 0;
    for a in &mut view.phases[0].agents {
        a.status = "completed".to_string();
        a.current_tool = None;
    }
    view.phases[1].status = "done".to_string();
    view
}

#[test]
fn reconciled_terminal_view_renders_completed_without_spinning() {
    // Phase-5 render-dump: when the reader hands the viewer an event-derived
    // terminal read model (a finished run whose snapshot dropped its final
    // write), the modal faithfully shows "completed" with done glyphs and no
    // live spinner frame — it stops spinning even though the snapshot's last
    // word was "running".
    let modal = WorkflowViewerModal::new(terminal_sample());
    let out = dump(&modal, 140, 30);
    assert!(out.contains("completed"), "header status reconciled: {out}");
    assert!(out.contains("Inspect code paths"), "completed Plan label: {out}");
    assert!(
        !out.contains("Inspecting code paths"),
        "a terminal phase must not render a stale activeForm: {out}"
    );
    assert!(out.contains('✓'), "terminal phases/agents show done glyph");
    for frame in SPINNER {
        assert!(
            !out.contains(frame),
            "a finished run must not render a live spinner ({frame}): {out}"
        );
    }
}

/// The per-agent marker is the **task**, not the output path: on a pane this
/// short only the first rows survive, and the card now spends them on what the
/// reader scans for (status → task → model → metrics). The 90-cell
/// `/tmp/<name>.md` path sank below the identity rows with `id`/`route`/`event`
/// — reference material, not the answer to "what is this executor doing".
#[test]
fn detail_panel_follows_selected_agent() {
    let mut modal = WorkflowViewerModal::new(sample());
    let before = dump(&modal, 120, 24);
    assert!(
        before.contains("Executor · read:dispatch")
            && before.contains("Inspect read:dispatch and report findings"),
        "running executor details shown first: {before}"
    );

    modal.handle_key(press(KeyCode::Up));
    let first = dump(&modal, 120, 24);
    assert!(
        first.contains("Executor · read:engine")
            && first.contains("Inspect read:engine and report findings"),
        "selected completed executor details shown: {first}"
    );
    modal.handle_key(press(KeyCode::Down));
    let after = dump(&modal, 120, 24);
    assert!(after.contains("read_file"), "current tool is visible");
}

fn dump(modal: &WorkflowViewerModal, w: u16, h: u16) -> String {
    let theme = Theme::zo();
    let backend = TestBackend::new(w, h);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| modal.draw(f, Rect::new(0, 0, w, h), &theme))
        .expect("draw");
    let buf = term.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

/// Render the whole modal into a `TestBackend` and hand back the buffer, so a
/// test can assert on *styles* and not only on glyphs.
fn buffer_dump(
    modal: &WorkflowViewerModal,
    theme: &Theme,
    w: u16,
    h: u16,
) -> ratatui::buffer::Buffer {
    let backend = TestBackend::new(w, h);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| modal.draw(f, Rect::new(0, 0, w, h), theme))
        .expect("draw");
    term.backend().buffer().clone()
}

/// The first row whose text starts at `col` with `needle`, as `(y, x)` of the
/// needle's first cell.
fn find_row(buf: &ratatui::buffer::Buffer, needle: &str) -> (u16, u16) {
    for y in 0..buf.area.height {
        let row: String = (0..buf.area.width)
            .map(|x| buf[(x, y)].symbol())
            .collect();
        if let Some(byte_idx) = row.find(needle) {
            let col = row[..byte_idx].chars().count();
            return (y, u16::try_from(col).unwrap_or(0));
        }
    }
    panic!("{needle:?} not found in:\n{buf:?}");
}

/// **대비 회귀 핀** — the detail pane's contrast ladder, asserted on the real
/// screen buffer.
///
/// The whole pane used to render through `Typography::key_hint` / `dim`, both
/// of which are `dim` *plus* `Modifier::DIM`: a terminal halves that again, so
/// every label and nearly every value sat at roughly 1.5:1 on the dialog
/// surface — the "dark grey on dark grey, cannot read any of it" report. This
/// pins the ladder that replaced it:
///
/// * label column = `muted`, never `faint`, and never `Modifier::DIM`
/// * value column = `fg`
/// * a semantic value (the running tool) keeps its own hue
/// * the selected list row actually carries the selection wash
#[test]
fn detail_pane_contrast_ladder_is_readable_and_selection_is_washed() {
    let theme = Theme::pi();
    let modal = WorkflowViewerModal::new(sample());
    let buf = buffer_dump(&modal, &theme, 160, 40);

    // `task` is an ordinary row: muted label, fg value.
    let (y, x) = find_row(&buf, "task     Inspect");
    for dx in 0..4 {
        let label = buf[(x + dx, y)].style();
        assert_eq!(
            label.fg,
            Some(theme.palette.muted),
            "detail labels must be `muted`, not the DIM-modified key hint"
        );
        assert_ne!(
            label.fg,
            Some(theme.palette.faint),
            "`faint` is for rules and decoration only — never for text"
        );
        assert!(
            !label
                .add_modifier
                .contains(ratatui::style::Modifier::DIM),
            "no detail label may stack `Modifier::DIM` on an already dim hue"
        );
    }
    for dx in 9..16 {
        assert_eq!(
            buf[(x + dx, y)].style().fg,
            Some(theme.palette.fg),
            "detail values sit at the top of the ladder (`fg`)"
        );
    }

    // A semantic value keeps its hue: the live tool is the tool-role teal.
    let (ty, tx) = find_row(&buf, "tool     read_file");
    assert_eq!(
        buf[(tx + 9, ty)].style().fg,
        Some(theme.palette.teal),
        "the running tool keeps the tool-role hue"
    );

    // ...and the label beside it is still `muted`, so the pair reads as one row.
    assert_eq!(buf[(tx, ty)].style().fg, Some(theme.palette.muted));

    // The selected executor row is washed behind its `→` cursor.
    let needle = "\u{2192} \u{280b} read:dispatch";
    let (sy, sx) = find_row(&buf, needle);
    assert_eq!(
        buf[(sx, sy)].style().bg,
        theme.selection_bg(),
        "{needle:?} must carry the shared selection wash"
    );

    // ...and the selected executor's name is its own identity hue *calmed*
    // toward `fg`, so the wash does the selecting instead of a hot pink name.
    // (`→ ` cursor + status glyph + space = 4 cells before the name.)
    let (ny, nx) = find_row(&buf, "\u{2192} \u{280b} read:dispatch");
    let raw = theme.agent_color("read:dispatch");
    let nx = nx + 4;
    assert_eq!(
        buf[(nx, ny)].style().fg,
        Some(theme.calm_identity(raw)),
        "a selected row's identity hue is toned down, not dropped"
    );
    assert_ne!(buf[(nx, ny)].style().fg, Some(raw));

    // Section headings are BOLD accent inlays on a `border` rule.
    let (hy, hx) = find_row(&buf, "LIVE OUTPUT");
    let heading = buf[(hx, hy)].style();
    assert_eq!(heading.fg, Some(theme.palette.accent));
    assert!(heading.add_modifier.contains(ratatui::style::Modifier::BOLD));
    assert_eq!(
        buf[(hx - 2, hy)].style().fg,
        Some(theme.palette.border),
        "the inlay rides a `border` rule, the sidebar's `section_rule` idiom"
    );
}

/// The dialog owns a surface of its own: every cell inside the modal rect is
/// painted with the elevation-2 glass, including the rows a short body never
/// reaches. Before this the viewer relied entirely on the host frosting the
/// backdrop, so the modal read as text pasted onto the transcript, and the
/// space under a hugged pane was raw terminal background.
#[test]
fn the_modal_paints_its_own_surface_edge_to_edge() {
    let theme = Theme::pi();
    let modal = WorkflowViewerModal::new(sample());
    let buf = buffer_dump(&modal, &theme, 160, 40);
    let surface = theme.surface2().expect("pi blends");
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            let bg = buf[(x, y)].style().bg;
            assert!(
                bg == Some(surface) || bg == theme.selection_bg(),
                "cell ({x},{y}) is neither the dialog surface nor a selection \
                 wash: {bg:?}"
            );
        }
    }
    // The bottom band — under every pane, where the old build left raw
    // terminal background — is pure surface.
    for y in buf.area.height - 4..buf.area.height {
        for x in 0..buf.area.width {
            assert_eq!(
                buf[(x, y)].style().bg,
                Some(surface),
                "cell ({x},{y}) below the panes must still be the dialog surface"
            );
        }
    }
}

/// **공간 배분 회귀 핀 (Ctrl+O)** — detail 패널이 충분히 넓으면 내부가
/// `메타 │ LIVE OUTPUT` 2단으로 갈라지고, 우측이 패널 바닥까지 채운다. 폭이
/// 모자라면 기존 1단 스택을 그대로 유지한다(3분기 레이아웃은 그대로).
#[test]
fn a_wide_detail_panel_splits_into_meta_and_output_columns() {
    let mut view = sample();
    view.phases[0].agents[1].output_tail = Some(
        (0..40)
            .map(|idx| format!("streamed prose row {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let modal = WorkflowViewerModal::new(view);

    let wide = dump(&modal, 200, 40);
    assert!(
        wide.contains("LIVE OUTPUT"),
        "the output section is present: {wide}"
    );
    // Two columns on one row: a meta label on the left, streamed prose on the
    // right, with the `│` rail between them.
    let split_row = wide
        .lines()
        .find(|line| line.contains("tool     read_file") && line.contains("streamed prose row"))
        .unwrap_or_else(|| panic!("meta and output must share a row:\n{wide}"));
    assert!(
        split_row.contains('\u{2502}'),
        "a `│` rail separates the columns: {split_row}"
    );
    assert!(
        wide.contains("streamed prose row 39"),
        "the column follows its own tail: {wide}"
    );

    // A genuinely narrow inspector keeps the single stack it always had.
    let narrow = dump(&modal, 108, 30);
    assert!(narrow.contains("status   running"), "{narrow}");
    assert!(
        !narrow
            .lines()
            .any(|line| line.contains("status   running") && line.contains("streamed prose row")),
        "a sub-split-width inspector must not split: {narrow}"
    );
}

/// **결함 D 회귀 핀 (재설계)** — 예전에는 한 항목짜리 구조 컬럼 두 개가
/// 각각 40~48칸을 먹고 나머지 행을 텅 비웠다. 이제 phase 와 executor 는 **한
/// 개의 FLEET 트리**라서, 고를 것이 하나뿐이어도 컬럼은 하나뿐이다.
#[test]
fn phases_and_executors_share_one_tree_column() {
    let mut view = sample();
    view.phases.truncate(1);
    view.phases[0].step_id = None;
    view.phases[0].plan_step = None;
    view.phases[0].agents.truncate(1);
    view.phases[0].total = 1;
    let modal = WorkflowViewerModal::new(view);
    let rendered = dump(&modal, 160, 40);

    assert!(
        !rendered.contains("Run scope \u{00b7} unlinked"),
        "the phase is a tree header, not a pane of its own: {rendered}"
    );
    assert!(
        rendered.contains("FLEET"),
        "the one selector column is the fleet tree: {rendered}"
    );
    // The group header and the executor it owns are in the same column: the
    // executor row starts to the right of the header's caret, not in a second
    // pane forty cells over.
    // Anchor on the tree caret: the header NOW line repeats the group's label
    // two rows above, but only the tree row folds.
    let buf = buffer_dump(&modal, &Theme::zo(), 160, 40);
    let (phase_y, phase_x) = find_row(&buf, "▾");
    let next: String = (0..buf.area.width)
        .map(|x| buf[(x, phase_y + 1)].symbol())
        .collect();
    let Some(agent_x) = next.find("read:engine").map(|idx| next[..idx].chars().count()) else {
        panic!("the executor sits under its group:\n{next}");
    };
    assert!(
        agent_x > usize::from(phase_x) && agent_x < usize::from(phase_x) + 12,
        "it is indented under the header, not in another column \
         (header col {phase_x}, executor col {agent_x})"
    );
}

/// **세로 공백 회귀 핀** — 인스펙터는 body 를 끝까지 쓴다. 예전에는 콘텐츠
/// 높이에 hug 돼서, 출력은 잘리는데 그 아래 30행이 비는 화면이 나왔다
/// (사용자 스크린샷의 "허접한" 화면).
#[test]
fn the_inspector_fills_the_body_instead_of_hugging() {
    let theme = Theme::zo();
    let mut view = sample();
    view.phases[0].agents[1].output_tail = Some(
        (0..60)
            .map(|idx| format!("streamed prose row {idx}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let modal = WorkflowViewerModal::new(view);
    let buf = buffer_dump(&modal, &theme, 160, 40);

    // Header (2) + the modal's own rules; the footer takes the last rows.
    let body_bottom = buf.area.height - 3;
    let inspector_x = buf.area.width - 20;
    let last_painted = (0..=body_bottom)
        .rev()
        .find(|y| !buf[(inspector_x, *y)].symbol().trim().is_empty())
        .expect("the inspector paints something");
    assert!(
        last_painted + 2 >= body_bottom,
        "a long output tail must reach the bottom of the body instead of \
         stopping above a dead band (last painted {last_painted}, body ends \
         at {body_bottom})"
    );
}

#[test]
fn running_agent_with_no_tokens_omits_the_bare_zero() {
    let mut view = sample();
    // A running agent whose token total isn't persisted until it finishes —
    // the live case from the screenshot. It must not render a bare "0 ·".
    view.phases[0].agents = vec![WorkflowAgentRow {
        id: "live-agent-id".to_string(),
        name: "live-agent".to_string(),
        status: "running".to_string(),
        current_tool: Some("bash".to_string()),
        model: "gpt-5.5-fast".to_string(),
        tool_calls: Some(1),
        tokens: 0,
        elapsed_secs: 91,
        ..WorkflowAgentRow::default()
    }];
    let out = dump(&WorkflowViewerModal::new(view), 100, 24);
    assert!(out.contains("1 tools"), "tool count still shown");
    assert!(!out.contains("0 ·"), "no broken-looking zero-token metric");
}

#[test]
fn unknown_tool_count_is_omitted_not_invented() {
    let mut view = sample();
    view.phases[0].agents = vec![WorkflowAgentRow {
        id: "legacy-agent-id".to_string(),
        name: "legacy-agent".to_string(),
        status: "running".to_string(),
        current_tool: Some("grep".to_string()),
        model: "gpt-5.5-fast".to_string(),
        tool_calls: None,
        tokens: 0,
        elapsed_secs: 91,
        ..WorkflowAgentRow::default()
    }];
    let out = dump(&WorkflowViewerModal::new(view), 100, 24);
    assert!(
        !out.contains("tools"),
        "unknown tool count must not be faked"
    );
    assert!(out.contains("1m 31s"), "elapsed still shown: {out}");
}

#[test]
fn mouse_wheel_scrolls_the_agent_pane() {
    let mut view = sample();
    view.phases[0].agents = (0..40)
        .map(|i| agent(&format!("agent-{i:02}"), "running", None))
        .collect();
    let mut modal = WorkflowViewerModal::new(view);

    assert!(
        dump(&modal, 80, 12).contains("agent-00"),
        "first agent visible at top"
    );
    modal.scroll_agents_down(20);
    assert!(
        !dump(&modal, 80, 12).contains("agent-00"),
        "wheel-down scrolls the first agent off-screen"
    );
    modal.scroll_agents_up(100); // saturates back to the top
    assert!(
        dump(&modal, 80, 12).contains("agent-00"),
        "wheel-up returns to the top"
    );
}

#[test]
fn overflowing_agent_pane_draws_scrollbar() {
    let mut view = sample();
    view.phases[0].agents = (0..40)
        .map(|i| agent(&format!("agent-{i:02}"), "running", None))
        .collect();
    let modal = WorkflowViewerModal::new(view);
    let theme = Theme::no_color();
    let backend = TestBackend::new(120, 24);
    let mut term = Terminal::new(backend).expect("backend");
    term.draw(|f| modal.draw(f, Rect::new(0, 0, 120, 24), &theme))
        .expect("draw");

    let buf = term.backend().buffer();
    let mut dump = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            dump.push_str(buf[(x, y)].symbol());
        }
        dump.push('\n');
    }
    assert!(dump.contains('#'), "scrollbar thumb should render: {dump}");
    assert!(dump.contains('.'), "scrollbar track should render: {dump}");
}

#[test]
fn fmt_helpers() {
    assert_eq!(fmt_tokens(950), "950");
    assert_eq!(fmt_tokens(103_000), "103.0k");
    assert_eq!(fmt_tokens(1_200_000), "1.2M");
    assert_eq!(format_elapsed(42), "42s");
    assert_eq!(format_elapsed(175), "2m 55s");
    assert_eq!(format_elapsed(3600), "1h 0m");
    assert_eq!(format_elapsed(72_248), "20h 4m");
}

#[test]
fn running_agents_counts_live_rows_not_recorded_still_running() {
    // sample()'s first phase is `running` with one completed + one running
    // agent row, and recorded still_running=1. Zero the recorded count to
    // prove the header reads the live rows, not the post-barrier tally.
    let mut view = sample();
    view.phases[0].still_running = 0;
    assert_eq!(view.running_agents(), 1);
    // The header string reflects it (K active, K>0).
    let theme = Theme::zo();
    let lines = WorkflowViewerModal::new(view).header_lines(&theme, 160);
    let text: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(text.contains("1 running"), "header was {text:?}");
    // Phase 1 running: 1 finished + 1 in-flight agent of 6 → (1 + 0.3)/6 ≈ 21%,
    // phase 2 pending (0): overall (21 + 0)/2 = 10% (was 8% before within-agent
    // credit lifted the active phase off its raw 1/6 = 16%).
    assert!(
        text.contains("10%"),
        "header should show the overall phase-progress percentage with in-flight credit: {text:?}"
    );
    assert!(
        !text.contains("% left"),
        "the redundant '% left' half must no longer be shown: {text:?}"
    );
}

/// 헤더 두 번째 줄(NOW)은 지금 무엇을 하는지 한 줄로 말한다. 헤더 높이는
/// `regions()` 가 2로 고정하므로 줄 수는 절대 2를 벗어나면 안 된다.
#[test]
fn header_now_line_names_the_running_phase_and_its_coordinate() {
    let theme = Theme::zo();
    let lines = WorkflowViewerModal::new(sample()).header_lines(&theme, 160);
    assert_eq!(lines.len(), 2, "the header owns exactly two rows");
    let now: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(
        now.contains("Inspecting code paths"),
        "the NOW line carries the active step's own sentence: {now:?}"
    );
    assert!(
        now.contains("1/2"),
        "the NOW line carries the phase coordinate: {now:?}"
    );
    assert!(
        now.contains("read_file"),
        "the running executor's live tool shows: {now:?}"
    );
    // The upcoming phase deliberately lives in the progress band's scope note,
    // not here — repeating it doubled "synthesize" on screen.
    assert!(
        !now.contains("next:"),
        "the NOW line does not restate the scope note's next phase: {now:?}"
    );
}

/// 계획 스텝과 연결되지 않은 페이즈는 자기 id 로 말한다(가짜 Plan 라벨 금지).
#[test]
fn header_now_line_falls_back_to_the_phase_id_when_unlinked() {
    let mut view = sample();
    view.phases[0].plan_step = None;
    let theme = Theme::zo();
    let lines = WorkflowViewerModal::new(view).header_lines(&theme, 160);
    assert_eq!(lines.len(), 2);
    let now: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(now.contains("read"), "unlinked phases show their id: {now:?}");
}

/// 종료된 런에서도 헤더는 여전히 정확히 두 줄이다.
#[test]
fn header_keeps_two_rows_on_a_terminal_view() {
    let mut view = sample();
    view.status = "completed".to_string();
    for phase in &mut view.phases {
        phase.status = "done".to_string();
        phase.agents.clear();
    }
    let theme = Theme::zo();
    let lines = WorkflowViewerModal::new(view).header_lines(&theme, 160);
    assert_eq!(lines.len(), 2, "a terminal view still owns two header rows");
    let now: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(!now.is_empty(), "the NOW row must say something: {now:?}");
    assert_eq!(
        WorkflowViewerModal::new(WorkflowView::default())
            .header_lines(&theme, 160)
            .len(),
        2,
        "an empty view too"
    );
}

/// 좁은 폭에서 한국어 라벨이 헤더를 넘치게 하면 안 된다.
#[test]
fn header_now_line_never_overflows_its_width() {
    let mut view = sample();
    view.phases[0].plan_step = Some(plan_step(
        "read",
        "아주 긴 한국어 계획 스텝 이름",
        "아주 긴 한국어 계획 스텝을 지금 실행하는 중입니다",
        TodoChecklistStatus::InProgress,
    ));
    let theme = Theme::zo();
    for width in [12_u16, 28, 40, 160] {
        let lines = WorkflowViewerModal::new(view.clone()).header_lines(&theme, width);
        assert_eq!(lines.len(), 2, "@{width}");
        assert!(
            line_width(&lines[1]) <= usize::from(width),
            "NOW row must fit {width} cells, was {}",
            line_width(&lines[1])
        );
    }
}

#[test]
fn tally_is_live_while_running_and_recorded_when_terminal() {
    let mut phase = sample().phases.remove(0); // running, 1 completed row
    phase.completed = 99; // recorded count ignored while running
    assert_eq!(phase.completed_now(), 1);
    assert_eq!(phase.running_now(), 1);
    // Past the barrier the recorded tallies win (rows may be budget-capped).
    phase.status = "done".to_string();
    phase.still_running = 3;
    assert_eq!(phase.completed_now(), 99);
    assert_eq!(phase.running_now(), 3);
}

#[test]
fn short_is_display_width_aware() {
    assert_eq!(short("hello", 10), "hello");
    let ascii = short("abcdefghij", 5);
    assert!(ascii.ends_with('…'));
    assert!(UnicodeWidthStr::width(ascii.as_str()) <= 5);
    // CJK glyphs are two columns each: budget 5 fits two (width 4) + `…`.
    let cjk = short("가나다라마", 5);
    assert!(cjk.ends_with('…'));
    assert!(
        UnicodeWidthStr::width(cjk.as_str()) <= 5,
        "width {} of {cjk:?}",
        UnicodeWidthStr::width(cjk.as_str())
    );
}

#[test]
fn overscroll_does_not_blank_the_agent_pane() {
    let theme = Theme::zo();
    let backend = TestBackend::new(100, 24);
    let mut term = Terminal::new(backend).expect("backend");
    let mut modal = WorkflowViewerModal::new(sample());
    for _ in 0..20 {
        modal.handle_key(press(KeyCode::PageDown));
    }
    term.draw(|f| modal.draw(f, Rect::new(0, 0, 100, 24), &theme))
        .expect("draw");
    let buf = term.backend().buffer();
    let mut dump = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            dump.push_str(buf[(x, y)].symbol());
        }
    }
    assert!(
        dump.contains("read:engine"),
        "an agent row must stay visible after overscroll (clamp)"
    );
}

#[test]
fn footer_compacts_on_narrow_width_without_losing_close() {
    let theme = Theme::no_color();
    let line = footer_line(&theme, 30, false, true);
    let text: String = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();

    assert!(
        line_width(&line) <= 30,
        "footer should fit narrow panes: {text:?}"
    );
    assert!(text.contains("executor"), "primary axis survives: {text}");
    assert!(text.contains("Esc close"), "close hint survives: {text}");
    assert!(
        !text.contains("PgUp/PgDn"),
        "page hint should be omitted first: {text}"
    );

    let event_line = footer_line(&theme, 31, true, true);
    let event_text: String = event_line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    assert!(line_width(&event_line) <= 31, "{event_text}");
    assert!(event_text.contains("^E back"), "{event_text}");
    assert!(event_text.contains("Esc close"), "{event_text}");
}

/// Defect F: the agent detail pane rendered rows of bare `{}`.
///
/// The braces came from the shared tool-input summarizer, which stringified an
/// argument-less payload; the manifest feed itself is prose. These assertions
/// cover the pane end of that contract: every activity row is readable, and a
/// running agent with nothing to show yet says so instead of showing nothing.
mod agent_detail_readability {
    use super::*;

    fn running(recent: &[&str], tail: Option<&str>) -> WorkflowAgentRow {
        WorkflowAgentRow {
            id: "agent-1".to_string(),
            name: "explorer".to_string(),
            status: "running".to_string(),
            elapsed_secs: 42,
            recent_tools: recent.iter().map(|entry| (*entry).to_string()).collect(),
            output_tail: tail.map(str::to_string),
            ..WorkflowAgentRow::default()
        }
    }

    /// The stacked card both viewers fall back to on a narrow terminal: the
    /// meta column with the output column appended under it.
    fn rendered(agent: &WorkflowAgentRow) -> Vec<String> {
        let theme = Theme::no_color();
        let mut column = agent_meta_column(agent, 80, &theme, DetailFolds::default());
        column.append(agent_output_column(agent, 80, &theme, false, None));
        column
            .into_lines()
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    /// Argument-less and argument-carrying tools alike read as tool names and
    /// human briefs — never as a JSON payload.
    #[test]
    fn activity_rows_are_readable_and_never_bare_braces() {
        // The shapes `record_current_tool` writes: a name alone when the call
        // had no arguments, `name · brief` when it had.
        let agent = running(
            &["InstrumentLog", "Read · src/main.rs", "Bash · cargo test -p zo-cli"],
            Some("분석을 시작합니다\n의존 그래프 확인 중"),
        );
        let lines = rendered(&agent);
        for expected in ["InstrumentLog", "src/main.rs", "cargo test -p zo-cli"] {
            assert!(
                lines.iter().any(|line| line.contains(expected)),
                "activity row {expected} must be readable in:\n{}",
                lines.join("\n")
            );
        }
        assert!(
            lines.iter().any(|line| line.contains("의존 그래프 확인 중")),
            "streamed prose renders as text:\n{}",
            lines.join("\n")
        );
        for line in &lines {
            assert_ne!(line.trim(), "{}", "no row may be a bare JSON object");
            assert!(
                !line.contains("{}"),
                "an argument-less call must not print its empty payload: {line:?}"
            );
        }
    }

    /// A running agent that has produced nothing yet gets a dim placeholder —
    /// an empty pane reads as a broken viewer.
    #[test]
    fn a_silent_running_agent_says_it_has_no_output_yet() {
        let lines = rendered(&running(&[], None));
        assert!(
            lines.iter().any(|line| line.contains("no output yet")),
            "expected a placeholder, got:\n{}",
            lines.join("\n")
        );
        assert!(
            lines.iter().any(|line| line.contains("42s")),
            "the placeholder says how long we have been waiting:\n{}",
            lines.join("\n")
        );
    }

    /// Once real output arrives the placeholder gets out of the way.
    #[test]
    fn the_placeholder_yields_to_real_output() {
        let lines = rendered(&running(&[], Some("first real line")));
        assert!(lines.iter().any(|line| line.contains("first real line")));
        assert!(
            !lines.iter().any(|line| line.contains("no output yet")),
            "placeholder must not sit beside real output:\n{}",
            lines.join("\n")
        );
    }
}


/// **스택 레이아웃 회귀 핀** — 좁은 터미널에서 트리가 위에 쌓일 때, 카드에
/// 자기 룰 2행을 안 빼주면 커서가 든 행을 보이게 하려고 한 행 스크롤되고,
/// 그 한 행이 하필 executor 들이 매달린 **phase 헤더**다(그룹 없는 평면 목록).
#[test]
fn the_stacked_tree_keeps_its_group_header() {
    let mut view = sample();
    view.phases.truncate(1);
    let modal = WorkflowViewerModal::new(view);
    let rendered = dump(&modal, 70, 24);
    let head = rendered
        .lines()
        // The header NOW line repeats the sentence; the group header is the
        // row with the fold caret.
        .position(|line| line.contains('▾') && line.contains("Inspecting code paths"))
        .unwrap_or_else(|| panic!("the group header survives the stack:\n{rendered}"));
    assert!(
        rendered
            .lines()
            .nth(head + 1)
            .is_some_and(|line| line.contains("read:engine")),
        "with its executors still under it:\n{rendered}"
    );
}

/// `↑`/`↓`/`End` 는 **런 전체**를 걷는다. 예전엔 선택된 phase 안에 갇혀 있어서,
/// phase 2 의 executor 는 아무도 모르는 `←`/`→` 축을 먼저 찾아야만 닿았다.
#[test]
fn executor_navigation_crosses_phase_groups() {
    let mut view = sample();
    view.phases[1].agents = vec![WorkflowAgentRow {
        id: "synth-id".to_string(),
        name: "synthesize".to_string(),
        status: "running".to_string(),
        ..WorkflowAgentRow::default()
    }];
    let mut modal = WorkflowViewerModal::new(view);
    assert_eq!((modal.selected_phase(), modal.selected_agent()), (0, 1));

    modal.handle_key(press(KeyCode::Down));
    assert_eq!(
        (modal.selected_phase(), modal.selected_agent()),
        (1, 0),
        "past the last executor of a group the cursor enters the next one"
    );
    modal.handle_key(press(KeyCode::Up));
    assert_eq!((modal.selected_phase(), modal.selected_agent()), (0, 1));

    modal.handle_key(press(KeyCode::End));
    assert_eq!(
        (modal.selected_phase(), modal.selected_agent()),
        (1, 0),
        "End is the last executor of the run, not of the selected phase"
    );
    modal.handle_key(press(KeyCode::Home));
    assert_eq!((modal.selected_phase(), modal.selected_agent()), (0, 0));
}
