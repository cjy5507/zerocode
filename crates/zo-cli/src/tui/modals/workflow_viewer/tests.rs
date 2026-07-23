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
    assert!(rendered.contains("Workflow → Executors"), "{rendered}");
    assert!(rendered.contains("Plan link unavailable"), "{rendered}");
    assert!(!rendered.contains("Plan → Executors"), "{rendered}");

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
    assert!(linked.contains("Workflow → Executors"), "{linked}");
    assert!(linked.contains("1/2 phases Plan linked"), "{linked}");
    assert!(linked.contains("Inspecting code paths · Plan linked"), "{linked}");
    assert!(linked.contains("Plan linked · Inspect code paths"), "{linked}");

    modal.handle_key(press(KeyCode::Right));
    let unlinked = dump(&modal, 140, 30);
    assert!(unlinked.contains("synthesize · Plan unlinked"), "{unlinked}");
    assert!(
        unlinked.contains("workflow 2/2 · Plan unlinked"),
        "{unlinked}"
    );
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
    assert!(rendered.contains("Run → Executors"), "{rendered}");
    assert!(rendered.contains("Run scope · unlinked"), "{rendered}");
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
    assert!(dumped.contains("activity"), "activity section: {dumped}");
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
    assert!(
        dumped.contains("learned-shadow-differs:gpt-5.6-sol"),
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
        dumped.contains("output tail") && dumped.contains("TAILLINE end"),
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
    assert!(dump.contains("Plan steps"), "Plan rail title present");
    assert!(
        !dump.contains("› 1/2"),
        "the rail must not present workflow-phase order as a Todo ordinal: {dump}"
    );
    assert!(dump.contains("Executors"), "Executor pane title present");
    assert!(dump.contains("PLAN"), "Plan-to-Executor path is visible");
    assert!(
        dump.contains("PLAN phase 1/2"),
        "the 1/2 counter is explicitly a workflow-phase index: {dump}"
    );
    assert!(
        !dump.contains("PLAN 1/2"),
        "workflow phase order must not impersonate Todo plan ordinals: {dump}"
    );
    assert!(
        dump.contains("Inspecting code paths"),
        "active Todo label is visible in the path: {dump}"
    );
    assert!(dump.contains("% done"), "overall progress percent present");
    assert!(
        !dump.contains("% left"),
        "the redundant '% left' half must no longer be shown: {dump}"
    );
    assert!(
        dump.contains("1/6") && dump.contains("21%"),
        "phase rail should show phase-local progress (1 of 6 done + 1 in-flight → 21%): {dump}"
    );
    assert!(dump.contains("read"), "phase id present");
    assert!(dump.contains("read:engine"), "agent name present");
    assert!(
        dump.contains("Executor · read:dispatch"),
        "active executor detail pane title present"
    );
    assert!(
        dump.contains("plan step -> read:dispatch"),
        "selected Plan-to-Executor relationship is explicit: {dump}"
    );
    assert!(
        dump.contains("21% done") && !dump.contains("% left"),
        "detail pane should show phase progress (with in-flight credit) and no redundant '% left': {dump}"
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
    assert!(rendered.contains("plan step -> read:engine"), "{rendered}");
    for width in [87, 88] {
        let narrow = dump(&modal, width, 24);
        assert!(
            narrow.contains("error   provider exhausted retries"),
            "width {width} must keep the failure cause visible: {narrow}"
        );
    }
}

#[test]
fn low_height_fallback_keeps_plan_executor_and_activity_visible() {
    let rendered = dump(&WorkflowViewerModal::new(sample()), 48, 8);
    assert!(rendered.contains("PLAN"), "compact path: {rendered}");
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
            rendered.contains("tool    read_file"),
            "width {width} must keep primary executor activity visible: {rendered}"
        );
        assert!(
            rendered.contains("Inspect code paths"),
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

#[test]
fn detail_panel_follows_selected_agent() {
    let mut modal = WorkflowViewerModal::new(sample());
    let before = dump(&modal, 120, 24);
    assert!(
        before.contains("Executor · read:dispatch") && before.contains("/tmp/read:dispatch.md"),
        "running executor details shown first: {before}"
    );

    modal.handle_key(press(KeyCode::Up));
    let first = dump(&modal, 120, 24);
    assert!(
        first.contains("Executor · read:engine") && first.contains("/tmp/read:engine.md"),
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
    let line = WorkflowViewerModal::new(view).header_line(&theme);
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(text.contains("1 active"), "header was {text:?}");
    // Phase 1 running: 1 finished + 1 in-flight agent of 6 → (1 + 0.3)/6 ≈ 21%,
    // phase 2 pending (0): overall (21 + 0)/2 = 10% (was 8% before within-agent
    // credit lifted the active phase off its raw 1/6 = 16%).
    assert!(
        text.contains("10% done"),
        "header should show the overall phase-progress percentage with in-flight credit: {text:?}"
    );
    assert!(
        !text.contains("% left"),
        "the redundant '% left' half must no longer be shown: {text:?}"
    );
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
