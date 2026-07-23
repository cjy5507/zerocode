use super::*;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use zo_cli::tui::modals::Effort;
use zo_cli::tui::theme::Theme;
use std::path::{Path, PathBuf};
use std::sync::MutexGuard;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

struct EnvPathGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvPathGuard {
    fn set(key: &'static str, value: &Path) -> Self {
        // Route through the single crate-wide env lock so this guard serializes
        // against env-mutating tests in OTHER modules too (a per-file `ENV_LOCK`
        // only serialized within this file, letting `ZO_TODO_STORE` writers
        // elsewhere race and stomp each other).
        let lock = crate::test_env_lock();
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self {
            key,
            previous,
            _lock: lock,
        }
    }
}

impl Drop for EnvPathGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("zo-{label}-{nanos}"))
}

fn test_app() -> App {
    let (_tx, rx) = mpsc::channel(8);
    let (cmd_tx, _cmd_rx) = mpsc::channel(8);
    App::new(Theme::no_color(), rx, cmd_tx)
}

#[test]
fn started_turn_error_exit_tears_down_activity_before_propagating() {
    for error in [
        TuiLoopError::Turn("turn task panicked: synthetic".to_string()),
        TuiLoopError::Tui("synthetic draw failure".to_string()),
    ] {
        let mut app = test_app();
        app.begin_turn_with_generation(1);
        let expected = error.to_string();

        let result = propagate_started_turn_error(&mut app, error);

        let error = result.expect_err("the original turn error must propagate");
        assert_eq!(error.to_string(), expected);
        assert!(
            app.turn_activity().is_none(),
            "an error exit must disarm permanent turn tick work"
        );
    }
}

fn draw_app_dump(app: &mut App, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    app.draw(&mut terminal).expect("draw");
    (0..height)
        .map(|y| {
            (0..width)
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
        .join("\n")
}

#[test]
fn exec_swap_and_edit_gate_follow_the_live_implementer_contract() {
    use std::future::Future;
    use std::pin::Pin;

    use runtime::RouteTaskComplexity::{Large, Medium, Trivial};

    struct NoopAsyncClient;
    impl runtime::AsyncApiClient for NoopAsyncClient {
        fn stream_async<'a>(
            &'a self,
            _request: runtime::ApiRequest,
            _render_tx: mpsc::Sender<RenderBlock>,
            _text_block_id: BlockId,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<Vec<runtime::AssistantEvent>, runtime::RuntimeError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    assert!(should_install_exec_implementer(
        tools::SmartExecSwap::Easy,
        Trivial
    ));
    assert!(!should_install_exec_implementer(
        tools::SmartExecSwap::Easy,
        Medium
    ));
    assert!(!should_install_exec_implementer(
        tools::SmartExecSwap::Easy,
        Large
    ));
    assert!(!should_install_exec_implementer(
        tools::SmartExecSwap::Never,
        Trivial
    ));
    assert!(should_install_exec_implementer(
        tools::SmartExecSwap::Always,
        Large
    ));

    let native_contract = runtime::ExecContract {
        impl_client: None,
        impl_model: "claude-sonnet-5".to_string(),
        plan_first: true,
    };
    assert!(
        !exec_contract_arms_edit_gate(Some(&native_contract)),
        "default easy policy on medium/hard turns and execSwap=never must leave direct edits native"
    );

    let swapped_contract = runtime::ExecContract {
        impl_client: Some(Arc::new(NoopAsyncClient)),
        impl_model: "claude-sonnet-5".to_string(),
        plan_first: false,
    };
    assert!(
        exec_contract_arms_edit_gate(Some(&swapped_contract)),
        "easy turns and execSwap=always must gate foreground edits while the swap is live"
    );
    assert!(!exec_contract_arms_edit_gate(None));
}

#[test]
fn turn_escalation_carries_the_router_complexity_band() {
    let easy = resolve_turn_escalation(0, Some(Effort::Smart), None, "fix the typo in the docs");
    assert_eq!(easy.complexity, runtime::RouteTaskComplexity::Trivial);

    let hard = resolve_turn_escalation(
        0,
        Some(Effort::Smart),
        None,
        "scan the whole repo and migrate every module",
    );
    assert_eq!(hard.complexity, runtime::RouteTaskComplexity::Large);
}

fn write_agent_manifest(store: &Path, id: &str, status: &str, created_at: u64) {
    std::fs::write(
        store.join(format!("{id}.json")),
        format!(
            r#"{{
                "agentId":"{id}",
                "name":"{id}",
                "status":"{status}",
                "createdAt":"{created_at}",
                "model":"openai/gpt-5.5-fast"
            }}"#
        ),
    )
    .expect("write agent manifest");
}

#[test]
fn remote_commands_target_only_their_turn_generation() {
    let current_generation = 7;
    assert!(command_targets_turn(
        &AgentCommand::RemoteCancelTurn {
            turn_generation: current_generation,
        },
        current_generation,
    ));
    assert!(command_targets_turn(
        &AgentCommand::RemoteSteer {
            turn_generation: current_generation,
            text: "current".to_string(),
        },
        current_generation,
    ));
    assert!(!command_targets_turn(
        &AgentCommand::RemoteCancelTurn {
            turn_generation: current_generation - 1,
        },
        current_generation,
    ));
    assert!(!command_targets_turn(
        &AgentCommand::RemoteSteer {
            turn_generation: current_generation - 1,
            text: "stale".to_string(),
        },
        current_generation,
    ));
    assert!(command_targets_turn(
        &AgentCommand::CancelTurn,
        current_generation,
    ));
    assert!(command_targets_turn(
        &AgentCommand::Steer("local".to_string()),
        current_generation,
    ));
}

#[test]
fn prelude_steers_are_folded_into_the_turn_input() {
    let input = fold_prelude_steers(
        "Project analysis".to_string(),
        &[
            "also check security".to_string(),
            "check test bottlenecks".to_string(),
        ],
    );

    assert!(input.contains("Project analysis"));
    assert!(input.contains("[Input received during pre-analysis]"));
    assert!(input.contains("- also check security"));
    assert!(input.contains("- check test bottlenecks"));
}

#[test]
fn prelude_steers_leave_input_unchanged_when_empty() {
    let input = fold_prelude_steers("Project analysis".to_string(), &[]);
    assert_eq!(input, "Project analysis");
}

#[test]
fn fanout_context_estimate_includes_system_prompt() {
    let system_prompt = vec!["x".repeat(36_000)];
    let tokens = estimated_fanout_context_tokens(0, &system_prompt);

    assert!(
        tokens >= 8_000,
        "system prompt should count toward fan-out scale"
    );
    // A broad Korean analyze intent surfaces the fan-out shape as a MODEL-LED
    // hint at every delegation-capable effort (CC-style: the model judges task
    // shape from the delegation rubric); the host itself never pre-spawns from
    // the phrase — see `should_host_prespawn`.
    assert_eq!(
        super::super::auto_fanout::build_route_hint(
            "프로젝트 전체 분석해줘",
            Some(Effort::Smart),
            tokens
        )
        .shape,
        super::super::auto_fanout::RouteShape::FanoutParallel,
        "smart + large natural-language analyze intent hints the fan-out shape"
    );
    assert_eq!(
        super::super::auto_fanout::build_route_hint(
            "프로젝트 전체 분석해줘",
            Some(Effort::High),
            tokens
        )
        .shape,
        super::super::auto_fanout::RouteShape::FanoutParallel,
        "below ultracode the fan-out shape is still a model-led nudge"
    );
}

#[test]
fn low_confidence_diagnose_triage_does_not_host_spawn() {
    let low = tools::IntentTriage {
        intent: "unsure vague verify/fix ask".to_string(),
        mode: FanoutMode::Diagnose,
        confidence: 0.12,
    };
    assert_eq!(trusted_triage_mode(Some(&low)), None);
    assert_eq!(
        fanout_branch(trusted_triage_mode(Some(&low)), false),
        FanoutBranch::Fallback,
        "low-confidence diagnose must not launch root-cause agents for a vague ask"
    );

    let high = tools::IntentTriage {
        confidence: 0.82,
        ..low
    };
    assert_eq!(trusted_triage_mode(Some(&high)), Some(FanoutMode::Diagnose));
    assert_eq!(
        fanout_branch(trusted_triage_mode(Some(&high)), false),
        FanoutBranch::Diagnose
    );
}

#[test]
fn fanout_input_includes_session_goal_when_present() {
    let input = fanout_decomposition_input(
        "continue implementation",
        Some("analyze the whole harness and fix agent orchestration"),
    );

    assert!(input.contains("Session goal: analyze the whole harness"));
    assert!(input.contains("User turn:\ncontinue implementation"));
}

#[test]
fn route_reminder_sets_for_model_led_and_clears_for_host_prespawn() {
    let model_led = super::super::auto_fanout::build_route_hint(
        "analyze the whole project",
        Some(Effort::Medium),
        20_000,
    );
    assert!(route_reminder_for_hint(&model_led)
        .expect("model-led broad work should remind the model")
        .contains("route-hint"));

    // A breadth host pre-spawn consumes the turn, so it never also nudges the
    // model (the reminder is suppressed in `model_reminder`).
    let host_led = super::super::auto_fanout::build_route_hint(
        "Use SpawnMultiAgent for this",
        Some(Effort::High),
        0,
    );
    assert!(host_led.should_host_prespawn());
    assert!(host_led.is_breadth());
    assert!(route_reminder_for_hint(&host_led).is_none());

    let solo =
        super::super::auto_fanout::build_route_hint("rename this symbol", Some(Effort::High), 500);
    assert_eq!(solo.shape, super::super::auto_fanout::RouteShape::Solo);
    assert!(route_reminder_for_hint(&solo).is_none());
}

#[test]
fn route_hint_uses_current_turn_not_broad_session_goal() {
    let goal = "analyze the whole codebase and perfectly implement the dynamic harness";
    let user_turn = "phase1 상태만 알려줘";
    let scoped_input = fanout_decomposition_input(user_turn, Some(goal));

    // Folding the broad goal into the gate input would look broad enough to
    // delegate (the old bug)...
    assert_ne!(
        super::super::auto_fanout::build_route_hint(&scoped_input, Some(Effort::High), 1_000).shape,
        super::super::auto_fanout::RouteShape::Solo,
        "the folded goal alone reproduces the old broad classification"
    );
    // ...so the pre-turn route hint must be built from the current user turn
    // only — a narrow status question, which never host-prespawns.
    assert!(
        !super::super::auto_fanout::build_route_hint(user_turn, Some(Effort::High), 1_000)
            .should_host_prespawn(),
        "a narrow follow-up must not host-prespawn even under a broad session goal"
    );
}

#[test]
fn fanout_input_ignores_empty_session_goal() {
    let input = fanout_decomposition_input("continue implementation", Some("   "));
    assert_eq!(input, "continue implementation");
}

#[test]
fn fanout_branch_non_breadth_engages_host_only_for_diagnose() {
    // The ship-blocker invariant: on a non-breadth pre-spawn (ultracode
    // Pipeline/DelegateOne) ONLY a `diagnose` verdict engages the host. A
    // `self_consistency` verdict must NOT spin up a council — it falls back to
    // the model-led turn — and neither does `decompose`, `solo`, or no triage.
    assert_eq!(
        fanout_branch(Some(FanoutMode::Diagnose), false),
        FanoutBranch::Diagnose,
        "non-breadth diagnose engages the host"
    );
    assert_eq!(
        fanout_branch(Some(FanoutMode::SelfConsistency), false),
        FanoutBranch::Fallback,
        "non-breadth self-consistency must stay model-led (the ship-blocker)"
    );
    for mode in [Some(FanoutMode::Solo), Some(FanoutMode::Decompose), None] {
        assert_eq!(
            fanout_branch(mode, false),
            FanoutBranch::Fallback,
            "non-breadth {mode:?} defers to the model-led turn"
        );
    }
}

#[test]
fn fanout_branch_breadth_runs_the_full_ladder() {
    // A breadth fan-out additionally runs self-consistency and decompose, and
    // an absent/decompose triage decomposes; solo still defers, diagnose still
    // diagnoses.
    assert_eq!(
        fanout_branch(Some(FanoutMode::Solo), true),
        FanoutBranch::Fallback
    );
    assert_eq!(
        fanout_branch(Some(FanoutMode::Diagnose), true),
        FanoutBranch::Diagnose
    );
    assert_eq!(
        fanout_branch(Some(FanoutMode::SelfConsistency), true),
        FanoutBranch::SelfConsistency
    );
    assert_eq!(
        fanout_branch(Some(FanoutMode::Decompose), true),
        FanoutBranch::Decompose
    );
    assert_eq!(
        fanout_branch(None, true),
        FanoutBranch::Decompose,
        "a breadth turn with no triage still decomposes"
    );
}

#[test]
fn stop_visible_agents_refreshes_hud_even_when_store_is_already_terminal() {
    let store = unique_temp_dir("terminal-agent-hud-refresh");
    std::fs::create_dir_all(&store).expect("agent store");
    let _guard = EnvPathGuard::set("ZO_AGENT_STORE", &store);
    // Isolate the todo store too: the HUD renders the todo list, and an
    // ambient `ZO_TODO_STORE` (e.g. inherited from a live zo session)
    // would leak unrelated todo text into the snapshot the assertions below
    // scan. `_guard` already holds the process-wide env lock for this test,
    // so set the var directly (a second `EnvPathGuard::set` would re-lock the
    // same non-reentrant mutex and deadlock) and clear it at the end.
    let prior_todo_store = std::env::var_os("ZO_TODO_STORE");
    std::env::set_var("ZO_TODO_STORE", store.join("absent-todos.json"));
    write_agent_manifest(&store, "stale-runner", "stopped", 200);

    let mut app = test_app();
    app.set_agent_manifest_started_after(150);
    app.update_hud_live_snapshot(
        1,
        Vec::new(),
        vec![AgentTaskSummary {
            name: "stale-runner".to_string(),
            status: "running".to_string(),
            model: "openai/gpt-5.5-fast".to_string(),
            elapsed_secs: 60,
            token_history: Vec::new(),
            current_tool: Some("bash".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        }],
        None,
    );

    let before = draw_app_dump(&mut app, 180, 28);
    assert!(
        before.contains("stale-runner") && before.contains("1 agents"),
        "fixture should start with a stale live HUD row:\n{before}"
    );

    let stopped = stop_visible_agents(&mut app);
    assert_eq!(
        stopped, 0,
        "manifest was already terminal; the stop call itself should not close a new agent"
    );

    let after = draw_app_dump(&mut app, 180, 28);
    assert!(
        !after.contains("stale-runner") && !after.contains("agents"),
        "stop request must refresh the HUD from terminal manifests before the next draw:\n{after}"
    );

    // Restore the prior `ZO_TODO_STORE` (still under `_guard`'s lock).
    match prior_todo_store {
        Some(value) => std::env::set_var("ZO_TODO_STORE", value),
        None => std::env::remove_var("ZO_TODO_STORE"),
    }
    let _ = std::fs::remove_dir_all(store);
}

#[test]
fn completed_agent_notice_hides_result_payload() {
    let completion = AgentCompletion {
        agent_id: "agent-decompose".to_string(),
        name: "decompose".to_string(),
        status: "completed".to_string(),
        result: Some(
            r#"{"subtasks":[{"role":"code structure analysis","prompt":"analyze the current project end to end"}]}"#
                .to_string(),
        ),
        structured: None,
        error: None,
        output_tokens: 0,
    };

    let (level, text) = format_agent_completion(&completion);

    assert_eq!(level, SystemLevel::Info);
    assert_eq!(text, "Agent 'decomposition' finished");
    assert!(!text.contains("subtasks"));
    assert!(!text.contains("prompt"));
    assert!(!text.contains("current project"));
}

#[test]
fn stopped_agent_notice_reads_like_cancel_not_failure() {
    let completion = AgentCompletion {
        agent_id: "agent-cli".to_string(),
        name: "cli".to_string(),
        status: "stopped".to_string(),
        result: None,
        structured: None,
        error: Some("cancelled by foreground turn".to_string()),
        output_tokens: 0,
    };

    let (level, text) = format_agent_completion(&completion);

    assert_eq!(level, SystemLevel::Warn);
    assert_eq!(text, "Agent 'cli' stopped: cancelled by foreground turn");
    assert!(!text.contains("failed"));
}

#[test]
fn decompose_collection_window_reap_is_suppressed_not_warned() {
    // Reproduces the reported noise: when the decompose wait elapses while
    // the model still streams, the collection-window reaper marks the
    // internal decompose agent "stopped". That completion must NOT surface
    // as a user-facing warning — the drain sites skip internal agents.
    let reaped = AgentCompletion {
        agent_id: "agent-decompose".to_string(),
        name: "decompose".to_string(),
        status: "stopped".to_string(),
        result: None,
        structured: None,
        error: Some("auto fan-out collection window closed".to_string()),
        output_tokens: 0,
    };
    assert!(
        agent_completion_is_internal(&reaped),
        "the decompose reap is internal plumbing and must be suppressed"
    );

    // A real spawned pre-analysis agent still surfaces normally.
    let visible = AgentCompletion {
        name: "api layer".to_string(),
        ..reaped
    };
    assert!(!agent_completion_is_internal(&visible));
}

#[test]
fn decompose_completion_advances_prelude_activity() {
    let completion = AgentCompletion {
        agent_id: "agent-decompose".to_string(),
        name: "decompose".to_string(),
        status: "completed".to_string(),
        result: None,
        structured: None,
        error: None,
        output_tokens: 0,
    };

    assert_eq!(
        auto_fanout_activity_for_completion(&completion),
        "Smart: preparing parallel pre-analysis agents"
    );
}

#[test]
fn fanout_launch_progress_summarizes_agent_roles() {
    let roles = vec![
        "runtime stream".to_string(),
        "ui status".to_string(),
        "agent store".to_string(),
        "tests".to_string(),
    ];

    assert_eq!(
        summarize_roles(&roles),
        " (runtime stream, ui status, agent store, +1 more)"
    );
    assert_eq!(pluralize("agent", 1), "agent");
    assert_eq!(pluralize("agent", 2), "agents");
}

#[test]
fn fanout_launch_progress_updates_central_progress_block() {
    let roles = vec![
        "runtime stream".to_string(),
        "ui status".to_string(),
        "agent store".to_string(),
        "tests".to_string(),
    ];

    let text = format_fanout_launch_progress_text(&roles);

    assert!(text.contains("Smart pre-analysis: launching · step 1/2 · 0% complete · 100% left"));
    assert!(text.contains("0/4 terminal, 4 queued"));
    assert!(text.contains("agent output pending (waiting for launch)"));
    assert!(text.contains("roles: runtime stream, ui status, agent store, +1 more"));
}

#[test]
fn fanout_launch_activity_reports_zero_progress_immediately() {
    let roles = vec![
        "runtime stream".to_string(),
        "ui status".to_string(),
        "agent store".to_string(),
        "tests".to_string(),
    ];

    let text = format_fanout_launch_activity(&roles);

    assert_eq!(
        text,
        "Smart: 0/4 complete · 0% · 100% left · 4 pre-analysis agents launching"
    );
}

#[test]
fn fanout_progress_text_shows_agents_and_pending_tokens() {
    let agents = vec![AgentTaskSummary {
        name: "CLI TUI".to_string(),
        status: "running".to_string(),
        model: "gpt-5.5".to_string(),
        elapsed_secs: 125,
        token_history: Vec::new(),
        current_tool: Some("bash".to_string()),
        current_phase: None,
        last_activity_at: None,
        ..Default::default()
    }];

    let text = format_fanout_progress_text("running", &agents, 1);

    assert!(text.contains("Smart pre-analysis: running · step 1/2 · 0% complete · 100% left"));
    assert!(text.contains("0/1 terminal, 1 active (1 running)"));
    assert!(text.contains("remaining: waiting for 1 agent result"));
    assert!(text.contains("models: gpt-5.5"));
    assert!(text.contains("agent output pending (waiting for usage)"));
    // Per-agent rows now render in the Claude-Code-style agent tree, not in
    // this summary block — the enumeration must NOT duplicate into the text.
    assert!(
        !text.contains("CLI TUI [running]"),
        "per-agent enumeration moved to the agent tree: {text}"
    );
}

#[test]
fn fanout_progress_text_reports_partial_completion_percentage() {
    let agents = vec![
        AgentTaskSummary {
            name: "runtime".to_string(),
            status: "completed".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 60,
            token_history: vec![120],
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "api".to_string(),
            status: "failed".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 70,
            token_history: vec![80],
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "ui".to_string(),
            status: "running".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 80,
            token_history: Vec::new(),
            current_tool: Some("bash".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "tests".to_string(),
            status: "running".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 90,
            token_history: Vec::new(),
            current_tool: Some("cargo test".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
    ];

    let text = format_fanout_progress_text("running", &agents, 2);

    assert!(text.contains("50% complete"));
    assert!(text.contains("50% left"));
    assert!(text.contains("2/4 terminal, 2 active (2 running), 1 failed"));
    assert!(text.contains("remaining: waiting for 2 agent results"));
    assert!(text.contains("models: gpt-5.5 x4"));
}

#[test]
fn fanout_activity_reports_completion_fraction_and_percentage() {
    let agents = vec![
        AgentTaskSummary {
            name: "runtime".to_string(),
            status: "completed".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 60,
            token_history: vec![120],
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "api".to_string(),
            status: "failed".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 70,
            token_history: vec![80],
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "ui".to_string(),
            status: "running".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 80,
            token_history: Vec::new(),
            current_tool: Some("bash".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
        AgentTaskSummary {
            name: "tests".to_string(),
            status: "running".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 90,
            token_history: Vec::new(),
            current_tool: Some("cargo test".to_string()),
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        },
    ];

    let text = format_live_fanout_activity(&agents, 2, 4);

    assert!(text.contains("Smart: 2/4 complete · 50%"));
    assert!(text.contains("50% left"));
    assert!(text.contains("2 pre-analysis agents active (2 running)"));
}

#[test]
fn fanout_activity_keeps_count_when_agent_rows_are_not_ready() {
    // No manifest yet (workflow None → spawned_total 0): fall back to the running
    // count for the denominator so an early frame still reads 0/4, not 0/0.
    let text = format_live_fanout_activity(&[], 4, 0);

    assert_eq!(
        text,
        "Smart: 0/4 complete · 0% · 100% left · 4 pre-analysis agents active (4 running)"
    );
}

#[test]
fn fanout_progress_text_settles_to_completed_with_real_model_and_tokens() {
    let agents = vec![AgentTaskSummary {
        name: "CLI TUI".to_string(),
        status: "completed".to_string(),
        model: "openai/gpt-5.5-fast".to_string(),
        elapsed_secs: 605,
        token_history: vec![120, 80],
        current_tool: None,
        current_phase: None,
        last_activity_at: None,
        ..Default::default()
    }];

    let text = format_fanout_progress_text("completed", &agents, 0);

    assert!(text.contains("Smart pre-analysis: completed · step 2/2 · 100% complete · 0% left"));
    assert!(text.contains("1/1 terminal, 0 active (0 running)"));
    assert!(text.contains("models: gpt-5.5-fast"));
    assert!(text.contains("~200 agent output tokens"));
    // Per-agent rows live in the agent tree now, not in this summary text.
    assert!(
        !text.contains("CLI TUI [completed]"),
        "per-agent enumeration moved to the agent tree: {text}"
    );
    assert!(
        !text.contains("agent output pending"),
        "terminal rows with token history should not look disconnected from token accounting"
    );
}

#[test]
fn fanout_collection_close_marks_active_rows_stopped_for_visible_snapshot() {
    let mut snapshot = LiveHudSnapshot {
        running: 1,
        todos: Vec::new(),
        agents: vec![AgentTaskSummary {
            name: "root cause".to_string(),
            status: "running".to_string(),
            model: "gpt-5.5".to_string(),
            elapsed_secs: 30,
            token_history: Vec::new(),
            current_tool: Some("grep_search".to_string()),
            current_phase: Some("searching".to_string()),
            last_activity_at: None,
            ..Default::default()
        }],
        workflow: None,
    };

    let phase = close_fanout_collection_snapshot(&mut snapshot);
    let text = format_fanout_progress_text(phase, &snapshot.agents, snapshot.running);

    assert_eq!(phase, "closed");
    assert_eq!(snapshot.running, 0);
    assert_eq!(snapshot.agents[0].status, "stopped");
    assert_eq!(snapshot.agents[0].current_tool, None);
    assert_eq!(
        snapshot.agents[0].current_phase.as_deref(),
        Some("collection window closed")
    );
    assert!(text.contains("Smart pre-analysis: closed · step 2/2 · 100% complete · 0% left"));
    assert!(text.contains("1/1 terminal, 0 active (0 running), 0 failed, 1 stopped"));
    assert!(
        !text.contains("remaining: waiting"),
        "closed pre-analysis must not leave stale waiting text: {text}"
    );
}

#[test]
fn fanout_collection_close_fallback_has_no_running_text() {
    let text = format_fanout_collection_closed_without_snapshot_text();

    assert!(text.contains("Smart pre-analysis: closed · step 2/2 · 100% complete · 0% left"));
    assert!(text.contains("continuing with the main model"));
    assert!(!text.contains("running"));
    assert!(!text.contains("waiting for 1 agent"));
}

#[test]
fn semantic_triage_prelude_defers_tool_call_until_agents_launch() {
    let labels = auto_fanout_prelude_labels(false);
    assert_eq!(labels.tool_label, "SemanticTriage");
    assert_eq!(
        labels.input_summary,
        "Smart prelude · choosing collaboration route"
    );
    assert!(labels.initial_activity.contains("Smart"));
    assert!(!labels.initial_note.contains("SpawnMultiAgent"));
    assert!(
        !auto_fanout_opens_tool_call_immediately(false),
        "semantic-triage-only fallback must not leave a running ToolCall with zero agents"
    );
}

#[test]
fn host_prespawn_prelude_keeps_spawn_multi_agent_label() {
    let labels = auto_fanout_prelude_labels(true);
    assert_eq!(labels.tool_label, "SpawnMultiAgent");
    assert_eq!(
        labels.input_summary,
        "Smart prelude · auto fan-out pre-analysis"
    );
    assert!(labels.initial_activity.contains("Smart"));
    assert!(auto_fanout_opens_tool_call_immediately(true));
}

#[test]
fn semantic_triage_selected_agent_preview_uses_smart_provenance() {
    assert_eq!(
        SMART_TRIAGE_SELECTED_PREVIEW,
        "Smart prelude · semantic triage selected agent pre-analysis"
    );
}

/// Smart prelude renders the Claude-Code-style agent tree in the transcript:
/// a synthetic spawn `ToolCall` opens a batch, the live HUD snapshot fills it
/// with per-agent rows, completions flip `⎿ Done`, and sealing swaps in the
/// `N agents finished` header — exactly like the model-invoked spawn path.

#[test]
fn auto_fanout_renders_agent_tree_in_transcript() {
    let store = unique_temp_dir("auto-fanout-tree-todos");
    std::fs::create_dir_all(&store).expect("store dir");
    // Isolate the todo store so an ambient one cannot leak into the dump.
    let _guard = EnvPathGuard::set("ZO_TODO_STORE", &store.join("absent-todos.json"));

    let mut app = test_app();
    let call_id = "auto-fanout-test";
    // Synthetic spawn ToolCall owns the tree (mirrors maybe_apply_auto_fanout_live).
    app.push_block(RenderBlock::ToolCall {
        id: runtime::message_stream::BlockIdGen::default().next(),
        tool_call_id: ToolCallId(call_id.to_string()),
        name: "SpawnMultiAgent".to_string(),
        summary: String::new(),
        preview: ToolPreview::Generic {
            name: "SpawnMultiAgent".to_string(),
            input_summary: "Smart prelude · auto fan-out pre-analysis".to_string(),
        },
        status: ToolCallStatus::Running,
    });
    app.begin_agent_batch_with_label(call_id, Some(SMART_PRELUDE_LABEL));

    // Live HUD snapshot fills the tree rows (spawn order by created_at). This
    // is the real production entry point (`update_hud_live_snapshot` →
    // `refresh_agent_batch`), so the test exercises the same path the turn loop
    // drives.
    let agent = |id: &str, status: &str, created: u64| AgentTaskSummary {
        id: id.to_string(),
        name: id.to_string(),
        status: status.to_string(),
        model: "haiku".to_string(),
        tool_calls: Some(7),
        tokens: 4_200,
        created_at: Some(created),
        ..AgentTaskSummary::default()
    };
    app.update_hud_live_snapshot(
        2,
        Vec::new(),
        vec![
            agent("runtime", "running", 100),
            agent("ui", "running", 200),
        ],
        None,
    );

    // The transcript side table now owns a CC-style tree for the synthetic
    // spawn call: spawn-order rows with tool/token meta.
    let tree = app
        .transcript_mut()
        .agent_tree(call_id)
        .cloned()
        .expect("auto fan-out must attach an agent tree to its synthetic spawn call");
    assert_eq!(
        tree.batch_label.as_deref(),
        Some(SMART_PRELUDE_LABEL),
        "Smart prelude provenance is carried to the agent tree header"
    );
    assert_eq!(tree.rows.len(), 2, "both agents joined the tree: {tree:?}");
    assert_eq!(tree.rows[0].name, "runtime", "spawn order by created_at");
    assert_eq!(tree.rows[1].name, "ui");
    assert_eq!(tree.rows[0].tool_calls, Some(7), "CC-style tool meta");
    assert!(tree.rows[0].tokens >= 4_200, "CC-style token meta");
    assert!(!tree.finished, "tree is still collecting while agents run");

    app.begin_turn_with_generation(0);
    app.set_turn_activity("Smart: 2 pre-analysis agents running");
    let running = draw_app_dump(&mut app, 120, 36);
    assert!(
        running.contains("Smart running 2 agents"),
        "Smart provenance must render in the live agent tree header:\n{running}"
    );
    assert!(
        running.contains("runtime") && running.contains("ui"),
        "live agent rows must render:\n{running}"
    );

    // A completion flips that row to Done in completion order.
    assert!(app.note_agent_completion_display("ui", "ui", "completed", 9_000));
    let tree = app
        .transcript_mut()
        .agent_tree(call_id)
        .cloned()
        .expect("tree persists");
    let ui_row = tree
        .rows
        .iter()
        .find(|r| r.name == "ui")
        .expect("ui row present");
    assert_eq!(
        ui_row.status, "completed",
        "completion flips the row: {tree:?}"
    );
    assert!(ui_row.done_order.is_some(), "completion order recorded");

    // Sealing the batch marks the tree finished → `N agents finished` header.
    app.finish_agent_batch(call_id);
    let tree = app
        .transcript_mut()
        .agent_tree(call_id)
        .cloned()
        .expect("tree persists after seal");
    assert!(
        tree.finished,
        "sealing flips the tree to the finished header state: {tree:?}"
    );
}

#[test]
fn fanout_evidence_instructs_consume_not_rederive() {
    let analysis = "### runtime stream\nThe stream loop coalesces draws to 30fps.";
    let evidence = build_fanout_evidence(analysis, 3);

    // Consume-not-rederive: the new instruction frames the pre-analysis as a
    // ready evidence base, never as preliminary work to re-derive.
    assert!(
        evidence.contains("evidence base"),
        "evidence must frame the pre-analysis as an evidence base: {evidence}"
    );
    assert!(
        !evidence.contains("verify and synthesize"),
        "evidence must not tell the model to re-derive the pre-analysis: {evidence}"
    );
    // The per-role analysis sections are carried verbatim...
    assert!(
        evidence.contains(analysis),
        "evidence must carry the per-role analysis sections: {evidence}"
    );
    assert!(evidence.contains("### runtime stream"));
    assert!(evidence.contains("3 independent subtasks"));
    // ...but the evidence body never prepends the user input (that now lives
    // in its own user message, outside this clearable tool-result).
    assert!(
        evidence.starts_with("[Smart pre-analysis]"),
        "evidence must not prepend the user input: {evidence}"
    );
}

#[test]
fn self_consistency_surface_copy_uses_smart_wording() {
    assert_eq!(
        SMART_SELF_CONSISTENCY_ACTIVITY,
        "Smart: self-consistency vote completed"
    );
    assert_eq!(
        SMART_SELF_CONSISTENCY_NOTE,
        "Smart self-consistency: independent answers reconciled by majority vote"
    );
}

#[test]
fn self_consistency_evidence_instructs_consume_not_rederive() {
    let answer = "The root cause is a stale cache key. (self-consistency: 3 agreed)";
    let evidence = build_self_consistency_evidence(answer);

    // Same consume-not-rederive contract as decompose fan-out evidence.
    assert!(
        evidence.contains("evidence base"),
        "must frame the reconciled vote as an evidence base: {evidence}"
    );
    assert!(
        !evidence.contains("verify and synthesize"),
        "must not tell the model to re-derive the reconciled vote: {evidence}"
    );
    // The reconciled answer is carried verbatim, with no user-input prepend.
    assert!(
        evidence.contains(answer),
        "must carry the reconciled answer: {evidence}"
    );
    assert!(
        evidence.starts_with("[Smart self-consistency]"),
        "must label the self-consistency path and not prepend user input: {evidence}"
    );
}

// ---------------------------------------------------------------------------
// Wire-safety guard for the synthetic evidence injection (Option A).
// Exercises `fanout_evidence_injection_is_wire_safe` directly on the three
// session shapes the bug report identified:
//   (a) empty session          → must NOT inject (would lead with assistant)
//   (b) last message assistant → must NOT inject (two consecutive assistant)
//   (c) last message user/tool → MUST inject    (valid alternation)
// ---------------------------------------------------------------------------

#[test]
fn fanout_evidence_pair_rolls_back_when_persist_fails() {
    let mut session = runtime::Session::new();
    session
        .push_message(runtime::ConversationMessage::user_text("user turn"))
        .expect("push user");
    let original_len = session.messages.len();
    let original_updated_at_ms = session.updated_at_ms;
    let tool_use = runtime::ConversationMessage::assistant(vec![runtime::ContentBlock::ToolUse {
        id: "auto_fanout_test".to_string(),
        name: "SpawnMultiAgent".to_string(),
        input: r#"{"reason":"parallel pre-analysis"}"#.to_string(),
    }]);
    let tool_result = runtime::ConversationMessage::tool_result(
        "auto_fanout_test",
        "SpawnMultiAgent",
        "evidence",
        false,
    );

    let error = push_synthetic_fanout_evidence_pair_with_persist(
        &mut session,
        tool_use,
        tool_result,
        |_| Err(runtime::SessionError::Format("persist failed".to_string())),
    )
    .expect_err("persist failure should propagate");

    assert!(error.to_string().contains("persist failed"));
    assert_eq!(
        session.messages.len(),
        original_len,
        "synthetic assistant(tool_use) must not remain orphaned after tool_result/persist failure"
    );
    assert_eq!(
        session.updated_at_ms, original_updated_at_ms,
        "rollback must restore session timestamp metadata"
    );
}

#[test]
fn fanout_injection_is_unsafe_for_empty_session() {
    // Case (a): empty session — the synthetic assistant(tool_use) would be the
    // very first wire message, which Anthropic rejects.
    let session = runtime::Session::new();
    assert!(
        !fanout_evidence_injection_is_wire_safe(&session),
        "empty session must not allow injection: leading assistant is invalid"
    );
}

#[test]
fn fanout_injection_is_unsafe_when_last_message_is_assistant() {
    // Case (b): last message is an assistant turn — a second assistant message
    // would create two consecutive assistant-wire roles, which Anthropic rejects.
    let mut session = runtime::Session::new();
    session
        .push_message(runtime::ConversationMessage::user_text("first user turn"))
        .expect("push user");
    session
        .push_message(runtime::ConversationMessage::assistant(vec![
            runtime::ContentBlock::Text {
                text: "assistant reply".to_string(),
            },
        ]))
        .expect("push assistant");
    assert!(
        !fanout_evidence_injection_is_wire_safe(&session),
        "session ending with assistant must not allow injection: would produce consecutive assistant messages"
    );
}

#[test]
fn fanout_injection_is_safe_when_last_message_is_user() {
    // Case (c-user): last message is a user turn — the synthetic assistant
    // follows correctly in the user→assistant alternation.
    let mut session = runtime::Session::new();
    session
        .push_message(runtime::ConversationMessage::user_text("user turn"))
        .expect("push user");
    assert!(
        fanout_evidence_injection_is_wire_safe(&session),
        "session ending with user message must allow injection"
    );
}

#[test]
fn fanout_injection_is_safe_when_last_message_is_tool_result() {
    // Case (c-tool): last message is a tool_result (stored role `Tool`, wire
    // role `user`) — valid user-wire before the new assistant(tool_use).
    let mut session = runtime::Session::new();
    session
        .push_message(runtime::ConversationMessage::user_text("first turn"))
        .expect("push user");
    session
        .push_message(runtime::ConversationMessage::assistant(vec![
            runtime::ContentBlock::ToolUse {
                id: "tu-1".to_string(),
                name: "Bash".to_string(),
                input: r#"{"command":"ls"}"#.to_string(),
            },
        ]))
        .expect("push assistant tool_use");
    session
        .push_message(runtime::ConversationMessage::tool_result(
            "tu-1", "Bash", "file.txt", false,
        ))
        .expect("push tool_result");
    assert!(
        fanout_evidence_injection_is_wire_safe(&session),
        "session ending with tool_result (wire role user) must allow injection"
    );
}

#[test]
fn fanout_counter_uses_fixed_spawned_total_not_live_list() {
    // Regression (HUD "0/4 → 0/3"): `list_running_agents_since` drops a terminal
    // agent after a grace window, so the live list shrinks as agents finish.
    // Deriving the denominator from `agents.len()` made the counter read 0/4 → 0/3
    // — the finished agent left the live list before it was ever counted as done —
    // instead of 1/4. With the fixed `spawned_total` (workflow manifest agent_ids)
    // the counter is monotonic: completed = spawned_total - running, denominator
    // fixed. `agents` empty exercises the fixed-vs-fallback path without building
    // summaries (a short live list IS exactly the regression condition).
    assert!(
        format_live_fanout_activity(&[], 4, 4).contains("0/4 complete"),
        "4 spawned, 4 running → 0/4"
    );
    assert!(
        format_live_fanout_activity(&[], 3, 4).contains("1/4 complete"),
        "1 finished (3 still running) must read 1/4, never 0/3"
    );
    assert!(
        format_live_fanout_activity(&[], 1, 4).contains("3/4 complete"),
        "3 finished (1 running) → 3/4"
    );
    assert!(
        format_live_fanout_activity(&[], 0, 4).contains("4/4 complete"),
        "all finished → 4/4 finalizing"
    );
}

#[tokio::test]
async fn wait_until_aborted_pends_until_flag_set_then_resolves() {
    use std::time::Duration;
    // The cancel primitive the spawned turn task races against the turn future:
    // when a Ctrl+C sets the shared abort flag, this resolves, the turn branch is
    // dropped (instant stream-kill), and the task hands its runtime back. It must
    // stay pending while the flag is clear, or every turn would self-cancel at once.
    let signal = runtime::HookAbortSignal::new();
    assert!(
        tokio::time::timeout(Duration::from_millis(80), wait_until_aborted(&signal))
            .await
            .is_err(),
        "must stay pending while the abort flag is clear"
    );
    // Once the flag is set it must resolve within a couple of poll cadences, so a
    // cancel is felt promptly rather than only at the next turn boundary.
    signal.abort();
    assert!(
        tokio::time::timeout(Duration::from_millis(300), wait_until_aborted(&signal))
            .await
            .is_ok(),
        "must resolve promptly once the abort flag is set"
    );
}

// The HUD/git-status snapshots must run on a runtime whose blocking pool is
// independent of the main tool pool, so a burst of slow tools that saturates
// the main pool can never starve the render loop's snapshot polling (the
// "SSH/MCP query freezes the UI" bug). Prove independence: saturate THIS
// runtime's blocking pool with parked workers, then confirm a HUD-runtime
// blocking task still completes promptly.
#[test]
fn hud_runtime_is_independent_of_a_saturated_main_pool() {
    use std::sync::mpsc as std_mpsc;
    use std::time::Duration;

    // A tiny main runtime with only 2 blocking threads, fully occupied by two
    // tasks that park until released — mimicking slow SSH/MCP tools pinning the
    // pool. Anything else queued on THIS pool would block indefinitely.
    let main_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .expect("main rt");

    let (release_tx, release_rx) = std_mpsc::channel::<()>();
    let release_rx = std::sync::Arc::new(std::sync::Mutex::new(release_rx));
    for _ in 0..2 {
        let rx = std::sync::Arc::clone(&release_rx);
        main_rt.spawn_blocking(move || {
            // Park until the test releases us, holding a main-pool worker.
            let _ = rx.lock().unwrap().recv();
        });
    }
    // Give the parking tasks a moment to occupy both workers.
    std::thread::sleep(Duration::from_millis(50));

    // A HUD-runtime blocking task must still finish quickly despite the main
    // pool being saturated — it has its own pool.
    let done = super::session_hud::hud_runtime().block_on(async {
        tokio::time::timeout(
            Duration::from_secs(2),
            super::session_hud::hud_runtime().spawn_blocking(|| 7_u8 + 35),
        )
        .await
    });

    // Release the parked main-pool workers regardless of outcome.
    let _ = release_tx.send(());
    let _ = release_tx.send(());

    let value = done
        .expect("HUD task must not time out while the main pool is saturated")
        .expect("HUD blocking task joins");
    assert_eq!(value, 42, "HUD runtime ran the task on its own pool");
}

// ── Phase 4 verdict channel — source #1: deep-gate VERIFY (turn_controller) ──

fn minimal_turn_summary() -> TurnSummary {
    TurnSummary {
        assistant_messages: Vec::new(),
        tool_results: Vec::new(),
        prompt_cache_events: Vec::new(),
        iterations: 1,
        usage: runtime::TokenUsage::default(),
        turn_output_tokens: 0,
        auto_compaction: None,
        microcompact: None,
        deep_verification: None,
        verification_issues: Vec::new(),
        deep_verifier_parse: None,
        deep_verifier_model: None,
        budget_exhausted: None,
    }
}

fn read_outcomes_at(cwd: &Path) -> Vec<runtime::RouteOutcomeRecord> {
    runtime::read_route_outcomes(cwd).unwrap_or_default()
}

#[test]
fn deep_verdict_records_nothing_when_no_verify_leg_ran() {
    let state_dir = unique_temp_dir("deep-verdict-no-leg");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    let _guard = EnvPathGuard::set("ZO_STATE_DIR", &state_dir);
    let cwd = std::env::current_dir().expect("cwd");

    // `deep_verifier_parse: None` — a no-edit chat turn, or deep-gate off.
    let outcome = TurnOutcome {
        summary: Some(minimal_turn_summary()),
    };
    record_deep_verdict_outcomes_for("claude-opus-4-8", &cwd, &outcome);

    assert!(
        read_outcomes_at(&cwd).is_empty(),
        "no VERIFY leg ran this turn — nothing may be recorded"
    );
}

#[test]
fn deep_verdict_records_nothing_when_the_turn_was_cancelled() {
    let state_dir = unique_temp_dir("deep-verdict-cancelled");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    let _guard = EnvPathGuard::set("ZO_STATE_DIR", &state_dir);
    let cwd = std::env::current_dir().expect("cwd");

    let outcome = TurnOutcome { summary: None };
    record_deep_verdict_outcomes_for("claude-opus-4-8", &cwd, &outcome);

    assert!(read_outcomes_at(&cwd).is_empty());
}

#[test]
fn deep_verdict_records_only_the_leg_when_the_verdict_is_ambiguous() {
    let state_dir = unique_temp_dir("deep-verdict-ambiguous");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    let _guard = EnvPathGuard::set("ZO_STATE_DIR", &state_dir);
    let cwd = std::env::current_dir().expect("cwd");

    let mut summary = minimal_turn_summary();
    // The VERIFY leg ran but the streaming sub-turn itself failed
    // (`verify_leg_failed_verdict`), so `deep_verification` stays `None` even
    // though a leg attempt happened.
    summary.deep_verifier_parse = Some(decision_core::deep_lane::VerifierParse::Timeout);
    summary.deep_verifier_model = Some("verifier-model".to_string());
    summary.deep_verification = None;
    let outcome = TurnOutcome { summary: Some(summary) };
    record_deep_verdict_outcomes_for("claude-opus-4-8", &cwd, &outcome);

    let outcomes = read_outcomes_at(&cwd);
    assert_eq!(
        outcomes.len(),
        1,
        "the leg's own did-run record is still written even on an ambiguous verdict"
    );
    assert_eq!(outcomes[0].route_key, "deep-verify:leg");
    assert_eq!(outcomes[0].selected_model, "verifier-model");
    assert_eq!(outcomes[0].status, "failed");
    assert_eq!(
        outcomes[0].signal, None,
        "the leg's did-run record is never itself a `signal:\"verdict\"` record"
    );
}

#[test]
fn deep_verdict_records_both_the_main_turn_and_the_leg_when_usable() {
    let state_dir = unique_temp_dir("deep-verdict-usable-pass");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    let _guard = EnvPathGuard::set("ZO_STATE_DIR", &state_dir);
    let cwd = std::env::current_dir().expect("cwd");

    let mut summary = minimal_turn_summary();
    summary.deep_verifier_parse = Some(decision_core::deep_lane::VerifierParse::Json);
    summary.deep_verifier_model = Some("verifier-model".to_string());
    summary.deep_verification = Some(true);
    let outcome = TurnOutcome { summary: Some(summary) };
    record_deep_verdict_outcomes_for("claude-opus-4-8", &cwd, &outcome);

    let outcomes = read_outcomes_at(&cwd);
    assert_eq!(outcomes.len(), 2, "both the main-turn verdict and the leg did-run record are written");

    let turn = outcomes
        .iter()
        .find(|record| record.route_key == "main:turn")
        .expect("main:turn record");
    assert_eq!(turn.selected_model, "claude-opus-4-8");
    assert_eq!(turn.status, "completed");
    assert_eq!(turn.signal.as_deref(), Some("verdict"));

    let leg = outcomes
        .iter()
        .find(|record| record.route_key == "deep-verify:leg")
        .expect("deep-verify:leg record");
    assert_eq!(leg.selected_model, "verifier-model");
    assert_eq!(leg.status, "completed");
    assert_eq!(leg.signal, None);
}

#[test]
fn deep_verdict_records_a_rejected_main_turn_as_failed() {
    let state_dir = unique_temp_dir("deep-verdict-usable-fail");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    let _guard = EnvPathGuard::set("ZO_STATE_DIR", &state_dir);
    let cwd = std::env::current_dir().expect("cwd");

    let mut summary = minimal_turn_summary();
    summary.deep_verifier_parse = Some(decision_core::deep_lane::VerifierParse::Salvaged);
    // No cross-model verifier installed this turn — falls back to the main model.
    summary.deep_verifier_model = None;
    summary.deep_verification = Some(false);
    let outcome = TurnOutcome { summary: Some(summary) };
    record_deep_verdict_outcomes_for("claude-opus-4-8", &cwd, &outcome);

    let outcomes = read_outcomes_at(&cwd);
    let turn = outcomes
        .iter()
        .find(|record| record.route_key == "main:turn")
        .expect("main:turn record");
    assert_eq!(turn.status, "failed");
    let leg = outcomes
        .iter()
        .find(|record| record.route_key == "deep-verify:leg")
        .expect("deep-verify:leg record");
    assert_eq!(
        leg.selected_model, "claude-opus-4-8",
        "no cross-model verifier installed — the leg ran on the native main model"
    );
}

#[test]
fn interactive_turn_budgets_default_env_override_and_disable() {
    let _lock = crate::test_env_lock();
    let restore = |key: &str, prev: Option<std::ffi::OsString>| match prev {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    };
    let prev_secs = std::env::var_os("ZO_TURN_DEADLINE_SECS");
    let prev_tokens = std::env::var_os("ZO_TURN_OUTPUT_TOKEN_BUDGET");
    let prev_input = std::env::var_os("ZO_TURN_INPUT_TOKEN_BUDGET");

    // Defaults (unset): all three bounds ON at the documented defaults.
    std::env::remove_var("ZO_TURN_DEADLINE_SECS");
    std::env::remove_var("ZO_TURN_OUTPUT_TOKEN_BUDGET");
    std::env::remove_var("ZO_TURN_INPUT_TOKEN_BUDGET");
    let (deadline, tokens, input) = interactive_turn_budgets();
    assert_eq!(
        deadline,
        Some(std::time::Duration::from_secs(
            runtime::DEFAULT_TURN_DEADLINE_SECS
        ))
    );
    assert_eq!(tokens, Some(runtime::DEFAULT_TURN_OUTPUT_TOKEN_BUDGET));
    assert_eq!(input, Some(runtime::DEFAULT_TURN_INPUT_TOKEN_BUDGET));

    // Explicit override.
    std::env::set_var("ZO_TURN_DEADLINE_SECS", "120");
    std::env::set_var("ZO_TURN_OUTPUT_TOKEN_BUDGET", "50000");
    std::env::set_var("ZO_TURN_INPUT_TOKEN_BUDGET", "700000");
    let (deadline, tokens, input) = interactive_turn_budgets();
    assert_eq!(deadline, Some(std::time::Duration::from_secs(120)));
    assert_eq!(tokens, Some(50_000));
    assert_eq!(input, Some(700_000));

    // `0` disables each bound (unbounded).
    std::env::set_var("ZO_TURN_DEADLINE_SECS", "0");
    std::env::set_var("ZO_TURN_OUTPUT_TOKEN_BUDGET", "0");
    std::env::set_var("ZO_TURN_INPUT_TOKEN_BUDGET", "0");
    let (deadline, tokens, input) = interactive_turn_budgets();
    assert_eq!(deadline, None, "0 secs disables the wall-clock bound");
    assert_eq!(tokens, None, "0 tokens disables the output-token bound");
    assert_eq!(input, None, "0 tokens disables the input-token bound");

    // Garbage falls back to the default, never silently disabling the net.
    std::env::set_var("ZO_TURN_DEADLINE_SECS", "not-a-number");
    let (deadline, _, _) = interactive_turn_budgets();
    assert_eq!(
        deadline,
        Some(std::time::Duration::from_secs(
            runtime::DEFAULT_TURN_DEADLINE_SECS
        )),
        "an unparseable value must fall back to the default, not disable"
    );

    restore("ZO_TURN_DEADLINE_SECS", prev_secs);
    restore("ZO_TURN_OUTPUT_TOKEN_BUDGET", prev_tokens);
    restore("ZO_TURN_INPUT_TOKEN_BUDGET", prev_input);
}
