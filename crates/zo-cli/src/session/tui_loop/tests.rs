use super::*;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use commands::SlashCommand;
use ratatui::backend::{Backend, TestBackend};
use ratatui::layout::Position;
use ratatui::{Terminal, TerminalOptions, Viewport};
use runtime::{ConversationMessage, MessageRole, Session, TokenUsage};
use zo_cli::tui::AppMode;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn remote_title_prefers_session_name_and_preserves_unnamed_projection() {
    let cwd = Path::new("/work/zo");
    assert_eq!(remote_session_title(cwd, None), "Zo · zo");
    assert_eq!(
        remote_session_title(cwd, Some("  배포 관찰  ")),
        "Zo · 배포 관찰"
    );
}

struct CurrentDirGuard {
    original: PathBuf,
    _lock: MutexGuard<'static, ()>,
}

impl CurrentDirGuard {
    fn enter(path: &Path) -> Self {
        let lock = cwd_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let original = env::current_dir().expect("cwd should exist");
        env::set_current_dir(path).expect("set current dir");
        Self {
            original,
            _lock: lock,
        }
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original);
    }
}

struct ApiKeyGuard {
    previous: Option<std::ffi::OsString>,
}

impl ApiKeyGuard {
    fn set_dummy() -> Self {
        let previous = env::var_os("ANTHROPIC_API_KEY");
        env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-tui-loop");
        Self { previous }
    }
}

impl Drop for ApiKeyGuard {
    fn drop(&mut self) {
        if let Some(value) = self.previous.take() {
            env::set_var("ANTHROPIC_API_KEY", value);
        } else {
            env::remove_var("ANTHROPIC_API_KEY");
        }
    }
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &Path) -> Self {
        let lock = crate::test_env_lock();
        let previous = env::var_os(key);
        env::set_var(key, value);
        Self {
            key,
            previous,
            _lock: lock,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.previous.take() {
            env::set_var(self.key, value);
        } else {
            env::remove_var(self.key);
        }
    }
}

/// Build the standard test `LiveCli`. The caller must ALREADY hold the crate
/// env lock (via `crate::test_env_lock()` or a live [`EnvVarGuard`]), acquired
/// BEFORE any [`CurrentDirGuard`] per the canonical env→cwd lock order:
/// `LiveCli::new` reads the process-global `ZO_CONFIG_HOME` (and
/// auto-installs bundled plugins into it), so an unlocked build races tests
/// that swap the config home — while locking here instead would self-deadlock
/// callers that hold the non-reentrant lock through a guard.
fn test_live_cli_with_env_lock_held() -> LiveCli {
    LiveCli::new(
        "sonnet".to_string(),
        true,
        None,
        runtime::PermissionMode::ReadOnly,
    )
    .expect("live cli should build")
}

fn unique_temp_dir(label: &str) -> PathBuf {
    // Keep test sessions out of the developer's real ~/.zo (see
    // isolate_global_zo_home_for_tests).
    crate::isolate_global_zo_home_for_tests();
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_millis();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "zo-tui-loop-{label}-{}-{millis}-{counter}",
        std::process::id()
    ))
}

fn cwd_lock() -> &'static Mutex<()> {
    crate::test_cwd_lock()
}

fn new_test_app() -> App {
    let (_block_tx, block_rx) = mpsc::channel::<RenderBlock>(16);
    let (cmd_tx, _cmd_rx) = mpsc::channel::<AgentCommand>(16);
    App::new(Theme::no_color(), block_rx, cmd_tx)
}

#[test]
fn session_start_payload_contains_source_cwd_and_session_id() {
    let payload = session_start_payload("session-123", Some("/tmp/workspace"));

    assert_eq!(payload["source"], "startup");
    assert_eq!(payload["cwd"], "/tmp/workspace");
    assert_eq!(payload["session_id"], "session-123");
}

#[test]
fn session_start_payload_keeps_cwd_key_when_unavailable() {
    let payload = session_start_payload("session-123", None);

    assert!(payload.get("cwd").is_some());
    assert!(payload["cwd"].is_null());
    assert_eq!(payload["session_id"], "session-123");
}

#[test]
fn session_end_payload_contains_session_id_and_reason_values() {
    let exit_payload = session_end_payload("session-123", "exit");
    let error_payload = session_end_payload("session-456", "error");

    assert_eq!(exit_payload["session_id"], "session-123");
    assert_eq!(exit_payload["reason"], "exit");
    assert_eq!(error_payload["session_id"], "session-456");
    assert_eq!(error_payload["reason"], "error");
}

#[test]
fn remote_submissions_never_dispatch_local_slash_commands() {
    for input in [
        "/exit",
        "/quit",
        "/remote approve 123456",
        "/permissions default",
    ] {
        assert!(!dispatches_local_slash(SubmissionOrigin::Remote, input));
        assert!(dispatches_local_slash(SubmissionOrigin::Local, input));
    }
    assert!(!dispatches_local_slash(
        SubmissionOrigin::Local,
        "ordinary prompt"
    ));
}

fn render_app_buffer(app: &mut App) -> String {
    let backend = TestBackend::new(220, 60);
    let mut terminal = Terminal::new(backend).expect("test backend");
    app.draw(&mut terminal).expect("draw");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect::<String>()
}

fn turn_summary_with_tools(tool_names: &[&str]) -> TurnSummary {
    TurnSummary {
        assistant_messages: Vec::new(),
        tool_results: tool_names
            .iter()
            .map(|tool_name| ConversationMessage {
                role: MessageRole::User,
                blocks: vec![ContentBlock::ToolResult {
                    tool_use_id: format!("tool-{tool_name}"),
                    tool_name: (*tool_name).to_string(),
                    output: String::new(),
                    is_error: false,
                    images: Vec::new(),
                }],
                usage: None,
                thought_signature: None,
                reasoning_replay: None,
                            model: None,
            })
            .collect(),
        prompt_cache_events: Vec::new(),
        iterations: 1,
        usage: TokenUsage::default(),
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

#[test]
fn auto_compaction_reseed_drops_old_transcript_blocks_and_cache() {
    let mut app = new_test_app();
    let ids = BlockIdGen::default();

    for index in 0..12 {
        app.push_block(RenderBlock::TextDelta {
            id: ids.next(),
            text: format!("old transcript block {index}"),
            done: true,
        });
    }
    let before_blocks = app.transcript_mut().len();
    assert_eq!(app.transcript_mut().rendered_cache_len(), before_blocks);

    let mut session = Session::new();
    session.messages = std::sync::Arc::new(vec![
        ConversationMessage::user_text("preserved user prompt"),
        ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "preserved assistant answer".to_string(),
        }]),
    ]);

    reseed_transcript_after_auto_compaction(&mut app, &ids, &session, 10);

    let after_blocks = app.transcript_mut().len();
    assert!(
        after_blocks < before_blocks,
        "compaction reseed should shrink transcript: before={before_blocks}, after={after_blocks}"
    );
    assert_eq!(app.transcript_mut().rendered_cache_len(), after_blocks);

    let rendered = render_app_buffer(&mut app);
    assert!(
        rendered.contains("preserved user prompt"),
        "rendered: {rendered}"
    );
    assert!(
        rendered.contains("preserved assistant answer"),
        "rendered: {rendered}"
    );
    assert!(
        rendered.contains("Compacted conversation · 10 messages summarized"),
        "rendered: {rendered}"
    );
    assert!(
        !rendered.contains("old transcript block"),
        "old blocks must not survive reseed: {rendered}"
    );
}

#[test]
fn inline_auto_compaction_appends_only_the_notice() {
    let mut app = new_test_app();
    app.set_terminal_mode(TerminalMode::Inline);
    let ids = BlockIdGen::default();
    app.push_block(RenderBlock::TextDelta {
        id: ids.next(),
        text: "already emitted turn".to_string(),
        done: true,
    });
    app.finalize_inline_transcript();

    let mut session = Session::new();
    session.messages = std::sync::Arc::new(vec![
        ConversationMessage::user_text("historical user prompt"),
        ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "historical assistant answer".to_string(),
        }]),
    ]);

    reseed_transcript_after_auto_compaction(&mut app, &ids, &session, 10);

    let blocks = app.transcript_mut().blocks();
    assert_eq!(blocks.len(), 1, "inline must not reseed append-only history");
    assert!(matches!(
        &blocks[0],
        RenderBlock::System { text, .. }
            if text.contains("Compacted conversation · 10 messages summarized")
    ));
}

#[test]
fn inline_shutdown_flush_does_not_redraw_a_nonzero_viewport() {
    let mut app = new_test_app();
    app.set_terminal_mode(TerminalMode::Inline);
    app.enable_input();

    let mut backend = TestBackend::new(110, 32);
    backend
        .set_cursor_position(Position::new(0, 10))
        .expect("position inline viewport below existing output");
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(zo_cli::tui::INLINE_VIEWPORT_HEIGHT),
        },
    )
    .expect("inline terminal");
    app.draw(&mut terminal).expect("initial viewport draw");
    let frame_count_before_shutdown = terminal.get_frame().count();

    app.push_block(RenderBlock::System {
        id: BlockIdGen::default().next(),
        level: SystemLevel::Info,
        text: "late finalized shutdown output".to_string(),
    });
    let mut result = Ok(());
    flush_inline_transcript_at_shutdown(&mut app, &mut terminal, &mut result);

    assert!(result.is_ok(), "shutdown flush failed: {result:?}");
    assert_eq!(
        terminal.get_frame().count(),
        frame_count_before_shutdown,
        "shutdown must emit scrollback without composing another viewport frame"
    );
    assert!(
        terminal.get_frame().area().y > 0,
        "regression requires an absolute nonzero viewport origin"
    );
    let backend = terminal.backend();
    let text = backend
        .scrollback()
        .content()
        .iter()
        .chain(backend.buffer().content().iter())
        .map(ratatui::buffer::Cell::symbol)
        .collect::<String>();
    assert!(text.contains("late finalized shutdown output"), "{text}");
}

#[test]
fn agent_auth_failure_render_is_actionable_and_compact() {
    let mut app = new_test_app();
    let ids = BlockIdGen::default();
    let completion = AgentCompletion {
        agent_id: "agent-auth".to_string(),
        name: "parallel-agent-2".to_string(),
        status: "failed".to_string(),
        result: None,
        structured: None,
        error: Some(
            "api returned 401 Unauthorized (authentication_error): Invalid authentication credentials"
                .to_string(),
        ),
        output_tokens: 0,
    };

    assert!(agent_completion_is_auth_failure(&completion));
    push_agent_completion(&mut app, &ids, &completion);

    let rendered = render_app_buffer(&mut app);
    assert!(
        rendered.contains("parallel-agent-2"),
        "rendered: {rendered}"
    );
    assert!(rendered.contains("auth failed"), "rendered: {rendered}");
    assert!(
        rendered.contains("ZO_AGENT_MODEL"),
        "rendered: {rendered}"
    );
    assert!(
        !rendered.contains("Invalid authentication credentials"),
        "rendered should hide low-level credential spam: {rendered}"
    );
}

#[test]
fn completed_agent_render_hides_result_payload() {
    let mut app = new_test_app();
    let ids = BlockIdGen::default();
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

    push_agent_completion(&mut app, &ids, &completion);

    let rendered = render_app_buffer(&mut app);
    assert!(
        rendered.contains("Agent 'decomposition' finished"),
        "rendered: {rendered}"
    );
    assert!(!rendered.contains("subtasks"), "rendered: {rendered}");
    assert!(!rendered.contains("prompt"), "rendered: {rendered}");
    assert!(
        !rendered.contains("current project"),
        "rendered: {rendered}"
    );
}

#[test]
fn agent_rate_limit_failure_render_is_actionable_and_compact() {
    let mut app = new_test_app();
    let ids = BlockIdGen::default();
    let completion = AgentCompletion {
        agent_id: "agent-rate".to_string(),
        name: "tools".to_string(),
        status: "failed".to_string(),
        result: None,
        structured: None,
        error: Some(
            "api failed after 6 attempts: api returned 429 Too Many Requests (rate_limit_error): Error"
                .to_string(),
        ),
        output_tokens: 0,
    };

    assert!(agent_completion_is_rate_limit_failure(&completion));
    push_agent_completion(&mut app, &ids, &completion);

    let rendered = render_app_buffer(&mut app);
    assert!(rendered.contains("tools"), "rendered: {rendered}");
    assert!(rendered.contains("rate limited"), "rendered: {rendered}");
    assert!(
        rendered.contains("throttled rapid requests") || rendered.contains("run fewer agents"),
        "rendered: {rendered}"
    );
    assert!(
        !rendered.contains("api failed after 6 attempts"),
        "rendered should hide low-level retry spam: {rendered}"
    );
}

#[test]
fn todo_items_reads_zo_todo_store_as_checklist() {
    let temp_dir = unique_temp_dir("todos");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let store_path = temp_dir.join(".zo-todos.json");
    fs::write(
        &store_path,
        r#"[
          {"stepId":"finish-sidebar","content":"Finish sidebar","activeForm":"Finishing sidebar","status":"in_progress"},
          {"content":"Run checks","activeForm":"Running checks","status":"pending"},
          {"content":"Ship","activeForm":"Shipping","status":"completed"}
        ]"#,
    )
    .expect("write todo store");

    // Pin the store path so reader and writer agree without depending on the
    // process cwd — `primary_store` maps a cwd through the project-state dir, not
    // to `cwd/.zo-todos.json`, so the cwd-relative form never matched. The
    // guard holds the shared lock, keeping this parallel-safe.
    let _env = EnvVarGuard::set("ZO_TODO_STORE", &store_path);
    let items = todo_items();

    assert_eq!(items.len(), 3);
    assert_eq!(items[0].content, "Finish sidebar");
    assert_eq!(items[0].step_id.as_deref(), Some("finish-sidebar"));
    assert!(matches!(items[0].status, TodoChecklistStatus::InProgress));
    assert_eq!(items[1].step_id, None, "legacy rows remain uncorrelated");
    assert!(matches!(items[1].status, TodoChecklistStatus::Pending));
    assert!(matches!(items[2].status, TodoChecklistStatus::Completed));

    fs::remove_dir_all(temp_dir).ok();
}

#[test]
fn todo_items_orders_active_work_before_completed_items() {
    let temp_dir = unique_temp_dir("todos-order");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let store_path = temp_dir.join(".zo-todos.json");
    fs::write(
        &store_path,
        r#"[
          {"stepId":"done","content":"Already done","activeForm":"Already done","status":"completed"},
          {"stepId":"pending-2","content":"Second pending","activeForm":"Second pending","status":"pending"},
          {"stepId":"current","content":"Current work","activeForm":"Doing current work","status":"in_progress"},
          {"stepId":"pending-1","content":"First pending","activeForm":"First pending","status":"pending"}
        ]"#,
    )
    .expect("write todo store");

    // Pin the store path (see `todo_items_reads_zo_todo_store_as_checklist`):
    // a cwd-relative store never resolved to where the writer wrote.
    let _env = EnvVarGuard::set("ZO_TODO_STORE", &store_path);
    let items = todo_items();

    assert_eq!(items.len(), 4);
    assert_eq!(items[0].content, "Current work");
    assert_eq!(items[0].step_id.as_deref(), Some("current"));
    assert!(matches!(items[0].status, TodoChecklistStatus::InProgress));
    assert_eq!(items[1].content, "Second pending");
    assert_eq!(items[1].step_id.as_deref(), Some("pending-2"));
    assert_eq!(items[2].content, "First pending");
    assert_eq!(items[2].step_id.as_deref(), Some("pending-1"));
    assert_eq!(items[3].content, "Already done");
    assert_eq!(items[3].step_id.as_deref(), Some("done"));

    fs::remove_dir_all(temp_dir).ok();
}

#[test]
fn todo_items_respects_zo_todo_store_override() {
    let temp_dir = unique_temp_dir("todos-env");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let override_path = temp_dir.join("custom-todos.json");
    fs::write(
        &override_path,
        r#"[
          {"content":"Use real store","activeForm":"Using real store","status":"in_progress"}
        ]"#,
    )
    .expect("write custom todo store");

    let _env = EnvVarGuard::set("ZO_TODO_STORE", &override_path);
    let items = todo_items();

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].content, "Use real store");
    assert_eq!(items[0].active_form, "Using real store");
    assert!(matches!(items[0].status, TodoChecklistStatus::InProgress));

    fs::remove_dir_all(temp_dir).ok();
}

#[test]
fn hud_context_usage_splits_cached_tokens_without_double_counting() {
    let usage = hud_context_usage(
        999,
        TokenUsage {
            input_tokens: 60,
            output_tokens: 20,
            cache_creation_input_tokens: 5,
            cache_read_input_tokens: 40,
        },
    );

    assert_eq!(
        usage,
        HudContextUsage {
            used: 105,
            new_input: 65,
            cached: 40,
        }
    );
}

#[test]
fn hud_context_usage_falls_back_to_estimate_before_provider_usage() {
    let usage = hud_context_usage(999, TokenUsage::default());

    assert_eq!(
        usage,
        HudContextUsage {
            used: 999,
            new_input: 0,
            cached: 0,
        }
    );
}

#[test]
fn persistent_slash_resume_renders_report_inside_transcript() {
    let temp_dir = unique_temp_dir("resume");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();

    let resume_path = temp_dir.join("resume-target.jsonl");
    let mut session = Session::new();
    session.session_id = "resume-target".to_string();
    session
        .push_user_text("resume me")
        .expect("session write should succeed");
    session
        .save_to_path(&resume_path)
        .expect("session should persist");

    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();

    let should_quit = handle_persistent_slash(
        &mut cli,
        &mut app,
        &BlockIdGen::default(),
        SlashCommand::Resume {
            session_path: Some(resume_path.display().to_string()),
        },
    )
    .expect("resume should succeed");

    assert!(!should_quit);
    assert!(!app.transcript_mut().is_empty());

    let rendered = render_app_buffer(&mut app);
    assert!(rendered.contains("Session resumed"), "rendered: {rendered}");
    assert!(rendered.contains("Messages"), "rendered: {rendered}");
    // The seeded prompt renders as a real `UserMessage` block — the same
    // widget a live turn pushes (visible `You` author label), not the old
    // `User …` system info row that broke multi-line rendering after /resume.
    assert!(rendered.contains("You"), "rendered: {rendered}");
    assert!(rendered.contains("resume me"), "rendered: {rendered}");
}

/// The Shift+Tab permission cycle must drive the runtime-facing plan flag
/// through the same seam the host loop uses (`set_plan_selected` fed by the
/// App's authoritative `plan_mode_active`), so the model is told Plan is active
/// only when the user actually selected it — never on the plain `ReadOnly` stop.
/// This mirrors the `AppAction::SelectPermission` arm without a live terminal.
#[test]
fn shift_tab_plan_stop_arms_runtime_plan_flag_but_read_only_stop_clears_it() {
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    let temp_dir = unique_temp_dir("plan-flag");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();

    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();
    let cwd = temp_dir.clone();
    // Normalize the badge to ReadOnly so the cycle starts at a known stop.
    app.set_session_meta(
        "sonnet",
        258_000,
        runtime::PermissionMode::ReadOnly,
        cwd.clone(),
        None,
    );

    let back_tab = KeyEvent {
        code: KeyCode::BackTab,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    };
    // Reproduce the host loop's SelectPermission handling: apply the permission
    // change, then mirror the App's authoritative plan flag onto the runtime.
    let cycle_once = |cli: &mut LiveCli, app: &mut App| {
        let action = app.handle_key(back_tab).expect("Shift+Tab handled");
        let super::AppAction::SelectPermission(mode) = action else {
            panic!("expected SelectPermission, got {action:?}");
        };
        cli.apply_permission_change(mode.as_str())
            .expect("permission change should apply");
        app.set_session_meta("sonnet", 258_000, mode, cwd.clone(), None);
        cli.set_plan_selected(app.plan_mode_active());
    };

    // Step 1: ReadOnly → Plan. Runtime enforces read-only, but the flag arms.
    cycle_once(&mut cli, &mut app);
    assert!(app.plan_mode_active(), "first stop is Plan");
    assert!(cli.plan_selected(), "Plan stop arms the runtime plan flag");
    assert!(
        cli.effective_system_prompt()
            .iter()
            .any(|s| s.contains("Plan mode is already active")),
        "Plan stop injects the per-turn Plan contract"
    );

    // Steps 2-4: Plan → Workspace → All → ReadOnly. The plain ReadOnly stop must
    // clear the flag and the contract — read-only is not Plan.
    cycle_once(&mut cli, &mut app); // Workspace
    assert!(!cli.plan_selected(), "Workspace stop is not Plan");
    cycle_once(&mut cli, &mut app); // All
    cycle_once(&mut cli, &mut app); // ReadOnly
    assert!(!app.plan_mode_active(), "back to plain ReadOnly stop");
    assert!(!cli.plan_selected(), "plain ReadOnly must not carry the plan flag");
    assert!(
        !cli.effective_system_prompt()
            .iter()
            .any(|s| s.contains("Plan mode is already active")),
        "plain ReadOnly receives no Plan contract"
    );
}

#[test]
fn persistent_slash_resume_hides_internal_deep_harness_turns() {
    let temp_dir = unique_temp_dir("resume-hide-deep-harness");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();

    let resume_path = temp_dir.join("resume-target.jsonl");
    let mut session = Session::new();
    session.messages = std::sync::Arc::new(vec![
        ConversationMessage::user_text("implement visible change"),
        ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "visible attempt".to_string(),
        }]),
        ConversationMessage::user_text(
            "[deep:VERIFY] You are a STRICT, adversarial verifier. This is internal.",
        ),
        ConversationMessage::assistant(vec![ContentBlock::ToolUse {
            id: "toolu_verify".to_string(),
            name: "grep_search".to_string(),
            input: "{}".to_string(),
        }]),
        ConversationMessage::tool_result(
            "toolu_verify",
            "grep_search",
            "verify output should be hidden",
            false,
        ),
        ConversationMessage::assistant(vec![ContentBlock::Text {
            text: r#"{"accepted": true, "issues": []}"#.to_string(),
        }]),
        ConversationMessage::user_text("[auto:RETRY] hidden repair contract"),
        ConversationMessage::assistant(vec![ContentBlock::Text {
            text: "visible retry answer".to_string(),
        }]),
        ConversationMessage::user_text("visible follow up"),
    ]);
    session
        .save_to_path(&resume_path)
        .expect("session should persist");

    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();

    let should_quit = handle_persistent_slash(
        &mut cli,
        &mut app,
        &BlockIdGen::default(),
        SlashCommand::Resume {
            session_path: Some(resume_path.display().to_string()),
        },
    )
    .expect("resume should succeed");

    assert!(!should_quit);
    let rendered = render_app_buffer(&mut app);
    assert!(rendered.contains("visible attempt"), "rendered: {rendered}");
    assert!(
        rendered.contains("visible retry answer"),
        "rendered: {rendered}"
    );
    assert!(
        rendered.contains("visible follow up"),
        "rendered: {rendered}"
    );
    assert!(!rendered.contains("STRICT"), "rendered: {rendered}");
    assert!(
        !rendered.contains("verify output should be hidden"),
        "rendered: {rendered}"
    );
    assert!(!rendered.contains("accepted"), "rendered: {rendered}");
    assert!(!rendered.contains("[auto:RETRY]"), "rendered: {rendered}");
}

#[test]
fn persistent_slash_session_list_renders_saved_sessions_inside_transcript() {
    let temp_dir = unique_temp_dir("session-list");
    let sessions_dir = temp_dir.join(".zo").join("sessions");
    fs::create_dir_all(&sessions_dir).expect("sessions dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();

    let saved_path = sessions_dir.join("session-saved.jsonl");
    let mut saved = Session::new();
    saved.session_id = "session-saved".to_string();
    saved.name = Some("deploy watch".to_string());
    saved
        .push_user_text("saved session")
        .expect("saved session write should succeed");
    saved
        .save_to_path(&saved_path)
        .expect("saved session should persist");

    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();

    let should_quit = handle_persistent_slash(
        &mut cli,
        &mut app,
        &BlockIdGen::default(),
        SlashCommand::Session {
            action: Some("list".to_string()),
            target: None,
        },
    )
    .expect("session list should succeed");

    assert!(!should_quit);
    assert!(!app.transcript_mut().is_empty());

    let rendered = render_app_buffer(&mut app);
    assert!(rendered.contains("Sessions"), "rendered: {rendered}");
    assert!(rendered.contains("session-saved"), "rendered: {rendered}");
    assert!(rendered.contains("● deploy watch"), "rendered: {rendered}");
}

#[test]
fn persistent_slash_name_updates_and_persists_current_session() {
    let temp_dir = unique_temp_dir("session-name");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _session_root = EnvVarGuard::set("ZO_SESSION_ROOT", &temp_dir);
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();

    // `_session_root` already holds the crate env lock for this test's
    // lifetime — re-locking here would self-deadlock.
    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();
    let ids = BlockIdGen::default();

    handle_persistent_slash(
        &mut cli,
        &mut app,
        &ids,
        SlashCommand::Name {
            name: Some("deploy watch".to_string()),
        },
    )
    .expect("name command should succeed");

    assert_eq!(cli.runtime.session().name.as_deref(), Some("deploy watch"));
    assert_eq!(
        Session::load_from_path(&cli.session.path)
            .expect("named session should persist")
            .name
            .as_deref(),
        Some("deploy watch")
    );

    handle_persistent_slash(
        &mut cli,
        &mut app,
        &ids,
        SlashCommand::Name { name: None },
    )
    .expect("bare name command should succeed");
    let rendered = render_app_buffer(&mut app);
    assert!(rendered.contains("deploy watch"), "rendered: {rendered}");
    assert!(rendered.contains("/name <name>"), "rendered: {rendered}");
}

#[test]
fn persistent_slash_group_b_reports_render_inside_transcript() {
    let temp_dir = unique_temp_dir("group-b");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();

    let cases = [
        (SlashCommand::Version, crate::VERSION),
        (
            SlashCommand::Config {
                section: Some("env".to_string()),
            },
            "Merged section: env",
        ),
        // Cards reformat the `Usage␣␣␣␣␣/x` key/value spacing into a
        // `Usage␣␣/x` row, so assert the spacing-stable bracket payload.
        (
            SlashCommand::Agents {
                args: Some("help".to_string()),
            },
            "[list|help]",
        ),
        (
            SlashCommand::Skills {
                args: Some("help".to_string()),
            },
            "[list|install <path>|help]",
        ),
        (
            SlashCommand::Mcp {
                action: Some("help".to_string()),
                target: None,
            },
            "[list|show <server>|auth [list|<server>]|logout <server>|help]",
        ),
    ];

    let mut cli = test_live_cli_with_env_lock_held();

    for (command, needle) in cases {
        let mut app = new_test_app();
        let should_quit =
            handle_persistent_slash(&mut cli, &mut app, &BlockIdGen::default(), command)
                .expect("group-b command should succeed");
        assert!(!should_quit);

        let rendered = render_app_buffer(&mut app);
        assert!(
            rendered.contains(needle),
            "missing `{needle}` in: {rendered}"
        );
    }
}

#[test]
fn persistent_slash_inbox_opens_team_inbox_modal() {
    let temp_dir = unique_temp_dir("inbox-modal");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();
    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();

    let should_quit = handle_persistent_slash(
        &mut cli,
        &mut app,
        &BlockIdGen::default(),
        SlashCommand::Inbox { args: None },
    )
    .expect("inbox command should succeed");

    assert!(!should_quit);
    assert_eq!(app.mode(), AppMode::ModalTeamInbox);
}

#[test]
fn persistent_slash_bare_tier_opens_deep_tier_modal() {
    let temp_dir = unique_temp_dir("tier-modal");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();
    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();

    let should_quit = handle_persistent_slash(
        &mut cli,
        &mut app,
        &BlockIdGen::default(),
        SlashCommand::DeepTier {
            action: commands::DeepTierAction::Show,
        },
    )
    .expect("tier command should succeed");

    assert!(!should_quit);
    assert_eq!(app.mode(), AppMode::ModalDeepTier);
    let rendered = render_app_buffer(&mut app);
    assert!(rendered.contains("Deep-tier models"), "{rendered}");
    assert!(rendered.contains("(built-in default)"), "{rendered}");
}

#[test]
fn persistent_tui_help_and_completions_classify_supported_commands() {
    let help = crate::session::slash_dispatch::render_persistent_tui_help();
    assert!(help.contains("Persistent TUI"));
    assert!(help.contains("Implemented"));
    assert!(help.contains("/config"));
    assert!(help.contains("/version"));
    assert!(help.contains("Deferred"));
    assert!(help.contains("Autocomplete"));

    assert!(crate::session::slash_dispatch::persistent_tui_candidate_supported("/config env"));
    assert!(crate::session::slash_dispatch::persistent_tui_candidate_supported("/agents help"));
    assert!(crate::session::slash_dispatch::persistent_tui_candidate_supported("/plugins list"));
    assert!(crate::session::slash_dispatch::persistent_tui_candidate_supported("/commit"));
    assert!(crate::session::slash_dispatch::persistent_tui_candidate_supported("/tools"));
}

#[test]
fn running_count_counts_only_non_terminal_agents() {
    fn agent(status: &str) -> AgentTaskSummary {
        AgentTaskSummary {
            name: "a".to_string(),
            status: status.to_string(),
            model: String::new(),
            elapsed_secs: 0,
            token_history: Vec::new(),
            current_tool: None,
            current_phase: None,
            last_activity_at: None,
            ..Default::default()
        }
    }
    let agents = vec![
        agent("running"),
        agent("running"),
        agent("pending"),
        agent("completed"),
        agent("failed"),
        agent("stopped"),
    ];
    // running + running + pending = 3; the three terminal rows are excluded from
    // the headline `⚡ N agents` count even while they linger in the tree.
    assert_eq!(running_count(&agents), 3);
    assert_eq!(running_count(&[]), 0);
}

#[test]
fn agent_stale_limits_reap_abandoned_running_rows_promptly() {
    assert_eq!(agent_stale_limit_secs("running"), 20 * 60);
    assert_eq!(agent_stale_limit_secs("pending"), 20 * 60);
    assert_eq!(agent_stale_limit_secs("stopped"), TERMINAL_GRACE_SECS);
    assert_eq!(agent_stale_limit_secs("completed"), TERMINAL_GRACE_SECS);
}

#[test]
fn agent_freshness_surfaces_a_quiet_running_agent_as_stalled_then_drops_it() {
    // Fresh while it keeps writing.
    assert_eq!(agent_freshness("running", 60), Some(false));
    assert_eq!(agent_freshness("running", RUNNING_STALE_SECS), Some(false));
    // Past the freshness limit it is kept but surfaced as stalled (instead of the
    // old behaviour where it vanished, leaving the HUD reading `agents 0/1`).
    assert_eq!(agent_freshness("running", RUNNING_STALE_SECS + 1), Some(true));
    assert_eq!(
        agent_freshness("pending", STALLED_SURFACE_LIMIT_SECS),
        Some(true)
    );
    // Past the hard ceiling an abandoned worker finally drops.
    assert_eq!(agent_freshness("running", STALLED_SURFACE_LIMIT_SECS + 1), None);
    // Terminal rows never surface as stalled — they drop on the short grace.
    assert_eq!(agent_freshness("completed", TERMINAL_GRACE_SECS), Some(false));
    assert_eq!(agent_freshness("completed", TERMINAL_GRACE_SECS + 1), None);
    assert_eq!(agent_freshness("failed", TERMINAL_GRACE_SECS + 1), None);
}

#[test]
fn liveness_rescue_keeps_a_live_worker_past_the_drop_window_but_never_a_dead_or_terminal_one() {
    let past_drop = STALLED_SURFACE_LIMIT_SECS + 1;
    // A running manifest quiet through one long tool call (cold cargo build):
    // a live in-process worker keeps the row, surfaced as stalled.
    assert_eq!(
        agent_freshness_with_liveness("running", past_drop, || true),
        Some(true)
    );
    // Dead worker → drops exactly as before.
    assert_eq!(
        agent_freshness_with_liveness("running", past_drop, || false),
        None
    );
    // Terminal rows never take the rescue, even with a (stale) live signal.
    assert_eq!(
        agent_freshness_with_liveness("stopped", TERMINAL_GRACE_SECS + 1, || true),
        None
    );
    // Inside the window the mtime verdict stands and the registry is not asked.
    assert_eq!(
        agent_freshness_with_liveness("running", 60, || unreachable!()),
        Some(false)
    );
}

#[test]
fn agent_elapsed_uses_manifest_timestamps_not_file_mtime() {
    let running = serde_json::json!({
        "createdAt": "100",
        "status": "running"
    });
    assert_eq!(
        agent_elapsed_secs(&running, 1_300, 3),
        1_200,
        "running agent duration should not reset when the manifest file is rewritten"
    );

    let completed = serde_json::json!({
        "createdAt": "100",
        "completedAt": "250",
        "status": "completed"
    });
    assert_eq!(
        agent_elapsed_secs(&completed, 1_300, 3),
        150,
        "terminal duration should be fixed from createdAt/completedAt"
    );

    let legacy = serde_json::json!({
        "status": "running"
    });
    assert_eq!(
        agent_elapsed_secs(&legacy, 1_300, 7),
        7,
        "legacy manifests without timestamps fall back to mtime age"
    );
}

#[test]
fn session_close_stops_only_agents_started_by_current_session() {
    fn write_manifest(
        store: &Path,
        id: &str,
        session_id: Option<&str>,
        status: &str,
        created_at: &str,
    ) {
        let manifest_path = store.join(format!("{id}.json"));
        let output_path = store.join(format!("{id}.md"));
        let mut value = serde_json::json!({
            "agentId": id,
            "name": id,
            "description": "test agent",
            "subagentType": null,
            "model": null,
            "status": status,
            "outputFile": output_path,
            "manifestFile": manifest_path,
            "createdAt": created_at,
            "startedAt": created_at
        });
        if let Some(session_id) = session_id {
            value["parentSessionId"] = serde_json::json!(session_id);
        }
        fs::write(&manifest_path, serde_json::to_string(&value).unwrap()).expect("manifest");
        fs::write(output_path, "").expect("output");
    }

    fn status(store: &Path, id: &str) -> String {
        serde_json::from_str::<serde_json::Value>(
            &fs::read_to_string(store.join(format!("{id}.json"))).unwrap(),
        )
        .unwrap()["status"]
            .as_str()
            .unwrap()
            .to_string()
    }

    let store = unique_temp_dir("session-close-agents");
    fs::create_dir_all(&store).expect("agent store");
    let _guard = EnvVarGuard::set("ZO_AGENT_STORE", &store);

    write_manifest(&store, "current", Some("session-a"), "running", "200");
    write_manifest(&store, "foreign", Some("session-b"), "running", "210");
    write_manifest(&store, "legacy", None, "running", "220");
    write_manifest(&store, "old", Some("session-a"), "running", "100");
    write_manifest(&store, "done", Some("session-a"), "completed", "230");

    let stopped = stop_agents_for_session_close(150, "session-a", "parent session closed");

    assert_eq!(stopped, 1);
    assert_eq!(status(&store, "current"), "stopped");
    assert_eq!(status(&store, "foreign"), "running");
    assert_eq!(status(&store, "legacy"), "running");
    assert_eq!(status(&store, "old"), "running");
    assert_eq!(status(&store, "done"), "completed");
    let _ = fs::remove_dir_all(store);
}

#[test]
fn agent_manifest_listing_ignores_previous_session_rows() {
    let store = unique_temp_dir("agent-baseline");
    fs::create_dir_all(&store).expect("agent store");
    let _guard = EnvVarGuard::set("ZO_AGENT_STORE", &store);

    fs::write(
        store.join("old.json"),
        r#"{
            "agentId":"old",
            "name":"old-session-agent",
            "status":"running",
            "createdAt":"100",
            "model":"openai/gpt-5.5-fast"
        }"#,
    )
    .expect("old manifest");
    fs::write(
        store.join("new.json"),
        r#"{
            "agentId":"new",
            "parentSessionId":"session-a",
            "label":"current session agent",
            "status":"running",
            "createdAt":"200",
            "model":"openai/gpt-5.5-fast"
        }"#,
    )
    .expect("new manifest");
    fs::write(
        store.join("foreign.json"),
        r#"{
            "agentId":"foreign",
            "parentSessionId":"session-b",
            "label":"foreign session agent",
            "status":"running",
            "createdAt":"250",
            "model":"claude-opus-4-8"
        }"#,
    )
    .expect("foreign manifest");

    let agents = list_running_agents_since(150, Some("session-a"));

    assert_eq!(agents.len(), 1, "only current-session agent should show");
    assert_eq!(agents[0].name, "current session agent");
    assert_eq!(running_count(&agents), 1);
    let _ = fs::remove_dir_all(store);
}

#[test]
fn dead_worker_manifest_surfaces_as_failed_instead_of_disappearing() {
    let store = unique_temp_dir("dead-agent-hud");
    fs::create_dir_all(&store).expect("agent store");
    let _guard = EnvVarGuard::set("ZO_AGENT_STORE", &store);
    let agent_id = "dead-agent-hud";
    let manifest_path = store.join(format!("{agent_id}.json"));
    let output_path = store.join(format!("{agent_id}.md"));
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string();
    fs::write(
        &manifest_path,
        serde_json::to_string(&serde_json::json!({
            "agentId": agent_id,
            "parentSessionId": "session-a",
            "name": "dead agent",
            "description": "test dead worker HUD surfacing",
            "subagentType": "general-purpose",
            "status": "running",
            "outputFile": output_path,
            "manifestFile": manifest_path,
            "createdAt": created_at,
            "ownerPid": std::process::id(),
            "runGeneration": 1,
            "startedAt": created_at,
            "currentTool": "cargo test"
        }))
        .expect("serialize manifest"),
    )
    .expect("write manifest");
    fs::write(&output_path, "# Agent Task\n").expect("write output");

    let agents = list_running_agents_since(0, Some("session-a"));

    assert_eq!(agents.len(), 1, "the failed row stays within terminal grace");
    assert_eq!(agents[0].status, "failed");
    assert_eq!(running_count(&agents), 0);
    let persisted: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&manifest_path).expect("read reconciled manifest"),
    )
    .expect("parse reconciled manifest");
    assert_eq!(persisted["status"], "failed");
    assert_eq!(
        persisted["error"],
        "worker died without delivering a result"
    );
    let _ = fs::remove_dir_all(store);
}

#[test]
fn agent_manifest_listing_does_not_fallback_to_unstamped_rows() {
    let store = unique_temp_dir("agent-unstamped-no-fallback");
    fs::create_dir_all(&store).expect("agent store");
    let _guard = EnvVarGuard::set("ZO_AGENT_STORE", &store);

    fs::write(
        store.join("current-unstamped.json"),
        r#"{
            "agentId":"current-unstamped",
            "name":"current unstamped agent",
            "status":"running",
            "createdAt":"200",
            "model":"claude-opus-4-8"
        }"#,
    )
    .expect("unstamped manifest");
    fs::write(
        store.join("foreign.json"),
        r#"{
            "agentId":"foreign",
            "parentSessionId":"session-b",
            "name":"foreign agent",
            "status":"running",
            "createdAt":"210",
            "model":"claude-opus-4-8"
        }"#,
    )
    .expect("foreign manifest");

    let agents = list_running_agents_since(150, Some("session-a"));

    assert!(
        agents.is_empty(),
        "unstamped manifests cannot be safely attributed to the current session"
    );
    assert_eq!(running_count(&agents), 0);
    let _ = fs::remove_dir_all(store);
}

#[test]
fn attached_image_blocks_decode_base64_and_skip_undecodable() {
    use base64::Engine as _;

    let ids = BlockIdGen(Arc::new(AtomicU64::new(0)));
    // A tiny "PNG" — content is irrelevant; we assert the round-tripped bytes.
    let png: &[u8] = b"\x89PNG\r\n\x1a\n-fake-bytes";
    let good = base64::engine::general_purpose::STANDARD.encode(png);
    let images = vec![
        ("image/png".to_string(), good),
        // `@` is outside the base64 alphabet → decode fails → block skipped.
        ("image/jpeg".to_string(), "@@@@not-base64@@@@".to_string()),
    ];

    let blocks = attached_image_blocks(&images, &ids);

    assert_eq!(
        blocks.len(),
        1,
        "the valid image is echoed and the undecodable one is skipped"
    );
    match &blocks[0] {
        RenderBlock::Image {
            data, media_type, ..
        } => {
            assert_eq!(media_type, "image/png");
            assert_eq!(data.as_slice(), png, "base64 round-trips to the raw bytes");
        }
        other => panic!("expected an Image block, got {other:?}"),
    }
}

#[test]
fn attached_image_blocks_is_empty_without_attachments() {
    let ids = BlockIdGen(Arc::new(AtomicU64::new(0)));
    assert!(attached_image_blocks(&[], &ids).is_empty());
}

#[test]
fn auto_review_diff_opens_when_turn_edits_meet_threshold() {
    let summary = turn_summary_with_tools(&["Read", "Edit", "Write"]);

    assert!(should_auto_open_review_diff(&summary, 1));
    assert!(should_auto_open_review_diff(&summary, 2));
    assert!(!should_auto_open_review_diff(&summary, 3));
}

#[test]
fn auto_review_diff_is_disabled_for_zero_threshold_or_read_only_turn() {
    let read_only = turn_summary_with_tools(&["Read", "Grep"]);
    let edited = turn_summary_with_tools(&["MultiEdit"]);

    assert!(!should_auto_open_review_diff(&edited, 0));
    assert!(!should_auto_open_review_diff(&read_only, 1));
}

/// 하단 스택(스피너 상태줄·입력·HUD)의 공유 거터 계약: 세 경계선 모두
/// 콘텐츠가 col 3 에서 시작한다 — 스피너 줄은 `── ⣾ …`, 입력은 `│❯ text`,
/// HUD 는 한 셀 heat marker를 가운데 둔 ` + mode …` 형태다. 스피너 리더가
/// 3셀 rule(`─── `)로 그려져 1칸 어긋나던 회귀를 고정한다.
#[test]
fn bottom_stack_boundaries_share_col3_content_gutter() {
    let mut app = new_test_app();
    app.begin_turn_with_generation(0);
    app.set_turn_activity("Delegating");
    app.set_input_text("workflow");
    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).expect("test backend");
    app.draw(&mut terminal).expect("draw");
    let buffer = terminal.backend().buffer();
    let row_text = |y: u16| -> String {
        (0..120u16)
            .map(|x| {
                buffer
                    .cell((x, y))
                    .map_or(" ", ratatui::buffer::Cell::symbol)
            })
            .collect()
    };
    let rows: Vec<String> = (0..30u16).map(row_text).collect();
    let spinner_y = rows
        .iter()
        .position(|row| row.contains("Delegating"))
        .expect("spinner row");
    let input_y = rows
        .iter()
        .position(|row| row.contains("❯ workflow"))
        .expect("input row");
    // HUD 줄은 sandbox 세그먼트로 식별 (사이드바의 `mode read-only` 행과 구분).
    let hud_y = rows
        .iter()
        .rposition(|row| row.contains("sandbox:"))
        .expect("hud row");

    let spinner = &rows[spinner_y];
    let input = &rows[input_y];
    let hud = &rows[hud_y];
    // no_color spinner stays `-- `; the modern HUD centers one heat marker in
    // the same three-cell gutter instead of restoring decorative rule chrome.
    assert!(
        spinner.starts_with("-- ") && !spinner.starts_with("---"),
        "spinner leader must be the shared 3-cell `── ` gutter: {spinner:?}"
    );
    assert!(
        hud.starts_with(" + "),
        "hud leader must be the shared 3-cell ` + ` gutter: {hud:?}"
    );
    // 콘텐츠 col3 정렬: 스피너 글리프, 입력 텍스트, HUD 텍스트.
    let col3_glyph = |row: &str| row.chars().nth(3).unwrap_or(' ');
    assert_ne!(
        col3_glyph(spinner),
        ' ',
        "spinner content at col3: {spinner:?}"
    );
    assert!(
        input.chars().take(3).filter(|c| *c != ' ').count() >= 2,
        "input leader `┃❯ ` occupies cols 0-2: {input:?}"
    );
    let input_text_at_col3: String = input.chars().skip(3).take(8).collect();
    assert_eq!(
        input_text_at_col3, "workflow",
        "input text starts at col3: {input:?}"
    );
    assert_ne!(col3_glyph(hud), ' ', "hud content at col3: {hud:?}");
}

fn channel_completion(id: &str, status: &str, error: Option<&str>) -> tools::AgentCompletion {
    // Mirrors the compact event the broadcast channel delivers: `result` is
    // always stripped (it lives in the completion store), `error` is preserved.
    tools::AgentCompletion {
        agent_id: id.to_string(),
        name: "explorer".to_string(),
        status: status.to_string(),
        result: None,
        structured: None,
        error: error.map(str::to_string),
        output_tokens: 0,
    }
}

#[test]
fn reinject_skips_non_background_completion() {
    let mut app = new_test_app();
    // Not marked background → a synchronous agent whose result already returned
    // inline to the model. Re-injecting would duplicate the answer.
    let completion = channel_completion("bg-bin-nonbg-1", "completed", None);
    assert!(!reinject_background_agent_completion(
        &mut app,
        &completion,
        "session-a"
    ));
    assert!(app.take_queued_messages().is_empty());
}

#[test]
fn reinject_skips_stopped_background_but_clears_marker() {
    let mut app = new_test_app();
    let id = "bg-bin-stopped-1";
    tools::mark_background_agent(id.to_string());
    // A user-cancelled (Ctrl+C-reaped) background agent must not resurrect a
    // turn, but its marker is still cleared so the id set never leaks.
    let completion = channel_completion(id, "stopped", Some("cancelled by foreground turn"));
    assert!(!reinject_background_agent_completion(
        &mut app,
        &completion,
        "session-a"
    ));
    assert!(!tools::is_background_agent(id));
    assert!(app.take_queued_messages().is_empty());
}

#[test]
fn reinject_completed_background_is_graceful_when_result_unavailable() {
    let mut app = new_test_app();
    let id = "bg-bin-evicted-1";
    tools::mark_background_agent(id.to_string());
    // Completed event but the store has no full result for this id (never
    // recorded / TTL-evicted). Re-injection must degrade to a no-op rather than
    // queue an empty follow-up turn — and still clear the marker.
    let completion = channel_completion(id, "completed", None);
    assert!(!reinject_background_agent_completion(
        &mut app,
        &completion,
        "session-a"
    ));
    assert!(!tools::is_background_agent(id));
    assert!(app.take_queued_messages().is_empty());
}

#[test]
fn reinject_failed_background_bash_surfaces_its_output_not_a_generic_message() {
    let mut app = new_test_app();
    let id = "bg-bin-failed-output-1";
    // Real bridge: a failed background *bash* task records its stdout/stderr +
    // the `[exit N]` line in the stored completion's `result`, with
    // `error: None` (see `notify_background_task_completion`). The broadcast
    // channel event then strips `result`, so reinject must read the output back
    // from the store instead of falling through to the canned
    // "failed without an error message" — otherwise the model never sees why
    // the background task failed.
    tools::notify_background_task_completion(
        id.to_string(),
        "failed",
        Some("build failed: error[E0433]: unresolved import\n[exit 101]".to_string()),
        Some("session-a".to_string()),
    );
    // The compact event the arm actually receives: `result` stripped,
    // `error` absent (a bash task carries no separate error string).
    let event = tools::AgentCompletion {
        agent_id: id.to_string(),
        name: "background bash".to_string(),
        status: "failed".to_string(),
        result: None,
        structured: None,
        error: None,
        output_tokens: 0,
    };
    assert!(
        reinject_background_agent_completion(&mut app, &event, "session-a"),
        "a failed background bash with real output must re-inject a follow-up turn"
    );
    let queued = app.take_queued_messages();
    assert_eq!(queued.len(), 1, "exactly one follow-up turn queued");
    let text = &queued[0].text;
    assert!(
        text.contains("build failed: error[E0433]: unresolved import"),
        "the model must receive the actual failure output: {text:?}"
    );
    assert!(text.contains("[exit 101]"), "the exit line must survive: {text:?}");
    assert!(
        !text.contains("failed without an error message"),
        "must not fall through to the generic placeholder: {text:?}"
    );
    assert!(!tools::is_background_agent(id), "marker cleared after consumption");
}

#[test]
fn reinject_suppresses_background_bash_from_another_session() {
    let mut app = new_test_app();
    let id = "bg-bin-cross-session-1";
    tools::notify_background_task_completion(
        id.to_string(),
        "completed",
        Some("session-a secret output".to_string()),
        Some("session-a".to_string()),
    );
    let event = channel_completion(id, "completed", None);

    assert!(!reinject_background_agent_completion(
        &mut app,
        &event,
        "session-b"
    ));
    assert!(app.take_queued_messages().is_empty());
    assert!(
        !tools::is_background_agent(id),
        "suppression consumes the marker so the event cannot leak as a generic notice"
    );
}

fn new_notification_inbox() -> runtime::AgentNotificationInbox {
    std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))
}

/// A background agent finishing while a turn is LIVE stages its result into
/// the mid-turn inbox (CC task-notification parity) instead of the follow-up
/// turn queue — the turn folds it in at its next tool-result boundary, so the
/// main model keeps working through the completion.
#[test]
fn mid_turn_delivery_stages_completed_background_into_inbox() {
    use crate::session::agent_notice::deliver_background_agent_completion_mid_turn;
    let mut app = new_test_app();
    let inbox = new_notification_inbox();
    let id = "bg-bin-midturn-1";
    // Seed the completion store with the full result (the channel event
    // strips it) — this also marks the id as background.
    tools::notify_background_task_completion(
        id.to_string(),
        "completed",
        Some("scout answer: the flag lives in config.rs".to_string()),
        Some("session-a".to_string()),
    );
    let event = channel_completion(id, "completed", None);

    assert!(
        deliver_background_agent_completion_mid_turn(&mut app, &inbox, &event, "session-a"),
        "a completed background agent must stage a mid-turn notification"
    );
    let staged = inbox.lock().expect("inbox");
    assert_eq!(staged.len(), 1, "exactly one notification staged");
    assert_eq!(staged[0].label, "explorer");
    assert_eq!(
        staged[0].status,
        runtime::message_stream::AgentResultStatus::Completed
    );
    assert!(
        staged[0].text.contains("the flag lives in config.rs"),
        "the model-facing text carries the stored result: {:?}",
        staged[0].text
    );
    assert!(
        staged[0].text.contains("finished"),
        "the header explains the completion: {:?}",
        staged[0].text
    );
    drop(staged);
    assert!(
        app.take_queued_messages().is_empty(),
        "mid-turn staging must not double-queue a follow-up turn"
    );
    assert!(!tools::is_background_agent(id), "marker cleared after consumption");
}

/// Non-background and stopped completions never stage a notification —
/// identical gating to the follow-up-turn path (shared builder).
#[test]
fn mid_turn_delivery_skips_non_background_completion() {
    use crate::session::agent_notice::deliver_background_agent_completion_mid_turn;
    let mut app = new_test_app();
    let inbox = new_notification_inbox();
    let event = channel_completion("bg-bin-midturn-nonbg-1", "completed", None);
    assert!(!deliver_background_agent_completion_mid_turn(
        &mut app,
        &inbox,
        &event,
        "session-a"
    ));
    assert!(inbox.lock().expect("inbox").is_empty());
    assert!(app.take_queued_messages().is_empty());
}

/// Notifications the turn never reached a boundary to fold are re-queued as
/// follow-up agent-result turns after the turn — the exactly-once tail of
/// mid-turn delivery. The queued entries carry agent-result meta so the
/// pop-time coalesce and card rendering behave exactly like the direct
/// re-injection path.
#[test]
fn undelivered_notifications_requeue_as_followup_agent_result_turns() {
    use crate::session::agent_notice::requeue_undelivered_agent_notifications;
    let mut app = new_test_app();
    let inbox = new_notification_inbox();
    inbox.lock().expect("inbox").extend([
        runtime::AgentNotification {
            label: "runtime-scout".to_string(),
            status: runtime::message_stream::AgentResultStatus::Completed,
            text: "[background agent `runtime-scout` finished]\n\nanswer A".to_string(),
        },
        runtime::AgentNotification {
            label: "fixture-writer".to_string(),
            status: runtime::message_stream::AgentResultStatus::Failed,
            text: "[background agent `fixture-writer` failed]\n\nboom".to_string(),
        },
    ]);

    assert_eq!(requeue_undelivered_agent_notifications(&mut app, &inbox), 2);
    assert!(inbox.lock().expect("inbox").is_empty(), "inbox fully drained");
    let queued = app.take_queued_messages();
    assert_eq!(queued.len(), 2);
    let first_meta = queued[0].agent_result.as_ref().expect("agent-result meta");
    assert_eq!(first_meta.label, "runtime-scout");
    assert!(queued[0].text.contains("answer A"));
    let second_meta = queued[1].agent_result.as_ref().expect("agent-result meta");
    assert_eq!(
        second_meta.status,
        runtime::message_stream::AgentResultStatus::Failed
    );
    assert!(queued[1].text.contains("boom"));
}

/// The idle-tick MCP poller and the full HUD rebuild share
/// `encoded_mcp_hud_rows`; this pins the state→row mapping at that seam: a
/// pending (still-discovering) server encodes as `discovering`, and a server
/// registered without a pending entry encodes as `ready` — so the
/// discovering→ready flip the poller must surface is purely an observable
/// state change, never a divergent second mapping.
#[test]
fn encoded_mcp_hud_rows_reflect_pending_and_ready_servers() {
    use zo_cli::tui::hud::McpHudStatusKind;
    use std::collections::BTreeMap;

    let servers = BTreeMap::from([(
        "alpha".to_string(),
        runtime::ScopedMcpServerConfig {
            scope: runtime::ConfigSource::User,
            config: runtime::McpServerConfig::Ws(runtime::McpWebSocketServerConfig {
                url: "ws://127.0.0.1:9".to_string(),
                headers: BTreeMap::new(),
                headers_helper: None,
            }),
        },
    )]);
    let manager = runtime::McpServerManager::from_servers(&servers);
    let state = Arc::new(Mutex::new(
        crate::session::mcp_runtime::RuntimeMcpState::from_manager_for_test(manager),
    ));

    let rows = encoded_mcp_hud_rows(&state);
    let decoded: Vec<McpHudStatus> = rows.iter().map(|row| McpHudStatus::decode(row)).collect();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].name, "alpha");
    assert_eq!(decoded[0].kind, McpHudStatusKind::Discovering);

    assert!(
        state
            .lock()
            .expect("mcp state")
            .add_ws_server("beta", "ws://127.0.0.1:10".to_string(), BTreeMap::new()),
        "registering a second server should succeed"
    );
    let rows = encoded_mcp_hud_rows(&state);
    let decoded: Vec<McpHudStatus> = rows.iter().map(|row| McpHudStatus::decode(row)).collect();
    let kinds: BTreeMap<&str, McpHudStatusKind> = decoded
        .iter()
        .map(|status| (status.name.as_str(), status.kind))
        .collect();
    assert_eq!(kinds.get("alpha"), Some(&McpHudStatusKind::Discovering));
    assert_eq!(
        kinds.get("beta"),
        Some(&McpHudStatusKind::Ready),
        "a server with no pending discovery entry must surface as ready"
    );
}

/// Read-only report commands open the centered report popup instead of
/// writing into the transcript: the popup shows the content, and closing it
/// leaves the conversation exactly as it was (nothing recorded — a report is
/// re-derivable by re-running its command).
#[test]
fn persistent_slash_report_command_opens_popup_and_leaves_transcript_clean() {
    let temp_dir = unique_temp_dir("report-popup");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Canonical test lock order: env BEFORE cwd (see `test_env_lock`).
    let _env = crate::test_env_lock();
    let _guard = CurrentDirGuard::enter(&temp_dir);
    let _api_key = ApiKeyGuard::set_dummy();
    let mut cli = test_live_cli_with_env_lock_held();
    let mut app = new_test_app();

    let should_quit = handle_persistent_slash(
        &mut cli,
        &mut app,
        &BlockIdGen::default(),
        SlashCommand::Version,
    )
    .expect("version command should succeed");

    assert!(!should_quit);
    assert_eq!(
        app.mode(),
        AppMode::ModalReport,
        "report commands must open the generic report popup"
    );
    let rendered = render_app_buffer(&mut app);
    assert!(
        rendered.contains(crate::VERSION),
        "popup must show the report content: {rendered}"
    );

    // Esc closes the popup; the transcript stayed clean, so the report text
    // is gone from the frame.
    app.handle_key(crossterm::event::KeyEvent::new(
        crossterm::event::KeyCode::Esc,
        crossterm::event::KeyModifiers::NONE,
    ))
    .expect("esc handled");
    assert_eq!(app.mode(), AppMode::Normal);
    let rendered = render_app_buffer(&mut app);
    assert!(
        !rendered.contains(crate::VERSION),
        "closing the popup must leave no report residue in the transcript: {rendered}"
    );
}

#[test]
fn boot_timing_line_formats_all_phases_in_ms() {
    use std::time::Duration;
    let line = format_boot_timing_line(
        Duration::from_millis(128),
        Duration::from_millis(7),
        Duration::from_millis(15),
        Duration::from_millis(163),
    );
    assert_eq!(
        line,
        "[boot] runtime_build=128ms status_context=7ms to_first_frame=15ms total=163ms"
    );
}
