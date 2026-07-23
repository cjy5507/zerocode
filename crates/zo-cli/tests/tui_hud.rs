//! Integration tests for the quiet, text-first HUD.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use runtime::message_stream::ActiveModel;
use zo_cli::tui::app::{ScheduledWakeHud, WakeSource};
use zo_cli::tui::hud::{
    self, HudState, PermissionMode as HudPermissionMode, SecurityPosture, SessionIdentity,
    TodoChecklistItem, TodoChecklistStatus,
};
use zo_cli::tui::theme::Theme;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn active_model(provider: &'static str, alias: &str) -> ActiveModel {
    ActiveModel {
        provider,
        alias: alias.to_string(),
        display_name: format!("{provider}:{alias}"),
        context_limit: 200_000,
    }
}

fn sample_state() -> HudState {
    HudState {
        session_identity: None,
        model: ActiveModel {
            provider: "anthropic",
            alias: "opus".to_string(),
            display_name: "claude-opus-4-8".to_string(),
            context_limit: 200_000,
        },
        turn_fallback_model: None,
        quota_fallback_model: None,
        ctx_used: 42_000,
        ctx_limit: 200_000,
        ctx_new_input: 0,
        ctx_cached: 0,
        compact_threshold: 0,
        cost_usd: 0.37,
        cost_approx: false,
        cwd: PathBuf::from("/Users/joe/dev/zo"),
        git_branch: Some("main".to_string()),
        perm_mode: HudPermissionMode::Workspace,
        security_posture: SecurityPosture::SandboxActive,
        effort: None,
        architect_impl: None,
        mcp_servers: vec!["almanac".to_string(), "context7".to_string()],
        bash_count: 14,
        read_count: 2,
        edit_count: 4,
        changed_files: 0,
        todo_summary: None,
        todo_items: Vec::new(),
        automation_lines: Vec::new(),
        lsp_servers: Vec::new(),
        running_agents: 0,
        agents: Vec::new(),
        workflow: None,
        last_tool: None,
        rate_limit: None,
        provider_quotas: Vec::new(),
        auth_origin: None,
        status_line: None,
        team_inbox_unread: 0,
        stale_binary: None,
        background_tasks: 0,
        scheduled_wake: None,
    }
}

fn render_buffer(cols: u16, state: &HudState, theme: &Theme) -> String {
    let backend = TestBackend::new(cols, 1);
    let mut terminal = Terminal::new(backend).expect("test backend");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, cols, 1);
            // ledger_visible=false: HUD is the single ctx/cost authority,
            // matching the session-summary contract exercised below.
            hud::draw(frame, area, state, theme, false, false);
        })
        .expect("draw");
    let buffer = terminal.backend().buffer().clone();
    buffer
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect::<String>()
}

#[test]
fn hud_renders_live_background_bash_badge_on_bottom_row() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.background_tasks = 2;

    let out = render_buffer(80, &state, &theme);

    assert!(out.contains("bg 2"), "background Bash badge missing: {out}");
}

#[test]
fn hud_renders_named_session_badge() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.session_identity = SessionIdentity::named("session-123", Some("deploy watch"));

    let out = render_buffer(100, &state, &theme);

    assert!(out.contains("● deploy watch"), "session badge missing: {out}");
}

#[test]
fn hud_renders_scheduled_wake_chip_only_when_armed() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    let unarmed = render_buffer(80, &state, &theme);
    assert!(!unarmed.contains("wake "), "unarmed wake chip leaked: {unarmed}");

    state.scheduled_wake = Some(ScheduledWakeHud {
        due_at_epoch: 0,
        reason: "check CI".to_string(),
        source: WakeSource::Wakeup,
    });
    let armed = render_buffer(80, &state, &theme);
    assert!(armed.contains("wake now"), "scheduled wake chip missing: {armed}");
}

#[test]
fn hud_background_badge_coexists_with_custom_status_line() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.status_line = Some("custom branch status".to_string());
    state.background_tasks = 1;

    let out = render_buffer(80, &state, &theme);

    assert!(out.contains("custom branch status"), "custom status missing: {out}");
    assert!(out.contains("bg 1"), "background Bash badge missing: {out}");
}

#[test]
fn hud_narrow_width_prioritizes_live_background_badge_over_agents() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.background_tasks = 1;
    state.running_agents = 12;

    let out = render_buffer(12, &state, &theme);

    assert!(out.contains("bg 1"), "active badge must not be right-clipped: {out}");
}

#[test]
fn hud_canonical_todo_order_replaces_stale_status_order() {
    let items = vec![
        TodoChecklistItem {
            step_id: None,
            content: "done".to_string(),
            active_form: "done".to_string(),
            status: TodoChecklistStatus::Completed,
        },
        TodoChecklistItem {
            step_id: None,
            content: "active".to_string(),
            active_form: "doing".to_string(),
            status: TodoChecklistStatus::InProgress,
        },
        TodoChecklistItem {
            step_id: None,
            content: "queued".to_string(),
            active_form: "queueing".to_string(),
            status: TodoChecklistStatus::Pending,
        },
    ];

    let ordered = hud::canonical_todo_items_for_hud(items);
    let contents = ordered
        .iter()
        .map(|item| item.content.as_str())
        .collect::<Vec<_>>();
    assert_eq!(contents, vec!["active", "queued", "done"]);
}

#[test]
fn hud_todo_store_honors_session_override_and_drops_stale_default() {
    let _env = env_lock();
    let base = std::env::temp_dir().join(format!(
        "zo-hud-todo-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let cwd = base.join("project");
    std::fs::create_dir_all(&cwd).expect("cwd");
    let stale_default = runtime::zo_state_base(&cwd).join(".zo-todos.json");
    std::fs::create_dir_all(stale_default.parent().expect("default parent"))
        .expect("default parent dir");
    std::fs::write(
        &stale_default,
        r#"[{"content":"stale","activeForm":"stale","status":"pending"}]"#,
    )
    .expect("stale todo store");
    let session_store = base.join("session-todos.json");
    std::fs::write(
        &session_store,
        r#"[{"content":"fresh","activeForm":"doing fresh","status":"in_progress"}]"#,
    )
    .expect("session todo store");

    std::env::set_var("ZO_TODO_STORE", &session_store);
    let resolved = hud::todo_store_path_for_hud(Some(&cwd)).expect("resolved store");
    assert_eq!(resolved, session_store);
    let items = hud::load_todo_items_for_hud(&resolved);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].content, "fresh");
    assert_eq!(items[0].active_form, "doing fresh");
    assert_eq!(items[0].status, TodoChecklistStatus::InProgress);

    std::env::remove_var("ZO_TODO_STORE");
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn hud_shows_model_name() {
    let theme = Theme::no_color();
    let state = sample_state();
    let out = render_buffer(80, &state, &theme);
    assert!(
        out.contains("claude-opus-4-8"),
        "resolved model id missing: {out}"
    );
}

#[test]
fn hud_shows_window_pressure_when_compaction_threshold_is_unknown() {
    let theme = Theme::no_color();
    let state = sample_state();
    let out = render_buffer(80, &state, &theme);
    // All live ctx surfaces use the canonical percentage helper. With no
    // compaction threshold, 42k of a 200k nominal window falls back to 21%.
    assert!(out.contains("ctx 21%"), "context pressure missing: {out}");
    assert!(
        !out.contains("ctx ~42.0k") && !out.contains("tokens"),
        "HUD must not restore the retired token-count format: {out}"
    );
}

#[test]
fn hud_shows_cost() {
    let theme = Theme::no_color();
    let state = sample_state();
    let out = render_buffer(80, &state, &theme);
    assert!(out.contains("$0.37"), "cost missing: {out}");
}

#[test]
fn hud_zero_tokens_show_pending_usage() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.ctx_used = 0;
    let out = render_buffer(80, &state, &theme);
    assert!(
        out.contains("ctx pending"),
        "fresh usage should read as pending, not a broken-looking zero: {out}"
    );
    assert!(
        !out.contains("~0 tokens") && !out.contains("0 tokens"),
        "HUD must not surface an authoritative zero before usage arrives: {out}"
    );
}

#[test]
fn hud_caps_over_limit_context_pressure_at_one_hundred_percent() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.ctx_used = 1_370_000;
    state.ctx_limit = 1_000_000;

    let out = render_buffer(120, &state, &theme);
    assert!(
        out.contains("ctx 100%"),
        "over-limit context pressure should saturate at 100%: {out}"
    );
    assert!(
        !out.contains("ctx ~1.0M+") && !out.contains("1.4M"),
        "HUD must not restore cumulative-looking token counts: {out}"
    );
}

#[test]
fn hud_sonnet_model() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.model = active_model("anthropic", "claude-sonnet-4-6");
    let out = render_buffer(80, &state, &theme);
    assert!(
        out.contains("claude-sonnet-4-6"),
        "sonnet model id missing: {out}"
    );
}

#[test]
fn hud_openai_model_uses_short_model_label() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.model = active_model("openai", "openai:gpt-5.5-fast");
    let out = render_buffer(80, &state, &theme);
    assert!(
        out.contains("gpt-5.5-fast"),
        "OpenAI model label missing: {out}"
    );
    assert!(
        !out.contains("openai:gpt-5.5-fast"),
        "provider prefix should not crowd the model label: {out}"
    );
}

#[test]
fn hud_openai_generic_alias_uses_resolved_display_model() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.model = ActiveModel {
        provider: "openai",
        alias: "gpt".to_string(),
        display_name: "OpenAI GPT-5.5 Fast".to_string(),
        context_limit: 1_000_000,
    };

    let out = render_buffer(90, &state, &theme);
    assert!(
        out.contains("gpt-5.5-fast"),
        "generic alias should show the resolved model display id: {out}"
    );
    assert!(
        !out.contains(" gpt "),
        "HUD should not collapse a resolved OpenAI model to generic 'gpt': {out}"
    );
}

#[test]
fn hud_compose_returns_line_with_spans() {
    let theme = Theme::no_color();
    let state = sample_state();
    let line = hud::compose(&state, &theme, 80, false);
    assert!(!line.spans.is_empty());
}

#[test]
fn hud_permission_mode_labels() {
    assert_eq!(HudPermissionMode::ReadOnly.label(), "read-only");
    assert_eq!(HudPermissionMode::Workspace.label(), "workspace-write");
    assert_eq!(HudPermissionMode::All.label(), "danger-full-access");
}

#[test]
fn hud_shows_security_posture() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.security_posture = SecurityPosture::SandboxBlocked;
    let out = render_buffer(100, &state, &theme);
    assert!(
        out.contains("sandbox:blocked"),
        "security posture missing: {out}"
    );
}

#[test]
fn hud_shows_edit_activity_when_edits_happened() {
    let theme = Theme::no_color();
    let state = sample_state();
    let out = render_buffer(120, &state, &theme);
    assert!(out.contains("+4 edits"), "edit activity missing: {out}");
}

#[test]
fn hud_omits_edit_activity_when_no_edits_happened() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.edit_count = 0;
    let out = render_buffer(120, &state, &theme);
    assert!(
        !out.contains(" edits"),
        "zero edit activity should stay quiet: {out}"
    );
}

#[test]
fn hud_prioritizes_changed_files_over_edit_activity() {
    let theme = Theme::no_color();
    let mut state = sample_state();
    state.changed_files = 6;
    let out = render_buffer(120, &state, &theme);
    assert!(out.contains("+6 files"), "changed files missing: {out}");
    assert!(
        !out.contains("+4 edits"),
        "dirty file count should be the primary change signal: {out}"
    );
}

#[test]
fn hud_narrow_still_shows_model() {
    let theme = Theme::no_color();
    let state = sample_state();
    let out = render_buffer(40, &state, &theme);
    assert!(
        out.contains("claude-opus-4-8"),
        "model should remain at 40 cols: {out}"
    );
}
