//! Default constructors for [`super::App`]-owned state snapshots.

use runtime::message_stream::ActiveModel;

use crate::tui::hud::{HudState, PermissionMode, SecurityPosture};

/// Neutral initial HUD state used before the session-backed sync
/// (`build_hud_state` in `session::tui_loop`) runs and replaces it.
/// The permission badge defaults to the safest mode (`ReadOnly`) —
/// the outer loop overwrites it with the real `LiveCli.permission_mode`
/// via `set_hud_state` before the first draw, so this value is never
/// user-visible in a live session.
pub(super) fn default_hud_state() -> HudState {
    HudState {
        session_identity: None,
        model: ActiveModel {
            provider: "claude",
            alias: "opus".to_string(),
            display_name: "Claude Opus".to_string(),
            context_limit: 200_000,
        },
        turn_fallback_model: None,
        quota_fallback_model: None,
        ctx_used: 0,
        ctx_limit: 200_000,
        ctx_new_input: 0,
        ctx_cached: 0,
        compact_threshold: 0,
        cost_usd: 0.0,
        cost_approx: false,
        cwd: std::path::PathBuf::from("."),
        git_branch: None,
        auth_origin: None,
        perm_mode: PermissionMode::ReadOnly,
        security_posture: SecurityPosture::SandboxBlocked,
        effort: None,
        architect_impl: None,
        mcp_servers: Vec::new(),
        bash_count: 0,
        read_count: 0,
        edit_count: 0,
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
        status_line: None,
        team_inbox_unread: 0,
        stale_binary: None,
        background_tasks: 0,
        scheduled_wake: None,
    }
}
