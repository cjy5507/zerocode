//! HUD snapshot session boundary for the interactive turn loop.
//!
//! The render loop in [`super`](../turn_controller.rs) polls four background
//! snapshots every tick — the live agent/todo/workflow HUD, the changed-files
//! git status, and the open workflow/agents viewers. Each one is a disk read
//! (a manifest scan or `git status`) that must never ride the render thread, so
//! they are offloaded onto a dedicated blocking runtime and drained by handle.
//!
//! This module owns that spawn → poll → assemble machinery so the turn loop
//! only *coordinates* it: it decides when to kick a snapshot off and where the
//! finished value goes, while the runtime plumbing and the DTO assembly live
//! here. The DTO is [`LiveHudSnapshot`]; it carries only neutral value types
//! (`u16`, `Vec<AgentTaskSummary>`, …) across the boundary, and the sole coupling
//! to the TUI is the `&mut App` sink the loop already owns — no new TUI type
//! dependency is introduced here.

use std::path::PathBuf;

use tokio::task::JoinHandle;

use zo_cli::tui::hud::{AgentTaskSummary, TodoChecklistItem};
use zo_cli::tui::modals::workflow_viewer::WorkflowView;
use zo_cli::tui::sidebar::GitStatusSnapshot;
use zo_cli::tui::workflow_progress::{AgentRowsSnapshot, WorkflowSummary};
use zo_cli::tui::App;

use super::super::freshness::{FreshnessDomain, SessionFreshness};

/// One assembled live-HUD frame: the running agent count, the todo checklist,
/// the per-agent rows, and the optional workflow summary. Fields are
/// crate-visible so the turn loop can read them directly (token totals, the
/// fanout-collection close) without accessor churn.
pub(super) struct LiveHudSnapshot {
    pub(super) running: u16,
    pub(super) todos: Vec<TodoChecklistItem>,
    pub(super) agents: Vec<AgentTaskSummary>,
    pub(super) workflow: Option<WorkflowSummary>,
}

/// Dedicated blocking runtime for the HUD/git-status snapshots the render loop
/// spawns every tick. If they shared the main runtime's blocking pool with tool
/// execution, a burst of slow tools (an SSH/DB query, several MCP calls) could
/// exhaust that pool and leave the snapshots queued indefinitely — so
/// `is_finished()` never trips, the render loop stops refreshing, and the UI
/// freezes for as long as the tools hold the pool. Giving the snapshots their
/// own runtime guarantees they always have a worker, no matter how saturated the
/// main pool is. It carries a small blocking pool of its own; the work is light
/// (a directory scan and a `git status`) and self-throttled by the caller.
pub(super) fn hud_runtime() -> &'static tokio::runtime::Runtime {
    static HUD_RT: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    HUD_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(4)
            .thread_name("zo-hud")
            .enable_all()
            .build()
            .expect("build HUD snapshot runtime")
    })
}

pub(super) fn spawn_live_hud_snapshot(
    agent_started_after: u64,
    session_id: Option<String>,
) -> JoinHandle<LiveHudSnapshot> {
    hud_runtime().spawn_blocking(move || {
        read_live_hud_snapshot(agent_started_after, session_id.as_deref())
    })
}

pub(super) fn read_live_hud_snapshot(
    agent_started_after: u64,
    session_id: Option<&str>,
) -> LiveHudSnapshot {
    let agents = super::super::tui_loop::list_running_agents_since(agent_started_after, session_id);
    let running = super::super::tui_loop::running_count(&agents);
    let todos = super::super::tui_loop::todo_items();
    let workflow = zo_cli::tui::workflow_progress::read_summary_since(
        agent_started_after,
        session_id,
    );
    LiveHudSnapshot {
        running,
        todos,
        agents,
        workflow,
    }
}

pub(super) fn apply_live_hud_snapshot(app: &mut App, snapshot: LiveHudSnapshot) {
    app.update_hud_live_snapshot(
        snapshot.running,
        snapshot.todos,
        snapshot.agents,
        snapshot.workflow,
    );
}

pub(super) async fn load_live_hud_snapshot(app: &App) -> Option<LiveHudSnapshot> {
    spawn_live_hud_snapshot(
        app.agent_manifest_started_after(),
        app.agent_manifest_session_id().map(str::to_string),
    )
    .await
    .ok()
}

pub(super) async fn refresh_live_hud_snapshot(app: &mut App) -> bool {
    if let Some(snapshot) = load_live_hud_snapshot(app).await {
        apply_live_hud_snapshot(app, snapshot);
        true
    } else {
        false
    }
}

pub(super) fn spawn_workflow_view_snapshot(
    agent_started_after: u64,
    session_id: Option<String>,
) -> JoinHandle<Option<WorkflowView>> {
    hud_runtime().spawn_blocking(move || {
        zo_cli::tui::workflow_progress::read_view_refresh_since(
            agent_started_after,
            session_id.as_deref(),
        )
    })
}

/// Background manifest read for the open Ctrl+G agents viewer — same offload
/// pattern as the workflow snapshot so disk IO never rides the render tick.
pub(super) fn spawn_agent_rows_snapshot(
    agent_started_after: u64,
    session_id: Option<String>,
    include_history: bool,
) -> JoinHandle<AgentRowsSnapshot> {
    hud_runtime().spawn_blocking(move || {
        zo_cli::tui::workflow_progress::read_agent_rows_since(
            agent_started_after,
            session_id.as_deref(),
            include_history,
        )
    })
}

pub(super) fn spawn_changed_files_snapshot(
    cwd: PathBuf,
    freshness: &SessionFreshness,
) -> JoinHandle<Option<GitStatusSnapshot>> {
    let source = freshness.workspace_status();
    let should_interrupt = freshness.dirty_flag(FreshnessDomain::Workspace);
    hud_runtime().spawn_blocking(move || {
        source.snapshot(&cwd, should_interrupt).ok()
    })
}
