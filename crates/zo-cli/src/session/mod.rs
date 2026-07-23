// L7c submodules. See the lane handoff for the directory split
// rationale — these modules are additive and don't disturb the
// existing `run_repl` / `LiveCli` surface. A follow-up lane wires
// the TUI driver into `run_repl` itself.
mod agent_notice;
mod auto_fanout;
mod automation;
mod built_runtime;
mod confidence_cascade;
mod freshness;
mod grind_escalation;
mod ide_bridge;
mod live_cli;
mod live_cli_actions;
mod live_cli_commands;
mod live_cli_pickers;
mod live_cli_reports;
mod loop_arms;
mod lsp_runtime;
mod mcp_runtime;
mod ndjson_summary;
pub(crate) use ndjson_summary::StreamPrompter;
pub(crate) mod permission_bridge;
pub(crate) mod report_services;
mod request_types;
pub(crate) mod restart;
pub(crate) mod runtime_bridge;
mod self_improve;
mod runtime_builder;
mod session_preferences;
pub(crate) mod smart_settings;
pub(crate) mod slash_dispatch;
pub(crate) mod socket_permission;
mod startup_snapshot;
pub(crate) mod status_line;
mod tool_toggles;
pub(crate) mod tui_loop;
pub(crate) mod turn_controller;
mod turn_harness;
pub(crate) mod user_question_bridge;
mod wakeups;

use std::path::PathBuf;
use std::time::Instant;

use runtime::PermissionMode;

use crate::cli_args::AllowedToolSet;

pub(crate) use built_runtime::{BuiltRuntime, RuntimePluginState};
pub(crate) use live_cli::{LiveCli, ManagedSessionSummary, SessionHandle};
pub(crate) use live_cli_commands::{
    ShareArtifact, delete_share_gist, share_gist_warning, upload_share_to_gist,
    write_share_artifact, write_to_clipboard,
};
pub(crate) use lsp_runtime::{RuntimeLspState, build_runtime_lsp_state};
pub(crate) use mcp_runtime::{PendingMcpImages, RuntimeMcpState, build_runtime_mcp_state};
#[cfg(test)]
pub(crate) use mcp_runtime::discover_pending_mcp_tools_now;
#[cfg(test)]
pub(crate) use ndjson_summary::write_ndjson_summary;
pub(crate) use session_preferences::project_effort_preference;
// Re-exported so the `serve` RPC layer can run the commit→push→PR flow against a
// remote session's own cwd (`session.commit_push_pr`), mirroring the local slash
// command path.
pub(crate) use request_types::{
    ListMcpResourcesRequest, McpToolRequest, ReadMcpResourceRequest, ToolSearchRequest,
};
pub(crate) use slash_dispatch::handle_commit_push_pr_at;

pub(crate) fn run_repl(
    mut model: String,
    model_pinned: bool,
    allowed_tools: Option<AllowedToolSet>,
    permission_mode: PermissionMode,
    mcp_config: Option<PathBuf>,
    inline: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::self_update::schedule_startup_check();
    let startup_start = Instant::now();
    // `/restart` re-execs a bare `zo` carrying only a resume-handoff env var
    // (see `session::restart`). When present, reopen that session's transcript
    // into this fresh interactive TUI, restoring the model/effort it was using
    // from the session's own preference sidecar (the pre-exec `persist_session`
    // wrote them there) rather than from re-passed argv. Resolving the model
    // *before* the runtime is built avoids a second runtime rebuild.
    let boot_resume = restart::take_boot_resume_request();
    let boot_resume_prefs = boot_resume
        .as_deref()
        .map(session_preferences::load_session_preferences);
    if let Some(persisted_model) = boot_resume_prefs
        .as_ref()
        .and_then(|prefs| prefs.model.clone())
    {
        model = persisted_model;
    }
    let mut cli = LiveCli::new_scoped_with_mcp_config_and_session_id(
        model,
        true,
        allowed_tools,
        permission_mode,
        crate::session_registry::SessionScope::Project,
        mcp_config,
        None,
        crate::runtime_support::StartupAuthPolicy::AllowUnauthenticated,
    )?;
    cli.set_model_user_pinned(model_pinned);
    if let Some(path) = &boot_resume {
        // Swap the fresh empty session for the persisted transcript. The fast
        // path keeps the just-built MCP/LSP/plugin runtime and only replaces the
        // session, so the boot cost of the redeploy stays low. Fail-open: a
        // missing/corrupt transcript degrades to a fresh session rather than
        // aborting the launch the user just triggered.
        match cli.resume_session_fast(Some(&path.display().to_string())) {
            Ok(_) => {
                if let Some(budget) =
                    boot_resume_prefs.as_ref().and_then(|prefs| prefs.effort_budget)
                {
                    // Restoring the resumed budget re-persists it. This runs at
                    // startup with no interactive user to read a warning, and the
                    // value came FROM disk, so a re-persist failure here is not
                    // actionable — the next `/effort`/model change surfaces it.
                    // Explicitly discard the warning so the `#[must_use]` return
                    // is consumed rather than silently dropped.
                    let _persist_warning = cli.set_effort_budget(budget);
                }
            }
            Err(error) => eprintln!(
                "[zo] restart: could not resume {} ({error}); starting a fresh session",
                path.display()
            ),
        }
    }
    let startup_elapsed = startup_start.elapsed();
    // Use a multi-thread runtime so that the synchronous tool runtimes
    // (bash, mcp_runtime, lsp_runtime, agent_tools, …) — each of which
    // builds its own private current-thread runtime and calls
    // `block_on` — can run on a worker thread distinct from the one
    // driving the TUI/select loop. With a single-threaded outer
    // runtime, those nested `block_on` calls panic with
    // "Cannot start a runtime from within a runtime", which surfaces
    // as a crash on focus-return / mid-turn tool execution.
    //
    // `max_blocking_threads` must be generous: every ordinary tool runs through
    // `spawn_blocking` (bash, MCP RPC, WebFetch, Agent, Workflow, session_recall
    // …) and a single slow one — an SSH/DB query, an MCP server that reasons for
    // tens of seconds — pins one worker for its whole lifetime. Pass 2 can also
    // fan out up to `MAX_PARALLEL_SAFE_TOOL_DISPATCHES` read-only tools at once.
    // At the old cap of 8, a handful of concurrent slow tools exhausted the pool
    // and any further `spawn_blocking` (including the per-turn tool dispatch
    // itself) queued behind them, stalling the turn. The HUD/git-status
    // snapshots that the render loop polls now run on their own dedicated
    // runtime (see `hud_runtime`) so they can never be starved by tool work, but
    // the main pool is still widened well above the worst-case concurrent-tool
    // count so tool dispatch itself is never queue-blocked. Blocking threads are
    // cheap (reclaimed when idle), so a high ceiling costs nothing at rest.
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(512)
        .thread_name("zo-rt")
        .enable_all()
        .build()?;
    // Fire-and-forget: pre-warm the HTTP connection pool (TCP + TLS handshake)
    // in the background so the first real API call reuses an established
    // connection. Errors are silently ignored.
    let warmup_client = cli.runtime.api_client().client();
    tokio_rt.spawn(async move {
        if let api::ProviderClient::Anthropic(client) = warmup_client {
            client.warm_connection().await;
        }
    });

    // Fire-and-forget: pre-warm the syntect syntax/theme assets on a blocking
    // worker during the idle gap before the first prompt. The assets live in a
    // `OnceLock` that is otherwise filled lazily inside `draw()` on the first
    // code block — and `draw()` runs on the TUI select! thread, so that first
    // load blocks the spinner/stream for tens of ms (the "first-output freeze"
    // on a code-heavy reply). Loading it here makes the first render a cache hit.
    tokio_rt.spawn_blocking(zo_cli::tui::markdown::prewarm_syntect_assets);

    // Fire-and-forget: run the Dreamer's between-sessions memory curation
    // automatically at startup (no slash command needed). It is throttled to at
    // most one pass per `DEFAULT_AUTO_DREAM_INTERVAL`, so frequent relaunches
    // coalesce, and it only promotes lessons that were repeated across distinct
    // sessions *and* verified — so a background pass can never pollute memory.
    // Runs on a blocking thread (it is filesystem IO) and never blocks startup.
    if should_run_startup_auto_dream(&cli.runtime.feature_config) {
        let dream_cwd = cli.cwd.clone();
        tokio_rt.spawn_blocking(move || {
            // Background, best-effort: a failed or empty pass must never disrupt
            // startup. Persist a tiny diagnostic so automatic failures are
            // observable without adding TUI noise. Successful promotions are
            // visible in memory recall on later sessions.
            if let Err(error) =
                runtime::maybe_auto_dream(&dream_cwd, runtime::memory::DEFAULT_AUTO_DREAM_INTERVAL)
            {
                let _ = runtime::record_auto_dream_failure(&dream_cwd, &error);
            }
        });
    }

    if should_run_startup_auto_self_improve(&cli.runtime.feature_config) {
        let improve_cwd = cli.cwd.clone();
        // Opt-in `autoImproveProposalsEnabled` (default off): when on, the
        // preflight also runs the headless generator and parks a gated proposal
        // automatically. Applying always stays an explicit `/improve apply`.
        let auto_propose = cli
            .runtime
            .feature_config
            .auto_improve_proposals_enabled();
        tokio_rt.spawn_blocking(move || {
            // Background, best-effort: a failed pass must never disrupt startup.
            // Off (default) it runs only the safe read-only scheduler preflight;
            // on, it may spend a minutes-long generator turn here, off the boot
            // path. Either way `/improve apply` remains a human gate.
            if let Err(error) =
                self_improve::maybe_auto_self_improve_preflight(&improve_cwd, auto_propose)
            {
                self_improve::record_auto_self_improve_failure(&improve_cwd, &error);
            }
        });
    }

    let terminal_mode = zo_cli::tui::TerminalMode::from_inline(
        inline || cli.runtime.feature_config.tui_inline_mode(),
    );
    let outcome = tokio_rt.block_on(tui_loop::run_repl_session(
        &mut cli,
        startup_elapsed,
        terminal_mode,
    ));

    // `/restart` set a re-exec plan and quit the loop cleanly. By here
    // `run_repl_session` has fully torn the TUI down — terminal restored, stderr
    // redirect dropped — so the child inherits a clean tty. `exec` replaces this
    // process on success and only returns on failure; on failure the session is
    // already persisted, so report the error and exit cleanly (never panic).
    if outcome.is_ok() {
        if let Some(plan) = cli.pending_restart.take() {
            let error = plan.exec();
            eprintln!("[zo] restart failed: {error}");
            eprintln!("[zo] {}", plan.manual_recovery_hint());
        }
    }

    outcome.map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })
}

pub(crate) use runtime_builder::build_runtime_plugin_state_with_loader;

fn should_run_startup_auto_dream(feature_config: &runtime::RuntimeFeatureConfig) -> bool {
    feature_config.dream_automation_enabled()
}

fn should_run_startup_auto_self_improve(feature_config: &runtime::RuntimeFeatureConfig) -> bool {
    feature_config.dream_automation_enabled()
}

#[cfg(test)]
mod tests {
    use super::{should_run_startup_auto_dream, should_run_startup_auto_self_improve};

    #[test]
    fn startup_auto_dream_gate_respects_feature_config() {
        assert!(should_run_startup_auto_dream(
            &runtime::RuntimeFeatureConfig::default()
        ));
        assert!(!should_run_startup_auto_dream(
            &runtime::RuntimeFeatureConfig::default().with_auto_dream_enabled(false)
        ));
    }

    #[test]
    fn startup_auto_self_improve_gate_respects_feature_config() {
        assert!(should_run_startup_auto_self_improve(
            &runtime::RuntimeFeatureConfig::default()
        ));
        assert!(!should_run_startup_auto_self_improve(
            &runtime::RuntimeFeatureConfig::default().with_auto_dream_enabled(false)
        ));
    }
}
