//! Session-lifecycle commands: new, clear, compact, resume, session,
//! fork, rename, rewind, undo, redo.
//!
//! These mutate session state and/or the transcript view, so they take
//! the full [`DispatchCtx`]. View reseeding goes through
//! [`seed_transcript_from_session`].

use zo_cli::tui::hud::PermissionMode as TuiPermissionMode;

use super::context::{DispatchCtx, DispatchError};
use super::helpers_tui::seed_transcript_from_session;
use super::output::CommandOutput;

fn sync_agent_manifest_scope(
    app: &mut zo_cli::tui::App,
    started_after: u64,
    session_id: &str,
) {
    app.set_agent_manifest_started_after(started_after);
    app.set_agent_manifest_session_id(session_id.to_string());
}

fn reset_visible_session(ctx: &mut DispatchCtx) {
    ctx.app.reset_session_view();
    sync_agent_manifest_scope(
        ctx.app,
        ctx.cli.agent_manifest_started_after,
        &ctx.cli.session.id,
    );
}

pub(super) fn new(ctx: &mut DispatchCtx) -> Result<CommandOutput, DispatchError> {
    let report = ctx.cli.new_session_report()?;
    reset_visible_session(ctx);
    ctx.cli.persist_session()?;
    Ok(CommandOutput::info(report))
}

pub(super) fn clear(ctx: &mut DispatchCtx, confirm: bool) -> Result<CommandOutput, DispatchError> {
    let report = ctx.cli.clear_session_report(confirm)?;
    if confirm {
        reset_visible_session(ctx);
        ctx.cli.persist_session()?;
    }
    Ok(CommandOutput::info(report))
}

pub(super) fn compact(
    ctx: &mut DispatchCtx,
    instructions: Option<String>,
) -> Result<CommandOutput, DispatchError> {
    let report = ctx.cli.compact_report(instructions)?;
    Ok(CommandOutput::info(report))
}

pub(super) fn resume(
    ctx: &mut DispatchCtx,
    session_path: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    if session_path.is_some() {
        // Fast path: swap session without rebuilding MCP/LSP/plugins.
        let report = ctx.cli.resume_session_fast(session_path)?;
        reset_visible_session(ctx);
        seed_transcript_from_session(ctx.app, ctx.ids, ctx.cli.runtime.session());
        return Ok(CommandOutput::info(report));
    }
    // No argument — show interactive session picker. Cap at the 10 most-recent:
    // the registry only parses those (cheap mtime sort first), so the picker
    // stays fast even when many sessions live on disk.
    let sessions: Vec<_> = crate::session_registry::list_managed_sessions_limited(Some(10))?
        .into_iter()
        .filter(|s| s.message_count > 0)
        .collect();
    if sessions.is_empty() {
        return Ok(CommandOutput::info("No saved sessions found."));
    }
    let labels: Vec<String> = sessions
        .iter()
        .map(|s| {
            let title = s
                .first_user_text
                .as_deref()
                .unwrap_or(&s.id[..s.id.len().min(12)]);
            format!("{title}  ({} msgs)", s.message_count)
        })
        .collect();
    let ids_list: Vec<String> = sessions.iter().map(|s| s.id.clone()).collect();
    ctx.app.open_session_modal(labels, ids_list);
    Ok(CommandOutput::Quiet)
}

pub(super) fn session(
    ctx: &mut DispatchCtx,
    action: Option<&str>,
    target: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    let (report, changed) = ctx.cli.session_command_report(action, target)?;
    if changed {
        reset_visible_session(ctx);
        seed_transcript_from_session(ctx.app, ctx.ids, ctx.cli.runtime.session());
    }
    Ok(CommandOutput::info(report))
}

pub(super) fn name(ctx: &mut DispatchCtx, name: Option<&str>) -> CommandOutput {
    // One implementation shared with the headless REPL: Ok → info (show/set),
    // Err → warn (rejected argument).
    match ctx.cli.set_display_name(name) {
        Ok(report) => CommandOutput::info(report),
        Err(usage) => CommandOutput::warn(usage),
    }
}

/// `/fork [name]` — exactly `/session fork [name]`. The old dedicated
/// implementation saved an unmanaged copy without switching to it; delegating
/// keeps one fork path (managed registry handle + runtime switch + reseed).
pub(super) fn fork(
    ctx: &mut DispatchCtx,
    name: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    session(ctx, Some("fork"), name)
}

pub(super) fn rename(ctx: &mut DispatchCtx, name: Option<&str>) -> CommandOutput {
    // A session's id IS its name; reuse the same `LiveCli::rename_session` the
    // REPL `/rename` uses (persist under the new id, drop the old file, rebuild
    // the runtime), then reseed the visible transcript from the renamed session.
    let Some(name) = name else {
        return CommandOutput::warn("Usage: /rename <name>".to_string());
    };
    match ctx.cli.rename_session(name) {
        Ok(report) => {
            reset_visible_session(ctx);
            seed_transcript_from_session(ctx.app, ctx.ids, ctx.cli.runtime.session());
            CommandOutput::info(report)
        }
        Err(error) => CommandOutput::error(format!("Rename failed: {error}")),
    }
}

pub(super) fn rewind(
    ctx: &mut DispatchCtx,
    action: &commands::WorkspaceRewindAction,
) -> CommandOutput {
    // Bare `/rewind` opens the interactive rewind viewer (the Esc-Esc modal)
    // instead of printing a static turn list; explicit restores keep the
    // textual receipt.
    if matches!(action, commands::WorkspaceRewindAction::List) {
        crate::session::tui_loop::open_rewind_viewer(ctx.cli, ctx.app, ctx.ids);
        return CommandOutput::Quiet;
    }
    match ctx.cli.workspace_rewind_report(action) {
        Ok(report) => CommandOutput::info(report),
        Err(error) => CommandOutput::error(format!("Workspace rewind failed: {error}")),
    }
}

pub(super) fn undo(ctx: &mut DispatchCtx, steps: Option<&str>) -> CommandOutput {
    let n: usize = steps.unwrap_or("1").parse().unwrap_or(1).max(1);
    let Some(snapshots) = ctx.cli.snapshot_stack.as_mut() else {
        return CommandOutput::warn("Undo\n  Not in a git repository — snapshot undo unavailable.");
    };
    let mut undone = 0;
    for _ in 0..n {
        match snapshots.undo() {
            Ok(result) => {
                undone += 1;
                if result.remaining == 0 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    if undone == 0 {
        CommandOutput::info("Undo\n  No previous state to undo to.")
    } else {
        CommandOutput::info(format!(
            "Undo\n  Reverted {undone} step(s). {remaining} snapshots remaining.",
            remaining = snapshots.depth()
        ))
    }
}

pub(super) fn redo(ctx: &mut DispatchCtx, steps: Option<&str>) -> CommandOutput {
    let n: usize = steps.unwrap_or("1").parse().unwrap_or(1).max(1);
    let Some(snapshots) = ctx.cli.snapshot_stack.as_mut() else {
        return CommandOutput::warn("Redo\n  Not in a git repository — snapshot redo unavailable.");
    };
    let mut redone = 0;
    for _ in 0..n {
        match snapshots.redo() {
            Ok(_) => redone += 1,
            Err(_) => break,
        }
    }
    if redone == 0 {
        CommandOutput::info("Redo\n  Nothing to redo.")
    } else {
        CommandOutput::info(format!(
            "Redo\n  Restored {redone} step(s). {remaining} redo(s) remaining.",
            remaining = snapshots.redo_depth()
        ))
    }
}

/// The runtime permission spelling the parser accepts for a TUI badge. Plan is
/// runtime read-only, so `/plan off` restoring a `Plan` badge would be
/// nonsensical — but [`App::exit_plan_mode`] never returns `Plan`, so map it to
/// the read-only spelling as a safe floor.
const fn runtime_spelling(mode: TuiPermissionMode) -> &'static str {
    match mode {
        TuiPermissionMode::ReadOnly | TuiPermissionMode::Plan => "read-only",
        TuiPermissionMode::Workspace => "workspace-write",
        TuiPermissionMode::All => "danger-full-access",
    }
}

/// `/plan on|off` — the TUI-visible plan-first gate.
///
/// `on` flips the HUD badge to `plan` (remembering the mode being left so
/// `off` can restore it) and makes the runtime read-only; the model can draft a
/// plan but cannot edit. `off` restores the remembered mode (or `Workspace`
/// when none was recorded — e.g. the gate was reached via the Shift+Tab cycle)
/// and re-grants the matching write access. Approval is always a human action;
/// the model never self-restores write.
pub(super) fn plan(
    ctx: &mut DispatchCtx,
    mode: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    match mode {
        Some("on" | "start") => {
            // Transactional: mutate the App plan-gate first, but commit only if
            // the runtime read-only switch succeeds. On failure, roll the App
            // back to the exact prior state and leave `plan_selected` unchanged
            // so the UI flag never diverges from the runtime.
            let rollback = ctx.app.plan_mode_snapshot();
            ctx.app.enter_plan_mode();
            let report = match ctx.cli.apply_permission_change("read-only") {
                Ok(report) => report,
                Err(error) => {
                    ctx.app.restore_plan_mode_snapshot(rollback);
                    return Err(error.into());
                }
            };
            ctx.cli.set_plan_selected(true);
            ctx.cli.persist_session()?;
            Ok(CommandOutput::info(format!(
                "Plan mode on — the session is now read-only. Ask the assistant to plan; run /plan off to approve and resume editing.\n{report}"
            )))
        }
        Some("off" | "stop") => {
            if !ctx.app.plan_mode_active() {
                return Ok(CommandOutput::info(
                    "Plan mode is not active — nothing to approve.",
                ));
            }
            let rollback = ctx.app.plan_mode_snapshot();
            let restored = ctx.app.exit_plan_mode();
            let spelling = runtime_spelling(restored);
            let report = match ctx.cli.apply_permission_change(spelling) {
                Ok(report) => report,
                Err(error) => {
                    ctx.app.restore_plan_mode_snapshot(rollback);
                    return Err(error.into());
                }
            };
            ctx.cli.set_plan_selected(false);
            ctx.cli.persist_session()?;
            Ok(CommandOutput::info(format!(
                "Plan mode off — approved; restored {spelling} permissions.\n{report}"
            )))
        }
        None => {
            ctx.app
                .open_arg_picker("plan", "/plan", vec!["on".to_string(), "off".to_string()]);
            Ok(CommandOutput::Quiet)
        }
        Some(_) => Ok(CommandOutput::info(
            "Plan\n  Usage            /plan [on|off]\n  on               make the session read-only (plan-first gate)\n  off               approve the plan and restore write access",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::sync_agent_manifest_scope;
    use zo_cli::tui::App;
    use zo_cli::tui::theme::Theme;
    use tokio::sync::mpsc;

    fn test_app() -> App {
        let (_tx, rx) = mpsc::channel(8);
        let (cmd_tx, _cmd_rx) = mpsc::channel(8);
        App::new(Theme::no_color(), rx, cmd_tx)
    }

    #[test]
    fn sync_agent_manifest_scope_updates_time_and_session_id() {
        let mut app = test_app();
        sync_agent_manifest_scope(&mut app, 111, "old-session");
        sync_agent_manifest_scope(&mut app, 222, "new-session");

        assert_eq!(app.agent_manifest_started_after(), 222);
        assert_eq!(app.agent_manifest_session_id(), Some("new-session"));
    }
}
