//! Actionable "system" commands that do real work (apply a model /
//! permission change, write a file, query a service) but don't fit the
//! session-lifecycle, git, auth, view, or toggle groups.

use commands::{GoalCommand, LoopCommand, SelfImproveAction};

use super::context::{DispatchCtx, DispatchError};
use super::handlers::{
    handle_ant_trace, handle_extra_usage, handle_perf_issue, handle_release_notes,
    handle_statusline,
};
use super::helpers_tui::build_model_entries;
use super::output::CommandOutput;
use crate::{
    build_bughunter_prompt, build_council_prompt, build_distill_prompt, build_ultraplan_prompt,
    render_export_text,
};

pub(super) fn model(
    ctx: &mut DispatchCtx,
    model: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    if let Some(model) = model {
        let report = ctx.cli.apply_model_change(model);
        ctx.cli.persist_session()?;
        Ok(CommandOutput::info(report))
    } else {
        let entries = build_model_entries(ctx.cli);
        ctx.app.open_model_modal(entries);
        Ok(CommandOutput::Quiet)
    }
}

pub(super) fn permissions(
    ctx: &mut DispatchCtx,
    mode: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    if let Some(mode) = mode {
        let report = ctx.cli.apply_permission_change(mode)?;
        ctx.cli.persist_session()?;
        Ok(CommandOutput::info(report))
    } else {
        let current = ctx.cli.permission_mode;
        ctx.app.open_permission_picker_modal(current);
        Ok(CommandOutput::Quiet)
    }
}

pub(super) fn init(ctx: &mut DispatchCtx) -> CommandOutput {
    // Scaffold the static structure first (`.zo/` dirs, settings, topic-doc
    // and sub-agent stubs, plus a placeholder `context.md`). Then queue an agent
    // turn that rewrites `context.md` from the real codebase — the Claude Code
    // `/init` behavior — instead of leaving the repo-marker-only template.
    let report = match crate::init_context_md() {
        Ok(report) => report,
        Err(e) => return CommandOutput::error(e.to_string()),
    };
    if let Err(error) = ctx.app.queue_message(crate::build_init_prompt()) {
        // The scaffold already wrote its files; only the follow-up analysis turn
        // failed to queue. Say so rather than implying the whole command failed.
        return CommandOutput::warn(format!(
            "{report}\n  Analysis         NOT queued ({error})\n  Status           scaffold written; run /init again to queue the repo-analysis turn"
        ));
    }
    CommandOutput::info(format!(
        "{report}\n  Analysis         queued repo-analysis turn (rewrites the instruction file from the codebase)\n  Status           will run after this command"
    ))
}

/// `/restart` — persist this session, tear the TUI down cleanly, and re-exec the
/// newest build on disk resuming the same conversation. Backs the stale-binary
/// sidebar badge.
///
/// This handler does everything that must happen *before* the terminal is torn
/// down — gate, resolve the re-exec, persist — and then records the plan and
/// asks the loop to exit via [`CommandOutput::Exit`]. `run_repl` runs the actual
/// `exec` after the loop returns and the terminal is restored (see
/// [`crate::session::restart`]). Nothing here is teardown itself, so a rejected
/// gate or an unresolvable binary leaves the live session fully intact.
pub(super) fn restart(ctx: &mut DispatchCtx) -> Result<CommandOutput, DispatchError> {
    use crate::session::restart::{evaluate_restart_readiness, RestartPlan, RestartReadiness};

    // Pre-flight gate: refuse while a turn is running or messages are queued, so
    // the re-exec never cuts off live work or silently drops the (unpersisted)
    // input queue. Nothing has been torn down yet, so this is a clean refusal.
    let turn_active = ctx.app.turn_activity().is_some();
    let queued = ctx.app.queued_message_count();
    if let RestartReadiness::Blocked(message) = evaluate_restart_readiness(turn_active, queued) {
        return Ok(CommandOutput::error(message));
    }

    // Resolve the re-exec up front (locates the running binary). A failure here
    // is reported as a command error while the session is still alive, never as
    // a crash mid-teardown.
    let plan = match RestartPlan::resolve(ctx.cli.session.path.clone(), ctx.cli.cwd.clone()) {
        Ok(plan) => plan,
        Err(error) => {
            return Ok(CommandOutput::error(format!(
                "Restart\n  Not restarted    {error}"
            )));
        }
    };

    // Persist so the child resumes exactly this state (transcript + model/effort
    // sidecar). A persist failure aborts the restart rather than relaunching into
    // a stale or empty session.
    ctx.cli.persist_session()?;

    // Hand the plan to `run_repl`, which execs it after the terminal is restored.
    ctx.cli.pending_restart = Some(plan);
    Ok(CommandOutput::Exit)
}

pub(super) fn export(ctx: &mut DispatchCtx, path: Option<&str>) -> CommandOutput {
    match crate::resolve_export_path(path, ctx.cli.runtime.session()) {
        Ok(export_path) => {
            let text = render_export_text(ctx.cli.runtime.session());
            match crate::write_atomic(&export_path, text.as_bytes()) {
                Ok(()) => CommandOutput::info(format!(
                    "Export\n  Result           wrote transcript\n  File             {}\n  Characters       {}",
                    export_path.display(),
                    text.len()
                )),
                Err(e) => CommandOutput::error(e.to_string()),
            }
        }
        Err(e) => CommandOutput::error(e.to_string()),
    }
}

/// `/dump` — serialize the transcript to a temp artifact and hand it to the
/// host loop to open in `$PAGER` (real `/pattern` search in `less`; `edit`
/// swaps in `$EDITOR`). The escape hatch out of the alt-screen: rendering
/// reuses the `/export` serializer so all three raw-text surfaces (/export,
/// /share, /dump) stay consistent.
pub(super) fn dump(ctx: &mut DispatchCtx, edit: bool) -> CommandOutput {
    let session = ctx.cli.runtime.session();
    if session.messages.is_empty() {
        return CommandOutput::warn("Dump\n  Nothing to show yet — the transcript is empty.");
    }
    match write_transcript_dump(&std::env::temp_dir(), &ctx.cli.session.id, session) {
        Ok((path, chars)) => {
            let viewer = if edit {
                "$EDITOR"
            } else {
                "$PAGER (less: `/` to search, `q` to quit)"
            };
            ctx.app.request_transcript_view(path.clone(), edit);
            CommandOutput::info(format!(
                "Dump\n  File             {}\n  Characters       {chars}\n  Viewer           {viewer}",
                path.display()
            ))
        }
        Err(e) => CommandOutput::error(format!("Dump\n  Write failed — {e}")),
    }
}

/// Core of [`dump`], parameterised on the destination directory so it is
/// testable without a full [`DispatchCtx`]: writes the `/export`-rendered
/// transcript to `<base>/zo-transcript-<session-id>.txt` and returns the
/// path plus character count.
fn write_transcript_dump(
    base: &std::path::Path,
    session_id: &str,
    session: &runtime::Session,
) -> std::io::Result<(std::path::PathBuf, usize)> {
    let path = base.join(format!("zo-transcript-{session_id}.txt"));
    let text = render_export_text(session);
    std::fs::write(&path, &text)?;
    Ok((path, text.len()))
}

pub(super) fn copy(ctx: &mut DispatchCtx, target: Option<&str>) -> CommandOutput {
    let session = ctx.cli.runtime.session();
    let payload = match target {
        None | Some("last") => session.messages.iter().rev().find_map(|m| {
            m.blocks
                .iter()
                .rev()
                .map(|b| match b {
                    runtime::ContentBlock::Text { text } => text.clone(),
                    runtime::ContentBlock::ToolResult { output, .. } => output.clone(),
                    runtime::ContentBlock::ToolUse { input, .. } => input.clone(),
                    runtime::ContentBlock::Image { media_type, .. } => {
                        format!("[image: {media_type}]")
                    }
                    runtime::ContentBlock::Thinking { .. } => "[thinking]".to_string(),
                    runtime::ContentBlock::RedactedThinking { .. } => {
                        "[redacted thinking]".to_string()
                    }
                })
                .next()
        }),
        Some("all") => Some(render_export_text(session)),
        Some(other) => {
            return CommandOutput::error(format!(
                "Unknown copy target: {other}\nUsage: /copy [last|all]"
            ));
        }
    };
    match payload {
        Some(text) => match crate::session::write_to_clipboard(&text) {
            Ok(sink) => CommandOutput::info(format!(
                "Copied to clipboard {} ({} chars)",
                sink.describe(),
                text.len()
            )),
            Err(e) => CommandOutput::error(format!("Clipboard error: {e}")),
        },

        None => CommandOutput::Quiet,
    }
}

pub(super) fn hooks(args: Option<&str>) -> CommandOutput {
    match crate::render_hooks_report() {
        Ok(r) => CommandOutput::popup(
            "/hooks",
            format!(
                "{r}{}",
                args.map(|a| format!("\n  Args             {a}"))
                    .unwrap_or_default()
            ),
        ),
        Err(e) => CommandOutput::error(e.to_string()),
    }
}

pub(super) fn reload_context(ctx: &mut DispatchCtx) -> Result<CommandOutput, DispatchError> {
    Ok(CommandOutput::info(ctx.cli.reload_context()?))
}

/// `/memory` → open the project `context.md` instruction file in `$EDITOR`.
/// The host loop owns the terminal, so we record the request on the app; the
/// loop suspends the TUI, runs the editor, then reloads context so the edits
/// take effect this session. (Structured global project-memory entries are
/// written with the `MemoryWrite` tool; this command edits the human-authored
/// instruction file.)
pub(super) fn edit_memory(ctx: &mut DispatchCtx) -> CommandOutput {
    let path = resolve_memory_edit_path(&ctx.cli.cwd);
    let display = path.display().to_string();
    ctx.app.request_file_edit(path);
    CommandOutput::info(format!(
        "Memory\n  Opening          {display}\n  Editor           $EDITOR (context reloads on save)"
    ))
}

/// Pick the `context.md` instruction file `/memory` always opens.
fn resolve_memory_edit_path(cwd: &std::path::Path) -> std::path::PathBuf {
    cwd.join("context.md")
}

/// `/dream` — force the between-sessions memory curation pass now (it also runs
/// automatically at startup, throttled). Promotes only lessons repeated across
/// distinct sessions *and* verified, then reports what it wrote.
pub(super) fn dream(ctx: &mut DispatchCtx) -> CommandOutput {
    use std::fmt::Write as _;
    match runtime::dream_at_cwd(&ctx.cli.cwd) {
        Ok(report) => {
            let mut body = format!("Dream\n  Result           {}", report.summary_line());
            for applied in &report.applied {
                // Infallible write into a String; the result is intentionally ignored.
                let _ = write!(
                    body,
                    "\n  + promoted        {} ({})",
                    applied.slug,
                    applied.outcome.as_str()
                );
            }
            if report.applied.is_empty() {
                body.push_str("\n  (no new lessons met the promotion bar)");
            }
            CommandOutput::info(body)
        }
        Err(error) => CommandOutput::warn(format!("Dream\n  {error}")),
    }
}

/// `/ide [name]` — discover a running IDE extension (Claude Code extension
/// lockfile protocol), connect to its WebSocket MCP server, and surface its
/// tools (`mcp__ide__*`). An optional name filters when several IDEs run.
pub(super) fn ide(ctx: &mut DispatchCtx, target: Option<&str>) -> CommandOutput {
    match super::super::ide_bridge::connect_ide(ctx.cli, target) {
        Ok(report) => CommandOutput::info(report),
        Err(message) => CommandOutput::warn(format!("IDE\n  {message}")),
    }
}

pub(super) fn add_dir(path: Option<&str>) -> CommandOutput {
    let Some(dir) = path else {
        return CommandOutput::warn("Usage: /add-dir <path>");
    };
    let resolved = if std::path::Path::new(dir).is_absolute() {
        std::path::PathBuf::from(dir)
    } else {
        std::env::current_dir().unwrap_or_default().join(dir)
    };
    if !resolved.is_dir() {
        return CommandOutput::error(format!("Directory not found: {}", resolved.display()));
    }
    // Canonicalize so symlinked spellings can't dodge the boundary comparison,
    // then merge into the live workspace roots so reads/writes under the new
    // directory actually pass the boundary check (CC `--add-dir` parity).
    let canonical = std::fs::canonicalize(&resolved).unwrap_or(resolved);
    let mut roots = runtime::file_ops::additional_workspace_roots();
    if roots.contains(&canonical) {
        return CommandOutput::info(format!(
            "Directory already in context: {}",
            canonical.display()
        ));
    }
    roots.push(canonical.clone());
    runtime::file_ops::set_additional_workspace_roots(roots);
    CommandOutput::info(format!("Added directory to context: {}", canonical.display()))
}

pub(super) fn teleport(target: Option<&str>) -> CommandOutput {
    match target {
        Some(t) => match crate::render_teleport_report(t) {
            Ok(r) => CommandOutput::popup("/teleport", r),
            Err(e) => CommandOutput::error(e.to_string()),
        },
        None => CommandOutput::warn("Usage: /teleport <symbol-or-path>"),
    }
}

pub(super) fn debug_tool_call(ctx: &mut DispatchCtx) -> CommandOutput {
    match crate::render_last_tool_debug_report(ctx.cli.runtime.session()) {
        Ok(r) => CommandOutput::popup("/debug-tool-call", r),
        Err(e) => CommandOutput::error(e.to_string()),
    }
}

pub(super) fn review(
    ctx: &mut DispatchCtx,
    scope: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    if let Some(prompt) = crate::build_review_prompt(scope)? {
        if let Err(error) = ctx.app.queue_message(prompt) {
            return Ok(CommandOutput::error(error.to_string()));
        }
        let target = scope.unwrap_or("current changes");
        return Ok(CommandOutput::info(format!(
            "Review\n  Scope            {target}\n  Method           queued code-reviewer subagent turn\n  Status           will run after this command"
        )));
    }

    Ok(CommandOutput::info(crate::render_review_report(scope)?))
}

pub(super) fn hunks(ctx: &mut DispatchCtx) -> CommandOutput {
    let context = ctx
        .cli
        .runtime
        .tool_executor_mut()
        .tool_registry_mut()
        .context()
        .clone();
    let ledger = match context.workspace_hunk_attribution() {
        Ok(ledger) => ledger,
        Err(error) => {
            return CommandOutput::error(format!("Workspace hunk review failed: {error}"));
        }
    };
    ctx.app.open_hunks_modal(zo_cli::tui::modals::ReviewModal::new(
        context, ledger,
    ));
    CommandOutput::Quiet
}

/// `/bughunter [scope]` — queue a real bug-hunting turn (same pattern as
/// `/council`), replacing the old static description of what a hunt would be.
pub(super) fn bughunter(ctx: &mut DispatchCtx, scope: Option<&str>) -> CommandOutput {
    if let Err(error) = ctx.app.queue_message(build_bughunter_prompt(scope)) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(format!(
        "Bughunter\n  Scope            {}\n  Method           queued a bug-hunt turn\n  Status           will run after this command",
        scope.unwrap_or("the current repository")
    ))
}

/// `/ultraplan [task]` — queue a real planning turn.
pub(super) fn ultraplan(ctx: &mut DispatchCtx, task: Option<&str>) -> CommandOutput {
    if let Err(error) = ctx.app.queue_message(build_ultraplan_prompt(task)) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(format!(
        "Ultraplan\n  Task             {}\n  Method           queued a planning turn\n  Status           will run after this command",
        task.unwrap_or("the current repo work")
    ))
}

pub(super) fn council(ctx: &mut DispatchCtx, task: Option<&str>) -> CommandOutput {
    if let Err(error) = ctx.app.queue_message(build_council_prompt(task)) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(format!(
        "Council\n  Task             {}\n  Method           queued SpawnMultiAgent + Council turn\n  Status           will run after this command",
        task.unwrap_or("the current task")
    ))
}

pub(super) fn distill(ctx: &mut DispatchCtx, topic: Option<&str>) -> CommandOutput {
    if let Err(error) = ctx.app.queue_message(build_distill_prompt(topic)) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(format!(
        "Distill\n  Topic            {}\n  Method           queued SkillDistill draft turn\n  Status           will run after this command",
        topic.unwrap_or("the reusable procedure from this session")
    ))
}

pub(super) fn self_improve(ctx: &mut DispatchCtx, action: &SelfImproveAction) -> CommandOutput {
    if matches!(action, SelfImproveAction::Status) {
        return match super::super::self_improve::status_report(
            &ctx.cli.cwd,
            ctx.cli.runtime.feature_config.dream_automation_enabled(),
        ) {
            Ok(report) => CommandOutput::popup("/improve status", report),
            Err(error) => CommandOutput::warn(format!(
                "Self-improve status\n  Status           {}",
                super::super::self_improve::escape_terminal(&error)
            )),
        };
    }

    // Show/review/reject are instant local state operations (no headless
    // turn), so they run inline exactly like status.
    match action {
        SelfImproveAction::Show { proposal_id } => {
            return match super::super::self_improve::show(&ctx.cli.cwd, proposal_id) {
                Ok(report) => CommandOutput::popup("/improve show", report),
                Err(error) => CommandOutput::warn(format!(
                    "Self-improve\n  Status           {}",
                    super::super::self_improve::escape_terminal(&error)
                )),
            };
        }
        SelfImproveAction::Review { proposal_id } => {
            return match super::super::self_improve::review(&ctx.cli.cwd, proposal_id) {
                Ok(report) => CommandOutput::info(report),
                Err(error) => CommandOutput::warn(format!(
                    "Self-improve\n  Status           {}",
                    super::super::self_improve::escape_terminal(&error)
                )),
            };
        }
        SelfImproveAction::Reject { proposal_id } => {
            return match super::super::self_improve::reject(&ctx.cli.cwd, proposal_id) {
                Ok(report) => CommandOutput::info(report),
                Err(error) => CommandOutput::warn(format!(
                    "Self-improve\n  Status           {}",
                    super::super::self_improve::escape_terminal(&error)
                )),
            };
        }
        _ => {}
    }

    // `/improve` generates its patch with a headless `zo -p` turn (minutes),
    // which would freeze the live TUI event loop. It runs in REPL/headless mode
    // instead: start `zo` (no `-p`) and run `/improve` there, then review the
    // proposal and `/improve apply`.
    let command = match action {
        SelfImproveAction::Propose => "/improve",
        SelfImproveAction::Apply { .. } => "/improve apply <patch-digest>",
        SelfImproveAction::Status
        | SelfImproveAction::Show { .. }
        | SelfImproveAction::Review { .. }
        | SelfImproveAction::Reject { .. } => unreachable!("handled above"),
    };
    CommandOutput::info(format!(
        "Self-improve\n  Command          {command}\n  Status           runs in REPL/headless mode\n  Reason           patch generation spawns a headless zo turn (would block the TUI)\n  How              run `zo` then `{command}` (review the proposal, then `/improve apply`)"
    ))
}

/// True only when `base` rejects a write with a permission / read-only error —
/// the case a `/goal` preflight should warn about. A clean writable dir (or any
/// other error, which we must not misreport) → false. Probes the base directly
/// with a uniquely-named marker so a clean tree never gains a stray `.zo/`.
fn state_base_unwritable(base: &std::path::Path) -> bool {
    let probe = base.join(format!(".zo-goal-write-probe-{}", std::process::id()));
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            false
        }
        Err(error) => matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
        ),
    }
}

/// One actionable, non-fatal warning line when zo cannot write its
/// per-project state for a `/goal` run, so the problem surfaces upfront instead
/// of mid-turn as a bare OS `Permission denied`. `None` when writable.
fn goal_state_writability_warning() -> Option<String> {
    let cwd = crate::current_cli_cwd().ok()?;
    // Probe the workspace cwd itself: it is the default home for both the todo
    // store and turn traces. (The todo store auto-falls back to ~/.zo on
    // EACCES; turn traces are best-effort and just stop persisting.)
    state_base_unwritable(&cwd).then(|| {
        format!(
            "  Warning          {} is not writable; the todo store auto-falls back to ~/.zo, \
             but turn traces may not persist. Set ZO_TRACE_ROOT (traces) and/or \
             ZO_STATE_DIR (todos) to a writable dir to relocate zo state.",
            cwd.display()
        )
    })
}

/// `/goal` — session-local bounded goal automation. The slash command layer
/// owns UX/queueing; the controller owns validation state, and the runtime deep
/// gate supplies semantic model verification/repair for the queued turns.
pub(super) fn goal(ctx: &mut DispatchCtx, command: GoalCommand) -> CommandOutput {
    match command {
        GoalCommand::Status => CommandOutput::popup("/goal status", ctx.cli.goal_status_report()),
        GoalCommand::Start { goal, options } => {
            let goal_text = goal.clone();
            let warning = goal_state_writability_warning();
            let (report, prompt) = ctx.cli.start_goal_controller(goal, options);
            // `None` prompt = the ambiguity gate held the goal back (a started
            // goal always has an action prompt): surface only the clarify
            // report — no todo, no queued turn.
            let Some(prompt) = prompt else {
                return CommandOutput::info(report);
            };
            if let Err(error) = ctx.app.queue_goal_message(prompt) {
                return CommandOutput::error(error.to_string());
            }
            let todo_line = super::super::LiveCli::goal_todo_sync_line(&goal_text);
            let mut body =
                format!("{report}\n{todo_line}\n  Status           queued first goal turn");
            if let Some(warning) = warning {
                body.push('\n');
                body.push_str(&warning);
            }
            CommandOutput::info(body)
        }
        GoalCommand::Verify => CommandOutput::info(ctx.cli.verify_goal_controller()),
        GoalCommand::Pause => CommandOutput::info(ctx.cli.pause_goal_controller()),
        GoalCommand::Resume => {
            let (report, prompt) = ctx.cli.resume_goal_controller();
            if let Some(prompt) = prompt {
                if let Err(error) = ctx.app.queue_goal_message(prompt) {
                    return CommandOutput::error(error.to_string());
                }
            }
            CommandOutput::info(format!(
                "{report}\n  Status           queued next goal turn"
            ))
        }
        GoalCommand::Clear => {
            let todo_line = super::super::LiveCli::goal_todo_clear_line();
            CommandOutput::info(format!("{}\n{todo_line}", ctx.cli.clear_goal_controller()))
        }
        GoalCommand::History => {
            CommandOutput::popup("/goal history", ctx.cli.goal_controller.history_report())
        }
        GoalCommand::Edit { goal } => {
            let todo_line = super::super::LiveCli::goal_todo_sync_line(&goal);
            CommandOutput::info(format!(
                "{}\n{todo_line}",
                ctx.cli.edit_goal_controller(goal)
            ))
        }
    }
}

/// `/loop` — session-local recurring prompt scheduler. Fixed-count loops enqueue
/// immediately; interval/watch loops are drained by the TUI idle loop.
pub(super) fn loop_cmd(ctx: &mut DispatchCtx, command: LoopCommand) -> CommandOutput {
    match ctx.cli.handle_loop_controller_command(command) {
        super::super::automation::LoopCommandResult::Report(report) => CommandOutput::info(report),
        super::super::automation::LoopCommandResult::Queue { report, prompts } => {
            for prompt in prompts {
                // Tag with the loop id so each run is gated at pop time and the
                // loop stays stoppable (`/loop stop|pause`) mid-flight.
                if let Err(error) = ctx.app.queue_loop_message(prompt.text, prompt.loop_id) {
                    return CommandOutput::error(error.to_string());
                }
            }
            CommandOutput::info(report)
        }
    }
}

/// `/security-review` — queue a security-focused hunt (it used to print the
/// generic bughunter description dressed up as a security scan).
pub(super) fn security_review(ctx: &mut DispatchCtx) -> CommandOutput {
    let prompt = build_bughunter_prompt(Some(
        "the current repository, focused on security: injection, path traversal, \
         command execution, secrets handling, unsafe deserialization, and permission checks",
    ));
    if let Err(error) = ctx.app.queue_message(prompt) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(
        "Security review\n  Method           queued a security-focused hunt turn\n  Status           will run after this command",
    )
}

pub(super) fn release_notes() -> CommandOutput {
    CommandOutput::popup("/release-notes", handle_release_notes())
}

pub(super) fn extra_usage(ctx: &mut DispatchCtx) -> CommandOutput {
    let usage = ctx.cli.runtime.usage();
    CommandOutput::popup("/extra-usage", handle_extra_usage(usage))
}

pub(super) fn perf_issue(ctx: &mut DispatchCtx) -> CommandOutput {
    let usage = ctx.cli.runtime.usage();
    CommandOutput::popup("/perf-issue", handle_perf_issue(usage))
}

pub(super) fn statusline() -> CommandOutput {
    CommandOutput::popup("/statusline", handle_statusline())
}

pub(super) fn ant_trace() -> CommandOutput {
    CommandOutput::popup("/ant-trace", handle_ant_trace())
}

#[cfg(test)]
mod tests {
    use super::resolve_memory_edit_path;
    use super::write_transcript_dump;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir() -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "zo-memory-path-{}-{nanos}-{counter}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn state_base_unwritable_flags_only_readonly_dirs() {
        // A normal writable dir → not flagged (and leaves no stray files behind).
        let writable = temp_dir();
        assert!(!super::state_base_unwritable(&writable));
        assert!(
            fs::read_dir(&writable).unwrap().next().is_none(),
            "probe must clean up after itself"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let ro = temp_dir();
            fs::set_permissions(&ro, fs::Permissions::from_mode(0o555)).expect("chmod ro");
            // Skip when this uid can write to a 0555 dir anyway (e.g. running as root):
            // the permission error we are detecting cannot occur.
            if fs::write(ro.join(".probe"), b"x").is_ok() {
                let _ = fs::remove_file(ro.join(".probe"));
            } else {
                assert!(
                    super::state_base_unwritable(&ro),
                    "a read-only dir must be flagged for the /goal preflight warning"
                );
            }
            let _ = fs::set_permissions(&ro, fs::Permissions::from_mode(0o755));
            let _ = fs::remove_dir_all(&ro);
        }
        let _ = fs::remove_dir_all(&writable);
    }

    #[test]
    fn transcript_dump_writes_export_text_to_session_scoped_file() {
        let base = temp_dir();
        let mut session = runtime::Session::new();
        session
            .push_message(runtime::ConversationMessage::user_text("hello dump"))
            .expect("push message");

        let (path, chars) =
            write_transcript_dump(&base, "sess-42", &session).expect("dump written");

        // Session-scoped name: re-running `/dump` overwrites the same artifact
        // instead of littering the temp dir.
        assert_eq!(path, base.join("zo-transcript-sess-42.txt"));
        let contents = fs::read_to_string(&path).expect("read dump");
        // Same serializer as /export — header plus the verbatim message.
        assert!(contents.contains("# Conversation Export"));
        assert!(contents.contains("hello dump"));
        // The reported char count must match what was written so the
        // "Characters" result line is trustworthy.
        assert_eq!(chars, contents.len());

        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn memory_edit_always_targets_context_md() {
        let root = temp_dir();
        let path = resolve_memory_edit_path(&root);
        assert_eq!(path, root.join("context.md"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn memory_edit_ignores_existing_legacy_instruction_file() {
        let root = temp_dir();
        let legacy_path = root.join(["CLAUDE", ".md"].concat());
        fs::write(&legacy_path, "legacy rules\n").expect("write legacy instructions");
        let path = resolve_memory_edit_path(&root);
        assert_eq!(path, root.join("context.md"));
        assert_eq!(
            fs::read_to_string(legacy_path).expect("read legacy instructions"),
            "legacy rules\n"
        );
        fs::remove_dir_all(root).expect("cleanup");
    }
}
