//! Git / VCS commands: commit preflight, PR, issue, pr-comments,
//! commit-push-pr, branch, backfill-sessions.
//!
//! `/commit`, `/pr-comments`, and the rest build textual reports from `git`
//! output and the crate-level formatters. `/pr` and `/issue` queue real
//! draft-and-create turns (`build_pr_prompt` / `build_issue_prompt`), and
//! `/branch <name>` actually switches branches.

use crate::git_helpers::{
    parse_git_status_branch, parse_git_workspace_summary, resolve_git_branch_for,
};
use crate::git_output;
use crate::{
    build_issue_prompt, build_pr_prompt, format_commit_preflight_report,
    format_commit_skipped_report,
};

use super::context::DispatchCtx;
use super::handlers::{handle_backfill_sessions, handle_commit_push_pr_at, handle_pr_comments};
use super::output::CommandOutput;

/// `/commit` preflight: summarize the working tree without committing.
pub(super) fn commit() -> CommandOutput {
    match git_output(&["status", "--short", "--branch"]) {
        Ok(status) => {
            let summary = parse_git_workspace_summary(Some(&status));
            let branch = parse_git_status_branch(Some(&status));
            let report = if summary.is_clean() {
                format_commit_skipped_report()
            } else {
                format_commit_preflight_report(branch.as_deref(), summary)
            };
            CommandOutput::popup("/commit", report)
        }
        Err(e) => CommandOutput::error(e.to_string()),
    }
}

/// `/pr [context]` — queue a turn that inspects the branch and creates the
/// pull request with `gh` (mirrors the REPL path's `run_turn`).
pub(super) fn pr(ctx: &mut DispatchCtx, context: Option<&str>) -> CommandOutput {
    let branch = resolve_git_branch_for(&ctx.cli.cwd).unwrap_or_else(|| "unknown".to_string());
    if let Err(error) = ctx.app.queue_message(build_pr_prompt(&branch, context)) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(format!(
        "PR\n  Branch           {branch}\n  Method           queued a draft-and-create turn (gh)\n  Status           will run after this command"
    ))
}

/// `/issue [context]` — queue a turn that drafts and files the GitHub issue
/// with `gh`.
pub(super) fn issue(ctx: &mut DispatchCtx, context: Option<&str>) -> CommandOutput {
    if let Err(error) = ctx.app.queue_message(build_issue_prompt(context)) {
        return CommandOutput::error(error.to_string());
    }
    CommandOutput::info(format!(
        "Issue\n  Context          {}\n  Method           queued a draft-and-create turn (gh)\n  Status           will run after this command",
        context.unwrap_or("from recent session context")
    ))
}

pub(super) fn pr_comments(pr_number: Option<&str>) -> CommandOutput {
    CommandOutput::popup("/pr-comments", handle_pr_comments(pr_number))
}

pub(super) fn commit_push_pr(cwd: &std::path::Path) -> CommandOutput {
    CommandOutput::info(handle_commit_push_pr_at(cwd))
}

pub(super) fn ship(cwd: &std::path::Path, message: &str) -> CommandOutput {
    let result = super::ship::handle_ship_at(cwd, message, |_| {});
    if result.success {
        CommandOutput::info(result.report)
    } else {
        CommandOutput::error(result.report)
    }
}

pub(super) fn backfill_sessions() -> CommandOutput {
    CommandOutput::info(handle_backfill_sessions())
}

/// `/branch` — show the current branch; `/branch <name>` genuinely switches:
/// create-and-checkout, falling back to a plain checkout when the branch
/// already exists (the same semantics as the REPL arm).
pub(super) fn branch(name: Option<&str>) -> CommandOutput {
    let current = resolve_git_branch_for(&std::env::current_dir().unwrap_or_default())
        .unwrap_or_else(|| "unknown".to_string());
    let Some(requested) = name.map(str::trim).filter(|value| !value.is_empty()) else {
        return CommandOutput::info(format!("Branch\n  Current          {current}"));
    };
    if requested.starts_with('-') || requested.contains(' ') || requested.contains("..") {
        return CommandOutput::error(format!("Branch\n  Invalid name     '{requested}'"));
    }
    match git_output(&["checkout", "-b", requested]) {
        Ok(_) => CommandOutput::info(format!(
            "Branch\n  Switched to      {requested} (new)\n  Previous         {current}"
        )),
        Err(create_error) => match git_output(&["checkout", requested]) {
            Ok(_) => CommandOutput::info(format!(
                "Branch\n  Switched to      {requested}\n  Previous         {current}"
            )),
            Err(_) => CommandOutput::error(format!("Branch switch failed: {create_error}")),
        },
    }
}
