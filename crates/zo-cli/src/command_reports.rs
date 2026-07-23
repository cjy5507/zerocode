#![allow(dead_code)]

use std::path::Path;
use std::process::Command;

use runtime::{ContentBlock, DeepGateConfig, DeepMode, Session};

use crate::git_helpers::GitWorkspaceSummary;

/// Shared `/deep` semantics: map the raw argument onto the deep-lane gate
/// config plus its user-facing confirmation. Both the TUI dispatcher
/// (`toggles::deep`) and the headless REPL call this, so the two surfaces can
/// never drift on what `/deep`, `/deep <cmd>`, or `/deep off` mean.
pub(crate) fn deep_gate_directive(arg: Option<&str>) -> (Option<DeepGateConfig>, String) {
    const MAX_ATTEMPTS: u32 = 3;
    // `on`/`off` mirror /plan so the muscle-memory slip `/deep on` enables the
    // gate instead of silently becoming a check command named "on". Any other
    // non-empty argument is the objective green command.
    match arg.map(str::trim) {
        Some("off" | "stop") => (
            None,
            "Deep mode off — turns run the ordinary single pass again.".to_string(),
        ),
        Some(cmd) if !cmd.is_empty() && !matches!(cmd, "on" | "start") => (
            Some(DeepGateConfig {
                mode: DeepMode::PlanFirst,
                check_command: Some(cmd.to_string()),
                max_attempts: MAX_ATTEMPTS,
            }),
            format!(
                "Deep mode on — plan → implement → verify → retry (up to {MAX_ATTEMPTS} attempts). \
                 Objective green = `{cmd}` exits 0. Run /deep off to disable."
            ),
        ),
        _ => (
            // bare `/deep`, `/deep on`, `/deep start`
            Some(DeepGateConfig {
                mode: DeepMode::PlanFirst,
                check_command: None,
                max_attempts: MAX_ATTEMPTS,
            }),
            "Deep mode on (verifier-only — no objective check). Add a green gate with \
             /deep <command> (e.g. /deep cargo test), or /deep off to disable."
                .to_string(),
        ),
    }
}

/// Shared `/auto` semantics — the reactive auto-verify gate. See
/// [`deep_gate_directive`]; both slash surfaces route through this.
pub(crate) fn auto_gate_directive(arg: Option<&str>) -> (Option<DeepGateConfig>, String) {
    const MAX_ATTEMPTS: u32 = 2;
    match arg.map(str::trim) {
        Some("off" | "stop") => (
            None,
            "Auto-verify off — turns run the ordinary single pass.".to_string(),
        ),
        Some(cmd) if !cmd.is_empty() && !matches!(cmd, "on" | "start") => (
            Some(DeepGateConfig {
                mode: DeepMode::Reactive,
                check_command: Some(cmd.to_string()),
                max_attempts: MAX_ATTEMPTS,
            }),
            format!(
                "Auto-verify on — after an edit, green = `{cmd}` exits 0 plus an adversarial \
                 verifier; retries up to {MAX_ATTEMPTS}×. /auto off to disable."
            ),
        ),
        _ => (
            // Bare `/auto on` must stay cheap: do not auto-detect `cargo test`
            // here, because that reintroduces the first-output / post-edit freeze
            // for ordinary chat. Users opt into a heavy objective gate explicitly
            // with `/auto <command>`.
            Some(DeepGateConfig {
                mode: DeepMode::Reactive,
                check_command: None,
                max_attempts: MAX_ATTEMPTS,
            }),
            format!(
                "Auto-verify on — edits are checked by the adversarial verifier; retries up to \
                 {MAX_ATTEMPTS}×. Use /auto <command> to add an objective check, /auto off to disable."
            ),
        ),
    }
}

pub(crate) fn format_commit_preflight_report(
    branch: Option<&str>,
    summary: GitWorkspaceSummary,
) -> String {
    format!(
        "Commit\n  Result           ready\n  Branch           {}\n  Workspace        {}\n  Changed files    {}\n  Action           create a git commit from the current workspace changes",
        branch.unwrap_or("unknown"),
        summary.headline(),
        summary.changed_files,
    )
}

pub(crate) fn format_commit_skipped_report() -> String {
    "Commit\n  Result           skipped\n  Reason           no workspace changes\n  Action           create a git commit from the current workspace changes\n  Next             /status to inspect context · /diff to inspect repo changes"
        .to_string()
}

pub(crate) fn render_teleport_report(target: &str) -> Result<String, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;

    let file_list = Command::new("rg")
        .args(["--files"])
        .current_dir(&cwd)
        .output()?;
    let file_matches = if file_list.status.success() {
        String::from_utf8(file_list.stdout)?
            .lines()
            .filter(|line| line.contains(target))
            .take(10)
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let content_output = Command::new("rg")
        .args(["-n", "-S", "--color", "never", target, "."])
        .current_dir(&cwd)
        .output()?;

    let mut lines = vec![
        "Teleport".to_string(),
        format!("  Target           {target}"),
        "  Action           search workspace files and content for the target".to_string(),
    ];
    if !file_matches.is_empty() {
        lines.push(String::new());
        lines.push("File matches".to_string());
        lines.extend(file_matches.into_iter().map(|path| format!("  {path}")));
    }

    if content_output.status.success() {
        let matches = String::from_utf8(content_output.stdout)?;
        if !matches.trim().is_empty() {
            lines.push(String::new());
            lines.push("Content matches".to_string());
            lines.push(truncate_for_prompt(&matches, 4_000));
        }
    }

    if lines.len() == 1 {
        lines.push("  Result           no matches found".to_string());
    }

    Ok(lines.join("\n"))
}

pub(crate) fn render_last_tool_debug_report(
    session: &Session,
) -> Result<String, Box<dyn std::error::Error>> {
    let last_tool_use = session
        .messages
        .iter()
        .rev()
        .find_map(|message| {
            message.blocks.iter().rev().find_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
        })
        .ok_or_else(|| "no prior tool call found in session".to_string())?;

    let tool_result = session.messages.iter().rev().find_map(|message| {
        message.blocks.iter().rev().find_map(|block| match block {
            ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
                ..
            } if tool_use_id == &last_tool_use.0 => {
                Some((tool_name.clone(), output.clone(), *is_error))
            }
            _ => None,
        })
    });

    let mut lines = vec![
        "Debug tool call".to_string(),
        "  Action           inspect the last recorded tool call and its result".to_string(),
        format!("  Tool id          {}", last_tool_use.0),
        format!("  Tool name        {}", last_tool_use.1),
        "  Input".to_string(),
        indent_block(&last_tool_use.2, 4),
    ];

    match tool_result {
        Some((tool_name, output, is_error)) => {
            lines.push("  Result".to_string());
            lines.push(format!("    name           {tool_name}"));
            lines.push(format!(
                "    status         {}",
                if is_error { "error" } else { "ok" }
            ));
            lines.push(indent_block(&output, 4));
        }
        None => lines.push("  Result           missing tool result".to_string()),
    }

    Ok(lines.join("\n"))
}

fn indent_block(value: &str, spaces: usize) -> String {
    let indent = " ".repeat(spaces);
    value
        .lines()
        .map(|line| format!("{indent}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn validate_no_args(
    command_name: &str,
    args: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(args) = args.map(str::trim).filter(|value| !value.is_empty()) {
        return Err(format!(
            "{command_name} does not accept arguments. Received: {args}\nUsage: {command_name}"
        )
        .into());
    }
    Ok(())
}

/// Prompt queued by `/bughunter [scope]` — a real bug-hunting turn instead of
/// the old static description of what a bug hunt would be.
pub(crate) fn build_bughunter_prompt(scope: Option<&str>) -> String {
    let scope = scope
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("the current repository");
    format!(
        "Hunt for real bugs in: {scope}\n\n\
         Steps:\n\
         1. Read the relevant code first; prioritize recently changed files (`git log --oneline -15`, `git diff HEAD~5 --stat`) and error-prone seams (unsafe casts, locking, index arithmetic, error paths).\n\
         2. Report only defects you can defend from the code you actually read — no style nits, no speculation. For each finding give: file:line, severity, the failure scenario (concrete input/state → wrong behavior), and a suggested fix.\n\
         3. Before reporting a finding, re-read the call sites and try to refute it yourself; drop anything that does not survive.\n\
         4. If nothing survives, say so plainly instead of inventing findings."
    )
}

/// Prompt queued by `/ultraplan [task]` — produces an actual execution plan.
pub(crate) fn build_ultraplan_prompt(task: Option<&str>) -> String {
    let task = task
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("the current repo work");
    format!(
        "Produce a thorough execution plan for: {task}\n\n\
         Ground the plan in this repository (read the relevant code and configs first — do not plan from assumptions). Cover:\n\
         1. Goal and non-goals — one sentence each.\n\
         2. Ordered implementation steps with the concrete files/modules each touches.\n\
         3. Risks and unknowns, each with how you would de-risk it.\n\
         4. Verification: the exact commands/tests that prove each step green.\n\
         5. Rollback: how to back out safely if a step goes wrong.\n\
         Do not start implementing — deliver the plan and stop."
    )
}

pub(crate) fn build_council_prompt(task: Option<&str>) -> String {
    let task = task
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("the current task");
    format!(
        "Run a council comparison for this task:\n\n{task}\n\n\
         Steps:\n\
         1. Use `SpawnMultiAgent` to launch exactly three independent candidate agents for the same task. Keep each candidate focused on producing its best answer, not on judging the others.\n\
         2. When the candidates finish, call `Council` with only each candidate's answer text and completion status. Do not include model names, agent names, or any other source identity in the `Council` candidates.\n\
         3. Follow the `Council` result and its judge budget fields. If it returns `best_of`, present the winning answer and briefly mention the supporting candidate indices. If it returns `tie` with `llm_judge_allowed: false`, report the tie honestly and stop.\n\
         4. Only when `Council` returns `tie` with `llm_judge_allowed: true`, the task genuinely needs adjudication, and `llm_judge_call_limit` is at least 1, you may call exactly one `Agent` with `subagent_type: \"judge\"`; `.zo/agents/judge.md` may override that judge harness.\n\
         5. Never call an LLM judge before `Council`, after `best_of`, or more than `llm_judge_call_limit` times."
    )
}

pub(crate) fn build_distill_prompt(topic: Option<&str>) -> String {
    let topic = topic
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("the reusable procedure from this session");
    format!(
        "Distill a reusable skill draft for:\n\n{topic}\n\n\
         Steps:\n\
         1. Review the recent session context and identify only repeatable procedure knowledge that would help future tasks. If there is no reusable procedure, explain that and do not call a tool.\n\
         2. Choose a lowercase kebab-case slug and a concise name/description.\n\
         3. Call `SkillDistill` exactly once with `slug`, `name`, `description`, and a complete `body` for the proposed `SKILL.md`. The tool will write `state: proposed`.\n\
         4. Do not call `Skill` on the new draft and do not claim it is approved or active; it remains proposed until human review."
    )
}

/// Prompt queued by `/init` so the live agent fleshes out the scaffolded
/// `context.md` instruction file from the *actual* codebase, rather than leaving
/// the static, repo-marker-only template. The scaffold (`.zo/` structure + the
/// instruction file + topic docs) is written synchronously first; this turn
/// replaces the generic template content with grounded guidance.
#[must_use]
pub(crate) fn build_init_prompt() -> String {
    "Analyze THIS repository and rewrite its project instruction file into \
     accurate, high-signal guidance grounded in the real code — never invented. \
     That file is what every future zo agent and sub-agent loads.\n\n\
     The instruction file is `context.md` at the repo root. Edit exactly that \
     file and do not create a competing instruction file.\n\n\
     The scaffold just wrote that file from a generic stack-detection template \
     (including a boilerplate '## Working agreement' section). Treat the \
     auto-generated content as a replaceable skeleton: rewrite it with real, \
     codebase-specific guidance. Only preserve text that is clearly \
     project-specific and was hand-written before this scaffold — do not keep \
     generic boilerplate merely because it is present.\n\n\
     Steps:\n\
     1. Discover the project from real files only: build manifests \
     (Cargo.toml / package.json / pyproject.toml / go.mod / …), the README, the \
     top-level directory layout, and CI config. Use read-only commands as needed \
     (list the tree, `git ls-files | head`, `cargo metadata --no-deps` for Rust). \
     Every claim you write must trace to something you read.\n\
     2. Rewrite the instruction file to be concise and high-signal:\n\
        - the build / lint / test / verify commands that actually exist here \
     (confirm them in the manifests or scripts — do not guess);\n\
        - a high-level architecture map: the main crates/packages/modules, what \
     each is responsible for, and the key directories;\n\
        - project-specific conventions and gotchas a new contributor must know.\n\
        Keep it short and link out to `.zo/docs/architecture.md` and \
     `.zo/docs/testing.md` for detail; flesh those out too when it helps.\n\
     3. Do not include secrets, machine-local absolute paths, or facts the repo \
     and git history already make obvious. Prefer accuracy over completeness.\n\n\
     When done, briefly summarize what you captured and which file you wrote."
        .to_string()
}

/// Prompt queued by `/pr [context]` — drafts and opens a real pull request
/// for the current branch via `gh`.
pub(crate) fn build_pr_prompt(branch: &str, context: Option<&str>) -> String {
    let context = context
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("none provided");
    format!(
        "Create a pull request for the current branch `{branch}` (user context: {context}).\n\n\
         Steps:\n\
         1. Inspect what the PR would contain: `git status`, `git log <base>..HEAD --oneline`, and the full diff vs the base branch (detect the default base with `gh repo view --json defaultBranchRef` or `git remote show origin`).\n\
         2. If the branch has no commits beyond the base, or you are on the default branch itself, stop and report that instead of creating anything.\n\
         3. Draft a concise title and a markdown body summarizing ALL commits in the range (not just the latest), including a test-plan section.\n\
         4. Push the branch if needed (`git push -u origin {branch}`) and create the PR with `gh pr create` using a heredoc body. Return the PR URL when done."
    )
}

/// Prompt queued by `/issue [context]` — drafts and files a real GitHub issue
/// via `gh`.
pub(crate) fn build_issue_prompt(context: Option<&str>) -> String {
    let context = context
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("the problem discussed most recently in this session");
    format!(
        "File a GitHub issue for: {context}\n\n\
         Steps:\n\
         1. Gather the concrete evidence from this session/repository (error output, reproduction steps, affected files) — the issue must stand on facts, not paraphrase.\n\
         2. Draft a specific title and a markdown body with: summary, reproduction steps, expected vs actual behavior, and environment details.\n\
         3. Create it with `gh issue create` using a heredoc body and return the issue URL. If `gh` is not authenticated or there is no GitHub remote, report that and print the drafted title/body instead."
    )
}

pub(crate) fn git_output(args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    git_output_in(&crate::current_cli_cwd()?, args)
}

/// Run `git <args>` in `cwd`, returning stdout on success or an error carrying
/// git's trimmed stderr. Shared core for the CLI's git invocations: [`git_output`]
/// targets the resolved CLI cwd, while workspace reports pass an explicit root.
pub(crate) fn git_output_in(
    cwd: &Path,
    args: &[&str],
) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git").args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!("git {} failed: {stderr}", args.join(" ")).into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn truncate_for_prompt(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.trim().to_string()
    } else {
        let truncated = value.chars().take(limit).collect::<String>();
        format!("{}\n…[truncated]", truncated.trim_end())
    }
}

#[cfg(test)]
mod deep_auto_directive_tests {
    use super::{auto_gate_directive, deep_gate_directive};
    use runtime::DeepMode;

    #[test]
    fn deep_off_clears_the_gate() {
        let (config, message) = deep_gate_directive(Some("off"));
        assert!(config.is_none());
        assert!(message.contains("Deep mode off"));
    }

    #[test]
    fn deep_bare_and_on_are_planfirst_verifier_only() {
        for arg in [None, Some("on"), Some("start")] {
            let (config, message) = deep_gate_directive(arg);
            let config = config.expect("bare/on installs the gate");
            assert!(matches!(config.mode, DeepMode::PlanFirst));
            assert!(config.check_command.is_none(), "no objective check for {arg:?}");
            assert!(message.contains("verifier-only"));
        }
    }

    #[test]
    fn deep_command_argument_becomes_the_objective_check() {
        let (config, message) = deep_gate_directive(Some("cargo test"));
        let config = config.expect("gate installed");
        assert!(matches!(config.mode, DeepMode::PlanFirst));
        assert_eq!(config.check_command.as_deref(), Some("cargo test"));
        assert!(message.contains("cargo test"));
    }

    #[test]
    fn auto_uses_reactive_mode_and_off_clears() {
        let (on, _) = auto_gate_directive(None);
        assert!(matches!(on.expect("gate installed").mode, DeepMode::Reactive));
        let (off, message) = auto_gate_directive(Some("stop"));
        assert!(off.is_none());
        assert!(message.contains("Auto-verify off"));
    }
}
