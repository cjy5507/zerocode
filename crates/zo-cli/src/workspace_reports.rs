use std::path::Path;
use std::process::Command;

use commands::render_slash_command_help;
use runtime::{resolve_sandbox_status, ConfigLoader, ProjectContext};

use crate::git_helpers::{parse_git_status_metadata, parse_git_workspace_summary};
use crate::{default_prompt_date, StatusContext, StatusUsage};

pub(crate) fn render_repl_help() -> String {
    [
        "REPL".to_string(),
        "  /exit                Quit the REPL".to_string(),
        "  /quit                Quit the REPL".to_string(),
        "  Up/Down              Navigate prompt history".to_string(),
        "  Tab                  Complete commands, modes, and recent sessions".to_string(),
        "  Ctrl-C               Clear input (or exit on empty prompt)".to_string(),
        "  Shift+Enter/Ctrl+J   Insert a newline".to_string(),
        "  Auto-save            ~/.zo/projects/<project>/sessions/<session-id>.jsonl"
            .to_string(),
        "  Resume latest        /resume latest".to_string(),
        "  Browse sessions      /session list".to_string(),
        String::new(),
        render_slash_command_help(),
    ]
    .join("\n")
}

pub(crate) fn status_context(
    session_path: Option<&Path>,
) -> Result<StatusContext, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    let loader = ConfigLoader::default_for(&cwd);
    let discovered_config_files = loader.discover().len();
    let runtime_config = loader.load()?;
    let project_context = ProjectContext::discover_with_git(&cwd, default_prompt_date())?;
    let (project_root, git_branch) =
        parse_git_status_metadata(project_context.git_status.as_deref());
    let git_summary = parse_git_workspace_summary(project_context.git_status.as_deref());
    let sandbox_status = resolve_sandbox_status(runtime_config.sandbox(), &cwd);
    Ok(StatusContext {
        cwd,
        session_path: session_path.map(Path::to_path_buf),
        loaded_config_files: runtime_config.loaded_entries().len(),
        discovered_config_files,
        instruction_file_count: project_context.instruction_files.len(),
        project_root,
        git_branch,
        git_summary,
        sandbox_status,
    })
}

pub(crate) fn format_status_report(
    model: &str,
    usage: StatusUsage,
    permission_mode: &str,
    context: &StatusContext,
) -> String {
    [
        format!(
            "Status\n  Model            {model}\n  Permission mode  {permission_mode}\n  Messages         {}\n  Turns            {}\n  Estimated tokens {}",
            usage.message_count, usage.turns, usage.estimated_tokens,
        ),
        format!(
            "Usage\n  Latest total     {}\n  Cumulative input {}\n  Cumulative output {}\n  Cumulative total {}",
            usage.latest.total_tokens(),
            usage.cumulative.input_tokens,
            usage.cumulative.output_tokens,
            usage.cumulative.total_tokens(),
        ),
        format!(
            "Workspace\n  Cwd              {}\n  Project root     {}\n  Git branch       {}\n  Git state        {}\n  Changed files    {}\n  Staged           {}\n  Unstaged         {}\n  Untracked        {}\n  Session          {}\n  Config files     loaded {}/{}\n  Instruction files {}\n  Suggested flow   /status → /diff → /commit",
            context.cwd.display(),
            context
                .project_root
                .as_ref()
                .map_or_else(|| "unknown".to_string(), |path| path.display().to_string()),
            context.git_branch.as_deref().unwrap_or("unknown"),
            context.git_summary.headline(),
            context.git_summary.changed_files,
            context.git_summary.staged_files,
            context.git_summary.unstaged_files,
            context.git_summary.untracked_files,
            context.session_path.as_ref().map_or_else(
                || "live-repl".to_string(),
                |path| path.display().to_string()
            ),
            context.loaded_config_files,
            context.discovered_config_files,
            context.instruction_file_count,
        ),
        format_sandbox_report(&context.sandbox_status),
    ]
    .join("\n\n")
}

pub(crate) fn format_sandbox_report(status: &runtime::SandboxStatus) -> String {
    format!(
        "Sandbox\n  Enabled           {}\n  Effective posture {}\n  Active            {}\n  Supported         {}\n  Platform backend  {}\n  In container      {}\n  Requested ns      {}\n  Active ns         {}\n  Requested net     {}\n  Active net        {}\n  Filesystem mode   {}\n  Filesystem active {}\n  Allowed mounts    {}\n  Markers           {}\n  Fallback reason   {}\n  Isolation note    {}",
        status.enabled,
        sandbox_effective_posture(status),
        status.active,
        status.supported,
        sandbox_platform_backend(),
        status.in_container,
        status.requested.namespace_restrictions,
        status.namespace_active,
        status.requested.network_isolation,
        status.network_active,
        status.filesystem_mode.as_str(),
        status.filesystem_active,
        if status.allowed_mounts.is_empty() {
            "<none>".to_string()
        } else {
            status.allowed_mounts.join(", ")
        },
        if status.container_markers.is_empty() {
            "<none>".to_string()
        } else {
            status.container_markers.join(", ")
        },
        status
            .fallback_reason
            .clone()
            .unwrap_or_else(|| "<none>".to_string()),
        sandbox_isolation_note(status),
    )
}

fn sandbox_effective_posture(status: &runtime::SandboxStatus) -> &'static str {
    if !status.enabled {
        "off"
    } else if status.namespace_active || status.network_active {
        "isolated"
    } else if status.filesystem_active {
        "filesystem-only"
    } else {
        "requested-but-inactive"
    }
}

fn sandbox_platform_backend() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux/unshare"
    } else if cfg!(target_os = "macos") {
        "macos/seatbelt opt-in"
    } else if cfg!(target_os = "windows") {
        "windows/unavailable"
    } else {
        "unsupported"
    }
}

fn sandbox_isolation_note(status: &runtime::SandboxStatus) -> &'static str {
    if !status.enabled {
        "sandbox not requested"
    } else if status.namespace_active || status.network_active {
        "namespace/network isolation active as requested"
    } else if status.filesystem_active {
        "only filesystem isolation is active; namespace/network isolation is not active"
    } else if status.fallback_reason.is_some() {
        "sandbox requested but unavailable; see fallback reason"
    } else {
        "sandbox requested but no isolation backend is active on this platform/configuration"
    }
}

pub(crate) fn render_config_report(
    section: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    let loader = ConfigLoader::default_for(&cwd);
    let discovered = loader.discover();
    let runtime_config = loader.load()?;

    let mut lines = vec![
        format!(
            "Config\n  Working directory {}\n  Loaded files      {}\n  Merged keys       {}",
            cwd.display(),
            runtime_config.loaded_entries().len(),
            runtime_config.merged().len()
        ),
        "Discovered files".to_string(),
    ];
    for entry in discovered {
        let source = match entry.source {
            runtime::ConfigSource::User => "user",
            runtime::ConfigSource::Project => "project",
            runtime::ConfigSource::Local => "local",
        };
        let status = if runtime_config
            .loaded_entries()
            .iter()
            .any(|loaded_entry| loaded_entry.path == entry.path)
        {
            "loaded"
        } else {
            "missing"
        };
        lines.push(format!(
            "  {source:<7} {status:<7} {}",
            entry.path.display()
        ));
    }

    if let Some(section) = section {
        lines.push(format!("Merged section: {section}"));
        let value = match section {
            "env" => runtime_config.get("env"),
            "hooks" => runtime_config.get("hooks"),
            "model" => runtime_config.get("model"),
            "plugins" => runtime_config
                .get("plugins")
                .or_else(|| runtime_config.get("enabledPlugins")),
            other => {
                lines.push(format!(
                    "  Unsupported config section '{other}'. Use env, hooks, model, or plugins."
                ));
                return Ok(lines.join("\n"));
            }
        };
        lines.push(format!(
            "  {}",
            match value {
                Some(value) => value.render(),
                None => "<unset>".to_string(),
            }
        ));
        return Ok(lines.join("\n"));
    }

    lines.push("Merged JSON".to_string());
    lines.push(format!("  {}", runtime_config.as_json().render()));
    Ok(lines.join("\n"))
}

pub(crate) fn render_memory_report() -> Result<String, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    let project_context = ProjectContext::discover(&cwd, default_prompt_date())?;
    let mut lines = vec![format!(
        "Memory\n  Working directory {}\n  Instruction files {}",
        cwd.display(),
        project_context.instruction_files.len()
    )];
    if project_context.instruction_files.is_empty() {
        lines.push("Discovered files".to_string());
        lines.push(
            "  No context.md instruction files discovered in the current directory ancestry."
                .to_string(),
        );
    } else {
        lines.push("Discovered files".to_string());
        for (index, file) in project_context.instruction_files.iter().enumerate() {
            let preview = file.content.lines().next().unwrap_or("").trim();
            let preview = if preview.is_empty() {
                "<empty>"
            } else {
                preview
            };
            lines.push(format!("  {}. {}", index + 1, file.path.display()));
            lines.push(format!(
                "     lines={} preview={}",
                file.content.lines().count(),
                preview
            ));
        }
    }
    Ok(lines.join("\n"))
}

pub(crate) fn render_diff_report() -> Result<String, Box<dyn std::error::Error>> {
    render_diff_report_for(&crate::current_cli_cwd()?)
}

pub(crate) fn render_hooks_report() -> Result<String, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    let loader = ConfigLoader::default_for(&cwd);
    let runtime_config = loader.load()?;
    let hooks = runtime_config.hooks();

    let mut lines = vec![
        "Hooks".to_string(),
        format!("  Working directory {}", cwd.display()),
        format!(
            "  Config files      {}",
            runtime_config.loaded_entries().len()
        ),
        format!(
            "  Counts            pre={} post={} failure={} subagent_start={} subagent_stop={}",
            hooks.pre_tool_use().len(),
            hooks.post_tool_use().len(),
            hooks.post_tool_use_failure().len(),
            hooks.subagent_start().len(),
            hooks.subagent_stop().len()
        ),
    ];

    if hooks.pre_tool_use().is_empty()
        && hooks.post_tool_use().is_empty()
        && hooks.post_tool_use_failure().is_empty()
        && hooks.subagent_start().is_empty()
        && hooks.subagent_stop().is_empty()
    {
        lines.push("  Result            no lifecycle hooks configured".to_string());
        return Ok(lines.join("\n"));
    }

    lines.push(String::new());
    lines.push("Configured hooks".to_string());
    for rule in hooks.pre_tool_use() {
        lines.push(format!("  PreToolUse         {rule}"));
    }
    for rule in hooks.post_tool_use() {
        lines.push(format!("  PostToolUse        {rule}"));
    }
    for rule in hooks.post_tool_use_failure() {
        lines.push(format!("  PostToolUseFailure {rule}"));
    }
    for rule in hooks.subagent_start() {
        lines.push(format!("  SubagentStart      {rule}"));
    }
    for rule in hooks.subagent_stop() {
        lines.push(format!("  SubagentStop       {rule}"));
    }

    Ok(lines.join("\n"))
}

pub(crate) fn render_review_report(
    scope: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    let target = scope.unwrap_or("current changes");
    let staged_stat = run_git_review_command_in(&cwd, &["diff", "--cached", "--stat"], scope)?;
    let unstaged_stat = run_git_review_command_in(&cwd, &["diff", "--stat"], scope)?;
    let diff_check = run_git_review_command_in(&cwd, &["diff", "--check"], scope)?;

    if staged_stat.trim().is_empty() && unstaged_stat.trim().is_empty() {
        return Ok(format!(
            "Review\n  Scope            {target}\n  Result           no staged or unstaged changes to review"
        ));
    }

    let mut lines = vec![
        "Review".to_string(),
        format!("  Scope            {target}"),
        "  Method           git diff preflight (stat + diff --check)".to_string(),
        format!(
            "  Diff check       {}",
            if diff_check.trim().is_empty() {
                "clean"
            } else {
                "issues found"
            }
        ),
    ];

    if !staged_stat.trim().is_empty() {
        lines.push(String::new());
        lines.push("Staged diff stat".to_string());
        lines.push(staged_stat.trim_end().to_string());
    }
    if !unstaged_stat.trim().is_empty() {
        lines.push(String::new());
        lines.push("Unstaged diff stat".to_string());
        lines.push(unstaged_stat.trim_end().to_string());
    }
    if !diff_check.trim().is_empty() {
        lines.push(String::new());
        lines.push("Diff check findings".to_string());
        lines.push(diff_check.trim_end().to_string());
    }

    Ok(lines.join("\n"))
}

pub(crate) fn build_review_prompt(
    scope: Option<&str>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let cwd = crate::current_cli_cwd()?;
    build_review_prompt_for(&cwd, scope)
}

fn build_review_prompt_for(
    cwd: &Path,
    scope: Option<&str>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let diff = run_git_review_diff_command_in(cwd, scope)?;
    let diff = diff.trim_end();
    if diff.is_empty() {
        return Ok(None);
    }

    let target = scope.unwrap_or("current changes");
    Ok(Some(format!(
        "Use the Agent tool to launch a `code-reviewer` subagent for this review.\n\
         \n\
         Agent tool arguments:\n\
         - subagent_type: code-reviewer\n\
         - description: Review {target}\n\
         - prompt: Review the git diff below for correctness, regressions, security, performance, and missing tests. Do not edit files. Return actionable findings first with file and line references when possible; if there are no findings, say so clearly.\n\
         \n\
         Review scope: {target}\n\
         Working directory: {}\n\
         \n\
         --- BEGIN GIT DIFF ---\n\
         {diff}\n\
         --- END GIT DIFF ---",
        cwd.display()
    )))
}

pub(crate) fn render_diff_report_for(cwd: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let staged = run_git_diff_command_in(cwd, &["diff", "--no-color", "--cached"])?;
    let staged_stat = run_git_diff_command_in(cwd, &["diff", "--no-color", "--cached", "--stat"])?;
    let unstaged = run_git_diff_command_in(cwd, &["diff", "--no-color"])?;
    let unstaged_stat = run_git_diff_command_in(cwd, &["diff", "--no-color", "--stat"])?;
    if staged.trim().is_empty() && unstaged.trim().is_empty() {
        return Ok(
            "# Diff\n\n### Result\nclean working tree\n\n### Detail\nno current changes"
                .to_string(),
        );
    }

    let mut sections = Vec::new();
    if !staged.trim().is_empty() {
        sections.push(format_diff_section(
            "Staged changes:",
            &staged,
            &staged_stat,
        ));
    }
    if !unstaged.trim().is_empty() {
        sections.push(format_diff_section(
            "Unstaged changes:",
            &unstaged,
            &unstaged_stat,
        ));
    }

    Ok(format!("# Diff\n\n{}", sections.join("\n\n")))
}

fn format_diff_section(title: &str, diff: &str, stat: &str) -> String {
    let mut section = format!("### {title}");
    if let Some(summary) = diff_stat_summary(stat) {
        section.push_str("\n\n");
        section.push_str(summary);
    }
    section.push_str("\n\n```diff\n");
    section.push_str(diff.trim_end());
    section.push_str("\n```");
    section
}

fn diff_stat_summary(stat: &str) -> Option<&str> {
    stat.lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
}

fn run_git_diff_command_in(
    cwd: &Path,
    args: &[&str],
) -> Result<String, Box<dyn std::error::Error>> {
    crate::command_reports::git_output_in(cwd, args)
}

fn run_git_review_command_in(
    cwd: &Path,
    args: &[&str],
    scope: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut command = Command::new("git");
    command.args(args).current_dir(cwd);
    if let Some(scope) = scope {
        command.args(["--", scope]);
    }
    let output = command.output()?;
    if output.status.success() {
        return Ok(String::from_utf8(output.stdout)?);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if stderr.contains("not a git repository") {
        return Ok(String::new());
    }

    Ok(format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr
    ))
}

fn run_git_review_diff_command_in(
    cwd: &Path,
    scope: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut command = Command::new("git");
    command
        .args(["diff", "--no-color", "HEAD"])
        .current_dir(cwd);
    if let Some(scope) = scope {
        command.args(["--", scope]);
    }
    let output = command.output()?;
    if output.status.success() {
        return Ok(String::from_utf8(output.stdout)?);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if stderr.contains("not a git repository") {
        return Ok(String::new());
    }

    Err(format!("git diff --no-color HEAD failed: {stderr}").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestRepo {
        path: PathBuf,
    }

    impl TestRepo {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time moves forward")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "zo-review-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temp repo");
            run_git(&path, &["init"]);
            run_git(&path, &["config", "user.email", "zo@example.invalid"]);
            run_git(&path, &["config", "user.name", "Zo Test"]);
            fs::write(path.join("tracked.txt"), "before\n").expect("write seed file");
            run_git(&path, &["add", "tracked.txt"]);
            run_git(&path, &["commit", "-m", "seed"]);
            Self { path }
        }
    }

    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn run_git(cwd: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn review_prompt_uses_code_reviewer_agent_and_full_head_diff() {
        let repo = TestRepo::new("dirty");
        fs::write(repo.path.join("tracked.txt"), "before\nafter\n").expect("modify tracked file");

        let prompt = build_review_prompt_for(&repo.path, None)
            .expect("build review prompt")
            .expect("dirty repo should produce prompt");

        assert!(prompt.contains("Agent tool"));
        assert!(prompt.contains("subagent_type: code-reviewer"));
        assert!(prompt.contains("--- BEGIN GIT DIFF ---"));
        assert!(prompt.contains("+after"));
    }

    #[test]
    fn review_prompt_returns_none_for_clean_tree() {
        let repo = TestRepo::new("clean");

        let prompt = build_review_prompt_for(&repo.path, None).expect("build review prompt");

        assert!(prompt.is_none());
    }
}
