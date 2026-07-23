//! Handlers for "promoted" slash commands.
//!
//! Each `handle_*` entry point is the per-command worker the main
//! dispatch (in `super::handle_persistent_slash`) calls when the user
//! types `/release-notes`, `/pr-comments`, etc. They run their side
//! effect (shell out to `gh`, read `CHANGELOG.md`, scrape session
//! files) and return the human-readable report that the caller pushes
//! into the transcript via `push_report`.
//!
//! Internal helpers (`parse_*`, `format_*`, `generate_*`) stay private
//! to this module — they exist to keep the handlers above readable and
//! are not consumed elsewhere.

use std::io;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use crate::git_output;

const COMMIT_PUSH_PR_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug)]
enum HardenedCommandError {
    Io(io::Error),
    TimedOut {
        program: String,
        args: Vec<String>,
        timeout: Duration,
    },
}

impl std::fmt::Display for HardenedCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::TimedOut {
                program,
                args,
                timeout,
            } => write!(
                f,
                "{} {} timed out after {}s",
                program,
                args.join(" "),
                timeout.as_secs()
            ),
        }
    }
}

fn run_hardened_command_in(
    program: &str,
    args: &[&str],
    timeout: Duration,
    cwd: Option<&Path>,
) -> Result<Output, HardenedCommandError> {
    let mut command = Command::new(program);
    command
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GCM_INTERACTIVE", "never")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }

    let mut child = command.spawn().map_err(HardenedCommandError::Io)?;

    let started = Instant::now();
    loop {
        match child.try_wait().map_err(HardenedCommandError::Io)? {
            Some(_) => return child.wait_with_output().map_err(HardenedCommandError::Io),
            None if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(HardenedCommandError::TimedOut {
                    program: program.to_string(),
                    args: args.iter().map(|arg| (*arg).to_string()).collect(),
                    timeout,
                });
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

pub(super) fn handle_release_notes() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut dir = cwd.as_path();
    loop {
        let candidate = dir.join("CHANGELOG.md");
        if candidate.is_file() {
            match std::fs::read_to_string(&candidate) {
                Ok(contents) => return parse_latest_changelog_section(&contents),
                Err(e) => {
                    return format!(
                        "Release Notes\n  Error            failed to read CHANGELOG.md: {e}"
                    );
                }
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }
    format!(
        "Release Notes\n  Version          {}\n  Fallback         No CHANGELOG.md found. See /version for build info.",
        env!("CARGO_PKG_VERSION")
    )
}

fn parse_latest_changelog_section(contents: &str) -> String {
    let mut start = None;
    let mut version_line = String::new();
    for (i, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") && start.is_none() {
            let rest = trimmed.strip_prefix("## ").unwrap_or("");
            if rest.starts_with('v')
                || rest.starts_with('[')
                || rest.chars().next().is_some_and(|c| c.is_ascii_digit())
            {
                start = Some(i);
                version_line = trimmed.to_string();
                continue;
            }
        }
        if let Some(start_idx) = start {
            if trimmed.starts_with("## ") {
                let section: String = contents
                    .lines()
                    .skip(start_idx + 1)
                    .take(i - start_idx - 1)
                    .collect::<Vec<_>>()
                    .join("\n");
                let body = section.trim();
                if body.is_empty() {
                    return format!("Release Notes\n  {version_line}\n  (no content in section)");
                }
                return format!("Release Notes\n  {version_line}\n\n{body}");
            }
        }
    }
    if let Some(idx) = start {
        let section: String = contents
            .lines()
            .skip(idx + 1)
            .collect::<Vec<_>>()
            .join("\n");
        let body = section.trim();
        if body.is_empty() {
            return format!("Release Notes\n  {version_line}\n  (no content in section)");
        }
        return format!("Release Notes\n  {version_line}\n\n{body}");
    }
    format!(
        "Release Notes\n  Version          {}\n  Fallback         CHANGELOG.md found but contains no version headers.",
        env!("CARGO_PKG_VERSION")
    )
}

pub(super) fn handle_pr_comments(pr_number: Option<&str>) -> String {
    if std::process::Command::new("gh")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_err()
    {
        return "PR Comments\n  Error            Install GitHub CLI (gh) to use this command."
            .to_string();
    }
    let remote_url = match git_output(&["remote", "get-url", "origin"]) {
        Ok(url) => url.trim().to_string(),
        Err(e) => return format!("PR Comments\n  Error            {e}"),
    };
    let owner_repo = parse_owner_repo(&remote_url);
    let Some(owner_repo) = owner_repo else {
        return format!(
            "PR Comments\n  Error            could not parse owner/repo from remote: {remote_url}"
        );
    };
    let number = match pr_number {
        Some(n) => n.trim_start_matches('#').to_string(),
        None => match std::process::Command::new("gh")
            .args(["pr", "view", "--json", "number", "-q", ".number"])
            .output()
        {
            Ok(output) if output.status.success() => {
                String::from_utf8_lossy(&output.stdout).trim().to_string()
            }
            _ => {
                return "PR Comments\n  Error            no PR# given and no PR found for current branch.\n  Usage            /pr-comments [PR#]".to_string();
            }
        },
    };
    match std::process::Command::new("gh")
        .args([
            "api",
            &format!("repos/{owner_repo}/pulls/{number}/comments"),
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let body = String::from_utf8_lossy(&output.stdout);
            format_pr_comments_response(&number, &body)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            format!(
                "PR Comments\n  Error            gh api failed: {}",
                stderr.trim()
            )
        }
        Err(e) => format!("PR Comments\n  Error            {e}"),
    }
}

fn parse_owner_repo(remote_url: &str) -> Option<String> {
    if let Some(rest) = remote_url.strip_prefix("git@github.com:") {
        return Some(rest.trim_end_matches(".git").to_string());
    }
    if let Some(rest) = remote_url
        .strip_prefix("https://github.com/")
        .or_else(|| remote_url.strip_prefix("http://github.com/"))
    {
        return Some(rest.trim_end_matches(".git").to_string());
    }
    None
}

fn format_pr_comments_response(pr_number: &str, json_body: &str) -> String {
    let comments: Vec<serde_json::Value> = match serde_json::from_str(json_body) {
        Ok(c) => c,
        Err(_) => return format!("PR Comments (#{pr_number})\n  (could not parse response)"),
    };
    if comments.is_empty() {
        return format!("PR Comments (#{pr_number})\n  No comments found.");
    }
    let mut lines = vec![format!(
        "PR Comments (#{pr_number})  {} comment(s)\n",
        comments.len()
    )];
    for comment in &comments {
        let author = comment
            .get("user")
            .and_then(|u| u.get("login"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let date = comment
            .get("created_at")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let date_short = if date.len() >= 10 { &date[..10] } else { date };
        let body = comment
            .get("body")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let body_truncated = if body.chars().count() > 500 {
            let truncated: String = body.chars().take(500).collect();
            format!("{truncated}...")
        } else {
            body.to_string()
        };
        lines.push(format!("  @{author}  {date_short}"));
        for line in body_truncated.lines() {
            lines.push(format!("    {line}"));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Run the commit → push → PR sequence in the given session's workspace cwd and
/// return a human-readable report (failures are encoded into the report string,
/// never an `Err`). Triggered by the local `/commit-push-pr` slash command.
pub(crate) fn handle_commit_push_pr_at(cwd: &Path) -> String {
    let mut report = Vec::new();
    report.push("Commit-Push-PR\n".to_string());
    let status = match git_output_at(&["status", "--short"], cwd) {
        Ok(s) => s,
        Err(e) => return format!("Commit-Push-PR\n  Step 1 failed    git status: {e}"),
    };
    let diff_stat = git_output_at(&["diff", "--stat"], cwd).unwrap_or_default();
    if status.trim().is_empty() {
        return "Commit-Push-PR\n  Result           nothing to commit, working tree clean"
            .to_string();
    }
    report.push(format!("  Status:\n{}", indent_lines(&status, "    ")));
    if !diff_stat.trim().is_empty() {
        report.push(format!(
            "  Diff stat:\n{}",
            indent_lines(&diff_stat, "    ")
        ));
    }
    if let Err(error) = stage_all_changes(&mut report, cwd) {
        return error;
    }
    let commit_msg = generate_commit_message(cwd);
    if let Err(error) = commit_staged_changes(&mut report, &commit_msg, cwd) {
        return error;
    }
    if let Err(line) = push_current_head(cwd) {
        report.push(line);
        return report.join("\n");
    }
    append_pr_create_step(&mut report, cwd);
    report.join("\n")
}

fn git_output_at(args: &[&str], cwd: &Path) -> io::Result<String> {
    let output = Command::new("git").args(args).current_dir(cwd).output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    }
}

fn stage_all_changes(report: &mut Vec<String>, cwd: &Path) -> Result<(), String> {
    match run_hardened_command_in(
        "git",
        &["add", "-A"],
        COMMIT_PUSH_PR_COMMAND_TIMEOUT,
        Some(cwd),
    ) {
        Ok(output) if output.status.success() => {
            report.push("  Step 1           git add -A ... ok".to_string());
            Ok(())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Commit-Push-PR\n  Step 1 failed    git add -A: {}",
                stderr.trim()
            ))
        }
        Err(error) => Err(format!(
            "Commit-Push-PR\n  Step 1 failed    git add: {error}"
        )),
    }
}

fn commit_staged_changes(
    report: &mut Vec<String>,
    commit_msg: &str,
    cwd: &Path,
) -> Result<(), String> {
    match run_hardened_command_in(
        "git",
        &["commit", "-m", commit_msg],
        COMMIT_PUSH_PR_COMMAND_TIMEOUT,
        Some(cwd),
    ) {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            report.push(format!(
                "  Step 2           git commit ... ok\n    {}",
                stdout.lines().next().unwrap_or("")
            ));
            Ok(())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "Commit-Push-PR\n  Step 2 failed    git commit: {}",
                stderr.trim()
            ))
        }
        Err(error) => Err(format!(
            "Commit-Push-PR\n  Step 2 failed    git commit: {error}"
        )),
    }
}

fn push_current_head(cwd: &Path) -> Result<(), String> {
    match run_hardened_command_in(
        "git",
        &["push", "-u", "origin", "HEAD"],
        COMMIT_PUSH_PR_COMMAND_TIMEOUT,
        Some(cwd),
    ) {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("  Step 3 failed    git push: {}", stderr.trim()))
        }
        Err(error) => Err(format!("  Step 3 failed    git push: {error}")),
    }
}

fn append_pr_create_step(report: &mut Vec<String>, cwd: &Path) {
    report.push("  Step 3           git push ... ok".to_string());
    match run_hardened_command_in(
        "gh",
        &["pr", "create", "--fill"],
        COMMIT_PUSH_PR_COMMAND_TIMEOUT,
        Some(cwd),
    ) {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            report.push(format!(
                "  Step 4           gh pr create ... ok\n    {}",
                stdout.trim()
            ));
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            report.push(format!(
                "  Step 4 warning   gh pr create: {}",
                stderr.trim()
            ));
        }
        Err(HardenedCommandError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            report.push(
                "  Step 4 skipped   gh not found. Install GitHub CLI to create PRs.".to_string(),
            );
        }
        Err(error) => {
            report.push(format!("  Step 4 warning   gh pr create: {error}"));
        }
    }
}

fn generate_commit_message(cwd: &Path) -> String {
    match git_output_at(&["diff", "--cached", "--stat"], cwd) {
        Ok(stat) => {
            let file_count = stat.lines().count().saturating_sub(1);
            let first_file = stat
                .lines()
                .next()
                .unwrap_or("")
                .split('|')
                .next()
                .unwrap_or("")
                .trim();
            if file_count <= 1 {
                format!("update {first_file}")
            } else {
                format!("update {file_count} files ({first_file}, ...)")
            }
        }
        Err(_) => "auto-commit via /commit-push-pr".to_string(),
    }
}

fn indent_lines(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn handle_backfill_sessions() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let data_dir = if cfg!(target_os = "macos") {
        std::path::PathBuf::from(&home)
            .join("Library")
            .join("Application Support")
            .join("zo-cli")
    } else {
        std::path::PathBuf::from(&home)
            .join(".local")
            .join("share")
            .join("zo-cli")
    };
    if !data_dir.is_dir() {
        return "Backfill Sessions\n  Result           No session directory found.\n  Searched         ~/.local/share/zo-cli/".to_string();
    }
    let mut migrated = 0u32;
    let mut errors = 0u32;
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "history"
                || path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("rustyline"))
            {
                let dest = path.with_extension("jsonl");
                if dest.exists() {
                    continue;
                }
                match std::fs::read_to_string(&path) {
                    Ok(contents) => {
                        let mut jsonl_lines = Vec::new();
                        for line in contents.lines() {
                            let line = line.trim();
                            if line.is_empty() {
                                continue;
                            }
                            let escaped = serde_json::to_string(line).unwrap_or_default();
                            jsonl_lines.push(format!("{{\"input\":{escaped}}}"));
                        }
                        if std::fs::write(&dest, jsonl_lines.join("\n")).is_err() {
                            errors += 1;
                        } else {
                            migrated += 1;
                        }
                    }
                    Err(_) => errors += 1,
                }
            }
        }
    }
    if migrated == 0 && errors == 0 {
        "Backfill Sessions\n  Result           No sessions to backfill.\n  Tip              .jsonl session files in .zo/sessions/ load via /resume"
            .to_string()
    } else {
        let mut lines = vec![format!(
            "Backfill Sessions\n  Migrated         {migrated} session(s)"
        )];
        if errors > 0 {
            lines.push(format!("  Errors           {errors}"));
        }
        lines.push(format!("  Directory        {}", data_dir.display()));
        lines.join("\n")
    }
}

pub(super) fn handle_extra_usage(usage: &runtime::UsageTracker) -> String {
    let cu = usage.cumulative_usage();
    let model = "claude-sonnet-4-20250514";
    let input_cost = f64::from(cu.input_tokens) * 3.0 / 1_000_000.0;
    let output_cost = f64::from(cu.output_tokens) * 15.0 / 1_000_000.0;
    let cache_write_cost = f64::from(cu.cache_creation_input_tokens) * 3.75 / 1_000_000.0;
    let cache_read_cost = f64::from(cu.cache_read_input_tokens) * 0.30 / 1_000_000.0;
    let total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost;
    format!(
        "Extra Usage\n\n  {:<20} {:>12} {:>10}\n  {:-<20} {:->12} {:->10}\n  {:<20} {:>12} ${:>8.4}\n  {:<20} {:>12} ${:>8.4}\n  {:<20} {:>12} ${:>8.4}\n  {:<20} {:>12} ${:>8.4}\n  {:-<20} {:->12} {:->10}\n  {:<20} {:>12} ${:>8.4}\n\n  Turns              {}\n  Model              {}",
        "Category",
        "Tokens",
        "Est. Cost",
        "",
        "",
        "",
        "Input",
        cu.input_tokens,
        input_cost,
        "Output",
        cu.output_tokens,
        output_cost,
        "Cache write",
        cu.cache_creation_input_tokens,
        cache_write_cost,
        "Cache read",
        cu.cache_read_input_tokens,
        cache_read_cost,
        "",
        "",
        "",
        "Total",
        cu.total_tokens(),
        total_cost,
        usage.turns(),
        model,
    )
}

pub(super) fn handle_perf_issue(usage: &runtime::UsageTracker) -> String {
    let cu = usage.cumulative_usage();
    let turns = usage.turns();
    let rss = match std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
    {
        Ok(output) if output.status.success() => {
            let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();
            raw.parse::<u64>().unwrap_or(0)
        }
        _ => 0,
    };
    #[allow(clippy::cast_precision_loss)]
    let rss_mb = rss as f64 / 1024.0;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let snapshot = serde_json::json!({
        "timestamp": timestamp,
        "rss_kb": rss,
        "rss_mb": format!("{rss_mb:.1}"),
        "session_turns": turns,
        "total_tokens": cu.total_tokens(),
        "input_tokens": cu.input_tokens,
        "output_tokens": cu.output_tokens,
    });
    let evidence_dir = std::env::current_dir()
        .unwrap_or_default()
        .join(".zo")
        .join("evidence");
    let _ = std::fs::create_dir_all(&evidence_dir);
    let path = evidence_dir.join(format!("perf-{timestamp}.json"));
    let saved = match std::fs::write(
        &path,
        serde_json::to_string_pretty(&snapshot).unwrap_or_default(),
    ) {
        Ok(()) => format!("  Saved            {}", path.display()),
        Err(e) => format!("  Save failed      {e}"),
    };
    format!(
        "Performance Snapshot\n\n{saved}\n\n  RSS memory         {rss_mb:.1} MB\n  Session turns      {turns}\n  Total tokens       {}\n  Input tokens       {}\n  Output tokens      {}",
        cu.total_tokens(),
        cu.input_tokens,
        cu.output_tokens,
    )
}

pub(super) fn handle_statusline() -> String {
    let fields = [
        ("model", true),
        ("permission_mode", true),
        ("tokens", true),
        ("cost", true),
        ("session_id", false),
        ("git_branch", true),
        ("cwd", false),
        ("thinking_budget", false),
    ];
    let mut lines = vec!["Statusline / HUD Configuration\n".to_string()];
    lines.push(format!("  {:<24} {}", "Field", "Status"));
    lines.push(format!("  {:-<24} {:-<10}", "", ""));
    for (field, on) in &fields {
        let status = if *on { "on" } else { "off" };
        lines.push(format!("  {field:<24} {status}"));
    }
    lines.push(String::new());
    lines.push(
        "  Tip              Use settings.json `statusLine` to customize the HUD.".to_string(),
    );
    lines.push(
        "  Path             ~/.zo/settings.json or .zo/settings.json -> \"statusLine\""
            .to_string(),
    );
    lines.join("\n")
}

pub(super) fn handle_ant_trace() -> String {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let evidence_dir = std::env::current_dir()
        .unwrap_or_default()
        .join(".zo")
        .join("evidence");
    let _ = std::fs::create_dir_all(&evidence_dir);
    let current = std::env::var("ANTHROPIC_LOG").unwrap_or_else(|_| "(not set)".to_string());
    let trace_info = serde_json::json!({
        "timestamp": timestamp,
        "instructions": "Set ANTHROPIC_LOG=debug before starting the CLI to capture full API traces.",
        "env_var": "ANTHROPIC_LOG=debug",
        "current_value": current,
    });
    let path = evidence_dir.join(format!("trace-{timestamp}.json"));
    let saved = match std::fs::write(
        &path,
        serde_json::to_string_pretty(&trace_info).unwrap_or_default(),
    ) {
        Ok(()) => format!("  Saved            {}", path.display()),
        Err(e) => format!("  Save failed      {e}"),
    };
    format!(
        "Anthropic API Trace\n\n{saved}\n\n  How to capture full API traces:\n\n  1. Set the environment variable before starting the CLI:\n     export ANTHROPIC_LOG=debug\n\n  2. Re-run your session. Full request/response payloads will\n     appear in stderr output.\n\n  3. To save to a file:\n     ANTHROPIC_LOG=debug zo 2> trace.log\n\n  Current ANTHROPIC_LOG   {current}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zo-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn backfill_sessions_converts_zo_history_in_place() {
        let _env_lock = crate::test_env_lock();
        let root = temp_dir("backfill");
        let zo_dir = if cfg!(target_os = "macos") {
            root.join("Library")
                .join("Application Support")
                .join("zo-cli")
        } else {
            root.join(".local").join("share").join("zo-cli")
        };
        std::fs::create_dir_all(&zo_dir).expect("create zo data dir");
        std::fs::write(zo_dir.join("session.history"), "first prompt\nsecond prompt\n")
            .expect("write history");

        let prior_home = std::env::var_os("HOME");
        std::env::set_var("HOME", &root);
        let result = std::panic::catch_unwind(handle_backfill_sessions);
        match prior_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        let report = result.unwrap_or_else(|payload| std::panic::resume_unwind(payload));

        assert!(report.contains("Migrated         1 session(s)"), "{report}");
        let migrated = std::fs::read_to_string(zo_dir.join("session.jsonl"))
            .expect("read converted history");
        assert!(migrated.contains(r#"{"input":"first prompt"}"#));
        assert!(migrated.contains(r#"{"input":"second prompt"}"#));

        std::fs::remove_dir_all(root).expect("cleanup temp dir");
    }

    #[test]
    fn statusline_report_points_to_statusline_settings() {
        let report = handle_statusline();
        assert!(report.contains("statusLine"));
        assert!(report.contains("~/.zo/settings.json"));
        assert!(!report.contains("\"hud\""));
    }

    #[test]
    fn hardened_command_sets_noninteractive_env_and_null_stdin() {
        let output = run_hardened_command_in(
            "sh",
            &[
                "-c",
                r#"test "$GIT_TERMINAL_PROMPT" = 0 && test "$GCM_INTERACTIVE" = never && ! read line"#,
            ],
            Duration::from_secs(2),
            None,
        )
        .expect("hardened command should run");

        assert!(
            output.status.success(),
            "command should see noninteractive env and EOF stdin: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn hardened_command_uses_supplied_cwd() {
        let cwd = temp_dir("commit-push-pr-cwd");
        let output =
            run_hardened_command_in("sh", &["-c", "pwd"], Duration::from_secs(2), Some(&cwd))
                .expect("hardened command should run in cwd");

        assert!(
            output.status.success(),
            "pwd command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let printed = std::path::PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        assert_eq!(
            std::fs::canonicalize(&printed).expect("printed cwd canonicalizes"),
            std::fs::canonicalize(&cwd).expect("expected cwd canonicalizes")
        );
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn commit_message_reads_index_from_supplied_cwd() {
        let cwd = temp_dir("commit-push-pr-git");
        run_hardened_command_in("git", &["init"], Duration::from_secs(2), Some(&cwd))
            .expect("git init");
        std::fs::write(cwd.join("note.txt"), "hello\n").expect("write note");
        run_hardened_command_in(
            "git",
            &["add", "note.txt"],
            Duration::from_secs(2),
            Some(&cwd),
        )
        .expect("git add");

        let message = generate_commit_message(&cwd);
        assert!(
            message.contains("note.txt"),
            "commit message should use supplied repo cwd, got {message:?}"
        );
        std::fs::remove_dir_all(cwd).ok();
    }

    #[test]
    fn hardened_command_times_out() {
        let result =
            run_hardened_command_in("sh", &["-c", "sleep 2"], Duration::from_millis(20), None);

        assert!(
            matches!(result, Err(HardenedCommandError::TimedOut { .. })),
            "expected timeout, got {result:?}"
        );
    }
}
