use std::fmt::Write as _;

use runtime::TokenUsage;

use crate::{LATEST_SESSION_REFERENCE, PRIMARY_SESSION_EXTENSION};

#[cfg(test)]
pub(crate) fn format_unknown_slash_command_message(name: &str) -> String {
    let suggestions = crate::cli_args::suggest_slash_commands(name);
    if suggestions.is_empty() {
        format!("unknown slash command: /{name}. Use /help to list available commands.")
    } else {
        format!(
            "unknown slash command: /{name}. Did you mean {}? Use /help to list available commands.",
            suggestions.join(", ")
        )
    }
}

pub(crate) fn format_model_report(model: &str, message_count: usize, turns: u32) -> String {
    format!(
        "Model
  Current model    {model}
  Session messages {message_count}
  Session turns    {turns}

Usage
  Inspect current model with /model
  Switch models with /model <name>"
    )
}

pub(crate) fn format_model_switch_report(
    previous: &str,
    next: &str,
    message_count: usize,
) -> String {
    format!(
        "Model updated
  Previous         {previous}
  Current          {next}
  Preserved msgs   {message_count}"
    )
}

pub(crate) fn format_permissions_report(mode: &str) -> String {
    let modes = [
        ("read-only", "Read/search tools only", mode == "read-only"),
        (
            "workspace-write",
            "Edit files inside the workspace",
            mode == "workspace-write",
        ),
        (
            "danger-full-access",
            "Unrestricted tool access",
            mode == "danger-full-access",
        ),
    ]
    .into_iter()
    .map(|(name, description, is_current)| {
        let marker = if is_current {
            "● current"
        } else {
            "○ available"
        };
        format!("  {name:<18} {marker:<11} {description}")
    })
    .collect::<Vec<_>>()
    .join(
        "
",
    );

    format!(
        "Permissions
  Active mode      {mode}
  Mode status      live session default

Modes
{modes}

Rules (settings.json -> permissions.allow / deny / ask)
  Form             tool(subject)
  Subject          exact (git status) | prefix `name:*` | glob `* ? [..]`
  Examples         bash(git:*), edit_file(*.env), bash(git push*)
  Precedence       deny > ask > allow

Usage
  Inspect current mode with /permissions
  Switch modes with /permissions <mode>"
    )
}

pub(crate) fn format_permissions_switch_report(previous: &str, next: &str) -> String {
    format!(
        "Permissions updated
  Result           mode switched
  Previous mode    {previous}
  Active mode      {next}
  Applies to       subsequent tool calls
  Usage            /permissions to inspect current mode"
    )
}

pub(crate) fn format_cost_report(usage: TokenUsage) -> String {
    format!(
        "Cost
  Input tokens     {}
  Output tokens    {}
  Cache create     {}
  Cache read       {}
  Total tokens     {}",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_read_input_tokens,
        usage.total_tokens(),
    )
}

pub(crate) fn format_resume_report(session_path: &str, message_count: usize, turns: u32) -> String {
    format!(
        "Session resumed
  Session file     {session_path}
  Messages         {message_count}
  Turns            {turns}"
    )
}

pub(crate) fn render_resume_usage() -> String {
    let mut out = format!(
        "Resume\n  Usage            /resume <session-path|session-id|{LATEST_SESSION_REFERENCE}>\n  Auto-save        ~/.zo/projects/<project>/sessions/<session-id>.{PRIMARY_SESSION_EXTENSION}\n"
    );
    match list_recent_sessions(10) {
        Ok(entries) if !entries.is_empty() => {
            out.push_str("  Recent sessions\n");
            for entry in entries {
                let _ = writeln!(out, "    • {entry}");
            }
            out.push_str(
                "  Tip              /resume <session-id> to restore; /session list for details",
            );
        }
        _ => {
            out.push_str("  Recent sessions  (none found in .zo/sessions/)\n");
            out.push_str("  Tip              use /session list to inspect saved sessions");
        }
    }
    out
}

fn list_recent_sessions(limit: usize) -> std::io::Result<Vec<String>> {
    use std::fs;
    let dir = std::path::Path::new(".zo/sessions");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<(std::time::SystemTime, String)> = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter_map(|e| {
            let path = e.path();
            let ext = path.extension()?.to_str()?;
            if ext != PRIMARY_SESSION_EXTENSION && ext != "json" {
                return None;
            }
            let meta = e.metadata().ok()?;
            let mtime = meta.modified().ok()?;
            let stem = path.file_stem()?.to_str()?.to_string();
            let size = meta.len();
            Some((mtime, format!("{stem}  ({size} bytes)")))
        })
        .collect();
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(entries.into_iter().take(limit).map(|(_, s)| s).collect())
}

pub(crate) fn format_compact_report(
    removed: usize,
    resulting_messages: usize,
    skipped: bool,
) -> String {
    if skipped {
        format!(
            "Compact
  Result           skipped
  Reason           session below compaction threshold
  Messages kept    {resulting_messages}"
        )
    } else {
        format!(
            "Compact
  Result           compacted
  Messages removed {removed}
  Messages kept    {resulting_messages}"
        )
    }
}

pub(crate) fn format_auto_compaction_notice(removed: usize) -> String {
    format!("Compacted conversation · {removed} messages summarized")
}
