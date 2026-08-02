//! Read-only "view" commands: status, cost, usage, cache, context,
//! doctor, version, config, memory, diff, agents, skills, mcp,
//! plugins, help.
//!
//! Each handler computes a [`CommandOutput`] (usually a rich
//! [`CardModel`]) from the live session and never mutates it — except
//! `/diff`, which may open the interactive viewer and return
//! [`CommandOutput::Quiet`].

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use core_types::{
    CardModel, CardTone, JsonValue, UsageDashboardRecord, UsageDashboardSnapshot,
    UsageTokenTotals,
};
use zo_cli::tui::modals::UsageDashboardModal;

use crate::git_output;

use super::super::report_services;
use super::context::{DispatchCtx, DispatchError};
use super::helpers_tui::build_help_card;
use super::output::CommandOutput;

pub(super) fn help(ctx: &mut DispatchCtx) -> CommandOutput {
    CommandOutput::popup_card(build_help_card(&ctx.cli.runtime.prompt_commands))
}

pub(super) fn status(ctx: &mut DispatchCtx) -> CommandOutput {
    CommandOutput::popup_card(ctx.cli.status_card())
}

pub(super) fn cost(ctx: &mut DispatchCtx) -> CommandOutput {
    CommandOutput::popup_card(ctx.cli.cost_card())
}

pub(super) fn config(section: Option<&str>) -> Result<CommandOutput, DispatchError> {
    let report = report_services::config_report(section)?;
    Ok(CommandOutput::popup_card(CardModel::from_report_text(
        " /config ",
        &report,
    )))
}

/// Working-tree diff vs `HEAD` → interactive viewer when non-empty, else
/// the textual summary card (clean tree / untracked-only).
pub(super) fn diff(ctx: &mut DispatchCtx) -> Result<CommandOutput, DispatchError> {
    let diff_text = git_output(&["diff", "HEAD", "--no-color"]).unwrap_or_default();
    let files = zo_cli::tui::modals::diff_viewer::parse_unified_diff(&diff_text);
    if files.is_empty() {
        let report = report_services::diff_report()?;
        Ok(CommandOutput::popup_card(CardModel::from_report_text(
            " /diff ", &report,
        )))
    } else {
        ctx.app.open_diff_viewer(files);
        Ok(CommandOutput::Quiet)
    }
}

/// Bare `/agents` opens the live agents viewer (the Ctrl+G modal) instead of a
/// static report; argumented forms (`/agents <name>`, filters) keep the
/// textual popup, which is where the detail listing lives.
pub(super) fn agents(
    ctx: &mut DispatchCtx,
    args: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    if args.map(str::trim).is_none_or(str::is_empty) {
        ctx.app.open_agents_viewer();
        return Ok(CommandOutput::Quiet);
    }
    let report = report_services::agents_report(args)?;
    Ok(CommandOutput::popup_card(CardModel::from_report_text(
        " /agents ",
        &report,
    )))
}

pub(super) fn inbox(ctx: &mut DispatchCtx, args: Option<&str>) -> CommandOutput {
    if args.map(str::trim).is_some_and(|args| !args.is_empty()) {
        let report = report_services::inbox_command(&ctx.cli.cwd, &ctx.cli.session.id, args);
        return CommandOutput::popup_card(CardModel::from_report_text(" /inbox ", &report));
    }
    let snapshot = runtime::team_inbox_snapshot(&ctx.cli.cwd, &ctx.cli.session.id, 50);
    ctx.app.open_team_inbox_viewer(snapshot);
    CommandOutput::Quiet
}

pub(super) fn skills(args: Option<&str>) -> Result<CommandOutput, DispatchError> {
    let report = report_services::skills_report(args)?;
    Ok(CommandOutput::popup_card(CardModel::from_report_text(
        " /skills ",
        &report,
    )))
}

pub(super) fn mcp(
    ctx: &mut DispatchCtx,
    action: Option<&str>,
    target: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    let args = match (action, target) {
        (None, None) => None,
        (Some(action), None) => Some(action.to_string()),
        (Some(action), Some(target)) => Some(format!("{action} {target}")),
        (None, Some(target)) => Some(target.to_string()),
    };
    let mut report = report_services::mcp_report(args.as_deref())?;
    // The bare `/mcp` overview also lists live-discovered prompts (the
    // dynamic `/mcp__server__prompt` slash commands, Claude Code parity).
    if args.is_none() {
        if let Some(section) = mcp_prompts_section(ctx) {
            report.push_str(&section);
        }
    }
    Ok(CommandOutput::popup_card(CardModel::from_report_text(
        " /mcp ", &report,
    )))
}

/// Render the discovered-prompts section for `/mcp`, or `None` when no
/// server advertises prompts (or discovery is still holding the lock —
/// the section degrades to hidden rather than blocking the input loop).
fn mcp_prompts_section(ctx: &DispatchCtx) -> Option<String> {
    use std::fmt::Write as _;

    let mcp_state = ctx.cli.runtime.mcp_state.as_ref()?;
    let prompts = mcp_state.try_lock().ok()?.prompts_snapshot();
    if prompts.is_empty() {
        return None;
    }

    let mut section = String::from("\n\nPrompts (slash commands)\n");
    for entry in &prompts {
        let summary = entry
            .prompt
            .description
            .clone()
            .or_else(|| entry.prompt.title.clone())
            .unwrap_or_else(|| format!("MCP prompt from `{}`", entry.server));
        let _ = writeln!(section, "  /{:<32} {summary}", entry.command);
    }
    section.push_str("  Run one to queue its rendered text as your next turn.");
    Some(section)
}

pub(super) fn version() -> CommandOutput {
    CommandOutput::popup_card(CardModel::from_report_text(
        " /version ",
        &report_services::version_report(),
    ))
}

pub(super) fn doctor(ctx: &mut DispatchCtx) -> CommandOutput {
    // Route through the shared doctor engine in default repair mode so the
    // interactive `/doctor`, headless `/doctor`, and `zo doctor` all diagnose
    // identically and apply the same automatic safe repairs. The engine is
    // secret-safe and performs no network I/O.
    let cwd = ctx.cli.cwd.clone();
    let report = crate::doctor::run(crate::doctor::DoctorMode::Repair, &cwd);
    CommandOutput::popup_card(CardModel::from_report_text(" /doctor ", &report.render()))
}

/// `/audit` — operator view of the `ToolGateway` ledger (WI-E2): per-tool call
/// counts, permission denials, and route decisions recorded this session.
pub(super) fn audit(ctx: &mut DispatchCtx) -> CommandOutput {
    let (invocations, summary) = {
        let tctx = ctx
            .cli
            .runtime
            .tool_executor_mut()
            .tool_registry_mut()
            .context();
        (tctx.tool_invocations(), tctx.audit_summary())
    };
    let mut by_tool: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for inv in &invocations {
        *by_tool.entry(inv.request.tool_name.clone()).or_default() += 1;
    }
    let mut card = CardModel::new(" /audit ")
        .section("Tools")
        .metric("total", summary.total.to_string(), CardTone::Accent)
        .metric("ok", summary.succeeded.to_string(), CardTone::Default)
        .metric("failed", summary.failed.to_string(), CardTone::Default);
    if !by_tool.is_empty() {
        card = card.section("By tool");
        for (name, count) in &by_tool {
            card = card.metric(name.clone(), count.to_string(), CardTone::Default);
        }
    }
    if !summary.denials.is_empty() {
        card = card.section("Denied").metric(
            "count",
            summary.denials.len().to_string(),
            CardTone::Default,
        );
    }
    if !summary.route_decisions.is_empty() {
        card = card.section("Routes").metric(
            "count",
            summary.route_decisions.len().to_string(),
            CardTone::Default,
        );
    }
    CommandOutput::popup_card(card)
}

pub(super) fn usage(ctx: &mut DispatchCtx, _scope: Option<&str>) -> CommandOutput {
    let snapshot = historical_usage_snapshot(ctx);
    ctx.app
        .open_usage_dashboard_modal(UsageDashboardModal::new(snapshot));
    CommandOutput::Quiet
}

fn historical_usage_snapshot(ctx: &mut DispatchCtx) -> UsageDashboardSnapshot {
    const MAX_USAGE_SESSIONS: usize = 512;

    let baseline_model = ctx.cli.model.clone();
    let active_session_id = ctx.cli.session.id.as_str();
    let mut seen_session_ids = HashSet::new();
    seen_session_ids.insert(active_session_id.to_string());
    let mut records = Vec::new();
    let mut scanned_sessions = 0usize;
    let mut skipped_sessions = 0usize;
    let mut duplicate_sessions = 0usize;

    match crate::session_registry::managed_session_paths_limited(Some(MAX_USAGE_SESSIONS)) {
        Ok(paths) => {
            for path in paths {
                let Ok(summary) = load_usage_summary(&path) else {
                    skipped_sessions = skipped_sessions.saturating_add(1);
                    continue;
                };
                if summary.usage.total_tokens() == 0 {
                    continue;
                }
                if !seen_session_ids.insert(summary.session_id.clone()) {
                    duplicate_sessions = duplicate_sessions.saturating_add(1);
                    continue;
                }
                scanned_sessions = scanned_sessions.saturating_add(1);
                records.push(UsageDashboardRecord {
                    session_id: summary.session_id,
                    occurred_at_ms: summary.updated_at_ms,
                    model: session_model(&path, &baseline_model),
                    usage: summary.usage,
                });
            }
        }
        Err(_) => skipped_sessions = skipped_sessions.saturating_add(1),
    }

    let live_usage = ctx.cli.runtime.usage();
    let live_total = live_usage.cumulative_usage();
    let mut live_record_included = false;
    if live_total.total_tokens_u64() > 0 || live_usage.turns() > 0 {
        live_record_included = true;
        records.push(UsageDashboardRecord {
            session_id: active_session_id.to_string(),
            occurred_at_ms: current_unix_millis(),
            model: baseline_model.clone(),
            usage: UsageTokenTotals::from_usage(live_total),
        });
    }

    let live_note = if live_record_included {
        " + current live session"
    } else {
        ""
    };
    let mut note = format!(
        "Historical estimates · {scanned_sessions} saved session(s){live_note} · session-level dates/models; turn-level ledger pending"
    );
    if duplicate_sessions > 0 {
        let _ = write!(note, " · {duplicate_sessions} duplicate file(s) ignored");
    }
    if skipped_sessions > 0 {
        let _ = write!(note, " · {skipped_sessions} unreadable file(s) skipped");
    }
    UsageDashboardSnapshot::from_records(baseline_model, records, note)
}

#[derive(Debug, Clone)]
struct UsageSessionSummary {
    session_id: String,
    updated_at_ms: u64,
    usage: UsageTokenTotals,
}

fn load_usage_summary(path: &Path) -> Result<UsageSessionSummary, String> {
    if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
        load_jsonl_usage_summary(path)
    } else {
        load_json_usage_summary(path)
    }
}

fn load_jsonl_usage_summary(path: &Path) -> Result<UsageSessionSummary, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let reader = BufReader::new(file);
    let mut session_id = None;
    let mut updated_at_ms = None;
    let mut usage = UsageTokenTotals::default();

    for line in reader.lines() {
        let line = line.map_err(|error| error.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.contains("\"session_meta\"") {
            let Ok(value) = JsonValue::parse(line) else {
                continue;
            };
            let Some(object) = value.as_object() else {
                continue;
            };
            if let Some(value) = object.get("session_id").and_then(JsonValue::as_str) {
                session_id = Some(value.to_string());
            }
            if let Some(value) = object.get("updated_at_ms").and_then(json_u64) {
                updated_at_ms = Some(value);
            }
        } else if line.contains("\"usage\"") {
            add_message_usage_from_line(line, &mut usage);
        }
    }

    Ok(UsageSessionSummary {
        session_id: session_id.ok_or_else(|| "missing session_id".to_string())?,
        updated_at_ms: updated_at_ms.unwrap_or_else(current_unix_millis),
        usage,
    })
}

fn load_json_usage_summary(path: &Path) -> Result<UsageSessionSummary, String> {
    const MAX_LEGACY_JSON_SESSION_BYTES: u64 = 8 * 1024 * 1024;

    let len = path.metadata().map_err(|error| error.to_string())?.len();
    if len > MAX_LEGACY_JSON_SESSION_BYTES {
        return Err(format!(
            "legacy JSON session is too large for synchronous usage scan ({len} bytes)"
        ));
    }
    let mut source = String::new();
    File::open(path)
        .map_err(|error| error.to_string())?
        .read_to_string(&mut source)
        .map_err(|error| error.to_string())?;
    let value = JsonValue::parse(&source).map_err(|error| error.to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "session must be an object".to_string())?;
    let session_id = object
        .get("session_id")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| "missing session_id".to_string())?
        .to_string();
    let updated_at_ms = object
        .get("updated_at_ms")
        .and_then(json_u64)
        .unwrap_or_else(current_unix_millis);
    let mut usage = UsageTokenTotals::default();
    if let Some(messages) = object.get("messages").and_then(JsonValue::as_array) {
        for message in messages {
            add_message_usage(message, &mut usage);
        }
    }
    Ok(UsageSessionSummary {
        session_id,
        updated_at_ms,
        usage,
    })
}

fn add_message_usage_from_line(line: &str, total: &mut UsageTokenTotals) {
    let Some(usage_source) = extract_usage_object(line) else {
        return;
    };
    let Ok(value) = JsonValue::parse(usage_source) else {
        return;
    };
    if let Some(usage) = parse_token_usage(&value) {
        add_usage(total, usage);
    }
}

fn extract_usage_object(line: &str) -> Option<&str> {
    let key_index = find_json_key_outside_strings(line, "usage")?;
    let after_key = &line[key_index + "\"usage\"".len()..];
    let brace_offset = after_key.find('{')?;
    let start = key_index + "\"usage\"".len() + brace_offset;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in line[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth = depth.saturating_add(1),
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return Some(&line[start..end]);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_json_key_outside_strings(line: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let bytes = line.as_bytes();
    let needle_bytes = needle.as_bytes();
    let mut idx = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    while idx < bytes.len() {
        let byte = bytes[idx];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            idx += 1;
            continue;
        }
        if byte == b'"' {
            if bytes[idx..].starts_with(needle_bytes) {
                let after = idx + needle_bytes.len();
                if line[after..].trim_start().starts_with(':') {
                    return Some(idx);
                }
            }
            in_string = true;
        }
        idx += 1;
    }
    None
}

fn add_message_usage(message: &JsonValue, total: &mut UsageTokenTotals) {
    let Some(usage) = message
        .as_object()
        .and_then(|object| object.get("usage"))
        .and_then(parse_token_usage)
    else {
        return;
    };
    add_usage(total, usage);
}

fn parse_token_usage(value: &JsonValue) -> Option<UsageTokenTotals> {
    let object = value.as_object()?;
    Some(UsageTokenTotals {
        input_tokens: object.get("input_tokens").and_then(json_u64)?,
        output_tokens: object.get("output_tokens").and_then(json_u64)?,
        cache_creation_input_tokens: object
            .get("cache_creation_input_tokens")
            .and_then(json_u64)
            .unwrap_or(0),
        cache_read_input_tokens: object
            .get("cache_read_input_tokens")
            .and_then(json_u64)
            .unwrap_or(0),
    })
}

fn json_u64(value: &JsonValue) -> Option<u64> {
    u64::try_from(value.as_i64()?).ok()
}

fn add_usage(total: &mut UsageTokenTotals, usage: UsageTokenTotals) {
    total.input_tokens = total.input_tokens.saturating_add(usage.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(usage.output_tokens);
    total.cache_creation_input_tokens = total
        .cache_creation_input_tokens
        .saturating_add(usage.cache_creation_input_tokens);
    total.cache_read_input_tokens = total
        .cache_read_input_tokens
        .saturating_add(usage.cache_read_input_tokens);
}

fn session_model(session_path: &Path, fallback_model: &str) -> String {
    super::super::session_preferences::load_session_preferences(session_path)
        .model
        .filter(|model| !model.trim().is_empty())
        .unwrap_or_else(|| fallback_model.to_string())
}

fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

pub(super) fn cache(ctx: &mut DispatchCtx) -> CommandOutput {
    let usage = ctx.cli.runtime.usage().cumulative_usage();
    let cache_created = usage.cache_creation_input_tokens;
    let cache_read = usage.cache_read_input_tokens;
    let total_input = usage.input_tokens;
    let cache_pct = if total_input > 0 {
        f64::from(cache_read) / f64::from(total_input) * 100.0
    } else {
        0.0
    };
    CommandOutput::popup(
        "/cache",
        format!(
            "Cache\n  Cache creation    {cache_created} tokens\n  Cache read        {cache_read} tokens\n  Total input       {total_input} tokens\n  Cache hit rate    {cache_pct:.1}%\n  Tip               Higher cache read = better reuse. System prompt changes break the cache.",
        ),
    )
}

pub(super) fn context(ctx: &mut DispatchCtx, action: Option<&str>) -> CommandOutput {
    let session = ctx.cli.runtime.session();
    let total = session.messages.len();
    let mut user_count = 0usize;
    let mut assistant_count = 0usize;
    let mut tool_count = 0usize;
    let mut system_count = 0usize;
    for msg in session.messages.iter() {
        match msg.role {
            runtime::MessageRole::User => user_count += 1,
            runtime::MessageRole::Assistant => assistant_count += 1,
            runtime::MessageRole::Tool => tool_count += 1,
            runtime::MessageRole::System => system_count += 1,
        }
    }
    let est_tokens = runtime::estimate_session_tokens(session);
    let cwd = std::env::current_dir()
        .map_or_else(|_| "(unknown)".to_string(), |p| p.display().to_string());

    let mut lines = vec![
        "Context".to_string(),
        format!("  Messages         {total}"),
        format!("    user           {user_count}"),
        format!("    assistant      {assistant_count}"),
        format!("    tool           {tool_count}"),
        format!("    system         {system_count}"),
        format!("  Est. tokens      {est_tokens}"),
    ];
    if let Some(ref compaction) = session.compaction {
        lines.push(format!("  Compactions      {}", compaction.count));
        lines.push(format!(
            "  Msgs removed     {}",
            compaction.removed_message_count
        ));
    } else {
        lines.push("  Compactions      0".to_string());
    }
    lines.push(format!("  Working dir      {cwd}"));
    if let Some(act) = action {
        lines.push(format!("  Action           {act}"));
    }
    CommandOutput::popup("/context", lines.join("\n"))
}

pub(super) fn plugins(
    ctx: &mut DispatchCtx,
    action: Option<&str>,
    target: Option<&str>,
) -> CommandOutput {
    let report = match (action, target) {
        (None | Some("list"), _) => {
            let summaries = ctx.cli.runtime.plugin_registry.summaries();
            if summaries.is_empty() {
                "Plugins\n  No plugins loaded.".to_string()
            } else {
                let mut lines = vec![format!("Plugins  ({} loaded)\n", summaries.len())];
                for s in &summaries {
                    let status = if s.enabled { "enabled" } else { "disabled" };
                    lines.push(format!(
                        "  {:<24} v{:<10} {:>8}  [{}]",
                        s.metadata.name, s.metadata.version, status, s.metadata.kind,
                    ));
                    if !s.metadata.description.is_empty() {
                        lines.push(format!("  {:<24} {}", "", s.metadata.description));
                    }
                }
                lines.join("\n")
            }
        }
        (Some("install" | "enable" | "disable" | "uninstall" | "update"), Some(t)) => format!(
            "Plugins\n  /plugins {} {t} — requires REPL mode for plugin manager access",
            action.unwrap_or("")
        ),
        (Some(a), Some(t)) => format!(
            "Plugins\n  /plugins {a} {t} — unknown action\n  Available: list, install, enable, disable, uninstall, update"
        ),
        (Some(a), None) => {
            format!("Plugins\n  /plugins {a} — requires REPL mode for plugin manager access")
        }
    };
    CommandOutput::popup("/plugins", report)
}


#[cfg(test)]
mod usage_dashboard_tests {
    use super::*;

    #[test]
    fn extracts_usage_object_without_parsing_large_message_line() {
        let line = format!(
            r#"{{"type":"message","message":{{"blocks":[{{"text":"{}"}}],"usage":{{"input_tokens":11,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":4}}}}}}"#,
            "x".repeat(16_384)
        );
        let usage_source = extract_usage_object(&line).expect("usage object");
        assert!(usage_source.starts_with('{'));
        assert!(usage_source.contains("\"input_tokens\":11"));
        assert!(usage_source.len() < 128);
    }

    #[test]
    fn jsonl_usage_summary_reads_meta_and_usage_without_full_session_load() {
        let path = std::env::temp_dir().join(format!(
            "zo-usage-summary-{}-{}.jsonl",
            std::process::id(),
            current_unix_millis()
        ));
        let body = r#"{"type":"session_meta","version":1,"session_id":"session-a","created_at_ms":10,"updated_at_ms":20}
{"type":"message","turn_index":0,"message":{"role":"assistant","blocks":[],"usage":{"input_tokens":7,"output_tokens":3,"cache_creation_input_tokens":2,"cache_read_input_tokens":1}}}
{"type":"message","turn_index":1,"message":{"role":"user","blocks":[]}}
"#;
        std::fs::write(&path, body).expect("write jsonl fixture");
        let summary = load_usage_summary(&path).expect("summary should load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(summary.session_id, "session-a");
        assert_eq!(summary.updated_at_ms, 20);
        assert_eq!(summary.usage.input_tokens, 7);
        assert_eq!(summary.usage.output_tokens, 3);
        assert_eq!(summary.usage.cache_creation_input_tokens, 2);
        assert_eq!(summary.usage.cache_read_input_tokens, 1);
    }
}
