//! Toggle + informational commands: theme, plan, tasks, focus and
//! the family of small preference toggles / status replies (feedback,
//! fast, brief, advisor, upgrade, output-style, desktop, insights,
//! thinkback, keybindings, privacy-settings, share, files).
//!
//! Kept together because each is a thin formatter or a single view
//! toggle — grouping them keeps the actionable `system` module focused.
//! (`share` is the one exception that does real work: it writes a local
//! transcript artifact, mirroring the non-TUI `LiveCli::Share` path.)

use zo_cli::tui::modals::ToolToggleRow;
use zo_cli::tui::theme::Theme as TuiTheme;

use super::context::DispatchCtx;
use super::output::CommandOutput;
use crate::session::smart_settings::{
    build_smart_settings_modal, execute_deep_tier_command, execute_smart_text_command,
};

pub(super) fn theme(ctx: &mut DispatchCtx, name: Option<&str>) -> CommandOutput {
    match name {
        None => {
            // No argument → open the selection modal (mirrors `/model`). The
            // chosen entry is re-submitted as `/theme <name>`, landing back in
            // the `Some(..)` arm below — one apply path, no duplicated logic.
            let names = TuiTheme::builtin_names()
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            ctx.app.open_arg_picker("theme", "/theme", names);
            CommandOutput::Quiet
        }
        Some(requested) => {
            // `default` is a friendly alias for the flagship Zo theme.
            let key = if requested == "default" {
                "zo"
            } else {
                requested
            };
            match TuiTheme::builtin(key) {
                Some(theme) => {
                    let applied = theme.name.clone();
                    // Apply the terminal's color policy (NO_COLOR / truecolor /
                    // ANSI-256 / ANSI-16) to what we render this session, while
                    // persistence below still writes the *raw* built-in palette
                    // so a NO_COLOR/neutral display never gets baked into the
                    // on-disk tokens file.
                    ctx.app.set_theme(theme.for_current_terminal());
                    let persisted = persist_theme_palette(key);
                    let footer = if persisted {
                        "  Persisted        .zo/design/tokens.json (applies on next launch)"
                    } else {
                        "  Note             applied for this session; persistence skipped"
                    };
                    CommandOutput::info(format!("Theme\n  Switched to      {applied}\n{footer}"))
                }
                None => CommandOutput::error(format!(
                    "Theme\n  Unknown theme    \"{requested}\"\n  Available        {}",
                    TuiTheme::builtin_names().join(", "),
                )),
            }
        }
    }
}

/// Persist the selected built-in theme's color palette into
/// `.zo/design/tokens.json` so the choice survives a relaunch (the
/// boot path reads that file via `Theme::load`). Only the `color`
/// section is rewritten; spacing / breakpoint / `border_usage` are left
/// untouched. Returns `false` (best-effort) if the file is missing or
/// cannot be written — the in-session `set_theme` still applies.
fn persist_theme_palette(name: &str) -> bool {
    use std::fs;

    let Some(theme) = TuiTheme::builtin(name) else {
        return false;
    };
    let Ok(cwd) = crate::current_cli_cwd() else {
        return false;
    };
    let path = cwd.join(".zo").join("design").join("tokens.json");

    // Merge into the existing document so we never drop the spacing,
    // breakpoint, or border sections.
    let mut doc: serde_json::Value = match fs::read_to_string(&path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({})),
        Err(_) => return false,
    };
    let p = &theme.palette;
    let entry = |c| serde_json::json!({ "hex": color_to_hex(c) });
    doc["color"] = serde_json::json!({
        "primary": {
            "accent": entry(p.accent),
            "accent_dim": entry(p.accent_dim),
        },
        "secondary": {
            "cyan": entry(p.cyan),
            "violet": entry(p.violet),
            "teal": entry(p.teal),
        },
        "neutral": {
            "fg": entry(p.fg),
            "bright": entry(p.bright),
            "dim": entry(p.dim),
            "muted": entry(p.muted),
            "faint": entry(p.faint),
            "code_bg": entry(p.code_bg),
        },
        "semantic": {
            "success": entry(p.success),
            "warn": entry(p.warn),
            "error": entry(p.error),
            "info": entry(p.info),
        },
    });

    let Ok(serialized) = serde_json::to_string_pretty(&doc) else {
        return false;
    };
    fs::write(&path, serialized).is_ok()
}

/// Render a `ratatui::style::Color` as a `#RRGGBB` hex string the tokens
/// loader can re-parse. Indexed/reset colors round-trip through the
/// ANSI-256 → RGB table so every built-in palette persists losslessly.
fn color_to_hex(color: ratatui::style::Color) -> String {
    use ratatui::style::Color;
    let (r, g, b) = match color {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Indexed(i) => ansi256_to_rgb(i),
        // Reset / named colors have no fixed RGB; anchor to neutral grey
        // so the written file stays well-formed and re-parseable.
        _ => (0x80, 0x80, 0x80),
    };
    format!("#{r:02X}{g:02X}{b:02X}")
}

/// Convert an ANSI-256 index to its canonical RGB triple (the xterm
/// 256-color cube + greyscale ramp + 16 base colors).
fn ansi256_to_rgb(idx: u8) -> (u8, u8, u8) {
    match idx {
        // 16 base ANSI colors (xterm defaults).
        0 => (0x00, 0x00, 0x00),
        1 => (0x80, 0x00, 0x00),
        2 => (0x00, 0x80, 0x00),
        3 => (0x80, 0x80, 0x00),
        4 => (0x00, 0x00, 0x80),
        5 => (0x80, 0x00, 0x80),
        6 => (0x00, 0x80, 0x80),
        7 => (0xC0, 0xC0, 0xC0),
        8 => (0x80, 0x80, 0x80),
        9 => (0xFF, 0x00, 0x00),
        10 => (0x00, 0xFF, 0x00),
        11 => (0xFF, 0xFF, 0x00),
        12 => (0x00, 0x00, 0xFF),
        13 => (0xFF, 0x00, 0xFF),
        14 => (0x00, 0xFF, 0xFF),
        15 => (0xFF, 0xFF, 0xFF),
        // 6×6×6 color cube (indices 16..=231).
        16..=231 => {
            let cube = idx - 16;
            let steps = [0u8, 95, 135, 175, 215, 255];
            let red = steps[(cube / 36) as usize];
            let green = steps[((cube % 36) / 6) as usize];
            let blue = steps[(cube % 6) as usize];
            (red, green, blue)
        }
        // Greyscale ramp (indices 232..=255).
        232..=255 => {
            let level = 8 + (idx - 232) * 10;
            (level, level, level)
        }
    }
}


/// `/auto [check-cmd|on|off]` — toggle reactive auto-verify, the interactive
/// default. After a turn that edits files, the harness auto-verifies the diff
/// (objective green command if known + an adversarial verifier) and retries on
/// failure, one-shot. No read-only phase, so no permission friction; chat and
/// analysis turns pass straight through. See
/// [`runtime::ConversationRuntime::run_auto_turn_streaming`].
pub(super) fn auto(ctx: &mut DispatchCtx, arg: Option<&str>) -> CommandOutput {
    let (config, message) = crate::auto_gate_directive(arg);
    ctx.cli.runtime.set_deep_gate(config);
    CommandOutput::info(message)
}

/// `/deep [check-cmd|off]` — toggle the deep-lane gate on the live session.
/// With a check command the objective gate is `<cmd>` exiting 0; bare `/deep`
/// is verifier-only; `/deep off` returns to the ordinary single-pass turn. The
/// gate itself (plan → implement → verify → retry, with a read-only planning and
/// verification phase) lives in
/// [`runtime::ConversationRuntime::run_deep_turn_streaming`]; here we just
/// install or clear its config so the next turn routes through it.
pub(super) fn deep(ctx: &mut DispatchCtx, arg: Option<&str>) -> CommandOutput {
    let (config, message) = crate::deep_gate_directive(arg);
    ctx.cli.runtime.set_deep_gate(config);
    CommandOutput::info(message)
}

pub(super) fn tasks(ctx: &DispatchCtx, args: Option<&str>) -> CommandOutput {
    // Real session task list (CC `/todos` parity): TaskCreate/Agent tasks and
    // background bash runs all live in the same registry.
    let registry = ctx
        .cli
        .runtime
        .api_client()
        .tool_registry()
        .context()
        .tasks
        .clone();
    let status_filter = args.and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
        "running" => Some(runtime::task_registry::TaskStatus::Running),
        "completed" | "done" => Some(runtime::task_registry::TaskStatus::Completed),
        "failed" => Some(runtime::task_registry::TaskStatus::Failed),
        "stopped" => Some(runtime::task_registry::TaskStatus::Stopped),
        _ => None,
    });
    let tasks = registry.list(status_filter);
    if tasks.is_empty() {
        return CommandOutput::popup(
            "/tasks",
            "Tasks\n  (none)\n  Tip              background bash (`run_in_background`) and TaskCreate register here",
        );
    }
    let mut lines = vec![format!("Tasks ({})", tasks.len())];
    for task in tasks {
        let prompt = task.prompt.lines().next().unwrap_or("");
        let prompt: String = prompt.chars().take(64).collect();
        lines.push(format!(
            "  {:<22} {:<10} {prompt}",
            task.task_id,
            task.status.to_string()
        ));
    }
    lines.push(
        "  Tip              TaskOutput(task_id) → output · TaskStop(task_id) → stop".to_string(),
    );
    CommandOutput::popup("/tasks", lines.join("\n"))
}

pub(super) fn focus(ctx: &mut DispatchCtx) -> CommandOutput {
    ctx.app.toggle_focus_mode();
    CommandOutput::info("Focus mode toggled. Press Esc or F11 to exit.")
}

pub(super) fn tools(ctx: &mut DispatchCtx) -> CommandOutput {
    let registry = ctx.cli.runtime.api_client().tool_registry();
    let rows = registry
        .toggleable_tools()
        .into_iter()
        .map(|tool| ToolToggleRow {
            name: tool.name,
            description: tool.description,
            source: tool.source.label().to_string(),
            enabled: tool.enabled,
        })
        .collect();
    ctx.app.open_tool_toggle_modal(rows);
    CommandOutput::Quiet
}

pub(super) fn feedback() -> CommandOutput {
    CommandOutput::popup(
        "/feedback",
        "Feedback\n  Report issues     https://github.com/anthropics/claude-code/issues\n  Command           /feedback",
    )
}

/// `/fast [on|off]` — toggle gpt-5.5's priority ("fast") service tier. The
/// state lives on the model (gpt-5.5 vs gpt-5.5-fast); the real logic, including
/// its orthogonality with reasoning effort and the `service_tier: "priority"`
/// wire mapping, lives in `LiveCli::toggle_fast`.
pub(super) fn fast(ctx: &mut DispatchCtx, mode: Option<&str>) -> CommandOutput {
    if mode.is_none() {
        // Bare `/fast` opens the on/off picker; `/fast status` still prints the
        // current serving tier for a non-modal read-out.
        ctx.app
            .open_arg_picker("fast", "/fast", vec!["on".to_string(), "off".to_string()]);
        return CommandOutput::Quiet;
    }
    CommandOutput::info(ctx.cli.toggle_fast(mode))
}

/// `/brief` — genuinely toggle between the `concise` and `markdown` output
/// styles (the same mechanism as `/output-style`, so persist + system-prompt
/// reload happen in one place). Mirrors the REPL `/brief` arm.
pub(super) fn brief(ctx: &mut DispatchCtx) -> CommandOutput {
    let current = tools::execute_config(tools::ConfigInput {
        setting: "outputStyle".to_string(),
        value: None,
    })
    .ok()
    .and_then(|output| {
        crate::session::live_cli_commands::config_string_value(&output).unwrap_or(None)
    })
    .unwrap_or_else(|| "markdown".to_string());
    let next = if current == "concise" {
        "markdown"
    } else {
        "concise"
    };
    output_style(ctx, Some(next))
}

/// `/advisor` — persist the `advisorModeEnabled` flag toggle (the same config
/// write the REPL arm performs); honest about being a preference for future
/// sessions rather than an immediate behavior change.
pub(super) fn advisor() -> CommandOutput {
    let current = tools::execute_config(tools::ConfigInput {
        setting: "advisorModeEnabled".to_string(),
        value: None,
    })
    .ok()
    .and_then(|output| {
        crate::session::live_cli_commands::config_bool_value(&output).unwrap_or(None)
    })
    .unwrap_or(false);
    let next = !current;
    if let Err(error) = tools::execute_config(tools::ConfigInput {
        setting: "advisorModeEnabled".to_string(),
        value: Some(tools::ConfigValue::Bool(next)),
    }) {
        return CommandOutput::error(format!("Failed to persist advisor mode: {error}"));
    }
    CommandOutput::info(format!(
        "Advisor mode {}\n  Effect           prompts guidance-oriented behavior for future sessions that honor this setting",
        if next { "enabled" } else { "disabled" }
    ))
}

pub(super) fn upgrade() -> CommandOutput {
    CommandOutput::popup(
        "/upgrade",
        format!(
            "Upgrade\n  Version          {}\n  Tip              rebuild from source: cd rust && cargo build --release",
            env!("CARGO_PKG_VERSION")
        ),
    )
}

pub(super) fn output_style(ctx: &mut DispatchCtx, style: Option<&str>) -> CommandOutput {
    let cwd = ctx.cli.cwd.clone();
    let Some(style) = style else {
        // Bare `/output-style` opens the picker over the *real* styles
        // (default + built-ins + `.zo/output-styles/*.md`); the choice
        // re-enters as `/output-style <name>`.
        let styles = runtime::output_style::list(&cwd)
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        ctx.app
            .open_arg_picker("output-style", "/output-style", styles);
        return CommandOutput::Quiet;
    };

    let style = style.trim();
    let resolved = runtime::output_style::resolve(&cwd, style);
    let is_default = style.eq_ignore_ascii_case(runtime::output_style::DEFAULT_STYLE);
    if resolved.is_none() && !is_default {
        let available = runtime::output_style::list(&cwd)
            .into_iter()
            .map(|entry| entry.name)
            .collect::<Vec<_>>()
            .join(", ");
        return CommandOutput::error(format!(
            "Unknown output style '{style}'\n  Available        {available}\n  Custom           .zo/output-styles/<name>.md"
        ));
    }

    // Persist (settings.local.json, Claude Code parity), then rebuild the live
    // system prompt so the style takes effect on the *next* turn.
    let persist = tools::execute_config(tools::ConfigInput {
        setting: "outputStyle".to_string(),
        value: Some(tools::ConfigValue::String(style.to_string())),
    });
    if let Err(error) = persist {
        return CommandOutput::error(format!("Failed to persist output style: {error}"));
    }
    if let Err(error) = ctx.cli.reload_context() {
        return CommandOutput::error(format!(
            "Output style saved, but reloading the system prompt failed: {error}"
        ));
    }
    let label = resolved.map_or_else(|| "default (stock prompt)".to_string(), |(name, _)| name);
    CommandOutput::info(format!(
        "Output style\n  Style            {label}\n  Saved to         .zo/settings.local.json\n  Applies          next turn (system prompt reloaded)"
    ))
}

/// `/desktop` — reveal the session file in the OS file manager (the same
/// `open`/`xdg-open` call the REPL arm makes).
pub(super) fn desktop(ctx: &mut DispatchCtx) -> CommandOutput {
    let path = ctx.cli.session.path.clone();
    match crate::session::live_cli_commands::open_in_desktop(&path) {
        Ok(()) => CommandOutput::info(format!(
            "Desktop open requested\n  Session file     {}",
            path.display()
        )),
        Err(error) => CommandOutput::error(format!("Desktop open failed: {error}")),
    }
}

/// `/insights` — live session analytics (turns, messages, token split),
/// mirroring the REPL arm as a popup report.
pub(super) fn insights(ctx: &mut DispatchCtx) -> CommandOutput {
    let usage = ctx.cli.runtime.usage().cumulative_usage();
    CommandOutput::popup(
        "/insights",
        format!(
            "Insights\n  Session ID       {}\n  Messages         {}\n  Turns            {}\n  Total tokens     {}\n  Input tokens     {}\n  Output tokens    {}",
            ctx.cli.session.id,
            ctx.cli.runtime.session().messages.len(),
            ctx.cli.runtime.usage().turns(),
            usage.total_tokens(),
            usage.input_tokens,
            usage.output_tokens
        ),
    )
}

/// `/thinkback` — replay the most recent assistant activity trail (tool calls
/// + reasoning summaries), mirroring the REPL arm as a popup report.
pub(super) fn thinkback(ctx: &mut DispatchCtx) -> CommandOutput {
    let lines = crate::session::live_cli_commands::last_thinkback_lines(ctx.cli.runtime.session());
    if lines.is_empty() {
        return CommandOutput::popup(
            "/thinkback",
            "Thinkback\n  Result           no prior assistant activity captured yet",
        );
    }
    CommandOutput::popup(
        "/thinkback",
        format!("Thinkback\n  Recent flow      {}", lines.join("\n  ")),
    )
}

pub(super) fn keybindings() -> CommandOutput {
    CommandOutput::popup(
        "/keybindings",
        "Keybindings\n  Enter            submit\n  Shift+Tab        cycle permission mode\n  Esc              cancel\n  Up/Down          scroll transcript",
    )
}

pub(super) fn privacy_settings() -> CommandOutput {
    CommandOutput::popup(
        "/privacy-settings",
        "Privacy Settings\n  Telemetry        off (local build)\n  Sessions         stored locally in .zo/sessions/\n  Sharing          /share is local-only; /share gist uploads a redacted, SECRET (unlisted ≠ private) gist you can revoke with /unshare\n  Tip              configure via settings.json",
    )
}

/// Write the current transcript to a local share artifact and, when `target`
/// is `Some("gist")`, additionally upload it to a secret GitHub gist.
///
/// A bare `/share` (`target == None`) is byte-identical to before: it writes
/// `.zo/share/<session-id>.txt` via `render_export_text` and copies the
/// *path* to the clipboard. `/share gist` keeps that local artifact (so the
/// command never hard-fails), prints a loud secret warning, then uploads a
/// redacted copy and copies the resulting *URL* instead.
pub(super) fn share(ctx: &mut DispatchCtx, target: Option<&str>) -> CommandOutput {
    let session_id = ctx.cli.session.id.clone();
    let artifact =
        match crate::session::write_share_artifact(&session_id, ctx.cli.runtime.session()) {
            Ok(artifact) => artifact,
            Err(e) => return CommandOutput::error(e.to_string()),
        };

    let path_display = artifact.path.display().to_string();
    let messages = ctx.cli.runtime.session().messages.len();

    if target == Some("gist") {
        return share_gist(ctx, &artifact, messages);
    }

    // Local-only: copy the artifact path so the user can paste it straight into
    // a message or terminal. Silently degrades when no clipboard sink exists —
    // the printed path is the real result either way.
    let clipboard_line = match crate::session::write_to_clipboard(&path_display) {
        Ok(sink) => format!("\n  Clipboard        path copied {}", sink.describe()),
        Err(_) => String::new(),
    };
    CommandOutput::info(format!(
        "Share\n  Result           wrote local share artifact\n  File             {path_display}\n  Messages         {messages}\n  Characters       {}{clipboard_line}",
        artifact.char_count
    ))
}

/// `/share gist`: warn, upload a redacted copy to a secret gist, copy the URL.
/// The local artifact written by the caller stays put, so an upload failure
/// degrades gracefully rather than losing the user's transcript.
fn share_gist(
    ctx: &mut DispatchCtx,
    artifact: &crate::session::ShareArtifact,
    messages: usize,
) -> CommandOutput {
    let warning = crate::session::share_gist_warning(artifact.char_count);
    let url = match crate::session::upload_share_to_gist(
        &ctx.cli.session.id,
        ctx.cli.runtime.session(),
    ) {
        Ok(url) => url,
        Err(e) => {
            return CommandOutput::error(format!(
                "{warning}\n\nShare gist\n  Error            {e}\n  Fallback         local artifact still at {}",
                artifact.path.display()
            ));
        }
    };
    let gist_id = url.rsplit('/').next().unwrap_or("<id>");
    let clipboard_line = match crate::session::write_to_clipboard(&url) {
        Ok(sink) => format!("\n  Clipboard        URL copied {}", sink.describe()),
        Err(_) => String::new(),
    };
    CommandOutput::info(format!(
        "{warning}\n\nShare gist\n  URL              {url}\n  Messages         {messages}\n  Revoke           /unshare {gist_id}{clipboard_line}"
    ))
}

/// `/unshare <id>`: delete a previously created share gist (the revoke handle).
pub(super) fn unshare(_ctx: &mut DispatchCtx, id: Option<&str>) -> CommandOutput {
    let Some(id) = id.map(str::trim).filter(|id| !id.is_empty()) else {
        return CommandOutput::error("Unshare\n  Usage            /unshare <gist-id>");
    };
    match crate::session::delete_share_gist(id) {
        Ok(()) => CommandOutput::info(format!("Unshare\n  Gist             {id} deleted")),
        Err(e) => CommandOutput::error(format!("Unshare\n  Error            {e}")),
    }
}

pub(super) fn files() -> CommandOutput {
    let cwd = std::env::current_dir().unwrap_or_default();
    CommandOutput::popup(
        "/files",
        format!(
            "Files\n  Working dir      {}\n  Tip              shows files in the current context",
            cwd.display()
        ),
    )
}

pub(super) fn smart(ctx: &mut DispatchCtx<'_>, arg: Option<&str>) -> CommandOutput {
    let action = arg.unwrap_or("").trim();

    if action.is_empty() {
        let options = vec![
            "Open settings dashboard".to_string(),
            "Turn Smart ON".to_string(),
            "Turn Smart OFF".to_string(),
            "Pin a role/subagent".to_string(),
            "Reset a role/subagent (Auto)".to_string(),
            "Reset all overrides".to_string(),
            "Show current status".to_string(),
        ];
        ctx.app.open_arg_picker("smart", "/smart Menu", options);
        return CommandOutput::Quiet;
    }

    let action_lower = action.to_ascii_lowercase();
    match action_lower.as_str() {
        "open settings dashboard" | "dashboard" => {
            return run_smart_gui_step(ctx);
        }
        "turn smart on" => {
            match execute_smart_text_command(&ctx.cli.model, Some("on")) {
                Ok(message) => return CommandOutput::info(message),
                Err(error) => return CommandOutput::error(error),
            }
        }
        "turn smart off" => {
            match execute_smart_text_command(&ctx.cli.model, Some("off")) {
                Ok(message) => return CommandOutput::info(message),
                Err(error) => return CommandOutput::error(error),
            }
        }
        "pin a role/subagent" => {
            let targets = all_smart_targets();
            let options = targets.iter().map(|t| format!("pin {t}")).collect();
            ctx.app.open_arg_picker("smart", "Select target to pin", options);
            return CommandOutput::Quiet;
        }
        "reset a role/subagent (auto)" => {
            let targets = all_smart_targets();
            let options = targets.iter().map(|t| format!("auto {t}")).collect();
            ctx.app.open_arg_picker("smart", "Select target to reset to Auto", options);
            return CommandOutput::Quiet;
        }
        "reset all overrides" => {
            match execute_smart_text_command(&ctx.cli.model, Some("reset")) {
                Ok(message) => return CommandOutput::info(message),
                Err(error) => return CommandOutput::error(error),
            }
        }
        "show current status" => {
            match execute_smart_text_command(&ctx.cli.model, Some("status")) {
                Ok(message) => return CommandOutput::info(message),
                Err(error) => return CommandOutput::error(error),
            }
        }
        _ => {}
    }

    let parts: Vec<&str> = action.split_whitespace().collect();
    if !parts.is_empty() {
        let subcommand = parts[0].to_ascii_lowercase();
        match subcommand.as_str() {
            "pin" => {
                if parts.len() == 1 {
                    let targets = all_smart_targets();
                    let options = targets.iter().map(|t| format!("pin {t}")).collect();
                    ctx.app.open_arg_picker("smart", "Select target to pin", options);
                    return CommandOutput::Quiet;
                } else if parts.len() == 2 {
                    let target = parts[1];
                    let inventory = runtime::connected_model_inventory(&ctx.cli.model);
                    let options = inventory
                        .models()
                        .iter()
                        .map(|m| format!("pin {target} {}", m.id()))
                        .collect();
                    ctx.app.open_arg_picker("smart", format!("Select model for {target}"), options);
                    return CommandOutput::Quiet;
                }
            }
            "auto" | "unpin" => {
                if parts.len() == 1 {
                    let targets = all_smart_targets();
                    let options = targets.iter().map(|t| format!("auto {t}")).collect();
                    ctx.app.open_arg_picker("smart", "Select target to reset to Auto", options);
                    return CommandOutput::Quiet;
                }
            }
            _ => {}
        }
    }

    match execute_smart_text_command(&ctx.cli.model, Some(action)) {
        Ok(message) => CommandOutput::info(message),
        Err(error) => CommandOutput::error(error),
    }
}

pub(super) fn deep_tier(
    ctx: &mut DispatchCtx<'_>,
    action: &commands::DeepTierAction,
) -> CommandOutput {
    if matches!(action, commands::DeepTierAction::Show) {
        let Some(setting) = tools::smart_deep_tier_models_for(&ctx.cli.cwd) else {
            return CommandOutput::error("Deep-tier pool: could not load merged settings");
        };
        ctx.app.open_deep_tier_modal(zo_cli::tui::modals::DeepTierView {
            models: setting.models,
            configured: setting.configured,
        });
        return CommandOutput::Quiet;
    }
    match execute_deep_tier_command(&ctx.cli.cwd, action) {
        Ok(message) => CommandOutput::info(message),
        Err(error) => CommandOutput::error(error),
    }
}

fn all_smart_targets() -> Vec<String> {
    vec![
        // Roles
        "default".to_string(),
        "fast".to_string(),
        "coding".to_string(),
        "debugging".to_string(),
        "verifier".to_string(),
        "reviewer".to_string(),
        "analysis".to_string(),
        "research".to_string(),
        "writing".to_string(),
        "design".to_string(),
        "judge".to_string(),
        "synthesizer".to_string(),
        // Subagents
        "general-purpose".to_string(),
        "Explore".to_string(),
        "Plan".to_string(),
        "Verification".to_string(),
        "deep-research".to_string(),
        "code-reviewer".to_string(),
        "debugger".to_string(),
        "data-analyst".to_string(),
        "refactor".to_string(),
        "frontend-design".to_string(),
        "zo-guide".to_string(),
        "statusline-setup".to_string(),
    ]
}

fn run_smart_gui_step(ctx: &mut DispatchCtx<'_>) -> CommandOutput {
    match build_smart_settings_modal(&ctx.cli.model, &ctx.cli.cwd, &ctx.cli.session.id) {
        Ok(modal) => {
            ctx.app.open_smart_settings_modal(modal);
            CommandOutput::Quiet
        }
        Err(error) => CommandOutput::error(error.to_string()),
    }
}

