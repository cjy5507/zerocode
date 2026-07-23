mod auth;
mod context;
mod git_cmds;
mod handlers;
mod helpers_tui;
mod output;
mod session_cmds;
mod ship;
mod system;
mod toggles;
mod view;

pub(crate) use auth::{
    ConnectReport, ProviderTokenLimits, connect_custom_provider, connect_preset, connect_preset_with_api_key,
};
pub(crate) use helpers_tui::{add_model_to_history, prompt_command_usage};
#[cfg(test)]
pub(super) use helpers_tui::{persistent_tui_candidate_supported, render_persistent_tui_help};
pub(super) use helpers_tui::{push_report, seed_transcript_from_session};
// Re-exported so the local `/commit-push-pr` slash path can call it via `git_cmds`.
pub(crate) use handlers::handle_commit_push_pr_at;
pub(crate) use ship::handle_ship_at;

use commands::SlashCommand;
use runtime::message_stream::BlockIdGen;

use self::context::{DispatchCtx, DispatchError};
use self::output::{CommandOutput, render};
use super::LiveCli;
use super::tui_loop::TuiLoopError;
use crate::cli_args::format_unknown_slash_command;
use zo_cli::tui::App;
use zo_cli::tui::modals::{Effort, effort_level_label};

/// Persistent-TUI slash entry point.
///
/// Builds the [`DispatchCtx`] once, routes the command to its group
/// handler (which returns a typed [`CommandOutput`]), then projects that
/// output onto the transcript through the single [`render`] funnel.
/// Returns `true` when the command requested process exit.
pub(super) fn handle_persistent_slash(
    cli: &mut LiveCli,
    app: &mut App,
    ids: &BlockIdGen,
    command: SlashCommand,
) -> Result<bool, TuiLoopError> {
    let mut ctx = DispatchCtx { cli, app, ids };
    let output = dispatch(&mut ctx, command)?;
    Ok(render(ctx.app, ctx.ids, output))
}

/// Route one parsed [`SlashCommand`] to its group handler.
///
/// Each arm is a single delegation; the command's behaviour lives in the
/// group module (`view`, `session_cmds`, `git_cmds`, `auth`, `system`,
/// `toggles`). Fallible handlers propagate [`DispatchError`] via `?`.
#[allow(clippy::too_many_lines)] // flat slash-command to handler dispatch
fn dispatch(ctx: &mut DispatchCtx, command: SlashCommand) -> Result<CommandOutput, DispatchError> {
    use SlashCommand as C;
    Ok(match command {
        // ── view (read-only reports / cards) ──────────────────────────
        C::Help => view::help(ctx),
        C::Status => view::status(ctx),
        C::Cost => view::cost(ctx),
        C::Config { section } => view::config(section.as_deref())?,
        C::Memory => system::edit_memory(ctx),
        C::Dream => system::dream(ctx),
        C::Diff => view::diff(ctx)?,
        C::Agents { args } => view::agents(ctx, args.as_deref())?,
        C::Inbox { args } => view::inbox(ctx, args.as_deref()),
        C::Skills { args } => view::skills(args.as_deref())?,
        C::Mcp { action, target } => view::mcp(ctx, action.as_deref(), target.as_deref())?,
        C::Tools => toggles::tools(ctx),
        C::Version => view::version(),
        C::Doctor => view::doctor(ctx),
        C::Audit => view::audit(ctx),
        C::Usage { scope } => view::usage(ctx, scope.as_deref()),
        C::Cache => view::cache(ctx),
        C::Context { action } => view::context(ctx, action.as_deref()),
        C::Plugins { action, target } => view::plugins(ctx, action.as_deref(), target.as_deref()),

        // ── session lifecycle ─────────────────────────────────────────
        C::New => session_cmds::new(ctx)?,
        C::Clear { confirm } => session_cmds::clear(ctx, confirm)?,
        C::Compact { instructions } => session_cmds::compact(ctx, instructions)?,
        C::Resume { session_path } => session_cmds::resume(ctx, session_path.as_deref())?,
        C::Session { action, target } => {
            session_cmds::session(ctx, action.as_deref(), target.as_deref())?
        }
        C::Name { name } => session_cmds::name(ctx, name.as_deref()),
        C::Fork { name } => session_cmds::fork(ctx, name.as_deref())?,
        C::Rename { name } => session_cmds::rename(ctx, name.as_deref()),
        C::Rewind { action } => session_cmds::rewind(ctx, &action),
        C::Undo { steps } => session_cmds::undo(ctx, steps.as_deref()),
        C::Redo { steps } => session_cmds::redo(ctx, steps.as_deref()),

        // ── git / VCS ─────────────────────────────────────────────────
        C::Commit => git_cmds::commit(),
        C::Ship { message } => git_cmds::ship(&ctx.cli.cwd, &message),
        C::Pr { context } => git_cmds::pr(ctx, context.as_deref()),
        C::Issue { context } => git_cmds::issue(ctx, context.as_deref()),
        C::PrComments { pr_number } => git_cmds::pr_comments(pr_number.as_deref()),
        C::CommitPushPr => git_cmds::commit_push_pr(&ctx.cli.cwd),
        C::BackfillSessions => git_cmds::backfill_sessions(),
        C::Branch { name } => git_cmds::branch(name.as_deref()),

        // ── auth / providers ──────────────────────────────────────────
        C::Connect { provider } => auth::connect(ctx, provider.as_deref()),
        C::Login { provider } => auth::login(ctx, provider.as_deref()),
        C::Logout => auth::logout(),

        // ── actionable system commands ────────────────────────────────
        C::Model { model } => system::model(ctx, model.as_deref())?,
        C::Permissions { mode } => system::permissions(ctx, mode.as_deref())?,
        C::Init => system::init(ctx),
        C::Restart => system::restart(ctx)?,
        C::Export { path } => system::export(ctx, path.as_deref()),
        C::Dump { edit } => system::dump(ctx, edit),
        C::Copy { target } => system::copy(ctx, target.as_deref()),
        C::Hooks { args } => system::hooks(args.as_deref()),
        C::ReloadContext => system::reload_context(ctx)?,
        C::Ide { target } => system::ide(ctx, target.as_deref()),
        C::AddDir { path } => system::add_dir(path.as_deref()),
        C::Teleport { target } => system::teleport(target.as_deref()),
        C::DebugToolCall => system::debug_tool_call(ctx),
        C::Review { scope } => system::review(ctx, scope.as_deref())?,
        C::Hunks => system::hunks(ctx),
        C::Bughunter { scope } => system::bughunter(ctx, scope.as_deref()),
        C::Ultraplan { task } => system::ultraplan(ctx, task.as_deref()),
        C::Council { task } => system::council(ctx, task.as_deref()),
        C::Distill { topic } => system::distill(ctx, topic.as_deref()),
        C::SelfImprove { action } => system::self_improve(ctx, &action),
        C::Remote { .. } => CommandOutput::error(
            "/remote is available only inside the live interactive TUI session.",
        ),
        C::Goal { command } => system::goal(ctx, command),
        C::Loop { command } => system::loop_cmd(ctx, command),
        C::SecurityReview => system::security_review(ctx),
        C::ReleaseNotes => system::release_notes(),
        C::ExtraUsage => system::extra_usage(ctx),
        C::PerfIssue => system::perf_issue(ctx),
        C::Statusline => system::statusline(),
        C::AntTrace => system::ant_trace(),

        // ── toggles + informational ───────────────────────────────────
        C::Effort { level } => handle_effort_command(ctx, level.as_deref()),
        C::Theme { name } => toggles::theme(ctx, name.as_deref()),
        C::Plan { mode } => session_cmds::plan(ctx, mode.as_deref())?,
        C::Deep { check } => toggles::deep(ctx, check.as_deref()),
        C::Auto { arg } => toggles::auto(ctx, arg.as_deref()),
        C::Tasks { args } => toggles::tasks(ctx, args.as_deref()),
        C::Focus => toggles::focus(ctx),
        C::Feedback => toggles::feedback(),
        C::Fast { mode } => toggles::fast(ctx, mode.as_deref()),
        C::Smart { arg } => toggles::smart(ctx, arg.as_deref()),
        C::DeepTier { action } => toggles::deep_tier(ctx, &action),
        C::Brief => toggles::brief(ctx),
        C::Advisor => toggles::advisor(),
        C::Upgrade => toggles::upgrade(),
        C::OutputStyle { style } => toggles::output_style(ctx, style.as_deref()),
        C::Desktop => toggles::desktop(ctx),
        C::Insights => toggles::insights(ctx),
        C::Thinkback => toggles::thinkback(ctx),
        C::Keybindings => toggles::keybindings(),
        C::PrivacySettings => toggles::privacy_settings(),
        C::Share { target } => toggles::share(ctx, target.as_deref()),
        C::Unshare { id } => toggles::unshare(ctx, id.as_deref()),
        C::Files => toggles::files(),

        // ── control ───────────────────────────────────────────────────
        C::Exit => CommandOutput::Exit,
        C::Unknown { name, args } => dispatch_plugin_or_unknown(ctx, &name, args.as_deref())?,
    })
}

/// Resolve an otherwise-unknown slash command against the loaded plugin
/// registry: plugin-contributed commands execute their script (stdout becomes
/// the command output); anything else falls back to the not-found message.
fn dispatch_plugin_or_unknown(
    ctx: &mut DispatchCtx,
    name: &str,
    args: Option<&str>,
) -> Result<CommandOutput, DispatchError> {
    if let Some(command) = ctx.cli.runtime.prompt_command(name).cloned() {
        return dispatch_prompt_command(ctx, &command, args.unwrap_or_default());
    }

    if name.starts_with("mcp__") {
        if let Some(output) = dispatch_mcp_prompt(ctx, name, args) {
            return Ok(output);
        }
    }

    let registry = &ctx.cli.runtime.plugin_registry;
    if registry.find_slash_command(name).is_none() {
        return Ok(CommandOutput::error(format_unknown_slash_command(name)));
    }
    Ok(
        match registry.run_slash_command(name, args.unwrap_or_default()) {
            Ok(output) if output.trim().is_empty() => {
                CommandOutput::info(format!("/{name} completed."))
            }
            Ok(output) => CommandOutput::info(output),
            Err(error) => CommandOutput::error(error.to_string()),
        },
    )
}

/// Resolve a discovered MCP prompt (`/mcp__<server>__<prompt>`): call
/// `prompts/get` with the positional args mapped onto the prompt's declared
/// argument names, then queue the resolved text as the next user turn
/// (Claude Code parity). Returns `None` when no discovered prompt matches so
/// the name falls through to the plugin/unknown path unchanged.
fn dispatch_mcp_prompt(
    ctx: &mut DispatchCtx,
    name: &str,
    args: Option<&str>,
) -> Option<CommandOutput> {
    use super::mcp_runtime::{map_prompt_arguments, prompt_messages_to_text};

    let mcp_state = ctx.cli.runtime.mcp_state.clone()?;
    // try_lock: background discovery may hold the lock through a slow server
    // handshake; blocking here would freeze the input loop for that duration.
    let mut state = match mcp_state.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            return Some(CommandOutput::info(
                "MCP discovery is still in progress — retry in a moment.",
            ));
        }
    };
    let entry = state.find_prompt(name)?;

    let arguments = map_prompt_arguments(&entry.prompt.arguments, args);
    let resolved = state.get_prompt(&entry.server, &entry.prompt.name, arguments);
    drop(state);

    Some(match resolved {
        Ok(result) => {
            let text = prompt_messages_to_text(&result);
            if text.trim().is_empty() {
                return Some(CommandOutput::error(format!(
                    "MCP prompt `/{name}` returned no text content."
                )));
            }
            ctx.app.queue_message(text).map_or_else(
                |error| CommandOutput::error(error.to_string()),
                |()| {
                    CommandOutput::info(format!(
                        "MCP prompt\n  Command          /{name}\n  Server           {}\n  Prompt           {}\n  Status           queued as next user turn",
                        entry.server, entry.prompt.name
                    ))
                },
            )
        }
        Err(error) => CommandOutput::error(format!("MCP prompt `/{name}` failed: {error}")),
    })
}

fn dispatch_prompt_command(
    ctx: &mut DispatchCtx,
    command: &commands::PromptCommandDef,
    args: &str,
) -> Result<CommandOutput, DispatchError> {
    if let Err(error) = ctx.app.ensure_can_queue_message() {
        return Ok(CommandOutput::error(error.to_string()));
    }

    let mut applied = Vec::new();
    if let Some(model) = command.model.as_deref() {
        let report = ctx.cli.apply_model_change(model);
        ctx.cli.persist_session()?;
        applied.push(format!("model={}", report.lines().next().unwrap_or(model)));
    }
    if let Some(effort) = command.effort.as_deref() {
        if let Some(preset) = Effort::from_token(effort) {
            let warning = ctx.cli.set_effort(preset);
            applied.push(format!("effort={}", preset.canonical()));
            if let Some(warning) = warning {
                applied.push(warning);
            }
        } else if let Ok(custom) = effort.parse::<u32>() {
            let warning = ctx.cli.set_effort_budget(custom);
            applied.push(format!("effort={custom}"));
            if let Some(warning) = warning {
                applied.push(warning);
            }
        } else {
            return Ok(CommandOutput::error(format!(
                "Prompt command\n  Command          /{}\n  Invalid effort   \"{effort}\"\n  Source           {}",
                command.name,
                command.path.display()
            )));
        }
    }

    if !command.allowed_tools.is_empty() {
        // Restrict the offered tool set for THIS turn only, reusing the same
        // normalizer the CLI `--allowed-tools` path uses. The turn build site
        // prefers this over the session-global set; it is cleared back to `None`
        // at turn completion so the restriction never leaks to the next turn.
        match crate::cli_args::normalize_allowed_tools(&command.allowed_tools) {
            Ok(turn_allowed) => {
                ctx.cli.turn_allowed_tools = turn_allowed;
                applied.push("allowed-tools=scoped".to_string());
            }
            Err(error) => {
                return Ok(CommandOutput::error(format!(
                    "Prompt command\n  Command          /{}\n  Invalid tools    {error}\n  Source           {}",
                    command.name,
                    command.path.display()
                )));
            }
        }
    }

    if let Err(error) = ctx.app.queue_message(command.render_prompt(args)) {
        return Ok(CommandOutput::error(error.to_string()));
    }
    let metadata = if applied.is_empty() {
        "none".to_string()
    } else {
        applied.join(", ")
    };
    Ok(CommandOutput::info(format!(
        "Prompt command\n  Command          /{}\n  Source           {}\n  Metadata         {metadata}\n  Status           queued as next user turn",
        command.name,
        command.path.display()
    )))
}

/// Named effort presets for extended-thinking budget.
///
/// Derived from [`Effort`] so `/effort` argument parsing, the status
/// banner, the help footer, and the interactive slider all read from
/// one source of truth and can never drift apart.
const EFFORT_PRESETS: &[Effort] = Effort::ALL;

/// Resolve the preset whose budget matches `value`, or `None` for a
/// custom numeric argument.
fn preset_for_budget(value: u32) -> Option<Effort> {
    EFFORT_PRESETS.iter().copied().find(|p| p.budget() == value)
}

/// Build the levels table shown after the current-level line.
fn format_effort_levels() -> String {
    let mut lines = Vec::new();
    for preset in EFFORT_PRESETS {
        let aliases = preset.display_aliases();
        let alias_suffix = if aliases.is_empty() {
            String::new()
        } else {
            format!(" (alias: {})", aliases.join(", "))
        };
        let budget_text = if preset.budget() == 0 {
            "—".to_string()
        } else {
            format!("{} tokens", zo_cli::util::format_thousands(preset.budget()))
        };
        lines.push(format!(
            "                   · {:<10} {:<14}  {}{alias_suffix}",
            preset.canonical(),
            budget_text,
            preset.description()
        ));
    }
    lines.push(
        "                   · <number>   custom budget    set an arbitrary token limit".to_string(),
    );
    lines.join("\n")
}

/// Render the `/effort` status banner. Used by both the no-arg form
/// and after a successful level change so the visible state matches
/// what is in the runtime.
///
/// `model` is the active model id: when the selected preset's effort tier is
/// clamped down on this model (e.g. `xhigh`/`ultra` on Sonnet or Gemini),
/// the level line shows the *effective* tier — `ultra → high` — so the user
/// sees what they actually get instead of a silent downgrade. `smart` is a
/// DYNAMIC band rather than one static tier, so it instead shows the
/// per-model band range it resolves within (e.g. `smart → dynamic
/// xhigh~ultra`).
/// Provider is not hardcoded here: the effective tier comes from the
/// provider-neutral `api::effective_effort_for_model`.
fn render_effort_status(budget: Option<u32>, model: &str) -> String {
    let (level_text, budget_text) = match budget {
        None | Some(0) => ("off".to_string(), "—".to_string()),
        Some(b) => {
            let preset = preset_for_budget(b);
            let label = preset.map_or_else(|| "custom".to_string(), |p| p.canonical().to_string());
            // Smart carries a DYNAMIC band, not one static tier — show the
            // resolved per-model range instead of the single-tier clamp
            // check below, which would only ever see the band's floor.
            let band = preset.and_then(|p| p.band_labels_for_model(model));
            let level_text = if let Some((floor, ceiling)) = band {
                if floor == ceiling {
                    format!("{label} → {floor} (fixed on {model})")
                } else {
                    format!("{label} → dynamic {floor}~{ceiling} (escalates on heavy turns)")
                }
            } else {
                // Annotate with the effective tier when the model clamps the
                // selection. `Effort::level()` is the cli→api effort seam;
                // `Off` carries no wire effort so it never annotates.
                preset
                    .and_then(Effort::level)
                    .map(|requested| {
                        let effective = api::effective_effort_for_model(requested, model);
                        if effective == requested {
                            label.clone()
                        } else {
                            format!(
                                "{label} → {} ({model} has no {label})",
                                effort_level_label(effective)
                            )
                        }
                    })
                    .unwrap_or(label)
            };
            (level_text, format!("{} tokens", zo_cli::util::format_thousands(b)))
        }
    };
    format!(
        "Effort\n  Level            {level_text}\n  Budget           {budget_text}\n  Levels           {}",
        format_effort_levels().trim_start()
    )
}

/// Apply the requested effort level and produce a status banner.
///
/// Bare `/effort` opens the interactive slider (returning
/// [`CommandOutput::Quiet`]); every other form mutates the runtime
/// budget and reports the resulting state through the funnel.
fn handle_effort_command(ctx: &mut DispatchCtx, level: Option<&str>) -> CommandOutput {
    match level.map(str::trim) {
        None | Some("") => {
            // Bare `/effort` opens the interactive slider, pre-positioned
            // on the current budget. `/effort show` keeps the text banner
            // for users who want a non-modal read-out.
            ctx.app.open_effort_modal(ctx.cli.thinking_budget);
            CommandOutput::Quiet
        }
        Some("show") => CommandOutput::info(render_effort_status(
            ctx.cli.thinking_budget,
            &ctx.cli.model,
        )),
        Some(raw) => {
            if let Some(preset) = Effort::from_token(raw) {
                let warning = ctx.cli.set_effort(preset);
                return CommandOutput::info(effort_status_with_warning(
                    render_effort_status(ctx.cli.thinking_budget, &ctx.cli.model),
                    warning,
                ));
            }
            if let Ok(custom) = raw.parse::<u32>() {
                let warning = ctx.cli.set_effort_budget(custom);
                return CommandOutput::info(effort_status_with_warning(
                    render_effort_status(ctx.cli.thinking_budget, &ctx.cli.model),
                    warning,
                ));
            }
            CommandOutput::error(format!(
                "Effort\n  Invalid level    \"{raw}\"\n  Try one of       off · low · medium · high · xhigh · max · ultra · smart (aliases: smartcode, ultracode, uc)\n  Or pass a number /effort 16000"
            ))
        }
    }
}

/// Append an effort-preference persistence warning (returned by `set_effort` /
/// `set_effort_budget`) to the `/effort` status banner so a save failure is
/// visible instead of swallowed, while a successful change shows the status
/// alone. Kept as one place so both the preset and numeric arms surface the
/// warning identically.
fn effort_status_with_warning(status: String, warning: Option<String>) -> String {
    match warning {
        Some(warning) => format!("{status}\n  {warning}"),
        None => status,
    }
}

#[cfg(test)]
mod effort_tests {
    use super::{Effort, format_effort_levels, preset_for_budget, render_effort_status};

    #[test]
    fn smart_is_its_own_preset_and_ultracode_still_parses() {
        let smart = Effort::from_token("smart").expect("smart preset present");
        assert_eq!(smart, Effort::Smart);
        assert_eq!(smart.budget(), 28_000);
        // Every spelling except `ultra` (which is now its own static level,
        // not a Smart alias) resolves to Smart, including the pre-rename
        // (P9) `ultracode` token — persisted settings/scripts must keep
        // working.
        for alias in ["smart", "smartcode", "ultracode", "uc", "SMART", "ULTRACODE", "UC"] {
            assert_eq!(
                Effort::from_token(alias),
                Some(Effort::Smart),
                "expected `smart` to match {alias}"
            );
        }
        // `ultra` is its own static top-tier level now (second meaning change
        // for this token — see effort_picker.rs).
        assert_eq!(Effort::from_token("ultra"), Some(Effort::Ultra));
        assert_eq!(Effort::Ultra.budget(), 26_000);
        assert_eq!(Effort::from_token("max"), Some(Effort::Max));
        assert_eq!(Effort::Max.budget(), 24_000);
    }

    #[test]
    fn preset_for_budget_finds_canonical() {
        assert_eq!(preset_for_budget(0).map(Effort::canonical), Some("off"));
        assert_eq!(preset_for_budget(1_024).map(Effort::canonical), Some("low"));
        assert_eq!(
            preset_for_budget(16_000).map(Effort::canonical),
            Some("xhigh")
        );
        assert_eq!(
            preset_for_budget(24_000).map(Effort::canonical),
            Some("max")
        );
        assert_eq!(
            preset_for_budget(26_000).map(Effort::canonical),
            Some("ultra")
        );
        assert_eq!(
            preset_for_budget(28_000).map(Effort::canonical),
            Some("smart")
        );
        assert!(preset_for_budget(12_345).is_none());
    }

    #[test]
    fn render_effort_status_shows_ultracode_state() {
        // `opus` accepts the full scale, so max is shown verbatim (no arrow).
        let banner = render_effort_status(Some(24_000), "opus");
        assert!(
            banner.contains("max"),
            "banner should name preset:\n{banner}"
        );
        assert!(
            banner.contains("24 000 tokens"),
            "banner missing budget:\n{banner}"
        );
        assert!(
            banner.contains("Levels"),
            "banner missing levels table:\n{banner}"
        );
    }

    #[test]
    fn render_effort_status_labels_off_when_disabled() {
        let banner = render_effort_status(None, "opus");
        assert!(banner.contains("Level            off"), "banner: {banner}");
    }

    #[test]
    fn render_effort_status_annotates_effective_tier_when_model_clamps() {
        // Provider-neutral clamp surfacing: xhigh (16k preset) on Sonnet has no
        // xhigh tier, so the level line must show the effective fallback rather
        // than a silent downgrade. Opus, which accepts xhigh, shows no arrow.
        let sonnet = render_effort_status(Some(16_000), "claude-sonnet-4-6");
        assert!(
            sonnet.contains("xhigh → high"),
            "sonnet xhigh must annotate the effective tier:\n{sonnet}"
        );
        let opus = render_effort_status(Some(16_000), "claude-opus-4-8");
        assert!(
            opus.contains("Level            xhigh") && !opus.contains("→"),
            "opus accepts xhigh, so no arrow:\n{opus}"
        );
        // Generalizes across providers. GPT fast mode is a serving-priority
        // signal, NOT an effort ceiling — explicit/budget-derived xhigh stays
        // intact (api's unified `gpt_for_model` clamping), so no arrow.
        let gpt_fast = render_effort_status(Some(16_000), "gpt-5.5-fast");
        assert!(
            gpt_fast.contains("Level            xhigh") && !gpt_fast.contains("→"),
            "gpt-5.5-fast accepts xhigh (fast is serving priority, not a ceiling):\n{gpt_fast}"
        );
        // Ultra (26k) is a STATIC pin, projected exactly like Max/Xhigh above.
        let gemini_ultra = render_effort_status(Some(26_000), "gemini-3.5-flash");
        assert!(
            gemini_ultra.contains("ultra → high"),
            "gemini caps at high, ultra annotates:\n{gemini_ultra}"
        );
        let sol_ultra = render_effort_status(Some(26_000), "gpt-5.6-sol");
        assert!(
            sol_ultra.contains("Level            ultra") && !sol_ultra.contains("→"),
            "{sol_ultra}"
        );
        let luna_ultra = render_effort_status(Some(26_000), "gpt-5.6-luna");
        assert!(luna_ultra.contains("ultra → xhigh"), "{luna_ultra}");
    }

    #[test]
    fn render_effort_status_shows_smart_dynamic_band_per_model() {
        // Smart (28k) is a DYNAMIC band, not a single static tier — the
        // banner shows the resolved per-model range, not a one-shot clamp.
        let gemini = render_effort_status(Some(28_000), "gemini-3.5-flash");
        assert!(
            gemini.contains("smart → high (fixed on gemini-3.5-flash)"),
            "gemini's ceiling collapses the whole band onto high:\n{gemini}"
        );
        let sol = render_effort_status(Some(28_000), "gpt-5.6-sol");
        assert!(
            sol.contains("smart → dynamic xhigh~ultra (escalates on heavy turns)"),
            "sol reaches the internal ultra selection ceiling:\n{sol}"
        );
        let luna = render_effort_status(Some(28_000), "gpt-5.6-luna");
        assert!(
            luna.contains("smart → dynamic xhigh~max (escalates on heavy turns)"),
            "luna tops out at max:\n{luna}"
        );
    }

    #[test]
    fn levels_table_mentions_ultracode_alias() {
        let table = format_effort_levels();
        assert!(table.contains("ultracode"), "table: {table}");
    }
}

#[cfg(test)]
mod allowed_tools_turn_scoping_tests {
    use super::{DispatchCtx, LiveCli, dispatch_prompt_command};
    use crate::session::runtime_bridge::build_message_request;
    use zo_cli::tui::App;
    use zo_cli::tui::theme::Theme;
    use runtime::ApiRequest;
    use runtime::message_stream::BlockIdGen;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, MutexGuard};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::mpsc;

    struct CurrentDirGuard {
        original: PathBuf,
        _lock: MutexGuard<'static, ()>,
    }

    impl CurrentDirGuard {
        fn enter(path: &Path) -> Self {
            // Every test entering a temp cwd would otherwise persist its
            // session into the developer's real ~/.zo/projects/ (see
            // isolate_global_zo_home_for_tests).
            crate::isolate_global_zo_home_for_tests();
            let lock = crate::test_cwd_lock()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let original = std::env::current_dir().expect("cwd should exist");
            std::env::set_current_dir(path).expect("set current dir");
            Self {
                original,
                _lock: lock,
            }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    struct ApiKeyGuard {
        previous: Option<std::ffi::OsString>,
    }

    impl ApiKeyGuard {
        fn set_dummy() -> Self {
            let previous = std::env::var_os("ANTHROPIC_API_KEY");
            std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-allowed-tools");
            Self { previous }
        }
    }

    impl Drop for ApiKeyGuard {
        fn drop(&mut self) {
            if let Some(value) = self.previous.take() {
                std::env::set_var("ANTHROPIC_API_KEY", value);
            } else {
                std::env::remove_var("ANTHROPIC_API_KEY");
            }
        }
    }

    fn test_app() -> App {
        let (_tx, rx) = mpsc::channel(8);
        let (cmd_tx, _cmd_rx) = mpsc::channel(8);
        App::new(Theme::no_color(), rx, cmd_tx)
    }

    fn prompt_command(allowed: &[&str]) -> commands::PromptCommandDef {
        commands::PromptCommandDef {
            name: "review-local".to_string(),
            description: Some("Review local diff".to_string()),
            argument_hint: None,
            model: None,
            effort: None,
            allowed_tools: allowed.iter().map(|s| (*s).to_string()).collect(),
            body: "Review $ARGUMENTS".to_string(),
            path: PathBuf::from(".zo/commands/review-local.md"),
        }
    }

    /// Names of the tools the wire request would advertise, given an optional
    /// allow-list. Mirrors the production turn build site's preference rule
    /// (`turn_allowed_tools` over the session-global `allowed_tools`).
    fn offered_tool_names(cli: &LiveCli) -> BTreeSet<String> {
        let allowed = cli
            .turn_allowed_tools
            .clone()
            .or_else(|| cli.allowed_tools.clone());
        let registry = cli.runtime.api_client().tool_registry();
        let request = ApiRequest {
            system_prompt: Arc::from(Vec::<String>::new()),
            wire_reminders: Arc::from(Vec::new()),
            messages: Arc::new(Vec::new()),
            tool_choice: None,
            effort_override: None,
            model_override: None,
        };
        let wire = build_message_request(
            &request,
            "sonnet",
            true,
            allowed.as_ref(),
            &registry,
            None,
            None,
            None,
        );
        wire.tools
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.name)
            .collect()
    }

    #[test]
    fn prompt_command_scopes_tools_for_one_turn_then_resets() {
        let _env_lock = crate::test_env_lock();
        let temp_dir = std::env::temp_dir().join(format!(
            "zo-allowed-tools-scope-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).expect("temp dir should exist");
        let _cwd = CurrentDirGuard::enter(&temp_dir);
        let _api_key = ApiKeyGuard::set_dummy();

        let mut cli = LiveCli::new(
            "sonnet".to_string(),
            true,
            None,
            runtime::PermissionMode::ReadOnly,
        )
        .expect("live cli should build");
        let mut app = test_app();
        let ids = BlockIdGen::default();

        // Baseline: no override, no global allow-list — the full builtin tool set
        // is offered, and crucially it is broader than the scoped set we request.
        let full = offered_tool_names(&cli);
        assert!(
            full.len() > 2,
            "expected the full builtin tool set to exceed the scoped two, got {full:?}"
        );
        assert!(full.contains("bash") && full.contains("read_file"), "{full:?}");

        // Dispatch a prompt command that scopes the turn to Bash + Read.
        let command = prompt_command(&["Bash", "Read"]);
        {
            let mut ctx = DispatchCtx {
                cli: &mut cli,
                app: &mut app,
                ids: &ids,
            };
            let output = dispatch_prompt_command(&mut ctx, &command, "")
                .expect("dispatch should succeed");
            assert!(
                format!("{output:?}").contains("allowed-tools=scoped"),
                "metadata should record the scoping: {output:?}"
            );
        }

        // (1) The override is set and normalized to canonical handler names.
        let scoped: BTreeSet<String> =
            ["bash".to_string(), "read_file".to_string()].into_iter().collect();
        assert_eq!(
            cli.turn_allowed_tools.as_ref(),
            Some(&scoped),
            "turn override should be the normalized scoped set"
        );

        // (2) The wire request offers ONLY the scoped tools for this turn.
        assert_eq!(
            offered_tool_names(&cli),
            scoped,
            "scoped turn must advertise only Bash + Read"
        );

        // (3) Turn completion seam: clearing the override restores the full set.
        // This mirrors `run_live_turn_with_images`'s reset; if that production
        // reset is removed the restriction leaks into the next turn and this
        // assertion fails.
        cli.turn_allowed_tools = None;
        assert_eq!(
            offered_tool_names(&cli),
            full,
            "after the turn completes the full tool set must return"
        );
    }

    #[test]
    fn cli_global_allowed_tools_still_apply_without_turn_override() {
        let _env_lock = crate::test_env_lock();
        let temp_dir = std::env::temp_dir().join(format!(
            "zo-allowed-tools-global-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).expect("temp dir should exist");
        let _cwd = CurrentDirGuard::enter(&temp_dir);
        let _api_key = ApiKeyGuard::set_dummy();

        let global: BTreeSet<String> = ["read_file".to_string()].into_iter().collect();
        let cli = LiveCli::new(
            "sonnet".to_string(),
            true,
            Some(global.clone()),
            runtime::PermissionMode::ReadOnly,
        )
        .expect("live cli should build");

        // No turn override: the session-global `--allowed-tools` set governs.
        assert!(cli.turn_allowed_tools.is_none());
        assert_eq!(offered_tool_names(&cli), global);
    }
}
