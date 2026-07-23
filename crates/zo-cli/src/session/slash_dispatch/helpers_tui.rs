//! Persistent-TUI helpers used by the main slash dispatcher.
//!
//! Three categories:
//!
//! - [`push_report`] funnels every dispatcher report into the
//!   transcript as a `RenderBlock::System` with a stable id.
//! - [`render_persistent_tui_help`] + supporting allow/deny lists
//!   answer "what slash commands actually work in persistent TUI?"
//!   so `/help` and the auto-suggester stay honest.
//! - [`seed_transcript_from_session`] + [`build_model_entries`]
//!   hydrate the transcript and model picker when a session resumes.

use api::{
    ProviderKind, context_window_for_model, custom_provider_catalog, fit_hint_for_model,
    provider_catalog, provider_enabled, resolve_model_alias,
};
use core_types::CardModel;
use runtime::model_catalog::{CatalogProvider, ModelCatalog};
use runtime::message_stream::anthropic::tools::{
    format_tool_result, preview_summary, preview_tool_input,
};
use runtime::message_stream::{
    ActiveModel, BlockIdGen, RenderBlock, SystemLevel, ToolCallId, ToolCallStatus,
};
use zo_cli::tui::{App, ModelPickerEntry};

#[cfg(test)]
use crate::render_repl_help;

use super::super::LiveCli;

pub(crate) fn push_report(
    app: &mut App,
    ids: &BlockIdGen,
    level: SystemLevel,
    text: impl Into<String>,
) {
    // `Into<String>` lets owned callers (the dispatch funnel) move their
    // `String` straight into the block instead of re-allocating via
    // `to_string`; `&str` callers still work and allocate exactly once.
    app.push_block(RenderBlock::System {
        id: ids.next(),
        level,
        text: text.into(),
    });
}

/// Push a structured command-output [`CardModel`] into the transcript as a
/// [`RenderBlock::Card`] — the rich counterpart of [`push_report`].
pub(crate) fn push_card(app: &mut App, ids: &BlockIdGen, card: CardModel) {
    app.push_block(RenderBlock::Card {
        id: ids.next(),
        card,
    });
}

fn push_seed_tool_result(
    app: &mut App,
    ids: &BlockIdGen,
    tool_name: &str,
    output: &str,
    is_error: bool,
) {
    let (level, text) = seed_tool_result_line(tool_name, output, is_error);
    push_report(app, ids, level, text);
}

/// Reconstruct a live-equivalent [`RenderBlock::ToolCall`] from a persisted
/// `ContentBlock::ToolUse`, so a resumed transcript shows the same structured
/// header (icon, path/command preview, ✓ status) the live session rendered —
/// not a flat grey `Tool call {name}` notice. The persisted `input` is a JSON
/// string; it is parsed through the *same* [`preview_tool_input`] builder the
/// streaming path uses, so resume and live are pixel-identical.
fn push_seed_tool_call(
    app: &mut App,
    ids: &BlockIdGen,
    tool_use_id: &str,
    name: &str,
    input: &str,
) {
    let parsed = serde_json::from_str::<serde_json::Value>(input)
        .unwrap_or_else(|_| serde_json::Value::String(input.to_string()));
    let preview = preview_tool_input(name, &parsed);
    let summary = preview_summary(&preview);
    app.push_block(RenderBlock::ToolCall {
        id: ids.next(),
        tool_call_id: ToolCallId(tool_use_id.to_string()),
        name: name.to_string(),
        summary,
        // A persisted call already ran to completion (its result is the next
        // block); seed it as `Ok` so it renders a settled ✓ header rather than
        // a spinner that never resolves. An errored result downgrades the
        // paired card via its own `is_error` flag.
        preview,
        status: ToolCallStatus::Ok,
    });
}

/// Reconstruct a live-equivalent [`RenderBlock::ToolResult`] from a persisted
/// `ContentBlock::ToolResult`, carrying the same `tool_call_id` as its paired
/// call so the layout engine collapses them into one tight card. The persisted
/// `output` is fed through [`format_tool_result`] — the same typed-body builder
/// the streaming path uses — so bash exit codes, diffs, and file listings come
/// back as rich bodies instead of a `Tool result {name} (ok)` notice.
fn push_seed_tool_result_rich(
    app: &mut App,
    ids: &BlockIdGen,
    tool_use_id: &str,
    tool_name: &str,
    output: &str,
    is_error: bool,
) {
    // AskUserQuestion stays a compact system line: its raw JSON verdict makes a
    // noisy result card, and the live TUI never showed it as one either.
    if tool_name.eq_ignore_ascii_case("AskUserQuestion") {
        push_seed_tool_result(app, ids, tool_name, output, is_error);
        return;
    }
    let parsed = serde_json::from_str::<serde_json::Value>(output)
        .unwrap_or_else(|_| serde_json::Value::String(output.to_string()));
    let body = format_tool_result(tool_name, &parsed, is_error);
    app.push_block(RenderBlock::ToolResult {
        id: ids.next(),
        tool_call_id: ToolCallId(tool_use_id.to_string()),
        is_error,
        body,
    });
}

fn seed_tool_result_line(tool_name: &str, output: &str, is_error: bool) -> (SystemLevel, String) {
    if tool_name.eq_ignore_ascii_case("AskUserQuestion") {
        let summary = ask_user_question_seed_summary(output, is_error);
        let level = if is_error {
            SystemLevel::Warn
        } else {
            SystemLevel::Info
        };
        return (level, format!("Question         {summary}"));
    }

    (
        if is_error {
            SystemLevel::Error
        } else {
            SystemLevel::Info
        },
        format!(
            "Tool result      {tool_name} ({})",
            if is_error { "error" } else { "ok" }
        ),
    )
}

fn ask_user_question_seed_summary(output: &str, is_error: bool) -> String {
    let trimmed = output.trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if value
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("answered"))
        {
            let answer = value
                .get("answer")
                .and_then(serde_json::Value::as_str)
                .map(compact_inline)
                .unwrap_or_default();
            return if answer.is_empty() {
                "answered".to_string()
            } else {
                format!("answered · {answer}")
            };
        }

        if value
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("unanswered"))
        {
            let reason = value
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .map(|reason| format!(" · {}", compact_inline(reason)))
                .unwrap_or_default();
            return format!("not answered{reason}");
        }
    }

    if is_error {
        let lower = trimmed.to_ascii_lowercase();
        if lower.contains("dismissed") || lower.contains("without an answer") {
            "dismissed before answer".to_string()
        } else if lower.contains("channel closed") {
            "question channel closed".to_string()
        } else if lower.contains("parse") {
            "invalid question payload".to_string()
        } else {
            "not answered".to_string()
        }
    } else {
        "answered".to_string()
    }
}

fn compact_inline(text: &str) -> String {
    const MAX: usize = 72;
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= MAX {
        return collapsed;
    }
    let end = collapsed
        .char_indices()
        .nth(MAX)
        .map_or(collapsed.len(), |(idx, _)| idx);
    format!("{}…", &collapsed[..end])
}

/// Build the `/help` overview as a structured [`CardModel`].
///
/// The old `/help` pushed a `RenderBlock::System` — a *single-line* notice
/// widget — holding dozens of `{name:<66}` fixed-width rows. With the
/// 4-cell gutter prefix and `Wrap` enabled, every summary spilled past the
/// pane edge and folded onto an unindented continuation line, shredding the
/// column alignment. Rendering through a card instead lays each command out
/// as an aligned two-column table row and truncates the summary with `…`
/// when the pane is narrow, so the catalogue stays tidy at any width.
pub(crate) fn build_help_card(
    prompt_commands: &[commands::PromptCommandDef],
) -> core_types::CardModel {
    use commands::{public_slash_command_specs_iter, slash_command_usage};
    use core_types::{CardModel, CommandCategory};

    fn help_row(key: &str, value: &str) -> Vec<String> {
        vec![key.to_string(), value.to_string()]
    }

    let mut card = CardModel::new("Help");

    // REPL / input shortcuts aren't slash commands, so they have no spec
    // entry — list them explicitly as the first group.
    card = card.section("REPL & input").table(
        Vec::new(),
        vec![
            help_row("/exit, /quit", "Quit the REPL"),
            help_row("Up / Down", "Navigate prompt history"),
            help_row("Tab", "Complete commands, modes, and recent sessions"),
            help_row("Ctrl-C", "Clear input (or exit on an empty prompt)"),
            help_row("Shift+Enter / Ctrl+J", "Insert a newline"),
            help_row("/resume latest", "Resume the most recent session"),
            help_row("/session list", "Browse saved sessions"),
        ],
    );

    // Slash commands grouped by registry category. Driving the rows off the
    // spec table (the single source of truth) means `/help` can never drift
    // from what the dispatcher actually handles.
    for category in CommandCategory::all() {
        let rows: Vec<Vec<String>> = public_slash_command_specs_iter()
            .filter(|spec| spec.category == *category)
            .map(|spec| {
                let summary = if spec.resume_supported {
                    format!("{}  · resume", spec.summary)
                } else {
                    spec.summary.to_string()
                };
                help_row(&slash_command_usage(spec), &summary)
            })
            .collect();
        if !rows.is_empty() {
            card = card
                .section(category.display_name())
                .table(Vec::new(), rows);
        }
    }

    if !prompt_commands.is_empty() {
        let rows = prompt_commands
            .iter()
            .map(|command| {
                help_row(
                    &prompt_command_usage(command),
                    &format!("{}  · prompt", command.summary()),
                )
            })
            .collect::<Vec<_>>();
        card = card
            .section("Project prompt commands")
            .table(Vec::new(), rows);
    }

    // Footer: how the catalogue maps onto persistent-TUI reality.
    let implemented = persistent_tui_supported_commands().len();
    card = card.section("Persistent TUI").table(
        Vec::new(),
        vec![
            help_row(
                "Implemented",
                &format!("{implemented} slash commands (see the categories above)"),
            ),
            help_row(
                "Deferred",
                "plugin-manager actions (install/enable/uninstall) need REPL mode",
            ),
            help_row(
                "Autocomplete",
                "type / to filter; unknown commands suggest the nearest match",
            ),
        ],
    );

    card
}

pub(crate) fn prompt_command_usage(command: &commands::PromptCommandDef) -> String {
    match command.argument_hint.as_deref() {
        Some(hint) if !hint.trim().is_empty() => format!("/{} {hint}", command.name),
        _ => format!("/{}", command.name),
    }
}

/// Plain-text catalogue footer. `/help` now renders the rich
/// [`build_help_card`] instead, so this survives only as the fixture the
/// completion-classification test asserts against — hence `#[cfg(test)]`.
#[cfg(test)]
pub(crate) fn render_persistent_tui_help() -> String {
    let supported = persistent_tui_supported_commands()
        .iter()
        .map(|name| format!("/{name}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{}\n\nPersistent TUI\n  Implemented      {supported}\n  Deferred         plugin-manager actions (install/enable/uninstall) need REPL mode\n  Autocomplete     type / to filter; unknown commands suggest the nearest match",
        render_repl_help(),
    )
}

/// Slash commands wired into the persistent-TUI dispatcher.
///
/// Derived from the registry's built-in specs — the single source of
/// truth for command names — rather than a hand-maintained array, so the
/// `/help` footer and the auto-suggester can never drift from what the
/// dispatcher actually handles (every [`commands::SlashCommand`] variant
/// now has a real arm). Canonical names only, without the leading `/`.
pub(crate) fn persistent_tui_supported_commands() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = commands::slash_command_specs()
        .iter()
        .map(|spec| spec.name)
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

#[cfg(test)]
pub(crate) fn persistent_tui_candidate_supported(candidate: &str) -> bool {
    let base = candidate
        .trim()
        .trim_start_matches('/')
        .split_whitespace()
        .next()
        .unwrap_or("");
    // Resolve through the registry so aliases (`/plugins` → `plugin`) and
    // case-insensitive forms count as supported, matching the dispatcher.
    commands::SlashCommandRegistry::with_builtins().contains(base)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeedUserVisibility {
    Show,
    HidePromptOnly,
    HideTurn,
}

fn seed_user_visibility(message: &runtime::ConversationMessage) -> SeedUserVisibility {
    let Some(text) = first_text_block(message) else {
        return SeedUserVisibility::Show;
    };
    let trimmed = text.trim_start();
    if trimmed.starts_with("[deep:VERIFY]") || trimmed.starts_with("[[ZO-DEEP:VERIFY]]") {
        return SeedUserVisibility::HideTurn;
    }
    if trimmed.starts_with("[auto:RETRY]")
        || trimmed.starts_with("[deep:PLAN]")
        || trimmed.starts_with("[deep:EXEC]")
    {
        return SeedUserVisibility::HidePromptOnly;
    }
    SeedUserVisibility::Show
}

fn first_text_block(message: &runtime::ConversationMessage) -> Option<&str> {
    message.blocks.iter().find_map(|block| match block {
        runtime::ContentBlock::Text { text } => Some(text.as_str()),
        _ => None,
    })
}

pub(crate) fn seed_transcript_from_session(
    app: &mut App,
    ids: &BlockIdGen,
    session: &runtime::Session,
) {
    let mut suppress_internal_turn = false;
    for message in session.messages.iter() {
        if suppress_internal_turn {
            if matches!(message.role, runtime::MessageRole::User) {
                suppress_internal_turn = false;
            } else {
                continue;
            }
        }

        match message.role {
            runtime::MessageRole::User => match seed_user_visibility(message) {
                SeedUserVisibility::Show => {
                    for block in &message.blocks {
                        if let runtime::ContentBlock::Text { text } = block {
                            // Seed as a real UserMessage block — the same widget a
                            // live turn pushes — so a resumed transcript renders
                            // identically to the session it restores (amber role
                            // rail + markdown body). The old `System` info row
                            // (`User  {text}`) reflowed multi-line prompts through
                            // the system-notice renderer and looked broken.
                            app.push_block(RenderBlock::UserMessage {
                                id: ids.next(),
                                text: text.clone(),
                            });
                        }
                    }
                }
                // Synthetic harness prompts were never echoed into the live TUI
                // as user messages, but they are persisted as normal session
                // input so the model can continue. Hide just the prompt and let
                // the visible assistant/tool output from that phase remain.
                SeedUserVisibility::HidePromptOnly => {}
                // The verifier phase is entirely internal: its prompt, tool
                // probes, and single-line JSON verdict make the resumed
                // transcript look corrupted. Live TUI narrates this phase via
                // ephemeral `auto: verifying…` notes instead, so omit the raw
                // persisted sub-turn until the next real user input.
                SeedUserVisibility::HideTurn => suppress_internal_turn = true,
            },
            runtime::MessageRole::Assistant => seed_assistant_message_blocks(app, ids, message),
            runtime::MessageRole::Tool => {
                for block in &message.blocks {
                    if let runtime::ContentBlock::ToolResult {
                        tool_use_id,
                        tool_name,
                        is_error,
                        output,
                        ..
                    } = block
                    {
                        push_seed_tool_result_rich(
                            app,
                            ids,
                            tool_use_id,
                            tool_name,
                            output,
                            *is_error,
                        );
                    }
                }
            }
            runtime::MessageRole::System => {
                for block in &message.blocks {
                    if let runtime::ContentBlock::Text { text } = block {
                        push_report(app, ids, SystemLevel::Info, text.clone());
                    }
                }
            }
        }
    }
}

/// Re-render one persisted assistant message's blocks into the resumed transcript.
/// Extracted from `seed_transcript_from_session` so that function stays within the
/// line budget; the behavior is identical to the inlined match it replaced.
fn seed_assistant_message_blocks(
    app: &mut App,
    ids: &BlockIdGen,
    message: &runtime::ConversationMessage,
) {
    for block in &message.blocks {
        match block {
            runtime::ContentBlock::Text { text } => {
                app.push_block(RenderBlock::TextDelta {
                    id: ids.next(),
                    text: text.clone(),
                    done: true,
                });
            }
            runtime::ContentBlock::ToolUse { id, name, input } => {
                push_seed_tool_call(app, ids, id, name, input);
            }
            runtime::ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                is_error,
                output,
                ..
            } => push_seed_tool_result_rich(app, ids, tool_use_id, tool_name, output, *is_error),
            runtime::ContentBlock::Image { media_type, data } => {
                // Rehydrate the image as a real render block (the bytes are
                // persisted as base64) so a resumed session shows the picture, not
                // just a text marker. An undecodable payload falls back to the marker.
                use base64::Engine as _;
                match base64::engine::general_purpose::STANDARD.decode(data) {
                    Ok(bytes) => app.push_block(RenderBlock::Image {
                        id: ids.next(),
                        data: bytes,
                        media_type: media_type.clone(),
                    }),
                    Err(_) => {
                        push_report(app, ids, SystemLevel::Info, format!("[image: {media_type}]"));
                    }
                }
            }
            // Reasoning blocks stay in storage for replay but are not
            // re-rendered to the TUI on resume.
            runtime::ContentBlock::Thinking { .. }
            | runtime::ContentBlock::RedactedThinking { .. } => {}
        }
    }
}

fn env_non_empty(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

fn google_model_picker_enabled() -> bool {
    api::google_code_assist_oauth_present()
        || env_non_empty("GOOGLE_API_KEY")
        || api::google_gemini_oauth_available()
}

#[cfg(test)]
fn default_model_rows(
    openai_enabled: bool,
    google_enabled: bool,
) -> Vec<(&'static str, &'static str, &'static str)> {
    runtime::model_catalog::builtin_rows()
        .iter()
        .filter(|(provider, _, _)| match provider {
            CatalogProvider::Anthropic => true,
            CatalogProvider::Openai => openai_enabled,
            CatalogProvider::Google => google_enabled,
        })
        .map(|(provider, id, display)| (provider.key(), *id, *display))
        .collect()
}

fn picker_provider_for_model(model: &str, resolved: &str) -> &'static str {
    if let Some(entry) = provider_catalog().iter().find(|entry| {
        entry.alias.eq_ignore_ascii_case(model)
            || entry.alias.eq_ignore_ascii_case(resolved)
            || entry.canonical_model_id.eq_ignore_ascii_case(model)
            || entry.canonical_model_id.eq_ignore_ascii_case(resolved)
    }) {
        return picker_provider_key(entry.provider);
    }

    if let Some((provider, _)) = custom_provider_catalog().into_iter().find(|(_, models)| {
        models.iter().any(|candidate| {
            candidate.eq_ignore_ascii_case(resolved) || candidate.eq_ignore_ascii_case(model)
        })
    }) {
        return provider;
    }

    let lower = resolved.to_ascii_lowercase();
    if lower.starts_with("claude") {
        "claude"
    } else if lower.starts_with("gpt")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("codex")
    {
        "openai"
    } else if lower.starts_with("gemini") {
        "google"
    } else if lower.starts_with("grok") {
        "xai"
    } else if lower.starts_with("ollama") {
        "ollama"
    } else {
        "custom"
    }
}

const fn picker_provider_key(provider: ProviderKind) -> &'static str {
    match provider {
        ProviderKind::Anthropic => "claude",
        ProviderKind::OpenAi => "openai",
        ProviderKind::Google => "google",
        ProviderKind::Xai => "xai",
        ProviderKind::Ollama => "ollama",
    }
}

use std::sync::{Mutex, OnceLock};

const MODEL_HISTORY_VERSION: u32 = 1;
const MODEL_HISTORY_FILE: &str = "model-history.json";

#[derive(serde::Serialize, serde::Deserialize)]
struct ModelHistoryDocument {
    version: u32,
    models: Vec<String>,
}

static MODEL_HISTORY: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn model_history_path() -> std::path::PathBuf {
    runtime::default_config_home().join(MODEL_HISTORY_FILE)
}

fn canonical_model_history(models: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut canonical = Vec::with_capacity(3);
    for model in models {
        let resolved = resolve_model_alias(&model);
        if !canonical.contains(&resolved) {
            canonical.push(resolved);
            if canonical.len() == 3 {
                break;
            }
        }
    }
    canonical
}

fn merge_model_history(history: &[String], alias: &str) -> Vec<String> {
    canonical_model_history(
        std::iter::once(alias.to_string()).chain(history.iter().cloned()),
    )
}

fn load_model_history(path: &std::path::Path) -> Vec<String> {
    let Some(document) = std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<ModelHistoryDocument>(&text).ok())
    else {
        return Vec::new();
    };
    if document.version != MODEL_HISTORY_VERSION {
        return Vec::new();
    }
    canonical_model_history(document.models)
}

fn save_model_history(path: &std::path::Path, history: &[String]) -> std::io::Result<()> {
    let document = ModelHistoryDocument {
        version: MODEL_HISTORY_VERSION,
        models: canonical_model_history(history.iter().cloned()),
    };
    let bytes = serde_json::to_vec_pretty(&document).map_err(std::io::Error::other)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    crate::write_atomic(path, &bytes)
}

fn model_history() -> &'static Mutex<Vec<String>> {
    MODEL_HISTORY.get_or_init(|| Mutex::new(load_model_history(&model_history_path())))
}

pub(crate) fn add_model_to_history(alias: &str) {
    if let Ok(mut history) = model_history().lock() {
        *history = merge_model_history(&history, alias);
        let _ = save_model_history(&model_history_path(), &history);
    }
}

pub(crate) fn get_model_history() -> Vec<String> {
    if let Ok(history) = model_history().lock() {
        history.clone()
    } else {
        Vec::new()
    }
}

pub(crate) fn build_model_entries(cli: &LiveCli) -> Vec<ModelPickerEntry> {
    let mut connected = vec![CatalogProvider::Anthropic];
    if provider_enabled(ProviderKind::OpenAi) {
        connected.push(CatalogProvider::Openai);
    }
    if google_model_picker_enabled() {
        connected.push(CatalogProvider::Google);
    }
    let catalog = ModelCatalog::load().unwrap_or_else(|_| {
        ModelCatalog::load_from(runtime::default_config_home().join("missing-model-catalog.json"))
            .expect("an absent fallback catalog is valid")
    });
    let mut entries: Vec<ModelPickerEntry> = catalog
        .rows(&connected, false)
        .into_iter()
        .map(|row| ModelPickerEntry {
            provider: row.provider.key().to_string(),
            model: ActiveModel {
                provider: row.provider.key(),
                alias: row.id.clone(),
                display_name: if row.builtin {
                    model_display_name(&row.id, &row.display_name)
                } else {
                    format!(
                        "{} · UNVERIFIED",
                        model_display_name(&row.id, &row.display_name)
                    )
                },
                context_limit: u32::try_from(context_window_for_model(&row.id)).unwrap_or(u32::MAX),
            },
        })
        .collect();

    // The active model is pinned into the suggested group by
    // `finalize_picker_entries`, which matches on the resolved canonical id so
    // a full id like `claude-fable-5` is recognized as the catalog's `fable`
    // row instead of being added as a second, duplicate entry.

    // Models from `/connect`-configured custom providers (Ollama / LM Studio /
    // DeepSeek / …) so a connected provider's models show up in the picker, not
    // just via `/model <name>` from memory. Skips any alias a built-in already
    // covers to avoid duplicate rows.
    for (provider, models) in custom_provider_catalog() {
        for alias in models {
            if entries.iter().any(|entry| entry.model.alias == alias) {
                continue;
            }
            let context_limit =
                u32::try_from(context_window_for_model(&alias)).unwrap_or(u32::MAX);
            entries.push(ModelPickerEntry {
                provider: provider.to_string(),
                model: ActiveModel {
                    provider,
                    alias: alias.clone(),
                    display_name: model_display_name(&alias, &alias),
                    context_limit,
                },
            });
        }
    }

    let current_resolved = resolve_model_alias(&cli.model);
    let current_provider = picker_provider_for_model(&cli.model, &current_resolved);
    let current_entry = ModelPickerEntry {
        provider: current_provider.to_string(),
        model: ActiveModel {
            provider: current_provider,
            alias: cli.model.clone(),
            display_name: model_display_name(&cli.model, &current_resolved),
            context_limit: u32::try_from(context_window_for_model(&current_resolved))
                .unwrap_or(u32::MAX),
        },
    };

    finalize_picker_entries(entries, current_entry, &get_model_history())
}

/// Pin the active model plus any recently-used models into a single leading
/// `suggested` group, de-duplicated by resolved canonical id.
///
/// Closes two picker bugs: (1) a full model id (`claude-fable-5`) is recognized
/// as the catalog's short-alias row (`fable`) instead of adding a second row;
/// (2) a model promoted into SUGGESTED is removed from its provider group so it
/// never renders twice. With no history only the current model is suggested —
/// the old fabricated defaults pointed partly at a retired alias that silently
/// dropped, leaving unrelated models as the only "suggestions".
fn finalize_picker_entries(
    mut entries: Vec<ModelPickerEntry>,
    current_entry: ModelPickerEntry,
    history: &[String],
) -> Vec<ModelPickerEntry> {
    let current_resolved = resolve_model_alias(&current_entry.model.alias);
    if !entries
        .iter()
        .any(|entry| resolve_model_alias(&entry.model.alias) == current_resolved)
    {
        entries.insert(0, current_entry);
    }

    // Always suggest the active model first, then any recently-used models.
    // The resolved-id dedup below drops the repeat when history also lists the
    // current model, so it is pinned into SUGGESTED regardless of history.
    let mut suggested_aliases: Vec<String> = vec![current_resolved.clone()];
    suggested_aliases.extend(history.iter().cloned());

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut suggested_entries: Vec<ModelPickerEntry> = Vec::new();
    for alias in suggested_aliases {
        let resolved = resolve_model_alias(&alias);
        if !seen.insert(resolved.clone()) {
            continue;
        }
        if let Some(matching) = entries
            .iter()
            .find(|entry| resolve_model_alias(&entry.model.alias) == resolved)
        {
            let mut suggested = matching.clone();
            suggested.provider = "suggested".to_string();
            suggested_entries.push(suggested);
        }
    }

    if !suggested_entries.is_empty() {
        let shown: std::collections::HashSet<String> = suggested_entries
            .iter()
            .map(|entry| resolve_model_alias(&entry.model.alias))
            .collect();
        entries.retain(|entry| !shown.contains(&resolve_model_alias(&entry.model.alias)));
        entries.splice(0..0, suggested_entries);
    }

    entries
}

fn model_display_name(model: &str, base: &str) -> String {
    fit_hint_for_model(model).map_or_else(
        || base.to_string(),
        |hint| format!("{base} ({})", hint.display_label()),
    )
}

#[cfg(test)]
mod tests {
    use core_types::CardElement;
    use runtime::message_stream::SystemLevel;
    use std::path::PathBuf;

    use super::{
        ModelPickerEntry, build_help_card, default_model_rows, finalize_picker_entries,
        load_model_history, merge_model_history, model_display_name, model_history_path,
        picker_provider_for_model, prompt_command_usage, resolve_model_alias, save_model_history,
        seed_tool_result_line, MODEL_HISTORY_FILE,
    };

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&std::ffi::OsStr>) -> Self {
            let original = std::env::var_os(key);
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn mk_entry(provider: &str, alias: &str, display: &str) -> ModelPickerEntry {
        ModelPickerEntry {
            provider: provider.to_string(),
            model: runtime::message_stream::ActiveModel {
                provider: "anthropic",
                alias: alias.to_string(),
                display_name: display.to_string(),
                context_limit: 1_000_000,
            },
        }
    }

    #[test]
    fn full_id_current_model_collapses_with_catalog_short_alias() {
        // The current model arrives as the full id while the catalog lists the
        // short alias — they must render as ONE row, not `claude-fable-5` +
        // `Fable 5`.
        let short = "fable";
        let full = resolve_model_alias(short);
        let entries = vec![
            mk_entry("claude", short, "Fable 5"),
            mk_entry("claude", "opus", "Opus 4.8"),
        ];
        let current = mk_entry("claude", &full, "Fable 5");
        let out = finalize_picker_entries(entries, current, &[]);
        let matching: Vec<_> = out
            .iter()
            .filter(|e| resolve_model_alias(&e.model.alias) == full)
            .collect();
        assert_eq!(matching.len(), 1, "one row per model: {out:?}");
        assert_eq!(matching[0].provider, "suggested");
    }

    #[test]
    fn empty_history_suggests_only_the_current_model() {
        // No history must NOT fabricate cross-provider defaults (the old
        // sonnet/gemini/gpt-5.5 list); only the current model is suggested.
        let entries = vec![
            mk_entry("claude", "fable", "Fable 5"),
            mk_entry("claude", "opus", "Opus 4.8"),
            mk_entry("claude", "sonnet", "Sonnet 5"),
        ];
        let current = mk_entry("claude", "opus", "Opus 4.8");
        let out = finalize_picker_entries(entries, current, &[]);
        let suggested: Vec<_> = out.iter().filter(|e| e.provider == "suggested").collect();
        assert_eq!(suggested.len(), 1, "only the current model is suggested");
        assert_eq!(
            resolve_model_alias(&suggested[0].model.alias),
            resolve_model_alias("opus")
        );
    }

    #[test]
    fn suggested_model_is_not_shown_twice() {
        let entries = vec![
            mk_entry("claude", "fable", "Fable 5"),
            mk_entry("claude", "opus", "Opus 4.8"),
        ];
        let current = mk_entry("claude", "fable", "Fable 5");
        let out = finalize_picker_entries(entries, current, &[]);
        let fable = resolve_model_alias("fable");
        let count = out
            .iter()
            .filter(|e| resolve_model_alias(&e.model.alias) == fable)
            .count();
        assert_eq!(count, 1, "fable renders exactly once");
        assert!(
            out.iter()
                .any(|e| e.provider == "suggested" && resolve_model_alias(&e.model.alias) == fable)
        );
    }

    #[test]
    fn history_dedups_by_resolved_id() {
        let entries = vec![
            mk_entry("claude", "fable", "Fable 5"),
            mk_entry("claude", "opus", "Opus 4.8"),
        ];
        let current = mk_entry("claude", "opus", "Opus 4.8");
        let fable_full = resolve_model_alias("fable");
        // The same model listed twice under two alias forms in history.
        let history = vec!["fable".to_string(), fable_full.clone()];
        let out = finalize_picker_entries(entries, current, &history);
        let suggested_fable = out
            .iter()
            .filter(|e| {
                e.provider == "suggested" && resolve_model_alias(&e.model.alias) == fable_full
            })
            .count();
        assert_eq!(suggested_fable, 1, "fable suggested once despite two forms");
    }

    #[test]
    fn model_history_merge_resolves_aliases_and_keeps_three_recents() {
        let history = vec![
            resolve_model_alias("fable"),
            resolve_model_alias("opus"),
            resolve_model_alias("haiku"),
        ];

        let merged = merge_model_history(&history, "sonnet");

        assert_eq!(
            merged,
            vec![
                resolve_model_alias("sonnet"),
                resolve_model_alias("fable"),
                resolve_model_alias("opus"),
            ]
        );
    }

    #[test]
    fn model_history_path_honors_config_home_override() {
        let _env_lock = crate::test_env_lock();
        let home = std::env::temp_dir().join(format!(
            "zo-model-history-home-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(home.as_os_str()));

        assert_eq!(model_history_path(), home.join(MODEL_HISTORY_FILE));
    }

    #[test]
    fn model_history_file_round_trips_versioned_format() {
        let path = std::env::temp_dir().join(format!(
            "zo-model-history-roundtrip-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let expected = vec![
            resolve_model_alias("fable"),
            resolve_model_alias("opus"),
            resolve_model_alias("gpt-5.6-sol"),
        ];

        save_model_history(&path, &expected).expect("save model history");
        let loaded = load_model_history(&path);
        let document: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&path).expect("read persisted model history"),
        )
        .expect("model history is valid JSON");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded, expected);
        assert_eq!(document["version"], 1);
        assert_eq!(document["models"].as_array().map(Vec::len), Some(3));
    }

    #[test]
    fn corrupt_model_history_file_loads_as_empty() {
        let path = std::env::temp_dir().join(format!(
            "zo-model-history-corrupt-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        assert!(
            load_model_history(&path).is_empty(),
            "a missing history file must be empty"
        );
        std::fs::write(&path, b"{not valid json").expect("write corrupt model history");

        let loaded = load_model_history(&path);

        let _ = std::fs::remove_file(path);
        assert!(loaded.is_empty(), "a corrupt history file must be empty");
    }

    #[test]
    fn non_empty_history_still_suggests_current_model() {
        // Regression guard: with a non-empty history the active model must
        // still be pinned into SUGGESTED, not dropped in favor of history only.
        let entries = vec![
            mk_entry("claude", "fable", "Fable 5"),
            mk_entry("claude", "opus", "Opus 4.8"),
            mk_entry("claude", "sonnet", "Sonnet 5"),
        ];
        let current = mk_entry("claude", "opus", "Opus 4.8");
        let history = vec!["fable".to_string(), "sonnet".to_string()];
        let out = finalize_picker_entries(entries, current, &history);
        let opus = resolve_model_alias("opus");
        assert!(
            out.iter()
                .any(|e| e.provider == "suggested" && resolve_model_alias(&e.model.alias) == opus),
            "current model stays in SUGGESTED with non-empty history: {out:?}"
        );
        let count = out
            .iter()
            .filter(|e| resolve_model_alias(&e.model.alias) == opus)
            .count();
        assert_eq!(count, 1, "current model renders exactly once");
    }

    fn prompt_command() -> commands::PromptCommandDef {
        commands::PromptCommandDef {
            name: "review-local".to_string(),
            description: Some("Review local diff".to_string()),
            argument_hint: Some("<scope>".to_string()),
            model: None,
            effort: None,
            body: "Review $ARGUMENTS".to_string(),
            allowed_tools: Vec::new(),
            path: PathBuf::from(".zo/commands/review-local.md"),
        }
    }

    #[test]
    fn prompt_command_usage_includes_argument_hint() {
        assert_eq!(
            prompt_command_usage(&prompt_command()),
            "/review-local <scope>"
        );
    }

    #[test]
    fn model_rows_include_current_sonnet_alias() {
        let rows = default_model_rows(false, false);

        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "claude" && *alias == "sonnet" && *display == "Sonnet 5"
        }));
    }

    #[test]
    fn model_rows_include_gemini_when_google_is_enabled() {
        let rows = default_model_rows(false, true);

        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "google"
                && *alias == "gemini-3.1-pro-preview"
                && *display == "Gemini 3.1 Pro Preview"
        }));
        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "google" && *alias == "gemini-3.5-flash" && *display == "Gemini 3.5 Flash"
        }));
        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "google"
                && *alias == "gemini-3.1-flash-lite"
                && *display == "Gemini 3.1 Flash Lite"
        }));
    }

    #[test]
    fn model_rows_omit_gemini_when_google_is_disabled() {
        let rows = default_model_rows(false, false);

        assert!(
            !rows
                .iter()
                .any(|(provider, alias, _)| *provider == "google" || alias.starts_with("gemini"))
        );
    }

    #[test]
    fn model_rows_include_openai_models_when_openai_is_enabled() {
        let rows = default_model_rows(true, false);

        // gpt-5.5는 카탈로그 퇴역(2026-07-11) — 피커에 있으면 안 된다.
        assert!(!rows
            .iter()
            .any(|(_, alias, _)| alias.starts_with("gpt-5.5")));
        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "openai" && *alias == "gpt-5.6-sol" && *display == "GPT-5.6-Sol"
        }));
        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "openai" && *alias == "gpt-5.6-terra" && *display == "GPT-5.6-Terra"
        }));
        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "openai" && *alias == "gpt-5.6-luna" && *display == "GPT-5.6-Luna"
        }));
        assert!(rows.iter().any(|(provider, alias, display)| {
            *provider == "openai"
                && *alias == "gpt-5.3-codex-spark"
                && *display == "GPT-5.3-Codex-Spark"
        }));
    }

    #[test]
    fn current_model_fallback_infers_provider_from_model_id() {
        // `gpt-5.5-fast` is intentionally not a default picker row (fast is a
        // serving-priority variant), but if it is the active session model the
        // inserted current-model row must still render under OpenAI, not Claude.
        assert_eq!(picker_provider_for_model("gpt-5.5-fast", "gpt-5.5-fast"), "openai");
        assert_eq!(picker_provider_for_model("gpt-6-preview", "gpt-6-preview"), "openai");
        assert_eq!(picker_provider_for_model("claude-sonnet", "claude-sonnet-5"), "claude");
        assert_eq!(picker_provider_for_model("gemini-3.5-flash", "gemini-3.5-flash"), "google");
    }

    #[test]
    fn model_display_name_omits_fit_hint_when_feature_is_disabled() {
        assert_eq!(
            model_display_name("qwen2.5-coder-32b", "Qwen Coder"),
            "Qwen Coder"
        );
    }

    #[test]
    fn build_help_card_includes_project_prompt_commands() {
        let command = prompt_command();
        let card = build_help_card(std::slice::from_ref(&command));

        assert!(card.elements.iter().any(|element| {
            matches!(
                element,
                CardElement::Section { label } if label == "Project prompt commands"
            )
        }));
        assert!(card.elements.iter().any(|element| {
            matches!(
                element,
                CardElement::Table { rows, .. }
                    if rows.iter().any(|row| {
                        row.len() == 2
                            && row[0] == "/review-local <scope>"
                            && row[1] == "Review local diff  · prompt"
                    })
            )
        }));
    }

    #[test]
    fn seed_ask_user_question_error_does_not_replay_as_tool_result_error() {
        let (level, text) = seed_tool_result_line(
            "AskUserQuestion",
            "User question dismissed without answer",
            true,
        );

        assert_eq!(level, SystemLevel::Warn);
        assert_eq!(text, "Question         dismissed before answer");
        assert!(!text.contains("Tool result"));
        assert!(!text.contains("AskUserQuestion"));
        assert!(!text.contains("(error)"));
    }

    #[test]
    fn resume_rebuilds_rich_tool_blocks_paired_by_id() {
        use runtime::message_stream::{
            BlockIdGen, RenderBlock, ToolCallId, ToolCallStatus, ToolResultBody,
        };
        use runtime::{ContentBlock, ConversationMessage, Session};
        use zo_cli::tui::{App, Theme};
        use tokio::sync::mpsc;

        // A persisted bash tool round-trip: ToolUse (assistant) → ToolResult.
        let mut session = Session::new();
        session
            .push_message(ConversationMessage::user_text("run the tests"))
            .unwrap();
        session
            .push_message(ConversationMessage::assistant(vec![
                ContentBlock::ToolUse {
                    id: "call-1".to_string(),
                    name: "bash".to_string(),
                    input: r#"{"command":"cargo test"}"#.to_string(),
                },
            ]))
            .unwrap();
        session
            .push_message(ConversationMessage::tool_result(
                "call-1",
                "bash",
                r#"{"exit_code":0,"stdout":"ok","stderr":""}"#,
                false,
            ))
            .unwrap();

        let (_tx, rx) = mpsc::channel::<RenderBlock>(16);
        let (cmd_tx, _cmd_rx) = mpsc::channel(16);
        let mut app = App::new(Theme::no_color(), rx, cmd_tx);
        let ids = BlockIdGen::default();

        super::seed_transcript_from_session(&mut app, &ids, &session);

        let blocks = app.transcript().blocks();
        // The tool round-trip must come back as a rich ToolCall + ToolResult,
        // never a pair of flat System notices.
        let call = blocks
            .iter()
            .find_map(|b| match b {
                RenderBlock::ToolCall {
                    tool_call_id,
                    name,
                    status,
                    ..
                } => Some((tool_call_id.clone(), name.clone(), *status)),
                _ => None,
            })
            .expect("resumed transcript must contain a rich ToolCall block");
        assert_eq!(call.0, ToolCallId("call-1".to_string()));
        assert_eq!(call.1, "bash");
        assert_eq!(call.2, ToolCallStatus::Ok);

        let result = blocks
            .iter()
            .find_map(|b| match b {
                RenderBlock::ToolResult {
                    tool_call_id, body, ..
                } => Some((tool_call_id.clone(), body.clone())),
                _ => None,
            })
            .expect("resumed transcript must contain a rich ToolResult block");
        // Paired by id so the layout collapses them into one tight card.
        assert_eq!(result.0, ToolCallId("call-1".to_string()));
        // bash result decoded into a structured Bash body, not a flat notice.
        assert!(
            matches!(result.1, ToolResultBody::Bash(_)),
            "bash output must rebuild as a structured Bash body, got {:?}",
            result.1
        );

        // No flat "Tool call bash" / "Tool result bash (ok)" System notices.
        assert!(
            !blocks.iter().any(|b| matches!(
                b,
                RenderBlock::System { text, .. } if text.contains("Tool call") || text.contains("Tool result")
            )),
            "resume must not emit flat System tool notices"
        );
    }
}
