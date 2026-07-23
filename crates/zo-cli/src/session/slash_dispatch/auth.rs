//! Authentication / provider commands: login, logout, connect.
//!
//! Provider metadata lives in one [`PROVIDERS`] table so `/connect` setup hints
//! stay in sync instead of drifting across inline `match` arms.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value};

use runtime::message_stream::SystemLevel;

use super::context::DispatchCtx;
use super::output::CommandOutput;

/// A non-Anthropic model provider reachable via an environment credential or
/// saved OAuth/ADC credential.
struct Provider {
    /// Accepted `/connect <name>` aliases (compared case-insensitively).
    /// The first entry is the canonical name used in hints.
    aliases: &'static [&'static str],
    /// Human-facing label.
    label: &'static str,
    /// Connection detection and setup hint.
    connection: ProviderConnection,
    /// Suggested `zo --model` value once connected.
    model_hint: &'static str,
}

#[derive(Clone, Copy)]
enum ProviderConnection {
    Env { env_key: &'static str },
    OpenAi,
    Google,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ProviderTokenLimits {
    pub(crate) context_window: Option<u64>,
    pub(crate) max_output_tokens: Option<u64>,
}

impl ProviderConnection {
    fn is_connected(self) -> bool {
        match self {
            Self::Env { env_key } => {
                env_non_empty(env_key) || api::load_openai_compat_api_key(env_key).ok().flatten().is_some()
            }
            Self::OpenAi => env_non_empty("OPENAI_API_KEY") || openai_oauth_present(),
            Self::Google => {
                api::google_code_assist_oauth_present()
                    || env_non_empty("GOOGLE_API_KEY")
                    || api::google_gemini_oauth_available()
            }
        }
    }

    fn connected_detail(self) -> &'static str {
        match self {
            Self::Env { env_key } => env_key,
            Self::OpenAi => "OPENAI_API_KEY or saved ChatGPT OAuth",
            Self::Google => "saved Gemini OAuth, GOOGLE_API_KEY, or Google ADC",
        }
    }

    fn setup_hint(self) -> String {
        match self {
            Self::Env { env_key } => format!(
                "Set the API key in your shell before starting zo:\n  \
                 export {env_key}=your-key-here"
            ),
            Self::OpenAi => "Run `/login openai` for ChatGPT subscription OAuth, or set:\n  export OPENAI_API_KEY=your-key-here".to_string(),
            Self::Google => "Run `/login google` for Gemini OAuth, or set:\n  \
                 export GOOGLE_API_KEY=your-key-here\n\n  \
                 Advanced ADC flow: `/login google-adc`"
                .to_string(),
        }
    }
}

fn env_non_empty(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|v| !v.trim().is_empty())
}

fn valid_auth_env_name(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn openai_oauth_present() -> bool {
    api::oauth_store::load_openai_oauth()
        .ok()
        .flatten()
        .is_some()
}

/// Single source of truth for connectable providers.
const PROVIDERS: &[Provider] = &[
    Provider {
        aliases: &["openai", "gpt", "codex"],
        label: "OpenAI",
        connection: ProviderConnection::OpenAi,
        model_hint: "gpt-5.5",
    },
    Provider {
        aliases: &["google", "gemini"],
        label: "Google",
        connection: ProviderConnection::Google,
        model_hint: "gemini-3.5-flash",
    },
    Provider {
        aliases: &["xai", "grok"],
        label: "xAI",
        connection: ProviderConnection::Env {
            env_key: "XAI_API_KEY",
        },
        model_hint: "grok",
    },
];

/// An OpenAI-compatible provider `/connect` can persist into the user's
/// `settings.json`, so the runtime and model picker pick it up on the next
/// start. Cloud presets carry a curated default model list; local servers
/// (Ollama / LM Studio) are probed live for the models they actually serve.
struct ConnectPreset {
    /// `/connect <alias>` names (case-insensitive); the first is the canonical
    /// provider name written to settings.
    aliases: &'static [&'static str],
    label: &'static str,
    /// OpenAI-compatible base URL.
    base_url: &'static str,
    /// Env var holding the API key, or `None` for keyless local servers.
    auth_env: Option<&'static str>,
    /// Curated default model ids (cloud), or the fallback list when a local
    /// server is unreachable or advertises nothing.
    models: &'static [&'static str],
    /// `true` for local servers probed with [`api::discover_models`].
    local: bool,
    /// Optional compatibility override for streaming usage chunks.
    include_usage: Option<bool>,
}

/// Writable OpenAI-compatible presets. Cloud endpoint paths are best-effort and
/// may need adjusting per the provider's current docs.
const CONNECT_PRESETS: &[ConnectPreset] = &[
    ConnectPreset {
        aliases: &["deepseek"],
        label: "DeepSeek",
        base_url: "https://api.deepseek.com",
        auth_env: Some("DEEPSEEK_API_KEY"),
        models: &["deepseek-chat", "deepseek-reasoner"],
        local: false,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["kimi", "moonshot"],
        label: "Kimi (Moonshot)",
        base_url: "https://api.moonshot.ai/v1",
        auth_env: Some("MOONSHOT_API_KEY"),
        models: &["kimi-k2-0905-preview", "moonshot-v1-32k"],
        local: false,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["qwen", "dashscope"],
        label: "Qwen (DashScope)",
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        auth_env: Some("DASHSCOPE_API_KEY"),
        models: &["qwen-max", "qwen-plus", "qwen-turbo"],
        local: false,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["nvidia", "nvidia-nim", "nim"],
        label: "NVIDIA NIM",
        base_url: "https://integrate.api.nvidia.com/v1",
        auth_env: Some("NVIDIA_API_KEY"),
        models: &["meta/llama-3.1-8b-instruct", "z-ai/glm-5.2"],
        local: false,
        include_usage: Some(false),
    },
    ConnectPreset {
        aliases: &["openrouter"],
        label: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        auth_env: Some("OPENROUTER_API_KEY"),
        models: &["openrouter/auto"],
        local: false,
        include_usage: Some(false),
    },
    ConnectPreset {
        aliases: &["ollama"],
        label: "Ollama",
        base_url: "http://localhost:11434/v1",
        auth_env: None,
        models: &[],
        local: true,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["lmstudio", "lm-studio"],
        label: "LM Studio",
        base_url: "http://localhost:1234/v1",
        auth_env: None,
        models: &[],
        local: true,
        include_usage: None,
    },
];

/// Outcome of a `/connect <preset>` attempt, mapped to a `CommandOutput` (TUI)
/// or printed (headless).
pub(crate) enum ConnectReport {
    Info(String),
    Warn(String),
    Error(String),
}

#[derive(Debug)]
enum ConnectProbe {
    NotNeeded,
    MissingKey(&'static str),
    Verified { model_count: usize },
    Failed(String),
}

impl ConnectProbe {
    fn message(&self) -> Option<String> {
        match self {
            Self::NotNeeded => None,
            Self::MissingKey(env) => Some(format!(
                "API key not found in this process. Set it before chatting:\n  export {env}=your-key-here"
            )),
            Self::Verified { model_count } => Some(format!(
                "API key verified via /models ({model_count} model(s) visible)."
            )),
            Self::Failed(error) => Some(format!("API key check failed: {error}")),
        }
    }

    const fn is_connected(&self) -> bool {
        matches!(self, Self::NotNeeded | Self::Verified { .. })
    }
}

fn cloud_preset_probe(preset: &ConnectPreset) -> ConnectProbe {
    if preset.local {
        return ConnectProbe::NotNeeded;
    }
    let Some(env) = preset.auth_env else {
        return ConnectProbe::NotNeeded;
    };
    let Some(key) = std::env::var(env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| api::load_openai_compat_api_key(env).ok().flatten())
    else {
        return ConnectProbe::MissingKey(env);
    };
    match api::sync_bridge::run_blocking(api::discover_models_with_bearer(
        preset.base_url,
        &key,
    )) {
        Ok(models) => ConnectProbe::Verified {
            model_count: models.len(),
        },
        Err(error) => ConnectProbe::Failed(error.to_string()),
    }
}

fn refresh_process_provider_catalog() -> Result<(), String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config = runtime::ConfigLoader::default_for(cwd)
        .load()
        .map_err(|error| error.to_string())?;
    let settings_json = config
        .custom_providers_json()
        .unwrap_or_else(|| "[]".to_string());
    if let Some(env_json) = std::env::var(api::CUSTOM_PROVIDERS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let live_json = merge_custom_provider_json_for_live_refresh(&env_json, &settings_json)?;
        return api::refresh_custom_providers_from_json(&live_json)
            .map_err(|error| error.to_string());
    }
    std::env::set_var(api::CUSTOM_PROVIDERS_ENV, &settings_json);
    api::refresh_custom_providers_from_json(&settings_json).map_err(|error| error.to_string())
}

fn merge_custom_provider_json_for_live_refresh(
    env_json: &str,
    settings_json: &str,
) -> Result<String, String> {
    let mut env_entries = parse_provider_json_array(api::CUSTOM_PROVIDERS_ENV, env_json)?;
    let settings_entries = parse_provider_json_array("settings.providers", settings_json)?;
    for entry in settings_entries {
        let Some(name) = entry.get("name").and_then(Value::as_str) else {
            env_entries.push(entry);
            continue;
        };
        if !env_entries
            .iter()
            .any(|existing| existing.get("name").and_then(Value::as_str) == Some(name))
        {
            env_entries.push(entry);
        }
    }
    serde_json::to_string(&Value::Array(env_entries)).map_err(|error| error.to_string())
}

fn parse_provider_json_array(label: &str, raw: &str) -> Result<Vec<Value>, String> {
    match serde_json::from_str::<Value>(raw).map_err(|error| error.to_string())? {
        Value::Array(entries) => Ok(entries),
        _ => Err(format!("{label} must be a JSON array")),
    }
}

/// Match `token` against [`CONNECT_PRESETS`] or treat a `http(s)://` token as a
/// custom OpenAI-compatible endpoint, persisting the provider to user settings —
/// discovering models first. Returns `None` when `token` is neither a known
/// preset nor a URL, so the caller can fall back to its status-check / OAuth
/// paths.
pub(crate) fn connect_preset(token: &str) -> Option<ConnectReport> {
    let lower = token.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return Some(connect_custom_url(token));
    }
    let preset = CONNECT_PRESETS
        .iter()
        .find(|preset| preset.aliases.iter().any(|alias| lower == *alias))?;

    let models = if preset.local {
        let discovered = api::sync_bridge::run_blocking(api::discover_models(preset.base_url));
        if discovered.is_empty() {
            return Some(ConnectReport::Warn(format!(
                "{}: couldn't reach a server at {}\n  Start {} and retry `/connect {}`.",
                preset.label, preset.base_url, preset.label, preset.aliases[0]
            )));
        }
        discovered
    } else {
        preset
            .models
            .iter()
            .map(|model| (*model).to_string())
            .collect()
    };

    let path = match write_user_provider_with_options(
        preset.aliases[0],
        preset.base_url,
        preset.auth_env,
        &models,
        ProviderTokenLimits::default(),
        preset.include_usage,
    ) {
        Ok(path) => path,
        Err(error) => {
            return Some(ConnectReport::Error(format!(
                "{}: failed to write settings: {error}",
                preset.label
            )));
        }
    };

    let refresh_note = match refresh_process_provider_catalog() {
        Ok(()) => "updated this session".to_string(),
        Err(error) => format!("saved, but live catalog refresh failed: {error}. Restart zo"),
    };
    let probe = cloud_preset_probe(preset);
    let first_model = models.first().map_or("<model>", String::as_str);
    let probe_note = probe
        .message()
        .map(|message| format!("\n  {message}"))
        .unwrap_or_default();
    let count_note = if preset.local {
        format!(" ({} model(s) discovered)", models.len())
    } else {
        String::new()
    };
    let message = format!(
        "{}: saved provider to {}{} and {refresh_note}.{}\n  Select it now: /model {}",
        preset.label,
        path.display(),
        count_note,
        probe_note,
        first_model
    );
    Some(if probe.is_connected() {
        ConnectReport::Info(message)
    } else {
        ConnectReport::Warn(message)
    })
}

/// Persist a cloud preset and save the provided API key into Zo's durable
/// credential store. Used by the TUI `/connect` setup modal so users do not
/// need to export an env var manually on every shell.
pub(crate) fn connect_preset_with_api_key(token: &str, api_key: &str) -> ConnectReport {
    let lower = token.to_ascii_lowercase();
    let Some(preset) = CONNECT_PRESETS
        .iter()
        .find(|preset| preset.aliases.iter().any(|alias| lower == *alias))
    else {
        return ConnectReport::Error(format!(
            "'{token}' is not a writable cloud preset. Presets: deepseek, kimi, qwen, nvidia, openrouter."
        ));
    };
    let Some(env_key) = preset.auth_env else {
        return ConnectReport::Error(format!(
            "{} does not use an API key; run /connect {} instead.",
            preset.label, preset.aliases[0]
        ));
    };
    if let Err(error) = api::save_openai_compat_api_key(env_key, api_key) {
        return ConnectReport::Error(format!(
            "{}: failed to save API key: {error}",
            preset.label
        ));
    }
    connect_preset(token).unwrap_or_else(|| {
        ConnectReport::Error(format!(
            "{}: failed to save provider after storing API key",
            preset.label
        ))
    })
}

/// Persist a custom OpenAI-compatible provider from the TUI onboarding wizard.
pub(crate) fn connect_custom_provider(
    name: &str,
    base_url: &str,
    auth_env: Option<&str>,
    api_key: Option<&str>,
    requested_models: &[String],
    token_limits: ProviderTokenLimits,
    include_usage: bool,
) -> ConnectReport {
    let name = name.trim();
    if name.is_empty() {
        return ConnectReport::Error("Custom provider: name is required".to_string());
    }
    let base_url = base_url.trim();
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        return ConnectReport::Error(format!(
            "{name}: base URL must start with http:// or https://"
        ));
    }
    let auth_env = auth_env.map(str::trim).filter(|value| !value.is_empty());
    if auth_env.is_some_and(|env| !valid_auth_env_name(env)) {
        return ConnectReport::Error(format!(
            "{name}: auth env must match [A-Za-z_][A-Za-z0-9_]*"
        ));
    }
    let api_key = api_key.map(str::trim).filter(|value| !value.is_empty());

    let mut models: Vec<String> = requested_models
        .iter()
        .map(|model| model.trim())
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    let discovery_note = if models.is_empty() {
        match discover_custom_provider_models(base_url, auth_env, api_key) {
            CustomModelDiscovery::Found(discovered) => {
                models = discovered;
                format!(" ({} model(s) discovered)", models.len())
            }
            CustomModelDiscovery::Empty => " (no models discovered)".to_string(),
            CustomModelDiscovery::Failed(error) => format!(" (model discovery failed: {error})"),
        }
    } else {
        format!(" ({} model(s) provided)", models.len())
    };

    let path = match write_user_provider_with_options(
        name,
        base_url,
        auth_env,
        &models,
        token_limits,
        Some(include_usage),
    ) {
        Ok(path) => path,
        Err(error) => {
            return ConnectReport::Error(format!("{name}: failed to write settings: {error}"));
        }
    };

    if let (Some(env_key), Some(key)) = (auth_env, api_key) {
        if let Err(error) = api::save_openai_compat_api_key(env_key, key) {
            return ConnectReport::Error(format!(
                "{name}: saved provider to {}, but failed to save API key: {error}",
                path.display()
            ));
        }
    }

    let refresh_note = match refresh_process_provider_catalog() {
        Ok(()) => "updated this session".to_string(),
        Err(error) => format!("saved, but live catalog refresh failed: {error}. Restart zo"),
    };
    if models.is_empty() {
        return ConnectReport::Warn(format!(
            "{name}: saved provider to {}{} and {refresh_note}, but no models are configured.\n  Add model ids to that provider's \"models\" list or rerun /connect custom.",
            path.display(),
            discovery_note,
        ));
    }

    let first_model = &models[0];
    match smoke_test_custom_provider(name, base_url, auth_env, api_key, first_model, include_usage) {
        SmokeTestResult::Passed => ConnectReport::Info(format!(
            "{name}: saved provider to {}{} and {refresh_note}; chat/completions smoke test passed.\n  Select it now: /model {first_model}",
            path.display(),
            discovery_note,
        )),
        SmokeTestResult::Failed(error) => ConnectReport::Warn(format!(
            "{name}: saved provider to {}{} and {refresh_note}, but chat/completions smoke test failed: {error}. Saved anyway.\n  Select it now: /model {first_model}",
            path.display(),
            discovery_note,
        )),
    }
}

#[derive(Debug)]
enum CustomModelDiscovery {
    Found(Vec<String>),
    Empty,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SmokeTestResult {
    Passed,
    Failed(String),
}

fn smoke_test_custom_provider(
    name: &str,
    base_url: &str,
    auth_env: Option<&str>,
    api_key: Option<&str>,
    model: &str,
    include_usage: bool,
) -> SmokeTestResult {
    let config = api::OpenAiCompatConfig::from_user(name, base_url, auth_env, include_usage);
    let client = if let Some(key) = api_key {
        Ok(api::OpenAiCompatClient::new(key.to_string(), config))
    } else {
        api::OpenAiCompatClient::from_env_optional_auth(config)
    };
    let client = match client {
        Ok(client) => client.with_retry_policy(0, Duration::from_millis(50), Duration::from_millis(50)),
        Err(error) => return SmokeTestResult::Failed(error.to_string()),
    };
    let request = api::MessageRequest {
        model: model.to_string(),
        max_tokens: 4,
        messages: vec![api::InputMessage::user_text("Reply with OK.")],
        system: None,
        tools: None,
        tool_choice: None,
        stream: false,
        thinking: None,
        output_config: None,
        effort: None,
        effort_band_ceiling: None,
    };
    let result = api::sync_bridge::run_blocking(async move {
        tokio::time::timeout(Duration::from_secs(20), client.send_message(&request)).await
    });
    match result {
        Ok(Ok(_)) => SmokeTestResult::Passed,
        Ok(Err(error)) => SmokeTestResult::Failed(error.to_string()),
        Err(_) => SmokeTestResult::Failed("chat/completions smoke test timed out".to_string()),
    }
}

fn discover_custom_provider_models(
    base_url: &str,
    auth_env: Option<&str>,
    api_key: Option<&str>,
) -> CustomModelDiscovery {
    if let Some(key) = api_key
        .map(ToOwned::to_owned)
        .or_else(|| auth_env.and_then(stored_or_env_openai_compat_key))
    {
        return match api::sync_bridge::run_blocking(api::discover_models_with_bearer(
            base_url,
            &key,
        )) {
            Ok(models) if models.is_empty() => CustomModelDiscovery::Empty,
            Ok(models) => CustomModelDiscovery::Found(models),
            Err(error) => CustomModelDiscovery::Failed(error.to_string()),
        };
    }

    let models = api::sync_bridge::run_blocking(api::discover_models(base_url));
    if models.is_empty() {
        CustomModelDiscovery::Empty
    } else {
        CustomModelDiscovery::Found(models)
    }
}

fn stored_or_env_openai_compat_key(env: &str) -> Option<String> {
    if !valid_auth_env_name(env) {
        return None;
    }
    std::env::var(env)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| api::load_openai_compat_api_key(env).ok().flatten())
}

/// Persist a custom `http(s)://host/v1` OpenAI-compatible endpoint, probing it
/// for its model list. The provider is named after the host and written keyless
/// (`requires_auth: false`); add an `auth_env` to the entry in settings.json if
/// the endpoint needs a key.
fn connect_custom_url(url: &str) -> ConnectReport {
    let name = provider_name_from_url(url);
    let models = api::sync_bridge::run_blocking(api::discover_models(url));
    let path = match write_user_provider(&name, url, None, &models) {
        Ok(path) => path,
        Err(error) => {
            return ConnectReport::Error(format!("{name}: failed to write settings: {error}"));
        }
    };
    let refresh_note = match refresh_process_provider_catalog() {
        Ok(()) => "updated this session".to_string(),
        Err(error) => format!("saved, but live catalog refresh failed: {error}. Restart zo"),
    };
    if models.is_empty() {
        ConnectReport::Warn(format!(
            "{name}: saved provider to {} and {refresh_note}, but found no models at {url}.\n  Add model ids to that provider's \"models\" list in settings.json (and an \"auth_env\" if it needs a key).",
            path.display()
        ))
    } else {
        ConnectReport::Info(format!(
            "{name}: saved provider to {} ({} model(s) discovered) and {refresh_note}.\n  Select it now: /model {}",
            path.display(),
            models.len(),
            models[0]
        ))
    }
}

/// Derive a provider name from an endpoint URL: its host without scheme, port,
/// or path (e.g. `https://api.together.xyz/v1` -> `api.together.xyz`). Falls
/// back to `custom` when no host can be isolated.
fn provider_name_from_url(url: &str) -> String {
    let host = url
        .split_once("://")
        .map_or(url, |(_, rest)| rest)
        .split(['/', ':'])
        .next()
        .unwrap_or("");
    if host.is_empty() {
        "custom".to_string()
    } else {
        host.to_string()
    }
}

/// Read-merge-write the user `settings.json`, upserting a provider into the
/// `providers` array (deduped by name). Mirrors `save_project_preferences`'s
/// serde-`Value` document handling.
fn write_user_provider(
    name: &str,
    base_url: &str,
    auth_env: Option<&str>,
    models: &[String],
) -> io::Result<PathBuf> {
    write_user_provider_with_options(
        name,
        base_url,
        auth_env,
        models,
        ProviderTokenLimits::default(),
        None,
    )
}

fn write_user_provider_with_options(
    name: &str,
    base_url: &str,
    auth_env: Option<&str>,
    models: &[String],
    token_limits: ProviderTokenLimits,
    include_usage: Option<bool>,
) -> io::Result<PathBuf> {
    if auth_env.is_some_and(|env| !valid_auth_env_name(env)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "auth_env must match [A-Za-z_][A-Za-z0-9_]*",
        ));
    }
    let path = runtime::ConfigLoader::default_for(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    )
    .config_home()
    .join("settings.json");
    upsert_provider_entry_with_options(
        &path,
        name,
        base_url,
        auth_env,
        models,
        token_limits,
        include_usage,
    )?;
    Ok(path)
}

/// Upsert one provider entry into the `providers` array of the settings document
/// at `path` (deduped by name), creating the file/object as needed. Split from
/// [`write_user_provider`] so the merge is testable with an explicit path.
#[cfg(test)]
fn upsert_provider_entry(
    path: &Path,
    name: &str,
    base_url: &str,
    auth_env: Option<&str>,
    models: &[String],
) -> io::Result<()> {
    upsert_provider_entry_with_options(
        path,
        name,
        base_url,
        auth_env,
        models,
        ProviderTokenLimits::default(),
        None,
    )
}

fn upsert_provider_entry_with_options(
    path: &Path,
    name: &str,
    base_url: &str,
    auth_env: Option<&str>,
    models: &[String],
    token_limits: ProviderTokenLimits,
    include_usage: Option<bool>,
) -> io::Result<()> {
    if auth_env.is_some_and(|env| !valid_auth_env_name(env)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "auth_env must match [A-Za-z_][A-Za-z0-9_]*",
        ));
    }
    let mut document = match fs::read_to_string(path) {
        Ok(contents) if contents.trim().is_empty() => Map::new(),
        Ok(contents) => serde_json::from_str::<Value>(&contents)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "settings.json must contain a JSON object",
                )
            })?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Map::new(),
        Err(error) => return Err(error),
    };

    let mut providers: Vec<Value> = document
        .get("providers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    providers.retain(|entry| entry.get("name").and_then(Value::as_str) != Some(name));

    let mut entry = Map::new();
    entry.insert("name".to_string(), Value::String(name.to_string()));
    entry.insert("base_url".to_string(), Value::String(base_url.to_string()));
    if let Some(env) = auth_env {
        entry.insert("auth_env".to_string(), Value::String(env.to_string()));
    }
    entry.insert(
        "requires_auth".to_string(),
        Value::Bool(auth_env.is_some()),
    );
    if let Some(context_window) = token_limits.context_window.filter(|&value| value > 0) {
        entry.insert(
            "context_window".to_string(),
            Value::Number(serde_json::Number::from(context_window)),
        );
    }
    if let Some(max_output_tokens) = token_limits.max_output_tokens.filter(|&value| value > 0) {
        entry.insert(
            "max_output_tokens".to_string(),
            Value::Number(serde_json::Number::from(max_output_tokens)),
        );
    }
    if let Some(include_usage) = include_usage {
        entry.insert("include_usage".to_string(), Value::Bool(include_usage));
    }
    entry.insert(
        "models".to_string(),
        Value::Array(
            models
                .iter()
                .map(|model| Value::String(model.clone()))
                .collect(),
        ),
    );
    providers.push(Value::Object(entry));
    document.insert("providers".to_string(), Value::Array(providers));

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_json::to_string_pretty(&document).map_err(io::Error::other)?;
    fs::write(path, rendered)?;
    Ok(())
}

impl Provider {
    fn matches(&self, token: &str) -> bool {
        self.aliases
            .iter()
            .any(|alias| token.eq_ignore_ascii_case(alias))
    }
}

pub(super) fn connect(ctx: &mut DispatchCtx<'_>, provider: Option<&str>) -> CommandOutput {
    let Some(prov) = provider else {
        open_provider_modal(ctx, "connect");
        return CommandOutput::Quiet;
    };
    let lower = prov.to_ascii_lowercase();
    if matches!(lower.as_str(), "claude" | "anthropic") {
        return CommandOutput::info("Claude: connected via OAuth. Use /login to re-authenticate.");
    }
    if matches!(lower.as_str(), "custom" | "openai-compatible" | "openai-compatible-custom") {
        ctx.app.open_custom_provider_modal();
        return CommandOutput::Quiet;
    }
    // Writable OpenAI-compatible presets (Ollama / LM Studio / DeepSeek / Kimi /
    // Qwen / NVIDIA / OpenRouter) persist a provider into settings.json; checked before the OAuth /
    // env status providers below.
    if let Some(report) = connect_preset(prov) {
        return match report {
            ConnectReport::Info(message) => CommandOutput::info(message),
            ConnectReport::Warn(message) => CommandOutput::warn(message),
            ConnectReport::Error(message) => CommandOutput::error(message),
        };
    }
    let Some(p) = PROVIDERS.iter().find(|p| p.matches(&lower)) else {
        return CommandOutput::error(format!(
            "Unknown provider: {prov}\nAvailable: deepseek, kimi, qwen, nvidia, openrouter, ollama, lmstudio, openai, google, xai, claude\nOr pass an OpenAI-compatible endpoint URL: /connect https://host/v1"
        ));
    };
    if p.connection.is_connected() {
        CommandOutput::info(format!(
            "{}: ✓ connected ({} is set)\nUse /model to select a model from this provider.",
            p.label,
            p.connection.connected_detail()
        ))
    } else {
        CommandOutput::warn(format!(
            "{}: ✗ not connected\n\n{}\n\nThen restart zo or use:\n  zo --model {}",
            p.label,
            p.connection.setup_hint(),
            p.model_hint
        ))
    }
}

/// Open the GUI provider picker shared by no-argument `/login` and
/// `/connect`. The selected provider is re-submitted to the command that opened
/// the modal (`<command>:<provider>` → `/<command> <provider>`), so `/login`
/// starts OAuth while `/connect` runs its preset/status path.
fn open_provider_modal(ctx: &mut DispatchCtx<'_>, command: &str) {
    let mut labels = vec![
        "Claude   —  Anthropic OAuth".to_string(),
        "ChatGPT  —  OpenAI subscription".to_string(),
        "Gemini   —  Google OAuth".to_string(),
    ];
    let mut ids: Vec<String> = ["claude", "openai", "google"]
        .into_iter()
        .map(|provider| format!("{command}:{provider}"))
        .collect();
    // `/connect` also sets up OpenAI-compatible local/cloud providers; list them
    // so they are discoverable without typing the alias. Each re-dispatches as
    // `/connect <id>` through the same selection path.
    if command == "connect" {
        for (id, label) in [
            ("nvidia", "NVIDIA  —  NIM free endpoint (API key)"),
            ("openrouter", "OpenRouter — OpenAI-compatible router (API key)"),
            ("deepseek", "DeepSeek  —  cloud (models + API key)"),
            ("kimi", "Kimi      —  Moonshot (models + API key)"),
            ("qwen", "Qwen      —  DashScope (models + API key)"),
        ] {
            ids.push(format!("connect-key:{id}"));
            labels.push(label.to_string());
        }
        for (id, label) in [
            ("ollama", "Ollama    —  local (auto-discovered)"),
            ("lmstudio", "LM Studio —  local (auto-discovered)"),
        ] {
            ids.push(format!("connect:{id}"));
            labels.push(label.to_string());
        }
        ids.push("connect-custom:openai-compatible".to_string());
        labels.push("Custom   —  OpenAI-compatible endpoint wizard".to_string());
    }
    let title = if command == "connect" {
        "Connect — select provider"
    } else {
        "Log in — select provider"
    };
    ctx.app.open_login_modal(title, labels, ids);
}

pub(super) fn login(ctx: &mut DispatchCtx<'_>, provider: Option<&str>) -> CommandOutput {
    let Some(prov) = provider else {
        open_provider_modal(ctx, "login");
        return CommandOutput::Quiet;
    };
    let opening = format!("Login — Opening browser for {prov} OAuth...");
    match crate::auth::run_login_provider(prov) {
        Ok(()) => CommandOutput::info(opening).and_report(
            SystemLevel::Info,
            format!(
                "{prov} OAuth login successful!\n\n  Use /model to switch models.\n\n  Other providers: /login openai | /login google | /login xai"
            ),
        ),
        Err(e) => CommandOutput::info(opening)
            .and_report(SystemLevel::Error, format!("Login failed: {e}")),
    }
}

pub(super) fn logout() -> CommandOutput {
    let claude = api::oauth_store::clear_oauth_credentials();
    let openai = api::oauth_store::clear_openai_oauth();
    let google = api::oauth_store::clear_google_code_assist_oauth();
    match (claude, openai, google) {
        (Ok(()), Ok(()), Ok(())) => CommandOutput::info(
            "Logout\n  Cleared saved Claude, ChatGPT, and Google Gemini OAuth credentials.\n  Note: env vars and Google ADC/gcloud credentials are still active if set.",
        ),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
            CommandOutput::error(format!("Logout failed: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConnectReport, ProviderTokenLimits, SmokeTestResult, connect_custom_provider, connect_preset,
        provider_name_from_url, smoke_test_custom_provider, upsert_provider_entry,
        upsert_provider_entry_with_options,
    };

    struct EnvVarGuard {
        key: &'static str,
        old: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let old = std::env::var(key).ok();
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, old }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.old.as_deref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn temp_config_home(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zo-connect-config-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp config home");
        dir
    }

    fn temp_settings_path(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "zo-connect-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join("settings.json")
    }

    #[test]
    fn connect_deepseek_refreshes_model_catalog_without_restart() {
        let _lock = crate::test_env_lock();
        let config_home = temp_config_home("deepseek-live-refresh");
        let config_home_str = config_home.to_str().expect("utf8 config home").to_string();
        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&config_home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", None);
        let _home = EnvVarGuard::set("HOME", Some(&config_home_str));
        let _custom_env = EnvVarGuard::set(api::CUSTOM_PROVIDERS_ENV, None);
        let _deepseek_key = EnvVarGuard::set("DEEPSEEK_API_KEY", None);

        api::refresh_custom_providers_from_json("[]").expect("clear live catalog");
        assert!(
            api::custom_provider_catalog().is_empty(),
            "test starts with an empty live custom-provider catalog"
        );

        let report = connect_preset("deepseek").expect("deepseek preset exists");
        let message = match report {
            ConnectReport::Warn(message) => message,
            ConnectReport::Info(message) | ConnectReport::Error(message) => {
                panic!("missing API key should warn after saving, got: {message}")
            }
        };
        assert!(
            message.contains("Select it now: /model deepseek-chat"),
            "connect should advertise immediate /model availability: {message}"
        );
        assert!(
            message.contains("API key not found in this process"),
            "connect should still report that the current process lacks the key: {message}"
        );
        assert!(
            !message.contains("Restart zo"),
            "successful live refresh must not require restart: {message}"
        );

        let catalog = api::custom_provider_catalog();
        let deepseek = catalog
            .iter()
            .find(|(provider, _)| *provider == "deepseek")
            .expect("/connect deepseek must refresh the live model catalog");
        assert_eq!(
            deepseek.1,
            vec![
                "deepseek-chat".to_string(),
                "deepseek-reasoner".to_string()
            ]
        );

        let settings_path = config_home.join("settings.json");
        let settings: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&settings_path).expect("settings file written"),
        )
        .expect("settings json");
        assert_eq!(settings["providers"][0]["name"], "deepseek");

        api::refresh_custom_providers_from_json("[]").expect("restore empty live catalog");
        std::fs::remove_dir_all(config_home).ok();
    }

    #[test]
    fn connect_deepseek_preserves_custom_providers_env_override() {
        let _lock = crate::test_env_lock();
        let config_home = temp_config_home("deepseek-env-override");
        let config_home_str = config_home.to_str().expect("utf8 config home").to_string();
        let env_json = r#"[{"name":"env-only","base_url":"http://env.example/v1","models":["env-model"],"requires_auth":false}]"#;
        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&config_home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", None);
        let _home = EnvVarGuard::set("HOME", Some(&config_home_str));
        let _custom_env = EnvVarGuard::set(api::CUSTOM_PROVIDERS_ENV, Some(env_json));
        let _deepseek_key = EnvVarGuard::set("DEEPSEEK_API_KEY", None);

        api::refresh_custom_providers_from_json("[]").expect("clear live catalog");
        let report = connect_preset("deepseek").expect("deepseek preset exists");
        assert!(
            matches!(report, ConnectReport::Warn(_)),
            "missing API key should warn while preserving env override"
        );
        assert_eq!(
            std::env::var(api::CUSTOM_PROVIDERS_ENV).as_deref(),
            Ok(env_json),
            "/connect must not clobber an operator-provided ZO_CUSTOM_PROVIDERS override"
        );

        let catalog = api::custom_provider_catalog();
        assert!(
            catalog
                .iter()
                .any(|(provider, models)| *provider == "env-only" && models == &["env-model"]),
            "live refresh must preserve env-provided custom models: {catalog:?}"
        );
        assert!(
            catalog.iter().any(|(provider, models)| {
                *provider == "deepseek" && models.iter().any(|model| model == "deepseek-chat")
            }),
            "live refresh should add the newly connected DeepSeek provider without removing env entries: {catalog:?}"
        );

        api::refresh_custom_providers_from_json("[]").expect("restore empty live catalog");
        std::fs::remove_dir_all(config_home).ok();
    }

    #[test]
    fn upsert_writes_provider_entry_with_expected_shape() {
        let path = temp_settings_path("shape");
        let _ = std::fs::remove_file(&path);
        upsert_provider_entry(
            &path,
            "deepseek",
            "https://api.deepseek.com",
            Some("DEEPSEEK_API_KEY"),
            &["deepseek-chat".to_string()],
        )
        .expect("write");

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        let providers = value["providers"].as_array().expect("providers array");
        assert_eq!(providers.len(), 1);
        let entry = &providers[0];
        assert_eq!(entry["name"], "deepseek");
        assert_eq!(entry["base_url"], "https://api.deepseek.com");
        assert_eq!(entry["auth_env"], "DEEPSEEK_API_KEY");
        assert_eq!(entry["requires_auth"], true);
        assert_eq!(entry["models"][0], "deepseek-chat");
    }




    #[test]
    fn smoke_test_custom_provider_calls_chat_completions() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = stream.read(&mut buf).expect("read");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request_text = String::from_utf8_lossy(&request);
            assert!(
                request_text.starts_with("POST /v1/chat/completions "),
                "request was: {request_text}"
            );
            let request_text_lower = request_text.to_ascii_lowercase();
            assert!(
                request_text_lower.contains("authorization: bearer sk-test"),
                "request should carry bearer auth: {request_text}"
            );
            let body = r#"{"id":"chatcmpl-test","model":"model-a","choices":[{"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        });

        let result = smoke_test_custom_provider(
            "test-provider",
            &format!("http://{addr}/v1"),
            Some("TEST_API_KEY"),
            Some("sk-test"),
            "model-a",
            false,
        );
        server.join().expect("server join");
        assert_eq!(result, SmokeTestResult::Passed);
    }

    #[test]
    fn connect_custom_provider_rejects_invalid_auth_env_name() {
        let report = connect_custom_provider(
            "bad-env",
            "https://example.com/v1",
            Some("FOO=bar"),
            Some("sk-secret"),
            &["model-a".to_string()],
            ProviderTokenLimits::default(),
            false,
        );
        match report {
            ConnectReport::Error(message) => {
                assert!(message.contains("auth env must match"), "message: {message}");
            }
            ConnectReport::Info(message) | ConnectReport::Warn(message) => {
                panic!("invalid auth env should fail before saving: {message}");
            }
        }
    }

    #[test]
    fn upsert_rejects_invalid_auth_env_name() {
        let path = temp_settings_path("invalid-auth-env");
        let error = upsert_provider_entry_with_options(
            &path,
            "bad-env",
            "https://example.com/v1",
            Some("FOO=bar"),
            &["model-a".to_string()],
            ProviderTokenLimits::default(),
            Some(false),
        )
        .expect_err("invalid auth env should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn upsert_can_write_include_usage_override() {
        let path = temp_settings_path("include-usage");
        let _ = std::fs::remove_file(&path);
        upsert_provider_entry_with_options(
            &path,
            "nvidia-nim",
            "https://integrate.api.nvidia.com/v1",
            Some("NVIDIA_API_KEY"),
            &["meta/llama-3.1-8b-instruct".to_string(), "z-ai/glm-5.2".to_string()],
            ProviderTokenLimits::default(),
            Some(false),
        )
        .expect("write");

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        let entry = &value["providers"][0];
        assert_eq!(entry["name"], "nvidia-nim");
        assert_eq!(entry["base_url"], "https://integrate.api.nvidia.com/v1");
        assert_eq!(entry["auth_env"], "NVIDIA_API_KEY");
        assert_eq!(entry["requires_auth"], true);
        assert_eq!(entry["include_usage"], false);
        assert_eq!(entry["models"][1], "z-ai/glm-5.2");
    }

    #[test]
    fn upsert_can_write_context_and_max_output_overrides() {
        let path = temp_settings_path("context-max-output");
        let _ = std::fs::remove_file(&path);
        upsert_provider_entry_with_options(
            &path,
            "xai-custom",
            "https://api.x.ai/v1",
            Some("XAI_API_KEY"),
            &["grok-4.5".to_string()],
            ProviderTokenLimits {
                context_window: Some(256_000),
                max_output_tokens: Some(32_000),
            },
            Some(false),
        )
        .expect("write");

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        let entry = &value["providers"][0];
        assert_eq!(entry["context_window"], 256_000);
        assert_eq!(entry["max_output_tokens"], 32_000);
        assert_eq!(entry["models"][0], "grok-4.5");
    }

    #[test]
    fn upsert_dedupes_by_name_and_keeps_other_providers() {
        let path = temp_settings_path("dedupe");
        let _ = std::fs::remove_file(&path);
        // Two distinct providers, then re-write the first: it must replace, not
        // duplicate, and must not disturb the other.
        upsert_provider_entry(
            &path,
            "deepseek",
            "https://api.deepseek.com",
            Some("DEEPSEEK_API_KEY"),
            &["deepseek-chat".to_string()],
        )
        .expect("write cloud");
        upsert_provider_entry(
            &path,
            "ollama",
            "http://localhost:11434/v1",
            None,
            &["llama3.1".to_string()],
        )
        .expect("write local");
        upsert_provider_entry(
            &path,
            "deepseek",
            "https://api.deepseek.com",
            Some("DEEPSEEK_API_KEY"),
            &["deepseek-reasoner".to_string()],
        )
        .expect("rewrite cloud");

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        let providers = value["providers"].as_array().expect("providers array");
        assert_eq!(providers.len(), 2, "rewrite must dedupe by name");

        let deepseek = providers
            .iter()
            .find(|entry| entry["name"] == "deepseek")
            .expect("deepseek entry");
        assert_eq!(
            deepseek["models"][0], "deepseek-reasoner",
            "rewrite must replace the model list"
        );
        // Keyless local provider omits auth_env and is not required.
        let ollama = providers
            .iter()
            .find(|entry| entry["name"] == "ollama")
            .expect("ollama entry");
        assert!(ollama.get("auth_env").is_none());
        assert_eq!(ollama["requires_auth"], false);
    }

    #[test]
    fn provider_name_is_derived_from_url_host() {
        assert_eq!(
            provider_name_from_url("https://api.together.xyz/v1"),
            "api.together.xyz"
        );
        assert_eq!(
            provider_name_from_url("http://localhost:1234/v1"),
            "localhost"
        );
        assert_eq!(provider_name_from_url("https://"), "custom");
    }
}
