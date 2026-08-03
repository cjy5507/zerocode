//! Authentication / provider commands: login, logout, connect.
//!
//! Provider metadata lives in one [`PROVIDERS`] table so `/connect` setup hints
//! stay in sync instead of drifting across inline `match` arms.

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use runtime::message_stream::SystemLevel;

use super::context::DispatchCtx;
use super::output::CommandOutput;
use super::provider_store::{
    self, ModelWrite, ProviderDraft, ProviderTokenLimits, valid_auth_env_name,
};

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
    Anthropic,
    OpenAi,
    Google,
}

impl ProviderConnection {
    fn is_connected(self) -> bool {
        match self {
            Self::Env { env_key } => {
                env_non_empty(env_key) || api::load_openai_compat_api_key(env_key).ok().flatten().is_some()
            }
            Self::Anthropic => api::oauth_store::load_oauth_credentials()
                .ok()
                .flatten()
                .is_some(),
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
            Self::Anthropic => "saved Anthropic OAuth",
            Self::OpenAi => "OPENAI_API_KEY or saved ChatGPT OAuth",
            Self::Google => "saved Gemini OAuth, GOOGLE_API_KEY, or Google ADC",
        }
    }

    /// Whether zo holds a credential it can actually clear for this account.
    fn is_disconnectable(self) -> bool {
        match self {
            Self::Anthropic | Self::OpenAi => true,
            // Cleared only when a key was saved into zo's own store; a shell
            // export or ADC login is not zo's to remove.
            Self::Env { env_key } => api::load_openai_compat_api_key(env_key)
                .ok()
                .flatten()
                .is_some(),
            Self::Google => api::google_code_assist_oauth_present(),
        }
    }

    /// Clear whatever zo saved for this account. `Ok(false)` means there was
    /// nothing of zo's to clear.
    fn clear_saved_credentials(self) -> io::Result<bool> {
        match self {
            Self::Anthropic => {
                let had = api::oauth_store::load_oauth_credentials()?.is_some();
                api::oauth_store::clear_oauth_credentials()?;
                Ok(had)
            }
            Self::OpenAi => {
                let had = openai_oauth_present();
                api::oauth_store::clear_openai_oauth()?;
                Ok(had)
            }
            Self::Google => {
                let had = api::google_code_assist_oauth_present();
                api::oauth_store::clear_google_code_assist_oauth()?;
                Ok(had)
            }
            Self::Env { env_key } => api::delete_openai_compat_api_key(env_key),
        }
    }

    fn setup_hint(self) -> String {
        match self {
            Self::Env { env_key } => format!(
                "Set the API key in your shell before starting zo:\n  \
                 export {env_key}=your-key-here"
            ),
            Self::Anthropic => "Run `/login claude` for Anthropic OAuth.".to_string(),
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

fn openai_oauth_present() -> bool {
    api::oauth_store::load_openai_oauth()
        .ok()
        .flatten()
        .is_some()
}

/// Single source of truth for connectable providers.
const PROVIDERS: &[Provider] = &[
    Provider {
        aliases: &["claude", "anthropic"],
        label: "Claude",
        connection: ProviderConnection::Anthropic,
        model_hint: "opus",
    },
    Provider {
        aliases: &["openai", "gpt", "codex"],
        label: "OpenAI",
        connection: ProviderConnection::OpenAi,
        model_hint: api::OPENAI_LATEST_MODEL_ALIAS,
    },
    Provider {
        aliases: &["google", "gemini"],
        label: "Google",
        connection: ProviderConnection::Google,
        model_hint: api::GOOGLE_LATEST_MODEL_ALIAS,
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
        models: api::DEEPSEEK_PRESET_MODELS,
        local: false,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["kimi", "moonshot"],
        label: "Kimi (Moonshot)",
        base_url: "https://api.moonshot.ai/v1",
        auth_env: Some("MOONSHOT_API_KEY"),
        models: api::KIMI_PRESET_MODELS,
        local: false,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["qwen", "dashscope"],
        label: "Qwen (DashScope)",
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        auth_env: Some("DASHSCOPE_API_KEY"),
        models: api::QWEN_PRESET_MODELS,
        local: false,
        include_usage: None,
    },
    ConnectPreset {
        aliases: &["nvidia", "nvidia-nim", "nim"],
        label: "NVIDIA NIM",
        base_url: "https://integrate.api.nvidia.com/v1",
        auth_env: Some("NVIDIA_API_KEY"),
        models: api::NVIDIA_PRESET_MODELS,
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

/// Re-read the merged global settings and make them the live provider catalog,
/// so a registration or a deletion takes effect in this session instead of at
/// the next restart. Ownership of the process bridge lives in
/// [`crate::custom_provider_env`].
fn refresh_process_provider_catalog() -> Result<(), String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let config = runtime::ConfigLoader::default_for(cwd)
        .load()
        .map_err(|error| error.to_string())?;
    let settings_json = config
        .custom_providers_json()
        .unwrap_or_else(|| "[]".to_string());
    crate::custom_provider_env::publish(&settings_json)
}

/// How a settings mutation landed, phrased for the report line.
pub(super) fn live_refresh_note() -> String {
    match refresh_process_provider_catalog() {
        Ok(()) if crate::custom_provider_env::mutations_apply_live() => {
            "updated this session".to_string()
        }
        // An operator export pins the catalog by design; say so instead of
        // claiming a live update that the override silently overrode.
        Ok(()) => format!(
            "saved, but {} is set in this shell and pins the live catalog. Unset it or restart zo",
            api::CUSTOM_PROVIDERS_ENV
        ),
        Err(error) => format!("saved, but live catalog refresh failed: {error}. Restart zo"),
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

    let refresh_note = live_refresh_note();
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

/// One custom OpenAI-compatible provider to register or edit.
///
/// A struct rather than a parameter list because the same shape crosses three
/// boundaries — the TUI wizard, the headless `/connect custom` path, and the
/// `session.connect_custom_provider` RPC — and a positional `Option<&str>` in
/// the middle of eight arguments is exactly where those three drift apart.
pub(crate) struct CustomProviderRequest<'a> {
    pub(crate) name: &'a str,
    pub(crate) base_url: &'a str,
    pub(crate) auth_env: Option<&'a str>,
    pub(crate) api_key: Option<&'a str>,
    pub(crate) models: &'a [String],
    pub(crate) token_limits: ProviderTokenLimits,
    pub(crate) include_usage: bool,
    /// Whether this endpoint's models accept OpenAI's `reasoning_effort`. Off
    /// means `/effort` and the Smart dynamic band cannot reach them at all.
    pub(crate) supports_reasoning_effort: bool,
    /// `true` when the wizard was opened on an existing provider, so the fields
    /// replace what is stored instead of merging into it.
    pub(crate) edit_existing: bool,
}

/// Persist a custom OpenAI-compatible provider from the TUI onboarding wizard.
pub(crate) fn connect_custom_provider(request: &CustomProviderRequest<'_>) -> ConnectReport {
    let name = request.name.trim();
    if name.is_empty() {
        return ConnectReport::Error("Custom provider: name is required".to_string());
    }
    let base_url = request.base_url.trim();
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        return ConnectReport::Error(format!(
            "{name}: base URL must start with http:// or https://"
        ));
    }
    let auth_env = request
        .auth_env
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if auth_env.is_some_and(|env| !valid_auth_env_name(env)) {
        return ConnectReport::Error(format!(
            "{name}: auth env must match [A-Za-z_][A-Za-z0-9_]*"
        ));
    }
    let api_key = request
        .api_key
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let mut models: Vec<String> = request
        .models
        .iter()
        .map(|model| model.trim())
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    // An edit that cleared the model list means "serve nothing here" and must be
    // honored; only a fresh registration falls back to probing /models.
    let discovery_note = if models.is_empty() && !request.edit_existing {
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

    let path = provider_store::global_settings_path();
    let draft = ProviderDraft::connect(name, base_url, auth_env, &models)
        .with_token_limits(request.token_limits)
        .with_include_usage(Some(request.include_usage))
        .with_reasoning_effort(Some(request.supports_reasoning_effort))
        .with_model_write(if request.edit_existing {
            ModelWrite::Replace
        } else {
            ModelWrite::Union
        });
    if let Err(error) = provider_store::upsert_provider(&path, &draft) {
        return ConnectReport::Error(format!("{name}: failed to write settings: {error}"));
    }

    if let (Some(env_key), Some(key)) = (auth_env, api_key) {
        if let Err(error) = api::save_openai_compat_api_key(env_key, key) {
            return ConnectReport::Error(format!(
                "{name}: saved provider to {}, but failed to save API key: {error}",
                path.display()
            ));
        }
    }

    let refresh_note = live_refresh_note();
    if models.is_empty() {
        return ConnectReport::Warn(format!(
            "{name}: saved provider to {}{} and {refresh_note}, but no models are configured.\n  Add model ids to that provider's \"models\" list or rerun /connect custom.",
            path.display(),
            discovery_note,
        ));
    }

    let first_model = &models[0];
    match smoke_test_custom_provider(
        name,
        base_url,
        auth_env,
        api_key,
        first_model,
        request.include_usage,
    ) {
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
    let refresh_note = live_refresh_note();
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

/// Merge one provider into the user-global `settings.json` and return the file
/// it landed in, so every report can name the exact (global) path it wrote.
///
/// Merge — not replace — is the whole point: a provider name is the account, and
/// its `auth_env` is that account's one key, so a second `/connect` under the
/// same name adds models to the account instead of resetting it. The rules live
/// in [`provider_store::upsert_provider`].
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
    let path = provider_store::global_settings_path();
    provider_store::upsert_provider(
        &path,
        &ProviderDraft::connect(name, base_url, auth_env, models)
            .with_token_limits(token_limits)
            .with_include_usage(include_usage),
    )?;
    Ok(path)
}

/// Account rows for the `/providers` manager: every subscription / OAuth
/// connection, with its live status.
///
/// Built from the same [`PROVIDERS`] table `/connect <name>` status-checks, so
/// the manager and the status line can never disagree about what is connected.
pub(crate) fn account_rows() -> Vec<zo_cli::tui::modals::ProviderAccountRow> {
    PROVIDERS
        .iter()
        .map(|provider| zo_cli::tui::modals::ProviderAccountRow {
            id: provider.aliases[0].to_string(),
            label: provider.label.to_string(),
            detail: provider.connection.connected_detail().to_string(),
            connected: provider.connection.is_connected(),
            disconnectable: provider.connection.is_disconnectable(),
        })
        .collect()
}

/// Forget one account's saved credentials.
///
/// Deliberately per-account: `/logout` used to clear all three OAuth stores at
/// once with no confirmation, which is not what "disconnect this one" means.
/// Only credentials zo owns are cleared — an exported env var or a `gcloud` ADC
/// login belongs to the shell, so the report says where it actually lives
/// instead of pretending to have removed it.
pub(crate) fn disconnect_account(id: &str) -> super::providers::ManagerOutcome {
    let Some(provider) = PROVIDERS.iter().find(|provider| provider.matches(id)) else {
        return super::providers::ManagerOutcome {
            report: ConnectReport::Warn(format!("{id}: not a known account.")),
            refresh_list: true,
        };
    };
    let report = match provider.connection.clear_saved_credentials() {
        Ok(true) => ConnectReport::Info(format!(
            "{}: disconnected. Reconnect with /login {}.",
            provider.label, provider.aliases[0]
        )),
        Ok(false) => ConnectReport::Warn(format!(
            "{}: nothing saved to clear — it is connected through {}.",
            provider.label,
            provider.connection.connected_detail()
        )),
        Err(error) => ConnectReport::Error(format!(
            "{}: failed to clear credentials: {error}",
            provider.label
        )),
    };
    super::providers::ManagerOutcome {
        report,
        refresh_list: true,
    }
}

impl Provider {
    fn matches(&self, token: &str) -> bool {
        self.aliases
            .iter()
            .any(|alias| token.eq_ignore_ascii_case(alias))
    }
}

pub(super) fn connect(ctx: &mut DispatchCtx<'_>, provider: Option<&str>) -> CommandOutput {
    // `/connect`, `/login`, `/logout` and `/providers` all answer one question,
    // so with no argument they all open the one manager. The argument forms are
    // unchanged, which is what keeps `/connect deepseek` in muscle memory.
    let Some(prov) = provider else {
        return super::providers::providers(ctx);
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
        let output = match report {
            ConnectReport::Info(message) => CommandOutput::info(message),
            ConnectReport::Warn(message) => CommandOutput::warn(message),
            ConnectReport::Error(message) => CommandOutput::error(message),
        };
        return reopen_manager_if_requested(ctx, output);
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
pub(crate) fn open_add_provider_picker(app: &mut zo_cli::tui::App) {
    open_provider_modal_on(app, "connect");
}

/// Which providers have a credential saved in zo's own store.
///
/// Every arm is a plain read of the credential file. Opening a picker is a
/// glance at state, so this must not stall the UI or change anything, and two
/// tempting probes are deliberately absent:
///
/// * The Claude Code keychain. [`api::read_claude_code_keychain_session`] forks
///   `security`, and on an expired blob it runs a token refresh with a
///   30-second budget, writes the result back to the keychain, and mirrors it
///   into zo's store — an authentication lifecycle, not a status read, on the
///   thread painting the frame. A Claude Code-only session therefore shows no
///   badge; under-claiming beats freezing the UI and rotating tokens.
/// * `GOOGLE_ACCESS_TOKEN`. A transient shell export is not a saved
///   credential, and counting it badged Gemini `[saved]` for someone who had
///   never logged in.
///
/// The badge reads `saved`, not `connected`, for the same reason: nothing here
/// contacts the provider or checks expiry.
fn saved_oauth_providers() -> [bool; 3] {
    let anthropic = api::oauth_store::load_oauth_credentials()
        .ok()
        .flatten()
        .is_some();
    let openai = api::oauth_store::load_openai_oauth().ok().flatten().is_some();
    let google = api::google_code_assist_oauth_present();
    [anthropic, openai, google]
}

fn open_provider_modal_on(app: &mut zo_cli::tui::App, command: &str) {
    use zo_cli::tui::modals::{ChoiceBadge, ChoiceRow};

    const SIGN_IN: &str = "Sign in";
    const API_KEY: &str = "API key";
    const ON_MACHINE: &str = "On this machine";

    let saved = saved_oauth_providers();
    let mut rows: Vec<ChoiceRow> = [
        ("Claude", "Anthropic OAuth"),
        ("ChatGPT", "OpenAI subscription"),
        ("Gemini", "Google OAuth"),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (label, description))| {
        let row = ChoiceRow::new(label).describe(description).in_group(SIGN_IN);
        if saved[index] {
            row.with_badge(ChoiceBadge::Saved)
        } else {
            row
        }
    })
    .collect();
    let mut ids: Vec<String> = ["claude", "openai", "google"]
        .into_iter()
        .map(|provider| format!("{command}:{provider}"))
        .collect();
    // `/connect` also sets up OpenAI-compatible local/cloud providers; list them
    // so they are discoverable without typing the alias. Each re-dispatches as
    // `/connect <id>` through the same selection path.
    if command == "connect" {
        for (id, label, description) in [
            ("nvidia", "NVIDIA", "NIM free endpoint"),
            ("openrouter", "OpenRouter", "OpenAI-compatible router"),
            ("deepseek", "DeepSeek", "cloud models"),
            ("kimi", "Kimi", "Moonshot"),
            ("qwen", "Qwen", "DashScope"),
        ] {
            ids.push(format!("connect-key:{id}"));
            rows.push(
                ChoiceRow::new(label)
                    .describe(description)
                    .with_badge(ChoiceBadge::NeedsKey)
                    .in_group(API_KEY),
            );
        }
        // The wizard closes the API-key section it belongs to. Appending it
        // after the local providers would leave the list ordered
        // API key → On this machine → API key.
        ids.push("connect-custom:openai-compatible".to_string());
        rows.push(
            ChoiceRow::new("Custom")
                .describe("OpenAI-compatible endpoint wizard")
                .in_group(API_KEY),
        );
        for (id, label) in [("ollama", "Ollama"), ("lmstudio", "LM Studio")] {
            ids.push(format!("connect:{id}"));
            rows.push(
                ChoiceRow::new(label)
                    .describe("auto-discovered")
                    .with_badge(ChoiceBadge::Local)
                    .in_group(ON_MACHINE),
            );
        }
    }
    let title = if command == "connect" {
        "Connect — select provider"
    } else {
        "Log in — select provider"
    };
    app.open_login_modal_rows(title, rows, ids);
}

pub(super) fn login(ctx: &mut DispatchCtx<'_>, provider: Option<&str>) -> CommandOutput {
    let Some(prov) = provider else {
        return super::providers::providers(ctx);
    };
    let opening = format!("Login — Opening browser for {prov} OAuth...");
    let output = match crate::auth::run_login_provider(prov) {
        Ok(()) => CommandOutput::info(opening).and_report(
            SystemLevel::Success,
            format!(
                "{prov} OAuth login successful!\n\n  Use /model to switch models.\n\n  Manage every connection: /providers"
            ),
        ),
        Err(e) => CommandOutput::info(opening)
            .and_report(SystemLevel::Error, format!("Login failed: {e}")),
    };
    reopen_manager_if_requested(ctx, output)
}

/// Land back on the manager when this flow was started from it, so adding two
/// providers in a row does not mean retyping the command between them.
fn reopen_manager_if_requested(
    ctx: &mut DispatchCtx<'_>,
    output: CommandOutput,
) -> CommandOutput {
    if ctx.app.take_provider_manager_return() {
        let reopened = super::providers::providers(ctx);
        debug_assert!(matches!(reopened, CommandOutput::Quiet));
    }
    output
}

/// `/logout` with no argument opens the manager, where `d` disconnects exactly
/// the account under the cursor. The old blanket behaviour — clear every saved
/// OAuth credential at once, with no confirmation — is still available, but now
/// has to be asked for by name.
pub(super) fn logout(ctx: &mut DispatchCtx<'_>, scope: Option<&str>) -> CommandOutput {
    match scope.map(str::trim).filter(|scope| !scope.is_empty()) {
        None => super::providers::providers(ctx),
        Some(scope) if scope.eq_ignore_ascii_case("all") => logout_all(),
        Some(other) => CommandOutput::error(format!(
            "Unknown /logout argument: {other}\n  /logout        open the provider manager and disconnect one account\n  /logout all    clear every saved OAuth credential"
        )),
    }
}

fn logout_all() -> CommandOutput {
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
        ConnectReport, CustomProviderRequest, ProviderTokenLimits, SmokeTestResult,
        connect_custom_provider, connect_preset,
        provider_name_from_url, saved_oauth_providers, smoke_test_custom_provider,
        write_user_provider_with_options,
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

    /// Point every canonical config root at one throwaway directory and hand
    /// back the guards plus the global `settings.json` the connect layer must
    /// write to. Holding the guards keeps the redirect alive for the test body.
    fn isolated_config_home(tag: &str) -> (Vec<EnvVarGuard>, std::path::PathBuf) {
        let home = temp_config_home(tag);
        let home_str = home.to_str().expect("utf8 config home").to_string();
        let guards = vec![
            EnvVarGuard::set("ZO_CONFIG_HOME", Some(&home_str)),
            EnvVarGuard::set("ZO_HOME", None),
            EnvVarGuard::set("HOME", Some(&home_str)),
        ];
        let settings = home.join("settings.json");
        (guards, settings)
    }

    fn read_settings(path: &std::path::Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).expect("settings written"))
            .expect("settings json")
    }

    /// The badge claims a credential is *saved*, so the probe must read saved
    /// state only. `GOOGLE_ACCESS_TOKEN` is a transient process env var — a
    /// shell export with nothing on disk — and counting it badged Gemini
    /// `[saved]` for someone who had never logged in.
    #[test]
    fn a_transient_google_access_token_is_not_a_saved_credential() {
        let _lock = crate::test_env_lock();
        let (_guards, _settings) = isolated_config_home("saved-google-env");
        let _token = EnvVarGuard::set("GOOGLE_ACCESS_TOKEN", Some("ya29.transient-export"));

        let [_anthropic, _openai, google] = saved_oauth_providers();

        assert!(
            !google,
            "an env-var access token is not a saved credential, so it must not badge Gemini"
        );
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

    /// Registration is user-global: the entry must land in the config home, and
    /// nothing may be written into the working directory.
    #[test]
    fn connect_writes_the_provider_into_the_global_config_home() {
        let _lock = crate::test_env_lock();
        let (_guards, settings) = isolated_config_home("shape");
        let cwd = std::env::current_dir().expect("cwd");

        let written = write_user_provider_with_options(
            "deepseek",
            "https://api.deepseek.com",
            Some("DEEPSEEK_API_KEY"),
            &["deepseek-chat".to_string()],
            ProviderTokenLimits::default(),
            None,
        )
        .expect("write");
        assert_eq!(written, settings, "providers are global, never per-project");
        assert!(
            !cwd.join(".zo").join("settings.json").exists(),
            "a registration must not create project-local settings"
        );

        let value = read_settings(&settings);
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
        let report = connect_custom_provider(&CustomProviderRequest {
            name: "bad-env",
            base_url: "https://example.com/v1",
            auth_env: Some("FOO=bar"),
            api_key: Some("sk-secret"),
            models: &["model-a".to_string()],
            token_limits: ProviderTokenLimits::default(),
            include_usage: false,
            supports_reasoning_effort: false,
            edit_existing: false,
        });
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
    fn connect_rejects_invalid_auth_env_name() {
        let _lock = crate::test_env_lock();
        let (_guards, settings) = isolated_config_home("invalid-auth-env");
        let error = write_user_provider_with_options(
            "bad-env",
            "https://example.com/v1",
            Some("FOO=bar"),
            &["model-a".to_string()],
            ProviderTokenLimits::default(),
            Some(false),
        )
        .expect_err("invalid auth env should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!settings.exists(), "a rejected draft writes nothing");
    }

    #[test]
    fn connect_can_write_include_usage_override() {
        let _lock = crate::test_env_lock();
        let (_guards, settings) = isolated_config_home("include-usage");
        write_user_provider_with_options(
            "nvidia-nim",
            "https://integrate.api.nvidia.com/v1",
            Some("NVIDIA_API_KEY"),
            &["meta/llama-3.1-8b-instruct".to_string(), "z-ai/glm-5.2".to_string()],
            ProviderTokenLimits::default(),
            Some(false),
        )
        .expect("write");

        let entry = read_settings(&settings)["providers"][0].clone();
        assert_eq!(entry["name"], "nvidia-nim");
        assert_eq!(entry["base_url"], "https://integrate.api.nvidia.com/v1");
        assert_eq!(entry["auth_env"], "NVIDIA_API_KEY");
        assert_eq!(entry["requires_auth"], true);
        assert_eq!(entry["include_usage"], false);
        assert_eq!(entry["models"][1], "z-ai/glm-5.2");
    }

    #[test]
    fn connect_can_write_context_and_max_output_overrides() {
        let _lock = crate::test_env_lock();
        let (_guards, settings) = isolated_config_home("context-max-output");
        write_user_provider_with_options(
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

        let entry = read_settings(&settings)["providers"][0].clone();
        assert_eq!(entry["context_window"], 256_000);
        assert_eq!(entry["max_output_tokens"], 32_000);
        assert_eq!(entry["models"][0], "grok-4.5");
    }

    /// Reconnecting a provider is the same account, not a fresh one: its models
    /// accumulate under the one key, and sibling providers are untouched.
    #[test]
    fn reconnecting_a_provider_accumulates_models_under_one_key() {
        let _lock = crate::test_env_lock();
        let (_guards, settings) = isolated_config_home("dedupe");
        write_user_provider_with_options(
            "deepseek",
            "https://api.deepseek.com",
            Some("DEEPSEEK_API_KEY"),
            &["deepseek-chat".to_string()],
            ProviderTokenLimits::default(),
            None,
        )
        .expect("write cloud");
        write_user_provider_with_options(
            "ollama",
            "http://localhost:11434/v1",
            None,
            &["llama3.1".to_string()],
            ProviderTokenLimits::default(),
            None,
        )
        .expect("write local");
        write_user_provider_with_options(
            "deepseek",
            "https://api.deepseek.com",
            Some("DEEPSEEK_API_KEY"),
            &["deepseek-reasoner".to_string()],
            ProviderTokenLimits::default(),
            None,
        )
        .expect("rewrite cloud");

        let value = read_settings(&settings);
        let providers = value["providers"].as_array().expect("providers array");
        assert_eq!(providers.len(), 2, "rewrite must dedupe by name");

        let deepseek = providers
            .iter()
            .find(|entry| entry["name"] == "deepseek")
            .expect("deepseek entry");
        assert_eq!(
            deepseek["models"], serde_json::json!(["deepseek-chat", "deepseek-reasoner"]),
            "a second connect adds to the account instead of resetting it"
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
