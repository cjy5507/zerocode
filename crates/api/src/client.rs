use crate::error::ApiError;
use crate::prompt_cache::{PromptCache, PromptCacheRecord, PromptCacheStats};
use crate::providers::anthropic::{self, AnthropicClient, AnthropicRetryNotice, AuthSource};
use crate::providers::chatgpt_backend::{self, ChatGptBackendClient};
use crate::providers::gemini_code_assist::{self, GeminiCodeAssistClient};
use crate::providers::openai_compat::{self, OpenAiCompatClient, OpenAiCompatConfig};
use crate::providers::{self, NON_CLAUDE_ADAPTERS_ENV, ProviderKind};
use crate::sync_bridge::run_blocking;
use crate::types::{MessageRequest, MessageResponse, StreamEvent};
use core_types::OpenAiOAuthTokens;
use serde::{Deserialize, Serialize};

/// Credential mechanism selected for one model-catalog row.
///
/// `Auto` preserves each provider's existing credential precedence. The other
/// variants are strict: construction either uses that mechanism or fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthRoute {
    #[default]
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "oauth")]
    OAuth,
    #[serde(rename = "api-key")]
    ApiKey,
}

impl AuthRoute {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::OAuth => "oauth",
            Self::ApiKey => "api-key",
        }
    }
}

fn provider_name(provider_kind: ProviderKind) -> &'static str {
    match provider_kind {
        ProviderKind::Anthropic => "Anthropic",
        ProviderKind::Xai => "xAI",
        ProviderKind::OpenAi => "OpenAI",
        ProviderKind::Google => "Google",
        ProviderKind::Ollama => "Ollama",
    }
}

fn anthropic_auth_for_route(
    auth_route: AuthRoute,
    auth: Option<AuthSource>,
) -> Result<AuthSource, ApiError> {
    let auth = match (auth_route, auth) {
        (_, Some(auth)) => auth,
        (AuthRoute::Auto, None) => AuthSource::from_env()?,
        (AuthRoute::OAuth, None) => AuthSource::from_oauth_only()?,
        (AuthRoute::ApiKey, None) => AuthSource::from_api_key_only()?,
    };
    match auth_route {
        AuthRoute::Auto => Ok(auth),
        AuthRoute::OAuth => auth
            .bearer_token()
            .map(|token| AuthSource::BearerToken(token.to_string()))
            .ok_or_else(|| {
                ApiError::missing_auth_route_credentials("Anthropic", auth_route.as_str())
            }),
        AuthRoute::ApiKey => auth
            .api_key()
            .map(|key| AuthSource::ApiKey(key.to_string()))
            .ok_or_else(|| {
                ApiError::missing_auth_route_credentials("Anthropic", auth_route.as_str())
            }),
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ProviderClient {
    Anthropic(AnthropicClient),
    Xai(OpenAiCompatClient),
    OpenAi(OpenAiCompatClient),
    Google(OpenAiCompatClient),
    /// Gemini CLI-compatible Code Assist backend over Google OAuth.
    GeminiCodeAssist(GeminiCodeAssistClient),
    Ollama(OpenAiCompatClient),
    /// ChatGPT subscription backend (Responses API over an OAuth access token).
    ChatGpt(ChatGptBackendClient),
}

impl ProviderClient {
    pub fn from_model(model: &str) -> Result<Self, ApiError> {
        Self::from_model_with_auth_route(model, AuthRoute::Auto)
    }

    /// Pin the ChatGPT backend's prompt-cache scope to a host-stable id (the
    /// zo session id) so provider cache keys survive client rebuilds — model
    /// swaps and OAuth rotations reconstruct the client, and its default
    /// per-instance scope would roll the key each time (see
    /// [`ChatGptBackendClient::with_cache_scope`]). No-op for every other
    /// provider: Anthropic caching is prefix-based (no key), and the
    /// OpenAI-compatible path derives its key without a session component.
    #[must_use]
    pub fn with_cache_scope(self, scope: &str) -> Self {
        match self {
            Self::ChatGpt(client) => Self::ChatGpt(client.with_cache_scope(scope)),
            other => other,
        }
    }

    /// The pinned ChatGPT cache scope, if this client carries one — read at
    /// 401-recovery rebuild time to carry the scope onto the fresh client.
    #[must_use]
    pub fn pinned_cache_scope(&self) -> Option<&str> {
        match self {
            Self::ChatGpt(client) => client.pinned_cache_scope(),
            _ => None,
        }
    }

    pub fn from_model_with_anthropic_auth(
        model: &str,
        anthropic_auth: Option<AuthSource>,
    ) -> Result<Self, ApiError> {
        Self::from_model_with_auth_route_and_anthropic_auth(
            model,
            AuthRoute::Auto,
            anthropic_auth,
        )
    }

    pub fn from_model_with_auth_route(
        model: &str,
        auth_route: AuthRoute,
    ) -> Result<Self, ApiError> {
        Self::from_model_with_auth_route_and_anthropic_auth(model, auth_route, None)
    }

    pub fn from_model_with_auth_route_and_anthropic_auth(
        model: &str,
        auth_route: AuthRoute,
        anthropic_auth: Option<AuthSource>,
    ) -> Result<Self, ApiError> {
        let resolved_model = providers::resolve_model_alias(model);

        // A user-defined provider (ZO_CUSTOM_PROVIDERS) takes over once the
        // static registry has missed. It reuses the OpenAI-compatible client,
        // wrapped in the existing `OpenAi` variant so no new `ProviderClient`
        // arm or exhaustive-match edit is needed. Defining the provider is its
        // own opt-in, so this MUST sit ahead of the built-in non-Claude adapter
        // gate: otherwise a custom model named `grok-*` is misclassified as the
        // first-party xAI adapter and rejected before the custom route can run.
        if let Some(custom) = providers::custom_provider_for_model(&resolved_model) {
            return match auth_route {
                AuthRoute::Auto | AuthRoute::ApiKey => {
                    Ok(Self::OpenAi(build_custom_client(&custom)?))
                }
                AuthRoute::OAuth => Err(ApiError::unsupported_auth_route(
                    custom.config.provider_name,
                    auth_route.as_str(),
                )),
            };
        }

        if auth_route == AuthRoute::Auto {
            if let Some(provider_kind) =
                providers::explicit_non_claude_provider_kind(&resolved_model)
            {
                if !providers::provider_enabled(provider_kind) {
                    return Err(ApiError::unsupported_provider(
                        provider_name(provider_kind),
                        NON_CLAUDE_ADAPTERS_ENV,
                    ));
                }
            }
        }

        Self::from_provider_kind_with_auth_route_and_anthropic_auth(
            providers::detect_provider_kind(&resolved_model),
            auth_route,
            anthropic_auth,
        )
    }

    /// Build the explicitly selected built-in provider without re-detecting it
    /// from the model ID. Model-catalog entries use this path so unfamiliar
    /// future IDs still route through the provider chosen by the operator.
    pub fn from_provider_kind_with_anthropic_auth(
        provider_kind: ProviderKind,
        anthropic_auth: Option<AuthSource>,
    ) -> Result<Self, ApiError> {
        Self::from_provider_kind_with_auth_route_and_anthropic_auth(
            provider_kind,
            AuthRoute::Auto,
            anthropic_auth,
        )
    }

    pub fn from_provider_kind_with_auth_route(
        provider_kind: ProviderKind,
        auth_route: AuthRoute,
    ) -> Result<Self, ApiError> {
        Self::from_provider_kind_with_auth_route_and_anthropic_auth(
            provider_kind,
            auth_route,
            None,
        )
    }

    pub fn from_provider_kind_with_auth_route_and_anthropic_auth(
        provider_kind: ProviderKind,
        auth_route: AuthRoute,
        anthropic_auth: Option<AuthSource>,
    ) -> Result<Self, ApiError> {
        if auth_route == AuthRoute::Auto && !providers::provider_enabled(provider_kind) {
            return Err(ApiError::unsupported_provider(
                provider_name(provider_kind),
                NON_CLAUDE_ADAPTERS_ENV,
            ));
        }

        match provider_kind {
            ProviderKind::Anthropic => Ok(Self::Anthropic(AnthropicClient::from_auth(
                anthropic_auth_for_route(auth_route, anthropic_auth)?,
            ))),
            ProviderKind::Xai => match auth_route {
                AuthRoute::Auto => Ok(Self::Xai(openai_compat_from_env(
                    OpenAiCompatConfig::xai(),
                )?)),
                AuthRoute::ApiKey => Ok(Self::Xai(openai_compat_api_key_for_route(
                    OpenAiCompatConfig::xai(),
                )?)),
                AuthRoute::OAuth => Err(ApiError::unsupported_auth_route(
                    "xAI",
                    auth_route.as_str(),
                )),
            },
            ProviderKind::OpenAi => match auth_route {
                AuthRoute::Auto => Ok(match load_fresh_openai_oauth() {
                    Some(tokens) => Self::ChatGpt(ChatGptBackendClient::new(
                        tokens.access_token,
                        tokens.account_id,
                    )),
                    None => Self::OpenAi(openai_compat_from_env(OpenAiCompatConfig::openai())?),
                }),
                AuthRoute::OAuth => load_fresh_openai_oauth().map_or_else(
                    || {
                        Err(ApiError::missing_auth_route_credentials(
                            "OpenAI",
                            auth_route.as_str(),
                        ))
                    },
                    |tokens| {
                        Ok(Self::ChatGpt(ChatGptBackendClient::new(
                            tokens.access_token,
                            tokens.account_id,
                        )))
                    },
                ),
                AuthRoute::ApiKey => Ok(Self::OpenAi(openai_compat_api_key_for_route(
                    OpenAiCompatConfig::openai(),
                )?)),
            },
            ProviderKind::Google => match auth_route {
                AuthRoute::Auto => Ok(
                    if let Some(tokens) = gemini_code_assist::load_fresh_oauth() {
                        Self::GeminiCodeAssist(GeminiCodeAssistClient::new(tokens.access_token))
                    } else {
                        Self::Google(google_openai_compat_from_env()?)
                    },
                ),
                AuthRoute::OAuth => gemini_code_assist::load_fresh_oauth().map_or_else(
                    || {
                        Err(ApiError::missing_auth_route_credentials(
                            "Google",
                            auth_route.as_str(),
                        ))
                    },
                    |tokens| {
                        Ok(Self::GeminiCodeAssist(GeminiCodeAssistClient::new(
                            tokens.access_token,
                        )))
                    },
                ),
                AuthRoute::ApiKey => Ok(Self::Google(openai_compat_api_key_for_route(
                    OpenAiCompatConfig::google(),
                )?)),
            },
            ProviderKind::Ollama => match auth_route {
                AuthRoute::Auto => Ok(Self::Ollama(
                    OpenAiCompatClient::from_env(OpenAiCompatConfig::ollama()).or_else(|_| {
                        OpenAiCompatClient::from_env_optional_auth(OpenAiCompatConfig::ollama())
                    })?,
                )),
                AuthRoute::OAuth | AuthRoute::ApiKey => Err(ApiError::unsupported_auth_route(
                    "Ollama",
                    auth_route.as_str(),
                )),
            },
        }
    }

    #[must_use]
    pub const fn provider_kind(&self) -> ProviderKind {
        match self {
            Self::Anthropic(_) => ProviderKind::Anthropic,
            Self::Xai(_) => ProviderKind::Xai,
            Self::OpenAi(_) | Self::ChatGpt(_) => ProviderKind::OpenAi,
            Self::Google(_) | Self::GeminiCodeAssist(_) => ProviderKind::Google,
            Self::Ollama(_) => ProviderKind::Ollama,
        }
    }

    #[must_use]
    pub fn with_prompt_cache(self, prompt_cache: PromptCache) -> Self {
        match self {
            Self::Anthropic(client) => Self::Anthropic(client.with_prompt_cache(prompt_cache)),
            other => other,
        }
    }

    /// Return a copy with the Anthropic auth swapped (no-op for other
    /// providers). Used to retry a 401'd streaming request with a freshly
    /// refreshed OAuth bearer without rebuilding the whole client.
    #[must_use]
    pub fn with_anthropic_auth(self, auth: AuthSource) -> Self {
        match self {
            Self::Anthropic(client) => Self::Anthropic(client.with_auth(auth)),
            other => other,
        }
    }

    /// Return a copy with a foreground retry notice callback installed on the
    /// Anthropic client. No-op for non-Anthropic providers.
    #[must_use]
    pub fn with_anthropic_retry_notice_callback(
        self,
        callback: impl Fn(AnthropicRetryNotice) + Send + Sync + 'static,
    ) -> Self {
        match self {
            Self::Anthropic(client) => Self::Anthropic(client.with_retry_notice_callback(callback)),
            other => other,
        }
    }

    /// Return a copy whose Anthropic client surfaces a rate limit immediately.
    /// No-op for non-Anthropic providers.
    #[must_use]
    pub fn with_rate_limit_fail_fast(self) -> Self {
        match self {
            Self::Anthropic(client) => Self::Anthropic(client.with_rate_limit_fail_fast()),
            other => other,
        }
    }

    #[must_use]
    pub fn prompt_cache_stats(&self) -> Option<PromptCacheStats> {
        match self {
            Self::Anthropic(client) => client.prompt_cache_stats(),
            Self::Xai(_)
            | Self::OpenAi(_)
            | Self::Google(_)
            | Self::GeminiCodeAssist(_)
            | Self::Ollama(_)
            | Self::ChatGpt(_) => None,
        }
    }

    #[must_use]
    pub fn take_last_prompt_cache_record(&self) -> Option<PromptCacheRecord> {
        match self {
            Self::Anthropic(client) => client.take_last_prompt_cache_record(),
            Self::Xai(_)
            | Self::OpenAi(_)
            | Self::Google(_)
            | Self::GeminiCodeAssist(_)
            | Self::Ollama(_)
            | Self::ChatGpt(_) => None,
        }
    }

    /// Whether this long-lived OAuth-backed provider client should be rebuilt
    /// before the next request because its saved bearer is expired or within the
    /// provider's refresh skew. This is intentionally a cheap local token-store
    /// check; rebuilding runs provider loaders that may perform network OAuth /
    /// account setup and must not happen on every turn.
    #[must_use]
    pub fn oauth_rebuild_needed(&self) -> bool {
        match self {
            Self::GeminiCodeAssist(_) => crate::oauth_store::load_google_code_assist_oauth()
                .ok()
                .flatten()
                .is_some_and(|tokens| gemini_code_assist::token_expired(&tokens)),
            Self::ChatGpt(_) => crate::oauth_store::load_openai_oauth()
                .ok()
                .flatten()
                .is_some_and(|tokens| openai_oauth_expired(&tokens)),
            Self::Anthropic(_)
            | Self::Xai(_)
            | Self::OpenAi(_)
            | Self::Google(_)
            | Self::Ollama(_) => false,
        }
    }

    pub async fn send_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageResponse, ApiError> {
        match self {
            Self::Anthropic(client) => client.send_message(request).await,
            Self::Xai(client)
            | Self::OpenAi(client)
            | Self::Google(client)
            | Self::Ollama(client) => client.send_message(request).await,
            Self::GeminiCodeAssist(client) => client.send_message(request).await,
            Self::ChatGpt(client) => client.send_message(request).await,
        }
    }

    pub async fn stream_message(
        &self,
        request: &MessageRequest,
    ) -> Result<MessageStream, ApiError> {
        match self {
            Self::Anthropic(client) => client
                .stream_message(request)
                .await
                .map(|stream| MessageStream::Anthropic(Box::new(stream))),
            Self::Xai(client)
            | Self::OpenAi(client)
            | Self::Google(client)
            | Self::Ollama(client) => client
                .stream_message(request)
                .await
                .map(|stream| MessageStream::OpenAiCompat(Box::new(stream))),
            Self::GeminiCodeAssist(client) => client
                .stream_message(request)
                .await
                .map(|stream| MessageStream::GeminiCodeAssist(Box::new(stream))),
            Self::ChatGpt(client) => client
                .stream_message(request)
                .await
                .map(|stream| MessageStream::ChatGpt(Box::new(stream))),
        }
    }
}

#[derive(Debug)]
pub enum MessageStream {
    // All variants boxed: each provider stream embeds a live response plus
    // retry state (the ChatGpt one carries a whole client + request), so
    // boxing keeps `MessageStream` itself pointer-sized per arm. Match arms
    // read through the box transparently via `Deref`.
    Anthropic(Box<anthropic::MessageStream>),
    OpenAiCompat(Box<openai_compat::MessageStream>),
    GeminiCodeAssist(Box<gemini_code_assist::GeminiCodeAssistStream>),
    ChatGpt(Box<chatgpt_backend::ChatGptStream>),
}

impl MessageStream {
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Self::Anthropic(stream) => stream.request_id(),
            Self::OpenAiCompat(stream) => stream.request_id(),
            Self::GeminiCodeAssist(stream) => stream.request_id(),
            Self::ChatGpt(_) => None,
        }
    }

    /// Unified rate-limit snapshot from the response headers, if the provider
    /// surfaced one. Only Anthropic (subscription / OAuth) carries the unified
    /// `anthropic-ratelimit-unified-*` headers; OpenAI-compatible providers
    /// return `None`.
    #[must_use]
    pub fn rate_limit(&self) -> Option<core_types::RateLimitSnapshot> {
        match self {
            Self::Anthropic(stream) => stream.rate_limit(),
            Self::OpenAiCompat(_) | Self::GeminiCodeAssist(_) | Self::ChatGpt(_) => None,
        }
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        match self {
            Self::Anthropic(stream) => stream.next_event().await,
            Self::OpenAiCompat(stream) => stream.next_event().await,
            Self::GeminiCodeAssist(stream) => stream.next_event().await,
            Self::ChatGpt(stream) => stream.next_event().await,
        }
    }

    /// Install a mid-stream retry sink. The `ChatGpt`, `OpenAiCompat`, and
    /// `GeminiCodeAssist` backends perform internal transparent pre-commit
    /// restarts that the establish-time retry layer never sees, so without
    /// this the reconnect pause reads as a freeze. `Anthropic` deliberately
    /// stays silent (see its `restart` doc: transparent recovery must not
    /// write into the TUI).
    #[must_use]
    pub fn with_stream_retry_notice(
        self,
        callback: impl Fn(core_types::StreamRetryNotice) + Send + Sync + 'static,
    ) -> Self {
        match self {
            Self::ChatGpt(stream) => {
                Self::ChatGpt(Box::new(stream.with_retry_notice_callback(callback)))
            }
            Self::OpenAiCompat(stream) => {
                Self::OpenAiCompat(Box::new(stream.with_retry_notice_callback(callback)))
            }
            Self::GeminiCodeAssist(stream) => {
                Self::GeminiCodeAssist(Box::new(stream.with_retry_notice_callback(callback)))
            }
            Self::Anthropic(_) => self,
        }
    }
}

pub use anthropic::{
    OAuthTokenSet, oauth_token_is_expired, resolve_saved_oauth_token, resolve_startup_auth_source,
};
#[must_use]
pub fn read_base_url() -> String {
    anthropic::read_base_url()
}

#[must_use]
pub fn read_xai_base_url() -> String {
    openai_compat::read_base_url(OpenAiCompatConfig::xai())
}

/// Build an OpenAI-compatible client from the environment. When the provider
/// points at a *custom* base URL (a self-hosted or proxy endpoint), fall back
/// to optional auth like Ollama so a keyless endpoint works; the official
/// cloud endpoints keep requiring a key, surfacing a clean missing-credentials
/// error instead of a downstream 401.
fn openai_compat_from_env(config: OpenAiCompatConfig) -> Result<OpenAiCompatClient, ApiError> {
    if openai_compat::has_custom_base_url(config) {
        OpenAiCompatClient::from_env(config)
            .or_else(|_| OpenAiCompatClient::from_env_optional_auth(config))
    } else {
        OpenAiCompatClient::from_env(config)
    }
}

fn openai_compat_api_key_for_route(
    config: OpenAiCompatConfig,
) -> Result<OpenAiCompatClient, ApiError> {
    OpenAiCompatClient::from_env(config).map_err(|error| match error {
        ApiError::MissingCredentials { .. } => {
            ApiError::missing_auth_route_credentials(config.provider_name, "api-key")
        }
        error => error,
    })
}

/// Build the Gemini/OpenAI-compatible fallback client. Custom `GOOGLE_BASE_URL`
/// endpoints keep the old optional-auth behavior so keyless proxies remain
/// supported. On the official Google endpoint, OAuth/ADC wins over
/// `GOOGLE_API_KEY` to keep Gemini on an OAuth-first policy.
fn google_openai_compat_from_env() -> Result<OpenAiCompatClient, ApiError> {
    let config = OpenAiCompatConfig::google();
    if openai_compat::has_custom_base_url(config) {
        return OpenAiCompatClient::from_env_optional_auth(config);
    }
    if crate::providers::google_auth::gemini_oauth_available() {
        return Ok(OpenAiCompatClient::google_oauth(config));
    }
    OpenAiCompatClient::from_env(config)
}

/// Build the OpenAI-compatible client for a user-defined provider. A provider
/// declaring `requires_auth: true` errors clearly when its key env is unset;
/// `false` lets a keyless self-host build with an empty key.
fn build_custom_client(
    custom: &providers::ResolvedCustomProvider,
) -> Result<OpenAiCompatClient, ApiError> {
    if custom.requires_auth {
        OpenAiCompatClient::from_env(custom.config)
    } else {
        OpenAiCompatClient::from_env_optional_auth(custom.config)
    }
}

/// Load the saved ChatGPT OAuth tokens, refreshing first when expired. Returns
/// `None` when no ChatGPT login exists so the caller falls back to the api-key
/// path. A refresh failure yields the existing (expired) tokens so the call can
/// surface a clear 401 rather than silently downgrading to the api-key path.
fn load_fresh_openai_oauth() -> Option<OpenAiOAuthTokens> {
    let tokens = crate::oauth_store::load_openai_oauth().ok().flatten()?;
    if !openai_oauth_expired(&tokens) {
        return Some(tokens);
    }
    let Some(refresh_token) = tokens.refresh_token.clone() else {
        return Some(tokens);
    };
    match run_blocking(crate::providers::openai_oauth::refresh_openai_tokens(
        &refresh_token,
    )) {
        Ok(mut refreshed) => {
            if refreshed.account_id.is_none() {
                refreshed.account_id = tokens.account_id;
            }
            let _ = crate::oauth_store::save_openai_oauth(&refreshed);
            Some(refreshed)
        }
        Err(_) => Some(tokens),
    }
}

/// Whether the saved ChatGPT token is within 60s of expiry (matching the
/// Anthropic OAuth buffer).
fn openai_oauth_expired(tokens: &OpenAiOAuthTokens) -> bool {
    tokens.expires_at.is_some_and(|expires_at| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0);
        expires_at <= now + 60
    })
}

#[cfg(test)]
mod tests {
    use crate::providers::{
        EXPERIMENTAL_PROVIDERS_ENV, NON_CLAUDE_ADAPTERS_ENV, ProviderKind, detect_provider_kind,
        non_claude_adapters_enabled, resolve_model_alias,
    };

    struct EnvVarGuard {
        key: &'static str,
        original: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
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

    #[test]
    fn resolves_existing_and_grok_aliases() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, Some("1"));
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-8");
        assert_eq!(resolve_model_alias("grok"), "grok-3");
    }

    #[test]
    fn provider_detection_prefers_model_family() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, Some("1"));
        assert_eq!(detect_provider_kind("grok-3"), ProviderKind::Xai);
        assert_eq!(
            detect_provider_kind("claude-sonnet-4-6"),
            ProviderKind::Anthropic
        );
    }

    #[test]
    fn non_claude_adapters_default_to_disabled() {
        let _lock = crate::test_env_lock();
        let _legacy_gate = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, None);
        let _adapter_gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        // Implicit activation now also keys off provider base URLs, so a clean
        // default requires every provider credential/endpoint env to be unset.
        let _guards: Vec<EnvVarGuard> = [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "GOOGLE_API_KEY",
            "GOOGLE_BASE_URL",
            "GOOGLE_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HOME",
            "XAI_API_KEY",
            "XAI_BASE_URL",
            "OLLAMA_API_KEY",
            "OLLAMA_BASE_URL",
        ]
        .into_iter()
        .map(|key| EnvVarGuard::set(key, None))
        .collect();
        assert!(!non_claude_adapters_enabled());
    }

    #[test]
    fn google_code_assist_oauth_takes_priority_over_custom_base_url() {
        let _lock = crate::test_env_lock();
        let temp_home = std::env::temp_dir().join(format!(
            "zo-google-oauth-priority-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&temp_home);
        let temp_home_str = temp_home.to_str().expect("utf-8 temp path").to_string();

        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&temp_home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", None);
        let _disable_external = EnvVarGuard::set("ZO_DISABLE_EXTERNAL_CREDENTIALS", None);
        let _base = EnvVarGuard::set("GOOGLE_BASE_URL", Some("http://localhost:9999/v1"));
        let _key = EnvVarGuard::set("GOOGLE_API_KEY", None);

        let expires_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_secs()
            + 3600;
        crate::oauth_store::save_google_code_assist_oauth(&core_types::OAuthTokenSet {
            access_token: "access-token".to_string(),
            refresh_token: None,
            expires_at: Some(expires_at),
            scopes: Vec::new(),
        })
        .expect("save google oauth token");

        let client =
            super::ProviderClient::from_model_with_anthropic_auth("gemini-3-flash-preview", None)
                .expect("google client");
        assert!(matches!(client, super::ProviderClient::GeminiCodeAssist(_)));

        let _ = std::fs::remove_dir_all(&temp_home);
    }

    fn provider_error(
        result: Result<super::ProviderClient, crate::ApiError>,
        message: &str,
    ) -> crate::ApiError {
        let Err(error) = result else {
            panic!("{message}");
        };
        error
    }

    #[test]
    fn auth_route_serde_round_trips() {
        for (route, encoded) in [
            (super::AuthRoute::Auto, "\"auto\""),
            (super::AuthRoute::OAuth, "\"oauth\""),
            (super::AuthRoute::ApiKey, "\"api-key\""),
        ] {
            assert_eq!(serde_json::to_string(&route).unwrap(), encoded);
            assert_eq!(serde_json::from_str::<super::AuthRoute>(encoded).unwrap(), route);
        }
    }

    #[test]
    fn forced_oauth_routes_do_not_fall_back_to_api_keys() {
        let _lock = crate::test_env_lock();
        let temp_home = std::env::temp_dir().join(format!(
            "zo-forced-oauth-route-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&temp_home);
        std::fs::create_dir_all(&temp_home).unwrap();
        let temp_home_str = temp_home.to_str().expect("utf-8 temp path").to_string();
        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&temp_home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", Some(&temp_home_str));
        let _home = EnvVarGuard::set("HOME", Some(&temp_home_str));
        let _disable_external = EnvVarGuard::set("ZO_DISABLE_EXTERNAL_CREDENTIALS", None);
        let _google_key = EnvVarGuard::set("GOOGLE_API_KEY", Some("google-test-key"));
        let _openai_key = EnvVarGuard::set("OPENAI_API_KEY", Some("openai-test-key"));

        let google_error = provider_error(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::Google,
                super::AuthRoute::OAuth,
            ),
            "Google OAuth must not fall back to the API key",
        );
        assert!(matches!(
            google_error,
            crate::ApiError::MissingAuthRouteCredentials {
                provider: "Google",
                route: "oauth"
            }
        ));
        let openai_error = provider_error(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::OpenAi,
                super::AuthRoute::OAuth,
            ),
            "OpenAI OAuth must not fall back to the API key",
        );
        assert!(matches!(
            openai_error,
            crate::ApiError::MissingAuthRouteCredentials {
                provider: "OpenAI",
                route: "oauth"
            }
        ));
        let _ = std::fs::remove_dir_all(temp_home);
    }

    #[test]
    fn forced_routes_select_exact_credentials_without_fallback() {
        let _lock = crate::test_env_lock();
        let temp_home = std::env::temp_dir().join(format!(
            "zo-forced-exact-route-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&temp_home);
        std::fs::create_dir_all(&temp_home).unwrap();
        let temp_home_str = temp_home.to_str().expect("utf-8 temp path").to_string();
        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&temp_home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", Some(&temp_home_str));
        let _home = EnvVarGuard::set("HOME", Some(&temp_home_str));
        let _disable_external = EnvVarGuard::set("ZO_DISABLE_EXTERNAL_CREDENTIALS", None);
        let _google_key = EnvVarGuard::set("GOOGLE_API_KEY", Some("google-test-key"));
        let _openai_key = EnvVarGuard::set("OPENAI_API_KEY", Some("openai-test-key"));
        let expires_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        crate::oauth_store::save_google_code_assist_oauth(&core_types::OAuthTokenSet {
            access_token: "google-oauth-token".to_string(),
            refresh_token: None,
            expires_at: Some(expires_at),
            scopes: Vec::new(),
        })
        .unwrap();
        crate::oauth_store::save_openai_oauth(&core_types::OpenAiOAuthTokens {
            access_token: "openai-oauth-token".to_string(),
            refresh_token: None,
            expires_at: Some(expires_at),
            account_id: Some("acct".to_string()),
            scopes: Vec::new(),
        })
        .unwrap();

        assert!(matches!(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::Google,
                super::AuthRoute::ApiKey,
            )
            .unwrap(),
            super::ProviderClient::Google(_)
        ));
        assert!(matches!(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::OpenAi,
                super::AuthRoute::ApiKey,
            )
            .unwrap(),
            super::ProviderClient::OpenAi(_)
        ));
        assert!(matches!(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::Google,
                super::AuthRoute::OAuth,
            )
            .unwrap(),
            super::ProviderClient::GeminiCodeAssist(_)
        ));
        assert!(matches!(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::OpenAi,
                super::AuthRoute::OAuth,
            )
            .unwrap(),
            super::ProviderClient::ChatGpt(_)
        ));

        std::env::remove_var("GOOGLE_API_KEY");
        let key_error = provider_error(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::Google,
                super::AuthRoute::ApiKey,
            ),
            "Google API key route must not use saved OAuth",
        );
        assert!(matches!(
            key_error,
            crate::ApiError::MissingAuthRouteCredentials {
                provider: "Google",
                route: "api-key"
            }
        ));
        let _ = std::fs::remove_dir_all(temp_home);
    }

    #[test]
    fn unsupported_explicit_auth_route_is_rejected() {
        let unsupported = provider_error(
            super::ProviderClient::from_provider_kind_with_auth_route(
                ProviderKind::Xai,
                super::AuthRoute::OAuth,
            ),
            "xAI OAuth is unsupported",
        );
        assert!(matches!(
            unsupported,
            crate::ApiError::UnsupportedAuthRoute { .. }
        ));
    }

    #[test]
    fn oauth_rebuild_needed_only_when_saved_oauth_is_near_expiry() {
        let _lock = crate::test_env_lock();
        let temp_home = std::env::temp_dir().join(format!(
            "zo-oauth-rebuild-needed-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&temp_home);
        std::fs::create_dir_all(&temp_home).expect("temp home");
        let temp_home_str = temp_home.to_str().expect("utf-8 temp path").to_string();

        let _config_home = EnvVarGuard::set("ZO_CONFIG_HOME", Some(&temp_home_str));
        let _zo_home = EnvVarGuard::set("ZO_HOME", None);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_secs();

        crate::oauth_store::save_google_code_assist_oauth(&core_types::OAuthTokenSet {
            access_token: "fresh-google-token".to_string(),
            refresh_token: Some("google-refresh".to_string()),
            expires_at: Some(now + 3600),
            scopes: Vec::new(),
        })
        .expect("save fresh google oauth");
        let google = super::ProviderClient::GeminiCodeAssist(
            crate::providers::gemini_code_assist::GeminiCodeAssistClient::new("fresh-google-token"),
        );
        assert!(
            !google.oauth_rebuild_needed(),
            "fresh Code Assist OAuth must not rebuild before every turn"
        );

        crate::oauth_store::save_google_code_assist_oauth(&core_types::OAuthTokenSet {
            access_token: "expired-google-token".to_string(),
            refresh_token: Some("google-refresh".to_string()),
            expires_at: Some(now.saturating_sub(1)),
            scopes: Vec::new(),
        })
        .expect("save expired google oauth");
        assert!(google.oauth_rebuild_needed());

        crate::oauth_store::save_openai_oauth(&core_types::OpenAiOAuthTokens {
            access_token: "fresh-openai-token".to_string(),
            refresh_token: Some("openai-refresh".to_string()),
            expires_at: Some(now + 3600),
            account_id: Some("acct".to_string()),
            scopes: Vec::new(),
        })
        .expect("save fresh openai oauth");
        let chatgpt = super::ProviderClient::ChatGpt(
            crate::providers::chatgpt_backend::ChatGptBackendClient::new(
                "fresh-openai-token",
                Some("acct".to_string()),
            ),
        );
        assert!(
            !chatgpt.oauth_rebuild_needed(),
            "fresh ChatGPT OAuth must not rebuild before every turn"
        );

        crate::oauth_store::save_openai_oauth(&core_types::OpenAiOAuthTokens {
            access_token: "expired-openai-token".to_string(),
            refresh_token: Some("openai-refresh".to_string()),
            expires_at: Some(now.saturating_sub(1)),
            account_id: Some("acct".to_string()),
            scopes: Vec::new(),
        })
        .expect("save expired openai oauth");
        assert!(chatgpt.oauth_rebuild_needed());

        let _ = std::fs::remove_dir_all(temp_home);
    }

    #[test]
    fn openai_compat_from_env_tolerates_missing_key_for_custom_base_url() {
        use crate::providers::openai_compat::OpenAiCompatConfig;
        let _lock = crate::test_env_lock();
        let _key = EnvVarGuard::set("OPENAI_API_KEY", None);

        // Official cloud endpoint, no key → clean missing-credentials error.
        let no_base = EnvVarGuard::set("OPENAI_BASE_URL", None);
        assert!(super::openai_compat_from_env(OpenAiCompatConfig::openai()).is_err());
        drop(no_base);

        // Self-hosted endpoint (custom base URL), no key → constructs with
        // optional auth instead of erroring.
        let _base = EnvVarGuard::set("OPENAI_BASE_URL", Some("http://localhost:8080/v1"));
        assert!(super::openai_compat_from_env(OpenAiCompatConfig::openai()).is_ok());
    }

    fn custom_provider(json: &str) -> crate::providers::ResolvedCustomProvider {
        let parsed = crate::providers::parse_custom_providers(&format!("[{json}]"))
            .expect("valid custom provider json");
        let custom = parsed.into_iter().next().expect("one provider");
        crate::providers::ResolvedCustomProvider {
            context_window: custom.context_window.filter(|&value| value > 0),
            max_output_tokens: custom.max_output_tokens.filter(|&value| value > 0),
            fit_hint: custom.to_fit_hint(),
            config: custom.to_static_config(),
            models: custom.models,
            requires_auth: custom.requires_auth,
        }
    }

    #[test]
    fn custom_grok_model_uses_custom_provider_before_xai_gate() {
        let _lock = crate::test_env_lock();
        let _legacy_gate = EnvVarGuard::set(crate::providers::EXPERIMENTAL_PROVIDERS_ENV, None);
        let _adapter_gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        let _xai_key = EnvVarGuard::set("XAI_API_KEY", None);
        let _xai_base = EnvVarGuard::set("XAI_BASE_URL", None);
        crate::providers::refresh_custom_providers_from_json(
            r#"[{"name":"xai-custom","base_url":"https://api.x.ai/v1","models":["grok-4.5"],"requires_auth":false}]"#,
        )
        .expect("refresh custom provider");

        let client = super::ProviderClient::from_model("grok-4.5")
            .expect("custom grok model should bypass built-in xAI adapter gate");
        assert!(matches!(client, super::ProviderClient::OpenAi(_)));

        crate::providers::refresh_custom_providers_from_json("[]")
            .expect("restore empty custom providers");
    }

    #[test]
    fn build_custom_client_allows_keyless_self_host() {
        let _lock = crate::test_env_lock();
        // requires_auth:false → builds even with no API key env set.
        let provider = custom_provider(
            r#"{"name":"Local","base_url":"http://localhost:11434/v1",
                "models":["llama-3.3"],"requires_auth":false}"#,
        );
        assert!(super::build_custom_client(&provider).is_ok());
    }

    #[test]
    fn build_custom_client_errors_when_required_key_missing() {
        let _lock = crate::test_env_lock();
        let _key = EnvVarGuard::set("CUSTOM_PROVIDER_KEY", None);
        // requires_auth:true (default) + unset key env → clean error, no panic.
        let provider = custom_provider(
            r#"{"name":"Cloudish","base_url":"https://api.cloudish.example/v1",
                "auth_env":"CUSTOM_PROVIDER_KEY","models":["cloud-large"]}"#,
        );
        let error = super::build_custom_client(&provider).expect_err("missing custom key");
        let rendered = error.to_string();
        assert!(
            rendered.contains("CUSTOM_PROVIDER_KEY"),
            "custom provider missing-credential error must name auth_env: {rendered}"
        );
        assert!(
            !rendered.contains("export  before"),
            "custom provider missing-credential error must not render a blank export hint: {rendered}"
        );
        assert!(
            !rendered.contains("zo login") && !rendered.contains("/login"),
            "custom adapter missing-credential error must not suggest OAuth login: {rendered}"
        );
    }
}
