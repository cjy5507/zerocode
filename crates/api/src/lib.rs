mod client;
mod error;
pub mod oauth_store;
pub mod otlp;
mod prompt_cache;
mod providers;
pub mod quota;
mod quota_shared;
mod sse;
pub mod sync_bridge;
pub mod token_counting;
mod types;

pub use client::{
    AuthRoute, MessageStream, OAuthTokenSet, ProviderClient, oauth_token_is_expired, read_base_url,
    read_xai_base_url, resolve_saved_oauth_token, resolve_startup_auth_source,
};
pub use core_types::{RateLimitSnapshot, RateLimitWindow, RateLimitWindowKind};
pub use error::{ApiError, ProviderErrorClass};
pub use prompt_cache::{
    CacheBreakEvent, PromptCache, PromptCacheConfig, PromptCachePaths, PromptCacheRecord,
    PromptCacheStats,
};
pub use providers::anthropic::keychain::{
    KeychainSession, claude_code_oauth_config, read_claude_code_keychain_session,
    read_claude_code_keychain_token,
};
pub use providers::anthropic::latest_claude_auth_origin;
pub use providers::anthropic::{
    AnthropicClient, AnthropicClient as ApiClient, AuthSource, ClaudeAuthOrigin,
    ResolvedClaudeAuth, anthropic_context_editing_enabled,
    refresh_claude_auth_after_unauthorized, resolve_claude_auth_fresh,
    resolve_claude_auth_fresh_detailed,
};
pub use providers::cloud_gateway::cloud_gateway_active;
pub use providers::gemini_code_assist::{
    GEMINI_CODE_ASSIST_OAUTH_CLIENT_ID_ENV, GEMINI_CODE_ASSIST_OAUTH_CLIENT_SECRET_ENV,
    GeminiCodeAssistClient, authorize_url as google_code_assist_authorize_url,
    exchange_code as exchange_google_code_assist_code,
    oauth_config as google_code_assist_oauth_config,
    oauth_present as google_code_assist_oauth_present,
    redirect_uri as google_code_assist_redirect_uri,
    refresh_tokens as refresh_google_code_assist_tokens,
    setup_saved_user as google_code_assist_setup_saved_user,
};
pub use providers::google_auth::{
    GOOGLE_ACCESS_TOKEN_ENV, GOOGLE_OAUTH_CLIENT_ID_FILE_ENV, GoogleOAuthClientConfig,
    SavedGoogleAdc, default_adc_credentials_path as google_default_adc_credentials_path,
    exchange_google_oauth_code_and_save_adc, gemini_access_token as google_gemini_access_token,
    gemini_oauth_available as google_gemini_oauth_available,
    gemini_oauth_scopes_csv as google_gemini_oauth_scopes_csv, google_oauth_authorize_url,
    load_google_oauth_client_config,
};
pub use providers::openai_compat::{
    OpenAiCompatClient, OpenAiCompatConfig, discover_models, discover_models_with_bearer,
};
pub use providers::openai_oauth::{
    OPENAI_OAUTH_CALLBACK_PORT, account_id_from_id_token, exchange_openai_code,
    openai_authorize_url, openai_oauth_config, refresh_openai_tokens,
};
pub use providers::{
    BandDifficulty, CUSTOM_PROVIDERS_ENV, EXPERIMENTAL_PROVIDERS_ENV, MODEL_CLASSES_ENV, MODEL_CONTEXT_WINDOWS_ENV, MODEL_EFFORT_CEILINGS_ENV, ULTRA_BAND_ENV, CustomProviderUsability, ModelClass, ModelFitHint,
    NON_CLAUDE_ADAPTERS_ENV, ProviderCatalogEntry, ProviderKind, ProviderMetadata,
    apply_non_anthropic_identity, band_difficulty_for_request,
    context_window_for_model, custom_provider_catalog, custom_provider_usability_catalog, custom_provider_usable_catalog, declared_model_class, detect_provider_kind,
    effective_effort_for_model,
    effort_budget_with_floor, effort_level_for_budget, effort_rank, explicit_non_claude_provider_kind,
    fit_hint_for_model, maker_for_provider, max_supported_effort, max_tokens_for_model, model_supports_xhigh,
    non_claude_adapters_enabled, openai_gpt_model_family, provider_catalog, provider_enabled, provider_usable_for_smart_inventory,
    refresh_custom_providers_from_env, refresh_custom_providers_from_json, resolve_effort_band, resolve_model_alias, resolve_registered_model_alias,
    uses_adaptive_thinking, wire_model_id,
};
pub use sse::{SseParser, parse_frame};
pub use types::{
    CacheControl, ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent,
    ContentBlockStopEvent, DocumentSource, EffortLevel, ImageSource, InputContentBlock,
    InputMessage, MessageDelta, MessageDeltaEvent, MessageRequest, MessageResponse,
    MessageStartEvent, MessageStopEvent, OutputConfig, OutputContentBlock, StreamEvent,
    SystemBlock, ThinkingConfig, ToolChoice, ToolDefinition, ToolResultContentBlock, Usage,
    system_from_string,
};

pub use token_counting::TokenCount;

pub use oauth_store::{
    clear_google_code_assist_oauth, clear_oauth_credentials, code_challenge_s256, credentials_path,
    generate_pkce_pair, generate_state, load_google_code_assist_oauth, load_oauth_credentials,
    load_openai_compat_api_key, loopback_redirect_uri, save_google_code_assist_oauth,
    save_oauth_credentials, save_openai_compat_api_key, saved_oauth_present,
    saved_oauth_present_effective, SavedOAuthProvider,
};

pub use telemetry::{
    AnalyticsEvent, AnthropicRequestProfile, ClientIdentity, DEFAULT_ANTHROPIC_VERSION,
    JsonlTelemetrySink, MemoryTelemetrySink, SessionTraceRecord, SessionTracer, TelemetryEvent,
    TelemetrySink,
};

#[cfg(test)]
pub(crate) fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| {
        // Unit tests must not detect the developer machine's real OAuth/ADC or
        // other external credential stores when asserting the default gate.
        std::env::set_var("ZO_DISABLE_EXTERNAL_CREDENTIALS", "1");
        Mutex::new(())
    })
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
pub(crate) mod test_env {
    //! Shared, panic-safe credential-store isolation for `api` unit tests.
    //!
    //! The saved-credential lookup merges every root in the
    //! `ZO_CONFIG_HOME` → `ZO_HOME` → `~/.zo` (HOME) chain, so overriding only
    //! `ZO_CONFIG_HOME` still leaks the developer machine's real
    //! `~/.zo/credentials.json` into a test. [`CredentialEnvIsolation`] is the
    //! single owner of that isolation policy: it snapshots and restores every
    //! credential-relevant variable (even on a panicking assertion) and points
    //! all home roots at one fresh, private temp dir. Tests never read or print
    //! real credential values — they only assert on the isolated empty store.

    use std::ffi::OsString;
    use std::path::Path;
    use tempfile::TempDir;

    /// Every environment variable that can steer the saved-credential lookup or
    /// the api-key/OAuth env fallbacks. Isolating all of them in one list keeps
    /// the policy defined once rather than re-listed per test.
    const CREDENTIAL_VARS: &[&str] = &[
        "HOME",
        core_types::paths::ZO_HOME_ENV,
        core_types::paths::ZO_CONFIG_HOME_ENV,
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ];

    /// RAII guard that isolates the credential-relevant environment for the
    /// duration of a test and restores the prior values on drop, including when
    /// the test unwinds from a failed assertion. Hold it alongside
    /// [`crate::test_env_lock`] so concurrent tests never observe the override.
    ///
    /// This is the single credential-store isolation helper for `api` tests —
    /// the Google ADC tests reuse it rather than re-deriving their own `HOME`
    /// override. The private root is a [`tempfile::TempDir`], so cleanup (even on
    /// a panicking assertion) is that `TempDir`'s own `Drop`; there is no
    /// hand-rolled timestamp path or manual `remove_dir_all`.
    pub(crate) struct CredentialEnvIsolation {
        saved: Vec<(&'static str, Option<OsString>)>,
        temp_home: TempDir,
    }

    impl CredentialEnvIsolation {
        /// Snapshot the credential vars, then point `HOME`/`ZO_HOME`/
        /// `ZO_CONFIG_HOME` at one fresh private temp dir and clear the api-key
        /// and auth-token fallbacks so the store is empty.
        pub(crate) fn empty() -> Self {
            let saved = CREDENTIAL_VARS
                .iter()
                .map(|&key| (key, std::env::var_os(key)))
                .collect();
            let temp_home = TempDir::new().expect("create isolated credential temp home");
            for &key in CREDENTIAL_VARS {
                std::env::remove_var(key);
            }
            std::env::set_var("HOME", temp_home.path());
            std::env::set_var(core_types::paths::ZO_HOME_ENV, temp_home.path());
            std::env::set_var(core_types::paths::ZO_CONFIG_HOME_ENV, temp_home.path());
            Self { saved, temp_home }
        }

        /// The isolated config home (also the isolated `HOME`), so a test can
        /// seed a credential file when it needs a non-empty store or resolve a
        /// `HOME`-relative path (e.g. the Google ADC location) under it.
        pub(crate) fn config_home(&self) -> &Path {
            self.temp_home.path()
        }
    }

    impl Drop for CredentialEnvIsolation {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
            // `temp_home` (a `TempDir`) removes its directory tree on drop.
        }
    }
}
