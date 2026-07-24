use std::env::VarError;
use std::fmt::{Display, Formatter};
use std::time::Duration;

#[derive(Debug)]
pub enum ApiError {
    MissingCredentials {
        provider: &'static str,
        env_vars: &'static [&'static str],
    },
    UnsupportedProvider {
        provider: &'static str,
        gate_env: &'static str,
    },
    MissingAuthRouteCredentials {
        provider: &'static str,
        route: &'static str,
    },
    UnsupportedAuthRoute {
        provider: &'static str,
        route: &'static str,
    },
    ExpiredOAuthToken,
    Auth(String),
    InvalidApiKeyEnv(VarError),
    Http(reqwest::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    Api {
        status: reqwest::StatusCode,
        error_type: Option<String>,
        message: Option<String>,
        body: String,
        retryable: bool,
        retry_after: Option<Duration>,
    },
    StreamApi {
        error_type: Option<String>,
        message: Option<String>,
        body: String,
        retryable: bool,
    },
    RetriesExhausted {
        attempts: u32,
        last_error: Box<ApiError>,
    },
    InvalidSseFrame(&'static str),
    BackoffOverflow {
        attempt: u32,
        base_delay: Duration,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorClass {
    RateLimit { retry_after: Option<Duration> },
    Transient,
    AuthExpired,
    ContextOverflow,
    InvalidToolProtocol,
    InvalidToolSchema,
    SafetyBlocked,
    NonRetryable,
}

impl ApiError {
    #[must_use]
    pub const fn missing_credentials(
        provider: &'static str,
        env_vars: &'static [&'static str],
    ) -> Self {
        Self::MissingCredentials { provider, env_vars }
    }

    #[must_use]
    pub const fn unsupported_provider(provider: &'static str, gate_env: &'static str) -> Self {
        Self::UnsupportedProvider { provider, gate_env }
    }

    #[must_use]
    pub const fn missing_auth_route_credentials(
        provider: &'static str,
        route: &'static str,
    ) -> Self {
        Self::MissingAuthRouteCredentials { provider, route }
    }

    #[must_use]
    pub const fn unsupported_auth_route(
        provider: &'static str,
        route: &'static str,
    ) -> Self {
        Self::UnsupportedAuthRoute { provider, route }
    }

    /// A streaming response went silent for longer than the idle budget.
    ///
    /// Marked retryable so the caller's retry policy can re-establish the
    /// stream instead of hanging forever on a quietly-reasoning backend.
    #[must_use]
    pub fn stream_idle_timeout(idle: Duration) -> Self {
        Self::StreamApi {
            error_type: Some("stream_idle_timeout".to_string()),
            message: Some(format!(
                "no stream data for {}s; backend went silent",
                idle.as_secs()
            )),
            body: String::new(),
            retryable: true,
        }
    }

    /// A provider kept the transport alive but produced no task action before
    /// the startup deadline. This is distinct from a byte-level idle timeout:
    /// keep-alive frames may have arrived, but they are not model progress.
    #[must_use]
    pub fn stream_startup_no_progress(budget: Duration, reasoning_extended: bool) -> Self {
        let extension = if reasoning_extended {
            " after one reasoning-based extension"
        } else {
            ""
        };
        Self::StreamApi {
            error_type: Some("stream_startup_no_progress".to_string()),
            message: Some(format!(
                "no text or tool action within {}s{extension}; transport keep-alives are not progress",
                budget.as_secs()
            )),
            body: String::new(),
            retryable: true,
        }
    }

    /// A pre-commit stream restart could not re-establish the connection
    /// within the remaining restart wall-clock budget (the reopen request was
    /// accepted but never answered). Retryable for parity with
    /// [`Self::stream_idle_timeout`], whose budget-exhausted path surfaces the
    /// same way; the caller's own retry policy decides what happens next.
    #[must_use]
    pub fn stream_restart_timeout(remaining: Duration) -> Self {
        Self::StreamApi {
            error_type: Some("stream_restart_timeout".to_string()),
            message: Some(format!(
                "stream reopen did not complete within the remaining {}ms restart budget",
                remaining.as_millis()
            )),
            body: String::new(),
            retryable: true,
        }
    }

    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Api { retry_after, .. } => *retry_after,
            Self::RetriesExhausted { last_error, .. } => last_error.retry_after(),
            _ => None,
        }
    }

    #[must_use]
    pub fn provider_error_class(&self) -> ProviderErrorClass {
        match self {
            Self::Api {
                status,
                error_type,
                message,
                body,
                retry_after,
                ..
            } => {
                let parts = [
                    error_type.as_deref(),
                    message.as_deref(),
                    Some(body.as_str()),
                ];
                if status.as_u16() == 401 {
                    // A 401 that rejects the *client* (provider fingerprint /
                    // WAF whitelist), not the credential, can never be fixed by
                    // an OAuth refresh or `zo login` — fail fast instead of
                    // routing it through the auth-recovery retry.
                    if is_client_rejection_text(parts.iter().copied().flatten()) {
                        return ProviderErrorClass::NonRetryable;
                    }
                    return ProviderErrorClass::AuthExpired;
                }
                if status.as_u16() == 429
                    || status.as_u16() == 529
                    || parts.iter().flatten().any(|part| is_rate_limit_part(part))
                {
                    return ProviderErrorClass::RateLimit {
                        retry_after: *retry_after,
                    };
                }
                if let Some(class) = classify_provider_error_text(parts.into_iter().flatten()) {
                    return class;
                }
                let code = status.as_u16();
                if matches!(code, 408 | 409) || code >= 500 {
                    ProviderErrorClass::Transient
                } else {
                    ProviderErrorClass::NonRetryable
                }
            }
            Self::StreamApi {
                error_type,
                message,
                body,
                retryable,
            } => {
                let parts = [
                    error_type.as_deref(),
                    message.as_deref(),
                    Some(body.as_str()),
                ];
                if parts.iter().flatten().any(|part| is_rate_limit_part(part)) {
                    return ProviderErrorClass::RateLimit { retry_after: None };
                }
                if let Some(class) = classify_provider_error_text(parts.into_iter().flatten()) {
                    return class;
                }
                if *retryable {
                    ProviderErrorClass::Transient
                } else {
                    ProviderErrorClass::NonRetryable
                }
            }
            Self::Http(error) if error.is_connect() || error.is_timeout() || error.is_request() => {
                ProviderErrorClass::Transient
            }
            Self::RetriesExhausted { last_error, .. } => last_error.provider_error_class(),
            Self::MissingCredentials { .. }
            | Self::UnsupportedProvider { .. }
            | Self::MissingAuthRouteCredentials { .. }
            | Self::UnsupportedAuthRoute { .. }
            | Self::ExpiredOAuthToken
            | Self::Auth(_)
            | Self::InvalidApiKeyEnv(_) => ProviderErrorClass::AuthExpired,
            Self::Http(_)
            | Self::Io(_)
            | Self::Json(_)
            | Self::InvalidSseFrame(_)
            | Self::BackoffOverflow { .. } => ProviderErrorClass::NonRetryable,
        }
    }

    /// True when the failure is a provider rate-limit / overload signal
    /// (HTTP 429 or 529), at any retry-exhaustion depth. Classified from the
    /// structured error — the status code for the non-stream path and the
    /// error type/message for the streaming path — instead of substring-matching
    /// a flattened display string, so the adaptive governor can tell a genuine
    /// throttle apart from an auth/validation error that must fail fast.
    #[must_use]
    pub fn is_rate_limit(&self) -> bool {
        match self {
            Self::Api { status, .. } => {
                let code = status.as_u16();
                code == 429 || code == 529
            }
            Self::StreamApi {
                error_type,
                message,
                body,
                ..
            } => {
                let haystack = error_type
                    .as_deref()
                    .into_iter()
                    .chain(message.as_deref())
                    .chain(std::iter::once(body.as_str()));
                // The capacity vocabulary (429 / 529 / overloaded / rate limit /
                // too many requests) is shared with the runtime retry + stream
                // layers via `core_types::retry_signal`, so a new overload
                // wording is recognised everywhere at once. The structured `Api`
                // arm above still classifies from the HTTP status code directly.
                haystack.into_iter().any(|part| {
                    core_types::retry_signal::is_rate_limit_text(&part.to_ascii_lowercase())
                })
            }
            Self::RetriesExhausted { last_error, .. } => last_error.is_rate_limit(),
            _ => false,
        }
    }

    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout() || error.is_request(),
            Self::Api { retryable, .. } | Self::StreamApi { retryable, .. } => {
                *retryable
                    && !matches!(self.provider_error_class(), ProviderErrorClass::ContextOverflow)
            }
            Self::RetriesExhausted { last_error, .. } => last_error.is_retryable(),
            Self::MissingCredentials { .. }
            | Self::UnsupportedProvider { .. }
            | Self::MissingAuthRouteCredentials { .. }
            | Self::UnsupportedAuthRoute { .. }
            | Self::ExpiredOAuthToken
            | Self::Auth(_)
            | Self::InvalidApiKeyEnv(_)
            | Self::Io(_)
            | Self::Json(_)
            | Self::InvalidSseFrame(_)
            | Self::BackoffOverflow { .. } => false,
        }
    }

    /// True when the failure is an HTTP 401 (expired/invalid credentials), at
    /// any retry-exhaustion depth. Drives a one-shot OAuth refresh + retry so a
    /// bearer that lapses mid-turn doesn't kill the turn until a restart.
    #[must_use]
    pub fn is_unauthorized(&self) -> bool {
        match self {
            Self::Api {
                status,
                error_type,
                message,
                body,
                ..
            } => {
                // A server 401 normally means a stale bearer worth one refresh +
                // retry. But a *client-rejection* 401 (provider fingerprint /
                // whitelist) is not a credential problem — refreshing and
                // retrying with the same client identity just 401s again, so it
                // must not drive the OAuth-recovery path.
                status.as_u16() == 401
                    && !is_client_rejection_text(
                        [error_type.as_deref(), message.as_deref(), Some(body.as_str())]
                            .into_iter()
                            .flatten(),
                    )
            }
            Self::RetriesExhausted { last_error, .. } => last_error.is_unauthorized(),
            _ => false,
        }
    }
}

fn is_rate_limit_part(part: &str) -> bool {
    core_types::retry_signal::is_rate_limit_text(&part.to_ascii_lowercase())
}

/// True when a 401 body signals that the *client itself* was rejected — the
/// provider's fingerprint check / WAF whitelist refused the request before the
/// credential was ever weighed. Relay backends that only accept traffic
/// matching an official client's wire image (e.g. agentrouter, which mimics
/// the Claude Code client) return this with a distinctive
/// `unauthorized_client` type. Such a failure is not fixable by re-login: the
/// token is fine, the caller's identity is not.
fn is_client_rejection_text<'a>(parts: impl IntoIterator<Item = &'a str>) -> bool {
    parts.into_iter().any(|part| {
        let part = part.to_ascii_lowercase();
        part.contains("unauthorized_client")
            || part.contains("unauthorized client")
            || part.contains("client_not_allowed")
            || part.contains("client not allowed")
    })
}

fn classify_provider_error_text<'a>(
    parts: impl IntoIterator<Item = &'a str>,
) -> Option<ProviderErrorClass> {
    let mut text = String::new();
    for part in parts {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&part.to_ascii_lowercase());
    }

    if text.contains("safety_blocked")
        || (text.contains("safety") && (text.contains("block") || text.contains("finish_reason")))
    {
        return Some(ProviderErrorClass::SafetyBlocked);
    }
    if core_types::retry_signal::is_request_buffer_overflow_text(&text)
        || text.contains("context_length_exceeded")
        || text.contains("context window")
        || text.contains("context length")
        || text.contains("maximum context")
        || text.contains("too many tokens")
        || text.contains("token limit")
        || text.contains("exceeds the context")
    {
        return Some(ProviderErrorClass::ContextOverflow);
    }
    if text.contains("thought_signature")
        || text.contains("thoughtsignature")
        || ((text.contains("functioncall")
            || text.contains("function call")
            || text.contains("functionresponse")
            || text.contains("function response"))
            && (text.contains("missing")
                || text.contains("mismatch")
                || text.contains("protocol")
                || text.contains("number of function response parts")))
    {
        return Some(ProviderErrorClass::InvalidToolProtocol);
    }
    if text.contains("function_declarations")
        || text.contains("function declaration")
        || text.contains("tool schema")
        || text.contains("parameters.properties")
        || (text.contains("tools[") && text.contains("schema"))
    {
        return Some(ProviderErrorClass::InvalidToolSchema);
    }
    None
}

impl Display for ApiError {
    #[allow(
        clippy::too_many_lines,
        reason = "one cohesive per-variant Display match; splitting it would scatter \
                  the error-message arms (surfaced by a rustc/clippy toolchain bump, \
                  not by a feature change — matches the codebase's existing convention)"
    )]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCredentials { provider, env_vars } => {
                let credential_hint = if env_vars.is_empty() {
                    "set the provider's configured api-key env var".to_string()
                } else {
                    format!("export {}", env_vars.join(" or "))
                };
                write!(
                    f,
                    "missing {provider} credentials; {credential_hint} before calling the {provider} API"
                )?;
                if matches!(*provider, "Anthropic" | "OpenAI" | "Google") {
                    write!(
                        f,
                        "\n\n  No credentials found — authenticate this provider, then retry:\n    • TUI:    /login [provider]      (e.g. /login google, /login openai; bare /login = Claude)\n    • shell:  zo login [provider]"
                    )
                } else {
                    write!(
                        f,
                        "\n\n  No API key found for this OpenAI-compatible adapter. Set the configured auth_env in the shell that launches zo, then retry."
                    )
                }
            }
            Self::UnsupportedProvider { provider, gate_env } => write!(
                f,
                "{provider} adapter is present but disabled in Claude-first mode; configure {provider} (set its API key or base URL) or set {gate_env}=1 to enable the experimental provider path"
            ),
            Self::MissingAuthRouteCredentials { provider, route } => write!(
                f,
                "missing {provider} {route} credentials; authenticate that exact route and retry (automatic credential fallback is disabled for this model)"
            ),
            Self::UnsupportedAuthRoute { provider, route } => write!(
                f,
                "{provider} does not support the explicit {route} authentication route"
            ),
            Self::ExpiredOAuthToken => {
                write!(
                    f,
                    "saved OAuth token is expired and no refresh token is available"
                )
            }
            Self::Auth(message) => write!(f, "auth error: {message}"),
            Self::InvalidApiKeyEnv(error) => {
                write!(f, "failed to read credential environment variable: {error}")
            }
            Self::Http(error) => write!(f, "http error: {error}"),
            Self::Io(error) => write!(f, "io error: {error}"),
            Self::Json(error) => write!(f, "json error: {error}"),
            Self::Api {
                status,
                error_type,
                message,
                body,
                ..
            } => {
                match (error_type, message) {
                    (Some(error_type), Some(message)) => {
                        write!(f, "api returned {status} ({error_type}): {message}")?;
                    }
                    _ => write!(f, "api returned {status}: {body}")?,
                }
                // A 401 means the credentials themselves are bad/expired, not a
                // transient fault — point the user at recovery instead of
                // leaving a bare "Invalid authentication credentials".
                if status.as_u16() == 401 {
                    if is_client_rejection_text(
                        [error_type.as_deref(), message.as_deref(), Some(body.as_str())]
                            .into_iter()
                            .flatten(),
                    ) {
                        // The credential is fine; the provider refused *zo* as
                        // an unauthorized client (fingerprint / whitelist). Re-
                        // login cannot fix this — say so plainly instead of
                        // sending the user in a login loop.
                        write!(
                            f,
                            "\n\n  This provider rejected zo as an unauthorized client — not an expired credential. Re-running /login will not help.\n  The endpoint only accepts requests whose wire image matches a whitelisted client (User-Agent / SDK headers).\n  Fix: give this provider a client fingerprint in settings.json — add \"client_fingerprint\": \"codex\" (or \"claude-code\") to its providers[] entry, or set a raw \"user_agent\". If the endpoint accepts generic API clients instead, no fingerprint is needed."
                        )?;
                    } else {
                        // Provider-neutral: a 401 can come from any backend (Claude,
                        // Gemini, ChatGPT, …), so point at re-login generically rather
                        // than assuming an Anthropic credential.
                        write!(
                            f,
                            "\n\n  Authentication failed — credentials expired or invalid.\n  Re-authenticate this model's provider, then retry:\n    • TUI:    /login [provider]      (e.g. /login google, /login openai; bare /login = Claude)\n    • shell:  zo login [provider]"
                        )?;
                    }
                }
                Ok(())
            }
            Self::StreamApi {
                error_type,
                message,
                body,
                ..
            } => match (error_type, message) {
                (Some(error_type), Some(message)) => {
                    write!(f, "api stream error ({error_type}): {message}")?;
                    if error_type == "overloaded_error" {
                        write!(
                            f,
                            "

  Provider capacity issue — this is not a local permission or workspace-trust failure. Wait and retry, or switch to a lighter/fallback model."
                        )?;
                    }
                    Ok(())
                }
                _ => write!(f, "api stream error: {body}"),
            },
            Self::RetriesExhausted {
                attempts,
                last_error,
            } => {
                write!(f, "api failed after {attempts} attempts: {last_error}")?;
                let msg = last_error.to_string();
                if msg.contains("429") || msg.contains("rate_limit") || msg.contains("529") {
                    write!(
                        f,
                        "\n\n  Rate limited. Try:\n    1. zo login     (get your own OAuth token)\n    2. --model sonnet  (lower rate limits)\n    3. Wait a minute and retry"
                    )?;
                }
                Ok(())
            }
            Self::InvalidSseFrame(message) => write!(f, "invalid sse frame: {message}"),
            Self::BackoffOverflow {
                attempt,
                base_delay,
            } => write!(
                f,
                "retry backoff overflowed on attempt {attempt} with base delay {base_delay:?}"
            ),
        }
    }
}

impl std::error::Error for ApiError {}

impl From<reqwest::Error> for ApiError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<std::io::Error> for ApiError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<VarError> for ApiError {
    fn from(value: VarError) -> Self {
        Self::InvalidApiKeyEnv(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;

    #[test]
    fn api_401_display_appends_reauth_hint() {
        let err = ApiError::Api {
            status: StatusCode::UNAUTHORIZED,
            error_type: Some("authentication_error".to_string()),
            message: Some("Invalid authentication credentials".to_string()),
            body: String::new(),
            retryable: false,
            retry_after: None,
        };
        let rendered = err.to_string();
        assert!(
            rendered.contains("zo login"),
            "401 must point at re-auth: {rendered}"
        );
        assert!(rendered.contains("authentication_error"));
    }

    #[test]
    fn missing_credentials_display_points_at_login() {
        let err = ApiError::missing_credentials("Anthropic", &["ANTHROPIC_API_KEY"]);
        let rendered = err.to_string();
        assert!(
            rendered.contains("ANTHROPIC_API_KEY"),
            "must still name the env var: {rendered}"
        );
        assert!(
            rendered.contains("/login") && rendered.contains("zo login"),
            "missing credentials must point at the login flow: {rendered}"
        );
    }

    #[test]
    fn api_500_display_has_no_reauth_hint() {
        let err = ApiError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: Some("api_error".to_string()),
            message: Some("boom".to_string()),
            body: String::new(),
            retryable: true,
            retry_after: None,
        };
        let rendered = err.to_string();
        assert!(
            !rendered.contains("zo login"),
            "non-401 must not nag about login: {rendered}"
        );
    }

    #[test]
    fn stream_idle_timeout_is_retryable() {
        let err = ApiError::stream_idle_timeout(Duration::from_secs(90));
        assert!(err.is_retryable(), "idle timeout must be retryable");
        assert!(err.to_string().contains("90s"));
    }

    #[test]
    fn stream_startup_no_progress_is_retryable_and_distinct_from_transport_idle() {
        let err = ApiError::stream_startup_no_progress(Duration::from_secs(480), true);
        assert!(err.is_retryable());
        let rendered = err.to_string();
        assert!(rendered.contains("stream_startup_no_progress"));
        assert!(rendered.contains("480s"));
        assert!(rendered.contains("reasoning-based extension"));
    }

    #[test]
    fn is_unauthorized_detects_server_401_only() {
        let api = |status| ApiError::Api {
            status,
            error_type: None,
            message: None,
            body: String::new(),
            retryable: false,
            retry_after: None,
        };
        assert!(api(reqwest::StatusCode::UNAUTHORIZED).is_unauthorized());
        assert!(!api(reqwest::StatusCode::INTERNAL_SERVER_ERROR).is_unauthorized());
        // A 401 wrapped by the retry layer is still recognised.
        assert!(
            ApiError::RetriesExhausted {
                attempts: 3,
                last_error: Box::new(api(reqwest::StatusCode::UNAUTHORIZED)),
            }
            .is_unauthorized()
        );
        // A local "expired" signal is not a server 401 (different recovery).
        assert!(!ApiError::ExpiredOAuthToken.is_unauthorized());
    }

    #[test]
    fn is_rate_limit_detects_429_and_529_and_overload() {
        let api = |status, retry_after| ApiError::Api {
            status,
            error_type: None,
            message: None,
            body: String::new(),
            retryable: true,
            retry_after,
        };
        // 429 and 529 (Anthropic overload) are rate-limit signals.
        assert!(api(StatusCode::TOO_MANY_REQUESTS, None).is_rate_limit());
        assert!(api(StatusCode::from_u16(529).unwrap(), None).is_rate_limit());
        // A 401 / 500 is not a rate limit (must fail fast, not absorb).
        assert!(!api(StatusCode::UNAUTHORIZED, None).is_rate_limit());
        assert!(!api(StatusCode::INTERNAL_SERVER_ERROR, None).is_rate_limit());
        // Wrapped by the retry layer, the classification still holds.
        assert!(
            ApiError::RetriesExhausted {
                attempts: 3,
                last_error: Box::new(api(StatusCode::TOO_MANY_REQUESTS, None)),
            }
            .is_rate_limit()
        );
        // A streamed overload error is classified from its type/message.
        let stream_overload = ApiError::StreamApi {
            error_type: Some("overloaded_error".to_string()),
            message: Some("Overloaded".to_string()),
            body: String::new(),
            retryable: true,
        };
        assert!(stream_overload.is_rate_limit());
        let stream_other = ApiError::StreamApi {
            error_type: Some("invalid_request_error".to_string()),
            message: Some("bad input".to_string()),
            body: String::new(),
            retryable: false,
        };
        assert!(!stream_other.is_rate_limit());
    }

    #[test]
    fn stream_overloaded_error_display_names_provider_capacity() {
        let error = ApiError::StreamApi {
            error_type: Some("overloaded_error".to_string()),
            message: Some("Overloaded".to_string()),
            body: String::new(),
            retryable: true,
        };
        let rendered = error.to_string();
        assert!(rendered.contains("Provider capacity issue"));
        assert!(rendered.contains("not a local permission or workspace-trust failure"));
        assert!(error.is_retryable());
    }

    #[test]
    fn retry_after_is_surfaced_through_retry_exhaustion() {
        let inner = ApiError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            error_type: None,
            message: None,
            body: String::new(),
            retryable: true,
            retry_after: Some(Duration::from_secs(42)),
        };
        assert_eq!(inner.retry_after(), Some(Duration::from_secs(42)));
        let wrapped = ApiError::RetriesExhausted {
            attempts: 5,
            last_error: Box::new(inner),
        };
        assert_eq!(
            wrapped.retry_after(),
            Some(Duration::from_secs(42)),
            "Retry-After must survive the retry-exhaustion wrapper"
        );
    }

    fn api_error(
        status: StatusCode,
        error_type: Option<&str>,
        message: Option<&str>,
        body: &str,
        retryable: bool,
    ) -> ApiError {
        ApiError::Api {
            status,
            error_type: error_type.map(str::to_string),
            message: message.map(str::to_string),
            body: body.to_string(),
            retryable,
            retry_after: None,
        }
    }

    #[test]
    fn provider_error_classifies_429_and_529_as_rate_limit() {
        let rate_limit = ApiError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            error_type: Some("rate_limit_error".to_string()),
            message: Some("slow down".to_string()),
            body: String::new(),
            retryable: true,
            retry_after: Some(Duration::from_secs(7)),
        };
        assert_eq!(
            rate_limit.provider_error_class(),
            ProviderErrorClass::RateLimit {
                retry_after: Some(Duration::from_secs(7)),
            }
        );
        assert_eq!(
            api_error(
                StatusCode::from_u16(529).unwrap(),
                Some("overloaded_error"),
                Some("busy"),
                "",
                true,
            )
            .provider_error_class(),
            ProviderErrorClass::RateLimit { retry_after: None }
        );
    }

    #[test]
    fn provider_error_class_preserves_retry_after_through_retries_exhausted() {
        let wrapped = ApiError::RetriesExhausted {
            attempts: 3,
            last_error: Box::new(ApiError::Api {
                status: StatusCode::TOO_MANY_REQUESTS,
                error_type: None,
                message: None,
                body: String::new(),
                retryable: true,
                retry_after: Some(Duration::from_secs(42)),
            }),
        };
        assert_eq!(
            wrapped.provider_error_class(),
            ProviderErrorClass::RateLimit {
                retry_after: Some(Duration::from_secs(42)),
            }
        );
    }

    #[test]
    fn provider_error_classifies_401_as_auth_expired() {
        assert_eq!(
            api_error(StatusCode::UNAUTHORIZED, None, None, "", false).provider_error_class(),
            ProviderErrorClass::AuthExpired
        );
    }

    #[test]
    fn client_rejection_401_is_not_auth_expired_and_not_unauthorized() {
        // agentrouter-style whitelist rejection: the token is valid, but the
        // provider refuses zo as an unauthorized client. This must not drive
        // the OAuth-refresh recovery path (which would 401 identically forever),
        // and must not be classified as an expired credential.
        let err = api_error(
            StatusCode::UNAUTHORIZED,
            Some("unauthorized_client_error"),
            Some("unauthorized client detected, contact support for assistance"),
            r#"{"type":"unauthorized_client_error","message":"UNAUTHENTICATED"}"#,
            false,
        );
        assert_eq!(
            err.provider_error_class(),
            ProviderErrorClass::NonRetryable,
            "client-rejection 401 must fail fast, not route through auth recovery"
        );
        assert!(
            !err.is_unauthorized(),
            "client-rejection 401 must not trigger the one-shot OAuth refresh + retry"
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains("unauthorized client")
                && rendered.contains("Re-running /login will not help"),
            "message must explain it is a client rejection, not a credential expiry: {rendered}"
        );
        assert!(
            !rendered.contains("zo login [provider]"),
            "must not send the user into a futile re-login loop: {rendered}"
        );
        assert!(
            rendered.contains("client_fingerprint") && rendered.contains("codex"),
            "message must point at the actual fix — a client_fingerprint preset — not a dead end: {rendered}"
        );
    }

    #[test]
    fn plain_401_still_drives_reauth() {
        // A genuine credential 401 (no client-rejection signal) is unchanged:
        // classified as AuthExpired, recognised by is_unauthorized, and shown
        // with the re-login hint.
        let err = api_error(
            StatusCode::UNAUTHORIZED,
            Some("authentication_error"),
            Some("Invalid authentication credentials"),
            "",
            false,
        );
        assert_eq!(err.provider_error_class(), ProviderErrorClass::AuthExpired);
        assert!(err.is_unauthorized());
        assert!(err.to_string().contains("zo login [provider]"));
    }

    #[test]
    fn provider_error_classifies_transient_http_and_5xx() {
        assert_eq!(
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                Some("api_error"),
                Some("try later"),
                "",
                true,
            )
            .provider_error_class(),
            ProviderErrorClass::Transient
        );
        assert_eq!(
            api_error(StatusCode::REQUEST_TIMEOUT, None, None, "", true).provider_error_class(),
            ProviderErrorClass::Transient
        );
    }

    #[test]
    fn provider_error_classifies_context_overflow() {
        assert_eq!(
            api_error(
                StatusCode::BAD_REQUEST,
                Some("invalid_request_error"),
                Some("context length exceeded"),
                "maximum context window exceeded",
                false,
            )
            .provider_error_class(),
            ProviderErrorClass::ContextOverflow
        );
    }

    #[test]
    fn request_buffer_overflow_is_context_overflow_and_not_retryable() {
        let error = api_error(
            StatusCode::INSUFFICIENT_STORAGE,
            None,
            None,
            "exceeded request buffer limit while retrying upstream",
            true,
        );
        assert_eq!(
            error.provider_error_class(),
            ProviderErrorClass::ContextOverflow
        );
        assert!(!error.is_retryable());
    }

    #[test]
    fn provider_error_classifies_gemini_invalid_tool_schema_as_non_retryable() {
        assert_eq!(
            api_error(
                StatusCode::BAD_REQUEST,
                Some("INVALID_ARGUMENT"),
                Some("GenerateContentRequest.tools[0].function_declarations[0].parameters.properties: invalid schema"),
                "bad tool schema",
                false,
            )
            .provider_error_class(),
            ProviderErrorClass::InvalidToolSchema
        );
        assert!(
            !api_error(
                StatusCode::BAD_REQUEST,
                Some("INVALID_ARGUMENT"),
                Some("function_declarations parameters invalid schema"),
                "",
                false,
            )
            .is_retryable()
        );
    }

    #[test]
    fn provider_error_classifies_gemini_missing_thought_signature_as_invalid_tool_protocol() {
        assert_eq!(
            api_error(
                StatusCode::BAD_REQUEST,
                Some("INVALID_ARGUMENT"),
                Some("functionCall read_file is missing a thought_signature"),
                "Gemini tool protocol rejected the request",
                false,
            )
            .provider_error_class(),
            ProviderErrorClass::InvalidToolProtocol
        );
    }

    #[test]
    fn provider_error_classifies_safety_blocked() {
        assert_eq!(
            ApiError::StreamApi {
                error_type: Some("safety_blocked".to_string()),
                message: Some("response blocked by safety policy".to_string()),
                body: String::new(),
                retryable: false,
            }
            .provider_error_class(),
            ProviderErrorClass::SafetyBlocked
        );
    }

    #[test]
    fn provider_error_classifies_plain_400_as_non_retryable() {
        assert_eq!(
            api_error(
                StatusCode::BAD_REQUEST,
                Some("invalid_request_error"),
                Some("bad request"),
                "plain validation error",
                false,
            )
            .provider_error_class(),
            ProviderErrorClass::NonRetryable
        );
    }

    #[test]
    fn provider_error_classifies_401_as_auth_even_with_rate_limit_text() {
        assert_eq!(
            api_error(
                StatusCode::UNAUTHORIZED,
                Some("invalid_api_key"),
                Some("unauthorized; previous request mentioned rate limit"),
                "quota and rate limit diagnostics are not auth class",
                false,
            )
            .provider_error_class(),
            ProviderErrorClass::AuthExpired
        );
    }

    #[test]
    fn provider_error_class_does_not_treat_generic_schema_text_as_tool_schema() {
        assert_eq!(
            api_error(
                StatusCode::BAD_REQUEST,
                Some("invalid_request_error"),
                Some("response_format json schema is invalid"),
                "schema validation failed",
                false,
            )
            .provider_error_class(),
            ProviderErrorClass::NonRetryable
        );
        assert_eq!(
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                Some("server_error"),
                Some("internal invalid schema cache"),
                "generic schema cache failure",
                true,
            )
            .provider_error_class(),
            ProviderErrorClass::Transient
        );
    }
}
