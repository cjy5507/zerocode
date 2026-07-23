//! Single source of truth for classifying a stringified provider error into a
//! retry signal.
//!
//! Several layers independently decide "is this a transient capacity failure?"
//! from a flattened error string:
//!
//! - the `runtime` retry layer picks a backoff *schedule* (longer for capacity
//!   stalls so an overload retry doesn't hammer the pool),
//! - the `runtime` conversation layer picks a live-UI *label* ("provider
//!   overloaded" / "rate limited" / "transient provider error"),
//! - the `runtime` Anthropic stream parser decides whether a `Transport` error
//!   is a *provider-emitted* frame (server already closed the turn — surface it
//!   for a fresh request) versus a recoverable connection drop.
//!
//! Before this module each site carried its own `contains("…")` predicate, so a
//! new provider wording (e.g. a different overload phrase) had to be added in
//! every place or one layer would silently disagree with the others. Keeping the
//! substring vocabulary here means every layer classifies from the *same* words
//! and only maps the resulting [`RetrySignal`] to its own concern.
//!
//! This is a deliberately text-based classifier for the flattened-display path.
//! Callers that hold the *structured* error (e.g. `api::ApiError` with an HTTP
//! status code) should classify from the status directly; this module is the
//! shared fallback for the many places that only see a `Display` string.

/// How a stringified provider error should be treated by a retry layer.
///
/// Ordering is by escalation, not severity: a [`RateLimit`](Self::RateLimit)
/// capacity stall is "more transient" than a generic [`Transient`](Self::Transient)
/// blip in that it wants a *longer* backoff, while [`Fatal`](Self::Fatal) must
/// fail fast (auth / validation errors that retrying can never fix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrySignal {
    /// A provider capacity signal — HTTP 429/529, an `overloaded_error`, or any
    /// "rate limit" / "overloaded" wording. Retryable on the *longer* schedule.
    RateLimit,
    /// A transient non-capacity failure — a 5xx or a connection/timeout drop.
    /// Retryable on the *standard* backoff schedule.
    Transient,
    /// Not retryable — an auth, validation, or otherwise permanent error.
    Fatal,
}

impl RetrySignal {
    /// True when the signal warrants a retry at all (rate-limit or transient).
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::RateLimit | Self::Transient)
    }

    /// True for a provider capacity signal (429 / 529 / overloaded / rate limit).
    #[must_use]
    pub const fn is_rate_limit(self) -> bool {
        matches!(self, Self::RateLimit)
    }
}

/// True when the lowercased error text carries a provider *capacity* signal:
/// HTTP 429 or 529, or any "overloaded" / "rate limit" wording.
///
/// `lower` MUST already be ASCII-lowercased by the caller (see
/// [`classify_error_text`], which lowercases once and reuses the result).
#[must_use]
pub fn is_rate_limit_text(lower: &str) -> bool {
    lower.contains("429")
        || lower.contains("529")
        || lower.contains("rate limit")
        || lower.contains("rate_limit")
        || lower.contains("too many requests")
        || lower.contains("overloaded")
}

/// True when a lowercased error says an upstream retry gateway could not buffer
/// the request body. Re-sending that same body is deterministic; callers must
/// reduce it before retrying.
#[must_use]
pub fn is_request_buffer_overflow_text(lower: &str) -> bool {
    lower.contains("request buffer limit")
        && (lower.contains("exceeded") || lower.contains("exceeds"))
}

/// True when the lowercased error text carries a *non-capacity* transient
/// signal: a retryable 5xx status or a connection/timeout/EOF drop.
///
/// The structured API layer treats every HTTP status `>= 500` as retryable.
/// This flattened-text fallback cannot safely parse arbitrary status tokens, so
/// keep the common provider/gateway server-error vocabulary explicit, including
/// Cloudflare-style 52x errors surfaced as `api returned 520 <unknown status code>`.
///
/// `lower` MUST already be ASCII-lowercased by the caller.
#[must_use]
pub fn is_transient_text(lower: &str) -> bool {
    lower.contains("500")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
        || lower.contains("520")
        || lower.contains("521")
        || lower.contains("522")
        || lower.contains("523")
        || lower.contains("524")
        || lower.contains("525")
        || lower.contains("526")
        || lower.contains("connection reset")
        || lower.contains("connection refused")
        || lower.contains("connection closed")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("broken pipe")
        || lower.contains("eof")
        // reqwest/hyper mid-body drops. `error decoding response body` is
        // reqwest's Decode error — for a streaming SSE body it means the
        // connection died mid-transfer (truncated chunked/compressed stream),
        // the same class as `connection reset`, but its text matched nothing
        // here, so the turn failed fatally instead of retrying (the reported
        // "✘ turn: … error decoding response body"). `incomplete message` is
        // hyper's IncompleteMessage (server closed mid-response); `error
        // reading a body` covers hyper's body-read transport variant.
        || lower.contains("error decoding response body")
        || lower.contains("incomplete message")
        || lower.contains("error reading a body")
        // Some provider backends close an accepted streaming request with this
        // generic terminal marker and no structured status. A fresh request is
        // safe; treating it as fatal forced the user to submit `continue`.
        || lower.contains("backend reported a terminal stream failure")
}

/// Classify a raw (not yet lowercased) provider error string into a
/// [`RetrySignal`]. Lowercases once, then applies the shared vocabulary:
/// capacity signals win over generic transient signals, and anything matching
/// neither is [`RetrySignal::Fatal`].
#[must_use]
pub fn classify_error_text(error_message: &str) -> RetrySignal {
    let lower = error_message.to_ascii_lowercase();
    if lower.contains("507") && is_request_buffer_overflow_text(&lower) {
        RetrySignal::Fatal
    } else if is_rate_limit_text(&lower) {
        RetrySignal::RateLimit
    } else if is_transient_text(&lower) {
        RetrySignal::Transient
    } else {
        RetrySignal::Fatal
    }
}

/// A provider stream *mid-flight* retry, surfaced so a live UI can show the
/// otherwise-silent reconnect pause as "reconnecting", not a freeze.
///
/// Distinct from the establish-time retry the `runtime` retry layer already
/// renders: this fires when a stream that has already started reading restarts
/// its own upstream connection internally (the provider closed the HTTP body
/// pre-commit). That internal restart never returns an error to the runtime
/// retry layer, so without this notice the turn just stalls for the backoff
/// delay. Carries the already-classified [`retry_notice_label`] text plus the
/// attempt counter and backoff delay so the consumer can render a message
/// without re-parsing the error.
#[derive(Debug, Clone)]
pub struct StreamRetryNotice {
    /// What this notice describes; drives the consumer's wording.
    pub kind: StreamNoticeKind,
    /// Human-readable cause label from [`retry_notice_label`].
    pub label: &'static str,
    /// 1-based restart attempt number.
    pub attempt: u32,
    /// Maximum attempts the stream will make before surfacing a hard error.
    pub max_attempts: u32,
    /// Backoff delay before this attempt's reconnect — or, for
    /// [`StreamNoticeKind::QuietReasoning`], how long the stream has been
    /// quiet so far.
    pub delay: std::time::Duration,
}

/// What a [`StreamRetryNotice`] describes: a reconnect backoff (the stream
/// stalled and is re-opening), or a quiet-reasoning heartbeat (the connection
/// is alive and delivering keep-alive chunks, but the model has produced no
/// visible event yet — deep reasoning on a large context can stay silent for
/// minutes, and without this signal it reads as a hang).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamNoticeKind {
    #[default]
    Reconnect,
    QuietReasoning,
}

/// Label of the quiet-reasoning heartbeat notice. A shared constant because
/// two layers key on the exact wording: the streaming backend stamps it on the
/// [`StreamNoticeKind::QuietReasoning`] notice, and the TUI matches the
/// transcript row's prefix to flip its "no output" stall badge into a calm
/// "reasoning · stream alive" state (the `STEERING_ECHO_PREFIX` precedent —
/// keep every emitter and matcher on this constant).
pub const QUIET_REASONING_LABEL: &str = "model reasoning silently — stream alive";

/// Prefix of the quota-hold warning the runtime emits when a hard 429 parks
/// the turn on the same model until its quota window resets (the wait band).
/// Shared for the same reason as [`QUIET_REASONING_LABEL`]: the runtime stamps
/// it on the transcript warning, and the TUI matches the prefix to flip the
/// spinner into a "rate-limited · holding for quota reset" state — without it
/// the (up to 15-minute) sleep reads as a plain "no output" hang and users
/// Esc out of a turn that would have resumed on its own.
pub const QUOTA_HOLD_NOTICE_PREFIX: &str = "Main model rate-limited";

/// Exact system warning emitted when a Fable/Mythos safety-classifier refusal
/// retries the current turn on Opus 4.8. Shared so renderers can identify the
/// transient one-turn fallback without duplicating user-facing prose.
pub const REFUSAL_FALLBACK_WARN: &str =
    "Fable safety classifier declined this request — retrying with Opus 4.8.";

/// Exact system warning emitted on the first turn pre-armed by the
/// session-scoped refusal cooldown. Kept beside [`REFUSAL_FALLBACK_WARN`] so
/// renderers and marker-stability tests share one refusal-fallback vocabulary.
pub const REFUSAL_DRY_PREARM_WARN: &str = "Fable safety classifier declined 2 consecutive turns — \
parking this session on claude-opus-4-8 for ~30m; Fable will retry automatically afterward.";

/// Prefix of both quota-fallback notices. The fallback model id follows this
/// prefix and ends at the first `;`, allowing renderers to distinguish an
/// active provider swap from [`QUOTA_HOLD_NOTICE_PREFIX`] without parsing the
/// rest of the human-readable explanation.
pub const QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX: &str = "Quota fallback active on ";

/// Extract the active fallback model from a quota-fallback system notice.
///
/// Returns `None` for quota-hold notices and malformed or empty fallback
/// markers. The returned slice borrows the model id directly from `text`.
#[must_use]
pub fn parse_quota_fallback_model(text: &str) -> Option<&str> {
    let rest = text.strip_prefix(QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX)?;
    let (model, _) = rest.split_once(';')?;
    (!model.is_empty()).then_some(model)
}

/// A short, human-readable label for a stringified provider error, for a live
/// "retrying in Ns" notice. Distinguishes an explicit *overload* from a plain
/// rate limit so the user can see which capacity wall they hit, and keeps that
/// wording in the same file as the classifier so it never drifts out of sync.
#[must_use]
pub fn retry_notice_label(error_message: &str) -> &'static str {
    let lower = error_message.to_ascii_lowercase();
    if lower.contains("overloaded") {
        "provider overloaded"
    } else if is_rate_limit_text(&lower) {
        "rate limited"
    } else {
        "transient provider error"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX, QUOTA_HOLD_NOTICE_PREFIX, REFUSAL_DRY_PREARM_WARN,
        REFUSAL_FALLBACK_WARN, RetrySignal, classify_error_text, parse_quota_fallback_model,
    };

    #[test]
    fn capacity_signals_classify_as_rate_limit() {
        for msg in [
            "HTTP 429 Too Many Requests",
            "overloaded_error: Overloaded",
            "rate_limit_error: rate limit exceeded",
            "upstream OVERLOADED",
            "api stream error (529)",
            "Too Many Requests",
        ] {
            assert_eq!(
                classify_error_text(msg),
                RetrySignal::RateLimit,
                "{msg:?} must be a rate-limit signal"
            );
        }
    }

    #[test]
    fn non_capacity_transient_signals_classify_as_transient() {
        for msg in [
            "HTTP 500 Internal Server Error",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout",
            "api returned 520 <unknown status code>: error code: 520",
            "Cloudflare 522 connection timed out",
            "request timed out",
            "connection reset by peer",
            "broken pipe",
            // A streaming body that dies mid-transfer surfaces through
            // reqwest/hyper as a decode/read failure, not a named
            // connection error — it must retry, not kill the turn.
            "transport error: http error: error decoding response body",
            "connection error: incomplete message",
            "error reading a body from connection",
            "turn: runtime: provider stream: transport error: api stream error: backend reported a terminal stream failure",
        ] {
            assert_eq!(
                classify_error_text(msg),
                RetrySignal::Transient,
                "{msg:?} must be a transient signal"
            );
        }
    }

    #[test]
    fn permanent_errors_classify_as_fatal() {
        for msg in [
            "authentication_error: invalid API key",
            "invalid_request_error: messages too long",
            "401 Unauthorized",
            "507 Insufficient Storage: exceeded request buffer limit while retrying upstream",
        ] {
            assert_eq!(
                classify_error_text(msg),
                RetrySignal::Fatal,
                "{msg:?} must be fatal (fail fast)"
            );
        }
    }

    #[test]
    fn capacity_wins_over_transient_when_both_present() {
        // A 529 alongside a 503-looking body is still a capacity stall.
        assert_eq!(
            classify_error_text("api stream error 529 (503 backend)"),
            RetrySignal::RateLimit
        );
    }

    #[test]
    fn retryability_helpers_agree_with_variant() {
        assert!(RetrySignal::RateLimit.is_retryable());
        assert!(RetrySignal::RateLimit.is_rate_limit());
        assert!(RetrySignal::Transient.is_retryable());
        assert!(!RetrySignal::Transient.is_rate_limit());
        assert!(!RetrySignal::Fatal.is_retryable());
        assert!(!RetrySignal::Fatal.is_rate_limit());
    }

    #[test]
    fn notice_label_distinguishes_overload_from_rate_limit() {
        use super::retry_notice_label;
        assert_eq!(
            retry_notice_label("api stream error (overloaded_error): Overloaded"),
            "provider overloaded"
        );
        assert_eq!(
            retry_notice_label("HTTP 429 Too Many Requests"),
            "rate limited"
        );
        assert_eq!(retry_notice_label("rate_limit exceeded"), "rate limited");
        assert_eq!(
            retry_notice_label("connection reset by peer"),
            "transient provider error"
        );
    }

    #[test]
    fn fallback_notice_markers_are_stable_and_unambiguous() {
        let model = "openai:gpt-5.6-sol";
        let swap = format!(
            "{QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX}{model}; the main model is rate-limited"
        );
        let prearm = format!(
            "{QUOTA_FALLBACK_ACTIVE_NOTICE_PREFIX}{model}; the main model is still cooling down"
        );
        assert_eq!(parse_quota_fallback_model(&swap), Some(model));
        assert_eq!(parse_quota_fallback_model(&prearm), Some(model));

        let hold = format!("{QUOTA_HOLD_NOTICE_PREFIX} (claude-fable-5); holding this turn");
        assert_eq!(parse_quota_fallback_model(&hold), None);
        assert_eq!(parse_quota_fallback_model("Quota fallback active on ; malformed"), None);
        assert_eq!(
            REFUSAL_FALLBACK_WARN,
            "Fable safety classifier declined this request — retrying with Opus 4.8."
        );
        assert_eq!(
            REFUSAL_DRY_PREARM_WARN,
            "Fable safety classifier declined 2 consecutive turns — parking this session on \
             claude-opus-4-8 for ~30m; Fable will retry automatically afterward."
        );
    }
}
