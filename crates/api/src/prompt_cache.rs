use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::types::{InputMessage, MessageRequest, MessageResponse, Usage};

/// Local TTL for the on-disk completion cache: how long a stored
/// `MessageResponse` may be replayed for a byte-identical request. Deliberately
/// short — the request fingerprint covers model/system/tools/messages but not
/// the files those messages reference, so a longer window risks replaying an
/// answer after the underlying tree changed. Identical requests are rare inside
/// a turn loop (messages grow each turn), so a small window loses little.
const DEFAULT_COMPLETION_TTL_SECS: u64 = 30;
/// Mirrors the *provider's* server-side prompt-cache lifetime so
/// [`detect_cache_break`] can tell a legitimate TTL expiry from an unexpected
/// break. Zo requests the extended 1-hour cache (`CacheControl::ephemeral_1h`,
/// the `extended-cache-ttl-2025-04-11` beta) on every breakpoint, so this must
/// be 1 hour to match — at the old 5-minute value any cache read 5–60 min after
/// the previous turn was misclassified as an *unexpected* break. Anthropic's
/// cache is a sliding window (each hit refreshes the TTL), so within an active
/// session the prefix effectively stays warm.
const DEFAULT_PROMPT_TTL_SECS: u64 = 60 * 60;
const DEFAULT_BREAK_MIN_DROP: u32 = 2_000;
const MAX_SANITIZED_LENGTH: usize = 80;
const REQUEST_FINGERPRINT_VERSION: u32 = 1;
const REQUEST_FINGERPRINT_PREFIX: &str = "v1";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Minimum re-billed volume (input + cache-creation tokens) for a request to
/// count toward the low-cache-hit-ratio streak. Below this floor a poor ratio
/// is cheap noise (a short request naturally has little to read from cache);
/// above it, a poor ratio means real money re-billed.
const LOW_CACHE_HIT_VOLUME_FLOOR: u64 = 50_000;
/// Consecutive low-cache-hit requests that trip the one-time warning. An edge
/// trigger — the warning fires only the request the streak first reaches this
/// value, not on every subsequent request, so a long-running degraded session
/// gets one line instead of one per turn.
const LOW_CACHE_HIT_STREAK_WARNING_THRESHOLD: u32 = 3;

#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    pub session_id: String,
    pub completion_ttl: Duration,
    pub prompt_ttl: Duration,
    pub cache_break_min_drop: u32,
}

impl PromptCacheConfig {
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            completion_ttl: Duration::from_secs(DEFAULT_COMPLETION_TTL_SECS),
            prompt_ttl: Duration::from_secs(DEFAULT_PROMPT_TTL_SECS),
            cache_break_min_drop: DEFAULT_BREAK_MIN_DROP,
        }
    }
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self::new("default")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCachePaths {
    pub root: PathBuf,
    pub session_dir: PathBuf,
    pub completion_dir: PathBuf,
    pub session_state_path: PathBuf,
    pub stats_path: PathBuf,
}

impl PromptCachePaths {
    #[must_use]
    pub fn for_session(session_id: &str) -> Self {
        let root = base_cache_root();
        let session_dir = root.join(sanitize_path_segment(session_id));
        let completion_dir = session_dir.join("completions");
        Self {
            root,
            session_state_path: session_dir.join("session-state.json"),
            stats_path: session_dir.join("stats.json"),
            session_dir,
            completion_dir,
        }
    }

    #[must_use]
    pub fn completion_entry_path(&self, request_hash: &str) -> PathBuf {
        self.completion_dir.join(format!("{request_hash}.json"))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheStats {
    pub tracked_requests: u64,
    pub completion_cache_hits: u64,
    pub completion_cache_misses: u64,
    pub completion_cache_writes: u64,
    pub expected_invalidations: u64,
    pub unexpected_cache_breaks: u64,
    pub total_cache_creation_input_tokens: u64,
    pub total_cache_read_input_tokens: u64,
    pub last_cache_creation_input_tokens: Option<u32>,
    pub last_cache_read_input_tokens: Option<u32>,
    pub last_request_hash: Option<String>,
    pub last_completion_cache_key: Option<String>,
    pub last_break_reason: Option<String>,
    pub last_cache_source: Option<String>,
    /// Index of the first message whose hash differs from the immediately
    /// preceding request's message at the same position. `None` when the
    /// current request's messages are a pure prefix-preserving extension of
    /// the previous request (ordinary turn growth) or when there is no prior
    /// request to compare against (first request this process has observed).
    /// See [`first_divergence`].
    #[serde(default)]
    pub last_first_divergence_index: Option<usize>,
    /// Length of the matching prefix between this request's messages and the
    /// previous request's — i.e. how many leading messages are byte-identical
    /// before [`Self::last_first_divergence_index`] (or the full overlap when
    /// there is no divergence).
    #[serde(default)]
    pub last_prefix_stable_messages: usize,
    /// Message count of the immediately preceding tracked request (0 if none).
    #[serde(default)]
    pub last_prev_message_count: usize,
    /// Message count of the most recently tracked request.
    #[serde(default)]
    pub last_message_count: usize,
    /// Consecutive requests (ending at the most recent) whose cache-hit ratio
    /// was below 20% while re-billing more than
    /// [`LOW_CACHE_HIT_VOLUME_FLOOR`] tokens. Resets to 0 the moment a request
    /// clears either threshold.
    #[serde(default)]
    pub low_cache_hit_streak: u32,
    /// Lifetime count of requests that counted toward a low-cache-hit streak
    /// (i.e. every request that incremented [`Self::low_cache_hit_streak`],
    /// including ones that did not themselves trip the warning).
    #[serde(default)]
    pub total_low_cache_hit_requests: u64,
    /// Re-billed tokens (input + cache-creation) accumulated across the
    /// in-progress low-cache-hit streak, reset whenever the streak breaks.
    /// Backs the "~`XXk` tokens" figure in [`format_low_cache_hit_warning`].
    /// Persisted (rather than kept in a transient field) for the same reason
    /// [`TrackedPromptState::message_hashes`] is: the non-Anthropic path
    /// reconstructs `PromptCache` fresh on every call, so only disk-backed
    /// state survives between consecutive requests in a streak.
    #[serde(default)]
    pub low_cache_hit_streak_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheBreakEvent {
    pub unexpected: bool,
    pub reason: String,
    pub previous_cache_read_input_tokens: u32,
    pub current_cache_read_input_tokens: u32,
    pub token_drop: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCacheRecord {
    pub cache_break: Option<CacheBreakEvent>,
    pub stats: PromptCacheStats,
    /// Set on the request where [`PromptCacheStats::low_cache_hit_streak`]
    /// first reaches [`LOW_CACHE_HIT_STREAK_WARNING_THRESHOLD`] — a one-line,
    /// one-time-per-streak notice for a caller to surface to the user
    /// (independent of `cache_break`, which stays `None` when the cache is
    /// merely *staying* cold rather than freshly dropping).
    pub low_cache_hit_warning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PromptCache {
    inner: Arc<Mutex<PromptCacheInner>>,
}

impl PromptCache {
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_config(PromptCacheConfig::new(session_id))
    }

    #[must_use]
    pub fn with_config(config: PromptCacheConfig) -> Self {
        let paths = PromptCachePaths::for_session(&config.session_id);
        let stats = read_json::<PromptCacheStats>(&paths.stats_path).unwrap_or_default();
        let previous = read_json::<TrackedPromptState>(&paths.session_state_path);
        Self {
            inner: Arc::new(Mutex::new(PromptCacheInner {
                config,
                paths,
                stats,
                previous,
            })),
        }
    }

    #[must_use]
    pub fn paths(&self) -> PromptCachePaths {
        self.lock().paths.clone()
    }

    #[must_use]
    pub fn stats(&self) -> PromptCacheStats {
        self.lock().stats.clone()
    }

    #[must_use]
    pub fn lookup_completion(&self, request: &MessageRequest) -> Option<MessageResponse> {
        let request_hash = request_hash_hex(request);
        let (paths, ttl) = {
            let inner = self.lock();
            (inner.paths.clone(), inner.config.completion_ttl)
        };
        let entry_path = paths.completion_entry_path(&request_hash);
        let entry = read_json::<CompletionCacheEntry>(&entry_path);
        let Some(entry) = entry else {
            let mut inner = self.lock();
            inner.stats.completion_cache_misses += 1;
            inner.stats.last_completion_cache_key = Some(request_hash);
            persist_state(&inner);
            return None;
        };

        if entry.fingerprint_version != current_fingerprint_version() {
            let mut inner = self.lock();
            inner.stats.completion_cache_misses += 1;
            inner.stats.last_completion_cache_key = Some(request_hash.clone());
            let _ = fs::remove_file(entry_path);
            persist_state(&inner);
            return None;
        }

        let expired = now_unix_secs().saturating_sub(entry.cached_at_unix_secs) >= ttl.as_secs();
        let mut inner = self.lock();
        inner.stats.last_completion_cache_key = Some(request_hash.clone());
        if expired {
            inner.stats.completion_cache_misses += 1;
            let _ = fs::remove_file(entry_path);
            persist_state(&inner);
            return None;
        }

        inner.stats.completion_cache_hits += 1;
        apply_usage_to_stats(
            &mut inner.stats,
            &entry.response.usage,
            &request_hash,
            "completion-cache",
        );
        inner.previous = Some(TrackedPromptState::from_usage(
            request,
            &entry.response.usage,
        ));
        persist_state(&inner);
        Some(entry.response)
    }

    #[must_use]
    pub fn record_response(
        &self,
        request: &MessageRequest,
        response: &MessageResponse,
    ) -> PromptCacheRecord {
        self.record_usage_internal(request, &response.usage, Some(response))
    }

    #[must_use]
    pub fn record_usage(&self, request: &MessageRequest, usage: &Usage) -> PromptCacheRecord {
        self.record_usage_internal(request, usage, None)
    }

    fn record_usage_internal(
        &self,
        request: &MessageRequest,
        usage: &Usage,
        response: Option<&MessageResponse>,
    ) -> PromptCacheRecord {
        let request_hash = request_hash_hex(request);
        let mut inner = self.lock();
        let previous = inner.previous.clone();
        let fingerprints = RequestFingerprints::from_request(request);
        let current = TrackedPromptState::from_fingerprints(&fingerprints, usage);

        // `previous.message_hashes` — NOT a separate in-memory field — is the
        // basis for divergence comparison. This matters: `record_usage_internal`
        // is called through two very different lifetimes. The Anthropic client
        // holds one `PromptCache` for the whole session, so an in-memory-only
        // field would survive there; but `record_non_anthropic_prompt_cache_usage`
        // (the GPT / OpenAI-compatible path) constructs a *fresh* `PromptCache`
        // on every single call — any state that lived only in `PromptCacheInner`
        // would be discarded before the next request and `first_divergence_index`
        // would silently degrade to always-`None` (worse: it would mislabel a
        // real mid-history edit as "append-only"). Riding along on
        // `TrackedPromptState`, which already round-trips through
        // `session-state.json` on every `PromptCache::new()` regardless of
        // instance lifetime, is what makes this work identically on both paths.
        let previous_message_hashes = previous.as_ref().map(|state| state.message_hashes.as_slice());
        let (first_divergence_index, prefix_stable_count) =
            first_divergence(previous_message_hashes, &fingerprints.message_hashes);
        let prev_message_count = previous.as_ref().map_or(0, |state| state.message_hashes.len());
        let current_message_count = fingerprints.message_hashes.len();

        let cache_break = detect_cache_break(
            &inner.config,
            previous.as_ref(),
            &current,
            first_divergence_index,
            current_message_count,
        );

        inner.stats.tracked_requests += 1;
        apply_usage_to_stats(&mut inner.stats, usage, &request_hash, "api-response");
        inner.stats.last_first_divergence_index = first_divergence_index;
        inner.stats.last_prefix_stable_messages = prefix_stable_count;
        inner.stats.last_prev_message_count = prev_message_count;
        inner.stats.last_message_count = current_message_count;
        if let Some(event) = &cache_break {
            if event.unexpected {
                inner.stats.unexpected_cache_breaks += 1;
            } else {
                inner.stats.expected_invalidations += 1;
            }
            inner.stats.last_break_reason = Some(event.reason.clone());
        }

        let low_cache_hit_warning = record_low_cache_hit_streak(
            &mut inner.stats,
            usage,
            first_divergence_index,
            current_message_count,
        );

        inner.previous = Some(current);
        if let Some(response) = response {
            write_completion_entry(&inner.paths, &request_hash, response);
            inner.stats.completion_cache_writes += 1;
        }
        persist_state(&inner);

        PromptCacheRecord {
            cache_break,
            stats: inner.stats.clone(),
            low_cache_hit_warning,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PromptCacheInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Debug)]
struct PromptCacheInner {
    config: PromptCacheConfig,
    paths: PromptCachePaths,
    stats: PromptCacheStats,
    previous: Option<TrackedPromptState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompletionCacheEntry {
    cached_at_unix_secs: u64,
    #[serde(default = "current_fingerprint_version")]
    fingerprint_version: u32,
    response: MessageResponse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TrackedPromptState {
    observed_at_unix_secs: u64,
    #[serde(default = "current_fingerprint_version")]
    fingerprint_version: u32,
    model_hash: u64,
    system_hash: u64,
    tools_hash: u64,
    messages_hash: u64,
    cache_read_input_tokens: u32,
    /// Per-message hash vector for this request, in order — the basis for
    /// [`first_divergence`] on the *next* request. Persisted alongside the
    /// rest of `TrackedPromptState` (mirrored to `session-state.json`)
    /// rather than kept in a separate process-memory field: `PromptCache`
    /// is reconstructed fresh on every call on the non-Anthropic path
    /// (`record_non_anthropic_prompt_cache_usage` builds a new instance per
    /// request), so anything that isn't disk-backed here would silently
    /// never see a "previous" vector to compare against on that path.
    /// `#[serde(default)]` so a `session-state.json` written before this
    /// field existed deserializes as an empty vector — divergence detection
    /// degrades to "no basis for comparison" for one request after an
    /// upgrade, then resumes normally, rather than failing to load at all.
    #[serde(default)]
    message_hashes: Vec<u64>,
}

impl TrackedPromptState {
    fn from_usage(request: &MessageRequest, usage: &Usage) -> Self {
        let hashes = RequestFingerprints::from_request(request);
        Self::from_fingerprints(&hashes, usage)
    }

    fn from_fingerprints(hashes: &RequestFingerprints, usage: &Usage) -> Self {
        Self {
            observed_at_unix_secs: now_unix_secs(),
            fingerprint_version: current_fingerprint_version(),
            model_hash: hashes.model,
            system_hash: hashes.system,
            tools_hash: hashes.tools,
            messages_hash: hashes.messages,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            message_hashes: hashes.message_hashes.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct RequestFingerprints {
    model: u64,
    system: u64,
    tools: u64,
    /// Aggregate hash of the whole `messages` array (single hash over the
    /// serialized Vec, `cache_control` markers stripped — see
    /// [`strip_message_cache_markers`]) — what [`detect_cache_break`]'s
    /// "message payload changed" check keys off of. Kept alongside
    /// `message_hashes` (below) rather than derived from it so the
    /// break-detection comparison stays a single-hash equality.
    messages: u64,
    /// Per-message hash, one entry per `messages[i]`, in order. Powers
    /// [`first_divergence`] — this is the piece the aggregate `messages` hash
    /// cannot answer ("which message changed", not just "something changed").
    message_hashes: Vec<u64>,
}

impl RequestFingerprints {
    fn from_request(request: &MessageRequest) -> Self {
        Self {
            model: hash_serializable(&request.model),
            system: hash_serializable(&request.system),
            tools: hash_serializable(&request.tools),
            messages: hash_serializable(&strip_message_cache_markers(&request.messages)),
            message_hashes: hash_messages(&request.messages),
        }
    }
}

/// Per-message FNV hash, one entry per element of `messages`, in order, with
/// `cache_control` markers stripped before hashing (see
/// [`strip_message_cache_markers`]). Reuses [`hash_serializable`] (the same
/// stable FNV hasher the aggregate request fingerprint uses) so a given
/// message hashes identically whether it's hashed alone or as part of the
/// whole array.
fn hash_messages(messages: &[InputMessage]) -> Vec<u64> {
    strip_message_cache_markers(messages)
        .iter()
        .map(hash_serializable)
        .collect()
}

/// Lower `messages` to JSON with every `cache_control` key removed, for
/// fingerprinting only.
///
/// The conversation breakpoint markers (`mark_conversation_cache_breakpoints`)
/// ride the newest two messages and therefore *move forward on every
/// request by design*. The provider's prefix cache keys on content, not on
/// the markers, so a moved marker is invisible to the cache — but hashing the
/// raw blocks made the fingerprints see a fake mid-history edit at the old
/// marker position on every call: `first_divergence` pinned a bogus
/// "history diverged at message N" a couple of messages from the tail, and
/// [`detect_cache_break`] misfiled genuinely *unexpected* token drops under
/// the expected "message payload changed" reason. Stripping the markers makes
/// the fingerprint track what the provider cache actually keys on.
fn strip_message_cache_markers(messages: &[InputMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|message| {
            let mut value = serde_json::to_value(message).unwrap_or_default();
            strip_cache_control(&mut value);
            value
        })
        .collect()
}

fn strip_cache_control(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.remove("cache_control");
            for nested in map.values_mut() {
                strip_cache_control(nested);
            }
        }
        serde_json::Value::Array(items) => {
            for nested in items {
                strip_cache_control(nested);
            }
        }
        _ => {}
    }
}

/// Find the first position where `current`'s per-message hashes diverge from
/// `previous`'s, returning `(first divergent index, matching-prefix length)`.
///
/// Returns `(None, 0)` when there is no previous vector to compare against
/// (the first request this process has observed). Returns `(None, n)` when
/// `current` is a prefix-preserving extension (or contraction) of
/// `previous` — the ordinary case of a turn appending new messages, which
/// must NOT be reported as a divergence even though the aggregate
/// `messages_hash` differs on every such turn. Otherwise returns
/// `(Some(index), index)` for the first index whose hash differs.
fn first_divergence(previous: Option<&[u64]>, current: &[u64]) -> (Option<usize>, usize) {
    let Some(previous) = previous else {
        return (None, 0);
    };
    let common = previous.len().min(current.len());
    match (0..common).find(|&index| previous[index] != current[index]) {
        Some(index) => (Some(index), index),
        None => (None, common),
    }
}

fn detect_cache_break(
    config: &PromptCacheConfig,
    previous: Option<&TrackedPromptState>,
    current: &TrackedPromptState,
    first_divergence_index: Option<usize>,
    current_message_count: usize,
) -> Option<CacheBreakEvent> {
    let previous = previous?;
    if previous.fingerprint_version != current.fingerprint_version {
        return Some(CacheBreakEvent {
            unexpected: false,
            reason: format!(
                "fingerprint version changed (v{} -> v{})",
                previous.fingerprint_version, current.fingerprint_version
            ),
            previous_cache_read_input_tokens: previous.cache_read_input_tokens,
            current_cache_read_input_tokens: current.cache_read_input_tokens,
            token_drop: previous
                .cache_read_input_tokens
                .saturating_sub(current.cache_read_input_tokens),
        });
    }
    let token_drop = previous
        .cache_read_input_tokens
        .saturating_sub(current.cache_read_input_tokens);
    if token_drop < config.cache_break_min_drop {
        return None;
    }

    let mut reasons: Vec<String> = Vec::new();
    if previous.model_hash != current.model_hash {
        reasons.push("model changed".to_string());
    }
    if previous.system_hash != current.system_hash {
        reasons.push("system prompt changed".to_string());
    }
    if previous.tools_hash != current.tools_hash {
        reasons.push("tool definitions changed".to_string());
    }
    if previous.messages_hash != current.messages_hash {
        // Enrich with *where* the history diverged — the piece the old
        // "message payload changed" wording could never answer, so every
        // ordinary tail-append turn (which also changes the aggregate hash)
        // looked identical to an actual mid-history edit in `last_break_reason`.
        let detail = match first_divergence_index {
            Some(index) => format!("history diverged at message {index}/{current_message_count}"),
            None => "append-only, no earlier message changed".to_string(),
        };
        reasons.push(format!("message payload changed ({detail})"));
    }

    let elapsed = current
        .observed_at_unix_secs
        .saturating_sub(previous.observed_at_unix_secs);

    let (unexpected, reason) = if reasons.is_empty() {
        if elapsed > config.prompt_ttl.as_secs() {
            (
                false,
                format!("possible prompt cache TTL expiry after {elapsed}s"),
            )
        } else {
            (
                true,
                "cache read tokens dropped while prompt fingerprint remained stable".to_string(),
            )
        }
    } else {
        (false, reasons.join(", "))
    };

    Some(CacheBreakEvent {
        unexpected,
        reason,
        previous_cache_read_input_tokens: previous.cache_read_input_tokens,
        current_cache_read_input_tokens: current.cache_read_input_tokens,
        token_drop,
    })
}

/// Ratio-based cache-efficiency streak tracker (spec item B). Updates
/// `stats.low_cache_hit_streak` / `stats.total_low_cache_hit_requests` every
/// call, and returns `Some(message)` only on the edge transition where the
/// streak first reaches [`LOW_CACHE_HIT_STREAK_WARNING_THRESHOLD`] — not on
/// every request past it, so a long degraded stretch produces one warning
/// instead of spamming one per turn, and a recovery-then-relapse produces a
/// fresh warning rather than staying permanently silent.
fn record_low_cache_hit_streak(
    stats: &mut PromptCacheStats,
    usage: &Usage,
    first_divergence_index: Option<usize>,
    current_message_count: usize,
) -> Option<String> {
    let cache_read = u64::from(usage.cache_read_input_tokens);
    let rebilled = u64::from(usage.input_tokens) + u64::from(usage.cache_creation_input_tokens);
    let denom = (cache_read + rebilled).max(1);
    // ratio = cache_read / denom < 0.2  <=>  cache_read * 5 < denom (integer
    // comparison — avoids floating point for a value that only ever gates a
    // streak counter).
    let low_ratio = cache_read.saturating_mul(5) < denom;
    let low_hit_request = low_ratio && rebilled > LOW_CACHE_HIT_VOLUME_FLOOR;

    if !low_hit_request {
        stats.low_cache_hit_streak = 0;
        stats.low_cache_hit_streak_tokens = 0;
        return None;
    }

    stats.low_cache_hit_streak = stats.low_cache_hit_streak.saturating_add(1);
    stats.total_low_cache_hit_requests = stats.total_low_cache_hit_requests.saturating_add(1);
    stats.low_cache_hit_streak_tokens = stats.low_cache_hit_streak_tokens.saturating_add(rebilled);

    if stats.low_cache_hit_streak != LOW_CACHE_HIT_STREAK_WARNING_THRESHOLD {
        return None;
    }

    Some(format_low_cache_hit_warning(
        stats.low_cache_hit_streak,
        stats.low_cache_hit_streak_tokens,
        first_divergence_index,
        current_message_count,
    ))
}

fn format_low_cache_hit_warning(
    streak: u32,
    rebilled_tokens: u64,
    first_divergence_index: Option<usize>,
    current_message_count: usize,
) -> String {
    let tokens_k = rebilled_tokens / 1_000;
    match first_divergence_index {
        Some(index) => format!(
            "prompt cache degraded: {streak} consecutive requests re-billed ~{tokens_k}k tokens (history diverges at message #{index}/{current_message_count})"
        ),
        None => format!(
            "prompt cache degraded: {streak} consecutive requests re-billed ~{tokens_k}k tokens"
        ),
    }
}

fn apply_usage_to_stats(
    stats: &mut PromptCacheStats,
    usage: &Usage,
    request_hash: &str,
    source: &str,
) {
    stats.total_cache_creation_input_tokens += u64::from(usage.cache_creation_input_tokens);
    stats.total_cache_read_input_tokens += u64::from(usage.cache_read_input_tokens);
    stats.last_cache_creation_input_tokens = Some(usage.cache_creation_input_tokens);
    stats.last_cache_read_input_tokens = Some(usage.cache_read_input_tokens);
    stats.last_request_hash = Some(request_hash.to_string());
    stats.last_cache_source = Some(source.to_string());
}

fn persist_state(inner: &PromptCacheInner) {
    let _ = ensure_cache_dirs(&inner.paths);
    let _ = write_json(&inner.paths.stats_path, &inner.stats);
    if let Some(previous) = &inner.previous {
        let _ = write_json(&inner.paths.session_state_path, previous);
    }
}

fn write_completion_entry(
    paths: &PromptCachePaths,
    request_hash: &str,
    response: &MessageResponse,
) {
    let _ = ensure_cache_dirs(paths);
    let entry = CompletionCacheEntry {
        cached_at_unix_secs: now_unix_secs(),
        fingerprint_version: current_fingerprint_version(),
        response: response.clone(),
    };
    let _ = write_json(&paths.completion_entry_path(request_hash), &entry);
}

fn ensure_cache_dirs(paths: &PromptCachePaths) -> std::io::Result<()> {
    ensure_private_dir(&paths.root)?;
    ensure_private_dir(&paths.session_dir)?;
    ensure_private_dir(&paths.completion_dir)
}

fn ensure_private_dir(path: &Path) -> std::io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => {},
        Ok(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("prompt cache directory is not a directory: {}", path.display()),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // The cache root lives two levels below the config home
            // (`<home>/cache/prompt-cache`), so the parent chain may not exist
            // yet. Create the ancestors best-effort with `create_dir_all` — we
            // deliberately do NOT tighten their permissions here: an ancestor
            // may be a shared, pre-existing directory (the config home, a temp
            // root) that this process does not own, and chmod-ing those would
            // fail with `EPERM`. Only the leaf cache directories (created here
            // and restricted below) are ours to make owner-only.
            if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            match fs::create_dir(path) {
                Ok(()) => {},
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    return ensure_private_dir(path);
                }
                Err(error) => return Err(error),
            }
        }
        Err(error) => return Err(error),
    }
    core_types::paths::restrict_permissions_owner_only(path)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let json = serde_json::to_vec_pretty(value)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    // The cache files are as sensitive as credentials (they hold prompt text),
    // so they use the same owner-only, symlink-rejecting, creation-time-`0o600`
    // write policy — reuse the single shared implementation rather than keeping
    // a second copy here. `LeaveParent` preserves the prompt cache's existing
    // directory semantics: `ensure_private_dir` already created and restricted
    // the leaf cache dirs, and their ancestors may be shared, pre-existing
    // directories this process does not own (chmod-ing those would `EPERM`), so
    // the writer must not touch the parent.
    core_types::paths::write_private_file(
        path,
        &json,
        &core_types::paths::ParentDirPolicy::LeaveParent,
    )
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let primary_root = base_cache_root();
    let relative = path.strip_prefix(&primary_root).ok()?;
    // Cache snapshots contain cumulative counters and a single latest prompt;
    // they cannot be merged safely, so use the first valid high-to-low copy.
    // A later persist writes that selected state to the primary root.
    for root in cache_roots() {
        let Ok(bytes) = fs::read(root.join(relative)) else {
            continue;
        };
        if let Ok(value) = serde_json::from_slice(&bytes) {
            return Some(value);
        }
    }
    None
}

fn request_hash_hex(request: &MessageRequest) -> String {
    format!(
        "{REQUEST_FINGERPRINT_PREFIX}-{:016x}",
        hash_serializable(request)
    )
}

fn hash_serializable<T: Serialize>(value: &T) -> u64 {
    let json = serde_json::to_vec(value).unwrap_or_default();
    stable_hash_bytes(&json)
}

fn sanitize_path_segment(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    if sanitized.len() <= MAX_SANITIZED_LENGTH {
        return sanitized;
    }
    let suffix = format!("-{:x}", hash_string(value));
    format!(
        "{}{}",
        &sanitized[..MAX_SANITIZED_LENGTH.saturating_sub(suffix.len())],
        suffix
    )
}

fn hash_string(value: &str) -> u64 {
    stable_hash_bytes(value.as_bytes())
}

fn cache_roots() -> Vec<PathBuf> {
    let homes = core_types::paths::zo_global_config_roots();
    let homes = if homes.is_empty() {
        vec![core_types::paths::default_config_home()]
    } else {
        homes
    };
    homes
        .into_iter()
        .map(|home| home.join("cache").join("prompt-cache"))
        .collect()
}

fn base_cache_root() -> PathBuf {
    cache_roots()
        .into_iter()
        .next()
        .expect("cache_roots always includes the primary config home")
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

const fn current_fingerprint_version() -> u32 {
    REQUEST_FINGERPRINT_VERSION
}

fn stable_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::{
        base_cache_root, detect_cache_break, ensure_private_dir, first_divergence, read_json,
        request_hash_hex, sanitize_path_segment, write_json, PromptCache, PromptCacheConfig,
        PromptCachePaths, TrackedPromptState, REQUEST_FINGERPRINT_PREFIX,
    };
    // Tests here mutate the process-wide ZO_CONFIG_HOME env var, so they must
    // serialize through the single crate-wide env lock rather than a private one;
    // two independent locks would let parallel tests race on the same env var.
    use crate::test_env_lock;
    use crate::types::{InputMessage, MessageRequest, MessageResponse, OutputContentBlock, Usage};

    #[test]
    fn path_builder_sanitizes_session_identifier() {
        let paths = PromptCachePaths::for_session("session:/with spaces");
        let session_dir = paths
            .session_dir
            .file_name()
            .and_then(|value| value.to_str())
            .expect("session dir name");
        assert_eq!(session_dir, "session--with-spaces");
        assert!(paths.completion_dir.ends_with("completions"));
        assert!(paths.stats_path.ends_with("stats.json"));
        assert!(paths.session_state_path.ends_with("session-state.json"));
    }

    #[test]
    fn request_fingerprint_drives_unexpected_break_detection() {
        let request = sample_request("same");
        let previous = TrackedPromptState::from_usage(
            &request,
            &Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 6_000,
                output_tokens: 0,
            },
        );
        let current = TrackedPromptState::from_usage(
            &request,
            &Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 1_000,
                output_tokens: 0,
            },
        );
        let event =
            detect_cache_break(&PromptCacheConfig::default(), Some(&previous), &current, None, 1)
                .expect("break should be detected");
        assert!(event.unexpected);
        assert!(event.reason.contains("stable"));
    }

    #[test]
    fn changed_prompt_marks_break_as_expected() {
        let previous_request = sample_request("first");
        let current_request = sample_request("second");
        let previous = TrackedPromptState::from_usage(
            &previous_request,
            &Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 6_000,
                output_tokens: 0,
            },
        );
        let current = TrackedPromptState::from_usage(
            &current_request,
            &Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 1_000,
                output_tokens: 0,
            },
        );
        // The two single-message requests differ at message index 0, matching
        // what `first_divergence` would compute for this exact pair — passed
        // explicitly here since this test drives `detect_cache_break` directly
        // rather than through `PromptCache::record_usage`.
        let event =
            detect_cache_break(&PromptCacheConfig::default(), Some(&previous), &current, Some(0), 1)
                .expect("break should be detected");
        assert!(!event.unexpected);
        assert!(event.reason.contains("message payload changed"));
        assert!(event.reason.contains("history diverged at message 0/1"));
    }

    #[test]
    fn completion_cache_round_trip_persists_recent_response() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let cache = PromptCache::new("unit-test-session");
        let request = sample_request("cache me");
        let response = sample_response(42, 12, "cached");

        assert!(cache.lookup_completion(&request).is_none());
        let record = cache.record_response(&request, &response);
        assert!(record.cache_break.is_none());

        let cached = cache
            .lookup_completion(&request)
            .expect("cached response should load");
        assert_eq!(cached.content, response.content);

        let stats = cache.stats();
        assert_eq!(stats.completion_cache_hits, 1);
        assert_eq!(stats.completion_cache_misses, 1);
        assert_eq!(stats.completion_cache_writes, 1);

        let persisted = read_json::<super::PromptCacheStats>(&cache.paths().stats_path)
            .expect("stats should persist");
        assert_eq!(persisted.completion_cache_hits, 1);

        std::fs::remove_dir_all(temp_root).expect("cleanup temp root");
        std::env::remove_var("ZO_CONFIG_HOME");
    }

    #[test]
    fn distinct_requests_do_not_collide_in_completion_cache() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-distinct-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let cache = PromptCache::new("distinct-request-session");
        let first_request = sample_request("first");
        let second_request = sample_request("second");

        let response = sample_response(42, 12, "cached");
        let _ = cache.record_response(&first_request, &response);

        assert!(cache.lookup_completion(&second_request).is_none());

        std::fs::remove_dir_all(temp_root).expect("cleanup temp root");
        std::env::remove_var("ZO_CONFIG_HOME");
    }

    #[test]
    fn expired_completion_entries_are_not_reused() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-expired-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let cache = PromptCache::with_config(PromptCacheConfig {
            session_id: "expired-session".to_string(),
            completion_ttl: Duration::ZERO,
            ..PromptCacheConfig::default()
        });
        let request = sample_request("expire me");
        let response = sample_response(7, 3, "stale");

        let _ = cache.record_response(&request, &response);

        assert!(cache.lookup_completion(&request).is_none());
        let stats = cache.stats();
        assert_eq!(stats.completion_cache_hits, 0);
        assert_eq!(stats.completion_cache_misses, 1);

        std::fs::remove_dir_all(temp_root).expect("cleanup temp root");
        std::env::remove_var("ZO_CONFIG_HOME");
    }

    #[test]
    fn distinct_zo_homes_do_not_share_cache_dir() {
        // Regression: base_cache_root() previously ignored ZO_CONFIG_HOME, so
        // two Zo homes sharing one HOME silently shared a legacy cache root.
        // The root must track ZO_CONFIG_HOME so the homes remain isolated.
        let _guard = test_env_lock();
        let prior = std::env::var_os("ZO_CONFIG_HOME");

        std::env::set_var("ZO_CONFIG_HOME", "/tmp/zo-home-a");
        let root_a = base_cache_root();

        std::env::set_var("ZO_CONFIG_HOME", "/tmp/zo-home-b");
        let root_b = base_cache_root();

        assert_ne!(
            root_a, root_b,
            "distinct ZO_CONFIG_HOME values must not share a cache root"
        );
        assert!(
            root_a.starts_with("/tmp/zo-home-a"),
            "cache root must live under ZO_CONFIG_HOME, got {}",
            root_a.display()
        );
        assert!(
            root_b.ends_with("cache/prompt-cache"),
            "cache root must keep its prompt-cache suffix, got {}",
            root_b.display()
        );

        match prior {
            Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
            None => std::env::remove_var("ZO_CONFIG_HOME"),
        }
    }

    #[test]
    fn sanitize_path_caps_long_values() {
        let long_value = "x".repeat(200);
        let sanitized = sanitize_path_segment(&long_value);
        assert!(sanitized.len() <= 80);
    }

    #[test]
    fn request_hashes_are_versioned_and_stable() {
        let request = sample_request("stable");
        let first = request_hash_hex(&request);
        let second = request_hash_hex(&request);
        assert_eq!(first, second);
        assert!(first.starts_with(REQUEST_FINGERPRINT_PREFIX));
    }

    #[test]
    fn env_guarded_tests_use_the_shared_crate_lock() {
        // Regression: previously this module owned a private env lock that did not
        // serialize against crate::test_env_lock, so tests in both could mutate
        // ZO_CONFIG_HOME concurrently. Acquiring the lock here must trip the
        // shared lock's side effect (set on first init), proving we route through
        // the single crate-wide lock instead of a separate private one.
        let _guard = test_env_lock();
        assert_eq!(
            std::env::var("ZO_DISABLE_EXTERNAL_CREDENTIALS").as_deref(),
            Ok("1"),
            "prompt_cache env-guarded tests must hold the shared crate::test_env_lock"
        );
    }

    // --- Prompt-cache forensics: per-message first-divergence index (spec A) ---

    /// The conversation cache-breakpoint markers move to the newest messages
    /// on every request BY DESIGN; the provider prefix cache keys on content,
    /// not markers. The fingerprints must therefore ignore `cache_control`
    /// entirely: two requests whose only difference is marker position hash
    /// identically — no bogus "history diverged at message N" and no break
    /// misfiled as "message payload changed".
    #[test]
    fn moving_cache_breakpoints_do_not_register_as_divergence() {
        with_temp_cache("marker-movement", |cache| {
            let marked = |marker_on: usize| {
                let mut request = sample_request_with_messages(&["one", "two", "three"]);
                let crate::types::InputContentBlock::Text { cache_control, .. } =
                    &mut request.messages[marker_on].content[0]
                else {
                    panic!("expected a Text block");
                };
                *cache_control = Some(crate::types::CacheControl::ephemeral_1h());
                request
            };

            let _ = cache.record_usage(&marked(1), &low_hit_usage());
            // Same content, marker advanced from message 1 to message 2 — the
            // exact shape every follow-up turn produces.
            let record = cache.record_usage(&marked(2), &low_hit_usage());

            assert_eq!(
                record.stats.last_first_divergence_index, None,
                "a moved marker must not read as a mid-history edit"
            );
            assert_eq!(record.stats.last_prefix_stable_messages, 3);
            // And a token drop under a stable content fingerprint must stay
            // classified as UNEXPECTED, not swallowed by "message payload
            // changed" at the marker position.
            if let Some(cache_break) = record.cache_break {
                assert!(
                    !cache_break.reason.contains("message payload changed"),
                    "marker movement leaked into break classification: {}",
                    cache_break.reason
                );
            }
        });
    }

    #[test]
    fn first_divergence_helper_handles_no_previous_and_pure_append() {
        // No previous vector at all (fresh process / first request): nothing
        // to compare, so no divergence is reported.
        assert_eq!(first_divergence(None, &[1, 2, 3]), (None, 0));
        // Pure prefix-preserving extension (ordinary turn growth): the
        // aggregate messages hash would differ, but per-message comparison
        // must still say "no divergence".
        assert_eq!(first_divergence(Some(&[1, 2]), &[1, 2, 3]), (None, 2));
        // A message inside the shared prefix changed.
        assert_eq!(first_divergence(Some(&[1, 2, 3]), &[1, 9, 3]), (Some(1), 1));
    }

    #[test]
    fn divergence_index_is_none_for_pure_append() {
        with_temp_cache("divergence-append", |cache| {
            let first = sample_request_with_messages(&["hello"]);
            let _ = cache.record_usage(&first, &low_hit_usage());

            let second = sample_request_with_messages(&["hello", "world"]);
            let record = cache.record_usage(&second, &low_hit_usage());

            assert_eq!(record.stats.last_first_divergence_index, None);
            assert_eq!(record.stats.last_prefix_stable_messages, 1);
            assert_eq!(record.stats.last_prev_message_count, 1);
            assert_eq!(record.stats.last_message_count, 2);
        });
    }

    #[test]
    fn divergence_index_flags_change_at_message_zero() {
        with_temp_cache("divergence-first", |cache| {
            let first = sample_request_with_messages(&["hello", "world"]);
            let _ = cache.record_usage(&first, &low_hit_usage());

            let second = sample_request_with_messages(&["goodbye", "world"]);
            let record = cache.record_usage(&second, &low_hit_usage());

            assert_eq!(record.stats.last_first_divergence_index, Some(0));
            assert_eq!(record.stats.last_prefix_stable_messages, 0);
        });
    }

    #[test]
    fn divergence_index_flags_change_in_the_middle() {
        with_temp_cache("divergence-middle", |cache| {
            let first = sample_request_with_messages(&["a", "b", "c"]);
            let _ = cache.record_usage(&first, &low_hit_usage());

            let second = sample_request_with_messages(&["a", "changed", "c"]);
            let record = cache.record_usage(&second, &low_hit_usage());

            assert_eq!(record.stats.last_first_divergence_index, Some(1));
            assert_eq!(record.stats.last_prefix_stable_messages, 1);
            assert_eq!(record.stats.last_prev_message_count, 3);
            assert_eq!(record.stats.last_message_count, 3);
        });
    }

    /// Regression: divergence tracking must not depend on a `PromptCache`
    /// instance staying alive between calls. The Anthropic client holds one
    /// long-lived `PromptCache` per session, but the non-Anthropic path
    /// (`record_non_anthropic_prompt_cache_usage` in `crates/runtime`)
    /// constructs a *fresh* `PromptCache::new(session_id)` on every single
    /// call. An earlier version of this instrumentation kept the per-message
    /// hash vector in a process-memory-only field on `PromptCacheInner`,
    /// which silently discarded it between such calls — divergence detection
    /// degraded to always-`None` on that path, and worse, mislabeled a real
    /// mid-history edit as "append-only, no earlier message changed". This
    /// test drives two independent `PromptCache::new()` instances against the
    /// same session id, exactly as the non-Anthropic seam does.
    #[test]
    fn divergence_index_survives_across_fresh_cache_instances() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-fresh-instance-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let session_id = "fresh-instance-divergence";

        let first_request = sample_request_with_messages(&["a", "b"]);
        let _ = PromptCache::new(session_id).record_usage(&first_request, &low_hit_usage());

        // A brand-new instance for the second call — never touches the first
        // instance's in-process state, only what it persisted to disk.
        let second_request = sample_request_with_messages(&["a", "changed"]);
        let record = PromptCache::new(session_id).record_usage(&second_request, &low_hit_usage());

        assert_eq!(
            record.stats.last_first_divergence_index,
            Some(1),
            "divergence must be detected purely from persisted state, \
             independent of whether the previous PromptCache instance is still alive"
        );

        std::fs::remove_dir_all(temp_root).expect("cleanup temp root");
        std::env::remove_var("ZO_CONFIG_HOME");
    }

    // --- Prompt-cache forensics: low-cache-hit-ratio streak warning (spec B) ---

    #[test]
    fn low_cache_hit_streak_warns_once_then_resets_on_recovery() {
        with_temp_cache("low-hit-streak", |cache| {
            let request = sample_request_with_messages(&["hello"]);

            let r1 = cache.record_usage(&request, &low_hit_usage());
            assert!(r1.low_cache_hit_warning.is_none());
            assert_eq!(r1.stats.low_cache_hit_streak, 1);

            let r2 = cache.record_usage(&request, &low_hit_usage());
            assert!(r2.low_cache_hit_warning.is_none());
            assert_eq!(r2.stats.low_cache_hit_streak, 2);

            let r3 = cache.record_usage(&request, &low_hit_usage());
            let warning = r3.low_cache_hit_warning.expect("streak of 3 should warn");
            assert!(warning.contains("3 consecutive requests"), "{warning}");
            assert_eq!(r3.stats.low_cache_hit_streak, 3);

            // 4th consecutive low-hit request: the streak keeps counting but
            // must NOT re-warn — an edge trigger, not a level trigger.
            let r4 = cache.record_usage(&request, &low_hit_usage());
            assert!(r4.low_cache_hit_warning.is_none());
            assert_eq!(r4.stats.low_cache_hit_streak, 4);

            // Recovery (a healthy-ratio request) clears the streak.
            let recovered = cache.record_usage(&request, &high_hit_usage());
            assert!(recovered.low_cache_hit_warning.is_none());
            assert_eq!(recovered.stats.low_cache_hit_streak, 0);

            // Relapse after recovery must warn again, not stay permanently
            // silent because the streak already fired once.
            let _ = cache.record_usage(&request, &low_hit_usage());
            let _ = cache.record_usage(&request, &low_hit_usage());
            let relapse = cache.record_usage(&request, &low_hit_usage());
            assert!(
                relapse.low_cache_hit_warning.is_some(),
                "warning should fire again after a recovery + relapse"
            );

            let stats = cache.stats();
            assert_eq!(stats.total_low_cache_hit_requests, 7);
        });
    }

    #[test]
    fn low_cache_hit_streak_ignores_small_requests() {
        // A poor ratio on a small request (below LOW_CACHE_HIT_VOLUME_FLOOR)
        // must not count toward the streak — otherwise short exchanges with
        // naturally little to read from cache would falsely alarm.
        with_temp_cache("low-hit-small", |cache| {
            let request = sample_request_with_messages(&["hi"]);
            let tiny_usage = Usage {
                input_tokens: 100,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 1,
                output_tokens: 5,
            };
            for _ in 0..5 {
                let record = cache.record_usage(&request, &tiny_usage);
                assert!(record.low_cache_hit_warning.is_none());
                assert_eq!(record.stats.low_cache_hit_streak, 0);
            }
        });
    }

    /// Regression, mirroring `divergence_index_survives_across_fresh_cache_instances`:
    /// the streak counters AND the accumulated re-billed-token figure behind
    /// the warning message must survive across independent `PromptCache`
    /// instances, since that is exactly how the non-Anthropic seam calls in
    /// (a fresh instance per request).
    #[test]
    fn low_cache_hit_streak_survives_across_fresh_cache_instances() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-fresh-instance-streak-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let session_id = "fresh-instance-streak";
        let request = sample_request_with_messages(&["hello"]);

        let r1 = PromptCache::new(session_id).record_usage(&request, &low_hit_usage());
        assert!(r1.low_cache_hit_warning.is_none());
        let r2 = PromptCache::new(session_id).record_usage(&request, &low_hit_usage());
        assert!(r2.low_cache_hit_warning.is_none());
        let r3 = PromptCache::new(session_id).record_usage(&request, &low_hit_usage());
        let warning = r3
            .low_cache_hit_warning
            .expect("streak of 3 should warn even across fresh instances");
        // With a fresh instance per call (the real non-Anthropic pattern), the
        // accumulated token figure must reflect all 3 requests, not just the
        // last one — proving the accumulator is disk-backed, not discarded
        // between calls.
        assert!(
            warning.contains("re-billed ~180k tokens"),
            "expected the cumulative 3x60k re-billed figure, got: {warning}"
        );

        std::fs::remove_dir_all(temp_root).expect("cleanup temp root");
        std::env::remove_var("ZO_CONFIG_HOME");
    }

    // --- Backward compatibility: old stats.json must still deserialize ---

    #[test]
    fn stats_deserializes_from_pre_instrumentation_json() {
        let old_json = r#"{
            "tracked_requests": 10,
            "completion_cache_hits": 2,
            "completion_cache_misses": 3,
            "completion_cache_writes": 4,
            "expected_invalidations": 1,
            "unexpected_cache_breaks": 1,
            "total_cache_creation_input_tokens": 100,
            "total_cache_read_input_tokens": 200,
            "last_cache_creation_input_tokens": 5,
            "last_cache_read_input_tokens": 6,
            "last_request_hash": "v1-deadbeef",
            "last_completion_cache_key": "v1-deadbeef",
            "last_break_reason": "model changed",
            "last_cache_source": "api-response"
        }"#;
        let stats: super::PromptCacheStats = serde_json::from_str(old_json)
            .expect("old-format stats.json (pre-instrumentation) must still deserialize");
        assert_eq!(stats.tracked_requests, 10);
        assert_eq!(stats.last_first_divergence_index, None);
        assert_eq!(stats.last_prefix_stable_messages, 0);
        assert_eq!(stats.last_prev_message_count, 0);
        assert_eq!(stats.last_message_count, 0);
        assert_eq!(stats.low_cache_hit_streak, 0);
        assert_eq!(stats.total_low_cache_hit_requests, 0);
        assert_eq!(stats.low_cache_hit_streak_tokens, 0);
    }

    /// `TrackedPromptState` also gained a field (`message_hashes`) and is
    /// mirrored to `session-state.json` — a resumed session whose
    /// `session-state.json` predates this instrumentation must still load
    /// (as "no basis for comparison yet" rather than failing to deserialize).
    #[test]
    fn tracked_prompt_state_deserializes_from_pre_instrumentation_json() {
        let old_json = r#"{
            "observed_at_unix_secs": 1700000000,
            "fingerprint_version": 1,
            "model_hash": 1,
            "system_hash": 2,
            "tools_hash": 3,
            "messages_hash": 4,
            "cache_read_input_tokens": 5000
        }"#;
        let state: TrackedPromptState = serde_json::from_str(old_json)
            .expect("old-format session-state.json (pre-instrumentation) must still deserialize");
        assert_eq!(state.message_hashes, Vec::<u64>::new());
    }

    // --- Owner-only persistence: fail safely on path-type / symlink surprises ---

    /// A freshly created cache directory and entry file must be owner-only
    /// (`0o700` / `0o600`) so other local users cannot read cached prompts and
    /// responses.
    #[cfg(unix)]
    #[test]
    fn persisted_cache_dir_and_file_are_owner_only() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = unique_temp_path("owner-only-dir");
        ensure_private_dir(&dir).expect("create private dir");
        let dir_mode = std::fs::metadata(&dir).expect("dir metadata").permissions().mode() & 0o777;
        assert_eq!(dir_mode, 0o700, "cache dir must be owner-only, got {dir_mode:o}");

        let file = dir.join("entry.json");
        write_json(&file, &serde_json::json!({ "k": "v" })).expect("write entry");
        let file_mode =
            std::fs::metadata(&file).expect("file metadata").permissions().mode() & 0o777;
        assert_eq!(file_mode, 0o600, "cache file must be owner-only, got {file_mode:o}");

        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// A symlink planted where a cache entry belongs must NOT be followed:
    /// `write_json` must fail and leave the symlink's target untouched, so a
    /// hostile link cannot redirect a cache write onto an arbitrary file.
    #[cfg(unix)]
    #[test]
    fn write_json_refuses_to_follow_a_symlink() {
        let dir = unique_temp_path("symlink-guard");
        std::fs::create_dir_all(&dir).expect("create dir");
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, "untouched\n").expect("write victim");
        let link = dir.join("entry.json");
        std::os::unix::fs::symlink(&victim, &link).expect("create symlink");

        let result = write_json(&link, &serde_json::json!({ "k": "v" }));
        assert!(result.is_err(), "write through a symlink must fail, not follow the link");
        assert_eq!(
            std::fs::read_to_string(&victim).expect("read victim"),
            "untouched\n",
            "the symlink target must be left byte-for-byte untouched"
        );

        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// If a directory already occupies the cache-entry path, `write_json` must
    /// surface a clear error rather than clobbering or panicking.
    #[test]
    fn write_json_refuses_a_non_file_path() {
        let dir = unique_temp_path("non-file");
        let occupied = dir.join("entry.json");
        std::fs::create_dir_all(&occupied).expect("create dir at entry path");

        let result = write_json(&occupied, &serde_json::json!({ "k": "v" }));
        assert!(result.is_err(), "writing onto a directory path must fail");

        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// If a non-directory (here, a plain file) sits where a cache directory
    /// belongs, `ensure_private_dir` must fail rather than treat it as usable.
    #[test]
    fn ensure_private_dir_refuses_a_non_directory() {
        let base = unique_temp_path("non-dir");
        std::fs::create_dir_all(&base).expect("create base");
        let occupied = base.join("cache");
        std::fs::write(&occupied, "not a dir\n").expect("write file at dir path");

        let result = ensure_private_dir(&occupied);
        assert!(result.is_err(), "a file where a directory belongs must fail");

        std::fs::remove_dir_all(&base).expect("cleanup");
    }

    fn unique_temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "prompt-cache-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    fn with_temp_cache(session_id: &str, body: impl FnOnce(&PromptCache)) {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-{session_id}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let cache = PromptCache::new(session_id);
        body(&cache);
        std::fs::remove_dir_all(&temp_root).expect("cleanup temp root");
        std::env::remove_var("ZO_CONFIG_HOME");
    }

    fn sample_request_with_messages(texts: &[&str]) -> MessageRequest {
        MessageRequest {
            model: "claude-3-7-sonnet-latest".to_string(),
            max_tokens: 64,
            messages: texts.iter().map(|text| InputMessage::user_text(*text)).collect(),
            system: Some(crate::types::system_from_string("system")),
            tools: None,
            tool_choice: None,
            stream: false,
            thinking: None,
            output_config: None,
            effort: None,
            effort_band_ceiling: None,
        }
    }

    fn low_hit_usage() -> Usage {
        Usage {
            input_tokens: 60_000,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 100,
            output_tokens: 10,
        }
    }

    fn high_hit_usage() -> Usage {
        Usage {
            input_tokens: 1_000,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 60_000,
            output_tokens: 10,
        }
    }

    fn sample_request(text: &str) -> MessageRequest {
        MessageRequest {
            model: "claude-3-7-sonnet-latest".to_string(),
            max_tokens: 64,
            messages: vec![InputMessage::user_text(text)],
            system: Some(crate::types::system_from_string("system")),
            tools: None,
            tool_choice: None,
            stream: false,
            thinking: None,
            output_config: None,
            effort: None,
            effort_band_ceiling: None,
        }
    }

    fn sample_response(
        cache_read_input_tokens: u32,
        output_tokens: u32,
        text: &str,
    ) -> MessageResponse {
        MessageResponse {
            id: "msg_test".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::Text {
                text: text.to_string(),
            }],
            model: "claude-3-7-sonnet-latest".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                cache_creation_input_tokens: 5,
                cache_read_input_tokens,
                output_tokens,
            },
            request_id: Some("req_test".to_string()),
            thought_signature: None,
            reasoning_replay: None,
            context_management: None,
        }
    }
}
