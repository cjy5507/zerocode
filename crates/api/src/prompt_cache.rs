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
/// gets one line instead of one per turn. Public so the doctor warns at the
/// same point the live session would have.
pub const LOW_CACHE_HIT_STREAK_WARNING_THRESHOLD: u32 = 3;

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
    /// Append-only per-request cache-break ledger (`breaks.jsonl`), one JSON
    /// line per [`CacheBreakEvent`]-bearing request. `stats.json` keeps only
    /// the LAST break's reason, which made multi-break sessions untraceable
    /// (which request broke, on which axis, was lost). No live code
    /// deserializes `PromptCachePaths` today (`for_session` is the only
    /// constructor); should an old serialized copy ever be loaded,
    /// `#[serde(default)]` yields an empty path and the best-effort ledger
    /// writer simply no-ops (`open("")` = `NotFound`) rather than failing.
    #[serde(default)]
    pub breaks_path: PathBuf,
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
            breaks_path: session_dir.join("breaks.jsonl"),
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
    /// Microcompact firings whose trim credit this session consumed, and the
    /// estimated tokens those trims cleared — the aggregate side of the
    /// per-row `trimmed_tokens_estimate` pairing, and the denominator that
    /// still counts a firing whose following drop stayed under the row
    /// threshold.
    #[serde(default)]
    pub context_trims_noted: u64,
    #[serde(default)]
    pub context_trim_tokens_noted: u64,
    pub completion_cache_hits: u64,
    pub completion_cache_misses: u64,
    pub completion_cache_writes: u64,
    pub expected_invalidations: u64,
    pub unexpected_cache_breaks: u64,
    pub total_cache_creation_input_tokens: u64,
    pub total_cache_read_input_tokens: u64,
    /// Lifetime uncached (`input_tokens`) volume. Required for an honest hit
    /// ratio: providers without a cache-creation concept (the OpenAI-compat
    /// path reports `cache_creation_input_tokens: 0` always) make
    /// `read/(read+creation)` degenerate to a constant 100%, hiding every
    /// cold request in the denominator it never entered.
    #[serde(default)]
    pub total_input_tokens: u64,
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

/// What one wire message IS, without its text: enough to name a prefix rewrite
/// without storing (or ever logging) prompt content.
///
/// The per-message hashes answer *where* history diverged; they cannot answer
/// *what* changed there, and that gap cost a full investigation: a ledger full
/// of "history diverged at message 275/1142" rows, five of them at the same
/// index, with no way to tell an in-memory microcompact clear from a
/// tool-history rewrite from a re-truncated result. A role, the block kinds,
/// and a byte count separate those three on sight.
/// Wire form: `"role|kinds|bytes"`, one line instead of six.
///
/// Not cosmetic. `session-state.json` is rewritten on every request and holds
/// one entry per message — 1,157 on a real session — and the shared writer
/// pretty-prints. As a struct that is ~170 kB re-written per request; as a
/// string it is ~40 kB. A diagnostic must not cost a gigabyte of disk writes a
/// day.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireMessageShape {
    /// Wire role — `"user"` or `"assistant"`.
    pub role: String,
    /// Block kinds in wire order, comma-joined, tool blocks named
    /// (`"tool_use:Agent,text"`). Truncated past
    /// [`MAX_SHAPE_KINDS`] blocks with a `"+N"` tail, since a coalesced tool
    /// run can carry dozens and the tail adds nothing to the diagnosis.
    pub kinds: String,
    /// Serialized bytes with `cache_control` stripped — the same form the
    /// hashes are taken over, so a size change here is exactly a change the
    /// provider's cache would see.
    pub bytes: u32,
}

/// Block kinds kept in [`WireMessageShape::kinds`] before summarizing the rest.
const MAX_SHAPE_KINDS: usize = 6;

impl Serialize for WireMessageShape {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("{}|{}|{}", self.role, self.kinds, self.bytes))
    }
}

impl<'de> Deserialize<'de> for WireMessageShape {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        // Split the byte count off the RIGHT and the role off the LEFT, so the
        // middle field is free to hold anything without a `|`.
        let (head, bytes) = raw
            .rsplit_once('|')
            .ok_or_else(|| serde::de::Error::custom("expected role|kinds|bytes"))?;
        let (role, kinds) = head
            .split_once('|')
            .ok_or_else(|| serde::de::Error::custom("expected role|kinds|bytes"))?;
        Ok(Self {
            role: role.to_string(),
            kinds: kinds.to_string(),
            bytes: bytes
                .parse()
                .map_err(|_| serde::de::Error::custom("shape byte count must be a u32"))?,
        })
    }
}

impl WireMessageShape {
    /// Build a shape from one already-`cache_control`-stripped message value.
    fn from_stripped(value: &serde_json::Value) -> Self {
        let role = value
            .get("role")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?")
            .to_string();
        let blocks = value.get("content").and_then(serde_json::Value::as_array);
        let kinds = blocks.map_or_else(String::new, |blocks| {
            let mut rendered: Vec<String> = blocks
                .iter()
                .take(MAX_SHAPE_KINDS)
                .map(Self::block_kind)
                .collect();
            if blocks.len() > MAX_SHAPE_KINDS {
                rendered.push(format!("+{}", blocks.len() - MAX_SHAPE_KINDS));
            }
            rendered.join(",")
        });
        Self {
            role,
            kinds,
            bytes: u32::try_from(
                serde_json::to_string(value).map_or(0, |serialized| serialized.len()),
            )
            .unwrap_or(u32::MAX),
        }
    }

    /// One block's kind, with the tool name appended for `tool_use` — naming
    /// the tool is what turns "a tool block changed" into "the `Agent` call was
    /// rewritten", which is the whole point of recording shapes.
    fn block_kind(block: &serde_json::Value) -> String {
        let kind = block
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        match block.get("name").and_then(serde_json::Value::as_str) {
            Some(name) => format!("{kind}:{name}"),
            None => kind.to_string(),
        }
    }

    /// `"user [tool_result] 12.4kB -> 96B"` — the one-line form the break
    /// reason carries.
    fn describe_change(previous: &Self, current: &Self) -> String {
        let role = if previous.role == current.role {
            previous.role.clone()
        } else {
            format!("{} -> {}", previous.role, current.role)
        };
        let kinds = if previous.kinds == current.kinds {
            format!("[{}]", previous.kinds)
        } else {
            format!("[{}] -> [{}]", previous.kinds, current.kinds)
        };
        format!(
            "{role} {kinds} {} -> {}",
            format_bytes(previous.bytes),
            format_bytes(current.bytes)
        )
    }
}

fn format_bytes(bytes: u32) -> String {
    if bytes >= 1024 {
        format!("{:.1}kB", f64::from(bytes) / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

/// The message a prefix rewrite happened at, in both its old and new shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergedWireMessage {
    pub index: usize,
    pub previous: WireMessageShape,
    pub current: WireMessageShape,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// The four axis flags mirror the four independent request-fingerprint hashes
// (model/system/tools/messages) — any subset can change together, so they are
// genuinely independent booleans, not an enum in disguise.
#[allow(clippy::struct_excessive_bools)]
pub struct CacheBreakEvent {
    pub unexpected: bool,
    pub reason: String,
    pub previous_cache_read_input_tokens: u32,
    pub current_cache_read_input_tokens: u32,
    pub token_drop: u32,
    /// Which fingerprint axes changed versus the previous request — the
    /// structured form of `reason`, so the per-request break ledger can be
    /// filtered without string parsing. Key signatures: `messages_changed`
    /// with `messages_truncated` is a history shrink (compaction/rewind/
    /// elision — the most common legitimate full-prefix rewrite); all four
    /// `false` with `unexpected: true` means our payload was byte-stable yet
    /// cache reads dropped (provider-side miss / eviction).
    #[serde(default)]
    pub model_changed: bool,
    #[serde(default)]
    pub system_changed: bool,
    #[serde(default)]
    pub tools_changed: bool,
    #[serde(default)]
    pub messages_changed: bool,
    /// True when the message history got SHORTER while its remaining prefix
    /// stayed intact — `first_divergence` alone cannot tell that apart from
    /// an ordinary tail append (both report no divergence index).
    #[serde(default)]
    pub messages_truncated: bool,
    /// Seconds since the previous tracked request — the TTL-expiry evidence
    /// for the all-axes-stable case.
    #[serde(default)]
    pub elapsed_secs: u64,
    /// What changed at the divergence, when there was one and both requests
    /// recorded a shape for that index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diverged_message: Option<DivergedWireMessage>,
    /// Tool names this request advertises that the previous one did not, and
    /// vice versa. Both empty on a `tools_changed` break means the NAMES are
    /// identical and something else about the definitions moved (order, a
    /// description, a schema) — a different defect with a different fix, and
    /// previously indistinguishable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_added: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_removed: Vec<String>,
    /// Wire model id of the request that broke, and its provider family. See
    /// [`CacheBreakLedgerRow::model`] for what they are for and how the family
    /// is derived — the row is the persisted copy, this is the live one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
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

/// Pending context-trim credits, keyed by prompt-cache session id.
///
/// The compaction planner (in `runtime`, which depends on this crate and so
/// cannot be called back into) deposits the estimated tokens a microcompact
/// cleared; the SAME session's next `record_usage` withdraws it and stamps
/// the figure on the break row that trim is about to cause. This is the
/// "missing third column" the ledger doc used to declare unrecordable: a row
/// said what got re-billed, never what the trim bought, so the firing could
/// not be priced. Keyed by session id so concurrent sessions/subagents in
/// one process can never claim each other's credit; a credit whose session
/// never sends another request idles harmlessly for the process lifetime.
fn pending_context_trims() -> &'static Mutex<std::collections::HashMap<String, u64>> {
    static PENDING: std::sync::OnceLock<Mutex<std::collections::HashMap<String, u64>>> = std::sync::OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Deposit a microcompact's estimated cleared tokens for `session_id`.
/// Accumulates: two firings before the next request price as one combined
/// trim, which is what the single following break row actually reflects.
pub fn note_context_trim(session_id: &str, estimated_tokens_cleared: u64) {
    if estimated_tokens_cleared == 0 {
        return;
    }
    let mut pending = pending_context_trims()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *pending.entry(session_id.to_string()).or_default() += estimated_tokens_cleared;
}

fn take_pending_context_trim(session_id: &str) -> u64 {
    pending_context_trims()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(session_id)
        .unwrap_or(0)
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
        maybe_spawn_stale_session_sweep(&paths);
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
        // Withdraw this session's pending trim credit up front: the trim
        // fired before THIS request regardless of whether a break row gets
        // written below, and the stats pair must count such firings too.
        let trimmed_tokens = take_pending_context_trim(&inner.config.session_id);
        if trimmed_tokens > 0 {
            inner.stats.context_trims_noted += 1;
            inner.stats.context_trim_tokens_noted += trimmed_tokens;
        }
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
            &request.model,
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
            // Durable per-request record: `last_break_reason` above only keeps
            // the final break, which made multi-break sessions untraceable.
            append_break_row(
                &inner.paths,
                &CacheBreakLedgerRow {
                    seq: inner.stats.tracked_requests,
                    ts_unix_secs: now_unix_secs(),
                    unexpected: event.unexpected,
                    reason: event.reason.clone(),
                    model_changed: event.model_changed,
                    system_changed: event.system_changed,
                    tools_changed: event.tools_changed,
                    messages_changed: event.messages_changed,
                    messages_truncated: event.messages_truncated,
                    first_divergence_index,
                    prefix_stable_messages: prefix_stable_count,
                    prev_message_count,
                    message_count: current_message_count,
                    prev_cache_read: event.previous_cache_read_input_tokens,
                    cache_read: event.current_cache_read_input_tokens,
                    cache_creation: usage.cache_creation_input_tokens,
                    token_drop: event.token_drop,
                    elapsed_secs: event.elapsed_secs,
                    diverged_message: event.diverged_message.clone(),
                    tools_added: event.tools_added.clone(),
                    tools_removed: event.tools_removed.clone(),
                    // Copied from the event rather than re-derived from
                    // `request` so the live event a caller surfaces and the
                    // persisted row can never disagree about which model broke.
                    model: event.model.clone(),
                    provider: event.provider.clone(),
                    trimmed_tokens_estimate: (trimmed_tokens > 0).then_some(trimmed_tokens),
                },
            );
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
    /// Per-message shape, parallel to [`Self::message_hashes`]. Persisted for
    /// the same reason the hashes are (the non-Anthropic path rebuilds
    /// `PromptCache` per request, so in-memory state would never see a
    /// previous), and `#[serde(default)]` so a state file written before this
    /// field existed still loads — the first break after an upgrade simply
    /// reports no shape.
    #[serde(default)]
    message_shapes: Vec<WireMessageShape>,
    /// Advertised tool names in wire order. Turns a `tools_changed` break from
    /// "some hash moved" into a name-level diff.
    #[serde(default)]
    tool_names: Vec<String>,
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
            message_shapes: hashes.message_shapes.clone(),
            tool_names: hashes.tool_names.clone(),
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
    /// Per-message shape, parallel to `message_hashes` — answers "what changed
    /// there", the piece a hash cannot.
    message_shapes: Vec<WireMessageShape>,
    /// Advertised tool names in wire order.
    tool_names: Vec<String>,
}

impl RequestFingerprints {
    fn from_request(request: &MessageRequest) -> Self {
        // Strip ONCE and derive all three message-side fingerprints from the
        // same values: the aggregate hash, the per-message hashes, and the
        // shapes. Serializing a 1,000-message history is not free, and the
        // three must agree about what they describe.
        let stripped = strip_message_cache_markers(&request.messages);
        Self {
            model: hash_serializable(&request.model),
            system: hash_serializable(&request.system),
            tools: hash_serializable(&request.tools),
            messages: hash_serializable(&stripped),
            message_hashes: stripped.iter().map(hash_serializable).collect(),
            message_shapes: stripped
                .iter()
                .map(WireMessageShape::from_stripped)
                .collect(),
            tool_names: request
                .tools
                .iter()
                .flatten()
                .map(|tool| tool.name.clone())
                .collect(),
        }
    }
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

/// Provider family for a wire model id, as
/// [`crate::ProviderKind::rate_limit_key`].
///
/// Reuses the provider registry's own classification (`metadata_for_model`,
/// which resolves aliases and `provider/model` refs on the way) instead of
/// prefix-matching model ids here — a second taxonomy would drift from the one
/// that actually routes requests, and this value is only useful if it agrees
/// with it. `""` when the id belongs to no known family (an unknown or
/// custom-only id), which the row then omits entirely.
///
/// Registry-only: no I/O and no auth probing (`detect_provider_kind`'s
/// credential fallbacks are deliberately NOT used — a row must describe the
/// request, not the machine's current logins). Called only when a break has
/// already been detected, so its allocations are off the per-request path.
fn provider_family_for_model(model: &str) -> &'static str {
    crate::providers::metadata_for_model(model)
        .map_or("", |metadata| metadata.provider.rate_limit_key())
}

/// The `fingerprint version changed` break: our own schema moved, so nothing
/// about the request can be compared across it. Split out of
/// [`detect_cache_break`] to keep that function readable, not because it varies.
fn fingerprint_bump_break(
    previous: &TrackedPromptState,
    current: &TrackedPromptState,
    elapsed: u64,
    model: &str,
) -> CacheBreakEvent {
    CacheBreakEvent {
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
        model_changed: false,
        system_changed: false,
        tools_changed: false,
        messages_changed: false,
        messages_truncated: false,
        elapsed_secs: elapsed,
        diverged_message: None,
        tools_added: Vec::new(),
        tools_removed: Vec::new(),
        model: model.to_string(),
        provider: provider_family_for_model(model).to_string(),
    }
}

/// Which fingerprint axes moved — the four independent hashes plus the shrink
/// flag, grouped so the reason builder takes one argument instead of five
/// booleans in a row.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools, reason = "one flag per independent fingerprint axis")]
struct BreakAxes {
    model_changed: bool,
    system_changed: bool,
    tools_changed: bool,
    messages_changed: bool,
    messages_truncated: bool,
}

/// Wire message counts on either side of the break.
#[derive(Debug, Clone, Copy)]
struct MessageCounts {
    previous: usize,
    current: usize,
}

/// One human-readable clause per changed axis, in fingerprint order. Empty when
/// nothing our side controls moved — the caller reads that as "provider-side or
/// TTL".
fn break_reasons(
    axes: BreakAxes,
    tools_added: &[String],
    tools_removed: &[String],
    diverged_message: Option<&DivergedWireMessage>,
    counts: MessageCounts,
    first_divergence_index: Option<usize>,
) -> Vec<String> {
    let mut reasons: Vec<String> = Vec::new();
    if axes.model_changed {
        reasons.push("model changed".to_string());
    }
    if axes.system_changed {
        reasons.push("system prompt changed".to_string());
    }
    if axes.tools_changed {
        reasons.push(format!(
            "tool definitions changed{}",
            format_tool_diff(tools_added, tools_removed)
        ));
    }
    if axes.messages_changed {
        // Enrich with *where* the history diverged, and with what changed there
        // — the pieces the old "message payload changed" wording could never
        // answer, so every ordinary tail-append turn (which also changes the
        // aggregate hash) looked identical to an actual mid-history edit.
        let current_message_count = counts.current;
        let detail = match first_divergence_index {
            Some(index) => {
                let shape = diverged_message.map_or_else(String::new, |diverged| {
                    format!(
                        " ({})",
                        WireMessageShape::describe_change(&diverged.previous, &diverged.current)
                    )
                });
                format!("history diverged at message {index}/{current_message_count}{shape}")
            }
            None if axes.messages_truncated => format!(
                "history truncated ({} -> {current_message_count} messages, shared prefix intact)",
                counts.previous
            ),
            None => "append-only, no earlier message changed".to_string(),
        };
        reasons.push(format!("message payload changed ({detail})"));
    }
    reasons
}

/// `model` is the wire model id of the request being recorded — the one thing
/// the tracked state cannot supply (it keeps only a hash of the model, so a
/// break row could name the axis that moved but never the model it moved on).
/// The event for a read drop on a PURE tail append (all invariants of that
/// branch hold by construction: only the messages axis changed, nothing
/// truncated or diverged, no tool diff). TTL when the gap explains it;
/// unexpected provider-side loss otherwise.
fn pure_append_break_event(
    config: &PromptCacheConfig,
    previous: &TrackedPromptState,
    current: &TrackedPromptState,
    token_drop: u32,
    elapsed: u64,
    model: &str,
) -> CacheBreakEvent {
    let (unexpected, reason) = if elapsed > config.prompt_ttl.as_secs() {
        (
            false,
            format!(
                "cache reads dropped on a pure append — possible prompt cache TTL expiry after {elapsed}s"
            ),
        )
    } else {
        (
            true,
            "cache reads dropped on a pure append (prefix unchanged) — provider-side miss or eviction"
                .to_string(),
        )
    };
    CacheBreakEvent {
        unexpected,
        reason,
        previous_cache_read_input_tokens: previous.cache_read_input_tokens,
        current_cache_read_input_tokens: current.cache_read_input_tokens,
        token_drop,
        model_changed: false,
        system_changed: false,
        tools_changed: false,
        messages_changed: true,
        messages_truncated: false,
        elapsed_secs: elapsed,
        diverged_message: None,
        tools_added: Vec::new(),
        tools_removed: Vec::new(),
        model: model.to_string(),
        provider: provider_family_for_model(model).to_string(),
    }
}

fn detect_cache_break(
    config: &PromptCacheConfig,
    previous: Option<&TrackedPromptState>,
    current: &TrackedPromptState,
    first_divergence_index: Option<usize>,
    current_message_count: usize,
    model: &str,
) -> Option<CacheBreakEvent> {
    let previous = previous?;
    let elapsed = current
        .observed_at_unix_secs
        .saturating_sub(previous.observed_at_unix_secs);
    if previous.fingerprint_version != current.fingerprint_version {
        return Some(fingerprint_bump_break(previous, current, elapsed, model));
    }
    let token_drop = previous
        .cache_read_input_tokens
        .saturating_sub(current.cache_read_input_tokens);
    if token_drop < config.cache_break_min_drop {
        return None;
    }

    let model_changed = previous.model_hash != current.model_hash;
    let system_changed = previous.system_hash != current.system_hash;
    let tools_changed = previous.tools_hash != current.tools_hash;
    let messages_changed = previous.messages_hash != current.messages_hash;
    // `first_divergence` compares only the shared prefix, so a history that
    // SHRANK with its remaining prefix intact (compaction, rewind, message
    // elision — all legitimate large cache breaks) returns `None` exactly like
    // an ordinary tail append. Distinguish it by count, or every truncation
    // masquerades as "append-only" — the mislabel that sent the cold-rewrite
    // investigation toward a phantom provider-side miss.
    let messages_truncated =
        messages_changed && current_message_count < previous.message_hashes.len();

    // Name the tools that came and went. A tool set that oscillates within one
    // conversation strands the entire prefix on every flip (the definitions sit
    // in front of the messages), and the axis flag alone never said which tools
    // — so the fix could only be guessed at.
    let (tools_added, tools_removed) = if tools_changed {
        tool_name_diff(&previous.tool_names, &current.tool_names)
    } else {
        (Vec::new(), Vec::new())
    };
    let diverged_message = first_divergence_index.and_then(|index| {
        Some(DivergedWireMessage {
            index,
            previous: previous.message_shapes.get(index)?.clone(),
            current: current.message_shapes.get(index)?.clone(),
        })
    });

    // A PURE tail append cannot legitimately drop reads: the previous request
    // is a byte-identical prefix of this one, and the provider's cache keys
    // on content, so everything it read last time is still there to read.
    // Reads dropping here is provider-side (miss, eviction, TTL) — yet the
    // reason machinery below files it under "message payload changed
    // (append-only…)" with `unexpected: false`, because an append does change
    // the aggregate messages hash. That misclassification buried 84M dropped
    // tokens across 577 ledger rows as self-inflicted-and-expected — the
    // wire-reminder silence pattern all over again. Classify it with the
    // fingerprint-stable case instead: TTL when the gap explains it,
    // unexpected otherwise.
    let pure_append = messages_changed
        && !messages_truncated
        && first_divergence_index.is_none()
        && current_message_count > previous.message_hashes.len()
        && !model_changed
        && !system_changed
        && !tools_changed;
    if pure_append {
        return Some(pure_append_break_event(
            config, previous, current, token_drop, elapsed, model,
        ));
    }

    let reasons = break_reasons(
        BreakAxes {
            model_changed,
            system_changed,
            tools_changed,
            messages_changed,
            messages_truncated,
        },
        &tools_added,
        &tools_removed,
        diverged_message.as_ref(),
        MessageCounts {
            previous: previous.message_hashes.len(),
            current: current_message_count,
        },
        first_divergence_index,
    );

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
        model_changed,
        system_changed,
        tools_changed,
        messages_changed,
        messages_truncated,
        elapsed_secs: elapsed,
        diverged_message,
        tools_added,
        tools_removed,
        model: model.to_string(),
        provider: provider_family_for_model(model).to_string(),
    })
}

/// Names present in exactly one of the two advertised tool lists, as
/// `(added, removed)`. Set semantics on purpose: a pure reordering yields two
/// empty vectors, which is itself the finding (the names are the same, so
/// something else about the definitions moved).
fn tool_name_diff(previous: &[String], current: &[String]) -> (Vec<String>, Vec<String>) {
    let before: std::collections::BTreeSet<&str> =
        previous.iter().map(String::as_str).collect();
    let after: std::collections::BTreeSet<&str> = current.iter().map(String::as_str).collect();
    let added = after
        .difference(&before)
        .map(|name| (*name).to_string())
        .collect();
    let removed = before
        .difference(&after)
        .map(|name| (*name).to_string())
        .collect();
    (added, removed)
}

/// `" (-Agent, -Workflow)"` / `" (+ToolSearch)"` / `" (same names; order or
/// definition changed)"` — the parenthetical the break reason appends.
fn format_tool_diff(added: &[String], removed: &[String]) -> String {
    if added.is_empty() && removed.is_empty() {
        return " (same names; order or definition changed)".to_string();
    }
    let names: Vec<String> = removed
        .iter()
        .map(|name| format!("-{name}"))
        .chain(added.iter().map(|name| format!("+{name}")))
        .collect();
    format!(" ({})", names.join(", "))
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
    stats.total_input_tokens += u64::from(usage.input_tokens);
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
    // The completion store is created here, by its writer, and nowhere else.
    let _ = ensure_private_dir(&paths.completion_dir);
    let entry = CompletionCacheEntry {
        cached_at_unix_secs: now_unix_secs(),
        fingerprint_version: current_fingerprint_version(),
        response: response.clone(),
    };
    let _ = write_json(&paths.completion_entry_path(request_hash), &entry);
}

/// Create the directories every session needs. NOT the completion store: that
/// one is created by its only writer, so an empty `completions/` never appears
/// for a session that stored no completion.
///
/// It used to be created here, unconditionally, per session — and since the
/// live turn loop is streaming while the completion store is written only from
/// the non-streaming `send_message`, the result was 548 directories on this
/// machine, every one of them empty. A directory that exists only to be empty
/// makes a dead feature look provisioned.
fn ensure_cache_dirs(paths: &PromptCachePaths) -> std::io::Result<()> {
    ensure_private_dir(&paths.root)?;
    ensure_private_dir(&paths.session_dir)
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

/// How long an idle per-session cache directory is retained before the sweep
/// removes it. Activity is judged by `stats.json` mtime (rewritten on every
/// tracked request), falling back to the directory's own mtime.
pub const PROMPT_CACHE_RETENTION_DAYS: u64 = 30;

/// Emergency cap on retained session directories: after the age sweep, the
/// oldest directories beyond this count are removed too. Sized as a backstop
/// well above real accumulation (measured: ~2,700 dirs over 41 days with
/// sub-agent fan-out) so the 30-day retention is what normally converges the
/// store — an earlier 512 cap did the exact opposite, deleting everything
/// but ~13 days of history on its first run.
pub const PROMPT_CACHE_MAX_SESSION_DIRS: usize = 8192;

/// Directories whose last activity is within this window are NEVER removed by
/// the count-cap branch (the age branch cannot reach them by definition). A
/// briefly-idle live session must not be collateral of a fan-out burst that
/// pushes the store past the cap.
pub const PROMPT_CACHE_CAP_TRIM_MIN_IDLE_DAYS: u64 = 7;

/// Root-level marker gating the sweep to roughly once per day; its mtime is
/// the last sweep time and its body records the last outcome
/// ([`SweepMarker`]) so `zo doctor` can attest what the janitor actually did
/// — a silent `let _ =` sweep left 2,194 deletions with zero written
/// evidence in adversarial review.
const SWEEP_MARKER_FILE: &str = ".last-sweep";

/// Rename-prefix applied to a directory before its recursive delete, so a
/// sweep interrupted mid-`remove_dir_all` leaves a clearly-marked husk that
/// the next sweep unconditionally removes — without it the half-deleted
/// directory's own mtime (bumped by the child deletions) made it look
/// freshly active and it squatted forever.
const SWEEPING_PREFIX: &str = ".sweeping-";

/// Persisted body of [`SWEEP_MARKER_FILE`]: when the last sweep ran and what
/// it did. `swept_at_unix_nanos` doubles as a content nonce so tests can
/// assert "the gate did not rewrite the marker" by bytes, not by mtime
/// (1-second-resolution filesystems can't distinguish a rewrite by mtime).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepMarker {
    pub swept_at_unix_nanos: u128,
    pub removed: usize,
    pub kept: usize,
}

/// Read the sweep marker's recorded outcome, if the body parses. A marker
/// from the touch-phase (empty) or an older build reads as `None`.
#[must_use]
pub fn read_sweep_marker(root: &Path) -> Option<SweepMarker> {
    let bytes = fs::read(root.join(SWEEP_MARKER_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Spawn the stale-session sweep on a background thread when the daily marker
/// says one is due. The marker is touched BEFORE spawning (losing the race
/// means an extra idempotent sweep; touching after would let every startup in
/// the window spawn) and rewritten AFTER the sweep with the outcome, so the
/// many fresh-instance constructions on the non-Anthropic path (one per
/// request) cost one `symlink_metadata` stat here and nothing more.
fn maybe_spawn_stale_session_sweep(paths: &PromptCachePaths) {
    let marker = paths.root.join(SWEEP_MARKER_FILE);
    if arm_sweep_marker(&marker) != SweepMarkerArming::Armed {
        return;
    }
    let root = paths.root.clone();
    let current_session_dir = paths.session_dir.clone();
    std::thread::spawn(move || {
        let outcome = sweep_stale_session_dirs(
            &root,
            &current_session_dir,
            SystemTime::now(),
            PROMPT_CACHE_MAX_SESSION_DIRS,
        );
        // Attest what happened (best-effort): the marker body is the only
        // durable record of the janitor's actions.
        if let Ok(outcome) = outcome {
            let marker_body = SweepMarker {
                swept_at_unix_nanos: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_nanos())
                    .unwrap_or(0),
                removed: outcome.removed,
                kept: outcome.kept,
            };
            if let Ok(json) = serde_json::to_vec(&marker_body) {
                let _ = core_types::paths::write_private_file(
                    &marker,
                    &json,
                    &core_types::paths::ParentDirPolicy::LeaveParent,
                );
            }
        }
    });
}

/// Outcome of [`arm_sweep_marker`]: whether this process won the right to run
/// today's sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SweepMarkerArming {
    /// Marker was missing or stale — this process re-armed it and must sweep.
    Armed,
    /// Someone swept within the last day (or the marker path is unusable) —
    /// do nothing.
    Declined,
}

/// Arm the daily sweep gate. A stale marker is re-armed by bumping its mtime
/// ONLY — truncating here erased the previous attestation, and a process
/// exiting before its detached sweep thread finished then left a permanent
/// zero-byte marker with a fresh mtime: no outcome on record and no retry
/// for 24 hours. A missing marker is created empty (there is no attestation
/// to preserve yet); anything that is not a regular file (a planted symlink
/// or directory) is refused outright.
fn arm_sweep_marker(marker: &Path) -> SweepMarkerArming {
    if let Ok(metadata) = fs::symlink_metadata(marker) {
        if !metadata.file_type().is_file() {
            return SweepMarkerArming::Declined;
        }
        let fresh = metadata
            .modified()
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|elapsed| elapsed < Duration::from_secs(24 * 60 * 60));
        if fresh {
            return SweepMarkerArming::Declined;
        }
        let rearmed = fs::OpenOptions::new()
            .write(true)
            .open(marker)
            .and_then(|file| file.set_modified(SystemTime::now()));
        if rearmed.is_err() {
            return SweepMarkerArming::Declined;
        }
    } else if core_types::paths::write_private_file(
        marker,
        b"",
        &core_types::paths::ParentDirPolicy::LeaveParent,
    )
    .is_err()
    {
        // No root yet (first session on this machine — created by the first
        // persist) or unwritable — skip; the next construction sweeps.
        return SweepMarkerArming::Declined;
    }
    SweepMarkerArming::Armed
}

/// What one sweep pass did: directories removed vs. left in place. `kept`
/// counts every surviving live session directory INCLUDING the calling
/// session's own, so it matches what a subsequent store listing (e.g. the
/// doctor's directory count) sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SweepOutcome {
    removed: usize,
    kept: usize,
}

/// Rename-then-delete: move the directory to a `.sweeping-` husk name first
/// (atomic), then delete the husk. An interrupted delete leaves a husk the
/// next sweep removes unconditionally instead of a half-empty directory
/// whose bumped mtime reads as fresh activity. Falls back to a direct
/// delete when the rename fails (e.g. a name collision).
fn remove_session_dir(root: &Path, path: &Path) -> bool {
    let husk_name = format!(
        "{SWEEPING_PREFIX}{}",
        path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()
    );
    let husk = root.join(husk_name);
    let target = if fs::rename(path, &husk).is_ok() { husk } else { path.to_path_buf() };
    fs::remove_dir_all(&target).is_ok()
}

/// Remove idle per-session cache directories under `root`, in one directory
/// pass (entries handled in `read_dir` order): leftover `.sweeping-` husks
/// unconditionally, everything whose last activity (`stats.json` mtime, else
/// directory mtime) predates the retention window, then — as an emergency
/// backstop over the survivors — the oldest beyond `max_session_dirs`,
/// EXCEPT directories active within
/// [`PROMPT_CACHE_CAP_TRIM_MIN_IDLE_DAYS`], which the cap may never touch.
/// The calling session's own directory (compared case-insensitively by name:
/// APFS default volumes are case-insensitive, and `--session-id MyRun` vs
/// `myrun` land in one directory) and anything that is not a plain directory
/// (symlinks included — `remove_dir_all` must never follow a planted link)
/// are left untouched. Best-effort throughout: a single unremovable entry
/// never aborts the sweep.
///
/// Known race, accepted: a session idle past the retention window that
/// resumes at the exact moment of a sweep can lose its cache-diagnostic
/// state (stats counters, break ledger, divergence baseline — rebuilt from
/// scratch by the next request; conversation state lives elsewhere and is
/// unaffected).
fn sweep_stale_session_dirs(
    root: &Path,
    current_session_dir: &Path,
    now: SystemTime,
    max_session_dirs: usize,
) -> std::io::Result<SweepOutcome> {
    let cutoff = now - Duration::from_secs(PROMPT_CACHE_RETENTION_DAYS * 24 * 60 * 60);
    let cap_guard =
        now - Duration::from_secs(PROMPT_CACHE_CAP_TRIM_MIN_IDLE_DAYS * 24 * 60 * 60);
    let current_name = current_session_dir
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase());
    let mut survivors: Vec<(SystemTime, PathBuf)> = Vec::new();
    let mut removed = 0usize;
    // The calling session's directory never enters `survivors` (it must not
    // be sort-fodder for the cap trim) but it IS a kept live directory.
    let mut kept_current = 0usize;
    for entry in fs::read_dir(root)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let entry_name = entry.file_name().to_string_lossy().to_lowercase();
        if Some(&entry_name) == current_name.as_ref() {
            kept_current = 1;
            continue;
        }
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if !metadata.file_type().is_dir() {
            continue; // files (the sweep marker) and symlinks stay
        }
        if entry_name.starts_with(SWEEPING_PREFIX) {
            // Husk from an interrupted earlier sweep — finish the job.
            if fs::remove_dir_all(&path).is_ok() {
                removed += 1;
            }
            continue;
        }
        let activity = fs::metadata(path.join("stats.json"))
            .and_then(|stats| stats.modified())
            .or_else(|_| metadata.modified())
            .unwrap_or(now);
        if activity < cutoff {
            if remove_session_dir(root, &path) {
                removed += 1;
            }
            continue;
        }
        survivors.push((activity, path));
    }
    let mut kept = survivors.len() + kept_current;
    if survivors.len() > max_session_dirs {
        survivors.sort_by_key(|(activity, _)| *activity);
        let excess = survivors.len() - max_session_dirs;
        for (activity, path) in survivors.into_iter().take(excess) {
            // The cap is an emergency backstop against unbounded growth, not
            // a recency contest: anything active in the last week stays even
            // when the store is over the cap (a fan-out burst must not evict
            // a briefly-idle live session).
            if activity >= cap_guard {
                continue;
            }
            if remove_session_dir(root, &path) {
                removed += 1;
                kept -= 1;
            }
        }
    }
    Ok(SweepOutcome { removed, kept })
}

/// Cache-health snapshot for `zo doctor`: the most recently active session's
/// stats and break ledger, plus store-wide retention facts.
#[derive(Debug, Clone)]
pub struct PromptCacheDoctorSummary {
    /// Sanitized directory name of the most recently active session.
    pub session_dir_name: String,
    pub stats: PromptCacheStats,
    pub breaks: Vec<CacheBreakLedgerRow>,
    /// Session directories currently in the store.
    pub store_session_dirs: usize,
    /// Age in days of the least recently active session directory.
    pub store_oldest_days: Option<u64>,
    /// The janitor's last attested outcome, if a sweep has completed on this
    /// machine. `None` also covers a marker still in its touch-phase.
    pub last_sweep: Option<SweepMarker>,
}

/// Read-only doctor probe over the prompt-cache store: locate the most
/// recently active session (by `stats.json` mtime), load its stats and break
/// ledger, and count store directories. `None` when the store does not exist
/// or holds no session with stats. Touches nothing — safe for `--check`.
#[must_use]
pub fn doctor_cache_summary() -> Option<PromptCacheDoctorSummary> {
    let root = base_cache_root();
    let entries = fs::read_dir(&root).ok()?;
    let now = SystemTime::now();
    let mut store_session_dirs = 0usize;
    let mut oldest: Option<SystemTime> = None;
    let mut latest: Option<(SystemTime, PathBuf)> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if !metadata.file_type().is_dir() {
            continue;
        }
        if entry.file_name().to_string_lossy().starts_with(SWEEPING_PREFIX) {
            // A husk mid-delete is not a live session, and its mtime (bumped
            // by the child deletions) would drag `store_oldest_days` younger.
            continue;
        }
        store_session_dirs += 1;
        let activity = fs::metadata(path.join("stats.json"))
            .and_then(|stats| stats.modified())
            .or_else(|_| metadata.modified())
            .unwrap_or(now);
        if oldest.is_none_or(|current| activity < current) {
            oldest = Some(activity);
        }
        if path.join("stats.json").exists()
            && latest.as_ref().is_none_or(|(current, _)| activity > *current)
        {
            latest = Some((activity, path));
        }
    }
    let (_, session_dir) = latest?;
    // A torn/unparseable stats.json (the writer is a non-atomic truncate)
    // must not erase the store-wide findings with it — degrade to zeroed
    // session stats (rendered as "no cache telemetry") instead of `None`.
    let stats =
        read_json::<PromptCacheStats>(&session_dir.join("stats.json")).unwrap_or_default();
    Some(PromptCacheDoctorSummary {
        session_dir_name: session_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        stats,
        breaks: read_break_ledger(&session_dir.join("breaks.jsonl")),
        store_session_dirs,
        store_oldest_days: oldest.and_then(|activity| {
            now.duration_since(activity).ok().map(|age| age.as_secs() / 86_400)
        }),
        last_sweep: read_sweep_marker(&root),
    })
}

/// One line of the per-request cache-break ledger (`breaks.jsonl`): which
/// request's cache reads DROPPED, on which fingerprint axis, and by how much.
/// `stats.json` only retains the LAST break's reason, so a session with
/// several breaks (the exact situation worth diagnosing) lost everything but
/// the final one.
///
/// Coverage contract — this is a ledger of break *transitions*, not of every
/// cold request: the session's first request (no previous to compare) and a
/// cache that STAYS cold (`token_drop` = 0) produce no row. Read it together
/// with `stats.json`'s `low_cache_hit_streak` counters, which do cover
/// sustained coldness. `seq` resets with `stats.json` (non-atomic overwrite;
/// a torn file re-zeroes `tracked_requests`), so treat `(seq, ts_unix_secs)`
/// as the ordering key, not `seq` alone.
///
/// Key signatures: `messages_truncated` = history shrank with its prefix
/// intact (compaction/rewind/elision — the common legitimate full-prefix
/// rewrite); all axis flags `false` with `unexpected: true` = our payload was
/// byte-stable yet reads dropped (provider-side miss / eviction).
///
/// `trimmed_tokens_estimate` is the trim-pricing column: the compaction
/// planner deposits each microcompact's cleared estimate through
/// [`note_context_trim`] (a session-keyed side channel — `runtime` depends on
/// this crate and cannot be called back into), and the same session's next
/// recorded request withdraws it here. Rows without a preceding trim carry
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// Same rationale as `CacheBreakEvent`: the axis flags are independent
// fingerprint dimensions, not mutually exclusive states.
#[allow(clippy::struct_excessive_bools)]
pub struct CacheBreakLedgerRow {
    /// 1-based request ordinal within this session's tracking
    /// (`stats.tracked_requests` at record time; see the coverage contract
    /// above for its reset caveat).
    pub seq: u64,
    pub ts_unix_secs: u64,
    pub unexpected: bool,
    pub reason: String,
    pub model_changed: bool,
    pub system_changed: bool,
    pub tools_changed: bool,
    pub messages_changed: bool,
    /// History got shorter while the surviving prefix stayed identical —
    /// indistinguishable from an append by divergence index alone.
    #[serde(default)]
    pub messages_truncated: bool,
    pub first_divergence_index: Option<usize>,
    /// Leading messages byte-identical to the previous request.
    #[serde(default)]
    pub prefix_stable_messages: usize,
    pub prev_message_count: usize,
    pub message_count: usize,
    pub prev_cache_read: u32,
    pub cache_read: u32,
    pub cache_creation: u32,
    pub token_drop: u32,
    /// Seconds since the previous tracked request (TTL-expiry evidence).
    pub elapsed_secs: u64,
    /// What the diverging message looked like before and after — the field that
    /// makes a rewrite attributable without a re-investigation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diverged_message: Option<DivergedWireMessage>,
    /// Name-level tool diff for a `tools_changed` break; both empty means the
    /// names matched and the definitions themselves moved.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_added: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_removed: Vec<String>,
    /// Wire model id this request was sent with, verbatim.
    ///
    /// WHY: a break's cost is provider-specific (cache-write premium, whether
    /// `cache_creation` is even reported), and the store holds rows from every
    /// provider a machine touched — foreground sessions and subagent scopes
    /// sit side by side under one root. Nothing in a row said which, so
    /// attributing cost meant inferring the provider from arithmetic
    /// artifacts. A real investigation did exactly that: it separated the
    /// populations by `cache_creation == 0` (the OpenAI-compat and Gemini
    /// paths hardcode it — see [`PromptCacheStats::total_input_tokens`]) and
    /// by `cache_read` being divisible by 128 (120/120 non-Anthropic rows
    /// were, 0/56 Anthropic rows were), and its headline median was quoted
    /// over the MIXED population before that split was found. Naming the
    /// model and family in the row ends that class of mistake.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// Provider family for [`Self::model`], as
    /// [`crate::ProviderKind::rate_limit_key`] (`anthropic` / `openai` /
    /// `google` / `xai` / `ollama`) — the repo's existing stable telemetry
    /// namespace, deliberately not a second taxonomy invented here.
    ///
    /// Derived from the wire model id, which is all `PromptCache` is handed:
    /// a custom OpenAI-compatible provider serving a Claude-named model
    /// therefore reads as `anthropic`. Threading the routed `ProviderKind` in
    /// instead would add an argument to `record_usage`/`record_response` at
    /// every call site in `runtime`/`tools`/`zo-cli`, and the model id already
    /// answers the question the ledger gets asked (which family's cache
    /// semantics and pricing apply to this row). Empty when the id matches no
    /// known family, and empty on rows written by builds before this field —
    /// which is what `skip_serializing_if` is for: old rows still load, and a
    /// row costs no bytes for a field it cannot fill.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    /// Estimated tokens the microcompact(s) firing right before this request
    /// cleared — the price-of-trim pairing: `cache_creation` on this row is
    /// what the trim COST, this field is what it BOUGHT. `None` on rows not
    /// preceded by a trim, and on rows written by builds before this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trimmed_tokens_estimate: Option<u64>,
}

/// Cause of a break whose axis flags are all `false` (the request fingerprint
/// itself did not change). Lives next to [`detect_cache_break`] — the sole
/// producer of the `reason` strings it matches — so the classification and the
/// wording can only drift together, and the pairing is pinned by tests in this
/// file rather than by a `contains()` in a far-away consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoAxisBreakCause {
    /// Byte-stable request inside the TTL window, yet reads dropped — the
    /// only genuinely provider-side case (recorded as `unexpected`).
    ProviderSide,
    /// Our own fingerprint schema version changed (an upgrade, not a leak).
    FingerprintBump,
    /// The gap since the previous request exceeded the prompt TTL.
    TtlExpiry,
    /// A reason wording this build does not recognize (e.g. a row written by
    /// a newer binary).
    Unknown,
}

impl CacheBreakLedgerRow {
    /// Classify a no-axis break by its recorded cause; `None` when any axis
    /// flag is set (the axis flags themselves are the explanation).
    #[must_use]
    pub fn no_axis_cause(&self) -> Option<NoAxisBreakCause> {
        if self.model_changed || self.system_changed || self.tools_changed || self.messages_changed
        {
            return None;
        }
        Some(if self.unexpected {
            NoAxisBreakCause::ProviderSide
        } else if self.reason.contains("fingerprint version") {
            NoAxisBreakCause::FingerprintBump
        } else if self.reason.contains("TTL") {
            NoAxisBreakCause::TtlExpiry
        } else {
            NoAxisBreakCause::Unknown
        })
    }
}

/// Append one break row to the session's `breaks.jsonl`. Best-effort by
/// contract (the ledger is a diagnostic aid; a write failure must never
/// affect the request path). Size stays naturally bounded: at most one row
/// per tracked request, a few hundred bytes each, in a per-session directory.
/// The session directory already exists here: a break requires a previous
/// tracked request, whose `persist_state` created and restricted it. Writes
/// through the shared `append_private_file` so the ledger keeps the exact
/// symlink-rejecting `O_NOFOLLOW`/`0o600` policy of every other cache file
/// (a raw `OpenOptions::append` here followed a planted symlink and chmod-ed
/// its target — caught in adversarial review).
fn append_break_row(paths: &PromptCachePaths, row: &CacheBreakLedgerRow) {
    let Ok(mut line) = serde_json::to_string(row) else {
        return;
    };
    line.push('\n');
    let _ = core_types::paths::append_private_file(&paths.breaks_path, line.as_bytes());
}

/// Read a break ledger (`breaks.jsonl`), oldest first. Lossy: unparseable
/// lines are skipped, a missing file reads as empty.
#[must_use]
pub fn read_break_ledger(breaks_path: &Path) -> Vec<CacheBreakLedgerRow> {
    let Ok(contents) = fs::read_to_string(breaks_path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
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
    use std::path::Path;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::{
        base_cache_root, detect_cache_break, ensure_private_dir, first_divergence, read_json,
        request_hash_hex, sanitize_path_segment, write_json, CacheBreakLedgerRow, PromptCache,
        PromptCacheConfig, PromptCachePaths, TrackedPromptState, REQUEST_FINGERPRINT_PREFIX,
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
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&previous),
            &current,
            None,
            1,
            TEST_MODEL,
        )
        .expect("break should be detected");
        assert!(event.unexpected);
        assert!(event.reason.contains("stable"));
    }

    /// The trim-pricing pair: a deposited microcompact credit rides the SAME
    /// session's next break row as `trimmed_tokens_estimate`, is consumed
    /// exactly once, and never crosses sessions.
    #[test]
    fn a_context_trim_credit_stamps_the_next_break_row_once_for_its_own_session() {
        use super::{note_context_trim, read_break_ledger, take_pending_context_trim};
        let _env = test_env_lock();
        let home = std::env::temp_dir().join(format!("zo-trim-pair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::env::set_var("ZO_CONFIG_HOME", &home);

        let session = format!("trim-pair-{}", std::process::id());
        let cache = PromptCache::new(session.clone());
        let usage_with_read = |cache_read: u32| Usage {
            input_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: cache_read,
            output_tokens: 0,
        };
        let _ = cache.record_usage(&sample_request_with_messages(&["a", "b"]), &usage_with_read(50_000));
        note_context_trim(&session, 12_345);
        note_context_trim("someone-else", 999);
        // The trim rewrote history mid-prefix → divergence → break row.
        let _ = cache.record_usage(
            &sample_request_with_messages(&["a", "REWRITTEN", "c"]),
            &usage_with_read(30_000),
        );
        let stats = cache.stats();
        assert_eq!(stats.context_trims_noted, 1);
        assert_eq!(stats.context_trim_tokens_noted, 12_345);
        let rows = read_break_ledger(&PromptCachePaths::for_session(&session).breaks_path);
        let row = rows.last().expect("break row written");
        assert_eq!(row.trimmed_tokens_estimate, Some(12_345), "{row:?}");
        // Consumed once: the next break carries nothing.
        let _ = cache.record_usage(
            &sample_request_with_messages(&["a", "DIFFERENT", "c"]),
            &usage_with_read(500),
        );
        let rows = read_break_ledger(&PromptCachePaths::for_session(&session).breaks_path);
        assert_eq!(rows.last().expect("second row").trimmed_tokens_estimate, None);
        // The other session's deposit is still waiting for ITS next request.
        assert_eq!(take_pending_context_trim("someone-else"), 999);
        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A pure tail append cannot legitimately drop cache reads — the previous
    /// request is a byte-identical prefix, and the provider caches on
    /// content. A drop there is provider-side, and it must be UNEXPECTED, not
    /// filed under "message payload changed (expected)": that misclass buried
    /// 84M dropped tokens across 577 production rows as self-inflicted.
    #[test]
    fn a_read_drop_on_a_pure_append_is_unexpected_not_self_inflicted() {
        let read = |request: &MessageRequest, cache_read: u32| {
            TrackedPromptState::from_usage(
                request,
                &Usage {
                    input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: cache_read,
                    output_tokens: 0,
                },
            )
        };
        let previous = read(&sample_request_with_messages(&["a", "b"]), 50_000);
        let appended = sample_request_with_messages(&["a", "b", "c"]);
        // Within the TTL: the gap cannot explain the drop — provider-side.
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&previous),
            &read(&appended, 1_000),
            None, // shared prefix identical: no divergence index, like a real append
            3,
            TEST_MODEL,
        )
        .expect("drop on append must produce a row");
        assert!(event.unexpected, "{}", event.reason);
        assert!(event.reason.contains("pure append"), "{}", event.reason);
        assert!(
            event.reason.contains("provider-side"),
            "{}",
            event.reason
        );
        // A drop AFTER the TTL window is the one self-explaining append case.
        let mut stale_previous = read(&sample_request_with_messages(&["a", "b"]), 50_000);
        stale_previous.observed_at_unix_secs =
            stale_previous.observed_at_unix_secs.saturating_sub(24 * 60 * 60);
        let ttl_event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&stale_previous),
            &read(&appended, 1_000),
            None,
            3,
            TEST_MODEL,
        )
        .expect("row still written");
        assert!(!ttl_event.unexpected, "{}", ttl_event.reason);
        assert!(ttl_event.reason.contains("TTL"), "{}", ttl_event.reason);
    }

    /// A `tools_changed` break must name the tools. Without this the ledger says
    /// only "tool definitions changed", which is where a real investigation had
    /// to start over from measurement: 41% of one day's cache-write came from
    /// this axis with no record of which tools moved.
    #[test]
    fn a_tools_changed_break_names_the_tools_that_came_and_went() {
        let with = |names: &[&str]| {
            let mut request = sample_request("same prompt");
            request.tools = Some(
                names
                    .iter()
                    .map(|name| crate::types::ToolDefinition {
                        name: (*name).to_string(),
                        description: None,
                        input_schema: serde_json::json!({"type": "object"}),
                    })
                    .collect(),
            );
            request
        };
        let state = |request: &MessageRequest, cache_read: u32| {
            TrackedPromptState::from_usage(
                request,
                &Usage {
                    input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: cache_read,
                    output_tokens: 0,
                },
            )
        };

        // A deep leg dropping its delegation tools, then the main lane restoring
        // them — the exact oscillation that stranded whole prefixes.
        let full = state(&with(&["read_file", "Agent", "Workflow"]), 200_000);
        let leg = state(&with(&["read_file"]), 0);
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&full),
            &leg,
            None,
            1,
            TEST_MODEL,
        )
        .expect("break");
        assert!(event.tools_changed);
        assert_eq!(event.tools_removed, vec!["Agent", "Workflow"]);
        assert!(event.tools_added.is_empty());
        assert!(
            event.reason.contains("(-Agent, -Workflow)"),
            "reason must name them: {}",
            event.reason
        );

        // An identical tool set is not a tools_changed break at all — whatever
        // else the request did, this axis stays quiet and names nothing.
        let back = state(&with(&["read_file", "Agent", "Workflow"]), 0);
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&full),
            &back,
            None,
            1,
            TEST_MODEL,
        )
        .expect("the token drop alone is still a break");
        assert!(!event.tools_changed);
        assert!(event.tools_added.is_empty() && event.tools_removed.is_empty());

        // Same names in a different order: both diffs empty, and the reason says
        // so rather than implying a membership change.
        let reordered = state(&with(&["Agent", "Workflow", "read_file"]), 0);
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&full),
            &reordered,
            None,
            1,
            TEST_MODEL,
        )
        .expect("break");
        assert!(event.tools_changed);
        assert!(event.tools_added.is_empty() && event.tools_removed.is_empty());
        assert!(
            event.reason.contains("same names; order or definition changed"),
            "{}",
            event.reason
        );
    }

    /// A mid-prefix rewrite must say WHAT changed at the divergence, not only
    /// where. Five identical "diverged at message 275/1142" rows could not tell
    /// an in-memory context trim from a tool-history rewrite; a role, the block
    /// kinds and a byte count do.
    #[test]
    fn a_mid_prefix_rewrite_describes_the_message_that_changed() {
        let long_body = "x".repeat(4_000);
        let previous_request = sample_request_with_messages(&[&long_body, "keep going"]);
        let current_request = sample_request_with_messages(&["[context trimmed]", "keep going"]);
        let state = |request: &MessageRequest, cache_read: u32| {
            TrackedPromptState::from_usage(
                request,
                &Usage {
                    input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: cache_read,
                    output_tokens: 0,
                },
            )
        };
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&state(&previous_request, 200_000)),
            &state(&current_request, 0),
            Some(0),
            2,
            TEST_MODEL,
        )
        .expect("break");

        let diverged = event.diverged_message.as_ref().expect("shape recorded");
        assert_eq!(diverged.index, 0);
        assert_eq!(diverged.previous.role, "user");
        assert_eq!(diverged.previous.kinds, "text");
        assert!(
            diverged.previous.bytes > 4_000 && diverged.current.bytes < 100,
            "the shrink is the whole signal: {diverged:?}"
        );
        assert!(
            event.reason.contains("history diverged at message 0/2 (user [text] "),
            "{}",
            event.reason
        );
        assert!(
            event.reason.contains("kB -> ") && event.reason.contains('B'),
            "the reason must carry both sizes: {}",
            event.reason
        );
    }

    /// The compact wire form must round-trip, including a `kinds` field that
    /// itself carries separators (`tool_use:Agent,text`) and the `+N` tail.
    #[test]
    fn a_message_shape_round_trips_through_its_compact_string_form() {
        let shape = super::WireMessageShape {
            role: "user".to_string(),
            kinds: "tool_use:Agent,text,+12".to_string(),
            bytes: 12_431,
        };
        let json = serde_json::to_string(&shape).expect("serialize");
        assert_eq!(json, "\"user|tool_use:Agent,text,+12|12431\"");
        assert_eq!(
            serde_json::from_str::<super::WireMessageShape>(&json).expect("deserialize"),
            shape
        );
        // A malformed entry must fail loudly rather than deserialize to zeros.
        assert!(serde_json::from_str::<super::WireMessageShape>("\"user|text\"").is_err());
        assert!(serde_json::from_str::<super::WireMessageShape>("\"user|text|huge\"").is_err());
    }

    /// The shapes must survive the same upgrade path the hashes did: a state
    /// file written before the field existed still loads, and the break it
    /// produces simply carries no shape rather than failing.
    #[test]
    fn a_state_file_without_shapes_still_loads_and_breaks_without_one() {
        let old_json = serde_json::json!({
            "observed_at_unix_secs": 1_000_000,
            "fingerprint_version": super::current_fingerprint_version(),
            "model_hash": 1,
            "system_hash": 2,
            "tools_hash": 3,
            "messages_hash": 4,
            "cache_read_input_tokens": 200_000,
            "message_hashes": [11, 22],
        })
        .to_string();
        let previous: TrackedPromptState =
            serde_json::from_str(&old_json).expect("pre-shape state must still deserialize");
        assert!(previous.message_shapes.is_empty());

        let current = TrackedPromptState::from_usage(
            &sample_request_with_messages(&["changed", "tail"]),
            &Usage {
                input_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 0,
            },
        );
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&previous),
            &current,
            Some(0),
            2,
            TEST_MODEL,
        )
        .expect("break");
        assert!(event.diverged_message.is_none());
        assert!(
            event.reason.contains("history diverged at message 0/2"),
            "{}",
            event.reason
        );
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
        let event = detect_cache_break(
            &PromptCacheConfig::default(),
            Some(&previous),
            &current,
            Some(0),
            1,
            TEST_MODEL,
        )
        .expect("break should be detected");
        assert!(!event.unexpected);
        assert!(event.reason.contains("message payload changed"));
        assert!(event.reason.contains("history diverged at message 0/1"));
    }

    /// End-to-end break ledger: every break-bearing request appends one row
    /// whose axis flags are structured (no string parsing), non-break requests
    /// append nothing, and the provider-side-miss signature (all axes stable,
    /// `unexpected: true`) is distinguishable from a tools-axis break — the
    /// exact per-request evidence `stats.json`'s single `last_break_reason`
    /// kept losing.
    #[test]
    fn break_ledger_records_each_break_with_axis_flags() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-breaks-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let cache = PromptCache::new("break-ledger-session");
        let warm = |cr: u32| Usage {
            input_tokens: 10,
            cache_creation_input_tokens: 100,
            cache_read_input_tokens: cr,
            output_tokens: 5,
        };

        // req1: baseline (no previous → no break).
        let base = sample_request("stable message");
        let record = cache.record_usage(&base, &warm(6_000));
        assert!(record.cache_break.is_none());

        // req2: tools axis changes, everything else identical → expected
        // break attributed to tools only.
        let mut with_tool = base.clone();
        with_tool.tools = Some(vec![crate::types::ToolDefinition {
            name: "probe".into(),
            description: Some("probe tool".into()),
            input_schema: serde_json::json!({"type": "object"}),
        }]);
        let record = cache.record_usage(&with_tool, &warm(0));
        let event = record.cache_break.expect("tools change must break");
        assert!(event.tools_changed && !event.model_changed && !event.system_changed);

        // req3: identical request repeated, cache reads recover → no break,
        // no ledger row.
        let record = cache.record_usage(&with_tool, &warm(6_000));
        assert!(record.cache_break.is_none());

        // req4: byte-identical payload yet reads collapse inside the TTL —
        // the provider-side-miss signature.
        let record = cache.record_usage(&with_tool, &warm(0));
        let event = record.cache_break.expect("stable-fingerprint drop must break");
        assert!(event.unexpected);
        assert!(
            !event.model_changed
                && !event.system_changed
                && !event.tools_changed
                && !event.messages_changed
        );

        // req5/req6: grow the history, then truncate it with the surviving
        // prefix intact (the compaction/rewind shape). The divergence index is
        // `None` both ways — the truncation flag and label must tell them
        // apart, or a compaction cold-rewrite reads as "append-only".
        let mut grown = with_tool.clone();
        grown.messages = vec![
            InputMessage::user_text("stable message"),
            InputMessage::user_text("second"),
            InputMessage::user_text("third"),
        ];
        let _ = cache.record_usage(&grown, &warm(6_000));
        let mut truncated = with_tool.clone();
        truncated.messages = vec![InputMessage::user_text("stable message")];
        let record = cache.record_usage(&truncated, &warm(0));
        let event = record.cache_break.expect("truncation must break");
        assert!(event.messages_changed && event.messages_truncated);
        assert!(
            event.reason.contains("history truncated (3 -> 1 messages"),
            "truncation must be labeled, got: {}",
            event.reason
        );

        let rows = super::read_break_ledger(&cache.paths().breaks_path);
        assert_eq!(rows.len(), 3, "one row per break, none for quiet requests: {rows:?}");
        assert_eq!(rows[0].seq, 2);
        assert!(rows[0].tools_changed && !rows[0].unexpected);
        assert!(rows[0].reason.contains("tool definitions changed"));
        assert_eq!(rows[0].prev_cache_read, 6_000);
        assert_eq!(rows[0].cache_read, 0);
        assert_eq!(rows[0].cache_creation, 100);
        assert_eq!(rows[1].seq, 4);
        assert!(rows[1].unexpected, "all-axes-stable drop is the provider-side-miss row");
        assert!(
            !rows[1].model_changed
                && !rows[1].system_changed
                && !rows[1].tools_changed
                && !rows[1].messages_changed
        );
        assert!(!rows[1].messages_truncated);
        assert_eq!(rows[2].seq, 6);
        assert!(rows[2].messages_truncated, "truncation row carries the flag: {rows:?}");
        assert_eq!(rows[2].prev_message_count, 3);
        assert_eq!(rows[2].message_count, 1);
        assert_eq!(rows[2].prefix_stable_messages, 1);

        // Reader is lossy: a corrupt line is skipped, not fatal.
        {
            use std::io::Write as _;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&cache.paths().breaks_path)
                .expect("ledger file exists");
            writeln!(file, "not json").expect("append corrupt line");
        }
        assert_eq!(super::read_break_ledger(&cache.paths().breaks_path).len(), 3);

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = std::fs::remove_dir_all(&temp_root);
    }

    /// A row must name the model the break happened on and that model's
    /// provider family. One store holds rows from every provider a machine
    /// touched, and until the fields existed the family could only be INFERRED
    /// from arithmetic artifacts (`cache_creation == 0`, `cache_read` divisible
    /// by 128) — an inference a real cost investigation got wrong, quoting a
    /// median over the mixed population before it found the split. Two
    /// providers inside one scope here, because that is the case that used to
    /// be unreadable.
    #[test]
    fn a_break_row_names_the_wire_model_and_provider_family_that_broke() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-attribution-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);
        let cache = PromptCache::new("break-attribution-session");
        let warm = |cr: u32| Usage {
            input_tokens: 10,
            cache_creation_input_tokens: 100,
            cache_read_input_tokens: cr,
            output_tokens: 5,
        };

        // Anthropic leg: a warm request, then a byte-stable read collapse.
        let claude = sample_request("attribute me");
        assert!(cache.record_usage(&claude, &warm(6_000)).cache_break.is_none());
        let event = cache
            .record_usage(&claude, &warm(0))
            .cache_break
            .expect("a byte-stable read collapse is a break");
        assert_eq!(event.model, TEST_MODEL);
        assert_eq!(event.provider, "anthropic");

        // Same scope, other provider. The model switch itself is not the break
        // under test — it recovers cache reads, so the row lands on the next
        // collapse, exactly as a real provider hand-off would.
        let mut gemini = claude.clone();
        gemini.model = "gemini-2.5-pro".to_string();
        assert!(cache.record_usage(&gemini, &warm(6_000)).cache_break.is_none());
        let event = cache
            .record_usage(&gemini, &warm(0))
            .cache_break
            .expect("second collapse is a break");
        assert_eq!(event.model, "gemini-2.5-pro");
        assert_eq!(event.provider, "google");

        let rows = super::read_break_ledger(&cache.paths().breaks_path);
        assert_eq!(rows.len(), 2, "one row per break: {rows:?}");
        assert_eq!(
            (rows[0].model.as_str(), rows[0].provider.as_str()),
            (TEST_MODEL, "anthropic")
        );
        assert_eq!(
            (rows[1].model.as_str(), rows[1].provider.as_str()),
            ("gemini-2.5-pro", "google"),
            "a mixed-provider scope must read without inference: {rows:?}"
        );

        std::env::remove_var("ZO_CONFIG_HOME");
        let _ = std::fs::remove_dir_all(&temp_root);
    }

    /// Attribution is additive in both directions: a row written before the
    /// fields existed still loads, and a row that cannot fill them does not pay
    /// bytes for them (the ledger gains a line per break for the life of every
    /// session, so an always-present empty key is a permanent tax).
    #[test]
    fn a_row_written_before_attribution_still_loads_and_empty_fields_stay_off_the_wire() {
        let old_line = serde_json::json!({
            "seq": 7,
            "ts_unix_secs": 1_000,
            "unexpected": false,
            "reason": "tool definitions changed",
            "model_changed": false,
            "system_changed": false,
            "tools_changed": true,
            "messages_changed": false,
            "first_divergence_index": serde_json::Value::Null,
            "prev_message_count": 40,
            "message_count": 41,
            "prev_cache_read": 700_000,
            "cache_read": 24_000,
            "cache_creation": 690_000,
            "token_drop": 676_000,
            "elapsed_secs": 12,
        })
        .to_string();
        let dir = tempfile::TempDir::new().expect("mkdir ledger dir");
        let path = dir.path().join("breaks.jsonl");
        std::fs::write(&path, format!("{old_line}\n")).expect("write pre-attribution ledger");

        let rows = super::read_break_ledger(&path);
        assert_eq!(rows.len(), 1, "an old row must still parse: {rows:?}");
        assert!(rows[0].tools_changed, "the rest of the row must survive too");
        assert!(
            rows[0].model.is_empty() && rows[0].provider.is_empty(),
            "an old row carries no attribution rather than failing to load"
        );

        let line = serde_json::to_string(&rows[0]).expect("serialize unfilled row");
        assert!(!line.contains("\"model\":"), "unfilled model must be omitted: {line}");
        assert!(!line.contains("\"provider\":"), "unfilled provider must be omitted: {line}");

        // The two assertions above are only meaningful if a FILLED row does
        // write the keys — otherwise they would pass on a field that never
        // serializes at all.
        let mut filled = rows[0].clone();
        filled.model = TEST_MODEL.to_string();
        filled.provider = "anthropic".to_string();
        let line = serde_json::to_string(&filled).expect("serialize filled row");
        assert!(line.contains("\"model\":\"claude-3-7-sonnet-latest\""), "{line}");
        assert!(line.contains("\"provider\":\"anthropic\""), "{line}");
    }

    /// The family string is the provider registry's own bucket key, not a
    /// second taxonomy invented next to it — if the two disagree the ledger
    /// misattributes cost, which is the whole failure the field exists to stop.
    #[test]
    fn the_recorded_provider_family_is_the_registrys_own_key() {
        let _guard = test_env_lock();
        assert_eq!(
            super::provider_family_for_model(TEST_MODEL),
            crate::ProviderKind::Anthropic.rate_limit_key()
        );
        assert_eq!(super::provider_family_for_model("gpt-5.5"), "openai");
        assert_eq!(super::provider_family_for_model("gemini-2.5-pro"), "google");

        // An id no family claims records nothing rather than guessing, so the
        // row omits the key. (`OLLAMA_BASE_URL` in the environment makes the
        // REGISTRY claim unknown ids for Ollama — that is its documented
        // behavior, not this helper's, so allow it instead of pretending.)
        let unknown = super::provider_family_for_model("not-a-real-model-family");
        assert!(
            unknown.is_empty() || unknown == "ollama",
            "an unknown id must not be attributed to a first-party family: {unknown}"
        );
    }

    /// A fresh private store root that cleans itself up on drop — including
    /// when an assertion unwinds (the previous hand-rolled temp path leaked
    /// its tree on every failing run).
    fn sweep_test_root() -> tempfile::TempDir {
        tempfile::TempDir::new().expect("mkdir root")
    }

    fn mk_session(root: &Path, name: &str) -> std::path::PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("mkdir session");
        std::fs::write(dir.join("stats.json"), "{}").expect("stats");
        dir
    }

    /// Age branch: idle directories past retention go; the calling session
    /// (even matched case-insensitively — APFS default volumes fold case, so
    /// `--session-id MyRun` and `myrun` share one directory), plain files,
    /// and symlinked directories all survive, and a planted symlink's target
    /// is never followed.
    #[test]
    fn stale_session_sweep_ages_out_idle_dirs_but_never_current_or_symlinks() {
        let store = sweep_test_root();
        let root = store.path();
        let current = mk_session(root, "Current-Session");
        let idle_a = mk_session(root, "idle-a");
        let idle_b = mk_session(root, "idle-b");
        // A planted symlink to a victim directory OUTSIDE the store root.
        let victim_home = sweep_test_root();
        let victim = victim_home.path();
        std::fs::write(victim.join("keep.txt"), "keep").expect("victim file");
        #[cfg(unix)]
        std::os::unix::fs::symlink(victim, root.join("planted-link")).expect("symlink");

        // A `now` far in the future ages every real mtime past retention —
        // idle sessions go; the current session is exempt regardless of age,
        // including when the caller spells its name in a different case.
        let future = SystemTime::now()
            + Duration::from_secs((super::PROMPT_CACHE_RETENTION_DAYS + 10) * 86_400);
        let outcome = super::sweep_stale_session_dirs(
            root,
            &root.join("current-session"),
            future,
            super::PROMPT_CACHE_MAX_SESSION_DIRS,
        )
        .expect("sweep");
        assert_eq!(outcome.removed, 2, "both idle sessions age out");
        assert_eq!(
            outcome.kept, 1,
            "the spared current session counts as kept (symlinks are not live dirs)"
        );
        assert!(
            current.exists(),
            "the calling session must never be swept (case-insensitive name match)"
        );
        assert!(!idle_a.exists() && !idle_b.exists());
        assert!(
            victim.join("keep.txt").exists(),
            "a planted symlink must not delete its target"
        );
    }

    /// Cap branch — the code path that actually fired in production (the age
    /// branch rarely finds anything on an active machine): oldest-first trim
    /// down to the cap, but NEVER a directory active within the cap-trim
    /// guard window, and never the current session. Exercised with an
    /// injected cap because the production constant cannot be reached with
    /// test-sized fixtures.
    #[test]
    fn stale_session_sweep_cap_trims_oldest_but_spares_recently_active() {
        let store = sweep_test_root();
        let root = store.path();
        let current = mk_session(root, "current");
        for index in 0..4 {
            mk_session(root, &format!("s{index}"));
        }
        // All five non-current dirs share "now" mtimes: within retention AND
        // within the cap-trim guard. Over-cap trimming must remove NOTHING.
        let now = SystemTime::now();
        let outcome =
            super::sweep_stale_session_dirs(root, &current, now, 2).expect("guarded sweep");
        assert_eq!(
            outcome.removed, 0,
            "recently-active dirs are never cap-trimmed even over the cap"
        );
        assert_eq!(outcome.kept, 5, "4 survivors + the calling session");

        // Push `now` past the guard window (but inside retention): the cap
        // may now trim oldest-first down to the injected cap of 2.
        let past_guard = now
            + Duration::from_secs(
                (super::PROMPT_CACHE_CAP_TRIM_MIN_IDLE_DAYS + 1) * 86_400,
            );
        let outcome =
            super::sweep_stale_session_dirs(root, &current, past_guard, 2).expect("cap sweep");
        assert_eq!(outcome.removed, 2, "trim down to the cap, oldest first");
        assert_eq!(outcome.kept, 3, "2 capped survivors + the calling session");
        assert!(current.exists(), "current survives the cap branch too");
    }

    /// An interrupted delete leaves a `.sweeping-` husk whose mtime looks
    /// fresh (child deletions bump it); the next sweep must remove husks
    /// unconditionally instead of letting them squat under the fresh mtime.
    #[test]
    fn stale_session_sweep_finishes_interrupted_husks() {
        let store = sweep_test_root();
        let root = store.path();
        let current = mk_session(root, "current");
        let husk = root.join(format!("{}old-session", super::SWEEPING_PREFIX));
        std::fs::create_dir_all(&husk).expect("husk");
        std::fs::write(husk.join("leftover.json"), "{}").expect("leftover");

        let outcome = super::sweep_stale_session_dirs(
            root,
            &current,
            SystemTime::now(),
            super::PROMPT_CACHE_MAX_SESSION_DIRS,
        )
        .expect("sweep");
        assert_eq!(outcome.removed, 1, "husks are finished regardless of mtime");
        assert!(!husk.exists());
        assert!(current.exists());
    }

    /// The daily marker gates the sweep: a missing marker arms one (touched
    /// BEFORE spawning, then rewritten with the outcome attestation), and a
    /// fresh marker short-circuits without rewriting. Gate behavior is
    /// asserted on marker BYTES (the attestation carries a nanosecond nonce),
    /// not mtime — a 1-second-resolution filesystem cannot distinguish a
    /// rewrite by mtime.
    #[test]
    fn sweep_marker_gates_to_once_per_day() {
        let store = sweep_test_root();
        let root = store.path().to_path_buf();
        let paths = PromptCachePaths {
            session_dir: root.join("current"),
            completion_dir: root.join("current").join("completions"),
            session_state_path: root.join("current").join("session-state.json"),
            stats_path: root.join("current").join("stats.json"),
            breaks_path: root.join("current").join("breaks.jsonl"),
            root: root.clone(),
        };
        let marker = root.join(super::SWEEP_MARKER_FILE);
        assert!(!marker.exists());
        super::maybe_spawn_stale_session_sweep(&paths);
        assert!(marker.exists(), "an armed sweep must leave the daily marker");
        // Wait for the detached sweep thread to write its attestation.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let attested = loop {
            if let Some(outcome) = super::read_sweep_marker(&root) {
                break outcome;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "sweep thread must attest its outcome in the marker"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        assert_eq!(attested.removed, 0, "an empty store sweeps nothing");
        let first_bytes = std::fs::read(&marker).expect("marker bytes");

        // Fresh marker → the gate short-circuits: neither a new touch nor a
        // new attestation may change the marker bytes.
        super::maybe_spawn_stale_session_sweep(&paths);
        std::thread::sleep(std::time::Duration::from_millis(50));
        let second_bytes = std::fs::read(&marker).expect("marker bytes");
        assert_eq!(
            first_bytes, second_bytes,
            "a fresh marker must gate without rewriting"
        );
    }

    /// Re-arming a STALE marker must bump its mtime without touching its
    /// bytes: the body is the last completed sweep's attestation, and the
    /// old truncate-on-arm left a permanent zero-byte marker whenever the
    /// process exited before its detached sweep thread finished.
    #[test]
    fn arm_sweep_marker_rearms_stale_marker_without_destroying_attestation() {
        let store = sweep_test_root();
        let root = store.path();
        let marker = root.join(super::SWEEP_MARKER_FILE);

        // Missing marker: created empty (nothing to preserve) and armed.
        assert_eq!(super::arm_sweep_marker(&marker), super::SweepMarkerArming::Armed);
        assert_eq!(std::fs::read(&marker).expect("marker").len(), 0);

        // Fresh marker (just created): the gate declines.
        assert_eq!(super::arm_sweep_marker(&marker), super::SweepMarkerArming::Declined);

        // Stale marker carrying a previous attestation: re-armed, bytes kept.
        let attestation = serde_json::to_vec(&super::SweepMarker {
            swept_at_unix_nanos: 42,
            removed: 7,
            kept: 3,
        })
        .expect("attestation json");
        std::fs::write(&marker, &attestation).expect("seed attestation");
        let stale = SystemTime::now() - Duration::from_secs(25 * 60 * 60);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&marker)
            .and_then(|file| file.set_modified(stale))
            .expect("age the marker");
        assert_eq!(super::arm_sweep_marker(&marker), super::SweepMarkerArming::Armed);
        assert_eq!(
            std::fs::read(&marker).expect("marker"),
            attestation,
            "re-arming must not destroy the previous attestation"
        );
        let rearmed_mtime =
            std::fs::metadata(&marker).and_then(|meta| meta.modified()).expect("mtime");
        assert!(
            rearmed_mtime > stale + Duration::from_secs(60),
            "re-arming must bump the gate mtime"
        );
        // And the bumped mtime now gates again.
        assert_eq!(super::arm_sweep_marker(&marker), super::SweepMarkerArming::Declined);

        // A planted symlink where the marker belongs is refused outright.
        #[cfg(unix)]
        {
            let linked = root.join("elsewhere");
            std::fs::write(&linked, b"target").expect("target");
            let link = root.join("link-marker");
            std::os::unix::fs::symlink(&linked, &link).expect("symlink");
            assert_eq!(super::arm_sweep_marker(&link), super::SweepMarkerArming::Declined);
            assert_eq!(std::fs::read(&linked).expect("target"), b"target");
        }
    }

    /// Pins the pairing between [`detect_cache_break`]'s reason wording and
    /// [`CacheBreakLedgerRow::no_axis_cause`], with rows built from REAL
    /// detector output rather than hand-written strings — a rewording that
    /// strands the classifier turns this red instead of silently downgrading
    /// every no-axis break to `Unknown` in the doctor.
    #[test]
    fn no_axis_cause_classifies_real_detector_reasons() {
        let config = PromptCacheConfig::new("no-axis-cause");
        let base = TrackedPromptState {
            observed_at_unix_secs: 1_000_000,
            fingerprint_version: super::current_fingerprint_version(),
            model_hash: 1,
            system_hash: 2,
            tools_hash: 3,
            messages_hash: 4,
            cache_read_input_tokens: 50_000,
            message_hashes: vec![11, 22],
            message_shapes: Vec::new(),
            tool_names: Vec::new(),
        };
        let row_for = |event: &super::CacheBreakEvent| CacheBreakLedgerRow {
            seq: 1,
            ts_unix_secs: 0,
            unexpected: event.unexpected,
            reason: event.reason.clone(),
            model_changed: event.model_changed,
            system_changed: event.system_changed,
            tools_changed: event.tools_changed,
            messages_changed: event.messages_changed,
            messages_truncated: event.messages_truncated,
            first_divergence_index: None,
            prefix_stable_messages: 0,
            prev_message_count: 0,
            message_count: 0,
            prev_cache_read: event.previous_cache_read_input_tokens,
            cache_read: event.current_cache_read_input_tokens,
            cache_creation: 0,
            token_drop: event.token_drop,
            elapsed_secs: event.elapsed_secs,
            diverged_message: event.diverged_message.clone(),
            tools_added: event.tools_added.clone(),
            tools_removed: event.tools_removed.clone(),
            model: event.model.clone(),
            provider: event.provider.clone(),
            trimmed_tokens_estimate: None,
        };

        // Provider-side: byte-stable fingerprint, inside the TTL, reads drop.
        let mut cold = base.clone();
        cold.observed_at_unix_secs += 10;
        cold.cache_read_input_tokens = 0;
        let event = super::detect_cache_break(&config, Some(&base), &cold, None, 2, TEST_MODEL)
            .expect("provider-side break");
        assert!(event.unexpected);
        assert_eq!(
            row_for(&event).no_axis_cause(),
            Some(super::NoAxisBreakCause::ProviderSide)
        );

        // TTL expiry: same fingerprint, but the gap exceeds the prompt TTL.
        let mut expired = cold.clone();
        expired.observed_at_unix_secs = base.observed_at_unix_secs
            + config.prompt_ttl.as_secs()
            + 60;
        let event = super::detect_cache_break(&config, Some(&base), &expired, None, 2, TEST_MODEL)
            .expect("ttl break");
        assert!(!event.unexpected);
        assert_eq!(
            row_for(&event).no_axis_cause(),
            Some(super::NoAxisBreakCause::TtlExpiry)
        );

        // Fingerprint bump: our own schema version changed.
        let mut bumped = cold.clone();
        bumped.fingerprint_version = base.fingerprint_version + 1;
        let event = super::detect_cache_break(&config, Some(&base), &bumped, None, 2, TEST_MODEL)
            .expect("fingerprint break");
        assert!(!event.unexpected);
        assert_eq!(
            row_for(&event).no_axis_cause(),
            Some(super::NoAxisBreakCause::FingerprintBump)
        );

        // An axis break carries its own explanation — no cause classification.
        let mut retooled = cold.clone();
        retooled.tools_hash = 99;
        let event = super::detect_cache_break(&config, Some(&base), &retooled, None, 2, TEST_MODEL)
            .expect("tools break");
        assert!(event.tools_changed);
        assert_eq!(row_for(&event).no_axis_cause(), None);
    }

    /// A session that stores no completion must leave no completion directory.
    ///
    /// The store is written only from the non-streaming `send_message`, while
    /// the live turn loop is streaming — so creating the directory per session
    /// produced 548 permanently-empty directories on this machine and made a
    /// feature that never ran look provisioned. Emptiness is the signal; do not
    /// manufacture it.
    #[test]
    fn a_session_that_stores_no_completion_leaves_no_completion_dir() {
        let _guard = test_env_lock();
        let temp_root = std::env::temp_dir().join(format!(
            "prompt-cache-emptydir-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        std::env::set_var("ZO_CONFIG_HOME", &temp_root);

        let cache = PromptCache::new("no-completion-session");
        let request = sample_request("stream me");
        // The streaming path's recording call: usage only, no response body.
        let _ = cache.record_usage(&request, &sample_response(1, 1, "x").usage);
        let paths = PromptCachePaths::for_session("no-completion-session");
        assert!(
            !paths.completion_dir.exists(),
            "no completion was stored, so nothing should have created its directory"
        );

        // …and the writer still creates it when there IS something to store.
        let _ = cache.record_response(&request, &sample_response(2, 2, "y"));
        assert!(paths.completion_dir.exists());
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

    /// Model id every fixture in this module sends. Shared with the direct
    /// [`detect_cache_break`] calls so the attribution fields under test always
    /// describe the same request the fixtures build.
    const TEST_MODEL: &str = "claude-3-7-sonnet-latest";

    fn sample_request_with_messages(texts: &[&str]) -> MessageRequest {
        MessageRequest {
            model: TEST_MODEL.to_string(),
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
            model: TEST_MODEL.to_string(),
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
