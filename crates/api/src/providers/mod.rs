use std::sync::{OnceLock, RwLock};
use std::time::Duration;

use serde::Deserialize;

use crate::types::{ContentBlockDelta, StreamEvent};

pub mod anthropic;
pub(crate) mod aws_sigv4;
pub mod chatgpt_backend;
pub(crate) mod cloud_gateway;
pub mod gemini_code_assist;
pub(crate) mod google_auth;
pub mod openai_compat;
pub mod openai_oauth;
mod retry_backoff;

pub const EXPERIMENTAL_PROVIDERS_ENV: &str = "ZO_EXPERIMENTAL_PROVIDERS";
pub const NON_CLAUDE_ADAPTERS_ENV: &str = "ZO_EXPERIMENTAL_PROVIDER_ADAPTERS";

/// Env var containing a JSON model-context capability catalog. This is an
/// operational escape hatch for providers whose `/models` endpoint does not
/// expose context-window metadata; it lets deployments update model limits
/// without recompiling. Shape: `{ "models": [{ "ids": ["model"],
/// "context_window": 1000000 }] }`.
pub const MODEL_CONTEXT_WINDOWS_ENV: &str = "ZO_MODEL_CONTEXT_WINDOWS";

const BUILTIN_MODEL_CONTEXT_WINDOWS_JSON: &str = include_str!("model_context_windows.json");

#[derive(Debug, Default, Deserialize)]
struct ModelContextCatalog {
    #[serde(default)]
    models: Vec<ModelContextEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelContextEntry {
    #[serde(default)]
    ids: Vec<String>,
    context_window: u64,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    /// Provider-declared positioning class (`"frontier"|"balanced"|"fast"`),
    /// when the provider (or an operator, via [`MODEL_CLASSES_ENV`]) has
    /// actually stated one. Absent for every entry whose provider has not
    /// published a positioning signal (Gemini/DeepSeek/Grok/etc. today) —
    /// see [`declared_model_class`]'s doc for the no-invented-quality-tables
    /// rule this enforces. Provenance for the declaration itself lives in
    /// this entry's existing `source` string rather than a parallel field:
    /// each entry already carries exactly one context-window/class fact set,
    /// so one provenance string covers both.
    #[serde(default)]
    class: Option<String>,
}

impl ModelContextCatalog {
    fn find_entry(&self, raw_model: &str, canonical_model: &str) -> Option<&ModelContextEntry> {
        let raw = raw_model.trim().to_ascii_lowercase();
        let canonical = canonical_model.trim().to_ascii_lowercase();

        // Prefer exact ids, then the longest segment-delimited family prefix.
        // Specificity matters because the catalog contains both `gpt` and
        // narrower families such as `gpt-5.6-sol`; a generic entry must not
        // shadow a dated/qualified member of the narrower family.
        if let Some(exact) = self.models.iter().find(|entry| {
            entry.ids.iter().any(|id| {
                let id = id.trim().to_ascii_lowercase();
                id == raw || id == canonical
            })
        }) {
            return Some(exact);
        }

        let mut best = None;
        for entry in &self.models {
            for id in &entry.ids {
                let id = id.trim().to_ascii_lowercase();
                // Bare aliases such as `gpt` are exact-only: treating them as
                // families would fabricate 258k/128k capabilities for every
                // unknown future GPT id. Versioned/model-specific ids retain
                // segment-delimited suffix matching.
                if id != "gpt"
                    && (model_id_matches_prefix_segment(&raw, &id)
                        || model_id_matches_prefix_segment(&canonical, &id))
                {
                    let specificity = id.len();
                    if best.is_none_or(|(best_specificity, _)| specificity > best_specificity) {
                        best = Some((specificity, entry));
                    }
                }
            }
        }
        best.map(|(_, entry)| entry)
    }

    fn context_window_for(&self, raw_model: &str, canonical_model: &str) -> Option<u64> {
        self.find_entry(raw_model, canonical_model)
            .map(|entry| entry.context_window)
            .filter(|&cw| cw > 0)
    }

    /// The docs-verified per-model synchronous output cap, when the catalog
    /// declares one. Previously dropped on the floor — `ModelContextEntry` only
    /// read `context_window`, so a non-Anthropic model's real cap (GPT-5.5 128k,
    /// `DeepSeek` V4 384k) never reached the wire and every such model was pinned
    /// to the conservative 64k default.
    fn max_output_tokens_for(&self, raw_model: &str, canonical_model: &str) -> Option<u64> {
        self.find_entry(raw_model, canonical_model)
            .and_then(|entry| entry.max_output_tokens)
    }

    /// The catalog's declared [`ModelClass`] for a model, via the exact same
    /// matcher [`Self::context_window_for`] uses (exact id first, then
    /// longest segment-delimited family prefix) — so a dated/`@`/`[`-suffixed
    /// id (`gpt-5.6-sol-2026-07-09`) resolves its declared class through the
    /// identical machinery as its context window.
    fn class_for(&self, raw_model: &str, canonical_model: &str) -> Option<ModelClass> {
        self.find_entry(raw_model, canonical_model)
            .and_then(|entry| entry.class.as_deref())
            .and_then(parse_model_class)
    }
}

/// Provider-declared positioning class for a model — the ONLY static
/// quality-like signal this codebase permits (design principle: no invented
/// quality tables; only provider-declared capability facts or explicitly
/// labeled cold-start priors). A model with no declared class returns `None`;
/// callers fall back to their own capability-derived heuristics rather than
/// guessing a class.
///
/// This is a distinct axis from [`ModelDescriptor`]-style marketing-name
/// classification (`opus`/`sonnet`/`fast`/... — see `model_inventory::
/// class_for_model` in the `runtime` crate): that string is a free-form
/// family/flavor label used for `RoleSelector` matching, while `ModelClass`
/// is the provider's own stated POSITIONING within its lineup (frontier vs.
/// balanced vs. fast), used to seed tier precedence. The two are not merged —
/// see `class_for_model`'s doc comment for why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelClass {
    /// Provider states this is (one of) their top/latest agentic model(s) —
    /// e.g. OpenAI's Codex model cache "Latest frontier agentic coding
    /// model", or Anthropic's public Mythos-class-above-Opus positioning.
    Frontier,
    /// Provider states this is their everyday/mid-tier model.
    Balanced,
    /// Provider states this is their fast/affordable model.
    Fast,
}

fn parse_model_class(value: &str) -> Option<ModelClass> {
    match value.trim().to_ascii_lowercase().as_str() {
        "frontier" => Some(ModelClass::Frontier),
        "balanced" => Some(ModelClass::Balanced),
        "fast" => Some(ModelClass::Fast),
        _ => None,
    }
}

/// Env var containing a JSON object of `{"model-or-family-prefix":
/// "frontier|balanced|fast"}` declared-class overrides, consulted before the
/// catalog in [`declared_model_class`] — the same zero-rebuild escape hatch
/// pattern as [`MODEL_EFFORT_CEILINGS_ENV`] (longest matching prefix wins,
/// segment-boundary aware via [`crate::types::model_id_matches_family`]).
pub const MODEL_CLASSES_ENV: &str = "ZO_MODEL_CLASSES";

/// Read fresh (not cached) so tests can set/unset the env var per-case —
/// mirrors [`env_effort_ceiling_override`].
fn env_model_class_override(canonical_lower: &str) -> Option<ModelClass> {
    let raw = std::env::var(MODEL_CLASSES_ENV).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let overrides: std::collections::HashMap<String, String> = serde_json::from_str(&raw).ok()?;
    let mut best: Option<(usize, ModelClass)> = None;
    for (prefix, value) in &overrides {
        let prefix_lower = prefix.trim().to_ascii_lowercase();
        if prefix_lower.is_empty() {
            continue;
        }
        if !crate::types::model_id_matches_family(canonical_lower, &prefix_lower) {
            continue;
        }
        let Some(class) = parse_model_class(value) else {
            continue;
        };
        if best.is_none_or(|(best_len, _)| prefix_lower.len() > best_len) {
            best = Some((prefix_lower.len(), class));
        }
    }
    best.map(|(_, class)| class)
}

/// Catalog-declared class lookup, mirroring [`catalog_max_output_tokens`]'s
/// env-catalog-then-builtin resolution order.
fn catalog_model_class(raw_model: &str, canonical_model: &str) -> Option<ModelClass> {
    if let Ok(raw) = std::env::var(MODEL_CONTEXT_WINDOWS_ENV) {
        if !raw.trim().is_empty() {
            if let Ok(catalog) = serde_json::from_str::<ModelContextCatalog>(&raw) {
                if let Some(class) = catalog.class_for(raw_model, canonical_model) {
                    return Some(class);
                }
            }
        }
    }
    builtin_model_context_catalog().class_for(raw_model, canonical_model)
}

/// Single source of truth for a model's provider-declared [`ModelClass`],
/// when one exists. Resolution order: [`MODEL_CLASSES_ENV`] override
/// (longest matching prefix wins) → [`MODEL_CONTEXT_WINDOWS_ENV`] catalog
/// override (if it happens to carry a `class` field) → the built-in
/// [`model_context_windows.json`] catalog. Aliases resolve through
/// [`resolve_model_alias`] first, then dated/`@`/`[`-suffixed ids resolve
/// through the same family-prefix matcher [`context_window_for_model`] uses.
///
/// Returns `None` for every model the provider (and no operator override) has
/// not declared a position for — e.g. Gemini/DeepSeek/Grok today. Callers
/// MUST treat `None` as "undeclared", not as any particular class; they fall
/// back to their own capability-derived heuristics (design principle:
/// no-hardcoding end-state — only provider-declared facts or labeled
/// cold-start priors are allowed as static data).
#[must_use]
pub fn declared_model_class(model: &str) -> Option<ModelClass> {
    let canonical = resolve_model_alias(model);
    let canonical_lower = canonical.to_ascii_lowercase();
    if let Some(class) = env_model_class_override(&canonical_lower) {
        return Some(class);
    }
    catalog_model_class(model, &canonical)
}

fn builtin_model_context_catalog() -> &'static ModelContextCatalog {
    static CATALOG: OnceLock<ModelContextCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_json::from_str(BUILTIN_MODEL_CONTEXT_WINDOWS_JSON).unwrap_or_else(|error| {
            eprintln!("[zo] built-in model context catalog is invalid: {error}");
            ModelContextCatalog::default()
        })
    })
}

fn env_model_context_window(raw_model: &str, canonical_model: &str) -> Option<u64> {
    let raw = std::env::var(MODEL_CONTEXT_WINDOWS_ENV).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let catalog: ModelContextCatalog = serde_json::from_str(&raw).ok()?;
    catalog.context_window_for(raw_model, canonical_model)
}

fn catalog_model_context_window(raw_model: &str, canonical_model: &str) -> Option<u64> {
    env_model_context_window(raw_model, canonical_model)
        .or_else(|| builtin_model_context_catalog().context_window_for(raw_model, canonical_model))
}

/// The docs-verified max synchronous output tokens for a model, from the env
/// override catalog first then the built-in — mirrors
/// [`catalog_model_context_window`] for the `max_output_tokens` field.
fn catalog_max_output_tokens(raw_model: &str, canonical_model: &str) -> Option<u64> {
    if let Ok(raw) = std::env::var(MODEL_CONTEXT_WINDOWS_ENV) {
        if !raw.trim().is_empty() {
            if let Ok(catalog) = serde_json::from_str::<ModelContextCatalog>(&raw) {
                if let Some(max_out) = catalog.max_output_tokens_for(raw_model, canonical_model) {
                    return Some(max_out);
                }
            }
        }
    }
    builtin_model_context_catalog().max_output_tokens_for(raw_model, canonical_model)
}

/// Process-wide shared `reqwest::Client` for every provider call.
///
/// 매 `ProviderClient` 가 `reqwest::Client::new()` 를 호출하면 connection
/// pool 이 인스턴스마다 분리되어 H2 multiplexing / TLS session resumption
/// 효과가 모두 소실된다. 단일 인스턴스로 일원화하면:
/// * 첫 호출 이후 TLS / TCP 재사용 → first-token 1-RTT 단축
/// * H2 stream multiplexing 으로 multi-agent burst 시 connection 폭증 방지
/// * `pool_idle_timeout` 5분으로 후속 turn 도 warm 유지
/// * H2 keep-alive 30s 로 NAT/방화벽이 idle 연결 끊는 것 차단
///
/// `reqwest::Client` 내부는 `Arc<Inner>` 이므로 `clone()` 비용 0. ALPN 으로
/// H2 자동 협상되므로 `http2_prior_knowledge` 는 사용하지 않는다 — 서버가
/// HTTP/1.1 만 지원해도 graceful fallback 보장.
#[must_use]
pub(crate) fn shared_http_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .tcp_nodelay(true)
                // Bound the connect phase only — a dead/blackholed host must not
                // wedge the SSE client forever. No blanket `.timeout()`: Anthropic
                // documents streaming as the timeout-avoidance path for large
                // `max_tokens` requests, and an active Opus/Fable stream can run
                // for minutes.
                .connect_timeout(Duration::from_secs(15))
                .pool_idle_timeout(Some(Duration::from_secs(300)))
                .http2_keep_alive_interval(Some(Duration::from_secs(30)))
                .http2_keep_alive_while_idle(true)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new())
        })
        .clone()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    Xai,
    OpenAi,
    Google,
    Ollama,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCacheStrategy {
    /// Anthropic prompt caching is expressed inside message/system content
    /// blocks with `cache_control` breakpoints plus local prompt-cache
    /// bookkeeping.
    AnthropicCacheControl,
    /// Official OpenAI request caching uses a stable `prompt_cache_key`.
    /// Extended retention is model-gated by [`openai_prompt_cache_retention`].
    OpenAiPromptCacheKey,
    /// OpenAI-compatible providers often reject unknown request fields, so
    /// zo sends no provider-specific cache controls unless a first-class
    /// strategy is known.
    NoRequestControls,
}

impl PromptCacheStrategy {
    /// Whether this strategy sends OpenAI's stable prompt-cache routing key.
    #[must_use]
    pub const fn sends_openai_prompt_cache_key(self) -> bool {
        matches!(self, Self::OpenAiPromptCacheKey)
    }

    /// Optional extended retention for strategies that support it.
    #[must_use]
    pub fn prompt_cache_retention(self, model: &str) -> Option<&'static str> {
        match self {
            Self::OpenAiPromptCacheKey => openai_prompt_cache_retention(model),
            Self::AnthropicCacheControl | Self::NoRequestControls => None,
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.profile().display_name)
    }
}

/// The complete, declarative description of one provider — its display name,
/// rate-limit namespace, connection wiring, and capability flags — in a single
/// place. Every [`ProviderKind`] accessor reads from this profile, so adding a
/// provider is exactly one [`ProviderKind`] variant plus one row in
/// [`ProviderKind::profile`], instead of an edit to six scattered `match` arms
/// (the old `connection` / `Display` / `supports_cache_tokens` /
/// `prompt_cache_strategy` / `supports_thinking` / `rate_limit_key` split, where
/// a forgotten arm silently gave a new provider the wrong capability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProviderProfile {
    /// Human-facing provider name (used by `Display`).
    display_name: &'static str,
    /// Stable namespace key for rate-limit bucketing / telemetry.
    rate_limit_key: &'static str,
    /// Environment variable holding the provider credential.
    auth_env: &'static str,
    /// Environment variable overriding the provider base URL.
    base_url_env: &'static str,
    /// Default API endpoint when `base_url_env` is unset.
    default_base_url: &'static str,
    /// Whether the provider reports cache token counts in its usage payload.
    supports_cache_tokens: bool,
    /// Whether the provider streams extended-thinking / reasoning content.
    supports_thinking: bool,
    /// Provider-specific prompt-cache request strategy.
    prompt_cache_strategy: PromptCacheStrategy,
}

impl ProviderKind {
    /// The single declarative profile row for this provider. All other
    /// accessors derive from it, keeping provider data in one place.
    const fn profile(self) -> ProviderProfile {
        match self {
            Self::Anthropic => ProviderProfile {
                display_name: "Anthropic",
                rate_limit_key: "anthropic",
                auth_env: "ANTHROPIC_API_KEY",
                base_url_env: "ANTHROPIC_BASE_URL",
                default_base_url: anthropic::DEFAULT_BASE_URL,
                supports_cache_tokens: true,
                supports_thinking: true,
                prompt_cache_strategy: PromptCacheStrategy::AnthropicCacheControl,
            },
            Self::Xai => ProviderProfile {
                display_name: "xAI",
                rate_limit_key: "xai",
                auth_env: "XAI_API_KEY",
                base_url_env: "XAI_BASE_URL",
                default_base_url: openai_compat::DEFAULT_XAI_BASE_URL,
                supports_cache_tokens: false,
                supports_thinking: false,
                prompt_cache_strategy: PromptCacheStrategy::NoRequestControls,
            },
            Self::OpenAi => ProviderProfile {
                display_name: "OpenAI",
                rate_limit_key: "openai",
                auth_env: "OPENAI_API_KEY",
                base_url_env: "OPENAI_BASE_URL",
                default_base_url: openai_compat::DEFAULT_OPENAI_BASE_URL,
                supports_cache_tokens: true,
                supports_thinking: false,
                prompt_cache_strategy: PromptCacheStrategy::OpenAiPromptCacheKey,
            },
            Self::Google => ProviderProfile {
                display_name: "Google",
                rate_limit_key: "google",
                auth_env: "GOOGLE_API_KEY",
                base_url_env: "GOOGLE_BASE_URL",
                default_base_url: openai_compat::DEFAULT_GOOGLE_BASE_URL,
                supports_cache_tokens: false,
                supports_thinking: false,
                prompt_cache_strategy: PromptCacheStrategy::NoRequestControls,
            },
            Self::Ollama => ProviderProfile {
                display_name: "Ollama",
                rate_limit_key: "ollama",
                auth_env: "OLLAMA_API_KEY",
                base_url_env: "OLLAMA_BASE_URL",
                default_base_url: openai_compat::DEFAULT_OLLAMA_BASE_URL,
                supports_cache_tokens: false,
                supports_thinking: false,
                prompt_cache_strategy: PromptCacheStrategy::NoRequestControls,
            },
        }
    }

    /// Connection metadata (credential env, base-url env, default endpoint).
    #[must_use]
    pub const fn metadata(self) -> ProviderMetadata {
        let profile = self.profile();
        ProviderMetadata {
            provider: self,
            auth_env: profile.auth_env,
            base_url_env: profile.base_url_env,
            default_base_url: profile.default_base_url,
        }
    }

    /// Whether the provider reports cache token counts in its usage payload.
    /// Anthropic reports explicit cache read/write fields; OpenAI reports
    /// prompt cache reads through `prompt_tokens_details.cached_tokens`.
    #[must_use]
    pub const fn supports_cache_tokens(self) -> bool {
        self.profile().supports_cache_tokens
    }

    /// Provider-specific prompt-cache request strategy.
    #[must_use]
    pub const fn prompt_cache_strategy(self) -> PromptCacheStrategy {
        self.profile().prompt_cache_strategy
    }

    /// Whether the provider streams extended-thinking / reasoning content.
    /// Only Anthropic surfaces thinking blocks through this client today.
    #[must_use]
    pub const fn supports_thinking(self) -> bool {
        self.profile().supports_thinking
    }

    /// Stable namespace key for rate-limit bucketing / telemetry.
    #[must_use]
    pub const fn rate_limit_key(self) -> &'static str {
        self.profile().rate_limit_key
    }
}

/// Extended OpenAI prompt-cache retention policy for model families that
/// support it. `None` means zo should still send `prompt_cache_key` for
/// OpenAI cache routing, but must not request extended retention.
#[must_use]
pub fn openai_prompt_cache_retention(model: &str) -> Option<&'static str> {
    let _family = openai_gpt_model_family(model)?;
    // The current ChatGPT/OpenAI GPT surface accepts `prompt_cache_key`, but
    // `gpt-5.5` rejects `prompt_cache_retention` and no currently exposed GPT
    // family is verified for 24h retention. Keep the policy hook explicit so a
    // future verified family can be added without touching request builders.
    None
}

const OPENAI_GPT_MODEL_FAMILIES: &[&str] = &[
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.3-codex-spark",
    "gpt-5.5",
];

/// Current GPT families exposed by Zo's OpenAI/ChatGPT picker.
#[must_use]
pub fn openai_gpt_model_family(model: &str) -> Option<&'static str> {
    let model = model.trim().to_ascii_lowercase();
    OPENAI_GPT_MODEL_FAMILIES
        .iter()
        .copied()
        .find(|family| crate::types::model_id_matches_family(&model, family))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderMetadata {
    pub provider: ProviderKind,
    pub auth_env: &'static str,
    pub base_url_env: &'static str,
    pub default_base_url: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelCapability {
    canonical_model_id: String,
    provider: Option<ProviderKind>,
    context_window: Option<u64>,
    max_output_tokens: Option<u32>,
    adaptive_thinking: Option<bool>,
}

impl ModelCapability {
    fn for_model(model: &str) -> Self {
        // Explicit `provider/model` keeps capability lookup on the wire model id
        // while forcing OpenAI-compatible custom routing when the named custom
        // provider exists. Bare Claude/GPT ids still use first-party paths.
        let wire = wire_model_id(model);
        let canonical_model_id = wire.clone();
        let canonical_lower = canonical_model_id.to_ascii_lowercase();
        let model_id = canonical_lower
            .split_once('[')
            .map_or(canonical_lower.as_str(), |(id, _)| id);

        let provider = if custom_provider_for_model(model).is_some()
            && split_provider_model_ref(model).is_some()
        {
            Some(ProviderKind::OpenAi)
        } else if let Some(entry) = catalog_entry_for_token(MODEL_REGISTRY, &canonical_model_id)
            .filter(|entry| entry.provider == ProviderKind::OpenAi)
        {
            // OpenAI's built-in ids have historically been prefix-detectable
            // (`gpt*`, `o*`, `codex*`). Future OpenAI families may not be. Treat
            // OpenAI catalog aliases/canonicals as explicit OpenAI selections so
            // a disabled/missing OpenAI provider produces a clean unsupported-
            // provider error instead of falling through to Anthropic.
            Some(entry.provider)
        } else {
            provider_kind_for_canonical(&canonical_model_id, &canonical_lower)
        };

        Self {
            provider,
            context_window: Some(context_window_for_canonical(&wire, &canonical_lower)),
            max_output_tokens: Some(max_output_tokens_for_model_id(&wire, model_id)),
            adaptive_thinking: Some(adaptive_thinking_for_canonical(&canonical_lower)),
            canonical_model_id,
        }
    }
}

fn model_capability_for_model(model: &str) -> ModelCapability {
    ModelCapability::for_model(model)
}

fn provider_kind_for_canonical(
    canonical_model_id: &str,
    canonical_lower: &str,
) -> Option<ProviderKind> {
    // OpenAI is the only built-in provider whose future model-family prefixes
    // are hard to predict from the id. Trust an explicit OpenAI registry row
    // before prefix guessing so a newly-added family does not fall through to
    // Anthropic; leave other providers on their existing gated/prefix paths.
    if let Some(provider) = catalog_provider_for_canonical(MODEL_REGISTRY, canonical_model_id)
        .filter(|provider| *provider == ProviderKind::OpenAi)
    {
        return Some(provider);
    }
    if canonical_lower.starts_with("claude") {
        return Some(ProviderKind::Anthropic);
    }
    if custom_provider_for_model(canonical_model_id).is_some() {
        return Some(ProviderKind::OpenAi);
    }
    if is_openai_builtin_model_prefix(canonical_lower) {
        return Some(ProviderKind::OpenAi);
    }
    if non_claude_adapters_enabled() && canonical_lower.starts_with("grok") {
        return Some(ProviderKind::Xai);
    }
    if canonical_lower.starts_with("gemini") {
        return Some(ProviderKind::Google);
    }
    if canonical_lower.starts_with("ollama") || std::env::var("OLLAMA_BASE_URL").is_ok() {
        return Some(ProviderKind::Ollama);
    }
    None
}

fn is_openai_builtin_model_prefix(lower: &str) -> bool {
    lower.starts_with("gpt")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("codex")
}

fn catalog_provider_for_canonical(
    catalog: &[ProviderCatalogEntry],
    canonical_model_id: &str,
) -> Option<ProviderKind> {
    let canonical = canonical_model_id.trim();
    catalog
        .iter()
        .find(|entry| entry.canonical_model_id.eq_ignore_ascii_case(canonical))
        .map(|entry| entry.provider)
}

fn catalog_entry_for_token<'a>(
    catalog: &'a [ProviderCatalogEntry],
    model: &str,
) -> Option<&'a ProviderCatalogEntry> {
    let lower = model.trim().to_ascii_lowercase();
    catalog.iter().find(|entry| {
        entry.alias.eq_ignore_ascii_case(&lower)
            || entry.canonical_model_id.eq_ignore_ascii_case(&lower)
    })
}

/// A single row of the provider catalog: a user-facing model `alias`, the
/// `canonical_model_id` it resolves to (the value actually sent to the API),
/// and the [`ProviderKind`] that serves it.
///
/// Connection wiring and capabilities are *derived* from the provider rather
/// than duplicated per row, so the catalog stays a flat alias→id table and a
/// new provider is one [`ProviderKind`] variant plus its registry rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCatalogEntry {
    pub alias: &'static str,
    pub canonical_model_id: &'static str,
    pub provider: ProviderKind,
    pub fit_hint: Option<ModelFitHint>,
}

impl ProviderCatalogEntry {
    #[must_use]
    const fn new(
        alias: &'static str,
        canonical_model_id: &'static str,
        provider: ProviderKind,
    ) -> Self {
        Self {
            alias,
            canonical_model_id,
            provider,
            fit_hint: None,
        }
    }

    /// Connection metadata for the entry's provider.
    #[must_use]
    pub const fn metadata(&self) -> ProviderMetadata {
        self.provider.metadata()
    }

    /// See [`ProviderKind::supports_cache_tokens`].
    #[must_use]
    pub const fn supports_cache_tokens(&self) -> bool {
        self.provider.supports_cache_tokens()
    }

    /// See [`ProviderKind::supports_thinking`].
    #[must_use]
    pub const fn supports_thinking(&self) -> bool {
        self.provider.supports_thinking()
    }

    /// See [`ProviderKind::rate_limit_key`].
    #[must_use]
    pub const fn rate_limit_key(&self) -> &'static str {
        self.provider.rate_limit_key()
    }
}

/// Read-only model fit metadata. This is advisory only: zo never selects,
/// downloads, starts, or rejects a model from these values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFitHint {
    pub estimated_vram_gb: u16,
    pub quantization: &'static str,
}

impl ModelFitHint {
    #[must_use]
    pub const fn new(estimated_vram_gb: u16, quantization: &'static str) -> Self {
        Self {
            estimated_vram_gb,
            quantization,
        }
    }

    #[must_use]
    pub fn display_label(self) -> String {
        format!("VRAM ~{}GB {}", self.estimated_vram_gb, self.quantization)
    }
}

/// The full provider catalog — the single source of truth mapping model
/// aliases to canonical ids and providers.
#[must_use]
pub fn provider_catalog() -> &'static [ProviderCatalogEntry] {
    MODEL_REGISTRY
}

const MODEL_REGISTRY: &[ProviderCatalogEntry] = &[
    // Anthropic — latest only. Both the bare (`opus`) and `claude-` prefixed
    // aliases resolve to the same canonical id, keeping a single source of
    // truth shared with the CLI's alias resolution.
    ProviderCatalogEntry::new("fable", "claude-fable-5", ProviderKind::Anthropic),
    ProviderCatalogEntry::new("claude-fable", "claude-fable-5", ProviderKind::Anthropic),
    ProviderCatalogEntry::new("opus", "claude-opus-4-8", ProviderKind::Anthropic),
    // `opus[1m]` is a visible label-only alias for the same `claude-opus-4-8`
    // (which already carries the native 1M Claude window). It resolves to the
    // bare canonical id — the `[1m]` suffix never reaches the wire — so it is a
    // picker-parity affordance with Claude Code, not a distinct model.
    ProviderCatalogEntry::new("opus[1m]", "claude-opus-4-8", ProviderKind::Anthropic),
    ProviderCatalogEntry::new("sonnet", "claude-sonnet-5", ProviderKind::Anthropic),
    ProviderCatalogEntry::new(
        "haiku",
        "claude-haiku-4-5-20251001",
        ProviderKind::Anthropic,
    ),
    ProviderCatalogEntry::new("claude-opus", "claude-opus-4-8", ProviderKind::Anthropic),
    ProviderCatalogEntry::new("claude-opus[1m]", "claude-opus-4-8", ProviderKind::Anthropic),
    ProviderCatalogEntry::new("claude-sonnet", "claude-sonnet-5", ProviderKind::Anthropic),
    ProviderCatalogEntry::new(
        "claude-haiku",
        "claude-haiku-4-5-20251001",
        ProviderKind::Anthropic,
    ),
    // OpenAI
    ProviderCatalogEntry::new("gpt-5.6-sol", "gpt-5.6-sol", ProviderKind::OpenAi),
    ProviderCatalogEntry::new("gpt-5.6-terra", "gpt-5.6-terra", ProviderKind::OpenAi),
    ProviderCatalogEntry::new("gpt-5.6-luna", "gpt-5.6-luna", ProviderKind::OpenAi),
    // Bare 세대 별칭: `gpt-5.6`은 terra로 — 사용자 확정(2026-07-11): 효율
    // 기준 기본값은 terra(gpt-5.5 계열 퇴역). 서빙 티어는 별칭에 하드코딩하지
    // 않는다(사용자 정책 개정) — bare id로 해석되고, fast 여부는 세션 fast
    // 상태(`/fast on|off`)가 결정한다. sol/luna는 명시 단축 별칭,
    // opus/sonnet/haiku 관례.
    ProviderCatalogEntry::new("gpt-5.6", "gpt-5.6-terra", ProviderKind::OpenAi),
    ProviderCatalogEntry::new("sol", "gpt-5.6-sol", ProviderKind::OpenAi),
    ProviderCatalogEntry::new("terra", "gpt-5.6-terra", ProviderKind::OpenAi),
    ProviderCatalogEntry::new("luna", "gpt-5.6-luna", ProviderKind::OpenAi),
    // gpt-5.5 계열은 카탈로그에서 퇴역(사용자 확정 2026-07-11) — terra가
    // 대체한다. 레거시 세션/핀이 들고 있는 5.5 id의 와이어 지원
    // (chatgpt_backend의 gpt-5.5-fast 별칭 파싱·effort 매핑)과 서브에이전트
    // 사다리의 5.5→terra 이관 arm은 유지된다.
    ProviderCatalogEntry::new(
        "gpt-5.3-codex-spark",
        "gpt-5.3-codex-spark",
        ProviderKind::OpenAi,
    ),
    // Google — Antigravity IDE OAuth, two Gemini families only (Flash + Pro).
    // These canonical ids are routing labels: the Gemini backend's `gemini_wire`
    // derives the actual Cloud Code wire id and `thinkingLevel` from the model
    // family (alias contains `flash` vs `pro`) plus the request effort, so any
    // alias here resolves to `gemini-3-flash` or `gemini-3-pro-{low,high}`.
    // Legacy public names stay accepted so existing settings keep resolving.
    ProviderCatalogEntry::new(
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview",
        ProviderKind::Google,
    ),
    ProviderCatalogEntry::new(
        "gemini-3.1-pro-preview-customtools",
        "gemini-3.1-pro-preview-customtools",
        ProviderKind::Google,
    ),
    ProviderCatalogEntry::new(
        "gemini-3-pro-preview",
        "gemini-3-pro-preview",
        ProviderKind::Google,
    ),
    ProviderCatalogEntry::new("gemini-3.5-flash", "gemini-3.5-flash", ProviderKind::Google),
    ProviderCatalogEntry::new("gemini-3-flash", "gemini-3-flash", ProviderKind::Google),
    ProviderCatalogEntry::new(
        "gemini-3-flash-preview",
        "gemini-3-flash-preview",
        ProviderKind::Google,
    ),
    ProviderCatalogEntry::new(
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
        ProviderKind::Google,
    ),
    ProviderCatalogEntry::new("gemini-pro", "gemini-3.1-pro-preview", ProviderKind::Google),
    ProviderCatalogEntry::new("gemini-flash", "gemini-3.5-flash", ProviderKind::Google),
    ProviderCatalogEntry::new(
        "gemini-flash-lite",
        "gemini-3.1-flash-lite",
        ProviderKind::Google,
    ),
    ProviderCatalogEntry::new(
        "gemini-3.5-pro",
        "gemini-3.1-pro-preview",
        ProviderKind::Google,
    ),
    // xAI — Grok 3 only
    ProviderCatalogEntry::new("grok", "grok-3", ProviderKind::Xai),
    // Ollama — generic passthrough (canonical id == alias).
    ProviderCatalogEntry::new("ollama", "ollama", ProviderKind::Ollama),
];

/// Env var holding a JSON array of user-defined OpenAI-compatible providers,
/// letting an operator add a provider without recompiling. Consulted only
/// after [`MODEL_REGISTRY`] misses, so built-in aliases always win.
pub const CUSTOM_PROVIDERS_ENV: &str = "ZO_CUSTOM_PROVIDERS";

/// A custom provider resolved for the current process: its owned model list and
/// auth requirement, plus the leaked `&'static` config the client consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomProviderUsability {
    pub name: &'static str,
    pub models: Vec<String>,
    pub requires_auth: bool,
    pub credential_env_vars: Vec<&'static str>,
    pub usable: bool,
}

#[derive(Clone)]
pub struct ResolvedCustomProvider {
    pub models: Vec<String>,
    pub requires_auth: bool,
    pub config: openai_compat::OpenAiCompatConfig,
    /// Optional operator-provided context window for every declared model.
    pub context_window: Option<u64>,
    /// Optional operator-provided max output cap for every declared model.
    pub max_output_tokens: Option<u64>,
    /// Only read behind the non-default `model-fit-hints` feature (see
    /// [`fit_hint_for_model`]) — dead in default builds by design.
    #[allow(dead_code)]
    pub fit_hint: Option<ModelFitHint>,
}

impl ResolvedCustomProvider {
    /// Whether this provider serves `model` (case-insensitive id match).
    fn serves(&self, model: &str) -> bool {
        self.canonical_model(model).is_some()
    }

    /// The registered model id matching `model`, used as the canonical id sent
    /// to the endpoint (preserves the operator's declared casing).
    fn canonical_model(&self, model: &str) -> Option<&str> {
        let lower = model.trim().to_ascii_lowercase();
        self.models
            .iter()
            .find(|m| m.to_ascii_lowercase() == lower)
            .map(String::as_str)
    }
}

/// Parse the `ZO_CUSTOM_PROVIDERS` JSON array. Pure, for unit testing.
pub(crate) fn parse_custom_providers(
    raw: &str,
) -> Result<Vec<openai_compat::CustomProviderConfig>, serde_json::Error> {
    serde_json::from_str(raw)
}

/// Resolve parsed custom-provider config into process-lifetime provider rows.
fn resolve_custom_providers(
    parsed: Vec<openai_compat::CustomProviderConfig>,
) -> Vec<ResolvedCustomProvider> {
    parsed
        .into_iter()
        .map(|custom| ResolvedCustomProvider {
            context_window: custom.context_window.filter(|&value| value > 0),
            max_output_tokens: custom.max_output_tokens.filter(|&value| value > 0),
            fit_hint: custom.to_fit_hint(),
            config: custom.to_static_config(),
            models: custom.models,
            requires_auth: custom.requires_auth,
        })
        .collect()
}

fn parse_resolved_custom_providers(
    raw: &str,
) -> Result<Vec<ResolvedCustomProvider>, serde_json::Error> {
    parse_custom_providers(raw).map(resolve_custom_providers)
}

fn load_custom_providers_from_env() -> Vec<ResolvedCustomProvider> {
    let Some(raw) = std::env::var(CUSTOM_PROVIDERS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Vec::new();
    };
    match parse_resolved_custom_providers(&raw) {
        Ok(providers) => providers,
        Err(error) => {
            eprintln!("[zo] {CUSTOM_PROVIDERS_ENV} is not a valid provider JSON array: {error}");
            Vec::new()
        }
    }
}

const DEEPSEEK_DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

/// `true` when a `DeepSeek` API key is reachable — set in the environment or
/// saved in the credential store via `/connect`. Gates the built-in `DeepSeek`
/// seed so the provider only surfaces once it is actually usable, mirroring how
/// OpenAI and Google built-ins appear only when their key/OAuth is present.
#[cfg(not(test))]
fn deepseek_credential_present() -> bool {
    std::env::var("DEEPSEEK_API_KEY").is_ok_and(|value| !value.trim().is_empty())
        || crate::oauth_store::load_openai_compat_api_key("DEEPSEEK_API_KEY")
            .ok()
            .flatten()
            .is_some_and(|value| !value.trim().is_empty())
}

/// Inert under unit tests: the custom-provider catalog tests assert exact
/// contents and must not depend on an ambient `DEEPSEEK_API_KEY` or a stored
/// credential on the developer's machine.
#[cfg(test)]
fn deepseek_credential_present() -> bool {
    false
}

/// The leaked, process-lifetime OpenAI-compatible config for the built-in
/// `DeepSeek` provider, initialized once so repeated catalog refreshes never
/// leak the strings again. `from_user` keeps `DeepSeek` on the same proven
/// data-driven path as a `/connect`-declared provider; `identity_maker` returns
/// `None` for
/// the `DeepSeek` name, so its identity is corrected without claiming a wrong
/// maker.
fn deepseek_seed_config() -> openai_compat::OpenAiCompatConfig {
    static CONFIG: OnceLock<openai_compat::OpenAiCompatConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        openai_compat::OpenAiCompatConfig::from_user(
            "DeepSeek",
            DEEPSEEK_DEFAULT_BASE_URL,
            Some("DEEPSEEK_API_KEY"),
            false,
        )
    })
}

/// zo's built-in provider seeds — known first-party OpenAI-compatible
/// providers (currently `DeepSeek`) that ship with zo so they work the moment
/// their key is present, without a `settings.json` / `ZO_CUSTOM_PROVIDERS`
/// entry. Each seed is gated on its credential so it never clutters the picker
/// unused. Model ids and context windows live in [`model_context_windows.json`].
fn builtin_seed_providers() -> Vec<ResolvedCustomProvider> {
    let mut seeds = Vec::new();
    if deepseek_credential_present() {
        seeds.push(ResolvedCustomProvider {
            models: vec![
                "deepseek-v4-pro".to_string(),
                "deepseek-v4-flash".to_string(),
            ],
            requires_auth: true,
            config: deepseek_seed_config(),
            context_window: None,
            max_output_tokens: None,
            fit_hint: None,
        });
    }
    seeds
}

/// Append built-in seeds the operator has not already declared. User/env-defined
/// providers are kept first so an explicit config wins on a model-id collision
/// (`custom_provider_for_model` returns the first match).
fn with_builtin_seed(mut providers: Vec<ResolvedCustomProvider>) -> Vec<ResolvedCustomProvider> {
    for seed in builtin_seed_providers() {
        let already_declared = providers
            .iter()
            .any(|existing| seed.models.iter().any(|model| existing.serves(model)));
        if !already_declared {
            providers.push(seed);
        }
    }
    providers
}

/// The user-defined provider catalog. It is initialized from
/// `ZO_CUSTOM_PROVIDERS` (plus zo's built-in seeds), but can be refreshed
/// after an in-session `/connect` writes settings so the current TUI does not
/// need a restart before `/model` sees newly configured providers.
fn custom_provider_store() -> &'static RwLock<Vec<ResolvedCustomProvider>> {
    static CACHE: OnceLock<RwLock<Vec<ResolvedCustomProvider>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(with_builtin_seed(load_custom_providers_from_env())))
}

fn custom_provider_snapshot() -> Vec<ResolvedCustomProvider> {
    custom_provider_store()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Replace the process-local custom-provider catalog from a JSON array. This is
/// the live companion to the startup env bridge: `/connect` writes settings,
/// then calls this so model routing and the picker reflect the new provider in
/// the already-running process.
pub fn refresh_custom_providers_from_json(raw: &str) -> Result<(), serde_json::Error> {
    let providers = with_builtin_seed(parse_resolved_custom_providers(raw)?);
    let mut guard = custom_provider_store()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = providers;
    Ok(())
}

/// Re-read the process-local custom-provider catalog from
/// `ZO_CUSTOM_PROVIDERS`.
pub fn refresh_custom_providers_from_env() {
    let mut guard = custom_provider_store()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = with_builtin_seed(load_custom_providers_from_env());
}

/// Split an explicit `provider/model` selection into `(provider, model)`.
///
/// Used when the same model id is served by both a built-in provider (e.g.
/// Anthropic OAuth `claude-opus-4-8`) and a custom gateway (e.g. agentrouter).
/// Bare ids keep the built-in route; only a qualified selection opts into the
/// custom provider.
#[must_use]
pub fn split_provider_model_ref(model: &str) -> Option<(&str, &str)> {
    let trimmed = model.trim();
    let (provider, model_id) = trimmed.rsplit_once('/')?;
    let provider = provider.trim();
    let model_id = model_id.trim();
    if provider.is_empty() || model_id.is_empty() {
        return None;
    }
    // Reject bare paths like "/model" and keep URLs out of this syntax.
    if provider.contains("://") {
        return None;
    }
    Some((provider, model_id))
}

fn provider_name_matches(configured: &str, requested: &str) -> bool {
    let configured = configured.trim();
    let requested = requested.trim();
    if configured.eq_ignore_ascii_case(requested) {
        return true;
    }
    // Allow `agent-router` / `agent_router` to match display name `agent router`.
    let normalize = |value: &str| {
        value
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    !normalize(configured).is_empty() && normalize(configured) == normalize(requested)
}

/// The custom provider that serves `model`, if any.
///
/// Bare model ids only match when the id is not a built-in Anthropic/OpenAI/
/// Google/xAI alias collision that should keep first-party routing. Explicit
/// `provider/model` selections always resolve to the named custom provider so
/// gateways like agentrouter can serve the same Claude id without hijacking
/// OAuth.
#[must_use]
pub fn custom_provider_for_model(model: &str) -> Option<ResolvedCustomProvider> {
    if let Some((provider_name, model_id)) = split_provider_model_ref(model) {
        return custom_provider_snapshot().into_iter().find(|provider| {
            provider_name_matches(provider.config.provider_name, provider_name)
                && provider.serves(model_id)
        });
    }

    custom_provider_snapshot()
        .into_iter()
        .find(|provider| provider.serves(model))
}

/// Format a picker / `/model` selection that must route through a specific
/// custom provider even when the model id collides with a built-in.
#[allow(dead_code)]
#[must_use]
pub fn format_provider_model_ref(provider: &str, model: &str) -> String {
    format!("{}/{}", provider.trim(), model.trim())
}

/// Wire model id to send to the remote API for `model`.
///
/// For bare ids this is [`resolve_model_alias`]. For `provider/model` this is
/// the model half only (e.g. `agent router/claude-opus-4-8` → `claude-opus-4-8`).
#[must_use]
pub fn wire_model_id(model: &str) -> String {
    if let Some((_provider, model_id)) = split_provider_model_ref(model) {
        return resolve_model_alias(model_id);
    }
    resolve_model_alias(model)
}

fn custom_provider_context_window(raw_model: &str, canonical_model: &str) -> Option<u64> {
    custom_provider_for_model(canonical_model)
        .or_else(|| custom_provider_for_model(raw_model))
        .and_then(|provider| provider.context_window)
        .filter(|&value| value > 0)
}

fn custom_provider_max_output_tokens(raw_model: &str, canonical_model: &str) -> Option<u64> {
    custom_provider_for_model(canonical_model)
        .or_else(|| custom_provider_for_model(raw_model))
        .and_then(|provider| provider.max_output_tokens)
        .filter(|&value| value > 0)
}

/// Configured custom providers as `(display name, model ids)` for UIs such as
/// the model picker, so models reached via `/connect` (Ollama / LM Studio /
/// `DeepSeek` / …) appear alongside the built-ins. Empty unless
/// `ZO_CUSTOM_PROVIDERS` is populated (the bootstrap mirrors settings.json
/// into it).
#[must_use]
pub fn custom_provider_catalog() -> Vec<(&'static str, Vec<String>)> {
    custom_provider_snapshot()
        .into_iter()
        .map(|provider| (provider.config.provider_name, provider.models))
        .collect()
}

fn custom_provider_is_usable(provider: &ResolvedCustomProvider) -> bool {
    !provider.requires_auth
        || provider.config.credential_env_vars.iter().any(|env| {
            env_non_empty(env)
                || crate::oauth_store::load_openai_compat_api_key(env)
                    .ok()
                    .flatten()
                    .is_some_and(|value| !value.trim().is_empty())
        })
}

/// Custom provider usability details for UIs that need to explain why a
/// configured provider is hidden from Smart Router's usable routing pool.
#[must_use]
pub fn custom_provider_usability_catalog() -> Vec<CustomProviderUsability> {
    custom_provider_snapshot()
        .into_iter()
        .map(|provider| {
            let usable = custom_provider_is_usable(&provider);
            CustomProviderUsability {
                name: provider.config.provider_name,
                models: provider.models,
                requires_auth: provider.requires_auth,
                credential_env_vars: provider.config.credential_env_vars.to_vec(),
                usable,
            }
        })
        .collect()
}

/// Custom providers whose declared models are usable for Smart Router display
/// and routing. Auth-required providers are included only when one of their
/// credential env vars is present in env or saved in the credential store.
#[must_use]
pub fn custom_provider_usable_catalog() -> Vec<(&'static str, Vec<String>)> {
    custom_provider_snapshot()
        .into_iter()
        .filter(custom_provider_is_usable)
        .map(|provider| (provider.config.provider_name, provider.models))
        .collect()
}

/// Optional read-only fit hint for `model`.
///
/// The feature is disabled by default and never drives selection or serving.
/// When enabled, callers may display the static/catalog metadata as an
/// operator hint.
#[must_use]
#[allow(unused_variables)]
pub fn fit_hint_for_model(model: &str) -> Option<ModelFitHint> {
    #[cfg(feature = "model-fit-hints")]
    {
        let canonical = resolve_model_alias(model);
        let lower = canonical.to_ascii_lowercase();
        if let Some(entry) = MODEL_REGISTRY.iter().find(|entry| {
            entry.alias == lower || entry.canonical_model_id.eq_ignore_ascii_case(&canonical)
        }) {
            return entry.fit_hint;
        }
        custom_provider_for_model(&canonical).and_then(|provider| provider.fit_hint)
    }

    #[cfg(not(feature = "model-fit-hints"))]
    {
        None
    }
}

#[must_use]
pub fn provider_enabled(kind: ProviderKind) -> bool {
    matches!(kind, ProviderKind::Anthropic)
        || explicit_non_claude_adapter_gate_enabled()
        || provider_configured(kind)
}

/// Strict provider predicate for Smart Router usable-model inventory.
///
/// Unlike [`provider_enabled`], this ignores broad experimental adapter flags.
/// Smart routing may display/select only providers that are actually configured,
/// plus Anthropic's default first-party path.
#[must_use]
pub fn provider_usable_for_smart_inventory(kind: ProviderKind) -> bool {
    provider_configured(kind)
}

/// `true` when a saved ChatGPT OAuth subscription token exists. A logged-in
/// ChatGPT subscription activates the OpenAI provider (model picker + routing)
/// without needing an API key or the experimental-adapters flag — the
/// subscription itself is the credential.
fn openai_oauth_present() -> bool {
    !external_credential_probes_disabled()
        && crate::oauth_store::load_openai_oauth()
            .ok()
            .flatten()
            .is_some()
}

fn external_credential_probes_disabled() -> bool {
    std::env::var("ZO_DISABLE_EXTERNAL_CREDENTIALS")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

fn explicit_non_claude_adapter_gate_enabled() -> bool {
    [EXPERIMENTAL_PROVIDERS_ENV, NON_CLAUDE_ADAPTERS_ENV]
        .into_iter()
        .any(|key| {
            std::env::var(key).ok().is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
        })
}

fn provider_configured(kind: ProviderKind) -> bool {
    match kind {
        ProviderKind::Anthropic => true,
        ProviderKind::OpenAi => {
            openai_oauth_present()
                || env_non_empty(kind.metadata().auth_env)
                || env_non_empty(kind.metadata().base_url_env)
        }
        ProviderKind::Google => {
            gemini_code_assist::oauth_present()
                || google_auth::gemini_oauth_available()
                || env_non_empty(kind.metadata().auth_env)
                || env_non_empty(kind.metadata().base_url_env)
        }
        ProviderKind::Xai | ProviderKind::Ollama => {
            env_non_empty(kind.metadata().auth_env) || env_non_empty(kind.metadata().base_url_env)
        }
    }
}

#[must_use]
pub fn non_claude_adapters_enabled() -> bool {
    // Explicit opt-in flag (either spelling) wins.
    if explicit_non_claude_adapter_gate_enabled() {
        return true;
    }
    // Implicit: any non-Claude provider the user has actually *configured* —
    // via OAuth/ADC, API key, or a custom base URL — auto-enables non-Claude
    // model-family detection. Individual provider routing still uses
    // `provider_enabled(kind)`, so configuring Google no longer enables xAI.
    [
        ProviderKind::Xai,
        ProviderKind::OpenAi,
        ProviderKind::Google,
        ProviderKind::Ollama,
    ]
    .into_iter()
    .any(provider_configured)
}

/// Whether environment variable `key` is set to a non-empty (after trim) value.
fn env_non_empty(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// Resolve a user-facing model alias to its canonical model id.
///
/// This is the single source of truth shared between the API client and the
/// CLI: it consults the [`provider_catalog`] first, then recovers an
/// unambiguous *near-miss* of a built-in alias (a typo like `fable5` → `fable`),
/// and finally falls back to normalising fully-qualified Claude ids
/// (dot-versioned → hyphenated). Experimental (non-Claude) aliases stay
/// untouched while their adapter gate is disabled, so the caller surfaces an
/// "unsupported provider" error rather than silently dialing out.
///
/// The near-miss recovery is what stops a mistyped alias from being passed
/// through verbatim to a provider that rejects it with an opaque `404
/// not_found` (the `fable5`/gemini-typo trap): a close typo of a real alias now
/// resolves to that alias's canonical id instead. It is deliberately
/// conservative — only a *unique*, *tight* match snaps — so distinct real
/// models (`gpt-5.6-sol` vs `gpt-5.5`), fully-qualified canonical ids, and
/// custom-provider model ids are never rerouted.
#[must_use]
pub fn resolve_model_alias(model: &str) -> String {
    let trimmed = model.trim();

    // Explicit provider/model keeps the selection token intact for routing /
    // session persistence. Wire id extraction is `wire_model_id`.
    if split_provider_model_ref(trimmed).is_some() {
        return trimmed.to_string();
    }

    let lower = trimmed.to_ascii_lowercase();
    let allow_experimental = non_claude_adapters_enabled();

    if let Some(entry) = MODEL_REGISTRY.iter().find(|entry| entry.alias == lower) {
        if provider_enabled(entry.provider) || allow_experimental {
            return entry.canonical_model_id.to_string();
        }
        return trimmed.to_string();
    }

    // Fallback: a user-defined provider (ZO_CUSTOM_PROVIDERS). The static
    // registry above always wins, so a bare custom id can never shadow a
    // built-in alias. Explicit `provider/model` is handled above.
    if let Some(custom) = custom_provider_for_model(&lower) {
        if let Some(canonical) = custom.canonical_model(&lower) {
            return canonical.to_string();
        }
    }

    // Near-miss recovery: snap a typo to the single built-in alias it clearly
    // meant (`fable5` → `fable` → `claude-fable-5`) instead of forwarding the
    // bogus id to a provider that 404s it. Skipped for an already-canonical
    // registry id (a valid fully-qualified target must never be snapped to a
    // shorter alias) and gated on the target provider being enabled, exactly
    // like the exact-alias path above, so a Gemini/GPT typo under a Claude-only
    // setup passes through as "unsupported" rather than dialing out.
    let is_known_canonical = MODEL_REGISTRY
        .iter()
        .any(|entry| entry.canonical_model_id.eq_ignore_ascii_case(&lower));
    if !is_known_canonical {
        if let Some(entry) = nearest_alias_entry(&lower) {
            if provider_enabled(entry.provider) || allow_experimental {
                return entry.canonical_model_id.to_string();
            }
        }
    }

    // Displayed short form of a known model → its canonical id. Canonical Claude
    // (and DeepSeek) ids spell the version with hyphens (`claude-opus-4-8`), but
    // the model and the user naturally type the *displayed* short form
    // (`opus-4.8`, `fable-5`, `sonnet-5`), which the provider 404s verbatim
    // (`not_found_error: model: opus-4.8`). Snap it to the canonical only when
    // dot→hyphen normalization — optionally under the `claude-` family prefix —
    // lands EXACTLY on a known canonical id. This is version-preserving by
    // construction: `opus-4.8` → `claude-opus-4-8` resolves, but a genuinely
    // different version like `opus-5` normalizes to `claude-opus-5`, which is no
    // canonical, so it still passes through untouched — and providers whose
    // canonical ids keep dots (GPT `gpt-5.5-mini`, Gemini `gemini-3.7-flash`)
    // never match the dashed candidate, so their short forms are unaffected.
    // The near-miss guard above deliberately refuses cross-version snaps and so
    // cannot cover this same-version reformatting.
    let dashed = lower.replace('.', "-");
    let prefixed = format!("claude-{dashed}");
    for candidate in [dashed.as_str(), prefixed.as_str()] {
        if let Some(entry) = MODEL_REGISTRY
            .iter()
            .find(|entry| entry.canonical_model_id.eq_ignore_ascii_case(candidate))
        {
            if provider_enabled(entry.provider) || allow_experimental {
                return entry.canonical_model_id.to_string();
            }
        }
    }

    normalize_model_id(trimmed)
}

/// Resolve model-authored spawn input to a registered Claude family target.
///
/// This composes with [`resolve_model_alias`] but adds one spawn-only rule:
/// unknown versions of the four built-in Claude families snap to that family's
/// registered canonical. Non-Claude ids, explicit `provider/model` references,
/// and operator-declared custom-provider ids retain normal passthrough
/// semantics. Main-session model selection deliberately continues to call the
/// permissive resolver directly.
#[must_use]
pub fn resolve_registered_model_alias(model: &str) -> String {
    let trimmed = model.trim();
    let resolved = resolve_model_alias(trimmed);

    // Explicit provider routing and operator-declared model ids are already
    // intentional, registered targets. Spawn hardening must not rewrite them.
    if split_provider_model_ref(trimmed).is_some() || custom_provider_for_model(trimmed).is_some() {
        return resolved;
    }
    if MODEL_REGISTRY
        .iter()
        .any(|entry| entry.canonical_model_id.eq_ignore_ascii_case(&resolved))
    {
        return resolved;
    }

    let lower = resolved.to_ascii_lowercase();
    let family_token = lower.strip_prefix("claude-");
    let is_family = |token: &&str| matches!(*token, "fable" | "opus" | "sonnet" | "haiku");
    let family = if let Some(claude_tokens) = family_token {
        claude_tokens.split(['-', '[', '@']).find(is_family)
    } else {
        lower.split(['-', '[', '@']).next().filter(is_family)
    };
    let Some(family) = family else {
        return resolved;
    };
    MODEL_REGISTRY
        .iter()
        .find(|entry| entry.provider == ProviderKind::Anthropic && entry.alias == family)
        .map_or(resolved, |entry| entry.canonical_model_id.to_string())
}

/// The single built-in alias that is an unambiguous near-miss of `lower`, or
/// `None` when nothing is close enough or the closest is a tie. Guards a typo
/// (`fable5` → `fable`) while refusing to reroute a distinct model: `gpt-5.6`
/// is edit-distance 1 from *both* `gpt-5.6-sol` and `gpt-5.5`, so it is ambiguous
/// and returns `None`.
///
/// Three guards keep it conservative, so it only ever fires on a confident
/// human typo, never on an intended-but-unknown id (custom-provider models,
/// new canonical ids):
/// - unique closest — a tie means we cannot know which was meant;
/// - edit distance ≤ 2 — single/double fat-finger only;
/// - shared 3-char prefix — you typed at least the first three characters
///   correctly, which stops junk (`op`) from snapping onto a short alias
///   (`opus`) and stops far matches that merely happen to fall within 2.
fn nearest_alias_entry(lower: &str) -> Option<&'static ProviderCatalogEntry> {
    // Only alias-like inputs are candidates; a long structured id is never a
    // "typo of an alias", and bounding the length keeps the DP trivially cheap.
    if lower.is_empty() || lower.chars().count() > 32 {
        return None;
    }
    let mut best_dist = usize::MAX;
    let mut best_entry: Option<&'static ProviderCatalogEntry> = None;
    let mut unique = false;
    for entry in MODEL_REGISTRY {
        let dist = levenshtein(lower, entry.alias);
        if dist < best_dist {
            best_dist = dist;
            best_entry = Some(entry);
            unique = true;
        } else if dist == best_dist {
            unique = false;
        }
    }
    let entry = best_entry?;
    (unique
        && best_dist <= 2
        && best_dist < entry.alias.chars().count()
        && shared_prefix_len(lower, entry.alias) >= 3
        // Never turn one model into a DIFFERENT version of another: a typo may
        // only differ from its alias in the *word*, never in an explicit
        // version/qualifier number. This is what keeps `opus-5`, `sonnet-4`,
        // `grok-4`, `gpt-5.5-mini`, and `gemini-3.7-flash` from silently
        // snapping onto an older/cheaper sibling — a semantic reroute that is
        // worse than a clean "unsupported model" error — while `fable5`,
        // `fabel`, and `opuss` still recover (they carry no version token).
        && version_tokens(lower) == version_tokens(entry.alias))
    .then_some(entry)
}

/// The explicit version/qualifier tokens of a model id, in order: each maximal
/// run that *starts with a digit at a field boundary* (string start, or right
/// after `-`/`.`) and extends over the following alphanumerics and dots.
///
/// This captures `5.6` in `gpt-5.6-sol`, `5` in `opus-5`, and `5.6x` in `gpt-5.6x`
/// (trailing letters glued to a version are part of the version spec, so a
/// suffix change is treated as a distinct model). It deliberately does NOT
/// capture the `5` in `fable5`: that digit follows a letter, so it is a
/// fat-finger on the `fable` alias rather than an explicit version field.
fn version_tokens(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let prev_is_boundary = i == 0 || matches!(chars[i - 1], '-' | '.');
        if chars[i].is_ascii_digit() && prev_is_boundary {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '.') {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else {
            i += 1;
        }
    }
    out
}

/// Levenshtein edit distance over `char`s (so multibyte input never panics on a
/// byte index). Bounded by the ≤32-char guard in [`nearest_alias_entry`], so
/// the two-row DP is cheap.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Count leading `char`s shared by `a` and `b`.
fn shared_prefix_len(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(x, y)| x == y)
        .count()
}

/// Normalize dot-separated version numbers to hyphens in fully-qualified
/// Claude model ids (e.g. `claude-opus-4.6` → `claude-opus-4-6`). Other
/// providers' ids are passed through verbatim.
fn normalize_model_id(model: &str) -> String {
    if model.starts_with("claude-") && model.contains('.') {
        model.replace('.', "-")
    } else {
        model.to_string()
    }
}

#[must_use]
pub fn metadata_for_model(model: &str) -> Option<ProviderMetadata> {
    model_capability_for_model(model)
        .provider
        .map(ProviderKind::metadata)
}

#[must_use]
pub fn explicit_non_claude_provider_kind(model: &str) -> Option<ProviderKind> {
    let lower = model.trim().to_ascii_lowercase();
    if lower.starts_with("grok") {
        return Some(ProviderKind::Xai);
    }
    if catalog_entry_for_token(MODEL_REGISTRY, &lower)
        .is_some_and(|entry| entry.provider == ProviderKind::OpenAi)
        || is_openai_builtin_model_prefix(&lower)
    {
        return Some(ProviderKind::OpenAi);
    }
    if lower.starts_with("gemini") {
        return Some(ProviderKind::Google);
    }
    if std::env::var("OLLAMA_BASE_URL").is_ok() {
        return Some(ProviderKind::Ollama);
    }
    None
}

/// The maker attribution used in the non-Anthropic identity override, or
/// `None` when zo must not assert a specific maker.
///
/// zo's base system prompt hardcodes a Claude Code identity ("You are Claude
/// Code…", "Model family: Claude Opus 4.8"). Served by any non-Anthropic
/// provider, that text makes the model introduce itself as Claude. Each
/// first-party non-Anthropic provider returns its real maker so the override
/// names it; `Ollama` returns `None` (zo cannot know which model/lab serves
/// a local endpoint, so it corrects the Claude claim without inventing a false
/// maker). `Anthropic` returns `None` because its path is never rewritten — the
/// verbatim identity line is a Claude Max OAuth fingerprint requirement.
#[must_use]
pub fn maker_for_provider(kind: ProviderKind) -> Option<&'static str> {
    match kind {
        ProviderKind::OpenAi => Some("OpenAI"),
        ProviderKind::Xai => Some("xAI"),
        ProviderKind::Google => Some("Google"),
        ProviderKind::Anthropic | ProviderKind::Ollama => None,
    }
}

/// Prepend the non-Anthropic identity override to a joined system prompt so a
/// non-Claude model does not introduce itself as Claude. Single source of truth
/// shared by every non-Anthropic backend (ChatGPT/Responses, Gemini Code
/// Assist, and the OpenAI-compatible chat path).
///
/// Returns `system` unchanged when it is empty (nothing to override). Otherwise
/// it prepends a one-paragraph override naming the model and — when `maker` is
/// `Some` — its maker, instructing the model to follow zo's Claude-authored
/// operating manual and provider-neutral response style contract while keeping
/// its own identity. With `maker == None` the override still corrects the
/// Claude claim but asserts no specific maker, so a custom/self-hosted endpoint
/// is never mislabeled.
///
/// The Anthropic backend must never call this: its first system block has to be
/// the verbatim Claude Code identity line (OAuth fingerprint requirement).
#[must_use]
pub fn apply_non_anthropic_identity(system: &str, model: &str, maker: Option<&str>) -> String {
    if system.is_empty() {
        return system.to_string();
    }
    let maker_clause = match maker {
        Some(maker) => format!("a large language model made by {maker}"),
        None => "a large language model".to_string(),
    };
    format!(
        "You are {model}, {maker_clause}, operating through the zo coding CLI. The operating \
         manual below was written for Claude Code; follow its tools, workflow, and provider-neutral \
         response style contract, but your identity is {model} — do not claim to be Claude or to \
         be made by Anthropic.\n\n{system}"
    )
}

#[must_use]
pub fn detect_provider_kind(model: &str) -> ProviderKind {
    if let Some(metadata) = metadata_for_model(model) {
        return metadata.provider;
    }
    if anthropic::has_auth_from_env_or_saved().unwrap_or(false) {
        return ProviderKind::Anthropic;
    }
    if non_claude_adapters_enabled() {
        if gemini_code_assist::oauth_present()
            || openai_compat::has_api_key("GOOGLE_API_KEY")
            || google_auth::gemini_oauth_available()
        {
            return ProviderKind::Google;
        }
        if openai_compat::has_api_key("OPENAI_API_KEY") {
            return ProviderKind::OpenAi;
        }
        if openai_compat::has_api_key("XAI_API_KEY") {
            return ProviderKind::Xai;
        }
    }
    if std::env::var("OLLAMA_BASE_URL").is_ok() {
        return ProviderKind::Ollama;
    }
    ProviderKind::Anthropic
}

/// Flatten a tool result's content blocks into a single plain-text string for
/// providers whose `tool`/`function_call_output` messages accept only text.
///
/// Blocks are joined by `\n`; JSON blocks are serialized; image blocks degrade
/// to a `[image <media-type>]` placeholder rather than being dropped silently.
/// Shared by the OpenAI-compatible, ChatGPT/Responses, and Gemini Code Assist
/// backends so the flattening stays byte-identical across them.
pub(crate) fn flatten_tool_result_content(
    content: &[crate::types::ToolResultContentBlock],
) -> String {
    use crate::types::ToolResultContentBlock;

    let mut flattened = String::new();
    for (index, block) in content.iter().enumerate() {
        if index > 0 {
            flattened.push('\n');
        }
        match block {
            ToolResultContentBlock::Text { text } => flattened.push_str(text),
            ToolResultContentBlock::Json { value } => flattened.push_str(&value.to_string()),
            ToolResultContentBlock::Image { source } => {
                flattened.push_str("[image ");
                flattened.push_str(&source.media_type);
                flattened.push(']');
            }
        }
    }
    flattened
}

/// The `data:<media-type>;base64,<data>` URL for an inline image, shared by the
/// OpenAI-compatible and ChatGPT/Responses backends (both send images as data
/// URLs inside an `image_url`). Gemini uses a distinct `inline_data` structure,
/// not a data URL, so it does not use this.
pub(crate) fn image_data_url(source: &crate::types::ImageSource) -> String {
    format!("data:{};base64,{}", source.media_type, source.data)
}

/// Exponential backoff schedule shared by the retrying provider clients
/// (Anthropic, OpenAI-compatible, ChatGPT/Responses): `initial_backoff`
/// doubled per attempt (`attempt` is 1-based), capped at `max_backoff`.
///
/// Returns [`crate::error::ApiError::BackoffOverflow`] when the doubling
/// multiplier (`1 << (attempt - 1)`) would overflow `u32`; the caller adds
/// jitter via `retry_backoff::spread_backoff`.
pub(crate) fn backoff_for_attempt(
    attempt: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
) -> Result<Duration, crate::error::ApiError> {
    let Some(multiplier) = 1_u32.checked_shl(attempt.saturating_sub(1)) else {
        return Err(crate::error::ApiError::BackoffOverflow {
            attempt,
            base_delay: initial_backoff,
        });
    };
    Ok(initial_backoff
        .checked_mul(multiplier)
        .map_or(max_backoff, |delay| delay.min(max_backoff)))
}

/// Default per-chunk idle budget for provider streams (ms). A streaming read that
/// receives no bytes for this long is aborted with a retryable
/// [`crate::error::ApiError::stream_idle_timeout`] instead of hanging the turn
/// forever — a half-open TCP connection or a backend that holds the socket open
/// while it reasons silently would otherwise freeze the spinner indefinitely
/// (and let a spawned sub-agent overrun its wall-clock budget, since the
/// iteration-boundary deadline check never runs while a read is parked).
/// Sized well above any normal inter-chunk gap.
pub(crate) const STREAM_IDLE_TIMEOUT_MS: u64 = 90_000;

/// Env override for [`STREAM_IDLE_TIMEOUT_MS`]; `0` disables the idle timeout
/// (restores the unbounded-wait behaviour). Shared by the Anthropic and
/// OpenAI-compatible streams; the ChatGPT/Responses backend keeps its own
/// `ZO_CHATGPT_STREAM_IDLE_TIMEOUT_MS` because it also drives an in-place
/// restart on idle.
pub(crate) const STREAM_IDLE_TIMEOUT_ENV: &str = "ZO_STREAM_IDLE_TIMEOUT_MS";

/// Resolve the per-chunk idle budget, honouring the env override. `None` means
/// "no timeout" (override set to `0`).
#[must_use]
pub(crate) fn stream_idle_timeout() -> Option<std::time::Duration> {
    let millis = std::env::var(STREAM_IDLE_TIMEOUT_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(STREAM_IDLE_TIMEOUT_MS);
    (millis > 0).then(|| std::time::Duration::from_millis(millis))
}

// ---------------------------------------------------------------------------
// Mid-stream restart predicate (shared by every streaming backend)
// ---------------------------------------------------------------------------
//
// A streaming turn can be transparently re-opened only while nothing has been
// surfaced yet — once a non-empty text or tool-argument delta has crossed into
// the caller's render path, restarting would duplicate output, so the fault
// must propagate instead. The Anthropic Messages API and the ChatGPT/Responses
// backend share this rule (neither has a resumable-offset token; recovery means
// re-sending the request), so the predicate lives here as the single source of
// truth rather than being duplicated per backend.

/// Pure restart predicate (unit-testable without a live socket): restart only
/// while the turn is uncommitted, the fault is retryable, and the restart
/// budget is not yet spent.
#[must_use]
pub(crate) fn should_restart(
    committed: bool,
    retryable: bool,
    attempts: u32,
    max_retries: u32,
) -> bool {
    !committed && retryable && attempts < max_retries
}

/// Default total wall-clock ceiling over a pre-commit restart sequence,
/// shared by the OpenAI-compatible and Gemini Code Assist streams (the
/// ChatGPT/Responses backend keeps its own equal `MAX_RESTART_WALLCLOCK`).
/// Attempt counting alone lets a *silent* backend hold the turn for
/// idle-timeout × restarts; this bounds the whole sequence by elapsed time.
pub(crate) const DEFAULT_MAX_RESTART_WALLCLOCK: Duration = Duration::from_secs(120);

/// Restart predicate with an added total wall-clock budget over the restart
/// sequence. A pre-commit stream can stall (idle-timeout) and re-open up to
/// `max_retries` times; each idle wait can be tens of seconds, so the attempt
/// count alone lets a *silent* backend hold the turn for minutes (the observed
/// ~275 s freeze: idle-timeout × restarts with no overall ceiling). This bounds
/// the whole sequence by elapsed time too, so a backend that never produces a
/// byte fails out promptly instead of looping the full budget. `elapsed` is the
/// time since the first restart in the sequence (`None` before any restart, so
/// the first restart is always allowed); the wall-clock gate only tightens
/// [`should_restart`], never loosens it.
#[must_use]
pub(crate) fn should_restart_within_budget(
    committed: bool,
    retryable: bool,
    attempts: u32,
    max_retries: u32,
    elapsed: Option<Duration>,
    max_wallclock: Duration,
) -> bool {
    should_restart(committed, retryable, attempts, max_retries)
        && elapsed.is_none_or(|e| e < max_wallclock)
}

/// Whether an event surfaces output that a restart would duplicate. Only
/// non-empty text and tool-call argument deltas commit; message/block framing,
/// reasoning deltas, and empty placeholders are bookkeeping — replaying them
/// after a stalled stream is preferable to wedging the turn forever.
#[must_use]
pub(crate) fn crosses_restart_commit_boundary(event: &StreamEvent) -> bool {
    match event {
        StreamEvent::ContentBlockDelta(delta) => match &delta.delta {
            ContentBlockDelta::TextDelta { text } => !text.is_empty(),
            ContentBlockDelta::InputJsonDelta { partial_json } => !partial_json.is_empty(),
            ContentBlockDelta::ThinkingDelta { .. } | ContentBlockDelta::SignatureDelta { .. } => {
                false
            }
        },
        StreamEvent::MessageStart(_)
        | StreamEvent::MessageDelta(_)
        | StreamEvent::ContentBlockStart(_)
        | StreamEvent::ContentBlockStop(_)
        | StreamEvent::MessageStop(_) => false,
    }
}

/// Shared sink for mid-stream restart notices, held by every streaming
/// backend that performs transparent pre-commit restarts (ChatGPT, the
/// OpenAI-compatible adapter, Gemini Code Assist).
///
/// A transparent restart never returns an error the establish-time retry
/// layer could render, so without a notice the turn just freezes for the
/// backoff. Wrapping the optional callback here keeps the "fire if installed"
/// pattern in one place and lets stream structs keep `#[derive(Debug)]`
/// (a bare `dyn Fn` is not `Debug`).
pub(crate) struct StreamRetryNotifier(
    Option<std::sync::Arc<dyn Fn(core_types::StreamRetryNotice) + Send + Sync>>,
);

impl StreamRetryNotifier {
    /// No sink installed (the default): notices are dropped, preserving the
    /// log-only behaviour for non-interactive callers.
    pub(crate) const fn none() -> Self {
        Self(None)
    }

    pub(crate) fn install(
        &mut self,
        callback: impl Fn(core_types::StreamRetryNotice) + Send + Sync + 'static,
    ) {
        self.0 = Some(std::sync::Arc::new(callback));
    }

    pub(crate) fn notify(&self, notice: core_types::StreamRetryNotice) {
        if let Some(callback) = &self.0 {
            callback(notice);
        }
    }
}

impl std::fmt::Debug for StreamRetryNotifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamRetryNotifier")
            .field("installed", &self.0.is_some())
            .finish()
    }
}

fn model_id_matches_prefix_segment(model_id: &str, prefix: &str) -> bool {
    model_id == prefix
        || model_id
            .strip_prefix(prefix)
            .is_some_and(|suffix| matches!(suffix.as_bytes().first(), Some(b'-' | b'@' | b'[')))
}

const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 64_000;
const EXTENDED_MAX_OUTPUT_TOKENS: u32 = 128_000;
const LEGACY_OPUS_MAX_OUTPUT_TOKENS: u32 = 32_000;

#[must_use]
pub fn max_tokens_for_model(model: &str) -> u32 {
    model_capability_for_model(model)
        .max_output_tokens
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
}

fn max_output_tokens_for_model_id(raw_model: &str, model_id: &str) -> u32 {
    // Anthropic's current model table documents 128k synchronous max output for
    // Fable/Mythos 5, Sonnet 5, and Opus 4.6+; the streaming guide explicitly
    // uses `claude-opus-4-8` with `max_tokens: 128000` as the large-output path
    // that avoids HTTP timeouts. Let those models use the full documented cap.
    // Older Opus 4.1/4.0 variants remain at their documented 32k cap; everything
    // else keeps the prior conservative 64k default, which still prevents the
    // old 32k mid-tool-call truncation without over-claiming unknown providers.
    if model_id.contains("fable")
        || model_id.contains("mythos")
        || model_id_matches_prefix_segment(model_id, "claude-sonnet-5")
        || model_id_matches_prefix_segment(model_id, "claude-opus-4-8")
        || model_id_matches_prefix_segment(model_id, "claude-opus-4-7")
        || model_id_matches_prefix_segment(model_id, "claude-opus-4-6")
    {
        return EXTENDED_MAX_OUTPUT_TOKENS;
    }
    if model_id_matches_prefix_segment(model_id, "claude-opus-4-1")
        || model_id_matches_prefix_segment(model_id, "claude-opus-4-0")
        || model_id == "claude-opus-4"
        || model_id_matches_prefix_segment(model_id, "claude-opus-4-20250514")
    {
        return LEGACY_OPUS_MAX_OUTPUT_TOKENS;
    }
    // Non-Anthropic models: honour the docs-verified catalog cap (GPT-5.5 128k,
    // DeepSeek V4 384k) before the conservative default, so large refactors and
    // full-file writes are not truncated mid-generation — the same failure the
    // Anthropic path was widened to 128k to avoid, previously still present for
    // every other provider because the catalog field was never read.
    if let Some(max_out) = custom_provider_max_output_tokens(raw_model, model_id) {
        return u32::try_from(max_out).unwrap_or(u32::MAX);
    }
    if let Some(max_out) = catalog_max_output_tokens(raw_model, model_id) {
        return u32::try_from(max_out).unwrap_or(u32::MAX);
    }
    DEFAULT_MAX_OUTPUT_TOKENS
}

/// Whether `model` uses Anthropic *adaptive* thinking — the server sizes the
/// thinking budget from `output_config.effort` rather than an explicit
/// `thinking.budget_tokens`. True for the Opus 4.6+/Fable generation (and the
/// matching Sonnet), false for Opus 4.5 and earlier, which still take a budget.
///
/// This is the SSOT that decides which Anthropic wire shape a request gets, so
/// the request builder never sends a deprecated `budget_tokens` to a model that
/// expects effort, nor an unsupported `output_config` to a legacy model.
#[must_use]
pub fn uses_adaptive_thinking(model: &str) -> bool {
    model_capability_for_model(model)
        .adaptive_thinking
        .unwrap_or(false)
}

fn adaptive_thinking_for_canonical(canonical_lower: &str) -> bool {
    // Fable is adaptive-only.
    if canonical_lower.contains("fable") {
        return true;
    }
    // Opus/Sonnet 4.6+ are adaptive; 4.5 and earlier are not. Match the
    // generation digits in the canonical id (e.g. `claude-opus-4-6`).
    if canonical_lower.contains("opus") || canonical_lower.contains("sonnet") {
        return !is_legacy_anthropic_generation(canonical_lower);
    }
    false
}

/// True when an Anthropic model id is the 4.5-or-earlier generation, which uses
/// legacy budget-based thinking. Recognizes the `4-5`/`4.5` (and older `-3`,
/// `4-0`…`4-5`) generation markers; anything newer (4-6+) is adaptive.
fn is_legacy_anthropic_generation(canonical_lower: &str) -> bool {
    const LEGACY_MARKERS: &[&str] = &[
        "4-5", "4.5", "4-4", "4.4", "4-3", "4.3", "4-2", "4.2", "4-1", "4.1", "4-0", "4.0", "-3-",
        "claude-3",
    ];
    LEGACY_MARKERS
        .iter()
        .any(|marker| canonical_lower.contains(marker))
}

/// Map a thinking budget (in tokens) to a provider-neutral [`EffortLevel`], so a
/// request that only carries a budget can still drive an adaptive model's
/// `output_config.effort` (and a GPT backend's `reasoning_effort`).
///
/// The thresholds are the exact inverse of the CLI `Effort` preset budgets
/// (`Effort::budget()`): each preset round-trips to the level it came from
/// (`Low` from 1024, `Medium` from 4096, `High` from 10000, `Xhigh` from 16000,
/// `Max` from 24000). This function has no `Ultra` bucket — it predates the
/// named Ultra tier and only derives a level from a raw budget number — so
/// `Effort::Ultra`'s 26000 budget and `Effort::Smart`'s 28000 budget both land
/// in the open-ended `Max` bucket rather than round-tripping to `Ultra`;
/// callers that carry the named `Effort::Ultra` preset (or `Effort::Smart`'s
/// dynamic band) must use `Effort::level()`/`Effort::band_ceiling()` directly
/// to get `Ultra`, not this budget-only fallback. The CLI-side
/// `Effort::level()` is the single source of truth for the non-Ultra
/// pairings, and a round-trip test in `effort_picker` fails loudly if the two
/// ever drift.
/// (P9 note: `Ultracode` was renamed to `Smart`, and the `ultra` token — a
/// mere alias of `Ultracode` before P9 — is now this separate static
/// `Effort::Ultra` preset in its own right; see `effort_picker.rs`.)
///
/// A previous calibration was tuned for large legacy budgets (`Max` needed a
/// budget above 48k), so every preset landed one to two tiers low on the wire:
/// a headless `ZO_EFFORT=max` reached Anthropic as `high`, and `xhigh`
/// reached GPT as `medium`. Boundaries now sit at the midpoint between
/// consecutive presets, so an off-preset custom budget snaps to the nearest
/// tier, and any budget above the top preset clamps to `Max`.
/// Combine a request's configured thinking budget with an optional escalation
/// **floor** (e.g. the deep-gate's `ApiRequest::effort_override`). The floor can
/// only raise effort, never lower it: the result is `max(configured, floor)`
/// when a floor is present, otherwise the configured budget unchanged. A zero or
/// absent floor is inert. Both inputs are `Option` so "no thinking configured"
/// (`None`) still escalates to the floor when one is set.
#[must_use]
pub fn effort_budget_with_floor(configured: Option<u32>, floor: Option<u32>) -> Option<u32> {
    match floor.filter(|&f| f > 0) {
        Some(f) => Some(configured.map_or(f, |c| c.max(f))),
        None => configured,
    }
}

#[must_use]
pub fn effort_level_for_budget(budget_tokens: u32) -> crate::types::EffortLevel {
    use crate::types::EffortLevel;
    match budget_tokens {
        0..=2_560 => EffortLevel::Low,         // Low preset = 1_024
        2_561..=7_048 => EffortLevel::Medium,  // Medium preset = 4_096
        7_049..=13_000 => EffortLevel::High,   // High preset = 10_000
        13_001..=20_000 => EffortLevel::Xhigh, // Xhigh preset = 16_000
        _ => EffortLevel::Max,                 // Max preset = 24_000
    }
}

/// The model-specific Zo tier used for selection, routing, and display for a
/// requested `level`. UI surfaces can ask this without branching on provider.
///
/// This is a **read-only capability projection**, not the final wire string.
/// A backend still applies its provider serializer after this selection; in
/// particular, GPT `Max`/`Ultra` are internal Zo tiers that both serialize
/// as the provider-supported `xhigh` value.
///
/// - Anthropic (`claude*`): [`EffortLevel::anthropic_for_model`] — Sonnet/Haiku
///   clamp `Xhigh -> High`; Opus/Fable keep the full scale.
/// - OpenAI (`gpt*`/`o3`/`o4`/`codex`, and custom OpenAI-compatible providers
///   detected by family): GPT-5.6 exposes internal `Max`; Sol/Terra also expose
///   internal `Ultra`. The final OpenAI wire projection maps both to `xhigh`.
///   GPT fast keeps `Xhigh` because `/fast` is service priority, not an effort
///   ceiling.
/// - Google (`gemini*`): Gemini 3 tops out at `high`, so top tiers project to High.
/// - Everything else (xAI, Ollama, unknown/custom non-OpenAI ids): **pass
///   through unchanged** — never silently downgrade a model whose ceiling we do
///   not know (the BUG-R14 trap: env-fallback misclassifying a custom provider).
///
/// Detection is **name/family-based**, deliberately *not* env-sensitive
/// [`detect_provider_kind`], so the result is deterministic and a custom
/// OpenAI-compatible model id is never re-tiered off ambient `OPENAI_API_KEY`.
#[must_use]
pub fn effective_effort_for_model(
    level: crate::types::EffortLevel,
    model: &str,
) -> crate::types::EffortLevel {
    use crate::types::{
        gpt_model_accepts_max, gpt_model_accepts_ultra, gpt_model_accepts_xhigh, EffortLevel,
    };

    let lower = resolve_model_alias(model).to_ascii_lowercase();

    // Anthropic: delegate to its model capability projection.
    if lower.starts_with("claude") {
        return level.anthropic_for_model(&lower);
    }

    // OpenAI family (builtin gpt/o3/o4/codex, or a registered custom provider
    // that resolves to an OpenAI-compatible model). GPT-5.6 exposes additional
    // internal selection tiers; gpt_for_model performs the separate, final
    // `Max | Ultra -> xhigh` wire projection. `/fast` is service priority and
    // does not lower the Xhigh ceiling.
    let is_openai = openai_gpt_model_family(&lower).is_some()
        || is_openai_builtin_model_prefix(&lower)
        || catalog_entry_for_token(MODEL_REGISTRY, &lower)
            .is_some_and(|entry| entry.provider == ProviderKind::OpenAi);
    if is_openai {
        if level == EffortLevel::Xhigh && !gpt_model_accepts_xhigh(&lower) {
            return EffortLevel::High;
        }
        if level == EffortLevel::Max && !gpt_model_accepts_max(&lower) {
            return EffortLevel::Xhigh;
        }
        if level == EffortLevel::Ultra && !gpt_model_accepts_ultra(&lower) {
            return EffortLevel::Xhigh;
        }
        return level;
    }

    // Google Gemini 3: tops out at `high`.
    if lower.starts_with("gemini") {
        return match level {
            EffortLevel::Xhigh | EffortLevel::Max | EffortLevel::Ultra => EffortLevel::High,
            other => other,
        };
    }

    // xAI / Ollama / unknown / custom non-OpenAI: conservative pass-through. We
    // do not know their ceiling, and guessing risks a silent downgrade; the
    // wire path applies whatever the provider actually enforces.
    level
}

/// Whether `model` actually accepts the `xhigh` reasoning tier — the
/// provider-neutral predicate UI surfaces use to decide whether to flag a
/// `xhigh`/`smart` selection as clamped. Defined in terms of
/// [`effective_effort_for_model`] so it can never disagree with the projection.
#[must_use]
pub fn model_supports_xhigh(model: &str) -> bool {
    use crate::types::EffortLevel;
    effective_effort_for_model(EffortLevel::Xhigh, model) == EffortLevel::Xhigh
}

/// Env var holding a JSON object of `{"model-or-family-prefix": "ultra|max|xhigh|high"}`
/// ceiling overrides, consulted before every built-in rule in
/// [`max_supported_effort`]. This is the zero-rebuild escape hatch for a newly
/// announced model's effort ceiling — mirrors how [`MODEL_CONTEXT_WINDOWS_ENV`]
/// lets an operator inject capability facts without recompiling. Longest
/// matching prefix wins (segment-boundary aware, see
/// [`crate::types::model_id_matches_family`]), so a specific override
/// (`gpt-5.7-nova`) beats a broader one (`gpt-5.7`) in the same JSON blob.
pub const MODEL_EFFORT_CEILINGS_ENV: &str = "ZO_MODEL_EFFORT_CEILINGS";

fn parse_effort_ceiling_value(value: &str) -> Option<crate::types::EffortLevel> {
    use crate::types::EffortLevel;
    match value.trim().to_ascii_lowercase().as_str() {
        "ultra" => Some(EffortLevel::Ultra),
        "max" => Some(EffortLevel::Max),
        "xhigh" => Some(EffortLevel::Xhigh),
        "high" => Some(EffortLevel::High),
        _ => None,
    }
}

/// Read fresh (not cached) so tests can set/unset the env var per-case — the
/// same non-caching choice [`env_model_context_window`] makes for the sibling
/// context-window override.
fn env_effort_ceiling_override(canonical_lower: &str) -> Option<crate::types::EffortLevel> {
    let raw = std::env::var(MODEL_EFFORT_CEILINGS_ENV).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let overrides: std::collections::HashMap<String, String> = serde_json::from_str(&raw).ok()?;
    let mut best: Option<(usize, crate::types::EffortLevel)> = None;
    for (prefix, value) in &overrides {
        let prefix_lower = prefix.trim().to_ascii_lowercase();
        if prefix_lower.is_empty() {
            continue;
        }
        if !crate::types::model_id_matches_family(canonical_lower, &prefix_lower) {
            continue;
        }
        let Some(level) = parse_effort_ceiling_value(value) else {
            continue;
        };
        if best.is_none_or(|(best_len, _)| prefix_lower.len() > best_len) {
            best = Some((prefix_lower.len(), level));
        }
    }
    best.map(|(_, level)| level)
}

/// Single source of truth for the highest internal [`crate::types::EffortLevel`]
/// tier Zo exposes for a `model` — consolidates what was previously
/// scattered across [`crate::types::gpt_model_accepts_ultra`]/
/// [`crate::types::gpt_model_accepts_max`], the Anthropic xhigh/max split baked
/// into [`crate::types::EffortLevel::anthropic_for_model`], and the Gemini
/// `high` cap applied ad hoc inside [`effective_effort_for_model`].
///
/// This is read as a **provider-declared capability fact** (design principle:
/// only capability-derived signals feed the router), never a routing
/// preference — it does not depend on connectivity, pins, or outcome data.
///
/// Resolution order: [`MODEL_EFFORT_CEILINGS_ENV`] override (longest matching
/// prefix wins) → alias resolution ([`resolve_model_alias`]) → per-provider
/// family rule. Dated/`@`/`[`-suffixed ids resolve through the same
/// alias/family-matching machinery every other capability lookup in this file
/// uses, so `gpt-5.6-sol-2026-07-09` and `gpt-5.6-terra@openai` get the same
/// ceiling as their bare family.
///
/// Per-provider rule:
/// - Anthropic (`claude*`): every Anthropic model accepts `max` verbatim
///   (Sonnet/Haiku's documented wire set is `low|medium|high|max`, simply
///   without `xhigh`); `Ultra` has no Anthropic wire value and always clamps
///   below `Max` (see `anthropic_for_model`), so `Max` is the true ceiling
///   regardless of the xhigh split.
/// - OpenAI (`gpt*`/`o3`/`o4`/`codex`, and OpenAI-compatible custom
///   providers): internal `Ultra` for Sol/Terra, internal `Max` for Luna, else
///   `Xhigh`. OpenAI's final wire enum tops out at `xhigh`, so `gpt_for_model`
///   projects both higher internal tiers before serialization.
/// - Google (`gemini*`): `High` — Gemini 3 tops out there.
/// - Everything else (xAI, Ollama, unknown/custom non-OpenAI): a conservative
///   `High` default. This is a capability *unknown*, not a capability grant —
///   it deliberately does NOT return `Ultra`/`Max` on the strength of
///   `effective_effort_for_model`'s pass-through wire tolerance, since that
///   would let an unrecognized model over-claim a Deep-tier promotion (see
///   `model_inventory::tiers_for_model` in the runtime crate).
#[must_use]
pub fn max_supported_effort(model: &str) -> crate::types::EffortLevel {
    use crate::types::{gpt_model_accepts_max, gpt_model_accepts_ultra, gpt_model_accepts_xhigh, EffortLevel};

    let canonical = resolve_model_alias(model);
    let lower = canonical.to_ascii_lowercase();

    if let Some(level) = env_effort_ceiling_override(&lower) {
        return level;
    }

    if lower.starts_with("claude") {
        return EffortLevel::Max;
    }

    let is_openai = openai_gpt_model_family(&lower).is_some()
        || is_openai_builtin_model_prefix(&lower)
        || catalog_entry_for_token(MODEL_REGISTRY, &lower)
            .is_some_and(|entry| entry.provider == ProviderKind::OpenAi);
    if is_openai {
        if gpt_model_accepts_ultra(&lower) {
            return EffortLevel::Ultra;
        }
        if gpt_model_accepts_max(&lower) {
            return EffortLevel::Max;
        }
        // `gpt_model_accepts_xhigh` is unconditionally true today, but kept as
        // a real predicate call (not inlined to `true`) so a future
        // family-specific carve-out degrades this ceiling gracefully too.
        if gpt_model_accepts_xhigh(&lower) {
            return EffortLevel::Xhigh;
        }
        return EffortLevel::High;
    }

    if lower.starts_with("gemini") {
        return EffortLevel::High;
    }

    EffortLevel::High
}

// ---------------------------------------------------------------------------
// Dynamic ultra-band resolver
//
// `Effort::Smart` (formerly `Ultracode`) no longer pins a single static
// wire level: it carries a FLOOR (`MessageRequest::effort`, always `Xhigh`
// in practice) plus a requested CEILING (`MessageRequest::effort_band_ceiling`).
// Each wire backend resolves the band to one concrete `EffortLevel` per
// request via `resolve_effort_band`, using the same per-request difficulty
// signals the GPT auto-effort ladder (`chatgpt_backend::dynamic_effort`)
// already computed — generalized here into the single shared source so no
// backend keeps its own copy of the keyword table.
// ---------------------------------------------------------------------------

/// Heavy-reasoning intent keywords (EN + KO), matched as substrings/stems of
/// the last user message. Single source of truth for per-request difficulty
/// classification: both the GPT auto-effort ladder
/// (`chatgpt_backend::dynamic_effort`) and [`band_difficulty_for_request`]
/// key off this ONE table — do not duplicate it elsewhere.
pub(crate) const HEAVY_INTENT: &[&str] = &[
    "analy",
    "debug",
    "refactor",
    "architect",
    "design",
    "review",
    "audit",
    "investigat",
    "optimiz",
    "compare",
    "benchmark",
    "분석",
    "디버",
    "리팩",
    "설계",
    "아키텍",
    "리뷰",
    "감사",
    "조사",
    "최적",
    "비교",
    "벤치",
];

/// Concatenate the plain-text blocks of one message (ignores tool calls,
/// results, and binary parts) — the per-request difficulty signal shared by
/// the GPT auto-effort ladder and the ultra-band resolver.
pub(crate) fn message_plain_text(message: &crate::types::InputMessage) -> String {
    use crate::types::InputContentBlock;
    let mut text = String::new();
    for block in &message.content {
        if let InputContentBlock::Text { text: value, .. } = block {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(value);
        }
    }
    text
}

/// The most recent user-authored message's plain text, or empty if none.
pub(crate) fn last_user_message_text(request: &crate::types::MessageRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(message_plain_text)
        .unwrap_or_default()
}

/// Total plain-text character count across every message — the large-context
/// difficulty proxy (~6k tokens at the 24_000-char threshold).
pub(crate) fn total_message_text_chars(request: &crate::types::MessageRequest) -> usize {
    use crate::types::InputContentBlock;
    request
        .messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            InputContentBlock::Text { text, .. } => Some(text.len()),
            _ => None,
        })
        .sum()
}

/// Per-request difficulty signals for the dynamic ultra band: a heavy-intent
/// keyword hit on the last user message, a large accumulated context
/// (>`24_000` chars, ~6k tokens), and a long single ask (>600 chars). Mirrors
/// (and is the sole source for) the GPT auto-effort ladder's existing triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BandDifficulty {
    pub heavy_intent: bool,
    pub large_context: bool,
    pub long_ask: bool,
}

impl BandDifficulty {
    /// Number of difficulty signals that fired (0..=3) — the escalation rung
    /// count [`resolve_effort_band`] steps by.
    #[must_use]
    pub fn signal_count(self) -> u8 {
        u8::from(self.heavy_intent) + u8::from(self.large_context) + u8::from(self.long_ask)
    }
}

/// Classify a request's difficulty signals from its message content — the
/// single source [`resolve_effort_band`] and the GPT auto-effort ladder both
/// read, so a heavy-intent Korean prompt escalates identically whether it
/// rides the dynamic ultra band or the legacy Auto ladder.
#[must_use]
pub fn band_difficulty_for_request(request: &crate::types::MessageRequest) -> BandDifficulty {
    let last_user = last_user_message_text(request);
    let total_chars = total_message_text_chars(request);
    let lower = last_user.to_lowercase();
    BandDifficulty {
        heavy_intent: HEAVY_INTENT.iter().any(|kw| lower.contains(kw)),
        large_context: total_chars > 24_000,
        long_ask: last_user.chars().count() > 600,
    }
}

/// Env var kill switch for the dynamic ultra band: `ZO_ULTRA_BAND=off`
/// restores the pre-band static-top behavior — [`resolve_effort_band`] always
/// returns the clamped ceiling regardless of per-request difficulty signals.
pub const ULTRA_BAND_ENV: &str = "ZO_ULTRA_BAND";

fn ultra_band_disabled() -> bool {
    std::env::var(ULTRA_BAND_ENV).is_ok_and(|value| value.trim().eq_ignore_ascii_case("off"))
}

/// Rank an [`crate::types::EffortLevel`] on the shared low→ultra scale.
/// Callers may compare named and budget-derived levels without changing their
/// independent budget, routing, or ceiling policies.
#[must_use]
pub const fn effort_rank(level: crate::types::EffortLevel) -> u8 {
    use crate::types::EffortLevel;
    match level {
        EffortLevel::Low => 0,
        EffortLevel::Medium => 1,
        EffortLevel::High => 2,
        EffortLevel::Xhigh => 3,
        EffortLevel::Max => 4,
        EffortLevel::Ultra => 5,
    }
}

fn effort_level_from_rank(rank: u8) -> crate::types::EffortLevel {
    use crate::types::EffortLevel;
    match rank {
        0 => EffortLevel::Low,
        1 => EffortLevel::Medium,
        2 => EffortLevel::High,
        3 => EffortLevel::Xhigh,
        4 => EffortLevel::Max,
        _ => EffortLevel::Ultra,
    }
}

/// Resolve a dynamic effort BAND — `[floor ..= min(ceiling, max_supported_effort(model))]`
/// — to the single concrete internal [`crate::types::EffortLevel`] this request
/// selects. Called once per request, BEFORE the backend's per-model
/// wire projection (`gpt_for_model`/`anthropic_for_model`/`gemini`) — the
/// resolved level then flows through that projection exactly as if it had
/// been the named `effort` all along.
///
/// Escalation rule: zero difficulty signals stay at `floor`; exactly one
/// signal steps up one rung (clamped to the ceiling); two or more signals
/// jump straight to the ceiling. In the one real caller (`Effort::Smart`,
/// floor=Xhigh, ceiling=Ultra) this is 0→xhigh, 1→max, 2+→ceiling (sol/terra
/// `ultra`, fable/luna `max`, gemini `high`).
///
/// `ZO_ULTRA_BAND=off` disables the dynamic behavior entirely: every call
/// returns the clamped ceiling, matching the pre-band static-top pin.
///
/// Properties (see the `resolve_effort_band_*` tests): the result never
/// exceeds `min(ceiling, max_supported_effort(model))`, never falls below
/// `floor` (a misconfigured `ceiling < floor` degenerates to always
/// returning the clamped ceiling, never below it), and is a deterministic,
/// pure function of its inputs (for a fixed `ZO_ULTRA_BAND`).
#[must_use]
pub fn resolve_effort_band(
    floor: crate::types::EffortLevel,
    ceiling: crate::types::EffortLevel,
    model: &str,
    difficulty: BandDifficulty,
) -> crate::types::EffortLevel {
    let model_ceiling_rank = effort_rank(max_supported_effort(model));
    let ceiling_rank = effort_rank(ceiling).min(model_ceiling_rank);
    let floor_rank = effort_rank(floor).min(ceiling_rank);

    if ultra_band_disabled() {
        return effort_level_from_rank(ceiling_rank);
    }

    let picked_rank = match difficulty.signal_count() {
        0 => floor_rank,
        1 => floor_rank.saturating_add(1),
        _ => ceiling_rank,
    };

    effort_level_from_rank(picked_rank.clamp(floor_rank, ceiling_rank))
}

/// Total input context window (in tokens) for `model`.
///
/// Anthropic windows are known from the Claude model ids/beta suffixes.
/// Providers that do not expose context-window fields through `/models` (notably
/// OpenAI's public model object) are resolved from the capability catalog loaded
/// via [`MODEL_CONTEXT_WINDOWS_ENV`] or the bundled docs-verified catalog. It is
/// **not** the per-response output cap — that is [`max_tokens_for_model`].
///
/// The lookup works off the canonical model id so both short aliases
/// (`opus`, `sonnet`, `gpt-5.5`) and fully-qualified ids
/// (`claude-opus-4-6`, `claude-opus-4-6[1m]`, `gpt-5.5-2026-04-23`) resolve
/// correctly.
#[must_use]
pub fn context_window_for_model(model: &str) -> u64 {
    model_capability_for_model(model)
        .context_window
        .unwrap_or(200_000)
}

fn context_window_for_canonical(raw_model: &str, canonical_lower: &str) -> u64 {
    // Current flagship Claude models with documented native 1M windows.
    // Sonnet 5 is the first Sonnet in Zo's built-in catalog with a documented
    // 1M default/max window; older Sonnet and Haiku variants stay capped below.
    if canonical_lower.contains("fable")
        || canonical_lower.contains("opus")
        || model_id_matches_prefix_segment(canonical_lower, "claude-sonnet-5")
    {
        return 1_000_000;
    }
    // Older Sonnet and Haiku top out at 258k in the environments Zo targets.
    // Apply that cap before the conservative Claude fallback so HUD/compaction
    // never advertises more context than these models can actually accept.
    if canonical_lower.contains("sonnet") || canonical_lower.contains("haiku") {
        return 258_000;
    }

    if canonical_lower.contains("claude") {
        return 200_000;
    }

    if let Some(window) = custom_provider_context_window(raw_model, canonical_lower) {
        return window;
    }

    // Provider/model capability metadata loaded from an external catalog.
    // OpenAI's public `/v1/models` object does not expose context-window
    // fields, so GPT-family limits must not be guessed in code; the built-in
    // catalog records docs-verified entries and `ZO_MODEL_CONTEXT_WINDOWS`
    // can override/add entries without recompilation.
    if let Some(window) = catalog_model_context_window(raw_model, canonical_lower) {
        return window;
    }

    // OpenAI-compatible fallback: conservative until a capability catalog entry
    // exists for the exact model. This avoids displaying fabricated precision
    // for new GPT/o/codex models.
    if canonical_lower.contains("gpt")
        || canonical_lower.contains("o3")
        || canonical_lower.contains("o4")
        || canonical_lower.contains("codex")
    {
        return 200_000;
    }

    // xAI Grok 3: 131k. Do not project that limit onto newer custom Grok ids;
    // use a custom-provider override or the conservative unknown-model fallback
    // until a docs-verified catalog entry exists.
    if model_id_matches_prefix_segment(canonical_lower, "grok-3") {
        return 131_072;
    }

    // Google Gemini: 1M
    if canonical_lower.contains("gemini") {
        return 1_000_000;
    }

    // Ollama: conservative default
    200_000
}

/// Read an environment variable, mapping an unset *or empty* value to `None`.
///
/// Shared by the Anthropic and OpenAI-compatible providers so the "empty
/// string means absent" credential-lookup rule has a single definition.
///
/// # Errors
/// Propagates a non-`NotPresent` [`crate::error::ApiError`] (e.g. a value that
/// is not valid Unicode).
pub(crate) fn read_env_non_empty(key: &str) -> Result<Option<String>, crate::error::ApiError> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Ok(Some(value)),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(crate::error::ApiError::from(error)),
    }
}

/// Append the *stable* prefix of a request's system blocks to `prefix`.
///
/// OpenAI prompt caching rewards stable content at the very beginning of the
/// prompt. Zo's system prompt is ordered identity/static/dynamic; the last
/// block is the per-session/project tail when a dynamic block exists. Keep the
/// routing key on the reusable prefix instead of fragmenting it by user/project
/// tail content. The request body still contains the full system text.
pub(crate) fn push_stable_system_prefix(prefix: &mut String, request: &crate::types::MessageRequest) {
    let Some(system) = &request.system else {
        return;
    };
    let stable_len = if system.len() > 1 {
        system.len() - 1
    } else {
        system.len()
    };
    for block in system.iter().take(stable_len) {
        let crate::types::SystemBlock::Text { text, .. } = block;
        prefix.push_str(text);
        prefix.push('\n');
    }
}

/// Append the conversation-stream discriminator — the text of the first user
/// message — to `prefix`. Distinct agents (main conversation, each fanout
/// spawn) open with distinct first messages, so this splits them into
/// distinct cache keys even when they share a session id; appending later
/// turns never changes it. Non-text opening blocks (images, tool results)
/// are skipped: text is present in every real opening prompt and is stable
/// across serialization details.
pub(crate) fn push_conversation_stream_discriminator(
    prefix: &mut String,
    request: &crate::types::MessageRequest,
) {
    let Some(first_user) = request
        .messages
        .iter()
        .find(|message| message.role == "user")
    else {
        return;
    };
    for block in &first_user.content {
        if let crate::types::InputContentBlock::Text { text, .. } = block {
            prefix.push_str(text);
        }
    }
    prefix.push('\n');
}

/// Derive the deterministic OpenAI `prompt_cache_key` for a request.
///
/// The routing key hashes the (wire) `model`, a non-empty session id, the
/// conversation-stream discriminator (the first user message's text), the
/// stable system prefix, and the tool schemas. `model` is passed explicitly
/// because the ChatGPT backend keys on the resolved wire model id rather than
/// `request.model`; the OpenAI-compatible path passes `&request.model`.
///
/// The stream discriminator is what keeps concurrent agents from sharing one
/// key: a fanout spawn re-stamps its requests with the parent's session id, so
/// under a session-only key every spawn's divergent transcript competed for
/// the same provider cache shard and evicted each other's prefixes (observed
/// live 07-20: sol cache reads pinned at the ~12k shared system prefix across
/// 400+ requests while each interleaved agent re-billed its full 100k+
/// history). The first user message is unique per conversation stream (the
/// user's opening prompt / the spawn's task prompt), never changes as turns
/// append, and when compaction rewrites it the key rolls over exactly when
/// the cached prefix is invalidated anyway.
pub(crate) fn prompt_cache_key(
    model: &str,
    request: &crate::types::MessageRequest,
    session_id: &str,
) -> String {
    use sha2::Digest;
    let mut prefix = String::new();
    prefix.push_str(model);
    prefix.push('\n');
    if !session_id.is_empty() {
        prefix.push_str(session_id);
        prefix.push('\n');
    }
    push_conversation_stream_discriminator(&mut prefix, request);
    push_stable_system_prefix(&mut prefix, request);
    if let Some(tools) = &request.tools {
        for tool in tools {
            prefix.push_str(&tool.name);
            prefix.push('\n');
            if let Some(description) = &tool.description {
                prefix.push_str(description);
                prefix.push('\n');
            }
            prefix.push_str(&tool.input_schema.to_string());
            prefix.push('\n');
        }
    }
    let digest = sha2::Sha256::digest(prefix.as_bytes());
    format!("zo-{digest:x}")[..64].to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        BandDifficulty, EXPERIMENTAL_PROVIDERS_ENV, MODEL_CLASSES_ENV, MODEL_CONTEXT_WINDOWS_ENV,
        MODEL_EFFORT_CEILINGS_ENV,
        ModelClass, NON_CLAUDE_ADAPTERS_ENV,
        PromptCacheStrategy, ProviderKind, STREAM_IDLE_TIMEOUT_ENV, STREAM_IDLE_TIMEOUT_MS,
        ULTRA_BAND_ENV,
        apply_non_anthropic_identity, band_difficulty_for_request, context_window_for_model,
        declared_model_class, detect_provider_kind,
        effective_effort_for_model, effort_budget_with_floor, effort_level_for_budget, effort_rank,
        ProviderCatalogEntry, catalog_entry_for_token, catalog_provider_for_canonical,
        explicit_non_claude_provider_kind, fit_hint_for_model, is_openai_builtin_model_prefix,
        levenshtein, maker_for_provider, max_supported_effort,
        max_tokens_for_model, model_capability_for_model, model_supports_xhigh,
        non_claude_adapters_enabled, openai_gpt_model_family, openai_prompt_cache_retention,
        provider_catalog, resolve_effort_band, resolve_model_alias, resolve_registered_model_alias,
        shared_prefix_len, stream_idle_timeout, uses_adaptive_thinking, version_tokens,
    };

    #[test]
    fn effort_rank_exhaustively_orders_the_six_levels() {
        use crate::types::EffortLevel::{High, Low, Max, Medium, Ultra, Xhigh};

        const ULTRA_RANK: u8 = effort_rank(Ultra);
        let levels = [Low, Medium, High, Xhigh, Max, Ultra];
        let ranks = levels.map(effort_rank);
        assert_eq!(ranks, [0, 1, 2, 3, 4, 5]);
        assert!(ranks.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(ULTRA_RANK, 5);
    }

    #[test]
    fn maker_for_provider_names_first_party_non_anthropic_only() {
        // First-party non-Anthropic providers name their real maker …
        assert_eq!(maker_for_provider(ProviderKind::OpenAi), Some("OpenAI"));
        assert_eq!(maker_for_provider(ProviderKind::Xai), Some("xAI"));
        assert_eq!(maker_for_provider(ProviderKind::Google), Some("Google"));
        // … while Anthropic (never rewritten) and Ollama (unknown local model)
        // assert no maker.
        assert_eq!(maker_for_provider(ProviderKind::Anthropic), None);
        assert_eq!(maker_for_provider(ProviderKind::Ollama), None);
    }

    #[test]
    fn apply_non_anthropic_identity_overrides_claude_and_names_maker() {
        let system = "You are Claude Code, Anthropic's official CLI for Claude.\nDo work.";
        let out = apply_non_anthropic_identity(system, "gpt-5.5", Some("OpenAI"));
        // The original Claude-authored body is preserved …
        assert!(out.contains("Do work."), "body preserved: {out}");
        // … under an explicit identity override naming the model + maker.
        assert!(out.contains("You are gpt-5.5, a large language model made by OpenAI"));
        assert!(out.contains("provider-neutral response style contract"));
        assert!(out.contains("do not claim to be Claude or to be made by Anthropic"));
        assert!(out.ends_with(system), "override is a prefix of the body");
    }

    #[test]
    fn apply_non_anthropic_identity_uses_neutral_wording_without_maker() {
        // A custom/self-hosted model (no known maker) is still identity-corrected
        // but must not be mislabeled as made by a specific lab.
        let out = apply_non_anthropic_identity("manual", "deepseek-chat", None);
        assert!(out.contains("You are deepseek-chat, a large language model, operating"));
        assert!(
            !out.contains("made by OpenAI")
                && !out.contains("made by xAI")
                && !out.contains("made by Google"),
            "no false maker attribution: {out}"
        );
        assert!(out.contains("do not claim to be Claude"));
    }

    #[test]
    fn apply_non_anthropic_identity_passes_empty_system_through() {
        // Empty system → unchanged empty output (no stray identity block); this
        // is the short-circuit that also protects every backend's empty path.
        assert_eq!(
            apply_non_anthropic_identity("", "gpt-5.5", Some("OpenAI")),
            ""
        );
    }

    #[test]
    fn stream_idle_timeout_defaults_and_env_override() {
        // Env access is process-global, and the Anthropic mid-stream-restart
        // tests also drive STREAM_IDLE_TIMEOUT_ENV, so serialise on the shared
        // lock to avoid a cross-test read of a half-written value.
        let _guard = crate::test_env_lock();
        let key = STREAM_IDLE_TIMEOUT_ENV;
        let restore = std::env::var(key).ok();

        std::env::remove_var(key);
        assert_eq!(
            stream_idle_timeout(),
            Some(std::time::Duration::from_millis(STREAM_IDLE_TIMEOUT_MS)),
            "default budget applies when unset"
        );

        std::env::set_var(key, "1500");
        assert_eq!(
            stream_idle_timeout(),
            Some(std::time::Duration::from_millis(1_500)),
            "valid override is honoured"
        );

        std::env::set_var(key, "0");
        assert_eq!(
            stream_idle_timeout(),
            None,
            "zero disables the idle timeout"
        );

        std::env::set_var(key, "not-a-number");
        assert_eq!(
            stream_idle_timeout(),
            Some(std::time::Duration::from_millis(STREAM_IDLE_TIMEOUT_MS)),
            "garbage falls back to the default"
        );

        match restore {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

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
    fn resolves_grok_aliases_only_when_adapter_gate_is_enabled() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        assert_eq!(resolve_model_alias("grok"), "grok");

        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        assert_eq!(resolve_model_alias("grok"), "grok-3");
    }

    /// The `fable5` trap: a near-miss of a real Anthropic alias must snap to the
    /// alias's canonical id, NOT pass through verbatim to Anthropic as a bogus
    /// `404 not_found` model. Anthropic is always provider-enabled, so this
    /// holds regardless of the non-Claude adapter gate.
    #[test]
    fn recovers_near_miss_of_a_builtin_alias() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        // Single fat-finger extra char and a transposition both recover.
        assert_eq!(resolve_model_alias("fable5"), "claude-fable-5");
        assert_eq!(resolve_model_alias("fabel"), "claude-fable-5");
        assert_eq!(resolve_model_alias("opuss"), "claude-opus-4-8");
    }

    /// A GPT/Gemini near-miss recovers only when the target provider is enabled,
    /// exactly like an exact-alias hit — so a typo under a Claude-only setup
    /// passes through as "unsupported" rather than silently dialing OpenAI.
    #[test]
    fn near_miss_recovery_honors_the_provider_gate() {
        let _lock = crate::test_env_lock();

        let _off = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        // Gate off: no snap to a disabled provider's alias — passthrough.
        assert_eq!(resolve_model_alias("gemini-flsh"), "gemini-flsh");

        let _on = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        // Gate on: `gemini-flsh` is a unique word-typo of `gemini-flash`.
        assert_eq!(resolve_model_alias("gemini-flsh"), "gemini-3.5-flash");
    }

    /// The critical conservatism boundary: a typo may differ from its alias only
    /// in the WORD, never in an explicit version/qualifier number. Silently
    /// running an older/cheaper sibling is worse than a clean "unsupported"
    /// error, so a version bump or suffix change must pass through untouched
    /// even when the provider is enabled and the edit distance is tiny.
    #[test]
    fn near_miss_recovery_never_reroutes_across_versions() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));

        // Anthropic is always enabled — these are the worst always-on reroutes.
        assert_eq!(resolve_model_alias("opus-5"), "opus-5");
        assert_eq!(resolve_model_alias("sonnet-4"), "sonnet-4");
        // Non-Claude version bumps / suffix changes with the gate ON.
        assert_eq!(resolve_model_alias("grok-4"), "grok-4");
        assert_eq!(resolve_model_alias("gpt-5.5-mini"), "gpt-5.5-mini");
        assert_eq!(resolve_model_alias("gpt-5.6o"), "gpt-5.6o");
        assert_eq!(resolve_model_alias("gemini-3.7-flash"), "gemini-3.7-flash");
    }

    /// The displayed short form of a model resolves to its canonical id. A live
    /// session spawned agents with `model: opus-4.8` (the version spelled with a
    /// dot, exactly as the HUD prints it) and the provider 404'd it verbatim
    /// (`not_found_error: model: opus-4.8`), even though `claude-opus-4-8` is the
    /// same model. This is version-*preserving* reformatting, distinct from the
    /// cross-version snaps `near_miss_recovery_never_reroutes_across_versions`
    /// forbids: only the separator (dot vs hyphen) and the `claude-` prefix
    /// differ, and the result must land on a real canonical id or pass through.
    #[test]
    fn resolves_dotted_and_bare_short_form_to_canonical() {
        let _lock = crate::test_env_lock();
        // Anthropic is always enabled, so these hold regardless of the gate.
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);

        // The reported 404: dotted version short form → canonical.
        assert_eq!(resolve_model_alias("opus-4.8"), "claude-opus-4-8");
        assert_eq!(resolve_model_alias("claude-opus-4.8"), "claude-opus-4-8");
        // Bare (no-minor / hyphen) short forms of other Claude families.
        assert_eq!(resolve_model_alias("fable-5"), "claude-fable-5");
        assert_eq!(resolve_model_alias("sonnet-5"), "claude-sonnet-5");

        // Conservatism preserved: a genuinely different version whose
        // normalization matches no canonical still passes through untouched, and
        // providers whose canonical ids keep dots are unaffected.
        assert_eq!(resolve_model_alias("opus-5"), "opus-5");
        assert_eq!(resolve_model_alias("gpt-5.5-mini"), "gpt-5.5-mini");
        assert_eq!(resolve_model_alias("gemini-3.7-flash"), "gemini-3.7-flash");
    }

    #[test]
    fn registered_spawn_alias_snaps_only_unknown_claude_family_versions() {
        let _lock = crate::test_env_lock();
        super::refresh_custom_providers_from_json("[]").expect("clear custom providers");
        for input in [
            "opus-4.6",
            "claude-opus-4.6",
            "claude-opus-4-6",
            "opus-5",
        ] {
            assert_eq!(
                resolve_registered_model_alias(input),
                "claude-opus-4-8",
                "input {input}"
            );
        }
        assert_eq!(
            resolve_registered_model_alias("opus"),
            "claude-opus-4-8"
        );
        assert_eq!(
            resolve_registered_model_alias("opus-4.8"),
            "claude-opus-4-8"
        );
        assert_eq!(
            resolve_registered_model_alias("opus[1m]"),
            "claude-opus-4-8"
        );
        assert_eq!(
            resolve_registered_model_alias("opus-4.6[1m]"),
            "claude-opus-4-8"
        );
        assert_eq!(
            resolve_registered_model_alias("sonnet-4.5"),
            "claude-sonnet-5"
        );
        assert_eq!(
            resolve_registered_model_alias("claude-3-5-haiku"),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(
            resolve_registered_model_alias("fable-4"),
            "claude-fable-5"
        );

        for input in [
            "gpt-5.6-sol",
            "gpt-5.5",
            "gemini-3-flash",
            "grok-3",
            "openai/gpt-x",
        ] {
            assert_eq!(resolve_registered_model_alias(input), input);
        }

        super::refresh_custom_providers_from_json(
            r#"[{"name":"custom claude","base_url":"https://example.invalid/v1",
                "models":["claude-opus-4.6"],"requires_auth":false}]"#,
        )
        .expect("load custom provider");
        assert_eq!(
            resolve_registered_model_alias("claude-opus-4.6"),
            "claude-opus-4.6",
            "operator-declared custom ids must not be snapped to Anthropic"
        );
        super::refresh_custom_providers_from_json("[]").expect("clear custom providers");
    }

    /// Conservatism guards: an ambiguous or too-far input must NEVER be
    /// rerouted to a distinct real model — passthrough (then the caller can
    /// surface a clean error) is safer than snapping to the wrong model.
    #[test]
    fn near_miss_recovery_refuses_ambiguous_or_distant_inputs() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));

        // Bare `gpt-5.6`은 한때 "형제로 스냅 금지" passthrough였으나, 사용자
        // 확정(2026-07-11)으로 terra를 가리키는 **명시 레지스트리 별칭**이 됐다
        // — near-miss 퍼지 스냅이 아니라 exact-alias 경로라 보수성 원칙과
        // 충돌하지 않는다. 서빙 티어는 별칭에 박지 않는다(세션 /fast 상태가
        // 결정) — bare id로 해석된다.
        assert_eq!(resolve_model_alias("gpt-5.6"), "gpt-5.6-terra");
        // No shared 3-char prefix with any alias → no snap.
        assert_eq!(resolve_model_alias("op"), "op");
        assert_eq!(resolve_model_alias("xyzzy"), "xyzzy");
        // A distinct real model is NOT snapped onto its sibling.
        assert_eq!(resolve_model_alias("gpt-5.6-sol"), "gpt-5.6-sol");
    }

    /// A fully-qualified canonical id already in the registry must resolve to
    /// itself, never get snapped to a shorter alias by the near-miss pass.
    #[test]
    fn near_miss_recovery_never_snaps_a_known_canonical_id() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        assert_eq!(resolve_model_alias("claude-fable-5"), "claude-fable-5");
        assert_eq!(resolve_model_alias("claude-opus-4-8"), "claude-opus-4-8");
    }

    #[test]
    fn levenshtein_and_prefix_helpers_are_correct() {
        assert_eq!(levenshtein("fable", "fable"), 0);
        assert_eq!(levenshtein("fable5", "fable"), 1);
        assert_eq!(levenshtein("fabel", "fable"), 2);
        assert_eq!(levenshtein("", "opus"), 4);
        // Multibyte input must not panic and counts by char.
        assert_eq!(levenshtein("한글", "opus"), 4);
        assert_eq!(shared_prefix_len("gpt-5.6x", "gpt-5.6-sol"), 7);
        assert_eq!(shared_prefix_len("op", "opus"), 2);
    }

    #[test]
    fn version_tokens_splits_only_field_boundary_numbers() {
        // A digit glued to a letter is a fat-finger, not a version field.
        assert!(version_tokens("fable5").is_empty());
        assert!(version_tokens("fable").is_empty());
        assert!(version_tokens("opus").is_empty());
        // Numbers at a `-`/`.`/start boundary ARE version fields.
        assert_eq!(version_tokens("opus-5"), vec!["5".to_string()]);
        assert_eq!(version_tokens("gpt-5.6-sol"), vec!["5.6".to_string()]);
        assert_eq!(version_tokens("gpt-5.6o"), vec!["5.6o".to_string()]);
        assert_eq!(version_tokens("gpt-5.5-mini"), vec!["5.5".to_string()]);
        assert_eq!(version_tokens("gemini-3.7-flash"), vec!["3.7".to_string()]);
    }

    #[test]
    fn detects_provider_from_model_name_only_when_adapter_gate_is_enabled() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        assert_eq!(detect_provider_kind("grok"), ProviderKind::Anthropic);

        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        assert_eq!(detect_provider_kind("grok"), ProviderKind::Xai);
        assert_eq!(
            detect_provider_kind("claude-sonnet-4-6"),
            ProviderKind::Anthropic
        );
    }

    #[test]
    fn catalog_provider_lookup_supports_non_prefix_openai_canonicals() {
        const FUTURE_CATALOG: &[ProviderCatalogEntry] = &[
            ProviderCatalogEntry::new(
                "future-openai",
                "future-openai-2026-01-01",
                ProviderKind::OpenAi,
            ),
            ProviderCatalogEntry::new("future-google", "future-google-1", ProviderKind::Google),
        ];

        assert_eq!(
            catalog_provider_for_canonical(FUTURE_CATALOG, "future-openai-2026-01-01"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(
            catalog_provider_for_canonical(FUTURE_CATALOG, "FUTURE-OPENAI-2026-01-01"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(
            catalog_entry_for_token(FUTURE_CATALOG, "future-openai")
                .map(|entry| entry.provider),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(
            catalog_entry_for_token(FUTURE_CATALOG, "future-openai-2026-01-01")
                .map(|entry| entry.provider),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(catalog_provider_for_canonical(FUTURE_CATALOG, "unknown"), None);
    }

    #[test]
    fn openai_catalog_tokens_are_explicit_even_without_a_gpt_prefix() {
        const FUTURE_CATALOG: &[ProviderCatalogEntry] = &[ProviderCatalogEntry::new(
            "future-openai",
            "future-openai-2026-01-01",
            ProviderKind::OpenAi,
        )];
        let entry = catalog_entry_for_token(FUTURE_CATALOG, "future-openai")
            .expect("future OpenAI alias found");

        assert_eq!(entry.provider, ProviderKind::OpenAi);
        assert!(is_openai_builtin_model_prefix("o1-preview"));
        assert_eq!(explicit_non_claude_provider_kind("o1-preview"), Some(ProviderKind::OpenAi));
    }

    #[test]
    fn detects_openai_provider_after_alias_resolution() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        let _key = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _base = EnvVarGuard::set("OPENAI_BASE_URL", Some("http://localhost:8080/v1"));

        // gpt-5.5 계열은 카탈로그 퇴역(2026-07-11) — 별칭 없이 passthrough.
        // 레거시 정확 id의 provider 감지는 아래에서 계속 보장한다.
        assert_eq!(resolve_model_alias("gpt-5.5"), "gpt-5.5");
        assert_eq!(resolve_model_alias("gpt-5.6"), "gpt-5.6-terra");
        assert_eq!(resolve_model_alias("gpt-5.6-luna"), "gpt-5.6-luna");
        assert_eq!(
            resolve_model_alias("gpt-5.3-codex-spark"),
            "gpt-5.3-codex-spark"
        );
        assert_eq!(
            detect_provider_kind("gpt-5.5-2026-04-23"),
            ProviderKind::OpenAi
        );
    }

    #[test]
    fn resolves_current_gemini_code_assist_aliases_when_google_is_enabled() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));

        assert_eq!(resolve_model_alias("gemini-flash"), "gemini-3.5-flash");
        assert_eq!(resolve_model_alias("gemini-pro"), "gemini-3.1-pro-preview");
        assert_eq!(
            resolve_model_alias("gemini-flash-lite"),
            "gemini-3.1-flash-lite"
        );
        assert_eq!(
            resolve_model_alias("gemini-3.5-pro"),
            "gemini-3.1-pro-preview"
        );
        assert_eq!(
            detect_provider_kind("gemini-3.1-pro-preview"),
            ProviderKind::Google
        );
    }

    #[test]
    fn adapter_gate_parses_common_falsey_values() {
        let _lock = crate::test_env_lock();
        let _legacy_unset = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, None);
        let _unset = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        assert!(!non_claude_adapters_enabled());

        let _legacy_false = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, Some("false"));
        let _false = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("false"));
        assert!(!non_claude_adapters_enabled());

        let _legacy_true = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, Some("1"));
        let _true = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        assert!(non_claude_adapters_enabled());
    }

    #[test]
    fn custom_base_url_enables_adapters_like_an_api_key() {
        let _lock = crate::test_env_lock();
        let _f1 = EnvVarGuard::set(EXPERIMENTAL_PROVIDERS_ENV, None);
        let _f2 = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, None);
        // Clear every provider credential/endpoint so the default is clean.
        let _k1 = EnvVarGuard::set("OPENAI_API_KEY", None);
        let _k2 = EnvVarGuard::set("GOOGLE_API_KEY", None);
        let _k3 = EnvVarGuard::set("XAI_API_KEY", None);
        let _k4 = EnvVarGuard::set("OLLAMA_API_KEY", None);
        let _b2 = EnvVarGuard::set("GOOGLE_BASE_URL", None);
        let _b3 = EnvVarGuard::set("XAI_BASE_URL", None);
        let _b4 = EnvVarGuard::set("OLLAMA_BASE_URL", None);

        // Nothing configured → Anthropic-only.
        let base_off = EnvVarGuard::set("OPENAI_BASE_URL", None);
        assert!(!non_claude_adapters_enabled());
        drop(base_off);

        // A self-hosted OpenAI-compatible endpoint (base URL, no key) now
        // auto-enables the adapters — matching how OLLAMA_BASE_URL already did.
        let _base_on = EnvVarGuard::set("OPENAI_BASE_URL", Some("http://localhost:8080/v1"));
        assert!(non_claude_adapters_enabled());
    }

    #[test]
    fn explicit_non_claude_provider_detection_works_without_gate() {
        assert_eq!(
            explicit_non_claude_provider_kind("grok-3"),
            Some(ProviderKind::Xai)
        );
        assert_eq!(
            explicit_non_claude_provider_kind("gpt-5.6-sol"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(explicit_non_claude_provider_kind("claude-sonnet-4-6"), None);
    }

    #[test]
    fn max_tokens_follow_docs_for_opus_fable_and_safe_fallbacks() {
        // Current Anthropic docs list 128K max output for Fable/Mythos and
        // Opus 4.6+ (including the `opus` alias -> Opus 4.8). Deprecated Opus
        // 4.1/4.0 variants stay at their documented 32K. Non-Anthropic models
        // now honour the docs-verified catalog cap; a model with no catalog entry
        // (or no max_output_tokens) keeps the safe 64K default.
        assert_eq!(max_tokens_for_model("opus"), 128_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-8"), 128_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-8-20260611"), 128_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-7"), 128_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-6[1m]"), 128_000);
        assert_eq!(max_tokens_for_model("fable"), 128_000);
        assert_eq!(max_tokens_for_model("sonnet"), 128_000);
        assert_eq!(max_tokens_for_model("claude-sonnet-5"), 128_000);
        assert_eq!(max_tokens_for_model("claude-sonnet-5-20260625"), 128_000);
        assert_eq!(max_tokens_for_model("claude-mythos-5"), 128_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-1"), 32_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-20250514"), 32_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-5"), 64_000);
        assert_eq!(max_tokens_for_model("claude-opus-4-10"), 64_000);
        assert_eq!(max_tokens_for_model("claude-sonnet-4-6"), 64_000);
        assert_eq!(max_tokens_for_model("grok-3"), 64_000);
        // Non-Anthropic models now honour the docs-verified catalog
        // max_output_tokens (previously all silently 64k because the field was
        // never deserialized).
        assert_eq!(max_tokens_for_model("gpt-5.6-sol"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.6-sol-2026-07-09"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.6-terra"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.6-terra@openai"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.6-luna"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.6-luna[fast]"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.5"), 128_000);
        assert_eq!(max_tokens_for_model("gpt-5.5-fast"), 128_000);
        assert_eq!(max_tokens_for_model("deepseek-v4-pro"), 384_000);
        // A catalog entry without max_output_tokens keeps the safe default.
        assert_eq!(max_tokens_for_model("gpt-5.3-codex-spark"), 64_000);
        assert_eq!(max_tokens_for_model("gpt-6-preview"), 64_000);
    }

    #[test]
    fn effort_floor_only_raises_never_lowers() {
        // No floor → configured budget passes through unchanged.
        assert_eq!(effort_budget_with_floor(Some(10_000), None), Some(10_000));
        assert_eq!(effort_budget_with_floor(None, None), None);
        // Floor below configured → keep the higher configured (never lower).
        assert_eq!(
            effort_budget_with_floor(Some(24_000), Some(16_000)),
            Some(24_000)
        );
        // Floor above configured → raise to the floor.
        assert_eq!(
            effort_budget_with_floor(Some(0), Some(16_000)),
            Some(16_000)
        );
        assert_eq!(
            effort_budget_with_floor(Some(10_000), Some(16_000)),
            Some(16_000)
        );
        // No configured thinking, floor set → escalate from nothing to the floor.
        assert_eq!(effort_budget_with_floor(None, Some(16_000)), Some(16_000));
        // A zero floor is inert.
        assert_eq!(
            effort_budget_with_floor(Some(10_000), Some(0)),
            Some(10_000)
        );
        assert_eq!(effort_budget_with_floor(None, Some(0)), None);
    }

    #[test]
    fn adaptive_thinking_gates_on_model_generation() {
        // Opus 4.6+/Fable use adaptive (output_config.effort); 4.5 and earlier
        // keep legacy budget thinking.
        assert!(uses_adaptive_thinking("claude-opus-4-8"));
        assert!(uses_adaptive_thinking("claude-opus-4-6"));
        assert!(uses_adaptive_thinking("fable"));
        assert!(uses_adaptive_thinking("claude-fable-5"));
        assert!(uses_adaptive_thinking("sonnet"));
        assert!(uses_adaptive_thinking("claude-sonnet-5"));
        assert!(uses_adaptive_thinking("claude-sonnet-4-6"));
        assert!(!uses_adaptive_thinking("claude-opus-4-5"));
        assert!(!uses_adaptive_thinking("claude-sonnet-4-5"));
        assert!(!uses_adaptive_thinking("claude-3-5-sonnet"));
        // Non-Anthropic models never use Anthropic adaptive thinking.
        assert!(!uses_adaptive_thinking("gpt-5.5"));
        assert!(!uses_adaptive_thinking("grok-3"));
    }

    #[test]
    fn budget_maps_to_effort_levels_monotonically() {
        use crate::types::EffortLevel;
        // Thresholds invert the CLI Effort preset budgets (1024/4096/10000/
        // 16000/24000/32000); each preset maps back to its own level, and the
        // boundaries are the midpoints between consecutive presets.
        assert_eq!(effort_level_for_budget(0), EffortLevel::Low);
        assert_eq!(effort_level_for_budget(1_024), EffortLevel::Low); // Low preset
        assert_eq!(effort_level_for_budget(2_560), EffortLevel::Low);
        assert_eq!(effort_level_for_budget(2_561), EffortLevel::Medium);
        assert_eq!(effort_level_for_budget(4_096), EffortLevel::Medium); // Medium preset
        assert_eq!(effort_level_for_budget(7_048), EffortLevel::Medium);
        assert_eq!(effort_level_for_budget(7_049), EffortLevel::High);
        assert_eq!(effort_level_for_budget(10_000), EffortLevel::High); // High preset
        assert_eq!(effort_level_for_budget(13_000), EffortLevel::High);
        assert_eq!(effort_level_for_budget(13_001), EffortLevel::Xhigh);
        assert_eq!(effort_level_for_budget(16_000), EffortLevel::Xhigh); // Xhigh preset
        assert_eq!(effort_level_for_budget(20_000), EffortLevel::Xhigh);
        assert_eq!(effort_level_for_budget(20_001), EffortLevel::Max);
        assert_eq!(effort_level_for_budget(24_000), EffortLevel::Max); // Max preset
        assert_eq!(effort_level_for_budget(28_000), EffortLevel::Max); // Smart preset (formerly Ultracode)
        assert_eq!(effort_level_for_budget(32_000), EffortLevel::Max);
        assert_eq!(effort_level_for_budget(200_000), EffortLevel::Max);
    }

    #[test]
    fn effective_effort_for_model_projects_every_provider_ceiling() {
        use crate::types::EffortLevel::{High, Low, Max, Medium, Ultra, Xhigh};
        // (model, requested, effective). Covers the full low..max scale across
        // every provider family so a UI can read the *actual* tier without
        // branching on provider.
        let cases = [
            // Anthropic: Opus/Fable keep the full scale; Sonnet/Haiku clamp
            // xhigh -> high but KEEP max (max is in their supported set).
            ("opus", Xhigh, Xhigh),
            ("opus", Max, Max),
            ("claude-opus-4-8", Xhigh, Xhigh),
            ("claude-fable-5", Xhigh, Xhigh),
            ("sonnet", Xhigh, High),
            ("sonnet", Max, Max),
            ("claude-sonnet-5", Xhigh, High),
            ("claude-sonnet-4-6", Xhigh, High),
            ("haiku", Xhigh, High),
            ("claude-haiku-4-5-20251001", Xhigh, High),
            // OpenAI: GPT-5.6 keeps Max; older GPT families keep the historical
            // Max -> Xhigh clamp. `/fast` is service priority, not an effort
            // ceiling.
            ("gpt-5.6-sol", Xhigh, Xhigh),
            ("gpt-5.6-sol", Max, Max),
            ("gpt-5.6-terra", Max, Max),
            ("gpt-5.6-luna", Max, Max),
            ("gpt-5.5", Xhigh, Xhigh),
            ("gpt-5.5", Max, Xhigh),
            ("gpt-5.5-fast", Xhigh, Xhigh),
            ("gpt-5.5-fast", Max, Xhigh),
            ("gpt-5.5-2026-04-23-fast", Max, Xhigh),
            ("gpt-5.3-codex-spark", Xhigh, Xhigh),
            ("gpt-5.3-codex-spark", Max, Xhigh),
            ("gpt-5.6-sol", Ultra, Ultra),
            ("gpt-5.6-sol-2026-07-09", Ultra, Ultra),
            ("gpt-5.6-terra@openai", Ultra, Ultra),
            ("gpt-5.6-terra[fast]", Ultra, Ultra),
            ("gpt-5.6-luna", Max, Max),
            ("gpt-5.6-luna", Ultra, Xhigh),
            ("gpt-5.5", Ultra, Xhigh),
            // Google Gemini 3: tops out at high.
            ("gemini-3.5-flash", Xhigh, High),
            ("gemini-3.5-flash", Max, High),
            ("gemini-3.5-flash", Ultra, High),
            ("gemini-pro", Max, High),
            ("gemini-pro", High, High),
            // Custom / unknown / xAI: conservative pass-through (never silently
            // downgraded off an ambient OPENAI_API_KEY — the BUG-R14 trap).
            ("deepseek-chat", Xhigh, Xhigh),
            ("deepseek-chat", Max, Max),
            ("my-self-hosted-model", Xhigh, Xhigh),
            // Lower tiers are identity on every provider.
            ("sonnet", Low, Low),
            ("gpt-5.5-fast", Medium, Medium),
            ("gemini-3.5-flash", High, High),
        ];
        for (model, requested, effective) in cases {
            assert_eq!(
                effective_effort_for_model(requested, model),
                effective,
                "model={model} requested={requested:?}"
            );
        }
    }

    #[test]
    fn model_supports_xhigh_truth_table() {
        // Mirrors the per-provider ceilings: Opus/Fable and GPT (including fast)
        // accept xhigh; Sonnet/Haiku and all Gemini do not; unknown/custom pass
        // through as "supported" (we don't claim to know their ceiling).
        assert!(model_supports_xhigh("opus"));
        assert!(model_supports_xhigh("claude-fable-5"));
        assert!(model_supports_xhigh("gpt-5.5"));
        assert!(!model_supports_xhigh("sonnet"));
        assert!(!model_supports_xhigh("claude-haiku-4-5-20251001"));
        assert!(model_supports_xhigh("gpt-5.5-fast"));
        assert!(model_supports_xhigh("gpt-5.3-codex-spark"));
        assert!(!model_supports_xhigh("gemini-3.5-flash"));
        assert!(!model_supports_xhigh("gemini-pro"));
        assert!(model_supports_xhigh("deepseek-chat"));
    }

    #[test]
    fn max_supported_effort_matches_every_provider_ceiling() {
        use crate::types::EffortLevel::{High, Max, Ultra, Xhigh};
        let cases = [
            // Anthropic: every model's true ceiling is Max regardless of the
            // xhigh split (Sonnet/Haiku accept max, just not xhigh).
            ("opus", Max),
            ("claude-opus-4-8", Max),
            ("claude-fable-5", Max),
            ("sonnet", Max),
            ("claude-sonnet-5", Max),
            ("haiku", Max),
            ("claude-haiku-4-5-20251001", Max),
            // OpenAI internal selection: Sol/Terra reach Ultra; Luna tops out
            // at Max; every other GPT family tops out at Xhigh.
            ("gpt-5.6-sol", Ultra),
            ("gpt-5.6-terra", Ultra),
            ("gpt-5.6-luna", Max),
            ("gpt-5.5", Xhigh),
            ("gpt-5.5-fast", Xhigh),
            ("gpt-5.3-codex-spark", Xhigh),
            // Dated / explicit-provider / service-tier suffixed ids inherit
            // their bare family's ceiling.
            ("gpt-5.6-sol-2026-07-09", Ultra),
            ("gpt-5.6-terra@openai", Ultra),
            ("gpt-5.6-terra[fast]", Ultra),
            ("gpt-5.6-luna-2026-07-09", Max),
            // Google Gemini 3 tops out at High.
            ("gemini-3.5-flash", High),
            ("gemini-pro", High),
            // Custom / unknown / xAI: conservative default — never Ultra/Max on
            // the strength of ambient pass-through tolerance.
            ("deepseek-chat", High),
            ("my-self-hosted-model", High),
        ];
        for (model, expected) in cases {
            assert_eq!(max_supported_effort(model), expected, "model={model}");
        }
    }

    #[test]
    fn max_supported_effort_env_override_wins_and_prefers_longest_prefix() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(
            MODEL_EFFORT_CEILINGS_ENV,
            Some(r#"{"gpt-5.7": "max", "gpt-5.7-nova": "ultra"}"#),
        );
        // Longest matching prefix wins: the specific `gpt-5.7-nova` override
        // beats the broader `gpt-5.7` one for a nova id.
        assert_eq!(max_supported_effort("gpt-5.7-nova"), crate::types::EffortLevel::Ultra);
        // A sibling family under the broader prefix gets the broader override.
        assert_eq!(max_supported_effort("gpt-5.7-other"), crate::types::EffortLevel::Max);
        // A model the override does not name is untouched by it.
        assert_eq!(max_supported_effort("gpt-5.5"), crate::types::EffortLevel::Xhigh);
    }

    #[test]
    fn max_supported_effort_env_override_ignores_garbage() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(MODEL_EFFORT_CEILINGS_ENV, Some("not-json"));
        assert_eq!(max_supported_effort("gpt-5.6-sol"), crate::types::EffortLevel::Ultra);

        let _override = EnvVarGuard::set(MODEL_EFFORT_CEILINGS_ENV, Some(""));
        assert_eq!(max_supported_effort("gpt-5.6-sol"), crate::types::EffortLevel::Ultra);
    }

    fn request_with_text(model: &str, text: &str) -> crate::types::MessageRequest {
        crate::types::MessageRequest {
            model: model.to_string(),
            max_tokens: 4_096,
            messages: vec![crate::types::InputMessage::user_text(text)],
            system: None,
            tools: None,
            tool_choice: None,
            stream: false,
            thinking: None,
            output_config: None,
            effort: None,
            effort_band_ceiling: None,
        }
    }

    #[test]
    fn band_difficulty_for_request_matches_dynamic_effort_signal_triple() {
        // heavy-intent keyword (EN + KO), plain trivial, and a long ask each
        // fire exactly the one signal they should — the shared classifier the
        // GPT auto ladder and the ultra-band resolver both key off.
        let trivial = band_difficulty_for_request(&request_with_text("gpt-5.5", "hi"));
        assert_eq!(trivial, BandDifficulty::default());
        assert_eq!(trivial.signal_count(), 0);

        let heavy_en = band_difficulty_for_request(&request_with_text(
            "gpt-5.5",
            "please refactor this module",
        ));
        assert!(heavy_en.heavy_intent);
        assert!(!heavy_en.large_context);
        assert!(!heavy_en.long_ask);
        assert_eq!(heavy_en.signal_count(), 1);

        let heavy_ko =
            band_difficulty_for_request(&request_with_text("gpt-5.5", "이 코드베이스를 분석해줘"));
        assert!(heavy_ko.heavy_intent);
        assert_eq!(heavy_ko.signal_count(), 1);

        let long_ask = "word ".repeat(150); // > 600 chars, no heavy keyword
        let long_only = band_difficulty_for_request(&request_with_text("gpt-5.5", &long_ask));
        assert!(!long_only.heavy_intent);
        assert!(long_only.long_ask);
        assert_eq!(long_only.signal_count(), 1);

        // Heavy keyword AND a long ask stack to two signals.
        let heavy_and_long = format!("please refactor this module. {long_ask}");
        let two_signal =
            band_difficulty_for_request(&request_with_text("gpt-5.5", &heavy_and_long));
        assert!(two_signal.heavy_intent && two_signal.long_ask);
        assert_eq!(two_signal.signal_count(), 2);
    }

    #[test]
    fn resolve_effort_band_examples_sol_fable_luna() {
        use crate::types::EffortLevel::{Max, Ultra, Xhigh};

        let none = BandDifficulty::default();
        let one = BandDifficulty {
            heavy_intent: true,
            ..BandDifficulty::default()
        };
        let two = BandDifficulty {
            heavy_intent: true,
            long_ask: true,
            ..BandDifficulty::default()
        };

        // Sol: internal ceiling is Ultra — the full 3-rung selection band is
        // reachable before the OpenAI wire projection.
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "gpt-5.6-sol", none), Xhigh);
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "gpt-5.6-sol", one), Max);
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "gpt-5.6-sol", two), Ultra);

        // Fable: Anthropic's ceiling is Max (Ultra has no Anthropic wire value),
        // so the one-signal and two-signal rungs both land on Max — there is no
        // higher named level to escalate to.
        assert_eq!(
            resolve_effort_band(Xhigh, Ultra, "claude-fable-5", none),
            Xhigh
        );
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "claude-fable-5", one), Max);
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "claude-fable-5", two), Max);

        // Luna's internal selection ceiling is Max — the same [xhigh..max]
        // shape as Fable before provider-specific wire projection.
        assert_eq!(
            resolve_effort_band(Xhigh, Ultra, "gpt-5.6-luna", none),
            Xhigh
        );
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "gpt-5.6-luna", one), Max);
        assert_eq!(resolve_effort_band(Xhigh, Ultra, "gpt-5.6-luna", two), Max);
    }

    #[test]
    fn resolve_effort_band_gemini_collapses_to_its_own_ceiling() {
        use crate::types::EffortLevel::{High, Ultra, Xhigh};
        // Gemini's declared ceiling is High regardless of signal count, so the
        // band degenerates to a single rung — matching the pre-band static
        // projection (`EffortLevel::gemini()` already collapses everything
        // >= high).
        let two = BandDifficulty {
            heavy_intent: true,
            long_ask: true,
            ..BandDifficulty::default()
        };
        assert_eq!(
            resolve_effort_band(Xhigh, Ultra, "gemini-3.5-flash", BandDifficulty::default()),
            High
        );
        assert_eq!(
            resolve_effort_band(Xhigh, Ultra, "gemini-3.5-flash", two),
            High
        );
    }

    #[test]
    fn resolve_effort_band_kill_switch_returns_static_ceiling() {
        use crate::types::EffortLevel::{Max, Ultra, Xhigh};
        let _lock = crate::test_env_lock();

        {
            let _guard = EnvVarGuard::set(ULTRA_BAND_ENV, Some("off"));
            // With the band disabled, every call collapses to the clamped
            // ceiling regardless of difficulty — the legacy static-top pin.
            assert_eq!(
                resolve_effort_band(Xhigh, Ultra, "gpt-5.6-sol", BandDifficulty::default()),
                Ultra
            );
            assert_eq!(
                resolve_effort_band(
                    Xhigh,
                    Ultra,
                    "claude-fable-5",
                    BandDifficulty {
                        heavy_intent: true,
                        ..BandDifficulty::default()
                    }
                ),
                Max
            );
        }
        {
            // Case-insensitive matching still gates on the exact "off" value.
            let _guard = EnvVarGuard::set(ULTRA_BAND_ENV, Some("OFF"));
            assert_eq!(
                resolve_effort_band(Xhigh, Ultra, "gpt-5.6-sol", BandDifficulty::default()),
                Ultra
            );
        }

        // Guard dropped (env restored/unset) — the band is active again.
        assert_eq!(
            resolve_effort_band(Xhigh, Ultra, "gpt-5.6-sol", BandDifficulty::default()),
            Xhigh
        );
    }

    #[test]
    fn resolve_effort_band_never_exceeds_ceiling_or_falls_below_floor() {
        // Exhaustive sweep: every (floor, ceiling, model, signal-count) combo
        // must land within [floor ..= min(ceiling, model_ceiling)] and must be
        // deterministic (same inputs -> same output across repeated calls).
        use crate::types::EffortLevel::{High, Low, Max, Medium, Ultra, Xhigh};

        // resolve_effort_band reads ULTRA_BAND_ENV, so this read side must
        // hold the crate env lock too — a sibling test's EnvVarGuard flipping
        // the kill switch between the paired calls is exactly the observed
        // workspace-parallel flake.
        let _lock = crate::test_env_lock();

        let levels = [Low, Medium, High, Xhigh, Max, Ultra];
        let models = [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "claude-fable-5",
            "sonnet",
            "gemini-3.5-flash",
            "deepseek-chat",
        ];
        let difficulties = [
            BandDifficulty::default(),
            BandDifficulty {
                heavy_intent: true,
                ..BandDifficulty::default()
            },
            BandDifficulty {
                large_context: true,
                ..BandDifficulty::default()
            },
            BandDifficulty {
                heavy_intent: true,
                large_context: true,
                long_ask: true,
            },
        ];

        for &floor in &levels {
            for &ceiling in &levels {
                for &model in &models {
                    let effective_ceiling_rank = effort_rank(ceiling)
                        .min(effort_rank(max_supported_effort(model)));
                    let floor_rank = effort_rank(floor).min(effective_ceiling_rank);
                    for &difficulty in &difficulties {
                        let first = resolve_effort_band(floor, ceiling, model, difficulty);
                        let second = resolve_effort_band(floor, ceiling, model, difficulty);
                        assert_eq!(
                            first, second,
                            "non-deterministic for floor={floor:?} ceiling={ceiling:?} model={model} difficulty={difficulty:?}"
                        );
                        assert!(
                            effort_rank(first) <= effective_ceiling_rank,
                            "exceeded ceiling: floor={floor:?} ceiling={ceiling:?} model={model} difficulty={difficulty:?} got={first:?}"
                        );
                        assert!(
                            effort_rank(first) >= floor_rank,
                            "fell below floor: floor={floor:?} ceiling={ceiling:?} model={model} difficulty={difficulty:?} got={first:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn declared_model_class_matches_catalog_declarations() {
        // `declared_model_class` consults ZO_MODEL_CLASSES. Serialize with
        // the override tests below so their process-global env cannot leak
        // into this catalog-only assertion under the parallel test runner.
        let _lock = crate::test_env_lock();
        let cases = [
            // Codex model cache 2026-07-09: sol=frontier(1), terra=balanced(2), luna=fast(3).
            ("gpt-5.6-sol", Some(ModelClass::Frontier)),
            ("gpt-5.6-terra", Some(ModelClass::Balanced)),
            ("gpt-5.6-luna", Some(ModelClass::Fast)),
            // Dated/service-tier-suffixed ids resolve through the same matcher.
            ("gpt-5.6-sol-2026-07-09", Some(ModelClass::Frontier)),
            ("gpt-5.6-terra@openai", Some(ModelClass::Balanced)),
            ("gpt-5.6-luna[fast]", Some(ModelClass::Fast)),
            // gpt-5.5 superseded as frontier by sol; declared balanced.
            ("gpt-5.5", Some(ModelClass::Balanced)),
            ("gpt-5.5-fast", Some(ModelClass::Balanced)),
            ("gpt-5.5-2026-04-23", Some(ModelClass::Balanced)),
            // Anthropic public positioning: fable-5 above opus; opus/sonnet/haiku
            // are executors per the user-declared hierarchy.
            ("claude-fable-5", Some(ModelClass::Frontier)),
            ("fable", Some(ModelClass::Frontier)),
            ("claude-opus-4-8", Some(ModelClass::Balanced)),
            ("opus", Some(ModelClass::Balanced)),
            ("claude-sonnet-5", Some(ModelClass::Balanced)),
            ("sonnet", Some(ModelClass::Balanced)),
            ("claude-haiku-4-5-20251001", Some(ModelClass::Fast)),
            ("haiku", Some(ModelClass::Fast)),
            // Never declared for these providers — undeclared falls back to the
            // runtime's own capability-derived derivation.
            ("gpt-5.3-codex-spark", None),
            ("deepseek-chat", None),
            ("gemini-3.5-flash", None),
            ("grok-4", None),
            ("my-self-hosted-model", None),
        ];
        for (model, expected) in cases {
            assert_eq!(declared_model_class(model), expected, "model={model}");
        }
    }

    #[test]
    fn declared_model_class_env_override_wins_and_prefers_longest_prefix() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(
            MODEL_CLASSES_ENV,
            Some(r#"{"gpt-5.7": "balanced", "gpt-5.7-nova": "frontier"}"#),
        );
        // Longest matching prefix wins: the specific `gpt-5.7-nova` override
        // beats the broader `gpt-5.7` one for a nova id.
        assert_eq!(declared_model_class("gpt-5.7-nova"), Some(ModelClass::Frontier));
        // A sibling family under the broader prefix gets the broader override.
        assert_eq!(declared_model_class("gpt-5.7-other"), Some(ModelClass::Balanced));
        // The override wins over an existing catalog declaration too.
        let _override2 = EnvVarGuard::set(MODEL_CLASSES_ENV, Some(r#"{"gpt-5.6-sol": "fast"}"#));
        assert_eq!(declared_model_class("gpt-5.6-sol"), Some(ModelClass::Fast));
        // A model the override does not name falls through to the catalog.
        assert_eq!(declared_model_class("gpt-5.6-terra"), Some(ModelClass::Balanced));
    }

    #[test]
    fn declared_model_class_env_override_ignores_garbage() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(MODEL_CLASSES_ENV, Some("not-json"));
        assert_eq!(declared_model_class("gpt-5.6-sol"), Some(ModelClass::Frontier));

        let _override = EnvVarGuard::set(MODEL_CLASSES_ENV, Some(""));
        assert_eq!(declared_model_class("gpt-5.6-sol"), Some(ModelClass::Frontier));
    }

    #[test]
    fn effective_effort_matches_the_wire_clamp_for_each_provider() {
        use crate::types::EffortLevel;
        // The tripwire: applying the provider's final wire serializer to the
        // effective internal tier must equal serializing the requested tier
        // directly. If capability selection and wire projection drift apart,
        // this fails loudly.
        let reps = [
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5-20251001",
            "claude-fable-5",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.5-fast",
            "gpt-5.3-codex-spark",
            "gemini-3.5-flash",
        ];
        for model in reps {
            let lower = resolve_model_alias(model).to_ascii_lowercase();
            for level in [
                EffortLevel::Low,
                EffortLevel::Medium,
                EffortLevel::High,
                EffortLevel::Xhigh,
                EffortLevel::Max,
                EffortLevel::Ultra,
            ] {
                let effective = effective_effort_for_model(level, model);
                let wire = if lower.starts_with("claude") {
                    level.anthropic_for_model(&lower).anthropic()
                } else if lower.starts_with("gpt") {
                    level.gpt_for_model(&lower)
                } else {
                    level.gemini()
                };
                let projected_wire = if lower.starts_with("claude") {
                    effective.anthropic()
                } else if lower.starts_with("gpt") {
                    effective.gpt_for_model(&lower)
                } else {
                    effective.gemini()
                };
                assert_eq!(
                    projected_wire,
                    wire,
                    "projection drifted from wire: model={model} level={level:?} \
                     projection={effective:?} wire={wire}"
                );
            }
        }
    }

    #[test]
    fn effort_level_wire_strings_clamp_for_gpt() {
        use crate::types::EffortLevel;
        // Anthropic accepts the full scale.
        assert_eq!(EffortLevel::Max.anthropic(), "max");
        assert_eq!(EffortLevel::Xhigh.anthropic(), "xhigh");
        assert_eq!(EffortLevel::Ultra.anthropic(), "xhigh");
        assert_eq!(EffortLevel::Low.anthropic(), "low");
        // GPT's provider enum tops out at xhigh. GPT-5.6 may expose higher
        // Zo-side selection tiers, but those names must never reach wire.
        assert_eq!(EffortLevel::Max.gpt(), "xhigh");
        assert_eq!(EffortLevel::Max.gpt_for_model("gpt-5.6-sol"), "xhigh");
        assert_eq!(EffortLevel::Ultra.gpt_for_model("gpt-5.6-sol"), "xhigh");
        assert_eq!(EffortLevel::Max.gpt_for_model("gpt-5.5"), "xhigh");
        assert_eq!(EffortLevel::Xhigh.gpt(), "xhigh");
        assert_eq!(EffortLevel::High.gpt(), "high");
    }

    #[test]
    fn opus_and_fable_are_the_only_1m_claude_windows() {
        assert_eq!(context_window_for_model("fable"), 1_000_000);
        assert_eq!(context_window_for_model("claude-fable-5"), 1_000_000);
        assert_eq!(context_window_for_model("opus"), 1_000_000);
        assert_eq!(context_window_for_model("claude-opus-4-8"), 1_000_000);
        assert_eq!(context_window_for_model("claude-opus-4-6[1m]"), 1_000_000);
    }

    #[test]
    fn opus_1m_label_alias_resolves_to_bare_opus_with_1m_window() {
        // The `opus[1m]` picker label is the same model as `opus`: it resolves
        // to the bare `claude-opus-4-8` (so the `[1m]` suffix never reaches the
        // wire and cannot 404) and still reports the native 1M window.
        assert_eq!(resolve_model_alias("opus[1m]"), "claude-opus-4-8");
        assert_eq!(resolve_model_alias("claude-opus[1m]"), "claude-opus-4-8");
        assert!(
            !resolve_model_alias("opus[1m]").contains('['),
            "the [1m] label must never reach the wire model id"
        );
        assert_eq!(context_window_for_model("opus[1m]"), 1_000_000);
        assert_eq!(context_window_for_model("claude-opus[1m]"), 1_000_000);
    }

    #[test]
    fn sonnet_5_uses_1m_context_window() {
        assert_eq!(context_window_for_model("sonnet"), 1_000_000);
        assert_eq!(context_window_for_model("claude-sonnet"), 1_000_000);
        assert_eq!(context_window_for_model("claude-sonnet-5"), 1_000_000);
        assert_eq!(context_window_for_model("claude-sonnet-5[1m]"), 1_000_000);
    }

    #[test]
    fn legacy_sonnet_and_haiku_are_capped_at_258k_even_with_1m_suffix() {
        assert_eq!(context_window_for_model("claude-sonnet-4-6"), 258_000);
        assert_eq!(context_window_for_model("claude-sonnet-4-6[1m]"), 258_000);
        assert_eq!(context_window_for_model("haiku"), 258_000);
        assert_eq!(
            context_window_for_model("claude-haiku-4-5-20251001"),
            258_000
        );
        assert_eq!(
            context_window_for_model("claude-haiku-4-5-20251001[1m]"),
            258_000
        );
    }

    #[test]
    fn codex_spark_context_window_comes_from_capability_catalog() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(MODEL_CONTEXT_WINDOWS_ENV, None);
        assert_eq!(context_window_for_model("gpt-5.3-codex-spark"), 122_000);
    }

    #[test]
    fn gpt_context_window_comes_from_capability_catalog() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(MODEL_CONTEXT_WINDOWS_ENV, None);
        assert_eq!(context_window_for_model("gpt"), 258_000);
        // User-directed 2026-07-14: the whole GPT family rides the 258k
        // effective window — the 5.6 trio's declared 353k proved unusable in
        // live sessions past ~256k.
        assert_eq!(context_window_for_model("gpt-5.6-sol"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.6-sol-2026-07-09"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.6-terra"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.6-terra@openai"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.6-luna"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.6-luna[fast]"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.5"), 258_000);
        assert_eq!(context_window_for_model("gpt-5.5-fast"), 258_000);
        assert_eq!(context_window_for_model("gpt-6-preview"), 200_000);
    }

    #[test]
    fn model_metadata_helpers_stay_consistent_across_alias_and_canonical_ids() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        let _override = EnvVarGuard::set(MODEL_CONTEXT_WINDOWS_ENV, None);

        for (alias, canonical, provider, context_window, adaptive) in [
            (
                "opus",
                "claude-opus-4-8",
                ProviderKind::Anthropic,
                1_000_000,
                true,
            ),
            (
                "sonnet",
                "claude-sonnet-5",
                ProviderKind::Anthropic,
                1_000_000,
                true,
            ),
            (
                "gpt-5.6-sol",
                "gpt-5.6-sol",
                ProviderKind::OpenAi,
                258_000,
                false,
            ),
            (
                "gpt-5.6-terra",
                "gpt-5.6-terra",
                ProviderKind::OpenAi,
                258_000,
                false,
            ),
            (
                "gpt-5.6-luna",
                "gpt-5.6-luna",
                ProviderKind::OpenAi,
                258_000,
                false,
            ),
            // gpt-5.5는 카탈로그 퇴역(2026-07-11) — bare 세대 별칭은 이제
            // gpt-5.6→terra다. 컨텍스트 윈도는 terra의 것.
            (
                "gpt-5.6",
                "gpt-5.6-terra",
                ProviderKind::OpenAi,
                258_000,
                false,
            ),
            (
                "gemini-flash",
                "gemini-3.5-flash",
                ProviderKind::Google,
                1_000_000,
                false,
            ),
        ] {
            assert_eq!(resolve_model_alias(alias), canonical, "alias {alias}");
            assert_eq!(
                detect_provider_kind(canonical),
                provider,
                "provider {alias}"
            );
            assert_eq!(
                context_window_for_model(alias),
                context_window,
                "alias context {alias}"
            );
            assert_eq!(
                context_window_for_model(canonical),
                context_window,
                "canonical context {canonical}"
            );
            assert_eq!(
                uses_adaptive_thinking(alias),
                adaptive,
                "alias adaptive thinking {alias}"
            );
            assert_eq!(
                uses_adaptive_thinking(canonical),
                adaptive,
                "canonical adaptive thinking {canonical}"
            );
        }
    }

    #[test]
    fn model_capability_is_a_narrow_projection_of_existing_helpers() {
        let _lock = crate::test_env_lock();
        let _gate = EnvVarGuard::set(NON_CLAUDE_ADAPTERS_ENV, Some("1"));
        let _override = EnvVarGuard::set(MODEL_CONTEXT_WINDOWS_ENV, None);

        for model in ["opus", "gpt-5.5", "gemini-flash", "grok", "unknown-model"] {
            let capability = model_capability_for_model(model);
            assert_eq!(capability.canonical_model_id, resolve_model_alias(model));
            assert_eq!(
                capability.provider.map(ProviderKind::metadata),
                super::metadata_for_model(model)
            );
            assert_eq!(
                capability.context_window,
                Some(context_window_for_model(model))
            );
            assert_eq!(
                capability.max_output_tokens,
                Some(max_tokens_for_model(model))
            );
            assert_eq!(
                capability.adaptive_thinking,
                Some(uses_adaptive_thinking(model))
            );
        }
    }

    #[test]
    fn custom_provider_context_and_output_overrides_are_used() {
        let _lock = crate::test_env_lock();
        let _catalog_override = EnvVarGuard::set(MODEL_CONTEXT_WINDOWS_ENV, None);
        super::refresh_custom_providers_from_json(
            r#"[{"name":"xai-custom","base_url":"https://api.x.ai/v1","models":["grok-4.5"],"requires_auth":false,"context_window":256000,"max_output_tokens":32000}]"#,
        )
        .expect("refresh custom provider");

        assert_eq!(context_window_for_model("grok-4.5"), 256_000);
        assert_eq!(max_tokens_for_model("grok-4.5"), 32_000);

        super::refresh_custom_providers_from_json("[]").expect("restore empty custom providers");
    }

    #[test]
    fn unknown_grok_versions_do_not_inherit_grok3_context_window() {
        let _lock = crate::test_env_lock();
        let _catalog_override = EnvVarGuard::set(MODEL_CONTEXT_WINDOWS_ENV, None);
        super::refresh_custom_providers_from_json("[]").expect("clear custom providers");
        assert_eq!(context_window_for_model("grok-3"), 131_072);
        assert_eq!(context_window_for_model("grok-4.5"), 200_000);
    }

    #[test]
    fn context_window_can_be_supplied_without_code_changes() {
        let _lock = crate::test_env_lock();
        let _override = EnvVarGuard::set(
            MODEL_CONTEXT_WINDOWS_ENV,
            Some(r#"{"models":[{"ids":["gpt-future"],"context_window":777000}]}"#),
        );
        assert_eq!(context_window_for_model("gpt-future"), 777_000);
    }

    #[test]
    fn resolves_claude_aliases_and_normalizes_dotted_versions() {
        // Anthropic is always enabled, so these hold regardless of the gate.
        assert_eq!(resolve_model_alias("fable"), "claude-fable-5");
        assert_eq!(resolve_model_alias("claude-fable"), "claude-fable-5");
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-8");
        assert_eq!(resolve_model_alias("claude-opus"), "claude-opus-4-8");
        assert_eq!(resolve_model_alias("sonnet"), "claude-sonnet-5");
        assert_eq!(resolve_model_alias("claude-sonnet"), "claude-sonnet-5");
        assert_eq!(
            resolve_model_alias("claude-haiku"),
            "claude-haiku-4-5-20251001"
        );
        // Dot-separated versions of fully-qualified ids normalise to hyphens.
        assert_eq!(resolve_model_alias("claude-opus-4.6"), "claude-opus-4-6");
        assert_eq!(
            resolve_model_alias("claude-sonnet-4.6"),
            "claude-sonnet-4-6"
        );
        // Non-Claude ids are passed through verbatim.
        assert_eq!(resolve_model_alias("grok-3"), "grok-3");
    }

    #[test]
    fn provider_capabilities_distinguish_cache_from_thinking_support() {
        assert!(ProviderKind::Anthropic.supports_cache_tokens());
        assert!(ProviderKind::OpenAi.supports_cache_tokens());
        assert!(ProviderKind::Anthropic.supports_thinking());
        assert_eq!(
            ProviderKind::Anthropic.prompt_cache_strategy(),
            PromptCacheStrategy::AnthropicCacheControl
        );
        assert_eq!(
            ProviderKind::OpenAi.prompt_cache_strategy(),
            PromptCacheStrategy::OpenAiPromptCacheKey
        );
        for kind in [
            ProviderKind::Google,
            ProviderKind::Xai,
            ProviderKind::Ollama,
        ] {
            assert!(!kind.supports_cache_tokens(), "{kind} should not cache");
            assert!(!kind.supports_thinking(), "{kind} should not think");
            assert_eq!(
                kind.prompt_cache_strategy(),
                PromptCacheStrategy::NoRequestControls
            );
        }
        assert_eq!(ProviderKind::Anthropic.rate_limit_key(), "anthropic");
        assert_eq!(ProviderKind::OpenAi.rate_limit_key(), "openai");
    }

    #[test]
    fn every_provider_profile_is_fully_populated() {
        // The single profile table is the source of truth: a new provider that
        // forgets a field would surface here rather than silently shipping an
        // empty display name or rate-limit key. Display must echo the profile,
        // and metadata must mirror the profile's connection wiring.
        for kind in [
            ProviderKind::Anthropic,
            ProviderKind::Xai,
            ProviderKind::OpenAi,
            ProviderKind::Google,
            ProviderKind::Ollama,
        ] {
            let profile = kind.profile();
            assert!(!profile.display_name.is_empty(), "{kind} display name");
            assert!(!profile.rate_limit_key.is_empty(), "{kind} rate-limit key");
            assert!(!profile.auth_env.is_empty(), "{kind} auth env");
            assert!(!profile.base_url_env.is_empty(), "{kind} base-url env");
            assert!(
                !profile.default_base_url.is_empty(),
                "{kind} default base url"
            );
            assert_eq!(kind.to_string(), profile.display_name, "{kind} Display");
            let metadata = kind.metadata();
            assert_eq!(metadata.auth_env, profile.auth_env);
            assert_eq!(metadata.base_url_env, profile.base_url_env);
            assert_eq!(metadata.default_base_url, profile.default_base_url);
        }
    }

    #[test]
    fn openai_gpt_family_policy_matches_supported_picker_models() {
        assert_eq!(openai_gpt_model_family("gpt-5.5"), Some("gpt-5.5"));
        assert_eq!(
            openai_gpt_model_family("gpt-5.5-2026-04-23"),
            Some("gpt-5.5")
        );
        assert_eq!(openai_gpt_model_family("gpt-5.5-fast"), Some("gpt-5.5"));
        assert_eq!(openai_gpt_model_family("gpt-5.6-sol"), Some("gpt-5.6-sol"));
        assert_eq!(
            openai_gpt_model_family("gpt-5.6-terra"),
            Some("gpt-5.6-terra")
        );
        assert_eq!(
            openai_gpt_model_family("gpt-5.6-luna"),
            Some("gpt-5.6-luna")
        );
        assert_eq!(
            openai_gpt_model_family("gpt-5.3-codex-spark"),
            Some("gpt-5.3-codex-spark")
        );
        assert_eq!(openai_gpt_model_family("gpt-5"), None);
        assert_eq!(openai_gpt_model_family("gpt-5.1-codex"), None);
        assert_eq!(openai_gpt_model_family("gpt-4.1"), None);
    }

    #[test]
    fn openai_extended_prompt_cache_policy_does_not_send_unverified_retention() {
        for model in [
            "gpt-5.5",
            "gpt-5.5-fast",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.3-codex-spark",
        ] {
            assert_eq!(openai_prompt_cache_retention(model), None, "{model}");
        }
    }

    #[test]
    fn catalog_entries_derive_metadata_from_their_provider() {
        for entry in provider_catalog() {
            assert_eq!(entry.metadata(), entry.provider.metadata());
            assert_eq!(entry.metadata().provider, entry.provider);
            assert_eq!(
                entry.supports_cache_tokens(),
                entry.provider.supports_cache_tokens()
            );
            // Every alias maps to a non-empty canonical id.
            assert!(!entry.canonical_model_id.is_empty());
        }
        // The catalog is the single source of truth for alias resolution.
        let opus = provider_catalog()
            .iter()
            .find(|entry| entry.alias == "opus")
            .expect("opus alias present in catalog");
        assert_eq!(opus.canonical_model_id, "claude-opus-4-8");
        assert_eq!(opus.provider, ProviderKind::Anthropic);
    }

    fn resolved(json: &str) -> super::ResolvedCustomProvider {
        let custom: super::openai_compat::CustomProviderConfig =
            super::parse_custom_providers(&format!("[{json}]"))
                .expect("valid custom provider json")
                .pop()
                .expect("one provider parsed");
        super::ResolvedCustomProvider {
            context_window: custom.context_window.filter(|&value| value > 0),
            max_output_tokens: custom.max_output_tokens.filter(|&value| value > 0),
            fit_hint: custom.to_fit_hint(),
            config: custom.to_static_config(),
            models: custom.models,
            requires_auth: custom.requires_auth,
        }
    }

    #[test]
    fn custom_provider_parses_models_and_auth_defaults() {
        let provider = resolved(
            r#"{"name":"Local","base_url":"http://localhost:8080/v1",
                "auth_env":"LOCAL_KEY","models":["llama-3.3","mistral-large"]}"#,
        );
        // `requires_auth` defaults to true when omitted.
        assert!(provider.requires_auth);
        assert_eq!(provider.config.provider_name, "Local");
        assert_eq!(provider.config.api_key_env, "LOCAL_KEY");
        assert_eq!(provider.config.credential_env_vars(), &["LOCAL_KEY"]);
        assert_eq!(provider.config.default_base_url, "http://localhost:8080/v1");
        assert!(provider.config.request_stream_usage);
        assert!(provider.fit_hint.is_none());
    }

    #[test]
    fn custom_provider_matches_models_case_insensitively() {
        let provider = resolved(
            r#"{"name":"Local","base_url":"http://localhost:8080/v1",
                "models":["Llama-3.3"],"requires_auth":false}"#,
        );
        assert!(provider.serves("llama-3.3"));
        assert!(provider.serves("LLAMA-3.3"));
        assert!(!provider.serves("gpt-5.6-sol"));
        // The registered casing is the canonical id sent to the endpoint.
        assert_eq!(provider.canonical_model("llama-3.3"), Some("Llama-3.3"));
        // Keyless self-host: no auth env, no key required.
        assert!(!provider.requires_auth);
        assert_eq!(provider.config.api_key_env, "");
    }

    #[test]
    fn custom_provider_preserves_read_only_fit_metadata() {
        let provider = resolved(
            r#"{"name":"LM Studio","base_url":"http://localhost:1234/v1",
                "models":["qwen2.5-coder-32b"],"requires_auth":false,
                "estimated_vram_gb":22,"quantization":"Q4_K_M"}"#,
        );
        let hint = provider.fit_hint.expect("fit hint");
        assert_eq!(hint.estimated_vram_gb, 22);
        assert_eq!(hint.quantization, "Q4_K_M");
        assert_eq!(hint.display_label(), "VRAM ~22GB Q4_K_M");
    }

    #[test]
    fn custom_provider_can_disable_stream_usage_opt_in() {
        let provider = resolved(
            r#"{"name":"LocalAI","base_url":"http://localhost:8080/v1",
                "models":["localai-model"],"requires_auth":false,"include_usage":false}"#,
        );
        assert!(!provider.config.request_stream_usage);
    }

    #[test]
    fn custom_provider_json_array_round_trips_and_rejects_garbage() {
        let parsed = super::parse_custom_providers(
            r#"[{"name":"A","base_url":"http://a/v1","models":["x"]},
                {"name":"B","base_url":"http://b/v1","models":["y"],"requires_auth":false}]"#,
        )
        .expect("two providers");
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].requires_auth);
        assert!(!parsed[1].requires_auth);
        assert!(parsed[0].include_usage);
        assert!(parsed[1].include_usage);
        // Malformed JSON is a hard parse error (surfaced, not silently dropped).
        assert!(super::parse_custom_providers("not json").is_err());
    }

    #[test]
    fn custom_provider_catalog_can_refresh_after_initial_empty_read() {
        let _lock = crate::test_env_lock();
        let _env = EnvVarGuard::set(super::CUSTOM_PROVIDERS_ENV, None);
        super::refresh_custom_providers_from_json("[]").expect("clear custom catalog");
        assert!(super::custom_provider_catalog().is_empty());

        super::refresh_custom_providers_from_json(
            r#"[{"name":"deepseek","base_url":"https://api.deepseek.com",
                "auth_env":"DEEPSEEK_API_KEY",
                "models":["deepseek-chat","deepseek-reasoner"]}]"#,
        )
        .expect("refresh custom catalog");

        let catalog = super::custom_provider_catalog();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].0, "deepseek");
        assert_eq!(
            catalog[0].1,
            vec![
                "deepseek-chat".to_string(),
                "deepseek-reasoner".to_string()
            ]
        );
        assert_eq!(resolve_model_alias("deepseek-chat"), "deepseek-chat");
        assert!(super::custom_provider_for_model("deepseek-reasoner").is_some());

        super::refresh_custom_providers_from_json("[]").expect("restore empty custom catalog");
    }

    #[test]
    fn resolve_model_alias_static_registry_wins_over_custom_collision() {
        // Built-in aliases are matched before the custom catalog, so a custom
        // provider can never shadow them. `opus` (Anthropic, always enabled)
        // demonstrates this without any adapter-gate setup.
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-8");
        // A model with neither a registry entry nor a custom provider passes
        // through unchanged — the custom fallback never fabricates an id.
        assert_eq!(
            resolve_model_alias("totally-unknown-model"),
            "totally-unknown-model"
        );
    }

    #[test]
    fn qualified_provider_model_selects_custom_without_shadowing_bare_claude() {
        let _lock = crate::test_env_lock();
        let _env = EnvVarGuard::set(super::CUSTOM_PROVIDERS_ENV, None);
        super::refresh_custom_providers_from_json(
            r#"[{"name":"agent router","base_url":"https://agentrouter.org/v1",
                "auth_env":"ZO_AGENT_ROUTER_API_KEY",
                "models":["claude-opus-4-8"],"requires_auth":false}]"#,
        )
        .expect("load agent router");

        // Bare Claude id stays first-party / OAuth.
        assert_eq!(resolve_model_alias("claude-opus-4-8"), "claude-opus-4-8");
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-8");
        assert!(super::custom_provider_for_model("claude-opus-4-8").is_some());
        assert_eq!(
            super::detect_provider_kind("claude-opus-4-8"),
            ProviderKind::Anthropic
        );
        assert_eq!(
            super::wire_model_id("claude-opus-4-8"),
            "claude-opus-4-8"
        );

        // Explicit provider/model opts into the custom gateway.
        let qualified = "agent router/claude-opus-4-8";
        assert_eq!(resolve_model_alias(qualified), qualified);
        assert_eq!(super::wire_model_id(qualified), "claude-opus-4-8");
        let custom = super::custom_provider_for_model(qualified).expect("qualified custom");
        assert_eq!(custom.config.provider_name, "agent router");
        assert_eq!(
            super::detect_provider_kind(qualified),
            ProviderKind::OpenAi
        );
        assert_eq!(
            super::split_provider_model_ref(qualified),
            Some(("agent router", "claude-opus-4-8"))
        );
        assert_eq!(
            super::format_provider_model_ref("agent router", "claude-opus-4-8"),
            qualified
        );

        super::refresh_custom_providers_from_json("[]").expect("clear");
    }

    #[test]
    fn fit_hints_are_hidden_when_feature_is_disabled() {
        assert!(fit_hint_for_model("qwen2.5-coder-32b").is_none());
    }

    #[test]
    fn flatten_tool_result_content_joins_and_degrades() {
        use crate::types::{ImageSource, ToolResultContentBlock};

        let blocks = vec![
            ToolResultContentBlock::Text {
                text: "first".to_string(),
            },
            ToolResultContentBlock::Json {
                value: serde_json::json!({ "ok": true }),
            },
            ToolResultContentBlock::Image {
                source: ImageSource {
                    kind: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: "ZGF0YQ==".to_string(),
                },
            },
        ];

        // Blocks join with '\n'; JSON serializes; images degrade to a placeholder.
        assert_eq!(
            super::flatten_tool_result_content(&blocks),
            "first\n{\"ok\":true}\n[image image/png]"
        );
        // Empty input yields an empty string (no leading newline).
        assert_eq!(super::flatten_tool_result_content(&[]), "");
    }

    #[test]
    fn backoff_for_attempt_doubles_caps_and_overflows() {
        use std::time::Duration;

        let initial = Duration::from_millis(500);
        let max = Duration::from_secs(8);

        // attempt 1 -> initial; then doubling each attempt …
        assert_eq!(
            super::backoff_for_attempt(1, initial, max).unwrap(),
            Duration::from_millis(500)
        );
        assert_eq!(
            super::backoff_for_attempt(2, initial, max).unwrap(),
            Duration::from_secs(1)
        );
        assert_eq!(
            super::backoff_for_attempt(4, initial, max).unwrap(),
            Duration::from_secs(4)
        );
        // … until the cap clamps it.
        assert_eq!(
            super::backoff_for_attempt(5, initial, max).unwrap(),
            Duration::from_secs(8)
        );
        // attempt 10 doubles to 256s but its shift still fits in u32, so it
        // clamps to the cap rather than overflowing (cf. attempt 40 below).
        assert_eq!(
            super::backoff_for_attempt(10, initial, max).unwrap(),
            Duration::from_secs(8)
        );

        // A multiplier shift that overflows u32 surfaces an error rather than
        // silently producing a bogus delay (drift the merge removed).
        assert!(matches!(
            super::backoff_for_attempt(40, initial, max),
            Err(crate::error::ApiError::BackoffOverflow { attempt: 40, .. })
        ));
    }

    #[test]
    fn should_restart_within_budget_adds_a_wallclock_ceiling() {
        use std::time::Duration;
        let cap = Duration::from_secs(120);

        // Before any restart (`elapsed == None`) it matches the plain predicate:
        // the first restart of a sequence is always allowed.
        assert!(super::should_restart_within_budget(
            false, true, 0, 5, None, cap
        ));
        // Within the wall-clock budget and under the attempt cap → still allowed.
        assert!(super::should_restart_within_budget(
            false,
            true,
            2,
            5,
            Some(Duration::from_secs(30)),
            cap
        ));
        // Past the wall-clock budget → denied even though attempts remain (the
        // silent-storm case: idle-timeout × restarts that never exhaust attempts
        // but hold the turn for minutes).
        assert!(!super::should_restart_within_budget(
            false,
            true,
            2,
            5,
            Some(Duration::from_secs(121)),
            cap
        ));
        // The wall-clock gate only tightens `should_restart`; it never overrides
        // commit/retryable/attempt denials.
        assert!(!super::should_restart_within_budget(
            true,
            true,
            0,
            5,
            Some(Duration::ZERO),
            cap
        ));
        assert!(!super::should_restart_within_budget(
            false,
            false,
            0,
            5,
            Some(Duration::ZERO),
            cap
        ));
        assert!(!super::should_restart_within_budget(
            false,
            true,
            5,
            5,
            Some(Duration::ZERO),
            cap
        ));
    }
}
