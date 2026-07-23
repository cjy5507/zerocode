use core_types::{TokenUsage, UsageCostEstimate, pricing_for_model};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Cache control directive for Anthropic prompt caching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    #[must_use]
    pub fn ephemeral() -> Self {
        Self {
            control_type: "ephemeral".to_string(),
            ttl: None,
        }
    }

    #[must_use]
    pub fn ephemeral_1h() -> Self {
        Self {
            control_type: "ephemeral".to_string(),
            ttl: Some("1h".to_string()),
        }
    }
}

/// A block in the `system` array of a [`MessageRequest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum SystemBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl SystemBlock {
    /// Convenience constructor for a plain text system block (no cache control).
    #[must_use]
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text {
            text: s.into(),
            cache_control: None,
        }
    }
}

/// Build a single-element system block vec from a plain string.
#[must_use]
pub fn system_from_string(s: impl Into<String>) -> Vec<SystemBlock> {
    vec![SystemBlock::text(s)]
}

/// Source descriptor for a document content block (e.g. PDF).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum DocumentSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

fn deserialize_system<'de, D>(deserializer: D) -> Result<Option<Vec<SystemBlock>>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(system_from_string(s))),
        Value::Array(arr) => {
            let blocks: Vec<SystemBlock> =
                serde_json::from_value(Value::Array(arr)).map_err(serde::de::Error::custom)?;
            Ok(Some(blocks))
        }
        other => Err(serde::de::Error::custom(format!(
            "system must be a string or an array of text blocks, got {other}"
        ))),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<InputMessage>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_system"
    )]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Extended thinking configuration. When set, the model uses a
    /// thinking budget before generating the final response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Anthropic effort control (Opus 4.6+/Fable/Sonnet 5). Sent on the wire
    /// as `output_config.effort` for models that use adaptive thinking; legacy
    /// models keep using `thinking.budget_tokens` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// Provider-neutral reasoning effort the caller selected, if any. Not part
    /// of the Anthropic wire format (the Anthropic body carries effort via
    /// [`Self::output_config`]); OpenAI/GPT backends read this to set
    /// `reasoning_effort`. Skipped from serialization entirely.
    #[serde(skip)]
    pub effort: Option<EffortLevel>,
    /// When `Some(ceiling)`, this request carries a *dynamic effort band*
    /// rather than a static pin: [`Self::effort`] holds the band FLOOR (always
    /// `Xhigh` in practice) and each wire backend resolves the concrete level
    /// to send — somewhere in `[floor ..= min(ceiling, model ceiling)]` —
    /// per-request from the message content via
    /// [`crate::providers::resolve_effort_band`]. `None` (the default at
    /// every pre-existing construction site) means fully static behavior:
    /// whatever [`Self::effort`] carries is sent verbatim, byte-identical to
    /// before this field existed. Never serialized (`serde(skip)`) — this is
    /// a resolver input, not part of any wire format.
    #[serde(skip)]
    pub effort_band_ceiling: Option<EffortLevel>,
}

/// Configuration for Anthropic extended thinking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Type discriminator — `"enabled"` for legacy budget-based thinking, or
    /// `"adaptive"` for Opus 4.6+/Fable models that let the server size the
    /// thinking budget from `output_config.effort`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Maximum tokens the model may spend on thinking. `None` (skipped on the
    /// wire) for adaptive thinking, where effort governs the budget instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Reasoning-summary visibility for adaptive thinking. `Some("summarized")`
    /// asks the server to stream summarized reasoning deltas; the Opus 4.8/4.7/
    /// Fable default is `"omitted"`, which streams *empty* thinking blocks and
    /// reads as a long, dead pause before the answer. `None` (skipped on the
    /// wire) keeps the server default. Adaptive-only; legacy budget thinking
    /// leaves it `None`.
    #[serde(rename = "display", skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

impl ThinkingConfig {
    /// Legacy budget-based thinking (`{"type":"enabled","budget_tokens":N}`),
    /// for models that still accept an explicit budget (Opus 4.5 and earlier,
    /// custom endpoints).
    #[must_use]
    pub fn enabled(budget_tokens: u32) -> Self {
        Self {
            kind: "enabled".to_string(),
            budget_tokens: Some(budget_tokens),
            display: None,
        }
    }

    /// Adaptive thinking (`{"type":"adaptive"}`), for Opus 4.6+/Fable models
    /// where the server sizes the budget from `output_config.effort`. Emits no
    /// `budget_tokens` on the wire.
    #[must_use]
    pub fn adaptive() -> Self {
        Self {
            kind: "adaptive".to_string(),
            budget_tokens: None,
            display: None,
        }
    }

    /// Adaptive thinking that streams a *visible* summary of the reasoning
    /// (`{"type":"adaptive","display":"summarized"}`). Without this, Opus 4.8's
    /// default `display:"omitted"` streams empty thinking blocks, so a long
    /// reasoning pass shows as dead "no output" before the answer. Only attach
    /// for models known to accept `display` (Opus/Fable — see
    /// [`anthropic_model_accepts_xhigh`]); other adaptive models keep
    /// [`Self::adaptive`].
    #[must_use]
    pub fn adaptive_summarized() -> Self {
        Self {
            kind: "adaptive".to_string(),
            budget_tokens: None,
            display: Some("summarized".to_string()),
        }
    }
}

/// Anthropic `output_config` block. Currently carries only the effort control
/// used by adaptive-thinking models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Reasoning effort, serialized as one of `low|medium|high|xhigh|max`.
    pub effort: EffortLevel,
}

impl OutputConfig {
    #[must_use]
    pub fn new(effort: EffortLevel) -> Self {
        Self { effort }
    }
}

/// Provider-neutral reasoning effort. Maps to Anthropic `output_config.effort`
/// and to OpenAI/GPT `reasoning_effort`, which use different scales at the
/// top end — see [`EffortLevel::anthropic`], [`EffortLevel::gpt`], and
/// [`EffortLevel::gemini`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    Ultra,
}

/// Provider-neutral reasoning intent derived from existing request fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningRequest {
    Auto,
    Effort(EffortLevel),
    /// Positive legacy thinking budget. `MessageRequest::reasoning_request`
    /// maps zero or absent budgets to `Auto`.
    BudgetTokens(u32),
}

impl EffortLevel {
    /// The Anthropic `output_config.effort` wire string. Anthropic has no
    /// `ultra` tier, so provider-neutral Ultra projects conservatively to xhigh.
    #[must_use]
    pub const fn anthropic(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            // Anthropic has no Ultra tier. Callers must project through
            // `anthropic_for_model` before serializing provider output.
            Self::Xhigh | Self::Ultra => "xhigh",
            Self::Max => "max",
        }
    }

    /// The conservative OpenAI/GPT `reasoning_effort` wire string for callers
    /// that do not know the concrete model. Older GPT families top out at
    /// `xhigh`, so use [`Self::gpt_for_model`] when the model id is available.
    #[must_use]
    pub const fn gpt(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh | Self::Max | Self::Ultra => "xhigh",
        }
    }

    /// The GPT `reasoning_effort` wire string for a specific `model`.
    ///
    /// GPT fast mode is a serving-priority signal, not a reasoning-effort
    /// ceiling. Zo's `Max` and `Ultra` remain useful internal selection
    /// tiers, but the OpenAI wire enum tops out at `xhigh`; sending either
    /// internal name produces a 400 response.
    #[must_use]
    pub fn gpt_for_model(self, model: &str) -> &'static str {
        match self {
            Self::Xhigh if !gpt_model_accepts_xhigh(model) => "high",
            Self::Max | Self::Ultra => "xhigh",
            other => other.gpt(),
        }
    }

    /// The Anthropic `output_config.effort` [`EffortLevel`] for a specific
    /// `model`, clamped to what that model accepts.
    ///
    /// Like [`Self::anthropic`], but honors the fact that not every adaptive
    /// Anthropic model accepts the `xhigh` tier. Sonnet/Haiku expose only
    /// `low|medium|high|max` and 400 on `xhigh`
    /// (`This model does not support effort level 'xhigh'`), which killed
    /// sub-agents that inherited an `Xhigh` budget (e.g. an opus→sonnet
    /// starvation-demoted `deep-research` agent carrying the 16k Xhigh preset).
    /// For those, `Xhigh` clamps to the model's real ceiling, [`Self::High`] —
    /// deliberately *not* [`Self::Max`], so the wire never silently sends a
    /// *higher* effort than was requested. Opus/Fable keep the full scale. This
    /// is the Anthropic analogue of [`Self::gpt_for_model`]; the `output_config`
    /// holds an [`EffortLevel`] (serialized directly), so this returns the
    /// clamped level rather than a wire string.
    #[must_use]
    pub fn anthropic_for_model(self, model: &str) -> EffortLevel {
        match self {
            EffortLevel::Ultra if anthropic_model_accepts_xhigh(model) => EffortLevel::Xhigh,
            EffortLevel::Ultra | EffortLevel::Xhigh
                if !anthropic_model_accepts_xhigh(model) => EffortLevel::High,
            _ => self,
        }
    }

    /// The Gemini 3 `generationConfig.thinkingConfig.thinkingLevel` wire string.
    /// Gemini 3 exposes only `low|medium|high` (it dropped 2.5's numeric
    /// `thinkingBudget`), so this is the building block: everything at or above
    /// `high` collapses to `high`. Family-specific clamping (Pro has no
    /// `medium`) is layered on top by the Gemini backend's `gemini_wire`.
    #[must_use]
    pub const fn gemini(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            // Gemini 3 tops out at "high"; xhigh/max collapse down to it.
            Self::High | Self::Xhigh | Self::Max | Self::Ultra => "high",
        }
    }
}

/// Whether a GPT-style `model` id accepts the `xhigh` reasoning-effort tier.
///
/// GPT fast mode is a serving-priority signal, not a reasoning-effort ceiling:
/// `gpt-5.5-fast` still accepts an explicit `xhigh`/`smart` request. Keep
/// this predicate centralized for any future model-specific ceiling, but do not
/// infer one from `fast` or `codex` tokens.
#[must_use]
pub fn gpt_model_accepts_xhigh(_model: &str) -> bool {
    true
}

/// Whether Zo exposes the internal `Max` reasoning-effort tier for a GPT
/// model.
///
/// This is an internal capability used by Smart/effort selection. It does not
/// imply that the provider accepts a literal `"max"`;
/// [`EffortLevel::gpt_for_model`] projects it to the supported wire ceiling.
#[must_use]
pub fn gpt_model_accepts_max(model: &str) -> bool {
    let lower = model.trim().to_ascii_lowercase();
    ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
        .iter()
        .any(|prefix| model_id_matches_family(&lower, prefix))
}

/// Whether Zo exposes the internal `Ultra` reasoning tier for a GPT model.
/// Only GPT-5.6 Sol and Terra expose it; Luna tops out at internal `Max`. The
/// provider wire value is still projected by [`EffortLevel::gpt_for_model`].
#[must_use]
pub fn gpt_model_accepts_ultra(model: &str) -> bool {
    let lower = model.trim().to_ascii_lowercase();
    ["gpt-5.6-sol", "gpt-5.6-terra"]
        .iter()
        .any(|prefix| model_id_matches_family(&lower, prefix))
}

/// Shared family/prefix matcher: `model` matches `prefix` exactly, or `prefix`
/// followed by a segment boundary (`-`, `@`, or `[`) — so a dated id
/// (`gpt-5.6-sol-2026-07-09`), an explicit-provider suffix (`gpt-5.6-terra@openai`),
/// or a service-tier suffix (`gpt-5.6-terra[fast]`) all match their bare family,
/// while a near-miss that merely shares a textual prefix (`gpt-5.56-sol` vs
/// `gpt-5.5`; `gpt-5.6-solar` vs `gpt-5.6-sol`) does not. `pub(crate)` so the
/// rest of the `api` crate (e.g. `providers::mod`) shares this ONE matcher
/// instead of a narrower ad hoc reimplementation — see
/// `providers::mod::model_matches_family`, which this superseded.
pub(crate) fn model_id_matches_family(model: &str, prefix: &str) -> bool {
    model == prefix
        || model
            .strip_prefix(prefix)
            .is_some_and(|suffix| matches!(suffix.as_bytes().first(), Some(b'-' | b'@' | b'[')))
}

#[cfg(test)]
mod model_id_matches_family_tests {
    use super::model_id_matches_family;

    #[test]
    fn dated_and_suffixed_ids_match_their_bare_family() {
        assert!(model_id_matches_family("gpt-5.6-sol-2026-07-09", "gpt-5.6-sol"));
        assert!(model_id_matches_family("gpt-5.6-terra@openai", "gpt-5.6-terra"));
        assert!(model_id_matches_family("gpt-5.6-terra[fast]", "gpt-5.6-terra"));
        assert!(model_id_matches_family("gpt-5.6-sol", "gpt-5.6-sol"));
    }

    #[test]
    fn near_miss_ids_do_not_match() {
        // A longer version digit run must not be swallowed by a shorter family.
        assert!(!model_id_matches_family("gpt-5.56-sol", "gpt-5.5"));
        // A distinct model that merely shares a textual prefix must not match.
        assert!(!model_id_matches_family("gpt-5.6-solar", "gpt-5.6-sol"));
    }
}

/// Whether an adaptive Anthropic `model` id accepts the `xhigh` reasoning-effort
/// tier on `output_config.effort`.
///
/// Sonnet and Haiku do **not**: their endpoint 400s with `This model does not
/// support effort level 'xhigh'` (supported: `low|medium|high|max`). Only the
/// Opus family (and Fable, which inherits the Opus scale) accepts `xhigh`.
/// Detected by family name so aliases and dated ids (`sonnet`,
/// `claude-sonnet-5`, `claude-haiku-4-5-20251001`) are all covered. This is
/// the Anthropic analogue of [`gpt_model_accepts_xhigh`]; the budget→effort and
/// explicit-effort paths both key off it via
/// [`EffortLevel::anthropic_for_model`].
#[must_use]
pub fn anthropic_model_accepts_xhigh(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    !(lower.contains("sonnet") || lower.contains("haiku"))
}

impl MessageRequest {
    #[must_use]
    pub fn reasoning_request(&self) -> ReasoningRequest {
        if let Some(effort) = self.effort {
            return ReasoningRequest::Effort(effort);
        }
        self.thinking
            .as_ref()
            .and_then(|thinking| thinking.budget_tokens)
            .filter(|&budget| budget > 0)
            .map_or(ReasoningRequest::Auto, ReasoningRequest::BudgetTokens)
    }

    #[must_use]
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: Vec<InputContentBlock>,
    /// Provider-opaque per-turn reasoning signature (Gemini `thoughtSignature`),
    /// carried from the session so the Gemini encoder can echo it back on the
    /// matching `functionCall`. Never serialized to the wire (`serde(skip)`):
    /// only the Gemini backend reads it, by field access, so it cannot leak into
    /// an Anthropic/OpenAI request even when those encoders serialize
    /// `InputMessage` directly.
    #[serde(skip)]
    pub thought_signature: Option<String>,
    /// Provider-opaque ChatGPT/Codex reasoning-replay payload for this turn
    /// (see [`core_types::ConversationMessage::reasoning_replay`]). Never
    /// serialized to the wire (`serde(skip)`): only the ChatGPT backend
    /// reads it, by field access, so it cannot leak into an
    /// Anthropic/Gemini request even when those encoders serialize
    /// `InputMessage` directly.
    #[serde(skip)]
    pub reasoning_replay: Option<Value>,
}

impl InputMessage {
    #[must_use]
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![InputContentBlock::Text {
                text: text.into(),
                cache_control: None,
            }],
            thought_signature: None,
            reasoning_replay: None,
        }
    }

    /// Build a user message that contains image blocks followed by a text prompt.
    #[must_use]
    pub fn user_with_images(text: impl Into<String>, images: Vec<ImageSource>) -> Self {
        let mut content: Vec<InputContentBlock> = images
            .into_iter()
            .map(|source| InputContentBlock::Image {
                source,
                cache_control: None,
            })
            .collect();
        content.push(InputContentBlock::Text {
            text: text.into(),
            cache_control: None,
        });
        Self {
            role: "user".to_string(),
            content,
            thought_signature: None,
            reasoning_replay: None,
        }
    }

    #[must_use]
    pub fn user_tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![InputContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: vec![ToolResultContentBlock::Text {
                    text: content.into(),
                }],
                is_error,
                            cache_control: None,
            }],
            thought_signature: None,
            reasoning_replay: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContentBlock>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Replayed Anthropic reasoning block (extended / interleaved thinking).
    /// Serializes to the wire `{"type":"thinking","thinking":...,"signature":...}`
    /// shape and MUST be sent verbatim with its signature — the API 400s on a
    /// modified or mis-ordered thinking block. Produced only by the runtime's
    /// `convert_blocks` for the Anthropic path, from a stored *signed* thinking
    /// block (unsigned / legacy blocks are dropped before lowering). The OpenAI
    /// and Gemini encoders match this variant explicitly and drop it, so a
    /// thinking block can never cross into a non-Anthropic request.
    Thinking {
        thinking: String,
        signature: String,
    },
    /// Replayed Anthropic `redacted_thinking` block. Serializes to
    /// `{"type":"redacted_thinking","data":...}`; Anthropic-only, same isolation
    /// as [`Self::Thinking`].
    RedactedThinking {
        data: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ToolLedgerView<'a> {
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: &'a Value,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: &'a [ToolResultContentBlock],
        is_error: bool,
    },
}

impl<'a> ToolLedgerView<'a> {
    #[must_use]
    pub fn from_input_block(block: &'a InputContentBlock) -> Option<Self> {
        match block {
            InputContentBlock::ToolUse { id, name, input, .. } => {
                Some(Self::ToolUse { id, name, input })
            }
            InputContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => Some(Self::ToolResult {
                tool_use_id,
                content,
                is_error: *is_error,
            }),
            InputContentBlock::Text { .. }
            | InputContentBlock::Image { .. }
            | InputContentBlock::Document { .. }
            | InputContentBlock::Thinking { .. }
            | InputContentBlock::RedactedThinking { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub kind: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    Text {
        text: String,
    },
    Json {
        value: Value,
    },
    /// An image a tool produced for the model to see. Serializes to the
    /// Anthropic `{"type":"image","source":{"type":"base64",...}}` shape.
    Image {
        source: ImageSource,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Server-side context edits Anthropic applied before inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextManagementResponse {
    #[serde(default)]
    pub applied_edits: Vec<AppliedContextEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedContextEdit {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub cleared_tool_uses: Option<u64>,
    #[serde(default)]
    pub cleared_thinking_turns: Option<u64>,
    #[serde(default)]
    pub cleared_input_tokens: Option<u64>,
}

impl ContextManagementResponse {
    #[must_use]
    pub fn cleared_tool_uses(&self) -> u64 {
        self.applied_edits
            .iter()
            .filter_map(|edit| edit.cleared_tool_uses)
            .sum()
    }

    #[must_use]
    pub fn cleared_input_tokens(&self) -> u64 {
        self.applied_edits
            .iter()
            .filter_map(|edit| edit.cleared_input_tokens)
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: String,
    pub content: Vec<OutputContentBlock>,
    pub model: String,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
    #[serde(default)]
    pub request_id: Option<String>,
    /// Provider-opaque reasoning signature for this assistant turn (Gemini 3's
    /// `thoughtSignature`, parsed from the first `functionCall`). Flows into the
    /// session and is echoed back on the next same-provider request; absent for
    /// providers that don't emit one. See [`InputMessage::thought_signature`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Provider-opaque ChatGPT/Codex reasoning-replay payload assembled for
    /// this assistant turn (the Responses reasoning items that precede each
    /// `function_call`, keyed by `call_id`), for a non-streaming response.
    /// Absent for providers that don't emit one. See
    /// [`InputMessage::reasoning_replay`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_replay: Option<Value>,
    /// Context edits the provider applied to this request, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<ContextManagementResponse>,
}

impl MessageResponse {
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.usage.total_tokens()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    #[must_use]
    pub const fn total_tokens(&self) -> u32 {
        self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
    }

    #[must_use]
    pub const fn token_usage(&self) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
        }
    }

    #[must_use]
    pub fn estimated_cost_usd(&self, model: &str) -> UsageCostEstimate {
        let usage = self.token_usage();
        pricing_for_model(model).map_or_else(
            || usage.estimate_cost_usd(),
            |pricing| usage.estimate_cost_usd_with_pricing(pricing),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageStartEvent {
    pub message: MessageResponse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDeltaEvent {
    pub delta: MessageDelta,
    pub usage: Usage,
    /// Context edits applied to this streaming request, reported on the final delta.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<ContextManagementResponse>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// End-of-turn reasoning signature (Gemini `thoughtSignature`), surfaced on
    /// the closing delta when the backend streams incrementally — at
    /// `message_start` time the signature isn't known yet because it rides on the
    /// late `functionCall` parts. Non-streaming/other providers leave it `None`
    /// (the signature, if any, arrives on `message_start` instead). See
    /// [`MessageResponse::thought_signature`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// End-of-turn ChatGPT/Codex reasoning-replay payload, assembled once
    /// the authoritative `response.completed` / `response.incomplete` output
    /// array is available (a streamed turn only knows the full set of
    /// `function_call`s and their preceding reasoning items at that point).
    /// See [`MessageResponse::reasoning_replay`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_replay: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockStartEvent {
    pub index: u32,
    pub content_block: OutputContentBlock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockDeltaEvent {
    pub index: u32,
    pub delta: ContentBlockDelta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentBlockStopEvent {
    pub index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageStopEvent {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart(MessageStartEvent),
    MessageDelta(MessageDeltaEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    MessageStop(MessageStopEvent),
}

#[cfg(test)]
mod tests {
    use core_types::format_usd;

    use super::*;

    #[test]
    fn input_message_never_serializes_thought_signature() {
        // Provider isolation: a Gemini `thoughtSignature` stored on an
        // InputMessage must never appear when a non-Gemini backend serializes
        // the request body. `serde(skip)` keeps it out of the wire entirely, so
        // a signature minted by Gemini cannot leak into a Claude/GPT request.
        let message = InputMessage {
            role: "assistant".to_string(),
            content: Vec::new(),
            thought_signature: Some("SECRET_SIG".to_string()),
            reasoning_replay: Some(serde_json::json!([{"call_id": "c1", "items": ["SECRET_REPLAY"]}])),
        };
        let wire = serde_json::to_string(&message).unwrap();
        assert!(
            !wire.contains("SECRET_SIG"),
            "signature leaked to wire: {wire}"
        );
        assert!(!wire.contains("thought_signature"));
        assert!(
            !wire.contains("SECRET_REPLAY"),
            "reasoning replay leaked to wire: {wire}"
        );
        assert!(!wire.contains("reasoning_replay"));
    }

    #[test]
    fn tool_ledger_view_projects_tool_use_without_cloning_wire_shape() {
        let input = serde_json::json!({ "path": "src/lib.rs" });
        let block = InputContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            input,
                    cache_control: None,
        };

        match ToolLedgerView::from_input_block(&block).expect("tool view") {
            ToolLedgerView::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "read_file");
                assert_eq!(input, &serde_json::json!({ "path": "src/lib.rs" }));
            }
            ToolLedgerView::ToolResult { .. } => panic!("expected tool use view"),
        }
    }

    #[test]
    fn tool_ledger_view_projects_tool_result_metadata_and_content() {
        let block = InputContentBlock::ToolResult {
            tool_use_id: "call_2".to_string(),
            content: vec![
                ToolResultContentBlock::Text {
                    text: "hello".to_string(),
                },
                ToolResultContentBlock::Json {
                    value: serde_json::json!({ "ok": true }),
                },
            ],
            is_error: true,
                    cache_control: None,
        };

        match ToolLedgerView::from_input_block(&block).expect("tool view") {
            ToolLedgerView::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "call_2");
                assert!(is_error);
                assert_eq!(content.len(), 2);
                assert_eq!(
                    content[1],
                    ToolResultContentBlock::Json {
                        value: serde_json::json!({ "ok": true }),
                    }
                );
            }
            ToolLedgerView::ToolUse { .. } => panic!("expected tool result view"),
        }
    }

    #[test]
    fn tool_ledger_view_ignores_non_tool_blocks() {
        let block = InputContentBlock::Text {
            text: "plain text".to_string(),
            cache_control: None,
        };
        assert!(ToolLedgerView::from_input_block(&block).is_none());
    }

    #[test]
    fn effort_level_wire_strings_per_provider() {
        // Each provider clamps the shared low..max scale to its own ceiling.
        // Anthropic keeps the full scale; model-agnostic GPT remains conservative
        // at xhigh; Gemini 3 tops out at high (and has no numeric budget).
        assert_eq!(EffortLevel::Max.anthropic(), "max");
        assert_eq!(EffortLevel::Max.gpt(), "xhigh");
        assert_eq!(EffortLevel::Xhigh.gpt(), "xhigh");
        assert_eq!(EffortLevel::Max.gemini(), "high");
        assert_eq!(EffortLevel::Xhigh.gemini(), "high");
        assert_eq!(EffortLevel::High.gemini(), "high");
        assert_eq!(EffortLevel::Medium.gemini(), "medium");
        assert_eq!(EffortLevel::Low.gemini(), "low");
    }

    #[test]
    fn gpt_for_model_projects_internal_top_tiers_to_xhigh() {
        // `/fast` is a serving-priority signal, not a reasoning-effort ceiling.
        // The provider's wire enum still tops out at xhigh for every GPT model;
        // Max/Ultra are Zo-side selection tiers, not literal wire values.
        assert_eq!(EffortLevel::Xhigh.gpt_for_model("gpt-5.5"), "xhigh");
        assert_eq!(EffortLevel::Max.gpt_for_model("gpt-5.5"), "xhigh");
        assert_eq!(EffortLevel::Xhigh.gpt_for_model("gpt-5.5-fast"), "xhigh");
        assert_eq!(EffortLevel::Max.gpt_for_model("gpt-5.5-fast"), "xhigh");
        assert_eq!(
            EffortLevel::Xhigh.gpt_for_model("gpt-5.5-2026-04-23-fast"),
            "xhigh"
        );
        assert_eq!(
            EffortLevel::Max.gpt_for_model("gpt-5.5-2026-04-23-fast"),
            "xhigh"
        );
        assert_eq!(EffortLevel::High.gpt_for_model("gpt-5.5-fast"), "high");
        assert_eq!(EffortLevel::Medium.gpt_for_model("gpt-5.5-fast"), "medium");
        assert_eq!(EffortLevel::Low.gpt_for_model("gpt-5.5-fast"), "low");
        assert_eq!(
            EffortLevel::Xhigh.gpt_for_model("gpt-5.3-codex-spark"),
            "xhigh"
        );
        assert_eq!(
            EffortLevel::Max.gpt_for_model("gpt-5.3-codex-spark"),
            "xhigh"
        );
        for model in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
            assert_eq!(EffortLevel::Xhigh.gpt_for_model(model), "xhigh", "{model}");
            assert_eq!(EffortLevel::Max.gpt_for_model(model), "xhigh", "{model}");
            assert_eq!(EffortLevel::Ultra.gpt_for_model(model), "xhigh", "{model}");
        }
        assert!(gpt_model_accepts_xhigh("gpt-5.5"));
        assert!(gpt_model_accepts_xhigh("gpt-5.5-fast"));
        assert!(gpt_model_accepts_xhigh("gpt-5.3-codex-spark"));
        assert!(gpt_model_accepts_max("gpt-5.6-sol"));
        assert!(gpt_model_accepts_max("gpt-5.6-terra"));
        assert!(gpt_model_accepts_max("gpt-5.6-luna"));
        assert!(!gpt_model_accepts_max("gpt-5.5"));
        assert!(!gpt_model_accepts_max("gpt-5.5-fast"));
        assert!(!gpt_model_accepts_max("gpt-5.3-codex-spark"));
    }

    #[test]
    fn anthropic_for_model_clamps_xhigh_on_sonnet_and_haiku() {
        use crate::types::anthropic_model_accepts_xhigh;
        // Regression: sub-agents inherit an `Xhigh` budget (the 16k preset, e.g.
        // a `deep-research` agent, or an opus→sonnet starvation demotion), but
        // adaptive Sonnet/Haiku reject `xhigh` on output_config.effort
        // (`This model does not support effort level 'xhigh'`), which 400'd and
        // killed the spawn. Sonnet/Haiku must clamp `xhigh` down to their real
        // ceiling, `high` — NOT `max` (never send a higher effort than asked).
        // Opus/Fable accept `xhigh` and keep it.
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("claude-opus-4-8"),
            EffortLevel::Xhigh
        );
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("opus"),
            EffortLevel::Xhigh
        );
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("claude-fable-5"),
            EffortLevel::Xhigh
        );
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("sonnet"),
            EffortLevel::High
        );
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("claude-sonnet-5"),
            EffortLevel::High
        );
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("haiku"),
            EffortLevel::High
        );
        assert_eq!(
            EffortLevel::Xhigh.anthropic_for_model("claude-haiku-4-5-20251001"),
            EffortLevel::High
        );
        // `max` is accepted by Sonnet (the 400's supported set lists it), so it
        // must pass through untouched — only `xhigh` is the gap.
        assert_eq!(
            EffortLevel::Max.anthropic_for_model("sonnet"),
            EffortLevel::Max
        );
        // Lower tiers are unaffected on every model.
        assert_eq!(
            EffortLevel::High.anthropic_for_model("sonnet"),
            EffortLevel::High
        );
        assert_eq!(
            EffortLevel::Medium.anthropic_for_model("haiku"),
            EffortLevel::Medium
        );
        assert_eq!(
            EffortLevel::Low.anthropic_for_model("sonnet"),
            EffortLevel::Low
        );
        // The shared predicate the wire path keys off.
        assert!(anthropic_model_accepts_xhigh("claude-opus-4-8"));
        assert!(anthropic_model_accepts_xhigh("claude-fable-5"));
        assert!(!anthropic_model_accepts_xhigh("claude-sonnet-5"));
        assert!(!anthropic_model_accepts_xhigh("haiku"));
    }

    fn request_for_reasoning(
        effort: Option<EffortLevel>,
        thinking: Option<ThinkingConfig>,
    ) -> MessageRequest {
        MessageRequest {
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            messages: vec![InputMessage::user_text("hi")],
            system: None,
            tools: None,
            tool_choice: None,
            stream: true,
            thinking,
            output_config: None,
            effort,
            effort_band_ceiling: None,
        }
    }

    #[test]
    fn reasoning_request_prefers_explicit_effort_over_budget() {
        let request =
            request_for_reasoning(Some(EffortLevel::Max), Some(ThinkingConfig::enabled(1_000)));
        assert_eq!(
            request.reasoning_request(),
            ReasoningRequest::Effort(EffortLevel::Max)
        );
    }

    #[test]
    fn reasoning_request_uses_positive_budget_when_effort_absent() {
        let request = request_for_reasoning(None, Some(ThinkingConfig::enabled(16_000)));
        assert_eq!(
            request.reasoning_request(),
            ReasoningRequest::BudgetTokens(16_000)
        );
    }

    #[test]
    fn reasoning_request_treats_zero_or_absent_budget_as_auto() {
        let zero = request_for_reasoning(None, Some(ThinkingConfig::enabled(0)));
        let absent = request_for_reasoning(None, None);
        assert_eq!(zero.reasoning_request(), ReasoningRequest::Auto);
        assert_eq!(absent.reasoning_request(), ReasoningRequest::Auto);
    }

    #[test]
    fn usage_total_tokens_includes_cache_tokens() {
        let usage = Usage {
            input_tokens: 10,
            cache_creation_input_tokens: 2,
            cache_read_input_tokens: 3,
            output_tokens: 4,
        };

        assert_eq!(usage.total_tokens(), 19);
        assert_eq!(usage.token_usage().total_tokens(), 19);
    }

    #[test]
    fn context_management_applied_edits_deserialize_on_stream_delta() {
        let event: MessageDeltaEvent = serde_json::from_value(serde_json::json!({
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "context_management": {
                "applied_edits": [{
                    "type": "clear_tool_uses_20250919",
                    "cleared_tool_uses": 5,
                    "cleared_input_tokens": 24_000
                }]
            }
        }))
        .expect("context-management delta");

        let context_management = event.context_management.expect("applied edits");
        assert_eq!(context_management.cleared_tool_uses(), 5);
        assert_eq!(context_management.cleared_input_tokens(), 24_000);
    }

    #[test]
    fn message_response_estimates_cost_from_model_usage() {
        let response = MessageResponse {
            id: "msg_cost".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: Vec::new(),
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1_000_000,
                cache_creation_input_tokens: 100_000,
                cache_read_input_tokens: 200_000,
                output_tokens: 500_000,
            },
            request_id: None,
            thought_signature: None,
            reasoning_replay: None,
            context_management: None,
        };

        // Sonnet 4.x prices at $3/$15 (+$3.75 cache-write, $0.30 cache-read),
        // its own tier — not the opus/default $15/$75 it used to fall through to.
        // 1M·$3 + 0.5M·$15 + 0.1M·$3.75 + 0.2M·$0.30 = $10.935.
        let cost = response.usage.estimated_cost_usd(&response.model);
        assert_eq!(format_usd(cost.total_cost_usd()), "$10.9350");
        assert_eq!(response.total_tokens(), 1_800_000);
    }

    #[test]
    fn cache_control_ephemeral_serializes_correctly() {
        let cc = CacheControl::ephemeral();
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json, serde_json::json!({"type": "ephemeral"}));
    }

    #[test]
    fn system_block_text_round_trips() {
        let block = SystemBlock::text("hello");
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json, serde_json::json!({"type": "text", "text": "hello"}));
        let back: SystemBlock = serde_json::from_value(json).unwrap();
        assert_eq!(back, block);
    }

    #[test]
    fn system_block_with_cache_control_round_trips() {
        let block = SystemBlock::Text {
            text: "cached".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}})
        );
        let back: SystemBlock = serde_json::from_value(json).unwrap();
        assert_eq!(back, block);
    }

    #[test]
    fn system_from_string_builds_single_text_block() {
        let blocks = system_from_string("test prompt");
        assert_eq!(blocks.len(), 1);
        assert_eq!(
            blocks[0],
            SystemBlock::Text {
                text: "test prompt".to_string(),
                cache_control: None,
            }
        );
    }

    #[test]
    fn deserialize_system_from_plain_string() {
        let json = serde_json::json!({
            "model": "claude-opus-4-6",
            "max_tokens": 64,
            "messages": [],
            "system": "plain string"
        });
        let req: MessageRequest = serde_json::from_value(json).unwrap();
        assert_eq!(
            req.system,
            Some(vec![SystemBlock::Text {
                text: "plain string".to_string(),
                cache_control: None,
            }])
        );
    }

    #[test]
    fn deserialize_system_from_array() {
        let json = serde_json::json!({
            "model": "claude-opus-4-6",
            "max_tokens": 64,
            "messages": [],
            "system": [
                {"type": "text", "text": "block one"},
                {"type": "text", "text": "block two", "cache_control": {"type": "ephemeral"}}
            ]
        });
        let req: MessageRequest = serde_json::from_value(json).unwrap();
        let blocks = req.system.unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[0],
            SystemBlock::Text {
                text: "block one".to_string(),
                cache_control: None,
            }
        );
        assert_eq!(
            blocks[1],
            SystemBlock::Text {
                text: "block two".to_string(),
                cache_control: Some(CacheControl::ephemeral()),
            }
        );
    }

    #[test]
    fn tool_choice_none_serializes_correctly() {
        let tc = ToolChoice::None;
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json, serde_json::json!({"type": "none"}));
        let back: ToolChoice = serde_json::from_value(json).unwrap();
        assert_eq!(back, ToolChoice::None);
    }

    #[test]
    fn input_content_block_text_with_cache_control() {
        let block = InputContentBlock::Text {
            text: "hello".to_string(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}})
        );
    }

    #[test]
    fn input_content_block_text_omits_cache_control_when_none() {
        let block = InputContentBlock::Text {
            text: "hello".to_string(),
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json, serde_json::json!({"type": "text", "text": "hello"}));
    }

    #[test]
    fn document_source_base64_round_trips() {
        let block = InputContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: "AAAA".to_string(),
            },
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "AAAA"}
            })
        );
        let back: InputContentBlock = serde_json::from_value(json).unwrap();
        assert_eq!(back, block);
    }
}
