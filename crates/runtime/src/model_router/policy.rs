use super::inventory::{ModelDescriptor, ModelInventory};
use super::learned::LearnedSpecialtyHint;
use super::outcome::CONFIDENT_DECISIVE_SAMPLES;
use super::selector::{select_model, RoleOverride, RoleSelector};
use super::target::{RouteRole, RoutingTarget};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterMode { MainOnly, Manual, Auto }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FreshnessPolicy { Latest, #[default] LatestStable }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelCapability {
    Default,
    Fast,
    Coding,
    Debugging,
    Verification,
    Analysis,
    Writing,
    Design,
    StructuredOutput,
    ToolUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelTier { Fast, Balanced, Strong, Deep }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus { Stable, Preview, Deprecated, Retired }

/// Provider-declared reasoning-effort ceiling for a model, mirrored locally
/// from `api::EffortLevel` so the router core stays free of a dependency on
/// the `api` crate (design principle: engine purity — no provider crate types
/// in the pure core, only in the edge adapter that builds descriptors,
/// `crate::model_inventory`). Only the top of the scale matters for routing
/// (whether a model clears the Ultra bar for Deep-tier promotion), so `Low`/
/// `Medium` collapse into the conservative `High` floor rather than getting
/// their own variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum EffortCeiling {
    #[default]
    High,
    Xhigh,
    Max,
    Ultra,
}

/// Provenance of a model's tier assignment, so `/smart doctor` (Phase 7) can
/// audit *why* a model landed in a tier instead of treating every entry as an
/// equally-confident fact (design principle: cold-start priors must be
/// labeled and self-retiring once learned data exists).
///
/// - `Fallback` — no capability data at all; the tier was guessed from a
///   marketing-name token (`is_deep_flagship`) or a generic frontier-family
///   grant. Weakest justification.
/// - `ColdStartPrior` — derived from a real provider-declared capability fact
///   (today: `effort_ceiling == Ultra` grants Deep) but still a static rule,
///   not outcome data — Phase 6 learned demotion can override it.
/// - `ProviderDeclared` — the model's tier set was derived directly from the
///   provider's OWN stated lineup positioning (`api::declared_model_class`:
///   e.g. OpenAI's Codex model cache "Latest frontier agentic coding model",
///   or Anthropic's public Mythos-class-above-Opus positioning), not a name-
///   token guess or an effort-ceiling side-effect. Checked FIRST in
///   `model_inventory::tiers_for_model` — the strongest static justification,
///   though still overridable by Phase 6 learned data once it exists.
/// - `Learned` — set from accumulated outcome data. Not produced anywhere
///   yet (Phase 6); the variant exists now so the field/enum shape is fixed
///   before that consumer lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TiersProvenance {
    #[default]
    Fallback,
    ColdStartPrior,
    ProviderDeclared,
    Learned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteTaskRisk { Low, Medium, High, Critical, #[default] Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteTaskComplexity { Trivial, Small, Medium, Large, #[default] Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteTaskKind {
    #[default]
    Default,
    Fast,
    Coding,
    Debugging,
    Verification,
    Review,
    Analysis,
    Research,
    Writing,
    Design,
    Judge,
    Synthesis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteContextNeed { None, LocalFiles, MultiFile, WholeRepo, #[default] Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteToolNeed { None, ReadOnly, Write, Shell, Network, #[default] Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteOutputNeed { FreeText, Structured, Patch, TestEvidence }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteVerificationNeed { None, Focused, Full, #[default] Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDiversityNeed { None, Helpful, Required }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteConfidence { High, Medium, Low }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteSignalSource {
    SubagentType,
    DescriptionKeyword,
    PromptKeyword,
    ToolSchema,
    WorkflowContext,
    UserDirective,
    /// A model-verbalized difficulty/risk self-assessment from the routing
    /// probe (`smart.autoClassifier: "probed"`): the only signal source that
    /// comes from a live model read of the task rather than static text
    /// matching. Bounded on purpose — fusion clamps its complexity influence
    /// to ±1 band around the deterministic classifier (see
    /// `fuse_probe_assessment`), so a hallucinated probe can never swing a
    /// trivial task to the Deep tier or vice versa.
    SelfAssessment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteAutoClassifierMode {
    Off,
    #[default]
    Deterministic,
    Assisted,
    /// Deterministic classification plus a single bounded Fast-tier model
    /// probe (~200 output tokens) whose verbalized `{complexity, risk,
    /// confidence}` self-assessment is fused on top of the keyword verdict.
    /// The deterministic classifier always runs first and remains the
    /// fallback whenever the probe fails, times out, or returns malformed
    /// JSON — this mode can only refine, never replace, the provider-free
    /// path.
    Probed,
}

impl RouteAutoClassifierMode {
    #[must_use]
    pub fn from_settings_value(value: Option<&serde_json::Value>) -> Self {
        let Some(value) = value.and_then(serde_json::Value::as_str).map(str::trim) else {
            return Self::Deterministic;
        };
        if value == "off" {
            Self::Off
        } else if value == "assisted" {
            Self::Assisted
        } else if value == "probed" {
            Self::Probed
        } else {
            Self::Deterministic
        }
    }

    #[must_use]
    pub fn audit_note(self) -> &'static str {
        match self {
            Self::Off => "smart-auto-classifier:off-provider-free-deterministic-active",
            Self::Deterministic => "smart-auto-classifier:deterministic-provider-free",
            Self::Assisted => "smart-auto-classifier:assisted-provider-free-deterministic",
            Self::Probed => "smart-auto-classifier:probed-model-fused-deterministic-floor",
        }
    }

    #[must_use]
    pub fn status_label(self) -> &'static str {
        match self {
            Self::Off => "off (deterministic fallback active)",
            Self::Deterministic => "deterministic",
            Self::Assisted => "assisted (provider-free deterministic)",
            Self::Probed => "probed (model self-assessment fused over deterministic)",
        }
    }
}

/// `smart.policy` — the execution-contract flavor Smart routing runs under.
///
/// `Architect` is the multi-model role-separation contract: reserved deep
/// reasoning models (see [`is_reserved_orchestrator_model`]) plan, orchestrate,
/// and verify, while implementation routes to standard implementer models. It
/// changes exactly two pure rules: [`implementation_route_model_allowed`]
/// drops the `complexity == Large` escape (a keyword-heuristic
/// misclassification could hand implementation to a reserved model — one of
/// the observed "routes wherever it wants" sources), and the
/// Verifier/Reviewer ladder in [`auto_selectors_for_role`] tries the Deep
/// rung first so the checker outclasses the implementer.
///
/// The TYPE default is `Classic` so an uninjected/default-constructed context
/// is byte-identical to pre-contract routing; the SETTINGS default is
/// `Architect` (see `tools::smart_setting_defaults` — absent key ⇒ Architect,
/// explicit `"classic"` or `ZO_SMART_POLICY=classic` opts out).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmartPolicy {
    /// Role-separation contract: reserved models plan/verify, implementers code.
    Architect,
    /// Pre-contract routing, byte-identical to before `smart.policy` existed.
    #[default]
    Classic,
}

impl SmartPolicy {
    /// Parse the `smart.policy` settings value. Absent or unrecognized ⇒ the
    /// documented live default `Architect` (NOT the type default — see the
    /// enum doc); only an explicit `"classic"` opts out. The
    /// `ZO_SMART_POLICY` env var, when set to a recognized value, wins over
    /// the settings key entirely (kill switch without a settings edit).
    #[must_use]
    pub fn from_settings_value(value: Option<&serde_json::Value>) -> Self {
        if let Ok(env) = std::env::var("ZO_SMART_POLICY") {
            match env.trim().to_ascii_lowercase().as_str() {
                "classic" => return Self::Classic,
                "architect" => return Self::Architect,
                _ => {}
            }
        }
        match value.and_then(serde_json::Value::as_str).map(str::trim) {
            Some("classic") => Self::Classic,
            _ => Self::Architect,
        }
    }

    /// Stable settings/wire token (`"architect"` / `"classic"`).
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Architect => "architect",
            Self::Classic => "classic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteShapeKind {
    Solo,
    OneSpecialist,
    SequentialWorkflow,
    ParallelLanes,
    RepairLoop,
    ParallelRepairLoop,
}

impl RouteShapeKind {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Solo => "solo",
            Self::OneSpecialist => "one-specialist",
            Self::SequentialWorkflow => "sequential-workflow",
            Self::ParallelLanes => "parallel-lanes",
            Self::RepairLoop => "repair-loop",
            Self::ParallelRepairLoop => "parallel-repair-loop",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneRouteMetadata {
    pub domain: String,
    pub lane_index: Option<usize>,
    pub lane_count: Option<usize>,
    pub merge_dependency: Option<String>,
}

impl LaneRouteMetadata {
    #[must_use]
    pub fn new(domain: impl Into<String>) -> Self {
        Self { domain: domain.into(), lane_index: None, lane_count: None, merge_dependency: None }
    }

    #[must_use]
    pub fn with_position(mut self, lane_index: usize, lane_count: usize) -> Self {
        self.lane_index = Some(lane_index);
        self.lane_count = Some(lane_count);
        self
    }

    #[must_use]
    pub fn with_merge_dependency(mut self, dependency: impl Into<String>) -> Self {
        self.merge_dependency = Some(dependency.into());
        self
    }
}

/// Bound on the outcome-feedback adjustment. Sized to **exceed** the recency
/// tiebreak (`release_rank`, capped at 100) but stay **under** the capability
/// (1000) and tier (300) gates: once enough real outcomes accumulate, learned
/// performance overrides the recency prior within a role's candidate pool, yet
/// feedback can never promote a model past the hard capability/tier requirement
/// it does not meet. (Was 50 — too small to ever beat recency, so the hardcoded
/// prior always won; this is the P3 reweight that lets the dynamic signal lead.)
pub(super) const MAX_FEEDBACK_ADJUSTMENT: i16 = 120;

/// Score bonus for a capability-tier selector match (`score_model_with_context`'s
/// `selector.tier` check) — the "tier gate" the cold-start specialty seed,
/// Phase 6's learned-specialty blend, and outcome feedback must all stay
/// safely under (see `learned_specialty_blend_worst_case_stays_under_the_tier_gate`
/// for the compile-time proof). Also reused, at the SAME value, by the
/// Default-role difficulty-tier bonus in `score_model_with_context` — that
/// call site's own comment already documented "same +300 weight as a
/// selector tier match" as an intentional invariant; naming the shared
/// constant makes the compiler enforce what was previously only a comment.
pub(super) const TIER_MATCH_BONUS: i32 = 300;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RouteFeedbackHint {
    pub enabled: bool,
    pub score_adjustment: i16,
    pub model_adjustments: Vec<(String, i16)>,
}

impl RouteFeedbackHint {
    #[must_use]
    pub fn disabled() -> Self { Self::default() }

    #[must_use]
    pub fn enabled(score_adjustment: i16) -> Self {
        Self { enabled: true, score_adjustment: score_adjustment.clamp(-MAX_FEEDBACK_ADJUSTMENT, MAX_FEEDBACK_ADJUSTMENT), model_adjustments: Vec::new() }
    }

    #[must_use]
    pub fn for_model(model: impl Into<String>, score_adjustment: i16) -> Self {
        let model = model.into();
        if model.trim().is_empty() {
            return Self::disabled();
        }
        Self {
            enabled: true,
            score_adjustment: 0,
            model_adjustments: vec![(model, score_adjustment.clamp(-MAX_FEEDBACK_ADJUSTMENT, MAX_FEEDBACK_ADJUSTMENT))],
        }
    }

    #[must_use]
    pub fn with_model_adjustment(mut self, model: impl Into<String>, score_adjustment: i16) -> Self {
        let model = model.into();
        if !model.trim().is_empty() {
            self.enabled = true;
            self.model_adjustments.push((model, score_adjustment.clamp(-MAX_FEEDBACK_ADJUSTMENT, MAX_FEEDBACK_ADJUSTMENT)));
        }
        self
    }

    #[must_use]
    pub fn bounded_adjustment(&self) -> i16 {
        if self.enabled { self.score_adjustment.clamp(-MAX_FEEDBACK_ADJUSTMENT, MAX_FEEDBACK_ADJUSTMENT) } else { 0 }
    }

    #[must_use]
    pub fn bounded_adjustment_for(&self, model_id: &str) -> i16 {
        if !self.enabled {
            return 0;
        }
        let model_specific: i16 = self
            .model_adjustments
            .iter()
            .filter(|(model, _)| model == model_id)
            .map(|(_, adjustment)| (*adjustment).clamp(-MAX_FEEDBACK_ADJUSTMENT, MAX_FEEDBACK_ADJUSTMENT))
            .sum();
        self.bounded_adjustment()
            .saturating_add(model_specific)
            .clamp(-MAX_FEEDBACK_ADJUSTMENT, MAX_FEEDBACK_ADJUSTMENT)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RoutePolicyContext {
    pub risk: RouteTaskRisk,
    pub complexity: RouteTaskComplexity,
    /// Number of prior implementation attempts that produced a real quality
    /// failure for this same work item. Provider throttling/retries never
    /// increment this counter. Premium implementation models may use it as an explicit
    /// escalation signal once ordinary implementers have failed repeatedly.
    pub prior_failures: u32,
    pub context_need: RouteContextNeed,
    pub tool_need: RouteToolNeed,
    pub verification_need: RouteVerificationNeed,
    pub route_shape: Option<RouteShapeKind>,
    pub lane: Option<LaneRouteMetadata>,
    pub allow_cross_provider_diversity: bool,
    /// Providers the AUTO selector may route to (`smart.providerAllowlist`).
    /// Empty (the default) allows every connected provider. Only auto
    /// candidates are constrained: an explicit model, a pin, and the
    /// main-model fallback stay untouched, so an allowlist that matches no
    /// candidate degrades to the parent model instead of failing the route.
    pub provider_allowlist: Vec<String>,
    pub feedback: RouteFeedbackHint,
    pub audit_notes: Vec<String>,
    /// Providers currently headroom-low (an active rate-limit cool-down, a
    /// recent 429, or OAuth subscription-window pressure), injected by the
    /// application layer (`smart_router::apply`) from the same governor state
    /// `agent_tools::rate_limit::rate_limit_headroom_low` reads. This is an
    /// OPERATIONAL fact, not a quality signal: the engine stays pure (no IO),
    /// it only ever sees the plain provider-name strings the caller computed.
    /// AUTO deprioritizes (never hard-filters) a candidate whose provider is
    /// in this set — see [`score_model_with_context`]'s cooldown penalty.
    pub cooldown_providers: Vec<String>,
    /// Per-provider *remaining* quota headroom (percent, `0..=100`), keyed by
    /// the same lowercase `ProviderKind::rate_limit_key` string
    /// [`Self::provider_in_cooldown`] matches — injected by the application
    /// layer (`smart_router::apply::provider_headroom_percents`) from
    /// `api::quota::provider_quota_views`. Plain operational data (engine
    /// purity: no IO, integers only — never an `f64`, so the struct stays
    /// `Eq`). Distinct from [`Self::cooldown_providers`]: cooldown is the
    /// binary "already throttled" (lagging) signal, this is the *graded*
    /// leading one — a provider at 8% remaining is penalized more than one at
    /// 22%. Empty (the default) is BYTE-IDENTICAL to pre-headroom routing —
    /// see [`score_model_with_context`]'s graded penalty and the
    /// `empty_headroom_set_is_byte_identical…` test.
    pub provider_headroom: Vec<(String, u8)>,
    /// Remaining-percent threshold below which [`Self::provider_headroom`]
    /// starts deducting score (`smart.headroomPenaltyThreshold`, default 25,
    /// injected by `apply.rs`). At/above it a provider is treated as healthy
    /// (no penalty); at `0%` remaining the penalty reaches the full
    /// [`COOLDOWN_DEPRIORITIZE_PENALTY`] (same ceiling as the binary cooldown).
    /// A `0` threshold disables the graded penalty entirely (a division guard),
    /// so a default-constructed context — empty `provider_headroom`, threshold
    /// `0` — is inert regardless.
    pub headroom_penalty_threshold: u8,
    /// Phase 5 deterministic exploration rotation slot, injected by the
    /// application layer (`smart_router::apply::exploration_slot_for` /
    /// [`exploration_slot_for_route`]) from the already-loaded outcome
    /// summary. `None` (the default) is byte-identical to pre-Phase-5
    /// routing — every existing caller that never sets this field sees no
    /// behavior change. `Some(slot)` only ever ROTATES the pick among
    /// candidates already in the winning selector rung (see
    /// `select_exploration_candidate`); it is never a score bonus, so it can
    /// never cross the capability/tier/rung gates.
    pub exploration_slot: Option<u32>,
    /// Per-canonical-model decisive-sample counts for the CURRENT `route_key`
    /// (`RouteOutcomeSummary::decisive_counts_for_route_key`), injected
    /// alongside `exploration_slot` so the engine can judge "under-sampled"
    /// (`< 2` decisive) with zero IO of its own (engine purity — mirrors how
    /// `cooldown_providers` carries an operational fact as plain data). A
    /// model absent from this list is treated as 0 decisive samples — the
    /// common brand-new/zero-history case this phase exists to fix.
    pub exploration_decisive_counts: Vec<(String, u32)>,
    /// Phase 6 learned-specialty hint, injected by `smart_router::apply`
    /// (application layer — engine purity, no IO in this module). Default
    /// (empty/[`LearnedSpecialtyHint::disabled`]) is BYTE-IDENTICAL to
    /// pre-Phase-6 routing: [`effective_specialty_adjustment`] falls back to
    /// [`cold_start_specialty_seed`] whenever this hint has no entry for a
    /// (role, model) pair. `smart.learnedSpecialty` gates what the caller
    /// injects here (`off`/`shadow` inject an empty hint into the REAL
    /// request; only `on` injects live data) — see that setting's doc in
    /// `tools::smart_router::settings`.
    pub learned_specialty: LearnedSpecialtyHint,
    /// The Smart execution-contract flavor this route runs under
    /// (`smart.policy`, injected by the application layer). The type default
    /// (`Classic`) keeps an uninjected context byte-identical to
    /// pre-contract routing. Deliberately NOT part of
    /// [`Self::is_default_policy`]: it is a session mode, not per-task
    /// context, so injecting it must not flip the "default task context"
    /// audit label.
    pub policy: SmartPolicy,
}

impl RoutePolicyContext {
    #[must_use]
    pub fn is_default_policy(&self) -> bool {
        self.risk == RouteTaskRisk::Unknown
            && self.complexity == RouteTaskComplexity::Unknown
            && self.prior_failures == 0
            && self.context_need == RouteContextNeed::Unknown
            && self.tool_need == RouteToolNeed::Unknown
            && self.verification_need == RouteVerificationNeed::Unknown
            && self.route_shape.is_none()
            && self.lane.is_none()
            && !self.allow_cross_provider_diversity
            && self.provider_allowlist.is_empty()
            && self.feedback == RouteFeedbackHint::default()
            && self.audit_notes.is_empty()
            && self.cooldown_providers.is_empty()
            && self.provider_headroom.is_empty()
            && self.exploration_slot.is_none()
            && self.exploration_decisive_counts.is_empty()
            && self.learned_specialty.is_empty()
    }

    #[must_use]
    pub fn provider_allowed(&self, provider: &str) -> bool {
        provider_allowlisted(&self.provider_allowlist, provider)
    }

    /// `true` when `provider` is in the caller-supplied cooldown set
    /// (case-insensitive, matching [`Self::provider_allowed`]'s convention).
    #[must_use]
    pub fn provider_in_cooldown(&self, provider: &str) -> bool {
        self.cooldown_providers
            .iter()
            .any(|candidate| candidate.trim().eq_ignore_ascii_case(provider))
    }

    /// Graded score penalty for a candidate whose provider's *remaining* quota
    /// headroom (from [`Self::provider_headroom`]) is below
    /// [`Self::headroom_penalty_threshold`]. Scales linearly from `0` at the
    /// threshold to the full [`COOLDOWN_DEPRIORITIZE_PENALTY`] at `0%` remaining
    /// (integer arithmetic; capped at that penalty so it can never exceed the
    /// binary-cooldown scale, keeping it well under the capability/tier gates —
    /// a squeezed provider is deprioritized, never disqualified). Returns `0` —
    /// byte-identical to pre-headroom routing — when `provider_headroom` has no
    /// entry for the provider, the remaining is at/above the threshold, or the
    /// threshold is `0` (feature off / division guard). Provider matching is
    /// case-insensitive, matching [`Self::provider_in_cooldown`].
    #[must_use]
    fn headroom_penalty_for(&self, provider: &str) -> i32 {
        let threshold = self.headroom_penalty_threshold;
        if threshold == 0 {
            return 0;
        }
        let Some(remaining) = self
            .provider_headroom
            .iter()
            .find(|(candidate, _)| candidate.trim().eq_ignore_ascii_case(provider))
            .map(|(_, remaining)| *remaining)
        else {
            return 0;
        };
        if remaining >= threshold {
            return 0;
        }
        let deficit = i32::from(threshold - remaining);
        (COOLDOWN_DEPRIORITIZE_PENALTY * deficit / i32::from(threshold))
            .min(COOLDOWN_DEPRIORITIZE_PENALTY)
    }
}

/// Shared allowlist predicate for the live route scorer and the
/// recommendation scorer (dashboard parity): empty list allows everything,
/// matching is case-insensitive on the provider name.
#[must_use]
pub(super) fn provider_allowlisted(allowlist: &[String], provider: &str) -> bool {
    allowlist.is_empty()
        || allowlist
            .iter()
            .any(|allowed| allowed.trim().eq_ignore_ascii_case(provider))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteRequest {
    pub role: RouteRole,
    pub target: RoutingTarget,
    pub main_model: String,
    pub explicit_model: Option<String>,
    pub mode: RouterMode,
    pub allow_fallback: bool,
    pub override_rule: Option<RoleOverride>,
    pub context: RoutePolicyContext,
}

impl RouteRequest {
    #[must_use]
    pub fn new(role: RouteRole, main_model: impl Into<String>) -> Self {
        Self {
            role,
            target: RoutingTarget::RoleFallback(role),
            main_model: main_model.into(),
            explicit_model: None,
            mode: RouterMode::Auto,
            allow_fallback: true,
            override_rule: None,
            context: RoutePolicyContext::default(),
        }
    }

    #[must_use]
    pub fn for_target(target: RoutingTarget, fallback_role: RouteRole, main_model: impl Into<String>) -> Self {
        Self {
            role: fallback_role,
            target,
            main_model: main_model.into(),
            explicit_model: None,
            mode: RouterMode::Auto,
            allow_fallback: true,
            override_rule: None,
            context: RoutePolicyContext::default(),
        }
    }

    #[must_use]
    pub fn with_context(mut self, context: RoutePolicyContext) -> Self {
        self.context = context;
        self
    }

    #[must_use]
    pub fn effective_role(&self) -> RouteRole { self.target.route_role_hint().unwrap_or(self.role) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDecisionSource {
    Explicit,
    MainOnly,
    Pinned,
    ManualSelector,
    AutoSelector,
    /// AUTO picked an under-sampled, same-rung rival via Phase 5's
    /// deterministic exploration rotation instead of the normal winner — see
    /// [`RoutePolicyContext::exploration_slot`] /
    /// `select_exploration_candidate`. Projected to the `"exploration"`
    /// `routeSource` value on the persisted outcome record (`apply.rs`'s
    /// `route_source_label`), distinct from `AutoSelector`'s `"auto"` so a
    /// later learning phase can tell an unforced best-of-breed pick apart
    /// from a deliberate cold-start sampling pick.
    Exploration,
    MainModelFallback,
    FallbackDisabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteAudit {
    pub reference_provider: Option<String>,
    pub selected_provider: Option<String>,
    pub cross_provider: Option<bool>,
    pub route_shape: Option<String>,
    pub lane_domain: Option<String>,
    pub guardrails: Vec<String>,
    pub feedback_adjustment: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub resolved_model: String,
    pub source: RouteDecisionSource,
    pub reason: String,
    pub audit: RouteAudit,
    /// Recommended reasoning-effort ceiling for this route, or `None` when the
    /// route carries no effort recommendation (the byte-identical default: a
    /// caller that ignores this field sees no behavior change). A PURE
    /// function of (role, task complexity, the resolved model's declared
    /// `effort_ceiling`) — see [`recommended_effort_for`]. Today the only rule
    /// is "Analysis-family role, Large complexity, Ultra-capable model ⇒
    /// recommend Ultra"; every other combination is `None`.
    pub recommended_effort: Option<EffortCeiling>,
}

#[must_use]
pub fn route_model(request: &RouteRequest, inventory: &ModelInventory) -> RouteDecision {
    if let Some(explicit) = request.explicit_model.as_ref().filter(|model| !model.is_empty()) {
        if model_allowed(explicit, request, inventory) {
            return decision(request, inventory, explicit.clone(), RouteDecisionSource::Explicit, "explicit model was provided");
        }
        return main_decision(request, inventory, RouteDecisionSource::MainModelFallback, "explicit model is outside usable inventory");
    }

    if request.mode == RouterMode::MainOnly {
        return main_decision(request, inventory, RouteDecisionSource::MainOnly, "router mainOnly mode");
    }

    match request.override_rule.as_ref() {
        Some(RoleOverride::Pin(model)) if !model.is_empty() => {
            if model_allowed(model, request, inventory) {
                // A pin may point outside `smart.providerAllowlist` — the
                // allowlist prefilters only the auto pool and a pin is
                // explicit user intent, so it still wins. But never silently:
                // name the escape in the reason (and the audit guardrails).
                let reason = if pin_escapes_provider_allowlist(model, request, inventory) {
                    "role exact pin (provider outside smart.providerAllowlist)"
                } else {
                    "role exact pin"
                };
                return decision(request, inventory, model.clone(), RouteDecisionSource::Pinned, reason);
            }
            if !request.allow_fallback {
                return main_decision(
                    request,
                    inventory,
                    RouteDecisionSource::FallbackDisabled,
                    "pinned model is outside usable inventory and fallback disabled",
                );
            }
        }
        Some(RoleOverride::Family(selector)) => {
            if let Some(model) = select_model(selector, inventory, |model| {
                route_candidate_allowed(model, request)
            }) {
                return decision(
                    request,
                    inventory,
                    model.id().to_string(),
                    RouteDecisionSource::ManualSelector,
                    "role family selector",
                );
            }
            if !request.allow_fallback {
                return main_decision(
                    request,
                    inventory,
                    RouteDecisionSource::FallbackDisabled,
                    "manual selector unavailable and fallback disabled",
                );
            }
        }
        Some(RoleOverride::Auto | RoleOverride::Pin(_)) | None => {}
    }

    if request.mode == RouterMode::Auto {
        let (winner, quota_degraded, exploration_fired) = select_auto_model_with_quota_note(request, inventory);
        if let Some(model) = winner {
            let source = if exploration_fired { RouteDecisionSource::Exploration } else { RouteDecisionSource::AutoSelector };
            let mut extra_guardrails: Vec<&str> = Vec::new();
            if quota_degraded {
                extra_guardrails.push("quota-degraded");
            }
            if exploration_fired {
                extra_guardrails.push("exploration");
            }
            return decision_with_note(
                request,
                inventory,
                model.id().to_string(),
                source,
                auto_reason(request),
                &extra_guardrails,
            );
        }
        if !request.allow_fallback {
            return main_decision(
                request,
                inventory,
                RouteDecisionSource::FallbackDisabled,
                "auto selector unavailable and fallback disabled",
            );
        }
    }

    main_decision(request, inventory, RouteDecisionSource::MainModelFallback, "main model fallback")
}

/// Ranked model fallbacks for a route decision, excluding the model that was
/// already selected. This is intentionally a *routing* fallback list, not a
/// provider retry budget: callers can try these when the selected provider is
/// rate-limited so a sub-agent does not sit parked on an exhausted quota window.
///
/// The scorer mirrors [`route_model`]: exact pins stay exact (no alternate),
/// manual selectors rank the same selector pool, and auto routes rank the same
/// capability/tier/context-filtered pool. Ties use the same "later catalog entry
/// wins" rule as `max_by_key`, so the first returned alternate is the real
/// second-best route under the current policy.
#[must_use]
pub fn route_model_fallback_candidates(
    request: &RouteRequest,
    inventory: &ModelInventory,
    selected_model: &str,
    limit: usize,
) -> Vec<String> {
    if limit == 0 || request.mode == RouterMode::MainOnly {
        return Vec::new();
    }

    let ranked = ranked_route_candidates(request, inventory);
    let selected = selected_model.trim();
    let mut fallbacks = Vec::new();
    for (model, _, _) in ranked {
        let id = model.id();
        if id == selected || fallbacks.iter().any(|existing| existing == id) {
            continue;
        }
        fallbacks.push(id.to_string());
        if fallbacks.len() >= limit {
            break;
        }
    }
    fallbacks
}

fn ranked_route_candidates<'a>(
    request: &RouteRequest,
    inventory: &'a ModelInventory,
) -> Vec<(&'a ModelDescriptor, i32, usize)> {
    let mut ranked = match request.override_rule.as_ref() {
        Some(RoleOverride::Pin(_)) => Vec::new(),
        Some(RoleOverride::Family(selector)) => ranked_selector_candidates(selector, inventory)
            .into_iter()
            .filter(|(model, _, _)| route_candidate_allowed(model, request))
            .collect(),
        Some(RoleOverride::Auto) | None => {
            if request.mode == RouterMode::Auto {
                ranked_auto_candidates(request, inventory)
            } else {
                Vec::new()
            }
        }
    };
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            // Preserve `Iterator::max_by_key` tie behavior: the later catalog entry
            // wins, so it must sort before an earlier equal-score entry.
            .then_with(|| right.2.cmp(&left.2))
    });
    ranked
}

fn ranked_selector_candidates<'a>(
    selector: &RoleSelector,
    inventory: &'a ModelInventory,
) -> Vec<(&'a ModelDescriptor, i32, usize)> {
    inventory
        .models()
        .iter()
        .enumerate()
        .filter(|(_, model)| selector_matches_context(model, selector))
        .map(|(index, model)| (model, selector_score(model, selector), index))
        .collect()
}

fn ranked_auto_candidates<'a>(
    request: &RouteRequest,
    inventory: &'a ModelInventory,
) -> Vec<(&'a ModelDescriptor, i32, usize)> {
    let selectors = auto_selectors_for_role(
        request.effective_role(),
        request.context.risk,
        request.context.complexity,
        request.context.policy,
    );
    let selector_count = selectors.len();
    let mut all = Vec::new();
    for (tier_offset, selector) in selectors.into_iter().enumerate() {
        let tier_penalty = selector_fallback_penalty(tier_offset);
        let mut ranked = ranked_context_candidates(&selector, request, inventory);
        if selector_count == 1 && !ranked.is_empty() {
            return ranked;
        }
        for (_, score, _) in &mut ranked {
            *score = score.saturating_sub(tier_penalty);
        }
        all.extend(ranked);
    }
    all
}

fn selector_fallback_penalty(tier_offset: usize) -> i32 {
    i32::try_from(tier_offset)
        .unwrap_or(i32::MAX / AUTO_SELECTOR_FALLBACK_PENALTY)
        .saturating_mul(AUTO_SELECTOR_FALLBACK_PENALTY)
}

fn ranked_context_candidates<'a>(
    selector: &RoleSelector,
    request: &RouteRequest,
    inventory: &'a ModelInventory,
) -> Vec<(&'a ModelDescriptor, i32, usize)> {
    inventory
        .models()
        .iter()
        .enumerate()
        .filter(|(_, model)| selector_matches_context(model, selector))
        .filter(|(_, model)| request.context.provider_allowed(model.provider()))
        .filter(|(_, model)| route_candidate_allowed(model, request))
        .map(|(index, model)| {
            (
                model,
                score_model_with_context(model, selector, request, inventory),
                index,
            )
        })
        .collect()
}

fn route_candidate_allowed(model: &ModelDescriptor, request: &RouteRequest) -> bool {
    !matches!(request.effective_role(), RouteRole::Coding | RouteRole::Debugging)
        || implementation_route_model_allowed(
            model.id(),
            request.context.complexity,
            request.context.prior_failures,
            request.context.policy,
        )
}

/// Keep the two premium reasoning-first models out of ordinary implementation
/// work. Under [`SmartPolicy::Classic`] they remain available to Smart AUTO
/// when the task is classified genuinely large or when two prior
/// implementation attempts for the same item have failed. Under
/// [`SmartPolicy::Architect`] the complexity escape is removed — a Large
/// keyword-misclassification must not hand implementation to a reserved
/// model; hard tasks get a reserved-model PLAN and a stronger VERIFY instead —
/// so only repeated real failures (`prior_failures >= 2`) escalate. Explicit
/// model choices and pins are handled before AUTO and remain explicit user
/// intent under both policies.
#[must_use]
pub fn implementation_route_model_allowed(
    model_id: &str,
    complexity: RouteTaskComplexity,
    prior_failures: u32,
    policy: SmartPolicy,
) -> bool {
    if !is_premium_implementation_escalation_model(model_id) {
        return true;
    }
    match policy {
        SmartPolicy::Architect => prior_failures >= 2,
        SmartPolicy::Classic => complexity == RouteTaskComplexity::Large || prior_failures >= 2,
    }
}

/// `true` for the reserved deep reasoning models the Architect contract keeps
/// on plan/orchestrate/verify duty (and out of ordinary implementation).
/// Public so the host's edit gate and HUD name the same set the router
/// enforces, instead of re-deriving it from tier heuristics.
#[must_use]
pub fn is_reserved_orchestrator_model(model_id: &str) -> bool {
    is_premium_implementation_escalation_model(model_id)
}

/// Built-in Architect PLAN/VERIFY pool used when `smart.deepTierModels` is
/// absent. Hosts may replace it per turn without changing the implementation
/// routing policy above.
pub const DEFAULT_DEEP_TIER_MODELS: [&str; 2] = ["claude-fable-5", "gpt-5.6-sol"];

/// Return the default Architect PLAN/VERIFY pool in preference order.
#[must_use]
pub fn default_deep_tier_models() -> Vec<String> {
    DEFAULT_DEEP_TIER_MODELS
        .iter()
        .map(|model| (*model).to_string())
        .collect()
}

/// Whether `model_id` belongs to the configured Architect PLAN/VERIFY pool.
/// Pool entries are already alias-resolved by the settings reader; resolving
/// the candidate too preserves the catalog's alias leniency, while the
/// provider-prefix comparison keeps future Claude short ids usable before a
/// catalog row ships.
#[must_use]
pub fn is_deep_tier_model(model_id: &str, pool: &[String]) -> bool {
    pool.iter()
        .any(|configured| deep_tier_model_matches(model_id, configured))
}

/// Whether `model_id` names the configured model under the routing alias,
/// family, and provider-prefix rules.
#[must_use]
pub fn deep_tier_model_matches(model_id: &str, configured: &str) -> bool {
    deep_tier_model_matches_one_way(model_id, configured)
}

fn is_premium_implementation_escalation_model(model_id: &str) -> bool {
    DEFAULT_DEEP_TIER_MODELS
        .iter()
        .any(|configured| deep_tier_model_matches(model_id, configured))
}

fn deep_tier_model_matches_one_way(model_id: &str, configured: &str) -> bool {
    let model = model_id.trim().to_ascii_lowercase();
    let configured = configured.trim().to_ascii_lowercase();
    if model.is_empty() || configured.is_empty() {
        return false;
    }

    let resolved_configured = api::resolve_model_alias(&configured).to_ascii_lowercase();
    let configured_leaf = resolved_configured
        .rsplit(['/', ':'])
        .next()
        .unwrap_or(&resolved_configured);
    if model_family_leaf_matches(&model, configured_leaf) {
        return true;
    }

    let model_leaf = model.rsplit(['/', ':']).next().unwrap_or(&model);
    if api::provider_catalog().iter().any(|entry| {
        entry
            .canonical_model_id
            .eq_ignore_ascii_case(configured_leaf)
            && model_alias_leaf_matches(model_leaf, entry.alias)
    }) {
        return true;
    }

    configured_leaf.strip_prefix("claude-").is_some_and(|short| {
        model_family_leaf_matches(&model, short)
            || model_alias_leaf_matches(model_leaf, &short.replace('-', ""))
    })
}

fn model_alias_leaf_matches(leaf: &str, alias: &str) -> bool {
    leaf == alias
        || leaf.strip_prefix(alias).is_some_and(|suffix| {
            matches!(suffix.as_bytes().first(), Some(b'@' | b'['))
        })
}

fn model_family_leaf_matches(model_id: &str, family: &str) -> bool {
    let leaf = model_id.rsplit(['/', ':']).next().unwrap_or(model_id);
    leaf == family
        || leaf.strip_prefix(family).is_some_and(|suffix| {
            matches!(suffix.as_bytes().first(), Some(b'-' | b'@' | b'['))
        })
}

fn selector_score(model: &ModelDescriptor, selector: &RoleSelector) -> i32 {
    let mut score = 0_i32;
    if selector.capability.is_some_and(|capability| model.has_capability(capability)) {
        score += 100_000;
    }
    if selector.tier.is_some_and(|tier| model.has_tier(tier)) {
        score += 50_000;
    }
    if selector.class.as_ref().is_some_and(|class| model.class_matches(class)) {
        score += 25_000;
    }
    score + i32::try_from(model.release_rank_value().min(10_000)).unwrap_or(0)
}

pub(crate) fn freshness_allows(status: ModelStatus, freshness: FreshnessPolicy) -> bool {
    match freshness {
        FreshnessPolicy::Latest => matches!(status, ModelStatus::Stable | ModelStatus::Preview),
        FreshnessPolicy::LatestStable => status == ModelStatus::Stable,
    }
}

fn model_allowed(model: &str, request: &RouteRequest, inventory: &ModelInventory) -> bool {
    model == request.main_model || inventory.find(model).is_some()
}

/// `true` when a pinned model resolves to a provider that
/// `smart.providerAllowlist` would exclude from the auto pool.
fn pin_escapes_provider_allowlist(
    model: &str,
    request: &RouteRequest,
    inventory: &ModelInventory,
) -> bool {
    if request.context.provider_allowlist.is_empty() {
        return false;
    }
    inventory
        .find(model)
        .is_some_and(|descriptor| !request.context.provider_allowed(descriptor.provider()))
}

fn main_decision(request: &RouteRequest, inventory: &ModelInventory, source: RouteDecisionSource, reason: &str) -> RouteDecision {
    decision(request, inventory, request.main_model.clone(), source, reason)
}

fn decision(request: &RouteRequest, inventory: &ModelInventory, resolved_model: String, source: RouteDecisionSource, reason: &str) -> RouteDecision {
    decision_with_note(request, inventory, resolved_model, source, reason, &[])
}

/// Same as [`decision`], with extra guardrail notes (today: the AUTO path's
/// `"quota-degraded"` and `"exploration"` stamps — every other call site
/// passes `&[]`, so this is byte-identical to the old `decision` for them).
fn decision_with_note(
    request: &RouteRequest,
    inventory: &ModelInventory,
    resolved_model: String,
    source: RouteDecisionSource,
    reason: &str,
    extra_guardrails: &[&str],
) -> RouteDecision {
    let mut audit = audit_decision(request, &resolved_model, inventory);
    audit.guardrails.extend(extra_guardrails.iter().map(|note| (*note).to_string()));
    let recommended_effort = recommended_effort_for_route(request, inventory, &resolved_model);
    // audit/effort는 bare 인벤토리 id로 계산을 마친 뒤에 승격한다 —
    // inventory.find는 `[fast]` 브래킷 id를 모른다.
    let resolved_model = propagate_fast_tier_to_terra(resolved_model, source, request);
    RouteDecision { audit, resolved_model, source, reason: reason.to_string(), recommended_effort }
}

/// 세션의 fast 상태를 라우터 픽에 전파한다(사용자 정책 2026-07-11 개정:
/// 하드코딩이 아니라 **실제 fast on일 때만** fast로). 세션 fast 상태는
/// 메인 모델 id에 인코딩되어 있으므로(`/fast on` = `…[fast]` 또는 legacy
/// `gpt-5.5-fast`), 메인이 fast-on이고 라우터가 **스스로** gpt-5.6-terra를
/// 골랐을 때만 priority 티어로 승격한다. 사용자가 지정한 id(Explicit/
/// Pinned/Main*)는 그대로 — 정확한 id가 곧 의도다. 이미 브래킷 티어가
/// 붙어 있으면 멱등 통과.
fn propagate_fast_tier_to_terra(
    resolved_model: String,
    source: RouteDecisionSource,
    request: &RouteRequest,
) -> String {
    let router_chosen = matches!(
        source,
        RouteDecisionSource::AutoSelector
            | RouteDecisionSource::Exploration
            | RouteDecisionSource::ManualSelector
    );
    if !router_chosen || resolved_model.contains('[') {
        return resolved_model;
    }
    let main = request.main_model.to_ascii_lowercase();
    let session_fast_on = main.ends_with("[fast]") || main == "gpt-5.5-fast";
    if !session_fast_on {
        return resolved_model;
    }
    let lower = resolved_model.to_ascii_lowercase();
    let is_terra = lower == "gpt-5.6-terra"
        || lower
            .strip_prefix("gpt-5.6-terra")
            .is_some_and(|suffix| suffix.starts_with('-') || suffix.starts_with('@'));
    if is_terra {
        return format!("{resolved_model}[fast]");
    }
    resolved_model
}

/// [`RouteDecision::recommended_effort`] for the model this route resolved
/// to — reads the resolved model's declared `effort_ceiling` from the
/// inventory (defaulting to the conservative [`EffortCeiling::High`] floor
/// when the model is not found, e.g. an explicit id outside the connected
/// pool) and applies [`recommended_effort_for`]. Runs for every decision
/// source (explicit/pinned/selector/auto/main-fallback) uniformly — it is a
/// pure function of the FINAL resolved model, not of how it was reached.
fn recommended_effort_for_route(request: &RouteRequest, inventory: &ModelInventory, resolved_model: &str) -> Option<EffortCeiling> {
    let ceiling = inventory
        .find(resolved_model)
        .map(ModelDescriptor::effort_ceiling_value)
        .unwrap_or_default();
    recommended_effort_for(request.effective_role(), request.context.complexity, ceiling)
}

/// (model × effort) co-routing rule: a PURE function of (role, task
/// complexity, the candidate model's effort ceiling). Analysis-family roles
/// (Analysis/Research/Judge/Synthesizer — the same family that races for the
/// Deep rung in [`auto_selectors_for_role`]) doing Large-complexity work on a
/// model whose provider-declared ceiling reaches [`EffortCeiling::Ultra`]
/// recommend running that route at Ultra (already clamped to the ceiling,
/// since the rule only fires when the ceiling IS Ultra). Every other
/// combination recommends nothing — the byte-identical default downstream.
#[must_use]
pub fn recommended_effort_for(role: RouteRole, complexity: RouteTaskComplexity, ceiling: EffortCeiling) -> Option<EffortCeiling> {
    let analysis_family = matches!(
        role,
        RouteRole::Analysis | RouteRole::Research | RouteRole::Judge | RouteRole::Synthesizer
    );
    (analysis_family && complexity == RouteTaskComplexity::Large && ceiling == EffortCeiling::Ultra)
        .then_some(EffortCeiling::Ultra)
}

/// Cooldown-aware AUTO selection: scores candidates with the (soft) cooldown
/// deprioritization applied, then re-scores with `cooldown_providers` treated
/// as empty to see whether the penalty actually changed the pick. Returns
/// `(winner, quota_degraded, exploration_fired)`. The second ranking only
/// runs when `cooldown_providers` is non-empty, so the empty-set (default)
/// case is a single ranking pass — byte-identical cost and result to before
/// this feature existed. `exploration_fired` reflects only the FIRST
/// (real-cooldown) ranking — the neutral re-rank exists solely to detect
/// whether cooldown changed the pick, not to attribute exploration.
fn select_auto_model_with_quota_note<'a>(
    request: &RouteRequest,
    inventory: &'a ModelInventory,
) -> (Option<&'a ModelDescriptor>, bool, bool) {
    let scored = ranked_auto_candidates(request, inventory);
    let (winner, exploration_fired) = select_ranked_auto_candidate(request, &scored);
    if request.context.cooldown_providers.is_empty() {
        return (winner, false, exploration_fired);
    }
    let mut neutral_request = request.clone();
    neutral_request.context.cooldown_providers = Vec::new();
    let neutral_scored = ranked_auto_candidates(&neutral_request, inventory);
    let (neutral_winner, _) = select_ranked_auto_candidate(&neutral_request, &neutral_scored);
    let quota_degraded = winner.map(ModelDescriptor::id) != neutral_winner.map(ModelDescriptor::id);
    (winner, quota_degraded, exploration_fired)
}

fn audit_decision(request: &RouteRequest, selected_model: &str, inventory: &ModelInventory) -> RouteAudit {
    let reference_provider = inventory.find(&request.main_model).map(|model| model.provider().to_string());
    let selected_provider = inventory.find(selected_model).map(|model| model.provider().to_string());
    let cross_provider = reference_provider
        .as_ref()
        .zip(selected_provider.as_ref())
        .map(|(reference, selected)| reference != selected);
    let mut guardrails = request.context.audit_notes.clone();
    if !request.context.allow_cross_provider_diversity {
        guardrails.push("cross-provider diversity disabled by default".to_string());
    }
    if !request.context.provider_allowlist.is_empty() {
        guardrails.push(format!(
            "provider allowlist: {}",
            request.context.provider_allowlist.join(", ")
        ));
        // Pins/explicit models can land outside the allowlist (it prefilters
        // only the auto pool) — record the escape so it is auditable.
        if let Some(provider) = selected_provider
            .as_deref()
            .filter(|provider| !request.context.provider_allowed(provider))
        {
            guardrails.push(format!("provider-allowlist-escape:{provider}"));
        }
    }
    if request.context.feedback.enabled {
        guardrails.push("feedback-informed scoring is bounded".to_string());
    }
    RouteAudit {
        reference_provider,
        selected_provider,
        cross_provider,
        route_shape: request.context.route_shape.map(|shape| shape.label().to_string()),
        lane_domain: request.context.lane.as_ref().map(|lane| lane.domain.clone()),
        guardrails,
        feedback_adjustment: request.context.feedback.bounded_adjustment_for(selected_model),
    }
}

fn auto_reason(request: &RouteRequest) -> &'static str {
    if request.context.is_default_policy() {
        "auto role selector"
    } else {
        "auto role selector with route context"
    }
}

/// Sibling spread window: candidates scoring within this of the top score
/// count as "near-best" for lane rotation. The window sits just above the
/// cold-start family seed but far below capability+tier gates, so fan-out lanes
/// can exercise comparable connected models (for example qwen next to
/// gpt/claude) without stepping down to a materially worse role fit.
const LANE_SPREAD_SCORE_WINDOW: i32 = 65;

/// Score gap that separates each AUTO selector fallback rung. It is larger
/// than any in-rung scorer contribution, preserving the fallback ladder while
/// still letting one ranked list back both route selection and fallback previews.
pub(super) const AUTO_SELECTOR_FALLBACK_PENALTY: i32 = 1_000_000;

/// Phase 5 risk-gated exploration score window: candidates scoring within
/// this of the top score are eligible for the deterministic exploration
/// rotation ([`select_exploration_candidate`]). Deliberately WIDER than
/// [`LANE_SPREAD_SCORE_WINDOW`] (65, sized only for near-tied lane siblings):
/// it must clear the full swing an established incumbent's outcome feedback
/// can carry (`MAX_FEEDBACK_ADJUSTMENT` = ±120), otherwise a zero-feedback
/// rival could never out-rank (or even get near) an incumbent sitting at the
/// positive end of that clamp — the exact live-data problem this phase
/// exists to fix (`gpt-5.6-sol` stuck at 2 samples while an incumbent held
/// +117~120). At the same time it stays two orders of magnitude below
/// [`AUTO_SELECTOR_FALLBACK_PENALTY`] (1,000,000, the gap between adjacent
/// AUTO selector fallback rungs), so exploration can structurally never
/// reach into a lower-fit selector rung — see
/// `exploration_rotation_never_crosses_rung_boundary` for the executable
/// proof that this bound holds even under a worst-case in-rung score spread.
pub(super) const EXPLORATION_SCORE_WINDOW: i32 = 200;

fn select_ranked_auto_candidate<'a>(
    request: &RouteRequest,
    scored: &[(&'a ModelDescriptor, i32, usize)],
) -> (Option<&'a ModelDescriptor>, bool) {
    // Preserve `max_by_key` semantics exactly (last of equal maxima) so a
    // non-fan-out route is deterministic and matches fallback ordering.
    let Some(winner) = scored
        .iter()
        .max_by_key(|(_, score, index)| (*score, *index))
        .map(|(model, _, _)| *model)
    else {
        return (None, false);
    };
    // Phase 5: deterministic risk-gated exploration rotation. Tried BEFORE
    // (and, when it fires, takes precedence over) lane rotation below —
    // `exploration_slot` is only ever set by `apply.rs` for a gate-cleared,
    // non-judging route (see `exploration_slot_for_route`), so the two
    // rotations rarely co-occur on the same call in practice, but a
    // deliberate cold-start sampling decision should not be silently
    // overridden by the lane sibling-diversity heuristic when they do.
    if request.context.exploration_slot.is_some() {
        let top_score = scored.iter().map(|(_, score, _)| *score).max().unwrap_or_default();
        if let Some(explored) = select_exploration_candidate(request, scored, top_score) {
            return (Some(explored), true);
        }
    }
    // Sibling spread (lane anti-affinity): a parallel fan-out routes N
    // same-role lanes, and a deterministic argmax sent every one to the same
    // model. When the route carries lane position, rotate by lane index among
    // the near-best candidates so siblings diversify — cross-model agreement
    // is worth a bounded near-best score edge — without ever leaving the
    // near-best set.
    let lane_position = request.context.lane.as_ref().and_then(|lane| {
        let index = lane.lane_index?;
        let count = lane.lane_count?;
        (count >= 2).then_some(index)
    });
    let Some(lane_index) = lane_position else {
        return (Some(winner), false);
    };
    let best = scored
        .iter()
        .map(|(_, score, _)| *score)
        .max()
        .unwrap_or_default();
    let mut near_best: Vec<(&ModelDescriptor, i32, usize)> = scored
        .iter()
        .filter(|(_, score, _)| best.saturating_sub(*score) <= LANE_SPREAD_SCORE_WINDOW)
        .map(|(model, score, index)| (*model, *score, *index))
        .collect();
    if near_best.len() < 2 {
        return (Some(winner), false);
    }
    // Rotate through the same best-to-worst ordering the scorer uses, rather
    // than raw inventory order, so lane 0 keeps the normal winner and siblings
    // fan out to progressively lower-but-still-near-best candidates.
    near_best.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.cmp(&left.2))
    });
    (
        near_best
            .get(lane_index % near_best.len())
            .map(|(model, _, _)| *model),
        false,
    )
}

/// Phase 5 exploration rotation: given `scored` (the SAME ranked list
/// `select_ranked_auto_candidate` computes for this call — every entry
/// already cleared every capability/tier prefilter for the role) and the top
/// score, deterministically picks the highest-scoring UNDER-SAMPLED
/// candidate within [`EXPLORATION_SCORE_WINDOW`] of the top score.
///
/// Two hard filters run before anything else — NEITHER is negotiable:
/// - a candidate whose provider is in `RoutePolicyContext::cooldown_providers`
///   is excluded outright (never explored INTO): a quota-degraded provider
///   would poison the very sample this phase exists to collect, and cooldown
///   is already an operational fact, not a quality one (see
///   `RoutePolicyContext::cooldown_providers`'s doc).
/// - a candidate whose decisive-sample count (from
///   `RoutePolicyContext::exploration_decisive_counts`, defaulting to 0 when
///   the model is absent — the common brand-new-model case) is `>= 2` is
///   excluded — exploration only ever targets genuinely under-sampled models.
///
/// When multiple candidates remain, `exploration_slot` rotates deterministically
/// among them (best-to-worst order, same tie-break as everywhere else in this
/// module) — NO randomness, so repeated calls with the same slot value always
/// pick the same candidate.
///
/// Returns `None` (caller falls through to the normal winner) when the slot
/// is unset or no in-window candidate survives both filters — a wasted slot,
/// never an error and never a forced pick outside the window.
fn select_exploration_candidate<'a>(
    request: &RouteRequest,
    scored: &[(&'a ModelDescriptor, i32, usize)],
    top_score: i32,
) -> Option<&'a ModelDescriptor> {
    let slot = request.context.exploration_slot?;
    // Defense in depth: `apply.rs`'s `exploration_slot_for_route` already
    // gates risk/role before ever setting `exploration_slot`, but these two
    // gates are the non-negotiable ones (design principle: gates are
    // inviolable, not just a slot-computation nicety) — so the engine
    // refuses to explore for a safety-sensitive risk or a judging role
    // structurally, even if some future caller sets `exploration_slot`
    // directly without going through that gate.
    if matches!(request.context.risk, RouteTaskRisk::High | RouteTaskRisk::Critical)
        || matches!(request.effective_role(), RouteRole::Verifier | RouteRole::Reviewer | RouteRole::Judge)
    {
        return None;
    }
    let mut eligible: Vec<(&'a ModelDescriptor, i32, usize)> = scored
        .iter()
        .filter(|(_, score, _)| top_score.saturating_sub(*score) <= EXPLORATION_SCORE_WINDOW)
        .filter(|(model, _, _)| !request.context.provider_in_cooldown(model.provider()))
        .filter(|(model, _, _)| exploration_decisive_count(request, model.id()) < 2)
        .map(|(model, score, index)| (*model, *score, *index))
        .collect();
    if eligible.is_empty() {
        return None;
    }
    eligible.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| right.2.cmp(&left.2)));
    let index = usize::try_from(slot).unwrap_or(usize::MAX) % eligible.len();
    eligible.get(index).map(|(model, _, _)| *model)
}

/// Decisive-sample count for `model_id` from
/// `RoutePolicyContext::exploration_decisive_counts`, defaulting to 0 (never
/// routed under this `route_key` before) when the model is absent from the
/// injected list.
fn exploration_decisive_count(request: &RouteRequest, model_id: &str) -> u32 {
    request
        .context
        .exploration_decisive_counts
        .iter()
        .find(|(id, _)| id == model_id)
        .map_or(0, |(_, count)| *count)
}

/// Phase 5 deterministic exploration slot for a route, given the
/// already-aggregated inputs `apply.rs` reads off the batch-loaded outcome
/// summary (`RouteOutcomeSummary::decisive_counts_for_route_key` /
/// `total_records_for_route_key`). A pure, IO-free predicate so its cadence
/// math and hard gates are unit-testable here directly, independent of
/// settings-file parsing or JSONL I/O — see the `exploration_slot_for_route_*`
/// tests below.
///
/// Fires (`Some(slot)`) only when EVERY one of:
/// - `exploration_enabled` (`smart.exploration`) is `true`,
/// - `risk` is not `High`/`Critical` — safety-sensitive routes never
///   explore,
/// - `role` is not `Verifier`/`Reviewer`/`Judge` — a judging role stays on
///   the most-proven model; exploring INTO the thing that checks everyone
///   else's work would undermine the check itself,
/// - the `route_key`'s cadence has come due
///   (`total_records_for_route_key % cadence == 0`; `cadence` is
///   `smart.explorationCadence`, apply.rs's default 5),
/// - the incumbent has reached full feedback confidence
///   (`top_incumbent_decisive >= CONFIDENT_DECISIVE_SAMPLES` — exploring
///   before any incumbent is even established would just be noise, not the
///   "stuck incumbent" case this phase targets), and
/// - `has_undersampled_rival` is `true` (apply.rs's coarse eligibility
///   proxy — see its doc; this fn does not re-derive it).
///
/// The returned slot is `total_records_for_route_key / cadence` — a
/// monotonically increasing generation counter used ONLY to rotate
/// deterministically across MULTIPLE equally-eligible under-sampled rivals
/// over successive fires (see [`select_exploration_candidate`]); it is never
/// a score input, so it cannot inflate a candidate past the capability/tier/
/// rung gates.
#[must_use]
pub fn exploration_slot_for_route(
    total_records_for_route_key: usize,
    cadence: usize,
    top_incumbent_decisive: usize,
    has_undersampled_rival: bool,
    risk: RouteTaskRisk,
    role: RouteRole,
    exploration_enabled: bool,
) -> Option<u32> {
    if !exploration_enabled {
        return None;
    }
    if matches!(risk, RouteTaskRisk::High | RouteTaskRisk::Critical) {
        return None;
    }
    if matches!(role, RouteRole::Verifier | RouteRole::Reviewer | RouteRole::Judge) {
        return None;
    }
    let cadence = cadence.max(1);
    if !total_records_for_route_key.is_multiple_of(cadence) {
        return None;
    }
    let confident_decisive_samples = usize::try_from(CONFIDENT_DECISIVE_SAMPLES).unwrap_or(8);
    if top_incumbent_decisive < confident_decisive_samples || !has_undersampled_rival {
        return None;
    }
    u32::try_from(total_records_for_route_key / cadence).ok()
}

pub(super) fn selector_matches_context(model: &ModelDescriptor, selector: &RoleSelector) -> bool {
    selector.provider.as_ref().is_none_or(|provider| model.provider() == provider)
        && selector.family.as_ref().is_none_or(|family| model.family() == family)
        && selector.class.as_ref().is_none_or(|class| model.class_matches(class))
        && selector.capability.is_none_or(|capability| model.has_capability(capability))
        && selector.tier.is_none_or(|tier| model.has_tier(tier))
        && freshness_allows(model.status_value(), selector.freshness)
}

fn score_model_with_context(
    model: &ModelDescriptor,
    selector: &RoleSelector,
    request: &RouteRequest,
    inventory: &ModelInventory,
) -> i32 {
    let mut score = 0_i32;
    if selector.capability.is_some_and(|capability| model.has_capability(capability)) { score += 1_000; }
    if selector.tier.is_some_and(|tier| model.has_tier(tier)) { score += TIER_MATCH_BONUS; }
    if selector.class.as_ref().is_some_and(|class| model.class_matches(class)) { score += 75; }
    if model.id() == request.main_model { score += same_main_bonus(request.effective_role()); }
    score += effective_specialty_adjustment(request.effective_role(), model, &request.context.learned_specialty);
    score += i32::try_from(model.release_rank_value().min(100)).unwrap_or(0);

    if matches!(request.context.risk, RouteTaskRisk::High | RouteTaskRisk::Critical)
        && matches!(request.effective_role(), RouteRole::Verifier | RouteRole::Reviewer | RouteRole::Judge)
        && model.has_capability(ModelCapability::Verification)
    {
        score += 40;
    }
    if matches!(request.context.complexity, RouteTaskComplexity::Large)
        && model.has_tier(ModelTier::Deep)
    {
        score += 35;
    }
    // Default-role difficulty routing: a generic agent has no specialty to anchor on,
    // so route it by task complexity (the Agent-tool contract's "auto-route by
    // difficulty: quick work to a cheap model, hard work to a strong one") instead of
    // always re-picking the parent. Specialty roles already carry an explicit tier in
    // their selector, so this only shapes the otherwise-tierless Default role. Same
    // +300 weight as a selector tier match, so the difficulty-fit tier beats the
    // parent's `same_main_bonus` when the parent is the wrong tier; when the parent
    // already is that tier it keeps the bonus and stays selected.
    if request.effective_role() == RouteRole::Default {
        if let Some(preferred) = default_difficulty_tier(request.context.complexity) {
            if model.has_tier(preferred) {
                score += TIER_MATCH_BONUS;
            }
        }
    }
    if request.context.route_shape.is_some_and(|shape| matches!(shape, RouteShapeKind::RepairLoop | RouteShapeKind::ParallelRepairLoop))
        && model.has_capability(ModelCapability::Debugging)
    {
        score += 35;
    }
    // Whole-repo context escalates toward a deep-analysis-capable model OR one
    // whose declared context window can actually hold a whole repo (>=300k) —
    // a capability-derived signal independent of tier, so a huge-context
    // model that has not (yet) earned a Deep tier can still be favored here.
    // A single `||` (not two separate `+= 35` blocks) so a model that
    // qualifies both ways is not double-counted.
    if matches!(request.context.context_need, RouteContextNeed::WholeRepo)
        && (model.has_tier(ModelTier::Deep)
            || model.context_window_value().is_some_and(|window| window >= 300_000))
    {
        score += 35;
    }
    // Writing code while requiring full verification escalates toward a model
    // that can also carry the verification evidence.
    if matches!(request.context.tool_need, RouteToolNeed::Write)
        && matches!(request.context.verification_need, RouteVerificationNeed::Full)
        && model.has_capability(ModelCapability::Verification)
    {
        score += 30;
    }
    score += provider_context_adjustment(model, request, inventory);
    // Quota deprioritization: the binary cooldown (lagging "already throttled")
    // takes precedence over the graded headroom penalty (leading "running low")
    // — a cooled-down provider gets the flat penalty ONLY, never both stacked.
    if request.context.provider_in_cooldown(model.provider()) {
        score -= COOLDOWN_DEPRIORITIZE_PENALTY;
    } else {
        score -= request.context.headroom_penalty_for(model.provider());
    }
    score + i32::from(request.context.feedback.bounded_adjustment_for(model.id()))
}

/// Soft deprioritization (never a hard filter) for a candidate whose provider
/// is in `RoutePolicyContext::cooldown_providers` — an operational quota fact,
/// not a quality signal. Sized to reliably flip an otherwise-tied AUTO pick
/// toward a healthy-provider rival (bigger than the same-main tie-break, `25`)
/// while staying well under the specialty seed (`60`) and the tier/capability
/// gates, so a cooled-down provider still wins when it is genuinely the only
/// or best-fit candidate — this degrades the pick, it does not disqualify it.
///
/// This is ALSO the ceiling of the graded headroom penalty
/// ([`RoutePolicyContext::headroom_penalty_for`]): a provider at `0%` remaining
/// is penalized exactly this much and no more, so the leading (headroom) and
/// lagging (cooldown) quota signals share one bounded scale that can never
/// cross the `+1000` capability / `+300` tier gates — quota only ever breaks a
/// near-tie, it never re-picks a worse-fit role model.
const COOLDOWN_DEPRIORITIZE_PENALTY: i32 = 30;

/// Provider diversity adjustment, driven by the user's
/// `allowCrossProviderDiversity` setting rather than hardcoded model pins.
///
/// Routing is otherwise best-of-breed: all auto roles can pick the strongest
/// capability/tier match across connected providers. Enabling cross-provider
/// diversity adds an independence reward only for error-checking roles
/// (verifier/reviewer/judge); other roles remain pure best-of-breed.
fn provider_context_adjustment(model: &ModelDescriptor, request: &RouteRequest, inventory: &ModelInventory) -> i32 {
    let Some(reference) = inventory.find(&request.main_model) else {
        return 0;
    };
    let diversity_role = matches!(
        request.effective_role(),
        RouteRole::Verifier | RouteRole::Reviewer | RouteRole::Judge
    );
    provider_anchor_adjustment(
        model,
        reference,
        diversity_role,
        request.context.allow_cross_provider_diversity,
    )
}

/// Shared provider-relationship adjustment, used by BOTH the live route scorer
/// ([`score_model_with_context`]) and the recommendation scorer
/// (`assignment::score_model_for_role`) so the `/smart` dashboard's recommended
/// models match what actually gets routed. `reference` is the parent/main model.
///
/// Routing is **best-of-breed**: each role picks the globally strongest model
/// (capability + tier + release rank), so there is NO cross-provider penalty — an
/// Analysis role on a GPT main can reach a deep-reasoning model on another
/// provider, a design/writing role the best writer, etc. The provider
/// relationship only adds an OPT-IN diversity *reward*:
///
/// - model == reference → neutral (the small same-main bonus lives in the scorer).
/// - diversity off (default) → neutral for everyone (pure best-of-breed; the main
///   model still wins ties via the scorer's `+25` same-main bonus).
/// - diversity on, error-diversity role (verifier/reviewer/judge) → prefer a
///   different provider (then a different family) so the checker is independent of
///   the model under review. Non-diversity roles stay pure best-of-breed.
pub(super) fn provider_anchor_adjustment(
    model: &ModelDescriptor,
    reference: &ModelDescriptor,
    diversity_role: bool,
    allow_cross_provider: bool,
) -> i32 {
    if model.id() == reference.id() {
        return 0;
    }
    if !allow_cross_provider || !diversity_role {
        return 0;
    }
    if model.provider() != reference.provider() {
        45
    } else if model.family() == reference.family() {
        10
    } else {
        25
    }
}

/// Same-main tie-break bonus, tapered by role. Shared by BOTH the live route
/// scorer ([`score_model_with_context`]) and the recommendation scorer
/// (`assignment::score_model_for_role`) so the dashboard preview matches routing.
///
/// Non-specialist roles (Default/Fast) keep the original `+25` — staying on the
/// main model is sensible there. Specialist worker roles get only a small
/// tie-break so a role's best specialty model wins over the main model instead of
/// the main model dominating every role it shares a tier with (the cause of
/// "sub-agents just follow the parent").
pub(super) fn same_main_bonus(role: RouteRole) -> i32 {
    match role {
        RouteRole::Default | RouteRole::Fast => 25,
        _ => 5,
    }
}

/// Tier a generic (Default-role) task should prefer, derived purely from its
/// classified complexity, so an unspecialized agent still routes by difficulty:
/// quick work to a cheap (Fast) tier, ordinary work to Balanced, heavy work to a
/// Strong tier. `Unknown` (an empty/uninferable task) yields `None`, leaving such
/// agents on the parent model rather than guessing a tier.
const fn default_difficulty_tier(complexity: RouteTaskComplexity) -> Option<ModelTier> {
    match complexity {
        RouteTaskComplexity::Trivial | RouteTaskComplexity::Small => Some(ModelTier::Fast),
        RouteTaskComplexity::Medium => Some(ModelTier::Balanced),
        RouteTaskComplexity::Large => Some(ModelTier::Strong),
        RouteTaskComplexity::Unknown => None,
    }
}

/// Cold-start specialty PRIOR (renamed from `specialty_seed_adjustment` —
/// Phase 6): a small per-role family preference so that, within a tier,
/// different roles pick different best-of-breed models instead of all
/// converging on one. Shared by both scorers (parity).
///
/// This is a SEED for the cold-start case ONLY — self-retiring per the
/// routing plan's "hardcoding only as cold-start seed" end-state:
/// [`effective_specialty_adjustment`] blends this static seed with Phase 6's
/// learned, verdict-weighted, time-decayed per-(role, model) score as
/// confidence in the learned signal ramps from 0 to 1, so once enough real
/// outcomes accrue for a (role, model) pair this table contributes
/// increasingly little (and, past `>=8` weighted decisive samples, nothing)
/// to that pair's score. Also still exceeded by the plain outcome-feedback
/// term (`MAX_FEEDBACK_ADJUSTMENT`), which continues to apply on top of
/// whatever this blend produces. `gpt` covers the codex line (see
/// `family_for_model`).
pub(super) fn cold_start_specialty_seed(role: RouteRole, model: &ModelDescriptor) -> i32 {
    const SEED: i32 = 60;
    let preferred: &[&str] = match role {
        RouteRole::Coding | RouteRole::Debugging => &["gpt", "claude"],
        RouteRole::Analysis | RouteRole::Research | RouteRole::Judge | RouteRole::Synthesizer => {
            &["claude", "gemini", "deepseek"]
        }
        RouteRole::Writing | RouteRole::Design => &["claude", "gemini"],
        RouteRole::Verifier | RouteRole::Reviewer | RouteRole::Fast | RouteRole::Default => &[],
    };
    if preferred.contains(&model.family()) { SEED } else { 0 }
}

/// Integer "round half away from zero" division — used by
/// [`effective_specialty_adjustment`] so the seed/learned blend never
/// depends on floating point (keeping `RoutePolicyContext`'s `Eq` derive
/// intact end-to-end, and staying exactly reproducible across platforms,
/// unlike `f64` rounding). `denominator` must be positive.
fn round_div(numerator: i32, denominator: i32) -> i32 {
    debug_assert!(denominator > 0, "round_div denominator must be positive");
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        -((-numerator + denominator / 2) / denominator)
    }
}

/// Phase 6 blend: `seed × (1 − c) + learned × c`, where `c` is the learned
/// entry's `confidence_permille / 1000`. Computed as an integer weighted
/// average (see [`round_div`]) rather than floating point.
///
/// Absent a learned entry for this EXACT (role, model) pair — the common
/// case: zero-data, `smart.learnedSpecialty=off`, shadow mode's real request,
/// or simply not enough weighted decisive samples yet — this returns the
/// cold-start seed UNCHANGED (not merely "close to" it): the byte-identical
/// zero-data-identity contract this phase's acceptance criteria require.
///
/// Shared by BOTH scorers (`score_model_with_context` here and
/// `assignment::score_model_for_role`) — a single function, not two
/// independently-maintained copies, so the two can never silently disagree
/// (mirrors `same_main_bonus`/`provider_anchor_adjustment`'s existing
/// shared-fn pattern).
pub(super) fn effective_specialty_adjustment(
    role: RouteRole,
    model: &ModelDescriptor,
    learned: &LearnedSpecialtyHint,
) -> i32 {
    let seed = cold_start_specialty_seed(role, model);
    let Some(entry) = learned.entry_for(role, model.id()) else {
        return seed;
    };
    let confidence = i32::from(entry.confidence_permille).clamp(0, 1000);
    let learned_value = i32::from(entry.model_adjustment);
    round_div(seed * (1000 - confidence) + learned_value * confidence, 1000)
}

/// `true` when a Verifier/Reviewer route should try the Strong tier before
/// Balanced — user decision "검증은 상황에 따라" (situational, not a flat
/// promotion): a High/Critical-risk route, or a Large-complexity one, gets a
/// shot at the stronger checker first; everything else (the common case)
/// keeps the exact pre-existing Balanced-only ladder, so this is
/// byte-identical for Low/Medium risk + non-Large complexity.
pub(super) fn verifier_should_try_strong_first(risk: RouteTaskRisk, complexity: RouteTaskComplexity) -> bool {
    matches!(risk, RouteTaskRisk::High | RouteTaskRisk::Critical) || matches!(complexity, RouteTaskComplexity::Large)
}

/// The AUTO selector ladder for `role`, situationally escalated for
/// Verifier/Reviewer by `risk`/`complexity` (see
/// [`verifier_should_try_strong_first`]). Every other role's ladder is
/// unconditioned by task context — only the checker roles read `risk`/
/// `complexity` here, mirroring how `score_model_with_context` already reads
/// them for scoring bonuses elsewhere in this file.
pub(super) fn auto_selectors_for_role(
    role: RouteRole,
    risk: RouteTaskRisk,
    complexity: RouteTaskComplexity,
    policy: SmartPolicy,
) -> Vec<RoleSelector> {
    match role {
        RouteRole::Default => vec![RoleSelector::new().capability(ModelCapability::Default)],
        RouteRole::Fast => vec![RoleSelector::new().capability(ModelCapability::Fast).tier(ModelTier::Fast)],
        RouteRole::Coding => vec![
            RoleSelector::new().capability(ModelCapability::Coding).tier(ModelTier::Strong),
            RoleSelector::new().capability(ModelCapability::Coding).tier(ModelTier::Balanced),
        ],
        RouteRole::Debugging => vec![
            RoleSelector::new().capability(ModelCapability::Debugging).tier(ModelTier::Strong),
            RoleSelector::new().capability(ModelCapability::Debugging).tier(ModelTier::Balanced),
        ],
        RouteRole::Verifier | RouteRole::Reviewer => {
            if policy == SmartPolicy::Architect {
                // Architect contract: verification is the quality bar, so the
                // checker ladder starts at the Deep reasoning rung (the
                // reserved plan/verify models) and only then falls through the
                // classic Strong/Balanced rungs. Cross-provider anchoring
                // (`provider_anchor_adjustment`, diversity role) still decides
                // WHICH Deep model within the rung, so the verifier lands on a
                // different provider than the anchor when the pool allows.
                vec![
                    RoleSelector::new().capability(ModelCapability::Verification).tier(ModelTier::Deep),
                    RoleSelector::new().capability(ModelCapability::Verification).tier(ModelTier::Strong),
                    RoleSelector::new().capability(ModelCapability::Verification).tier(ModelTier::Balanced),
                ]
            } else if verifier_should_try_strong_first(risk, complexity) {
                vec![
                    RoleSelector::new().capability(ModelCapability::Verification).tier(ModelTier::Strong),
                    RoleSelector::new().capability(ModelCapability::Verification).tier(ModelTier::Balanced),
                ]
            } else {
                vec![RoleSelector::new().capability(ModelCapability::Verification).tier(ModelTier::Balanced)]
            }
        }
        RouteRole::Analysis | RouteRole::Research | RouteRole::Judge | RouteRole::Synthesizer => vec![
            RoleSelector::new().capability(ModelCapability::Analysis).tier(ModelTier::Deep),
            RoleSelector::new().capability(ModelCapability::Analysis).tier(ModelTier::Strong),
            RoleSelector::new().capability(ModelCapability::Analysis).tier(ModelTier::Balanced),
        ],
        RouteRole::Writing => vec![RoleSelector::new().capability(ModelCapability::Writing).tier(ModelTier::Balanced)],
        RouteRole::Design => vec![RoleSelector::new().capability(ModelCapability::Design).tier(ModelTier::Balanced)],
    }
}

#[cfg(test)]
mod headroom_penalty_tests {
    use super::{RoutePolicyContext, COOLDOWN_DEPRIORITIZE_PENALTY};

    fn context(headroom: &[(&str, u8)], threshold: u8) -> RoutePolicyContext {
        RoutePolicyContext {
            provider_headroom: headroom
                .iter()
                .map(|(provider, remaining)| ((*provider).to_string(), *remaining))
                .collect(),
            headroom_penalty_threshold: threshold,
            ..RoutePolicyContext::default()
        }
    }

    /// The graded penalty scales linearly from `0` at the threshold to the full
    /// `COOLDOWN_DEPRIORITIZE_PENALTY` at `0%` remaining, is capped there, and is
    /// inert for absent providers / at-or-above-threshold / a `0` threshold.
    #[test]
    fn graded_penalty_scales_and_caps() {
        // Absent provider → no penalty (byte-identical to no headroom).
        assert_eq!(context(&[], 25).headroom_penalty_for("openai"), 0);
        // At/above the threshold → healthy, no penalty (boundary inclusive).
        assert_eq!(context(&[("openai", 25)], 25).headroom_penalty_for("openai"), 0);
        assert_eq!(context(&[("openai", 40)], 25).headroom_penalty_for("openai"), 0);
        // Below the threshold → graded: 30*(25-5)/25 = 24.
        assert_eq!(context(&[("openai", 5)], 25).headroom_penalty_for("openai"), 24);
        // 0% remaining → the full ceiling (30), never more.
        assert_eq!(
            context(&[("openai", 0)], 25).headroom_penalty_for("openai"),
            COOLDOWN_DEPRIORITIZE_PENALTY
        );
        // A `0` threshold disables the feature (division guard).
        assert_eq!(context(&[("openai", 0)], 0).headroom_penalty_for("openai"), 0);
        // Case-insensitive provider match, mirroring `provider_in_cooldown`.
        assert_eq!(context(&[("OpenAI", 0)], 25).headroom_penalty_for("openai"), 30);
    }

    /// The default context (empty headroom, `0` threshold) never penalizes — the
    /// engine-purity identity contract at the method level.
    #[test]
    fn default_context_is_penalty_free() {
        let default = RoutePolicyContext::default();
        assert_eq!(default.headroom_penalty_for("openai"), 0);
        assert_eq!(default.headroom_penalty_for("anthropic"), 0);
    }
}
