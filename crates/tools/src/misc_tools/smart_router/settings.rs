use std::collections::BTreeMap;
use std::path::Path;

use runtime::{
    default_config_home, FreshnessPolicy, RoleOverride, RoleSelector, RouteAutoClassifierMode,
    RouteTaskComplexity, SmartPolicy, SubagentProfileId,
};
use serde_json::Value;

use super::infer::{role_key, route_role_from_key};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepTierModelsSetting {
    pub models: Vec<String>,
    pub configured: bool,
}

/// Default number of ranked fallback candidates a smart route carries for
/// quota/rate-limit escape (`smart.fallbackCandidateLimit`). Unchanged from
/// the previous hardcoded literal — only promoted to a settings knob so it is
/// tunable without a rebuild; Phase 7 is expected to surface it on `/smart`.
pub(super) const DEFAULT_FALLBACK_CANDIDATE_LIMIT: usize = 2;

/// Default Phase 5 exploration cadence (`smart.explorationCadence`): a
/// `route_key`'s exploration slot fires every Nth recorded outcome
/// (`total_records_for_route_key % N == 0`). N=5 matches the routing plan's
/// "1-in-5 spawns" acceptance criterion — frequent enough that a genuinely
/// under-sampled rival accumulates decisive samples in a reasonable number
/// of turns, infrequent enough that exploration stays a minority of traffic
/// even while the incumbent's dominance is being contested.
pub(super) const DEFAULT_EXPLORATION_CADENCE: usize = 5;

/// Default `smart.headroomPenaltyThreshold` — the *remaining* quota-headroom
/// percent below which the router starts DEDUCTING a candidate's score (graded,
/// capped at the binary-cooldown penalty). 25%: leaves the healthy band
/// untouched while nudging AUTO off a provider whose 5h/7d window (or
/// recent-429 estimate) is running thin. Clamped to `1..=100` on read. Kept in
/// lockstep by the CLI's `cli_snapshot_defaults_match_tools_crate_runtime_defaults`.
pub(super) const DEFAULT_HEADROOM_PENALTY_THRESHOLD: u8 = 25;

/// Default `smart.quotaWaitBandMinutes` — how close (minutes) to a quota
/// window's reset the runtime turn loop HOLDS on the main model instead of
/// falling back to another provider. 15 minutes: ride out a short subscription
/// throttle on the configured model, but fall back when the wall is hours off.
/// `0` disables the band (pure fallback). Kept in lockstep by the CLI's
/// `cli_snapshot_defaults_match_tools_crate_runtime_defaults`.
pub(super) const DEFAULT_QUOTA_WAIT_BAND_MINUTES: u64 = 15;

/// When the foreground Architect contract may swap an EXEC leg from the
/// session model to the routed implementer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmartExecSwap {
    /// Swap only the router's lowest implementation-capable complexity band.
    #[default]
    Easy,
    /// Swap every implementation-shaped turn (the original Architect behavior).
    Always,
    /// Keep every EXEC leg on the session model.
    Never,
}

impl SmartExecSwap {
    /// Parse `smart.execSwap`; absent or unrecognized values use the documented
    /// `easy` default.
    #[must_use]
    pub fn from_settings_value(value: Option<&Value>) -> Self {
        match value.and_then(Value::as_str).map(str::trim) {
            Some(value) if value.eq_ignore_ascii_case("always") => Self::Always,
            Some(value) if value.eq_ignore_ascii_case("never") => Self::Never,
            _ => Self::Easy,
        }
    }

    /// Whether this mode arms the implementer client for the classified turn.
    #[must_use]
    pub fn arms_for(self, complexity: RouteTaskComplexity) -> bool {
        match self {
            // `Trivial` is the classifier's lowest implementation-capable
            // band (write intent is checked separately by the host). `Small`
            // is the next graded band, so it intentionally stays native.
            Self::Easy => complexity == RouteTaskComplexity::Trivial,
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// Phase 6 `smart.learnedSpecialty` mode. Default `Shadow` — see each
/// variant's doc; `apply.rs`'s `SmartRouteContext` decides what to compute
/// and inject based on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum LearnedSpecialtyMode {
    /// Never compute or inject the learned-specialty hint at all — routing
    /// behaves exactly as if Phase 6 did not exist (the cold-start seed
    /// alone decides specialty).
    Off,
    /// Compute the hint, but route with it EMPTY (c=0 — the real decision
    /// stays seed-only) while additionally recording whether injecting it
    /// for real WOULD have changed the pick (`learned-shadow-differs:<model>`
    /// on the route audit) — a soak period before flipping to `On`.
    #[default]
    Shadow,
    /// Inject the computed hint for real; the live route blends seed and
    /// learned per [`runtime::RoutePolicyContext::learned_specialty`]'s
    /// confidence ramp.
    On,
}

impl LearnedSpecialtyMode {
    fn from_settings_value(value: Option<&Value>) -> Self {
        match value.and_then(Value::as_str).map(str::trim) {
            Some("off") => Self::Off,
            Some("on") => Self::On,
            // "shadow", missing, or any unrecognized value: the documented
            // default — same fail-closed-to-default convention as
            // `RouteAutoClassifierMode::from_settings_value`.
            _ => Self::Shadow,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // each bool is an independent smart.* feature gate, not a state machine
pub(super) struct SmartRuntimeSettings {
    pub enabled: bool,
    pub allow_cross_provider_diversity: bool,
    /// Whether the always-on deep-gate VERIFY leg may cross to a different
    /// provider than the main model (`smart.verifyCrossProvider`, default
    /// true). Decoupled from `allow_cross_provider_diversity` on purpose: the
    /// global worker-diversity flag can be off while verify stays cross-model
    /// (the default), or verify can be pinned to the native provider while
    /// worker diversity stays on. The ONLY live consumer is the CLI crate's
    /// `route_deep_verify_model`; this field exists here so the tools reader
    /// parses the same key and the cross-crate defaults contract
    /// (`smart_setting_defaults`) stays in lockstep — hence the `dead_code`
    /// allow (no in-crate routing path reads it yet).
    #[allow(dead_code)]
    pub verify_cross_provider: bool,
    /// Whether a main-model turn that exhausts its subscription/quota window
    /// (`RateLimit` after the retry budget) auto-falls-back to an equivalent
    /// model on a different provider for that turn (`smart.quotaFallback`,
    /// default true). Same lockstep-only rationale as [`Self::verify_cross_provider`]:
    /// the ONLY live consumer is the CLI crate's `route_quota_fallback_model` /
    /// turn loop; this field exists here so the tools reader parses the same key
    /// and the cross-crate defaults contract (`smart_setting_defaults`) stays in
    /// lockstep — hence the `dead_code` allow.
    #[allow(dead_code)]
    pub quota_fallback: bool,
    /// Minutes-to-reset within which the runtime turn loop HOLDS on the main
    /// model instead of falling back (`smart.quotaWaitBandMinutes`, default
    /// [`DEFAULT_QUOTA_WAIT_BAND_MINUTES`]; `0` disables). Same lockstep-only
    /// rationale as [`Self::quota_fallback`]: the live consumer is the CLI's
    /// turn-entry `set_quota_wait_band`; parsed here so the dual-reader defaults
    /// contract stays in lockstep — hence the `dead_code` allow.
    #[allow(dead_code)]
    pub quota_wait_band_minutes: u64,
    /// Providers the auto route may pick (`smart.providerAllowlist`); empty =
    /// all connected providers (the default). Explicit models, pins, and the
    /// main-model fallback are never constrained by this.
    pub provider_allowlist: Vec<String>,
    /// Ordered Architect PLAN/VERIFY pool (`smart.deepTierModels`). Missing or
    /// empty uses the built-in pool; a non-empty array replaces it.
    pub deep_tier_models: Vec<String>,
    pub deep_tier_models_configured: bool,
    pub feedback_informed_auto: bool,
    pub auto_classifier: RouteAutoClassifierMode,
    pub subagents: BTreeMap<String, RoleOverride>,
    pub roles: BTreeMap<String, RoleOverride>,
    /// Ranked fallback-candidate count for quota/rate-limit escape
    /// (`smart.fallbackCandidateLimit`, default [`DEFAULT_FALLBACK_CANDIDATE_LIMIT`]).
    pub fallback_candidate_limit: usize,
    /// Master switch for Phase 5 deterministic exploration (`smart.exploration`).
    /// On by default: without it a zero/thin-history model that already
    /// cleared every capability/tier prefilter can never accumulate outcome
    /// samples (an established incumbent's feedback bound outlives any lane
    /// window), the exact live-data problem this phase exists to fix. Still
    /// gated per-route by risk/role hard gates regardless of this flag.
    pub exploration: bool,
    /// Cadence divisor for Phase 5 exploration (`smart.explorationCadence`,
    /// default [`DEFAULT_EXPLORATION_CADENCE`]).
    pub exploration_cadence: usize,
    /// Phase 6 `smart.learnedSpecialty` mode (default [`LearnedSpecialtyMode::Shadow`]).
    pub learned_specialty: LearnedSpecialtyMode,
    /// Remaining-percent threshold below which low quota headroom starts
    /// deducting route score (`smart.headroomPenaltyThreshold`, default
    /// [`DEFAULT_HEADROOM_PENALTY_THRESHOLD`], clamped `1..=100`). Injected into
    /// `RoutePolicyContext::headroom_penalty_threshold` for the graded penalty.
    pub headroom_penalty_threshold: u8,
    /// The Smart execution-contract flavor (`smart.policy`, default
    /// `architect`; explicit `"classic"` or `ZO_SMART_POLICY=classic` opts
    /// out — see [`SmartPolicy::from_settings_value`]). Injected into
    /// `RoutePolicyContext::policy` so the implementation gate and the
    /// Verifier ladder enforce the contract on every smart-routed spawn.
    pub policy: SmartPolicy,
    /// When the foreground deep gate may swap EXEC legs to the routed
    /// implementer (`smart.execSwap`, default `easy`). This does not affect
    /// spawn routing or verifier selection.
    pub exec_swap: SmartExecSwap,
}

/// Plain-data snapshot of every `smart.*` runtime default this module's
/// reader ([`read_smart_runtime_settings`]) falls back to when a key is
/// absent from `settings.json`. Exposed publicly (re-exported at the crate
/// root as `tools::smart_setting_defaults`) ONLY so the CLI crate's
/// `SmartSettingsSnapshot` reader (`snapshot_from_root`) can assert its own
/// defaults stay byte-identical to this crate's — the two crates parse the
/// SAME `settings.json` independently for two different surfaces (the
/// dashboard preview vs. live routing), and the smart-auto routing plan
/// documents this as the dual-reader drift point (P7). Not consumed by any
/// routing path — read-only, for a cross-crate test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // each bool mirrors an independent smart.* feature gate default, not a state machine
pub struct SmartSettingDefaults {
    pub enabled: bool,
    pub allow_cross_provider_diversity: bool,
    /// Default for `smart.verifyCrossProvider` — the deep-gate VERIFY leg's own
    /// cross-provider switch (`true`, same as the global diversity flag).
    pub verify_cross_provider: bool,
    /// Default for `smart.quotaFallback` — the automatic cross-provider fallback
    /// on main-model quota exhaustion (`true`).
    pub quota_fallback: bool,
    /// Default `smart.quotaWaitBandMinutes` ([`DEFAULT_QUOTA_WAIT_BAND_MINUTES`]).
    pub quota_wait_band_minutes: u64,
    /// Default `smart.deepTierModels` preference order.
    pub deep_tier_models: &'static [&'static str],
    pub feedback_informed_auto: bool,
    pub fallback_candidate_limit: usize,
    pub exploration: bool,
    pub exploration_cadence: usize,
    /// `true` when the default `learnedSpecialty` mode is `shadow` (today's
    /// documented default) rather than `off`/`on`.
    pub learned_specialty_defaults_to_shadow: bool,
    /// Default `smart.headroomPenaltyThreshold` ([`DEFAULT_HEADROOM_PENALTY_THRESHOLD`]).
    pub headroom_penalty_threshold: u8,
    /// Default `smart.policy` (`Architect` — the role-separation contract is
    /// the live default; `classic` opts out).
    pub policy: SmartPolicy,
    /// Default `smart.execSwap` (`easy`).
    pub exec_swap: SmartExecSwap,
}

#[must_use]
pub fn smart_setting_defaults() -> SmartSettingDefaults {
    SmartSettingDefaults {
        enabled: true,
        allow_cross_provider_diversity: true,
        verify_cross_provider: true,
        quota_fallback: true,
        quota_wait_band_minutes: DEFAULT_QUOTA_WAIT_BAND_MINUTES,
        deep_tier_models: &runtime::DEFAULT_DEEP_TIER_MODELS,
        feedback_informed_auto: true,
        fallback_candidate_limit: DEFAULT_FALLBACK_CANDIDATE_LIMIT,
        exploration: true,
        exploration_cadence: DEFAULT_EXPLORATION_CADENCE,
        learned_specialty_defaults_to_shadow: matches!(
            LearnedSpecialtyMode::default(),
            LearnedSpecialtyMode::Shadow
        ),
        headroom_penalty_threshold: DEFAULT_HEADROOM_PENALTY_THRESHOLD,
        policy: SmartPolicy::Architect,
        exec_swap: SmartExecSwap::Easy,
    }
}

pub(super) fn read_smart_runtime_settings() -> Option<SmartRuntimeSettings> {
    // Route policy follows the SAME merged settings the session resolved —
    // global settings.json, the project's .zo/settings*.json, and a
    // `--settings` overlay (ConfigLoader holds that flag as a process-wide
    // override). The old direct read of only the global file meant a project
    // or CLI-level `smart.enabled: false` was silently ignored by spawn
    // routing. Missing files still merge to an empty root (defaults ON); a
    // load failure bails to None (fail-safe: no routing), matching the old
    // malformed-file behavior.
    let cwd = std::env::current_dir().unwrap_or_else(|_| default_config_home());
    read_smart_runtime_settings_for(&cwd)
}

#[allow(clippy::too_many_lines)] // flat, one-block-per-key parser for the merged smart.* object
fn read_smart_runtime_settings_for(cwd: &Path) -> Option<SmartRuntimeSettings> {
    let root = merged_settings_root(cwd)?;
    let smart = root.get("smart").and_then(Value::as_object);
    let enabled = smart
        .and_then(|smart| smart.get("enabled"))
        .and_then(Value::as_bool)
        // On by default: smart AUTO is the product's routing brain — subagent
        // spawns land on the best connected model per role (verify cross-checks
        // on a different provider, workers get best-of-breed) with zero setup.
        // Note the blast radius includes single-provider users too: the
        // inventory carries every catalog entry of a usable provider, so spawns
        // can land on a different same-provider model/effort tier. An explicit
        // `smart.enabled: false` (or `/smart off`) still wins. Keep in lockstep
        // with `snapshot_from_root` in the CLI and `smart_setting_defaults()`.
        .unwrap_or(true);
    let allow_cross_provider_diversity = smart
        .and_then(|smart| smart.get("allowCrossProviderDiversity"))
        .and_then(Value::as_bool)
        // On by default: cross-checking is the point of Smart auto — verifier/
        // reviewer roles should land on a *different* provider than the main
        // model without any setup, and worker roles get best-of-breed instead of
        // same-provider anchoring. Single-provider pools are unaffected (the
        // inventory only ever contains connected models), and `/smart` can still
        // turn it off. Keep in lockstep with `snapshot_from_root` in the CLI.
        .unwrap_or(true);
    let verify_cross_provider = smart
        .and_then(|smart| smart.get("verifyCrossProvider"))
        .and_then(Value::as_bool)
        // On by default: the always-on deep-gate VERIFY leg should cross to a
        // different provider than the main model without any setup. Governs
        // ONLY the verify leg — decoupled from `allowCrossProviderDiversity`
        // (the worker-diversity flag) so one can be off while the other is on.
        // Keep in lockstep with `snapshot_from_root` in the CLI and
        // `smart_setting_defaults()`.
        .unwrap_or(true);
    let quota_fallback = smart
        .and_then(|smart| smart.get("quotaFallback"))
        .and_then(Value::as_bool)
        // On by default: a main-model quota exhaustion should auto-continue on an
        // equivalent different-provider model for that turn instead of killing
        // the turn. Keep in lockstep with `snapshot_from_root` in the CLI and
        // `smart_setting_defaults()`.
        .unwrap_or(true);
    let quota_wait_band_minutes = quota_wait_band_minutes_from_smart(smart);
    let provider_allowlist = provider_allowlist_from_smart(smart);
    let deep_tier_setting = deep_tier_models_from_smart(smart);
    let feedback_informed_auto = smart
        .and_then(|smart| smart.get("feedbackInformedAuto"))
        .and_then(Value::as_bool)
        // On by default: the outcome loop is how routing stays dynamic (learns who
        // performs the role) instead of frozen on the model-name/recency prior. It
        // is bounded + confidence-weighted (see `RouteOutcomeBucket`), and the user
        // can still disable it via `/smart`.
        .unwrap_or(true);
    let auto_classifier = RouteAutoClassifierMode::from_settings_value(
        smart.and_then(|smart| smart.get("autoClassifier")),
    );
    let fallback_candidate_limit = smart
        .and_then(|smart| smart.get("fallbackCandidateLimit"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_FALLBACK_CANDIDATE_LIMIT);
    let exploration = smart
        .and_then(|smart| smart.get("exploration"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let exploration_cadence = smart
        .and_then(|smart| smart.get("explorationCadence"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_EXPLORATION_CADENCE);
    let learned_specialty =
        LearnedSpecialtyMode::from_settings_value(smart.and_then(|smart| smart.get("learnedSpecialty")));
    let policy = SmartPolicy::from_settings_value(smart.and_then(|smart| smart.get("policy")));
    let exec_swap =
        SmartExecSwap::from_settings_value(smart.and_then(|smart| smart.get("execSwap")));
    let headroom_penalty_threshold = smart
        .and_then(|smart| smart.get("headroomPenaltyThreshold"))
        .and_then(Value::as_u64)
        // Clamp to 1..=100 (not `filter(>0)`): a 0 threshold would divide-by-zero
        // the graded penalty, and >100 is meaningless — pin both ends into range
        // rather than silently falling back to the default. Keep in lockstep with
        // `snapshot_from_root` in the CLI.
        .map(|value| value.clamp(1, 100))
        .and_then(|value| u8::try_from(value).ok())
        .unwrap_or(DEFAULT_HEADROOM_PENALTY_THRESHOLD);
    let mut subagents = BTreeMap::new();
    if let Some(router) = root.get("modelRouter").and_then(Value::as_object) {
        if let Some(subagent_object) = router.get("subagents").and_then(Value::as_object) {
            for (subagent, value) in subagent_object {
                let Some(profile) = SubagentProfileId::parse(subagent) else {
                    continue;
                };
                if let Ok(Some(override_rule)) = role_override_from_json(value) {
                    subagents.insert(profile.key().to_string(), override_rule);
                }
            }
        }
    }
    let mut roles = BTreeMap::new();
    if let Some(router) = root.get("modelRouter").and_then(Value::as_object) {
        if let Some(role_object) = router.get("roles").and_then(Value::as_object) {
            for (role, value) in role_object {
                let Some(route_role) = route_role_from_key(role) else {
                    continue;
                };
                if let Ok(Some(override_rule)) = role_override_from_json(value) {
                    roles.insert(role_key(route_role).to_string(), override_rule);
                }
            }
        }
    }
    Some(SmartRuntimeSettings {
        enabled,
        allow_cross_provider_diversity,
        verify_cross_provider,
        quota_fallback,
        quota_wait_band_minutes,
        provider_allowlist,
        deep_tier_models: deep_tier_setting.models,
        deep_tier_models_configured: deep_tier_setting.configured,
        feedback_informed_auto,
        auto_classifier,
        subagents,
        roles,
        fallback_candidate_limit,
        exploration,
        exploration_cadence,
        learned_specialty,
        headroom_penalty_threshold,
        policy,
        exec_swap,
    })
}

fn merged_settings_root(cwd: &Path) -> Option<Value> {
    let config = runtime::ConfigLoader::default_for(cwd).load().ok()?;
    serde_json::from_str(&config.as_json().render()).ok()
}

/// The foreground Architect EXEC-swap policy from merged
/// [`runtime::ConfigLoader`] settings, so project and `--settings` overlays
/// participate. A loader failure keeps EXEC native.
#[must_use]
pub fn smart_exec_swap() -> SmartExecSwap {
    read_smart_runtime_settings().map_or(SmartExecSwap::Never, |settings| settings.exec_swap)
}

/// Ordered Architect PLAN/VERIFY pool from merged [`runtime::ConfigLoader`]
/// settings. A load failure keeps the built-in safety pool.
#[must_use]
pub fn smart_deep_tier_models() -> Vec<String> {
    read_smart_runtime_settings().map_or_else(
        runtime::default_deep_tier_models,
        |settings| settings.deep_tier_models,
    )
}

/// Ordered Architect PLAN/VERIFY pool and whether a non-empty merged setting
/// replaced the built-in default for `cwd`.
#[must_use]
pub fn smart_deep_tier_models_for(cwd: &Path) -> Option<DeepTierModelsSetting> {
    read_smart_runtime_settings_for(cwd).map(|settings| DeepTierModelsSetting {
        models: settings.deep_tier_models,
        configured: settings.deep_tier_models_configured,
    })
}

/// The live `smart.policy` for callers outside the batch route context (the
/// spawn-time implementation gate and its rate-limit fallback filter). A
/// settings LOAD FAILURE degrades to [`SmartPolicy::Classic`] — the
/// pre-contract behavior — matching [`read_smart_runtime_settings`]'s
/// fail-safe direction; a merely absent key still resolves to the documented
/// `Architect` default via [`SmartPolicy::from_settings_value`].
pub(crate) fn live_smart_policy() -> SmartPolicy {
    read_smart_runtime_settings().map_or(SmartPolicy::Classic, |settings| settings.policy)
}

/// Parse `smart.providerAllowlist` — a JSON array of provider names. Blank
/// entries are dropped; a missing/non-array value means "no restriction".
/// Keep in lockstep with `snapshot_from_root` in the CLI.
/// Parse `smart.quotaWaitBandMinutes` — minutes-to-reset within which the turn
/// loop holds on the main model instead of falling back. Absent → the default
/// band; `0` disables. Keep in lockstep with `snapshot_from_root` in the CLI.
fn quota_wait_band_minutes_from_smart(smart: Option<&serde_json::Map<String, Value>>) -> u64 {
    smart
        .and_then(|smart| smart.get("quotaWaitBandMinutes"))
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_QUOTA_WAIT_BAND_MINUTES)
}

fn deep_tier_models_from_smart(
    smart: Option<&serde_json::Map<String, Value>>,
) -> DeepTierModelsSetting {
    let configured = smart
        .and_then(|smart| smart.get("deepTierModels"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(api::resolve_model_alias)
                .collect()
        })
        .filter(|models: &Vec<String>| !models.is_empty());
    configured.map_or_else(
        || DeepTierModelsSetting {
            models: runtime::default_deep_tier_models(),
            configured: false,
        },
        |models| DeepTierModelsSetting {
            models,
            configured: true,
        },
    )
}

pub(super) fn provider_allowlist_from_smart(
    smart: Option<&serde_json::Map<String, Value>>,
) -> Vec<String> {
    smart
        .and_then(|smart| smart.get("providerAllowlist"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn role_override_from_json(value: &Value) -> Result<Option<RoleOverride>, ()> {
    let object = value.as_object().ok_or(())?;
    match object.get("mode").and_then(Value::as_str).ok_or(())? {
        // Deletion tombstone: the CLI writes `{"mode":"deleted"}` into the
        // primary canonical root to durably mask a same-key override that lives
        // only in a lower root. The router must read it as "no override" (the
        // same as `auto`/absent), not as a parse error.
        "auto" | "deleted" => Ok(None),
        "pinned" => {
            let model = object.get("model").and_then(Value::as_str).ok_or(())?.trim();
            if model.is_empty() {
                return Err(());
            }
            Ok(Some(RoleOverride::Pin(model.to_string())))
        }
        "manualPreferred" => {
            let selector = object.get("selector").and_then(Value::as_object).ok_or(())?;
            let provider = required_selector_string(selector, "provider").ok_or(())?;
            let family = required_selector_string(selector, "family").ok_or(())?;
            let class = required_selector_string(selector, "class").ok_or(())?;
            let freshness = match required_selector_string(selector, "freshness").ok_or(())?.as_str() {
                "latest" => FreshnessPolicy::Latest,
                "latestStable" => FreshnessPolicy::LatestStable,
                _ => return Err(()),
            };
            Ok(Some(RoleOverride::Family(
                RoleSelector::new()
                    .provider(provider)
                    .family(family)
                    .class(class)
                    .freshness(freshness),
            )))
        }
        _ => Err(()),
    }
}

fn required_selector_string(
    selector: &serde_json::Map<String, Value>,
    key: &str,
) -> Option<String> {
    let value = selector.get(key)?.as_str()?.trim();
    (!value.is_empty()).then_some(value.to_string())
}

// Outcome-feedback lookups live on `apply::SmartRouteContext::feedback_for`,
// which serves them from a batch-loaded summary instead of re-parsing the
// outcome log per agent. The keying contract (resolved subagent type,
// `subagent:{type}`) is documented there.
