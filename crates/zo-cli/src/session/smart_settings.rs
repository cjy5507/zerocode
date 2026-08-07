use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use runtime::{
    connected_model_inventory, default_config_home, read_route_outcomes,
    recommend_auto_assignments_with_learned_specialty,
    recommend_role_fallbacks_with_learned_specialty, route_model,
    summarize_route_outcomes_with_canonicalizer,
    AssignmentSource, AutoAssignmentOptions, FreshnessPolicy, LearnedSpecialtyHint, RoleOverride,
    RoleSelector, RouteAutoClassifierMode, RouteContextNeed, RouteFeedbackHint, RoutePolicyContext,
    RouteRequest, RouteRole, RouteTaskComplexity, RouteTaskRisk, RouteToolNeed,
    RouteVerificationNeed, RoutingTarget, SubagentProfileId, CONFIDENT_DECISIVE_SAMPLES,
};
use runtime::model_router::AssignmentConfidence;
use zo_cli::tui::glyphs;
use zo_cli::tui::modals::{
    SmartSettingsCommit, SmartSettingsFreshness, SmartSettingsModal, SmartSettingsModel,
    SmartSettingsObservedRoute, SmartSettingsRecommendation, SmartSettingsTarget,
    SmartSettingsTargetKind, SmartSettingsUpdate, SmartSettingsView,
};
use serde_json::{Map as JsonMap, Value};

#[cfg(test)]
thread_local! {
    static SMART_TEST_CONFIG_HOME: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_test_config_home<T>(config_home: &Path, run: impl FnOnce() -> T) -> T {
    struct RestoreTestConfigHome(Option<PathBuf>);

    impl Drop for RestoreTestConfigHome {
        fn drop(&mut self) {
            SMART_TEST_CONFIG_HOME.with(|home| {
                home.replace(self.0.take());
            });
        }
    }

    let prior = SMART_TEST_CONFIG_HOME.with(|home| home.replace(Some(config_home.to_path_buf())));
    let _restore = RestoreTestConfigHome(prior);
    run()
}

/// Default `smart.fallbackCandidateLimit` — mirrors the tools crate's
/// `misc_tools::smart_router::settings::DEFAULT_FALLBACK_CANDIDATE_LIMIT`
/// (private to that crate). Kept in lockstep by the
/// `cli_snapshot_defaults_match_tools_crate_runtime_defaults` test.
const DEFAULT_FALLBACK_CANDIDATE_LIMIT: usize = 2;

/// Default `smart.explorationCadence` — mirrors the tools crate's
/// `DEFAULT_EXPLORATION_CADENCE` (private to that crate). Same lockstep test.
const DEFAULT_EXPLORATION_CADENCE: usize = 5;

/// Default `smart.headroomPenaltyThreshold` — mirrors the tools crate's
/// `DEFAULT_HEADROOM_PENALTY_THRESHOLD` (private to that crate). Same lockstep test.
const DEFAULT_HEADROOM_PENALTY_THRESHOLD: u8 = 25;

/// Default `smart.quotaWaitBandMinutes` — mirrors the tools crate's
/// `DEFAULT_QUOTA_WAIT_BAND_MINUTES`. 15 minutes: hold the turn on the main
/// model for a quota window that clears within a quarter hour, else fall back.
const DEFAULT_QUOTA_WAIT_BAND_MINUTES: u64 = 15;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // each bool mirrors an independent smart.* feature gate, not a state machine
pub(crate) struct SmartSettingsSnapshot {
    pub enabled: bool,
    pub allow_cross_provider_diversity: bool,
    /// Whether the always-on deep-gate VERIFY leg may cross to a different
    /// provider than the main model (`smart.verifyCrossProvider`, default
    /// true). The verify leg's cross-provider behavior is governed SOLELY by
    /// this key — decoupled from `allow_cross_provider_diversity` (the global
    /// worker-diversity flag) — so a user can turn worker diversity off yet
    /// keep verify cross-model, or pin verify native while workers stay
    /// diverse. Lockstep with the tools crate's `read_smart_runtime_settings`.
    pub verify_cross_provider: bool,
    /// Whether a main-model turn that exhausts its subscription/quota window
    /// auto-falls-back to an equivalent different-provider model for that turn
    /// (`smart.quotaFallback`, default true). The live consumer is
    /// [`route_quota_fallback_model`] + the runtime turn loop; kept in lockstep
    /// with the tools crate's `read_smart_runtime_settings`.
    pub quota_fallback: bool,
    /// How close (in minutes) to a quota window's reset the runtime turn loop
    /// HOLDS on the main model instead of falling back (`smart.quotaWaitBandMinutes`,
    /// default 15). `0` disables the band (pure fallback). Lockstep with the
    /// tools crate's `read_smart_runtime_settings`; consumed by the host on turn
    /// entry via `runtime::set_quota_wait_band`.
    pub quota_wait_band_minutes: u64,
    /// Providers the auto route may pick (`smart.providerAllowlist`); empty =
    /// all connected providers. Read-only here (no modal editing yet) — kept
    /// in lockstep with the tools crate's `read_smart_runtime_settings`.
    pub provider_allowlist: Vec<String>,
    /// Ordered Architect PLAN/VERIFY pool (`smart.deepTierModels`). Missing or
    /// empty uses the built-in pool; a non-empty array replaces it.
    pub deep_tier_models: Vec<String>,
    pub feedback_informed_auto: bool,
    pub auto_classifier: RouteAutoClassifierMode,
    pub subagents: BTreeMap<String, SmartRoleUpdate>,
    pub roles: BTreeMap<String, SmartRoleUpdate>,
    /// Ranked fallback-candidate count for quota/rate-limit escape
    /// (`smart.fallbackCandidateLimit`) — P7 lockstep with the tools crate's
    /// `SmartRuntimeSettings::fallback_candidate_limit`
    /// (`DEFAULT_FALLBACK_CANDIDATE_LIMIT` = 2).
    pub fallback_candidate_limit: usize,
    /// Master switch for Phase 5 deterministic exploration
    /// (`smart.exploration`) — P7 lockstep with the tools crate's
    /// `SmartRuntimeSettings::exploration` (defaults `true`).
    pub exploration: bool,
    /// Cadence divisor for Phase 5 exploration (`smart.explorationCadence`) —
    /// P7 lockstep with the tools crate's `DEFAULT_EXPLORATION_CADENCE` = 5.
    pub exploration_cadence: usize,
    /// Phase 6 `smart.learnedSpecialty` mode — P7 lockstep with the tools
    /// crate's `LearnedSpecialtyMode` (that enum is private to the `tools`
    /// crate, so this is a CLI-local mirror, same pattern as
    /// [`SmartFreshness`] mirroring `runtime::FreshnessPolicy`).
    pub learned_specialty: SmartLearnedSpecialtyMode,
    /// Remaining-percent threshold for the router's graded headroom penalty
    /// (`smart.headroomPenaltyThreshold`) — lockstep mirror of the tools crate's
    /// `SmartRuntimeSettings::headroom_penalty_threshold` (default
    /// [`DEFAULT_HEADROOM_PENALTY_THRESHOLD`] = 25). Read-only here (no modal
    /// editing yet); carried for the dual-reader defaults contract.
    pub headroom_penalty_threshold: u8,
    /// The Smart execution-contract flavor (`smart.policy`, default
    /// `architect`) — lockstep mirror of the tools crate's
    /// `SmartRuntimeSettings::policy`. Drives the deep-verify/quota-fallback
    /// route contexts and the dashboard preview options so the CLI shows the
    /// same Verifier ladder and implementation gate the runtime enforces.
    pub policy: runtime::SmartPolicy,
    /// When the foreground deep gate may swap EXEC legs to the routed
    /// implementer (`smart.execSwap`, default `always`). Parsed here for the
    /// dual-reader defaults contract; the live gate reads the merged
    /// `ConfigLoader` value through [`tools::smart_exec_swap`].
    pub exec_swap: tools::SmartExecSwap,
}



pub(crate) fn build_smart_settings_modal(
    default_model: &str,
    cwd: &Path,
    session_id: &str,
) -> io::Result<SmartSettingsModal> {
    let snapshot = read_global_smart_settings()?;
    let inventory = connected_model_inventory(default_model);
    let models = inventory
        .models()
        .iter()
        .map(|model| SmartSettingsModel {
            id: model.id().to_string(),
            provider: model.provider().to_string(),
            family: model.family().to_string(),
            class: model.class_label().unwrap_or("balanced").to_string(),
        })
        .collect();
    let roles = smart_role_keys()
        .iter()
        .map(|role| SmartSettingsTarget {
            key: (*role).to_string(),
            label: role_display_label(role).to_string(),
            kind: SmartSettingsTargetKind::Role,
            update: snapshot
                .roles
                .get(*role)
                .map_or(SmartSettingsUpdate::Auto, smart_settings_update_from_role_update),
        })
        .collect();
    let subagents = runtime::BuiltinSubagentProfile::all()
        .iter()
        .map(|profile| {
            let key = profile.key().to_string();
            SmartSettingsTarget {
                label: subagent_display_label(&key),
                update: snapshot
                    .subagents
                    .get(&key)
                    .map_or(SmartSettingsUpdate::Auto, smart_settings_update_from_role_update),
                key,
                kind: SmartSettingsTargetKind::Subagent,
            }
        })
        .collect();
    // Precompute BOTH diversity sets so toggling cross-provider diversity in the
    // modal previews live (the recommendation is pure/deterministic in the
    // option), instead of being frozen at the snapshot's setting until save. When
    // feedback-informed auto is on, fold the durable outcome history into the
    // recommendation scorer so the auto preview reflects what actually performed
    // (the same bounded term the live router applies), not just the static
    // model-name/recency prior — otherwise "auto" looks like it never learns.
    let outcome_feedback = snapshot
        .feedback_informed_auto
        .then(|| runtime::read_route_outcome_summary(cwd).ok())
        .flatten();
    let learned = dashboard_learned_specialty_hint(&snapshot);
    let recommendations = smart_recommendations(
        &inventory,
        false,
        &snapshot.provider_allowlist,
        outcome_feedback.as_ref(),
        &learned,
        snapshot.policy,
    );
    let recommendations_with_diversity = smart_recommendations(
        &inventory,
        true,
        &snapshot.provider_allowlist,
        outcome_feedback.as_ref(),
        &learned,
        snapshot.policy,
    );
    Ok(SmartSettingsModal::new(SmartSettingsView {
        enabled: snapshot.enabled,
        allow_cross_provider_diversity: snapshot.allow_cross_provider_diversity,
        feedback_informed_auto: snapshot.feedback_informed_auto,
        auto_classifier: snapshot.auto_classifier.status_label().to_string(),
        main_model: inventory.main_model().to_string(),
        settings_path: global_settings_path().display().to_string(),
        models,
        model_notes: smart_model_pool_notes(),
        roles,
        subagents,
        recommendations,
        recommendations_with_diversity,
        turn_output_tokens: session_turn_output_usage(cwd, session_id),
        observed_routes: build_observed_routes(cwd),
    }))
}

/// Observed routing outcomes for the dashboard, read from the durable
/// route-outcome log and **aggregated per target across every model that ran**.
/// Each bucket's `route_key` is `"<kind>:<key>"` (e.g. `subagent:Verification`,
/// `role:coding`); `split_once(':')` maps it back to the modal's target kind/key
/// (keys never contain a colon). Counts are decisive runs only (completed +
/// failed) so user cancels don't distort the `ok` ratio. The model is reported
/// only when a single model ran for the target — naming one of several would
/// mislead. Best-effort: a missing/unreadable log yields an empty list.
fn build_observed_routes(cwd: &Path) -> Vec<SmartSettingsObservedRoute> {
    runtime::read_route_outcome_summary(cwd)
        .map(|summary| observed_routes_from_summary(&summary))
        .unwrap_or_default()
}

/// Pure aggregation half of [`build_observed_routes`], split out so the mapping
/// is testable without touching the filesystem.
fn observed_routes_from_summary(summary: &runtime::RouteOutcomeSummary) -> Vec<SmartSettingsObservedRoute> {
    // (kind, key) -> (completed, decisive, distinct models)
    let mut groups: Vec<(SmartSettingsTargetKind, String, usize, usize, Vec<String>)> = Vec::new();
    for bucket in &summary.by_route {
        let Some((kind, key)) = bucket.route_key.split_once(':') else {
            continue;
        };
        let kind = match kind {
            "subagent" => SmartSettingsTargetKind::Subagent,
            "role" => SmartSettingsTargetKind::Role,
            _ => continue,
        };
        let decisive = bucket.completed.saturating_add(bucket.failed);
        if let Some(group) = groups.iter_mut().find(|(k, existing, ..)| *k == kind && existing == key) {
            group.2 = group.2.saturating_add(bucket.completed);
            group.3 = group.3.saturating_add(decisive);
            if !group.4.contains(&bucket.selected_model) {
                group.4.push(bucket.selected_model.clone());
            }
        } else {
            groups.push((kind, key.to_string(), bucket.completed, decisive, vec![bucket.selected_model.clone()]));
        }
    }
    groups
        .into_iter()
        .map(|(kind, key, completed, decisive, mut models)| SmartSettingsObservedRoute {
            kind,
            key,
            completed,
            decisive,
            model: if models.len() == 1 { models.pop() } else { None },
        })
        .collect()
}

/// Recommendation previews for every subagent profile AND every route role, for
/// one cross-provider-diversity setting. Roles are surfaced too so the Roles tab
/// previews the model the runtime would actually auto-route to, instead of
/// always showing the main-model fallback.
///
/// `learned` is the Phase 6 hint, already mode-gated by the caller
/// ([`dashboard_learned_specialty_hint`]: empty in `off`/`shadow`, live in
/// `on`) — passed through to the `_with_learned_specialty` scorer variants so
/// the dashboard preview matches what live routing (`apply.rs`'s
/// `learned_specialty_for_real_request`) actually does under the current
/// mode, instead of always previewing the seed-only pick.
fn smart_recommendations(
    inventory: &runtime::ModelInventory,
    allow_cross_provider_diversity: bool,
    provider_allowlist: &[String],
    feedback: Option<&runtime::RouteOutcomeSummary>,
    learned: &LearnedSpecialtyHint,
    policy: runtime::SmartPolicy,
) -> Vec<SmartSettingsRecommendation> {
    let options = AutoAssignmentOptions {
        allow_cross_provider_diversity,
        provider_allowlist: provider_allowlist.to_vec(),
        policy,
    };
    let plan = recommend_auto_assignments_with_learned_specialty(inventory, &options, feedback, learned);
    let mut recommendations: Vec<SmartSettingsRecommendation> = plan
        .assignments
        .iter()
        .filter_map(|assignment| match &assignment.target {
            RoutingTarget::Subagent(profile) => Some(SmartSettingsRecommendation {
                kind: SmartSettingsTargetKind::Subagent,
                key: profile.key().to_string(),
                selected_model: assignment.selected_model.clone(),
                confidence: confidence_label(assignment.confidence).to_string(),
                reason: assignment.reason.clone(),
                audit: assignment.audit.clone(),
            }),
            _ => None,
        })
        .collect();
    let role_fallbacks = recommend_role_fallbacks_with_learned_specialty(inventory, &options, learned);
    recommendations.extend(role_fallbacks.iter().filter_map(|assignment| {
        match &assignment.target {
            RoutingTarget::RoleFallback(role) => Some(SmartSettingsRecommendation {
                kind: SmartSettingsTargetKind::Role,
                key: role.key().to_string(),
                selected_model: assignment.selected_model.clone(),
                confidence: confidence_label(assignment.confidence).to_string(),
                reason: assignment.reason.clone(),
                audit: assignment.audit.clone(),
            }),
            _ => None,
        }
    }));
    recommendations
}

/// Per-turn output-token usage for the dashboard's usage sparkline, oldest
/// first. The durable turn trace records *cumulative* output tokens at each
/// turn's end, so deltas between consecutive turns give the per-turn cost. A
/// drop (session resume resets the cumulative baseline) is treated as a fresh
/// turn rather than producing a bogus underflow. Best-effort: a missing trace
/// (no turns yet, or `-p`/headless runs that don't persist) yields an empty
/// series and the dashboard omits the chart.
fn session_turn_output_usage(cwd: &Path, session_id: &str) -> Vec<u32> {
    let records = runtime::turn_trace::read_session(cwd, session_id);
    let mut previous = 0u32;
    let mut series = Vec::with_capacity(records.len());
    for record in records {
        let current = record.output_tokens;
        series.push(current.checked_sub(previous).unwrap_or(current));
        previous = current;
    }
    series
}

pub(crate) fn apply_smart_settings_commit(commit: &SmartSettingsCommit) -> io::Result<String> {
    write_global_smart_enabled(commit.enabled)?;
    write_global_smart_allow_cross_provider_diversity(commit.allow_cross_provider_diversity)?;
    write_global_smart_feedback_informed_auto(commit.feedback_informed_auto)?;
    for (role, update) in &commit.roles {
        write_global_smart_role(role, &role_update_from_smart_settings(update))?;
    }
    for (subagent, update) in &commit.subagents {
        write_global_smart_subagent(subagent, &role_update_from_smart_settings(update))?;
    }
    let override_count = commit
        .roles
        .iter()
        .chain(commit.subagents.iter())
        .filter(|(_, update)| !matches!(update, SmartSettingsUpdate::Auto))
        .count();
    Ok(format!(
        "Saved Smart Router settings. Smart is {} with {override_count} override(s).",
        if commit.enabled { "ON" } else { "OFF" }
    ))
}

fn smart_model_pool_notes() -> Vec<String> {
    api::custom_provider_usability_catalog()
        .into_iter()
        .filter(|provider| provider.requires_auth && !provider.usable && !provider.models.is_empty())
        .map(|provider| {
            let credentials = if provider.credential_env_vars.is_empty() {
                "API key".to_string()
            } else {
                provider.credential_env_vars.join("/")
            };
            format!(
                "{} hidden: missing {credentials} for {} model(s)",
                provider.name,
                provider.models.len()
            )
        })
        .collect()
}

fn smart_settings_update_from_role_update(update: &SmartRoleUpdate) -> SmartSettingsUpdate {
    match update {
        SmartRoleUpdate::Auto => SmartSettingsUpdate::Auto,
        SmartRoleUpdate::ExactPin { model } => SmartSettingsUpdate::ExactPin {
            model: model.clone(),
        },
        SmartRoleUpdate::FamilyLock {
            provider,
            family,
            class,
            freshness,
        } => SmartSettingsUpdate::FamilyLock {
            provider: provider.clone(),
            family: family.clone(),
            class: class.clone(),
            freshness: match freshness {
                SmartFreshness::Latest => SmartSettingsFreshness::Latest,
                SmartFreshness::LatestStable => SmartSettingsFreshness::LatestStable,
            },
        },
    }
}

fn role_update_from_smart_settings(update: &SmartSettingsUpdate) -> SmartRoleUpdate {
    match update {
        SmartSettingsUpdate::Auto => SmartRoleUpdate::Auto,
        SmartSettingsUpdate::ExactPin { model } => SmartRoleUpdate::ExactPin {
            model: model.clone(),
        },
        SmartSettingsUpdate::FamilyLock {
            provider,
            family,
            class,
            freshness,
        } => SmartRoleUpdate::FamilyLock {
            provider: provider.clone(),
            family: family.clone(),
            class: class.clone(),
            freshness: match freshness {
                SmartSettingsFreshness::Latest => SmartFreshness::Latest,
                SmartSettingsFreshness::LatestStable => SmartFreshness::LatestStable,
            },
        },
    }
}

fn confidence_label(confidence: AssignmentConfidence) -> &'static str {
    match confidence {
        AssignmentConfidence::High => "High",
        AssignmentConfidence::Medium => "Medium",
        AssignmentConfidence::Low => "Low",
    }
}

fn assignment_options(snapshot: &SmartSettingsSnapshot) -> AutoAssignmentOptions {
    AutoAssignmentOptions {
        allow_cross_provider_diversity: snapshot.allow_cross_provider_diversity,
        provider_allowlist: snapshot.provider_allowlist.clone(),
        policy: snapshot.policy,
    }
}

fn role_display_label(role: &str) -> &str {
    match role {
        "default" => "Default",
        "fast" => "Fast",
        "coding" => "Coding",
        "debugging" => "Debugging",
        "verifier" => "Verifier",
        "reviewer" => "Reviewer",
        "analysis" => "Analysis",
        "research" => "Research",
        "writing" => "Writing",
        "design" => "Design",
        "judge" => "Judge",
        "synthesizer" => "Synthesizer",
        _ => role,
    }
}

fn subagent_display_label(key: &str) -> String {
    key.trim_start_matches("custom:")
        .replace('-', " ")
        .split_whitespace()
        .map(|part| {
            let mut chars = part.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + chars.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn smart_role_keys() -> &'static [&'static str] {
    &[
        "default", "fast", "coding", "debugging", "verifier", "reviewer",
        "analysis", "research", "writing", "design", "judge", "synthesizer",
    ]
}

fn resolve_smart_pin_model(default_model: &str, model_query: &str) -> Result<String, String> {
    let inventory = connected_model_inventory(default_model);
    if let Some(exact) = inventory.find(model_query) {
        return Ok(exact.id().to_string());
    }

    let query_lower = model_query.to_ascii_lowercase();
    let matches: Vec<_> = inventory
        .models()
        .iter()
        .filter(|m| m.id().to_ascii_lowercase().contains(&query_lower))
        .collect();

    match matches.len().cmp(&1) {
        std::cmp::Ordering::Equal => Ok(matches[0].id().to_string()),
        std::cmp::Ordering::Greater => {
            let list = matches.iter().map(|m| m.id()).collect::<Vec<_>>().join(", ");
            Err(format!(
                "Multiple models match `{model_query}`: {list}. Please be more specific."
            ))
        }
        std::cmp::Ordering::Less => {
            let list = inventory
                .models()
                .iter()
                .map(runtime::ModelDescriptor::id)
                .collect::<Vec<_>>()
                .join("\n  ");
            Err(format!(
                "Model `{model_query}` not found in usable models.\nUsable models:\n  {list}"
            ))
        }
    }
}

/// One-line reminder that saved overrides take priority over Auto routing.
/// Surfaced at the moment Smart is turned ON (and on `/smart status`) when
/// overrides exist: a stale pin silently freezes routing for that role, which
/// reads as "auto is broken" — say so exactly where the user is looking.
fn override_bypass_warning() -> Option<String> {
    let snapshot = read_global_smart_settings().ok()?;
    let count = snapshot.roles.len() + snapshot.subagents.len();
    (count > 0).then(|| {
        format!(
            "Note: {count} saved override(s) (pinned roles/subagents) take priority over Auto \
             routing for those targets. Run `/smart reset` to go full-auto."
        )
    })
}

/// The cross-provider-diversity setting the deep-gate VERIFY leg routes under.
///
/// Verify leg diversity is governed SOLELY by `smart.verifyCrossProvider`
/// (default true), decoupled from the global worker-diversity flag
/// (`allow_cross_provider_diversity`): the verify leg stays cross-model by
/// default even when global diversity is off, and honors
/// `verifyCrossProvider: false` even when global diversity is on. Extracted as
/// a named seam so the decoupling is one unit-testable decision rather than an
/// inline field read.
fn deep_verify_allow_cross_provider(snapshot: &SmartSettingsSnapshot) -> bool {
    snapshot.verify_cross_provider
}

/// Resolve the model the deep gate's always-on VERIFY legs should run on: the
/// Smart route for the **Verifier role** against the current main model,
/// honoring a saved `verifier` role override and the verify-leg cross-provider
/// switch (`smart.verifyCrossProvider`, decoupled from the global
/// worker-diversity flag — see [`deep_verify_allow_cross_provider`]). `None`
/// when Smart is off, settings are unreadable, or the route resolves to the
/// main model itself (single-provider pool, diversity off, or a same-model
/// pin) — callers keep the native same-model verify in those cases.
#[cfg(test)]
pub(crate) fn route_deep_verify_model(main_model: &str) -> Option<String> {
    let deep_tier_models = read_global_smart_settings().ok()?.deep_tier_models;
    route_deep_verify_candidates(main_model, &deep_tier_models)
        .into_iter()
        .next()
}

fn deep_verify_request_and_primary(
    request: RouteRequest,
    inventory: &runtime::ModelInventory,
) -> (RouteRequest, String) {
    let resolved = resolve_deep_verify_primary(&request, inventory);
    (request, resolved)
}

/// Resolve the configured deep-verifier primary without silently replacing an
/// exact pin when that model is absent from the current connected inventory.
/// The generic router may auto-select in that case, but for this leg a pin is
/// explicit user intent: the host either constructs that pinned client or
/// skips the cross-model leg and keeps native verification.
fn resolve_deep_verify_primary(
    request: &RouteRequest,
    inventory: &runtime::ModelInventory,
) -> String {
    match request.override_rule.as_ref() {
        Some(RoleOverride::Pin(model)) if !model.is_empty() => model.clone(),
        _ => route_model(request, inventory).resolved_model,
    }
}

/// Build the Verifier-role route request and inventory used by
/// [`route_deep_verify_candidates`], returning the trimmed main model alongside
/// them. `None` in exactly the cases the caller treats as "no cross verifier":
/// empty main model, Smart off, or unreadable settings.
fn deep_verify_route_request(
    main_model: &str,
) -> Option<(
    RouteRequest,
    runtime::ModelInventory,
    String,
    SmartSettingsSnapshot,
)> {
    let main_model = main_model.trim();
    if main_model.is_empty() {
        return None;
    }
    let snapshot = read_global_smart_settings().ok()?;
    if !snapshot.enabled {
        return None;
    }
    let inventory = connected_model_inventory(main_model);
    let mut request = RouteRequest::for_target(
        RoutingTarget::RoleFallback(RouteRole::Verifier),
        RouteRole::Verifier,
        main_model,
    );
    request.override_rule = snapshot
        .roles
        .get("verifier")
        .and_then(role_override_from_update);
    request = request.with_context(RoutePolicyContext {
        risk: RouteTaskRisk::Low,
        // Large, deliberately: the verifier is the quality bar, and Medium kept
        // the router's Deep tier ineligible, so classic-policy verification
        // landed on mid-tier models. One smart verification beats several
        // mediocre ones — route the verify leg like a hard task.
        complexity: RouteTaskComplexity::Large,
        prior_failures: 0,
        context_need: RouteContextNeed::LocalFiles,
        tool_need: RouteToolNeed::ReadOnly,
        verification_need: RouteVerificationNeed::Focused,
        route_shape: None,
        lane: None,
        // Verify leg diversity is governed solely by smart.verifyCrossProvider
        // (default true), decoupled from the global worker-diversity flag.
        allow_cross_provider_diversity: deep_verify_allow_cross_provider(&snapshot),
        provider_allowlist: snapshot.provider_allowlist.clone(),
        feedback: deep_verify_feedback_hint(&snapshot),
        audit_notes: vec!["deep-gate-verify".to_string()],
        cooldown_providers: Vec::new(),
        // The deep-verify leg builds its own context and never reads the
        // sub-agent-batch headroom state, so it stays headroom-neutral (empty +
        // threshold 0 = no graded penalty), same as its empty cooldown set.
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        // Phase 6 learned-specialty is not wired into this leg yet — it has
        // its own settings snapshot path (`read_global_smart_settings`, not
        // `smart_router::apply::SmartRouteContext`), so computing the hint
        // here would duplicate that batch-load rather than reuse it. Stays
        // disabled (byte-identical) until a future phase decides to route
        // this leg through the same shadow/on/off machinery.
        learned_specialty: LearnedSpecialtyHint::disabled(),
        // Architect ladders the verify leg Deep→Strong→Balanced so the
        // checker outclasses the implementer; classic keeps the historical
        // Strong/Balanced ladder this leg's Large-complexity context opens.
        policy: snapshot.policy,
    });
    Some((request, inventory, main_model.to_string(), snapshot))
}

/// Resolve the deep gate's VERIFY leg as an **ordered candidate list**, top
/// choice first, so a verifier that is itself hard rate-limited can fail over
/// to the next available model. Architect policy uses the configured deep-tier
/// pool verbatim in preference order; classic policy keeps the configured
/// verifier primary plus Smart Router fallbacks. The list is deduplicated, and
/// the main model remains the terminal native fallback rather than appearing
/// as a duplicate cross-model candidate.
pub(crate) fn route_deep_verify_candidates(
    main_model: &str,
    deep_tier_models: &[String],
) -> Vec<String> {
    let Some((request, inventory, main_model, snapshot)) = deep_verify_route_request(main_model)
    else {
        return Vec::new();
    };
    if snapshot.policy == runtime::SmartPolicy::Architect {
        let mut ordered = Vec::with_capacity(deep_tier_models.len());
        for candidate in deep_tier_models {
            if !same_deep_tier_model(candidate, &main_model)
                && !ordered
                    .iter()
                    .any(|existing| same_deep_tier_model(candidate, existing))
            {
                ordered.push(candidate.clone());
            }
        }
        return ordered;
    }

    let (request, primary) = deep_verify_request_and_primary(request, &inventory);
    if primary == main_model {
        return Vec::new();
    }

    let mut ordered = vec![primary.clone()];
    for candidate in runtime::route_model_fallback_candidates(
        &request,
        &inventory,
        &primary,
        snapshot.fallback_candidate_limit,
    ) {
        if candidate != main_model
            && !ordered.contains(&candidate)
        {
            ordered.push(candidate);
        }
    }
    ordered
}

fn same_deep_tier_model(left: &str, right: &String) -> bool {
    runtime::is_deep_tier_model(left, std::slice::from_ref(right))
        || runtime::is_deep_tier_model(right, &[left.to_string()])
}

/// Ordered Architect PLAN/VERIFY pool from the merged smart-settings SSOT.
pub(crate) fn configured_deep_tier_models() -> Vec<String> {
    tools::smart_deep_tier_models()
}

pub(crate) fn execute_deep_tier_command(
    cwd: &Path,
    action: &commands::DeepTierAction,
) -> Result<String, String> {
    let setting = tools::smart_deep_tier_models_for(cwd)
        .ok_or_else(|| "Deep-tier pool: could not load merged settings".to_string())?;
    match action {
        commands::DeepTierAction::Show => {
            let source = if setting.configured {
                "configured"
            } else {
                "built-in default"
            };
            let mut lines = vec![format!("Deep-tier pool ({source})")];
            lines.extend(
                setting
                    .models
                    .iter()
                    .enumerate()
                    .map(|(index, model)| format!("  {}. {model}", index + 1)),
            );
            lines.push(commands::DEEP_TIER_USAGE.to_string());
            Ok(lines.join("\n"))
        }
        commands::DeepTierAction::Add { model } => {
            let model = api::resolve_model_alias(model.trim());
            if let Some(index) = setting
                .models
                .iter()
                .position(|existing| runtime::deep_tier_model_matches(&model, existing))
            {
                return Ok(format!(
                    "{model} is already in the deep-tier pool at #{}. No changes made.",
                    index + 1
                ));
            }
            let mut models = setting.models;
            models.push(model.clone());
            write_project_deep_tier_models(cwd, Some(&models)).map_err(|error| {
                format!("Deep-tier pool: could not update project settings: {error}")
            })?;
            Ok(format!(
                "Added {model} to the deep-tier pool at #{}.\nApplies from next turn.",
                models.len()
            ))
        }
        commands::DeepTierAction::Remove { target } => {
            let index = target
                .parse::<usize>()
                .ok()
                .filter(|index| (1..=setting.models.len()).contains(index))
                .map(|index| index - 1)
                .or_else(|| {
                    setting
                        .models
                        .iter()
                        .position(|model| runtime::deep_tier_model_matches(target, model))
                })
                .ok_or_else(|| {
                    format!(
                        "No deep-tier model matches `{target}`.\n{}",
                        commands::DEEP_TIER_USAGE
                    )
                })?;
            if setting.models.len() == 1 {
                return Err(
                    "Cannot remove the last deep-tier model; use /tier reset to restore the built-in default."
                        .to_string(),
                );
            }
            let mut models = setting.models;
            let removed = models.remove(index);
            write_project_deep_tier_models(cwd, Some(&models)).map_err(|error| {
                format!("Deep-tier pool: could not update project settings: {error}")
            })?;
            Ok(format!(
                "Removed #{position} ({removed}) from the deep-tier pool.\nApplies from next turn.",
                position = index + 1
            ))
        }
        commands::DeepTierAction::Move { from, to } => {
            move_deep_tier_model(cwd, setting.models, *from, *to)
        }
        commands::DeepTierAction::Reset => {
            write_project_deep_tier_models(cwd, None).map_err(|error| {
                format!("Deep-tier pool: could not reset project settings: {error}")
            })?;
            if tools::smart_deep_tier_models_for(cwd).is_some_and(|setting| setting.configured) {
                write_project_deep_tier_models(cwd, Some(&[])).map_err(|error| {
                    format!("Deep-tier pool: could not reset project settings: {error}")
                })?;
            }
            Ok("Reset the deep-tier pool to the built-in default.\nApplies from next turn."
                .to_string())
        }
    }
}

fn move_deep_tier_model(
    cwd: &Path,
    mut models: Vec<String>,
    from: usize,
    to: usize,
) -> Result<String, String> {
    let len = models.len();
    if !(1..=len).contains(&from) || !(1..=len).contains(&to) {
        return Err(format!(
            "Deep-tier positions must be between 1 and {len}.\n{}",
            commands::DEEP_TIER_USAGE
        ));
    }
    if from == to {
        return Ok(format!(
            "Deep-tier model #{from} is already at #{to}. No changes made."
        ));
    }
    let model = models.remove(from - 1);
    models.insert(to - 1, model.clone());
    write_project_deep_tier_models(cwd, Some(&models))
        .map_err(|error| format!("Deep-tier pool: could not update project settings: {error}"))?;
    Ok(format!(
        "Moved #{from} ({model}) to #{to} in the deep-tier pool.\nApplies from next turn."
    ))
}

fn write_project_deep_tier_models(cwd: &Path, models: Option<&[String]>) -> io::Result<()> {
    let path = super::session_preferences::project_settings_path(cwd);
    update_settings_file(&path, |root| {
        let smart = object_child(root, "smart")?;
        match models {
            Some(models) => {
                smart.insert(
                    "deepTierModels".to_string(),
                    Value::Array(models.iter().cloned().map(Value::String).collect()),
                );
            }
            None => {
                smart.remove("deepTierModels");
            }
        }
        Ok(())
    })
}

/// Whether Smart routing is on at all, for surfaces that only need to say
/// whether delegation is being routed (the `/model` switch report). Unreadable
/// settings answer `false` — the same fail-safe direction as every other
/// reader here, and the honest answer for a message that would otherwise
/// promise routing that will not happen.
pub(crate) fn smart_routing_enabled() -> bool {
    read_global_smart_settings().is_ok_and(|snapshot| snapshot.enabled)
}

/// Whether this turn's deep gate must keep PLAN and VERIFY on the configured
/// deep-tier pool. Independent of `smart.execSwap` and turn difficulty.
pub(crate) fn architect_deep_lanes_enabled() -> bool {
    read_global_smart_settings().is_ok_and(|snapshot| {
        snapshot.enabled && snapshot.policy == runtime::SmartPolicy::Architect
    })
}

/// Resolve the model a quota-exhausted main-model turn should fall back to: the
/// top-ranked auto route candidate on a **different provider** than the main
/// model. `None` when Smart is off, `smart.quotaFallback` is off, settings are
/// unreadable, the pool is single-provider, or every ranked alternative shares
/// the main model's provider — in every such case the caller installs no
/// fallback and a quota-exhausted turn fails exactly as it did before the
/// feature existed.
///
/// Cross-provider is the whole point: falling back to another model on the SAME
/// throttled account/provider would just hit the same quota wall, so the
/// candidate's [`api::detect_provider_kind`] must differ from the main model's.
/// Mirrors the deep-verifier candidate route's inventory/request construction but
/// routes the general `Default` role (an equivalent main-turn peer, not the
/// read-only Verifier) and ranks alternates via
/// [`runtime::route_model_fallback_candidates`] — the same best-of-breed scorer
/// the sub-agent quota-escape path uses — instead of the single primary route.
/// The quota-wait band as a [`Duration`], from `smart.quotaWaitBandMinutes`
/// (default 15 min; `0` disables). Read every turn entry and pushed to the
/// runtime via `set_quota_wait_band`. Unlike [`route_quota_fallback_model`] this
/// is NOT gated on `smart.quotaFallback`: holding on the main model for an
/// imminent reset is worthwhile even when no cross-provider peer is available.
/// A read error degrades to the default band rather than disabling the feature.
pub(crate) fn quota_wait_band() -> std::time::Duration {
    let minutes = read_global_smart_settings()
        .map(|snapshot| snapshot.quota_wait_band_minutes)
        .unwrap_or(DEFAULT_QUOTA_WAIT_BAND_MINUTES);
    std::time::Duration::from_secs(minutes.saturating_mul(60))
}

pub(crate) fn route_quota_fallback_model(main_model: &str) -> Option<String> {
    let main_model = main_model.trim();
    if main_model.is_empty() {
        return None;
    }
    let snapshot = read_global_smart_settings().ok()?;
    if !snapshot.enabled || !snapshot.quota_fallback {
        return None;
    }
    let main_provider = api::detect_provider_kind(main_model);
    let inventory = connected_model_inventory(main_model);
    let mut request = RouteRequest::for_target(
        RoutingTarget::RoleFallback(RouteRole::Default),
        RouteRole::Default,
        main_model,
    );
    request = request.with_context(RoutePolicyContext {
        risk: RouteTaskRisk::Medium,
        complexity: RouteTaskComplexity::Medium,
        prior_failures: 0,
        context_need: RouteContextNeed::LocalFiles,
        tool_need: RouteToolNeed::Write,
        verification_need: RouteVerificationNeed::Full,
        route_shape: None,
        lane: None,
        // A quota fallback that stays on the same (throttled) provider is
        // pointless, so this leg always asks for cross-provider candidates
        // regardless of the global worker-diversity flag; the explicit
        // provider-kind filter below is the hard guarantee.
        allow_cross_provider_diversity: true,
        provider_allowlist: snapshot.provider_allowlist.clone(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: vec!["quota-fallback".to_string()],
        cooldown_providers: Vec::new(),
        // The hard different-provider filter below already guarantees this leg
        // escapes the throttled provider, so the graded headroom penalty adds
        // nothing here — stay headroom-neutral (empty + threshold 0), same as
        // the deep-verify leg above.
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: snapshot.policy,
    });
    let candidates = runtime::route_model_fallback_candidates(
        &request,
        &inventory,
        main_model,
        snapshot.fallback_candidate_limit,
    );
    candidates
        .into_iter()
        .find(|candidate| api::detect_provider_kind(candidate) != main_provider)
}

/// Resolve the Architect contract's implementer: the Smart route for the
/// **Coding role** against the current main model, honoring a saved `coding`
/// role override. This is what makes the contract dynamic — "the snappy,
/// hard-working model does the work" follows the router's best-of-breed
/// Coding pick (tier ladder + specialty seed + learned outcomes + headroom),
/// never a hardcoded model name.
///
/// `None` in every case the contract must step aside: Smart off, policy
/// `classic`, unreadable settings, or no candidate outside the planner pool —
/// an explicit `coding` pin to a pool member is user intent, so the native
/// model implements and no contract is installed.
///
/// `deep_tier_models` is the caller's already-resolved `smart.deepTierModels`
/// pool, so the implementer search excludes exactly the models the contract
/// gate treats as planners. An AUTO route that lands on a pool member walks
/// the ranked fallbacks rather than giving up: with a customized pool the
/// top Coding pick can legitimately be a planner, and abandoning the contract
/// there would silently return the turn to pre-contract behavior.
pub(crate) fn route_exec_impl_model(
    main_model: &str,
    deep_tier_models: &[String],
) -> Option<String> {
    let main_model = main_model.trim();
    if main_model.is_empty() {
        return None;
    }
    let snapshot = read_global_smart_settings().ok()?;
    if !snapshot.enabled || snapshot.policy != runtime::SmartPolicy::Architect {
        return None;
    }
    let inventory = connected_model_inventory(main_model);
    let mut request = RouteRequest::for_target(
        RoutingTarget::RoleFallback(RouteRole::Coding),
        RouteRole::Coding,
        main_model,
    );
    request.override_rule = snapshot
        .roles
        .get("coding")
        .and_then(role_override_from_update);
    request = request.with_context(RoutePolicyContext {
        risk: RouteTaskRisk::Medium,
        // Medium, deliberately NOT Large: under architect the premium gate has
        // no complexity escape anyway, and a neutral midpoint keeps the pick
        // stable across the classifier's per-turn complexity swings.
        complexity: RouteTaskComplexity::Medium,
        context_need: RouteContextNeed::MultiFile,
        tool_need: RouteToolNeed::Write,
        verification_need: RouteVerificationNeed::Focused,
        allow_cross_provider_diversity: snapshot.allow_cross_provider_diversity,
        provider_allowlist: snapshot.provider_allowlist.clone(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: vec!["architect-exec-impl".to_string()],
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: snapshot.policy,
        ..Default::default()
    });
    // An explicit `coding` pin is user intent and is never second-guessed: a
    // pin to a planner means "the native model implements", so the contract
    // steps aside rather than shopping for a substitute.
    if let Some(RoleOverride::Pin(model)) = request.override_rule.as_ref() {
        if !model.is_empty() {
            return (model != main_model
                && !runtime::is_deep_tier_model(model, deep_tier_models))
            .then(|| model.clone());
        }
    }
    let decision = route_model(&request, &inventory);
    let usable = |candidate: &String| {
        candidate != main_model && !runtime::is_deep_tier_model(candidate, deep_tier_models)
    };
    if usable(&decision.resolved_model) {
        return Some(decision.resolved_model);
    }
    runtime::route_model_fallback_candidates(
        &request,
        &inventory,
        &decision.resolved_model,
        snapshot.fallback_candidate_limit,
    )
    .into_iter()
    .find(|candidate| usable(candidate))
}

/// Resolve the confidence-cascade escalation target: the model an armed
/// cascade turn (the previous turn verbalized LOW confidence in its own
/// result) runs on instead of the session model. Routed like a hard
/// implementation task — complexity `Large`, deliberately, so the Deep pool
/// (fable/sol) is eligible (`implementation_route_model_allowed` admits
/// premium models only at Large; the same reasoning as the deep-verify leg).
///
/// SAME-provider only: the escalation rides the bound client's wire model id
/// (`ConversationRuntime::set_escalation_model_override`), it never swaps
/// clients — so a route that resolves cross-provider or back to the main
/// model yields `None` and the turn escalates on effort alone.
pub(crate) fn route_cascade_escalation_model(main_model: &str) -> Option<String> {
    let main_model = main_model.trim();
    if main_model.is_empty() {
        return None;
    }
    let snapshot = read_global_smart_settings().ok()?;
    if !snapshot.enabled {
        return None;
    }
    let main_provider = api::detect_provider_kind(main_model);
    let inventory = connected_model_inventory(main_model);
    let mut request = RouteRequest::for_target(
        RoutingTarget::RoleFallback(RouteRole::Coding),
        RouteRole::Coding,
        main_model,
    );
    // Only the fields that make this leg what it is; everything defaulted is
    // the type's own neutral value (`..Default::default()`), so adding a
    // field to `RoutePolicyContext` never forces an edit here — the literal-
    // cascade landmine the sibling legs still carry.
    request = request.with_context(RoutePolicyContext {
        risk: RouteTaskRisk::Medium,
        // Large, deliberately: Deep-pool eligibility
        // (`implementation_route_model_allowed` admits premium models only
        // at Large — the deep-verify leg's reasoning).
        complexity: RouteTaskComplexity::Large,
        // The previous turn effectively failed (the model itself said so):
        // a real prior-failure signal, not a synthetic one.
        prior_failures: 1,
        context_need: RouteContextNeed::MultiFile,
        tool_need: RouteToolNeed::Write,
        verification_need: RouteVerificationNeed::Full,
        provider_allowlist: snapshot.provider_allowlist.clone(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: vec!["confidence-cascade".to_string()],
        learned_specialty: LearnedSpecialtyHint::disabled(),
        // `policy` rides the `Default` (Classic) DELIBERATELY: this leg IS the
        // failure-escalation path (the previous turn verbalized low
        // confidence), so it keeps the classic Large-complexity escape into
        // the premium pool even when `smart.policy=architect` gates ordinary
        // implementation routes — architect's own escape hatch is exactly
        // "repeated real failures", which is what a cascade arm represents.
        ..Default::default()
    });
    let decision = route_model(&request, &inventory);
    let resolved = decision.resolved_model;
    (resolved != main_model && api::detect_provider_kind(&resolved) == main_provider)
        .then_some(resolved)
}

/// Env kill switch for the Smart per-turn dynamic effort floor:
/// `ZO_SMART_DYNAMIC_FLOOR=off` (or `0`) restores the pre-floor behavior —
/// every Smart turn rides the static heavy band (`xhigh..=max`) regardless
/// of how simple the ask reads.
pub(crate) const SMART_DYNAMIC_FLOOR_ENV: &str = "ZO_SMART_DYNAMIC_FLOOR";

fn smart_dynamic_floor_enabled() -> bool {
    !std::env::var(SMART_DYNAMIC_FLOOR_ENV).is_ok_and(|value| {
        let value = value.trim();
        value.eq_ignore_ascii_case("off") || value == "0"
    })
}

/// Per-turn dynamic effort band for `/effort smart`: `Some((floor, ceiling))`
/// when a trusted probe reads the turn simple enough to ride a LOWER band than
/// Smart's default `xhigh..=max`, or when deterministic broad-scope evidence
/// raises it. `None` keeps the default.
///
/// This is what makes Smart difficulty-proportional in latency, not just in
/// orchestration without letting keyword tables demote a turn: a probe-backed
/// trivial verdict floors at `low`, a probe-backed small verdict at `medium`,
/// and both keep `xhigh` as the CEILING so wire-side difficulty signals
/// (`api::resolve_effort_band`: heavy intent, large context, long ask) can
/// climb a mis-read turn back to exactly the old always-on floor — never
/// beyond.
///
/// The heavy end is proportional too: a `Large` turn (a migration, a
/// repo-wide sweep — what the classifier already gates `plan_first` on) floors
/// at `max` instead of `xhigh`, so the hardest asks start at the tier the
/// wire-side signals would otherwise have to earn one rung at a time. `Medium`
/// and `Unknown` return `None` and keep the default `xhigh..=max` band
/// unchanged — `Medium` is the classifier's bucket for every ordinary
/// implement/fix verb, so raising ITS floor would raise almost every real turn.
///
/// Provenance comes directly from probe fusion; this function never re-runs or
/// compares classifiers to guess where a verdict came from. The model itself
/// is never swapped (a per-turn model change would cold the prompt cache).
/// Escalation floors (grind/cascade) run AFTER this in
/// `grind_escalation::effective_turn_effort` and override it to `xhigh` — a
/// floor, so the `Large` band's higher `max` survives it.
///
/// A trusted non-`Large` `Design` verdict caps the ceiling at `xhigh`: measured
/// design quality saturated below max reasoning, while a Korean brief took
/// 842s against a 558s reference on the uncapped band. There is no special
/// design floor; the obsolete `high` override only compensated for the
/// keyword-derived `Small` demotion that provenance now forbids.
///
/// Env kill switch for the design-intent guidance reminder and its effort
/// band: `ZO_DESIGN_GUIDANCE=off` (or `0`) restores the pre-design-axis
/// behavior — no `[zo:design-guidance]` reminder, and a resolved `Design` turn
/// bands exactly like any other turn of its complexity (no `xhigh` cap).
pub(crate) const DESIGN_GUIDANCE_ENV: &str = "ZO_DESIGN_GUIDANCE";

pub(crate) fn design_guidance_enabled() -> bool {
    !std::env::var(DESIGN_GUIDANCE_ENV).is_ok_and(|value| {
        let value = value.trim();
        value.eq_ignore_ascii_case("off") || value == "0"
    })
}

pub(crate) fn smart_turn_effort_band_for_complexity(
    complexity: RouteTaskComplexity,
    intent: runtime::RouteTaskIntent,
    provenance: runtime::RouteAssessmentProvenance,
) -> Option<(api::EffortLevel, api::EffortLevel)> {
    if !smart_dynamic_floor_enabled() {
        return None;
    }
    // Broad-scope evidence raises unconditionally, including when it came from
    // the deterministic tables. No intent override may lower this floor.
    if complexity == RouteTaskComplexity::Large {
        return Some((api::EffortLevel::Max, api::EffortLevel::Ultra));
    }
    // Only a probe verdict that cleared fusion's confidence gate may lower the
    // default Smart floor or cap its ceiling.
    if provenance != runtime::RouteAssessmentProvenance::TrustedProbe {
        return None;
    }
    match complexity {
        RouteTaskComplexity::Trivial => Some((api::EffortLevel::Low, api::EffortLevel::Xhigh)),
        RouteTaskComplexity::Small => Some((api::EffortLevel::Medium, api::EffortLevel::Xhigh)),
        RouteTaskComplexity::Medium | RouteTaskComplexity::Unknown
            if intent == runtime::RouteTaskIntent::Design && design_guidance_enabled() =>
        {
            Some((api::EffortLevel::Xhigh, api::EffortLevel::Xhigh))
        }
        RouteTaskComplexity::Medium
        | RouteTaskComplexity::Large
        | RouteTaskComplexity::Unknown => None,
    }
}

/// Real feedback hint for the deep-verify leg's own route (`route_key`
/// `"deep-verify:leg"` — the SAME key `turn_controller::
/// record_deep_verdict_outcomes`'s did-run record writes to), gated on
/// `feedbackInformedAuto` exactly like the sub-agent spawn path's
/// `SmartRouteContext` (`tools::misc_tools::smart_router::apply`). Was
/// unconditionally `RouteFeedbackHint::disabled()` before Phase 4 — the
/// verify leg neither learned from history nor could BE learned from (no
/// outcome recorder wrote its route at all until Task 2 of that phase).
/// Best-effort: any read/parse failure degrades to `disabled()`, never a hard
/// error — a missing/corrupt outcome file must not break VERIFY routing.
fn deep_verify_feedback_hint(snapshot: &SmartSettingsSnapshot) -> RouteFeedbackHint {
    if !snapshot.feedback_informed_auto {
        return RouteFeedbackHint::disabled();
    }
    let Ok(cwd) = std::env::current_dir() else {
        return RouteFeedbackHint::disabled();
    };
    deep_verify_feedback_hint_at(snapshot, &cwd)
}

/// [`deep_verify_feedback_hint`] with the workspace injected. Split from the
/// wrapper so tests can pin the outcome-store location: the process cwd is
/// global mutable state, and a test reading it through `current_dir()` raced
/// the chdir-ing tests (which serialize on a *different* lock, `cwd_lock`) —
/// under parallel load the read resolved a foreign project slug mid-test and
/// found an empty store.
fn deep_verify_feedback_hint_at(
    snapshot: &SmartSettingsSnapshot,
    cwd: &std::path::Path,
) -> RouteFeedbackHint {
    if !snapshot.feedback_informed_auto {
        return RouteFeedbackHint::disabled();
    }
    let Ok(records) = read_route_outcomes(cwd) else {
        return RouteFeedbackHint::disabled();
    };
    // Same write-time canonicalization the spawn/verdict recorders use
    // (`api::resolve_model_alias`) — `tools::canonicalize_route_model_id` is
    // the identical one-liner, but that fn is crate-private to `tools`.
    let summary =
        summarize_route_outcomes_with_canonicalizer(&records, |raw| api::resolve_model_alias(raw.trim()));
    summary.feedback_hint_for_route_key("deep-verify:leg")
}

/// `/refine`'s promotion-gate evidence: how many learned-specialty entries
/// exist at all, plus display lines for the ones clearing the router's own
/// rung-admission predicate — the exact bar that changes routing once
/// `smart.learnedSpecialty` is `on`. Divergence stamps alone prove the
/// learned hint DIFFERS; an armed entry is the measured claim that it is
/// RIGHT, so recommendations key off this and never off stamp volume.
pub(crate) fn learned_promotion_evidence(cwd: &std::path::Path) -> (usize, Vec<String>) {
    let records = read_route_outcomes(cwd).unwrap_or_default();
    let hint = LearnedSpecialtyHint::compute(&records, epoch_seconds_now(), |raw| {
        api::resolve_model_alias(raw.trim())
    });
    let armed = hint
        .entries()
        .iter()
        .filter(|(_, _, entry)| entry.earns_rung_admission())
        .map(|(role, model, entry)| {
            format!(
                "{}: {model} ({:+}, {}\u{2030} confidence)",
                role_display_label(role.key()),
                entry.model_adjustment,
                entry.confidence_permille
            )
        })
        .collect();
    (hint.entries().len(), armed)
}

/// Seconds-since-epoch for [`LearnedSpecialtyHint::compute`]'s recency decay.
/// Mirrors `tools::misc_tools::smart_router::apply::epoch_seconds_now`
/// (crate-private there, so duplicated here rather than plumbed across the
/// crate boundary for one clock read).
fn epoch_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Phase 6 learned-specialty hint for the `/smart` dashboard/status preview,
/// gated EXACTLY like the live route's `apply.rs::learned_specialty_for_real_
/// request`: `off`/`shadow` inject nothing (the preview must show what live
/// routing ACTUALLY does today — shadow never changes the real pick, it only
/// logs a delta), `on` injects the same computed hint the live router blends
/// in for real. Without this gating, the dashboard would preview a
/// still-soaking or disabled learned signal as if it were already live.
/// Best-effort: any read/parse failure degrades to `disabled()`, never a hard
/// error — a missing/corrupt outcome file must not break the preview.
fn dashboard_learned_specialty_hint(snapshot: &SmartSettingsSnapshot) -> LearnedSpecialtyHint {
    if snapshot.learned_specialty != SmartLearnedSpecialtyMode::On {
        return LearnedSpecialtyHint::disabled();
    }
    let Ok(cwd) = std::env::current_dir() else {
        return LearnedSpecialtyHint::disabled();
    };
    let Ok(records) = read_route_outcomes(&cwd) else {
        return LearnedSpecialtyHint::disabled();
    };
    LearnedSpecialtyHint::compute(&records, epoch_seconds_now(), |raw| {
        api::resolve_model_alias(raw.trim())
    })
}

/// Lower a saved role override into the router's [`RoleOverride`]. `Auto` is
/// "no override" (`None`), mirroring `role_override_from_json` in the tools
/// crate's smart-router settings reader.
fn role_override_from_update(update: &SmartRoleUpdate) -> Option<RoleOverride> {
    match update {
        SmartRoleUpdate::Auto => None,
        SmartRoleUpdate::ExactPin { model } => Some(RoleOverride::Pin(model.clone())),
        SmartRoleUpdate::FamilyLock {
            provider,
            family,
            class,
            freshness,
        } => Some(RoleOverride::Family(
            RoleSelector::new()
                .provider(provider.clone())
                .family(family.clone())
                .class(class.clone())
                .freshness(match freshness {
                    SmartFreshness::Latest => FreshnessPolicy::Latest,
                    SmartFreshness::LatestStable => FreshnessPolicy::LatestStable,
                }),
        )),
    }
}

#[allow(clippy::too_many_lines)] // one flat command-grammar dispatch table, subcommand-per-arm
pub(crate) fn execute_smart_text_command(
    default_model: &str,
    arg: Option<&str>,
) -> Result<String, String> {
    let raw_arg = arg.unwrap_or("").trim();
    if raw_arg.is_empty() {
        return Ok("Smart Router settings open in the TUI. Use `/smart status` to view the current setup.\n\nQuick CLI commands:\n  /smart agents         Show builtin and loaded custom agent types\n  /smart on | off       Enable or disable Smart Router\n  /smart reset          Reset all subagent and role overrides to Auto\n  /smart pin <target> <model>   Pin a role or subagent to a model\n  /smart auto <target>          Reset a role or subagent override to Auto\n  /smart doctor         Show aggregate route outcome feedback + provenance/exploration/learned status\n  /smart execswap always|easy|never   When the Architect contract hands EXEC to the routed implementer\n  /smart explore on | off        Toggle deterministic exploration\n  /smart explore cadence <n>     Set the exploration cadence divisor (default 5)\n  /smart classifier deterministic | assisted | probed | off   Set the auto-route classifier mode (probed = fuse a Fast-tier self-assessment)\n  /smart learned shadow | on | off   Set the learned-specialty mode\n  /smart feedback on | off       Toggle feedback-informed auto routing\n  /smart diversity on | off      Toggle cross-provider diversity\n  /smart verify-cross on | off   Toggle the deep-gate VERIFY leg's cross-provider routing\n  /smart quota-fallback on | off Toggle auto cross-provider fallback on main-model quota exhaustion\n  /smart providers <csv> | clear Restrict (or unrestrict) the AUTO provider pool".to_string());
    }

    let parts: Vec<&str> = raw_arg.split_whitespace().collect();
    let subcommand = parts[0].to_ascii_lowercase();

    match subcommand.as_str() {
        "status" => render_smart_status(default_model, None).map_err(|error| error.to_string()),
        "agents" => render_smart_agents(default_model).map_err(|error| error.to_string()),
        "doctor" | "outcomes" => {
            render_smart_doctor(default_model).map_err(|error| error.to_string())
        }
        "on" | "enable" => {
            write_global_smart_enabled(true).map_err(|e| format!("Failed to enable Smart Router: {e}"))?;
            let mut message = "Smart Router enabled.".to_string();
            if let Some(warning) = override_bypass_warning() {
                message.push('\n');
                message.push_str(&warning);
            }
            Ok(message)
        }
        "off" | "disable" => {
            write_global_smart_enabled(false).map_err(|e| format!("Failed to disable Smart Router: {e}"))?;
            Ok("Smart Router disabled.".to_string())
        }
        "reset" => {
            write_global_reset_overrides().map_err(|e| format!("Failed to reset overrides: {e}"))?;
            Ok("All Smart Router overrides reset to Auto.".to_string())
        }
        "pin" => {
            if parts.len() < 3 {
                return Err(format!(
                    "Usage: /smart pin <target> <model>\nExample: /smart pin coding {}",
                    api::OPENAI_LATEST_MODEL_ALIAS
                ));
            }
            let target = parts[1];
            let model_query = parts[2..].join(" ");

            let is_role = normalize_role(target).is_some();
            let is_subagent = SubagentProfileId::parse(target).is_some();

            if !is_role && !is_subagent {
                return Err(format!(
                    "Unknown target `{target}`. Must be a valid role or subagent.\n\
                     Roles: default, fast, coding, debugging, verifier, reviewer, analysis, research, writing, design, judge, synthesizer\n\
                     Subagents: general-purpose, Explore, Plan, Verification, deep-research, code-reviewer, debugger, data-analyst, refactor, frontend-design, zo-guide, statusline-setup"
                ));
            }

            let resolved_model = resolve_smart_pin_model(default_model, &model_query)?;

            let update = SmartRoleUpdate::ExactPin { model: resolved_model.clone() };

            if is_role {
                write_global_smart_role(target, &update).map_err(|e| format!("Failed to save role pin: {e}"))?;
                Ok(format!("Pinned role `{target}` to model `{resolved_model}`."))
            } else {
                write_global_smart_subagent(target, &update).map_err(|e| format!("Failed to save subagent pin: {e}"))?;
                Ok(format!("Pinned subagent `{target}` to model `{resolved_model}`."))
            }
        }
        "auto" | "unpin" => {
            if parts.len() < 2 {
                return Err("Usage: /smart auto <target>  or  /smart unpin <target>\nExample: /smart auto coding".to_string());
            }
            let target = parts[1];
            let is_role = normalize_role(target).is_some();
            let is_subagent = SubagentProfileId::parse(target).is_some();

            if !is_role && !is_subagent {
                return Err(format!(
                    "Unknown target `{target}`. Must be a valid role or subagent."
                ));
            }

            let update = SmartRoleUpdate::Auto;
            if is_role {
                write_global_smart_role(target, &update).map_err(|e| format!("Failed to reset role: {e}"))?;
                Ok(format!("Reset role `{target}` to Auto recommendation."))
            } else {
                write_global_smart_subagent(target, &update).map_err(|e| format!("Failed to reset subagent: {e}"))?;
                Ok(format!("Reset subagent `{target}` to Auto recommendation."))
            }
        }
        "explore" => {
            let Some(action) = parts.get(1) else {
                return Err(
                    "Usage: /smart explore on|off  or  /smart explore cadence <n>".to_string(),
                );
            };
            match action.to_ascii_lowercase().as_str() {
                "on" => {
                    write_global_smart_exploration(true)
                        .map_err(|e| format!("Failed to enable exploration: {e}"))?;
                    Ok("Deterministic exploration enabled.".to_string())
                }
                "off" => {
                    write_global_smart_exploration(false)
                        .map_err(|e| format!("Failed to disable exploration: {e}"))?;
                    Ok("Deterministic exploration disabled.".to_string())
                }
                "cadence" => {
                    let Some(raw) = parts.get(2) else {
                        return Err("Usage: /smart explore cadence <n>\nExample: /smart explore cadence 5".to_string());
                    };
                    let cadence: usize = raw
                        .parse()
                        .ok()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| format!("Invalid cadence `{raw}`; must be a positive integer."))?;
                    write_global_smart_exploration_cadence(cadence)
                        .map_err(|e| format!("Failed to set exploration cadence: {e}"))?;
                    Ok(format!(
                        "Exploration cadence set to every {cadence} recorded outcome(s) per route."
                    ))
                }
                other => Err(format!(
                    "Unknown `/smart explore {other}`. Use `on`, `off`, or `cadence <n>`."
                )),
            }
        }
        "classifier" => {
            let Some(mode_arg) = parts.get(1) else {
                return Err(
                    "Usage: /smart classifier deterministic|assisted|probed|off".to_string(),
                );
            };
            let mode = mode_arg.to_ascii_lowercase();
            if !matches!(mode.as_str(), "deterministic" | "assisted" | "probed" | "off") {
                return Err(format!(
                    "Unknown `/smart classifier {mode}`. Use `deterministic`, `assisted`, `probed`, or `off`."
                ));
            }
            write_global_smart_auto_classifier(&mode)
                .map_err(|e| format!("Failed to set auto classifier mode: {e}"))?;
            let label = runtime::RouteAutoClassifierMode::from_settings_value(Some(
                &Value::String(mode.clone()),
            ))
            .status_label();
            Ok(format!("Auto classifier mode set to `{mode}` — {label}."))
        }
        "learned" => {
            let Some(mode_arg) = parts.get(1) else {
                return Err("Usage: /smart learned shadow|on|off".to_string());
            };
            let mode = match mode_arg.to_ascii_lowercase().as_str() {
                "off" => SmartLearnedSpecialtyMode::Off,
                "shadow" => SmartLearnedSpecialtyMode::Shadow,
                "on" => SmartLearnedSpecialtyMode::On,
                other => {
                    return Err(format!(
                        "Unknown `/smart learned {other}`. Use `shadow`, `on`, or `off`."
                    ));
                }
            };
            write_global_smart_learned_specialty(mode)
                .map_err(|e| format!("Failed to set learned specialty mode: {e}"))?;
            Ok(format!(
                "Learned specialty mode set to `{}`.",
                mode.as_settings_str()
            ))
        }
        "feedback" => {
            let Some(action) = parts.get(1) else {
                return Err("Usage: /smart feedback on|off".to_string());
            };
            match action.to_ascii_lowercase().as_str() {
                "on" => {
                    write_global_smart_feedback_informed_auto(true)
                        .map_err(|e| format!("Failed to enable feedback-informed auto: {e}"))?;
                    Ok("Feedback-informed auto routing enabled.".to_string())
                }
                "off" => {
                    write_global_smart_feedback_informed_auto(false)
                        .map_err(|e| format!("Failed to disable feedback-informed auto: {e}"))?;
                    Ok("Feedback-informed auto routing disabled.".to_string())
                }
                other => Err(format!("Unknown `/smart feedback {other}`. Use `on` or `off`.")),
            }
        }
        "diversity" => {
            let Some(action) = parts.get(1) else {
                return Err("Usage: /smart diversity on|off".to_string());
            };
            match action.to_ascii_lowercase().as_str() {
                "on" => {
                    write_global_smart_allow_cross_provider_diversity(true)
                        .map_err(|e| format!("Failed to enable cross-provider diversity: {e}"))?;
                    Ok("Cross-provider diversity enabled.".to_string())
                }
                "off" => {
                    write_global_smart_allow_cross_provider_diversity(false)
                        .map_err(|e| format!("Failed to disable cross-provider diversity: {e}"))?;
                    Ok("Cross-provider diversity disabled.".to_string())
                }
                other => Err(format!("Unknown `/smart diversity {other}`. Use `on` or `off`.")),
            }
        }
        "verify-cross" => {
            let Some(action) = parts.get(1) else {
                return Err("Usage: /smart verify-cross on|off".to_string());
            };
            match action.to_ascii_lowercase().as_str() {
                "on" => {
                    write_global_smart_verify_cross_provider(true)
                        .map_err(|e| format!("Failed to enable verify cross-provider: {e}"))?;
                    Ok("Deep-gate VERIFY leg cross-provider routing enabled.".to_string())
                }
                "off" => {
                    write_global_smart_verify_cross_provider(false)
                        .map_err(|e| format!("Failed to disable verify cross-provider: {e}"))?;
                    Ok("Deep-gate VERIFY leg cross-provider routing disabled (native-preferred).".to_string())
                }
                other => Err(format!("Unknown `/smart verify-cross {other}`. Use `on` or `off`.")),
            }
        }
        "policy" => {
            let Some(action) = parts.get(1) else {
                return Err("Usage: /smart policy architect|classic".to_string());
            };
            match action.to_ascii_lowercase().as_str() {
                "architect" => {
                    write_global_smart_policy(runtime::SmartPolicy::Architect)
                        .map_err(|e| format!("Failed to set smart policy: {e}"))?;
                    Ok("Smart policy set to architect — reserved deep models plan/orchestrate/verify; implementation routes to standard implementer models (escalation on repeated failures or an explicit pin).".to_string())
                }
                "classic" => {
                    write_global_smart_policy(runtime::SmartPolicy::Classic)
                        .map_err(|e| format!("Failed to set smart policy: {e}"))?;
                    Ok("Smart policy set to classic — pre-contract routing (Large-complexity tasks may route implementation to reserved models; Balanced-first verify ladder).".to_string())
                }
                other => Err(format!(
                    "Unknown `/smart policy {other}`. Use `architect` or `classic`."
                )),
            }
        }
        "execswap" | "exec-swap" => {
            let Some(action) = parts.get(1) else {
                return Err("Usage: /smart execswap always|easy|never".to_string());
            };
            let exec_swap = match action.to_ascii_lowercase().as_str() {
                "always" => tools::SmartExecSwap::Always,
                "easy" => tools::SmartExecSwap::Easy,
                "never" => tools::SmartExecSwap::Never,
                other => {
                    return Err(format!(
                        "Unknown `/smart execswap {other}`. Use `always`, `easy`, or `never`."
                    ))
                }
            };
            write_global_smart_exec_swap(exec_swap)
                .map_err(|e| format!("Failed to set exec swap: {e}"))?;
            Ok(match exec_swap {
                tools::SmartExecSwap::Always => "EXEC swap set to always (default) — the deep session model plans and verifies, the routed implementer writes every diff. Two failed implementer attempts escalate back to the session model.".to_string(),
                tools::SmartExecSwap::Easy => "EXEC swap set to easy — only the lowest complexity band hands off; the session model implements everything else itself.".to_string(),
                tools::SmartExecSwap::Never => "EXEC swap set to never — every EXEC leg stays on the session model. PLAN and VERIFY still use the deep-tier pool.".to_string(),
            })
        }
        "quota-fallback" => {
            let Some(action) = parts.get(1) else {
                return Err("Usage: /smart quota-fallback on|off".to_string());
            };
            match action.to_ascii_lowercase().as_str() {
                "on" => {
                    write_global_smart_quota_fallback(true)
                        .map_err(|e| format!("Failed to enable quota fallback: {e}"))?;
                    Ok("Quota fallback enabled — a rate-limited main model auto-continues the turn on an equivalent other-provider model.".to_string())
                }
                "off" => {
                    write_global_smart_quota_fallback(false)
                        .map_err(|e| format!("Failed to disable quota fallback: {e}"))?;
                    Ok("Quota fallback disabled — a quota-exhausted turn now fails instead of switching provider.".to_string())
                }
                other => Err(format!("Unknown `/smart quota-fallback {other}`. Use `on` or `off`.")),
            }
        }
        "providers" => {
            if parts.len() < 2 {
                return Err(
                    "Usage: /smart providers <csv>|clear\nExample: /smart providers anthropic,openai".to_string(),
                );
            }
            let raw = parts[1..].join(" ");
            if raw.eq_ignore_ascii_case("clear") {
                write_global_smart_provider_allowlist(&[])
                    .map_err(|e| format!("Failed to clear provider allowlist: {e}"))?;
                Ok("Provider allowlist cleared — AUTO may pick any connected provider.".to_string())
            } else {
                let providers: Vec<String> = raw
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string)
                    .collect();
                if providers.is_empty() {
                    return Err(
                        "Usage: /smart providers <csv>|clear\nExample: /smart providers anthropic,openai".to_string(),
                    );
                }
                write_global_smart_provider_allowlist(&providers)
                    .map_err(|e| format!("Failed to set provider allowlist: {e}"))?;
                Ok(format!(
                    "Provider allowlist set to: {}.",
                    providers.join(", ")
                ))
            }
        }
        _ => Err(format!(
            "Unsupported subcommand `{subcommand}`. Use `/smart` to see available commands."
        )),
    }
}

fn render_smart_doctor(default_model: &str) -> io::Result<String> {
    let cwd = std::env::current_dir()?;
    let path = runtime::route_outcome_log_path(&cwd);
    let records = read_route_outcomes(&cwd)?;
    let summary = runtime::summarize_route_outcomes(&records);
    let mut report = render_smart_doctor_summary(&path, &summary);
    // P7: every routing decision explainable from `/smart doctor` alone,
    // instead of JSONL archaeology — sections (a)-(f) of the smart-auto
    // routing plan's Phase 7. Best-effort throughout: a settings-read failure
    // degrades individual sections rather than failing the whole doctor.
    let snapshot = read_global_smart_settings().ok();
    // First, deliberately: before any aggregate can be interpreted, the reader
    // needs to know which features actually RAN to produce it.
    report.push_str(&smart_doctor_harness_section());
    report.push_str(&smart_doctor_verdict_section(&records));
    if let Some(section) = smart_doctor_verify_pair_section(&records) {
        report.push_str(&section);
    }
    report.push_str(&smart_doctor_learned_shadow_section(
        &records,
        snapshot.as_ref(),
    ));
    report.push_str(&smart_doctor_exploration_section(snapshot.as_ref(), &summary));
    report.push_str(&smart_doctor_canonical_merge_section(&records));
    report.push_str(&smart_doctor_provenance_section(default_model));
    report.push_str(&smart_doctor_pin_awareness_section(&records));
    report.push_str(&smart_doctor_calibration_section(&records));
    Ok(report)
}

/// Column width for a feature label in the harness table — COMPUTED from the
/// longest [`telemetry::HarnessFeature::label`] at build time, not written
/// down. It was a hand-measured `32`, which silently became wrong the moment a
/// feature with a longer label was added: that row's count jumped a column and
/// the "scan straight down the counts" property the table is FOR quietly
/// broke (caught by `harness_rows_align_their_count_column`, but only after
/// the fact). Deriving it means adding a feature widens the column instead.
///
/// Labels are ASCII by construction, so byte length is display width.
const HARNESS_LABEL_WIDTH: usize = {
    let features = telemetry::HarnessFeature::all();
    let mut widest = 0;
    let mut index = 0;
    while index < features.len() {
        let width = features[index].label().len();
        if width > widest {
            widest = width;
        }
        index += 1;
    }
    widest
};

/// Harness firing attestation: which fail-open features actually RAN in this
/// process, and — for the ones that did not — whether they FAILED or a gate
/// declined them.
///
/// This section answers "is the feature I believe is on actually on?" with no
/// env var, no grep, and no JSONL archaeology. Every past instance of that
/// question was answered days or weeks late precisely because a fail-open path
/// emitted nothing at all: a routing probe that failed on 100% of calls, and a
/// design axis that a pinned effort made structurally unreachable.
///
/// Scope note, stated in the output too: the ledger is PROCESS-wide. It starts
/// empty at launch and counts this binary's own work, sub-agents included.
/// That is deliberate — liveness is a property of the running build, so a
/// persisted count would report some earlier build's behavior as if it were
/// this one's.
fn smart_doctor_harness_section() -> String {
    let ledger = telemetry::harness_attest_snapshot();
    let mut lines = vec![
        String::new(),
        "Harness firings (this process)".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    // Printed before the empty check: a run's arm is worth stating even when
    // nothing was reached, since "ablated and never reached" and "not ablated
    // and never reached" are the same empty table with opposite meanings.
    let ablation = telemetry::ablation_set();
    if !ablation.is_empty() {
        lines.push(format!(
            "  ablation active ({}): {} — held out of this run",
            telemetry::ABLATION_ENV,
            ablation.keys().join(", ")
        ));
    }
    if ledger.is_empty() {
        lines.push(
            "  (nothing recorded yet — no attested feature has been reached in this process)"
                .to_string(),
        );
        return lines.join("\n");
    }
    let leaks = ledger.ablation_leaks();
    if !leaks.is_empty() {
        lines.push(format!(
            "  ! {} fired while ablated — this run is NOT a valid control arm",
            leaks
                .iter()
                .map(|feature| feature.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let failing = ledger.failing_features();
    if !failing.is_empty() {
        lines.push(format!(
            "  ! {} never succeeded and failed at least once — see the rows below",
            failing
                .iter()
                .map(|feature| feature.label())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    for (feature, attestation) in ledger.rows() {
        lines.extend(harness_attest_rows(feature, &attestation));
    }
    lines.join("\n")
}

/// One feature's rows: the counter line, plus a precondition line only when the
/// feature recorded NOTHING — that is the single case where the counts alone
/// cannot say whether zero is a defect or the expected reading.
fn harness_attest_rows(
    feature: telemetry::HarnessFeature,
    attestation: &telemetry::FeatureAttestation,
) -> Vec<String> {
    use std::fmt::Write as _;

    let health = attestation.health();
    let tag = match health {
        telemetry::FeatureHealth::Alive => "fired",
        telemetry::FeatureHealth::Failing => "FAILING",
        telemetry::FeatureHealth::GatedOff => "gated",
        telemetry::FeatureHealth::Silent => "idle",
        telemetry::FeatureHealth::Ablated => "ablated",
        telemetry::FeatureHealth::AblationLeak => "LEAKED",
    };
    let mut row = format!(
        "  [{tag:>7}] {:<width$} {:>4}x",
        feature.label(),
        attestation.fired,
        width = HARNESS_LABEL_WIDTH
    );
    if attestation.failed_total() > 0 {
        let _ = write!(
            row,
            "  failed {}: {}",
            attestation.failed_total(),
            render_attest_reasons(&attestation.failures_by_frequency())
        );
    }
    if attestation.declined_total() > 0 {
        let _ = write!(
            row,
            "  declined {}: {}",
            attestation.declined_total(),
            render_attest_reasons(&attestation.declines_by_frequency())
        );
    }
    let mut rows = vec![row];
    if health == telemetry::FeatureHealth::Silent {
        rows.push(format!("            {}", feature.precondition()));
    }
    rows
}

/// `"provider_failure x17, timeout x4"` — every reason, most frequent first.
/// Not truncated: the reason set is bounded by the code's own `&'static str`
/// constants, so the line cannot grow without a source change.
fn render_attest_reasons(reasons: &[(&'static str, u64)]) -> String {
    reasons
        .iter()
        .map(|(reason, count)| format!("{reason} x{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Learned complexity calibration: which (role, complexity) classes THIS
/// project's outcome history promotes one band up at routing time, plus the
/// cross-project merged view (the `JacobianLens.merge()` analog — a thin
/// local history can preview what its siblings' evidence would add; only the
/// local table actually routes).
fn smart_doctor_calibration_section(records: &[runtime::RouteOutcomeRecord]) -> String {
    let mut lines = vec![
        String::new(),
        "Complexity calibration (learned)".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let local = runtime::ComplexityCalibration::compute(records);
    let local_rows = local.promoted_classes();
    if local_rows.is_empty() {
        lines.push(
            "  (no promoted classes — every (role, complexity) class is healthy or under-sampled)"
                .to_string(),
        );
    } else {
        for (role, complexity, completed, failed) in &local_rows {
            lines.push(format!(
                "  {role}·{complexity} → one band up  ({failed} failed / {} decisive)",
                completed + failed
            ));
        }
    }
    let merged_records = runtime::read_route_outcomes_across_projects();
    if !merged_records.is_empty() {
        let merged = runtime::ComplexityCalibration::compute(&merged_records);
        let extra: Vec<_> = merged
            .promoted_classes()
            .into_iter()
            .filter(|(role, complexity, ..)| {
                !local_rows.iter().any(|(local_role, local_complexity, ..)| {
                    local_role == role && local_complexity == complexity
                })
            })
            .collect();
        lines.push(format!(
            "  cross-project view: {} records across projects{}",
            merged_records.len(),
            if extra.is_empty() {
                " — no additional promotions".to_string()
            } else {
                String::new()
            }
        ));
        for (role, complexity, completed, failed) in extra {
            lines.push(format!(
                "    would also promote {role}·{complexity}  ({failed} failed / {} decisive; advisory only)",
                completed + failed
            ));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

/// (a) Per-`route_key` count of `signal:"verdict"` records — surfaces
/// whether Phase 4's cross-check channel (deep-gate VERIFY / planner-bound
/// reviewer / repair-loop / ad-hoc review) is actually filling, without
/// grepping the JSONL by hand.
fn smart_doctor_verdict_section(records: &[runtime::RouteOutcomeRecord]) -> String {
    let mut lines = vec![
        String::new(),
        "Verdict channel".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for record in records {
        if record.signal.as_deref() == Some("verdict") {
            *counts.entry(record.route_key.as_str()).or_insert(0) += 1;
        }
    }
    let total: usize = counts.values().sum();
    if total == 0 {
        lines.push(
            "0 verdict-signal records. The cross-check channel (deep-gate VERIFY, planner-bound \
             reviewer→worker, repair-loop, ad-hoc code review) has not attributed a verdict to \
             any route_key in this store yet."
                .to_string(),
        );
    } else {
        lines.push(format!(
            "{total} verdict-signal record(s) across {} route_key bucket(s):",
            counts.len()
        ));
        for (route_key, count) in &counts {
            lines.push(format!("  {route_key}: {count}"));
        }
    }
    lines.join("\n")
}

/// (a′) Per-`(implementation model, verifier model)` pass-rate for
/// `signal:"verdict"` records that carry a cross-model verifier (P1). Surfaces
/// which implementation/verification pairings the deep-gate VERIFY channel has
/// actually judged, and how often the implementation's work passed — the pair
/// attribution that plain verdict counts (the section above) cannot show.
/// `None` when no such paired record exists yet, so the section is omitted
/// rather than rendering an empty table (the routing plan's "0 samples ⇒ no
/// section" convention). Reuses the store's decisive convention (`completed`
/// = pass, `failed` = fail; user-cancelled `stopped`/non-terminal never
/// counts) — verdict records only ever carry completed/failed, but any other
/// status is skipped defensively so a stray record cannot skew the rate.
fn smart_doctor_verify_pair_section(records: &[runtime::RouteOutcomeRecord]) -> Option<String> {
    // (implementation, verifier) -> (passed, decisive). BTreeMap so the table
    // renders in a deterministic, testable order.
    let mut pairs: BTreeMap<(&str, &str), (usize, usize)> = BTreeMap::new();
    for record in records {
        if record.signal.as_deref() != Some("verdict") {
            continue;
        }
        let Some(verifier) = record.verifier_model.as_deref() else {
            continue;
        };
        let passed = match record.status.as_str() {
            "completed" => 1,
            "failed" => 0,
            // Non-decisive (stopped / still_running): excluded from the rate.
            _ => continue,
        };
        let entry = pairs
            .entry((record.selected_model.as_str(), verifier))
            .or_insert((0, 0));
        entry.0 += passed;
        entry.1 += 1;
    }
    if pairs.is_empty() {
        return None;
    }
    let total: usize = pairs.values().map(|(_, decisive)| decisive).sum();
    let mut lines = vec![
        String::new(),
        "Verify pair attribution (P1)".to_string(),
        "────────────────────────────────────".to_string(),
        format!(
            "{total} paired verdict sample(s) across {} (implementation → verifier) pair(s):",
            pairs.len()
        ),
    ];
    for ((implementation, verifier), &(passed, decisive)) in &pairs {
        // `decisive` is always >= 1 (every entry is inserted on a decisive
        // record), so the guard is defensive, not reachable.
        let pct = if decisive == 0 {
            0
        } else {
            passed.saturating_mul(100) / decisive
        };
        lines.push(format!(
            "  {implementation} → {verifier}: {decisive} sample(s), {passed}/{decisive} pass ({pct}%)"
        ));
    }
    Some(lines.join("\n"))
}

/// Model ids whose `routeReason` carries a `learned-shadow-differs:<model>`
/// stamp (Phase 6 shadow-mode probe), read from recent agent manifests — the
/// cheapest available read-back path (`apply.rs::annotate_learned_shadow_
/// delta`'s doc comment names this exact source: the stamp already persists
/// to the manifest via `compose_route_reason`, so no new plumbing is needed).
/// Returns `(agent name-or-id, model the learned hint would have picked)`
/// pairs, newest-manifest-first is NOT guaranteed (directory read order);
/// best-effort — an unreadable store or a malformed manifest is skipped, not
/// an error.
pub(crate) fn scan_learned_shadow_stamps() -> Vec<(String, String)> {
    let Ok(dir) = tools::agent_store_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut hits = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        let Some(reason) = value.get("routeReason").and_then(Value::as_str) else {
            continue;
        };
        let Some(model) = reason.split("learned-shadow-differs:").nth(1) else {
            continue;
        };
        // Defensive: the stamp is documented as the LAST segment
        // `compose_route_reason` appends, but split on the " · " joiner too
        // in case a future guardrail is appended after it.
        let model = model.split(" \u{b7} ").next().unwrap_or(model).trim();
        if model.is_empty() {
            continue;
        }
        let name = value
            .get("agentId")
            .and_then(Value::as_str)
            .or_else(|| value.get("name").and_then(Value::as_str))
            .unwrap_or("agent")
            .to_string();
        hits.push((name, model.to_string()));
    }
    hits
}

/// (b) Learned-vs-seed shadow delta: the manifest-stamp scan above, plus —
/// when cheaply computable from the SAME already-loaded records — a table of
/// every (role, model) the Phase 6 learned-specialty engine currently has an
/// opinion on, with its adjustment/confidence. Computed fresh here regardless
/// of the live `smart.learnedSpecialty` mode (this is a read-only report, not
/// a routing decision), but the mode itself is shown so the reader knows
/// whether these numbers are currently influencing routing (`on`) or only
/// being computed for observation (`off`/`shadow`).
fn smart_doctor_learned_shadow_section(
    records: &[runtime::RouteOutcomeRecord],
    snapshot: Option<&SmartSettingsSnapshot>,
) -> String {
    let mut lines = vec![
        String::new(),
        "Learned specialty (shadow-first)".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let mode_label = snapshot.map_or("unknown (settings unreadable)", |snapshot| {
        snapshot.learned_specialty.status_label()
    });
    lines.push(format!("Mode: {mode_label}"));
    let stamps = scan_learned_shadow_stamps();
    if stamps.is_empty() {
        lines.push(
            "No `learned-shadow-differs:<model>` stamps found in recent agent manifests — shadow \
             mode has not yet found a case where the learned pick would differ from the seed pick."
                .to_string(),
        );
    } else {
        lines.push(format!(
            "{} agent manifest(s) recorded a shadow delta (learned would have picked differently):",
            stamps.len()
        ));
        for (name, model) in stamps.iter().take(12) {
            lines.push(format!("  {name} \u{2192} {model}"));
        }
    }
    let hint = LearnedSpecialtyHint::compute(records, epoch_seconds_now(), |raw| {
        api::resolve_model_alias(raw.trim())
    });
    if hint.is_empty() {
        lines.push(
            "No learned-specialty entries yet (needs >=4 weighted decisive verdict/run samples \
             per role/model pair)."
                .to_string(),
        );
    } else {
        lines.push("Learned entries (role: model — adjustment, confidence):".to_string());
        for (role, model, entry) in hint.entries() {
            // Same predicate the router uses: an armed row earns candidacy
            // one rung up (at tier-score par), subject to the rung's other
            // requirements. Positive-only — a losing streak never arms.
            let armed = if entry.earns_rung_admission() { " — rung-admission armed" } else { "" };
            lines.push(format!(
                "  {}: {model} — {:+}, {}\u{2030}{armed}",
                role_display_label(role.key()),
                entry.model_adjustment,
                entry.confidence_permille
            ));
        }
    }
    lines.join("\n")
}

/// (c) Exploration status: master switch + cadence, and per-`route_key`
/// whether the incumbent has reached `CONFIDENT_DECISIVE_SAMPLES` decisive
/// runs (Phase 5's "established enough to justify exploring rivals" gate —
/// mirrors `policy::select_ranked_auto_candidate`'s eligibility check).
fn smart_doctor_exploration_section(
    snapshot: Option<&SmartSettingsSnapshot>,
    summary: &runtime::RouteOutcomeSummary,
) -> String {
    let mut lines = vec![
        String::new(),
        "Exploration".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let Some(snapshot) = snapshot else {
        lines.push("Settings unreadable — exploration status unknown.".to_string());
        return lines.join("\n");
    };
    lines.push(format!(
        "Exploration: {} \u{b7} cadence every {} recorded outcome(s) per route",
        if snapshot.exploration { "on" } else { "off" },
        snapshot.exploration_cadence
    ));
    let threshold = usize::try_from(CONFIDENT_DECISIVE_SAMPLES).unwrap_or(8);
    let mut route_keys: Vec<&str> = summary
        .by_route
        .iter()
        .map(|bucket| bucket.route_key.as_str())
        .collect();
    route_keys.sort_unstable();
    route_keys.dedup();
    for route_key in route_keys.iter().take(12) {
        let counts = summary.decisive_counts_for_route_key(route_key);
        let Some((incumbent_model, incumbent_decisive)) =
            counts.iter().max_by_key(|(_, decisive)| *decisive)
        else {
            continue;
        };
        let eligible = *incumbent_decisive >= threshold;
        lines.push(format!(
            "  {route_key}: incumbent {incumbent_model} at {incumbent_decisive} decisive — {}",
            if eligible {
                "exploration-eligible (under-sampled rivals may rotate in)"
            } else {
                "not yet eligible"
            }
        ));
    }
    lines.join("\n")
}

/// (d) Canonical merge summary: run the SAME canonicalizer the CLI's own
/// feedback reads use (`api::resolve_model_alias`) over every distinct raw
/// `selectedModel` id in the store, and show which canonical models absorb
/// more than one raw id fragment — the P3 fix's audit trail (raw string
/// keying fragmented `claude-opus-4-8`/`claude-opus-4.8`, `gpt-5.5`/dated
/// ids, etc. into separate buckets before canonicalization-at-read).
fn smart_doctor_canonical_merge_section(records: &[runtime::RouteOutcomeRecord]) -> String {
    let mut lines = vec![
        String::new(),
        "Canonical id merges".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for record in records {
        let raw = record.selected_model.trim();
        if raw.is_empty() {
            continue;
        }
        let canonical = api::resolve_model_alias(raw);
        let raws = groups.entry(canonical).or_default();
        if !raws.iter().any(|existing| existing == raw) {
            raws.push(raw.to_string());
        }
    }
    let merged: Vec<(&String, &Vec<String>)> =
        groups.iter().filter(|(_, raws)| raws.len() > 1).collect();
    if merged.is_empty() {
        lines.push(
            "No raw id fragments merge under canonicalization — every raw `selectedModel` id in \
             the store already maps to a distinct canonical model."
                .to_string(),
        );
    } else {
        lines.push(format!(
            "{} canonical model(s) absorb multiple raw id fragments:",
            merged.len()
        ));
        for (canonical, raws) in merged {
            lines.push(format!("  {canonical} \u{2190} {}", raws.join(", ")));
        }
    }
    lines.join("\n")
}

fn provenance_label(provenance: runtime::TiersProvenance) -> &'static str {
    match provenance {
        runtime::TiersProvenance::Fallback => {
            "fallback (no capability data — marketing-token/generic guess)"
        }
        runtime::TiersProvenance::ColdStartPrior => {
            "cold-start-prior (capability-derived, static)"
        }
        runtime::TiersProvenance::ProviderDeclared => {
            "provider-declared (provider's own stated lineup position)"
        }
        runtime::TiersProvenance::Learned => "learned (from outcome data)",
    }
}

/// (e) Per-model tier provenance (`ModelDescriptor::tiers_provenance_value`,
/// the P1 field) for the CURRENT connected inventory — lets the reader see
/// which models' Deep/Strong/etc. tier grants are a real provider-declared
/// capability fact (`ColdStartPrior`), a weak marketing-token guess
/// (`Fallback`), or (once Phase 6 wires a demotion path) `Learned`.
fn smart_doctor_provenance_section(default_model: &str) -> String {
    let mut lines = vec![
        String::new(),
        "Model tier provenance".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let inventory = connected_model_inventory(default_model);
    for model in inventory.models() {
        lines.push(format!(
            "  {} \u{2014} {}",
            model.id(),
            provenance_label(model.tiers_provenance_value())
        ));
    }
    lines.join("\n")
}

/// (f) Pin-contamination warning: the existing config-override warning
/// (saved role/subagent pins take priority over AUTO) extended with a
/// STORE-side check — when the majority of a `route_key`'s outcome records
/// were recorded under a pin (`routeSource == "pin"`), that bucket's history
/// measures pin AVAILABILITY, not AUTO selection quality (the exact live-data
/// finding that motivated Phase 6's pin zero-weighting).
fn smart_doctor_pin_awareness_section(records: &[runtime::RouteOutcomeRecord]) -> String {
    let mut lines = vec![
        String::new(),
        "Pin awareness".to_string(),
        "────────────────────────────────────".to_string(),
    ];
    let mut any = false;
    if let Some(warning) = override_bypass_warning() {
        lines.push(warning);
        any = true;
    }
    let mut per_route: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for record in records {
        let entry = per_route.entry(record.route_key.as_str()).or_insert((0, 0));
        entry.1 += 1;
        if record.route_source.as_deref() == Some("pin") {
            entry.0 += 1;
        }
    }
    let dominated: Vec<(&str, usize, usize)> = per_route
        .into_iter()
        .filter(|(_, (pin, total))| *total > 0 && *pin * 2 > *total)
        .map(|(key, (pin, total))| (key, pin, total))
        .collect();
    if !dominated.is_empty() {
        any = true;
        lines.push(
            "! The following route_key buckets are DOMINATED by pinned routes — their history \
             measures pin AVAILABILITY, not AUTO selection quality:"
                .to_string(),
        );
        for (route_key, pin, total) in dominated {
            // Outcome counts are bounded by the store's global retention cap
            // (2048) — well within f64's exact-integer range, so widen via
            // u32 first rather than allow-suppressing the precision lint.
            let pin_f = f64::from(u32::try_from(pin).unwrap_or(u32::MAX));
            let total_f = f64::from(u32::try_from(total).unwrap_or(u32::MAX));
            let pct = (pin_f / total_f) * 100.0;
            lines.push(format!("  {route_key}: {pin}/{total} records ({pct:.0}%) are pinned"));
        }
    }
    if !any {
        lines.push("No saved overrides and no pin-dominated route_key buckets.".to_string());
    }
    lines.join("\n")
}

fn render_smart_doctor_summary(
    path: &Path,
    summary: &runtime::RouteOutcomeSummary,
) -> String {
    let mut lines = smart_doctor_header_lines(path, summary);
    if summary.total == 0 {
        append_smart_doctor_empty_state(&mut lines);
        return lines.join("\n");
    }
    append_smart_doctor_buckets(&mut lines, summary);
    append_smart_doctor_note(&mut lines);
    lines.join("\n")
}

fn smart_doctor_header_lines(path: &Path, summary: &runtime::RouteOutcomeSummary) -> Vec<String> {
    vec![
        format!("{} Smart Router Doctor", glyphs::SMART_AUTO_NC),
        format!("Outcome log: {}", path.display()),
        format!("Recorded outcomes: {}", summary.total),
        format!(
            "Status: completed {} / failed {} / stopped {} / still-running {}",
            summary.completed, summary.failed, summary.stopped, summary.still_running
        ),
        format!("Output tokens: {}", summary.output_tokens),
    ]
}

fn append_smart_doctor_empty_state(lines: &mut Vec<String>) {
    lines.push("No route outcomes recorded yet. Run agent/workflow tasks with Smart Router enabled to build aggregate feedback.".to_string());
}

fn append_smart_doctor_buckets(lines: &mut Vec<String>, summary: &runtime::RouteOutcomeSummary) {
    lines.push(String::new());
    lines.push("Route outcome buckets".to_string());
    lines.push("────────────────────────────────────".to_string());
    lines.extend(summary.by_route.iter().take(12).map(smart_doctor_bucket_line));
}

fn smart_doctor_bucket_line(bucket: &runtime::RouteOutcomeBucket) -> String {
    format!(
        "{} via {} — total {}, completed {}, failed {}, stopped {}, tokens {}, provider errors {}",
        bucket.route_key,
        bucket.selected_model,
        bucket.total,
        bucket.completed,
        bucket.failed,
        bucket.stopped,
        bucket.output_tokens,
        smart_doctor_provider_errors(bucket)
    )
}

fn smart_doctor_provider_errors(bucket: &runtime::RouteOutcomeBucket) -> String {
    if bucket.provider_errors.is_empty() {
        return "none".to_string();
    }
    bucket
        .provider_errors
        .iter()
        .map(|(class, count)| format!("{class}:{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn append_smart_doctor_note(lines: &mut Vec<String>) {
    lines.push(String::new());
    // Honest, state-independent note. The old text claimed feedback-informed auto
    // was "enabled by a later phase" — it ships and is wired into the live scorer;
    // it is a /smart toggle. (Kept free of runtime-global reads so the rendered
    // doctor output stays deterministic.)
    lines.push("Note: when feedback-informed auto is enabled in /smart, these bounded outcomes nudge subagent auto routing; explicit pins and family locks are never overridden.".to_string());
}

/// The flat `key: value` header of `/smart status` — one row per resolved
/// `smart.*` setting. Split out of [`render_smart_status`] so adding a setting
/// row never pushes the renderer over the line budget.
fn smart_status_setting_rows(
    snapshot: &SmartSettingsSnapshot,
    inventory: &runtime::ModelInventory,
    enabled: bool,
    override_count: usize,
) -> Vec<String> {
    vec![
        format!(
            "{} Smart Model Router: {}",
            glyphs::SMART_AUTO_NC,
            if enabled { "ON" } else { "OFF" }
        ),
        format!("Settings path: {}", global_settings_path().display()),
        format!("Usable models: {}", inventory.models().len()),
        format!("Main model: {}", inventory.main_model()),
        format!("Overrides: {override_count}"),
        format!(
            "Policy: {}",
            match snapshot.policy {
                runtime::SmartPolicy::Architect =>
                    "architect (default) — deep models plan/verify, implementers code",
                runtime::SmartPolicy::Classic => "classic (pre-contract routing)",
            }
        ),
        format!(
            "EXEC swap: {}{}",
            match snapshot.exec_swap {
                tools::SmartExecSwap::Always =>
                    "always — implementer writes every diff, planner plans/verifies",
                tools::SmartExecSwap::Easy => "easy — only the lowest complexity band hands off",
                tools::SmartExecSwap::Never => "never — EXEC stays on the session model",
            },
            // Derived from the enum's own Default so this line can never claim
            // a stale default again (it shipped saying "always (default)" for
            // a mode whose real default had become Never).
            if snapshot.exec_swap == tools::SmartExecSwap::default() {
                " (default)"
            } else {
                ""
            }
        ),
        format!(
            "Cross-provider diversity: {}",
            if snapshot.allow_cross_provider_diversity { "allowed" } else { "off" }
        ),
        format!(
            "Verify cross-provider: {}",
            if snapshot.verify_cross_provider { "on (default)" } else { "off (native-preferred)" }
        ),
        format!(
            "Quota fallback: {}",
            if snapshot.quota_fallback { "on (default)" } else { "off (turn fails on quota exhaustion)" }
        ),
        format!(
            "Feedback-informed auto: {}",
            if snapshot.feedback_informed_auto { "on (bounded)" } else { "off" }
        ),
        format!(
            "Provider allowlist: {}",
            if snapshot.provider_allowlist.is_empty() {
                "all connected providers".to_string()
            } else {
                snapshot.provider_allowlist.join(", ")
            }
        ),
        format!("Auto classifier: {}", snapshot.auto_classifier.status_label()),
        format!("Saved subagent overrides: {}", snapshot.subagents.len()),
        format!("Saved role fallbacks: {}", snapshot.roles.len()),
    ]
}

pub(crate) fn render_smart_status(
    default_model: &str,
    forced_enabled: Option<bool>,
) -> io::Result<String> {
    let snapshot = read_global_smart_settings()?;
    let enabled = forced_enabled.unwrap_or(snapshot.enabled);
    let inventory = connected_model_inventory(default_model);
    let learned = dashboard_learned_specialty_hint(&snapshot);
    let plan = recommend_auto_assignments_with_learned_specialty(
        &inventory,
        &assignment_options(&snapshot),
        None,
        &learned,
    );
    let override_count = snapshot.subagents.len() + snapshot.roles.len();
    let mut lines = smart_status_setting_rows(&snapshot, &inventory, enabled, override_count);
    if enabled && override_count > 0 {
        lines.push(format!(
            "! {override_count} override(s) take priority over Auto for those targets — run `/smart reset` for full-auto routing."
        ));
    }
    if !enabled {
        lines.push(
            "Execution is unchanged while Smart Router is OFF; preview shows what would apply if enabled."
                .to_string(),
        );
    }
    lines.push(String::new());
    lines.push("Usable Models".to_string());
    lines.push("────────────────────────────────────".to_string());
    lines.extend(render_model_pool_rows(&inventory));
    if inventory.models().len() <= 1 {
        lines.push("! Single-model pool: cross-model verification is unavailable.".to_string());
    }
    lines.push(String::new());
    lines.push("Recommended Subagent Setup".to_string());
    lines.push("────────────────────────────────────".to_string());
    lines.extend(render_subagent_assignments(&plan));
    lines.extend(smart_route_legend());
    lines.push(String::new());
    lines.push("Agent Types".to_string());
    lines.push("────────────────────────────────────".to_string());
    lines.push("  Unspecified spawns are auto-classified before routing.".to_string());
    lines.extend(render_agent_type_rows(&plan, false));
    lines.push(String::new());
    lines.push("Saved Overrides".to_string());
    lines.push("────────────────────────────────────".to_string());
    lines.extend(render_saved_overrides(&snapshot, &inventory));
    lines.push(String::new());
    lines.push("Execution priority: explicit model > subagent override > role fallback > auto recommendation > main model.".to_string());
    lines.push("Run `/smart` to edit settings in the GUI.".to_string());
    Ok(lines.join("\n"))
}

fn render_smart_agents(default_model: &str) -> io::Result<String> {
    let snapshot = read_global_smart_settings()?;
    let inventory = connected_model_inventory(default_model);
    let learned = dashboard_learned_specialty_hint(&snapshot);
    let plan = recommend_auto_assignments_with_learned_specialty(
        &inventory,
        &assignment_options(&snapshot),
        None,
        &learned,
    );
    let mut lines = vec![
        "Agent Types".to_string(),
        "────────────────────────────────────".to_string(),
        "Unspecified spawns are auto-classified before routing.".to_string(),
        String::new(),
        "Builtin Agents".to_string(),
    ];
    lines.extend(render_agent_type_rows(&plan, true));
    lines.push(String::new());
    lines.push("Custom Agents".to_string());
    let custom_agents = tools::loaded_custom_agents();
    if custom_agents.is_empty() {
        lines.push("  <none loaded>".to_string());
    } else {
        for agent in custom_agents {
            let model = agent.model.as_deref().unwrap_or("inherit");
            let permission = agent
                .permission_mode
                .map_or("danger-full-access (default)", runtime::PermissionMode::as_str);
            lines.push(format!(
                "  {} model={model} permission={permission} source={}",
                agent.name,
                agent.source_path.display(),
            ));
        }
    }
    Ok(lines.join("\n"))
}

fn render_model_pool_rows(inventory: &runtime::ModelInventory) -> Vec<String> {
    if inventory.models().is_empty() {
        return vec!["  <none>".to_string()];
    }
    inventory
        .models()
        .iter()
        .map(|model| {
            format!(
                "  {} {:<28} {:<10} {}/{}",
                glyphs::SMART_MODEL_NC,
                model.id(),
                model.provider(),
                model.family(),
                model.class_label().unwrap_or("balanced")
            )
        })
        .collect()
}

fn render_subagent_assignments(plan: &runtime::AutoAssignmentPlan) -> Vec<String> {
    if plan.assignments.is_empty() {
        return vec!["  <none>".to_string()];
    }
    let mut rows = Vec::new();
    for assignment in &plan.assignments {
        let icon = route_icon_for_target(&assignment.target);
        let target = target_label(&assignment.target);
        let source = match assignment.source {
            AssignmentSource::Auto => "Auto",
            AssignmentSource::MainFallback => "Main",
        };
        let confidence = confidence_meter(assignment.confidence);
        rows.push(format!(
            "  {icon} {target:<18} -> {:<28} {source:<4} {confidence} {}",
            assignment.selected_model, assignment.reason
        ));
        if let Some(audit) = assignment.audit.first() {
            rows.push(format!("      audit: {audit}"));
        }
    }
    rows
}

fn render_agent_type_rows(
    plan: &runtime::AutoAssignmentPlan,
    include_purpose: bool,
) -> Vec<String> {
    runtime::BuiltinSubagentProfile::all()
        .iter()
        .copied()
        .map(|profile| {
            let selected_model = plan
                .assignments
                .iter()
                .find(|assignment| {
                    matches!(
                        &assignment.target,
                        RoutingTarget::Subagent(target) if target.key() == profile.key()
                    )
                })
                .map_or("<unavailable>", |assignment| assignment.selected_model.as_str());
            let row = format!(
                "  {:<18} role={:<11} model={:<28} tools={}",
                profile.key(),
                profile.route_role().key(),
                selected_model,
                tools::subagent_toolset_class(profile.key()),
            );
            if include_purpose {
                format!("{row} — {}", profile.purpose())
            } else {
                row
            }
        })
        .collect()
}

fn confidence_meter(confidence: AssignmentConfidence) -> &'static str {
    match confidence {
        AssignmentConfidence::High => "[###]",
        AssignmentConfidence::Medium => "[##-]",
        AssignmentConfidence::Low => "[#--]",
    }
}

fn smart_route_legend() -> Vec<String> {
    vec![
        format!(
            "  Legend: {} fast | {} code | {} verify | {} review",
            glyphs::SMART_FAST_NC,
            glyphs::SMART_CODE_NC,
            glyphs::SMART_VERIFY_NC,
            glyphs::SMART_REVIEW_NC
        ),
        format!(
            "          {} research | {} design | {} auto | [###]/[##-]/[#--] fit",
            glyphs::SMART_RESEARCH_NC,
            glyphs::SMART_DESIGN_NC,
            glyphs::SMART_AUTO_NC
        ),
    ]
}

fn route_icon_for_target(target: &RoutingTarget) -> &'static str {
    match target.route_role_hint() {
        Some(RouteRole::Fast) => glyphs::SMART_FAST_NC,
        Some(RouteRole::Coding | RouteRole::Debugging) => glyphs::SMART_CODE_NC,
        Some(RouteRole::Verifier) => glyphs::SMART_VERIFY_NC,
        Some(RouteRole::Reviewer) => glyphs::SMART_REVIEW_NC,
        Some(
            RouteRole::Analysis
            | RouteRole::Research
            | RouteRole::Judge
            | RouteRole::Synthesizer,
        ) => glyphs::SMART_RESEARCH_NC,
        Some(RouteRole::Writing | RouteRole::Design) => glyphs::SMART_DESIGN_NC,
        Some(RouteRole::Default) | None => glyphs::SMART_AUTO_NC,
    }
}

fn render_saved_overrides(
    snapshot: &SmartSettingsSnapshot,
    inventory: &runtime::ModelInventory,
) -> Vec<String> {
    let mut rows = Vec::new();
    for (subagent, update) in &snapshot.subagents {
        rows.push(format!(
            "  subagent:{subagent:<18} {}",
            describe_update_for_status(update, inventory)
        ));
    }
    for (role, update) in &snapshot.roles {
        rows.push(format!(
            "  role:{role:<22} {}",
            describe_update_for_status(update, inventory)
        ));
    }
    if rows.is_empty() {
        rows.push("  <none>".to_string());
    }
    rows
}

fn describe_update_for_status(update: &SmartRoleUpdate, inventory: &runtime::ModelInventory) -> String {
    match update {
        SmartRoleUpdate::Auto => "Auto".to_string(),
        SmartRoleUpdate::ExactPin { model } => {
            if inventory.find(model).is_some() {
                format!("{} Pinned: {model}", glyphs::SMART_PIN_NC)
            } else {
                format!(
                    "{} Missing pin: {model} (will fallback)",
                    glyphs::SMART_FALLBACK_NC
                )
            }
        }
        SmartRoleUpdate::FamilyLock {
            provider,
            family,
            class,
            freshness,
        } => format!(
            "{} Track: {provider}/{family}/{class}/{}",
            glyphs::SMART_PIN_NC,
            freshness_label(*freshness)
        ),
    }
}

fn target_label(target: &RoutingTarget) -> String {
    match target {
        RoutingTarget::Subagent(profile) => profile.key().to_string(),
        RoutingTarget::RoleFallback(_) => "role-fallback".to_string(),
        RoutingTarget::Foreground => "foreground".to_string(),
        RoutingTarget::WorkflowPhase { phase_id, .. } => format!("workflow:{phase_id}"),
        RoutingTarget::Synthesis => "synthesis".to_string(),
        RoutingTarget::Judge => "judge".to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmartFreshness {
    Latest,
    LatestStable,
}

/// CLI-local mirror of the tools crate's `smart_router::settings::
/// LearnedSpecialtyMode` (private to that crate, so it cannot be imported
/// directly — same reason [`SmartFreshness`] mirrors `FreshnessPolicy`
/// instead of re-exporting a tools-crate type). Keep the three variants and
/// the `shadow` default in lockstep by hand; the
/// `cli_snapshot_defaults_match_tools_crate_runtime_defaults` test guards
/// against the two drifting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SmartLearnedSpecialtyMode {
    Off,
    #[default]
    Shadow,
    On,
}

impl SmartLearnedSpecialtyMode {
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

    fn as_settings_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::On => "on",
        }
    }

    pub(crate) fn status_label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow (soaking — seed still routes; delta logged for doctor)",
            Self::On => "on (live)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SmartRoleUpdate {
    Auto,
    ExactPin {
        model: String,
    },
    FamilyLock {
        provider: String,
        family: String,
        class: String,
        freshness: SmartFreshness,
    },
}

fn global_settings_paths() -> Vec<PathBuf> {
    #[cfg(test)]
    if let Some(path) = SMART_TEST_CONFIG_HOME.with(|home| home.borrow().clone()) {
        return vec![path.join("settings.json")];
    }
    let mut roots = runtime::zo_global_config_roots();
    if roots.is_empty() {
        roots.push(default_config_home());
    }
    roots
        .into_iter()
        .map(|root| root.join("settings.json"))
        .collect()
}

pub(crate) fn global_settings_path() -> PathBuf {
    global_settings_paths()
        .into_iter()
        .next()
        .unwrap_or_else(|| default_config_home().join("settings.json"))
}

pub(crate) fn read_global_smart_settings() -> io::Result<SmartSettingsSnapshot> {
    let root = read_merged_global_settings_object(&global_settings_paths())?;
    Ok(snapshot_from_root(&Value::Object(root)))
}

/// The merged user-global settings object (`~/.zo/settings.json`, honoring
/// `ZO_CONFIG_HOME` and every root `runtime::zo_global_config_roots` reports).
///
/// Exposed so `session_preferences` can read `model` / `reasoningEffort` /
/// `thinkingBudget` from the SAME place `smart.*` is already read from. Without
/// this layer a user-global `reasoningEffort` was silently inert — preferences
/// were resolved from the cwd's `.zo/settings*.json` and the CLI settings file
/// only, so every directory without its own project settings fell through to
/// the `DEFAULT_EFFORT` startup constant no matter what the user had configured
/// globally. A read failure degrades to `None` (the preference layer must never
/// fail a run).
pub(crate) fn global_preferences_object() -> Option<JsonMap<String, Value>> {
    read_merged_global_settings_object(&global_settings_paths()).ok()
}

/// Marker recording that the one-time smart-AUTO default banner was shown.
/// Lives under the config home (`~/.zo` unless `ZO_CONFIG_HOME`
/// redirects it — same `default_config_home()` root as `settings.json`), so
/// the banner is per-user rather than per-session, and tests are isolated the
/// same way `global_settings_path` is.
fn smart_default_banner_marker_path() -> PathBuf {
    smart_config_home().join("notices").join("smart-default-banner-shown")
}

/// The config home every banner read/write resolves against — the test
/// override and `default_config_home()` in one place, so the marker and the
/// prior-use scan can never disagree about whose home they are looking at.
fn smart_config_home() -> PathBuf {
    #[cfg(test)]
    if let Some(path) = SMART_TEST_CONFIG_HOME.with(|home| home.borrow().clone()) {
        return path;
    }
    default_config_home()
}

/// One-time boot notice for the smart-AUTO default flip (`smart.enabled` now
/// defaults to `true`).
///
/// Returns `Some(text)` at most once per user — a marker file under the
/// config home records the showing — and ONLY on the default path: an
/// explicit `smart.enabled` key in `settings.json` (`true` OR `false`) means
/// the user already decided, so there is nothing to announce and the marker
/// is left unburned. Unreadable settings also stay quiet. The injection
/// point is gated by the caller: only the interactive REPL/TUI boot pushes
/// this notice (headless `-p`, `serve`, and spawned sub-agents never enter
/// that path).
/// Whether any managed project under `root` holds at least one session file —
/// the cheapest durable "this user has used zo before" signal. Bounded walk:
/// first hit wins, and a fresh install short-circuits on the missing
/// `projects/` directory.
fn any_prior_session_under(root: &Path) -> bool {
    let Ok(projects) = fs::read_dir(root.join("projects")) else {
        return false;
    };
    for project in projects.flatten() {
        let Ok(sessions) = fs::read_dir(project.path().join("sessions")) else {
            continue;
        };
        for entry in sessions.flatten() {
            if entry.path().extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                return true;
            }
        }
    }
    false
}

pub(crate) fn smart_default_banner_notice() -> Option<String> {
    let root = read_merged_global_settings_object(&global_settings_paths()).ok()?;
    // Match the readers' semantics (`as_bool`): only a *boolean* value counts
    // as a decision. A non-bool value (e.g. a hand-edited `"enabled": "false"`)
    // is treated as absent by both routing readers — the default kicks in — so
    // suppressing the banner on mere key presence would hide the notice from
    // exactly the user whose behavior silently changed.
    let explicit = root
        .get("smart")
        .and_then(Value::as_object)
        .and_then(|smart| smart.get("enabled"))
        .and_then(Value::as_bool)
        .is_some();
    if explicit {
        return None;
    }
    let marker = smart_default_banner_marker_path();
    if marker.exists() {
        return None;
    }
    // Best-effort marker write: an unwritable config home must not error the
    // boot path — worst case the banner reappears on the next boot.
    if let Some(parent) = marker.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&marker, b"shown\n");
    // The banner announces a CHANGED default — it only means something to a
    // user whose earlier sessions ran under the old one. A first boot has no
    // prior expectation to correct, and the notice was the very first thing
    // a brand-new user saw (100% firing, and it demoted the fresh launchpad
    // into the transcript layout). The marker above is still written, so the
    // notice never fires later either: their default was AUTO from day one.
    if !any_prior_session_under(&smart_config_home()) {
        return None;
    }
    Some(
        "smart AUTO routing is now enabled by default: subagent turns route to the best \
         connected model per role. Run /smart off to disable it, or /smart doctor to \
         inspect routing status."
            .to_string(),
    )
}

// A flat, one-key-per-block `settings.json` reader: it grows by one uniform
// `smart.<key>` block per feature gate (lockstep with the tools crate's reader),
// so the length is inherent, not a cohesion smell.
#[allow(clippy::too_many_lines)]
fn snapshot_from_root(root: &Value) -> SmartSettingsSnapshot {
    let smart = root.get("smart").and_then(Value::as_object);
    let enabled = smart
        .and_then(|smart| smart.get("enabled"))
        .and_then(Value::as_bool)
        // On by default — keep in lockstep with the runtime settings default
        // (`read_smart_runtime_settings` in the tools crate) so the dashboard
        // reflects the same smart-AUTO state the router actually uses. An
        // explicit `smart.enabled: false` (or `/smart off`) still wins.
        .unwrap_or(true);
    let allow_cross_provider_diversity = smart
        .and_then(|smart| smart.get("allowCrossProviderDiversity"))
        .and_then(Value::as_bool)
        // On by default — keep in lockstep with the runtime settings default
        // (`read_smart_runtime_settings`) so the dashboard reflects the same
        // cross-provider behavior the router actually uses.
        .unwrap_or(true);
    let verify_cross_provider = smart
        .and_then(|smart| smart.get("verifyCrossProvider"))
        .and_then(Value::as_bool)
        // On by default — keep in lockstep with the tools crate's
        // `read_smart_runtime_settings`. Governs the deep-gate VERIFY leg's
        // cross-provider routing on its own, decoupled from the global
        // worker-diversity flag above.
        .unwrap_or(true);
    let quota_fallback = smart
        .and_then(|smart| smart.get("quotaFallback"))
        .and_then(Value::as_bool)
        // On by default — keep in lockstep with the tools crate's
        // `read_smart_runtime_settings`. Governs the automatic cross-provider
        // fallback when the main model's quota window is exhausted.
        .unwrap_or(true);
    let quota_wait_band_minutes = smart
        .and_then(|smart| smart.get("quotaWaitBandMinutes"))
        .and_then(Value::as_u64)
        // 15 minutes by default — keep in lockstep with the tools crate's
        // `read_smart_runtime_settings`. When a quota window lifts within this
        // many minutes the turn HOLDS on the main model instead of falling
        // back; `0` disables the band (pure fallback).
        .unwrap_or(DEFAULT_QUOTA_WAIT_BAND_MINUTES);
    let provider_allowlist = smart
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
        // Empty by default (all providers allowed) — keep in lockstep with the
        // tools crate's `read_smart_runtime_settings`.
        .unwrap_or_default();
    let deep_tier_models = smart
        .and_then(|smart| smart.get("deepTierModels"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|model| !model.is_empty())
                // Catalog-first, in lockstep with the tools crate's
                // `deep_tier_models_from_smart`: the gated resolver would leave
                // `openai-latest` unresolved on a machine without OpenAI
                // connected, so this reader and the live router would disagree
                // about what the same configured pool names.
                .map(api::resolve_catalog_alias)
                .collect()
        })
        .filter(|models: &Vec<String>| !models.is_empty())
        .unwrap_or_else(runtime::default_deep_tier_models);
    let feedback_informed_auto = smart
        .and_then(|smart| smart.get("feedbackInformedAuto"))
        .and_then(Value::as_bool)
        // On by default — keep in lockstep with the runtime settings default so the
        // dashboard reflects the same dynamic-routing state the router actually uses.
        .unwrap_or(true);
    let auto_classifier = RouteAutoClassifierMode::from_settings_value(
        smart.and_then(|smart| smart.get("autoClassifier")),
    );
    let fallback_candidate_limit = smart
        .and_then(|smart| smart.get("fallbackCandidateLimit"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        // Keep in lockstep with the tools crate's `DEFAULT_FALLBACK_CANDIDATE_LIMIT`.
        .unwrap_or(DEFAULT_FALLBACK_CANDIDATE_LIMIT);
    let exploration = smart
        .and_then(|smart| smart.get("exploration"))
        .and_then(Value::as_bool)
        // On by default — keep in lockstep with the tools crate's
        // `read_smart_runtime_settings` (Phase 5).
        .unwrap_or(true);
    let exploration_cadence = smart
        .and_then(|smart| smart.get("explorationCadence"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        // Keep in lockstep with the tools crate's `DEFAULT_EXPLORATION_CADENCE`.
        .unwrap_or(DEFAULT_EXPLORATION_CADENCE);
    let learned_specialty = SmartLearnedSpecialtyMode::from_settings_value(
        smart.and_then(|smart| smart.get("learnedSpecialty")),
    );
    let headroom_penalty_threshold = smart
        .and_then(|smart| smart.get("headroomPenaltyThreshold"))
        .and_then(Value::as_u64)
        .map(|value| value.clamp(1, 100))
        .and_then(|value| u8::try_from(value).ok())
        // Keep in lockstep with the tools crate's `DEFAULT_HEADROOM_PENALTY_THRESHOLD`.
        .unwrap_or(DEFAULT_HEADROOM_PENALTY_THRESHOLD);
    // Absent ⇒ Architect (the live default); explicit "classic" or
    // ZO_SMART_POLICY opts out — keep in lockstep with the tools crate's
    // `read_smart_runtime_settings` (both delegate to the shared runtime parser).
    let policy = runtime::SmartPolicy::from_settings_value(smart.and_then(|smart| smart.get("policy")));
    let exec_swap =
        tools::SmartExecSwap::from_settings_value(smart.and_then(|smart| smart.get("execSwap")));
    let mut subagents = BTreeMap::new();
    if let Some(subagent_object) = root
        .get("modelRouter")
        .and_then(Value::as_object)
        .and_then(|router| router.get("subagents"))
        .and_then(Value::as_object)
    {
        for (subagent, value) in subagent_object {
            if let Some(profile) = SubagentProfileId::parse(subagent) {
                if let Some(update) = role_update_from_json(value) {
                    subagents.insert(profile.key().to_string(), update);
                }
            }
        }
    }
    let mut roles = BTreeMap::new();
    if let Some(role_object) = root
        .get("modelRouter")
        .and_then(Value::as_object)
        .and_then(|router| router.get("roles"))
        .and_then(Value::as_object)
    {
        for (role, value) in role_object {
            if let Some(role) = normalize_role(role) {
                if let Some(update) = role_update_from_json(value) {
                    roles.insert(role.to_string(), update);
                }
            }
        }
    }
    SmartSettingsSnapshot {
        enabled,
        allow_cross_provider_diversity,
        verify_cross_provider,
        quota_fallback,
        quota_wait_band_minutes,
        provider_allowlist,
        deep_tier_models,
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
    }
}

/// `mode` value marking a deletion tombstone in `modelRouter.{roles,subagents}`.
/// A tombstone written into the PRIMARY root durably masks a same-key override
/// that lives only in a lower canonical root: the deep merge keeps the primary
/// object (so `mode` reads `"deleted"`), and every reader treats that as "no
/// override" — the same as an absent key — restoring Auto. A later explicit
/// pin/family write to the same key overwrites the tombstone, so masking is not
/// sticky. Writes stay primary-only; lower roots are never rewritten.
const DELETED_MODE: &str = "deleted";

/// A deletion tombstone object (`{"mode":"deleted"}`) for a role/subagent key.
fn deleted_role_json() -> JsonMap<String, Value> {
    let mut config = JsonMap::new();
    config.insert("mode".to_string(), Value::String(DELETED_MODE.to_string()));
    config
}

fn role_update_from_json(value: &Value) -> Option<SmartRoleUpdate> {
    let object = value.as_object()?;
    match object.get("mode")?.as_str()? {
        // A tombstone masks any lower-root override for this key: readers see
        // "no override" exactly as if the key were absent.
        DELETED_MODE => None,
        "pinned" => Some(SmartRoleUpdate::ExactPin {
            model: object.get("model")?.as_str()?.to_string(),
        }),
        "manualPreferred" => {
            let selector = object.get("selector")?.as_object()?;
            Some(SmartRoleUpdate::FamilyLock {
                provider: selector.get("provider")?.as_str()?.to_string(),
                family: selector.get("family")?.as_str()?.to_string(),
                class: selector.get("class")?.as_str()?.to_string(),
                freshness: match selector.get("freshness")?.as_str()? {
                    "latest" => SmartFreshness::Latest,
                    _ => SmartFreshness::LatestStable,
                },
            })
        }
        "auto" => Some(SmartRoleUpdate::Auto),
        _ => None,
    }
}

pub(crate) fn write_global_reset_overrides() -> io::Result<()> {
    update_global_settings(|root| {
        let model_router = object_child(root, "modelRouter")?;
        // The passed-in `root` is the MERGED view across every canonical root,
        // so `roles`/`subagents` here also carry overrides that live only in
        // lower roots. A bare `remove` would drop them from the primary write
        // but leave the lower-root files untouched, so the next merged read
        // resurrects them. Replace each live key with a deletion tombstone
        // instead: the tombstone is written to the PRIMARY root and durably
        // masks the lower-root override on every subsequent read. Writes stay
        // primary-only.
        tombstone_all_overrides(object_child(model_router, "roles")?);
        tombstone_all_overrides(object_child(model_router, "subagents")?);
        Ok(())
    })
}

/// Replace every non-tombstone override in a merged `roles`/`subagents` object
/// with a deletion tombstone, so a reset durably masks overrides sourced from
/// lower canonical roots. Keys already tombstoned stay tombstoned.
fn tombstone_all_overrides(overrides: &mut JsonMap<String, Value>) {
    let keys = overrides.keys().cloned().collect::<Vec<_>>();
    for key in keys {
        overrides.insert(key, Value::Object(deleted_role_json()));
    }
}

fn write_global_smart_bool(key: &'static str, enabled: bool) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        smart.insert(key.to_string(), Value::Bool(enabled));
        Ok(())
    })
}

pub(crate) fn write_global_smart_enabled(enabled: bool) -> io::Result<()> {
    write_global_smart_bool("enabled", enabled)
}

/// Persist `smart.policy` (`"architect"` / `"classic"`) to the global settings.
pub(crate) fn write_global_smart_policy(policy: runtime::SmartPolicy) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        smart.insert("policy".to_string(), Value::String(policy.key().to_string()));
        Ok(())
    })
}

pub(crate) fn write_global_smart_exec_swap(exec_swap: tools::SmartExecSwap) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        smart.insert(
            "execSwap".to_string(),
            Value::String(exec_swap.key().to_string()),
        );
        Ok(())
    })
}

pub(crate) fn write_global_smart_allow_cross_provider_diversity(enabled: bool) -> io::Result<()> {
    write_global_smart_bool("allowCrossProviderDiversity", enabled)
}

pub(crate) fn write_global_smart_verify_cross_provider(enabled: bool) -> io::Result<()> {
    write_global_smart_bool("verifyCrossProvider", enabled)
}

pub(crate) fn write_global_smart_quota_fallback(enabled: bool) -> io::Result<()> {
    write_global_smart_bool("quotaFallback", enabled)
}

pub(crate) fn write_global_smart_feedback_informed_auto(enabled: bool) -> io::Result<()> {
    write_global_smart_bool("feedbackInformedAuto", enabled)
}

pub(crate) fn write_global_smart_exploration(enabled: bool) -> io::Result<()> {
    write_global_smart_bool("exploration", enabled)
}

pub(crate) fn write_global_smart_exploration_cadence(cadence: usize) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        smart.insert(
            "explorationCadence".to_string(),
            Value::Number(serde_json::Number::from(cadence)),
        );
        Ok(())
    })
}

pub(crate) fn write_global_smart_auto_classifier(mode: &str) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        smart.insert("autoClassifier".to_string(), Value::String(mode.to_string()));
        Ok(())
    })
}

pub(crate) fn write_global_smart_learned_specialty(
    mode: SmartLearnedSpecialtyMode,
) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        smart.insert(
            "learnedSpecialty".to_string(),
            Value::String(mode.as_settings_str().to_string()),
        );
        Ok(())
    })
}

pub(crate) fn write_global_smart_provider_allowlist(providers: &[String]) -> io::Result<()> {
    update_global_settings(|root| {
        let smart = object_child(root, "smart")?;
        if providers.is_empty() {
            smart.remove("providerAllowlist");
        } else {
            smart.insert(
                "providerAllowlist".to_string(),
                Value::Array(providers.iter().cloned().map(Value::String).collect()),
            );
        }
        Ok(())
    })
}

pub(crate) fn write_global_smart_role(role: &str, update: &SmartRoleUpdate) -> io::Result<()> {
    let key = normalize_role(role).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported smart role `{role}`"),
        )
    })?;
    update_global_settings(|root| {
        let model_router = object_child(root, "modelRouter")?;
        let roles = object_child(model_router, "roles")?;
        match update {
            SmartRoleUpdate::Auto => {
                // Tombstone rather than remove: a bare remove drops this role
                // from the primary write but leaves a lower-root override in
                // place, which the next merged read resurrects. The tombstone
                // durably masks it back to Auto. A later explicit pin/family
                // write to the same key overwrites the tombstone below.
                roles.insert(key.to_string(), Value::Object(deleted_role_json()));
            }
            SmartRoleUpdate::ExactPin { model } => {
                roles.insert(key.to_string(), Value::Object(pinned_role_json(model)));
            }
            SmartRoleUpdate::FamilyLock { provider, family, class, freshness } => {
                roles.insert(
                    key.to_string(),
                    Value::Object(family_role_json(provider, family, class, *freshness)),
                );
            }
        }
        Ok(())
    })
}

/// Mask every stored key that resolves to `canonical_key`'s profile — the
/// canonical key itself and any alias spellings — with a deletion tombstone in
/// the merged object, so a lower-root override under any of those spellings is
/// durably masked once written to the primary root. The caller then overwrites
/// the canonical key with the real override (pin/family/Auto tombstone). A bare
/// remove would only drop the keys from the primary write and let a lower-root
/// alias resurrect on the next merged read.
fn tombstone_subagent_aliases(subagents: &mut JsonMap<String, Value>, canonical_key: &str) {
    let aliases = subagents
        .keys()
        .filter(|existing| {
            SubagentProfileId::parse(existing)
                .is_some_and(|profile| profile.key() == canonical_key)
        })
        .cloned()
        .collect::<Vec<_>>();
    for alias in aliases {
        subagents.insert(alias, Value::Object(deleted_role_json()));
    }
}

pub(crate) fn write_global_smart_subagent(
    subagent: &str,
    update: &SmartRoleUpdate,
) -> io::Result<()> {
    let profile = SubagentProfileId::parse(subagent).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported smart subagent `{subagent}`"),
        )
    })?;
    let key = profile.key().to_string();
    update_global_settings(|root| {
        let model_router = object_child(root, "modelRouter")?;
        let subagents = object_child(model_router, "subagents")?;
        // Mask the canonical key and every alias spelling with tombstones so a
        // lower-root override (under any spelling) cannot resurrect; the arms
        // below then overwrite the canonical key with the real state.
        tombstone_subagent_aliases(subagents, &key);
        match update {
            SmartRoleUpdate::Auto => {
                // Auto keeps the canonical tombstone from above: it durably
                // masks any lower-root override back to Auto. A later explicit
                // pin/family write overwrites it via the arms below.
                subagents.insert(key, Value::Object(deleted_role_json()));
            }
            SmartRoleUpdate::ExactPin { model } => {
                subagents.insert(key, Value::Object(pinned_role_json(model)));
            }
            SmartRoleUpdate::FamilyLock { provider, family, class, freshness } => {
                subagents.insert(
                    key,
                    Value::Object(family_role_json(provider, family, class, *freshness)),
                );
            }
        }
        Ok(())
    })
}

fn update_global_settings(
    update: impl FnOnce(&mut JsonMap<String, Value>) -> io::Result<()>,
) -> io::Result<()> {
    let paths = global_settings_paths();
    let path = paths
        .first()
        .cloned()
        .unwrap_or_else(|| default_config_home().join("settings.json"));
    update_settings_file_from_paths(&path, &paths, update)
}

pub(super) fn update_settings_file(
    path: &Path,
    update: impl FnOnce(&mut JsonMap<String, Value>) -> io::Result<()>,
) -> io::Result<()> {
    update_settings_file_from_paths(path, &[path.to_path_buf()], update)
}

fn update_settings_file_from_paths(
    path: &Path,
    read_paths: &[PathBuf],
    update: impl FnOnce(&mut JsonMap<String, Value>) -> io::Result<()>,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let _lock = SettingsFileLock::acquire(path)?;
    let mut root = read_merged_global_settings_object(read_paths)?;
    update(&mut root)?;
    let rendered = serde_json::to_string_pretty(&Value::Object(root))?;
    crate::write_atomic(path, format!("{rendered}\n").as_bytes())
}

struct SettingsFileLock {
    path: PathBuf,
}

impl SettingsFileLock {
    fn acquire(settings_path: &Path) -> io::Result<Self> {
        let lock_path = settings_path.with_extension("json.lock");
        for _ in 0..200 {
            match OpenOptions::new().write(true).create_new(true).open(&lock_path) {
                Ok(mut file) => {
                    writeln!(file, "{}", std::process::id())?;
                    return Ok(Self { path: lock_path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            format!("timed out waiting for {}", lock_path.display()),
        ))
    }
}

impl Drop for SettingsFileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn read_merged_global_settings_object(
    paths: &[PathBuf],
) -> io::Result<JsonMap<String, Value>> {
    let mut merged = JsonMap::new();
    for path in paths.iter().rev() {
        merge_settings_objects(&mut merged, read_global_settings_object(path)?);
    }
    Ok(merged)
}

fn merge_settings_objects(target: &mut JsonMap<String, Value>, incoming: JsonMap<String, Value>) {
    for (key, value) in incoming {
        match (target.get_mut(&key), value) {
            (Some(Value::Object(target_object)), Value::Object(incoming_object)) => {
                merge_settings_objects(target_object, incoming_object);
            }
            (_, value) => {
                target.insert(key, value);
            }
        }
    }
}

fn read_global_settings_object(path: &Path) -> io::Result<JsonMap<String, Value>> {
    let mut last_invalid = None;
    for attempt in 0..3 {
        match fs::read_to_string(path) {
            Ok(text) if text.trim().is_empty() => return Ok(JsonMap::new()),
            Ok(text) => match parse_settings_object(path, &text) {
                Ok(root) => return Ok(root),
                Err(error) if attempt < 2 && error.kind() == io::ErrorKind::InvalidData => {
                    last_invalid = Some(error);
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(error),
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(JsonMap::new()),
            Err(error) => return Err(error),
        }
    }
    Err(last_invalid.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid JSON in {}", path.display()),
        )
    }))
}

fn parse_settings_object(path: &Path, text: &str) -> io::Result<JsonMap<String, Value>> {
    let value: Value = serde_json::from_str(text).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid JSON in {}: {error}", path.display()),
        )
    })?;
    value.as_object().cloned().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} must contain a JSON object", path.display()),
        )
    })
}

fn normalize_role(role: &str) -> Option<&'static str> {
    match role.trim().to_ascii_lowercase().as_str() {
        "default" => Some("default"),
        "fast" => Some("fast"),
        "coding" => Some("coding"),
        "debugging" => Some("debugging"),
        "verifier" => Some("verifier"),
        "reviewer" => Some("reviewer"),
        "analysis" => Some("analysis"),
        "research" => Some("research"),
        "writing" => Some("writing"),
        "design" => Some("design"),
        "judge" => Some("judge"),
        "synthesizer" => Some("synthesizer"),
        _ => None,
    }
}

fn freshness_label(freshness: SmartFreshness) -> &'static str {
    match freshness {
        SmartFreshness::Latest => "latest",
        SmartFreshness::LatestStable => "latestStable",
    }
}

fn object_child<'a>(
    root: &'a mut JsonMap<String, Value>,
    key: &str,
) -> io::Result<&'a mut JsonMap<String, Value>> {
    if !root.contains_key(key) {
        root.insert(key.to_string(), Value::Object(JsonMap::new()));
    }
    match root.get_mut(key) {
        Some(Value::Object(object)) => Ok(object),
        Some(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("settings key `{key}` must be a JSON object"),
        )),
        None => unreachable!("object child was inserted"),
    }
}

fn pinned_role_json(model: &str) -> JsonMap<String, Value> {
    let mut config = JsonMap::new();
    config.insert("mode".to_string(), Value::String("pinned".to_string()));
    config.insert("model".to_string(), Value::String(model.to_string()));
    config
}

fn family_role_json(
    provider: &str,
    family: &str,
    class: &str,
    freshness: SmartFreshness,
) -> JsonMap<String, Value> {
    let mut selector = JsonMap::new();
    selector.insert("provider".to_string(), Value::String(provider.to_string()));
    selector.insert("family".to_string(), Value::String(family.to_string()));
    selector.insert("class".to_string(), Value::String(class.to_string()));
    selector.insert(
        "freshness".to_string(),
        Value::String(freshness_label(freshness).to_string()),
    );

    let mut config = JsonMap::new();
    config.insert(
        "mode".to_string(),
        Value::String("manualPreferred".to_string()),
    );
    config.insert("selector".to_string(), Value::Object(selector));
    config
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_config_home(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zo-smart-settings-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn with_config_home<T>(config_home: &std::path::Path, run: impl FnOnce() -> T) -> T {
        with_test_config_home(config_home, run)
    }

    /// Run `body` with `ZO_SMART_DYNAMIC_FLOOR` set to `value` (`None` =
    /// unset), under the crate env lock, restoring the prior value after.
    fn with_dynamic_floor_env<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
        let _guard = crate::test_env_lock();
        let prior = std::env::var_os(SMART_DYNAMIC_FLOOR_ENV);
        match value {
            Some(value) => std::env::set_var(SMART_DYNAMIC_FLOOR_ENV, value),
            None => std::env::remove_var(SMART_DYNAMIC_FLOOR_ENV),
        }
        let output = body();
        match prior {
            Some(prior) => std::env::set_var(SMART_DYNAMIC_FLOOR_ENV, prior),
            None => std::env::remove_var(SMART_DYNAMIC_FLOOR_ENV),
        }
        output
    }

    fn with_agent_defs_env<T>(path: &Path, body: impl FnOnce() -> T) -> T {
        let _guard = crate::test_env_lock();
        let prior = std::env::var_os("ZO_AGENT_DEFS_DIR");
        std::env::set_var("ZO_AGENT_DEFS_DIR", path);
        let output = body();
        match prior {
            Some(prior) => std::env::set_var("ZO_AGENT_DEFS_DIR", prior),
            None => std::env::remove_var("ZO_AGENT_DEFS_DIR"),
        }
        output
    }

    fn with_merged_config_home<T>(config_home: &Path, body: impl FnOnce() -> T) -> T {
        let _guard = crate::test_env_lock();
        let prior_config_home = std::env::var_os("ZO_CONFIG_HOME");
        let prior_zo_home = std::env::var_os("ZO_HOME");
        let prior_home = std::env::var_os("HOME");
        std::env::set_var("ZO_CONFIG_HOME", config_home);
        std::env::remove_var("ZO_HOME");
        // `zo_global_config_roots` also discovers `$HOME/.zo` as a lower-priority
        // user root that merges UNDER `ZO_CONFIG_HOME`, so every key this temp
        // home leaves unset would still come from the developer's real global
        // settings — and these tests assert built-in defaults.
        std::env::set_var("HOME", config_home);
        runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides::default());
        let output = body();
        runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides::default());
        match prior_config_home {
            Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
            None => std::env::remove_var("ZO_CONFIG_HOME"),
        }
        match prior_zo_home {
            Some(value) => std::env::set_var("ZO_HOME", value),
            None => std::env::remove_var("ZO_HOME"),
        }
        match prior_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
        output
    }

    /// The failing row is the one the whole section exists to print, so pin its
    /// exact shape: the tag must be visually distinct from an ordinary decline,
    /// the failure reasons must be ranked, and declines must appear WITHOUT
    /// being counted as failures.
    #[test]
    fn harness_row_marks_a_never_succeeding_feature_as_failing() {
        let mut ledger = telemetry::HarnessAttestLedger::new();
        for _ in 0..17 {
            ledger.record_failed(telemetry::HarnessFeature::RoutingProbe, "provider_failure");
        }
        for _ in 0..4 {
            ledger.record_failed(telemetry::HarnessFeature::RoutingProbe, "timeout");
        }
        for _ in 0..9 {
            ledger.record_declined(telemetry::HarnessFeature::RoutingProbe, "not_worth_it");
        }
        let attestation = ledger.attestation(telemetry::HarnessFeature::RoutingProbe);
        let rows = harness_attest_rows(telemetry::HarnessFeature::RoutingProbe, &attestation);
        assert_eq!(rows.len(), 1, "a failing row needs no precondition hint: {rows:?}");
        let row = &rows[0];
        assert!(row.contains("[FAILING]"), "{row}");
        assert!(row.contains("routing probe"), "{row}");
        assert!(
            row.contains("failed 21: provider_failure x17, timeout x4"),
            "failures must be ranked most-frequent first: {row}"
        );
        assert!(
            row.contains("declined 9: not_worth_it x9"),
            "declines must be shown separately from failures: {row}"
        );
    }

    /// A session with no design turn declines the reminder on every turn. That
    /// must NOT print as a defect — the moment this table cries wolf on normal
    /// operation, its reader stops believing the rows that matter.
    #[test]
    fn harness_row_keeps_ordinary_declines_out_of_the_failing_tag() {
        let mut ledger = telemetry::HarnessAttestLedger::new();
        for _ in 0..40 {
            ledger.record_declined(telemetry::HarnessFeature::DesignGuidance, "intent_not_design");
        }
        let attestation = ledger.attestation(telemetry::HarnessFeature::DesignGuidance);
        let rows = harness_attest_rows(telemetry::HarnessFeature::DesignGuidance, &attestation);
        let row = &rows[0];
        assert!(row.contains("[  gated]"), "{row}");
        assert!(!row.contains("FAILING"), "{row}");
        assert!(!row.contains("failed"), "a decline is not a failure: {row}");
        assert!(row.contains("declined 40: intent_not_design x40"), "{row}");
    }

    /// A feature that recorded NOTHING is the one case the counts cannot
    /// explain on their own, so — and only then — the precondition is printed
    /// beneath it. Without it, an Anthropic-only session reads its structurally
    /// silent replay rows as two defects.
    #[test]
    fn harness_row_explains_a_silent_feature_with_its_precondition() {
        let ledger = telemetry::HarnessAttestLedger::new();
        let feature = telemetry::HarnessFeature::ReasoningReplayTurnFinal;
        let rows = harness_attest_rows(feature, &ledger.attestation(feature));
        assert_eq!(rows.len(), 2, "silent rows carry a precondition line: {rows:?}");
        assert!(rows[0].contains("[   idle]"), "{:?}", rows[0]);
        assert!(
            rows[1].contains("OpenAI Responses wire only"),
            "{:?}",
            rows[1]
        );
    }

    /// The table is scanned vertically — a reader looks down the count column
    /// for zeros. A label that outgrows [`HARNESS_LABEL_WIDTH`] would silently
    /// break that scan, so the alignment is a contract, not a cosmetic.
    #[test]
    fn harness_rows_align_their_count_column() {
        let ledger = telemetry::HarnessAttestLedger::new();
        let mut columns = std::collections::BTreeSet::new();
        for (feature, attestation) in ledger.rows() {
            let row = harness_attest_rows(feature, &attestation).remove(0);
            // A fresh ledger renders no trailing reason text, so the last `x`
            // is the count suffix.
            columns.insert(row.rfind('x').expect("count column present"));
        }
        assert_eq!(
            columns.len(),
            1,
            "every feature row must place its count in the same column: {columns:?}"
        );
    }

    #[test]
    fn harness_section_reports_emptiness_instead_of_an_all_zero_table() {
        // The process-wide ledger may already hold entries from sibling tests,
        // so assert on the two mutually exclusive shapes rather than on one.
        let section = smart_doctor_harness_section();
        assert!(section.contains("Harness firings (this process)"), "{section}");
        if telemetry::harness_attest_snapshot().is_empty() {
            assert!(section.contains("nothing recorded yet"), "{section}");
        } else {
            assert!(
                section.contains("routing probe"),
                "a non-empty ledger must enumerate every feature: {section}"
            );
        }
    }

    #[test]
    fn smart_settings_merge_roots_and_write_primary_only() {
        let root = temp_config_home("merged-global-roots");
        let primary = root.join("primary/settings.json");
        let secondary = root.join("secondary/settings.json");
        let legacy = root.join("legacy/settings.json");
        for path in [&primary, &secondary, &legacy] {
            fs::create_dir_all(path.parent().expect("settings parent")).expect("create root");
        }
        fs::write(
            &legacy,
            r#"{"smart":{"enabled":false,"quotaFallback":false},"modelRouter":{"roles":{"coding":{"mode":"pinned","model":"legacy-model"}}}}"#,
        )
        .expect("write legacy settings");
        fs::write(
            &secondary,
            r#"{"smart":{"quotaFallback":true,"allowCrossProviderDiversity":false}}"#,
        )
        .expect("write secondary settings");
        fs::write(&primary, r#"{"smart":{"enabled":true}}"#)
            .expect("write primary settings");
        let secondary_before = fs::read_to_string(&secondary).expect("secondary before");
        let legacy_before = fs::read_to_string(&legacy).expect("legacy before");
        let paths = vec![primary.clone(), secondary.clone(), legacy.clone()];

        let merged = read_merged_global_settings_object(&paths).expect("merged settings");
        let snapshot = snapshot_from_root(&Value::Object(merged));
        assert!(snapshot.enabled, "primary value must win");
        assert!(snapshot.quota_fallback, "secondary value must win");
        assert!(!snapshot.allow_cross_provider_diversity);
        assert_eq!(
            snapshot.roles.get("coding"),
            Some(&SmartRoleUpdate::ExactPin {
                model: "legacy-model".to_string()
            })
        );

        update_settings_file_from_paths(&primary, &paths, |root| {
            object_child(root, "smart")?.insert("enabled".to_string(), Value::Bool(false));
            Ok(())
        })
        .expect("write primary merged settings");
        let written = read_settings(&primary);
        assert_eq!(written["smart"]["enabled"], Value::Bool(false));
        assert_eq!(written["smart"]["quotaFallback"], Value::Bool(true));
        assert_eq!(written["modelRouter"]["roles"]["coding"]["model"], "legacy-model");
        assert_eq!(
            fs::read_to_string(&secondary).expect("secondary after"),
            secondary_before
        );
        assert_eq!(fs::read_to_string(&legacy).expect("legacy after"), legacy_before);
    }

    /// Set up a primary + lower canonical root pair where the OVERRIDE lives
    /// only in the lower root, returning the two settings paths and the ordered
    /// path vec (primary first). The lower root pins `roles.coding` and
    /// `modelRouter.subagents.Verification`; the primary starts empty.
    fn lower_root_override_fixture(label: &str) -> (PathBuf, PathBuf, Vec<PathBuf>) {
        let root = temp_config_home(label);
        let primary = root.join("primary/settings.json");
        let lower = root.join("lower/settings.json");
        for path in [&primary, &lower] {
            fs::create_dir_all(path.parent().expect("settings parent")).expect("create root");
        }
        fs::write(
            &lower,
            r#"{"modelRouter":{"roles":{"coding":{"mode":"pinned","model":"lower-model"}},"subagents":{"Verification":{"mode":"pinned","model":"lower-verify"}}}}"#,
        )
        .expect("write lower settings");
        fs::write(&primary, "{}").expect("write primary settings");
        let paths = vec![primary.clone(), lower.clone()];
        (primary, lower, paths)
    }

    fn merged_snapshot(paths: &[PathBuf]) -> SmartSettingsSnapshot {
        let merged = read_merged_global_settings_object(paths).expect("merged settings");
        snapshot_from_root(&Value::Object(merged))
    }

    #[test]
    fn reset_all_tombstones_mask_lower_root_overrides_primary_only() {
        let (primary, lower, paths) = lower_root_override_fixture("reset-all-lower-root");
        let lower_before = fs::read_to_string(&lower).expect("lower before");

        // Sanity: the lower-root overrides are visible before the reset.
        let before = merged_snapshot(&paths);
        assert_eq!(
            before.roles.get("coding"),
            Some(&SmartRoleUpdate::ExactPin { model: "lower-model".to_string() })
        );
        assert_eq!(
            before.subagents.get("Verification"),
            Some(&SmartRoleUpdate::ExactPin { model: "lower-verify".to_string() })
        );

        // Reset-all: the same closure `write_global_reset_overrides` runs,
        // driven against the multi-root path vec (the writers themselves resolve
        // a single test path via `SMART_TEST_CONFIG_HOME`).
        update_settings_file_from_paths(&primary, &paths, |root| {
            let model_router = object_child(root, "modelRouter")?;
            tombstone_all_overrides(object_child(model_router, "roles")?);
            tombstone_all_overrides(object_child(model_router, "subagents")?);
            Ok(())
        })
        .expect("reset all");

        // Readers now see no overrides — the lower-root pins are masked.
        let after = merged_snapshot(&paths);
        assert!(!after.roles.contains_key("coding"), "role override must be masked");
        assert!(
            !after.subagents.contains_key("Verification"),
            "subagent override must be masked"
        );

        // The mask is a tombstone in the PRIMARY root; the lower root is intact.
        let written = read_settings(&primary);
        assert_eq!(written["modelRouter"]["roles"]["coding"]["mode"], "deleted");
        assert_eq!(written["modelRouter"]["subagents"]["Verification"]["mode"], "deleted");
        assert_eq!(fs::read_to_string(&lower).expect("lower after"), lower_before);
    }

    #[test]
    fn individual_auto_tombstones_mask_single_lower_root_override() {
        let (primary, lower, paths) = lower_root_override_fixture("individual-auto-lower-root");
        let lower_before = fs::read_to_string(&lower).expect("lower before");

        // Set only the `coding` role to Auto — the same tombstone
        // `write_global_smart_role(SmartRoleUpdate::Auto)` writes.
        update_settings_file_from_paths(&primary, &paths, |root| {
            let model_router = object_child(root, "modelRouter")?;
            let roles = object_child(model_router, "roles")?;
            roles.insert("coding".to_string(), Value::Object(deleted_role_json()));
            Ok(())
        })
        .expect("reset coding to auto");

        let after = merged_snapshot(&paths);
        assert!(!after.roles.contains_key("coding"), "coding must be masked to Auto");
        // The untouched subagent override still resolves from the lower root.
        assert_eq!(
            after.subagents.get("Verification"),
            Some(&SmartRoleUpdate::ExactPin { model: "lower-verify".to_string() })
        );

        let written = read_settings(&primary);
        assert_eq!(written["modelRouter"]["roles"]["coding"]["mode"], "deleted");
        assert_eq!(fs::read_to_string(&lower).expect("lower after"), lower_before);
    }

    #[test]
    fn explicit_write_replaces_tombstone_and_stays_primary_only() {
        let (primary, lower, paths) = lower_root_override_fixture("write-replaces-tombstone");
        let lower_before = fs::read_to_string(&lower).expect("lower before");

        // First mask `coding` to Auto with a tombstone.
        update_settings_file_from_paths(&primary, &paths, |root| {
            let roles = object_child(object_child(root, "modelRouter")?, "roles")?;
            roles.insert("coding".to_string(), Value::Object(deleted_role_json()));
            Ok(())
        })
        .expect("tombstone coding");
        assert!(!merged_snapshot(&paths).roles.contains_key("coding"));

        // A later explicit pin overwrites the tombstone in the primary root.
        update_settings_file_from_paths(&primary, &paths, |root| {
            let roles = object_child(object_child(root, "modelRouter")?, "roles")?;
            roles.insert("coding".to_string(), Value::Object(pinned_role_json("primary-model")));
            Ok(())
        })
        .expect("pin coding");

        let after = merged_snapshot(&paths);
        assert_eq!(
            after.roles.get("coding"),
            Some(&SmartRoleUpdate::ExactPin { model: "primary-model".to_string() }),
            "explicit write must replace the tombstone"
        );

        // The pin landed in the primary root only; the lower root is untouched.
        let written = read_settings(&primary);
        assert_eq!(written["modelRouter"]["roles"]["coding"]["mode"], "pinned");
        assert_eq!(written["modelRouter"]["roles"]["coding"]["model"], "primary-model");
        assert_eq!(fs::read_to_string(&lower).expect("lower after"), lower_before);
    }

    #[test]
    fn tombstone_reads_as_no_override_and_preserves_nested_precedence() {
        // A tombstone in the primary over a pin in the lower reads as no
        // override, while an unrelated role pinned only in the lower still wins
        // — the deletion is scoped to its key, nested precedence is intact.
        let root = temp_config_home("tombstone-nested-precedence");
        let primary = root.join("primary/settings.json");
        let lower = root.join("lower/settings.json");
        for path in [&primary, &lower] {
            fs::create_dir_all(path.parent().expect("settings parent")).expect("create root");
        }
        fs::write(
            &lower,
            r#"{"modelRouter":{"roles":{"coding":{"mode":"pinned","model":"lower-coding"},"reviewer":{"mode":"pinned","model":"lower-reviewer"}}}}"#,
        )
        .expect("write lower settings");
        fs::write(
            &primary,
            r#"{"modelRouter":{"roles":{"coding":{"mode":"deleted"}}}}"#,
        )
        .expect("write primary settings");
        let paths = vec![primary.clone(), lower.clone()];

        let snapshot = merged_snapshot(&paths);
        assert!(!snapshot.roles.contains_key("coding"), "tombstoned role masked");
        assert_eq!(
            snapshot.roles.get("reviewer"),
            Some(&SmartRoleUpdate::ExactPin { model: "lower-reviewer".to_string() }),
            "sibling lower-root override still resolves"
        );
    }

    #[test]
    fn deep_tier_command_add_materializes_default_and_deduplicates_aliases() {
        let root = temp_config_home("deep-tier-command-add");
        let config_home = root.join("config");
        let cwd = root.join("project");
        fs::create_dir_all(&cwd).expect("project dir");

        with_merged_config_home(&config_home, || {
            let shown = execute_deep_tier_command(&cwd, &commands::DeepTierAction::Show)
                .expect("show default pool");
            assert!(shown.contains("Deep-tier pool (built-in default)"), "{shown}");
            assert!(shown.contains("1. fable"), "{shown}");
            assert!(shown.contains("2. openai-latest"), "{shown}");
            assert!(shown.contains(commands::DEEP_TIER_USAGE), "{shown}");

            let duplicate = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Add {
                    model: "fable".to_string(),
                },
            )
            .expect("duplicate add is a no-op");
            assert!(duplicate.contains("already in the deep-tier pool"));
            assert!(!super::super::session_preferences::project_settings_path(&cwd).exists());

            let requested = "opus-5.0";
            let resolved = api::resolve_model_alias(requested);
            let added = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Add {
                    model: requested.to_string(),
                },
            )
            .expect("add model");
            assert!(added.contains("Applies from next turn."), "{added}");

            let project =
                read_settings(&super::super::session_preferences::project_settings_path(&cwd));
            assert_eq!(
                project["smart"]["deepTierModels"],
                serde_json::json!([
                    api::ANTHROPIC_FABLE_MODEL_ALIAS,
                    api::OPENAI_LATEST_MODEL_ALIAS,
                    resolved
                ])
            );
            let active = tools::smart_deep_tier_models_for(&cwd).expect("merged pool");
            assert!(active.configured);
            assert_eq!(
                active.models,
                vec![
                    api::latest_anthropic_family_model(api::ANTHROPIC_FABLE_MODEL_ALIAS)
                        .expect("provider catalog must define the current Fable head")
                        .to_string(),
                    api::latest_model_for_provider(api::ProviderKind::OpenAi)
                        .expect("provider catalog must define the current OpenAI head")
                        .to_string(),
                    api::resolve_model_alias(requested)
                ]
            );
        });
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn deep_tier_command_removes_by_index_and_alias_then_resets() {
        let root = temp_config_home("deep-tier-command-remove-reset");
        let config_home = root.join("config");
        let cwd = root.join("project");
        let settings_path = super::super::session_preferences::project_settings_path(&cwd);
        fs::create_dir_all(settings_path.parent().expect("settings parent"))
            .expect("project settings dir");
        fs::write(
            &settings_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "smart": {
                    "deepTierModels": ["claude-fable-5", "gpt-5.6-sol", "claude-opus-5"]
                }
            }))
            .expect("json"),
        )
        .expect("write project settings");

        with_merged_config_home(&config_home, || {
            let removed = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Remove {
                    target: "2".to_string(),
                },
            )
            .expect("remove by index");
            assert!(removed.contains("#2 (gpt-5.6-sol)"), "{removed}");

            let removed = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Remove {
                    target: "fable".to_string(),
                },
            )
            .expect("remove by alias");
            assert!(removed.contains("claude-fable-5"), "{removed}");
            let error = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Remove {
                    target: "1".to_string(),
                },
            )
            .expect_err("last removal is rejected");
            assert!(error.contains("Cannot remove the last deep-tier model"));

            let reset = execute_deep_tier_command(&cwd, &commands::DeepTierAction::Reset)
                .expect("reset pool");
            assert!(reset.contains("built-in default"), "{reset}");
            assert!(reset.contains("Applies from next turn."), "{reset}");
            let project = read_settings(&settings_path);
            assert!(project["smart"].get("deepTierModels").is_none());
            let active = tools::smart_deep_tier_models_for(&cwd).expect("default pool");
            assert!(!active.configured);
            assert_eq!(active.models, runtime::default_deep_tier_models());

            fs::write(
                &settings_path,
                serde_json::to_string_pretty(
                    &serde_json::json!({"smart": {"deepTierModels": []}}),
                )
                .expect("json"),
            )
            .expect("write empty pool");
            let empty = tools::smart_deep_tier_models_for(&cwd).expect("empty pool fallback");
            assert!(!empty.configured);
            assert_eq!(empty.models, runtime::default_deep_tier_models());
        });
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn deep_tier_command_move_persists_preference_order() {
        let root = temp_config_home("deep-tier-command-move");
        let config_home = root.join("config");
        let cwd = root.join("project");
        let settings_path = super::super::session_preferences::project_settings_path(&cwd);
        fs::create_dir_all(settings_path.parent().expect("settings parent"))
            .expect("project settings dir");
        fs::write(
            &settings_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "smart": {
                    "deepTierModels": ["architect-a", "architect-b", "architect-c"]
                }
            }))
            .expect("json"),
        )
        .expect("write project settings");

        with_merged_config_home(&config_home, || {
            let moved = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Move { from: 3, to: 1 },
            )
            .expect("move model");
            assert!(moved.contains("Moved #3 (architect-c) to #1"), "{moved}");

            let active = tools::smart_deep_tier_models_for(&cwd).expect("moved pool");
            assert!(active.configured);
            assert_eq!(
                active.models,
                vec![
                    "architect-c".to_string(),
                    "architect-a".to_string(),
                    "architect-b".to_string()
                ]
            );

            let error = execute_deep_tier_command(
                &cwd,
                &commands::DeepTierAction::Move { from: 4, to: 1 },
            )
            .expect_err("out-of-range move is rejected");
            assert!(error.contains("between 1 and 3"), "{error}");
        });
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn trusted_probe_downshifts_simple_turns_while_keyword_verdicts_only_raise() {
        use api::EffortLevel as L;
        use runtime::RouteAssessmentProvenance as P;
        use runtime::RouteTaskComplexity as C;
        use runtime::RouteTaskIntent as I;

        with_dynamic_floor_env(None, || {
            // The same easy complexity is authoritative only after a probe
            // clears fusion's confidence gate.
            for complexity in [C::Trivial, C::Small] {
                assert_eq!(
                    smart_turn_effort_band_for_complexity(
                        complexity,
                        I::Other,
                        P::Deterministic,
                    ),
                    None,
                    "{complexity:?}: keyword-only verdicts must not lower effort"
                );
            }
            assert_eq!(
                smart_turn_effort_band_for_complexity(
                    C::Trivial,
                    I::Other,
                    P::TrustedProbe,
                ),
                Some((L::Low, L::Xhigh))
            );
            assert_eq!(
                smart_turn_effort_band_for_complexity(
                    C::Small,
                    I::Other,
                    P::TrustedProbe,
                ),
                Some((L::Medium, L::Xhigh))
            );
            for provenance in [P::Deterministic, P::TrustedProbe] {
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Medium, I::Other, provenance),
                    None
                );
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Unknown, I::Other, provenance),
                    None
                );
                // Repo-scale work raises unconditionally.
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Large, I::Other, provenance),
                    Some((L::Max, L::Ultra))
                );
            }
        });
    }

    fn with_design_guidance_env<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
        let _guard = crate::test_env_lock();
        let prior = std::env::var_os(DESIGN_GUIDANCE_ENV);
        match value {
            Some(value) => std::env::set_var(DESIGN_GUIDANCE_ENV, value),
            None => std::env::remove_var(DESIGN_GUIDANCE_ENV),
        }
        let output = body();
        match prior {
            Some(prior) => std::env::set_var(DESIGN_GUIDANCE_ENV, prior),
            None => std::env::remove_var(DESIGN_GUIDANCE_ENV),
        }
        output
    }

    #[test]
    fn design_intent_keeps_the_complexity_floor_and_caps_only_non_large_work() {
        use api::EffortLevel as L;
        use runtime::RouteAssessmentProvenance as P;
        use runtime::RouteTaskComplexity as C;
        use runtime::RouteTaskIntent as I;

        with_design_guidance_env(None, || {
            // Trusted easy verdicts keep the ordinary complexity-derived floor.
            assert_eq!(
                smart_turn_effort_band_for_complexity(C::Trivial, I::Design, P::TrustedProbe),
                Some((L::Low, L::Xhigh))
            );
            assert_eq!(
                smart_turn_effort_band_for_complexity(C::Small, I::Design, P::TrustedProbe),
                Some((L::Medium, L::Xhigh))
            );
            // Medium/Unknown normally keep xhigh..=max; Design only applies
            // the independently measured xhigh ceiling.
            for complexity in [C::Medium, C::Unknown] {
                assert_eq!(
                    smart_turn_effort_band_for_complexity(
                        complexity,
                        I::Design,
                        P::TrustedProbe,
                    ),
                    Some((L::Xhigh, L::Xhigh)),
                    "{complexity:?}"
                );
            }
            // Large raises unconditionally and no intent may lower its floor.
            assert_eq!(
                smart_turn_effort_band_for_complexity(C::Large, I::Design, P::TrustedProbe),
                Some((L::Max, L::Ultra))
            );
            // Other trusted intents retain the ordinary mapping.
            for intent in [I::Other, I::Implementation, I::Analysis] {
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Small, intent, P::TrustedProbe),
                    Some((L::Medium, L::Xhigh)),
                    "{intent:?}"
                );
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Trivial, intent, P::TrustedProbe),
                    Some((L::Low, L::Xhigh)),
                    "{intent:?}"
                );
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Large, intent, P::TrustedProbe),
                    Some((L::Max, L::Ultra)),
                    "{intent:?}"
                );
                assert_eq!(
                    smart_turn_effort_band_for_complexity(C::Medium, intent, P::TrustedProbe),
                    None,
                    "{intent:?}"
                );
            }
            // Even an impossible hand-built deterministic Design value cannot
            // authorize a lower floor or a cap.
            assert_eq!(
                smart_turn_effort_band_for_complexity(C::Small, I::Design, P::Deterministic),
                None
            );
            assert_eq!(
                smart_turn_effort_band_for_complexity(C::Medium, I::Design, P::Deterministic),
                None
            );
            assert_eq!(
                smart_turn_effort_band_for_complexity(C::Large, I::Design, P::Deterministic),
                Some((L::Max, L::Ultra))
            );
        });
    }

    #[test]
    fn design_guidance_kill_switch_removes_only_the_design_ceiling() {
        use api::EffortLevel as L;
        use runtime::RouteAssessmentProvenance as P;
        use runtime::RouteTaskIntent as I;

        for value in ["off", "OFF", "0"] {
            with_design_guidance_env(Some(value), || {
                assert!(!design_guidance_enabled(), "{value}");
                assert_eq!(
                    smart_turn_effort_band_for_complexity(
                        RouteTaskComplexity::Small,
                        I::Design,
                        P::TrustedProbe,
                    ),
                    Some((L::Medium, L::Xhigh)),
                    "{value}"
                );
                assert_eq!(
                    smart_turn_effort_band_for_complexity(
                        RouteTaskComplexity::Medium,
                        I::Design,
                        P::TrustedProbe,
                    ),
                    None,
                    "{value}"
                );
            });
        }
        with_design_guidance_env(None, || {
            assert!(design_guidance_enabled());
            assert_eq!(
                smart_turn_effort_band_for_complexity(
                    RouteTaskComplexity::Medium,
                    I::Design,
                    P::TrustedProbe,
                ),
                Some((L::Xhigh, L::Xhigh))
            );
        });
    }

    #[test]
    fn smart_turn_effort_band_kill_switch_disables_downshift() {
        use runtime::RouteAssessmentProvenance as P;
        use runtime::RouteTaskIntent as I;

        with_dynamic_floor_env(Some("off"), || {
            assert_eq!(
                smart_turn_effort_band_for_complexity(
                    RouteTaskComplexity::Small,
                    I::Other,
                    P::TrustedProbe,
                ),
                None
            );
        });
    }

    fn read_settings(path: &std::path::Path) -> Value {
        serde_json::from_str(&fs::read_to_string(path).expect("read settings"))
            .expect("settings json")
    }

    fn assert_no_smart_rich_or_emoji_glyphs(output: &str) {
        let banned = [
            glyphs::SMART_AUTO,
            glyphs::SMART_MODEL,
            glyphs::SMART_FAST,
            glyphs::SMART_CODE,
            glyphs::SMART_VERIFY,
            glyphs::SMART_RESEARCH,
            glyphs::SMART_REVIEW,
            glyphs::SMART_DESIGN,
            glyphs::SMART_PIN,
            glyphs::SMART_FALLBACK,
            "\u{2713}",
            "\u{26a1}",
            "\u{1f50e}",
            "\u{1f4cc}",
        ];
        for glyph in banned {
            assert!(
                !output.contains(glyph),
                "unexpected rich/emoji glyph in Smart output: {glyph}"
            );
        }
    }

    #[test]
    fn smart_status_renders_usable_models_and_subagent_recommendations() {
        let config_home = temp_config_home("status-board");
        with_config_home(&config_home, || {
            let status = render_smart_status("unlisted-current-model", Some(false))
                .expect("status");
            assert!(status.contains("Usable models:"));
            assert!(status.contains("Main model:"));
            assert!(status.contains("Usable Models"));
            assert!(status.contains("Recommended Subagent Setup"));
            assert!(status.contains("C general-purpose"));
            assert!(status.contains("Agent Types"));
            assert!(status.contains("Unspecified spawns are auto-classified before routing."));
            assert!(status.contains("tools=read-only"));
            assert!(
                ["[###]", "[##-]", "[#--]"].iter().any(|meter| status.contains(meter)),
                "Smart status should render an ASCII confidence meter: {status}"
            );
            assert!(status.contains("Legend: F fast | C code | V verify | Q review"));
            assert!(status.contains("R research | D design | S auto | [###]/[##-]/[#--] fit"));
            assert_no_smart_rich_or_emoji_glyphs(&status);
            assert!(status.contains("Run `/smart` to edit settings in the GUI."));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_agents_renders_builtin_and_loaded_custom_agents() {
        let config_home = temp_config_home("agent-types");
        let defs = config_home.join("agent-defs");
        fs::create_dir_all(&defs).expect("agent defs dir");
        let custom_path = defs.join("custom-linter.md");
        fs::write(
            &custom_path,
            "---\nname: custom-linter\nmodel: claude-test\npermission_mode: read-only\n---\nLint the requested files.",
        )
        .expect("custom agent");

        with_agent_defs_env(&defs, || {
            with_config_home(&config_home, || {
                let rendered = execute_smart_text_command(
                    "unlisted-current-model",
                    Some("agents"),
                )
                .expect("agent catalog");
                assert!(rendered.contains("Builtin Agents"));
                assert!(rendered.contains("Explore the codebase read-only"));
                assert!(rendered.contains(
                    "custom-linter model=claude-test permission=read-only source="
                ));
                assert!(rendered.contains(
                    std::fs::canonicalize(&custom_path)
                        .expect("canonical custom path")
                        .to_string_lossy()
                        .as_ref()
                ));
            });
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn execute_text_command_exposes_only_status_or_gui_entrypoint() {
        let config_home = temp_config_home("text-command");
        with_config_home(&config_home, || {
            assert!(execute_smart_text_command("main", Some("status"))
                .expect("status")
                .contains("Smart Model Router"));
            assert!(execute_smart_text_command("main", Some("set verifier auto")).is_err());
            assert!(execute_smart_text_command("main", Some("configure")).is_err());
            assert!(execute_smart_text_command("main", Some("_gui fake-token Turn Smart ON")).is_err());
            assert!(execute_smart_text_command("main", None)
                .expect("gui hint")
                .contains("/smart status"));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_on_and_status_surface_override_bypass_warning() {
        let config_home = temp_config_home("override-bypass-warning");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            // A stale pin left in settings silently outranks Auto for that role;
            // turning Smart ON (and status) must say so, and a clean full-auto
            // config must not carry the note.
            fs::write(
                global_settings_path(),
                r#"{"smart":{"enabled":false},"modelRouter":{"roles":{"coding":{"mode":"pinned","model":"some-model"}}}}"#,
            )
            .expect("seed settings");

            let res = execute_smart_text_command("main", Some("on")).unwrap();
            assert!(res.contains("Smart Router enabled."));
            assert!(res.contains("1 saved override(s)"), "warning missing: {res}");
            assert!(res.contains("/smart reset"));

            let status = render_smart_status("main", None).expect("status");
            assert!(
                status.contains("! 1 override(s) take priority over Auto"),
                "status warning missing: {status}"
            );

            execute_smart_text_command("main", Some("reset")).unwrap();
            let res = execute_smart_text_command("main", Some("on")).unwrap();
            assert!(res.contains("Smart Router enabled."));
            assert!(!res.contains("saved override(s)"), "unexpected warning: {res}");
            let status = render_smart_status("main", None).expect("status");
            assert!(!status.contains("take priority over Auto"), "unexpected status warning: {status}");
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn deep_verify_route_is_none_when_smart_is_off() {
        let config_home = temp_config_home("deep-verify-off");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(global_settings_path(), r#"{"smart":{"enabled":false}}"#)
                .expect("seed settings");
            assert_eq!(
                route_deep_verify_model("some-main-model"),
                None,
                "deep-gate verify must stay on the native model while Smart is OFF"
            );
            assert!(
                route_deep_verify_candidates(
                    "some-main-model",
                    &runtime::default_deep_tier_models()
                )
                .is_empty()
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn deep_verify_candidates_preserve_exact_primary_without_inserting_a_fallback() {
        let pinned_model = std::env::var("ZO_AGENT_MODEL")
            .ok()
            .filter(|model| !model.trim().is_empty())
            .unwrap_or_else(|| "gpt-5.6-sol".to_string());
        let config_home = temp_config_home("deep-verify-pinned-primary");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(
                global_settings_path(),
                serde_json::json!({
                    "smart": {"enabled": true, "policy": "classic"},
                    "modelRouter": {
                        "roles": {
                            "verifier": {"mode": "pinned", "model": pinned_model}
                        }
                    }
                })
                .to_string(),
            )
            .expect("seed settings");
            assert_eq!(
                route_deep_verify_candidates(
                    "some-main-model",
                    &runtime::default_deep_tier_models()
                ),
                vec![pinned_model],
                "an exact verifier pin stays first and exact; no hardcoded fallback is appended"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn deep_verify_candidates_honor_an_explicit_native_verify_primary() {
        let config_home = temp_config_home("deep-verify-native-primary");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(
                global_settings_path(),
                r#"{"smart":{"enabled":true,"policy":"classic"},"modelRouter":{"roles":{"verifier":{"mode":"pinned","model":"some-main-model"}}}}"#,
            )
            .expect("seed settings");
            assert!(
                route_deep_verify_candidates(
                    "some-main-model",
                    &runtime::default_deep_tier_models()
                )
                .is_empty(),
                "pinning verifier to the main model must keep native verify instead of choosing an alternate"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn architect_plan_and_verify_routes_stay_deep_under_every_exec_swap_mode() {
        for exec_swap in ["easy", "always", "never"] {
            let config_home = temp_config_home(&format!("deep-lanes-{exec_swap}"));
            with_config_home(&config_home, || {
                fs::create_dir_all(config_home.as_path()).expect("create config home");
                fs::write(
                    global_settings_path(),
                    serde_json::json!({
                        "smart": {
                            "enabled": true,
                            "policy": "architect",
                            "execSwap": exec_swap
                        },
                        "modelRouter": {
                            "roles": {
                                "verifier": {
                                    "mode": "pinned",
                                    "model": "gpt-5.6-terra"
                                }
                            }
                        }
                    })
                    .to_string(),
                )
                .expect("seed settings");

                assert!(architect_deep_lanes_enabled(), "{exec_swap}");
                let candidates = route_deep_verify_candidates(
                    "claude-opus-4-8",
                    &runtime::default_deep_tier_models(),
                );
                assert_eq!(candidates, runtime::default_deep_tier_models(), "{exec_swap}");
            });
            let _ = fs::remove_dir_all(config_home);
        }
    }

    #[test]
    fn custom_deep_tier_pool_replaces_membership_and_orders_candidates() {
        let single = snapshot_from_root(&serde_json::json!({
            "smart": {"deepTierModels": ["claude-opus-5"]}
        }));
        assert!(runtime::is_deep_tier_model("opus-5", &single.deep_tier_models));
        assert!(!runtime::is_deep_tier_model(
            "claude-fable-5",
            &single.deep_tier_models
        ));

        let config_home = temp_config_home("deep-tier-custom-order");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(
                global_settings_path(),
                serde_json::json!({
                    "smart": {
                        "enabled": true,
                        "policy": "architect",
                        "deepTierModels": ["claude-opus-5", "claude-fable-5"]
                    }
                })
                .to_string(),
            )
            .expect("seed settings");
            let pool = read_global_smart_settings()
                .expect("settings")
                .deep_tier_models;
            assert_eq!(
                route_deep_verify_candidates("gpt-5.6-terra", &pool),
                pool,
                "configured preference order must drive the PLAN/VERIFY ladder"
            );
        });
        let _ = fs::remove_dir_all(config_home);

        let future = snapshot_from_root(&serde_json::json!({
            "smart": {"deepTierModels": ["future-flagship-9"]}
        }));
        assert_eq!(
            future.deep_tier_models,
            vec!["future-flagship-9".to_string()],
            "an unresolvable future id must survive parsing"
        );
        assert_eq!(
            snapshot_from_root(&serde_json::json!({"smart": {"deepTierModels": []}}))
                .deep_tier_models,
            runtime::default_deep_tier_models(),
            "an empty configured pool must behave as unset"
        );
    }

    /// Env isolation for `deep_verify_feedback_hint`'s outcome-store read:
    /// `ZO_STATE_DIR` redirects `runtime::record_route_outcome`/
    /// `read_route_outcomes` to a fresh temp dir regardless of the real
    /// process cwd, guarded by the crate-wide env lock (other modules assert
    /// on the same global).
    struct FeedbackHintTestEnv {
        _guard: std::sync::MutexGuard<'static, ()>,
        state_dir: PathBuf,
        prior_state_dir: Option<std::ffi::OsString>,
    }

    impl FeedbackHintTestEnv {
        fn setup(tag: &str) -> Self {
            let guard = crate::test_env_lock();
            let state_dir = temp_config_home(&format!("deep-verify-feedback-{tag}"));
            fs::create_dir_all(&state_dir).expect("state dir");
            let prior_state_dir = std::env::var_os("ZO_STATE_DIR");
            std::env::set_var("ZO_STATE_DIR", &state_dir);
            Self {
                _guard: guard,
                state_dir,
                prior_state_dir,
            }
        }
    }

    impl Drop for FeedbackHintTestEnv {
        fn drop(&mut self) {
            match self.prior_state_dir.take() {
                Some(value) => std::env::set_var("ZO_STATE_DIR", value),
                None => std::env::remove_var("ZO_STATE_DIR"),
            }
            let _ = fs::remove_dir_all(&self.state_dir);
        }
    }

    #[test]
    fn deep_verify_feedback_hint_disabled_when_feedback_informed_auto_is_off() {
        // No env isolation needed: `feedback_informed_auto: false` must return
        // before touching env/fs at all.
        let snapshot = SmartSettingsSnapshot {
            feedback_informed_auto: false,
            ..SmartSettingsSnapshot::default()
        };
        assert_eq!(deep_verify_feedback_hint(&snapshot), RouteFeedbackHint::disabled());
    }

    #[test]
    fn deep_verify_feedback_hint_is_disabled_shape_when_the_outcome_store_is_empty() {
        let _env = FeedbackHintTestEnv::setup("empty");
        let snapshot = SmartSettingsSnapshot {
            feedback_informed_auto: true,
            ..SmartSettingsSnapshot::default()
        };
        assert_eq!(
            deep_verify_feedback_hint(&snapshot),
            RouteFeedbackHint::disabled(),
            "an empty/missing outcome store must degrade to disabled, never panic"
        );
    }

    #[test]
    fn deep_verify_feedback_hint_reads_learned_history_under_its_own_route_key() {
        let env = FeedbackHintTestEnv::setup("learned");
        let cwd = std::env::current_dir().expect("cwd");
        // Two decisive (>=2) completions for the SAME model under the exact
        // route_key `record_deep_verdict_outcomes` (turn_controller.rs) writes
        // to — `"deep-verify:leg"` — so the bucket clears the decisive floor.
        for _ in 0..2 {
            let record = runtime::RouteOutcomeRecord::new(
                "deep-verify",
                "leg",
                "verifier-model-a",
                "completed",
            );
            runtime::record_route_outcome(&cwd, &record).expect("seed outcome");
        }

        let snapshot = SmartSettingsSnapshot {
            feedback_informed_auto: true,
            ..SmartSettingsSnapshot::default()
        };
        // Read through the cwd-injected seam with the exact cwd the seeds
        // were written under: the process cwd can be swapped mid-test by the
        // chdir-ing tests (they hold `cwd_lock`, not this test's env lock).
        let hint = deep_verify_feedback_hint_at(&snapshot, &cwd);
        assert!(hint.enabled, "a learned bucket must produce an enabled hint");
        assert!(
            hint.model_adjustments
                .iter()
                .any(|(model, adjustment)| model == "verifier-model-a" && *adjustment > 0),
            "a fully-completed bucket must boost the model that reliably finishes VERIFY legs: {:?}",
            hint.model_adjustments
        );
        drop(env);
    }

    #[test]
    fn role_override_lowering_matches_router_semantics() {
        assert!(role_override_from_update(&SmartRoleUpdate::Auto).is_none());
        assert!(matches!(
            role_override_from_update(&SmartRoleUpdate::ExactPin { model: "m-1".to_string() }),
            Some(RoleOverride::Pin(model)) if model == "m-1"
        ));
        assert!(matches!(
            role_override_from_update(&SmartRoleUpdate::FamilyLock {
                provider: "openai".to_string(),
                family: "gpt".to_string(),
                class: "strong".to_string(),
                freshness: SmartFreshness::Latest,
            }),
            Some(RoleOverride::Family(_))
        ));
    }

    #[test]
    fn execute_text_command_manages_settings() {
        let config_home = temp_config_home("text-command-manage");
        with_config_home(&config_home, || {
            write_global_smart_enabled(false).unwrap();
            assert!(!read_global_smart_settings().unwrap().enabled);

            let res = execute_smart_text_command("main", Some("on")).unwrap();
            assert!(res.contains("enabled"));
            assert!(read_global_smart_settings().unwrap().enabled);

            let res = execute_smart_text_command("main", Some("off")).unwrap();
            assert!(res.contains("disabled"));
            assert!(!read_global_smart_settings().unwrap().enabled);

            let inventory = connected_model_inventory("main");
            if let Some(model) = inventory.models().first() {
                let model_id = model.id();

                let res = execute_smart_text_command("main", Some(&format!("pin coding {model_id}"))).unwrap();
                assert!(res.contains("Pinned"));
                let snapshot = read_global_smart_settings().unwrap();
                assert!(snapshot.roles.contains_key("coding"));

                let res = execute_smart_text_command("main", Some("auto coding")).unwrap();
                assert!(res.contains("Reset"));
                let snapshot = read_global_smart_settings().unwrap();
                assert!(!snapshot.roles.contains_key("coding"));

                let res = execute_smart_text_command("main", Some(&format!("pin Plan {model_id}"))).unwrap();
                assert!(res.contains("Pinned"));
                let snapshot = read_global_smart_settings().unwrap();
                assert!(snapshot.subagents.contains_key("Plan"));

                let res = execute_smart_text_command("main", Some("reset")).unwrap();
                assert!(res.contains("reset"));
                let snapshot = read_global_smart_settings().unwrap();
                assert!(snapshot.subagents.is_empty());
                assert!(snapshot.roles.is_empty());
            }
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn writes_global_subagent_settings_and_snapshot_reads_them() {
        let config_home = temp_config_home("subagent-write");
        with_config_home(&config_home, || {
            write_global_smart_subagent(
                "verification",
                &SmartRoleUpdate::ExactPin {
                    model: "Verifier/Model-X".to_string(),
                },
            )
            .expect("write subagent");
            let snapshot = read_global_smart_settings().expect("snapshot");
            assert_eq!(
                snapshot.subagents["Verification"],
                SmartRoleUpdate::ExactPin {
                    model: "Verifier/Model-X".to_string()
                }
            );
            let settings = read_settings(&global_settings_path());
            assert_eq!(
                settings["modelRouter"]["subagents"]["Verification"]["model"],
                "Verifier/Model-X"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    fn commit_with(
        roles: Vec<(String, SmartSettingsUpdate)>,
        subagents: Vec<(String, SmartSettingsUpdate)>,
    ) -> SmartSettingsCommit {
        SmartSettingsCommit {
            enabled: true,
            allow_cross_provider_diversity: true,
            feedback_informed_auto: false,
            roles,
            subagents,
        }
    }

    /// Enter on the dashboard is the only way a user's edits reach disk, so the
    /// global toggles must survive the commit into the next snapshot.
    #[test]
    fn apply_smart_settings_commit_persists_global_toggles() {
        let config_home = temp_config_home("commit-toggles");
        with_config_home(&config_home, || {
            apply_smart_settings_commit(&commit_with(Vec::new(), Vec::new())).expect("apply");

            let snapshot = read_global_smart_settings().expect("snapshot");
            assert_eq!(
                (
                    snapshot.enabled,
                    snapshot.allow_cross_provider_diversity,
                    snapshot.feedback_informed_auto
                ),
                (true, true, false),
                "committed toggles did not round-trip: {snapshot:?}"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    /// Both target kinds share one commit but different settings roots, and the
    /// modal's `SmartSettingsUpdate` has to translate into the stored
    /// `SmartRoleUpdate` shape — pin the pin and the family lock together.
    #[test]
    fn apply_smart_settings_commit_persists_role_and_subagent_updates() {
        let config_home = temp_config_home("commit-targets");
        with_config_home(&config_home, || {
            let commit = commit_with(
                vec![(
                    "coding".to_string(),
                    SmartSettingsUpdate::ExactPin {
                        model: "Coder/Model-9".to_string(),
                    },
                )],
                vec![(
                    "verification".to_string(),
                    SmartSettingsUpdate::FamilyLock {
                        provider: "anthropic".to_string(),
                        family: "claude".to_string(),
                        class: "balanced".to_string(),
                        freshness: SmartSettingsFreshness::LatestStable,
                    },
                )],
            );

            apply_smart_settings_commit(&commit).expect("apply");

            let snapshot = read_global_smart_settings().expect("snapshot");
            assert_eq!(
                (
                    snapshot.roles.get("coding"),
                    snapshot.subagents.get("Verification")
                ),
                (
                    Some(&SmartRoleUpdate::ExactPin {
                        model: "Coder/Model-9".to_string()
                    }),
                    Some(&SmartRoleUpdate::FamilyLock {
                        provider: "anthropic".to_string(),
                        family: "claude".to_string(),
                        class: "balanced".to_string(),
                        freshness: SmartFreshness::LatestStable,
                    })
                ),
                "committed targets did not round-trip: {snapshot:?}"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    /// The confirmation line is the only feedback the user gets that a save
    /// landed, and `Auto` means "no override" — counting it would report edits
    /// the user did not make.
    #[test]
    fn apply_smart_settings_commit_counts_only_non_auto_targets_as_overrides() {
        let config_home = temp_config_home("commit-override-count");
        with_config_home(&config_home, || {
            let commit = commit_with(
                vec![
                    ("coding".to_string(), SmartSettingsUpdate::Auto),
                    (
                        "verifier".to_string(),
                        SmartSettingsUpdate::ExactPin {
                            model: "Verifier/Model-X".to_string(),
                        },
                    ),
                ],
                vec![("verification".to_string(), SmartSettingsUpdate::Auto)],
            );

            let message = apply_smart_settings_commit(&commit).expect("apply");

            assert!(
                message.contains("1 override(s)"),
                "only the pinned role counts as an override: {message}"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn auto_clears_all_parse_equivalent_subagent_aliases() {
        let config_home = temp_config_home("subagent-alias-clear");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(
                global_settings_path(),
                r#"{"modelRouter":{"subagents":{"verification":{"mode":"pinned","model":"A"},"Verification":{"mode":"pinned","model":"B"}}}}"#,
            )
            .expect("seed settings");
            write_global_smart_subagent("Verification", &SmartRoleUpdate::Auto)
                .expect("clear aliases");
            // Every parse-equivalent alias is masked back to Auto: no spelling
            // resolves to an override. Auto now writes deletion tombstones
            // rather than physically removing keys, so a same-profile override
            // in a lower canonical root cannot resurrect; readers ignore the
            // tombstones exactly as they ignore absent keys.
            let snapshot = read_global_smart_settings().expect("snapshot");
            assert!(
                !snapshot.subagents.contains_key("Verification")
                    && !snapshot.subagents.contains_key("verification"),
                "aliases still resolve to an override: {:?}",
                snapshot.subagents
            );
            let settings = read_settings(&global_settings_path());
            let subagents = settings["modelRouter"]["subagents"]
                .as_object()
                .expect("subagents object");
            for (alias, value) in subagents {
                assert_eq!(
                    value["mode"], "deleted",
                    "alias {alias} must be a tombstone, got {value:?}"
                );
            }
        });
        let _ = fs::remove_dir_all(config_home);
    }




    #[test]
    fn status_is_read_only_and_reports_missing_pins() {
        let config_home = temp_config_home("status-readonly-missing");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(
                global_settings_path(),
                r#"{"smart":{"enabled":true},"modelRouter":{"subagents":{"Verification":{"mode":"pinned","model":"missing-model"}}}}"#,
            )
            .expect("seed settings");
            let before = fs::read_to_string(global_settings_path()).expect("before");
            let status = render_smart_status("unlisted-current-model", None).expect("status");
            let after = fs::read_to_string(global_settings_path()).expect("after");
            assert_eq!(before, after, "status must be read-only");
            assert!(status.contains("Saved Overrides"));
            assert!(status.contains("! Missing pin: missing-model"));
            assert!(status.contains("Execution priority:"));
        });
        let _ = fs::remove_dir_all(config_home);
    }



    #[test]
    fn invalid_json_is_preserved_on_write_error() {
        let config_home = temp_config_home("invalid-json");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(global_settings_path(), "{not valid json").expect("write invalid");
            let error = write_global_smart_enabled(true).expect_err("invalid json fails");
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert_eq!(
                fs::read_to_string(global_settings_path()).expect("read invalid"),
                "{not valid json"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_policy_settings_are_persisted_and_reported() {
        let config_home = temp_config_home("smart-policy-settings");
        with_config_home(&config_home, || {
            write_global_smart_allow_cross_provider_diversity(true).expect("write diversity");
            write_global_smart_feedback_informed_auto(true).expect("write feedback");

            let snapshot = read_global_smart_settings().expect("snapshot");
            assert!(snapshot.allow_cross_provider_diversity);
            assert!(snapshot.feedback_informed_auto);

            let status = render_smart_status("unlisted-current-model", None).expect("status");
            assert!(status.contains("Cross-provider diversity: allowed"));
            assert!(status.contains("Feedback-informed auto: on (bounded)"));

            let settings = read_settings(&global_settings_path());
            assert_eq!(settings["smart"]["allowCrossProviderDiversity"], true);
            assert_eq!(settings["smart"]["feedbackInformedAuto"], true);
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_status_reports_auto_classifier_without_enabling_provider_calls() {
        let config_home = temp_config_home("smart-auto-classifier-status");
        with_config_home(&config_home, || {
            fs::create_dir_all(config_home.as_path()).expect("create config home");
            fs::write(
                global_settings_path(),
                r#"{"smart":{"enabled":true,"autoClassifier":"assisted"}}"#,
            )
            .expect("seed settings");

            let snapshot = read_global_smart_settings().expect("snapshot");
            assert_eq!(snapshot.auto_classifier, RouteAutoClassifierMode::Assisted);

            let status = render_smart_status("unlisted-current-model", None).expect("status");
            assert!(status.contains("Auto classifier: assisted (provider-free deterministic)"));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_doctor_renders_aggregate_route_outcomes() {
        let summary = runtime::RouteOutcomeSummary {
            total: 2,
            completed: 1,
            failed: 1,
            stopped: 0,
            still_running: 0,
            output_tokens: 99,
            by_route: vec![runtime::RouteOutcomeBucket {
                route_key: "subagent:Verification".to_string(),
                target_kind: "subagent".to_string(),
                target: "Verification".to_string(),
                selected_model: "gpt-5.5".to_string(),
                total: 2,
                completed: 1,
                failed: 1,
                stopped: 0,
                output_tokens: 99,
                provider_errors: [("rateLimit".to_string(), 1)].into_iter().collect(),
            }],
        };

        let rendered = render_smart_doctor_summary(
            Path::new("/tmp/zo/.zo/smart-router/route-outcomes.jsonl"),
            &summary,
        );

        assert!(rendered.contains("Smart Router Doctor"));
        assert!(rendered.contains("Recorded outcomes: 2"));
        assert!(rendered.contains("subagent:Verification via gpt-5.5"));
        assert!(rendered.contains("provider errors rateLimit:1"));
        assert!(rendered.contains("feedback-informed auto is enabled in /smart"));
    }

    fn bucket(route_key: &str, model: &str, completed: usize, failed: usize, stopped: usize) -> runtime::RouteOutcomeBucket {
        let (kind, target) = route_key.split_once(':').unwrap();
        runtime::RouteOutcomeBucket {
            route_key: route_key.to_string(),
            target_kind: kind.to_string(),
            target: target.to_string(),
            selected_model: model.to_string(),
            total: completed + failed + stopped,
            completed,
            failed,
            stopped,
            output_tokens: 0,
            provider_errors: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn observed_routes_aggregate_per_target_and_exclude_cancels() {
        let summary = runtime::RouteOutcomeSummary {
            total: 0,
            completed: 0,
            failed: 0,
            stopped: 0,
            still_running: 0,
            output_tokens: 0,
            by_route: vec![
                // Same target ran on two models → aggregate, hide the model name.
                bucket("subagent:Verification", "model-a", 8, 1, 2),
                bucket("subagent:Verification", "model-b", 0, 3, 0),
                // Single-model target → name the model; the 2 cancels are excluded
                // from the decisive ratio.
                bucket("subagent:debugger", "solo-model", 5, 0, 2),
            ],
        };

        let observed = observed_routes_from_summary(&summary);

        let verification = observed
            .iter()
            .find(|route| route.kind == SmartSettingsTargetKind::Subagent && route.key == "Verification")
            .expect("verification observed");
        assert_eq!(verification.completed, 8); // 8 + 0
        assert_eq!(verification.decisive, 12); // (8+1) + (0+3); cancels excluded
        assert_eq!(verification.model, None, "two models ran → no single model named");

        let debugger = observed
            .iter()
            .find(|route| route.key == "debugger")
            .expect("debugger observed");
        assert_eq!(debugger.completed, 5);
        assert_eq!(debugger.decisive, 5, "2 user-cancels excluded from decisive runs");
        assert_eq!(debugger.model.as_deref(), Some("solo-model"));
    }

    // ---- P7: doctor section unit tests ----

    #[test]
    fn smart_doctor_verdict_section_reports_zero_and_counts() {
        let empty: Vec<runtime::RouteOutcomeRecord> = Vec::new();
        let rendered = smart_doctor_verdict_section(&empty);
        assert!(rendered.contains("0 verdict-signal records"), "{rendered}");

        let records = vec![
            runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                .with_signal("verdict"),
            runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                .with_signal("verdict"),
            runtime::RouteOutcomeRecord::new("subagent", "Explore", "gpt-5.5", "completed"),
        ];
        let rendered = smart_doctor_verdict_section(&records);
        assert!(
            rendered.contains("2 verdict-signal record(s) across 1 route_key bucket(s)"),
            "{rendered}"
        );
        assert!(rendered.contains("subagent:code-reviewer: 2"), "{rendered}");
    }

    #[test]
    fn smart_doctor_verify_pair_section_groups_pairs_and_reports_pass_rate() {
        // No paired verdicts at all ⇒ the section is omitted (None), not an
        // empty table.
        let bare = vec![
            runtime::RouteOutcomeRecord::new("main", "turn", "claude-opus-4-8", "completed")
                .with_signal("verdict"),
            runtime::RouteOutcomeRecord::new("subagent", "Explore", "gpt-5.6-sol", "completed"),
        ];
        assert!(
            smart_doctor_verify_pair_section(&bare).is_none(),
            "verdicts without a verifier_model must not render a pair section"
        );

        // Two passes + one fail for the same (impl, verifier) pair, plus a
        // second distinct pair, plus a user-cancelled `stopped` that must not
        // count toward the rate.
        let records = vec![
            runtime::RouteOutcomeRecord::new("main", "turn", "claude-opus-4-8", "completed")
                .with_signal("verdict")
                .with_verifier_model(Some("gpt-5.6-sol".to_string())),
            runtime::RouteOutcomeRecord::new("main", "turn", "claude-opus-4-8", "completed")
                .with_signal("verdict")
                .with_verifier_model(Some("gpt-5.6-sol".to_string())),
            runtime::RouteOutcomeRecord::new("main", "turn", "claude-opus-4-8", "failed")
                .with_signal("verdict")
                .with_verifier_model(Some("gpt-5.6-sol".to_string())),
            runtime::RouteOutcomeRecord::new("main", "turn", "gpt-5.6-sol", "completed")
                .with_signal("verdict")
                .with_verifier_model(Some("claude-opus-4-8".to_string())),
            // stopped: excluded from the decisive denominator.
            runtime::RouteOutcomeRecord::new("main", "turn", "gpt-5.6-sol", "stopped")
                .with_signal("verdict")
                .with_verifier_model(Some("claude-opus-4-8".to_string())),
        ];
        let rendered =
            smart_doctor_verify_pair_section(&records).expect("paired verdicts must render");
        assert!(rendered.contains("Verify pair attribution (P1)"), "{rendered}");
        // 3 decisive for the first pair + 1 for the second = 4 across 2 pairs.
        assert!(
            rendered.contains("4 paired verdict sample(s) across 2 (implementation → verifier) pair(s)"),
            "{rendered}"
        );
        assert!(
            rendered.contains("claude-opus-4-8 → gpt-5.6-sol: 3 sample(s), 2/3 pass (66%)"),
            "{rendered}"
        );
        assert!(
            rendered.contains("gpt-5.6-sol → claude-opus-4-8: 1 sample(s), 1/1 pass (100%)"),
            "{rendered}"
        );
    }

    #[test]
    fn smart_doctor_canonical_merge_section_groups_raw_fragments() {
        let records = vec![
            runtime::RouteOutcomeRecord::new("subagent", "Plan", "claude-opus-4-8", "completed"),
            runtime::RouteOutcomeRecord::new("subagent", "Plan", "claude-opus-4.8", "completed"),
            runtime::RouteOutcomeRecord::new("subagent", "Explore", "totally-unique-test-model-xyz", "completed"),
        ];
        let rendered = smart_doctor_canonical_merge_section(&records);
        assert!(
            rendered.contains("claude-opus-4-8") && rendered.contains("claude-opus-4.8"),
            "merged raw fragments both listed: {rendered}"
        );
        assert!(
            !rendered.contains("totally-unique-test-model-xyz \u{2190}"),
            "a singleton raw id must not appear as a merge group: {rendered}"
        );
    }

    #[test]
    fn smart_doctor_canonical_merge_section_reports_none_when_no_merges() {
        let records = vec![
            runtime::RouteOutcomeRecord::new("subagent", "Explore", "unique-model-a-xyz", "completed"),
            runtime::RouteOutcomeRecord::new("subagent", "Plan", "unique-model-b-xyz", "completed"),
        ];
        let rendered = smart_doctor_canonical_merge_section(&records);
        assert!(rendered.contains("No raw id fragments merge"), "{rendered}");
    }

    #[test]
    fn smart_doctor_exploration_section_reports_on_off_and_eligibility() {
        let snapshot = SmartSettingsSnapshot {
            exploration: true,
            exploration_cadence: 5,
            ..SmartSettingsSnapshot::default()
        };
        let summary = runtime::RouteOutcomeSummary {
            total: 9,
            by_route: vec![
                bucket("subagent:code-reviewer", "gpt-5.5-fast", 8, 0, 0),
                bucket("subagent:Explore", "gpt-5.6-sol", 1, 0, 0),
            ],
            ..runtime::RouteOutcomeSummary::default()
        };
        let rendered = smart_doctor_exploration_section(Some(&snapshot), &summary);
        assert!(rendered.contains("Exploration: on"), "{rendered}");
        assert!(rendered.contains("cadence every 5"), "{rendered}");
        assert!(
            rendered.contains("subagent:code-reviewer: incumbent gpt-5.5-fast at 8 decisive — exploration-eligible"),
            "{rendered}"
        );
        assert!(
            rendered.contains("subagent:Explore: incumbent gpt-5.6-sol at 1 decisive — not yet eligible"),
            "{rendered}"
        );
    }

    #[test]
    fn smart_doctor_exploration_section_handles_unreadable_settings() {
        let summary = runtime::RouteOutcomeSummary::default();
        let rendered = smart_doctor_exploration_section(None, &summary);
        assert!(rendered.contains("Settings unreadable"), "{rendered}");
    }

    #[test]
    fn smart_doctor_pin_awareness_section_warns_when_dominated() {
        let config_home = temp_config_home("pin-awareness-dominated");
        with_config_home(&config_home, || {
            let mut records: Vec<runtime::RouteOutcomeRecord> = (0..3)
                .map(|_| {
                    runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                        .with_route_source(Some("pin".to_string()))
                })
                .collect();
            records.push(
                runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                    .with_route_source(Some("auto".to_string())),
            );
            let rendered = smart_doctor_pin_awareness_section(&records);
            assert!(rendered.contains("DOMINATED by pinned routes"), "{rendered}");
            assert!(
                rendered.contains("subagent:code-reviewer: 3/4 records (75%) are pinned"),
                "{rendered}"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_doctor_pin_awareness_section_quiet_when_not_dominated() {
        let config_home = temp_config_home("pin-awareness-quiet");
        with_config_home(&config_home, || {
            let records = vec![
                runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                    .with_route_source(Some("auto".to_string())),
                runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                    .with_route_source(Some("pin".to_string())),
                runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                    .with_route_source(Some("auto".to_string())),
            ];
            let rendered = smart_doctor_pin_awareness_section(&records);
            assert!(!rendered.contains("DOMINATED"), "{rendered}");
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn scan_learned_shadow_stamps_reads_manifest_route_reason() {
        // `ZO_AGENT_STORE` is process-global, and this test points it at a store
        // containing a STAMPED manifest. Without the crate-wide lock the sibling
        // `..._reports_empty_state` test can observe that store and fail its
        // "no stamps found" assertion — a real intermittent red (seen under a
        // loaded `--workspace` run, invisible when the target runs alone).
        let _guard = crate::test_env_lock();
        let dir = std::env::temp_dir().join(format!(
            "zo-smart-doctor-shadow-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("temp dir");
        fs::write(
            dir.join("agent-1.json"),
            r#"{"agentId":"agent-1","name":"reviewer","routeReason":"Coding·Medium — auto pick · learned-shadow-differs:gpt-5.6-sol"}"#,
        )
        .expect("write stamped manifest");
        fs::write(
            dir.join("agent-2.json"),
            r#"{"agentId":"agent-2","name":"explorer","routeReason":"Coding·Medium — auto pick"}"#,
        )
        .expect("write unstamped manifest");

        let prior = std::env::var("ZO_AGENT_STORE").ok();
        std::env::set_var("ZO_AGENT_STORE", &dir);
        let hits = scan_learned_shadow_stamps();
        match prior {
            Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
            None => std::env::remove_var("ZO_AGENT_STORE"),
        }
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(hits.len(), 1, "only the stamped manifest counts: {hits:?}");
        assert_eq!(hits[0].0, "agent-1");
        assert_eq!(hits[0].1, "gpt-5.6-sol");
    }

    #[test]
    fn smart_doctor_learned_shadow_section_renders_mode_and_entries() {
        let mut records = Vec::new();
        for _ in 0..5 {
            records.push(
                runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.6-sol", "completed")
                    .with_role(Some("coding".to_string())),
            );
        }
        // The peer pool must clear the same >=4.0 weighted floor as the
        // model's own history (a near-empty peer pool yields no entry at
        // all), so give the rival a real run of samples too.
        for _ in 0..4 {
            records.push(
                runtime::RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.5-fast", "completed")
                    .with_role(Some("coding".to_string())),
            );
        }
        let snapshot = SmartSettingsSnapshot {
            learned_specialty: SmartLearnedSpecialtyMode::Shadow,
            ..SmartSettingsSnapshot::default()
        };
        let rendered = smart_doctor_learned_shadow_section(&records, Some(&snapshot));
        assert!(rendered.contains("Mode: shadow"), "{rendered}");
        assert!(rendered.contains("Learned entries"), "{rendered}");
        assert!(rendered.contains("Coding: gpt-5.6-sol"), "{rendered}");
    }

    #[test]
    fn smart_doctor_learned_shadow_section_reports_empty_state() {
        // Same lock as the writer above: this asserts the ABSENCE of stamps, so it
        // must not run while a sibling has `ZO_AGENT_STORE` pointed at a stamped
        // fixture store. The store is then pinned at an EMPTY temp dir rather than
        // trusting the ambient one, so the assertion cannot depend on which agent
        // manifests this machine happens to have written.
        let _guard = crate::test_env_lock();
        let empty_store = std::env::temp_dir().join(format!(
            "zo-smart-doctor-empty-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&empty_store).expect("temp dir");
        let prior = std::env::var("ZO_AGENT_STORE").ok();
        std::env::set_var("ZO_AGENT_STORE", &empty_store);
        let rendered = smart_doctor_learned_shadow_section(&[], None);
        match prior {
            Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
            None => std::env::remove_var("ZO_AGENT_STORE"),
        }
        let _ = fs::remove_dir_all(&empty_store);
        assert!(rendered.contains("Mode: unknown"), "{rendered}");
        assert!(rendered.contains("No `learned-shadow-differs"), "{rendered}");
        assert!(rendered.contains("No learned-specialty entries yet"), "{rendered}");
    }

    #[test]
    fn smart_doctor_provenance_section_lists_models_with_provenance_labels() {
        let rendered = smart_doctor_provenance_section("main");
        // Best-effort against whatever inventory this test environment resolves
        // (no credentials assumed) — just assert the section renders its header
        // and, if any model line exists at all, that it carries one of the three
        // documented provenance labels rather than a raw enum debug string.
        assert!(rendered.contains("Model tier provenance"), "{rendered}");
        if rendered.lines().count() > 3 {
            assert!(
                rendered.contains("fallback (")
                    || rendered.contains("cold-start-prior (")
                    || rendered.contains("learned ("),
                "{rendered}"
            );
        }
    }

    /// `smart.enabled` defaults ON (user decision, 2026-07-10): a missing key
    /// must read as enabled, while an explicit `false` still wins. Pinned here
    /// so a regression back to default-off cannot land silently.
    #[test]
    fn smart_enabled_defaults_on_and_explicit_false_wins() {
        let absent = snapshot_from_root(&Value::Object(JsonMap::new()));
        assert!(absent.enabled, "missing smart.enabled must default ON");

        let empty_smart = snapshot_from_root(&serde_json::json!({"smart": {}}));
        assert!(empty_smart.enabled, "smart object without enabled key must default ON");

        let explicit_off = snapshot_from_root(&serde_json::json!({"smart": {"enabled": false}}));
        assert!(!explicit_off.enabled, "explicit smart.enabled=false must win over the default");
    }

    /// Seed one managed session file under `config_home` so the banner's
    /// prior-use gate reads this as an EXISTING user's home.
    fn seed_prior_session(config_home: &std::path::Path) {
        let sessions = config_home.join("projects").join("proj-a").join("sessions");
        fs::create_dir_all(&sessions).expect("create sessions dir");
        fs::write(sessions.join("session-1.jsonl"), "{}\n").expect("seed session");
    }

    #[test]
    fn smart_default_banner_never_fires_for_a_brand_new_user() {
        let config_home = temp_config_home("default-banner-new-user");
        with_config_home(&config_home, || {
            // Fresh install: no settings.json AND no prior sessions. The
            // banner announces a changed default, which a first boot has no
            // expectation about — it must stay silent AND burn the marker so
            // it cannot fire later once sessions exist.
            assert!(
                smart_default_banner_notice().is_none(),
                "a brand-new user must never see the changed-default banner"
            );
            assert!(
                config_home
                    .join("notices")
                    .join("smart-default-banner-shown")
                    .exists(),
                "the silent first boot must still burn the marker"
            );
            seed_prior_session(&config_home);
            assert!(
                smart_default_banner_notice().is_none(),
                "the burned marker must hold once the user has sessions"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_default_banner_shows_once_then_marker_suppresses_it() {
        let config_home = temp_config_home("default-banner-once");
        with_config_home(&config_home, || {
            // An EXISTING user (has prior sessions) on the pure default path.
            seed_prior_session(&config_home);
            let banner = smart_default_banner_notice().expect("first boot must banner");
            assert!(banner.contains("/smart off"), "{banner}");
            assert!(banner.contains("/smart doctor"), "{banner}");
            assert!(
                config_home
                    .join("notices")
                    .join("smart-default-banner-shown")
                    .exists(),
                "showing the banner must persist the per-user marker"
            );
            assert!(
                smart_default_banner_notice().is_none(),
                "the marker must suppress every later boot"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_default_banner_suppressed_by_explicit_enabled_key() {
        // An explicit key — true OR false — means the user already decided;
        // no banner, and the one-shot marker must stay unburned so a later
        // return to the default path still gets the announcement.
        for explicit in [true, false] {
            let config_home = temp_config_home("default-banner-explicit");
            with_config_home(&config_home, || {
                fs::create_dir_all(&config_home).expect("create config home");
                fs::write(
                    global_settings_path(),
                    serde_json::json!({"smart": {"enabled": explicit}}).to_string(),
                )
                .expect("seed settings");
                assert!(
                    smart_default_banner_notice().is_none(),
                    "explicit smart.enabled={explicit} must suppress the banner"
                );
                assert!(
                    !config_home
                        .join("notices")
                        .join("smart-default-banner-shown")
                        .exists(),
                    "a suppressed banner must not burn the marker"
                );
            });
            let _ = fs::remove_dir_all(config_home);
        }
    }

    /// The documented dual-reader drift point (smart-auto routing plan, P7):
    /// the tools crate's live-routing reader (`read_smart_runtime_settings`)
    /// and this crate's dashboard reader (`snapshot_from_root`) parse the same
    /// `settings.json` independently and must fall back to the SAME defaults
    /// when a key is absent, or the dashboard preview and live routing
    /// silently disagree about what "default" means.
    #[test]
    fn cli_snapshot_defaults_match_tools_crate_runtime_defaults() {
        let defaults = tools::smart_setting_defaults();
        let snapshot = snapshot_from_root(&Value::Object(JsonMap::new()));
        assert_eq!(snapshot.enabled, defaults.enabled);
        assert_eq!(
            snapshot.allow_cross_provider_diversity,
            defaults.allow_cross_provider_diversity
        );
        assert_eq!(snapshot.verify_cross_provider, defaults.verify_cross_provider);
        assert_eq!(snapshot.quota_fallback, defaults.quota_fallback);
        assert_eq!(snapshot.deep_tier_models, defaults.deep_tier_models);
        assert_eq!(snapshot.feedback_informed_auto, defaults.feedback_informed_auto);
        assert_eq!(snapshot.fallback_candidate_limit, defaults.fallback_candidate_limit);
        assert_eq!(snapshot.exploration, defaults.exploration);
        assert_eq!(snapshot.exploration_cadence, defaults.exploration_cadence);
        assert_eq!(
            snapshot.learned_specialty == SmartLearnedSpecialtyMode::Shadow,
            defaults.learned_specialty_defaults_to_shadow
        );
        assert_eq!(
            snapshot.headroom_penalty_threshold,
            defaults.headroom_penalty_threshold
        );
        assert_eq!(snapshot.policy, defaults.policy);
        assert_eq!(snapshot.exec_swap, defaults.exec_swap);
        assert_eq!(defaults.exec_swap, tools::SmartExecSwap::Never);
        assert_eq!(
            defaults.policy,
            runtime::SmartPolicy::Architect,
            "the live smart.policy default is the architect contract"
        );
    }

    /// The same dual-reader drift point for a CONFIGURED pool. Both readers
    /// parse `smart.deepTierModels` independently, so they must resolve an
    /// entry identically. `resolve_model_alias` is gated on the target provider
    /// being connected, so using it here left `openai-latest` unresolved while
    /// the live router (catalog-first) resolved it to the OpenAI head — the
    /// same configured pool naming different models on different machines.
    #[test]
    fn cli_snapshot_resolves_a_configured_pool_like_the_tools_reader() {
        let snapshot = snapshot_from_root(&serde_json::json!({
            "smart": {"deepTierModels": [api::OPENAI_LATEST_MODEL_ALIAS]}
        }));
        assert_eq!(
            snapshot.deep_tier_models,
            vec![api::resolve_catalog_alias(api::OPENAI_LATEST_MODEL_ALIAS)],
            "the dashboard reader must resolve a configured pool exactly like \
             the tools crate's `deep_tier_models_from_smart`"
        );
    }

    /// `smart.policy` parses `classic` as the opt-out and defaults everything
    /// else (absent, unrecognized) to the architect contract — lockstep with
    /// the tools crate reader (both delegate to `SmartPolicy::from_settings_value`).
    #[test]
    fn smart_policy_parses_classic_opt_out_and_defaults_architect() {
        let absent = snapshot_from_root(&Value::Object(JsonMap::new()));
        assert_eq!(absent.policy, runtime::SmartPolicy::Architect);
        let classic = snapshot_from_root(&serde_json::json!({"smart": {"policy": "classic"}}));
        assert_eq!(classic.policy, runtime::SmartPolicy::Classic);
        let bogus = snapshot_from_root(&serde_json::json!({"smart": {"policy": "bogus"}}));
        assert_eq!(bogus.policy, runtime::SmartPolicy::Architect);
    }

    #[test]
    fn exec_swap_parses_the_three_modes_and_defaults_never() {
        let absent = snapshot_from_root(&Value::Object(JsonMap::new()));
        assert_eq!(absent.exec_swap, tools::SmartExecSwap::Never);
        for (value, expected) in [
            ("easy", tools::SmartExecSwap::Easy),
            ("always", tools::SmartExecSwap::Always),
            ("never", tools::SmartExecSwap::Never),
            ("bogus", tools::SmartExecSwap::Never),
        ] {
            let snapshot =
                snapshot_from_root(&serde_json::json!({"smart": {"execSwap": value}}));
            assert_eq!(snapshot.exec_swap, expected, "{value}");
        }
    }

    /// P0: the deep-gate VERIFY leg's cross-provider switch defaults ON and is
    /// decoupled from the global worker-diversity flag. Turning global
    /// diversity off must leave the verify leg cross-model (the `Verifier`
    /// route request still asks for diversity); turning `verifyCrossProvider`
    /// off must drop the verify leg's diversity even while global diversity is
    /// on. `deep_verify_allow_cross_provider` is the single seam that decision
    /// flows through, so asserting on it proves the decoupling without needing
    /// a live multi-provider inventory.
    #[test]
    fn verify_cross_provider_defaults_on_and_decouples_from_global_diversity() {
        // Absent key ⇒ ON (lockstep with the tools crate reader).
        let absent = snapshot_from_root(&Value::Object(JsonMap::new()));
        assert!(absent.verify_cross_provider, "missing verifyCrossProvider must default ON");
        assert!(deep_verify_allow_cross_provider(&absent));

        // Global diversity OFF, verify-cross unspecified: verify stays cross-model.
        let global_off = snapshot_from_root(&serde_json::json!({
            "smart": {"allowCrossProviderDiversity": false}
        }));
        assert!(!global_off.allow_cross_provider_diversity);
        assert!(global_off.verify_cross_provider, "verify-cross must not follow the global flag off");
        assert!(
            deep_verify_allow_cross_provider(&global_off),
            "verify leg must route cross-model even when global worker-diversity is off"
        );

        // Global diversity ON, verify-cross OFF: verify leg prefers native.
        let verify_off = snapshot_from_root(&serde_json::json!({
            "smart": {"allowCrossProviderDiversity": true, "verifyCrossProvider": false}
        }));
        assert!(verify_off.allow_cross_provider_diversity);
        assert!(!verify_off.verify_cross_provider);
        assert!(
            !deep_verify_allow_cross_provider(&verify_off),
            "verifyCrossProvider=false must drop verify-leg diversity even when global is on"
        );
    }

    #[test]
    fn execute_text_command_toggles_verify_cross_and_status_reports_it() {
        let config_home = temp_config_home("verify-cross-toggle");
        with_config_home(&config_home, || {
            let res = execute_smart_text_command("main", Some("verify-cross off")).unwrap();
            assert!(res.contains("disabled"), "{res}");
            assert!(!read_global_smart_settings().unwrap().verify_cross_provider);
            let status = render_smart_status("main", None).expect("status");
            assert!(
                status.contains("Verify cross-provider: off (native-preferred)"),
                "{status}"
            );

            let res = execute_smart_text_command("main", Some("verify-cross on")).unwrap();
            assert!(res.contains("enabled"), "{res}");
            assert!(read_global_smart_settings().unwrap().verify_cross_provider);
            let status = render_smart_status("main", None).expect("status");
            assert!(status.contains("Verify cross-provider: on (default)"), "{status}");

            let err = execute_smart_text_command("main", Some("verify-cross bogus")).unwrap_err();
            assert!(err.contains("Unknown"), "{err}");
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn execute_text_command_sets_smart_policy_and_status_reports_it() {
        let config_home = temp_config_home("smart-policy-toggle");
        with_config_home(&config_home, || {
            let res = execute_smart_text_command("main", Some("policy classic")).unwrap();
            assert!(res.contains("classic"), "{res}");
            assert_eq!(
                read_global_smart_settings().unwrap().policy,
                runtime::SmartPolicy::Classic
            );
            let status = render_smart_status("main", None).expect("status");
            assert!(status.contains("Policy: classic"), "{status}");

            let res = execute_smart_text_command("main", Some("policy architect")).unwrap();
            assert!(res.contains("architect"), "{res}");
            assert_eq!(
                read_global_smart_settings().unwrap().policy,
                runtime::SmartPolicy::Architect
            );
            let status = render_smart_status("main", None).expect("status");
            assert!(status.contains("Policy: architect"), "{status}");

            let err = execute_smart_text_command("main", Some("policy bogus")).unwrap_err();
            assert!(err.contains("Unknown"), "{err}");
        });
        let _ = fs::remove_dir_all(config_home);
    }

    /// P3: `smart.quotaFallback` defaults ON and is parsed independently of the
    /// other cross-provider flags — the CLI reader mirrors the tools crate's.
    #[test]
    fn quota_fallback_defaults_on_and_parses_explicit_off() {
        let absent = snapshot_from_root(&Value::Object(JsonMap::new()));
        assert!(absent.quota_fallback, "missing quotaFallback must default ON");

        let off = snapshot_from_root(&serde_json::json!({"smart": {"quotaFallback": false}}));
        assert!(!off.quota_fallback, "explicit quotaFallback=false must be honored");

        // Independent of the deep-verify switch: one off, the other on.
        let mixed = snapshot_from_root(&serde_json::json!({
            "smart": {"quotaFallback": false, "verifyCrossProvider": true}
        }));
        assert!(!mixed.quota_fallback);
        assert!(mixed.verify_cross_provider);
    }

    #[test]
    fn execute_text_command_toggles_quota_fallback_and_status_reports_it() {
        let config_home = temp_config_home("quota-fallback-toggle");
        with_config_home(&config_home, || {
            let res = execute_smart_text_command("main", Some("quota-fallback off")).unwrap();
            assert!(res.contains("disabled"), "{res}");
            assert!(!read_global_smart_settings().unwrap().quota_fallback);
            let status = render_smart_status("main", None).expect("status");
            assert!(
                status.contains("Quota fallback: off (turn fails on quota exhaustion)"),
                "{status}"
            );

            let res = execute_smart_text_command("main", Some("quota-fallback on")).unwrap();
            assert!(res.contains("enabled"), "{res}");
            assert!(read_global_smart_settings().unwrap().quota_fallback);
            let status = render_smart_status("main", None).expect("status");
            assert!(status.contains("Quota fallback: on (default)"), "{status}");

            let err = execute_smart_text_command("main", Some("quota-fallback bogus")).unwrap_err();
            assert!(err.contains("Unknown"), "{err}");
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn execute_text_command_manages_new_p7_knobs() {
        let config_home = temp_config_home("p7-knobs");
        with_config_home(&config_home, || {
            let res = execute_smart_text_command("main", Some("explore off")).unwrap();
            assert!(res.contains("disabled"), "{res}");
            assert!(!read_global_smart_settings().unwrap().exploration);

            let res = execute_smart_text_command("main", Some("explore on")).unwrap();
            assert!(res.contains("enabled"), "{res}");
            assert!(read_global_smart_settings().unwrap().exploration);

            let res = execute_smart_text_command("main", Some("explore cadence 7")).unwrap();
            assert!(res.contains('7'), "{res}");
            assert_eq!(read_global_smart_settings().unwrap().exploration_cadence, 7);

            let err = execute_smart_text_command("main", Some("explore cadence 0")).unwrap_err();
            assert!(err.contains("Invalid cadence"), "{err}");

            let res = execute_smart_text_command("main", Some("learned on")).unwrap();
            assert!(res.contains("on"), "{res}");
            assert_eq!(
                read_global_smart_settings().unwrap().learned_specialty,
                SmartLearnedSpecialtyMode::On
            );

            let res = execute_smart_text_command("main", Some("learned off")).unwrap();
            assert!(res.contains("off"), "{res}");
            assert_eq!(
                read_global_smart_settings().unwrap().learned_specialty,
                SmartLearnedSpecialtyMode::Off
            );

            let err = execute_smart_text_command("main", Some("learned bogus")).unwrap_err();
            assert!(err.contains("Unknown"), "{err}");

            let res = execute_smart_text_command("main", Some("feedback off")).unwrap();
            assert!(res.contains("disabled"), "{res}");
            assert!(!read_global_smart_settings().unwrap().feedback_informed_auto);

            let res = execute_smart_text_command("main", Some("feedback on")).unwrap();
            assert!(res.contains("enabled"), "{res}");
            assert!(read_global_smart_settings().unwrap().feedback_informed_auto);

            let res = execute_smart_text_command("main", Some("diversity off")).unwrap();
            assert!(res.contains("disabled"), "{res}");
            assert!(!read_global_smart_settings().unwrap().allow_cross_provider_diversity);

            let res = execute_smart_text_command("main", Some("diversity on")).unwrap();
            assert!(res.contains("enabled"), "{res}");
            assert!(read_global_smart_settings().unwrap().allow_cross_provider_diversity);

            let res = execute_smart_text_command("main", Some("providers anthropic, openai")).unwrap();
            assert!(res.contains("anthropic"), "{res}");
            assert_eq!(
                read_global_smart_settings().unwrap().provider_allowlist,
                vec!["anthropic".to_string(), "openai".to_string()]
            );

            let res = execute_smart_text_command("main", Some("providers clear")).unwrap();
            assert!(res.contains("cleared"), "{res}");
            assert!(read_global_smart_settings().unwrap().provider_allowlist.is_empty());
        });
        let _ = fs::remove_dir_all(config_home);
    }
}
