use api::ProviderKind;
use crate::model_inventory::{connected_model_inventory, model_inventory_from_authorized_providers};

use super::*;

fn inventory() -> ModelInventory {
    ModelInventory::new(
        "claude-sonnet-main",
        vec![
            ModelDescriptor::new("claude-sonnet-main", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .class("sonnet")
                .capabilities([
                    ModelCapability::Default,
                    ModelCapability::Coding,
                    ModelCapability::Verification,
                ])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("gpt-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .class("strong")
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(2),
            ModelDescriptor::new("claude-opus-analysis", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .class("opus")
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Deep])
                .release_rank(3),
            ModelDescriptor::new("gpt-preview", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .class("strong")
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .status(ModelStatus::Preview)
                .release_rank(99),
        ],
    )
}

#[test]
fn authorized_provider_inventory_does_not_expose_full_catalog() {
    let inventory = model_inventory_from_authorized_providers("claude-sonnet-main", &[ProviderKind::Anthropic], &[]);
    assert!(inventory.models().iter().any(|model| model.provider() == "anthropic"));
    assert!(
        inventory.models().iter().all(|model| model.provider() == "anthropic" || model.id() == "claude-sonnet-main"),
        "unauthorized provider leaked into inventory: {:?}",
        inventory.models()
    );
    assert!(inventory.find("gpt-5.5-fast").is_none());
}

#[test]
fn custom_inventory_includes_only_declared_custom_models() {
    let custom = vec![("Local".to_string(), vec!["llama-3.3".to_string(), "qwen-coder".to_string()])];
    let inventory = model_inventory_from_authorized_providers("main", &[], &custom);
    assert!(inventory.find("llama-3.3").is_some());
    assert!(inventory.find("qwen-coder").is_some());
    assert!(inventory.find("deepseek-chat").is_none());
    assert!(inventory.find("gpt-5.5-fast").is_none());
}

#[test]
fn custom_provider_models_are_classified_for_smart_auto_routing() {
    let custom = vec![
        (
            "gme-litellm".to_string(),
            vec![
                "qwen3.6-35b-a3b".to_string(),
                "qwen3.6-35b-a3b-thinking".to_string(),
                "ornith-1.0-9b".to_string(),
            ],
        ),
        (
            "nvidia".to_string(),
            vec![
                "meta/llama-3.1-8b-instruct".to_string(),
                "z-ai/glm-5.2".to_string(),
            ],
        ),
    ];
    let inventory = model_inventory_from_authorized_providers("main", &[], &custom);

    let qwen = inventory.find("qwen3.6-35b-a3b").expect("qwen model");
    assert_eq!(qwen.family(), "qwen");
    assert!(qwen.has_capability(ModelCapability::Coding));
    assert!(qwen.has_capability(ModelCapability::Verification));
    assert!(qwen.has_capability(ModelCapability::Analysis));
    assert!(qwen.has_tier(ModelTier::Strong));

    let thinking = inventory
        .find("qwen3.6-35b-a3b-thinking")
        .expect("thinking model");
    assert_eq!(thinking.family(), "qwen");
    assert!(thinking.has_tier(ModelTier::Deep));

    let llama = inventory
        .find("meta/llama-3.1-8b-instruct")
        .expect("llama model");
    assert_eq!(llama.family(), "llama");
    assert!(llama.has_capability(ModelCapability::Fast));
    assert!(!llama.has_capability(ModelCapability::Coding));
    assert!(!llama.has_capability(ModelCapability::Verification));
    assert!(!llama.has_capability(ModelCapability::Analysis));
    assert!(llama.has_tier(ModelTier::Fast));
    assert!(!llama.has_tier(ModelTier::Strong));

    let glm = inventory.find("z-ai/glm-5.2").expect("glm model");
    assert_eq!(glm.family(), "glm");
    assert!(glm.has_capability(ModelCapability::Coding));
    assert!(glm.has_tier(ModelTier::Strong));

    let ornith = inventory.find("ornith-1.0-9b").expect("ornith model");
    assert_eq!(ornith.family(), "ornith");
    assert!(ornith.has_capability(ModelCapability::Fast));
    assert!(!ornith.has_capability(ModelCapability::Coding));
    assert!(!ornith.has_capability(ModelCapability::Verification));
    assert!(!ornith.has_capability(ModelCapability::Analysis));
    assert!(ornith.has_tier(ModelTier::Fast));
}

#[test]
fn small_custom_frontier_models_do_not_satisfy_specialist_fallbacks() {
    let custom = vec![(
        "nvidia".to_string(),
        vec![
            "meta/llama-3.1-8b-instruct".to_string(),
            "ornith-1.0-9b".to_string(),
        ],
    )];
    let inventory = model_inventory_from_authorized_providers("main-model", &[], &custom);

    for role in [
        RouteRole::Coding,
        RouteRole::Debugging,
        RouteRole::Verifier,
        RouteRole::Reviewer,
        RouteRole::Analysis,
        RouteRole::Research,
        RouteRole::Judge,
        RouteRole::Synthesizer,
        RouteRole::Writing,
        RouteRole::Design,
    ] {
        let decision = route_model(&RouteRequest::new(role, "main-model"), &inventory);
        assert_eq!(
            decision.resolved_model, "main-model",
            "small custom model should not satisfy specialist fallback for {role:?}",
        );
        assert_eq!(decision.source, RouteDecisionSource::MainModelFallback);
    }

    let fast = route_model(&RouteRequest::new(RouteRole::Fast, "main-model"), &inventory);
    assert_ne!(fast.resolved_model, "main-model", "small custom models still serve Fast routes");
}

#[test]
fn builtin_codex_spark_and_sonnet_are_classified_by_role_fit() {
    let inventory = model_inventory_from_authorized_providers(
        "main",
        &[ProviderKind::OpenAi, ProviderKind::Anthropic],
        &[],
    );

    let spark = inventory
        .find("gpt-5.3-codex-spark")
        .expect("codex spark");
    assert_eq!(spark.provider(), "openai");
    assert_eq!(spark.family(), "gpt");
    assert_eq!(spark.class_label(), Some("fast"));
    assert!(spark.has_capability(ModelCapability::Fast));
    assert!(spark.has_capability(ModelCapability::Coding));
    assert!(spark.has_tier(ModelTier::Fast));
    assert!(spark.has_tier(ModelTier::Balanced));
    assert!(!spark.has_tier(ModelTier::Strong));
    assert!(!spark.has_tier(ModelTier::Deep));

    let sonnet = inventory.find("claude-sonnet-5").expect("sonnet");
    assert_eq!(sonnet.provider(), "anthropic");
    assert_eq!(sonnet.family(), "claude");
    assert_eq!(sonnet.class_label(), Some("sonnet"));
    assert!(sonnet.has_capability(ModelCapability::Coding));
    assert!(sonnet.has_capability(ModelCapability::Analysis));
    assert!(sonnet.has_tier(ModelTier::Strong));
    assert!(!sonnet.has_tier(ModelTier::Fast));
}

#[test]
fn unknown_custom_models_need_size_signal_before_specialist_promotion() {
    let custom = vec![(
        "local".to_string(),
        vec![
            "local-large".to_string(),
            "local-9b".to_string(),
            "local-70b".to_string(),
        ],
    )];
    let inventory = model_inventory_from_authorized_providers("main", &[], &custom);

    let opaque = inventory.find("local-large").expect("opaque custom");
    assert_eq!(opaque.family(), "custom");
    assert!(opaque.has_capability(ModelCapability::Default));
    assert!(!opaque.has_capability(ModelCapability::Coding));
    assert!(opaque.has_tier(ModelTier::Balanced));
    assert!(!opaque.has_tier(ModelTier::Strong));

    let small = inventory.find("local-9b").expect("small custom");
    assert_eq!(small.family(), "custom");
    assert!(small.has_capability(ModelCapability::Fast));
    assert!(!small.has_capability(ModelCapability::Coding));
    assert!(small.has_tier(ModelTier::Fast));
    assert!(!small.has_tier(ModelTier::Strong));

    let large = inventory.find("local-70b").expect("large custom");
    assert_eq!(large.family(), "custom");
    assert!(large.has_capability(ModelCapability::Coding));
    assert!(large.has_capability(ModelCapability::Analysis));
    assert!(large.has_tier(ModelTier::Strong));
}

#[test]
fn custom_deepseek_models_keep_provider_family_and_capabilities() {
    let inventory = model_inventory_from_authorized_providers(
        "main",
        &[],
        &[(
            "DeepSeek".to_string(),
            vec!["DeepSeek-Chat".to_string(), "DeepSeek-Reasoner".to_string()],
        )],
    );

    let chat = inventory.find("DeepSeek-Chat").expect("deepseek chat");
    assert_eq!(chat.provider(), "DeepSeek");
    assert_eq!(chat.family(), "deepseek");
    assert_eq!(chat.class_label(), Some("chat"));

    let reasoner = inventory.find("DeepSeek-Reasoner").expect("deepseek reasoner");
    assert_eq!(reasoner.family(), "deepseek");
    assert_eq!(reasoner.class_label(), Some("reasoner"));

    let coding = route_model(&RouteRequest::new(RouteRole::Coding, "main"), &inventory);
    assert_ne!(coding.resolved_model, "main", "DeepSeek should be usable for coding routes");
    let analysis = route_model(&RouteRequest::new(RouteRole::Analysis, "main"), &inventory);
    assert_eq!(
        analysis.resolved_model, "DeepSeek-Reasoner",
        "mixed-case DeepSeek reasoner should keep analysis capability and deep tier"
    );
}

#[test]
fn default_model_does_not_authorize_its_builtin_provider_catalog() {
    let inventory = model_inventory_from_authorized_providers("gpt-5.5-fast", &[], &[]);
    let ids: Vec<&str> = inventory.models().iter().map(ModelDescriptor::id).collect();
    assert_eq!(ids, vec!["gpt-5.5-fast"]);
    let descriptor = inventory.find("gpt-5.5-fast").expect("fallback descriptor");
    assert_eq!(descriptor.provider(), "unknown");
    assert_eq!(descriptor.family(), "custom");
    assert_eq!(descriptor.source_value(), ModelSource::CurrentMainModel);
}

#[test]
fn connected_inventory_always_keeps_current_default_as_fallback() {
    let inventory = connected_model_inventory("unlisted-current-model");
    assert!(inventory.find("unlisted-current-model").is_some());
}

#[test]
fn dated_and_suffixed_main_model_ids_inherit_known_family() {
    // Task 4: a main-session model pinned to a dated/`@`/`[`-suffixed variant
    // of an already-classified inventory entry (e.g. `gpt-5.6-sol-2026-07-09`)
    // must inherit that entry's family/provider/tiers/effort-ceiling/context
    // window instead of degrading to the zero-capability `unknown`/`custom`
    // fallback — the asymmetry the effort predicates already handled (dated
    // ids) but the inventory fallback did not.
    let known_models = || {
        vec![ModelDescriptor::new("gpt-5.6-sol", "openai", "gpt")
            .source(ModelSource::EnabledBuiltinProvider)
            .class("strong")
            .capabilities([ModelCapability::Default, ModelCapability::Coding])
            .tiers([ModelTier::Deep, ModelTier::Strong])
            .release_rank(56)
            .effort_ceiling(EffortCeiling::Ultra)
            .context_window(258_000)]
    };

    for dated_id in ["gpt-5.6-sol-2026-07-09", "gpt-5.6-sol@openai", "gpt-5.6-sol[fast]"] {
        let inventory = ModelInventory::new(dated_id, known_models());
        let main = inventory.find(dated_id).unwrap_or_else(|| panic!("{dated_id} missing from inventory"));
        assert_eq!(main.provider(), "openai", "{dated_id}");
        assert_eq!(main.family(), "gpt", "{dated_id}");
        assert!(main.has_tier(ModelTier::Deep), "{dated_id} should inherit the Deep tier");
        assert_eq!(main.effort_ceiling_value(), EffortCeiling::Ultra, "{dated_id}");
        assert_eq!(main.context_window_value(), Some(258_000), "{dated_id}");
        assert_eq!(main.source_value(), ModelSource::CurrentMainModel, "{dated_id} is still the main model");
    }
}

#[test]
fn genuinely_unknown_main_model_id_still_degrades_to_unknown() {
    // The negative case: an id sharing no known family prefix with anything
    // in the inventory must still degrade exactly as before this phase.
    let inventory = ModelInventory::new(
        "totally-unrecognized-model-xyz",
        vec![ModelDescriptor::new("gpt-5.6-sol", "openai", "gpt")
            .source(ModelSource::EnabledBuiltinProvider)
            .tiers([ModelTier::Deep, ModelTier::Strong])
            .release_rank(56)],
    );
    let main = inventory.find("totally-unrecognized-model-xyz").expect("fallback descriptor present");
    assert_eq!(main.provider(), "unknown");
    assert_eq!(main.family(), "custom");
    assert!(!main.has_tier(ModelTier::Deep));
    assert!(!main.has_tier(ModelTier::Strong));
}

#[test]
fn route_auto_classifier_mode_is_provider_free_and_fail_closed() {
    assert_eq!(RouteAutoClassifierMode::from_settings_value(None), RouteAutoClassifierMode::Deterministic);
    assert_eq!(
        RouteAutoClassifierMode::from_settings_value(Some(&serde_json::json!(" surprise "))),
        RouteAutoClassifierMode::Deterministic,
    );
    assert_eq!(
        RouteAutoClassifierMode::from_settings_value(Some(&serde_json::json!(" off "))),
        RouteAutoClassifierMode::Off,
    );
    assert_eq!(
        RouteAutoClassifierMode::from_settings_value(Some(&serde_json::json!("assisted"))),
        RouteAutoClassifierMode::Assisted,
    );
    assert_eq!(
        RouteAutoClassifierMode::Assisted.audit_note(),
        "smart-auto-classifier:assisted-provider-free-deterministic",
    );
    assert_eq!(
        RouteAutoClassifierMode::Assisted.status_label(),
        "assisted (provider-free deterministic)",
    );
}

#[test]
fn subagent_profile_id_normalizes_builtin_and_custom_keys() {
    let verification = SubagentProfileId::parse("verification").expect("builtin");
    assert_eq!(verification.key(), "Verification");
    assert_eq!(verification.kind(), SubagentProfileKind::Builtin);

    let custom = SubagentProfileId::parse("Reviewer Agent").expect("custom");
    assert_eq!(custom.key(), "custom:reviewer-agent");
    assert_eq!(custom.kind(), SubagentProfileKind::Custom);
}

#[test]
fn builtin_subagent_profiles_have_route_roles() {
    assert_eq!(BuiltinSubagentProfile::Verification.route_role(), RouteRole::Verifier);
    assert_eq!(BuiltinSubagentProfile::CodeReviewer.route_role(), RouteRole::Reviewer);
    assert_eq!(BuiltinSubagentProfile::Plan.route_role(), RouteRole::Analysis);
}

#[test]
fn phase0_route_model_auto_selector_baseline_by_role() {
    let cases = [
        (RouteRole::Coding, "gpt-coder", RouteDecisionSource::AutoSelector, "auto role selector"),
        (
            RouteRole::Verifier,
            "claude-sonnet-main",
            RouteDecisionSource::AutoSelector,
            "auto role selector",
        ),
        (
            RouteRole::Reviewer,
            "claude-sonnet-main",
            RouteDecisionSource::AutoSelector,
            "auto role selector",
        ),
        (
            RouteRole::Research,
            "claude-opus-analysis",
            RouteDecisionSource::AutoSelector,
            "auto role selector",
        ),
        (
            RouteRole::Design,
            "claude-sonnet-main",
            RouteDecisionSource::MainModelFallback,
            "main model fallback",
        ),
    ];

    for (role, expected_model, expected_source, expected_reason) in cases {
        let decision = route_model(&RouteRequest::new(role, "claude-sonnet-main"), &inventory());
        assert_eq!(decision.resolved_model, expected_model, "unexpected model for {role:?}");
        assert_eq!(decision.source, expected_source, "unexpected source for {role:?}");
        assert_eq!(decision.reason, expected_reason, "unexpected reason for {role:?}");
    }
}


#[test]
fn explicit_model_wins_before_router_policy() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.explicit_model = Some("gpt-coder".to_string());
    request.override_rule = Some(RoleOverride::Pin("claude-opus-analysis".to_string()));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "gpt-coder");
    assert_eq!(decision.source, RouteDecisionSource::Explicit);
    assert_eq!(decision.audit.cross_provider, Some(true));
}

#[test]
fn explicit_model_outside_inventory_falls_back_to_main() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.explicit_model = Some("not-connected".to_string());
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "claude-sonnet-main");
    assert_eq!(decision.source, RouteDecisionSource::MainModelFallback);
}

#[test]
fn exact_pin_outside_inventory_never_routes_to_missing_model() {
    let mut request = RouteRequest::new(RouteRole::Verifier, "claude-sonnet-main");
    request.override_rule = Some(RoleOverride::Pin("Vendor/Model-X".to_string()));
    let decision = route_model(&request, &inventory());
    assert_ne!(decision.resolved_model, "Vendor/Model-X");
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
}

#[test]
fn manual_family_selector_filters_provider_family_class_and_freshness() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.override_rule = Some(RoleOverride::Family(
        RoleSelector::new()
            .provider("openai")
            .family("gpt")
            .class("strong")
            .capability(ModelCapability::Coding)
            .tier(ModelTier::Strong)
            .freshness(FreshnessPolicy::LatestStable),
    ));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "gpt-coder");
    assert_eq!(decision.source, RouteDecisionSource::ManualSelector);
    assert_eq!(decision.audit.cross_provider, Some(true));
}

#[test]
fn coding_family_selector_respects_premium_gate_but_exact_pin_remains_explicit() {
    let inventory = ModelInventory::new(
        "claude-sonnet-5",
        vec![
            ModelDescriptor::new("claude-sonnet-5", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("claude-fable-5", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong, ModelTier::Deep])
                .release_rank(100),
        ],
    );
    let family = RoleOverride::Family(
        RoleSelector::new()
            .provider("anthropic")
            .family("claude")
            .capability(ModelCapability::Coding)
            .tier(ModelTier::Strong),
    );

    let mut ordinary = RouteRequest::new(RouteRole::Coding, "claude-sonnet-5")
        .with_context(RoutePolicyContext {
            complexity: RouteTaskComplexity::Medium,
            ..RoutePolicyContext::default()
        });
    ordinary.override_rule = Some(family.clone());
    assert_eq!(route_model(&ordinary, &inventory).resolved_model, "claude-sonnet-5");
    assert!(
        route_model_fallback_candidates(&ordinary, &inventory, "claude-sonnet-5", 8)
            .is_empty(),
        "family-selector fallback candidates obey the same premium gate"
    );

    let mut escalated = ordinary.clone();
    escalated.context.prior_failures = 2;
    assert_eq!(route_model(&escalated, &inventory).resolved_model, "claude-fable-5");

    let mut exact_pin = ordinary;
    exact_pin.override_rule = Some(RoleOverride::Pin("claude-fable-5".to_string()));
    assert_eq!(route_model(&exact_pin, &inventory).resolved_model, "claude-fable-5");
    assert_eq!(route_model(&exact_pin, &inventory).source, RouteDecisionSource::Pinned);
}

#[test]
fn latest_stable_selector_skips_preview_even_when_newer() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.override_rule = Some(RoleOverride::Family(
        RoleSelector::new()
            .provider("openai")
            .capability(ModelCapability::Coding)
            .tier(ModelTier::Strong)
            .freshness(FreshnessPolicy::LatestStable),
    ));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "gpt-coder");
}

#[test]
fn latest_selector_can_choose_preview_when_it_scores_highest() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.override_rule = Some(RoleOverride::Family(
        RoleSelector::new()
            .provider("openai")
            .capability(ModelCapability::Coding)
            .tier(ModelTier::Strong)
            .freshness(FreshnessPolicy::Latest),
    ));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "gpt-preview");
}

#[test]
fn manual_mode_does_not_run_auto_selector() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.mode = RouterMode::Manual;
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "claude-sonnet-main");
    assert_eq!(decision.source, RouteDecisionSource::MainModelFallback);
}

#[test]
fn fallback_disabled_fails_closed_to_main_model() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.allow_fallback = false;
    request.override_rule = Some(RoleOverride::Family(RoleSelector::new().provider("missing").family("none")));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "claude-sonnet-main");
    assert_eq!(decision.source, RouteDecisionSource::FallbackDisabled);
}

#[test]
fn fallback_disabled_also_applies_when_auto_selector_has_no_match() {
    let sparse_inventory = ModelInventory::new("main-custom", vec![ModelDescriptor::new("main-custom", "custom", "custom")]);
    let mut request = RouteRequest::new(RouteRole::Coding, "main-custom");
    request.allow_fallback = false;
    let decision = route_model(&request, &sparse_inventory);
    assert_eq!(decision.resolved_model, "main-custom");
    assert_eq!(decision.source, RouteDecisionSource::FallbackDisabled);
}

#[test]
fn subagent_target_drives_effective_auto_role() {
    let target = RoutingTarget::Subagent(SubagentProfileId::builtin(BuiltinSubagentProfile::Plan));
    let request = RouteRequest::for_target(target, RouteRole::Coding, "claude-sonnet-main");
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "claude-opus-analysis");
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
}

#[test]
fn session_fast_state_propagates_to_router_picked_terra() {
    // 세션 fast on(메인 모델이 `[fast]`)이면 라우터가 스스로 고른 terra는
    // priority 티어로 나간다; fast off(bare 메인)면 bare 그대로 — 티어는
    // 하드코딩이 아니라 세션 상태 추종(사용자 정책 2026-07-11).
    let inventory_for = |main: &str| {
        ModelInventory::new(
            main,
            vec![ModelDescriptor::new("gpt-5.6-terra", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(56)],
        )
    };
    let fast_main = "gpt-5.6-sol[fast]";
    let decision = route_model(
        &RouteRequest::new(RouteRole::Coding, fast_main),
        &inventory_for(fast_main),
    );
    assert_eq!(decision.resolved_model, "gpt-5.6-terra[fast]");

    let standard_main = "gpt-5.6-sol";
    let decision = route_model(
        &RouteRequest::new(RouteRole::Coding, standard_main),
        &inventory_for(standard_main),
    );
    assert_eq!(decision.resolved_model, "gpt-5.6-terra");
}

#[test]
fn auto_selector_routes_coding_to_strong_coder() {
    let request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "gpt-coder");
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
}

#[test]
fn coding_premium_models_require_large_or_repeated_failure_escalation() {
    for (main, premium, standard, provider, family) in [
        ("gpt-5.6-sol", "gpt-5.6-sol", "gpt-5.6-terra", "openai", "gpt"),
        (
            "claude-fable-5",
            "claude-fable-5",
            "claude-sonnet-5",
            "anthropic",
            "claude",
        ),
    ] {
        let inventory = ModelInventory::new(
            main,
            vec![
                ModelDescriptor::new(premium, provider, family)
                    .source(ModelSource::CurrentMainModel)
                    .capabilities([ModelCapability::Coding])
                    .tiers([ModelTier::Deep, ModelTier::Strong])
                    .release_rank(100),
                ModelDescriptor::new(standard, provider, family)
                    .source(ModelSource::EnabledBuiltinProvider)
                    .capabilities([ModelCapability::Coding])
                    .tiers([ModelTier::Strong])
                    .release_rank(50),
            ],
        );
        let request = RouteRequest::new(RouteRole::Coding, main).with_context(RoutePolicyContext {
            complexity: RouteTaskComplexity::Medium,
            ..RoutePolicyContext::default()
        });

        let decision = route_model(&request, &inventory);
        assert_eq!(
            decision.resolved_model, standard,
            "{premium} is reserved for genuinely hard or repeatedly failing implementation"
        );
        assert!(
            !route_model_fallback_candidates(&request, &inventory, standard, 8)
                .iter()
                .any(|model| model == premium),
            "rate-limit candidates must not silently reintroduce {premium}"
        );
        let dashboard_coding = recommend_role_fallbacks(
            &inventory,
            &AutoAssignmentOptions::default(),
        )
        .into_iter()
        .find(|assignment| {
            assignment.target == RoutingTarget::RoleFallback(RouteRole::Coding)
        })
        .expect("coding recommendation");
        assert_eq!(
            dashboard_coding.selected_model, standard,
            "dashboard/default recommendations must match the live ordinary Coding route"
        );

        let one_failure = RouteRequest::new(RouteRole::Coding, main).with_context(
            RoutePolicyContext {
                complexity: RouteTaskComplexity::Medium,
                prior_failures: 1,
                ..RoutePolicyContext::default()
            },
        );
        assert_eq!(route_model(&one_failure, &inventory).resolved_model, standard);

        for context in [
            RoutePolicyContext {
                complexity: RouteTaskComplexity::Large,
                ..RoutePolicyContext::default()
            },
            RoutePolicyContext {
                complexity: RouteTaskComplexity::Medium,
                prior_failures: 2,
                ..RoutePolicyContext::default()
            },
        ] {
            assert_eq!(
                route_model(
                    &RouteRequest::new(RouteRole::Coding, main).with_context(context),
                    &inventory,
                )
                .resolved_model,
                premium,
                "{premium} should be available only after an explicit escalation signal"
            );
        }
    }
}

#[test]
fn debugging_is_an_implementation_route_for_premium_gating() {
    let inventory = ModelInventory::new(
        "claude-fable-5",
        vec![
            ModelDescriptor::new("claude-fable-5", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Debugging])
                .tiers([ModelTier::Deep, ModelTier::Strong])
                .release_rank(100),
            ModelDescriptor::new("claude-sonnet-5", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Debugging])
                .tiers([ModelTier::Strong])
                .release_rank(50),
        ],
    );
    let ordinary = RouteRequest::new(RouteRole::Debugging, "claude-fable-5")
        .with_context(RoutePolicyContext {
            complexity: RouteTaskComplexity::Medium,
            ..RoutePolicyContext::default()
        });

    assert_eq!(route_model(&ordinary, &inventory).resolved_model, "claude-sonnet-5");
    assert!(
        !route_model_fallback_candidates(&ordinary, &inventory, "claude-sonnet-5", 8)
            .iter()
            .any(|model| model == "claude-fable-5")
    );
    let dashboard = recommend_role_fallbacks(
        &inventory,
        &AutoAssignmentOptions::default(),
    )
    .into_iter()
    .find(|assignment| {
        assignment.target == RoutingTarget::RoleFallback(RouteRole::Debugging)
    })
    .expect("debugging recommendation");
    assert_eq!(dashboard.selected_model, "claude-sonnet-5");

    for context in [
        RoutePolicyContext {
            complexity: RouteTaskComplexity::Large,
            ..RoutePolicyContext::default()
        },
        RoutePolicyContext {
            complexity: RouteTaskComplexity::Medium,
            prior_failures: 2,
            ..RoutePolicyContext::default()
        },
    ] {
        assert_eq!(
            route_model(
                &RouteRequest::new(RouteRole::Debugging, "claude-fable-5")
                    .with_context(context),
                &inventory,
            )
            .resolved_model,
            "claude-fable-5"
        );
    }
}

#[test]
fn premium_implementation_model_matching_respects_family_boundaries_and_qualifiers() {
    for model in [
        "gpt-5.6-sol",
        "openai/gpt-5.6-sol-2026-07-09",
        "gpt-5.6-sol@openai",
        "claude-fable-5",
        "claude-fable",
        "anthropic/claude-fable-5[1m]",
        "sol",
        "openai/sol",
        "fable",
        "anthropic:fable",
        "fable5",
        "anthropic/fable5",
    ] {
        assert!(!implementation_route_model_allowed(
            model,
            RouteTaskComplexity::Medium,
            0,
            SmartPolicy::Classic
        ));
        assert!(
            is_reserved_orchestrator_model(model),
            "the public reserved-set predicate must match the gate: {model}"
        );
    }
    for near_miss in ["gpt-5.6-solar", "claude-fable-50", "console", "fabled"] {
        assert!(implementation_route_model_allowed(
            near_miss,
            RouteTaskComplexity::Medium,
            0,
            SmartPolicy::Classic
        ));
        assert!(!is_reserved_orchestrator_model(near_miss));
    }
}

#[test]
fn configured_deep_tier_pool_replaces_default_membership() {
    let defaults = default_deep_tier_models();
    assert!(is_deep_tier_model("fable", &defaults));
    assert!(is_deep_tier_model("sol", &defaults));
    assert!(!is_deep_tier_model("gpt-5.6-terra", &defaults));

    let custom = vec!["claude-opus-5".to_string()];
    assert!(is_deep_tier_model("opus-5", &custom));
    assert!(is_deep_tier_model("anthropic/claude-opus-5[1m]", &custom));
    assert!(!is_deep_tier_model("claude-fable-5", &custom));
}

#[test]
fn architect_policy_drops_the_large_complexity_escape_but_keeps_failure_escalation() {
    // Classic: a Large classification alone admits a reserved model to
    // implementation. Architect: it does not — only repeated real failures do.
    for reserved in ["claude-fable-5", "gpt-5.6-sol"] {
        assert!(implementation_route_model_allowed(
            reserved,
            RouteTaskComplexity::Large,
            0,
            SmartPolicy::Classic
        ));
        assert!(!implementation_route_model_allowed(
            reserved,
            RouteTaskComplexity::Large,
            0,
            SmartPolicy::Architect
        ));
        assert!(implementation_route_model_allowed(
            reserved,
            RouteTaskComplexity::Large,
            2,
            SmartPolicy::Architect
        ));
        assert!(implementation_route_model_allowed(
            reserved,
            RouteTaskComplexity::Small,
            2,
            SmartPolicy::Architect
        ));
    }
    // Non-reserved implementers are never gated, under either policy.
    assert!(implementation_route_model_allowed(
        "gpt-5.6-terra",
        RouteTaskComplexity::Large,
        0,
        SmartPolicy::Architect
    ));
    assert!(implementation_route_model_allowed(
        "claude-opus-4-8",
        RouteTaskComplexity::Unknown,
        0,
        SmartPolicy::Architect
    ));
}

#[test]
fn main_only_mode_always_uses_main_model() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    request.mode = RouterMode::MainOnly;
    request.override_rule = Some(RoleOverride::Pin("gpt-coder".to_string()));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.resolved_model, "claude-sonnet-main");
    assert_eq!(decision.source, RouteDecisionSource::MainOnly);
}

#[test]
fn fallback_candidates_return_second_best_route_without_selected_model() {
    let inv = ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("coder-a", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(10),
            ModelDescriptor::new("coder-b", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(12),
        ],
    );
    let request = RouteRequest::new(RouteRole::Coding, "main-model");
    let decision = route_model(&request, &inv);
    assert_eq!(decision.resolved_model, "coder-b");

    let candidates = route_model_fallback_candidates(&request, &inv, &decision.resolved_model, 4);

    assert_eq!(
        candidates,
        vec!["coder-a".to_string()],
        "router fallback candidates are ranked selector alternates; the runtime adds the parent model separately"
    );
}

#[test]
fn fallback_candidates_include_lower_auto_selector_tiers_for_retry() {
    let inv = ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("deep-a", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Deep])
                .release_rank(10),
            ModelDescriptor::new("strong-a", "gme-litellm", "qwen")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Strong])
                .release_rank(99),
            ModelDescriptor::new("balanced-a", "local", "custom")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Balanced])
                .release_rank(100),
        ],
    );
    let request = RouteRequest::new(RouteRole::Analysis, "main-model");
    let decision = route_model(&request, &inv);
    assert_eq!(decision.resolved_model, "deep-a");

    let candidates = route_model_fallback_candidates(&request, &inv, &decision.resolved_model, 4);

    assert_eq!(
        candidates,
        vec!["strong-a".to_string(), "balanced-a".to_string()],
        "provider retry fallbacks should keep walking the auto selector ladder after the selected Deep tier",
    );
}

#[test]
fn fallback_candidates_respect_limit_main_only_and_exact_pins() {
    let inv = inventory();
    let request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main");
    assert!(route_model_fallback_candidates(&request, &inv, "gpt-coder", 0).is_empty());

    let mut main_only = request.clone();
    main_only.mode = RouterMode::MainOnly;
    assert!(route_model_fallback_candidates(&main_only, &inv, "claude-sonnet-main", 2).is_empty());

    let mut pin = RouteRequest::new(RouteRole::Verifier, "claude-sonnet-main");
    pin.override_rule = Some(RoleOverride::Pin("gpt-coder".to_string()));
    assert!(
        route_model_fallback_candidates(&pin, &inv, "gpt-coder", 2).is_empty(),
        "an exact pin is explicit user intent; runtime fallback can still use the parent, but router has no second selector choice"
    );
}


#[test]
fn phase0_auto_assignment_snapshot_for_representative_builtin_profiles() {
    let plan = recommend_auto_assignments(&inventory());
    let selected_for = |profile: BuiltinSubagentProfile| {
        plan.assignments
            .iter()
            .find(|assignment| {
                assignment.target
                    == RoutingTarget::Subagent(SubagentProfileId::builtin(profile))
            })
            .unwrap_or_else(|| panic!("missing assignment for {profile:?}"))
    };

    // GeneralPurpose → Coding role. Best-of-breed: the strongest Coding model wins
    // regardless of provider. gpt-preview is excluded (Preview fails the freshness
    // prefilter, matching the live route), so the Stable gpt-coder is recommended —
    // the same model `route_model(Coding)` picks (the Anthropic main has no
    // Strong-tier coder of its own, so the route crosses to the best one).
    let general = selected_for(BuiltinSubagentProfile::GeneralPurpose);
    assert_eq!(general.selected_model, "gpt-coder");
    assert_eq!(general.source, AssignmentSource::Auto);
    assert_eq!(general.confidence, AssignmentConfidence::High);
    assert!(general.reason.contains("balanced worker"));

    let plan_profile = selected_for(BuiltinSubagentProfile::Plan);
    assert_eq!(plan_profile.selected_model, "claude-opus-analysis");
    assert_eq!(plan_profile.confidence, AssignmentConfidence::High);
    assert!(plan_profile.reason.contains("deep reasoning"));

    let verifier = selected_for(BuiltinSubagentProfile::Verification);
    assert_eq!(verifier.selected_model, "claude-sonnet-main");
    assert_eq!(verifier.confidence, AssignmentConfidence::High);
    assert!(verifier.reason.contains("verification"));
    assert!(verifier.reason.contains("diversity"));

    // StatuslineSetup → Fast role. No model has the Fast capability/tier, so the
    // recommendation falls back to the main model — exactly what the runtime does
    // (the live Fast route also finds no candidate and uses the main model).
    let statusline = selected_for(BuiltinSubagentProfile::StatuslineSetup);
    assert_eq!(statusline.selected_model, "claude-sonnet-main");
    assert_eq!(statusline.source, AssignmentSource::MainFallback);
    assert_eq!(statusline.confidence, AssignmentConfidence::Low);
    assert!(statusline.reason.contains("using main model"));
}


#[test]
fn auto_assignment_single_model_pool_uses_main_fallback() {
    let inventory = ModelInventory::new(
        "solo-main",
        vec![ModelDescriptor::new("solo-main", "custom", "custom")],
    );
    let plan = recommend_auto_assignments(&inventory);
    assert_eq!(plan.inventory_summary.usable_model_count, 1);
    assert!(plan.assignments.iter().all(|assignment| {
        assignment.selected_model == "solo-main"
            && assignment.source == AssignmentSource::MainFallback
            && !assignment.reason.is_empty()
    }));
}

#[test]
fn auto_assignment_prefers_verifier_diversity_when_available() {
    // Verifier diversity (a different model for error-diversity) is a
    // cross-provider feature, so it only applies when the user opts into diversity.
    // With diversity off the hard anchor keeps every role on the main provider.
    let plan = recommend_auto_assignments_with_options(
        &inventory(),
        &AutoAssignmentOptions { allow_cross_provider_diversity: true, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
    );
    let worker = plan
        .assignments
        .iter()
        .find(|assignment| {
            assignment.target
                == RoutingTarget::Subagent(SubagentProfileId::builtin(
                    BuiltinSubagentProfile::GeneralPurpose,
                ))
        })
        .expect("worker assignment");
    let verifier = plan
        .assignments
        .iter()
        .find(|assignment| {
            assignment.target
                == RoutingTarget::Subagent(SubagentProfileId::builtin(
                    BuiltinSubagentProfile::Verification,
                ))
        })
        .expect("verifier assignment");
    assert_ne!(worker.selected_model, verifier.selected_model);
    assert!(verifier.reason.contains("diversity"));
}

#[test]
fn auto_assignment_never_emits_model_outside_inventory() {
    let inventory = model_inventory_from_authorized_providers(
        "main-custom",
        &[],
        &[("local".to_string(), vec!["local-coder".to_string()])],
    );
    let plan = recommend_auto_assignments(&inventory);
    for assignment in &plan.assignments {
        assert!(
            inventory.find(&assignment.selected_model).is_some(),
            "assignment selected model outside inventory: {assignment:?}"
        );
        assert!(!assignment.reason.is_empty());
    }
}

#[test]
fn feedback_aware_recommendation_learns_from_outcome_history() {
    // The /smart dashboard's auto preview must reflect what actually performed,
    // not just the static recency prior. Two Coding-capable workers share a
    // provider+tier so only release_rank separates them: `newer-untested` has the
    // higher rank (would win the plain recommendation), but `proven-coder` has a
    // clean outcome history. With feedback folded in, the recommendation flips to
    // the proven model — mirroring how the live router (`route_model`) routes.
    let inventory = ModelInventory::new(
        "main-x",
        vec![
            ModelDescriptor::new("main-x", "x", "x")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("newer-untested", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(100),
            ModelDescriptor::new("proven-coder", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(1),
        ],
    );
    let options = AutoAssignmentOptions::default();

    // Without feedback, the higher release_rank wins the GeneralPurpose (Coding) slot.
    let plain = recommend_auto_assignments_with_options(&inventory, &options);
    let general = |plan: &AutoAssignmentPlan| {
        plan.assignments
            .iter()
            .find(|assignment| {
                assignment.target
                    == RoutingTarget::Subagent(SubagentProfileId::builtin(
                        BuiltinSubagentProfile::GeneralPurpose,
                    ))
            })
            .expect("general-purpose assignment")
            .selected_model
            .clone()
    };
    assert_eq!(general(&plain), "newer-untested");

    // GeneralPurpose records its outcomes under `subagent:general-purpose`; eight
    // clean runs earn the full feedback weight, enough to override recency.
    let records: Vec<_> = (0..8)
        .map(|_| {
            RouteOutcomeRecord::new("subagent", "general-purpose", "proven-coder", "completed")
        })
        .collect();
    let summary = summarize_route_outcomes(&records);
    let learned = recommend_auto_assignments_with_feedback(&inventory, &options, &summary);
    assert_eq!(
        general(&learned),
        "proven-coder",
        "auto recommendation must learn from the durable outcome history",
    );

    // Role fallbacks are NOT feedback-keyed: outcomes are only ever recorded
    // under `subagent:*` (both the live router and the spawn recorder file role
    // routes there too, never `role:*`), so a `subagent:general-purpose` history
    // must NOT bleed into the Coding role fallback — it stays on the static prior.
    let coding_role = |plan: &[TargetAssignment]| {
        plan.iter()
            .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Coding))
            .expect("coding role fallback")
            .selected_model
            .clone()
    };
    let role_plain = recommend_role_fallbacks(&inventory, &options);
    assert_eq!(
        coding_role(&role_plain),
        "newer-untested",
        "role fallback has no matching `role:*` history and must stay on the static prior",
    );
}

#[test]
fn code_reviewer_is_treated_as_verifier_group() {
    let plan = recommend_auto_assignments(&inventory());
    let reviewer = plan
        .assignments
        .iter()
        .find(|assignment| {
            assignment.target
                == RoutingTarget::Subagent(SubagentProfileId::builtin(
                    BuiltinSubagentProfile::CodeReviewer,
                ))
        })
        .expect("reviewer assignment");
    assert!(reviewer.reason.contains("diversity"));
    assert_ne!(reviewer.selected_model, "gpt-coder");
}

#[test]
fn provider_allowlist_constrains_auto_route_and_falls_back_to_main() {
    // Without an allowlist, Coding routes best-of-breed to the OpenAI coder.
    let open = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main")
        .with_context(neutral_route_context(true));
    assert_eq!(route_model(&open, &inventory()).resolved_model, "gpt-coder");

    // Allowlisting anthropic keeps the auto route off OpenAI entirely —
    // matching is case-insensitive, and the audit names the active allowlist.
    let mut anthropic_only = neutral_route_context(true);
    anthropic_only.provider_allowlist = vec!["Anthropic".to_string()];
    let anchored = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main")
        .with_context(anthropic_only);
    let decision = route_model(&anchored, &inventory());
    assert_eq!(
        decision.audit.selected_provider.as_deref(),
        Some("anthropic"),
        "a disallowed provider must never win on score: {decision:?}"
    );
    assert!(
        decision
            .audit
            .guardrails
            .iter()
            .any(|line| line.contains("provider allowlist")),
        "the active allowlist must be audited: {:?}",
        decision.audit.guardrails
    );

    // An allowlist matching no connected provider degrades to the main model
    // (the parent is the user's explicit choice, so it stays reachable).
    let mut nothing_connected = neutral_route_context(true);
    nothing_connected.provider_allowlist = vec!["gemini".to_string()];
    let fallback = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main")
        .with_context(nothing_connected);
    assert_eq!(
        route_model(&fallback, &inventory()).resolved_model,
        "claude-sonnet-main"
    );
}

#[test]
fn provider_allowlist_applies_to_recommendations_for_dashboard_parity() {
    let options = AutoAssignmentOptions {
        allow_cross_provider_diversity: true,
        provider_allowlist: vec!["anthropic".to_string()],
        policy: SmartPolicy::Classic,
    };
    let fallbacks = recommend_role_fallbacks(&inventory(), &options);
    let coding = fallbacks
        .iter()
        .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Coding))
        .expect("coding fallback");
    assert_ne!(
        coding.selected_model, "gpt-coder",
        "the dashboard preview must honor the same allowlist the live route does"
    );
}

#[test]
fn phase4_route_context_is_audited_without_overriding_hard_policy() {
    let mut request = RouteRequest::new(RouteRole::Coding, "claude-sonnet-main").with_context(RoutePolicyContext {
        risk: RouteTaskRisk::High,
        complexity: RouteTaskComplexity::Large,
        prior_failures: 0,
        context_need: RouteContextNeed::Unknown,
        tool_need: RouteToolNeed::Unknown,
        verification_need: RouteVerificationNeed::Unknown,
        route_shape: Some(RouteShapeKind::ParallelLanes),
        lane: Some(LaneRouteMetadata::new("backend").with_position(0, 2)),
        allow_cross_provider_diversity: false,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::enabled(999),
        audit_notes: vec!["unit-test-context".to_string()],
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    });
    request.explicit_model = Some("gpt-coder".to_string());

    let decision = route_model(&request, &inventory());

    assert_eq!(decision.resolved_model, "gpt-coder");
    assert_eq!(decision.source, RouteDecisionSource::Explicit);
    assert_eq!(decision.audit.route_shape.as_deref(), Some("parallel-lanes"));
    assert_eq!(decision.audit.lane_domain.as_deref(), Some("backend"));
    assert_eq!(decision.audit.feedback_adjustment, 120);
    assert!(decision.audit.guardrails.iter().any(|line| line.contains("bounded")));
    assert!(decision.audit.guardrails.iter().any(|line| line == "unit-test-context"));
}

#[test]
fn phase4_whole_repo_context_escalates_to_deep_tier() {
    // Two equally-matching coders; whole-repo context must escalate to the one
    // that also has a Deep tier, overcoming the other's release-rank edge.
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            // Same provider as the main: this test isolates the whole-repo Deep
            // escalation, so the candidates must clear the diversity-off hard anchor
            // (a cross-provider coder would be excluded before scoring).
            ModelDescriptor::new("coder-plain", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("coder-deep", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong, ModelTier::Deep])
                .release_rank(40),
        ],
    );
    let ctx = |context_need| RoutePolicyContext {
        risk: RouteTaskRisk::Low,
        complexity: RouteTaskComplexity::Small,
        prior_failures: 0,
        context_need,
        tool_need: RouteToolNeed::Unknown,
        verification_need: RouteVerificationNeed::Unknown,
        route_shape: None,
        lane: None,
        allow_cross_provider_diversity: false,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: Vec::new(),
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    };

    let local = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(ctx(RouteContextNeed::LocalFiles)),
        &inventory,
    );
    assert_eq!(local.resolved_model, "coder-plain", "local context keeps the higher-ranked coder");

    let whole = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(ctx(RouteContextNeed::WholeRepo)),
        &inventory,
    );
    assert_eq!(whole.resolved_model, "coder-deep", "whole-repo context escalates to the deep-tier coder");
}

#[test]
fn phase1_whole_repo_context_bonus_also_fires_on_large_context_window_without_deep_tier() {
    // Task 6: the WholeRepo bonus must ALSO fire for a model whose declared
    // context window is >= 300k, independent of tier — a huge-context model
    // that has not (yet) earned a Deep tier can still be favored here. Since
    // the user-directed 2026-07-14 change caps the whole GPT family at 258k,
    // no builtin GPT model qualifies; the path stays for big-window Claude
    // models and future entrants, exercised here with synthetic descriptors.
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("coder-plain", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50)
                .context_window(258_000),
            // Strong-tier only (deliberately NOT Deep) but a declared context
            // window >= 300k — isolates the new context_window path from the
            // pre-existing Deep-tier path.
            ModelDescriptor::new("coder-huge-context", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(40)
                .context_window(353_000),
        ],
    );
    let ctx = |context_need| RoutePolicyContext {
        risk: RouteTaskRisk::Low,
        complexity: RouteTaskComplexity::Small,
        prior_failures: 0,
        context_need,
        tool_need: RouteToolNeed::Unknown,
        verification_need: RouteVerificationNeed::Unknown,
        route_shape: None,
        lane: None,
        allow_cross_provider_diversity: false,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: Vec::new(),
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    };

    let local = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(ctx(RouteContextNeed::LocalFiles)),
        &inventory,
    );
    assert_eq!(local.resolved_model, "coder-plain", "local context keeps the higher-ranked coder (no bonus applies)");

    let whole = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(ctx(RouteContextNeed::WholeRepo)),
        &inventory,
    );
    assert_eq!(
        whole.resolved_model, "coder-huge-context",
        "whole-repo context escalates to the huge-context-window coder even without a Deep tier"
    );
}

#[test]
fn deep_tier_promoted_ultra_model_wins_analysis_role_on_both_scorers() {
    // Task 7 (scorer parity gate): a model promoted to Deep tier via its
    // Ultra effort ceiling (Phase 1's capability-derived rule, applied
    // upstream in `model_inventory::tiers_for_model`) must win the Analysis
    // role identically on the live route scorer (`policy::route_model`) AND
    // the dashboard recommendation scorer (`assignment::recommend_role_fallbacks`)
    // — the two must never silently disagree.
    let inventory = ModelInventory::new(
        "gpt-5.5",
        vec![
            ModelDescriptor::new("gpt-5.5", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default, ModelCapability::Analysis])
                .tiers([ModelTier::Balanced, ModelTier::Strong])
                .release_rank(55),
            ModelDescriptor::new("gpt-5.6-sol", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Deep, ModelTier::Strong])
                .effort_ceiling(EffortCeiling::Ultra)
                .release_rank(56),
        ],
    );
    let live = route_model(
        &RouteRequest::new(RouteRole::Analysis, "gpt-5.5").with_context(neutral_route_context(false)),
        &inventory,
    );
    assert_eq!(live.resolved_model, "gpt-5.6-sol", "live route should reach the Deep-promoted model");

    let role_preview = recommend_role_fallbacks(
        &inventory,
        &AutoAssignmentOptions { allow_cross_provider_diversity: false, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
    );
    let analysis = role_preview
        .iter()
        .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Analysis))
        .expect("analysis fallback");
    assert_eq!(
        analysis.selected_model, live.resolved_model,
        "dashboard recommendation must agree with the live route on the Deep-promoted model"
    );
}

#[test]
fn phase4_write_plus_full_verification_prefers_verification_capable() {
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            // Same provider as the main: isolates the write+full-verification
            // escalation from the diversity-off hard anchor (a cross-provider coder
            // would be excluded before the context bonus could promote it).
            ModelDescriptor::new("coder-plain", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("coder-verify", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding, ModelCapability::Verification])
                .tiers([ModelTier::Strong])
                .release_rank(40),
        ],
    );
    let ctx = |tool_need, verification_need| RoutePolicyContext {
        risk: RouteTaskRisk::Low,
        complexity: RouteTaskComplexity::Small,
        prior_failures: 0,
        context_need: RouteContextNeed::LocalFiles,
        tool_need,
        verification_need,
        route_shape: None,
        lane: None,
        allow_cross_provider_diversity: false,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: Vec::new(),
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    };

    let read_only = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic")
            .with_context(ctx(RouteToolNeed::ReadOnly, RouteVerificationNeed::Focused)),
        &inventory,
    );
    assert_eq!(read_only.resolved_model, "coder-plain");

    let write_full = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic")
            .with_context(ctx(RouteToolNeed::Write, RouteVerificationNeed::Full)),
        &inventory,
    );
    assert_eq!(
        write_full.resolved_model, "coder-verify",
        "write + full verification escalates to the verification-capable coder",
    );
}

#[test]
fn phase5_cross_provider_diversity_is_default_off_and_opt_in() {
    fn selected_verifier(plan: &AutoAssignmentPlan) -> &TargetAssignment {
        plan.assignments
            .iter()
            .find(|assignment| {
                assignment.target
                    == RoutingTarget::Subagent(SubagentProfileId::builtin(
                        BuiltinSubagentProfile::Verification,
                    ))
            })
            .expect("verification assignment")
    }

    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("openai-worker", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(10),
            ModelDescriptor::new("openai-verifier", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("anthropic-verifier", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(2),
        ],
    );

    let default_plan = recommend_auto_assignments(&inventory);
    let opt_in_plan = recommend_auto_assignments_with_options(
        &inventory,
        &AutoAssignmentOptions { allow_cross_provider_diversity: true, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
    );
    let default_verifier = selected_verifier(&default_plan);
    let opt_in_verifier = selected_verifier(&opt_in_plan);
    // Unified with the live route scorer: provider anchoring is referenced to the
    // MAIN model. Off (default) anchors the verifier to the main provider
    // (anthropic); On rewards crossing to a different provider (openai).
    assert_eq!(default_verifier.selected_model, "anthropic-verifier");
    assert!(default_verifier.audit.iter().any(|line| line.contains("disabled by default")));
    assert_eq!(opt_in_verifier.selected_model, "openai-verifier");
    assert!(opt_in_verifier.audit.iter().any(|line| line.contains("allowed")));
}

#[test]
fn phase5_live_route_diversity_toggle_flips_selected_verifier() {
    // The recommendation path is covered above; this pins the LIVE route_model
    // path (diversity_context_adjustment): the same Verifier request must pick a
    // same-provider verifier when cross-provider diversity is off, and the
    // cross-provider verifier only when it is explicitly allowed.
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            // Reference (parent) model: not a verifier candidate.
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            // anthropic-verifier outranks openai-verifier, so best-of-breed (off)
            // picks it on rank alone; the diversity reward (on) must overcome that
            // rank gap to flip the verifier cross-provider.
            ModelDescriptor::new("anthropic-verifier", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(3),
            ModelDescriptor::new("openai-verifier", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(2),
        ],
    );

    let context = |allow_cross_provider_diversity| RoutePolicyContext {
        risk: RouteTaskRisk::Low,
        complexity: RouteTaskComplexity::Small,
        prior_failures: 0,
        context_need: RouteContextNeed::Unknown,
        tool_need: RouteToolNeed::Unknown,
        verification_need: RouteVerificationNeed::Unknown,
        route_shape: None,
        lane: None,
        allow_cross_provider_diversity,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: Vec::new(),
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    };

    let off = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-anthropic").with_context(context(false)),
        &inventory,
    );
    assert_eq!(off.resolved_model, "anthropic-verifier", "diversity off → best-of-breed picks the higher-ranked verifier");

    let on = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-anthropic").with_context(context(true)),
        &inventory,
    );
    assert_eq!(on.resolved_model, "openai-verifier", "diversity on → cross-provider verifier reward flips the pick");
}

#[test]
fn worker_role_picks_best_of_breed_coder_across_providers() {
    // A worker (Coding) role on a GPT main: routing is best-of-breed, so the
    // strongest coder wins regardless of provider — here the higher-ranked
    // claude-coder, not the same-provider gpt-coder — for BOTH the default and the
    // diversity opt-in (a non-diversity role gets no cross-provider reward, so the
    // pick is rank-driven either way).
    let inventory = ModelInventory::new(
        "main-gpt",
        vec![
            ModelDescriptor::new("main-gpt", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("gpt-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(40),
            ModelDescriptor::new("claude-coder", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
        ],
    );
    for allow_cross in [false, true] {
        let decision = route_model(
            &RouteRequest::new(RouteRole::Coding, "main-gpt").with_context(neutral_route_context(allow_cross)),
            &inventory,
        );
        assert_eq!(
            decision.resolved_model, "claude-coder",
            "best-of-breed picks the strongest coder cross-provider (cross={allow_cross})",
        );
    }
}

#[test]
fn recommendation_matches_best_of_breed_live_route_for_worker() {
    // The recommendation path must agree with the live path (above): best-of-breed,
    // so the worker (Coding) role recommends the strongest coder regardless of
    // provider — the higher-ranked claude-coder — for both off and on.
    let inventory = ModelInventory::new(
        "main-gpt",
        vec![
            ModelDescriptor::new("main-gpt", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("gpt-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(40),
            ModelDescriptor::new("claude-coder", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
        ],
    );
    let worker = |plan: &AutoAssignmentPlan| {
        plan.assignments
            .iter()
            .find(|assignment| {
                assignment.target
                    == RoutingTarget::Subagent(SubagentProfileId::builtin(
                        BuiltinSubagentProfile::GeneralPurpose,
                    ))
            })
            .expect("worker assignment")
            .selected_model
            .clone()
    };
    let off = recommend_auto_assignments(&inventory);
    let on = recommend_auto_assignments_with_options(
        &inventory,
        &AutoAssignmentOptions { allow_cross_provider_diversity: true, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
    );
    assert_eq!(worker(&off), "claude-coder", "best-of-breed worker (default)");
    assert_eq!(worker(&on), "claude-coder", "best-of-breed worker (opt-in)");
}

#[test]
fn role_fallback_recommendations_cover_every_role_within_inventory() {
    let fallbacks = recommend_role_fallbacks(&inventory(), &AutoAssignmentOptions::default());
    assert_eq!(fallbacks.len(), RouteRole::all().len());
    for assignment in &fallbacks {
        assert!(matches!(assignment.target, RoutingTarget::RoleFallback(_)));
        assert!(
            inventory().find(&assignment.selected_model).is_some(),
            "role fallback selected model outside inventory: {assignment:?}",
        );
    }
    // Coding role: best-of-breed picks the Stable gpt-coder (gpt-preview is excluded
    // by the freshness prefilter), crossing from the Anthropic main because it has
    // no Strong-tier coder of its own — the same model the live `route_model(Coding)`
    // resolves.
    let coding = fallbacks
        .iter()
        .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Coding))
        .expect("coding role fallback");
    assert_eq!(coding.selected_model, "gpt-coder");
}

/// A production-representative but otherwise neutral route context. Every task
/// signal stays Unknown/None so no task bonus fires, isolating the same
/// best-of-breed selection (prefilter + score) shared with the recommendation
/// scorer while still letting tests toggle route-level policy such as diversity.
fn neutral_route_context(allow_cross_provider_diversity: bool) -> RoutePolicyContext {
    RoutePolicyContext {
        risk: RouteTaskRisk::Unknown,
        complexity: RouteTaskComplexity::Unknown,
        prior_failures: 0,
        context_need: RouteContextNeed::Unknown,
        tool_need: RouteToolNeed::Unknown,
        verification_need: RouteVerificationNeed::Unknown,
        route_shape: Some(RouteShapeKind::Solo),
        lane: None,
        allow_cross_provider_diversity,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::disabled(),
        audit_notes: Vec::new(),
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    }
}

#[test]
fn default_role_routes_generic_work_by_difficulty() {
    // A generic (Default-role) agent has no specialty to anchor on, so it routes by
    // task *difficulty*: trivial/small work to a cheap Fast-tier model, heavy work to a
    // Strong-tier model. An uninferable task (Unknown complexity) has no difficulty
    // signal, so best-of-breed keeps the parent (via its same-main bonus). All three
    // candidates share the main's provider, isolating the tier preference from provider
    // anchoring, and Default carries no specialty seed — so tier is the only mover.
    let inventory = ModelInventory::new(
        "claude-main",
        vec![
            ModelDescriptor::new("claude-main", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .class("sonnet")
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(5),
            ModelDescriptor::new("claude-cheap", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .class("haiku")
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Fast])
                .release_rank(1),
            ModelDescriptor::new("claude-heavy", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .class("opus")
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Strong])
                .release_rank(1),
        ],
    );
    let ctx = |complexity| RoutePolicyContext { complexity, ..neutral_route_context(false) };

    let small = route_model(
        &RouteRequest::new(RouteRole::Default, "claude-main").with_context(ctx(RouteTaskComplexity::Small)),
        &inventory,
    );
    assert_eq!(
        small.resolved_model, "claude-cheap",
        "small generic work routes to the cheap Fast-tier model, not the parent"
    );

    let large = route_model(
        &RouteRequest::new(RouteRole::Default, "claude-main").with_context(ctx(RouteTaskComplexity::Large)),
        &inventory,
    );
    assert_eq!(
        large.resolved_model, "claude-heavy",
        "large generic work routes to the Strong-tier model"
    );

    let unknown = route_model(
        &RouteRequest::new(RouteRole::Default, "claude-main").with_context(ctx(RouteTaskComplexity::Unknown)),
        &inventory,
    );
    assert_eq!(
        unknown.resolved_model, "claude-main",
        "an uninferable generic task (Unknown complexity) stays on the parent model"
    );
}

#[test]
fn fast_role_picks_best_of_breed_fast_model_across_providers() {
    // Explore/statusline subagents route to the Fast role. Best-of-breed: the
    // strongest (highest-ranked) fast model wins regardless of provider — here the
    // higher-ranked gpt-fast over the Anthropic main's own haiku.
    let inventory = ModelInventory::new(
        "main-claude",
        vec![
            ModelDescriptor::new("main-claude", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("claude-haiku", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Fast])
                .tiers([ModelTier::Fast, ModelTier::Balanced])
                .release_rank(5),
            ModelDescriptor::new("gpt-fast", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Fast])
                .tiers([ModelTier::Fast, ModelTier::Balanced])
                .release_rank(9),
        ],
    );
    let decision = route_model(
        &RouteRequest::new(RouteRole::Fast, "main-claude").with_context(neutral_route_context(false)),
        &inventory,
    );
    assert_eq!(
        decision.resolved_model, "gpt-fast",
        "best-of-breed Fast picks the highest-ranked fast model regardless of provider",
    );
}

#[test]
fn analysis_crosses_to_best_deep_model_when_main_provider_lacks_one() {
    // The user's contract: each role uses the best specialist, cross-provider. A GPT
    // main asking for Analysis (Deep-tier) — which OpenAI's catalog lacks — reaches
    // the best Deep model on another provider (claude-deep) instead of being stuck
    // on a weaker same-provider model. Best-of-breed has no cross penalty, so this
    // holds whether or not the user enables the (verifier-only) diversity reward.
    let inventory = ModelInventory::new(
        "gpt-main",
        vec![
            ModelDescriptor::new("gpt-main", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default, ModelCapability::Coding])
                .tiers([ModelTier::Balanced, ModelTier::Strong])
                .release_rank(55),
            ModelDescriptor::new("claude-deep", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Deep])
                .release_rank(40),
        ],
    );
    for allow_cross in [false, true] {
        let decision = route_model(
            &RouteRequest::new(RouteRole::Analysis, "gpt-main").with_context(neutral_route_context(allow_cross)),
            &inventory,
        );
        assert_eq!(
            decision.resolved_model, "claude-deep",
            "Analysis reaches the best Deep model cross-provider (cross={allow_cross})",
        );
    }
}

#[test]
fn analysis_role_falls_back_to_strong_then_balanced_when_no_deep_exists() {
    let strong_inventory = ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("balanced-analyst", "local", "custom")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Balanced])
                .release_rank(90),
            ModelDescriptor::new("qwen-strong-analyst", "gme-litellm", "qwen")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Strong])
                .release_rank(10),
        ],
    );
    let live = route_model(
        &RouteRequest::new(RouteRole::Analysis, "main-model").with_context(neutral_route_context(true)),
        &strong_inventory,
    );
    assert_eq!(
        live.resolved_model, "qwen-strong-analyst",
        "Analysis should use a Strong analysis-capable model when no Deep model exists",
    );
    let role_preview = recommend_role_fallbacks(
        &strong_inventory,
        &AutoAssignmentOptions { allow_cross_provider_diversity: true, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
    );
    let analysis = role_preview
        .iter()
        .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Analysis))
        .expect("analysis fallback");
    assert_eq!(analysis.selected_model, live.resolved_model);

    let balanced_inventory = ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("balanced-analyst", "local", "custom")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Balanced])
                .release_rank(90),
        ],
    );
    let balanced = route_model(
        &RouteRequest::new(RouteRole::Analysis, "main-model").with_context(neutral_route_context(true)),
        &balanced_inventory,
    );
    assert_eq!(
        balanced.resolved_model, "balanced-analyst",
        "Analysis should still avoid main fallback when only a Balanced analysis model exists",
    );
}

#[test]
fn role_recommendation_equals_live_route_for_each_role() {
    // Parity guard: the dashboard's per-role recommendation must equal the model
    // the live (production, with-context) `route_model` resolves for that role,
    // including the freshness/tier prefilter and the main-model fallback. This is
    // the divergence the recommend-vs-runtime audit flagged (a high-rank Preview
    // model recommended but never routed to).
    let inv = inventory();
    let fallbacks = recommend_role_fallbacks(&inv, &AutoAssignmentOptions::default());
    for role in RouteRole::all().iter().copied() {
        let live = route_model(
            &RouteRequest::new(role, "claude-sonnet-main").with_context(neutral_route_context(false)),
            &inv,
        );
        let recommended = fallbacks
            .iter()
            .find(|assignment| assignment.target == RoutingTarget::RoleFallback(role))
            .expect("role fallback");
        assert_eq!(
            recommended.selected_model, live.resolved_model,
            "recommendation for {role:?} must equal the live route",
        );
    }
}

#[test]
fn role_recommendation_equals_default_policy_live_route_for_each_role() {
    // Default RouteRequest must use the same AUTO scorer as the dashboard
    // recommendation path. This guards against regressing to the older
    // selector-only scorer for default-policy routes.
    let inv = inventory();
    let fallbacks = recommend_role_fallbacks(&inv, &AutoAssignmentOptions::default());
    for role in RouteRole::all().iter().copied() {
        let live = route_model(&RouteRequest::new(role, "claude-sonnet-main"), &inv);
        let recommended = fallbacks
            .iter()
            .find(|assignment| assignment.target == RoutingTarget::RoleFallback(role))
            .expect("role fallback");
        assert_eq!(
            recommended.selected_model, live.resolved_model,
            "default-policy recommendation for {role:?} must equal the live route",
        );
    }
}

#[test]
fn role_recommendation_equals_anchored_live_route_with_multiple_candidates() {
    // The fixture above has ≤1 candidate per role, so it can't exercise the
    // scorer's anchoring tie-break. This inventory has two same-tier coders from
    // different providers, so provider anchoring decides — and the recommendation
    // must still equal the production (anchored) live route, both off and on.
    let inventory = ModelInventory::new(
        "main-gpt",
        vec![
            ModelDescriptor::new("main-gpt", "openai", "gpt")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("gpt-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(40),
            ModelDescriptor::new("claude-coder", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
        ],
    );
    for allow_cross in [false, true] {
        let live = route_model(
            &RouteRequest::new(RouteRole::Coding, "main-gpt")
                .with_context(neutral_route_context(allow_cross)),
            &inventory,
        );
        let fallbacks = recommend_role_fallbacks(
            &inventory,
            &AutoAssignmentOptions { allow_cross_provider_diversity: allow_cross, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
        );
        let coding = fallbacks
            .iter()
            .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Coding))
            .expect("coding role fallback");
        assert_eq!(
            coding.selected_model, live.resolved_model,
            "recommendation must equal the anchored live route (cross={allow_cross})",
        );
        // Best-of-breed: the higher-ranked claude-coder wins regardless of the
        // diversity setting (Coding is not an error-diversity role, so `on` adds no
        // cross-provider reward).
        assert_eq!(coding.selected_model, "claude-coder");
    }
}

#[test]
fn phase6_feedback_hint_is_bounded_and_opt_in() {
    let disabled = RouteFeedbackHint::disabled();
    assert_eq!(disabled.bounded_adjustment(), 0);

    let enabled = RouteFeedbackHint::enabled(-999);
    assert_eq!(enabled.bounded_adjustment(), -120);

    let request = RouteRequest::new(RouteRole::Research, "claude-sonnet-main").with_context(RoutePolicyContext {
        risk: RouteTaskRisk::Medium,
        complexity: RouteTaskComplexity::Large,
        prior_failures: 0,
        context_need: RouteContextNeed::Unknown,
        tool_need: RouteToolNeed::Unknown,
        verification_need: RouteVerificationNeed::Unknown,
        route_shape: Some(RouteShapeKind::RepairLoop),
        lane: None,
        allow_cross_provider_diversity: false,
        provider_allowlist: Vec::new(),
        feedback: RouteFeedbackHint::enabled(999),
        audit_notes: Vec::new(),
        cooldown_providers: Vec::new(),
        provider_headroom: Vec::new(),
        headroom_penalty_threshold: 0,
        exploration_slot: None,
        exploration_decisive_counts: Vec::new(),
        learned_specialty: LearnedSpecialtyHint::disabled(),
        policy: SmartPolicy::Classic,
    });
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
    assert_eq!(decision.reason, "auto role selector with route context");
    assert_eq!(decision.audit.feedback_adjustment, 120);
}

#[test]
fn outcome_summary_builds_bounded_feedback_hint_for_route_key() {
    let one_sample = summarize_route_outcomes(&[RouteOutcomeRecord::new(
        "subagent",
        "Verification",
        "steady-coder",
        "completed",
    )]);
    assert_eq!(
        one_sample
            .feedback_hint_for_route_key("subagent:Verification")
            .bounded_adjustment_for("steady-coder"),
        0,
        "single samples are too weak to affect routing",
    );

    let summary = summarize_route_outcomes(&[
        RouteOutcomeRecord::new("subagent", "Verification", "steady-coder", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "steady-coder", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "flaky-coder", "failed"),
        RouteOutcomeRecord::new("subagent", "Verification", "flaky-coder", "failed"),
        RouteOutcomeRecord::new("subagent", "Verification", "flaky-coder", "stopped"),
        RouteOutcomeRecord::new("subagent", "Verification", "cancelled-coder", "stopped"),
        RouteOutcomeRecord::new("subagent", "Verification", "cancelled-coder", "stopped"),
        RouteOutcomeRecord::new("subagent", "debugger", "debugger-model", "completed"),
        RouteOutcomeRecord::new("subagent", "debugger", "debugger-model", "completed"),
    ]);
    let hint = summary.feedback_hint_for_route_key("subagent:Verification");

    // steady-coder: 2 completed / 0 failed. Confidence-weighted (P3): a perfect but
    // THIN history (2 of 8 confident samples) earns only 2/8 of the full bound →
    // 120 * 2/8 = 30. It will climb toward 120 as more decisive runs accumulate.
    assert_eq!(hint.bounded_adjustment_for("steady-coder"), 30);
    // flaky-coder: 2 real failures + 1 user-cancel. The cancel is fully neutral
    // (excluded from numerator AND denominator), so the score is -30 over the 2
    // decisive runs (same 2/8 confidence ramp) — the cancel neither penalizes nor dilutes.
    assert_eq!(hint.bounded_adjustment_for("flaky-coder"), -30);
    // cancelled-coder: only user-cancels → 0 decisive runs → no adjustment.
    assert_eq!(hint.bounded_adjustment_for("cancelled-coder"), 0);
    assert_eq!(hint.bounded_adjustment_for("debugger-model"), 0);
}

#[test]
fn outcome_feedback_reaches_full_weight_and_overrides_recency() {
    // P3: a well-evidenced route earns the full feedback weight and that weight is
    // large enough to override the recency (release_rank) prior — the whole point
    // of the reweight. With the old ±40 bound, the newer model's rank always won.
    let records: Vec<_> = (0..8)
        .map(|_| RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"))
        .collect();
    let hint =
        summarize_route_outcomes(&records).feedback_hint_for_route_key("subagent:Verification");
    assert_eq!(
        hint.bounded_adjustment_for("proven-model"),
        120,
        "8 clean decisive runs earn the full feedback weight",
    );

    // A lower release-rank but well-proven model beats a brand-new top-rank one.
    let inventory = ModelInventory::new(
        "main-x",
        vec![
            ModelDescriptor::new("main-x", "x", "x")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("newer-untested", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(100),
            ModelDescriptor::new("proven-model", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
        ],
    );
    let decision = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-x").with_context(RoutePolicyContext {
            feedback: hint,
            ..neutral_route_context(false)
        }),
        &inventory,
    );
    assert_eq!(
        decision.resolved_model, "proven-model",
        "a well-evidenced model overrides the recency (release_rank) prior",
    );
}

#[test]
fn phase3_route_decision_is_byte_identical_on_a_fixed_fixture() {
    // HARD CONSTRAINT regression lock for the P3 outcome-store-v2 phase
    // (schema/retention/canonicalization/half-life/manifest-reread/still_running
    // doctrine): none of that work may move a real routing decision. This
    // fixture pins the exact `route_model` output — resolved model AND the
    // numeric feedback adjustment baked into the audit — against a fixed set
    // of outcome records, INCLUDING v2-schema fields (role/complexity/risk/
    // routeSource) on some of them, so the assertions also double as proof
    // that the new fields are inert to scoring (only `summarize_route_outcomes`
    // — the identity-canonicalizer default every existing caller still uses —
    // feeds real routing in this phase; `summarize_route_outcomes_with_canonicalizer`/
    // `weighted_feedback_hint_for_route_key` are additive, unwired siblings).
    let records = vec![
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed")
            .with_role(Some("verifier".to_string()))
            .with_complexity(Some("medium".to_string()))
            .with_risk(Some("low".to_string()))
            .with_route_source(Some("auto".to_string())),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
        RouteOutcomeRecord::new("subagent", "Verification", "proven-model", "completed"),
    ];

    let summary = summarize_route_outcomes(&records);
    let hint = summary.feedback_hint_for_route_key("subagent:Verification");
    assert_eq!(
        hint.bounded_adjustment_for("proven-model"),
        120,
        "8 clean decisive runs earn the full feedback weight",
    );

    let inventory = ModelInventory::new(
        "main-x",
        vec![
            ModelDescriptor::new("main-x", "x", "x")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("newer-untested", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(100),
            ModelDescriptor::new("proven-model", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
        ],
    );
    let decision = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-x").with_context(RoutePolicyContext {
            feedback: hint,
            ..neutral_route_context(false)
        }),
        &inventory,
    );

    assert_eq!(decision.resolved_model, "proven-model");
    assert_eq!(decision.audit.feedback_adjustment, 120);
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
}

#[test]
fn outcome_feedback_is_model_specific_bounded_and_cannot_override_explicit_pin() {
    let inventory = ModelInventory::new(
        "main-model",
        vec![
            // All three share one provider so the diversity-off hard anchor keeps
            // the coders in play — this test isolates the feedback adjustment, not
            // provider anchoring.
            ModelDescriptor::new("main-model", "local", "main")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("steady-coder", "local", "steady")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("flaky-coder", "local", "flaky")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(51),
        ],
    );
    let feedback = RouteFeedbackHint::for_model("steady-coder", 40)
        .with_model_adjustment("flaky-coder", -40);
    let auto = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model")
            .with_context(RoutePolicyContext { feedback, ..RoutePolicyContext::default() }),
        &inventory,
    );

    assert_eq!(auto.resolved_model, "steady-coder");
    assert_eq!(auto.audit.feedback_adjustment, 40);

    let pinned = route_model(
        &RouteRequest {
            explicit_model: Some("flaky-coder".to_string()),
            context: RoutePolicyContext {
                feedback: RouteFeedbackHint::for_model("steady-coder", 50),
                ..RoutePolicyContext::default()
            },
            ..RouteRequest::new(RouteRole::Coding, "main-model")
        },
        &inventory,
    );

    assert_eq!(pinned.resolved_model, "flaky-coder");
    assert_eq!(pinned.source, RouteDecisionSource::Explicit);
}

#[test]
fn specialty_seed_and_same_main_bonus_taper_by_role() {
    use super::policy::{cold_start_specialty_seed, same_main_bonus};

    // same-main bonus: generic roles keep the strong +25 (staying on the main model
    // is sensible); specialist worker roles get only a small tie-break so a role's
    // best specialty model can win instead of the main model dominating its tier.
    assert_eq!(same_main_bonus(RouteRole::Default), 25);
    assert_eq!(same_main_bonus(RouteRole::Fast), 25);
    assert_eq!(same_main_bonus(RouteRole::Coding), 5);
    assert_eq!(same_main_bonus(RouteRole::Analysis), 5);
    assert_eq!(same_main_bonus(RouteRole::Verifier), 5);

    // specialty seed: per-role family preference (cold-start prior, overridden by
    // the feedback loop once outcomes accrue). `gpt` covers the codex line.
    let gpt = ModelDescriptor::new("x", "openai", "gpt");
    let claude = ModelDescriptor::new("y", "anthropic", "claude");
    let deepseek = ModelDescriptor::new("z", "deepseek", "deepseek");
    assert_eq!(cold_start_specialty_seed(RouteRole::Coding, &gpt), 60);
    assert_eq!(cold_start_specialty_seed(RouteRole::Coding, &claude), 60);
    assert_eq!(cold_start_specialty_seed(RouteRole::Coding, &deepseek), 0);
    assert_eq!(cold_start_specialty_seed(RouteRole::Analysis, &deepseek), 60);
    assert_eq!(cold_start_specialty_seed(RouteRole::Writing, &claude), 60);
    assert_eq!(cold_start_specialty_seed(RouteRole::Writing, &gpt), 0);
    // verifier/fast/default are neutral — diversity, tier, and recency decide.
    assert_eq!(cold_start_specialty_seed(RouteRole::Verifier, &claude), 0);
    assert_eq!(cold_start_specialty_seed(RouteRole::Fast, &gpt), 0);
}

#[test]
fn verifier_auto_selectors_escalate_to_strong_first_for_high_risk_or_large_complexity() {
    use super::policy::{auto_selectors_for_role, verifier_should_try_strong_first};

    // Low/Medium risk + non-Large complexity: byte-identical to the original
    // single Balanced-only ladder (the common case, unconditioned by Phase 8).
    for risk in [RouteTaskRisk::Low, RouteTaskRisk::Medium, RouteTaskRisk::Unknown] {
        for complexity in [
            RouteTaskComplexity::Trivial,
            RouteTaskComplexity::Small,
            RouteTaskComplexity::Medium,
            RouteTaskComplexity::Unknown,
        ] {
            assert!(
                !verifier_should_try_strong_first(risk, complexity),
                "{risk:?}/{complexity:?} must not escalate"
            );
            for role in [RouteRole::Verifier, RouteRole::Reviewer] {
                let selectors = auto_selectors_for_role(role, risk, complexity, SmartPolicy::Classic);
                assert_eq!(selectors.len(), 1, "{role:?} must keep the single Balanced-only rung");
                assert_eq!(selectors[0].tier, Some(ModelTier::Balanced));
            }
        }
    }

    // High/Critical risk OR Large complexity (either alone suffices): Strong
    // tried first, Balanced kept as the fallback rung.
    for (risk, complexity) in [
        (RouteTaskRisk::High, RouteTaskComplexity::Unknown),
        (RouteTaskRisk::Critical, RouteTaskComplexity::Small),
        (RouteTaskRisk::Low, RouteTaskComplexity::Large),
    ] {
        assert!(
            verifier_should_try_strong_first(risk, complexity),
            "{risk:?}/{complexity:?} must escalate"
        );
        for role in [RouteRole::Verifier, RouteRole::Reviewer] {
            let selectors = auto_selectors_for_role(role, risk, complexity, SmartPolicy::Classic);
            assert_eq!(selectors.len(), 2, "{role:?} must try Strong before Balanced");
            assert_eq!(selectors[0].tier, Some(ModelTier::Strong));
            assert_eq!(selectors[1].tier, Some(ModelTier::Balanced));
        }
    }
}

#[test]
fn architect_policy_verifier_ladder_tries_deep_then_strong_then_balanced() {
    use super::policy::auto_selectors_for_role;
    // Architect: the checker ladder starts at the Deep rung regardless of
    // risk/complexity (verification is the quality bar), then falls through
    // the classic rungs — so a pool with no Deep verifier degrades exactly to
    // the classic behavior instead of failing the route.
    for role in [RouteRole::Verifier, RouteRole::Reviewer] {
        for (risk, complexity) in [
            (RouteTaskRisk::Low, RouteTaskComplexity::Small),
            (RouteTaskRisk::Critical, RouteTaskComplexity::Large),
        ] {
            let selectors = auto_selectors_for_role(role, risk, complexity, SmartPolicy::Architect);
            assert_eq!(
                selectors.iter().map(|selector| selector.tier).collect::<Vec<_>>(),
                vec![Some(ModelTier::Deep), Some(ModelTier::Strong), Some(ModelTier::Balanced)],
                "{role:?} {risk:?}/{complexity:?} must ladder Deep→Strong→Balanced under architect"
            );
        }
    }
    // Non-checker roles are untouched by the policy.
    assert_eq!(
        auto_selectors_for_role(RouteRole::Coding, RouteTaskRisk::Low, RouteTaskComplexity::Small, SmartPolicy::Architect),
        auto_selectors_for_role(RouteRole::Coding, RouteTaskRisk::Low, RouteTaskComplexity::Small, SmartPolicy::Classic),
    );
}

#[test]
fn architect_policy_routes_large_implementation_away_from_reserved_models() {
    // End-to-end through `route_model`: under classic, a Large Coding route may
    // land on the reserved flagship; under architect the same request lands on
    // a standard implementer, and only a repeated-failure escalation readmits
    // the reserved model.
    let inventory = ModelInventory::new(
        "claude-fable-5",
        vec![
            ModelDescriptor::new("claude-fable-5", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default, ModelCapability::Coding, ModelCapability::Analysis])
                .tiers([ModelTier::Deep, ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("gpt-5.6-terra", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Default, ModelCapability::Coding])
                .tiers([ModelTier::Balanced, ModelTier::Strong])
                .release_rank(56),
        ],
    );
    let context = |policy, prior_failures| RoutePolicyContext {
        complexity: RouteTaskComplexity::Large,
        prior_failures,
        policy,
        ..neutral_route_context(false)
    };
    let architect = route_model(
        &RouteRequest::new(RouteRole::Coding, "claude-fable-5")
            .with_context(context(SmartPolicy::Architect, 0)),
        &inventory,
    );
    assert_eq!(
        architect.resolved_model, "gpt-5.6-terra",
        "architect must keep Large implementation on a standard implementer"
    );
    let escalated = route_model(
        &RouteRequest::new(RouteRole::Coding, "claude-fable-5")
            .with_context(context(SmartPolicy::Architect, 2)),
        &inventory,
    );
    assert_eq!(
        escalated.resolved_model, "claude-fable-5",
        "two real failures escalate implementation to the reserved model"
    );
}

#[test]
fn architect_policy_verifier_route_lands_on_a_cross_provider_deep_checker() {
    // With a Deep-capable checker on another provider, the architect verifier
    // route must pick it over the same-provider Balanced checker the classic
    // ladder would choose.
    let inventory = ModelInventory::new(
        "claude-fable-5",
        vec![
            ModelDescriptor::new("claude-fable-5", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default, ModelCapability::Verification, ModelCapability::Analysis])
                .tiers([ModelTier::Deep, ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("gpt-5.6-sol", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Default, ModelCapability::Verification, ModelCapability::Analysis])
                .tiers([ModelTier::Deep, ModelTier::Strong])
                .release_rank(56),
            ModelDescriptor::new("gpt-5.6-terra", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Default, ModelCapability::Verification])
                .tiers([ModelTier::Balanced, ModelTier::Strong])
                .release_rank(56),
        ],
    );
    let mut context = neutral_route_context(true);
    context.policy = SmartPolicy::Architect;
    let decision = route_model(
        &RouteRequest::new(RouteRole::Verifier, "claude-fable-5").with_context(context),
        &inventory,
    );
    assert_eq!(
        decision.resolved_model, "gpt-5.6-sol",
        "architect verify must land on the cross-provider Deep checker"
    );
}

#[test]
fn smart_policy_settings_value_defaults_to_architect_and_honors_classic() {
    assert_eq!(SmartPolicy::from_settings_value(None), SmartPolicy::Architect);
    assert_eq!(
        SmartPolicy::from_settings_value(Some(&serde_json::json!("classic"))),
        SmartPolicy::Classic
    );
    assert_eq!(
        SmartPolicy::from_settings_value(Some(&serde_json::json!("architect"))),
        SmartPolicy::Architect
    );
    assert_eq!(
        SmartPolicy::from_settings_value(Some(&serde_json::json!("bogus"))),
        SmartPolicy::Architect,
        "unrecognized values fail to the documented live default"
    );
    assert_eq!(SmartPolicy::default(), SmartPolicy::Classic, "type default stays byte-identical");
}

#[test]
fn high_risk_or_large_complexity_verifier_route_escalates_to_a_strong_tier_checker() {
    // End-to-end (through `route_model`, not just the selector-ladder fn):
    // a Strong-tier verification-capable model only wins a Verifier route
    // when the situational escalation gate opens.
    let inventory = ModelInventory::new(
        "main-x",
        vec![
            ModelDescriptor::new("main-x", "x", "x")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("balanced-checker", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Balanced])
                .release_rank(10),
            ModelDescriptor::new("strong-checker", "x", "x")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Verification])
                .tiers([ModelTier::Strong])
                .release_rank(10),
        ],
    );

    // Low risk, non-Large complexity: byte-identical to pre-Phase-8 routing —
    // the Balanced-only selector never even considers the Strong-tier model.
    let low_risk = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-x")
            .with_context(RoutePolicyContext { risk: RouteTaskRisk::Low, ..neutral_route_context(false) }),
        &inventory,
    );
    assert_eq!(low_risk.resolved_model, "balanced-checker");

    // High risk: the Strong-tier selector is tried FIRST.
    let high_risk = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-x")
            .with_context(RoutePolicyContext { risk: RouteTaskRisk::High, ..neutral_route_context(false) }),
        &inventory,
    );
    assert_eq!(high_risk.resolved_model, "strong-checker");

    // Large complexity alone (risk left Unknown) also escalates.
    let large_complexity = route_model(
        &RouteRequest::new(RouteRole::Verifier, "main-x").with_context(RoutePolicyContext {
            complexity: RouteTaskComplexity::Large,
            ..neutral_route_context(false)
        }),
        &inventory,
    );
    assert_eq!(large_complexity.resolved_model, "strong-checker");

    // Reviewer shares the exact same escalation (same selector fn).
    let reviewer_high_risk = route_model(
        &RouteRequest::new(RouteRole::Reviewer, "main-x")
            .with_context(RoutePolicyContext { risk: RouteTaskRisk::Critical, ..neutral_route_context(false) }),
        &inventory,
    );
    assert_eq!(reviewer_high_risk.resolved_model, "strong-checker");
}

#[test]
fn learned_specialty_absent_entry_leaves_the_seed_score_unchanged() {
    // Phase 6 zero-data-identity contract, at the shared-blend-fn level: no
    // learned entry for this EXACT (role, model) pair must return the
    // cold-start seed UNCHANGED — not "close to" it — whether the hint is
    // fully empty or merely has no entry relevant to this pair.
    use super::policy::{cold_start_specialty_seed, effective_specialty_adjustment};

    let gpt = ModelDescriptor::new("x", "openai", "gpt");
    let seed_only = cold_start_specialty_seed(RouteRole::Coding, &gpt);

    let empty_hint = LearnedSpecialtyHint::disabled();
    assert_eq!(effective_specialty_adjustment(RouteRole::Coding, &gpt, &empty_hint), seed_only);

    // A hint that carries data, but none for THIS (role, model) pair, must
    // not leak in either — proves `entry_for`'s exact-match lookup, not just
    // "the whole hint object happened to be default".
    let unrelated_hint = LearnedSpecialtyHint::default()
        .with_entry(RouteRole::Analysis, "x", 90, 1000)
        .with_entry(RouteRole::Coding, "some-other-model", -90, 1000);
    assert_eq!(effective_specialty_adjustment(RouteRole::Coding, &gpt, &unrelated_hint), seed_only);
}

#[test]
fn learned_specialty_zero_data_route_is_byte_identical_to_seed_only_routing() {
    // End-to-end (through `route_model`, not just the scorer fn directly):
    // a route request carrying the DEFAULT (empty) learned-specialty hint
    // must resolve identically to one that never mentions the field at all.
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("gpt-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("deepseek-coder", "deepseek", "deepseek")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(51),
        ],
    );
    let bare_default = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(neutral_route_context(false)),
        &inventory,
    );
    let explicit_empty_hint = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(RoutePolicyContext {
            learned_specialty: LearnedSpecialtyHint::disabled(),
            ..neutral_route_context(false)
        }),
        &inventory,
    );
    assert_eq!(bare_default, explicit_empty_hint);
}

#[test]
fn learned_specialty_blend_worst_case_stays_under_the_tier_gate() {
    use super::learned::LEARNED_SPECIALTY_BOUND;
    use super::policy::{MAX_FEEDBACK_ADJUSTMENT, TIER_MATCH_BONUS};

    // A convex combination `seed * (1-c) + learned * c` over `seed in
    // [-60, 60]` and `learned in [-LEARNED_SPECIALTY_BOUND,
    // LEARNED_SPECIALTY_BOUND]` is maximized in magnitude at one of its
    // endpoints (c=0 or c=1) — so the worst-case |effective_specialty| is
    // `max(60, LEARNED_SPECIALTY_BOUND)` = `LEARNED_SPECIALTY_BOUND` (90,
    // since it exceeds the seed's 60). Stacked with the full outcome-feedback
    // swing (`MAX_FEEDBACK_ADJUSTMENT`, ±120) it must still land strictly
    // under the tier-selector gate (`TIER_MATCH_BONUS`, 300) — otherwise a
    // model that only qualifies for a WEAKER tier could out-score one that
    // actually matches the role's required tier. Both sides are compile-time
    // constants (the P5 `exploration_score_window_cannot_reach_a_lower_selector_rung`
    // pattern), so this is a genuine build-time invariant.
    const {
        assert!(
            (LEARNED_SPECIALTY_BOUND as i32) + (MAX_FEEDBACK_ADJUSTMENT as i32) < TIER_MATCH_BONUS,
            "learned-specialty + feedback worst case must stay under the tier gate"
        );
    }
}

#[test]
fn learned_specialty_ramp_blends_seed_and_learned_at_half_confidence() {
    use super::policy::effective_specialty_adjustment;

    // At the eligibility floor (4 weighted decisive samples out of an
    // 8-sample ramp denominator), confidence lands at exactly 500 permille
    // (c=0.5) — the natural boundary case `LearnedSpecialtyHint::compute`
    // produces. deepseek has no Coding seed (seed=0); a learned entry that
    // fully favors it (learned=+90) at c=0.5 must land HALFWAY between the
    // two: round(0 * 0.5 + 90 * 0.5) = 45.
    let deepseek = ModelDescriptor::new("z", "deepseek", "deepseek");
    let hint = LearnedSpecialtyHint::default().with_entry(RouteRole::Coding, "z", 90, 500);
    assert_eq!(effective_specialty_adjustment(RouteRole::Coding, &deepseek, &hint), 45);

    // claude DOES carry the Coding seed (60); a learned entry pulling it
    // toward -90 at c=0.5 must land at round(60*0.5 + (-90)*0.5) = -15.
    let claude = ModelDescriptor::new("y", "anthropic", "claude");
    let pulling_down = LearnedSpecialtyHint::default().with_entry(RouteRole::Coding, "y", -90, 500);
    assert_eq!(effective_specialty_adjustment(RouteRole::Coding, &claude, &pulling_down), -15);
}

#[test]
fn learned_specialty_blend_agrees_between_live_route_and_dashboard_scorer() {
    // Scorer-parity gate (task 3): a shared fixture (role × candidates ×
    // learned data) must produce the SAME ranking on the live route scorer
    // (`policy::route_model`) and the dashboard recommendation scorer
    // (`assignment::recommend_role_fallbacks_with_learned_specialty`) — the
    // two must never silently disagree, since both call the SAME
    // `effective_specialty_adjustment` fn.
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            // Neither family is Coding-seeded (`cold_start_specialty_seed`'s
            // Coding preference list is `gpt`/`claude` only) — deliberately,
            // so the ONLY thing separating them at baseline is release_rank,
            // isolating the learned-entry effect asserted below.
            ModelDescriptor::new("model-a", "xai", "grok")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("model-b", "deepseek", "deepseek")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(10),
        ],
    );

    // Sanity: without learned data, model-a's release_rank (50) beats
    // model-b's (10) — this isolates the effect of the learned entry added
    // below.
    let baseline = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(neutral_route_context(false)),
        &inventory,
    );
    assert_eq!(baseline.resolved_model, "model-a");

    let learned = LearnedSpecialtyHint::default().with_entry(RouteRole::Coding, "model-b", 90, 1000);
    let live = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(RoutePolicyContext {
            learned_specialty: learned.clone(),
            ..neutral_route_context(false)
        }),
        &inventory,
    );
    assert_eq!(live.resolved_model, "model-b", "full-confidence learned entry must flip the live route");

    let dashboard = recommend_role_fallbacks_with_learned_specialty(
        &inventory,
        &AutoAssignmentOptions { allow_cross_provider_diversity: false, provider_allowlist: Vec::new(), policy: SmartPolicy::Classic },
        &learned,
    );
    let coding = dashboard
        .iter()
        .find(|assignment| assignment.target == RoutingTarget::RoleFallback(RouteRole::Coding))
        .expect("coding fallback");
    assert_eq!(
        coding.selected_model, live.resolved_model,
        "dashboard recommendation must agree with the live route on the learned-flipped model"
    );
}

#[test]
fn parallel_lanes_spread_across_near_best_models() {
    // Three Coding-capable models with near-enough scores (same tier/capability,
    // cold-start seed/release-rank gaps within LANE_SPREAD_SCORE_WINDOW): a
    // 4-lane fan-out must spread while still excluding the first model outside
    // the near-best window.
    let inventory = ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("coder-a", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(10),
            ModelDescriptor::new("coder-b", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(12),
            // Qwen has no Coding cold-start family seed, so it used to sit just
            // outside the narrow lane-spread window. It is still a comparable
            // Strong coder and should get one sibling lane.
            ModelDescriptor::new("coder-qwen", "gme-litellm", "qwen")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(15),
            ModelDescriptor::new("too-weak-balanced", "local", "custom")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Balanced])
                .release_rank(100),
            ModelDescriptor::new("outside-window", "local", "custom")
                .source(ModelSource::CustomProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(1),
        ],
    );
    let lane_request = |index: usize| {
        let mut context = neutral_route_context(true);
        context.lane = Some(LaneRouteMetadata::new("backend").with_position(index, 4));
        RouteRequest::new(RouteRole::Coding, "main-model").with_context(context)
    };

    let picks: Vec<String> = (0..4)
        .map(|index| route_model(&lane_request(index), &inventory).resolved_model)
        .collect();

    let distinct: std::collections::BTreeSet<&String> = picks.iter().collect();
    assert_eq!(
        distinct.len(),
        3,
        "4 parallel lanes should rotate across the three near-best models: {picks:?}"
    );
    assert!(picks.iter().any(|model| model == "coder-qwen"), "qwen lane missing: {picks:?}");
    assert!(
        picks.iter().all(|model| model != "too-weak-balanced" && model != "outside-window"),
        "lane spread must stay within the active selector and score window: {picks:?}",
    );
    // Without lane metadata the selector keeps its argmax pick (regression
    // guard: non-fan-out routes are untouched by the spread).
    let solo = RouteRequest::new(RouteRole::Coding, "main-model")
        .with_context(neutral_route_context(true));
    let first = route_model(&solo, &inventory).resolved_model;
    let second = route_model(&solo, &inventory).resolved_model;
    assert_eq!(first, second, "solo routing must stay deterministic");
}

#[test]
fn pinned_model_outside_provider_allowlist_is_flagged_not_blocked() {
    let mut context = neutral_route_context(true);
    context.provider_allowlist = vec!["anthropic".to_string()];
    let mut request =
        RouteRequest::new(RouteRole::Coding, "claude-sonnet-main").with_context(context);
    request.override_rule = Some(RoleOverride::Pin("gpt-coder".to_string()));

    let decision = route_model(&request, &inventory());

    // The pin is explicit user intent: it still wins…
    assert_eq!(decision.resolved_model, "gpt-coder");
    assert_eq!(decision.source, RouteDecisionSource::Pinned);
    // …but the allowlist escape is named, in both the reason and the audit.
    assert!(
        decision.reason.contains("outside smart.providerAllowlist"),
        "reason must surface the escape: {}",
        decision.reason
    );
    assert!(
        decision
            .audit
            .guardrails
            .iter()
            .any(|line| line.contains("provider-allowlist-escape:openai")),
        "audit must record the escape: {:?}",
        decision.audit.guardrails
    );

    // An in-list pin stays unflagged.
    let mut context = neutral_route_context(true);
    context.provider_allowlist = vec!["openai".to_string()];
    let mut request =
        RouteRequest::new(RouteRole::Coding, "claude-sonnet-main").with_context(context);
    request.override_rule = Some(RoleOverride::Pin("gpt-coder".to_string()));
    let decision = route_model(&request, &inventory());
    assert_eq!(decision.reason, "role exact pin");
}

fn cooldown_test_inventory() -> ModelInventory {
    ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "local", "main")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            // Equal score rivals (same tier/capability/rank) so the ONLY thing
            // that can distinguish them is inventory order (no cooldown) vs.
            // the cooldown penalty.
            ModelDescriptor::new("healthy-coder", "healthy-provider", "healthy")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
            ModelDescriptor::new("cooled-coder", "cooled-provider", "cooled")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
        ],
    )
}

#[test]
fn cooldown_provider_loses_a_tie_to_a_healthy_rival_and_stamps_quota_degraded() {
    let inventory = cooldown_test_inventory();

    // Baseline (no cooldown): the equal-score tie resolves to the LATER
    // inventory entry (`max_by_key` last-of-equal-maxima) — `cooled-coder`.
    let baseline = route_model(&RouteRequest::new(RouteRole::Coding, "main-model"), &inventory);
    assert_eq!(baseline.resolved_model, "cooled-coder");
    assert!(
        !baseline.audit.guardrails.iter().any(|g| g == "quota-degraded"),
        "no cooldown set → no quota-degraded stamp: {:?}",
        baseline.audit.guardrails
    );

    // With `cooled-provider` in cooldown, the soft deprioritization flips the
    // tie to the healthy rival — NOT a hard filter, just enough to lose a tie
    // — and the decision is stamped `quota-degraded` because the pick changed.
    let degraded = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            cooldown_providers: vec!["cooled-provider".to_string()],
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(degraded.resolved_model, "healthy-coder");
    assert!(
        degraded.audit.guardrails.iter().any(|g| g == "quota-degraded"),
        "cooldown-changed pick must be stamped: {:?}",
        degraded.audit.guardrails
    );
}

#[test]
fn empty_cooldown_set_is_byte_identical_to_no_cooldown_context() {
    let inventory = cooldown_test_inventory();
    let request = RouteRequest::new(RouteRole::Coding, "main-model");
    let plain = route_model(&request, &inventory);
    let with_empty_cooldown = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            cooldown_providers: Vec::new(),
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(plain, with_empty_cooldown);
}

/// P1 identity contract (mirrors `empty_cooldown_set_is_byte_identical…`): an
/// EMPTY `provider_headroom` is byte-identical to no-headroom routing — even
/// with a non-default threshold set, because the graded penalty branch is never
/// entered without headroom data.
#[test]
fn empty_headroom_set_is_byte_identical_to_no_headroom_context() {
    let inventory = cooldown_test_inventory();
    let request = RouteRequest::new(RouteRole::Coding, "main-model");
    let plain = route_model(&request, &inventory);
    let with_empty_headroom = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            provider_headroom: Vec::new(),
            headroom_penalty_threshold: 25,
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(plain, with_empty_headroom);
}

/// P1 graded headroom penalty: a provider whose remaining headroom is below the
/// threshold loses an otherwise-equal tie (the baseline tie resolves to the
/// LATER inventory entry, `cooled-coder`), exactly like the binary cooldown but
/// driven by the leading remaining-percent signal — and stays a soft nudge, so
/// a healthy (at-threshold) reading keeps the baseline pick.
#[test]
fn graded_headroom_penalty_flips_a_tie_below_threshold_only() {
    let inventory = cooldown_test_inventory();
    let baseline = route_model(&RouteRequest::new(RouteRole::Coding, "main-model"), &inventory);
    assert_eq!(baseline.resolved_model, "cooled-coder");

    // 5% remaining vs a 25% threshold ⇒ a real penalty ⇒ the tie flips to the
    // healthy rival.
    let squeezed = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            provider_headroom: vec![("cooled-provider".to_string(), 5)],
            headroom_penalty_threshold: 25,
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(squeezed.resolved_model, "healthy-coder");

    // Exactly at the threshold ⇒ no penalty ⇒ the baseline pick survives.
    let at_threshold = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            provider_headroom: vec![("cooled-provider".to_string(), 25)],
            headroom_penalty_threshold: 25,
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(at_threshold.resolved_model, "cooled-coder");
}

/// P1 no-double-penalty: a provider in BOTH the cooldown set AND at `0%`
/// headroom is deprioritized by the binary cooldown ONLY (the scorer's
/// `if cooldown { -30 } else { headroom }` skips the graded branch), so adding
/// the `0%` headroom entry to an already-cooled provider leaves the pick exactly
/// where the cooldown alone put it — never a compounded penalty. (The exact
/// arithmetic no-double is proven in `policy::headroom_penalty_tests`.)
///
/// NOTE: we assert on the resolved model, NOT the `quota-degraded` audit stamp:
/// that stamp comes from a neutral re-rank that clears only `cooldown_providers`
/// (not `provider_headroom`), so the two contexts legitimately attribute it
/// differently — an out-of-scope quirk of `select_auto_model_with_quota_note`,
/// not the penalty stacking this test is about.
#[test]
fn cooldown_takes_precedence_over_graded_headroom() {
    let inventory = cooldown_test_inventory();
    let cooldown_only = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            cooldown_providers: vec!["cooled-provider".to_string()],
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    let cooldown_and_headroom = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            cooldown_providers: vec!["cooled-provider".to_string()],
            provider_headroom: vec![("cooled-provider".to_string(), 0)],
            headroom_penalty_threshold: 25,
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(cooldown_only.resolved_model, "healthy-coder");
    assert_eq!(
        cooldown_and_headroom.resolved_model, cooldown_only.resolved_model,
        "adding a 0% headroom entry for an already-cooled provider must not change the pick"
    );
}

#[test]
fn cooldown_never_hard_filters_the_only_qualifying_provider() {
    // A cooled-down provider that is the ONLY candidate for a role still wins
    // — the penalty degrades the pick, it never disqualifies it.
    let inventory = ModelInventory::new(
        "main-model",
        vec![
            ModelDescriptor::new("main-model", "local", "main")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("only-coder", "only-provider", "only")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(50),
        ],
    );
    let decision = route_model(
        &RouteRequest::new(RouteRole::Coding, "main-model").with_context(RoutePolicyContext {
            cooldown_providers: vec!["only-provider".to_string()],
            ..RoutePolicyContext::default()
        }),
        &inventory,
    );
    assert_eq!(decision.resolved_model, "only-coder");
}

#[test]
fn recommended_effort_for_route_fires_only_for_analysis_family_large_and_ultra_ceiling() {
    // The rule: Analysis-family role + Large complexity + Ultra ceiling ⇒
    // recommend Ultra. Every other combination ⇒ None.
    assert_eq!(
        recommended_effort_for(RouteRole::Analysis, RouteTaskComplexity::Large, EffortCeiling::Ultra),
        Some(EffortCeiling::Ultra)
    );
    assert_eq!(
        recommended_effort_for(RouteRole::Research, RouteTaskComplexity::Large, EffortCeiling::Ultra),
        Some(EffortCeiling::Ultra)
    );
    assert_eq!(
        recommended_effort_for(RouteRole::Judge, RouteTaskComplexity::Large, EffortCeiling::Ultra),
        Some(EffortCeiling::Ultra)
    );
    assert_eq!(
        recommended_effort_for(RouteRole::Synthesizer, RouteTaskComplexity::Large, EffortCeiling::Ultra),
        Some(EffortCeiling::Ultra)
    );
    // Non-analysis role: no recommendation even at Large + Ultra.
    assert_eq!(
        recommended_effort_for(RouteRole::Coding, RouteTaskComplexity::Large, EffortCeiling::Ultra),
        None
    );
    // Analysis family but not Large: no recommendation.
    assert_eq!(
        recommended_effort_for(RouteRole::Analysis, RouteTaskComplexity::Medium, EffortCeiling::Ultra),
        None
    );
    // Analysis family + Large but ceiling only Max (not Ultra): no recommendation.
    assert_eq!(
        recommended_effort_for(RouteRole::Analysis, RouteTaskComplexity::Large, EffortCeiling::Max),
        None
    );
}

#[test]
fn route_decision_carries_recommended_effort_for_a_deep_ultra_ceiling_pick() {
    let inventory = ModelInventory::new(
        "claude-sonnet-main",
        vec![
            ModelDescriptor::new("claude-sonnet-main", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("gpt-5.6-sol", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Analysis])
                .tiers([ModelTier::Deep, ModelTier::Strong])
                .effort_ceiling(EffortCeiling::Ultra)
                .release_rank(56),
        ],
    );
    let request = RouteRequest::new(RouteRole::Analysis, "claude-sonnet-main").with_context(RoutePolicyContext {
        complexity: RouteTaskComplexity::Large,
        ..RoutePolicyContext::default()
    });
    let decision = route_model(&request, &inventory);
    assert_eq!(decision.resolved_model, "gpt-5.6-sol");
    assert_eq!(decision.recommended_effort, Some(EffortCeiling::Ultra));

    // The SAME model at Medium complexity gets no effort recommendation.
    let medium_request = RouteRequest::new(RouteRole::Analysis, "claude-sonnet-main").with_context(RoutePolicyContext {
        complexity: RouteTaskComplexity::Medium,
        ..RoutePolicyContext::default()
    });
    let medium_decision = route_model(&medium_request, &inventory);
    assert_eq!(medium_decision.recommended_effort, None);
}

// ---------------------------------------------------------------------
// Phase 5 — deterministic risk-gated exploration
// ---------------------------------------------------------------------

/// Single-selector-rung (Coding, Strong tier) fixture with an established
/// incumbent and a comparably-scored, thin-history challenger — both within
/// [`super::policy::EXPLORATION_SCORE_WINDOW`] of each other so exploration's
/// window/decisive-count filtering is the only thing distinguishing them.
fn coding_exploration_inventory() -> ModelInventory {
    ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("incumbent-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(90),
            ModelDescriptor::new("challenger-coder", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(80),
        ],
    )
}

fn coding_exploration_request(context: RoutePolicyContext) -> RouteRequest {
    RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(context)
}

#[test]
fn exploration_slot_for_route_fires_only_on_cadence_multiples() {
    // Determinism: a fixed fixture (top incumbent confident, an
    // under-sampled rival exists) fires at EXACT cadence multiples and only
    // those — 0, 5, 10 fire; 1-4, 6-9 do not.
    let cadence = 5;
    let fired: Vec<usize> = (0..=10)
        .filter(|&total| {
            exploration_slot_for_route(total, cadence, 8, true, RouteTaskRisk::Low, RouteRole::Coding, true).is_some()
        })
        .collect();
    assert_eq!(fired, vec![0, 5, 10], "must fire at exactly the cadence multiples in 0..=10");

    // The returned slot is `total / cadence` — a monotonically increasing
    // generation counter, not the raw record count itself.
    assert_eq!(
        exploration_slot_for_route(0, cadence, 8, true, RouteTaskRisk::Low, RouteRole::Coding, true),
        Some(0)
    );
    assert_eq!(
        exploration_slot_for_route(5, cadence, 8, true, RouteTaskRisk::Low, RouteRole::Coding, true),
        Some(1)
    );
    assert_eq!(
        exploration_slot_for_route(10, cadence, 8, true, RouteTaskRisk::Low, RouteRole::Coding, true),
        Some(2)
    );
}

#[test]
fn exploration_slot_for_route_hard_gates_never_explore() {
    // Baseline: this combination fires (sanity check that every gate below
    // is actually the thing flipping the result to `None`).
    let fires = |risk, role, enabled| {
        exploration_slot_for_route(5, 5, 8, true, risk, role, enabled)
    };
    assert!(fires(RouteTaskRisk::Low, RouteRole::Coding, true).is_some());

    // Safety-sensitive routes never explore.
    assert_eq!(fires(RouteTaskRisk::High, RouteRole::Coding, true), None);
    assert_eq!(fires(RouteTaskRisk::Critical, RouteRole::Coding, true), None);

    // Judging roles never explore (exploring INTO the thing that checks
    // everyone else's work would undermine the check itself).
    assert_eq!(fires(RouteTaskRisk::Low, RouteRole::Verifier, true), None);
    assert_eq!(fires(RouteTaskRisk::Low, RouteRole::Reviewer, true), None);
    assert_eq!(fires(RouteTaskRisk::Low, RouteRole::Judge, true), None);

    // Master switch off.
    assert_eq!(fires(RouteTaskRisk::Low, RouteRole::Coding, false), None);

    // Incumbent not yet confident (< 8 decisive) — exploring before any
    // incumbent is established would just be noise.
    assert_eq!(
        exploration_slot_for_route(5, 5, 7, true, RouteTaskRisk::Low, RouteRole::Coding, true),
        None
    );

    // No under-sampled rival at all.
    assert_eq!(
        exploration_slot_for_route(5, 5, 8, false, RouteTaskRisk::Low, RouteRole::Coding, true),
        None
    );
}

#[test]
fn exploration_score_window_cannot_reach_a_lower_selector_rung() {
    use super::policy::{AUTO_SELECTOR_FALLBACK_PENALTY, EXPLORATION_SCORE_WINDOW, MAX_FEEDBACK_ADJUSTMENT};

    // The rung boundary between adjacent AUTO selector fallback tiers is a
    // constant multiple of `AUTO_SELECTOR_FALLBACK_PENALTY` (1,000,000),
    // applied UNIFORMLY to every candidate in that rung — independent of a
    // candidate's underlying feature scores. Two candidates in ADJACENT
    // rungs are therefore separated by at least `AUTO_SELECTOR_FALLBACK_
    // PENALTY` minus a generous overestimate of the maximum plausible
    // in-rung score spread (every documented positive scorer contribution
    // summed is nowhere near this). `EXPLORATION_SCORE_WINDOW` structurally
    // cannot bridge that gap — this is the invariant
    // `select_exploration_candidate` relies on instead of tracking a
    // separate "rung id" per candidate; see
    // `exploration_never_reaches_a_lower_selector_rung` for the same
    // invariant proven end-to-end through `route_model`.
    const GENEROUS_MAX_IN_RUNG_SCORE_SPREAD: i32 = 3_000;
    // Both sides are compile-time constants, so this is a genuine build-time
    // invariant, not merely a test-time check — an inline const block makes
    // that explicit (and satisfies clippy's `assertions_on_constants`).
    const {
        assert!(
            EXPLORATION_SCORE_WINDOW < AUTO_SELECTOR_FALLBACK_PENALTY - GENEROUS_MAX_IN_RUNG_SCORE_SPREAD,
            "exploration window must stay far below the selector-rung gap"
        );
    }
    // And the window must clear the full outcome-feedback swing an
    // incumbent can carry (±MAX_FEEDBACK_ADJUSTMENT) — the live-data problem
    // this phase exists to fix: an established incumbent's feedback bound
    // must not outrun the exploration window the way it outran
    // `LANE_SPREAD_SCORE_WINDOW` (65 < 120).
    const {
        assert!(
            (MAX_FEEDBACK_ADJUSTMENT as i32) < EXPLORATION_SCORE_WINDOW,
            "exploration window must clear the full outcome-feedback clamp"
        );
    }
}

#[test]
fn exploration_rotates_into_an_undersampled_same_rung_candidate() {
    let inventory = coding_exploration_inventory();
    let request = coding_exploration_request(RoutePolicyContext {
        exploration_slot: Some(0),
        // Incumbent is fully confident (10 decisive); the challenger is
        // absent (0 decisive, the common brand-new-model case) — the exact
        // live-data shape this phase targets.
        exploration_decisive_counts: vec![("incumbent-coder".to_string(), 10)],
        ..RoutePolicyContext::default()
    });

    // Sanity: without exploration, the incumbent's higher release_rank wins
    // the plain argmax.
    let plain = route_model(&coding_exploration_request(RoutePolicyContext::default()), &inventory);
    assert_eq!(plain.resolved_model, "incumbent-coder");
    assert_eq!(plain.source, RouteDecisionSource::AutoSelector);

    let decision = route_model(&request, &inventory);
    assert_eq!(decision.resolved_model, "challenger-coder");
    assert_eq!(decision.source, RouteDecisionSource::Exploration);
    assert!(
        decision.audit.guardrails.iter().any(|line| line == "exploration"),
        "audit must record the exploration guardrail — `smart_router::apply::compose_route_reason` \
         (tools crate) turns this into the human-readable routeReason's `· exploration` suffix: {:?}",
        decision.audit.guardrails
    );
}

#[test]
fn exploration_falls_through_to_the_normal_winner_when_nothing_qualifies() {
    let inventory = coding_exploration_inventory();
    // Both models already have >= 2 decisive samples — no under-sampled
    // rival exists, so the slot is wasted (no error, no forced pick).
    let request = coding_exploration_request(RoutePolicyContext {
        exploration_slot: Some(0),
        exploration_decisive_counts: vec![
            ("incumbent-coder".to_string(), 10),
            ("challenger-coder".to_string(), 5),
        ],
        ..RoutePolicyContext::default()
    });

    let decision = route_model(&request, &inventory);
    assert_eq!(decision.resolved_model, "incumbent-coder");
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
    assert!(!decision.audit.guardrails.iter().any(|line| line == "exploration"));
}

#[test]
fn exploration_never_explores_into_a_cooldown_provider() {
    let inventory = coding_exploration_inventory();
    // The challenger is under-sampled (would normally be explored into) but
    // its provider (`anthropic`) is in cooldown — a hard exclusion, not a
    // soft deprioritization, so exploration must skip it entirely rather
    // than risk poisoning the very sample this phase exists to collect.
    let request = coding_exploration_request(RoutePolicyContext {
        exploration_slot: Some(0),
        exploration_decisive_counts: vec![("incumbent-coder".to_string(), 10)],
        cooldown_providers: vec!["anthropic".to_string()],
        ..RoutePolicyContext::default()
    });

    let decision = route_model(&request, &inventory);
    assert_eq!(
        decision.resolved_model, "incumbent-coder",
        "cooldown provider must never be explored INTO"
    );
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
}

#[test]
fn exploration_hard_gates_apply_even_if_a_caller_sets_the_slot_directly() {
    // Defense in depth: `select_exploration_candidate` re-checks risk/role
    // itself (not just `apply.rs`'s `exploration_slot_for_route`) — even a
    // context that sets `exploration_slot` directly (bypassing the gated
    // computation) must never explore for a High-risk task or a judging role.
    let inventory = coding_exploration_inventory();
    let exploration_context = RoutePolicyContext {
        exploration_slot: Some(0),
        exploration_decisive_counts: vec![("incumbent-coder".to_string(), 10)],
        ..RoutePolicyContext::default()
    };

    let high_risk = RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(RoutePolicyContext {
        risk: RouteTaskRisk::High,
        ..exploration_context.clone()
    });
    let decision = route_model(&high_risk, &inventory);
    assert_eq!(decision.resolved_model, "incumbent-coder");
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector, "High risk must never explore");

    let verifier = RouteRequest::new(RouteRole::Verifier, "main-anthropic").with_context(exploration_context);
    let decision = route_model(&verifier, &inventory);
    assert_ne!(decision.source, RouteDecisionSource::Exploration, "Verifier role must never explore");
}

#[test]
fn exploration_never_reaches_a_lower_selector_rung() {
    // Adversarial worst case: the top rung (Strong tier) winner has the
    // WEAKEST possible in-rung score (no specialty-seed family, minimum
    // release_rank), while a lower rung (Balanced tier) candidate has the
    // STRONGEST possible in-rung score (seeded family, maximum release_rank)
    // AND is flagged maximally attractive to exploration (0 decisive
    // samples). Even so, the lower-rung candidate must never be picked —
    // `AUTO_SELECTOR_FALLBACK_PENALTY` (1,000,000) separates the rungs by far
    // more than `EXPLORATION_SCORE_WINDOW` (200) could ever bridge (see
    // `exploration_score_window_cannot_reach_a_lower_selector_rung`'s
    // arithmetic proof of the same invariant).
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            // Top rung (Strong tier): the ONLY Strong-tier candidate, given
            // the weakest possible in-rung score (deepseek has no Coding
            // specialty seed; release_rank 1).
            ModelDescriptor::new("weak-strong-tier", "deepseek", "deepseek")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(1),
            // Lower rung (Balanced tier): maximal in-rung score (gpt has a
            // Coding specialty seed; release_rank 100) — deliberately more
            // attractive than the top-rung winner by every non-rung signal.
            ModelDescriptor::new("strong-balanced-tier", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Balanced])
                .release_rank(100),
        ],
    );
    let request = coding_exploration_request(RoutePolicyContext {
        exploration_slot: Some(0),
        // The top-rung winner is itself sampled (excluded on decisive
        // count), and the lower-rung candidate is flagged as having ZERO
        // decisive samples — maximally attractive to exploration by every
        // signal EXCEPT the rung it lives in.
        exploration_decisive_counts: vec![("weak-strong-tier".to_string(), 10)],
        ..RoutePolicyContext::default()
    });

    let decision = route_model(&request, &inventory);
    assert_eq!(
        decision.resolved_model, "weak-strong-tier",
        "exploration must never cross into a lower selector rung, however attractive the rival"
    );
    assert_eq!(decision.source, RouteDecisionSource::AutoSelector);
}

#[test]
fn exploration_rotates_deterministically_across_multiple_eligible_rivals() {
    // Two under-sampled rivals in-window: successive exploration generations
    // (slot 0, 1, 2, 3, ...) rotate between them in a fixed, reproducible
    // order — NO randomness. The same slot value always yields the same pick.
    let inventory = ModelInventory::new(
        "main-anthropic",
        vec![
            ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                .source(ModelSource::CurrentMainModel)
                .capabilities([ModelCapability::Default])
                .tiers([ModelTier::Balanced])
                .release_rank(1),
            ModelDescriptor::new("incumbent-coder", "openai", "gpt")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(90),
            ModelDescriptor::new("rival-a", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(80),
            ModelDescriptor::new("rival-b", "anthropic", "claude")
                .source(ModelSource::EnabledBuiltinProvider)
                .capabilities([ModelCapability::Coding])
                .tiers([ModelTier::Strong])
                .release_rank(70),
        ],
    );
    let decisive_counts = vec![("incumbent-coder".to_string(), 10)];
    let pick_for_slot = |slot: u32| {
        let request = coding_exploration_request(RoutePolicyContext {
            exploration_slot: Some(slot),
            exploration_decisive_counts: decisive_counts.clone(),
            ..RoutePolicyContext::default()
        });
        route_model(&request, &inventory).resolved_model
    };

    let picks: Vec<String> = (0..4).map(pick_for_slot).collect();
    assert_eq!(
        picks,
        vec!["rival-a", "rival-b", "rival-a", "rival-b"],
        "rotation must be deterministic and reproducible across repeated calls: {picks:?}"
    );
}
