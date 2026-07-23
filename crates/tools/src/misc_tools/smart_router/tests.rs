
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;
    use crate::misc_tools::{AgentInput, SpawnMultiAgentInput};
    use runtime::{
        RouteAutoClassifierMode, RouteConfidence, RouteContextNeed, RouteOutputNeed, RouteRole,
        RouteShapeKind, RouteSignalSource, RouteTaskComplexity, RouteTaskKind, RouteTaskRisk,
        RouteToolNeed, RouteVerificationNeed,
    };
    use serde_json::Value;

    use super::evidence::{
        infer_route_shape_evidence, shape_input_with_evidence, RouteEvidenceInput,
    };
    use super::apply::{
        apply_smart_models_to_spawn_input, routing_subagent_type,
        smart_parent_model_for_agent,
    };
    use super::infer::{infer_route_role, task_has_write_intent};
    use super::metadata::{
        apply_probe_to_metadata, classify_task_metadata, TaskMetadataInput,
    };
    use super::planner::plan_agent_needs;
    use super::turn::assess_turn_orchestration;
    use super::settings::read_smart_runtime_settings;
    use super::shape::{
        select_route_shape, RouteShapeInput, UserOrchestrationRequestOutcome,
    };

    fn env_lock() -> &'static Mutex<()> {
        // Crate-wide lock: this module scopes ZO_CONFIG_HOME / ZO_HOME,
        // which memory and permission tests in OTHER modules also mutate — a
        // module-local mutex excluded none of them, and the wandering
        // memory_write_local_targets flake was this module's config-home
        // override landing mid-write in that test.
        crate::tests::env_lock()
    }

    fn temp_config_home(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zo-tools-smart-router-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn with_config_home<T>(config_home: &std::path::Path, run: impl FnOnce() -> T) -> T {
        // Recover from a poisoned lock instead of cascade-panicking: one test that
        // fails its assertions must not turn into a wall of unrelated "PoisonError"
        // failures across every other env-scoped test in this module.
        let _guard = env_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior_config_home = std::env::var_os("ZO_CONFIG_HOME");
        let prior_zo_home = std::env::var_os("ZO_HOME");
        let prior_custom_providers = std::env::var_os(api::CUSTOM_PROVIDERS_ENV);
        std::env::set_var("ZO_CONFIG_HOME", config_home);
        std::env::remove_var("ZO_HOME");
        std::env::set_var(
            api::CUSTOM_PROVIDERS_ENV,
            r#"[{"name":"Verifier","base_url":"http://verifier.local/v1","models":["Verifier/Model-X","Role/Model-Y","Subagent/Model-Z"],"requires_auth":false}]"#,
        );
        api::refresh_custom_providers_from_env();
        let output = run();
        match prior_config_home {
            Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
            None => std::env::remove_var("ZO_CONFIG_HOME"),
        }
        match prior_zo_home {
            Some(value) => std::env::set_var("ZO_HOME", value),
            None => std::env::remove_var("ZO_HOME"),
        }
        match prior_custom_providers {
            Some(value) => std::env::set_var(api::CUSTOM_PROVIDERS_ENV, value),
            None => std::env::remove_var(api::CUSTOM_PROVIDERS_ENV),
        }
        api::refresh_custom_providers_from_env();
        output
    }

    #[test]
    fn builtin_profile_keys_use_the_profile_role_mapping() {
        for profile in runtime::BuiltinSubagentProfile::all() {
            if profile.key() == "general-purpose" {
                continue;
            }
            assert_eq!(
                infer_route_role(Some(profile.key()), "", ""),
                profile.route_role(),
                "{} must route through its builtin profile mapping",
                profile.key(),
            );
        }
    }

    #[test]
    fn auto_general_type_keeps_legacy_text_routing() {
        assert_eq!(
            routing_subagent_type(Some("general-purpose"), Some("general-purpose")),
            None
        );
        assert_eq!(
            routing_subagent_type(Some("general-purpose"), None),
            Some("general-purpose")
        );
        assert_eq!(
            routing_subagent_type(Some("Explore"), Some("Explore")),
            Some("Explore")
        );
    }

    fn write_settings(config_home: &std::path::Path, value: &Value) {
        fs::create_dir_all(config_home).expect("config dir");
        fs::write(
            config_home.join("settings.json"),
            serde_json::to_string_pretty(&value).expect("json"),
        )
        .expect("write settings");
    }

    fn agent_input(subagent_type: Option<&str>) -> AgentInput {
        AgentInput {
            allow_cross_provider: false,
            description: "verify the code".to_string(),
            prompt: "run verification".to_string(),
            subagent_type: subagent_type.map(str::to_string),
            name: None,
            model: None,
            cwd: None,
            schema: None,
            background: Some(false),
            workflow_member: false,
            api_concurrency: None,
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            time_budget: None,
            prior_failures: 0,
            route_reason: None,
            route_role: None,
            route_complexity: None,
            route_risk: None,
            route_source: None,
            route_model: None,
            route_fallback_models: Vec::new(),
            route_effort: None,
            judged_agent: None,
        }
    }

    #[test]
    fn route_shape_evidence_extracts_parallel_lanes_without_selecting_models() {
        let evidence = infer_route_shape_evidence(&RouteEvidenceInput {
            subagent_type: Some("general-purpose"),
            name: Some("parser"),
            description: "implement parser lane",
            prompt: "Split parser/executor/docs into three separate lanes and use parallel worktrees.",
            workflow_member: true,
            fanout_position: Some((1, 3)),
            auto_classifier: RouteAutoClassifierMode::Deterministic,
        });

        assert_eq!(evidence.requested_shape, Some(RouteShapeKind::ParallelLanes));
        assert_eq!(evidence.independent_lanes, 3);
        let lane = evidence.lane.expect("lane metadata");
        assert_eq!(lane.domain, "parser");
        assert_eq!(lane.lane_index, Some(1));
        assert_eq!(lane.lane_count, Some(3));
        assert!(evidence
            .audit_notes
            .iter()
            .any(|note| note == "smart-route-independent-lanes:3"));
    }

    #[test]
    fn route_shape_evidence_changes_shape_only_when_there_is_real_agent_need() {
        let metadata = classify_task_metadata(
            &TaskMetadataInput::new(
                Some("general-purpose"),
                "implement workflow route parser",
                "Split parser/executor/docs into parallel lanes and edit the workflow contract.",
            )
            .with_workflow_member(true),
            RouteRole::Coding,
        );
        let needs = plan_agent_needs(&metadata);
        let evidence = infer_route_shape_evidence(&RouteEvidenceInput {
            subagent_type: Some("general-purpose"),
            name: Some("parser"),
            description: "implement parser lane",
            prompt: "Split parser/executor/docs into parallel lanes and edit the workflow contract.",
            workflow_member: true,
            fanout_position: Some((0, 3)),
            auto_classifier: RouteAutoClassifierMode::Deterministic,
        });
        let shape = select_route_shape(&shape_input_with_evidence(&metadata, &needs, &evidence));
        assert_eq!(shape.shape, RouteShapeKind::ParallelLanes);
        assert_eq!(shape.outcome, UserOrchestrationRequestOutcome::AcceptRequestedShape);

        let trivial_metadata = classify_task_metadata(
            &TaskMetadataInput::new(
                Some("general-purpose"),
                "small label typo",
                "Split this typo into parallel lanes but only answer the spelling fix.",
            ),
            RouteRole::Default,
        );
        let trivial_needs = plan_agent_needs(&trivial_metadata);
        // A bare text "parallel lanes" mention (requested_shape) with NO real
        // fan-out structure (fanout_position=None → independent_lanes=0) and no
        // agent-need plan stays Solo: a weak text signal does not over-orchestrate
        // trivial work. (An actual N-way fan-out, which carries fanout_position,
        // does route per-lane — see `route_shape_routes_fanout_members_without_needs`.)
        let trivial_evidence = infer_route_shape_evidence(&RouteEvidenceInput {
            subagent_type: Some("general-purpose"),
            name: Some("docs"),
            description: "small label typo",
            prompt: "Split this typo into parallel lanes but only answer the spelling fix.",
            workflow_member: false,
            fanout_position: None,
            auto_classifier: RouteAutoClassifierMode::Deterministic,
        });
        let trivial_shape = select_route_shape(&shape_input_with_evidence(
            &trivial_metadata,
            &trivial_needs,
            &trivial_evidence,
        ));
        assert_eq!(trivial_shape.shape, RouteShapeKind::Solo);
    }


    #[test]
    fn assisted_classifier_adds_provider_free_lane_evidence() {
        let deterministic = infer_route_shape_evidence(&RouteEvidenceInput {
            subagent_type: Some("general-purpose"),
            name: Some("planner"),
            description: "plan the work",
            prompt: "Workstreams: parser, executor, docs.",
            workflow_member: true,
            fanout_position: None,
            auto_classifier: RouteAutoClassifierMode::Deterministic,
        });
        assert_eq!(deterministic.requested_shape, None);
        assert_eq!(deterministic.independent_lanes, 0);

        let assisted = infer_route_shape_evidence(&RouteEvidenceInput {
            subagent_type: Some("general-purpose"),
            name: Some("planner"),
            description: "plan the work",
            prompt: "Workstreams: parser, executor, docs.",
            workflow_member: true,
            fanout_position: None,
            auto_classifier: RouteAutoClassifierMode::Assisted,
        });
        assert_eq!(assisted.requested_shape, Some(RouteShapeKind::ParallelLanes));
        assert_eq!(assisted.independent_lanes, 3);
        assert!(assisted
            .audit_notes
            .iter()
            .any(|note| note == "smart-assisted-classifier:provider-free-deterministic"));
    }

    /// Labeled complexity-band corpus (the `data/evaluations` analog from the
    /// jacobian-lens reference): a fixed EN/KR fixture set pinning the
    /// deterministic classifier's band assignments, so a keyword-table edit
    /// that silently demotes a class of tasks (the 91c412b0 live bug: a
    /// Korean whole-repo migration classified `Small` → routed to a
    /// Fast-tier model) fails here instead of in production.
    #[test]
    fn complexity_band_evaluation_corpus() {
        let corpus: &[(&str, RouteTaskComplexity)] = &[
            // Large: repo-wide / multi-subsystem, both languages.
            ("scan the whole repo and migrate every module", RouteTaskComplexity::Large),
            ("레포 전체를 훑어서 모든 모듈을 마이그레이션해줘", RouteTaskComplexity::Large),
            ("코드베이스 전체에서 deprecated API를 제거", RouteTaskComplexity::Large),
            ("대규모 리팩터링: 여러 서브시스템의 의존성 정리", RouteTaskComplexity::Large),
            // Medium: implementation/fix verbs, both languages.
            ("implement the retry backoff in the client", RouteTaskComplexity::Medium),
            ("fix the failing integration parser", RouteTaskComplexity::Medium),
            ("이 함수의 버그를 수정해줘", RouteTaskComplexity::Medium),
            ("스트리밍 경로를 리팩토링해줘", RouteTaskComplexity::Medium),
            ("에러 핸들링 로직을 변경해줘", RouteTaskComplexity::Medium),
            // Trivial: label/typo/docs class.
            ("fix the typo in the docs", RouteTaskComplexity::Trivial),
            ("버튼 라벨 오타 고쳐줘", RouteTaskComplexity::Trivial),
            // Compound: a typo mention coexisting with real work must NOT
            // demote that work (the symmetric regression of the typo-first
            // reorder, caught by the review pass).
            (
                "refactor the auth module and fix a typo in the error message",
                RouteTaskComplexity::Medium,
            ),
            ("인증 모듈 로직 변경하면서 오타도 고쳐줘", RouteTaskComplexity::Medium),
            ("refactor the config loader", RouteTaskComplexity::Medium),
            // Small: short lookups with no implementation verbs.
            ("what does this flag do", RouteTaskComplexity::Small),
            ("이 설정값이 뭔지 알려줘", RouteTaskComplexity::Small),
        ];
        let mut misses = Vec::new();
        for (prompt, expected) in corpus {
            // Mirror the real routing path: the effective role is inferred
            // from the task text first (`smart_routing_identity`), and the
            // complexity tables read it (e.g. the parallel/integration →
            // Large rule exempts Coding/Debugging roles).
            let role = infer_route_role(None, "", prompt);
            let input = TaskMetadataInput::new(None, "", prompt);
            let metadata = classify_task_metadata(&input, role);
            if metadata.complexity != *expected {
                misses.push(format!(
                    "{prompt:?} (role {role:?}): expected {expected:?}, got {:?}",
                    metadata.complexity
                ));
            }
        }
        assert!(misses.is_empty(), "complexity corpus misses:\n{}", misses.join("\n"));
    }

    /// The whole-turn complexity entry point (`assess_turn_complexity`) must
    /// agree with the corpus-pinned per-agent classifier — it feeds the Smart
    /// dynamic effort band's per-turn floor, so a drift here would silently
    /// under- or over-think live turns.
    #[test]
    fn assess_turn_complexity_mirrors_the_per_agent_classifier() {
        use super::super::assess_turn_complexity;
        assert_eq!(
            assess_turn_complexity("fix the typo in the docs"),
            RouteTaskComplexity::Trivial
        );
        assert_eq!(
            assess_turn_complexity("이 설정값이 뭔지 알려줘"),
            RouteTaskComplexity::Small
        );
        assert_eq!(
            assess_turn_complexity("이 함수의 버그를 수정해줘"),
            RouteTaskComplexity::Medium
        );
        assert_eq!(
            assess_turn_complexity("레포 전체를 훑어서 모든 모듈을 마이그레이션해줘"),
            RouteTaskComplexity::Large
        );
        assert_eq!(assess_turn_complexity("   "), RouteTaskComplexity::Unknown);
    }

    #[test]
    fn probed_classifier_does_not_join_assisted_marker_contract() {
        // Probed's contract is the live self-assessment probe, NOT the
        // assisted "trust authored lane/shape markers" opt-in — a probed
        // task's text markers must classify exactly like deterministic mode,
        // while the probed audit note still lands for the trail.
        let probed = infer_route_shape_evidence(&RouteEvidenceInput {
            subagent_type: Some("general-purpose"),
            name: Some("planner"),
            description: "plan the work",
            prompt: "Workstreams: parser, executor, docs.",
            workflow_member: true,
            fanout_position: None,
            auto_classifier: RouteAutoClassifierMode::Probed,
        });
        assert_eq!(probed.requested_shape, None);
        assert_eq!(probed.independent_lanes, 0);
        assert!(probed
            .audit_notes
            .iter()
            .any(|note| note == RouteAutoClassifierMode::Probed.audit_note()));
    }

    #[test]
    fn probe_fusion_moves_complexity_one_band_and_records_signal() {
        // A hard-but-atypically-phrased Korean task: no complexity keyword
        // hits, so the deterministic classifier floors at Small — the exact
        // gap class the probe exists to close (the 91c412b0 live bug).
        let input = TaskMetadataInput::new(None, "작업 하나 부탁해", "이 시스템 전반을 손봐줘");
        let mut metadata = classify_task_metadata(&input, RouteRole::Default);
        assert_eq!(metadata.complexity, RouteTaskComplexity::Small);

        let consumed = apply_probe_to_metadata(
            &mut metadata,
            runtime::ProbeAssessment {
                complexity: RouteTaskComplexity::Large,
                risk: RouteTaskRisk::Medium,
                confidence: RouteConfidence::High,
            },
        );
        assert!(consumed);
        // ±1 band clamp: Small + a Large probe lands on Medium, never Large.
        assert_eq!(metadata.complexity, RouteTaskComplexity::Medium);
        assert_eq!(metadata.risk, RouteTaskRisk::Medium);
        assert!(metadata
            .signals
            .iter()
            .any(|signal| signal.source == RouteSignalSource::SelfAssessment));

        // A low-confidence probe is discarded outright.
        let mut untouched = classify_task_metadata(&input, RouteRole::Default);
        let consumed = apply_probe_to_metadata(
            &mut untouched,
            runtime::ProbeAssessment {
                complexity: RouteTaskComplexity::Large,
                risk: RouteTaskRisk::Critical,
                confidence: RouteConfidence::Low,
            },
        );
        assert!(!consumed);
        assert_eq!(untouched.complexity, RouteTaskComplexity::Small);
        assert!(untouched
            .signals
            .iter()
            .all(|signal| signal.source != RouteSignalSource::SelfAssessment));
    }

    #[test]
    fn role_inference_does_not_treat_all_test_mentions_as_verifier() {
        assert_eq!(
            infer_route_role(Some("general-purpose"), "implement feature", "write code and tests"),
            RouteRole::Coding
        );
        assert_eq!(
            infer_route_role(Some("Verification"), "verify", "run tests"),
            RouteRole::Verifier
        );
    }

    #[test]
    fn smart_router_selects_saved_verifier_model_for_agent_execution() {
        let config_home = temp_config_home("verifier");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification")),
            )
            .expect("smart router records a verifier route reason");
            assert!(
                choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "reason names the pinned verifier route: {:?}",
                choice.reason
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_router_choice_carries_fallback_candidates_for_rate_limit_escape() {
        let config_home = temp_config_home("fallback-candidates");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification")),
            )
            .expect("verifier pin should produce a routed choice");
            assert_eq!(choice.model.as_deref(), Some("Verifier/Model-X"));
            assert!(
                choice.fallback_models.is_empty(),
                "exact pins do not invent router alternates; the runtime fallback list adds the parent model separately"
            );

            let mut generic = agent_input(None);
            generic.description = "small repo cleanup".to_string();
            generic.prompt = "Tidy a small typo in the docs.".to_string();
            let generic_choice = smart_parent_model_for_agent(Some("claude-sonnet-main"), &generic)
                .expect("generic route should still expose fallback-only routing state");
            assert!(
                generic_choice
                    .fallback_models
                    .iter()
                    .all(|model| model != "claude-sonnet-main"),
                "fallback-only smart candidates never repeat the selected parent model"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn settings_overlay_from_cli_reaches_spawn_routing() {
        // Global settings say routing ON; a `--settings` overlay (highest
        // precedence, held by ConfigLoader as a process-wide override) says
        // OFF. The router reader must see the MERGED result — its old direct
        // read of only the global file silently ignored the overlay, so a
        // bench/automation run could never disable routing from the CLI.
        let config_home = temp_config_home("cli-overlay-disables-routing");
        write_settings(&config_home, &json!({"smart": {"enabled": true}}));
        let overlay = config_home.join("overlay-settings.json");
        fs::write(
            &overlay,
            serde_json::to_string(&json!({"smart": {"enabled": false}})).expect("json"),
        )
        .expect("write overlay");
        let result = with_config_home(&config_home, || {
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides {
                settings_file: Some(overlay.clone()),
                ..Default::default()
            });
            let result = read_smart_runtime_settings();
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides::default());
            result
        });
        assert!(
            !result.expect("merged settings still parse").enabled,
            "smart.enabled=false via --settings must reach spawn routing \
             (SmartRouteContext::load gates on this flag)"
        );
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn exec_swap_defaults_easy_and_reads_the_merged_settings_overlay() {
        let config_home = temp_config_home("exec-swap-merged-default");
        write_settings(&config_home, &json!({"smart": {"execSwap": "always"}}));
        let overlay = config_home.join("overlay-settings.json");
        fs::write(
            &overlay,
            serde_json::to_string(&json!({"smart": {"execSwap": "never"}})).expect("json"),
        )
        .expect("write overlay");
        with_config_home(&config_home, || {
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides {
                settings_file: Some(overlay.clone()),
                ..Default::default()
            });
            let settings = read_smart_runtime_settings().expect("merged settings");
            assert_eq!(
                settings.exec_swap,
                super::settings::SmartExecSwap::Never,
                "highest-precedence overlay must disable the swap"
            );
            assert_eq!(
                super::settings::smart_exec_swap(),
                super::settings::SmartExecSwap::Never
            );
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides::default());
            assert_eq!(
                super::settings::smart_exec_swap(),
                super::settings::SmartExecSwap::Always,
                "without the overlay, explicit smart.execSwap=always must preserve the original swap behavior"
            );
        });
        let _ = fs::remove_dir_all(config_home);

        let default_home = temp_config_home("exec-swap-default-easy");
        write_settings(&default_home, &json!({"smart": {}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.exec_swap, super::settings::SmartExecSwap::Easy);
            assert_eq!(
                super::settings::smart_setting_defaults().exec_swap,
                super::settings::SmartExecSwap::Easy
            );
        });
        let _ = fs::remove_dir_all(default_home);

        let empty_home = temp_config_home("deep-tier-models-empty");
        write_settings(&empty_home, &json!({"smart": {"deepTierModels": []}}));
        with_config_home(&empty_home, || {
            assert_eq!(
                super::settings::smart_deep_tier_models(),
                runtime::default_deep_tier_models(),
                "an empty configured pool must behave as unset"
            );
            let setting = super::settings::smart_deep_tier_models_for(
                &std::env::current_dir().expect("cwd"),
            )
            .expect("merged settings");
            assert!(!setting.configured);
            assert_eq!(setting.models, runtime::default_deep_tier_models());
        });
        let _ = fs::remove_dir_all(empty_home);
    }

    #[test]
    fn deep_tier_models_use_merged_replacement_order_and_keep_future_ids() {
        let config_home = temp_config_home("deep-tier-models-merged");
        write_settings(
            &config_home,
            &json!({"smart": {"deepTierModels": ["claude-fable-5"]}}),
        );
        let overlay = config_home.join("overlay-settings.json");
        fs::write(
            &overlay,
            serde_json::to_string(&json!({
                "smart": {
                    "deepTierModels": ["claude-opus-5", "future-flagship-9"]
                }
            }))
            .expect("json"),
        )
        .expect("write overlay");

        with_config_home(&config_home, || {
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides {
                settings_file: Some(overlay.clone()),
                ..Default::default()
            });
            assert_eq!(
                super::settings::smart_deep_tier_models(),
                vec![
                    "claude-opus-5".to_string(),
                    "future-flagship-9".to_string()
                ]
            );
            let setting = super::settings::smart_deep_tier_models_for(
                &std::env::current_dir().expect("cwd"),
            )
            .expect("merged settings");
            assert!(setting.configured);
            assert_eq!(
                setting.models,
                vec![
                    "claude-opus-5".to_string(),
                    "future-flagship-9".to_string()
                ]
            );
            runtime::ConfigLoader::set_cli_overrides(runtime::CliConfigOverrides::default());
            assert_eq!(
                super::settings::smart_deep_tier_models(),
                vec!["claude-fable-5".to_string()]
            );
        });
        let _ = fs::remove_dir_all(config_home);

        let default_home = temp_config_home("deep-tier-models-default");
        write_settings(&default_home, &json!({"smart": {}}));
        with_config_home(&default_home, || {
            assert_eq!(
                read_smart_runtime_settings()
                    .expect("settings")
                    .deep_tier_models,
                runtime::default_deep_tier_models()
            );
            assert_eq!(
                super::settings::smart_setting_defaults().deep_tier_models,
                runtime::DEFAULT_DEEP_TIER_MODELS
            );
        });
        let _ = fs::remove_dir_all(default_home);
    }

    #[test]
    fn exec_swap_easy_maps_only_to_the_lowest_implementation_band() {
        use super::settings::SmartExecSwap;

        assert!(SmartExecSwap::Easy.arms_for(RouteTaskComplexity::Trivial));
        for complexity in [
            RouteTaskComplexity::Small,
            RouteTaskComplexity::Medium,
            RouteTaskComplexity::Large,
            RouteTaskComplexity::Unknown,
        ] {
            assert!(!SmartExecSwap::Easy.arms_for(complexity), "{complexity:?}");
        }
        assert!(SmartExecSwap::Always.arms_for(RouteTaskComplexity::Large));
        assert!(!SmartExecSwap::Never.arms_for(RouteTaskComplexity::Trivial));
    }

    #[test]
    fn headroom_penalty_threshold_parses_default_clamp_and_value() {
        // Absent key → the documented default (25).
        let absent = temp_config_home("headroom-absent");
        write_settings(&absent, &json!({"smart": {"enabled": true}}));
        with_config_home(&absent, || {
            assert_eq!(
                read_smart_runtime_settings().expect("settings").headroom_penalty_threshold,
                25
            );
        });
        let _ = fs::remove_dir_all(absent);

        // In-range value → used verbatim.
        let explicit = temp_config_home("headroom-explicit");
        write_settings(&explicit, &json!({"smart": {"headroomPenaltyThreshold": 40}}));
        with_config_home(&explicit, || {
            assert_eq!(
                read_smart_runtime_settings().expect("settings").headroom_penalty_threshold,
                40
            );
        });
        let _ = fs::remove_dir_all(explicit);

        // Out-of-range values clamp into 1..=100 rather than falling to default.
        let low = temp_config_home("headroom-low");
        write_settings(&low, &json!({"smart": {"headroomPenaltyThreshold": 0}}));
        with_config_home(&low, || {
            assert_eq!(
                read_smart_runtime_settings().expect("settings").headroom_penalty_threshold,
                1
            );
        });
        let _ = fs::remove_dir_all(low);

        let high = temp_config_home("headroom-high");
        write_settings(&high, &json!({"smart": {"headroomPenaltyThreshold": 250}}));
        with_config_home(&high, || {
            assert_eq!(
                read_smart_runtime_settings().expect("settings").headroom_penalty_threshold,
                100
            );
        });
        let _ = fs::remove_dir_all(high);
    }

    /// apply.rs injection core: the quota views fold into one binding
    /// (minimum-remaining) entry per provider, unknown rows are skipped, and the
    /// `rate_limit_key` string is what the router matches against.
    #[test]
    fn binding_headroom_keeps_the_minimum_remaining_per_provider() {
        use api::quota::ProviderQuotaView;
        use api::ProviderKind;
        let view = |provider, label: &str, remaining| ProviderQuotaView {
            provider,
            window_label: label.to_string(),
            remaining_percent: remaining,
            resets_at_unix: None,
            estimated: false,
        };
        let folded = super::apply::binding_headroom_from_views(vec![
            // Anthropic 5h/7d → keep the hotter (lower-remaining) window.
            view(ProviderKind::Anthropic, "5h", Some(70)),
            view(ProviderKind::Anthropic, "7d", Some(10)),
            // A row with unknown remaining is dropped entirely.
            view(ProviderKind::OpenAi, "429", None),
            view(ProviderKind::Google, "429", Some(0)),
        ]);
        assert!(folded.contains(&("anthropic".to_string(), 10)), "binding = min window: {folded:?}");
        assert!(folded.contains(&("google".to_string(), 0)));
        assert!(
            !folded.iter().any(|(provider, _)| provider == "openai"),
            "unknown remaining must not fabricate a headroom entry: {folded:?}"
        );
    }

    #[test]
    fn workflow_general_purpose_implementation_records_effective_coding_role() {
        let config_home = temp_config_home("workflow-effective-coding-role");
        write_settings(&config_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&config_home, || {
            let mut input = agent_input(Some("general-purpose"));
            input.description = "workflow phase `implement` item 0".to_string();
            input.prompt = "Implement this scoped provider fallback, add integration tests, and avoid the unrelated parallel workstream.".to_string();

            let choice = smart_parent_model_for_agent(Some("claude-sonnet-main"), &input)
                .expect("ordinary workflow implementation should produce Smart route metadata");
            assert_eq!(choice.decision_meta.role, "coding");
            assert_eq!(choice.decision_meta.complexity, "medium");
            assert!(
                choice
                    .model
                    .as_deref()
                    .is_none_or(|model| !model.contains("fable") && !model.contains("5.6-sol"))
            );
            assert!(choice
                .fallback_models
                .iter()
                .all(|model| !model.contains("fable") && !model.contains("5.6-sol")));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn effective_implementation_role_controls_weak_large_signals() {
        let config_home = temp_config_home("effective-coding-complexity");
        write_settings(&config_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&config_home, || {
            let mut input = agent_input(Some("general-purpose"));
            input.description = "service task".to_string();
            input.prompt = "parallel integration across two services".to_string();

            let choice = smart_parent_model_for_agent(Some("claude-fable-5"), &input)
                .expect("Coding metadata must survive even when text alone looks generic");
            assert_eq!(choice.decision_meta.role, "coding");
            assert_ne!(choice.decision_meta.complexity, "large");
            assert_ne!(choice.model.as_deref(), Some("claude-fable-5"));
            assert!(choice
                .fallback_models
                .iter()
                .all(|model| !model.contains("fable") && !model.contains("5.6-sol")));

            let mut debugger = agent_input(Some("debugger"));
            debugger.description = "debug a scoped failure".to_string();
            debugger.prompt = "debug this parallel integration worker".to_string();
            let debugger_choice = smart_parent_model_for_agent(Some("claude-fable-5"), &debugger)
                .expect("ordinary debugger route metadata");
            assert_eq!(debugger_choice.decision_meta.role, "debugging");
            assert_eq!(debugger_choice.decision_meta.complexity, "medium");
            assert!(debugger_choice
                .model
                .as_deref()
                .is_some_and(|model| !model.contains("fable") && !model.contains("5.6-sol")));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn coding_pin_metadata_survives_when_selected_model_equals_parent() {
        let config_home = temp_config_home("coding-pin-same-parent");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {
                    "roles": {
                        "coding": {"mode": "pinned", "model": "claude-fable-5"}
                    }
                }
            }),
        );
        with_config_home(&config_home, || {
            let mut input = agent_input(Some("general-purpose"));
            input.description = "implement a scoped fix".to_string();
            input.prompt = "Implement the change and run its focused test.".to_string();

            let choice = smart_parent_model_for_agent(Some("claude-fable-5"), &input)
                .expect("same-parent Coding pin still needs policy metadata");
            assert!(choice.model.is_none(), "the selected model already is the parent");
            assert_eq!(choice.decision_meta.role, "coding");
            assert_eq!(choice.decision_meta.complexity, "medium");
            assert_eq!(choice.decision_meta.route_source, "pin");
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn custom_role_name_collision_uses_task_write_intent() {
        let config_home = temp_config_home("custom-role-name-collision");
        write_settings(&config_home, &json!({"smart": {"enabled": true}}));
        let definitions = config_home.join("agents");
        fs::create_dir_all(&definitions).expect("custom definitions directory");
        fs::write(
            definitions.join("analysis.md"),
            "---\nname: analysis\ndescription: Custom implementer\n---\nImplement scoped changes.",
        )
        .expect("custom analysis definition");
        with_config_home(&config_home, || {
            let prior = std::env::var_os("ZO_AGENT_DEFS_DIR");
            std::env::set_var("ZO_AGENT_DEFS_DIR", &definitions);
            let mut input = agent_input(Some("analysis"));
            input.description = "provider ticket 123".to_string();
            input.prompt = "Handle this ticket end-to-end.".to_string();
            let choice = smart_parent_model_for_agent(Some("claude-fable-5"), &input)
                .expect("custom implementation route");
            match prior {
                Some(value) => std::env::set_var("ZO_AGENT_DEFS_DIR", value),
                None => std::env::remove_var("ZO_AGENT_DEFS_DIR"),
            }

            assert_eq!(choice.decision_meta.role, "coding");
            assert_ne!(choice.decision_meta.complexity, "large");
            assert!(choice
                .model
                .as_deref()
                .is_some_and(|model| !model.contains("fable") && !model.contains("5.6-sol")));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn implementation_intent_recognizes_code_writing_phrases_without_matching_nouns() {
        assert!(task_has_write_intent("", "Write a Rust function for the endpoint"));
        assert!(task_has_write_intent("", "Write the code for a new endpoint"));
        assert!(task_has_write_intent("", "Adding a route for the new endpoint"));
        assert!(task_has_write_intent("", "Rust 함수를 작성해줘"));
        assert!(!task_has_write_intent("", "Review the implementation and write a report"));
    }

    /// `smart.enabled` defaults ON (user decision, 2026-07-10): a settings.json
    /// without the key — or without a `smart` object at all — must read as
    /// enabled, while an explicit `false` still wins. Pinned here so a
    /// regression back to default-off cannot land silently; the CLI's
    /// `snapshot_from_root` lockstep test covers the dual-reader side.
    #[test]
    fn smart_enabled_defaults_on_when_key_absent_and_explicit_false_wins() {
        let absent_home = temp_config_home("enabled-default-absent");
        write_settings(&absent_home, &json!({}));
        with_config_home(&absent_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(settings.enabled, "missing smart object must default ON");
        });
        let _ = fs::remove_dir_all(absent_home);

        let empty_smart_home = temp_config_home("enabled-default-empty-smart");
        write_settings(&empty_smart_home, &json!({"smart": {}}));
        with_config_home(&empty_smart_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(settings.enabled, "smart object without enabled key must default ON");
        });
        let _ = fs::remove_dir_all(empty_smart_home);

        let off_home = temp_config_home("enabled-explicit-off");
        write_settings(&off_home, &json!({"smart": {"enabled": false}}));
        with_config_home(&off_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(!settings.enabled, "explicit smart.enabled=false must win");
        });
        let _ = fs::remove_dir_all(off_home);

        // The cross-crate defaults contract mirrors the same flip.
        assert!(super::settings::smart_setting_defaults().enabled);
    }

    /// A missing settings.json is a fresh install, not an opt-out: the reader
    /// must fall through to the enabled-by-default path — otherwise live
    /// routing stays OFF while the CLI reader (`NotFound` → empty object) shows
    /// ON and the boot banner announces a default that is not in effect.
    /// Malformed JSON still bails to None: no routing on an unreadable config.
    #[test]
    fn smart_enabled_defaults_on_when_settings_file_is_missing() {
        let missing_home = temp_config_home("enabled-default-missing-file");
        fs::create_dir_all(&missing_home).expect("config dir");
        with_config_home(&missing_home, || {
            let settings =
                read_smart_runtime_settings().expect("missing settings.json must not bail");
            assert!(settings.enabled, "fresh install must default ON");
        });
        let _ = fs::remove_dir_all(missing_home);

        let malformed_home = temp_config_home("enabled-malformed-file");
        fs::create_dir_all(&malformed_home).expect("config dir");
        fs::write(malformed_home.join("settings.json"), b"{not json").expect("write settings");
        with_config_home(&malformed_home, || {
            assert!(
                read_smart_runtime_settings().is_none(),
                "malformed settings.json must fail safe to no routing"
            );
        });
        let _ = fs::remove_dir_all(malformed_home);
    }

    #[test]
    fn provider_allowlist_parses_trimmed_and_defaults_to_empty() {
        let default_home = temp_config_home("provider-allowlist-default");
        write_settings(&default_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(
                settings.provider_allowlist.is_empty(),
                "missing key must mean no restriction"
            );
        });
        let _ = fs::remove_dir_all(default_home);

        let set_home = temp_config_home("provider-allowlist-set");
        write_settings(
            &set_home,
            &json!({"smart": {
                "enabled": true,
                "providerAllowlist": ["anthropic", "  openai  ", "", 42]
            }}),
        );
        with_config_home(&set_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(
                settings.provider_allowlist,
                vec!["anthropic".to_string(), "openai".to_string()],
                "entries are trimmed; blanks and non-strings are dropped"
            );
        });
        let _ = fs::remove_dir_all(set_home);
    }

    #[test]
    fn fallback_candidate_limit_defaults_to_two_and_honors_override() {
        let default_home = temp_config_home("fallback-candidate-limit-default");
        write_settings(&default_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(
                settings.fallback_candidate_limit,
                super::settings::DEFAULT_FALLBACK_CANDIDATE_LIMIT
            );
            assert_eq!(settings.fallback_candidate_limit, 2);
        });
        let _ = fs::remove_dir_all(default_home);

        let override_home = temp_config_home("fallback-candidate-limit-override");
        write_settings(
            &override_home,
            &json!({"smart": {"enabled": true, "fallbackCandidateLimit": 4}}),
        );
        with_config_home(&override_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.fallback_candidate_limit, 4);
        });
        let _ = fs::remove_dir_all(override_home);

        // Zero (and any non-positive/invalid value) falls back to the default
        // rather than disabling fallback candidates entirely by accident.
        let zero_home = temp_config_home("fallback-candidate-limit-zero");
        write_settings(
            &zero_home,
            &json!({"smart": {"enabled": true, "fallbackCandidateLimit": 0}}),
        );
        with_config_home(&zero_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.fallback_candidate_limit, 2);
        });
        let _ = fs::remove_dir_all(zero_home);
    }

    #[test]
    fn exploration_settings_default_on_with_cadence_five_and_honor_overrides() {
        let default_home = temp_config_home("exploration-settings-default");
        write_settings(&default_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(settings.exploration, "smart.exploration defaults to on");
            assert_eq!(
                settings.exploration_cadence,
                super::settings::DEFAULT_EXPLORATION_CADENCE
            );
            assert_eq!(settings.exploration_cadence, 5);
        });
        let _ = fs::remove_dir_all(default_home);

        let off_home = temp_config_home("exploration-settings-off");
        write_settings(
            &off_home,
            &json!({"smart": {"enabled": true, "exploration": false, "explorationCadence": 3}}),
        );
        with_config_home(&off_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(!settings.exploration);
            assert_eq!(settings.exploration_cadence, 3);
        });
        let _ = fs::remove_dir_all(off_home);

        // Zero (and any non-positive/invalid value) falls back to the
        // default rather than dividing by zero or exploring every route.
        let zero_home = temp_config_home("exploration-cadence-zero");
        write_settings(
            &zero_home,
            &json!({"smart": {"enabled": true, "explorationCadence": 0}}),
        );
        with_config_home(&zero_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.exploration_cadence, 5);
        });
        let _ = fs::remove_dir_all(zero_home);
    }

    #[test]
    fn learned_specialty_settings_default_to_shadow_and_honor_overrides() {
        use super::settings::LearnedSpecialtyMode;

        let default_home = temp_config_home("learned-specialty-default");
        write_settings(&default_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.learned_specialty, LearnedSpecialtyMode::Shadow, "default is shadow");
        });
        let _ = fs::remove_dir_all(default_home);

        let off_home = temp_config_home("learned-specialty-off");
        write_settings(&off_home, &json!({"smart": {"enabled": true, "learnedSpecialty": "off"}}));
        with_config_home(&off_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.learned_specialty, LearnedSpecialtyMode::Off);
        });
        let _ = fs::remove_dir_all(off_home);

        let on_home = temp_config_home("learned-specialty-on");
        write_settings(&on_home, &json!({"smart": {"enabled": true, "learnedSpecialty": "on"}}));
        with_config_home(&on_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.learned_specialty, LearnedSpecialtyMode::On);
        });
        let _ = fs::remove_dir_all(on_home);

        // An unrecognized value fails closed to the documented default
        // (same convention as `RouteAutoClassifierMode`), not an error.
        let bogus_home = temp_config_home("learned-specialty-bogus");
        write_settings(&bogus_home, &json!({"smart": {"enabled": true, "learnedSpecialty": "yolo"}}));
        with_config_home(&bogus_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.learned_specialty, LearnedSpecialtyMode::Shadow);
        });
        let _ = fs::remove_dir_all(bogus_home);
    }

    /// `smart.verifyCrossProvider` defaults ON (like the global diversity flag)
    /// and is parsed independently of `allowCrossProviderDiversity`: turning the
    /// worker-diversity flag off must NOT drag the verify-leg switch down with
    /// it. Mirrors the CLI-side `snapshot_from_root` parse in lockstep.
    #[test]
    fn verify_cross_provider_defaults_on_and_is_independent_of_global_diversity() {
        let default_home = temp_config_home("verify-cross-default");
        write_settings(&default_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(settings.verify_cross_provider, "missing key must default ON");
        });
        let _ = fs::remove_dir_all(default_home);

        // Global worker-diversity OFF, but verify-cross unspecified: verify-cross
        // must still read ON — the two keys are decoupled.
        let decoupled_home = temp_config_home("verify-cross-decoupled");
        write_settings(
            &decoupled_home,
            &json!({"smart": {"enabled": true, "allowCrossProviderDiversity": false}}),
        );
        with_config_home(&decoupled_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(!settings.allow_cross_provider_diversity);
            assert!(
                settings.verify_cross_provider,
                "verify-cross must stay ON when only the global flag is off"
            );
        });
        let _ = fs::remove_dir_all(decoupled_home);

        let off_home = temp_config_home("verify-cross-off");
        write_settings(
            &off_home,
            &json!({"smart": {"enabled": true, "verifyCrossProvider": false}}),
        );
        with_config_home(&off_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert!(!settings.verify_cross_provider, "explicit false must win");
        });
        let _ = fs::remove_dir_all(off_home);

        // The cross-crate defaults contract mirrors the same ON default.
        assert!(super::settings::smart_setting_defaults().verify_cross_provider);
    }

    #[test]
    fn smart_router_auto_classifier_settings_are_conservative_and_auditable() {
        let default_home = temp_config_home("auto-classifier-default");
        write_settings(&default_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&default_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.auto_classifier, RouteAutoClassifierMode::Deterministic);
            assert_eq!(
                settings.auto_classifier.audit_note(),
                "smart-auto-classifier:deterministic-provider-free",
            );
        });
        let _ = fs::remove_dir_all(default_home);

        let invalid_home = temp_config_home("auto-classifier-invalid");
        write_settings(
            &invalid_home,
            &json!({"smart": {"enabled": true, "autoClassifier": "surprise"}}),
        );
        with_config_home(&invalid_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.auto_classifier, RouteAutoClassifierMode::Deterministic);
        });
        let _ = fs::remove_dir_all(invalid_home);

        let assisted_home = temp_config_home("auto-classifier-assisted");
        write_settings(
            &assisted_home,
            &json!({"smart": {"enabled": true, "autoClassifier": "assisted"}}),
        );
        with_config_home(&assisted_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.auto_classifier, RouteAutoClassifierMode::Assisted);
            assert_eq!(
                settings.auto_classifier.audit_note(),
                "smart-auto-classifier:assisted-provider-free-deterministic",
            );
        });
        let _ = fs::remove_dir_all(assisted_home);

        let probed_home = temp_config_home("auto-classifier-probed");
        write_settings(
            &probed_home,
            &json!({"smart": {"enabled": true, "autoClassifier": "probed"}}),
        );
        with_config_home(&probed_home, || {
            let settings = read_smart_runtime_settings().expect("settings");
            assert_eq!(settings.auto_classifier, RouteAutoClassifierMode::Probed);
            assert_eq!(
                settings.auto_classifier.audit_note(),
                "smart-auto-classifier:probed-model-fused-deterministic-floor",
            );
        });
        let _ = fs::remove_dir_all(probed_home);
    }

    #[test]
    fn smart_router_feedback_opt_in_preserves_saved_verifier_model() {
        let config_home = temp_config_home("verifier-feedback-opt-in");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true, "feedbackInformedAuto": true, "autoClassifier": "assisted"},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification")),
            )
            .expect("smart router records a verifier route reason");
            assert!(
                choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "reason names the pinned verifier route: {:?}",
                choice.reason
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn phase0_role_inference_baseline_covers_representative_roles() {
        let cases = [
            (
                Some("Verification"),
                "verify the patch",
                "run tests for the changed code",
                RouteRole::Verifier,
            ),
            (
                Some("code-reviewer"),
                "review the diff",
                "perform adversarial review",
                RouteRole::Reviewer,
            ),
            (
                Some("debugger"),
                "reproduce the failure",
                "debug the failing check",
                RouteRole::Debugging,
            ),
            (
                Some("frontend-design"),
                "polish frontend UI",
                "improve design details",
                RouteRole::Design,
            ),
            (
                None,
                "plan the approach",
                "compare options before editing",
                RouteRole::Analysis,
            ),
            (
                None,
                "summarize the result",
                "synthesize a concise report",
                RouteRole::Synthesizer,
            ),
        ];

        for (subagent_type, description, prompt, expected) in cases {
            assert_eq!(
                infer_route_role(subagent_type, description, prompt),
                expected,
                "unexpected route for {subagent_type:?} / {description:?} / {prompt:?}"
            );
        }
    }

    #[test]
    fn long_keyword_less_brief_lifts_complexity_off_the_small_floor() {
        // No complexity keyword matches, but an 800+ char brief is itself a
        // difficulty signal — without the length lift the Default role
        // difficulty-routes this to a Fast-tier model.
        let long_brief = "Trace how the settlement reconciliation ledger drifts \
            from the upstream statement when partial captures arrive out of \
            order, and report the exact invariant that breaks. "
            .repeat(6);
        assert!(long_brief.chars().count() >= 800, "fixture must be long");
        let metadata = classify_task_metadata(
            &TaskMetadataInput::new(None, "", &long_brief),
            RouteRole::Default,
        );
        assert_eq!(metadata.complexity, RouteTaskComplexity::Medium);

        // Short keyword-less briefs keep the Small floor.
        let metadata = classify_task_metadata(
            &TaskMetadataInput::new(None, "", "list the routes in this service"),
            RouteRole::Default,
        );
        assert_eq!(metadata.complexity, RouteTaskComplexity::Small);
    }

    #[test]
    fn korean_mixed_verify_and_modify_preserves_coding_intent() {
        let prompt = "플랜설정후 적대적검증 하고 1번부터 5번까지 완벽하게 수정";
        assert_eq!(infer_route_role(None, "", prompt), RouteRole::Coding);

        let input = TaskMetadataInput::new(None, "", prompt);
        let metadata = classify_task_metadata(&input, RouteRole::Coding);
        assert_eq!(metadata.kind, RouteTaskKind::Coding);
        assert_eq!(metadata.tool_need, RouteToolNeed::Write);
        assert!(matches!(
            metadata.verification_need,
            RouteVerificationNeed::Focused | RouteVerificationNeed::Full
        ));
        let needs = plan_agent_needs(&metadata);
        assert!(
            needs.iter().any(|need| need.candidate_role == RouteRole::Verifier),
            "verification evidence must be retained: {needs:?}"
        );
    }

    #[test]
    fn role_inference_recognizes_korean_specialties() {
        assert_eq!(infer_route_role(None, "", "변경사항 검토"), RouteRole::Reviewer);
        assert_eq!(infer_route_role(None, "", "수정 내용 리뷰"), RouteRole::Reviewer);

        // English-only matching routed every Korean task to Default/Fast, so a
        // "프로젝트 분석" fan-out ran entirely on the cheap fast model. Korean
        // keywords let each specialty reach its proper role (and, best-of-breed, its
        // specialist model). This is the screenshot scenario: "...분석" → Analysis.
        let cases = [
            (None, "소스 아키텍처와 핵심 기능 분석", "구조를 분석", RouteRole::Analysis),
            (None, "추론 작업", "단계별로 추론", RouteRole::Analysis),
            (None, "디자인 작업", "UI 디자인 개선", RouteRole::Design),
            (None, "품질 검증", "테스트 실행", RouteRole::Verifier),
            (None, "기능 구현", "코드 구현", RouteRole::Coding),
            (None, "리서치", "관련 자료 조사", RouteRole::Research),
        ];
        for (subagent_type, description, prompt, expected) in cases {
            assert_eq!(
                infer_route_role(subagent_type, description, prompt),
                expected,
                "unexpected route for Korean {description:?} / {prompt:?}",
            );
        }
    }

    #[test]
    fn phase0_weak_or_support_signals_do_not_create_verifier_role() {
        assert_eq!(
            infer_route_role(Some("general-purpose"), "update label copy", "adjust docs and wording"),
            RouteRole::Writing
        );
        assert_eq!(
            infer_route_role(None, "small typo", "rename a label"),
            RouteRole::Default
        );
        assert_ne!(
            infer_route_role(None, "write docs", "mention tests in a changelog"),
            RouteRole::Verifier
        );
    }


    #[test]
    fn phase1_metadata_classifier_emits_signals_without_selecting_models() {
        let fallback_role = infer_route_role(
            Some("Verification"),
            "verify security contract",
            "run tests with structured schema",
        );
        let input = TaskMetadataInput::new(
            Some("Verification"),
            "verify security contract",
            "run tests with structured schema",
        )
        .with_schema(true)
        .with_workflow_member(true);
        let metadata = classify_task_metadata(&input, fallback_role);

        assert_eq!(metadata.fallback_role, RouteRole::Verifier);
        assert_eq!(metadata.risk, RouteTaskRisk::Medium);
        assert_eq!(metadata.context_need, RouteContextNeed::MultiFile);
        assert_eq!(metadata.tool_need, RouteToolNeed::Shell);
        assert_eq!(metadata.output_need, RouteOutputNeed::Structured);
        assert_eq!(metadata.verification_need, RouteVerificationNeed::Focused);
        assert_eq!(metadata.confidence, RouteConfidence::High);
        assert!(metadata
            .signals
            .iter()
            .any(|signal| signal.source == RouteSignalSource::ToolSchema));
        assert!(metadata
            .signals
            .iter()
            .any(|signal| signal.source == RouteSignalSource::WorkflowContext));
    }

    #[test]
    fn phase1_classifier_scores_kind_and_confidence_by_signal_weight() {
        // Scored, not flat: confidence scales with accumulated signal weight
        // (old behavior was "any signal → High"), and kind is accumulated from
        // weighted signals rather than read straight off the role.
        let weak = classify_task_metadata(
            &TaskMetadataInput::new(None, "please fix", ""),
            RouteRole::Coding,
        );
        assert_eq!(weak.confidence, RouteConfidence::Low, "one weak signal is not high confidence");
        assert_eq!(weak.kind, RouteTaskKind::Coding);

        let strong = classify_task_metadata(
            &TaskMetadataInput::new(Some("code-reviewer"), "review the auth change", "verify and run tests")
                .with_schema(true),
            RouteRole::Reviewer,
        );
        assert_eq!(strong.confidence, RouteConfidence::High);
        // Review signals (subagent + keyword) outweigh the lone verify keyword.
        assert_eq!(strong.kind, RouteTaskKind::Review);
        // The accumulated weight is actually read (no longer a dead field).
        assert!(strong.signals.iter().map(|signal| signal.weight).sum::<i32>() >= 100);
    }

    #[test]
    fn assess_turn_orchestration_reports_need_for_risky_turns_and_none_for_chat() {
        // Risk-bearing turn with no obvious delegation keyword → the planner still
        // finds agent need, so the host can drive a non-Solo shape from it.
        let risky = assess_turn_orchestration("harden the auth token handling");
        assert!(risky.need_count >= 1, "auth/token risk should plan an agent");
        assert_ne!(risky.shape, RouteShapeKind::Solo);

        // Plain chat → no need, Solo (the host keeps its own default).
        let chat = assess_turn_orchestration("hello there");
        assert_eq!(chat.need_count, 0);
        assert_eq!(chat.shape, RouteShapeKind::Solo);
    }

    #[test]
    fn goal_pivot_marker_promotes_the_turn_to_large() {
        // A pivot turn carries the goal controller's re-approach marker: the
        // problem already exhausted an approach on the routed model, so the
        // turn classifies Large (→ the strong tier) even when its own wording
        // would only be Medium.
        let pivot_input = TaskMetadataInput::new(
            None,
            "goal pivot",
            "[zo:goal-pivot] The current approach has failed repeatedly — fix the parser",
        );
        let pivot = classify_task_metadata(&pivot_input, RouteRole::Default);
        assert_eq!(pivot.complexity, RouteTaskComplexity::Large);

        // The same wording WITHOUT the marker stays Medium ("fix").
        let plain_input = TaskMetadataInput::new(None, "goal repair", "fix the parser");
        let plain = classify_task_metadata(&plain_input, RouteRole::Default);
        assert_eq!(plain.complexity, RouteTaskComplexity::Medium);
    }

    #[test]
    fn korean_task_text_classifies_complexity_like_english() {
        // Large parity: the Large keyword table was English-only, so a Korean
        // whole-repo task fell through to Small — and the Default role
        // difficulty-routes Small to a Fast-tier model. A Korean "레포 전체"
        // task must classify Large exactly like "whole repo" would.
        let large_input = TaskMetadataInput::new(
            None,
            "대규모 작업",
            "레포 전체를 훑어서 모든 모듈을 마이그레이션해줘. 여러 단계에 걸쳐 진행해야 한다.",
        );
        let large = classify_task_metadata(&large_input, RouteRole::Default);
        assert_eq!(large.complexity, RouteTaskComplexity::Large);
        assert_eq!(large.context_need, RouteContextNeed::WholeRepo);

        // Medium parity: a lone 마이그레이션 (no whole-repo cue) is a Medium
        // edit-shaped task, like "fix"/"implement".
        let medium_input =
            TaskMetadataInput::new(None, "정리", "이 함수를 새 api로 마이그레이션해줘.");
        let medium = classify_task_metadata(&medium_input, RouteRole::Default);
        assert_eq!(medium.complexity, RouteTaskComplexity::Medium);

        // `integration` and `parallel` describe work shape, not necessarily
        // genuine difficulty. A scoped integration-test change must not unlock
        // premium implementation models on its first attempt.
        let scoped_integration = classify_task_metadata(
            &TaskMetadataInput::new(
                None,
                "add an integration test",
                "implement two parallel test cases for this one provider",
            ),
            RouteRole::Coding,
        );
        assert_eq!(scoped_integration.complexity, RouteTaskComplexity::Medium);

        let analysis_integration = classify_task_metadata(
            &TaskMetadataInput::new(
                None,
                "analyze parallel integration",
                "compare the integration paths across services",
            ),
            RouteRole::Analysis,
        );
        assert_eq!(
            analysis_integration.complexity,
            RouteTaskComplexity::Large,
            "the stricter signal is Coding-only; deep analysis keeps its prior escalation"
        );

        // Trivial parity: 오타 == typo.
        let trivial_input = TaskMetadataInput::new(None, "간단", "readme 오타 알려줘.");
        let trivial = classify_task_metadata(&trivial_input, RouteRole::Default);
        assert_eq!(trivial.complexity, RouteTaskComplexity::Trivial);

        // Risk parity: 삭제 == delete (High), so Korean destructive tasks get
        // the same reviewer-need planning as English ones.
        let risky_input =
            TaskMetadataInput::new(None, "정리", "이 테이블 데이터를 전부 삭제해줘.");
        let risky = classify_task_metadata(&risky_input, RouteRole::Default);
        assert_eq!(risky.risk, RouteTaskRisk::High);
    }

    #[test]
    fn dependent_multi_file_coding_task_classifies_large() {
        let input = TaskMetadataInput::new(
            Some("general-purpose"),
            "implement dependent checkout boundaries",
            "checkout/domain.py, checkout/discounts.py, checkout/inventory.py, \
             checkout/ledger.py, checkout/service.py, checkout/api.py를 수정해줘. \
             도메인 검증, 가격 계산, 원자적 재고 예약, 멱등 저장, API 직렬화가 \
             서로 의존하므로 전체 테스트가 통과할 때까지 구현해.",
        );

        let metadata = classify_task_metadata(&input, RouteRole::Coding);
        assert_eq!(metadata.complexity, RouteTaskComplexity::Large);
        assert_eq!(metadata.context_need, RouteContextNeed::MultiFile);

        let bounded = classify_task_metadata(
            &TaskMetadataInput::new(
                Some("general-purpose"),
                "implement one module",
                "rate_limiter.py 하나를 구현하고 테스트해.",
            ),
            RouteRole::Coding,
        );
        assert_eq!(bounded.complexity, RouteTaskComplexity::Medium);
    }

    #[test]
    fn phase2_need_planner_emits_evidence_plans_but_no_shape_or_model() {
        let quiet_input = TaskMetadataInput::new(None, "small typo", "rename a label");
        let quiet_metadata = classify_task_metadata(&quiet_input, RouteRole::Default);
        assert_eq!(quiet_metadata.complexity, RouteTaskComplexity::Trivial);
        assert!(
            plan_agent_needs(&quiet_metadata).is_empty(),
            "trivial metadata should not create agent work"
        );

        let risky_input = TaskMetadataInput::new(
            Some("code-reviewer"),
            "review sandbox permission handling",
            "verify auth token safety",
        );
        let risky_metadata = classify_task_metadata(&risky_input, RouteRole::Reviewer);
        let needs = plan_agent_needs(&risky_metadata);

        assert!(needs.iter().any(|need| need.candidate_role == RouteRole::Reviewer));
        assert!(needs.iter().all(|need| !need.need.is_empty()));
        assert!(needs.iter().all(|need| !need.evidence_target.is_empty()));
        assert!(needs.iter().all(|need| !need.stop_condition.is_empty()));
        assert!(needs.iter().all(|need| !need.fallback.is_empty()));
    }

    #[test]
    fn phase2_need_planner_emits_multiple_distinct_role_plans_without_a_fleet() {
        // A genuine multi-need turn (debugging + high-risk + full verification)
        // must produce >=2 plans, each with a DISTINCT role and unique evidence
        // target — never a duplicated reviewer fleet.
        let input = TaskMetadataInput::new(
            Some("debugger"),
            "reproduce the auth permission crash",
            "full verification of the failing security path is required",
        );
        let metadata = classify_task_metadata(&input, RouteRole::Debugging);
        let needs = plan_agent_needs(&metadata);

        assert!(needs.len() >= 2, "multi-need turn should plan >=2 agents: {needs:?}");
        // Verification (Full) → Verifier, kind Debugging → Debugging, risk High → Reviewer.
        assert!(needs.iter().any(|need| need.candidate_role == RouteRole::Verifier));
        assert!(needs.iter().any(|need| need.candidate_role == RouteRole::Debugging));
        assert!(needs.iter().any(|need| need.candidate_role == RouteRole::Reviewer));

        let mut roles: Vec<RouteRole> = needs.iter().map(|need| need.candidate_role).collect();
        roles.sort_by_key(|role| format!("{role:?}"));
        let unique = roles.len();
        roles.dedup();
        assert_eq!(roles.len(), unique, "no duplicate-role fleet: {needs:?}");

        let mut targets: Vec<&str> = needs.iter().map(|need| need.evidence_target.as_str()).collect();
        targets.sort_unstable();
        let total = targets.len();
        targets.dedup();
        assert_eq!(targets.len(), total, "each plan must have a distinct evidence target: {needs:?}");
    }

    #[test]
    fn phase3_route_shape_selector_chooses_shape_without_choosing_model() {
        let solo_metadata = classify_task_metadata(
            &TaskMetadataInput::new(None, "small typo", "rename a label"),
            RouteRole::Default,
        );
        let solo_needs = plan_agent_needs(&solo_metadata);
        let solo = select_route_shape(&RouteShapeInput::new(&solo_metadata, &solo_needs));
        assert_eq!(solo.shape, RouteShapeKind::Solo);
        assert_eq!(solo.outcome, UserOrchestrationRequestOutcome::NotRequested);

        let verifier_metadata = classify_task_metadata(
            &TaskMetadataInput::new(Some("Verification"), "verify patch", "run tests"),
            RouteRole::Verifier,
        );
        let verifier_needs = plan_agent_needs(&verifier_metadata);
        let one = select_route_shape(&RouteShapeInput::new(&verifier_metadata, &verifier_needs));
        assert_eq!(one.shape, RouteShapeKind::OneSpecialist);

        let parallel = select_route_shape(
            &RouteShapeInput::new(&verifier_metadata, &verifier_needs).with_independent_lanes(3),
        );
        assert_eq!(parallel.shape, RouteShapeKind::ParallelLanes);
        assert!(parallel.reason.contains("independent lanes"));

        let forced_parallel_for_solo = select_route_shape(
            &RouteShapeInput::new(&solo_metadata, &solo_needs)
                .with_requested_shape(RouteShapeKind::ParallelLanes),
        );
        assert_eq!(forced_parallel_for_solo.shape, RouteShapeKind::Solo);
        assert_eq!(
            forced_parallel_for_solo.outcome,
            UserOrchestrationRequestOutcome::RecommendDifferentShape
        );
    }

    #[test]
    fn phase3_route_shape_selector_handles_findings_and_unsafe_requests() {
        let metadata = classify_task_metadata(
            &TaskMetadataInput::new(Some("debugger"), "debug failing workflow", "reproduce failure"),
            RouteRole::Debugging,
        );
        let needs = plan_agent_needs(&metadata);

        let repair = select_route_shape(&RouteShapeInput::new(&metadata, &needs).with_findings(true));
        assert_eq!(repair.shape, RouteShapeKind::RepairLoop);

        let parallel_repair = select_route_shape(
            &RouteShapeInput::new(&metadata, &needs)
                .with_findings(true)
                .with_independent_lanes(2),
        );
        assert_eq!(parallel_repair.shape, RouteShapeKind::ParallelRepairLoop);

        let unsafe_request = select_route_shape(
            &RouteShapeInput::new(&metadata, &needs)
                .with_requested_shape(RouteShapeKind::ParallelLanes)
                .with_unsafe_request(true),
        );
        assert_eq!(unsafe_request.shape, RouteShapeKind::SequentialWorkflow);
        assert_eq!(
            unsafe_request.outcome,
            UserOrchestrationRequestOutcome::RefuseUnsafeShape
        );

        let ambiguous_request = select_route_shape(
            &RouteShapeInput::new(&metadata, &needs)
                .with_requested_shape(RouteShapeKind::ParallelLanes)
                .with_ambiguous_ownership(true),
        );
        assert_eq!(ambiguous_request.shape, RouteShapeKind::SequentialWorkflow);
        assert_eq!(
            ambiguous_request.outcome,
            UserOrchestrationRequestOutcome::RecommendDifferentShape
        );
    }


    #[test]
    fn smart_router_does_not_route_agents_when_disabled() {
        let config_home = temp_config_home("disabled");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": false},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            assert!(smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification"))
            )
            .is_none());
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_router_does_not_override_explicit_agent_model() {
        let config_home = temp_config_home("explicit");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let mut input = agent_input(Some("Verification"));
            input.model = Some("User/Explicit".to_string());
            assert!(smart_parent_model_for_agent(Some("claude-sonnet-main"), &input).is_none());
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn smart_router_smuggles_resolved_model_and_reason_for_spawn_multi_agent_members() {
        let config_home = temp_config_home("spawn");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({"description": "verify", "prompt": "run tests", "subagent_type": "Verification"}),
                    json!({"description": "review", "prompt": "review", "model": "User/Explicit"}),
                ],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);
            // A smart route does NOT masquerade as a user-typed `model` (that
            // field stays clear, so the on-wire same-provider-family gate never
            // second-guesses a trusted host decision)…
            assert!(
                input.agents[0].get("model").is_none(),
                "a smart route must not pose as an explicit user model override"
            );
            // …instead the resolved model travels under the trusted smuggle key
            // and is applied verbatim by the spawn path — this is what makes the
            // routed agent actually run on the verifier pin, not the parent.
            assert_eq!(
                input.agents[0][super::apply::ROUTE_MODEL_SMUGGLE_KEY],
                "Verifier/Model-X",
                "the verifier pin drives the spawned agent's actual model"
            );
            assert_eq!(input.agents[1]["model"], "User/Explicit");
            // The WHY travels alongside the model (smuggle key → manifest
            // routeReason); an explicit-model member carries neither.
            let reason = input.agents[0][super::apply::ROUTE_REASON_SMUGGLE_KEY]
                .as_str()
                .expect("routed member carries a route reason");
            assert!(
                reason.contains("pin"),
                "reason names the pin decision: {reason}"
            );
            assert!(input.agents[1]
                .get(super::apply::ROUTE_REASON_SMUGGLE_KEY)
                .is_none());
            assert!(
                input.agents[1]
                    .get(super::apply::ROUTE_MODEL_SMUGGLE_KEY)
                    .is_none(),
                "an explicit-model member is never overridden by a smart route"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    #[allow(clippy::too_many_lines)] // one adversarial test covering all 6 smuggle keys × 3 cases
    fn crafted_route_smuggle_keys_in_untrusted_agent_json_are_scrubbed_not_trusted() {
        // SECURITY: a SpawnMultiAgent member's JSON is untrusted (model/user-
        // authored). A crafted `__zo_route_model` must NEVER survive
        // `apply_smart_models_to_spawn_input` to be read verbatim by the spawn
        // path — that would bypass the on-wire `model` provider-family fence and
        // force an arbitrary cross-provider model. The host is the SOLE populator
        // of the smuggle keys; any caller-supplied copy is scrubbed up front,
        // regardless of whether routing runs, skips, or overrides the member.

        // A member with a forged route model (and reason) crafted into its JSON.
        let zo_smuggle = |mut agent: Value| -> Value {
            let object = agent.as_object_mut().expect("agent must be an object");
            object.insert(
                super::apply::ROUTE_MODEL_SMUGGLE_KEY.to_string(),
                json!("evil/cross-provider-model"),
            );
            object.insert(
                super::apply::ROUTE_REASON_SMUGGLE_KEY.to_string(),
                json!("forged reason"),
            );
            object.insert(
                super::apply::ROUTE_FALLBACK_MODELS_SMUGGLE_KEY.to_string(),
                json!(["evil/fallback-model"]),
            );
            object.insert(
                super::apply::ROUTE_EFFORT_SMUGGLE_KEY.to_string(),
                json!("ultra"),
            );
            object.insert(
                super::apply::ROUTE_DECISION_META_SMUGGLE_KEY.to_string(),
                json!({"role": "forged-role", "complexity": "forged", "risk": "forged", "routeSource": "forged"}),
            );
            object.insert(
                super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY.to_string(),
                json!("evil-worker-name"),
            );
            agent
        };

        // Case 1 — routing DISABLED: the router early-returns before the loop, yet
        // the forged keys must still be gone (this is the "even when routing is
        // disabled" bypass).
        let off_home = temp_config_home("spoof-routing-off");
        write_settings(&off_home, &json!({ "smart": {"enabled": false} }));
        with_config_home(&off_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![zo_smuggle(
                    json!({"description": "verify", "prompt": "run tests"}),
                )],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_MODEL_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-model key must be scrubbed even when /smart is disabled"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_REASON_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-reason key must be scrubbed too"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_EFFORT_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-effort key must be scrubbed even when /smart is disabled — \
                 otherwise untrusted JSON could force an Ultra effort (and its paired thinking \
                 budget) onto an arbitrary agent"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_DECISION_META_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-decision-meta key must be scrubbed even when /smart is disabled"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-judged-agent key must be scrubbed even when /smart is disabled — \
                 otherwise untrusted JSON could point verdict credit at an arbitrary route"
            );
        });
        let _ = fs::remove_dir_all(off_home);

        // Case 2/3 — routing ENABLED with a verifier pin: the forged key must not
        // survive alongside an explicit `model` (that member stays provider-family
        // fenced), and for a routable member it is OVERWRITTEN by the host's pin,
        // never left as the crafted value.
        let on_home = temp_config_home("spoof-routing-on");
        write_settings(
            &on_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&on_home, || {
            let mut explicit_member = zo_smuggle(json!({
                "description": "review",
                "prompt": "review",
                "model": "User/Explicit",
            }));
            // Keep the explicit `model` (zo_smuggle only adds smuggle keys).
            assert_eq!(explicit_member["model"], "User/Explicit");
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    explicit_member.take(),
                    zo_smuggle(json!({
                        "description": "verify",
                        "prompt": "run tests",
                        "subagent_type": "Verification",
                    })),
                ],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);
            // Explicit-model member: forged key gone, explicit model untouched
            // (so it flows through the on-wire same-provider-family fence).
            assert_eq!(input.agents[0]["model"], "User/Explicit");
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_MODEL_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-model key must not survive alongside an explicit model"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_EFFORT_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-effort key must not survive alongside an explicit model either"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_DECISION_META_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-decision-meta key must not survive alongside an explicit model"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY)
                    .is_none(),
                "a crafted route-judged-agent key must not survive alongside an explicit model either"
            );
            // Routable member: the host pin replaces the forged value entirely.
            assert_eq!(
                input.agents[1][super::apply::ROUTE_MODEL_SMUGGLE_KEY],
                "Verifier/Model-X",
                "a routable member's forged key is replaced by the host-resolved pin"
            );
            assert_ne!(
                input.agents[1][super::apply::ROUTE_DECISION_META_SMUGGLE_KEY]["role"],
                "forged-role",
                "a routable member's forged decision-meta must be replaced by the host's own classification"
            );
            assert_ne!(
                input.agents[1]
                    .get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY)
                    .cloned()
                    .unwrap_or(Value::Null),
                json!("evil-worker-name"),
                "a routable member's forged judged-agent value must never survive verbatim, \
                 whether the host overwrites it with a real binding or leaves it absent"
            );
        });
        let _ = fs::remove_dir_all(on_home);
    }

    #[test]
    fn planner_bound_judged_agent_smuggled_for_unambiguous_two_member_pair() {
        let config_home = temp_config_home("judged-agent-pair");
        write_settings(&config_home, &json!({ "smart": {"enabled": true} }));
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({
                        "description": "implement the fix",
                        "prompt": "modify the parser to handle the new token",
                        "name": "worker-1"
                    }),
                    json!({
                        "description": "review",
                        "prompt": "review the change",
                        "subagent_type": "code-reviewer",
                        "name": "reviewer-1"
                    }),
                ],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);

            assert_eq!(
                input.agents[1][super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY],
                "worker-1",
                "the sole reviewer-shaped member of an unambiguous 2-member batch is bound to the other member's name"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY)
                    .is_none(),
                "a worker member is never its own judge"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn planner_bound_judged_agent_is_absent_for_batches_larger_than_two() {
        let config_home = temp_config_home("judged-agent-triple");
        write_settings(&config_home, &json!({ "smart": {"enabled": true} }));
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({"description": "implement part a", "prompt": "modify module a", "name": "worker-a"}),
                    json!({"description": "implement part b", "prompt": "modify module b", "name": "worker-b"}),
                    json!({
                        "description": "review",
                        "prompt": "review the change",
                        "subagent_type": "code-reviewer",
                        "name": "reviewer-1"
                    }),
                ],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);

            for agent in &input.agents {
                assert!(
                    agent.get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY).is_none(),
                    "a 3+ member batch is ambiguous — which sibling a reviewer targets is \
                     unknowable, so nothing may be bound: {agent:?}"
                );
            }
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn planner_bound_judged_agent_is_absent_when_the_worker_has_no_name() {
        let config_home = temp_config_home("judged-agent-no-name");
        write_settings(&config_home, &json!({ "smart": {"enabled": true} }));
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({"description": "implement the fix", "prompt": "modify the parser"}),
                    json!({
                        "description": "review",
                        "prompt": "review the change",
                        "subagent_type": "code-reviewer",
                        "name": "reviewer-1"
                    }),
                ],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);

            assert!(
                input.agents[1]
                    .get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY)
                    .is_none(),
                "the worker has no `name` for the spawn loop to resolve later, so the binding \
                 cannot be carried — no guessing"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn planner_bound_judged_agent_is_absent_when_both_members_are_judge_shaped() {
        let config_home = temp_config_home("judged-agent-both-judges");
        write_settings(&config_home, &json!({ "smart": {"enabled": true} }));
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({
                        "description": "verify",
                        "prompt": "run the test suite",
                        "subagent_type": "Verification",
                        "name": "verifier-1"
                    }),
                    json!({
                        "description": "review",
                        "prompt": "review the change",
                        "subagent_type": "code-reviewer",
                        "name": "reviewer-1"
                    }),
                ],
                concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);

            for agent in &input.agents {
                assert!(
                    agent.get(super::apply::ROUTE_JUDGED_AGENT_SMUGGLE_KEY).is_none(),
                    "two judge-shaped members with no worker between them is ambiguous — \
                     neither is unambiguously 'the work' the other judges: {agent:?}"
                );
            }
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn route_shape_solo_routes_trivial_default_work_by_difficulty_not_specialty() {
        let config_home = temp_config_home("solo-single-agent");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"writing": {"mode": "pinned", "model": "Writing/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let input = AgentInput {
                allow_cross_provider: false,
                description: "small typo".to_string(),
                prompt: "rename a label".to_string(),
                subagent_type: None,
                name: None,
                model: None,
                cwd: None,
                schema: None,
                background: Some(false),
                workflow_member: false,
                api_concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
                time_budget: None,
                prior_failures: 0,
                route_reason: None,
                route_role: None,
                route_complexity: None,
                route_risk: None,
                route_source: None,
                route_model: None,
                route_fallback_models: Vec::new(),
                route_effort: None,
                judged_agent: None,
            };
            // A trivial solo Default task can produce a difficulty-based audit route,
            // but smart routing must NOT inject an unrelated specialist model. The
            // spawned agent still inherits the parent/session model.
            let routed = smart_parent_model_for_agent(Some("claude-sonnet-main"), &input);
            assert!(
                routed.is_some(),
                "a trivial solo Default task routes by difficulty rather than inheriting the parent"
            );
            assert!(
                routed
                    .as_ref()
                    .is_none_or(|choice| !choice.reason.as_deref().is_some_and(|reason| reason.contains("Writing/Model-X"))),
                "difficulty audit must not promote a generic task to an unrelated specialist"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn route_shape_preserves_solo_members_in_spawn_fanout() {
        let config_home = temp_config_home("spawn-preserve-solo");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {"roles": {"verifier": {"mode": "pinned", "model": "Verifier/Model-X"}}}
            }),
        );
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({"description": "small typo", "prompt": "rename a label"}),
                    json!({"description": "verify", "prompt": "run tests", "subagent_type": "Verification"}),
                ],
                concurrency: Some(4),
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            // Smart routing may enrich agents with a resolved model + reason, but
            // it must never delete agents or overwrite a user's explicit model.
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);

            assert_eq!(input.agents.len(), 2, "smart routing must not delete user-provided agents");
            assert_eq!(input.agents[0]["description"], "small typo");
            // Neither member gets a user-facing `model` (a smart route is trusted
            // host state, not a user override), yet each carries the resolved
            // model under the smuggle key so the spawn path runs it verbatim.
            assert!(
                input.agents[0].get("model").is_none(),
                "a smart route must not pose as an explicit user model override"
            );
            let routed_0 = input.agents[0][super::apply::ROUTE_MODEL_SMUGGLE_KEY]
                .as_str()
                .expect("a trivial-difficulty member still routes to a resolved model");
            assert_ne!(
                routed_0, "claude-sonnet-main",
                "a smuggled route model is only ever a genuine override, never the parent"
            );
            assert!(
                input.agents[0]
                    .get(super::apply::ROUTE_REASON_SMUGGLE_KEY)
                    .is_some(),
                "the resolved model travels with its human-readable reason"
            );
            assert_eq!(input.agents[1]["subagent_type"], "Verification");
            assert!(
                input.agents[1].get("model").is_none(),
                "a smart route must not pose as an explicit user model override"
            );
            assert_eq!(
                input.agents[1][super::apply::ROUTE_MODEL_SMUGGLE_KEY],
                "Verifier/Model-X",
                "the verifier pin drives the spawned agent's actual model"
            );
            assert_eq!(input.concurrency, Some(4));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn route_shape_preserves_all_user_models_and_agent_count() {
        let config_home = temp_config_home("spawn-preserve-explicit");
        write_settings(
            &config_home,
            &json!({"smart": {"enabled": true}}),
        );
        with_config_home(&config_home, || {
            let mut input = SpawnMultiAgentInput {
                agents: vec![
                    json!({"description": "small typo", "prompt": "rename a label"}),
                    json!({"description": "small typo", "prompt": "rename a label", "model": "User/Explicit"}),
                ],
                concurrency: Some(2),
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            // The implicit Default member may receive route-reason audit metadata;
            // the explicit User/Explicit model must be left untouched.
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut input);

            assert_eq!(input.agents.len(), 2, "explicit and implicit user agents are both preserved");
            assert!(
                input.agents[0].get("model").is_none(),
                "implicit Small-difficulty member inherits the parent/session model"
            );
            assert_eq!(
                input.agents[1]["model"], "User/Explicit",
                "an explicit user model is never overwritten by smart routing"
            );
            assert_eq!(input.concurrency, Some(2));
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn route_shape_routes_fanout_members_without_needs() {
        // A fan-out member carries its lane count from the fanout position. Even
        // with NO synthesized agent-need plan, a parallel spawn must route per-role
        // (ParallelLanes) rather than collapsing to Solo — otherwise every lane
        // silently inherits the parent model (the "sub-agents follow the parent" bug).
        let metadata = classify_task_metadata(
            &TaskMetadataInput::new(Some("general-purpose"), "analyze the module", "perform a deep analysis"),
            RouteRole::Analysis,
        );
        let needs = plan_agent_needs(&metadata);
        assert!(needs.is_empty(), "precondition: this task synthesizes no agent-need plan");
        let shape = select_route_shape(&RouteShapeInput::new(&metadata, &needs).with_independent_lanes(4));
        assert_eq!(shape.shape, RouteShapeKind::ParallelLanes);
    }

    #[test]
    fn smart_router_routes_specialty_and_generic_but_gates_uninferable() {
        let config_home = temp_config_home("single-specialty");
        write_settings(&config_home, &json!({"smart": {"enabled": true}}));
        with_config_home(&config_home, || {
            // An explicit specialty role (inferred Analysis) routes to a specialist
            // model even as a single (non-fanout) Solo spawn.
            let mut analysis = agent_input(None);
            analysis.description = "analyze the module".to_string();
            analysis.prompt = "perform a deep analysis".to_string();
            assert!(
                smart_parent_model_for_agent(Some("claude-sonnet-main"), &analysis).is_some(),
                "explicit specialty single agent should route to a specialist model"
            );

            // A generic (Default-role) single task that carries a difficulty signal
            // routes by difficulty instead of always inheriting the parent — this is
            // the "sub-agents follow the parent" fix.
            let mut generic = agent_input(None);
            generic.description = "do the thing".to_string();
            generic.prompt = "just handle it".to_string();
            assert!(
                smart_parent_model_for_agent(Some("claude-sonnet-main"), &generic).is_some(),
                "a generic single task with a difficulty signal routes by difficulty"
            );

            // Safety boundary: a generic task with NO difficulty signal (Unknown
            // complexity — empty description/prompt) has nothing to route on, so it
            // stays on the parent rather than guessing. This gates *before* any
            // inventory lookup, so it holds regardless of which models are connected.
            let mut uninferable = agent_input(None);
            uninferable.description = String::new();
            uninferable.prompt = String::new();
            assert!(
                smart_parent_model_for_agent(Some("claude-sonnet-main"), &uninferable).is_none(),
                "an uninferable generic task (Unknown complexity) stays on the parent model"
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }


    #[test]
    fn smart_router_subagent_override_wins_over_role_fallback() {
        let config_home = temp_config_home("subagent-wins");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {
                    "subagents": {"Verification": {"mode": "pinned", "model": "Subagent/Model-Z"}},
                    "roles": {"verifier": {"mode": "pinned", "model": "Role/Model-Y"}}
                }
            }),
        );
        with_config_home(&config_home, || {
            let choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification")),
            )
            .expect("smart router records a subagent override route reason");
            assert!(
                choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "reason names the pinned subagent route: {:?}",
                choice.reason
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }


    #[test]
    fn malformed_smart_role_sibling_does_not_disable_valid_role() {
        let config_home = temp_config_home("malformed-role-sibling");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {
                    "roles": {
                        "verifier": {"mode": "pinned", "model": "Verifier/Model-X"},
                        "badrole": {"mode": "pinned", "model": "Ignored/Model"},
                        "coding": {"mode": "manualPreferred", "selector": {"provider": "Verifier"}}
                    }
                }
            }),
        );
        with_config_home(&config_home, || {
            let choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification")),
            )
            .expect("valid role route survives malformed siblings");
            assert!(
                choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "reason names the pinned valid role route: {:?}",
                choice.reason
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn malformed_smart_subagent_sibling_does_not_disable_valid_subagent() {
        let config_home = temp_config_home("malformed-subagent-sibling");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {
                    "subagents": {
                        "Verification": {"mode": "pinned", "model": "Subagent/Model-Z"},
                        "bad subagent": {"mode": "pinned", "model": "Ignored/Model"},
                        "Plan": {"mode": "manualPreferred", "selector": {"provider": "Verifier"}}
                    },
                    "roles": {"verifier": {"mode": "pinned", "model": "Role/Model-Y"}}
                }
            }),
        );
        with_config_home(&config_home, || {
            let choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification")),
            )
            .expect("valid subagent route survives malformed siblings");
            assert!(
                choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "reason names the pinned valid subagent route: {:?}",
                choice.reason
            );
        });
        let _ = fs::remove_dir_all(config_home);
    }


    #[test]
    // Cohesive end-to-end routing scenario in one config context: the Agent path
    // (choice.model → route_model) and the SpawnMultiAgent path (smuggle keys)
    // share the same settings/inventory, so keeping them unsplit reads clearer.
    #[allow(clippy::too_many_lines)]
    fn smart_router_e2e_agent_spawn_and_workflow_like_inputs() {
        let config_home = temp_config_home("e2e-agent-spawn-workflow");
        write_settings(
            &config_home,
            &json!({
                "smart": {"enabled": true},
                "modelRouter": {
                    "subagents": {
                        "Verification": {"mode": "pinned", "model": "Subagent/Model-Z"}
                    },
                    "roles": {
                        "reviewer": {"mode": "pinned", "model": "Role/Model-Y"}
                    }
                }
            }),
        );
        with_config_home(&config_home, || {
            let verification = agent_input(Some("Verification"));
            let verification_choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &verification,
            )
            .expect("verification route reason");
            assert!(
                verification_choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "verification reason names the pin decision: {:?}",
                verification_choice.reason
            );
            // The single-`Agent` path (dispatch.rs) sets `route_model` from THIS
            // choice's model, so the resolved subagent pin must be the model —
            // not just the reason — that the routed agent will actually run on.
            assert_eq!(
                verification_choice.model.as_deref(), Some("Subagent/Model-Z"),
                "the Agent path resolves the Verification subagent pin as the model"
            );

            let mut explicit = agent_input(Some("Verification"));
            explicit.model = Some("User/Explicit".to_string());
            assert!(smart_parent_model_for_agent(Some("claude-sonnet-main"), &explicit).is_none());

            let reviewer = AgentInput {
                allow_cross_provider: false,
                description: "review the diff".to_string(),
                prompt: "perform code review".to_string(),
                subagent_type: None,
                name: None,
                model: None,
                cwd: None,
                schema: None,
                background: Some(false),
                workflow_member: true,
                api_concurrency: None,
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
                time_budget: None,
                prior_failures: 0,
                route_reason: None,
                route_role: None,
                route_complexity: None,
                route_risk: None,
                route_source: None,
                route_model: None,
                route_fallback_models: Vec::new(),
                route_effort: None,
                judged_agent: None,
            };
            let reviewer_choice = smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &reviewer,
            )
            .expect("reviewer route reason");
            assert!(
                reviewer_choice.reason.as_deref().is_some_and(|reason| reason.contains("pin")),
                "reviewer reason names the pin decision: {:?}",
                reviewer_choice.reason
            );
            assert_eq!(
                reviewer_choice.model.as_deref(), Some("Role/Model-Y"),
                "the Agent path resolves the reviewer role pin as the model"
            );

            let mut spawn = SpawnMultiAgentInput {
                agents: vec![
                    json!({"prompt": "run tests", "description": "verify", "subagent_type": "Verification"}),
                    json!({"prompt": "review", "description": "review the code"}),
                    json!({"prompt": "verify", "description": "verify", "subagent_type": "Verification", "model": "User/Explicit"}),
                ],
                concurrency: Some(2),
                parent_session_id: None,
                tool_call_id: None,
                mcp_passthrough: None,
                parent_permission_mode: None,
            };
            apply_smart_models_to_spawn_input(Some("claude-sonnet-main"), &mut spawn);
            assert!(spawn.agents[0].get("model").is_none());
            assert!(spawn.agents[1].get("model").is_none());
            assert_eq!(spawn.agents[2]["model"], "User/Explicit");
            // The resolved models travel under the smuggle key and drive the real
            // spawned-agent model: the Verification subagent pin wins for [0]…
            assert_eq!(
                spawn.agents[0][super::apply::ROUTE_MODEL_SMUGGLE_KEY],
                "Subagent/Model-Z",
                "the Verification subagent pin drives agent[0]'s actual model"
            );
            // …the implicit member routes to a genuine (non-parent) override…
            let routed_1 = spawn.agents[1][super::apply::ROUTE_MODEL_SMUGGLE_KEY]
                .as_str()
                .expect("the implicit member routes to a resolved model");
            assert_ne!(routed_1, "claude-sonnet-main");
            // …and the explicit-model member is never touched by the router.
            assert!(spawn.agents[2]
                .get(super::apply::ROUTE_MODEL_SMUGGLE_KEY)
                .is_none());
            assert!(spawn.agents[0]
                .get(super::apply::ROUTE_REASON_SMUGGLE_KEY)
                .is_some());
            assert!(spawn.agents[1]
                .get(super::apply::ROUTE_REASON_SMUGGLE_KEY)
                .is_some());
            assert!(spawn.agents[2]
                .get(super::apply::ROUTE_REASON_SMUGGLE_KEY)
                .is_none());
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn invalid_global_settings_fail_closed() {
        let config_home = temp_config_home("invalid");
        fs::create_dir_all(&config_home).expect("config dir");
        fs::write(config_home.join("settings.json"), "{not valid json").expect("settings");
        with_config_home(&config_home, || {
            assert!(smart_parent_model_for_agent(
                Some("claude-sonnet-main"),
                &agent_input(Some("Verification"))
            )
            .is_none());
        });
        let _ = fs::remove_dir_all(config_home);
    }

    #[test]
    fn shadow_mode_annotates_the_delta_without_changing_the_real_route() {
        use super::apply::annotate_learned_shadow_delta;
        use super::settings::LearnedSpecialtyMode;
        use runtime::model_router::ModelSource;
        use runtime::{
            route_model, LearnedSpecialtyHint, ModelCapability, ModelDescriptor, ModelInventory,
            ModelTier, RouteFeedbackHint, RoutePolicyContext, RouteRequest,
        };

        let inventory = ModelInventory::new(
            "main-anthropic",
            vec![
                ModelDescriptor::new("main-anthropic", "anthropic", "claude")
                    .source(ModelSource::CurrentMainModel)
                    .capabilities([ModelCapability::Default])
                    .tiers([ModelTier::Balanced])
                    .release_rank(1),
                // Neither candidate carries the Coding cold-start seed
                // (`gpt`/`claude` only), so release_rank alone decides the
                // seed-only baseline — isolating the shadow hint's effect.
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
        let context = RoutePolicyContext {
            feedback: RouteFeedbackHint::disabled(),
            learned_specialty: LearnedSpecialtyHint::disabled(),
            ..RoutePolicyContext::default()
        };
        let request = RouteRequest::new(RouteRole::Coding, "main-anthropic").with_context(context);
        let real_decision = route_model(&request, &inventory);
        assert_eq!(
            real_decision.resolved_model, "model-a",
            "seed-only baseline: the real request always carries an empty hint outside `on` mode"
        );

        let computed_hint =
            LearnedSpecialtyHint::default().with_entry(RouteRole::Coding, "model-b", 90, 1000);

        // Shadow: the real pick must NOT move, but the delta must be recorded.
        let annotated = annotate_learned_shadow_delta(
            real_decision.clone(),
            &request,
            &inventory,
            LearnedSpecialtyMode::Shadow,
            &computed_hint,
        );
        assert_eq!(annotated.resolved_model, "model-a", "shadow mode must never change the real pick");
        assert!(
            annotated
                .audit
                .guardrails
                .iter()
                .any(|line| line == "learned-shadow-differs:model-b"),
            "shadow delta must be recorded: {:?}",
            annotated.audit.guardrails
        );

        // Off/On are no-ops for THIS fn: `off` never computes a hint at all
        // (mirrored here by calling with the mode explicitly, still over the
        // same non-empty `computed_hint` to prove the mode check itself
        // gates it, not just hint emptiness); `on` already routed for real
        // via the injected hint, so there is nothing left for this fn to add.
        let off_noop = annotate_learned_shadow_delta(
            real_decision.clone(),
            &request,
            &inventory,
            LearnedSpecialtyMode::Off,
            &computed_hint,
        );
        assert!(off_noop
            .audit
            .guardrails
            .iter()
            .all(|line| !line.starts_with("learned-shadow-differs:")));

        let on_noop = annotate_learned_shadow_delta(
            real_decision,
            &request,
            &inventory,
            LearnedSpecialtyMode::On,
            &computed_hint,
        );
        assert!(on_noop
            .audit
            .guardrails
            .iter()
            .all(|line| !line.starts_with("learned-shadow-differs:")));
    }

    #[test]
    fn shadow_mode_is_a_pure_no_op_when_the_computed_hint_is_empty() {
        use super::apply::annotate_learned_shadow_delta;
        use super::settings::LearnedSpecialtyMode;
        use runtime::{route_model, LearnedSpecialtyHint, ModelInventory, RoutePolicyContext, RouteRequest};

        let inventory = ModelInventory::new("main-anthropic", Vec::new());
        let request = RouteRequest::new(RouteRole::Coding, "main-anthropic")
            .with_context(RoutePolicyContext::default());
        let decision = route_model(&request, &inventory);
        let annotated = annotate_learned_shadow_delta(
            decision.clone(),
            &request,
            &inventory,
            LearnedSpecialtyMode::Shadow,
            &LearnedSpecialtyHint::disabled(),
        );
        assert_eq!(
            annotated, decision,
            "an empty computed hint must skip the second route_model call entirely — byte-identical decision"
        );
    }
