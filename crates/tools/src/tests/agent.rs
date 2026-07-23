//! Agent / `SpawnMultiAgent` / council / worker lifecycle behavior.
//!
//! Split out of the old flat `tests.rs` (4,133 lines) by domain;
//! shared fixtures live in the parent module.

use super::*;

/// The Agent schema advertises `model` + `allow_cross_provider` so an
/// explicit user ask ("opus 에이전트로 실행해") has a legitimate, auditable
/// path. Before this, the field was accepted by the parser but hidden from
/// the schema, and a cross-family request dead-ended: the live push-session
/// incident saw the model flail into a Config override and end up running a
/// different model under an "opus-agent" name.
#[test]
fn agent_schema_advertises_model_with_cross_provider_guard() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "Agent")
        .expect("Agent spec should be present");
    let properties = &spec.input_schema["properties"];
    assert_eq!(properties["model"]["type"], "string");
    assert_eq!(properties["allow_cross_provider"]["type"], "boolean");
    assert!(
        properties["allow_cross_provider"]["description"]
            .as_str()
            .is_some_and(|text| text.contains("ONLY when the user explicitly asked")),
        "the guard flag must document its user-explicit-ask contract"
    );
    assert!(properties.get("subagent_type").is_some());
    assert!(properties.get("description").is_some());
    assert!(properties.get("prompt").is_some());
}

#[test]
fn subagent_toolset_classes_follow_static_capabilities() {
    assert_eq!(crate::subagent_toolset_class("general-purpose"), "full");
    assert_eq!(crate::subagent_toolset_class("Explore"), "read-only");
    assert_eq!(crate::subagent_toolset_class("refactor"), "edit");
    assert_eq!(crate::subagent_toolset_class("my-custom-agent"), "custom");
}

#[test]
fn agent_background_omitted_parses_to_none_for_host_default() {
    // `background` omitted must parse to `None` — the dispatch layer resolves
    // it against `ToolContext::background_agent_default` (detached in the
    // interactive main session, blocking for sub-agents/headless). A plain
    // `false` default here would erase the "unspecified" signal and pin every
    // caller to blocking regardless of host.
    let omitted: AgentInput =
        serde_json::from_value(serde_json::json!({"description": "d", "prompt": "p"}))
            .expect("omitted background parses");
    assert_eq!(omitted.background, None);

    let explicit: AgentInput = serde_json::from_value(
        serde_json::json!({"description": "d", "prompt": "p", "background": false}),
    )
    .expect("explicit background parses");
    assert_eq!(explicit.background, Some(false));
}

#[test]
fn background_agent_default_is_off_until_an_interactive_host_opts_in() {
    // Fresh contexts (sub-agent executors, headless) must default to blocking:
    // nothing consumes the completion channel there, so a detached result
    // would be silently lost. The shared cell propagates the interactive
    // host's opt-in to every registry clone.
    let context = ToolContext::new();
    assert!(!context.background_agent_default());
    let clone = context.clone();
    context.set_background_agent_default(true);
    assert!(clone.background_agent_default());
}

#[test]
fn agent_description_routes_parallel_work_to_spawn_multi_agent() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "Agent")
        .expect("Agent spec should be present");
    let description = spec.description;
    assert!(
        description.contains("Use `Agent` for a single focused specialist"),
        "Agent must be described as the one-specialist primitive"
    );
    assert!(
        description.contains("do not treat several blocking `Agent` calls as a parallel swarm"),
        "Agent must explicitly reject several blocking Agent calls as a swarm"
    );
    assert!(
        description.contains("For real parallel fan-out")
            && description.contains("use `SpawnMultiAgent`"),
        "Agent description must route real fan-out to SpawnMultiAgent"
    );
    assert!(
        description.contains("dependent plan→implement→verify")
            && description.contains("use `Workflow`"),
        "Agent description must route dependent pipeline work to Workflow"
    );
    assert!(
        !description.contains("emit multiple Agent calls")
            && !description.contains("to run several agents at once"),
        "Agent must not teach the model that multiple blocking Agent calls form a parallel swarm"
    );
    assert!(
        description.contains("work directly instead of spawning anything"),
        "Agent must tell the model to handle simple asks directly (proportionality)"
    );
}

/// Proportionality contract in the spawn tools' own descriptions (CC-style:
/// the when-NOT-to-use bar travels with the tool, and the model applies it
/// per ask — there is no per-turn mode reminder or host difficulty classifier
/// steering orchestration).
#[test]
fn spawn_multi_agent_description_sizes_fanout_to_the_ask() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "SpawnMultiAgent")
        .map(|spec| spec.description)
        .expect("SpawnMultiAgent spec should be present");
    assert!(
        spec.contains("Size the fan-out to the ask, not the session mode"),
        "fan-out size must track the ask, never the effort mode"
    );
    assert!(
        spec.contains("never spawn a swarm or a multi-perspective verification panel")
            && spec.contains("simple question"),
        "swarms/panels must be banned for simple questions, lookups, and bounded fixes"
    );
}

#[test]
fn agent_descriptions_match_parent_model_inheritance_policy() {
    let specs = mvp_tool_specs();
    for tool_name in ["Agent", "SpawnMultiAgent"] {
        let spec = specs
            .iter()
            .find(|spec| spec.name == tool_name)
            .expect("tool spec should be present");
        let description = spec.description;
        assert!(
            description.contains("inherit the active parent/session model"),
            "{tool_name} must describe parent/session model inheritance"
        );
        assert!(
            description.contains("same provider family"),
            "{tool_name} must describe same-provider-family explicit model limits"
        );
        assert!(
            description.contains("reasoning") && description.contains("budget"),
            "{tool_name} must describe difficulty as effort/budget tuning, not model routing"
        );
        for stale in [
            "quick OpenAI/GPT work uses Spark",
            "quick work to a cheap model",
            "hard work to a strong one",
            "auto-route by difficulty",
        ] {
            assert!(
                !description.contains(stale),
                "{tool_name} description must not advertise stale model auto-routing phrase: {stale}"
            );
        }
    }
}

#[test]
fn spawn_multi_agent_schema_advertises_per_agent_model_override() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "SpawnMultiAgent")
        .expect("SpawnMultiAgent spec should be present");
    let per_agent = &spec.input_schema["properties"]["agents"]["items"]["properties"];
    assert_eq!(
        per_agent["model"]["type"], "string",
        "the model can override the inherited model per sub-agent within the same provider family (BUG-D5)"
    );
    assert!(per_agent.get("name").is_some());
    assert!(per_agent.get("description").is_some());
    assert!(per_agent.get("prompt").is_some());
}

#[test]
fn spawn_multi_agent_schema_exposes_fanout_limit() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "SpawnMultiAgent")
        .expect("SpawnMultiAgent spec should be present");
    let agents = &spec.input_schema["properties"]["agents"];

    assert_eq!(
        agents["maxItems"].as_u64(),
        Some(MAX_SPAWN_MULTI_AGENT_AGENTS as u64)
    );
}

#[test]
fn spawn_multi_agent_schema_exposes_concurrency_window() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "SpawnMultiAgent")
        .expect("SpawnMultiAgent spec should be present");
    let concurrency = &spec.input_schema["properties"]["concurrency"];
    assert_eq!(concurrency["type"], json!("integer"));
    assert_eq!(concurrency["minimum"].as_u64(), Some(1));
    assert_eq!(
        concurrency["maximum"].as_u64(),
        Some(MAX_SPAWN_MULTI_AGENT_AGENTS as u64),
        "the permit-before-spawn window must not exceed the hard agent cap"
    );
}

#[test]
fn spawn_multi_agent_respects_concurrency() {
    // The permit-before-spawn window bounds how many agents run at once.
    // Unset → the workflow execution bound (min(16, cores-2), the CC model);
    // a request is clamped to [1, cap] and may widen beyond the default.
    assert_eq!(
        effective_spawn_window(None, 4),
        workflow_concurrency_limit().min(4),
        "the default window is the execution bound, never more than the count"
    );
    assert_eq!(
        effective_spawn_window(None, 100),
        workflow_concurrency_limit().min(MAX_SPAWN_MULTI_AGENT_AGENTS),
        "a large fan-out defaults to the execution bound and queues the rest"
    );
    assert_eq!(
        effective_spawn_window(Some(2), 4),
        2,
        "a request is honored"
    );
    assert_eq!(
        effective_spawn_window(Some(0), 4),
        1,
        "zero is clamped up to one (never a zero-width window)"
    );
    assert_eq!(
        effective_spawn_window(Some(999), 4),
        4,
        "a request never exceeds the agent count or the cap"
    );
    assert_eq!(
        effective_spawn_window(Some(MAX_SPAWN_MULTI_AGENT_AGENTS), 100),
        MAX_SPAWN_MULTI_AGENT_AGENTS,
        "an explicit request may widen the window up to the hard cap"
    );
}

#[test]
fn spawn_multi_agent_rejects_too_many_agents_before_spawning() {
    let agents = (0..=MAX_SPAWN_MULTI_AGENT_AGENTS)
        .map(|index| json!({ "prompt": format!("candidate {index}") }))
        .collect::<Vec<_>>();
    let error = run_tool("SpawnMultiAgent", &json!({ "agents": agents }))
        .expect_err("oversized fan-out should be rejected before spawning");

    assert!(matches!(
        error,
        ToolError::InvalidInput(message)
            if message.contains("at most")
                && message.contains(&MAX_SPAWN_MULTI_AGENT_AGENTS.to_string())
    ));
}

#[test]
fn council_schema_hides_candidate_source_identity() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "Council")
        .expect("Council spec should be present");
    let properties = &spec.input_schema["properties"]["candidates"]["items"]["properties"];

    assert!(properties.get("text").is_some());
    assert!(properties.get("status").is_some());
    assert!(
        properties.get("model").is_none(),
        "candidate model identity must not be part of the judge-facing schema"
    );
    assert!(
        properties.get("name").is_none(),
        "candidate agent identity must not be part of the judge-facing schema"
    );
}

#[test]
fn council_schema_exposes_budget_limits() {
    let spec = mvp_tool_specs()
        .iter()
        .find(|spec| spec.name == "Council")
        .expect("Council spec should be present");
    let candidates = &spec.input_schema["properties"]["candidates"];
    let text = &candidates["items"]["properties"]["text"];

    assert_eq!(
        candidates["maxItems"].as_u64(),
        Some(MAX_COUNCIL_CANDIDATES as u64)
    );
    assert_eq!(
        text["maxLength"].as_u64(),
        Some(MAX_COUNCIL_CANDIDATE_CHARS as u64)
    );
}

#[test]
fn council_tool_selects_self_consistent_majority() {
    let output = run_tool(
        "Council",
        &json!({
            "candidates": [
                { "text": "Keep provider routing provider-aware" },
                { "text": "keep provider routing provider-aware", "status": "completed" },
                { "text": "Rewrite the whole runtime" }
            ]
        }),
    )
    .expect("Council should execute");
    let value: serde_json::Value = serde_json::from_str(&output).expect("json output");

    assert_eq!(value["source_hidden"], true);
    assert_eq!(value["llm_judge_allowed"], false);
    assert_eq!(value["llm_judge_call_limit"], 0);
    assert_eq!(value["outcome"]["type"], "best_of");
    assert_eq!(value["outcome"]["winner_index"], 0);
    assert_eq!(value["outcome"]["supporting_indices"], json!([0, 1]));
}

#[test]
fn council_tool_allows_one_llm_judge_only_for_actionable_ties() {
    let output = run_tool(
        "Council",
        &json!({
            "candidates": [
                { "text": "A" },
                { "text": "B" },
                { "text": "C" }
            ]
        }),
    )
    .expect("Council should execute");
    let value: serde_json::Value = serde_json::from_str(&output).expect("json output");

    assert_eq!(value["outcome"]["type"], "tie");
    assert_eq!(value["llm_judge_allowed"], true);
    assert_eq!(
        value["llm_judge_call_limit"].as_u64(),
        Some(MAX_COUNCIL_LLM_JUDGE_CALLS as u64)
    );
}

#[test]
fn council_tool_rejects_oversized_candidate_text() {
    let error = run_tool(
        "Council",
        &json!({
            "candidates": [
                { "text": "x".repeat(MAX_COUNCIL_CANDIDATE_CHARS + 1) }
            ]
        }),
    )
    .expect_err("oversized candidate should be rejected");

    assert!(matches!(
        error,
        ToolError::InvalidInput(message)
            if message.contains("candidate 0")
                && message.contains(&MAX_COUNCIL_CANDIDATE_CHARS.to_string())
    ));
}

#[test]
fn worker_tools_gate_prompt_delivery_until_ready_and_support_auto_trust() {
    let created = run_tool(
        "WorkerCreate",
        &json!({
            "cwd": "/tmp/worktree/repo",
            "trusted_roots": ["/tmp/worktree"]
        }),
    )
    .expect("WorkerCreate should succeed");
    let created_output: serde_json::Value = serde_json::from_str(&created).expect("json");
    let worker_id = created_output["worker_id"]
        .as_str()
        .expect("worker id")
        .to_string();
    assert_eq!(created_output["status"], "spawning");
    assert_eq!(created_output["trust_auto_resolve"], true);

    let gated = run_tool(
        "WorkerSendPrompt",
        &json!({
            "worker_id": worker_id,
            "prompt": "ship the change"
        }),
    )
    .expect_err("prompt delivery before ready should fail");
    assert!(gated.to_string().contains("not ready for prompt delivery"));

    let observed = run_tool(
        "WorkerObserve",
        &json!({
            "worker_id": created_output["worker_id"],
            "screen_text": "Do you trust the files in this folder?\n1. Yes, proceed\n2. No"
        }),
    )
    .expect("WorkerObserve should auto-resolve trust");
    let observed_output: serde_json::Value = serde_json::from_str(&observed).expect("json");
    assert_eq!(observed_output["status"], "spawning");
    assert_eq!(observed_output["trust_gate_cleared"], true);
    assert_eq!(
        observed_output["events"][1]["payload"]["type"],
        "trust_prompt"
    );
    assert_eq!(
        observed_output["events"][2]["payload"]["resolution"],
        "auto_allowlisted"
    );

    let ready = run_tool(
        "WorkerObserve",
        &json!({
            "worker_id": created_output["worker_id"],
            "screen_text": "Ready for your input\n>"
        }),
    )
    .expect("WorkerObserve should mark worker ready");
    let ready_output: serde_json::Value = serde_json::from_str(&ready).expect("json");
    assert_eq!(ready_output["status"], "ready_for_prompt");

    let await_ready = run_tool(
        "WorkerAwaitReady",
        &json!({
            "worker_id": created_output["worker_id"]
        }),
    )
    .expect("WorkerAwaitReady should succeed");
    let await_ready_output: serde_json::Value = serde_json::from_str(&await_ready).expect("json");
    assert_eq!(await_ready_output["ready"], true);

    let accepted = run_tool(
        "WorkerSendPrompt",
        &json!({
            "worker_id": created_output["worker_id"],
            "prompt": "ship the change"
        }),
    )
    .expect("WorkerSendPrompt should succeed after ready");
    let accepted_output: serde_json::Value = serde_json::from_str(&accepted).expect("json");
    assert_eq!(accepted_output["status"], "running");
    assert_eq!(accepted_output["prompt_delivery_attempts"], 1);
    assert_eq!(accepted_output["prompt_in_flight"], true);
}

#[test]
fn worker_tools_detect_misdelivery_and_arm_prompt_replay() {
    let created = run_tool(
        "WorkerCreate",
        &json!({
            "cwd": "/tmp/repo/worker-misdelivery"
        }),
    )
    .expect("WorkerCreate should succeed");
    let created_output: serde_json::Value = serde_json::from_str(&created).expect("json");
    let worker_id = created_output["worker_id"]
        .as_str()
        .expect("worker id")
        .to_string();

    run_tool(
        "WorkerObserve",
        &json!({
            "worker_id": worker_id,
            "screen_text": "Ready for input\n>"
        }),
    )
    .expect("worker should become ready");

    run_tool(
        "WorkerSendPrompt",
        &json!({
            "worker_id": worker_id,
            "prompt": "Investigate flaky boot"
        }),
    )
    .expect("prompt send should succeed");

    let recovered = run_tool(
        "WorkerObserve",
        &json!({
            "worker_id": worker_id,
            "screen_text": "% Investigate flaky boot\nzsh: command not found: Investigate"
        }),
    )
    .expect("misdelivery observe should succeed");
    let recovered_output: serde_json::Value = serde_json::from_str(&recovered).expect("json");
    assert_eq!(recovered_output["status"], "ready_for_prompt");
    assert_eq!(recovered_output["last_error"]["kind"], "prompt_delivery");
    assert_eq!(recovered_output["replay_prompt"], "Investigate flaky boot");
    assert_eq!(
        recovered_output["events"][3]["payload"]["observed_target"],
        "shell"
    );
    assert_eq!(
        recovered_output["events"][4]["payload"]["recovery_armed"],
        true
    );

    let replayed = run_tool(
        "WorkerSendPrompt",
        &json!({
            "worker_id": worker_id
        }),
    )
    .expect("WorkerSendPrompt should replay recovered prompt");
    let replayed_output: serde_json::Value = serde_json::from_str(&replayed).expect("json");
    assert_eq!(replayed_output["status"], "running");
    assert_eq!(replayed_output["prompt_delivery_attempts"], 2);
    assert_eq!(replayed_output["prompt_in_flight"], true);
}

#[test]
fn subagent_tool_executor_denies_blocked_tool_before_dispatch() {
    // given
    let policy = permission_policy_for_mode(PermissionMode::ReadOnly);
    let mut executor = SubagentToolExecutor::new(BTreeSet::from([String::from("write_file")]))
        .with_enforcer(PermissionEnforcer::new(policy));

    // when
    let error = executor
        .execute(
            "write_file",
            &json!({
                "path": "blocked.txt",
                "content": "blocked"
            })
            .to_string(),
        )
        .expect_err("subagent write tool should be denied before dispatch");

    // then
    assert!(error
        .to_string()
        .contains("requires workspace-write permission"));
}

#[test]
#[allow(clippy::too_many_lines)] // cohesive end-to-end handoff-metadata persistence scenario
fn agent_persists_handoff_metadata() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = temp_path("agent-store");
    std::env::set_var("ZO_AGENT_STORE", &dir);
    // The `run_tool("Agent")` calls below spawn *real* agent threads and
    // *wait* for their completion. Starve credential resolution completely —
    // empty env keys (treated as absent), keychain disabled via the env-lock
    // init, and an empty credentials home — so each agent fails fast at
    // construction instead of discovering the developer's real tokens and
    // burning retry back-offs against an unreachable API. The dead loopback
    // base URL is belt-and-suspenders for anything that still dials out.
    let original_api_key = std::env::var_os("ANTHROPIC_API_KEY");
    let original_auth_token = std::env::var_os("ANTHROPIC_AUTH_TOKEN");
    let original_base_url = std::env::var_os("ANTHROPIC_BASE_URL");
    let original_config_home = std::env::var_os("ZO_CONFIG_HOME");
    std::env::set_var("ANTHROPIC_API_KEY", "");
    std::env::set_var("ANTHROPIC_AUTH_TOKEN", "");
    std::env::set_var("ANTHROPIC_BASE_URL", "http://127.0.0.1:9");
    std::env::set_var("ZO_CONFIG_HOME", temp_path("agent-config-home"));
    let captured = Arc::new(Mutex::new(None::<AgentJob>));
    let captured_for_spawn = Arc::clone(&captured);

    let manifest = execute_agent_with_spawn(
        AgentInput {
            allow_cross_provider: false,
            description: "Audit the branch".to_string(),
            prompt: "Check tests and outstanding work.".to_string(),
            subagent_type: Some("Explore".to_string()),
            name: Some("ship-audit".to_string()),
            model: None,
            cwd: None,
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: Some("session-a".to_string()),
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
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
        },
        move |job| {
            *captured_for_spawn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(job);
            Ok(())
        },
    )
    .expect("Agent should succeed");
    std::env::remove_var("ZO_AGENT_STORE");

    assert_eq!(manifest.name, "ship-audit");
    assert_eq!(manifest.subagent_type.as_deref(), Some("Explore"));
    assert_eq!(manifest.status, "running");
    assert!(!manifest.created_at.is_empty());
    assert!(manifest.started_at.is_some());
    assert!(manifest.completed_at.is_none());
    let contents = std::fs::read_to_string(&manifest.output_file).expect("agent file exists");
    let manifest_contents =
        std::fs::read_to_string(&manifest.manifest_file).expect("manifest file exists");
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_contents).expect("manifest should be valid json");
    assert!(contents.contains("Audit the branch"));
    assert!(contents.contains("Check tests and outstanding work."));
    assert!(manifest_contents.contains("\"subagentType\": \"Explore\""));
    assert!(manifest_contents.contains("\"parentSessionId\": \"session-a\""));
    assert_eq!(manifest_json["parentSessionId"], "session-a");
    assert!(manifest_contents.contains("\"status\": \"running\""));
    assert_eq!(manifest_json["laneEvents"][0]["event"], "lane.started");
    assert_eq!(manifest_json["laneEvents"][0]["status"], "running");
    assert!(manifest_json["currentBlocker"].is_null());
    let captured_job = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("spawn job should be captured");
    assert_eq!(captured_job.prompt, "Check tests and outstanding work.");
    assert!(captured_job.allowed_tools.contains("read_file"));
    assert!(!captured_job.allowed_tools.contains("Agent"));

    let normalized = run_tool(
        "Agent",
        &json!({
            "description": "Verify the branch",
            "prompt": "Check tests.",
            "subagent_type": "explorer"
        }),
    )
    .expect("Agent should normalize built-in aliases");
    let normalized_output: serde_json::Value =
        serde_json::from_str(&normalized).expect("valid json");
    assert_eq!(normalized_output["subagentType"], "Explore");

    let named = run_tool(
        "Agent",
        &json!({
            "description": "Review the branch",
            "prompt": "Inspect diff.",
            "name": "Ship Audit!!!"
        }),
    )
    .expect("Agent should normalize explicit names");
    let named_output: serde_json::Value = serde_json::from_str(&named).expect("valid json");
    assert_eq!(named_output["name"], "ship-audit");
    for (key, original) in [
        ("ANTHROPIC_API_KEY", original_api_key),
        ("ANTHROPIC_AUTH_TOKEN", original_auth_token),
        ("ANTHROPIC_BASE_URL", original_base_url),
        ("ZO_CONFIG_HOME", original_config_home),
    ] {
        match original {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn agent_input_cwd_threads_into_job() {
    use std::path::{Path, PathBuf};
    // 8a-②: a per-agent cwd on `AgentInput` must reach the spawned `AgentJob`
    // so `build_agent_runtime` can confine the sub-agent's tools to its
    // worktree. `None` (every other test) keeps the process-cwd default.
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = temp_path("agent-cwd");
    std::env::set_var("ZO_AGENT_STORE", &dir);
    let captured = Arc::new(Mutex::new(None::<AgentJob>));
    let captured_for_spawn = Arc::clone(&captured);

    execute_agent_with_spawn(
        AgentInput {
            allow_cross_provider: false,
            description: "Worktree task".to_string(),
            prompt: "do isolated work".to_string(),
            subagent_type: None,
            name: Some("wt-iso".to_string()),
            model: None,
            cwd: Some(PathBuf::from("/tmp/zo-wt-iso")),
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
            time_budget: Some(std::time::Duration::from_secs(7)),
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
        },
        move |job| {
            *captured_for_spawn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(job);
            Ok(())
        },
    )
    .expect("Agent should succeed");
    std::env::remove_var("ZO_AGENT_STORE");

    let captured_job = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("spawn job should be captured");
    assert_eq!(
        captured_job.cwd.as_deref(),
        Some(Path::new("/tmp/zo-wt-iso")),
        "AgentInput.cwd must thread into AgentJob.cwd"
    );
    assert_eq!(
        captured_job.time_budget,
        Some(std::time::Duration::from_secs(7)),
        "AgentInput.time_budget must thread into AgentJob.time_budget"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn agent_hook_config_threads_into_job() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = temp_path("agent-hooks");
    std::env::set_var("ZO_AGENT_STORE", &dir);
    let hook_config = runtime::RuntimeHookConfig::default()
        .with_subagent_lifecycle(
            vec!["echo start".to_string()],
            vec!["echo stop".to_string()],
        )
        // Main-agent-only rule: must be stripped by the sub-agent view below.
        .with_turn_end(vec!["echo stop-gate".to_string()]);
    let captured = Arc::new(Mutex::new(None::<AgentJob>));
    let captured_for_spawn = Arc::clone(&captured);

    execute_agent_with_spawn_and_parent_model_and_hooks(
        AgentInput {
            allow_cross_provider: false,
            description: "Hooked task".to_string(),
            prompt: "check hooks".to_string(),
            subagent_type: None,
            name: Some("hooked".to_string()),
            model: None,
            cwd: None,
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
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
        },
        move |job| {
            *captured_for_spawn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(job);
            Ok(())
        },
        None,
        None,
        Some(&hook_config),
    )
    .expect("Agent should succeed");
    std::env::remove_var("ZO_AGENT_STORE");

    let captured_job = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("spawn job should be captured");
    assert_eq!(captured_job.hook_config.subagent_start().len(), 1);
    assert_eq!(captured_job.hook_config.subagent_stop().len(), 1);
    // The job carries the sub-agent VIEW of the parent hooks: Stop/TurnEnd is a
    // main-agent contract (CC parity), so it must not reach the sub-agent.
    assert!(captured_job.hook_config.turn_end().is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn agent_manifest_records_requested_and_resolved_model_separately() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = temp_path("agent-resolved-model");
    std::env::set_var("ZO_AGENT_STORE", &dir);
    let captured = Arc::new(Mutex::new(None::<AgentJob>));
    let captured_for_spawn = Arc::clone(&captured);

    let output = execute_agent_with_spawn_and_parent_model_and_hooks(
        AgentInput {
            allow_cross_provider: false,
            description: "Review hard provider routing bug".to_string(),
            prompt: "debug a hard correctness issue".to_string(),
            subagent_type: None,
            name: Some("resolved-model".to_string()),
            model: None,
            cwd: None,
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
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
        },
        move |job| {
            *captured_for_spawn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(job);
            Ok(())
        },
        Some("gpt"),
        None,
        None,
    )
    .expect("Agent should succeed");
    std::env::remove_var("ZO_AGENT_STORE");

    let manifest_text =
        std::fs::read_to_string(&output.manifest_file).expect("manifest should exist");
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("manifest json");
    let requested = manifest_json["requestedModel"]
        .as_str()
        .expect("requested model");
    let resolved = manifest_json["resolvedModel"]
        .as_str()
        .expect("resolved model");
    let model = manifest_json["model"].as_str().expect("runtime model");

    assert_eq!(requested, "gpt");
    assert_eq!(
        resolved, "gpt",
        "CC parity (2026-06-11): the sub-agent inherits the parent model token \
         verbatim; alias resolution happens downstream in the provider client"
    );
    assert_eq!(
        model, resolved,
        "runtime model and resolved model should describe the same model"
    );

    let captured_job = captured
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("spawn job should be captured");
    assert_eq!(
        captured_job.manifest.requested_model.as_deref(),
        Some("gpt")
    );
    assert_eq!(
        captured_job.manifest.resolved_model.as_deref(),
        Some(resolved)
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[allow(clippy::too_many_lines)]
fn agent_fake_runner_can_persist_completion_and_failure() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = temp_path("agent-runner");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let completed = execute_agent_with_spawn(
        AgentInput {
            allow_cross_provider: false,
            description: "Complete the task".to_string(),
            prompt: "Do the work".to_string(),
            subagent_type: Some("Explore".to_string()),
            name: Some("complete-task".to_string()),
            model: Some("claude-sonnet-4-6".to_string()),
            cwd: None,
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
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
        },
        |job| {
            Ok(persist_agent_terminal_state(
                &job.manifest,
                "completed",
                Some("Finished successfully"),
                None,
            )?)
        },
    )
    .expect("completed agent should succeed");

    let completed_manifest =
        std::fs::read_to_string(&completed.manifest_file).expect("completed manifest should exist");
    let completed_manifest_json: serde_json::Value =
        serde_json::from_str(&completed_manifest).expect("completed manifest json");
    let completed_output =
        std::fs::read_to_string(&completed.output_file).expect("completed output should exist");
    assert!(completed_manifest.contains("\"status\": \"completed\""));
    assert!(completed_output.contains("Finished successfully"));
    assert_eq!(
        completed_manifest_json["laneEvents"][0]["event"],
        "lane.started"
    );
    assert_eq!(
        completed_manifest_json["laneEvents"][1]["event"],
        "lane.finished"
    );
    assert!(completed_manifest_json["currentBlocker"].is_null());

    let failed = execute_agent_with_spawn(
        AgentInput {
            allow_cross_provider: false,
            description: "Fail the task".to_string(),
            prompt: "Do the failing work".to_string(),
            subagent_type: Some("Verification".to_string()),
            name: Some("fail-task".to_string()),
            model: None,
            cwd: None,
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
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
        },
        |job| {
            Ok(persist_agent_terminal_state(
                &job.manifest,
                "failed",
                None,
                Some(String::from("tool failed: simulated failure")),
            )?)
        },
    )
    .expect("failed agent should still spawn");

    let failed_manifest =
        std::fs::read_to_string(&failed.manifest_file).expect("failed manifest should exist");
    let failed_manifest_json: serde_json::Value =
        serde_json::from_str(&failed_manifest).expect("failed manifest json");
    let failed_output =
        std::fs::read_to_string(&failed.output_file).expect("failed output should exist");
    assert!(failed_manifest.contains("\"status\": \"failed\""));
    assert!(failed_manifest.contains("simulated failure"));
    assert!(failed_output.contains("simulated failure"));
    assert!(failed_output.contains("failure_class: tool_runtime"));
    assert_eq!(
        failed_manifest_json["currentBlocker"]["failureClass"],
        "tool_runtime"
    );
    assert_eq!(
        failed_manifest_json["laneEvents"][1]["event"],
        "lane.blocked"
    );
    assert_eq!(
        failed_manifest_json["laneEvents"][2]["event"],
        "lane.failed"
    );
    assert_eq!(
        failed_manifest_json["laneEvents"][2]["failureClass"],
        "tool_runtime"
    );

    let spawn_error = execute_agent_with_spawn(
        AgentInput {
            allow_cross_provider: false,
            description: "Spawn error task".to_string(),
            prompt: "Never starts".to_string(),
            subagent_type: None,
            name: Some("spawn-error".to_string()),
            model: None,
            cwd: None,
            schema: None,
            workflow_member: false,
            background: Some(false),
            parent_session_id: None,
            tool_call_id: None,
            mcp_passthrough: None,
            parent_permission_mode: None,
            api_concurrency: None,
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
        },
        |_| Err(ToolError::Execution("thread creation failed".into())),
    )
    .expect_err("spawn errors should surface");
    assert!(spawn_error
        .to_string()
        .contains("failed to spawn sub-agent"));
    let spawn_error_manifest = std::fs::read_dir(&dir)
        .expect("agent dir should exist")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .find_map(|path| {
            let contents = std::fs::read_to_string(&path).ok()?;
            contents
                .contains("\"name\": \"spawn-error\"")
                .then_some(contents)
        })
        .expect("failed manifest should still be written");
    let spawn_error_manifest_json: serde_json::Value =
        serde_json::from_str(&spawn_error_manifest).expect("spawn error manifest json");
    assert!(spawn_error_manifest.contains("\"status\": \"failed\""));
    assert!(spawn_error_manifest.contains("thread creation failed"));
    assert_eq!(
        spawn_error_manifest_json["currentBlocker"]["failureClass"],
        "infra"
    );

    std::env::remove_var("ZO_AGENT_STORE");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn lane_failure_taxonomy_normalizes_common_blockers() {
    let cases = [
        (
            "prompt delivery failed in tmux pane",
            LaneFailureClass::PromptDelivery,
        ),
        (
            "trust prompt is still blocking startup",
            LaneFailureClass::TrustGate,
        ),
        (
            "branch stale against main after divergence",
            LaneFailureClass::BranchDivergence,
        ),
        (
            "compile failed after cargo check",
            LaneFailureClass::Compile,
        ),
        ("targeted tests failed", LaneFailureClass::Test),
        ("plugin bootstrap failed", LaneFailureClass::PluginStartup),
        ("mcp handshake timed out", LaneFailureClass::McpHandshake),
        (
            "mcp startup failed before listing tools",
            LaneFailureClass::McpStartup,
        ),
        (
            "gateway routing rejected the request",
            LaneFailureClass::GatewayRouting,
        ),
        (
            "api returned 401 Unauthorized (authentication_error): Invalid authentication credentials",
            LaneFailureClass::GatewayRouting,
        ),
        (
            "api returned 403 Forbidden (permission_error): OAuth token does not meet scope requirement",
            LaneFailureClass::GatewayRouting,
        ),
        (
            "api failed after 6 attempts: api returned 429 Too Many Requests (rate_limit_error)",
            LaneFailureClass::GatewayRouting,
        ),
        (
            "tool failed: denied tool execution from hook",
            LaneFailureClass::ToolRuntime,
        ),
        (
            "conversation loop exceeded the maximum number of iterations",
            LaneFailureClass::ToolRuntime,
        ),
        (
            "agent exceeded its time budget",
            LaneFailureClass::ToolRuntime,
        ),
        ("thread creation failed", LaneFailureClass::Infra),
    ];

    for (message, expected) in cases {
        assert_eq!(classify_lane_failure(message), expected, "{message}");
    }
}

#[test]
fn lane_event_schema_serializes_to_canonical_names() {
    let cases = [
        (LaneEventName::Started, "lane.started"),
        (LaneEventName::Ready, "lane.ready"),
        (LaneEventName::PromptMisdelivery, "lane.prompt_misdelivery"),
        (LaneEventName::Blocked, "lane.blocked"),
        (LaneEventName::Red, "lane.red"),
        (LaneEventName::Green, "lane.green"),
        (LaneEventName::CommitCreated, "lane.commit.created"),
        (LaneEventName::PrOpened, "lane.pr.opened"),
        (LaneEventName::MergeReady, "lane.merge.ready"),
        (LaneEventName::Finished, "lane.finished"),
        (LaneEventName::Failed, "lane.failed"),
        (
            LaneEventName::BranchStaleAgainstMain,
            "branch.stale_against_main",
        ),
    ];

    for (event, expected) in cases {
        assert_eq!(
            serde_json::to_value(event).expect("serialize lane event"),
            json!(expected)
        );
    }
}

#[test]
fn agent_tool_subset_mapping_is_expected() {
    let general = allowed_tools_for_subagent("general-purpose");
    assert!(general.contains("bash"));
    assert!(general.contains("write_file"));
    assert!(!general.contains("Agent"));

    let explore = allowed_tools_for_subagent("Explore");
    assert!(explore.contains("read_file"));
    assert!(explore.contains("grep_search"));
    assert!(!explore.contains("bash"));

    let plan = allowed_tools_for_subagent("Plan");
    assert!(plan.contains("TodoWrite"));
    assert!(plan.contains("StructuredOutput"));
    assert!(!plan.contains("Agent"));

    let verification = allowed_tools_for_subagent("Verification");
    assert!(verification.contains("bash"));
    assert!(verification.contains("PowerShell"));
    assert!(!verification.contains("write_file"));
}

#[derive(Debug)]
struct MockSubagentApiClient {
    calls: usize,
    input_path: String,
}

impl runtime::ApiClient for MockSubagentApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        self.calls += 1;
        match self.calls {
            1 => {
                assert_eq!(request.messages.len(), 1);
                Ok(vec![
                    AssistantEvent::ToolUse {
                        id: "tool-1".to_string(),
                        name: "read_file".to_string(),
                        input: json!({ "path": self.input_path }).to_string(),
                    },
                    AssistantEvent::MessageStop,
                ])
            }
            2 => {
                assert!(request.messages.len() >= 3);
                Ok(vec![
                    AssistantEvent::TextDelta("Scope: completed mock review".to_string()),
                    AssistantEvent::MessageStop,
                ])
            }
            _ => unreachable!("extra mock stream call"),
        }
    }
}

#[test]
fn subagent_runtime_executes_tool_loop_with_isolated_session() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = temp_path("subagent-input.txt");
    std::fs::write(&path, "hello from child").expect("write input file");

    let mut runtime = ConversationRuntime::new(
        Session::new(),
        MockSubagentApiClient {
            calls: 0,
            input_path: path.display().to_string(),
        },
        SubagentToolExecutor::new(BTreeSet::from([String::from("read_file")])),
        agent_permission_policy(runtime::PermissionMode::DangerFullAccess, None),
        vec![String::from("system prompt")],
    );

    let summary = runtime
        .run_turn("Inspect the delegated file", None)
        .expect("subagent loop should succeed");

    assert_eq!(
        final_assistant_text(&summary),
        "Scope: completed mock review"
    );
    assert!(runtime
        .session()
        .messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .any(|block| matches!(
            block,
            runtime::ContentBlock::ToolResult { output, .. }
                if output.contains("hello from child")
        )));

    let _ = std::fs::remove_file(path);
}

/// End-to-end check of a per-agent permission override through the *same*
/// policy+enforcer the spawned `SubagentToolExecutor` uses (`build_agent_runtime`
/// → `agent_permission_policy` → `PermissionEnforcer::check`). Mirrors the config
/// a `reviewer.md` with `permissionMode: read-only` +
/// `permission: bash(git *)=allow, bash(rm *)=deny` parses into.
#[test]
fn per_agent_permission_override_enforced_through_executor_policy() {
    let rules = RuntimePermissionRuleConfig::new(
        vec!["bash(git *)".to_string()],
        vec!["bash(rm *)".to_string()],
        Vec::new(),
    );
    let policy = agent_permission_policy(PermissionMode::ReadOnly, Some(&rules));
    let enforcer = PermissionEnforcer::new(policy);

    // Denied by mode: read-only cannot satisfy the write tools' requirement.
    assert!(!enforcer.is_allowed("edit_file", r#"{"file_path":"src/main.rs"}"#));
    assert!(!enforcer.is_allowed("write_file", r#"{"file_path":"src/new.rs"}"#));
    // Denied by deny rule, even though bash would otherwise need escalation.
    assert!(!enforcer.is_allowed("bash", r#"{"command":"rm -rf x"}"#));

    // Allowed by mode: read_file's requirement is met by read-only.
    assert!(enforcer.is_allowed("read_file", r#"{"file_path":"src/main.rs"}"#));
    // Allowed by allow rule, which short-circuits the read-only mode gate.
    assert!(enforcer.is_allowed("bash", r#"{"command":"git status"}"#));
    assert!(enforcer.is_allowed("bash", r#"{"command":"git diff"}"#));
}

/// A built-in sub-agent (no custom definition → both override fields `None`)
/// must keep the historical policy: `DangerFullAccess`, zero rules.
#[test]
fn builtin_agent_policy_is_byte_identical_default() {
    let with_defaults = agent_permission_policy(PermissionMode::DangerFullAccess, None);
    assert_eq!(
        with_defaults.active_mode(),
        PermissionMode::DangerFullAccess
    );
    let enforcer = PermissionEnforcer::new(with_defaults);
    // DangerFullAccess satisfies every tool requirement → all allowed.
    assert!(enforcer.is_allowed("edit_file", r#"{"file_path":"src/main.rs"}"#));
    assert!(enforcer.is_allowed("bash", r#"{"command":"rm -rf x"}"#));
}

#[test]
fn agent_rejects_blank_required_fields() {
    let missing_description = run_tool(
        "Agent",
        &json!({
            "description": "  ",
            "prompt": "Inspect"
        }),
    )
    .expect_err("blank description should fail");
    assert!(missing_description
        .to_string()
        .contains("description must not be empty"));

    let missing_prompt = run_tool(
        "Agent",
        &json!({
            "description": "Inspect branch",
            "prompt": " "
        }),
    )
    .expect_err("blank prompt should fail");
    assert!(missing_prompt
        .to_string()
        .contains("prompt must not be empty"));
}
