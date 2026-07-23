//! Tool specs, alias routing, gateway metadata, and registry surface.
//!
//! Split out of the old flat `tests.rs` (4,133 lines) by domain;
//! shared fixtures live in the parent module.

use super::*;

const TEST_MODEL: &str = "claude-sonnet-4-6";

#[test]
fn typed_action_tools_are_registered_and_gated() {
    // Phase-2 typed actions are advertised with the right permission tier...
    let cargo = mvp_tool_specs()
        .iter()
        .find(|s| s.name == "Cargo")
        .expect("Cargo tool registered");
    assert_eq!(cargo.required_permission, PermissionMode::WorkspaceWrite);
    let git = mvp_tool_specs()
        .iter()
        .find(|s| s.name == "Git")
        .expect("Git tool registered");
    assert_eq!(git.required_permission, PermissionMode::ReadOnly);

    // ...and the gating is real end-to-end: `Cargo` (workspace-write) is denied
    // before dispatch under read-only, and the gateway records the denial under
    // the Cargo family — so the typed action is never silently more permissive
    // than the equivalent `bash "cargo build"`.
    let policy = permission_policy_for_mode(PermissionMode::ReadOnly);
    let registry = GlobalToolRegistry::builtin().with_enforcer(PermissionEnforcer::new(policy));
    let error = registry
        .execute("Cargo", &json!({ "action": "build" }))
        .expect_err("cargo build denied under read-only");
    assert!(error.to_string().contains("permission denied"));
    let invocations = registry.context().tool_invocations();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].request.family, ToolFamily::Cargo);
    assert!(matches!(
        invocations[0].policy_decision,
        ToolPolicyDecision::Denied { .. }
    ));
}

#[test]
fn exposes_mvp_tools() {
    let names = mvp_tool_specs()
        .iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"bash"));
    assert!(names.contains(&"read_file"));
    assert!(names.contains(&"WebFetch"));
    assert!(names.contains(&"WebSearch"));
    assert!(names.contains(&"TodoWrite"));
    assert!(names.contains(&"Skill"));
    assert!(names.contains(&"SkillDistill"));
    assert!(names.contains(&"SkillReview"));
    assert!(names.contains(&"Agent"));
    assert!(names.contains(&"ToolSearch"));
    assert!(names.contains(&"NotebookEdit"));
    assert!(names.contains(&"Sleep"));
    assert!(names.contains(&"SendUserMessage"));
    assert!(names.contains(&"Brief"));
    assert!(names.contains(&"SyntheticOutput"));
    assert!(names.contains(&"SpawnMultiAgent"));
    assert!(names.contains(&"Council"));
    // Unimplemented orchestration tools are intentionally not exposed to the
    // model: exposing them would advertise behavior the build cannot honor.
    assert!(!names.contains(&"ForkSubagent"));
    assert!(!names.contains(&"LoadAgentsDir"));
    assert!(!names.contains(&"VerificationAgent"));
    assert!(names.contains(&"EnterWorktree"));
    assert!(names.contains(&"ExitWorktree"));
    assert!(names.contains(&"ExitPlanModeV2"));
    assert!(names.contains(&"Workflow"));
    assert!(names.contains(&"WorkflowValidate"));
    assert!(names.contains(&"WorkflowLibrary"));
    assert!(names.contains(&"WorkflowRuns"));
    assert!(names.contains(&"WorkflowSkillProject"));
    assert!(names.contains(&"Config"));
    assert!(names.contains(&"EnterPlanMode"));
    assert!(names.contains(&"ExitPlanMode"));
    assert!(names.contains(&"StructuredOutput"));
    assert!(names.contains(&"CronRunDue"));
    assert!(names.contains(&"REPL"));
    assert!(names.contains(&"PowerShell"));
    assert!(names.contains(&"WorkerCreate"));
    assert!(names.contains(&"WorkerObserve"));
    assert!(names.contains(&"WorkerAwaitReady"));
    assert!(names.contains(&"WorkerSendPrompt"));
    assert!(names.contains(&"Monitor"));
    assert!(names.contains(&"SendMessage"));
    assert!(names.contains(&"ScheduleWakeup"));
    assert!(!names.contains(&"ListMcpResources"));
    assert!(!names.contains(&"ReadMcpResource"));
    assert!(!names.contains(&"McpAuth"));
    assert!(!names.contains(&"MCP"));
}

#[test]
fn legacy_mcp_tools_are_not_model_facing_builtin_tools() {
    let registry = GlobalToolRegistry::builtin();
    let legacy_names = ["ListMcpResources", "ReadMcpResource", "McpAuth", "MCP"];
    let advertised = registry
        .definitions(TEST_MODEL, None)
        .into_iter()
        .map(|definition| definition.name)
        .collect::<BTreeSet<_>>();

    for name in legacy_names {
        assert!(
            !advertised.contains(name),
            "legacy MCP tool `{name}` must not be advertised; live MCP uses runtime tools"
        );
        assert!(
            registry.execute(name, &json!({})).is_err(),
            "legacy MCP tool `{name}` must not execute through the model-facing registry"
        );
        assert!(
            registry
                .normalize_allowed_tools(&[name.to_string()])
                .is_err(),
            "legacy MCP tool `{name}` must not be allow-listable without a runtime MCP wrapper"
        );
    }
}

#[test]
fn rejects_unknown_tool_names() {
    let error = run_tool("nope", &json!({})).expect_err("tool should be rejected");
    assert!(error.to_string().contains("unsupported tool"));
}

// ------------------------------------------------------------------
// Lane A — tool name canonicalization
// ------------------------------------------------------------------

#[test]
fn canonical_tool_name_maps_pascal_and_short_aliases_to_handlers() {
    use crate::canonical_tool_name;

    // File tools: PascalCase, short, and snake_case all collapse to the
    // single handler name used by `file_tools::dispatch`.
    assert_eq!(canonical_tool_name("Read"), "read_file");
    assert_eq!(canonical_tool_name("read"), "read_file");
    assert_eq!(canonical_tool_name("read_file"), "read_file");
    assert_eq!(canonical_tool_name("Write"), "write_file");
    assert_eq!(canonical_tool_name("write"), "write_file");
    assert_eq!(canonical_tool_name("write_file"), "write_file");
    assert_eq!(canonical_tool_name("Edit"), "edit_file");
    assert_eq!(canonical_tool_name("edit"), "edit_file");
    assert_eq!(canonical_tool_name("Glob"), "glob_search");
    assert_eq!(canonical_tool_name("glob"), "glob_search");
    assert_eq!(canonical_tool_name("Grep"), "grep_search");
    assert_eq!(canonical_tool_name("grep"), "grep_search");

    // Bash is canonical as "bash"; PascalCase should round-trip.
    assert_eq!(canonical_tool_name("bash"), "bash");
    assert_eq!(canonical_tool_name("Bash"), "bash");

    // Already-PascalCase handlers stay put.
    assert_eq!(canonical_tool_name("TodoWrite"), "TodoWrite");
    assert_eq!(canonical_tool_name("WebFetch"), "WebFetch");
    assert_eq!(canonical_tool_name("NotebookEdit"), "NotebookEdit");

    // Snake_case mirror aliases resolve to their PascalCase handlers.
    assert_eq!(canonical_tool_name("todo_write"), "TodoWrite");
    assert_eq!(canonical_tool_name("web_fetch"), "WebFetch");
    assert_eq!(canonical_tool_name("notebook_edit"), "NotebookEdit");
}

#[test]
fn tool_registry_dispatch_accepts_read_aliases_for_same_handler() {
    use std::fs;

    let tmp = std::env::temp_dir().join(format!("lane-a-read-alias-{}.txt", std::process::id()));
    fs::write(&tmp, "hello world\n").expect("seed temp file");
    let path = tmp.to_string_lossy().to_string();

    let registry = GlobalToolRegistry::builtin();
    let input = json!({ "path": path });

    let via_pascal = registry
        .execute("Read", &input)
        .expect("PascalCase Read must dispatch");
    let via_short = registry
        .execute("read", &input)
        .expect("short-form `read` must dispatch");
    let via_snake = registry
        .execute("read_file", &input)
        .expect("snake_case read_file must dispatch");

    assert_eq!(
        via_pascal, via_snake,
        "Read and read_file should produce identical output"
    );
    assert_eq!(
        via_short, via_snake,
        "`read` and read_file should produce identical output"
    );
    assert!(via_snake.contains("hello world"));

    let _ = fs::remove_file(&tmp);
}

#[test]
fn tool_registry_dispatch_accepts_write_and_bash_aliases() {
    use std::fs;

    // This case runs a real `bash` command, which resolves `env::current_dir()`;
    // serialize it against the cwd-mutating tests (which all hold `env_lock`) so
    // a concurrent `set_current_dir` cannot invalidate the cwd mid-spawn.
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let tmp = std::env::temp_dir().join(format!("lane-a-write-alias-{}.txt", std::process::id()));
    let path = tmp.to_string_lossy().to_string();

    let registry = GlobalToolRegistry::builtin();

    // PascalCase Write should hit the same handler as write_file.
    let _ = registry
        .execute(
            "Write",
            &json!({ "path": path.clone(), "content": "from-pascal\n" }),
        )
        .expect("PascalCase Write must dispatch");
    assert_eq!(
        fs::read_to_string(&tmp).expect("file written"),
        "from-pascal\n"
    );

    let _ = registry
        .execute(
            "write_file",
            &json!({ "path": path.clone(), "content": "from-snake\n" }),
        )
        .expect("snake_case write_file must dispatch");
    assert_eq!(
        fs::read_to_string(&tmp).expect("file rewritten"),
        "from-snake\n"
    );

    let _ = fs::remove_file(&tmp);

    // Bash vs bash: both must land on the single bash handler.
    let cwd = sandbox_disabled_cwd("bash-alias-cwd");
    let mut ctx = ToolContext::new();
    ctx.cwd = Some(cwd.clone());
    let registry = GlobalToolRegistry::builtin().with_context(ctx);
    let bash_input = json!({ "command": "echo lane-a-bash" });
    let via_pascal = registry
        .execute("Bash", &bash_input)
        .expect("PascalCase Bash must dispatch");
    let via_snake = registry
        .execute("bash", &bash_input)
        .expect("snake_case bash must dispatch");
    assert!(via_pascal.contains("lane-a-bash"));
    assert!(via_snake.contains("lane-a-bash"));
    let _ = fs::remove_dir_all(cwd);
}

#[test]
fn resolve_allowed_tools_accepts_pascal_case_alias_inputs() {
    let registry = GlobalToolRegistry::builtin();
    let allowed = registry
        .normalize_allowed_tools(&[
            "Read".to_string(),
            "Write".to_string(),
            "Bash".to_string(),
            "Grep".to_string(),
        ])
        .expect("PascalCase aliases must be allow-listable")
        .expect("allow-list should be populated");
    assert!(allowed.contains("read_file"));
    assert!(allowed.contains("write_file"));
    assert!(allowed.contains("bash"));
    assert!(allowed.contains("grep_search"));
}

#[test]
fn global_tool_registry_denies_blocked_tool_before_dispatch() {
    // given
    let policy = permission_policy_for_mode(PermissionMode::ReadOnly);
    let registry = GlobalToolRegistry::builtin().with_enforcer(PermissionEnforcer::new(policy));

    // when
    let error = registry
        .execute(
            "write_file",
            &json!({
                "path": "blocked.txt",
                "content": "blocked"
            }),
        )
        .expect_err("write tool should be denied before dispatch");

    // then
    assert!(error
        .to_string()
        .contains("requires workspace-write permission"));
}

#[test]
fn tool_gateway_records_builtin_success_metadata() {
    let tmp = temp_path("gateway-read-success.txt");
    fs::write(&tmp, "gateway ok\n").expect("seed temp file");
    let ctx = ToolContext::new();

    let output = execute_tool(&ctx, "Read", &json!({ "path": tmp.display().to_string() }))
        .expect("read_file should succeed");

    assert!(output.contains("gateway ok"));
    let invocations = ctx.tool_invocations();
    assert_eq!(invocations.len(), 1);
    let invocation = &invocations[0];
    assert_eq!(invocation.request.requested_name, "Read");
    assert_eq!(invocation.request.tool_name, "read_file");
    assert_eq!(invocation.request.family, ToolFamily::File);
    assert!(matches!(
        invocation.policy_decision,
        ToolPolicyDecision::NotConfigured
    ));
    match &invocation.result {
        ToolInvocationResult::Succeeded { metadata } => {
            assert!(metadata.output_chars >= output.chars().count());
            assert_eq!(metadata.returned_chars, output.chars().count());
            assert!(!metadata.truncated);
        }
        other @ ToolInvocationResult::Failed { .. } => {
            panic!("expected successful invocation, got {other:?}")
        }
    }

    let _ = fs::remove_file(tmp);
}

#[test]
fn tool_gateway_records_permission_denials() {
    let policy = permission_policy_for_mode(PermissionMode::ReadOnly);
    let registry = GlobalToolRegistry::builtin().with_enforcer(PermissionEnforcer::new(policy));

    let error = registry
        .execute(
            "write_file",
            &json!({
                "path": "blocked.txt",
                "content": "blocked"
            }),
        )
        .expect_err("write tool should be denied");

    assert!(error.to_string().contains("permission denied"));
    let invocations = registry.context().tool_invocations();
    assert_eq!(invocations.len(), 1);
    let invocation = &invocations[0];
    assert_eq!(invocation.request.tool_name, "write_file");
    assert_eq!(invocation.request.family, ToolFamily::File);
    match &invocation.policy_decision {
        ToolPolicyDecision::Denied {
            check,
            active_mode,
            required_mode,
            reason,
        } => {
            assert_eq!(*check, ToolPolicyCheck::ToolPermission);
            assert_eq!(active_mode, "read-only");
            assert_eq!(required_mode, "workspace-write");
            assert!(reason.contains("requires workspace-write permission"));
        }
        other => panic!("expected denied policy decision, got {other:?}"),
    }
    assert!(matches!(
        &invocation.result,
        ToolInvocationResult::Failed { .. }
    ));
}

#[test]
fn tool_gateway_records_tool_toggle_denials() {
    let registry = GlobalToolRegistry::builtin()
        .with_disabled_tools(BTreeSet::from(["WebSearch".to_string()]));

    let error = registry
        .execute("WebSearch", &json!({ "query": "zo" }))
        .expect_err("disabled tool should be denied");

    assert!(error.to_string().contains("tool disabled"));
    let invocations = registry.context().tool_invocations();
    assert_eq!(invocations.len(), 1);
    let invocation = &invocations[0];
    assert_eq!(invocation.request.tool_name, "WebSearch");
    assert_eq!(invocation.request.family, ToolFamily::Web);
    assert!(matches!(
        invocation.policy_decision,
        ToolPolicyDecision::Denied {
            check: ToolPolicyCheck::ToolToggle,
            ..
        }
    ));
}

#[test]
fn tool_gateway_records_bash_web_and_workflow_families() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let cwd = sandbox_disabled_cwd("gateway-families-cwd");
    let bash_ctx = ToolContext::new().with_cwd(cwd.clone());
    execute_tool(
        &bash_ctx,
        "bash",
        &json!({ "command": "printf gateway-bash" }),
    )
    .expect("bash should succeed");
    let bash_invocations = bash_ctx.tool_invocations();
    assert_eq!(bash_invocations.len(), 1);
    assert_eq!(bash_invocations[0].request.family, ToolFamily::Bash);
    assert!(matches!(
        bash_invocations[0].result,
        ToolInvocationResult::Succeeded { .. }
    ));
    let _ = fs::remove_dir_all(cwd);

    let web_ctx = ToolContext::new();
    let web_error = execute_tool(
        &web_ctx,
        "WebFetch",
        &json!({ "url": "not a url", "prompt": "summarize" }),
    )
    .expect_err("invalid URL should fail before network IO");
    assert!(web_error.to_string().contains("invalid input"));
    let web_invocations = web_ctx.tool_invocations();
    assert_eq!(web_invocations.len(), 1);
    assert_eq!(web_invocations[0].request.family, ToolFamily::Web);
    assert!(matches!(
        web_invocations[0].result,
        ToolInvocationResult::Failed { .. }
    ));

    let workflow_ctx = ToolContext::new();
    let workflow_error = execute_tool(
        &workflow_ctx,
        "Workflow",
        &json!({ "name": "gateway", "phases": [] }),
    )
    .expect_err("invalid workflow should fail validation");
    assert!(workflow_error.to_string().contains("at least one phase"));
    let workflow_invocations = workflow_ctx.tool_invocations();
    assert_eq!(workflow_invocations.len(), 1);
    assert_eq!(workflow_invocations[0].request.family, ToolFamily::Workflow);
    assert!(matches!(
        workflow_invocations[0].result,
        ToolInvocationResult::Failed { .. }
    ));
}

#[test]
fn permission_mode_from_plugin_rejects_invalid_inputs() {
    let unknown_permission =
        permission_mode_from_plugin("admin").expect_err("unknown plugin permission should fail");
    assert!(unknown_permission
        .to_string()
        .contains("unsupported plugin permission: admin"));

    let empty_permission =
        permission_mode_from_plugin("").expect_err("empty plugin permission should fail");
    assert!(empty_permission
        .to_string()
        .contains("unsupported plugin permission: "));
}

#[test]
fn runtime_tools_extend_registry_definitions_permissions_and_search() {
    let registry = GlobalToolRegistry::builtin()
        .with_runtime_tools(vec![crate::RuntimeToolDefinition {
            name: "mcp__demo__echo".to_string(),
            description: Some("Echo text from the demo MCP server".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "additionalProperties": false
            }),
            required_permission: runtime::PermissionMode::ReadOnly,
        }])
        .expect("runtime tools should register");

    let allowed = registry
        .normalize_allowed_tools(&["mcp__demo__echo".to_string()])
        .expect("runtime tool should be allow-listable")
        .expect("allow-list should be populated");
    assert!(allowed.contains("mcp__demo__echo"));

    let definitions = registry.definitions(TEST_MODEL, Some(&allowed));
    assert_eq!(definitions.len(), 1);
    assert_eq!(definitions[0].name, "mcp__demo__echo");

    let permissions = registry
        .permission_specs(Some(&allowed))
        .expect("runtime tool permissions should resolve");
    assert_eq!(
        permissions,
        vec![(
            "mcp__demo__echo".to_string(),
            runtime::PermissionMode::ReadOnly
        )]
    );

    let search = registry.search(
        "demo echo",
        5,
        Some(vec!["pending-server".to_string()]),
        Some(runtime::McpDegradedReport::new(
            vec!["demo".to_string()],
            vec![runtime::McpFailedServer {
                server_name: "pending-server".to_string(),
                phase: runtime::McpLifecyclePhase::ToolDiscovery,
                error: runtime::McpErrorSurface::new(
                    runtime::McpLifecyclePhase::ToolDiscovery,
                    Some("pending-server".to_string()),
                    "tool discovery failed",
                    BTreeMap::new(),
                    true,
                ),
            }],
            vec!["mcp__demo__echo".to_string()],
            vec!["mcp__demo__echo".to_string()],
        )),
    );
    let output = serde_json::to_value(search).expect("search output should serialize");
    assert_eq!(output["matches"][0], "mcp__demo__echo");
    assert_eq!(output["pending_mcp_servers"][0], "pending-server");
    assert_eq!(
        output["mcp_degraded"]["failed_servers"][0]["phase"],
        "tool_discovery"
    );
}

#[test]
fn normalize_allowed_tools_accepts_aliases_and_mixed_separators() {
    let registry = GlobalToolRegistry::builtin();

    let allowed = registry
        .normalize_allowed_tools(&["read, write   grep".to_string()])
        .expect("aliases should normalize")
        .expect("allow-list should be populated");

    assert_eq!(
        allowed,
        BTreeSet::from([
            "grep_search".to_string(),
            "read_file".to_string(),
            "write_file".to_string(),
        ])
    );
}

#[test]
fn normalize_shell_command_collapses_whitespace_before_lowercasing() {
    assert_eq!(
        normalize_shell_command("  cargo   test   --workspace \n --all-targets "),
        "cargo test --workspace --all-targets"
    );
}

#[test]
fn builtin_registry_hides_lsp_from_public_surface_without_registered_servers() {
    let registry = GlobalToolRegistry::builtin();

    let definitions = registry.definitions(TEST_MODEL, None);
    assert!(!definitions
        .iter()
        .any(|definition| definition.name == "LSP"));

    let search = registry.search("lsp", 5, None, None);
    assert!(!search.matches.iter().any(|entry| entry == "LSP"));
}

#[test]
fn builtin_registry_exposes_lsp_when_server_is_registered() {
    let context = ToolContext::new();
    context.lsp.register(
        "rust",
        runtime::lsp_client::LspServerStatus::Connected,
        None,
        vec!["hover".into()],
    );
    let registry = GlobalToolRegistry::builtin().with_context(context);

    let definitions = registry.definitions(TEST_MODEL, None);
    assert!(definitions
        .iter()
        .any(|definition| definition.name == "LSP"));

    let search = registry.search("lsp", 5, None, None);
    assert!(search.matches.iter().any(|entry| entry == "LSP"));
}

#[test]
fn pending_tools_preserve_multiple_streaming_tool_calls_by_index() {
    let mut events = Vec::new();
    let mut pending_tools = BTreeMap::new();

    push_output_block(
        OutputContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "read_file".to_string(),
            input: json!({}),
        },
        1,
        &mut events,
        &mut pending_tools,
        true,
    );
    push_output_block(
        OutputContentBlock::ToolUse {
            id: "tool-2".to_string(),
            name: "grep_search".to_string(),
            input: json!({}),
        },
        2,
        &mut events,
        &mut pending_tools,
        true,
    );

    pending_tools
        .get_mut(&1)
        .expect("first tool pending")
        .2
        .push_str("{\"path\":\"src/main.rs\"}");
    pending_tools
        .get_mut(&2)
        .expect("second tool pending")
        .2
        .push_str("{\"pattern\":\"TODO\"}");

    assert_eq!(
        pending_tools.remove(&1),
        Some((
            "tool-1".to_string(),
            "read_file".to_string(),
            "{\"path\":\"src/main.rs\"}".to_string(),
        ))
    );
    assert_eq!(
        pending_tools.remove(&2),
        Some((
            "tool-2".to_string(),
            "grep_search".to_string(),
            "{\"pattern\":\"TODO\"}".to_string(),
        ))
    );
}

#[test]
fn tool_search_supports_keyword_and_select_queries() {
    let keyword = run_tool(
        "ToolSearch",
        &json!({"query": "web current", "max_results": 3}),
    )
    .expect("ToolSearch should succeed");
    let keyword_output: serde_json::Value = serde_json::from_str(&keyword).expect("valid json");
    let matches = keyword_output["matches"].as_array().expect("matches");
    assert!(matches.iter().any(|value| value == "WebSearch"));

    let selected = run_tool("ToolSearch", &json!({"query": "select:Agent,Skill"}))
        .expect("ToolSearch should succeed");
    let selected_output: serde_json::Value = serde_json::from_str(&selected).expect("valid json");
    assert_eq!(selected_output["matches"][0], "Agent");
    assert_eq!(selected_output["matches"][1], "Skill");

    let aliased = run_tool("ToolSearch", &json!({"query": "AgentTool"}))
        .expect("ToolSearch should support tool aliases");
    let aliased_output: serde_json::Value = serde_json::from_str(&aliased).expect("valid json");
    assert_eq!(aliased_output["matches"][0], "Agent");
    assert_eq!(aliased_output["normalized_query"], "agent");

    let selected_with_alias = run_tool("ToolSearch", &json!({"query": "select:AgentTool,Skill"}))
        .expect("ToolSearch alias select should succeed");
    let selected_with_alias_output: serde_json::Value =
        serde_json::from_str(&selected_with_alias).expect("valid json");
    assert_eq!(selected_with_alias_output["matches"][0], "Agent");
    assert_eq!(selected_with_alias_output["matches"][1], "Skill");
}

#[test]
fn tool_search_respects_disabled_tools_from_context() {
    let ctx = ToolContext::new();
    ctx.set_disabled_tools(BTreeSet::from(["WebSearch".to_string()]));

    let result = execute_tool(
        &ctx,
        "ToolSearch",
        &json!({"query": "select:WebSearch,read_file", "max_results": 5}),
    )
    .expect("ToolSearch should succeed");
    let output: serde_json::Value = serde_json::from_str(&result).expect("valid json");
    let matches = output["matches"].as_array().expect("matches");
    assert!(!matches.iter().any(|value| value == "WebSearch"));
    assert!(matches.iter().any(|value| value == "read_file"));
}

#[test]
fn task_creation_tools_are_not_exposed_and_specs_are_honest() {
    let specs = mvp_tool_specs();

    // Creation tools must not be advertised to the model: no runner drives
    // a created task past `Created`, so exposing them invites hang-over.
    let names: Vec<_> = specs.iter().map(|spec| spec.name).collect();
    assert!(!names.contains(&"TaskCreate"));
    assert!(!names.contains(&"RunTaskPacket"));

    // Read/manage tools stay exposed.
    assert!(names.contains(&"TaskGet"));
    assert!(names.contains(&"TaskList"));

    // Even though they are hidden, the spec descriptions must no longer
    // claim subprocess execution that the build cannot honor.
    let task_specs = crate::task_tools::tool_specs_for_test();
    for spec in &task_specs {
        if matches!(spec.name, "TaskCreate" | "RunTaskPacket") {
            assert!(
                !spec.description.to_lowercase().contains("subprocess"),
                "{} still claims subprocess execution",
                spec.name
            );
            assert!(
                spec.description.contains("not yet wired"),
                "{} should disclose that execution is not wired",
                spec.name
            );
        }
    }
}
