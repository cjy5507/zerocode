use std::collections::BTreeSet;

use runtime::{
    lsp_client::LspRegistry, permission_enforcer::PermissionEnforcer, ConversationRuntime,
    PermissionMode, PermissionPolicy, RuntimeFeatureConfig, Session, ToolError as RuntimeToolError,
    ToolExecutor,
};
use serde_json::json;

use super::super::{execute_tool_with_context, ToolContext};
use super::custom::CustomAgent;
use super::manifest::{record_current_tool, record_tool_finished};
use super::provider_client::ProviderRuntimeClient;
use super::{agent_permission_policy, allowed_tools_for_subagent, AgentJob, DEFAULT_AGENT_MODEL};
use crate::context::Probe;

pub(crate) struct SubagentToolExecutor {
    allowed_tools: BTreeSet<String>,
    enforcer: Option<PermissionEnforcer>,
    pub(super) context: ToolContext,
    /// Manifest path to stamp with the live `currentTool` on each call, so the
    /// parent sidebar shows what this agent is doing right now. `None` for
    /// non-persisted executors (tests).
    manifest_path: Option<std::path::PathBuf>,
    /// Parent-session MCP passthrough: calls to its advertised tools route
    /// back through the parent session's MCP runtime instead of the builtin
    /// dispatcher (which has no MCP families). `None` for non-MCP sessions.
    mcp_passthrough: Option<crate::registry::McpPassthrough>,
}

impl SubagentToolExecutor {
    pub(crate) fn new(allowed_tools: BTreeSet<String>) -> Self {
        Self {
            allowed_tools,
            enforcer: None,
            context: ToolContext::new(),
            manifest_path: None,
            mcp_passthrough: None,
        }
    }

    /// Route calls to the parent session's MCP tools through this passthrough
    /// (see the field docs). Builder-style, mirroring `with_manifest_path`.
    pub(crate) fn with_mcp_passthrough(
        mut self,
        passthrough: Option<crate::registry::McpPassthrough>,
    ) -> Self {
        let passthrough = passthrough.map(|passthrough| {
            passthrough.filtered_to_allowed(&self.allowed_tools)
        });
        if let Some(passthrough) = passthrough.as_ref() {
            self.context.install_mcp_passthrough(passthrough.clone());
        }
        self.mcp_passthrough = passthrough;
        self
    }

    pub(crate) fn with_enforcer(mut self, enforcer: PermissionEnforcer) -> Self {
        self.enforcer = Some(enforcer);
        self
    }

    pub(crate) fn with_manifest_path(mut self, path: std::path::PathBuf) -> Self {
        self.manifest_path = Some(path);
        self
    }

    /// Inherit the foreground TUI session that owns this sub-agent. Nested
    /// delegation tools stamp their manifests/progress from this context; leave
    /// it unset for legacy/headless agents whose parent manifest is unstamped.
    pub(crate) fn with_parent_session_id(self, session_id: Option<&str>) -> Self {
        if let Some(session_id) = session_id {
            self.context.set_session_id(session_id);
        }
        self
    }

    /// Inherit the resolved model this sub-agent is actually running on. Nested
    /// delegation tools read this through `spawn_parent_model()` so their model
    /// selection stays in the same provider/model context unless explicitly
    /// overridden by the nested tool input.
    pub(crate) fn with_parent_model(self, model: Option<&str>) -> Self {
        if let Some(model) = non_empty_model(model) {
            self.context.set_active_model(model);
        }
        self
    }

    /// Mark this agent's model as explicitly chosen (see
    /// [`crate::ToolContext::active_model_pinned`]) so its own nested spawns
    /// inherit it instead of smart-routing onto a different model.
    pub(crate) fn with_parent_model_pinned(self, pinned: bool) -> Self {
        self.context.set_active_model_pinned(pinned);
        self
    }

    /// Set this agent's write-lease owner id (track 4-2), so its `write_file` /
    /// `edit_file` calls take a per-path lease distinguishable from sibling
    /// agents. See [`ToolContext::with_lease_owner`].
    pub(crate) fn with_lease_owner(mut self, owner: String) -> Self {
        self.context = self.context.with_lease_owner(owner);
        self
    }

    /// Pin the working directory for this agent's tools *and* confine shell git
    /// operations to it (worktree isolation): `bash` runs there, relative file
    /// paths resolve against it, and `git -C` / `--git-dir` / `GIT_DIR=` cannot
    /// redirect into the shared checkout. See
    /// [`ToolContext::with_worktree_confinement`].
    pub(crate) fn with_worktree_confinement(mut self, cwd: std::path::PathBuf) -> Self {
        self.context = self.context.with_worktree_confinement(cwd);
        self
    }

    /// Share the parent session's LSP registry into this executor's context, so
    /// the edit/write diagnostics enrichment surfaces diagnostics to the
    /// sub-agent. Mirrors [`Self::with_cwd`]; see [`ToolContext::with_lsp`].
    pub(crate) fn with_lsp(mut self, lsp: LspRegistry) -> Self {
        self.context = self.context.with_lsp(lsp);
        self
    }

    /// Revert every instrumentation probe this agent staged via `InstrumentLog`
    /// (see [`ToolContext::revert_probes`]). Called at run completion - on both
    /// the success and error paths - so debug markers never survive into the
    /// working tree. Returns the number of probes reverted.
    pub(crate) fn revert_probes(&self) -> usize {
        self.context.revert_probes()
    }

    /// Release every write lease this agent holds (track 4-2), so its edited
    /// paths free up for the next sequential agent without waiting out the TTL.
    /// A no-op unless the context carries a lease owner (guard opt-in enabled);
    /// the lease store is resolved from the agent's cwd, falling back to the
    /// process cwd when unpinned — the same resolution the acquire path uses.
    pub(crate) fn release_write_leases(&self) {
        let Some(owner) = self.context.lease_owner.as_deref() else {
            return;
        };
        let cwd = self
            .context
            .cwd
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        crate::release_write_leases(owner, &cwd);
    }

    /// Clone of this executor's probe-sink `Arc`, for building a
    /// `ProbeRevertGuard` that reverts instrumentation even on a panic unwind
    /// (the `catch_unwind` path where an explicit `revert_probes()` is skipped).
    pub(crate) fn probe_sink_handle(&self) -> std::sync::Arc<std::sync::Mutex<Vec<Probe>>> {
        self.context.probe_sink.clone()
    }
}

impl ToolExecutor for SubagentToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, RuntimeToolError> {
        // Accept PascalCase aliases on the sub-agent boundary as well; the
        // allow-list stores canonical handler names, so resolve first.
        let canonical = crate::canonical_tool_name(tool_name);
        if !self.allowed_tools.contains(canonical.as_str()) {
            return Err(RuntimeToolError::new(format!(
                "tool `{tool_name}` is not enabled for this sub-agent"
            )));
        }
        let value = serde_json::from_str(input)
            .map_err(|error| RuntimeToolError::new(format!("invalid tool input JSON: {error}")))?;
        // Stamp only after input validation, then clear on both the success and
        // error return paths. `currentTool` is a live state, not a last-tool
        // label; leaving it set made completed calls look permanently active.
        if let Some(path) = self.manifest_path.as_ref() {
            record_current_tool(path, &canonical, input);
        }
        // Parent-session MCP tools are installed into this executor's
        // ToolContext. Let execute_tool_with_context route them so permission,
        // audit, truncation, and artifact handling stay identical to builtins.
        let result = execute_tool_with_context(
            &self.context,
            self.enforcer.as_ref(),
            canonical.as_str(),
            &value,
        );
        if let Some(path) = self.manifest_path.as_ref() {
            record_tool_finished(path);
        }
        result.map_err(|e| RuntimeToolError::new(e.to_string()))
    }

    fn take_pending_images(&mut self) -> Vec<(String, String)> {
        // Drain images a sub-agent tool (e.g. read_image) staged into this
        // executor's own context sink, so the sub-agent's model sees them too.
        std::mem::take(
            &mut *self
                .context
                .image_sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

pub(super) fn subagent_hook_context(
    job: &AgentJob,
    status: &str,
    result: Option<&str>,
    error: Option<&str>,
) -> serde_json::Value {
    json!({
        "status": status,
        "agent": {
            "id": &job.manifest.agent_id,
            "name": &job.manifest.name,
            "label": &job.manifest.label,
            "description": &job.manifest.description,
            "subagent_type": &job.manifest.subagent_type,
            "model": &job.manifest.model,
            "cwd": job.cwd.as_ref().map(|cwd| cwd.display().to_string()),
        },
        "result": result,
        "error": error,
    })
}

fn non_empty_model(model: Option<&str>) -> Option<&str> {
    model.filter(|model| !model.trim().is_empty())
}

fn subagent_parent_model(job: &AgentJob) -> Option<&str> {
    non_empty_model(job.manifest.model.as_deref())
        .or_else(|| non_empty_model(job.manifest.resolved_model.as_deref()))
}

/// This agent's model counts as an explicit pin for its own nested spawns when
/// it was NOT smart-routed (no `routeSource` on the manifest) and a model is
/// actually recorded. Covers both a pinned parent session (members inherit the
/// session model, routing skipped) and a member-level explicit `model` field —
/// in either case the chosen model cascades to nested delegation instead of
/// being re-routed one level deeper. Routed members (`routeSource` present)
/// keep dynamic routing for their children. Derived, not persisted: no new
/// manifest field or input plumbing.
fn subagent_model_pinned(job: &AgentJob) -> bool {
    job.manifest.route_source.is_none() && subagent_parent_model(job).is_some()
}

fn build_subagent_tool_executor(
    job: &AgentJob,
    allowed_tools: BTreeSet<String>,
    permission_policy: &PermissionPolicy,
) -> SubagentToolExecutor {
    let mut tool_executor = SubagentToolExecutor::new(allowed_tools)
        .with_enforcer(PermissionEnforcer::new(permission_policy.clone()))
        .with_mcp_passthrough(job.mcp_passthrough.clone())
        .with_manifest_path(std::path::PathBuf::from(&job.manifest.manifest_file))
        .with_parent_session_id(job.manifest.parent_session_id.as_deref())
        .with_parent_model(subagent_parent_model(job))
        .with_parent_model_pinned(subagent_model_pinned(job))
        .with_lease_owner(job.manifest.agent_id.clone());
    // Worktree isolation: confine this agent's bash/file tools to its own
    // directory, and refuse shell git redirects (`git -C`, `--git-dir`,
    // `GIT_DIR=`) that would reach the shared checkout. Absent a cwd the
    // executor keeps the process-cwd default (a non-isolated agent).
    if let Some(cwd) = job.cwd.clone() {
        tool_executor = tool_executor.with_worktree_confinement(cwd);
    }
    // Share the parent session's LSP servers so the existing edit/write
    // diagnostics enrichment surfaces diagnostics to e.g. a debugger sub-agent -
    // but only for a non-isolated agent (see `inherited_lsp` for why isolation
    // must skip it).
    if let Some(lsp) = inherited_lsp(job.cwd.as_deref(), job.lsp.clone()) {
        tool_executor = tool_executor.with_lsp(lsp);
    }
    tool_executor
}

pub(super) fn build_agent_runtime(
    job: &AgentJob,
    token_history: std::sync::Arc<std::sync::Mutex<Vec<u32>>>,
    output_tokens_total: std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> Result<ConversationRuntime<ProviderRuntimeClient, SubagentToolExecutor>, String> {
    let model = job
        .manifest
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_AGENT_MODEL.to_string());
    let mut allowed_tools = job.allowed_tools.clone();
    // Structured-output schema (8c): the agent must be able to call
    // `StructuredOutput` to return its result as a captured tool input.
    if job.schema.is_some() {
        allowed_tools.insert("StructuredOutput".to_string());
    }
    // Inherited MCP tools (parent-session passthrough): the schemas the client
    // advertises and, below, the permission requirements the policy registers.
    // Filtered by the allow-set, which an explicit custom `tools:` list may
    // have narrowed to a subset.
    let mcp_tools: Vec<crate::registry::RuntimeToolDefinition> = job
        .mcp_passthrough
        .as_ref()
        .map(|mcp| {
            mcp.definitions_snapshot()
                .into_iter()
                .filter(|definition| allowed_tools.contains(&definition.name))
                .collect()
        })
        .unwrap_or_default();
    let api_client = ProviderRuntimeClient::new_with_history(
        &model,
        allowed_tools.clone(),
        token_history,
        output_tokens_total,
        job.workflow_member,
        job.thinking_budget_tokens,
        job.route_effort,
        job.api_concurrency,
        job.route_fallback_models.clone(),
        Some(job.cancel_signal.clone()),
    )?
    .with_mcp_tools(
        mcp_tools
            .iter()
            .map(|definition| api::ToolDefinition {
                name: definition.name.clone(),
                description: definition.description.clone(),
                input_schema: definition.input_schema.clone(),
            })
            .collect(),
    )
    // Live activity: the provider client stamps wait-phases (governor queue,
    // rate-limit cool-down) and the streamed output tail onto the same
    // manifest the tool executor stamps `currentTool` on.
    .with_manifest_path(std::path::PathBuf::from(&job.manifest.manifest_file))
    // W9-3: identity for the one-shot starvation notice on the parent
    // transcript (label wins over the raw name, mirroring the HUD).
    .with_agent_identity(
        job.manifest.agent_id.clone(),
        job.manifest
            .label
            .clone()
            .unwrap_or_else(|| job.manifest.name.clone()),
    );
    let mut permission_policy = agent_permission_policy(
        job.permission_mode
            .unwrap_or(PermissionMode::DangerFullAccess),
        job.permission_rules.as_ref(),
    );
    // Register each inherited MCP tool's own permission requirement so the
    // sub-agent's enforcer grades it exactly like the main session does — an
    // unregistered name falls back to the most-restrictive default and would
    // deny MCP everywhere below danger-full-access.
    for definition in &mcp_tools {
        permission_policy = permission_policy
            .with_tool_requirement(definition.name.as_str(), definition.required_permission);
    }
    let tool_executor = build_subagent_tool_executor(job, allowed_tools, &permission_policy);
    let feature_config = RuntimeFeatureConfig::default()
        .with_hooks(job.hook_config.clone())
        .with_model(model.clone());
    let runtime = ConversationRuntime::new_with_features(
        agent_session(job)?,
        api_client,
        tool_executor,
        permission_policy,
        job.system_prompt.clone(),
        &feature_config,
    )
    // Share the spawn-time steering queue (already registered in the parent-
    // side registry) so a `SendMessage` to this agent lands in the queue the
    // turn loop actually drains.
    .with_steering_queue(job.steering.clone());
    // 8c: a schema phase forces a final `StructuredOutput` call so its result is
    // a captured tool input, not parsed-from-prose text.
    let runtime = if job.schema.is_some() {
        runtime.with_structured_output_tool("StructuredOutput")
    } else {
        runtime
    };
    Ok(runtime)
}

/// The [`Session`] a sub-agent's runtime starts from. A fresh spawn gets a new
/// session persisted live to the agent-store transcript; a `SendMessage`
/// resume rehydrates the terminal snapshot (context intact — the whole point),
/// and a missing/unreadable transcript is a hard error rather than a silent
/// fresh start that would masquerade as a continuation. Bare test harnesses
/// (`transcript_path: None`) keep the in-memory-only session.
fn agent_session(job: &AgentJob) -> Result<Session, String> {
    let Some(path) = job.transcript_path.as_ref() else {
        return Ok(Session::new());
    };
    if job.resume {
        return Session::load_from_secure_path(path).map_err(|error| {
            format!(
                "cannot rehydrate agent transcript `{}`: {error}",
                path.display()
            )
        });
    }
    Ok(Session::new().with_secure_persistence_path(path.clone()))
}

/// The LSP registry a sub-agent should inherit from its parent: shared only
/// when the agent runs in the parent cwd (`cwd` is `None`).
///
/// A worktree-isolated agent (`cwd` is `Some`) gets `None`: LSP file URIs are
/// formed against the process cwd (`LspStdioTransport::path_to_uri` /
/// `uri_to_path` use `std::env::current_dir`), while isolation scopes a sub-agent
/// only via `ctx.cwd` and never calls `set_current_dir`. So the parent's servers,
/// rooted in the parent tree, would sync/read the wrong tree for an isolated
/// agent. Keying on `cwd` (not on the isolation request) is also right for the
/// workflow worktree fallback, which degrades to `cwd: None` and then genuinely
/// runs in the parent tree.
pub(super) fn inherited_lsp(
    cwd: Option<&std::path::Path>,
    parent_lsp: Option<LspRegistry>,
) -> Option<LspRegistry> {
    if cwd.is_some() {
        return None;
    }
    parent_lsp
}

/// Tool allowlist for a resolved harness. A file-based custom agent uses its
/// declared `tools` when present, otherwise inherits the general-purpose
/// default set so a body-only definition still works. Built-in types fall
/// through to their static [`allowed_tools_for_subagent`] sets unchanged.
pub(super) fn allowed_tools_for_resolved(
    subagent_type: &str,
    custom: Option<&CustomAgent>,
) -> BTreeSet<String> {
    let Some(custom) = custom else {
        return allowed_tools_for_subagent(subagent_type);
    };
    match &custom.tools {
        Some(tools) if !tools.is_empty() => tools.iter().cloned().collect(),
        // Declared empty or omitted -> inherit the general-purpose default set.
        _ => allowed_tools_for_subagent("general-purpose"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::misc_tools::agent_tools::AgentActivityTelemetry;
    use crate::misc_tools::agent_tools::AgentOutput;
    use crate::registry::{McpPassthrough, RuntimeToolDefinition};

    fn test_passthrough() -> McpPassthrough {
        McpPassthrough::for_tests(
            vec![RuntimeToolDefinition {
                name: "mcp__ctx7__query".to_string(),
                description: Some("docs lookup".to_string()),
                input_schema: json!({"type": "object"}),
                required_permission: PermissionMode::Prompt,
            }],
            std::sync::Arc::new(|name, input| {
                Ok(format!(
                    "mcp:{name}:{}",
                    input.get("q").and_then(serde_json::Value::as_str).unwrap_or("")
                ))
            }),
        )
    }

    #[test]
    fn subagent_executes_inherited_mcp_tools_through_the_passthrough() {
        let mut allowed = BTreeSet::new();
        allowed.insert("mcp__ctx7__query".to_string());
        let mut executor =
            SubagentToolExecutor::new(allowed).with_mcp_passthrough(Some(test_passthrough()));

        let output = executor
            .execute("mcp__ctx7__query", r#"{"q": "hi"}"#)
            .expect("passthrough dispatch");
        assert_eq!(output, "mcp:mcp__ctx7__query:hi");
    }

    #[test]
    fn subagent_tool_search_finds_inherited_mcp_schemas() {
        let mut allowed = BTreeSet::new();
        allowed.insert("ToolSearch".to_string());
        allowed.insert("mcp__ctx7__query".to_string());
        let mut executor =
            SubagentToolExecutor::new(allowed).with_mcp_passthrough(Some(test_passthrough()));

        let output = executor
            .execute(
                "ToolSearch",
                r#"{"query":"select:mcp__ctx7__query","max_results":5}"#,
            )
            .expect("ToolSearch should see inherited MCP passthrough definitions");
        let value: serde_json::Value = serde_json::from_str(&output).expect("valid ToolSearch JSON");
        assert_eq!(value["matches"][0], "mcp__ctx7__query");
        assert!(
            value["schemas"]
                .as_array()
                .expect("schemas")
                .iter()
                .any(|schema| schema["name"] == "mcp__ctx7__query"),
            "ToolSearch should return the inherited MCP schema, not just the name"
        );
    }

    #[test]
    fn subagent_tool_search_hides_unallowed_inherited_mcp_schemas() {
        let passthrough = McpPassthrough::for_tests(
            vec![
                RuntimeToolDefinition {
                    name: "mcp__ctx7__query".to_string(),
                    description: Some("docs lookup".to_string()),
                    input_schema: json!({"type": "object"}),
                    required_permission: PermissionMode::Prompt,
                },
                RuntimeToolDefinition {
                    name: "mcp__secret__dump".to_string(),
                    description: Some("secret server".to_string()),
                    input_schema: json!({"type": "object"}),
                    required_permission: PermissionMode::DangerFullAccess,
                },
            ],
            std::sync::Arc::new(|name, _input| Ok(format!("mcp:{name}"))),
        );
        let mut allowed = BTreeSet::new();
        allowed.insert("ToolSearch".to_string());
        allowed.insert("mcp__ctx7__query".to_string());
        let mut executor = SubagentToolExecutor::new(allowed).with_mcp_passthrough(Some(passthrough));

        let output = executor
            .execute(
                "ToolSearch",
                r#"{"query":"select:mcp__secret__dump,mcp__ctx7__query","max_results":5}"#,
            )
            .expect("ToolSearch should run");
        let value: serde_json::Value = serde_json::from_str(&output).expect("valid ToolSearch JSON");
        let matches = value["matches"].as_array().expect("matches");
        assert!(matches.iter().any(|name| name == "mcp__ctx7__query"));
        assert!(!matches.iter().any(|name| name == "mcp__secret__dump"));
        assert!(
            value["schemas"]
                .as_array()
                .expect("schemas")
                .iter()
                .all(|schema| schema["name"] != "mcp__secret__dump"),
            "ToolSearch must not reveal MCP schemas outside the sub-agent allow-list"
        );
    }

    #[test]
    fn subagent_tool_search_respects_the_permission_gate() {
        let mut allowed = BTreeSet::new();
        allowed.insert("ToolSearch".to_string());
        allowed.insert("mcp__ctx7__query".to_string());
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
            .with_tool_requirement("ToolSearch", PermissionMode::WorkspaceWrite);
        let mut executor = SubagentToolExecutor::new(allowed)
            .with_enforcer(PermissionEnforcer::new(policy))
            .with_mcp_passthrough(Some(test_passthrough()));

        let error = executor
            .execute(
                "ToolSearch",
                r#"{"query":"select:mcp__ctx7__query","max_results":5}"#,
            )
            .expect_err("read-only sub-agent must not bypass ToolSearch permission checks");
        assert!(
            error.to_string().to_lowercase().contains("permission"),
            "unexpected error: {error}"
        );
        let invocations = executor.context.tool_invocations();
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].request.tool_name, "ToolSearch");
        assert!(matches!(
            invocations[0].policy_decision,
            crate::ToolPolicyDecision::Denied { .. }
        ));
    }

    #[test]
    fn subagent_mcp_passthrough_uses_shared_audit_and_truncation_path() {
        let passthrough = McpPassthrough::for_tests(
            vec![RuntimeToolDefinition {
                name: "mcp__ctx7__query".to_string(),
                description: Some("docs lookup".to_string()),
                input_schema: json!({"type": "object"}),
                required_permission: PermissionMode::Prompt,
            }],
            std::sync::Arc::new(|_name, _input| Ok("mcp-line detail detail detail
".repeat(2_000))),
        );
        let mut allowed = BTreeSet::new();
        allowed.insert("mcp__ctx7__query".to_string());
        let mut executor = SubagentToolExecutor::new(allowed).with_mcp_passthrough(Some(passthrough));

        let output = executor
            .execute("mcp__ctx7__query", r#"{"q":"hi"}"#)
            .expect("passthrough dispatch");
        assert!(
            output.contains("retrieve_tool_output"),
            "oversized MCP passthrough output should be truncated with a recovery notice"
        );
        assert!(
            output.len() < "mcp-line detail detail detail
".repeat(2_000).len(),
            "model-facing MCP passthrough output should be bounded"
        );

        let invocations = executor.context.tool_invocations();
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].request.tool_name, "mcp__ctx7__query");
        assert!(matches!(
            invocations[0].result,
            crate::ToolInvocationResult::Succeeded { .. }
        ));
        let crate::ToolInvocationResult::Succeeded { metadata } = &invocations[0].result else {
            panic!("expected successful MCP passthrough invocation");
        };
        assert!(
            metadata.artifact.is_some(),
            "oversized MCP passthrough output should preserve the full result as an artifact"
        );
    }

    #[test]
    fn subagent_mcp_call_respects_the_permission_gate() {
        let mut allowed = BTreeSet::new();
        allowed.insert("mcp__ctx7__query".to_string());
        // A read-only policy that knows the tool requires full access — the
        // exact grading `build_agent_runtime` registers from the definition.
        let policy = PermissionPolicy::new(PermissionMode::ReadOnly)
            .with_tool_requirement("mcp__ctx7__query", PermissionMode::DangerFullAccess);
        let mut executor = SubagentToolExecutor::new(allowed)
            .with_enforcer(PermissionEnforcer::new(policy))
            .with_mcp_passthrough(Some(test_passthrough()));

        let error = executor
            .execute("mcp__ctx7__query", r#"{"q": "hi"}"#)
            .expect_err("read-only sub-agent must not dispatch a full-access MCP tool");
        assert!(
            error.to_string().to_lowercase().contains("permission"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn subagent_without_passthrough_rejects_mcp_names() {
        let mut allowed = BTreeSet::new();
        allowed.insert("mcp__ctx7__query".to_string());
        let mut executor = SubagentToolExecutor::new(allowed);
        let error = executor
            .execute("mcp__ctx7__query", r#"{"q": "hi"}"#)
            .expect_err("no passthrough installed");
        assert!(error.to_string().contains("unsupported tool") || error.to_string().contains("not"));
    }

    fn manifest_with_parent_session(parent_session_id: Option<&str>) -> AgentOutput {
        AgentOutput {
            agent_id: "agent-1".to_string(),
            parent_session_id: parent_session_id.map(str::to_string),
            tool_call_id: None,
            name: "nested-runner".to_string(),
            label: Some("Nested runner".to_string()),
            description: "runs nested delegation".to_string(),
            subagent_type: Some("general-purpose".to_string()),
            requested_model: None,
            resolved_model: Some("gpt-5.5-fast".to_string()),
            route_reason: None,
            route_role: None,
            route_complexity: None,
            route_risk: None,
            route_source: None,
            model: Some("gpt-5.5-fast".to_string()),
            status: "running".to_string(),
            output_file: "/tmp/zo-agent-output.md".to_string(),
            manifest_file: "/tmp/zo-agent-manifest.json".to_string(),
            created_at: "1".to_string(),
            owner_pid: None,
            run_generation: 0,
            started_at: Some("1".to_string()),
            completed_at: None,
            completion_published_at: None,
            lane_events: Vec::new(),
            current_blocker: None,
            error: None,
            token_history: Vec::new(),
            current_tool: None,
            recent_tools: Vec::new(),
            tool_calls: 0,
            current_phase: None,
            output_tail: String::new(),
            last_activity_at: None,
            activity: AgentActivityTelemetry::default(),
        }
    }

    fn job_with_parent_session(parent_session_id: Option<&str>) -> AgentJob {
        AgentJob {
            manifest: manifest_with_parent_session(parent_session_id),
            prompt: "prompt".to_string(),
            system_prompt: Vec::new(),
            allowed_tools: BTreeSet::new(),
            permission_rules: None,
            permission_mode: None,
            cwd: None,
            lsp: None,
            schema: None,
            workflow_member: false,
            time_budget: None,
            thinking_budget_tokens: None,
            route_effort: None,
            api_concurrency: None,
            route_fallback_models: Vec::new(),
            mcp_passthrough: None,
            hook_config: runtime::RuntimeHookConfig::default(),
            cancel_signal: runtime::HookAbortSignal::new(),
            judged_agent: None,
            parent_model: None,
            steering: runtime::SteeringQueue::default(),
            transcript_path: None,
            resume: false,
        }
    }

    #[test]
    fn subagent_tool_executor_inherits_parent_session_id_for_nested_delegation() {
        let job = job_with_parent_session(Some("session-visible-in-tui"));
        let permission_policy = agent_permission_policy(PermissionMode::DangerFullAccess, None);

        let executor = build_subagent_tool_executor(&job, BTreeSet::new(), &permission_policy);

        assert_eq!(
            executor.context.session_id().as_deref(),
            Some("session-visible-in-tui"),
            "nested Agent/SpawnMultiAgent/Workflow calls must inherit the root TUI session stamp"
        );
    }

    #[test]
    fn subagent_tool_executor_keeps_unstamped_context_when_parent_session_is_absent() {
        let job = job_with_parent_session(None);
        let permission_policy = agent_permission_policy(PermissionMode::DangerFullAccess, None);

        let executor = build_subagent_tool_executor(&job, BTreeSet::new(), &permission_policy);

        assert_eq!(executor.context.session_id(), None);
    }

    #[test]
    fn subagent_tool_executor_inherits_parent_model_for_nested_delegation() {
        let job = job_with_parent_session(Some("session-visible-in-tui"));
        let permission_policy = agent_permission_policy(PermissionMode::DangerFullAccess, None);

        let executor = build_subagent_tool_executor(&job, BTreeSet::new(), &permission_policy);

        assert_eq!(
            executor.context.spawn_parent_model().as_deref(),
            Some("gpt-5.5-fast"),
            "nested Agent/SpawnMultiAgent calls must inherit the running sub-agent model"
        );
    }

    #[test]
    fn subagent_pin_cascades_to_nested_delegation_only_for_unrouted_models() {
        let permission_policy = agent_permission_policy(PermissionMode::DangerFullAccess, None);

        // Un-routed model (pinned-session inherit or an explicit member-level
        // `model`): nested spawns must see the pin and inherit it verbatim.
        let job = job_with_parent_session(Some("session"));
        let executor = build_subagent_tool_executor(&job, BTreeSet::new(), &permission_policy);
        assert!(
            executor.context.active_model_pinned(),
            "an un-routed agent model cascades as a pin to nested delegation"
        );

        // Smart-routed model (`routeSource` stamped): children keep dynamic routing.
        let mut routed = job_with_parent_session(Some("session"));
        routed.manifest.route_source = Some("auto".to_string());
        let executor = build_subagent_tool_executor(&routed, BTreeSet::new(), &permission_policy);
        assert!(
            !executor.context.active_model_pinned(),
            "a smart-routed agent model must not pin its nested delegation"
        );
    }

    #[test]
    fn subagent_parent_model_uses_resolved_model_fallback_and_ignores_blank_models() {
        let mut job = job_with_parent_session(Some("session-visible-in-tui"));
        job.manifest.model = Some("   ".to_string());
        job.manifest.resolved_model = Some("claude-opus-4-8".to_string());
        let permission_policy = agent_permission_policy(PermissionMode::DangerFullAccess, None);

        let executor = build_subagent_tool_executor(&job, BTreeSet::new(), &permission_policy);

        assert_eq!(
            executor.context.spawn_parent_model().as_deref(),
            Some("claude-opus-4-8")
        );
    }

    #[test]
    fn subagent_tool_executor_keeps_model_context_empty_when_manifest_has_no_model() {
        let mut job = job_with_parent_session(Some("session-visible-in-tui"));
        job.manifest.model = None;
        job.manifest.resolved_model = None;
        let permission_policy = agent_permission_policy(PermissionMode::DangerFullAccess, None);

        let executor = build_subagent_tool_executor(&job, BTreeSet::new(), &permission_policy);

        assert_eq!(executor.context.spawn_parent_model(), None);
    }
}
