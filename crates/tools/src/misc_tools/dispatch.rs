//! Name → runner dispatch for every misc tool.
//!
//! Each arm of the match deserialises the JSON input into the typed
//! struct, then forwards to the `run_*` wrapper that lives in
//! `super`. The wrappers in turn call the concrete `execute_*`
//! implementation and pretty-print the result.

use runtime::lsp_client::LspRegistry;
use runtime::permission_enforcer::PermissionEnforcer;
use serde_json::Value;

use super::agent_tools::resolve_subagent_type;
use super::smart_router::{
    apply_smart_models_to_spawn_input_with_auto_types,
    smart_parent_model_for_agent_with_auto_type, ROUTE_MODEL_SMUGGLE_KEY,
    ROUTE_REASON_SMUGGLE_KEY,
};
use super::{
    from_value, maybe_enforce_permission_check, run_agent, run_ask_user_question, run_audit,
    run_config, run_council, run_enter_plan_mode, run_exit_plan_mode, run_memory_write,
    run_monitor, run_notebook_edit, run_remote_trigger, run_retrieve_tool_output,
    run_schedule_wakeup, run_send_message, run_send_to_user, run_session_recall,
    run_skill, run_skill_distill, run_skill_review, run_sleep, run_spawn_multi_agent,
    run_structured_output, run_synthetic_output, run_testing_permission, run_tool_search,
    AgentInput, AskUserQuestionInput, ConfigInput, CouncilInput, EnterPlanModeInput,
    ExitPlanModeInput, MemoryWriteInput, MonitorInput, NotebookEditInput, RemoteTriggerInput,
    RetrieveToolOutputInput, ScheduleWakeupInput, SendMessageInput, SendToUserInput,
    SessionRecallInput,
    SkillDistillInput, SkillInput, SkillReviewInput, SleepInput, SpawnMultiAgentInput,
    StructuredOutputInput, SyntheticOutputInput, TestingPermissionInput, ToolContext, ToolError,
    ToolSearchInput,
};

/// The parent session's LSP registry to share with a spawned sub-agent, or
/// `None` when no server is running. Returning `None` for an empty registry
/// keeps the headless / no-LSP case a clean no-op: no empty-registry `Arc` is
/// cloned across the spawn boundary, and the sub-agent's enrich gate
/// (`!ctx.lsp.is_empty()`) stays skipped exactly as today.
fn parent_lsp(ctx: &ToolContext) -> Option<&LspRegistry> {
    (!ctx.lsp.is_empty()).then_some(&ctx.lsp)
}

/// Owning `tool_use` id the runtime dispatcher smuggles into Spawn-family
/// execution input (`spawn_tool_execution_input` in `runtime::conversation`),
/// stamped onto spawned agent manifests so the TUI attributes each agent to
/// the right transcript batch. `None` for direct/headless invocations.
fn smuggled_tool_call_id(input: &Value) -> Option<String> {
    input
        .get("__zo_tool_call_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// The spawning session's active permission mode: the registry enforcer's
/// live mode where one exists (sub-agent / headless / serve), else the
/// foreground session mode the TUI records on [`ToolContext`]. Spawn-family
/// dispatch stamps this so child agents are clamped to the parent's
/// privilege instead of the historical `DangerFullAccess` default.
fn active_parent_mode(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
) -> Option<runtime::PermissionMode> {
    enforcer
        .map(PermissionEnforcer::active_mode)
        .or_else(|| ctx.session_permission_mode())
}

fn stamp_inferred_agent_type(input: &mut AgentInput) -> Option<String> {
    if input.subagent_type.is_some() {
        return None;
    }
    let inferred = resolve_subagent_type(None, &input.description, &input.prompt);
    input.subagent_type = Some(inferred.clone());
    Some(inferred)
}

/// Whether an explicit `subagent_type` resolves to a user-defined CUSTOM agent
/// (`.zo/agents/*.md`) — a deliberate, user-authored specialization worth
/// isolating even on the same model. Built-in read-only types (Explore,
/// code-reviewer, …) need no special case: their read-only prompts already fail
/// the guard's write-intent clause. Generic aliases and invented/bogus names
/// carry no custom definition, so a model cannot dodge the guard by naming a
/// made-up `subagent_type`. Derived from the custom-agent SSOT — no hardcoded
/// type table.
fn spawn_uses_custom_agent(explicit: Option<&str>, description: &str, prompt: &str) -> bool {
    let Some(raw) = explicit.map(str::trim).filter(|kind| !kind.is_empty()) else {
        return false;
    };
    let (_resolved, custom) =
        super::agent_tools::resolve_subagent_type_and_custom_agent(Some(raw), description, prompt);
    custom.is_some()
}

/// CC-style implementation-delegation guard. A MODEL-DRIVEN single `Agent`
/// spawn is WASTEFUL — the main loop should implement it inline instead — when
/// every clause below holds. Each early `return false` is an escape that KEEPS
/// the spawn: unknown per-turn policy (fail open), a user turn that itself
/// warrants delegation, a user-defined custom agent, no parent to compare, a
/// genuinely different model, read-only delegated work, or a hard delegated
/// slice. Only a same-model, simple, generic, non-delegation-worthy
/// implementation folds inline. All signals come from the corpus-pinned SSOT
/// classifiers — no hardcoded tables.
fn same_model_impl_spawn_is_wasteful(
    resolved_model: &str,
    parent_model: Option<&str>,
    delegated: crate::AgentTaskAssessment,
    spawn_is_custom_agent: bool,
    policy: Option<crate::TurnAgentPolicy>,
) -> bool {
    // Unknown per-turn policy (background / non-turn dispatch) → fail open.
    let Some(policy) = policy else {
        return false;
    };
    // The user's turn must itself be a simple Solo ask with no delegation
    // value/request; otherwise the turn genuinely warrants sub-agents.
    if !policy.user_turn_is_solo_simple() {
        return false;
    }
    // A user-defined custom agent is a deliberate specialization — keep it.
    if spawn_is_custom_agent {
        return false;
    }
    // No parent model to compare against (non-live/test harness) → fail open.
    let Some(parent) = parent_model.map(str::trim).filter(|p| !p.is_empty()) else {
        return false;
    };
    // A genuinely different effective model is real diversity — keep it.
    if crate::misc_tools::canonicalize_route_model_id(resolved_model)
        != crate::misc_tools::canonicalize_route_model_id(parent)
    {
        return false;
    }
    // Read-only / research delegated work is a valid same-model sub-agent use.
    if !delegated.has_write_intent {
        return false;
    }
    // Only a SIMPLE delegated slice is wasteful to hand to a same-model clone.
    matches!(
        delegated.complexity,
        runtime::RouteTaskComplexity::Trivial | runtime::RouteTaskComplexity::Small
    )
}

/// The successful tool result returned when the guard folds a spawn: a clear
/// `folded_inline` status so the model implements in the main loop and does not
/// masquerade the (non-)spawn as completed work or retry it.
fn inline_fold_result(model: &str) -> String {
    format!(
        "status: folded_inline — no sub-agent was spawned. This is a simple, single \
         implementation task and you are already running `{model}`, so implement it inline in \
         the main loop now (Claude-Code style: the main agent implements; reserve sub-agents for \
         parallel fan-out, read-only search across many files, a genuinely complex/large task, or \
         a different model). Do NOT retry this spawn."
    )
}

fn single_spawn_member_inline_fold_model(
    input: &SpawnMultiAgentInput,
    parent_model: Option<&str>,
    policy: Option<crate::TurnAgentPolicy>,
) -> Option<String> {
    let [member] = input.agents.as_slice() else {
        return None;
    };
    let prompt = member
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let subagent_type = member.get("subagent_type").and_then(Value::as_str);
    let member_model = |key| {
        member
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(str::to_string)
    };
    let resolved_model = std::env::var(super::agent_tools::AGENT_MODEL_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| member_model(ROUTE_MODEL_SMUGGLE_KEY))
        .or_else(|| member_model("model"))
        .or_else(|| {
            parent_model
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_default();
    let delegated = super::assess_agent_task(prompt);
    let spawn_is_custom = spawn_uses_custom_agent(subagent_type, "", prompt);

    same_model_impl_spawn_is_wasteful(
        &resolved_model,
        parent_model,
        delegated,
        spawn_is_custom,
        policy,
    )
    .then_some(resolved_model)
}

fn stamp_inferred_spawn_types(input: &mut SpawnMultiAgentInput) -> Vec<Option<String>> {
    input
        .agents
        .iter_mut()
        .map(|agent| {
            let object = agent.as_object_mut()?;
            object.remove(ROUTE_REASON_SMUGGLE_KEY);
            if object.contains_key("subagent_type") || object.contains_key("subagentType") {
                return None;
            }
            let inferred = resolve_subagent_type(
                None,
                object
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                object
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            object.insert(
                "subagent_type".to_string(),
                Value::String(inferred.clone()),
            );
            Some(inferred)
        })
        .collect()
}

fn annotate_auto_type_reason(reason: Option<String>, auto_type: Option<&str>) -> Option<String> {
    let auto_type = auto_type?;
    Some(reason.map_or_else(
        || format!("type={auto_type} (auto)"),
        |reason| format!("{reason} · type={auto_type} (auto)"),
    ))
}

fn annotate_spawn_auto_type_reasons(
    input: &mut SpawnMultiAgentInput,
    auto_types: &[Option<String>],
) {
    for (agent, auto_type) in input.agents.iter_mut().zip(auto_types) {
        let Some(auto_type) = auto_type.as_deref() else {
            continue;
        };
        let Some(object) = agent.as_object_mut() else {
            continue;
        };
        let existing = object
            .remove(ROUTE_REASON_SMUGGLE_KEY)
            .and_then(|value| value.as_str().map(str::to_string));
        let reason = annotate_auto_type_reason(existing, Some(auto_type))
            .expect("an auto type always produces a route reason");
        object.insert(ROUTE_REASON_SMUGGLE_KEY.to_string(), Value::String(reason));
    }
}

#[allow(clippy::too_many_lines)] // a flat name → runner table, clearer unsplit
pub(crate) fn dispatch(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    match name {
        "Skill" => Some(from_value::<SkillInput>(input).and_then(run_skill)),
        "SkillDistill" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<SkillDistillInput>(input).and_then(|inp| run_skill_distill(&inp))
            }),
        ),
        "SkillReview" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<SkillReviewInput>(input).and_then(|inp| run_skill_review(&inp))
            }),
        ),
        "Agent" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<AgentInput>(input).and_then(|mut inp| {
                    // Capture the model's EXPLICIT agent-type choice before
                    // `stamp_inferred_agent_type` fills an inferred one, so the
                    // guard can tell a user-defined custom agent from a bare
                    // "go implement this" spawn.
                    let explicit_subagent_type = inp.subagent_type.clone();
                    let auto_type = stamp_inferred_agent_type(&mut inp);
                    let parent = ctx.spawn_parent_model();
                    // A user-pinned session model is inherited verbatim: smart
                    // routing only tunes spawns when the session itself rides
                    // defaults (member-level `model` fields still win inside).
                    let route = !ctx.active_model_pinned();
                    if let Some(choice) =
                        route
                            .then(|| {
                                smart_parent_model_for_agent_with_auto_type(
                                    parent.as_deref(),
                                    &inp,
                                    auto_type.as_deref(),
                                )
                            })
                            .flatten()
                    {
                        inp.route_model = choice.model;
                        inp.route_reason = choice.reason;
                        inp.route_fallback_models = choice.fallback_models;
                        inp.route_effort = choice.effort;
                        inp.route_role = Some(choice.decision_meta.role);
                        inp.route_complexity = Some(choice.decision_meta.complexity);
                        inp.route_risk = Some(choice.decision_meta.risk);
                        inp.route_source = Some(choice.decision_meta.route_source);
                    }
                    // CC-style guard: fold a wasteful same-model, simple
                    // implementation spawn back to the main loop. The resolved
                    // model is what the spawn will ACTUALLY run — the global
                    // env override wins over everything (mirrors
                    // `try_resolve_agent_model_selection` /
                    // `smart_routed_model_selection`), then route → explicit →
                    // inherited parent. Custom-agent spawns are exempt inside
                    // the guard, so this non-custom precedence is exact.
                    let resolved_model: String =
                        std::env::var(super::agent_tools::AGENT_MODEL_ENV)
                            .ok()
                            .map(|value| value.trim().to_string())
                            .filter(|value| !value.is_empty())
                            .or_else(|| {
                                inp.route_model
                                    .as_deref()
                                    .map(str::trim)
                                    .filter(|m| !m.is_empty())
                                    .map(str::to_string)
                            })
                            .or_else(|| {
                                inp.model
                                    .as_deref()
                                    .map(str::trim)
                                    .filter(|m| !m.is_empty())
                                    .map(str::to_string)
                            })
                            .or_else(|| {
                                parent
                                    .as_deref()
                                    .map(str::trim)
                                    .filter(|m| !m.is_empty())
                                    .map(str::to_string)
                            })
                            .unwrap_or_default();
                    let delegated = super::assess_agent_task(&inp.prompt);
                    let spawn_is_custom = spawn_uses_custom_agent(
                        explicit_subagent_type.as_deref(),
                        &inp.description,
                        &inp.prompt,
                    );
                    if same_model_impl_spawn_is_wasteful(
                        &resolved_model,
                        parent.as_deref(),
                        delegated,
                        spawn_is_custom,
                        ctx.turn_agent_policy(),
                    ) {
                        return Ok(inline_fold_result(&resolved_model));
                    }
                    inp.route_reason = annotate_auto_type_reason(
                        inp.route_reason.take(),
                        auto_type.as_deref(),
                    );
                    inp.parent_session_id = ctx.session_id();
                    inp.parent_permission_mode = active_parent_mode(ctx, enforcer);
                    inp.tool_call_id = smuggled_tool_call_id(input);
                    inp.mcp_passthrough = ctx.mcp_passthrough();
                    // An omitted `background` defers to the host: detached in
                    // the interactive main session (its REPL re-injects the
                    // completion), blocking anywhere the result would be lost.
                    if inp.background.is_none() {
                        inp.background = Some(ctx.background_agent_default());
                    }
                    run_agent(
                        inp,
                        parent.as_deref(),
                        parent_lsp(ctx),
                        Some(ctx.hook_config()),
                    )
                })
            }),
        ),
        "ToolSearch" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<ToolSearchInput>(input).and_then(|inp| run_tool_search(&inp, ctx))
            }),
        ),
        // Read-only ledger rollup — takes no input, so it ignores `input`.
        "Audit" => Some(run_audit(ctx)),
        "session_recall" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<SessionRecallInput>(input).and_then(|inp| run_session_recall(inp, ctx))
            }),
        ),
        "retrieve_tool_output" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<RetrieveToolOutputInput>(input)
                    .and_then(|inp| run_retrieve_tool_output(&inp))
            }),
        ),
        "NotebookEdit" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| from_value::<NotebookEditInput>(input).and_then(run_notebook_edit)),
        ),
        "Sleep" => Some(from_value::<SleepInput>(input).and_then(|input| run_sleep(&input))),
        // `send_to_user` is the mid-run push tool; `SendUserMessage`/`Brief`
        // are legacy aliases kept for models trained on those names — all three
        // route through one runner so the behavior can never drift.
        "send_to_user" | "SendUserMessage" | "Brief" => Some(
            from_value::<SendToUserInput>(input).and_then(|inp| run_send_to_user(inp, ctx)),
        ),
        "SyntheticOutput" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<SyntheticOutputInput>(input)
                    .and_then(|input| run_synthetic_output(&input))
            }),
        ),
        "SpawnMultiAgent" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<SpawnMultiAgentInput>(input).and_then(|mut inp| {
                    let auto_types = stamp_inferred_spawn_types(&mut inp);
                    let parent = ctx.spawn_parent_model();
                    // Same pinned-session rule as the `Agent` arm above.
                    if !ctx.active_model_pinned() {
                        apply_smart_models_to_spawn_input_with_auto_types(
                            parent.as_deref(),
                            &mut inp,
                            &auto_types,
                        );
                    }
                    annotate_spawn_auto_type_reasons(&mut inp, &auto_types);
                    if let Some(resolved_model) = single_spawn_member_inline_fold_model(
                        &inp,
                        parent.as_deref(),
                        ctx.turn_agent_policy(),
                    ) {
                        return Ok(inline_fold_result(&resolved_model));
                    }
                    inp.parent_session_id = ctx.session_id();
                    inp.parent_permission_mode = active_parent_mode(ctx, enforcer);
                    inp.tool_call_id = smuggled_tool_call_id(input);
                    inp.mcp_passthrough = ctx.mcp_passthrough();
                    run_spawn_multi_agent(
                        &inp,
                        parent.as_deref(),
                        parent_lsp(ctx),
                        Some(ctx.hook_config()),
                    )
                })
            }),
        ),
        "Council" => Some(from_value::<CouncilInput>(input).and_then(|inp| run_council(&inp))),
        "Config" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| from_value::<ConfigInput>(input).and_then(run_config)),
        ),
        "EnterPlanMode" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<EnterPlanModeInput>(input).and_then(run_enter_plan_mode)
            }),
        ),
        "ExitPlanMode" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .map_err(|error| match error {
                    // Models trained on the CC-style tool of the same name call
                    // this to submit a plan; in read-only plan mode that denial
                    // is a dead end without a pointer to the real submission
                    // tool, costing a wasted round-trip per turn.
                    ToolError::PermissionDenied { tool, reason } => {
                        ToolError::PermissionDenied {
                            tool,
                            reason: format!(
                                "{reason} · To submit a plan for approval while in plan mode, call ExitPlanModeV2 — plan mode is lifted only by the user."
                            ),
                        }
                    }
                    other => other,
                })
                .and_then(|()| from_value::<ExitPlanModeInput>(input).and_then(run_exit_plan_mode)),
        ),
        "StructuredOutput" => {
            Some(from_value::<StructuredOutputInput>(input).and_then(run_structured_output))
        }
        "AskUserQuestion" => Some(
            from_value::<AskUserQuestionInput>(input)
                .and_then(|inp| run_ask_user_question(inp, ctx.user_question_channel().as_deref())),
        ),
        "MemoryWrite" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<MemoryWriteInput>(input).and_then(|inp| run_memory_write(&inp, ctx))
            }),
        ),
        "RemoteTrigger" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<RemoteTriggerInput>(input).and_then(run_remote_trigger)
            }),
        ),
        "TestingPermission" => Some(
            from_value::<TestingPermissionInput>(input)
                .and_then(|inp| run_testing_permission(&inp, enforcer)),
        ),
        "Monitor" => Some(from_value::<MonitorInput>(input).and_then(run_monitor)),
        "SendMessage" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<SendMessageInput>(input).and_then(|inp| {
                    run_send_message(
                        &inp,
                        parent_lsp(ctx),
                        Some(ctx.hook_config()),
                        ctx.mcp_passthrough(),
                        active_parent_mode(ctx, enforcer),
                    )
                })
            }),
        ),
        "ScheduleWakeup" => Some(
            from_value::<ScheduleWakeupInput>(input).and_then(|input| {
                let session_id = ctx.session_id();
                run_schedule_wakeup(&input, session_id.as_deref())
            }),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::*;

    struct EnvGuard {
        key: &'static str,
        prior: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let prior = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prior }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prior.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn solo_simple_policy() -> crate::TurnAgentPolicy {
        crate::TurnAgentPolicy {
            user_complexity: runtime::RouteTaskComplexity::Small,
            user_shape: runtime::RouteShapeKind::Solo,
            user_need_count: 0,
            user_requested_delegation: false,
        }
    }

    fn write_simple_task() -> crate::AgentTaskAssessment {
        crate::AgentTaskAssessment {
            complexity: runtime::RouteTaskComplexity::Small,
            has_write_intent: true,
        }
    }

    /// The guard folds ONLY a same-model, simple, generic, non-delegation-worthy
    /// implementation spawn on a simple Solo turn; every genuine reason to spawn
    /// keeps it, and unknown per-turn policy fails open.
    #[test]
    fn same_model_impl_spawn_guard_folds_only_the_wasteful_case() {
        let sol = Some("gpt-5.6-sol");
        let policy = Some(solo_simple_policy());
        let task = write_simple_task();

        // Baseline: same model, simple write slice, no custom, simple Solo turn.
        assert!(same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            task,
            false,
            policy,
        ));
        // Fail open — unknown per-turn policy (background / non-turn dispatch).
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            task,
            false,
            None,
        ));
        // Escape — the user turn itself warrants delegation (non-Solo shape).
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            task,
            false,
            Some(crate::TurnAgentPolicy {
                user_shape: runtime::RouteShapeKind::ParallelLanes,
                ..solo_simple_policy()
            }),
        ));
        // Escape — the user turn is hard (complex whole-turn).
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            task,
            false,
            Some(crate::TurnAgentPolicy {
                user_complexity: runtime::RouteTaskComplexity::Large,
                ..solo_simple_policy()
            }),
        ));
        // Escape — the turn has planned agent needs.
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            task,
            false,
            Some(crate::TurnAgentPolicy {
                user_need_count: 2,
                ..solo_simple_policy()
            }),
        ));
        // Escape — the user EXPLICITLY requested delegation (non-Solo requested
        // shape), even though the natural shape is Solo with no needs.
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            task,
            false,
            Some(crate::TurnAgentPolicy {
                user_requested_delegation: true,
                ..solo_simple_policy()
            }),
        ));
        // Escape — a user-defined custom agent.
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol", sol, task, true, policy,
        ));
        // Escape — a genuinely different effective model.
        assert!(!same_model_impl_spawn_is_wasteful(
            "claude-opus-4-8",
            sol,
            task,
            false,
            policy,
        ));
        // Escape — read-only / research delegated work.
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            crate::AgentTaskAssessment {
                complexity: runtime::RouteTaskComplexity::Small,
                has_write_intent: false,
            },
            false,
            policy,
        ));
        // Escape — a genuinely hard delegated slice.
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol",
            sol,
            crate::AgentTaskAssessment {
                complexity: runtime::RouteTaskComplexity::Large,
                has_write_intent: true,
            },
            false,
            policy,
        ));
        // Escape — no parent model (non-live/test harness).
        assert!(!same_model_impl_spawn_is_wasteful(
            "gpt-5.6-sol", None, task, false, policy,
        ));
    }

    #[test]
    fn single_spawn_member_guard_folds_only_a_wasteful_solo_member() {
        let _lock = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _model = EnvGuard::set(super::super::agent_tools::AGENT_MODEL_ENV, "");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "zo-dispatch-single-spawn-custom-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("custom agent dir");
        std::fs::write(
            dir.join("myrefactorer.md"),
            "---\nname: myrefactorer\ndescription: Custom\ntools: edit_file\n---\nRefactor locally.",
        )
        .expect("custom agent definition");
        let _defs = EnvGuard::set("ZO_AGENT_DEFS_DIR", &dir);
        let spawn = |agents: Vec<Value>| -> SpawnMultiAgentInput {
            serde_json::from_value(json!({ "agents": agents })).expect("spawn input")
        };
        let write_prompt = "Write a Rust function for the endpoint";
        assert_eq!(
            super::super::assess_agent_task(write_prompt),
            write_simple_task()
        );

        let same_model = json!({
            "prompt": write_prompt,
            "subagent_type": "general-purpose",
            "model": "claude-opus-4-8",
            "__zo_route_model": "gpt-5.6-sol"
        });
        assert_eq!(
            single_spawn_member_inline_fold_model(
                &spawn(vec![same_model.clone()]),
                Some("gpt-5.6-sol"),
                Some(solo_simple_policy()),
            )
            .as_deref(),
            Some("gpt-5.6-sol"),
            "the routed model wins over the member model and folds"
        );
        assert_eq!(
            single_spawn_member_inline_fold_model(
                &spawn(vec![same_model.clone(), same_model]),
                Some("gpt-5.6-sol"),
                Some(solo_simple_policy()),
            ),
            None,
            "two members are a real swarm"
        );
        assert_eq!(
            single_spawn_member_inline_fold_model(
                &spawn(vec![json!({
                    "prompt": write_prompt,
                    "subagent_type": "general-purpose",
                    "__zo_route_model": "claude-opus-4-8"
                })]),
                Some("gpt-5.6-sol"),
                Some(solo_simple_policy()),
            ),
            None,
            "a different routed model is genuine diversity"
        );
        assert_eq!(
            single_spawn_member_inline_fold_model(
                &spawn(vec![json!({
                    "prompt": write_prompt,
                    "subagent_type": "myrefactorer",
                    "__zo_route_model": "gpt-5.6-sol"
                })]),
                Some("gpt-5.6-sol"),
                Some(solo_simple_policy()),
            ),
            None,
            "a custom agent remains isolated"
        );
        assert_eq!(
            single_spawn_member_inline_fold_model(
                &spawn(vec![json!({
                    "prompt": "Inspect dispatch.rs and report the relevant code",
                    "subagent_type": "Explore",
                    "__zo_route_model": "gpt-5.6-sol"
                })]),
                Some("gpt-5.6-sol"),
                Some(solo_simple_policy()),
            ),
            None,
            "read-only work remains delegated"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Only a user-defined CUSTOM agent (.md) counts as a specialization. No
    /// explicit type, generic aliases, built-in labels, and invented names all
    /// lack a custom definition, so the guard cannot be dodged with a made-up
    /// `subagent_type`.
    #[test]
    fn spawn_uses_custom_agent_only_for_real_custom_definitions() {
        let _lock = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "zo-dispatch-custom-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("custom agent dir");
        std::fs::write(
            dir.join("myrefactorer.md"),
            "---\nname: myrefactorer\ndescription: Custom\ntools: edit_file\n---\nRefactor locally.",
        )
        .expect("custom agent definition");
        let _defs = EnvGuard::set("ZO_AGENT_DEFS_DIR", &dir);

        assert!(!spawn_uses_custom_agent(None, "", ""));
        assert!(!spawn_uses_custom_agent(Some("general-purpose"), "", ""));
        assert!(!spawn_uses_custom_agent(Some("general"), "", ""));
        assert!(!spawn_uses_custom_agent(Some("implementer"), "", ""));
        assert!(!spawn_uses_custom_agent(Some("code-reviewer"), "", ""));
        assert!(spawn_uses_custom_agent(Some("myrefactorer"), "", ""));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn untyped_agent_is_stamped_before_routing() {
        let mut input: AgentInput = serde_json::from_value(json!({
            "description": "Debug the parser crash",
            "prompt": "reproduce the panic, find the root cause, and fix the bug"
        }))
        .expect("agent input");

        let auto_type = stamp_inferred_agent_type(&mut input);

        assert_eq!(auto_type.as_deref(), Some("debugger"));
        assert_eq!(input.subagent_type.as_deref(), Some("debugger"));
        let role = runtime::SubagentProfileId::parse("debugger")
            .and_then(|profile| profile.route_role_hint());
        assert_eq!(role, Some(runtime::RouteRole::Debugging));
        assert_eq!(
            super::super::agent_tools::display_agent_label(
                Some("Fix parser"),
                &input.description,
                "fix-parser",
                input.subagent_type.as_deref().expect("stamped type"),
            ),
            Some("debugger·Fix parser".to_string())
        );
        assert_eq!(
            annotate_auto_type_reason(None, auto_type.as_deref()).as_deref(),
            Some("type=debugger (auto)")
        );
    }

    #[test]
    fn spawn_members_stamp_only_missing_types() {
        let mut input: SpawnMultiAgentInput = serde_json::from_value(json!({
            "agents": [
                {
                    "description": "Find where sessions are loaded",
                    "prompt": "explore the codebase and return file references"
                },
                {
                    "description": "Classify task",
                    "prompt": "return one label",
                    "subagent_type": "classifier"
                }
            ]
        }))
        .expect("spawn input");

        let auto_types = stamp_inferred_spawn_types(&mut input);
        annotate_spawn_auto_type_reasons(&mut input, &auto_types);

        assert_eq!(input.agents[0]["subagent_type"], "Explore");
        assert_eq!(input.agents[1]["subagent_type"], "classifier");
        assert_eq!(auto_types[0].as_deref(), Some("Explore"));
        assert_eq!(auto_types[1], None);
        assert_eq!(
            input.agents[0][ROUTE_REASON_SMUGGLE_KEY],
            "type=Explore (auto)"
        );
        assert!(input.agents[1].get(ROUTE_REASON_SMUGGLE_KEY).is_none());
    }

    #[test]
    fn explicit_custom_type_survives_dispatch_and_job_resolution() {
        let _lock = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "zo-dispatch-custom-agent-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("custom agent dir");
        std::fs::write(
            dir.join("reviewer.md"),
            "---\nname: reviewer\ndescription: Custom reviewer\ntools: read_file\n---\nReview locally.",
        )
        .expect("custom agent definition");
        let _defs = EnvGuard::set("ZO_AGENT_DEFS_DIR", &dir);
        let mut input: AgentInput = serde_json::from_value(json!({
            "description": "Review the patch",
            "prompt": "review this code",
            "subagent_type": "reviewer"
        }))
        .expect("agent input");

        assert_eq!(stamp_inferred_agent_type(&mut input), None);
        let (resolved, custom) = super::super::agent_tools::resolve_subagent_type_and_custom_agent(
            input.subagent_type.as_deref(),
            &input.description,
            &input.prompt,
        );

        assert_eq!(resolved, "reviewer");
        assert_eq!(custom.expect("custom definition wins").name, "reviewer");
        let _ = std::fs::remove_dir_all(dir);
    }
}
