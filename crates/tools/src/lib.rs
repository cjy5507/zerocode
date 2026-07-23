//! `tools` crate root.
//!
//! Production surface lives in dedicated modules; this file is the public
//! API + crate-internal re-export hub only.
//!
//! ```text
//! lib.rs (this file)        ── module dispatch + pub/pub(crate) use
//!   ├── context.rs          ── ToolContext + UserQuestionChannel
//!   ├── aliases.rs          ── canonical_tool_name + TOOL_NAME_ALIASES
//!   ├── registry.rs         ── ToolSpec + GlobalToolRegistry + mvp_tool_specs
//!   ├── dispatch.rs         ── execute_tool + dispatch_tool_inner + helpers
//!   ├── preflight.rs        ── workspace test branch divergence guard
//!   └── error.rs            ── ToolError
//! ```
//!
//! Tool-family handlers (`bash_tools`, `file_tools`, `web_tools`, ...) are
//! unchanged — only the cross-cutting glue that used to live inline was
//! extracted as part of Phase 3.1 of the refactor.

mod aliases;
mod artifacts;
mod bash_tools;
mod codegraph_tools;
mod context;
mod dispatch;
pub mod error;
mod fanout;
mod file_tools;
mod file_write_lease;
mod gateway;
mod hunk_attribution;
mod http_bridge;
mod mcp_tools;
mod misc_tools;
mod model_json;
mod plan_mode_v2;
mod preflight;
mod registry;
mod task_tools;
mod team_inbox_store;
mod team_tools;
mod typed_actions;
mod web_tools;
mod worker_tools;
mod workflow_tools;
mod workspace_scope_guard;
mod workspace_checkpoint;
mod worktree_tools;

// Public API.
pub use aliases::canonical_tool_name;
pub use artifacts::{read_artifact, store_artifact, ArtifactKind, ArtifactRef};
pub use context::{ToolContext, TurnAgentPolicy, UserQuestionChannel};
pub use dispatch::{enforce_permission_check, execute_tool};
pub use error::ToolError;
pub use fanout::{
    clarify_intent, decompose_for_fanout, decompose_for_fanout_with_timeout,
    decompose_for_fanout_with_timeout_and_hooks, run_fanout_spawn, run_fanout_spawn_with_timeout,
    diagnose_lens_labels, run_diagnose_fanout, run_fanout_spawn_with_timeout_and_hooks,
    run_self_consistency_fanout, FanoutMode, FanoutSubtask, IntentTriage, AUTO_FANOUT_AGENT_TIMEOUT,
    AUTO_FANOUT_DECOMPOSE_TIMEOUT, MAX_FANOUT_SUBTASKS, SELF_CONSISTENCY_K,
};
pub use gateway::{
    summarize_invocations, AuditDenial, AuditSummary, RouteDecisionRecord, ToolErrorKind,
    ToolFamily, ToolInvocation, ToolInvocationRequest, ToolInvocationResult, ToolPolicyCheck,
    ToolPolicyDecision, ToolResultMetadata,
};
pub use runtime::live_output;
pub use hunk_attribution::{
    AttributionLine, AttributionLineKind, AttributionOrigin, AttributionStatus, AttributedHunk,
    HunkAttributionLedger, ReviewHunkError, apply_reverse_patch,
};
pub use misc_tools::agent_store_dir;
pub use misc_tools::{loaded_custom_agents, LoadedCustomAgent};
pub use misc_tools::{send_agent_message, AgentSendOutcome};
pub use misc_tools::{
    drain_background_completions_for_session, fold_background_completions_into_input,
};
pub use misc_tools::{
    assess_agent_task, assess_turn_complexity, assess_turn_orchestration, turn_has_write_intent,
    AgentTaskAssessment, TurnOrchestrationHint,
};
pub use misc_tools::{
    smart_deep_tier_models, smart_deep_tier_models_for, smart_exec_swap, smart_setting_defaults,
    DeepTierModelsSetting, SmartExecSwap, SmartSettingDefaults,
};
pub use misc_tools::ToolSearchOutput;
pub use misc_tools::{
    agent_worker_is_live, background_completion_matches_session, clear_background_agent,
    execute_config, execute_enter_plan_mode, execute_exit_plan_mode, is_background_agent,
    reconcile_dead_agent_worker,
    mark_background_agent,
    notify_background_task_completion, notify_remote,
    parent_session_belongs,
    register_agent_completion_channel,
    reap_orphaned_agents, stop_running_agents_since,
    stop_running_agents_since_for_session,
    stop_running_agents_since_for_strict_session, wait_for_agent_completions, AgentCompletion,
    ConfigInput, ConfigOutput, ConfigValue, EnterPlanModeInput, ExitPlanModeInput, PlanModeOutput,
    AGENT_STARVED_STATUS, provider_error_class_from_completion, provider_error_class_metadata,
};
pub use registry::{
    deferred_tool_manifest_section, mvp_tool_specs, GlobalToolRegistry, RuntimeToolDefinition,
    ToggleableTool, ToggleableToolSource, ToolManifestEntry, ToolRegistry, ToolSource, ToolSpec,
};
pub use team_tools::{ensure_team_inbox_store, host_post_team_inbox_update};
pub use typed_actions::{run_process_spec, CargoAction, GitAction, ProcessOutcome, ProcessSpec};
pub use workflow_tools::{
    event_log_terminal_status, event_phase_statuses, event_timeline_lines, read_event_log,
    request_foreground_workflow_cancel, EventPhase, WorkflowEventKind, WorkflowEventRecord,
};
pub use workspace_checkpoint::{
    render_workspace_checkpoint_list, render_workspace_restore_summary, WorkspaceCheckpoint,
    WorkspaceCheckpointFile, WorkspaceFileSnapshot, WorkspaceRestoreSkippedPath,
    WorkspaceRestoreSummary, WorkspaceSnapshotSkip,
    MAX_CHECKPOINT_FILE_BYTES, MAX_WORKSPACE_CHECKPOINTS,
};

/// Compact capability tag for the static toolset behind a subagent type.
#[must_use]
pub fn subagent_toolset_class(subagent_type: &str) -> &'static str {
    let tools = misc_tools::allowed_tools_for_subagent(subagent_type);
    let builtin = runtime::BuiltinSubagentProfile::all()
        .iter()
        .any(|profile| profile.key().eq_ignore_ascii_case(subagent_type));
    if !builtin {
        "custom"
    } else if tools.contains("Config") || tools.contains("Sleep") {
        "full"
    } else if tools.contains("edit_file") || tools.contains("write_file") {
        "edit"
    } else {
        "read-only"
    }
}

// Crate-internal helpers — `*_tools.rs` siblings reach these through
// `crate::xxx` so the re-export here is what keeps the existing call sites
// working without a per-file path rewrite. `permission_mode_from_plugin`
// and `normalize_shell_command` are only touched from `#[cfg(test)]`, so
// silence the unused-import warning that fires on a non-test build.
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use aliases::permission_mode_from_plugin;
pub(crate) use dispatch::{
    epoch_seconds_now, execute_tool_with_context, from_value, maybe_enforce_permission_check,
    to_pretty_json,
};
pub(crate) use file_write_lease::{
    acquire as acquire_write_lease, release_all_for_owner as release_write_leases, LeaseOutcome,
};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use preflight::normalize_shell_command;
pub(crate) use preflight::workspace_test_branch_preflight;
pub(crate) use registry::SearchableToolSpec;
pub(crate) use workspace_scope_guard::{workspace_guard_enabled, workspace_scope_guard};

#[cfg(test)]
mod tests;
