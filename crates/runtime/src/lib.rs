//! Core runtime primitives for the `zo` CLI and supporting crates.
//!
//! This crate owns session persistence, permission evaluation, prompt assembly,
//! MCP plumbing, tool-facing file operations, and the core conversation loop
//! that drives interactive and one-shot turns.

pub mod auto_format;
mod bash;
pub mod bash_validation;
mod bootstrap;
pub mod command_palette;
mod compact;
mod compact_diff;
mod config;
pub mod context_compression;
mod conversation;
mod convert_messages;
pub mod file_ops;
pub mod file_read_registry;
pub mod fuzzy_file_picker;
pub mod git_snapshot;
pub mod green_contract;
mod hooks;
pub mod image_guard;
mod jsonl_log;
pub mod lsp_client;
pub mod live_output;
mod mcp;
mod mcp_client;
pub mod mcp_http;
pub mod mcp_http_common;
pub mod mcp_lifecycle_hardened;
mod mcp_limits;
pub mod mcp_oauth;
pub mod mcp_sse;
mod mcp_stdio;
pub mod mcp_ws;
pub mod memory;
pub mod message_stream;
pub mod model_inventory;
pub mod model_catalog;
pub mod model_router;
pub mod notifications;
mod oauth;
pub mod permission;
pub mod permission_enforcer;
mod permissions;
pub mod plugin_lifecycle;
mod policy_engine;
mod prompt;
pub mod recovery_recipes;
mod registry_io;
mod remote;
pub mod retry;
pub mod sandbox;
pub mod secure_fs;
pub mod session_control;
pub mod skills;
pub mod stale_branch;
pub mod summary_compression;
pub mod task_packet;
pub mod task_registry;
mod team_inbox_digest;
pub mod team_cron_registry;
pub mod todo_progress;
pub mod todo_store;
pub mod trust_resolver;
pub mod worker_boot;

// Re-export modules from core-types so that `crate::json`, `crate::session`,
// `crate::usage`, `crate::lane_events`, and `crate::sse` still resolve for
// internal consumers within this crate.
pub use core_types::json;
pub use core_types::lane_events;
pub use core_types::session;
pub use core_types::sse;
pub use core_types::usage;

pub use bash::{execute_bash, execute_bash_with_tasks, BashCommandInput, BashCommandOutput};

#[must_use]
pub fn available_disk_bytes(dir: &std::path::Path) -> Option<u64> {
    bash::available_disk_bytes(dir)
}

#[must_use]
pub fn low_disk_warning(dir: &std::path::Path) -> Option<String> {
    bash::low_disk_warning(dir)
}
pub use bootstrap::{BootstrapPhase, BootstrapPlan};
pub use compact::{
    apply_compaction, compact_session, compact_session_with, compaction_system_prompt,
    distill_session_state, edited_file_paths, estimate_session_tokens, format_compact_summary,
    get_compact_continuation_message, is_edit_result_tool, microcompact_clearable_estimate,
    microcompact_session, prepare_compaction, should_compact, summary_fabricates_identifiers,
    CompactionConfig, CompactionPlan, CompactionResult, CompactionSummarizer, FocusSummarizer,
    LocalSummarizer, MicrocompactEvent, COMPACTION_SYSTEM_PROMPT, MICROCOMPACT_PLACEHOLDER,
};
pub use compact_diff::{
    CompactDiffHunk, CompactDiffLine, CompactDiffLineKind, compact_line_diff,
};
pub use config::{
    default_config_home, zo_global_config_roots, zo_project_state_dir, zo_state_base,
    CliConfigOverrides, ConfigEntry, ConfigError, ConfigLoader, ConfigSource, HookMatcher,
    HookRule, McpConfigCollection, McpManagedProxyServerConfig, McpOAuthConfig,
    McpRemoteServerConfig, McpSdkServerConfig, McpServerConfig, McpStdioServerConfig, McpTransport,
    McpWebSocketServerConfig, OAuthConfig, ResolvedPermissionMode, RuntimeConfig,
    RuntimeFeatureConfig, RuntimeHookConfig, RuntimePermissionRuleConfig, RuntimePluginConfig,
    RuntimeShipConfig, ScopedMcpServerConfig, UntrustedMcpServer, ZO_SETTINGS_SCHEMA_NAME,
};
pub use conversation::SteeringQueue;
pub use conversation::{AgentNotification, AgentNotificationInbox};
pub use conversation::{
    auto_compaction_threshold_for_model, auto_compaction_threshold_from_env, detect_check_command,
    env_deadline_extension, env_turn_budgets, final_assistant_text, read_only_bash_allow_rules,
    flush_pending_tool_events, prompt_cache_record_to_event, record_non_anthropic_prompt_cache_usage, push_output_block, redacted_thinking_data_to_string, response_to_events, ApiClient, ApiRequest,
    AssistantEvent, AsyncApiClient, AutoCompactionEvent, BudgetExhausted, ConcurrentDispatchFn,
    ConversationRuntime, DeepGateConfig, DeepMode,
    DeepOutcome, ExecContract, PromptCacheEvent, ProviderStateBlob, RuntimeError, StaticToolExecutor,
    StreamingTurnError, ToolError, ToolExecutor, TurnSummary, DEFAULT_STREAMING_CHANNEL_CAPACITY,
    DEFAULT_TURN_DEADLINE_SECS, DEFAULT_TURN_INPUT_TOKEN_BUDGET, DEFAULT_TURN_OUTPUT_TOKEN_BUDGET,
    STEERING_ECHO_PREFIX,
};
/// Provider failure classification carried on [`RuntimeError`]. Re-exported so
/// callers (and tests) can construct/inspect a classified error — notably the
/// `RateLimit` class the quota-fallback turn loop keys off — without reaching
/// past `runtime` into the `api` crate.
pub use api::ProviderErrorClass;
pub use convert_messages::{
    append_wire_reminders, convert_messages, mark_conversation_cache_breakpoints,
    mark_conversation_cache_breakpoints_short_ttl,
};
pub use core_types::{format_usd, pricing_for_model};
pub use core_types::{
    ContentBlock, ConversationMessage, CouncilOutcome, IncrementalSseParser, JsonError, JsonValue,
    LaneEvent, LaneEventBlocker, LaneEventName, LaneEventStatus, LaneFailureClass, MemoryEntry,
    MemoryHit, MemoryRetriever, MessageRole, ModelPricing, Session, SessionCompaction,
    SessionError, SessionFork, SseEvent, TokenUsage, UsageCostEstimate, UsageTracker,
};
pub use file_ops::{
    edit_file, glob_search, grep_search, read_file, write_file, EditFileOutput, GlobSearchOutput,
    GrepSearchInput, GrepSearchOutput, ReadFileOutput, StructuredPatchHunk, TextFilePayload,
    WriteFileOutput,
};
pub use file_read_registry::{FileFreshness, FileReadRegistry};
pub use hooks::{
    HookAbortOrigin, HookAbortSignal, HookEvent, HookProgressEvent, HookProgressReporter,
    HookRunResult, HookRunner,
};
pub use team_inbox_digest::{
    ensure_session_channel_subscription, team_inbox_manual_ack, team_inbox_snapshot,
    team_inbox_store_root, team_inbox_unread_count, TeamInboxSnapshot, TeamInboxSnapshotRow,
};
pub use mcp::{
    mcp_server_signature, mcp_tool_name, mcp_tool_prefix, normalize_name_for_mcp,
    scoped_mcp_config_hash, unwrap_ccr_proxy_url,
};
pub use mcp_client::{
    McpClientAuth, McpClientBootstrap, McpClientTransport, McpManagedProxyTransport,
    McpRemoteTransport, McpSdkTransport, McpStdioTransport,
};
pub use mcp_http::{connect_mcp_http, McpHttpProcess};
pub use mcp_lifecycle_hardened::{
    McpDegradedReport, McpErrorSurface, McpFailedServer, McpLifecyclePhase, McpLifecycleState,
    McpLifecycleValidator, McpPhaseResult,
};
pub use mcp_sse::{connect_mcp_sse, McpSseProcess};
pub use mcp_stdio::{
    spawn_mcp_stdio_process, InboundEvent, JsonRpcError, JsonRpcId, JsonRpcRequest,
    JsonRpcResponse, ManagedMcpTool, McpDiscoveryClass, McpDiscoveryFailure, McpGetPromptParams,
    McpGetPromptResult, McpInitializeClientInfo, McpInitializeParams, McpInitializeResult,
    McpInitializeServerInfo, McpListPromptsParams, McpListPromptsResult, McpListResourcesParams,
    McpListResourcesResult, McpListToolsParams, McpListToolsResult, McpPrompt, McpPromptArgument,
    McpPromptMessage, McpReadResourceParams, McpReadResourceResult, McpResource,
    McpResourceContents, McpServerManager, McpServerManagerError, McpStdioProcess, McpTool,
    McpToolCallContent, McpToolCallParams, McpToolCallResult, McpToolDiscoveryReport,
    UnsupportedMcpServer,
};
pub use mcp_ws::{connect_mcp_ws, McpWsProcess};
pub use model_inventory::{connected_model_inventory, model_inventory_from_authorized_providers};
pub use model_router::{
    deep_tier_model_matches, default_deep_tier_models, exploration_slot_for_route,
    fuse_probe_assessment, implementation_route_model_allowed, is_deep_tier_model,
    is_reserved_orchestrator_model,
    is_terminal_outcome_status, parse_probe_response, probe_prompt, ComplexityCalibration,
    ProbeAssessment, ProbeFusion, ProbeFusionEffect, read_route_outcome_summary,
    read_route_outcomes, read_route_outcomes_across_projects, recommend_auto_assignments,
    recommend_auto_assignments_with_feedback, recommend_auto_assignments_with_learned_specialty,
    recommend_auto_assignments_with_options,
    recommend_role_fallbacks, recommend_role_fallbacks_with_learned_specialty, record_route_outcome,
    recommended_effort_for,
    route_model, route_model_fallback_candidates,
    route_outcome_log_path, summarize_route_outcomes, summarize_route_outcomes_with_canonicalizer,
    weighted_feedback_hint_for_route_key, AssignmentSource, AutoAssignmentOptions,
    AutoAssignmentPlan, BuiltinSubagentProfile, EffortCeiling, FreshnessPolicy, LaneRouteMetadata,
    LearnedSpecialtyEntry, LearnedSpecialtyHint, CONFIDENT_DECISIVE_SAMPLES,
    ModelCapability, ModelDescriptor, ModelInventory, ModelStatus, ModelTier, RoleOverride,
    RoleSelector, RouteAudit, RouteAutoClassifierMode, RouteConfidence, RouteContextNeed,
    RouteDecision, RouteDecisionSource, RouteDiversityNeed, RouteFeedbackHint,
    RouteOutputNeed, RouteOutcomeBucket, RouteOutcomeRecord, RouteOutcomeSummary,
    RoutePolicyContext, RouteRequest, RouteRole, RouteShapeKind, RouteSignalSource,
    RouteTaskComplexity, RouteTaskKind, RouteTaskRisk, RouteToolNeed, RouteVerificationNeed,
    RouterMode, RoutingTarget, SmartPolicy, SubagentProfileId, SubagentProfileKind, TiersProvenance,
    DEFAULT_DEEP_TIER_MODELS,
};
pub use memory::{
    dream_at_cwd, load_lexical_memory_retriever, load_memory_retriever, maybe_auto_dream,
    parse_memory_index, record_auto_dream_failure, record_automation_event, record_observation,
    record_verified_check, render_recalled_memory_section, DreamReport, Dreamer,
    LexicalMemoryRetriever,
};
pub use mcp_oauth::open_browser;
pub use oauth::{
    clear_mcp_oauth_token, clear_oauth_credentials, clear_openai_oauth, code_challenge_s256,
    credentials_path, generate_pkce_pair, generate_state, is_mcp_token_expired,
    list_mcp_oauth_servers, load_mcp_oauth_token, load_oauth_credentials, load_openai_oauth,
    loopback_redirect_uri, parse_oauth_callback_query, parse_oauth_callback_request_target,
    save_mcp_oauth_token, save_oauth_credentials, save_openai_oauth, OAuthAuthorizationRequest,
    OAuthCallbackParams, OAuthRefreshRequest, OAuthTokenExchangeRequest, OAuthTokenSet,
    OpenAiOAuthTokens, PkceChallengeMethod, PkceCodePair,
};
pub use permissions::{
    PermissionContext, PermissionMode, PermissionOutcome, PermissionOverride, PermissionPolicy,
    PermissionPromptDecision, PermissionPrompter, PermissionRequest, TemporaryAllowGrant,
};
pub use plugin_lifecycle::{
    DegradedMode, DiscoveryResult, PluginHealthcheck, PluginLifecycle, PluginLifecycleEvent,
    PluginState, ResourceInfo, ServerHealth, ServerStatus, ToolInfo,
};
pub use policy_engine::{
    DiffScope, GreenLevel, LaneBlocker, LaneContext, PolicyAction, PolicyCondition, PolicyEngine,
    PolicyRule, ReconcileReason, ReviewStatus,
};
pub use prompt::{
    discover_skills, load_system_prompt, load_system_prompt_for_main,
    load_system_prompt_for_main_with_mode, output_style, prepend_bullets, skill_search_roots,
    split_system_with_identity, ContextFile, ProjectContext, PromptBuildError, PromptMode,
    SkillIndexEntry, SkillInvocationMode, SkillTriggers, SystemPromptBuilder,
    CLAUDE_CODE_IDENTITY, FRONTIER_MODEL_NAME, SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
};
pub use skills::{
    build_skill_recommendation_reminder, recommend_skills, SkillDecision, SkillMatchInput,
    SkillRecommendation, SKILL_RECOMMENDATION_REMINDER_PREFIX,
};
pub use recovery_recipes::{
    attempt_recovery, recipe_for, EscalationPolicy, FailureScenario, RecoveryContext,
    RecoveryEvent, RecoveryRecipe, RecoveryResult, RecoveryStep,
};
pub use remote::{
    inherited_upstream_proxy_env, no_proxy_list, read_token, upstream_proxy_ws_url,
    RemoteSessionContext, UpstreamProxyBootstrap, UpstreamProxyState, DEFAULT_REMOTE_BASE_URL,
    DEFAULT_SESSION_TOKEN_PATH, DEFAULT_SYSTEM_CA_BUNDLE, NO_PROXY_HOSTS, UPSTREAM_PROXY_ENV_KEYS,
};
pub use sandbox::{
    build_linux_sandbox_command, detect_container_environment, detect_container_environment_from,
    resolve_sandbox_status, resolve_sandbox_status_for_request, ContainerEnvironment,
    FilesystemIsolationMode, LinuxSandboxCommand, SandboxConfig, SandboxDetectionInputs,
    SandboxRequest, SandboxStatus,
};
mod tool_output_truncation;
pub use tool_output_truncation::{truncate_tool_output, TruncatedOutput, TruncationConfig};
pub mod turn_trace;
pub use turn_trace::{TurnOutcome, TurnRecord};

pub use stale_branch::{
    apply_policy, check_freshness, BranchFreshness, StaleBranchAction, StaleBranchEvent,
    StaleBranchPolicy,
};
pub use task_packet::{validate_packet, TaskPacket, TaskPacketValidationError, ValidatedPacket};

/// Env flag enabling per-phase turn timing logs (set to any value).
pub const PROFILE_TURN_ENV: &str = "ZO_PROFILE_TURN";

/// Whether `ZO_PROFILE_TURN` turn profiling is on. Read per call (no
/// memoization) so long-lived sessions observe changes; every caller sits on
/// a >=50ms slow path where one getenv is noise.
#[must_use]
pub fn turn_profiling_enabled() -> bool {
    std::env::var(PROFILE_TURN_ENV).is_ok()
}
pub use trust_resolver::{TrustConfig, TrustDecision, TrustEvent, TrustPolicy, TrustResolver};
pub use worker_boot::{
    Worker, WorkerEvent, WorkerEventKind, WorkerEventPayload, WorkerFailure, WorkerFailureKind,
    WorkerPromptTarget, WorkerReadySnapshot, WorkerRegistry, WorkerStatus, WorkerTrustResolution,
};

#[cfg(test)]
pub(crate) fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
