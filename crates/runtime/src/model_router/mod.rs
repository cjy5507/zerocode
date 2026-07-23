//! Smart Model Router core.
//!
//! Routing policy stays pure: callers provide a source-guarded usable model
//! inventory and routing preferences. The sibling outcome module only persists
//! aggregate-safe route results for later diagnostics; it never selects models,
//! reads settings, builds provider clients, or performs network discovery.

mod assignment;
mod calibration;
mod inventory;
mod learned;
mod outcome;
mod policy;
mod probe;
mod selector;
mod target;

#[cfg(test)]
mod tests;

pub use assignment::{
    recommend_auto_assignments, recommend_auto_assignments_with_feedback,
    recommend_auto_assignments_with_learned_specialty, recommend_auto_assignments_with_options,
    recommend_role_fallbacks, recommend_role_fallbacks_with_learned_specialty, AssignmentConfidence,
    AssignmentSource, AutoAssignmentOptions, AutoAssignmentPlan, InventorySummary, TargetAssignment,
};
pub use inventory::{
    ModelDescriptor, ModelInventory, ModelSource, UsableModel, UsableModelInventory,
};
pub use learned::{LearnedSpecialtyEntry, LearnedSpecialtyHint};
pub use outcome::{
    is_terminal_outcome_status, read_route_outcome_summary, read_route_outcomes,
    record_route_outcome, route_outcome_log_path, summarize_route_outcomes,
    summarize_route_outcomes_with_canonicalizer, weighted_feedback_hint_for_route_key,
    RouteOutcomeBucket, RouteOutcomeRecord, RouteOutcomeSummary, CONFIDENT_DECISIVE_SAMPLES,
};
pub use policy::{
    deep_tier_model_matches, default_deep_tier_models, exploration_slot_for_route,
    implementation_route_model_allowed, is_deep_tier_model, is_reserved_orchestrator_model,
    recommended_effort_for,
    route_model, route_model_fallback_candidates, EffortCeiling, FreshnessPolicy, LaneRouteMetadata, ModelCapability, ModelStatus,
    ModelTier, RouteAudit, RouteAutoClassifierMode, RouteConfidence, RouteContextNeed,
    RouteDecision, RouteDecisionSource, RouteDiversityNeed, RouteFeedbackHint,
    RouteOutputNeed, RoutePolicyContext, RouteRequest, RouterMode, RouteShapeKind,
    RouteSignalSource, RouteTaskComplexity, RouteTaskKind, RouteTaskRisk, RouteToolNeed,
    RouteVerificationNeed, SmartPolicy, TiersProvenance, DEFAULT_DEEP_TIER_MODELS,
};
pub use calibration::{
    read_route_outcomes_across_projects, route_outcome_log_paths_across_projects,
    ComplexityCalibration, CALIBRATION_FAILURE_SHARE, CALIBRATION_MIN_SAMPLES,
};
pub use probe::{
    fuse_probe_assessment, parse_probe_response, probe_prompt, ProbeAssessment, ProbeFusion,
    ProbeFusionEffect,
};
pub use selector::{RoleOverride, RoleSelector};
pub use target::{
    BuiltinSubagentProfile, RouteRole, RoutingTarget, SubagentProfileId, SubagentProfileKind,
};
