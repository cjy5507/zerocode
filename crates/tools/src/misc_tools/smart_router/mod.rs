mod apply;
mod canonical;
mod evidence;
mod infer;
mod metadata;
mod planner;
mod probe_exec;
mod settings;
mod shape;
mod turn;

#[cfg(test)]
mod tests;

pub(crate) use apply::{
    apply_smart_models_to_spawn_input_with_auto_types, smart_parent_model_for_agent,
    smart_parent_model_for_agent_with_auto_type,
    ROUTE_DECISION_META_SMUGGLE_KEY, ROUTE_EFFORT_SMUGGLE_KEY,
    ROUTE_FALLBACK_MODELS_SMUGGLE_KEY, ROUTE_JUDGED_AGENT_SMUGGLE_KEY,
    ROUTE_MODEL_SMUGGLE_KEY, ROUTE_REASON_SMUGGLE_KEY,
};
pub(crate) use canonical::canonicalize_route_model_id;
pub(crate) use settings::live_smart_policy;
pub use settings::{
    smart_deep_tier_models, smart_deep_tier_models_for, smart_exec_swap, smart_setting_defaults,
    DeepTierModelsSetting, SmartExecSwap, SmartSettingDefaults,
};
pub use turn::{
    assess_agent_task, assess_turn_complexity, assess_turn_orchestration, turn_has_write_intent,
    AgentTaskAssessment, TurnOrchestrationHint,
};

pub(crate) fn agent_task_has_write_intent(description: &str, prompt: &str) -> bool {
    infer::task_has_write_intent(description, prompt)
}
