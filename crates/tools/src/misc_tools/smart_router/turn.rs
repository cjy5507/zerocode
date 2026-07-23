use runtime::{RouteAutoClassifierMode, RouteConfidence, RouteShapeKind};

use super::evidence::{infer_route_shape_evidence, shape_input_with_evidence, RouteEvidenceInput};
use super::infer::infer_route_role;
use super::metadata::{classify_task_metadata, TaskMetadataInput};
use super::planner::plan_agent_needs;
use super::shape::select_route_shape;

/// Smart-routing assessment of a whole turn (the user's prompt). The host
/// orchestrator consults this to *drive* a route decision when its own
/// (multilingual) keyword classifier finds no delegation signal, so the smart
/// routing layer can influence orchestration rather than only model selection.
///
/// Pure and deterministic: it reads no settings and performs no IO, so it is
/// safe to call from the host's per-turn route classifier (which is unit-tested
/// as a pure function). Uses provider-free deterministic classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnOrchestrationHint {
    /// The canonical route shape the evidence suggests for this turn.
    pub shape: RouteShapeKind,
    /// How many distinct agent needs the planner found (0 → no delegation value).
    pub need_count: usize,
    /// Confidence in the suggested shape.
    pub confidence: RouteConfidence,
    /// Deterministic task-risk band from the same metadata classification.
    pub risk: runtime::RouteTaskRisk,
    /// The user EXPLICITLY requested a delegation shape in the turn text (the
    /// evidence pipeline's `requested_shape` resolved to a non-`Solo` shape,
    /// e.g. "use one specialist", "in parallel"). Surfaced separately because
    /// the decision `shape` is the *natural* shape, which stays `Solo` whenever
    /// the need plan is empty — so an explicit request would otherwise be lost.
    /// The same-model spawn guard honors this as "the user asked to delegate".
    pub user_requested_delegation: bool,
}

/// Classify a whole turn's task complexity with the same deterministic
/// classifier the per-agent router uses (role inference + metadata tables,
/// Korean parity, typo/coexisting-work guards — all pinned by the complexity
/// evaluation corpus). The host consults this for RESOURCE allocation only —
/// e.g. the Smart dynamic effort band's per-turn floor, so a trivial ask
/// answers fast — never to steer the model's own orchestration judgment,
/// which lives in the base prompt's delegation rubric.
#[must_use]
pub fn assess_turn_complexity(user_text: &str) -> runtime::RouteTaskComplexity {
    let role = infer_route_role(None, "", user_text);
    classify_task_metadata(&TaskMetadataInput::new(None, "", user_text), role).complexity
}

/// Whether a whole turn's text carries concrete implementation/write intent,
/// using the SAME multilingual keyword classifier the per-agent router's
/// write-intent gate uses (`task_has_write_intent`). The host's Architect
/// contract consults this to decide whether a turn is implementation-shaped
/// (EXEC legs swap to the implementer client) — pure and deterministic, safe
/// for the per-turn host path.
#[must_use]
pub fn turn_has_write_intent(user_text: &str) -> bool {
    super::infer::task_has_write_intent("", user_text)
}

/// Complexity + write-intent of a DELEGATED agent task (its `description` +
/// `prompt` slice), distinct from the whole-turn assessment. The guard needs
/// the slice's own difficulty/effect, not the user turn's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentTaskAssessment {
    pub complexity: runtime::RouteTaskComplexity,
    pub has_write_intent: bool,
}

/// Classify a delegated agent task from the EXECUTABLE task text — the spawn's
/// `prompt`, which is what the child actually runs (`AgentJob::prompt`) — using
/// the SAME corpus-pinned role/metadata and write-intent classifiers as the
/// whole-turn helpers. Deliberately ignores `description`, `name`,
/// `subagent_type`, and model/route fields: those are model-authored labels the
/// child never executes, so classifying them would let a spawn dodge the guard
/// (e.g. an inflated `description`) without changing the real task. Pure and
/// deterministic.
#[must_use]
pub fn assess_agent_task(prompt: &str) -> AgentTaskAssessment {
    let role = infer_route_role(None, "", prompt);
    let complexity =
        classify_task_metadata(&TaskMetadataInput::new(None, "", prompt), role).complexity;
    let has_write_intent = super::infer::task_has_write_intent("", prompt);
    AgentTaskAssessment {
        complexity,
        has_write_intent,
    }
}

/// Assess a turn's orchestration shape from smart-routing evidence (metadata
/// classifier + need planner + shape selector), reusing the exact pipeline the
/// per-agent router uses so the host and model layers agree on the taxonomy.
#[must_use]
pub fn assess_turn_orchestration(user_text: &str) -> TurnOrchestrationHint {
    let role = infer_route_role(None, "", user_text);
    let metadata = classify_task_metadata(&TaskMetadataInput::new(None, "", user_text), role);
    let needs = plan_agent_needs(&metadata);
    let evidence = infer_route_shape_evidence(&RouteEvidenceInput {
        subagent_type: None,
        name: None,
        description: "",
        prompt: user_text,
        workflow_member: false,
        fanout_position: None,
        auto_classifier: RouteAutoClassifierMode::Deterministic,
    });
    // The user's explicitly requested shape (from the evidence pipeline) is
    // discarded by `select_route_shape`, which returns the natural shape — so
    // capture it here. A non-`Solo` requested shape means the user asked to
    // delegate, even when the natural shape is `Solo` (empty need plan).
    let user_requested_delegation = evidence
        .requested_shape
        .is_some_and(|requested| !matches!(requested, RouteShapeKind::Solo));
    let decision = select_route_shape(&shape_input_with_evidence(&metadata, &needs, &evidence));
    TurnOrchestrationHint {
        shape: decision.shape,
        need_count: needs.len(),
        confidence: decision.confidence,
        risk: metadata.risk,
        user_requested_delegation,
    }
}
