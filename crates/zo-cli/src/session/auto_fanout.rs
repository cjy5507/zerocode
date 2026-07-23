//! Adaptive route hint for a live turn — the *host* side of the router.
//!
//! This is a cheap, model-free, **advisory** classifier. It does not force a
//! shape: the real delegation decision moves to the model, which is taught the
//! four routing shapes by the always-on delegation rubric in the base system
//! prompt (`runtime::prompt::sections::get_delegation_section`). The host's job
//! here is narrow:
//!
//! 1. Produce a [`RouteHint`] (best-guess [`RouteShape`] + confidence + reasons)
//!    surfaced to the model as a one-line system reminder, so it starts from a
//!    sensible default instead of re-deriving intent from scratch.
//! 2. Decide whether the host should *pre-spawn* a flat fan-out before the model
//!    turn ([`RouteHint::should_host_prespawn`]). To avoid the split-brain where
//!    the host pre-spawns analysis **and** the model also spawns agents (BUG-D2),
//!    the host now pre-spawns only on an explicit parallel request or in
//!    ultracode mode — every other turn defers to the model.
//!
//! Per the work-order, the fix is *not* a bigger keyword table (a dead end);
//! the buckets below stay deliberately small and each token lives in exactly one
//! bucket (so no signal is double-counted — the old BUG-R1).

use std::collections::BTreeSet;

use zo_cli::tui::modals::Effort;

/// Prefix for the per-turn route-hint system reminder. Stable so the turn
/// controller can replace/clear it each turn via
/// `replace_transient_system_reminder_by_prefix` (no stale hint lingers).
pub(crate) const ROUTE_HINT_REMINDER_PREFIX: &str = "[zo:route-hint]";

/// Route-hint reminder installed *after* a host prelude actually fanned out this
/// turn (its findings are seated in context as a synthetic `SpawnMultiAgent`
/// result). It replaces the original "consider delegating" nudge so the model
/// builds on the pre-analysis instead of re-running its own fan-out — enforcing,
/// not merely nudging, the host-XOR-model spawn invariant (BUG-D2). Starts with
/// [`ROUTE_HINT_REMINDER_PREFIX`] so it occupies the same transient slot.
pub(crate) const PRELUDE_FANNED_OUT_REMINDER: &str = "[zo:route-hint] Pre-analysis already ran this turn and its findings are in this turn's context; build on them — do NOT start another fan-out (Agent/SpawnMultiAgent/Workflow) unless you uncover genuinely new, independent work the pre-analysis did not cover.";

/// Exact tool/API directives for parallel orchestration. Natural-language
/// collaboration requests are intentionally NOT listed here: the LLM triage
/// prompt decides those from semantic meaning. This table is only for users who
/// explicitly name the host orchestration primitive.
const EXPLICIT_PARALLEL_TARGETS: &[&str] = &[
    "spawnmultiagent",
    "spawn multi agent",
    "spawn_multi_agent",
    "workflow fanout",
    "workflow fan-out",
];

const EXPLICIT_PARALLEL_VERBS: &[&str] =
    &["use", "call", "run", "invoke", "start", "launch", "execute"];

/// Explicit requests to *not* delegate — force [`RouteShape::Solo`].
const EXPLICIT_NO_DELEGATION: &[&str] = &[
    "에이전트 쓰지",
    "에이전트 없이",
    "혼자 처리",
    "혼자서",
    "위임하지",
    "don't delegate",
    "do not delegate",
    "without agents",
    "no agents",
    "no delegation",
];

/// Implementation/change intent — the model should plan→implement→verify
/// (a pipeline), not return analysis only.
const IMPLEMENT_INTENT: &[&str] = &[
    "구현",
    "수정",
    "고쳐",
    "버그",
    "디버그",
    "리팩터",
    "리팩토링",
    "마이그",
    "적용해",
    "개선",
    "향상",
    "최적화",
    "문서화",
    "implement",
    "improve",
    "improvement",
    "optimize",
    "document ",
    "documentation",
    "fix",
    "debug",
    "refactor",
    "migrat",
];

/// Verification intent — paired with implementation it strengthens Pipeline.
const VERIFY_INTENT: &[&str] = &[
    "검증",
    "테스트",
    "통과",
    "verify",
    "verif",
    "test",
    "make it pass",
    "make the tests",
];

/// Strong bug/root-cause signals that justify ultracode's cheap semantic
/// triage consuming a model-led Pipeline if `clarify_intent` returns Diagnose.
/// Generic implementation/improvement prompts intentionally do not match here:
/// they should keep their plan→implement→verify route instead of being
/// reinterpreted as `BUG:\n{prompt}` for root-cause fan-out.
const DIAGNOSE_TRIAGE_INTENT: &[&str] = &[
    "버그",
    "디버그",
    "원인",
    "왜 ",
    "실패",
    "재현",
    "오류",
    "에러",
    "멈춤",
    "hang",
    "crash",
    "panic",
    "regression",
    "failing",
    "failure",
    "debug",
    "root cause",
    "why ",
    "repro",
    "reproduce",
    "broken",
];

/// Breadth-first analysis/search intent — read-only investigation that may
/// split into independent slices.
const BREADTH_INTENT: &[&str] = &[
    "분석",
    "조사",
    "리뷰",
    "감사",
    "비교",
    "전수",
    "광범위",
    "찾아",
    "취약",
    "analyz",
    "investigat",
    "review",
    "audit",
    "compare",
    "survey",
    "find ",
];

/// Multi-scope signals — several directories/modules/perspectives, which makes a
/// breadth task a fan-out candidate rather than a single-area investigation.
/// Tokens that used to also live in [`BREADTH_INTENT`] (`각각`, `전반`, `across`)
/// live *only* here now, so a single word is never counted twice (BUG-R1).
const MULTI_SCOPE_INTENT: &[&str] = &[
    "프로젝트",
    "코드베이스",
    "전체",
    "전반",
    "여러",
    "각각",
    "하네스",
    "across",
    "multiple",
    "crates/",
    "workspace",
    "codebase",
    "whole project",
    "entire project",
    "runtime",
    "frontend",
    "backend",
    "harness",
];

/// Small, local edits — bias toward [`RouteShape::Solo`].
const SMALL_SINGLE_STEP_INTENT: &[&str] = &[
    "one function",
    "single function",
    "rename",
    "typo",
    "format only",
    "quick",
    "간단",
    "한 함수",
    "한 줄",
    "이름만",
    "오타",
];

/// Explicit scope boundaries that make implementation cheaper to complete in
/// the main turn than to hand off. These are scope declarations, not generic
/// intent keywords, and intentionally stay separate from the routing buckets.
const BOUNDED_IMPLEMENTATION_SCOPE: &[&str] = &[
    "one file",
    "single file",
    "one module",
    "single module",
    "한 파일",
    "파일 하나",
    "단일 파일",
    "한 모듈",
    "모듈 하나",
    "단일 모듈",
];

/// Extensions that identify an explicitly named implementation target. Docs
/// and settings formats are excluded: reading `SPEC.md` or `Cargo.toml` is
/// context, not a second implementation scope.
const IMPLEMENTATION_FILE_EXTENSIONS: &[&str] = &[
    "c", "cc", "cpp", "cs", "css", "go", "h", "hpp", "html", "java", "js", "jsx",
    "kt", "kts", "php", "proto", "py", "rb", "rs", "scala", "sh", "sql", "svelte",
    "swift", "ts", "tsx", "vue",
];

/// Accumulated-context size (tokens) above which a breadth-first intent is large
/// enough to be worth splitting across agents.
const LARGE_CONTEXT_TOKENS: usize = 8_000;

/// The four execution shapes the host can hint and the model chooses among.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteShape {
    /// Handle directly, no delegation.
    Solo,
    /// One focused specialist on a single bounded area.
    DelegateOne,
    /// Independent slices run in parallel and synthesized.
    FanoutParallel,
    /// Dependent phases: plan → implement → verify.
    Pipeline,
}

impl RouteShape {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Solo => "Solo",
            Self::DelegateOne => "DelegateOne",
            Self::FanoutParallel => "FanoutParallel",
            Self::Pipeline => "Pipeline",
        }
    }

    /// The canonical model-routing shape (`runtime::RouteShapeKind`) this host
    /// orchestration shape corresponds to. The host pre-spawn decision (this
    /// 4-shape taxonomy) and the per-agent model router (the 6-shape
    /// `RouteShapeKind`) previously had no shared vocabulary; this mapping lets
    /// the recorded route decision express the host shape in the same taxonomy
    /// the model router uses. The host has no repair-loop shape (that is a
    /// per-phase workflow concern), so it never produces
    /// `RepairLoop`/`ParallelRepairLoop`.
    pub(crate) fn to_route_shape_kind(self) -> runtime::RouteShapeKind {
        match self {
            Self::Solo => runtime::RouteShapeKind::Solo,
            Self::DelegateOne => runtime::RouteShapeKind::OneSpecialist,
            Self::FanoutParallel => runtime::RouteShapeKind::ParallelLanes,
            Self::Pipeline => runtime::RouteShapeKind::SequentialWorkflow,
        }
    }

    /// One step up the escalation ladder used after a prior-turn failure: toward
    /// more structure and verification. `FanoutParallel` collapses into
    /// `Pipeline` (plan→implement→verify) rather than staying breadth-only, and
    /// `Pipeline` is the ceiling. Never escalates *to* `FanoutParallel`; under
    /// ultracode the non-breadth shapes (`DelegateOne`/`Pipeline`) host-prespawn
    /// too, so `escalate` (not this fn) guards the no-split-brain invariant by
    /// reverting a bump that would newly flip prespawn on.
    fn escalated(self) -> Self {
        match self {
            Self::Solo => Self::DelegateOne,
            Self::DelegateOne | Self::FanoutParallel | Self::Pipeline => Self::Pipeline,
        }
    }
}

/// A turn-level failure observed by the host, fed into the next turn's route as
/// an escalation signal (WI-B). Kept in this crate (route input lives here); no
/// cross-crate consumer needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FailureSignal {
    /// A test the turn relied on went red.
    RedTest,
    /// Repeated empty/again-empty model responses.
    RepeatedEmptyResponse,
    /// The turn hit a tool-call / iteration budget.
    ToolBudgetExceeded,
    /// The turn hit a wall-clock deadline / timeout.
    Deadline,
    /// The turn crossed a cumulative token budget (output or full-price input).
    TokenBudgetExceeded,
}

impl FailureSignal {
    /// The static note appended to the model reminder explaining the escalation.
    fn note(self) -> &'static str {
        match self {
            Self::RedTest => "the previous turn left a test red",
            Self::RepeatedEmptyResponse => "the previous turn kept returning empty responses",
            Self::ToolBudgetExceeded => "the previous turn ran out of its tool-call budget",
            Self::Deadline => "the previous turn hit its deadline",
            Self::TokenBudgetExceeded => "the previous turn ran out of its token budget",
        }
    }
}

/// Map a graceful budget stop to the escalation-ladder vocabulary. A turn that
/// exhausts a budget ends `Ok` (work preserved), but for routing purposes it
/// failed to converge — without this mapping the runaway breaker's graceful
/// stops were invisible to WI-B, and the exhaust→continue→exhaust grind never
/// escalated.
pub(crate) fn failure_signal_for_budget(kind: runtime::BudgetExhausted) -> FailureSignal {
    match kind {
        runtime::BudgetExhausted::Deadline => FailureSignal::Deadline,
        runtime::BudgetExhausted::Iterations
        | runtime::BudgetExhausted::ToolCalls
        | runtime::BudgetExhausted::VerificationTreadmill => FailureSignal::ToolBudgetExceeded,
        runtime::BudgetExhausted::OutputTokens | runtime::BudgetExhausted::InputTokens => {
            FailureSignal::TokenBudgetExceeded
        }
    }
}

/// Decide what escalation signal (if any) to apply on the *next* turn, given the
/// last failure seen and the new one. A new or different failure escalates; a
/// 2nd **consecutive identical** failure returns `None` so the host stops
/// climbing the ladder and lets the turn fail honestly instead of looping
/// escalate→fail→escalate forever (WI-B infinite-loop guard).
pub(crate) fn decide_escalation(
    last: Option<FailureSignal>,
    new_failure: FailureSignal,
) -> Option<FailureSignal> {
    if last == Some(new_failure) {
        None
    } else {
        Some(new_failure)
    }
}

/// Classify a turn's terminal error into an escalation-worthy [`FailureSignal`],
/// or `None` for failures escalation can't help (auth, network, validation) —
/// climbing the route ladder won't fix a missing API key (WI-B). Matched on the
/// stringified error, the only failure-kind signal the turn loop carries.
pub(crate) fn classify_turn_failure(error: &str) -> Option<FailureSignal> {
    let lower = error.to_ascii_lowercase();
    if lower.contains("test failed")
        || lower.contains("tests failed")
        || lower.contains("test failure")
        || lower.contains("red test")
    {
        Some(FailureSignal::RedTest)
    } else if lower.contains("tool call budget")
        || lower.contains("tool-call budget")
        || lower.contains("max_iterations")
        || lower.contains("max iterations")
        || lower.contains("iteration cap")
    {
        Some(FailureSignal::ToolBudgetExceeded)
    } else if lower.contains("deadline") || lower.contains("timed out") || lower.contains("timeout")
    {
        Some(FailureSignal::Deadline)
    } else if lower.contains("empty") || lower.contains("produced no content") {
        Some(FailureSignal::RepeatedEmptyResponse)
    } else {
        None
    }
}

/// The host's advisory routing hint for a turn.
#[derive(Debug, Clone)]
pub(crate) struct RouteHint {
    pub shape: RouteShape,
    pub confidence: f32,
    pub reasons: Vec<&'static str>,
    explicit_parallel: bool,
    ultracode: bool,
    semantic_triage: bool,
    /// Static note from a prior-turn failure escalation (WI-B). `reasons` is
    /// `Vec<&'static str>`, so the dynamic escalation cause rides here instead.
    prior_failure_note: Option<&'static str>,
    /// The true canonical (6-shape) routing verdict when the host shape is a
    /// lossy projection of it — e.g. smart-routing evidence classified the turn as
    /// `RepairLoop`/`ParallelLanes` but the host has no such shape and projects it
    /// onto `Pipeline`/`FanoutParallel`. `None` for keyword-classified turns,
    /// where `shape.to_route_shape_kind()` is already exact. Used only for audit
    /// fidelity; it does not change routing.
    canonical_override: Option<runtime::RouteShapeKind>,
}

impl RouteHint {
    /// Whether the host should pre-spawn a multi-agent pre-analysis *before* the
    /// model turn — which also runs the cheap intent triage that can route a turn
    /// to `diagnose`.
    ///
    /// - `FanoutParallel`: a host fast-path only for exact orchestration
    ///   directives or ultracode breadth turns. Natural-language collaboration
    ///   requests remain model/triage-led so no phrase table becomes the first
    ///   decision maker.
    /// - `Pipeline`/`DelegateOne`: keep the model-led route; non-breadth host
    ///   pre-analysis is reserved for ultracode only via `FanoutParallel` breadth
    ///   or explicit future gates, not generic High/Max effort.
    ///
    /// Every other case defers to the model so the host and model never both spawn
    /// for the same turn (split-brain, BUG-D2).
    pub(crate) fn should_host_prespawn(&self) -> bool {
        match self.shape {
            RouteShape::FanoutParallel => self.explicit_parallel || self.ultracode,
            RouteShape::Pipeline | RouteShape::DelegateOne | RouteShape::Solo => false,
        }
    }

    /// Whether to run cheap LLM semantic triage before the model turn without
    /// directly committing to host fan-out.
    ///
    /// Armed only on the explicit `ultracode` opt-in for the non-prespawn shapes
    /// (`Pipeline`/`DelegateOne`): triage runs `clarify_intent`, which can route a
    /// bug-shaped request to the adversarial `Diagnose` fan-out, then falls back to
    /// the model-led turn on any other verdict. Lower efforts keep this off, so an
    /// unconfigured user never gets a surprise pre-analysis and ordinary turns
    /// defer entirely to the model (which self-invokes `SpawnMultiAgent` when it
    /// needs collaborators). Suppressed whenever the host will already pre-spawn.
    pub(crate) fn should_run_semantic_triage(&self) -> bool {
        self.semantic_triage && !self.should_host_prespawn()
    }

    /// Whether this is the breadth fan-out shape (`FanoutParallel`), which fans
    /// out on any `decompose` triage. Non-breadth shapes stay model-led: their
    /// route reminder teaches the model when to call Agent/SpawnMultiAgent, but
    /// the host does not pre-consume the turn from a natural-language phrase.
    pub(crate) fn is_breadth(&self) -> bool {
        self.shape == RouteShape::FanoutParallel
    }

    /// The canonical (6-shape) routing verdict for the audit record: the
    /// `canonical_override` when the host shape is a lossy projection (e.g.
    /// evidence `RepairLoop`/`ParallelLanes`), otherwise the host shape's own
    /// canonical mapping. Behavior-neutral — affects only what `/audit` records.
    pub(crate) fn canonical_shape_kind(&self) -> runtime::RouteShapeKind {
        self.canonical_override.unwrap_or_else(|| self.shape.to_route_shape_kind())
    }

    /// A concise, user-facing one-line summary of the host's route decision for
    /// the turn, or `None` for `Solo` (handling a turn directly needs no banner).
    ///
    /// This is the **deterministic, host-emitted surfacing** of the routing
    /// choice (principle ①): it shows in the live TUI even when the model skips
    /// its own narration, closing the gap where the host classified every turn
    /// but the decision was only visible via a debug eprintln / an `/audit`
    /// count. It is deliberately labeled a *hint* because the shape is advisory —
    /// the model may escalate (e.g. an ultracode audit from one `Agent` to
    /// `SpawnMultiAgent`), and its own narration remains the authoritative
    /// decision. Shape names use the model's tool vocabulary so the host line and
    /// the model's narration read in the same terms.
    pub(crate) fn user_hint_line(&self, host_prespawn: bool) -> Option<String> {
        let tool = match self.shape {
            RouteShape::Solo => return None,
            RouteShape::DelegateOne => "Agent",
            RouteShape::FanoutParallel => "SpawnMultiAgent",
            RouteShape::Pipeline => "Workflow",
        };
        let spawn = if host_prespawn { " · pre-spawning" } else { "" };
        let reasons = self.reasons.join(", ");
        Some(format!("route hint · {tool}{spawn} — {reasons}"))
    }

    /// Strengthen this hint after a prior-turn failure (WI-B). One step up the
    /// ladder (`escalated`), a confidence bump, and a static cause note surfaced
    /// to the model. Returns `self` unchanged when there is no failure, or when
    /// the hint is already a host pre-spawn fast-path — escalation is advisory
    /// (model-led) and must never enable or disable host pre-spawn, preserving
    /// the no-split-brain invariant. Under ultracode `escalated` *can* yield a
    /// prespawn shape (`Solo`→`DelegateOne` both host-prespawn there), so the
    /// shape bump is reverted whenever it would newly flip prespawn on — the
    /// confidence bump and cause note still ride through as the advisory signal.
    #[must_use]
    pub(crate) fn escalate(mut self, prior_failure: Option<FailureSignal>) -> Self {
        let Some(failure) = prior_failure else {
            return self;
        };
        if self.should_host_prespawn() {
            return self;
        }
        let escalated = self.shape.escalated();
        let prior_shape = self.shape;
        self.shape = escalated;
        if self.should_host_prespawn() {
            // Escalation would newly enable the host pre-spawn (e.g. ultracode
            // Solo→DelegateOne): keep the model-led shape so escalation stays
            // purely advisory and never flips prespawn on (no split-brain).
            self.shape = prior_shape;
        }
        self.confidence = (self.confidence + 0.15).min(0.95);
        self.reasons.push("escalated after prior-turn failure");
        self.prior_failure_note = Some(failure.note());
        self
    }

    /// A one-line reminder surfaced to the model unless the host will actually
    /// *consume* the turn itself — i.e. a **breadth** pre-spawn, which fans out
    /// on a `decompose` verdict and carries the intent. A non-breadth pre-spawn
    /// (ultracode `Pipeline`/`DelegateOne`) runs the triage but engages the host
    /// only on a `diagnose` verdict; it usually falls back to the model-led turn,
    /// so it must keep its shape guidance. `None` for `Solo` (let the model just
    /// answer) and for a breadth host fast-path (the pre-spawned analysis already
    /// carries the intent).
    pub(crate) fn model_reminder(&self) -> Option<String> {
        if self.should_host_prespawn() && self.is_breadth() {
            return None;
        }
        self.shape_guidance_reminder()
    }

    /// A one-line reminder for entry points that **never** host pre-spawn
    /// (headless `-p`, `zo serve`): the model itself must choose the route,
    /// so unlike [`Self::model_reminder`] this also nudges the `FanoutParallel`
    /// shape instead of deferring it to a host fast-path that those paths do not
    /// have. `None` only for `Solo`. Shares the wording with `model_reminder`
    /// via [`Self::shape_guidance_reminder`] so the two never drift.
    pub(crate) fn model_led_reminder(&self) -> Option<String> {
        self.shape_guidance_reminder()
    }

    /// Shared body of the route reminders: the shape-specific guidance line, or
    /// `None` for `Solo`. The host-vs-model-led decision (whether a pre-spawn
    /// suppresses it) lives in the callers, so this single source of truth keeps
    /// `model_reminder` and `model_led_reminder` byte-identical in wording.
    fn shape_guidance_reminder(&self) -> Option<String> {
        if self.shape == RouteShape::Solo {
            return None;
        }
        let guidance = match self.shape {
            RouteShape::DelegateOne => {
                "consider delegating it to one focused `Agent` (pick a fitting subagent_type)"
            }
            RouteShape::FanoutParallel => {
                "consider splitting the independent slices across `SpawnMultiAgent` and synthesizing"
            }
            RouteShape::Pipeline => {
                "for dependent implementation spanning multiple files or subsystems, prefer a `Workflow`: plan→implement→verify; bounded one-file/module work should stay solo; make real changes and verify them, do not return analysis only"
            }
            RouteShape::Solo => unreachable!(),
        };
        let escalation = self.prior_failure_note.map_or(String::new(), |note| {
            format!(" Note: {note}, so a stronger route is advised this turn.")
        });
        Some(format!(
            "{ROUTE_HINT_REMINDER_PREFIX} Route hint: {}. {guidance}. Advisory only; small/direct work can stay solo.{escalation}",
            self.shape.as_str(),
        ))
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn implementation_target_from_token(token: &str) -> Option<String> {
    let candidate = token.trim_matches(|ch: char| {
        matches!(
            ch,
            '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';'
        )
    });
    let (stem, suffix) = candidate.rsplit_once('.')?;
    if stem.is_empty() || stem.contains("://") {
        return None;
    }
    // Korean case particles can immediately follow an ASCII extension
    // (`rate_limiter.py를`). Stop at the first non-ASCII-alphanumeric byte so
    // the structural filename is still recognized without language keywords.
    let extension = suffix
        .chars()
        .take_while(char::is_ascii_alphanumeric)
        .collect::<String>();
    IMPLEMENTATION_FILE_EXTENSIONS
        .iter()
        .any(|known| extension.eq_ignore_ascii_case(known))
        .then(|| format!("{stem}.{}", extension.to_ascii_lowercase()))
}

fn names_one_implementation_target(input: &str) -> bool {
    input
        .split_whitespace()
        .filter_map(implementation_target_from_token)
        .collect::<BTreeSet<_>>()
        .len()
        == 1
}

fn is_explicit_parallel_directive(input: &str) -> bool {
    let trimmed = input
        .trim_start_matches(|ch: char| !ch.is_alphanumeric())
        .trim();
    if trimmed.is_empty() || starts_with_negation(trimmed) {
        return false;
    }

    EXPLICIT_PARALLEL_TARGETS
        .iter()
        .any(|target| target_is_directive(trimmed, target))
}

fn starts_with_negation(input: &str) -> bool {
    ["do not ", "don't ", "dont ", "no ", "never "]
        .iter()
        .any(|prefix| input.starts_with(prefix))
}

fn target_is_directive(input: &str, target: &str) -> bool {
    EXPLICIT_PARALLEL_VERBS.iter().any(|verb| {
        input
            .strip_prefix(verb)
            .and_then(|rest| rest.trim_start().strip_prefix(target))
            .is_some_and(|tail| directive_tail(Some(tail)))
    })
}

fn directive_tail(tail: Option<&str>) -> bool {
    match tail.map(str::trim_start) {
        Some("") => true,
        Some(rest) => {
            rest.starts_with(':')
                || rest.starts_with(';')
                || rest.starts_with(',')
                || rest.starts_with("for ")
                || rest.starts_with("on ")
                || rest.starts_with("to ")
                || rest.starts_with("with ")
        }
        None => false,
    }
}

/// Build the host's advisory [`RouteHint`] for a turn. Model-free and cheap.
///
/// Routing is **model-driven**: keyword classification (implement / breadth /
/// default) only shapes the advisory reminder and never arms host triage, so the
/// host stops auto-firing fan-out on ordinary turns regardless of language or
/// model. The host pre-spawns a fan-out only for an explicit orchestration
/// directive (`is_explicit_parallel_directive`); for anything else the turn goes
/// straight to the model, which calls `SpawnMultiAgent` itself when it needs
/// collaborators.
///
/// `effort == None` (the user never set a level) is treated conservatively —
/// like a low signal — so an unconfigured user never gets a surprise fan-out
/// (BUG-R2). `Off`/`Low` force `Solo`.
pub(crate) fn build_route_hint(
    user_text: &str,
    effort: Option<Effort>,
    context_tokens: usize,
) -> RouteHint {
    let lower = user_text.to_lowercase();
    let ultracode = effort == Some(Effort::Smart);
    let low_effort = matches!(effort, Some(Effort::Off | Effort::Low));

    let solo = |confidence: f32, reason: &'static str, semantic_triage: bool| RouteHint {
        shape: RouteShape::Solo,
        confidence,
        reasons: vec![reason],
        explicit_parallel: false,
        ultracode,
        semantic_triage,
        prior_failure_note: None,
        canonical_override: None,
    };

    // Off/Low effort, or an explicit "don't delegate", forces solo.
    if low_effort {
        return solo(0.9, "effort opted out of delegation", false);
    }
    if contains_any(&lower, EXPLICIT_NO_DELEGATION) {
        return solo(0.95, "explicit no-delegation request", false);
    }
    // Exact tool/API directives are explicit host orchestration requests.
    // Natural-language collaboration/parallelism stays semantic and model-led.
    if is_explicit_parallel_directive(&lower) {
        return RouteHint {
            shape: RouteShape::FanoutParallel,
            confidence: 0.95,
            reasons: vec!["explicit parallel orchestration directive"],
            explicit_parallel: true,
            ultracode,
            semantic_triage: false,
            prior_failure_note: None,
            canonical_override: None,
        };
    }

    let small = contains_any(&lower, SMALL_SINGLE_STEP_INTENT);
    let implement = contains_any(&lower, IMPLEMENT_INTENT);
    let verify = contains_any(&lower, VERIFY_INTENT);
    let diagnose_shaped = contains_any(&lower, DIAGNOSE_TRIAGE_INTENT);
    let breadth = contains_any(&lower, BREADTH_INTENT);
    let multi_scope = contains_any(&lower, MULTI_SCOPE_INTENT);
    let large_ctx = context_tokens >= LARGE_CONTEXT_TOKENS;
    let bounded_implementation = !multi_scope
        && !diagnose_shaped
        && (contains_any(&lower, BOUNDED_IMPLEMENTATION_SCOPE)
            || names_one_implementation_target(&lower));

    // A clearly small, local edit with no implementation breadth stays solo.
    if small && !implement && !multi_scope {
        return solo(0.85, "small single-step edit", false);
    }

    // A spec or test command does not turn one named implementation target
    // into a dependent workflow. Keep this ahead of the generic implementation
    // branch so the host reminder agrees with the base delegation rubric.
    if implement && bounded_implementation {
        return solo(0.85, "bounded single-target implementation", false);
    }

    // Unbounded implementation/change intent → pipeline
    // (plan→implement→verify). This is exactly the case the old gate
    // mis-handled: it scored such turns as fan-out candidates but returned
    // analysis only (BUG-D1).
    if implement {
        let mut confidence: f32 = 0.6;
        let mut reasons = vec!["implementation intent"];
        if verify {
            confidence += 0.2;
            reasons.push("plus verification");
        }
        if multi_scope || large_ctx {
            confidence += 0.1;
            reasons.push("multi-step / broad");
        }
        return RouteHint {
            shape: RouteShape::Pipeline,
            confidence: confidence.min(0.95),
            reasons,
            explicit_parallel: false,
            ultracode,
            // Ultracode-only: implementation turns with a strong bug/root-cause
            // signal arm cheap LLM intent triage. Generic improvement/doc/update
            // requests deliberately skip triage so a spurious `diagnose` verdict
            // cannot consume their plan→implement→verify Pipeline as a BUG fan-out.
            semantic_triage: ultracode && diagnose_shaped,
            prior_failure_note: None,
            canonical_override: None,
        };
    }

    // Breadth-first analysis/search. Natural language stays model-led in every
    // language: the hint surfaces the shape — one specialist for a single area,
    // independent slices for a multi-scope ask — but only an exact orchestration
    // directive above directly host-spawns a fan-out.
    if breadth {
        return breadth_route_hint(ultracode, multi_scope, large_ctx);
    }

    // No multilingual keyword matched. Let the smart-routing need planner drive
    // the decision instead of blindly defaulting to Solo.
    if let Some(hint) = evidence_driven_fallthrough(user_text, ultracode) {
        return hint;
    }
    solo(0.6, "no delegation signal", false)
}

/// The breadth arm of [`build_route_hint`], split out purely for function
/// length: one focused specialist for a single-area investigation, a model-led
/// fan-out nudge when the ask names several areas.
fn breadth_route_hint(ultracode: bool, multi_scope: bool, large_ctx: bool) -> RouteHint {
    let mut confidence: f32 = 0.6;
    if ultracode {
        confidence += 0.1;
    }
    if multi_scope {
        confidence += 0.1;
    }
    if large_ctx {
        confidence += 0.1;
    }
    let confidence = confidence.min(0.95);
    if multi_scope {
        let mut reasons = vec!["multi-scope breadth investigation"];
        if large_ctx {
            reasons.push("large context");
        }
        // Several areas in one breadth ask → surface the fan-out shape as a
        // MODEL-LED nudge (CC-style: the model judges task shape and spawns
        // its own agents from the delegation rubric). The hint's `ultracode`
        // is forced off — mirroring `evidence_driven_fallthrough` — so
        // `should_host_prespawn` stays false and the host never pre-consumes
        // a turn from a natural-language phrase (BUG-D2 stays closed). Under
        // the real ultracode opt-in, cheap LLM triage still gates any host
        // fan-out (`decompose`/`Diagnose`) exactly as before.
        return RouteHint {
            shape: RouteShape::FanoutParallel,
            confidence,
            reasons,
            explicit_parallel: false,
            ultracode: false,
            semantic_triage: ultracode,
            prior_failure_note: None,
            canonical_override: None,
        };
    }
    let mut reasons = vec!["deep single-area investigation"];
    if ultracode {
        reasons.push("ultracode breadth investigation");
    }
    if large_ctx {
        reasons.push("large context");
    }
    RouteHint {
        shape: RouteShape::DelegateOne,
        confidence,
        reasons,
        explicit_parallel: false,
        ultracode,
        // Ultracode-only: a breadth investigation under the explicit ultracode
        // opt-in arms cheap LLM intent triage. The shape stays DelegateOne (no
        // host-prespawn), so this only lets `clarify_intent` route a bug-shaped
        // request to the adversarial `Diagnose` fan-out; otherwise it falls
        // back to the model-led turn. Lower efforts stay fully model-led.
        semantic_triage: ultracode,
        prior_failure_note: None,
        canonical_override: None,
    }
}

/// Smart-routing fallthrough: when the host's (multilingual) keyword classifier
/// finds no delegation signal, ask the need planner whether agents add value and
/// drive the shape from that evidence. This is the seam where smart routing
/// actually *drives* host orchestration rather than only per-agent model
/// selection. `None` keeps the host's Solo default. Safety:
/// - `ParallelLanes` evidence surfaces as a **model-led** `FanoutParallel`
///   (breadth nudge the model can act on), but the host still NEVER pre-spawns
///   from evidence alone (anti-self-DoS): for that hint both `explicit_parallel`
///   and `ultracode` are forced off, so `should_host_prespawn` stays false even
///   under ultracode. Other shapes map to `DelegateOne`/`Pipeline`, which never
///   host-prespawn either;
/// - empty evidence (e.g. a turn the planner cannot classify) keeps Solo, so the
///   confident multilingual routing above is never regressed;
/// - the planner's true 6-shape verdict is carried in `canonical_override` so the
///   audit record stays faithful even though the host shape is a projection.
fn evidence_driven_fallthrough(user_text: &str, ultracode: bool) -> Option<RouteHint> {
    let evidence = tools::assess_turn_orchestration(user_text);
    if evidence.need_count == 0 {
        return None;
    }
    // `hint_ultracode` gates host pre-spawn (and semantic triage) for the hint.
    // For the breadth (`FanoutParallel`) projection it is forced off so evidence
    // can never make the host spawn; for the others it carries the real value so
    // an ultracode evidence turn can still arm `Diagnose` triage as before.
    let (shape, hint_ultracode) = match evidence.shape {
        runtime::RouteShapeKind::Solo => return None,
        runtime::RouteShapeKind::OneSpecialist => (RouteShape::DelegateOne, ultracode),
        runtime::RouteShapeKind::ParallelLanes => (RouteShape::FanoutParallel, false),
        // SequentialWorkflow / RepairLoop / ParallelRepairLoop → structured pipeline.
        _ => (RouteShape::Pipeline, ultracode),
    };
    Some(RouteHint {
        shape,
        confidence: 0.6,
        reasons: vec!["smart-routing evidence detected agent need"],
        explicit_parallel: false,
        ultracode: hint_ultracode,
        semantic_triage: hint_ultracode,
        prior_failure_note: None,
        canonical_override: Some(evidence.shape),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_edit_is_solo() {
        let hint = build_route_hint(
            "이 함수 이름만 foo에서 bar로 바꿔줘",
            Some(Effort::Medium),
            500,
        );
        assert_eq!(hint.shape, RouteShape::Solo);
        assert!(!hint.should_host_prespawn());
    }

    #[test]
    fn rename_symbol_stays_solo_even_in_ultracode() {
        let hint = build_route_hint("rename this symbol", Some(Effort::Smart), 500);
        assert_eq!(hint.shape, RouteShape::Solo);
        assert!(!hint.should_host_prespawn());
    }

    #[test]
    fn simple_question_is_solo_even_under_smart_effort() {
        // A simple question under Smart effort reads Solo: the host never
        // pre-spawns for it, and the model's own delegation rubric (base
        // prompt + tool descriptions) is what keeps the response direct.
        let simple = build_route_hint("이 설정값이 뭔지 알려줘", Some(Effort::Smart), 500);
        assert_eq!(simple.shape, RouteShape::Solo);
        let broad = build_route_hint(
            "이 기능 구현하고 테스트까지 통과시켜줘",
            Some(Effort::Smart),
            1_000,
        );
        assert_ne!(broad.shape, RouteShape::Solo);
    }

    #[test]
    fn implementation_is_pipeline() {
        let hint = build_route_hint(
            "이 기능 구현하고 테스트까지 통과시켜줘",
            Some(Effort::High),
            1_000,
        );
        assert_eq!(hint.shape, RouteShape::Pipeline);
        assert!(
            !hint.should_host_prespawn(),
            "implementation remains model-led; the model decides whether to call agents"
        );
        // Model-driven: a keyword match shapes the advisory reminder but never
        // arms host triage on an ordinary turn.
        assert!(!hint.should_run_semantic_triage());
    }

    #[test]
    fn bounded_single_target_implementation_stays_solo() {
        // Regression from the 2026-07-14 T2 benchmark: reading a spec and
        // verifying the result does not make a one-file implementation a
        // dependent multi-stage workflow. `tests/` is a verification boundary,
        // not a second implementation scope.
        for prompt in [
            "SPEC.md를 읽고 rate_limiter.py를 새로 구현해줘. tests/ 디렉토리는 절대 수정 금지. python3 -m unittest discover -s tests가 전부 통과해야 완료야. 완료 전에 직접 테스트를 실행해서 확인해.",
            "Read the spec, implement src/parser.rs, and run cargo test before finishing.",
            "Implement this in a single module and verify it.",
        ] {
            let hint = build_route_hint(prompt, Some(Effort::Smart), 1_000);

            assert_eq!(hint.shape, RouteShape::Solo, "prompt: {prompt}");
            assert!(
                hint.model_led_reminder().is_none(),
                "bounded one-file work must not receive a Workflow reminder: {prompt}"
            );
        }
    }

    #[test]
    fn dependent_multi_target_implementation_keeps_pipeline() {
        let prompt =
            "Implement src/parser.rs and src/lexer.rs, then update tests/parser_test.rs and verify.";
        let hint = build_route_hint(prompt, Some(Effort::Smart), 1_000);

        assert_eq!(hint.shape, RouteShape::Pipeline, "prompt: {prompt}");
        let reminder = hint
            .model_led_reminder()
            .expect("dependent multi-file work needs Workflow guidance");
        assert!(reminder.contains("multiple files or subsystems"));
        assert!(reminder.contains("bounded one-file/module work should stay solo"));
    }

    #[test]
    fn ultracode_general_improvement_and_documentation_are_pipeline() {
        // Regression: generic "improve/document" requests in ultracode looked
        // like ordinary text and fell through to Solo, despite being codebase
        // changes that should get plan→implement→verify structure.
        for prompt in [
            "ultracode multiagent claude agent 개선하고 ultracode logic 문서화",
            "improve multiagent Claude-agent routing and document ultracode logic",
            "다음 개선 진행 완벽하게 적대적 검증까지",
            "개선하고 검증까지 해줘",
        ] {
            let hint = build_route_hint(prompt, Some(Effort::Smart), 2_000);
            assert_eq!(hint.shape, RouteShape::Pipeline, "prompt: {prompt}");
            assert!(
                !hint.should_host_prespawn(),
                "generic implementation remains model-led: {prompt}"
            );
            assert!(!hint.is_breadth(), "prompt: {prompt}");
            assert!(
                !hint.should_run_semantic_triage(),
                "generic Pipeline improvements must not let a spurious diagnose triage consume the turn: {prompt}"
            );
        }
    }

    #[test]
    fn migration_plan_apply_verify_is_pipeline() {
        let hint = build_route_hint(
            "마이그레이션 계획 세우고 적용하고 검증해줘",
            Some(Effort::High),
            2_000,
        );
        assert_eq!(hint.shape, RouteShape::Pipeline);
    }

    #[test]
    fn multi_scope_breadth_hints_model_led_fanout_and_never_host_prespawns() {
        let prompt = "runtime/tools/api 각각에서 timeout 취약점 찾아줘";
        // Multi-scope breadth surfaces the fan-out shape as a model-led nudge at
        // every delegation-capable effort — the model judges task shape and
        // spawns its own agents (CC-style) — but the host never pre-spawns from
        // a natural-language phrase.
        let high = build_route_hint(prompt, Some(Effort::High), 1_000);
        assert_eq!(high.shape, RouteShape::FanoutParallel);
        assert!(!high.should_host_prespawn());
        assert!(!high.should_run_semantic_triage());
        let uc = build_route_hint(prompt, Some(Effort::Smart), 1_000);
        assert_eq!(uc.shape, RouteShape::FanoutParallel);
        assert!(!uc.should_host_prespawn());
        // Ultracode still arms the cheap LLM intent triage — the only gate that
        // can turn this hint into a host fan-out (`decompose`) or `Diagnose`.
        assert!(uc.should_run_semantic_triage());

        // Single-area breadth keeps the one-specialist hint.
        let single =
            build_route_hint("auth 모듈의 timeout 취약점 찾아줘", Some(Effort::High), 1_000);
        assert_eq!(single.shape, RouteShape::DelegateOne);
        assert!(!single.should_host_prespawn());
    }

    #[test]
    fn user_hint_line_surfaces_non_solo_shapes_in_tool_vocab() {
        // Solo handling needs no banner — the gap-closing surface is for
        // delegation-shaped turns only, so trivial chat stays clean.
        let solo = build_route_hint("hi", Some(Effort::High), 0);
        assert_eq!(solo.shape, RouteShape::Solo);
        assert!(solo.user_hint_line(false).is_none());

        // DelegateOne → the model's `Agent` vocabulary.
        let one = build_route_hint("이 failing test 하나 원인 찾아줘", Some(Effort::High), 1_000);
        assert_eq!(one.shape, RouteShape::DelegateOne);
        let line = one.user_hint_line(false).expect("a non-solo turn surfaces a line");
        assert!(line.starts_with("route hint · Agent —"), "{line}");

        // Pipeline → `Workflow`.
        let pipe = build_route_hint("이거 구현하고 테스트 통과시켜줘", Some(Effort::High), 1_000);
        assert_eq!(pipe.shape, RouteShape::Pipeline);
        assert!(pipe.user_hint_line(false).unwrap().contains("Workflow"));

        // FanoutParallel with a host pre-spawn → `SpawnMultiAgent` + the
        // pre-spawning note so the user sees the host actually consumed the turn.
        let fan = build_route_hint("Use SpawnMultiAgent for this", Some(Effort::High), 0);
        assert_eq!(fan.shape, RouteShape::FanoutParallel);
        let fan_line = fan.user_hint_line(true).expect("non-solo surfaces a line");
        assert!(fan_line.contains("SpawnMultiAgent"), "{fan_line}");
        assert!(fan_line.contains("pre-spawning"), "{fan_line}");
    }

    #[test]
    fn smart_routing_evidence_drives_shape_when_host_keywords_miss() {
        // "harden the auth token handling" matches none of the host's (multilingual)
        // intent keyword lists, so the keyword classifier alone would default to
        // Solo. The smart-routing need planner sees the auth/token risk and plans a
        // reviewer, so the evidence now *drives* a non-Solo shape — orchestration
        // influenced by smart routing, not just model selection.
        let hint = build_route_hint("harden the auth token handling", Some(Effort::High), 1_000);
        assert_ne!(hint.shape, RouteShape::Solo, "evidence should drive a delegation shape");
        assert!(
            hint.reasons.iter().any(|reason| reason.contains("smart-routing evidence")),
            "{:?}",
            hint.reasons,
        );

        // A turn with no host keyword AND no smart-routing need stays Solo — the
        // evidence path only adds capability, never forces delegation.
        let plain = build_route_hint("hello there", Some(Effort::High), 0);
        assert_eq!(plain.shape, RouteShape::Solo);
    }

    #[test]
    fn canonical_override_is_used_for_the_audit_shape() {
        use runtime::RouteShapeKind;
        // A host shape that is a lossy projection (Pipeline) of a richer planner
        // verdict (RepairLoop) reports the true 6-shape for the audit.
        let with_override = RouteHint {
            shape: RouteShape::Pipeline,
            confidence: 0.6,
            reasons: vec!["x"],
            explicit_parallel: false,
            ultracode: false,
            semantic_triage: false,
            prior_failure_note: None,
            canonical_override: Some(RouteShapeKind::RepairLoop),
        };
        assert_eq!(with_override.canonical_shape_kind(), RouteShapeKind::RepairLoop);
        // Without an override it falls back to the host shape's own mapping.
        let plain = RouteHint {
            canonical_override: None,
            ..with_override.clone()
        };
        assert_eq!(plain.canonical_shape_kind(), RouteShapeKind::SequentialWorkflow);
    }

    #[test]
    fn evidence_parallel_lanes_is_model_led_fanout_and_never_prespawns() {
        // Call the fallthrough directly (bypassing the host keyword gate) with a
        // prompt the planner reads as separable parallel lanes. ParallelLanes
        // surfaces as a model-led FanoutParallel breadth nudge — but the host must
        // NEVER pre-spawn from evidence alone, even under ultracode (anti-self-DoS).
        // Three slash-separated lanes drive independent_lanes >= 2 → the planner's
        // natural shape is ParallelLanes (the "parallel" keyword alone does not).
        let hint = evidence_driven_fallthrough(
            "review the backend/frontend/database security layers",
            true,
        )
        .expect("evidence should detect a need for this prompt");
        assert_eq!(hint.shape, RouteShape::FanoutParallel);
        assert!(!hint.should_host_prespawn(), "evidence must never host-prespawn, even ultracode");
        assert!(!hint.should_run_semantic_triage());
        assert!(hint.is_breadth());
        // Audit fidelity: the true 6-shape is preserved despite the projection.
        assert_eq!(hint.canonical_shape_kind(), runtime::RouteShapeKind::ParallelLanes);
    }

    #[test]
    fn host_shapes_map_to_canonical_model_routing_taxonomy() {
        // The host's 4-shape pre-spawn taxonomy shares one canonical vocabulary
        // with the model router's 6-shape RouteShapeKind, so the recorded route
        // decision is expressed consistently across both layers.
        use runtime::RouteShapeKind;
        assert_eq!(RouteShape::Solo.to_route_shape_kind(), RouteShapeKind::Solo);
        assert_eq!(RouteShape::DelegateOne.to_route_shape_kind(), RouteShapeKind::OneSpecialist);
        assert_eq!(RouteShape::FanoutParallel.to_route_shape_kind(), RouteShapeKind::ParallelLanes);
        assert_eq!(RouteShape::Pipeline.to_route_shape_kind(), RouteShapeKind::SequentialWorkflow);
    }

    #[test]
    fn single_area_investigation_is_delegate_one() {
        let hint = build_route_hint(
            "이 failing test 하나 원인 찾아줘",
            Some(Effort::High),
            1_000,
        );
        assert_eq!(hint.shape, RouteShape::DelegateOne);
        assert!(!hint.should_host_prespawn());
    }

    #[test]
    fn exact_parallel_orchestration_directive_prespawns() {
        let hint = build_route_hint("Use SpawnMultiAgent for this", Some(Effort::High), 0);
        assert_eq!(hint.shape, RouteShape::FanoutParallel);
        assert!(hint.should_host_prespawn());
    }

    #[test]
    fn tool_mentions_and_negations_do_not_directly_prespawn() {
        for prompt in [
            "fix the SpawnMultiAgent bug",
            "review why SpawnMultiAgent hangs",
            "SpawnMultiAgent",
            "spawn_multi_agent",
            "workflow fanout",
            "SpawnMultiAgent: hangs when agents finish",
            "SpawnMultiAgent, bug report: dropped result",
            "Workflow fanout: broken on retry",
            "do not spawn multi agent workers",
            "never use SpawnMultiAgent here",
        ] {
            let hint = build_route_hint(prompt, Some(Effort::High), 0);
            assert_ne!(
                hint.shape,
                RouteShape::FanoutParallel,
                "ordinary mention or negation must not direct-spawn: {prompt}"
            );
            assert!(!hint.should_host_prespawn(), "prompt: {prompt}");
        }
    }

    #[test]
    fn natural_parallel_language_does_not_directly_prespawn() {
        for prompt in [
            "use parallel agents for this",
            "여러 에이전트 관점으로 검토해",
            "병렬로 조사해",
        ] {
            let hint = build_route_hint(prompt, Some(Effort::High), 0);
            assert_ne!(
                hint.shape,
                RouteShape::FanoutParallel,
                "natural language must not directly force fan-out: {prompt}"
            );
            assert!(
                !hint.should_host_prespawn(),
                "natural language stays model-led: {prompt}"
            );
            assert!(
                !hint.should_run_semantic_triage(),
                "model-driven: natural-language parallelism defers to the model, no host triage: {prompt}"
            );
        }
    }

    #[test]
    fn high_effort_pipeline_stays_model_led_without_phrase_table() {
        let prompt = "adversarially verify the fix, repair functional mismatches, and use the right collaboration strategy from context";

        let high = build_route_hint(prompt, Some(Effort::High), 0);
        assert_eq!(high.shape, RouteShape::Pipeline);
        assert!(
            !high.explicit_parallel,
            "collaboration strategy must be decided by model triage, not a phrase table"
        );
        assert!(
            !high.should_host_prespawn(),
            "high effort must not host-prespawn from natural-language collaboration hints"
        );

        let ultracode = build_route_hint(prompt, Some(Effort::Smart), 0);
        assert_eq!(ultracode.shape, RouteShape::Pipeline);
        assert!(
            !ultracode.explicit_parallel,
            "do not preserve one wording as a hardcoded explicit-parallel token"
        );
        assert!(
            !ultracode.should_host_prespawn(),
            "ultracode implementation stays model-led unless it is breadth or an exact orchestration directive"
        );
        assert!(
            !ultracode.is_breadth(),
            "fix/verify intent remains a plan→implement→verify pipeline unless triage dynamically diagnoses otherwise"
        );
    }

    #[test]
    fn explicit_no_delegation_is_solo() {
        let hint = build_route_hint("에이전트 쓰지 말고 답해", Some(Effort::High), 50_000);
        assert_eq!(hint.shape, RouteShape::Solo);
        assert!(!hint.should_host_prespawn());
    }

    #[test]
    fn low_effort_opts_out_even_when_heavy() {
        let hint = build_route_hint("please analyze the codebase", Some(Effort::Low), 50_000);
        assert_eq!(hint.shape, RouteShape::Solo);
    }

    #[test]
    fn unset_effort_is_conservative() {
        // BUG-R2: a user who never set an effort level must not get a surprise
        // host fan-out — only explicit parallelism pre-spawns for them.
        let analysis = build_route_hint("analyze the whole project", None, 20_000);
        assert!(
            !analysis.should_host_prespawn(),
            "unset effort must not host-prespawn from heuristics alone"
        );
        let explicit = build_route_hint("Use SpawnMultiAgent", None, 0);
        assert!(explicit.should_host_prespawn());
    }

    #[test]
    fn route_matrix_pins_pipeline_fanout_and_no_delegation_boundaries() {
        // One table covers the routing boundaries that must stay model-agnostic:
        // explicit host fan-out, ultracode fast-path, model-led analysis,
        // implementation pipelines, and explicit no-delegation.
        let cases = [
            (
                "Use SpawnMultiAgent for this",
                Some(Effort::High),
                0,
                RouteShape::FanoutParallel,
                true,
            ),
            (
                "프로젝트 전체 분석해줘",
                Some(Effort::Smart),
                20_000,
                RouteShape::FanoutParallel,
                false,
            ),
            (
                "프로젝트 전체 분석해줘",
                Some(Effort::Max),
                20_000,
                RouteShape::FanoutParallel,
                false,
            ),
            (
                "p1~p5 까지 구현하고 검증해",
                Some(Effort::Smart),
                20_000,
                RouteShape::Pipeline,
                false,
            ),
            (
                "에이전트 쓰지 말고 전체 분석해줘",
                Some(Effort::Smart),
                20_000,
                RouteShape::Solo,
                false,
            ),
        ];

        for (prompt, effort, tokens, shape, prespawn) in cases {
            let hint = build_route_hint(prompt, effort, tokens);
            assert_eq!(hint.shape, shape, "prompt: {prompt}");
            assert_eq!(hint.should_host_prespawn(), prespawn, "prompt: {prompt}");
            // The model nudge is suppressed only when the host will CONSUME the
            // turn — a breadth pre-spawn. Model-led Pipeline/DelegateOne keeps
            // its shape reminder.
            if prespawn && hint.is_breadth() {
                assert!(
                    hint.model_reminder().is_none(),
                    "a breadth host pre-spawn must not also nudge the model to spawn: {prompt}"
                );
            }
            if !prespawn && shape != RouteShape::Solo {
                assert!(
                    hint.model_reminder().is_some(),
                    "a model-led non-solo route must keep its shape reminder: {prompt}"
                );
            }
        }
    }

    #[test]
    fn no_split_brain_every_fanout_is_host_prespawned() {
        // A host-prespawned fan-out (exact orchestration directive) does NOT
        // also nudge the model — so host and model never both spawn for the
        // same turn (BUG-D2).
        let uc = build_route_hint(
            "Use SpawnMultiAgent for the whole project",
            Some(Effort::Smart),
            20_000,
        );
        assert_eq!(uc.shape, RouteShape::FanoutParallel);
        assert!(uc.should_host_prespawn());
        assert!(
            uc.model_reminder().is_none(),
            "a host-prespawned fan-out must not also nudge the model to spawn"
        );
        // Natural-language multi-scope breadth is the inverse split: the fan-out
        // shape is surfaced as a model-led nudge and the host never pre-spawns,
        // so again exactly one side owns the spawning decision.
        let high = build_route_hint("analyze the whole project", Some(Effort::Max), 20_000);
        assert_eq!(high.shape, RouteShape::FanoutParallel);
        assert!(!high.should_host_prespawn());
        assert!(high.model_reminder().is_some());
    }

    #[test]
    fn ultracode_broad_natural_language_is_model_led_not_direct_fanout() {
        let hint = build_route_hint("review this", Some(Effort::Smart), 500);
        assert_eq!(hint.shape, RouteShape::DelegateOne);
        assert!(!hint.should_host_prespawn());
        // Ultracode arms cheap LLM triage (no direct host fan-out); the DelegateOne
        // reminder still lets the model decide whether to delegate on fallback.
        assert!(hint.should_run_semantic_triage());
        assert!(hint.model_reminder().is_some());
    }

    #[test]
    fn ultracode_bug_fix_is_pipeline_model_led_unless_breadth_or_directive() {
        let uc = build_route_hint("이 스트리밍 버그 고쳐줘", Some(Effort::Smart), 0);
        assert_eq!(uc.shape, RouteShape::Pipeline);
        assert!(
            !uc.should_host_prespawn(),
            "natural-language bug fixes stay model-led; the model may still call Agent/SpawnMultiAgent from the reminder"
        );
        assert!(uc.model_reminder().is_some());
    }

    #[test]
    fn below_ultracode_a_bug_fix_stays_model_led() {
        // Only ultracode auto-engages the pre-analysis; below it a fix is an
        // ordinary advisory Pipeline so an unconfigured user never gets a surprise
        // swarm on every bug report.
        let med = build_route_hint("이 스트리밍 버그 고쳐줘", Some(Effort::Medium), 0);
        assert_eq!(med.shape, RouteShape::Pipeline);
        assert!(!med.should_host_prespawn());
    }

    #[test]
    fn ultracode_bug_shaped_turn_arms_triage_but_lower_effort_does_not() {
        // The distinctive harness (cheap intent triage → adversarial `Diagnose`
        // fan-out) was dormant because every route hard-coded semantic_triage:false.
        // Under the explicit ultracode opt-in a bug-shaped turn must now ARM the
        // triage gate (`should_run_semantic_triage`) — the single gate that lets
        // `AutoFanoutPlan::from_hint` build a triage prelude, whose `clarify_intent`
        // can route the request to `FanoutMode::Diagnose`. Crucially it stays
        // model-led: triage is not a host-prespawn fan-out, it falls back to the
        // single turn unless the LLM diagnoses a bug.
        for bug_prompt in [
            "이 스트리밍 버그 고쳐줘",
            "fix the streaming bug that reproduces on every turn",
            "debug why the parser drops the last token",
        ] {
            let uc = build_route_hint(bug_prompt, Some(Effort::Smart), 0);
            assert_eq!(uc.shape, RouteShape::Pipeline, "prompt: {bug_prompt}");
            assert!(
                uc.should_run_semantic_triage(),
                "ultracode bug-shaped turn must arm triage so it can reach Diagnose: {bug_prompt}"
            );
            assert!(
                !uc.should_host_prespawn(),
                "armed triage is model-led, never a host-prespawn fan-out: {bug_prompt}"
            );

            // The same request below ultracode stays fully model-led: no host
            // triage, so an unconfigured user never gets a surprise pre-analysis.
            for effort in [Effort::Medium, Effort::High, Effort::Max] {
                let lower = build_route_hint(bug_prompt, Some(effort), 0);
                assert!(
                    !lower.should_run_semantic_triage(),
                    "{effort:?} must not arm host triage: {bug_prompt}"
                );
                assert!(!lower.should_host_prespawn(), "prompt: {bug_prompt}");
            }
        }
    }

    #[test]
    fn model_led_reminder_nudges_fanout_that_host_prespawn_would_suppress() {
        // Gap B: headless `-p` / serve never host pre-spawn, so they use
        // `model_led_reminder`, which nudges EVERY non-Solo shape — including the
        // ultracode FanoutParallel case where the host-aware `model_reminder`
        // returns None (because the TUI would host pre-spawn instead).
        let uc = build_route_hint(
            "Use SpawnMultiAgent for the whole project",
            Some(Effort::Smart),
            20_000,
        );
        assert_eq!(uc.shape, RouteShape::FanoutParallel);
        assert!(uc.should_host_prespawn());
        assert!(
            uc.model_reminder().is_none(),
            "host-aware reminder defers a prespawned fan-out to the host"
        );
        let led = uc
            .model_led_reminder()
            .expect("model-led path must still nudge the fan-out shape");
        assert!(led.contains(ROUTE_HINT_REMINDER_PREFIX));
        assert!(
            led.contains("SpawnMultiAgent"),
            "nudges the fan-out tool: {led}"
        );

        // Solo stays None on both paths (just answer).
        let solo = build_route_hint("rename this symbol", Some(Effort::Smart), 500);
        assert_eq!(solo.shape, RouteShape::Solo);
        assert!(solo.model_reminder().is_none());
        assert!(solo.model_led_reminder().is_none());

        // When the host would NOT pre-spawn (DelegateOne), the two paths agree
        // verbatim — `model_led_reminder` shares the wording, never drifts.
        let one = build_route_hint(
            "이 failing test 하나 원인 찾아줘",
            Some(Effort::High),
            1_000,
        );
        assert_eq!(one.shape, RouteShape::DelegateOne);
        assert!(!one.should_host_prespawn());
        assert_eq!(one.model_reminder(), one.model_led_reminder());
    }

    #[test]
    fn route_hint_reminders_keep_pipeline_contract_without_leaky_framing() {
        let hint = build_route_hint(
            "p1~p5 까지 구현하고 검증해",
            Some(Effort::High),
            20_000,
        );
        assert_eq!(hint.shape, RouteShape::Pipeline);
        assert!(!hint.should_host_prespawn());

        for reminder in [
            hint.model_reminder()
                .expect("model path must receive pipeline guidance"),
            hint.model_led_reminder()
                .expect("model-led path must receive pipeline guidance"),
        ] {
            assert!(reminder.contains(ROUTE_HINT_REMINDER_PREFIX));
            assert!(reminder.contains("Route hint: Pipeline"));
            assert!(reminder.contains("prefer a `Workflow`"));
            assert!(reminder.contains("plan→implement→verify"));
            assert!(reminder.contains("make real changes"));
            assert!(reminder.contains("verify them"));
            assert!(reminder.contains("do not return analysis only"));

            let lower = reminder.to_lowercase();
            for banned in [
                "this turn looks like",
                "this looks like implementation",
                "use your own judgment",
            ] {
                assert!(
                    !lower.contains(banned),
                    "route hint must preserve pipeline guidance without visible filler phrase {banned:?}: {reminder}"
                );
            }
        }
    }

    #[test]
    fn broad_analysis_hints_fanout_but_the_host_never_spawns_the_swarm() {
        // Successor to the "swarm starves the quota and hangs" regression: the
        // hazard there was the HOST auto-fan-out. That stays closed — no effort
        // level host-prespawns from a natural-language phrase. What changed
        // (CC-parity rebalance): a multi-scope analysis now SURFACES the
        // fan-out shape as a model-led nudge at every delegation-capable
        // effort, and the model — taught the quota-sharing and minimum-agents
        // rules by the delegation rubric — owns the actual spawn decision.
        let prompt = "프로젝트 분석 cc와의 갭";
        // Default effort resolves to `Off` → solo (no delegation at all).
        assert_eq!(
            build_route_hint(prompt, Some(Effort::Off), 20_000).shape,
            RouteShape::Solo
        );
        for effort in [Effort::Medium, Effort::High, Effort::Max] {
            let hint = build_route_hint(prompt, Some(effort), 20_000);
            assert_eq!(
                hint.shape,
                RouteShape::FanoutParallel,
                "{effort:?} surfaces the fan-out shape as a model-led hint"
            );
            assert!(
                !hint.should_host_prespawn(),
                "{effort:?} must not pre-spawn a swarm"
            );
        }
        // Unset effort gets the same advisory shape and, critically, still
        // never a host pre-spawn (BUG-R2 stays closed).
        let unset = build_route_hint(prompt, None, 20_000);
        assert_eq!(unset.shape, RouteShape::FanoutParallel);
        assert!(!unset.should_host_prespawn());
        // Ultracode still does not direct-spawn a swarm from natural language;
        // it arms cheap LLM triage, the only gate that can host-fan-out
        // (`decompose`) or reach `Diagnose` before falling back to the model.
        let uc = build_route_hint(prompt, Some(Effort::Smart), 20_000);
        assert_eq!(uc.shape, RouteShape::FanoutParallel);
        assert!(!uc.should_host_prespawn());
        assert!(uc.should_run_semantic_triage());
    }

    // --- WI-B: failure escalation ---

    #[test]
    fn red_test_escalates_route_hint() {
        // A small-edit turn is normally Solo; after a prior red test the next
        // turn's hint climbs one step (Solo→DelegateOne) and surfaces the cause.
        let base = build_route_hint("tweak this helper", Some(Effort::Medium), 500);
        assert_eq!(base.shape, RouteShape::Solo);

        let escalated = base.clone().escalate(Some(FailureSignal::RedTest));
        assert_eq!(escalated.shape, RouteShape::DelegateOne);
        assert!(escalated.confidence >= base.confidence);
        let reminder = escalated.model_reminder().expect("escalated hint reminds");
        assert!(
            reminder.contains("test red"),
            "reminder names the cause: {reminder}"
        );
    }

    #[test]
    fn escalation_ladder_tops_out_at_pipeline() {
        let pipeline =
            build_route_hint("implement and verify the parser", Some(Effort::High), 9_000);
        assert_eq!(pipeline.shape, RouteShape::Pipeline);
        // Escalating a Pipeline stays Pipeline (the ceiling) — never FanoutParallel.
        let escalated = pipeline.escalate(Some(FailureSignal::ToolBudgetExceeded));
        assert_eq!(escalated.shape, RouteShape::Pipeline);
    }

    #[test]
    fn repeated_failure_stops_escalation() {
        // First failure of a kind escalates; the 2nd consecutive identical
        // failure stops (no infinite escalate→fail loop). A different failure
        // re-escalates.
        assert_eq!(
            decide_escalation(None, FailureSignal::RedTest),
            Some(FailureSignal::RedTest)
        );
        assert_eq!(
            decide_escalation(Some(FailureSignal::RedTest), FailureSignal::RedTest),
            None,
            "a 2nd consecutive identical failure must not keep escalating"
        );
        assert_eq!(
            decide_escalation(Some(FailureSignal::RedTest), FailureSignal::Deadline),
            Some(FailureSignal::Deadline),
            "a different failure re-escalates"
        );
    }

    #[test]
    fn classify_turn_failure_maps_escalation_worthy_errors_only() {
        assert_eq!(
            classify_turn_failure("verification stage: 2 tests failed"),
            Some(FailureSignal::RedTest)
        );
        assert_eq!(
            classify_turn_failure("agent exceeded its tool call budget"),
            Some(FailureSignal::ToolBudgetExceeded)
        );
        assert_eq!(
            classify_turn_failure("stream idle: request timed out"),
            Some(FailureSignal::Deadline)
        );
        assert_eq!(
            classify_turn_failure("assistant stream produced no content"),
            Some(FailureSignal::RepeatedEmptyResponse)
        );
        // Auth/network failures are not escalation-worthy — climbing the ladder
        // won't fix a missing key, so the route is left untouched.
        assert_eq!(classify_turn_failure("401 unauthorized"), None);
        assert_eq!(classify_turn_failure("connection refused"), None);
    }

    #[test]
    fn escalation_preserves_no_split_brain() {
        // Escalating any non-prespawn hint never enables host pre-spawn (it never
        // becomes FanoutParallel), so the model stays in charge of delegation.
        for text in [
            "tweak this",
            "investigate the auth bug",
            "implement the feature",
        ] {
            let base = build_route_hint(text, Some(Effort::High), 500);
            let before = base.should_host_prespawn();
            let after = base
                .escalate(Some(FailureSignal::RedTest))
                .should_host_prespawn();
            assert_eq!(
                before, after,
                "escalation must not change host pre-spawn for {text:?}"
            );
            assert!(!after, "escalated hints never host pre-spawn");
        }

        // And the explicit-parallel fast-path is left entirely untouched.
        let fast = build_route_hint("Use SpawnMultiAgent", Some(Effort::High), 0);
        assert!(fast.should_host_prespawn());
        let escalated = fast.clone().escalate(Some(FailureSignal::RedTest));
        assert!(
            escalated.should_host_prespawn(),
            "the host fast-path is not overridden by escalation"
        );
        assert_eq!(escalated.shape, fast.shape, "fast-path shape is preserved");
    }

    #[test]
    fn ultracode_escalation_never_flips_prespawn_on() {
        // Under ultracode the non-breadth shapes host-prespawn, so escalating a
        // Solo turn (Solo→DelegateOne) would NEWLY enable the host triage —
        // breaking the no-split-brain invariant escalation is meant to preserve.
        // escalate must revert that shape bump; the confidence/cause signal still
        // rides through as the advisory nudge.
        let solo = build_route_hint("rename this symbol", Some(Effort::Smart), 0);
        assert_eq!(solo.shape, RouteShape::Solo);
        assert!(
            !solo.should_host_prespawn(),
            "an ultracode small edit is Solo (no prespawn)"
        );
        let escalated = solo.clone().escalate(Some(FailureSignal::RedTest));
        assert!(
            !escalated.should_host_prespawn(),
            "escalation must not newly flip on host pre-spawn under ultracode"
        );
        assert!(
            escalated.confidence > solo.confidence,
            "the advisory confidence bump still applies even when the shape is reverted"
        );
    }

    #[test]
    fn nonbreadth_ultracode_pipeline_still_reminds_the_model() {
        // A non-breadth ultracode bug-shaped turn does NOT host-prespawn a
        // fan-out, but it arms cheap LLM triage (so the request can reach
        // Diagnose) and still surfaces its plan→implement→verify reminder for
        // the fallback turn.
        let hint = build_route_hint(
            "이 기능 버그 원인 찾아서 고치고 검증해줘",
            Some(Effort::Smart),
            20_000,
        );
        assert_eq!(hint.shape, RouteShape::Pipeline);
        assert!(!hint.should_host_prespawn());
        assert!(hint.should_run_semantic_triage());
        assert!(
            !hint.is_breadth(),
            "an implementation pipeline is not breadth"
        );
        let reminder = hint
            .model_reminder()
            .expect("a non-breadth pipeline must keep its shape reminder");
        assert!(
            reminder.contains("implementation"),
            "the fallback reminder must carry the plan→implement→verify nudge: {reminder}"
        );
    }
}
