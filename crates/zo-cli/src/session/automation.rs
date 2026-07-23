use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use commands::{DurationSpec, GoalOptions, LoopCommand};
use decision_core::{
    decide_goal_completion, decide_loop_termination, failure_signature, finding_from_text,
    triage_failures, BlockTracker, BlockedNeed, BudgetExhaustion, BudgetLedger, CheckpointAction,
    CheckpointLedger, CheckpointPolicy, ConvergenceLedger, ConvergencePolicy, ConvergenceVerdict,
    CriteriaProgress, DivergingReason, DoneReason, GoalCompletion, LoopBudget, LoopTermination,
    PivotLedger, Progress, ProgressTracker, StallResponse, BLOCK_ESCALATION_THRESHOLD,
    GOAL_PIVOT_BUDGET,
};
use runtime::{BudgetExhausted, DeepGateConfig, DeepMode, PermissionMode};
use tools::{run_process_spec, ProcessSpec};

const DEFAULT_GOAL_MAX_TURNS: u32 = 3;
const WATCH_POLL_INTERVAL: Duration = Duration::from_secs(1);
const MAX_WATCH_FILES: usize = 10_000;
const MAX_LOOP_FIXED_COUNT: u32 = 50;
const MAX_ACTIVE_SESSION_LOOPS: usize = 8;
/// Default run cap applied to a recurring (`every` / `watch`) loop when the user
/// gives no explicit `--max-runs`. Without it a recurring loop bills forever; a
/// finite default keeps cost bounded while still being generous. Mirrors the
/// fixed-count ceiling. The user can still raise it explicitly with `--max-runs`.
const DEFAULT_RECURRING_MAX_RUNS: u32 = MAX_LOOP_FIXED_COUNT;
const PLAN_FIRST_MARKER: &str = "[zo:automation-plan-first]";
/// Marker embedded on an automation prompt's control line when the owning
/// `/loop` or `/goal` was started with `--allow-writes`. The host's turn-scoped
/// permission gate ([`automation_permission_gate_change`]) reads it to decide
/// whether the unattended turn inherits the session's write permission or is
/// forced read-only + propose-only. Placed AFTER `PLAN_FIRST_MARKER` so
/// [`is_plan_first_automation_prompt`]'s `starts_with` check is unaffected.
const AUTOMATION_ALLOW_WRITES_MARKER: &str = "[zo:automation-allow-writes]";
const DEFAULT_PLAN_FIRST_CHECK_COMMAND: &str = "git diff --check";
/// Extra transient allow rules a forced-read-only automation turn needs on top of
/// the deep gate's vetted read-only bash/cargo inspection allowlist (which
/// [`automation_read_only_allow_rules`] prepends):
/// - `bash(gh *)` so the ci/pr presets can read CI status and PR comments;
///   `bash_validation` still blocks `gh` *mutations* under read-only, so only
///   reads pass.
/// - `TeamInboxPost` / `send_to_user` so the read-only turn can still record its
///   proposal and push a must-read finding — the "propose only" half of the
///   policy. Both otherwise require `WorkspaceWrite`. `send_to_user` is named
///   ahead of its landing (an allow rule for an absent tool is an inert no-op).
const AUTOMATION_READ_ONLY_EXTRA_ALLOW_RULES: &[&str] =
    &["bash(gh *)", "TeamInboxPost", "send_to_user"];

fn record_automation_trace(cwd: &Path, session_id: &str, kind: &str, event: &str, verified: bool) {
    if session_id.trim().is_empty() {
        return;
    }
    let _ = runtime::record_automation_event(cwd, session_id, kind, event, verified);
}

/// Consecutive-`Blocked` turns before a `/goal` escalates the blocker to the
/// human instead of grinding the turn budget. `ZO_GOAL_BLOCK_ESCALATION_TURNS`
/// overrides the default; `0` disables the escalation entirely (the loop falls
/// back to its stall + turn-cap guards). Mirrors `ZO_VERIFY_TREADMILL_ROUNDS`.
fn block_escalation_turns() -> u32 {
    std::env::var("ZO_GOAL_BLOCK_ESCALATION_TURNS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(BLOCK_ESCALATION_THRESHOLD)
}

/// Build the escalation digest for a goal stopped by an external blocker: the
/// validation report, the specific blocker and a concrete sample failure, and
/// the next action the human can take to unblock. The sample is truncated so a
/// verbose cargo/git tail cannot swamp the digest.
fn build_block_escalation_report(
    report_text: &str,
    need: BlockedNeed,
    sample: Option<&str>,
    turn: u32,
    max: u32,
) -> String {
    let mut out = format!(
        "{report_text}\n  Result           BLOCKED — needs {} ({turn}/{max})",
        need.label()
    );
    if let Some(sample) = sample.map(str::trim).filter(|sample| !sample.is_empty()) {
        const MAX_SAMPLE: usize = 200;
        let sample = if sample.chars().count() > MAX_SAMPLE {
            let truncated: String = sample.chars().take(MAX_SAMPLE).collect();
            format!("{truncated}…")
        } else {
            sample.to_string()
        };
        let _ = write!(out, "\n  Blocker          {sample}");
    }
    let _ = write!(out, "\n  Next             {}", need.remedy());
    out
}

/// Whether the goal-loop verification-convergence ledger is enabled.
/// `ZO_VERIFY_CONVERGENCE=0` disables it (the loop falls back to its
/// stall + turn-cap guards); anything else, or unset, keeps the default on.
/// Mirrors `ZO_GOAL_BLOCK_ESCALATION_TURNS` (re-read per use, no rebuild).
fn verify_convergence_enabled() -> bool {
    std::env::var("ZO_VERIFY_CONVERGENCE")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        != Some(0)
}

/// Pivot turns a stalled goal may spend re-approaching before it gives up.
/// `ZO_GOAL_PIVOTS` overrides; `0` disables (immediate give-up, the
/// pre-pivot behavior). Re-read per use, mirroring the other guards.
fn goal_pivots() -> u32 {
    std::env::var("ZO_GOAL_PIVOTS")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(GOAL_PIVOT_BUDGET)
}

/// Control-line marker embedded in a pivot turn's prompt. The smart router's
/// complexity classifier recognizes it and promotes the turn (a problem that
/// stalled the current model deserves a stronger one).
pub(crate) const GOAL_PIVOT_MARKER: &str = "[zo:goal-pivot]";

/// Goal-contract enforcement level. `0` = off, `1` = REPL clarify reminder
/// only, `2` (default) = also hard-gate `/goal` on an ambiguous goal with no
/// objective check. Re-read per use, mirroring the other automation guards.
pub(crate) fn goal_contract_level() -> u8 {
    std::env::var("ZO_GOAL_CONTRACT")
        .ok()
        .and_then(|value| value.trim().parse::<u8>().ok())
        .unwrap_or(2)
}

/// Whether any of the `/goal --check` strings parses to an *objective*
/// validator (cargo/git/grep). A goal with one has a decidable success
/// criterion, so the ambiguity gate must never hold it back.
pub(crate) fn has_objective_checks(checks: &[String]) -> bool {
    parse_validators(checks)
        .iter()
        .any(|validator| !matches!(validator, GoalValidator::ModelRubric { .. }))
}

/// Build the clarify-first report for an ambiguous `/goal`: name each fired
/// metric with its standard readings and ask the user to restate (or attach an
/// objective `--check`). This is the pre-execution half of the goal contract —
/// one question here is cheaper than hours on the wrong reading.
pub(crate) fn build_goal_clarify_report(goal: &str, cues: &[decision_core::AmbiguityCue]) -> String {
    let mut out = format!(
        "Goal needs one clarification (not started)\n  Goal             {goal}"
    );
    for cue in cues {
        let _ = write!(out, "\n  \"{}\" 해석      ", cue.term);
        for (index, reading) in cue.interpretations.iter().enumerate() {
            let _ = write!(out, "\n    {}. {reading}", index + 1);
        }
    }
    let _ = write!(
        out,
        "\n  Next             해석을 명시해 `/goal`을 다시 실행하거나, 객관 기준을 붙이세요 \
         (예: `--check cargo:test`, `--check grep:PATTERN`). ZO_GOAL_CONTRACT=0 으로 게이트 해제."
    );
    out
}

/// Unattended-checkpoint thresholds for `/goal` runs, from the environment.
/// `ZO_CHECKPOINT_TURNS` / `ZO_CHECKPOINT_MINUTES` /
/// `ZO_CHECKPOINT_TOKENS` override the per-axis windows (`0` disables that
/// axis; all three `0` disables checkpointing), `ZO_CHECKPOINT_MAX_UNACKED`
/// the unacknowledged-checkpoint limit. Re-read per use (no rebuild to retune),
/// mirroring the other automation guards.
fn checkpoint_policy() -> CheckpointPolicy {
    fn env_u64(key: &str, default: u64) -> u64 {
        std::env::var(key)
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(default)
    }
    let defaults = CheckpointPolicy::default();
    CheckpointPolicy {
        every_turns: u32::try_from(env_u64(
            "ZO_CHECKPOINT_TURNS",
            u64::from(defaults.every_turns),
        ))
        .unwrap_or(defaults.every_turns),
        every_wall_secs: env_u64("ZO_CHECKPOINT_MINUTES", defaults.every_wall_secs / 60)
            .saturating_mul(60),
        every_output_tokens: env_u64("ZO_CHECKPOINT_TOKENS", defaults.every_output_tokens),
        max_unacked: u32::try_from(env_u64(
            "ZO_CHECKPOINT_MAX_UNACKED",
            u64::from(defaults.max_unacked),
        ))
        .unwrap_or(defaults.max_unacked),
    }
}

/// Build the stop digest for a goal whose verification rounds are provably not
/// converging: the validation report, why more verification cannot help
/// (churn / no net progress), the findings still open, and the human's next
/// step. Mirrors [`build_block_escalation_report`] — a named cause + open list
/// is strictly more actionable than "failed".
fn build_unconverged_report(
    report_text: &str,
    reason: DivergingReason,
    unresolved: &[&str],
    turn: u32,
    max: u32,
) -> String {
    let mut out = format!(
        "{report_text}\n  Result           UNCONVERGED — {} ({turn}/{max})",
        reason.label()
    );
    for finding in unresolved {
        let _ = write!(out, "\n  Open finding     {finding}");
    }
    let _ = write!(
        out,
        "\n  Next             review the open findings and decide what to accept or fix — \
         more verification rounds will not settle this"
    );
    out
}

fn record_goal_trace(cwd: &Path, session_id: &str, event: &str, verified: bool) {
    record_automation_trace(cwd, session_id, "goal", event, verified);
    if event == "failed" {
        let _ = runtime::memory::record_self_improve_pulse(
            cwd,
            decision_core::dreamer::CandidateKind::GoalFailure,
            session_id,
            "goal",
            "goal automation failed",
            event,
            verified,
        );
    }
}

fn record_loop_trace(cwd: &Path, session_id: &str, event: &str, verified: bool) {
    record_automation_trace(cwd, session_id, "loop", event, verified);
}

fn plan_first_automation_prompt(kind: &str, body: &str, allow_writes: bool) -> String {
    // An opted-in (`--allow-writes`) loop/goal embeds the allow-writes marker on
    // the control line so the host permission gate lets the turn inherit the
    // session's write permission; the default (no marker) keeps the unattended
    // turn read-only + propose-only.
    let writes_marker = if allow_writes {
        format!(" {AUTOMATION_ALLOW_WRITES_MARKER}")
    } else {
        String::new()
    };
    format!(
        "{PLAN_FIRST_MARKER}{writes_marker} {kind} automation must plan before acting.\n\
         PLAN first: briefly state a concrete markdown plan with EXACTLY these four section headers, in order: ## Target files, ## Invariants, ## Expected tests, ## Risks. These correspond to Target files, Invariants, Expected tests, Risks and require non-placeholder content. Empty/TODO/TBD/N/A/none-only sections are invalid.\n\
         EXEC after the plan: implement the smallest correct change, validate it, and repair once if validation fails. As you begin EXEC, call TodoWrite to record your plan's concrete steps as individual checklist items (one per target file or expected test), replacing any single broad placeholder, and keep it updated as you progress. Do not stop after planning unless blocked.\n\n{body}"
    )
}

pub(crate) fn is_plan_first_automation_prompt(input: &str) -> bool {
    input.trim_start().starts_with(PLAN_FIRST_MARKER)
}

/// Cap on the deep gate's *internal* PLAN→EXEC→VERIFY self-correction attempts
/// per turn. This is deliberately decoupled from a goal's `--max-turns`: the
/// goal controller owns the *outer* repair loop (it has stall detection, a token
/// budget, and richer typed validators), so binding the inner cap to `max_turns`
/// would make `--max-turns N` silently authorize ≈N×N model legs (PLAN+EXEC+VERIFY
/// × N inner × N outer). One bounded inner self-correction (so the within-turn
/// effort escalation still fires once) is enough; the outer loop handles the rest.
pub(crate) const DEEP_INNER_MAX_ATTEMPTS: u32 = 2;

pub(crate) fn automation_plan_first_deep_gate_config() -> DeepGateConfig {
    DeepGateConfig {
        mode: DeepMode::PlanFirst,
        check_command: Some(DEFAULT_PLAN_FIRST_CHECK_COMMAND.to_string()),
        max_attempts: DEEP_INNER_MAX_ATTEMPTS,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AutomationPlanGateChange {
    /// Deep-gate state to restore after the one automation turn finishes.
    pub(crate) restore: Option<DeepGateConfig>,
    /// Temporary plan-first gate to install before the turn, if needed.
    pub(crate) install: Option<DeepGateConfig>,
}

pub(crate) fn should_install_automation_plan_gate(previous: Option<&DeepGateConfig>) -> bool {
    !matches!(
        previous.map(|config| config.mode),
        Some(DeepMode::PlanFirst)
    )
}

pub(crate) fn automation_plan_gate_change(
    input: &str,
    previous: Option<&DeepGateConfig>,
) -> Option<AutomationPlanGateChange> {
    // Centralize the one-turn install/restore policy so TUI and headless live
    // paths cannot disagree about plan-first automation scoping.
    if !is_plan_first_automation_prompt(input) {
        return None;
    }
    Some(AutomationPlanGateChange {
        restore: previous.cloned(),
        install: should_install_automation_plan_gate(previous)
            .then(automation_plan_first_deep_gate_config),
    })
}

/// Whether an automation prompt carries the `--allow-writes` opt-in marker.
///
/// Only the CONTROL LINE (the first line, which the controller composes and
/// which already starts with [`PLAN_FIRST_MARKER`]) is consulted — never the
/// body. Goal repair prompts embed model-authored text (validator output,
/// verifier objections) in the body, so a whole-string `contains` would let a
/// crafted objection carrying the literal marker escape the unattended
/// read-only downgrade. The marker is only ever legitimately placed on line 1
/// by [`plan_first_automation_prompt`].
pub(crate) fn automation_prompt_allows_writes(input: &str) -> bool {
    input
        .lines()
        .next()
        .is_some_and(|control_line| control_line.contains(AUTOMATION_ALLOW_WRITES_MARKER))
}

/// The transient allow rules to grant a forced-read-only automation turn so it
/// can still inspect the repo (read-only `bash`/`cargo`/`git`, plus `gh` reads)
/// and record its proposal into the team inbox — the "propose only" half of the
/// policy. Combines the deep gate's vetted read-only inspection allowlist
/// ([`runtime::read_only_bash_allow_rules`], the single owner) with the automation
/// extras. Without this, a forced-read-only turn's `gh run list` and
/// `TeamInboxPost` would both be denied, breaking the presets.
pub(crate) fn automation_read_only_allow_rules() -> Vec<&'static str> {
    runtime::read_only_bash_allow_rules()
        .iter()
        .copied()
        .chain(AUTOMATION_READ_ONLY_EXTRA_ALLOW_RULES.iter().copied())
        .collect()
}

/// The turn-scoped permission change for an automation-spoken (`/loop`/`/goal`
/// schedule) turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AutomationPermissionGateChange {
    /// `Some(ReadOnly)` to install a turn-scoped downgrade before the turn;
    /// `None` when the session is already read-only (nothing to change, yet the
    /// propose-only allow rules below still apply).
    pub(crate) downgrade_to: Option<PermissionMode>,
    /// The mode to restore after the one automation turn finishes.
    pub(crate) restore: PermissionMode,
}

/// Decide the turn-scoped permission policy for an automation-spoken turn.
///
/// Returns `None` for a user-typed turn (no automation marker) or an opted-in
/// (`--allow-writes`) automation turn — both keep the session's permission. For
/// an unattended, propose-only automation turn it returns `Some`, so the host
/// forces read-only + grants the propose-only allowlist ([`automation_read_only_allow_rules`]).
///
/// The downgrade only ever *lowers* privilege: `ReadOnly` is the floor of the
/// privilege ladder ([`PermissionMode::satisfies`]/`privilege_rank`), so
/// installing it can never raise a session mode. An already-read-only session is
/// left as-is (`downgrade_to = None`) — the requirement's explicit no-raise case
/// — while the allowlist still applies so a read-only session's loop can post.
/// Pure so the TUI and headless paths share one policy, mirroring
/// [`automation_plan_gate_change`].
pub(crate) fn automation_permission_gate_change(
    input: &str,
    current: PermissionMode,
) -> Option<AutomationPermissionGateChange> {
    if !is_plan_first_automation_prompt(input) || automation_prompt_allows_writes(input) {
        return None;
    }
    Some(AutomationPermissionGateChange {
        // Force read-only unless the session is already read-only. ReadOnly is the
        // ladder floor, so this only ever restricts (never raises) — including the
        // decision modes (Prompt/Allow), which unattended must not honor.
        downgrade_to: (current != PermissionMode::ReadOnly).then_some(PermissionMode::ReadOnly),
        restore: current,
    })
}

/// One-word label for a loop/goal's unattended permission mode, shown in
/// `/loop list`/`status` and `/goal status`.
fn automation_permission_label(allow_writes: bool) -> &'static str {
    if allow_writes {
        "allow-writes (inherits session)"
    } else {
        "read-only + propose"
    }
}

/// Map a turn's [`BudgetExhausted`] cause to a short human label for the
/// loop-pause digest note and system notice.
pub(crate) fn budget_exhausted_kind_label(kind: BudgetExhausted) -> &'static str {
    match kind {
        BudgetExhausted::Iterations => "iteration budget",
        BudgetExhausted::Deadline => "time budget",
        BudgetExhausted::ToolCalls => "tool-call budget",
        BudgetExhausted::OutputTokens => "output-token budget",
        BudgetExhausted::InputTokens => "input-token budget",
        BudgetExhausted::VerificationTreadmill => "verification loop",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GoalRunState {
    Active,
    Paused,
    Succeeded,
    Failed,
    /// Terminal: repeated failures are rooted in something outside the loop's
    /// control — a missing credential/permission/tool, or an unreachable/failing
    /// external service (classified by [`triage_failures`], streak-counted by
    /// [`BlockTracker`]). Retrying cannot resolve it, so instead of grinding the
    /// whole turn budget the goal escalates to the human with the specific
    /// blocker. Catches the "re-plan around an impossible goal" runaway that the
    /// identical-failure stall provably misses (surface text drifts each turn;
    /// the *class* stays Blocked).
    Blocked,
    /// Terminal: successive verification rounds are provably not converging —
    /// repaired findings keep reappearing (churn) or new blocking findings
    /// keep arriving at the round cap (classified by [`ConvergenceLedger`]).
    /// More verification cannot settle it, so the goal stops and hands the
    /// open findings to the human. Catches the repair⇄re-verify oscillation
    /// that the treadmill guard (reset by edits) and the identical-failure
    /// stall (every round's findings differ) both provably miss.
    Unconverged,
    /// Terminal: the turn budget was spent without ever producing a positive
    /// verification signal (no deterministic validators passed *and* no
    /// semantic verdict accepted). The work was neither confirmed done nor
    /// confirmed failing — reported honestly instead of a false "succeeded".
    Unverified,
    Cleared,
}

#[derive(Debug, Clone)]
pub(crate) struct GoalState {
    pub(crate) id: u64,
    pub(crate) text: String,
    pub(crate) validators: Vec<GoalValidator>,
    pub(crate) max_turns: u32,
    pub(crate) turn_count: u32,
    pub(crate) state: GoalRunState,
    pub(crate) last_report: Option<ValidationReport>,
    /// Cumulative assistant output tokens charged across the goal's turns, for
    /// the optional token budget. Distinct from `turn_count` (the turn axis).
    pub(crate) output_tokens_used: u64,
    /// Optional cap on `output_tokens_used`; `None` means no token cap (the turn
    /// cap and stall detector still bound the loop).
    pub(crate) token_budget: Option<u64>,
    /// No-progress detector: stalls the goal early when the same validation
    /// failure repeats, instead of burning the rest of the turn budget.
    progress: ProgressTracker,
    /// External-blocker detector: escalates the goal to the human when it fails
    /// on an out-of-the-loop's-control cause (auth/permission/tool/service) for
    /// several turns running. Complements `progress` — it keys on the failure
    /// *class*, so it fires even when a re-planning loop's failure text drifts
    /// and the identical-failure stall cannot. Persisted with the goal so a
    /// restart cannot re-buy the streak.
    blocks: BlockTracker,
    /// Verification-convergence ledger: tracks the *content* of the verifier's
    /// objections across turns and stops the goal when repair⇄re-verify rounds
    /// provably stop converging (churn / no net progress). Complements
    /// `progress` (which needs identical failures) and the treadmill guard
    /// (which resets on edits). Persisted with the goal so a restart cannot
    /// re-buy the verification rounds.
    convergence: ConvergenceLedger,
    /// Unattended-checkpoint pacing: after a window of unacknowledged work the
    /// goal surfaces a progress digest, and after too many unacknowledged
    /// digests it pauses (work preserved, `/goal resume` to continue) — a
    /// human who saw several checkpoints and said nothing is not watching.
    /// Any user input acknowledges. Not persisted: a restored goal loads as
    /// Paused, and the resume itself is the acknowledgement.
    checkpoint: CheckpointLedger,
    /// Monotone objective-criteria gauge: a turn that newly passes a check is
    /// demonstrably not stuck, so it resets the stall/block streaks. Only a
    /// decidable check flipping green counts — research and re-planning "feel
    /// like progress" without bound.
    criteria: CriteriaProgress,
    /// Pivot budget: on a stall (or non-converging verification), spend one
    /// re-approach turn instead of terminating; once spent, give up exactly as
    /// before. Persisted with the goal so a restart cannot refill the budget.
    pivots: PivotLedger,
    /// `--allow-writes` opt-in: when `true`, this goal's unattended action/repair
    /// turns inherit the session's write permission (the action prompt embeds the
    /// allow-writes marker). Default `false` = read-only + propose-only.
    allow_writes: bool,
}

impl GoalState {
    fn has_objective_validators(&self) -> bool {
        self.validators
            .iter()
            .any(|validator| !matches!(validator, GoalValidator::ModelRubric { .. }))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GoalController {
    active: Option<GoalState>,
    history: Vec<String>,
    next_id: u64,
}

impl Default for GoalController {
    fn default() -> Self {
        Self {
            active: None,
            history: Vec::new(),
            next_id: 1,
        }
    }
}

impl GoalController {
    // Owned `goal`/`options` are the natural API here (the goal text is moved
    // into `GoalState`); rust 1.94's pedantic needless_pass_by_value fires only
    // because this particular path clones instead of moving.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn start(&mut self, goal: String, options: GoalOptions) -> String {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let validators = parse_validators(&options.checks);
        let max_turns = options.max_turns.unwrap_or(DEFAULT_GOAL_MAX_TURNS).max(1);
        let state = GoalState {
            id,
            text: goal.clone(),
            validators,
            max_turns,
            turn_count: 0,
            state: GoalRunState::Active,
            last_report: None,
            output_tokens_used: 0,
            token_budget: options.token_budget,
            progress: ProgressTracker::default(),
            blocks: BlockTracker::default(),
            convergence: ConvergenceLedger::default(),
            checkpoint: CheckpointLedger::default(),
            criteria: CriteriaProgress::default(),
            pivots: PivotLedger::default(),
            allow_writes: options.allow_writes,
        };
        self.active = Some(state);
        self.history.push(format!(
            "goal-{id}: started `{goal}` ({max_turns} turn cap)"
        ));
        self.status_report()
    }

    pub(crate) fn edit(&mut self, goal: String) -> String {
        let Some(active) = self.active.as_mut() else {
            let options = GoalOptions::default();
            return self.start(goal, options);
        };
        active.text.clone_from(&goal);
        active.turn_count = 0;
        active.state = GoalRunState::Active;
        active.last_report = None;
        active.output_tokens_used = 0;
        active.progress = ProgressTracker::default();
        active.blocks = BlockTracker::default();
        active.convergence = ConvergenceLedger::default();
        active.checkpoint = CheckpointLedger::default();
        active.criteria = CriteriaProgress::default();
        active.pivots = PivotLedger::default();
        self.history
            .push(format!("goal-{}: edited to `{goal}`", active.id));
        self.status_report()
    }

    pub(crate) fn pause(&mut self) -> String {
        match self.active.as_mut() {
            Some(active) => {
                active.state = GoalRunState::Paused;
                self.history.push(format!("goal-{}: paused", active.id));
                self.status_report()
            }
            None => "Goal paused\n  Status           no active goal".to_string(),
        }
    }

    pub(crate) fn resume(&mut self) -> Option<(String, String)> {
        let active = self.active.as_mut()?;
        // Only a paused goal may resume. Terminal states (Succeeded/Failed/
        // Unverified) must not be revived by `/goal resume` — start or edit a new
        // goal instead. (Cleared already drops `active`, so it never reaches here.)
        if active.state != GoalRunState::Paused {
            return None;
        }
        active.state = GoalRunState::Active;
        // Resuming IS the acknowledgement — without this, a goal auto-paused at
        // an unattended checkpoint would re-pause on its very next crossing.
        active.checkpoint.acknowledge();
        self.history.push(format!("goal-{}: resumed", active.id));
        let prompt = build_goal_action_prompt(active);
        Some((self.status_report(), prompt))
    }

    pub(crate) fn clear(&mut self) -> String {
        if let Some(active) = self.active.as_mut() {
            active.state = GoalRunState::Cleared;
            self.history.push(format!("goal-{}: cleared", active.id));
        }
        self.active = None;
        "Goal cleared\n  Runtime          removed from next turn".to_string()
    }

    /// The user spoke (their own submission, or any `/goal` command): reset the
    /// active goal's unattended-checkpoint window. An actively-supervised goal
    /// never checkpoints, let alone auto-pauses. No-op without an active goal.
    pub(crate) fn acknowledge_user_input(&mut self) {
        if let Some(active) = self.active.as_mut() {
            active.checkpoint.acknowledge();
        }
    }

    /// Verify the goal's deterministic validators and fold in the turn's
    /// semantic verdict (the deep-lane verifier). `semantic` is `Some(true)` if
    /// the adversarial verifier accepted this turn's change, `Some(false)` if it
    /// rejected, `None` if no semantic judgment was produced. The report is
    /// `ok` only when [`decide_goal_completion`] returns `Satisfied`, so a goal
    /// with no objective check is never reported "passed" on the mere absence of
    /// a failure.
    pub(crate) fn verify(&mut self, cwd: &Path, semantic: Option<bool>) -> ValidationReport {
        let Some(active) = self.active.as_mut() else {
            return ValidationReport::pass("No active goal to verify.".to_string());
        };
        let report = run_validators(cwd, &active.validators, semantic);
        active.last_report = Some(report.clone());
        report
    }

    /// Run the goal's validators inline and fold the verdict into the stop
    /// decision. The validators may block on `cargo`/`git`, so the *interactive*
    /// path runs [`run_validators`] off the async loop via `spawn_blocking` and
    /// calls [`record_turn_with_report`] directly. This blocking convenience is
    /// for the headless path and tests, where there is no UI to freeze.
    pub(crate) fn record_turn_and_advance(
        &mut self,
        cwd: &Path,
        session_id: &str,
        semantic: Option<bool>,
        output_tokens: u32,
    ) -> GoalAdvance {
        let report = {
            let Some(active) = self.active.as_ref() else {
                return GoalAdvance::Idle;
            };
            if active.state != GoalRunState::Active {
                return GoalAdvance::Idle;
            }
            run_validators(cwd, &active.validators, semantic)
        };
        self.record_turn_with_report(cwd, session_id, &report, output_tokens)
    }

    /// Fold a *pre-computed* [`ValidationReport`] into the machine-verified stop
    /// decision and advance the goal's state. Split from validator execution so
    /// the interactive caller can run the (blocking) validators on a worker
    /// thread and keep only this cheap state-mutation on the async event loop.
    // Cohesive turn-folding orchestration: completion verdict + stall + blocker
    // escalation + resource budget all mutate one goal's state in lock-step;
    // splitting it would scatter the tightly-coupled decision across helpers.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn record_turn_with_report(
        &mut self,
        cwd: &Path,
        session_id: &str,
        report: &ValidationReport,
        output_tokens: u32,
    ) -> GoalAdvance {
        let Some(active) = self.active.as_mut() else {
            return GoalAdvance::Idle;
        };
        if active.state != GoalRunState::Active {
            return GoalAdvance::Idle;
        }
        active.turn_count = active.turn_count.saturating_add(1);
        active.output_tokens_used = active
            .output_tokens_used
            .saturating_add(u64::from(output_tokens));
        let report_text = report.render("Goal validation");
        active.last_report = Some(report.clone());

        // Positive progress (decision_core::CriteriaProgress): a turn that
        // newly passes an objective check is demonstrably not stuck — reset the
        // negative-streak detectors so a goal climbing its own success criteria
        // never stalls or block-escalates mid-climb. Only a new all-time best
        // counts; oscillation and research turns reset nothing.
        if active.criteria.observe(report.objective_passed) {
            active.progress = ProgressTracker::default();
            active.blocks = BlockTracker::default();
        }

        // Fold the completion verdict with stall detection and the resource
        // budget (turn cap + optional token cap) into one machine-verified stop
        // decision (decision-core `decide_loop_termination`). A genuine success
        // still wins; otherwise a repeated-failure stall or an exhausted budget
        // stops the loop early and honestly instead of burning the rest of it.
        let completion = if report.ok {
            GoalCompletion::Satisfied
        } else if report.unverifiable {
            GoalCompletion::Unverifiable
        } else {
            GoalCompletion::Continue
        };
        // Stall needs a concrete, comparable OBJECTIVE failure. Two guards:
        // (1) never observe before turn 2 — the first failure is not a repeat, so
        //     a goal always gets at least one repair attempt (stall fires at 3+);
        // (2) only an *objective* validator failure (cargo/git/grep) is a
        //     comparable repeat. `objective_failures` excludes the constant
        //     "semantic verifier rejected this turn" marker, so two cases both
        //     skip the observe and run to their cap honestly: a goal with NO
        //     objective validators (the marker is its only failure, and distinct
        //     rejections would hash identically), AND a goal whose objective
        //     checks PASSED this turn but whose verifier rejected it (a static
        //     `has_objective_validators()` gate would wrongly stall the latter).
        let progress = if !report.objective_failures.is_empty() && active.turn_count >= 2 {
            active
                .progress
                .observe(failure_signature(&report.objective_failures))
        } else {
            Progress::Advancing
        };
        // Resource budget: turn cap + optional token cap, folded by the pure
        // decision-core ledger (the single source of truth for exhaustion — the
        // ledger guards `max_turns > 0`, and `start`/`restore` clamp it to >= 1).
        let budget = BudgetLedger {
            turns: active.turn_count,
            output_tokens: active.output_tokens_used,
        }
        .exhaustion(&LoopBudget {
            max_turns: active.max_turns,
            max_output_tokens: active.token_budget,
        });

        // External-blocker escalation (decision_core::failure_triage). A goal
        // that is impossible *as scoped* — a missing credential/permission/tool,
        // an unreachable/failing service — re-plans each turn, so its surface
        // failure text drifts and the identical-failure stall above never fires;
        // it would grind the whole budget before the turn cap. Classify this
        // turn's objective failures by *why* they failed; on a short consecutive-
        // Blocked streak, stop and escalate to the human with the specific
        // blocker instead. Takes precedence over the stall/budget stop because a
        // named blocker + next step is strictly more actionable than "failed".
        // Gated on a failing turn (`!report.ok`, so a genuine success still wins)
        // and disabled by `ZO_GOAL_BLOCK_ESCALATION_TURNS=0`.
        let block_threshold = block_escalation_turns();
        let escalate_need = (block_threshold > 0 && !report.ok).then(|| {
            active
                .blocks
                .observe(triage_failures(&report.objective_failures), block_threshold)
        });
        if let Some(need) = escalate_need.flatten() {
            active.state = GoalRunState::Blocked;
            record_goal_trace(cwd, session_id, "blocked", false);
            self.history.push(format!(
                "goal-{}: blocked (needs {}) after {} turn(s)",
                active.id,
                need.label(),
                active.turn_count
            ));
            return GoalAdvance::Done(build_block_escalation_report(
                &report_text,
                need,
                report.objective_failures.first().map(String::as_str),
                active.turn_count,
                active.max_turns,
            ));
        }

        // Verification-convergence (decision_core::verify_convergence). A round
        // is folded only when the verifier produced CONCRETE objections this
        // turn (`semantic_issues` — empty unless it rejected); a bare rejection
        // or an objective-only failure carries no discriminating content and
        // leaves the ledger untouched, so a goal that never verifies can never
        // stop on it. Diverging means more verification provably cannot settle
        // the goal — repaired findings keep reappearing (churn) or new blocking
        // findings keep arriving at the round cap — so stop and hand the open
        // findings to the human instead of buying another repair⇄verify round.
        // `Converged` is advisory only and changes nothing here: acceptance
        // still flows exclusively through `decide_goal_completion`, preserving
        // the anti-optimistic-stop guarantee.
        if verify_convergence_enabled() && !report.semantic_issues.is_empty() {
            let findings: Vec<_> = report
                .semantic_issues
                .iter()
                .filter_map(|issue| finding_from_text(issue))
                .collect();
            let verdict = active
                .convergence
                .observe_round(&findings, &ConvergencePolicy::default());
            if let ConvergenceVerdict::Diverging(reason) = verdict {
                // Churn (repairs undoing each other) terminates outright — a
                // NEW approach on an oscillating patch-set makes it worse. But
                // no-net-progress only proves more of the SAME verification
                // cannot converge, so spend a pivot on a re-approach first if
                // the budget allows (never past the resource budget).
                let response = if reason == DivergingReason::NoNetProgress && budget.is_none() {
                    active.pivots.respond_to_stall(goal_pivots())
                } else {
                    // Churn always gives up; so does an exhausted budget.
                    StallResponse::GiveUp
                };
                if let StallResponse::Pivot { pivots_left } = response {
                    record_goal_trace(cwd, session_id, "pivoted", false);
                    self.history.push(format!(
                        "goal-{}: pivoted (verification not converging) after {} turn(s)",
                        active.id, active.turn_count
                    ));
                    let prompt = build_goal_pivot_prompt(active, report, pivots_left);
                    return GoalAdvance::Queue {
                        report: format!(
                            "{report_text}\n  Result           pivot queued — verification not \
                             converging; forcing a re-approach ({}/{})",
                            active.turn_count, active.max_turns
                        ),
                        prompt,
                    };
                }
                active.state = GoalRunState::Unconverged;
                record_goal_trace(cwd, session_id, "unconverged", false);
                self.history.push(format!(
                    "goal-{}: unconverged ({}) after {} turn(s)",
                    active.id,
                    reason.label(),
                    active.turn_count
                ));
                return GoalAdvance::Done(build_unconverged_report(
                    &report_text,
                    reason,
                    &active.convergence.unresolved_samples(3),
                    active.turn_count,
                    active.max_turns,
                ));
            }
        }

        match decide_loop_termination(completion, budget, progress) {
            LoopTermination::Done(DoneReason::Satisfied) => {
                record_goal_trace(cwd, session_id, "succeeded", true);
                active.state = GoalRunState::Succeeded;
                self.history.push(format!(
                    "goal-{}: succeeded after {} turn(s)",
                    active.id, active.turn_count
                ));
                GoalAdvance::Done(report_text)
            }
            LoopTermination::Done(reason) => {
                // Strategy pivot (decision_core::strategy_pivot): a stall proves
                // the current APPROACH is exhausted, not the goal — spend a pivot
                // on a forced re-approach turn before giving up, while budget
                // remains. Once the pivot budget is spent (or with
                // `ZO_GOAL_PIVOTS=0`), fall through to the honest terminal.
                if matches!(reason, DoneReason::Stalled(_)) && budget.is_none() {
                    if let StallResponse::Pivot { pivots_left } =
                        active.pivots.respond_to_stall(goal_pivots())
                    {
                        record_goal_trace(cwd, session_id, "pivoted", false);
                        self.history.push(format!(
                            "goal-{}: pivoted (stall) after {} turn(s)",
                            active.id, active.turn_count
                        ));
                        let prompt = build_goal_pivot_prompt(active, report, pivots_left);
                        return GoalAdvance::Queue {
                            report: format!(
                                "{report_text}\n  Result           pivot queued — same failure \
                                 repeated; forcing a re-approach ({}/{})",
                                active.turn_count, active.max_turns
                            ),
                            prompt,
                        };
                    }
                }
                // Distinguish an honest "could not verify" from a real failure: a
                // turn with no positive signal at all is `Unverified`; any negative
                // signal (red validator / rejected verifier) is `Failed`. The
                // reason names the specific cause (stall / token budget / turn cap).
                let unverifiable = report.unverifiable;
                active.state = if unverifiable {
                    GoalRunState::Unverified
                } else {
                    GoalRunState::Failed
                };
                let event = if unverifiable { "unverified" } else { "failed" };
                record_goal_trace(cwd, session_id, event, false);
                let result = match reason {
                    DoneReason::Stalled(_) => "stalled: same failure repeated with no progress",
                    DoneReason::BudgetExhausted(BudgetExhaustion::Tokens) => {
                        "stopped: output token budget exhausted"
                    }
                    DoneReason::BudgetExhausted(BudgetExhaustion::Turns) if unverifiable => {
                        "unverified: max turns reached with no verification signal"
                    }
                    DoneReason::BudgetExhausted(BudgetExhaustion::Turns) => {
                        "failed: max turns reached"
                    }
                    DoneReason::Satisfied => unreachable!("Satisfied handled above"),
                };
                self.history.push(format!(
                    "goal-{}: {event} after {} turn(s)",
                    active.id, active.turn_count
                ));
                GoalAdvance::Done(format!(
                    "{report_text}\n  Result           {result} ({}/{})",
                    active.turn_count, active.max_turns
                ))
            }
            LoopTermination::Continue => {
                // Unattended checkpoint (decision_core::checkpoint) — evaluated
                // only when every stop signal above said "keep going", so a
                // stop always outranks a report. A crossed window surfaces a
                // progress digest on the queued report; too many unacknowledged
                // digests auto-pause the goal (work preserved, `/goal resume`),
                // because a human who saw several checkpoints and said nothing
                // is not watching. Any user input resets via
                // [`GoalController::acknowledge_user_input`].
                let policy = checkpoint_policy();
                let action =
                    active
                        .checkpoint
                        .observe_turn(now_unix_secs(), u64::from(output_tokens), &policy);
                // Objective-criteria progress line shared by both checkpoint
                // surfaces: "how far along its own success criteria is this
                // goal" is the first thing an absent user wants to know.
                let criteria_line = if report.objective_total > 0 {
                    format!(
                        " · criteria {}/{} green",
                        active.criteria.best(),
                        report.objective_total
                    )
                } else {
                    String::new()
                };
                if action == CheckpointAction::Pause {
                    active.state = GoalRunState::Paused;
                    record_goal_trace(cwd, session_id, "checkpoint_paused", false);
                    self.history.push(format!(
                        "goal-{}: auto-paused at an unattended checkpoint after {} turn(s)",
                        active.id, active.turn_count
                    ));
                    return GoalAdvance::Pause(format!(
                        "{report_text}\n  Result           paused — {} checkpoint(s) went \
                         unacknowledged ({}/{})\n  Spent            {} output token(s){criteria_line}\
                         \n  Next             /goal resume to continue, /goal edit to rescope",
                        active.checkpoint.unacked(),
                        active.turn_count,
                        active.max_turns,
                        active.output_tokens_used,
                    ));
                }
                record_goal_trace(cwd, session_id, "repair_queued", false);
                let prompt = build_goal_repair_prompt(active, report);
                let mut queue_report = format!(
                    "{report_text}\n  Result           repair queued ({}/{})",
                    active.turn_count, active.max_turns
                );
                if action == CheckpointAction::Report {
                    let _ = write!(
                        queue_report,
                        "\n  Checkpoint       unattended for {} turn(s) · {} output token(s)\
                         {criteria_line} — auto-pause after {} more silent checkpoint(s)",
                        active.turn_count,
                        active.output_tokens_used,
                        policy.max_unacked.saturating_sub(active.checkpoint.unacked()),
                    );
                }
                GoalAdvance::Queue {
                    report: queue_report,
                    prompt,
                }
            }
        }
    }

    /// Snapshot the active goal's validators so the interactive caller can run
    /// them off the async loop (`spawn_blocking`) and feed the report back to
    /// [`record_turn_with_report`]. `None` when there is no Active goal — the
    /// caller treats that as "no goal turn to advance".
    pub(crate) fn active_goal_validators(&self) -> Option<Vec<GoalValidator>> {
        let active = self.active.as_ref()?;
        (active.state == GoalRunState::Active).then(|| active.validators.clone())
    }

    pub(crate) fn active_prompt(&self) -> Option<String> {
        self.active.as_ref().and_then(|active| {
            (active.state == GoalRunState::Active).then(|| build_goal_action_prompt(active))
        })
    }

    pub(crate) fn deep_gate_config(&self) -> Option<DeepGateConfig> {
        let active = self.active.as_ref()?;
        (active.state == GoalRunState::Active).then_some(DeepGateConfig {
            mode: DeepMode::PlanFirst,
            // Goal validators are typed labels; do not convert them into a shell
            // command. The outer goal controller runs typed validators. If there
            // are no objective validators, give the deep gate a safe read-only
            // deterministic check so semantic-only repair turns still produce
            // objective validation evidence.
            check_command: (!active.has_objective_validators())
                .then(|| DEFAULT_PLAN_FIRST_CHECK_COMMAND.to_string()),
            // Inner self-correction cap is independent of the goal turn budget —
            // the goal controller owns the outer repair loop. See
            // [`DEEP_INNER_MAX_ATTEMPTS`]: `max_turns` here would compound to
            // ≈N×N model legs.
            max_attempts: active.max_turns.min(DEEP_INNER_MAX_ATTEMPTS),
        })
    }

    pub(crate) fn status_report(&self) -> String {
        let Some(active) = self.active.as_ref() else {
            return "No goal set\n  Hint             /goal <text> --check cargo:test".to_string();
        };
        let checks = if active.validators.is_empty() {
            "model evaluator only".to_string()
        } else {
            active
                .validators
                .iter()
                .map(GoalValidator::label)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut out = format!(
            "Current goal\n  Id               goal-{}\n  Goal             {}\n  State            {:?}\n  Turns            {}/{}\n  Checks           {}\n  Permission       {}\n  Stop policy      green objective + no veto, else explicit semantic accept",
            active.id,
            active.text,
            active.state,
            active.turn_count,
            active.max_turns,
            checks,
            automation_permission_label(active.allow_writes),
        );
        if let Some(report) = &active.last_report {
            let _ = write!(
                out,
                "\n  Last validation  {}",
                if report.ok { "passed" } else { "failed" }
            );
        }
        out
    }

    pub(crate) fn history_report(&self) -> String {
        if self.history.is_empty() {
            return "Goal history\n  (none)".to_string();
        }
        format!("Goal history\n  {}", self.history.join("\n  "))
    }

    pub(crate) fn hud_label(&self) -> Option<String> {
        let active = self.active.as_ref()?;
        let state = match active.state {
            GoalRunState::Active => "active",
            GoalRunState::Paused => "paused",
            GoalRunState::Succeeded => "done",
            GoalRunState::Failed => "failed",
            GoalRunState::Blocked => "blocked",
            GoalRunState::Unconverged => "unconverged",
            GoalRunState::Unverified => "unverified",
            GoalRunState::Cleared => "cleared",
        };
        Some(format!(
            "goal-{} {state} · {}/{} · {}",
            active.id,
            active.turn_count,
            active.max_turns,
            truncate_label(&active.text, 36)
        ))
    }

    pub(crate) fn active_goal_text(&self) -> Option<&str> {
        self.active.as_ref().map(|active| active.text.as_str())
    }

    /// Snapshot the active goal for cross-restart persistence. Only a resumable
    /// goal (Active or Paused) is captured; terminal/cleared states need not
    /// survive a restart.
    pub(crate) fn snapshot_persist(&self) -> Option<persist::GoalPersist> {
        let active = self.active.as_ref()?;
        if !matches!(active.state, GoalRunState::Active | GoalRunState::Paused) {
            return None;
        }
        Some(persist::GoalPersist {
            id: active.id,
            text: active.text.clone(),
            checks: active.validators.iter().map(GoalValidator::label).collect(),
            max_turns: active.max_turns,
            turn_count: active.turn_count,
            state: format!("{:?}", active.state),
            output_tokens_used: active.output_tokens_used,
            token_budget: active.token_budget,
            progress: active.progress,
            allow_writes: active.allow_writes,
            saved_at: now_unix_secs(),
            blocks: active.blocks,
            convergence: active.convergence.clone(),
            criteria: active.criteria,
            pivots: active.pivots,
        })
    }

    /// Restore a persisted goal. Resume policy: a goal that was Active when the
    /// process exited reloads as **Paused** — a restart must never auto-run
    /// unattended repair turns; the user reactivates it with `/goal resume`.
    ///
    /// An **abandoned** goal is dropped instead of restored (see
    /// [`goal_is_abandoned`]): a one-off `/goal` that never advanced a single turn
    /// would otherwise linger in the HUD forever across restarts. A goal with any
    /// progress (`turn_count >= 1`) is always restored so real, resumable work is
    /// never silently discarded.
    pub(crate) fn restore_persist(&mut self, goal: persist::GoalPersist) {
        let state = match goal.state.as_str() {
            "Active" | "Paused" => GoalRunState::Paused,
            _ => return,
        };
        if goal_is_abandoned(goal.turn_count, goal.saved_at, now_unix_secs()) {
            // Bump `next_id` past the dropped goal so a freshly started goal does
            // not reuse its id, then leave `active` unset (nothing to restore).
            self.next_id = self.next_id.max(goal.id.saturating_add(1));
            return;
        }
        let validators = parse_validators(&goal.checks);
        self.next_id = self.next_id.max(goal.id.saturating_add(1));
        self.active = Some(GoalState {
            id: goal.id,
            text: goal.text,
            validators,
            // Clamp to >= 1 (mirrors `start`): a corrupt `max_turns: 0` would
            // otherwise let an unbounded goal run, since the budget ledger only
            // treats a positive cap as a turn limit.
            max_turns: goal.max_turns.max(1),
            turn_count: goal.turn_count,
            state,
            last_report: None,
            output_tokens_used: goal.output_tokens_used,
            token_budget: goal.token_budget,
            progress: goal.progress,
            // The runaway-guard ledgers survive the restart so their budgets
            // cannot be re-bought by restart-resume cycles (`#[serde(default)]`
            // keeps pre-ledger state files loading as fresh ledgers). The
            // checkpoint window alone restarts: the goal loads as Paused and
            // the resume itself is the acknowledgement its pacing wants.
            blocks: goal.blocks,
            convergence: goal.convergence,
            checkpoint: CheckpointLedger::default(),
            criteria: goal.criteria,
            pivots: goal.pivots,
            allow_writes: goal.allow_writes,
        });
    }
}

/// Seconds since the Unix epoch, or `0` if the clock is before the epoch (which
/// cannot happen in practice). Kept as a tiny wrapper so the persist timestamp
/// and the restore-age check read from one source.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// How long a never-progressed, persisted goal survives across restarts before
/// it is treated as abandoned and dropped on restore. A goal the user actually
/// worked (any completed turn) is exempt and lives until explicitly cleared.
const ABANDONED_GOAL_TTL_SECS: u64 = 7 * 24 * 60 * 60;

/// Whether a persisted goal should be dropped rather than restored.
///
/// Pure (time is injected) so the policy is unit-testable. A goal is abandoned
/// only when it made **no progress at all** (`turn_count == 0`) AND it is either
/// a legacy timestamp-less record (`saved_at == 0`) or older than
/// [`ABANDONED_GOAL_TTL_SECS`]. A goal with any progress is never abandoned, and
/// a recently-saved zero-progress goal is kept so a normal restart still restores
/// the goal the user just set.
fn goal_is_abandoned(turn_count: u32, saved_at: u64, now: u64) -> bool {
    if turn_count >= 1 {
        return false;
    }
    if saved_at == 0 {
        // Legacy record (written before timestamps existed): a zero-progress goal
        // from an unknown, pre-timestamp past is the lingering-HUD case the policy
        // targets — drop it.
        return true;
    }
    now.saturating_sub(saved_at) > ABANDONED_GOAL_TTL_SECS
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GoalValidator {
    Cargo(CargoValidator),
    GitDiffCheck,
    Grep { pattern: String },
    ModelRubric { label: String },
}

impl GoalValidator {
    fn label(&self) -> String {
        match self {
            Self::Cargo(action) => format!("cargo:{}", action.label()),
            Self::GitDiffCheck => "git:diff-check".to_string(),
            Self::Grep { pattern } => format!("grep:{pattern}"),
            Self::ModelRubric { label } => label.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CargoValidator {
    Fmt,
    Check,
    Test,
    Clippy,
}

impl CargoValidator {
    const fn label(self) -> &'static str {
        match self {
            Self::Fmt => "fmt",
            Self::Check => "check",
            Self::Test => "test",
            Self::Clippy => "clippy",
        }
    }

    fn process_spec(self, cwd: &Path) -> ProcessSpec {
        let args = match self {
            Self::Fmt => vec!["fmt".to_string(), "--check".to_string()],
            Self::Check => vec!["check".to_string()],
            Self::Test => vec!["test".to_string()],
            Self::Clippy => vec!["clippy".to_string()],
        };
        ProcessSpec {
            binary: "cargo".to_string(),
            args,
            cwd: Some(cwd.to_path_buf()),
            timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidationReport {
    pub(crate) ok: bool,
    /// True when the turn produced *no* verification signal at all — neither a
    /// deterministic objective validator nor a semantic verdict. Lets the
    /// caller report an honest "unverified" outcome instead of a false
    /// "failed"/"succeeded" at the turn cap.
    pub(crate) unverifiable: bool,
    pub(crate) summary: String,
    pub(crate) failures: Vec<String>,
    /// Failures from *objective* validators only (cargo/git/grep) — i.e.
    /// `failures` minus the synthetic "semantic verifier rejected this turn"
    /// marker. Used as the stall signature input so distinct semantic rejections
    /// (which carry no comparable signal) never look like a repeated failure.
    pub(crate) objective_failures: Vec<String>,
    /// The adversarial verifier's CONCRETE objections for this turn (from the
    /// deep-lane `verification_issues`), attached by the live caller. Rendered
    /// into the repair prompt so a rejected turn re-prompts the model with the
    /// specific defects to fix — never folded into the stall signature
    /// (`objective_failures`), which must stay a comparable objective signal.
    pub(crate) semantic_issues: Vec<String>,
    /// Objective validators that passed this turn (of `objective_total`). Feeds
    /// the monotone [`CriteriaProgress`] gauge: a turn that newly passes a
    /// criterion is demonstrably not stuck.
    pub(crate) objective_passed: u32,
    pub(crate) objective_total: u32,
}

impl ValidationReport {
    fn pass(summary: String) -> Self {
        Self {
            ok: true,
            unverifiable: false,
            summary,
            failures: Vec::new(),
            objective_failures: Vec::new(),
            semantic_issues: Vec::new(),
            objective_passed: 0,
            objective_total: 0,
        }
    }

    pub(crate) fn render(&self, title: &str) -> String {
        let mut out = format!(
            "{title}\n  Result           {}\n  Summary          {}",
            if self.ok {
                "passed"
            } else if self.unverifiable {
                "unverified"
            } else {
                "failed"
            },
            self.summary
        );
        for failure in &self.failures {
            let _ = write!(out, "\n  Failure          {failure}");
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GoalAdvance {
    Idle,
    Queue { report: String, prompt: String },
    Done(String),
    /// The goal auto-paused at an unattended checkpoint: too many progress
    /// digests went unacknowledged, so it stops burning budget until the human
    /// resumes. NOT terminal — the driver must surface the report but leave
    /// the goal (and its todo) intact for `/goal resume`.
    Pause(String),
}

fn parse_validators(checks: &[String]) -> Vec<GoalValidator> {
    checks
        .iter()
        .map(|check| {
            let trimmed = check.trim();
            match trimmed {
                "cargo:fmt" | "fmt" => GoalValidator::Cargo(CargoValidator::Fmt),
                "cargo:check" | "check" => GoalValidator::Cargo(CargoValidator::Check),
                "cargo:test" | "test" => GoalValidator::Cargo(CargoValidator::Test),
                "cargo:clippy" | "clippy" => GoalValidator::Cargo(CargoValidator::Clippy),
                "git:diff" | "git:diff-check" | "diff" => GoalValidator::GitDiffCheck,
                _ => trimmed.strip_prefix("grep:").map_or_else(
                    || GoalValidator::ModelRubric {
                        label: trimmed.to_string(),
                    },
                    |pattern| GoalValidator::Grep {
                        pattern: pattern.to_string(),
                    },
                ),
            }
        })
        .collect()
}

/// Run the goal's deterministic validators and fold in the turn's semantic
/// verdict to decide whether the goal may stop. The deterministic verdict comes
/// only from *objective* validators (cargo/git/grep); `ModelRubric` labels are
/// not objective checks (they are a no-op marker), so a goal configured with
/// only rubric labels — or none — has `deterministic = None` and defers to the
/// semantic verdict. The fold is [`decide_goal_completion`], so completion is
/// only ever `ok` on a positive signal, never on the absence of a negative one.
pub(crate) fn run_validators(
    cwd: &Path,
    validators: &[GoalValidator],
    semantic: Option<bool>,
) -> ValidationReport {
    let mut failures = Vec::new();
    let mut passed = Vec::new();
    let mut objective_count = 0usize;
    for validator in validators {
        if matches!(validator, GoalValidator::ModelRubric { .. }) {
            // Not an objective check — contributes no deterministic verdict.
            continue;
        }
        objective_count += 1;
        match run_validator(cwd, validator) {
            Ok(summary) => passed.push(summary),
            Err(failure) => failures.push(failure),
        }
    }

    // The deterministic verdict: `Some(true)` if every objective validator
    // passed, `Some(false)` if any failed, `None` if there were no objective
    // validators at all.
    let deterministic = if objective_count == 0 {
        None
    } else {
        Some(failures.is_empty())
    };
    // Snapshot the objective failures BEFORE appending the semantic marker: the
    // stall signature must hash only comparable, discriminating failures.
    let objective_failures = failures.clone();
    if semantic == Some(false) {
        failures.push("semantic verifier rejected this turn".to_string());
    }

    let completion = decide_goal_completion(deterministic, semantic);
    let ok = completion == GoalCompletion::Satisfied;
    let unverifiable = completion == GoalCompletion::Unverifiable;

    let summary = build_validation_summary(deterministic, semantic, &passed, &failures);
    let objective_passed = u32::try_from(passed.len()).unwrap_or(u32::MAX);
    let objective_total = u32::try_from(objective_count).unwrap_or(u32::MAX);
    ValidationReport {
        ok,
        unverifiable,
        summary,
        failures,
        objective_failures,
        // The live caller attaches the verifier's concrete objections (it owns
        // the turn summary); validators alone produce no semantic issues.
        semantic_issues: Vec::new(),
        objective_passed,
        objective_total,
    }
}

/// Human-readable one-line summary of what the gate saw, for the report.
fn build_validation_summary(
    deterministic: Option<bool>,
    semantic: Option<bool>,
    passed: &[String],
    failures: &[String],
) -> String {
    let semantic_note = match semantic {
        Some(true) => "semantic verifier accepted",
        Some(false) => "semantic verifier rejected",
        None => "no semantic verdict",
    };
    match deterministic {
        None => format!("No deterministic validators configured; {semantic_note}."),
        Some(true) => {
            if passed.is_empty() {
                format!("objective checks passed; {semantic_note}")
            } else {
                format!("{}; {semantic_note}", passed.join("; "))
            }
        }
        Some(false) => format!(
            "{} passed, {} failed; {semantic_note}",
            passed.len(),
            failures.len()
        ),
    }
}

fn run_validator(cwd: &Path, validator: &GoalValidator) -> Result<String, String> {
    match validator {
        GoalValidator::Cargo(action) => {
            let spec = action.process_spec(cwd);
            let outcome = run_process_spec(&spec).map_err(|error| error.to_string())?;
            if outcome.exit_code == 0 && !outcome.timed_out {
                Ok(format!("cargo:{} passed", action.label()))
            } else {
                Err(format!(
                    "cargo:{} failed (exit {}, timed_out={}): {}{}",
                    action.label(),
                    outcome.exit_code,
                    outcome.timed_out,
                    tail_line(&outcome.stderr),
                    tail_line(&outcome.stdout)
                ))
            }
        }
        GoalValidator::GitDiffCheck => {
            let spec = ProcessSpec {
                binary: "git".to_string(),
                args: vec!["diff".to_string(), "--check".to_string()],
                cwd: Some(cwd.to_path_buf()),
                timeout: Duration::from_secs(60),
            };
            let outcome = run_process_spec(&spec).map_err(|error| error.to_string())?;
            if outcome.exit_code == 0 && !outcome.timed_out {
                Ok("git diff --check passed".to_string())
            } else {
                Err(format!(
                    "git diff --check failed (exit {}): {}{}",
                    outcome.exit_code,
                    tail_line(&outcome.stderr),
                    tail_line(&outcome.stdout)
                ))
            }
        }
        GoalValidator::Grep { pattern } => {
            let found = grep_workspace(cwd, pattern);
            if found {
                Ok(format!("grep:{pattern} found"))
            } else {
                Err(format!("grep:{pattern} not found in workspace text files"))
            }
        }
        // Unreachable from `run_validators` (it skips `ModelRubric` before
        // dispatching here) — the rubric is graded by the independent evaluator
        // in `LiveCli::grade_active_rubric` and folded into the semantic verdict,
        // not run as a deterministic check. Kept as an honest fallback label.
        GoalValidator::ModelRubric { label } => {
            Ok(format!("{label}: graded by the independent rubric evaluator"))
        }
    }
}

fn tail_line(text: &str) -> String {
    text.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(|line| format!(" {line}"))
        .unwrap_or_default()
}

fn grep_workspace(cwd: &Path, pattern: &str) -> bool {
    let mut stack = vec![cwd.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == ".git" || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else if is_probably_text(&path)
                && fs::read_to_string(&path).is_ok_and(|content| content.contains(pattern))
            {
                return true;
            }
        }
    }
    false
}

fn is_probably_text(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(
            "rs" | "toml"
                | "md"
                | "txt"
                | "json"
                | "yaml"
                | "yml"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "css"
                | "html"
                | "sh"
                | "py"
        )
    )
}

fn build_goal_action_prompt(active: &GoalState) -> String {
    let body = format!(
        "Goal loop objective:\n{}\n\nLoop engineering cycle: Discover → Plan → Execute → Verify → Iterate. Run one bounded act → validate → repair loop. Prefer typed tools over bash. Stop only when every objective check is green and the verifier did not veto it; with no objective check, stop only on an explicit semantic-verifier accept.\n\nChecks: {}\nMax turns: {}\nCurrent turn: {}",
        active.text,
        if active.validators.is_empty() {
            "semantic model evaluator".to_string()
        } else {
            active
                .validators
                .iter()
                .map(GoalValidator::label)
                .collect::<Vec<_>>()
                .join(", ")
        },
        active.max_turns,
        active.turn_count.saturating_add(1)
    );
    plan_first_automation_prompt("goal", &body, active.allow_writes)
}

fn build_goal_repair_prompt(active: &GoalState, report: &ValidationReport) -> String {
    let validation_requirement = if active
        .validators
        .iter()
        .all(|validator| matches!(validator, GoalValidator::ModelRubric { .. }))
        && report
            .failures
            .iter()
            .any(|failure| failure == "semantic verifier rejected this turn")
    {
        "\n\nRepair validation requirement: the previous turn had no deterministic validators and the semantic verifier rejected it. In the next attempt, include a concrete validation command or typed check in Expected tests and run it before claiming completion."
    } else {
        ""
    };
    // Surface the verifier's CONCRETE objections so the repair turn fixes the
    // exact defects instead of guessing from a generic "rejected, try again".
    // This is the same signal the deep gate's own inner retry already trusts;
    // here it crosses to the outer goal repair loop.
    let verifier_objections = if report.semantic_issues.is_empty() {
        String::new()
    } else {
        let mut out = String::from(
            "\n\nThe adversarial verifier rejected the previous turn with these specific objections — resolve EACH one before claiming completion:",
        );
        for issue in &report.semantic_issues {
            let _ = write!(out, "\n  - {issue}");
        }
        out
    };
    let body = format!(
        "Goal validation failed. Repair the implementation and verify again.\n\nGoal: {}\nValidation report:\n{}{}{}\n\nUse typed tools first and avoid bash unless no typed tool can perform the check. Iterate from the validation observation; stop only when every objective check is green and the verifier did not veto it, or on an explicit semantic-verifier accept when no objective check exists.",
        active.text,
        report.render("Validation"),
        verifier_objections,
        validation_requirement,
    );
    plan_first_automation_prompt("goal repair", &body, active.allow_writes)
}

/// Build the forced re-approach prompt for a pivot turn. Unlike a repair
/// prompt (fix the specific defects, same approach), a pivot forbids the
/// failed approach outright and demands alternatives whose MEANS differ.
/// Carries [`GOAL_PIVOT_MARKER`] so the smart router promotes the turn — a
/// problem that stalled the current model deserves a stronger one.
fn build_goal_pivot_prompt(
    active: &GoalState,
    report: &ValidationReport,
    pivots_left: u32,
) -> String {
    let body = format!(
        "{GOAL_PIVOT_MARKER} The current approach has failed repeatedly with no progress — \
         re-running it is FORBIDDEN this turn.\n\nGoal: {}\nLatest validation report:\n{}\n\n\
         Pivot contract ({pivots_left} pivot(s) left after this):\n\
         1. In your plan, add an `## Alternatives` section: 2-3 approaches whose MEANS differ \
         from the failed one (a different tool/algorithm/design, a narrower scope, or a \
         prerequisite fix), each with one line on why it can succeed where the failed approach \
         could not.\n\
         2. Pick exactly ONE under `## Chosen` and implement it this turn.\n\
         3. Do NOT re-run the failed approach: same commands, same edit sites, same plan shape \
         all count as re-running it. If every alternative is worse, say so and stop honestly \
         instead of retrying.",
        active.text,
        report.render("Validation"),
    );
    plan_first_automation_prompt("goal pivot", &body, active.allow_writes)
}

#[derive(Debug, Clone)]
pub(crate) struct LoopController {
    loops: Vec<LoopState>,
    history: Vec<String>,
    next_id: u64,
}

impl Default for LoopController {
    fn default() -> Self {
        Self {
            loops: Vec::new(),
            history: Vec::new(),
            next_id: 1,
        }
    }
}

impl LoopController {
    pub(crate) fn handle_command(
        &mut self,
        cwd: &Path,
        session_id: &str,
        command: LoopCommand,
    ) -> LoopCommandResult {
        match command {
            LoopCommand::List => LoopCommandResult::Report(self.list_report()),
            LoopCommand::Status { id } => {
                LoopCommandResult::Report(self.status_report(id.as_deref()))
            }
            LoopCommand::StartFixedCount { count, prompt } => {
                self.start_fixed_count(count, prompt)
            }
            LoopCommand::StartInterval { every, prompt } => self.start_interval(every, prompt),
            LoopCommand::StartWatch { glob, prompt } => self.start_watch(cwd, glob, prompt),
            LoopCommand::RunNow { id } => self.run_now(cwd, session_id, id.as_deref()),
            LoopCommand::Pause { id } => LoopCommandResult::Report(self.pause(id.as_deref())),
            LoopCommand::Resume { id } => LoopCommandResult::Report(self.resume(id.as_deref())),
            LoopCommand::Stop { id, all } => {
                LoopCommandResult::Report(self.stop(cwd, session_id, id.as_deref(), all))
            }
            LoopCommand::Clear => {
                self.loops
                    .retain(|loop_state| loop_state.status == LoopStatus::Active);
                self.history.clear();
                LoopCommandResult::Report("Loop history cleared.".to_string())
            }
        }
    }

    pub(crate) fn drain_due_prompts(
        &mut self,
        cwd: &Path,
        session_id: &str,
        now: Instant,
    ) -> Vec<QueuedLoopPrompt> {
        let mut prompts = Vec::new();
        for loop_state in &mut self.loops {
            if loop_state.status != LoopStatus::Active {
                continue;
            }
            // A bounded recurring loop whose `--max-runs` / `--token-budget` is
            // already spent completes here instead of charging another run and
            // emitting a phantom 'fired' trace for a tick the pop-gate would drop.
            // `run_count` is the runs already completed, so this stops the loop the
            // moment a further run would exceed the cap. Unbounded loops (the
            // default) never exhaust, preserving the prior fire-until-exit behavior.
            if matches!(
                loop_state.kind,
                LoopKind::Interval { .. } | LoopKind::Watch { .. }
            ) && loop_state.recurring_budget_spent()
            {
                loop_state.status = LoopStatus::Completed;
                continue;
            }
            match &mut loop_state.kind {
                LoopKind::FixedCount { .. } => {}
                LoopKind::Interval { every, next_due } => {
                    if now >= *next_due {
                        loop_state.run_count = loop_state.run_count.saturating_add(1);
                        *next_due = now + *every;
                        record_loop_trace(cwd, session_id, "fired", false);
                        prompts.push(QueuedLoopPrompt::new(loop_state, None));
                    }
                }
                LoopKind::Watch {
                    glob,
                    snapshot,
                    next_poll,
                } => {
                    if now < *next_poll {
                        continue;
                    }
                    *next_poll = now + WATCH_POLL_INTERVAL;
                    let fresh = collect_watch_snapshot(cwd, glob);
                    let changed = changed_files(snapshot, &fresh);
                    *snapshot = fresh;
                    if !changed.is_empty() {
                        loop_state.run_count = loop_state.run_count.saturating_add(1);
                        record_loop_trace(cwd, session_id, "fired", false);
                        prompts.push(QueuedLoopPrompt::new(loop_state, Some(changed)));
                    }
                }
            }
        }
        prompts
    }

    pub(crate) fn next_due_in(&self, now: Instant) -> Option<Duration> {
        self.next_due_info(now).map(|(due_in, _)| due_in)
    }

    pub(crate) fn next_due_info(&self, now: Instant) -> Option<(Duration, &str)> {
        self.loops
            .iter()
            .filter(|loop_state| loop_state.status == LoopStatus::Active)
            .filter_map(|loop_state| match &loop_state.kind {
                LoopKind::FixedCount { .. } => None,
                LoopKind::Interval { next_due, .. } => {
                    Some((next_due.saturating_duration_since(now), loop_state.prompt.as_str()))
                }
                LoopKind::Watch { next_poll, .. } => Some((
                    next_poll.saturating_duration_since(now),
                    loop_state.prompt.as_str(),
                )),
            })
            .min_by_key(|(due_in, _)| *due_in)
    }

    pub(crate) fn hud_label(&self) -> Option<String> {
        let active = self
            .loops
            .iter()
            .filter(|loop_state| loop_state.status == LoopStatus::Active)
            .count();
        (active > 0).then(|| format!("{active} loop(s) active"))
    }

    /// Pop-time gate: the caller dequeued a loop-owned prompt and asks whether to
    /// actually run it. This is what makes `/loop` stoppable — a prompt queued
    /// before `/loop stop|pause` (or before the turn cap was reached) is dropped
    /// here instead of running. The turn cap is folded by the same decision-core
    /// brain as `/goal` ([`decide_loop_termination`] over a [`BudgetLedger`]). The
    /// pop-gate itself folds only the resource budget (completion `Continue`, no
    /// objective failure to fold), so a budget-only loop stops on the fixed-count
    /// cap or the recurring `--max-runs` / `--token-budget` ceiling, never on a
    /// fabricated "satisfied"/"stalled". A `/loop --until <check>` completion
    /// condition is evaluated *after* each turn (`loop_until_validators` +
    /// `complete_loop`), which marks the loop `Completed` so the next pop is a
    /// clean `Skip` — it is not folded here.
    pub(crate) fn begin_loop_turn(
        &mut self,
        cwd: &Path,
        session_id: &str,
        loop_id: &str,
    ) -> LoopTurnGate {
        let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) else {
            // The loop was `/loop clear`ed after this run was queued.
            return LoopTurnGate::Skip;
        };
        if loop_state.status != LoopStatus::Active {
            // Stopped, paused, or already completed → drop this stale queued run.
            return LoopTurnGate::Skip;
        }
        match loop_state.kind {
            LoopKind::FixedCount { count } => {
                let budget = BudgetLedger {
                    turns: loop_state.run_count,
                    output_tokens: loop_state.output_tokens,
                }
                .exhaustion(&loop_state.budget);
                match decide_loop_termination(
                    GoalCompletion::Continue,
                    budget,
                    Progress::Advancing,
                ) {
                    LoopTermination::Continue => {
                        loop_state.run_count = loop_state.run_count.saturating_add(1);
                        record_loop_trace(cwd, session_id, "fired", false);
                        // The final queued run completes the loop, so any further
                        // stale pop (or a no-op `/loop stop`) is a clean Skip.
                        if loop_state.run_count >= count {
                            loop_state.status = LoopStatus::Completed;
                        }
                        LoopTurnGate::Run
                    }
                    LoopTermination::Done(_) => {
                        loop_state.status = LoopStatus::Completed;
                        LoopTurnGate::Skip
                    }
                }
            }
            // Recurring loops already charged `run_count` and traced the run when
            // `drain_due_prompts` produced this prompt. The pop-gate now also folds
            // any per-loop budget (`--max-runs` / `--token-budget`) through the same
            // decision-core ledger as fixed-count loops, so a bounded recurring loop
            // stops once its cap/ceiling is hit instead of firing until session
            // exit. An all-unset budget never exhausts, preserving the prior
            // unbounded behavior of a plain `/loop every 30s`.
            LoopKind::Interval { .. } | LoopKind::Watch { .. } => {
                let exhaustion = BudgetLedger {
                    // `run_count` was already charged for this tick in
                    // `drain_due_prompts`/`run_now`, so the runs COMPLETED before
                    // this one are `run_count - 1`. Folding against that count makes
                    // `--max-runs N` permit exactly N runs (the (N+1)-th tick finds
                    // `turns == N` exhausted and is dropped), matching the
                    // pre-increment check the fixed-count arm does itself.
                    turns: loop_state.run_count.saturating_sub(1),
                    output_tokens: loop_state.output_tokens,
                }
                .exhaustion(&loop_state.budget);
                match decide_loop_termination(
                    GoalCompletion::Continue,
                    exhaustion,
                    Progress::Advancing,
                ) {
                    LoopTermination::Continue => LoopTurnGate::Run,
                    LoopTermination::Done(_) => {
                        loop_state.status = LoopStatus::Completed;
                        LoopTurnGate::Skip
                    }
                }
            }
        }
    }

    /// Charge a recurring loop's completed turn with the assistant output tokens
    /// it produced, so a `--token-budget` ceiling folds through the pop-gate on the
    /// next tick (mirrors the per-turn token charge `/goal` does in
    /// [`GoalController::record_turn_with_report`]). Fixed-count loops are bounded
    /// by their run count, so this is a no-op for them. Unknown / inactive loops
    /// are ignored — the run may have been cleared since it fired.
    pub(crate) fn charge_loop_output(&mut self, loop_id: &str, output_tokens: u32) {
        if let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_state.output_tokens = loop_state
                .output_tokens
                .saturating_add(u64::from(output_tokens));
        }
    }

    /// The `--until` completion validators for an active loop, or `None` when the
    /// loop is inactive/unknown or has no completion check. The caller runs them
    /// off the async loop (they may block on `cargo`/`grep`) and, if they pass,
    /// calls [`Self::complete_loop`] — splitting validator execution from state
    /// mutation exactly as `/goal` does (`advance_goal_after_turn`).
    pub(crate) fn loop_until_validators(&self, loop_id: &str) -> Option<Vec<GoalValidator>> {
        let loop_state = self.loops.iter().find(|l| l.id == loop_id)?;
        if loop_state.status != LoopStatus::Active || loop_state.until.is_empty() {
            return None;
        }
        Some(loop_state.until.clone())
    }

    /// Mark a loop done because its `--until` condition was met. A subsequent
    /// pop-gate finds it non-`Active` and drops any further queued run.
    pub(crate) fn complete_loop(&mut self, loop_id: &str) {
        if let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_state.status = LoopStatus::Completed;
            self.history.push(format!("{loop_id}: completed (--until met)"));
        }
    }

    /// Fold this turn's `--until` *objective* failures into the loop's stall AND
    /// external-blocker trackers (the same anti-no-progress / escalate-to-human
    /// brains `/goal` uses), and say how the loop should react.
    ///
    /// `Blocked` wins over `Stalled` (a named blocker + remedy is strictly more
    /// actionable) and — unlike the stall — is not gated on the 2nd run: the
    /// triage class is stable under text drift, so two consecutive blocked runs
    /// escalate even when every run's failure text differs. The stall keeps its
    /// original gates: never before the loop's 2nd run (the first failure is not
    /// a repeat), and only an *objective* failure is a comparable repeat (a
    /// `--until` with no objective signal can never stall, so it keeps its prior
    /// run-to-budget behavior).
    pub(crate) fn observe_loop_stall(
        &mut self,
        loop_id: &str,
        objective_failures: &[String],
    ) -> LoopStallVerdict {
        let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) else {
            return LoopStallVerdict::Continue;
        };
        if objective_failures.is_empty() {
            return LoopStallVerdict::Continue;
        }
        let block_threshold = block_escalation_turns();
        if block_threshold > 0 {
            if let Some(need) = loop_state
                .blocks
                .observe(triage_failures(objective_failures), block_threshold)
            {
                return LoopStallVerdict::Blocked(need);
            }
        }
        if loop_state.run_count < 2 {
            return LoopStallVerdict::Continue;
        }
        if matches!(
            loop_state.progress.observe(failure_signature(objective_failures)),
            Progress::Stalled(_)
        ) {
            LoopStallVerdict::Stalled
        } else {
            LoopStallVerdict::Continue
        }
    }

    /// Stop a loop that has stalled on its `--until` condition. Distinct from
    /// [`Self::complete_loop`] (`--until` met = success): a stall is a give-up, so
    /// the loop is `Stopped`, not `Completed`. A subsequent pop-gate drops any
    /// further queued run.
    pub(crate) fn stall_loop(&mut self, loop_id: &str) {
        if let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_state.status = LoopStatus::Stopped;
            self.history.push(format!(
                "{loop_id}: stopped (--until stalled — same failure repeated with no progress)"
            ));
        }
    }

    /// Stop a loop whose `--until` condition is failing on an external blocker
    /// (auth/permission/tool/service — [`triage_failures`]). Retrying cannot
    /// resolve it, so the loop stops and the caller escalates the specific
    /// blocker + remedy to the human. Mirrors the goal's Blocked terminal.
    pub(crate) fn block_loop(&mut self, loop_id: &str, need: BlockedNeed) {
        if let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) {
            loop_state.status = LoopStatus::Stopped;
            self.history.push(format!(
                "{loop_id}: stopped (--until blocked — needs {})",
                need.label()
            ));
        }
    }

    /// Pause a loop because its just-finished turn exhausted a turn budget
    /// (iteration cap / deadline / tool-call budget). Only an `Active` loop is
    /// paused; returns `true` when it actually transitioned so the caller emits
    /// the digest note + system notice exactly once (a loop already Paused/Stopped
    /// by the user in the meantime is left alone). Distinct from a stall: a budget
    /// pause is recoverable — the user resumes it with `/loop resume` after
    /// deciding what to do.
    pub(crate) fn pause_for_budget(&mut self, loop_id: &str) -> bool {
        let Some(loop_state) = self.loops.iter_mut().find(|l| l.id == loop_id) else {
            return false;
        };
        if loop_state.status != LoopStatus::Active {
            return false;
        }
        loop_state.status = LoopStatus::Paused;
        self.history
            .push(format!("{loop_id}: paused (turn budget exhausted — awaiting user decision)"));
        true
    }

    #[allow(clippy::needless_pass_by_value)]
    fn start_fixed_count(&mut self, count: u32, prompt: String) -> LoopCommandResult {
        if count > MAX_LOOP_FIXED_COUNT {
            return LoopCommandResult::Report(format!(
                "Loop rejected\n  Reason           fixed-count budget exceeded\n  Requested        {count} run(s)\n  Budget           max {MAX_LOOP_FIXED_COUNT} run(s) per command"
            ));
        }
        // Fixed-count loops take their prompt verbatim (they never run
        // `split_loop_budget_flags`), so strip a leading `--allow-writes` opt-in
        // here for parity with recurring loops.
        let (allow_writes, prompt) = strip_leading_allow_writes(&prompt);
        let id = self.next_loop_id();
        // Stay `Active` and own the loop in the controller (run_count starts at 0
        // and is charged per actual run in `begin_loop_turn`). The old design set
        // `Completed` up front and dumped all N prompts into the message queue as
        // plain text — so `/loop stop|pause` could never halt an in-flight loop
        // (the controller no longer tracked it, and the queued prompts ran no
        // matter what). Controller-owned + a pop-time gate makes `/loop` stoppable
        // and routes the turn cap through the same decision-core ledger as `/goal`.
        let loop_state = LoopState {
            id: id.clone(),
            prompt: prompt.clone(),
            status: LoopStatus::Active,
            run_count: 0,
            output_tokens: 0,
            // The fixed-count cap IS the turn budget — folded by the same ledger as
            // recurring loops in `begin_loop_turn`.
            budget: LoopBudget {
                max_turns: count.max(1),
                max_output_tokens: None,
            },
            // Fixed-count loops are bounded by `count`; no `--until` completion check.
            until: Vec::new(),
            progress: ProgressTracker::default(),
            blocks: BlockTracker::default(),
            allow_writes,
            kind: LoopKind::FixedCount { count },
        };
        self.loops.push(loop_state);
        self.history
            .push(format!("{id}: queued fixed-count loop ({count} run(s))"));
        let prompts = (1..=count)
            .map(|run| QueuedLoopPrompt {
                text: format_loop_prompt(&id, run, Some(count), &prompt, None, allow_writes),
                loop_id: id.clone(),
            })
            .collect();
        LoopCommandResult::Queue {
            report: format!(
                "Loop created\n  Id               {id}\n  Mode             fixed-count\n  Runs             {count}\n  Status           queued {count} session turn(s)
  Scope            persisted; reloads paused after restart
  Budget           fixed-count max {MAX_LOOP_FIXED_COUNT} run(s)
  Control          /loop stop|pause {id} halts remaining runs
  Plan             required before each run"
            ),
            prompts,
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn start_interval(&mut self, every: DurationSpec, prompt: String) -> LoopCommandResult {
        if let Some(report) = self.reject_if_active_loop_budget_exceeded("interval") {
            return LoopCommandResult::Report(report);
        }
        let id = self.next_loop_id();
        let (budget, until, allow_writes, prompt) = split_loop_budget_flags(&prompt);
        let loop_state = LoopState {
            id: id.clone(),
            prompt,
            status: LoopStatus::Active,
            run_count: 0,
            output_tokens: 0,
            budget,
            until,
            progress: ProgressTracker::default(),
            blocks: BlockTracker::default(),
            allow_writes,
            kind: LoopKind::Interval {
                every: every.duration,
                next_due: Instant::now() + every.duration,
            },
        };
        self.loops.push(loop_state);
        self.history
            .push(format!("{id}: started interval loop every {}", every.raw));
        LoopCommandResult::Report(format!(
            "Loop created\n  Id               {id}\n  Mode             interval\n  Every            {}\n  Status           active
  Scope            persisted; reloads paused after restart
  Budget           active recurring loops {}/{}{}
  Plan             required before each run",
            every.raw,
            self.active_loop_count(),
            MAX_ACTIVE_SESSION_LOOPS,
            recurring_budget_note(budget),
        ))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn start_watch(&mut self, cwd: &Path, glob: String, prompt: String) -> LoopCommandResult {
        if let Some(report) = self.reject_if_active_loop_budget_exceeded("watch") {
            return LoopCommandResult::Report(report);
        }
        let id = self.next_loop_id();
        let snapshot = collect_watch_snapshot(cwd, &glob);
        let (budget, until, allow_writes, prompt) = split_loop_budget_flags(&prompt);
        let loop_state = LoopState {
            id: id.clone(),
            prompt,
            status: LoopStatus::Active,
            run_count: 0,
            output_tokens: 0,
            budget,
            until,
            progress: ProgressTracker::default(),
            blocks: BlockTracker::default(),
            allow_writes,
            kind: LoopKind::Watch {
                glob: glob.clone(),
                snapshot,
                next_poll: Instant::now() + WATCH_POLL_INTERVAL,
            },
        };
        self.loops.push(loop_state);
        self.history
            .push(format!("{id}: started watch loop on {glob}"));
        LoopCommandResult::Report(format!(
            "Loop created\n  Id               {id}\n  Mode             watch\n  Glob             {glob}\n  Status           active (polling)
  Scope            persisted; reloads paused after restart
  Budget           active recurring loops {}/{}{}
  Plan             required before each run",
            self.active_loop_count(),
            MAX_ACTIVE_SESSION_LOOPS,
            recurring_budget_note(budget),
        ))
    }

    fn run_now(&mut self, cwd: &Path, session_id: &str, id: Option<&str>) -> LoopCommandResult {
        let Some(index) = self.resolve_loop_index(id) else {
            return LoopCommandResult::Report(format!(
                "Loop run-now\n  Target           {}\n  Status           no matching loop",
                id.unwrap_or("latest")
            ));
        };
        if matches!(self.loops[index].kind, LoopKind::FixedCount { .. }) {
            return LoopCommandResult::Report(format!(
                "Loop run-now\n  Id               {}\n  Status           fixed-count loops cannot be run manually",
                self.loops[index].id
            ));
        }
        // A recurring loop that is stopped or paused must not fire via run-now:
        // resume it explicitly first. Without this guard `/loop run <id>` revives a
        // Stopped/Paused loop and bumps its run count, bypassing its lifecycle.
        if self.loops[index].status != LoopStatus::Active {
            return LoopCommandResult::Report(format!(
                "Loop run-now\n  Id               {}\n  Status           not active ({:?}); resume before running",
                self.loops[index].id, self.loops[index].status
            ));
        }

        let loop_state = &mut self.loops[index];
        loop_state.run_count = loop_state.run_count.saturating_add(1);
        record_loop_trace(cwd, session_id, "fired", false);
        let prompt = QueuedLoopPrompt::new(loop_state, None);
        LoopCommandResult::Queue {
            report: format!(
                "Loop run-now\n  Id               {}\n  Status           queued one run
  Plan             required before this run",
                prompt.loop_id
            ),
            prompts: vec![prompt],
        }
    }

    fn pause(&mut self, id: Option<&str>) -> String {
        let Some(loop_state) = self.resolve_loop_mut(id) else {
            return format!(
                "Loop pause\n  Target           {}\n  Status           no matching loop",
                id.unwrap_or("latest")
            );
        };
        loop_state.status = LoopStatus::Paused;
        format!("Loop paused\n  Id               {}", loop_state.id)
    }

    fn resume(&mut self, id: Option<&str>) -> String {
        let Some(index) = self.resolve_loop_index(id) else {
            return format!(
                "Loop resume\n  Target           {}\n  Status           no matching loop",
                id.unwrap_or("latest")
            );
        };
        // Fixed-count loops are now controller-owned and Active while runs remain,
        // so `/loop pause` can halt one mid-flight; resume must therefore un-pause
        // it symmetrically (its queued runs resume firing through the pop-gate).
        // The recurring active-budget check below only counts recurring loops, so
        // resuming a fixed-count loop never trips it.
        let would_activate_recurring = self.loops[index].status != LoopStatus::Active
            && !matches!(self.loops[index].kind, LoopKind::FixedCount { .. });
        if would_activate_recurring && self.active_loop_count() >= MAX_ACTIVE_SESSION_LOOPS {
            return format!(
                "Loop resume\n  Id               {}\n  Status           active recurring loop budget exceeded\n  Budget           {}/{} active",
                self.loops[index].id,
                self.active_loop_count(),
                MAX_ACTIVE_SESSION_LOOPS
            );
        }

        let loop_state = &mut self.loops[index];
        loop_state.status = LoopStatus::Active;
        match &mut loop_state.kind {
            LoopKind::Interval { every, next_due } => *next_due = Instant::now() + *every,
            LoopKind::Watch { next_poll, .. } => *next_poll = Instant::now() + WATCH_POLL_INTERVAL,
            LoopKind::FixedCount { .. } => {}
        }
        format!("Loop resumed\n  Id               {}", loop_state.id)
    }

    fn stop(&mut self, cwd: &Path, session_id: &str, id: Option<&str>, all: bool) -> String {
        if all {
            let stopped = self
                .loops
                .iter_mut()
                .filter(|loop_state| loop_state.status == LoopStatus::Active)
                .map(|loop_state| {
                    loop_state.status = LoopStatus::Stopped;
                    1usize
                })
                .sum::<usize>();
            if stopped > 0 {
                record_loop_trace(cwd, session_id, "stopped", false);
            }
            return format!("Loop stopped\n  Target           all\n  Count            {stopped}");
        }
        let Some(loop_state) = self.resolve_loop_mut(id) else {
            return format!(
                "Loop stop\n  Target           {}\n  Status           no matching loop",
                id.unwrap_or("latest")
            );
        };
        loop_state.status = LoopStatus::Stopped;
        record_loop_trace(cwd, session_id, "stopped", false);
        format!("Loop stopped\n  Id               {}", loop_state.id)
    }

    fn list_report(&self) -> String {
        if self.loops.is_empty() {
            return "Loops\n  (none)\n  Presets          /loop ci|pr|audit — recurring recipes (read-only + propose; --allow-writes to inherit)".to_string();
        }
        let mut out = format!(
            "Loops\n  Budget           active recurring loops {}/{}",
            self.active_loop_count(),
            MAX_ACTIVE_SESSION_LOOPS
        );
        for loop_state in &self.loops {
            let _ = write!(
                out,
                "\n  {}              {:?} · {} run(s) · {} · {} · {}",
                loop_state.id,
                loop_state.status,
                loop_state.run_count,
                loop_state.kind.label(),
                automation_permission_label(loop_state.allow_writes),
                truncate_label(&loop_state.prompt, 38)
            );
        }
        out
    }

    fn status_report(&self, id: Option<&str>) -> String {
        let target = if let Some(id) = id {
            let target = self.loops.iter().find(|loop_state| loop_state.id == id);
            if target.is_none() {
                return format!(
                    "Loop status\n  Target           {id}\n  Status           no matching loop"
                );
            }
            target
        } else {
            self.loops
                .iter()
                .rev()
                .find(|loop_state| loop_state.status == LoopStatus::Active)
                .or_else(|| self.loops.last())
        };
        let Some(loop_state) = target else {
            return "Loop status\n  (none)".to_string();
        };
        format!(
            "Loop status\n  Id               {}\n  State            {:?}\n  Mode             {}\n  Permission       {}\n  Runs             {}\n  Scope            persisted; reloads paused after restart\n  Stop policy      {}\n  Prompt           {}",
            loop_state.id,
            loop_state.status,
            loop_state.kind.label(),
            automation_permission_label(loop_state.allow_writes),
            loop_state.run_count,
            loop_state.kind.stop_policy(self.active_loop_count()),
            loop_state.prompt
        )
    }

    fn active_loop_count(&self) -> usize {
        self.loops
            .iter()
            .filter(|loop_state| {
                loop_state.status == LoopStatus::Active
                    && !matches!(loop_state.kind, LoopKind::FixedCount { .. })
            })
            .count()
    }

    fn reject_if_active_loop_budget_exceeded(&self, mode: &str) -> Option<String> {
        let active = self.active_loop_count();
        (active >= MAX_ACTIVE_SESSION_LOOPS).then(|| {
            format!(
                "Loop rejected\n  Mode             {mode}\n  Reason           active recurring loop budget exceeded\n  Budget           {active}/{MAX_ACTIVE_SESSION_LOOPS} active"
            )
        })
    }

    fn resolve_loop_index(&self, id: Option<&str>) -> Option<usize> {
        if let Some(id) = id {
            self.loops.iter().position(|loop_state| loop_state.id == id)
        } else {
            self.loops
                .iter()
                .rposition(|loop_state| loop_state.status == LoopStatus::Active)
                .or_else(|| {
                    self.loops
                        .iter()
                        .rposition(|loop_state| loop_state.status == LoopStatus::Paused)
                })
        }
    }

    fn resolve_loop_mut(&mut self, id: Option<&str>) -> Option<&mut LoopState> {
        let index = self.resolve_loop_index(id)?;
        self.loops.get_mut(index)
    }

    fn next_loop_id(&mut self) -> String {
        let id = format!("loop-{}", self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Snapshot resumable recurring loops for cross-restart persistence.
    /// Fixed-count loops are excluded (their prompts are eagerly drained at
    /// creation, so reviving one would re-fire it), as are stopped/completed
    /// loops.
    pub(crate) fn snapshot_persist(&self) -> Vec<persist::LoopPersist> {
        self.loops
            .iter()
            .filter(|loop_state| {
                matches!(loop_state.status, LoopStatus::Active | LoopStatus::Paused)
            })
            .filter_map(|loop_state| {
                let kind = match &loop_state.kind {
                    LoopKind::FixedCount { .. } => return None,
                    LoopKind::Interval { every, .. } => persist::LoopKindPersist::Interval {
                        every_secs: every.as_secs().max(1),
                    },
                    LoopKind::Watch { glob, .. } => persist::LoopKindPersist::Watch {
                        glob: glob.clone(),
                    },
                };
                Some(persist::LoopPersist {
                    id: loop_state.id.clone(),
                    prompt: loop_state.prompt.clone(),
                    status: format!("{:?}", loop_state.status),
                    run_count: loop_state.run_count,
                    output_tokens: loop_state.output_tokens,
                    budget: loop_state.budget,
                    until: loop_state.until.iter().map(GoalValidator::label).collect(),
                    progress: loop_state.progress,
                    allow_writes: loop_state.allow_writes,
                    kind,
                })
            })
            .collect()
    }

    /// Restore persisted recurring loops, re-arming each schedule relative to NOW
    /// (a past due time never replays a backlog) and re-snapshotting watches (a
    /// change made while the process was down is intentionally not retro-fired).
    pub(crate) fn restore_persist(&mut self, cwd: &Path, loops: Vec<persist::LoopPersist>) {
        for persisted in loops {
            // Resume policy: a loop that was Active at exit reloads as **Paused**
            // (mirrors the goal restore policy). A restart must never silently
            // resume an unattended, billing recurring loop; the user reactivates
            // it deliberately with `/loop resume`.
            let status = match persisted.status.as_str() {
                "Active" | "Paused" => LoopStatus::Paused,
                _ => continue,
            };
            let kind = match persisted.kind {
                persist::LoopKindPersist::Interval { every_secs } => {
                    let every = Duration::from_secs(every_secs.max(1));
                    LoopKind::Interval {
                        every,
                        next_due: Instant::now() + every,
                    }
                }
                persist::LoopKindPersist::Watch { glob } => {
                    let snapshot = collect_watch_snapshot(cwd, &glob);
                    LoopKind::Watch {
                        glob,
                        snapshot,
                        next_poll: Instant::now() + WATCH_POLL_INTERVAL,
                    }
                }
            };
            if let Some(seq) = persisted
                .id
                .strip_prefix("loop-")
                .and_then(|n| n.parse::<u64>().ok())
            {
                self.next_id = self.next_id.max(seq.saturating_add(1));
            }
            self.loops.push(LoopState {
                id: persisted.id,
                prompt: persisted.prompt,
                status,
                run_count: persisted.run_count,
                // Round-trip the resource budget and spent tokens so a restored
                // `--max-runs` / `--token-budget` loop resumes bounded (and stops
                // at its cap) instead of reloading unbounded and running forever.
                output_tokens: persisted.output_tokens,
                budget: persisted.budget,
                until: parse_validators(&persisted.until),
                // Restore the stall streak so a loop that was already repeating a
                // failure before restart does not get a fresh budget to do it again.
                progress: persisted.progress,
                // Not persisted (mirrors the goal's tracker): a restored loop
                // restarts its blocked streak from zero — at worst one extra run.
                blocks: BlockTracker::default(),
                // Restore the write opt-in so a `--allow-writes` loop resumes with
                // its permission mode instead of silently reverting to read-only.
                allow_writes: persisted.allow_writes,
                kind,
            });
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LoopCommandResult {
    Report(String),
    Queue {
        report: String,
        prompts: Vec<QueuedLoopPrompt>,
    },
}

/// The pop-time verdict for a loop-owned queued prompt (see
/// [`LoopController::begin_loop_turn`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopTurnGate {
    /// Run this loop turn.
    Run,
    /// Drop this stale run: the loop was stopped/paused/cleared or its budget is
    /// spent. No turn is dispatched.
    Skip,
}

/// How a recurring loop should react to a failing `--until` check (see
/// [`LoopController::observe_loop_stall`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopStallVerdict {
    /// Keep running — the failure is new (or not yet a comparable repeat).
    Continue,
    /// The same objective failure repeated with no progress: stop the loop.
    Stalled,
    /// The failure is rooted outside the loop's control (auth/permission/tool/
    /// service): stop and escalate the specific blocker to the human.
    Blocked(BlockedNeed),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QueuedLoopPrompt {
    pub(crate) text: String,
    pub(crate) loop_id: String,
}

impl QueuedLoopPrompt {
    #[allow(clippy::needless_pass_by_value)]
    fn new(loop_state: &LoopState, changed: Option<Vec<String>>) -> Self {
        let total = match loop_state.kind {
            LoopKind::FixedCount { count } => Some(count),
            LoopKind::Interval { .. } | LoopKind::Watch { .. } => None,
        };
        Self {
            text: format_loop_prompt(
                &loop_state.id,
                loop_state.run_count,
                total,
                &loop_state.prompt,
                changed.as_deref(),
                loop_state.allow_writes,
            ),
            loop_id: loop_state.id.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct LoopState {
    id: String,
    prompt: String,
    status: LoopStatus,
    run_count: u32,
    /// Cumulative assistant output tokens charged across this loop's runs, for the
    /// optional per-loop token budget. Recurring loops fold this through the same
    /// [`BudgetLedger`] as fixed-count loops; charged in [`LoopController::charge_loop_output`].
    output_tokens: u64,
    /// Optional resource budget. Fixed-count loops always carry their `count` as
    /// the turn cap; recurring (interval/watch) loops carry it only when started
    /// with `--max-runs` / `--token-budget`. An all-unset budget never exhausts on
    /// its own, so a plain `/loop every 30s` keeps its prior unbounded behavior.
    budget: LoopBudget,
    /// Optional `--until <check>` completion condition(s): objective validators
    /// (`grep:` / `cargo:` / `git:diff-check`) re-checked after each loop turn. An
    /// empty list (the default) means "no completion check — run until the budget
    /// is spent", preserving the prior behavior. When every check passes the loop
    /// is genuinely done and stops, instead of repeating to its cap.
    until: Vec<GoalValidator>,
    /// Stall tracker for the `--until` condition (the same anti-no-progress brain
    /// as `/goal`). Each post-turn `--until` failure folds its objective signature
    /// here; when the SAME failure repeats with no progress the loop stops instead
    /// of firing forever. Only meaningful for recurring `--until` loops.
    progress: ProgressTracker,
    /// External-blocker detector for the `--until` condition (the same
    /// escalate-to-the-human brain as `/goal`). Keys on the failure *class*, so
    /// it fires even when the failing check's surface text drifts run to run
    /// and the identical-failure stall above cannot. Not persisted (mirrors the
    /// goal's tracker): a restored loop restarts the streak from zero.
    blocks: BlockTracker,
    /// `--allow-writes` opt-in: when `true`, this loop's unattended turns inherit
    /// the session's write permission (each queued prompt embeds the allow-writes
    /// marker). Default `false` = read-only + propose-only.
    allow_writes: bool,
    kind: LoopKind,
}

impl LoopState {
    /// Whether a recurring loop's resource budget (`--max-runs` / `--token-budget`)
    /// is already spent, so a further run would exceed it. `run_count` here is the
    /// number of runs already completed (the prospective tick is not yet charged),
    /// matching the pre-increment check the pop-gate uses. An unbounded budget
    /// never returns `true`.
    fn recurring_budget_spent(&self) -> bool {
        BudgetLedger {
            turns: self.run_count,
            output_tokens: self.output_tokens,
        }
        .exhaustion(&self.budget)
        .is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopStatus {
    Active,
    Paused,
    Stopped,
    Completed,
}

#[derive(Debug, Clone)]
enum LoopKind {
    FixedCount {
        count: u32,
    },
    Interval {
        every: Duration,
        next_due: Instant,
    },
    Watch {
        glob: String,
        snapshot: BTreeMap<String, Option<SystemTime>>,
        next_poll: Instant,
    },
}

impl LoopKind {
    fn label(&self) -> String {
        match self {
            Self::FixedCount { count } => format!("fixed-count ({count} run budget)"),
            Self::Interval { every, .. } => {
                format!("interval every {}", humantime_duration(*every))
            }
            Self::Watch { glob, .. } => format!("watch {glob}"),
        }
    }

    fn stop_policy(&self, active_recurring: usize) -> String {
        match self {
            Self::FixedCount { count } => {
                format!("/loop stop or /loop pause; otherwise after {count} queued run(s)")
            }
            Self::Interval { .. } | Self::Watch { .. } => format!(
                "/loop stop, /loop pause, or session exit; active recurring budget {active_recurring}/{MAX_ACTIVE_SESSION_LOOPS}"
            ),
        }
    }
}

/// One-line summary of a recurring loop's optional `--max-runs` / `--token-budget`
/// caps, appended to the "Loop created" report. Empty when the loop is unbounded.
fn recurring_budget_note(budget: LoopBudget) -> String {
    match (budget.max_turns, budget.max_output_tokens) {
        (0, None) => String::new(),
        (runs, None) => format!("\n  Max runs         {runs}"),
        (0, Some(tokens)) => format!("\n  Token budget     {tokens}"),
        (runs, Some(tokens)) => format!("\n  Max runs         {runs}\n  Token budget     {tokens}"),
    }
}

fn humantime_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs.is_multiple_of(86_400) {
        format!("{}d", secs / 86_400)
    } else if secs.is_multiple_of(3_600) {
        format!("{}h", secs / 3_600)
    } else if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

fn format_loop_prompt(
    id: &str,
    run: u32,
    total: Option<u32>,
    prompt: &str,
    changed: Option<&[String]>,
    allow_writes: bool,
) -> String {
    let run_label = total.map_or_else(|| run.to_string(), |total| format!("{run}/{total}"));
    let mut body = format!(
        "Session loop {id} run {run_label}. Execute this prompt and summarize the result:\n\nLoop engineering contract: Discover → Plan → Execute → Verify → Iterate. Treat this as one bounded iteration, validate before claiming completion, and summarize the observation for the next run.\n\n{prompt}"
    );
    if let Some(changed) = changed.filter(|changed| !changed.is_empty()) {
        let _ = write!(body, "\n\nChanged files:\n- {}", changed.join("\n- "));
    }
    plan_first_automation_prompt("loop", &body, allow_writes)
}

/// Pull leading `--max-runs N` / `--token-budget N` flags off a recurring loop's
/// prompt, returning the parsed [`LoopBudget`] and the remaining prompt text.
///
/// The slash parser hands a recurring loop's prompt through verbatim (it is free
/// text), so these resource flags are parsed here at the start of the prompt
/// rather than in the `commands` crate's `LoopCommand` contract. Only flags at
/// the very front are consumed (`--max-runs 5 watch the deploy`); anything that
/// is not a recognized leading flag ends the scan and stays in the prompt, so a
/// prompt that legitimately mentions `--max-runs` later is untouched. An invalid
/// or non-positive value is left in the prompt (fail-open: never silently drop a
/// budget — an unparsed flag just means no cap, the prior behavior).
fn split_loop_budget_flags(prompt: &str) -> (LoopBudget, Vec<GoalValidator>, bool, String) {
    // Recurring loops start bounded: an omitted `--max-runs` defaults to a finite
    // cap rather than "run forever". An explicit `--max-runs N` below overrides it.
    let mut budget = LoopBudget {
        max_turns: DEFAULT_RECURRING_MAX_RUNS,
        max_output_tokens: None,
    };
    let mut until: Vec<GoalValidator> = Vec::new();
    // `--allow-writes` opts out of the unattended read-only default (see the
    // permission gate). A bare boolean flag with no value.
    let mut allow_writes = false;
    // A borrowed cursor that always points into `prompt`, so no allocation is
    // needed until the final remainder is cloned out.
    let mut rest = prompt.trim_start();
    loop {
        let (flag, after) = split_first_word(rest);
        // Parse the flag's value word: a positive `u32`, plus the prompt tail after
        // it. `None` (missing / non-numeric / zero) ends the scan and leaves the
        // flag in the prompt rather than consuming a bogus budget.
        let parse_value = || -> Option<(u32, &str)> {
            let (word, tail) = split_first_word(after);
            word.parse::<u32>()
                .ok()
                .filter(|&n| n > 0)
                .map(|n| (n, tail))
        };
        match flag {
            "--max-runs" => {
                let Some((runs, tail)) = parse_value() else {
                    break;
                };
                budget.max_turns = runs;
                rest = tail;
            }
            "--token-budget" => {
                let Some((tokens, tail)) = parse_value() else {
                    break;
                };
                budget.max_output_tokens = Some(u64::from(tokens));
                rest = tail;
            }
            "--until" => {
                // One objective check word (e.g. `grep:DONE`, `cargo:test`); an
                // empty/unparseable value leaves the flag in the prompt (fail-open).
                let (word, tail) = split_first_word(after);
                let Some(validator) = parse_until_validator(word) else {
                    break;
                };
                until.push(validator);
                rest = tail;
            }
            "--allow-writes" => {
                allow_writes = true;
                rest = after;
            }
            _ => break,
        }
    }
    (budget, until, allow_writes, rest.trim_start().to_string())
}

/// Parse one `--until <check>` value into an *objective* validator. Mirrors the
/// `/goal --check` grammar but rejects a bare rubric label: a loop completion
/// condition must be deterministically checkable (`grep:` / `cargo:` /
/// `git:diff-check`), never a model-graded rubric (`None`).
fn parse_until_validator(word: &str) -> Option<GoalValidator> {
    if word.is_empty() {
        return None;
    }
    match parse_validators(std::slice::from_ref(&word.to_string())).into_iter().next() {
        Some(GoalValidator::ModelRubric { .. }) | None => None,
        Some(validator) => Some(validator),
    }
}

fn split_first_word(input: &str) -> (&str, &str) {
    let trimmed = input.trim_start();
    match trimmed.split_once(char::is_whitespace) {
        Some((first, rest)) => (first, rest.trim_start()),
        None => (trimmed, ""),
    }
}

/// Pull a single leading `--allow-writes` opt-in flag off a loop prompt, returning
/// `(allow_writes, remaining_prompt)`. Used by the fixed-count path, which takes
/// its prompt verbatim and so never runs [`split_loop_budget_flags`]. Only a flag
/// at the very front is consumed; a prompt that merely mentions `--allow-writes`
/// later is untouched.
fn strip_leading_allow_writes(prompt: &str) -> (bool, String) {
    let trimmed = prompt.trim_start();
    let (first, rest) = split_first_word(trimmed);
    if first == "--allow-writes" {
        (true, rest.trim_start().to_string())
    } else {
        (false, trimmed.to_string())
    }
}

fn collect_watch_snapshot(cwd: &Path, pattern: &str) -> BTreeMap<String, Option<SystemTime>> {
    let mut snapshot = BTreeMap::new();
    let mut stack = vec![cwd.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if snapshot.len() >= MAX_WATCH_FILES {
            break;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == ".git" || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let rel = relative_unix(cwd, &path);
            if glob_match(pattern, &rel) {
                let modified = entry
                    .metadata()
                    .ok()
                    .and_then(|metadata| metadata.modified().ok());
                snapshot.insert(rel, modified);
            }
        }
    }
    snapshot
}

fn changed_files(
    old: &BTreeMap<String, Option<SystemTime>>,
    new: &BTreeMap<String, Option<SystemTime>>,
) -> Vec<String> {
    let mut changed = Vec::new();
    for (path, modified) in new {
        if old.get(path) != Some(modified) {
            changed.push(path.clone());
        }
    }
    for path in old.keys() {
        if !new.contains_key(path) {
            changed.push(path.clone());
        }
    }
    changed.sort();
    changed
}

fn relative_unix(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd)
        .unwrap_or(path)
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern = pattern.trim_start_matches("./");
    let path = path.trim_start_matches("./");
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();
    glob_match_parts(&pattern_parts, &path_parts)
}

// The `(None, Some)` and `(Some, None)` arms share a `false` body but cannot be
// merged: the `**` arm between them must keep matching a `None` path first, so
// reordering to combine them would change the glob semantics.
#[allow(clippy::match_same_arms)]
fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    match (pattern.split_first(), path.split_first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some((&"**", rest)), _) => {
            glob_match_parts(rest, path)
                || path
                    .split_first()
                    .is_some_and(|(_, path_rest)| glob_match_parts(pattern, path_rest))
        }
        (Some((segment, rest)), Some((path_segment, path_rest))) => {
            segment_match(segment, path_segment) && glob_match_parts(rest, path_rest)
        }
        (Some(_), None) => false,
    }
}

fn segment_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut star_text = 0usize;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            star_text = ti;
            pi += 1;
        } else if let Some(star_pi) = star {
            pi = star_pi + 1;
            star_text += 1;
            ti = star_text;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn truncate_label(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

pub(crate) mod persist;

#[cfg(test)]
mod tests;
