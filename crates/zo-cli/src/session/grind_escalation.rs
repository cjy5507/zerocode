//! Grind escalation: raise intelligence — not another warning — when a session
//! keeps exhausting turn budgets on the same task.
//!
//! The runaway circuit breaker ends a grinding turn *gracefully* (see
//! [`runtime::BudgetExhausted`]), so a user who answers "계속"/"continue"
//! simply re-arms the same approach at the same effort and the cycle repeats —
//! the push-session incident re-planned one deploy for hours this way, one
//! budget-exhausted turn after another. Warnings are the wrong lever there:
//! each round genuinely progresses, so the churn/repetition guards rightly
//! stay quiet. What helps is a smarter turn. After
//! [`grind_escalation_threshold`] consecutive budget-exhausted turns, the next
//! turn (a) runs with its reasoning effort floored at `xhigh` and (b) carries
//! a strategy-review directive to batch-discover the remaining failures
//! instead of resuming the fix-one-then-reverify cycle. The same signal
//! separately feeds the one-step WI-B route escalation
//! (`auto_fanout::decide_escalation`, see
//! `LiveCli::record_turn_budget_exhausted`).
//!
//! Scope: the interactive REPL turn path (`turn_controller`). The streak is
//! session-scoped and in-memory — a `/restart` starts fresh.

use api::EffortLevel;

/// Transient wire-reminder prefix (replace-by-prefix, see
/// `reminders::replace_transient_system_reminder_by_prefix`). Set-or-cleared
/// at every turn entry, so a prior turn's directive never lingers into a turn
/// that is not grinding.
pub(crate) const GRIND_ESCALATION_REMINDER_PREFIX: &str = "[zo:grind-escalation]";

/// Consecutive budget-exhausted turns before escalation arms. The first
/// exhaustion is often legitimate scale; the second in a row is a pattern.
const GRIND_ESCALATION_DEFAULT_THRESHOLD: u32 = 2;

/// The armed threshold, or `None` when the guard is disabled.
/// `ZO_GRIND_ESCALATION` overrides the default (`0` disables; an empty or
/// non-numeric value falls back rather than silently disabling). Read per turn
/// (not memoized) so an operator can retune without a rebuild — the
/// escape-hatch idiom of `ZO_VERIFY_TREADMILL_ROUNDS`.
pub(crate) fn grind_escalation_threshold() -> Option<u32> {
    threshold_from(std::env::var("ZO_GRIND_ESCALATION").ok().as_deref())
}

fn threshold_from(raw: Option<&str>) -> Option<u32> {
    let threshold = raw
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .unwrap_or(GRIND_ESCALATION_DEFAULT_THRESHOLD);
    (threshold > 0).then_some(threshold)
}

/// Whether `streak` consecutive budget-exhausted turns arm escalation for the
/// coming turn.
pub(crate) fn armed(streak: u32) -> bool {
    armed_for(streak, grind_escalation_threshold())
}

fn armed_for(streak: u32, threshold: Option<u32>) -> bool {
    threshold.is_some_and(|threshold| streak >= threshold)
}

/// The directive for the coming turn, `None` when the streak is zero or the
/// guard is disabled. A two-step ladder: one exhausted turn arms the
/// *checkpoint* directive (report-then-ask before resuming — a noncommittal
/// reply like "응?" must not silently re-arm heavy execution, the free-session
/// incident); reaching the threshold arms the full *escalation* directive
/// alongside the effort floor.
pub(crate) fn reminder(streak: u32) -> Option<String> {
    reminder_for(streak, grind_escalation_threshold())
}

fn reminder_for(streak: u32, threshold: Option<u32>) -> Option<String> {
    if threshold.is_none() || streak == 0 {
        return None;
    }
    Some(if armed_for(streak, threshold) {
        reminder_text(streak)
    } else {
        checkpoint_text()
    })
}

fn checkpoint_text() -> String {
    format!(
        "{GRIND_ESCALATION_REMINDER_PREFIX} <system-reminder>The previous turn was stopped by a \
         turn-budget breaker (time/tokens), not by finishing. Unless the user's message clearly \
         and explicitly asks to continue executing, do NOT resume heavy execution: first report \
         in a few lines what was completed, what remains, and what (if anything) is blocking, \
         then ask how to proceed. If the user does ask to continue, do not repeat the previous \
         cycle unchanged — state in one line what you will do differently.</system-reminder>"
    )
}

fn reminder_text(streak: u32) -> String {
    format!(
        "{GRIND_ESCALATION_REMINDER_PREFIX} <system-reminder>The previous {streak} consecutive \
         turns each ended by exhausting their execution budget (time/tokens) on this task, so \
         the current fix-one-then-reverify cycle is grinding, not converging. This turn runs at \
         escalated reasoning effort. Step back and re-plan before resuming: (1) batch-discover \
         ALL remaining failures in one pass — run the full pipeline/deployment to completion \
         collecting every error, not just the first; (2) apply the fixes as one batch — or, if \
         the fix APPROACH itself is what keeps failing, run 2-3 independent attempts in parallel \
         (a Workflow judge panel) and keep the best instead of iterating on one; (3) run the \
         expensive verification suite ONCE at the end, not after every small fix. If the goal \
         may be unachievable as stated, or a decision is the user's to make, stop and report \
         instead of repeating the cycle.</system-reminder>"
    )
}

/// Marker carried by a synthesized auto-continue prompt, so the next turn can
/// be told apart from a user-typed message (a user-typed turn starts a fresh
/// auto-continue chain).
pub(crate) const AUTO_CONTINUE_MARKER: &str = "[zo:auto-continue]";

/// Consecutive automatic continuations allowed per chain. Two continuations on
/// top of the original turn give a legitimately long task three full turn
/// budgets unattended; anything still unfinished then genuinely needs the
/// user.
const AUTO_CONTINUE_DEFAULT_CAP: u32 = 2;

/// The auto-continue cap, or `None` when disabled. `ZO_AUTO_CONTINUE`
/// overrides (`0` disables); non-numeric falls back to the default.
pub(crate) fn auto_continue_cap() -> Option<u32> {
    let cap = std::env::var("ZO_AUTO_CONTINUE")
        .ok()
        .and_then(|raw| raw.trim().parse::<u32>().ok())
        .unwrap_or(AUTO_CONTINUE_DEFAULT_CAP);
    (cap > 0).then_some(cap)
}

/// Whether a budget-exhausted turn should automatically continue instead of
/// waiting for the user to type "계속". All of:
///
/// - the stop was a plain budget (never [`VerificationTreadmill`] — that stop
///   exists precisely to hand back to the user),
/// - the turn externalized progress (successful edit/write/plan results — the
///   same evidence class as the in-turn deadline extension),
/// - the chain has continuations left, and
/// - the grind streak is at most the escalation threshold (the continuation AT
///   the threshold runs escalated — the "smartest retry" — and past it the
///   session must stop and ask).
///
/// Pure over its inputs; env is read by the caller via [`auto_continue_cap`]
/// and [`grind_escalation_threshold`].
pub(crate) fn should_auto_continue(
    kind: runtime::BudgetExhausted,
    progress_results: usize,
    chain: u32,
    cap: Option<u32>,
    streak: u32,
    threshold: Option<u32>,
) -> bool {
    if matches!(kind, runtime::BudgetExhausted::VerificationTreadmill) {
        return false;
    }
    let Some(cap) = cap else { return false };
    let Some(threshold) = threshold else {
        return false;
    };
    progress_results > 0 && chain < cap && streak <= threshold
}

/// The synthesized continuation prompt for an auto-continued turn.
pub(crate) fn auto_continue_prompt() -> String {
    format!(
        "{AUTO_CONTINUE_MARKER} The previous turn hit its execution budget while making real \
         progress. Continue exactly where it stopped and finish the remaining work — do not redo \
         completed steps. If everything is already complete, summarize the outcome and stop."
    )
}

/// One-turn effort escalation: floor the turn's effort at `xhigh` when armed.
/// A user pin of `max`/`ultra` is never lowered; any smart-band ceiling is
/// lifted alongside (a ceiling below `xhigh` would silently undo the floor).
/// Pure over `(armed, named, ceiling)` so the mapping is testable without env
/// or client plumbing.
pub(crate) fn effective_turn_effort(
    armed: bool,
    named: Option<EffortLevel>,
    ceiling: Option<EffortLevel>,
) -> (Option<EffortLevel>, Option<EffortLevel>) {
    if !armed {
        return (named, ceiling);
    }
    match named {
        Some(EffortLevel::Max | EffortLevel::Ultra) => (named, ceiling),
        _ => (Some(EffortLevel::Xhigh), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_defaults_to_two_and_zero_disables() {
        assert_eq!(threshold_from(None), Some(GRIND_ESCALATION_DEFAULT_THRESHOLD));
        assert_eq!(threshold_from(Some("0")), None);
        assert_eq!(threshold_from(Some("3")), Some(3));
        assert_eq!(threshold_from(Some(" 4 ")), Some(4));
        // Non-numeric falls back to the default rather than silently disabling
        // the safety net.
        assert_eq!(
            threshold_from(Some("junk")),
            Some(GRIND_ESCALATION_DEFAULT_THRESHOLD)
        );
        assert_eq!(
            threshold_from(Some("")),
            Some(GRIND_ESCALATION_DEFAULT_THRESHOLD)
        );
    }

    #[test]
    fn effort_floors_at_xhigh_and_lifts_the_ceiling_when_armed() {
        use EffortLevel::*;
        // Unarmed: everything passes through untouched.
        assert_eq!(
            effective_turn_effort(false, Some(High), Some(Medium)),
            (Some(High), Some(Medium))
        );
        // Armed: any effort at or below xhigh is floored to xhigh, and the
        // band ceiling is lifted so it cannot undo the floor.
        assert_eq!(effective_turn_effort(true, None, None), (Some(Xhigh), None));
        assert_eq!(
            effective_turn_effort(true, Some(Low), None),
            (Some(Xhigh), None)
        );
        assert_eq!(
            effective_turn_effort(true, Some(High), Some(Medium)),
            (Some(Xhigh), None)
        );
        assert_eq!(
            effective_turn_effort(true, Some(Xhigh), Some(High)),
            (Some(Xhigh), None)
        );
    }

    #[test]
    fn effort_never_lowers_a_max_or_ultra_pin() {
        use EffortLevel::*;
        assert_eq!(
            effective_turn_effort(true, Some(Max), None),
            (Some(Max), None)
        );
        assert_eq!(
            effective_turn_effort(true, Some(Ultra), Some(Max)),
            (Some(Ultra), Some(Max))
        );
    }

    #[test]
    fn reminder_carries_the_prefix_streak_and_batch_directive() {
        let text = reminder_text(3);
        assert!(text.starts_with(GRIND_ESCALATION_REMINDER_PREFIX), "{text:?}");
        assert!(text.contains("previous 3 consecutive"), "{text:?}");
        assert!(text.contains("batch-discover"), "{text:?}");
        assert!(text.contains("ONCE at the end"), "{text:?}");
    }

    #[test]
    fn auto_continue_requires_progress_chain_room_and_a_sane_streak() {
        use runtime::BudgetExhausted::{Deadline, VerificationTreadmill};
        let cap = Some(2);
        let threshold = Some(2);
        // The healthy case: budget stop + progress + room in the chain.
        assert!(should_auto_continue(Deadline, 3, 0, cap, 1, threshold));
        // At the escalation threshold the continuation still runs (escalated).
        assert!(should_auto_continue(Deadline, 3, 1, cap, 2, threshold));
        // Past the threshold the session must stop and ask.
        assert!(!should_auto_continue(Deadline, 3, 1, cap, 3, threshold));
        // No externalized progress → no silent continuation.
        assert!(!should_auto_continue(Deadline, 0, 0, cap, 1, threshold));
        // Chain exhausted → hand back to the user.
        assert!(!should_auto_continue(Deadline, 3, 2, cap, 1, threshold));
        // The treadmill stop exists to hand back — never auto-continue it.
        assert!(!should_auto_continue(
            VerificationTreadmill,
            3,
            0,
            cap,
            1,
            threshold
        ));
        // Either guard disabled → off.
        assert!(!should_auto_continue(Deadline, 3, 0, None, 1, threshold));
        assert!(!should_auto_continue(Deadline, 3, 0, cap, 1, None));
    }

    #[test]
    fn auto_continue_prompt_carries_the_marker_and_no_redo_directive() {
        let prompt = auto_continue_prompt();
        assert!(prompt.starts_with(AUTO_CONTINUE_MARKER), "{prompt:?}");
        assert!(prompt.contains("do not redo"), "{prompt:?}");
    }

    #[test]
    fn reminder_ladder_steps_from_none_to_checkpoint_to_escalation() {
        let threshold = Some(2);
        // Streak 0 → no directive at all.
        assert_eq!(reminder_for(0, threshold), None);
        // Streak 1 (below threshold) → checkpoint directive: report-then-ask,
        // never a silent resume on a noncommittal reply.
        let checkpoint = reminder_for(1, threshold).expect("checkpoint step");
        assert!(checkpoint.starts_with(GRIND_ESCALATION_REMINDER_PREFIX), "{checkpoint:?}");
        assert!(checkpoint.contains("report"), "{checkpoint:?}");
        assert!(checkpoint.contains("ask how to proceed"), "{checkpoint:?}");
        // At threshold → full strategy-review escalation.
        let escalation = reminder_for(2, threshold).expect("escalation step");
        assert!(escalation.contains("batch-discover"), "{escalation:?}");
        // Disabled guard → nothing, regardless of streak.
        assert_eq!(reminder_for(5, None), None);
    }
}
