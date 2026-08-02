//! Turn-end gate: the harness-side "check your last paragraph" lint.
//!
//! The system prompt's turn discipline asks the model to do promised work
//! before ending the turn; strong models comply, weaker ones end on
//! "I'll do X next." and hand the promise back to the user. This module makes
//! the contract enforceable at the only place it can be observed — the natural
//! end of a streaming turn (no tool calls in the final assistant message):
//! when the reply's last paragraph is a promise of future work, or (on an
//! autonomous surface) a question nobody is present to answer, the loop
//! re-prompts with a bounded reminder instead of ending the turn.
//!
//! Deterministic and conservative by design: a missed promise costs the user
//! one manual nudge, while a false positive costs a whole extra model
//! iteration — so the marker lists are tight, and every "waiting on the user"
//! phrasing is an explicit skip. Complements (never replaces) the budget
//! breakers: the gate only ever runs at a natural end, so a turn stopped by
//! deadline/token/treadmill closers is never re-prompted.

/// Why the gate wants one more iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnEndingIssue {
    /// The final paragraph promises work the model has not done ("I'll …").
    Promise,
    /// The final paragraph is a question, and nobody can answer mid-run.
    Question,
}

/// Hard cap on gate re-prompts per turn, so a model that keeps promising
/// (or genuinely needs the user) cannot be looped indefinitely.
pub(super) const TURN_END_GATE_MAX_REPROMPTS: usize = 2;

/// Gate level from `ZO_TURN_END_GATE`: `0` disables, `1` (default) lints
/// promise endings everywhere and question endings on autonomous surfaces,
/// `2` also lints question endings in interactive sessions. Read per turn
/// (not memoized) so an operator can retune without a rebuild — the
/// escape-hatch idiom of `ZO_MAX_ITERATIONS`.
pub(super) fn env_turn_end_gate_level() -> u8 {
    std::env::var("ZO_TURN_END_GATE")
        .ok()
        .and_then(|raw| raw.trim().parse::<u8>().ok())
        .unwrap_or(1)
}

/// Promise markers: first-person future commitments. Matched against the last
/// paragraph only — a promise earlier in the reply followed by a completed
/// summary is fine.
const PROMISE_MARKERS: [&str; 12] = [
    "I'll ",
    "I will ",
    "I'm going to ",
    "Let me now ",
    "Next, I'll",
    "Next I'll",
    "하겠습니다",
    "해보겠습니다",
    "할게요",
    "진행하겠",
    "시작하겠",
    "할 예정입니다",
];

/// Phrasings that mean the model is legitimately stopping or handing the
/// decision to the user. Any of these in the last paragraph suppresses the
/// promise lint — ending on "I'll wait for your decision" is the *correct*
/// blocked-on-user ending, not a broken promise.
const WAITING_ON_USER_MARKERS: [&str; 16] = [
    "I'll wait",
    "I'll stop",
    "I'll hold",
    "I'll leave",
    "let me know",
    "your call",
    "if you want",
    "if you'd like",
    "once you confirm",
    "알려주세요",
    "알려 주세요",
    "말씀해 주세요",
    "말씀해주세요",
    "확인해 주세요",
    "기다리겠",
    "중단하겠",
];

/// The last paragraph of a reply: the text after the final blank line,
/// trimmed. Falls back to the whole (trimmed) text when there is no blank
/// line, so single-paragraph replies are still screened.
fn last_paragraph(text: &str) -> &str {
    let trimmed = text.trim_end();
    trimmed
        .rsplit("\n\n")
        .next()
        .unwrap_or(trimmed)
        .trim()
}

/// Screen a naturally-ending reply's final paragraph. `lint_questions` is true
/// when nobody can answer a mid-run question (autonomous surface, or gate
/// level 2). Returns `None` when the ending is fine.
pub(super) fn screen_turn_ending(
    final_text: &str,
    lint_questions: bool,
) -> Option<TurnEndingIssue> {
    let paragraph = last_paragraph(final_text);
    if paragraph.is_empty() {
        // Empty replies belong to the empty-stream retry machinery.
        return None;
    }
    if lint_questions
        && paragraph
            .trim_end_matches(['*', '`', ')', ']', '"', '\''])
            .ends_with(['?', '？'])
    {
        return Some(TurnEndingIssue::Question);
    }
    if WAITING_ON_USER_MARKERS
        .iter()
        .any(|marker| paragraph.contains(marker))
    {
        return None;
    }
    if PROMISE_MARKERS
        .iter()
        .any(|marker| paragraph.contains(marker))
    {
        return Some(TurnEndingIssue::Promise);
    }
    None
}

/// The reminder folded in as a fresh user message when the gate fires. A user
/// message (not a wire reminder) so the re-prompt survives replay on every
/// provider and reads as an explicit course correction.
pub(super) fn turn_end_gate_reminder(issue: TurnEndingIssue) -> &'static str {
    match issue {
        TurnEndingIssue::Promise => {
            "[zo:turn-end-gate] <system-reminder>Your reply ended by promising work you have \
             not done yet. Do not end the turn on a promise: do that work now with tool calls — \
             including retrying after errors and gathering missing information yourself. If you \
             are genuinely blocked on input only the user can provide, say exactly what you need \
             and why instead. Then finish with a complete final answer.</system-reminder>"
        }
        TurnEndingIssue::Question => {
            "[zo:turn-end-gate] <system-reminder>Your reply ended with a question, but nobody \
             is present to answer it mid-run. Make the most reasonable assumption, state it \
             explicitly, and continue the work with tool calls. Only end on a question if the \
             task truly cannot proceed without the user's decision — and then say exactly what \
             input is needed and what you completed.</system-reminder>"
        }
    }
}

/// Short transcript banner shown when the gate re-prompts, so the extra
/// iteration is never a silent mystery.
pub(super) fn turn_end_gate_banner(issue: TurnEndingIssue) -> &'static str {
    match issue {
        TurnEndingIssue::Promise => {
            "[turn-gate] reply ended on a promise — asking the model to do the work now"
        }
        TurnEndingIssue::Question => {
            "[turn-gate] reply ended on a question nobody can answer — asking the model to proceed"
        }
    }
}

impl<C, T> super::ConversationRuntime<C, T>
where
    C: super::ApiClient,
    T: super::ToolExecutor,
{
    /// Evaluate the gate against a naturally-ending assistant message. Returns
    /// the issue to re-prompt on, bumping `reprompts`, or `None` when the turn
    /// may end. Reads the env level per call so the knob works without a
    /// rebuild; bounded by [`TURN_END_GATE_MAX_REPROMPTS`] per turn.
    pub(super) fn take_turn_end_gate_issue(
        &self,
        final_text: &str,
        reprompts: &mut usize,
    ) -> Option<TurnEndingIssue> {
        let level = env_turn_end_gate_level();
        if level == 0 || *reprompts >= TURN_END_GATE_MAX_REPROMPTS {
            return None;
        }
        let lint_questions = self.autonomous_surface || level >= 2;
        let issue = screen_turn_ending(final_text, lint_questions)?;
        *reprompts += 1;
        Some(issue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promise_ending_is_flagged() {
        let text = "Found the bug in the parser.\n\nNext, I'll fix the tokenizer and rerun the tests.";
        assert_eq!(
            screen_turn_ending(text, false),
            Some(TurnEndingIssue::Promise)
        );
    }

    #[test]
    fn korean_promise_ending_is_flagged() {
        let text = "원인을 찾았습니다.\n\n이제 수정을 진행하겠습니다.";
        assert_eq!(
            screen_turn_ending(text, false),
            Some(TurnEndingIssue::Promise)
        );
    }

    #[test]
    fn completed_report_is_not_flagged() {
        let text = "Fixed the tokenizer and reran the tests — all 42 pass.\n\nThe root cause was an off-by-one in the span math.";
        assert_eq!(screen_turn_ending(text, false), None);
    }

    #[test]
    fn earlier_promise_with_completed_ending_is_not_flagged() {
        let text = "I'll start with the parser.\n\nDone: parser fixed, tests green.";
        assert_eq!(screen_turn_ending(text, false), None);
    }

    #[test]
    fn waiting_on_user_is_not_flagged() {
        let text = "Two viable designs exist.\n\nI'll wait for your decision — let me know which one.";
        assert_eq!(screen_turn_ending(text, false), None);
        let korean = "설계가 두 가지입니다.\n\n어느 쪽으로 갈지 알려주세요.";
        assert_eq!(screen_turn_ending(korean, false), None);
    }

    #[test]
    fn question_ending_only_flagged_when_linting_questions() {
        let text = "The migration is ready.\n\nShould I also update the staging config?";
        assert_eq!(screen_turn_ending(text, false), None);
        assert_eq!(
            screen_turn_ending(text, true),
            Some(TurnEndingIssue::Question)
        );
    }

    #[test]
    fn question_lint_sees_through_trailing_markup() {
        let text = "다음 단계로 갈까요?**";
        assert_eq!(
            screen_turn_ending(text, true),
            Some(TurnEndingIssue::Question)
        );
    }

    #[test]
    fn empty_text_is_ignored() {
        assert_eq!(screen_turn_ending("", true), None);
        assert_eq!(screen_turn_ending("\n\n", true), None);
    }
}
