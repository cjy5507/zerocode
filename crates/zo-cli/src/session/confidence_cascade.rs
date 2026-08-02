//! Confidence cascade: escalate the NEXT turn to a stronger configuration
//! when the model itself reports low confidence in what it just produced.
//!
//! The jacobian-lens principle behind this: a model's verbalized read of its
//! own state is a cheap, faithful predictor of outcome quality (the paper's
//! "verbalizable representations form a global workspace" claim) — so a
//! turn-end `low` self-report is treated as a routing signal, exactly like
//! the grind streak treats budget exhaustion.
//!
//! The contract is asymmetric on purpose: the model appends one marker line
//! ONLY when not confident (see [`contract_reminder`]), so a confident turn
//! costs zero output tokens. The harness parses the marker at turn end
//! ([`parse_turn_confidence`]), arms the cascade, and the following turn runs
//! with (a) its effort floored at `xhigh` (the grind ladder's
//! `effective_turn_effort`), (b) a re-approach directive, and (c) when Smart
//! can route one, a same-provider Deep-tier wire-model escalation
//! (`ConversationRuntime::set_escalation_model_override`).
//!
//! Scope: the interactive REPL turn path (`turn_controller`). Armed state is
//! session-scoped and in-memory — a `/restart` starts fresh.

/// The marker the model appends as its FINAL line when it lacks confidence.
/// Grammar: `[zo:turn-confidence] low|medium|high — <one line why>`.
/// The literal lives with the renderer (which restyles it as a dim chip) so
/// the two can never drift.
pub(crate) use zo_cli::tui::markdown::TURN_CONFIDENCE_MARKER;

/// Transient wire-reminder prefix for the marker contract (replace-by-prefix,
/// set-or-cleared at every turn entry).
pub(crate) const CONFIDENCE_CONTRACT_REMINDER_PREFIX: &str = "[zo:turn-confidence-contract]";

/// Transient wire-reminder prefix for the escalated turn's directive.
pub(crate) const CASCADE_ESCALATION_REMINDER_PREFIX: &str = "[zo:confidence-cascade]";

/// The model's verbalized turn-end confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnConfidence {
    Low,
    Medium,
    High,
}

/// Whether the cascade is enabled. `ZO_CONFIDENCE_CASCADE=0` disables;
/// missing/other values leave it on (same idiom as `ZO_GRIND_ESCALATION`).
/// Read per turn so an operator can retune without a rebuild.
pub(crate) fn cascade_enabled() -> bool {
    std::env::var("ZO_CONFIDENCE_CASCADE")
        .ok()
        .is_none_or(|raw| raw.trim() != "0")
}

/// Parse the model's turn-end confidence readout from its final text.
///
/// The contract says FINAL line, and that is enforced literally: only the
/// last non-empty line is considered. A marker quoted mid-text (a model
/// explaining the contract with a line-start example, then closing with
/// confident prose) must not false-arm the cascade; a model that appends
/// prose after its marker forfeits the readout — the asymmetric contract
/// makes silence the cheap, safe default. Unknown tokens after the marker
/// parse as `None`, never as a guess.
pub(crate) fn parse_turn_confidence(text: &str) -> Option<TurnConfidence> {
    let line = final_marker_line(text)?;
    let rest = line[TURN_CONFIDENCE_MARKER.len()..].trim_start();
    let token: String = rest
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_ascii_lowercase();
    match token.as_str() {
        "low" => Some(TurnConfidence::Low),
        "medium" => Some(TurnConfidence::Medium),
        "high" => Some(TurnConfidence::High),
        _ => None,
    }
}

/// Whether a turn-end readout arms the cascade for the coming turn.
pub(crate) fn should_arm(readout: Option<TurnConfidence>) -> bool {
    readout == Some(TurnConfidence::Low)
}

/// The standing marker contract, taught as a transient reminder while the
/// cascade is enabled. Asymmetric: emit only on low confidence, so the
/// common confident turn costs nothing.
pub(crate) fn contract_reminder() -> String {
    format!(
        "{CONFIDENCE_CONTRACT_REMINDER_PREFIX} <system-reminder>If — and ONLY if — you finish \
         this turn NOT confident that your result is correct and complete (unverified edits, \
         guesses about APIs, failing or skipped tests, an approach you doubt), append ONE final \
         line, exactly: {TURN_CONFIDENCE_MARKER} low — <one short reason>. Do not add this line \
         when you are confident; never add it to purely conversational replies. It judges work \
         YOU completed THIS turn — do not add it to turns that merely report progress on or \
         wait for still-running background agents/workflows (the harness already tracks \
         those).</system-reminder>"
    )
}

/// The directive for the escalated turn following a low-confidence readout.
pub(crate) fn escalation_reminder(reason_hint: Option<&str>) -> String {
    let reason = reason_hint
        .map(|reason| format!(" Its stated reason: {reason}."))
        .unwrap_or_default();
    format!(
        "{CASCADE_ESCALATION_REMINDER_PREFIX} <system-reminder>The previous turn ended with the \
         model reporting LOW confidence in its own result.{reason} This turn runs escalated \
         (stronger reasoning). Do not simply continue: first re-derive what the previous turn \
         was uncertain about, verify or falsify it against the actual code/tests, then either \
         repair the result or state precisely why it was already correct.</system-reminder>"
    )
}

/// The last non-empty line when — and only when — it carries the marker.
fn final_marker_line(text: &str) -> Option<&str> {
    text.lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .filter(|line| line.starts_with(TURN_CONFIDENCE_MARKER))
}

/// The free-text reason after the confidence token, for the escalated turn's
/// directive and the HUD banner. Same line-selection rule as
/// [`parse_turn_confidence`].
pub(crate) fn parse_confidence_reason(text: &str) -> Option<String> {
    let line = final_marker_line(text)?;
    let rest = line[TURN_CONFIDENCE_MARKER.len()..].trim_start();
    let reason = rest
        .trim_start_matches(|ch: char| ch.is_ascii_alphabetic())
        .trim_start_matches([' ', '—', '-', ':', '–'])
        .trim();
    (!reason.is_empty()).then(|| {
        // Bounded: this is re-injected into a reminder, not an essay slot.
        reason.chars().take(200).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_final_low_marker_with_reason() {
        let text = "작업을 마쳤지만 확신이 없습니다.\n\n[zo:turn-confidence] low — verify를 못 돌렸음";
        assert_eq!(parse_turn_confidence(text), Some(TurnConfidence::Low));
        assert_eq!(
            parse_confidence_reason(text).as_deref(),
            Some("verify를 못 돌렸음")
        );
        assert!(should_arm(parse_turn_confidence(text)));
    }

    #[test]
    fn absent_marker_or_confident_readout_does_not_arm() {
        assert_eq!(parse_turn_confidence("done, all tests green"), None);
        assert!(!should_arm(None));
        assert!(!should_arm(parse_turn_confidence(
            "[zo:turn-confidence] high — verified end to end"
        )));
    }

    #[test]
    fn quoted_contract_mid_text_does_not_false_arm() {
        // The model explaining the contract inline (not at line start) must
        // not read as a readout.
        let text = "규약상 [zo:turn-confidence] low 를 붙여야 하는 경우가 있는데, \
                    이번 턴은 확신이 있어 붙이지 않습니다.";
        assert_eq!(parse_turn_confidence(text), None);
        // A LINE-START example quoted mid-text must not false-arm either
        // when the actual final line is confident prose — only the last
        // non-empty line counts (the contract says final line).
        let example_mid_text =
            "예:\n[zo:turn-confidence] low — 이유\n위처럼 붙이면 됩니다. 이번 턴은 확신 있음.";
        assert_eq!(parse_turn_confidence(example_mid_text), None);
    }

    #[test]
    fn unknown_token_is_rejected() {
        assert_eq!(
            parse_turn_confidence("[zo:turn-confidence] shaky — hmm"),
            None
        );
    }

    #[test]
    fn last_marker_wins() {
        let text = "[zo:turn-confidence] low — first pass\nfixed it.\n\
                    [zo:turn-confidence] high — re-ran tests, green";
        assert_eq!(parse_turn_confidence(text), Some(TurnConfidence::High));
    }
}
