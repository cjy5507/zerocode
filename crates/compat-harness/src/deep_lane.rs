//! Deep-lane orchestration glue for the agent-eval harness.
//!
//! The portable state machine lives in `decision_core::deep_lane`; this crate keeps
//! the compatibility boundary that needs benchmark-specific diff hygiene.

pub use decision_core::deep_lane::{
    decide, fold_verification_attempt, parse_verifier, validate_plan, verifier_gate_accepts,
    DeepDecision, PlanVerdict, VerificationAttempt, VerifierParse, VerifierVerdict,
    MAX_SUMMARY_CHARS, REQUIRED_PLAN_SECTIONS,
};

use crate::diff_hygiene::{run_passed, DiffHygiene, TestStatus};

/// Whether the objective gate passed for one attempt: the agent exited 0, the
/// test did not fail, the diff is clean, and any declared intended set changed.
#[must_use]
pub fn objective_passed(
    exit_code: i32,
    test: TestStatus,
    hygiene: &DiffHygiene,
    intended_provided: bool,
) -> bool {
    run_passed(exit_code, test, hygiene, 0, intended_provided)
}

fn tail(text: &str, max: usize) -> String {
    let trimmed = text.trim_end();
    let char_count = trimmed.chars().count();
    if char_count <= max {
        return trimmed.to_string();
    }
    let start = char_count - max;
    let kept: String = trimmed.chars().skip(start).collect();
    format!("...(truncated {start} chars)...\n{kept}")
}

/// Builds the compact failure summary fed to the next execute pass. It lists
/// only what must change and clamps the result to [`MAX_SUMMARY_CHARS`].
#[must_use]
pub fn failure_summary(
    hygiene: &DiffHygiene,
    test: TestStatus,
    test_log: &str,
    verifier: &VerifierVerdict,
) -> String {
    // `tail` prepends its truncation banner *outside* the `max` budget, so
    // clamp with headroom reserved for the banner — otherwise a long input
    // overshoots MAX_SUMMARY_CHARS by the banner length.
    const TAIL_BANNER_HEADROOM: usize = 64;
    let mut out = String::new();
    out.push_str("Your previous attempt did not pass. Fix ONLY the problems below. ");
    out.push_str(
        "Do not edit unrelated files, and do not modify or delete tests to make them pass.\n",
    );
    out.push_str(
        "If the verifier flags behavior the task did not request, remove that behavior even when the tests already pass.\n",
    );

    if test == TestStatus::Fail {
        out.push_str("\n## Failing test (tail)\n");
        out.push_str(&tail(test_log, MAX_SUMMARY_CHARS.saturating_sub(600)));
        out.push('\n');
    }
    if !hygiene.unexpected.is_empty() {
        out.push_str("\n## Files you changed that the task did not ask for (revert these)\n");
        for path in &hygiene.unexpected {
            out.push_str("- ");
            out.push_str(path);
            out.push('\n');
        }
    }
    if !hygiene.pollution.is_empty() {
        out.push_str("\n## Runtime artifacts leaked into the repo (remove these)\n");
        for path in &hygiene.pollution {
            out.push_str("- ");
            out.push_str(path);
            out.push('\n');
        }
    }
    if !verifier.issues.is_empty() {
        out.push_str("\n## Verifier findings\n");
        for issue in &verifier.issues {
            out.push_str("- ");
            out.push_str(issue);
            out.push('\n');
        }
    }
    out.push_str("\n## Mandatory repair checklist\n");
    out.push_str("- Make the objective test pass first; do not stop on a red test.\n");
    out.push_str(
        "- For every verifier finding above, change the code until that exact defect is gone.\n",
    );
    out.push_str(
        "- If a finding names a stale symbol, wrong receiver, or missed call site, search all intended files and fix every occurrence.\n",
    );
    out.push_str(
        "- If the task threads options or a new parameter, audit every caller, wrapper, and cache path; do not use lossy stringified cache keys unless the task explicitly requires them.\n",
    );
    out.push_str(
        "- Re-run the exact failing test command after edits and inspect any remaining failure before stopping.\n",
    );
    // `tail` prepends its truncation banner *outside* the `max` budget; the
    // headroom was reserved at the top of the function.
    tail(&out, MAX_SUMMARY_CHARS.saturating_sub(TAIL_BANNER_HEADROOM))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff_hygiene::score;

    #[test]
    fn compat_reexports_core_deep_lane_decisions() {
        assert!(
            validate_plan(
                "Files: a.js\nInvariants: stable API\nTests: cargo test\nRisks: edge cases"
            )
            .valid
        );
        assert_eq!(decide(1, 2, false, false), DeepDecision::Retry);
        assert_eq!(
            parse_verifier(r#"{"accepted": true}"#).parse,
            VerifierParse::Json
        );
        let verifier = VerifierVerdict {
            accepted: true,
            issues: Vec::new(),
            parse: VerifierParse::Json,
            evidence: None,
        };
        let folded: VerificationAttempt = fold_verification_attempt(1, 2, true, &verifier, &[]);
        assert_eq!(folded.decision, DeepDecision::Accept);
        assert!(folded.gate_accepted);
    }

    #[test]
    fn objective_passed_mirrors_run_passed() {
        let clean = score(" M src/x.js\n", &["src/x.js"]);
        assert!(objective_passed(0, TestStatus::Pass, &clean, true));
        let dirty = score(" M other.js\n", &["src/x.js"]);
        assert!(!objective_passed(0, TestStatus::Pass, &dirty, true));
        let noop = score("", &["src/x.js"]);
        assert!(!objective_passed(0, TestStatus::Skipped, &noop, true));
    }

    #[test]
    fn summary_includes_test_tail_and_issues() {
        let hygiene = score(" M src/x.js\n", &["src/x.js"]);
        let verdict = VerifierVerdict {
            accepted: false,
            issues: vec!["missing edge case".to_string()],
            parse: VerifierParse::Json,
            evidence: None,
        };
        let summary = failure_summary(
            &hygiene,
            TestStatus::Fail,
            "AssertionError: expected 2 got 3",
            &verdict,
        );
        assert!(summary.contains("Failing test"));
        assert!(summary.contains("expected 2 got 3"));
        assert!(summary.contains("missing edge case"));
        assert!(summary.contains("do not modify or delete tests"));
        assert!(summary.contains("behavior the task did not request"));
        assert!(summary.contains("Mandatory repair checklist"));
        assert!(summary.contains("stale symbol"));
        assert!(summary.contains("cache path"));
    }

    #[test]
    fn summary_lists_unexpected_and_pollution() {
        let hygiene = score(" M src/x.js\n M evil.js\n?? .zo/\n", &["src/x.js"]);
        let verdict = VerifierVerdict {
            accepted: false,
            issues: vec![],
            parse: VerifierParse::Json,
            evidence: None,
        };
        let summary = failure_summary(&hygiene, TestStatus::Pass, "", &verdict);
        assert!(summary.contains("evil.js"));
        assert!(summary.contains(".zo/"));
        assert!(summary.contains("revert these"));
        assert!(summary.contains("remove these"));
    }

    #[test]
    fn summary_is_clamped() {
        let hygiene = score("", &[]);
        let verdict = VerifierVerdict {
            accepted: false,
            issues: vec![],
            parse: VerifierParse::Json,
            evidence: None,
        };
        let huge = "x".repeat(100_000);
        let summary = failure_summary(&hygiene, TestStatus::Fail, &huge, &verdict);
        assert!(summary.chars().count() <= MAX_SUMMARY_CHARS);
        assert!(summary.contains("truncated"));
    }

    #[test]
    fn tail_keeps_recent_content() {
        let value = tail("abcdefghij", 4);
        assert!(value.ends_with("ghij"));
        assert!(value.contains("truncated"));
        assert_eq!(tail("abc", 10), "abc");
    }
}
