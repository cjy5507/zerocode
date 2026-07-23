use std::collections::BTreeMap;

use crate::ToolError;
use runtime::CouncilOutcome;
use serde::{Deserialize, Serialize};

pub(crate) const MAX_COUNCIL_CANDIDATES: usize = 7;
pub(crate) const MAX_COUNCIL_CANDIDATE_CHARS: usize = 8_000;
pub(crate) const MAX_COUNCIL_LLM_JUDGE_CALLS: usize = 1;

#[derive(Debug, Deserialize)]
pub(crate) struct CouncilInput {
    pub candidates: Vec<CouncilCandidateInput>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CouncilCandidateInput {
    pub text: String,
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CouncilOutput {
    pub outcome: CouncilOutcome,
    pub candidate_count: usize,
    pub successful_count: usize,
    pub source_hidden: bool,
    pub llm_judge_allowed: bool,
    pub llm_judge_call_limit: usize,
}

pub(crate) fn execute_council(input: &CouncilInput) -> Result<CouncilOutput, ToolError> {
    validate_council_input(input)?;

    let candidate_count = input.candidates.len();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for (index, candidate) in input.candidates.iter().enumerate() {
        if !is_success_status(candidate.status.as_deref()) {
            continue;
        }
        let normalized = normalize_candidate_text(&candidate.text);
        if normalized.is_empty() {
            continue;
        }
        groups.entry(normalized).or_default().push(index);
    }

    let successful_count = groups.values().map(Vec::len).sum();
    let outcome = select_self_consistency(&groups, successful_count);
    let llm_judge_allowed = should_allow_llm_judge(&outcome, successful_count);
    let llm_judge_call_limit = if llm_judge_allowed {
        MAX_COUNCIL_LLM_JUDGE_CALLS
    } else {
        0
    };
    Ok(CouncilOutput {
        outcome,
        candidate_count,
        successful_count,
        source_hidden: true,
        llm_judge_allowed,
        llm_judge_call_limit,
    })
}

fn validate_council_input(input: &CouncilInput) -> Result<(), ToolError> {
    if input.candidates.len() > MAX_COUNCIL_CANDIDATES {
        return Err(ToolError::InvalidInput(format!(
            "Council accepts at most {MAX_COUNCIL_CANDIDATES} candidates (got {})",
            input.candidates.len()
        )));
    }

    for (index, candidate) in input.candidates.iter().enumerate() {
        let char_count = candidate.text.chars().count();
        if char_count > MAX_COUNCIL_CANDIDATE_CHARS {
            return Err(ToolError::InvalidInput(format!(
                "Council candidate {index} text must be at most {MAX_COUNCIL_CANDIDATE_CHARS} characters (got {char_count})"
            )));
        }
    }

    Ok(())
}

fn select_self_consistency(
    groups: &BTreeMap<String, Vec<usize>>,
    successful_count: usize,
) -> CouncilOutcome {
    let Some(max_support) = groups.values().map(Vec::len).max() else {
        return CouncilOutcome::Tie {
            reason: "no successful candidates".to_string(),
        };
    };
    if max_support < 2 || max_support * 2 <= successful_count {
        return CouncilOutcome::Tie {
            reason: "no self-consistency majority".to_string(),
        };
    }

    let winners = groups
        .values()
        .filter(|indices| indices.len() == max_support)
        .collect::<Vec<_>>();
    if winners.len() != 1 {
        return CouncilOutcome::Tie {
            reason: "multiple candidate answers tied".to_string(),
        };
    }

    let supporting_indices = winners[0].clone();
    CouncilOutcome::BestOf {
        winner_index: supporting_indices[0],
        supporting_indices,
        rationale: "selected by self-consistency majority".to_string(),
    }
}

fn should_allow_llm_judge(outcome: &CouncilOutcome, successful_count: usize) -> bool {
    matches!(outcome, CouncilOutcome::Tie { .. }) && successful_count >= 2
}

fn is_success_status(status: Option<&str>) -> bool {
    status.is_none_or(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "ok" | "success" | "succeeded" | "completed"
        )
    })
}

fn normalize_candidate_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{
        execute_council, normalize_candidate_text, CouncilCandidateInput, CouncilInput,
        MAX_COUNCIL_CANDIDATES, MAX_COUNCIL_CANDIDATE_CHARS, MAX_COUNCIL_LLM_JUDGE_CALLS,
    };
    use crate::ToolError;
    use runtime::CouncilOutcome;

    #[test]
    fn self_consistency_selects_majority_candidate() {
        let output = execute_council(&CouncilInput {
            candidates: vec![
                CouncilCandidateInput {
                    text: "Use ProviderClient routing".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "Use providerclient routing".to_string(),
                    status: Some("completed".to_string()),
                },
                CouncilCandidateInput {
                    text: "Rewrite the runtime".to_string(),
                    status: None,
                },
            ],
        })
        .expect("valid council input");

        assert_eq!(output.candidate_count, 3);
        assert_eq!(output.successful_count, 3);
        assert!(output.source_hidden);
        assert!(!output.llm_judge_allowed);
        assert_eq!(output.llm_judge_call_limit, 0);
        assert_eq!(
            output.outcome,
            CouncilOutcome::BestOf {
                winner_index: 0,
                supporting_indices: vec![0, 1],
                rationale: "selected by self-consistency majority".to_string(),
            }
        );
    }

    #[test]
    fn all_failed_candidates_return_honest_tie() {
        let output = execute_council(&CouncilInput {
            candidates: vec![CouncilCandidateInput {
                text: "ignored".to_string(),
                status: Some("failed".to_string()),
            }],
        })
        .expect("valid council input");

        assert_eq!(output.successful_count, 0);
        assert!(!output.llm_judge_allowed);
        assert_eq!(output.llm_judge_call_limit, 0);
        assert_eq!(
            output.outcome,
            CouncilOutcome::Tie {
                reason: "no successful candidates".to_string(),
            }
        );
    }

    #[test]
    fn plurality_without_majority_does_not_fake_a_winner() {
        let output = execute_council(&CouncilInput {
            candidates: vec![
                CouncilCandidateInput {
                    text: "A".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "A".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "B".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "C".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "D".to_string(),
                    status: None,
                },
            ],
        })
        .expect("valid council input");

        assert_eq!(output.successful_count, 5);
        assert!(output.llm_judge_allowed);
        assert_eq!(
            output.outcome,
            CouncilOutcome::Tie {
                reason: "no self-consistency majority".to_string(),
            }
        );
    }

    #[test]
    fn unique_answers_do_not_fake_a_winner() {
        let output = execute_council(&CouncilInput {
            candidates: vec![
                CouncilCandidateInput {
                    text: "A".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "B".to_string(),
                    status: None,
                },
                CouncilCandidateInput {
                    text: "C".to_string(),
                    status: None,
                },
            ],
        })
        .expect("valid council input");

        assert!(output.llm_judge_allowed);
        assert_eq!(output.llm_judge_call_limit, MAX_COUNCIL_LLM_JUDGE_CALLS);
        assert_eq!(
            output.outcome,
            CouncilOutcome::Tie {
                reason: "no self-consistency majority".to_string(),
            }
        );
    }

    #[test]
    fn normalization_ignores_case_and_whitespace() {
        assert_eq!(
            normalize_candidate_text("  Same\n answer\tagain "),
            "same answer again"
        );
    }

    #[test]
    fn rejects_too_many_candidates() {
        let candidates = (0..=MAX_COUNCIL_CANDIDATES)
            .map(|index| CouncilCandidateInput {
                text: format!("candidate {index}"),
                status: None,
            })
            .collect();
        let error = execute_council(&CouncilInput { candidates })
            .expect_err("candidate count should be bounded");

        assert!(matches!(
            error,
            ToolError::InvalidInput(message)
                if message.contains("at most")
                    && message.contains(&MAX_COUNCIL_CANDIDATES.to_string())
        ));
    }

    #[test]
    fn rejects_oversized_candidate_text() {
        let error = execute_council(&CouncilInput {
            candidates: vec![CouncilCandidateInput {
                text: "x".repeat(MAX_COUNCIL_CANDIDATE_CHARS + 1),
                status: None,
            }],
        })
        .expect_err("candidate text should be bounded");

        assert!(matches!(
            error,
            ToolError::InvalidInput(message)
                if message.contains("candidate 0")
                    && message.contains(&MAX_COUNCIL_CANDIDATE_CHARS.to_string())
        ));
    }
}
