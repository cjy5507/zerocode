//! Built-in workflow presets.
//!
//! Presets are only an input-expansion layer: they turn a small, named request
//! into an ordinary [`WorkflowSpec`](super::spec::WorkflowSpec) JSON value. The
//! engine still owns execution, routing, retries, and verification.

use serde_json::{json, Value};

use crate::ToolError;

const CROSS_MODEL_VERIFIED: &str = "cross_model_verified";
const GPT_CLAUDE_VERIFIED: &str = "gpt_claude_verified";
const DEFAULT_VERIFY_COMMAND: &str = "cargo check --workspace --all-targets";
const DEFAULT_MAX_ROUNDS: u32 = 3;

#[derive(Debug)]
pub(super) struct PresetExpansion {
    pub spec: Value,
    pub input: Value,
}

pub(super) fn expand_preset(input: &Value) -> Result<Option<PresetExpansion>, ToolError> {
    let Some(preset) = input.get("preset") else {
        return Ok(None);
    };
    if input.get("spec").is_some() {
        return Err(invalid("`preset` and `spec` are mutually exclusive"));
    }

    let preset = preset
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid("`preset` must be a non-empty string"))?;

    match preset {
        CROSS_MODEL_VERIFIED | GPT_CLAUDE_VERIFIED => Ok(Some(cross_model_verified(input)?)),
        other => Err(invalid(format!(
            "unknown workflow preset `{other}` (expected `{CROSS_MODEL_VERIFIED}` or `{GPT_CLAUDE_VERIFIED}`)"
        ))),
    }
}

fn cross_model_verified(input: &Value) -> Result<PresetExpansion, ToolError> {
    let forced = forced_agent_model_override();
    cross_model_verified_with_override(input, forced.as_deref())
}

fn cross_model_verified_with_override(
    input: &Value,
    forced_agent_model: Option<&str>,
) -> Result<PresetExpansion, ToolError> {
    if let Some(forced) = forced_agent_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Err(invalid(format!(
            "`ZO_AGENT_MODEL` is set to `{forced}`, which forces every sub-agent onto one model and disables the `cross_model_verified` contract; unset it or use an explicit workflow spec"
        )));
    }

    let task = input.get("input").cloned().unwrap_or(Value::Null);
    if task_is_empty(&task) {
        return Err(invalid(
            "`input` is required for `cross_model_verified` and must not be empty",
        ));
    }

    let verify_command = match string_field(input, "verify_command")? {
        Some(command) => command,
        None => string_field(input, "verification_command")?
            .unwrap_or_else(|| DEFAULT_VERIFY_COMMAND.to_string()),
    };
    if verify_command.trim().is_empty() {
        return Err(invalid("`verify_command` must not be empty"));
    }

    let coding_model = required_string_field(input, "coding_model")?;
    let review_model = required_string_field(input, "review_model")?;
    if coding_model == review_model {
        return Err(invalid(
            "`coding_model` and `review_model` must be different for `cross_model_verified`",
        ));
    }
    let verification_model = string_field(input, "verification_model")?
        .unwrap_or_else(|| review_model.clone());
    let synthesis_model = string_field(input, "synthesis_model")?
        .unwrap_or_else(|| review_model.clone());

    let max_rounds = u32_field(input, "max_rounds")?.unwrap_or(DEFAULT_MAX_ROUNDS);
    if max_rounds == 0 {
        return Err(invalid("`max_rounds` must be at least 1"));
    }

    Ok(PresetExpansion {
        spec: json!({
            "name": "cross-model-verified",
            "description": "Built-in preset: preflight minimality, coding-role implementation, reviewer-role critique, coding-role repair, verifier-role final audit, and a hard command_green verification gate.",
            "mode": "phases",
            "phases": [
                {
                    "id": "preflight",
                    "subagent_type": "Plan",
                    "model": review_model,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "can_avoid_new_code": { "type": "boolean" },
                            "minimal_approach": { "type": "string" },
                            "files_to_inspect": { "type": "array", "items": { "type": "string" } },
                            "risks": { "type": "array", "items": { "type": "string" } }
                        },
                        "required": ["can_avoid_new_code", "minimal_approach"]
                    },
                    "prompt": "Preflight the task before any implementation. Decide whether existing code, standard library APIs, existing dependencies, configuration, or documentation can solve it with less new code. Be conservative: do not invent APIs. Return the required JSON only. Task:\n{input}"
                },
                {
                    "id": "implement",
                    "over": "preflight",
                    "subagent_type": "general-purpose",
                    "model": coding_model,
                    "prompt": "Implement the task using the preflight result below. Prefer the smallest correct change, clean code, single responsibility, no speculative abstractions, and no unrelated cleanup. If preflight says new code can be avoided, do that instead of generating needless code. Do not commit.\n\nTask:\n{input}\n\nPreflight result:\n{item}"
                },
                {
                    "id": "adversarial_review",
                    "over": "implement",
                    "subagent_type": "code-reviewer",
                    "model": review_model,
                    "prompt": "Adversarially review the current working tree after the implementation. Assume a different model wrote it. Check correctness, security, concurrency, error handling, edge cases, tests, literal/spec mismatches, SRP, and unnecessary complexity. Report only actionable blockers or high-value fixes; if clean, say so clearly.\n\nTask:\n{input}\n\nImplementation handoff:\n{item}"
                },
                {
                    "id": "repair_until_green",
                    "over": "adversarial_review",
                    "subagent_type": "general-purpose",
                    "model": coding_model,
                    "prompt": "Apply only valid fixes from the review, then ensure the verification command is likely to pass. Keep changes surgical and SRP-friendly. If the review is already clean, do not churn code; explain the no-op. Previous repair rounds are below.\n\nTask:\n{input}\n\nReview:\n{item}\n\nPrior repair rounds:\n{seen}",
                    "repeat": {
                        "max_rounds": max_rounds,
                        "until": { "command_green": { "command": verify_command } }
                    }
                },
                {
                    "id": "final_verification",
                    "over": "repair_until_green",
                    "subagent_type": "Verification",
                    "model": verification_model,
                    "prompt": "Final verification pass from an independent verifier role. Inspect the current diff and relevant tests. Confirm whether the task is complete, the verification command is appropriate, and no avoidable complexity or SRP violation remains. Do not edit files unless a small fix is absolutely necessary.\n\nTask:\n{input}\n\nRepair handoff:\n{item}"
                }
            ],
            "synthesize": {
                "subagent_type": "Plan",
                "model": synthesis_model,
                "prompt": "Summarize the cross-model workflow result for the parent agent. Include: final status, decisive verification evidence, remaining risks, and whether commit/push is safe. Use only the completed workflow outputs below.\n\n{all}"
            }
        }),
        input: task,
    })
}

fn required_string_field(input: &Value, key: &str) -> Result<String, ToolError> {
    string_field(input, key)?.ok_or_else(|| invalid(format!("`{key}` is required")))
}

fn string_field(input: &Value, key: &str) -> Result<Option<String>, ToolError> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.trim().to_string())),
        Some(_) => Err(invalid(format!("`{key}` must be a string"))),
    }
}

fn u32_field(input: &Value, key: &str) -> Result<Option<u32>, ToolError> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_u64()
            .and_then(|number| u32::try_from(number).ok())
            .map(Some)
            .ok_or_else(|| invalid(format!("`{key}` must be a positive integer"))),
        Some(_) => Err(invalid(format!("`{key}` must be a positive integer"))),
    }
}

fn task_is_empty(task: &Value) -> bool {
    match task {
        Value::Null => true,
        Value::String(value) => value.trim().is_empty(),
        _ => false,
    }
}

#[cfg(not(test))]
fn forced_agent_model_override() -> Option<String> {
    std::env::var("ZO_AGENT_MODEL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
fn forced_agent_model_override() -> Option<String> {
    None
}

fn invalid(message: impl Into<String>) -> ToolError {
    ToolError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_cross_model_verified_preset() {
        let expanded = expand_preset(&json!({
            "preset": "cross_model_verified",
            "input": "fix the parser",
            "verify_command": "cargo test -p tools",
            "coding_model": "gpt-5.5-fast",
            "review_model": "claude-opus-4-8",
            "verification_model": "claude-sonnet-verify",
            "synthesis_model": "gpt-synth",
            "max_rounds": 2
        }))
        .expect("preset should parse")
        .expect("preset should expand");

        assert_eq!(expanded.input, json!("fix the parser"));
        assert_eq!(expanded.spec["name"], "cross-model-verified");
        assert_eq!(expanded.spec["phases"].as_array().expect("phases").len(), 5);
        assert_eq!(
            expanded.spec["phases"][1]["subagent_type"],
            "general-purpose",
            "implementation must route through the coding role"
        );
        assert_eq!(
            expanded.spec["phases"][2]["subagent_type"],
            "code-reviewer",
            "review must route through an independent reviewer role"
        );
        assert_eq!(
            expanded.spec["phases"][1]["model"],
            "gpt-5.5-fast",
            "implementation must use the explicit coding model"
        );
        assert_eq!(
            expanded.spec["phases"][2]["model"],
            "claude-opus-4-8",
            "review must use the explicit review model"
        );
        assert_eq!(
            expanded.spec["phases"][4]["model"],
            "claude-sonnet-verify",
            "final verifier may use a dedicated verification model"
        );
        assert_eq!(
            expanded.spec["synthesize"]["model"],
            "gpt-synth",
            "synthesis may use a dedicated synthesis model"
        );
        assert_eq!(
            expanded.spec["phases"][3]["repeat"]["until"]["command_green"]["command"],
            "cargo test -p tools"
        );
    }

    #[test]
    fn alias_expands_like_primary_preset() {
        let primary = expand_preset(&json!({
            "preset": "cross_model_verified",
            "input": "fix the parser",
            "coding_model": "gpt-5.5-fast",
            "review_model": "claude-opus-4-8"
        }))
        .expect("primary preset should parse")
        .expect("primary preset should expand");
        let alias = expand_preset(&json!({
            "preset": "gpt_claude_verified",
            "input": "fix the parser",
            "coding_model": "gpt-5.5-fast",
            "review_model": "claude-opus-4-8"
        }))
        .expect("alias preset should parse")
        .expect("alias preset should expand");

        assert_eq!(alias.spec, primary.spec);
        assert_eq!(alias.input, primary.input);
    }

    #[test]
    fn forced_agent_model_override_rejects_cross_model_preset() {
        let error = cross_model_verified_with_override(
            &json!({
                "preset": "cross_model_verified",
                "input": "fix the parser",
                "coding_model": "gpt-5.5-fast",
                "review_model": "claude-opus-4-8"
            }),
            Some("forced-model"),
        )
        .expect_err("forced single-model env disables cross-model preset")
        .to_string();

        assert!(error.contains("ZO_AGENT_MODEL"), "got {error}");
        assert!(error.contains("forced-model"), "got {error}");
    }

    #[test]
    fn unknown_preset_error_mentions_alias() {
        let error = expand_preset(&json!({ "preset": "unknown" }))
            .expect_err("unknown preset should fail")
            .to_string();
        assert!(error.contains("cross_model_verified"), "got {error}");
        assert!(error.contains("gpt_claude_verified"), "got {error}");
    }

    #[test]
    fn preset_rejects_missing_task() {
        let error = expand_preset(&json!({
            "preset": "cross_model_verified",
            "coding_model": "gpt-5.5-fast",
            "review_model": "claude-opus-4-8"
        }))
        .expect_err("task input is required")
        .to_string();
        assert!(error.contains("input"), "got {error}");
    }

    #[test]
    fn preset_rejects_missing_models() {
        let error = expand_preset(&json!({
            "preset": "cross_model_verified",
            "input": "fix the parser"
        }))
        .expect_err("cross-model preset needs explicit models")
        .to_string();
        assert!(error.contains("coding_model"), "got {error}");
    }

    #[test]
    fn preset_rejects_same_coding_and_review_model() {
        let error = expand_preset(&json!({
            "preset": "cross_model_verified",
            "input": "fix the parser",
            "coding_model": "gpt-5.5-fast",
            "review_model": " gpt-5.5-fast "
        }))
        .expect_err("models must differ")
        .to_string();
        assert!(error.contains("must be different"), "got {error}");
    }
}
