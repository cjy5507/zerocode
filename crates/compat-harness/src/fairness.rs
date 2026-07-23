//! Per-run fairness contract: the recorded conditions that make one run
//! comparable to another.
//!
//! The spec requires *every* run to emit one `fairness_contract.json`. This
//! module computes the parts that must be tested and reproducible — the input
//! hashes, the model/effort normalization, and the per-run validity verdict —
//! while the shell supplies the online context it alone knows (git commit/tree
//! hash, dirty flags, timestamps, runner/harness versions, permission mode).
//!
//! `fairness_level` (strict/normalized/comparable) is a *cross-runner* judgment
//! made later by the scorer from two contracts; a single run records `unknown`.

use std::fmt::Write;

use serde::Serialize;
use sha2::{Digest, Sha256};

use decision_core::decision::FairnessStatus;

/// Lowercase hex SHA-256 of a string. Used for prompt / test-command /
/// intended-path-set hashes so two runs are provably given identical inputs.
#[must_use]
pub fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Normalize a declared model label to its family token for cross-runner
/// comparison (e.g. `claude-opus-4-8` → `opus`, `gpt-5.5` → `gpt-5`).
///
/// Data-driven: the first matching family substring wins; extend `FAMILIES` to
/// teach a new provider. Returns `unknown` when nothing matches, which the
/// validity verdict treats as a comparability gap, not a hard failure.
#[must_use]
pub fn normalize_model_family(declared: &str) -> String {
    const FAMILIES: &[&str] = &[
        "opus", "sonnet", "haiku", "gpt-5", "gpt-4", "o3", "o1", "gemini", "grok",
    ];
    let lower = declared.to_ascii_lowercase();
    FAMILIES
        .iter()
        .find(|family| lower.contains(*family))
        .map_or_else(|| "unknown".to_string(), |family| (*family).to_string())
}

/// Normalize an effort label to a coarse tier so two runners declared at the
/// same effort compare as equal even when they spell it differently.
#[must_use]
pub fn normalize_effort(declared: &str) -> String {
    match declared.trim().to_ascii_lowercase().as_str() {
        "" | "default" => "default",
        "low" | "minimal" => "low",
        "medium" | "med" => "medium",
        "high" => "high",
        "max" | "maximum" => "max",
        other => return other.to_string(),
    }
    .to_string()
}

/// The online context the shell supplies; the pure fields (hashes,
/// normalization, status) are derived by [`build_contract`].
#[derive(Debug, Clone, Default)]
pub struct FairnessInput {
    pub runner: String,
    pub lane: String,
    pub fixture_id: String,
    pub fixture_commit: String,
    pub fixture_tree_hash: String,
    pub fixture_dirty_before: bool,
    pub fixture_dirty_after: bool,
    pub prompt_path: String,
    /// The raw prompt text, hashed into `prompt_sha256`.
    pub prompt: String,
    pub test_command: String,
    pub intended_path_set: Vec<String>,
    pub declared_model: String,
    pub declared_effort: String,
    pub permission_mode: String,
    pub timeout_seconds: u64,
    pub runner_version: String,
    pub harness_version: String,
    pub benchmark_suite_version: String,
    pub started_at: String,
    pub finished_at: String,
}

/// One run's fairness contract, serialized to `fairness_contract.json`.
#[derive(Debug, Clone, Serialize)]
pub struct FairnessContract {
    pub fairness_contract_version: String,
    pub status: String,
    pub fairness_level: String,
    pub runner: String,
    pub lane: String,
    pub fixture_id: String,
    pub fixture_commit: String,
    pub fixture_tree_hash: String,
    pub fixture_dirty_before: bool,
    pub fixture_dirty_after: bool,
    pub prompt_path: String,
    pub prompt_sha256: String,
    pub test_command: String,
    pub test_command_sha256: String,
    pub intended_path_set: Vec<String>,
    pub intended_path_set_sha256: String,
    pub declared_model: String,
    pub normalized_model_family: String,
    pub declared_effort: String,
    pub normalized_effort: String,
    pub permission_mode: String,
    pub timeout_seconds: u64,
    pub runner_version: String,
    pub harness_version: String,
    pub benchmark_suite_version: String,
    pub started_at: String,
    pub finished_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_reason: Option<String>,
}

/// Decide whether a single run's recorded conditions are trustworthy. A dirty
/// fixture or a missing required input invalidates the run (spec Invalid
/// Policy); a model whose family cannot be normalized is `partial` — usable for
/// same-runner trends but not a clean cross-runner comparison.
fn judge(input: &FairnessInput, normalized_model: &str) -> (FairnessStatus, Option<String>) {
    if input.prompt.trim().is_empty() {
        return (FairnessStatus::Invalid, Some("prompt is empty".to_string()));
    }
    if input.test_command.trim().is_empty() {
        return (
            FairnessStatus::Invalid,
            Some("test_command is empty".to_string()),
        );
    }
    if input.fixture_dirty_before {
        return (
            FairnessStatus::Invalid,
            Some("fixture was dirty before the run".to_string()),
        );
    }
    // The fixture's starting state must be pinned by *something* — a commit or a
    // tree hash. With neither, the diff cannot be replayed against a known base,
    // so the run is not reproducible.
    if input.fixture_commit.trim().is_empty() && input.fixture_tree_hash.trim().is_empty() {
        return (
            FairnessStatus::Invalid,
            Some("fixture state unidentified (no commit or tree hash)".to_string()),
        );
    }

    // Partial: usable for same-runner trends, but a clean cross-runner comparison
    // needs the declared conditions on record. An unrecognized model family, or a
    // missing run condition (permission mode, runner/harness version), leaves the
    // comparison incomplete rather than wrong — so the run is recorded, not voided.
    if normalized_model == "unknown" {
        return (
            FairnessStatus::Partial,
            Some(format!(
                "model family not recognized: {}",
                input.declared_model
            )),
        );
    }
    let mut undeclared = Vec::new();
    if input.permission_mode.trim().is_empty() {
        undeclared.push("permission_mode");
    }
    if input.runner_version.trim().is_empty() {
        undeclared.push("runner_version");
    }
    if input.harness_version.trim().is_empty() {
        undeclared.push("harness_version");
    }
    if !undeclared.is_empty() {
        return (
            FairnessStatus::Partial,
            Some(format!("undeclared conditions: {}", undeclared.join(", "))),
        );
    }

    (FairnessStatus::Valid, None)
}

/// Build a per-run fairness contract: compute the input hashes, normalize the
/// model/effort, and judge the run's validity. Intended paths are sorted before
/// hashing so the set hash is order-independent.
#[must_use]
pub fn build_contract(input: &FairnessInput) -> FairnessContract {
    let normalized_model_family = normalize_model_family(&input.declared_model);
    let (status, status_reason) = judge(input, &normalized_model_family);

    let mut intended = input.intended_path_set.clone();
    intended.sort();
    intended.dedup();
    let intended_joined = intended.join("\n");

    FairnessContract {
        fairness_contract_version: "1.0".to_string(),
        status: status.as_str().to_string(),
        fairness_level: "unknown".to_string(),
        runner: input.runner.clone(),
        lane: input.lane.clone(),
        fixture_id: input.fixture_id.clone(),
        fixture_commit: input.fixture_commit.clone(),
        fixture_tree_hash: input.fixture_tree_hash.clone(),
        fixture_dirty_before: input.fixture_dirty_before,
        fixture_dirty_after: input.fixture_dirty_after,
        prompt_path: input.prompt_path.clone(),
        prompt_sha256: sha256_hex(&input.prompt),
        test_command: input.test_command.clone(),
        test_command_sha256: sha256_hex(&input.test_command),
        intended_path_set_sha256: sha256_hex(&intended_joined),
        intended_path_set: intended,
        declared_model: input.declared_model.clone(),
        normalized_model_family,
        declared_effort: input.declared_effort.clone(),
        normalized_effort: normalize_effort(&input.declared_effort),
        permission_mode: input.permission_mode.clone(),
        timeout_seconds: input.timeout_seconds,
        runner_version: input.runner_version.clone(),
        harness_version: input.harness_version.clone(),
        benchmark_suite_version: input.benchmark_suite_version.clone(),
        started_at: input.started_at.clone(),
        finished_at: input.finished_at.clone(),
        status_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        // SHA-256 of the empty string is the well-known all-zeros-input digest.
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn model_family_normalizes_from_declared_label() {
        assert_eq!(normalize_model_family("claude-opus-4-8"), "opus");
        assert_eq!(normalize_model_family("claude-sonnet-4-6"), "sonnet");
        assert_eq!(normalize_model_family("gpt-5.5"), "gpt-5");
        assert_eq!(normalize_model_family("something-else"), "unknown");
    }

    #[test]
    fn effort_normalizes_synonyms() {
        assert_eq!(normalize_effort("Maximum"), "max");
        assert_eq!(normalize_effort("MED"), "medium");
        assert_eq!(normalize_effort(""), "default");
        assert_eq!(normalize_effort("custom-tier"), "custom-tier");
    }

    fn valid_input() -> FairnessInput {
        FairnessInput {
            runner: "zo".to_string(),
            lane: "deep".to_string(),
            fixture_id: "wide-rename".to_string(),
            fixture_tree_hash: "abc123treehash".to_string(),
            prompt: "Rename fetch to load.".to_string(),
            test_command: "node --test".to_string(),
            intended_path_set: vec!["src/b.js".to_string(), "src/a.js".to_string()],
            declared_model: "claude-opus-4-8".to_string(),
            declared_effort: "high".to_string(),
            permission_mode: "danger-full-access".to_string(),
            timeout_seconds: 300,
            runner_version: "zo 0.1.0".to_string(),
            harness_version: "1.0".to_string(),
            ..FairnessInput::default()
        }
    }

    #[test]
    fn build_contract_hashes_and_normalizes_a_clean_run() {
        let contract = build_contract(&valid_input());
        assert_eq!(contract.status, "valid");
        assert_eq!(contract.normalized_model_family, "opus");
        assert_eq!(contract.normalized_effort, "high");
        assert_eq!(contract.prompt_sha256.len(), 64);
        // Intended paths are sorted before hashing.
        assert_eq!(contract.intended_path_set, vec!["src/a.js", "src/b.js"]);
        assert!(contract.status_reason.is_none());
    }

    #[test]
    fn intended_path_set_hash_is_order_independent() {
        let mut a = valid_input();
        a.intended_path_set = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let mut b = valid_input();
        b.intended_path_set = vec!["z".to_string(), "x".to_string(), "y".to_string()];
        assert_eq!(
            build_contract(&a).intended_path_set_sha256,
            build_contract(&b).intended_path_set_sha256
        );
    }

    #[test]
    fn dirty_fixture_is_invalid() {
        let mut input = valid_input();
        input.fixture_dirty_before = true;
        let contract = build_contract(&input);
        assert_eq!(contract.status, "invalid");
        assert!(contract.status_reason.unwrap().contains("dirty"));
    }

    #[test]
    fn empty_required_inputs_are_invalid() {
        let mut input = valid_input();
        input.prompt = "  ".to_string();
        assert_eq!(build_contract(&input).status, "invalid");

        let mut input = valid_input();
        input.test_command = String::new();
        assert_eq!(build_contract(&input).status, "invalid");
    }

    #[test]
    fn unrecognized_model_is_partial_not_invalid() {
        let mut input = valid_input();
        input.declared_model = "mystery-model-9".to_string();
        let contract = build_contract(&input);
        assert_eq!(contract.status, "partial");
        assert_eq!(contract.normalized_model_family, "unknown");
    }

    #[test]
    fn fixture_without_commit_or_tree_hash_is_invalid() {
        // risk 4: neither identifier pins the base ⇒ not reproducible.
        let mut input = valid_input();
        input.fixture_tree_hash = String::new(); // fixture_commit already empty
        let contract = build_contract(&input);
        assert_eq!(contract.status, "invalid");
        assert!(contract.status_reason.unwrap().contains("fixture state"));
    }

    #[test]
    fn fixture_pinned_by_commit_alone_is_not_invalid() {
        let mut input = valid_input();
        input.fixture_tree_hash = String::new();
        input.fixture_commit = "deadbeefcafe".to_string();
        // A commit identifies the base even without a tree hash.
        assert_ne!(build_contract(&input).status, "invalid");
    }

    #[test]
    fn undeclared_run_conditions_are_partial() {
        // risk 4: an empty permission mode / runner / harness version is no longer
        // silently "valid" — the comparison is incomplete, so the run is partial.
        let mut input = valid_input();
        input.permission_mode = String::new();
        let contract = build_contract(&input);
        assert_eq!(contract.status, "partial");
        assert!(contract.status_reason.unwrap().contains("undeclared"));

        let mut input = valid_input();
        input.runner_version = String::new();
        assert_eq!(build_contract(&input).status, "partial");

        let mut input = valid_input();
        input.harness_version = String::new();
        assert_eq!(build_contract(&input).status, "partial");
    }
}
