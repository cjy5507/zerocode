//! Data-driven benchmark task manifests.
//!
//! A fixture stops being defined by a human runbook plus CLI flags and instead
//! declares itself in `task.toml`: its lane, prompt, test command, intended
//! paths, and timeout. A `lanes.toml` catalog declares each lane's policy. The
//! harness discovers and validates these, so adding a task or activating a lane
//! is a *data* change — no shell edit, no recompile, no runbook drift.
//!
//! The portable lane *names* live in [`decision_core::decision::BenchmarkLane`];
//! this module owns the on-disk *shape* and the discovery/validation IO.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use decision_core::decision::BenchmarkLane;
use serde::{Deserialize, Serialize};

/// One self-describing benchmark task (`task.toml` in a fixture directory).
///
/// The prompt, test command, and intended paths are the *contract the harness
/// scores* — recorded verbatim into the fairness contract so two runners are
/// compared on identical inputs. It round-trips: parsed from `task.toml`, and
/// re-serialized to JSON by `deep-eval discover` for the shell to iterate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskManifest {
    /// Manifest schema version, for forward-compatible evolution.
    pub schema_version: String,
    /// Stable, unique, URL-safe task id (also the artifact directory name).
    pub id: String,
    /// Lane token, validated against [`BenchmarkLane`] and the lane catalog.
    pub lane: String,
    /// Optional difficulty hint (`easy` | `medium` | `hard` | `expert`); kept
    /// separate from the lane so same-lane tasks of differing difficulty do not
    /// confound pass-rate interpretation.
    #[serde(default)]
    pub difficulty: Option<String>,
    /// Canonical regression axes this fixture exercises. Kept as a strict
    /// taxonomy so coverage reports stay machine-readable rather than drifting
    /// into one-off prose labels.
    #[serde(default)]
    pub coverage_tags: Vec<String>,
    /// The raw task prompt handed to every runner verbatim.
    pub prompt: String,
    /// Deterministic, offline test command that drives the objective gate.
    pub test_command: String,
    /// Files the agent is expected to modify; diff hygiene is scored on these.
    #[serde(default)]
    pub intended_paths: Vec<String>,
    /// Per-task timeout override; falls back to the lane policy when unset.
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    /// Tests that must flip fail→pass (SWE-bench `FAIL_TO_PASS`). Empty means
    /// the whole `test_command` result is the objective gate.
    #[serde(default)]
    pub tests_that_must_flip: Vec<String>,
    /// Tests that must stay green (SWE-bench `PASS_TO_PASS`; no regression).
    #[serde(default)]
    pub tests_that_must_not_regress: Vec<String>,
    /// Optional reference solution proving the task is solvable.
    #[serde(default)]
    pub oracle_solution: Option<String>,
}

impl TaskManifest {
    /// The parsed lane, or `None` when the lane token is not a known lane.
    #[must_use]
    pub fn parsed_lane(&self) -> Option<BenchmarkLane> {
        BenchmarkLane::from_token(&self.lane)
    }

    /// Parse a task manifest from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }
}

/// One lane's scoring policy (a `[lanes.<name>]` entry in `lanes.toml`).
#[derive(Debug, Clone, Deserialize)]
pub struct LanePolicy {
    /// How the objective gate is computed (e.g. `test_and_diff`).
    pub objective_gate: String,
    /// Verifier strictness: `none` | `strict` | `salvage_allowed`.
    pub verifier_policy: String,
    /// Maximum agent retries in the deep loop for this lane.
    pub retry_budget: u32,
    /// Allowed-diff policy (e.g. `intended_paths_only` | `any`).
    pub diff_policy: String,
    /// Default per-task timeout when a task does not override it.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}

const fn default_timeout() -> u64 {
    120
}

/// The lane catalog (`lanes.toml`): one policy per lane, keyed by lane token.
#[derive(Debug, Clone, Deserialize)]
pub struct LaneCatalog {
    /// Catalog schema version.
    pub schema_version: String,
    /// Lane token → policy. A lane is *active* exactly when it appears here, so
    /// a task may only target a lane the catalog declares.
    pub lanes: BTreeMap<String, LanePolicy>,
}

impl LaneCatalog {
    /// Look up a lane's policy by its [`BenchmarkLane`].
    #[must_use]
    pub fn policy(&self, lane: BenchmarkLane) -> Option<&LanePolicy> {
        self.lanes.get(lane.as_str())
    }

    /// Parse a catalog from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    /// Load and parse `lanes.toml` from a path.
    pub fn load(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_toml(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }
}

/// A discovered task: its fixture directory and the parsed manifest.
#[derive(Debug, Clone)]
pub struct DiscoveredTask {
    /// The fixture directory containing `task.toml`.
    pub dir: PathBuf,
    /// The parsed manifest.
    pub manifest: TaskManifest,
}

/// Discover every `task.toml` directly under `fixtures_root/<name>/`, sorted by
/// task id for stable, replayable iteration.
///
/// A directory without a `task.toml` is skipped (it may be a shared helper, not
/// a task), so the suite grows by dropping in a fixture directory.
pub fn discover_tasks(fixtures_root: &Path) -> io::Result<Vec<DiscoveredTask>> {
    let mut tasks = Vec::new();
    for entry in fs::read_dir(fixtures_root)? {
        let dir = entry?.path();
        if !dir.is_dir() {
            continue;
        }
        let manifest_path = dir.join("task.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let text = fs::read_to_string(&manifest_path)?;
        let manifest = TaskManifest::from_toml(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", manifest_path.display()),
            )
        })?;
        tasks.push(DiscoveredTask { dir, manifest });
    }
    tasks.sort_by(|a, b| a.manifest.id.cmp(&b.manifest.id));
    Ok(tasks)
}

/// Canonical coverage taxonomy for fixture metadata. Tags intentionally describe
/// what the fixture actually exercises; absence of a P1/P2/P3 tag is signal, not
/// a failure, until a real fixture for that regression axis exists.
pub const KNOWN_COVERAGE_TAGS: &[&str] = &[
    "async-rollback",
    "cross-file-rename",
    "edge-case",
    "general-deep-loop",
    "general-fast-loop",
    "hidden-invariant",
    "js-unit-test",
    "multi-stage-debug",
    "schema-propagation",
    "streaming-parser",
];

#[must_use]
fn is_known_coverage_tag(tag: &str) -> bool {
    KNOWN_COVERAGE_TAGS.binary_search(&tag).is_ok()
}

#[must_use]
fn is_slug_tag(tag: &str) -> bool {
    let bytes = tag.as_bytes();
    !bytes.is_empty()
        && bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.last().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.iter().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
        && !tag.contains("--")
}

/// Validate one task manifest against the lane catalog; returns every problem
/// (empty == valid) so a fixture-admission step reports all issues at once
/// instead of failing on the first.
#[must_use]
pub fn validate_task(manifest: &TaskManifest, catalog: &LaneCatalog) -> Vec<String> {
    let mut problems = Vec::new();
    if manifest.id.trim().is_empty() {
        problems.push("id is empty".to_string());
    }
    if manifest.prompt.trim().is_empty() {
        problems.push("prompt is empty".to_string());
    }
    if manifest.test_command.trim().is_empty() {
        problems.push("test_command is empty".to_string());
    }
    let mut seen_coverage_tags = std::collections::BTreeSet::new();
    for tag in &manifest.coverage_tags {
        if tag.trim() != tag || !is_slug_tag(tag) {
            problems.push(format!("coverage tag '{tag}' is not a canonical slug"));
        } else if !is_known_coverage_tag(tag) {
            problems.push(format!("unknown coverage tag '{tag}'"));
        }
        if !seen_coverage_tags.insert(tag) {
            problems.push(format!("duplicate coverage tag '{tag}'"));
        }
    }
    match manifest.parsed_lane() {
        None => problems.push(format!("unknown lane '{}'", manifest.lane)),
        Some(lane) => {
            if catalog.policy(lane).is_none() {
                problems.push(format!(
                    "lane '{}' is not declared in lanes.toml",
                    manifest.lane
                ));
            }
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> LaneCatalog {
        LaneCatalog::from_toml(
            r#"
schema_version = "1.0"

[lanes.fast]
objective_gate = "test_and_diff"
verifier_policy = "none"
retry_budget = 0
diff_policy = "intended_paths_only"

[lanes.deep]
objective_gate = "test_and_diff"
verifier_policy = "strict"
retry_budget = 2
diff_policy = "intended_paths_only"
timeout_seconds = 300
"#,
        )
        .expect("catalog parses")
    }

    #[test]
    fn parses_minimal_task_manifest() {
        let manifest = TaskManifest::from_toml(
            r#"
schema_version = "1.0"
id = "off-by-one"
lane = "fast"
prompt = "A test in test/range.test.js fails. Fix the bug in src/range.js."
test_command = "node --test"
coverage_tags = ["general-fast-loop", "js-unit-test", "edge-case"]
intended_paths = ["src/range.js"]
"#,
        )
        .expect("manifest parses");
        assert_eq!(manifest.id, "off-by-one");
        assert_eq!(manifest.parsed_lane(), Some(BenchmarkLane::Fast));
        assert_eq!(manifest.intended_paths, vec!["src/range.js"]);
        assert_eq!(
            manifest.coverage_tags,
            vec!["general-fast-loop", "js-unit-test", "edge-case"]
        );
        // Unset optionals default cleanly.
        assert_eq!(manifest.timeout_seconds, None);
        assert!(manifest.tests_that_must_flip.is_empty());
        assert!(manifest.oracle_solution.is_none());
    }

    #[test]
    fn parses_lane_catalog_and_looks_up_policy() {
        let catalog = sample_catalog();
        let fast = catalog.policy(BenchmarkLane::Fast).expect("fast policy");
        assert_eq!(fast.verifier_policy, "none");
        assert_eq!(fast.retry_budget, 0);
        // Default timeout applies when the lane omits it.
        assert_eq!(fast.timeout_seconds, 120);

        let deep = catalog.policy(BenchmarkLane::Deep).expect("deep policy");
        assert_eq!(deep.retry_budget, 2);
        assert_eq!(deep.timeout_seconds, 300);

        // A lane absent from the catalog has no policy.
        assert!(catalog.policy(BenchmarkLane::Migration).is_none());
    }

    #[test]
    fn validate_passes_a_clean_manifest() {
        let manifest = TaskManifest::from_toml(
            "schema_version=\"1.0\"\nid=\"t\"\nlane=\"deep\"\nprompt=\"p\"\ntest_command=\"npm test\"\n",
        )
        .unwrap();
        assert!(validate_task(&manifest, &sample_catalog()).is_empty());
    }

    #[test]
    fn validate_flags_unknown_lane_and_undeclared_lane() {
        let catalog = sample_catalog();

        let unknown = TaskManifest::from_toml(
            "schema_version=\"1.0\"\nid=\"t\"\nlane=\"nope\"\nprompt=\"p\"\ntest_command=\"c\"\n",
        )
        .unwrap();
        let problems = validate_task(&unknown, &catalog);
        assert!(problems.iter().any(|p| p.contains("unknown lane")));

        // 'migration' is a real lane name but is not declared in this catalog.
        let undeclared = TaskManifest::from_toml(
            "schema_version=\"1.0\"\nid=\"t\"\nlane=\"migration\"\nprompt=\"p\"\ntest_command=\"c\"\n",
        )
        .unwrap();
        let problems = validate_task(&undeclared, &catalog);
        assert!(problems.iter().any(|p| p.contains("not declared")));
    }

    #[test]
    fn validate_flags_empty_required_fields() {
        let blank = TaskManifest::from_toml(
            "schema_version=\"1.0\"\nid=\"  \"\nlane=\"fast\"\nprompt=\"\"\ntest_command=\"  \"\n",
        )
        .unwrap();
        let problems = validate_task(&blank, &sample_catalog());
        assert!(problems.iter().any(|p| p.contains("id is empty")));
        assert!(problems.iter().any(|p| p.contains("prompt is empty")));
        assert!(problems.iter().any(|p| p.contains("test_command is empty")));
    }

    #[test]
    fn validate_flags_invalid_unknown_and_duplicate_coverage_tags() {
        let manifest = TaskManifest::from_toml(
            r#"schema_version="1.0"
id="t"
lane="fast"
prompt="p"
test_command="c"
coverage_tags=["general-fast-loop", "Bad Tag", "not-in-taxonomy", "general-fast-loop"]
"#,
        )
        .unwrap();
        let problems = validate_task(&manifest, &sample_catalog());
        assert!(problems
            .iter()
            .any(|p| p.contains("not a canonical slug")));
        assert!(problems.iter().any(|p| p.contains("unknown coverage tag")));
        assert!(problems.iter().any(|p| p.contains("duplicate coverage tag")));
    }

    #[test]
    fn discover_finds_and_sorts_task_tomls_skipping_bare_dirs() {
        let root = std::env::temp_dir().join(format!("zo-disc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let make = |name: &str, id: &str| {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("task.toml"),
                format!(
                    "schema_version=\"1.0\"\nid=\"{id}\"\nlane=\"fast\"\nprompt=\"p\"\ntest_command=\"node --test\"\n"
                ),
            )
            .unwrap();
        };
        make("zeta", "zeta-task");
        make("alpha", "alpha-task");
        // A directory with no task.toml is skipped, not an error.
        fs::create_dir_all(root.join("shared-helper")).unwrap();

        let tasks = discover_tasks(&root).expect("discovery succeeds");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].manifest.id, "alpha-task");
        assert_eq!(tasks[1].manifest.id, "zeta-task");
    }
}
