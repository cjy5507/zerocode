use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use std::time::Instant;

use commands::{CommandManifestEntry, CommandRegistry, CommandSource};
use runtime::{BootstrapPhase, BootstrapPlan};
use tools::{ToolManifestEntry, ToolRegistry, ToolSource};

pub mod deep;
pub mod deep_lane;
pub mod diff_hygiene;
pub mod fairness;
pub mod manifest;
pub mod regression;
pub mod runner;
pub mod semantic_probes;
pub mod suite;
pub mod summary;

pub use decision_core::decision::{
    decide_final, BenchmarkLane, FailureClass, FairnessStatus, FinalDecision, ObjectiveGate,
    RunVerdict, VerifierDecision,
};
pub use deep::{DeepConfig, DeepVerdict};
pub use deep_lane::{
    decide, failure_summary, fold_verification_attempt, objective_passed, parse_verifier,
    validate_plan, DeepDecision, PlanVerdict, VerificationAttempt, VerifierParse, VerifierVerdict,
};
pub use diff_hygiene::{
    fail_reasons, permission_denial_fatal, run_passed, score as score_diff_hygiene, warnings,
    DiffHygiene, StatusEntry, TestStatus,
};
pub use fairness::{
    build_contract, normalize_effort, normalize_model_family, sha256_hex, FairnessContract,
    FairnessInput,
};
pub use manifest::{
    discover_tasks, validate_task, DiscoveredTask, LaneCatalog, LanePolicy, TaskManifest,
};
pub use regression::{
    CaseComparison, PerfBaseline, PerfBaselineFile, PerfVerdict, RegressionGate, SuiteReport,
};
pub use runner::{
    normalize_effort_label, normalize_model_label, run_one, RunMetrics, RunResult, RunSpec, Tokens,
};
pub use suite::{run_suite, SuiteConfig, SuiteRunner};
pub use summary::{parse_ledger, summarize_ledger, LaneRunnerSummary, LedgerRow};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamPaths {
    repo_root: PathBuf,
}

impl UpstreamPaths {
    #[must_use]
    pub fn from_repo_root(repo_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_root: repo_root.into(),
        }
    }

    #[must_use]
    pub fn from_workspace_dir(workspace_dir: impl AsRef<Path>) -> Self {
        let workspace_dir = workspace_dir
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| workspace_dir.as_ref().to_path_buf());
        let primary_repo_root = workspace_dir
            .parent()
            .map_or_else(|| PathBuf::from(".."), Path::to_path_buf);
        let repo_root = resolve_upstream_repo_root(&primary_repo_root);
        Self { repo_root }
    }

    #[must_use]
    pub fn commands_path(&self) -> PathBuf {
        self.repo_root.join("src/commands.ts")
    }

    #[must_use]
    pub fn tools_path(&self) -> PathBuf {
        self.repo_root.join("src/tools.ts")
    }

    #[must_use]
    pub fn cli_path(&self) -> PathBuf {
        self.repo_root.join("src/entrypoints/cli.tsx")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedManifest {
    pub commands: CommandRegistry,
    pub tools: ToolRegistry,
    pub bootstrap: BootstrapPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfCase {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub warmups: usize,
    pub runs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfSample {
    pub elapsed: Duration,
    pub peak_rss_kib: Option<u64>,
    pub status_code: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerfSummary {
    pub name: String,
    pub runs: usize,
    pub min_elapsed: Duration,
    pub max_elapsed: Duration,
    pub mean_elapsed: Duration,
    /// Median (p50) — robust central tendency. The regression gate compares
    /// this rather than the mean because a single slow run (OS scheduling,
    /// page fault, cold cache) drags the mean but barely moves the median.
    pub median_elapsed: Duration,
    /// Sample standard deviation of elapsed time. Persisted into the baseline
    /// so a captured measurement carries its own noise context.
    pub stddev_elapsed: Duration,
    pub peak_rss_kib: Option<u64>,
}

pub struct PerfHarness;

impl PerfHarness {
    pub fn run(case: &PerfCase) -> std::io::Result<PerfSummary> {
        for _ in 0..case.warmups {
            let sample = run_perf_sample(case)?;
            if sample.status_code != Some(0) {
                return Err(std::io::Error::other(format!(
                    "warmup command `{}` exited with {:?}",
                    case.name, sample.status_code
                )));
            }
        }

        let run_count = case.runs.max(1);
        let mut samples = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            let sample = run_perf_sample(case)?;
            if sample.status_code != Some(0) {
                return Err(std::io::Error::other(format!(
                    "measured command `{}` exited with {:?}",
                    case.name, sample.status_code
                )));
            }
            samples.push(sample);
        }

        Ok(summarize_perf_samples(&case.name, &samples))
    }
}

#[must_use]
pub fn render_perf_summary_json(summary: &PerfSummary) -> String {
    let peak = summary
        .peak_rss_kib
        .map_or_else(|| "null".to_string(), |value| value.to_string());
    format!(
        "{{\"name\":\"{}\",\"runs\":{},\"min_elapsed_micros\":{},\"median_elapsed_micros\":{},\"mean_elapsed_micros\":{},\"max_elapsed_micros\":{},\"stddev_elapsed_micros\":{},\"peak_rss_kib\":{}}}",
        escape_json_string(&summary.name),
        summary.runs,
        summary.min_elapsed.as_micros(),
        summary.median_elapsed.as_micros(),
        summary.mean_elapsed.as_micros(),
        summary.max_elapsed.as_micros(),
        summary.stddev_elapsed.as_micros(),
        peak
    )
}

fn escape_json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                let _ = write!(escaped, "\\u{:04x}", ch as u32);
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn run_perf_sample(case: &PerfCase) -> std::io::Result<PerfSample> {
    let started = Instant::now();
    // Silence the child's own stdout/stderr: the harness measures wall-clock
    // and RSS, and a chatty command (e.g. `zo --help`) would otherwise drown
    // the harness's own report. stdin is closed so nothing blocks on a prompt.
    let mut child = Command::new(&case.command)
        .args(&case.args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let mut peak_rss_kib = sample_child_rss_kib(&child);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        peak_rss_kib = max_optional(peak_rss_kib, sample_child_rss_kib(&child));
        std::thread::sleep(Duration::from_millis(1));
    };
    peak_rss_kib = max_optional(peak_rss_kib, sample_child_rss_kib(&child));

    Ok(PerfSample {
        elapsed: started.elapsed(),
        peak_rss_kib,
        status_code: status.code(),
    })
}

fn summarize_perf_samples(name: &str, samples: &[PerfSample]) -> PerfSummary {
    let runs = samples.len();
    let mut min_elapsed = Duration::MAX;
    let mut max_elapsed = Duration::ZERO;
    let mut total_nanos = 0_u128;
    let mut peak_rss_kib = None;

    for sample in samples {
        min_elapsed = min_elapsed.min(sample.elapsed);
        max_elapsed = max_elapsed.max(sample.elapsed);
        total_nanos = total_nanos.saturating_add(sample.elapsed.as_nanos());
        peak_rss_kib = max_optional(peak_rss_kib, sample.peak_rss_kib);
    }

    let mean_elapsed = if runs == 0 {
        Duration::ZERO
    } else {
        duration_from_nanos(total_nanos / runs as u128)
    };

    PerfSummary {
        name: name.to_string(),
        runs,
        min_elapsed: if runs == 0 {
            Duration::ZERO
        } else {
            min_elapsed
        },
        max_elapsed,
        mean_elapsed,
        median_elapsed: median_elapsed(samples),
        stddev_elapsed: stddev_elapsed(samples, mean_elapsed),
        peak_rss_kib,
    }
}

/// p50 of the elapsed samples. Even sample counts average the two middle
/// values. Empty input yields `Duration::ZERO`.
fn median_elapsed(samples: &[PerfSample]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut nanos: Vec<u128> = samples.iter().map(|s| s.elapsed.as_nanos()).collect();
    nanos.sort_unstable();
    let mid = nanos.len() / 2;
    let median = if nanos.len() % 2 == 1 {
        nanos[mid]
    } else {
        u128::midpoint(nanos[mid - 1], nanos[mid])
    };
    duration_from_nanos(median)
}

/// Sample standard deviation (Bessel-corrected, `n - 1`). Fewer than two
/// samples have no dispersion, so the result is `Duration::ZERO`.
///
/// Statistics need floating point for the square root; nanosecond counts for
/// any realistic command duration (< 100 days) are well within f64's exact
/// integer range, so the casts do not lose meaningful precision.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn stddev_elapsed(samples: &[PerfSample], mean: Duration) -> Duration {
    if samples.len() < 2 {
        return Duration::ZERO;
    }
    let mean_nanos = mean.as_nanos() as f64;
    let variance = samples
        .iter()
        .map(|sample| {
            let diff = sample.elapsed.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    let stddev = variance.sqrt();
    if stddev.is_finite() && stddev >= 0.0 {
        duration_from_nanos(stddev as u128)
    } else {
        Duration::ZERO
    }
}

fn duration_from_nanos(nanos: u128) -> Duration {
    let bounded = u64::try_from(nanos).unwrap_or(u64::MAX);
    Duration::from_nanos(bounded)
}

fn max_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn sample_child_rss_kib(child: &Child) -> Option<u64> {
    let pid = child.id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
}

fn resolve_upstream_repo_root(primary_repo_root: &Path) -> PathBuf {
    let candidates = upstream_repo_candidates(primary_repo_root);
    candidates
        .into_iter()
        .find(|candidate| candidate.join("src/commands.ts").is_file())
        .unwrap_or_else(|| primary_repo_root.to_path_buf())
}

fn upstream_repo_candidates(primary_repo_root: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![primary_repo_root.to_path_buf()];

    if let Some(explicit) = std::env::var_os("CLAUDE_CODE_UPSTREAM") {
        candidates.push(PathBuf::from(explicit));
    }

    for ancestor in primary_repo_root.ancestors().take(4) {
        candidates.push(ancestor.join("zo"));
        candidates.push(ancestor.join("zo"));
    }

    candidates.push(
        primary_repo_root
            .join("reference-source")
            .join("zo"),
    );
    candidates.push(primary_repo_root.join("vendor").join("zo"));

    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped.iter().any(|seen: &PathBuf| seen == &candidate) {
            deduped.push(candidate);
        }
    }
    deduped
}

pub fn extract_manifest(paths: &UpstreamPaths) -> std::io::Result<ExtractedManifest> {
    let commands_source = fs::read_to_string(paths.commands_path())?;
    let tools_source = fs::read_to_string(paths.tools_path())?;
    let cli_source = fs::read_to_string(paths.cli_path())?;

    Ok(ExtractedManifest {
        commands: extract_commands(&commands_source),
        tools: extract_tools(&tools_source),
        bootstrap: extract_bootstrap_plan(&cli_source),
    })
}

#[must_use]
pub fn extract_commands(source: &str) -> CommandRegistry {
    let mut entries = Vec::new();
    let mut in_internal_block = false;

    for raw_line in source.lines() {
        let line = raw_line.trim();

        if line.starts_with("export const INTERNAL_ONLY_COMMANDS = [") {
            in_internal_block = true;
            continue;
        }

        if in_internal_block {
            if line.starts_with(']') {
                in_internal_block = false;
                continue;
            }
            if let Some(name) = first_identifier(line) {
                entries.push(CommandManifestEntry {
                    name,
                    source: CommandSource::InternalOnly,
                });
            }
            continue;
        }

        if line.starts_with("import ") {
            for imported in imported_symbols(line) {
                entries.push(CommandManifestEntry {
                    name: imported,
                    source: CommandSource::Builtin,
                });
            }
        }

        if line.contains("feature('") && line.contains("./commands/") {
            if let Some(name) = first_assignment_identifier(line) {
                entries.push(CommandManifestEntry {
                    name,
                    source: CommandSource::FeatureGated,
                });
            }
        }
    }

    dedupe_commands(entries)
}

#[must_use]
pub fn extract_tools(source: &str) -> ToolRegistry {
    let mut entries = Vec::new();

    for raw_line in source.lines() {
        let line = raw_line.trim();
        if line.starts_with("import ") && line.contains("./tools/") {
            for imported in imported_symbols(line) {
                if imported.ends_with("Tool") {
                    entries.push(ToolManifestEntry {
                        name: imported,
                        source: ToolSource::Base,
                    });
                }
            }
        }

        if line.contains("feature('") && line.contains("Tool") {
            if let Some(name) = first_assignment_identifier(line) {
                if name.ends_with("Tool") || name.ends_with("Tools") {
                    entries.push(ToolManifestEntry {
                        name,
                        source: ToolSource::Conditional,
                    });
                }
            }
        }
    }

    dedupe_tools(entries)
}

#[must_use]
pub fn extract_bootstrap_plan(source: &str) -> BootstrapPlan {
    let mut phases = vec![BootstrapPhase::CliEntry];

    if source.contains("--version") {
        phases.push(BootstrapPhase::FastPathVersion);
    }
    if source.contains("startupProfiler") {
        phases.push(BootstrapPhase::StartupProfiler);
    }
    if source.contains("--dump-system-prompt") {
        phases.push(BootstrapPhase::SystemPromptFastPath);
    }
    if source.contains("--claude-in-chrome-mcp") {
        phases.push(BootstrapPhase::ChromeMcpFastPath);
    }
    if source.contains("--daemon-worker") {
        phases.push(BootstrapPhase::DaemonWorkerFastPath);
    }
    if source.contains("remote-control") {
        phases.push(BootstrapPhase::BridgeFastPath);
    }
    if source.contains("args[0] === 'daemon'") {
        phases.push(BootstrapPhase::DaemonFastPath);
    }
    if source.contains("args[0] === 'ps'") || source.contains("args.includes('--bg')") {
        phases.push(BootstrapPhase::BackgroundSessionFastPath);
    }
    if source.contains("args[0] === 'new' || args[0] === 'list' || args[0] === 'reply'") {
        phases.push(BootstrapPhase::TemplateFastPath);
    }
    if source.contains("environment-runner") {
        phases.push(BootstrapPhase::EnvironmentRunnerFastPath);
    }
    phases.push(BootstrapPhase::MainRuntime);

    BootstrapPlan::from_phases(phases)
}

fn imported_symbols(line: &str) -> Vec<String> {
    let Some(after_import) = line.strip_prefix("import ") else {
        return Vec::new();
    };

    let before_from = after_import
        .split(" from ")
        .next()
        .unwrap_or_default()
        .trim();
    if before_from.starts_with('{') {
        return before_from
            .trim_matches(|c| c == '{' || c == '}')
            .split(',')
            .filter_map(|part| {
                let trimmed = part.trim();
                if trimmed.is_empty() {
                    return None;
                }
                Some(trimmed.split_whitespace().next()?.to_string())
            })
            .collect();
    }

    let first = before_from.split(',').next().unwrap_or_default().trim();
    if first.is_empty() {
        Vec::new()
    } else {
        vec![first.to_string()]
    }
}

fn first_assignment_identifier(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let candidate = trimmed.split('=').next()?.trim();
    first_identifier(candidate)
}

fn first_identifier(line: &str) -> Option<String> {
    let mut out = String::new();
    for ch in line.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else if !out.is_empty() {
            break;
        }
    }
    (!out.is_empty()).then_some(out)
}

fn dedupe_commands(entries: Vec<CommandManifestEntry>) -> CommandRegistry {
    let mut deduped = Vec::new();
    for entry in entries {
        let exists = deduped.iter().any(|seen: &CommandManifestEntry| {
            seen.name == entry.name && seen.source == entry.source
        });
        if !exists {
            deduped.push(entry);
        }
    }
    CommandRegistry::new(deduped)
}

fn dedupe_tools(entries: Vec<ToolManifestEntry>) -> ToolRegistry {
    let mut deduped = Vec::new();
    for entry in entries {
        let exists = deduped
            .iter()
            .any(|seen: &ToolManifestEntry| seen.name == entry.name && seen.source == entry.source);
        if !exists {
            deduped.push(entry);
        }
    }
    ToolRegistry::new(deduped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_paths() -> UpstreamPaths {
        let workspace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        UpstreamPaths::from_workspace_dir(workspace_dir)
    }

    fn has_upstream_fixture(paths: &UpstreamPaths) -> bool {
        paths.commands_path().is_file()
            && paths.tools_path().is_file()
            && paths.cli_path().is_file()
    }

    #[test]
    fn extracts_non_empty_manifests_from_upstream_repo() {
        let paths = fixture_paths();
        if !has_upstream_fixture(&paths) {
            return;
        }
        let manifest = extract_manifest(&paths).expect("manifest should load");
        assert!(!manifest.commands.entries().is_empty());
        assert!(!manifest.tools.entries().is_empty());
        assert!(!manifest.bootstrap.phases().is_empty());
    }

    #[test]
    fn detects_known_upstream_command_symbols() {
        let paths = fixture_paths();
        if !paths.commands_path().is_file() {
            return;
        }
        let commands =
            extract_commands(&fs::read_to_string(paths.commands_path()).expect("commands.ts"));
        let names: Vec<_> = commands
            .entries()
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert!(names.contains(&"addDir"));
        assert!(names.contains(&"review"));
        assert!(!names.contains(&"INTERNAL_ONLY_COMMANDS"));
    }

    #[test]
    fn detects_known_upstream_tool_symbols() {
        let paths = fixture_paths();
        if !paths.tools_path().is_file() {
            return;
        }
        let tools = extract_tools(&fs::read_to_string(paths.tools_path()).expect("tools.ts"));
        let names: Vec<_> = tools
            .entries()
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert!(names.contains(&"AgentTool"));
        assert!(names.contains(&"BashTool"));
    }

    #[test]
    fn perf_harness_summarizes_command_latency_and_memory() {
        let case = PerfCase {
            name: "echo-smoke".to_string(),
            command: "sh".to_string(),
            args: vec!["-c".to_string(), ":".to_string()],
            warmups: 1,
            runs: 3,
        };

        let summary = PerfHarness::run(&case).expect("perf command should run");

        assert_eq!(summary.name, "echo-smoke");
        assert_eq!(summary.runs, 3);
        assert!(summary.min_elapsed <= summary.mean_elapsed);
        assert!(summary.mean_elapsed <= summary.max_elapsed);
        assert!(summary.min_elapsed <= summary.median_elapsed);
        assert!(summary.median_elapsed <= summary.max_elapsed);
    }

    #[test]
    fn perf_summary_uses_peak_rss_when_available() {
        let summary = summarize_perf_samples(
            "fixture",
            &[
                PerfSample {
                    elapsed: Duration::from_millis(3),
                    peak_rss_kib: Some(10),
                    status_code: Some(0),
                },
                PerfSample {
                    elapsed: Duration::from_millis(9),
                    peak_rss_kib: Some(30),
                    status_code: Some(0),
                },
            ],
        );

        assert_eq!(summary.min_elapsed, Duration::from_millis(3));
        assert_eq!(summary.max_elapsed, Duration::from_millis(9));
        assert_eq!(summary.mean_elapsed, Duration::from_millis(6));
        // Median of [3ms, 9ms] is their average; stddev (n-1) is sqrt(18)ms ≈ 4.24ms.
        assert_eq!(summary.median_elapsed, Duration::from_millis(6));
        assert!(summary.stddev_elapsed >= Duration::from_millis(4));
        assert!(summary.stddev_elapsed < Duration::from_millis(5));
        assert_eq!(summary.peak_rss_kib, Some(30));
    }

    #[test]
    fn perf_summary_renders_stable_json() {
        let summary = PerfSummary {
            name: "zo \"version\"".to_string(),
            runs: 5,
            min_elapsed: Duration::from_micros(10),
            mean_elapsed: Duration::from_micros(20),
            max_elapsed: Duration::from_micros(30),
            median_elapsed: Duration::from_micros(18),
            stddev_elapsed: Duration::from_micros(7),
            peak_rss_kib: Some(4096),
        };

        assert_eq!(
            render_perf_summary_json(&summary),
            "{\"name\":\"zo \\\"version\\\"\",\"runs\":5,\"min_elapsed_micros\":10,\"median_elapsed_micros\":18,\"mean_elapsed_micros\":20,\"max_elapsed_micros\":30,\"stddev_elapsed_micros\":7,\"peak_rss_kib\":4096}"
        );
    }
}
