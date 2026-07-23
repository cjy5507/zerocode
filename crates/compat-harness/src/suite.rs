//! Native benchmark suite orchestrator — the Rust replacement for
//! `agent-eval-suite.sh` + `agent-eval-harness.sh`.
//!
//! Discovers `task.toml` fixtures, runs each `(runner × task)` through the native
//! [`crate::runner::run_one`], builds the per-run fairness contract and final
//! decision from the already-unit-tested rules ([`build_contract`],
//! [`decide_final`]), and rolls up lane×runner denominators
//! ([`summarize_ledger`]). No shell, no `jq`, no `deep-eval` subprocess: every
//! step is a direct function call, so the pipeline can never drift from its tests.
//!
//! Two correctness wins over the shell suite:
//! - the fairness contract's run conditions (permission mode, runner/harness
//!   version) are filled, so a complete run is `valid` rather than `partial`;
//! - `artifacts_preserved` is a **real file-existence check**, not the shell's
//!   hardcoded `true` — an accepted run with no preserved artifact is caught.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use serde_json::json;
use sha2::Digest;
use sha2::Sha256;

use decision_core::decision::decide_final;
use decision_core::decision::FailureClass;
use decision_core::decision::FairnessStatus;
use decision_core::decision::FinalDecision;
use decision_core::decision::ObjectiveGate;
use decision_core::decision::RunVerdict;
use decision_core::decision::VerifierDecision;
use decision_core::deep_lane::VerifierParse;

use crate::deep::DeepConfig;
use crate::fairness::build_contract;
use crate::fairness::FairnessContract;
use crate::fairness::FairnessInput;
use crate::manifest::discover_tasks;
use crate::manifest::validate_task;
use crate::manifest::DiscoveredTask;
use crate::manifest::LaneCatalog;
use crate::manifest::TaskManifest;
use crate::runner::normalize_effort_label;
use crate::runner::normalize_model_label;
use crate::runner::required_artifacts_present;
use crate::runner::run_one;
use crate::runner::spawned_process_groups_snapshot;
use crate::runner::RunMetrics;
use crate::runner::RunResult;
use crate::runner::RunSpec;
use crate::summary::summarize_ledger;
use crate::summary::LedgerRow;

const SCORER_VERSION: &str = "compat-harness-native-scorer-v1";
const REPORT_GENERATOR_VERSION: &str = "compat-harness-native-report-v1";

/// One runner's declared identity and how to invoke it.
pub struct SuiteRunner {
    pub name: String,
    pub kind: String,
    pub bin: PathBuf,
    pub args: Vec<String>,
    pub model_label: String,
    pub effort_label: String,
    pub permission_mode: String,
    pub version: String,
}

/// Everything one suite run needs. The caller stamps ambient provenance
/// (`timestamp`/`git_commit`/`command_invocation`), keeping this module free of
/// ambient IO.
pub struct SuiteConfig {
    pub fixtures: PathBuf,
    pub lanes: PathBuf,
    pub runners: Vec<SuiteRunner>,
    pub out_dir: PathBuf,
    pub suite_version: String,
    pub timestamp: String,
    pub git_commit: String,
    pub command_invocation: Option<Vec<String>>,
    /// How many times to run every `(task × runner)` cell. `1` (the default)
    /// keeps the single-shot layout; `N > 1` runs each cell N times — writing the
    /// k-th run under `<runner>/<lane>-<id>/rep-k` — and emits `repeats.json` with
    /// per-cell pass@N and median wall/tokens. Repeated measurement is what turns
    /// a one-off number into a trustworthy self-improvement signal (a single run
    /// conflates the model's variance with a real regression).
    pub repeat: usize,
}

/// Run the whole suite: every discovered task × every runner, scored and rolled
/// up. Writes per-run `result.json` / `fairness_contract.json` / `decide.json`
/// plus `ledger.jsonl`, `summary.json`, and `manifest.json` under `out_dir`.
///
/// # Errors
/// Propagates I/O errors from discovery, running, or writing artifacts. The lane
/// catalog (`lanes.toml`) is **required**: a missing or unparseable catalog is
/// fatal (risk R0a) rather than a silent `None`, because lane policy
/// (verifier strictness, retry budget, timeout) is load-bearing — a suite run
/// without it would mis-score deep lanes and lose timeout fallback.
pub fn run_suite(cfg: &SuiteConfig) -> io::Result<()> {
    let catalog = LaneCatalog::load(&cfg.lanes)?;
    let tasks = discover_tasks(&cfg.fixtures)?;
    validate_discovered_tasks(&tasks, &catalog)?;
    fs::create_dir_all(&cfg.out_dir)?;

    let mut ledger_rows: Vec<LedgerRow> = Vec::new();
    let mut ledger_text = String::new();

    let repeat = cfg.repeat.max(1);
    for task in &tasks {
        for runner in &cfg.runners {
            for rep in 0..repeat {
                let row = run_cell(cfg, task, runner, &catalog, rep, repeat)?;
                let _ = writeln!(ledger_text, "{}", row.line);
                ledger_rows.push(row.row);
            }
        }
    }

    fs::write(cfg.out_dir.join("ledger.jsonl"), &ledger_text)?;
    let summary = summarize_ledger(&ledger_rows);
    fs::write(
        cfg.out_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    // Cross-runner dominance leaderboard (per lane): accuracy + geomean token/wall
    // ratios + composite. Useful even at repeat=1; with ≥2 runners it is the
    // "who wins" ranking the four-way report otherwise computes by hand.
    let leaderboard = crate::summary::build_leaderboard_with_views(&ledger_rows);
    fs::write(
        cfg.out_dir.join("leaderboard.json"),
        serde_json::to_string_pretty(&leaderboard)?,
    )?;
    // pass@N + median wall/tokens per fixture cell — only meaningful with >1 run,
    // so it is emitted exactly when repeats were requested (the doc's recommended
    // next step). The lane×runner `summary.json` already aggregates every rep.
    if repeat > 1 {
        let repeats = crate::summary::summarize_repeats(&ledger_rows);
        fs::write(
            cfg.out_dir.join("repeats.json"),
            serde_json::to_string_pretty(&repeats)?,
        )?;
    }
    let fixture_coverage = fixture_coverage_summary(&tasks);
    fs::write(
        cfg.out_dir.join("manifest.json"),
        manifest_json(cfg, &fixture_coverage),
    )?;
    fs::write(
        cfg.out_dir.join("process_warnings.json"),
        serde_json::to_string_pretty(&detect_leftover_processes(cfg))?,
    )?;
    fs::write(
        cfg.out_dir.join("report.md"),
        render_report_markdown(cfg, &ledger_rows, &fixture_coverage),
    )?;
    Ok(())
}

fn validate_discovered_tasks(tasks: &[DiscoveredTask], catalog: &LaneCatalog) -> io::Result<()> {
    let mut problems = Vec::new();
    for task in tasks {
        problems.extend(
            validate_task(&task.manifest, catalog)
                .into_iter()
                .map(|problem| format!("{}: {problem}", task.dir.join("task.toml").display())),
        );
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        problems.join("
"),
    ))
}

/// One `(task, runner)` cell's outputs: the ledger row (for the rollup) and its
/// JSON line (for the on-disk ledger).
struct Cell {
    row: LedgerRow,
    line: String,
}

fn run_cell(
    cfg: &SuiteConfig,
    task: &DiscoveredTask,
    runner: &SuiteRunner,
    catalog: &LaneCatalog,
    rep: usize,
    repeat: usize,
) -> io::Result<Cell> {
    let m = &task.manifest;
    let mut run_dir = cfg
        .out_dir
        .join(&runner.name)
        .join(format!("{}-{}", m.lane, m.id));
    // Single-shot keeps the flat `<lane>-<id>` layout (back-compat); repeats nest
    // each run under `rep-k` so no run overwrites another's preserved artifacts.
    if repeat > 1 {
        run_dir = run_dir.join(format!("rep-{}", rep + 1));
    }
    fs::create_dir_all(&run_dir)?;

    // R0b admission gate: a fixture that fails validation never runs. It is
    // recorded as invalid (fixture_invalid) with its problems on disk, so the
    // summary denominator counts it instead of silently dropping it — or, worse,
    // letting an undeclared-lane fixture reach run_one and be accepted.
    let problems = validate_task(m, catalog);
    if !problems.is_empty() {
        return invalid_cell(&run_dir, runner, m, &problems);
    }
    let policy = catalog
        .policy(m.parsed_lane().expect("validated task lane"))
        .expect("validated lane policy");

    let spec = RunSpec {
        runner: runner.name.clone(),
        runner_kind: runner.kind.clone(),
        bin: runner.bin.clone(),
        args: runner.args.clone(),
        fixture: task.dir.clone(),
        prompt: m.prompt.clone(),
        test_command: Some(m.test_command.clone()),
        intended: m.intended_paths.clone(),
        lane: m.lane.clone(),
        model: normalize_model_label(&runner.model_label),
        effort: normalize_effort_label(&runner.effort_label),
        objective_gate: policy.objective_gate.clone(),
        diff_policy: policy.diff_policy.clone(),
        timeout_seconds: effective_timeout(m, catalog),
        artifacts_dir: Some(run_dir.join("artifacts")),
        keep_failed: false,
        deep: deep_config(catalog, m),
    };
    let result = run_one(&spec)?;
    fs::write(run_dir.join("result.json"), serde_json::to_string(&result)?)?;
    let metrics = result.metrics();
    fs::write(
        run_dir.join("metrics.json"),
        serde_json::to_string(&metrics)?,
    )?;

    let contract = build_fairness(cfg, runner, m, &task.dir, &result, catalog);
    fs::write(
        run_dir.join("fairness_contract.json"),
        serde_json::to_string(&contract)?,
    )?;

    let verdict = derive_decision(&result, &contract.status);
    fs::write(run_dir.join("decide.json"), verdict_json(verdict))?;

    Ok(cell_from_verdict(runner, m, verdict, Some(&metrics)))
}

/// Record a fixture that failed admission (risk R0b): write its problems and emit
/// an `invalid` / `fixture_invalid` ledger row **without running it**. Never
/// leaderboard-eligible — an unvalidated fixture cannot produce a trusted score.
fn invalid_cell(
    run_dir: &Path,
    runner: &SuiteRunner,
    m: &TaskManifest,
    problems: &[String],
) -> io::Result<Cell> {
    fs::write(
        run_dir.join("problems.json"),
        serde_json::to_string(&json!({ "fixture_invalid": problems }))?,
    )?;
    Ok(cell_from_verdict(
        runner,
        m,
        RunVerdict {
            decision: FinalDecision::Invalid,
            failure: Some(FailureClass::FixtureInvalid),
            leaderboard_eligible: false,
        },
        None,
    ))
}

/// Build the ledger row and its on-disk JSON line from a scored verdict. Shared
/// by the run path and the admission-rejected path so the two can never disagree
/// on the row shape.
fn cell_from_verdict(
    runner: &SuiteRunner,
    m: &TaskManifest,
    verdict: RunVerdict,
    metrics: Option<&RunMetrics>,
) -> Cell {
    let failure_class = verdict.failure.map(|f| f.as_str().to_string());
    let line = json!({
        "runner": runner.name,
        "lane": m.lane,
        "fixture": m.id,
        "final_decision": verdict.decision.as_str(),
        "failure_class": failure_class,
        "leaderboard_eligible": verdict.leaderboard_eligible,
        "wall_seconds": metrics.map(|m| m.wall_seconds),
        "token_total": metrics.and_then(|m| m.token_total),
        "token_output": metrics.and_then(|m| m.token_output),
        "total_tokens_per_second": metrics.and_then(|m| m.total_tokens_per_second),
        "output_tokens_per_second": metrics.and_then(|m| m.output_tokens_per_second),
        "deterministic_probe_failure_count": metrics.map(|m| m.deterministic_probe_failure_count),
        "phase_plan_millis": metrics.and_then(|m| m.phase_timings.as_ref().map(|t| t.plan_millis)),
        "phase_exec_millis": metrics.and_then(|m| m.phase_timings.as_ref().map(|t| t.exec_millis)),
        "phase_test_millis": metrics.and_then(|m| m.phase_timings.as_ref().map(|t| t.test_millis)),
        "phase_verify_millis": metrics.and_then(|m| m.phase_timings.as_ref().map(|t| t.verify_millis)),
        "phase_repair_millis": metrics.and_then(|m| m.phase_timings.as_ref().map(|t| t.repair_millis)),
        "retry_count": metrics.map(|m| m.retry_count),
        "dirty_diff": metrics.map(|m| m.dirty_diff),
        "deep_verifier_failed": metrics.map(|m| m.deep_verifier_failed),
        "timeout": metrics.map(|m| m.timeout),
        "provider_error": metrics.map(|m| m.provider_error),
        "cost_usd": metrics.and_then(|m| m.cost_usd),
        "cost_normalized_score": metrics.and_then(|m| m.cost_normalized_score),
    })
    .to_string();
    Cell {
        row: LedgerRow {
            runner: runner.name.clone(),
            lane: m.lane.clone(),
            fixture: m.id.clone(),
            final_decision: verdict.decision.as_str().to_string(),
            failure_class,
            leaderboard_eligible: verdict.leaderboard_eligible,
            wall_seconds: metrics.map(|m| m.wall_seconds),
            token_total: metrics.and_then(|m| m.token_total),
            token_output: metrics.and_then(|m| m.token_output),
            total_tokens_per_second: metrics.and_then(|m| m.total_tokens_per_second),
            output_tokens_per_second: metrics.and_then(|m| m.output_tokens_per_second),
            deterministic_probe_failure_count: metrics.map(|m| m.deterministic_probe_failure_count),
            phase_plan_millis: metrics
                .and_then(|m| m.phase_timings.as_ref().map(|t| t.plan_millis)),
            phase_exec_millis: metrics
                .and_then(|m| m.phase_timings.as_ref().map(|t| t.exec_millis)),
            phase_test_millis: metrics
                .and_then(|m| m.phase_timings.as_ref().map(|t| t.test_millis)),
            phase_verify_millis: metrics
                .and_then(|m| m.phase_timings.as_ref().map(|t| t.verify_millis)),
            phase_repair_millis: metrics
                .and_then(|m| m.phase_timings.as_ref().map(|t| t.repair_millis)),
            retry_count: metrics.map(|m| m.retry_count),
            dirty_diff: metrics.map(|m| m.dirty_diff),
            deep_verifier_failed: metrics.map(|m| m.deep_verifier_failed),
            timeout: metrics.map(|m| m.timeout),
            provider_error: metrics.map(|m| m.provider_error),
            cost_usd: metrics.and_then(|m| m.cost_usd),
            cost_normalized_score: metrics.and_then(|m| m.cost_normalized_score),
        },
        line,
    }
}

/// A lane whose policy demands a strict verifier runs the deep loop; its
/// `retry_budget` becomes the attempt cap (at least one).
fn deep_config(catalog: &LaneCatalog, m: &TaskManifest) -> Option<DeepConfig> {
    let policy = catalog.policy(m.parsed_lane()?)?;
    (policy.verifier_policy == "strict").then(|| DeepConfig {
        max_attempts: policy.retry_budget.max(1),
    })
}

/// Build the per-run fairness contract with the run conditions filled in, so a
/// complete run judges `valid` (the shell suite left these empty → `partial`).
fn build_fairness(
    cfg: &SuiteConfig,
    runner: &SuiteRunner,
    m: &TaskManifest,
    dir: &Path,
    result: &RunResult,
    catalog: &LaneCatalog,
) -> FairnessContract {
    let input = FairnessInput {
        runner: runner.name.clone(),
        lane: m.lane.clone(),
        fixture_id: m.id.clone(),
        fixture_commit: String::new(),
        fixture_tree_hash: fixture_tree_hash(dir),
        fixture_dirty_before: false,
        fixture_dirty_after: !result.clean_diff,
        prompt_path: String::new(),
        prompt: m.prompt.clone(),
        test_command: m.test_command.clone(),
        intended_path_set: m.intended_paths.clone(),
        declared_model: runner.model_label.clone(),
        declared_effort: runner.effort_label.clone(),
        permission_mode: runner.permission_mode.clone(),
        timeout_seconds: effective_timeout(m, catalog),
        runner_version: runner.version.clone(),
        harness_version: cfg.suite_version.clone(),
        benchmark_suite_version: cfg.suite_version.clone(),
        started_at: cfg.timestamp.clone(),
        finished_at: cfg.timestamp.clone(),
    };
    build_contract(&input)
}

/// The timeout the run actually had (risk R1): a task's own `timeout_seconds`
/// wins; absent that, it inherits its lane policy's `timeout_seconds` from
/// `lanes.toml` (deep = 300, fast = 120, …) so the fairness contract records the
/// real budget rather than a misleading `0`. `0` only when neither pins one.
fn effective_timeout(m: &TaskManifest, catalog: &LaneCatalog) -> u64 {
    m.timeout_seconds
        .or_else(|| {
            m.parsed_lane()
                .and_then(|lane| catalog.policy(lane))
                .map(|policy| policy.timeout_seconds)
        })
        .unwrap_or(0)
}

/// Map a scored run to the decision matrix. The fast lane has no verifier (the
/// objective gate IS the decision, modeled as a strict accept/reject); the deep
/// lane carries the verifier diagnostics. `artifacts_preserved` is a real check.
fn derive_decision(result: &RunResult, fairness_status: &str) -> RunVerdict {
    let fairness = FairnessStatus::from_token(fairness_status).unwrap_or(FairnessStatus::Unknown);

    let (objective, parse, accepted) = result.deep.as_ref().map_or_else(
        || {
            let green = result.pass && result.clean_diff;
            let obj = if green {
                ObjectiveGate::Green
            } else {
                ObjectiveGate::Red
            };
            (obj, VerifierParse::Json, green)
        },
        |deep| {
            let obj = if deep.diagnostics.objective_passed {
                ObjectiveGate::Green
            } else {
                ObjectiveGate::Red
            };
            let parse = VerifierParse::from_token(&deep.diagnostics.verifier_parse)
                .unwrap_or(VerifierParse::Empty);
            (obj, parse, deep.verifier_accepted)
        },
    );

    // risk 1 + R2: the *whole* required artifact set must be on disk, not just
    // one assumed file — an accepted run is replayable only if its full set
    // (agent output + git before/after/diff) survived. A deep run additionally
    // owes its plan and verifier raw output. The set is the SSOT in `runner`.
    let artifacts_preserved = result
        .artifact_dir
        .as_ref()
        .is_some_and(|d| required_artifacts_present(Path::new(d), result.deep.is_some()));

    let decision = VerifierDecision::from_verdict(accepted, parse);
    decide_final(fairness, objective, parse, decision, artifacts_preserved)
}

fn verdict_json(v: RunVerdict) -> String {
    json!({
        "final_decision": v.decision.as_str(),
        "failure_class": v.failure.map(FailureClass::as_str),
        "leaderboard_eligible": v.leaderboard_eligible,
    })
    .to_string()
}

fn detect_leftover_processes(cfg: &SuiteConfig) -> serde_json::Value {
    // Diagnostic-only and intentionally non-destructive. Scope the scan to
    // process groups that this harness actually spawned; this catches surviving
    // descendants even after their direct parent exited and they became orphans.
    #[cfg(unix)]
    {
        let _ = cfg;
        let groups = spawned_process_groups_snapshot();
        let output = Command::new("ps")
            .args(["-axo", "pid=,ppid=,pgid=,etime=,command="])
            .output();
        let rows = output
            .ok()
            .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
            .unwrap_or_default()
            .lines()
            .filter_map(|line| leftover_process_row(line, &groups))
            .collect::<Vec<_>>();
        json!({
            "leftover_processes": rows,
            "scan_scope": "recorded_runner_process_groups",
            "recorded_process_groups": groups.len(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = cfg;
        json!({
            "leftover_processes": [],
            "scan_scope": "unsupported_platform",
            "recorded_process_groups": 0,
        })
    }
}

#[cfg(unix)]
fn leftover_process_row(
    line: &str,
    groups: &[crate::runner::SpawnedProcessGroup],
) -> Option<serde_json::Value> {
    let mut parts = line.split_whitespace();
    let process_id = parts.next()?;
    let parent_id = parts.next()?;
    let group_id = parts.next()?;
    let elapsed = parts.next().unwrap_or("");
    let command = parts.collect::<Vec<_>>().join(" ");
    let group_id_num = group_id.parse::<u32>().ok()?;
    let group = groups.iter().find(|group| group.pgid == group_id_num)?;
    Some(json!({
        "pid": process_id,
        "ppid": parent_id,
        "pgid": group_id,
        "elapsed": elapsed,
        "command": command,
        "root_pid": group.root_pid,
        "label": group.label,
        "spawn_command": group.command,
    }))
}

fn render_report_markdown(
    cfg: &SuiteConfig,
    rows: &[LedgerRow],
    fixture_coverage: &FixtureCoverageSummary,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Deep eval report");
    let _ = writeln!(out);
    let _ = writeln!(out, "- Result dir: `{}`", cfg.out_dir.to_string_lossy());
    let _ = writeln!(out, "- Suite version: `{}`", cfg.suite_version);
    let _ = writeln!(out, "- Runs: {}", rows.len());
    let accepted = rows
        .iter()
        .filter(|r| r.final_decision == "accepted")
        .count();
    let _ = writeln!(out, "- Accepted: {accepted}/{}", rows.len());
    let token_total: u64 = rows.iter().filter_map(|r| r.token_total).sum();
    let wall_total: u64 = rows.iter().filter_map(|r| r.wall_seconds).sum();
    let probe_failures: usize = rows
        .iter()
        .filter_map(|r| r.deterministic_probe_failure_count)
        .sum();
    let _ = writeln!(out, "- Wall seconds total: {wall_total}");
    let _ = writeln!(out, "- Token total: {token_total}");
    let _ = writeln!(out, "- Deterministic probe failures: {probe_failures}");
    let _ = writeln!(out);
    let _ = writeln!(out, "## By runner");
    let summary = summarize_ledger(rows);
    for (key, cell) in summary {
        let _ = writeln!(
            out,
            "- `{key}`: accepted {}/{}, strict pass {:.2}%, median wall {:?}, tokens {}, output tokens {}, probe failures {}",
            cell.accepted,
            cell.attempted,
            cell.strict_pass_rate * 100.0,
            cell.median_wall_seconds,
            cell.total_token_usage,
            cell.total_output_token_usage,
            cell.deterministic_probe_failure_count
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "## Fixture coverage");
    let _ = writeln!(out, "- Tasks: {}", fixture_coverage.task_count);
    let _ = writeln!(out, "- Tagged tasks: {}", fixture_coverage.tagged_task_count);
    let _ = writeln!(out, "- Untagged tasks: {}", fixture_coverage.untagged_task_count);
    for (tag, count) in &fixture_coverage.tag_counts {
        let tasks = fixture_coverage
            .tag_tasks
            .get(tag)
            .map(|ids| ids.join(", "))
            .unwrap_or_default();
        let _ = writeln!(out, "- `{tag}`: {count} task(s) — {tasks}");
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Artifacts: `ledger.jsonl`, `summary.json`, `leaderboard.json`, `repeats.json` when repeated, and each run's `artifacts/` directory."
    );
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FixtureCoverageSummary {
    task_count: usize,
    tagged_task_count: usize,
    untagged_task_count: usize,
    tag_counts: BTreeMap<String, usize>,
    tag_tasks: BTreeMap<String, Vec<String>>,
}

fn fixture_coverage_summary(tasks: &[DiscoveredTask]) -> FixtureCoverageSummary {
    let mut tag_counts = BTreeMap::new();
    let mut tag_tasks: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut untagged_task_count = 0;
    for task in tasks {
        if task.manifest.coverage_tags.is_empty() {
            untagged_task_count += 1;
            continue;
        }
        let unique_tags = task
            .manifest
            .coverage_tags
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        for tag in unique_tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            tag_tasks
                .entry(tag.clone())
                .or_default()
                .push(task.manifest.id.clone());
        }
    }
    for tasks in tag_tasks.values_mut() {
        tasks.sort();
        tasks.dedup();
    }
    FixtureCoverageSummary {
        task_count: tasks.len(),
        tagged_task_count: tasks.len().saturating_sub(untagged_task_count),
        untagged_task_count,
        tag_counts,
        tag_tasks,
    }
}

fn manifest_json(cfg: &SuiteConfig, fixture_coverage: &FixtureCoverageSummary) -> String {
    let runners: Vec<&str> = cfg.runners.iter().map(|r| r.name.as_str()).collect();
    let runner_kinds: BTreeMap<&str, &str> = cfg
        .runners
        .iter()
        .map(|r| (r.name.as_str(), r.kind.as_str()))
        .collect();
    let runner_versions: BTreeMap<&str, &str> = cfg
        .runners
        .iter()
        .map(|r| (r.name.as_str(), r.version.as_str()))
        .collect();
    let ledger_path = cfg.out_dir.join("ledger.jsonl");
    json!({
        "result_dir": cfg.out_dir.to_string_lossy(),
        "created_at": cfg.timestamp,
        "harness_git_commit": cfg.git_commit,
        "benchmark_suite_version": cfg.suite_version,
        "fixture_set_hash": fixture_set_hash(&cfg.fixtures),
        "fixture_coverage": {
            "task_count": fixture_coverage.task_count,
            "tagged_task_count": fixture_coverage.tagged_task_count,
            "untagged_task_count": fixture_coverage.untagged_task_count,
            "tag_counts": fixture_coverage.tag_counts,
            "tag_tasks": fixture_coverage.tag_tasks,
        },
        "runners": runners,
        "runner_kinds": runner_kinds,
        "runner_versions": runner_versions,
        "command_invocation": cfg.command_invocation.as_ref(),
        "ledger_path": ledger_path.to_string_lossy(),
        "scorer_version": SCORER_VERSION,
        "report_generator_version": REPORT_GENERATOR_VERSION,
    })
    .to_string()
}

/// SHA-256 of a fixture's source tree (each file's relative path + bytes, sorted),
/// excluding `.git`, `node_modules`, and the manifest — pins the exact starting
/// state so the fairness contract identifies what the agent was handed.
fn fixture_tree_hash(dir: &Path) -> String {
    tree_hash(dir, false)
}

/// SHA-256 of the whole fixture set, including `task.toml` manifests. This pins
/// both the starting files and the benchmark instructions recorded in
/// `manifest.json`.
fn fixture_set_hash(dir: &Path) -> String {
    tree_hash(dir, true)
}

fn tree_hash(dir: &Path, include_task_manifest: bool) -> String {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    collect_tree(dir, dir, &mut entries, include_task_manifest);
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, bytes) in &entries {
        hasher.update(rel.as_bytes());
        hasher.update([0u8]);
        hasher.update(bytes);
    }
    let mut out = String::with_capacity(64);
    for b in hasher.finalize() {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn collect_tree(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, Vec<u8>)>,
    include_task_manifest: bool,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == ".git"
            || name == "node_modules"
            || (!include_task_manifest && name == "task.toml")
        {
            continue;
        }
        if path.is_dir() {
            collect_tree(root, &path, out, include_task_manifest);
        } else if let (Ok(rel), Ok(bytes)) = (path.strip_prefix(root), fs::read(&path)) {
            out.push((rel.to_string_lossy().into_owned(), bytes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tempfile::TempDir;

    const LANES: &str = "schema_version = \"1.0\"\n\
        [lanes.fast]\nobjective_gate=\"test_and_diff\"\nverifier_policy=\"none\"\n\
        retry_budget=0\ndiff_policy=\"intended_paths_only\"\ntimeout_seconds=120\n\
        [lanes.deep]\nobjective_gate=\"test_and_diff\"\nverifier_policy=\"strict\"\n\
        retry_budget=2\ndiff_policy=\"intended_paths_only\"\ntimeout_seconds=300\n";

    /// A deterministic mock runner: it creates the intended file `out.txt` and
    /// prints a zo-shaped result, ignoring its args (the harness appends
    /// `--output-format json -p <prompt>`). So one script drives every cell to a
    /// green, clean, would-be-accepted run with no network or real agent.
    #[cfg(unix)]
    fn write_mock_runner(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let bin = dir.join("mock-runner.sh");
        fs::write(
            &bin,
            "#!/bin/sh\necho patched > out.txt\nprintf '%s' \
             '{\"message\":\"done\",\"iterations\":1,\"usage\":{\"input_tokens\":10,\
             \"output_tokens\":5,\"cache_creation_input_tokens\":1,\
             \"cache_read_input_tokens\":2}}'\n",
        )
        .unwrap();
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).unwrap();
        bin
    }

    fn write_task(fixtures: &Path, dir_name: &str, toml: &str) {
        let dir = fixtures.join(dir_name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("task.toml"), toml).unwrap();
        // A source file so the fixture tree hash is non-empty (fairness needs a
        // pinned starting state) and there is something for git_init to commit.
        fs::write(dir.join("seed.txt"), "seed\n").unwrap();
    }

    #[cfg(unix)]
    fn suite_cfg(root: &Path, bin: PathBuf, lanes_name: &str) -> SuiteConfig {
        SuiteConfig {
            fixtures: root.join("fixtures"),
            lanes: root.join(lanes_name),
            runners: vec![SuiteRunner {
                name: "zo".to_string(),
                kind: "zo".to_string(),
                bin,
                args: vec![],
                model_label: "opus".to_string(),
                effort_label: "high".to_string(),
                permission_mode: "danger-full-access".to_string(),
                version: "mock 1.0".to_string(),
            }],
            out_dir: root.join("out"),
            suite_version: "1.0".to_string(),
            timestamp: "1700000000".to_string(),
            git_commit: "deadbeef".to_string(),
            command_invocation: Some(vec!["deep-eval".to_string(), "suite".to_string()]),
            repeat: 1,
        }
    }

    fn read_json(path: &Path) -> Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    fn assert_fixture_coverage(manifest: &Value, tag: &str, task_id: &str) {
        assert_eq!(manifest["fixture_coverage"]["task_count"], 1);
        assert_eq!(manifest["fixture_coverage"]["tagged_task_count"], 1);
        assert_eq!(manifest["fixture_coverage"]["untagged_task_count"], 0);
        assert_eq!(manifest["fixture_coverage"]["tag_counts"][tag], 1);
        assert_eq!(manifest["fixture_coverage"]["tag_tasks"][tag][0], task_id);
    }

    #[cfg(unix)]
    #[test]
    fn e2e_mock_benchmark_accepts_with_declared_permission_mode_and_scores() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        fs::write(root.path().join("lanes.toml"), LANES).unwrap();
        write_task(
            &root.path().join("fixtures"),
            "smoke",
            "schema_version=\"1.0\"\nid=\"smoke\"\nlane=\"fast\"\nprompt=\"Create out.txt with patched content\"\n\
             test_command=\"test \\\"$(cat out.txt)\\\" = patched\"\ncoverage_tags=[\"general-fast-loop\", \"js-unit-test\"]\nintended_paths=[\"out.txt\"]\n",
        );
        let mut cfg = suite_cfg(root.path(), bin, "lanes.toml");
        cfg.runners[0].args = vec!["--permission-mode".into(), "acceptEdits".into()];
        cfg.runners[0].permission_mode = "acceptEdits".into();
        run_suite(&cfg).unwrap();

        let run_dir = cfg.out_dir.join("zo").join("fast-smoke");
        let fairness = read_json(&run_dir.join("fairness_contract.json"));
        assert_eq!(fairness["status"], "valid");
        assert_eq!(fairness["permission_mode"], "acceptEdits");

        let result = read_json(&run_dir.join("result.json"));
        assert_eq!(result["pass"], true);
        assert_eq!(result["clean_diff"], true);
        let metrics = read_json(&run_dir.join("metrics.json"));
        assert_eq!(metrics["token_total"], 18);

        let decide = read_json(&run_dir.join("decide.json"));
        assert_eq!(decide["final_decision"], "accepted");
        assert_eq!(decide["leaderboard_eligible"], true);

        let ledger = fs::read_to_string(cfg.out_dir.join("ledger.jsonl")).unwrap();
        let row: Value = serde_json::from_str(ledger.lines().next().unwrap()).unwrap();
        assert_eq!(row["runner"], "zo");
        assert_eq!(row["lane"], "fast");
        assert_eq!(row["fixture"], "smoke");
        assert_eq!(row["final_decision"], "accepted");
        assert_eq!(row["leaderboard_eligible"], true);
        assert_eq!(row["token_total"], 18);

        let summary = read_json(&cfg.out_dir.join("summary.json"));
        assert_eq!(summary["fast/zo"]["accepted"], 1);
        assert_eq!(summary["fast/zo"]["eligible"], 1);
        assert_eq!(summary["fast/zo"]["total_token_usage"], 18);

        let leaderboard = read_json(&cfg.out_dir.join("leaderboard.json"));
        assert_eq!(leaderboard["fast"][0]["runner"], "zo");
        assert_eq!(leaderboard["fast"][0]["fixtures_scored"], 1);
        assert_eq!(leaderboard["fast"][0]["accepted"], 1);

        let report = fs::read_to_string(cfg.out_dir.join("report.md")).unwrap();
        assert!(report.contains("## Fixture coverage"));
        assert!(report.contains("`general-fast-loop`: 1 task(s) — smoke"));
        let manifest = read_json(&cfg.out_dir.join("manifest.json"));
        assert_fixture_coverage(&manifest, "general-fast-loop", "smoke");

        assert!(!root.path().join("out.txt").exists());
    }

    // --- R0a: a missing/unparseable lane catalog is fatal, not a silent None ---

    #[cfg(unix)]
    #[test]
    fn r0a_missing_lane_catalog_is_fatal() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        write_task(
            &root.path().join("fixtures"),
            "t1",
            "schema_version=\"1.0\"\nid=\"t1\"\nlane=\"fast\"\nprompt=\"do it\"\n\
             test_command=\"true\"\nintended_paths=[\"out.txt\"]\n",
        );
        // No lanes.toml on disk → load fails → run_suite must error, not proceed.
        let cfg = suite_cfg(root.path(), bin, "absent-lanes.toml");
        let err = run_suite(&cfg).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[cfg(unix)]
    #[test]
    fn invalid_coverage_tags_are_admission_rejected_not_reported() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        fs::write(root.path().join("lanes.toml"), LANES).unwrap();
        write_task(
            &root.path().join("fixtures"),
            "bad-tags",
            "schema_version=\"1.0\"\nid=\"bad-tags\"\nlane=\"fast\"\nprompt=\"do it\"\n\
             test_command=\"true\"\ncoverage_tags=[\"general-fast-loop\", \"not-in-taxonomy\", \"general-fast-loop\"]\nintended_paths=[\"out.txt\"]\n",
        );

        let cfg = suite_cfg(root.path(), bin, "lanes.toml");
        let err = run_suite(&cfg).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let message = err.to_string();
        assert!(message.contains("unknown coverage tag"));
        assert!(message.contains("duplicate coverage tag"));
        assert!(!cfg.out_dir.join("manifest.json").exists());
    }

    // --- happy path: a valid fixture is accepted, artifacts preserved, and the
    // lane-policy timeout flows into the fairness contract (R1, R2) ---

    #[cfg(unix)]
    #[test]
    fn happy_path_accepts_preserves_artifacts_and_inherits_lane_timeout() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        fs::write(root.path().join("lanes.toml"), LANES).unwrap();
        // Task omits timeout_seconds → it must inherit the fast lane's 120 (R1).
        write_task(
            &root.path().join("fixtures"),
            "t1",
            "schema_version=\"1.0\"\nid=\"t1\"\nlane=\"fast\"\nprompt=\"do it\"\n\
             test_command=\"true\"\nintended_paths=[\"out.txt\"]\n",
        );
        let cfg = suite_cfg(root.path(), bin, "lanes.toml");
        run_suite(&cfg).unwrap();

        let run_dir = cfg.out_dir.join("zo").join("fast-t1");
        // R2: the whole required artifact set is on disk, under the doc-aligned
        // names, including the new before/after/diff git snapshots.
        let artifacts = run_dir.join("artifacts");
        assert!(artifacts.join("agent_stdout.json").is_file());
        assert!(artifacts.join("git_status.before.txt").is_file());
        assert!(artifacts.join("git_status.after.txt").is_file());
        assert!(artifacts.join("git_diff.patch").is_file());
        // A fast-lane run owes no deep evidence, so the base set alone preserves it.
        assert!(required_artifacts_present(&artifacts, false));
        // The patch records the agent's new file — the staged diff includes
        // additions, so it is real evidence, not an empty placeholder.
        let diff = fs::read_to_string(artifacts.join("git_diff.patch")).unwrap();
        assert!(
            diff.contains("out.txt"),
            "diff should record the change: {diff}"
        );

        let decide = read_json(&run_dir.join("decide.json"));
        assert_eq!(decide["final_decision"], "accepted");
        assert!(decide["failure_class"].is_null());

        let fairness = read_json(&run_dir.join("fairness_contract.json"));
        assert_eq!(fairness["status"], "valid");
        // R1: inherited from the lane policy, not the old misleading 0.
        assert_eq!(fairness["timeout_seconds"], 120);

        let metrics = read_json(&run_dir.join("metrics.json"));
        assert_eq!(metrics["token_total"], 18);
        assert_eq!(metrics["retry_count"], 0);
        assert_eq!(metrics["dirty_diff"], false);
        assert_eq!(metrics["timeout"], false);

        let ledger = fs::read_to_string(cfg.out_dir.join("ledger.jsonl")).unwrap();
        assert!(ledger.contains("\"token_total\":18"));

        let summary = read_json(&cfg.out_dir.join("summary.json"));
        assert_eq!(summary["fast/zo"]["accepted"], 1);
        assert_eq!(summary["fast/zo"]["attempted"], 1);
        assert_eq!(summary["fast/zo"]["eligible"], 1);
        assert_eq!(summary["fast/zo"]["total_token_usage"], 18);
        assert_eq!(summary["fast/zo"]["total_output_token_usage"], 5);
        assert_eq!(summary["fast/zo"]["unknown_token_usage"], 0);
        assert_eq!(summary["fast/zo"]["dirty_diff_rate"], 0.0);
        assert!(summary["fast/zo"]["mean_wall_seconds"].is_number());
        assert!(
            summary["fast/zo"]["aggregate_total_tokens_per_second"].is_number()
                || summary["fast/zo"]["aggregate_total_tokens_per_second"].is_null()
        );
        assert!(
            summary["fast/zo"]["aggregate_output_tokens_per_second"].is_number()
                || summary["fast/zo"]["aggregate_output_tokens_per_second"].is_null()
        );
        assert_eq!(
            summary["fast/zo"]["deterministic_probe_failure_count"],
            0
        );
        assert!(summary["fast/zo"]["cost_normalized_score"].is_null());

        let leaderboard = read_json(&cfg.out_dir.join("leaderboard.json"));
        assert!(leaderboard["fast"].is_array());
        assert!(leaderboard["_views"]["accuracy_first"]["fast"].is_array());
        assert!(leaderboard["_views"]["latency_first"]["fast"].is_array());
        assert!(leaderboard["_views"]["token_first"]["fast"].is_array());

        let report = fs::read_to_string(cfg.out_dir.join("report.md")).unwrap();
        assert!(report.contains("# Deep eval report"));
        assert!(report.contains("Accepted: 1/1"));
        assert!(report.contains("Token total: 18"));
        assert!(report.contains("Artifacts:"));

        let process_warnings = read_json(&cfg.out_dir.join("process_warnings.json"));
        assert!(process_warnings["leftover_processes"].as_array().is_some());
        assert_eq!(
            process_warnings["scan_scope"],
            "recorded_runner_process_groups"
        );
        assert!(
            process_warnings["recorded_process_groups"]
                .as_u64()
                .unwrap()
                >= 1
        );

        let manifest = read_json(&cfg.out_dir.join("manifest.json"));
        assert_eq!(
            manifest["result_dir"],
            cfg.out_dir.to_string_lossy().as_ref()
        );
        assert_eq!(manifest["harness_git_commit"], "deadbeef");
        assert_eq!(manifest["benchmark_suite_version"], "1.0");
        assert_eq!(manifest["fixture_set_hash"].as_str().unwrap().len(), 64);
        assert_eq!(manifest["runner_versions"]["zo"], "mock 1.0");
        assert_eq!(manifest["command_invocation"][0], "deep-eval");
        assert_eq!(manifest["command_invocation"][1], "suite");
        assert_eq!(
            manifest["ledger_path"],
            cfg.out_dir.join("ledger.jsonl").to_string_lossy().as_ref()
        );
        assert_eq!(manifest["scorer_version"], SCORER_VERSION);
        assert_eq!(
            manifest["report_generator_version"],
            REPORT_GENERATOR_VERSION
        );
    }

    #[cfg(unix)]
    #[test]
    fn leftover_detector_matches_orphaned_descendant_by_recorded_process_group() {
        let groups = vec![crate::runner::SpawnedProcessGroup {
            pgid: 4242,
            root_pid: 1111,
            label: "agent".into(),
            command: "zo -p task".into(),
        }];
        let row = leftover_process_row("2222 1 4242 00:01 orphaned-child", &groups)
            .expect("same process group should be reported even after reparenting");
        assert_eq!(row["pid"], "2222");
        assert_eq!(row["ppid"], "1");
        assert_eq!(row["pgid"], "4242");
        assert_eq!(row["root_pid"], 1111);
        assert_eq!(row["label"], "agent");
        assert!(leftover_process_row("3333 1 7777 00:01 unrelated", &groups).is_none());
    }

    #[test]
    fn repeat_runs_each_cell_n_times_into_rep_dirs_and_writes_repeats_json() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        fs::write(root.path().join("lanes.toml"), LANES).unwrap();
        write_task(
            &root.path().join("fixtures"),
            "t1",
            "schema_version=\"1.0\"\nid=\"t1\"\nlane=\"fast\"\nprompt=\"do it\"\n\
             test_command=\"true\"\ncoverage_tags=[\"general-fast-loop\"]\nintended_paths=[\"out.txt\"]\n",
        );
        let mut cfg = suite_cfg(root.path(), bin, "lanes.toml");
        cfg.repeat = 2;
        run_suite(&cfg).unwrap();

        // Each run is isolated under rep-k; the flat path is not used at repeat>1.
        let cell = cfg.out_dir.join("zo").join("fast-t1");
        assert!(cell.join("rep-1").join("decide.json").is_file());
        assert!(cell.join("rep-2").join("decide.json").is_file());
        assert!(!cell.join("decide.json").exists());

        // The ledger has one row per rep, and repeats.json reports pass@N.
        let ledger = fs::read_to_string(cfg.out_dir.join("ledger.jsonl")).unwrap();
        assert_eq!(ledger.lines().filter(|l| !l.trim().is_empty()).count(), 2);

        let repeats = read_json(&cfg.out_dir.join("repeats.json"));
        let agg = &repeats["fast/zo/t1"];
        assert_eq!(agg["repeats"], 2);
        assert_eq!(agg["accepted"], 2);
        assert_eq!(agg["pass_at_n"], 1.0);
        assert!(agg["median_token_total"].is_number());
        assert!(agg["median_wall_seconds"].is_number());

        let manifest = read_json(&cfg.out_dir.join("manifest.json"));
        assert_fixture_coverage(&manifest, "general-fast-loop", "t1");
    }

    #[test]
    fn single_shot_keeps_flat_layout_and_writes_no_repeats_json() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        fs::write(root.path().join("lanes.toml"), LANES).unwrap();
        write_task(
            &root.path().join("fixtures"),
            "t1",
            "schema_version=\"1.0\"\nid=\"t1\"\nlane=\"fast\"\nprompt=\"do it\"\n\
             test_command=\"true\"\nintended_paths=[\"out.txt\"]\n",
        );
        let cfg = suite_cfg(root.path(), bin, "lanes.toml"); // repeat: 1
        run_suite(&cfg).unwrap();

        // Back-compat: flat path, no rep-k nesting, no repeats.json.
        assert!(cfg
            .out_dir
            .join("zo")
            .join("fast-t1")
            .join("decide.json")
            .is_file());
        assert!(!cfg.out_dir.join("repeats.json").exists());
    }

    #[test]
    fn fixture_set_hash_includes_task_manifests() {
        let root = TempDir::new().unwrap();
        let fixtures = root.path().join("fixtures");
        write_task(
            &fixtures,
            "t1",
            "schema_version=\"1.0\"\nid=\"t1\"\nlane=\"fast\"\nprompt=\"a\"\n\
             test_command=\"true\"\n",
        );

        let before = fixture_set_hash(&fixtures);
        fs::write(
            fixtures.join("t1").join("task.toml"),
            "schema_version=\"1.0\"\nid=\"t1\"\nlane=\"fast\"\nprompt=\"b\"\n\
             test_command=\"true\"\n",
        )
        .unwrap();

        let after = fixture_set_hash(&fixtures);
        assert_ne!(before, after);
    }

    #[cfg(unix)]
    #[test]
    fn lane_diff_policy_any_accepts_unlisted_source_edit() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        let lanes = LANES.replace(
            "diff_policy=\"intended_paths_only\"\ntimeout_seconds=120",
            "diff_policy=\"any\"\ntimeout_seconds=120",
        );
        fs::write(root.path().join("lanes.toml"), lanes).unwrap();
        // No intended paths: with the old hard-coded intended_paths_only policy,
        // the mock runner's out.txt edit was a dirty diff. The lane's `any`
        // policy now permits that source edit while still preserving artifacts.
        write_task(
            &root.path().join("fixtures"),
            "reviewish",
            "schema_version=\"1.0\"\nid=\"reviewish\"\nlane=\"fast\"\nprompt=\"do it\"\n\
             test_command=\"true\"\n",
        );
        let cfg = suite_cfg(root.path(), bin, "lanes.toml");
        run_suite(&cfg).unwrap();

        let run_dir = cfg.out_dir.join("zo").join("fast-reviewish");
        let result = read_json(&run_dir.join("result.json"));
        assert_eq!(result["clean_diff"], true);
        assert_eq!(result["pass"], true);
        let decide = read_json(&run_dir.join("decide.json"));
        assert_eq!(decide["final_decision"], "accepted");
    }

    // --- R0b: an undeclared-lane fixture is rejected at admission, never run ---

    #[cfg(unix)]
    #[test]
    fn r0b_unknown_lane_fixture_is_admission_rejected_not_run() {
        let root = TempDir::new().unwrap();
        let bin = write_mock_runner(root.path());
        fs::write(root.path().join("lanes.toml"), LANES).unwrap();
        // 'nope' is not a known lane → validate_task flags it → must not run.
        write_task(
            &root.path().join("fixtures"),
            "bad",
            "schema_version=\"1.0\"\nid=\"bad\"\nlane=\"nope\"\nprompt=\"p\"\n\
             test_command=\"true\"\n",
        );
        let cfg = suite_cfg(root.path(), bin, "lanes.toml");
        let err = run_suite(&cfg).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("unknown lane"));

        let run_dir = cfg.out_dir.join("zo").join("nope-bad");
        assert!(!run_dir.exists(), "invalid fixture should not create run artifacts");
        assert!(!cfg.out_dir.join("ledger.jsonl").exists());
        assert!(!cfg.out_dir.join("summary.json").exists());
    }

    // --- R2 (targeted): the artifact-set check and the decision gate it drives ---

    #[test]
    fn r2_required_artifacts_present_needs_every_file() {
        use crate::runner::{REQUIRED_ARTIFACTS, REQUIRED_DEEP_ARTIFACTS};
        let dir = TempDir::new().unwrap();
        let d = dir.path();
        // The base set is "present" only once the WHOLE list is on disk.
        for (i, name) in REQUIRED_ARTIFACTS.iter().enumerate() {
            assert!(!required_artifacts_present(d, false), "premature at {name}");
            fs::write(d.join(name), "").unwrap();
            let complete = i + 1 == REQUIRED_ARTIFACTS.len();
            assert_eq!(required_artifacts_present(d, false), complete);
        }
        // A deep run additionally owes the deep evidence set.
        assert!(!required_artifacts_present(d, true));
        for name in REQUIRED_DEEP_ARTIFACTS {
            fs::write(d.join(name), "").unwrap();
        }
        assert!(required_artifacts_present(d, true));
    }

    fn green_fast_result(artifact_dir: Option<String>) -> RunResult {
        RunResult {
            runner: "zo".into(),
            model: "claude-opus-4-8".into(),
            effort: "high".into(),
            lane: "fast".into(),
            exit_code: 0,
            wall_seconds: 1,
            startup_seconds: None,
            test: crate::diff_hygiene::TestStatus::Pass,
            intended_changed: 1,
            permission_denials: 0,
            pollution: vec![],
            unexpected: vec![],
            clean_diff: true,
            pass: true,
            tokens: None,
            iterations: Some(1),
            fail_reasons: vec![],
            warnings: vec![],
            artifact_dir,
            deep: None,
        }
    }

    #[test]
    fn r2_derive_decision_invalid_when_artifact_set_incomplete() {
        use crate::runner::REQUIRED_ARTIFACTS;
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_string_lossy().into_owned();
        // All but the last required file present → incomplete → not replayable.
        let (last, rest) = REQUIRED_ARTIFACTS.split_last().unwrap();
        for name in rest {
            fs::write(dir.path().join(name), "").unwrap();
        }
        let verdict = derive_decision(&green_fast_result(Some(path.clone())), "valid");
        assert_eq!(verdict.decision, FinalDecision::Invalid);
        assert_eq!(
            verdict.failure,
            Some(FailureClass::ArtifactPreservationFailed)
        );

        // Complete the set → the same green run is now accepted.
        fs::write(dir.path().join(last), "").unwrap();
        let verdict = derive_decision(&green_fast_result(Some(path)), "valid");
        assert_eq!(verdict.decision, FinalDecision::Accepted);
        assert!(verdict.failure.is_none());
    }

    /// A green deep run whose deciding objective passed and verifier accepted via
    /// strict JSON — the same shape as an accepted single-shot, but `deep` is set
    /// so the gate also demands the deep evidence set.
    fn green_deep_result(artifact_dir: Option<String>) -> RunResult {
        let mut r = green_fast_result(artifact_dir);
        r.lane = "deep".into();
        r.deep = Some(crate::deep::DeepVerdict {
            attempts: 1,
            max_attempts: 2,
            plan_valid: true,
            verifier_accepted: true,
            outcome: "accept".into(),
            diagnostics: crate::deep::DeepDiagnostics {
                plan_missing: vec![],
                verifier_parse: "json".into(),
                verifier_issues: 0,
                objective_passed: true,
                phase_timed_out: false,
                plan_recovered: false,
                verifier_recovered_by_objective: false,
                deterministic_probe_issues: 0,
                failure: "none".into(),
            },
            phase_timings: crate::deep::DeepPhaseTimings::default(),
        });
        r
    }

    #[test]
    fn r2_deep_run_requires_deep_evidence() {
        use crate::runner::{REQUIRED_ARTIFACTS, REQUIRED_DEEP_ARTIFACTS};
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_string_lossy().into_owned();
        // Full base set present, but the deep evidence (plan + verifier raw) is
        // missing → a green deep run is invalid, not accepted.
        for name in REQUIRED_ARTIFACTS {
            fs::write(dir.path().join(name), "").unwrap();
        }
        let verdict = derive_decision(&green_deep_result(Some(path.clone())), "valid");
        assert_eq!(verdict.decision, FinalDecision::Invalid);
        assert_eq!(
            verdict.failure,
            Some(FailureClass::ArtifactPreservationFailed)
        );

        // Add the deep evidence → the same green deep run is now accepted.
        for name in REQUIRED_DEEP_ARTIFACTS {
            fs::write(dir.path().join(name), "").unwrap();
        }
        let verdict = derive_decision(&green_deep_result(Some(path)), "valid");
        assert_eq!(verdict.decision, FinalDecision::Accepted);
        assert!(verdict.failure.is_none());
    }
}
