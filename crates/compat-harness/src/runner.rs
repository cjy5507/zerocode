//! Native single-runner executor — the Rust core of the benchmark harness.
//!
//! Replaces the shell `run_one`/`invoke_agent` online glue (copy → git → spawn →
//! test → diff → usage) while reusing the already-unit-tested pure verdict logic
//! in [`crate::diff_hygiene`]. The shell mirror can retire once the suite drives
//! this.
//!
//! Two deliberate departures from the shell, both wins:
//! - **Agent stdout/stderr stay in memory**, never written into the work tree, so
//!   no scratch file can ever count as diff pollution (the shell wrote
//!   `.agent-out.json` into `$work` and then had to filter it back out).
//! - **An accepted run's artifacts are preserved** for replayability — the shell
//!   discarded passing runs (`rm -rf "$work"`), which broke the goal document's
//!   accepted-run replayability requirement.

use std::fs;
use std::io;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::ChildStderr;
use std::process::ChildStdout;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[derive(Debug, Clone)]
pub(crate) struct SpawnedProcessGroup {
    pub(crate) pgid: u32,
    pub(crate) root_pid: u32,
    pub(crate) label: String,
    pub(crate) command: String,
}

static SPAWNED_PROCESS_GROUPS: OnceLock<Mutex<Vec<SpawnedProcessGroup>>> = OnceLock::new();

fn record_spawned_process_group(pgid: u32, root_pid: u32, label: &str, command: String) {
    let registry = SPAWNED_PROCESS_GROUPS.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut groups) = registry.lock() {
        groups.push(SpawnedProcessGroup {
            pgid,
            root_pid,
            label: label.to_string(),
            command,
        });
    }
}

pub(crate) fn spawned_process_groups_snapshot() -> Vec<SpawnedProcessGroup> {
    SPAWNED_PROCESS_GROUPS
        .get()
        .and_then(|groups| groups.lock().ok().map(|groups| groups.clone()))
        .unwrap_or_default()
}

use serde::Serialize;
use serde_json::json;
use serde_json::Value;
use tempfile::TempDir;

use crate::diff_hygiene::fail_reasons;
use crate::diff_hygiene::permission_denial_fatal;
use crate::diff_hygiene::run_passed;
use crate::diff_hygiene::score;
use crate::diff_hygiene::warnings;
use crate::diff_hygiene::DiffHygiene;
use crate::diff_hygiene::TestStatus;

/// Everything one runner needs to execute one fixture once. `bin` is expected to
/// be absolute (the caller canonicalizes); `model`/`effort` are already the
/// normalized labels the result should record.
#[derive(Debug, Clone)]
pub struct RunSpec {
    /// Stable display/ledger identity for this benchmark cell.
    pub runner: String,
    /// Executable protocol to use when spawning the runner (`zo`, `claude`,
    /// or `codex`). Kept separate from `runner` so a suite can compare aliases
    /// like `zo_claude` and `zo_gpt` without losing readable results.
    pub runner_kind: String,
    pub bin: PathBuf,
    pub args: Vec<String>,
    pub fixture: PathBuf,
    pub prompt: String,
    pub test_command: Option<String>,
    pub intended: Vec<String>,
    pub lane: String,
    pub model: String,
    pub effort: String,
    /// Lane objective policy (`test_and_diff` or `verifier_only`).
    pub objective_gate: String,
    /// Lane diff policy (`intended_paths_only` or `any`).
    pub diff_policy: String,
    /// Whole-run timeout budget in seconds. `0` means no timeout.
    pub timeout_seconds: u64,
    /// When set, the run's artifacts are written here (created if absent),
    /// regardless of pass/fail — so an accepted run stays replayable.
    pub artifacts_dir: Option<PathBuf>,
    /// On failure, also snapshot the whole work tree under `<artifacts_dir>/worktree`.
    pub keep_failed: bool,
    /// `Some` selects the deep lane (forced plan→execute→verify→retry loop);
    /// `None` runs a single-shot fast-lane invocation.
    pub deep: Option<crate::deep::DeepConfig>,
}

/// Billed token breakdown, mirroring the shell `tokens_object` and the runtime's
/// `TokenUsage::total_tokens`. `cache_*` are `None` when the runner omitted them,
/// which forces `complete=false` (the bare `input` headline can hide cached
/// context). `total` is the billed sum across the classes present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Tokens {
    pub input: u64,
    pub output: u64,
    pub cache_creation: Option<u64>,
    pub cache_read: Option<u64>,
    pub total: u64,
    pub complete: bool,
}

/// One runner's scored result. Field order and names mirror the shell harness's
/// per-run JSON object exactly, so the two are directly comparable (the fast lane
/// leaves `deep` null).
#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub runner: String,
    pub model: String,
    pub effort: String,
    pub lane: String,
    pub exit_code: i32,
    pub wall_seconds: u64,
    /// Spawn → first stdout byte of the agent process, when observed. This is the
    /// runner's fixed CLI start-up + plugin-scan cost, separated from task work so
    /// a fast-lane comparison isn't distorted by it (codex pays a large fixed cost
    /// here; zo/claude pay almost none). Additive: `wall_seconds` is unchanged.
    /// `None` on the deep lane, whose synthesized result spans many spawns.
    pub startup_seconds: Option<u64>,
    pub test: TestStatus,
    pub intended_changed: usize,
    pub permission_denials: usize,
    pub pollution: Vec<String>,
    pub unexpected: Vec<String>,
    pub clean_diff: bool,
    pub pass: bool,
    pub tokens: Option<Tokens>,
    pub iterations: Option<u64>,
    pub fail_reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub artifact_dir: Option<String>,
    pub deep: Option<crate::deep::DeepVerdict>,
}

/// Compact, directly-observed metrics for reporting. Cost is intentionally
/// optional: without real billing/pricing data the harness must not invent it.
#[derive(Debug, Clone, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct RunMetrics {
    pub wall_seconds: u64,
    /// Fixed runner start-up cost (see [`RunResult::startup_seconds`]); subtract
    /// from `wall_seconds` for the task-only wall.
    pub startup_seconds: Option<u64>,
    pub token_total: Option<u64>,
    pub token_input: Option<u64>,
    pub token_output: Option<u64>,
    pub token_cache_creation: Option<u64>,
    pub token_cache_read: Option<u64>,
    pub total_tokens_per_second: Option<f64>,
    pub output_tokens_per_second: Option<f64>,
    pub phase_timings: Option<crate::deep::DeepPhaseTimings>,
    pub deterministic_probe_failure_count: usize,
    pub retry_count: u32,
    pub dirty_diff: bool,
    pub deep_verifier_failed: bool,
    pub timeout: bool,
    pub provider_error: bool,
    pub cost_usd: Option<f64>,
    pub cost_normalized_score: Option<f64>,
}

impl RunResult {
    #[must_use]
    pub fn metrics(&self) -> RunMetrics {
        let timeout = self
            .fail_reasons
            .iter()
            .any(|r| r == "agent_timeout" || r == "test_timeout")
            || self
                .deep
                .as_ref()
                .is_some_and(|d| d.diagnostics.verifier_parse == "timeout");
        let deep_verifier_failed = self.deep.as_ref().is_some_and(|d| {
            !d.verifier_accepted
                || d.diagnostics.verifier_parse != "json"
                || d.diagnostics.failure.starts_with("verifier_")
        });
        let token_total = self.tokens.as_ref().map(|t| t.total);
        let token_input = self.tokens.as_ref().map(|t| t.input);
        let token_output = self.tokens.as_ref().map(|t| t.output);
        let token_cache_creation = self.tokens.as_ref().and_then(|t| t.cache_creation);
        let token_cache_read = self.tokens.as_ref().and_then(|t| t.cache_read);
        let total_tokens_per_second = tokens_per_second(token_total, self.wall_seconds);
        let output_tokens_per_second = tokens_per_second(token_output, self.wall_seconds);
        RunMetrics {
            wall_seconds: self.wall_seconds,
            startup_seconds: self.startup_seconds,
            token_total,
            token_input,
            token_output,
            token_cache_creation,
            token_cache_read,
            total_tokens_per_second,
            output_tokens_per_second,
            phase_timings: self.deep.as_ref().map(|d| d.phase_timings.clone()),
            deterministic_probe_failure_count: self
                .deep
                .as_ref()
                .map_or(0, |d| d.diagnostics.deterministic_probe_issues),
            retry_count: self
                .deep
                .as_ref()
                .map_or(0, |d| d.attempts.saturating_sub(1)),
            dirty_diff: !self.clean_diff,
            deep_verifier_failed,
            timeout,
            provider_error: self.exit_code != 0 && !timeout,
            cost_usd: None,
            cost_normalized_score: None,
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn tokens_per_second(tokens: Option<u64>, wall_seconds: u64) -> Option<f64> {
    tokens
        .filter(|_| wall_seconds > 0)
        .map(|tokens| tokens as f64 / wall_seconds as f64)
}

/// Raw output of one agent invocation, kept in memory. The deep loop reuses this
/// shape (with an empty stderr) for its synthesized whole-loop result.
pub(crate) struct AgentOutput {
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) exit_code: i32,
    pub(crate) timed_out: bool,
    /// Spawn → first stdout byte (the runner's start-up cost), when observed.
    pub(crate) startup_seconds: Option<u64>,
}

pub(crate) struct TestRun {
    pub(crate) status: TestStatus,
    pub(crate) log: Option<String>,
    pub(crate) timed_out: bool,
}

impl TestRun {
    fn skipped() -> Self {
        Self {
            status: TestStatus::Skipped,
            log: None,
            timed_out: false,
        }
    }
}

pub(crate) struct RunBudget {
    started: Instant,
    timeout: Option<Duration>,
}

impl RunBudget {
    fn new(timeout_seconds: u64) -> Self {
        Self {
            started: Instant::now(),
            timeout: (timeout_seconds > 0).then(|| Duration::from_secs(timeout_seconds)),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_duration(timeout: Duration) -> Self {
        Self {
            started: Instant::now(),
            timeout: Some(timeout),
        }
    }

    pub(crate) fn remaining(&self) -> Option<Duration> {
        self.timeout
            .map(|timeout| timeout.saturating_sub(self.started.elapsed()))
    }

    pub(crate) fn reserving_capped(&self, reserve: Duration, cap: Duration) -> Self {
        Self {
            started: Instant::now(),
            timeout: self
                .remaining()
                .map(|remaining| remaining.saturating_sub(reserve).min(cap)),
        }
    }
}

struct CapturedCommand {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    exit_code: i32,
    success: bool,
    timed_out: bool,
    /// Spawn → first stdout byte, when the child produced any. `None` if it never
    /// wrote to stdout (or was never spawned).
    startup: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectiveGatePolicy {
    TestAndDiff,
    VerifierOnly,
}

impl ObjectiveGatePolicy {
    fn from_token(token: &str) -> Self {
        match token.trim() {
            "verifier_only" => Self::VerifierOnly,
            _ => Self::TestAndDiff,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffPolicy {
    IntendedPathsOnly,
    Any,
}

impl DiffPolicy {
    fn from_token(token: &str) -> Self {
        match token.trim() {
            "any" => Self::Any,
            _ => Self::IntendedPathsOnly,
        }
    }
}

/// Execute one runner on one fixture and score it. The work tree is a throwaway
/// temp dir (RAII-cleaned on drop); copy the fixture in, make it a git repo so
/// `git status` shows only the agent's edits, run the agent, verify, then score
/// the diff with the shared [`crate::diff_hygiene`] rules.
///
/// # Errors
/// Returns an error only for I/O the run cannot proceed without (temp dir, fixture
/// copy, artifact write). A non-zero agent or test exit is data, not an error.
pub fn run_one(spec: &RunSpec) -> io::Result<RunResult> {
    let work_tmp = TempDir::new()?;
    let work = work_tmp.path();
    copy_dir(&spec.fixture, work)?;
    git_init(work)?;
    // The clean baseline `git_init` just committed — the run's before-state
    // snapshot (empty when we initialized the repo ourselves), preserved so the
    // diff is replayable against a known starting point.
    let git_status_before = git_porcelain(work)?;

    let start = Instant::now();
    let budget = RunBudget::new(spec.timeout_seconds);
    let (agent, deep_verdict, deep_artifacts) = match spec.deep.as_ref() {
        Some(cfg) => {
            let r = crate::deep::run_deep_loop(spec, work, cfg, &budget)?;
            (
                AgentOutput {
                    stdout: r.stdout,
                    stderr: String::new(),
                    exit_code: r.exit_code,
                    timed_out: r.timed_out,
                    // The deep result spans many spawns; per-spawn start-up is not
                    // meaningfully a single number, so the deep lane omits it.
                    startup_seconds: None,
                },
                Some(r.verdict),
                Some(r.artifacts),
            )
        }
        None => (spawn_agent(spec, work, &budget)?, None, None),
    };
    let wall_seconds = start.elapsed().as_secs();

    let test_run = match spec.test_command.as_deref() {
        None => TestRun::skipped(),
        Some(cmd) => run_test(work, cmd, &budget),
    };
    let test = test_run.status;

    let porcelain = filter_scratch(&git_porcelain(work)?);
    let intended_refs: Vec<&str> = spec.intended.iter().map(String::as_str).collect();
    let (hygiene, intended_provided) =
        score_for_policy(&porcelain, &intended_refs, &spec.diff_policy);

    let tokens = parse_tokens(&agent.stdout);
    let iterations = parse_iterations(&agent.stdout);
    let denials = parse_denials(&agent.stdout);

    let base_pass = run_passed_for_policy(
        agent.exit_code,
        test,
        &hygiene,
        denials,
        intended_provided,
        &spec.objective_gate,
    );
    // The deep lane adds one gate on top of the objective rule: the strict
    // verifier must have ACCEPTED (outcome "accept"). A loop that ran out of
    // attempts is not a pass even with a clean objective diff. Only ever blocks.
    let deep_accepted = deep_verdict.as_ref().is_none_or(|v| v.outcome == "accept");
    let pass = base_pass && deep_accepted;

    let reasons = run_fail_reasons(
        &agent,
        &test_run,
        &hygiene,
        denials,
        intended_provided,
        &spec.objective_gate,
        deep_verdict.as_ref(),
    );
    let warns = warnings_for_policy(
        denials,
        test,
        &hygiene,
        intended_provided,
        &spec.objective_gate,
    );

    let artifact_dir = preserve_artifacts(
        spec,
        work,
        &agent,
        test_run.log.as_deref(),
        pass,
        &git_status_before,
        deep_artifacts.as_ref(),
    )?;

    Ok(RunResult {
        runner: spec.runner.clone(),
        model: spec.model.clone(),
        effort: spec.effort.clone(),
        lane: spec.lane.clone(),
        exit_code: agent.exit_code,
        wall_seconds,
        startup_seconds: agent.startup_seconds,
        test,
        intended_changed: hygiene.intended_count,
        permission_denials: denials,
        pollution: hygiene.pollution.clone(),
        unexpected: hygiene.unexpected.clone(),
        clean_diff: hygiene.clean,
        pass,
        tokens,
        iterations,
        fail_reasons: reasons,
        warnings: warns.iter().map(|s| (*s).to_string()).collect(),
        artifact_dir,
        deep: deep_verdict,
    })
}

fn run_fail_reasons(
    agent: &AgentOutput,
    test: &TestRun,
    hygiene: &DiffHygiene,
    permission_denials: usize,
    intended_provided: bool,
    objective_gate: &str,
    deep_verdict: Option<&crate::deep::DeepVerdict>,
) -> Vec<String> {
    let mut reasons: Vec<String> = fail_reasons_for_policy(
        agent.exit_code,
        test.status,
        hygiene,
        permission_denials,
        intended_provided,
        objective_gate,
    )
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    if agent.timed_out {
        reasons.push("agent_timeout".to_string());
    }
    if test.timed_out {
        reasons.push("test_timeout".to_string());
    }
    if deep_verdict.is_some_and(|v| v.outcome != "accept") {
        reasons.push("deep_unverified".to_string());
    }
    reasons
}

pub(crate) fn score_for_policy(
    porcelain: &str,
    intended: &[&str],
    diff_policy: &str,
) -> (DiffHygiene, bool) {
    let mut hygiene = score(porcelain, intended);
    let mut intended_provided = !intended.is_empty();
    if DiffPolicy::from_token(diff_policy) == DiffPolicy::Any {
        hygiene.unexpected.clear();
        hygiene.clean = hygiene.pollution.is_empty();
        intended_provided = false;
    }
    (hygiene, intended_provided)
}

pub(crate) fn run_passed_for_policy(
    exit_code: i32,
    test: TestStatus,
    hygiene: &DiffHygiene,
    permission_denials: usize,
    intended_provided: bool,
    objective_gate: &str,
) -> bool {
    match ObjectiveGatePolicy::from_token(objective_gate) {
        ObjectiveGatePolicy::TestAndDiff => run_passed(
            exit_code,
            test,
            hygiene,
            permission_denials,
            intended_provided,
        ),
        ObjectiveGatePolicy::VerifierOnly => {
            exit_code == 0
                && hygiene.clean
                && !permission_denial_fatal(
                    permission_denials,
                    TestStatus::Skipped,
                    hygiene,
                    intended_provided,
                )
        }
    }
}

pub(crate) fn objective_evidence_passed_for_policy(
    test: TestStatus,
    hygiene: &DiffHygiene,
    permission_denials: usize,
    intended_provided: bool,
    objective_gate: &str,
) -> bool {
    // Deep lanes run an explicit test/diff objective gate after each agent
    // phase. Once that evidence is green, a process timeout from the agent turn
    // is a runner-control signal, not proof the produced patch is bad. The
    // final semantic verifier still has to accept before the run can pass.
    run_passed_for_policy(
        0,
        test,
        hygiene,
        permission_denials,
        intended_provided,
        objective_gate,
    )
}

fn fail_reasons_for_policy(
    exit_code: i32,
    test: TestStatus,
    hygiene: &DiffHygiene,
    permission_denials: usize,
    intended_provided: bool,
    objective_gate: &str,
) -> Vec<&'static str> {
    let effective_test = match ObjectiveGatePolicy::from_token(objective_gate) {
        ObjectiveGatePolicy::TestAndDiff => test,
        ObjectiveGatePolicy::VerifierOnly => TestStatus::Skipped,
    };
    fail_reasons(
        exit_code,
        effective_test,
        hygiene,
        permission_denials,
        intended_provided,
    )
}

fn warnings_for_policy(
    permission_denials: usize,
    test: TestStatus,
    hygiene: &DiffHygiene,
    intended_provided: bool,
    objective_gate: &str,
) -> Vec<&'static str> {
    let effective_test = match ObjectiveGatePolicy::from_token(objective_gate) {
        ObjectiveGatePolicy::TestAndDiff => test,
        ObjectiveGatePolicy::VerifierOnly => TestStatus::Skipped,
    };
    warnings(
        permission_denials,
        effective_test,
        hygiene,
        intended_provided,
    )
}

/// Spawn the agent in `work` with `spec.prompt` (single-shot / fast lane).
fn spawn_agent(spec: &RunSpec, work: &Path, budget: &RunBudget) -> io::Result<AgentOutput> {
    spawn_with_prompt(spec, work, &spec.prompt, budget)
}

fn apply_zo_benchmark_env(cmd: &mut Command, spec: &RunSpec) {
    let Some(artifacts_dir) = spec.artifacts_dir.as_ref() else {
        return;
    };
    let root = artifacts_dir.join("zo-runtime");
    set_env_default(cmd, "ZO_ARTIFACT_STORE", root.join("artifacts"));
    set_env_default(cmd, "ZO_SESSION_ROOT", root.join("sessions"));
    set_env_default(cmd, "ZO_WORKFLOW_STORE", root.join("workflows"));
    set_env_default(cmd, "ZO_TODO_STORE", root.join("todos.json"));
}

fn set_env_default(cmd: &mut Command, key: &str, value: PathBuf) {
    if std::env::var_os(key).is_none() {
        cmd.env(key, absolute_path(value));
    }
}

fn absolute_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

/// Spawn the agent in `work` with an explicit prompt, enforcing each runner's
/// argument order (Zo's `-p` swallows the rest of argv, so its flags must
/// precede it; codex emits a JSONL event stream we fold into the zo/claude
/// shape). The deep loop reuses this for its plan/exec/verify turns.
pub(crate) fn spawn_with_prompt(
    spec: &RunSpec,
    work: &Path,
    prompt: &str,
    budget: &RunBudget,
) -> io::Result<AgentOutput> {
    let mut cmd = Command::new(&spec.bin);
    cmd.current_dir(work);
    match spec.runner_kind.as_str() {
        "zo" => {
            apply_zo_benchmark_env(&mut cmd, spec);
            if spec.effort != "unknown" {
                cmd.env("ZO_EFFORT", &spec.effort);
            }
            if let Some(root) = spec
                .artifacts_dir
                .as_ref()
                .map(|dir| dir.join("zo-trace"))
            {
                cmd.env("ZO_TRACE_ROOT", absolute_path(root));
            }
            cmd.args(&spec.args)
                .arg("--output-format")
                .arg("json")
                .arg("-p")
                .arg(prompt);
        }
        "claude" => {
            cmd.args(&spec.args)
                .arg("-p")
                .arg(prompt)
                .arg("--output-format")
                .arg("json");
        }
        "codex" => {
            cmd.arg("exec")
                .arg("--json")
                .arg("--skip-git-repo-check")
                .args(&spec.args)
                .arg(prompt);
            let out = run_command(cmd, budget.remaining(), "agent")?;
            return Ok(AgentOutput {
                stdout: convert_codex_jsonl(&String::from_utf8_lossy(&out.stdout)),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                exit_code: out.exit_code,
                timed_out: out.timed_out,
                startup_seconds: out.startup.map(|d| d.as_secs()),
            });
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown runner: {other}"),
            ));
        }
    }
    let out = run_command(cmd, budget.remaining(), "agent")?;
    Ok(AgentOutput {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        exit_code: out.exit_code,
        timed_out: out.timed_out,
        startup_seconds: out.startup.map(|d| d.as_secs()),
    })
}

/// Fold codex's JSONL event stream into the single zo/claude-shaped object the
/// scorer reads: the last `agent_message` text as the result, and the last
/// `turn.completed.usage` remapped to the Anthropic four-counter. codex
/// `input_tokens` is TOTAL input per the OpenAI convention, so uncached =
/// `input - cached`; output gains reasoning tokens; `cache_read` = cached.
/// Mirrors the shell `jq -s` exactly.
fn convert_codex_jsonl(jsonl: &str) -> String {
    let mut message = String::new();
    let mut usage = json!({});
    for line in jsonl.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match v["type"].as_str() {
            Some("item.completed") if v["item"]["type"] == "agent_message" => {
                if let Some(t) = v["item"]["text"].as_str() {
                    message = t.to_string();
                }
            }
            Some("turn.completed") => {
                let u = &v["usage"];
                let input = u["input_tokens"].as_u64().unwrap_or(0);
                let cached = u["cached_input_tokens"].as_u64().unwrap_or(0);
                let output = u["output_tokens"].as_u64().unwrap_or(0);
                let reasoning = u["reasoning_output_tokens"].as_u64().unwrap_or(0);
                usage = json!({
                    "input_tokens": input.saturating_sub(cached),
                    "output_tokens": output + reasoning,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": cached,
                });
            }
            _ => {}
        }
    }
    json!({ "message": message, "usage": usage }).to_string()
}

/// Run the task's verification command in the work tree, capturing combined
/// output. A non-zero exit is `Fail`.
pub(crate) fn run_test(work: &Path, cmd: &str, budget: &RunBudget) -> TestRun {
    let mut command = Command::new("sh");
    command.arg("-c").arg(cmd).current_dir(work);
    match run_command(command, budget.remaining(), "test") {
        Ok(o) => {
            let mut log = o.stdout;
            log.extend_from_slice(&o.stderr);
            let status = if o.success {
                TestStatus::Pass
            } else {
                TestStatus::Fail
            };
            TestRun {
                status,
                log: Some(String::from_utf8_lossy(&log).into_owned()),
                timed_out: o.timed_out,
            }
        }
        Err(e) => TestRun {
            status: TestStatus::Fail,
            log: Some(format!("test spawn failed: {e}")),
            timed_out: false,
        },
    }
}

fn run_command(
    mut cmd: Command,
    timeout: Option<Duration>,
    label: &str,
) -> io::Result<CapturedCommand> {
    // A budget that is already fully spent (saturated to zero) can't run anything.
    if timeout == Some(Duration::ZERO) {
        return Ok(timeout_without_spawn(label, Duration::ZERO));
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    cmd.process_group(0);
    let command_debug = format!("{cmd:?}");
    let mut child = cmd.spawn()?;
    #[cfg(unix)]
    record_spawned_process_group(child.id(), child.id(), label, command_debug);
    let started = Instant::now();

    // Drain both pipes on their own threads. A child that out-writes the OS pipe
    // buffer (~64KB) before exiting would otherwise block on write and stall the
    // poll loop until the timeout fires; continuous draining removes that hazard.
    // The stdout reader also stamps the first byte, giving spawn → first-output —
    // the runner's fixed start-up cost, which we record separately from task work.
    let first_byte = Arc::new(Mutex::new(None));
    let stdout_reader = spawn_stdout_reader(child.stdout.take(), started, Arc::clone(&first_byte));
    let stderr_reader = spawn_stderr_reader(child.stderr.take());

    let status;
    let timed_out;
    loop {
        if let Some(exited) = child.try_wait()? {
            status = exited;
            timed_out = false;
            break;
        }
        match timeout {
            Some(limit) if started.elapsed() >= limit => {
                terminate_child_group(&mut child);
                status = child.wait()?;
                timed_out = true;
                break;
            }
            _ => std::thread::sleep(Duration::from_millis(20)),
        }
    }

    // Readers exit on pipe EOF, which the child's exit (or kill above) guarantees.
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    let startup = first_byte.lock().ok().and_then(|g| *g);

    let mut captured = captured_from_output(stdout, stderr, status, timed_out);
    captured.startup = startup;
    if timed_out {
        captured = captured.with_timeout_marker(label, timeout.unwrap_or_default());
    }
    Ok(captured)
}

/// Drain a child's stdout to completion, stamping the arrival of the first byte
/// into `first_byte` as `spawn → first-output` (the runner's start-up cost).
fn spawn_stdout_reader(
    stdout: Option<ChildStdout>,
    started: Instant,
    first_byte: Arc<Mutex<Option<Duration>>>,
) -> JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        let Some(mut reader) = stdout else {
            return buf;
        };
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    if buf.is_empty() {
                        if let Ok(mut cell) = first_byte.lock() {
                            *cell = Some(started.elapsed());
                        }
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        buf
    })
}

/// Drain a child's stderr to completion (no first-byte stamp — start-up is
/// measured on stdout, which every runner writes its result to).
fn spawn_stderr_reader(stderr: Option<ChildStderr>) -> JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut reader) = stderr {
            let _ = reader.read_to_end(&mut buf);
        }
        buf
    })
}

fn terminate_child_group(child: &mut Child) {
    #[cfg(unix)]
    {
        let pgid = format!("-{}", child.id());
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg("--")
            .arg(&pgid)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        std::thread::sleep(Duration::from_millis(50));
        let _ = Command::new("kill")
            .arg("-KILL")
            .arg("--")
            .arg(&pgid)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .arg("/PID")
            .arg(child.id().to_string())
            .arg("/T")
            .arg("/F")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = child.kill();
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        let _ = child.kill();
    }
}

fn captured_from_output(
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: std::process::ExitStatus,
    timed_out: bool,
) -> CapturedCommand {
    CapturedCommand {
        stdout,
        stderr,
        exit_code: if timed_out {
            -1
        } else {
            status.code().unwrap_or(-1)
        },
        success: !timed_out && status.success(),
        timed_out,
        startup: None,
    }
}

fn timeout_without_spawn(label: &str, timeout: Duration) -> CapturedCommand {
    CapturedCommand {
        stdout: Vec::new(),
        stderr: timeout_marker(label, timeout).into_bytes(),
        exit_code: -1,
        success: false,
        timed_out: true,
        startup: None,
    }
}

impl CapturedCommand {
    fn with_timeout_marker(mut self, label: &str, timeout: Duration) -> Self {
        self.stderr
            .extend_from_slice(timeout_marker(label, timeout).as_bytes());
        self.exit_code = -1;
        self.success = false;
        self.timed_out = true;
        self
    }
}

fn timeout_marker(label: &str, timeout: Duration) -> String {
    format!(
        "\n[{label} timed out after {:.3}s]\n",
        timeout.as_secs_f64()
    )
}

fn run_git(work: &Path, args: &[&str]) -> io::Result<std::process::Output> {
    Command::new("git").current_dir(work).args(args).output()
}

/// Make the freshly copied fixture a git repo so `git status` reflects only the
/// agent's changes. `--allow-empty` keeps a degenerate fixture off git's
/// "nothing to commit" path. A fixture that already carries `.git` is left as-is.
fn git_init(work: &Path) -> io::Result<()> {
    if work.join(".git").is_dir() {
        return Ok(());
    }
    run_git(work, &["init", "-q"])?;
    run_git(work, &["add", "-A"])?;
    run_git(
        work,
        &[
            "-c",
            "user.email=h@h",
            "-c",
            "user.name=h",
            "commit",
            "-qm",
            "base",
            "--allow-empty",
        ],
    )?;
    Ok(())
}

pub(crate) fn git_porcelain(work: &Path) -> io::Result<String> {
    let out = run_git(work, &["status", "--porcelain"])?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Drop our own scratch files from a porcelain dump before scoring, so they can
/// never count as pollution. The native runner keeps agent output in memory and
/// writes nothing into the work tree, so this is belt-and-suspenders — it also
/// covers a test command that happens to drop a log next to the sources.
pub(crate) fn filter_scratch(porcelain: &str) -> String {
    porcelain
        .lines()
        .filter(|line| {
            let rest = line.get(3..).map_or("", str::trim);
            let path = rest.rsplit(" -> ").next().unwrap_or(rest).trim_matches('"');
            !matches!(
                path,
                ".agent-out.json"
                    | ".agent-err.log"
                    | ".test.log"
                    | ".codex.jsonl"
                    | ".git-status.txt"
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `usage` object lives at the top level of zo/claude/(converted) codex
/// result JSON.
fn find_usage(v: &Value) -> Option<&Value> {
    v.get("usage").filter(|u| u.is_object())
}

/// Parse the billed token breakdown, or `None` when the runner reported no usage
/// (or `input`/`output` are absent) — never fabricated as 0, matching the shell.
fn parse_tokens(stdout: &str) -> Option<Tokens> {
    let v: Value = serde_json::from_str(stdout).ok()?;
    let u = find_usage(&v)?;
    let input = u["input_tokens"].as_u64()?;
    let output = u["output_tokens"].as_u64()?;
    let cache_creation = u["cache_creation_input_tokens"].as_u64();
    let cache_read = u["cache_read_input_tokens"].as_u64();
    let total = input + output + cache_creation.unwrap_or(0) + cache_read.unwrap_or(0);
    let complete = cache_creation.is_some() && cache_read.is_some();
    Some(Tokens {
        input,
        output,
        cache_creation,
        cache_read,
        total,
        complete,
    })
}

/// zo reports `iterations`, claude `num_turns`; either is the turn count.
fn parse_iterations(stdout: &str) -> Option<u64> {
    let v: Value = serde_json::from_str(stdout).ok()?;
    v.get("iterations")
        .and_then(Value::as_u64)
        .or_else(|| v.get("num_turns").and_then(Value::as_u64))
}

/// Count Claude-style top-level `permission_denials` objects (Zo omits the
/// field ⇒ 0). serde array length counts top-level elements only, so nested
/// arrays/objects in a denial's tool input neither inflate nor truncate the count.
fn parse_denials(stdout: &str) -> usize {
    serde_json::from_str::<Value>(stdout)
        .ok()
        .and_then(|v| {
            v.get("permission_denials")
                .and_then(|d| d.as_array().map(Vec::len))
        })
        .unwrap_or(0)
}

/// Recursively copy a directory's contents to seed the work tree from a fixture,
/// matching the shell `cp -R "$FIXTURE/."`.
fn copy_dir(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

// Artifact filenames, aligned with the goal document's recommended per-run set
// (the retired shell harness used hyphenated names; nothing external depends on
// them now). `preserve_artifacts` writes them and `required_artifacts_present`
// gates on them, so writer and checker share one source of truth.
const AGENT_STDOUT_FILE: &str = "agent_stdout.json";
const STDERR_FILE: &str = "stderr.log";
const GIT_STATUS_BEFORE_FILE: &str = "git_status.before.txt";
const GIT_STATUS_AFTER_FILE: &str = "git_status.after.txt";
const GIT_DIFF_FILE: &str = "git_diff.patch";
const TEST_LOG_FILE: &str = "test.log";
const DEEP_PLAN_FILE: &str = "deep_plan.json";
const VERIFIER_RAW_FILE: &str = "verifier_output.raw.txt";
const VERIFIER_PARSED_FILE: &str = "verifier_output.parsed.json";

/// The always-present evidence set every run must leave on disk for
/// replayability: the agent's stdout/stderr and the git state before, after, and
/// as a patch. [`preserve_artifacts`] always writes exactly these and the suite's
/// decision gate checks them via [`required_artifacts_present`] — one shared list,
/// so the writer and the checker can never drift (the goal document's "required
/// artifact set"). `test.log` is *not* listed: a fixture with no test command
/// legitimately produces none, so it is written when present but never gated.
pub const REQUIRED_ARTIFACTS: &[&str] = &[
    AGENT_STDOUT_FILE,
    STDERR_FILE,
    GIT_STATUS_BEFORE_FILE,
    GIT_STATUS_AFTER_FILE,
    GIT_DIFF_FILE,
];

/// The extra evidence a deep-lane run must preserve on top of
/// [`REQUIRED_ARTIFACTS`]: the plan it committed to and the verifier's raw
/// output. `verifier_output.parsed.json` is deliberately absent — it exists only
/// when the verifier output was parseable (the document's "when parseable"), so a
/// malformed/missing/timeout verifier legitimately leaves no parsed file.
pub const REQUIRED_DEEP_ARTIFACTS: &[&str] = &[DEEP_PLAN_FILE, VERIFIER_RAW_FILE];

/// Whether the required artifact set exists as files under `dir` — the real
/// replayability check an accepted run is gated on. A deep-lane run
/// (`deep == true`) must additionally carry [`REQUIRED_DEEP_ARTIFACTS`].
#[must_use]
pub fn required_artifacts_present(dir: &Path, deep: bool) -> bool {
    let present = |name: &&str| dir.join(name).is_file();
    REQUIRED_ARTIFACTS.iter().all(present) && (!deep || REQUIRED_DEEP_ARTIFACTS.iter().all(present))
}

/// Persist the run's diagnosis artifacts when `artifacts_dir` is set. Unlike the
/// shell harness (which preserved only *failed* runs and `rm -rf`'d passing ones),
/// an accepted run is kept too, so the goal document's replayability holds for
/// every leaderboard-eligible run.
fn preserve_artifacts(
    spec: &RunSpec,
    work: &Path,
    agent: &AgentOutput,
    test_log: Option<&str>,
    pass: bool,
    git_status_before: &str,
    deep: Option<&crate::deep::DeepArtifacts>,
) -> io::Result<Option<String>> {
    let Some(dir) = spec.artifacts_dir.as_ref() else {
        return Ok(None);
    };
    fs::create_dir_all(dir)?;
    fs::write(dir.join(AGENT_STDOUT_FILE), &agent.stdout)?;
    fs::write(dir.join(STDERR_FILE), &agent.stderr)?;
    fs::write(dir.join(GIT_STATUS_BEFORE_FILE), git_status_before)?;
    if let Some(log) = test_log {
        fs::write(dir.join(TEST_LOG_FILE), log)?;
    }
    // Capture the after-state porcelain (untracked shown as `??`) *before* staging
    // for the patch, so the snapshot reflects the tree as the agent left it.
    if let Ok(after) = git_porcelain(work) {
        fs::write(dir.join(GIT_STATUS_AFTER_FILE), after)?;
    }
    fs::write(dir.join(GIT_DIFF_FILE), git_diff_patch(work))?;
    if let Some(deep) = deep {
        fs::write(dir.join(DEEP_PLAN_FILE), &deep.plan_json)?;
        fs::write(dir.join(VERIFIER_RAW_FILE), &deep.verifier_raw)?;
        if let Some(parsed) = &deep.verifier_parsed {
            fs::write(dir.join(VERIFIER_PARSED_FILE), parsed)?;
        }
    }
    if spec.keep_failed && !pass {
        copy_dir(work, &dir.join("worktree"))?;
    }
    Ok(Some(dir.to_string_lossy().into_owned()))
}

/// The run's full change set as a unified diff, including new files: stage
/// everything so additions appear, then diff the index against HEAD. The work
/// tree is a throwaway temp dir and this runs after scoring, so mutating its
/// index cannot affect the verdict. Best-effort — a git failure yields an empty
/// patch rather than aborting artifact preservation.
fn git_diff_patch(work: &Path) -> String {
    let _ = run_git(work, &["add", "-A"]);
    run_git(work, &["diff", "--cached"])
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default()
}

/// Canonicalize a model label to the family-versioned name the result records
/// (`opus` → `claude-opus-4-8`), mirroring the shell `normalize_model_label`.
#[must_use]
pub fn normalize_model_label(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase().replace('.', "-");
    match lower.as_str() {
        "opus" | "claude-opus" | "claude-opus-4" | "claude-opus-4-8" => {
            "claude-opus-4-8".to_string()
        }
        "sonnet" | "claude-sonnet" | "claude-sonnet-4" | "claude-sonnet-4-6" => {
            "claude-sonnet-4-6".to_string()
        }
        "" => "unknown".to_string(),
        _ => lower,
    }
}

/// Normalize an effort label, mirroring the shell `normalize_effort_label`.
#[must_use]
pub fn normalize_effort_label(raw: &str) -> String {
    let r = raw.trim().to_ascii_lowercase();
    match r.as_str() {
        "" => "unknown".to_string(),
        "none" | "disable" | "disabled" => "off".to_string(),
        "med" => "medium".to_string(),
        // P9 rename: `ultracode` -> `smart` (persisted/legacy spellings keep
        // normalizing to the current preset name). `ultra` is now its own
        // static top-tier level, NOT an alias of `smart` — it normalizes to
        // itself, matching the `_ => r` fallthrough (kept as an explicit arm
        // for discoverability of the label-vocabulary change).
        "smart" | "smartcode" | "ultracode" | "uc" => "smart".to_string(),
        "ultra" => "ultra".to_string(),
        _ => r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_label_canonicalizes() {
        assert_eq!(normalize_model_label("opus"), "claude-opus-4-8");
        assert_eq!(normalize_model_label("Sonnet"), "claude-sonnet-4-6");
        assert_eq!(normalize_model_label(""), "unknown");
        assert_eq!(normalize_model_label("gpt-5.5"), "gpt-5-5");
    }

    #[test]
    fn effort_label_normalizes() {
        assert_eq!(normalize_effort_label("med"), "medium");
        assert_eq!(normalize_effort_label("none"), "off");
        assert_eq!(normalize_effort_label(""), "unknown");
        assert_eq!(normalize_effort_label("high"), "high");
        // `ultracode`/`smartcode`/`uc` are all pre-rename or short spellings
        // of the current `smart` preset.
        assert_eq!(normalize_effort_label("ultracode"), "smart");
        assert_eq!(normalize_effort_label("smartcode"), "smart");
        assert_eq!(normalize_effort_label("uc"), "smart");
        assert_eq!(normalize_effort_label("smart"), "smart");
        // `ultra` is its own static level now (label-vocabulary change) — no
        // longer normalizes to the orchestration preset.
        assert_eq!(normalize_effort_label("ultra"), "ultra");
    }

    #[test]
    fn zo_benchmark_env_defaults_point_under_artifacts_dir() {
        if [
            "ZO_ARTIFACT_STORE",
            "ZO_SESSION_ROOT",
            "ZO_WORKFLOW_STORE",
            "ZO_TODO_STORE",
        ]
        .iter()
        .any(|key| std::env::var_os(key).is_some())
        {
            return;
        }
        let artifacts = PathBuf::from("bench/results/sample/run/artifacts");
        let spec = RunSpec {
            runner: "zo_gpt".into(),
            runner_kind: "zo".into(),
            bin: PathBuf::from("/bin/true"),
            args: vec![],
            fixture: PathBuf::from("fixture"),
            prompt: "do it".into(),
            test_command: Some("true".into()),
            intended: vec!["src/main.js".into()],
            lane: "deep".into(),
            model: "gpt-5.5".into(),
            effort: "xhigh".into(),
            objective_gate: "test_and_diff".into(),
            diff_policy: "intended_paths_only".into(),
            timeout_seconds: 300,
            artifacts_dir: Some(artifacts.clone()),
            keep_failed: false,
            deep: None,
        };
        let mut cmd = Command::new("true");
        apply_zo_benchmark_env(&mut cmd, &spec);
        let envs: std::collections::BTreeMap<_, _> = cmd
            .get_envs()
            .filter_map(|(key, value)| {
                Some((key.to_string_lossy().to_string(), PathBuf::from(value?)))
            })
            .collect();
        let root = absolute_path(artifacts.join("zo-runtime"));
        assert_eq!(envs["ZO_ARTIFACT_STORE"], root.join("artifacts"));
        assert_eq!(envs["ZO_SESSION_ROOT"], root.join("sessions"));
        assert_eq!(envs["ZO_WORKFLOW_STORE"], root.join("workflows"));
        assert_eq!(envs["ZO_TODO_STORE"], root.join("todos.json"));
    }

    #[test]
    fn tokens_total_is_billed_sum_with_complete_flag() {
        let out = r#"{"usage":{"input_tokens":10,"output_tokens":20,"cache_creation_input_tokens":5,"cache_read_input_tokens":100}}"#;
        let t = parse_tokens(out).unwrap();
        assert_eq!(t.total, 135);
        assert!(t.complete);
    }

    #[test]
    fn tokens_incomplete_without_cache_counters() {
        let out = r#"{"usage":{"input_tokens":10,"output_tokens":20}}"#;
        let t = parse_tokens(out).unwrap();
        assert_eq!(t.total, 30);
        assert!(!t.complete);
        assert!(t.cache_read.is_none());
    }

    #[test]
    fn tokens_absent_usage_is_none() {
        assert!(parse_tokens(r#"{"result":"ok"}"#).is_none());
        assert!(parse_tokens("not json").is_none());
    }

    #[test]
    fn denials_counts_top_level_objects_only() {
        let out =
            r#"{"permission_denials":[{"tool":"Read"},{"tool":"Bash","input":{"nested":[1,2]}}]}"#;
        assert_eq!(parse_denials(out), 2);
        assert_eq!(parse_denials(r#"{"result":"ok"}"#), 0);
    }

    #[test]
    fn codex_jsonl_folds_to_zo_shape() {
        let jsonl = "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"done\"}}\n{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":100,\"cached_input_tokens\":80,\"output_tokens\":30,\"reasoning_output_tokens\":10}}";
        let folded = convert_codex_jsonl(jsonl);
        let v: Value = serde_json::from_str(&folded).unwrap();
        assert_eq!(v["message"], "done");
        assert_eq!(v["usage"]["input_tokens"], 20); // 100 - 80
        assert_eq!(v["usage"]["output_tokens"], 40); // 30 + 10
        assert_eq!(v["usage"]["cache_read_input_tokens"], 80);
        let t = parse_tokens(&folded).unwrap();
        assert_eq!(t.total, 20 + 40 + 80);
    }

    #[test]
    fn filter_scratch_drops_harness_files() {
        let porc = " M src/range.js\n?? .agent-out.json\n?? .test.log";
        assert_eq!(filter_scratch(porc), " M src/range.js");
    }

    #[test]
    fn diff_policy_any_allows_source_edits_but_not_runtime_pollution() {
        let (hygiene, intended_provided) = score_for_policy(" M docs/review.md\n", &[], "any");
        assert!(hygiene.clean);
        assert!(!intended_provided);
        assert!(hygiene.unexpected.is_empty());

        let (polluted, intended_provided) =
            score_for_policy(" M docs/review.md\n?? .zo/session.json\n", &[], "any");
        assert!(!polluted.clean);
        assert!(!intended_provided);
        assert_eq!(polluted.pollution, vec![".zo/session.json"]);
    }

    #[test]
    fn verifier_only_objective_ignores_test_failure_but_keeps_diff_policy() {
        let (hygiene, intended_provided) = score_for_policy(" M docs/review.md\n", &[], "any");
        assert!(run_passed_for_policy(
            0,
            TestStatus::Fail,
            &hygiene,
            0,
            intended_provided,
            "verifier_only",
        ));

        let (polluted, intended_provided) =
            score_for_policy("?? .zo/session.json\n", &[], "any");
        assert!(!run_passed_for_policy(
            0,
            TestStatus::Pass,
            &polluted,
            0,
            intended_provided,
            "verifier_only",
        ));
    }

    #[test]
    fn deep_objective_evidence_can_salvage_agent_timeout_after_green_tests() {
        let (hygiene, intended_provided) =
            score_for_policy(" M src/store.js\n", &["src/store.js"], "test_and_diff");
        assert!(!run_passed_for_policy(
            -1,
            TestStatus::Pass,
            &hygiene,
            0,
            intended_provided,
            "test_and_diff",
        ));
        assert!(objective_evidence_passed_for_policy(
            TestStatus::Pass,
            &hygiene,
            0,
            intended_provided,
            "test_and_diff",
        ));

        let dirty = score_for_policy(
            " M src/store.js\n M src/unrelated.js\n",
            &["src/store.js"],
            "test_and_diff",
        );
        assert!(!objective_evidence_passed_for_policy(
            TestStatus::Pass,
            &dirty.0,
            0,
            dirty.1,
            "test_and_diff",
        ));
        assert!(!objective_evidence_passed_for_policy(
            TestStatus::Fail,
            &hygiene,
            0,
            intended_provided,
            "test_and_diff",
        ));
    }

    #[cfg(unix)]
    #[test]
    fn command_timeout_kills_process_group_and_marks_output() {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("sleep 5; echo late");
        let out = run_command(cmd, Some(Duration::from_millis(10)), "agent").unwrap();
        assert!(out.timed_out);
        assert!(!out.success);
        assert_eq!(out.exit_code, -1);
        assert!(!String::from_utf8_lossy(&out.stdout).contains("late"));
        assert!(String::from_utf8_lossy(&out.stderr).contains("agent timed out"));
    }

    #[test]
    fn run_command_stamps_startup_at_first_output() {
        // First output arrives only after the 0.4s sleep, so the recorded start-up
        // (spawn → first byte) must reflect that delay — proving it measures
        // time-to-first-output, not merely spawn cost.
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("sleep 0.4; printf hello");
        let out = run_command(cmd, Some(Duration::from_secs(10)), "agent").unwrap();
        assert!(out.success);
        assert_eq!(out.stdout, b"hello");
        let startup = out.startup.expect("first output observed");
        assert!(
            startup >= Duration::from_millis(300),
            "startup {startup:?} should reflect the 0.4s pre-output delay"
        );
    }

    #[test]
    fn run_command_drains_large_output_without_deadlock() {
        // 200KB exceeds the OS pipe buffer (~64KB). Without continuous draining the
        // child would block on write and never exit, so a buffer-at-end reader
        // would time out. Streaming readers capture it all and the run completes.
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("yes x | head -c 200000");
        let out = run_command(cmd, Some(Duration::from_secs(30)), "agent").unwrap();
        assert!(out.success);
        assert!(!out.timed_out);
        assert_eq!(out.stdout.len(), 200_000);
    }

    #[test]
    fn run_budget_reserving_capped_can_reduce_child_to_zero_without_spending_parent() {
        let parent = RunBudget::from_duration(Duration::from_secs(30));
        let child = parent.reserving_capped(Duration::from_secs(45), Duration::from_secs(75));
        assert_eq!(child.remaining(), Some(Duration::ZERO));
        assert!(
            parent.remaining().unwrap_or_default() > Duration::from_secs(20),
            "parent budget should remain available for objective validation"
        );
    }

    #[test]
    fn run_budget_reserving_capped_preserves_validation_time_and_limits_child() {
        let parent = RunBudget::from_duration(Duration::from_secs(300));
        let child = parent.reserving_capped(Duration::from_secs(60), Duration::from_secs(75));
        assert!(child.remaining().unwrap_or_default() <= Duration::from_secs(75));
        assert!(child.remaining().unwrap_or_default() > Duration::from_secs(70));
        assert!(
            parent.remaining().unwrap_or_default() > Duration::from_secs(290),
            "parent budget should remain available for later phases"
        );
    }

    #[cfg(unix)]
    #[test]
    fn run_test_timeout_fails_and_surfaces_log_marker() {
        let dir = TempDir::new().unwrap();
        let budget = RunBudget::from_duration(Duration::from_millis(10));
        let test = run_test(dir.path(), "sleep 5; echo late", &budget);
        assert_eq!(test.status, TestStatus::Fail);
        assert!(test.timed_out);
        let log = test.log.unwrap();
        assert!(!log.contains("late"));
        assert!(log.contains("test timed out"));
    }

    #[test]
    fn run_result_json_matches_shell_field_shape() {
        // A fast-lane result serializes with deep:null and the shell's field names.
        let r = RunResult {
            runner: "zo".into(),
            model: "claude-opus-4-8".into(),
            effort: "medium".into(),
            lane: "fast".into(),
            exit_code: 0,
            wall_seconds: 12,
            startup_seconds: Some(2),
            test: TestStatus::Pass,
            intended_changed: 1,
            permission_denials: 0,
            pollution: vec![],
            unexpected: vec![],
            clean_diff: true,
            pass: true,
            tokens: Some(Tokens {
                input: 10,
                output: 20,
                cache_creation: None,
                cache_read: None,
                total: 30,
                complete: false,
            }),
            iterations: Some(3),
            fail_reasons: vec![],
            warnings: vec![],
            artifact_dir: None,
            deep: None,
        };
        let v: Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["test"], "pass");
        assert!(v["deep"].is_null());
        assert!(v["artifact_dir"].is_null());
        assert_eq!(v["clean_diff"], true);
        assert_eq!(v["tokens"]["complete"], false);
        assert_eq!(v["startup_seconds"], 2);
    }

    #[test]
    fn run_metrics_are_derived_from_observed_result_fields() {
        let r = RunResult {
            runner: "zo".into(),
            model: "gpt-5.5-fast".into(),
            effort: "high".into(),
            lane: "deep".into(),
            exit_code: -1,
            wall_seconds: 42,
            startup_seconds: Some(5),
            test: TestStatus::Fail,
            intended_changed: 0,
            permission_denials: 0,
            pollution: vec![".zo/session.json".into()],
            unexpected: vec![],
            clean_diff: false,
            pass: false,
            tokens: Some(Tokens {
                input: 11,
                output: 7,
                cache_creation: Some(3),
                cache_read: Some(2),
                total: 23,
                complete: true,
            }),
            iterations: Some(2),
            fail_reasons: vec!["agent_timeout".into(), "deep_unverified".into()],
            warnings: vec![],
            artifact_dir: None,
            deep: Some(crate::deep::DeepVerdict {
                attempts: 3,
                max_attempts: 3,
                plan_valid: true,
                verifier_accepted: false,
                outcome: "reject".into(),
                diagnostics: crate::deep::DeepDiagnostics {
                    plan_missing: vec![],
                    verifier_parse: "malformed".into(),
                    verifier_issues: 1,
                    objective_passed: false,
                    phase_timed_out: true,
                    plan_recovered: false,
                    verifier_recovered_by_objective: false,
                    deterministic_probe_issues: 0,
                    failure: "verifier_malformed".into(),
                },
                phase_timings: crate::deep::DeepPhaseTimings::default(),
            }),
        };
        let metrics = r.metrics();
        assert_eq!(metrics.wall_seconds, 42);
        assert_eq!(metrics.startup_seconds, Some(5));
        assert_eq!(metrics.token_total, Some(23));
        assert_eq!(metrics.retry_count, 2);
        assert!(metrics.dirty_diff);
        assert!(metrics.deep_verifier_failed);
        assert!(metrics.timeout);
        assert!(!metrics.provider_error);
        assert!(metrics.cost_usd.is_none());
        assert!(metrics.cost_normalized_score.is_none());
    }
}
