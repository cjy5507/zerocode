//! Deep-lane gate: a live `plan → implement → verify → retry` turn orchestrator.
//!
//! This wires the pure decision brain in [`decision_core::deep_lane`] (already
//! unit-tested and shared with the benchmark harness) into the interactive
//! streaming loop. It is **not** a second copy of that policy —
//! [`validate_plan`], [`parse_lens_verifier`] and [`fold_verification_attempt`]
//! are called directly, so the accept/retry rules can never drift
//! from the benchmark. Only the *live IO* lives here, because it genuinely
//! differs from the benchmark's subprocess path:
//!
//! - each phase is one streaming sub-turn (`run_turn_streaming_with_images`)
//!   instead of a spawned `zo -p`;
//! - the objective gate is the project's own check command, run through the
//!   shared [`crate::execute_bash`] chokepoint (the same green source the
//!   workflow `command_green` check converges on);
//! - the PLAN and VERIFY sub-turns run under [`PermissionMode::ReadOnly`] so the
//!   model cannot edit before a valid plan exists and the verifier inspects but
//!   never mutates.
//!
//! See `docs/design-deep-lane-live-wiring.md` for the full rationale and the
//! crate-dependency constraints that make a focused live gate (rather than a
//! full extraction of the benchmark loop) the right shape.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use decision_core::deep_lane::{
    DeepDecision, MAX_SUMMARY_CHARS, PlanVerdict, VerificationAttempt, VerifierParse,
    VerifierVerdict, fold_verification_attempt, parse_lens_verifier, validate_plan,
};

use crate::hooks::HookEvent;
use crate::message_stream::types::{BlockIdGen, RenderBlock, SystemLevel};
use crate::model_router::{RouteTaskComplexity, RouteTaskRisk};
use crate::permission::PermissionPrompter as AsyncPermissionPrompter;
use crate::session::{ContentBlock, ConversationMessage, MessageRole};
use crate::usage::TokenUsage;
use crate::{BashCommandInput, PermissionMode, execute_bash};
use crate::permissions::TemporaryAllowGrant;

use super::{build_turn_end_hook_context, changed_files_snapshot_async};

use super::{
    ApiClient, AsyncApiClient, AutoCompactionEvent, BudgetExhausted, ConversationRuntime,
    PromptCacheEvent, StreamingTurnError, ToolExecutor, TurnSummary,
};

/// How the gate structures a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeepMode {
    /// Reactive (default): run the turn normally with full tools, then — only if
    /// it actually edited files — auto-verify the diff and retry on failure. No
    /// read-only planning phase, so there is no permission friction; chat and
    /// analysis turns pass straight through with zero overhead.
    #[default]
    Reactive,
    /// Plan-first: force a structured read-only PLAN before any edit, then
    /// implement → verify → retry. Stronger guarantee for hard tasks at the cost
    /// of a read-only planning phase (bash is blocked there).
    PlanFirst,
}

/// How many implementer attempts the Architect contract runs before
/// escalating the EXEC leg back to the native (reserved) model. Mirrors the
/// router's `implementation_route_model_allowed` escape (`prior_failures >=
/// 2`): two real implementer failures are the contract's own escalation
/// signal, so the third attempt runs on the session's premium model.
const ARCHITECT_IMPL_ATTEMPTS: u32 = 2;
const CHECK_OUTPUT_TAIL_BYTES: usize = 4_000;
const VERIFY_EDITED_PATHS_BYTES: usize = 2_000;
const VERIFY_ASSISTANT_CLAIM_BYTES: usize = 4_000;
const EXEC_PRIOR_DIFF_BYTES: usize = 6_000;
const EXEC_PRIOR_EDITED_PATHS_BYTES: usize = 2_000;
// Two paths still covers one focused edit plus a directly coupled companion file.
const FILES_TRIVIAL_MAX: usize = 2;
// Twenty-four changed lines keeps skip eligibility to genuinely tiny patches.
const CHURN_TRIVIAL_MAX: usize = 24;
// Four paths bounds spec-only review to a small, locally auditable change.
const FILES_SMALL_MAX: usize = 4;
// One hundred sixty changed lines is a conservative ceiling for modest churn.
const CHURN_SMALL_MAX: usize = 160;
const SECURITY_PATH_MARKERS: &[&str] = &[
    "auth",
    "secret",
    "credential",
    "token",
    "crypto",
    "password",
];
const TEST_PATH_MARKERS: &[&str] = &["test", "_test.", ".test.", "spec"];

/// Per-turn Architect execution contract (`smart.policy=architect`): the
/// metadata for an implementation-shaped turn whose session main model is
/// reserved for plan/orchestrate/verify duty
/// ([`crate::is_reserved_orchestrator_model`]).
///
/// Installed by the host on every turn entry via
/// [`ConversationRuntime::set_exec_contract`] — set-or-cleared, mirroring
/// `set_deep_verify_candidates`, so it can never outlive its turn. `None`
/// keeps the pre-contract behavior. The optional implementer client gates only
/// the EXEC swap (`smart.execSwap`); plan-first promotion can remain active
/// while EXEC runs on the native client, but the foreground edit gate arms only
/// for a live swap. PLAN and VERIFY use their independent deep-lane clients.
#[derive(Clone)]
pub struct ExecContract {
    /// The implementer client EXEC legs swap to (attempts
    /// `1..=ARCHITECT_IMPL_ATTEMPTS`). `None` means the configured policy did
    /// not arm a swap for this turn's difficulty.
    pub impl_client: Option<Arc<dyn AsyncApiClient>>,
    /// The implementer's model id, for narration and telemetry.
    pub impl_model: String,
    /// Run the read-only PLAN phase before the first EXEC even when the gate
    /// mode is Reactive (complex/multi-scope implementation turns).
    pub plan_first: bool,
}

impl ExecContract {
    /// Whether this contract swaps EXEC legs away from the session client.
    #[must_use]
    pub fn exec_swap_enabled(&self) -> bool {
        self.impl_client.is_some()
    }
}

impl std::fmt::Debug for ExecContract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecContract")
            .field("impl_model", &self.impl_model)
            .field("plan_first", &self.plan_first)
            .field("exec_swap_enabled", &self.exec_swap_enabled())
            .finish_non_exhaustive()
    }
}

/// Which upstream client a deep-gate sub-turn (leg) runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubturnClient {
    /// The session's native client (reserved PLAN or EXEC without a contract).
    Native,
    /// The Architect deep-tier PLAN client; falls back to Native only when the
    /// session model is itself reserved deep-tier.
    Plan,
    /// A native-client Architect EXEC leg. It remains exempt from the edit
    /// gate without pretending that an implementer client was swapped in.
    NativeExec,
    /// The ranked cross-model verifier candidate selected by `verify_subturn`
    /// (`deep_verify_candidate_idx`).
    Verify,
    /// The Architect contract's implementer ([`ExecContract::impl_client`]).
    /// Falls back to Native when no contract is installed.
    Implementer,
}

/// Configuration for the deep-lane gate. Installed on a [`ConversationRuntime`]
/// via [`ConversationRuntime::set_deep_gate`]; absent (`None`) means a turn runs
/// the ordinary single-pass loop with no verification.
#[derive(Debug, Clone)]
pub struct DeepGateConfig {
    /// How the turn is structured (reactive vs plan-first).
    pub mode: DeepMode,
    /// Project check command whose exit 0 is treated as objectively green. When
    /// `None`, there is no objective gate and the adversarial verifier alone
    /// decides acceptance. The host can fill this from [`detect_check_command`].
    pub check_command: Option<String>,
    /// Upper bound on the implement→verify retries (and, in plan-first mode, the
    /// plan re-tries).
    pub max_attempts: u32,
}

impl Default for DeepGateConfig {
    fn default() -> Self {
        Self {
            mode: DeepMode::Reactive,
            check_command: None,
            max_attempts: 2,
        }
    }
}

/// Best-effort detection of the project's check command from the working
/// directory, for the reactive objective gate. Returns `None` when no known
/// project marker is present (the verifier then decides alone). First match in a
/// fixed order wins, so a mixed repo gets a deterministic choice.
///
/// This is auto-wired as the per-coding-turn reactive gate (it runs after every
/// edited turn — see `install_reactive_verify_gate_if_coding`), so the default
/// must be a *cheap* objective signal, not a full multi-minute test run on a
/// large repo. For Rust the default is therefore `cargo build --tests`: it
/// compiles every test target (catching the same build/type errors a
/// `cargo test` build would surface) without paying for the test *execution*.
/// The heavier full `cargo test` stays available where a turn explicitly asks
/// for it; the reactive auto default favors a fast green-build gate.
#[must_use]
pub fn detect_check_command() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let has = |name: &str| cwd.join(name).is_file();
    if has("Cargo.toml") {
        return Some("cargo build --tests".to_string());
    }
    if has("package.json") {
        return Some("npm test".to_string());
    }
    if has("pyproject.toml") || has("pytest.ini") || has("setup.cfg") {
        return Some("pytest".to_string());
    }
    if has("go.mod") {
        return Some("go test ./...".to_string());
    }
    if let Ok(entries) = std::fs::read_dir(&cwd) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension() {
                if ext == "sln" || ext == "csproj" {
                    return Some("dotnet test".to_string());
                }
            }
        }
    }
    None
}

/// The structured result of a deep turn, for telemetry and the final note.
#[derive(Debug, Clone)]
pub struct DeepOutcome {
    pub decision: DeepDecision,
    pub attempts: u32,
    pub plan_valid: bool,
    pub plan_missing: Vec<String>,
    /// The adversarial verifier's semantic verdict on the change, surfaced for
    /// goal-completion gating (anti "optimistic stop"). `Some(true)` = the
    /// VERIFY phase accepted an edit-making turn, `Some(false)` = it rejected
    /// or the gate gave up, `None` = no semantic judgment was made this turn
    /// (a chat/analysis turn that changed nothing, or a trivial green edit whose
    /// proportional depth skipped VERIFY). Distinct from `decision`,
    /// which also reports `Accept` for a no-edit turn that was never verified.
    pub verification: Option<bool>,
    /// The concrete problems the adversarial verifier raised on the FINAL
    /// attempt (the unresolved rejection that ended the inner loop). Empty when
    /// the change was accepted or no verify ran. Surfaced so the goal-level
    /// repair prompt can tell the model *what specifically to fix* instead of a
    /// generic "rejected, try again" — the inner loop already feeds these back
    /// (`failure_summary`); this carries the same signal to the outer loop.
    pub issues: Vec<String>,
    /// How the most recent VERIFY sub-turn's verdict was recovered (Phase 4
    /// verdict-channel seam — NOT consumed by any accept/retry/stall policy,
    /// which stays entirely in `decision-core`). `None` when no VERIFY
    /// sub-turn ran this turn (a no-edit chat/analysis turn or proportional
    /// VERIFY skip). `Some(
    /// VerifierParse::Json | VerifierParse::Salvaged)` means the verifier
    /// actually produced a usable verdict — `verification` reflects it and a
    /// verdict-outcome recorder may safely record it. `Some(VerifierParse::
    /// Empty | Unparseable | Timeout)` means a VERIFY sub-turn ran but
    /// recovered no usable signal (`verification` is still gated
    /// conservatively for goal-completion purposes) — a verdict-outcome
    /// recorder MUST treat this as "no signal, do not record" per the
    /// ambiguous-verdicts-are-never-recorded doctrine.
    pub verifier_parse: Option<VerifierParse>,
    /// The cross-model verifier's model id when [`ConversationRuntime::
    /// set_deep_verify_client`] installed one for this turn's VERIFY leg
    /// (`None` when the leg ran on the turn's own native model, or when no
    /// VERIFY leg ran this turn — same absence condition as `verifier_parse`).
    /// Surfaced so a verdict-outcome recorder can credit/blame the verifier's
    /// OWN run under its real model, distinct from the main turn's model.
    pub verifier_model: Option<String>,
}

// ── Pure helpers (no IO; unit-tested below) ──────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum VerifyDepth {
    Skip,
    SingleLens,
    Full,
}

fn verify_depth(
    complexity: RouteTaskComplexity,
    risk: RouteTaskRisk,
    files_changed: usize,
    line_churn: usize,
    objective_ok: bool,
    touches_security: bool,
    touches_tests: bool,
) -> VerifyDepth {
    if files_changed == 0
        || matches!(
            complexity,
            RouteTaskComplexity::Medium
                | RouteTaskComplexity::Large
                | RouteTaskComplexity::Unknown
        ) || matches!(
        risk,
        RouteTaskRisk::High | RouteTaskRisk::Critical | RouteTaskRisk::Unknown
    ) || !objective_ok
        || files_changed > FILES_SMALL_MAX
        || line_churn > CHURN_SMALL_MAX
        || touches_security
        || touches_tests
    {
        return VerifyDepth::Full;
    }

    if complexity == RouteTaskComplexity::Small
        && matches!(risk, RouteTaskRisk::Low | RouteTaskRisk::Medium)
    {
        return VerifyDepth::SingleLens;
    }

    if complexity == RouteTaskComplexity::Trivial
        && risk == RouteTaskRisk::Low
        && files_changed <= FILES_TRIVIAL_MAX
        && line_churn <= CHURN_TRIVIAL_MAX
    {
        return VerifyDepth::Skip;
    }

    VerifyDepth::Full
}

fn verify_depth_for_band(
    band: Option<(RouteTaskComplexity, RouteTaskRisk)>,
    files_changed: usize,
    line_churn: usize,
    objective_ok: bool,
    touches_security: bool,
    touches_tests: bool,
) -> VerifyDepth {
    let Some((complexity, risk)) = band else {
        return VerifyDepth::Full;
    };
    verify_depth(
        complexity,
        risk,
        files_changed,
        line_churn,
        objective_ok,
        touches_security,
        touches_tests,
    )
}

fn paths_touch_security(paths: &[String]) -> bool {
    paths.iter().any(|path| {
        let path = path.to_ascii_lowercase();
        SECURITY_PATH_MARKERS
            .iter()
            .any(|marker| path.contains(marker))
    })
}

fn paths_touch_tests(paths: &[String]) -> bool {
    paths.iter().any(|path| {
        let path = path.to_ascii_lowercase();
        TEST_PATH_MARKERS
            .iter()
            .any(|marker| path.contains(marker))
    })
}

/// Map a bash `return_code_interpretation` to "green" (exit 0). `None` means the
/// command exited 0 and produced the expected output; any other interpretation
/// (`exit_code:N`, `timeout`, …) is not green. Mirrors the workflow
/// `command_green` reader so both paths agree on what green means.
fn interpret_green(interpretation: Option<&str>) -> bool {
    match interpretation {
        None => true,
        Some(code) => {
            code.strip_prefix("exit_code:")
                .and_then(|n| n.parse::<i32>().ok())
                == Some(0)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CheckObservation {
    green: bool,
    output_tail: String,
}

/// Run the objective check command and report whether it was green. A command
/// that fails to even start is treated as not-green, never a panic. Runs in the
/// live process cwd (the working tree) via the shared [`crate::execute_bash`]
/// chokepoint.
///
/// `execute_bash` is a *blocking* call (it drives a child process to completion
/// via `block_in_place`). The reactive/deep gate runs inside the host's live
/// streaming turn, whose `select!` polls the turn future on the same task; a
/// `block_in_place` there suspends the whole task, freezing the TUI's input,
/// mouse and spinner for the entire (potentially multi-minute) check run. So
/// offload it to a dedicated blocking thread via `spawn_blocking` and `await`
/// the result — the turn task yields, keeping the event loop live throughout.
async fn run_check_command(command: &str) -> CheckObservation {
    let command = command.to_string();
    let Ok(observation) = tokio::task::spawn_blocking(move || {
        let Ok(input) = serde_json::from_value::<BashCommandInput>(json!({ "command": command }))
        else {
            return CheckObservation {
                green: false,
                output_tail: "check command could not be parsed".to_string(),
            };
        };
        match execute_bash(input) {
            Ok(out) => CheckObservation {
                green: interpret_green(out.return_code_interpretation.as_deref()),
                output_tail: bounded_check_output_tail(&out.stdout, &out.stderr),
            },
            Err(error) => CheckObservation {
                green: false,
                output_tail: format!("check command failed to start: {error}"),
            },
        }
    })
    .await
    else {
        // The blocking task panicked or was cancelled — treat as not-green
        // rather than propagating a panic into the turn loop.
        return CheckObservation {
            green: false,
            output_tail: "check command runner was cancelled".to_string(),
        };
    };
    observation
}

async fn command_is_green(command: &str) -> bool {
    run_check_command(command).await.green
}

fn bounded_check_output_tail(stdout: &str, stderr: &str) -> String {
    let mut output = String::new();
    if !stdout.trim().is_empty() {
        let _ = writeln!(output, "stdout:\n{}", stdout.trim_end());
    }
    if !stderr.trim().is_empty() {
        let _ = writeln!(output, "stderr:\n{}", stderr.trim_end());
    }
    if output.is_empty() {
        return "(no output)".to_string();
    }
    truncate_to_tail_on_boundary(&mut output, CHECK_OUTPUT_TAIL_BYTES);
    output
}

/// A bounded `git diff` of the relevant working-tree paths for the verifier
/// prompt. When `paths` is empty it falls back to the full working-tree diff;
/// otherwise it asks git for only those pathspecs so unrelated pre-existing
/// dirt cannot crowd the actual attempt out of the bounded verifier prompt.
/// Read-only; an unavailable git or oversized diff degrades to a truncated best
/// effort.
///
/// Offloaded to a blocking thread for the same reason as [`command_is_green`]:
/// even a scoped `git diff` can walk the index and must not freeze the live TUI
/// event loop while it runs.
async fn bounded_git_diff_for_paths(paths: Vec<String>, max: usize) -> (String, usize) {
    let mut diff = tokio::task::spawn_blocking(move || scoped_git_diff(&paths))
        .await
        .unwrap_or_default();
    let line_churn = diff_line_churn(&diff);
    truncate_on_boundary(&mut diff, max);
    (diff, line_churn)
}

fn diff_line_churn(diff: &str) -> usize {
    diff.lines()
        .filter(|line| {
            (line.starts_with('+') && !line.starts_with("+++ "))
                || (line.starts_with('-') && !line.starts_with("--- "))
        })
        .count()
}

fn scoped_git_diff(paths: &[String]) -> String {
    if paths.is_empty() {
        return run_git_diff(&[]);
    }

    let mut diff = run_git_diff(paths);
    append_untracked_file_diffs(&mut diff, paths);
    if diff.trim().is_empty() {
        let _ = writeln!(
            diff,
            "(no git diff for scoped attempt paths: {})",
            paths.join(", ")
        );
    }
    diff
}

fn run_git_diff(paths: &[String]) -> String {
    let mut command = std::process::Command::new("git");
    command.args(["--no-optional-locks"]);
    if !paths.is_empty() {
        command.args(["--literal-pathspecs"]);
    }
    command.args(["diff"]);
    if !paths.is_empty() {
        command.arg("--");
        command.args(paths);
    }
    match command.output() {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout).into(),
        _ => String::new(),
    }
}

fn append_untracked_file_diffs(diff: &mut String, paths: &[String]) {
    for path in paths {
        if !path_is_untracked_file(path) {
            continue;
        }
        if !diff.is_empty() && !diff.ends_with('\n') {
            diff.push('\n');
        }
        diff.push_str(&run_no_index_new_file_diff(path));
    }
}

fn path_is_untracked_file(path: &str) -> bool {
    if !std::path::Path::new(path).is_file() {
        return false;
    }
    let output = std::process::Command::new("git")
        .args([
            "--no-optional-locks",
            "--literal-pathspecs",
            "ls-files",
            "--error-unmatch",
            "--",
            path,
        ])
        .output();
    matches!(output, Ok(output) if !output.status.success())
}

fn run_no_index_new_file_diff(path: &str) -> String {
    let empty = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let output = std::process::Command::new("git")
        .args([
            "--no-optional-locks",
            "diff",
            "--no-index",
            "--",
            empty,
            path,
        ])
        .output();
    match output {
        // `git diff --no-index` returns 1 when files differ; stdout is still the
        // useful patch. Treat any stdout as best-effort diff content.
        Ok(output) if !output.stdout.is_empty() => String::from_utf8_lossy(&output.stdout).into(),
        _ => String::new(),
    }
}

fn edited_file_paths(summary: &TurnSummary) -> Vec<String> {
    let mut paths = BTreeSet::new();
    for message in &summary.tool_results {
        for block in &message.blocks {
            let ContentBlock::ToolResult {
                tool_name,
                output,
                is_error,
                ..
            } = block
            else {
                continue;
            };
            if *is_error || !super::is_edit_or_write_tool(tool_name) {
                continue;
            }
            if let Some(path) = tool_result_path(output) {
                paths.insert(path);
            }
        }
    }
    paths.into_iter().collect()
}

fn tool_result_path(output: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    ["filePath", "path", "file_path"]
        .into_iter()
        .find_map(|key| value.get(key).and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(ToOwned::to_owned)
}

fn attempt_diff_paths(
    baseline_files: &[String],
    after_files: &[String],
    edited_paths: &[String],
) -> Vec<String> {
    let baseline: BTreeSet<&str> = baseline_files.iter().map(String::as_str).collect();
    let mut paths: BTreeSet<String> = edited_paths
        .iter()
        .map(|path| path.trim())
        .filter(|path| !path.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    paths.extend(
        after_files
            .iter()
            .map(String::as_str)
            .filter(|path| !baseline.contains(path))
            .map(ToOwned::to_owned),
    );
    paths.into_iter().collect()
}

/// Truncate `s` to at most `max` bytes, never splitting a UTF-8 char.
fn truncate_on_boundary(s: &mut String, max: usize) {
    if s.len() <= max {
        return;
    }
    let mut n = max;
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    s.truncate(n);
}

fn truncate_to_tail_on_boundary(s: &mut String, max: usize) {
    if s.len() <= max {
        return;
    }
    let mut start = s.len().saturating_sub(max);
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    s.drain(..start);
}

/// Build the retry repair contract from the failing attempt, bounded to
/// [`MAX_SUMMARY_CHARS`] so a retry's added model cost stays small.
fn failure_summary(objective_ok: bool, verifier: &VerifierVerdict) -> String {
    let mut s =
        String::from("Your previous attempt did NOT pass. Treat this as the repair contract:\n");
    if !objective_ok {
        let _ = writeln!(
            s,
            "- The objective check is RED. Make it pass without weakening, modifying, or deleting tests."
        );
    }
    if verifier.issues.is_empty() {
        if !verifier.accepted {
            let _ = writeln!(
                s,
                "- The verifier rejected the change (no specific issues were itemized)."
            );
        }
    } else {
        let _ = writeln!(s, "- The verifier raised these issues:");
        for issue in &verifier.issues {
            let _ = writeln!(s, "  - {issue}");
        }
    }
    let _ = writeln!(
        s,
        "Use the current working tree (your prior edits are still applied) to find the exact code to fix or narrow. Do not keep verifier-rejected behavior just because tests are green."
    );
    s.push_str("\nMandatory repair checklist:\n");
    s.push_str("- Make the objective check pass first; do not stop on a red check.\n");
    s.push_str(
        "- For every verifier finding above, change the code until that exact defect is gone.\n",
    );
    s.push_str("- If a finding names a stale symbol, wrong receiver, or missed call site, search intended files and fix every occurrence.\n");
    s.push_str("- If the task threads options or a new argument, audit wrappers and cache paths for stale or mixed-mode results.\n");
    s.push_str(
        "- Re-run the exact failing check after edits and inspect any remaining failure before stopping.\n",
    );
    truncate_on_boundary(&mut s, MAX_SUMMARY_CHARS);
    s
}

fn exec_retry_context(
    repair: &str,
    diff: &str,
    edited_paths: &[String],
    check: Option<(&str, &CheckObservation)>,
) -> String {
    let mut paths = if edited_paths.is_empty() {
        "(none reported)".to_string()
    } else {
        edited_paths
            .iter()
            .map(|path| format!("- {path}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    truncate_on_boundary(&mut paths, EXEC_PRIOR_EDITED_PATHS_BYTES);

    let mut diff = if diff.trim().is_empty() {
        "(no scoped diff available)".to_string()
    } else {
        diff.to_string()
    };
    truncate_on_boundary(&mut diff, EXEC_PRIOR_DIFF_BYTES);

    let check = match check {
        Some((command, observation)) if !observation.green => {
            let mut output_tail = observation.output_tail.clone();
            truncate_to_tail_on_boundary(&mut output_tail, CHECK_OUTPUT_TAIL_BYTES);
            format!(
                "Latest check `{command}`: FAIL\nLatest failing output (bounded tail):\n{output_tail}"
            )
        }
        Some((command, _)) => format!("Latest check `{command}`: PASS"),
        None => "No objective check command was configured for this turn.".to_string(),
    };

    format!(
        "{repair}\n\nPrior attempt edited paths (bounded):\n{paths}\n\n\
         {check}\n\nPrior attempt scoped diff (bounded):\n{diff}"
    )
}

fn verification_outcome_note(
    scope: &str,
    decision: DeepDecision,
    attempt: u32,
    max_attempts: u32,
    objective_ok: bool,
    verifier: &VerifierVerdict,
) -> String {
    let objective = if objective_ok {
        "objective ok"
    } else {
        "objective red"
    };
    // Codex-style work citation: show WHAT the verifier checked next to its
    // verdict, so one glance at the note replaces a second verification round.
    let evidence = verifier
        .evidence
        .as_deref()
        .map(|cited| format!(" · checked: {cited}"))
        .unwrap_or_default();
    let verifier = verifier_display_summary(verifier);
    match decision {
        DeepDecision::Accept => {
            format!("{scope}: accepted — verification passed ({objective}; {verifier}){evidence}")
        }
        DeepDecision::Retry => format!(
            "{scope}: retrying — {verifier} ({objective}; attempt {attempt}/{max_attempts}){evidence}"
        ),
        DeepDecision::GiveUp => {
            format!("{scope}: stopped — out of attempts ({objective}; {verifier}){evidence}")
        }
    }
}

fn verifier_display_summary(verifier: &VerifierVerdict) -> String {
    let mode = verifier_mode_label(verifier.parse);
    if verifier.accepted {
        return format!("{mode} accepted");
    }
    if !verifier.issues.is_empty() {
        return format!("{mode} found {}", issue_count_label(verifier.issues.len()));
    }
    match verifier.parse {
        VerifierParse::Json | VerifierParse::Salvaged => format!("{mode} rejected"),
        VerifierParse::Empty => "verifier returned no output".to_string(),
        VerifierParse::Unparseable => "verifier returned no usable verdict".to_string(),
        VerifierParse::Timeout => "verifier timed out".to_string(),
    }
}

const fn verifier_mode_label(parse: VerifierParse) -> &'static str {
    match parse {
        VerifierParse::Json => "strict verifier",
        VerifierParse::Salvaged => "salvaged verifier",
        VerifierParse::Empty | VerifierParse::Unparseable | VerifierParse::Timeout => "verifier",
    }
}

fn issue_count_label(count: usize) -> String {
    if count == 1 {
        "1 issue".to_string()
    } else {
        format!("{count} issues")
    }
}

/// The conservative verdict for a VERIFY sub-turn that did not produce a usable
/// verdict because the sub-turn itself failed (a transient streaming error). A
/// non-accept tagged `Timeout` so [`fold_verification_attempt`] retries (or
/// gives up at the cap) rather than the `?` it used to take throwing away the
/// EXEC edits that are already applied to the work tree. `Timeout` keeps the
/// display honest ("verifier timed out") and never gate-accepts.
fn verify_leg_failed_verdict() -> VerifierVerdict {
    VerifierVerdict {
        accepted: false,
        issues: Vec::new(),
        parse: VerifierParse::Timeout,
        evidence: None,
    }
}

/// Whether a completed turn actually edited the workspace — true when any
/// non-error tool result came from a write-class tool. Reactive verification
/// only engages when this holds, so chat/analysis turns are never taxed.
fn made_edits(summary: &TurnSummary) -> bool {
    summary.tool_results.iter().any(|message| {
        message.blocks.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolResult { tool_name, is_error, .. }
                    if !is_error && super::is_edit_or_write_tool(tool_name)
            )
        })
    })
}

fn task_with_retry_context(task: &str, retry: Option<&str>) -> String {
    match retry {
        Some(retry) if !retry.trim().is_empty() => {
            format!("{task}\n\nLatest repair/update context:\n{retry}")
        }
        _ => task.to_string(),
    }
}

/// The reactive retry prompt: restate the failure repair contract and the
/// current request context so the next attempt fixes the rejected change.
fn reactive_retry_prompt(task: &str, repair: &str) -> String {
    format!(
        "[auto:RETRY] Your previous change did not pass verification.\n\n{repair}\n\n\
         Fix every issue above before stopping; treat the objective check and verifier findings as blocking. Current request context:\n{task}\n\n\
         Edit only what the fix requires; do not weaken, modify, or delete tests to force a pass."
    )
}

/// The read-only `bash` inspection commands permitted during a write-capable
/// session's PLAN/VERIFY phases (see `deep_subturn`). Single owner of the
/// allowlist. Each entry is an OpenCode-style `tool(<glob>)` allow rule matched
/// against the tool's permission subject — `bash(<glob>)` against the command,
/// and `Cargo(<verb>)` against the typed `action` so the shell-free `Cargo`
/// tool gets the same read-only inspection relaxation as `bash "cargo …"`.
/// Restricted to inspection verbs (build, status, search, read); destructive
/// `bash` outside these stays gated by `bash`'s `DangerFullAccess` requirement.
/// A glob `*` can still span shell chaining, so this is a pragmatic relaxation
/// for a user-initiated full-access session's read-only phases, not a sandbox.
#[must_use]
pub fn read_only_bash_allow_rules() -> &'static [&'static str] {
    &[
        "bash(cargo check*)",
        "bash(cargo test*)",
        "bash(cargo clippy*)",
        "bash(cargo fmt*)",
        "bash(cargo build*)",
        "bash(cargo metadata*)",
        "bash(git status*)",
        "bash(git diff*)",
        "bash(git log*)",
        "bash(git show*)",
        "bash(git branch*)",
        "bash(pwd)",
        "bash(diff *)",
        "bash(ls *)",
        "bash(ls)",
        "bash(cat *)",
        "bash(rg *)",
        "bash(grep *)",
        "bash(find *)",
        "bash(head *)",
        "bash(tail *)",
        "bash(wc *)",
        "bash(echo *)",
        // Typed-action equivalents of the read-only `cargo` verbs above. `Cargo`
        // requires WorkspaceWrite (it writes `target/`), so under a downgraded
        // ReadOnly phase it is denied unless explicitly allowed here — the same
        // relaxation the `bash(cargo …)` rules grant the shell form. Subjects
        // are the discrete `action` verb (see `extract_permission_subject`), so
        // each rule names one verb. Inspection verbs only: `run`/`build` (which
        // execute arbitrary or heavier writes) stay gated. `Git` is already
        // ReadOnly, so it needs no scoped grant.
        "Cargo(check)",
        "Cargo(test)",
        "Cargo(clippy)",
        "Cargo(fmt)",
    ]
}

struct DeepSubturnPermissionGuard<'a, C, T> {
    runtime: &'a mut ConversationRuntime<C, T>,
    saved_mode: PermissionMode,
    bash_grant: Option<TemporaryAllowGrant>,
    /// Prior conversation messages while a cross-model VERIFY or swapped EXEC
    /// runs on an isolated packet. Drop appends the leg's messages back so the
    /// existing parsing and rendering seams remain unchanged.
    saved_isolated_messages: Option<Arc<Vec<ConversationMessage>>>,
    /// `Some(prior)` when this sub-turn swapped a leg client (verify or
    /// implementer) into `async_api_client`; Drop restores `prior`, so a
    /// cancelled or errored leg can never leak its client into a later leg
    /// or turn.
    #[allow(
        clippy::option_option,
        reason = "tri-state: None = no swap performed, Some(prior) = restore prior (which may itself be None)"
    )]
    saved_async_client: Option<Option<Arc<dyn AsyncApiClient>>>,
    /// Which leg flag the swap set, cleared by Drop in lockstep with the client
    /// restore.
    swapped: Option<SubturnClient>,
    /// `true` for a native-client Architect EXEC leg; cleared on Drop even
    /// though no client swap occurred.
    native_exec_leg: bool,
}

impl<'a, C, T> DeepSubturnPermissionGuard<'a, C, T> {
    fn new(
        runtime: &'a mut ConversationRuntime<C, T>,
        mode: PermissionMode,
        client: SubturnClient,
    ) -> Self {
        // `begin_phase_clamp` (vs plain `set_active_mode`) records the
        // stronger base mode so a mutating-tool denial during PLAN/VERIFY
        // names the phase clamp instead of telling the model to ask the user
        // for a permission the session already has.
        let saved_mode = runtime.permission_policy.begin_phase_clamp(mode);
        // When this phase downgrades a write-capable base mode to ReadOnly (the
        // PLAN/VERIFY phases of a full-access `/goal`/`/loop` turn), grant a
        // small read-only `bash` allowlist so inspection commands (cargo / git
        // status / rg ...) are not denied with the confusing "requires
        // danger-full-access; current mode is read-only". Removed by Drop, so a
        // cancelled sub-turn cannot leak the transient allowlist.
        let bash_grant = (mode == PermissionMode::ReadOnly
            && saved_mode.satisfies(PermissionMode::WorkspaceWrite))
        .then(|| {
            runtime
                .permission_policy
                .add_temporary_allow_rules(read_only_bash_allow_rules())
        });
        // Leg-scoped client swap. Marking the leg flag makes the
        // quota-fallback override defer to the swapped client for the
        // duration of this sub-turn (restored in Drop alongside the client
        // swap, so the two stay in lockstep).
        let native_exec_leg = client == SubturnClient::NativeExec;
        if native_exec_leg {
            runtime.exec_native_leg_active = true;
        }
        let swap = match client {
            SubturnClient::Native | SubturnClient::NativeExec => None,
            SubturnClient::Plan => runtime
                .deep_plan_client
                .as_ref()
                .map(|(client, _)| client.clone()),
            // `verify_subturn` picks which ranked candidate this leg runs on
            // by setting `deep_verify_candidate_idx` first; swap that one in.
            SubturnClient::Verify => runtime
                .deep_verify_candidates
                .get(runtime.deep_verify_candidate_idx)
                .map(|(client, _)| client.clone()),
            SubturnClient::Implementer => runtime
                .exec_contract
                .as_ref()
                .and_then(|contract| contract.impl_client.clone()),
        };
        let saved_isolated_messages = (client == SubturnClient::Verify
            || (client == SubturnClient::Implementer && swap.is_some()))
        .then(|| std::mem::replace(&mut runtime.session.messages, Arc::new(Vec::new())));
        let swapped = swap.is_some().then_some(client);
        let saved_async_client = swap.map(|client_arc| {
            match client {
                SubturnClient::Plan => runtime.deep_plan_leg_active = true,
                SubturnClient::Verify => runtime.deep_verify_leg_active = true,
                SubturnClient::Implementer => runtime.exec_impl_leg_active = true,
                SubturnClient::Native | SubturnClient::NativeExec => {}
            }
            runtime.async_api_client.replace(client_arc)
        });
        Self {
            runtime,
            saved_mode,
            bash_grant,
            saved_isolated_messages,
            saved_async_client,
            swapped,
            native_exec_leg,
        }
    }
}

impl<C, T> DeepSubturnPermissionGuard<'_, C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    async fn run(
        self,
        prompt: String,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        self.runtime
            .run_internal_subturn_streaming_with_images(prompt, images, render_tx, prompter)
            .await
    }
}

impl<C, T> Drop for DeepSubturnPermissionGuard<'_, C, T> {
    fn drop(&mut self) {
        if let Some(mut prior) = self.saved_isolated_messages.take() {
            let leg_messages =
                std::mem::replace(&mut self.runtime.session.messages, Arc::new(Vec::new()));
            Arc::make_mut(&mut prior).extend(leg_messages.iter().cloned());
            self.runtime.session.messages = prior;
        }
        if let Some(grant) = self.bash_grant.take() {
            self.runtime
                .permission_policy
                .remove_temporary_allow_rules(grant);
        }
        if let Some(prior) = self.saved_async_client.take() {
            self.runtime.async_api_client = prior;
            match self.swapped {
                Some(SubturnClient::Plan) => self.runtime.deep_plan_leg_active = false,
                Some(SubturnClient::Verify) => self.runtime.deep_verify_leg_active = false,
                Some(SubturnClient::Implementer) => self.runtime.exec_impl_leg_active = false,
                Some(SubturnClient::Native | SubturnClient::NativeExec) | None => {}
            }
        }
        if self.native_exec_leg {
            self.runtime.exec_native_leg_active = false;
        }
        self.runtime
            .permission_policy
            .end_phase_clamp(self.saved_mode);
    }
}

/// The PLAN-phase prompt. Forces a structured plan whose four headers match
/// [`decision_core::deep_lane::REQUIRED_PLAN_SECTIONS`], so [`validate_plan`]
/// can confirm it before any edit is allowed.
fn plan_prompt(task: &str, baseline: Option<&str>, missing: &[String]) -> String {
    let mut s = String::from(
        "[deep:PLAN] You are in the PLANNING phase of a deliberate change. Do NOT edit any files \
         yet — use read-only tools (read, grep, list) to inspect the repository, then write a \
         concrete plan. You MUST do this planning YOURSELF, inline. Do NOT spawn sub-agents, \
         delegate, or call Agent, SpawnMultiAgent, Workflow, or SendMessage.\n\n",
    );
    let _ = write!(s, "Task:\n{task}\n\n");
    if let Some(baseline) = baseline {
        let _ = write!(s, "{baseline}\n\n");
    }
    if !missing.is_empty() {
        let _ = write!(
            s,
            "Your previous plan had missing, empty, or placeholder-only required sections: {}. Add concrete, non-placeholder content for each one.\n\n",
            missing.join(", ")
        );
    }
    s.push_str(
        "Produce a short markdown plan with EXACTLY these four section headers, in order. Each section must contain concrete, non-placeholder content; empty/TODO/TBD/N/A/none-only sections are invalid.\n\n\
         ## Target files\n\
         For each file you will change, say what changes. Treat this as a contract across files: a \
         field/type/signature introduced in one file must be threaded through every file and test \
         that consumes it.\n\n\
         ## Invariants\n\
         Behavior that must NOT change; public APIs/signatures to preserve.\n\n\
         ## Expected tests\n\
         Which tests must pass — and any test you must NOT modify or delete.\n\n\
         ## Risks\n\
         Edge cases, hidden invariants, and failure modes to watch.\n\n\
         Output ONLY the plan. No code, no edits.\n",
    );
    s
}

/// The IMPLEMENT-phase prompt, carrying the validated plan and (on a retry) the
/// failure repair contract.
fn exec_prompt(task: &str, plan: &str, retry: Option<&str>) -> String {
    let mut s = format!(
        "[deep:EXEC] You are in the IMPLEMENTATION phase. Apply the change now, following the \
         plan.\n\n\
         Task:\n{task}\n\n\
         Plan (from the planning phase):\n{plan}\n"
    );
    if let Some(extra) = retry {
        let _ = write!(s, "\n{extra}\n");
        s.push_str(
            "\nRetry rules:\n\
             - Treat every failing check line and every verifier finding as blocking; do not stop while any remains true.\n\
             - If an Immediate mechanical edits section is present, apply those exact edits first, then rerun the failing check before broader rewrites.\n\
             - If repair hints list exact receiver replacements, apply those replacements unless the candidate is truly not in scope.\n\
             - Search intended files for stale symbols, wrong receiver/type names, and missed call sites when the task renames or threads an API.\n\
             - If the task threads options or a new argument, audit wrappers and cache paths for stale or mixed-mode results.\n",
        );
    }
    s.push_str(
        "\nRules:\n\
         - Edit only the files the plan targets.\n\
         - Preserve call receivers during renames: `thing.oldName(...)` should become `thing.newName(...)`, not `TypeName.newName(...)`, unless the task explicitly asks for a static/type call.\n\
         - Before stopping, any new identifier used as a call receiver must be imported, defined, or passed in that file.\n\
         - Do NOT modify, weaken, or delete tests to make them pass.\n\
         - If you are FIXING A BUG: first add a test (or assertion) that REPRODUCES it — one that FAILS on the current, unfixed code — then make it pass. A fix with no failing-first reproduction is how a plausible-but-wrong fix slips through.\n\
         - If a recent change already touched what you are now being asked to fix again, treat that prior change as SUSPECT, not ground truth — reproduce the symptom and confirm the real root cause before re-patching.\n\
         - Do NOT leave stray or scratch files in the repository.\n",
    );
    s
}

/// The VERIFY-phase prompt examples consumed by [`parse_lens_verifier`].
const VERIFY_JSON_ACCEPT_EXAMPLE: &str = r#"{"spec": true, "regression": true, "security": true, "issues": [], "evidence": "read diff + both call sites; ran `cargo test -p core` (ok, 42 passed)"}"#;
const VERIFY_JSON_REJECT_EXAMPLE: &str = r#"{"spec": false, "regression": true, "security": true, "issues": ["specific problem"], "evidence": "read diff; ran `cargo test -p core` (2 failed: parse_roundtrip)"}"#;
const VERIFY_SCALAR_ACCEPT_EXAMPLE: &str = r#"{"accepted": true, "issues": [], "evidence": "read the scoped diff and checked every task requirement"}"#;
const VERIFY_SCALAR_REJECT_EXAMPLE: &str = r#"{"accepted": false, "issues": ["specific unmet task requirement"], "evidence": "read the scoped diff and found the missing requirement"}"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerifyLensMode {
    SpecOnly,
    Full,
}

fn latest_assistant_text(messages: &[ConversationMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::Assistant)
        .map(|message| {
            let mut out = String::new();
            for block in &message.blocks {
                if let ContentBlock::Text { text } = block {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
            }
            out
        })
        .unwrap_or_default()
}

fn verify_prompt(
    task: &str,
    diff: &str,
    check: Option<(&str, &CheckObservation)>,
    edited_paths: &[String],
    assistant_claim: &str,
    lens_mode: VerifyLensMode,
) -> String {
    let objective = match check {
        Some((cmd, observation)) => format!(
            "Objective check `{cmd}`: {}\nLatest check output (bounded tail):\n{}",
            if observation.green { "PASS" } else { "FAIL" },
            observation.output_tail
        ),
        None => "No objective check command was configured for this turn.".to_string(),
    };
    let mut paths = if edited_paths.is_empty() {
        "(none reported)".to_string()
    } else {
        edited_paths
            .iter()
            .map(|path| format!("- {path}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    truncate_on_boundary(&mut paths, VERIFY_EDITED_PATHS_BYTES);
    let mut assistant_claim = if assistant_claim.trim().is_empty() {
        "(no final assistant claim was produced)".to_string()
    } else {
        assistant_claim.to_string()
    };
    truncate_on_boundary(&mut assistant_claim, VERIFY_ASSISTANT_CLAIM_BYTES);
    if lens_mode == VerifyLensMode::SpecOnly {
        return format!(
            "[deep:VERIFY] You are ONE strict, adversarial verifier. You MUST judge ONLY the \
             spec/task-compliance dimension YOURSELF, inline, in this sub-turn. Do NOT spawn \
             sub-agents, delegate, or call Agent, SpawnMultiAgent, Workflow, or SendMessage. You \
             may use read-only tools to inspect further — prefer read_file/grep; if you run bash, \
             use ONE simple command from the current directory (no `cd`, no `&&`/`;` chaining — \
             compound commands are denied in this read-only phase).\n\n\
             Task:\n{task}\n\n\
             {objective}\n\n\
             Paths changed this attempt:\n{paths}\n\n\
             Assistant's final claim for this attempt (bounded):\n{assistant_claim}\n\n\
             Diff (scoped git diff, bounded):\n{diff}\n\n\
             Does the change FULLY and CORRECTLY satisfy the task, including every requirement, \
             correct error handling, and edge cases? If the change FIXES A BUG, prefer a test that \
             fails on the unfixed code and passes now; reject a bug fix that lacks such a test ONLY \
             where a failing-first reproduction is feasible. If reproduction is genuinely \
             impractical (a timing/heisenbug fix, a TUI/rendering glitch, a config or dependency \
             bump, or a change whose only feasible repro is manual), accept when the diff documents \
             WHY reproduction is impractical AND the objective check is green. If a checked-in test \
             explicitly requires behavior not spelled out in the task, treat that test as part of \
             the contract. Do not emit a partial lens object.\n\n\
             Respond with ONLY a single-line JSON object and NOTHING else — no prose, no markdown \
             code fences, no extra keys, and no text before or after it. Use exactly these scalar \
             verdict keys (accepted, issues, evidence). List each concrete unmet task requirement \
             in issues, and cite what you actually inspected in evidence:\n\
             {VERIFY_SCALAR_ACCEPT_EXAMPLE}\n\
             or\n\
             {VERIFY_SCALAR_REJECT_EXAMPLE}\n"
        );
    }
    format!(
        "[deep:VERIFY] You are ONE strict, adversarial verifier. You MUST judge all three lenses \
         YOURSELF, inline, in this sub-turn, assessing each lens independently. Do NOT spawn \
         sub-agents, delegate, or call Agent, SpawnMultiAgent, Workflow, or SendMessage. The whole change is \
         REJECTED if ANY lens finds a real defect in its dimension (a single credible objection \
         blocks acceptance). You may use read-only tools to inspect further — prefer \
         read_file/grep; if you run bash, use ONE simple command from the current directory (no \
         `cd`, no `&&`/`;` chaining — compound commands are denied in this read-only phase).\n\n\
         Task:\n{task}\n\n\
         {objective}\n\n\
         Paths changed this attempt:\n{paths}\n\n\
         Assistant's final claim for this attempt (bounded):\n{assistant_claim}\n\n\
         Diff (scoped git diff, bounded):\n{diff}\n\n\
         Judge each lens (true = accept, false = reject):\n\
         - \"spec\": does the change FULLY and CORRECTLY satisfy the task? Every requirement \
         implemented; correct error handling and edge cases. If the change FIXES A BUG, prefer a \
         test that fails on the unfixed code and passes now; reject a bug fix that lacks such a test \
         ONLY where a failing-first reproduction is feasible — that is exactly how a \
         plausible-but-wrong fix slips through. If a reproduction is genuinely impractical (a \
         timing/heisenbug fix, a TUI/rendering glitch, a config or dependency bump, a change whose \
         only feasible repro is manual), accept when the diff documents WHY a reproduction is \
         impractical AND the objective check is green. If a checked-in test explicitly requires \
         behavior not spelled out in the task, treat that test as part of the contract instead of \
         rejecting the behavior as scope creep.\n\
         - \"regression\": does the change avoid breaking anything? Only the intended files changed \
         (no unrelated edits, no leaked artifacts); NO test modified or deleted to force a pass; no \
         existing behavior silently broken.\n\
         - \"security\": does the change avoid introducing a vulnerability (injection, unsafe input \
         handling, secret/credential leak, path traversal) in THIS diff? If the change is not \
         security-relevant, accept this lens (true).\n\n\
         Respond with ONLY a single-line JSON object and NOTHING else — no prose, no markdown code \
         fences, no extra keys, and no text before or after it. Use exactly these keys (spec, \
         regression, security, issues, evidence). List every rejecting lens's concrete problem in \
         issues, and cite your work in evidence — ONE line naming what you actually inspected and \
         every command you ran with its observed result. A verdict without evidence is not \
         auditable and wastes the reviewer's trust:\n\
         {VERIFY_JSON_ACCEPT_EXAMPLE}\n\
         or\n\
         {VERIFY_JSON_REJECT_EXAMPLE}\n"
    )
}

/// Accumulates per-phase [`TurnSummary`]s into one combined summary for the
/// whole deep turn (iterations summed, message vectors concatenated, usage
/// folded field-by-field).
#[derive(Default)]
struct DeepSummaryAcc {
    assistant_messages: Vec<ConversationMessage>,
    tool_results: Vec<ConversationMessage>,
    prompt_cache_events: Vec<PromptCacheEvent>,
    iterations: usize,
    usage: TokenUsage,
    turn_output_tokens: u32,
    auto_compaction: Option<AutoCompactionEvent>,
    microcompact: Option<crate::MicrocompactEvent>,
    budget_exhausted: Option<BudgetExhausted>,
}

impl DeepSummaryAcc {
    fn fold(&mut self, summary: TurnSummary) {
        self.assistant_messages.extend(summary.assistant_messages);
        self.tool_results.extend(summary.tool_results);
        self.prompt_cache_events.extend(summary.prompt_cache_events);
        self.iterations += summary.iterations;
        // Each sub-turn's usage is the *cumulative* session usage at that point
        // (both `TurnSummary.usage` assignment sites use `cumulative_usage()`), so
        // the deep turn's usage is the LATEST sub-turn's cumulative — NOT the sum
        // of the snapshots. Summing multiplied the total by the sub-turn count and,
        // downstream, inflated the goal token budget and tripped auto-compaction
        // early. Sub-turns run in sequence so the last fold carries the highest
        // cumulative.
        self.usage = summary.usage;
        // `turn_output_tokens`, by contrast, is each sub-turn's OWN in-turn delta,
        // so the deep turn's output is their SUM (the goal budget charges the whole
        // multi-sub-turn deep turn, not just the last leg).
        self.turn_output_tokens = self
            .turn_output_tokens
            .saturating_add(summary.turn_output_tokens);
        if summary.auto_compaction.is_some() {
            self.auto_compaction = summary.auto_compaction;
        }
        if summary.microcompact.is_some() {
            self.microcompact = summary.microcompact;
        }
        // A budget stop in ANY leg marks the whole deep turn budget-stopped:
        // dropping it here silently disarmed every downstream consumer (the
        // `/loop` budget-pause and the grind-escalation streak) whenever the
        // deep gate wrapped the turn. Sub-turns run in sequence, so a later
        // leg's stop simply overwrites an earlier one.
        if summary.budget_exhausted.is_some() {
            self.budget_exhausted = summary.budget_exhausted;
        }
    }

    fn into_summary(self) -> TurnSummary {
        TurnSummary {
            assistant_messages: self.assistant_messages,
            tool_results: self.tool_results,
            prompt_cache_events: self.prompt_cache_events,
            iterations: self.iterations,
            usage: self.usage,
            turn_output_tokens: self.turn_output_tokens,
            auto_compaction: self.auto_compaction,
            microcompact: self.microcompact,
            // The deep methods return `(TurnSummary, DeepOutcome)` separately;
            // the verdict is stamped onto the summary at the wrapper seam
            // (`run_turn_streaming_maybe_deep`), so the accumulator leaves it
            // `None` here.
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted: self.budget_exhausted,
        }
    }
}

/// The goal-facing verification scalar exported on `summary.deep_verification`.
///
/// This must mean "the deep loop as a whole accepted", NOT merely "the verifier
/// gate accepted". `gate_accepted` can be true on an objective-RED turn (a strict
/// JSON accept is trusted for the deep loop's own retry/stall policy), but a goal
/// must never be marked succeeded while its objective check is red. `decision` is
/// `Accept` only when `objective_ok && gate_accepted` (see `decide_with_progress`),
/// so gating the export on it is the correct, conservative goal-facing signal.
fn goal_facing_accept(folded: &VerificationAttempt) -> bool {
    folded.decision == DeepDecision::Accept
}

/// Emit a non-critical deep-phase progress note into the render stream. A closed
/// channel just means the turn is unwinding, so the error is ignored.
async fn deep_note(
    render_tx: &mpsc::Sender<RenderBlock>,
    ids: &BlockIdGen,
    text: impl Into<String>,
) {
    let _ = render_tx
        .send(RenderBlock::System {
            id: ids.next(),
            level: SystemLevel::Info,
            text: text.into(),
        })
        .await;
}

impl<C, T> ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    /// Install (or clear) the deep-lane gate. With a config set, the host should
    /// drive turns through [`Self::run_deep_turn_streaming`] instead of
    /// [`Self::run_turn_streaming_with_images`].
    pub fn set_deep_gate(&mut self, config: Option<DeepGateConfig>) {
        self.deep_gate = config;
    }

    /// The installed deep-lane config, if any.
    #[must_use]
    pub fn deep_gate(&self) -> Option<&DeepGateConfig> {
        self.deep_gate.as_ref()
    }

    /// Set the stable logical workspace directory used to root durable external
    /// traces (`.zo/dream/`, `.zo/turns/`). Hosts whose process cwd can
    /// diverge from the workspace — the interactive TUI (where `EnterWorktree`
    /// chdirs) and `zo serve` (many sessions, one process) — must call this so
    /// trace producers and the auto-dream consumer agree on one `.zo/`.
    pub fn set_workspace_cwd(&mut self, cwd: std::path::PathBuf) {
        self.workspace_cwd = Some(cwd);
    }

    /// Resolve the directory to root durable traces at: the configured stable
    /// workspace if set, else the live process cwd. Centralizes the producer
    /// rule so every trace site (deep-gate accept, turn completion) agrees.
    pub(crate) fn trace_cwd(&self) -> Option<std::path::PathBuf> {
        if let Some(root) = std::env::var_os("ZO_TRACE_ROOT") {
            return Some(std::path::PathBuf::from(root));
        }
        match &self.workspace_cwd {
            Some(cwd) => Some(cwd.clone()),
            None => std::env::current_dir().ok(),
        }
    }

    /// Record a green-verified acceptance as a Dreamer candidate lesson.
    ///
    /// Called from both accept paths (reactive and plan-first) when a change is
    /// accepted *and* the objective check ran green. Best-effort and silent: it
    /// appends one candidate to `.zo/dream/` rooted at the session's stable
    /// workspace ([`Self::trace_cwd`]), which a later between-sessions
    /// [`crate::maybe_auto_dream`] pass may promote once the same project check
    /// has been green-verified across enough distinct sessions. A turn is never
    /// failed or slowed by a recording problem.
    fn record_verified_accept(&self, objective_ok: bool) {
        if !objective_ok {
            return;
        }
        let Some(check) = self
            .deep_gate
            .as_ref()
            .and_then(|c| c.check_command.clone())
        else {
            return;
        };
        if let Some(cwd) = self.trace_cwd() {
            let _ = crate::record_verified_check(&cwd, &self.session.session_id, Some(&check));
            let _ = crate::memory::record_self_improve_pulse_if_enabled(
                self.dream_automation_enabled,
                &cwd,
                decision_core::dreamer::CandidateKind::VerifiedAccept,
                &self.session.session_id,
                "deep_gate",
                "deep gate accepted after objective check",
                &check,
                true,
            );
        }
    }

    /// Drop-in streaming entry point for hosts: routes to the deep gate when one
    /// is installed (discarding the structured [`DeepOutcome`], which is already
    /// narrated into the render stream), otherwise runs the ordinary turn. Keeps
    /// the caller's `select!`/render loop on a single `Result<TurnSummary, _>`
    /// shape regardless of mode.
    ///
    /// Honors the `TurnEnd` (Stop) hook exactly like the synchronous
    /// [`Self::run_turn`] loop: a hook returning a `followupMessage` (or the
    /// Claude Code `decision: "block"` shape) re-enters the turn with that
    /// message as the next user input, bounded by `max_stop_loops`. This used
    /// to exist only on the sync path, which made Stop-hook gates (e.g. a
    /// session-goal "keep working until done" check) dead in the interactive
    /// TUI — the one place users actually run them.
    ///
    /// # Errors
    /// Propagates any [`StreamingTurnError`] from the underlying turn.
    pub async fn run_turn_streaming_maybe_deep(
        &mut self,
        user_input: impl Into<String>,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        let mut input: String = user_input.into();
        // Follow-up rounds are text-only; the original images belong to the
        // first round (mirrors the sync Stop-loop).
        let mut images = images;
        let mut loop_count = 0usize;
        // Baseline for the WHOLE turn's output delta across Stop-loop legs — each
        // leg's summary carries only its own delta, but a multi-leg (TurnEnd
        // followup) turn must charge the goal budget the sum. Cumulative is
        // monotonic within this runtime instance, so re-derive it at return.
        let turn_base_output = self.usage_tracker.cumulative_usage().output_tokens;
        let result = loop {
            let round_images = std::mem::take(&mut images);
            let deep_mode = self.deep_gate.as_ref().map(|cfg| cfg.mode);
            if deep_mode.is_some() {
                if let Err(error) = self.run_user_prompt_submit_for_streaming_user_entry(&input) {
                    break Err(error);
                }
            }
            // A per-turn Architect contract can promote a Reactive gate to the
            // plan-first driver for THIS turn only (complex implementation
            // turns get a read-only PLAN by the reserved model before the
            // implementer edits); the installed gate config itself is untouched.
            let plan_first_contract = self
                .exec_contract
                .as_ref()
                .is_some_and(|contract| contract.plan_first);
            let mut summary = match deep_mode {
                Some(DeepMode::Reactive) if !plan_first_contract => {
                    let (mut summary, outcome) = match self
                        .run_auto_turn_streaming(
                            input.clone(),
                            round_images,
                            render_tx.clone(),
                            Arc::clone(&prompter),
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(error) => break Err(error),
                    };
                    // Surface the adversarial verifier's verdict to the host so
                    // the goal controller can gate completion on it.
                    summary.deep_verification = outcome.verification;
                    summary.verification_issues = outcome.issues;
                    summary.deep_verifier_parse = outcome.verifier_parse;
                    summary.deep_verifier_model = outcome.verifier_model;
                    summary
                }
                Some(DeepMode::PlanFirst | DeepMode::Reactive) => {
                    let (mut summary, outcome) = match self
                        .run_deep_turn_streaming(
                            input.clone(),
                            round_images,
                            render_tx.clone(),
                            Arc::clone(&prompter),
                        )
                        .await
                    {
                        Ok(value) => value,
                        Err(error) => break Err(error),
                    };
                    summary.deep_verification = outcome.verification;
                    summary.verification_issues = outcome.issues;
                    summary.deep_verifier_parse = outcome.verifier_parse;
                    summary.deep_verifier_model = outcome.verifier_model;
                    summary
                }
                None => {
                    match self
                        .run_turn_streaming_with_images(
                            input.clone(),
                            round_images,
                            render_tx.clone(),
                            Arc::clone(&prompter),
                        )
                        .await
                    {
                        Ok(summary) => summary,
                        Err(error) => break Err(error),
                    }
                }
            };
            // TurnEnd (Stop) hook — same contract and bound as the sync loop
            // (`run_turn`). Hook commands run synchronously with the shared 5s
            // timeout; with no TurnEnd rules configured this is a no-op, so the
            // render loop only ever pauses when the user opted into a gate.
            let files_changed = changed_files_snapshot_async().await;
            let context = build_turn_end_hook_context(
                &summary,
                loop_count,
                &files_changed,
                self.session.session_goal.as_deref(),
            );
            let outcome = self.run_lifecycle_hook(HookEvent::TurnEnd, &context);
            match outcome.followup().map(str::to_owned) {
                Some(followup) if loop_count < self.max_stop_loops => {
                    loop_count += 1;
                    input = followup;
                }
                _ => {
                    summary.turn_output_tokens = self
                        .usage_tracker
                        .cumulative_usage()
                        .output_tokens
                        .saturating_sub(turn_base_output);
                    break Ok(summary);
                }
            }
        };
        self.settle_team_inbox_turn_for_result(&result);
        result
    }

    /// Reactive auto-verify turn — the default [`DeepMode::Reactive`]. Runs the
    /// user's request as an ordinary turn (full tools, no read-only phase), then
    /// **only if it edited files** selects proportional VERIFY depth for the diff
    /// and retries verified failures, bounded by `max_attempts`. A chat/analysis
    /// turn that changes nothing returns immediately with zero verification
    /// overhead and no permission friction.
    ///
    /// This is the Reactive phase driver behind
    /// [`Self::run_turn_streaming_maybe_deep`]: it drives internal sub-turns
    /// only and deliberately skips the per-turn host lifecycle — the
    /// `UserPromptSubmit` hook and the `TeamInbox` digest injection/settle run
    /// in that outer loop. Calling this directly bypasses those policies.
    ///
    /// # Errors
    /// Propagates any [`StreamingTurnError`] from a sub-turn.
    #[allow(clippy::too_many_lines)]
    pub async fn run_auto_turn_streaming(
        &mut self,
        user_input: impl Into<String>,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<(TurnSummary, DeepOutcome), StreamingTurnError> {
        let cfg = self.deep_gate.clone().unwrap_or_default();
        let max = cfg.max_attempts.max(1);
        let task = user_input.into();
        let ids = BlockIdGen::default();
        let mut acc = DeepSummaryAcc::default();
        let mut pending_images = Some(images);
        let mut extra: Option<String> = None;
        let mut exec_retry: Option<String> = None;
        let mut decision = DeepDecision::Accept;
        let mut attempts = 0u32;
        // The verifier's semantic verdict for goal-completion gating. `None`
        // until an edit-making turn is actually verified, so a no-edit turn or
        // proportional trivial-change skip reports no semantic judgment rather
        // than a spurious verifier accept.
        let mut verification: Option<bool> = None;
        // The final attempt's verifier issues, surfaced on the outcome so the
        // goal-level repair prompt can name the concrete defects to fix.
        let mut verifier_issues: Vec<String> = Vec::new();
        // Phase 4 verdict-channel seam: the final attempt's raw verifier parse
        // confidence and the verifier model, surfaced on the outcome alongside
        // `verification`/`verifier_issues`. See `DeepOutcome::verifier_parse`.
        let mut verifier_parse: Option<VerifierParse> = None;
        let mut verifier_model: Option<String> = None;
        // The previous attempt's verifier issues, for the ALP §3 "no more
        // progress" stop condition: if a retry fails for the same reason, give
        // up early instead of burning the remaining attempt budget.
        let mut prev_issues: Vec<String> = Vec::new();
        let mut verify_depth_floor = VerifyDepth::Skip;
        // Do not run the objective command before the first model stream. In the
        // interactive default this command can be a heavyweight project test (for
        // example `cargo test`), and running it here delays the first token even
        // for a no-edit chat turn. Reactive mode verifies only after an edit is
        // observed below; no-edit turns keep the zero-overhead contract.

        for attempt in 1..=max {
            attempts = attempt;
            // Attempt 1 is the user's request verbatim — behaves exactly like an
            // ordinary turn. A retry restates it with the failure repair contract.
            let prompt = match exec_retry.as_deref() {
                None => task.clone(),
                Some(repair) => reactive_retry_prompt(&task, repair),
            };
            let baseline_files = changed_files_snapshot_async().await;
            let phase_images = pending_images.take().unwrap_or_default();
            // Runtime effort escalation, mirroring the plan-first EXEC leg
            // (`run_deep_turn_streaming`): the first attempt runs at the
            // configured effort, but a retry means that effort did not solve the
            // task, so power up every retry to at least `Xhigh` (a floor, never a
            // downgrade). Cleared immediately after the EXEC sub-turn — before `?`
            // and before VERIFY — so the override never leaks into the read-only
            // verify turn or a later turn on error.
            if attempt > 1 {
                self.set_effort_override(Some(super::ESCALATION_EFFORT_BUDGET));
                deep_note(
                    &render_tx,
                    &ids,
                    "auto: escalating reasoning effort (xhigh) for retry…",
                )
                .await;
            }
            if let Some(note) = self.exec_leg_note(attempt) {
                deep_note(&render_tx, &ids, note).await;
            }
            if self.exec_swap_enabled() && attempt > ARCHITECT_IMPL_ATTEMPTS {
                // Failure escalation: the native (reserved) model implements
                // from here on, so the edit gate stands down for this turn.
                self.reserved_edit_gate = false;
            }
            // The EXEC leg keeps the caller's permission mode (a no-op
            // set/restore); the guard exists so an Architect contract can swap
            // the leg onto the implementer client — without one this is
            // byte-identical to the old direct sub-turn call.
            let base_mode = self.permission_policy.active_mode();
            let exec_result = self
                .deep_subturn(
                    prompt,
                    phase_images,
                    base_mode,
                    self.exec_leg_client(attempt),
                    &render_tx,
                    &prompter,
                )
                .await;
            self.set_effort_override(None);
            let summary = exec_result?;
            let edited = made_edits(&summary);
            let edited_paths = edited_file_paths(&summary);
            let assistant_claim = latest_assistant_text(&summary.assistant_messages);
            acc.fold(summary);

            // No edits ⇒ a question/analysis/chat turn. Done — never tax a turn
            // that changed nothing.
            if !edited {
                decision = DeepDecision::Accept;
                break;
            }

            let check_observation = match cfg.check_command.as_deref() {
                Some(cmd) => {
                    let observation = run_check_command(cmd).await;
                    deep_note(
                        &render_tx,
                        &ids,
                        format!(
                            "auto: check `{cmd}` → {}",
                            if observation.green {
                                "green ✓"
                            } else {
                                "red ✗"
                            }
                        ),
                    )
                    .await;
                    Some(observation)
                }
                None => None,
            };
            let objective_ok = check_observation.as_ref().is_none_or(|check| check.green);

            let after_files = changed_files_snapshot_async().await;
            let diff_paths = attempt_diff_paths(&baseline_files, &after_files, &edited_paths);
            let (diff, line_churn) =
                bounded_git_diff_for_paths(diff_paths.clone(), 6000).await;
            let selected_depth = verify_depth_for_band(
                self.verify_band,
                diff_paths.len(),
                line_churn,
                objective_ok,
                paths_touch_security(&diff_paths),
                paths_touch_tests(&diff_paths),
            );
            let depth = selected_depth.max(verify_depth_floor);
            verify_depth_floor = depth;
            if depth == VerifyDepth::Skip {
                deep_note(
                    &render_tx,
                    &ids,
                    "auto: trivial green change — skipping deep verify",
                )
                .await;
                decision = DeepDecision::Accept;
                break;
            }
            let verify_note = match self.deep_verify_primary_model_label() {
                Some(model) => format!(
                    "auto: verifying the change with {model} (cross-model, attempt {attempt}/{max})…"
                ),
                None => format!("auto: verifying the change (attempt {attempt}/{max})…"),
            };
            deep_note(&render_tx, &ids, verify_note).await;
            // VERIFY runs read-only just like plan-first: an adversarial verifier
            // inspects the diff but must never edit or delete files. `deep_subturn`
            // downgrades a write-capable session to ReadOnly (with the scoped
            // read-only `bash` grant) and always restores the prior mode. When a
            // cross-model verifier is installed the leg runs on it (native
            // fallback inside `verify_subturn`).
            let verify_result = self
                .verify_subturn(
                    verify_prompt(
                        &task_with_retry_context(&task, extra.as_deref()),
                        &diff,
                        cfg.check_command
                            .as_deref()
                            .zip(check_observation.as_ref()),
                        &diff_paths,
                        &assistant_claim,
                        if depth == VerifyDepth::SingleLens {
                            VerifyLensMode::SpecOnly
                        } else {
                            VerifyLensMode::Full
                        },
                    ),
                    &render_tx,
                    &ids,
                    &prompter,
                )
                .await;
            // A failed VERIFY leg (transient streaming error) must NOT throw away
            // the EXEC edits already applied this attempt. Fold a conservative
            // non-accept (Timeout) so the loop retries or gives up at the cap,
            // preserving the completed implementation in the work tree.
            let verifier = match verify_result {
                Ok(verify_summary) => {
                    acc.fold(verify_summary);
                    parse_lens_verifier(&self.last_assistant_text())
                }
                Err(_) => verify_leg_failed_verdict(),
            };
            // Keep accept/retry/stall policy in decision-core; this runtime only
            // supplies observed IO facts from the live VERIFY sub-turn. Reactive
            // mode intentionally does not run a pre-model baseline command, so an
            // objective-red post-edit check remains blocking rather than delaying
            // the first token to classify it as a pre-existing failure.
            let gating_objective_ok = objective_ok;
            let folded = fold_verification_attempt(
                attempt,
                max,
                gating_objective_ok,
                &verifier,
                &prev_issues,
            );

            // Record the goal-facing verdict for goal-completion gating: export
            // "the deep loop accepted overall" (decision == Accept), NOT the raw
            // verifier gate — an objective-red turn must never read as accepted by
            // a downstream goal that has no objective validators of its own.
            verification = Some(goal_facing_accept(&folded));
            verifier_issues = verifier.issues.clone();
            verifier_parse = Some(verifier.parse);
            verifier_model = self
                .deep_verify_succeeded_model_label()
                .map(str::to_string);
            decision = folded.decision;
            deep_note(
                &render_tx,
                &ids,
                verification_outcome_note("auto", decision, attempt, max, objective_ok, &verifier),
            )
            .await;

            match decision {
                DeepDecision::Accept | DeepDecision::GiveUp => {
                    if decision == DeepDecision::Accept {
                        self.record_verified_accept(objective_ok);
                    }
                    break;
                }
                DeepDecision::Retry => {
                    let repair = failure_summary(objective_ok, &verifier);
                    exec_retry = Some(exec_retry_context(
                        &repair,
                        &diff,
                        &diff_paths,
                        cfg.check_command
                            .as_deref()
                            .zip(check_observation.as_ref()),
                    ));
                    extra = Some(repair);
                    prev_issues = verifier.issues.clone();
                }
            }
        }

        let outcome = DeepOutcome {
            decision,
            attempts,
            plan_valid: true,
            plan_missing: Vec::new(),
            verification,
            issues: verifier_issues,
            verifier_parse,
            verifier_model,
        };
        Ok((acc.into_summary(), outcome))
    }

    /// Concatenated text of the most recent assistant message (the phase output
    /// the gate inspects). Empty when the last turn produced no assistant text.
    fn last_assistant_text(&self) -> String {
        latest_assistant_text(&self.session.messages)
    }

    /// Run one phase sub-turn under `mode`, always restoring the prior
    /// permission mode (even on error) so a `ReadOnly` PLAN/VERIFY never leaks
    /// past its phase.
    async fn deep_subturn(
        &mut self,
        prompt: String,
        images: Vec<(String, String)>,
        mode: PermissionMode,
        client: SubturnClient,
        render_tx: &mpsc::Sender<RenderBlock>,
        prompter: &Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        let guard = DeepSubturnPermissionGuard::new(self, mode, client);
        guard
            .run(prompt, images, render_tx.clone(), Arc::clone(prompter))
            .await
    }

    fn exec_swap_enabled(&self) -> bool {
        self.exec_contract
            .as_ref()
            .is_some_and(ExecContract::exec_swap_enabled)
    }

    fn native_model_is_deep_tier(&self) -> bool {
        self.context_model
            .as_deref()
            .is_some_and(|model| crate::is_deep_tier_model(model, &self.deep_tier_models))
    }

    fn plan_leg_client(&self) -> Result<SubturnClient, StreamingTurnError> {
        if self.deep_plan_client.as_ref().is_some_and(|(_, model)| {
            !self.deep_tier_only || crate::is_deep_tier_model(model, &self.deep_tier_models)
        }) {
            return Ok(SubturnClient::Plan);
        }
        if !self.deep_tier_only || self.native_model_is_deep_tier() {
            return Ok(SubturnClient::Native);
        }
        Err(StreamingTurnError::runtime(
            "architect PLAN requires an available configured deep-tier client",
        ))
    }

    /// Which client the EXEC leg of `attempt` runs on. When `smart.execSwap`
    /// did not arm for this turn, every attempt remains native while retaining
    /// the Architect EXEC edit-gate exemption. When armed, two implementer
    /// failures escalate to the native model, mirroring the router escape.
    fn exec_leg_client(&self, attempt: u32) -> SubturnClient {
        if let Some(contract) = &self.exec_contract {
            if !contract.exec_swap_enabled() {
                return SubturnClient::NativeExec;
            }
            if attempt <= ARCHITECT_IMPL_ATTEMPTS {
                return SubturnClient::Implementer;
            }
        }
        SubturnClient::Native
    }

    /// One-line narration for the EXEC leg's contract state, `None` when the
    /// leg runs native without a contract (nothing to announce).
    fn exec_leg_note(&self, attempt: u32) -> Option<String> {
        let contract = self.exec_contract.as_ref()?;
        if !contract.exec_swap_enabled() {
            return None;
        }
        let native = self.context_model.as_deref().unwrap_or("the main model");
        if attempt <= ARCHITECT_IMPL_ATTEMPTS {
            (attempt == 1).then(|| {
                format!(
                    "architect: implementing with {} — {native} stays on plan/verify",
                    contract.impl_model
                )
            })
        } else if attempt == ARCHITECT_IMPL_ATTEMPTS + 1 {
            Some(format!(
                "architect: {ARCHITECT_IMPL_ATTEMPTS} implementer attempts failed — escalating implementation to {native}"
            ))
        } else {
            None
        }
    }

    /// One VERIFY sub-turn (read-only). When cross-model verifier candidates are
    /// installed ([`Self::set_deep_verify_candidates`]), the leg walks them
    /// top-ranked first: a hard `RateLimit` skips every remaining candidate on
    /// that provider and tries the next different-provider candidate, while any
    /// other stream error advances to the next candidate. The walk is bounded
    /// by the candidate count. Exhausting an installed ranked list falls back
    /// to the native main client once unless the Architect deep-tier invariant
    /// forbids that non-deep fallback.
    async fn verify_subturn(
        &mut self,
        prompt: String,
        render_tx: &mpsc::Sender<RenderBlock>,
        ids: &BlockIdGen,
        prompter: &Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<TurnSummary, StreamingTurnError> {
        self.deep_verify_succeeded_model = None;
        let candidate_count = self.deep_verify_candidates.len();
        if candidate_count > 0 {
            let mut rate_limited_providers: Vec<api::ProviderKind> = Vec::new();
            for idx in 0..candidate_count {
                let model = self.deep_verify_candidates[idx].1.clone();
                if self.deep_tier_only
                    && !crate::is_deep_tier_model(&model, &self.deep_tier_models)
                {
                    continue;
                }
                let provider = api::detect_provider_kind(&model);
                if rate_limited_providers.contains(&provider) {
                    continue;
                }
                self.deep_verify_candidate_idx = idx;
                match self
                    .deep_subturn(
                        prompt.clone(),
                        Vec::new(),
                        PermissionMode::ReadOnly,
                        SubturnClient::Verify,
                        render_tx,
                        prompter,
                    )
                    .await
                {
                    Ok(summary) => {
                        self.deep_verify_succeeded_model = Some(model);
                        return Ok(summary);
                    }
                    Err(err) => {
                        if matches!(
                            err.provider_error_class(),
                            Some(api::ProviderErrorClass::RateLimit { .. })
                        ) {
                            rate_limited_providers.push(provider);
                            deep_note(
                                render_tx,
                                ids,
                                format!(
                                    "auto: verifier {model} rate-limited — trying the next-ranked provider…"
                                ),
                            )
                            .await;
                        } else {
                            deep_note(
                                render_tx,
                                ids,
                                format!(
                                    "auto: verifier {model} unavailable — trying the next-ranked candidate…"
                                ),
                            )
                            .await;
                        }
                    }
                }
            }
            deep_note(
                render_tx,
                ids,
                if self.deep_tier_only && !self.native_model_is_deep_tier() {
                    "auto: all deep-tier verifier candidates unavailable — non-deep native fallback disabled"
                } else {
                    "auto: all ranked verifier candidates unavailable — retrying with the main model…"
                },
            )
            .await;
        }
        if self.deep_tier_only && !self.native_model_is_deep_tier() {
            return Err(StreamingTurnError::runtime(
                "architect VERIFY requires an available configured deep-tier client",
            ));
        }
        self.deep_subturn(
            prompt,
            Vec::new(),
            PermissionMode::ReadOnly,
            SubturnClient::Native,
            render_tx,
            prompter,
        )
        .await
    }

    /// Top-ranked verifier model shown before a VERIFY attempt starts.
    fn deep_verify_primary_model_label(&self) -> Option<&str> {
        self.deep_verify_candidates
            .first()
            .map(|(_, model)| model.as_str())
    }

    /// Verifier model that actually produced the current attempt's verdict.
    fn deep_verify_succeeded_model_label(&self) -> Option<&str> {
        self.deep_verify_succeeded_model.as_deref()
    }

    /// Run a deliberate turn: PLAN (read-only, re-tried until structurally
    /// valid) → IMPLEMENT → objective check → VERIFY (read-only) → decide, with
    /// bounded retries fed the failure contract. `Accept` and `GiveUp` both end
    /// the turn honestly; `GiveUp` leaves the work tree in its last state rather
    /// than pretending success. Returns the combined summary of every phase.
    ///
    /// This is the `PlanFirst` phase driver behind
    /// [`Self::run_turn_streaming_maybe_deep`]: it drives internal sub-turns
    /// only and deliberately skips the per-turn host lifecycle — the
    /// `UserPromptSubmit` hook and the `TeamInbox` digest injection/settle run
    /// in that outer loop. Calling this directly bypasses those policies.
    ///
    /// # Errors
    /// Propagates any [`StreamingTurnError`] from a phase sub-turn.
    #[allow(clippy::too_many_lines)]
    pub async fn run_deep_turn_streaming(
        &mut self,
        user_input: impl Into<String>,
        images: Vec<(String, String)>,
        render_tx: mpsc::Sender<RenderBlock>,
        prompter: Arc<dyn AsyncPermissionPrompter>,
    ) -> Result<(TurnSummary, DeepOutcome), StreamingTurnError> {
        let cfg = self.deep_gate.clone().unwrap_or_default();
        let max = cfg.max_attempts.max(1);
        let task = user_input.into();
        let ids = BlockIdGen::default();
        let base_mode = self.permission_policy.active_mode();
        let mut acc = DeepSummaryAcc::default();
        // Fresh deep turn: forget any verifier model recorded by a previous
        // turn so the summary reports this turn's successful verifier (or none).
        self.deep_verify_succeeded_model = None;

        // Baseline: show the check's starting state to the planner (cheap, and
        // it tells the model whether it is starting red or must keep green). The
        // check itself is offloaded to a blocking thread inside `command_is_green`
        // so it never freezes the host event loop while it runs.
        let baseline = match cfg.check_command.as_deref() {
            Some(cmd) => {
                let green = command_is_green(cmd).await;
                Some(if green {
                    format!("Baseline check `{cmd}` currently PASSES; keep it green.")
                } else {
                    format!("Baseline check `{cmd}` currently FAILS; this is the red state to fix.")
                })
            }
            None => None,
        };

        // ── PLAN phase (ReadOnly): re-plan until structurally valid. ──
        deep_note(&render_tx, &ids, "deep: PLAN phase (read-only)…").await;
        let mut plan_md = String::new();
        let mut plan_verdict = PlanVerdict {
            valid: false,
            missing: Vec::new(),
        };
        let mut pending_images = Some(images);
        let mut missing: Vec<String> = Vec::new();
        let plan_client = self.plan_leg_client()?;
        for _ in 1..=max {
            let prompt = plan_prompt(&task, baseline.as_deref(), &missing);
            let phase_images = pending_images.take().unwrap_or_default();
            let summary = self
                .deep_subturn(
                    prompt,
                    phase_images,
                    PermissionMode::ReadOnly,
                    plan_client,
                    &render_tx,
                    &prompter,
                )
                .await?;
            acc.fold(summary);
            plan_md = self.last_assistant_text();
            plan_verdict = validate_plan(&plan_md);
            if plan_verdict.valid {
                deep_note(&render_tx, &ids, "deep: plan valid ✓").await;
                break;
            }
            missing = plan_verdict.missing.clone();
            deep_note(
                &render_tx,
                &ids,
                format!("deep: plan missing [{}] — re-planning", missing.join(", ")),
            )
            .await;
        }
        // If still invalid after `max` tries, proceed honestly: plan validity is
        // surfaced in the outcome, and the verifier still gates acceptance.

        // ── IMPLEMENT → check → VERIFY → decide, bounded retries. ──
        let mut extra: Option<String> = None;
        let mut exec_retry: Option<String> = None;
        let mut decision = DeepDecision::GiveUp;
        let mut attempts = 0u32;
        // Previous attempt's verifier issues, for the ALP §3 "no more progress"
        // stop condition (same as the reactive loop).
        let mut prev_issues: Vec<String> = Vec::new();
        let mut verify_depth_floor = VerifyDepth::Skip;
        let mut verification: Option<bool> = None;
        // Final attempt's verifier issues, surfaced on the outcome for the
        // goal-level repair prompt (same as the reactive loop).
        let mut verifier_issues: Vec<String> = Vec::new();
        // Phase 4 verdict-channel seam (same as the reactive loop): the final
        // attempt's raw verifier parse confidence and the verifier model.
        let mut verifier_parse: Option<VerifierParse> = None;
        let mut verifier_model: Option<String> = None;
        // Pre-edit objective baseline (see `run_auto_turn_streaming`): a check that
        // is already red before this deep turn edits anything is an out-of-scope
        // baseline failure, so it does not force the retry loop; an edit-introduced
        // green→red regression still gates.
        let baseline_objective_green = match cfg.check_command.as_deref() {
            Some(cmd) => command_is_green(cmd).await,
            None => true,
        };
        for attempt in 1..=max {
            attempts = attempt;

            // Runtime effort escalation (the mechanism `auto_effort_for_prompt`'s
            // doc delegates to): the first attempt runs at the configured effort,
            // but a retry means that effort did not solve the task — the bench
            // shows hard tasks `High` cannot pass do pass at `Xhigh`. So power up
            // every retry to at least `Xhigh` (a floor, so a task already above
            // it is not lowered). Cleared after the loop.
            if attempt > 1 {
                self.set_effort_override(Some(super::ESCALATION_EFFORT_BUDGET));
                deep_note(
                    &render_tx,
                    &ids,
                    "deep: escalating reasoning effort (xhigh) for retry…",
                )
                .await;
            }

            deep_note(
                &render_tx,
                &ids,
                format!("deep: EXEC attempt {attempt}/{max}…"),
            )
            .await;
            let baseline_files = changed_files_snapshot_async().await;
            if let Some(note) = self.exec_leg_note(attempt) {
                deep_note(&render_tx, &ids, note).await;
            }
            if self.exec_swap_enabled() && attempt > ARCHITECT_IMPL_ATTEMPTS {
                // Failure escalation: the native (reserved) model implements
                // from here on, so the edit gate stands down for this turn.
                self.reserved_edit_gate = false;
            }
            let exec_result = self
                .deep_subturn(
                    exec_prompt(&task, &plan_md, exec_retry.as_deref()),
                    Vec::new(),
                    base_mode,
                    self.exec_leg_client(attempt),
                    &render_tx,
                    &prompter,
                )
                .await;
            // Clear the escalation floor immediately after the (possibly
            // escalated) EXEC sub-turn — before `?` and before VERIFY — so it
            // never leaks into the read-only verify turn or a later turn on
            // error. Idempotent when no escalation was set.
            self.set_effort_override(None);
            let summary = exec_result?;
            let edited_paths = edited_file_paths(&summary);
            let assistant_claim = latest_assistant_text(&summary.assistant_messages);
            acc.fold(summary);

            // Objective gate: the project's own check command, when configured.
            let check_observation = match cfg.check_command.as_deref() {
                Some(cmd) => {
                    let observation = run_check_command(cmd).await;
                    deep_note(
                        &render_tx,
                        &ids,
                        format!(
                            "deep: check `{cmd}` → {}",
                            if observation.green {
                                "green ✓"
                            } else {
                                "red ✗"
                            }
                        ),
                    )
                    .await;
                    Some(observation)
                }
                None => None,
            };
            let objective_ok = check_observation.as_ref().is_none_or(|check| check.green);

            let after_files = changed_files_snapshot_async().await;
            let diff_paths = attempt_diff_paths(&baseline_files, &after_files, &edited_paths);
            let (diff, line_churn) =
                bounded_git_diff_for_paths(diff_paths.clone(), 6000).await;
            let selected_depth = verify_depth_for_band(
                self.verify_band,
                diff_paths.len(),
                line_churn,
                objective_ok,
                paths_touch_security(&diff_paths),
                paths_touch_tests(&diff_paths),
            );
            let depth = selected_depth.max(verify_depth_floor);
            verify_depth_floor = depth;
            if depth == VerifyDepth::Skip {
                deep_note(
                    &render_tx,
                    &ids,
                    "deep: trivial green change — skipping deep verify",
                )
                .await;
                decision = DeepDecision::Accept;
                break;
            }
            let verify_note = match self.deep_verify_primary_model_label() {
                Some(model) => format!("deep: VERIFY phase (read-only, cross-model {model})…"),
                None => "deep: VERIFY phase (read-only)…".to_string(),
            };
            deep_note(&render_tx, &ids, verify_note).await;
            let verify_result = self
                .verify_subturn(
                    verify_prompt(
                        &task_with_retry_context(&task, extra.as_deref()),
                        &diff,
                        cfg.check_command
                            .as_deref()
                            .zip(check_observation.as_ref()),
                        &diff_paths,
                        &assistant_claim,
                        if depth == VerifyDepth::SingleLens {
                            VerifyLensMode::SpecOnly
                        } else {
                            VerifyLensMode::Full
                        },
                    ),
                    &render_tx,
                    &ids,
                    &prompter,
                )
                .await;
            // A failed VERIFY leg (transient streaming error) must NOT throw away
            // the EXEC edits already applied this attempt via `?`. Fold a
            // conservative non-accept (Timeout) so the loop retries or gives up at
            // the cap, preserving the completed implementation in the work tree.
            let verifier = match verify_result {
                Ok(summary) => {
                    acc.fold(summary);
                    parse_lens_verifier(&self.last_assistant_text())
                }
                Err(_) => verify_leg_failed_verdict(),
            };
            // A still-red baseline failure is out of scope: only an edit-introduced
            // regression gates the deep loop. The verifier still sees raw objective.
            let gating_objective_ok = objective_ok || !baseline_objective_green;
            // Keep accept/retry/stall policy in decision-core; this runtime only
            // supplies observed IO facts from the live VERIFY sub-turn.
            let folded = fold_verification_attempt(
                attempt,
                max,
                gating_objective_ok,
                &verifier,
                &prev_issues,
            );

            verification = Some(goal_facing_accept(&folded));
            verifier_issues = verifier.issues.clone();
            verifier_parse = Some(verifier.parse);
            verifier_model = self
                .deep_verify_succeeded_model_label()
                .map(str::to_string);
            decision = folded.decision;
            deep_note(
                &render_tx,
                &ids,
                verification_outcome_note("deep", decision, attempt, max, objective_ok, &verifier),
            )
            .await;

            match decision {
                DeepDecision::Accept | DeepDecision::GiveUp => {
                    if decision == DeepDecision::Accept {
                        self.record_verified_accept(objective_ok);
                    }
                    break;
                }
                DeepDecision::Retry => {
                    let repair = failure_summary(objective_ok, &verifier);
                    exec_retry = Some(exec_retry_context(
                        &repair,
                        &diff,
                        &diff_paths,
                        cfg.check_command
                            .as_deref()
                            .zip(check_observation.as_ref()),
                    ));
                    extra = Some(repair);
                    prev_issues = verifier.issues.clone();
                }
            }
        }

        let outcome = DeepOutcome {
            decision,
            attempts,
            plan_valid: plan_verdict.valid,
            plan_missing: plan_verdict.missing,
            // Preserve the semantic verifier gate observed when VERIFY ran;
            // proportional trivial-change skips leave it as `None`.
            verification,
            issues: verifier_issues,
            verifier_parse,
            verifier_model,
        };
        deep_note(
            &render_tx,
            &ids,
            format!(
                "deep: {} after {attempts} attempt(s) · plan {}",
                decision.as_str(),
                if outcome.plan_valid {
                    "valid"
                } else {
                    "invalid"
                }
            ),
        )
        .await;

        Ok((acc.into_summary(), outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_check_command_finds_cargo_in_this_crate() {
        // `cargo test -p runtime` runs with the crate root as cwd, which has a
        // Cargo.toml, so detection must pick the Rust check command.
        assert_eq!(
            detect_check_command().as_deref(),
            Some("cargo build --tests")
        );
    }

    #[test]
    fn detect_check_command_reactive_default_is_a_cheap_build_not_a_full_test_run() {
        // The detected command is auto-wired as the reactive per-coding-turn gate
        // (it runs after *every* edited turn — see
        // `install_reactive_verify_gate_if_coding`). On a large repo a full
        // `cargo test` would force a multi-minute test *build + run* after each
        // edit, freezing the loop on the objective check. The Rust auto default
        // must therefore compile the test targets without running them: a green
        // build is a real objective signal at a fraction of the cost.
        let cmd =
            detect_check_command().expect("this crate has a Cargo.toml, so a command is detected");

        // It must still drive the test targets through the compiler (so it catches
        // the same build/type errors `cargo test` would surface)…
        assert!(
            cmd.starts_with("cargo build") && cmd.contains("--tests"),
            "Rust reactive default must build the test targets, got {cmd:?}"
        );
        // …but it must NOT be the heavy full `cargo test` run that this finding
        // replaces.
        assert_ne!(
            cmd, "cargo test",
            "the reactive auto default must not be a full multi-minute test run"
        );
    }

    #[test]
    fn interpret_green_reads_exit_codes() {
        assert!(interpret_green(None));
        assert!(interpret_green(Some("exit_code:0")));
        assert!(!interpret_green(Some("exit_code:1")));
        assert!(!interpret_green(Some("exit_code:137")));
        assert!(!interpret_green(Some("timeout")));
        assert!(!interpret_green(Some("garbage")));
    }

    #[test]
    fn verify_depth_is_conservative_across_band_and_change_matrix() {
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Trivial,
                RouteTaskRisk::Low,
                1,
                CHURN_TRIVIAL_MAX,
                true,
                false,
                false,
            ),
            VerifyDepth::Skip
        );
        for risk in [RouteTaskRisk::Low, RouteTaskRisk::Medium] {
            assert_eq!(
                verify_depth(
                    RouteTaskComplexity::Small,
                    risk,
                    FILES_SMALL_MAX,
                    CHURN_SMALL_MAX,
                    true,
                    false,
                    false,
                ),
                VerifyDepth::SingleLens
            );
        }
        for complexity in [RouteTaskComplexity::Medium, RouteTaskComplexity::Large] {
            assert_eq!(
                verify_depth(complexity, RouteTaskRisk::Low, 1, 1, true, false, false),
                VerifyDepth::Full
            );
        }
        for risk in [RouteTaskRisk::High, RouteTaskRisk::Critical] {
            assert_eq!(
                verify_depth(
                    RouteTaskComplexity::Trivial,
                    risk,
                    1,
                    1,
                    true,
                    false,
                    false,
                ),
                VerifyDepth::Full
            );
        }
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Small,
                RouteTaskRisk::Low,
                FILES_SMALL_MAX + 1,
                1,
                true,
                false,
                false,
            ),
            VerifyDepth::Full
        );
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Small,
                RouteTaskRisk::Low,
                1,
                CHURN_SMALL_MAX + 1,
                true,
                false,
                false,
            ),
            VerifyDepth::Full
        );
    }

    #[test]
    fn verify_depth_forces_full_on_failed_sensitive_or_unknown_changes() {
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Trivial,
                RouteTaskRisk::Low,
                1,
                1,
                false,
                false,
                false,
            ),
            VerifyDepth::Full
        );
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Trivial,
                RouteTaskRisk::Low,
                0,
                0,
                true,
                false,
                false,
            ),
            VerifyDepth::Full,
            "an unscoped edit must not bypass verification"
        );

        let security_paths = vec!["src/Auth/session.rs".to_string()];
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Trivial,
                RouteTaskRisk::Low,
                1,
                1,
                true,
                paths_touch_security(&security_paths),
                false,
            ),
            VerifyDepth::Full
        );
        let test_paths = vec!["tests/parser_spec.rs".to_string()];
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Trivial,
                RouteTaskRisk::Low,
                1,
                1,
                true,
                false,
                paths_touch_tests(&test_paths),
            ),
            VerifyDepth::Full
        );

        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Unknown,
                RouteTaskRisk::Low,
                1,
                1,
                true,
                false,
                false,
            ),
            VerifyDepth::Full
        );
        assert_eq!(
            verify_depth(
                RouteTaskComplexity::Trivial,
                RouteTaskRisk::Unknown,
                1,
                1,
                true,
                false,
                false,
            ),
            VerifyDepth::Full
        );
        assert_eq!(
            verify_depth_for_band(None, 1, 1, true, false, false),
            VerifyDepth::Full,
            "an absent host band must preserve full verification"
        );
    }

    #[test]
    fn verify_depth_retry_floor_never_downgrades() {
        assert_eq!(
            VerifyDepth::SingleLens.max(VerifyDepth::Full),
            VerifyDepth::Full
        );
        assert_eq!(
            VerifyDepth::Skip.max(VerifyDepth::SingleLens),
            VerifyDepth::SingleLens
        );
    }

    #[test]
    fn diff_line_churn_excludes_patch_headers() {
        let diff =
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n-old\n+new\n----deleted\n++++added\n context\n";
        assert_eq!(diff_line_churn(diff), 4);
    }

    #[test]
    fn read_only_allow_rules_unblock_typed_cargo_inspection_not_writes() {
        // End-to-end: the exact rule set `deep_subturn` injects must let the
        // shell-free `Cargo` typed tool run inspection verbs under a downgraded
        // ReadOnly phase (the deep VERIFY denial), while `run`/`build` and other
        // write tools stay gated — matching the `bash(cargo …)` relaxation.
        use crate::permissions::PermissionOutcome;

        let mut policy = crate::PermissionPolicy::new(crate::PermissionMode::ReadOnly)
            .with_tool_requirement("Cargo", crate::PermissionMode::WorkspaceWrite)
            .with_tool_requirement("bash", crate::PermissionMode::DangerFullAccess)
            .with_tool_requirement("write_file", crate::PermissionMode::WorkspaceWrite);

        let grant = policy.add_temporary_allow_rules(read_only_bash_allow_rules());

        // Typed inspection verbs are now allowed…
        for verb in ["check", "test", "clippy", "fmt"] {
            let input = format!(r#"{{"action":"{verb}"}}"#);
            assert_eq!(
                policy.authorize("Cargo", &input, None),
                PermissionOutcome::Allow,
                "Cargo({verb}) should be permitted by the scoped read-only grant"
            );
        }
        // …and so is the equivalent shell form.
        assert_eq!(
            policy.authorize("bash", r#"{"command":"cargo test --all"}"#, None),
            PermissionOutcome::Allow
        );

        // But heavier/arbitrary `Cargo` verbs and unrelated writes stay denied.
        for verb in ["run", "build"] {
            let input = format!(r#"{{"action":"{verb}"}}"#);
            assert!(
                matches!(
                    policy.authorize("Cargo", &input, None),
                    PermissionOutcome::Deny { .. }
                ),
                "Cargo({verb}) must remain gated"
            );
        }
        assert!(matches!(
            policy.authorize("write_file", r#"{"path":"a.rs","content":"x"}"#, None),
            PermissionOutcome::Deny { .. }
        ));

        // No leak once the phase restores.
        policy.remove_temporary_allow_rules(grant);
        assert!(matches!(
            policy.authorize("Cargo", r#"{"action":"test"}"#, None),
            PermissionOutcome::Deny { .. }
        ));
    }

    #[test]
    fn plan_prompt_headers_satisfy_validate_plan() {
        // The PLAN prompt instructs four headers; a plan literally echoing them
        // must validate, proving the prompt and the policy agree.
        let echoed = "## Target files\nx\n## Invariants\ny\n## Expected tests\nz\n## Risks\nw";
        assert!(validate_plan(echoed).valid);
        // And the prompt itself names the canonical sections.
        let prompt = plan_prompt("do a thing", None, &[]);
        assert!(prompt.contains("[deep:PLAN]"));
        for header in [
            "## Target files",
            "## Invariants",
            "## Expected tests",
            "## Risks",
        ] {
            assert!(prompt.contains(header), "missing {header}");
        }
        assert!(
            prompt.contains("Do NOT spawn sub-agents, delegate, or call Agent, SpawnMultiAgent, Workflow, or SendMessage"),
            "bounded PLAN must explicitly forbid delegation"
        );
    }

    #[test]
    fn plan_prompt_carries_baseline_and_missing_feedback() {
        let prompt = plan_prompt(
            "t",
            Some("Baseline check `ct` currently FAILS"),
            &["tests".into(), "risks".into()],
        );
        assert!(prompt.contains("Baseline check `ct` currently FAILS"));
        assert!(
            prompt.contains("missing, empty, or placeholder-only required sections: tests, risks")
        );
        assert!(prompt.contains("concrete, non-placeholder content"));
    }

    #[test]
    fn exec_prompt_includes_retry_only_when_present() {
        assert!(!exec_prompt("t", "p", None).contains("repair contract"));
        let retry = exec_prompt("t", "p", Some("Your previous attempt did NOT pass."));
        assert!(retry.contains("[deep:EXEC]"));
        assert!(retry.contains("Your previous attempt did NOT pass."));
        assert!(retry.contains("Immediate mechanical edits"));
        assert!(retry.contains("exact receiver replacements"));
        assert!(retry.contains("Preserve call receivers during renames"));
    }

    #[test]
    fn exec_demands_reproduction_first_and_distrusts_a_prior_fix() {
        // The two disciplines that beat zo's repeated surface fixes, taught to
        // the implementer as rules (no keyword gate): write a failing-first
        // reproduction before fixing a bug, and treat a recent change to the same
        // code as suspect rather than ground truth.
        let exec = exec_prompt("the streaming stutter is still there", "plan", None);
        assert!(
            exec.to_lowercase().contains("reproduces"),
            "exec must require a failing-first reproduction for a bug fix"
        );
        assert!(
            exec.contains("SUSPECT"),
            "exec must distrust a recent prior fix to the same code"
        );
    }

    #[test]
    fn verify_rejects_a_bug_fix_without_a_reproduction_test() {
        // The verifier closes the hole the prior failed commit fell through: a
        // plausible bug-fix diff that passes the pre-existing (toothless) suite.
        let check = CheckObservation {
            green: true,
            output_tail: "42 passed".to_string(),
        };
        let verify = verify_prompt(
            "fix the bug",
            "diff",
            Some(("cargo test", &check)),
            &[],
            "implemented the fix",
            VerifyLensMode::Full,
        );
        assert!(
            verify.contains("fails on the unfixed code"),
            "verifier must require a test that reproduces the bug (RED before, green after)"
        );
        assert!(
            verify.contains("ONLY where a failing-first reproduction is feasible"),
            "verifier must carve out genuinely-untestable fixes (heisenbug/TUI/config) \
             instead of hard-rejecting them and burning retries"
        );
    }

    #[test]
    fn retry_context_is_visible_to_retry_and_verify_prompts() {
        let repair = "Mandatory repair checklist:\n- Fix every issue above\n- Also handle the MCP ToolSearch path";
        let retry = reactive_retry_prompt("why did it stop?", repair);
        assert!(retry.contains("Current request context:"));
        assert!(retry.contains("why did it stop?"));
        assert_eq!(
            retry.matches("Also handle the MCP ToolSearch path").count(),
            1,
            "auto:RETRY should not duplicate the repair contract inside the task"
        );

        let task = task_with_retry_context("why did it stop?", Some(repair));
        let verify = verify_prompt(
            &task,
            "diff",
            None,
            &[],
            "updated the implementation",
            VerifyLensMode::Full,
        );
        assert!(verify.contains("Task:\nwhy did it stop?"));
        assert!(verify.contains("Latest repair/update context:"));
        assert!(verify.contains("Also handle the MCP ToolSearch path"));
    }

    #[test]
    fn verify_prompt_is_strict_json_and_states_objective() {
        let check = CheckObservation {
            green: false,
            output_tail: "CHECK_OUTPUT_TAIL_MARKER".to_string(),
        };
        let with = verify_prompt(
            "t",
            "diff",
            Some(("cargo test", &check)),
            &["src/changed.rs".to_string()],
            "ASSISTANT_CLAIM_MARKER",
            VerifyLensMode::Full,
        );
        assert!(with.contains("[deep:VERIFY]"));
        assert!(with.contains("Objective check `cargo test`: FAIL"));
        assert!(with.contains("CHECK_OUTPUT_TAIL_MARKER"));
        assert!(with.contains("src/changed.rs"));
        assert!(with.contains("ASSISTANT_CLAIM_MARKER"));
        assert!(with.contains("scoped git diff"));
        assert!(with.contains(VERIFY_JSON_ACCEPT_EXAMPLE));
        assert!(with.contains(VERIFY_JSON_REJECT_EXAMPLE));
        assert!(
            with.contains("Do NOT spawn sub-agents, delegate, or call Agent, SpawnMultiAgent, Workflow, or SendMessage"),
            "VERIFY must imperatively require one inline verifier"
        );
        let without = verify_prompt(
            "t",
            "diff",
            None,
            &[],
            "claim",
            VerifyLensMode::Full,
        );
        assert!(without.contains("No objective check command was configured"));
    }

    #[test]
    fn verify_prompt_examples_are_strict_parseable_json() {
        // The examples are the per-lens contract; parse_lens_verifier folds them
        // under AnyReject. The all-true example accepts; the one-false (spec)
        // example rejects on that single lens objection.
        let accept = parse_lens_verifier(VERIFY_JSON_ACCEPT_EXAMPLE);
        assert_eq!(accept.parse, VerifierParse::Json);
        assert!(accept.accepted);

        let reject = parse_lens_verifier(VERIFY_JSON_REJECT_EXAMPLE);
        assert_eq!(reject.parse, VerifierParse::Json);
        assert!(!reject.accepted, "any single lens reject blocks acceptance");
        assert_eq!(reject.issues.len(), 1);
    }

    #[test]
    fn verify_prompt_uses_scalar_spec_contract_only_for_single_lens() {
        let single = verify_prompt(
            "task",
            "diff",
            None,
            &[],
            "claim",
            VerifyLensMode::SpecOnly,
        );
        assert!(single.contains("ONLY the spec/task-compliance dimension"));
        assert!(single.contains(VERIFY_SCALAR_ACCEPT_EXAMPLE));
        assert!(single.contains(VERIFY_SCALAR_REJECT_EXAMPLE));
        assert!(!single.contains(VERIFY_JSON_ACCEPT_EXAMPLE));
        assert!(!single.contains("\"regression\""));
        assert!(!single.contains("\"security\""));

        let scalar_accept = parse_lens_verifier(VERIFY_SCALAR_ACCEPT_EXAMPLE);
        assert_eq!(scalar_accept.parse, VerifierParse::Json);
        assert!(scalar_accept.accepted);
        let scalar_reject = parse_lens_verifier(VERIFY_SCALAR_REJECT_EXAMPLE);
        assert_eq!(scalar_reject.parse, VerifierParse::Json);
        assert!(!scalar_reject.accepted);

        let full = verify_prompt(
            "task",
            "diff",
            None,
            &[],
            "claim",
            VerifyLensMode::Full,
        );
        assert!(full.contains(VERIFY_JSON_ACCEPT_EXAMPLE));
        assert!(full.contains(VERIFY_JSON_REJECT_EXAMPLE));
        assert!(full.contains("\"regression\": does"));
    }

    #[test]
    fn lens_verify_rejects_when_only_security_lens_objects() {
        // The whole change is rejected if ANY lens objects, even when spec and
        // regression accept — the multi-lens rigor BB1-lite adds over a single
        // holistic verdict.
        let verdict = parse_lens_verifier(
            r#"{"spec": true, "regression": true, "security": false, "issues": ["logs a secret"]}"#,
        );
        assert!(!verdict.accepted);
        assert_eq!(verdict.parse, VerifierParse::Json);
        assert_eq!(verdict.issues, vec!["logs a secret".to_string()]);
    }

    #[test]
    fn lens_verify_falls_back_to_single_verdict_contract() {
        // A model that ignores the per-lens rubric and returns the old
        // {accepted,issues} shape still resolves correctly via the fallback.
        let accept = parse_lens_verifier(r#"{"accepted": true, "issues": []}"#);
        assert!(accept.accepted);
        // An unusable response is a conservative non-accept, never a silent pass.
        assert!(!parse_lens_verifier("not json at all").accepted);
    }

    #[test]
    fn goal_facing_accept_requires_deep_accept_not_just_verifier_gate() {
        // Objective check is RED but the verifier JSON-accepted. The verifier gate
        // still accepts (that is the deep loop's own retry/stall policy), and the
        // deep decision is GiveUp at the attempt cap — so the goal-facing scalar
        // must NOT export accept. Otherwise a goal with no objective validators
        // could be marked Succeeded on an objective-red turn (silent false success).
        let verifier = VerifierVerdict {
            accepted: true,
            issues: Vec::new(),
            parse: VerifierParse::Json,
            evidence: None,
        };
        let folded = fold_verification_attempt(2, 2, false, &verifier, &[]);

        assert!(folded.gate_accepted);
        assert_eq!(folded.decision, DeepDecision::GiveUp);
        assert!(!goal_facing_accept(&folded));
    }

    #[test]
    fn edited_file_paths_extracts_successful_write_targets() {
        let summary = TurnSummary {
            assistant_messages: Vec::new(),
            tool_results: vec![
                ConversationMessage::tool_result(
                    "edit-1",
                    "edit_file",
                    r#"{"filePath":"crates/runtime/src/lib.rs"}"#,
                    false,
                ),
                ConversationMessage::tool_result(
                    "write-1",
                    "write_file",
                    r#"{"path":"crates/runtime/src/new.rs"}"#,
                    false,
                ),
                ConversationMessage::tool_result(
                    "read-ignored",
                    "read_file",
                    r#"{"filePath":"README.md"}"#,
                    false,
                ),
                ConversationMessage::tool_result(
                    "failed-ignored",
                    "edit_file",
                    r#"{"filePath":"crates/runtime/src/failed.rs"}"#,
                    true,
                ),
            ],
            prompt_cache_events: Vec::new(),
            iterations: 1,
            usage: TokenUsage::default(),
            turn_output_tokens: 0,
            auto_compaction: None,
            microcompact: None,
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted: None,
        };

        assert_eq!(
            edited_file_paths(&summary),
            vec![
                "crates/runtime/src/lib.rs".to_string(),
                "crates/runtime/src/new.rs".to_string()
            ]
        );
    }

    #[test]
    fn deep_summary_fold_last_wins_cumulative_but_sums_turn_output() {
        // `usage` is the *cumulative* session usage, so a folded deep turn's usage
        // must be the LATEST snapshot, never the sum — summing multiplied the total
        // by the sub-turn count and inflated the goal budget / tripped compaction.
        // `turn_output_tokens` is each sub-turn's OWN delta, so it DOES sum (the
        // goal budget charges the whole multi-sub-turn deep turn). Iterations sum.
        let sub_turn = |iterations: usize, cumulative_output: u32, turn_delta: u32| TurnSummary {
            assistant_messages: Vec::new(),
            tool_results: Vec::new(),
            prompt_cache_events: Vec::new(),
            iterations,
            usage: TokenUsage {
                output_tokens: cumulative_output,
                ..Default::default()
            },
            turn_output_tokens: turn_delta,
            auto_compaction: None,
            microcompact: None,
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted: None,
        };
        let mut acc = DeepSummaryAcc::default();
        acc.fold(sub_turn(2, 100, 100)); // cumulative 100, this leg produced 100
        acc.fold(sub_turn(3, 250, 150)); // cumulative 250, this leg produced 150
        let summary = acc.into_summary();
        assert_eq!(
            summary.usage.output_tokens, 250,
            "usage is the latest cumulative snapshot, not 100 + 250"
        );
        assert_eq!(
            summary.turn_output_tokens, 250,
            "turn_output is the SUM of per-leg deltas (100 + 150)"
        );
        assert_eq!(
            summary.iterations, 5,
            "iterations accumulate across sub-turns"
        );
    }

    #[test]
    fn deep_summary_fold_preserves_a_sub_turn_budget_stop() {
        // A budget stop in ANY leg must survive into the composed summary:
        // dropping it silently disarmed the `/loop` budget-pause and the
        // grind-escalation streak whenever the deep gate wrapped the turn.
        let sub_turn = |budget_exhausted: Option<BudgetExhausted>| TurnSummary {
            assistant_messages: Vec::new(),
            tool_results: Vec::new(),
            prompt_cache_events: Vec::new(),
            iterations: 1,
            usage: TokenUsage::default(),
            turn_output_tokens: 0,
            auto_compaction: None,
            microcompact: None,
            deep_verification: None,
            verification_issues: Vec::new(),
            deep_verifier_parse: None,
            deep_verifier_model: None,
            budget_exhausted,
        };
        let mut acc = DeepSummaryAcc::default();
        acc.fold(sub_turn(None));
        acc.fold(sub_turn(Some(BudgetExhausted::Deadline)));
        // A later clean leg must not erase the earlier stop.
        acc.fold(sub_turn(None));
        assert_eq!(
            acc.into_summary().budget_exhausted,
            Some(BudgetExhausted::Deadline),
            "a sub-turn budget stop must survive the deep-turn fold"
        );
    }

    #[test]
    fn attempt_diff_paths_excludes_preexisting_unedited_dirty_files() {
        let baseline = vec!["crates/api/src/client.rs".to_string()];
        let after = vec![
            "crates/api/src/client.rs".to_string(),
            "crates/runtime/src/conversation/deep_gate.rs".to_string(),
        ];
        let edited =
            vec!["crates/zo-cli/src/session/slash_dispatch/helpers_tui.rs".to_string()];

        assert_eq!(
            attempt_diff_paths(&baseline, &after, &edited),
            vec![
                "crates/runtime/src/conversation/deep_gate.rs".to_string(),
                "crates/zo-cli/src/session/slash_dispatch/helpers_tui.rs".to_string(),
            ]
        );
    }

    #[test]
    fn attempt_diff_paths_keeps_preexisting_file_when_attempt_edited_it() {
        let baseline = vec!["crates/runtime/src/conversation/deep_gate.rs".to_string()];
        let after = baseline.clone();
        let edited = vec!["crates/runtime/src/conversation/deep_gate.rs".to_string()];

        assert_eq!(attempt_diff_paths(&baseline, &after, &edited), edited);
    }

    #[test]
    fn verification_outcome_note_summarizes_without_wire_parse_tokens() {
        let verifier = VerifierVerdict {
            accepted: false,
            issues: vec!["leaked settings file".into()],
            parse: VerifierParse::Json,
            evidence: None,
        };
        let note = verification_outcome_note("auto", DeepDecision::GiveUp, 2, 2, true, &verifier);

        assert_eq!(
            note,
            "auto: stopped — out of attempts (objective ok; strict verifier found 1 issue)"
        );
        assert!(!note.contains("verifier json"));
        assert!(!note.contains("accepted\":"));
    }

    #[test]
    fn verification_outcome_note_handles_retry_and_missing_verdict() {
        let verifier = VerifierVerdict {
            accepted: false,
            issues: Vec::new(),
            parse: VerifierParse::Unparseable,
            evidence: None,
        };
        let note = verification_outcome_note("deep", DeepDecision::Retry, 1, 3, false, &verifier);

        assert_eq!(
            note,
            "deep: retrying — verifier returned no usable verdict (objective red; attempt 1/3)"
        );
    }

    #[test]
    fn failure_summary_is_bounded_and_lists_issues() {
        let verifier = VerifierVerdict {
            accepted: false,
            issues: vec!["off-by-one".into(), "missing null check".into()],
            parse: decision_core::deep_lane::VerifierParse::Json,
            evidence: None,
        };
        let summary = failure_summary(false, &verifier);
        assert!(summary.contains("objective check is RED"));
        assert!(summary.contains("off-by-one"));
        assert!(summary.contains("Mandatory repair checklist"));
        assert!(summary.contains("stale symbol"));
        assert!(summary.contains("cache path"));
        assert!(summary.len() <= MAX_SUMMARY_CHARS);

        // A huge issue list is truncated on a char boundary, never panicking.
        let big = VerifierVerdict {
            accepted: false,
            issues: vec!["x".repeat(5000)],
            parse: decision_core::deep_lane::VerifierParse::Json,
            evidence: None,
        };
        assert!(failure_summary(true, &big).len() <= MAX_SUMMARY_CHARS);
    }

    #[test]
    fn exec_retry_context_carries_bounded_prior_attempt_evidence() {
        let check = CheckObservation {
            green: false,
            output_tail: format!("{}CHECK_TAIL", "é".repeat(CHECK_OUTPUT_TAIL_BYTES)),
        };
        let context = exec_retry_context(
            "repair every verifier issue",
            &"d".repeat(EXEC_PRIOR_DIFF_BYTES * 2),
            &[format!(
                "src/{}path.rs",
                "é".repeat(EXEC_PRIOR_EDITED_PATHS_BYTES)
            )],
            Some(("cargo test -p runtime", &check)),
        );

        assert!(context.contains("repair every verifier issue"));
        assert!(context.contains("cargo test -p runtime"));
        assert!(context.contains("CHECK_TAIL"));
        assert!(!context.contains(&"d".repeat(EXEC_PRIOR_DIFF_BYTES + 1)));
        assert!(
            context.len()
                <= "repair every verifier issue".len()
                    + EXEC_PRIOR_DIFF_BYTES
                    + EXEC_PRIOR_EDITED_PATHS_BYTES
                    + CHECK_OUTPUT_TAIL_BYTES
                    + 512
        );
    }

    #[test]
    fn truncate_on_boundary_respects_utf8() {
        let mut s = "héllo wörld".to_string();
        truncate_on_boundary(&mut s, 2); // byte 2 splits 'é' (2 bytes from 'h')
        assert!(s.is_char_boundary(s.len()));
        assert!(s.len() <= 2);

        let mut tail = format!("{}TAIL", "é".repeat(CHECK_OUTPUT_TAIL_BYTES));
        truncate_to_tail_on_boundary(&mut tail, CHECK_OUTPUT_TAIL_BYTES);
        assert!(tail.is_char_boundary(tail.len()));
        assert!(tail.len() <= CHECK_OUTPUT_TAIL_BYTES);
        assert!(tail.ends_with("TAIL"));
    }

    #[test]
    fn verify_leg_failure_folds_to_conservative_non_accept_not_abort() {
        // A failed VERIFY sub-turn (transient streaming error) used to abort the
        // whole deep turn via `?`, discarding the EXEC edits already applied this
        // attempt. The Err-fold path now folds `verify_leg_failed_verdict()`
        // instead. That verdict must be a non-accept tagged `Timeout` so the gate
        // never accepts on a failed verify, and the loop continues honestly.
        let verdict = verify_leg_failed_verdict();
        assert!(!verdict.accepted, "a failed verify leg must never accept");
        assert_eq!(verdict.parse, VerifierParse::Timeout);
        assert!(verdict.issues.is_empty());

        // Mid-loop (attempts remain): fold ⇒ Retry, preserving the applied edits
        // rather than throwing them away. Never Accept.
        let mid = fold_verification_attempt(1, 2, true, &verdict, &[]);
        assert!(!mid.gate_accepted, "Timeout is not a salvage accept");
        assert_eq!(mid.decision, DeepDecision::Retry);

        // Last attempt: fold ⇒ GiveUp, which still ends the turn honestly and
        // leaves the completed implementation in the work tree (no `?` unwind).
        let last = fold_verification_attempt(2, 2, true, &verdict, &[]);
        assert!(!last.gate_accepted);
        assert_eq!(last.decision, DeepDecision::GiveUp);
        // The goal-facing scalar must NOT read as accepted on a failed verify.
        assert!(!goal_facing_accept(&last));
    }

    #[test]
    fn verify_leg_failure_display_is_honest_timeout() {
        // The conservative verdict drives an honest "verifier timed out" note in
        // both reactive and plan-first paths (display already handles Timeout).
        let verdict = verify_leg_failed_verdict();
        assert_eq!(verifier_display_summary(&verdict), "verifier timed out");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deep_subturn_restores_permission_mode_when_dropped_mid_stream() {
        use std::future::{Future, pending};
        use std::pin::Pin;

        use crate::conversation::{
            ApiRequest, AssistantEvent, AsyncApiClient, RuntimeError, StaticToolExecutor,
        };
        use crate::permission::{
            PermissionDecision as AsyncPermissionDecision, PermissionError,
            PermissionRequest as AsyncPermissionRequest,
        };
        use crate::session::Session;

        struct NoopApiClient;

        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }

        struct PendingAsyncClient {
            entered: Arc<tokio::sync::Notify>,
        }

        impl AsyncApiClient for PendingAsyncClient {
            fn stream_async<'a>(
                &'a self,
                _request: ApiRequest,
                _render_tx: mpsc::Sender<RenderBlock>,
                _text_block_id: crate::message_stream::types::BlockId,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<Vec<AssistantEvent>, RuntimeError>> + Send + 'a,
                >,
            > {
                let entered = Arc::clone(&self.entered);
                Box::pin(async move {
                    entered.notify_one();
                    pending().await
                })
            }
        }

        struct DenyAsyncPrompter;

        impl AsyncPermissionPrompter for DenyAsyncPrompter {
            fn decide<'a>(
                &'a self,
                _request: AsyncPermissionRequest,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<AsyncPermissionDecision, PermissionError>>
                        + Send
                        + 'a,
                >,
            > {
                Box::pin(async { Ok(AsyncPermissionDecision::Deny) })
            }
        }

        let entered_stream = Arc::new(tokio::sync::Notify::new());
        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite)
                .with_tool_requirement("bash", PermissionMode::DangerFullAccess),
            vec!["system".to_string()],
        )
        .with_async_api_client(Arc::new(PendingAsyncClient {
            entered: Arc::clone(&entered_stream),
        }));
        let (render_tx, mut render_rx) = mpsc::channel(64);
        let _drain = tokio::spawn(async move { while render_rx.recv().await.is_some() {} });
        let prompter: Arc<dyn AsyncPermissionPrompter> = Arc::new(DenyAsyncPrompter);

        let mut subturn = Box::pin(runtime.deep_subturn(
            "inspect before editing".to_string(),
            Vec::new(),
            PermissionMode::ReadOnly,
            SubturnClient::Native,
            &render_tx,
            &prompter,
        ));
        tokio::select! {
            result = subturn.as_mut() => panic!("pending stream unexpectedly completed: {result:?}"),
            () = entered_stream.notified() => {}
        }

        drop(subturn);

        assert_eq!(
            runtime.permission_policy.active_mode(),
            PermissionMode::WorkspaceWrite,
            "dropping a PLAN/VERIFY sub-turn future must restore the previous permission mode"
        );
        assert!(
            matches!(
                runtime.permission_policy.authorize(
                    "bash",
                    r#"{"command":"cargo test -p runtime"}"#,
                    None,
                ),
                crate::PermissionOutcome::Deny { .. }
            ),
            "dropping the sub-turn future must remove the temporary read-only bash/Cargo allow grant"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[allow(clippy::too_many_lines)] // stub-heavy async harness, mirrors the drop test above
    async fn verify_subturn_sends_focused_packet_and_restores_native_after_drop() {
        use std::future::{Future, pending};
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        use crate::conversation::{
            ApiRequest, AssistantEvent, AsyncApiClient, RuntimeError, StaticToolExecutor,
        };
        use crate::permission::{
            PermissionDecision as AsyncPermissionDecision, PermissionError,
            PermissionRequest as AsyncPermissionRequest,
        };
        use crate::session::Session;

        struct NoopApiClient;
        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }

        /// Counts entries and then hangs, so the test can observe *which*
        /// client a leg streamed on and cancel it mid-flight.
        struct CountingPendingClient {
            calls: Arc<AtomicUsize>,
            entered: Arc<tokio::sync::Notify>,
            captured: Option<Arc<Mutex<Option<ApiRequest>>>>,
        }
        impl AsyncApiClient for CountingPendingClient {
            fn stream_async<'a>(
                &'a self,
                request: ApiRequest,
                _render_tx: mpsc::Sender<RenderBlock>,
                _text_block_id: crate::message_stream::types::BlockId,
            ) -> Pin<Box<dyn Future<Output = Result<Vec<AssistantEvent>, RuntimeError>> + Send + 'a>>
            {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if let Some(captured) = &self.captured {
                    *captured.lock().expect("request lock") = Some(request);
                }
                let entered = Arc::clone(&self.entered);
                Box::pin(async move {
                    entered.notify_one();
                    pending().await
                })
            }
        }

        struct DenyAsyncPrompter;
        impl AsyncPermissionPrompter for DenyAsyncPrompter {
            fn decide<'a>(
                &'a self,
                _request: AsyncPermissionRequest,
            ) -> Pin<
                Box<dyn Future<Output = Result<AsyncPermissionDecision, PermissionError>> + Send + 'a>,
            > {
                Box::pin(async { Ok(AsyncPermissionDecision::Deny) })
            }
        }

        let native_calls = Arc::new(AtomicUsize::new(0));
        let native_entered = Arc::new(tokio::sync::Notify::new());
        let cross_calls = Arc::new(AtomicUsize::new(0));
        let cross_entered = Arc::new(tokio::sync::Notify::new());
        let captured = Arc::new(Mutex::new(None));
        let prior_marker = "PRIOR_SESSION_MARKER_MUST_NOT_REACH_VERIFY";
        let mut session = Session::new();
        session
            .push_user_text(prior_marker)
            .expect("seed prior conversation");

        let mut runtime = ConversationRuntime::new(
            session,
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        )
        .with_async_api_client(Arc::new(CountingPendingClient {
            calls: Arc::clone(&native_calls),
            entered: Arc::clone(&native_entered),
            captured: None,
        }));
        runtime.set_deep_verify_client(Some((
            Arc::new(CountingPendingClient {
                calls: Arc::clone(&cross_calls),
                entered: Arc::clone(&cross_entered),
                captured: Some(Arc::clone(&captured)),
            }),
            "cross-verifier-model".to_string(),
        )));

        let (render_tx, mut render_rx) = mpsc::channel(64);
        let _drain = tokio::spawn(async move { while render_rx.recv().await.is_some() {} });
        let prompter: Arc<dyn AsyncPermissionPrompter> = Arc::new(DenyAsyncPrompter);
        let ids = BlockIdGen::default();
        let check = CheckObservation {
            green: true,
            output_tail: "FOCUSED_CHECK_OUTPUT_MARKER".to_string(),
        };
        let packet = verify_prompt(
            "FOCUSED_TASK_MARKER",
            "FOCUSED_DIFF_MARKER",
            Some(("cargo test -p runtime", &check)),
            &["src/focused.rs".to_string()],
            "FOCUSED_ASSISTANT_CLAIM_MARKER",
            VerifyLensMode::Full,
        );

        // The VERIFY leg must stream on the cross-model client, not the native one.
        let mut subturn = Box::pin(runtime.verify_subturn(
            packet,
            &render_tx,
            &ids,
            &prompter,
        ));
        tokio::select! {
            result = subturn.as_mut() => panic!("pending cross stream unexpectedly completed: {result:?}"),
            () = cross_entered.notified() => {}
        }
        let request = captured
            .lock()
            .expect("request lock")
            .clone()
            .expect("captured verifier request");
        let request_text = request
            .messages
            .iter()
            .flat_map(|message| &message.blocks)
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(request.messages.len(), 1);
        assert!(request_text.contains("FOCUSED_TASK_MARKER"));
        assert!(request_text.contains("FOCUSED_DIFF_MARKER"));
        assert!(request_text.contains("FOCUSED_CHECK_OUTPUT_MARKER"));
        assert!(request_text.contains("src/focused.rs"));
        assert!(request_text.contains("FOCUSED_ASSISTANT_CLAIM_MARKER"));
        assert!(!request_text.contains(prior_marker));
        drop(subturn);
        assert!(runtime.session.messages.iter().any(|message| {
            message.blocks.iter().any(
                |block| matches!(block, ContentBlock::Text { text } if text == prior_marker),
            )
        }));
        assert_eq!(cross_calls.load(Ordering::SeqCst), 1, "verify leg must use the cross client");
        assert_eq!(
            native_calls.load(Ordering::SeqCst),
            0,
            "the native client must not stream during a cross verify leg"
        );

        // Dropping the leg mid-stream must restore the native client for
        // subsequent (non-verify) sub-turns.
        let mut ordinary = Box::pin(runtime.deep_subturn(
            "plan next".to_string(),
            Vec::new(),
            PermissionMode::ReadOnly,
            SubturnClient::Native,
            &render_tx,
            &prompter,
        ));
        tokio::select! {
            result = ordinary.as_mut() => panic!("pending native stream unexpectedly completed: {result:?}"),
            () = native_entered.notified() => {}
        }
        drop(ordinary);
        assert_eq!(
            native_calls.load(Ordering::SeqCst),
            1,
            "after the cancelled verify leg, ordinary sub-turns must be back on the native client"
        );
        assert_eq!(cross_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn exec_leg_client_swaps_for_two_attempts_then_escalates_native() {
        use crate::conversation::{ApiRequest, AssistantEvent, RuntimeError, StaticToolExecutor};
        use crate::session::Session;

        struct NoopApiClient;
        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }
        struct NeverAsyncClient;
        impl AsyncApiClient for NeverAsyncClient {
            fn stream_async<'a>(
                &'a self,
                _request: ApiRequest,
                _render_tx: mpsc::Sender<RenderBlock>,
                _text_block_id: crate::message_stream::types::BlockId,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = Result<Vec<AssistantEvent>, RuntimeError>>
                        + Send
                        + 'a,
                >,
            > {
                Box::pin(async { Ok(vec![AssistantEvent::MessageStop]) })
            }
        }

        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        );
        // No contract: every attempt runs native.
        assert_eq!(runtime.exec_leg_client(1), SubturnClient::Native);
        assert_eq!(runtime.exec_leg_client(3), SubturnClient::Native);
        assert!(runtime.exec_leg_note(1).is_none(), "no contract, nothing to announce");

        runtime.set_exec_contract(Some(ExecContract {
            impl_client: None,
            impl_model: "gpt-5.6-terra".to_string(),
            plan_first: true,
        }));
        // Unarmed contract (default medium/hard, or `never`): plan-first
        // metadata stays installed, but every EXEC attempt uses the native
        // client and nothing is announced.
        assert!(runtime.exec_contract().is_some_and(|contract| contract.plan_first));
        assert_eq!(runtime.exec_leg_client(1), SubturnClient::NativeExec);
        assert_eq!(runtime.exec_leg_client(ARCHITECT_IMPL_ATTEMPTS + 1), SubturnClient::NativeExec);
        assert!(runtime.exec_leg_note(1).is_none());

        runtime.set_exec_contract(Some(ExecContract {
            impl_client: Some(Arc::new(NeverAsyncClient)),
            impl_model: "gpt-5.6-terra".to_string(),
            plan_first: false,
        }));
        // Contract: the first ARCHITECT_IMPL_ATTEMPTS run on the implementer,
        // then implementation escalates back to the native (reserved) model —
        // the same "two real failures" rule as the router's premium gate.
        assert_eq!(runtime.exec_leg_client(1), SubturnClient::Implementer);
        assert_eq!(runtime.exec_leg_client(ARCHITECT_IMPL_ATTEMPTS), SubturnClient::Implementer);
        assert_eq!(
            runtime.exec_leg_client(ARCHITECT_IMPL_ATTEMPTS + 1),
            SubturnClient::Native
        );
        let first = runtime.exec_leg_note(1).expect("attempt 1 announces the contract");
        assert!(first.contains("gpt-5.6-terra"), "{first}");
        assert!(runtime.exec_leg_note(2).is_none(), "no re-announcement mid-loop");
        let escalated = runtime
            .exec_leg_note(ARCHITECT_IMPL_ATTEMPTS + 1)
            .expect("escalation announces the native takeover");
        assert!(escalated.contains("escalating"), "{escalated}");

        // Clearing the contract (the host does this every turn entry) restores
        // native legs.
        runtime.set_exec_contract(None);
        assert_eq!(runtime.exec_leg_client(1), SubturnClient::Native);
    }

    #[test]
    fn architect_plan_uses_a_deep_client_or_reserved_native_only() {
        use std::future::Future;
        use std::pin::Pin;

        use crate::conversation::{ApiRequest, AssistantEvent, RuntimeError, StaticToolExecutor};
        use crate::session::Session;

        struct NoopApiClient;
        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }
        struct NeverAsyncClient;
        impl AsyncApiClient for NeverAsyncClient {
            fn stream_async<'a>(
                &'a self,
                _request: ApiRequest,
                _render_tx: mpsc::Sender<RenderBlock>,
                _text_block_id: crate::message_stream::types::BlockId,
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<Vec<AssistantEvent>, RuntimeError>> + Send + 'a,
                >,
            > {
                Box::pin(std::future::pending())
            }
        }

        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        );
        runtime.set_context_model("gpt-5.6-terra");
        runtime.set_deep_tier_only(true);
        runtime.set_deep_plan_client(Some((
            Arc::new(NeverAsyncClient),
            "claude-fable-5".to_string(),
        )));
        assert_eq!(runtime.plan_leg_client().unwrap(), SubturnClient::Plan);

        runtime.set_deep_plan_client(None);
        assert!(
            runtime.plan_leg_client().is_err(),
            "an implementer-tier native model must not inherit PLAN"
        );
        runtime.set_context_model("gpt-5.6-sol");
        assert_eq!(runtime.plan_leg_client().unwrap(), SubturnClient::Native);

        runtime.set_deep_tier_models(vec!["claude-opus-5".to_string()]);
        runtime.set_context_model("opus-5");
        assert_eq!(runtime.plan_leg_client().unwrap(), SubturnClient::Native);
        runtime.set_context_model("claude-fable-5");
        assert!(
            runtime.plan_leg_client().is_err(),
            "an explicit pool replaces the built-in membership"
        );
    }

    #[test]
    fn architect_edit_gate_denies_reserved_foreground_edits_until_exempt() {
        use crate::conversation::{ApiRequest, AssistantEvent, RuntimeError, StaticToolExecutor};
        use crate::session::Session;

        struct NoopApiClient;
        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }

        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        );
        // Not armed (the default; sub-agent/headless runtimes stay here):
        // nothing is denied.
        assert!(runtime.architect_edit_gate_denial("edit_file").is_none());

        runtime.set_reserved_edit_gate(true);
        let denial = runtime
            .architect_edit_gate_denial("edit_file")
            .expect("armed gate must deny a foreground edit");
        assert!(denial.contains("swapped implementer EXEC leg"), "{denial}");
        assert!(
            runtime.architect_edit_gate_denial("Write").is_some(),
            "every edit-result tool is gated"
        );
        assert!(
            runtime.architect_edit_gate_denial("read_file").is_none(),
            "read tools pass"
        );
        assert!(
            runtime.architect_edit_gate_denial("bash").is_none(),
            "non-edit tools pass (bash has its own permission ladder)"
        );

        // An EXEC leg on the implementer client is the contract being honored.
        runtime.exec_impl_leg_active = true;
        assert!(runtime.architect_edit_gate_denial("edit_file").is_none());
        runtime.exec_impl_leg_active = false;

        // A scoped native EXEC leg remains exempt if an armed contract reaches
        // its native escalation attempt.
        runtime.exec_native_leg_active = true;
        assert!(runtime.architect_edit_gate_denial("edit_file").is_none());
        runtime.exec_native_leg_active = false;
        assert!(runtime.architect_edit_gate_denial("edit_file").is_some());

        // A ReadOnly phase already denies writes with mode messaging.
        let prior = runtime.permission_policy.set_active_mode(PermissionMode::ReadOnly);
        assert!(runtime.architect_edit_gate_denial("edit_file").is_none());
        runtime.permission_policy.set_active_mode(prior);

        // Disarming (host does this every turn entry; the deep gate does it on
        // failure escalation) restores ordinary behavior.
        runtime.set_reserved_edit_gate(false);
        assert!(runtime.architect_edit_gate_denial("edit_file").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[allow(clippy::too_many_lines)] // stub-heavy async harness, mirrors the verify swap test above
    async fn exec_impl_leg_sends_focused_packet_and_native_escalation_keeps_history() {
        use std::future::{Future, pending};
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        use crate::conversation::{
            ApiRequest, AssistantEvent, AsyncApiClient, RuntimeError, StaticToolExecutor,
        };
        use crate::permission::{
            PermissionDecision as AsyncPermissionDecision, PermissionError,
            PermissionRequest as AsyncPermissionRequest,
        };
        use crate::session::Session;

        struct NoopApiClient;
        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }
        struct CountingPendingClient {
            calls: Arc<AtomicUsize>,
            entered: Arc<tokio::sync::Notify>,
            captured: Arc<Mutex<Option<ApiRequest>>>,
        }
        impl AsyncApiClient for CountingPendingClient {
            fn stream_async<'a>(
                &'a self,
                request: ApiRequest,
                _render_tx: mpsc::Sender<RenderBlock>,
                _text_block_id: crate::message_stream::types::BlockId,
            ) -> Pin<Box<dyn Future<Output = Result<Vec<AssistantEvent>, RuntimeError>> + Send + 'a>>
            {
                self.calls.fetch_add(1, Ordering::SeqCst);
                *self.captured.lock().expect("request lock") = Some(request);
                let entered = Arc::clone(&self.entered);
                Box::pin(async move {
                    entered.notify_one();
                    pending().await
                })
            }
        }
        struct DenyAsyncPrompter;
        impl AsyncPermissionPrompter for DenyAsyncPrompter {
            fn decide<'a>(
                &'a self,
                _request: AsyncPermissionRequest,
            ) -> Pin<
                Box<dyn Future<Output = Result<AsyncPermissionDecision, PermissionError>> + Send + 'a>,
            > {
                Box::pin(async { Ok(AsyncPermissionDecision::Deny) })
            }
        }

        let native_calls = Arc::new(AtomicUsize::new(0));
        let native_entered = Arc::new(tokio::sync::Notify::new());
        let impl_calls = Arc::new(AtomicUsize::new(0));
        let impl_entered = Arc::new(tokio::sync::Notify::new());
        let native_request = Arc::new(Mutex::new(None));
        let impl_request = Arc::new(Mutex::new(None));
        let prior_marker = "PRIOR_SESSION_MARKER_MUST_NOT_REACH_IMPLEMENTER";
        let mut session = Session::new();
        session
            .push_user_text(prior_marker)
            .expect("seed prior conversation");

        let mut runtime = ConversationRuntime::new(
            session,
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        )
        .with_async_api_client(Arc::new(CountingPendingClient {
            calls: Arc::clone(&native_calls),
            entered: Arc::clone(&native_entered),
            captured: Arc::clone(&native_request),
        }));
        runtime.set_exec_contract(Some(ExecContract {
            impl_client: Some(Arc::new(CountingPendingClient {
                calls: Arc::clone(&impl_calls),
                entered: Arc::clone(&impl_entered),
                captured: Arc::clone(&impl_request),
            })),
            impl_model: "gpt-5.6-terra".to_string(),
            plan_first: true,
        }));

        let (render_tx, mut render_rx) = mpsc::channel(64);
        let _drain = tokio::spawn(async move { while render_rx.recv().await.is_some() {} });
        let prompter: Arc<dyn AsyncPermissionPrompter> = Arc::new(DenyAsyncPrompter);

        let check = CheckObservation {
            green: false,
            output_tail: "FOCUSED_FAILING_CHECK_OUTPUT".to_string(),
        };
        let retry = exec_retry_context(
            "FOCUSED_REPAIR_CONTEXT",
            "FOCUSED_PRIOR_DIFF",
            &["src/focused.rs".to_string()],
            Some(("cargo test -p runtime", &check)),
        );
        let packet = exec_prompt("FOCUSED_TASK", "FOCUSED_PLAN", Some(&retry));

        // A later EXEC attempt still runs on the implementer client, but sends
        // only its self-contained packet. Cancelling it mid-stream must restore
        // the prior transcript and native client.
        let implementer_client = runtime.exec_leg_client(ARCHITECT_IMPL_ATTEMPTS);
        let mut exec = Box::pin(runtime.deep_subturn(
            packet,
            Vec::new(),
            PermissionMode::WorkspaceWrite,
            implementer_client,
            &render_tx,
            &prompter,
        ));
        tokio::select! {
            result = exec.as_mut() => panic!("pending impl stream unexpectedly completed: {result:?}"),
            () = impl_entered.notified() => {}
        }
        let request = impl_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("captured implementer request");
        let request_text = request
            .messages
            .iter()
            .flat_map(|message| &message.blocks)
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(request.messages.len(), 1);
        assert!(request_text.contains("FOCUSED_TASK"));
        assert!(request_text.contains("FOCUSED_PLAN"));
        assert!(request_text.contains("FOCUSED_REPAIR_CONTEXT"));
        assert!(request_text.contains("FOCUSED_PRIOR_DIFF"));
        assert!(request_text.contains("src/focused.rs"));
        assert!(request_text.contains("cargo test -p runtime"));
        assert!(request_text.contains("FOCUSED_FAILING_CHECK_OUTPUT"));
        assert!(!request_text.contains(prior_marker));
        drop(exec);
        assert!(runtime.session.messages.iter().any(|message| {
            message.blocks.iter().any(
                |block| matches!(block, ContentBlock::Text { text } if text == prior_marker),
            )
        }));
        assert_eq!(impl_calls.load(Ordering::SeqCst), 1, "EXEC leg must use the implementer client");
        assert_eq!(native_calls.load(Ordering::SeqCst), 0);
        assert!(
            !runtime.exec_impl_leg_active,
            "dropping the EXEC leg must clear the implementer-leg flag"
        );

        // The post-failure escalation is a native EXEC leg and keeps the full,
        // restored conversation in the session model's cache namespace.
        let native_client = runtime.exec_leg_client(ARCHITECT_IMPL_ATTEMPTS + 1);
        let mut ordinary = Box::pin(runtime.deep_subturn(
            exec_prompt("NATIVE_TASK", "NATIVE_PLAN", Some(&retry)),
            Vec::new(),
            PermissionMode::WorkspaceWrite,
            native_client,
            &render_tx,
            &prompter,
        ));
        tokio::select! {
            result = ordinary.as_mut() => panic!("pending native stream unexpectedly completed: {result:?}"),
            () = native_entered.notified() => {}
        }
        let request = native_request
            .lock()
            .expect("request lock")
            .clone()
            .expect("captured native escalation request");
        let request_text = request
            .messages
            .iter()
            .flat_map(|message| &message.blocks)
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(request_text.contains(prior_marker));
        assert!(request_text.contains("NATIVE_TASK"));
        drop(ordinary);
        assert_eq!(native_calls.load(Ordering::SeqCst), 1);
        assert_eq!(impl_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[allow(clippy::too_many_lines)] // two end-to-end candidate-walk scenarios
    async fn verify_subturn_uses_ranked_candidates_then_native_fallback() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::conversation::{
            ApiRequest, AssistantEvent, AsyncApiClient, RuntimeError, StaticToolExecutor,
        };
        use crate::permission::{
            PermissionDecision as AsyncPermissionDecision, PermissionError,
            PermissionRequest as AsyncPermissionRequest,
        };
        use crate::session::Session;

        struct NoopApiClient;
        impl ApiClient for NoopApiClient {
            fn stream(&mut self, _request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
                Ok(vec![AssistantEvent::MessageStop])
            }
        }

        #[derive(Clone, Copy)]
        enum Outcome {
            Stop,
            RateLimit,
        }

        struct CountingAsyncClient {
            calls: Arc<AtomicUsize>,
            outcome: Outcome,
        }
        impl AsyncApiClient for CountingAsyncClient {
            fn stream_async<'a>(
                &'a self,
                _request: ApiRequest,
                _render_tx: mpsc::Sender<RenderBlock>,
                _text_block_id: crate::message_stream::types::BlockId,
            ) -> Pin<Box<dyn Future<Output = Result<Vec<AssistantEvent>, RuntimeError>> + Send + 'a>>
            {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let outcome = self.outcome;
                Box::pin(async move {
                    match outcome {
                        Outcome::Stop => Ok(vec![
                            AssistantEvent::TextDelta(
                                r#"{"accepted":true,"issues":[]}"#.to_string(),
                            ),
                            AssistantEvent::MessageStop,
                        ]),
                        Outcome::RateLimit => Err(RuntimeError::with_provider_error_class(
                            "verifier rate-limited",
                            api::ProviderErrorClass::RateLimit { retry_after: None },
                        )),
                    }
                })
            }
        }

        struct DenyAsyncPrompter;
        impl AsyncPermissionPrompter for DenyAsyncPrompter {
            fn decide<'a>(
                &'a self,
                _request: AsyncPermissionRequest,
            ) -> Pin<
                Box<dyn Future<Output = Result<AsyncPermissionDecision, PermissionError>> + Send + 'a>,
            > {
                Box::pin(async { Ok(AsyncPermissionDecision::Deny) })
            }
        }

        let prompter: Arc<dyn AsyncPermissionPrompter> = Arc::new(DenyAsyncPrompter);

        // A 429 on the first provider skips lower-ranked models on that same
        // provider and uses the next different-provider candidate. It must not
        // call either the native client or the main-turn quota fallback.
        let native_calls = Arc::new(AtomicUsize::new(0));
        let quota_fallback_calls = Arc::new(AtomicUsize::new(0));
        let first_calls = Arc::new(AtomicUsize::new(0));
        let same_provider_calls = Arc::new(AtomicUsize::new(0));
        let next_provider_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        )
        .with_async_api_client(Arc::new(CountingAsyncClient {
            calls: Arc::clone(&native_calls),
            outcome: Outcome::Stop,
        }));
        runtime.set_quota_wait_band(std::time::Duration::ZERO);
        runtime.set_quota_fallback_client(Some((
            Arc::new(CountingAsyncClient {
                calls: Arc::clone(&quota_fallback_calls),
                outcome: Outcome::Stop,
            }),
            "gemini-3.5-flash".to_string(),
        )));
        runtime.set_deep_verify_candidates(vec![
            (
                Arc::new(CountingAsyncClient {
                    calls: Arc::clone(&first_calls),
                    outcome: Outcome::RateLimit,
                }),
                "claude-fable-4-5".to_string(),
            ),
            (
                Arc::new(CountingAsyncClient {
                    calls: Arc::clone(&same_provider_calls),
                    outcome: Outcome::Stop,
                }),
                "claude-opus-4-8".to_string(),
            ),
            (
                Arc::new(CountingAsyncClient {
                    calls: Arc::clone(&next_provider_calls),
                    outcome: Outcome::Stop,
                }),
                "gpt-5.6-sol".to_string(),
            ),
        ]);
        let (render_tx, mut render_rx) = mpsc::channel(64);
        let _drain = tokio::spawn(async move { while render_rx.recv().await.is_some() {} });
        let result = runtime
            .verify_subturn(
                "judge the diff".to_string(),
                &render_tx,
                &BlockIdGen::default(),
                &prompter,
            )
            .await;
        assert!(result.is_ok(), "next-ranked verify must succeed: {result:?}");
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(same_provider_calls.load(Ordering::SeqCst), 0);
        assert_eq!(next_provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(native_calls.load(Ordering::SeqCst), 0);
        assert_eq!(quota_fallback_calls.load(Ordering::SeqCst), 0);
        assert!(!runtime.quota_fallback_active);
        assert!(runtime.quota_dry_until.is_none());
        assert!(parse_lens_verifier(&runtime.last_assistant_text()).accepted);
        assert_eq!(
            runtime.deep_verify_succeeded_model_label(),
            Some("gpt-5.6-sol")
        );
        assert!(!runtime.deep_verify_leg_active);

        // If every ranked provider is rate-limited, the same-model native
        // verifier remains the final safety net and its real verdict is used.
        let native_calls = Arc::new(AtomicUsize::new(0));
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        )
        .with_async_api_client(Arc::new(CountingAsyncClient {
            calls: Arc::clone(&native_calls),
            outcome: Outcome::Stop,
        }));
        runtime.set_deep_verify_candidates(vec![
            (
                Arc::new(CountingAsyncClient {
                    calls: Arc::clone(&first_calls),
                    outcome: Outcome::RateLimit,
                }),
                "claude-fable-4-5".to_string(),
            ),
            (
                Arc::new(CountingAsyncClient {
                    calls: Arc::clone(&second_calls),
                    outcome: Outcome::RateLimit,
                }),
                "gpt-5.6-sol".to_string(),
            ),
        ]);
        let (render_tx, mut render_rx) = mpsc::channel(64);
        let _drain = tokio::spawn(async move { while render_rx.recv().await.is_some() {} });
        let result = runtime
            .verify_subturn(
                "judge the diff".to_string(),
                &render_tx,
                &BlockIdGen::default(),
                &prompter,
            )
            .await;
        assert!(result.is_ok(), "native fallback must produce a verdict: {result:?}");
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
        assert_eq!(native_calls.load(Ordering::SeqCst), 1);
        assert!(parse_lens_verifier(&runtime.last_assistant_text()).accepted);
        assert_eq!(runtime.deep_verify_succeeded_model_label(), None);
        assert!(!runtime.deep_verify_leg_active);

        // Architect + implementer-tier session: exhausting the deep pool must
        // fail closed instead of running VERIFY on the implementer.
        let native_calls = Arc::new(AtomicUsize::new(0));
        let deep_calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = ConversationRuntime::new(
            Session::new(),
            NoopApiClient,
            StaticToolExecutor::new(),
            crate::PermissionPolicy::new(PermissionMode::WorkspaceWrite),
            vec!["system".to_string()],
        )
        .with_async_api_client(Arc::new(CountingAsyncClient {
            calls: Arc::clone(&native_calls),
            outcome: Outcome::Stop,
        }));
        runtime.set_context_model("gpt-5.6-terra");
        runtime.set_deep_tier_only(true);
        runtime.set_deep_verify_candidates(vec![(
            Arc::new(CountingAsyncClient {
                calls: Arc::clone(&deep_calls),
                outcome: Outcome::RateLimit,
            }),
            "claude-fable-5".to_string(),
        )]);
        let (render_tx, mut render_rx) = mpsc::channel(64);
        let _drain = tokio::spawn(async move { while render_rx.recv().await.is_some() {} });
        let result = runtime
            .verify_subturn(
                "judge the diff".to_string(),
                &render_tx,
                &BlockIdGen::default(),
                &prompter,
            )
            .await;
        assert!(result.is_err(), "no non-deep native fallback is allowed");
        assert_eq!(deep_calls.load(Ordering::SeqCst), 1);
        assert_eq!(native_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn reactive_verify_uses_same_read_only_allowlist_as_plan_first() {
        // Fix A routes the reactive VERIFY through `deep_subturn(.., ReadOnly, ..)`
        // exactly like plan-first, so the verifier inspects but never mutates.
        // The end-to-end ReadOnly downgrade is exercised by the live integration
        // test; here we pin the invariant that downgrade relies on — the scoped
        // grant `deep_subturn` injects unblocks read-only inspection while every
        // write-class tool stays denied under ReadOnly.
        use crate::permissions::PermissionOutcome;

        let mut policy = crate::PermissionPolicy::new(crate::PermissionMode::ReadOnly)
            .with_tool_requirement("bash", crate::PermissionMode::DangerFullAccess)
            .with_tool_requirement("write_file", crate::PermissionMode::WorkspaceWrite)
            .with_tool_requirement("edit_file", crate::PermissionMode::WorkspaceWrite);
        let grant = policy.add_temporary_allow_rules(read_only_bash_allow_rules());

        // Read-only inspection (git diff) the verifier needs is permitted by the
        // scoped grant `deep_subturn` injects…
        assert_eq!(
            policy.authorize("bash", r#"{"command":"git diff"}"#, None),
            PermissionOutcome::Allow
        );
        // …but the verifier can never edit or delete files.
        for tool in ["write_file", "edit_file"] {
            assert!(
                matches!(
                    policy.authorize(tool, r#"{"path":"a.rs","content":"x"}"#, None),
                    PermissionOutcome::Deny { .. }
                ),
                "{tool} must stay denied during the read-only VERIFY phase"
            );
        }
        policy.remove_temporary_allow_rules(grant);
    }

    /// The objective check must run off the async task (via `spawn_blocking`) so
    /// it never freezes the host's `select!` event loop. We can't easily assert
    /// loop-liveness in this shared unit binary (other tests mutate the global
    /// cwd, racing the subprocess), so the end-to-end non-starvation property is
    /// covered by the isolated integration test
    /// `tests/deep_gate_live.rs::reactive_check_does_not_starve_the_render_loop`.
    /// Here we only confirm the helper still computes the right verdict when run
    /// on the blocking pool.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn command_is_green_runs_on_blocking_pool() {
        assert!(command_is_green("true").await, "`true` exits 0 ⇒ green");
        assert!(
            !command_is_green("false").await,
            "`false` exits 1 ⇒ not green"
        );
        let observed = run_check_command("printf check-output-marker").await;
        assert!(observed.green);
        assert!(observed.output_tail.contains("check-output-marker"));
    }

    /// The deep-gate computes `changed_files_snapshot` on every edit-making
    /// attempt (baseline + after, plus the `TurnEnd` hook context) — twice per
    /// attempt, regardless of whether an objective `check_command` is set. It
    /// spawns a blocking `git diff`, so before the fix it ran synchronously on
    /// the host `select!` task and froze the spinner/stream mid-turn on a large
    /// or index-locked working tree (the reported "도구 사용 중 멈춤"). The async
    /// wrapper must run it off-thread via `spawn_blocking` so the await yields
    /// and the event loop stays live, exactly like `command_is_green` above.
    ///
    /// We can't assert loop-liveness deterministically in this shared unit
    /// binary (other tests mutate the global cwd, racing the subprocess), and a
    /// PATH-shimmed slow `git` would require mutating process-global env in a
    /// multi-threaded test binary — exactly the flakiness this file avoids. The
    /// off-thread guarantee instead rests on two robust facts: (1) every
    /// deep-gate call site now awaits this async wrapper (no sync
    /// `changed_files_snapshot()` remains on an async path — see the call sites
    /// in `run_auto_turn_streaming` / `run_deep_turn_streaming` / the Stop loop),
    /// and (2) the wrapper delegates to `tokio::task::spawn_blocking`, the same
    /// offload `command_is_green` uses (covered live by
    /// `tests/deep_gate_live.rs::reactive_check_does_not_starve_the_render_loop`).
    /// Here we confirm the helper still returns and never panics when driven on
    /// the runtime (a panicking/cancelled blocking task degrades to an empty
    /// snapshot rather than propagating into the turn loop).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn changed_files_snapshot_async_runs_on_blocking_pool() {
        // Whatever the ambient repo state, the call must resolve to a Vec (never
        // hang, never panic) when awaited from the async context.
        let _snapshot: Vec<String> = changed_files_snapshot_async().await;
    }
}
