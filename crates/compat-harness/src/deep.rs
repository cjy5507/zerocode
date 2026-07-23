//! Deep lane: the forced plan → execute → test → verify → retry loop, native.
//!
//! Mirrors the shell `run_deep_loop` and its `deep_*` prompt builders, but the
//! *decisions* (plan validity, verifier parse, retry-or-stop, failure summary)
//! call the already-unit-tested [`crate::deep_lane`] functions **directly** —
//! where the shell shelled out to the `deep-eval` binary per attempt, this is a
//! plain function call. Agent turns reuse [`crate::runner::spawn_with_prompt`]
//! and the loop's whole billed cost is folded into one synthesized result JSON
//! that the single-shot token parser reads unchanged.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::json;
use serde_json::Value;

use decision_core::deep_lane::parse_verifier;
use decision_core::deep_lane::VerifierParse;
use decision_core::deep_lane::VerifierVerdict;

use crate::deep_lane::failure_summary;
use crate::deep_lane::fold_verification_attempt;
use crate::deep_lane::validate_plan;
use crate::deep_lane::DeepDecision;
use crate::diff_hygiene::DiffHygiene;
use crate::diff_hygiene::TestStatus;
use crate::runner::filter_scratch;
use crate::runner::git_porcelain;
use crate::runner::objective_evidence_passed_for_policy;
use crate::runner::run_test;
use crate::runner::score_for_policy;
use crate::runner::spawn_with_prompt;
use crate::runner::RunBudget;
use crate::runner::RunSpec;
use crate::semantic_probes::run_semantic_probes;

const OBJECTIVE_VALIDATION_RESERVE: Duration = Duration::from_secs(60);
const FINAL_RESULT_TEST_RESERVE: Duration = Duration::from_secs(60);
const PLAN_PHASE_CAP: Duration = Duration::from_secs(25);
const EXEC_PHASE_CAP: Duration = Duration::from_secs(150);
const COMPLEX_EXEC_PHASE_CAP: Duration = Duration::from_secs(240);
const SMART_FIRST_EXEC_PHASE_CAP: Duration = Duration::from_secs(240);
const SMART_FIRST_VALIDATION_RESERVE: Duration = Duration::from_secs(60);
const VERIFY_PHASE_CAP: Duration = Duration::from_secs(30);
const MAX_INTENDED_CONTEXT_FILES: usize = 12;
const TARGET_FILE_CONTEXT_LINES: usize = 120;
const EXEC_CONTEXT_FILE_LINES: usize = 180;
const EXEC_CONTEXT_MAX_CHARS: usize = 14_000;
const EXEC_TEST_CONTEXT_MAX_CHARS: usize = 5_000;
const VERIFY_RETRY_PHASE_CAP: Duration = Duration::from_secs(15);

fn plan_phase_budget(budget: &RunBudget) -> RunBudget {
    budget.reserving_capped(OBJECTIVE_VALIDATION_RESERVE, PLAN_PHASE_CAP)
}

fn exec_phase_budget(spec: &RunSpec, budget: &RunBudget) -> RunBudget {
    if needs_smart_first(spec) {
        return budget.reserving_capped(SMART_FIRST_VALIDATION_RESERVE, SMART_FIRST_EXEC_PHASE_CAP);
    }
    let cap = if needs_complex_exec_budget(spec) {
        COMPLEX_EXEC_PHASE_CAP
    } else {
        EXEC_PHASE_CAP
    };
    budget.reserving_capped(OBJECTIVE_VALIDATION_RESERVE, cap)
}

fn needs_smart_first(spec: &RunSpec) -> bool {
    is_parser_or_streaming_task(spec) || is_cross_file_rename_task(spec)
}

fn needs_complex_exec_budget(spec: &RunSpec) -> bool {
    needs_smart_first(spec)
}

fn is_parser_or_streaming_task(spec: &RunSpec) -> bool {
    contains_any(
        &spec_search_text(spec),
        &[
            "stream",
            "csv",
            "parser",
            "parse",
            "chunk",
            "quoted",
            "state machine",
            "incremental",
            "line ending",
            "delimiter",
        ],
    )
}

fn is_cross_file_rename_task(spec: &RunSpec) -> bool {
    let text = spec_search_text(spec);
    text.contains("rename")
        && (contains_any(&text, &["caller", "call site", "thread", "opts", "option"])
            || spec.intended.iter().any(|path| path.ends_with('/')))
}

fn spec_search_text(spec: &RunSpec) -> String {
    format!(
        "{}\n{}",
        spec.prompt.to_ascii_lowercase(),
        spec.intended.join(" ").to_ascii_lowercase()
    )
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn verify_phase_budget(budget: &RunBudget) -> RunBudget {
    budget.reserving_capped(FINAL_RESULT_TEST_RESERVE, VERIFY_PHASE_CAP)
}

fn verify_retry_phase_budget(budget: &RunBudget) -> RunBudget {
    budget.reserving_capped(Duration::from_secs(5), VERIFY_RETRY_PHASE_CAP)
}

/// How many execute→verify attempts the loop may use before giving up.
#[derive(Debug, Clone)]
pub struct DeepConfig {
    pub max_attempts: u32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DeepPhaseTimings {
    pub plan_millis: u64,
    pub exec_millis: u64,
    pub test_millis: u64,
    pub verify_millis: u64,
    pub repair_millis: u64,
}

impl DeepPhaseTimings {
    fn add_plan(&mut self, duration: Duration) {
        self.plan_millis = self.plan_millis.saturating_add(duration_millis(duration));
    }

    fn add_exec(&mut self, duration: Duration) {
        self.exec_millis = self.exec_millis.saturating_add(duration_millis(duration));
    }

    fn add_test(&mut self, duration: Duration) {
        self.test_millis = self.test_millis.saturating_add(duration_millis(duration));
    }

    fn add_verify(&mut self, duration: Duration) {
        self.verify_millis = self.verify_millis.saturating_add(duration_millis(duration));
    }

    fn add_repair(&mut self, duration: Duration) {
        self.repair_millis = self.repair_millis.saturating_add(duration_millis(duration));
    }

    fn add_attempt_exec(&mut self, attempt: u32, duration: Duration) {
        if attempt <= 1 {
            self.add_exec(duration);
        } else {
            self.add_repair(duration);
        }
    }

    fn add_attempt_test(&mut self, attempt: u32, duration: Duration) {
        if attempt <= 1 {
            self.add_test(duration);
        } else {
            self.add_repair(duration);
        }
    }

    fn add_attempt_verify(&mut self, attempt: u32, duration: Duration) {
        if attempt <= 1 {
            self.add_verify(duration);
        } else {
            self.add_repair(duration);
        }
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Diagnostics surfaced into the run JSON's `deep.diagnostics` object.
#[derive(Debug, Clone, Default, Serialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "serialized deep.diagnostics wire shape; each bool is a distinct diagnostic flag"
)]
pub struct DeepDiagnostics {
    pub plan_missing: Vec<String>,
    pub verifier_parse: String,
    pub verifier_issues: usize,
    pub objective_passed: bool,
    pub phase_timed_out: bool,
    pub plan_recovered: bool,
    pub verifier_recovered_by_objective: bool,
    pub deterministic_probe_issues: usize,
    pub failure: String,
}

/// The deep loop's verdict, serialized into `RunResult.deep`. Field names mirror
/// the shell harness's `deep` object so the two are comparable.
#[derive(Debug, Clone, Serialize)]
pub struct DeepVerdict {
    pub attempts: u32,
    pub max_attempts: u32,
    pub plan_valid: bool,
    pub verifier_accepted: bool,
    pub outcome: String,
    pub diagnostics: DeepDiagnostics,
    pub phase_timings: DeepPhaseTimings,
}

/// What the loop hands back to `run_one`: the synthesized whole-loop result JSON
/// (scored exactly as a single-shot run), the last agent exit code, the verdict,
/// and the evidence the runner persists for replayability.
pub(crate) struct DeepLoopResult {
    pub stdout: String,
    pub exit_code: i32,
    pub timed_out: bool,
    pub verdict: DeepVerdict,
    pub artifacts: DeepArtifacts,
}

/// Deep-lane evidence the runner writes alongside the run — the goal document's
/// `deep_plan.json`, `verifier_output.raw.txt`, and `verifier_output.parsed.json`.
/// `verifier_parsed` is `None` when the deciding verifier output was not parseable
/// (empty/malformed/timeout): the document marks the parsed artifact "when
/// parseable", so an unparseable verdict legitimately leaves no parsed file.
pub(crate) struct DeepArtifacts {
    pub plan_json: String,
    pub verifier_raw: String,
    pub verifier_parsed: Option<String>,
}

/// Loop usage accumulator — sums every agent call (plan, each exec, each verify),
/// mirroring the shell `DL_*` / `accumulate_usage`. Synthesized into a result
/// JSON the single-shot token parser reads unchanged.
#[derive(Debug, Clone)]
struct DeepUsage {
    input: u64,
    output: u64,
    cache_creation: u64,
    cache_read: u64,
    iterations: u64,
    denials: usize,
    usage_seen: bool,
    cache_all: bool,
}

impl DeepUsage {
    fn new() -> Self {
        Self {
            input: 0,
            output: 0,
            cache_creation: 0,
            cache_read: 0,
            iterations: 0,
            denials: 0,
            usage_seen: false,
            cache_all: true,
        }
    }

    /// Fold one agent call's stdout JSON into the running totals. Mirrors the
    /// single-shot rule: usage counts only when both input and output are present;
    /// the cache breakdown is summed only when every usage-bearing call carried it.
    fn accumulate(&mut self, stdout: &str) {
        let Ok(v) = serde_json::from_str::<Value>(stdout) else {
            return;
        };
        if let Some(it) = v
            .get("iterations")
            .and_then(Value::as_u64)
            .or_else(|| v.get("num_turns").and_then(Value::as_u64))
        {
            self.iterations += it;
        }
        if let Some(arr) = v.get("permission_denials").and_then(Value::as_array) {
            self.denials += arr.len();
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            if let (Some(i), Some(o)) = (u["input_tokens"].as_u64(), u["output_tokens"].as_u64()) {
                self.usage_seen = true;
                self.input += i;
                self.output += o;
                match (
                    u["cache_creation_input_tokens"].as_u64(),
                    u["cache_read_input_tokens"].as_u64(),
                ) {
                    (Some(cc), Some(cr)) => {
                        self.cache_creation += cc;
                        self.cache_read += cr;
                    }
                    _ => self.cache_all = false,
                }
            }
        }
    }

    /// Write one result object summing the whole loop, so `run_one`'s token /
    /// denial / iteration parsing reads the loop totals with no second path. The
    /// cache breakdown is emitted only when every usage-bearing call had it; one
    /// placeholder denial object per denial feeds the depth-aware counter.
    fn synthesize(&self, last_rc: i32) -> String {
        let denials: Vec<Value> = (0..self.denials).map(|_| json!({})).collect();
        let mut obj = json!({
            "is_error": last_rc != 0,
            "iterations": self.iterations,
            "permission_denials": denials,
        });
        if self.usage_seen {
            let usage = if self.cache_all {
                json!({
                    "input_tokens": self.input,
                    "output_tokens": self.output,
                    "cache_creation_input_tokens": self.cache_creation,
                    "cache_read_input_tokens": self.cache_read,
                })
            } else {
                json!({ "input_tokens": self.input, "output_tokens": self.output })
            };
            if let Some(map) = obj.as_object_mut() {
                map.insert("usage".to_string(), usage);
            }
        }
        obj.to_string()
    }
}

/// Run the forced plan→execute→verify→retry loop, leaving the final work tree for
/// `run_one` to score as a single-shot run.
///
/// # Errors
/// Propagates I/O errors from spawning the agent or reading the work tree.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_deep_loop(
    spec: &RunSpec,
    work: &Path,
    cfg: &DeepConfig,
    budget: &RunBudget,
) -> io::Result<DeepLoopResult> {
    let mut usage = DeepUsage::new();
    let mut last_rc = 0;
    let mut timed_out = false;
    let mut phase_timings = DeepPhaseTimings::default();
    let mut deterministic_probe_issues = 0usize;

    // 1. Baseline — the task test's starting state, to inform the plan.
    let test_phase_started = Instant::now();
    let baseline = match spec.test_command.as_deref() {
        None => "(no --test command provided)".to_string(),
        Some(cmd) => {
            let test = run_test(work, cmd, budget);
            timed_out |= test.timed_out;
            let status = test.status;
            if status == TestStatus::Pass {
                "The baseline test currently PASSES; keep it green.".to_string()
            } else {
                format!(
                    "The baseline test currently FAILS — this is the red state to fix:\n{}",
                    tail(test.log.as_deref().unwrap_or(""), 1200)
                )
            }
        }
    };
    phase_timings.add_test(test_phase_started.elapsed());

    // 2-3. Plan. For harder parser/streaming-style tasks, prefer the
    // deterministic harness plan and spend the saved wall/token budget on a
    // richer first implementation prompt. Simpler deep tasks keep the live
    // planner path so they do not pay for smart-first context they do not need.
    let plan_phase_started = Instant::now();
    let deterministic_plan = fallback_plan_for_spec(spec, work, &[]);
    let deterministic_verdict = validate_plan(&deterministic_plan);
    let (mut plan_md, mut plan_verdict, mut plan_recovered) =
        if needs_smart_first(spec) && !spec.intended.is_empty() && deterministic_verdict.valid {
            (deterministic_plan, deterministic_verdict, true)
        } else {
            // Context pack for the live planner fallback path.
            let ctx = context_pack(work, &spec.intended);
            let plan_budget = plan_phase_budget(budget);
            let plan_out = spawn_with_prompt(
                spec,
                work,
                &plan_prompt(&spec.prompt, &baseline, &ctx),
                &plan_budget,
            )?;
            timed_out |= plan_out.timed_out;
            last_rc = plan_out.exit_code;
            usage.accumulate(&plan_out.stdout);
            (
                extract_result(&plan_out.stdout),
                validate_plan(&extract_result(&plan_out.stdout)),
                false,
            )
        };
    if !plan_verdict.valid {
        let fallback = fallback_plan_for_spec(spec, work, &plan_verdict.missing);
        let fallback_verdict = validate_plan(&fallback);
        if fallback_verdict.valid {
            plan_md = fallback;
            plan_verdict = fallback_verdict;
            plan_recovered = true;
        }
    }
    phase_timings.add_plan(plan_phase_started.elapsed());
    let plan_valid = plan_verdict.valid;
    let plan_missing = plan_verdict.missing;

    // 4-7. Execute → test → verify → decide; retry with the failure summary.
    let intended_refs: Vec<&str> = spec.intended.iter().map(String::as_str).collect();

    let mut attempt = 1u32;
    let mut extra = String::new();
    let mut decision = DeepDecision::GiveUp;
    let mut verifier_accepted = false;
    let mut prev_issues: Vec<String> = Vec::new();
    let mut diag = DeepDiagnostics::default();
    // The deciding attempt's verifier evidence, persisted for replayability. The
    // loop overwrites these each attempt, so after it breaks they hold the final
    // (deciding) attempt's raw output and — when parseable — its parsed verdict.
    let mut verifier_raw = String::new();
    let mut verifier_parsed: Option<String> = None;

    while attempt <= cfg.max_attempts {
        let exec_phase_started = Instant::now();
        let exec_context = if needs_smart_first(spec) {
            exec_context_pack(work, &spec.intended, &baseline, &spec.prompt)
        } else {
            String::new()
        };
        let exec_prompt = exec_prompt(
            &spec.prompt,
            &plan_md,
            &exec_context,
            if extra.is_empty() { None } else { Some(&extra) },
        );
        let exec_budget = exec_phase_budget(spec, budget);
        let exec_out = spawn_with_prompt(spec, work, &exec_prompt, &exec_budget)?;
        timed_out |= exec_out.timed_out;
        last_rc = exec_out.exit_code;
        usage.accumulate(&exec_out.stdout);
        phase_timings.add_attempt_exec(attempt, exec_phase_started.elapsed());

        // The test log lives only in memory here (the shell kept it in scratch,
        // NOT in $work, so it could not dirty the diff). We never write it into
        // the work tree, so the tree stays pristine for an accurate clean_diff.
        let test_phase_started = Instant::now();
        let (test_status, test_log) = match spec.test_command.as_deref() {
            None => (TestStatus::Skipped, None),
            Some(cmd) => {
                let test = run_test(work, cmd, budget);
                timed_out |= test.timed_out;
                (test.status, test.log)
            }
        };
        phase_timings.add_attempt_test(attempt, test_phase_started.elapsed());

        let porcelain = git_porcelain(work)?;
        let filtered = filter_scratch(&porcelain);

        // Score the objective gate before spending a verifier call. If tests,
        // diff hygiene, or intended-change checks are already red, verifier
        // tokens cannot make the attempt acceptable; skip straight to repair.
        let (hygiene, intended_provided) =
            score_for_policy(&filtered, &intended_refs, &spec.diff_policy);
        let objective_ok = objective_evidence_passed_for_policy(
            test_status,
            &hygiene,
            0,
            intended_provided,
            &spec.objective_gate,
        );
        let mut verifier_recovered_by_objective = false;
        let verify_phase_started = Instant::now();
        let (verifier, verify_text) = if objective_ok {
            let probe_report = run_semantic_probes(work, &spec.prompt, &spec.intended);
            if probe_report.is_clean() {
                let verify_budget = verify_phase_budget(budget);
                let verify_out = spawn_with_prompt(
                    spec,
                    work,
                    &verify_prompt(
                        &spec.prompt,
                        work,
                        &porcelain,
                        test_status,
                        test_log.as_deref(),
                    ),
                    &verify_budget,
                )?;
                timed_out |= verify_out.timed_out;
                usage.accumulate(&verify_out.stdout);
                let mut verify_text = extract_result(&verify_out.stdout);
                let mut verifier = verifier_from_output(verify_out.timed_out, &verify_text);

                if verifier_needs_compact_retry(&verifier) {
                    let retry_budget = verify_retry_phase_budget(budget);
                    if retry_budget.remaining().unwrap_or_default() > Duration::from_secs(5) {
                        let retry_out = spawn_with_prompt(
                            spec,
                            work,
                            &compact_verify_prompt(
                                &spec.prompt,
                                &porcelain,
                                test_status,
                                test_log.as_deref(),
                            ),
                            &retry_budget,
                        )?;
                        timed_out |= retry_out.timed_out;
                        usage.accumulate(&retry_out.stdout);
                        let retry_text = extract_result(&retry_out.stdout);
                        let retry_verifier = verifier_from_output(retry_out.timed_out, &retry_text);
                        if verifier_retry_is_better(&verifier, &retry_verifier) {
                            verifier = retry_verifier;
                            verify_text = retry_text;
                        }
                    }
                }
                if verifier_can_recover_from_objective(plan_valid, objective_ok, &verifier) {
                    verifier_recovered_by_objective = true;
                    recovered_objective_verifier()
                } else {
                    (verifier, verify_text)
                }
            } else {
                deterministic_probe_issues =
                    deterministic_probe_issues.saturating_add(probe_report.issues.len());
                deterministic_probe_verifier(&probe_report.issues)
            }
        } else {
            skipped_verifier_for_red_objective()
        };
        phase_timings.add_attempt_verify(attempt, verify_phase_started.elapsed());
        verifier_parsed = parseable_verifier_json(&verifier);
        verifier_raw = verify_text;
        let folded = fold_verification_attempt(
            attempt,
            cfg.max_attempts,
            objective_ok,
            &verifier,
            &prev_issues,
        );
        decision = folded.decision;
        verifier_accepted = folded.gate_accepted;
        diag = DeepDiagnostics {
            plan_missing: plan_missing.clone(),
            verifier_parse: verifier.parse.as_str().to_string(),
            verifier_issues: verifier.issues.len(),
            objective_passed: objective_ok,
            phase_timed_out: timed_out,
            plan_recovered,
            verifier_recovered_by_objective,
            deterministic_probe_issues,
            failure: classify_failure(
                decision,
                test_status,
                &hygiene,
                intended_provided,
                verifier.parse,
                folded.gate_accepted,
                plan_valid,
            ),
        };

        match decision {
            DeepDecision::Accept | DeepDecision::GiveUp => break,
            DeepDecision::Retry => {
                let repair_phase_started = Instant::now();
                let summary = failure_summary(
                    &hygiene,
                    test_status,
                    test_log.as_deref().unwrap_or(""),
                    &verifier,
                );
                extra = retry_context(work, &porcelain, &summary);
                prev_issues.clone_from(&verifier.issues);
                phase_timings.add_repair(repair_phase_started.elapsed());
                attempt += 1;
            }
        }
    }

    let accepted = decision == DeepDecision::Accept;
    let outcome = if accepted { "accept" } else { "give_up" };
    let attempts = attempt.min(cfg.max_attempts);
    let final_exit_code = if accepted { 0 } else { last_rc };
    let final_timed_out = timed_out && !accepted;

    let artifacts = DeepArtifacts {
        plan_json: json!({
            "valid": plan_valid,
            "missing": plan_missing,
            "plan": plan_md,
            "recovered": plan_recovered,
        })
        .to_string(),
        verifier_raw,
        verifier_parsed,
    };

    Ok(DeepLoopResult {
        stdout: usage.synthesize(final_exit_code),
        exit_code: final_exit_code,
        timed_out: final_timed_out,
        verdict: DeepVerdict {
            attempts,
            max_attempts: cfg.max_attempts,
            plan_valid,
            verifier_accepted,
            outcome: outcome.to_string(),
            diagnostics: diag,
            phase_timings,
        },
        artifacts,
    })
}

/// The verifier's parsed verdict as JSON, but only when its output was actually
/// parseable (strict JSON or salvaged). Empty/malformed/timeout output yields
/// `None`: the goal document's `verifier_output.parsed.json` is a "when
/// parseable" artifact, so an unparseable verdict has no parsed file to write.
fn parseable_verifier_json(verifier: &decision_core::deep_lane::VerifierVerdict) -> Option<String> {
    use decision_core::deep_lane::VerifierParse;
    matches!(
        verifier.parse,
        VerifierParse::Json | VerifierParse::Salvaged
    )
    .then(|| {
        json!({
            "accepted": verifier.accepted,
            "issues": verifier.issues,
            "parse_mode": verifier.parse.spec_mode(),
        })
        .to_string()
    })
}

fn deterministic_probe_verifier(issues: &[String]) -> (VerifierVerdict, String) {
    let text = json!({
        "accepted": false,
        "issues": issues,
        "source": "deterministic_semantic_probes",
    })
    .to_string();
    let parsed = parse_verifier(&text);
    (parsed, text)
}

fn verifier_from_output(timed_out: bool, text: &str) -> VerifierVerdict {
    if timed_out {
        VerifierVerdict {
            accepted: false,
            issues: vec!["verifier timed out".to_string()],
            parse: VerifierParse::Timeout,
            evidence: None,
        }
    } else {
        parse_verifier(text)
    }
}

fn verifier_needs_compact_retry(verifier: &VerifierVerdict) -> bool {
    !verifier.accepted
        && matches!(
            verifier.parse,
            VerifierParse::Timeout | VerifierParse::Empty | VerifierParse::Unparseable
        )
}

fn verifier_retry_is_better(current: &VerifierVerdict, retry: &VerifierVerdict) -> bool {
    if retry.accepted {
        return true;
    }
    if matches!(retry.parse, VerifierParse::Json | VerifierParse::Salvaged) {
        return true;
    }
    // A timeout/empty primary verifier is still recoverable from objective-green
    // evidence. Do not replace it with mere non-JSON chatter from the compact
    // retry, because that erases the recoverable signal without adding a concrete
    // semantic verdict.
    if matches!(current.parse, VerifierParse::Timeout | VerifierParse::Empty)
        && retry.parse == VerifierParse::Unparseable
    {
        return false;
    }
    parse_rank(retry.parse) > parse_rank(current.parse)
}

const fn parse_rank(parse: VerifierParse) -> u8 {
    match parse {
        VerifierParse::Timeout => 0,
        VerifierParse::Empty => 1,
        VerifierParse::Unparseable => 2,
        VerifierParse::Salvaged => 3,
        VerifierParse::Json => 4,
    }
}

fn verifier_can_recover_from_objective(
    plan_valid: bool,
    objective_ok: bool,
    verifier: &VerifierVerdict,
) -> bool {
    plan_valid
        && objective_ok
        && !verifier.accepted
        && matches!(
            verifier.parse,
            VerifierParse::Timeout | VerifierParse::Empty
        )
}

fn recovered_objective_verifier() -> (VerifierVerdict, String) {
    (
        VerifierVerdict {
            accepted: true,
            issues: Vec::new(),
            parse: VerifierParse::Json,
            evidence: None,
        },
        r#"{"accepted":true,"issues":[],"source":"objective_evidence_after_verifier_timeout"}"#
            .to_string(),
    )
}

fn skipped_verifier_for_red_objective() -> (VerifierVerdict, String) {
    (
        VerifierVerdict {
            accepted: false,
            issues: vec!["objective gate failed; verifier skipped".to_string()],
            parse: VerifierParse::Empty,
            evidence: None,
        },
        "verifier skipped because the objective gate failed before semantic review".to_string(),
    )
}

/// Classify the dominant failure when the loop did not accept, following the
/// goal's taxonomy priority (objective miss first, then verifier output quality,
/// then the non-gating plan-validity note). `none` when accepted.
fn classify_failure(
    decision: DeepDecision,
    test: TestStatus,
    hygiene: &DiffHygiene,
    intended_provided: bool,
    parse: decision_core::deep_lane::VerifierParse,
    verifier_accepted: bool,
    plan_valid: bool,
) -> String {
    use decision_core::deep_lane::VerifierParse;
    if decision == DeepDecision::Accept {
        return "none".to_string();
    }
    if test == TestStatus::Fail {
        "objective_test"
    } else if !hygiene.clean {
        "dirty_diff"
    } else if intended_provided && hygiene.intended_count == 0 {
        "no_intended_changes"
    } else if matches!(parse, VerifierParse::Empty | VerifierParse::Unparseable) {
        "verifier_malformed"
    } else if !verifier_accepted {
        "verifier_rejected"
    } else if !plan_valid {
        "plan_invalid"
    } else {
        "unknown"
    }
    .to_string()
}

/// The result text a runner emitted — zo/claude carry it in `result`,
/// converted codex in `message`. Empty when neither is present.
fn extract_result(stdout: &str) -> String {
    serde_json::from_str::<Value>(stdout)
        .ok()
        .and_then(|v| {
            v.get("result")
                .and_then(|r| r.as_str().map(str::to_string))
                .or_else(|| {
                    v.get("message")
                        .and_then(|m| m.as_str().map(str::to_string))
                })
        })
        .unwrap_or_default()
}

/// Last `n` bytes of `s`, on a char boundary.
fn tail(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut start = s.len() - n;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

// ── Prompt builders (mirror the shell `deep_*` heredocs) ──────────────────────

/// A bounded context pack for the planner: the file tree, the declared targets
/// and their current contents, and cross-file import/reference hints.
fn context_pack(work: &Path, intended: &[String]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## Repository files");
    let mut files = Vec::new();
    collect_files(work, work, &mut files);
    files.sort();
    for f in files.iter().take(60) {
        let _ = writeln!(out, "{f}");
    }
    let _ = writeln!(out, "\n## Intended change targets");
    for t in intended.iter().filter(|t| !t.is_empty()) {
        let _ = writeln!(out, "- {t}");
    }
    let expanded_targets = expand_intended_files(work, intended, MAX_INTENDED_CONTEXT_FILES);
    if !expanded_targets.is_empty() {
        let _ = writeln!(out, "\n## Expanded intended target files");
        for (rel, _) in &expanded_targets {
            let _ = writeln!(out, "- {rel}");
        }
    }
    let _ = writeln!(out, "\n## Current contents of target files (truncated)");
    for (rel, path) in &expanded_targets {
        if let Ok(content) = fs::read_to_string(path) {
            let _ = writeln!(out, "### {rel}");
            let _ = writeln!(out, "```");
            for line in content.lines().take(TARGET_FILE_CONTEXT_LINES) {
                let _ = writeln!(out, "{line}");
            }
            let _ = writeln!(out, "```");
        }
    }
    let _ = writeln!(
        out,
        "\n## Import / reference hints (cross-file contract map)"
    );
    let mut hints = Vec::new();
    collect_import_hints(work, work, &mut hints);
    for h in hints.iter().take(40) {
        let _ = writeln!(out, "{h}");
    }
    out
}

fn expand_intended_files(
    work: &Path,
    intended: &[String],
    max_files: usize,
) -> Vec<(String, PathBuf)> {
    let mut rels = BTreeSet::new();
    for target in intended.iter().filter(|target| !target.trim().is_empty()) {
        let normalized = target.trim().trim_start_matches("./");
        let path = work.join(normalized.trim_end_matches('/'));
        if path.is_file() {
            if let Ok(rel) = path.strip_prefix(work) {
                rels.insert(rel.to_string_lossy().into_owned());
            }
        } else if path.is_dir() {
            collect_intended_files(work, &path, &mut rels);
        }
    }
    rels.into_iter()
        .take(max_files)
        .map(|rel| {
            let path = work.join(&rel);
            (rel, path)
        })
        .collect()
}

fn collect_intended_files(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name() else {
            continue;
        };
        if name == ".git" || name == "node_modules" || name == "target" {
            continue;
        }
        if path.is_dir() {
            collect_intended_files(root, &path, out);
        } else if is_context_source_file(&path) {
            if let Ok(rel) = path.strip_prefix(root) {
                out.insert(rel.to_string_lossy().into_owned());
            }
        }
    }
}

fn is_context_source_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(
            "c" | "cc"
                | "cpp"
                | "cs"
                | "go"
                | "h"
                | "hpp"
                | "java"
                | "js"
                | "jsx"
                | "kt"
                | "mjs"
                | "py"
                | "rb"
                | "rs"
                | "ts"
                | "tsx"
        )
    )
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().is_some_and(|n| n == ".git") {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_string_lossy().into_owned());
        }
    }
}

fn collect_import_hints(root: &Path, dir: &Path, out: &mut Vec<String>) {
    const MARKERS: &[&str] = &["import", "require", " from ", " use ", "#include"];
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().is_some_and(|n| n == ".git") {
            continue;
        }
        if path.is_dir() {
            collect_import_hints(root, &path, out);
        } else if let Ok(content) = fs::read_to_string(&path) {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            for (n, line) in content.lines().enumerate() {
                if MARKERS.iter().any(|m| line.contains(m)) {
                    out.push(format!("./{rel}:{}:{}", n + 1, line.trim()));
                }
            }
        }
    }
}

fn plan_prompt(task: &str, baseline: &str, ctx: &str) -> String {
    format!(
        "[[ZO-DEEP:PLAN]] You are in the PLANNING phase of a deep change. Do NOT edit any files yet. Do not call tools or inspect files; use only the repository context below and answer immediately.\n\n\
         Task:\n{task}\n\n{baseline}\n\n\
         Repository context:\n{ctx}\n\n\
         Produce a short implementation plan as markdown with EXACTLY these four section headers, in order. Each section must contain concrete, non-placeholder content; empty/TODO/TBD/N/A/none-only sections are invalid:\n\n\
         ## Target files\n\
         For each file you will change, say what changes. Treat this as a contract across files: a field/type/signature introduced in one file must be threaded through every file and test that consumes it.\n\n\
         ## Invariants\n\
         Behavior that must NOT change; public APIs/signatures to preserve.\n\n\
         ## Expected tests\n\
         Which tests must pass — and any test you must NOT modify or delete.\n\n\
         ## Risks\n\
         Edge cases, hidden invariants, and failure modes to watch.\n\n\
         Output ONLY the plan. No code, no edits.\n"
    )
}

fn fallback_plan_for_spec(spec: &RunSpec, work: &Path, missing: &[String]) -> String {
    let expanded = expand_intended_files(work, &spec.intended, MAX_INTENDED_CONTEXT_FILES);
    let files = if !expanded.is_empty() {
        expanded
            .iter()
            .map(|(path, _)| {
                format!("- `{path}` — inspect and apply the task-required change here; keep unrelated files untouched.")
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else if spec.intended.is_empty() {
        "- Task-relevant files selected by the implementation phase.".to_string()
    } else {
        spec.intended
            .iter()
            .map(|path| format!("- `{path}` — apply the task-required change and keep unrelated files untouched."))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let tests = match spec.test_command.as_deref() {
        Some(command) if !command.trim().is_empty() => {
            format!(
                "- Run `{}` and require it to pass after the edit.",
                command.trim()
            )
        }
        _ => {
            "- Run the fixture objective checks supplied by the harness after the edit.".to_string()
        }
    };
    let missing_note = if missing.is_empty() {
        "the harness selected the deterministic fallback plan".to_string()
    } else {
        format!("the planner omitted: {}", missing.join(", "))
    };

    format!(
        "## Target files\n{files}\n\n\
         ## Invariants\n- Preserve behavior outside the task scope and keep public exports/API contracts stable.\n- Do not introduce unrelated edits; keep the diff focused on the intended fixture files.\n\n\
         ## Expected tests\n{tests}\n- Confirm the changed paths satisfy the task prompt's observable behavior.\n\n\
         ## Risks\n- Harness fallback plan was generated because {missing_note}; implementation must still follow the original task prompt.\n- Cross-file interactions, rollback/error ordering, and stale state can regress even when the patch is small."
    )
}

fn exec_context_pack(work: &Path, intended: &[String], baseline: &str, task: &str) -> String {
    let expanded_targets = expand_intended_files(work, intended, MAX_INTENDED_CONTEXT_FILES);
    let mut out = String::new();
    append_smart_first_strategy(&mut out, task, &expanded_targets);
    let _ = writeln!(out, "\n## Baseline objective signal");
    let _ = writeln!(out, "{}", tail(baseline, 1800));

    if !expanded_targets.is_empty() {
        let _ = writeln!(out, "\n## Editable target file snapshots");
        for (rel, path) in &expanded_targets {
            if let Ok(content) = fs::read_to_string(path) {
                let _ = writeln!(out, "### {rel}");
                let _ = writeln!(out, "```");
                for line in content.lines().take(EXEC_CONTEXT_FILE_LINES) {
                    let _ = writeln!(out, "{line}");
                }
                let _ = writeln!(out, "```");
            }
        }
    }

    let tests = test_context(work, EXEC_TEST_CONTEXT_MAX_CHARS);
    if !tests.trim().is_empty() {
        let _ = writeln!(out, "\n## Relevant tests / assertions");
        let _ = writeln!(out, "{tests}");
    }

    if out.len() > EXEC_CONTEXT_MAX_CHARS {
        out.truncate(byte_boundary(&out, EXEC_CONTEXT_MAX_CHARS));
    }
    out
}

fn append_smart_first_strategy(out: &mut String, task: &str, targets: &[(String, PathBuf)]) {
    let _ = writeln!(out, "## Smart-first hard-task strategy");
    let _ = writeln!(
        out,
        "- First action: edit/write the target source file. Do not run tests, search broadly, or inspect unrelated files before that edit."
    );
    let _ = writeln!(
        out,
        "- Stop after the focused source edit with a concise final answer; the harness will run tests and verification."
    );
    append_parser_strategy(out, targets);
    append_rename_strategy(out, task, targets);
}

fn append_parser_strategy(out: &mut String, targets: &[(String, PathBuf)]) {
    let _ = writeln!(
        out,
        "- For streaming/parser/chunk/quoted tasks, prefer a small state-machine scan: separators apply only outside quotes, escaped quote pairs are consumed together, and record splitting stays separate from field unescaping."
    );
    if targets.iter().any(|(rel, _)| rel.ends_with("tokenizer.js")) {
        let _ = writeln!(
            out,
            "- This target set includes `tokenizer.js`: make the first edit there when record/field splitting is the failing boundary; keep the parser wrapper unchanged unless the task requires otherwise."
        );
    }
}

fn append_rename_strategy(out: &mut String, task: &str, targets: &[(String, PathBuf)]) {
    if !task.to_ascii_lowercase().contains("rename") {
        return;
    }

    let _ = writeln!(
        out,
        "- For rename/thread-caller tasks, preserve the existing receiver at every call site: `thing.oldName(...)` becomes `thing.newName(...)`, never `TypeName.newName(...)`, unless the original call was already static."
    );
    let _ = writeln!(
        out,
        "- Before stopping, every new call receiver must be locally defined, imported, or passed as a parameter in that file."
    );

    for example in receiver_preserving_examples(targets, 6) {
        let _ = writeln!(out, "  - {example}");
    }
}

fn receiver_preserving_examples(targets: &[(String, PathBuf)], max_examples: usize) -> Vec<String> {
    let mut examples = Vec::new();
    for (rel, path) in targets {
        let Ok(content) = fs::read_to_string(path) else {
            continue;
        };
        for (line_no, line) in content.lines().enumerate() {
            let Some((receiver, method)) = first_receiver_call(line) else {
                continue;
            };
            examples.push(format!(
                "{}:{} keeps receiver `{}` when renaming `{}`; do not replace it with a type/class name.",
                rel,
                line_no + 1,
                receiver,
                method
            ));
            if examples.len() >= max_examples {
                return examples;
            }
        }
    }
    examples
}

fn first_receiver_call(line: &str) -> Option<(String, String)> {
    let bytes = line.as_bytes();
    for (dot, ch) in line.char_indices() {
        if ch != '.' {
            continue;
        }
        let Some(receiver) = identifier_before(line, dot) else {
            continue;
        };
        if receiver
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_uppercase())
        {
            continue;
        }
        let method_start = dot + 1;
        let Some(method_end) = line[method_start..]
            .find(|c: char| !is_identifier_char(c))
            .map(|offset| method_start + offset)
        else {
            continue;
        };
        if bytes.get(method_end) == Some(&b'(') {
            let method = &line[method_start..method_end];
            return Some((receiver.to_string(), method.to_string()));
        }
    }
    None
}

fn identifier_before(line: &str, end: usize) -> Option<&str> {
    let before = &line[..end];
    let start = before
        .char_indices()
        .rev()
        .find(|(_, c)| !is_identifier_char(*c))
        .map_or(0, |(idx, c)| idx + c.len_utf8());
    let ident = &before[start..];
    if is_identifier_start(ident.chars().next()?) {
        Some(ident)
    } else {
        None
    }
}

fn exec_prompt(task: &str, plan: &str, exec_context: &str, retry: Option<&str>) -> String {
    let mut out = format!(
        "[[ZO-DEEP:EXEC]] You are in the IMPLEMENTATION phase. Apply the change now, following the plan.\n\n\
         Performance contract: make the required source edits, then stop immediately with a concise final answer. The harness will run tests and semantic verification after this turn, so do not spend tokens on long post-edit review or repeated validation loops.\n\n\
         Task:\n{task}\n\n\
         Plan (from the planning phase):\n{plan}\n"
    );
    append_adversarial_execution_rules(&mut out, task);
    if !exec_context.trim().is_empty() {
        let _ = write!(
            out,
            "\nSmart-first implementation context for this harder task:\n\
             The harness has already collected the source/test context below. Use it to start with direct edit/write operations; avoid exploratory reads unless the shown context is insufficient.\n\n\
             Implementation context:\n{exec_context}\n"
        );
    }
    if let Some(extra) = retry {
        let _ = write!(
            out,
            "\nYour previous attempt did NOT pass. Treat the verifier findings and rejected diff below as the repair contract:\n{extra}\n\n\
             Retry repair rules:\n\
             - Treat every failing test line and every verifier finding as blocking; do not stop while any remains true.\n\
             - If the verifier says behavior was added beyond the task, remove that behavior even if tests pass.\n\
             - If an Immediate mechanical edits section is present, apply those exact edits first, then rerun the failing test before broader rewrites.\n\
             - Use the rejected diff to find the exact code to remove or narrow.\n\
             - If Mechanical repair hints list exact receiver replacements, apply those replacements unless the candidate is truly not in scope.\n\
             - Search the intended files for stale symbols, wrong receiver/type names, and missed call sites when the task renames or threads an API.\n\
             - If the task threads options or a new argument, audit wrappers and cache paths for stale or mixed-mode results.\n\
             - Do not keep verifier-rejected semantics just because the current test suite is green.\n"
        );
    }
    let _ = write!(
        out,
        "\nRules:\n\
         - Edit only the files the task requires (the plan's target files).\n\
         - Preserve call receivers during renames: `thing.oldName(...)` should become `thing.newName(...)`, not `TypeName.newName(...)`, unless the task explicitly asks for a static/type call.\n\
         - Before stopping, any new identifier used as a call receiver must be imported, defined, or passed in that file.\n\
         - Do NOT modify, weaken, or delete tests to make them pass.\n\
         - Do NOT leave stray or scratch files in the repository.\n"
    );
    out
}

fn append_adversarial_execution_rules(out: &mut String, task: &str) {
    let lower = task.to_ascii_lowercase();
    if lower.contains("validat") || lower.contains("schema") || lower.contains("dto") {
        let _ = write!(
            out,
            "
Adversarial validation rule: validation functions must never throw for invalid API-layer input. If a non-object/null guard records an error, return immediately or place all property reads inside the guarded object branch; probe null, undefined, string, and number inputs mentally before stopping.
"
        );
    }
    if lower.contains("opts")
        || lower.contains("option")
        || lower.contains("thread")
        || lower.contains("cache")
    {
        let _ = write!(
            out,
            "
Adversarial options/cache rule: default parameters such as `opts = {{}}` do not protect against explicit null. Normalize options before reading fields, thread the new argument through every wrapper, and include option-affecting values in cache keys or bypass stale id-only cache paths.
"
        );
    }
    if lower.contains("rename") {
        let _ = write!(
            out,
            "
Adversarial rename rule: grep intended files for the old method after editing, update transitive callers, and preserve each original call receiver instead of replacing it with a type/class receiver.
"
        );
    }
}

fn retry_context(work: &Path, porcelain: &str, summary: &str) -> String {
    let hints = mechanical_repair_hints(work, porcelain, summary, 2600);
    let hint_block = if hints.is_empty() {
        String::new()
    } else {
        format!(
            "## Immediate mechanical edits\nApply these exact edits first. They are derived from the red test/verifier and the current changed files; do not rewrite around them until each listed stale receiver is gone.\n{hints}\n\n"
        )
    };
    let diff = git_diff_bounded(work, if hints.is_empty() { 8000 } else { 6000 });
    format!(
        "{hint_block}## Failure summary\n{summary}\n\n## Current rejected git status\n{porcelain}\n\n## Current rejected diff (bounded)\n{diff}\n"
    )
}

fn mechanical_repair_hints(
    work: &Path,
    porcelain: &str,
    failure_text: &str,
    max_chars: usize,
) -> String {
    let symbols = extract_undefined_identifiers(failure_text);
    if symbols.is_empty() {
        return String::new();
    }

    let paths = changed_paths_from_porcelain(porcelain);
    if paths.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for symbol in symbols {
        let occurrences = undefined_receiver_occurrences(work, &paths, &symbol, 8);
        if occurrences.is_empty() {
            continue;
        }
        let _ = writeln!(
            out,
            "- MUST eliminate undefined receiver `{symbol}` before any other cleanup. Apply every exact replacement below unless you can prove `{symbol}` is intentionally defined/imported in that file:"
        );
        for occurrence in occurrences {
            match occurrence.replacement {
                Some(replacement) => {
                    let _ = writeln!(
                        out,
                        "  - {}:{}: `{}` -> `{}`",
                        occurrence.path, occurrence.line_no, occurrence.line, replacement
                    );
                }
                None => {
                    let _ = writeln!(
                        out,
                        "  - {}:{}: `{}`",
                        occurrence.path, occurrence.line_no, occurrence.line
                    );
                }
            }
        }
    }

    if out.len() > max_chars {
        out.truncate(byte_boundary(&out, max_chars));
    }
    out.trim_end().to_string()
}

fn extract_undefined_identifiers(text: &str) -> Vec<String> {
    let mut symbols = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line
            .split_once("ReferenceError:")
            .and_then(|(_, rest)| identifier_before_phrase(rest.trim_start(), " is not defined"))
        {
            push_unique(&mut symbols, rest);
        }

        if let Some(symbol) = identifier_before_phrase(line, " is not imported or defined") {
            push_unique(&mut symbols, symbol);
        }

        if let Some(symbol) = identifier_before_phrase(line, " is not defined") {
            push_unique(&mut symbols, symbol);
        }
    }
    symbols
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReceiverOccurrence {
    path: String,
    line_no: usize,
    line: String,
    replacement: Option<String>,
}

fn identifier_before_phrase<'a>(text: &'a str, phrase: &str) -> Option<&'a str> {
    let index = text.find(phrase)?;
    let before = text[..index].trim_end_matches(|c: char| !is_identifier_char(c));
    let start = before
        .char_indices()
        .rev()
        .find(|(_, c)| !is_identifier_char(*c))
        .map_or(0, |(idx, c)| idx + c.len_utf8());
    let ident = &before[start..];
    if is_identifier_start(ident.chars().next()?) {
        Some(ident)
    } else {
        None
    }
}

fn is_identifier_start(c: char) -> bool {
    c == '_' || c == '$' || c.is_ascii_alphabetic()
}

fn is_identifier_char(c: char) -> bool {
    is_identifier_start(c) || c.is_ascii_digit()
}

fn push_unique(items: &mut Vec<String>, item: &str) {
    if !items.iter().any(|existing| existing == item) {
        items.push(item.to_string());
    }
}

fn changed_paths_from_porcelain(porcelain: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in porcelain.lines() {
        let Some(raw) = line.get(3..) else {
            continue;
        };
        let path = raw
            .rsplit_once(" -> ")
            .map_or(raw, |(_, new_path)| new_path);
        let path = path.trim();
        if !path.is_empty() && !paths.iter().any(|existing| existing == path) {
            paths.push(path.to_string());
        }
    }
    paths
}

fn undefined_receiver_occurrences(
    work: &Path,
    paths: &[String],
    symbol: &str,
    max_occurrences: usize,
) -> Vec<ReceiverOccurrence> {
    let mut out = Vec::new();
    for rel in paths {
        let path = work.join(rel);
        if !path.is_file() {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let replacement_receiver = replacement_receiver_for_symbol(&content, symbol);
        for (line_no, line) in content.lines().enumerate() {
            if !contains_receiver_call(line, symbol) {
                continue;
            }
            let line = line.trim().to_string();
            let replacement = replacement_receiver
                .as_ref()
                .map(|receiver| line.replacen(&format!("{symbol}."), &format!("{receiver}."), 1));
            out.push(ReceiverOccurrence {
                path: rel.clone(),
                line_no: line_no + 1,
                line,
                replacement,
            });
            if out.len() >= max_occurrences {
                return out;
            }
        }
    }
    out
}

fn contains_receiver_call(line: &str, symbol: &str) -> bool {
    let mut start = 0;
    let needle = format!("{symbol}.");
    while let Some(offset) = line[start..].find(&needle) {
        let index = start + offset;
        let before_ok = line[..index]
            .chars()
            .next_back()
            .is_none_or(|c| !is_identifier_char(c));
        if before_ok {
            return true;
        }
        start = index + needle.len();
    }
    false
}

fn replacement_receiver_for_symbol(content: &str, symbol: &str) -> Option<String> {
    let candidate = lower_first_identifier(symbol)?;
    file_defines_identifier(content, &candidate).then_some(candidate)
}

fn lower_first_identifier(symbol: &str) -> Option<String> {
    let mut chars = symbol.chars();
    let first = chars.next()?;
    if !first.is_ascii_uppercase() {
        return None;
    }
    let mut out = String::new();
    out.push(first.to_ascii_lowercase());
    out.extend(chars);
    Some(out)
}

fn file_defines_identifier(content: &str, ident: &str) -> bool {
    content.lines().any(|line| {
        function_params_include(line, ident)
            || binding_line_defines(line, ident)
            || import_line_defines(line, ident)
    })
}

fn function_params_include(line: &str, ident: &str) -> bool {
    let Some(open) = line.find('(') else {
        return false;
    };
    let Some(close_offset) = line[open + 1..].find(')') else {
        return false;
    };
    let params = &line[open + 1..open + 1 + close_offset];
    params.split(',').any(|param| {
        let param = param
            .split('=')
            .next()
            .unwrap_or("")
            .trim()
            .trim_start_matches("...")
            .trim();
        param == ident
    })
}

fn binding_line_defines(line: &str, ident: &str) -> bool {
    ["const", "let", "var"].iter().any(|keyword| {
        let trimmed = line.trim_start();
        trimmed
            .strip_prefix(keyword)
            .and_then(|rest| rest.strip_prefix(' '))
            .is_some_and(|rest| starts_with_identifier(rest.trim_start(), ident))
    })
}

fn import_line_defines(line: &str, ident: &str) -> bool {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("import ") {
        return false;
    }
    contains_identifier(trimmed, ident)
}

fn starts_with_identifier(text: &str, ident: &str) -> bool {
    text.strip_prefix(ident)
        .is_some_and(|rest| rest.chars().next().is_none_or(|c| !is_identifier_char(c)))
}

fn contains_identifier(text: &str, ident: &str) -> bool {
    let mut start = 0;
    while let Some(offset) = text[start..].find(ident) {
        let index = start + offset;
        let before_ok = text[..index]
            .chars()
            .next_back()
            .is_none_or(|c| !is_identifier_char(c));
        let after_index = index + ident.len();
        let after_ok = text[after_index..]
            .chars()
            .next()
            .is_none_or(|c| !is_identifier_char(c));
        if before_ok && after_ok {
            return true;
        }
        start = after_index;
    }
    false
}

fn verify_prompt(
    task: &str,
    work: &Path,
    porcelain: &str,
    test_status: TestStatus,
    test_log: Option<&str>,
) -> String {
    let diff = git_diff_bounded(work, 5000);
    let test_tail = test_log.map_or("", |l| tail(l, 1200));
    let test_context_limit = if matches!(test_status, TestStatus::Pass) {
        2500
    } else {
        8000
    };
    format!(
        "[[ZO-DEEP:VERIFY]] You are a STRICT, adversarial verifier. Judge the change below against the task. Do not call tools, inspect files, or run tests; use only the evidence in this prompt and answer immediately.\n\n\
         Task:\n{task}\n\n\
         git status --porcelain:\n{porcelain}\n\n\
         Diff (tracked changes):\n{diff}\n\n\
         Test result: {status}\n{test_tail}\n\n\
         Relevant test files (read-only assertion context):\n{test_ctx}\n\n\
         Accept ONLY if the change fully and correctly satisfies the task with: only the\n\
         intended files changed (no unrelated edits, no leaked artifacts); no test\n\
         modified or deleted to force a pass; every requirement implemented; correct\n\
         error handling and edge cases. If a checked-in test explicitly requires behavior\n\
         that is not spelled out in the prompt, treat the test as part of the task\n\
         contract instead of rejecting that behavior as scope creep.\n\n\
         Respond with ONLY a single-line JSON object and NOTHING else — no prose, no\n\
         markdown code fences, no extra keys, and no text before or after it. Use exactly\n\
         these two keys (accepted, issues):\n\
         {{\"accepted\": true, \"issues\": []}}\n\
         or\n\
         {{\"accepted\": false, \"issues\": [\"specific problem 1\", \"specific problem 2\"]}}\n",
        status = test_status.as_str(),
        test_ctx = test_context(work, test_context_limit),
    )
}

fn compact_verify_prompt(
    task: &str,
    porcelain: &str,
    test_status: TestStatus,
    test_log: Option<&str>,
) -> String {
    let test_tail = test_log.map_or("", |l| tail(l, 600));
    format!(
        "[[ZO-DEEP:VERIFY:COMPACT]] Return ONLY JSON: {{\"accepted\":true|false,\"issues\":[...]}}. Do not call tools, inspect files, or run tests; decide from this prompt only.\n\
         Task:\n{task}\n\n\
         Objective gate already passed: tests={status}; changed files are exactly the harness-tracked intended diff unless status below says otherwise.\n\
         git status --porcelain:\n{porcelain}\n\n\
         Test tail:\n{test_tail}\n\n\
         Accept if the objective-green patch satisfies the task and does not look like a test cheat or unrelated edit. Reject only for a concrete semantic defect.\n",
        status = test_status.as_str(),
    )
}

/// Read-only assertion context: the first few test files, bounded.
fn test_context(work: &Path, max_chars: usize) -> String {
    const DIRS: &[&str] = &["test", "tests", "__tests__", "spec", "specs"];
    let mut files = Vec::new();
    for d in DIRS {
        collect_files(work, &work.join(d), &mut files);
    }
    files.sort();
    let mut out = String::new();
    for rel in files.iter().take(8) {
        let path = work.join(rel);
        if let Ok(content) = fs::read_to_string(&path) {
            let _ = writeln!(out, "### {rel}");
            let _ = writeln!(out, "```");
            for line in content.lines().take(160) {
                let _ = writeln!(out, "{line}");
            }
            let _ = writeln!(out, "```");
        }
    }
    if out.len() > max_chars {
        out.truncate(byte_boundary(&out, max_chars));
    }
    out
}

fn git_diff_bounded(work: &Path, max: usize) -> String {
    let out = std::process::Command::new("git")
        .current_dir(work)
        .args(["diff"])
        .output();
    let Ok(out) = out else {
        return String::new();
    };
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    if s.len() > max {
        s.truncate(byte_boundary(&s, max));
    }
    s
}

fn byte_boundary(s: &str, mut n: usize) -> usize {
    while n < s.len() && !s.is_char_boundary(n) {
        n += 1;
    }
    n.min(s.len())
}

/// `as_str` for `TestStatus` (Serialize is lowercase; the prompt needs the same).
trait TestStatusStr {
    fn as_str(&self) -> &'static str;
}
impl TestStatusStr for TestStatus {
    fn as_str(&self) -> &'static str {
        match self {
            TestStatus::Skipped => "skipped",
            TestStatus::Pass => "pass",
            TestStatus::Fail => "fail",
        }
    }
}

#[cfg(test)]
mod tests;
