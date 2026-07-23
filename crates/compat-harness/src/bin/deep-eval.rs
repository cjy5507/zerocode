//! `deep-eval` — the benchmark CLI. It exposes the pure decision rules in
//! [`compat_harness`] (deep-lane plan/verifier/retry, the final-decision matrix,
//! fairness contracts, ledger denominators) and drives the native runner
//! ([`compat_harness::run_one`] / [`compat_harness::run_suite`]).
//!
//! All *policy* (is the plan complete? did the verifier accept? retry or stop?
//! the final verdict?) lives in unit-tested Rust, and the runner does the
//! *online* work — running the real agent, git, the test command — in-process,
//! so the scoring can never silently drift from its tests.
//!
//! Modes:
//!   validate-plan FILE            → `{"valid":bool,"missing":[..]}`
//!   decide [flags]                → `{"decision":"accept|retry|give_up", ..}`
//!   `extract-result JSON_FILE`      → the runner's `result` text, raw on stdout
//!
//! All modes exit 0 on a well-formed call (the verdict is in stdout, so the
//! caller never has to fight `set -e`); only a usage error exits non-zero.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::ExitCode;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use compat_harness::deep_lane::{
    failure_summary, fold_verification_attempt, objective_passed, parse_verifier, validate_plan,
    DeepDecision, VerifierParse,
};
use compat_harness::diff_hygiene::{score, TestStatus};
use compat_harness::{
    build_contract, decide_final, discover_tasks, normalize_effort_label, normalize_model_label,
    parse_ledger, run_one, summarize_ledger, validate_task, DeepConfig, FairnessInput,
    FairnessStatus, LaneCatalog, ObjectiveGate, RunSpec, TaskManifest, VerifierDecision,
};
use compat_harness::{run_suite, SuiteConfig, SuiteRunner};
use serde::Serialize;

const USAGE: &str = "usage:
  deep-eval validate-plan PLAN_FILE
  deep-eval decide --attempt N --max N --exit-code N --test pass|fail|skipped \\
      [--intended PATH]... --porcelain-file FILE --verifier-file FILE \\
      [--test-log-file FILE] [--summary-out FILE]
  deep-eval decide-final --fairness valid|invalid|partial|unknown \\
      --objective green|red|not_run|invalid \\
      --verifier-parse json|salvaged|empty|unparseable|timeout \\
      --verifier-accepted true|false --artifacts-preserved true|false
  deep-eval discover FIXTURES_ROOT [--lanes LANES_TOML]
  deep-eval fairness-contract --runner R --lane L --fixture-id ID --prompt-file F \\
      --test-command CMD [--intended PATH]... --declared-model M --declared-effort E \\
      [--permission-mode PM] [--timeout-seconds N] [--fixture-commit C] \\
      [--fixture-tree-hash H] [--fixture-dirty-before BOOL] [--fixture-dirty-after BOOL] \\
      [--runner-version V] [--harness-version V] [--suite-version V] \\
      [--started-at T] [--finished-at T]
  deep-eval summary LEDGER_FILE   (includes phase timing, probe, output-token/sec fields)
  deep-eval run --runner R --bin PATH [--arg A]... --fixture DIR \\
      (--prompt P | --prompt-file F) [--test CMD] [--intended PATH]... \\
      --lane L --model M --effort E [--objective-gate G] [--diff-policy P] \\
      [--timeout-seconds N] \\
      [--artifacts-dir DIR] [--keep-failed] [--deep-attempts N]
  deep-eval suite [--fixtures DIR] [--lanes TOML] --out DIR [--runners CSV] \\
      [--suite-version V] [--repeat N]   (per-runner config from <NAME>_BIN/_KIND/_MODEL_LABEL/_EFFORT/_ARGS env;
      set <NAME>_ARGS=\"--permission-mode acceptEdits\" or another explicit permission mode for fairness-valid runs;
      --repeat N runs each cell N times -> repeats.json with pass@N + median wall/tokens;
      writes ledger.jsonl, summary.json with phase/probe/output-token metrics, leaderboard.json with _views,
      report.md, process_warnings.json, manifest.json, and per-run metrics.json/result.json)

Each mode prints JSON to stdout and exits 0 on a well-formed call.";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(mode) = args.next() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest: Vec<String> = args.collect();
    match mode.as_str() {
        "validate-plan" => run_validate_plan(&rest),
        "decide" => run_decide(&rest),
        "decide-final" => run_decide_final(&rest),
        "discover" => run_discover(&rest),
        "fairness-contract" => run_fairness_contract(&rest),
        "summary" => run_summary(&rest),
        "run" => run_run(&rest),
        "suite" => run_suite_cmd(&rest),
        "extract-result" => run_extract_result(&rest),
        "-h" | "--help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("deep-eval: unknown mode '{other}'\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// `run` — execute one runner on one fixture natively (the Rust replacement for
/// the shell `run_one`/`invoke_agent`), printing the scored per-run JSON. The
/// binary is canonicalized here so a relative `--bin` resolves regardless of the
/// work dir the runner is spawned in — the shell-harness `cd "$work"` foot-gun,
/// gone in one typed line.
#[allow(clippy::too_many_lines)]
fn run_run(args: &[String]) -> ExitCode {
    let mut runner = String::new();
    let mut bin = String::new();
    let mut cmd_args: Vec<String> = Vec::new();
    let mut fixture = String::new();
    let mut prompt = String::new();
    let mut prompt_file: Option<String> = None;
    let mut test_command: Option<String> = None;
    let mut intended: Vec<String> = Vec::new();
    let mut lane = String::new();
    let mut model = String::new();
    let mut effort = String::new();
    let mut objective_gate = "test_and_diff".to_string();
    let mut diff_policy = "intended_paths_only".to_string();
    let mut artifacts_dir: Option<String> = None;
    let mut keep_failed = false;
    let mut deep_attempts: Option<u32> = None;
    let mut timeout_seconds = 0_u64;

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--keep-failed" {
            keep_failed = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1).cloned();
        match arg {
            "--runner" => runner = val.unwrap_or_default(),
            "--bin" => bin = val.unwrap_or_default(),
            "--arg" => {
                if let Some(v) = val {
                    cmd_args.push(v);
                }
            }
            "--fixture" => fixture = val.unwrap_or_default(),
            "--prompt" => prompt = val.unwrap_or_default(),
            "--prompt-file" => prompt_file = val,
            "--test" => test_command = val,
            "--intended" => {
                if let Some(v) = val {
                    intended.push(v);
                }
            }
            "--lane" => lane = val.unwrap_or_default(),
            "--model" => model = val.unwrap_or_default(),
            "--effort" => effort = val.unwrap_or_default(),
            "--objective-gate" => objective_gate = val.unwrap_or(objective_gate),
            "--diff-policy" => diff_policy = val.unwrap_or(diff_policy),
            "--timeout-seconds" => timeout_seconds = val.and_then(|v| v.parse().ok()).unwrap_or(0),
            "--artifacts-dir" => artifacts_dir = val,
            "--deep-attempts" => deep_attempts = val.and_then(|v| v.parse().ok()),
            other => {
                eprintln!("deep-eval run: unknown flag '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 2;
    }

    let prompt = match prompt_file {
        Some(pf) => match fs::read_to_string(&pf) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("deep-eval run: cannot read --prompt-file '{pf}': {e}");
                return ExitCode::from(2);
            }
        },
        None => prompt,
    };

    let bin = fs::canonicalize(&bin).unwrap_or_else(|_| PathBuf::from(&bin));

    let spec = RunSpec {
        runner_kind: runner.clone(),
        runner,
        bin,
        args: cmd_args,
        fixture: PathBuf::from(fixture),
        prompt,
        test_command,
        intended,
        lane,
        model: normalize_model_label(&model),
        effort: normalize_effort_label(&effort),
        objective_gate,
        diff_policy,
        timeout_seconds,
        artifacts_dir: artifacts_dir.map(PathBuf::from),
        keep_failed,
        deep: deep_attempts.map(|n| DeepConfig { max_attempts: n }),
    };

    match run_one(&spec) {
        Ok(result) => match serde_json::to_string(&result) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("deep-eval run: serialize failed: {e}");
                ExitCode::from(1)
            }
        },
        Err(e) => {
            eprintln!("deep-eval run: {e}");
            ExitCode::from(1)
        }
    }
}

/// `suite` — run the whole benchmark suite natively (replaces agent-eval-suite.sh
/// and the shell harness). Per-runner bin/model/effort/args come from `<NAME>_*`
/// env vars (NAME = the uppercased runner), matching the shell suite's contract.
fn run_suite_cmd(args: &[String]) -> ExitCode {
    let mut fixtures = "bench/fixtures".to_string();
    let mut lanes = "bench/lanes.toml".to_string();
    let mut out = String::new();
    let mut runners_csv = "zo".to_string();
    let mut suite_version = "1.0".to_string();
    let mut repeat = 1usize;

    let mut i = 0;
    while i < args.len() {
        let val = args.get(i + 1).cloned();
        match args[i].as_str() {
            "--fixtures" => fixtures = val.unwrap_or(fixtures),
            "--lanes" => lanes = val.unwrap_or(lanes),
            "--out" => out = val.unwrap_or_default(),
            "--runners" => runners_csv = val.unwrap_or(runners_csv),
            "--suite-version" => suite_version = val.unwrap_or(suite_version),
            // Runs every cell N times and writes repeats.json (pass@N + median
            // wall/tokens). A 0/garbage value floors to 1 rather than skipping
            // every cell, so a typo can never silently produce an empty suite.
            "--repeat" => {
                repeat = val
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(1)
                    .max(1);
            }
            other => {
                eprintln!("deep-eval suite: unknown flag '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 2;
    }
    if out.is_empty() {
        eprintln!("deep-eval suite: --out DIR is required\n{USAGE}");
        return ExitCode::from(2);
    }

    let runners = match build_runners(&runners_csv) {
        Ok(runners) if !runners.is_empty() => runners,
        Ok(_) => {
            eprintln!("deep-eval suite: --runners listed no runners\n{USAGE}");
            return ExitCode::from(2);
        }
        Err(missing) => {
            eprintln!(
                "deep-eval suite: unset runner binary env var(s):\n  {}\n\
                 set each to the runner's executable — there is no PATH fallback, so the\n\
                 binary recorded in the fairness contract is always reproducible — or drop\n\
                 the runner from --runners.\n{USAGE}",
                missing.join("\n  ")
            );
            return ExitCode::from(2);
        }
    };

    let cfg = SuiteConfig {
        fixtures: PathBuf::from(fixtures),
        lanes: PathBuf::from(lanes),
        runners,
        out_dir: PathBuf::from(out),
        suite_version,
        timestamp: now_unix_string(),
        git_commit: git_head(),
        command_invocation: Some(std::env::args().collect()),
        repeat,
    };
    match run_suite(&cfg) {
        Ok(()) => {
            println!("{}", suite_success_json(&cfg.out_dir));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("deep-eval suite: {e}");
            ExitCode::from(1)
        }
    }
}

fn suite_success_json(out_dir: &Path) -> String {
    let obj = serde_json::json!({
        "result_dir": out_dir,
        "ledger": out_dir.join("ledger.jsonl"),
        "summary": out_dir.join("summary.json"),
        "leaderboard": out_dir.join("leaderboard.json"),
        "leaderboard_views": ["composite", "accuracy_first", "latency_first", "token_first"],
        "report": out_dir.join("report.md"),
        "process_warnings": out_dir.join("process_warnings.json"),
        "manifest": out_dir.join("manifest.json"),
        "per_run_metrics": "<result_dir>/<runner>/<lane-fixture>[/rep-N]/metrics.json",
        "new_metrics": [
            "phase_plan_millis",
            "phase_exec_millis",
            "phase_test_millis",
            "phase_verify_millis",
            "phase_repair_millis",
            "deterministic_probe_failure_count",
            "total_tokens_per_second",
            "output_tokens_per_second"
        ]
    });
    serde_json::to_string(&obj)
        .unwrap_or_else(|_| format!("{{\"result_dir\":\"{}\"}}", out_dir.display()))
}

/// Build each runner from its `<NAME>_*` env (bin canonicalized; permission mode
/// and model lifted from `<NAME>_ARGS` when not given explicitly). `<NAME>_KIND`
/// optionally names the spawn protocol (`zo`, `claude`, or `codex`) so one
/// suite can compare aliases such as `zo_claude` and `zo_gpt`.
///
/// Policy R5(a): a runner's binary is **explicit** — `<NAME>_BIN` must be set,
/// with no PATH fallback — so the binary recorded in the fairness contract is
/// always exactly the one that ran (reproducible). A requested runner whose
/// `<NAME>_BIN` is unset is collected into the `Err` list and **never silently
/// dropped**: the suite compares exactly the runners asked for, or says which
/// env var is missing, consistent with the harness's no-silent-degradation rule.
fn build_runners(csv: &str) -> Result<Vec<SuiteRunner>, Vec<String>> {
    let mut runners = Vec::new();
    let mut missing = Vec::new();
    for name in csv.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        let up = name.to_uppercase();
        let Ok(bin) = std::env::var(format!("{up}_BIN")) else {
            missing.push(format!("{up}_BIN (for runner '{name}')"));
            continue;
        };
        let bin = fs::canonicalize(&bin).unwrap_or_else(|_| PathBuf::from(&bin));
        let kind = std::env::var(format!("{up}_KIND"))
            .unwrap_or_else(|_| name.to_string())
            .to_ascii_lowercase();
        let args: Vec<String> = std::env::var(format!("{up}_ARGS"))
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .collect();
        let model_label = std::env::var(format!("{up}_MODEL_LABEL"))
            .ok()
            .or_else(|| flag_value(&args, "--model"))
            .unwrap_or_else(|| "unknown".to_string());
        let effort_label = std::env::var(format!("{up}_EFFORT"))
            .or_else(|_| std::env::var(format!("{up}_EFFORT_LABEL")))
            .ok()
            .or_else(|| flag_value(&args, "--effort"))
            .unwrap_or_else(|| "unknown".to_string());
        let permission_mode = runner_permission_mode(&kind, &args);
        let version = runner_version(&bin);
        runners.push(SuiteRunner {
            name: name.to_string(),
            kind,
            bin,
            args,
            model_label,
            effort_label,
            permission_mode,
            version,
        });
    }
    if missing.is_empty() {
        Ok(runners)
    } else {
        Err(missing)
    }
}

/// The value following `flag` in an argument vector, if present.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn runner_permission_mode(runner: &str, args: &[String]) -> String {
    if let Some(mode) = flag_value(args, "--permission-mode") {
        return mode;
    }
    if runner == "codex" {
        if args
            .iter()
            .any(|arg| arg == "--dangerously-bypass-approvals-and-sandbox")
        {
            return "danger-full-access".to_string();
        }
        if let Some(sandbox) = flag_value(args, "--sandbox") {
            return sandbox;
        }
    }
    String::new()
}

fn now_unix_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_default()
}

fn git_head() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn runner_version(bin: &Path) -> String {
    Command::new(bin)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn run_validate_plan(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("deep-eval validate-plan: PLAN_FILE required\n{USAGE}");
        return ExitCode::from(2);
    };
    let markdown = fs::read_to_string(path).unwrap_or_default();
    let verdict = validate_plan(&markdown);
    let missing = json_string_array(&verdict.missing);
    println!("{{\"valid\":{},\"missing\":{}}}", verdict.valid, missing);
    ExitCode::SUCCESS
}

/// Print the runner's final assistant text from a `--output-format json` object.
/// The deep loop feeds this to the next phase: the plan text into the execute
/// prompt, the verifier's verdict into `decide`. `serde_json` does the unescaping
/// so embedded quotes/newlines survive intact (a bash extractor would mangle
/// them). Prints nothing (exit 0) when the file is absent, unparseable, or
/// carries no text key — a runner that produced no text simply yields an empty
/// phase input.
fn run_extract_result(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("deep-eval extract-result: JSON_FILE required\n{USAGE}");
        return ExitCode::from(2);
    };
    let raw = fs::read_to_string(path).unwrap_or_default();
    print!("{}", extract_result_text(&raw));
    ExitCode::SUCCESS
}

/// Pull the final assistant text out of a `--output-format json` result object.
///
/// The two runners label that text under different keys: Claude Code uses
/// `result`, Zo uses `message` (see `prompt_result_json` in the Zo CLI).
/// The extractor accepts either — Claude's `result` first, then Zo's
/// `message` — so the deep loop reads real plan/verifier text regardless of which
/// runner produced it. (Before this, only `result` was read, so every Zo phase
/// silently extracted an empty string: empty plan → `plan_valid:false`, empty
/// verifier → `verifier_accepted:false`, i.e. a guaranteed `deep_unverified`.)
///
/// Returns an empty string when `raw` is empty, not a JSON object, or has neither
/// key — the genuinely-empty case the caller renders as an empty phase input.
fn extract_result_text(raw: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| {
            v.get("result")
                .or_else(|| v.get("message"))
                .and_then(|text| text.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

#[derive(Default)]
struct DecideFlags {
    attempt: u32,
    max: u32,
    exit_code: i32,
    test: TestStatus,
    intended: Vec<String>,
    porcelain_file: Option<PathBuf>,
    verifier_file: Option<PathBuf>,
    test_log_file: Option<PathBuf>,
    summary_out: Option<PathBuf>,
}

/// `decide` — the deep loop's **loop-control** verdict (`accept` / `retry` /
/// `give_up`): should the loop stop, or run another attempt? This is NOT the
/// leaderboard score. The final per-run decision is [`run_decide_final`], which
/// applies the strict matrix and can legitimately differ — e.g. a salvaged
/// verifier lets this loop *stop-accept* (retrying a malformed verifier is
/// pointless), while `decide-final` records `inconclusive` because a salvaged
/// parse is not a trustworthy accept. Read this verdict only as "keep going or
/// not", never as the run's result.
fn run_decide(args: &[String]) -> ExitCode {
    let mut f = DecideFlags {
        max: 1,
        ..DecideFlags::default()
    };
    // Index-based scan: each value flag consumes the following token. A missing
    // value is reported by the typed parsers (or simply leaves the field unset
    // for the optional path flags).
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let value = args.get(i + 1).cloned();
        match arg {
            "--attempt" => f.attempt = parse_or_exit(value, "--attempt"),
            "--max" => f.max = parse_or_exit(value, "--max"),
            "--exit-code" => f.exit_code = parse_or_exit(value, "--exit-code"),
            "--test" => f.test = parse_test_status(value.as_deref()),
            "--intended" => {
                if let Some(v) = value {
                    f.intended.push(v);
                }
            }
            "--porcelain-file" => f.porcelain_file = value.map(PathBuf::from),
            "--verifier-file" => f.verifier_file = value.map(PathBuf::from),
            "--test-log-file" => f.test_log_file = value.map(PathBuf::from),
            "--summary-out" => f.summary_out = value.map(PathBuf::from),
            other => {
                eprintln!("deep-eval decide: unknown flag '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        // Every recognized flag here takes exactly one value, so advance by two.
        i += 2;
    }

    let porcelain = read_opt(f.porcelain_file.as_ref());
    let verifier_raw = read_opt(f.verifier_file.as_ref());
    let test_log = read_opt(f.test_log_file.as_ref());

    let intended_refs: Vec<&str> = f.intended.iter().map(String::as_str).collect();
    let intended_provided = !intended_refs.is_empty();
    let hygiene = score(&porcelain, &intended_refs);
    let verifier = parse_verifier(&verifier_raw);
    let objective_ok = objective_passed(f.exit_code, f.test, &hygiene, intended_provided);
    let folded = fold_verification_attempt(f.attempt, f.max, objective_ok, &verifier, &[]);
    let verifier_gate = folded.gate_accepted;
    let decision = folded.decision;

    // Only build/persist a retry summary when the loop is going to continue or
    // stop short — an accepted run has nothing to summarize.
    if decision != DeepDecision::Accept {
        if let Some(out) = f.summary_out.as_ref() {
            let summary = failure_summary(&hygiene, f.test, &test_log, &verifier);
            let _ = fs::write(out, summary);
        }
    }

    println!(
        "{{\"decision\":\"{}\",\"objective_passed\":{},\"verifier_accepted\":{},\"verifier_parse\":\"{}\",\"verifier_issues\":{},\"clean_diff\":{},\"intended_changed\":{},\"attempt\":{},\"max\":{}}}",
        decision.as_str(),
        objective_ok,
        verifier_gate,
        verifier.parse.as_str(),
        verifier.issues.len(),
        hygiene.clean,
        hygiene.intended_count,
        f.attempt,
        f.max,
    );
    ExitCode::SUCCESS
}

/// Apply the final-decision matrix (spec lines 261-270) at the shell boundary —
/// the **single source of truth** for a run's leaderboard verdict.
///
/// Unlike [`run_decide`] (which only steers the deep loop's retry/stop), this is
/// the verdict the ledger and pass-rate denominators are built from. The two can
/// differ by design: the loop may stop-accept a salvaged verifier, but here a
/// salvaged parse is `inconclusive`, not `accepted`. When interpreting results,
/// trust this — never the loop-control `decide`.
///
/// The shell already knows each classified axis — the fairness contract status,
/// the objective gate, how the verifier parsed and whether it accepted, and
/// whether the required artifacts were preserved. This turns those into exactly
/// one typed final verdict (`accepted | rejected | inconclusive | invalid |
/// blocked`) plus a precise failure class, so the scorer never re-derives the
/// matrix in bash and drifts from the unit-tested Rust.
fn run_decide_final(args: &[String]) -> ExitCode {
    let mut fairness = None;
    let mut objective = None;
    let mut parse = None;
    let mut accepted = false;
    let mut artifacts = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let value = args.get(i + 1).cloned();
        match arg {
            "--fairness" => fairness = value.as_deref().and_then(FairnessStatus::from_token),
            "--objective" => objective = value.as_deref().and_then(ObjectiveGate::from_token),
            "--verifier-parse" => parse = value.as_deref().and_then(VerifierParse::from_token),
            "--verifier-accepted" => accepted = matches!(value.as_deref(), Some("true")),
            "--artifacts-preserved" => artifacts = matches!(value.as_deref(), Some("true")),
            other => {
                eprintln!("deep-eval decide-final: unknown flag '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 2;
    }

    let (Some(fairness), Some(objective), Some(parse)) = (fairness, objective, parse) else {
        eprintln!(
            "deep-eval decide-final: --fairness, --objective, and --verifier-parse are required\n{USAGE}"
        );
        return ExitCode::from(2);
    };

    // The verifier's semantic decision is derived from its accept flag and how
    // its output parsed: a non-acceptance is a real reject only with a decision
    // signal (strict/salvage), never on empty/malformed/timeout output.
    let decision = VerifierDecision::from_verdict(accepted, parse);
    let verdict = decide_final(fairness, objective, parse, decision, artifacts);
    let failure = verdict
        .failure
        .map_or_else(|| "null".to_string(), |f| format!("\"{}\"", f.as_str()));
    println!(
        "{{\"final_decision\":\"{}\",\"failure_class\":{},\"leaderboard_eligible\":{},\"verifier_decision\":\"{}\",\"verifier_parse_spec\":\"{}\"}}",
        verdict.decision.as_str(),
        failure,
        verdict.leaderboard_eligible,
        decision.as_str(),
        parse.spec_mode(),
    );
    ExitCode::SUCCESS
}

/// One discovered task, re-serialized to JSON for the shell to iterate. The
/// manifest fields are flattened in so `jq` reads `.id`, `.lane`, `.prompt`,
/// `.test_command`, `.intended_paths` directly alongside the discovery metadata.
#[derive(Serialize)]
struct DiscoverItem<'a> {
    dir: String,
    valid: bool,
    problems: Vec<String>,
    #[serde(flatten)]
    manifest: &'a TaskManifest,
}

/// `discover FIXTURES_ROOT [--lanes LANES_TOML]` — scan every `task.toml` under
/// the fixtures root and print them as a JSON array. With a lane catalog, each
/// task is validated against it and carries `valid` + `problems`, so the shell
/// can skip or fail invalid fixtures instead of hard-coding the task list.
fn run_discover(args: &[String]) -> ExitCode {
    let mut root: Option<String> = None;
    let mut lanes_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--lanes" => {
                lanes_path = args.get(i + 1).cloned();
                i += 2;
            }
            other if root.is_none() => {
                root = Some(other.to_string());
                i += 1;
            }
            other => {
                eprintln!("deep-eval discover: unexpected argument '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(root) = root else {
        eprintln!("deep-eval discover: FIXTURES_ROOT required\n{USAGE}");
        return ExitCode::from(2);
    };

    let tasks = match discover_tasks(Path::new(&root)) {
        Ok(tasks) => tasks,
        Err(e) => {
            eprintln!("deep-eval discover: {e}");
            return ExitCode::from(1);
        }
    };

    let catalog = match lanes_path.as_ref() {
        Some(path) => match LaneCatalog::load(Path::new(path)) {
            Ok(catalog) => Some(catalog),
            Err(e) => {
                eprintln!("deep-eval discover: {path}: {e}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    let items: Vec<DiscoverItem> = tasks
        .iter()
        .map(|task| {
            let problems = catalog
                .as_ref()
                .map(|catalog| validate_task(&task.manifest, catalog))
                .unwrap_or_default();
            DiscoverItem {
                dir: task.dir.display().to_string(),
                valid: problems.is_empty(),
                problems,
                manifest: &task.manifest,
            }
        })
        .collect();

    match serde_json::to_string(&items) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("deep-eval discover: serialize failed: {e}");
            ExitCode::from(1)
        }
    }
}

/// `fairness-contract [flags]` — compute one run's `fairness_contract.json`. The
/// shell passes the online context it alone knows (git commit/tree hash, dirty
/// flags, timestamps, versions, permission mode); deep-eval reads the prompt
/// file, hashes the inputs, normalizes model/effort, judges per-run validity,
/// and prints the contract JSON for the shell to save with the run's artifacts.
fn run_fairness_contract(args: &[String]) -> ExitCode {
    let mut input = FairnessInput::default();
    let mut prompt_file: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let value = args.get(i + 1).cloned();
        match arg {
            "--runner" => input.runner = value.unwrap_or_default(),
            "--lane" => input.lane = value.unwrap_or_default(),
            "--fixture-id" => input.fixture_id = value.unwrap_or_default(),
            "--prompt-file" => prompt_file = value,
            "--test-command" => input.test_command = value.unwrap_or_default(),
            "--intended" => {
                if let Some(path) = value {
                    input.intended_path_set.push(path);
                }
            }
            "--declared-model" => input.declared_model = value.unwrap_or_default(),
            "--declared-effort" => input.declared_effort = value.unwrap_or_default(),
            "--permission-mode" => input.permission_mode = value.unwrap_or_default(),
            "--timeout-seconds" => {
                input.timeout_seconds = value.and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            "--fixture-commit" => input.fixture_commit = value.unwrap_or_default(),
            "--fixture-tree-hash" => input.fixture_tree_hash = value.unwrap_or_default(),
            "--fixture-dirty-before" => {
                input.fixture_dirty_before = matches!(value.as_deref(), Some("true"));
            }
            "--fixture-dirty-after" => {
                input.fixture_dirty_after = matches!(value.as_deref(), Some("true"));
            }
            "--runner-version" => input.runner_version = value.unwrap_or_default(),
            "--harness-version" => input.harness_version = value.unwrap_or_default(),
            "--suite-version" => input.benchmark_suite_version = value.unwrap_or_default(),
            "--started-at" => input.started_at = value.unwrap_or_default(),
            "--finished-at" => input.finished_at = value.unwrap_or_default(),
            other => {
                eprintln!("deep-eval fairness-contract: unknown flag '{other}'\n{USAGE}");
                return ExitCode::from(2);
            }
        }
        // Every recognized flag consumes exactly one value.
        i += 2;
    }

    if let Some(path) = prompt_file {
        input.prompt = fs::read_to_string(&path).unwrap_or_default();
        input.prompt_path = path;
    }

    let contract = build_contract(&input);
    match serde_json::to_string(&contract) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("deep-eval fairness-contract: serialize failed: {e}");
            ExitCode::from(1)
        }
    }
}

/// `summary LEDGER_FILE` — aggregate the run ledger into per-(lane, runner)
/// pass-rate denominators (strict / adjudicated / inconclusive / invalid /
/// blocked / artifact-preservation) and print them as JSON. Lanes are never
/// merged; an empty or missing ledger yields `{}`.
fn run_summary(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("deep-eval summary: LEDGER_FILE required\n{USAGE}");
        return ExitCode::from(2);
    };
    let text = fs::read_to_string(path).unwrap_or_default();
    let summary = summarize_ledger(&parse_ledger(&text));
    match serde_json::to_string(&summary) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("deep-eval summary: serialize failed: {e}");
            ExitCode::from(1)
        }
    }
}

fn parse_or_exit<T: std::str::FromStr>(value: Option<String>, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let Some(v) = value else {
        eprintln!("deep-eval decide: {flag} requires a value");
        std::process::exit(2);
    };
    v.parse().unwrap_or_else(|e| {
        eprintln!("deep-eval decide: {flag} expects a number ({e})");
        std::process::exit(2);
    })
}

fn parse_test_status(value: Option<&str>) -> TestStatus {
    match value {
        Some("pass") => TestStatus::Pass,
        Some("fail") => TestStatus::Fail,
        _ => TestStatus::Skipped,
    }
}

fn read_opt(path: Option<&PathBuf>) -> String {
    path.and_then(|p| fs::read_to_string(p).ok())
        .unwrap_or_default()
}

/// Render a slice of strings as a JSON array, escaping each element. Kept local
/// (no `serde_json::to_string` round-trip) because the elements are short,
/// known section names and the harness stays dependency-light at this boundary.
fn json_string_array(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for ch in item.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                _ => out.push(ch),
            }
        }
        out.push('"');
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::extract_result_text;
    use super::USAGE;

    #[test]
    fn suite_usage_and_success_json_surface_new_artifacts_and_metrics() {
        assert!(USAGE.contains("report.md"));
        assert!(USAGE.contains("process_warnings.json"));
        assert!(USAGE.contains("leaderboard.json with _views"));
        assert!(USAGE.contains("phase/probe/output-token metrics"));

        let json = super::suite_success_json(Path::new("/tmp/deep-out"));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["result_dir"], "/tmp/deep-out");
        assert!(value["report"].as_str().unwrap().ends_with("report.md"));
        assert!(value["process_warnings"]
            .as_str()
            .unwrap()
            .ends_with("process_warnings.json"));
        assert!(value["leaderboard_views"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "accuracy_first"));
        assert!(value["new_metrics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "phase_repair_millis"));
        assert!(value["new_metrics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "deterministic_probe_failure_count"));
    }

    #[test]
    fn suite_usage_documents_permission_mode_for_fairness_valid_runs() {
        assert!(USAGE.contains("<NAME>_ARGS=\"--permission-mode acceptEdits\""));
        assert!(USAGE.contains("fairness-valid runs"));
    }

    #[test]
    fn build_runners_errors_naming_the_unset_bin_env() {
        // R5(a): a requested runner with no <NAME>_BIN is a *named* error, never a
        // silent drop. 'zznosuchrunner' has no ZZNOSUCHRUNNER_BIN in any env, so
        // this is deterministic and spawns nothing (the unset path returns early).
        // let...else (not unwrap_err) because SuiteRunner has no Debug.
        let Err(missing) = super::build_runners("zznosuchrunner") else {
            panic!("unset ZZNOSUCHRUNNER_BIN must be a named error, not a runner");
        };
        assert!(missing.iter().any(|m| m.contains("ZZNOSUCHRUNNER_BIN")));
    }

    #[test]
    fn build_runners_keeps_alias_and_uses_kind_for_spawn_protocol() {
        std::env::set_var("ZO_ALIAS_TEST_BIN", "/bin/echo");
        std::env::set_var("ZO_ALIAS_TEST_KIND", "zo");
        std::env::set_var(
            "ZO_ALIAS_TEST_ARGS",
            "--model gpt-5.5 --permission-mode danger-full-access",
        );
        std::env::set_var("ZO_ALIAS_TEST_EFFORT", "xhigh");

        let runners = super::build_runners("zo_alias_test").expect("alias runner config");
        let runner = &runners[0];

        assert_eq!(runner.name, "zo_alias_test");
        assert_eq!(runner.kind, "zo");
        assert_eq!(runner.model_label, "gpt-5.5");
        assert_eq!(runner.effort_label, "xhigh");
        assert_eq!(runner.permission_mode, "danger-full-access");

        std::env::remove_var("ZO_ALIAS_TEST_BIN");
        std::env::remove_var("ZO_ALIAS_TEST_KIND");
        std::env::remove_var("ZO_ALIAS_TEST_ARGS");
        std::env::remove_var("ZO_ALIAS_TEST_EFFORT");
    }

    #[test]
    fn extracts_claude_result_field() {
        let raw = "{\"is_error\":false,\"result\":\"## Plan\\n- step 1\"}";
        assert_eq!(extract_result_text(raw), "## Plan\n- step 1");
    }

    #[test]
    fn extracts_zo_message_field() {
        // Zo's --output-format json carries the text under "message".
        let raw = "{\"is_error\":false,\"message\":\"## Plan\\n- step 1\",\"model\":\"sonnet\"}";
        assert_eq!(extract_result_text(raw), "## Plan\n- step 1");
    }

    #[test]
    fn result_wins_when_both_present() {
        let raw = "{\"result\":\"claude text\",\"message\":\"zo text\"}";
        assert_eq!(extract_result_text(raw), "claude text");
    }

    #[test]
    fn serde_unescapes_embedded_quotes_and_newlines() {
        // A verifier verdict the way Zo emits it: a JSON string whose value is
        // itself a JSON object with escaped quotes/newlines.
        let raw = "{\"message\":\"{\\\"accepted\\\": false, \\\"issues\\\": [\\\"a\\nb\\\"]}\"}";
        assert_eq!(
            extract_result_text(raw),
            "{\"accepted\": false, \"issues\": [\"a\nb\"]}"
        );
    }

    #[test]
    fn empty_when_no_text_key() {
        assert_eq!(extract_result_text("{\"is_error\":false,\"usage\":{}}"), "");
    }

    #[test]
    fn empty_when_not_json_or_blank() {
        assert_eq!(extract_result_text(""), "");
        assert_eq!(extract_result_text("not json at all"), "");
    }
}
