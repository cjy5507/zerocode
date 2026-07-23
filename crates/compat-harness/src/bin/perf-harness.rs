//! `perf-harness` — measure a command's latency + peak RSS and, optionally,
//! gate it against a checked-in baseline for CI.
//!
//! Modes:
//!   run    (default)  measure a command, print summary JSON
//!   bless             measure, then write/update a baseline entry
//!   gate              measure, compare to baseline, exit 3 on regression
//!   suite             run the built-in `zo` scenario bundle (gate or bless)
//!
//! The first token may be a mode name; with no mode it behaves as `run`, so
//! the historical `perf-harness --name N -- CMD` invocation still works.

use std::path::PathBuf;

use compat_harness::regression::{
    default_scenarios, render_case_comparison, render_suite_report, PerfBaseline, PerfBaselineFile,
    RegressionGate, SuiteReport, DEFAULT_RSS_TOLERANCE, DEFAULT_TIME_TOLERANCE, EXIT_OK,
    EXIT_REGRESSION,
};
use compat_harness::{render_perf_summary_json, PerfCase, PerfHarness};

const USAGE: &str = "usage:
  perf-harness [run]  --name NAME [--warmups N] [--runs N] -- COMMAND [ARGS...]
  perf-harness bless  --baseline FILE --name NAME [--warmups N] [--runs N] -- COMMAND [ARGS...]
  perf-harness gate   --baseline FILE [--name NAME] [--tolerance R] [--rss-tolerance R] [--gate-rss] [--warmups N] [--runs N] -- COMMAND [ARGS...]
  perf-harness suite  --bin PATH-TO-zo --baseline FILE [--bless] [--tolerance R] [--rss-tolerance R] [--gate-rss] [--warmups N] [--runs N]

RSS is measured and reported for every case but does not fail the gate unless
--gate-rss is set, because peak RSS of a sub-10ms process is noisy.";

const DEFAULT_WARMUPS: usize = 2;
const DEFAULT_RUNS: usize = 5;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Run,
    Gate,
    Bless,
    Suite,
}

fn parse_mode(token: &str) -> Option<Mode> {
    match token {
        "run" => Some(Mode::Run),
        "gate" => Some(Mode::Gate),
        "bless" => Some(Mode::Bless),
        "suite" => Some(Mode::Suite),
        _ => None,
    }
}

#[derive(Default, Debug)]
struct Flags {
    name: Option<String>,
    warmups: Option<usize>,
    runs: Option<usize>,
    baseline: Option<PathBuf>,
    bin: Option<String>,
    time_tolerance: Option<f64>,
    rss_tolerance: Option<f64>,
    gate_rss: bool,
    bless: bool,
    command: Option<String>,
    command_args: Vec<String>,
}

impl Flags {
    fn build_case(&self) -> Result<PerfCase, String> {
        let command = self
            .command
            .clone()
            .ok_or_else(|| "missing command after `--`".to_string())?;
        let name = self.name.clone().unwrap_or_else(|| command.clone());
        Ok(PerfCase {
            name,
            command,
            args: self.command_args.clone(),
            warmups: self.warmups.unwrap_or(DEFAULT_WARMUPS),
            runs: self.runs.unwrap_or(DEFAULT_RUNS),
        })
    }

    fn regression_gate(&self) -> RegressionGate {
        RegressionGate::new(
            self.time_tolerance.unwrap_or(DEFAULT_TIME_TOLERANCE),
            self.rss_tolerance.unwrap_or(DEFAULT_RSS_TOLERANCE),
        )
        .gating_rss(self.gate_rss)
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (mode, rest) = match raw.first().map(String::as_str).and_then(parse_mode) {
        Some(mode) => (mode, &raw[1..]),
        None => (Mode::Run, &raw[..]),
    };

    let flags = match parse_flags(rest) {
        Ok(flags) => flags,
        Err(message) => fail_usage(&message),
    };

    let outcome = match mode {
        Mode::Run => run_mode(&flags),
        Mode::Gate => gate_mode(&flags),
        Mode::Bless => bless_mode(&flags),
        Mode::Suite => suite_mode(&flags),
    };

    match outcome {
        Ok(code) => std::process::exit(code),
        Err(message) => fail_usage(&message),
    }
}

fn fail_usage(message: &str) -> ! {
    eprintln!("perf-harness: {message}");
    eprintln!("{USAGE}");
    std::process::exit(2);
}

fn run_mode(flags: &Flags) -> Result<i32, String> {
    let case = flags.build_case()?;
    let summary = PerfHarness::run(&case).map_err(|error| error.to_string())?;
    println!("{}", render_perf_summary_json(&summary));
    Ok(EXIT_OK)
}

fn bless_mode(flags: &Flags) -> Result<i32, String> {
    let baseline_path = flags
        .baseline
        .clone()
        .ok_or_else(|| "bless requires --baseline FILE".to_string())?;
    let case = flags.build_case()?;
    let summary = PerfHarness::run(&case).map_err(|error| error.to_string())?;

    let mut file = load_or_empty(&baseline_path)?;
    file.upsert(PerfBaseline::from_summary(&summary));
    file.save(&baseline_path)
        .map_err(|error| format!("save baseline: {error}"))?;

    eprintln!(
        "perf-harness: blessed `{}` -> {}",
        case.name,
        baseline_path.display()
    );
    println!("{}", render_perf_summary_json(&summary));
    Ok(EXIT_OK)
}

fn gate_mode(flags: &Flags) -> Result<i32, String> {
    let baseline_path = flags
        .baseline
        .clone()
        .ok_or_else(|| "gate requires --baseline FILE".to_string())?;
    let case = flags.build_case()?;
    let baseline_file = PerfBaselineFile::load(&baseline_path)
        .map_err(|error| format!("load baseline: {error}"))?;
    let summary = PerfHarness::run(&case).map_err(|error| error.to_string())?;

    let Some(baseline) = baseline_file.get(&case.name) else {
        eprintln!(
            "perf-harness: no baseline for `{}` in {} — run `bless` first; passing",
            case.name,
            baseline_path.display()
        );
        println!("{}", render_perf_summary_json(&summary));
        return Ok(EXIT_OK);
    };

    let comparison = flags.regression_gate().compare(baseline, &summary);
    println!("{}", render_case_comparison(&comparison));
    Ok(if comparison.is_regression() {
        EXIT_REGRESSION
    } else {
        EXIT_OK
    })
}

fn suite_mode(flags: &Flags) -> Result<i32, String> {
    let bin = flags
        .bin
        .clone()
        .ok_or_else(|| "suite requires --bin PATH-TO-zo".to_string())?;
    let baseline_path = flags
        .baseline
        .clone()
        .ok_or_else(|| "suite requires --baseline FILE".to_string())?;

    let cases = default_scenarios(
        &bin,
        flags.warmups.unwrap_or(DEFAULT_WARMUPS),
        flags.runs.unwrap_or(DEFAULT_RUNS),
    );
    let mut summaries = Vec::with_capacity(cases.len());
    for case in &cases {
        let summary =
            PerfHarness::run(case).map_err(|error| format!("case `{}`: {error}", case.name))?;
        summaries.push(summary);
    }

    if flags.bless {
        let mut file = load_or_empty(&baseline_path)?;
        for summary in &summaries {
            file.upsert(PerfBaseline::from_summary(summary));
        }
        file.save(&baseline_path)
            .map_err(|error| format!("save baseline: {error}"))?;
        eprintln!(
            "perf-harness: blessed {} case(s) -> {}",
            summaries.len(),
            baseline_path.display()
        );
        return Ok(EXIT_OK);
    }

    let file = PerfBaselineFile::load(&baseline_path)
        .map_err(|error| format!("load baseline: {error}"))?;
    let report = SuiteReport::evaluate(&flags.regression_gate(), &file, &summaries);
    println!("{}", render_suite_report(&report));
    Ok(report.exit_code())
}

/// Load a baseline file, treating "not found" as an empty file so the first
/// `bless` can create it, while a malformed existing file is a hard error
/// rather than being silently clobbered.
fn load_or_empty(path: &PathBuf) -> Result<PerfBaselineFile, String> {
    match PerfBaselineFile::load(path) {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(PerfBaselineFile::empty()),
        Err(error) => Err(format!("load baseline: {error}")),
    }
}

fn parse_flags(args: &[String]) -> Result<Flags, String> {
    let mut flags = Flags::default();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--" => {
                let tail = &args[(index + 1).min(args.len())..];
                flags.command = tail.first().cloned();
                flags.command_args = tail.get(1..).map(<[String]>::to_vec).unwrap_or_default();
                return Ok(flags);
            }
            "--name" => {
                index += 1;
                flags.name = Some(take(args, index, "--name")?);
            }
            "--baseline" => {
                index += 1;
                flags.baseline = Some(PathBuf::from(take(args, index, "--baseline")?));
            }
            "--bin" => {
                index += 1;
                flags.bin = Some(take(args, index, "--bin")?);
            }
            "--warmups" => {
                index += 1;
                flags.warmups = Some(parse_count(&take(args, index, "--warmups")?, "--warmups")?);
            }
            "--runs" => {
                index += 1;
                flags.runs = Some(parse_count(&take(args, index, "--runs")?, "--runs")?);
            }
            "--tolerance" => {
                index += 1;
                flags.time_tolerance = Some(parse_ratio(
                    &take(args, index, "--tolerance")?,
                    "--tolerance",
                )?);
            }
            "--rss-tolerance" => {
                index += 1;
                flags.rss_tolerance = Some(parse_ratio(
                    &take(args, index, "--rss-tolerance")?,
                    "--rss-tolerance",
                )?);
            }
            "--gate" => { /* explicit affirmation of the default suite action */ }
            "--gate-rss" => flags.gate_rss = true,
            "--bless" => flags.bless = true,
            other => return Err(format!("unknown option `{other}`")),
        }
        index += 1;
    }

    Ok(flags)
}

fn take(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_count(value: &str, flag: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid value for {flag}: `{value}`"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_ratio(value: &str, flag: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid value for {flag}: `{value}`"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!(
            "{flag} must be a non-negative number (e.g. 0.05 for 5%)"
        ));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn into(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_string()).collect()
    }

    #[test]
    fn parses_named_command_case_legacy() {
        let flags = parse_flags(&into(&[
            "--name",
            "zo-version",
            "--warmups",
            "1",
            "--runs",
            "3",
            "--",
            "zo",
            "--version",
        ]))
        .expect("parse flags");

        assert_eq!(flags.name.as_deref(), Some("zo-version"));
        assert_eq!(flags.warmups, Some(1));
        assert_eq!(flags.runs, Some(3));

        let case = flags.build_case().expect("build case");
        assert_eq!(case.name, "zo-version");
        assert_eq!(case.command, "zo");
        assert_eq!(case.args, vec!["--version"]);
        assert_eq!(case.warmups, 1);
        assert_eq!(case.runs, 3);
    }

    #[test]
    fn name_defaults_to_command_when_absent() {
        let flags = parse_flags(&into(&["--", "zo", "--version"])).expect("parse");
        let case = flags.build_case().expect("build");
        assert_eq!(case.name, "zo");
        assert_eq!(case.warmups, DEFAULT_WARMUPS);
        assert_eq!(case.runs, DEFAULT_RUNS);
    }

    #[test]
    fn parses_gate_tolerances_and_baseline() {
        let flags = parse_flags(&into(&[
            "--baseline",
            "docs/perf/baselines/zo-local.json",
            "--tolerance",
            "0.1",
            "--rss-tolerance",
            "0.2",
            "--",
            "zo",
            "--help",
        ]))
        .expect("parse");

        assert_eq!(
            flags.baseline,
            Some(PathBuf::from("docs/perf/baselines/zo-local.json"))
        );
        let gate = flags.regression_gate();
        assert!((gate.time_tolerance - 0.1).abs() < 1e-9);
        assert!((gate.rss_tolerance - 0.2).abs() < 1e-9);
    }

    #[test]
    fn suite_bless_flag_parses() {
        let flags = parse_flags(&into(&[
            "--bin",
            "target/release/zo",
            "--baseline",
            "b.json",
            "--bless",
        ]))
        .expect("parse");
        assert!(flags.bless);
        assert_eq!(flags.bin.as_deref(), Some("target/release/zo"));
    }

    #[test]
    fn rejects_zero_runs_and_bad_tolerance() {
        assert!(parse_flags(&into(&["--runs", "0", "--", "zo"]))
            .unwrap_err()
            .contains("greater than zero"));
        assert!(parse_flags(&into(&["--tolerance", "nope", "--", "zo"]))
            .unwrap_err()
            .contains("invalid value"));
    }

    #[test]
    fn parse_mode_recognizes_subcommands() {
        assert_eq!(parse_mode("gate"), Some(Mode::Gate));
        assert_eq!(parse_mode("bless"), Some(Mode::Bless));
        assert_eq!(parse_mode("suite"), Some(Mode::Suite));
        assert_eq!(parse_mode("run"), Some(Mode::Run));
        assert_eq!(parse_mode("zo"), None);
    }

    #[test]
    fn missing_command_is_an_error() {
        let flags = parse_flags(&into(&["--name", "x"])).expect("parse");
        assert!(flags.build_case().is_err());
    }
}
