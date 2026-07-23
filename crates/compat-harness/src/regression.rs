//! Performance regression gate: turns raw [`PerfSummary`](crate::PerfSummary)
//! measurements into a pass/fail verdict against a checked-in baseline.
//!
//! The harness in [`crate`] *measures* a command (latency + peak RSS). This
//! module is the *judgment* layer: it stores a baseline, compares a fresh run
//! against it within a noise band, and classifies each case as Improved,
//! Unchanged, or Regressed. A batch of cases rolls up into a single CI exit
//! code so a regression fails the build.
//!
//! Why the median, not the mean: subprocess timing is dominated by occasional
//! OS noise (scheduling, page faults, cold caches). The median (p50) shrugs
//! off a single slow run; the mean does not. The gate therefore compares
//! medians, and the baseline carries the standard deviation so the noise level
//! travels with the measurement.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{PerfCase, PerfSummary};

/// Current on-disk schema version for baseline files.
pub const BASELINE_SCHEMA_VERSION: u32 = 1;

/// Default noise band for latency: a median change within ±5% is treated as
/// noise. Subprocess end-to-end timing is noisier than in-process
/// microbenchmarks (criterion defaults to 1%), so the band is wider.
pub const DEFAULT_TIME_TOLERANCE: f64 = 0.05;

/// Default noise band for peak RSS: ±10%. Allocator behavior and page
/// granularity make memory measurements coarser than time.
pub const DEFAULT_RSS_TOLERANCE: f64 = 0.10;

/// Exit code when the gate finds no regression.
pub const EXIT_OK: i32 = 0;
/// Exit code when at least one case regressed — distinct from a harness error
/// (which the binary reports as 1/2) so CI can tell "slower" from "broke".
pub const EXIT_REGRESSION: i32 = 3;

/// One captured measurement — the unit a gate compares against. Durations are
/// stored in microseconds so the baseline file stays compact and diffable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfBaseline {
    pub name: String,
    pub runs: usize,
    pub median_micros: u64,
    pub min_micros: u64,
    pub mean_micros: u64,
    pub stddev_micros: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_kib: Option<u64>,
}

impl PerfBaseline {
    /// Snapshot a fresh summary into a storable baseline.
    #[must_use]
    pub fn from_summary(summary: &PerfSummary) -> Self {
        Self {
            name: summary.name.clone(),
            runs: summary.runs,
            median_micros: duration_micros(summary.median_elapsed),
            min_micros: duration_micros(summary.min_elapsed),
            mean_micros: duration_micros(summary.mean_elapsed),
            stddev_micros: duration_micros(summary.stddev_elapsed),
            peak_rss_kib: summary.peak_rss_kib,
        }
    }

    #[must_use]
    pub fn median(&self) -> Duration {
        Duration::from_micros(self.median_micros)
    }
}

/// A checked-in baseline file: a versioned bundle of per-case baselines, kept
/// sorted by name for stable diffs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerfBaselineFile {
    pub schema: u32,
    pub cases: Vec<PerfBaseline>,
}

impl Default for PerfBaselineFile {
    fn default() -> Self {
        Self {
            schema: BASELINE_SCHEMA_VERSION,
            cases: Vec::new(),
        }
    }
}

impl PerfBaselineFile {
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parse a baseline file from JSON.
    ///
    /// # Errors
    /// Returns the `serde_json` error when the text is not a valid baseline.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Render as pretty JSON with a trailing newline (diff-friendly).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string());
        out.push('\n');
        out
    }

    /// Load a baseline file from disk.
    ///
    /// # Errors
    /// Propagates IO errors and maps malformed JSON to [`std::io::ErrorKind::InvalidData`].
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
    }

    /// Persist the baseline file, creating parent directories as needed.
    ///
    /// # Errors
    /// Propagates IO errors from directory creation or the write itself.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, self.to_json())
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PerfBaseline> {
        self.cases.iter().find(|case| case.name == name)
    }

    /// Insert or replace the baseline for a case, keeping `cases` sorted by name.
    pub fn upsert(&mut self, baseline: PerfBaseline) {
        if let Some(existing) = self
            .cases
            .iter_mut()
            .find(|case| case.name == baseline.name)
        {
            *existing = baseline;
        } else {
            self.cases.push(baseline);
        }
        self.cases.sort_by(|a, b| a.name.cmp(&b.name));
    }
}

/// Classification of one metric's change against its baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfVerdict {
    /// Beyond the noise band in the good direction (faster / less memory).
    Improved,
    /// Within the noise band.
    Unchanged,
    /// Beyond the noise band in the bad direction (slower / more memory).
    Regressed,
    /// No comparable baseline value (new case, or metric absent on one side).
    Unknown,
}

impl PerfVerdict {
    /// Bucket a relative change into a verdict. The band is inclusive: a delta
    /// exactly equal to the tolerance counts as `Unchanged`.
    #[must_use]
    pub fn classify(delta_ratio: f64, tolerance: f64) -> Self {
        if delta_ratio > tolerance {
            Self::Regressed
        } else if delta_ratio < -tolerance {
            Self::Improved
        } else {
            Self::Unchanged
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Improved => "improved",
            Self::Unchanged => "unchanged",
            Self::Regressed => "REGRESSED",
            Self::Unknown => "unknown",
        }
    }
}

/// Noise bands for the two metrics the gate judges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionGate {
    pub time_tolerance: f64,
    pub rss_tolerance: f64,
    /// Whether an RSS regression *fails* the gate. Off by default: peak RSS of
    /// a sub-10ms process (e.g. `zo --version`) swings widely between runs
    /// because `ps` samples it mid-fault, so gating on it produces noise-driven
    /// false failures. RSS is always measured and reported; enable this only
    /// for longer scenarios where memory is the thing under test.
    pub gate_rss: bool,
}

impl Default for RegressionGate {
    fn default() -> Self {
        Self {
            time_tolerance: DEFAULT_TIME_TOLERANCE,
            rss_tolerance: DEFAULT_RSS_TOLERANCE,
            gate_rss: false,
        }
    }
}

impl RegressionGate {
    #[must_use]
    pub fn new(time_tolerance: f64, rss_tolerance: f64) -> Self {
        Self {
            time_tolerance,
            rss_tolerance,
            gate_rss: false,
        }
    }

    /// Builder: make an RSS regression count toward the pass/fail verdict.
    #[must_use]
    pub fn gating_rss(mut self, gate_rss: bool) -> Self {
        self.gate_rss = gate_rss;
        self
    }

    /// Compare a fresh measurement against its baseline.
    #[must_use]
    pub fn compare(&self, baseline: &PerfBaseline, summary: &PerfSummary) -> CaseComparison {
        let current_median_micros = duration_micros(summary.median_elapsed);
        let time_delta = ratio(baseline.median_micros, current_median_micros);
        let time_tolerance = effective_time_tolerance(
            self.time_tolerance,
            baseline.median_micros,
            baseline.stddev_micros,
            duration_micros(summary.stddev_elapsed),
        );
        let time_verdict = time_delta.map_or(PerfVerdict::Unknown, |delta| {
            PerfVerdict::classify(delta, time_tolerance)
        });

        let (rss_delta, rss_verdict) = match (baseline.peak_rss_kib, summary.peak_rss_kib) {
            (Some(base), Some(current)) => {
                let delta = ratio(base, current);
                let verdict = delta.map_or(PerfVerdict::Unknown, |delta| {
                    PerfVerdict::classify(delta, self.rss_tolerance)
                });
                (delta, verdict)
            }
            _ => (None, PerfVerdict::Unknown),
        };

        CaseComparison {
            name: summary.name.clone(),
            baseline_median: baseline.median(),
            current_median: summary.median_elapsed,
            time_delta_ratio: time_delta,
            time_verdict,
            baseline_rss_kib: baseline.peak_rss_kib,
            current_rss_kib: summary.peak_rss_kib,
            rss_delta_ratio: rss_delta,
            rss_verdict,
            rss_gated: self.gate_rss,
        }
    }
}

/// The judged result for a single case: per-metric deltas and verdicts.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseComparison {
    pub name: String,
    pub baseline_median: Duration,
    pub current_median: Duration,
    /// `(current - baseline) / baseline` for the median; `None` if unjudgeable.
    pub time_delta_ratio: Option<f64>,
    pub time_verdict: PerfVerdict,
    pub baseline_rss_kib: Option<u64>,
    pub current_rss_kib: Option<u64>,
    pub rss_delta_ratio: Option<f64>,
    pub rss_verdict: PerfVerdict,
    /// Whether the RSS verdict counts toward [`Self::is_regression`]. When
    /// false the RSS delta is reported for trend visibility but never fails
    /// the gate (see [`RegressionGate::gate_rss`]).
    pub rss_gated: bool,
}

impl CaseComparison {
    /// A case is a regression if the latency regressed, or — when RSS gating is
    /// enabled — if peak memory regressed.
    #[must_use]
    pub fn is_regression(&self) -> bool {
        self.time_verdict == PerfVerdict::Regressed
            || (self.rss_gated && self.rss_verdict == PerfVerdict::Regressed)
    }
}

/// Roll-up of comparing a batch of fresh summaries against a baseline file.
#[derive(Debug, Clone, PartialEq)]
pub struct SuiteReport {
    pub comparisons: Vec<CaseComparison>,
    /// Cases that ran but had no baseline entry (cannot be judged yet).
    pub new_cases: Vec<String>,
    /// Cases present in the baseline but not run this time.
    pub missing_cases: Vec<String>,
}

impl SuiteReport {
    /// Compare every summary against the baseline file using `gate`.
    #[must_use]
    pub fn evaluate(
        gate: &RegressionGate,
        baseline: &PerfBaselineFile,
        summaries: &[PerfSummary],
    ) -> Self {
        let mut comparisons = Vec::new();
        let mut new_cases = Vec::new();
        for summary in summaries {
            if let Some(base) = baseline.get(&summary.name) {
                comparisons.push(gate.compare(base, summary));
            } else {
                new_cases.push(summary.name.clone());
            }
        }
        let missing_cases = baseline
            .cases
            .iter()
            .filter(|case| !summaries.iter().any(|summary| summary.name == case.name))
            .map(|case| case.name.clone())
            .collect();
        Self {
            comparisons,
            new_cases,
            missing_cases,
        }
    }

    #[must_use]
    pub fn has_regression(&self) -> bool {
        self.comparisons.iter().any(CaseComparison::is_regression)
    }

    /// CI exit code: 0 when clean, 3 when any case regressed.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        if self.has_regression() {
            EXIT_REGRESSION
        } else {
            EXIT_OK
        }
    }
}

/// The built-in scenario bundle, parameterized by the path to the `zo`
/// binary. Every case is a subprocess entry point that runs **deterministically
/// without a network or a TUI session**, so the gate measures zo's own work
/// rather than API latency or terminal interaction.
///
/// Coverage is layered from the trivial fast paths to progressively more
/// startup work:
/// * `startup-version` / `startup-help` — the argv fast paths (minimal work).
/// * `startup-system-prompt` — assembles the full system prompt (tool
///   manifests + prompt template + environment injection). Output is
///   byte-for-byte stable across runs, so latency is the only variable.
/// * `startup-bootstrap-plan` — enumerates the startup bootstrap plan,
///   exercising the planner without touching the network.
///
/// Excluded on purpose: `-p` prompt turns (need a live API → non-deterministic
/// latency) and TUI render (interactive only). Slash dispatch / git snapshot
/// scenarios land here once the binary grows headless entry points for them;
/// the gate and baseline format already support an arbitrary set of cases.
#[must_use]
pub fn default_scenarios(zo_bin: &str, warmups: usize, runs: usize) -> Vec<PerfCase> {
    let case = |name: &str, args: &[&str]| PerfCase {
        name: name.to_string(),
        command: zo_bin.to_string(),
        args: args.iter().map(|arg| (*arg).to_string()).collect(),
        warmups,
        runs,
    };
    vec![
        case("startup-version", &["--version"]),
        case("startup-help", &["--help"]),
        case("startup-system-prompt", &["system-prompt"]),
        case("startup-bootstrap-plan", &["bootstrap-plan"]),
    ]
}

/// Render one comparison as a single terminal line.
#[must_use]
pub fn render_case_comparison(comparison: &CaseComparison) -> String {
    let rss_note = if comparison.rss_gated { "" } else { "*" };
    format!(
        "{name:<22} time {time_verdict:<10} {time_delta:>8}  rss{rss_note} {rss_verdict:<10} {rss_delta:>8}  [{base:.1?} -> {cur:.1?}]",
        name = comparison.name,
        time_verdict = comparison.time_verdict.label(),
        time_delta = fmt_pct(comparison.time_delta_ratio),
        rss_verdict = comparison.rss_verdict.label(),
        rss_delta = fmt_pct(comparison.rss_delta_ratio),
        base = comparison.baseline_median,
        cur = comparison.current_median,
    )
}

/// Render the suite roll-up: one line per case, then new/missing/verdict notes.
#[must_use]
pub fn render_suite_report(report: &SuiteReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for comparison in &report.comparisons {
        let _ = writeln!(out, "{}", render_case_comparison(comparison));
    }
    if !report.new_cases.is_empty() {
        let _ = writeln!(out, "new (no baseline): {}", report.new_cases.join(", "));
    }
    if !report.missing_cases.is_empty() {
        let _ = writeln!(
            out,
            "missing (not run): {}",
            report.missing_cases.join(", ")
        );
    }
    if report
        .comparisons
        .iter()
        .any(|comparison| !comparison.rss_gated)
    {
        let _ = writeln!(
            out,
            "* rss shown for trend only; not gating (enable with --gate-rss)"
        );
    }
    let regressions = report
        .comparisons
        .iter()
        .filter(|comparison| comparison.is_regression())
        .count();
    if regressions == 0 {
        let _ = write!(
            out,
            "verdict: PASS ({} case(s) within band)",
            report.comparisons.len()
        );
    } else {
        let _ = write!(out, "verdict: REGRESSION ({regressions} case(s) over band)");
    }
    out
}

/// Format a relative change as a signed percentage, or `n/a` when absent.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn fmt_pct(ratio: Option<f64>) -> String {
    match ratio {
        Some(value) => format!("{:+.1}%", value * 100.0),
        None => "n/a".to_string(),
    }
}

/// `(current - baseline) / baseline`. `None` when the baseline is zero, where
/// a relative change is undefined.
///
/// Micro-second counts for any realistic command duration are far below
/// f64's exact-integer range, so the casts do not lose meaningful precision.
#[allow(clippy::cast_precision_loss)]
fn ratio(baseline: u64, current: u64) -> Option<f64> {
    if baseline == 0 {
        return None;
    }
    Some((current as f64 - baseline as f64) / baseline as f64)
}

/// Keep the configured relative tolerance as the floor, but widen it when the
/// baseline/current samples themselves show more noise. Two combined standard
/// deviations is intentionally conservative for sub-10ms subprocess cases.
#[allow(clippy::cast_precision_loss)]
fn effective_time_tolerance(
    configured: f64,
    baseline_micros: u64,
    baseline_stddev_micros: u64,
    current_stddev_micros: u64,
) -> f64 {
    if baseline_micros == 0 {
        return configured;
    }
    let combined_stddev = (baseline_stddev_micros as f64).hypot(current_stddev_micros as f64);
    let noise_floor = (combined_stddev * 2.0) / baseline_micros as f64;
    configured.max(noise_floor)
}

fn duration_micros(value: Duration) -> u64 {
    u64::try_from(value.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(name: &str, median_ms: u64, rss: Option<u64>) -> PerfSummary {
        PerfSummary {
            name: name.to_string(),
            runs: 5,
            min_elapsed: Duration::from_millis(median_ms.saturating_sub(1)),
            max_elapsed: Duration::from_millis(median_ms + 1),
            mean_elapsed: Duration::from_millis(median_ms),
            median_elapsed: Duration::from_millis(median_ms),
            stddev_elapsed: Duration::from_micros(500),
            peak_rss_kib: rss,
        }
    }

    #[test]
    fn baseline_round_trips_through_json() {
        let mut file = PerfBaselineFile::empty();
        file.upsert(PerfBaseline::from_summary(&summary(
            "startup-version",
            100,
            Some(8000),
        )));
        file.upsert(PerfBaseline::from_summary(&summary(
            "startup-help",
            120,
            Some(9000),
        )));

        let json = file.to_json();
        let parsed = PerfBaselineFile::from_json(&json).expect("round trip");
        assert_eq!(parsed, file);
        assert_eq!(parsed.schema, BASELINE_SCHEMA_VERSION);
    }

    #[test]
    fn upsert_replaces_and_keeps_cases_sorted() {
        let mut file = PerfBaselineFile::empty();
        file.upsert(PerfBaseline::from_summary(&summary("zeta", 10, None)));
        file.upsert(PerfBaseline::from_summary(&summary("alpha", 20, None)));
        // Replace zeta in place rather than duplicating it.
        file.upsert(PerfBaseline::from_summary(&summary("zeta", 30, None)));

        let names: Vec<_> = file.cases.iter().map(|case| case.name.as_str()).collect();
        assert_eq!(names, ["alpha", "zeta"]);
        assert_eq!(file.get("zeta").unwrap().median_micros, 30_000);
    }

    #[test]
    fn classify_respects_inclusive_band() {
        // Exactly at the tolerance is still within band.
        assert_eq!(PerfVerdict::classify(0.05, 0.05), PerfVerdict::Unchanged);
        assert_eq!(PerfVerdict::classify(0.0501, 0.05), PerfVerdict::Regressed);
        assert_eq!(PerfVerdict::classify(-0.0501, 0.05), PerfVerdict::Improved);
        assert_eq!(PerfVerdict::classify(0.0, 0.05), PerfVerdict::Unchanged);
    }

    #[test]
    fn gate_flags_latency_regression_beyond_band() {
        let gate = RegressionGate::default();
        let baseline = PerfBaseline::from_summary(&summary("c", 100, Some(8000)));
        // +20% slower than the 100ms baseline.
        let current = summary("c", 120, Some(8000));
        let comparison = gate.compare(&baseline, &current);
        assert_eq!(comparison.time_verdict, PerfVerdict::Regressed);
        assert_eq!(comparison.rss_verdict, PerfVerdict::Unchanged);
        assert!(comparison.is_regression());
    }

    #[test]
    fn gate_treats_short_process_jitter_inside_noise_floor_as_unchanged() {
        let gate = RegressionGate::default();
        let baseline = PerfBaseline::from_summary(&PerfSummary {
            name: "startup-help".to_string(),
            runs: 9,
            min_elapsed: Duration::from_micros(5_719),
            max_elapsed: Duration::from_micros(6_840),
            mean_elapsed: Duration::from_micros(6_179),
            median_elapsed: Duration::from_micros(6_080),
            stddev_elapsed: Duration::from_micros(372),
            peak_rss_kib: Some(2_048),
        });
        let current = PerfSummary {
            name: "startup-help".to_string(),
            runs: 15,
            min_elapsed: Duration::from_micros(6_488),
            max_elapsed: Duration::from_micros(8_613),
            mean_elapsed: Duration::from_micros(7_336),
            median_elapsed: Duration::from_micros(7_357),
            stddev_elapsed: Duration::from_micros(752),
            peak_rss_kib: Some(2_688),
        };

        let comparison = gate.compare(&baseline, &current);

        assert_eq!(comparison.time_verdict, PerfVerdict::Unchanged);
        assert!(!comparison.is_regression());
    }

    #[test]
    fn gate_flags_memory_regression_only_when_rss_gating_enabled() {
        let baseline = PerfBaseline::from_summary(&summary("c", 100, Some(8000)));
        // Same speed, +25% memory.
        let current = summary("c", 100, Some(10_000));

        // Default: RSS is report-only, so a memory jump does NOT fail the gate.
        let report_only = RegressionGate::default().compare(&baseline, &current);
        assert_eq!(report_only.rss_verdict, PerfVerdict::Regressed);
        assert!(!report_only.rss_gated);
        assert!(!report_only.is_regression());

        // Opt in: now the same memory jump fails the gate.
        let gated = RegressionGate::default()
            .gating_rss(true)
            .compare(&baseline, &current);
        assert_eq!(gated.time_verdict, PerfVerdict::Unchanged);
        assert_eq!(gated.rss_verdict, PerfVerdict::Regressed);
        assert!(gated.rss_gated);
        assert!(gated.is_regression());
    }

    #[test]
    fn gate_reports_improvement_and_unknown_rss() {
        let gate = RegressionGate::default();
        let baseline = PerfBaseline::from_summary(&summary("c", 100, None));
        // 30% faster, and no RSS on either side -> rss unknown.
        let current = summary("c", 70, None);
        let comparison = gate.compare(&baseline, &current);
        assert_eq!(comparison.time_verdict, PerfVerdict::Improved);
        assert_eq!(comparison.rss_verdict, PerfVerdict::Unknown);
        assert!(!comparison.is_regression());
    }

    #[test]
    fn suite_report_tracks_new_missing_and_exit_code() {
        let gate = RegressionGate::default();
        let mut baseline = PerfBaselineFile::empty();
        baseline.upsert(PerfBaseline::from_summary(&summary(
            "kept",
            100,
            Some(8000),
        )));
        baseline.upsert(PerfBaseline::from_summary(&summary(
            "gone",
            100,
            Some(8000),
        )));

        let summaries = vec![
            summary("kept", 102, Some(8000)), // within band
            summary("fresh", 50, Some(4000)), // no baseline -> new
        ];
        let report = SuiteReport::evaluate(&gate, &baseline, &summaries);

        assert_eq!(report.comparisons.len(), 1);
        assert_eq!(report.new_cases, ["fresh"]);
        assert_eq!(report.missing_cases, ["gone"]);
        assert!(!report.has_regression());
        assert_eq!(report.exit_code(), EXIT_OK);
    }

    #[test]
    fn suite_report_exit_code_signals_regression() {
        let gate = RegressionGate::default();
        let mut baseline = PerfBaselineFile::empty();
        baseline.upsert(PerfBaseline::from_summary(&summary("hot", 100, Some(8000))));
        let summaries = vec![summary("hot", 200, Some(8000))];

        let report = SuiteReport::evaluate(&gate, &baseline, &summaries);
        assert!(report.has_regression());
        assert_eq!(report.exit_code(), EXIT_REGRESSION);
    }

    #[test]
    fn default_scenarios_target_the_given_binary() {
        let cases = default_scenarios("/tmp/zo", 2, 5);
        assert_eq!(cases.len(), 4);
        assert!(cases.iter().all(|case| case.command == "/tmp/zo"));
        assert!(cases.iter().all(|case| case.warmups == 2 && case.runs == 5));
        assert_eq!(cases[0].name, "startup-version");
        // The deterministic, network-free startup scenarios are all present.
        let names: Vec<_> = cases.iter().map(|case| case.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "startup-version",
                "startup-help",
                "startup-system-prompt",
                "startup-bootstrap-plan",
            ]
        );
        // No scenario shells out to a prompt turn (that would need the network).
        assert!(cases
            .iter()
            .all(|case| !case.args.iter().any(|arg| arg == "-p")));
    }

    #[test]
    fn render_suite_report_is_human_readable() {
        let gate = RegressionGate::default();
        let mut baseline = PerfBaselineFile::empty();
        baseline.upsert(PerfBaseline::from_summary(&summary("hot", 100, Some(8000))));
        let report = SuiteReport::evaluate(&gate, &baseline, &[summary("hot", 200, Some(8000))]);
        let text = render_suite_report(&report);
        assert!(text.contains("REGRESSED"));
        assert!(text.contains("verdict: REGRESSION"));
    }

    #[test]
    fn ratio_guards_zero_baseline() {
        assert_eq!(ratio(0, 100), None);
        assert_eq!(ratio(100, 150), Some(0.5));
    }
}
