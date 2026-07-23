//! Lane×runner pass-rate denominators computed from the run ledger.
//!
//! The spec forbids hiding invalid or inconclusive runs behind a single pass
//! rate, and forbids averaging lanes together. So every `(lane, runner)` cell
//! is keyed separately and reports the full set of denominators — strict,
//! adjudicated, inconclusive, invalid, blocked, and artifact-preservation —
//! with the raw counts alongside the rates so an excluded count is never
//! silently dropped.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One ledger row: one recorded run's final verdict, as written by the
/// orchestrator's `emit_decisions`.
#[derive(Debug, Clone, Deserialize)]
pub struct LedgerRow {
    pub runner: String,
    pub lane: String,
    #[serde(default)]
    pub fixture: String,
    pub final_decision: String,
    #[serde(default)]
    pub failure_class: Option<String>,
    #[serde(default)]
    pub leaderboard_eligible: bool,
    #[serde(default)]
    pub wall_seconds: Option<u64>,
    #[serde(default)]
    pub token_total: Option<u64>,
    #[serde(default)]
    pub token_output: Option<u64>,
    #[serde(default)]
    pub total_tokens_per_second: Option<f64>,
    #[serde(default)]
    pub output_tokens_per_second: Option<f64>,
    #[serde(default)]
    pub deterministic_probe_failure_count: Option<usize>,
    #[serde(default)]
    pub phase_plan_millis: Option<u64>,
    #[serde(default)]
    pub phase_exec_millis: Option<u64>,
    #[serde(default)]
    pub phase_test_millis: Option<u64>,
    #[serde(default)]
    pub phase_verify_millis: Option<u64>,
    #[serde(default)]
    pub phase_repair_millis: Option<u64>,
    #[serde(default)]
    pub retry_count: Option<u32>,
    #[serde(default)]
    pub dirty_diff: Option<bool>,
    #[serde(default)]
    pub deep_verifier_failed: Option<bool>,
    #[serde(default)]
    pub timeout: Option<bool>,
    #[serde(default)]
    pub provider_error: Option<bool>,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub cost_normalized_score: Option<f64>,
}

impl LedgerRow {
    fn has_observed_metrics(&self) -> bool {
        self.wall_seconds.is_some()
            || self.token_total.is_some()
            || self.token_output.is_some()
            || self.retry_count.is_some()
            || self.dirty_diff.is_some()
            || self.deep_verifier_failed.is_some()
            || self.timeout.is_some()
            || self.provider_error.is_some()
            || self.cost_usd.is_some()
            || self.cost_normalized_score.is_some()
    }
}

/// Pass-rate denominators for one `(lane, runner)` cell. Counts accompany rates
/// so a report can show the excluded/inconclusive counts the spec requires.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LaneRunnerSummary {
    pub attempted: u32,
    pub eligible: u32,
    pub accepted: u32,
    pub rejected: u32,
    pub inconclusive: u32,
    pub invalid: u32,
    pub blocked: u32,
    pub artifact_preservation_failed: u32,
    pub mean_wall_seconds: Option<f64>,
    pub median_wall_seconds: Option<f64>,
    pub total_token_usage: u64,
    pub total_output_token_usage: u64,
    pub unknown_token_usage: u32,
    pub aggregate_total_tokens_per_second: Option<f64>,
    pub aggregate_output_tokens_per_second: Option<f64>,
    pub mean_phase_millis: BTreeMap<String, f64>,
    pub deterministic_probe_failure_count: usize,
    pub total_retry_count: u32,
    pub mean_retry_count: Option<f64>,
    pub dirty_diff_count: u32,
    pub deep_verifier_failure_count: u32,
    pub timeout_count: u32,
    pub provider_error_count: u32,
    pub total_cost_usd: Option<f64>,
    pub cost_normalized_score: Option<f64>,
    pub strict_pass_rate: f64,
    pub adjudicated_pass_rate: f64,
    pub inconclusive_rate: f64,
    pub invalid_rate: f64,
    pub blocked_rate: f64,
    pub artifact_preservation_failure_rate: f64,
    pub dirty_diff_rate: f64,
    pub deep_verifier_failure_rate: f64,
    pub timeout_rate: f64,
    pub provider_error_rate: f64,
    #[serde(skip)]
    wall_samples: Vec<u64>,
    #[serde(skip)]
    retry_samples: Vec<u32>,
    #[serde(skip)]
    output_token_samples: Vec<u64>,
    #[serde(skip)]
    phase_samples: BTreeMap<String, Vec<u64>>,
}

/// `num / den` as a fraction, or `0.0` when the denominator is empty. `f64::from`
/// is lossless for these counts, so no precision is lost.
fn rate(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        f64::from(num) / f64::from(den)
    }
}

fn mean_u64(samples: &[u64]) -> Option<f64> {
    let den = sample_len_f64(samples.len())?;
    let total: Duration = samples
        .iter()
        .map(|seconds| Duration::from_secs(*seconds))
        .sum();
    Some(total.as_secs_f64() / den)
}

fn mean_u32(samples: &[u32]) -> Option<f64> {
    let den = sample_len_f64(samples.len())?;
    let sum: u32 = samples.iter().sum();
    Some(f64::from(sum) / den)
}

fn median_u64(samples: &mut [u64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        Some(f64::midpoint(
            Duration::from_secs(samples[mid - 1]).as_secs_f64(),
            Duration::from_secs(samples[mid]).as_secs_f64(),
        ))
    } else {
        Some(Duration::from_secs(samples[mid]).as_secs_f64())
    }
}

fn sample_len_f64(len: usize) -> Option<f64> {
    if len == 0 {
        return None;
    }
    u32::try_from(len).ok().map(f64::from)
}

#[allow(clippy::cast_precision_loss)]
fn ratio_u64(numerator: u64, denominator: u64) -> f64 {
    numerator as f64 / denominator as f64
}

/// Plain numeric mean of `samples` (no `Duration` reinterpretation — used for
/// token counts, where the values are not seconds). `None` for an empty slice.
#[allow(clippy::cast_precision_loss)] // token counts are far below f64's exact-int range
fn mean_plain_u64(samples: &[u64]) -> Option<f64> {
    let den = sample_len_f64(samples.len())?;
    let total: u128 = samples.iter().map(|&v| u128::from(v)).sum();
    Some(total as f64 / den)
}

/// Plain numeric median of `samples` (sorts in place). `None` for an empty slice.
#[allow(clippy::cast_precision_loss)]
fn median_plain_u64(samples: &mut [u64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        Some(f64::midpoint(samples[mid - 1] as f64, samples[mid] as f64))
    } else {
        Some(samples[mid] as f64)
    }
}

/// One `(lane, runner, fixture)` cell aggregated across repeated runs: the
/// reliability view the lane×runner rollup cannot give. `pass_at_n` is the
/// fraction of repeats that were accepted, and the medians are robust to the
/// single-run outliers that a one-shot benchmark mistakes for signal.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RepeatCellSummary {
    pub repeats: u32,
    pub accepted: u32,
    pub pass_at_n: f64,
    pub mean_wall_seconds: Option<f64>,
    pub median_wall_seconds: Option<f64>,
    pub mean_token_total: Option<f64>,
    pub median_token_total: Option<f64>,
    #[serde(skip)]
    wall_samples: Vec<u64>,
    #[serde(skip)]
    token_samples: Vec<u64>,
}

/// Aggregate repeated runs into per-`(lane, runner, fixture)` reliability cells,
/// keyed `"<lane>/<runner>/<fixture>"`. Distinct from [`summarize_ledger`], which
/// rolls up by lane×runner: this keeps each fixture separate so pass@N and the
/// wall/token medians describe one concrete task, not a lane average.
#[must_use]
pub fn summarize_repeats(rows: &[LedgerRow]) -> BTreeMap<String, RepeatCellSummary> {
    let mut groups: BTreeMap<String, RepeatCellSummary> = BTreeMap::new();
    for row in rows {
        let entry = groups
            .entry(format!("{}/{}/{}", row.lane, row.runner, row.fixture))
            .or_default();
        entry.repeats += 1;
        if row.final_decision == "accepted" {
            entry.accepted += 1;
        }
        if let Some(wall) = row.wall_seconds {
            entry.wall_samples.push(wall);
        }
        if let Some(tokens) = row.token_total {
            entry.token_samples.push(tokens);
        }
    }
    for cell in groups.values_mut() {
        cell.pass_at_n = rate(cell.accepted, cell.repeats);
        cell.mean_wall_seconds = mean_u64(&cell.wall_samples);
        let mut wall = cell.wall_samples.clone();
        cell.median_wall_seconds = median_u64(&mut wall);
        cell.mean_token_total = mean_plain_u64(&cell.token_samples);
        let mut tokens = cell.token_samples.clone();
        cell.median_token_total = median_plain_u64(&mut tokens);
    }
    groups
}

fn push_phase_sample(samples: &mut BTreeMap<String, Vec<u64>>, name: &str, value: Option<u64>) {
    if let Some(value) = value {
        samples.entry(name.to_string()).or_default().push(value);
    }
}

/// Parse a ledger file: one JSON object per line. Blank and malformed lines are
/// skipped so a partially written ledger still summarizes the rows it has.
#[must_use]
pub fn parse_ledger(text: &str) -> Vec<LedgerRow> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// Aggregate ledger rows into per-`(lane, runner)` denominators, keyed
/// `"<lane>/<runner>"`. Lanes are never merged.
///
/// `strict_pass_rate` divides by leaderboard-eligible runs (so invalid-fairness
/// runs drop out of the denominator); the inconclusive/invalid/blocked rates
/// divide by all attempted runs so nothing is hidden.
#[must_use]
pub fn summarize_ledger(rows: &[LedgerRow]) -> BTreeMap<String, LaneRunnerSummary> {
    let mut groups: BTreeMap<String, LaneRunnerSummary> = BTreeMap::new();
    for row in rows {
        let entry = groups
            .entry(format!("{}/{}", row.lane, row.runner))
            .or_default();
        entry.attempted += 1;
        if row.leaderboard_eligible {
            entry.eligible += 1;
        }
        match row.final_decision.as_str() {
            "accepted" => entry.accepted += 1,
            "rejected" => entry.rejected += 1,
            "inconclusive" => entry.inconclusive += 1,
            "invalid" => entry.invalid += 1,
            "blocked" => entry.blocked += 1,
            _ => {}
        }
        if row.failure_class.as_deref() == Some("artifact_preservation_failed") {
            entry.artifact_preservation_failed += 1;
        }
        if let Some(wall) = row.wall_seconds {
            entry.wall_samples.push(wall);
        }
        if let Some(tokens) = row.token_total {
            entry.total_token_usage += tokens;
        } else if row.has_observed_metrics() {
            entry.unknown_token_usage += 1;
        }
        if let Some(tokens) = row.token_output {
            entry.total_output_token_usage += tokens;
            entry.output_token_samples.push(tokens);
        }
        entry.deterministic_probe_failure_count = entry
            .deterministic_probe_failure_count
            .saturating_add(row.deterministic_probe_failure_count.unwrap_or(0));
        push_phase_sample(&mut entry.phase_samples, "plan", row.phase_plan_millis);
        push_phase_sample(&mut entry.phase_samples, "exec", row.phase_exec_millis);
        push_phase_sample(&mut entry.phase_samples, "test", row.phase_test_millis);
        push_phase_sample(&mut entry.phase_samples, "verify", row.phase_verify_millis);
        push_phase_sample(&mut entry.phase_samples, "repair", row.phase_repair_millis);
        if let Some(retry_count) = row.retry_count {
            entry.total_retry_count += retry_count;
            entry.retry_samples.push(retry_count);
        }
        if row.dirty_diff.unwrap_or(false) {
            entry.dirty_diff_count += 1;
        }
        if row.deep_verifier_failed.unwrap_or(false) {
            entry.deep_verifier_failure_count += 1;
        }
        if row.timeout.unwrap_or(false) {
            entry.timeout_count += 1;
        }
        if row.provider_error.unwrap_or(false) {
            entry.provider_error_count += 1;
        }
        if let Some(cost) = row.cost_usd {
            entry.total_cost_usd = Some(entry.total_cost_usd.unwrap_or(0.0) + cost);
        }
    }
    for summary in groups.values_mut() {
        summary.mean_wall_seconds = mean_u64(&summary.wall_samples);
        let mut wall_samples = summary.wall_samples.clone();
        summary.median_wall_seconds = median_u64(&mut wall_samples);
        let wall_sum: u64 = summary.wall_samples.iter().sum();
        if wall_sum > 0 {
            summary.aggregate_total_tokens_per_second =
                Some(ratio_u64(summary.total_token_usage, wall_sum));
            summary.aggregate_output_tokens_per_second =
                Some(ratio_u64(summary.total_output_token_usage, wall_sum));
        }
        summary.mean_phase_millis = summary
            .phase_samples
            .iter()
            .filter_map(|(name, samples)| mean_plain_u64(samples).map(|mean| (name.clone(), mean)))
            .collect();
        summary.mean_retry_count = mean_u32(&summary.retry_samples);
        summary.cost_normalized_score = summary
            .total_cost_usd
            .filter(|cost| *cost > 0.0)
            .map(|cost| f64::from(summary.accepted) / cost);
        summary.strict_pass_rate = rate(summary.accepted, summary.eligible);
        summary.adjudicated_pass_rate = rate(summary.accepted, summary.accepted + summary.rejected);
        summary.inconclusive_rate = rate(summary.inconclusive, summary.attempted);
        summary.invalid_rate = rate(summary.invalid, summary.attempted);
        summary.blocked_rate = rate(summary.blocked, summary.attempted);
        summary.artifact_preservation_failure_rate =
            rate(summary.artifact_preservation_failed, summary.attempted);
        summary.dirty_diff_rate = rate(summary.dirty_diff_count, summary.attempted);
        summary.deep_verifier_failure_rate =
            rate(summary.deep_verifier_failure_count, summary.attempted);
        summary.timeout_rate = rate(summary.timeout_count, summary.attempted);
        summary.provider_error_rate = rate(summary.provider_error_count, summary.attempted);
    }
    groups
}

/// One runner's normalized standing within a lane: the "who dominates" numbers a
/// one-shot table cannot give. Token/wall are scored as the **geometric mean of
/// per-fixture ratios against the per-fixture best runner** (SWE-bench style), so
/// a single huge fixture can't dominate the average and the best runner on every
/// fixture scores exactly 100. `composite` is the four-way report's published
/// formula: `0.50*accuracy + 0.25*speed + 0.25*token`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RunnerScore {
    pub runner: String,
    pub fixtures_scored: u32,
    pub accepted: u32,
    /// Accepted ÷ all runs for this runner in the lane (a "did it work" rate).
    /// Deliberately not `summarize_ledger`'s `strict_pass_rate`, which divides by
    /// leaderboard-eligible runs — different denominator, hence a different name.
    pub pass_rate: f64,
    /// Geomean of (this runner's median tokens ÷ the fixture's best median
    /// tokens) over fixtures it ran. `1.0` = best everywhere; higher is worse.
    pub geomean_token_ratio: Option<f64>,
    pub geomean_wall_ratio: Option<f64>,
    /// `100 / geomean_*_ratio` (best → 100, `0` when no data). Higher is better.
    pub token_score: f64,
    pub speed_score: f64,
    pub accuracy_score: f64,
    pub composite: f64,
}

/// Geometric mean of strictly-positive ratios, ignoring non-positive entries.
/// `None` when nothing usable remains (so a runner with no measured fixtures
/// scores 0 rather than a fabricated 1.0).
#[allow(clippy::cast_precision_loss)] // fixture counts are tiny
fn geomean(ratios: &[f64]) -> Option<f64> {
    let usable: Vec<f64> = ratios.iter().copied().filter(|r| *r > 0.0).collect();
    if usable.is_empty() {
        return None;
    }
    let sum_ln: f64 = usable.iter().map(|r| r.ln()).sum();
    Some((sum_ln / usable.len() as f64).exp())
}

/// Per-`(lane, runner, fixture)` central tendency, used to normalize the
/// leaderboard. Medians make it robust to a single noisy repeat.
#[derive(Default)]
struct CellAgg {
    wall_samples: Vec<u64>,
    token_samples: Vec<u64>,
    accepted: u32,
    total: u32,
}

/// Rank runners within each lane on accuracy, token efficiency, and speed, keyed
/// by lane. Distinct from [`summarize_ledger`] (lane×runner denominators) and
/// [`summarize_repeats`] (per-fixture reliability): this is the cross-runner
/// **dominance** view — the leaderboard the four-way report computes by hand.
///
/// Lanes are never merged. Token/wall scores normalize per fixture against the
/// best runner on that fixture, so they are only meaningful when ≥2 runners ran
/// the same fixtures; with a single runner every ratio is `1.0` (it is its own
/// best) and the scores collapse to accuracy.
#[must_use]
pub fn build_leaderboard(rows: &[LedgerRow]) -> BTreeMap<String, Vec<RunnerScore>> {
    // lane -> fixture -> runner -> aggregate
    type LaneCells = BTreeMap<String, BTreeMap<String, BTreeMap<String, CellAgg>>>;
    let mut cells: LaneCells = BTreeMap::new();
    for row in rows {
        let cell = cells
            .entry(row.lane.clone())
            .or_default()
            .entry(row.fixture.clone())
            .or_default()
            .entry(row.runner.clone())
            .or_default();
        cell.total += 1;
        if row.final_decision == "accepted" {
            cell.accepted += 1;
        }
        if let Some(wall) = row.wall_seconds {
            cell.wall_samples.push(wall);
        }
        if let Some(tokens) = row.token_total {
            cell.token_samples.push(tokens);
        }
    }

    let mut leaderboard: BTreeMap<String, Vec<RunnerScore>> = BTreeMap::new();
    for (lane, fixtures) in &cells {
        // Per fixture: the best (minimum) median tokens / wall across runners,
        // and this runner's own median, so we can form per-fixture ratios.
        let mut token_ratios: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut wall_ratios: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut accepted: BTreeMap<&str, u32> = BTreeMap::new();
        let mut total: BTreeMap<&str, u32> = BTreeMap::new();

        for runners in fixtures.values() {
            let medians: BTreeMap<&str, (Option<f64>, Option<f64>)> = runners
                .iter()
                .map(|(runner, agg)| {
                    let mut tok = agg.token_samples.clone();
                    let mut wall = agg.wall_samples.clone();
                    (
                        runner.as_str(),
                        (median_plain_u64(&mut tok), median_u64(&mut wall)),
                    )
                })
                .collect();
            let best_token = medians
                .values()
                .filter_map(|(t, _)| *t)
                .filter(|t| *t > 0.0)
                .fold(f64::INFINITY, f64::min);
            let best_wall = medians
                .values()
                .filter_map(|(_, w)| *w)
                .filter(|w| *w > 0.0)
                .fold(f64::INFINITY, f64::min);
            for (runner, (tok, wall)) in &medians {
                if let Some(t) = tok {
                    if best_token.is_finite() && best_token > 0.0 {
                        token_ratios.entry(runner).or_default().push(t / best_token);
                    }
                }
                if let Some(w) = wall {
                    if best_wall.is_finite() && best_wall > 0.0 {
                        wall_ratios.entry(runner).or_default().push(w / best_wall);
                    }
                }
            }
            for (runner, agg) in runners {
                *accepted.entry(runner.as_str()).or_default() += agg.accepted;
                *total.entry(runner.as_str()).or_default() += agg.total;
            }
        }

        let mut scores: Vec<RunnerScore> = total
            .keys()
            .map(|runner| {
                let acc = accepted.get(runner).copied().unwrap_or(0);
                let tot = total.get(runner).copied().unwrap_or(0);
                let strict = rate(acc, tot);
                let token_geomean = token_ratios.get(runner).and_then(|r| geomean(r));
                let wall_geomean = wall_ratios.get(runner).and_then(|r| geomean(r));
                let token_score = token_geomean.map_or(0.0, |g| 100.0 / g);
                let speed_score = wall_geomean.map_or(0.0, |g| 100.0 / g);
                let accuracy_score = strict * 100.0;
                RunnerScore {
                    runner: (*runner).to_string(),
                    fixtures_scored: u32::try_from(fixtures.len()).unwrap_or(u32::MAX),
                    accepted: acc,
                    pass_rate: strict,
                    geomean_token_ratio: token_geomean,
                    geomean_wall_ratio: wall_geomean,
                    token_score,
                    speed_score,
                    accuracy_score,
                    composite: 0.50 * accuracy_score + 0.25 * speed_score + 0.25 * token_score,
                }
            })
            .collect();
        // Highest composite first; ties broken by accuracy then runner name so the
        // ordering is deterministic (NaN cannot appear — all inputs are finite).
        scores.sort_by(|a, b| {
            b.composite
                .total_cmp(&a.composite)
                .then(b.pass_rate.total_cmp(&a.pass_rate))
                .then(a.runner.cmp(&b.runner))
        });
        leaderboard.insert(lane.clone(), scores);
    }
    leaderboard
}

/// Back-compatible leaderboard JSON: lane keys stay at the top level, while
/// `_views` adds alternative ranking orders for reports that care about one axis.
#[must_use]
pub fn build_leaderboard_with_views(rows: &[LedgerRow]) -> Value {
    let composite = build_leaderboard(rows);
    let mut root = serde_json::Map::new();
    for (lane, scores) in &composite {
        root.insert(lane.clone(), json!(scores));
    }
    root.insert(
        "_views".to_string(),
        json!({
            "composite": composite,
            "accuracy_first": ranked_view(rows, ViewRank::Accuracy),
            "latency_first": ranked_view(rows, ViewRank::Latency),
            "token_first": ranked_view(rows, ViewRank::Token),
        }),
    );
    Value::Object(root)
}

#[derive(Clone, Copy)]
enum ViewRank {
    Accuracy,
    Latency,
    Token,
}

fn ranked_view(rows: &[LedgerRow], rank: ViewRank) -> BTreeMap<String, Vec<RunnerScore>> {
    let mut board = build_leaderboard(rows);
    for scores in board.values_mut() {
        scores.sort_by(|a, b| match rank {
            ViewRank::Accuracy => b
                .accuracy_score
                .total_cmp(&a.accuracy_score)
                .then(b.composite.total_cmp(&a.composite))
                .then(a.runner.cmp(&b.runner)),
            ViewRank::Latency => b
                .speed_score
                .total_cmp(&a.speed_score)
                .then(b.accuracy_score.total_cmp(&a.accuracy_score))
                .then(a.runner.cmp(&b.runner)),
            ViewRank::Token => b
                .token_score
                .total_cmp(&a.token_score)
                .then(b.accuracy_score.total_cmp(&a.accuracy_score))
                .then(a.runner.cmp(&b.runner)),
        });
    }
    board
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEDGER: &str = r#"
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true}
{"runner":"zo","lane":"deep","fixture":"b","final_decision":"rejected","failure_class":"verifier_semantic_reject","leaderboard_eligible":true}
{"runner":"zo","lane":"deep","fixture":"c","final_decision":"inconclusive","failure_class":"verifier_timeout","leaderboard_eligible":true}
{"runner":"zo","lane":"deep","fixture":"d","final_decision":"invalid","failure_class":"fairness_contract_invalid","leaderboard_eligible":false}
{"runner":"claude","lane":"fast","fixture":"a","final_decision":"accepted","leaderboard_eligible":true}
"#;

    #[test]
    fn parse_skips_blank_and_malformed_lines() {
        let rows = parse_ledger(
            "\n{not json}\n{\"runner\":\"x\",\"lane\":\"y\",\"final_decision\":\"accepted\"}\n",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].runner, "x");
    }

    #[test]
    fn strict_rate_excludes_invalid_fairness_from_denominator() {
        let summary = summarize_ledger(&parse_ledger(LEDGER));
        let deep = &summary["deep/zo"];
        assert_eq!(deep.attempted, 4);
        // The invalid-fairness run is not leaderboard-eligible.
        assert_eq!(deep.eligible, 3);
        assert_eq!(deep.accepted, 1);
        assert_eq!(deep.unknown_token_usage, 0);
        // strict = accepted / eligible = 1/3.
        assert!((deep.strict_pass_rate - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn adjudicated_rate_divides_by_accepted_plus_rejected() {
        let summary = summarize_ledger(&parse_ledger(LEDGER));
        let deep = &summary["deep/zo"];
        // adjudicated = accepted / (accepted + rejected) = 1/2.
        assert!((deep.adjudicated_pass_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn inconclusive_and_invalid_rates_divide_by_all_attempted() {
        let summary = summarize_ledger(&parse_ledger(LEDGER));
        let deep = &summary["deep/zo"];
        assert_eq!(deep.inconclusive, 1);
        assert_eq!(deep.invalid, 1);
        // both over attempted=4.
        assert!((deep.inconclusive_rate - 0.25).abs() < 1e-9);
        assert!((deep.invalid_rate - 0.25).abs() < 1e-9);
    }

    #[test]
    fn lanes_are_keyed_separately_never_merged() {
        let summary = summarize_ledger(&parse_ledger(LEDGER));
        assert!(summary.contains_key("deep/zo"));
        assert!(summary.contains_key("fast/claude"));
        assert_eq!(summary["fast/claude"].accepted, 1);
        assert_eq!(summary["fast/claude"].attempted, 1);
    }

    #[test]
    fn empty_denominators_are_zero_not_nan() {
        // A cell with no accepted/rejected runs must not divide by zero.
        let rows = parse_ledger(
            r#"{"runner":"zo","lane":"deep","final_decision":"inconclusive","leaderboard_eligible":false}"#,
        );
        let summary = summarize_ledger(&rows);
        let cell = &summary["deep/zo"];
        // Both rates are exactly 0.0 (a zero numerator), asserted with the same
        // epsilon idiom this file uses elsewhere rather than a strict float ==.
        assert!(cell.strict_pass_rate.abs() < 1e-9);
        assert!(cell.adjudicated_pass_rate.abs() < 1e-9);
        assert!(cell.strict_pass_rate.is_finite());
    }

    #[test]
    fn summarizes_observed_metrics_per_lane_runner() {
        let rows = parse_ledger(
            r#"
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":10,"token_total":100,"retry_count":0,"dirty_diff":false,"deep_verifier_failed":false,"timeout":false,"provider_error":false,"cost_usd":2.0}
{"runner":"zo","lane":"deep","fixture":"b","final_decision":"rejected","leaderboard_eligible":true,"wall_seconds":20,"token_total":300,"retry_count":2,"dirty_diff":true,"deep_verifier_failed":true,"timeout":true,"provider_error":false,"cost_usd":3.0}
{"runner":"zo","lane":"deep","fixture":"c","final_decision":"inconclusive","leaderboard_eligible":false,"wall_seconds":40,"retry_count":1,"dirty_diff":false,"deep_verifier_failed":false,"timeout":false,"provider_error":true}
"#,
        );
        let summary = summarize_ledger(&rows);
        let deep = &summary["deep/zo"];
        assert!((deep.mean_wall_seconds.unwrap() - (70.0 / 3.0)).abs() < 1e-9);
        assert!((deep.median_wall_seconds.unwrap() - 20.0).abs() < 1e-9);
        assert_eq!(deep.total_token_usage, 400);
        assert_eq!(deep.unknown_token_usage, 1);
        assert_eq!(deep.total_retry_count, 3);
        assert!((deep.mean_retry_count.unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(deep.dirty_diff_count, 1);
        assert!((deep.dirty_diff_rate - (1.0 / 3.0)).abs() < 1e-9);
        assert_eq!(deep.deep_verifier_failure_count, 1);
        assert_eq!(deep.timeout_count, 1);
        assert_eq!(deep.provider_error_count, 1);
        assert!((deep.total_cost_usd.unwrap() - 5.0).abs() < 1e-9);
        assert!((deep.cost_normalized_score.unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn summarize_repeats_computes_pass_at_n_and_medians_per_fixture() {
        // Same fixture run 3x (2 accepted, 1 rejected) → pass@3 = 2/3, median
        // wall = middle of {10,20,40} = 20, median tokens = middle of
        // {100,200,400} = 200. A different fixture stays its own cell.
        let rows = parse_ledger(
            r#"
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":10,"token_total":100}
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"rejected","leaderboard_eligible":true,"wall_seconds":40,"token_total":400}
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":20,"token_total":200}
{"runner":"zo","lane":"deep","fixture":"b","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":5,"token_total":50}
"#,
        );
        let repeats = summarize_repeats(&rows);

        let a = &repeats["deep/zo/a"];
        assert_eq!(a.repeats, 3);
        assert_eq!(a.accepted, 2);
        assert!((a.pass_at_n - 2.0 / 3.0).abs() < 1e-9);
        assert!((a.median_wall_seconds.unwrap() - 20.0).abs() < 1e-9);
        assert!((a.median_token_total.unwrap() - 200.0).abs() < 1e-9);
        assert!((a.mean_token_total.unwrap() - (700.0 / 3.0)).abs() < 1e-9);

        // Fixtures are never merged: b is its own perfect cell.
        let b = &repeats["deep/zo/b"];
        assert_eq!(b.repeats, 1);
        assert!((b.pass_at_n - 1.0).abs() < 1e-9);
        assert!((b.median_token_total.unwrap() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn leaderboard_ranks_by_composite_with_geomean_efficiency() {
        // Two runners over two deep fixtures, both passing all. zo is best on
        // tokens and wall for every fixture; codex uses exactly 2x on each. So
        // accuracy ties at 100 and the geomean efficiency decides the ranking.
        let rows = parse_ledger(
            r#"
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":10,"token_total":100}
{"runner":"zo","lane":"deep","fixture":"b","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":15,"token_total":150}
{"runner":"codex","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":20,"token_total":200}
{"runner":"codex","lane":"deep","fixture":"b","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":30,"token_total":300}
"#,
        );
        let board = build_leaderboard(&rows);
        let deep = &board["deep"];
        assert_eq!(deep.len(), 2);

        // zo dominates: ratio 1.0 on every fixture → score 100, composite 100.
        let zo = &deep[0];
        assert_eq!(zo.runner, "zo");
        assert_eq!(zo.fixtures_scored, 2);
        assert!((zo.geomean_token_ratio.unwrap() - 1.0).abs() < 1e-9);
        assert!((zo.token_score - 100.0).abs() < 1e-9);
        assert!((zo.speed_score - 100.0).abs() < 1e-9);
        assert!((zo.pass_rate - 1.0).abs() < 1e-9);
        assert!((zo.composite - 100.0).abs() < 1e-9);

        // codex: 2x tokens & wall on both → geomean 2.0 → score 50; same accuracy.
        let codex = &deep[1];
        assert_eq!(codex.runner, "codex");
        assert!((codex.geomean_token_ratio.unwrap() - 2.0).abs() < 1e-9);
        assert!((codex.token_score - 50.0).abs() < 1e-9);
        assert!((codex.speed_score - 50.0).abs() < 1e-9);
        // 0.50*100 + 0.25*50 + 0.25*50 = 75.
        assert!((codex.composite - 75.0).abs() < 1e-9);
    }

    #[test]
    fn leaderboard_json_keeps_lane_keys_and_adds_axis_views() {
        let rows = parse_ledger(
            r#"
{"runner":"zo","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":20,"token_total":100}
{"runner":"codex","lane":"deep","fixture":"a","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":10,"token_total":200}
"#,
        );
        let board = build_leaderboard_with_views(&rows);
        assert!(board.get("deep").is_some(), "lane key stays top-level");
        assert!(board["_views"]["composite"]["deep"].is_array());
        assert_eq!(
            board["_views"]["latency_first"]["deep"][0]["runner"],
            "codex"
        );
        assert_eq!(board["_views"]["token_first"]["deep"][0]["runner"], "zo");
    }

    #[test]
    fn leaderboard_keeps_lanes_separate_and_scores_single_runner_as_self_best() {
        // One runner alone is its own per-fixture best, so its ratios are 1.0 and
        // the efficiency scores are 100 — the ranking collapses to accuracy. A
        // second lane stays its own board.
        let rows = parse_ledger(
            r#"
{"runner":"zo","lane":"fast","fixture":"x","final_decision":"accepted","leaderboard_eligible":true,"wall_seconds":5,"token_total":50}
{"runner":"zo","lane":"deep","fixture":"y","final_decision":"rejected","leaderboard_eligible":true,"wall_seconds":9,"token_total":90}
"#,
        );
        let board = build_leaderboard(&rows);
        assert!(board.contains_key("fast"));
        assert!(board.contains_key("deep"));

        let fast = &board["fast"][0];
        assert!((fast.token_score - 100.0).abs() < 1e-9);
        assert!((fast.composite - 100.0).abs() < 1e-9); // accuracy 100 + efficiency 100

        // Rejected → pass_rate 0; efficiency still 100 (its own best), so the
        // composite is purely the efficiency weight: 0.25*100 + 0.25*100 = 50.
        let deep = &board["deep"][0];
        assert!((deep.pass_rate).abs() < 1e-9);
        assert!((deep.composite - 50.0).abs() < 1e-9);
    }
}
