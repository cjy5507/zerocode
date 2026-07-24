//! Aggregate-safe Smart Router outcome history.
//!
//! This module owns the small persisted signal that later phases can use to
//! evaluate routing quality. It deliberately stores only route/model/status
//! metadata and bounded counters — never prompts, agent output, or raw errors.

use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead as _, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::policy::{RouteFeedbackHint, MAX_FEEDBACK_ADJUSTMENT};
use crate::jsonl_log::rewrite_jsonl_lines_if_changed;

/// Decisive-sample count at which outcome feedback reaches its full weight. Below
/// this, the adjustment ramps up linearly with the sample count so a thin history
/// (the `>=2` minimum) only nudges, while an well-evidenced route can override the
/// recency prior. Sized so ~8 decisive runs earn full trust.
///
/// `pub` (not `pub(super)`): also the exploration-eligibility threshold
/// (`policy::select_ranked_auto_candidate`'s "incumbent >=8 decisive" gate)
/// and the `/smart doctor` exploration-status section's eligibility check
/// (P7) both need the SAME number a caller outside this module can name,
/// rather than a silently-duplicated magic `8`.
pub const CONFIDENT_DECISIVE_SAMPLES: i32 = 8;
const OUTCOME_DIR: &str = "smart-router";
const OUTCOME_FILE: &str = "route-outcomes.jsonl";

/// Per-`(route_key, selectedModel)` bucket cap (P3 retention v2). Replaces the
/// old flat [`OUTCOME_GLOBAL_RETENTION`]-only cap, which let a single
/// high-traffic route (observed live: `code-reviewer:gpt-5.5-fast` at 195+
/// decisive samples) crowd out a low-traffic route's ENTIRE history before it
/// ever reached the `>=2` decisive minimum. A bucket over this cap drops its
/// oldest records first; append order is otherwise preserved.
const OUTCOME_BUCKET_RETENTION: usize = 48;
/// Global cap across all buckets (P3 retention v2), replacing the old flat
/// 512-record cap. Sized well above `OUTCOME_BUCKET_RETENTION` times a
/// realistic number of concurrently-active routes, so it only engages when
/// the store holds many distinct routes at once. When exceeded, the
/// globally-oldest record is evicted from whichever bucket is CURRENTLY
/// LARGEST, repeated until back under budget — so a handful of hot routes
/// absorb the trim and a thin, low-traffic route's history is the last thing
/// cut.
const OUTCOME_GLOBAL_RETENTION: usize = 2048;
/// Half-life (in days) for [`weighted_feedback_hint_for_route_key`]'s
/// recency weighting: a decisive record's contribution to the confidence-
/// weighted adjustment halves every this many days. Sized so a signal from
/// the project's typical ~1-week outcome-store window still counts heavily,
/// while a month-old-only bucket has decayed toward near-zero influence
/// instead of freezing at the full bound forever.
const FEEDBACK_HALF_LIFE_DAYS: f64 = 14.0;

// NOTE: `PartialEq` only (not `Eq`) — `signal_weight: Option<f32>` (P3 schema
// v2) has no total ordering, so the whole struct cannot derive `Eq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteOutcomeRecord {
    pub recorded_at: u64,
    pub route_key: String,
    pub target_kind: String,
    pub target: String,
    pub selected_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_model: Option<String>,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_error_class: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub output_tokens: u64,
    /// Evidence source: absent for run-level outcomes (the spawn recorder),
    /// `"verdict"` for cross-check attribution — a validator's judgement of a
    /// worker's output folded back onto the worker's model. Provenance only;
    /// aggregation treats both as decisive samples on purpose (run outcomes
    /// say "the model finished", verdict outcomes say "the work was right").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    /// The verification model that judged this record's `selected_model`, for a
    /// `signal:"verdict"` pair-attribution record: `(selected_model =
    /// implementation model, verifier_model = the model that cross-checked it)`.
    /// Populated only on the main-turn verdict leg when a cross-model verifier
    /// actually ran; `None` for run outcomes, native same-model verify, and
    /// every pre-v2 record. Recorded for provenance/`/smart doctor` only — no
    /// scorer learns from the pair key yet. Optional + serde-default so pre-v2
    /// lines still deserialize (see `route_outcome_v2_schema_parses_pre_v2_live_records`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier_model: Option<String>,
    // --- P3 schema v2: all OPTIONAL, serde-default so every pre-v2 record on
    // disk (see the `route_outcome_v2_schema_parses_pre_v2_live_records` test,
    // built from real captured lines) still deserializes with `None`/`0`.
    // Populated only by routes the Smart router actually classified — a
    // record with these absent means routing-off, an explicit model, or a
    // pre-v2 write, never a parse failure.
    /// Smart-router role label (`RouteRole`, e.g. `"coding"`/`"analysis"`) at
    /// the time this route was decided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Task complexity (`RouteTaskComplexity`, lowercased) at decision time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub complexity: Option<String>,
    /// Task risk (`RouteTaskRisk`, lowercased) at decision time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk: Option<String>,
    /// Reasoning-effort tier actually used for the run, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort_level: Option<String>,
    /// Wall-clock run duration in milliseconds, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// How this route was decided: `"auto"` | `"pin"` | `"explicit"` |
    /// `"fallback"` | `"exploration"`. Distinguishes an AUTO-selector pick
    /// (real routing-quality signal) from a config-forced pin/family-selector
    /// override (an AVAILABILITY signal, not a quality one — the live store's
    /// largest bucket was a pin's residue, per the routing plan's live-data
    /// audit) so a later learning phase can weight them differently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_source: Option<String>,
    /// Reserved for a future weighted-signal source (e.g. a verdict weighted
    /// differently from a run completion). Not populated or consumed yet;
    /// the field exists now so the schema shape is fixed before a consumer
    /// lands (mirrors the `TiersProvenance::Learned` reservation pattern).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_weight: Option<f32>,
}

impl RouteOutcomeRecord {
    #[must_use]
    pub fn new(
        target_kind: impl Into<String>,
        target: impl Into<String>,
        selected_model: impl Into<String>,
        status: impl Into<String>,
    ) -> Self {
        let target_kind = target_kind.into();
        let target = target.into();
        Self {
            recorded_at: epoch_seconds_now(),
            route_key: format!("{target_kind}:{target}"),
            target_kind,
            target,
            selected_model: selected_model.into(),
            requested_model: None,
            status: status.into(),
            provider_error_class: None,
            output_tokens: 0,
            signal: None,
            verifier_model: None,
            role: None,
            complexity: None,
            risk: None,
            effort_level: None,
            duration_ms: None,
            route_source: None,
            signal_weight: None,
        }
    }

    #[must_use]
    pub fn with_signal(mut self, signal: impl Into<String>) -> Self {
        self.signal = Some(signal.into());
        self
    }

    /// Attach the verification model that judged this record's `selected_model`
    /// (the `(implementation, verifier)` pair, P1). Blank/whitespace is dropped
    /// to `None` — same empty-filtering convention as [`with_requested_model`].
    #[must_use]
    pub fn with_verifier_model(mut self, verifier_model: Option<String>) -> Self {
        self.verifier_model = verifier_model.filter(|model| !model.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_requested_model(mut self, requested_model: Option<String>) -> Self {
        self.requested_model = requested_model.filter(|model| !model.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_provider_error_class(mut self, provider_error_class: Option<String>) -> Self {
        self.provider_error_class = provider_error_class.filter(|class| !class.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_output_tokens(mut self, output_tokens: u64) -> Self {
        self.output_tokens = output_tokens;
        self
    }

    #[must_use]
    pub fn with_role(mut self, role: Option<String>) -> Self {
        self.role = role.filter(|value| !value.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_complexity(mut self, complexity: Option<String>) -> Self {
        self.complexity = complexity.filter(|value| !value.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_risk(mut self, risk: Option<String>) -> Self {
        self.risk = risk.filter(|value| !value.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_effort_level(mut self, effort_level: Option<String>) -> Self {
        self.effort_level = effort_level.filter(|value| !value.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_duration_ms(mut self, duration_ms: Option<u64>) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    #[must_use]
    pub fn with_route_source(mut self, route_source: Option<String>) -> Self {
        self.route_source = route_source.filter(|value| !value.trim().is_empty());
        self
    }

    #[must_use]
    pub fn with_signal_weight(mut self, signal_weight: Option<f32>) -> Self {
        self.signal_weight = signal_weight;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteOutcomeSummary {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub stopped: usize,
    pub still_running: usize,
    pub output_tokens: u64,
    pub by_route: Vec<RouteOutcomeBucket>,
}

impl RouteOutcomeSummary {
    #[must_use]
    pub fn feedback_hint_for_route_key(&self, route_key: &str) -> RouteFeedbackHint {
        self.by_route
            .iter()
            .filter(|bucket| bucket.route_key == route_key)
            .filter_map(RouteOutcomeBucket::feedback_adjustment)
            .fold(RouteFeedbackHint::disabled(), |hint, (model, adjustment)| {
                hint.with_model_adjustment(model, adjustment)
            })
    }

    /// Per-canonical-model decisive (`completed + failed`) sample counts for
    /// a `route_key` — Phase 5's eligibility input. Exposed here so `apply.rs`
    /// can determine exploration eligibility straight off the already-loaded
    /// summary, with no second JSONL read. `selected_model` is already
    /// canonical whenever this summary was built via
    /// [`summarize_route_outcomes_with_canonicalizer`] (the live call site,
    /// P3) — this fn does no canonicalization itself, it only reads the
    /// bucket key the summary already computed.
    #[must_use]
    pub fn decisive_counts_for_route_key(&self, route_key: &str) -> Vec<(String, usize)> {
        self.by_route
            .iter()
            .filter(|bucket| bucket.route_key == route_key)
            .map(|bucket| (bucket.selected_model.clone(), bucket.completed.saturating_add(bucket.failed)))
            .collect()
    }

    /// Total records (every status, not just decisive) ever seen for a
    /// `route_key` — the Phase 5 exploration cadence input
    /// (`total % smart.explorationCadence == 0`). Sums `RouteOutcomeBucket::total`
    /// across every model bucket sharing the `route_key`, so it reflects the
    /// SAME retained history `decisive_counts_for_route_key` reads (subject
    /// to the same P3 bucket/global caps).
    #[must_use]
    pub fn total_records_for_route_key(&self, route_key: &str) -> usize {
        self.by_route
            .iter()
            .filter(|bucket| bucket.route_key == route_key)
            .map(|bucket| bucket.total)
            .fold(0usize, usize::saturating_add)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteOutcomeBucket {
    pub route_key: String,
    pub target_kind: String,
    pub target: String,
    pub selected_model: String,
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub stopped: usize,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_errors: BTreeMap<String, usize>,
}

impl RouteOutcomeBucket {
    fn feedback_adjustment(&self) -> Option<(String, i16)> {
        // Score over DECISIVE runs only: completed + failed. `stopped` =
        // user-cancelled runs (and abandoned permission prompts) are NOT model
        // outcomes, so they are excluded from both the numerator and the
        // denominator — a cancel neither penalizes the model nor dilutes its
        // score. Need >=2 decisive samples before nudging routing.
        let decisive = self.completed.saturating_add(self.failed);
        if decisive < 2 {
            return None;
        }
        let completed = i32::try_from(self.completed).unwrap_or(i32::MAX);
        let failed = i32::try_from(self.failed).unwrap_or(i32::MAX);
        let decisive = i32::try_from(decisive).unwrap_or(i32::MAX).max(1);
        // Confidence-weighted (P3): the win-rate `(completed - failed)/decisive`
        // is scaled to the full feedback bound only once there are enough decisive
        // samples (`CONFIDENT_DECISIVE_SAMPLES`); below that it ramps up linearly,
        // so two lucky runs nudge gently while a well-evidenced model can reach the
        // full `MAX_FEEDBACK_ADJUSTMENT` — large enough to override the recency
        // (`release_rank`) prior within the role's candidate pool, which the old
        // ±40 bound never could. Still bounded under the capability/tier gates.
        let confidence = decisive.min(CONFIDENT_DECISIVE_SAMPLES);
        let max = i32::from(MAX_FEEDBACK_ADJUSTMENT);
        let score = ((completed - failed) * max * confidence)
            / (decisive * CONFIDENT_DECISIVE_SAMPLES);
        Some((
            self.selected_model.clone(),
            i16::try_from(score.clamp(-max, max)).unwrap_or(0),
        ))
    }
}

#[must_use]
pub fn route_outcome_log_path(cwd: &Path) -> PathBuf {
    crate::zo_project_state_dir(cwd).join(OUTCOME_DIR).join(OUTCOME_FILE)
}

/// Statuses a route outcome may actually persist. A route outcome represents
/// a FINISHED run; `still_running` (and any other in-flight/placeholder
/// label) is a live-progress signal only — see [`record_route_outcome`]'s
/// doctrine guard.
#[must_use]
pub fn is_terminal_outcome_status(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "stopped")
}

/// Recorder-side doctrine guard (shared by every recorder — the spawn
/// completion path, verdict attribution, and any future source): a route
/// outcome record is only ever appended for a TERMINAL status
/// (`completed`/`failed`/`stopped`). `still_running` is a live/in-flight
/// placeholder (HUD, collection-window stragglers) — persisting it would
/// silently poison the decisive aggregate as a false failure the moment it
/// landed (see `normalized_status`/`add_status_to_bucket`).
///
/// `debug_assert!` means a violation PANICS in development/test builds (the
/// bug is caught immediately, loudly, at the call site that introduced it)
/// but the write is unconditionally skipped either way, so a release build
/// degrades to "one record silently dropped" instead of ever crashing on it.
pub fn record_route_outcome(cwd: &Path, record: &RouteOutcomeRecord) -> io::Result<()> {
    debug_assert!(
        is_terminal_outcome_status(&record.status),
        "route-outcome recorder doctrine violation: attempted to persist non-terminal status {:?} for route_key {:?} — every recorder must record only completed/failed/stopped",
        record.status,
        record.route_key,
    );
    if !is_terminal_outcome_status(&record.status) {
        return Ok(());
    }
    record_route_outcome_at_path(&route_outcome_log_path(cwd), record)
}

fn record_route_outcome_at_path(path: &Path, record: &RouteOutcomeRecord) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, record).map_err(io::Error::other)?;
        writeln!(file)?;
    }
    prune_route_outcome_store(path)
}

/// P3 retention v2: per-bucket cap + global cap (see the module-level consts)
/// in place of the old flat line-count cap. Operates on the file's OWN lines
/// via [`rewrite_jsonl_lines_if_changed`] (symlink-safe, atomic rename) so a
/// retained record's on-disk bytes are untouched — only dropped lines change.
fn prune_route_outcome_store(path: &Path) -> io::Result<()> {
    rewrite_jsonl_lines_if_changed(path, |lines| {
        let parsed: Option<Vec<RouteOutcomeRecord>> = lines
            .iter()
            .map(|line| serde_json::from_str(line).ok())
            .collect();
        let Some(records) = parsed else {
            // A line this build cannot parse (corrupt write, or a future/
            // foreign schema) cannot be bucketed safely — fall back to the
            // old flat newest-cap on raw lines so the file still stays
            // bounded instead of growing without limit.
            return flat_cap_newest_lines(lines, OUTCOME_GLOBAL_RETENTION);
        };
        let keep = outcome_prune_keep_mask(&records);
        lines
            .into_iter()
            .zip(keep)
            .filter_map(|(line, keep)| keep.then_some(line))
            .collect()
    })
}

fn flat_cap_newest_lines(lines: Vec<String>, cap: usize) -> Vec<String> {
    if lines.len() <= cap {
        return lines;
    }
    let skip = lines.len() - cap;
    lines.into_iter().skip(skip).collect()
}

/// Which of `records` (append/file order — oldest first) survive retention.
/// Two passes:
/// 1. Per-`(route_key, selectedModel)` bucket cap: a single bucket keeps only
///    its newest [`OUTCOME_BUCKET_RETENTION`] records.
/// 2. Global cap: if bucket-capped survivors still exceed
///    [`OUTCOME_GLOBAL_RETENTION`], evict the globally-oldest surviving
///    record from whichever bucket is CURRENTLY LARGEST, repeated until back
///    under budget — protecting a low-traffic route's thin history until the
///    very end.
fn outcome_prune_keep_mask(records: &[RouteOutcomeRecord]) -> Vec<bool> {
    let mut keep = vec![true; records.len()];
    let mut order: BTreeMap<(String, String), VecDeque<usize>> = BTreeMap::new();
    for (idx, record) in records.iter().enumerate() {
        order
            .entry((record.route_key.clone(), record.selected_model.clone()))
            .or_default()
            .push_back(idx);
    }
    for indices in order.values_mut() {
        while indices.len() > OUTCOME_BUCKET_RETENTION {
            if let Some(oldest) = indices.pop_front() {
                keep[oldest] = false;
            }
        }
    }
    let mut survivors: usize = order.values().map(VecDeque::len).sum();
    if survivors > OUTCOME_GLOBAL_RETENTION {
        let mut heap: BinaryHeap<(usize, (String, String))> = order
            .iter()
            .filter(|(_, indices)| !indices.is_empty())
            .map(|(key, indices)| (indices.len(), key.clone()))
            .collect();
        while survivors > OUTCOME_GLOBAL_RETENTION {
            let Some((size, key)) = heap.pop() else { break };
            let Some(indices) = order.get_mut(&key) else { continue };
            if indices.len() != size {
                // Stale heap entry (this bucket's size already changed since
                // it was pushed) — push the current size back and retry.
                if !indices.is_empty() {
                    heap.push((indices.len(), key));
                }
                continue;
            }
            if let Some(oldest) = indices.pop_front() {
                keep[oldest] = false;
                survivors -= 1;
                if !indices.is_empty() {
                    heap.push((indices.len(), key));
                }
            }
        }
    }
    keep
}

/// Pure record-level pruning (the same policy as [`prune_route_outcome_store`],
/// without the file I/O) — exercised directly by the retention tests below,
/// which is cheaper and more precise than round-tripping through real files.
#[cfg(test)]
#[must_use]
fn prune_outcome_records(records: Vec<RouteOutcomeRecord>) -> Vec<RouteOutcomeRecord> {
    let keep = outcome_prune_keep_mask(&records);
    records
        .into_iter()
        .zip(keep)
        .filter_map(|(record, keep)| keep.then_some(record))
        .collect()
}

pub fn read_route_outcomes(cwd: &Path) -> io::Result<Vec<RouteOutcomeRecord>> {
    read_route_outcomes_from_path(&route_outcome_log_path(cwd))
}

pub(super) fn read_route_outcomes_from_path(path: &Path) -> io::Result<Vec<RouteOutcomeRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = OpenOptions::new().read(true).open(path)?;
    Ok(BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<RouteOutcomeRecord>(&line).ok())
        .collect())
}

pub fn read_route_outcome_summary(cwd: &Path) -> io::Result<RouteOutcomeSummary> {
    Ok(summarize_route_outcomes(&read_route_outcomes(cwd)?))
}

#[must_use]
pub fn summarize_route_outcomes(records: &[RouteOutcomeRecord]) -> RouteOutcomeSummary {
    summarize_route_outcomes_with_canonicalizer(records, ToString::to_string)
}

/// Same aggregation as [`summarize_route_outcomes`], but `canonicalize_model`
/// is applied to each record's `selected_model` BEFORE it becomes (part of)
/// a bucket key — so historical id fragments that name the same model
/// (`claude-opus-4-8` vs `claude-opus-4.8`, `fable`/`fable5` vs
/// `claude-fable-5`, `gpt-5.5` vs its dated canonical id) merge into one
/// bucket instead of diluting the decisive sample count across N near-
/// duplicate buckets.
///
/// This is the pure engine's half of P3 canonicalization — it stays free of
/// any alias-resolution logic itself (`crate::model_router` has no
/// dependency on `api`); the tools layer builds the actual canonicalizer
/// (on `api::resolve_model_alias`) and injects it here. [`summarize_route_outcomes`]
/// passes the identity closure, so every EXISTING caller keeps its current,
/// byte-identical behavior; only a caller that opts into this variant sees
/// merged buckets.
#[must_use]
pub fn summarize_route_outcomes_with_canonicalizer(
    records: &[RouteOutcomeRecord],
    canonicalize_model: impl Fn(&str) -> String,
) -> RouteOutcomeSummary {
    let mut summary = RouteOutcomeSummary::default();
    let mut buckets: BTreeMap<(String, String, String, String), RouteOutcomeBucket> = BTreeMap::new();
    for record in records {
        summary.total = summary.total.saturating_add(1);
        summary.output_tokens = summary.output_tokens.saturating_add(record.output_tokens);
        add_status_to_summary(&mut summary, record.status.as_str());
        let canonical_model = canonicalize_model(&record.selected_model);
        let key = (
            record.route_key.clone(),
            record.target_kind.clone(),
            record.target.clone(),
            canonical_model.clone(),
        );
        let bucket = buckets.entry(key).or_insert_with(|| RouteOutcomeBucket {
            route_key: record.route_key.clone(),
            target_kind: record.target_kind.clone(),
            target: record.target.clone(),
            selected_model: canonical_model,
            total: 0,
            completed: 0,
            failed: 0,
            stopped: 0,
            output_tokens: 0,
            provider_errors: BTreeMap::new(),
        });
        bucket.total = bucket.total.saturating_add(1);
        bucket.output_tokens = bucket.output_tokens.saturating_add(record.output_tokens);
        add_status_to_bucket(
            bucket,
            record.status.as_str(),
            record.provider_error_class.as_deref(),
        );
        if let Some(class) = record.provider_error_class.as_deref() {
            *bucket.provider_errors.entry(class.to_string()).or_insert(0) += 1;
        }
    }
    summary.by_route = buckets.into_values().collect();
    summary.by_route.sort_by(|left, right| {
        right
            .total
            .cmp(&left.total)
            .then_with(|| left.route_key.cmp(&right.route_key))
    });
    summary
}

fn add_status_to_summary(summary: &mut RouteOutcomeSummary, status: &str) {
    match normalized_status(status) {
        OutcomeStatus::Completed => summary.completed = summary.completed.saturating_add(1),
        OutcomeStatus::Failed => summary.failed = summary.failed.saturating_add(1),
        OutcomeStatus::Stopped => summary.stopped = summary.stopped.saturating_add(1),
        OutcomeStatus::StillRunning => {
            summary.still_running = summary.still_running.saturating_add(1);
        }
    }
}

fn add_status_to_bucket(
    bucket: &mut RouteOutcomeBucket,
    status: &str,
    provider_error_class: Option<&str>,
) {
    match normalized_status(status) {
        OutcomeStatus::Completed => bucket.completed = bucket.completed.saturating_add(1),
        OutcomeStatus::Stopped => bucket.stopped = bucket.stopped.saturating_add(1),
        OutcomeStatus::Failed | OutcomeStatus::StillRunning => {
            // Provider-infrastructure failures (throttling, transient faults,
            // expired credentials) are not model-quality outcomes: a throttled
            // provider's model must not lose win-rate for non-quality reasons.
            // They stay in `total` and `provider_errors` (dashboard) but are
            // excluded from the decisive denominator, like `stopped`. Model
            // faults (contextOverflow / invalid tool protocol / safetyBlocked /
            // nonRetryable) still count as failures.
            if is_infra_provider_error(provider_error_class) {
                return;
            }
            bucket.failed = bucket.failed.saturating_add(1);
        }
    }
}

/// Shared with [`decisive_outcome`] so there is exactly one place that
/// decides which provider-error classes are infrastructure noise (never a
/// model-quality signal), instead of two independently-maintained copies of
/// the same literal set.
fn is_infra_provider_error(provider_error_class: Option<&str>) -> bool {
    matches!(provider_error_class, Some("rateLimit" | "transient" | "authExpired"))
}

/// Whether a record is a decisive win (`Some(true)`), a decisive loss
/// (`Some(false)`), or excluded from the decisive pool entirely (`None`) —
/// `stopped` (user-cancelled) and an infra-class failure never count either
/// way. Reuses [`normalized_status`]/[`is_infra_provider_error`] so this is
/// the SAME classification [`add_status_to_bucket`] applies, just projected
/// to a signed outcome instead of an aggregate mutation — used by
/// [`weighted_feedback_hint_for_route_key`], which needs a per-record verdict
/// (not a pre-aggregated bucket count) to apply recency weighting.
pub(super) fn decisive_outcome(status: &str, provider_error_class: Option<&str>) -> Option<bool> {
    match normalized_status(status) {
        OutcomeStatus::Completed => Some(true),
        OutcomeStatus::Stopped => None,
        OutcomeStatus::Failed | OutcomeStatus::StillRunning => {
            if is_infra_provider_error(provider_error_class) {
                None
            } else {
                Some(false)
            }
        }
    }
}

/// Recency-half-life-weighted analogue of
/// [`RouteOutcomeSummary::feedback_hint_for_route_key`]: each decisive
/// (`completed`/`failed`) record's contribution to the confidence-weighted
/// adjustment is scaled by `0.5^(age_days / FEEDBACK_HALF_LIFE_DAYS)` before
/// the win-rate/confidence-ramp math runs, so a model with only aging
/// evidence decays back toward a neutral (0) adjustment instead of staying
/// pinned at the full bound forever (the plain-count aggregate in
/// [`RouteOutcomeBucket::feedback_adjustment`] has no notion of "when" — a
/// bucket frozen at its `CONFIDENT_DECISIVE_SAMPLES` ramp stays at the same
/// score whether its evidence is an hour or a year old).
///
/// `now` is INJECTED (epoch seconds) — no hidden clock read — so this stays a
/// pure, deterministically-testable function; callers pass live epoch
/// seconds, tests pass fixed values. `canonicalize_model` mirrors
/// [`summarize_route_outcomes_with_canonicalizer`]'s seam so weighted buckets
/// merge the same historical id fragments the unweighted summary does.
///
/// NOT wired into the live routing path for this phase: `apply.rs`'s
/// `SmartRouteContext::feedback_for` still calls the unweighted
/// [`RouteOutcomeSummary::feedback_hint_for_route_key`], which is the ONLY
/// routing-affecting feedback source today — P3 is schema/substrate-only
/// (routing behavior stays byte-identical); a later phase can switch the
/// live call site to this function once that behavior change is explicitly
/// signed off.
///
/// **Still unwired as of Phase 6** (`model_router::learned`): Phase 6 reuses
/// this fn's underlying MACHINERY — [`recency_weight`] and [`decisive_outcome`]
/// are shared, `pub(super)`, with the learned-specialty aggregator — but does
/// NOT call `weighted_feedback_hint_for_route_key` itself, and does not flip
/// `apply.rs`'s plain per-route-key feedback path onto it either. That
/// remains a deliberately separate, deferred behavior-freeze decision until
/// the sibling path is explicitly approved after soak.
#[must_use]
pub fn weighted_feedback_hint_for_route_key(
    records: &[RouteOutcomeRecord],
    route_key: &str,
    now: u64,
    canonicalize_model: impl Fn(&str) -> String,
) -> RouteFeedbackHint {
    let mut per_model: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for record in records {
        if record.route_key != route_key {
            continue;
        }
        let Some(win) = decisive_outcome(record.status.as_str(), record.provider_error_class.as_deref()) else {
            continue;
        };
        let weight = recency_weight(record.recorded_at, now);
        let entry = per_model
            .entry(canonicalize_model(&record.selected_model))
            .or_insert((0.0, 0.0));
        if win {
            entry.0 += weight;
        } else {
            entry.1 += weight;
        }
    }
    per_model
        .into_iter()
        .fold(RouteFeedbackHint::disabled(), |hint, (model, (weighted_completed, weighted_failed))| {
            match weighted_feedback_adjustment(weighted_completed, weighted_failed) {
                Some(adjustment) => hint.with_model_adjustment(model, adjustment),
                None => hint,
            }
        })
}

/// `0.5^(age_days / FEEDBACK_HALF_LIFE_DAYS)`, clamped implicitly to `(0, 1]`
/// by construction (`age_seconds` saturates at 0 for a `recorded_at` at or
/// after `now`, e.g. a fixed test `now` or minor clock skew — that reads as
/// "fresh", weight 1.0, never > 1.0 or negative).
#[allow(clippy::cast_precision_loss)]
pub(super) fn recency_weight(recorded_at: u64, now: u64) -> f64 {
    let age_seconds = now.saturating_sub(recorded_at);
    let age_days = age_seconds as f64 / 86_400.0;
    0.5_f64.powf(age_days / FEEDBACK_HALF_LIFE_DAYS)
}

/// Same confidence-ramp shape as [`RouteOutcomeBucket::feedback_adjustment`]
/// (see its doc for the rationale), generalized from integer decisive counts
/// to recency-weighted `f64` sums. At weight 1.0 per record (i.e. every
/// record dated exactly `now`) this produces the IDENTICAL score the integer
/// version would for the same raw completed/failed counts — the ramp and
/// bound math are the same formula, just over weighted sums.
fn weighted_feedback_adjustment(weighted_completed: f64, weighted_failed: f64) -> Option<i16> {
    let decisive = weighted_completed + weighted_failed;
    if decisive < 2.0 {
        return None;
    }
    let confidence = decisive.min(f64::from(CONFIDENT_DECISIVE_SAMPLES));
    let max = f64::from(MAX_FEEDBACK_ADJUSTMENT);
    let score = (weighted_completed - weighted_failed) * max * confidence / (decisive * f64::from(CONFIDENT_DECISIVE_SAMPLES));
    #[allow(clippy::cast_possible_truncation)]
    let bounded = score.round().clamp(-max, max) as i16;
    Some(bounded)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutcomeStatus {
    Completed,
    Failed,
    Stopped,
    StillRunning,
}

fn normalized_status(status: &str) -> OutcomeStatus {
    match status {
        "completed" => OutcomeStatus::Completed,
        "stopped" => OutcomeStatus::Stopped,
        "still_running" => OutcomeStatus::StillRunning,
        _ => OutcomeStatus::Failed,
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

fn epoch_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn summary_counts_statuses_without_raw_outputs() {
        let records = vec![
            RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
                .with_output_tokens(7),
            RouteOutcomeRecord::new("subagent", "Verification", "model-a", "failed")
                .with_provider_error_class(Some("rateLimit".to_string())),
            RouteOutcomeRecord::new("subagent", "debugger", "model-b", "stopped"),
        ];

        let summary = summarize_route_outcomes(&records);

        assert_eq!(summary.total, 3);
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.stopped, 1);
        assert_eq!(summary.output_tokens, 7);
        assert_eq!(summary.by_route[0].route_key, "subagent:Verification");
        assert_eq!(summary.by_route[0].provider_errors["rateLimit"], 1);
    }

    #[test]
    fn decisive_counts_and_total_records_are_scoped_to_the_route_key() {
        let records = vec![
            RouteOutcomeRecord::new("subagent", "code-reviewer", "model-hot", "completed"),
            RouteOutcomeRecord::new("subagent", "code-reviewer", "model-hot", "completed"),
            RouteOutcomeRecord::new("subagent", "code-reviewer", "model-hot", "failed"),
            RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.6-sol", "completed"),
            // A cancelled run: counts toward `total` but not decisive.
            RouteOutcomeRecord::new("subagent", "code-reviewer", "gpt-5.6-sol", "stopped"),
            // A different route_key entirely must not leak in.
            RouteOutcomeRecord::new("subagent", "debugger", "model-hot", "completed"),
        ];

        let summary = summarize_route_outcomes(&records);
        let route_key = "subagent:code-reviewer";

        let mut counts = summary.decisive_counts_for_route_key(route_key);
        counts.sort();
        assert_eq!(
            counts,
            vec![("gpt-5.6-sol".to_string(), 1), ("model-hot".to_string(), 3)]
        );
        assert_eq!(summary.total_records_for_route_key(route_key), 5);
        assert_eq!(summary.total_records_for_route_key("subagent:debugger"), 1);
        assert_eq!(summary.decisive_counts_for_route_key("subagent:unknown"), Vec::new());
    }

    #[test]
    fn provider_infra_failures_do_not_count_against_model_quality() {
        // 2 wins + 3 rate-limit failures: infra faults must not turn a good
        // model's feedback negative. They stay visible (total/provider_errors)
        // but leave the decisive pool, so the bucket scores as 2-0.
        let mut records = vec![
            RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed"),
            RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed"),
        ];
        for _ in 0..3 {
            records.push(
                RouteOutcomeRecord::new("subagent", "Verification", "model-a", "failed")
                    .with_provider_error_class(Some("rateLimit".to_string())),
            );
        }
        // A model fault (no provider_error_class) still counts as a failure.
        records.push(RouteOutcomeRecord::new(
            "subagent",
            "Verification",
            "model-a",
            "failed",
        ));

        let summary = summarize_route_outcomes(&records);
        let bucket = &summary.by_route[0];

        assert_eq!(bucket.total, 6);
        assert_eq!(bucket.completed, 2);
        assert_eq!(bucket.failed, 1, "only the model fault is decisive");
        assert_eq!(bucket.provider_errors["rateLimit"], 3);
        let hint = summary.feedback_hint_for_route_key("subagent:Verification");
        assert!(
            hint.bounded_adjustment_for("model-a") > 0,
            "2-1 decisive record must stay positive despite 3 throttle faults"
        );
    }

    #[test]
    fn store_round_trips_and_prunes_a_hot_bucket_to_its_cap() {
        // All records share one (route_key, model) bucket, so this exercises
        // the real file-level write→prune→read path against the per-bucket
        // cap (the old test asserted the flat 512 cap; P3 replaces that with
        // a per-bucket cap of `OUTCOME_BUCKET_RETENTION`).
        let root = tempfile::tempdir().expect("tempdir");
        let path = root.path().join("route-outcomes.jsonl");
        let extra = 3;
        for index in 0..(OUTCOME_BUCKET_RETENTION + extra) {
            let record = RouteOutcomeRecord::new("subagent", "agent", "model-a", "completed")
                .with_output_tokens(index as u64);
            record_route_outcome_at_path(&path, &record).expect("record outcome");
        }

        let records = read_route_outcomes_from_path(&path).expect("read outcomes");

        assert_eq!(records.len(), OUTCOME_BUCKET_RETENTION);
        // The oldest `extra` records (output_tokens 0..extra) were evicted;
        // the newest `OUTCOME_BUCKET_RETENTION` survive in append order.
        assert_eq!(records[0].output_tokens, extra as u64);
        assert!(!serde_json::to_string(&records[0]).expect("json").contains("prompt"));
    }

    #[test]
    fn bucket_cap_enforced_keeps_newest_records() {
        let mut records = Vec::new();
        for index in 0..60 {
            records.push(
                RouteOutcomeRecord::new("subagent", "agent", "model-a", "completed")
                    .with_output_tokens(index),
            );
        }

        let pruned = prune_outcome_records(records);

        assert_eq!(pruned.len(), OUTCOME_BUCKET_RETENTION);
        // The 12 oldest (0..12) were dropped; 12..60 survive, oldest-first.
        assert_eq!(pruned[0].output_tokens, 12);
        assert_eq!(pruned.last().unwrap().output_tokens, 59);
    }

    #[test]
    fn hot_bucket_burst_cannot_evict_a_quiet_routes_thin_history() {
        // A quiet route with only 3 records must survive a 200-record burst on
        // a totally different (route_key, model) bucket — the live-data
        // problem this retention redesign exists to fix (a hot
        // code-reviewer×gpt-5.5-fast bucket was crowding out thin routes).
        let mut records = Vec::new();
        for index in 0..3 {
            records.push(RouteOutcomeRecord::new(
                "subagent",
                "deep-research",
                "model-quiet",
                "completed",
            ).with_output_tokens(index));
        }
        for index in 0..200 {
            records.push(RouteOutcomeRecord::new(
                "subagent",
                "code-reviewer",
                "model-hot",
                "completed",
            ).with_output_tokens(1000 + index));
        }

        let pruned = prune_outcome_records(records);

        let quiet: Vec<_> = pruned
            .iter()
            .filter(|record| record.selected_model == "model-quiet")
            .collect();
        let hot: Vec<_> = pruned
            .iter()
            .filter(|record| record.selected_model == "model-hot")
            .collect();
        assert_eq!(quiet.len(), 3, "the quiet route's entire history must survive");
        assert_eq!(hot.len(), OUTCOME_BUCKET_RETENTION);
    }

    #[test]
    fn global_cap_evicts_from_the_largest_surviving_bucket_first() {
        // One hot bucket already at the per-bucket cap (48) plus enough
        // singleton (1-record) buckets to push the total one over the global
        // cap. The single excess record must come from the hot bucket, NOT
        // from any of the singleton buckets — that is the whole point of
        // "protect low-traffic route history".
        let mut records = Vec::new();
        for index in 0..OUTCOME_BUCKET_RETENTION {
            records.push(RouteOutcomeRecord::new(
                "subagent",
                "code-reviewer",
                "model-hot",
                "completed",
            ).with_output_tokens(index as u64));
        }
        let singleton_count = OUTCOME_GLOBAL_RETENTION - OUTCOME_BUCKET_RETENTION + 1;
        for index in 0..singleton_count {
            records.push(RouteOutcomeRecord::new(
                "subagent",
                format!("agent-{index}"),
                "model-a",
                "completed",
            ));
        }
        assert_eq!(records.len(), OUTCOME_GLOBAL_RETENTION + 1);

        let pruned = prune_outcome_records(records);

        assert_eq!(pruned.len(), OUTCOME_GLOBAL_RETENTION);
        let hot_survivors = pruned.iter().filter(|r| r.selected_model == "model-hot").count();
        let singleton_survivors = pruned.iter().filter(|r| r.selected_model == "model-a").count();
        assert_eq!(
            hot_survivors,
            OUTCOME_BUCKET_RETENTION - 1,
            "the excess record must come from the largest (hot) bucket"
        );
        assert_eq!(
            singleton_survivors, singleton_count,
            "every low-traffic singleton bucket must survive untouched"
        );
    }

    #[test]
    fn canonicalization_merges_historical_model_id_fragments_at_summarize_time() {
        // Live data showed `claude-opus-4-8` and `claude-opus-4.8` (and
        // similar fragments) diluting the decisive count across two buckets
        // that name the same model. `summarize_route_outcomes` (the identity
        // canonicalizer) keeps them split; the injected canonicalizer merges
        // them into one bucket.
        let records = vec![
            RouteOutcomeRecord::new("subagent", "Plan", "claude-opus-4-8", "completed"),
            RouteOutcomeRecord::new("subagent", "Plan", "claude-opus-4.8", "completed"),
            RouteOutcomeRecord::new("subagent", "Plan", "claude-opus-4.8", "failed"),
        ];

        let unmerged = summarize_route_outcomes(&records);
        assert_eq!(unmerged.by_route.len(), 2, "identity canonicalizer keeps fragments split");

        let merged = summarize_route_outcomes_with_canonicalizer(&records, |model| {
            if model == "claude-opus-4.8" {
                "claude-opus-4-8".to_string()
            } else {
                model.to_string()
            }
        });
        assert_eq!(merged.by_route.len(), 1, "canonicalizer merges the dot/dash fragments");
        let bucket = &merged.by_route[0];
        assert_eq!(bucket.selected_model, "claude-opus-4-8");
        assert_eq!(bucket.total, 3);
        assert_eq!(bucket.completed, 2);
        assert_eq!(bucket.failed, 1);
    }

    #[test]
    fn weighted_feedback_fresh_bucket_still_reaches_full_bound() {
        // 8 decisive, all recorded exactly `now` (weight 1.0 each) must
        // reproduce the SAME full-confidence ±120 the unweighted ramp gives
        // for 8 fresh decisive samples — the half-life axis must not weaken
        // fresh evidence.
        let now = 1_800_000_000_u64;
        let mut records = Vec::new();
        for _ in 0..8 {
            records.push(RouteOutcomeRecord {
                recorded_at: now,
                ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
            });
        }

        let hint = weighted_feedback_hint_for_route_key(
            &records,
            "subagent:Verification",
            now,
            ToString::to_string,
        );

        assert_eq!(hint.bounded_adjustment_for("model-a"), MAX_FEEDBACK_ADJUSTMENT);
    }

    #[test]
    fn weighted_feedback_decays_substantially_for_stale_only_evidence() {
        let now = 1_800_000_000_u64;
        let thirty_days_ago = now - 30 * 86_400;
        let mut fresh = Vec::new();
        let mut stale = Vec::new();
        for _ in 0..20 {
            fresh.push(RouteOutcomeRecord {
                recorded_at: now,
                ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
            });
            stale.push(RouteOutcomeRecord {
                recorded_at: thirty_days_ago,
                ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
            });
        }

        let fresh_hint = weighted_feedback_hint_for_route_key(
            &fresh,
            "subagent:Verification",
            now,
            ToString::to_string,
        );
        let stale_hint = weighted_feedback_hint_for_route_key(
            &stale,
            "subagent:Verification",
            now,
            ToString::to_string,
        );

        assert_eq!(fresh_hint.bounded_adjustment_for("model-a"), MAX_FEEDBACK_ADJUSTMENT);
        let stale_adjustment = stale_hint.bounded_adjustment_for("model-a");
        // 20 decisive samples at 30 days old (weight 0.5^(30/14) ≈ 0.226 each)
        // works out to ≈68, well short of the fresh case's 120 — "substantial"
        // decay without pinning an over-precise hand-computed value.
        assert!(
            stale_adjustment > 0 && stale_adjustment < 100,
            "30-day-old-only evidence must decay substantially below the fresh full bound, got {stale_adjustment}"
        );
    }

    #[test]
    fn weighted_feedback_extremely_stale_thin_evidence_decays_to_disabled() {
        // Only 2 raw decisive samples, both ~90 days old: the weighted
        // decisive sum drops under the `>=2` floor, so the model gets NO
        // adjustment (equivalent to fully decayed to 0) rather than freezing.
        let now = 1_800_000_000_u64;
        let ninety_days_ago = now - 90 * 86_400;
        let records = vec![
            RouteOutcomeRecord {
                recorded_at: ninety_days_ago,
                ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
            },
            RouteOutcomeRecord {
                recorded_at: ninety_days_ago,
                ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
            },
        ];

        let hint = weighted_feedback_hint_for_route_key(
            &records,
            "subagent:Verification",
            now,
            ToString::to_string,
        );

        assert_eq!(hint.bounded_adjustment_for("model-a"), 0);
    }

    #[test]
    fn weighted_feedback_mixed_ages_weight_each_record_independently() {
        // One very fresh failure should outweigh several old wins once they
        // have decayed enough — this is the "mixed ages weight correctly"
        // acceptance case, asserted against the exact hand-computed score.
        let now = 1_800_000_000_u64;
        let sixty_days_ago = now - 60 * 86_400;
        let mut records = Vec::new();
        for _ in 0..6 {
            records.push(RouteOutcomeRecord {
                recorded_at: sixty_days_ago,
                ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "completed")
            });
        }
        records.push(RouteOutcomeRecord {
            recorded_at: now,
            ..RouteOutcomeRecord::new("subagent", "Verification", "model-a", "failed")
        });

        let hint = weighted_feedback_hint_for_route_key(
            &records,
            "subagent:Verification",
            now,
            ToString::to_string,
        );

        // weight(60d) = 0.5^(60/14) ≈ 0.0513; weighted_completed = 6 * that
        // ≈ 0.308; weighted_failed = 1.0 (fresh, weight 1.0). decisive ≈ 1.31
        // stays under the `>=2` confidence floor, so the model gets NO
        // adjustment here — vs. the OLD unweighted formula, which would score
        // this bucket +75 (6 completed / 1 failed, confidence-ramped). That
        // gap is "mixed ages weight correctly": the aged wins do not drown out
        // the one fresh loss the way raw counts alone would.
        assert_eq!(hint.bounded_adjustment_for("model-a"), 0);
    }

    #[test]
    fn recency_weight_is_one_at_zero_age_and_decays_monotonically() {
        let now = 1_800_000_000_u64;
        assert!((recency_weight(now, now) - 1.0).abs() < f64::EPSILON);
        // 14 days == `FEEDBACK_HALF_LIFE_DAYS`, spelled as a literal (rather
        // than cast from the `f64` const) to avoid a lossy-cast lint on a
        // whole-number-valued constant.
        debug_assert!((FEEDBACK_HALF_LIFE_DAYS - 14.0).abs() < f64::EPSILON);
        let at_half_life = recency_weight(now - 14 * 86_400, now);
        assert!(
            (at_half_life - 0.5).abs() < 0.001,
            "one half-life must decay to ~0.5, got {at_half_life}"
        );
        assert!(recency_weight(now - 86_400, now) > recency_weight(now - 2 * 86_400, now));
    }

    #[test]
    #[should_panic(expected = "route-outcome recorder doctrine violation")]
    // The doctrine guard is a `debug_assert!`, which compiles out under
    // `--release`; without this gate a release-mode test run fails on "did
    // not panic as expected" even though the guard works as designed.
    #[cfg(debug_assertions)]
    fn record_route_outcome_debug_asserts_on_non_terminal_status() {
        let root = tempfile::tempdir().expect("tempdir");
        let record = RouteOutcomeRecord::new("subagent", "agent", "model-a", "still_running");
        // Must panic in a debug/test build (the doctrine guard) rather than
        // silently writing a `still_running` placeholder to disk.
        let _ = record_route_outcome(root.path(), &record);
    }

    #[test]
    fn is_terminal_outcome_status_accepts_only_finished_states() {
        assert!(is_terminal_outcome_status("completed"));
        assert!(is_terminal_outcome_status("failed"));
        assert!(is_terminal_outcome_status("stopped"));
        assert!(!is_terminal_outcome_status("still_running"));
        assert!(!is_terminal_outcome_status("running"));
    }

    #[test]
    fn v2_fields_round_trip_through_json() {
        let record = RouteOutcomeRecord::new("subagent", "Plan", "gpt-5.6-sol", "completed")
            .with_role(Some("analysis".to_string()))
            .with_complexity(Some("large".to_string()))
            .with_risk(Some("low".to_string()))
            .with_effort_level(Some("ultra".to_string()))
            .with_duration_ms(Some(12_345))
            .with_route_source(Some("auto".to_string()))
            .with_signal_weight(Some(1.5));

        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"role\":\"analysis\""));
        assert!(json.contains("\"complexity\":\"large\""));
        assert!(json.contains("\"risk\":\"low\""));
        assert!(json.contains("\"effortLevel\":\"ultra\""));
        assert!(json.contains("\"durationMs\":12345"));
        assert!(json.contains("\"routeSource\":\"auto\""));
        assert!(json.contains("\"signalWeight\":1.5"));

        let round_tripped: RouteOutcomeRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round_tripped, record);
    }

    /// P1 pair-attribution: a verdict record carrying `verifier_model`
    /// serializes as `verifierModel` and round-trips; a record WITHOUT one
    /// omits the key entirely (so pre-v2 readers and run outcomes stay
    /// byte-identical), and a line lacking the key deserializes to `None`.
    #[test]
    fn verifier_model_round_trips_and_is_omitted_when_absent() {
        let paired = RouteOutcomeRecord::new("main", "turn", "claude-opus-4-8", "completed")
            .with_signal("verdict")
            .with_signal_weight(Some(1.0))
            .with_verifier_model(Some("gpt-5.6-sol".to_string()));
        let json = serde_json::to_string(&paired).expect("serialize");
        assert!(json.contains("\"verifierModel\":\"gpt-5.6-sol\""), "{json}");
        let round_tripped: RouteOutcomeRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round_tripped, paired);
        assert_eq!(round_tripped.verifier_model.as_deref(), Some("gpt-5.6-sol"));

        // No verifier: the key must not appear at all (Option::is_none skip).
        let unpaired = RouteOutcomeRecord::new("subagent", "Explore", "gpt-5.6-sol", "completed");
        let json = serde_json::to_string(&unpaired).expect("serialize");
        assert!(!json.contains("verifierModel"), "absent verifier must omit the key: {json}");

        // A pre-v2/no-verifier line deserializes with `None`, and blank input
        // is filtered to `None` by the builder.
        let parsed: RouteOutcomeRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.verifier_model, None);
        let blanked = RouteOutcomeRecord::new("main", "turn", "m", "completed")
            .with_verifier_model(Some("   ".to_string()));
        assert_eq!(blanked.verifier_model, None);
    }

    /// Backward-compat: 10 REAL lines captured from a live
    /// `route-outcomes.jsonl` (pre-v2 schema — no `role`/`complexity`/`risk`/
    /// `effortLevel`/`durationMs`/`routeSource`/`signalWeight` fields at all).
    /// Every v2 field must deserialize to `None` and every line must parse —
    /// the store must never need a migration to accept the schema change.
    /// This is a fixed string fixture (NOT a read of the live file at test
    /// time), so it stays reproducible independent of the developer's
    /// machine/session state.
    const LIVE_PRE_V2_LINES: &str = r#"{"recordedAt":1783010203,"routeKey":"subagent:Explore","targetKind":"subagent","target":"Explore","selectedModel":"gemini-3-pro","requestedModel":"gemini-3-pro","status":"failed","providerErrorClass":"invalidToolSchema"}
{"recordedAt":1783011453,"routeKey":"subagent:Explore","targetKind":"subagent","target":"Explore","selectedModel":"gpt-5.5","requestedModel":"gpt-5.5","status":"completed","outputTokens":4306}
{"recordedAt":1783044777,"routeKey":"subagent:code-reviewer","targetKind":"subagent","target":"code-reviewer","selectedModel":"gpt-5.3-codex-spark","requestedModel":"gpt-5.3-codex-spark","status":"completed","outputTokens":570}
{"recordedAt":1783173713,"routeKey":"subagent:code-reviewer","targetKind":"subagent","target":"code-reviewer","selectedModel":"gpt-5.5-fast","requestedModel":"gpt-5.5-fast","status":"completed","outputTokens":4580}
{"recordedAt":1783305219,"routeKey":"subagent:code-reviewer","targetKind":"subagent","target":"code-reviewer","selectedModel":"gpt-5.5-fast","requestedModel":"gpt-5.5-fast","status":"completed","outputTokens":12177}
{"recordedAt":1783435871,"routeKey":"subagent:debugger","targetKind":"subagent","target":"debugger","selectedModel":"gpt-5.5-fast","requestedModel":"gpt-5.5-fast","status":"completed","outputTokens":2985}
{"recordedAt":1783500533,"routeKey":"subagent:Explore","targetKind":"subagent","target":"Explore","selectedModel":"gpt-5.5-fast","requestedModel":"gpt-5.5-fast","status":"completed","outputTokens":9431}
{"recordedAt":1783606855,"routeKey":"subagent:Plan","targetKind":"subagent","target":"Plan","selectedModel":"claude-opus-4-8","requestedModel":"gpt-5.5-fast","status":"failed","providerErrorClass":"nonRetryable"}
{"recordedAt":1783613059,"routeKey":"subagent:code-reviewer","targetKind":"subagent","target":"code-reviewer","selectedModel":"gpt-5.5-fast","requestedModel":"gpt-5.5-fast","status":"completed","outputTokens":13112}
{"recordedAt":1783621615,"routeKey":"subagent:code-reviewer","targetKind":"subagent","target":"code-reviewer","selectedModel":"gpt-5.5-fast","requestedModel":"gpt-5.5-fast","status":"completed","outputTokens":8291}"#;

    #[test]
    fn route_outcome_v2_schema_parses_pre_v2_live_records() {
        let records: Vec<RouteOutcomeRecord> = LIVE_PRE_V2_LINES
            .lines()
            .map(|line| serde_json::from_str(line).expect("pre-v2 live line must still parse"))
            .collect();

        assert_eq!(records.len(), 10);
        for record in &records {
            assert_eq!(record.role, None);
            assert_eq!(record.complexity, None);
            assert_eq!(record.risk, None);
            assert_eq!(record.effort_level, None);
            assert_eq!(record.duration_ms, None);
            assert_eq!(record.route_source, None);
            assert_eq!(record.signal_weight, None);
            assert_eq!(record.verifier_model, None);
        }
        // Sanity: the fixture still carries the real field values through the
        // v2 struct shape (the whole point of `#[serde(default)]`, not just
        // "does it fail to error out").
        assert_eq!(records[0].selected_model, "gemini-3-pro");
        assert_eq!(records[0].provider_error_class.as_deref(), Some("invalidToolSchema"));
        assert_eq!(records[7].requested_model.as_deref(), Some("gpt-5.5-fast"));
        assert_eq!(records[7].selected_model, "claude-opus-4-8");

        // And the whole fixture summarizes without error through the SAME
        // (unchanged) public entry point the store's readers use.
        let summary = summarize_route_outcomes(&records);
        assert_eq!(summary.total, 10);
    }
}
