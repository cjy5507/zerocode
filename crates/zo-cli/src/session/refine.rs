//! `/refine` — the evidence-backed tuning loop, sealed from parts that
//! already exist: the harness attest ledger (what fired / failed / was
//! gated), the Phase 6 learned-specialty shadow stamps (what routing WOULD
//! have picked), and the Dreamer candidate store (what `/improve` can turn
//! into a quarantined patch).
//!
//! Doctrine, in order of importance:
//! - ZERO model calls. Everything is a local read over evidence the sessions
//!   already produced; invoking `/refine` costs no tokens.
//! - The live attest ledger is process-scoped BY DESIGN (`/smart doctor`'s
//!   liveness doctrine: a persisted count must never report an earlier
//!   build's behavior as this one's). Persistence here honors that by
//!   stamping every row with the build SHA and aggregating the current build
//!   SEPARATELY from older ones — older builds are context, never health.
//! - `/refine` APPLIES nothing. Every finding names the smallest change that
//!   would act on it as the exact existing command, so the human stays the
//!   apply gate — the same review-first contract `/improve` ships with. The
//!   one thing it records is a Dreamer candidate for a FAILING feature,
//!   which is itself only an input to that same gated pipeline.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use decision_core::dreamer::{CandidateEvidence, CandidateKind, SelfImproveCandidate};
use serde::{Deserialize, Serialize};

/// Evidence rows live under the canonical writable Zo home, next to the
/// other durable evidence stores.
const ATTEST_EVIDENCE_FILE: &str = "attest.jsonl";
const ATTEST_EVIDENCE_DIR: &str = "evidence";
/// Compact the append-only row file once it outgrows this, keeping only
/// [`JANITOR_KEEP_DAYS`] of rows. One row per session lands well under 2 KiB,
/// so this bound is years of normal use, not months.
const JANITOR_MAX_BYTES: u64 = 1024 * 1024;
const JANITOR_KEEP_DAYS: u64 = 30;
/// Default `/refine` evidence window.
const DEFAULT_WINDOW_DAYS: u64 = 14;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn attest_evidence_path() -> PathBuf {
    runtime::default_config_home()
        .join(ATTEST_EVIDENCE_DIR)
        .join(ATTEST_EVIDENCE_FILE)
}

// ── Persisted schema ─────────────────────────────────────────────────────────

/// One feature's counters as persisted. Mirrors
/// `telemetry::FeatureAttestation` with owned reason keys — the in-memory
/// ledger uses `&'static str` reasons by construction, which cannot cross a
/// serialization boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedAttestation {
    #[serde(default)]
    pub(crate) fired: u64,
    #[serde(default)]
    pub(crate) leaked: u64,
    #[serde(default)]
    pub(crate) failed: BTreeMap<String, u64>,
    #[serde(default)]
    pub(crate) declined: BTreeMap<String, u64>,
}

/// The health classifier over persisted counters, mirroring
/// `telemetry::FeatureHealth` over the domain `/refine` can actually reach
/// (a cross-check test below feeds the same counters to both). The two
/// ablation states are deliberately absent: `aggregate_window` excludes any
/// row recorded under an active ablation BEFORE health is consulted, so a
/// leak or a holdout can never arrive here — a real failure outranks
/// everything else, and a zero with only deliberate declines is a gate, not
/// a defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistedHealth {
    Alive,
    Failing,
    GatedOff,
    Silent,
}

impl PersistedAttestation {
    fn merge_from(&mut self, other: &Self) {
        self.fired += other.fired;
        self.leaked += other.leaked;
        for (reason, count) in &other.failed {
            *self.failed.entry(reason.clone()).or_default() += count;
        }
        for (reason, count) in &other.declined {
            *self.declined.entry(reason.clone()).or_default() += count;
        }
    }

    fn failed_total(&self) -> u64 {
        self.failed.values().sum()
    }

    fn declined_total(&self) -> u64 {
        self.declined.values().sum()
    }

    /// Most frequent reason (name breaks ties, matching
    /// `telemetry::rank_reasons` order).
    fn top_reason(map: &BTreeMap<String, u64>) -> Option<(&str, u64)> {
        map.iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(reason, count)| (reason.as_str(), *count))
    }

    pub(crate) fn health(&self) -> PersistedHealth {
        if self.fired > 0 {
            PersistedHealth::Alive
        } else if !self.failed.is_empty() {
            PersistedHealth::Failing
        } else if !self.declined.is_empty() {
            PersistedHealth::GatedOff
        } else {
            PersistedHealth::Silent
        }
    }
}

/// One session's final attest snapshot. Append-only; the reader keeps the
/// newest row per (session id, build SHA), so re-persisting mid-session
/// (e.g. `/refine` before exit) is harmless, and a session resumed under a
/// newer binary can never shadow what the original build recorded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedAttestRow {
    pub(crate) ts_ms: u64,
    pub(crate) git_sha: String,
    pub(crate) session_id: String,
    #[serde(default)]
    pub(crate) ablation: Vec<String>,
    #[serde(default)]
    pub(crate) features: BTreeMap<String, PersistedAttestation>,
}

fn snapshot_to_row(session_id: &str, git_sha: &str, ts_ms: u64) -> Option<PersistedAttestRow> {
    let ledger = telemetry::harness_attest_snapshot();
    if ledger.is_empty() {
        return None;
    }
    let mut features = BTreeMap::new();
    for (feature, attestation) in ledger.rows() {
        let persisted = PersistedAttestation {
            fired: attestation.fired,
            leaked: attestation.leaked,
            failed: attestation
                .failed
                .iter()
                .map(|(reason, count)| ((*reason).to_string(), *count))
                .collect(),
            declined: attestation
                .declined
                .iter()
                .map(|(reason, count)| ((*reason).to_string(), *count))
                .collect(),
        };
        features.insert(feature.key().to_string(), persisted);
    }
    Some(PersistedAttestRow {
        ts_ms,
        git_sha: git_sha.to_string(),
        session_id: session_id.to_string(),
        ablation: ledger
            .ablation()
            .keys()
            .into_iter()
            .map(ToString::to_string)
            .collect(),
        features,
    })
}

fn append_row(path: &Path, row: &PersistedAttestRow) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_string(row).map_err(std::io::Error::other)?;
    line.push('\n');
    core_types::paths::append_private_file(path, line.as_bytes())?;
    janitor_compact_if_oversized(path, row.ts_ms)?;
    Ok(())
}

/// Rewrite the row file keeping only recent rows once it outgrows the cap.
/// Losing old rows is fine — `/refine` never reads past its window — but a
/// failed compaction must never lose the file, hence write-then-rename via
/// the private-file helper on a sibling temp path.
fn janitor_compact_if_oversized(path: &Path, now_ms: u64) -> std::io::Result<()> {
    let len = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        Err(_) => return Ok(()),
    };
    if len <= JANITOR_MAX_BYTES {
        return Ok(());
    }
    let keep_after = now_ms.saturating_sub(JANITOR_KEEP_DAYS * 24 * 60 * 60 * 1000);
    let kept: Vec<String> = std::fs::read_to_string(path)?
        .lines()
        .filter(|line| {
            serde_json::from_str::<PersistedAttestRow>(line)
                .map(|row| row.ts_ms >= keep_after)
                .unwrap_or(false)
        })
        .map(str::to_string)
        .collect();
    let mut payload = kept.join("\n");
    if !payload.is_empty() {
        payload.push('\n');
    }
    let tmp = path.with_extension("jsonl.compact");
    core_types::paths::write_private_file(
        &tmp,
        payload.as_bytes(),
        &core_types::paths::ParentDirPolicy::CreateAndRestrict,
    )?;
    std::fs::rename(&tmp, path)
}

/// Persist this process's live attest snapshot under `session_id`. Empty
/// ledger ⇒ no row (a run that reached no attested feature says nothing).
/// Best-effort by contract: evidence persistence must never fail a session.
pub(crate) fn persist_attest_snapshot(session_id: &str) {
    let sha = crate::GIT_SHA.unwrap_or("unknown");
    let Some(row) = snapshot_to_row(session_id, sha, now_ms()) else {
        return;
    };
    let _ = append_row(&attest_evidence_path(), &row);
}

/// RAII teardown hook: entry paths create one, and the drop persists the
/// session's attest evidence on every exit route — including panics, where
/// the evidence of what the session reached is worth the most.
pub(crate) struct AttestPersistGuard {
    session_id: String,
}

impl AttestPersistGuard {
    pub(crate) fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
        }
    }
}

impl Drop for AttestPersistGuard {
    fn drop(&mut self) {
        persist_attest_snapshot(&self.session_id);
    }
}

// ── Aggregation ──────────────────────────────────────────────────────────────

fn read_rows(path: &Path) -> Vec<PersistedAttestRow> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    // Newest row per (session, build) wins: a ledger only ever grows within
    // a process, so the highest timestamp supersedes earlier snapshots of
    // the SAME build — but a session resumed under a different binary must
    // not shadow the evidence the original build recorded (the per-build
    // aggregation downstream depends on both rows surviving).
    let mut by_session_build: BTreeMap<(String, String), PersistedAttestRow> = BTreeMap::new();
    for line in text.lines() {
        let Ok(row) = serde_json::from_str::<PersistedAttestRow>(line) else {
            continue;
        };
        let key = (row.session_id.clone(), row.git_sha.clone());
        match by_session_build.get(&key) {
            Some(existing) if existing.ts_ms >= row.ts_ms => {}
            _ => {
                by_session_build.insert(key, row);
            }
        }
    }
    by_session_build.into_values().collect()
}

#[derive(Debug, Default)]
pub(crate) struct WindowAggregate {
    pub(crate) current: BTreeMap<String, PersistedAttestation>,
    /// Session ids contributing to `current`, per feature — the provenance a
    /// dream candidate needs.
    pub(crate) current_sessions_by_feature: BTreeMap<String, BTreeSet<String>>,
    pub(crate) current_session_count: usize,
    pub(crate) older_failing: BTreeSet<String>,
    pub(crate) older_session_count: usize,
    pub(crate) older_build_count: usize,
    /// Sessions that ran with an ablation active; their zeros are results,
    /// not findings, so they are excluded from `current` health entirely.
    pub(crate) ablated_session_count: usize,
}

pub(crate) fn aggregate_window(
    rows: &[PersistedAttestRow],
    current_sha: &str,
    now_ms: u64,
    window_days: u64,
) -> WindowAggregate {
    let cutoff = now_ms.saturating_sub(window_days * 24 * 60 * 60 * 1000);
    let mut aggregate = WindowAggregate::default();
    let mut older_builds = BTreeSet::new();
    for row in rows {
        if row.ts_ms < cutoff {
            continue;
        }
        if !row.ablation.is_empty() {
            aggregate.ablated_session_count += 1;
            continue;
        }
        if row.git_sha == current_sha {
            aggregate.current_session_count += 1;
            for (key, attestation) in &row.features {
                aggregate
                    .current
                    .entry(key.clone())
                    .or_default()
                    .merge_from(attestation);
                aggregate
                    .current_sessions_by_feature
                    .entry(key.clone())
                    .or_default()
                    .insert(row.session_id.clone());
            }
        } else {
            aggregate.older_session_count += 1;
            older_builds.insert(row.git_sha.clone());
            for (key, attestation) in &row.features {
                if attestation.health() == PersistedHealth::Failing {
                    aggregate.older_failing.insert(key.clone());
                }
            }
        }
    }
    aggregate.older_build_count = older_builds.len();
    aggregate
}

// ── Findings ─────────────────────────────────────────────────────────────────

/// The one command that opens a gated feature's door, for the features whose
/// door IS a setting. Everything else is gated by usage or by wire (the
/// precondition text says which), so there is no command to suggest.
fn door_command(key: &str) -> Option<&'static str> {
    telemetry::HarnessFeature::from_key(key).and_then(|feature| match feature {
        telemetry::HarnessFeature::RoutingProbe => Some("/smart autoClassifier probed"),
        _ => None,
    })
}

fn feature_label(key: &str) -> String {
    telemetry::HarnessFeature::from_key(key)
        .map_or_else(|| key.to_string(), |feature| feature.label().to_string())
}

fn feature_precondition(key: &str) -> Option<&'static str> {
    telemetry::HarnessFeature::from_key(key).map(telemetry::HarnessFeature::precondition)
}

/// A FAILING feature promoted into the Dreamer candidate store, so the
/// existing `/improve` pipeline (fusion → quarantined patch → review-first
/// apply) can plan a repair for it.
pub(crate) fn failing_candidate(
    key: &str,
    attestation: &PersistedAttestation,
    sessions: &BTreeSet<String>,
    git_sha: &str,
) -> SelfImproveCandidate {
    let top_reason = PersistedAttestation::top_reason(&attestation.failed)
        .map_or_else(|| "unknown".to_string(), |(reason, _)| reason.to_string());
    let evidence = sessions
        .iter()
        .take(3)
        .map(|session_id| CandidateEvidence {
            session_id: session_id.clone(),
            source: "refine_attest".to_string(),
            detail: format!(
                "build {git_sha}: harness feature `{key}` tried {}x, fired 0x; top reason: {top_reason}",
                attestation.failed_total(),
            ),
            // Counted by instrumentation at the failure site, not inferred
            // by a model — the strongest evidence class this store carries.
            verified: true,
        })
        .collect();
    SelfImproveCandidate::new(
        CandidateKind::HarnessDefect,
        format!("harness feature `{key}` failing: {top_reason}"),
        evidence,
    )
}

// ── Report ───────────────────────────────────────────────────────────────────

struct ShadowEvidence {
    mode_label: String,
    /// Whether the learned hint is already routing for real (`on`), read
    /// from the enum rather than parsed back out of the display label.
    mode_is_on: bool,
    stamp_count: usize,
    distinct_models: BTreeSet<String>,
}

fn shadow_evidence() -> ShadowEvidence {
    let (mode_label, mode_is_on) = super::smart_settings::read_global_smart_settings()
        .map_or_else(
            |_| ("unknown (settings unreadable)".to_string(), false),
            |snapshot| {
                (
                    snapshot.learned_specialty.status_label().to_string(),
                    matches!(
                        snapshot.learned_specialty,
                        super::smart_settings::SmartLearnedSpecialtyMode::On
                    ),
                )
            },
        );
    let stamps = super::smart_settings::scan_learned_shadow_stamps();
    let distinct_models = stamps.iter().map(|(_, model)| model.clone()).collect();
    ShadowEvidence {
        mode_label,
        mode_is_on,
        stamp_count: stamps.len(),
        distinct_models,
    }
}

/// Run `/refine`: persist the live snapshot, aggregate the evidence window,
/// and render the findings. Local reads only — no model call anywhere.
pub(crate) fn run_refine(cwd: &Path, session_id: &str, window_days: Option<u64>) -> String {
    // This session's evidence joins the window before it is read, so the
    // very first `/refine` a user ever runs already has something to say.
    persist_attest_snapshot(session_id);
    let sha = crate::GIT_SHA.unwrap_or("unknown");
    let rows = read_rows(&attest_evidence_path());
    let window = window_days.unwrap_or(DEFAULT_WINDOW_DAYS);
    let aggregate = aggregate_window(&rows, sha, now_ms(), window);
    let shadow = shadow_evidence();
    render_report(cwd, sha, window, &aggregate, &shadow)
}

#[allow(clippy::too_many_lines)] // one linear report assembly, sectioned by comments
fn render_report(
    cwd: &Path,
    sha: &str,
    window_days: u64,
    aggregate: &WindowAggregate,
    shadow: &ShadowEvidence,
) -> String {
    let mut lines = vec![
        "Refine — evidence-backed tuning report (applies nothing)".to_string(),
        "────────────────────────────────────────────────────────".to_string(),
        format!(
            "Window: last {window_days} day(s) · {} session(s) on this build ({}) · {} older-build session(s)",
            aggregate.current_session_count,
            short_sha(sha),
            aggregate.older_session_count,
        ),
    ];
    if aggregate.ablated_session_count > 0 {
        lines.push(format!(
            "  {} ablated session(s) excluded — their zeros are control-arm results, not findings",
            aggregate.ablated_session_count
        ));
    }

    // Harness health — findings first, then the alive digest.
    lines.push(String::new());
    lines.push("Harness health (this build)".to_string());
    if aggregate.current.is_empty() {
        lines.push(
            "  (no attested evidence on this build yet — finish a coding turn or two and rerun)"
                .to_string(),
        );
    }
    let mut alive = Vec::new();
    let mut candidates_written = 0usize;
    for (key, attestation) in &aggregate.current {
        match attestation.health() {
            PersistedHealth::Failing => {
                let (reason, count) = PersistedAttestation::top_reason(&attestation.failed)
                    .map_or(("unknown", 0), |(reason, count)| (reason, count));
                lines.push(format!(
                    "  ! FAILING  {} — tried {}x, fired 0x; top reason: {reason} ({count})",
                    feature_label(key),
                    attestation.failed_total(),
                ));
                let sessions = aggregate
                    .current_sessions_by_feature
                    .get(key)
                    .cloned()
                    .unwrap_or_default();
                let candidate = failing_candidate(key, attestation, &sessions, sha);
                match runtime::memory::record_self_improve_candidate(cwd, &candidate) {
                    Ok(()) => {
                        candidates_written += 1;
                        lines.push(format!(
                            "             -> recorded dream candidate `{}` — `/improve` can now plan a repair",
                            candidate.id
                        ));
                    }
                    Err(error) => lines.push(format!(
                        "             -> could not record a dream candidate ({error}); evidence stays in this report"
                    )),
                }
            }
            PersistedHealth::GatedOff => {
                let gate = PersistedAttestation::top_reason(&attestation.declined)
                    .map_or_else(String::new, |(reason, count)| {
                        format!("top gate: {reason} ({count})")
                    });
                lines.push(format!(
                    "  - gated    {} — declined {}x; {gate}",
                    feature_label(key),
                    attestation.declined_total(),
                ));
                if let Some(precondition) = feature_precondition(key) {
                    lines.push(format!("             {precondition}"));
                }
                if let Some(command) = door_command(key) {
                    lines.push(format!("             -> smallest change: {command}"));
                }
            }
            PersistedHealth::Alive => alive.push(format!(
                "{} {}x",
                feature_label(key),
                attestation.fired
            )),
            PersistedHealth::Silent => {}
        }
    }
    if !alive.is_empty() {
        lines.push(format!("  + alive    {}", alive.join(" · ")));
    }

    // Shadow routing evidence.
    lines.push(String::new());
    lines.push("Shadow routing (Phase 6 learned specialty)".to_string());
    lines.push(format!("  mode: {}", shadow.mode_label));
    if shadow.stamp_count == 0 {
        lines.push(
            "  no shadow-delta stamps yet — the learned pick has not differed from the seed pick"
                .to_string(),
        );
    } else {
        lines.push(format!(
            "  {} manifest stamp(s) where the learned pick would have differed ({} distinct model(s))",
            shadow.stamp_count,
            shadow.distinct_models.len(),
        ));
        if !shadow.mode_is_on {
            lines.push(
                "  -> smallest change: /smart learnedSpecialty on   (observation-only until then; stamps accrue at zero cost)"
                    .to_string(),
            );
        }
    }

    // Older builds: context only.
    if aggregate.older_session_count > 0 {
        lines.push(String::new());
        lines.push(format!(
            "Older builds ({} build(s), context only — never merged into this build's health)",
            aggregate.older_build_count
        ));
        if aggregate.older_failing.is_empty() {
            lines.push("  nothing was failing there".to_string());
        } else {
            lines.push(format!(
                "  failing there: {}",
                aggregate
                    .older_failing
                    .iter()
                    .map(|key| feature_label(key))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    lines.push(String::new());
    lines.push(if candidates_written > 0 {
        format!(
            "Nothing was changed. {candidates_written} dream candidate(s) recorded — review with /improve status; every other suggestion is the exact command that would apply it."
        )
    } else {
        "Nothing was changed. Every suggestion above is the exact command that would apply it."
            .to_string()
    });
    lines.join("\n")
}

fn short_sha(sha: &str) -> &str {
    if sha.len() > 12 { &sha[..12] } else { sha }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attestation(
        fired: u64,
        failed: &[(&str, u64)],
        declined: &[(&str, u64)],
    ) -> PersistedAttestation {
        PersistedAttestation {
            fired,
            leaked: 0,
            failed: failed
                .iter()
                .map(|(reason, count)| ((*reason).to_string(), *count))
                .collect(),
            declined: declined
                .iter()
                .map(|(reason, count)| ((*reason).to_string(), *count))
                .collect(),
        }
    }

    fn row(
        ts_ms: u64,
        sha: &str,
        session: &str,
        features: &[(&str, PersistedAttestation)],
    ) -> PersistedAttestRow {
        PersistedAttestRow {
            ts_ms,
            git_sha: sha.to_string(),
            session_id: session.to_string(),
            ablation: Vec::new(),
            features: features
                .iter()
                .map(|(key, attestation)| ((*key).to_string(), attestation.clone()))
                .collect(),
        }
    }

    /// The persisted health classifier must agree with the live one — the
    /// doctrine (failure outranks gates, gates outrank silence) lives in
    /// telemetry and this mirror may not drift from it.
    #[test]
    fn persisted_health_matches_telemetry_health() {
        let cases: Vec<(PersistedAttestation, telemetry::FeatureHealth, PersistedHealth)> = vec![
            (
                attestation(2, &[], &[]),
                telemetry::FeatureHealth::Alive,
                PersistedHealth::Alive,
            ),
            (
                attestation(0, &[("boom", 3)], &[("gate", 1)]),
                telemetry::FeatureHealth::Failing,
                PersistedHealth::Failing,
            ),
            (
                attestation(0, &[], &[("gate", 4)]),
                telemetry::FeatureHealth::GatedOff,
                PersistedHealth::GatedOff,
            ),
            (
                attestation(0, &[], &[]),
                telemetry::FeatureHealth::Silent,
                PersistedHealth::Silent,
            ),
        ];
        for (persisted, live_expected, persisted_expected) in cases {
            let static_reason = |reason: &str| -> &'static str {
                match reason {
                    "boom" => "boom",
                    "gate" => "gate",
                    other => panic!("unexpected reason {other}"),
                }
            };
            let live = telemetry::FeatureAttestation {
                fired: persisted.fired,
                leaked: persisted.leaked,
                failed: persisted
                    .failed
                    .iter()
                    .map(|(reason, count)| (static_reason(reason), *count))
                    .collect(),
                declined: persisted
                    .declined
                    .iter()
                    .map(|(reason, count)| (static_reason(reason), *count))
                    .collect(),
            };
            assert_eq!(live.health(), live_expected);
            assert_eq!(persisted.health(), persisted_expected);
        }
    }

    #[test]
    fn window_aggregation_separates_builds_and_respects_the_window() {
        const DAY: u64 = 24 * 60 * 60 * 1000;
        let now = 100 * DAY;
        let rows = vec![
            // Current build, in window: two sessions merge.
            row(
                now - DAY,
                "sha-new",
                "s1",
                &[("routing_probe", attestation(0, &[("http_400", 2)], &[]))],
            ),
            row(
                now - 2 * DAY,
                "sha-new",
                "s2",
                &[("routing_probe", attestation(0, &[("http_400", 3)], &[]))],
            ),
            // Older build in window: failing there must NOT poison current.
            row(
                now - 3 * DAY,
                "sha-old",
                "s3",
                &[("workflow_relay", attestation(0, &[("dead", 1)], &[]))],
            ),
            // Out of window entirely.
            row(
                now - 40 * DAY,
                "sha-new",
                "s4",
                &[("routing_probe", attestation(0, &[("http_400", 9)], &[]))],
            ),
        ];
        let aggregate = aggregate_window(&rows, "sha-new", now, 14);
        assert_eq!(aggregate.current_session_count, 2);
        assert_eq!(aggregate.older_session_count, 1);
        assert_eq!(aggregate.older_build_count, 1);
        let probe = aggregate.current.get("routing_probe").expect("merged");
        assert_eq!(probe.failed_total(), 5, "in-window sessions sum, s4 excluded");
        assert_eq!(probe.health(), PersistedHealth::Failing);
        assert!(!aggregate.current.contains_key("workflow_relay"));
        assert!(aggregate.older_failing.contains("workflow_relay"));
    }

    #[test]
    fn ablated_sessions_are_excluded_from_health_entirely() {
        const DAY: u64 = 24 * 60 * 60 * 1000;
        let now = 100 * DAY;
        let mut ablated = row(
            now - DAY,
            "sha-new",
            "s1",
            &[("routing_probe", attestation(0, &[("http_400", 2)], &[]))],
        );
        ablated.ablation = vec!["routing_probe".to_string()];
        let aggregate = aggregate_window(&[ablated], "sha-new", now, 14);
        assert_eq!(aggregate.ablated_session_count, 1);
        assert_eq!(aggregate.current_session_count, 0);
        assert!(aggregate.current.is_empty(), "control arms are not findings");
    }

    #[test]
    fn newest_row_per_session_wins_and_duplicates_do_not_double_count() {
        let dir = std::env::temp_dir().join(format!("zo-refine-rows-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("attest.jsonl");
        let _ = std::fs::remove_file(&path);
        let early = row(
            1_000,
            "sha",
            "s1",
            &[("routing_probe", attestation(1, &[], &[]))],
        );
        let late = row(
            2_000,
            "sha",
            "s1",
            &[("routing_probe", attestation(4, &[], &[]))],
        );
        append_row(&path, &early).expect("append early");
        append_row(&path, &late).expect("append late");
        let rows = read_rows(&path);
        assert_eq!(rows.len(), 1, "one logical session per build");
        assert_eq!(
            rows[0].features.get("routing_probe").expect("feature").fired,
            4,
            "the newest snapshot supersedes, never sums"
        );
        // The same session resumed under a DIFFERENT build keeps both rows:
        // per-build aggregation depends on the original build's evidence
        // never being shadowed by a later binary's snapshot.
        let resumed = row(
            3_000,
            "sha-next",
            "s1",
            &[("routing_probe", attestation(0, &[], &[])) ],
        );
        append_row(&path, &resumed).expect("append resumed");
        assert_eq!(read_rows(&path).len(), 2, "one row per (session, build)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn janitor_compacts_oversized_files_keeping_recent_rows() {
        const DAY: u64 = 24 * 60 * 60 * 1000;
        let dir = std::env::temp_dir().join(format!("zo-refine-janitor-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("attest.jsonl");
        let _ = std::fs::remove_file(&path);
        let now = 100 * DAY;
        // An old row, then enough padding rows to cross the cap on append.
        let mut payload = String::new();
        let old = row(now - 60 * DAY, "sha", "old", &[]);
        payload.push_str(&serde_json::to_string(&old).expect("json"));
        payload.push('\n');
        let filler = "x".repeat(4096);
        for index in 0..300 {
            let mut recent = row(now - DAY, "sha", &format!("s{index}"), &[]);
            recent.ablation = vec![filler.clone()];
            payload.push_str(&serde_json::to_string(&recent).expect("json"));
            payload.push('\n');
        }
        std::fs::write(&path, &payload).expect("seed file");
        assert!(std::fs::metadata(&path).expect("meta").len() > JANITOR_MAX_BYTES);
        janitor_compact_if_oversized(&path, now).expect("compact");
        let rows = read_rows(&path);
        assert!(rows.iter().all(|row| row.session_id != "old"));
        assert!(!rows.is_empty(), "recent rows survive compaction");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn failing_candidate_is_actionable_with_verified_evidence() {
        let mut sessions = BTreeSet::new();
        sessions.insert("session-1".to_string());
        sessions.insert("session-2".to_string());
        let candidate = failing_candidate(
            "routing_probe",
            &attestation(0, &[("http_400", 14)], &[]),
            &sessions,
            "sha-new",
        );
        assert_eq!(candidate.kind, CandidateKind::HarnessDefect);
        assert!(candidate.kind.is_actionable());
        assert_eq!(candidate.evidence.len(), 2);
        assert!(candidate.evidence.iter().all(|evidence| evidence.verified));
        assert!(candidate.summary.contains("routing_probe"));
        assert!(candidate.summary.contains("http_400"));
        assert!(
            candidate.id.starts_with("harness_defect-"),
            "deterministic id namespaced by kind: {}",
            candidate.id
        );
    }

    #[test]
    fn door_commands_exist_only_for_setting_gated_features() {
        assert_eq!(
            door_command("routing_probe"),
            Some("/smart autoClassifier probed")
        );
        for feature in telemetry::HarnessFeature::all() {
            if *feature == telemetry::HarnessFeature::RoutingProbe {
                continue;
            }
            assert_eq!(door_command(feature.key()), None, "{}", feature.key());
        }
        assert_eq!(door_command("not-a-feature"), None);
    }

    #[test]
    fn report_renders_failing_gated_alive_and_shadow_sections() {
        let dir = std::env::temp_dir().join(format!("zo-refine-report-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let mut aggregate = WindowAggregate {
            current_session_count: 2,
            ..WindowAggregate::default()
        };
        aggregate
            .current
            .insert("design_guidance".to_string(), attestation(0, &[], &[("intent", 7)]));
        aggregate
            .current
            .insert("info_topology".to_string(), attestation(9, &[], &[]));
        let shadow = ShadowEvidence {
            mode_label: "shadow".to_string(),
            mode_is_on: false,
            stamp_count: 3,
            distinct_models: ["a".to_string(), "b".to_string()].into_iter().collect(),
        };
        let report = render_report(&dir, "sha-current", 14, &aggregate, &shadow);
        assert!(report.contains("gated    design guidance reminder"), "{report}");
        assert!(
            report.contains("requires a turn whose resolved intent is Design"),
            "{report}"
        );
        assert!(report.contains("alive    information topology"), "{report}");
        assert!(report.contains("/smart learnedSpecialty on"), "{report}");
        assert!(report.contains("Nothing was changed."), "{report}");
        assert!(
            !report.contains("FAILING"),
            "no failing rows in this fixture: {report}"
        );
    }
}
