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
/// Door commands are the report's contract: the EXACT existing command that
/// applies a suggestion. Both are proven against the live `/smart` grammar
/// by `refine_door_commands_resolve_to_real_smart_subcommands` — a door that
/// the dispatcher rejects is a defect (the first two shipped here were: they
/// used the settings-key spellings `autoClassifier`/`learnedSpecialty`,
/// which the command grammar does not accept).
const ROUTING_PROBE_DOOR: &str = "/smart classifier probed";
const SHADOW_DOOR: &str = "/smart learned on";

fn door_command(key: &str) -> Option<&'static str> {
    telemetry::HarnessFeature::from_key(key).and_then(|feature| match feature {
        telemetry::HarnessFeature::RoutingProbe => Some(ROUTING_PROBE_DOOR),
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
    /// Learned-specialty entries the engine currently holds at all.
    learned_entry_count: usize,
    /// Display lines for entries clearing the router's own rung-admission
    /// predicate — the promotion gate's evidence. Promotion is only ever
    /// recommended when this is non-empty: turning the mode on with nothing
    /// armed changes no routing decision.
    armed_entries: Vec<String>,
}

fn shadow_evidence(cwd: &Path) -> ShadowEvidence {
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
    let (learned_entry_count, armed_entries) =
        super::smart_settings::learned_promotion_evidence(cwd);
    ShadowEvidence {
        mode_label,
        mode_is_on,
        stamp_count: stamps.len(),
        distinct_models,
        learned_entry_count,
        armed_entries,
    }
}

// ── Head-to-head bench evidence ──────────────────────────────────────────────

/// One tool's aggregate line from a `zo-bench` `scoreboard.json`.
struct BenchToolSummary {
    tool: String,
    tasks: u64,
    successes: u64,
    median_wall_ms: u64,
    total_cost_usd: f64,
    total_tokens: u64,
}

/// The newest bench run's scoreboard, reduced to what the report says:
/// the per-tool digest, the axes where this tool trails each rival, and
/// the tasks it lost outright. Absence of a scoreboard is not a finding —
/// callers render nothing when this is `None`.
struct BenchEvidence {
    run_name: String,
    age_days: u64,
    tools: Vec<BenchToolSummary>,
    verdicts: Vec<String>,
    lost_tasks: Vec<String>,
}

/// Which scoreboard row is "us". Derived from the running binary's name so
/// the report never pins a product name the bench config didn't use; a tool
/// list that doesn't include it still renders, just without verdict lines.
fn self_tool_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.file_stem().map(|stem| stem.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "zo".to_string())
}

fn bench_evidence(cwd: &Path) -> Option<BenchEvidence> {
    bench_evidence_for(cwd, &self_tool_name(), now_ms() / 1000)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn percent_over(mine: f64, rival: f64) -> u64 {
    if rival <= 0.0 {
        return 0;
    }
    (((mine - rival) / rival) * 100.0).round().max(0.0) as u64
}

#[allow(clippy::cast_precision_loss)]
fn format_wall(ms: u64) -> String {
    format!("{:.1}s", ms as f64 / 1000.0)
}

fn format_tokens(tokens: u64) -> String {
    if tokens >= 1000 {
        format!("{}k", (tokens + 500) / 1000)
    } else {
        tokens.to_string()
    }
}

fn bench_evidence_for(cwd: &Path, self_tool: &str, now_secs: u64) -> Option<BenchEvidence> {
    let results_dir = cwd.join("bench").join("results");
    let mut newest: Option<(u64, String)> = None;
    for entry in std::fs::read_dir(&results_dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(epoch) = name
            .strip_prefix("run-")
            .and_then(|suffix| suffix.parse::<u64>().ok())
        else {
            continue;
        };
        if !entry.path().join("scoreboard.json").is_file() {
            continue;
        }
        if newest.as_ref().is_none_or(|(top, _)| epoch > *top) {
            newest = Some((epoch, name));
        }
    }
    let (epoch, run_name) = newest?;
    let raw = std::fs::read_to_string(results_dir.join(&run_name).join("scoreboard.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;

    let tools: Vec<BenchToolSummary> = value
        .get("tools")?
        .as_array()?
        .iter()
        .filter_map(|tool| {
            Some(BenchToolSummary {
                tool: tool.get("tool")?.as_str()?.to_string(),
                tasks: tool.get("tasks")?.as_u64()?,
                successes: tool.get("successes")?.as_u64()?,
                median_wall_ms: tool.get("median_wall_ms")?.as_u64()?,
                total_cost_usd: tool.get("total_cost_usd")?.as_f64()?,
                total_tokens: tool.get("total_tokens")?.as_u64()?,
            })
        })
        .collect();
    if tools.is_empty() {
        return None;
    }

    Some(BenchEvidence {
        run_name,
        age_days: now_secs.saturating_sub(epoch) / (24 * 60 * 60),
        verdicts: bench_verdicts(&tools, self_tool),
        lost_tasks: bench_lost_tasks(&value, self_tool),
        tools,
    })
}

/// Per-task outcomes: a task is lost when none of our trials passed it; the
/// rivals that DID pass it are named so the loss is reproducible.
fn bench_lost_tasks(scoreboard: &serde_json::Value, self_tool: &str) -> Vec<String> {
    let mut self_task_passed: BTreeMap<String, bool> = BTreeMap::new();
    let mut rivals_passed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let empty = Vec::new();
    for row in scoreboard
        .get("rows")
        .and_then(serde_json::Value::as_array)
        .unwrap_or(&empty)
    {
        let (Some(task), Some(tool), Some(success)) = (
            row.get("task").and_then(serde_json::Value::as_str),
            row.get("tool").and_then(serde_json::Value::as_str),
            row.get("success").and_then(serde_json::Value::as_bool),
        ) else {
            continue;
        };
        if tool == self_tool {
            *self_task_passed.entry(task.to_string()).or_insert(false) |= success;
        } else if success {
            rivals_passed.entry(task.to_string()).or_default().insert(tool.to_string());
        }
    }
    self_task_passed
        .iter()
        .filter(|(_, passed)| !**passed)
        .map(|(task, _)| {
            rivals_passed.get(task).map_or_else(
                || format!("{task} (no tool passed)"),
                |winners| {
                    format!(
                        "{task} ({} passed)",
                        winners.iter().cloned().collect::<Vec<_>>().join(", ")
                    )
                },
            )
        })
        .collect()
}

/// One line per rival naming every axis where our row trails theirs. No self
/// row in the scoreboard ⇒ no verdicts — the summary still renders.
#[allow(clippy::cast_precision_loss)]
fn bench_verdicts(tools: &[BenchToolSummary], self_tool: &str) -> Vec<String> {
    let Some(mine) = tools.iter().find(|tool| tool.tool == self_tool) else {
        return Vec::new();
    };
    tools
        .iter()
        .filter(|rival| rival.tool != self_tool)
        .map(|rival| {
            let mut behind = Vec::new();
            if mine.successes * rival.tasks < rival.successes * mine.tasks {
                behind.push(format!(
                    "pass rate ({}/{} vs {}/{})",
                    mine.successes, mine.tasks, rival.successes, rival.tasks
                ));
            }
            if mine.median_wall_ms > rival.median_wall_ms {
                behind.push(format!(
                    "median wall (+{}%)",
                    percent_over(mine.median_wall_ms as f64, rival.median_wall_ms as f64)
                ));
            }
            if mine.total_cost_usd > rival.total_cost_usd {
                behind.push(format!(
                    "cost (+{}%)",
                    percent_over(mine.total_cost_usd, rival.total_cost_usd)
                ));
            }
            if mine.total_tokens > rival.total_tokens {
                behind.push(format!(
                    "tokens (+{}%)",
                    percent_over(mine.total_tokens as f64, rival.total_tokens as f64)
                ));
            }
            if behind.is_empty() {
                format!("vs {}: ahead or even on every axis", rival.tool)
            } else {
                format!("vs {}: behind on {}", rival.tool, behind.join(", "))
            }
        })
        .collect()
}

// ── Repeated-workflow evidence (skill distill trigger) ───────────────────────

/// A cluster of recent sessions that opened with near-identical asks — the
/// evidence axis for "this workflow repeats; a skill would compress it".
/// Detection only: creating the skill stays a model+human action, and the
/// section renders nothing when nothing repeats.
struct RepeatedWorkflow {
    session_count: usize,
    sample: String,
    shared_terms: Vec<String>,
}

/// Meaningful tokens of one opening ask. ASCII words are lowercased and kept
/// at ≥3 chars minus a tiny stopword list; CJK runs are kept whole AND as
/// their leading two syllables, so Korean verb-suffix variation
/// ("배포해줘" / "배포 절차") still overlaps on the stem. A heuristic, not
/// a parser — the cluster thresholds below carry the precision.
fn ask_tokens(text: &str) -> BTreeSet<String> {
    const STOPWORDS: [&str; 10] = [
        "the", "and", "for", "this", "that", "with", "into", "from", "please", "run",
    ];
    fn flush(word: &mut String, is_cjk: bool, tokens: &mut BTreeSet<String>) {
        if word.is_empty() {
            return;
        }
        let token = word.to_lowercase();
        word.clear();
        if is_cjk {
            let syllables: Vec<char> = token.chars().collect();
            if syllables.len() >= 2 {
                tokens.insert(syllables.iter().take(2).collect());
                tokens.insert(token);
            }
        } else if token.chars().count() >= 3
            && !STOPWORDS.contains(&token.as_str())
            && !token.chars().all(|c| c.is_ascii_digit())
        {
            tokens.insert(token);
        }
    }
    let mut tokens = BTreeSet::new();
    let mut word = String::new();
    let mut word_cjk = false;
    for ch in text.chars() {
        let is_ascii_word = ch.is_ascii_alphanumeric();
        let is_cjk = matches!(ch,
            '\u{AC00}'..='\u{D7AF}' | '\u{4E00}'..='\u{9FFF}' | '\u{3040}'..='\u{30FF}');
        if is_ascii_word || is_cjk {
            if !word.is_empty() && word_cjk != is_cjk {
                flush(&mut word, word_cjk, &mut tokens);
            }
            word_cjk = is_cjk;
            word.push(ch);
        } else {
            flush(&mut word, word_cjk, &mut tokens);
        }
    }
    flush(&mut word, word_cjk, &mut tokens);
    tokens
}

/// Greedy seed clustering over opening asks: a member shares ≥2 tokens with
/// the seed at Jaccard ≥ 0.4, and only clusters of ≥3 sessions count —
/// two similar asks are coincidence, three are a workflow. Top two clusters
/// by size, so the report never scrolls on repetition evidence.
fn cluster_repeated_asks(asks: &[String]) -> Vec<RepeatedWorkflow> {
    let tokenized: Vec<(&String, BTreeSet<String>)> = asks
        .iter()
        .map(|ask| (ask, ask_tokens(ask)))
        .filter(|(_, tokens)| tokens.len() >= 2)
        .collect();
    let mut assigned = vec![false; tokenized.len()];
    let mut out = Vec::new();
    for seed in 0..tokenized.len() {
        if assigned[seed] {
            continue;
        }
        let mut members = vec![seed];
        for candidate in (seed + 1)..tokenized.len() {
            if assigned[candidate] {
                continue;
            }
            let shared = tokenized[seed].1.intersection(&tokenized[candidate].1).count();
            let union = tokenized[seed].1.union(&tokenized[candidate].1).count();
            if shared >= 2 && shared * 10 >= union * 4 {
                members.push(candidate);
            }
        }
        assigned[seed] = true;
        if members.len() < 3 {
            continue;
        }
        for &member in &members {
            assigned[member] = true;
        }
        // Terms every member shares with the seed — each member shares ≥2
        // seed tokens pairwise, but not necessarily the same two, so this
        // can legitimately come up short; the sample ask still identifies
        // the cluster.
        let shared_terms: Vec<String> = tokenized[seed]
            .1
            .iter()
            .filter(|token| {
                members[1..]
                    .iter()
                    .all(|&member| tokenized[member].1.contains(*token))
            })
            .take(4)
            .cloned()
            .collect();
        out.push(RepeatedWorkflow {
            session_count: members.len(),
            sample: tokenized[seed].0.chars().take(48).collect(),
            shared_terms,
        });
    }
    out.sort_by(|left, right| right.session_count.cmp(&left.session_count));
    out.truncate(2);
    out
}

/// Opening asks of this project's recent worked sessions, from the registry's
/// already-computed summaries. "Worked" = enough messages that the session
/// plausibly held a workflow, not a one-line Q&A.
fn repeated_workflow_evidence(window_days: u64) -> Vec<RepeatedWorkflow> {
    let Ok(sessions) = crate::session_registry::list_managed_sessions_limited(Some(48)) else {
        return Vec::new();
    };
    let now = u128::from(now_ms());
    let window = u128::from(window_days) * 24 * 60 * 60 * 1000;
    let asks: Vec<String> = sessions
        .iter()
        .filter(|session| session.message_count >= 6)
        .filter(|session| now.saturating_sub(session.modified_epoch_millis) <= window)
        .filter_map(|session| session.first_user_text.clone())
        .collect();
    cluster_repeated_asks(&asks)
}

/// The launchpad's one-line pointer, computed at boot on the startup loader
/// thread. Pure read — no persist, no candidate write; those stay behind the
/// explicit `/refine`.
///
/// Deliberately fires on current-build FAILING features ONLY. Gated features
/// and shadow stamps exist on most healthy installs, so a boot line for them
/// would be a permanent nag (the `needs_onboarding` OR-predicate mistake all
/// over again); a FAILING attestation is rare, actionable, and worth the
/// interruption. Zero findings ⇒ `None` ⇒ the launchpad says nothing.
pub(crate) fn startup_notice() -> Option<String> {
    let sha = crate::GIT_SHA.unwrap_or("unknown");
    let rows = read_rows(&attest_evidence_path());
    let aggregate = aggregate_window(&rows, sha, now_ms(), DEFAULT_WINDOW_DAYS);
    startup_notice_from(&aggregate)
}

fn startup_notice_from(aggregate: &WindowAggregate) -> Option<String> {
    let failing = aggregate
        .current
        .values()
        .filter(|attestation| attestation.health() == PersistedHealth::Failing)
        .count();
    (failing > 0).then(|| {
        format!(
            "{failing} harness feature{} failing — /refine",
            if failing == 1 { "" } else { "s" },
        )
    })
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
    let shadow = shadow_evidence(cwd);
    let proposed_skills = tools::stranded_proposed_skills(cwd);
    let bench = bench_evidence(cwd);
    let repeated = repeated_workflow_evidence(window);
    render_report(
        cwd,
        sha,
        window,
        &aggregate,
        &shadow,
        &proposed_skills,
        bench.as_ref(),
        &repeated,
    )
}

// One flat evidence-section-per-argument report; bundling them into a struct
// would only rename the same eight things.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn render_report(
    cwd: &Path,
    sha: &str,
    window_days: u64,
    aggregate: &WindowAggregate,
    shadow: &ShadowEvidence,
    proposed_skills: &[tools::ProposedSkill],
    bench: Option<&BenchEvidence>,
    repeated: &[RepeatedWorkflow],
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

    // Head-to-head bench — the "are we winning" evidence, straight from the
    // newest scoreboard artifact. Absence of a run renders nothing: not
    // having measured is not a finding, and a nag row would push people to
    // burn tokens on benches they didn't ask for.
    if let Some(bench) = bench {
        lines.push(String::new());
        lines.push(format!(
            "Head-to-head bench (bench/results/{} · {})",
            bench.run_name,
            if bench.age_days == 0 {
                "today".to_string()
            } else {
                format!("{} day(s) ago", bench.age_days)
            },
        ));
        let name_width = bench
            .tools
            .iter()
            .map(|tool| tool.tool.len())
            .max()
            .unwrap_or(0);
        for tool in &bench.tools {
            lines.push(format!(
                "  {:<name_width$}  {}/{} pass · {} median · ${:.2} · {} tokens",
                tool.tool,
                tool.successes,
                tool.tasks,
                format_wall(tool.median_wall_ms),
                tool.total_cost_usd,
                format_tokens(tool.total_tokens),
            ));
        }
        for verdict in &bench.verdicts {
            lines.push(format!("  {verdict}"));
        }
        if !bench.lost_tasks.is_empty() {
            lines.push(format!("  lost task(s): {}", bench.lost_tasks.join(" · ")));
        }
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
    lines.push("Shadow routing (learned specialty)".to_string());
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
    }
    // Promotion gate: `/smart learned on` is recommended ONLY when an entry
    // clears the router's own rung-admission predicate — the exact bar that
    // changes a routing decision once the mode is on. Stamps prove the
    // learned hint DIFFERS; an armed entry is the measured claim that it is
    // RIGHT. Recommending on stamp volume alone would be advice without
    // evidence — and turning the mode on with nothing armed changes nothing.
    if !shadow.mode_is_on {
        if !shadow.armed_entries.is_empty() {
            lines.push(format!(
                "  {} learned entr{} clear the router's rung-admission bar:",
                shadow.armed_entries.len(),
                if shadow.armed_entries.len() == 1 { "y" } else { "ies" },
            ));
            for armed in shadow.armed_entries.iter().take(4) {
                lines.push(format!("    {armed}"));
            }
            lines.push(format!(
                "  -> promotion evidence met — smallest change: {SHADOW_DOOR}"
            ));
        } else if shadow.learned_entry_count > 0 {
            lines.push(format!(
                "  0 of {} learned entr{} clear the rung-admission bar yet — promotion would not change routing; the shadow soak continues at zero cost",
                shadow.learned_entry_count,
                if shadow.learned_entry_count == 1 { "y" } else { "ies" },
            ));
        }
    }

    // Distilled skills stranded behind the review gate. The gate that keeps
    // a proposed skill unusable is only honest if something shows the human
    // the queue — this is that surface.
    if !proposed_skills.is_empty() {
        lines.push(String::new());
        lines.push("Distilled skills awaiting review".to_string());
        for skill in proposed_skills {
            lines.push(format!(
                "  ~ proposed  {} ({}) — say \"approve the skill {}\" (or discard it) to decide; it stays unusable until then",
                skill.slug, skill.origin, skill.slug,
            ));
        }
    }

    // Repetition evidence — recent sessions that opened with the same ask.
    // Detection only: the distill itself stays a model+human action, and
    // silence is the default (no repetition ⇒ no section, never a nag).
    if !repeated.is_empty() {
        lines.push(String::new());
        lines.push("Repeated workflows (skill evidence)".to_string());
        for workflow in repeated {
            let shared = if workflow.shared_terms.is_empty() {
                String::new()
            } else {
                format!(" (shared: {})", workflow.shared_terms.join(", "))
            };
            lines.push(format!(
                "  ~ {} recent sessions opened with variants of \"{}\"{shared}",
                workflow.session_count, workflow.sample,
            ));
        }
        lines.push(
            "  -> a skill would make this one command — say \"distill this workflow into a skill\" in such a session"
                .to_string(),
        );
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
    fn startup_notice_fires_only_on_current_build_failing_features() {
        let mut aggregate = WindowAggregate::default();
        assert_eq!(startup_notice_from(&aggregate), None, "empty window is silent");
        aggregate
            .current
            .insert("design_guidance".to_string(), attestation(0, &[], &[("intent", 7)]));
        aggregate
            .current
            .insert("info_topology".to_string(), attestation(3, &[], &[]));
        assert_eq!(
            startup_notice_from(&aggregate),
            None,
            "gated and alive features never nag the launchpad"
        );
        aggregate
            .current
            .insert("routing_probe".to_string(), attestation(0, &[("http_400", 5)], &[]));
        assert_eq!(
            startup_notice_from(&aggregate).as_deref(),
            Some("1 harness feature failing — /refine")
        );
        aggregate
            .current
            .insert("workflow_relay".to_string(), attestation(0, &[("dead", 1)], &[]));
        assert_eq!(
            startup_notice_from(&aggregate).as_deref(),
            Some("2 harness features failing — /refine")
        );
    }

    #[test]
    fn door_commands_exist_only_for_setting_gated_features() {
        assert_eq!(door_command("routing_probe"), Some(ROUTING_PROBE_DOOR));
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
        let shadow = shadow_fixture();
        let report = render_report(&dir, "sha-current", 14, &aggregate, &shadow, &[], None, &[]);
        assert!(report.contains("gated    design guidance reminder"), "{report}");
        assert!(
            report.contains("requires a turn whose resolved intent is Design"),
            "{report}"
        );
        assert!(report.contains("alive    information topology"), "{report}");
        assert!(
            report.contains("1 learned entry clear the router's rung-admission bar"),
            "{report}"
        );
        assert!(
            report.contains("executor: model-x (+72, 910‰ confidence)"),
            "{report}"
        );
        assert!(
            report.contains("promotion evidence met — smallest change: /smart learned on"),
            "{report}"
        );
        assert!(report.contains("Nothing was changed."), "{report}");
        assert!(
            !report.contains("FAILING"),
            "no failing rows in this fixture: {report}"
        );
        assert!(
            !report.contains("Distilled skills"),
            "no skills section without proposed drafts: {report}"
        );

        // With nothing armed the door must NOT print: turning the mode on
        // would change no routing decision, so recommending it is advice
        // without evidence.
        let unarmed = ShadowEvidence {
            armed_entries: Vec::new(),
            learned_entry_count: 3,
            ..shadow_fixture()
        };
        let soaking = render_report(&dir, "sha-current", 14, &aggregate, &unarmed, &[], None, &[]);
        assert!(
            soaking.contains("0 of 3 learned entries clear the rung-admission bar yet"),
            "{soaking}"
        );
        assert!(
            !soaking.contains("/smart learned on"),
            "no promotion door without armed evidence: {soaking}"
        );

        let proposed = vec![tools::ProposedSkill {
            slug: "tui-palette-fixes".to_string(),
            origin: "project-zo".to_string(),
            path: "/tmp/x/SKILL.md".to_string(),
        }];
        let with_skills =
            render_report(&dir, "sha-current", 14, &aggregate, &shadow, &proposed, None, &[]);
        assert!(
            with_skills.contains("~ proposed  tui-palette-fixes (project-zo)"),
            "{with_skills}"
        );
        assert!(
            with_skills.contains("approve the skill tui-palette-fixes"),
            "{with_skills}"
        );
        assert!(
            !with_skills.contains("Head-to-head bench"),
            "no bench section without a scoreboard: {with_skills}"
        );
    }

    fn shadow_fixture() -> ShadowEvidence {
        ShadowEvidence {
            mode_label: "shadow".to_string(),
            mode_is_on: false,
            stamp_count: 3,
            distinct_models: ["a".to_string(), "b".to_string()].into_iter().collect(),
            learned_entry_count: 2,
            armed_entries: vec!["executor: model-x (+72, 910‰ confidence)".to_string()],
        }
    }

    fn write_scoreboard(dir: &Path, run: &str, body: &serde_json::Value) {
        let run_dir = dir.join("bench").join("results").join(run);
        std::fs::create_dir_all(&run_dir).expect("mkdir");
        std::fs::write(
            run_dir.join("scoreboard.json"),
            serde_json::to_string(body).expect("json"),
        )
        .expect("write scoreboard");
    }

    #[test]
    fn bench_evidence_reads_the_newest_run_and_names_losing_axes_and_lost_tasks() {
        const DAY: u64 = 24 * 60 * 60;
        let dir = std::env::temp_dir().join(format!("zo-refine-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // A stale run that must NOT be picked, and a run directory without a
        // scoreboard artifact that must be skipped entirely.
        write_scoreboard(&dir, "run-100", &serde_json::json!({"tools": [], "rows": []}));
        std::fs::create_dir_all(dir.join("bench").join("results").join("run-999"))
            .expect("empty run dir");
        write_scoreboard(
            &dir,
            "run-200",
            &serde_json::json!({
                "tools": [
                    {"tool": "zo", "tasks": 6, "successes": 5, "median_wall_ms": 31_000,
                     "total_cost_usd": 1.29, "total_tokens": 910_921},
                    {"tool": "claude-code", "tasks": 6, "successes": 6, "median_wall_ms": 29_000,
                     "total_cost_usd": 1.98, "total_tokens": 1_098_867},
                ],
                "rows": [
                    {"task": "rust-compile-fix", "tool": "zo", "success": false},
                    {"task": "rust-compile-fix", "tool": "claude-code", "success": true},
                    {"task": "fix-off-by-one", "tool": "zo", "success": true},
                    {"task": "fix-off-by-one", "tool": "claude-code", "success": true},
                ],
            }),
        );

        let bench = bench_evidence_for(&dir, "zo", 200 + 3 * DAY).expect("bench evidence");
        assert_eq!(bench.run_name, "run-200", "newest run with an artifact wins");
        assert_eq!(bench.age_days, 3);
        assert_eq!(bench.verdicts.len(), 1);
        assert!(
            bench.verdicts[0].contains("behind on pass rate (5/6 vs 6/6)"),
            "{}",
            bench.verdicts[0]
        );
        assert!(
            bench.verdicts[0].contains("median wall (+7%)"),
            "{}",
            bench.verdicts[0]
        );
        assert!(
            !bench.verdicts[0].contains("cost") && !bench.verdicts[0].contains("tokens"),
            "axes we lead must not be listed as behind: {}",
            bench.verdicts[0]
        );
        assert_eq!(bench.lost_tasks, vec!["rust-compile-fix (claude-code passed)"]);

        let report = render_report(
            &dir,
            "sha-current",
            14,
            &WindowAggregate::default(),
            &ShadowEvidence {
                mode_label: "shadow".to_string(),
                mode_is_on: false,
                stamp_count: 0,
                distinct_models: BTreeSet::new(),
                learned_entry_count: 0,
                armed_entries: Vec::new(),
            },
            &[],
            Some(&bench),
            &[],
        );
        assert!(
            report.contains("Head-to-head bench (bench/results/run-200 · 3 day(s) ago)"),
            "{report}"
        );
        assert!(report.contains("zo           5/6 pass · 31.0s median · $1.29 · 911k tokens"), "{report}");
        assert!(
            report.contains("claude-code  6/6 pass · 29.0s median · $1.98 · 1099k tokens"),
            "{report}"
        );
        assert!(report.contains("lost task(s): rust-compile-fix (claude-code passed)"), "{report}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bench_evidence_without_a_self_row_still_summarizes_but_stays_verdict_free() {
        let dir = std::env::temp_dir().join(format!("zo-refine-bench-noself-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_scoreboard(
            &dir,
            "run-50",
            &serde_json::json!({
                "tools": [
                    {"tool": "claude-code", "tasks": 2, "successes": 2, "median_wall_ms": 1000,
                     "total_cost_usd": 0.10, "total_tokens": 900},
                ],
                "rows": [],
            }),
        );
        let bench = bench_evidence_for(&dir, "zo", 50).expect("bench evidence");
        assert!(bench.verdicts.is_empty(), "no self row, no verdict");
        assert!(bench.lost_tasks.is_empty());
        assert_eq!(bench.tools.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every door the report can print must be a command the live `/smart`
    /// grammar accepts. Probing with the SUBCOMMAND ALONE keeps the check
    /// side-effect free: a real subcommand missing its argument answers with
    /// a usage error, while a made-up one answers "Unsupported subcommand" —
    /// exactly how both original doors (settings-key spellings) were broken.
    #[test]
    fn refine_door_commands_resolve_to_real_smart_subcommands() {
        let mut doors: Vec<&str> = vec![SHADOW_DOOR];
        doors.extend(
            telemetry::HarnessFeature::all()
                .iter()
                .filter_map(|feature| door_command(feature.key())),
        );
        for door in doors {
            let subcommand = door
                .strip_prefix("/smart ")
                .unwrap_or_else(|| panic!("door must be a /smart command: {door}"))
                .split_whitespace()
                .next()
                .expect("subcommand token");
            let outcome = super::super::smart_settings::execute_smart_text_command(
                "claude-opus-5",
                Some(subcommand),
            );
            if let Err(error) = outcome {
                assert!(
                    !error.contains("Unsupported subcommand"),
                    "door `{door}` names a subcommand the dispatcher rejects: {error}"
                );
            }
        }
    }

    #[test]
    fn repeated_asks_cluster_across_suffix_variants_and_ignore_generic_asks() {
        let asks: Vec<String> = [
            "release 배포 절차 실행해줘",
            "release 배포해줘",
            "이번 버전 release 배포 진행",
            // Generic one-token asks must never form a cluster.
            "진행",
            "고쳐줘",
            "계속",
            // Two similar asks are coincidence, not a workflow.
            "bench 스코어보드 돌려줘",
            "bench 스코어보드 실행",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let clusters = cluster_repeated_asks(&asks);
        assert_eq!(clusters.len(), 1, "only the 3-session release cluster counts");
        assert_eq!(clusters[0].session_count, 3);
        assert!(
            clusters[0].shared_terms.iter().any(|term| term == "release")
                && clusters[0].shared_terms.iter().any(|term| term == "배포"),
            "{:?}",
            clusters[0].shared_terms
        );
        assert!(clusters[0].sample.contains("release"));
    }

    #[test]
    fn repeated_workflow_section_renders_only_with_evidence() {
        let dir = std::env::temp_dir().join(format!("zo-refine-repeat-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let repeated = vec![RepeatedWorkflow {
            session_count: 4,
            sample: "release 배포 절차 실행해줘".to_string(),
            shared_terms: vec!["release".to_string(), "배포".to_string()],
        }];
        let report = render_report(
            &dir,
            "sha-current",
            14,
            &WindowAggregate::default(),
            &shadow_fixture(),
            &[],
            None,
            &repeated,
        );
        assert!(
            report.contains(
                "~ 4 recent sessions opened with variants of \"release 배포 절차 실행해줘\" (shared: release, 배포)"
            ),
            "{report}"
        );
        assert!(
            report.contains("say \"distill this workflow into a skill\""),
            "{report}"
        );

        let without = render_report(
            &dir,
            "sha-current",
            14,
            &WindowAggregate::default(),
            &shadow_fixture(),
            &[],
            None,
            &[],
        );
        assert!(
            !without.contains("Repeated workflows"),
            "no repetition, no section: {without}"
        );
    }

    #[test]
    fn bench_evidence_is_none_without_any_scoreboard() {
        let dir = std::env::temp_dir().join(format!("zo-refine-bench-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        assert!(bench_evidence_for(&dir, "zo", 0).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
