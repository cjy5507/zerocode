//! `/improve` — the dreamer self-repair pipeline, end to end and human-gated.
//!
//! `/improve` reads the top self-improvement candidate, runs the `DreamFusion`
//! advisors, GENERATES a fix as a unified diff in an isolated git worktree (never
//! touching the user's tree), validates it through the tested quarantine (clean
//! re-apply + focused checks) and the manual apply gate, and persists the
//! proposal. `/improve apply` then applies it — but only when the gate is
//! satisfied with explicit human approval.
//!
//! Patch generation is the one piece the dreamer infra lacked, so it is the only
//! new behavior here. It is delegated to a [`PatchGenerator`] so the
//! orchestration is unit-testable with a stub; the production generator runs a
//! headless `zo -p` turn inside the worktree (reusing the whole zo binary
//! rather than a bespoke inference path).

use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use decision_core::dreamer::{DreamFusionReport, DreamJudgeDecision, QuarantinePatchRun};
use runtime::memory::{
    evaluate_manual_apply_gate, mark_self_improve_candidate_applied,
    mark_self_improve_candidate_rejected, read_self_improve_schedule_state,
    record_self_improve_attempt, run_dream_fusion_v0, run_quarantine_patch,
    should_run_self_improve, try_acquire_self_improve_lock, ManualApplyGateRequest,
    QuarantineCheckCommand, QuarantinePatchRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Max agentic turns the headless generation pass may take.
const GENERATION_MAX_TURNS: usize = 12;
/// Wall-clock ceiling on the headless generation pass. A stalled provider must
/// not wedge `/improve`, so the child is killed once this elapses.
const GENERATION_TIMEOUT: Duration = Duration::from_secs(600);
/// Upper bound on captured child stdio so a chatty/looping child cannot grow the
/// buffer without limit; output past this is dropped (it is diagnostic only —
/// the proposal is the worktree `git diff`, not the child's stdout).
const GENERATION_OUTPUT_CAP: usize = 64 * 1024;
/// Source roots a self-improve patch may touch; anything else is gate-blocked.
const ALLOWED_PATH_ROOTS: &[&str] = &["crates", "docs", "Cargo.toml"];
const HARD_MIN_DISK_BYTES: u64 = 128 * 1024 * 1024;
const APPROVAL_ENVELOPE_VERSION: &str = "zo-self-improve-approval-v1";
const CHECK_POLICY_VERSION: &str = "cargo-check-workspace-all-targets-locked-v1";
/// Minimum spacing between automatic self-improve preflight attempts. The
/// preflight is cheap and safe, but marker/backoff throttling prevents startup
/// churn and preserves room for the explicit `/improve` pipeline.
const AUTO_SELF_IMPROVE_MIN_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// Backoff after an automatic self-improve preflight failure.
const AUTO_SELF_IMPROVE_FAILURE_BACKOFF: Duration = Duration::from_secs(6 * 60 * 60);

fn disk_refusal(available: u64, parent: &Path) -> Option<String> {
    (available < HARD_MIN_DISK_BYTES).then(|| {
        format!(
            "refusing to run: only {}MB left on the filesystem holding {} — free disk space first (reclaim Rust target/ build dirs, temp scratch, or orphaned worktrees)",
            available / (1024 * 1024),
            parent.display(),
        )
    })
}

fn disk_preflight(parent: &Path) -> Result<Option<String>, String> {
    if let Some(error) = runtime::available_disk_bytes(parent)
        .and_then(|available| disk_refusal(available, parent))
    {
        return Err(error);
    }
    Ok(runtime::low_disk_warning(parent))
}

fn log_disk_warning(context: &str, warning: Option<String>) {
    if let Some(warning) = warning {
        eprintln!("Self-improve {context}: {warning}");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoSelfImproveOutcome {
    SkippedBackoff,
    SkippedLocked,
    PendingProposalExists,
    NoCandidate,
    /// Read-only path (opt-in off): the fusion report was (re)generated and a
    /// candidate is ready for a human `/improve`.
    CandidateReady,
    /// Opt-in on: the preflight ran the generator and parked a gated patch
    /// proposal (`state: proposed`). Applying still needs `/improve apply`.
    ProposalGenerated,
}

/// Generates the candidate's improvement as a unified diff, in isolation.
/// Abstracted so the orchestration is unit-testable with a stub; the production
/// impl ([`ZoSubprocessGenerator`]) runs a headless `zo -p` turn inside a
/// git worktree checked out at HEAD.
pub(crate) trait PatchGenerator {
    /// Implement the improvement described by `prompt` inside `worktree` and
    /// return the resulting `git diff` (empty string ⇒ no change produced).
    fn generate(&self, worktree: &Path, prompt: &str) -> Result<String, String>;
}

/// Production generator: a headless `zo -p` turn in the worktree, full-access
/// and turn-capped, then `git diff` as the proposal.
pub(crate) struct ZoSubprocessGenerator {
    zo_bin: PathBuf,
}

impl ZoSubprocessGenerator {
    pub(crate) fn from_current_exe() -> Result<Self, String> {
        std::env::current_exe()
            .map(|zo_bin| Self { zo_bin })
            .map_err(|error| format!("cannot locate the zo binary: {error}"))
    }
}

fn generation_args(prompt: &str) -> Vec<String> {
    vec![
        "-p".to_string(),
        prompt.to_string(),
        "--permission-mode".to_string(),
        "workspace-write".to_string(),
        "--allowedTools".to_string(),
        "read_file,grep_search,glob_search,edit_file,write_file".to_string(),
        "--max-turns".to_string(),
        GENERATION_MAX_TURNS.to_string(),
    ]
}

const TRUSTED_EXECUTABLE_PATH: &str = "/usr/local/bin:/usr/bin:/bin";
const GENERATION_PASSTHROUGH_ENV: &[&str] = &[
    "ZO_CONFIG_HOME",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_VERTEX_ACCESS_TOKEN",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CLOUD_ML_REGION",
    "CODE_ASSIST_ENDPOINT",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "ZO_AGENT_MODEL",
    "ZO_AGENT_ROUTER_API_KEY",
    "ZO_COMPACTION_MODEL",
    "ZO_CUSTOM_API_KEY",
    "ZO_CUSTOM_OPENAI_API_KEY",
    "ZO_NVIDIA_NIM_API_KEY",
    "GOOGLE_ACCESS_TOKEN",
    "GOOGLE_API_KEY",
    "LOCALAI_API_KEY",
    "MOONSHOT_API_KEY",
    "NVIDIA_API_KEY",
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
];

fn generation_environment(worktree: &Path) -> Result<Vec<(OsString, OsString)>, String> {
    generation_environment_with(worktree, |name| std::env::var_os(name))
}

fn generation_environment_with<F>(
    worktree: &Path,
    mut lookup: F,
) -> Result<Vec<(OsString, OsString)>, String>
where
    F: FnMut(&str) -> Option<OsString>,
{
    let run_dir = worktree
        .parent()
        .ok_or_else(|| "generation worktree has no run directory".to_string())?;
    let home = run_dir.join("home");
    let temp = run_dir.join("tmp");
    if !home.is_dir() || !temp.is_dir() {
        return Err("generation isolation directories are unavailable".to_string());
    }

    let mut environment = vec![
        (OsString::from("HOME"), home.into_os_string()),
        (OsString::from("TMPDIR"), temp.into_os_string()),
        (
            OsString::from("PATH"),
            OsString::from(TRUSTED_EXECUTABLE_PATH),
        ),
    ];
    for name in GENERATION_PASSTHROUGH_ENV {
        if let Some(value) = lookup(name) {
            environment.push((OsString::from(name), value));
        }
    }
    Ok(environment)
}

impl PatchGenerator for ZoSubprocessGenerator {
    fn generate(&self, worktree: &Path, prompt: &str) -> Result<String, String> {
        #[cfg(windows)]
        {
            let _ = (worktree, prompt);
            Err("self-improve generation is unavailable on Windows until descendant cleanup is enforced by a safe Job Object integration".to_string())
        }
        #[cfg(not(windows))]
        {
            log_disk_warning("headless generation", disk_preflight(worktree)?);
            let mut command = Command::new(&self.zo_bin);
            command
                .current_dir(worktree)
                .args(generation_args(prompt))
                .env_clear()
                .envs(generation_environment(worktree)?)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            configure_generation_process_group(&mut command);
            let child = command
                .spawn()
                .map_err(|_| "failed to spawn headless zo (diagnostics redacted)".to_string())?;
            let outcome = run_to_completion(child, GENERATION_TIMEOUT)?;
            if !outcome.status.success() {
                return Err(format!(
                    "headless zo generation failed with {} (diagnostics redacted)",
                    outcome.status
                ));
            }
            capture_worktree_patch(worktree)
        }
    }
}

/// What a bounded, timed child run yielded.
#[derive(Debug)]
struct ChildOutcome {
    status: std::process::ExitStatus,
}

fn configure_generation_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.process_group(0);
    }
}

fn terminate_generation_process_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, killpg};
        use nix::unistd::Pid;

        if let Ok(process_group) = i32::try_from(child.id()) {
            let process_group = Pid::from_raw(process_group);
            let _ = killpg(process_group, Signal::SIGTERM);
            std::thread::sleep(Duration::from_millis(50));
            let _ = killpg(process_group, Signal::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = child;
    }
    let _ = child.kill();
}

/// Drive `child` to completion, draining stdout/stderr into bounded buffers on
/// background threads (so a full pipe cannot deadlock the child) and killing its
/// whole process tree if it outlives `timeout`. The drain threads bound capture
/// at [`GENERATION_OUTPUT_CAP`] bytes each and are joined on every exit path.
fn run_to_completion(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<ChildOutcome, String> {
    let stdout = child.stdout.take().map(drain_capped);
    let stderr = child.stderr.take().map(drain_capped);
    let deadline = Instant::now() + timeout;
    let wait_result = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // The primary process may exit after spawning a descendant that
                // still owns stdout/stderr. Reap the remaining process group
                // before joining drain threads so successful completion cannot
                // bypass the wall-clock bound.
                terminate_generation_process_tree(&mut child);
                break Ok((status, false));
            }
            Ok(None) if Instant::now() >= deadline => {
                terminate_generation_process_tree(&mut child);
                break child
                    .wait()
                    .map(|status| (status, true))
                    .map_err(|_| "reaping timed-out headless zo failed".to_string());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(error) => {
                terminate_generation_process_tree(&mut child);
                let _ = child.wait();
                let _ = error;
                break Err("waiting on headless zo failed".to_string());
            }
        }
    };

    if let Some(handle) = stdout {
        let _ = handle.join();
    }
    if let Some(handle) = stderr {
        let _ = handle.join();
    }
    let (status, timed_out) = wait_result?;
    if timed_out {
        return Err(format!(
            "headless zo generation timed out after {}s and its process tree was killed",
            timeout.as_secs()
        ));
    }
    Ok(ChildOutcome { status })
}

/// Read `reader` to EOF on a background thread, keeping at most
/// [`GENERATION_OUTPUT_CAP`] bytes (the rest is read and discarded so the pipe
/// never blocks the child).
fn drain_capped<R: std::io::Read + Send + 'static>(
    mut reader: R,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut captured = Vec::new();
        let mut buf = [0u8; 8 * 1024];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if captured.len() < GENERATION_OUTPUT_CAP {
                        let room = GENERATION_OUTPUT_CAP - captured.len();
                        captured.extend_from_slice(&buf[..n.min(room)]);
                    }
                }
            }
        }
        String::from_utf8_lossy(&captured).into_owned()
    })
}

/// What `/improve` produced: the fusion verdict, the validated quarantine run,
/// and the gate's blocking reasons (approval still withheld). The diff itself is
/// persisted for `/improve apply`, not carried here — nothing displays it inline.
pub(crate) struct ImproveProposal {
    pub proposal_id: String,
    pub summary: String,
    pub decision: DreamJudgeDecision,
    pub patch_diff: String,
    pub run: QuarantinePatchRun,
    pub blocking_reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
enum ProposalPhase {
    #[default]
    Ready,
    Reviewed,
    Applying,
    PatchApplied,
    CandidateApplied,
}

impl ProposalPhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready_for_review",
            Self::Reviewed => "reviewed",
            Self::Applying => "applying",
            Self::PatchApplied => "patch_applied",
            Self::CandidateApplied => "candidate_applied",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedProposal {
    #[serde(default)]
    phase: ProposalPhase,
    patch_diff: String,
    patch_digest: String,
    approval_digest: String,
    base_commit: String,
    changed_paths: Vec<String>,
    risk: decision_core::dreamer::PatchRisk,
    run: QuarantinePatchRun,
}

/// Run the propose half of `/improve`. `Ok(None)` ⇒ no candidate to act on.
pub(crate) fn propose(
    cwd: &Path,
    generator: &dyn PatchGenerator,
) -> Result<Option<ImproveProposal>, String> {
    let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? else {
        return Err("self-improve is already running for this repository".to_string());
    };
    propose_inner(cwd, generator)
}

/// Lock-free core of [`propose`]. The caller MUST already hold the self-improve
/// lock: `propose` acquires it, and the startup preflight
/// ([`maybe_auto_self_improve_preflight_at`]) already holds it when it opts into
/// auto-proposal — so both routes go through one generator path without a
/// second lock acquisition (which would self-deadlock).
fn propose_inner(
    cwd: &Path,
    generator: &dyn PatchGenerator,
) -> Result<Option<ImproveProposal>, String> {
    // Proposal generation never consumes recovery state. An existing proposal
    // or apply receipt must remain byte-identical until an explicit recovery or
    // apply path resolves it.
    if load_proposal(cwd)?.is_some() {
        return Err(
            "a pending /improve proposal or apply receipt already exists; review it with /improve status and apply or resolve it before generating a new proposal"
                .to_string(),
        );
    }
    let run_id = improve_run_id();
    let Some(report) = run_dream_fusion_v0(cwd, &run_id).map_err(|e| e.to_string())? else {
        return Ok(None);
    };
    if !decision_is_actionable(report.decision) {
        return Ok(None);
    }
    let patch_diff = generate_in_worktree(cwd, &run_id, &report, generator)?;
    if patch_diff.trim().is_empty() {
        return Err("the generation pass produced no changes to propose".to_string());
    }
    let request = QuarantinePatchRequest {
        run_id: run_id.clone(),
        candidate_id: report.candidate_id.clone(),
        patch_diff: patch_diff.clone(),
        allowed_paths: allowed_paths(),
        checks_authorized: false,
        checks: Vec::new(),
        risk: report.risk,
    };
    let run = run_quarantine_patch(cwd, &request).map_err(|e| e.to_string())?;
    let decision = evaluate_manual_apply_gate(
        cwd,
        &ManualApplyGateRequest {
            approved_by_user: false,
            run: run.clone(),
            allowed_paths: allowed_paths(),
            reviewer_accepted: false,
        },
    );
    let proposal_id = persist_proposal(cwd, &patch_diff, &run)?;
    Ok(Some(ImproveProposal {
        proposal_id,
        summary: report.summary,
        decision: report.decision,
        patch_diff,
        run,
        blocking_reasons: decision.reasons,
    }))
}

/// Startup self-improve preflight. `auto_propose` is the opt-in
/// `autoImproveProposalsEnabled` gate: off (default) keeps the historical
/// read-only behavior (fusion report only); on constructs the production
/// [`ZoSubprocessGenerator`] so an actionable candidate is turned into a
/// parked, gated proposal automatically — no manual `/improve` needed. Applying
/// still requires `/improve apply`. Runs on a background thread at startup, so
/// the minutes-long generator never blocks the boot path.
pub(crate) fn maybe_auto_self_improve_preflight(
    cwd: &Path,
    auto_propose: bool,
) -> Result<AutoSelfImproveOutcome, String> {
    // Build the real generator only when opted in; a failure to locate the zo
    // binary degrades to the read-only path rather than failing the preflight.
    let generator = if auto_propose {
        ZoSubprocessGenerator::from_current_exe().ok()
    } else {
        None
    };
    maybe_auto_self_improve_preflight_at(
        cwd,
        SystemTime::now(),
        generator.as_ref().map(|g| g as &dyn PatchGenerator),
    )
}

fn maybe_auto_self_improve_preflight_at(
    cwd: &Path,
    now: SystemTime,
    auto_propose_with: Option<&dyn PatchGenerator>,
) -> Result<AutoSelfImproveOutcome, String> {
    // Do not let an already-applied candidate leave a stale receipt pausing the
    // scheduler. The first readonly check preserves status' no-state behavior.
    if load_proposal_readonly(cwd)?.is_some() {
        let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? else {
            return Ok(AutoSelfImproveOutcome::PendingProposalExists);
        };
        if !reconcile_applied_proposal(cwd)? {
            return Ok(AutoSelfImproveOutcome::PendingProposalExists);
        }
    }

    let schedule = read_self_improve_schedule_state(cwd).map_err(|e| e.to_string())?;
    if !should_run_self_improve(
        schedule.last_attempt,
        schedule.last_failure,
        now,
        AUTO_SELF_IMPROVE_MIN_INTERVAL,
        AUTO_SELF_IMPROVE_FAILURE_BACKOFF,
    ) {
        return Ok(AutoSelfImproveOutcome::SkippedBackoff);
    }

    let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? else {
        return Ok(AutoSelfImproveOutcome::SkippedLocked);
    };
    if load_proposal(cwd)?.is_some() && !reconcile_applied_proposal(cwd)? {
        return Ok(AutoSelfImproveOutcome::PendingProposalExists);
    }

    record_self_improve_attempt(cwd).map_err(|e| e.to_string())?;
    let candidates = runtime::memory::read_self_improve_candidates(cwd);
    if !candidates
        .iter()
        .any(|candidate| candidate.kind.is_actionable() && !candidate.status.is_terminal())
    {
        return Ok(AutoSelfImproveOutcome::NoCandidate);
    }

    if let Some(generator) = auto_propose_with {
        // Opt-in path: run the generator and park a gated proposal. We already
        // hold the self-improve lock (acquired above), so call the lock-free
        // core directly — `propose` itself would try to re-acquire and fail.
        // Applying is still a separate, human `/improve apply`.
        return match propose_inner(cwd, generator)? {
            Some(_) => Ok(AutoSelfImproveOutcome::ProposalGenerated),
            // The advisors declined this candidate mid-generation (not
            // actionable / no diff); nothing was parked.
            None => Ok(AutoSelfImproveOutcome::NoCandidate),
        };
    }

    // Read-only path (opt-in off): generate the DreamFusion report only — pure
    // heuristics over the top-ranked candidate, a JSON file under
    // `.zo/dream/fusion/`; no patch is generated and nothing is applied.
    // Without this, the automatic loop ended at a bare backoff timestamp and the
    // operator had to arm `/improve` blind; now `/improve status` opens on a
    // concrete "this is what /improve would act on". A failure records the
    // normal failure backoff via the caller.
    run_dream_fusion_v0(cwd, &improve_run_id()).map_err(|e| e.to_string())?;
    Ok(AutoSelfImproveOutcome::CandidateReady)
}

pub(crate) fn record_auto_self_improve_failure(cwd: &Path, error: &str) {
    let error = runtime::memory::DreamError::Io(std::io::Error::other(error.to_string()));
    let _ = runtime::memory::record_self_improve_failure(cwd, &error);
}

/// Phase-aware "what next" line for `/improve status`.
fn improve_next_action(
    active_proposal: Option<&PersistedProposal>,
    has_candidates: bool,
    automation_enabled: bool,
) -> String {
    active_proposal.map_or_else(
        || {
            if has_candidates {
                "run /improve to generate a proposal".to_string()
            } else if automation_enabled {
                "wait for a verified self-improve candidate".to_string()
            } else {
                "enable autoDreamEnabled or run /improve manually".to_string()
            }
        },
        |proposal| match proposal.phase {
            ProposalPhase::Ready => format!(
                "review or reject proposal {} before apply",
                proposal.approval_digest
            ),
            ProposalPhase::Reviewed => format!("apply proposal {}", proposal.approval_digest),
            _ => "complete or recover the pending apply receipt".to_string(),
        },
    )
}

pub(crate) fn status_report(cwd: &Path, automation_enabled: bool) -> Result<String, String> {
    let now = SystemTime::now();
    let schedule = runtime::memory::read_self_improve_schedule_state_readonly(cwd)
        .map_err(|error| error.to_string())?;
    let mut active_proposal = load_proposal_readonly(cwd)?;
    if active_proposal.is_some() {
        if let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? {
            let _ = reconcile_applied_proposal(cwd)?;
            active_proposal = load_proposal_readonly(cwd)?;
        }
    }
    let pending_proposal = active_proposal.is_some();
    let candidates = runtime::memory::read_self_improve_candidates(cwd);
    let active_candidates: Vec<_> = candidates
        .iter()
        .filter(|candidate| candidate.kind.is_actionable() && !candidate.status.is_terminal())
        .collect();
    let candidate_line = active_candidates.first().map_or_else(
        || "none".to_string(),
        |candidate| {
            format!(
                "ready ({} active; top {} · {})",
                active_candidates.len(),
                candidate.kind.as_str(),
                candidate.status.as_str()
            )
        },
    );
    // Per-signature breakdown of the actionable backlog. Candidates are
    // segmented by error signature at record time, so each row is one concrete
    // failure mode with its own evidence volume and recency — not one opaque
    // "turn failed" aggregate.
    let mut candidate_rows = String::new();
    for (index, candidate) in active_candidates.iter().take(3).enumerate() {
        let _ = write!(
            candidate_rows,
            "\n    {}. {} · {} · {} evidence · last {}",
            index + 1,
            candidate.kind.as_str(),
            sanitize_status_text(&candidate.summary),
            candidate.evidence.len(),
            format_epoch_ms(candidate.last_observed_at_ms, now),
        );
    }
    // The read-only fusion report the startup preflight (or a prior /improve)
    // persisted — what /improve would act on right now.
    let fusion_line = runtime::memory::latest_dream_fusion_report(cwd).map_or_else(
        || "none".to_string(),
        |(report, modified)| {
            format!(
                "{} — {} ({})",
                report.decision.as_str(),
                sanitize_status_text(&report.summary),
                format_time_delta(modified, now)
            )
        },
    );
    let scheduler_ready = automation_enabled
        && should_run_self_improve(
            schedule.last_attempt,
            schedule.last_failure,
            now,
            AUTO_SELF_IMPROVE_MIN_INTERVAL,
            AUTO_SELF_IMPROVE_FAILURE_BACKOFF,
        );
    let scheduler = if !automation_enabled {
        "disabled"
    } else if pending_proposal {
        "paused (pending proposal)"
    } else if scheduler_ready {
        "ready"
    } else {
        "backoff"
    };
    let proposal_id = active_proposal
        .as_ref()
        .map_or("none", |proposal| proposal.approval_digest.as_str());
    let proposal_phase = active_proposal
        .as_ref()
        .map_or("none", |proposal| proposal.phase.as_str());
    let next_action = improve_next_action(
        active_proposal.as_ref(),
        !active_candidates.is_empty(),
        automation_enabled,
    );

    Ok(format!(
        "Self-improve status\n  Automation      {}\n  Scheduler       {}\n  Last attempt    {}\n  Last failure    {}\n  Pending proposal {}\n  Proposal ID     {}\n  Proposal phase  {}\n  Candidate        {}{}\n  Fusion report    {}\n  Next action      {}",
        if automation_enabled { "enabled" } else { "disabled" },
        scheduler,
        format_optional_time(schedule.last_attempt, now),
        format_optional_time(schedule.last_failure, now),
        yes_no(pending_proposal),
        proposal_id,
        proposal_phase,
        candidate_line,
        candidate_rows,
        fusion_line,
        next_action
    ))
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn format_optional_time(value: Option<SystemTime>, now: SystemTime) -> String {
    value.map_or_else(|| "never".to_string(), |time| format_time_delta(time, now))
}

/// Format a candidate-store epoch-millisecond stamp as a relative age.
/// Zero marks a record predating timestamped aggregation.
fn format_epoch_ms(ms: u64, now: SystemTime) -> String {
    if ms == 0 {
        return "unknown".to_string();
    }
    format_time_delta(
        std::time::UNIX_EPOCH + Duration::from_millis(ms),
        now,
    )
}

fn format_time_delta(time: SystemTime, now: SystemTime) -> String {
    match now.duration_since(time) {
        Ok(delta) => format!("{} ago", format_duration(delta)),
        Err(_) => "in the future".to_string(),
    }
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 60 * 60 {
        format!("{}m", secs / 60)
    } else if secs < 24 * 60 * 60 {
        format!("{}h", secs / (60 * 60))
    } else {
        format!("{}d", secs / (24 * 60 * 60))
    }
}

/// Whether a fusion verdict warrants generating a patch. `Reject` and
/// `NeedMoreEvidence` are not actionable — generating on them wastes a headless
/// `zo -p` turn on a candidate the advisors did not endorse.
const fn decision_is_actionable(decision: DreamJudgeDecision) -> bool {
    matches!(
        decision,
        DreamJudgeDecision::PlanPatch | DreamJudgeDecision::Quarantine
    )
}

fn record_candidate_and_clear_receipt(
    cwd: &Path,
    proposal: &mut PersistedProposal,
    candidate_id: &str,
    expected_diff: &str,
) -> Result<(), String> {
    mark_self_improve_candidate_applied(cwd, candidate_id).map_err(|error| {
        format!(
            "patch applied but candidate completion was not recorded; proposal was retained for recovery: {error}"
        )
    })?;
    proposal.phase = ProposalPhase::CandidateApplied;
    write_persisted_proposal(cwd, proposal)?;
    validate_applied_state(cwd, proposal, expected_diff).map_err(|error| {
        format!(
            "candidate completion was recorded but the repository changed before receipt cleanup; proposal was retained for recovery: {error}"
        )
    })?;
    clear_proposal(cwd)
        .map_err(|error| format!("patch applied but proposal cleanup failed: {error}"))
}

fn select_proposal(cwd: &Path, proposal_id: &str) -> Result<PersistedProposal, String> {
    let proposal = load_proposal(cwd)?
        .ok_or_else(|| "no pending /improve proposal — run /improve first".to_string())?;
    if proposal_id != proposal.approval_digest {
        return Err("stale or unknown self-improve proposal ID".to_string());
    }
    validate_receipt_binding(&proposal)?;
    Ok(proposal)
}

/// Show the complete persisted proposal selected by its stable ID.
pub(crate) fn show(cwd: &Path, proposal_id: &str) -> Result<String, String> {
    let proposal = select_proposal(cwd, proposal_id)?;
    let mut out = format!(
        "Self-improve proposal\n  Proposal ID     {}\n  State           {}\n  Candidate       {}\n  Risk            {}\n  Changed files   {}",
        proposal.approval_digest,
        proposal.phase.as_str(),
        escape_terminal(&proposal.run.candidate_id),
        proposal.risk,
        proposal.changed_paths.len(),
    );
    for path in &proposal.changed_paths {
        let _ = write!(out, "\n    {}", escape_terminal(path));
    }
    let _ = write!(
        out,
        "\n  Patch\n{}",
        escape_terminal_multiline(&proposal.patch_diff)
    );
    Ok(out)
}

/// Persist explicit reviewer acceptance for an exact proposal ID. Review does
/// not run checks or mutate the repository; apply revalidates every safety gate.
pub(crate) fn review(cwd: &Path, proposal_id: &str) -> Result<String, String> {
    let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? else {
        return Err("self-improve is already running for this repository".to_string());
    };
    let mut proposal = select_proposal(cwd, proposal_id)?;
    match proposal.phase {
        ProposalPhase::Ready => {
            proposal.phase = ProposalPhase::Reviewed;
            write_persisted_proposal(cwd, &proposal)?;
        }
        ProposalPhase::Reviewed => {}
        _ => return Err("proposal is no longer available for review".to_string()),
    }
    Ok(format!("Self-improve proposal {proposal_id} reviewed and accepted."))
}

/// Reject an exact proposal and terminally retire its candidate before removing
/// the active receipt, preventing immediate regeneration of the declined patch.
pub(crate) fn reject(cwd: &Path, proposal_id: &str) -> Result<String, String> {
    let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? else {
        return Err("self-improve is already running for this repository".to_string());
    };
    let proposal = select_proposal(cwd, proposal_id)?;
    if !matches!(proposal.phase, ProposalPhase::Ready | ProposalPhase::Reviewed) {
        return Err("proposal is no longer available for rejection".to_string());
    }
    mark_self_improve_candidate_rejected(cwd, &proposal.run.candidate_id)
        .map_err(|error| error.to_string())?;
    clear_proposal(cwd)?;
    Ok(format!("Self-improve proposal {proposal_id} rejected."))
}

/// Apply the selected `/improve` proposal — only when a separate review has
/// accepted it and the manual apply gate is satisfied with explicit approval.
/// Returns the human-readable outcome.
pub(crate) fn apply(cwd: &Path, approved_digest: &str) -> Result<String, String> {
    let Some(_lock) = try_acquire_self_improve_lock(cwd).map_err(|e| e.to_string())? else {
        return Err("self-improve is already running for this repository".to_string());
    };
    let _ = reconcile_applied_proposal(cwd)?;
    let mut proposal = select_proposal(cwd, approved_digest)?;
    if proposal.phase != ProposalPhase::Reviewed {
        return Err(format!(
            "proposal has not been reviewer-accepted; review proposal {approved_digest} before applying"
        ));
    }
    let Some(recomputed) =
        run_dream_fusion_v0(cwd, &improve_run_id()).map_err(|e| e.to_string())?
    else {
        return Err("proposal candidate is no longer actionable".to_string());
    };
    if recomputed.candidate_id != proposal.run.candidate_id || recomputed.risk != proposal.risk {
        return Err(
            "persisted proposal risk or candidate no longer matches the current repository state"
                .to_string(),
        );
    }
    let verified_run = run_quarantine_patch(
        cwd,
        &QuarantinePatchRequest {
            run_id: improve_run_id(),
            candidate_id: proposal.run.candidate_id.clone(),
            patch_diff: proposal.patch_diff.clone(),
            allowed_paths: allowed_paths(),
            checks_authorized: true,
            checks: default_checks(cwd),
            risk: proposal.risk,
        },
    )
    .map_err(|e| e.to_string())?;
    if verified_run.base_commit != proposal.base_commit
        || verified_run.patch_digest != proposal.patch_digest
        || verified_run.changed_paths != proposal.changed_paths
        || verified_run.risk != proposal.risk
        || approval_digest_for_run(&verified_run) != proposal.approval_digest
    {
        return Err("proposal changed during quarantine revalidation".to_string());
    }
    let decision = evaluate_manual_apply_gate(
        cwd,
        &ManualApplyGateRequest {
            approved_by_user: true,
            run: verified_run.clone(),
            allowed_paths: allowed_paths(),
            reviewer_accepted: proposal.phase == ProposalPhase::Reviewed,
        },
    );
    if !decision.eligible {
        let failed_checks = verified_run
            .check_results
            .iter()
            .filter(|check| !check.success)
            .map(|check| {
                format!(
                    "{}:exit={:?}:{}",
                    escape_terminal(&check.name),
                    check.exit_code,
                    escape_terminal(&check.stderr)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "apply gate blocked the patch: {}{}",
            decision.reasons.join(", "),
            if failed_checks.is_empty() {
                String::new()
            } else {
                format!("; checks: {failed_checks}")
            }
        ));
    }
    let expected_diff = quarantine_verified_diff(cwd, &verified_run.run_id)?;
    final_apply_revalidation(cwd, &proposal, approved_digest)?;
    proposal.phase = ProposalPhase::Applying;
    write_persisted_proposal(cwd, &proposal)?;
    if let Err(error) = apply_patch_and_validate(cwd, &proposal.patch_diff, || {
        validate_applied_state(cwd, &proposal, &expected_diff)
    }) {
        proposal.phase = ProposalPhase::Reviewed;
        let _ = write_persisted_proposal(cwd, &proposal);
        return Err(error);
    }
    proposal.phase = ProposalPhase::PatchApplied;
    write_persisted_proposal(cwd, &proposal)?;
    record_candidate_and_clear_receipt(
        cwd,
        &mut proposal,
        &verified_run.candidate_id,
        &expected_diff,
    )?;
    let _ = record_self_improve_attempt(cwd);
    Ok(format!(
        "Applied self-improve patch — {} file(s) changed.",
        verified_run.changed_paths.len()
    ))
}

/// Render a proposal for the REPL: fusion verdict, changed files, check results,
/// and what `/improve apply` still needs.
#[must_use]
pub(crate) fn format_proposal(proposal: &ImproveProposal) -> String {
    let mut out = format!(
        "Self-improve proposal\n  Proposal ID     {}\n  Candidate        {}\n  Fusion           {}\n  Risk             {}\n  Changed files    {}",
        proposal.proposal_id,
        escape_terminal(&proposal.summary),
        proposal.decision.as_str(),
        proposal.run.risk,
        proposal.run.changed_paths.len(),
    );
    for path in &proposal.run.changed_paths {
        let _ = write!(out, "\n    {}", escape_terminal(path));
    }
    for check in &proposal.run.check_results {
        let _ = write!(
            out,
            "\n  Check            {} — {}{}",
            escape_terminal(&check.name),
            if check.success { "PASS" } else { "FAIL" },
            if check.stderr.is_empty() {
                String::new()
            } else {
                format!(" ({})", escape_terminal(&check.stderr))
            },
        );
    }
    let _ = write!(
        out,
        "\n  Patch\n{}\n  Review           required for proposal {} before apply",
        escape_terminal_multiline(&proposal.patch_diff),
        proposal.proposal_id
    );
    if !proposal.blocking_reasons.is_empty() {
        let reasons = proposal
            .blocking_reasons
            .iter()
            .map(|reason| escape_terminal(reason))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(out, " ({reasons})");
    }
    out
}

pub(crate) fn escape_terminal(value: &str) -> String {
    value.chars().flat_map(char::escape_default).collect()
}

/// Neutralize control characters for the status report while keeping normal
/// Unicode readable. `escape_terminal` (`escape_default`) would mangle the
/// signature separator `·` into `\u{b7}`; candidate summaries are built from
/// bounded classifier output, so stripping controls is the only hardening the
/// status lines need.
fn sanitize_status_text(value: &str) -> String {
    value
        .chars()
        .map(|character| if character.is_control() { ' ' } else { character })
        .collect()
}

fn escape_terminal_multiline(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| {
            if character == '\n' {
                "\n".chars().collect::<Vec<_>>()
            } else {
                character.escape_default().collect::<Vec<_>>()
            }
        })
        .collect()
}

fn patch_digest(patch: &str) -> String {
    format!("{:x}", Sha256::digest(patch.as_bytes()))
}

fn approval_digest_for_run(run: &QuarantinePatchRun) -> String {
    let mut hasher = Sha256::new();
    for field in [
        APPROVAL_ENVELOPE_VERSION,
        run.patch_digest.as_str(),
        run.base_commit.as_str(),
        run.candidate_id.as_str(),
        run.risk.as_str(),
        CHECK_POLICY_VERSION,
    ] {
        hasher.update(field.len().to_be_bytes());
        hasher.update(field.as_bytes());
    }
    for path in &run.changed_paths {
        hasher.update(path.len().to_be_bytes());
        hasher.update(path.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn validate_persisted_proposal(
    proposal: &PersistedProposal,
    approved_digest: &str,
) -> Result<(), String> {
    if approved_digest != proposal.approval_digest {
        return Err("approval digest does not match the displayed proposal".to_string());
    }
    validate_receipt_binding(proposal)
}

/// Internal-consistency checks of a persisted proposal/receipt: the patch
/// digest, base commit, changed paths, risk, candidate binding, and approval
/// digest must all agree with each other before any automatic action.
fn validate_receipt_binding(proposal: &PersistedProposal) -> Result<(), String> {
    if proposal.run.candidate_id.trim().is_empty() {
        return Err("persisted proposal binding is invalid".to_string());
    }
    if patch_digest(&proposal.patch_diff) != proposal.patch_digest
        || proposal.run.patch_digest != proposal.patch_digest
        || proposal.run.base_commit != proposal.base_commit
        || proposal.run.changed_paths != proposal.changed_paths
        || proposal.run.risk != proposal.risk
        || approval_digest_for_run(&proposal.run) != proposal.approval_digest
    {
        return Err("persisted proposal binding is invalid".to_string());
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ApplyState {
    head: String,
    status: Vec<u8>,
    staged_diff: Vec<u8>,
    worktree_diff: Vec<u8>,
}

fn capture_apply_state(cwd: &Path) -> Result<ApplyState, String> {
    Ok(ApplyState {
        head: git_head(cwd)?,
        status: git_diff(cwd, &["status", "--porcelain", "-z"] )?,
        staged_diff: git_diff(cwd, &["diff", "--cached", "--binary", "HEAD"] )?,
        worktree_diff: git_diff(cwd, &["diff", "--binary"] )?,
    })
}

fn trusted_git_executable(cwd: &Path) -> Result<PathBuf, String> {
    runtime::memory::trusted_git_binary(cwd)
        .map_err(|_| "trusted Git executable is unavailable".to_string())
}

fn git_command(cwd: &Path) -> Result<Command, String> {
    let mut command = Command::new(trusted_git_executable(cwd)?);
    command.arg("-C").arg(cwd);
    Ok(command)
}

fn rollback_self_improve_patch(cwd: &Path, patch_diff: &str) -> Result<(), String> {
    git_apply_bytes_with_mode(cwd, patch_diff.as_bytes(), true, true).map_err(|error| {
        format!(
            "approved patch could not be safely reversed; no files were reset or overwritten: {error}"
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestoreApplyOutcome {
    ExactPreApplyState,
    ExternalEditsPreserved,
}

fn restore_apply_state(
    cwd: &Path,
    state: &ApplyState,
    patch_diff: &str,
) -> Result<RestoreApplyOutcome, String> {
    rollback_self_improve_patch(cwd, patch_diff)?;
    let restored = capture_apply_state(cwd)?;
    if restored == *state {
        Ok(RestoreApplyOutcome::ExactPreApplyState)
    } else {
        Ok(RestoreApplyOutcome::ExternalEditsPreserved)
    }
}

fn apply_patch_and_validate<F>(cwd: &Path, patch_diff: &str, validate: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String>,
{
    let before = capture_apply_state(cwd)?;
    git_apply(cwd, patch_diff)?;
    if let Err(error) = validate() {
        return match restore_apply_state(cwd, &before, patch_diff) {
            Ok(RestoreApplyOutcome::ExactPreApplyState) => {
                Err(format!("{error}; repository state was restored"))
            }
            Ok(RestoreApplyOutcome::ExternalEditsPreserved) => Err(format!(
                "{error}; approved patch was reversed and external edits were preserved"
            )),
            Err(restore_error) => Err(format!(
                "{error}; rollback was refused to preserve external edits; manual recovery is required and the proposal was retained: {restore_error}"
            )),
        };
    }
    Ok(())
}

fn staged_changed_paths(cwd: &Path) -> Result<Vec<String>, String> {
    let output = git_command(cwd)?
        .args(["diff", "--cached", "--name-only", "--no-renames", "-z", "HEAD"])
        .output()
        .map_err(|_| "git diff --cached failed".to_string())?;
    if !output.status.success() {
        return Err("git diff --cached failed".to_string());
    }
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| {
            String::from_utf8(path.to_vec()).map_err(|_| "git path was not UTF-8".to_string())
        })
        .collect()
}

fn git_diff(cwd: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = git_command(cwd)?
        .args(args)
        .output()
        .map_err(|_| "git verification command failed".to_string())?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err("git verification command failed".to_string())
    }
}

fn final_apply_revalidation(
    cwd: &Path,
    proposal: &PersistedProposal,
    approved_digest: &str,
) -> Result<(), String> {
    if git_head(cwd)? != proposal.base_commit {
        return Err("repository HEAD changed after proposal review".to_string());
    }
    let status = git_diff(cwd, &["status", "--porcelain", "-z"])?;
    if !status.is_empty() {
        return Err("repository worktree is no longer clean".to_string());
    }
    if proposal
        .changed_paths
        .iter()
        .any(|path| path_has_symlink(cwd, path))
    {
        return Err("approved path has a symlinked ancestor".to_string());
    }
    validate_persisted_proposal(proposal, approved_digest)
}

fn quarantine_verified_diff(cwd: &Path, run_id: &str) -> Result<String, String> {
    let relative = PathBuf::from(".zo/dream/quarantine")
        .join(run_id)
        .join("patch.diff");
    runtime::secure_fs::read_to_string_no_symlink(cwd, &relative)
        .map_err(|_| "verified quarantine diff is unavailable".to_string())
}

fn validate_applied_state(
    cwd: &Path,
    proposal: &PersistedProposal,
    expected_diff: &str,
) -> Result<(), String> {
    if git_head(cwd)? != proposal.base_commit
        || staged_changed_paths(cwd)? != proposal.changed_paths
        || git_diff(cwd, &["diff", "--cached", "--binary", "HEAD"])?
            != expected_diff.as_bytes()
        || !git_diff(cwd, &["diff", "--binary"])?.is_empty()
        || proposal
            .changed_paths
            .iter()
            .any(|path| path_has_symlink(cwd, path))
    {
        return Err("applied patch did not match the approved postcondition".to_string());
    }
    Ok(())
}

fn path_has_symlink(cwd: &Path, path: &str) -> bool {
    let mut current = cwd.to_path_buf();
    for component in Path::new(path).components() {
        let std::path::Component::Normal(component) = component else {
            return true;
        };
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => return true,
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(_) => return true,
        }
    }
    false
}

fn finish_generation(
    generation: Result<String, String>,
    cleanup: Result<(), String>,
) -> Result<String, String> {
    match (generation, cleanup) {
        (Ok(patch), Ok(())) => Ok(patch),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(generation_error), Err(cleanup_error)) => Err(format!(
            "{generation_error}; generation worktree cleanup also failed: {cleanup_error}"
        )),
    }
}

fn generate_in_worktree(
    cwd: &Path,
    run_id: &str,
    report: &DreamFusionReport,
    generator: &dyn PatchGenerator,
) -> Result<String, String> {
    let worktree = add_worktree(cwd, run_id)?;
    let generation = generator.generate(&worktree, &improvement_prompt(report));
    let cleanup = remove_worktree(cwd, &worktree);
    finish_generation(generation, cleanup)
}

fn improvement_prompt(report: &DreamFusionReport) -> String {
    let findings = report
        .findings
        .iter()
        .map(|finding| {
            format!(
                "- [{}] {} (risk {}; checks: {})",
                finding.role.as_str(),
                finding.summary,
                finding.risk,
                finding.recommended_checks.join("; ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "[zo:self-improve] You are in an isolated git worktree. Implement the SMALLEST correct \
         change that resolves the self-improvement candidate below; make surgical edits to source \
         only and ensure it still builds. Do not commit.\n\n\
         Candidate: {summary}\n\nAdvisor findings:\n{findings}\n",
        summary = report.summary,
    )
}

fn default_checks(cwd: &Path) -> Vec<QuarantineCheckCommand> {
    // The focused check that gates the patch. A cargo workspace gets `cargo
    // check`; a non-cargo repo gets no check, so the apply gate stays blocked
    // (it never applies a patch it could not verify).
    if cwd.join("Cargo.toml").exists() {
        vec![QuarantineCheckCommand {
            name: "cargo check".to_string(),
            program: "cargo".to_string(),
            args: vec![
                "check".to_string(),
                "--workspace".to_string(),
                "--all-targets".to_string(),
                "--locked".to_string(),
                "--quiet".to_string(),
            ],
        }]
    } else {
        Vec::new()
    }
}

fn allowed_paths() -> Vec<String> {
    ALLOWED_PATH_ROOTS
        .iter()
        .map(|root| (*root).to_string())
        .collect()
}

fn improve_run_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("improve-{}-{nanos}", std::process::id())
}

const STALE_GENERATION_RUN_RETENTION: usize = 3;

fn generation_run_relative(run_id: &str) -> Result<PathBuf, String> {
    let relative = PathBuf::from(IMPROVE_RELATIVE_DIR).join(run_id);
    let mut components = Path::new(run_id).components();
    if !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err("generation run identifier is invalid".to_string());
    }
    Ok(relative)
}

fn generation_run_dir(cwd: &Path, worktree: &Path) -> Result<PathBuf, String> {
    if worktree.file_name().and_then(|name| name.to_str()) != Some("gen") {
        return Err("generation worktree path is invalid".to_string());
    }
    let run_dir = worktree
        .parent()
        .ok_or_else(|| "generation worktree has no run directory".to_string())?;
    let improve = improve_dir(cwd)?;
    let relative = run_dir
        .strip_prefix(&improve)
        .map_err(|_| "generation worktree is outside the private run directory".to_string())?;
    let mut components = relative.components();
    if !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err("generation worktree is outside the private run directory".to_string());
    }
    Ok(run_dir.to_path_buf())
}

fn remove_generation_run_dir(cwd: &Path, worktree: &Path) -> Result<(), String> {
    let run_dir = generation_run_dir(cwd, worktree)?;
    match std::fs::symlink_metadata(&run_dir) {
        Ok(metadata) if metadata.file_type().is_dir() => std::fs::remove_dir_all(&run_dir)
            .map_err(|_| "generation worktree run directory cleanup failed".to_string()),
        Ok(_) => Err("generation worktree run directory is not a real directory".to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("generation worktree run directory cleanup failed".to_string()),
    }
}

fn cleanup_stale_runs(cwd: &Path, current_run_id: &str) -> Result<(), String> {
    let improve = improve_dir(cwd)?;
    let entries = std::fs::read_dir(&improve)
        .map_err(|_| "listing stale generation runs failed (diagnostics redacted)".to_string())?;
    let mut stale_runs = Vec::new();
    let mut errors = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            errors.push("listing a stale generation run failed (diagnostics redacted)".to_string());
            continue;
        };
        if entry.file_name() == current_run_id || entry.file_name() == "last-proposal.json" {
            continue;
        }
        match std::fs::symlink_metadata(entry.path()) {
            Ok(metadata) if metadata.file_type().is_dir() => stale_runs.push((
                metadata.modified().unwrap_or(std::time::UNIX_EPOCH),
                entry.path(),
            )),
            Ok(_) => {}
            Err(_) => errors.push("inspecting a stale generation run failed (diagnostics redacted)".to_string()),
        }
    }
    stale_runs.sort_by(|left, right| right.0.cmp(&left.0));
    for (_, run_dir) in stale_runs.into_iter().skip(STALE_GENERATION_RUN_RETENTION) {
        let worktree = run_dir.join("gen");
        let cleanup = match std::fs::symlink_metadata(&worktree) {
            Ok(metadata) if metadata.file_type().is_dir() => remove_worktree(cwd, &worktree),
            Ok(_) => Err("stale generation worktree is not a real directory".to_string()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                remove_generation_run_dir(cwd, &worktree)
            }
            Err(_) => Err("inspecting a stale generation worktree failed (diagnostics redacted)".to_string()),
        };
        if let Err(error) = cleanup {
            errors.push(error);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "stale generation run cleanup failed: {}",
            errors.join("; ")
        ))
    }
}

fn add_worktree(cwd: &Path, run_id: &str) -> Result<PathBuf, String> {
    cleanup_stale_runs(cwd, run_id)?;
    let run_relative = generation_run_relative(run_id)?;
    let run_dir = runtime::secure_fs::ensure_private_dir(cwd, &run_relative)
        .map_err(|_| "generation run directory setup failed (diagnostics redacted)".to_string())?;
    for subdir in ["home", "tmp"] {
        runtime::secure_fs::ensure_private_dir(cwd, &run_relative.join(subdir)).map_err(|_| {
            "generation isolation directory setup failed (diagnostics redacted)".to_string()
        })?;
    }
    let path = run_dir.join("gen");
    let head = git_head(cwd)?;
    log_disk_warning("worktree creation", disk_preflight(&run_dir)?);
    let output = git_command(cwd)?
        .args(["worktree", "add", "--detach"])
        .arg(&path)
        .arg(&head)
        .output()
        .map_err(|_| "git worktree add failed (diagnostics redacted)".to_string())?;
    if !output.status.success() {
        let cleanup = remove_worktree(cwd, &path);
        return match cleanup {
            Ok(()) => Err("git worktree add failed (diagnostics redacted)".to_string()),
            Err(error) => Err(format!(
                "git worktree add failed (diagnostics redacted); setup cleanup also failed: {error}"
            )),
        };
    }
    Ok(path)
}

fn remove_worktree(cwd: &Path, path: &Path) -> Result<(), String> {
    let mut errors = Vec::new();
    match trusted_git_executable(cwd) {
        Ok(git) => match Command::new(git)
            .arg("-C")
            .arg(cwd)
            .args(["worktree", "remove", "--force"])
            .arg(path)
            .output()
        {
            Ok(output) if output.status.success() => {}
            Ok(_) | Err(_) => {
                errors.push("git worktree remove failed (diagnostics redacted)".to_string());
            }
        },
        Err(error) => errors.push(error),
    }
    if let Err(error) = remove_generation_run_dir(cwd, path) {
        errors.push(error);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn git_head(cwd: &Path) -> Result<String, String> {
    let output = git_command(cwd)?
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| format!("git rev-parse failed: {e}"))?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed (no commits yet?)".to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn capture_worktree_patch(worktree: &Path) -> Result<String, String> {
    let intent = git_command(worktree)?
        .args(["add", "--intent-to-add", "--", "."])
        .output()
        .map_err(|_| "capturing generated files failed (diagnostics redacted)".to_string())?;
    if !intent.status.success() {
        return Err("capturing generated files failed (diagnostics redacted)".to_string());
    }

    let output = git_command(worktree)?
        .args(["diff", "--binary", "HEAD"])
        .output()
        .map_err(|_| "capturing generated patch failed (diagnostics redacted)".to_string())?;
    if !output.status.success() {
        return Err("capturing generated patch failed (diagnostics redacted)".to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_apply(cwd: &Path, patch_diff: &str) -> Result<(), String> {
    git_apply_bytes(cwd, patch_diff.as_bytes(), true)
}

fn git_apply_bytes(cwd: &Path, patch: &[u8], update_index: bool) -> Result<(), String> {
    git_apply_bytes_with_mode(cwd, patch, update_index, false)
}

fn git_apply_bytes_with_mode(
    cwd: &Path,
    patch: &[u8],
    update_index: bool,
    reverse: bool,
) -> Result<(), String> {
    let mut args = vec!["apply", "--whitespace=nowarn"];
    if update_index {
        args.push("--index");
    }
    if reverse {
        args.push("--reverse");
    }
    args.push("-");
    let mut child = git_command(cwd)?
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| "git apply failed (diagnostics redacted)".to_string())?;
    let write_result = child
        .stdin
        .take()
        .ok_or_else(|| "git apply stdin was unavailable".to_string())
        .and_then(|mut stdin| {
            let result = stdin
                .write_all(patch)
                .map_err(|_| "writing patch to git apply failed".to_string());
            drop(stdin);
            result
        });
    if let Err(error) = write_result {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    let status = child
        .wait()
        .map_err(|_| "git apply wait failed (diagnostics redacted)".to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err("git apply failed (diagnostics redacted)".to_string())
    }
}

const IMPROVE_RELATIVE_DIR: &str = ".zo/dream/improve";
const PROPOSAL_RELATIVE_PATH: &str = ".zo/dream/improve/last-proposal.json";

fn improve_dir(cwd: &Path) -> Result<PathBuf, String> {
    runtime::secure_fs::ensure_private_dir(cwd, Path::new(IMPROVE_RELATIVE_DIR))
        .map_err(|error| error.to_string())
}

fn write_proposal_atomic(cwd: &Path, bytes: &[u8]) -> Result<(), String> {
    improve_dir(cwd)?;
    runtime::secure_fs::write_atomic_owner_only(
        cwd,
        Path::new(PROPOSAL_RELATIVE_PATH),
        bytes,
    )
    .map_err(|error| error.to_string())
}

fn persist_proposal(
    cwd: &Path,
    patch_diff: &str,
    run: &QuarantinePatchRun,
) -> Result<String, String> {
    let approval_digest = approval_digest_for_run(run);
    let payload = PersistedProposal {
        phase: ProposalPhase::Ready,
        patch_diff: patch_diff.to_string(),
        patch_digest: run.patch_digest.clone(),
        approval_digest: approval_digest.clone(),
        base_commit: run.base_commit.clone(),
        changed_paths: run.changed_paths.clone(),
        risk: run.risk,
        run: run.clone(),
    };
    if patch_digest(&payload.patch_diff) != payload.patch_digest {
        return Err("quarantine returned a mismatched patch digest".to_string());
    }
    let json = serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?;
    write_proposal_atomic(cwd, &json)?;
    Ok(approval_digest)
}

fn write_persisted_proposal(cwd: &Path, proposal: &PersistedProposal) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(proposal).map_err(|error| error.to_string())?;
    write_proposal_atomic(cwd, &json)
}

fn read_proposal(cwd: &Path) -> Result<Option<PersistedProposal>, String> {
    let contents = match runtime::secure_fs::read_to_string_no_symlink(
        cwd,
        Path::new(PROPOSAL_RELATIVE_PATH),
    ) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => {
            return Err("proposal state path is not a real directory or regular file".to_string());
        }
    };
    serde_json::from_str(&contents)
        .map(Some)
        .map_err(|error| error.to_string())
}

fn load_proposal_readonly(cwd: &Path) -> Result<Option<PersistedProposal>, String> {
    read_proposal(cwd)
}

fn load_proposal(cwd: &Path) -> Result<Option<PersistedProposal>, String> {
    read_proposal(cwd)
}

fn clear_proposal(cwd: &Path) -> Result<(), String> {
    runtime::secure_fs::remove_file_no_symlink(cwd, Path::new(PROPOSAL_RELATIVE_PATH))
        .map_err(|error| error.to_string())
}

/// Complete an interrupted post-apply sequence. The persisted receipt moves
/// through `Applying` → `PatchApplied` → `CandidateApplied`; cleanup occurs
/// only after the receipt binding and the actual repository state validate
/// and the matching candidate is durably terminal. A malformed, tampered,
/// stale, or ambiguous receipt is never auto-deleted and never terminally
/// marks a candidate: it stays pending with a recovery error.
fn reconcile_applied_proposal(cwd: &Path) -> Result<bool, String> {
    let Some(mut proposal) = load_proposal(cwd)? else {
        return Ok(false);
    };
    validate_receipt_binding(&proposal).map_err(|error| {
        format!(
            "persisted apply receipt failed validation and was retained for manual recovery: {error}"
        )
    })?;
    let candidates = runtime::memory::read_self_improve_candidates(cwd);
    let Some(candidate) = candidates
        .iter()
        .find(|candidate| candidate.id == proposal.run.candidate_id)
    else {
        return Err(
            "persisted apply receipt references a missing candidate and was retained for manual recovery"
                .to_string(),
        );
    };
    if !candidate.kind.is_actionable() {
        return Err(
            "persisted apply receipt references a non-actionable candidate and was retained for manual recovery"
                .to_string(),
        );
    }
    let candidate_is_applied =
        candidate.status == decision_core::dreamer::CandidateStatus::Applied;
    if candidate_is_applied {
        if !matches!(
            proposal.phase,
            ProposalPhase::PatchApplied | ProposalPhase::CandidateApplied
        ) {
            return Err(
                "persisted apply receipt is ambiguous (its candidate is already terminal but the receipt never recorded a patch application); it was retained for manual recovery"
                    .to_string(),
            );
        }
        validate_applied_state(cwd, &proposal, &proposal.patch_diff).map_err(|error| {
            format!(
                "persisted apply receipt does not match the repository state and was retained for manual recovery: {error}"
            )
        })?;
        proposal.phase = ProposalPhase::CandidateApplied;
        write_persisted_proposal(cwd, &proposal)?;
        validate_applied_state(cwd, &proposal, &proposal.patch_diff).map_err(|error| {
            format!(
                "persisted apply receipt does not match the repository state and was retained for manual recovery: {error}"
            )
        })?;
        clear_proposal(cwd)?;
        return Ok(true);
    }
    if candidate.status.is_terminal() {
        return Err(
            "persisted apply receipt references a terminal candidate with a conflicting outcome and was retained for manual recovery"
                .to_string(),
        );
    }
    if proposal.phase == ProposalPhase::CandidateApplied {
        return Err(
            "persisted apply receipt claims candidate completion without a durable candidate event and was retained for manual recovery"
                .to_string(),
        );
    }
    if proposal.phase == ProposalPhase::Applying
        && validate_applied_state(cwd, &proposal, &proposal.patch_diff).is_ok()
    {
        proposal.phase = ProposalPhase::PatchApplied;
        write_persisted_proposal(cwd, &proposal)?;
    }
    if !matches!(proposal.phase, ProposalPhase::PatchApplied | ProposalPhase::CandidateApplied) {
        return Ok(false);
    }

    // A receipt that claims the patch landed must match the actual staged
    // state before the candidate is terminally marked.
    validate_applied_state(cwd, &proposal, &proposal.patch_diff).map_err(|error| {
        format!(
            "persisted apply receipt does not match the repository state and was retained for manual recovery: {error}"
        )
    })?;
    mark_self_improve_candidate_applied(cwd, &proposal.run.candidate_id).map_err(|error| {
        format!(
            "applied patch receipt could not record candidate completion; proposal was retained for recovery: {error}"
        )
    })?;
    proposal.phase = ProposalPhase::CandidateApplied;
    write_persisted_proposal(cwd, &proposal)?;
    validate_applied_state(cwd, &proposal, &proposal.patch_diff).map_err(|error| {
        format!(
            "persisted apply receipt does not match the repository state and was retained for manual recovery: {error}"
        )
    })?;
    clear_proposal(cwd)?;
    Ok(true)
}

#[cfg(test)]
mod tests;
