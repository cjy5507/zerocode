use super::{
    apply, decision_is_actionable, disk_refusal, format_proposal, load_proposal,
    maybe_auto_self_improve_preflight_at, propose, reject, review, run_to_completion, show,
    status_report, AutoSelfImproveOutcome, PatchGenerator,
};
use decision_core::dreamer::{
    CandidateEvidence, CandidateKind, DreamJudgeDecision, SelfImproveCandidate,
};
use runtime::memory::{mark_self_improve_candidate_applied, record_self_improve_candidate};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn unique_repo(label: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "zo-improve-{label}-{}-{nanos}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

#[test]
fn low_disk_refusal_matches_shared_bash_wording() {
    let parent = Path::new("/tmp/self-improve-parent");
    assert_eq!(
        disk_refusal(12 * 1024 * 1024, parent).as_deref(),
        Some(
            "refusing to run: only 12MB left on the filesystem holding /tmp/self-improve-parent — free disk space first (reclaim Rust target/ build dirs, temp scratch, or orphaned worktrees)"
        )
    );
}

fn git(repo: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .status()
        .expect("run git");
    assert!(status.success(), "git {args:?} failed");
}

/// A repo with one commit containing `crates/x.rs` and `crates/external.rs`, plus a seeded candidate.
fn seed_repo(label: &str) -> std::path::PathBuf {
    let repo = unique_repo(label);
    std::fs::create_dir_all(repo.join("crates")).expect("mkdir crates");
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "zo@example.com"]);
    git(&repo, &["config", "user.name", "Zo Test"]);
    std::fs::write(repo.join(".gitignore"), ".zo/\n").expect("write gitignore");
    std::fs::write(repo.join("crates/x.rs"), "pub fn x() -> i32 { 1 }\n").expect("write source");
    std::fs::write(
        repo.join("crates/external.rs"),
        "pub fn external() -> i32 { 10 }\n",
    )
    .expect("write external source");
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "seed"]);

    let candidate = SelfImproveCandidate::new(
        CandidateKind::GoalFailure,
        "x() should return 2",
        vec![CandidateEvidence {
            session_id: "sess".to_string(),
            source: "goal".to_string(),
            detail: "x() returns the wrong value".to_string(),
            verified: true,
        }],
    );
    record_self_improve_candidate(&repo, &candidate).expect("seed candidate");
    repo
}

fn seed_cargo_repo(label: &str) -> std::path::PathBuf {
    let repo = seed_repo(label);
    std::fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"self-improve-test\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lib]\npath = \"crates/x.rs\"\n",
    )
    .expect("write Cargo.toml");
    let status = Command::new("cargo")
        .arg("generate-lockfile")
        .current_dir(&repo)
        .status()
        .expect("generate Cargo.lock");
    assert!(status.success(), "cargo generate-lockfile failed");
    git(&repo, &["add", "Cargo.toml", "Cargo.lock"]);
    git(&repo, &["commit", "-q", "-m", "cargo fixture"]);
    repo
}
/// yields a clean patch that re-applies in the quarantine worktree.
struct StubGenerator;
impl PatchGenerator for StubGenerator {
    fn generate(&self, worktree: &Path, _prompt: &str) -> Result<String, String> {
        std::fs::write(worktree.join("crates/x.rs"), "pub fn x() -> i32 { 2 }\n")
            .map_err(|e| e.to_string())?;
        let output = Command::new("git")
            .arg("-C")
            .arg(worktree)
            .args(["diff", "HEAD"])
            .output()
            .map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// Generator that creates an untracked binary file in the isolated worktree.
struct UntrackedBinaryGenerator;
impl PatchGenerator for UntrackedBinaryGenerator {
    fn generate(&self, worktree: &Path, _prompt: &str) -> Result<String, String> {
        std::fs::write(worktree.join("crates/generated.bin"), [0, 1, b'f', b'o', b'r', b'g', b'e', 0xff])
            .map_err(|error| error.to_string())?;
        super::capture_worktree_patch(worktree)
    }
}

/// Generator that makes no edits → empty diff.
struct NoopGenerator;
impl PatchGenerator for NoopGenerator {
    fn generate(&self, _worktree: &Path, _prompt: &str) -> Result<String, String> {
        Ok(String::new())
    }
}

/// Generator that records whether it was ever invoked. The gate must never reach
/// it for a non-actionable verdict (and must reach it for an actionable one).
struct SpyGenerator {
    called: AtomicBool,
}
impl PatchGenerator for SpyGenerator {
    fn generate(&self, worktree: &Path, _prompt: &str) -> Result<String, String> {
        self.called.store(true, Ordering::Relaxed);
        std::fs::write(worktree.join("crates/x.rs"), "pub fn x() -> i32 { 2 }\n")
            .map_err(|e| e.to_string())?;
        let output = Command::new("git")
            .arg("-C")
            .arg(worktree)
            .args(["diff", "HEAD"])
            .output()
            .map_err(|e| e.to_string())?;
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

fn prepare_patch_applied_receipt(
    repo: &Path,
    stage_patch: bool,
) -> super::PersistedProposal {
    let proposal = propose(repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal exists");
    if stage_patch {
        super::git_apply(repo, &proposal.patch_diff).expect("stage approved patch");
    }
    let mut receipt = load_proposal(repo).unwrap().unwrap();
    receipt.phase = super::ProposalPhase::PatchApplied;
    super::write_persisted_proposal(repo, &receipt).expect("persist apply receipt");
    receipt
}

#[test]
fn propose_generates_quarantined_patch_and_persists_it() {
    let repo = seed_repo("propose");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("a candidate produces a proposal");

    // The generated diff was quarantined and its changed path captured.
    assert!(
        proposal.run.changed_paths.iter().any(|p| p == "crates/x.rs"),
        "changed paths: {:?}",
        proposal.run.changed_paths
    );
    // Approval is withheld at propose time, so the gate names it as blocking.
    assert!(
        proposal
            .blocking_reasons
            .iter()
            .any(|r| r == "missing_user_approval"),
        "reasons: {:?}",
        proposal.blocking_reasons
    );
    // The proposal is persisted for a later `/improve apply`.
    assert!(repo
        .join(".zo/dream/improve/last-proposal.json")
        .exists());

    // The generation worktree was cleaned up (no stray worktrees linger).
    let worktrees = Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["worktree", "list"])
        .output()
        .expect("worktree list");
    assert_eq!(
        String::from_utf8_lossy(&worktrees.stdout).lines().count(),
        1,
        "only the main worktree should remain"
    );

    let rendered = format_proposal(&proposal);
    assert!(rendered.contains("Self-improve proposal"));
    assert!(rendered.contains("crates/x.rs"));
    assert!(rendered.contains(&proposal.proposal_id));
    assert!(rendered.contains("diff --git"));
    assert!(rendered.contains("Review           required"));

    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn proposal_lifecycle_requires_exact_selection_and_review_before_apply() {
    let repo = seed_cargo_repo("proposal-lifecycle");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");

    assert_eq!(proposal.proposal_id.len(), 64);
    assert!(show(&repo, &"0".repeat(64)).unwrap_err().contains("stale"));
    assert!(show(&repo, &proposal.proposal_id)
        .expect("show exact proposal")
        .contains(&proposal.proposal_id));
    assert!(apply(&repo, &proposal.proposal_id)
        .unwrap_err()
        .contains("review"));

    let reviewed = review(&repo, &proposal.proposal_id).expect("review exact proposal");
    assert!(reviewed.contains("reviewed"));
    assert!(apply(&repo, &proposal.proposal_id)
        .expect("reviewed proposal applies")
        .contains("Applied self-improve patch"));
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn proposal_reject_refuses_stale_id_and_retires_exact_candidate() {
    let repo = seed_repo("proposal-reject");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");

    assert!(reject(&repo, &"f".repeat(64)).unwrap_err().contains("stale"));
    assert!(reject(&repo, &proposal.proposal_id)
        .expect("reject exact proposal")
        .contains("rejected"));
    assert!(load_proposal(&repo).expect("load proposal").is_none());
    let candidate = runtime::memory::read_self_improve_candidates(&repo)
        .into_iter()
        .find(|candidate| candidate.id == proposal.run.candidate_id)
        .expect("proposal candidate remains auditable");
    assert_eq!(candidate.status, decision_core::dreamer::CandidateStatus::Rejected);
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn propose_captures_untracked_binary_files_as_a_complete_patch() {
    let repo = seed_repo("untracked-binary");
    let proposal = propose(&repo, &UntrackedBinaryGenerator)
        .expect("propose succeeds")
        .expect("proposal created");

    assert!(proposal.patch_diff.contains("GIT binary patch"), "{}", proposal.patch_diff);
    assert!(proposal
        .run
        .changed_paths
        .iter()
        .any(|path| path == "crates/generated.bin"));
    super::git_apply(&repo, &proposal.patch_diff).expect("binary proposal re-applies");
    assert_eq!(
        std::fs::read(repo.join("crates/generated.bin")).unwrap(),
        vec![0, 1, b'f', b'o', b'r', b'g', b'e', 0xff]
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn generation_environment_projects_config_and_credentials_without_ambient_leaks() {
    let repo = unique_repo("generation-environment");
    let worktree = repo.join("run/gen");
    std::fs::create_dir_all(worktree.parent().unwrap().join("home")).unwrap();
    std::fs::create_dir_all(worktree.parent().unwrap().join("tmp")).unwrap();
    std::fs::create_dir_all(&worktree).unwrap();

    let environment: std::collections::BTreeMap<_, _> = super::generation_environment_with(
        &worktree,
        |name| match name {
            "ZO_CONFIG_HOME" => Some("/trusted/config".into()),
            "OPENAI_API_KEY" => Some("provider-secret".into()),
            "UNRELATED_SECRET" => Some("must-not-pass".into()),
            _ => None,
        },
    )
    .unwrap()
    .into_iter()
    .collect();
    assert_eq!(environment.get(&std::ffi::OsString::from("ZO_CONFIG_HOME")), Some(&std::ffi::OsString::from("/trusted/config")));
    assert_eq!(environment.get(&std::ffi::OsString::from("OPENAI_API_KEY")), Some(&std::ffi::OsString::from("provider-secret")));
    assert_eq!(environment.get(&std::ffi::OsString::from("PATH")), Some(&std::ffi::OsString::from(super::TRUSTED_EXECUTABLE_PATH)));
    assert_eq!(environment.get(&std::ffi::OsString::from("HOME")), Some(&worktree.parent().unwrap().join("home").into_os_string()));
    assert!(!environment.contains_key(&std::ffi::OsString::from("UNRELATED_SECRET")));

    #[cfg(not(windows))]
    {
        let error = super::ZoSubprocessGenerator {
            zo_bin: "/missing/zo-provider-secret".into(),
        }
        .generate(&worktree, "candidate")
        .expect_err("spawn failure is redacted");
        assert!(error.contains("diagnostics redacted"), "{error}");
        assert!(!error.contains("provider-secret"), "{error}");
    }
    let _ = std::fs::remove_dir_all(repo);
}

#[cfg(unix)]
#[test]
fn generation_run_directories_are_private_and_stale_runs_are_bounded() {
    use std::os::unix::fs::PermissionsExt as _;

    let repo = seed_repo("generation-retention");
    for index in 0..(super::STALE_GENERATION_RUN_RETENTION + 2) {
        runtime::secure_fs::ensure_private_dir(
            &repo,
            &Path::new(".zo/dream/improve").join(format!("stale-{index}")),
        )
        .unwrap();
    }
    let worktree = super::add_worktree(&repo, "retained-run").expect("create worktree");
    let run_dir = worktree.parent().unwrap();
    let home = run_dir.join("home");
    let temp = run_dir.join("tmp");
    for directory in [run_dir, home.as_path(), temp.as_path()] {
        assert_eq!(
            std::fs::metadata(directory).unwrap().permissions().mode() & 0o777,
            0o700,
            "{} must be owner-only",
            directory.display()
        );
    }
    let run_count = std::fs::read_dir(repo.join(".zo/dream/improve"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.metadata().map(|metadata| metadata.is_dir()).unwrap_or(false))
        .count();
    assert!(
        run_count <= super::STALE_GENERATION_RUN_RETENTION + 1,
        "retention left {run_count} run directories"
    );
    super::remove_worktree(&repo, &worktree).expect("remove worktree");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn status_report_is_read_only_when_no_dream_state_exists() {
    let repo = unique_repo("status-readonly");
    std::fs::create_dir_all(&repo).expect("create repo dir");

    let report = status_report(&repo, true).expect("status succeeds");

    assert!(report.contains("Self-improve status"));
    assert!(report.contains("Automation      enabled"));
    assert!(report.contains("Pending proposal no"));
    assert!(report.contains("Candidate        none"));
    assert!(
        !repo.join(".zo").exists(),
        "status must not create .zo state"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn status_report_surfaces_candidate_ready() {
    let repo = seed_repo("status-candidate");

    let report = status_report(&repo, true).expect("status succeeds");

    assert!(report.contains("Scheduler       ready"), "{report}");
    assert!(report.contains("Candidate        ready (1 active; top goal_failure · proposed)"), "{report}");
    assert!(
        report.contains("Fusion report    none"),
        "no fusion report exists before the first preflight/improve: {report}"
    );
    assert!(report.contains("Next action      run /improve to generate a proposal"), "{report}");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn status_report_surfaces_pending_proposal() {
    let repo = seed_repo("status-pending");
    propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");

    let report = status_report(&repo, true).expect("status succeeds");

    assert!(report.contains("Scheduler       paused (pending proposal)"), "{report}");
    assert!(report.contains("Pending proposal yes"), "{report}");
    assert!(report.contains("Proposal phase  ready_for_review"), "{report}");
    assert!(report.contains("Next action      review or reject proposal"), "{report}");
    let _ = std::fs::remove_dir_all(repo);
}

#[cfg(unix)]
#[test]
fn status_report_rejects_symlinked_proposal_file() {
    let repo = seed_repo("status-symlink-proposal-file");
    let improve_dir = repo.join(".zo/dream/improve");
    std::fs::create_dir_all(&improve_dir).unwrap();
    let target = repo.join("outside-proposal.json");
    std::fs::write(&target, b"{}").unwrap();
    let proposal = improve_dir.join("last-proposal.json");
    let _ = std::fs::remove_file(&proposal);
    std::os::unix::fs::symlink(&target, &proposal).unwrap();

    let error = status_report(&repo, true).expect_err("symlinked proposal file rejected");
    assert!(!error.is_empty());
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn status_report_surfaces_disabled_automation() {
    let repo = unique_repo("status-disabled");
    std::fs::create_dir_all(&repo).expect("create repo dir");

    let report = status_report(&repo, false).expect("status succeeds");

    assert!(report.contains("Automation      disabled"), "{report}");
    assert!(report.contains("Scheduler       disabled"), "{report}");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn auto_self_improve_reports_candidate_ready_without_generating_patch() {
    let repo = seed_repo("auto-ready");
    let outcome = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect("auto preflight succeeds");

    assert_eq!(outcome, AutoSelfImproveOutcome::CandidateReady);
    assert!(
        load_proposal(&repo).expect("load proposal").is_none(),
        "automatic startup preflight must not generate or persist a patch"
    );
    let (fusion, _) = runtime::memory::latest_dream_fusion_report(&repo)
        .expect("preflight persists a read-only fusion report for the ready candidate");
    assert_eq!(fusion.summary, "x() should return 2");
    let _ = std::fs::remove_dir_all(repo);
}

/// Opt-in path: when a generator is supplied (the `autoImproveProposalsEnabled`
/// gate is on), the startup preflight runs it and parks a *gated* proposal
/// automatically — no manual `/improve` — but leaves it in the proposed phase,
/// never applied. Applying stays an explicit human `/improve apply`.
#[test]
fn auto_self_improve_parks_a_gated_proposal_when_opted_in() {
    let repo = seed_repo("auto-propose");
    let outcome = maybe_auto_self_improve_preflight_at(
        &repo,
        std::time::SystemTime::now(),
        Some(&StubGenerator),
    )
    .expect("auto preflight succeeds");

    assert_eq!(outcome, AutoSelfImproveOutcome::ProposalGenerated);

    let parked = load_proposal(&repo)
        .expect("load proposal")
        .expect("a proposal was persisted for review");
    assert_eq!(
        parked.phase,
        super::ProposalPhase::Ready,
        "the proposal must be parked for review, not auto-applied"
    );
    let _ = std::fs::remove_dir_all(repo);
}

/// After the startup preflight, `/improve status` surfaces the persisted
/// fusion report and the per-signature candidate rows — the operator opens the
/// next session to "this is what /improve would act on", not a bare backoff
/// timestamp.
#[test]
fn status_report_surfaces_preflight_fusion_report_and_segmented_rows() {
    let repo = seed_repo("status-fusion");
    let outcome = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect("auto preflight succeeds");
    assert_eq!(outcome, AutoSelfImproveOutcome::CandidateReady);

    let report = status_report(&repo, true).expect("status succeeds");

    assert!(
        report.contains("Fusion report    plan_patch — x() should return 2 ("),
        "{report}"
    );
    assert!(
        report.contains("1. goal_failure · x() should return 2 · 1 evidence · last "),
        "{report}"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn auto_self_improve_ignores_terminal_candidates() {
    let repo = seed_repo("auto-terminal-only");
    let id = runtime::memory::read_self_improve_candidates(&repo)
        .into_iter()
        .next()
        .expect("seed candidate exists")
        .id;
    mark_self_improve_candidate_applied(&repo, &id).unwrap();

    let outcome = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect("auto preflight succeeds");

    assert_eq!(outcome, AutoSelfImproveOutcome::NoCandidate);
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn auto_self_improve_failure_recorder_writes_sanitized_backoff_marker() {
    let repo = seed_repo("auto-failure-marker");
    super::record_auto_self_improve_failure(&repo, "sensitive provider token: abc123");

    let failure = std::fs::read_to_string(repo.join(".zo/dream/.last_self_improve_error.json"))
        .expect("failure marker written");
    assert!(failure.contains("dreamer_io_error"));
    assert!(!failure.contains("abc123"));
    let state = runtime::memory::read_self_improve_schedule_state(&repo)
        .expect("failure marker participates in backoff state");
    assert!(state.last_failure.is_some());
    let _ = std::fs::remove_dir_all(repo);
}

#[cfg(unix)]
#[test]
fn auto_self_improve_rejects_symlinked_proposal_dir_before_pending_check() {
    let repo = seed_repo("auto-symlink-proposal-dir");
    let improve_dir = repo.join(".zo/dream/improve");
    let outside = repo.join("outside-improve");
    let _ = std::fs::remove_dir_all(&improve_dir);
    std::fs::create_dir_all(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, &improve_dir).unwrap();

    let error = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect_err("symlinked proposal dir must be rejected");
    assert!(
        error.contains("not a real directory"),
        "unexpected error: {error}"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn auto_self_improve_respects_backoff_without_generating_patch() {
    let repo = seed_repo("auto-backoff");
    runtime::memory::record_self_improve_attempt(&repo).expect("record attempt");

    let outcome = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect("auto runner skips");

    assert_eq!(outcome, AutoSelfImproveOutcome::SkippedBackoff);
    assert!(load_proposal(&repo).expect("load proposal").is_none());
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn auto_self_improve_does_not_duplicate_existing_proposal() {
    let repo = seed_repo("auto-existing-proposal");
    propose(&repo, &StubGenerator)
        .expect("manual propose succeeds")
        .expect("proposal exists");

    let outcome = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect("auto runner skips existing proposal");

    assert_eq!(outcome, AutoSelfImproveOutcome::PendingProposalExists);
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn ambiguous_applied_receipt_is_retained_and_surfaces_recovery_error() {
    let repo = seed_repo("reconcile-applied");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal exists");
    mark_self_improve_candidate_applied(&repo, &proposal.run.candidate_id).unwrap();

    // The candidate is terminal but the receipt never recorded a patch
    // application, so reconciliation must refuse to guess: the receipt is
    // retained on disk and both status and the scheduler surface the error.
    let error = status_report(&repo, true).expect_err("ambiguous receipt fails status");
    assert!(error.contains("ambiguous"), "{error}");
    assert!(repo.join(".zo/dream/improve/last-proposal.json").exists());

    let error = maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None)
        .expect_err("ambiguous receipt pauses the scheduler with an error");
    assert!(error.contains("ambiguous"), "{error}");
    assert!(repo.join(".zo/dream/improve/last-proposal.json").exists());
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn propose_preserves_pending_proposal_without_calling_generator() {
    let repo = seed_repo("propose-pending");
    propose(&repo, &StubGenerator)
        .expect("first propose succeeds")
        .expect("proposal exists");
    let proposal_path = repo.join(super::PROPOSAL_RELATIVE_PATH);
    let before = std::fs::read(&proposal_path).unwrap();
    let spy = SpyGenerator {
        called: AtomicBool::new(false),
    };

    let Err(error) = propose(&repo, &spy) else {
        panic!("second propose must be refused while a proposal is pending");
    };
    assert!(error.contains("already exists"), "{error}");
    assert!(!spy.called.load(Ordering::Relaxed));
    assert_eq!(std::fs::read(&proposal_path).unwrap(), before);
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn propose_preserves_apply_receipt_without_calling_generator() {
    let repo = seed_repo("propose-apply-receipt");
    prepare_patch_applied_receipt(&repo, true);
    record_self_improve_candidate(
        &repo,
        &SelfImproveCandidate::new(
            CandidateKind::TurnFailure,
            "a second actionable candidate",
            Vec::new(),
        ),
    )
    .unwrap();
    let proposal_path = repo.join(super::PROPOSAL_RELATIVE_PATH);
    let before = std::fs::read(&proposal_path).unwrap();
    let spy = SpyGenerator {
        called: AtomicBool::new(false),
    };

    let Err(error) = propose(&repo, &spy) else {
        panic!("proposal generation must be refused while an apply receipt exists");
    };
    assert!(error.contains("already exists"), "{error}");
    assert!(!spy.called.load(Ordering::Relaxed));
    assert_eq!(std::fs::read(&proposal_path).unwrap(), before);
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn success_telemetry_is_not_reported_as_candidate_ready() {
    let repo = unique_repo("success-telemetry");
    std::fs::create_dir_all(&repo).unwrap();
    for kind in [CandidateKind::VerifiedAccept, CandidateKind::PostTurn, CandidateKind::GoalTerminal] {
        record_self_improve_candidate(
            &repo,
            &SelfImproveCandidate::new(kind, kind.as_str(), Vec::new()),
        )
        .unwrap();
    }

    let report = status_report(&repo, true).expect("status succeeds");
    assert!(report.contains("Candidate        none"), "{report}");
    assert_eq!(
        maybe_auto_self_improve_preflight_at(&repo, std::time::SystemTime::now(), None).unwrap(),
        AutoSelfImproveOutcome::NoCandidate
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn propose_with_no_changes_is_an_error_not_a_phantom_patch() {
    let repo = seed_repo("noop");
    let result = propose(&repo, &NoopGenerator);
    assert!(result.is_err(), "an empty generation must not be proposed");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn non_actionable_fusion_verdicts_gate_out_generation() {
    // Reject / NeedMoreEvidence must short-circuit BEFORE the headless `zo -p`
    // generation pass; PlanPatch / Quarantine are the only actionable verdicts.
    assert!(!decision_is_actionable(DreamJudgeDecision::Reject));
    assert!(!decision_is_actionable(DreamJudgeDecision::NeedMoreEvidence));
    assert!(decision_is_actionable(DreamJudgeDecision::PlanPatch));
    assert!(decision_is_actionable(DreamJudgeDecision::Quarantine));
}

#[test]
fn propose_reaches_the_generator_on_an_actionable_verdict() {
    // The seeded GoalTerminal candidate synthesizes to an actionable PlanPatch
    // verdict, so `propose` must drive the generator — proving the gate added for
    // BB3-A is wired in (it lets actionable candidates through) rather than dead.
    let repo = seed_repo("spy-actionable");
    let spy = SpyGenerator {
        called: AtomicBool::new(false),
    };
    propose(&repo, &spy)
        .expect("propose succeeds")
        .expect("an actionable candidate produces a proposal");
    assert!(
        spy.called.load(Ordering::Relaxed),
        "an actionable verdict must reach the generator"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn generation_cleanup_failure_is_reported() {
    let repo = seed_repo("cleanup-failure");
    let run = runtime::secure_fs::ensure_private_dir(
        &repo,
        Path::new(".zo/dream/improve/cleanup-failure-run"),
    )
    .expect("create private run");
    let cleanup_error = super::remove_worktree(&repo, &run.join("gen"))
        .expect_err("removing an unregistered worktree must fail");
    assert!(cleanup_error.contains("git worktree remove failed"));

    let error = super::finish_generation(Ok("patch".to_string()), Err(cleanup_error.clone()))
        .expect_err("successful generation must not hide cleanup failure");
    assert_eq!(error, cleanup_error);

    let combined = super::finish_generation(
        Err("generation failed".to_string()),
        Err("cleanup failed".to_string()),
    )
    .expect_err("both failures must be reported");
    assert!(combined.contains("generation failed"));
    assert!(combined.contains("cleanup failed"));
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn successful_worktree_cleanup_removes_run_directory() {
    let repo = seed_repo("cleanup-parent");
    let worktree = super::add_worktree(&repo, "cleanup-parent-run").unwrap();
    let run_dir = worktree.parent().unwrap().to_path_buf();
    assert!(run_dir.is_dir());
    super::remove_worktree(&repo, &worktree).unwrap();
    assert!(!run_dir.exists(), "empty generation run directory must be removed");
    let _ = std::fs::remove_dir_all(repo);
}

#[cfg(unix)]
#[test]
fn git_apply_does_not_follow_legacy_apply_diff_symlink() {
    use std::os::unix::fs::symlink;

    let repo = seed_repo("apply-symlink");
    let improve = super::improve_dir(&repo).unwrap();
    let target = repo.join("outside.txt");
    std::fs::write(&target, "sentinel").unwrap();
    symlink(&target, improve.join("apply.diff")).unwrap();
    let patch = "diff --git a/crates/x.rs b/crates/x.rs\n--- a/crates/x.rs\n+++ b/crates/x.rs\n@@ -1 +1 @@\n-pub fn x() -> i32 { 1 }\n+pub fn x() -> i32 { 2 }\n";

    super::git_apply(&repo, patch).unwrap();
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "sentinel");
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/x.rs")).unwrap(),
        "pub fn x() -> i32 { 2 }\n"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn generation_child_is_bounded_and_has_no_command_or_network_tools() {
    let args = super::generation_args("candidate text");
    assert!(!args.iter().any(|arg| arg == "--dangerously-skip-permissions"));
    assert!(args.windows(2).any(|pair| pair == ["--permission-mode", "workspace-write"]));
    let tools = args
        .windows(2)
        .find(|pair| pair[0] == "--allowedTools")
        .map(|pair| pair[1].as_str())
        .expect("allowed tools");
    assert!(!tools.split(',').any(|tool| matches!(tool, "bash" | "WebFetch" | "WebSearch" | "ToolSearch")));
}

#[test]
fn apply_without_a_proposal_is_a_clear_error() {
    let repo = seed_repo("apply-missing");
    let error = apply(&repo, "0").expect_err("apply with no proposal must error");
    assert!(error.contains("no pending"), "error: {error}");
    let _ = std::fs::remove_dir_all(repo);
}

#[cfg(unix)]
#[test]
fn run_to_completion_reaps_descendant_after_parent_success() {
    let mut command = Command::new("sh");
    command
        .args(["-c", "sleep 30 & exit 0"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    super::configure_generation_process_group(&mut command);
    let child = command.spawn().expect("spawn parent with descendant");

    let start = std::time::Instant::now();
    let outcome = run_to_completion(child, std::time::Duration::from_secs(2))
        .expect("successful parent should reap its remaining process group");
    assert!(outcome.status.success());
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "descendant-held pipes must not block drain joins: {:?}",
        start.elapsed()
    );
}

#[cfg(unix)]
#[test]
fn run_to_completion_kills_a_child_that_outlives_the_timeout() {
    // A stalled generation child must be killed (not inherited / left running) so
    // a wedged provider cannot wedge `/improve`. Use a child that would sleep far
    // past the timeout and assert the wall-clock ceiling reaps it.
    let mut command = Command::new("sh");
    command
        .args(["-c", "sleep 30 & wait"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    super::configure_generation_process_group(&mut command);
    let child = command.spawn().expect("spawn sleeper process tree");
    let start = std::time::Instant::now();
    let result = run_to_completion(child, std::time::Duration::from_millis(200));
    assert!(
        result.is_err(),
        "a child outliving the timeout must error, not hang"
    );
    assert!(
        result.unwrap_err().contains("timed out"),
        "the error must name the timeout"
    );
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "the timeout must fire well before the child's own 30s sleep"
    );
}

#[test]
fn successful_verified_apply_applies_the_patch_once_and_preserves_the_expected_state() {
    let repo = seed_cargo_repo("apply-success");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");
    assert!(
        proposal.run.check_results.iter().all(|check| check.success),
        "proposal checks were not green: {:?}",
        proposal.run.check_results
    );

    review(&repo, &proposal.proposal_id).expect("review succeeds");
    let outcome = apply(&repo, &proposal.proposal_id).expect("verified apply succeeds");
    assert!(outcome.contains("Applied self-improve patch"), "{outcome}");
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/x.rs")).unwrap(),
        "pub fn x() -> i32 { 2 }\n"
    );
    let state = super::capture_apply_state(&repo).expect("capture final state");
    assert!(state.worktree_diff.is_empty(), "working tree is unstaged");
    assert!(!state.staged_diff.is_empty(), "approved patch is staged exactly once");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn patch_applied_receipt_recovery_completes_once_with_expected_state() {
    let repo = seed_repo("recover-patch-applied");
    let receipt = prepare_patch_applied_receipt(&repo, true);
    let candidate_file_stem: String = receipt
        .run
        .candidate_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect();
    let candidate_path = repo
        .join(".zo/dream/candidates")
        .join(format!("{candidate_file_stem}.jsonl"));

    assert!(super::reconcile_applied_proposal(&repo).unwrap());
    assert!(load_proposal(&repo).unwrap().is_none());
    super::validate_applied_state(&repo, &receipt, &receipt.patch_diff)
        .expect("recovered staged state matches approved patch");
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/x.rs")).unwrap(),
        "pub fn x() -> i32 { 2 }\n"
    );
    let candidates = runtime::memory::read_self_improve_candidates(&repo);
    assert_eq!(
        candidates
            .iter()
            .find(|candidate| candidate.id == receipt.run.candidate_id)
            .unwrap()
            .status,
        decision_core::dreamer::CandidateStatus::Applied
    );

    let candidate_after_first_recovery = std::fs::read(&candidate_path).unwrap();
    assert!(!super::reconcile_applied_proposal(&repo).unwrap());
    assert_eq!(
        std::fs::read(&candidate_path).unwrap(),
        candidate_after_first_recovery,
        "a second recovery pass must not append another Applied event"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn patch_applied_receipt_mismatch_is_retained_with_clear_error() {
    let repo = seed_repo("recover-staged-mismatch");
    let receipt = prepare_patch_applied_receipt(&repo, false);
    std::fs::write(repo.join("crates/x.rs"), "pub fn x() -> i32 { 3 }\n").unwrap();
    mark_self_improve_candidate_applied(&repo, &receipt.run.candidate_id).unwrap();
    let proposal_path = repo.join(super::PROPOSAL_RELATIVE_PATH);
    let before = std::fs::read(&proposal_path).unwrap();

    let error = super::reconcile_applied_proposal(&repo)
        .expect_err("a receipt must not complete against mismatched repository state");
    assert!(error.contains("does not match the repository state"), "{error}");
    assert_eq!(std::fs::read(&proposal_path).unwrap(), before);
    assert_eq!(
        runtime::memory::read_self_improve_candidates(&repo)
            .into_iter()
            .find(|candidate| candidate.id == receipt.run.candidate_id)
            .unwrap()
            .status,
        decision_core::dreamer::CandidateStatus::Applied
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn patch_applied_receipt_recovery_rejects_each_tampered_binding_field() {
    let repo = seed_repo("recover-tampered-receipt");
    let receipt = prepare_patch_applied_receipt(&repo, true);
    let proposal_path = repo.join(super::PROPOSAL_RELATIVE_PATH);
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&proposal_path).unwrap()).unwrap();

    for (field, tamper) in [
        ("patch digest", ("patch_digest", serde_json::json!("0".repeat(64)))),
        ("approval digest", ("approval_digest", serde_json::json!("1".repeat(64)))),
        ("base commit", ("base_commit", serde_json::json!("2".repeat(40)))),
        ("changed paths", ("changed_paths", serde_json::json!(["crates/external.rs"]))),
        ("risk", ("risk", serde_json::json!("high"))),
    ] {
        let mut payload = original.clone();
        payload[tamper.0] = tamper.1;
        let tampered = serde_json::to_vec_pretty(&payload).unwrap();
        std::fs::write(&proposal_path, &tampered).unwrap();

        let error = super::reconcile_applied_proposal(&repo)
            .expect_err("tampered receipt binding must be rejected");
        assert!(error.contains("failed validation"), "{field}: {error}");
        assert_eq!(std::fs::read(&proposal_path).unwrap(), tampered, "{field}");
    }

    let mut payload = original;
    payload["run"]["candidate_id"] = serde_json::json!("tampered-candidate");
    let tampered = serde_json::to_vec_pretty(&payload).unwrap();
    std::fs::write(&proposal_path, &tampered).unwrap();
    let error = super::reconcile_applied_proposal(&repo)
        .expect_err("tampered candidate id must be rejected");
    assert!(error.contains("failed validation"), "candidate ID: {error}");
    assert_eq!(std::fs::read(&proposal_path).unwrap(), tampered);
    assert_ne!(
        runtime::memory::read_self_improve_candidates(&repo)
            .into_iter()
            .find(|candidate| candidate.id == receipt.run.candidate_id)
            .unwrap()
            .status,
        decision_core::dreamer::CandidateStatus::Applied
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn failed_apply_postcondition_restores_the_exact_pre_apply_state() {
    let repo = seed_repo("apply-restore");
    let patch = "diff --git a/crates/x.rs b/crates/x.rs\n--- a/crates/x.rs\n+++ b/crates/x.rs\n@@ -1 +1 @@\n-pub fn x() -> i32 { 1 }\n+pub fn x() -> i32 { 2 }\n";
    let before = super::capture_apply_state(&repo).expect("capture before state");

    let error = super::apply_patch_and_validate(&repo, patch, || {
        Err("induced postcondition failure".to_string())
    })
    .expect_err("failed postcondition restores repository state");
    assert!(error.contains("repository state was restored"), "{error}");

    let after = super::capture_apply_state(&repo).expect("capture restored state");
    assert_eq!(after.head, before.head);
    assert_eq!(after.status, before.status);
    assert_eq!(after.staged_diff, before.staged_diff);
    assert_eq!(after.worktree_diff, before.worktree_diff);
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/x.rs")).unwrap(),
        "pub fn x() -> i32 { 1 }\n"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn failed_apply_postcondition_reverses_approved_patch_and_preserves_external_tracked_edit() {
    let repo = seed_repo("apply-preserve-external-file-edit");
    let patch = "diff --git a/crates/x.rs b/crates/x.rs\n--- a/crates/x.rs\n+++ b/crates/x.rs\n@@ -1 +1 @@\n-pub fn x() -> i32 { 1 }\n+pub fn x() -> i32 { 2 }\n";

    let error = super::apply_patch_and_validate(&repo, patch, || {
        std::fs::write(
            repo.join("crates/external.rs"),
            "pub fn external() -> i32 { 11 }\n",
        )
        .expect("write external tracked edit");
        Err("induced postcondition failure".to_string())
    })
    .expect_err("failed postcondition must reverse only the approved patch");

    assert!(
        error.contains("approved patch was reversed and external edits were preserved"),
        "{error}"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/x.rs")).unwrap(),
        "pub fn x() -> i32 { 1 }\n"
    );
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/external.rs")).unwrap(),
        "pub fn external() -> i32 { 11 }\n"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn failed_apply_postcondition_refuses_rollback_over_same_hunk_external_edit() {
    let repo = seed_repo("apply-preserve-same-hunk-edit");
    let patch = "diff --git a/crates/x.rs b/crates/x.rs\n--- a/crates/x.rs\n+++ b/crates/x.rs\n@@ -1 +1 @@\n-pub fn x() -> i32 { 1 }\n+pub fn x() -> i32 { 2 }\n";

    let error = super::apply_patch_and_validate(&repo, patch, || {
        std::fs::write(repo.join("crates/x.rs"), "pub fn x() -> i32 { 3 }\n")
            .expect("write same-hunk external edit");
        Err("induced postcondition failure".to_string())
    })
    .expect_err("same-hunk edit must prevent a destructive rollback");

    assert!(
        error.contains("rollback was refused to preserve external edits"),
        "{error}"
    );
    assert!(error.contains("manual recovery is required"), "{error}");
    assert!(error.contains("proposal was retained"), "{error}");
    assert_eq!(
        std::fs::read_to_string(repo.join("crates/x.rs")).unwrap(),
        "pub fn x() -> i32 { 3 }\n"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn apply_is_blocked_by_the_gate_when_checks_are_not_green() {
    // The temp repo has no Cargo.toml, so the quarantine runs no focused check;
    // the manual apply gate must then refuse to apply (never apply unverified).
    let repo = seed_repo("apply-gate");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("a candidate produces a proposal");
    review(&repo, &proposal.proposal_id).expect("review succeeds");
    let error = apply(&repo, &proposal.proposal_id).expect_err("gate must block an unverified patch");
    assert!(
        error.contains("focused_checks_not_green"),
        "error: {error}"
    );
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn apply_requires_the_exact_displayed_digest_and_rejects_substitution() {
    let repo = seed_repo("digest-binding");
    let proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");

    let error = apply(&repo, &"0".repeat(64)).expect_err("stale ID must not approve");
    assert!(error.contains("stale or unknown"), "{error}");

    let proposal_path = repo.join(".zo/dream/improve/last-proposal.json");
    let mut payload: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&proposal_path).unwrap()).unwrap();
    payload["patch_diff"] = serde_json::Value::String("substituted patch".to_string());
    std::fs::write(&proposal_path, serde_json::to_vec(&payload).unwrap()).unwrap();
    let error = apply(&repo, &proposal.proposal_id).expect_err("substituted patch must fail");
    assert!(error.contains("persisted proposal binding"), "{error}");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn final_gate_revalidates_head_before_index_apply() {
    let repo = seed_repo("final-head");
    propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");
    let proposal = load_proposal(&repo).unwrap().unwrap();
    std::fs::write(repo.join("crates/x.rs"), "pub fn x() -> i32 { 3 }\n").unwrap();
    git(&repo, &["add", "crates/x.rs"]);
    git(&repo, &["commit", "-q", "-m", "head moved"]);

    let approval_digest = proposal.approval_digest.clone();
    let error = super::final_apply_revalidation(&repo, &proposal, &approval_digest)
        .expect_err("HEAD move blocks apply");
    assert!(error.contains("HEAD changed"), "{error}");
    let _ = std::fs::remove_dir_all(repo);
}

#[test]
fn proposal_renderer_escapes_all_terminal_controls() {
    let repo = seed_repo("terminal-escape");
    let mut proposal = propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");
    proposal.summary = "bad\u{1b}]8;;https://example.invalid\u{7}name\nnext".to_string();
    proposal.run.changed_paths = vec!["crates/\u{1b}[31mx.rs".to_string()];
    proposal.blocking_reasons = vec!["bad\rreason".to_string()];

    let rendered = format_proposal(&proposal);
    assert!(!rendered.contains('\u{1b}'));
    assert!(!rendered.contains('\r'));
    assert!(rendered.contains("\\u{1b}"), "{rendered}");
    assert!(rendered.contains("\\n"), "{rendered}");
    let _ = std::fs::remove_dir_all(repo);
}

#[cfg(unix)]
#[test]
fn proposal_state_is_owner_only() {
    use std::os::unix::fs::PermissionsExt as _;

    let repo = seed_repo("proposal-permissions");
    propose(&repo, &StubGenerator)
        .expect("propose succeeds")
        .expect("proposal created");
    let improve = repo.join(".zo/dream/improve");
    let proposal = improve.join("last-proposal.json");
    assert_eq!(std::fs::metadata(&improve).unwrap().permissions().mode() & 0o777, 0o700);
    assert_eq!(std::fs::metadata(&proposal).unwrap().permissions().mode() & 0o777, 0o600);
    let _ = std::fs::remove_dir_all(repo);
}
