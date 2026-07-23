use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};
use std::time::Duration;

use runtime::{ConfigLoader, RuntimeShipConfig};
use sha2::{Digest, Sha256};

const OUTPUT_TAIL_LINES: usize = 40;
const PUSH_RETRY_DELAY: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub(crate) struct ShipResult {
    pub(crate) report: String,
    pub(crate) success: bool,
}

#[derive(Debug)]
struct GateRecord {
    number: usize,
    command: String,
    status: ExitStatus,
    tail: String,
}

#[derive(Debug)]
struct CommandCapture {
    status: ExitStatus,
    text: String,
}

pub(crate) fn handle_ship_at(
    cwd: &Path,
    message: &str,
    progress: impl FnMut(String),
) -> ShipResult {
    let config = match ConfigLoader::default_for(cwd).load() {
        Ok(config) => config,
        Err(error) => {
            return ShipResult::failure(format!(
                "Ship refused\n  Reason            failed to load settings: {error}"
            ));
        }
    };
    run_ship_with_config(cwd, message, config.ship(), PUSH_RETRY_DELAY, progress)
}

#[allow(clippy::too_many_lines)] // The foreground gate-to-push transaction stays linear so no step can detach or reorder.
fn run_ship_with_config(
    cwd: &Path,
    message: &str,
    config: &RuntimeShipConfig,
    push_retry_delay: Duration,
    mut progress: impl FnMut(String),
) -> ShipResult {
    if message.trim().is_empty() {
        return ShipResult::failure(
            "Ship refused\n  Reason            commit message is required".to_string(),
        );
    }
    if let Err(reason) = ensure_git_worktree(cwd) {
        return ShipResult::failure(format!("Ship refused\n  Reason            {reason}"));
    }

    let captured = match capture_file_set(cwd) {
        Ok(paths) => paths,
        Err(reason) => {
            return ShipResult::failure(format!("Ship refused\n  Reason            {reason}"));
        }
    };
    match git_operation_in_progress(cwd) {
        Ok(Some(operation)) => {
            return ShipResult::failure(format!(
                "Ship refused\n  Reason            {operation} is in progress"
            ));
        }
        Ok(None) => {}
        Err(reason) => {
            return ShipResult::failure(format!("Ship refused\n  Reason            {reason}"));
        }
    }
    if captured.is_empty() {
        return ShipResult::failure(
            "Ship refused\n  Reason            working tree has no modified or added paths".to_string(),
        );
    }
    let captured_fingerprint = match fingerprint_capture(cwd, &captured) {
        Ok(fingerprint) => fingerprint,
        Err(reason) => {
            return ShipResult::failure(format!("Ship refused\n  Reason            {reason}"));
        }
    };
    progress(format!(
        "Ship\n  Captured files    {}\n  Status            running configured gates",
        captured.len()
    ));

    let mut gate_records = Vec::with_capacity(config.gates().len());
    for (index, gate) in config.gates().iter().enumerate() {
        progress(format!(
            "Ship gate {}/{}\n  Command           {gate}\n  Status            running",
            index + 1,
            config.gates().len()
        ));
        let output = match run_shell(cwd, gate) {
            Ok(output) => output,
            Err(error) => {
                let mut report = format!(
                    "Ship aborted\n  Reason            gate {} could not start: {error}\n  Gate              {gate}",
                    index + 1
                );
                append_gate_tails(&mut report, &gate_records);
                return ShipResult::failure(report);
            }
        };
        let record = GateRecord {
            number: index + 1,
            command: gate.clone(),
            status: output.status,
            tail: verbatim_tail(&output.text),
        };
        let passed = record.status.success();
        gate_records.push(record);
        if !passed {
            let failed = gate_records.last().expect("just pushed failed gate");
            let mut report = format!(
                "Ship aborted\n  Reason            gate {} failed ({})\n  Gate              {}\n  Staging           not started\n  Commit            not started",
                failed.number,
                display_status(failed.status),
                failed.command
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
        progress(format!(
            "Ship gate {}/{}\n  Status            passed",
            index + 1,
            config.gates().len()
        ));
    }

    progress("Ship\n  Status            checking concurrent workspace drift".to_string());
    let after_gates = match capture_file_set(cwd) {
        Ok(paths) => paths,
        Err(reason) => {
            let mut report = format!(
                "Ship aborted\n  Reason            could not re-capture paths after gates: {reason}"
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    if after_gates != captured {
        let mut report = "Ship aborted\n  Reason            modified-file set changed while gates ran\n  Staging           not started\n  Commit            not started".to_string();
        append_path_list(&mut report, "Captured files before gates", &captured);
        append_path_list(&mut report, "Files after gates", &after_gates);
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    // Same path set is not enough: a gate (or a concurrent session) can rewrite
    // the CONTENT of an already-captured path, and committing it would ship
    // bytes the gates never validated.
    match fingerprint_capture(cwd, &captured) {
        Ok(after) if after == captured_fingerprint => {}
        Ok(_) => {
            let mut report = "Ship aborted\n  Reason            captured file contents or modes changed while gates ran\n  Staging           not started\n  Commit            not started".to_string();
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
        Err(reason) => {
            let mut report = format!(
                "Ship aborted\n  Reason            could not re-fingerprint captured paths after gates: {reason}"
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    }

    progress(format!(
        "Ship\n  Status            staging exactly {} captured paths",
        captured.len()
    ));
    let index_snapshot = match snapshot_index(cwd) {
        Ok(tree) => tree,
        Err(reason) => {
            let mut report = format!(
                "Ship aborted\n  Reason            {reason}\n  Staging           not started\n  Commit            not started"
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    let staged_output = match stage_paths(cwd, &captured) {
        Ok(output) => output,
        Err(error) => {
            let mut report = format!(
                "Ship aborted\n  Reason            git add could not start: {error}\n  Commit            not started"
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    if !staged_output.status.success() {
        let rolled_back = restore_index(cwd, &index_snapshot);
        let mut report = format!(
            "Ship aborted\n  Reason            git add failed ({})\n  Staging           {}\n  Commit            not started",
            display_status(staged_output.status),
            if rolled_back { "rolled back" } else { "rollback failed" }
        );
        append_verbatim_tail(&mut report, "git add", &verbatim_tail(&staged_output.text));
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }

    let staged = match capture_staged_file_set(cwd) {
        Ok(paths) => paths,
        Err(reason) => {
            let rolled_back = restore_index(cwd, &index_snapshot);
            let mut report = format!(
                "Ship aborted\n  Reason            staged-count guard failed to inspect the index: {reason}\n  Staging           {}\n  Commit            not started",
                if rolled_back { "rolled back" } else { "rollback failed" }
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    if staged.len() != captured.len() || staged != captured {
        // Restore the pre-staging index so an aborted ship leaves no
        // half-staged state behind (best-effort).
        let rolled_back = restore_index(cwd, &index_snapshot);
        let mut report = format!(
            "Ship aborted\n  Reason            staged paths do not match captured paths\n  Captured count    {}\n  Staged count      {}\n  Staging           {}\n  Commit            not started",
            captured.len(),
            staged.len(),
            if rolled_back { "rolled back" } else { "rollback failed" }
        );
        append_path_list(&mut report, "Captured files", &captured);
        append_path_list(&mut report, "Staged files", &staged);
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    // `git commit` snapshots the INDEX, so prove the index still holds exactly
    // the bytes the gates validated: the staged entries must match the
    // worktree for every captured path AND the worktree must still match the
    // pre-gate fingerprint. This closes the window between the post-gate check
    // and `git add`; the remaining race (another process rewriting the index
    // between here and commit) requires a concurrent git writer, which the
    // merge/rebase refusal already treats as out of scope.
    let staged_matches = staged_matches_worktree(cwd, &captured);
    let refreshed = fingerprint_capture(cwd, &captured);
    let staged_is_validated = matches!(&staged_matches, Ok(true))
        && matches!(&refreshed, Ok(fingerprint) if *fingerprint == captured_fingerprint);
    if !staged_is_validated {
        let rolled_back = restore_index(cwd, &index_snapshot);
        let reason = match (&staged_matches, &refreshed) {
            (Err(reason), _) | (_, Err(reason)) => {
                format!("could not re-verify staged content: {reason}")
            }
            _ => "staged content does not match the gate-validated capture".to_string(),
        };
        let mut report = format!(
            "Ship aborted\n  Reason            {reason}\n  Staging           {}\n  Commit            not started",
            if rolled_back { "rolled back" } else { "rollback failed" }
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    // Pin the publication target now: the ship pushes the created commit BY
    // SHA to this branch's configured upstream, so a hook or concurrent
    // writer moving HEAD after the commit cannot smuggle unvalidated history
    // into the push.
    let branch_ref = match run_git(cwd, ["symbolic-ref", "--quiet", "HEAD"]) {
        Ok(output) if output.status.success() => output.text.trim().to_string(),
        _ => {
            let rolled_back = restore_index(cwd, &index_snapshot);
            let mut report = format!(
                "Ship aborted\n  Reason            HEAD is detached; /ship requires a branch checkout\n  Staging           {}\n  Commit            not started",
                if rolled_back { "rolled back" } else { "rollback failed" }
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    // `%(push:...)` honors branch.<name>.pushRemote / remote.pushDefault /
    // push.default (triangular workflows); fall back to the fetch upstream
    // when git cannot resolve a push-specific destination.
    let push_remote = branch_field(cwd, &branch_ref, "%(push:remotename)")
        .or_else(|| branch_field(cwd, &branch_ref, "%(upstream:remotename)"));
    let push_target_ref = branch_field(cwd, &branch_ref, "%(push:remoteref)")
        .or_else(|| branch_field(cwd, &branch_ref, "%(upstream:remoteref)"));
    let (Some(push_remote), Some(push_target_ref)) = (push_remote, push_target_ref) else {
        let rolled_back = restore_index(cwd, &index_snapshot);
        let mut report = format!(
            "Ship aborted\n  Reason            no push destination is configured for {branch_ref}\n  Staging           {}\n  Commit            not started",
            if rolled_back { "rolled back" } else { "rollback failed" }
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    };
    // Record the rollback point and the exact tree the validated index holds
    // right now: after `git commit`, the created commit must carry this tree
    // or it is rolled back (commit hooks can mutate the index mid-commit).
    let head_before_commit = match run_git(cwd, ["rev-parse", "HEAD"]) {
        Ok(output) if output.status.success() => output.text.trim().to_string(),
        _ => {
            let rolled_back = restore_index(cwd, &index_snapshot);
            let mut report = format!(
                "Ship aborted\n  Reason            could not record HEAD before committing\n  Staging           {}\n  Commit            not started",
                if rolled_back { "rolled back" } else { "rollback failed" }
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    let expected_tree = match run_git(cwd, ["write-tree"]) {
        Ok(output) if output.status.success() => output.text.trim().to_string(),
        _ => {
            let rolled_back = restore_index(cwd, &index_snapshot);
            let mut report = format!(
                "Ship aborted\n  Reason            could not record the gate-validated staged tree\n  Staging           {}\n  Commit            not started",
                if rolled_back { "rolled back" } else { "rollback failed" }
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };

    progress("Ship\n  Status            committing captured paths".to_string());
    let commit_output = match Command::new("git")
        .current_dir(cwd)
        .arg("commit")
        .arg("-m")
        .arg(message)
        .output()
    {
        Ok(output) => capture_output(output),
        Err(error) => {
            let rolled_back = restore_index(cwd, &index_snapshot);
            let mut report = format!(
                "Ship aborted\n  Reason            git commit could not start: {error}\n  Staging           {}\n  Push              not started",
                if rolled_back { "rolled back" } else { "rollback failed" }
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    if !commit_output.status.success() {
        // The commit did not land (e.g. a rejecting pre-commit hook), so put
        // back the exact index the user had before the ship staged its
        // capture (best-effort).
        let rolled_back = restore_index(cwd, &index_snapshot);
        let mut report = format!(
            "Ship aborted\n  Reason            git commit failed ({})\n  Staging           {}\n  Push              not started",
            display_status(commit_output.status),
            if rolled_back { "rolled back" } else { "rollback failed" }
        );
        append_verbatim_tail(
            &mut report,
            "git commit",
            &verbatim_tail(&commit_output.text),
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    // Successful commit hooks may still mutate the index or extend history
    // during `git commit` (a formatter re-staging files, a commit-creating
    // post-commit hook). Observe HEAD once and require it to (a) carry the
    // gate-validated staged tree and (b) sit directly on the pre-ship HEAD;
    // anything else is rolled back and the push refused. The push publishes
    // this observed commit BY SHA, so later HEAD movement cannot change what
    // is published.
    let observed = match run_git(cwd, ["rev-parse", "HEAD", "HEAD^{tree}", "HEAD^"]) {
        Ok(output) if output.status.success() => {
            let lines = output.text.lines().map(str::to_string).collect::<Vec<_>>();
            (lines.len() == 3).then_some(lines)
        }
        _ => None,
    };
    let Some(observed) = observed else {
        let rolled_back = restore_index(cwd, &index_snapshot);
        let mut report = format!(
            "Ship aborted\n  Reason            could not inspect the created commit\n  Commit            left in place\n  Staging           {}\n  Push              refused",
            if rolled_back { "rolled back" } else { "rollback failed" }
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    };
    let (ship_commit, observed_tree, observed_parent) = (&observed[0], &observed[1], &observed[2]);
    if *observed_tree != expected_tree || *observed_parent != head_before_commit {
        // Compare-and-swap rollback: rewind the branch ONLY if it still names
        // the exact commit observed above, so a concurrent writer's later
        // update is never discarded.
        let commit_undone = run_git(
            cwd,
            [
                "update-ref",
                branch_ref.as_str(),
                head_before_commit.as_str(),
                ship_commit.as_str(),
            ],
        )
        .is_ok_and(|output| output.status.success());
        // Restore the index only when the rewind actually happened: a failed
        // CAS means a concurrent writer owns the branch now, and their index
        // state must not be overwritten with our stale snapshot. (No
        // deterministic repro exists — nothing can interleave between the
        // in-process observation and the CAS without an external writer.)
        let rolled_back = commit_undone && restore_index(cwd, &index_snapshot);
        let reason = if *observed_tree == expected_tree {
            "the created commit does not sit directly on the pre-ship HEAD (a commit-creating hook or concurrent writer?)"
        } else {
            "commit tree does not match the gate-validated staged tree (an index-mutating commit hook?)"
        };
        let mut report = format!(
            "Ship aborted\n  Reason            {reason}\n  Commit            {}\n  Staging           {}\n  Push              refused",
            if commit_undone {
                "rolled back"
            } else {
                "left in place (branch moved concurrently)"
            },
            if rolled_back {
                "rolled back"
            } else if commit_undone {
                "rollback failed"
            } else {
                "left untouched"
            }
        );
        append_verbatim_tail(
            &mut report,
            "git commit",
            &verbatim_tail(&commit_output.text),
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    let ship_commit = ship_commit.clone();

    progress("Ship\n  Status            pushing commit".to_string());
    // The branch must still name the exact observed ship commit when the push
    // starts; movement means a concurrent writer owns the branch and the ship
    // result can no longer be attributed safely. (The push below names the
    // ship commit BY SHA, so movement after this check cannot change the
    // published content either way.)
    if !branch_still_at(cwd, &branch_ref, &ship_commit) {
        let mut report = format!(
            "Ship incomplete\n  Commit            created ({ship_commit})\n  Reason            the branch moved past the ship commit before the push (concurrent writer?)\n  Push              refused"
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    let push_refspec = format!("{ship_commit}:{push_target_ref}");
    let first_push = match run_git(cwd, ["push", push_remote.as_str(), push_refspec.as_str()]) {
        Ok(output) => output,
        Err(error) => {
            let mut report = format!(
                "Ship incomplete\n  Commit            created\n  Reason            git push could not start: {error}"
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    };
    let (push_output, first_push_failure) = if first_push.status.success() {
        (first_push, None)
    } else {
        progress(format!(
            "Ship\n  Push              failed ({})\n  Status            retrying in {}s",
            display_status(first_push.status),
            push_retry_delay.as_secs()
        ));
        std::thread::sleep(push_retry_delay);
        // Recheck before EVERY attempt: the branch may have moved during the
        // retry delay.
        if !branch_still_at(cwd, &branch_ref, &ship_commit) {
            let mut report = format!(
                "Ship incomplete\n  Commit            created ({ship_commit})\n  Reason            the branch moved past the ship commit before the push retry (concurrent writer?)\n  Push              refused"
            );
            append_verbatim_tail(
                &mut report,
                "git push attempt 1",
                &verbatim_tail(&first_push.text),
            );
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
        let retry = match run_git(cwd, ["push", push_remote.as_str(), push_refspec.as_str()]) {
            Ok(output) => output,
            Err(error) => {
                let mut report = format!(
                    "Ship incomplete\n  Commit            created\n  Reason            push retry could not start: {error}"
                );
                append_verbatim_tail(
                    &mut report,
                    "git push attempt 1",
                    &verbatim_tail(&first_push.text),
                );
                append_gate_tails(&mut report, &gate_records);
                return ShipResult::failure(report);
            }
        };
        (retry, Some(first_push))
    };
    if !push_output.status.success() {
        let mut report = format!(
            "Ship incomplete\n  Commit            created\n  Reason            git push failed after one retry ({})",
            display_status(push_output.status)
        );
        if let Some(first) = &first_push_failure {
            append_verbatim_tail(
                &mut report,
                "git push attempt 1",
                &verbatim_tail(&first.text),
            );
        }
        append_verbatim_tail(
            &mut report,
            "git push attempt 2",
            &verbatim_tail(&push_output.text),
        );
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }
    // Final contract check: the branch must still name the ship commit after
    // a successful push. (Movement inside the push-to-recheck window has no
    // deterministic repro; the pre-attempt rechecks are the testable
    // contract.)
    if !branch_still_at(cwd, &branch_ref, &ship_commit) {
        let mut report = format!(
            "Ship incomplete\n  Commit            created ({ship_commit})\n  Push              succeeded\n  Reason            the branch moved past the ship commit during the push (concurrent writer?)"
        );
        append_verbatim_tail(&mut report, "git push", &verbatim_tail(&push_output.text));
        append_gate_tails(&mut report, &gate_records);
        return ShipResult::failure(report);
    }

    let deploy_output = if let Some(deploy) = config.deploy() {
        progress(format!(
            "Ship deploy\n  Command           {deploy}\n  Status            running"
        ));
        match run_shell(cwd, deploy) {
            Ok(output) => Some((deploy, output)),
            Err(error) => {
                let mut report = format!(
                    "Ship incomplete\n  Commit            created\n  Push              succeeded\n  Reason            deploy could not start: {error}"
                );
                append_gate_tails(&mut report, &gate_records);
                return ShipResult::failure(report);
            }
        }
    } else {
        None
    };

    if let Some((deploy, output)) = &deploy_output {
        if !output.status.success() {
            let mut report = format!(
                "Ship incomplete\n  Commit            created\n  Push              succeeded\n  Reason            deploy failed ({})\n  Deploy            {deploy}",
                display_status(output.status)
            );
            append_verbatim_tail(&mut report, "deploy", &verbatim_tail(&output.text));
            append_gate_tails(&mut report, &gate_records);
            return ShipResult::failure(report);
        }
    }

    let mut report = format!(
        "Ship complete\n  Captured files    {}\n  Staged count      {}\n  Commit            created\n  Push              succeeded\n  Deploy            {}",
        captured.len(),
        staged.len(),
        if deploy_output.is_some() {
            "succeeded"
        } else {
            "not configured"
        }
    );
    append_verbatim_tail(
        &mut report,
        "git commit",
        &verbatim_tail(&commit_output.text),
    );
    append_verbatim_tail(&mut report, "git push", &verbatim_tail(&push_output.text));
    if let Some((_, output)) = &deploy_output {
        append_verbatim_tail(&mut report, "deploy", &verbatim_tail(&output.text));
    }
    append_gate_tails(&mut report, &gate_records);
    ShipResult::success(report)
}

impl ShipResult {
    fn success(report: String) -> Self {
        Self {
            report,
            success: true,
        }
    }

    fn failure(report: String) -> Self {
        Self {
            report,
            success: false,
        }
    }
}

fn ensure_git_worktree(cwd: &Path) -> Result<(), String> {
    let output = run_git(cwd, ["rev-parse", "--is-inside-work-tree"])
        .map_err(|error| format!("could not run git: {error}"))?;
    if output.status.success() && output.text.trim() == "true" {
        Ok(())
    } else {
        Err("current directory is not a git worktree".to_string())
    }
}

fn git_operation_in_progress(cwd: &Path) -> Result<Option<&'static str>, String> {
    for (marker, operation) in [
        ("MERGE_HEAD", "merge"),
        ("rebase-merge", "rebase"),
        ("rebase-apply", "rebase"),
    ] {
        let output = run_git(cwd, ["rev-parse", "--git-path", marker])
            .map_err(|error| format!("could not inspect git operation state: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "could not inspect git operation state ({})",
                display_status(output.status)
            ));
        }
        let marker_path = PathBuf::from(output.text.trim());
        let marker_path = if marker_path.is_absolute() {
            marker_path
        } else {
            cwd.join(marker_path)
        };
        if marker_path.exists() {
            return Ok(Some(operation));
        }
    }
    Ok(None)
}

fn capture_file_set(cwd: &Path) -> Result<BTreeSet<PathBuf>, String> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .output()
        .map_err(|error| format!("could not run git status: {error}"))?;
    if !output.status.success() {
        let captured = capture_output(output);
        return Err(format!(
            "git status failed ({}): {}",
            display_status(captured.status),
            verbatim_tail(&captured.text)
        ));
    }
    parse_porcelain_paths(&output.stdout)
}

fn parse_porcelain_paths(bytes: &[u8]) -> Result<BTreeSet<PathBuf>, String> {
    let entries = bytes.split(|byte| *byte == 0).collect::<Vec<_>>();
    let mut paths = BTreeSet::new();
    let mut index = 0;
    while index < entries.len() {
        let record = entries[index];
        if record.is_empty() {
            index += 1;
            continue;
        }
        if record.len() < 4 || record[2] != b' ' {
            return Err("git status returned an unexpected porcelain record".to_string());
        }
        paths.insert(path_from_git_bytes(&record[3..])?);
        let is_rename_or_copy = matches!(record[0], b'R' | b'C') || matches!(record[1], b'R' | b'C');
        if is_rename_or_copy {
            index += 1;
            let Some(source) = entries.get(index).filter(|entry| !entry.is_empty()) else {
                return Err("git status returned an incomplete rename record".to_string());
            };
            paths.insert(path_from_git_bytes(source)?);
        }
        index += 1;
    }
    Ok(paths)
}

fn capture_staged_file_set(cwd: &Path) -> Result<BTreeSet<PathBuf>, String> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(["diff", "--cached", "--name-only", "--no-renames", "-z"])
        .output()
        .map_err(|error| format!("could not inspect staged paths: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git diff --cached failed ({})",
            display_status(output.status)
        ));
    }
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .map(path_from_git_bytes)
        .collect()
}

fn stage_paths(cwd: &Path, paths: &BTreeSet<PathBuf>) -> std::io::Result<CommandCapture> {
    let mut command = Command::new("git");
    // Pathspec magic (`:(glob)`, `:(top)`, …) survives `--`; literal mode
    // keeps a file literally named like a magic prefix from expanding to
    // unrelated paths.
    command
        .current_dir(cwd)
        .env("GIT_LITERAL_PATHSPECS", "1")
        .args(["add", "-A", "--"]);
    command.args(paths);
    command.output().map(capture_output)
}

/// Order-stable digest of HEAD plus the kind and content of every captured
/// path. The set-equality drift guard reports readable path diffs, but only
/// this fingerprint catches a gate (or concurrent session) rewriting the
/// CONTENT of an already-captured path — which would commit bytes the gates
/// never validated.
fn fingerprint_capture(cwd: &Path, paths: &BTreeSet<PathBuf>) -> Result<String, String> {
    let head = run_git(cwd, ["rev-parse", "HEAD"])
        .map_err(|error| format!("could not read HEAD for the drift fingerprint: {error}"))?;
    if !head.status.success() {
        return Err(format!(
            "could not read HEAD for the drift fingerprint ({})",
            display_status(head.status)
        ));
    }
    let mut hasher = Sha256::new();
    hasher.update(head.text.trim().as_bytes());
    for path in paths {
        let absolute = cwd.join(path);
        hasher.update([0u8]);
        hasher.update(path.as_os_str().as_encoded_bytes());
        hasher.update([0u8]);
        match std::fs::symlink_metadata(&absolute) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                hasher.update(b"link:");
                match std::fs::read_link(&absolute) {
                    Ok(target) => {
                        let target = target.as_os_str().as_encoded_bytes();
                        hasher.update(
                            u64::try_from(target.len()).unwrap_or(u64::MAX).to_le_bytes(),
                        );
                        hasher.update(target);
                    }
                    Err(_) => hasher.update(b"<unreadable>"),
                }
            }
            Ok(metadata) if metadata.is_dir() => {
                // A directory with a `.git` entry is a submodule working
                // tree: what `git add` stages for it is the checked-out
                // commit (a mode-160000 gitlink), so that commit joins the
                // digest instead of a constant directory tag.
                if absolute.join(".git").exists() {
                    hasher.update(b"gitlink:");
                    match Command::new("git")
                        .current_dir(&absolute)
                        .args(["rev-parse", "HEAD"])
                        .output()
                    {
                        Ok(output) if output.status.success() => {
                            hasher.update(output.stdout.trim_ascii());
                        }
                        _ => hasher.update(b"<unreadable>"),
                    }
                } else {
                    hasher.update(b"dir");
                }
            }
            Ok(metadata) => {
                // Git tracks exactly one permission — the executable bit
                // (100644 vs 100755). Frame it unambiguously: a mode tag
                // hashed for BOTH states plus a content length prefix, so no
                // content byte-string can imitate a different mode/content
                // split (e.g. non-executable "x:payload" vs executable
                // "payload").
                let executable = {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt as _;
                        metadata.permissions().mode() & 0o111 != 0
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = &metadata;
                        false
                    }
                };
                hasher.update(if executable {
                    b"file mode=1:"
                } else {
                    b"file mode=0:"
                });
                match std::fs::read(&absolute) {
                    Ok(bytes) => {
                        hasher
                            .update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
                        hasher.update(&bytes);
                    }
                    Err(_) => hasher.update(b"<unreadable>"),
                }
            }
            Err(_) => hasher.update(b"absent"),
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Path of the active index file for this worktree (`--git-path` resolves
/// linked worktrees).
fn index_file_path(cwd: &Path) -> Result<PathBuf, String> {
    let output = run_git(cwd, ["rev-parse", "--git-path", "index"])
        .map_err(|error| format!("could not locate the git index: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "could not locate the git index ({})",
            display_status(output.status)
        ));
    }
    let path = PathBuf::from(output.text.trim());
    Ok(if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    })
}

/// Byte-exact snapshot of the pre-staging index file so any abort after
/// `git add` can restore EXACTLY the index the user had — including partially
/// staged hunks and index-only state (intent-to-add, assume-unchanged,
/// skip-worktree) that a tree-level `git read-tree` restore would lose.
fn snapshot_index(cwd: &Path) -> Result<Vec<u8>, String> {
    let path = index_file_path(cwd)?;
    std::fs::read(&path)
        .map_err(|error| format!("could not snapshot the index before staging: {error}"))
}

/// Best-effort byte-exact restore of the index snapshot taken before staging.
/// Takes git's own `index.lock` exclusively and swaps the bytes in with an
/// atomic same-directory rename, refusing without touching the index while a
/// concurrent git process holds the lock. (A torn write mid-restore cannot be
/// reproduced deterministically; the lock refusal and rename path are the
/// testable contract.)
fn restore_index(cwd: &Path, snapshot: &[u8]) -> bool {
    let Ok(path) = index_file_path(cwd) else {
        return false;
    };
    let Some(parent) = path.parent() else {
        return false;
    };
    let lock = parent.join("index.lock");
    if std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .is_err()
    {
        return false;
    }
    let temp = parent.join(format!("index.zo-ship-{}", std::process::id()));
    let swapped = std::fs::write(&temp, snapshot).is_ok() && std::fs::rename(&temp, &path).is_ok();
    if !swapped {
        let _ = std::fs::remove_file(&temp);
    }
    let _ = std::fs::remove_file(&lock);
    swapped
}

/// Whether the branch still names the exact observed ship commit.
fn branch_still_at(cwd: &Path, branch_ref: &str, ship_commit: &str) -> bool {
    run_git(cwd, ["rev-parse", branch_ref])
        .is_ok_and(|output| output.status.success() && output.text.trim() == ship_commit)
}

/// One `%(...)` for-each-ref field of the branch, `None` when unset.
fn branch_field(cwd: &Path, branch_ref: &str, format: &str) -> Option<String> {
    let format_arg = format!("--format={format}");
    let output = run_git(cwd, ["for-each-ref", format_arg.as_str(), branch_ref]).ok()?;
    if !output.status.success() {
        return None;
    }
    let value = output.text.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// Whether the staged index entries match the worktree for every captured
/// path (`git diff --quiet` compares the index against the worktree).
fn staged_matches_worktree(cwd: &Path, paths: &BTreeSet<PathBuf>) -> Result<bool, String> {
    let mut command = Command::new("git");
    command
        .current_dir(cwd)
        .env("GIT_LITERAL_PATHSPECS", "1")
        .args(["diff", "--quiet", "--"]);
    command.args(paths);
    let output = command
        .output()
        .map_err(|error| format!("could not compare the index to the worktree: {error}"))?;
    match output.status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(format!(
            "git diff --quiet failed ({})",
            display_status(output.status)
        )),
    }
}

fn run_git<const N: usize>(cwd: &Path, args: [&str; N]) -> std::io::Result<CommandCapture> {
    Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .map(capture_output)
}

#[cfg(not(windows))]
fn run_shell(cwd: &Path, command: &str) -> std::io::Result<CommandCapture> {
    Command::new("sh")
        .current_dir(cwd)
        .args(["-lc", "exec 2>&1; eval \"$1\"", "zo-ship", command])
        .output()
        .map(capture_output)
}

#[cfg(windows)]
fn run_shell(cwd: &Path, command: &str) -> std::io::Result<CommandCapture> {
    Command::new("cmd")
        .current_dir(cwd)
        .args(["/D", "/S", "/C", command])
        .output()
        .map(capture_output)
}

fn capture_output(output: Output) -> CommandCapture {
    let Output {
        status,
        stdout,
        stderr,
    } = output;
    let mut text = String::from_utf8_lossy(&stdout).into_owned();
    if !stderr.is_empty() {
        text.push_str(&String::from_utf8_lossy(&stderr));
    }
    CommandCapture { status, text }
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)] // Windows rejects non-UTF-8 paths through this shared fallible signature.
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    use std::os::unix::ffi::OsStringExt as _;
    Ok(PathBuf::from(OsString::from_vec(bytes.to_vec())))
}

#[cfg(not(unix))]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    String::from_utf8(bytes.to_vec())
        .map(PathBuf::from)
        .map_err(|_| "git returned a non-UTF-8 path".to_string())
}

fn verbatim_tail(text: &str) -> String {
    let lines = text.split_inclusive('\n').collect::<Vec<_>>();
    let start = lines.len().saturating_sub(OUTPUT_TAIL_LINES);
    lines[start..].concat()
}

fn display_status(status: ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| "terminated by signal".to_string(), |code| format!("exit {code}"))
}

fn append_path_list(report: &mut String, title: &str, paths: &BTreeSet<PathBuf>) {
    let _ = write!(report, "\n\n{title} ({}):", paths.len());
    if paths.is_empty() {
        report.push_str("\n  (none)");
        return;
    }
    for path in paths {
        let _ = write!(report, "\n  {}", path.display());
    }
}

fn append_gate_tails(report: &mut String, records: &[GateRecord]) {
    for record in records {
        append_verbatim_tail(
            report,
            &format!(
                "gate {} ({}, {})",
                record.number,
                record.command,
                display_status(record.status)
            ),
            &record.tail,
        );
    }
}

fn append_verbatim_tail(report: &mut String, label: &str, tail: &str) {
    let _ = write!(report, "\n\n--- {label} output tail (verbatim) ---\n");
    if tail.is_empty() {
        report.push_str("<empty>\n");
    } else {
        report.push_str(tail);
        if !tail.ends_with('\n') {
            report.push('\n');
        }
    }
    let _ = write!(report, "--- end {label} output tail ---");
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestRepo {
        root: PathBuf,
        repo: PathBuf,
    }

    impl TestRepo {
        fn new() -> Self {
            let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "zo-ship-test-{}-{sequence}",
                std::process::id()
            ));
            let repo = root.join("repo");
            let remote = root.join("remote.git");
            fs::create_dir_all(&repo).expect("create test repo");
            git(&repo, &["init", "-q"]);
            git(&repo, &["config", "user.name", "Zo Ship Test"]);
            git(&repo, &["config", "user.email", "ship@example.com"]);
            fs::write(repo.join("tracked.txt"), "base\n").expect("write tracked file");
            git(&repo, &["add", "tracked.txt"]);
            git(&repo, &["commit", "-q", "-m", "base"]);
            git(&root, &["init", "--bare", "-q", remote.to_str().expect("utf8 remote")]);
            git(
                &repo,
                &["remote", "add", "origin", remote.to_str().expect("utf8 remote")],
            );
            git(&repo, &["push", "-q", "-u", "origin", "HEAD"]);
            Self { root, repo }
        }

        fn head(&self) -> String {
            git_stdout(&self.repo, &["rev-parse", "HEAD"])
                .trim()
                .to_string()
        }
    }

    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn gate_failure_reports_verbatim_tail_without_staging_or_commit() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let head_before = test.head();
        let config = RuntimeShipConfig::new(
            vec!["printf 'gate-red-line\\n'; false".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must not commit",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success);
        assert!(result.report.contains("gate-red-line\n"));
        assert!(result.report.contains("Staging           not started"));
        assert_eq!(test.head(), head_before);
        assert!(git_status(&test.repo, &["diff", "--cached", "--quiet"])
                .status
                .success());
    }

    #[test]
    fn gate_created_file_triggers_concurrent_drift_guard() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let head_before = test.head();
        let config = RuntimeShipConfig::new(
            vec!["printf drift > drift.txt; echo drift-gate-tail".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must not commit drift",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success);
        assert!(result.report.contains("modified-file set changed while gates ran"));
        assert!(result.report.contains("Captured files before gates (1):"));
        assert!(result.report.contains("Files after gates (2):"));
        assert!(result.report.contains("drift.txt"));
        assert!(result.report.contains("drift-gate-tail\n"));
        assert_eq!(test.head(), head_before);
        assert!(git_status(&test.repo, &["diff", "--cached", "--quiet"])
                .status
                .success());
    }

    #[test]
    fn green_flow_commits_exact_capture_and_round_trips_quoted_message() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        fs::write(test.repo.join("new file.txt"), "new\n").expect("write added file");
        let message = r#"release "quoted" candidate; touch message-was-interpolated"#;
        let config = RuntimeShipConfig::new(
            vec!["true".to_string(), "echo green-gate-tail".to_string()],
            Some("echo deploy-tail".to_string()),
        );

        let result = run_ship_with_config(
            &test.repo,
            message,
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(result.success, "{}", result.report);
        assert_eq!(
            git_stdout(&test.repo, &["log", "-1", "--pretty=%B"]).trim_end(),
            message
        );
        assert!(!test.repo.join("message-was-interpolated").exists());
        let committed = git_stdout_bytes(
            &test.repo,
            &["show", "--pretty=format:", "--name-only", "--no-renames", "-z", "HEAD"],
        );
        let committed = committed
            .split(|byte| *byte == 0)
            .filter(|entry| !entry.is_empty())
            .map(path_from_git_bytes)
            .collect::<Result<BTreeSet<_>, _>>()
            .expect("parse committed paths");
        assert_eq!(
            committed,
            BTreeSet::from([PathBuf::from("new file.txt"), PathBuf::from("tracked.txt")])
        );
        assert!(result.report.contains("Staged count      2"));
        assert!(result.report.contains("green-gate-tail\n"));
        assert!(result.report.contains("deploy-tail\n"));
        assert!(git_status(&test.repo, &["status", "--porcelain"]).stdout.is_empty());
    }

    #[test]
    fn gate_modified_content_of_captured_file_triggers_drift_guard() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let head_before = test.head();
        // The gate rewrites an ALREADY-CAPTURED path: the path set stays
        // identical, so only a content fingerprint can catch the drift.
        let config = RuntimeShipConfig::new(
            vec!["printf tampered > tracked.txt; echo content-drift-tail".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must not commit tampered content",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result.report.contains("changed while gates ran"),
            "{}",
            result.report
        );
        assert_eq!(test.head(), head_before);
        assert!(git_status(&test.repo, &["diff", "--cached", "--quiet"])
            .status
            .success());
    }

    #[test]
    fn gate_mode_change_of_captured_file_triggers_drift_guard() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let head_before = test.head();
        // The gate flips only the executable bit: bytes and path set are
        // unchanged, so the fingerprint must cover the git-tracked mode.
        let config = RuntimeShipConfig::new(
            vec!["chmod +x tracked.txt; echo mode-drift-tail".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must not commit tampered mode",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result.report.contains("changed while gates ran"),
            "{}",
            result.report
        );
        assert_eq!(test.head(), head_before);
        assert!(git_status(&test.repo, &["diff", "--cached", "--quiet"])
            .status
            .success());
    }

    #[test]
    fn submodule_commit_drift_during_gates_triggers_drift_guard() {
        let test = TestRepo::new();
        // Embedded repo acting as a submodule: the parent tracks it as a
        // gitlink (mode 160000) whose value is the checked-out commit.
        let sub = test.repo.join("sub");
        fs::create_dir_all(&sub).expect("create submodule dir");
        git(&sub, &["init", "-q"]);
        git(&sub, &["config", "user.name", "Zo Ship Test"]);
        git(&sub, &["config", "user.email", "ship@example.com"]);
        fs::write(sub.join("inner.txt"), "one\n").expect("write inner file");
        git(&sub, &["add", "inner.txt"]);
        git(&sub, &["commit", "-q", "-m", "sub one"]);
        git(&test.repo, &["add", "sub"]);
        git(&test.repo, &["commit", "-q", "-m", "add gitlink"]);
        // Move the submodule ahead so the parent captures it as modified.
        git(&sub, &["commit", "-q", "--allow-empty", "-m", "sub two"]);
        let head_before = test.head();
        // The gate advances the submodule again: the path set and a plain
        // directory tag are unchanged — only the gitlink commit moved.
        let config = RuntimeShipConfig::new(
            vec!["git -C sub commit -q --allow-empty -m 'sub three'".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must not commit unvalidated gitlink",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result.report.contains("changed while gates ran"),
            "{}",
            result.report
        );
        assert_eq!(test.head(), head_before);
    }

    #[test]
    fn index_mutating_commit_hook_cannot_alter_the_shipped_tree() {
        use std::os::unix::fs::PermissionsExt as _;

        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let git_dir = git_stdout(&test.repo, &["rev-parse", "--git-dir"]);
        let hook_dir = test.repo.join(git_dir.trim()).join("hooks");
        fs::create_dir_all(&hook_dir).expect("create hooks dir");
        let hook = hook_dir.join("pre-commit");
        // The hook SUCCEEDS but smuggles an extra staged path into the index,
        // so the created commit's tree differs from the validated capture.
        fs::write(
            &hook,
            "#!/bin/sh\nprintf smuggled > hook-smuggled.txt\ngit add hook-smuggled.txt\nexit 0\n",
        )
        .expect("write pre-commit hook");
        let mut perms = fs::metadata(&hook).expect("hook metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&hook, perms).expect("chmod hook");
        let head_before = test.head();
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        let result = run_ship_with_config(
            &test.repo,
            "hook must not smuggle content",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result
                .report
                .contains("does not match the gate-validated staged tree"),
            "{}",
            result.report
        );
        assert_eq!(test.head(), head_before);
    }

    #[test]
    fn same_tree_head_advance_by_commit_hook_refuses_push() {
        use std::os::unix::fs::PermissionsExt as _;

        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let git_dir = git_stdout(&test.repo, &["rev-parse", "--git-dir"]);
        let hook_dir = test.repo.join(git_dir.trim()).join("hooks");
        fs::create_dir_all(&hook_dir).expect("create hooks dir");
        let hook = hook_dir.join("post-commit");
        // The hook advances HEAD with an empty commit carrying the SAME tree,
        // so a tree-only guard would still publish the extra, unvalidated
        // commit. (The marker file keeps the hook from recursing.)
        fs::write(
            &hook,
            "#!/bin/sh\nif [ ! -f hook-ran ]; then touch hook-ran; git commit -q --allow-empty -m extra; fi\nexit 0\n",
        )
        .expect("write post-commit hook");
        let mut perms = fs::metadata(&hook).expect("hook metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&hook, perms).expect("chmod hook");
        let head_before = test.head();
        let remote = test.root.join("remote.git");
        let remote_head_before = git_stdout(&remote, &["rev-parse", "HEAD"]);
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        let result = run_ship_with_config(
            &test.repo,
            "hook must not extend history",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result
                .report
                .contains("does not sit directly on the pre-ship HEAD"),
            "{}",
            result.report
        );
        assert_eq!(test.head(), head_before);
        assert_eq!(
            git_stdout(&remote, &["rev-parse", "HEAD"]),
            remote_head_before
        );
    }

    #[test]
    fn restore_index_refuses_while_git_holds_the_index_lock() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "staged version\n").expect("write staged version");
        git(&test.repo, &["add", "tracked.txt"]);
        let snapshot = snapshot_index(&test.repo).expect("snapshot index");
        fs::write(test.repo.join("tracked.txt"), "worktree version\n")
            .expect("write worktree version");
        git(&test.repo, &["add", "tracked.txt"]);
        let index_lock = index_file_path(&test.repo)
            .expect("locate index")
            .with_file_name("index.lock");
        fs::write(&index_lock, b"").expect("hold index lock");
        // A concurrent git writer holds the lock: the restore must refuse and
        // leave the index untouched instead of racing the writer.
        assert!(!restore_index(&test.repo, &snapshot));
        assert_eq!(
            git_stdout(&test.repo, &["show", ":tracked.txt"]),
            "worktree version\n"
        );
        fs::remove_file(&index_lock).expect("release index lock");
        assert!(restore_index(&test.repo, &snapshot));
        assert_eq!(
            git_stdout(&test.repo, &["show", ":tracked.txt"]),
            "staged version\n"
        );
    }

    #[test]
    fn push_honors_configured_push_remote_over_fetch_upstream() {
        let test = TestRepo::new();
        let publish = test.root.join("publish.git");
        git(
            &test.root,
            &["init", "--bare", "-q", publish.to_str().expect("utf8 publish")],
        );
        git(
            &test.repo,
            &["remote", "add", "publish", publish.to_str().expect("utf8 publish")],
        );
        let branch = git_stdout(&test.repo, &["symbolic-ref", "--short", "HEAD"]);
        let push_remote_key = format!("branch.{}.pushRemote", branch.trim());
        git(&test.repo, &["config", &push_remote_key, "publish"]);
        git(&test.repo, &["config", "push.default", "current"]);
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let origin = test.root.join("remote.git");
        let origin_head_before = git_stdout(&origin, &["rev-parse", "HEAD"]);
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        let result = run_ship_with_config(
            &test.repo,
            "ship to the configured push remote",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(result.success, "{}", result.report);
        assert_eq!(
            git_stdout(&publish, &["rev-parse", "HEAD"]).trim(),
            test.head()
        );
        assert_eq!(
            git_stdout(&origin, &["rev-parse", "HEAD"]),
            origin_head_before
        );
    }

    #[test]
    fn branch_movement_between_commit_and_push_refuses_push() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let remote = test.root.join("remote.git");
        let remote_head_before = git_stdout(&remote, &["rev-parse", "HEAD"]);
        let repo = test.repo.clone();
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        // A "concurrent writer" advances the branch inside the
        // observation-to-push window, which the progress callback exposes
        // deterministically.
        let result = run_ship_with_config(
            &test.repo,
            "must not push after branch movement",
            &config,
            Duration::ZERO,
            |progress| {
                if progress.contains("pushing commit") {
                    let output = Command::new("git")
                        .current_dir(&repo)
                        .args(["commit", "-q", "--allow-empty", "-m", "concurrent"])
                        .output()
                        .expect("advance branch");
                    assert!(output.status.success());
                }
            },
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result.report.contains("moved past the ship commit"),
            "{}",
            result.report
        );
        assert_eq!(
            git_stdout(&remote, &["rev-parse", "HEAD"]),
            remote_head_before
        );
    }

    #[test]
    fn branch_movement_during_push_retry_delay_refuses_retry() {
        let test = TestRepo::new();
        // Make the remote one commit AHEAD so the first push attempt fails
        // (non-fast-forward), opening the retry window deterministically.
        git(&test.repo, &["commit", "-q", "--allow-empty", "-m", "ahead"]);
        git(&test.repo, &["push", "-q"]);
        git(&test.repo, &["reset", "-q", "--hard", "HEAD^"]);
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let remote = test.root.join("remote.git");
        let remote_head_before = git_stdout(&remote, &["rev-parse", "HEAD"]);
        let repo = test.repo.clone();
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        // The "concurrent writer" advances the branch inside the retry delay,
        // which the retry progress message exposes deterministically.
        let result = run_ship_with_config(
            &test.repo,
            "must not retry after branch movement",
            &config,
            Duration::ZERO,
            |progress| {
                if progress.contains("retrying in") {
                    let output = Command::new("git")
                        .current_dir(&repo)
                        .args(["commit", "-q", "--allow-empty", "-m", "concurrent"])
                        .output()
                        .expect("advance branch");
                    assert!(output.status.success());
                }
            },
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result
                .report
                .contains("moved past the ship commit before the push retry"),
            "{}",
            result.report
        );
        assert_eq!(
            git_stdout(&remote, &["rev-parse", "HEAD"]),
            remote_head_before
        );
    }

    #[test]
    fn mode_content_collision_cannot_slip_past_the_fingerprint() {
        let test = TestRepo::new();
        // Framing-collision attempt: under naive tagging a NON-executable
        // file containing "x:payload" and an executable file containing
        // "payload" hash identically ("file:" + "x:" + content).
        fs::write(test.repo.join("tracked.txt"), "x:payload").expect("modify tracked file");
        let head_before = test.head();
        let config = RuntimeShipConfig::new(
            vec!["printf payload > tracked.txt; chmod +x tracked.txt".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must not commit the collision",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(
            result.report.contains("changed while gates ran"),
            "{}",
            result.report
        );
        assert_eq!(test.head(), head_before);
    }

    #[test]
    fn pathspec_magic_filename_is_staged_literally() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        // `--` does not disable git pathspec magic; only literal-pathspec mode
        // keeps a file literally named like a magic prefix from expanding.
        fs::write(test.repo.join(":(top)magic.txt"), "literal\n").expect("write magic-named file");
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        let result = run_ship_with_config(
            &test.repo,
            "ship literal magic filename",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(result.success, "{}", result.report);
        let committed = git_stdout_bytes(
            &test.repo,
            &[
                "show",
                "--pretty=format:",
                "--name-only",
                "--no-renames",
                "-z",
                "HEAD",
            ],
        );
        let committed = committed
            .split(|byte| *byte == 0)
            .filter(|entry| !entry.is_empty())
            .map(path_from_git_bytes)
            .collect::<Result<BTreeSet<_>, _>>()
            .expect("parse committed paths");
        assert_eq!(
            committed,
            BTreeSet::from([
                PathBuf::from(":(top)magic.txt"),
                PathBuf::from("tracked.txt")
            ])
        );
    }

    #[test]
    fn merge_in_progress_refuses_before_running_gates() {
        let test = TestRepo::new();
        fs::write(test.repo.join("tracked.txt"), "changed\n").expect("modify tracked file");
        let git_dir = git_stdout(&test.repo, &["rev-parse", "--git-dir"]);
        fs::write(test.repo.join(git_dir.trim()).join("MERGE_HEAD"), test.head())
            .expect("create merge marker");
        let config = RuntimeShipConfig::new(
            vec!["touch gate-must-not-run".to_string()],
            None,
        );

        let result = run_ship_with_config(
            &test.repo,
            "must refuse merge",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success);
        assert!(result.report.contains("merge is in progress"));
        assert!(!test.repo.join("gate-must-not-run").exists());
    }

    #[test]
    fn aborted_staging_restores_pre_staged_index_content() {
        let test = TestRepo::new();
        // Tracked second file whose index-only assume-unchanged flag must
        // survive the rollback byte-for-byte.
        fs::write(test.repo.join("flagged.txt"), "flag\n").expect("write flagged file");
        git(&test.repo, &["add", "flagged.txt"]);
        git(&test.repo, &["commit", "-q", "-m", "flagged base"]);
        git(&test.repo, &["update-index", "--assume-unchanged", "flagged.txt"]);
        // The user pre-staged one version, then kept editing the worktree: an
        // aborted ship must put back exactly the staged version, not HEAD (a
        // whole-index `git reset` would lose the staged selection).
        fs::write(test.repo.join("tracked.txt"), "staged version\n").expect("write staged version");
        git(&test.repo, &["add", "tracked.txt"]);
        fs::write(test.repo.join("tracked.txt"), "worktree version\n")
            .expect("write worktree version");

        let snapshot = snapshot_index(&test.repo).expect("snapshot index");
        git(&test.repo, &["add", "-A"]);
        assert_eq!(
            git_stdout(&test.repo, &["show", ":tracked.txt"]),
            "worktree version\n"
        );
        assert!(restore_index(&test.repo, &snapshot));
        assert_eq!(
            git_stdout(&test.repo, &["show", ":tracked.txt"]),
            "staged version\n"
        );
        let flags = git_stdout(&test.repo, &["ls-files", "-v", "flagged.txt"]);
        assert!(flags.starts_with('h'), "index-only flag lost: {flags}");
    }

    #[test]
    fn failing_commit_hook_restores_pre_staged_index() {
        use std::os::unix::fs::PermissionsExt as _;

        let test = TestRepo::new();
        // The user pre-staged one version and kept editing; when the commit
        // itself fails (here: a rejecting pre-commit hook), the abort must
        // restore the pre-ship index instead of leaving the ship's staging.
        fs::write(test.repo.join("tracked.txt"), "staged version\n").expect("write staged version");
        git(&test.repo, &["add", "tracked.txt"]);
        fs::write(test.repo.join("tracked.txt"), "worktree version\n")
            .expect("write worktree version");
        let git_dir = git_stdout(&test.repo, &["rev-parse", "--git-dir"]);
        let hook_dir = test.repo.join(git_dir.trim()).join("hooks");
        fs::create_dir_all(&hook_dir).expect("create hooks dir");
        let hook = hook_dir.join("pre-commit");
        fs::write(&hook, "#!/bin/sh\nexit 1\n").expect("write pre-commit hook");
        let mut perms = fs::metadata(&hook).expect("hook metadata").permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&hook, perms).expect("chmod hook");
        let head_before = test.head();
        let config = RuntimeShipConfig::new(vec!["true".to_string()], None);

        let result = run_ship_with_config(
            &test.repo,
            "hook must reject this commit",
            &config,
            Duration::ZERO,
            |_| {},
        );

        assert!(!result.success, "{}", result.report);
        assert!(result.report.contains("git commit failed"), "{}", result.report);
        assert_eq!(test.head(), head_before);
        assert_eq!(
            git_stdout(&test.repo, &["show", ":tracked.txt"]),
            "staged version\n"
        );
    }

    fn git(cwd: &Path, args: &[&str]) {
        let output = git_status(cwd, args);
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn git_stdout(cwd: &Path, args: &[&str]) -> String {
        String::from_utf8(git_stdout_bytes(cwd, args)).expect("git output is utf8")
    }

    fn git_stdout_bytes(cwd: &Path, args: &[&str]) -> Vec<u8> {
        let output = git_status(cwd, args);
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    fn git_status(cwd: &Path, args: &[&str]) -> Output {
        Command::new("git")
            .current_dir(cwd)
            .args(args)
            .output()
            .expect("run git")
    }
}
