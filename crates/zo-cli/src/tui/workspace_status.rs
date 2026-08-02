//! Workspace status sources for the Changes panel.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::sidebar::{
    ChangedFile, FileStatus, GitStatusSnapshot, MAX_SIDEBAR_FILES,
    is_workspace_status_path_filtered,
};

/// A failed or interrupted workspace status scan.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct WorkspaceStatusError {
    message: String,
}

impl WorkspaceStatusError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Source boundary for producing Changes-panel snapshots.
pub trait WorkspaceStatusSource: Send + Sync {
    /// Read one complete snapshot, aborting if a newer workspace event arrives.
    fn snapshot(
        &self,
        cwd: &Path,
        should_interrupt: Arc<AtomicBool>,
    ) -> Result<GitStatusSnapshot, WorkspaceStatusError>;
}

/// Gitoxide-backed workspace status source used by interactive sessions.
pub struct GixStatus;

impl GixStatus {
    /// Verify that `cwd` belongs to a non-bare repository.
    pub fn open(cwd: &Path) -> Result<Self, WorkspaceStatusError> {
        let repo = gix::discover(cwd).map_err(|error| WorkspaceStatusError::new(error.to_string()))?;
        if repo.workdir().is_none() {
            return Err(WorkspaceStatusError::new(
                "bare repositories do not have workspace status",
            ));
        }
        Ok(Self)
    }
}

impl WorkspaceStatusSource for GixStatus {
    fn snapshot(
        &self,
        cwd: &Path,
        should_interrupt: Arc<AtomicBool>,
    ) -> Result<GitStatusSnapshot, WorkspaceStatusError> {
        gix_status_snapshot(cwd, &should_interrupt)
    }
}

/// Subprocess-backed compatibility source retained as the per-session fallback.
pub struct GitCliStatus;

impl WorkspaceStatusSource for GitCliStatus {
    fn snapshot(
        &self,
        cwd: &Path,
        should_interrupt: Arc<AtomicBool>,
    ) -> Result<GitStatusSnapshot, WorkspaceStatusError> {
        let snapshot = fetch_git_cli_status(cwd);
        if should_interrupt.load(Ordering::Acquire) {
            Err(WorkspaceStatusError::new("git CLI status scan interrupted"))
        } else {
            Ok(snapshot)
        }
    }
}

/// Select gitoxide once for the session, falling back permanently if opening fails.
#[must_use]
pub fn session_workspace_status_source(cwd: &Path) -> Arc<dyn WorkspaceStatusSource> {
    match GixStatus::open(cwd) {
        Ok(source) => Arc::new(source),
        Err(error) => {
            eprintln!(
                "[zo] gitoxide status unavailable ({error}); using git CLI for this session"
            );
            Arc::new(GitCliStatus)
        }
    }
}

fn gix_status_snapshot(
    cwd: &Path,
    should_interrupt: &Arc<AtomicBool>,
) -> Result<GitStatusSnapshot, WorkspaceStatusError> {
    let repo = gix::discover(cwd).map_err(|error| WorkspaceStatusError::new(error.to_string()))?;
    let platform = repo
        .status(gix::progress::Discard)
        .map_err(|error| WorkspaceStatusError::new(error.to_string()))?
        .tree_index_track_renames(gix::status::tree_index::TrackRenames::Disabled)
        .index_worktree_rewrites(None)
        .should_interrupt_owned(Arc::clone(should_interrupt));
    let mut iter = platform
        .into_iter(std::iter::empty::<gix::bstr::BString>())
        .map_err(|error| WorkspaceStatusError::new(error.to_string()))?;
    let mut statuses = BTreeMap::<String, (FileStatus, bool)>::new();

    for item in iter.by_ref() {
        let item = item.map_err(|error| WorkspaceStatusError::new(error.to_string()))?;
        let path = String::from_utf8_lossy(item.location()).into_owned();
        if is_workspace_status_path_filtered(&path) {
            continue;
        }
        let (status, tracked_for_diff) = match item {
            gix::status::Item::IndexWorktree(change) => {
                let tracked_for_diff = !matches!(
                    &change,
                    gix::status::index_worktree::Item::DirectoryContents { .. }
                );
                let Some(summary) = change.summary() else {
                    continue;
                };
                let status = match summary {
                    gix::status::index_worktree::iter::Summary::Removed => FileStatus::Deleted,
                    gix::status::index_worktree::iter::Summary::Added
                    | gix::status::index_worktree::iter::Summary::IntentToAdd => FileStatus::Added,
                    gix::status::index_worktree::iter::Summary::Modified
                    | gix::status::index_worktree::iter::Summary::TypeChange
                    | gix::status::index_worktree::iter::Summary::Renamed
                    | gix::status::index_worktree::iter::Summary::Copied
                    | gix::status::index_worktree::iter::Summary::Conflict => FileStatus::Modified,
                };
                (status, tracked_for_diff)
            }
            gix::status::Item::TreeIndex(change) => (
                match change {
                    gix::diff::index::ChangeRef::Addition { .. } => FileStatus::Added,
                    gix::diff::index::ChangeRef::Deletion { .. } => FileStatus::Deleted,
                    gix::diff::index::ChangeRef::Modification { .. }
                    | gix::diff::index::ChangeRef::Rewrite { .. } => FileStatus::Modified,
                },
                true,
            ),
        };
        statuses
            .entry(path)
            .and_modify(|(current, tracked)| {
                *current = combined_status(*current, status);
                *tracked |= tracked_for_diff;
            })
            .or_insert((status, tracked_for_diff));
    }

    if iter.into_outcome().is_none() {
        return Err(WorkspaceStatusError::new("gitoxide status scan interrupted"));
    }

    let total = statuses.len();
    let selected = statuses
        .into_iter()
        .take(MAX_SIDEBAR_FILES)
        .collect::<Vec<_>>();
    let tracked_for_diff = selected
        .iter()
        .filter(|(_, (_, tracked))| *tracked)
        .map(|(path, _)| path.clone())
        .collect::<HashSet<_>>();
    let mut files = selected
        .into_iter()
        .map(|(path, (status, _))| ChangedFile {
            path,
            status,
            adds: 0,
            rems: 0,
        })
        .collect::<Vec<_>>();
    merge_gix_line_tallies(&repo, &mut files, &tracked_for_diff, should_interrupt)?;
    Ok(GitStatusSnapshot { files, total })
}

fn combined_status(left: FileStatus, right: FileStatus) -> FileStatus {
    if left == FileStatus::Deleted || right == FileStatus::Deleted {
        FileStatus::Deleted
    } else if left == FileStatus::Added || right == FileStatus::Added {
        FileStatus::Added
    } else {
        FileStatus::Modified
    }
}

fn merge_gix_line_tallies(
    repo: &gix::Repository,
    files: &mut [ChangedFile],
    tracked_for_diff: &HashSet<String>,
    should_interrupt: &AtomicBool,
) -> Result<(), WorkspaceStatusError> {
    let Some(worktree_root) = repo.workdir().map(Path::to_path_buf) else {
        return Ok(());
    };
    let head_tree = repo.head_tree().ok();
    let mut cache = repo
        .diff_resource_cache(
            gix::diff::blob::pipeline::Mode::ToGit,
            gix::diff::blob::pipeline::WorktreeRoots {
                old_root: None,
                new_root: Some(worktree_root),
            },
        )
        .map_err(|error| WorkspaceStatusError::new(error.to_string()))?;

    for file in files {
        if should_interrupt.load(Ordering::Acquire) {
            return Err(WorkspaceStatusError::new("gitoxide status scan interrupted"));
        }
        if !tracked_for_diff.contains(&file.path) {
            continue;
        }
        let relative_path = Path::new(&file.path);
        let head_entry = head_tree
            .as_ref()
            .and_then(|tree| tree.lookup_entry_by_path(relative_path).ok().flatten());
        let old_id = head_entry
            .as_ref()
            .map_or_else(|| repo.object_hash().null(), gix::object::tree::Entry::object_id);
        let mode = head_entry
            .as_ref()
            .map_or(gix::objs::tree::EntryKind::Blob, |entry| entry.mode().kind());
        let path = gix::bstr::BString::from(file.path.as_bytes());

        if cache
            .set_resource(
                old_id,
                mode,
                path.as_ref(),
                gix::diff::blob::ResourceKind::OldOrSource,
                repo,
            )
            .is_err()
            || cache
                .set_resource(
                    repo.object_hash().null(),
                    mode,
                    path.as_ref(),
                    gix::diff::blob::ResourceKind::NewOrDestination,
                    repo,
                )
                .is_err()
        {
            continue;
        }
        let mut diff = gix::object::blob::diff::Platform {
            resource_cache: &mut cache,
        };
        if let Ok(Some(counts)) = diff.line_counts() {
            file.adds = usize::try_from(counts.insertions).unwrap_or(usize::MAX);
            file.rems = usize::try_from(counts.removals).unwrap_or(usize::MAX);
        }
    }
    Ok(())
}

fn fetch_git_cli_status(cwd: &Path) -> GitStatusSnapshot {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["status", "--porcelain=v1", "-z", "--no-renames"])
        .output();
    let Ok(output) = output else {
        return GitStatusSnapshot::EMPTY;
    };
    if !output.status.success() {
        return GitStatusSnapshot::EMPTY;
    }
    let mut snapshot = parse_porcelain_status(&output.stdout);
    merge_cli_line_tallies(cwd, &mut snapshot);
    snapshot
}

/// Fold per-file line magnitude (`+N -M` vs HEAD) into the CLI snapshot.
fn merge_cli_line_tallies(cwd: &Path, snapshot: &mut GitStatusSnapshot) {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["diff", "--numstat", "--no-renames", "HEAD"])
        .output();
    let Ok(output) = output else { return };
    if !output.status.success() {
        return;
    }
    let tallies = parse_numstat(&output.stdout);
    for file in &mut snapshot.files {
        if let Some(&(adds, rems)) = tallies.get(&file.path) {
            file.adds = adds;
            file.rems = rems;
        }
    }
}

fn parse_numstat(raw: &[u8]) -> HashMap<String, (usize, usize)> {
    let mut map = HashMap::new();
    for line in String::from_utf8_lossy(raw).lines() {
        let mut parts = line.splitn(3, '\t');
        let (Some(adds), Some(rems), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        map.insert(
            path.to_string(),
            (adds.parse().unwrap_or(0), rems.parse().unwrap_or(0)),
        );
    }
    map
}

fn parse_porcelain_status(raw: &[u8]) -> GitStatusSnapshot {
    let mut files = Vec::new();
    let mut total = 0;
    let mut rest = raw;

    while !rest.is_empty() {
        let end = rest.iter().position(|&byte| byte == 0).unwrap_or(rest.len());
        let entry = &rest[..end];
        rest = if end < rest.len() {
            &rest[end + 1..]
        } else {
            &[]
        };
        if entry.len() < 4 {
            continue;
        }
        let status_bytes = &entry[..2];
        let is_rename = status_bytes.contains(&b'R') || status_bytes.contains(&b'C');
        let path = String::from_utf8_lossy(&entry[3..]).into_owned();
        if !is_workspace_status_path_filtered(&path) {
            total += 1;
            if files.len() < MAX_SIDEBAR_FILES {
                files.push(ChangedFile {
                    path,
                    status: file_status_from_porcelain(status_bytes),
                    adds: 0,
                    rems: 0,
                });
            }
        }
        if is_rename {
            let skip_end = rest.iter().position(|&byte| byte == 0).unwrap_or(rest.len());
            rest = if skip_end < rest.len() {
                &rest[skip_end + 1..]
            } else {
                &[]
            };
        }
    }

    files.sort_unstable_by(|left, right| left.path.cmp(&right.path));
    GitStatusSnapshot { files, total }
}

fn file_status_from_porcelain(status: &[u8]) -> FileStatus {
    if status[0] == b'D' || status[1] == b'D' {
        FileStatus::Deleted
    } else if status[0] == b'A' || status[1] == b'A' || status[0] == b'?' || status[1] == b'?' {
        FileStatus::Added
    } else {
        FileStatus::Modified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn gix_and_cli_status_match_staged_unstaged_untracked_and_rename_matrix() {
        if Command::new("git").arg("--version").output().is_err() {
            eprintln!("skipping status equivalence test: git is unavailable");
            return;
        }
        let repo = std::env::temp_dir().join(format!(
            "zo-workspace-status-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&repo).expect("temp repo");
        git(&repo, &["init"]);
        git(&repo, &["config", "user.email", "zo@example.com"]);
        git(&repo, &["config", "user.name", "Zo Test"]);
        std::fs::write(repo.join("both.txt"), "one\ntwo\n").expect("seed both");
        std::fs::write(repo.join("unstaged.txt"), "before\n").expect("seed unstaged");
        std::fs::write(repo.join("old.txt"), "rename me\n").expect("seed rename");
        std::fs::write(repo.join("deleted.txt"), "delete me\n").expect("seed delete");
        std::fs::write(repo.join(".gitignore"), "ignored.txt\n").expect("seed ignore");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-m", "seed"]);

        std::fs::write(repo.join("both.txt"), "one staged\ntwo\n").expect("stage modification");
        git(&repo, &["add", "both.txt"]);
        std::fs::write(repo.join("both.txt"), "one staged\ntwo unstaged\n").expect("unstaged tail");
        std::fs::write(repo.join("unstaged.txt"), "after\n").expect("unstaged modification");
        std::fs::write(repo.join("untracked.txt"), "new\n").expect("untracked file");
        std::fs::write(repo.join("staged-new.txt"), "staged new\n").expect("staged new");
        git(&repo, &["add", "staged-new.txt"]);
        git(&repo, &["mv", "old.txt", "renamed.txt"]);
        std::fs::remove_file(repo.join("deleted.txt")).expect("delete tracked");
        std::fs::write(repo.join("ignored.txt"), "ignored\n").expect("ignored file");

        let interrupt = Arc::new(AtomicBool::new(false));
        let cli = GitCliStatus
            .snapshot(&repo, Arc::clone(&interrupt))
            .expect("CLI snapshot");
        let gix = GixStatus::open(&repo)
            .expect("open gix")
            .snapshot(&repo, interrupt)
            .expect("gix snapshot");
        assert_eq!(gix, cli);

        std::fs::remove_dir_all(repo).ok();
    }

    fn git(repo: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
