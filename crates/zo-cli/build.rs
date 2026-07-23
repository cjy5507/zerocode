//! Build-time version stamping for the `zo` binary.
//!
//! Populates the `GIT_SHA`, `TARGET`, and `BUILD_DATE` compile-time env vars
//! that `main.rs` reads through `option_env!`, so `zo --version` reports the
//! real commit, host triple, and build date instead of unfilled placeholders.
//! Pure stdlib — no build dependencies. Each piece degrades to `unknown`
//! rather than failing the build (e.g. a source tarball with no `.git`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-env-changed=ZO_RELEASE_CHANNEL");
    let manifest_dir = PathBuf::from(env_or_default("CARGO_MANIFEST_DIR", "."));

    emit(
        "GIT_SHA",
        &git_sha(&manifest_dir).unwrap_or_else(|| "unknown".to_string()),
    );
    emit("TARGET", &env_or_default("TARGET", "unknown"));
    emit("BUILD_DATE", &build_date());

    rerun_when_head_moves(&manifest_dir);
}

/// Emit a compile-time env var consumed by `option_env!` in `main.rs`.
fn emit(key: &str, value: &str) {
    println!("cargo:rustc-env={key}={value}");
}

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Short commit SHA, suffixed with `-dirty` when the working tree has
/// uncommitted changes. `None` when git is unavailable or this is not a repo.
fn git_sha(dir: &Path) -> Option<String> {
    let sha = run_git(dir, &["rev-parse", "--short=12", "HEAD"])?;
    let dirty = run_git(dir, &["status", "--porcelain"])
        .is_some_and(|status| !status.trim().is_empty());
    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

fn run_git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git").current_dir(dir).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Current UTC date as `YYYY-MM-DD`. This is the binary's build date shown by
/// `--version`; the prompt-context "today" is resolved live at session start
/// (see `default_prompt_date` in `main.rs`).
fn build_date() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let (year, month, day) = civil_from_unix_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Civil (Gregorian) date from days since the Unix epoch (1970-01-01).
/// Howard Hinnant's `civil_from_days`, valid for the full proleptic calendar
/// and dependency-free. All arithmetic stays in `i64` (month/day are small
/// positive values) so there are no lossy casts.
fn civil_from_unix_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day)
}

/// Re-run this script (re-stamping `GIT_SHA`) when the checked-out commit
/// changes: a branch switch rewrites `HEAD`, a new commit rewrites the branch
/// ref, and a `git gc` moves refs into `packed-refs`.
fn rerun_when_head_moves(manifest_dir: &Path) {
    let Some(git_dir) = find_git_dir(manifest_dir) else {
        return;
    };
    watch(&git_dir.join("HEAD"));
    watch(&git_dir.join("packed-refs"));
    if let Some(ref_path) = head_ref_path(&git_dir) {
        watch(&ref_path);
    }
}

fn watch(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// Resolve the `.git` directory by walking up from the crate. Handles both a
/// normal repository (`.git/` directory) and a worktree/submodule (`.git` file
/// containing `gitdir: <path>`).
fn find_git_dir(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let dot_git = ancestor.join(".git");
        if dot_git.is_dir() {
            return Some(dot_git);
        }
        if dot_git.is_file() {
            let contents = std::fs::read_to_string(&dot_git).ok()?;
            let target = contents.strip_prefix("gitdir:")?.trim();
            return Some(ancestor.join(target));
        }
    }
    None
}

/// The on-disk file backing the symbolic ref in `HEAD`, if HEAD points at one.
fn head_ref_path(git_dir: &Path) -> Option<PathBuf> {
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let reference = head.strip_prefix("ref:")?.trim();
    Some(git_dir.join(reference))
}
