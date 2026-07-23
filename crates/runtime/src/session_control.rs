use std::env;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::session::{Session, SessionError};

pub const PRIMARY_SESSION_EXTENSION: &str = "jsonl";
pub const LEGACY_SESSION_EXTENSION: &str = "json";
pub const LATEST_SESSION_REFERENCE: &str = "latest";

const SESSION_REFERENCE_ALIASES: &[&str] = &[LATEST_SESSION_REFERENCE, "last", "recent"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionHandle {
    pub id: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedSessionSummary {
    pub id: String,
    pub path: PathBuf,
    pub modified_epoch_millis: u128,
    pub message_count: usize,
    pub parent_session_id: Option<String>,
    pub branch_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedManagedSession {
    pub handle: SessionHandle,
    pub session: Session,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForkedManagedSession {
    pub parent_session_id: String,
    pub handle: SessionHandle,
    pub session: Session,
    pub branch_name: Option<String>,
}

#[derive(Debug)]
pub enum SessionControlError {
    Io(std::io::Error),
    Session(SessionError),
    Format(String),
}

impl Display for SessionControlError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Session(error) => write!(f, "{error}"),
            Self::Format(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SessionControlError {}

impl From<std::io::Error> for SessionControlError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<SessionError> for SessionControlError {
    fn from(value: SessionError) -> Self {
        Self::Session(value)
    }
}

pub fn sessions_dir() -> Result<PathBuf, SessionControlError> {
    managed_sessions_dir_for(env::current_dir()?)
}

pub fn managed_sessions_dir_for(
    base_dir: impl AsRef<Path>,
) -> Result<PathBuf, SessionControlError> {
    let path = managed_sessions_dir_path_for(base_dir);
    fs::create_dir_all(&path)?;
    // Sessions hold prompts and file contents; keep the directory owner-only so
    // other local users cannot read another user's transcripts. Best-effort: the
    // directory may live under a root the process does not own (e.g. a shared
    // `ZO_SESSION_ROOT`), and that must not break session creation.
    let _ = core_types::paths::restrict_permissions_owner_only(&path);
    Ok(path)
}

#[must_use]
pub fn managed_sessions_dir_path_for(base_dir: impl AsRef<Path>) -> PathBuf {
    if let Some(root) = env::var_os("ZO_SESSION_ROOT") {
        if !root.is_empty() {
            return PathBuf::from(root).join("sessions");
        }
    }
    global_project_sessions_dir_for(base_dir)
}

#[must_use]
pub fn global_project_sessions_dir_for(base_dir: impl AsRef<Path>) -> PathBuf {
    global_project_session_dirs_for(base_dir.as_ref())
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            crate::default_config_home()
                .join("projects")
                .join(crate::config::project_slug(base_dir.as_ref()))
                .join("sessions")
        })
}

fn global_project_session_dirs_for(base_dir: &Path) -> Vec<PathBuf> {
    let slug = crate::config::project_slug(base_dir);
    let mut dirs = Vec::new();
    for root in core_types::paths::zo_global_config_roots() {
        push_unique_path(
            &mut dirs,
            root.join("projects").join(&slug).join("sessions"),
        );
    }
    if dirs.is_empty() {
        push_unique_path(
            &mut dirs,
            crate::default_config_home()
                .join("projects")
                .join(slug)
                .join("sessions"),
        );
    }
    dirs
}

#[must_use]
pub fn workspace_sessions_dir_for(base_dir: impl AsRef<Path>) -> PathBuf {
    base_dir.as_ref().join(".zo").join("sessions")
}

#[must_use]
pub fn managed_session_search_dirs_for(base_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(root) = env::var_os("ZO_SESSION_ROOT") {
        if !root.is_empty() {
            push_unique_path(&mut dirs, PathBuf::from(root).join("sessions"));
        }
    }
    for directory in global_project_session_dirs_for(base_dir) {
        push_unique_path(&mut dirs, directory);
    }
    push_unique_path(&mut dirs, workspace_sessions_dir_for(base_dir));
    dirs
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

/// Default retention for session transcripts by last-activity date, mirroring
/// Claude Code's `cleanupPeriodDays` default.
pub const DEFAULT_SESSION_RETENTION_DAYS: u32 = 30;

/// What one retention sweep removed. Counts only — the sweep is best-effort
/// and silent about individual I/O failures (a locked or vanished file is
/// simply left for the next sweep).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionCleanupReport {
    pub removed_sessions: usize,
    pub removed_prefs: usize,
    pub removed_dirs: usize,
    pub reclaimed_bytes: u64,
}

impl SessionCleanupReport {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Delete session transcripts older than `retention_days` (by mtime — resume
/// touches a session, so an actively revisited transcript keeps renewing its
/// lease) across every project slug under the global Zo home, along with
/// each removed session's preferences file. Emptied `sessions/`,
/// `session-prefs/`, and slug directories are pruned afterwards, so orphan
/// slugs (e.g. from a tempdir cwd) age out of `~/.zo/projects/` on their
/// own instead of accumulating forever.
///
/// `session_recall` and `/resume` read these transcripts, so an expired
/// session is no longer recallable — the same trade Claude Code's
/// `cleanupPeriodDays` makes. An active session's file was appended moments
/// ago and can never be older than the cutoff.
#[must_use]
pub fn cleanup_expired_sessions(retention_days: u32) -> SessionCleanupReport {
    // checked_sub: an absurd retention (u32::MAX days from a clamped config
    // value) must degrade to "nothing expires", not panic on pre-epoch time.
    let Some(cutoff) = SystemTime::now().checked_sub(std::time::Duration::from_secs(
        u64::from(retention_days) * 24 * 60 * 60,
    )) else {
        return SessionCleanupReport::default();
    };
    cleanup_expired_sessions_under(&crate::default_config_home().join("projects"), cutoff)
}

/// Env-free core of [`cleanup_expired_sessions`]: sweep `projects_root`
/// removing session files last modified before `cutoff`. Split out so tests
/// drive the cutoff directly instead of manipulating file mtimes.
#[must_use]
pub fn cleanup_expired_sessions_under(
    projects_root: &Path,
    cutoff: SystemTime,
) -> SessionCleanupReport {
    let mut report = SessionCleanupReport::default();
    let Ok(slugs) = fs::read_dir(projects_root) else {
        return report;
    };
    for slug in slugs.flatten() {
        let slug_dir = slug.path();
        if !slug_dir.is_dir() {
            continue;
        }
        let sessions_dir = slug_dir.join("sessions");
        let prefs_dir = slug_dir.join("session-prefs");
        for entry in fs::read_dir(&sessions_dir).into_iter().flatten().flatten() {
            let path = entry.path();
            let is_session_file = path.extension().and_then(|ext| ext.to_str()).is_some_and(
                |ext| ext == PRIMARY_SESSION_EXTENSION || ext == LEGACY_SESSION_EXTENSION,
            );
            if !is_session_file {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let expired = metadata
                .modified()
                .is_ok_and(|modified| modified < cutoff);
            if !expired || fs::remove_file(&path).is_err() {
                continue;
            }
            report.removed_sessions += 1;
            report.reclaimed_bytes += metadata.len();
            // The session's preferences ride its file stem (see
            // session_preferences::preferences_path).
            if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                let prefs = prefs_dir.join(format!("{stem}.json"));
                if let Ok(prefs_metadata) = fs::metadata(&prefs) {
                    if fs::remove_file(&prefs).is_ok() {
                        report.removed_prefs += 1;
                        report.reclaimed_bytes += prefs_metadata.len();
                    }
                }
            }
        }
        // Prune what the sweep emptied. `remove_dir` refuses non-empty
        // directories, so a slug still holding state (todos, memory, live
        // sessions) is never touched.
        for dir in [&sessions_dir, &prefs_dir] {
            if fs::remove_dir(dir).is_ok() {
                report.removed_dirs += 1;
            }
        }
        if fs::remove_dir(&slug_dir).is_ok() {
            report.removed_dirs += 1;
        }
    }
    report
}

pub fn create_managed_session_handle(
    session_id: &str,
) -> Result<SessionHandle, SessionControlError> {
    create_managed_session_handle_for(env::current_dir()?, session_id)
}

pub fn create_managed_session_handle_for(
    base_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<SessionHandle, SessionControlError> {
    let id = session_id.to_string();
    let path =
        managed_sessions_dir_for(base_dir)?.join(format!("{id}.{PRIMARY_SESSION_EXTENSION}"));
    Ok(SessionHandle { id, path })
}

pub fn resolve_session_reference(reference: &str) -> Result<SessionHandle, SessionControlError> {
    resolve_session_reference_for(env::current_dir()?, reference)
}

pub fn resolve_session_reference_for(
    base_dir: impl AsRef<Path>,
    reference: &str,
) -> Result<SessionHandle, SessionControlError> {
    let base_dir = base_dir.as_ref();
    if is_session_reference_alias(reference) {
        let latest = latest_managed_session_for(base_dir)?;
        return Ok(SessionHandle {
            id: latest.id,
            path: latest.path,
        });
    }

    let direct = PathBuf::from(reference);
    let candidate = if direct.is_absolute() {
        direct.clone()
    } else {
        base_dir.join(&direct)
    };
    let looks_like_path = direct.extension().is_some() || direct.components().count() > 1;
    let path = if candidate.exists() {
        candidate
    } else if looks_like_path {
        return Err(SessionControlError::Format(
            format_missing_session_reference(reference),
        ));
    } else {
        resolve_managed_session_path_for(base_dir, reference)?
    };

    Ok(SessionHandle {
        id: session_id_from_path(&path).unwrap_or_else(|| reference.to_string()),
        path,
    })
}

pub fn resolve_managed_session_path(session_id: &str) -> Result<PathBuf, SessionControlError> {
    resolve_managed_session_path_for(env::current_dir()?, session_id)
}

pub fn resolve_managed_session_path_for(
    base_dir: impl AsRef<Path>,
    session_id: &str,
) -> Result<PathBuf, SessionControlError> {
    let base_dir = base_dir.as_ref();
    for directory in managed_session_search_dirs_for(base_dir) {
        for extension in [PRIMARY_SESSION_EXTENSION, LEGACY_SESSION_EXTENSION] {
            let path = directory.join(format!("{session_id}.{extension}"));
            if path.exists() {
                return Ok(path);
            }
        }
    }
    Err(SessionControlError::Format(
        format_missing_session_reference(session_id),
    ))
}

#[must_use]
pub fn is_managed_session_file(path: &Path) -> bool {
    let has_session_extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|extension| {
            extension == PRIMARY_SESSION_EXTENSION || extension == LEGACY_SESSION_EXTENSION
        });
    if !has_session_extension {
        return false;
    }
    // Sidecar files live in the sessions directory and share transcript
    // extensions but are NOT resumable transcripts: the append-only Raw Vault
    // (`<id>.vault.jsonl`), rotated transcript fragments
    // (`<id>.rot-<ts>.jsonl`), per-session todo stores (`<id>.todos.json`), and
    // per-session preference stores (`<id>.prefs.json`). Excluding them keeps
    // them out of session listings and out of `resume latest` (which picks by
    // mtime — sidecars can be touched after their owning transcript and would
    // otherwise win).
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    !name.contains(".vault.")
        && !name.contains(".rot-")
        && !name.ends_with(".todos.json")
        && !name.ends_with(".prefs.json")
}

pub fn list_managed_sessions() -> Result<Vec<ManagedSessionSummary>, SessionControlError> {
    list_managed_sessions_for(env::current_dir()?)
}

pub fn list_managed_sessions_for(
    base_dir: impl AsRef<Path>,
) -> Result<Vec<ManagedSessionSummary>, SessionControlError> {
    let base_dir = base_dir.as_ref();
    let mut sessions = Vec::new();
    for directory in managed_session_search_dirs_for(base_dir) {
        if !directory.exists() {
            continue;
        }
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if !is_managed_session_file(&path) {
                continue;
            }
            let metadata = entry.metadata()?;
            let modified_epoch_millis = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis())
                .unwrap_or_default();
            let (id, message_count, parent_session_id, branch_name) =
                match Session::load_from_path(&path) {
                    Ok(session) => {
                        let parent_session_id = session
                            .fork
                            .as_ref()
                            .map(|fork| fork.parent_session_id.clone());
                        let branch_name = session
                            .fork
                            .as_ref()
                            .and_then(|fork| fork.branch_name.clone());
                        (
                            session.session_id,
                            session.messages.len(),
                            parent_session_id,
                            branch_name,
                        )
                    }
                    Err(_) => (
                        path.file_stem()
                            .and_then(|value| value.to_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        0,
                        None,
                        None,
                    ),
                };
            sessions.push(ManagedSessionSummary {
                id,
                path,
                modified_epoch_millis,
                message_count,
                parent_session_id,
                branch_name,
            });
        }
    }
    sessions.sort_by(|left, right| {
        right
            .modified_epoch_millis
            .cmp(&left.modified_epoch_millis)
            .then_with(|| right.id.cmp(&left.id))
    });
    sessions.dedup_by(|left, right| left.id == right.id);
    Ok(sessions)
}

pub fn latest_managed_session() -> Result<ManagedSessionSummary, SessionControlError> {
    latest_managed_session_for(env::current_dir()?)
}

pub fn latest_managed_session_for(
    base_dir: impl AsRef<Path>,
) -> Result<ManagedSessionSummary, SessionControlError> {
    latest_managed_session_for_excluding(base_dir, None)
}

pub fn latest_managed_session_for_excluding(
    base_dir: impl AsRef<Path>,
    exclude_id: Option<&str>,
) -> Result<ManagedSessionSummary, SessionControlError> {
    list_managed_sessions_for(base_dir)?
        .into_iter()
        .find(|summary| exclude_id.is_none_or(|id| summary.id != id))
        .ok_or_else(|| SessionControlError::Format(format_no_managed_sessions()))
}

pub fn load_managed_session(reference: &str) -> Result<LoadedManagedSession, SessionControlError> {
    load_managed_session_for(env::current_dir()?, reference)
}

pub fn load_managed_session_for(
    base_dir: impl AsRef<Path>,
    reference: &str,
) -> Result<LoadedManagedSession, SessionControlError> {
    load_managed_session_for_excluding(base_dir, reference, None)
}

pub fn load_managed_session_for_excluding(
    base_dir: impl AsRef<Path>,
    reference: &str,
    exclude_id: Option<&str>,
) -> Result<LoadedManagedSession, SessionControlError> {
    let base_dir = base_dir.as_ref();
    let handle = if is_session_reference_alias(reference) {
        let latest = latest_managed_session_for_excluding(base_dir, exclude_id)?;
        SessionHandle {
            id: latest.id,
            path: latest.path,
        }
    } else {
        resolve_session_reference_for(base_dir, reference)?
    };
    let session = Session::load_from_path(&handle.path)?;
    Ok(LoadedManagedSession {
        handle: SessionHandle {
            id: session.session_id.clone(),
            path: handle.path,
        },
        session,
    })
}

pub fn fork_managed_session(
    session: &Session,
    branch_name: Option<String>,
) -> Result<ForkedManagedSession, SessionControlError> {
    fork_managed_session_for(env::current_dir()?, session, branch_name)
}

pub fn fork_managed_session_for(
    base_dir: impl AsRef<Path>,
    session: &Session,
    branch_name: Option<String>,
) -> Result<ForkedManagedSession, SessionControlError> {
    let parent_session_id = session.session_id.clone();
    let forked = session.fork(branch_name);
    let handle = create_managed_session_handle_for(base_dir, &forked.session_id)?;
    let branch_name = forked
        .fork
        .as_ref()
        .and_then(|fork| fork.branch_name.clone());
    let forked = forked.with_persistence_path(handle.path.clone());
    forked.save_to_path(&handle.path)?;
    Ok(ForkedManagedSession {
        parent_session_id,
        handle,
        session: forked,
        branch_name,
    })
}

#[must_use]
pub fn is_session_reference_alias(reference: &str) -> bool {
    SESSION_REFERENCE_ALIASES
        .iter()
        .any(|alias| reference.eq_ignore_ascii_case(alias))
}

fn session_id_from_path(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .and_then(|name| {
            name.strip_suffix(&format!(".{PRIMARY_SESSION_EXTENSION}"))
                .or_else(|| name.strip_suffix(&format!(".{LEGACY_SESSION_EXTENSION}")))
        })
        .map(ToOwned::to_owned)
}

/// Message for a `--resume <ref>` that matched no managed session.
///
/// Single source of truth for this hint — the CLI re-exports it (see
/// `zo_cli::session_format`) so both surfaces stay in sync. The
/// wording reflects the real layout: managed sessions live in the global
/// per-project store or the workspace `.zo/sessions/` directory.
#[must_use]
pub fn format_missing_session_reference(reference: &str) -> String {
    format!(
        "session not found: {reference}\nHint: managed sessions live in your Zo per-project store (~/.zo/projects/<project>/sessions) or the workspace .zo/sessions/ directory. Try `{LATEST_SESSION_REFERENCE}` for the most recent session or `/session list` in the REPL."
    )
}

/// Message when no managed sessions exist at all. Single source of truth (see
/// [`format_missing_session_reference`]); the CLI re-exports it.
#[must_use]
pub fn format_no_managed_sessions() -> String {
    format!(
        "no managed sessions found in your Zo per-project store (~/.zo/projects/<project>/sessions)\nStart `zo` to create a session, then rerun with `--resume {LATEST_SESSION_REFERENCE}`."
    )
}

#[cfg(test)]
mod tests {
    use super::{
        create_managed_session_handle_for, fork_managed_session_for, is_managed_session_file,
        is_session_reference_alias, list_managed_sessions_for, load_managed_session_for,
        load_managed_session_for_excluding, managed_session_search_dirs_for,
        managed_sessions_dir_path_for, resolve_session_reference_for, ManagedSessionSummary,
        LATEST_SESSION_REFERENCE,
    };
    use crate::session::Session;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos();
        let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "runtime-session-control-{}-{counter}-{nanos}",
            std::process::id()
        ))
    }

    fn with_config_home<T>(home: &Path, f: impl FnOnce() -> T) -> T {
        let _lock = crate::test_env_lock();
        let prior_config_home = std::env::var_os("ZO_CONFIG_HOME");
        let prior_session_root = std::env::var_os("ZO_SESSION_ROOT");
        std::env::set_var("ZO_CONFIG_HOME", home);
        std::env::remove_var("ZO_SESSION_ROOT");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        match prior_config_home {
            Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
            None => std::env::remove_var("ZO_CONFIG_HOME"),
        }
        match prior_session_root {
            Some(value) => std::env::set_var("ZO_SESSION_ROOT", value),
            None => std::env::remove_var("ZO_SESSION_ROOT"),
        }
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[test]
    fn search_dirs_include_every_canonical_global_home() {
        let _lock = crate::test_env_lock();
        let root = temp_dir();
        let workspace = root.join("workspace");
        let config_home = root.join("config-home");
        let zo_home = root.join("zo-home");
        let user_home = root.join("user-home");
        let prior_config_home = std::env::var_os("ZO_CONFIG_HOME");
        let prior_zo_home = std::env::var_os("ZO_HOME");
        let prior_home = std::env::var_os("HOME");
        let prior_session_root = std::env::var_os("ZO_SESSION_ROOT");
        std::env::set_var("ZO_CONFIG_HOME", &config_home);
        std::env::set_var("ZO_HOME", &zo_home);
        std::env::set_var("HOME", &user_home);
        std::env::remove_var("ZO_SESSION_ROOT");

        let dirs = managed_session_search_dirs_for(&workspace);

        for (key, value) in [
            ("ZO_CONFIG_HOME", prior_config_home),
            ("ZO_HOME", prior_zo_home),
            ("HOME", prior_home),
            ("ZO_SESSION_ROOT", prior_session_root),
        ] {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }

        let slug = crate::config::project_slug(&workspace);
        assert_eq!(
            dirs,
            vec![
                config_home.join("projects").join(&slug).join("sessions"),
                zo_home.join("projects").join(&slug).join("sessions"),
                user_home
                    .join(".zo")
                    .join("projects")
                    .join(&slug)
                    .join("sessions"),
                user_home
                    .join(".forge")
                    .join("projects")
                    .join(&slug)
                    .join("sessions"),
                workspace.join(".zo").join("sessions"),
            ]
        );
    }

    #[test]
    fn retention_sweep_removes_expired_keeps_fresh_and_prunes_empty_slugs() {
        use super::cleanup_expired_sessions_under;
        let root = temp_dir();
        // Slug A: one expired session (+ prefs) and one fresh session.
        let a_sessions = root.join("slug-a").join("sessions");
        let a_prefs = root.join("slug-a").join("session-prefs");
        fs::create_dir_all(&a_sessions).expect("a sessions");
        fs::create_dir_all(&a_prefs).expect("a prefs");
        fs::write(a_sessions.join("old.jsonl"), "x".repeat(100)).expect("old session");
        fs::write(a_prefs.join("old.json"), "{}").expect("old prefs");
        fs::write(a_sessions.join("new.jsonl"), "y").expect("new session");
        // Slug B: only an expired session — the whole slug should vanish.
        let b_sessions = root.join("slug-b").join("sessions");
        fs::create_dir_all(&b_sessions).expect("b sessions");
        fs::write(b_sessions.join("done.jsonl"), "z").expect("b session");
        // Slug C: no sessions at all, but live state — must never be touched.
        let c_state = root.join("slug-c").join("state");
        fs::create_dir_all(&c_state).expect("c state");
        fs::write(c_state.join("todos.json"), "[]").expect("c todos");

        // Cutoff in the future relative to `old`/`done`, in the past for `new`.
        let now = SystemTime::now();
        let old_time = now - std::time::Duration::from_secs(60);
        // Backdate nothing: instead pick a cutoff between file creation times.
        // All files were just written, so make `new` fresh by rewriting it
        // after choosing the cutoff boundary of "now".
        std::thread::sleep(std::time::Duration::from_millis(20));
        let cutoff = SystemTime::now();
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(a_sessions.join("new.jsonl"), "y2").expect("touch new");
        let _ = old_time;

        let report = cleanup_expired_sessions_under(&root, cutoff);

        assert_eq!(report.removed_sessions, 2, "old + done expire");
        assert_eq!(report.removed_prefs, 1, "old's prefs follow it");
        assert!(!a_sessions.join("old.jsonl").exists());
        assert!(a_sessions.join("new.jsonl").exists(), "fresh session survives");
        assert!(!a_prefs.exists(), "emptied prefs dir is pruned");
        assert!(!root.join("slug-b").exists(), "fully-emptied slug is pruned");
        assert!(
            c_state.join("todos.json").exists(),
            "slugs holding live state are untouched"
        );
        assert!(report.reclaimed_bytes >= 100);

        // Idempotent: nothing left to expire.
        let second = cleanup_expired_sessions_under(&root, cutoff);
        assert!(second.is_empty(), "second sweep finds nothing: {second:?}");
        let _ = fs::remove_dir_all(&root);
    }

    fn persist_session(root: &Path, text: &str) -> Session {
        let mut session = Session::new();
        session
            .push_user_text(text)
            .expect("session message should save");
        let handle = create_managed_session_handle_for(root, &session.session_id)
            .expect("managed session handle should build");
        let session = session.with_persistence_path(handle.path.clone());
        session
            .save_to_path(&handle.path)
            .expect("session should persist");
        session
    }

    fn wait_for_next_millisecond() {
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_millis();
        while SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_millis()
            <= start
        {}
    }

    fn summary_by_id<'a>(
        summaries: &'a [ManagedSessionSummary],
        id: &str,
    ) -> &'a ManagedSessionSummary {
        summaries
            .iter()
            .find(|summary| summary.id == id)
            .expect("session summary should exist")
    }

    #[test]
    fn creates_and_lists_managed_sessions() {
        // given
        let root = temp_dir();
        fs::create_dir_all(&root).expect("root dir should exist");
        let config_home = root.join("home").join(".zo");
        with_config_home(&config_home, || {
            let older = persist_session(&root, "older session");
            wait_for_next_millisecond();
            let newer = persist_session(&root, "newer session");

            // when
            let sessions = list_managed_sessions_for(&root).expect("managed sessions should list");

            // then
            assert_eq!(sessions.len(), 2);
            assert_eq!(sessions[0].id, newer.session_id);
            assert_eq!(summary_by_id(&sessions, &older.session_id).message_count, 1);
            assert_eq!(summary_by_id(&sessions, &newer.session_id).message_count, 1);
            assert!(managed_sessions_dir_path_for(&root).starts_with(&config_home));
        });
        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn lists_skip_vault_rotation_and_todo_sidecars() {
        let root = temp_dir();
        fs::create_dir_all(&root).expect("root dir should exist");
        let config_home = root.join("home").join(".zo");
        with_config_home(&config_home, || {
            let real = persist_session(&root, "real session");
            let dir = managed_sessions_dir_path_for(&root);

            // Sidecars next to the real transcript: a Raw Vault and a rotated
            // fragment. Both share the `.jsonl` extension but must never surface
            // as resumable sessions (a vault is touched on every compaction, so
            // it would otherwise win `resume latest` by mtime).
            fs::write(
                dir.join(format!("{}.vault.jsonl", real.session_id)),
                "{\"type\":\"vault\",\"vault_seq\":0,\"message\":{}}\n",
            )
            .expect("vault sidecar should write");
            fs::write(
                dir.join(format!("{}.rot-123.jsonl", real.session_id)),
                "not a session\n",
            )
            .expect("rotation sidecar should write");
            fs::write(dir.join(format!("{}.todos.json", real.session_id)), "[]")
                .expect("todo sidecar should write");

            let sessions =
                list_managed_sessions_for(&root).expect("managed sessions should list");

            assert_eq!(
                sessions.len(),
                1,
                "only the real transcript lists; vault/rotation/todo sidecars are skipped"
            );
            assert_eq!(sessions[0].id, real.session_id);
        });
        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn sidecars_are_not_managed_session_files() {
        assert!(is_managed_session_file(Path::new("session-1-2.jsonl")));
        assert!(is_managed_session_file(Path::new("session-1-2.json")));
        assert!(!is_managed_session_file(Path::new(
            "session-1-2.vault.jsonl"
        )));
        assert!(!is_managed_session_file(Path::new(
            "session-1-2.rot-123.jsonl"
        )));
        assert!(!is_managed_session_file(Path::new(
            "session-1-2.todos.json"
        )));
        assert!(!is_managed_session_file(Path::new(
            "session-1-2.prefs.json"
        )));
        assert!(!is_managed_session_file(Path::new("notes.txt")));
    }

    #[test]
    fn resolves_latest_alias_and_loads_session_from_workspace_root() {
        // given
        let root = temp_dir();
        fs::create_dir_all(&root).expect("root dir should exist");
        let config_home = root.join("home").join(".zo");
        with_config_home(&config_home, || {
            let older = persist_session(&root, "older session");
            wait_for_next_millisecond();
            let newer = persist_session(&root, "newer session");

            // when
            let handle = resolve_session_reference_for(&root, LATEST_SESSION_REFERENCE)
                .expect("latest alias should resolve");
            let loaded = load_managed_session_for(&root, "recent")
                .expect("recent alias should load the latest session");

            // then
            assert_eq!(handle.id, newer.session_id);
            assert_eq!(loaded.handle.id, newer.session_id);
            assert_eq!(loaded.session.messages.len(), 1);
            assert_ne!(loaded.handle.id, older.session_id);
            assert!(is_session_reference_alias("last"));
        });
        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn latest_alias_ignores_todo_sidecar_when_excluding_current_session() {
        let root = temp_dir();
        fs::create_dir_all(&root).expect("root dir should exist");
        let config_home = root.join("home").join(".zo");
        with_config_home(&config_home, || {
            let prior = persist_session(&root, "prior session");
            wait_for_next_millisecond();
            let current = persist_session(&root, "current session");
            wait_for_next_millisecond();
            let dir = managed_sessions_dir_path_for(&root);
            fs::write(dir.join(format!("{}.todos.json", current.session_id)), "[]")
                .expect("todo sidecar should write");

            let loaded = load_managed_session_for_excluding(
                &root,
                LATEST_SESSION_REFERENCE,
                Some(&current.session_id),
            )
            .expect("latest should skip current session and its sidecars");

            assert_eq!(loaded.handle.id, prior.session_id);
            assert_eq!(loaded.session.messages.len(), 1);
        });
        fs::remove_dir_all(root).expect("temp dir should clean up");
    }

    #[test]
    fn forks_session_into_managed_storage_with_lineage() {
        // given
        let root = temp_dir();
        fs::create_dir_all(&root).expect("root dir should exist");
        let config_home = root.join("home").join(".zo");
        with_config_home(&config_home, || {
            let source = persist_session(&root, "parent session");

            // when
            let forked =
                fork_managed_session_for(&root, &source, Some("incident-review".to_string()))
                    .expect("session should fork");
            let sessions = list_managed_sessions_for(&root).expect("managed sessions should list");
            let summary = summary_by_id(&sessions, &forked.handle.id);

            // then
            assert_eq!(forked.parent_session_id, source.session_id);
            assert_eq!(forked.branch_name.as_deref(), Some("incident-review"));
            assert_eq!(
                summary.parent_session_id.as_deref(),
                Some(source.session_id.as_str())
            );
            assert_eq!(summary.branch_name.as_deref(), Some("incident-review"));
            assert_eq!(
                forked.session.persistence_path(),
                Some(forked.handle.path.as_path())
            );
        });
        fs::remove_dir_all(root).expect("temp dir should clean up");
    }
}
