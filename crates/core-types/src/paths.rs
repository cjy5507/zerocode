//! Canonical resolution of zo's per-user state home (`~/.zo`).
//!
//! Single source of truth for the `ZO_CONFIG_HOME` → `ZO_HOME` →
//! `$HOME/.zo` chain, with a read-only legacy `$HOME/.forge` fallback
//! appended last. Before this module, the credential stores, the log path,
//! and the config tools each re-implemented the chain and drifted: all of
//! them silently ignored `ZO_HOME` (splitting user state away from the
//! configured home) and one read the variable UTF-8-only, dropping non-UTF-8
//! paths. `runtime::config` delegates here so every crate — including `api`,
//! which cannot depend on `runtime` — resolves the same directories.

use rand::Rng as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Environment variable naming the highest-priority zo home.
pub const ZO_CONFIG_HOME_ENV: &str = "ZO_CONFIG_HOME";
/// Secondary home override honored after [`ZO_CONFIG_HOME_ENV`].
pub const ZO_HOME_ENV: &str = "ZO_HOME";
/// Directory name of the conventional per-user home under `$HOME`, and of
/// the per-project state directory under a workspace root.
pub const ZO_DIR_NAME: &str = ".zo";

const LEGACY_FORGE_DIR_NAME: &str = ".forge";

/// All per-user global config homes, highest priority first: the canonical
/// `ZO_CONFIG_HOME` → `ZO_HOME` → `~/.zo` chain, de-duplicated. When `HOME` is
/// set and non-empty, the read-only legacy `~/.forge` fallback is appended
/// last; an unset `HOME` contributes neither conventional home.
#[must_use]
pub fn zo_global_config_roots() -> Vec<PathBuf> {
    canonical_config_roots_from(
        std::env::var_os(ZO_CONFIG_HOME_ENV).map(PathBuf::from),
        std::env::var_os(ZO_HOME_ENV).map(PathBuf::from),
        std::env::var_os("HOME").map(PathBuf::from),
    )
}

fn canonical_config_roots_from(
    config_home: Option<PathBuf>,
    zo_home: Option<PathBuf>,
    user_home: Option<PathBuf>,
) -> Vec<PathBuf> {
    let user_home = user_home.filter(|home| !home.as_os_str().is_empty());
    dedupe_paths(
        config_home
            .into_iter()
            .chain(zo_home)
            .chain(user_home.iter().map(|home| home.join(ZO_DIR_NAME)))
            .chain(
                user_home
                    .iter()
                    .map(|home| home.join(LEGACY_FORGE_DIR_NAME)),
            ),
    )
}

fn dedupe_paths(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for path in paths {
        if !path.as_os_str().is_empty() && !roots.iter().any(|existing| existing == &path) {
            roots.push(path);
        }
    }
    roots
}

/// The single canonical write location for user state (sessions,
/// credentials, generated settings): the first entry of
/// [`zo_global_config_roots`]. When no user home can be resolved, use one
/// process-scoped, unpredictably named, owner-only temporary Zo home rather
/// than leaking global state into the current working directory.
#[must_use]
pub fn default_config_home() -> PathBuf {
    zo_global_config_roots()
        .into_iter()
        .next()
        .unwrap_or_else(secure_unresolved_config_home)
}

static UNRESOLVED_CONFIG_HOME: OnceLock<PathBuf> = OnceLock::new();

fn secure_unresolved_config_home() -> PathBuf {
    UNRESOLVED_CONFIG_HOME
        .get_or_init(|| {
            create_secure_unresolved_config_home(&std::env::temp_dir())
                .unwrap_or_else(|_| persistence_disabled_home())
        })
        .clone()
}

fn create_secure_unresolved_config_home(temp_dir: &Path) -> std::io::Result<PathBuf> {
    let base = if temp_dir.is_absolute() {
        temp_dir.to_path_buf()
    } else {
        std::env::current_dir()?.join(temp_dir)
    }
    .canonicalize()?;

    for _ in 0..128 {
        let token = rand::rng().random::<u128>();
        let private_root = base.join(format!("zo-unresolved-home-{token:032x}"));
        match create_owner_only_dir(&private_root) {
            Ok(()) => {
                let home = private_root.join(ZO_DIR_NAME);
                if let Err(error) = create_owner_only_dir(&home) {
                    let _ = std::fs::remove_dir(&private_root);
                    return Err(error);
                }
                return Ok(home);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
            Err(error) => return Err(error),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a private temporary Zo home",
    ))
}

fn create_owner_only_dir(path: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder.create(path)?;
    restrict_permissions_owner_only(path)
}

fn persistence_disabled_home() -> PathBuf {
    std::env::current_exe()
        .unwrap_or_else(|_| PathBuf::from(std::path::MAIN_SEPARATOR_STR))
        .join("zo-persistence-disabled")
        .join(ZO_DIR_NAME)
}

/// Environment variable that redirects all of zo's per-project `.zo/*`
/// operational state (todos, turn traces, …) out of the working directory. Set
/// it to a writable directory when the cwd is read-only or must stay clean
/// (a graded benchmark tree, a read-only mount). Unset → state lives under the
/// cwd as before. Distinct from the home chain above, which holds *global*
/// user state (credentials, sessions); this relocates *per-project* state.
pub const ZO_STATE_DIR_ENV: &str = "ZO_STATE_DIR";

/// Base directory under which a workspace's `.zo/` state is read and written:
/// the [`ZO_STATE_DIR_ENV`] override when set (and non-empty), else `cwd`.
/// Callers append `ZO_DIR_NAME`/… as before, so a single override relocates
/// every reader and writer consistently (no per-writer fallback drift).
#[must_use]
pub fn zo_state_base(cwd: &Path) -> PathBuf {
    state_base_from(std::env::var_os(ZO_STATE_DIR_ENV), cwd)
}

/// Env-free core of [`zo_state_base`], so the precedence is unit-testable
/// without mutating process state.
fn state_base_from(override_dir: Option<std::ffi::OsString>, cwd: &Path) -> PathBuf {
    match override_dir {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => cwd.to_path_buf(),
    }
}

/// Restrict a path to owner-only access (`0o700` for directories, `0o600` for
/// files) so other local users cannot read zo's prompts, transcripts, or
/// credentials.
///
/// This is the single source of truth for the permission policy that the
/// credential store, session persistence, and turn-trace writers all share.
/// On non-Unix platforms it is a no-op (POSIX permission bits do not exist),
/// which mirrors how the rest of the codebase treats filesystem permissions.
///
/// # Errors
/// Returns the underlying I/O error if the path exists but its permissions
/// cannot be changed.
pub fn restrict_permissions_owner_only(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = if path.is_dir() { 0o700 } else { 0o600 };
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Symlink-safe, owner-only (`0o600`) write of secret bytes (OAuth refresh
/// tokens, client secrets, ADC credentials) to `path`.
///
/// This writes in place: the permission bits are set at file creation, but the
/// write itself is a plain truncating overwrite — there is no atomic
/// temp-file-plus-rename and no `fsync` durability barrier, so a crash mid-write
/// can leave a truncated file. That is acceptable for these credentials (they
/// are re-fetched on the next run); do not describe it as atomic.
///
/// This is the single source of truth for the "write a private credential file"
/// policy so callers do not each re-derive the symlink/permission handling and
/// drift:
/// - the parent directory is created if missing and restricted to owner-only;
/// - a pre-existing target must be a regular file (never a symlink or special
///   file), so a planted symlink cannot redirect the write to an attacker's
///   target;
/// - on Unix the file is created with mode `0o600` *at creation* and opened
///   with `O_NOFOLLOW | O_NONBLOCK`, closing the TOCTOU window between the
///   pre-check and the open (a symlink is refused, and a swapped FIFO/device
///   cannot block the open); the opened fd is then re-checked to be a regular
///   file, and permissions are re-restricted for the case where the file
///   already existed.
///
/// On non-Unix platforms the symlink/regular-file pre-check still applies and
/// the bytes are written; POSIX permission bits do not exist there.
///
/// # Errors
/// Returns an I/O error if the parent cannot be prepared, the existing target
/// is not a regular file, a symlink is encountered on open, or the write fails.
pub fn write_secret_file(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    write_private_file(path, contents, &ParentDirPolicy::CreateAndRestrict)
}

/// How [`write_private_file`] treats `path`'s parent directory before writing.
pub enum ParentDirPolicy {
    /// Create the parent (and its ancestors) if missing and restrict it to
    /// owner-only. Correct for a credential file whose whole directory chain the
    /// caller owns (OAuth/ADC under the config home).
    CreateAndRestrict,
    /// Leave the parent directory entirely alone: the caller has already created
    /// and permissioned the leaf directory, and its ancestors may be shared,
    /// pre-existing directories this process does not own (chmod-ing those would
    /// `EPERM`). Used by the prompt cache, whose `ensure_private_dir` restricts
    /// the leaf dirs separately.
    LeaveParent,
}

/// The shared symlink-safe, owner-only (`0o600`) file write behind
/// [`write_secret_file`], parameterized by how the parent directory is handled
/// so the credential writers and the prompt cache reuse one implementation of
/// the symlink-rejection + `O_NOFOLLOW`/`0o600` policy rather than duplicating
/// it. Writes in place (no atomic rename / `fsync`), exactly as
/// [`write_secret_file`] documents.
///
/// # Errors
/// Returns an I/O error if the parent cannot be prepared (when requested), the
/// existing target is not a regular file, a symlink is encountered on open, or
/// the write fails.
pub fn write_private_file(
    path: &Path,
    contents: &[u8],
    parent_policy: &ParentDirPolicy,
) -> std::io::Result<()> {
    if matches!(parent_policy, ParentDirPolicy::CreateAndRestrict) {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
            restrict_permissions_owner_only(parent)?;
        }
    }

    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => {
            restrict_permissions_owner_only(path)?;
        }
        Ok(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("private file path is not a regular file: {}", path.display()),
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // `O_NOFOLLOW` rejects a symlink at `path`; `O_NONBLOCK` ensures the
        // open of a swapped FIFO/device returns instead of blocking on it.
        options.mode(0o600);
        options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    // Re-check the *opened* fd, not the earlier `symlink_metadata`: a FIFO or
    // device swapped in between the pre-check and the open would pass the
    // `is_file` check yet not be a regular file, so reject anything the open
    // actually landed on that is not one.
    if !file.metadata()?.file_type().is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("private file path is not a regular file: {}", path.display()),
        ));
    }
    restrict_permissions_owner_only(path)?;
    std::io::Write::write_all(&mut file, contents)
}

/// Lexically normalize a path: drop `.` (current-dir) components and resolve
/// each `..` (parent-dir) by popping the previous component. Performs no
/// filesystem access, so it is purely syntactic.
///
/// This is the single source of truth for the `..`-traversal collapsing that
/// both the config path resolver and the workspace-trust gate rely on; two
/// independent copies of this logic are exactly the divergence risk that
/// matters for trust and path decisions.
#[must_use]
pub fn normalize_path_components(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn normalize_collapses_dot_and_parent_components() {
        assert_eq!(
            normalize_path_components(Path::new("/a/./b/../c")),
            PathBuf::from("/a/c")
        );
        assert_eq!(
            normalize_path_components(Path::new("a/b/../../d")),
            PathBuf::from("d")
        );
    }

    #[test]
    fn canonical_roots_skip_empty_home_before_appending_zo() {
        assert_eq!(
            canonical_config_roots_from(
                Some(PathBuf::from("/config")),
                Some(PathBuf::new()),
                Some(PathBuf::new()),
            ),
            vec![PathBuf::from("/config")]
        );
        assert!(canonical_config_roots_from(
            Some(PathBuf::new()),
            Some(PathBuf::new()),
            Some(PathBuf::new()),
        )
        .is_empty());
    }

    #[test]
    fn global_roots_append_legacy_forge_last_without_changing_primary() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = [
            (ZO_CONFIG_HOME_ENV, std::env::var_os(ZO_CONFIG_HOME_ENV)),
            (ZO_HOME_ENV, std::env::var_os(ZO_HOME_ENV)),
            ("HOME", std::env::var_os("HOME")),
        ];
        std::env::remove_var(ZO_CONFIG_HOME_ENV);
        std::env::remove_var(ZO_HOME_ENV);
        let home = std::env::temp_dir().join(format!(
            "zo-global-roots-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::env::set_var("HOME", &home);

        let roots = zo_global_config_roots();
        assert_eq!(
            roots,
            vec![home.join(ZO_DIR_NAME), home.join(LEGACY_FORGE_DIR_NAME)]
        );
        assert_eq!(roots.first(), Some(&home.join(ZO_DIR_NAME)));
        assert_eq!(default_config_home(), home.join(ZO_DIR_NAME));

        std::env::remove_var("HOME");
        assert!(zo_global_config_roots().is_empty());

        for (key, value) in prior {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn no_home_fallback_is_private_absolute_and_process_stable() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = [
            (ZO_CONFIG_HOME_ENV, std::env::var_os(ZO_CONFIG_HOME_ENV)),
            (ZO_HOME_ENV, std::env::var_os(ZO_HOME_ENV)),
            ("HOME", std::env::var_os("HOME")),
        ];
        for (key, _) in &prior {
            std::env::remove_var(key);
        }

        let first = default_config_home();
        let second = default_config_home();

        for (key, value) in prior {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }

        assert!(first.is_absolute());
        assert_eq!(first, second);
        assert_eq!(first.file_name(), Some(std::ffi::OsStr::new(ZO_DIR_NAME)));
        assert!(first.is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&first)
                    .expect("temporary Zo home metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(first.parent().expect("private parent"))
                    .expect("private parent metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn state_base_prefers_non_empty_override_else_cwd() {
        let cwd = Path::new("/work/proj");
        assert_eq!(
            state_base_from(Some("/state".into()), cwd),
            PathBuf::from("/state")
        );
        assert_eq!(state_base_from(None, cwd), PathBuf::from("/work/proj"));
        // An empty override must not silently send state to the filesystem root.
        assert_eq!(
            state_base_from(Some("".into()), cwd),
            PathBuf::from("/work/proj")
        );
    }

    // A FIFO swapped in at the target passes the `symlink_metadata` pre-check
    // yet is not a regular file; the post-open re-check must reject it (and
    // `O_NONBLOCK` keeps the open from blocking on the FIFO).
    #[cfg(unix)]
    #[test]
    fn write_private_file_rejects_a_fifo_target() {
        let dir = std::env::temp_dir().join(format!(
            "zo-fifo-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("target");
        nix::unistd::mkfifo(&path, nix::sys::stat::Mode::S_IRWXU).expect("mkfifo");

        // A pre-existing FIFO is caught by the `symlink_metadata` pre-check
        // (`AlreadyExists`, "not a regular file"). The post-open `is_file`
        // re-check plus `O_NONBLOCK` are the defense for the harder case — a FIFO
        // swapped in *after* the pre-check — which cannot be raced
        // deterministically here; either layer rejects, and the call never
        // blocks or writes through the FIFO.
        let error = write_private_file(&path, b"secret", &ParentDirPolicy::LeaveParent)
            .expect_err("a FIFO target must be rejected");
        assert!(
            matches!(
                error.kind(),
                std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::InvalidInput
            ) || error.raw_os_error().is_some(),
            "unexpected error kind for FIFO rejection: {error:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
