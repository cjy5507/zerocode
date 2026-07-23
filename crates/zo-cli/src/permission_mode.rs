//! Permission-mode resolution helpers.
//!
//! One cohesive concern lifted out of `main.rs`: turning the various
//! permission-mode representations (CLI label, config-resolved enum, env
//! override) into the runtime's [`PermissionMode`]. The crate root re-exports
//! the `pub(crate)` entry points so existing `crate::…` call sites are
//! unchanged; the resolution-detail helpers stay private to this module.

use std::path::Path;

use runtime::{ConfigLoader, PermissionMode, ResolvedPermissionMode};

use crate::current_cli_cwd;

/// Map a CLI/label string to a [`PermissionMode`], warning and defaulting to
/// `workspace-write` on an unrecognised label.
pub(crate) fn permission_mode_from_label(mode: &str) -> PermissionMode {
    match mode {
        "read-only" => PermissionMode::ReadOnly,
        "workspace-write" => PermissionMode::WorkspaceWrite,
        "danger-full-access" => PermissionMode::DangerFullAccess,
        other => {
            eprintln!(
                "warning: unsupported permission mode '{other}', falling back to workspace-write"
            );
            PermissionMode::WorkspaceWrite
        }
    }
}

fn permission_mode_from_resolved(mode: ResolvedPermissionMode) -> PermissionMode {
    match mode {
        ResolvedPermissionMode::ReadOnly => PermissionMode::ReadOnly,
        ResolvedPermissionMode::WorkspaceWrite => PermissionMode::WorkspaceWrite,
        ResolvedPermissionMode::DangerFullAccess => PermissionMode::DangerFullAccess,
    }
}

/// The effective default permission mode: env override first, then the config
/// for the current directory, then a workspace-risk-aware fallback.
pub(crate) fn default_permission_mode() -> PermissionMode {
    resolve_permission_mode(fallback_permission_mode_for_current_dir)
}

/// Like [`default_permission_mode`] but routes the cwd fallback through the
/// interactive workspace-trust gate (the Claude-Code-style "trust this folder?"
/// prompt). Used only by the TUI repl entry point, so the prompt never fires on
/// a headless `-p`/serve path — those keep going through
/// [`default_permission_mode`].
pub(crate) fn interactive_default_permission_mode(inline: bool) -> PermissionMode {
    resolve_permission_mode(|| interactive_fallback_permission_mode_for_current_dir(inline))
}

/// Shared config → `fallback` resolution. The environment override is
/// intentionally NOT consulted: the permission mode comes only from the actual
/// configured value for the current directory (the live `/permissions`
/// selection persisted to settings), falling back to the workspace-risk-aware
/// default when nothing is configured. Only the cwd fallback differs between
/// the headless and interactive entry points.
fn resolve_permission_mode(fallback: impl FnOnce() -> PermissionMode) -> PermissionMode {
    config_permission_mode_for_current_dir().unwrap_or_else(fallback)
}

fn interactive_fallback_permission_mode_for_current_dir(inline: bool) -> PermissionMode {
    current_cli_cwd().ok().as_deref().map_or(
        PermissionMode::DangerFullAccess,
        |cwd| {
            crate::workspace_trust::resolve_trust_for_cwd(
                cwd,
                fallback_permission_mode_for_cwd,
                inline,
            )
        },
    )
}

/// Whether merged settings request primary-screen inline TUI rendering.
///
/// Config load errors fail open here because the normal runtime bootstrap will
/// report them with full context moments later; this early read exists only so
/// the first-visit workspace trust prompt uses the correct terminal strategy.
pub(crate) fn configured_tui_inline_mode() -> bool {
    let Some(cwd) = current_cli_cwd().ok() else {
        return false;
    };
    ConfigLoader::default_for(&cwd)
        .load()
        .is_ok_and(|config| config.tui_inline_mode())
}

fn config_permission_mode_for_current_dir() -> Option<PermissionMode> {
    let cwd = current_cli_cwd().ok()?;
    let loader = ConfigLoader::default_for(&cwd);
    loader
        .load()
        .ok()?
        .permission_mode()
        .map(permission_mode_from_resolved)
}

fn fallback_permission_mode_for_current_dir() -> PermissionMode {
    current_cli_cwd().ok().as_deref().map_or(
        PermissionMode::DangerFullAccess,
        fallback_permission_mode_for_cwd,
    )
}

fn fallback_permission_mode_for_cwd(cwd: &Path) -> PermissionMode {
    if crate::workspace_trust::classify_cwd(cwd).requires_safe_default() {
        PermissionMode::ReadOnly
    } else {
        PermissionMode::DangerFullAccess
    }
}

/// Canonicalise a permission-mode label, or `None` when it is not one of the
/// three supported modes.
pub(crate) fn normalize_permission_mode(mode: &str) -> Option<&'static str> {
    match mode.trim() {
        "read-only" => Some("read-only"),
        "workspace-write" => Some("workspace-write"),
        "danger-full-access" => Some("danger-full-access"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_permission_mode;

    #[test]
    fn normalize_permission_mode_accepts_supported_modes() {
        assert_eq!(normalize_permission_mode("read-only"), Some("read-only"));
        assert_eq!(
            normalize_permission_mode("workspace-write"),
            Some("workspace-write")
        );
        assert_eq!(
            normalize_permission_mode("danger-full-access"),
            Some("danger-full-access")
        );
        assert_eq!(
            normalize_permission_mode("  read-only  "),
            Some("read-only")
        );
        assert_eq!(normalize_permission_mode("unknown"), None);
    }
}
