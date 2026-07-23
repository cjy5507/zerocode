use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(test)]
use runtime::PermissionMode;
use zo_cli::tui::{RecentSession, StartupAuthState, StartupScreen};

#[cfg(test)]
use crate::resolve_model_alias;
use crate::{format_session_modified_age, status_context, VERSION};

/// Number of prior sessions surfaced on the launchpad splash.
const RECENT_SESSION_LIMIT: usize = 3;
/// Max display width (cells) for a recent-session title before it is
/// ellipsized; keeps long first-prompts from overflowing the centered
/// splash on narrow terminals.
const RECENT_LABEL_CELLS: usize = 40;

/// Returns the current process resident memory in KB (macOS/Linux).
///
/// On macOS uses `/usr/bin/memory_pressure` avoidance: reads
/// `rusage.ru_maxrss` which is available via the std library's
/// `resource_usage` on nightly. On stable, falls back to a cached
/// value from `ps` spawned once at startup.
pub(crate) fn resident_memory_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        return std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|pages| pages * 4)
            .unwrap_or(0);
    }
    #[cfg(target_os = "macos")]
    {
        use std::sync::OnceLock;
        static CACHED_RSS: OnceLock<u64> = OnceLock::new();
        *CACHED_RSS.get_or_init(|| {
            let pid = std::process::id();
            std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &pid.to_string()])
                .output()
                .ok()
                .and_then(|out| {
                    String::from_utf8_lossy(&out.stdout)
                        .trim()
                        .parse::<u64>()
                        .ok()
                })
                .unwrap_or(0)
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

// Consumed only by the test-gated plain-text banner path in `LiveCli`;
// the TUI renders `build_startup_screen` below.
#[cfg(test)]
pub(crate) fn build_startup_banner(
    model: &str,
    permission_mode: PermissionMode,
    session_id: &str,
    startup_elapsed: Option<Duration>,
) -> String {
    let status = status_context(None).ok();
    let git_branch = status
        .as_ref()
        .and_then(|context| context.git_branch.as_deref())
        .unwrap_or("unknown");
    let mem_kb = resident_memory_kb();
    let startup_suffix = startup_elapsed
        .map(|d| format!(" · {}ms", d.as_millis()))
        .unwrap_or_default();
    let mem_suffix = if mem_kb > 0 {
        format!(
            " · {:.1}MB",
            f64::from(u32::try_from(mem_kb).unwrap_or(u32::MAX)) / 1024.0
        )
    } else {
        String::new()
    };

    format!(
        "zo v{version}  {model}  ·  {perms}  ·  {branch}{startup}{mem}\n\
         {session_id}\n\
         /help commands  /model select  /status context  Shift+Enter newline",
        version = VERSION,
        perms = permission_mode.as_str(),
        branch = git_branch,
        startup = startup_suffix,
        mem = mem_suffix,
    )
}

pub(crate) fn build_startup_screen(
    model: &str,
    permissions: &str,
    session_id: &str,
    session_path: &Path,
    startup_elapsed: Option<Duration>,
) -> StartupScreen {
    let cwd = crate::current_cli_cwd().unwrap_or_else(|_| PathBuf::from("."));
    let status = status_context(Some(session_path)).ok();
    let workspace = status.as_ref().map_or_else(
        || "unknown".to_string(),
        |context| context.git_summary.headline(),
    );
    let project_root = status
        .as_ref()
        .and_then(|context| context.project_root.clone());
    let branch = status
        .as_ref()
        .and_then(|context| context.git_branch.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let mem_kb = resident_memory_kb();
    let recent_sessions = recent_sessions_for_launchpad(session_id);
    let auth = StartupAuthState {
        anthropic_oauth: runtime::load_oauth_credentials().ok().flatten().is_some(),
        chatgpt_oauth: runtime::load_openai_oauth().ok().flatten().is_some(),
    };

    StartupScreen {
        version: VERSION.to_string(),
        model: model.to_string(),
        permissions: permissions.to_string(),
        branch,
        workspace,
        directory: cwd,
        project_root,
        session_id: session_id.to_string(),
        autosave_path: session_path.to_path_buf(),
        startup_ms: startup_elapsed.map(|elapsed| elapsed.as_millis()),
        memory_mb: (mem_kb > 0)
            .then(|| f64::from(u32::try_from(mem_kb).unwrap_or(u32::MAX)) / 1024.0),
        auth,
        recent_sessions,
    }
}

/// Build the launchpad's recent-session list: the most recently modified
/// non-empty sessions, excluding the one we are about to open. Returns an
/// empty vec on any registry error (the splash then omits the section).
fn recent_sessions_for_launchpad(active_session_id: &str) -> Vec<RecentSession> {
    // Parse only a few extra beyond what we display: the cheap mtime sort caps
    // the expensive JSONL parse, and the `+ 1` backfills if the active session
    // (excluded below) happens to be among the most recent on disk.
    let Ok(sessions) =
        crate::session_registry::list_managed_sessions_limited(Some(RECENT_SESSION_LIMIT + 1))
    else {
        return Vec::new();
    };
    sessions
        .into_iter()
        .filter(|s| s.message_count > 0 && s.id != active_session_id)
        .take(RECENT_SESSION_LIMIT)
        .map(|s| {
            let raw_title = s.first_user_text.as_deref().map_or_else(
                || s.id.chars().take(12).collect::<String>(),
                |text| text.lines().next().unwrap_or(text).trim().to_string(),
            );
            RecentSession {
                label: truncate_to_cells(&raw_title, RECENT_LABEL_CELLS),
                age: format_session_modified_age(s.modified_epoch_millis),
            }
        })
        .collect()
}

/// Width-aware truncation to `max_cells` display columns, appending an
/// ellipsis when clipped. Uses `unicode-width` so CJK (e.g. Korean) titles
/// count as 2 cells each — matching the project's unified width handling.
fn truncate_to_cells(s: &str, max_cells: usize) -> String {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

    if UnicodeWidthStr::width(s) <= max_cells {
        return s.to_string();
    }
    // Reserve one cell for the ellipsis.
    let budget = max_cells.saturating_sub(1);
    let mut acc = 0usize;
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if acc + w > budget {
            break;
        }
        acc += w;
        out.push(ch);
    }
    out.push('\u{2026}');
    out
}

// Same test-only status as `build_startup_banner` above.
#[cfg(test)]
pub(crate) fn input_box_label(model: &str, session_id: &str) -> String {
    let model = resolve_model_alias(model);
    let session_head: String = session_id.chars().take(8).collect();
    if session_head.is_empty() {
        model
    } else {
        format!("{model} · {session_head}")
    }
}
