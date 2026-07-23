use runtime::{
    summary_compression::compress_summary_text, LaneEvent, LaneEventBlocker, LaneEventName,
    LaneEventStatus, LaneFailureClass,
};

use super::super::epoch_seconds_now;
use super::AgentOutput;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Per-agent locks serialize every status-affecting manifest read-modify-write
/// without making unrelated agents wait on filesystem I/O. Weak entries vanish
/// once a manifest operation finishes, so this registry does not retain old ids.
static AGENT_MANIFEST_LOCKS: OnceLock<Mutex<HashMap<String, Weak<Mutex<()>>>>> = OnceLock::new();
static AGENT_FILE_TEMP_SEQUENCE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// Every spawned agent owns one dedicated OS thread. Pin the durable run
    /// generation to that thread so late provider/tool activity from an older
    /// resume leg cannot mutate the current manifest generation.
    static ACTIVE_AGENT_RUN_GENERATION: std::cell::Cell<Option<u64>> = const {
        std::cell::Cell::new(None)
    };
    /// `run_if_agent_manifest_running` keeps the per-agent mutex while it
    /// creates the worker. Test runners and embedders may complete synchronously
    /// on that same thread, so nested persistence must reuse the outer claim.
    static HELD_AGENT_MANIFEST_LOCKS: std::cell::RefCell<Vec<String>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

struct ActiveRunGenerationGuard(Option<u64>);

struct HeldAgentManifestLockGuard(String);

impl Drop for HeldAgentManifestLockGuard {
    fn drop(&mut self) {
        HELD_AGENT_MANIFEST_LOCKS.with(|held| {
            let mut held = held.borrow_mut();
            let position = held
                .iter()
                .rposition(|agent_id| agent_id == &self.0)
                .expect("held agent manifest lock is tracked");
            held.remove(position);
        });
    }
}

impl Drop for ActiveRunGenerationGuard {
    fn drop(&mut self) {
        ACTIVE_AGENT_RUN_GENERATION.with(|generation| generation.set(self.0));
    }
}

pub(super) fn with_agent_run_generation<R>(generation: u64, action: impl FnOnce() -> R) -> R {
    let previous = ACTIVE_AGENT_RUN_GENERATION.with(|active| active.replace(Some(generation)));
    let _guard = ActiveRunGenerationGuard(previous);
    action()
}

fn active_agent_run_generation() -> Option<u64> {
    ACTIVE_AGENT_RUN_GENERATION.with(std::cell::Cell::get)
}

fn agent_manifest_locks() -> &'static Mutex<HashMap<String, Weak<Mutex<()>>>> {
    AGENT_MANIFEST_LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn agent_manifest_lock(agent_id: &str) -> Arc<Mutex<()>> {
    let mut locks = agent_manifest_locks()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(agent_id).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    locks.insert(agent_id.to_string(), Arc::downgrade(&lock));
    lock
}

fn with_agent_manifest_lock<R>(agent_id: &str, action: impl FnOnce() -> R) -> R {
    if HELD_AGENT_MANIFEST_LOCKS.with(|held| held.borrow().iter().any(|held| held == agent_id)) {
        return action();
    }

    let lock = agent_manifest_lock(agent_id);
    let _guard = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    HELD_AGENT_MANIFEST_LOCKS.with(|held| held.borrow_mut().push(agent_id.to_string()));
    let _held_guard = HeldAgentManifestLockGuard(agent_id.to_string());
    action()
}

pub(super) fn expected_manifest_filename(agent_id: &str) -> Result<String, String> {
    if agent_id.is_empty()
        || Path::new(agent_id).components().count() != 1
        || agent_id.contains(std::path::MAIN_SEPARATOR)
        || agent_id.contains('/')
        || agent_id.contains('\\')
    {
        return Err("agent manifest has an unsafe agent id".to_string());
    }
    Ok(format!("{agent_id}.json"))
}

fn reject_unsafe_existing_file(path: &Path, kind: &str) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("agent {kind} target is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.nlink() > 1 {
            return Err(format!("agent {kind} target has hard-link aliases"));
        }
    }
    Ok(())
}

fn validate_open_regular_file(file: &std::fs::File, kind: &str) -> Result<(), String> {
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.file_type().is_file() {
        return Err(format!("agent {kind} target is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.nlink() > 1 {
            return Err(format!("agent {kind} target has hard-link aliases"));
        }
    }
    Ok(())
}

fn open_existing_regular_file(path: &Path, kind: &str, append: bool) -> Result<std::fs::File, String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(!append).append(append);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|error| error.to_string())?;
    validate_open_regular_file(&file, kind)?;
    Ok(file)
}

fn read_existing_regular_file(path: &Path, kind: &str) -> Result<String, String> {
    use std::io::Read as _;

    let mut file = open_existing_regular_file(path, kind, false)?;
    let mut text = String::new();
    file.read_to_string(&mut text).map_err(|error| error.to_string())?;
    Ok(text)
}

fn write_new_regular_file(path: &Path, kind: &str, contents: &[u8]) -> Result<(), String> {
    use std::io::Write as _;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options
            .mode(0o600)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|error| error.to_string())?;
    validate_open_regular_file(&file, kind)?;
    file.write_all(contents).map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())
}

fn write_regular_file_atomically(path: &Path, kind: &str, contents: &[u8]) -> Result<(), String> {
    write_regular_file_atomically_with(path, kind, contents, |source, destination| {
        std::fs::rename(source, destination)
    })
}

fn write_regular_file_atomically_with(
    path: &Path,
    kind: &str,
    contents: &[u8],
    rename: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => reject_unsafe_existing_file(path, kind)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("agent {kind} target has no parent directory"))?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("agent {kind} target has an invalid filename"))?;
    let sequence = AGENT_FILE_TEMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{filename}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let result = (|| {
        write_new_regular_file(&temporary, kind, contents)?;
        rename(&temporary, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// Resolve constructor-supplied manifest fields to sibling paths under one
/// canonical store directory. Serialized fields are checked, never trusted as
/// an authority for either directory.
fn trusted_manifest_paths(manifest: &AgentOutput) -> Result<(PathBuf, PathBuf), String> {
    let filename = expected_manifest_filename(&manifest.agent_id)?;
    let supplied_manifest = Path::new(&manifest.manifest_file);
    if supplied_manifest.file_name().and_then(|name| name.to_str()) != Some(filename.as_str()) {
        return Err("agent manifest filename does not match its id".to_string());
    }
    let parent = supplied_manifest
        .parent()
        .ok_or_else(|| "agent manifest has no parent directory".to_string())?;
    let store = std::fs::canonicalize(parent).map_err(|error| error.to_string())?;
    let manifest_path = store.join(&filename);
    if supplied_manifest.exists() {
        reject_unsafe_existing_file(supplied_manifest, "manifest")?;
        if std::fs::canonicalize(supplied_manifest).map_err(|error| error.to_string())?
            != manifest_path
        {
            return Err("agent manifest is outside its trusted store".to_string());
        }
    }
    let output_filename = format!("{}.md", manifest.agent_id);
    let output_path = store.join(&output_filename);
    let supplied_output = Path::new(&manifest.output_file);
    let lexical_output = parent.join(&output_filename);
    if supplied_output != lexical_output
        && supplied_output != output_path
        && std::fs::canonicalize(supplied_output).map_err(|error| error.to_string())? != output_path
    {
        return Err("agent output filename does not match its manifest id".to_string());
    }
    match std::fs::symlink_metadata(&output_path) {
        Ok(_) => reject_unsafe_existing_file(&output_path, "output")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    Ok((manifest_path, output_path))
}

fn trusted_resume_snapshot_path(manifest: &AgentOutput) -> Result<PathBuf, String> {
    let (manifest_path, _) = trusted_manifest_paths(manifest)?;
    let parent = manifest_path
        .parent()
        .ok_or_else(|| "agent manifest has no parent directory".to_string())?;
    Ok(parent.join(format!("{}.resume.json", manifest.agent_id)))
}

pub(super) fn trusted_agent_transcript_path(manifest: &AgentOutput) -> Result<PathBuf, String> {
    let (manifest_path, _) = trusted_manifest_paths(manifest)?;
    let parent = manifest_path
        .parent()
        .ok_or_else(|| "agent manifest has no parent directory".to_string())?;
    let path = parent.join(format!("{}.session.jsonl", manifest.agent_id));
    match std::fs::symlink_metadata(&path) {
        Ok(_) => reject_unsafe_existing_file(&path, "transcript")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    Ok(path)
}

pub(super) fn write_new_agent_output(
    output_path: &Path,
    agent_id: &str,
    contents: &str,
) -> Result<(), String> {
    expected_manifest_filename(agent_id)?;
    let expected_filename = format!("{agent_id}.md");
    if output_path.file_name().and_then(|name| name.to_str()) != Some(&expected_filename) {
        return Err("agent output filename does not match its id".to_string());
    }
    let parent = output_path
        .parent()
        .ok_or_else(|| "agent output has no parent directory".to_string())?;
    let store = std::fs::canonicalize(parent).map_err(|error| error.to_string())?;
    let trusted_output = store.join(expected_filename);
    write_new_regular_file(&trusted_output, "output", contents.as_bytes())
}

pub(super) fn write_agent_resume_snapshot_file(
    manifest: &AgentOutput,
    contents: &str,
) -> Result<(), String> {
    let path = trusted_resume_snapshot_path(manifest)?;
    write_regular_file_atomically(&path, "resume snapshot", contents.as_bytes())
}

pub(super) fn read_agent_resume_snapshot_file(manifest: &AgentOutput) -> Result<String, String> {
    let path = trusted_resume_snapshot_path(manifest)?;
    read_existing_regular_file(&path, "resume snapshot")
}

/// Bind a scanned document to the actual directory-entry that discovered it.
/// This is the only scanner entry point: `manifestFile` and `outputFile` in
/// the JSON must agree with the physical sibling paths or the document is
/// ignored.
pub(super) fn load_agent_manifest_from_scanned_path(path: &Path) -> Result<AgentOutput, String> {
    reject_unsafe_existing_file(path, "manifest")?;
    let text = read_existing_regular_file(path, "manifest")?;
    let mut manifest = serde_json::from_str::<AgentOutput>(&text).map_err(|error| error.to_string())?;
    let (manifest_path, output_path) = trusted_manifest_paths(&manifest)?;
    let discovered = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
    if discovered != manifest_path {
        return Err("scanned manifest does not match its agent id".to_string());
    }
    manifest.manifest_file = manifest_path.display().to_string();
    manifest.output_file = output_path.display().to_string();
    Ok(manifest)
}

/// Read a known, constructor-trusted manifest path and bind its fields before
/// an update. The caller may not redirect persistence through serialized JSON.
fn load_bound_manifest(manifest: &AgentOutput) -> Result<AgentOutput, String> {
    let (manifest_path, _) = trusted_manifest_paths(manifest)?;
    load_agent_manifest_from_scanned_path(&manifest_path)
}

/// Execute a worker-start action only while the durable manifest still says
/// `running`. The same per-agent lock guards external stop persistence. A
/// synchronous runner may persist completion before returning; the lock helper
/// treats that same-thread nested persistence as reentrant while other threads
/// remain serialized.
pub(super) fn run_if_agent_manifest_running<R, E>(
    manifest: &AgentOutput,
    action: impl FnOnce() -> Result<R, E>,
) -> Result<Option<R>, E>
where
    E: From<String>,
{
    with_agent_manifest_lock(&manifest.agent_id, || {
        let current = load_bound_manifest(manifest).map_err(E::from)?;
        if current.status != "running" || current.run_generation != manifest.run_generation {
            return Ok(None);
        }
        action().map(Some)
    })
}

/// Read-only projection used by the workflow startup watchdog. It deliberately
/// keeps transport liveness, model reasoning, and task progress as distinct
/// signals so a keep-alive cannot masquerade as useful work.
#[derive(Debug, Clone, Default)]
pub(crate) struct AgentActivitySnapshot {
    pub(crate) started_at: Option<u64>,
    pub(crate) stream_open_at: Option<u64>,
    pub(crate) last_reasoning_at: Option<u64>,
    pub(crate) first_task_action_at: Option<u64>,
    pub(crate) last_task_progress_at: Option<u64>,
    pub(crate) current_tool: Option<String>,
    pub(crate) effective_effort: Option<String>,
    pub(crate) selected_model: Option<String>,
    pub(crate) fallback_models: Vec<String>,
    pub(crate) loaded_skills: Vec<String>,
}

pub(super) fn write_agent_manifest(manifest: &AgentOutput) -> Result<(), String> {
    let (manifest_path, output_path) = trusted_manifest_paths(manifest)?;
    let mut persisted = manifest.clone();
    persisted.manifest_file = manifest_path.display().to_string();
    persisted.output_file = output_path.display().to_string();
    let json = serde_json::to_string_pretty(&persisted).map_err(|error| error.to_string())?;
    write_regular_file_atomically(&manifest_path, "manifest", json.as_bytes())
}

pub(crate) fn persist_agent_terminal_state(
    manifest: &AgentOutput,
    status: &str,
    result: Option<&str>,
    error: Option<String>,
) -> Result<(), String> {
    persist_agent_terminal_state_with_history(manifest, status, result, error, Vec::new())
}

/// Same as [`persist_agent_terminal_state`] but also writes the per-turn
/// `token_history` series back into the manifest. Callers that care about
/// the sparkline data path (`run_agent_job`) use this; legacy/failure
/// callers stay on the no-history variant so they don't have to thread an
/// always-empty `Vec` through their code paths.
pub(super) fn persist_agent_terminal_state_with_history(
    manifest: &AgentOutput,
    status: &str,
    result: Option<&str>,
    error: Option<String>,
    token_history: Vec<u32>,
) -> Result<(), String> {
    persist_agent_terminal_state_with_history_if_running(
        manifest,
        status,
        result,
        error,
        token_history,
    )
    .map(|_| ())
}

pub(super) fn persist_agent_failed_state_if_running(
    manifest: &AgentOutput,
    error: String,
) -> Result<bool, String> {
    persist_agent_terminal_state_with_history_if_running(
        manifest,
        "failed",
        None,
        Some(error),
        Vec::new(),
    )
}

fn persist_agent_terminal_state_with_history_if_running(
    manifest: &AgentOutput,
    status: &str,
    result: Option<&str>,
    error: Option<String>,
    token_history: Vec<u32>,
) -> Result<bool, String> {
    with_agent_manifest_lock(&manifest.agent_id, || {
        let mut next_manifest = load_bound_manifest(manifest)?;
        if next_manifest.run_generation != manifest.run_generation {
            return Ok(false);
        }
        if super::agent_output_status_is_terminal(&next_manifest.status) {
            // A stop has already won. Keep that terminal status, but retain the
            // worker's richer final text and token history for later inspection.
            if next_manifest.status == "stopped" {
                append_agent_output(
                    &next_manifest,
                    &format_agent_terminal_enrichment(result, error.as_deref()),
                )?;
                if !token_history.is_empty() {
                    next_manifest.token_history = token_history;
                    write_agent_manifest(&next_manifest)?;
                }
            }
            return Ok(false);
        }

        let blocker = error.as_deref().map(classify_lane_blocker);
        append_agent_output(
            &next_manifest,
            &format_agent_terminal_output(status, result, blocker.as_ref(), error.as_deref()),
        )?;
        next_manifest.status = status.to_string();
        next_manifest.completed_at = Some(epoch_seconds_now());
        next_manifest.current_tool = None;
        next_manifest.current_phase = None;
        next_manifest.last_activity_at = Some(epoch_seconds_now_u64());
        next_manifest.current_blocker.clone_from(&blocker);
        next_manifest.error = error;
        if !token_history.is_empty() {
            next_manifest.token_history = token_history;
        }
        if let Some(blocker) = blocker {
            next_manifest
                .lane_events
                .push(LaneEvent::blocked(epoch_seconds_now(), &blocker));
            next_manifest
                .lane_events
                .push(LaneEvent::failed(epoch_seconds_now(), &blocker));
        } else {
            next_manifest.current_blocker = None;
            let compressed_detail = result
                .filter(|value| !value.trim().is_empty())
                .map(|value| compress_summary_text(value.trim()));
            next_manifest
                .lane_events
                .push(LaneEvent::finished(epoch_seconds_now(), compressed_detail));
        }
        write_agent_manifest(&next_manifest)?;
        Ok(true)
    })
}

/// Reset a TERMINAL manifest back to `running` for a `SendMessage` resume and
/// append the follow-up prompt to the human-readable output file. Reads the
/// CURRENT on-disk manifest (mid-run model swaps and the final terminal write
/// both live there) and returns the refreshed copy the new [`super::AgentJob`]
/// is built from. `started_at` is re-stamped so the resumed leg's duration is
/// measured on its own.
///
/// Atomically claim a terminal generation for resume and install the live
/// in-process handles before the new `running` state becomes discoverable.
/// A failed durable write rolls those handles back while the same per-agent
/// lock is still held, so a losing concurrent resume never touches the winner.
pub(super) fn persist_agent_resumed_state_with(
    manifest: &AgentOutput,
    follow_up: &str,
    before_publish: impl FnOnce(&AgentOutput),
    rollback: impl FnOnce(&AgentOutput),
) -> Result<AgentOutput, String> {
    with_agent_manifest_lock(&manifest.agent_id, || {
        let mut next_manifest = load_bound_manifest(manifest)?;
        if !super::agent_output_status_is_terminal(&next_manifest.status) {
            return Err("agent is already running and cannot be resumed".to_string());
        }
        if next_manifest.run_generation != manifest.run_generation {
            return Err("agent resume generation is stale".to_string());
        }
        append_agent_output(
            &next_manifest,
            &format!("\n## Follow-up (resumed)\n\n{}\n", follow_up.trim()),
        )?;
        next_manifest.status = String::from("running");
        next_manifest.run_generation = next_manifest.run_generation.saturating_add(1);
        next_manifest.completed_at = None;
        // The marker is per-generation: the resumed run has not delivered yet.
        next_manifest.completion_published_at = None;
        next_manifest.error = None;
        next_manifest.current_blocker = None;
        next_manifest.current_tool = None;
        next_manifest.current_phase = None;
        next_manifest.started_at = Some(epoch_seconds_now());
        next_manifest.last_activity_at = Some(epoch_seconds_now_u64());
        next_manifest
            .lane_events
            .push(LaneEvent::started(epoch_seconds_now()));
        before_publish(&next_manifest);
        if let Err(error) = write_agent_manifest(&next_manifest) {
            rollback(&next_manifest);
            return Err(error);
        }
        Ok(next_manifest)
    })
}

/// Best-effort post-mortem marker: stamp the manifest when a completion for
/// this agent crossed the host's delivery edge — store publication on
/// channel-less hosts (serve/headless sweep the store at turn boundaries)
/// plus a successful interactive wakeup send where a channel is registered
/// (the notify caller gates on that). A terminal manifest WITHOUT this stamp
/// is the "died without delivering a result" signature that made the
/// silent-death incident undiagnosable.
///
/// The stamp is generation-safe: only a TERMINAL manifest whose
/// `run_generation` matches the publishing run may carry it, so a delayed
/// publication from generation N can never fake "delivered" on a concurrently
/// resumed N+1 (resume also clears the marker). Every error leg is swallowed —
/// publication must never fail on a stamp problem.
pub(super) fn stamp_completion_published(agent_id: &str, expected_generation: Option<u64>) {
    let Ok(filename) = expected_manifest_filename(agent_id) else {
        return;
    };
    let Ok(store) = super::agent_store_dir() else {
        return;
    };
    let path = store.join(filename);
    with_agent_manifest_lock(agent_id, || {
        let Ok(mut manifest) = load_agent_manifest_from_scanned_path(&path) else {
            return;
        };
        if !super::agent_output_status_is_terminal(&manifest.status) {
            return;
        }
        if expected_generation.is_some_and(|generation| generation != manifest.run_generation) {
            return;
        }
        manifest.completion_published_at = Some(epoch_seconds_now());
        let _ = write_agent_manifest(&manifest);
    });
}

pub(super) fn persist_agent_stopped_state(
    manifest: &AgentOutput,
    reason: &str,
) -> Result<bool, String> {
    persist_agent_stopped_state_with(manifest, reason, || {})
}

/// Atomically claims a running manifest for an external stop. The callback runs
/// only for the claimant and while its per-agent lock is held, so cancellation
/// signal ownership and the durable terminal transition cannot split. The
/// callback runs only after the stopped output and manifest have been written
/// successfully, while the per-agent lock is still held.
pub(super) fn persist_agent_stopped_state_with(
    manifest: &AgentOutput,
    reason: &str,
    after_transition: impl FnOnce(),
) -> Result<bool, String> {
    with_agent_manifest_lock(&manifest.agent_id, || {
        let mut next_manifest = load_bound_manifest(manifest)?;
        if next_manifest.run_generation != manifest.run_generation
            || super::agent_output_status_is_terminal(&next_manifest.status)
        {
            return Ok(false);
        }
        append_agent_output(&next_manifest, &format_agent_stopped_output(reason))?;
        next_manifest.status = String::from("stopped");
        next_manifest.completed_at = Some(epoch_seconds_now());
        next_manifest.current_tool = None;
        next_manifest.current_phase = None;
        next_manifest.last_activity_at = Some(epoch_seconds_now_u64());
        next_manifest.current_blocker = None;
        next_manifest.error = None;
        next_manifest.lane_events.push(
            LaneEvent::new(
                LaneEventName::Closed,
                LaneEventStatus::Closed,
                epoch_seconds_now(),
            )
            .with_detail(reason.trim().to_string()),
        );
        write_agent_manifest(&next_manifest)?;
        after_transition();
        Ok(true)
    })
}

fn append_agent_output(manifest: &AgentOutput, suffix: &str) -> Result<(), String> {
    use std::io::Write as _;

    let (_, output_path) = trusted_manifest_paths(manifest)?;
    let mut file = open_existing_regular_file(&output_path, "output", true)?;
    file.write_all(suffix.as_bytes())
        .map_err(|error| error.to_string())
}

pub(super) fn read_agent_output(manifest: &AgentOutput) -> Result<String, String> {
    let (_, output_path) = trusted_manifest_paths(manifest)?;
    read_existing_regular_file(&output_path, "output")
}

pub(super) fn manifest_generation_is_current(manifest: &AgentOutput) -> bool {
    load_bound_manifest(manifest)
        .is_ok_and(|current| current.run_generation == manifest.run_generation)
}

fn format_agent_terminal_enrichment(result: Option<&str>, error: Option<&str>) -> String {
    let mut sections = Vec::new();
    if let Some(result) = result.filter(|value| !value.trim().is_empty()) {
        sections.push(format!("\n### Final response\n\n{}\n", result.trim()));
    }
    if let Some(error) = error.filter(|value| !value.trim().is_empty()) {
        sections.push(format!("\n### Worker detail\n\n{}\n", error.trim()));
    }
    sections.join("")
}

fn format_agent_stopped_output(reason: &str) -> String {
    let reason = reason.trim();
    if reason.is_empty() {
        "\n## Result\n\n- status: stopped\n".to_string()
    } else {
        format!("\n## Result\n\n- status: stopped\n\n### Stop reason\n\n{reason}\n")
    }
}

fn format_agent_terminal_output(
    status: &str,
    result: Option<&str>,
    blocker: Option<&LaneEventBlocker>,
    error: Option<&str>,
) -> String {
    let mut sections = vec![format!("\n## Result\n\n- status: {status}\n")];
    if let Some(blocker) = blocker {
        sections.push(format!(
            "\n### Blocker\n\n- failure_class: {}\n- detail: {}\n",
            serde_json::to_string(&blocker.failure_class)
                .unwrap_or_else(|_| "\"infra\"".to_string())
                .trim_matches('"'),
            blocker.detail.trim()
        ));
    }
    if let Some(result) = result.filter(|value| !value.trim().is_empty()) {
        sections.push(format!("\n### Final response\n\n{}\n", result.trim()));
    }
    if let Some(error) = error.filter(|value| !value.trim().is_empty()) {
        sections.push(format!("\n### Error\n\n{}\n", error.trim()));
    }
    sections.join("")
}

fn classify_lane_blocker(error: &str) -> LaneEventBlocker {
    let detail = error.trim().to_string();
    LaneEventBlocker {
        failure_class: classify_lane_failure(error),
        detail,
    }
}

pub(crate) fn classify_lane_failure(error: &str) -> LaneFailureClass {
    let normalized = error.to_ascii_lowercase();

    if normalized.contains("prompt") && normalized.contains("deliver") {
        LaneFailureClass::PromptDelivery
    } else if normalized.contains("trust") {
        LaneFailureClass::TrustGate
    } else if normalized.contains("branch")
        && (normalized.contains("stale") || normalized.contains("diverg"))
    {
        LaneFailureClass::BranchDivergence
    } else if normalized.contains("gateway")
        || normalized.contains("routing")
        || normalized.contains("429")
        || normalized.contains("too many requests")
        || normalized.contains("rate_limit")
        || normalized.contains("rate limit")
        || normalized.contains("401")
        || normalized.contains("unauthorized")
        || normalized.contains("403")
        || normalized.contains("forbidden")
        || normalized.contains("permission_error")
        || normalized.contains("authentication")
        || normalized.contains("api key")
        || normalized.contains("credentials")
    {
        LaneFailureClass::GatewayRouting
    } else if normalized.contains("compile")
        || normalized.contains("build failed")
        || normalized.contains("cargo check")
    {
        LaneFailureClass::Compile
    } else if normalized.contains("test") {
        LaneFailureClass::Test
    } else if normalized.contains("tool failed")
        || normalized.contains("runtime tool")
        || normalized.contains("tool runtime")
        || normalized.contains("conversation loop exceeded")
        || normalized.contains("maximum number of iterations")
        || normalized.contains("time budget")
    {
        LaneFailureClass::ToolRuntime
    } else if normalized.contains("plugin") {
        LaneFailureClass::PluginStartup
    } else if normalized.contains("mcp") && normalized.contains("handshake") {
        LaneFailureClass::McpHandshake
    } else if normalized.contains("mcp") {
        LaneFailureClass::McpStartup
    } else {
        LaneFailureClass::Infra
    }
}

/// Maximum entries kept in a manifest's `recentTools` rolling feed.
pub(crate) const RECENT_TOOLS_CAP: usize = 12;

/// Maximum chars kept in a manifest's `outputTail` rolling text buffer.
pub(crate) const OUTPUT_TAIL_CAP: usize = 2_000;

pub(super) fn retain_output_tail_window(output_tail: &mut String) {
    let excess_chars = output_tail.chars().count().saturating_sub(OUTPUT_TAIL_CAP);
    if excess_chars > 0 {
        let cut = output_tail
            .char_indices()
            .nth(excess_chars)
            .map_or(0, |(idx, _)| idx);
        output_tail.drain(..cut);
    }
}

/// Epoch seconds as a number, for the `lastActivityAt` heartbeat field.
pub(super) fn epoch_seconds_now_u64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// Best-effort read–modify–write of one agent manifest. Silently no-ops on
/// any I/O / parse error — a missed frame just means a slightly stale
/// sidebar, never a failure.
fn update_manifest_best_effort(
    manifest_path: &std::path::Path,
    apply: impl FnOnce(&mut AgentOutput),
) {
    let expected_generation = active_agent_run_generation();
    let Ok(initial) = load_agent_manifest_from_scanned_path(manifest_path) else {
        return;
    };
    with_agent_manifest_lock(&initial.agent_id, || {
        let Ok(mut manifest) = load_bound_manifest(&initial) else {
            return;
        };
        // Re-check under the same lock used by terminal writers. A live frame
        // must never overwrite a just-committed terminal state with `running`,
        // and an old resumed worker must not stamp the newer run generation.
        if super::agent_output_status_is_terminal(&manifest.status)
            || expected_generation
                .is_some_and(|generation| manifest.run_generation != generation)
        {
            return;
        }
        apply(&mut manifest);
        manifest.last_activity_at = Some(epoch_seconds_now_u64());
        let _ = write_agent_manifest(&manifest);
    });
}

/// Best-effort: stamp the manifest's `currentTool` (and append to the
/// `recentTools` rolling feed) so the parent sidebar / agent viewer can show
/// this agent's live activity. A starting tool also clears any wait-phase
/// (`currentPhase`) — the agent is demonstrably past waiting.
pub(super) fn record_current_tool(
    manifest_path: &std::path::Path,
    tool_name: &str,
    input_json: &str,
) {
    let entry = match brief_tool_arg(input_json) {
        Some(brief) => format!("{tool_name} \u{00b7} {brief}"),
        None => tool_name.to_string(),
    };
    update_manifest_best_effort(manifest_path, |manifest| {
        let now = epoch_seconds_now_u64();
        manifest.current_tool = Some(tool_name.to_string());
        manifest.current_phase = None;
        manifest.tool_calls = manifest.tool_calls.saturating_add(1);
        manifest.activity.first_task_action_at.get_or_insert(now);
        manifest.activity.last_task_progress_at = Some(now);
        if tool_name == "Skill" {
            if let Some(skill) = loaded_skill_name(input_json) {
                if !manifest.activity.loaded_skills.contains(&skill) {
                    manifest.activity.loaded_skills.push(skill);
                }
            }
        }
        manifest.recent_tools.push(entry);
        if manifest.recent_tools.len() > RECENT_TOOLS_CAP {
            let overflow = manifest.recent_tools.len() - RECENT_TOOLS_CAP;
            manifest.recent_tools.drain(..overflow);
        }
    });
}

/// Best-effort: clear `currentTool` once execution returns. Previously the
/// field survived until the next tool/terminal event, making a completed bash
/// call look like a 16-minute live operation in the HUD.
pub(super) fn record_tool_finished(manifest_path: &std::path::Path) {
    update_manifest_best_effort(manifest_path, |manifest| {
        manifest.current_tool = None;
        manifest.activity.last_task_progress_at = Some(epoch_seconds_now_u64());
    });
}

/// Stamp the beginning of a provider request, before opening its stream. If no
/// decoded event follows, `streamOpenAt` + absent `firstProviderEventAt` is the
/// observable signature of a provider/startup stall.
pub(super) fn record_agent_stream_open(
    manifest_path: &std::path::Path,
    effective_effort: Option<&str>,
    thinking_budget_tokens: Option<u32>,
) {
    update_manifest_best_effort(manifest_path, |manifest| {
        manifest
            .activity
            .stream_open_at
            .get_or_insert_with(epoch_seconds_now_u64);
        manifest.activity.effective_effort = effective_effort.map(str::to_string);
        manifest.activity.thinking_budget_tokens = thinking_budget_tokens;
    });
}

/// Stamp the first decoded provider event separately from raw transport life.
pub(super) fn record_agent_provider_event(manifest_path: &std::path::Path) {
    update_manifest_best_effort(manifest_path, |manifest| {
        let now = epoch_seconds_now_u64();
        manifest.activity.first_provider_event_at.get_or_insert(now);
        manifest.activity.last_transport_at = Some(now);
        manifest.activity.quiet_stream_since_at = None;
    });
}

/// Record an API backend's structured quiet/reconnect notice. A quiet notice is
/// evidence that keep-alives reached the client, not reasoning/task progress;
/// the startup watchdog therefore never treats this field as an extension.
pub(super) fn record_agent_stream_notice(
    manifest_path: &std::path::Path,
    notice: &core_types::StreamRetryNotice,
) {
    update_manifest_best_effort(manifest_path, |manifest| {
        let now = epoch_seconds_now_u64();
        manifest.activity.last_transport_at = Some(now);
        match notice.kind {
            core_types::StreamNoticeKind::QuietReasoning => {
                manifest.activity.quiet_stream_since_at =
                    Some(now.saturating_sub(notice.delay.as_secs()));
                manifest.activity.quiet_notice_count =
                    manifest.activity.quiet_notice_count.saturating_add(1);
                manifest.current_phase = Some(format!(
                    "reasoning stream alive · {}s+ without decoded output",
                    notice.delay.as_secs()
                ));
            }
            core_types::StreamNoticeKind::Reconnect => {
                manifest.activity.reconnect_count =
                    manifest.activity.reconnect_count.saturating_add(1);
                manifest.activity.retry_cause = Some(format!("stream_reconnect: {}", notice.label));
                manifest.current_phase = Some(format!(
                    "{} · reconnect {}/{}",
                    notice.label, notice.attempt, notice.max_attempts
                ));
            }
        }
    });
}

/// Reasoning deltas are decoded model activity (unlike keep-alives). Callers
/// throttle this write; the first delta is always recorded immediately.
pub(super) fn record_agent_reasoning_activity(manifest_path: &std::path::Path) {
    update_manifest_best_effort(manifest_path, |manifest| {
        manifest.activity.last_reasoning_at = Some(epoch_seconds_now_u64());
    });
}

/// Mark the first model-visible task action while a tool call is still being
/// streamed (or before a small text delta reaches the tail flusher).
pub(super) fn record_agent_task_activity(manifest_path: &std::path::Path) {
    update_manifest_best_effort(manifest_path, |manifest| {
        let now = epoch_seconds_now_u64();
        manifest.activity.first_task_action_at.get_or_insert(now);
        manifest.activity.last_task_progress_at = Some(now);
        manifest.current_phase = None;
    });
}

/// Structured failure classification for provider-open retries that happen
/// before a stream object exists (and therefore cannot emit a stream notice).
pub(super) fn record_agent_retry_cause(manifest_path: &std::path::Path, cause: &str) {
    update_manifest_best_effort(manifest_path, |manifest| {
        manifest.activity.retry_cause = Some(cause.to_string());
    });
}

/// Best-effort: stamp the agent's transient wait/stream phase (`waiting for
/// api slot`, `rate-limited · resumes in ~90s`, `thinking`, …) so a parked
/// agent reads as visibly alive in the HUD instead of a frozen `[running]`.
/// `None` clears the phase.
pub(super) fn record_agent_phase(manifest_path: &std::path::Path, phase: Option<&str>) {
    update_manifest_best_effort(manifest_path, |manifest| {
        manifest.current_phase = phase.map(str::to_string);
    });
}

/// Best-effort: update the model shown for a running agent after a runtime
/// quota/rate-limit fallback switches the provider client. The original
/// `requestedModel` stays unchanged; `model`/`resolvedModel` reflect where the
/// agent is actually retrying now.
pub(super) fn record_agent_runtime_model(manifest_path: &std::path::Path, model: &str) {
    update_manifest_best_effort(manifest_path, |manifest| {
        let model = model.to_string();
        manifest.model = Some(model.clone());
        manifest.resolved_model = Some(model);
    });
}

/// Best-effort: append a streamed assistant-text chunk to the manifest's
/// rolling `outputTail` (last [`OUTPUT_TAIL_CAP`] chars, trimmed on a char
/// boundary) and bump the heartbeat. Callers throttle; this just writes.
pub(super) fn append_agent_output_tail(manifest_path: &std::path::Path, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    update_manifest_best_effort(manifest_path, |manifest| {
        let now = epoch_seconds_now_u64();
        manifest.output_tail.push_str(chunk);
        manifest.current_phase = None;
        manifest.activity.first_task_action_at.get_or_insert(now);
        manifest.activity.last_task_progress_at = Some(now);
        retain_output_tail_window(&mut manifest.output_tail);
    });
}

/// Best-effort: remove a suffix previously exposed through `outputTail`.
///
/// This is used when a provider stream must retry the same turn from scratch:
/// any live partial prose from the failed attempt would otherwise be shown
/// again when the retry re-streams it. The suffix guard keeps the operation
/// conservative — if the rolling tail no longer ends with this exact attempt's
/// text, leave the manifest untouched instead of risking unrelated output.
pub(super) fn trim_agent_output_tail_suffix(manifest_path: &std::path::Path, suffix: &str) {
    if suffix.is_empty() {
        return;
    }
    update_manifest_best_effort(manifest_path, |manifest| {
        let Some(trimmed) = manifest.output_tail.strip_suffix(suffix) else {
            return;
        };
        manifest.output_tail = trimmed.to_string();
    });
}

/// One-line argument brief for the activity feed: the most identifying
/// string field of the tool input (command / path / pattern / …), trimmed
/// to a display-friendly length. `None` when the input has no such field.
fn brief_tool_arg(input_json: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(input_json).ok()?;
    let brief = [
        "command",
        "file_path",
        "path",
        "pattern",
        "query",
        "url",
        "description",
        "prompt",
    ]
    .iter()
    .find_map(|key| value.get(key).and_then(serde_json::Value::as_str))?;
    let brief = brief.trim().replace('\n', " ");
    if brief.is_empty() {
        return None;
    }
    let mut out = String::new();
    for (count, ch) in brief.chars().enumerate() {
        if count >= 60 {
            out.push('\u{2026}');
            break;
        }
        out.push(ch);
    }
    Some(out)
}

fn loaded_skill_name(input_json: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(input_json)
        .ok()?
        .get("skill")?
        .as_str()
        .map(str::trim)
        .filter(|skill| !skill.is_empty())
        .map(str::to_string)
}

fn read_activity_snapshot(manifest_path: &std::path::Path) -> Option<AgentActivitySnapshot> {
    let text = std::fs::read_to_string(manifest_path).ok()?;
    let manifest = serde_json::from_str::<AgentOutput>(&text).ok()?;
    Some(AgentActivitySnapshot {
        started_at: manifest.started_at.as_deref().and_then(|value| value.parse().ok()),
        stream_open_at: manifest.activity.stream_open_at,
        last_reasoning_at: manifest.activity.last_reasoning_at,
        first_task_action_at: manifest.activity.first_task_action_at,
        last_task_progress_at: manifest.activity.last_task_progress_at,
        current_tool: manifest.current_tool,
        effective_effort: manifest.activity.effective_effort,
        selected_model: manifest.model,
        fallback_models: manifest.activity.fallback_models,
        loaded_skills: manifest.activity.loaded_skills,
    })
}

pub(crate) fn agent_activity_snapshot_by_id(agent_id: &str) -> Option<AgentActivitySnapshot> {
    let path = super::agent_store_dir().ok()?.join(format!("{agent_id}.json"));
    read_activity_snapshot(&path)
}

#[cfg(test)]
mod activity_tests {
    use super::*;

    fn test_manifest(tag: &str) -> (std::path::PathBuf, AgentOutput) {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zo-agent-activity-{tag}-{unique}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let manifest_path = dir.join("agent-test.json");
        let output_path = dir.join("agent-test.md");
        std::fs::write(&output_path, "# agent\n").expect("write output");
        let manifest = serde_json::from_value::<AgentOutput>(serde_json::json!({
            "agentId": "agent-test",
            "name": "agent-test",
            "description": "test agent",
            "subagentType": "Explore",
            "model": "gpt-5.6-sol",
            "status": "running",
            "outputFile": output_path,
            "manifestFile": manifest_path,
            "createdAt": "100",
            "startedAt": "100"
        }))
        .expect("minimal legacy manifest should deserialize");
        (dir, manifest)
    }

    #[test]
    fn agent_manifest_lock_allows_same_thread_reentry() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let value = with_agent_manifest_lock("agent-reentrant-lock-test", || {
                with_agent_manifest_lock("agent-reentrant-lock-test", || 42)
            });
            let _ = sender.send(value);
        });

        let value = receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("same-thread manifest lock reentry must not deadlock");
        worker.join().expect("reentry worker should finish");
        assert_eq!(value, 42);
    }

    #[test]
    fn telemetry_is_backward_compatible_and_skill_receipt_is_tool_observed() {
        let (dir, manifest) = test_manifest("skill");
        assert!(manifest.activity.is_empty(), "legacy manifest defaults telemetry");
        write_agent_manifest(&manifest).expect("write manifest");
        let path = std::path::Path::new(&manifest.manifest_file);

        // Reading or mentioning a skill is not a receipt.
        record_current_tool(path, "Read", r#"{"path":"skills/karpathy/SKILL.md"}"#);
        let running = read_activity_snapshot(path).expect("read running activity snapshot");
        assert_eq!(running.current_tool.as_deref(), Some("Read"));
        record_tool_finished(path);
        // Only an actual Skill tool call is recorded, and repeats dedupe.
        record_current_tool(path, "Skill", r#"{"skill":"karpathy"}"#);
        record_tool_finished(path);
        record_current_tool(path, "Skill", r#"{"skill":"karpathy"}"#);
        record_tool_finished(path);

        let current: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(current.activity.loaded_skills, vec!["karpathy"]);
        assert!(current.current_tool.is_none(), "finished tools never remain stale");
        assert_eq!(current.tool_calls, 3);
        assert!(current.activity.first_task_action_at.is_some());
        assert!(current.activity.last_task_progress_at.is_some());
        let snapshot = read_activity_snapshot(path).expect("read activity snapshot");
        assert_eq!(
            snapshot.last_task_progress_at,
            current.activity.last_task_progress_at
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn quiet_transport_notice_is_not_reasoning_or_task_progress() {
        let (dir, manifest) = test_manifest("quiet");
        write_agent_manifest(&manifest).expect("write manifest");
        let path = std::path::Path::new(&manifest.manifest_file);
        record_agent_stream_open(path, Some("xhigh"), Some(10_000));
        record_agent_stream_notice(
            path,
            &core_types::StreamRetryNotice {
                kind: core_types::StreamNoticeKind::QuietReasoning,
                label: core_types::QUIET_REASONING_LABEL,
                attempt: 0,
                max_attempts: 0,
                delay: std::time::Duration::from_secs(60),
            },
        );

        let quiet: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert!(quiet.activity.last_transport_at.is_some());
        assert!(quiet.activity.quiet_stream_since_at.is_some());
        assert!(quiet.activity.last_reasoning_at.is_none());
        assert!(quiet.activity.first_task_action_at.is_none());

        record_agent_reasoning_activity(path);
        let reasoning: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert!(reasoning.activity.last_reasoning_at.is_some());
        assert!(reasoning.activity.first_task_action_at.is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn stale_run_generation_ignores_live_activity() {
        let (dir, mut manifest) = test_manifest("generation");
        manifest.run_generation = 2;
        write_agent_manifest(&manifest).expect("write current generation");
        let path = std::path::Path::new(&manifest.manifest_file);

        with_agent_run_generation(1, || {
            record_agent_provider_event(path);
            record_current_tool(path, "bash", r#"{"command":"stale"}"#);
            record_agent_phase(path, Some("stale"));
        });
        let stale: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(stale.run_generation, 2);
        assert!(stale.activity.is_empty());
        assert!(stale.current_tool.is_none());
        assert!(stale.current_phase.is_none());
        assert_eq!(stale.tool_calls, 0);

        with_agent_run_generation(2, || {
            record_current_tool(path, "bash", r#"{"command":"current"}"#);
        });
        let current: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(current.current_tool.as_deref(), Some("bash"));
        assert_eq!(current.tool_calls, 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn concurrent_resume_installs_one_generation_owner() {
        let (dir, mut manifest) = test_manifest("concurrent-resume");
        manifest.status = String::from("completed");
        manifest.completed_at = Some(String::from("200"));
        manifest.run_generation = 4;
        write_agent_manifest(&manifest).expect("write terminal manifest");

        let claims = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rollbacks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for follow_up in ["first", "second"] {
            let manifest = manifest.clone();
            let claims = claims.clone();
            let rollbacks = rollbacks.clone();
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                persist_agent_resumed_state_with(
                    &manifest,
                    follow_up,
                    |_| {
                        claims.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    },
                    |_| {
                        rollbacks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    },
                )
            }));
        }
        barrier.wait();
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().expect("resume worker"))
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(claims.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(rollbacks.load(std::sync::atomic::Ordering::SeqCst), 0);

        let current = load_bound_manifest(&manifest).expect("load resumed manifest");
        assert_eq!(current.status, "running");
        assert_eq!(current.run_generation, 5);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn atomic_writer_cleans_temp_after_rename_failure() {
        let (dir, _) = test_manifest("atomic-cleanup");
        let target = dir.join("target.json");
        let error = write_regular_file_atomically_with(
            &target,
            "test",
            b"secret",
            |_, _| Err(std::io::Error::other("forced rename failure")),
        )
        .expect_err("forced rename failure must surface");
        assert!(error.contains("forced rename failure"));
        let leaked_temp = std::fs::read_dir(&dir)
            .expect("read test dir")
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"));
        assert!(!leaked_temp, "failed atomic writes must remove temporary files");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn terminal_manifest_ignores_late_live_activity() {
        let (dir, mut manifest) = test_manifest("terminal");
        manifest.status = String::from("stopped");
        manifest.completed_at = Some(String::from("200"));
        manifest.last_activity_at = Some(200);
        write_agent_manifest(&manifest).expect("write terminal manifest");
        let path = std::path::Path::new(&manifest.manifest_file);

        record_agent_provider_event(path);
        record_agent_reasoning_activity(path);
        record_agent_task_activity(path);
        record_current_tool(path, "bash", r#"{"command":"late"}"#);
        record_agent_phase(path, Some("thinking"));

        let current: AgentOutput = serde_json::from_str(
            &std::fs::read_to_string(&manifest.manifest_file).expect("read manifest"),
        )
        .expect("parse manifest");
        assert_eq!(current.status, "stopped");
        assert_eq!(current.completed_at.as_deref(), Some("200"));
        assert_eq!(current.last_activity_at, Some(200));
        assert!(current.activity.is_empty());
        assert!(current.current_tool.is_none());
        assert!(current.current_phase.is_none());
        assert_eq!(current.tool_calls, 0);
        std::fs::remove_dir_all(dir).ok();
    }
}
