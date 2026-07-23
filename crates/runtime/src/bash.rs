use std::collections::BTreeMap;
use std::env;
use std::io;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command as TokioCommand;
use tokio::runtime::Builder;
use tokio::time::timeout;

use crate::sandbox::{
    resolve_sandbox_status_for_request, sandbox_scratch_dirs, wrap_sandbox_command,
    FilesystemIsolationMode, SandboxConfig, SandboxStatus,
};
use crate::ConfigLoader;

/// Input schema for the built-in bash execution tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BashCommandInput {
    pub command: String,
    pub timeout: Option<u64>,
    pub description: Option<String>,
    #[serde(rename = "run_in_background")]
    pub run_in_background: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    pub dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "namespaceRestrictions")]
    pub namespace_restrictions: Option<bool>,
    #[serde(rename = "isolateNetwork")]
    pub isolate_network: Option<bool>,
    #[serde(rename = "filesystemMode")]
    pub filesystem_mode: Option<FilesystemIsolationMode>,
    #[serde(rename = "allowedMounts")]
    pub allowed_mounts: Option<Vec<String>>,
    /// Working directory for the spawned command. `None` (the default and the
    /// wire behavior) runs in the live process cwd; a per-agent tool context
    /// sets it so `isolation:"worktree"` executes inside the agent's worktree
    /// without mutating the shared process cwd. Not part of the tool's public
    /// JSON schema — injected server-side after deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<std::path::PathBuf>,
}

/// Output returned from a bash tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BashCommandOutput {
    pub stdout: String,
    pub stderr: String,
    #[serde(rename = "rawOutputPath")]
    pub raw_output_path: Option<String>,
    pub interrupted: bool,
    #[serde(rename = "isImage")]
    pub is_image: Option<bool>,
    #[serde(rename = "backgroundTaskId")]
    pub background_task_id: Option<String>,
    #[serde(rename = "backgroundedByUser")]
    pub backgrounded_by_user: Option<bool>,
    #[serde(rename = "assistantAutoBackgrounded")]
    pub assistant_auto_backgrounded: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    pub dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "returnCodeInterpretation")]
    pub return_code_interpretation: Option<String>,
    #[serde(rename = "noOutputExpected")]
    pub no_output_expected: Option<bool>,
    #[serde(rename = "structuredContent")]
    pub structured_content: Option<Vec<serde_json::Value>>,
    #[serde(rename = "persistedOutputPath")]
    pub persisted_output_path: Option<String>,
    #[serde(rename = "persistedOutputSize")]
    pub persisted_output_size: Option<u64>,
    #[serde(rename = "sandboxStatus")]
    pub sandbox_status: Option<SandboxStatus>,
    /// Non-blocking advisory raised when the command matches a known
    /// destructive pattern (e.g. `rm -rf /`, `dd if=…`, a fork bomb). The
    /// command still ran; this surfaces the risk to the user/model. Omitted
    /// from the wire format when absent.
    #[serde(
        rename = "safetyWarning",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub safety_warning: Option<String>,
}

/// Default wall-clock deadline applied to a bash call that carries no explicit
/// `timeout`. Without it a headless `zo -p` run can wedge forever: a `curl`
/// to a slow-but-alive host keeps the child + pipe open, so `child.wait()`
/// never returns and the collect future awaits with no deadline. 120 s matches
/// the bash tool's documented default and is generous for normal builds while
/// still bounding the worst case. Override with `ZO_BASH_TIMEOUT_MS`.
const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;

/// Smallest timeout we accept as a genuine deadline. A model that sends
/// `timeout: 10` (ms — almost always a ms/s unit slip, `10` meaning *seconds*)
/// would otherwise SIGKILL every command before it can even spawn, turning each
/// tool call into a silent retry loop. Sub-second values are treated as bogus
/// and fall back to the default, mirroring `default_bash_timeout_ms`'s
/// "0 must not instant-timeout" rule.
const MIN_BASH_TIMEOUT_MS: u64 = 1_000;

/// Resolve the default bash timeout, honoring a `ZO_BASH_TIMEOUT_MS`
/// override. A malformed or non-positive value falls back to the compiled
/// default rather than degrading to an instant timeout on every command.
fn default_bash_timeout_ms() -> u64 {
    env::var("ZO_BASH_TIMEOUT_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .unwrap_or(DEFAULT_BASH_TIMEOUT_MS)
}

/// Polling slice for the background-task watcher: how quickly a `TaskStop`
/// kills the process and how promptly the terminal status lands.
const BACKGROUND_WATCH_SLICE: Duration = Duration::from_millis(200);

/// Stream one pipe of a background task into its registry output log.
fn spawn_background_pipe_drain<R: io::Read + Send + 'static>(
    registry: crate::task_registry::TaskRegistry,
    task_id: String,
    pipe: R,
    label: Option<&'static str>,
) {
    std::thread::spawn(move || {
        let reader = io::BufReader::new(pipe);
        for line in io::BufRead::lines(reader) {
            let Ok(line) = line else { break };
            let entry = match label {
                Some(label) => format!("[{label}] {line}\n"),
                None => format!("{line}\n"),
            };
            if registry.append_output(&task_id, &entry).is_err() {
                break;
            }
        }
    });
}

/// Drain both stdio pipes of a background child into the task output log.
fn spawn_background_task_drain(
    registry: crate::task_registry::TaskRegistry,
    task_id: String,
    child: &mut std::process::Child,
) {
    if let Some(stdout) = child.stdout.take() {
        spawn_background_pipe_drain(registry.clone(), task_id.clone(), stdout, None);
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_background_pipe_drain(registry, task_id, stderr, Some("stderr"));
    }
}

/// Watch a background child until it exits or its task is stopped.
///
/// * Process exits → `[exit N]` appended, status `Completed` / `Failed`.
/// * Task flips to `Stopped` (a `TaskStop` call) → the process is killed, so
///   stopping a background bash actually terminates it instead of orphaning.
fn spawn_background_task_watcher(
    registry: crate::task_registry::TaskRegistry,
    task_id: String,
    mut child: std::process::Child,
) {
    use crate::task_registry::TaskStatus;
    std::thread::spawn(move || {
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let code = status.code();
                    let _ = registry.append_output(
                        &task_id,
                        &format!(
                            "[exit {}]\n",
                            code.map_or_else(|| "signal".to_string(), |c| c.to_string())
                        ),
                    );
                    let final_status = if status.success() {
                        TaskStatus::Completed
                    } else {
                        TaskStatus::Failed
                    };
                    // A user-stopped task keeps its `Stopped` verdict.
                    if registry
                        .get(&task_id)
                        .is_none_or(|task| task.status != TaskStatus::Stopped)
                    {
                        let _ = registry.set_status(&task_id, final_status);
                    }
                    registry.confirm_background_exit(&task_id);
                    break;
                }
                Ok(None) => {
                    if registry
                        .get(&task_id)
                        .is_some_and(|task| task.status == TaskStatus::Stopped)
                    {
                        let _ = child.kill();
                        if child.wait().is_ok() {
                            let _ = registry.append_output(&task_id, "[killed by TaskStop]\n");
                            registry.confirm_background_exit(&task_id);
                            break;
                        }
                    }
                    std::thread::sleep(BACKGROUND_WATCH_SLICE);
                }
                Err(error) => {
                    let error = classify_enospc(error, &std::env::temp_dir());
                    let _ = registry.append_output(
                        &task_id,
                        &format!("[watch error: {error}]\n"),
                    );
                    let _ = registry.set_status(&task_id, TaskStatus::Failed);
                    break;
                }
            }
        }
    });
}

/// Re-describe an out-of-disk (`ENOSPC`) failure with an actionable cleanup
/// hint; every other error passes through unchanged. Matched by errno as well
/// as [`io::ErrorKind::StorageFull`] because the kind mapping differs across
/// platforms — a bare `os error 28` gives the user nothing to reclaim.
fn classify_enospc(error: io::Error, cwd: &std::path::Path) -> io::Error {
    let out_of_space =
        error.raw_os_error() == Some(28) || error.kind() == io::ErrorKind::StorageFull;
    if !out_of_space {
        return error;
    }
    io::Error::new(
        error.kind(),
        format!(
            "no space left on device while running in {}: free disk space and retry — the usual reclaim targets are Rust target/ build dirs, scratch under {}, and orphaned worktrees ({error})",
            cwd.display(),
            std::env::temp_dir().display(),
        ),
    )
}

/// Bytes still available on the filesystem holding `dir`; `None` when the
/// probe fails or the platform has no probe — the guard must fail open.
/// Public so agent/worktree/self-improve spawn paths can preflight the same
/// way without re-implementing the probe.
#[cfg(unix)]
pub fn available_disk_bytes(dir: &std::path::Path) -> Option<u64> {
    let stat = rustix::fs::statvfs(dir).ok()?;
    Some(stat.f_bavail.saturating_mul(stat.f_frsize))
}

#[cfg(not(unix))]
pub fn available_disk_bytes(_dir: &std::path::Path) -> Option<u64> {
    None
}

/// Warn-only headroom floor: agent builds die on ENOSPC well before literal
/// zero, so surface the pressure while there is still room to act. A warning
/// (never a refusal) keeps a conservative threshold from blocking normal work.
const LOW_DISK_WARN_BYTES: u64 = 1024 * 1024 * 1024;

fn low_disk_warning_at(available: u64, threshold: u64, dir: &std::path::Path) -> Option<String> {
    (available < threshold).then(|| {
        format!(
            "disk space low: {}MB available on the filesystem holding {} — heavy commands may fail with ENOSPC; reclaim Rust target/ build dirs, temp scratch, or orphaned worktrees before starting large builds",
            available / (1024 * 1024),
            dir.display(),
        )
    })
}

/// Probe both the working directory and the temp filesystem (worktrees and
/// scratch land there) and return whichever has the least headroom.
fn tightest_disk(dir: &std::path::Path) -> Option<(u64, std::path::PathBuf)> {
    let temp = std::env::temp_dir();
    [dir, temp.as_path()]
        .into_iter()
        .filter_map(|probe| available_disk_bytes(probe).map(|bytes| (bytes, probe.to_path_buf())))
        .min_by_key(|(bytes, _)| *bytes)
}

/// Warn about the working or temp filesystem, whichever has the least
/// headroom.
pub fn low_disk_warning(dir: &std::path::Path) -> Option<String> {
    let (available, tightest) = tightest_disk(dir)?;
    low_disk_warning_at(available, LOW_DISK_WARN_BYTES, &tightest)
}

/// Below this floor the run is refused outright: work started here dies on
/// ENOSPC mid-flight anyway, and the refusal names what to reclaim. The gap
/// between the two thresholds separates "warn and continue" from hard-stop.
pub(crate) const HARD_MIN_DISK_BYTES: u64 = 128 * 1024 * 1024;

pub(crate) fn disk_critical_error(
    available: u64,
    floor: u64,
    dir: &std::path::Path,
) -> Option<io::Error> {
    (available < floor).then(|| {
        io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "refusing to run: only {}MB left on the filesystem holding {} — free disk space first (reclaim Rust target/ build dirs, temp scratch, or orphaned worktrees)",
                available / (1024 * 1024),
                dir.display(),
            ),
        )
    })
}

/// Hard-stop preflight at the shared execution chokepoint.
fn refuse_when_disk_critical(cwd: &std::path::Path) -> io::Result<()> {
    match tightest_disk(cwd).and_then(|(available, tightest)| {
        disk_critical_error(available, HARD_MIN_DISK_BYTES, &tightest)
    }) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Merge a low-disk warning into an output that does not already carry a
/// stronger safety warning.
fn attach_disk_warning(
    mut output: BashCommandOutput,
    warning: Option<String>,
) -> BashCommandOutput {
    if output.safety_warning.is_none() {
        output.safety_warning = warning;
    }
    output
}

/// Executes a shell command with the requested sandbox settings.
///
/// Background commands (`run_in_background: true`) on this entry point keep
/// the legacy fire-and-forget behavior (PID returned, output discarded). The
/// tool path uses [`execute_bash_with_tasks`] so background runs register in
/// the session [`TaskRegistry`] and stay observable via `TaskOutput`.
pub fn execute_bash(input: BashCommandInput) -> io::Result<BashCommandOutput> {
    execute_bash_with_tasks(input, None, None)
}

/// [`execute_bash`] plus an optional task registry for background runs.
///
/// With a registry, `run_in_background: true` creates a real task: stdout and
/// stderr stream into the task's output log, `backgroundTaskId` is the
/// **task id** (`task_…`) that `TaskOutput`/`TaskGet` accept, and `TaskStop`
/// terminates the process (a watcher thread observes the `Stopped` status).
/// This closes the old design gap where the tool returned a bare OS PID that
/// no other tool could look up — with the output already discarded.
// The dispatch reads as one unit (validation → preflight → three exit paths);
// splitting it would thread five locals through helper signatures.
#[allow(clippy::too_many_lines)]
pub fn execute_bash_with_tasks(
    input: BashCommandInput,
    tasks: Option<&crate::task_registry::TaskRegistry>,
    session_id: Option<&str>,
) -> io::Result<BashCommandOutput> {
    // Safety backstop (goal G8): refuse the catastrophic command set at the
    // single execution chokepoint — in every permission mode and even on the
    // paths that bypass the permission enforcer (workflow check commands,
    // enforcer-less dispatch). `check_destructive` is the single source of
    // truth; the enforcer surfaces the same block as a clean PermissionDenied
    // when it is wired (`PermissionEnforcer::check_bash`).
    if let crate::bash_validation::ValidationResult::Block { reason } =
        crate::bash_validation::check_destructive(&input.command)
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("refused to run a catastrophic command — {reason}"),
        ));
    }

    // A per-agent tool context can pin the working directory (worktree
    // isolation); absent that, fall back to the live process cwd so the
    // default behavior is unchanged.
    let cwd = match input.cwd.clone() {
        Some(dir) => dir,
        None => env::current_dir()?,
    };
    let sandbox_status = sandbox_status_for_input(&input, &cwd);
    // Disk preflight at the shared chokepoint: every agent/workflow bash run
    // funnels through here, so ENOSPC pressure is surfaced BEFORE the long
    // build that would otherwise die on it — a warning with headroom left,
    // a refusal below the critical floor.
    let low_disk = low_disk_warning(&cwd);
    refuse_when_disk_critical(&cwd)?;

    if input.run_in_background.unwrap_or(false) {
        let mut command = prepare_command(&input.command, &cwd, &sandbox_status, false)?;

        // Registry-integrated path: the background run becomes a real task so
        // `TaskOutput`/`TaskGet`/`TaskStop` work on it.
        if let Some(registry) = tasks {
            let mut child = command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|error| classify_enospc(error, &cwd))?;
            let pid = child.id();
            let task = registry.create_background_process(
                &input.command,
                Some("background bash"),
                session_id,
            );
            let _ = registry.set_status(&task.task_id, crate::task_registry::TaskStatus::Running);
            let _ = registry.append_output(
                &task.task_id,
                &format!("$ {}\n[pid {pid}]\n", input.command),
            );
            spawn_background_task_drain(registry.clone(), task.task_id.clone(), &mut child);
            spawn_background_task_watcher(registry.clone(), task.task_id.clone(), child);

            return Ok(BashCommandOutput {
                stdout: format!(
                    "Started background task {} (pid {pid}). Poll output with TaskOutput(task_id), stop with TaskStop.",
                    task.task_id
                ),
                stderr: String::new(),
                raw_output_path: None,
                interrupted: false,
                is_image: None,
                background_task_id: Some(task.task_id),
                backgrounded_by_user: Some(false),
                assistant_auto_backgrounded: Some(false),
                dangerously_disable_sandbox: input.dangerously_disable_sandbox,
                return_code_interpretation: None,
                no_output_expected: Some(false),
                structured_content: None,
                persisted_output_path: None,
                persisted_output_size: None,
                sandbox_status: Some(sandbox_status),
                safety_warning: low_disk,
            });
        }

        // Legacy fire-and-forget (no registry in scope): PID only, output
        // discarded. Internal callers that never poll keep this contract.
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| classify_enospc(error, &cwd))?;

        return Ok(BashCommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            raw_output_path: None,
            interrupted: false,
            is_image: None,
            background_task_id: Some(child.id().to_string()),
            backgrounded_by_user: Some(false),
            assistant_auto_backgrounded: Some(false),
            dangerously_disable_sandbox: input.dangerously_disable_sandbox,
            return_code_interpretation: None,
            no_output_expected: Some(true),
            structured_content: None,
            persisted_output_path: None,
            persisted_output_size: None,
            sandbox_status: Some(sandbox_status),
            safety_warning: low_disk,
        });
    }

    // The handle owns registry cleanup on every exit path. Reader tasks get
    // cloneable writers, while this guard remains on the dispatch thread until
    // the foreground command has completed, timed out, or unwound.
    let live_output = crate::live_output::register(crate::live_output::dispatch_key().as_deref());
    let live_writer = live_output.writer();

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // Already inside a tokio runtime — use `block_in_place` which lets
        // the current worker thread run blocking code while migrating other
        // async tasks to sibling workers.  This avoids the previous approach
        // of `std::thread::spawn` + a fresh single-threaded runtime per bash
        // command, eliminating ~1ms OS thread creation overhead (P0 L2).
        //
        // For current-thread runtimes (tests), fall back to a dedicated
        // thread since `block_in_place` panics on single-threaded executors.
        let result = if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
            let live_writer = live_writer.clone();
            std::thread::spawn(move || {
                let runtime = Builder::new_current_thread().enable_all().build()?;
                runtime.block_on(execute_bash_async(
                    input,
                    sandbox_status,
                    cwd,
                    live_writer,
                ))
            })
            .join()
            .map_err(|_| io::Error::other("bash execution thread panicked"))?
        } else {
            tokio::task::block_in_place(|| {
                handle.block_on(execute_bash_async(
                    input,
                    sandbox_status,
                    cwd,
                    live_writer,
                ))
            })
        };
        return result.map(|output| attach_disk_warning(output, low_disk));
    }

    let runtime = Builder::new_current_thread().enable_all().build()?;
    runtime
        .block_on(execute_bash_async(
            input,
            sandbox_status,
            cwd,
            live_writer,
        ))
        .map(|output| attach_disk_warning(output, low_disk))
}

async fn execute_bash_async(
    input: BashCommandInput,
    sandbox_status: SandboxStatus,
    cwd: std::path::PathBuf,
    live_output: crate::live_output::LiveOutputWriter,
) -> io::Result<BashCommandOutput> {
    let mut command = prepare_tokio_command(&input.command, &cwd, &sandbox_status, true)?;

    // Stream stdout/stderr instead of `command.output()` (which buffers the
    // entire payload before we get a chance to truncate). With unbounded
    // buffering a `find /` or `cat huge.log` could consume GB of RAM. `read_capped`
    // keeps memory bounded to `MAX_OUTPUT_BYTES + TAIL_OUTPUT_BYTES` per stream
    // (head buffer + trailing ring) and silently drains the overflow so the child
    // doesn't block on a full pipe — `render_capped` then keeps both the head and
    // the tail (where errors live) instead of head-only.
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| classify_enospc(error, &cwd))?;

    // Captured before `child` is borrowed by the collect future. On Unix this is
    // the process-group leader's pid (== pgid), used to reap the whole tree on a
    // timeout below.
    #[cfg(unix)]
    let child_pid = child.id();

    let stdout_reader = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("child stdout pipe missing"))?;
    let stderr_reader = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("child stderr pipe missing"))?;
    // Box::pin the join future: it nests three sub-futures plus their
    // buffers (~16 KB on the stack). Heap-allocating it once keeps this large
    // frame off the caller's stack and avoids repeated moves while `timeout`
    // polls it, instead of carrying the whole future by value.
    let collect_all = Box::pin(async {
        let (out, err, status) = tokio::join!(
            read_capped(
                stdout_reader,
                MAX_OUTPUT_BYTES,
                TAIL_OUTPUT_BYTES,
                live_output.clone(),
                crate::live_output::LiveOutputWriter::append_stdout,
            ),
            read_capped(
                stderr_reader,
                MAX_OUTPUT_BYTES,
                TAIL_OUTPUT_BYTES,
                live_output,
                crate::live_output::LiveOutputWriter::append_stderr,
            ),
            child.wait(),
        );
        Ok::<_, io::Error>((out?, err?, status?))
    });

    // The deadline is unconditional: a model-supplied `timeout` stays
    // authoritative, but its absence — or an implausibly small value, almost
    // always a ms/s unit slip like `timeout: 10` that would SIGKILL the command
    // before it spawns — falls back to the default instead of an instant or
    // unbounded await, so neither a slow-but-alive child nor a bogus deadline can
    // wedge the run. `run_in_background` commands early-return above and stay exempt.
    let effective_timeout_ms = input
        .timeout
        .filter(|&ms| ms >= MIN_BASH_TIMEOUT_MS)
        .unwrap_or_else(default_bash_timeout_ms);
    let output_result = if let Ok(result) =
        timeout(Duration::from_millis(effective_timeout_ms), collect_all).await
    {
        let (stdout_cap, stderr_cap, status) = result?;
        (stdout_cap, stderr_cap, status, false)
    } else {
        // Timeout: signal the whole process group first so backgrounded
        // grandchildren are reaped too, then return — dropping `child`
        // triggers `kill_on_drop` as the leader backstop (WI-G).
        #[cfg(unix)]
        terminate_process_group(child_pid);
        return Ok(BashCommandOutput {
            stdout: String::new(),
            stderr: format!("Command exceeded timeout of {effective_timeout_ms} ms"),
            raw_output_path: None,
            interrupted: true,
            is_image: None,
            background_task_id: None,
            backgrounded_by_user: None,
            assistant_auto_backgrounded: None,
            dangerously_disable_sandbox: input.dangerously_disable_sandbox,
            return_code_interpretation: Some(String::from("timeout")),
            no_output_expected: Some(true),
            structured_content: None,
            persisted_output_path: None,
            persisted_output_size: None,
            sandbox_status: Some(sandbox_status),
            safety_warning: None,
        });
    };

    let (stdout_cap, stderr_cap, status, interrupted) = output_result;
    let (stdout_head, stdout_tail, stdout_total) = stdout_cap;
    let (stderr_head, stderr_tail, stderr_total) = stderr_cap;
    let stdout = render_capped(&stdout_head, &stdout_tail, stdout_total);
    let stderr = render_capped(&stderr_head, &stderr_tail, stderr_total);
    let no_output_expected = Some(stdout.trim().is_empty() && stderr.trim().is_empty());
    let return_code_interpretation = match status.code() {
        Some(0) => None,
        Some(code) => Some(format!("exit_code:{code}")),
        // `ExitStatus::code()` is absent when the shell is terminated by a
        // signal. `None` is otherwise reserved for a clean exit, so preserve a
        // non-success interpretation instead of reporting SIGKILL/OOM as green.
        None => Some(String::from("signal")),
    };

    Ok(BashCommandOutput {
        stdout,
        stderr,
        raw_output_path: None,
        interrupted,
        is_image: None,
        background_task_id: None,
        backgrounded_by_user: None,
        assistant_auto_backgrounded: None,
        dangerously_disable_sandbox: input.dangerously_disable_sandbox,
        return_code_interpretation,
        no_output_expected,
        structured_content: None,
        persisted_output_path: None,
        persisted_output_size: None,
        sandbox_status: Some(sandbox_status),
        safety_warning: None,
    })
}

fn sandbox_status_for_input(input: &BashCommandInput, cwd: &std::path::Path) -> SandboxStatus {
    static CACHED_SANDBOX_CONFIG: OnceLock<SandboxConfig> = OnceLock::new();

    let config = CACHED_SANDBOX_CONFIG
        .get_or_init(|| {
            ConfigLoader::default_for(cwd).load().map_or_else(
                |_| SandboxConfig::default(),
                |runtime_config| runtime_config.sandbox().clone(),
            )
        })
        .clone();
    let request = config.resolve_request(
        input.dangerously_disable_sandbox.map(|disabled| !disabled),
        input.namespace_restrictions,
        input.isolate_network,
        input.filesystem_mode,
        input.allowed_mounts.clone(),
    );
    resolve_sandbox_status_for_request(&request, cwd)
}

/// User-declared `settings.env`, cached once per process exactly like
/// [`sandbox_status_for_input`]'s sandbox config. Injected into every bash child
/// so an `env` block in settings.json actually reaches the shell (CC parity)
/// instead of only being surfaced by `/config env`. Cache keyed off the first
/// bash call's `cwd`, matching the sandbox cache's single-workspace assumption.
fn cached_settings_env(cwd: &std::path::Path) -> &'static BTreeMap<String, String> {
    static CACHED_SETTINGS_ENV: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    CACHED_SETTINGS_ENV.get_or_init(|| {
        ConfigLoader::default_for(cwd).load().map_or_else(
            |_| BTreeMap::new(),
            |runtime_config| runtime_config.env().clone(),
        )
    })
}

/// Environment that pins child processes to non-interactive credential flow.
///
/// The bash tool runs inside a TUI whose renderer owns the terminal. A command
/// that needs credentials — classically `git push` over HTTPS — otherwise opens
/// the *controlling terminal* (`/dev/tty`) directly to print `Username for
/// 'https://github.com':` and read the answer. Because `Stdio::null()` only
/// redirects the inherited stdin (not `/dev/tty`), that prompt lands straight on
/// the ratatui frame: it garbles in-flight glyphs (CJK / box-drawing break) and
/// silently steals keystrokes while the command hangs forever waiting on input.
///
/// `GIT_TERMINAL_PROMPT=0` makes git fail fast to *stderr* (captured) with
/// "terminal prompts disabled" instead of touching the terminal; `GCM_INTERACTIVE`
/// stops Git Credential Manager from popping its own prompt. Legitimate
/// non-interactive auth (token-in-URL, credential helper, ssh-agent) is
/// untouched — only the dead-end interactive path is closed.
const NONINTERACTIVE_ENV: &[(&str, &str)] =
    &[("GIT_TERMINAL_PROMPT", "0"), ("GCM_INTERACTIVE", "never")];

/// Git environment variables that silently redirect *every* git invocation at a
/// different repository. Stripped from every shell child so an inherited
/// `GIT_DIR`/`GIT_WORK_TREE` — leaked from the parent process, or from a
/// worktree-isolated agent's environment — cannot reach another checkout. Only
/// the *inherited* value is removed (before `settings.env` and the login
/// profile run), so a user that re-exports them from their own config is still
/// respected.
const STRIPPED_GIT_ENV: &[&str] = &["GIT_DIR", "GIT_WORK_TREE"];

/// Apply [`NONINTERACTIVE_ENV`] and detach stdin so no child can reach the
/// controlling terminal to prompt. Generic over the env/stdin surface shared by
/// `std` and `tokio` `Command`s.
fn harden_noninteractive_std(prepared: &mut Command) {
    prepared.envs(NONINTERACTIVE_ENV.iter().copied());
    prepared.stdin(Stdio::null());
}

fn harden_noninteractive_tokio(prepared: &mut TokioCommand) {
    prepared.envs(NONINTERACTIVE_ENV.iter().copied());
    prepared.stdin(Stdio::null());
}

fn prepare_command(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    create_dirs: bool,
) -> io::Result<Command> {
    fail_closed_if_sandbox_unavailable(sandbox_status)?;
    if create_dirs {
        prepare_sandbox_dirs(cwd);
    }

    let mut prepared = if let Some(launcher) = wrap_sandbox_command(command, cwd, sandbox_status) {
        let mut prepared = Command::new(launcher.program);
        prepared.args(launcher.args);
        prepared.current_dir(cwd);
        prepared.envs(launcher.env);
        prepared
    } else {
        let mut prepared = Command::new("sh");
        prepared.arg("-lc").arg(command).current_dir(cwd);
        if sandbox_status.filesystem_active {
            let (home, tmp) = sandbox_scratch_dirs(cwd);
            prepared.env("HOME", home);
            prepared.env("TMPDIR", tmp);
        }
        prepared
    };
    // Drop inherited GIT_DIR/GIT_WORK_TREE before settings.env re-adds any, so a
    // leaked value can't silently redirect git while an explicit user override
    // still applies.
    for var in STRIPPED_GIT_ENV {
        prepared.env_remove(var);
    }
    // User settings.env first, so the non-interactive safety vars applied by
    // `harden_*` (GIT_TERMINAL_PROMPT/GCM_INTERACTIVE) always win and a stray
    // settings.env can't re-open the /dev/tty prompt that corrupts the TUI.
    prepared.envs(cached_settings_env(cwd));
    harden_noninteractive_std(&mut prepared);
    Ok(prepared)
}

fn prepare_tokio_command(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    create_dirs: bool,
) -> io::Result<TokioCommand> {
    fail_closed_if_sandbox_unavailable(sandbox_status)?;
    if create_dirs {
        prepare_sandbox_dirs(cwd);
    }

    let mut prepared = if let Some(launcher) = wrap_sandbox_command(command, cwd, sandbox_status) {
        let mut prepared = TokioCommand::new(launcher.program);
        prepared.args(launcher.args);
        prepared.current_dir(cwd);
        prepared.envs(launcher.env);
        prepared
    } else {
        let mut prepared = TokioCommand::new("sh");
        prepared.arg("-lc").arg(command).current_dir(cwd);
        if sandbox_status.filesystem_active {
            let (home, tmp) = sandbox_scratch_dirs(cwd);
            prepared.env("HOME", home);
            prepared.env("TMPDIR", tmp);
        }
        prepared
    };
    // Drop inherited GIT_DIR/GIT_WORK_TREE before settings.env (see
    // `prepare_command`), so a leaked value can't silently redirect git.
    for var in STRIPPED_GIT_ENV {
        prepared.env_remove(var);
    }
    // User settings.env first (see `prepare_command`): safety hardening wins.
    prepared.envs(cached_settings_env(cwd));
    harden_noninteractive_tokio(&mut prepared);
    // Make the child its own process-group leader so a timeout/cancel can signal
    // the whole tree (`kill -- -pgid`), not just the direct child. Without this,
    // `kill_on_drop` SIGKILLs only the `sh`/`unshare` pid and any backgrounded
    // grandchildren orphan (WI-G). Unix-only; Windows keeps `kill_on_drop`.
    #[cfg(unix)]
    prepared.process_group(0);
    Ok(prepared)
}

/// Kill an entire process group on Unix — the child plus any grandchildren it
/// spawned. The child is its own group leader (`process_group(0)`), so its pid
/// doubles as the pgid and a negated pid signals the whole group. Mirrors the
/// dependency-free `kill` shell-out the compat-harness runner already uses:
/// SIGTERM for a graceful stop, then SIGKILL after a short grace. `kill_on_drop`
/// remains the backstop for the leader itself.
#[cfg(unix)]
fn terminate_process_group(pid: Option<u32>) {
    let Some(pid) = pid else {
        return;
    };
    let group = format!("-{pid}");
    let _ = std::process::Command::new("kill")
        .arg("-TERM")
        .arg("--")
        .arg(&group)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    std::thread::sleep(Duration::from_millis(50));
    let _ = std::process::Command::new("kill")
        .arg("-KILL")
        .arg("--")
        .arg(&group)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn fail_closed_if_sandbox_unavailable(sandbox_status: &SandboxStatus) -> io::Result<()> {
    // The per-platform policy lives in `sandbox::current_sandbox_unavailability`
    // (WI-E): a Linux host whose `unshare` is genuinely broken — or macOS with
    // `ZO_MACOS_SEATBELT` opted in but `sandbox-exec` missing — is a real,
    // fixable failure and fails closed. The default macOS/Windows path has no
    // native isolation to fail on, so it degrades to filesystem-scratch isolation
    // instead (failing closed there would reject every command on a dev machine —
    // the "frozen" 5-minute retry loop bug).
    if let Some(reason) = crate::sandbox::current_sandbox_unavailability(sandbox_status) {
        return Err(io::Error::other(format!(
            "sandbox requested but unavailable: {reason}. Set dangerouslyDisableSandbox=true or sandbox.enabled=false to run without sandbox."
        )));
    }
    Ok(())
}

/// Ensure the external sandbox scratch directories exist before a command
/// redirects `HOME`/`TMPDIR` into them. These live outside the working tree
/// (see [`sandbox_scratch_dirs`]) so they never pollute the target repo.
fn prepare_sandbox_dirs(cwd: &std::path::Path) {
    let (home, tmp) = sandbox_scratch_dirs(cwd);
    let _ = std::fs::create_dir_all(home);
    let _ = std::fs::create_dir_all(tmp);
}

#[cfg(test)]
mod tests {
    use super::{
        default_bash_timeout_ms, execute_bash, execute_bash_with_tasks, BashCommandInput,
        DEFAULT_BASH_TIMEOUT_MS,
    };
    use crate::sandbox::{FilesystemIsolationMode, SandboxStatus};
    use crate::task_registry::{TaskRegistry, TaskStatus};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    /// Serializes the tests that mutate the process-global
    /// `ZO_BASH_TIMEOUT_MS` so they never observe each other's value.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn enospc_spawn_error_gains_cleanup_hint() {
        let classified = super::classify_enospc(
            std::io::Error::from_raw_os_error(28),
            std::path::Path::new("/work/repo"),
        );
        let message = classified.to_string();
        assert!(message.contains("no space left on device"), "{message}");
        assert!(message.contains("/work/repo"), "{message}");
        assert!(message.contains("target/"), "{message}");
    }

    #[test]
    fn non_enospc_spawn_error_passes_through_unchanged() {
        let original = std::io::Error::from_raw_os_error(2);
        let expected = original.to_string();
        let classified =
            super::classify_enospc(original, std::path::Path::new("/work/repo"));
        assert_eq!(classified.to_string(), expected);
        assert_eq!(classified.raw_os_error(), Some(2));
    }

    #[test]
    fn low_disk_threshold_logic_is_deterministic() {
        let dir = std::path::Path::new("/work/repo");
        let warning = super::low_disk_warning_at(5, 10, dir).expect("below threshold warns");
        assert!(warning.contains("disk space low"), "{warning}");
        assert!(warning.contains("target/"), "{warning}");
        assert!(super::low_disk_warning_at(10, 10, dir).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn disk_probe_reports_available_bytes_for_real_paths() {
        let available = super::available_disk_bytes(&std::env::temp_dir());
        assert!(available.is_some_and(|bytes| bytes > 0), "{available:?}");
    }

    #[test]
    fn critical_disk_floor_refuses_with_reclaim_hint() {
        let dir = std::path::Path::new("/work/repo");
        let error = super::disk_critical_error(64 * 1024 * 1024, 128 * 1024 * 1024, dir)
            .expect("below the floor refuses");
        assert!(error.to_string().contains("refusing to run"), "{error}");
        assert!(error.to_string().contains("target/"), "{error}");
        assert!(super::disk_critical_error(256 * 1024 * 1024, 128 * 1024 * 1024, dir).is_none());
    }

    /// A `BashCommandInput` with the given command/timeout and otherwise
    /// default (sandboxed, foreground) settings — keeps the timeout tests
    /// focused on the deadline behavior rather than field boilerplate.
    fn input(command: &str, timeout: Option<u64>) -> BashCommandInput {
        BashCommandInput {
            command: String::from(command),
            timeout,
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
            cwd: None,
        }
    }

    fn wait_for<T>(deadline: Duration, mut probe: impl FnMut() -> Option<T>) -> T {
        let until = Instant::now() + deadline;
        loop {
            if let Some(value) = probe() {
                return value;
            }
            assert!(Instant::now() < until, "condition not reached in time");
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    /// 설계결함 회귀: 백그라운드 bash 는 `TaskRegistry` 에 등록되고,
    /// `backgroundTaskId` 는 `TaskOutput` 이 받는 **task id** 이며, stdout 과
    /// stderr 가 태스크 출력으로 스트리밍되고 종료시 상태가 종결된다.
    #[test]
    fn background_bash_registers_task_and_streams_output() {
        let registry = TaskRegistry::new_in_memory();
        let mut request = input("sleep 1; echo bg-out; echo bg-err 1>&2", None);
        request.run_in_background = Some(true);
        let live = registry.live_background_process_count(Some("session-a"));
        let output =
            execute_bash_with_tasks(request, Some(&registry), Some("session-a"))
                .expect("background spawn works");
        assert_eq!(live.load(), 1, "successful launch increments once");
        let task_id = output.background_task_id.expect("task id present");
        assert!(
            task_id.starts_with("task_"),
            "must be a registry task id, not a bare PID: {task_id}"
        );

        wait_for(Duration::from_secs(5), || {
            registry
                .get(&task_id)
                .filter(|task| task.status == TaskStatus::Completed)
        });
        let log = registry.output(&task_id).expect("task output");
        assert!(log.contains("bg-out"), "stdout streamed: {log}");
        assert!(log.contains("[stderr] bg-err"), "stderr streamed: {log}");
        assert!(log.contains("[exit 0]"), "exit recorded: {log}");
        assert_eq!(live.load(), 0, "watcher terminal outcome decrements once");
    }

    /// `TaskStop` 이 백그라운드 프로세스를 실제로 죽인다 (워처가 Stopped
    /// 상태를 보고 kill).
    #[test]
    fn task_stop_kills_background_bash() {
        let registry = TaskRegistry::new_in_memory();
        let mut request = input("sleep 30", None);
        request.run_in_background = Some(true);
        let live = registry.live_background_process_count(Some("session-a"));
        let output =
            execute_bash_with_tasks(request, Some(&registry), Some("session-a"))
                .expect("background spawn works");
        assert_eq!(live.load(), 1);
        let task_id = output.background_task_id.expect("task id present");

        registry.stop(&task_id).expect("stop succeeds");
        assert_eq!(
            live.load(),
            1,
            "TaskStop keeps the process live until the watcher reaps it"
        );
        wait_for(Duration::from_secs(5), || {
            registry
                .output(&task_id)
                .ok()
                .filter(|log| log.contains("[killed by TaskStop]") && live.load() == 0)
        });
        assert_eq!(live.load(), 0, "the watcher confirms exit exactly once");
        let task = registry.get(&task_id).expect("task exists");
        assert_eq!(task.status, TaskStatus::Stopped);
    }

    #[test]
    fn background_spawn_failure_never_increments_live_count() {
        let registry = TaskRegistry::new_in_memory();
        let live = registry.live_background_process_count(Some("session-a"));
        let mut request = input("echo unreachable", None);
        request.run_in_background = Some(true);
        let missing_cwd = std::env::temp_dir().join(format!(
            "zo-missing-background-cwd-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&missing_cwd);
        request.cwd = Some(missing_cwd);

        let result = execute_bash_with_tasks(request, Some(&registry), Some("session-a"));

        assert!(result.is_err(), "missing cwd must fail process spawn");
        assert_eq!(live.load(), 0);
        assert!(registry.is_empty(), "failed spawn must not create task history");
    }

    #[test]
    fn default_timeout_interrupts_runaway_command() {
        // A `timeout: None` call must still inherit a deadline so a headless
        // run can never hang on a slow-but-alive child. Inject a short default
        // via the env override; a 5 s sleep must trip it.
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("ZO_BASH_TIMEOUT_MS", "250");
        let output = execute_bash(input("sleep 5", None)).expect("bash command should execute");
        std::env::remove_var("ZO_BASH_TIMEOUT_MS");

        assert!(
            output.interrupted,
            "missing timeout must fall back to the default deadline"
        );
        assert_eq!(
            output.return_code_interpretation.as_deref(),
            Some("timeout")
        );
    }

    #[test]
    fn default_timeout_leaves_fast_commands_untouched() {
        // The unconditional deadline must not clip a command that finishes
        // well within it. Uses the real 120 s default (no env mutation).
        let _guard = ENV_LOCK.lock().unwrap();
        let output =
            execute_bash(input("printf 'hello'", None)).expect("bash command should execute");

        assert_eq!(output.stdout, "hello");
        assert!(
            !output.interrupted,
            "a fast command must not be interrupted by the default"
        );
    }

    #[cfg(unix)]
    #[test]
    fn signal_terminated_command_is_not_reported_as_success() {
        let output = execute_bash(input("kill -KILL $$", Some(5_000)))
            .expect("the shell should report its signal termination");

        assert_eq!(
            output.return_code_interpretation.as_deref(),
            Some("signal")
        );
    }

    #[test]
    fn explicit_timeout_overrides_short_default() {
        // A model-supplied `timeout` stays authoritative even when the default
        // is tightened: a 1 s sleep under a short default but explicit 5 s
        // timeout must complete normally.
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("ZO_BASH_TIMEOUT_MS", "200");
        let output = execute_bash(input("sleep 1; printf 'done'", Some(5_000)))
            .expect("bash should execute");
        std::env::remove_var("ZO_BASH_TIMEOUT_MS");

        assert!(
            !output.interrupted,
            "explicit timeout must override a shorter default"
        );
        assert_eq!(output.stdout, "done");
    }

    #[test]
    fn default_timeout_env_override_parsing() {
        // Pure resolution test: a malformed or non-positive override falls back
        // to the compiled default; a valid one wins.
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("ZO_BASH_TIMEOUT_MS");
        assert_eq!(default_bash_timeout_ms(), DEFAULT_BASH_TIMEOUT_MS);

        std::env::set_var("ZO_BASH_TIMEOUT_MS", "5000");
        assert_eq!(default_bash_timeout_ms(), 5_000);

        std::env::set_var("ZO_BASH_TIMEOUT_MS", "0");
        assert_eq!(
            default_bash_timeout_ms(),
            DEFAULT_BASH_TIMEOUT_MS,
            "0 must not instant-timeout"
        );

        std::env::set_var("ZO_BASH_TIMEOUT_MS", "not-a-number");
        assert_eq!(default_bash_timeout_ms(), DEFAULT_BASH_TIMEOUT_MS);

        std::env::remove_var("ZO_BASH_TIMEOUT_MS");
    }

    #[test]
    fn implausibly_small_timeout_falls_back_to_default() {
        // A model that sends `timeout: 10` (ms — almost always a ms/s unit slip)
        // must not turn every command into an instant SIGKILL. Sub-second values
        // are discarded in favor of the default deadline, so the command runs to
        // completion instead of being killed before it can spawn.
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("ZO_BASH_TIMEOUT_MS");
        let output =
            execute_bash(input("printf 'survived'", Some(10))).expect("bash should execute");

        assert!(
            !output.interrupted,
            "a sub-second timeout must fall back to the default, not instant-kill"
        );
        assert_eq!(output.stdout, "survived");
    }

    #[test]
    fn executes_simple_command() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
            cwd: None,
        })
        .expect("bash command should execute");

        assert_eq!(output.stdout, "hello");
        assert!(!output.interrupted);
        assert!(output.sandbox_status.is_some());
    }

    #[test]
    fn runs_in_explicit_cwd_when_set() {
        // A per-agent tool context can pin the working directory (worktree
        // isolation); `pwd` must report that directory, not the live process
        // cwd. Default `cwd: None` is exercised by every other test here.
        let dir = std::env::temp_dir().join(format!(
            "zo-bash-cwd-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create cwd");
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.clone());

        let output = execute_bash(BashCommandInput {
            command: String::from("pwd"),
            timeout: Some(2_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: None,
            isolate_network: None,
            filesystem_mode: None,
            allowed_mounts: None,
            cwd: Some(canonical.clone()),
        })
        .expect("bash command should execute");
        let reported = std::path::PathBuf::from(output.stdout.trim());
        let reported = reported.canonicalize().unwrap_or(reported);
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            reported, canonical,
            "explicit cwd must be the command's working directory"
        );
    }

    #[test]
    fn git_prompts_are_disabled_for_child_processes() {
        // Regression: `git push` over HTTPS used to open the controlling
        // terminal to print `Username for 'https://github.com':`, corrupting the
        // TUI frame (garbled CJK) and hanging. The runner now pins
        // `GIT_TERMINAL_PROMPT=0` so git fails fast to stderr instead — verify
        // the env actually reaches the spawned shell.
        let output = execute_bash(BashCommandInput {
            command: String::from("printf '%s' \"$GIT_TERMINAL_PROMPT\""),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
            cwd: None,
        })
        .expect("bash command should execute");
        assert_eq!(
            output.stdout, "0",
            "GIT_TERMINAL_PROMPT must be 0 so children never prompt on the terminal"
        );
    }

    #[test]
    fn child_stdin_is_detached_so_readers_get_eof_not_the_tui() {
        // stdin is `/dev/null`, so a command that reads stdin (`cat`) hits EOF
        // immediately and returns instead of blocking on — and stealing input
        // from — the TUI's terminal. A hang would trip the timeout and surface
        // as `interrupted`.
        let output = execute_bash(BashCommandInput {
            command: String::from("cat; printf 'END'"),
            timeout: Some(2_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
            cwd: None,
        })
        .expect("bash command should execute");
        assert!(!output.interrupted, "stdin EOF must not hang the command");
        assert_eq!(output.stdout, "END");
    }

    #[test]
    fn disables_sandbox_when_requested() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: None,
            isolate_network: None,
            filesystem_mode: None,
            allowed_mounts: None,
            cwd: None,
        })
        .expect("bash command should execute");

        assert!(!output.sandbox_status.expect("sandbox status").enabled);
    }

    #[test]
    fn sandbox_fallback_fails_closed_on_linux_degrades_elsewhere() {
        let workspace = std::env::temp_dir().join(format!(
            "zo-bash-fail-closed-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&workspace);
        std::fs::create_dir_all(&workspace).expect("create workspace");
        let marker = workspace.join("marker");
        let status = SandboxStatus {
            enabled: true,
            fallback_reason: Some(
                "namespace isolation unavailable (requires Linux with `unshare`)".to_string(),
            ),
            ..SandboxStatus::default()
        };

        let result = super::prepare_command("printf 'ran' > marker", &workspace, &status, false);
        let marker_exists = marker.exists();
        let _ = std::fs::remove_dir_all(&workspace);

        if cfg!(target_os = "linux") {
            // A genuinely broken `unshare` on Linux is fixable — fail closed.
            let error = result.expect_err("Linux: sandbox fallback must fail closed before spawn");
            assert_eq!(error.kind(), std::io::ErrorKind::Other);
            assert!(
                error
                    .to_string()
                    .contains("sandbox requested but unavailable"),
                "error should explain the sandbox failure, got: {error}"
            );
        } else {
            // macOS/Windows have no namespace sandbox to begin with — refusing
            // would wedge every command, so preparation must degrade, not error.
            result.expect("non-Linux: must degrade to filesystem isolation, not fail closed");
        }
        assert!(
            !marker_exists,
            "prepare_command must never spawn the command itself"
        );
    }

    #[tokio::test]
    async fn executes_simple_command_inside_existing_runtime() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello from runtime'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
            cwd: None,
        })
        .expect("bash command should execute inside runtime");

        assert_eq!(output.stdout, "hello from runtime");
        assert!(!output.interrupted);
        assert!(output.sandbox_status.is_some());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_kills_process_group() {
        use std::process::Stdio;
        use std::time::{Duration, Instant};

        // Background a long-lived grandchild, record its pid, then block in the
        // foreground past the timeout. `kill_on_drop` alone SIGKILLs only the
        // `sh` leader and would orphan the backgrounded `sleep`; the WI-G group
        // kill (`kill -- -pgid`) must reap it too.
        // No parentheses/special chars in the path — it is interpolated into a
        // `sh -lc` command, and `ThreadId(..)`'s Debug form would break parsing.
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let pidfile =
            std::env::temp_dir().join(format!("zo-pgroup-{}-{unique}.pid", std::process::id()));
        let _ = std::fs::remove_file(&pidfile);
        let command = format!("sleep 30 & echo $! > {}; sleep 30", pidfile.display());

        let output = execute_bash(BashCommandInput {
            command,
            // Must be >= MIN_BASH_TIMEOUT_MS or it is ignored for the 120s default.
            timeout: Some(1_500),
            description: None,
            run_in_background: Some(false),
            // Disable zo's sandbox so the test exercises only the kill path,
            // not scratch-dir redirection (memory: spawn tests use this flag).
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
            cwd: None,
        })
        .expect("bash command should execute");
        assert!(output.interrupted, "command should have timed out");

        let pid_text = std::fs::read_to_string(&pidfile).expect("grandchild pidfile written");
        let gpid: u32 = pid_text.trim().parse().expect("valid grandchild pid");
        let _ = std::fs::remove_file(&pidfile);

        // The backgrounded grandchild would live ~30s; the group kill must have
        // reaped it within a short poll window.
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut alive = true;
        while Instant::now() < deadline {
            let still_alive = std::process::Command::new("kill")
                .arg("-0")
                .arg(gpid.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false);
            if !still_alive {
                alive = false;
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        if alive {
            // Don't leak the runaway if we're about to fail.
            let _ = std::process::Command::new("kill")
                .arg("-KILL")
                .arg(gpid.to_string())
                .status();
        }
        assert!(
            !alive,
            "process-group kill must reap the backgrounded grandchild (pid {gpid})"
        );
    }

    /// Regression guard: preparing the sandbox scratch directories must not
    /// create `.sandbox-home` / `.sandbox-tmp` inside the working tree. A
    /// non-interactive agent run would otherwise leave these artifacts in the
    /// target repository, polluting `git status` (observed in the Claude Code
    /// vs Zo benchmark).
    #[test]
    fn prepare_sandbox_dirs_does_not_pollute_workspace() {
        let workspace = std::env::temp_dir().join(format!(
            "zo-bash-pollution-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&workspace);
        std::fs::create_dir_all(&workspace).expect("create workspace");

        super::prepare_sandbox_dirs(&workspace);

        let polluted_home = workspace.join(".sandbox-home").exists();
        let polluted_tmp = workspace.join(".sandbox-tmp").exists();
        let _ = std::fs::remove_dir_all(&workspace);

        assert!(
            !polluted_home,
            ".sandbox-home must not be created inside the workspace"
        );
        assert!(
            !polluted_tmp,
            ".sandbox-tmp must not be created inside the workspace"
        );
    }

    /// Regression guard: a sandboxed command whose `$HOME` / `$TMPDIR` writes
    /// would land in the workspace must instead write to an external scratch
    /// location, leaving the working tree untouched.
    #[test]
    fn sandboxed_home_writes_stay_outside_workspace() {
        let workspace = std::env::temp_dir().join(format!(
            "zo-bash-home-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&workspace);
        std::fs::create_dir_all(&workspace).expect("create workspace");

        let status = super::sandbox_status_for_input(
            &BashCommandInput {
                command: String::new(),
                timeout: None,
                description: None,
                run_in_background: Some(false),
                dangerously_disable_sandbox: Some(false),
                namespace_restrictions: Some(false),
                isolate_network: Some(false),
                filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
                allowed_mounts: None,
                cwd: None,
            },
            &workspace,
        );
        // The fallback (non-Linux / no namespaces) path redirects HOME/TMPDIR
        // via env vars; assert those point outside the workspace.
        let command =
            super::prepare_command("printf hi > \"$HOME/marker\"", &workspace, &status, true)
                .expect("sandbox env fallback without fallback reason should prepare");
        let home = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("HOME"))
            .and_then(|(_, value)| value)
            .map(std::path::PathBuf::from);

        let polluted = workspace.join(".sandbox-home").exists();
        let _ = std::fs::remove_dir_all(&workspace);

        if let Some(home) = home {
            assert!(
                !home.starts_with(&workspace),
                "sandbox HOME ({home:?}) must live outside the workspace ({workspace:?})"
            );
        }
        assert!(
            !polluted,
            ".sandbox-home must not be created inside the workspace"
        );
    }
}

/// Maximum output bytes before truncation (16 KiB, matching upstream).
const MAX_OUTPUT_BYTES: usize = 16_384;

/// When output overflows `MAX_OUTPUT_BYTES`, this many bytes from the END are
/// preserved alongside the head. A command's actual failure — the compiler
/// error, the panic, the failing test summary — almost always sits at the tail,
/// and head-only truncation discarded it, forcing the model to re-run the whole
/// command (a second full invocation re-billing its output plus another turn).
const TAIL_OUTPUT_BYTES: usize = 4 * 1024;

/// Read at most `cap` bytes from `reader` into a `Vec<u8>`; any further
/// bytes are read and discarded so the child process can finish writing
/// (a full pipe would otherwise block the producer forever).
///
/// Trade-off: we still pay the kernel-side I/O of reading every byte, but
/// the heap footprint stays bounded at `cap`. For a `find /` style output
/// this turns a multi-GB resident-set spike into ~16 KiB.
async fn read_capped<R>(
    mut reader: R,
    cap: usize,
    tail_cap: usize,
    live_output: crate::live_output::LiveOutputWriter,
    append_live: fn(&crate::live_output::LiveOutputWriter, &[u8]),
) -> io::Result<(Vec<u8>, Vec<u8>, usize)>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut head: Vec<u8> = Vec::with_capacity(cap.min(8192));
    let mut tail: Vec<u8> = Vec::new();
    let mut total: usize = 0;
    let mut scratch = [0u8; 8192];
    loop {
        let n = reader.read(&mut scratch).await?;
        if n == 0 {
            break;
        }
        let chunk = &scratch[..n];
        append_live(&live_output, chunk);
        if head.len() < cap {
            let room = cap - head.len();
            head.extend_from_slice(&chunk[..room.min(n)]);
        }
        retain_tail(&mut tail, chunk, tail_cap);
        total = total.saturating_add(n);
    }
    Ok((head, tail, total))
}

/// Keep at most `tail_cap` trailing bytes of a stream in `tail`, chunk-wise so a
/// multi-GB drain stays O(total) (no per-byte shifting): the common 8 KiB read
/// is `>= tail_cap`, so it replaces the buffer outright; only a small final read
/// drains the front.
fn retain_tail(tail: &mut Vec<u8>, chunk: &[u8], tail_cap: usize) {
    if tail_cap == 0 {
        return;
    }
    if chunk.len() >= tail_cap {
        tail.clear();
        tail.extend_from_slice(&chunk[chunk.len() - tail_cap..]);
        return;
    }
    let overflow = (tail.len() + chunk.len()).saturating_sub(tail_cap);
    if overflow > 0 {
        tail.drain(0..overflow);
    }
    tail.extend_from_slice(chunk);
}

/// Render the captured stream within `MAX_OUTPUT_BYTES`, preserving BOTH the head
/// and the tail (where errors live) with an explicit middle-elision marker.
/// `head` holds up to `MAX_OUTPUT_BYTES` bytes; `tail` holds the last
/// `TAIL_OUTPUT_BYTES`; `total` is the full byte count seen. When the whole
/// output fit in `head` it is returned verbatim (lossless).
fn render_capped(head: &[u8], tail: &[u8], total: usize) -> String {
    if total <= head.len() {
        // Everything fit in the head buffer — lossless.
        return String::from_utf8_lossy(head).into_owned();
    }
    let head_keep = MAX_OUTPUT_BYTES.saturating_sub(TAIL_OUTPUT_BYTES);
    let head_slice = &head[..head_keep.min(head.len())];
    let elided = total
        .saturating_sub(head_slice.len())
        .saturating_sub(tail.len());
    format!(
        "{}\n\n[output truncated — kept first {} B and last {} B of {} B total; {} B elided in the middle. A command's error usually appears at the end.]\n\n{}",
        String::from_utf8_lossy(head_slice),
        head_slice.len(),
        tail.len(),
        total,
        elided,
        String::from_utf8_lossy(tail),
    )
}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    #[test]
    fn short_output_unchanged() {
        let head = b"hello world".to_vec();
        assert_eq!(render_capped(&head, &[], head.len()), "hello world");
    }

    #[test]
    fn within_cap_is_lossless() {
        let head = b"a".repeat(MAX_OUTPUT_BYTES);
        assert_eq!(render_capped(&head, &[], head.len()).len(), MAX_OUTPUT_BYTES);
    }

    #[test]
    fn overflow_keeps_head_and_tail_with_marker() {
        // Head buffer holds the first MAX_OUTPUT_BYTES; tail holds the last
        // TAIL_OUTPUT_BYTES; the original was much larger.
        let head = b"H".repeat(MAX_OUTPUT_BYTES);
        let tail = b"TAIL_ERROR".repeat(TAIL_OUTPUT_BYTES / 10);
        let total = 1_000_000;
        let out = render_capped(&head, &tail, total);
        assert!(out.contains("[output truncated"), "marker present");
        assert!(out.contains("1000000 B total"), "reports the true total");
        // The tail (where the error lives) survives.
        assert!(out.ends_with("TAIL_ERROR"), "tail preserved at the end");
        // The head is trimmed to leave room for the tail.
        assert!(out.starts_with(&"H".repeat(64)), "head preserved at the start");
    }

    #[test]
    fn retain_tail_keeps_last_bytes_chunkwise() {
        let mut tail = Vec::new();
        // A chunk larger than tail_cap replaces the buffer with its own suffix.
        retain_tail(&mut tail, &b"0123456789".repeat(2), 8);
        assert_eq!(tail, b"23456789");
        // A small follow-up chunk drains the front to stay within tail_cap.
        retain_tail(&mut tail, b"AB", 8);
        assert_eq!(tail, b"456789AB");
    }
}
