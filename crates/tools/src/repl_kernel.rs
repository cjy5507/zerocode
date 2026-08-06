//! Session-persistent Python kernel behind the `REPL` tool.
//!
//! The one-shot REPL path (`python3 -c <code>`) forgets everything between
//! calls, so every follow-up pays to re-read and re-parse state the model
//! already computed. This module keeps ONE Python process per (session,
//! language) alive across tool calls: variables, imports, functions, and
//! parsed data survive turns — and context compaction, since the state lives
//! in the child process rather than in the transcript.
//!
//! Design constraints, in order:
//! - **No runtime dependencies.** The bootstrap is stdlib-only Python fed via
//!   `-c`; there is no ipykernel/Jupyter/venv to provision, so the kernel
//!   works on any machine where `python3` resolves.
//! - **An exception is a result, not a crash.** A raised exception returns
//!   `ok:false` with the traceback while the kernel (and its namespace)
//!   stays alive — matching interactive REPL semantics.
//! - **A dead kernel is disposable.** Timeout or EOF kills and unregisters
//!   the kernel; the next call spawns a fresh one. State loss is reported in
//!   the error text so the model knows to rebuild.
//! - **Orphan-free.** The bootstrap's read loop ends on stdin EOF, so kernels
//!   exit on their own when the zo process goes away.
//!
//! Protocol: one JSON object per line on stdin (`{"id","code"}`), one JSON
//! object per line on stdout (`{"id","ok","value","stdout","stderr","error"}`).
//! stdout/stderr of the executed code are captured and never interleave with
//! protocol frames.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;

use super::error::ToolError;

/// Default per-exec deadline. A persistent kernel read with no deadline would
/// hang the tool forever on an infinite loop, so unlike the one-shot path the
/// persistent path always has one; mirrors the `bash` tool's default.
const DEFAULT_EXEC_TIMEOUT: Duration = Duration::from_millis(120_000);

/// Byte caps applied on the Rust side before the result enters the transcript.
/// Generous enough for real data inspection, bounded enough that one stray
/// `print(huge)` cannot flood a turn. The Python side additionally caps the
/// repr it produces (`MAX_VALUE_REPR_CHARS` in the bootstrap).
const MAX_STREAM_BYTES: usize = 200_000;

/// Spawn-to-first-frame deadline. Interpreter startup is tens of
/// milliseconds; seconds mean a broken install and should fail loudly.
const SPAWN_PROBE_TIMEOUT: Duration = Duration::from_millis(10_000);

/// Most kernels one zo process keeps alive at once. Sessions are serial in
/// practice, but `/clear` mints a new session id while the old session's
/// kernel would otherwise idle forever — beyond this bound the
/// longest-idle kernel is killed when a new one is admitted.
const MAX_KERNELS: usize = 8;

/// Stdlib-only bootstrap. Executes each request in a single shared namespace
/// and evaluates a trailing expression like a REPL. `__name__` is set so
/// `if __name__ == "__main__"` blocks behave as in a script.
const PYTHON_BOOTSTRAP: &str = r#"
import ast, contextlib, io, json, sys, traceback

MAX_VALUE_REPR_CHARS = 10_000
ns = {"__name__": "__main__"}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    out, err = io.StringIO(), io.StringIO()
    resp = {"id": req.get("id"), "ok": True, "value": None, "error": None}
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            tree = ast.parse(req.get("code", ""), "<zo-repl>", "exec")
            value = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop(-1).value)
                if tree.body:
                    exec(compile(tree, "<zo-repl>", "exec"), ns)
                value = eval(compile(last, "<zo-repl>", "eval"), ns)
            else:
                exec(compile(tree, "<zo-repl>", "exec"), ns)
            if value is not None:
                rendered = repr(value)
                if len(rendered) > MAX_VALUE_REPR_CHARS:
                    rendered = rendered[:MAX_VALUE_REPR_CHARS] + "…(truncated)"
                resp["value"] = rendered
    except BaseException:
        resp["ok"] = False
        resp["error"] = traceback.format_exc()
    resp["stdout"] = out.getvalue()
    resp["stderr"] = err.getvalue()
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"#;

/// One live kernel: the child process, its stdin, and the reader-thread
/// channel its stdout frames arrive on. Held behind a per-kernel mutex so two
/// concurrent tool calls against the same session serialize (one namespace,
/// one cell at a time — the same rule Jupyter enforces) without blocking
/// kernels of other sessions.
struct KernelHandle {
    child: Child,
    stdin: ChildStdin,
    frames: Receiver<String>,
    /// Monotonic request counter; responses are matched on it so a stale
    /// frame from a timed-out call can never satisfy a later call.
    next_request: u64,
    started: Instant,
    /// Last successful exec — the LRU axis for [`MAX_KERNELS`] eviction.
    last_used: Instant,
}

/// Result of one persistent execution, shaped by the caller into the tool's
/// JSON reply.
#[derive(Debug)]
pub(crate) struct KernelExecOutput {
    pub ok: bool,
    pub value: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub duration_ms: u128,
    /// Age of the kernel process at reply time — lets the tool output say
    /// "same kernel as before" without the model tracking it.
    pub kernel_uptime_secs: u64,
    pub fresh_kernel: bool,
}

#[derive(Debug, Deserialize)]
struct KernelFrame {
    id: u64,
    ok: bool,
    value: Option<String>,
    error: Option<String>,
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
}

/// Why one exec failed, typed so the caller can tell "this kernel is dead —
/// respawning is safe" apart from "the USER'S code hung" (a timeout), which
/// must never be silently re-run. String matching here would repeat the
/// overflow-classifier landmine: a message-shape change silently turning a
/// recovery path off.
#[derive(Debug)]
enum KernelError {
    /// The child rejected input or exited: the process is gone, the code
    /// never ran to completion in it, and a fresh kernel may retry safely.
    Dead(String),
    /// The code ran past its deadline. The kernel is discarded, but the same
    /// code must not be auto-retried — it would just hang again.
    Timeout(u128),
}

impl KernelError {
    fn message(&self) -> String {
        match self {
            Self::Dead(detail) => detail.clone(),
            Self::Timeout(ms) => format!("kernel execution exceeded timeout of {ms} ms"),
        }
    }
}

fn registry() -> &'static Mutex<HashMap<String, Arc<Mutex<KernelHandle>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Mutex<KernelHandle>>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Execute `code` in the persistent kernel for `scope`, spawning one if
/// needed. `reset` discards any existing kernel first. `scope` is the
/// isolation key — callers pass the session id so concurrent sessions (and
/// sub-agents with their own session ids) never share a namespace.
pub(crate) fn execute_persistent_python(
    scope: &str,
    code: &str,
    program: &str,
    cwd: Option<&std::path::Path>,
    timeout: Option<Duration>,
    reset: bool,
) -> Result<KernelExecOutput, ToolError> {
    let timeout = timeout.unwrap_or(DEFAULT_EXEC_TIMEOUT);
    if reset {
        remove_kernel(scope);
    }

    // Two attempts, but only for a DEAD kernel: a concurrent call on the same
    // scope may have discarded the kernel between this call taking the Arc and
    // acquiring its mutex (the discarding call holds the mutex first), leaving
    // a killed child whose stdin returns EPIPE. That is transport failure, not
    // a property of the user's code, so one respawn-and-retry heals it. A
    // timeout never retries — the code would just hang again.
    let mut attempted_respawn = false;
    loop {
        let (kernel, fresh_kernel) = {
            let mut map = registry()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(existing) = map.get(scope) {
                (Arc::clone(existing), false)
            } else {
                evict_longest_idle_if_full(&mut map);
                let handle = spawn_kernel(program, cwd)?;
                let handle = Arc::new(Mutex::new(handle));
                map.insert(scope.to_string(), Arc::clone(&handle));
                (handle, true)
            }
        };

        let mut guard = kernel
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let started = Instant::now();
        match exec_on_kernel(&mut guard, code, timeout) {
            Ok(frame) => {
                guard.last_used = Instant::now();
                return Ok(KernelExecOutput {
                    ok: frame.ok,
                    value: frame.value,
                    stdout: cap_stream(frame.stdout),
                    stderr: cap_stream(frame.stderr),
                    error: frame.error,
                    duration_ms: started.elapsed().as_millis(),
                    kernel_uptime_secs: guard.started.elapsed().as_secs(),
                    fresh_kernel,
                });
            }
            Err(error) => {
                // The kernel is unusable: kill it and unregister so the next
                // call starts clean. Unregister only OUR handle — a concurrent
                // call may already have replaced the map entry with a fresh
                // kernel that must not be torn down.
                let _ = guard.child.kill();
                let _ = guard.child.wait();
                drop(guard);
                remove_kernel_handle(scope, &kernel);

                if matches!(error, KernelError::Dead(_)) && !attempted_respawn {
                    attempted_respawn = true;
                    continue;
                }
                return Err(ToolError::Execution(format!(
                    "{} — the kernel was discarded and its variables are lost; \
                     the next REPL call starts a fresh kernel",
                    error.message()
                )));
            }
        }
    }
}

/// Drop `scope`'s kernel if present, killing the process.
pub(crate) fn remove_kernel(scope: &str) {
    let handle = {
        let mut map = registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.remove(scope)
    };
    if let Some(handle) = handle {
        let mut guard = handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = guard.child.kill();
        let _ = guard.child.wait();
    }
}

/// Kill and unregister the longest-idle kernel when the registry is at
/// [`MAX_KERNELS`]. Only idle kernels are candidates: a kernel whose mutex is
/// held is mid-exec, and `try_lock` skipping it is what keeps eviction from
/// ever tearing down live work. All-busy at the cap leaves the cap soft — one
/// extra kernel beats blocking a session behind another session's cell.
fn evict_longest_idle_if_full(map: &mut HashMap<String, Arc<Mutex<KernelHandle>>>) {
    if map.len() < MAX_KERNELS {
        return;
    }
    let victim = map
        .iter()
        .filter_map(|(scope, handle)| {
            let guard = handle.try_lock().ok()?;
            Some((scope.clone(), guard.last_used))
        })
        .min_by_key(|(_, last_used)| *last_used)
        .map(|(scope, _)| scope);
    if let Some(scope) = victim {
        if let Some(handle) = map.remove(&scope) {
            // try_lock only: a blocking lock here would hold the REGISTRY
            // mutex while waiting out another call's exec (up to its full
            // timeout), stalling every session's kernel ops. If a racer
            // grabbed the kernel between the probe above and now (it cloned
            // the Arc before we took the map lock), the kernel is merely left
            // unregistered — it idles until zo exits and its stdin EOFs, a
            // bounded leak instead of a global stall.
            if let Ok(mut guard) = handle.try_lock() {
                let _ = guard.child.kill();
                let _ = guard.child.wait();
            }
        }
    }
}

/// Unregister `scope` only if the map still holds exactly `expected` — the
/// compare-and-remove a failing call uses so it never tears down a fresh
/// kernel a concurrent call has already installed in its place.
fn remove_kernel_handle(scope: &str, expected: &Arc<Mutex<KernelHandle>>) {
    let mut map = registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if map
        .get(scope)
        .is_some_and(|current| Arc::ptr_eq(current, expected))
    {
        map.remove(scope);
    }
}

fn spawn_kernel(program: &str, cwd: Option<&std::path::Path>) -> Result<KernelHandle, ToolError> {
    let mut command = Command::new(program);
    command
        .args(["-u", "-c", PYTHON_BOOTSTRAP])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // Same scoping rule as the one-shot path: the parent session's todo store
    // is a zo implementation detail and must not leak into children.
    command.env_remove("ZO_TODO_STORE");
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let mut child = command
        .spawn()
        .map_err(|e| ToolError::Execution(format!("failed to spawn python kernel: {e}")))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| ToolError::Execution("kernel stdin unavailable".into()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ToolError::Execution("kernel stdout unavailable".into()))?;

    let (sender, frames) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("zo-repl-kernel-reader".into())
        .spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if sender.send(line.trim_end().to_string()).is_err() {
                            break;
                        }
                    }
                }
            }
        })
        .map_err(|e| ToolError::Execution(format!("failed to start kernel reader: {e}")))?;

    let mut handle = KernelHandle {
        child,
        stdin,
        frames,
        next_request: 0,
        started: Instant::now(),
        last_used: Instant::now(),
    };
    // Readiness probe: a no-op cell proves the interpreter parsed the
    // bootstrap and is serving frames, so a broken install fails here with a
    // clear message instead of as a timeout on the user's first real cell.
    exec_on_kernel(&mut handle, "None", SPAWN_PROBE_TIMEOUT).map_err(|e| {
        let _ = handle.child.kill();
        let _ = handle.child.wait();
        ToolError::Execution(format!(
            "python kernel failed its readiness probe: {}",
            e.message()
        ))
    })?;
    Ok(handle)
}

fn exec_on_kernel(
    kernel: &mut KernelHandle,
    code: &str,
    timeout: Duration,
) -> Result<KernelFrame, KernelError> {
    kernel.next_request += 1;
    let id = kernel.next_request;
    let request = json!({ "id": id, "code": code }).to_string();
    kernel
        .stdin
        .write_all(request.as_bytes())
        .and_then(|()| kernel.stdin.write_all(b"\n"))
        .and_then(|()| kernel.stdin.flush())
        .map_err(|e| KernelError::Dead(format!("kernel is not accepting input: {e}")))?;

    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(KernelError::Timeout(timeout.as_millis()));
        }
        match kernel.frames.recv_timeout(remaining) {
            Ok(line) => {
                // Frames from a request that already timed out are drained and
                // dropped; only the current id answers this call.
                if let Ok(frame) = serde_json::from_str::<KernelFrame>(&line) {
                    if frame.id == id {
                        return Ok(frame);
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                return Err(KernelError::Timeout(timeout.as_millis()));
            }
            Err(RecvTimeoutError::Disconnected) => {
                let status = kernel
                    .child
                    .try_wait()
                    .ok()
                    .flatten()
                    .map_or_else(|| "unknown".to_string(), |s| s.to_string());
                return Err(KernelError::Dead(format!(
                    "kernel process exited mid-call (status: {status})"
                )));
            }
        }
    }
}

fn cap_stream(mut text: String) -> String {
    if text.len() > MAX_STREAM_BYTES {
        text.truncate(floor_char_boundary(&text, MAX_STREAM_BYTES));
        text.push_str("\n…(output truncated)");
    }
    text
}

/// Largest byte index `<= at` that lands on a char boundary. (Stable-Rust
/// stand-in for `str::floor_char_boundary`.)
fn floor_char_boundary(text: &str, at: usize) -> usize {
    let mut index = at.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests need a real interpreter; every dev/CI box this repo targets has
    /// one. A hard panic (not a silent skip) keeps the suite honest — a
    /// missing python would otherwise turn the whole module into a 0-fire
    /// feature with green tests.
    fn python() -> &'static str {
        super::super::bash_tools::detect_first_command(&["python3", "python"])
            .expect("python3 is required to test the persistent kernel")
    }

    fn scope(name: &str) -> String {
        format!("test-{name}-{}", std::process::id())
    }

    #[test]
    fn state_persists_across_calls() {
        let scope = scope("persist");
        let a = execute_persistent_python(&scope, "x = 41", python(), None, None, false)
            .expect("first exec");
        assert!(a.ok && a.fresh_kernel);
        let b = execute_persistent_python(&scope, "x + 1", python(), None, None, false)
            .expect("second exec");
        assert!(b.ok && !b.fresh_kernel);
        assert_eq!(b.value.as_deref(), Some("42"));
        remove_kernel(&scope);
    }

    #[test]
    fn exception_reports_but_kernel_survives() {
        let scope = scope("survive");
        execute_persistent_python(&scope, "y = 7", python(), None, None, false).expect("seed");
        let boom = execute_persistent_python(&scope, "1/0", python(), None, None, false)
            .expect("exception is a result, not a transport error");
        assert!(!boom.ok);
        assert!(
            boom.error.as_deref().unwrap_or("").contains("ZeroDivisionError"),
            "traceback expected: {:?}",
            boom.error
        );
        let after = execute_persistent_python(&scope, "y", python(), None, None, false)
            .expect("kernel survives the exception");
        assert_eq!(after.value.as_deref(), Some("7"));
        remove_kernel(&scope);
    }

    #[test]
    fn reset_discards_the_namespace() {
        let scope = scope("reset");
        execute_persistent_python(&scope, "z = 1", python(), None, None, false).expect("seed");
        let fresh = execute_persistent_python(&scope, "'z' in dir()", python(), None, None, true)
            .expect("reset exec");
        assert!(fresh.fresh_kernel);
        assert_eq!(fresh.value.as_deref(), Some("False"));
        remove_kernel(&scope);
    }

    #[test]
    fn timeout_discards_kernel_and_next_call_starts_fresh() {
        let scope = scope("timeout");
        execute_persistent_python(&scope, "w = 5", python(), None, None, false).expect("seed");
        let hung = execute_persistent_python(
            &scope,
            "import time\ntime.sleep(60)",
            python(),
            None,
            Some(Duration::from_millis(300)),
            false,
        );
        let message = hung.expect_err("must time out").to_string();
        assert!(message.contains("timeout"), "{message}");
        assert!(message.contains("variables are lost"), "{message}");
        let fresh = execute_persistent_python(&scope, "'w' in dir()", python(), None, None, false)
            .expect("fresh kernel after timeout");
        assert!(fresh.fresh_kernel);
        assert_eq!(fresh.value.as_deref(), Some("False"));
        remove_kernel(&scope);
    }

    #[test]
    fn kernel_death_mid_call_errors_once_then_next_call_is_fresh() {
        let scope = scope("death");
        execute_persistent_python(&scope, "v = 9", python(), None, None, false).expect("seed");
        // `os._exit` kills the kernel before any frame is written. The dead
        // kernel triggers the single respawn-retry, which re-runs the same
        // cell and dies again — so the call errors instead of looping.
        let died = execute_persistent_python(
            &scope,
            "import os\nos._exit(0)",
            python(),
            None,
            None,
            false,
        );
        let message = died.expect_err("kernel death must surface").to_string();
        assert!(message.contains("exited mid-call"), "{message}");
        assert!(message.contains("variables are lost"), "{message}");

        let fresh = execute_persistent_python(&scope, "'v' in dir()", python(), None, None, false)
            .expect("fresh kernel after death");
        assert!(fresh.fresh_kernel);
        assert_eq!(fresh.value.as_deref(), Some("False"));
        remove_kernel(&scope);
    }

    #[test]
    fn scopes_are_isolated() {
        let left = scope("iso-left");
        let right = scope("iso-right");
        execute_persistent_python(&left, "secret = 'left'", python(), None, None, false)
            .expect("left seed");
        let other = execute_persistent_python(&right, "'secret' in dir()", python(), None, None, false)
            .expect("right probe");
        assert_eq!(other.value.as_deref(), Some("False"));
        remove_kernel(&left);
        remove_kernel(&right);
    }

    /// Hermetic eviction test on a LOCAL map: filling the global registry to
    /// its cap from a test would evict other parallel tests' kernels and make
    /// the suite flaky.
    #[test]
    fn eviction_kills_the_longest_idle_kernel_and_skips_busy_ones() {
        let mut map: HashMap<String, Arc<Mutex<KernelHandle>>> = HashMap::new();
        for i in 0..MAX_KERNELS {
            let mut handle = spawn_kernel(python(), None).expect("spawn");
            // Deterministic idleness ranking without sleeping: older index =
            // longer idle.
            handle.last_used = Instant::now()
                .checked_sub(Duration::from_secs((MAX_KERNELS - i) as u64))
                .expect("process uptime exceeds a few seconds");
            map.insert(format!("k{i}"), Arc::new(Mutex::new(handle)));
        }

        evict_longest_idle_if_full(&mut map);
        assert_eq!(map.len(), MAX_KERNELS - 1);
        assert!(!map.contains_key("k0"), "longest-idle kernel must be evicted");

        // A busy kernel (mutex held) is never the victim even when longest idle.
        let busy = Arc::clone(map.get("k1").expect("k1 present"));
        let busy_guard = busy.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // Refill to the cap so eviction triggers again.
        let mut extra = spawn_kernel(python(), None).expect("spawn extra");
        extra.last_used = Instant::now();
        map.insert("fresh".into(), Arc::new(Mutex::new(extra)));
        evict_longest_idle_if_full(&mut map);
        assert!(
            map.contains_key("k1"),
            "a mid-exec kernel must be skipped by eviction"
        );
        drop(busy_guard);

        for (_, handle) in map.drain() {
            if let Ok(mut guard) = handle.try_lock() {
                let _ = guard.child.kill();
                let _ = guard.child.wait();
            }
        }
    }

    #[test]
    fn stdout_is_captured_not_interleaved() {
        let scope = scope("stdout");
        let out = execute_persistent_python(
            &scope,
            "print('hello'); print('world'); 3 * 3",
            python(),
            None,
            None,
            false,
        )
        .expect("exec");
        assert_eq!(out.stdout, "hello\nworld\n");
        assert_eq!(out.value.as_deref(), Some("9"));
        remove_kernel(&scope);
    }

    #[test]
    fn statement_only_cells_return_no_value() {
        let scope = scope("stmt");
        let out = execute_persistent_python(&scope, "for _ in range(3): pass", python(), None, None, false)
            .expect("exec");
        assert!(out.ok);
        assert_eq!(out.value, None);
        remove_kernel(&scope);
    }
}
