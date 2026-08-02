//! `/ide` — bridge to a running IDE extension's MCP server.
//!
//! Zo-compatible VS Code / `JetBrains` extensions run a local WebSocket MCP
//! server and advertise it through lockfiles in the Zo home at `ide/<port>.lock`:
//!
//! ```json
//! {"workspaceFolders":["/path"],"pid":3215,"ideName":"IntelliJ IDEA",
//!  "transport":"ws","authToken":"…","runningInWindows":false}
//! ```
//!
//! Zo uses its existing WebSocket MCP transport to discover the lockfile whose
//! workspace matches the current directory and whose process is alive, connect
//! with the protocol-defined `x-claude-code-ide-authorization` header, and add
//! the IDE's tools (`mcp__ide__getDiagnostics`, `openDiff`, …) to the registry.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use tools::GlobalToolRegistry;

use super::RuntimeMcpState;

/// One parsed IDE lockfile.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct IdeLock {
    #[serde(default)]
    pub(crate) workspace_folders: Vec<String>,
    pub(crate) pid: u32,
    #[serde(default)]
    pub(crate) ide_name: String,
    #[serde(default)]
    pub(crate) transport: String,
    pub(crate) auth_token: String,
    /// Not in the file — filled from the lockfile name (`<port>.lock`).
    #[serde(skip)]
    pub(crate) port: u16,
}

/// Auth header the IDE extension's WebSocket server requires.
const IDE_AUTH_HEADER: &str = "x-claude-code-ide-authorization";

/// Upper bound on the synchronous WS handshake + `initialize` + `tools/list`
/// round-trip. The lockfile only proves the port was *opened*, not that the
/// server still answers; a hung-but-accepting socket would otherwise block the
/// TUI input thread forever. 5s is generous for a localhost handshake yet short
/// enough that the spinner/input stay responsive on a wedged IDE.
const IDE_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Lockfile directories written by Zo-compatible IDE extensions, in canonical
/// priority order. The fallback keeps no-home environments private and usable.
fn ide_lock_dirs() -> Vec<PathBuf> {
    let mut roots = runtime::zo_global_config_roots();
    if roots.is_empty() {
        roots.push(runtime::default_config_home());
    }
    roots.into_iter().map(|root| root.join("ide")).collect()
}

fn parse_ide_lock(path: &Path) -> Option<IdeLock> {
    let port: u16 = path.file_stem()?.to_str()?.parse().ok()?;
    let raw = std::fs::read_to_string(path).ok()?;
    let mut lock: IdeLock = serde_json::from_str(&raw).ok()?;
    lock.port = port;
    Some(lock)
}

/// `kill -0`: does the lockfile's process still exist? Dead IDEs leave stale
/// lockfiles behind; connecting to a recycled port would hit who-knows-what.
fn pid_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .is_ok_and(|status| status.success())
}

fn workspace_matches(lock: &IdeLock, cwd: &Path) -> bool {
    lock.workspace_folders
        .iter()
        .any(|folder| cwd.starts_with(folder))
}

/// All live, WebSocket-transport IDE locks, workspace matches first.
pub(crate) fn discover_ide_locks(dir: &Path, cwd: &Path) -> Vec<IdeLock> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut locks: Vec<IdeLock> = entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "lock"))
        .filter_map(|entry| parse_ide_lock(&entry.path()))
        .filter(|lock| lock.transport == "ws" && pid_alive(lock.pid))
        .collect();
    locks.sort_by_key(|lock| !workspace_matches(lock, cwd));
    locks
}

/// Connect the best-matching IDE and splice its tools into the live registry.
/// `target` filters by IDE name substring when several IDEs are running.
/// Returns the user-facing report text.
fn discover_ide_locks_in_dirs(dirs: &[PathBuf], cwd: &Path) -> Vec<IdeLock> {
    let mut seen_ports = BTreeSet::new();
    let mut locks = Vec::new();
    for dir in dirs {
        for lock in discover_ide_locks(dir, cwd) {
            if seen_ports.insert(lock.port) {
                locks.push(lock);
            }
        }
    }
    locks.sort_by_key(|lock| !workspace_matches(lock, cwd));
    locks
}

pub(crate) fn connect_ide(
    cli: &mut super::LiveCli,
    target: Option<&str>,
) -> Result<String, String> {
    let dirs = ide_lock_dirs();
    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    let mut locks = discover_ide_locks_in_dirs(&dirs, &cwd);
    if let Some(filter) = target {
        let needle = filter.to_lowercase();
        locks.retain(|lock| lock.ide_name.to_lowercase().contains(&needle));
    }
    let Some(lock) = locks.first() else {
        return Err(format!(
            "No running Zo-compatible IDE extension found{} ({}). Configure the \
             extension to publish lockfiles in the Zo home and open this project.",
            target
                .map(|t| format!(" matching `{t}`"))
                .unwrap_or_default(),
            dirs.first()
                .map_or_else(|| "Zo home".to_string(), |dir| dir.display().to_string())
        ));
    };
    let workspace_note = if workspace_matches(lock, &cwd) {
        ""
    } else {
        " (workspace does not match the current directory)"
    };
    let url = format!("ws://127.0.0.1:{}", lock.port);
    let headers = BTreeMap::from([(IDE_AUTH_HEADER.to_string(), lock.auth_token.clone())]);
    let registry = cli.runtime.api_client().tool_registry();
    // `connect_ws_server_now` mutates the shared `Arc<Mutex<RuntimeMcpState>>`
    // and the `Arc`-backed registry, so running it on a background thread still
    // propagates the spliced tools to the live session — the timeout only
    // bounds how long the input thread *waits* for that to finish.
    let tool_count = connect_ws_with_timeout(
        &mut cli.runtime.mcp_state,
        &registry,
        url,
        headers,
        IDE_CONNECT_TIMEOUT,
    )?;
    Ok(format!(
        "IDE\n  Connected        {}{workspace_note}\n  Port             {}\n  Tools            {tool_count} (mcp__ide__*)",
        if lock.ide_name.is_empty() {
            "IDE"
        } else {
            &lock.ide_name
        },
        lock.port
    ))
}

/// Run [`super::mcp_runtime::connect_ws_server_now`] off the input thread and
/// wait at most `timeout` for it. The shared `Arc<Mutex<RuntimeMcpState>>` and
/// the `Arc`-backed registry are cloned into the worker so a successful connect
/// still splices its tools into the live session; on timeout the worker is left
/// detached (the wedged WS handshake drops on its own, mirroring the
/// best-effort background MCP discovery), and the caller gets a clear error
/// instead of a frozen TUI.
fn connect_ws_with_timeout(
    mcp_state_slot: &mut Option<Arc<Mutex<RuntimeMcpState>>>,
    registry: &GlobalToolRegistry,
    url: String,
    headers: BTreeMap<String, String>,
    timeout: Duration,
) -> Result<usize, String> {
    // Populate the slot here (rather than inside the worker) so the worker's
    // clone and the caller's slot point at the *same* `Mutex` — the splice the
    // worker performs is then visible to the live session even though the call
    // ran on another thread.
    let state = mcp_state_slot
        .get_or_insert_with(|| Arc::new(Mutex::new(RuntimeMcpState::empty())))
        .clone();
    let registry = registry.clone();
    run_with_timeout(timeout, move || {
        super::mcp_runtime::connect_ws_server_now(
            &mut Some(state),
            &registry,
            "ide",
            url,
            headers,
        )
    })
}

/// Generic timeout wrapper: run `op` on a worker thread and block the caller for
/// at most `timeout`. Returns `op`'s result, or a timeout error if it does not
/// finish in time. A timed-out worker is detached, not joined — the caller must
/// not be held hostage to a hung operation. Kept generic so the timeout
/// behaviour is unit-testable without a live MCP/IDE server.
fn run_with_timeout<F>(timeout: Duration, op: F) -> Result<usize, String>
where
    F: FnOnce() -> Result<usize, String> + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    // Detached on purpose: if `op` hangs, dropping the receiver lets the send
    // fail silently when the worker eventually unblocks. The thread name aids
    // diagnosis if it lingers.
    let spawned = std::thread::Builder::new()
        .name("ide-connect".to_string())
        .spawn(move || {
            let _ = tx.send(op());
        });
    if spawned.is_err() {
        // `op` was moved into the (failed) spawn, so it cannot be re-run inline;
        // report the spawn failure rather than risk an unbounded inline connect.
        return Err("could not spawn IDE connect worker thread".to_string());
    }
    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "IDE did not respond within {}s — the editor's MCP server is \
             unreachable or hung. Reopen the project in the IDE and retry.",
            timeout.as_secs()
        )),
        // The worker dropped the sender without sending (e.g. panicked); surface
        // it as a generic failure rather than blocking forever.
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("IDE connect worker exited unexpectedly".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique per-test scratch dir without a tempfile dependency.
    fn temp_lock_dir(tag: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "zo-ide-bridge-{}-{tag}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_lock(dir: &Path, port: u16, json: &str) {
        std::fs::write(dir.join(format!("{port}.lock")), json).expect("write lockfile");
    }

    /// CC 확장이 쓰는 실제 lockfile 스키마가 파싱되고, 포트는 파일명에서 온다.
    #[test]
    fn parses_cc_extension_lockfile_shape() {
        let dir = temp_lock_dir("parse");
        write_lock(
            &dir,
            52224,
            r#"{"workspaceFolders":["/Users/joe/2026/zo"],"pid":3215,
                "ideName":"IntelliJ IDEA","transport":"ws","runningInWindows":false,
                "authToken":"secret-token"}"#,
        );
        let lock = parse_ide_lock(&dir.join("52224.lock")).expect("parsed");
        assert_eq!(lock.port, 52224);
        assert_eq!(lock.ide_name, "IntelliJ IDEA");
        assert_eq!(lock.transport, "ws");
        assert_eq!(lock.auth_token, "secret-token");
        assert!(workspace_matches(
            &lock,
            Path::new("/Users/joe/2026/zo/crates")
        ));
        assert!(!workspace_matches(&lock, Path::new("/tmp/elsewhere")));
    }

    /// 죽은 pid 의 stale lockfile 은 후보에서 제외된다.
    #[test]
    fn discovery_skips_stale_and_non_ws_locks() {
        let dir = temp_lock_dir("stale");
        // A live pid (our own) with the wrong transport must still be
        // excluded.
        write_lock(
            &dir,
            1111,
            &format!(
                r#"{{"workspaceFolders":[],"pid":{},"ideName":"SSE IDE",
                "transport":"sse","authToken":"t"}}"#,
                std::process::id()
            ),
        );
        // A near-max PID is effectively guaranteed dead.
        write_lock(
            &dir,
            2222,
            r#"{"workspaceFolders":[],"pid":99999999,"ideName":"Dead IDE",
                "transport":"ws","authToken":"t"}"#,
        );
        // Garbage that must not panic the scan.
        write_lock(&dir, 3333, "not-json");
        let locks = discover_ide_locks(&dir, Path::new("/anywhere"));
        assert!(
            locks.is_empty(),
            "stale/non-ws/garbage locks must be filtered: {locks:?}"
        );
    }

    /// 워크스페이스가 cwd 와 일치하는 lock 이 우선 정렬된다.
    #[test]
    fn workspace_matching_lock_sorts_first() {
        let dir = temp_lock_dir("sort");
        let pid = std::process::id();
        write_lock(
            &dir,
            1000,
            &format!(
                r#"{{"workspaceFolders":["/other/project"],"pid":{pid},
                "ideName":"Other","transport":"ws","authToken":"t"}}"#
            ),
        );
        write_lock(
            &dir,
            2000,
            &format!(
                r#"{{"workspaceFolders":["/this/project"],"pid":{pid},
                "ideName":"Match","transport":"ws","authToken":"t"}}"#
            ),
        );
        let locks = discover_ide_locks(&dir, Path::new("/this/project/sub"));
        assert_eq!(locks.len(), 2);
        assert_eq!(locks[0].ide_name, "Match");
    }

    #[test]
    fn discovery_merges_canonical_roots_with_primary_precedence() {
        let primary = temp_lock_dir("primary-root");
        let secondary = temp_lock_dir("secondary-root");
        let pid = std::process::id();
        write_lock(
            &primary,
            4100,
            &format!(
                r#"{{"workspaceFolders":["/other"],"pid":{pid},
                "ideName":"Primary","transport":"ws","authToken":"primary"}}"#
            ),
        );
        write_lock(
            &secondary,
            4100,
            &format!(
                r#"{{"workspaceFolders":["/project"],"pid":{pid},
                "ideName":"Shadowed","transport":"ws","authToken":"secondary"}}"#
            ),
        );
        write_lock(
            &secondary,
            4200,
            &format!(
                r#"{{"workspaceFolders":["/project"],"pid":{pid},
                "ideName":"Legacy IDE","transport":"ws","authToken":"legacy"}}"#
            ),
        );

        let locks = discover_ide_locks_in_dirs(
            &[primary, secondary],
            Path::new("/project/subdirectory"),
        );
        assert_eq!(locks.len(), 2);
        assert_eq!(locks[0].ide_name, "Legacy IDE");
        assert!(locks.iter().any(|lock| lock.ide_name == "Primary"));
        assert!(!locks.iter().any(|lock| lock.ide_name == "Shadowed"));
    }

    /// 응답 없는(행 걸린) IDE 핸드셰이크는 입력 스레드를 묶어두지 않고,
    /// timeout 안에 에러로 반환된다. 기존 코드는 무기한 블록되어 TUI 가 멈췄다.
    #[test]
    fn hung_connect_returns_within_timeout() {
        let start = std::time::Instant::now();
        let result = run_with_timeout(Duration::from_millis(50), || {
            // Simulate a hung-but-accepting IDE: the handshake never returns
            // anywhere near the timeout window.
            std::thread::sleep(Duration::from_secs(30));
            Ok(7)
        });
        let waited = start.elapsed();
        let error = result.expect_err("a hung connect must surface a timeout error");
        assert!(
            error.contains("did not respond"),
            "timeout error should explain the hang: {error}"
        );
        // The wait must be bounded by the timeout, not the 30s sleep — proving
        // the caller (TUI input thread) is freed promptly.
        assert!(
            waited < Duration::from_secs(5),
            "timeout did not bound the wait: {waited:?}"
        );
    }

    /// 정상적으로 빠르게 끝나는 연결은 결과를 그대로 통과시킨다(타임아웃 래퍼가 투명).
    #[test]
    fn fast_connect_passes_result_through() {
        let result = run_with_timeout(Duration::from_secs(5), || Ok(3));
        assert_eq!(result, Ok(3));
    }
}
