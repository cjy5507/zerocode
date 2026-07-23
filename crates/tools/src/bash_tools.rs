use std::path::Path;
use std::process::{Child, Command, Output};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use super::{
    from_value, maybe_enforce_permission_check, to_pretty_json, workspace_guard_enabled,
    workspace_scope_guard, workspace_test_branch_preflight, ToolContext, ToolError, ToolSpec,
};
use runtime::{
    permission_enforcer::PermissionEnforcer, resolve_sandbox_status, BashCommandInput,
    BashCommandOutput, ConfigLoader, PermissionMode, SandboxConfig, SandboxStatus,
};

#[derive(Debug, Deserialize)]
pub(crate) struct ReplInput {
    pub code: String,
    pub language: String,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PowerShellInput {
    pub command: String,
    pub timeout: Option<u64>,
    pub description: Option<String>,
    pub run_in_background: Option<bool>,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct ReplOutput {
    pub language: String,
    pub stdout: String,
    pub stderr: String,
    #[serde(rename = "exitCode")]
    pub exit_code: i32,
    #[serde(rename = "durationMs")]
    pub duration_ms: u128,
}

struct ReplRuntime {
    program: &'static str,
    args: &'static [&'static str],
}

pub(crate) fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "bash",
            description: "Execute a shell command in the current workspace. Prefer the dedicated tools when one fits — read_file/grep_search/find_files instead of cat/grep/find, edit_file instead of sed -i — they are cheaper, permission-scoped, and keep the harness's file-state tracking accurate. Independent commands belong in one response so they run in parallel; a long-running command (server, watcher, slow build) should set run_in_background instead of blocking the turn. A denied command means the user declined it — change approach; do not re-issue it verbatim.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout": { "type": "integer", "minimum": 1 },
                    "description": { "type": "string" },
                    "run_in_background": { "type": "boolean" },
                    "dangerouslyDisableSandbox": { "type": "boolean" },
                    "namespaceRestrictions": { "type": "boolean" },
                    "isolateNetwork": { "type": "boolean" },
                    "filesystemMode": { "type": "string", "enum": ["off", "workspace-only", "allow-list"] },
                    "allowedMounts": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
        ToolSpec {
            name: "REPL",
            description: "Execute code in a REPL-like subprocess.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "code": { "type": "string" },
                    "language": { "type": "string" },
                    "timeout_ms": { "type": "integer", "minimum": 1 }
                },
                "required": ["code", "language"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
        ToolSpec {
            name: "PowerShell",
            description: "Execute a PowerShell command with optional timeout.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout": { "type": "integer", "minimum": 1 },
                    "description": { "type": "string" },
                    "run_in_background": { "type": "boolean" }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
            required_permission: PermissionMode::DangerFullAccess,
        },
    ]
}

pub(crate) fn dispatch(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    // A per-agent context can pin the working directory (worktree isolation);
    // `None` leaves every exec tool on the process cwd, the historical default.
    let cwd = ctx.cwd.as_deref();
    // Worktree isolation: a confined agent's shell must not redirect git at the
    // shared main checkout. Refuse `git -C` / `--git-dir` / `GIT_DIR=` targets
    // that escape the worktree before the command runs. File-tool writes are
    // already confined by the workspace boundary; this closes the shell path.
    if let Some(root) = ctx.worktree_confinement.as_deref() {
        if matches!(name, "bash" | "PowerShell") {
            if let Some(command) = input.get("command").and_then(Value::as_str) {
                if let Some(reason) =
                    runtime::bash_validation::git_worktree_escape_reason(command, root)
                {
                    return Some(Err(ToolError::PermissionDenied {
                        tool: name.to_string(),
                        reason,
                    }));
                }
            }
        }
    }
    match name {
        "bash" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                let tool_call_id = input.get("__zo_tool_call_id").and_then(Value::as_str);
                runtime::live_output::with_dispatch_key(tool_call_id, || {
                    let session_id = ctx.session_id();
                    from_value::<BashCommandInput>(input).and_then(|inp| {
                        mark_shell_checkpoint_if_write_intent(ctx, &inp.command);
                        run_bash(inp, cwd, Some(&ctx.tasks), session_id.as_deref())
                    })
                })
            }),
        ),
        "REPL" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                // REPL has no background mode, so it creates no task to session-stamp.
                from_value::<ReplInput>(input).and_then(|inp| run_repl(inp, cwd))
            }),
        ),
        "PowerShell" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<PowerShellInput>(input).and_then(|inp| {
                    mark_shell_checkpoint_if_write_intent(ctx, &inp.command);
                    run_powershell(&inp, cwd)
                })
            }),
        ),
        _ => None,
    }
}

fn mark_shell_checkpoint_if_write_intent(ctx: &ToolContext, command: &str) {
    if runtime::bash_validation::required_mode_for_command(command) != PermissionMode::ReadOnly {
        ctx.mark_workspace_checkpoint_incomplete();
    }
}

pub(crate) fn run_bash(
    mut input: BashCommandInput,
    cwd: Option<&Path>,
    tasks: Option<&runtime::task_registry::TaskRegistry>,
    session_id: Option<&str>,
) -> Result<String, ToolError> {
    // Pin the spawn cwd from the tool context unless the input already carries
    // one (it never does from the wire — `cwd` is server-injected, not in the
    // public schema). Default `None` keeps the live process cwd. Remember whether
    // the command was already pinned before injection: only an explicit isolated
    // cwd should bypass the shared-worktree guard.
    let had_explicit_cwd = input.cwd.is_some();
    if input.cwd.is_none() {
        input.cwd = cwd.map(Path::to_path_buf);
    }
    if let Some(output) = workspace_test_branch_preflight(&input.command) {
        return Ok(serde_json::to_string_pretty(&output)?);
    }
    // Shared-working-tree scope guard (track 4-1): when opt-in via
    // `ZO_WORKSPACE_GUARD`, block broad mutating commands (`cargo fmt`,
    // `git add -A`, `git reset --hard`, …) on the shared process tree so one
    // agent cannot tangle another's uncommitted work. The guard must run before
    // a server-injected `ctx.cwd` makes the command look artificially isolated.
    if !had_explicit_cwd && workspace_guard_enabled() {
        if let Some(output) = workspace_scope_guard(&input.command) {
            return Ok(serde_json::to_string_pretty(&output)?);
        }
    }
    // Surface a non-blocking advisory for known destructive patterns
    // (`rm -rf /`, `dd if=…`, fork bombs, …). The command still runs —
    // this only makes the risk visible alongside the result.
    let safety_warning = runtime::bash_validation::check_destructive(&input.command)
        .warning_message()
        .map(ToOwned::to_owned);
    // Background runs register in the session task registry (when the tool
    // context provides one) so `TaskOutput`/`TaskStop` can observe and kill
    // them — the returned `backgroundTaskId` is a real `task_…` id.
    let mut output = runtime::execute_bash_with_tasks(input, tasks, session_id)
        .map_err(|error| ToolError::Execution(error.to_string()))?;
    if output.safety_warning.is_none() {
        output.safety_warning = safety_warning;
    }
    Ok(serde_json::to_string_pretty(&output)?)
}

pub(crate) fn run_repl(input: ReplInput, cwd: Option<&Path>) -> Result<String, ToolError> {
    to_pretty_json(execute_repl(input, cwd)?)
}

pub(crate) fn run_powershell(
    input: &PowerShellInput,
    cwd: Option<&Path>,
) -> Result<String, ToolError> {
    to_pretty_json(execute_powershell(input, cwd)?)
}

fn execute_repl(input: ReplInput, cwd: Option<&Path>) -> Result<ReplOutput, ToolError> {
    if input.code.trim().is_empty() {
        return Err(ToolError::InvalidInput("code must not be empty".into()));
    }
    // Catastrophic backstop for EVERY language. A REPL interpreter (python,
    // node, …) can shell out (os.system / subprocess / child_process.exec), so a
    // host-destroying command is reachable regardless of language — gating this
    // on a shell language left python/node as an unguarded, strictly-MORE-capable
    // path that ran even in DangerFullAccess (zo's normal mode). This mirrors
    // the hard-block `execute_bash_with_tasks` enforces for `bash` in every mode.
    if let Some(reason) = repl_catastrophic_reason(&input.code) {
        return Err(ToolError::Execution(format!(
            "refused to run a catastrophic command — {reason}"
        )));
    }
    let runtime = resolve_repl_runtime(&input.language)?;
    let started = Instant::now();
    let mut process = Command::new(runtime.program);
    process
        .args(runtime.args)
        .arg(&input.code)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }

    let output = if let Some(timeout_ms) = input.timeout_ms {
        let child = process.spawn()?;
        wait_child_with_timeout(child, Duration::from_millis(timeout_ms)).map_err(|e| {
            ToolError::Execution(if e.kind() == std::io::ErrorKind::TimedOut {
                format!("REPL execution exceeded timeout of {timeout_ms} ms")
            } else {
                e.to_string()
            })
        })?
    } else {
        process.spawn()?.wait_with_output()?
    };

    Ok(ReplOutput {
        language: input.language,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        exit_code: output.status.code().unwrap_or(1),
        duration_ms: started.elapsed().as_millis(),
    })
}

/// Catastrophic-command reason for a REPL `code` body, or `None`. Best-effort
/// defense-in-depth behind REPL's `DangerFullAccess` requirement — not a
/// sandbox. First pass scans the whole body, so a shell-language REPL
/// (`bash`/`sh`) gets the same hard-block as the `bash` tool. Second pass scans
/// each embedded string literal, so an interpreter REPL that shells out
/// (`os.system('rm -rf /')`, `execSync("…")`, backticks) is refused even though
/// the surrounding code is not shell — `check_destructive` tokenizes on the
/// first command, so it only sees `rm` once the literal is unwrapped.
/// Obfuscation (string concatenation, base64) is out of scope, as it is for the
/// `bash` path; the permission gate remains the primary control.
fn repl_catastrophic_reason(code: &str) -> Option<String> {
    if let runtime::bash_validation::ValidationResult::Block { reason } =
        runtime::bash_validation::check_destructive(code)
    {
        return Some(reason);
    }
    for literal in string_literals(code) {
        if let runtime::bash_validation::ValidationResult::Block { reason } =
            runtime::bash_validation::check_destructive(&literal)
        {
            return Some(reason);
        }
    }
    None
}

/// Contents of every single-, double-, or back-quoted string literal in `code`.
/// Best-effort scanner (honors `\`-escapes, no nesting/concatenation) used to
/// vet interpreter-REPL payloads that smuggle a shell command through a literal.
fn string_literals(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = code.chars();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' || c == '`' {
            let quote = c;
            let mut buf = String::new();
            let mut escaped = false;
            for ch in chars.by_ref() {
                if escaped {
                    buf.push(ch);
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == quote {
                    break;
                } else {
                    buf.push(ch);
                }
            }
            if !buf.is_empty() {
                out.push(buf);
            }
        }
    }
    out
}

fn resolve_repl_runtime(language: &str) -> Result<ReplRuntime, ToolError> {
    match language.trim().to_ascii_lowercase().as_str() {
        "python" | "py" => Ok(ReplRuntime {
            program: detect_first_command(&["python3", "python"])
                .ok_or_else(|| ToolError::Execution("python runtime not found".into()))?,
            args: &["-c"],
        }),
        "javascript" | "js" | "node" => Ok(ReplRuntime {
            program: detect_first_command(&["node"])
                .ok_or_else(|| ToolError::Execution("node runtime not found".into()))?,
            args: &["-e"],
        }),
        "sh" | "shell" | "bash" => Ok(ReplRuntime {
            program: detect_first_command(&["bash", "sh"])
                .ok_or_else(|| ToolError::Execution("shell runtime not found".into()))?,
            args: &["-lc"],
        }),
        other => Err(ToolError::InvalidInput(format!(
            "unsupported REPL language: {other}"
        ))),
    }
}

fn detect_first_command(commands: &[&'static str]) -> Option<&'static str> {
    commands
        .iter()
        .copied()
        .find(|command| command_exists(command))
}

pub(crate) fn command_exists(command: &str) -> bool {
    std::process::Command::new("sh")
        .arg("-lc")
        .arg("command -v \"$1\" >/dev/null 2>&1")
        .arg("--")
        .arg(command)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn execute_powershell(
    input: &PowerShellInput,
    cwd: Option<&Path>,
) -> std::io::Result<BashCommandOutput> {
    if let runtime::bash_validation::ValidationResult::Block { reason } =
        runtime::bash_validation::check_destructive(&input.command)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("refused to run a catastrophic command — {reason}"),
        ));
    }
    let _ = &input.description;
    if let Some(output) = workspace_test_branch_preflight(&input.command) {
        return Ok(output);
    }
    let shell = detect_powershell_shell()?;
    execute_shell_command(
        shell,
        &input.command,
        input.timeout,
        input.run_in_background,
        cwd,
    )
}

fn detect_powershell_shell() -> std::io::Result<&'static str> {
    if command_exists("pwsh") {
        Ok("pwsh")
    } else if command_exists("powershell") {
        Ok("powershell")
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "PowerShell executable not found (expected `pwsh` or `powershell` in PATH)",
        ))
    }
}

fn execute_shell_command(
    shell: &str,
    command: &str,
    timeout: Option<u64>,
    run_in_background: Option<bool>,
    exec_cwd: Option<&Path>,
) -> std::io::Result<BashCommandOutput> {
    // A per-agent context can pin the working directory; otherwise use the
    // live process cwd (unchanged default). Both the spawn and the sandbox
    // config below resolve against this directory.
    let cwd = match exec_cwd {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir()?,
    };
    let loaded_config = ConfigLoader::default_for(&cwd).load();
    let sandbox_config = loaded_config
        .as_ref()
        .map_or_else(|_| SandboxConfig::default(), |cfg| cfg.sandbox().clone());
    // settings.env parity with the bash tool: a declared `env` block reaches the
    // PowerShell child too (this path previously dropped it silently).
    let settings_env = loaded_config
        .as_ref()
        .map(|cfg| cfg.env().clone())
        .unwrap_or_default();
    let sandbox_status = resolve_sandbox_status(&sandbox_config, &cwd);
    fail_closed_if_sandbox_unavailable(&sandbox_status)?;

    if run_in_background.unwrap_or(false) {
        let child = std::process::Command::new(shell)
            .arg("-NoProfile")
            .arg("-NonInteractive")
            .arg("-Command")
            .arg(command)
            .current_dir(&cwd)
            .envs(&settings_env)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        return Ok(backgrounded_command_output(
            child.id().to_string(),
            sandbox_status,
        ));
    }

    let mut process = std::process::Command::new(shell);
    process
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-Command")
        .arg(command)
        .current_dir(&cwd)
        .envs(&settings_env);
    process
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    if let Some(timeout_ms) = timeout {
        let child = process.spawn()?;
        let timeout_dur = Duration::from_millis(timeout_ms);
        return match wait_child_with_timeout(child, timeout_dur) {
            Ok(output) => Ok(completed_command_output(&output, sandbox_status)),
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                Ok(timed_out_command_output(timeout_ms, sandbox_status))
            }
            Err(e) => Err(e),
        };
    }

    let output = process.output()?;
    Ok(completed_command_output(&output, sandbox_status))
}

fn fail_closed_if_sandbox_unavailable(sandbox_status: &SandboxStatus) -> std::io::Result<()> {
    if sandbox_status.enabled {
        if let Some(reason) = &sandbox_status.fallback_reason {
            return Err(std::io::Error::other(format!(
                "sandbox requested but unavailable: {reason}. Set sandbox.enabled=false to run PowerShell without sandbox."
            )));
        }
    }
    Ok(())
}

/// Output shape for a command spawned detached via `run_in_background`: no
/// captured streams, only the spawned OS task id.
fn backgrounded_command_output(
    task_id: String,
    sandbox_status: SandboxStatus,
) -> BashCommandOutput {
    BashCommandOutput {
        stdout: String::new(),
        stderr: String::new(),
        raw_output_path: None,
        interrupted: false,
        is_image: None,
        background_task_id: Some(task_id),
        backgrounded_by_user: Some(true),
        assistant_auto_backgrounded: Some(false),
        dangerously_disable_sandbox: None,
        return_code_interpretation: None,
        no_output_expected: Some(true),
        structured_content: None,
        persisted_output_path: None,
        persisted_output_size: None,
        sandbox_status: Some(sandbox_status),
        safety_warning: None,
    }
}

/// Output shape for a command that ran to completion (with or without a timeout
/// guard), carrying captured stdout/stderr and the non-zero exit-code reading.
fn completed_command_output(output: &Output, sandbox_status: SandboxStatus) -> BashCommandOutput {
    BashCommandOutput {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        raw_output_path: None,
        interrupted: false,
        is_image: None,
        background_task_id: None,
        backgrounded_by_user: None,
        assistant_auto_backgrounded: None,
        dangerously_disable_sandbox: None,
        return_code_interpretation: output
            .status
            .code()
            .filter(|code| *code != 0)
            .map(|code| format!("exit_code:{code}")),
        no_output_expected: Some(output.stdout.is_empty() && output.stderr.is_empty()),
        structured_content: None,
        persisted_output_path: None,
        persisted_output_size: None,
        sandbox_status: Some(sandbox_status),
        safety_warning: None,
    }
}

/// Output shape for a command killed because it exceeded its timeout budget.
fn timed_out_command_output(timeout_ms: u64, sandbox_status: SandboxStatus) -> BashCommandOutput {
    BashCommandOutput {
        stdout: String::new(),
        stderr: format!("Command exceeded timeout of {timeout_ms} ms"),
        raw_output_path: None,
        interrupted: true,
        is_image: None,
        background_task_id: None,
        backgrounded_by_user: None,
        assistant_auto_backgrounded: None,
        dangerously_disable_sandbox: None,
        return_code_interpretation: Some(String::from("timeout")),
        no_output_expected: Some(false),
        structured_content: None,
        persisted_output_path: None,
        persisted_output_size: None,
        sandbox_status: Some(sandbox_status),
        safety_warning: None,
    }
}

/// Wait for a child process to complete, killing it if the timeout expires.
///
/// The `Child` stays on this thread so its kill handle survives a timeout —
/// the earlier design moved the whole child into a `wait_with_output` thread,
/// which left no way to terminate it and leaked a runaway process (and its
/// descendants) past the deadline. Here the stdout/stderr pipes are drained by
/// dedicated reader threads (so a child that fills a pipe buffer can't
/// deadlock), while this thread polls `try_wait`; on timeout it kills and reaps
/// the child before returning `TimedOut`.
pub(crate) fn wait_child_with_timeout(
    child: Child,
    timeout: Duration,
) -> std::io::Result<Output> {
    wait_child_with_timeout_for_key(child, timeout, None)
}

fn wait_child_with_timeout_for_key(
    mut child: Child,
    timeout: Duration,
    live_key: Option<&str>,
) -> std::io::Result<Output> {
    use std::io::Read as _;
    use std::sync::{Arc, Mutex};

    // Detach the pipes so the child can never block on a full buffer while we
    // poll for exit. The full shared buffers preserve the final Output exactly;
    // the registry writer separately keeps a bounded tail for live readers.
    // The handle is the RAII cleanup guard for normal exit, timeout, and unwind.
    let live_output = runtime::live_output::register(live_key);
    let stdout_capture = Arc::new(Mutex::new(Vec::new()));
    let stderr_capture = Arc::new(Mutex::new(Vec::new()));
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_buffer = Arc::clone(&stdout_capture);
    let stdout_live = live_output.writer();
    let stdout_reader = std::thread::spawn(move || {
        if let Some(mut pipe) = stdout {
            let mut chunk = [0u8; 8192];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        stdout_buffer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .extend_from_slice(&chunk[..n]);
                        stdout_live.append_stdout(&chunk[..n]);
                    }
                }
            }
        }
    });
    let stderr_buffer = Arc::clone(&stderr_capture);
    let stderr_live = live_output.writer();
    let stderr_reader = std::thread::spawn(move || {
        if let Some(mut pipe) = stderr {
            let mut chunk = [0u8; 8192];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        stderr_buffer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .extend_from_slice(&chunk[..n]);
                        stderr_live.append_stderr(&chunk[..n]);
                    }
                }
            }
        }
    });

    // Poll for exit until the deadline. A short sleep keeps this responsive
    // without meaningfully burning CPU for a process-scale timeout.
    let deadline = Instant::now() + timeout;
    let poll_interval = Duration::from_millis(5);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            // Past the budget: terminate and reap the child. `kill` then
            // `wait` guarantees the OS slot is released before we return.
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "child process exceeded timeout",
            ));
        }
        std::thread::sleep(poll_interval);
    };

    // Process exited within budget — collect whatever the readers captured.
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    let stdout = std::mem::take(
        &mut *stdout_capture
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    let stderr = std::mem::take(
        &mut *stderr_capture
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_write_intent_marks_a_file_checkpoint_incomplete() {
        let path = std::env::temp_dir().join(format!(
            "zo-shell-checkpoint-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_file(&path);
        let context = ToolContext::new();
        context.begin_workspace_checkpoint(1);
        context
            .record_workspace_checkpoint_before(&path)
            .expect("capture before");
        std::fs::write(&path, b"created").expect("write fixture");
        context.record_workspace_checkpoint_write(&path);

        mark_shell_checkpoint_if_write_intent(&context, "printf changed > file.txt");

        let checkpoint = context
            .finish_workspace_checkpoint()
            .expect("finish checkpoint")
            .expect("checkpoint");
        assert!(checkpoint.incomplete);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn shell_read_only_command_does_not_mark_checkpoint_incomplete() {
        let context = ToolContext::new();
        context.begin_workspace_checkpoint(1);

        mark_shell_checkpoint_if_write_intent(&context, "git status --short");

        assert!(context
            .finish_workspace_checkpoint()
            .expect("finish checkpoint")
            .is_none());
    }

    #[test]
    fn confined_agent_rejects_git_escape_out_of_worktree() {
        // A worktree-isolated agent's shell must not redirect git at another
        // checkout. The guard returns before the command ever runs.
        let ctx = ToolContext::new().with_worktree_confinement(std::path::PathBuf::from("/work/wt"));
        for command in [
            "git -C /repo status",
            "git --git-dir=/repo/.git log",
            "GIT_DIR=/repo/.git git status",
        ] {
            let input = serde_json::json!({ "command": command });
            match dispatch(&ctx, None, "bash", &input) {
                Some(Err(ToolError::PermissionDenied { reason, .. })) => {
                    assert!(
                        reason.contains("isolated worktree"),
                        "unexpected reason for {command:?}: {reason}"
                    );
                }
                other => panic!("expected PermissionDenied for {command:?}, got {other:?}"),
            }
        }
    }

    /// Spawn `sh -c <script>` with both pipes captured, as the timeout callers do.
    fn spawn(script: &str) -> Child {
        Command::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn sh")
    }

    #[test]
    fn returns_output_when_command_finishes_in_budget() {
        // Spawning inherits the process-global cwd; serialize against the
        // cwd-mutating tests (which hold `env_lock`) so a concurrent
        // `set_current_dir` cannot fail the spawn and poison the lock.
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let child = spawn("printf 'hello'; printf 'oops' 1>&2; exit 3");
        let output = wait_child_with_timeout(child, Duration::from_secs(5)).expect("within budget");
        assert_eq!(output.stdout, b"hello");
        assert_eq!(output.stderr, b"oops");
        assert_eq!(output.status.code(), Some(3));
    }

    #[test]
    fn live_registry_sees_output_before_lossless_final_result() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let child = spawn("printf 'first\\n'; sleep 0.3; printf 'second\\n'");
        let runner = std::thread::spawn(move || {
            wait_child_with_timeout_for_key(
                child,
                Duration::from_secs(5),
                Some("test-live-capture"),
            )
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        let snapshot = loop {
            if let Some(snapshot) = runtime::live_output::snapshot("test-live-capture", 8192) {
                if snapshot.stdout_tail.contains("first\n") {
                    break snapshot;
                }
            }
            assert!(Instant::now() < deadline, "first chunk was never published live");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(!snapshot.stdout_tail.contains("second"));

        let output = runner.join().expect("wait thread").expect("within budget");
        assert_eq!(output.stdout, b"first\nsecond\n");
        assert!(runtime::live_output::snapshot("test-live-capture", 8192).is_none());
    }

    #[test]
    fn foreground_bash_uses_tool_call_id_for_live_output() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let runner = std::thread::spawn(|| {
            let context = ToolContext::new();
            dispatch(
                &context,
                None,
                "bash",
                &serde_json::json!({
                    "command": "printf 'bash-first\\n'; sleep 0.3; printf 'bash-second\\n'",
                    "__zo_tool_call_id": "test-foreground-bash"
                }),
            )
            .expect("bash dispatch")
            .expect("bash result")
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if runtime::live_output::snapshot("test-foreground-bash", 8192)
                .is_some_and(|snapshot| snapshot.stdout_tail.contains("bash-first\n"))
            {
                break;
            }
            assert!(Instant::now() < deadline, "foreground Bash output was not published");
            std::thread::sleep(Duration::from_millis(10));
        }

        let raw = runner.join().expect("bash thread");
        let output: BashCommandOutput = serde_json::from_str(&raw).expect("Bash output JSON");
        assert_eq!(output.stdout, "bash-first\nbash-second\n");
        assert!(runtime::live_output::snapshot("test-foreground-bash", 8192).is_none());
    }

    #[test]
    fn timeout_drops_live_registry_entry() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let child = spawn("sleep 60");
        let runner = std::thread::spawn(move || {
            wait_child_with_timeout_for_key(
                child,
                Duration::from_millis(300),
                Some("test-live-timeout"),
            )
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while runtime::live_output::snapshot("test-live-timeout", 8192).is_none() {
            assert!(Instant::now() < deadline, "live entry was never registered");
            std::thread::sleep(Duration::from_millis(10));
        }
        let error = runner
            .join()
            .expect("wait thread")
            .expect_err("must time out");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(runtime::live_output::snapshot("test-live-timeout", 8192).is_none());
    }

    #[test]
    fn live_output_sanitizer_resolves_cr_rewrites_and_terminal_sequences() {
        let raw = "\u{1b}]0;watch\u{7}old 1%\r\u{1b}[32m진행 · 2%\u{1b}[0m\nnext";
        assert_eq!(
            runtime::live_output::sanitize_live_output(raw),
            "진행 · 2%\nnext"
        );
    }

    #[test]
    fn timeout_kills_the_child_instead_of_leaking_it() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A process that would run for 60s, far beyond the 100ms budget.
        let child = spawn("sleep 60");
        let pid = child.id();
        // The helper takes the child by value and is responsible for killing
        // and reaping it on timeout.
        let err =
            wait_child_with_timeout(child, Duration::from_millis(100)).expect_err("must time out");
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);

        // The real proof: the pid is gone. `kill -0` returns non-zero once the
        // process no longer exists, confirming we terminated it rather than
        // detaching a runaway. Allow a brief moment for the OS to reap.
        std::thread::sleep(Duration::from_millis(50));
        let alive = Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(!alive, "child pid {pid} should be dead after timeout");
    }

    #[test]
    fn large_output_does_not_deadlock_under_timeout_guard() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // ~1MB to stdout — well past a pipe buffer. If the pipes were not
        // drained concurrently the child would block on write and we would hit
        // the timeout; instead it must complete and return the full payload.
        let child = spawn("yes a | head -c 1000000");
        let output = wait_child_with_timeout(child, Duration::from_secs(10))
            .expect("drains without deadlock");
        assert_eq!(output.stdout.len(), 1_000_000);
        assert!(output.status.success());
    }

    fn bash_input(command: &str) -> BashCommandInput {
        BashCommandInput {
            command: command.to_string(),
            timeout: None,
            description: None,
            run_in_background: None,
            dangerously_disable_sandbox: None,
            namespace_restrictions: None,
            isolate_network: None,
            filesystem_mode: None,
            allowed_mounts: None,
            cwd: None,
        }
    }

    #[test]
    fn workspace_guard_applies_after_server_injected_cwd() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("ZO_WORKSPACE_GUARD", "1");
        let cwd = std::env::current_dir().expect("cwd");
        let output = run_bash(bash_input("cargo fmt -p tools"), Some(&cwd), None, None)
            .expect("guard returns synthetic output");
        std::env::remove_var("ZO_WORKSPACE_GUARD");

        assert!(output.contains("preflight_blocked:workspace_scope"));
        assert!(output.contains("cargo fmt (broad reformat)"));
    }

    #[test]
    fn workspace_guard_allows_file_scoped_fmt_after_server_injected_cwd() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("ZO_WORKSPACE_GUARD", "1");
        let cwd = std::env::current_dir().expect("cwd");
        let output = run_bash(
            bash_input("cargo fmt --check -- crates/tools/src/workspace_scope_guard.rs"),
            Some(&cwd),
            None,
            None,
        )
        .expect("file-scoped check should execute");
        std::env::remove_var("ZO_WORKSPACE_GUARD");

        assert!(!output.contains("preflight_blocked:workspace_scope"));
    }

    #[test]
    fn powershell_refuses_catastrophic_commands_before_shell_detection() {
        // Use a definitely-missing cwd so the unfixed path cannot execute the
        // dangerous payload even on machines with PowerShell installed. The fix
        // must refuse before shell detection or process spawn reaches that cwd.
        let missing_cwd = std::env::temp_dir().join(format!(
            "zo-powershell-missing-cwd-{}",
            std::process::id()
        ));

        let catastrophic = execute_powershell(
            &PowerShellInput {
                command: "rm -rf /".to_string(),
                timeout: None,
                description: None,
                run_in_background: None,
            },
            Some(&missing_cwd),
        );
        let ordinary = execute_powershell(
            &PowerShellInput {
                command: "echo hi".to_string(),
                timeout: None,
                description: None,
                run_in_background: None,
            },
            Some(&missing_cwd),
        );

        let err = catastrophic.expect_err("catastrophic PowerShell command must be refused");
        assert!(
            err.kind() == std::io::ErrorKind::PermissionDenied
                || err.to_string().contains("catastrophic"),
            "catastrophic PowerShell command must be refused, got {err:?}"
        );
        if let Err(err) = ordinary {
            assert_ne!(
                err.kind(),
                std::io::ErrorKind::PermissionDenied,
                "ordinary PowerShell command must not be refused by catastrophic backstop"
            );
            assert!(
                !err.to_string().contains("catastrophic"),
                "ordinary PowerShell command false-positived: {err}"
            );
        }
    }

    #[test]
    fn repl_refuses_catastrophic_in_any_language() {
        // The catastrophic hard-block must hold for EVERY REPL language, not just
        // shell: python/node can shell out, so they were a strictly-more-capable
        // unguarded path before this. The refusal happens before any subprocess
        // spawn, so this is hermetic (no python/node needed).
        for (language, code) in [
            ("bash", "rm -rf /"),
            ("sh", "rm -rf / --no-preserve-root"),
            ("python", "import os; os.system('rm -rf / --no-preserve-root')"),
            ("py", "os.system('rm -rf /')"),
            ("node", "require('child_process').execSync('rm -rf /')"),
            ("javascript", "child_process.execSync(`rm -rf /`)"),
        ] {
            let err = execute_repl(
                ReplInput {
                    code: code.to_string(),
                    language: language.to_string(),
                    timeout_ms: None,
                },
                None,
            )
            .unwrap_err();
            assert!(
                matches!(&err, ToolError::Execution(msg) if msg.contains("catastrophic")),
                "{language} REPL must refuse `{code}`, got {err:?}"
            );
        }
    }

    #[test]
    fn repl_allows_benign_code_in_any_language() {
        // The backstop must not false-positive on ordinary code with quoted
        // strings. These resolve a runtime then spawn; assert no catastrophic
        // refusal (the call may still fail if the interpreter is absent, which is
        // a different error).
        for (language, code) in [
            ("python", "print('hello world')"),
            ("node", "console.log('removing nothing')"),
            ("bash", "echo 'rm is a normal word here'"),
        ] {
            let result = execute_repl(
                ReplInput {
                    code: code.to_string(),
                    language: language.to_string(),
                    timeout_ms: Some(5_000),
                },
                None,
            );
            if let Err(ToolError::Execution(msg)) = &result {
                assert!(
                    !msg.contains("catastrophic"),
                    "{language} REPL false-positived on benign `{code}`: {msg}"
                );
            }
        }
    }

    #[test]
    fn repl_dispatch_is_permission_gated() {
        // A ReadOnly sub-agent must not run REPL ungated: REPL requires
        // DangerFullAccess, so the enforcer denies it at the dispatch seam
        // (previously the REPL arm carried no permission check at all).
        let ctx = ToolContext::new();
        let enforcer =
            PermissionEnforcer::new(runtime::PermissionPolicy::new(PermissionMode::ReadOnly));
        let input = json!({ "code": "touch x", "language": "bash" });
        let result = dispatch(&ctx, Some(&enforcer), "REPL", &input).expect("REPL is a known tool");
        assert!(result.is_err(), "ReadOnly REPL must be denied, got {result:?}");
    }
}
