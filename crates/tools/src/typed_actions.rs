//! Phase-2 typed actions (No-SH-by-default).
//!
//! These tools run real binaries through a [`ProcessSpec`] — an explicit
//! `binary` + `args` + `cwd` + `timeout`, never a shell command string — so the
//! model and workflows can run `cargo`/`git` without the quoting, word-splitting,
//! injection, and policy ambiguity of `bash -lc`. `bash` stays as the escape
//! hatch for anything these typed actions don't cover, exactly the doc's "No-SH
//! by default, shell as controlled escape hatch" (§5.1).
//!
//! Scope is a deliberate prototype: [`ProcessSpec`] captures §9's *essential*
//! safety property (binary/args separation + a mandatory timeout + bounded
//! capture). The richer §9 fields (env allowlist, stdin artifact, sandbox,
//! expected-exit policy) are intentionally deferred — adding them before a caller
//! needs them would be speculative.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    from_value, maybe_enforce_permission_check, to_pretty_json, ToolContext, ToolError, ToolSpec,
};
use crate::bash_tools::wait_child_with_timeout;
use runtime::{permission_enforcer::PermissionEnforcer, PermissionMode};

/// Hard ceiling on a single typed-action run. Mirrors the bash tool's default
/// budget so a hung `cargo`/`git` can never wedge a turn.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Cap on captured stdout/stderr returned to the model (chars, not bytes, so the
/// slice always lands on a UTF-8 boundary). Build logs can be huge; the tail is
/// where the error usually is, so truncation keeps the tail.
const MAX_CAPTURE_CHARS: usize = 20_000;

// ---------------------------------------------------------------------------
// ProcessSpec (§9 prototype)
// ---------------------------------------------------------------------------

/// A typed, shell-free description of one external process to run. The binary
/// and args reach the OS verbatim — no quoting, no word-splitting, no `sh -lc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessSpec {
    pub binary: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub timeout: Duration,
}

impl ProcessSpec {
    /// A spec for `binary args...` on the process cwd with the default timeout.
    #[must_use]
    pub fn new(binary: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            binary: binary.into(),
            args,
            cwd: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    #[must_use]
    fn with_cwd(mut self, cwd: Option<PathBuf>) -> Self {
        self.cwd = cwd;
        self
    }
}

/// Structured result of running a [`ProcessSpec`] — the typed counterpart to the
/// bash tool's `BashCommandOutput`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessOutcome {
    pub binary: String,
    pub args: Vec<String>,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub duration_ms: u128,
}

/// Run a [`ProcessSpec`] with bounded capture and a hard timeout. Never invokes a
/// shell: `binary` + `args` are handed to the OS as-is. A timeout is reported as
/// a structured `timed_out` outcome (not an error) so callers can branch on it;
/// only a failure to *start* the process is an `Err`.
pub fn run_process_spec(spec: &ProcessSpec) -> Result<ProcessOutcome, ToolError> {
    let started = Instant::now();
    // Inherit the same sandbox as bash (WI-E): on Linux the argv is prepended
    // with the `unshare` launcher, on macOS (Seatbelt opt-in) with
    // `sandbox-exec -p <profile>`. A no-op on the default macOS/Windows path, so
    // a plain typed action still execs its binary directly.
    let cwd = spec
        .cwd
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let status =
        runtime::sandbox::resolve_sandbox_status(&runtime::sandbox::SandboxConfig::default(), &cwd);
    let (program, args, env) =
        runtime::sandbox::sandbox_wrap_argv(&spec.binary, &spec.args, &cwd, &status);
    let mut command = Command::new(&program);
    command
        .args(&args)
        .envs(env)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = spec.cwd.as_deref() {
        command.current_dir(cwd);
    }
    let child = command.spawn().map_err(|e| {
        ToolError::Execution(format!(
            "failed to start `{}` (is it installed and on PATH?): {e}",
            spec.binary
        ))
    })?;
    match wait_child_with_timeout(child, spec.timeout) {
        Ok(output) => Ok(ProcessOutcome {
            binary: spec.binary.clone(),
            args: spec.args.clone(),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: bounded(&output.stdout),
            stderr: bounded(&output.stderr),
            timed_out: false,
            duration_ms: started.elapsed().as_millis(),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => Ok(ProcessOutcome {
            binary: spec.binary.clone(),
            args: spec.args.clone(),
            exit_code: -1,
            stdout: String::new(),
            stderr: format!("process exceeded timeout of {}s", spec.timeout.as_secs()),
            timed_out: true,
            duration_ms: started.elapsed().as_millis(),
        }),
        Err(e) => Err(ToolError::Execution(e.to_string())),
    }
}

/// Lossy-decode and cap captured output to the trailing [`MAX_CAPTURE_CHARS`]
/// characters (errors usually surface at the end of a build log).
fn bounded(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let count = text.chars().count();
    if count <= MAX_CAPTURE_CHARS {
        return text.into_owned();
    }
    let tail: String = text.chars().skip(count - MAX_CAPTURE_CHARS).collect();
    format!("…[truncated, kept last {MAX_CAPTURE_CHARS} chars]…\n{tail}")
}

// ---------------------------------------------------------------------------
// Cargo / Git typed actions
// ---------------------------------------------------------------------------

/// A `cargo` subcommand the model can run as a typed action. Each maps to one
/// fixed argv head; user-supplied `args` are appended verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CargoAction {
    Check,
    Test,
    Fmt,
    Clippy,
    Build,
    Run,
}

impl CargoAction {
    const fn subcommand(self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Test => "test",
            Self::Fmt => "fmt",
            Self::Clippy => "clippy",
            Self::Build => "build",
            Self::Run => "run",
        }
    }

    /// Compile to a shell-free [`ProcessSpec`]: `cargo <sub> <extra...>`.
    #[must_use]
    pub fn to_process_spec(self, extra: &[String], cwd: Option<PathBuf>) -> ProcessSpec {
        let mut args = Vec::with_capacity(1 + extra.len());
        args.push(self.subcommand().to_string());
        args.extend(extra.iter().cloned());
        ProcessSpec::new("cargo", args).with_cwd(cwd)
    }
}

/// A read-only `git` inspection the model can run as a typed action. Mutating
/// git (commit/push/merge) intentionally stays on `bash`, where the existing
/// branch-guard / preflight and the doc's §14 human-gate apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitAction {
    Status,
    Diff,
    Log,
    Show,
    Branch,
}

impl GitAction {
    const fn subcommand(self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Diff => "diff",
            Self::Log => "log",
            Self::Show => "show",
            Self::Branch => "branch",
        }
    }

    /// Compile to a shell-free [`ProcessSpec`]: `git <sub> <extra...>`.
    #[must_use]
    pub fn to_process_spec(self, extra: &[String], cwd: Option<PathBuf>) -> ProcessSpec {
        let mut args = Vec::with_capacity(1 + extra.len());
        args.push(self.subcommand().to_string());
        args.extend(extra.iter().cloned());
        ProcessSpec::new("git", args).with_cwd(cwd)
    }
}

// ---------------------------------------------------------------------------
// Model-facing tools
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CargoToolInput {
    action: CargoAction,
    #[serde(default)]
    args: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GitToolInput {
    action: GitAction,
    #[serde(default)]
    args: Vec<String>,
}

const CARGO_DESCRIPTION: &str =
    "Run a Cargo subcommand as a typed action — no shell. Prefer this over \
`bash \"cargo …\"` for Rust build/test/lint: args are passed verbatim (no quoting), with a \
120s timeout and bounded output. `action` is one of check|test|fmt|clippy|build|run; `args` are \
extra arguments, e.g. {\"action\":\"test\",\"args\":[\"-p\",\"tools\"]}.";

const GIT_DESCRIPTION: &str = "Run a read-only Git inspection as a typed action — no shell. Prefer this \
over `bash \"git …\"` for inspecting repository state. `action` is one of status|diff|log|show|branch; \
`args` are extra arguments, e.g. {\"action\":\"log\",\"args\":[\"-n\",\"5\",\"--oneline\"]}. Mutating git \
(commit/push) still goes through bash.";

pub(crate) fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "Cargo",
            description: CARGO_DESCRIPTION,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["check", "test", "fmt", "clippy", "build", "run"]
                    },
                    "args": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["action"],
                "additionalProperties": false
            }),
            // cargo build/test write to `target/`; treat as workspace-write.
            required_permission: PermissionMode::WorkspaceWrite,
        },
        ToolSpec {
            name: "Git",
            description: GIT_DESCRIPTION,
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "show", "branch"]
                    },
                    "args": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["action"],
                "additionalProperties": false
            }),
            // Read-only inspection only — safe in the most restrictive mode.
            required_permission: PermissionMode::ReadOnly,
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
    // `None` leaves the typed action on the process cwd, like the bash tool.
    let cwd = ctx.cwd.as_deref();
    match name {
        "Cargo" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<CargoToolInput>(input).and_then(|inp| run_cargo(inp, cwd))
            }),
        ),
        "Git" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| from_value::<GitToolInput>(input).and_then(|inp| run_git(inp, cwd))),
        ),
        _ => None,
    }
}

fn run_cargo(input: CargoToolInput, cwd: Option<&Path>) -> Result<String, ToolError> {
    let CargoToolInput { action, args } = input;
    let spec = action.to_process_spec(&args, cwd.map(Path::to_path_buf));
    to_pretty_json(run_process_spec(&spec)?)
}

fn run_git(input: GitToolInput, cwd: Option<&Path>) -> Result<String, ToolError> {
    let GitToolInput { action, args } = input;
    let spec = action.to_process_spec(&args, cwd.map(Path::to_path_buf));
    to_pretty_json(run_process_spec(&spec)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_action_compiles_to_shell_free_spec() {
        let spec =
            CargoAction::Test.to_process_spec(&["-p".to_string(), "tools".to_string()], None);
        assert_eq!(spec.binary, "cargo");
        assert_eq!(spec.args, vec!["test", "-p", "tools"]);
        assert_eq!(spec.cwd, None);
        assert_eq!(spec.timeout, DEFAULT_TIMEOUT);
    }

    #[test]
    fn git_action_compiles_to_shell_free_spec() {
        let spec = GitAction::Log.to_process_spec(&["--oneline".to_string()], None);
        assert_eq!(spec.binary, "git");
        assert_eq!(spec.args, vec!["log", "--oneline"]);
    }

    #[test]
    fn cargo_tool_input_parses_action_and_defaults_args() {
        let parsed: CargoToolInput = serde_json::from_value(json!({ "action": "clippy" })).unwrap();
        assert_eq!(parsed.action, CargoAction::Clippy);
        assert!(parsed.args.is_empty());
    }

    #[test]
    fn run_process_spec_executes_without_a_shell() {
        // `cargo --version` is always available in the test toolchain and never
        // mutates anything — a hermetic check that the typed runner actually
        // spawns a real binary (no shell) and captures structured output.
        // Spawning inherits the process-global cwd, so serialize against the
        // cwd-mutating tests (which hold `env_lock`); otherwise a concurrent
        // `set_current_dir` can invalidate the cwd mid-spawn and fail the run.
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let spec = ProcessSpec::new("cargo", vec!["--version".to_string()]);
        let outcome = run_process_spec(&spec).expect("cargo --version runs");
        assert_eq!(outcome.exit_code, 0);
        assert!(!outcome.timed_out);
        assert!(
            outcome.stdout.starts_with("cargo "),
            "stdout: {}",
            outcome.stdout
        );
    }

    #[test]
    fn run_process_spec_reports_missing_binary_as_error() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let spec = ProcessSpec::new("zo-no-such-binary-xyz", vec![]);
        let err = run_process_spec(&spec).expect_err("missing binary fails to start");
        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[test]
    fn run_process_spec_times_out_without_hanging() {
        let _guard = crate::tests::env_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A 1ms budget against a process that sleeps far longer must come back as
        // a structured timeout, not a hang or an error.
        let spec = ProcessSpec {
            binary: "sleep".to_string(),
            args: vec!["5".to_string()],
            cwd: None,
            timeout: Duration::from_millis(1),
        };
        let outcome = run_process_spec(&spec).expect("timeout is a structured outcome");
        assert!(outcome.timed_out);
        assert_ne!(outcome.exit_code, 0);
    }
}
