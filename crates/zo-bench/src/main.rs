//! `zo-bench` — the scorecard harness behind the "better than every CLI/TUI
//! coding tool" claim. Runs the SAME task set through each configured tool in
//! an isolated workspace and measures the three axes that matter: did it
//! succeed (an objective check command, never a model's opinion), how long it
//! took (runner wall clock, never the tool's self-report), and what it cost
//! (the tool's own `--output-format json` usage record).
//!
//! Everything is data: tools, models, tasks, fixtures, and checks live in a
//! TOML file the operator edits. The binary knows how to run a matrix and
//! score it — it does not know what "zo" or "claude" are.
//!
//! Output contract (consumed by `/refine` in M5): one `rows.jsonl` with a
//! record per (task, tool, trial), plus `scoreboard.json` and a rendered
//! `scoreboard.md` per run directory.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
struct BenchConfig {
    #[serde(default)]
    defaults: Defaults,
    tools: Vec<ToolConfig>,
    tasks: Vec<TaskConfig>,
}

#[derive(Debug, Deserialize)]
struct Defaults {
    #[serde(default = "default_timeout_secs")]
    timeout_secs: u64,
    #[serde(default = "default_trials")]
    trials: u32,
    #[serde(default = "default_check_timeout_secs")]
    check_timeout_secs: u64,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            timeout_secs: default_timeout_secs(),
            trials: default_trials(),
            check_timeout_secs: default_check_timeout_secs(),
        }
    }
}

const fn default_timeout_secs() -> u64 {
    300
}
const fn default_trials() -> u32 {
    1
}
const fn default_check_timeout_secs() -> u64 {
    120
}

#[derive(Debug, Deserialize)]
struct ToolConfig {
    name: String,
    /// Full argv; `{prompt}` is replaced with the (stage) prompt, `{run_root}`
    /// and `{workdir}` with the run/workspace directories, and `{session_id}`
    /// with the per-(task, tool, trial) session id (UUID-shaped, so tools that
    /// require a UUID accept it).
    argv: Vec<String>,
    /// Argv for stage 2+ of a multi-stage task, for tools whose "continue this
    /// session" flag differs from their "name this session" flag (Claude Code:
    /// `--session-id X` creates, `--resume X` continues). Absent = stages
    /// reuse `argv` (zo: `--session-id X` both creates and continues).
    #[serde(default)]
    resume_argv: Option<Vec<String>>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TaskConfig {
    id: String,
    /// Single-stage prompt. Exactly one of `prompt` / `stages` must be set.
    #[serde(default)]
    prompt: Option<String>,
    /// Multi-stage prompts, run in order against the SAME session and the
    /// SAME workspace — the long-horizon lane: later stages measure whether
    /// the tool actually carries the earlier conversation (and at what cost).
    #[serde(default)]
    stages: Vec<StageConfig>,
    /// Objective success gate, run with `sh -c` in the workspace after the
    /// tool exits. Exit 0 = success. This is the ONLY success signal.
    check: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    files: Vec<TaskFile>,
    /// Optional tag (e.g. "smoke", "rust", "long") for --filter selection.
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StageConfig {
    prompt: String,
    /// Optional mid-stage gate; a red stage check fails the whole task even
    /// if a later stage papers over it.
    #[serde(default)]
    check: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskFile {
    path: String,
    content: String,
    /// A file the task owns rather than the tool: a test the prompt says not
    /// to touch, or a check script. Every such file is written back verbatim
    /// before the check runs, so "do not modify the test" stops being an
    /// honour rule that a tool can score a pass by breaking.
    #[serde(default)]
    readonly: bool,
}

#[derive(Debug, Serialize)]
struct RowRecord {
    task: String,
    tool: String,
    trial: u32,
    success: bool,
    wall_ms: u128,
    check_exit: Option<i32>,
    tool_exit: Option<i32>,
    timed_out: bool,
    cost_usd: Option<f64>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_creation_tokens: u64,
    tool_reported_error: bool,
    /// Number of prompts this row spans (1 = classic single-shot task).
    stages: u32,
    /// Task-owned files (tests, check scripts) the tool had modified, restored
    /// verbatim before the check ran. Non-empty means the tool edited the
    /// measurement — the row still scores on the declared check, and this names
    /// what it touched.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    restored_files: Vec<String>,
    /// Per-stage breakdown for multi-stage tasks (`None` on single-shot rows,
    /// keeping legacy row shape byte-compatible). This is where the
    /// long-horizon economics live: a stage-3 cost far above stage 1's on the
    /// same tool means the session carry is re-billing history instead of
    /// caching it.
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_detail: Option<Vec<StageDetail>>,
}

#[derive(Debug, Serialize)]
struct StageDetail {
    stage: u32,
    wall_ms: u128,
    cost_usd: Option<f64>,
    tokens: u64,
    tool_exit: Option<i32>,
    timed_out: bool,
    check_exit: Option<i32>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("zo-bench: {error}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    "usage: zo-bench [--tasks crates/zo-bench/tasks.toml] [--out bench/results] \
     [--only id[,id…]] [--tag tag] [--tool name[,name…]] [--trials N]"
        .to_string()
}

struct CliArgs {
    tasks_path: PathBuf,
    out_dir: PathBuf,
    only: Option<Vec<String>>,
    tag: Option<String>,
    tools: Option<Vec<String>>,
    trials: Option<u32>,
}

fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    let mut parsed = CliArgs {
        tasks_path: PathBuf::from("crates/zo-bench/tasks.toml"),
        // `/bench/` is the repo's ignored output area by convention.
        out_dir: PathBuf::from("bench/results"),
        only: None,
        tag: None,
        tools: None,
        trials: None,
    };
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let mut value = |name: &str| {
            iter.next()
                .cloned()
                .ok_or_else(|| format!("{name} requires a value\n{}", usage()))
        };
        match flag.as_str() {
            "--tasks" => parsed.tasks_path = PathBuf::from(value("--tasks")?),
            "--out" => parsed.out_dir = PathBuf::from(value("--out")?),
            "--only" => {
                parsed.only = Some(value("--only")?.split(',').map(str::to_string).collect());
            }
            "--tag" => parsed.tag = Some(value("--tag")?),
            "--tool" => {
                parsed.tools = Some(value("--tool")?.split(',').map(str::to_string).collect());
            }
            "--trials" => {
                parsed.trials = Some(
                    value("--trials")?
                        .parse()
                        .map_err(|e| format!("--trials: {e}"))?,
                );
            }
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown flag {other}\n{}", usage())),
        }
    }
    Ok(parsed)
}

#[allow(clippy::too_many_lines)] // one linear run loop; helpers carry the logic
fn run(args: &[String]) -> Result<(), String> {
    let cli = parse_args(args)?;
    let raw = std::fs::read_to_string(&cli.tasks_path)
        .map_err(|e| format!("cannot read {}: {e}", cli.tasks_path.display()))?;
    let config: BenchConfig = toml::from_str(&raw).map_err(|e| format!("tasks file: {e}"))?;

    let tools: Vec<&ToolConfig> = config
        .tools
        .iter()
        .filter(|tool| {
            cli.tools
                .as_ref()
                .is_none_or(|names| names.iter().any(|name| name == &tool.name))
        })
        .collect();
    if tools.is_empty() {
        return Err("no tools selected".to_string());
    }
    let tasks: Vec<&TaskConfig> = config
        .tasks
        .iter()
        .filter(|task| {
            cli.only
                .as_ref()
                .is_none_or(|ids| ids.iter().any(|id| id == &task.id))
        })
        .filter(|task| {
            cli.tag
                .as_ref()
                .is_none_or(|tag| task.tags.iter().any(|candidate| candidate == tag))
        })
        .collect();
    if tasks.is_empty() {
        return Err("no tasks selected".to_string());
    }
    let trials = cli.trials.unwrap_or(config.defaults.trials).max(1);

    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let run_root = cli.out_dir.join(format!("run-{epoch}"));
    std::fs::create_dir_all(&run_root).map_err(|e| e.to_string())?;
    // Workspaces live OUTSIDE the repo tree: a fixture inside it is captured
    // by ancestor build config (the first full run had cargo claiming the
    // fixture for this repo's own workspace — one tool diagnosed and
    // detached itself, the other lost the task to the environment, which is
    // exactly the kind of confound a benchmark must not contain).
    let work_base = std::env::temp_dir().join(format!("zo-bench-{epoch}"));
    std::fs::create_dir_all(&work_base).map_err(|e| e.to_string())?;
    eprintln!("workspaces: {}", work_base.display());
    let rows_path = run_root.join("rows.jsonl");
    let mut rows_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rows_path)
        .map_err(|e| e.to_string())?;

    let mut rows: Vec<RowRecord> = Vec::new();
    for task in &tasks {
        for tool in &tools {
            for trial in 1..=trials {
                eprintln!("── {} × {} (trial {trial}/{trials})", task.id, tool.name);
                let row = run_one(&config.defaults, &run_root, &work_base, task, tool, trial)?;
                let line = serde_json::to_string(&row).map_err(|e| e.to_string())?;
                writeln!(rows_file, "{line}").map_err(|e| e.to_string())?;
                eprintln!(
                    "   {} · {}ms · ${:.4} · check_exit={:?}",
                    if row.success { "SUCCESS" } else { "FAIL" },
                    row.wall_ms,
                    row.cost_usd.unwrap_or(0.0),
                    row.check_exit,
                );
                rows.push(row);
            }
        }
    }

    let mut scoreboard = build_scoreboard(&rows);
    // Stamp the run's own reproduction command: whoever reads this artifact
    // later (e.g. /refine's staleness note) can name the EXACT invocation
    // that refreshes it instead of guessing at flags.
    scoreboard["rerun_args"] = serde_json::json!(std::env::args().skip(1).collect::<Vec<_>>());
    std::fs::write(
        run_root.join("scoreboard.json"),
        serde_json::to_string_pretty(&scoreboard).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let rendered = render_scoreboard(&scoreboard);
    std::fs::write(run_root.join("scoreboard.md"), &rendered).map_err(|e| e.to_string())?;
    println!("{rendered}");
    println!("rows: {}", rows_path.display());
    Ok(())
}

/// The stage prompts a task runs: its `stages` when declared, else the single
/// `prompt` as a one-stage list.
fn task_stages(task: &TaskConfig) -> Result<Vec<StageConfig>, String> {
    if !task.stages.is_empty() {
        if task.prompt.is_some() {
            return Err(format!(
                "task {}: set either `prompt` or `stages`, not both",
                task.id
            ));
        }
        return Ok(task
            .stages
            .iter()
            .map(|stage| StageConfig {
                prompt: stage.prompt.clone(),
                check: stage.check.clone(),
            })
            .collect());
    }
    match &task.prompt {
        Some(prompt) => Ok(vec![StageConfig {
            prompt: prompt.clone(),
            check: None,
        }]),
        None => Err(format!("task {}: needs `prompt` or `stages`", task.id)),
    }
}

/// UUID-v4-shaped session id (Claude Code validates the flag as a UUID; zo
/// accepts any `[A-Za-z0-9-]`). Derived from time + pid + a counter through
/// two hash rounds — uniqueness per bench run is all that matters here, not
/// cryptographic randomness.
fn make_session_id(seed: u64) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (
        seed,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    )
        .hash(&mut hasher);
    let hi = hasher.finish();
    hi.hash(&mut hasher);
    let lo = hasher.finish();
    let bytes: Vec<u8> = hi
        .to_be_bytes()
        .into_iter()
        .chain(lo.to_be_bytes())
        .collect();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-4{:01x}{:02x}-{:01x}{:01x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6] & 0x0f, bytes[7],
        8 + (bytes[8] & 0x03), bytes[8] & 0x0f, bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

#[allow(clippy::too_many_lines)] // one linear stage loop; helpers carry the logic
fn run_one(
    defaults: &Defaults,
    run_root: &Path,
    work_base: &Path,
    task: &TaskConfig,
    tool: &ToolConfig,
    trial: u32,
) -> Result<RowRecord, String> {
    let workdir = work_base
        .join(&task.id)
        .join(&tool.name)
        .join(format!("t{trial}"));
    std::fs::create_dir_all(&workdir).map_err(|e| e.to_string())?;
    for file in &task.files {
        let path = workdir.join(&file.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, &file.content).map_err(|e| e.to_string())?;
    }
    seed_git(&workdir);

    let run_root_abs = run_root
        .canonicalize()
        .unwrap_or_else(|_| run_root.to_path_buf());
    let workdir_abs = workdir.canonicalize().unwrap_or_else(|_| workdir.clone());
    let stages = task_stages(task)?;
    let session_id = make_session_id(u64::from(trial));
    let timeout = Duration::from_secs(task.timeout_secs.unwrap_or(defaults.timeout_secs));

    let mut total_wall: u128 = 0;
    let mut total_metrics = ToolMetrics::default();
    let mut last_tool_exit: Option<i32> = None;
    let mut any_timed_out = false;
    let mut stage_details: Vec<StageDetail> = Vec::new();
    let mut all_stage_checks_green = true;
    // Task-owned files the tool changed and the runner put back before judging.
    let mut restored_files: Vec<String> = Vec::new();

    for (index, stage) in stages.iter().enumerate() {
        let stage_no = u32::try_from(index).unwrap_or(u32::MAX) + 1;
        let template = if index > 0 {
            tool.resume_argv.as_ref().unwrap_or(&tool.argv)
        } else {
            &tool.argv
        };
        let argv: Vec<String> = template
            .iter()
            .map(|arg| {
                expand_placeholders(arg, &stage.prompt, &run_root_abs, &workdir_abs, &session_id)
            })
            .collect();
        let (program, rest) = argv
            .split_first()
            .ok_or_else(|| format!("tool {} has an empty argv", tool.name))?;

        let stdout_path = workdir.join(format!(".bench-stdout-s{stage_no}.json"));
        let stderr_path = workdir.join(format!(".bench-stderr-s{stage_no}.log"));
        let mut command = Command::new(program);
        command
            .args(rest)
            .current_dir(&workdir)
            .stdin(Stdio::null())
            .stdout(std::fs::File::create(&stdout_path).map_err(|e| e.to_string())?)
            .stderr(std::fs::File::create(&stderr_path).map_err(|e| e.to_string())?);
        for (key, value) in &tool.env {
            command.env(
                key,
                expand_placeholders(
                    value,
                    &stage.prompt,
                    &run_root_abs,
                    &workdir_abs,
                    &session_id,
                ),
            );
        }

        let started = Instant::now();
        let mut child = command
            .spawn()
            .map_err(|e| format!("spawn {program}: {e}"))?;
        let (tool_exit, timed_out) = wait_with_timeout(&mut child, timeout);
        let wall_ms = started.elapsed().as_millis();

        let stdout = std::fs::read_to_string(&stdout_path).unwrap_or_default();
        let metrics = parse_tool_metrics(&stdout);
        let stage_check_exit = stage.check.as_deref().map(|check| {
            restored_files.extend(restore_readonly_files(&task.files, &workdir));
            run_check(check, &workdir, defaults.check_timeout_secs).unwrap_or(-1)
        });
        if let Some(exit) = stage_check_exit {
            if exit != 0 {
                all_stage_checks_green = false;
            }
        }

        total_wall += wall_ms;
        total_metrics.accumulate(&metrics);
        last_tool_exit = tool_exit;
        any_timed_out |= timed_out;
        stage_details.push(StageDetail {
            stage: stage_no,
            wall_ms,
            cost_usd: metrics.cost_usd,
            tokens: metrics.input_tokens
                + metrics.output_tokens
                + metrics.cache_read_tokens
                + metrics.cache_creation_tokens,
            tool_exit,
            timed_out,
            check_exit: stage_check_exit,
        });
        if stages.len() > 1 {
            eprintln!(
                "   stage {stage_no}/{} · {}ms · ${:.4}",
                stages.len(),
                wall_ms,
                metrics.cost_usd.unwrap_or(0.0),
            );
        }
        // A timed-out stage leaves the session in an unknown state; later
        // stages would measure recovery, not carry. Stop the trial here.
        if timed_out {
            all_stage_checks_green = false;
            break;
        }
    }

    restored_files.extend(restore_readonly_files(&task.files, &workdir));
    restored_files.sort_unstable();
    restored_files.dedup();
    let check_exit = run_check(&task.check, &workdir, defaults.check_timeout_secs);
    let stage_count = u32::try_from(stage_details.len()).unwrap_or(u32::MAX);
    Ok(RowRecord {
        task: task.id.clone(),
        tool: tool.name.clone(),
        trial,
        success: check_exit == Some(0) && all_stage_checks_green,
        restored_files,
        wall_ms: total_wall,
        check_exit,
        tool_exit: last_tool_exit,
        timed_out: any_timed_out,
        cost_usd: total_metrics.cost_usd,
        input_tokens: total_metrics.input_tokens,
        output_tokens: total_metrics.output_tokens,
        cache_read_tokens: total_metrics.cache_read_tokens,
        cache_creation_tokens: total_metrics.cache_creation_tokens,
        tool_reported_error: total_metrics.reported_error,
        stages: stage_count,
        stage_detail: (stages.len() > 1).then_some(stage_details),
    })
}

/// A git repo is part of a realistic coding workspace (several harnesses key
/// verification and diff scoping off it), so every fixture gets one. Failure
/// is non-fatal: a tool must also cope with a bare directory.
fn seed_git(workdir: &Path) {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(workdir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()
    };
    git(&["init", "-q"]);
    git(&["add", "-A"]);
    git(&[
        "-c",
        "user.email=bench@zo.local",
        "-c",
        "user.name=zo-bench",
        "commit",
        "-qm",
        "fixture seed",
        "--no-gpg-sign",
    ]);
}

fn expand_placeholders(
    arg: &str,
    prompt: &str,
    run_root: &Path,
    workdir: &Path,
    session_id: &str,
) -> String {
    arg.replace("{prompt}", prompt)
        .replace("{run_root}", &run_root.display().to_string())
        .replace("{workdir}", &workdir.display().to_string())
        .replace("{session_id}", session_id)
}

/// Poll-wait with a deadline. On expiry the direct child is killed (grand-
/// children may linger until their stdin closes — acceptable for a bench).
fn wait_with_timeout(child: &mut std::process::Child, timeout: Duration) -> (Option<i32>, bool) {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return (status.code(), false),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (None, true);
                }
                std::thread::sleep(Duration::from_millis(120));
            }
            Err(_) => return (None, false),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
struct ToolMetrics {
    cost_usd: Option<f64>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_creation_tokens: u64,
    reported_error: bool,
}

impl ToolMetrics {
    /// Fold one stage's metrics into a multi-stage total. Cost stays `None`
    /// only while EVERY stage reported none — a single reporting stage makes
    /// the total honest-but-partial rather than silently zero.
    fn accumulate(&mut self, stage: &ToolMetrics) {
        if let Some(cost) = stage.cost_usd {
            *self.cost_usd.get_or_insert(0.0) += cost;
        }
        self.input_tokens += stage.input_tokens;
        self.output_tokens += stage.output_tokens;
        self.cache_read_tokens += stage.cache_read_tokens;
        self.cache_creation_tokens += stage.cache_creation_tokens;
        self.reported_error |= stage.reported_error;
    }
}

/// Pull the terminal metrics record out of a tool's stdout. Both supported
/// output shapes are handled: NDJSON (scan lines, keep the LAST record that
/// carries `total_cost_usd`) and one whole-stream JSON document.
fn parse_tool_metrics(stdout: &str) -> ToolMetrics {
    let mut record: Option<Value> = None;
    for line in stdout.lines() {
        if let Ok(value) = serde_json::from_str::<Value>(line) {
            if value.get("total_cost_usd").is_some() {
                record = Some(value);
            }
        }
    }
    if record.is_none() {
        if let Ok(value) = serde_json::from_str::<Value>(stdout) {
            if value.get("total_cost_usd").is_some() {
                record = Some(value);
            }
        }
    }
    let Some(record) = record else {
        return ToolMetrics::default();
    };
    let usage = record.get("usage").cloned().unwrap_or_else(|| json!({}));
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    ToolMetrics {
        cost_usd: record.get("total_cost_usd").and_then(Value::as_f64),
        input_tokens: count("input_tokens"),
        output_tokens: count("output_tokens"),
        cache_read_tokens: count("cache_read_input_tokens"),
        cache_creation_tokens: count("cache_creation_input_tokens"),
        reported_error: record
            .get("is_error")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || record.get("subtype").and_then(Value::as_str) == Some("error"),
    }
}

/// Put the task-owned files back exactly as declared and report which ones had
/// been changed.
///
/// The suite tells tools not to edit its tests, and until this ran, nothing
/// checked: a tool that rewrote `test_edge.py` — or the perf check that times
/// it — scored a clean pass. Restoring immediately before the check makes the
/// verdict independent of what the tool did to the measurement, and the
/// returned list turns a silent cheat into a recorded fact.
fn restore_readonly_files(files: &[TaskFile], workdir: &Path) -> Vec<String> {
    let mut restored = Vec::new();
    for file in files.iter().filter(|file| file.readonly) {
        let path = workdir.join(&file.path);
        let current = std::fs::read_to_string(&path).ok();
        if current.as_deref() == Some(file.content.as_str()) {
            continue;
        }
        if std::fs::write(&path, &file.content).is_ok() {
            restored.push(file.path.clone());
        }
    }
    restored
}

fn run_check(check: &str, workdir: &Path, timeout_secs: u64) -> Option<i32> {
    let mut child = Command::new("sh")
        .args(["-c", check])
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let (code, timed_out) = wait_with_timeout(&mut child, Duration::from_secs(timeout_secs));
    if timed_out {
        return None;
    }
    code
}

fn build_scoreboard(rows: &[RowRecord]) -> Value {
    let mut tools: BTreeMap<&str, Vec<&RowRecord>> = BTreeMap::new();
    for row in rows {
        tools.entry(&row.tool).or_default().push(row);
    }
    let summaries: Vec<Value> = tools
        .iter()
        .map(|(tool, rows)| {
            let successes = rows.iter().filter(|row| row.success).count();
            let mut walls: Vec<u128> = rows.iter().map(|row| row.wall_ms).collect();
            walls.sort_unstable();
            let median_wall = walls.get(walls.len() / 2).copied().unwrap_or(0);
            let cost: f64 = rows.iter().filter_map(|row| row.cost_usd).sum();
            let tokens: u64 = rows
                .iter()
                .map(|row| {
                    row.input_tokens
                        + row.output_tokens
                        + row.cache_read_tokens
                        + row.cache_creation_tokens
                })
                .sum();
            json!({
                "tool": tool,
                "tasks": rows.len(),
                "successes": successes,
                "median_wall_ms": median_wall,
                "total_cost_usd": cost,
                "total_tokens": tokens,
            })
        })
        .collect();
    let per_task: Vec<Value> = rows
        .iter()
        .map(|row| serde_json::to_value(row).unwrap_or(Value::Null))
        .collect();
    json!({ "tools": summaries, "rows": per_task })
}

fn render_scoreboard(scoreboard: &Value) -> String {
    let mut out = String::new();
    out.push_str("| tool | success | median wall | cost | tokens |\n");
    out.push_str("|---|---|---|---|---|\n");
    for tool in scoreboard
        .get("tools")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let get_u64 = |key: &str| tool.get(key).and_then(Value::as_u64).unwrap_or(0);
        out.push_str(&format!(
            "| {} | {}/{} | {}ms | ${:.4} | {} |\n",
            tool.get("tool").and_then(Value::as_str).unwrap_or("?"),
            get_u64("successes"),
            get_u64("tasks"),
            get_u64("median_wall_ms"),
            tool.get("total_cost_usd")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            get_u64("total_tokens"),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every "do not modify the test" in the suite was an honour rule until the
    /// runner started putting task-owned files back before judging. A tool that
    /// rewrites the test it is measured by must not score a pass, and the row
    /// must name what it touched.
    #[test]
    fn a_rewritten_test_is_restored_before_the_check_and_reported() {
        let workdir = std::env::temp_dir().join(format!(
            "zo-bench-restore-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&workdir).expect("workdir");
        let files = vec![
            TaskFile {
                path: "test_calc.py".to_string(),
                content: "assert calc() == 6\n".to_string(),
                readonly: true,
            },
            TaskFile {
                path: "untouched_test.py".to_string(),
                content: "assert True\n".to_string(),
                readonly: true,
            },
            TaskFile {
                path: "calc.py".to_string(),
                content: "def calc():\n    return 0\n".to_string(),
                readonly: false,
            },
        ];
        for file in &files {
            std::fs::write(workdir.join(&file.path), &file.content).expect("materialize");
        }
        // The tool neuters the test it is judged by and edits the file it is
        // supposed to edit.
        std::fs::write(workdir.join("test_calc.py"), "pass\n").expect("tamper");
        std::fs::write(workdir.join("calc.py"), "def calc():\n    return 6\n").expect("real work");

        let restored = restore_readonly_files(&files, &workdir);

        assert_eq!(restored, vec!["test_calc.py".to_string()], "only the changed task-owned file is reported");
        assert_eq!(
            std::fs::read_to_string(workdir.join("test_calc.py")).expect("read"),
            "assert calc() == 6\n",
            "the check judges the declared test, not the tool's rewrite"
        );
        assert_eq!(
            std::fs::read_to_string(workdir.join("calc.py")).expect("read"),
            "def calc():\n    return 6\n",
            "the tool's own work is left alone"
        );
        let _ = std::fs::remove_dir_all(&workdir);
    }

    #[test]
    fn placeholder_expansion_covers_prompt_paths_and_session() {
        let expanded = expand_placeholders(
            "--session-root={run_root}/s --task={prompt} --dir={workdir} --sid={session_id}",
            "fix the bug",
            Path::new("/tmp/run"),
            Path::new("/tmp/run/w"),
            "abc-123",
        );
        assert_eq!(
            expanded,
            "--session-root=/tmp/run/s --task=fix the bug --dir=/tmp/run/w --sid=abc-123"
        );
    }

    #[test]
    fn session_ids_are_uuid_shaped_and_distinct() {
        let a = make_session_id(1);
        let b = make_session_id(2);
        assert_ne!(a, b);
        for id in [&a, &b] {
            let parts: Vec<&str> = id.split('-').collect();
            assert_eq!(parts.len(), 5, "{id}");
            assert_eq!(
                parts.iter().map(|p| p.len()).collect::<Vec<_>>(),
                vec![8, 4, 4, 4, 12],
                "{id}"
            );
            assert!(
                id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'),
                "{id}"
            );
            // v4 marker: tools that validate the UUID version accept it.
            assert_eq!(parts[2].chars().next(), Some('4'), "{id}");
        }
    }

    #[test]
    fn task_stages_normalizes_single_prompt_and_rejects_ambiguity() {
        let single = TaskConfig {
            id: "t".into(),
            prompt: Some("do the thing".into()),
            stages: Vec::new(),
            check: "true".into(),
            timeout_secs: None,
            files: Vec::new(),
            tags: Vec::new(),
        };
        let stages = task_stages(&single).expect("single prompt is one stage");
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].prompt, "do the thing");

        let multi = TaskConfig {
            id: "t".into(),
            prompt: None,
            stages: vec![
                StageConfig {
                    prompt: "one".into(),
                    check: None,
                },
                StageConfig {
                    prompt: "two".into(),
                    check: Some("true".into()),
                },
            ],
            check: "true".into(),
            timeout_secs: None,
            files: Vec::new(),
            tags: Vec::new(),
        };
        assert_eq!(task_stages(&multi).expect("multi").len(), 2);

        let both = TaskConfig {
            id: "t".into(),
            prompt: Some("x".into()),
            stages: vec![StageConfig {
                prompt: "one".into(),
                check: None,
            }],
            check: "true".into(),
            timeout_secs: None,
            files: Vec::new(),
            tags: Vec::new(),
        };
        assert!(task_stages(&both).is_err(), "prompt+stages must be rejected");
        let neither = TaskConfig {
            id: "t".into(),
            prompt: None,
            stages: Vec::new(),
            check: "true".into(),
            timeout_secs: None,
            files: Vec::new(),
            tags: Vec::new(),
        };
        assert!(task_stages(&neither).is_err());
    }

    #[test]
    fn metrics_accumulate_sums_stages_and_keeps_cost_honest() {
        let mut total = ToolMetrics::default();
        total.accumulate(&ToolMetrics {
            cost_usd: None,
            input_tokens: 5,
            ..ToolMetrics::default()
        });
        assert_eq!(total.cost_usd, None, "no stage reported cost yet");
        total.accumulate(&ToolMetrics {
            cost_usd: Some(0.25),
            output_tokens: 7,
            reported_error: true,
            ..ToolMetrics::default()
        });
        assert_eq!(total.cost_usd, Some(0.25));
        assert_eq!(total.input_tokens, 5);
        assert_eq!(total.output_tokens, 7);
        assert!(total.reported_error);
    }

    #[test]
    fn metrics_parser_handles_ndjson_last_record_and_whole_doc() {
        let ndjson = concat!(
            "{\"type\":\"delta\"}\n",
            "{\"total_cost_usd\":0.5,\"usage\":{\"input_tokens\":10,\"output_tokens\":2}}\n",
            "{\"total_cost_usd\":0.9,\"usage\":{\"input_tokens\":20,\"output_tokens\":4,\
             \"cache_read_input_tokens\":7},\"is_error\":false}\n",
        );
        let metrics = parse_tool_metrics(ndjson);
        assert_eq!(metrics.cost_usd, Some(0.9), "last record wins");
        assert_eq!(metrics.input_tokens, 20);
        assert_eq!(metrics.cache_read_tokens, 7);
        assert!(!metrics.reported_error);

        let whole = "{\n  \"total_cost_usd\": 1.25,\n  \"usage\": {\"output_tokens\": 3},\n  \
                     \"subtype\": \"error\"\n}";
        let metrics = parse_tool_metrics(whole);
        assert_eq!(metrics.cost_usd, Some(1.25));
        assert_eq!(metrics.output_tokens, 3);
        assert!(metrics.reported_error);

        assert_eq!(parse_tool_metrics("no json here"), ToolMetrics::default());
    }

    #[test]
    fn scoreboard_aggregates_per_tool() {
        let row = |tool: &str, success: bool, wall: u128, cost: f64| RowRecord {
            task: "t".into(),
            tool: tool.into(),
            trial: 1,
            success,
            wall_ms: wall,
            check_exit: Some(i32::from(!success)),
            tool_exit: Some(0),
            timed_out: false,
            cost_usd: Some(cost),
            input_tokens: 100,
            output_tokens: 10,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            tool_reported_error: false,
            stages: 1,
            restored_files: Vec::new(),
            stage_detail: None,
        };
        let rows = vec![
            row("a", true, 1000, 0.10),
            row("a", true, 3000, 0.20),
            row("b", false, 2000, 0.40),
        ];
        let scoreboard = build_scoreboard(&rows);
        let tools = scoreboard["tools"].as_array().expect("tools");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["tool"], "a");
        assert_eq!(tools[0]["successes"], 2);
        assert_eq!(tools[0]["median_wall_ms"], 3000);
        assert_eq!(tools[1]["successes"], 0);
        let rendered = render_scoreboard(&scoreboard);
        assert!(rendered.contains("| a | 2/2 |"), "{rendered}");
    }
}
