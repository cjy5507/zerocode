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
    /// Full argv; `{prompt}` is replaced with the task prompt, `{run_root}`
    /// and `{workdir}` with the run/workspace directories.
    argv: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TaskConfig {
    id: String,
    prompt: String,
    /// Objective success gate, run with `sh -c` in the workspace after the
    /// tool exits. Exit 0 = success. This is the ONLY success signal.
    check: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    files: Vec<TaskFile>,
    /// Optional tag (e.g. "smoke", "rust") for --filter selection.
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct TaskFile {
    path: String,
    content: String,
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
    let argv: Vec<String> = tool
        .argv
        .iter()
        .map(|arg| expand_placeholders(arg, &task.prompt, &run_root_abs, &workdir_abs))
        .collect();
    let (program, rest) = argv
        .split_first()
        .ok_or_else(|| format!("tool {} has an empty argv", tool.name))?;

    let stdout_path = workdir.join(".bench-stdout.json");
    let stderr_path = workdir.join(".bench-stderr.log");
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
            expand_placeholders(value, &task.prompt, &run_root_abs, &workdir_abs),
        );
    }

    let timeout = Duration::from_secs(task.timeout_secs.unwrap_or(defaults.timeout_secs));
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|e| format!("spawn {program}: {e}"))?;
    let (tool_exit, timed_out) = wait_with_timeout(&mut child, timeout);
    let wall_ms = started.elapsed().as_millis();

    let stdout = std::fs::read_to_string(&stdout_path).unwrap_or_default();
    let metrics = parse_tool_metrics(&stdout);

    let check_exit = run_check(&task.check, &workdir, defaults.check_timeout_secs);
    Ok(RowRecord {
        task: task.id.clone(),
        tool: tool.name.clone(),
        trial,
        success: check_exit == Some(0),
        wall_ms,
        check_exit,
        tool_exit,
        timed_out,
        cost_usd: metrics.cost_usd,
        input_tokens: metrics.input_tokens,
        output_tokens: metrics.output_tokens,
        cache_read_tokens: metrics.cache_read_tokens,
        cache_creation_tokens: metrics.cache_creation_tokens,
        tool_reported_error: metrics.reported_error,
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

fn expand_placeholders(arg: &str, prompt: &str, run_root: &Path, workdir: &Path) -> String {
    arg.replace("{prompt}", prompt)
        .replace("{run_root}", &run_root.display().to_string())
        .replace("{workdir}", &workdir.display().to_string())
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

    #[test]
    fn placeholder_expansion_covers_prompt_and_paths() {
        let expanded = expand_placeholders(
            "--session-root={run_root}/s --task={prompt} --dir={workdir}",
            "fix the bug",
            Path::new("/tmp/run"),
            Path::new("/tmp/run/w"),
        );
        assert_eq!(
            expanded,
            "--session-root=/tmp/run/s --task=fix the bug --dir=/tmp/run/w"
        );
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
