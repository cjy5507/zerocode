//! Phase-0 evaluation for the context-compression layer
//! (docs/goal-headroom-context-compression-eval-2026-06-11.md).
//!
//! Measures, on *this* repository with the *real* tool implementations, how
//! many model-facing chars the structural compressors in
//! `runtime::context_compression` save versus the current pipeline
//! (pretty-JSON envelope → `truncate_tool_output`). No network, no model —
//! the corpus is generated live by calling the same `file_ops`/process code
//! the dispatch path uses, then both pipelines are applied side by side.
//!
//! Run: `cargo run -p compat-harness --bin compression-eval`

// A measurement/report binary: usize→f64 casts are percentage displays where
// 52-bit precision is far beyond what the table prints, and the replay
// aggregator is a flat report function, clearer unsplit.
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use runtime::context_compression::compress_tool_output;
use runtime::{truncate_tool_output, TruncationConfig};

/// Mirrors `bash.rs::MAX_OUTPUT_BYTES` — stdout/stderr are capped to this many
/// bytes *before* the JSON envelope is built.
const BASH_STREAM_CAP_BYTES: usize = 16_384;

struct Sample {
    tool: &'static str,
    label: String,
    /// Exactly what the dispatch seam sees today: the pretty-JSON envelope.
    envelope: String,
    /// For `read_file`: the raw file content, to verify losslessness.
    lossless_reference: Option<String>,
}

struct Row {
    tool: &'static str,
    label: String,
    envelope_chars: usize,
    baseline_chars: usize,
    compressed_chars: usize,
    baseline_truncated: bool,
    compressed_truncated: bool,
    lossless_verified: Option<bool>,
    /// The large-code-file outline view was chosen (reversible by marker
    /// construction — verified by unit tests — but not byte-identical).
    outline: bool,
    /// Of the original payload lines, how many survive verbatim in each
    /// model-facing view (None when there is no line-oriented reference).
    baseline_coverage: Option<f64>,
    compressed_coverage: Option<f64>,
}

/// Fraction of non-blank `reference` lines present in `view`. The baseline
/// view is a (possibly truncated) JSON envelope, so there the needle is each
/// line's JSON-escaped form; the compressed view carries lines verbatim.
fn line_coverage(reference: &str, view: &str, json_escaped_needles: bool) -> f64 {
    let needles: Vec<String> = reference
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            if json_escaped_needles {
                let quoted = serde_json::to_string(line).expect("escape line");
                quoted[1..quoted.len() - 1].to_string()
            } else {
                line.to_string()
            }
        })
        .collect();
    if needles.is_empty() {
        return 100.0;
    }
    let hits = needles
        .iter()
        .filter(|needle| view.contains(needle.as_str()))
        .count();
    hits as f64 / needles.len() as f64 * 100.0
}

fn main() {
    let root = workspace_root();
    std::env::set_current_dir(&root).expect("chdir to workspace root");
    println!("# compression-eval — Phase 0 measurement\n");
    println!("workspace: {}\n", root.display());

    let config = TruncationConfig::default();
    let mut samples: Vec<Sample> = Vec::new();
    samples.extend(read_file_samples(&root));
    samples.extend(grep_samples(&root));
    samples.extend(glob_samples());
    samples.extend(bash_samples());

    let mut rows: Vec<Row> = Vec::new();
    for sample in &samples {
        rows.push(evaluate(sample, &root, &config));
    }

    print_table(&rows);
    print_verdict(&rows);

    // Replay every recorded session: the decisive number. Each logged
    // tool_result is exactly what the model was sent at the time; normalizing
    // it to what *today's* code would send (the `originalFile` echo has since
    // been `#[serde(skip)]`ped) and re-running both pipelines yields the
    // savings on the true tool-call distribution, not a synthetic corpus.
    let sessions_dir = root.join(".zo").join("sessions");
    if sessions_dir.is_dir() {
        replay_sessions(&sessions_dir, &root, &config);
    } else {
        eprintln!(
            "no session log dir at {} — replay skipped",
            sessions_dir.display()
        );
    }
}

// ---------------------------------------------------------------------------
// session replay
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ReplayAgg {
    calls: usize,
    compressed_calls: usize,
    baseline_chars: usize,
    compressed_chars: usize,
}

fn replay_sessions(dir: &Path, root: &Path, config: &TruncationConfig) {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect(),
        Err(error) => {
            eprintln!("cannot read sessions dir: {error}");
            return;
        }
    };
    files.sort();

    let mut by_tool: std::collections::BTreeMap<String, ReplayAgg> =
        std::collections::BTreeMap::new();
    let mut parse_failures = 0usize;
    let mut museum_records = 0usize;

    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            parse_failures += 1;
            continue;
        };
        for line in text.lines() {
            let Ok(record) = serde_json::from_str::<serde_json::Value>(line) else {
                parse_failures += 1;
                continue;
            };
            if record.get("type").and_then(|v| v.as_str()) != Some("message") {
                continue;
            }
            let Some(blocks) = record
                .get("message")
                .and_then(|m| m.get("blocks"))
                .and_then(|b| b.as_array())
            else {
                continue;
            };
            for block in blocks {
                let Some(tool) = block.get("tool_name").and_then(|v| v.as_str()) else {
                    continue;
                };
                let Some(output) = block.get("output").and_then(|v| v.as_str()) else {
                    continue;
                };
                // edit/write envelopes cut mid-JSON predate the
                // `originalFile` `#[serde(skip)]` fix — today's wire never
                // produces them (the envelope no longer carries the whole
                // file), and the cut destroyed the bytes needed to normalize
                // them. Excluding beats counting them as incompressible.
                if (tool == "edit_file" || tool == "write_file")
                    && output.contains("[output truncated")
                    && serde_json::from_str::<serde_json::Value>(output).is_err()
                {
                    museum_records += 1;
                    continue;
                }
                let baseline_raw = normalize_to_current_wire(tool, output);
                let baseline = truncate_tool_output(&baseline_raw, tool, config);
                let view = compress_tool_output(&baseline.content, tool, Some(root));
                let compressed = truncate_tool_output(&view.content, tool, config);

                let agg = by_tool.entry(tool.to_string()).or_default();
                agg.calls += 1;
                if view.was_compressed {
                    agg.compressed_calls += 1;
                }
                agg.baseline_chars += baseline.content.chars().count();
                agg.compressed_chars += compressed.content.chars().count();
            }
        }
    }

    println!("## Session replay ({} session files)\n", files.len());
    println!("| tool | calls | rewritten | baseline chars | compressed chars | saved |");
    println!("|---|---:|---:|---:|---:|---:|");
    let mut total = ReplayAgg::default();
    for agg in by_tool.values() {
        total.calls += agg.calls;
        total.compressed_calls += agg.compressed_calls;
        total.baseline_chars += agg.baseline_chars;
        total.compressed_chars += agg.compressed_chars;
    }
    let mut ordered: Vec<(&String, &ReplayAgg)> = by_tool.iter().collect();
    ordered.sort_by_key(|(_, agg)| std::cmp::Reverse(agg.baseline_chars));
    for (tool, agg) in &ordered {
        // Keep the table readable: skip tools that contribute < 0.1% volume.
        if agg.baseline_chars * 1000 < total.baseline_chars {
            continue;
        }
        println!(
            "| {} | {} | {} | {} | {} | {:.1}% |",
            tool,
            agg.calls,
            agg.compressed_calls,
            agg.baseline_chars,
            agg.compressed_chars,
            pct_saved(agg.baseline_chars, agg.compressed_chars),
        );
    }
    println!(
        "| **TOTAL** | {} | {} | {} | {} | **{:.1}%** |",
        total.calls,
        total.compressed_calls,
        total.baseline_chars,
        total.compressed_chars,
        pct_saved(total.baseline_chars, total.compressed_chars),
    );
    if parse_failures > 0 {
        println!("\n(unparseable lines/files skipped: {parse_failures})");
    }
    if museum_records > 0 {
        println!(
            "(excluded {museum_records} edit/write records cut mid-JSON — a pre-`originalFile`-skip \
             wire shape today's code no longer produces)"
        );
    }
    println!(
        "\nReplay verdict: {:.1}% of all model-facing tool-output chars saved on the real \
         session corpus (baseline normalized to today's wire format).",
        pct_saved(total.baseline_chars, total.compressed_chars)
    );
}

/// What today's dispatch would send for this logged output: drop fields that
/// have since been `#[serde(skip)]`ped off the wire (`originalFile` on
/// edit/write envelopes). Logged outputs predating that fix would otherwise
/// inflate the baseline with chars current zo no longer sends.
fn normalize_to_current_wire(tool: &str, output: &str) -> String {
    if tool != "edit_file" && tool != "write_file" {
        return output.to_string();
    }
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(output) else {
        return output.to_string();
    };
    let Some(obj) = value.as_object_mut() else {
        return output.to_string();
    };
    if obj.remove("originalFile").is_none() {
        return output.to_string();
    }
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| output.to_string())
}

fn workspace_root() -> PathBuf {
    // compat-harness lives at <root>/crates/compat-harness.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// corpus
// ---------------------------------------------------------------------------

fn read_file_samples(root: &Path) -> Vec<Sample> {
    let targets = [
        ("large rust source", "crates/runtime/src/bash.rs"),
        ("mid rust source", "crates/tools/src/dispatch.rs"),
        (
            "small rust source",
            "crates/runtime/src/tool_output_truncation.rs",
        ),
        (
            "markdown doc",
            "docs/goal-headroom-context-compression-eval-2026-06-11.md",
        ),
        (
            "very large source",
            "crates/runtime/src/conversation/mod.rs",
        ),
    ];
    let mut samples = Vec::new();
    for (label, rel) in targets {
        let path = root.join(rel);
        if !path.exists() {
            eprintln!("skip read_file sample (missing): {rel}");
            continue;
        }
        let path_str = path.to_string_lossy().into_owned();
        match runtime::read_file(&path_str, None, None) {
            Ok(output) => {
                let reference = output.file.content.clone();
                let envelope =
                    serde_json::to_string_pretty(&output).expect("serialize read_file envelope");
                samples.push(Sample {
                    tool: "read_file",
                    label: format!("{label} ({rel})"),
                    envelope,
                    lossless_reference: Some(reference),
                });
            }
            Err(error) => eprintln!("skip read_file sample {rel}: {error}"),
        }
    }
    samples
}

fn grep_input(value: serde_json::Value) -> runtime::GrepSearchInput {
    serde_json::from_value(value).expect("grep input")
}

fn grep_samples(root: &Path) -> Vec<Sample> {
    let crates_dir = root.join("crates").to_string_lossy().into_owned();
    let specs = [
        (
            "content mode, common symbol",
            serde_json::json!({
                "pattern": "fn execute",
                "path": crates_dir,
                "output_mode": "content",
            }),
        ),
        (
            "content mode, broad keyword",
            serde_json::json!({
                "pattern": "truncat",
                "path": crates_dir,
                "output_mode": "content",
                "head_limit": 400,
            }),
        ),
        (
            "files_with_matches mode",
            serde_json::json!({
                "pattern": "tool_result",
                "path": crates_dir,
            }),
        ),
    ];
    let mut samples = Vec::new();
    for (label, spec) in specs {
        match runtime::grep_search(&grep_input(spec)) {
            Ok(output) => samples.push(Sample {
                tool: "grep_search",
                label: label.to_string(),
                envelope: serde_json::to_string_pretty(&output).expect("serialize grep envelope"),
                lossless_reference: output.content.clone(),
            }),
            Err(error) => eprintln!("skip grep sample {label}: {error}"),
        }
    }
    samples
}

fn glob_samples() -> Vec<Sample> {
    match runtime::glob_search("crates/**/*.rs", None) {
        Ok(output) => vec![Sample {
            tool: "glob_search",
            label: "crates/**/*.rs".to_string(),
            envelope: serde_json::to_string_pretty(&output).expect("serialize glob envelope"),
            lossless_reference: None,
        }],
        Err(error) => {
            eprintln!("skip glob sample: {error}");
            Vec::new()
        }
    }
}

fn bash_samples() -> Vec<Sample> {
    let specs: [(&str, &str); 4] = [
        ("git log --stat", "git log --stat -25"),
        (
            "ls -la (small listing)",
            "ls -la crates/runtime/src | head -40",
        ),
        (
            "cargo metadata (large JSON line)",
            "cargo metadata --no-deps --format-version 1 2>/dev/null",
        ),
        (
            "colored output (ANSI best case)",
            "git -c color.ui=always log --oneline --decorate -60",
        ),
    ];
    let mut samples = Vec::new();
    for (label, command) in specs {
        match run_shell(command) {
            Ok((stdout, stderr)) => {
                let envelope = bash_envelope(&stdout, &stderr);
                samples.push(Sample {
                    tool: "bash",
                    label: label.to_string(),
                    envelope,
                    lossless_reference: None,
                });
            }
            Err(error) => eprintln!("skip bash sample {label}: {error}"),
        }
    }
    samples
}

fn run_shell(command: &str) -> std::io::Result<(String, String)> {
    let output = Command::new("/bin/zsh").arg("-c").arg(command).output()?;
    Ok((
        cap_stream(&String::from_utf8_lossy(&output.stdout)),
        cap_stream(&String::from_utf8_lossy(&output.stderr)),
    ))
}

/// Reproduce `bash.rs::truncate_output`: byte cap + human-readable notice.
fn cap_stream(text: &str) -> String {
    if text.len() <= BASH_STREAM_CAP_BYTES {
        return text.to_string();
    }
    let mut end = BASH_STREAM_CAP_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[output truncated — exceeded {BASH_STREAM_CAP_BYTES} bytes]",
        &text[..end]
    )
}

/// The same envelope shape `BashCommandOutput` serializes to on the live path.
fn bash_envelope(stdout: &str, stderr: &str) -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "stdout": stdout,
        "stderr": stderr,
        "rawOutputPath": null,
        "interrupted": false,
        "isImage": null,
        "backgroundTaskId": null,
        "backgroundedByUser": null,
        "assistantAutoBackgrounded": null,
        "dangerouslyDisableSandbox": null,
        "returnCodeInterpretation": null,
        "noOutputExpected": null,
        "structuredContent": null,
        "persistedOutputPath": null,
        "persistedOutputSize": null,
        "sandboxStatus": null,
    }))
    .expect("serialize bash envelope")
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------

fn evaluate(sample: &Sample, root: &Path, config: &TruncationConfig) -> Row {
    let baseline = truncate_tool_output(&sample.envelope, sample.tool, config);
    let view = compress_tool_output(&sample.envelope, sample.tool, Some(root));
    let compressed = truncate_tool_output(&view.content, sample.tool, config);
    let outline = compressed.content.starts_with("[file:outline]");

    // Losslessness check (read_file / grep content): when nothing was cut by
    // the truncation layer, the compressed view must still contain the exact
    // reference text. The outline view is reversible-by-marker rather than
    // byte-identical; its round-trip is proven by unit tests, so it is
    // reported as its own category instead of pass/fail here.
    let lossless_verified = sample.lossless_reference.as_ref().map(|reference| {
        if compressed.was_truncated || outline {
            return false;
        }
        match sample.tool {
            "read_file" => match compressed.content.split_once('\n') {
                Some((_, body)) => body == reference,
                None => false,
            },
            "grep_search" => {
                // Every original content row must survive (possibly with the
                // path prefix moved into a group header).
                reference.lines().all(|row| {
                    let rest = row.splitn(3, ':').nth(2).unwrap_or(row);
                    compressed.content.contains(rest)
                })
            }
            _ => true,
        }
    });

    // Line coverage: how much of the original payload survives in each view.
    // For grep the model-facing rows drop the path prefix into a group header,
    // so compare on the text after the path to keep both sides comparable.
    let coverage_reference = sample.lossless_reference.as_ref().map(|reference| {
        if sample.tool == "grep_search" {
            reference
                .lines()
                .map(|row| row.splitn(3, ':').nth(2).unwrap_or(row).to_string())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            reference.clone()
        }
    });
    let (baseline_coverage, compressed_coverage) = match &coverage_reference {
        Some(reference) => (
            Some(line_coverage(reference, &baseline.content, true)),
            Some(line_coverage(reference, &compressed.content, false)),
        ),
        None => (None, None),
    };

    Row {
        tool: sample.tool,
        label: sample.label.clone(),
        envelope_chars: sample.envelope.chars().count(),
        baseline_chars: baseline.content.chars().count(),
        compressed_chars: compressed.content.chars().count(),
        baseline_truncated: baseline.was_truncated,
        compressed_truncated: compressed.was_truncated,
        lossless_verified,
        outline,
        baseline_coverage,
        compressed_coverage,
    }
}

fn pct_saved(baseline: usize, compressed: usize) -> f64 {
    if baseline == 0 {
        return 0.0;
    }
    (1.0 - compressed as f64 / baseline as f64) * 100.0
}

fn print_table(rows: &[Row]) {
    println!(
        "| tool | sample | envelope | model-facing (now) | model-facing (compressed) | saved | coverage now→new | lossless |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---|");
    for row in rows {
        let saved = pct_saved(row.baseline_chars, row.compressed_chars);
        let lossless = if row.outline {
            "outline (reversible)"
        } else {
            match row.lossless_verified {
                Some(true) => "✓",
                Some(false) if row.compressed_truncated => "n/a (truncated)",
                Some(false) => "✗ FAIL",
                None => "—",
            }
        };
        let baseline_mark = if row.baseline_truncated { " (cut)" } else { "" };
        let compressed_mark = if row.compressed_truncated {
            " (cut)"
        } else {
            ""
        };
        let coverage = match (row.baseline_coverage, row.compressed_coverage) {
            (Some(before), Some(after)) => format!("{before:.0}%→{after:.0}%"),
            _ => "—".to_string(),
        };
        println!(
            "| {} | {} | {} | {}{} | {}{} | {:.1}% | {} | {} |",
            row.tool,
            row.label,
            row.envelope_chars,
            row.baseline_chars,
            baseline_mark,
            row.compressed_chars,
            compressed_mark,
            saved,
            coverage,
            lossless,
        );
    }
    println!();
}

fn print_verdict(rows: &[Row]) {
    let mut by_tool: std::collections::BTreeMap<&str, (usize, usize)> =
        std::collections::BTreeMap::new();
    for row in rows {
        let entry = by_tool.entry(row.tool).or_insert((0, 0));
        entry.0 += row.baseline_chars;
        entry.1 += row.compressed_chars;
    }

    println!("## Aggregate by tool\n");
    let mut report = String::new();
    let mut read_heavy_saved = 0.0_f64;
    for (tool, (baseline, compressed)) in &by_tool {
        let saved = pct_saved(*baseline, *compressed);
        let _ = writeln!(
            report,
            "- {tool}: {baseline} → {compressed} chars ({saved:.1}% saved)"
        );
        if *tool == "read_file" {
            read_heavy_saved = saved;
        }
    }
    print!("{report}");

    let total_baseline: usize = by_tool.values().map(|(b, _)| *b).sum();
    let total_compressed: usize = by_tool.values().map(|(_, c)| *c).sum();
    let total_saved = pct_saved(total_baseline, total_compressed);
    println!("- TOTAL: {total_baseline} → {total_compressed} chars ({total_saved:.1}% saved)\n");

    let grep_glob_saved = {
        let baseline: usize = by_tool
            .iter()
            .filter(|(tool, _)| **tool == "grep_search" || **tool == "glob_search")
            .map(|(_, (b, _))| *b)
            .sum();
        let compressed: usize = by_tool
            .iter()
            .filter(|(tool, _)| **tool == "grep_search" || **tool == "glob_search")
            .map(|(_, (_, c))| *c)
            .sum();
        pct_saved(baseline, compressed)
    };

    let lossless_failures = rows
        .iter()
        .filter(|row| {
            row.lossless_verified == Some(false) && !row.compressed_truncated && !row.outline
        })
        .count();

    println!("## Phase-0 gate\n");
    println!("- read_file savings: {read_heavy_saved:.1}% (gate: material, target ≥30% on read-heavy turns)");
    println!("- grep+glob savings: {grep_glob_saved:.1}%");
    println!("- lossless verification failures: {lossless_failures}");
    let verdict = if lossless_failures == 0 && (read_heavy_saved >= 30.0 || total_saved >= 30.0) {
        "PASS — proceed to Phase 1 (wire into dispatch seam)"
    } else if lossless_failures == 0 && total_saved >= 10.0 {
        "PARTIAL — savings are real but below the 30% target; decide with the per-tool table"
    } else {
        "FAIL — do not adopt; savings not material or losslessness broken"
    };
    println!("- verdict: {verdict}");
}
