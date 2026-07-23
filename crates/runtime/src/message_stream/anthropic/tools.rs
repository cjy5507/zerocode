#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

//! Pure, structured ports of the 15 ad-hoc tool formatters at
//! `zo-cli/src/main.rs:2252–2622`.
//!
//! Per `code-rules.md` R4 **and** R1, these functions never return
//! pre-rendered ANSI strings. They return the typed
//! [`ToolPreview`] / [`ToolResultBody`] / [`BashResult`] /
//! [`DiffView`] variants from
//! [`crate::message_stream::types`]. Stringification is the TUI
//! renderer's job.
//!
//! The functions are also deliberately *tolerant* of shape — the
//! legacy site uses `serde_json::Value` lookups with defaults
//! because the Anthropic API surfaces schemas it does not strictly
//! validate. We preserve that tolerance: unknown shapes become
//! [`ToolPreview::Generic`] / [`ToolResultBody::Generic`] rather
//! than hard errors.

use serde_json::Value;

use crate::compact_diff::{CompactDiffLineKind, compact_line_diff};
use crate::message_stream::types::{
    BashResult, DiffHunk, DiffLine, DiffLineKind, DiffView, TodoResultItem, TodoResultStatus,
    ToolPreview, ToolResultBody,
};

const TOOL_OUTPUT_DISPLAY_MAX_LINES: usize = 60;
const TOOL_OUTPUT_DISPLAY_MAX_CHARS: usize = 4_000;
const READ_DISPLAY_MAX_LINES: usize = 80;
const READ_DISPLAY_MAX_CHARS: usize = 6_000;

// ============================================================================
// Tool input previews
// ============================================================================

/// Convert a tool-call input JSON payload into a typed preview.
///
/// Recognises the 15 canonical tool names handled by the legacy
/// formatter at `main.rs:2252`. Unknown names fall through to
/// [`ToolPreview::Generic`].
#[must_use]
pub fn preview_tool_input(name: &str, input: &Value) -> ToolPreview {
    // Accept the short aliases (`read`, `write`, `edit`, `glob`, `grep`)
    // that the tools crate's dispatcher canonicalizes; PascalCase and
    // snake_case are already listed explicitly in the match arms below so
    // that both Anthropic API payloads and legacy snake_case callers
    // round-trip to the same typed preview. The original `name` is kept for
    // the `Generic` fallback so display output preserves the caller spelling.
    let canonical = canonical_preview_alias(name);
    match canonical {
        "bash" | "Bash" => ToolPreview::Bash {
            command: extract_str(input, "command").unwrap_or_default(),
        },
        "read_file" | "Read" => ToolPreview::Read {
            path: extract_tool_path(input),
            range: extract_read_range(input),
        },
        "write_file" | "Write" => {
            let path = extract_tool_path(input);
            let byte_count = input
                .get("content")
                .and_then(Value::as_str)
                .map_or(0, str::len);
            ToolPreview::Write { path, byte_count }
        }
        "edit_file" | "Edit" => {
            let path = extract_tool_path(input);
            let hunk_count = input
                .get("structuredPatch")
                .or_else(|| input.get("structured_patch"))
                .and_then(Value::as_array)
                .map_or(1, Vec::len);
            ToolPreview::Edit { path, hunk_count }
        }
        "glob_search" | "Glob" => ToolPreview::Glob {
            pattern: extract_str(input, "pattern").unwrap_or_else(|| "?".to_string()),
        },
        "grep_search" | "Grep" => ToolPreview::Grep {
            pattern: extract_str(input, "pattern").unwrap_or_else(|| "?".to_string()),
            path: extract_str(input, "path"),
        },
        "web_search" | "WebSearch" | "web_fetch" | "WebFetch" => ToolPreview::Search {
            query: extract_str(input, "query")
                .or_else(|| extract_str(input, "url"))
                .unwrap_or_else(|| "?".to_string()),
        },
        _ => ToolPreview::Generic {
            name: name.to_string(),
            input_summary: summarize_json(input, 96),
        },
    }
}

// ============================================================================
// Tool result bodies
// ============================================================================

/// Convert a tool-result JSON payload into a typed result body.
///
/// `is_error` drives the caller's [`crate::message_stream::types::RenderBlock::ToolResult::is_error`] flag;
/// here we only shape the body.
#[must_use]
pub fn format_tool_result(name: &str, output: &Value, is_error: bool) -> ToolResultBody {
    if is_error {
        return format_error_value_result(name, output);
    }

    let canonical = canonical_preview_alias(name);
    match canonical {
        "bash" | "Bash" => ToolResultBody::Bash(format_bash_result(output)),
        "read_file" | "Read" => format_read_result(output),
        "write_file" | "Write" => format_write_result(name, output),
        "edit_file" | "Edit" => format_edit_result(output),
        "glob_search" | "Glob" => format_glob_result(output),
        "grep_search" | "Grep" => format_grep_result(output),
        "TodoWrite" | "TaskList" => {
            format_todos_result(output).unwrap_or_else(|| format_generic_result(name, output))
        }
        _ => format_generic_result(name, output),
    }
}

/// Parse a `TodoWrite` / `TaskList` result into a typed checklist body so the
/// TUI renders a Claude-Code-style titled block instead of raw (and upstream
/// truncated) JSON. Reads the writer's `newTodos` array
/// (`tools::task_tools::TodoWriteOutput`); a `TaskList` `tasks` array is also
/// accepted. Returns `None` when neither shape is present so the caller falls
/// back to the generic renderer.
fn format_todos_result(output: &Value) -> Option<ToolResultBody> {
    let array = output
        .get("newTodos")
        .or_else(|| output.get("new_todos"))
        .or_else(|| output.get("tasks"))
        .or_else(|| output.get("todos"))
        .and_then(Value::as_array)?;
    let items: Vec<TodoResultItem> = array.iter().filter_map(parse_todo_result_item).collect();
    // An empty array (all work cleared) still gets the typed body so the row
    // reads "Updated Plan · all done" rather than a stray `{}` generic.
    Some(ToolResultBody::Todos(items))
}

fn parse_todo_result_item(value: &Value) -> Option<TodoResultItem> {
    let content = value.get("content").and_then(Value::as_str)?.to_string();
    let active_form = value
        .get("activeForm")
        .or_else(|| value.get("active_form"))
        .and_then(Value::as_str)
        .unwrap_or(&content)
        .to_string();
    let status = match value.get("status").and_then(Value::as_str) {
        Some("in_progress" | "inProgress") => TodoResultStatus::InProgress,
        Some("completed" | "complete" | "done") => TodoResultStatus::Completed,
        _ => TodoResultStatus::Pending,
    };
    Some(TodoResultItem {
        content,
        active_form,
        status,
    })
}

/// Convert a raw tool-result string into a typed result body.
///
/// This is the streaming hot-path entry point: obvious non-JSON generic/error
/// output is rendered directly from `&str`, avoiding both a failed
/// `serde_json` parse and the fallback `Value::String` allocation. JSON-shaped
/// output still goes through [`format_tool_result`] so structured read/edit/bash
/// cards keep their existing behavior.
#[must_use]
pub fn format_tool_result_from_raw(name: &str, raw: &str, is_error: bool) -> ToolResultBody {
    if might_be_json_value(raw) {
        if let Ok(output) = serde_json::from_str::<Value>(raw) {
            return format_tool_result(name, &output, is_error);
        }
    }

    if is_error {
        return format_plain_text_error_result(name, raw);
    }

    let canonical = canonical_preview_alias(name);
    if is_specialized_result_tool(canonical) {
        // Preserve the tolerant legacy fallback for malformed outputs from
        // tools with structured renderers. Some of those renderers intentionally
        // turn a string Value into a tool-specific generic/fallback body.
        let output = Value::String(raw.to_string());
        return format_tool_result(name, &output, false);
    }

    format_generic_text_result(name, raw)
}

fn format_plain_text_error_result(name: &str, raw: &str) -> ToolResultBody {
    format_error_text_result(name, raw)
}

fn format_error_value_result(name: &str, value: &Value) -> ToolResultBody {
    match value {
        Value::String(text) => format_error_text_result(name, text),
        other => format_error_text_result(name, &other.to_string()),
    }
}

fn format_error_text_result(name: &str, text: &str) -> ToolResultBody {
    let (content, truncated) = truncate(
        text,
        TOOL_OUTPUT_DISPLAY_MAX_LINES,
        TOOL_OUTPUT_DISPLAY_MAX_CHARS,
    );
    if name == "AskUserQuestion" {
        ToolResultBody::Generic {
            name: name.to_string(),
            content,
            truncated,
        }
    } else {
        ToolResultBody::Text { content, truncated }
    }
}

fn format_generic_text_result(name: &str, raw: &str) -> ToolResultBody {
    let (content, truncated) = truncate(
        raw,
        TOOL_OUTPUT_DISPLAY_MAX_LINES,
        TOOL_OUTPUT_DISPLAY_MAX_CHARS,
    );
    ToolResultBody::Generic {
        name: name.to_string(),
        content,
        truncated,
    }
}

fn is_specialized_result_tool(canonical: &str) -> bool {
    matches!(
        canonical,
        "bash"
            | "Bash"
            | "read_file"
            | "Read"
            | "write_file"
            | "Write"
            | "edit_file"
            | "Edit"
            | "glob_search"
            | "Glob"
            | "grep_search"
            | "Grep"
    )
}

fn might_be_json_value(raw: &str) -> bool {
    let trimmed = trim_json_whitespace_start(raw);
    match trimmed.as_bytes().first().copied() {
        Some(b'{') => looks_like_json_object_start(trimmed),
        Some(b'[') => looks_like_json_array_start(trimmed),
        Some(b'"') => is_json_string_document(trimmed),
        Some(b't') => is_exact_json_literal(trimmed, "true"),
        Some(b'f') => is_exact_json_literal(trimmed, "false"),
        Some(b'n') => is_exact_json_literal(trimmed, "null"),
        Some(b'-' | b'0'..=b'9') => is_json_number_document(trimmed),
        _ => false,
    }
}

fn trim_json_whitespace_start(raw: &str) -> &str {
    let first_non_ws = raw
        .as_bytes()
        .iter()
        .position(|byte| !is_json_whitespace(*byte))
        .unwrap_or(raw.len());
    &raw[first_non_ws..]
}

fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

fn has_only_json_whitespace(raw: &str) -> bool {
    raw.as_bytes().iter().all(|byte| is_json_whitespace(*byte))
}

fn looks_like_json_object_start(raw: &str) -> bool {
    let after_open = trim_json_whitespace_start(&raw[1..]);
    matches!(after_open.as_bytes().first().copied(), Some(b'}' | b'"'))
}

fn looks_like_json_array_start(raw: &str) -> bool {
    let after_open = trim_json_whitespace_start(&raw[1..]);
    matches!(
        after_open.as_bytes().first().copied(),
        Some(b']' | b'{' | b'[' | b'"' | b'-' | b'0'..=b'9' | b't' | b'f' | b'n')
    )
}

fn is_exact_json_literal(raw: &str, literal: &str) -> bool {
    raw.strip_prefix(literal)
        .is_some_and(has_only_json_whitespace)
}

fn is_json_string_document(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    let mut idx = 1;
    while idx < bytes.len() {
        match bytes[idx] {
            b'"' => return has_only_json_whitespace(&raw[idx + 1..]),
            b'\\' => {
                idx += 1;
                let Some(escaped) = bytes.get(idx).copied() else {
                    return false;
                };
                if !matches!(
                    escaped,
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                ) {
                    return false;
                }
                if escaped == b'u' {
                    let Some(hex) = bytes.get(idx + 1..idx + 5) else {
                        return false;
                    };
                    if !hex.iter().all(u8::is_ascii_hexdigit) {
                        return false;
                    }
                    idx += 4;
                }
            }
            0x00..=0x1f => return false,
            _ => {}
        }
        idx += 1;
    }
    false
}

fn is_json_number_document(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    let mut idx = 0;

    if matches!(bytes.get(idx), Some(b'-')) {
        idx += 1;
    }

    match bytes.get(idx).copied() {
        Some(b'0') => idx += 1,
        Some(b'1'..=b'9') => {
            idx += 1;
            while matches!(bytes.get(idx), Some(b'0'..=b'9')) {
                idx += 1;
            }
        }
        _ => return false,
    }

    if matches!(bytes.get(idx), Some(b'.')) {
        idx += 1;
        let start = idx;
        while matches!(bytes.get(idx), Some(b'0'..=b'9')) {
            idx += 1;
        }
        if idx == start {
            return false;
        }
    }

    if matches!(bytes.get(idx), Some(b'e' | b'E')) {
        idx += 1;
        if matches!(bytes.get(idx), Some(b'+' | b'-')) {
            idx += 1;
        }
        let start = idx;
        while matches!(bytes.get(idx), Some(b'0'..=b'9')) {
            idx += 1;
        }
        if idx == start {
            return false;
        }
    }

    has_only_json_whitespace(&raw[idx..])
}

/// Resolve short-form aliases (`read`, `write`, ...) to the long-form name
/// used by the match arms in this module. `PascalCase` and `snake_case` spellings
/// pass through unchanged so existing arms still fire.
fn canonical_preview_alias(name: &str) -> &str {
    match name {
        "read" => "read_file",
        "write" => "write_file",
        "edit" => "edit_file",
        "glob" => "glob_search",
        "grep" => "grep_search",
        other => other,
    }
}

// ============================================================================
// Individual formatters
// ============================================================================

/// Bash result formatter. Pure; returns the structured
/// [`BashResult`] the TUI expects.
#[must_use]
pub fn format_bash_result(value: &Value) -> BashResult {
    let exit_code = value
        .get("exit_code")
        .or_else(|| value.get("exitCode"))
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let (stdout, stdout_truncated) = truncate(
        value
            .get("stdout")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        TOOL_OUTPUT_DISPLAY_MAX_LINES,
        TOOL_OUTPUT_DISPLAY_MAX_CHARS,
    );
    let (stderr, stderr_truncated) = truncate(
        value
            .get("stderr")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        TOOL_OUTPUT_DISPLAY_MAX_LINES,
        TOOL_OUTPUT_DISPLAY_MAX_CHARS,
    );
    BashResult {
        exit_code,
        stdout,
        stderr,
        truncated: stdout_truncated || stderr_truncated,
    }
}

/// File read result formatter.
#[must_use]
pub fn format_read_result(value: &Value) -> ToolResultBody {
    let file = value.get("file").unwrap_or(value);
    let path = extract_tool_path(file);
    let content = file
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let (content, truncated) = truncate(content, READ_DISPLAY_MAX_LINES, READ_DISPLAY_MAX_CHARS);
    let language = detect_language(&path);
    ToolResultBody::Read {
        path,
        content,
        language,
        truncated,
    }
}

/// Write result formatter.
#[must_use]
pub fn format_write_result(name: &str, value: &Value) -> ToolResultBody {
    let path = extract_tool_path(value);
    let line_count = value
        .get("content")
        .and_then(Value::as_str)
        .map_or(0, |content| content.lines().count());
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("write");
    let verb = if kind == "create" { "Wrote" } else { "Updated" };
    ToolResultBody::Generic {
        name: name.to_string(),
        content: format!("{verb} {path} ({line_count} lines)"),
        truncated: false,
    }
}

/// Edit result formatter. Parses `structuredPatch` when present.
#[must_use]
pub fn format_edit_result(value: &Value) -> ToolResultBody {
    let path = extract_tool_path(value);
    if let Some(view) = parse_structured_patch(&path, value) {
        return ToolResultBody::Diff(view);
    }
    let old_value = value
        .get("oldString")
        .or_else(|| value.get("old_string"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let new_value = value
        .get("newString")
        .or_else(|| value.get("new_string"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    // Guard against the phantom `Edit: ? · +0 -0` block. When neither a
    // `structuredPatch` nor a non-empty `oldString`/`newString` is present —
    // the result payload was a plain string (e.g. an upstream JSON parse/
    // truncation failure) or an object missing those fields — fabricating a
    // `Diff` body yields an empty hunk with no path: the TUI then renders an
    // "Edit" header with `?` for the path and `+0 -0` counts even though no
    // diff exists. Fall back to a readable generic line instead, surfacing the
    // raw payload so the real outcome (or error) stays visible.
    if old_value.is_empty() && new_value.is_empty() {
        return ToolResultBody::Generic {
            name: "edit_file".to_string(),
            content: value_to_text(value),
            truncated: false,
        };
    }

    ToolResultBody::Diff(DiffView {
        old_path: Some(path.clone()),
        new_path: Some(path.clone()),
        language: detect_language(&path),
        hunks: diff_hunks_from_replace(old_value, new_value),
    })
}

/// Glob result formatter — structured listing.
#[must_use]
pub fn format_glob_result(value: &Value) -> ToolResultBody {
    let filenames = value
        .get("filenames")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let num_files = value
        .get("numFiles")
        .and_then(Value::as_u64)
        .unwrap_or(filenames.len() as u64) as usize;
    let truncated = filenames.len() < num_files;
    ToolResultBody::Listing {
        entries: filenames,
        truncated,
    }
}

/// Grep result formatter — structured listing.
#[must_use]
pub fn format_grep_result(value: &Value) -> ToolResultBody {
    if let Some(content) = value.get("content").and_then(Value::as_str) {
        if !content.trim().is_empty() {
            let (content, truncated) = truncate(
                content,
                TOOL_OUTPUT_DISPLAY_MAX_LINES,
                TOOL_OUTPUT_DISPLAY_MAX_CHARS,
            );
            return ToolResultBody::Text { content, truncated };
        }
    }
    let entries = value
        .get("filenames")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    ToolResultBody::Listing {
        entries,
        truncated: false,
    }
}

/// Fallback result formatter for tools we don't specialise on.
#[must_use]
pub fn format_generic_result(name: &str, value: &Value) -> ToolResultBody {
    let rendered = value_to_text(value);
    let (content, truncated) = truncate(
        &rendered,
        TOOL_OUTPUT_DISPLAY_MAX_LINES,
        TOOL_OUTPUT_DISPLAY_MAX_CHARS,
    );
    ToolResultBody::Generic {
        name: name.to_string(),
        content,
        truncated,
    }
}

/// One-line collapsed summary of a tool *input* preview, used as the
/// `RenderBlock::ToolCall::summary` header. Lives here next to
/// [`preview_tool_input`] so the live stream path (`parser.rs`) and the
/// resume path (`seed_transcript_from_session`) build identical headers.
#[must_use]
pub fn preview_summary(preview: &ToolPreview) -> String {
    match preview {
        ToolPreview::Bash { command } => truncate_one_line(command, 80),
        ToolPreview::Read { path, .. } => format!("read {path}"),
        ToolPreview::Write { path, byte_count } => format!("write {path} ({byte_count} bytes)"),
        ToolPreview::Edit { path, hunk_count } => format!("edit {path} ({hunk_count} hunks)"),
        ToolPreview::Glob { pattern } => format!("glob {pattern}"),
        ToolPreview::Grep { pattern, path } => match path {
            Some(p) => format!("grep {pattern} in {p}"),
            None => format!("grep {pattern}"),
        },
        ToolPreview::Search { query } => format!("search {query}"),
        ToolPreview::Generic {
            name,
            input_summary,
        } => format!("{name} {input_summary}"),
    }
}

fn truncate_one_line(text: &str, limit: usize) -> String {
    let first = text.lines().next().unwrap_or(text);
    if first.chars().count() <= limit {
        first.to_string()
    } else {
        let mut out: String = first.chars().take(limit).collect();
        out.push('\u{2026}');
        out
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn extract_tool_path(value: &Value) -> String {
    value
        .get("file_path")
        .or_else(|| value.get("filePath"))
        .or_else(|| value.get("path"))
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string()
}

fn extract_str(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn extract_read_range(value: &Value) -> Option<(u64, u64)> {
    // Result-shape / native `Read` params: a 1-based start line plus a line
    // count.
    if let (Some(start), Some(num)) = (
        value.get("startLine").and_then(Value::as_u64),
        value.get("numLines").and_then(Value::as_u64),
    ) {
        return Some((start, start.saturating_add(num.saturating_sub(1)).max(start)));
    }
    // Zo `read_file` *input* params: a 0-based `offset` and a `limit` line
    // count. A windowed read carries both, so two windows of one file (a model
    // paging a large file 80 lines at a time) get distinct `(start, end)` ranges
    // instead of collapsing to the bare path — the "read the same file 4 times"
    // illusion in the collapsed tool group. An offset-only tail read has no
    // computable end, so it stays rangeless (the whole tail; the path alone is
    // honest).
    if let Some(limit) = value.get("limit").and_then(Value::as_u64) {
        let start = value
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(1);
        return Some((start, start.saturating_add(limit.saturating_sub(1)).max(start)));
    }
    None
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        Value::Object(_) | Value::Array(_) => {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        }
        _ => value.to_string(),
    }
}

fn summarize_json(value: &Value, limit: usize) -> String {
    let compact = match value {
        Value::String(s) => s.clone(),
        _ => value.to_string(),
    };
    if compact.chars().count() <= limit {
        compact
    } else {
        let mut out: String = compact.chars().take(limit).collect();
        out.push('…');
        out
    }
}

fn truncate(text: &str, max_lines: usize, max_chars: usize) -> (String, bool) {
    let mut lines = text.lines();
    let mut kept_lines = Vec::with_capacity(max_lines.min(16));
    for _ in 0..max_lines {
        let Some(line) = lines.next() else {
            break;
        };
        kept_lines.push(line);
    }
    let line_truncated = lines.next().is_some();
    let mut joined = kept_lines.join("\n");
    let mut char_truncated = false;
    // Cap at `max_chars` *characters*, cutting on a UTF-8 boundary. The old
    // `joined.truncate(max_chars)` used a byte index and panicked whenever the
    // cut landed inside a multi-byte char (한글/CJK/emoji) —
    // `is_char_boundary` assertion. `char_indices().nth(max_chars)` gives the
    // byte offset of the (max_chars+1)th char (a valid boundary) and only scans
    // up to the limit, not the whole string.
    if let Some((byte_idx, _)) = joined.char_indices().nth(max_chars) {
        joined.truncate(byte_idx);
        char_truncated = true;
    }
    (joined, line_truncated || char_truncated)
}

fn detect_language(path: &str) -> Option<String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)?;
    Some(
        match ext {
            "rs" => "rust",
            "ts" | "tsx" => "typescript",
            "js" | "jsx" => "javascript",
            "py" => "python",
            "go" => "go",
            "md" => "markdown",
            "toml" => "toml",
            "json" => "json",
            "sh" => "bash",
            other => other,
        }
        .to_string(),
    )
}

fn parse_structured_patch(path: &str, value: &Value) -> Option<DiffView> {
    let hunks_json = value
        .get("structuredPatch")
        .or_else(|| value.get("structured_patch"))?
        .as_array()?;
    let mut hunks = Vec::with_capacity(hunks_json.len());
    for hunk in hunks_json {
        let old_start = hunk.get("oldStart").and_then(Value::as_u64).unwrap_or(1) as u32;
        let old_lines = hunk.get("oldLines").and_then(Value::as_u64).unwrap_or(0) as u32;
        let new_start = hunk.get("newStart").and_then(Value::as_u64).unwrap_or(1) as u32;
        let new_lines = hunk.get("newLines").and_then(Value::as_u64).unwrap_or(0) as u32;
        let lines = hunk
            .get("lines")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(classify_diff_line)
                    .collect()
            })
            .unwrap_or_default();
        hunks.push(DiffHunk {
            old_start,
            old_lines,
            new_start,
            new_lines,
            lines,
        });
    }
    if hunks.is_empty() {
        return None;
    }
    Some(DiffView {
        old_path: Some(path.to_string()),
        new_path: Some(path.to_string()),
        language: detect_language(path),
        hunks,
    })
}

fn classify_diff_line(raw: &str) -> DiffLine {
    let mut chars = raw.chars();
    match chars.next() {
        Some('+') => DiffLine {
            kind: DiffLineKind::Added,
            text: chars.as_str().to_string(),
        },
        Some('-') => DiffLine {
            kind: DiffLineKind::Removed,
            text: chars.as_str().to_string(),
        },
        Some(' ') => DiffLine {
            kind: DiffLineKind::Context,
            text: chars.as_str().to_string(),
        },
        _ => DiffLine {
            kind: DiffLineKind::Context,
            text: raw.to_string(),
        },
    }
}

fn diff_hunks_from_replace(old_value: &str, new_value: &str) -> Vec<DiffHunk> {
    compact_line_diff(old_value, new_value)
        .into_iter()
        .map(|hunk| DiffHunk {
            old_start: hunk.old_start as u32,
            old_lines: hunk.old_lines as u32,
            new_start: hunk.new_start as u32,
            new_lines: hunk.new_lines as u32,
            lines: hunk
                .lines
                .into_iter()
                .map(|line| DiffLine {
                    kind: match line.kind {
                        CompactDiffLineKind::Context => DiffLineKind::Context,
                        CompactDiffLineKind::Removed => DiffLineKind::Removed,
                        CompactDiffLineKind::Added => DiffLineKind::Added,
                    },
                    text: line.text,
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Lane A — `PascalCase`, short-form, and `snake_case` spellings must all
    /// produce the same typed [`ToolPreview`] variant, so the dispatcher in
    /// the `tools` crate can canonicalize freely without breaking the
    /// renderer.
    #[test]
    fn preview_tool_input_accepts_read_aliases_for_same_variant() {
        let input = json!({ "path": "/tmp/example.txt" });
        let via_pascal = preview_tool_input("Read", &input);
        let via_short = preview_tool_input("read", &input);
        let via_snake = preview_tool_input("read_file", &input);
        for preview in [&via_pascal, &via_short, &via_snake] {
            assert!(
                matches!(preview, ToolPreview::Read { .. }),
                "expected ToolPreview::Read, got {preview:?}"
            );
        }
    }

    #[test]
    fn read_range_maps_offset_limit_so_windowed_reads_are_distinguishable() {
        // Zo's `read_file` carries a 0-based `offset` + `limit` line count. The
        // preview must surface a 1-based `(start, end)` window so the collapsed
        // tool group can show `path:start-end` — two windows of one file no longer
        // collapse to identical rows.
        let win = preview_tool_input(
            "read_file",
            &json!({ "path": "src/App.tsx", "offset": 80, "limit": 80 }),
        );
        assert!(
            matches!(&win, ToolPreview::Read { range: Some((81, 160)), .. }),
            "offset 80 + limit 80 → 1-based lines 81-160, got {win:?}"
        );
        // limit with no offset starts at line 1.
        let head = preview_tool_input(
            "read_file",
            &json!({ "path": "a.rs", "offset": 0, "limit": 80 }),
        );
        assert!(
            matches!(&head, ToolPreview::Read { range: Some((1, 80)), .. }),
            "offset 0 + limit 80 → lines 1-80, got {head:?}"
        );
        // A whole-file read (no window) stays rangeless — the bare path is honest.
        let whole = preview_tool_input("read_file", &json!({ "path": "a.rs" }));
        assert!(
            matches!(&whole, ToolPreview::Read { range: None, .. }),
            "a rangeless read carries no window, got {whole:?}"
        );
        // The result-shape `startLine`/`numLines` path still works (resume seed).
        let result_shape = preview_tool_input(
            "Read",
            &json!({ "path": "a.rs", "startLine": 5, "numLines": 10 }),
        );
        assert!(
            matches!(&result_shape, ToolPreview::Read { range: Some((5, 14)), .. }),
            "startLine 5 + numLines 10 → lines 5-14, got {result_shape:?}"
        );
    }

    #[test]
    fn truncate_cuts_multibyte_text_on_a_char_boundary_without_panicking() {
        // Regression: a tool result of 한글 (3 bytes/char) whose char-cap lands
        // mid-character used to panic in `String::truncate` (is_char_boundary).
        let text = "한".repeat(5_000); // 5_000 chars, 15_000 bytes
        let (out, truncated) = truncate(&text, 80, 4_000);
        assert!(truncated, "over the cap, so flagged truncated");
        assert_eq!(
            out.chars().count(),
            4_000,
            "capped at max_chars *characters*"
        );
        assert!(
            out.chars().all(|c| c == '한'),
            "no split/garbled char at the cut"
        );

        // Under the cap is returned untouched.
        let short = "가나다".to_string();
        let (out, truncated) = truncate(&short, 80, 4_000);
        assert!(!truncated);
        assert_eq!(out, "가나다");
    }

    #[test]
    fn preview_tool_input_accepts_bash_and_write_aliases() {
        let bash = json!({ "command": "echo hi" });
        assert!(matches!(
            preview_tool_input("Bash", &bash),
            ToolPreview::Bash { .. }
        ));
        assert!(matches!(
            preview_tool_input("bash", &bash),
            ToolPreview::Bash { .. }
        ));

        let write = json!({ "path": "/tmp/x", "content": "hi" });
        for name in ["Write", "write", "write_file"] {
            assert!(
                matches!(preview_tool_input(name, &write), ToolPreview::Write { .. }),
                "`{name}` should preview as ToolPreview::Write"
            );
        }
    }

    /// Regression for the phantom `Edit: ? · +0 -0` block: a result payload
    /// with no `structuredPatch` and no `oldString`/`newString` (e.g. an
    /// upstream JSON parse failure delivered the body as a bare string, or an
    /// object missing those fields) must NOT fabricate an empty `Diff`. It
    /// renders as a readable `Generic` instead.
    #[test]
    fn format_edit_result_does_not_fabricate_phantom_diff() {
        // String payload (JSON parse failed upstream).
        let as_string = Value::String("auto-format ran; no structured diff".to_string());
        assert!(
            matches!(
                format_edit_result(&as_string),
                ToolResultBody::Generic { .. }
            ),
            "string payload must not become a phantom Diff"
        );

        // Object missing the diff-bearing fields entirely.
        let bare = json!({ "ok": true });
        assert!(
            matches!(format_edit_result(&bare), ToolResultBody::Generic { .. }),
            "fieldless object must not become a phantom Diff"
        );

        // A real edit (oldString/newString present) still renders a Diff.
        let real = json!({
            "filePath": "/tmp/a.rs",
            "oldString": "let x = 0;",
            "newString": "let x = 1;"
        });
        assert!(
            matches!(format_edit_result(&real), ToolResultBody::Diff(_)),
            "a real edit must still render a Diff"
        );
    }

    #[test]
    fn format_tool_result_accepts_grep_and_glob_aliases() {
        let glob = json!({ "filenames": ["a.rs", "b.rs"], "numFiles": 2 });
        for name in ["Glob", "glob", "glob_search"] {
            assert!(
                matches!(
                    format_tool_result(name, &glob, false),
                    ToolResultBody::Listing { .. }
                ),
                "`{name}` should format as ToolResultBody::Listing"
            );
        }

        let grep = json!({ "filenames": ["a.rs"] });
        for name in ["Grep", "grep", "grep_search"] {
            assert!(
                matches!(
                    format_tool_result(name, &grep, false),
                    ToolResultBody::Listing { .. }
                ),
                "`{name}` should format as ToolResultBody::Listing"
            );
        }
    }

    #[test]
    fn format_tool_result_preserves_ask_user_question_name_on_error() {
        let body = format_tool_result(
            "AskUserQuestion",
            &Value::String("User question dismissed without answer".to_string()),
            true,
        );

        match body {
            ToolResultBody::Generic {
                name,
                content,
                truncated,
            } => {
                assert_eq!(name, "AskUserQuestion");
                assert_eq!(content, "User question dismissed without answer");
                assert!(!truncated);
            }
            other => panic!("AskUserQuestion errors need a named generic body, got {other:?}"),
        }
    }

    #[test]
    fn format_tool_result_from_raw_uses_plain_text_fast_path_for_generic_output() {
        let raw = "plain grep-like output that is not json\n".repeat(200);
        assert!(!might_be_json_value(&raw));

        let body = format_tool_result_from_raw("CustomTool", &raw, false);
        match body {
            ToolResultBody::Generic {
                name,
                content,
                truncated,
            } => {
                assert_eq!(name, "CustomTool");
                assert!(content.starts_with("plain grep-like output"));
                assert!(
                    truncated,
                    "large generic text should still be display-capped"
                );
            }
            other => panic!("generic raw output should render as Generic, got {other:?}"),
        }
    }

    #[test]
    fn raw_json_candidate_rejects_jsonish_plain_text_prefixes() {
        for raw in [
            "false positive tool output\n".repeat(200),
            "found 12 files\n".repeat(200),
            "null device warning\n".repeat(200),
            "2026-03-31 log line\n".repeat(200),
            "- warning: not a JSON number\n".repeat(200),
            "\"unterminated quoted output\n".repeat(200),
            "[INFO] formatter output\n".repeat(200),
            "{not actually json}\n".repeat(200),
        ] {
            assert!(
                !might_be_json_value(&raw),
                "jsonish plain text should not enter serde_json parsing: {raw:?}"
            );
        }
    }

    #[test]
    fn raw_json_candidate_accepts_complete_json_scalars() {
        for raw in [
            "true",
            " false \n",
            "null\t",
            "0",
            "-12.5e+3\r\n",
            r#""quoted string""#,
            r#""escaped \" string""#,
        ] {
            assert!(
                might_be_json_value(raw),
                "valid JSON scalar should parse: {raw}"
            );
        }

        for raw in [
            "01",
            "1.",
            "1e",
            "true-ish",
            "null device",
            r#""bad \q escape""#,
        ] {
            assert!(
                !might_be_json_value(raw),
                "invalid scalar should be rejected: {raw}"
            );
        }
    }

    #[test]
    fn raw_error_results_are_display_capped() {
        let raw = "plain failure ".repeat(1_000);
        let body = format_tool_result_from_raw("Bash", &raw, true);

        match body {
            ToolResultBody::Text { content, truncated } => {
                assert!(truncated, "large plain errors should be display-capped");
                assert_eq!(content.chars().count(), TOOL_OUTPUT_DISPLAY_MAX_CHARS);
            }
            other => panic!("expected text error body, got {other:?}"),
        }
    }

    #[test]
    fn structured_error_results_are_display_capped_but_keep_question_name() {
        let body = format_tool_result(
            "AskUserQuestion",
            &Value::String("dismissed ".repeat(1_000)),
            true,
        );

        match body {
            ToolResultBody::Generic {
                name,
                content,
                truncated,
            } => {
                assert_eq!(name, "AskUserQuestion");
                assert!(
                    truncated,
                    "large structured errors should be display-capped"
                );
                assert_eq!(content.chars().count(), TOOL_OUTPUT_DISPLAY_MAX_CHARS);
            }
            other => panic!("expected named generic question error body, got {other:?}"),
        }
    }

    #[test]
    fn format_tool_result_from_raw_keeps_structured_json_rendering() {
        let raw = r#"{"filePath":"/tmp/a.rs","structuredPatch":[{"oldStart":1,"oldLines":1,"newStart":1,"newLines":1,"lines":["-old","+new"]}]}"#;
        assert!(might_be_json_value(raw));
        assert!(matches!(
            format_tool_result_from_raw("Edit", raw, false),
            ToolResultBody::Diff(_)
        ));
    }

    #[test]
    fn format_tool_result_from_raw_preserves_plain_error_shape() {
        let body = format_tool_result_from_raw("Bash", "plain failure", true);
        assert!(matches!(
            body,
            ToolResultBody::Text {
                ref content,
                truncated: false
            } if content == "plain failure"
        ));
    }

    #[test]
    fn todowrite_result_parses_into_typed_todos_body() {
        let raw = r#"{"oldTodos":[],"newTodos":[
            {"content":"Wire it","activeForm":"Wiring it","status":"completed"},
            {"content":"Render it","activeForm":"Rendering it","status":"in_progress"},
            {"content":"Test it","activeForm":"Testing it","status":"pending"}
        ],"verificationNudgeNeeded":null}"#;
        let body = format_tool_result_from_raw("TodoWrite", raw, false);
        let ToolResultBody::Todos(items) = body else {
            panic!("TodoWrite must format as a typed Todos body, got {body:?}");
        };
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].status, TodoResultStatus::Completed);
        assert_eq!(items[1].status, TodoResultStatus::InProgress);
        assert_eq!(items[1].active_form, "Rendering it");
        assert_eq!(items[2].status, TodoResultStatus::Pending);
    }

    #[test]
    fn todowrite_empty_result_is_still_typed_todos() {
        let raw = r#"{"oldTodos":[],"newTodos":[],"verificationNudgeNeeded":null}"#;
        let body = format_tool_result_from_raw("TodoWrite", raw, false);
        assert!(
            matches!(body, ToolResultBody::Todos(ref items) if items.is_empty()),
            "an all-cleared TodoWrite still gets the typed (empty) Todos body: {body:?}"
        );
    }
}
