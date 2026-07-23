//! Tool execution pipeline.
//!
//! Three layers of indirection collapse into one entry point:
//!
//! ```text
//! execute_tool / GlobalToolRegistry::execute
//!     └── execute_tool_with_context
//!             ├── dispatch_tool_inner           (per-family fan-out)
//!             ├── maybe_enrich_file_tool_result (auto-format + LSP diagnostics)
//!             └── runtime::truncate_tool_output (per-tool size cap)
//! ```
//!
//! Helpers shared by every `*_tools.rs` submodule (`from_value`,
//! `to_pretty_json`, `maybe_enforce_permission_check`) live here too so the
//! root namespace stays uncluttered.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use runtime::{
    context_compression::compress_tool_output,
    permission_enforcer::{EnforcementResult, PermissionEnforcer},
    truncate_tool_output, TruncationConfig,
};
use serde::Deserialize;
use serde_json::Value;

use crate::aliases::canonical_tool_name;
use crate::context::{disabled_tool_error, ToolContext, TOOL_TOGGLE_DENIAL_REASON};
use crate::error::ToolError;
use crate::gateway::{
    self, failed_result, successful_result, toggle_denied_decision, ToolResultMetadata,
};
use crate::{
    bash_tools, codegraph_tools, file_tools, mcp_tools, misc_tools, plan_mode_v2, task_tools,
    team_tools, typed_actions, web_tools, worker_tools, workflow_tools, worktree_tools,
};

type BuiltinDispatcher = fn(
    &ToolContext,
    Option<&PermissionEnforcer>,
    &str,
    &Value,
) -> Option<Result<String, ToolError>>;

const BUILTIN_DISPATCHERS: &[BuiltinDispatcher] = &[
    bash_tools::dispatch,
    codegraph_tools::dispatch,
    typed_actions::dispatch,
    file_tools::dispatch,
    web_tools::dispatch,
    task_tools::dispatch,
    worker_tools::dispatch,
    team_tools::dispatch,
    mcp_tools::dispatch,
    misc_tools::dispatch,
    dispatch_worktree_family,
    dispatch_plan_mode_v2_family,
    dispatch_workflow_family,
];

/// Check permission before executing a tool. Returns Err with denial reason if blocked.
pub fn enforce_permission_check(
    enforcer: &PermissionEnforcer,
    tool_name: &str,
    input: &Value,
) -> Result<(), ToolError> {
    // Command-intent gate first for the bash tool: a command the shared
    // `bash_validation` pipeline forbids under the active mode (a write or
    // `sed -i` under read-only) fails here with its command-specific reason
    // instead of the generic mode-ladder denial below. Ordering is safe —
    // both gates must pass before execution, and deny rules still apply in
    // the generic gate.
    if tool_name == "bash" {
        if let Some(command) = input.get("command").and_then(Value::as_str) {
            if let EnforcementResult::Denied {
                active_mode,
                required_mode,
                reason,
                ..
            } = enforcer.check_bash(command)
            {
                return Err(ToolError::PermissionDenied {
                    tool: tool_name.to_owned(),
                    reason: permission_denial_reason(reason, &active_mode, &required_mode),
                });
            }
        }
    }

    let input_str = serde_json::to_string(input).unwrap_or_default();
    if let EnforcementResult::Denied {
        active_mode,
        required_mode,
        reason,
        ..
    } = enforcer.check(tool_name, &input_str)
    {
        return Err(ToolError::PermissionDenied {
            tool: tool_name.to_owned(),
            reason: permission_denial_reason(reason, &active_mode, &required_mode),
        });
    }

    Ok(())
}

fn permission_denial_reason(reason: String, active_mode: &str, required_mode: &str) -> String {
    if reason.contains("Permission audit:") {
        return reason;
    }

    format!(
        "{reason}. Permission audit: active mode is {active_mode}; required mode is {required_mode}. \
         This denial is mode-based and deterministic — the identical call will be denied again, \
         so do not retry it. Continue with tools allowed under {active_mode} mode, or ask the \
         user to escalate (TUI: `/permissions {required_mode}`, or restart with \
         `--permission-mode {required_mode}`)."
    )
}

/// Execute a tool using an externally-provided context. Prefer `GlobalToolRegistry::execute`.
pub fn execute_tool(ctx: &ToolContext, name: &str, input: &Value) -> Result<String, ToolError> {
    execute_tool_with_context(ctx, None, name, input)
}

/// Global default truncation config used for all tool dispatch results.
fn global_truncation_config() -> &'static TruncationConfig {
    static CONFIG: OnceLock<TruncationConfig> = OnceLock::new();
    CONFIG.get_or_init(TruncationConfig::default)
}

pub(crate) fn execute_tool_with_context(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Result<String, ToolError> {
    execute_tool_with_context_and_artifact_dir(ctx, enforcer, name, input, None)
}

fn execute_tool_with_context_and_artifact_dir(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
    artifact_dir: Option<&Path>,
) -> Result<String, ToolError> {
    let canonical = canonical_tool_name(name);
    let canonical_ref = canonical.as_str();
    let invocation = gateway::begin_tool_invocation(name, canonical_ref, input, enforcer);
    if ctx.is_tool_disabled(canonical_ref) {
        let error = disabled_tool_error(canonical_ref);
        ctx.record_tool_invocation(
            invocation
                .with_policy_decision(toggle_denied_decision(TOOL_TOGGLE_DENIAL_REASON))
                .finish(failed_result(&error), gateway::epoch_millis_now()),
        );
        return Err(error);
    }
    let raw = match dispatch_tool_inner(ctx, enforcer, canonical_ref, input) {
        Ok(raw) => raw,
        Err(error) => {
            ctx.record_tool_invocation(
                invocation.finish(failed_result(&error), gateway::epoch_millis_now()),
            );
            return Err(error);
        }
    };

    Ok(finish_successful_tool_output(
        ctx,
        invocation,
        canonical_ref,
        input,
        raw,
        artifact_dir,
    ))
}

pub(crate) fn finish_successful_tool_output(
    ctx: &ToolContext,
    invocation: gateway::ToolInvocationStart,
    canonical_ref: &str,
    input: &Value,
    raw: String,
    artifact_dir: Option<&Path>,
) -> String {
    let enriched = maybe_enrich_file_tool_result(ctx, canonical_ref, input, raw);

    // Compute the model-facing compressed/outline view first: it is what
    // bounds the wire size for JSON-envelope tools (read_file) whose raw
    // envelope must stay valid JSON and therefore cannot be char-truncated.
    let compressed = compress_tool_output(
        &enriched,
        canonical_ref,
        ctx.workspace_root.as_deref().or(ctx.cwd.as_deref()),
    );
    let truncated = truncate_tool_output(&enriched, canonical_ref, global_truncation_config());
    // A JSON-envelope tool (read_file) is exempt from raw-envelope truncation,
    // so `was_truncated` is always false for it even when the output is huge.
    // We must still preserve the full bytes as an artifact in that case so the
    // model can recover them via `retrieve_tool_output`. Treat any output that
    // exceeds the global ceiling as "oversized" for artifact purposes, even
    // when the structural rewrite did not happen to save enough to flip
    // `was_compressed`.
    let oversized = truncated.original_len > global_truncation_config().default_max_chars;
    // Phase-4: transformed output keeps its full pre-transform content as a
    // content-addressed artifact (best-effort) so it stays recoverable without
    // rendering raw bytes without bound into the transcript. The `None`
    // override resolves the global `.zo/artifacts` store; tests inject a dir.
    let artifact = crate::artifacts::store_transformed(
        artifact_dir,
        &enriched,
        compressed.was_compressed,
        truncated.was_truncated || oversized,
    );
    let mut content = truncated.content;
    // CCR retrieve half: tell the model how to get the cut bytes back. Only
    // when the artifact actually persisted — a store failure must not
    // advertise an id that cannot be resolved. For compressed-only artifacts we
    // keep JSON outputs parseable by embedding the same notice as a structured
    // field; the model-facing compression seam extracts and appends it after
    // rewriting the body.
    if let Some(artifact_ref) = &artifact {
        let notice = recovery_notice(&artifact_ref.sha256);
        if truncated.was_truncated {
            content.push('\n');
            content.push_str(&notice);
        } else {
            content = embed_recovery_notice(content, &notice);
        }
    }
    let returned_chars = content.chars().count();
    let metadata = ToolResultMetadata {
        output_chars: truncated.original_len.max(returned_chars),
        returned_chars,
        truncated: truncated.was_truncated,
        artifact,
    };
    ctx.record_tool_invocation(
        invocation.finish(successful_result(metadata), gateway::epoch_millis_now()),
    );
    content
}

fn recovery_notice(sha256: &str) -> String {
    format!(
        "[full output preserved — call retrieve_tool_output {{\"sha256\": \"{sha256}\"}}; \
         window large outputs with offset/limit (0-based lines)]"
    )
}

fn embed_recovery_notice(content: String, notice: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(serde_json::Value::Object(mut map)) => {
            map.insert(
                "recoveryNotice".to_string(),
                serde_json::Value::String(notice.to_string()),
            );
            serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or(content)
        }
        _ => format!("{content}\n{notice}"),
    }
}

/// Wall-clock budget for in-place auto-formatting in
/// [`maybe_enrich_file_tool_result`]. `runtime::auto_format` owns the child
/// process and kills it on timeout, so a misconfigured or wedged formatter does
/// not leak a worker thread/subprocess or stall every write/edit indefinitely.
/// 5 s is generous for any single-file format.
const AUTO_FORMAT_BUDGET_MS: u64 = 5_000;

/// Whether any instrumentation probe is currently staged in the context. Used
/// to suppress in-place auto-formatting that would corrupt a debugger's probe
/// snippet before it can be reverted. Only the debugger sub-agent stages probes,
/// so this is always false for the main session.
fn file_has_staged_probe(ctx: &ToolContext) -> bool {
    !ctx.probe_sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_empty()
}

/// Resolve the file that post-write enrichment should inspect/format.
///
/// Keep this path calculation in terms of the original tool input instead of
/// reparsing the raw JSON result: `write_file` can return large `content`, and
/// the enrichment hot path only needs the target path. The precedence mirrors
/// `file_tools`: an absolute path is already anchored; otherwise a configured
/// workspace root wins, then an execution cwd, then the historical process-cwd
/// relative path.
fn enrichment_target_path(ctx: &ToolContext, input: &serde_json::Value) -> Option<PathBuf> {
    let input_path = input.get("path").and_then(|v| v.as_str())?;
    Some(resolve_file_tool_path(ctx, Path::new(input_path)))
}

fn resolve_file_tool_path(ctx: &ToolContext, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return file_tools::resolve_for_boundary_check(path);
    }

    if let Some(root) = ctx.workspace_root.as_deref() {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        return file_tools::resolve_for_boundary_check(&root.join(path));
    }

    if let Some(cwd) = ctx.cwd.as_deref() {
        return file_tools::resolve_for_boundary_check(&cwd.join(path));
    }

    file_tools::resolve_for_boundary_check(path)
}

fn maybe_enrich_file_tool_result(
    ctx: &ToolContext,
    tool_name: &str,
    input: &serde_json::Value,
    raw_output: String,
) -> String {
    if !matches!(tool_name, "write_file" | "edit_file") {
        return raw_output;
    }
    let Some(path) = enrichment_target_path(ctx, input) else {
        return raw_output;
    };
    let path_lossy = path.to_string_lossy();
    let path = path_lossy.as_ref();

    let mut enrichments = Vec::new();

    // Auto-format rewrites the file in place. While an instrumentation probe is
    // staged (debugger debug-mode), skip it: a reformat (e.g. rustfmt re-indenting
    // or line-splitting a column-0 `/*ZO_PROBE*/` line) would shift the probe's
    // bytes out from under `ToolContext::revert_probes`, which strips the exact
    // inserted snippet — leaving the marker stranded in the diff. Probes are
    // reverted at run end, so formatting resumes after the debug window. Only the
    // debugger sub-agent can stage probes, so the main session is unaffected. LSP
    // diagnostics below only READ the file, so they still run.
    if !file_has_staged_probe(ctx) {
        if let Some(fmt_result) = runtime::auto_format::format_file_with_timeout(
            path,
            Duration::from_millis(AUTO_FORMAT_BUDGET_MS),
        ) {
            enrichments.push(format!("[auto-format] {fmt_result}"));
            // 포맷터가 방금 파일을 재작성했다 — write/edit 성공 시점에 기록된
            // read-registry 스냅샷이 즉시 stale해지므로 여기서 재기록한다.
            // 이 재기록이 없으면 다음 edit_file마다 거짓 "외부 변경" 거부 →
            // 재읽기 강제의 라이브락성 마찰이 생긴다 (CC의 "modified by a
            // linter" 사이클과 같은 클래스). 무변경 포맷이어도 재기록은
            // 멱등이라 무해하다.
            file_tools::record_file_observation(&ctx.file_reads, path);
        }
    }

    if !ctx.lsp.is_empty() {
        let diagnostics = std::fs::read_to_string(path).map_or_else(
            |_| ctx.lsp.get_diagnostics(path),
            |text| {
                ctx.lsp
                    .sync_and_collect_diagnostics(path, &text, Duration::from_millis(500))
            },
        );
        if !diagnostics.is_empty() {
            let diag_lines: Vec<String> = diagnostics
                .iter()
                .take(10)
                .map(|d| {
                    format!(
                        "  {}:{}: [{}] {}",
                        d.line, d.character, d.severity, d.message
                    )
                })
                .collect();
            enrichments.push(format!(
                "--- LSP Diagnostics ({} issue{}) ---\n{}",
                diagnostics.len(),
                if diagnostics.len() == 1 { "" } else { "s" },
                diag_lines.join("\n")
            ));
        }
    }

    fold_enrichment_into_output(raw_output, &enrichments)
}

/// Attach `enrichments` (auto-format / LSP feedback) to a tool's `raw_output`
/// without breaking JSON-shaped results.
///
/// `write_file`/`edit_file` return a JSON object whose fields the TUI
/// formatters read (`filePath`, `content`, `structuredPatch`) to render the
/// diff header and line counts. Appending the text as a raw trailer made the
/// whole string stop being valid JSON, so the renderer fell back to a plain
/// string and drew a phantom `? · +0 -0` even though the edit succeeded. When
/// the output is a JSON object we fold the feedback into a `toolFeedback`
/// field — the output stays parseable for the renderer while the model still
/// sees the formatter/diagnostic text. Non-object output (e.g. an error
/// already shaped as text) falls back to the readable text trailer.
fn fold_enrichment_into_output(raw_output: String, enrichments: &[String]) -> String {
    if enrichments.is_empty() {
        return raw_output;
    }
    let enrichment_text = enrichments.join("\n");
    match serde_json::from_str::<serde_json::Value>(&raw_output) {
        Ok(serde_json::Value::Object(mut map)) => {
            map.insert(
                "toolFeedback".to_string(),
                serde_json::Value::String(enrichment_text),
            );
            serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or(raw_output)
        }
        _ => format!("{raw_output}\n\n{enrichment_text}"),
    }
}

fn dispatch_tool_inner(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Result<String, ToolError> {
    if let Some(mcp) = ctx.mcp_passthrough() {
        if mcp.covers(name) {
            maybe_enforce_permission_check(enforcer, name, input)?;
            return mcp.dispatch(name, input).map_err(ToolError::Execution);
        }
    }

    for dispatch in BUILTIN_DISPATCHERS {
        if let Some(result) = dispatch(ctx, enforcer, name, input) {
            return result;
        }
    }
    Err(ToolError::NotFound(name.to_owned()))
}

fn dispatch_worktree_family(
    _ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    dispatch_worktree(enforcer, name, input)
}

fn dispatch_plan_mode_v2_family(
    _ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    dispatch_plan_mode_v2(enforcer, name, input)
}

fn dispatch_workflow_family(
    ctx: &ToolContext,
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    dispatch_workflow(
        enforcer,
        name,
        input,
        ctx.active_model().as_deref(),
        ctx.active_model_pinned(),
        ctx.hook_config(),
        ctx.session_id().as_deref(),
        ctx.mcp_passthrough(),
        // The spawning session's mode: the enforcer's live mode where one
        // exists, else the foreground mode the TUI records on the context —
        // workflow agents are clamped to it like every other spawn.
        enforcer
            .map(PermissionEnforcer::active_mode)
            .or_else(|| ctx.session_permission_mode()),
    )
}

fn dispatch_worktree(
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    match name {
        "EnterWorktree" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<worktree_tools::EnterWorktreeInput>(input)
                    .and_then(|input| worktree_tools::run_enter_worktree(&input))
            }),
        ),
        "ExitWorktree" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<worktree_tools::ExitWorktreeInput>(input)
                    .and_then(worktree_tools::run_exit_worktree)
            }),
        ),
        _ => None,
    }
}

fn dispatch_plan_mode_v2(
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Option<Result<String, ToolError>> {
    match name {
        "ExitPlanModeV2" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                from_value::<plan_mode_v2::ExitPlanModeV2Input>(input)
                    .and_then(plan_mode_v2::run_exit_plan_mode_v2)
            }),
        ),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_workflow(
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
    parent_model: Option<&str>,
    parent_model_pinned: bool,
    hook_config: &runtime::RuntimeHookConfig,
    parent_session_id: Option<&str>,
    mcp_passthrough: Option<crate::registry::McpPassthrough>,
    parent_permission_mode: Option<runtime::PermissionMode>,
) -> Option<Result<String, ToolError>> {
    match name {
        // `run_workflow` parses the raw value itself (it accepts both
        // `{spec, input}` and a bare spec, and tolerates stringified JSON),
        // so the dispatcher hands the `Value` through untouched.
        "Workflow" => Some(
            maybe_enforce_permission_check(enforcer, name, input).and_then(|()| {
                workflow_tools::run_workflow(
                    input,
                    parent_model,
                    parent_model_pinned,
                    hook_config,
                    parent_session_id,
                    mcp_passthrough,
                    enforcer,
                    parent_permission_mode,
                )
            }),
        ),
        "WorkflowValidate" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| workflow_tools::validate_workflow(input)),
        ),
        "WorkflowLibrary" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| workflow_tools::run_workflow_library(input)),
        ),
        "WorkflowRuns" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| workflow_tools::run_workflow_runs(input)),
        ),
        "WorkflowSkillProject" => Some(
            maybe_enforce_permission_check(enforcer, name, input)
                .and_then(|()| workflow_tools::run_workflow_skill_project(input)),
        ),
        _ => None,
    }
}

pub(crate) fn maybe_enforce_permission_check(
    enforcer: Option<&PermissionEnforcer>,
    tool_name: &str,
    input: &Value,
) -> Result<(), ToolError> {
    if let Some(enforcer) = enforcer {
        enforce_permission_check(enforcer, tool_name, input)?;
    }
    Ok(())
}

pub(crate) fn from_value<T: for<'de> Deserialize<'de>>(input: &Value) -> Result<T, ToolError> {
    serde_json::from_value(input.clone()).map_err(|e| ToolError::InvalidInput(e.to_string()))
}

pub(crate) fn to_pretty_json<T: serde::Serialize>(value: T) -> Result<String, ToolError> {
    Ok(serde_json::to_string_pretty(&value)?)
}

/// Current wall-clock time as a Unix epoch seconds string.
///
/// Renamed from `iso8601_now` (code-review-2026-05): the previous name
/// implied an ISO-8601 timestamp but the returned value was always a
/// raw epoch-seconds string. Callers stamp this into JSON fields like
/// `sentAt` / `scheduledAt` where a stable numeric ordering is all
/// that matters — switching to a true ISO timestamp would change every
/// downstream consumer's parse expectations, so the safer fix is to
/// align the name with the actual format.
pub(crate) fn epoch_seconds_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

#[cfg(test)]
mod auto_format_guard_tests {
    use super::{enrichment_target_path, file_has_staged_probe, maybe_enrich_file_tool_result};
    use crate::context::{Probe, ToolContext};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zo-tools-{name}-{}-{unique}",
            std::process::id()
        ))
    }

    fn stage_probe(ctx: &ToolContext, path: std::path::PathBuf, snippet: &str) {
        ctx.probe_sink
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Probe {
                path,
                snippet: snippet.to_string(),
            });
    }

    #[test]
    fn file_has_staged_probe_reflects_the_sink() {
        let ctx = ToolContext::new();
        assert!(!file_has_staged_probe(&ctx), "empty sink → false");
        stage_probe(&ctx, "x.rs".into(), "\n/*ZO_PROBE:1*/ x");
        assert!(file_has_staged_probe(&ctx), "staged probe → true");
    }

    #[test]
    fn enrichment_target_resolves_relative_path_against_tool_cwd() {
        let cwd = unique_temp_path("enrich-cwd");
        std::fs::create_dir_all(&cwd).expect("create cwd");
        let ctx = ToolContext::new().with_cwd(cwd.clone());
        let input = serde_json::json!({ "path": "src/lib.rs" });

        let target = enrichment_target_path(&ctx, &input).expect("target path");

        assert_eq!(
            target,
            cwd.canonicalize()
                .expect("canonical cwd")
                .join("src/lib.rs"),
            "file-tool enrichment must follow the tool cwd/workspace root, not the process cwd"
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn enrichment_target_keeps_file_tool_workspace_root_precedence() {
        let workspace = unique_temp_path("enrich-workspace");
        let cwd = unique_temp_path("enrich-other-cwd");
        std::fs::create_dir_all(&workspace).expect("create workspace");
        std::fs::create_dir_all(&cwd).expect("create cwd");
        let mut ctx = ToolContext::new().with_workspace_root(workspace.clone());
        ctx.cwd = Some(cwd.clone());
        let input = serde_json::json!({ "path": "relative.rs" });

        let target = enrichment_target_path(&ctx, &input).expect("target path");

        assert_eq!(
            target,
            workspace
                .canonicalize()
                .expect("canonical workspace")
                .join("relative.rs"),
            "enrichment should mirror file_tools: workspace_root wins before cwd"
        );
        let _ = std::fs::remove_dir_all(&workspace);
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn enrichment_target_normalizes_missing_segments_like_file_tools() {
        let workspace = unique_temp_path("enrich-normalized-workspace");
        std::fs::create_dir_all(&workspace).expect("create workspace");
        let ctx = ToolContext::new().with_workspace_root(workspace.clone());
        let input = serde_json::json!({ "path": "missing_dir/../relative.rs" });

        let target = enrichment_target_path(&ctx, &input).expect("target path");

        assert_eq!(
            target,
            workspace
                .canonicalize()
                .expect("canonical workspace")
                .join("relative.rs"),
            "enrichment must normalize `..` even when an intermediate path segment is missing"
        );
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[test]
    fn enrich_skips_autoformat_while_a_probe_is_staged() {
        // A staged probe must suppress in-place formatting so the probe's bytes
        // survive for revert (rustfmt would reflow a column-0 marker line). With
        // no LSP configured the enrich step then adds nothing and returns the raw
        // output, leaving the file untouched — deterministic regardless of whether
        // a formatter is installed.
        let path = unique_temp_path("fmt").with_extension("rs");
        let misformatted = "fn   f( ){let x=1;}\n"; // rustfmt would rewrite this
        std::fs::write(&path, misformatted).expect("write fixture");

        let ctx = ToolContext::new();
        stage_probe(&ctx, path.clone(), "\n/*ZO_PROBE:1*/ //p");

        let input = serde_json::json!({ "path": path.to_string_lossy() });
        let out = maybe_enrich_file_tool_result(&ctx, "edit_file", &input, "RAW".to_string());

        assert_eq!(out, "RAW", "no enrichment while a probe is staged");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            misformatted,
            "file left unformatted while a probe is staged"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn recovery_notice_embedding_keeps_json_parseable() {
        let raw = r#"{"filePath":"/tmp/a.rs","content":"body"}"#.to_string();
        let notice = super::recovery_notice(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        );

        let enriched = super::embed_recovery_notice(raw, &notice);
        let value: serde_json::Value = serde_json::from_str(&enriched)
            .expect("compressed-only artifact notice must not corrupt JSON output");

        assert_eq!(
            value.get("filePath").and_then(|value| value.as_str()),
            Some("/tmp/a.rs")
        );
        assert_eq!(
            value.get("recoveryNotice").and_then(|value| value.as_str()),
            Some(notice.as_str())
        );
    }

    /// Regression: enriching a JSON tool result must keep it parseable as a
    /// JSON object so the TUI diff/write formatters can still read `filePath`,
    /// `content`, and `structuredPatch`. The old text-trailer form produced a
    /// non-JSON string and surfaced a phantom `? · +0 -0` even on a real edit.
    #[test]
    fn enrichment_preserves_json_object_for_renderer() {
        let raw = r#"{"filePath":"/tmp/a.rs","content":"line1\nline2","structuredPatch":[]}"#;
        let enriched = super::fold_enrichment_into_output(
            raw.to_string(),
            &["[auto-format] formatted".to_string()],
        );
        let value: serde_json::Value =
            serde_json::from_str(&enriched).expect("enriched output must stay valid JSON");
        assert_eq!(
            value.get("filePath").and_then(|v| v.as_str()),
            Some("/tmp/a.rs"),
            "renderer fields survive enrichment"
        );
        assert_eq!(
            value.get("toolFeedback").and_then(|v| v.as_str()),
            Some("[auto-format] formatted"),
            "model still receives the enrichment text"
        );
    }

    /// Non-JSON output (e.g. an already-stringified error) falls back to the
    /// readable text trailer rather than being dropped.
    #[test]
    fn enrichment_falls_back_to_text_for_non_json() {
        let enriched = super::fold_enrichment_into_output(
            "plain text result".to_string(),
            &["[auto-format] formatted".to_string()],
        );
        assert!(enriched.contains("plain text result"));
        assert!(enriched.contains("[auto-format] formatted"));
    }

    /// Regression: a `read_file` result larger than the 30k ceiling must stay
    /// valid JSON on the wire and produce an *outline* view — not a JSON
    /// envelope cut mid-string. The old path char-truncated the envelope
    /// (`chars().take(30_000)`), storing invalid JSON, which then made the wire
    /// compressor fail-open so the outline never fired. This reproduces the
    /// exact dispatch combination (truncate → recovery-notice embed) and then
    /// the wire seam, asserting both invariants.
    #[test]
    fn oversized_read_file_stays_valid_json_and_outlines_on_the_wire() {
        use runtime::context_compression::wire_tool_output;
        use runtime::file_ops::{ReadFileOutput, TextFilePayload};
        use runtime::{truncate_tool_output, TruncationConfig};
        use std::fmt::Write as _;

        // A code file whose deep bodies push the lossless envelope well past the
        // 30k ceiling. Bodies are indented deeper than `OUTLINE_KEEP_INDENT` (4)
        // so the outline elides them; the `fn` signatures sit at column 0 and
        // are kept.
        let mut content = String::new();
        for n in 0..400 {
            let _ = writeln!(content, "fn item_{n}() {{");
            for _ in 0..12 {
                content.push_str("        let _ = deep_body_line_that_is_fairly_long();\n");
            }
            content.push_str("}\n");
        }
        let total_lines = content.lines().count();
        let envelope = serde_json::to_string(&ReadFileOutput {
            kind: "text".to_string(),
            file: TextFilePayload {
                file_path: "/tmp/big.rs".to_string(),
                content,
                num_lines: total_lines,
                start_line: 1,
                total_lines,
                notice: None,
            },
        })
        .expect("serialize read_file envelope");
        assert!(
            envelope.chars().count() > 30_000,
            "fixture must exceed the truncation ceiling"
        );

        // Dispatch step 1: truncation must NOT cut the envelope — read_file is
        // exempt — so the stored JSON stays parseable.
        let truncated =
            truncate_tool_output(&envelope, "read_file", &TruncationConfig::default());
        assert!(
            !truncated.was_truncated,
            "read_file envelope must not be raw-truncated"
        );
        serde_json::from_str::<serde_json::Value>(&truncated.content)
            .expect("stored read_file output must remain valid JSON");

        // Dispatch step 2: the recovery notice is embedded as a structured
        // field (the non-truncated branch), keeping the content valid JSON.
        let stored = super::embed_recovery_notice(
            truncated.content,
            &super::recovery_notice(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
        );
        serde_json::from_str::<serde_json::Value>(&stored)
            .expect("stored output with recovery notice must remain valid JSON");

        // Wire seam: because the stored JSON is valid, the compressor can parse
        // it and emit the outline view (full file structure, deep bodies elided
        // with re-read instructions) instead of failing open.
        let wired = wire_tool_output(&stored, "read_file", false);
        let preview: String = wired.chars().take(200).collect();
        assert!(
            wired.contains("[file:outline]"),
            "the wire view must be the outline, got: {preview}"
        );
        assert!(
            wired.contains("retrieve_tool_output"),
            "the wire view must still tell the model how to recover full bytes"
        );
    }

    fn extract_notice_sha(content: &str) -> String {
        let marker = "{\"sha256\": \"";
        let start = content
            .find(marker)
            .map(|idx| idx + marker.len())
            .expect("recovery notice sha marker present");
        content[start..start + 64].to_string()
    }

    #[test]
    fn over_cap_bash_dispatch_returns_digest_and_recovery_notice() {
        let artifact_dir = unique_temp_path("bash-digest-artifacts");
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");
        let ctx = ToolContext::new();
        let command = "i=0; while [ $i -lt 1200 ]; do printf 'stdout-line-%04d detail detail detail\n' \"$i\"; i=$((i+1)); done; printf 'stderr-tail failure detail\n' >&2; exit 1";
        let input = serde_json::json!({ "command": command, "timeout": 20 });

        let content = super::execute_tool_with_context_and_artifact_dir(
            &ctx,
            None,
            "bash",
            &input,
            Some(&artifact_dir),
        )
        .expect("bash dispatch succeeds even with nonzero command exit");

        assert!(content.starts_with("[bash]"));
        assert!(content.contains("stdout-line-0000"));
        assert!(content.contains("stdout-line-1199"));
        assert!(content.contains("stderr-tail failure detail"));
        assert!(content.contains("middle elided:"));
        assert!(content.contains("retrieve_tool_output"));
        let sha = extract_notice_sha(&content);
        assert!(artifact_dir.join(sha).exists(), "artifact notice must resolve");
    }

    #[test]
    fn over_cap_cargo_test_bash_model_view_keeps_failures_and_recovery_notice() {
        use runtime::context_compression::wire_tool_output;

        let artifact_dir = unique_temp_path("cargo-test-bash-artifacts");
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");
        let ctx = ToolContext::new();
        let command = "echo 'running 901 tests'; i=0; while [ $i -lt 900 ]; do printf 'test pass_%04d ... ok\n' \"$i\"; i=$((i+1)); done; printf 'test fail_case ... FAILED\n\nfailures:\n\n---- fail_case stdout ----\nthread '\''fail_case'\'' panicked at crates/demo.rs:42:9:\nexpected true\n\nfailures:\n    fail_case\n\ntest result: FAILED. 900 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'; exit 1";
        let input = serde_json::json!({ "command": command, "timeout": 20 });

        let session_content = super::execute_tool_with_context_and_artifact_dir(
            &ctx,
            None,
            "bash",
            &input,
            Some(&artifact_dir),
        )
        .expect("bash dispatch succeeds even with failing tests");
        let model_view = wire_tool_output(&session_content, "bash", false);

        assert!(model_view.contains("test result: FAILED. 900 passed; 1 failed"));
        assert!(model_view.contains("failures:"));
        assert!(model_view.contains("---- fail_case stdout ----"));
        assert!(model_view.contains("retrieve_tool_output"));
        let summary = model_view.find("test result: FAILED").expect("summary kept");
        let failure = model_view.find("---- fail_case stdout ----").expect("failure detail kept");
        assert!(summary < failure, "summary should lead failure details");
        let digest_body = model_view
            .split("[full output preserved")
            .next()
            .unwrap_or(&model_view);
        let cargo_section_start = digest_body.find("[cargo test]").expect("cargo digest marker kept");
        let summary_in_cargo_digest = summary - cargo_section_start;
        assert!(
            summary_in_cargo_digest < digest_body[cargo_section_start..].chars().count() / 2,
            "summary should land in the cargo digest front half"
        );
        if let Some(pass_noise) = model_view.find("test pass_0000 ... ok") {
            assert!(summary < pass_noise, "summary must precede passing noise");
        }
        let sha = extract_notice_sha(&model_view);
        assert!(artifact_dir.join(sha).exists(), "artifact notice must resolve");
    }

}
