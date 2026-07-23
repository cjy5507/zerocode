//! Item assembly: fan-out expansion, round accumulation, dedup keys,
//! and the schema-retry pass.
use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::misc_tools::AgentCompletion;

use super::super::spec::{NormalizedPhase, PhaseSource, RepeatPolicy};
use super::prompts::{
    extract_structured, phase_agent_input, render_prompt, value_to_prompt_string,
};
use super::spawn::{wait_and_account, Spawned};
use super::{
    AgentBackend, EngineState, Finding, FindingState, ItemResult, PassReceipt, Risk, RunOptions,
    SemanticVerdict, INPUT_SENTINEL, STATUS_COMPLETED, STATUS_FAILED, STATUS_STILL_RUNNING,
};

/// Text injected as `{seen}`: the accumulated prior-round completed results,
/// one per line, so the next round can avoid repeating them (loop-until-dry).
pub(super) fn seen_text(accumulated: &[ItemResult]) -> String {
    accumulated
        .iter()
        .filter(|item| item.status == STATUS_COMPLETED)
        .map(seen_line_for_mapping)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Merge a round's items into the accumulator, deduping by `dedup_by` when set.
/// Returns how many *new* (previously-unseen-key) items this round added — the
/// signal `until: no_new` uses to stop.
pub(super) fn accumulate_round(
    accumulated: &mut Vec<ItemResult>,
    seen_keys: &mut HashSet<String>,
    round_items: Vec<ItemResult>,
    repeat: Option<&RepeatPolicy>,
) -> usize {
    let dedup_by = repeat.and_then(|policy| policy.dedup_by.as_deref());
    let mut new_count = 0;
    for item in round_items {
        let Some(path) = dedup_by else {
            // No dedup key → keep every item; the whole round counts as new.
            accumulated.push(item);
            new_count += 1;
            continue;
        };
        let keys = item_dedup_keys(&item, path);
        if keys.is_empty() {
            // Keyless (failed / no JSON): kept for the report but not counted as
            // new, so a round of only keyless items can still end `no_new`.
            accumulated.push(item);
        } else if keys.iter().all(|key| seen_keys.contains(key)) {
            // Every key already seen → a duplicate; drop from the union.
        } else {
            for key in keys {
                seen_keys.insert(key);
            }
            accumulated.push(item);
            new_count += 1;
        }
    }
    new_count
}

/// This item's dedup keys via the `dedup_by` JSON path, from its structured
/// output (or its raw result parsed as JSON).
pub(super) fn item_dedup_keys(item: &ItemResult, path: &str) -> Vec<String> {
    let value = item.structured.clone().or_else(|| {
        item.result
            .as_deref()
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
    });
    value.map_or_else(Vec::new, |value| extract_dedup_keys(&value, path))
}

/// Resolve a simple JSON path against a value into zero or more string keys.
/// Supports dotted fields and `[]` array spreads — `bugs[].title`, `id`, `a.b`,
/// `items[].id`. Each `[]` fans the remaining path over every array element.
pub(super) fn extract_dedup_keys(value: &Value, path: &str) -> Vec<String> {
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    let mut keys = Vec::new();
    collect_keys(value, &segments, &mut keys);
    keys
}

pub(super) fn collect_keys(value: &Value, segments: &[&str], out: &mut Vec<String>) {
    let Some((segment, rest)) = segments.split_first() else {
        if let Some(key) = value_to_key(value) {
            out.push(key);
        }
        return;
    };
    let (field, spread) = match segment.strip_suffix("[]") {
        Some(field) => (field, true),
        None => (*segment, false),
    };
    let target = if field.is_empty() {
        Some(value)
    } else {
        value.get(field)
    };
    let Some(target) = target else { return };
    if spread {
        if let Value::Array(elements) = target {
            for element in elements {
                collect_keys(element, rest, out);
            }
        }
    } else {
        collect_keys(target, rest, out);
    }
}

pub(super) fn value_to_key(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

/// Resolve the list of `{item}` strings a phase fans out over.
pub(super) fn resolve_items(
    source: &PhaseSource,
    input: &Value,
    prior: &HashMap<String, Vec<ItemResult>>,
) -> Vec<String> {
    match source {
        PhaseSource::Single => vec![String::new()],
        PhaseSource::Fanout(items) => expand_fanout(items, input),
        // `over` maps the *completed* results of the earlier phase only —
        // failed / still-running items are dropped from the mapping input.
        PhaseSource::Over(phase_id) => prior
            .get(phase_id)
            .map(|results| {
                results
                    .iter()
                    .filter(|item| item.status == STATUS_COMPLETED)
                    .map(item_text_for_mapping)
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Expand a static fan-out list, substituting the `$input` sentinel.
pub(super) fn expand_fanout(items: &[String], input: &Value) -> Vec<String> {
    let mut expanded = Vec::with_capacity(items.len());
    for item in items {
        if item == INPUT_SENTINEL {
            match input {
                Value::Array(values) => expanded.extend(values.iter().map(value_to_prompt_string)),
                Value::Null => {}
                other => expanded.push(value_to_prompt_string(other)),
            }
        } else {
            expanded.push(item.clone());
        }
    }
    expanded
}

/// The text an `over`/`synthesize` consumer sees for a completed item: the
/// validated JSON when present, else the raw result text.
pub(super) fn item_text_for_mapping(item: &ItemResult) -> String {
    let body = item
        .structured
        .as_ref()
        .map(serde_json::Value::to_string)
        .or_else(|| item.result.clone())
        .unwrap_or_default();
    if item.loaded_skills.is_empty() {
        return body;
    }
    format!(
        "[Workflow skill receipt: the prior phase actually loaded {} through the Skill tool. Its output below already reflects that guidance; do not reload a skill solely because its name appears in the plan or output.]\n\n{body}",
        item.loaded_skills.join(", ")
    )
}

fn seen_line_for_mapping(item: &ItemResult) -> String {
    let text = item_text_for_mapping(item);
    if !item.carried {
        return text;
    }
    let reason = item
        .carry_reason
        .as_deref()
        .unwrap_or("semantic pass carried into next repeat round");
    format!("{text} [carried: {reason}]")
}

/// Turn a spawn record into its [`ItemResult`], looking up the completion and
/// (when a schema is declared) extracting structured JSON.
pub(super) fn assemble_item(
    spawned: Spawned,
    completions: &[AgentCompletion],
    schema: Option<&Value>,
    loaded_skills: Vec<String>,
) -> ItemResult {
    let Spawned {
        index,
        item,
        outcome,
    } = spawned;

    let agent_id = match outcome {
        Ok(id) => id,
        Err(error) => {
            let retry_key = Some(retry_key(index, &item, None, None));
            return ItemResult {
                index,
                input: item,
                agent_id: String::new(),
                status: STATUS_FAILED.to_string(),
                result: None,
                error: Some(error),
                structured: None,
                output_tokens: 0,
                loaded_skills,
                semantic_verdict: Some("spawn_error".to_string()),
                retry_key,
                carry_reason: None,
                carried: false,
            };
        }
    };

    let (status, result, error, captured, output_tokens) =
        match completions.iter().find(|c| c.agent_id == agent_id) {
            Some(completion) => (
                completion.status.clone(),
                completion.result.clone(),
                completion.error.clone(),
                completion.structured.clone(),
                completion.output_tokens,
            ),
            None => (STATUS_STILL_RUNNING.to_string(), None, None, None, 0),
        };

    // 8c: prefer the `StructuredOutput` tool call captured by the agent runtime
    // (exact); fall back to parsing JSON out of the free-text result (MVP path)
    // so an agent that ignored the tool still yields structure when it can.
    let structured = if status == STATUS_COMPLETED {
        captured.or_else(|| {
            schema
                .is_some()
                .then(|| result.as_deref().and_then(extract_structured))
                .flatten()
        })
    } else {
        None
    };

    let semantic_verdict = semantic_verdict(&status, structured.as_ref()).map(str::to_string);
    let retry_key = Some(retry_key(index, &item, structured.as_ref(), result.as_deref()));

    ItemResult {
        index,
        input: item,
        agent_id,
        status,
        result,
        error,
        structured,
        output_tokens,
        loaded_skills,
        semantic_verdict,
        retry_key,
        carry_reason: None,
        carried: false,
    }
}

/// Classify a completed agent's structured tool output into a coarse verdict
/// label: `Some("pass")` (structured pass with coverage evidence), `Some(
/// "finding")` (structured rejection/issue), `Some("retry")` (status
/// completed but no usable structured verdict — ambiguous, never a signal),
/// or `None` (the run itself never reached `STATUS_COMPLETED`). `pub(crate)`
/// so both the workflow engine's repair loop AND the general single-agent
/// spawn path (`misc_tools::agent_tools::spawn`, Phase 4 verdict widening —
/// planner-bound reviewer→worker pairs and ad-hoc standalone reviews) share
/// this ONE structured-verdict classifier instead of a second parser.
pub(crate) fn semantic_verdict(status: &str, structured: Option<&Value>) -> Option<&'static str> {
    match normalize_semantic_verdict(usize::MAX, "", status, structured) {
        SemanticVerdict::Pass(_) => Some("pass"),
        SemanticVerdict::Finding(_) => Some("finding"),
        SemanticVerdict::Unknown => None,
        SemanticVerdict::Invalid(_) => Some("retry"),
    }
}

pub(super) fn normalize_semantic_verdict(
    index: usize,
    input: &str,
    status: &str,
    structured: Option<&Value>,
) -> SemanticVerdict {
    if status != STATUS_COMPLETED {
        return SemanticVerdict::Unknown;
    }
    let Some(value) = structured else {
        return SemanticVerdict::Invalid("missing structured validator output".to_string());
    };
    if structured_has_finding(value) {
        return SemanticVerdict::Finding(finding_from_value(index, input, value));
    }
    if structured_is_pass(value) {
        return SemanticVerdict::Pass(pass_receipt_from_value(index, input, value));
    }
    SemanticVerdict::Unknown
}

fn finding_from_value(index: usize, input: &str, value: &Value) -> Finding {
    let title = first_string(value, &["title", "summary", "finding", "message"])
        .unwrap_or_else(|| format!("finding for item {index}"));
    let evidence = first_string(value, &["evidence", "reason", "details", "error"])
        .unwrap_or_else(|| value.to_string());
    let affected_paths = value
        .get("affected_paths")
        .or_else(|| value.get("paths"))
        .or_else(|| value.get("files"))
        .and_then(Value::as_array)
        .map(|paths| {
            paths
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Finding {
        id: stable_finding_id(index, input, &title, &evidence),
        title,
        affected_paths,
        risk: risk_from_value(value),
        evidence,
        state: FindingState::Queued,
        attempts: 0,
    }
}

/// The validator's declared blast radius for a finding/pass, defaulting to
/// `Local` when absent or unrecognized. This is the evidence-based risk source
/// (§9) — no hardcoded path list.
fn risk_from_value(value: &Value) -> Risk {
    match value
        .get("risk")
        .and_then(Value::as_str)
        .map(|risk| risk.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("global") => Risk::Global,
        Some("shared") => Risk::Shared,
        _ => Risk::Local,
    }
}

fn pass_receipt_from_value(index: usize, input: &str, value: &Value) -> PassReceipt {
    let coverage = first_string(value, &["coverage", "evidence", "checked", "tests"])
        .unwrap_or_else(|| value.to_string());
    PassReceipt {
        item_index: index,
        receipt_key: stable_finding_id(index, input, "pass", &coverage),
        coverage,
        risk: risk_from_value(value),
    }
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(stringish_value)
}

fn stringish_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => (!value.trim().is_empty()).then(|| value.trim().to_string()),
        Value::Array(values) => values.iter().find_map(stringish_value),
        Value::Object(_) if value_has_content(value) => Some(value.to_string()),
        _ => None,
    }
}

fn stable_finding_id(index: usize, input: &str, title: &str, evidence: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    index.hash(&mut hasher);
    input.hash(&mut hasher);
    title.hash(&mut hasher);
    evidence.hash(&mut hasher);
    format!("finding-{index}-{:016x}", hasher.finish())
}

pub(super) fn retry_key(
    index: usize,
    input: &str,
    structured: Option<&Value>,
    result: Option<&str>,
) -> String {
    let finding_key = structured
        .and_then(|value| value.get("finding_id").or_else(|| value.get("id")))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    if let Some(key) = finding_key {
        return format!("finding:{key}");
    }
    let issue_key = structured
        .and_then(|value| value.get("issues"))
        .and_then(Value::as_array)
        .and_then(|issues| issues.first())
        .map(value_to_prompt_string)
        .filter(|value| !value.trim().is_empty());
    if let Some(key) = issue_key {
        return format!("issue:{key}");
    }
    let result_key = result.filter(|value| !value.trim().is_empty()).unwrap_or(input);
    format!("item:{index}:{result_key}")
}

fn structured_has_finding(value: &Value) -> bool {
    if value.get("issues").and_then(Value::as_array).is_some_and(|issues| !issues.is_empty()) {
        return true;
    }
    if value
        .get("verdict")
        .and_then(Value::as_str)
        .is_some_and(|verdict| matches!(verdict, "fail" | "failed" | "finding"))
    {
        return true;
    }
    ["spec", "regression", "security", "passed", "pass", "ok", "success"]
        .iter()
        .any(|key| value.get(*key).and_then(Value::as_bool) == Some(false))
}

fn structured_is_pass(value: &Value) -> bool {
    if ["spec", "regression", "security"]
        .iter()
        .all(|key| value.get(*key).and_then(Value::as_bool) == Some(true))
    {
        return true;
    }
    if value
        .get("verdict")
        .and_then(Value::as_str)
        .is_some_and(|verdict| matches!(verdict, "pass" | "passed" | "ok" | "success"))
    {
        return structured_has_coverage(value);
    }
    ["passed", "pass", "ok", "success"]
        .iter()
        .any(|key| value.get(*key).and_then(Value::as_bool) == Some(true))
        && structured_has_coverage(value)
}

fn structured_has_coverage(value: &Value) -> bool {
    ["coverage", "evidence", "checked", "tests", "files", "citations"]
        .iter()
        .any(|key| value.get(*key).is_some_and(value_has_content))
}

fn value_has_content(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(_) => true,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
    }
}

/// One re-spawn per item whose schema extraction failed, with an emphasized
/// JSON-only instruction. After this single retry, items that still don't
/// yield JSON keep `structured: None` and their raw `result` (never discarded).
pub(super) fn schema_retry_pass(
    phase: &NormalizedPhase,
    input_text: &str,
    schema: &Value,
    items: &mut [ItemResult],
    backend: &mut dyn AgentBackend,
    opts: &RunOptions,
    state: &mut EngineState,
) {
    for item in items.iter_mut() {
        if item.status != STATUS_COMPLETED || item.structured.is_some() {
            continue;
        }
        if state.budget_blocks_spawn() {
            state.mark_exhausted();
            break;
        }

        // The retry only needs to re-state the task + force JSON; `{seen}`
        // context (if any) was already present in the first attempt, so pass "".
        let base = render_prompt(
            &phase.prompt,
            &item.input,
            item.index,
            input_text,
            "",
            Some(schema),
        );
        let prompt = format!(
            "{base}\n\nIMPORTANT: your previous reply was not valid JSON. Reply with ONLY the JSON value — no prose, no code fences."
        );
        // A schema retry only re-asks for JSON-shaped output (no new file work),
        // so it runs without a worktree; 8c removes this pass entirely.
        let Ok(agent_id) = backend.spawn(phase_agent_input(
            phase,
            item.index,
            &item.input,
            prompt,
            0,
            true,
            None,
        )) else {
            continue;
        };
        state.record_spawn();

        let completions = wait_and_account(
            backend,
            state,
            std::slice::from_ref(&agent_id),
            opts.phase_timeout,
        );
        let Some(completion) = completions.into_iter().find(|c| c.agent_id == agent_id) else {
            continue;
        };
        if completion.status == STATUS_COMPLETED {
            if let Some(text) = completion.result {
                if let Some(value) = extract_structured(&text) {
                    // Keep structured + raw consistent — both from the retry.
                    item.structured = Some(value);
                    item.result = Some(text);
                    item.semantic_verdict = semantic_verdict(&item.status, item.structured.as_ref()).map(str::to_string);
                    item.retry_key = Some(retry_key(
                        item.index,
                        &item.input,
                        item.structured.as_ref(),
                        item.result.as_deref(),
                    ));
                }
                // else: leave the original result in place; structured stays None.
            }
        }
    }
}
