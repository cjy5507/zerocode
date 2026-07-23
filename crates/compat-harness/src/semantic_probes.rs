//! Deterministic semantic probes for deep-lane benchmark attempts.
//!
//! These checks are intentionally narrow and evidence-backed. They catch common
//! benchmark edge-case failures before spending a verifier turn, but they do not
//! try to prove general correctness.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct SemanticProbeReport {
    pub issues: Vec<String>,
}

impl SemanticProbeReport {
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Run deterministic probes that are gated by the task text and changed target
/// files. Each issue is phrased like a strict verifier finding so the deep retry
/// path can reuse the same repair contract.
#[must_use]
pub fn run_semantic_probes(work: &Path, task: &str, intended: &[String]) -> SemanticProbeReport {
    let task_lower = task.to_ascii_lowercase();
    let files = intended_source_files(work, intended);
    let mut issues = Vec::new();

    if is_validation_task(&task_lower) {
        push_unique_all(&mut issues, probe_validation_non_object_deref(&files));
    }
    if is_option_threading_task(&task_lower) {
        push_unique_all(&mut issues, probe_null_opts_deref(&files));
        push_unique_all(&mut issues, probe_opts_id_only_cache(&files));
    }
    if task_lower.contains("rename") {
        if let Some(old_method) = renamed_old_method(task) {
            push_unique_all(&mut issues, probe_stale_renamed_method(&files, &old_method));
        }
    }

    SemanticProbeReport { issues }
}

fn is_validation_task(task_lower: &str) -> bool {
    ["validat", "schema", "dto", "api layer", "required field"]
        .iter()
        .any(|needle| task_lower.contains(needle))
}

fn is_option_threading_task(task_lower: &str) -> bool {
    ["opts", "option", "thread", "include", "cache", "caller"]
        .iter()
        .any(|needle| task_lower.contains(needle))
}

fn intended_source_files(work: &Path, intended: &[String]) -> Vec<(String, PathBuf, String)> {
    let mut rels = BTreeSet::new();
    for raw in intended.iter().filter(|s| !s.trim().is_empty()) {
        let normalized = raw.trim().trim_start_matches("./");
        let path = work.join(normalized.trim_end_matches('/'));
        if path.is_file() {
            if let Ok(rel) = path.strip_prefix(work) {
                rels.insert(rel.to_string_lossy().into_owned());
            }
        } else if path.is_dir() {
            collect_source_files(work, &path, &mut rels);
        }
    }
    rels.into_iter()
        .filter_map(|rel| {
            let path = work.join(&rel);
            let content = fs::read_to_string(&path).ok()?;
            Some((rel, path, content))
        })
        .collect()
}

fn collect_source_files(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name() else {
            continue;
        };
        if name == ".git" || name == "node_modules" || name == "target" || name == "test" {
            continue;
        }
        if path.is_dir() {
            collect_source_files(root, &path, out);
        } else if is_source_file(&path) {
            if let Ok(rel) = path.strip_prefix(root) {
                out.insert(rel.to_string_lossy().into_owned());
            }
        }
    }
}

fn is_source_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("js" | "jsx" | "mjs" | "ts" | "tsx" | "rs" | "py" | "go" | "java")
    )
}

fn probe_validation_non_object_deref(files: &[(String, PathBuf, String)]) -> Vec<String> {
    let mut issues = Vec::new();
    for (rel, _, content) in files {
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let Some(var) = non_object_guard_variable(line) else {
                continue;
            };
            if block_exits_early(&lines, idx) {
                continue;
            }
            if let Some((line_no, deref)) = later_property_deref(&lines, idx + 1, &var) {
                issues.push(format!(
                    "{rel}:{line_no} dereferences `{deref}` after detecting non-object/null `{var}` input; validation must return errors instead of throwing for invalid API-layer inputs."
                ));
            }
        }
    }
    issues
}

fn non_object_guard_variable(line: &str) -> Option<String> {
    let compact = line.split_whitespace().collect::<String>();
    if !(compact.contains("typeof") && compact.contains("!==\"object\"")
        || compact.contains("!=='object'"))
    {
        return None;
    }
    if !compact.contains("===null") {
        return None;
    }
    let after_typeof = compact.split("typeof").nth(1)?;
    let var = after_typeof
        .split("!==")
        .next()?
        .trim_matches(|c: char| c == '(' || c == ')' || c == '!')
        .to_string();
    if var.is_empty() {
        None
    } else {
        Some(var)
    }
}

fn block_exits_early(lines: &[&str], guard_idx: usize) -> bool {
    let mut brace_depth = brace_delta(lines[guard_idx]);
    for line in lines.iter().skip(guard_idx + 1).take(8) {
        if line.contains("return") || line.contains("throw") {
            return true;
        }
        brace_depth += brace_delta(line);
        if brace_depth <= 0 {
            return false;
        }
    }
    false
}

fn brace_delta(line: &str) -> i32 {
    let opens = i32::try_from(line.chars().filter(|c| *c == '{').count()).unwrap_or(i32::MAX);
    let closes = i32::try_from(line.chars().filter(|c| *c == '}').count()).unwrap_or(i32::MAX);
    opens - closes
}

fn later_property_deref(lines: &[&str], start: usize, var: &str) -> Option<(usize, String)> {
    let needle = format!("{var}.");
    let end = (start + 24).min(lines.len());
    for (offset, line) in lines[start..end].iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("function ") {
            break;
        }
        if let Some(col) = trimmed.find(&needle) {
            let rest = &trimmed[col..];
            let field = rest
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.'))
                .next()
                .unwrap_or(rest);
            return Some((start + offset + 1, field.to_string()));
        }
    }
    None
}

fn probe_null_opts_deref(files: &[(String, PathBuf, String)]) -> Vec<String> {
    let mut issues = Vec::new();
    for (rel, _, content) in files {
        if !content.contains("opts = {}") || !content.contains("opts.") {
            continue;
        }
        for (idx, line) in content.lines().enumerate() {
            if line.contains("opts.")
                && !line.contains("opts?.")
                && !line.contains("opts &&")
                && !line.contains("opts !== null")
            {
                issues.push(format!(
                    "{rel}:{} defaults `opts` only when it is undefined; passing null can still throw when reading `{}` instead of treating missing options defensively.",
                    idx + 1,
                    line.trim()
                ));
                break;
            }
        }
    }
    issues
}

fn probe_opts_id_only_cache(files: &[(String, PathBuf, String)]) -> Vec<String> {
    let mut issues = Vec::new();
    for (rel, _, content) in files {
        let compact = content.split_whitespace().collect::<String>();
        let mentions_opts = content.contains("opts") || content.contains("options");
        let loads_with_opts =
            compact.contains(".load(id,opts)") || compact.contains(".load(id,options)");
        let id_only_cache = compact.contains("cache.has(id)")
            && compact.contains("cache.set(id,")
            && compact.contains("cache.get(id)");
        if mentions_opts && loads_with_opts && id_only_cache {
            issues.push(format!(
                "{rel} caches entries only by id while threading opts; different opts for the same id can return a stale value from a prior call."
            ));
        }
    }
    issues
}

fn renamed_old_method(task: &str) -> Option<String> {
    let lower = task.to_ascii_lowercase();
    let rename_pos = lower.find("rename")?;
    let to_pos = lower[rename_pos..].find(" to ").map(|p| rename_pos + p)?;
    let before_to = &task[rename_pos..to_pos];
    let open = before_to.find('(')?;
    let before_args = &before_to[..open];
    let dot = before_args.rfind('.')?;
    let method = before_args[dot + 1..].trim();
    (!method.is_empty()).then(|| method.to_string())
}

fn probe_stale_renamed_method(
    files: &[(String, PathBuf, String)],
    old_method: &str,
) -> Vec<String> {
    let mut issues = Vec::new();
    let needle = format!(".{old_method}(");
    for (rel, _, content) in files {
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue;
            }
            if trimmed.contains(&needle) {
                issues.push(format!(
                    "{rel}:{} still calls `{}` after the rename; audit every intended call site and preserve the existing receiver when replacing it.",
                    idx + 1,
                    needle
                ));
            }
        }
    }
    issues
}

fn push_unique_all(out: &mut Vec<String>, incoming: Vec<String>) {
    for issue in incoming {
        if !out.contains(&issue) {
            out.push(issue);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, content: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    #[test]
    fn catches_validation_non_object_deref() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/validate.js",
            "function validateMoney(value) {\n  const errors = [];\n  if (typeof value !== 'object' || value === null) {\n    errors.push('money must be an object');\n  }\n  if (!Number.isFinite(value.amount)) { errors.push('amount'); }\n  return errors;\n}\n",
        );
        let report = run_semantic_probes(
            tmp.path(),
            "Add a required currency field to Money and validate the API DTO.",
            &["src/validate.js".into()],
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("dereferences `value.amount`")),
            "{:?}",
            report.issues
        );
    }

    #[test]
    fn allows_validation_guard_that_returns() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/validate.js",
            "function validateMoney(value) {\n  if (typeof value !== 'object' || value === null) {\n    return ['money must be an object'];\n  }\n  return Number.isFinite(value.amount) ? [] : ['amount'];\n}\n",
        );
        let report =
            run_semantic_probes(tmp.path(), "validate API DTO", &["src/validate.js".into()]);
        assert!(report.is_clean(), "{:?}", report.issues);
    }

    #[test]
    fn catches_null_opts_default_pitfall() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/repository.js",
            "class Repository {\n  load(id, opts = {}) {\n    const record = this.records.get(id);\n    if (record.deleted && !opts.includeDeleted) return null;\n    return record;\n  }\n}\n",
        );
        let report = run_semantic_probes(
            tmp.path(),
            "Rename Repository.fetch(id) to Repository.load(id, opts) and thread opts through callers.",
            &["src/repository.js".into()],
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("passing null can still throw")),
            "{:?}",
            report.issues
        );
    }

    #[test]
    fn catches_opts_id_only_cache() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/cache.js",
            "function cachedUser(repository, id, cache, opts) {\n  if (!cache.has(id)) { cache.set(id, repository.load(id, opts)); }\n  return cache.get(id);\n}\n",
        );
        let report = run_semantic_probes(
            tmp.path(),
            "thread opts through every caller and cache",
            &["src/cache.js".into()],
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("caches entries only by id")),
            "{:?}",
            report.issues
        );
    }

    #[test]
    fn catches_stale_renamed_call_site() {
        let tmp = TempDir::new().unwrap();
        write(
            tmp.path(),
            "src/service.js",
            "function get(repository, id) { return repository.fetch(id); }\n",
        );
        let report = run_semantic_probes(
            tmp.path(),
            "Rename Repository.fetch(id) to Repository.load(id, opts) and thread opts through every caller.",
            &["src/service.js".into()],
        );
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.contains("still calls `.fetch(`")),
            "{:?}",
            report.issues
        );
    }
}
