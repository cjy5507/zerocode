use super::*;

#[test]
fn retry_attempt_phase_time_is_charged_to_repair_bucket() {
    let mut timings = DeepPhaseTimings::default();
    timings.add_attempt_exec(1, Duration::from_millis(10));
    timings.add_attempt_test(1, Duration::from_millis(20));
    timings.add_attempt_verify(1, Duration::from_millis(30));
    timings.add_attempt_exec(2, Duration::from_millis(40));
    timings.add_attempt_test(2, Duration::from_millis(50));
    timings.add_attempt_verify(2, Duration::from_millis(60));

    assert_eq!(timings.exec_millis, 10);
    assert_eq!(timings.test_millis, 20);
    assert_eq!(timings.verify_millis, 30);
    assert_eq!(timings.repair_millis, 150);
}

#[test]
fn deep_usage_sums_calls_and_synthesizes() {
    let mut u = DeepUsage::new();
    u.accumulate(r#"{"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":1,"cache_read_input_tokens":2},"iterations":1}"#);
    u.accumulate(r#"{"usage":{"input_tokens":20,"output_tokens":7,"cache_creation_input_tokens":3,"cache_read_input_tokens":4},"num_turns":2,"permission_denials":[{}]}"#);
    let s = u.synthesize(0);
    let v: Value = serde_json::from_str(&s).unwrap();
    assert_eq!(v["usage"]["input_tokens"], 30);
    assert_eq!(v["usage"]["output_tokens"], 12);
    assert_eq!(v["usage"]["cache_creation_input_tokens"], 4);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 6);
    assert_eq!(v["iterations"], 3);
    assert_eq!(v["permission_denials"].as_array().unwrap().len(), 1);
    assert_eq!(v["is_error"], false);
}

#[test]
fn deep_usage_incomplete_cache_drops_breakdown() {
    let mut u = DeepUsage::new();
    u.accumulate(r#"{"usage":{"input_tokens":10,"output_tokens":5}}"#);
    let v: Value = serde_json::from_str(&u.synthesize(1)).unwrap();
    assert_eq!(v["usage"]["input_tokens"], 10);
    assert!(v["usage"].get("cache_read_input_tokens").is_none());
    assert_eq!(v["is_error"], true);
}

#[test]
fn extract_result_reads_result_or_message() {
    assert_eq!(extract_result(r#"{"result":"hi"}"#), "hi");
    assert_eq!(extract_result(r#"{"message":"yo"}"#), "yo");
    assert_eq!(extract_result("not json"), "");
}

#[test]
fn parsed_verifier_artifact_only_when_parseable() {
    use decision_core::deep_lane::parse_verifier;
    // Strict JSON → a parsed artifact carrying the verdict and spec parse mode.
    let parsed =
        parseable_verifier_json(&parse_verifier(r#"{"accepted": false, "issues": ["x"]}"#))
            .expect("strict JSON is parseable");
    let v: Value = serde_json::from_str(&parsed).unwrap();
    assert_eq!(v["accepted"], false);
    assert_eq!(v["parse_mode"], "strict_valid");
    assert_eq!(v["issues"][0], "x");
    // Empty / unparseable output → no parsed file (the doc's "when parseable").
    assert!(parseable_verifier_json(&parse_verifier("")).is_none());
    assert!(parseable_verifier_json(&parse_verifier("hmm, not sure")).is_none());
}

#[test]
fn skipped_verifier_is_empty_parse_and_no_accept() {
    let (verifier, raw) = skipped_verifier_for_red_objective();
    assert!(!verifier.accepted);
    assert_eq!(verifier.parse, VerifierParse::Empty);
    assert_eq!(
        verifier.issues,
        vec!["objective gate failed; verifier skipped"]
    );
    assert!(raw.contains("verifier skipped"));
    assert!(parseable_verifier_json(&verifier).is_none());
}

#[test]
fn fallback_plan_is_valid_and_concrete() {
    let spec = RunSpec {
        runner: "zo_claude".into(),
        runner_kind: "zo".into(),
        bin: std::path::PathBuf::from("/bin/true"),
        args: Vec::new(),
        fixture: std::path::PathBuf::from("fixture"),
        prompt: "Implement rollback when async batch writes fail.".into(),
        test_command: Some("npm test".into()),
        intended: vec!["src/store.js".into(), "src/batch.js".into()],
        lane: "deep".into(),
        model: "claude-opus-4-8".into(),
        effort: "max".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 300,
        artifacts_dir: None,
        keep_failed: false,
        deep: None,
    };
    let plan = fallback_plan_for_spec(&spec, Path::new("."), &["files".into(), "tests".into()]);
    let verdict = validate_plan(&plan);
    assert!(
        verdict.valid,
        "fallback plan missing: {:?}\n{plan}",
        verdict.missing
    );
    assert!(plan.contains("src/store.js"));
    assert!(plan.contains("npm test"));
    assert!(plan.contains("Harness fallback plan"));
}

#[test]
fn intended_directories_expand_into_context_and_fallback_plan() {
    let root = std::env::temp_dir().join(format!(
        "zo-deep-context-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("src/nested")).unwrap();
    std::fs::write(
        root.join("src/parser.js"),
        "export function parseCsv() {}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/nested/state.ts"),
        "export const state = {};\n",
    )
    .unwrap();
    std::fs::write(root.join("src/notes.txt"), "not source context\n").unwrap();

    let intended = vec!["src/".to_string()];
    let expanded = expand_intended_files(&root, &intended, MAX_INTENDED_CONTEXT_FILES);
    assert_eq!(
        expanded
            .iter()
            .map(|(path, _)| path.as_str())
            .collect::<Vec<_>>(),
        vec!["src/nested/state.ts", "src/parser.js"]
    );
    let context = context_pack(&root, &intended);
    assert!(context.contains("## Expanded intended target files"));
    assert!(context.contains("### src/parser.js"));
    assert!(context.contains("export function parseCsv"));
    assert!(!context.contains("not source context"));
    std::fs::create_dir_all(root.join("test")).unwrap();
    std::fs::write(
        root.join("test/parser.test.js"),
        "const assert = require('node:assert/strict');\n",
    )
    .unwrap();
    let exec_context = exec_context_pack(
        &root,
        &intended,
        "baseline red",
        "Implement a streaming CSV parser.",
    );
    assert!(exec_context.contains("## Smart-first hard-task strategy"));
    assert!(exec_context.contains("First action: edit/write"));
    assert!(exec_context.contains("state-machine scan"));
    assert!(exec_context.contains("## Baseline objective signal"));
    assert!(exec_context.contains("baseline red"));
    assert!(exec_context.contains("## Editable target file snapshots"));
    assert!(exec_context.contains("### src/parser.js"));
    assert!(exec_context.contains("## Relevant tests / assertions"));
    assert!(exec_context.contains("test/parser.test.js"));

    let spec = RunSpec {
        runner: "zo_claude".into(),
        runner_kind: "zo".into(),
        bin: std::path::PathBuf::from("/bin/true"),
        args: Vec::new(),
        fixture: root.clone(),
        prompt: "Implement a streaming CSV parser.".into(),
        test_command: Some("npm test".into()),
        intended,
        lane: "deep".into(),
        model: "claude-opus-4-8".into(),
        effort: "max".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 300,
        artifacts_dir: None,
        keep_failed: false,
        deep: None,
    };
    let plan = fallback_plan_for_spec(&spec, &root, &["files".into()]);
    assert!(plan.contains("src/parser.js"));
    assert!(plan.contains("src/nested/state.ts"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn cross_file_rename_tasks_get_receiver_preserving_smart_context() {
    let root = std::env::temp_dir().join(format!(
        "zo-deep-rename-context-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/service.js"),
        "function getDisplayName(repository, id) {\n  const user = repository.fetch(id);\n}\n",
    )
    .unwrap();

    let spec = RunSpec {
        runner: "zo_gpt".into(),
        runner_kind: "zo".into(),
        bin: std::path::PathBuf::from("/bin/true"),
        args: Vec::new(),
        fixture: root.clone(),
        prompt: "Rename Repository.fetch(id) to Repository.load(id, opts) and thread opts through every caller.".into(),
        test_command: Some("node --test".into()),
        intended: vec!["src/".into()],
        lane: "deep".into(),
        model: "gpt-5.5".into(),
        effort: "xhigh".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 300,
        artifacts_dir: None,
        keep_failed: false,
        deep: None,
    };

    assert!(needs_smart_first(&spec));
    let context = exec_context_pack(&root, &spec.intended, "baseline red", &spec.prompt);
    assert!(context.contains("For rename/thread-caller tasks"));
    assert!(context.contains("preserve the existing receiver"));
    assert!(context.contains("src/service.js:2 keeps receiver `repository`"));
    assert!(context.contains("renaming `fetch`"));
    assert!(context.contains("do not replace it with a type/class name"));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn complex_parser_tasks_get_more_exec_budget_without_spending_parent() {
    let mut spec = RunSpec {
        runner: "zo_claude".into(),
        runner_kind: "zo".into(),
        bin: std::path::PathBuf::from("/bin/true"),
        args: Vec::new(),
        fixture: std::path::PathBuf::from("fixture"),
        prompt: "Implement a streaming CSV parser across arbitrary chunks.".into(),
        test_command: Some("npm test".into()),
        intended: vec!["src/".into()],
        lane: "deep".into(),
        model: "claude-opus-4-8".into(),
        effort: "max".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 300,
        artifacts_dir: None,
        keep_failed: false,
        deep: None,
    };
    let parent = RunBudget::from_duration(Duration::from_secs(300));
    assert!(needs_smart_first(&spec));
    let complex = exec_phase_budget(&spec, &parent)
        .remaining()
        .unwrap_or_default();
    spec.prompt = "Rename one helper.".into();
    spec.intended = vec!["src/helper.js".into()];
    assert!(!needs_smart_first(&spec));
    let simple = exec_phase_budget(&spec, &parent)
        .remaining()
        .unwrap_or_default();
    assert!(complex > Duration::from_secs(239), "complex={complex:?}");
    assert!(complex <= Duration::from_secs(240), "complex={complex:?}");
    assert!(simple <= Duration::from_secs(150), "simple={simple:?}");
    assert!(complex > simple + Duration::from_secs(80));
    assert!(parent.remaining().unwrap_or_default() > Duration::from_secs(290));
}

#[test]
fn phase_budgets_are_token_fast_bounded() {
    let parent = RunBudget::from_duration(Duration::from_secs(300));
    let plan = plan_phase_budget(&parent).remaining().unwrap_or_default();
    let verify = verify_phase_budget(&parent).remaining().unwrap_or_default();
    let retry = verify_retry_phase_budget(&parent)
        .remaining()
        .unwrap_or_default();
    assert!(plan <= Duration::from_secs(25));
    assert!(verify <= Duration::from_secs(30));
    assert!(retry <= Duration::from_secs(15));
    assert!(plan > Duration::from_secs(24));
    assert!(verify > Duration::from_secs(29));
    assert!(retry > Duration::from_secs(14));
}

#[test]
fn verifier_retry_prefers_parseable_or_accepted_output() {
    let timeout = VerifierVerdict {
        accepted: false,
        issues: vec!["verifier timed out".into()],
        parse: VerifierParse::Timeout,
        evidence: None,
    };
    let unparseable = VerifierVerdict {
        accepted: false,
        issues: vec!["bad".into()],
        parse: VerifierParse::Unparseable,
        evidence: None,
    };
    let json_reject = VerifierVerdict {
        accepted: false,
        issues: vec!["real defect".into()],
        parse: VerifierParse::Json,
        evidence: None,
    };
    let accepted = VerifierVerdict {
        accepted: true,
        issues: Vec::new(),
        parse: VerifierParse::Salvaged,
        evidence: None,
    };
    assert!(verifier_needs_compact_retry(&timeout));
    assert!(!verifier_retry_is_better(&timeout, &unparseable));
    assert!(verifier_retry_is_better(&timeout, &json_reject));
    assert!(verifier_retry_is_better(&unparseable, &accepted));
    assert!(!verifier_needs_compact_retry(&accepted));
}

#[test]
fn objective_recovery_accepts_only_missing_verifier_signal() {
    let timeout = VerifierVerdict {
        accepted: false,
        issues: vec!["verifier timed out".into()],
        parse: VerifierParse::Timeout,
        evidence: None,
    };
    let empty = VerifierVerdict {
        accepted: false,
        issues: Vec::new(),
        parse: VerifierParse::Empty,
        evidence: None,
    };
    let unparseable = VerifierVerdict {
        accepted: false,
        issues: vec!["not json".into()],
        parse: VerifierParse::Unparseable,
        evidence: None,
    };

    assert!(verifier_can_recover_from_objective(true, true, &timeout));
    assert!(verifier_can_recover_from_objective(true, true, &empty));
    assert!(!verifier_can_recover_from_objective(false, true, &timeout));
    assert!(!verifier_can_recover_from_objective(true, false, &timeout));
    assert!(!verifier_can_recover_from_objective(
        true,
        true,
        &unparseable
    ));

    let (verifier, raw) = recovered_objective_verifier();
    assert!(verifier.accepted);
    assert_eq!(verifier.parse, VerifierParse::Json);
    assert!(raw.contains("objective_evidence_after_verifier_timeout"));
}

#[test]
fn prompts_carry_phase_markers_and_task() {
    assert!(plan_prompt("T", "B", "C").contains("[[ZO-DEEP:PLAN]]"));
    assert!(plan_prompt("T", "B", "C").contains("## Target files"));
    assert!(plan_prompt("T", "B", "C").contains("Do not call tools"));
    let ex = exec_prompt("T", "P", "CTX", Some("RETRY"));
    assert!(ex.contains("[[ZO-DEEP:EXEC]]"));
    assert!(ex.contains("Performance contract"));
    assert!(ex.contains("The harness will run tests"));
    assert!(ex.contains("stop immediately"));
    assert!(ex.contains("RETRY"));
    assert!(ex.contains("Immediate mechanical edits"));
    assert!(ex.contains("exact receiver replacements"));
    assert!(ex.contains("Preserve call receivers during renames"));
    assert!(ex.contains("Implementation context"));
    assert!(ex.contains("CTX"));
    assert!(ex.contains("direct edit/write"));
    let lean_ex = exec_prompt("T", "P", "", None);
    assert!(!lean_ex.contains("Smart-first implementation context"));
    assert!(!lean_ex.contains("Implementation context"));
    assert!(!exec_prompt("T", "P", "CTX", None).contains("previous attempt"));
    let verify = verify_prompt("T", Path::new("/nonexistent"), "", TestStatus::Pass, None);
    assert!(verify.contains("[[ZO-DEEP:VERIFY]]"));
    assert!(verify.contains("Do not call tools"));
    assert!(compact_verify_prompt("T", "", TestStatus::Pass, None).contains("Do not call tools"));
}

#[test]
fn repair_hints_extract_undefined_symbols() {
    let text = "ReferenceError: Repository is not defined\nRepository is not imported or defined\ncall site fails because Repository is not defined";
    assert_eq!(extract_undefined_identifiers(text), vec!["Repository"]);
}

#[test]
fn repair_hints_point_to_changed_occurrences() {
    let root = std::env::temp_dir().join(format!("zo-deep-repair-hints-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/service.js"),
        "function getDisplayName(repository, id) {\n  return Repository.load(id);\n}\n",
    )
    .unwrap();

    let hints = mechanical_repair_hints(
        &root,
        " M src/service.js\n",
        "ReferenceError: Repository is not defined",
        2000,
    );

    assert!(hints.contains("MUST eliminate undefined receiver `Repository`"));
    assert!(hints.contains("src/service.js:2"));
    assert!(hints.contains("`return Repository.load(id);` -> `return repository.load(id);`"));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn exec_prompt_includes_adversarial_validation_options_and_rename_rules() {
    let prompt = exec_prompt(
        "Rename Repository.fetch(id) to Repository.load(id, opts), thread options through cache, and validate the schema.",
        "## Target files\n- src/repository.js\n\n## Invariants\n- Preserve API callers.\n\n## Expected tests\n- npm test\n\n## Risks\n- null opts",
        "",
        None,
    );
    assert!(prompt.contains("validation functions must never throw"));
    assert!(prompt.contains("explicit null"));
    assert!(prompt.contains("id-only cache"));
    assert!(prompt.contains("preserve each original call receiver"));
}

#[test]
fn retry_context_prioritizes_immediate_mechanical_edits() {
    let root =
        std::env::temp_dir().join(format!("zo-deep-retry-context-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/service.js"),
        "function getDisplayName(repository, id) {\n  return Repository.load(id);\n}\n",
    )
    .unwrap();

    let context = retry_context(
        &root,
        " M src/service.js\n",
        "ReferenceError: Repository is not defined",
    );

    assert!(context.starts_with("## Immediate mechanical edits"));
    assert!(context.contains("`return Repository.load(id);` -> `return repository.load(id);`"));
    assert!(context.contains("## Failure summary"));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn repair_hints_ignore_non_receiver_symbol_mentions() {
    let root = std::env::temp_dir().join(format!(
        "zo-deep-repair-hints-nonreceiver-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/repository.js"),
        "class Repository {}\nmodule.exports = { Repository };\n",
    )
    .unwrap();

    let hints = mechanical_repair_hints(
        &root,
        " M src/repository.js\n",
        "ReferenceError: Repository is not defined",
        2000,
    );

    assert!(hints.is_empty());

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn repair_hints_cover_multiple_changed_call_sites() {
    let root = std::env::temp_dir().join(format!(
        "zo-deep-repair-hints-multiple-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/service.js"),
        "function getDisplayName(repository, id, opts) {\n  const user = Repository.load(id, opts);\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("src/cache.js"),
        "function cachedUser(repository, id, cache, opts) {\n  cache.set(id, Repository.load(id, opts));\n}\n",
    )
    .unwrap();

    let hints = mechanical_repair_hints(
        &root,
        " M src/service.js\n M src/cache.js\n",
        "Repository is not defined",
        4000,
    );

    assert!(hints.contains(
        "`const user = Repository.load(id, opts);` -> `const user = repository.load(id, opts);`"
    ));
    assert!(hints.contains(
        "`cache.set(id, Repository.load(id, opts));` -> `cache.set(id, repository.load(id, opts));`"
    ));

    let _ = fs::remove_dir_all(&root);
}
