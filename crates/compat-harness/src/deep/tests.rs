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
    u.accumulate(r#"{"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":1,"cache_read_input_tokens":2},"iterations":1}"#, true);
    u.accumulate(r#"{"usage":{"input_tokens":20,"output_tokens":7,"cache_creation_input_tokens":3,"cache_read_input_tokens":4},"num_turns":2,"permission_denials":[{}]}"#, true);
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
    u.accumulate(r#"{"usage":{"input_tokens":10,"output_tokens":5}}"#, true);
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
        ablation: Vec::new(),
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
        true,
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
        ablation: Vec::new(),
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
        ablation: Vec::new(),
        deep: None,
    };

    assert!(needs_smart_first(&spec));
    let context = exec_context_pack(&root, &spec.intended, "baseline red", &spec.prompt, true);
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
        ablation: Vec::new(),
        deep: None,
    };
    let phases = PhaseBudgetPolicy::default();
    let parent = RunBudget::from_duration(Duration::from_secs(300));
    assert!(needs_smart_first(&spec));
    let smart_first = exec_phase_budget(&phases, &spec, &parent)
        .remaining()
        .unwrap_or_default();
    spec.prompt = "Rename one helper.".into();
    spec.intended = vec!["src/helper.js".into()];
    assert!(!needs_smart_first(&spec));
    let simple = exec_phase_budget(&phases, &spec, &parent)
        .remaining()
        .unwrap_or_default();
    assert!(
        smart_first > Duration::from_secs(239),
        "smart_first={smart_first:?}"
    );
    assert!(
        smart_first <= Duration::from_secs(240),
        "smart_first={smart_first:?}"
    );
    assert!(simple <= Duration::from_secs(150), "simple={simple:?}");
    assert!(smart_first > simple + Duration::from_secs(80));
    assert!(parent.remaining().unwrap_or_default() > Duration::from_secs(290));
}

#[test]
fn phase_budgets_are_token_fast_bounded() {
    let phases = PhaseBudgetPolicy::default();
    let parent = RunBudget::from_duration(Duration::from_secs(300));
    let plan = plan_phase_budget(&phases, &parent)
        .remaining()
        .unwrap_or_default();
    let verify = verify_phase_budget(&phases, &parent)
        .remaining()
        .unwrap_or_default();
    let retry = verify_retry_phase_budget(&phases, &parent)
        .remaining()
        .unwrap_or_default();
    assert!(plan <= Duration::from_secs(25));
    assert!(verify <= Duration::from_secs(30));
    assert!(retry <= Duration::from_secs(15));
    assert!(plan > Duration::from_secs(24));
    assert!(verify > Duration::from_secs(29));
    assert!(retry > Duration::from_secs(14));
}

/// Pins every default to the constant it replaced.
///
/// The budgets moved out of `deep.rs` constants into `lanes.toml`, and every other
/// assertion here is a bound (`<= 25s`, `> 24s`), not an equality — so a mistyped
/// default would have shifted every benchmark score while the suite stayed green.
/// The reserves had no assertion at all.
#[test]
fn phase_budget_defaults_match_the_constants_they_replaced() {
    let phases = PhaseBudgetPolicy::default();

    assert_eq!(phases.plan_cap(), Duration::from_secs(25));
    assert_eq!(phases.exec_cap(), Duration::from_secs(150));
    assert_eq!(phases.smart_first_exec_cap(), Duration::from_secs(240));
    assert_eq!(phases.verify_cap(), Duration::from_secs(30));
    assert_eq!(phases.verify_retry_cap(), Duration::from_secs(15));
    assert_eq!(
        phases.objective_validation_reserve(),
        Duration::from_secs(60)
    );
    assert_eq!(
        phases.smart_first_validation_reserve(),
        Duration::from_secs(60)
    );
    assert_eq!(phases.final_result_test_reserve(), Duration::from_secs(60));
    assert_eq!(phases.verify_retry_reserve(), Duration::from_secs(5));
}

/// Retuning a lane's phases is a data change now — the promise `lanes.toml` makes
/// in its own header and could not keep while these were constants. A lane that
/// declares one key gets that key and defaults for the rest, and the override has
/// to reach the budget the loop actually spends, not just the parsed struct.
#[test]
fn lane_phase_budget_overrides_only_the_keys_it_declares() {
    use crate::manifest::LaneCatalog;

    let catalog = LaneCatalog::from_toml(
        "schema_version = \"1.0\"\n\
         [lanes.fast]\n\
         objective_gate=\"test_and_diff\"\nverifier_policy=\"none\"\n\
         retry_budget=0\ndiff_policy=\"intended_paths_only\"\ntimeout_seconds=120\n\
         [lanes.deep]\n\
         objective_gate=\"test_and_diff\"\nverifier_policy=\"strict\"\n\
         retry_budget=2\ndiff_policy=\"intended_paths_only\"\ntimeout_seconds=600\n\
         [lanes.deep.phase_budget]\nexec_cap_seconds=300\n",
    )
    .expect("a catalog carrying a phase_budget table parses");

    // The lane that declared nothing is untouched.
    assert_eq!(
        catalog.lanes["fast"].phase_budget.exec_cap(),
        Duration::from_secs(150)
    );

    let phases = catalog.lanes["deep"].phase_budget;
    assert_eq!(phases.exec_cap(), Duration::from_secs(300));
    // Keys it did not declare keep the default instead of resetting to zero.
    assert_eq!(phases.plan_cap(), Duration::from_secs(25));
    assert_eq!(
        phases.objective_validation_reserve(),
        Duration::from_secs(60)
    );

    let mut spec = plain_deep_spec();
    spec.prompt = "Add a currency field to Money and thread it through the DTO.".into();
    assert!(!needs_smart_first(&spec));
    let parent = RunBudget::from_duration(Duration::from_secs(600));
    let spent = exec_phase_budget(&phases, &spec, &parent)
        .remaining()
        .unwrap_or_default();
    assert!(spent > Duration::from_secs(299), "spent={spent:?}");
    assert!(spent <= Duration::from_secs(300), "spent={spent:?}");
}

/// A misspelled key is an error, not a silent default. Both the catalog's own
/// header and `PhaseBudgetPolicy`'s docs promise this, and the failure it prevents
/// is the quiet one: a benchmark that ran on numbers nobody chose while its
/// operator believed the file they edited had taken effect.
#[test]
fn a_misspelled_phase_budget_key_is_rejected_rather_than_ignored() {
    use crate::manifest::LaneCatalog;

    let with_typo = "schema_version = \"1.0\"\n\
         [lanes.deep]\n\
         objective_gate=\"test_and_diff\"\nverifier_policy=\"strict\"\n\
         retry_budget=2\ndiff_policy=\"intended_paths_only\"\ntimeout_seconds=300\n\
         [lanes.deep.phase_budget]\nexec_cap_second=300\n";

    let error = LaneCatalog::from_toml(with_typo)
        .expect_err("`exec_cap_second` is not a field and must not parse");
    assert!(
        error.to_string().contains("exec_cap_second"),
        "the error should name the offending key, got: {error}"
    );
}

/// A configured retry cap must shorten the verifier's retry, never switch it off.
///
/// The spawn floor stayed a hard-coded five seconds while the cap became data, so
/// `verify_retry_cap_seconds = 5` produced a five-second budget that failed a
/// `> 5s` check: the catalog advertised a cap and the retry silently never ran.
/// Two numbers that happened to be equal in the old code had different jobs — one
/// the reserve subtracted from the parent, one the floor worth spawning for — and
/// only the reserve moved into the policy.
#[test]
fn a_small_verify_retry_cap_shortens_the_retry_instead_of_disabling_it() {
    let phases = PhaseBudgetPolicy {
        verify_retry_cap_seconds: 5,
        ..PhaseBudgetPolicy::default()
    };
    let parent = RunBudget::from_duration(Duration::from_secs(300));
    let retry = verify_retry_phase_budget(&phases, &parent);
    let remaining = retry.remaining().unwrap_or_default();

    // The cap is honoured as a cap.
    assert!(remaining > Duration::from_secs(4), "remaining={remaining:?}");
    assert!(remaining <= Duration::from_secs(5), "remaining={remaining:?}");

    // And the loop's own gate — not a recomputed copy of it — lets the turn spawn.
    assert!(
        retry_is_worth_spawning(&phases, &retry),
        "a 5s cap must still spawn, remaining={remaining:?}"
    );
}

/// A large cap must not *raise* the bar on the retry it configures.
///
/// This is the case an unclamped `cap / 3` floor broke: at `cap = 60` it demanded
/// twenty seconds, so a parent with only twenty to spare skipped a retry the old
/// hard-coded five-second floor would have spawned. A cap silently disabling a
/// retry by being too large is the same defect as one being too small, pointed the
/// other way.
#[test]
fn a_large_verify_retry_cap_still_spawns_a_budget_the_loop_used_to_accept() {
    let phases = PhaseBudgetPolicy {
        verify_retry_cap_seconds: 60,
        ..PhaseBudgetPolicy::default()
    };
    // A 25s parent, less the 5s reserve, leaves just under 20s — far inside the cap.
    let parent = RunBudget::from_duration(Duration::from_secs(25));
    let retry = verify_retry_phase_budget(&phases, &parent);
    let remaining = retry.remaining().unwrap_or_default();

    assert!(remaining > Duration::from_secs(19), "remaining={remaining:?}");
    assert!(remaining <= Duration::from_secs(20), "remaining={remaining:?}");
    assert!(
        retry_is_worth_spawning(&phases, &retry),
        "a 60s cap must not demand more than the legacy 5s, remaining={remaining:?}"
    );
}

/// The shipped catalog's own numbers, not a synthetic one.
///
/// Four of five recorded `schema-propagation` deep runs sat at 25,079–25,091ms
/// against the legacy 25s planning cap — a 12ms spread across runs weeks apart, so
/// the cap rather than variance — and fell back to the deterministic plan. The
/// fifth finished a live plan in 20,832ms and used it. The cap sat inside the
/// planner's own latency, which wasted the turn and, because a killed phase
/// reports no usage, withheld every deep run's token total. This pins the override
/// that fixes it, and pins that it moved nothing else.
#[test]
fn shipped_deep_lane_gives_the_planner_sixty_seconds() {
    use crate::manifest::LaneCatalog;

    let catalog = LaneCatalog::from_toml(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../bench/lanes.toml"
        ))
        .expect("the shipped lane catalog is readable"),
    )
    .expect("the shipped lane catalog parses");

    let phases = catalog.lanes["deep"].phase_budget;
    assert_eq!(phases.plan_cap(), Duration::from_secs(60));

    // The budget the loop actually spends, not just the parsed field.
    let parent = RunBudget::from_duration(Duration::from_secs(335));
    let plan = plan_phase_budget(&phases, &parent)
        .remaining()
        .unwrap_or_default();
    assert!(plan > Duration::from_secs(59), "plan={plan:?}");
    assert!(plan <= Duration::from_secs(60), "plan={plan:?}");

    // One number moved; every other phase keeps the value it always had.
    assert_eq!(phases.exec_cap(), Duration::from_secs(150));
    assert_eq!(phases.smart_first_exec_cap(), Duration::from_secs(240));
    assert_eq!(phases.verify_cap(), Duration::from_secs(30));
    assert_eq!(phases.verify_retry_cap(), Duration::from_secs(15));

    // The compiled default is untouched: this is a catalog override, not a new
    // global default, so lanes and direct paths that opt out still get 25s.
    assert_eq!(
        PhaseBudgetPolicy::default().plan_cap(),
        Duration::from_secs(25)
    );
}

/// The arithmetic raising the plan cap nearly broke.
///
/// Phase budgets are drawn from the parent's REMAINING time, so a longer plan is
/// subtracted from every phase after it. Measuring each phase against a fresh
/// parent — which is all the test above does — cannot see that. This walks the
/// budget down the way the loop actually spends it, with real overhead in between.
///
/// At the 300s total this override was first written against, a 60s plan left
/// verify on `min(30 - B - T, 30)`: ten seconds instead of thirty once the
/// baseline and per-attempt tests took ten seconds each. Raising the lane total by
/// the same 35s the cap took buys that back.
#[test]
fn the_raised_plan_cap_does_not_starve_the_phases_after_it() {
    let phases = PhaseBudgetPolicy {
        plan_cap_seconds: 60,
        ..PhaseBudgetPolicy::default()
    };
    let spec = plain_deep_spec();
    assert!(!needs_smart_first(&spec), "exec must use the plain 150s cap");

    // Baseline test and context assembly before planning; per-attempt tests
    // between exec and verify. Ten seconds each is modest for a real fixture.
    let (baseline, between) = (10u64, 10u64);
    let verify_at = |total: u64| -> Duration {
        let secs = |b: &RunBudget| b.remaining().unwrap_or_default().as_secs();
        let plan = secs(&plan_phase_budget(
            &phases,
            &RunBudget::from_duration(Duration::from_secs(total - baseline)),
        ));
        let exec = secs(&exec_phase_budget(
            &phases,
            &spec,
            &RunBudget::from_duration(Duration::from_secs(total - baseline - plan)),
        ));
        verify_phase_budget(
            &phases,
            &RunBudget::from_duration(Duration::from_secs(
                total - baseline - plan - exec - between,
            )),
        )
        .remaining()
        .unwrap_or_default()
    };

    // The shipped 335s total: the verifier still gets its whole cap.
    let shipped = verify_at(335);
    assert!(
        shipped > Duration::from_secs(29),
        "verify at the shipped 335s total = {shipped:?}"
    );

    // 300s would have starved it. This is the regression the test exists for.
    let starved = verify_at(300);
    assert!(
        starved < Duration::from_secs(15),
        "verify at a 300s total = {starved:?}, so the 335s total is not what saves it"
    );
}

/// The floor itself, pinned to literal seconds rather than to the helper the
/// implementation consults: a test that recomputes the production expression
/// cannot fail when that expression is the thing that is wrong.
///
/// Both bounds are covered. The third scales a small cap down so it stays usable;
/// the clamp holds every large cap at the five seconds the loop required while the
/// floor was hard-coded.
#[test]
fn verify_retry_spawn_floor_scales_down_but_never_exceeds_five_seconds() {
    let floor_for = |cap: u64| {
        PhaseBudgetPolicy {
            verify_retry_cap_seconds: cap,
            ..PhaseBudgetPolicy::default()
        }
        .verify_retry_min_spawn()
    };

    assert_eq!(floor_for(0), Duration::ZERO, "a zero cap disables the retry");
    assert_eq!(floor_for(3), Duration::from_secs(1), "cap 3");
    assert_eq!(floor_for(5), Duration::from_secs(1), "cap 5");
    assert_eq!(floor_for(15), Duration::from_secs(5), "the shipped default");
    assert_eq!(floor_for(60), Duration::from_secs(5), "clamped, not 20s");
    assert_eq!(floor_for(3_600), Duration::from_secs(5), "clamped, not 20m");
}

/// A minimal deep-lane spec for budget tests: no fixture IO, no agent spawn.
fn plain_deep_spec() -> RunSpec {
    RunSpec {
        runner: "zo_claude".into(),
        runner_kind: "zo".into(),
        bin: std::path::PathBuf::from("/bin/true"),
        args: Vec::new(),
        fixture: std::path::PathBuf::from("fixture"),
        prompt: "Add a field.".into(),
        test_command: Some("node --test".into()),
        intended: vec!["src/model.js".into()],
        lane: "deep".into(),
        model: "claude-opus-4-8".into(),
        effort: "max".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 600,
        artifacts_dir: None,
        keep_failed: false,
        ablation: Vec::new(),
        deep: None,
    }
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

/// A phase killed at its wall-clock cap never prints the closing envelope, so
/// whole-buffer parsing charged it zero tokens and the suite reported the
/// surviving phases' partial sum as the run's total. The streamed events it did
/// emit are the evidence that it spent anything at all.
#[test]
fn deep_usage_survives_a_phase_killed_mid_stream() {
    let mut u = DeepUsage::new();
    u.accumulate(concat!(
        r#"{"type":"assistant","text":"reading the parser"}"#,
        "\n",
        r#"{"type":"usage","ctx_tokens":900,"input_tokens":40,"output_tokens":9,"cache_read_tokens":2,"cache_creation_tokens":1}"#,
        "\n",
        r#"{"type":"usage","ctx_tokens":1800,"input_tokens":90,"output_tokens":21,"cache_read_tokens":6,"cache_creation_tokens":3}"#,
        "\n",
        r#"{"type":"assistant","text":"applying the pa"#,
    ), true);

    let v: Value = serde_json::from_str(&u.synthesize(-1)).unwrap();
    assert_eq!(v["usage"]["input_tokens"], 90);
    assert_eq!(v["usage"]["output_tokens"], 21);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 6);
    assert_eq!(v["usage"]["cache_creation_input_tokens"], 3);
}

/// Streamed `usage` events restate session-cumulative counters, so a call must
/// contribute its newest event once. Summing the events instead would inflate a
/// run's cost by however often the runner happened to report progress.
#[test]
fn deep_usage_counts_cumulative_stream_events_once() {
    let mut u = DeepUsage::new();
    u.accumulate(concat!(
        r#"{"type":"usage","ctx_tokens":10,"input_tokens":5,"output_tokens":1,"cache_read_tokens":0,"cache_creation_tokens":0}"#,
        "\n",
        r#"{"type":"usage","ctx_tokens":20,"input_tokens":30,"output_tokens":8,"cache_read_tokens":4,"cache_creation_tokens":2}"#,
        "\n",
        r#"{"type":"result","subtype":"success","is_error":false,"result":"done","num_turns":3,"usage":{"input_tokens":30,"output_tokens":8,"cache_creation_input_tokens":2,"cache_read_input_tokens":4}}"#,
    ), true);

    let v: Value = serde_json::from_str(&u.synthesize(0)).unwrap();
    assert_eq!(v["usage"]["input_tokens"], 30);
    assert_eq!(v["usage"]["output_tokens"], 8);
    assert_eq!(v["usage"]["cache_creation_input_tokens"], 2);
    assert_eq!(v["usage"]["cache_read_input_tokens"], 4);
    assert_eq!(v["iterations"], 3);
}

/// Tool-result events carry a `result` field of their own, so "last object with
/// a `result`" would hand the verifier a tool's output instead of the agent's
/// verdict. The tagged terminal envelope is the only thing that means the turn.
#[test]
fn extract_result_prefers_the_terminal_envelope() {
    let stdout = concat!(
        r#"{"type":"tool_result","result":"3 tests failed"}"#,
        "\n",
        r#"{"type":"result","subtype":"success","is_error":false,"result":"{\"accepted\":true}"}"#,
        "\n",
        r#"{"type":"usage","ctx_tokens":5,"input_tokens":1,"output_tokens":1,"cache_read_tokens":0,"cache_creation_tokens":0}"#,
    );

    assert_eq!(extract_result(stdout), "{\"accepted\":true}");
}

/// The failure that produced a false "16x fewer tokens" report, reduced to its
/// mechanism: one phase reports usage, a later phase is killed and reports none.
/// The loop must publish NO total, because the only number it could publish is
/// the surviving phase's subtotal — which reads exactly like a complete
/// measurement and is not one.
#[test]
fn deep_usage_publishes_no_total_when_a_phase_went_unmeasured() {
    let mut u = DeepUsage::new();
    u.accumulate(
        r#"{"type":"result","usage":{"input_tokens":100,"output_tokens":40,"cache_creation_input_tokens":5,"cache_read_input_tokens":7},"iterations":2}"#,
        true,
    );
    // A killed phase: the process died before printing its closing envelope.
    u.accumulate(
        "{\"type\":\"assistant\",\"text\":\"still working on the pa",
        true,
    );

    let v: Value = serde_json::from_str(&u.synthesize(-1)).unwrap();
    assert!(
        v.get("usage").is_none(),
        "a run with an unmeasured phase must report no usage, got {v}"
    );
    // The turn count is withheld for the same reason as the tokens. This used to
    // assert `iterations == 2`, i.e. the one phase that reported — but the killed
    // phase took turns too, so publishing 2 describes a shorter run than happened.
    assert!(
        v.get("iterations").is_none(),
        "a killed phase's turns are unknown, so no count may be published, got {v}"
    );
}

/// A phase the loop never started is not an unmeasured phase — it is a phase that
/// cost exactly nothing.
///
/// When the budget is already spent `run_command` returns without spawning, and
/// its output is indistinguishable from a process killed before it wrote: empty
/// stdout, `timed_out` set. Treating that as unmeasured voided the token total of
/// a run whose real phases were measured perfectly well, which is the opposite of
/// the honesty this accounting is for.
#[test]
fn a_phase_that_never_spawned_leaves_the_run_measured() {
    let mut u = DeepUsage::new();
    u.accumulate(
        r#"{"type":"result","usage":{"input_tokens":100,"output_tokens":40,"cache_creation_input_tokens":5,"cache_read_input_tokens":7},"iterations":2}"#,
        true,
    );
    // Budget exhausted: no process, no request, no cost.
    u.accumulate("", false);

    let v: Value = serde_json::from_str(&u.synthesize(0)).unwrap();
    assert_eq!(v["usage"]["input_tokens"], 100, "got {v}");
    assert_eq!(v["usage"]["output_tokens"], 40, "got {v}");
    assert_eq!(v["iterations"], 2, "got {v}");
}

/// An absent `permission_denials` field means two different things, and only one
/// of them is zero.
///
/// zo omits the field when there were none, so a call that reported anything else
/// genuinely had zero. A call that reported NOTHING — killed before its envelope —
/// leaves the count unknown, and the empty array `synthesize` emits would read as
/// "observed, and there were none". That reading decides runs: the `verifier_only`
/// gate has no failing-test or no-op clause, so the denial term is the only thing
/// between a run whose refusal blocked the work and a pass.
#[test]
fn only_a_phase_that_reported_nothing_leaves_the_denial_count_unknown() {
    let mut killed = DeepUsage::new();
    killed.accumulate(
        "{\"type\":\"assistant\",\"text\":\"still working on the pa",
        true,
    );
    let v: Value = serde_json::from_str(&killed.synthesize(-1)).unwrap();
    assert_eq!(v["permission_denials_unknown"], true, "got {v}");

    // Reported its turns and tokens, just no denials: that is a real zero.
    let mut reported = DeepUsage::new();
    reported.accumulate(
        r#"{"type":"result","usage":{"input_tokens":10,"output_tokens":5},"iterations":1}"#,
        true,
    );
    let v: Value = serde_json::from_str(&reported.synthesize(0)).unwrap();
    assert!(v.get("permission_denials_unknown").is_none(), "got {v}");

    // A phase that never started reported nothing and cost nothing, so it cannot
    // make the count unknown either.
    let mut unspawned = DeepUsage::new();
    unspawned.accumulate(
        r#"{"type":"result","usage":{"input_tokens":10,"output_tokens":5},"iterations":1}"#,
        true,
    );
    unspawned.accumulate("", false);
    let v: Value = serde_json::from_str(&unspawned.synthesize(0)).unwrap();
    assert!(v.get("permission_denials_unknown").is_none(), "got {v}");
}

/// The context pack carries two different kinds of thing, and only one of them is
/// safe to widen. The strategy preamble tells the agent what to DO — edit first,
/// inspect nothing else, stop after one edit — so giving it to a task the keyword
/// predicate never selected changes that task's behavior and breaks comparability
/// with every run recorded before. Letting a task READ the files it was told to
/// change is not in that category.
///
/// So a non-smart-first task must lose the preamble and keep everything else.
/// This is the fixture shape that went 0/3: a schema propagation prompt none of
/// the keywords match, which used to be handed an empty context.
#[test]
fn non_smart_first_tasks_get_context_without_the_strategy_preamble() {
    let root = std::env::temp_dir().join(format!(
        "zo-deep-plain-context-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::create_dir_all(root.join("test")).unwrap();
    fs::write(
        root.join("src/model.js"),
        "class Money {\n  constructor(amount) {\n    this.amount = amount;\n  }\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("test/model.test.js"),
        "const assert = require('node:assert/strict');\n",
    )
    .unwrap();

    let task = "Add a required 'currency' field to Money and propagate it through serialize, validate, and the API DTO.";
    let intended = vec!["src/".to_string()];

    let spec = RunSpec {
        runner: "zo_claude".into(),
        runner_kind: "zo".into(),
        bin: std::path::PathBuf::from("/bin/true"),
        args: Vec::new(),
        fixture: root.clone(),
        prompt: task.into(),
        test_command: Some("node --test".into()),
        intended: intended.clone(),
        lane: "deep".into(),
        model: "claude-opus-4-8".into(),
        effort: "max".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 300,
        artifacts_dir: None,
        keep_failed: false,
        ablation: Vec::new(),
        deep: None,
    };
    // Guards the premise: this really is a task the predicate does not select, so
    // the loop passes `false` here and the assertions below describe production.
    assert!(!needs_smart_first(&spec));

    let context = exec_context_pack(&root, &intended, "baseline red", task, false);

    // The behavior-changing preamble, and every directive inside it, stays out.
    assert!(!context.contains("## Smart-first hard-task strategy"));
    assert!(!context.contains("First action: edit/write"));
    assert!(!context.contains("Stop after the focused source edit"));
    assert!(!context.contains("state-machine scan"));

    // Everything that merely lets the task see the code it must change stays in.
    assert!(context.contains("## Baseline objective signal"));
    assert!(context.contains("baseline red"));
    assert!(context.contains("## Editable target file snapshots"));
    assert!(context.contains("### src/model.js"));
    assert!(context.contains("class Money"));
    assert!(context.contains("## Relevant tests / assertions"));
    assert!(context.contains("test/model.test.js"));

    let _ = fs::remove_dir_all(&root);
}

/// The three planner outcomes the diagnostics have to tell apart, read straight
/// off what a planner turn leaves behind.
///
/// A phase killed at its cap never prints its closing envelope, so it has no
/// result text; a planner that finished can still answer with nothing. Both end
/// up on the fallback plan, so the plan itself cannot distinguish them.
#[test]
fn planner_signals_separate_running_out_of_time_from_answering_nothing() {
    // Killed mid-stream: no terminal envelope, so no result text to extract.
    let killed = PlanTurn::observe(
        true,
        &extract_result("{\"type\":\"assistant\",\"text\":\"## Ste"),
    );
    assert!(killed.timed_out);
    assert!(!killed.result_present);

    // Finished inside its cap, but the envelope carried only whitespace.
    let said_nothing = PlanTurn::observe(false, &extract_result(r#"{"type":"result","result":"  "}"#));
    assert!(!said_nothing.timed_out);
    assert!(!said_nothing.result_present);

    // Finished and answered.
    let answered = PlanTurn::observe(
        false,
        &extract_result(r#"{"type":"result","result":"1. edit src/a.js"}"#),
    );
    assert!(!answered.timed_out);
    assert!(answered.result_present);
}

/// The distinction has to survive into `deep.diagnostics`, because the JSON is
/// the only place a later plan-cap decision can read it from.
///
/// `plan_recovered` is true both when the deterministic plan is taken up front
/// and when a live plan comes back unusable — that conflation is why these
/// fields were added. Recording them as plain `bool` reinstated it from the
/// other side: the deterministic path, having no planner at all, would report
/// the same `false`/`false` as a planner that ran and returned nothing.
#[test]
fn a_run_with_no_planner_is_not_reported_as_a_planner_that_answered_nothing() {
    let diagnostics = |plan_turn: Option<PlanTurn>| {
        serde_json::to_value(DeepDiagnostics {
            plan_recovered: true,
            plan_timed_out: plan_turn.map(|turn| turn.timed_out),
            plan_result_present: plan_turn.map(|turn| turn.result_present),
            ..DeepDiagnostics::default()
        })
        .expect("diagnostics serialize")
    };

    // Deterministic plan: no planner turn exists to describe.
    // `.get` rather than indexing: indexing yields `Null` for an absent key too,
    // so it could not tell "reported as not applicable" from "never emitted".
    let deterministic = diagnostics(None);
    assert_eq!(
        deterministic.get("plan_timed_out"),
        Some(&Value::Null),
        "got {deterministic}"
    );
    assert_eq!(
        deterministic.get("plan_result_present"),
        Some(&Value::Null),
        "got {deterministic}"
    );

    // A live planner that returned nothing: same `plan_recovered`, different run.
    let empty_answer = diagnostics(Some(PlanTurn::observe(false, "")));
    assert_eq!(empty_answer["plan_timed_out"], false, "got {empty_answer}");
    assert_eq!(empty_answer["plan_result_present"], false, "got {empty_answer}");

    assert_ne!(
        deterministic, empty_answer,
        "two different planner outcomes must not serialize identically"
    );

    // And the killed planner stays distinct from both.
    let killed = diagnostics(Some(PlanTurn::observe(true, "")));
    assert_eq!(killed["plan_timed_out"], true, "got {killed}");
    assert_ne!(killed, empty_answer);
}

/// `true(1)`, wherever this platform keeps it.
///
/// The budget-only fixtures above hard-code `/bin/true` because they never
/// spawn it; macOS ships the binary only under `/usr/bin`, so a test that
/// actually runs the agent has to look it up.
fn stub_agent_bin() -> std::path::PathBuf {
    ["/usr/bin/true", "/bin/true"]
        .iter()
        .map(std::path::PathBuf::from)
        .find(|path| path.exists())
        .expect("a `true` binary to stand in for the agent")
}

/// A deep spec whose "agent" is `true(1)`: it spawns, exits 0, and prints
/// nothing. That is exactly the planner-answered-nothing case, with a real
/// process behind it.
fn spec_with_stub_agent(work: &Path, prompt: &str, intended: Vec<String>) -> RunSpec {
    RunSpec {
        runner: "zo_claude".into(),
        runner_kind: "zo".into(),
        bin: stub_agent_bin(),
        args: Vec::new(),
        fixture: work.to_path_buf(),
        prompt: prompt.into(),
        test_command: Some("true".into()),
        intended,
        lane: "deep".into(),
        model: "claude-opus-4-8".into(),
        effort: "max".into(),
        objective_gate: "test_and_diff".into(),
        diff_policy: "intended_paths_only".into(),
        timeout_seconds: 60,
        artifacts_dir: None,
        keep_failed: false,
        ablation: Vec::new(),
        deep: None,
    }
}

fn stub_work_dir(tag: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "zo-deep-loop-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/helper.js"), "export function helper() {}\n").unwrap();
    root
}

/// Run the loop against the stub agent with `parent_seconds` of total budget.
///
/// The total is a parameter because the planning phase is drawn from what is
/// left after a 60s validation reserve: below that the planner is never spawned
/// at all, which is a different outcome from one that ran.
fn run_stub_loop(work: &Path, spec: &RunSpec, parent_seconds: u64) -> DeepDiagnostics {
    let cfg = DeepConfig {
        max_attempts: 1,
        phase_budget: PhaseBudgetPolicy::default(),
    };
    let budget = RunBudget::from_duration(Duration::from_secs(parent_seconds));
    run_deep_loop(spec, work, &cfg, &budget)
        .expect("the deep loop completes against a stub agent")
        .verdict
        .diagnostics
}

/// The shipped deep lane's total, which leaves the planner its whole cap.
const SHIPPED_DEEP_TOTAL_SECONDS: u64 = 335;

/// The live-planner branch, recorded by the loop itself rather than by a helper
/// called next to it.
///
/// `PlanTurn::observe` being correct is not the same claim as the loop storing
/// what it observed — the fields are filled inside a function that spawns agent
/// turns, and that gap is where the first version of this change went wrong.
#[test]
fn the_loop_reports_a_live_planner_that_returned_nothing() {
    let root = stub_work_dir("live-planner");
    let spec = spec_with_stub_agent(&root, "Rename one helper.", vec!["src/helper.js".into()]);
    assert!(
        !needs_smart_first(&spec),
        "this spec must take the live-planner branch"
    );

    let diagnostics = run_stub_loop(&root, &spec, SHIPPED_DEEP_TOTAL_SECONDS);

    // The planner ran and finished inside its cap, but said nothing.
    assert_eq!(diagnostics.plan_timed_out, Some(false));
    assert_eq!(diagnostics.plan_result_present, Some(false));

    let _ = fs::remove_dir_all(&root);
}

/// A planner the clock never let start is not a planner that ran out of its own
/// time, and the loop must not report it as one.
///
/// `run_command` returns without spawning once the budget is spent, and that
/// output is byte-identical to a process killed mid-write: empty stdout with
/// `timed_out` set. Reading it as a planner timeout would aim a plan-cap change
/// at a phase that never executed — the exact mistake `plan_timed_out` exists to
/// prevent. `spawned` is what tells them apart, and the usage accounting one
/// line above already relies on it.
#[test]
fn a_planner_the_budget_never_started_is_not_reported_as_one_that_timed_out() {
    let root = stub_work_dir("starved-planner");
    let spec = spec_with_stub_agent(&root, "Rename one helper.", vec!["src/helper.js".into()]);
    assert!(!needs_smart_first(&spec));

    // 60s is exactly the validation reserve, so the planning phase is left zero.
    let diagnostics = run_stub_loop(&root, &spec, 60);

    assert_eq!(
        diagnostics.plan_timed_out, None,
        "no planner process ran, so it cannot have timed out"
    );
    assert_eq!(diagnostics.plan_result_present, None);
    // The run really did run out of time — that fact belongs to the aggregate,
    // which is why attributing it to the planner was both wrong and redundant.
    assert!(diagnostics.phase_timed_out);

    let _ = fs::remove_dir_all(&root);
}

/// The deterministic branch, where no planner turn exists to describe.
///
/// This is the case a plain `bool` got wrong: both fields stayed `false`, which
/// is the same output the test above produces — so the two runs, which differ in
/// whether a planner ran at all, were indistinguishable in the JSON.
#[test]
fn the_loop_reports_no_planner_turn_when_it_takes_the_deterministic_plan() {
    let root = stub_work_dir("deterministic");
    let spec = spec_with_stub_agent(
        &root,
        "Implement a streaming CSV parser across arbitrary chunks.",
        vec!["src/".into()],
    );
    assert!(
        needs_smart_first(&spec),
        "this spec must take the deterministic-plan branch"
    );

    let diagnostics = run_stub_loop(&root, &spec, SHIPPED_DEEP_TOTAL_SECONDS);

    assert_eq!(diagnostics.plan_timed_out, None, "no planner turn ran");
    assert_eq!(diagnostics.plan_result_present, None, "no planner turn ran");

    let _ = fs::remove_dir_all(&root);
}
