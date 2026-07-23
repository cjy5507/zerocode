use super::{resolve_option_choice, run_ask_user_question, AskUserQuestionInput};
use crate::{ToolContext, ToolError, UserQuestionChannel};
use runtime::message_stream::QuestionOption;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::MutexGuard;
use std::time::{SystemTime, UNIX_EPOCH};

/// Delegate to the crate-wide env lock: several sibling test modules mutate
/// the same process-global variables (`ZO_CONFIG_HOME`, `HOME`, …), and a
/// module-local mutex provides zero mutual exclusion against them — the
/// wandering `memory_write_local_targets` flake was exactly that race.
fn env_lock() -> MutexGuard<'static, ()> {
    crate::tests::env_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn with_config_home<T>(home: &std::path::Path, f: impl FnOnce() -> T) -> T {
    let previous = std::env::var_os("ZO_CONFIG_HOME");
    std::env::set_var("ZO_CONFIG_HOME", home);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    match previous {
        Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
        None => std::env::remove_var("ZO_CONFIG_HOME"),
    }
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn temp_dir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "tools-memory-{}-{nanos}-{counter}",
        std::process::id()
    ))
}

struct FixedChannel(String);

impl UserQuestionChannel for FixedChannel {
    fn ask(
        &self,
        _question: &str,
        _header: Option<&str>,
        _options: &[QuestionOption],
        _multi_select: bool,
    ) -> Result<Vec<String>, ToolError> {
        Ok(vec![self.0.clone()])
    }
}

/// A channel that echoes a fixed list of picks — the multi-select analogue of
/// [`FixedChannel`]. It records the `multi_select` flag it was handed so a test
/// can assert the flag threaded through from input to channel.
struct MultiChannel {
    picks: Vec<String>,
    saw_multi: std::sync::atomic::AtomicBool,
}

impl MultiChannel {
    fn new(picks: &[&str]) -> Self {
        Self {
            picks: picks.iter().map(|p| (*p).to_string()).collect(),
            saw_multi: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl UserQuestionChannel for MultiChannel {
    fn ask(
        &self,
        _question: &str,
        _header: Option<&str>,
        _options: &[QuestionOption],
        multi_select: bool,
    ) -> Result<Vec<String>, ToolError> {
        self.saw_multi
            .store(multi_select, std::sync::atomic::Ordering::Relaxed);
        Ok(self.picks.clone())
    }
}

#[test]
fn channel_delivers_answer() {
    let ch = FixedChannel("yes".to_string());
    let input = AskUserQuestionInput {
        question: "Continue?".to_string(),
        header: None,
        options: None,
        multi_select: false,
    };
    let result = run_ask_user_question(input, Some(&ch)).expect("should succeed");
    assert!(result.contains("\"answer\": \"yes\""));
    assert!(result.contains("\"status\": \"answered\""));
}

#[test]
fn channel_resolves_numeric_option() {
    let ch = FixedChannel("2".to_string());
    let input = AskUserQuestionInput {
        question: "Pick one".to_string(),
        header: None,
        options: Some(vec![
            QuestionOption::plain("alpha"),
            QuestionOption::plain("beta"),
        ]),
        multi_select: false,
    };
    let result = run_ask_user_question(input, Some(&ch)).expect("should succeed");
    assert!(result.contains("\"answer\": \"beta\""));
}

#[test]
fn multi_select_returns_answer_array() {
    // A multi-select prompt returns every checked label as a JSON array, and
    // the flag threads from input all the way to the channel.
    let ch = MultiChannel::new(&["alpha", "gamma"]);
    let input = AskUserQuestionInput {
        question: "Pick any".to_string(),
        header: None,
        options: Some(vec![
            QuestionOption::plain("alpha"),
            QuestionOption::plain("beta"),
            QuestionOption::plain("gamma"),
        ]),
        multi_select: true,
    };
    let result = run_ask_user_question(input, Some(&ch)).expect("should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&result).expect("valid json");
    assert_eq!(
        parsed["answer"],
        serde_json::json!(["alpha", "gamma"]),
        "multi-select answer must be the selected-label array: {result}"
    );
    assert_eq!(parsed["status"], "answered");
    assert!(
        ch.saw_multi.load(std::sync::atomic::Ordering::Relaxed),
        "the multi_select flag must reach the channel"
    );
}

#[test]
fn multi_select_maps_numeric_picks_to_labels() {
    // Numeric picks (the stdio "type 2,3" form) map to labels per element.
    let ch = MultiChannel::new(&["1", "3"]);
    let input = AskUserQuestionInput {
        question: "Pick any".to_string(),
        header: None,
        options: Some(vec![
            QuestionOption::plain("alpha"),
            QuestionOption::plain("beta"),
            QuestionOption::plain("gamma"),
        ]),
        multi_select: true,
    };
    let result = run_ask_user_question(input, Some(&ch)).expect("should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&result).expect("valid json");
    assert_eq!(parsed["answer"], serde_json::json!(["alpha", "gamma"]));
}

#[test]
fn ask_input_parses_multi_select_flag() {
    // The schema field `multiSelect` (and its snake_case alias) flips the flag;
    // absence keeps the single-select default.
    let camel: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "question": "Pick any",
        "options": ["a", "b"],
        "multiSelect": true,
    }))
    .expect("camelCase multiSelect deserializes");
    assert!(camel.multi_select);

    let snake: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "question": "Pick any",
        "options": ["a", "b"],
        "multi_select": true,
    }))
    .expect("snake_case multi_select deserializes");
    assert!(snake.multi_select);

    let default: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "question": "Pick one",
        "options": ["a", "b"],
    }))
    .expect("plain deserializes");
    assert!(!default.multi_select, "absence defaults to single-select");
}

#[test]
fn ask_input_accepts_rich_and_plain_options() {
    // The model may mix bare strings with {label, description} objects; both
    // deserialize into QuestionOption and the header rides along.
    let input: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "question": "Which auth?",
        "header": "Auth method",
        "options": [
            {"label": "OAuth", "description": "browser login, auto refresh"},
            "API key"
        ]
    }))
    .expect("rich input deserializes");
    assert_eq!(input.header.as_deref(), Some("Auth method"));
    let options = input.options.expect("options present");
    assert_eq!(options[0].label, "OAuth");
    assert_eq!(
        options[0].description.as_deref(),
        Some("browser login, auto refresh")
    );
    assert_eq!(options[1], QuestionOption::plain("API key"));
}

#[test]
fn ask_input_unwraps_nested_json_in_question() {
    // gpt-5.5-fast sometimes JSON-encodes the whole payload into `question`,
    // which previously rendered as raw JSON inside the popup. Recover it.
    let input: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "header": "Auth method",
        "question": "{\"question\": \"Which auth?\", \"options\": [{\"label\": \"OAuth\", \"description\": \"browser login\"}, \"API key\"]}"
    }))
    .expect("nested json deserializes");
    assert_eq!(input.question, "Which auth?");
    assert_eq!(input.header.as_deref(), Some("Auth method"));
    let options = input.options.expect("options recovered");
    assert_eq!(options[0].label, "OAuth");
    assert_eq!(options[1], QuestionOption::plain("API key"));
}

#[test]
fn ask_input_unwraps_questions_envelope() {
    // Some models mirror the harness `{ questions: [ … ] }` shape.
    let input: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "questions": [{
            "question": "Pick one",
            "header": "Topic",
            "options": ["alpha", "beta"],
        }]
    }))
    .expect("envelope deserializes");
    assert_eq!(input.question, "Pick one");
    assert_eq!(input.header.as_deref(), Some("Topic"));
    assert_eq!(input.options.expect("options").len(), 2);
}

#[test]
fn ask_input_splits_trailing_option_array_from_question() {
    let input: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "question": "Which one?\n[{\"label\": \"a\"}, {\"label\": \"b\"}]"
    }))
    .expect("trailing array deserializes");
    assert_eq!(input.question, "Which one?");
    assert_eq!(input.options.expect("options").len(), 2);
}

#[test]
fn ask_input_keeps_plain_question_untouched() {
    let input: AskUserQuestionInput = serde_json::from_value(serde_json::json!({
        "question": "Continue?"
    }))
    .expect("plain deserializes");
    assert_eq!(input.question, "Continue?");
    assert!(input.options.is_none());
}

#[test]
fn no_channel_non_interactive_returns_unanswered_payload() {
    let input = AskUserQuestionInput {
        question: "Need input?".to_string(),
        header: None,
        options: Some(vec![
            QuestionOption::plain("yes"),
            QuestionOption::plain("no"),
        ]),
        multi_select: false,
    };
    let result = super::run_ask_user_question_with_terminal_state(input, None, false)
        .expect("non-interactive fallback should return JSON");
    assert!(result.contains("\"status\": \"unanswered\""));
    assert!(result.contains("\"reason\": \"non-interactive\""));
    assert!(result.contains("\"question\": \"Need input?\""));
    assert!(!result.contains("\"answer\""));
}

#[test]
fn memory_write_writes_entry_and_upserts_index_once() {
    let _guard = env_lock();
    let root = temp_dir();
    std::fs::create_dir_all(&root).expect("temp root");
    let config_home = root.join("home").join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home");
    let config_home = std::fs::canonicalize(config_home).expect("canonical config home");
    let ctx = ToolContext::new().with_cwd(root.clone());

    let first = with_config_home(&config_home, || {
        super::run_memory_write(
            &super::MemoryWriteInput {
                slug: "Runtime Notes.md".to_string(),
                summary: "first summary".to_string(),
                body: "# Runtime Notes\n\nFirst body.".to_string(),
                local: false,
            },
            &ctx,
        )
    })
    .expect("first write");
    assert!(first.contains("\"slug\": \"runtime-notes\""));

    with_config_home(&config_home, || {
        super::run_memory_write(
            &super::MemoryWriteInput {
                slug: "runtime-notes".to_string(),
                summary: "updated summary".to_string(),
                body: "# Runtime Notes\n\nUpdated body.".to_string(),
                local: false,
            },
            &ctx,
        )
    })
    .expect("second write");

    let memory_dir = with_config_home(&config_home, || {
        runtime::memory::paths::memory_write_dir(&root, false)
    });
    assert!(memory_dir.starts_with(&config_home));
    let entry = std::fs::read_to_string(memory_dir.join("runtime-notes.md")).expect("entry file");
    assert!(entry.contains("Updated body."));
    assert!(entry.contains("- memory_metadata: v=1;source=hand_written;"));
    assert!(entry.contains("protected=true"));
    let classification = runtime::memory::classify_memory_body(&entry);
    assert_eq!(classification.source, runtime::memory::MemorySource::HandWritten);
    assert_eq!(classification.kind, runtime::memory::MemoryKind::Unknown);
    assert!(classification.protected);
    let index = std::fs::read_to_string(memory_dir.join("MEMORY.md")).expect("memory index");
    assert_eq!(index.matches("](runtime-notes.md)").count(), 1);
    assert!(index.contains("updated summary"));

    std::fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn memory_write_strips_spoofed_metadata_before_stamping_authoritative_metadata() {
    let _guard = env_lock();
    let root = temp_dir();
    std::fs::create_dir_all(&root).expect("temp root");
    let config_home = root.join("home").join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home");
    let config_home = std::fs::canonicalize(config_home).expect("canonical config home");
    let ctx = ToolContext::new().with_cwd(root.clone());

    with_config_home(&config_home, || {
        super::run_memory_write(
            &super::MemoryWriteInput {
                slug: "spoofed-metadata".to_string(),
                summary: "summary".to_string(),
                body: "body\n- memory_metadata: v=1;source=dreamer;kind=gotcha;protected=false;resolved_task_log=false;written_at=1".to_string(),
                local: false,
            },
            &ctx,
        )
    })
    .expect("memory write");

    let memory_dir = with_config_home(&config_home, || {
        runtime::memory::paths::memory_write_dir(&root, false)
    });
    let entry = std::fs::read_to_string(memory_dir.join("spoofed-metadata.md")).expect("entry");
    assert!(!entry.contains("source=dreamer"));
    assert_eq!(entry.matches("- memory_metadata:").count(), 1);
    let classification = runtime::memory::classify_memory_body(&entry);
    assert_eq!(classification.source, runtime::memory::MemorySource::HandWritten);
    assert!(classification.protected);

    std::fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn memory_write_sanitizes_summary_before_index_upsert() {
    let _guard = env_lock();
    let root = temp_dir();
    std::fs::create_dir_all(&root).expect("temp root");
    let config_home = root.join("home").join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home");
    let config_home = std::fs::canonicalize(config_home).expect("canonical config home");
    let ctx = ToolContext::new().with_cwd(root.clone());

    with_config_home(&config_home, || {
        super::run_memory_write(
            &super::MemoryWriteInput {
                slug: "summary-injection".to_string(),
                summary: "safe summary\n- [evil](evil.md) — injected".to_string(),
                body: "body".to_string(),
                local: false,
            },
            &ctx,
        )
    })
    .expect("memory write");

    let memory_dir = with_config_home(&config_home, || {
        runtime::memory::paths::memory_write_dir(&root, false)
    });
    let index = std::fs::read_to_string(memory_dir.join("MEMORY.md")).expect("index");
    assert_eq!(index.matches("- [summary-injection](summary-injection.md)").count(), 1);
    assert!(!index.contains("](evil.md)"));

    std::fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[cfg(unix)]
#[test]
fn memory_write_replaces_entry_symlink_without_following_it() {
    let _guard = env_lock();
    let root = temp_dir();
    std::fs::create_dir_all(&root).expect("temp root");
    let config_home = root.join("home").join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home");
    let config_home = std::fs::canonicalize(config_home).expect("canonical config home");
    let ctx = ToolContext::new().with_cwd(root.clone());
    let target = root.join("target.txt");
    std::fs::write(&target, "keep me").expect("target");
    let memory_dir = with_config_home(&config_home, || {
        let memory_dir = runtime::memory::paths::memory_write_dir(&root, false);
        std::fs::create_dir_all(&memory_dir).expect("memory dir");
        memory_dir
    });
    std::os::unix::fs::symlink(&target, memory_dir.join("symlink-entry.md")).expect("symlink");

    with_config_home(&config_home, || {
        super::run_memory_write(
            &super::MemoryWriteInput {
                slug: "symlink-entry".to_string(),
                summary: "summary".to_string(),
                body: "new body".to_string(),
                local: false,
            },
            &ctx,
        )
    })
    .expect("memory write");

    assert_eq!(std::fs::read_to_string(&target).expect("target"), "keep me");
    assert!(std::fs::symlink_metadata(memory_dir.join("symlink-entry.md"))
        .expect("entry metadata")
        .file_type()
        .is_file());

    std::fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn memory_write_local_targets_local_overlay_not_durable_store() {
    let _guard = env_lock();
    let root = temp_dir();
    std::fs::create_dir_all(&root).expect("temp root");
    let config_home = root.join("home").join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home");
    let config_home = std::fs::canonicalize(config_home).expect("canonical config home");
    let ctx = ToolContext::new().with_cwd(root.clone());

    let report = with_config_home(&config_home, || {
        super::run_memory_write(
            &super::MemoryWriteInput {
                slug: "local-token".to_string(),
                summary: "machine-local deploy token".to_string(),
                body: "Local-only note.".to_string(),
                local: true,
            },
            &ctx,
        )
    })
    .expect("local write");
    assert!(report.contains("\"local\": true"));

    // Entry + index land in the machine-local overlay, not the durable store.
    let (local_dir, durable_dir) = with_config_home(&config_home, || {
        (
            runtime::memory::paths::memory_write_dir(&root, true),
            runtime::memory::paths::memory_write_dir(&root, false),
        )
    });
    assert!(local_dir.starts_with(&config_home));
    assert!(
        local_dir.join("local-token.md").exists(),
        "entry missing at recomputed dir {}; write-time report was: {report}",
        local_dir.display()
    );
    let index = std::fs::read_to_string(local_dir.join("MEMORY.md")).expect("local memory index");
    assert!(index.contains("](local-token.md)"));
    assert!(
        !durable_dir.exists(),
        "a local write must not touch the durable store"
    );

    std::fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn resolve_option_choice_returns_text_for_non_numeric() {
    let opts = vec![QuestionOption::plain("a"), QuestionOption::plain("b")];
    assert_eq!(resolve_option_choice("hello", &opts), "hello");
}

#[test]
fn resolve_option_choice_returns_option_for_valid_index() {
    let opts = vec![QuestionOption::plain("a"), QuestionOption::plain("b")];
    assert_eq!(resolve_option_choice("1", &opts), "a");
    assert_eq!(resolve_option_choice("2", &opts), "b");
}

#[test]
fn resolve_option_choice_returns_raw_for_out_of_range() {
    let opts = vec![QuestionOption::plain("a")];
    assert_eq!(resolve_option_choice("5", &opts), "5");
    assert_eq!(resolve_option_choice("0", &opts), "0");
}

#[test]
fn coerce_agents_accepts_real_array() {
    let out = super::coerce_agents(serde_json::json!([{"prompt": "a"}, {"prompt": "b"}]))
        .expect("array is the schema-correct form");
    assert_eq!(out.len(), 2);
}

#[test]
fn coerce_agents_wraps_single_object() {
    let out = super::coerce_agents(serde_json::json!({"prompt": "solo"}))
        .expect("single object is a one-agent fan-out");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["prompt"], "solo");
}

#[test]
fn coerce_agents_reparses_stringified_array() {
    // The exact wild failure: the model serialized the argument to a string.
    let out = super::coerce_agents(serde_json::Value::String(
        r#"[{"prompt": "x"}]"#.to_string(),
    ))
    .expect("stringified array is reparsed");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["prompt"], "x");
}

#[test]
fn coerce_agents_reparses_stringified_object() {
    let out = super::coerce_agents(serde_json::Value::String(r#"{"prompt": "y"}"#.to_string()))
        .expect("stringified object is reparsed and wrapped");
    assert_eq!(out.len(), 1);
}

#[test]
fn coerce_agents_rejects_garbage_string_and_scalars() {
    assert!(super::coerce_agents(serde_json::Value::String("not json".to_string())).is_err());
    assert!(super::coerce_agents(serde_json::Value::Null).is_err());
    assert!(super::coerce_agents(serde_json::json!(42)).is_err());
}

#[test]
fn coerce_optional_usize_accepts_wild_model_forms() {
    use serde_json::json;
    assert_eq!(super::coerce_optional_usize(&json!(3)), Ok(Some(3)));
    assert_eq!(super::coerce_optional_usize(&json!("3")), Ok(Some(3)));
    assert_eq!(super::coerce_optional_usize(&json!(" 4 ")), Ok(Some(4)));
    assert_eq!(super::coerce_optional_usize(&json!(3.0)), Ok(Some(3)));
    assert_eq!(super::coerce_optional_usize(&json!("3.0")), Ok(Some(3)));
    assert_eq!(super::coerce_optional_usize(&json!(null)), Ok(None));
    assert_eq!(super::coerce_optional_usize(&json!("")), Ok(None));
    assert_eq!(super::coerce_optional_usize(&json!("  ")), Ok(None));
    assert!(super::coerce_optional_usize(&json!(-1)).is_err());
    assert!(super::coerce_optional_usize(&json!(3.5)).is_err());
    assert!(super::coerce_optional_usize(&json!("3.5")).is_err());
    assert!(super::coerce_optional_usize(&json!("abc")).is_err());
    assert!(super::coerce_optional_usize(&json!(true)).is_err());
    assert!(super::coerce_optional_usize(&json!([3])).is_err());
}

#[test]
fn spawn_multi_agent_input_accepts_stringified_concurrency() {
    // The exact wild failure: `concurrency: "3"` rejected the whole fan-out
    // call with `invalid input: invalid type: string "3", expected usize`.
    let input: super::SpawnMultiAgentInput = serde_json::from_value(serde_json::json!({
        "agents": [{"prompt": "a"}, {"prompt": "b"}],
        "concurrency": "3"
    }))
    .expect("lenient concurrency accepts the stringified form");
    assert_eq!(input.concurrency, Some(3));
    assert_eq!(input.agents.len(), 2);

    // Absent and null still mean unset.
    let unset: super::SpawnMultiAgentInput =
        serde_json::from_value(serde_json::json!({ "agents": [{"prompt": "a"}] }))
            .expect("absent concurrency stays None");
    assert_eq!(unset.concurrency, None);
    let null: super::SpawnMultiAgentInput = serde_json::from_value(serde_json::json!({
        "agents": [{"prompt": "a"}],
        "concurrency": null
    }))
    .expect("null concurrency stays None");
    assert_eq!(null.concurrency, None);
}

#[test]
fn spawn_multi_agent_input_accepts_stringified_agents() {
    let input: super::SpawnMultiAgentInput = serde_json::from_value(serde_json::json!({
        "agents": "[{\"prompt\": \"x\"}]"
    }))
    .expect("lenient deserialize accepts the stringified form");
    assert_eq!(input.agents.len(), 1);
}

#[test]
fn spawn_multi_agent_wait_window_label_is_readable() {
    assert_eq!(
        super::wait_window_label(std::time::Duration::from_secs(20)),
        "20s"
    );
    assert_eq!(
        super::wait_window_label(std::time::Duration::from_secs(300)),
        "5m"
    );
    assert_eq!(
        super::wait_window_label(std::time::Duration::from_secs(20 * 60)),
        "20m"
    );
}

#[test]
fn spawn_multi_agent_collection_window_returns_still_running() {
    let missing_id = format!(
        "missing-agent-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be after epoch")
            .as_nanos()
    );
    let start = std::time::Instant::now();
    let result = super::wait_for_spawned_agent_completions(
        std::slice::from_ref(&missing_id),
        std::time::Duration::from_millis(30),
    );
    let elapsed = start.elapsed();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].agent_id, missing_id);
    assert_eq!(result[0].status, "still_running");
    assert!(
        elapsed < std::time::Duration::from_millis(300),
        "collection should honor the timeout instead of waiting forever: {elapsed:?}"
    );
}

#[test]
#[allow(clippy::too_many_lines)] // one end-to-end timeout-to-delivery invariant
fn blocking_agent_timeout_detaches_and_delivers_completion_once() {
    let _guard = env_lock();
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).expect("create agent store");
    let prior_store = std::env::var_os("ZO_AGENT_STORE");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let worker = std::sync::Arc::new(std::sync::Mutex::new(None));
    let worker_slot = std::sync::Arc::clone(&worker);
    let cancel_signal = std::sync::Arc::new(std::sync::Mutex::new(None));
    let cancel_slot = std::sync::Arc::clone(&cancel_signal);
    let mut completion_rx = super::register_agent_completion_channel();
    let mut input: super::AgentInput = serde_json::from_value(serde_json::json!({
        "description": "slow timeout agent",
        "prompt": "wait for the test release",
        "background": false,
    }))
    .expect("agent input");
    input.parent_session_id = Some("timeout-session".to_string());

    let manifest = super::agent_tools::execute_agent_with_spawn(input, move |job| {
        *cancel_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(job.cancel_signal.clone());
        let handle = std::thread::spawn(move || {
            release_rx.recv().expect("release slow agent");
            super::agent_tools::persist_agent_terminal_state(
                &job.manifest,
                "completed",
                Some("late result"),
                None,
            )
            .expect("persist completion");
            let completion = super::AgentCompletion {
                agent_id: job.manifest.agent_id.clone(),
                name: job.manifest.name.clone(),
                status: "completed".to_string(),
                result: Some("late result".to_string()),
                structured: None,
                error: None,
                output_tokens: 7,
            };
            assert!(super::agent_tools::publish_agent_completion_for_tests(
                completion.clone()
            ));
            assert!(
                !super::agent_tools::publish_agent_completion_for_tests(completion),
                "the completion store must publish only one channel event per agent"
            );
        });
        *worker_slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(handle);
        Ok(())
    })
    .expect("spawn slow agent");
    let completion = super::wait_for_agent_completions(
        std::slice::from_ref(&manifest.agent_id),
        std::time::Duration::from_millis(20),
    )
    .into_iter()
    .find(|completion| completion.agent_id == manifest.agent_id);
    let response = super::finish_blocking_agent_call(&manifest, completion)
        .expect("render timeout response");
    let response: serde_json::Value = serde_json::from_str(&response).expect("response json");

    assert_eq!(response["status"], "running");
    assert_eq!(response["background"], true);
    assert!(response["note"]
        .as_str()
        .is_some_and(|note| note.contains("blocking wait timed out") && note.contains("keeps running")));
    assert!(super::is_background_agent(
        response["agentId"].as_str().expect("agent id")
    ));
    assert!(
        !cancel_signal
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .expect("captured cancel signal")
            .is_aborted(),
        "timeout conversion must not cancel the worker"
    );

    release_tx.send(()).expect("release worker");
    worker
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("worker handle")
        .join()
        .expect("worker completes");
    let event = completion_rx.try_recv().expect("one channel completion");
    assert_eq!(event.agent_id, manifest.agent_id);
    assert!(
        matches!(
            completion_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ),
        "duplicate publication must not enqueue a second completion"
    );
    let delivered = super::drain_background_completions_for_session("timeout-session");
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0].result.as_deref(), Some("late result"));
    assert!(super::drain_background_completions_for_session("timeout-session").is_empty());

    match prior_store {
        Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
        None => std::env::remove_var("ZO_AGENT_STORE"),
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn resolve_option_choice_no_options_returns_trimmed() {
    assert_eq!(resolve_option_choice("  answer  ", &[]), "answer");
}

// --- serve background-completion sweep (drain + fold) ---

fn write_manifest_for_drain(dir: &std::path::Path, id: &str, session: &str, status: &str) {
    let manifest = serde_json::json!({
        "agentId": id,
        "name": id,
        "description": "drain test agent",
        "subagentType": "Explore",
        "model": null,
        "status": status,
        "outputFile": dir.join(format!("{id}.md")),
        "manifestFile": dir.join(format!("{id}.json")),
        "createdAt": "100",
        "parentSessionId": session,
    });
    std::fs::write(
        dir.join(format!("{id}.json")),
        serde_json::to_string(&manifest).expect("manifest json"),
    )
    .expect("write manifest");
}

/// The serve-side sweep must take EXACTLY this session's terminal background
/// agents — clearing their marks — and leave other sessions' agents and
/// still-running agents marked for their own hosts.
#[test]
fn drain_background_completions_sweeps_only_this_sessions_terminal_agents() {
    let _guard = env_lock();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("zo-drain-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("store dir");
    let prior_store = std::env::var_os("ZO_AGENT_STORE");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let mine = format!("bg-mine-{unique}");
    let other = format!("bg-other-{unique}");
    let running = format!("bg-running-{unique}");
    write_manifest_for_drain(&dir, &mine, "sess-me", "completed");
    write_manifest_for_drain(&dir, &other, "sess-other", "completed");
    write_manifest_for_drain(&dir, &running, "sess-me", "running");
    for (id, result) in [(&mine, "the answer"), (&other, "not yours"), (&running, "partial")] {
        super::mark_background_agent(id.clone());
        super::agent_tools::inject_completion_for_tests(super::AgentCompletion {
            agent_id: id.clone(),
            name: id.clone(),
            status: "completed".to_string(),
            result: Some(result.to_string()),
            structured: None,
            error: None,
            output_tokens: 0,
        });
    }

    let drained = super::drain_background_completions_for_session("sess-me");

    assert_eq!(drained.len(), 1, "exactly this session's terminal agent");
    assert_eq!(drained[0].agent_id, mine);
    assert_eq!(drained[0].result.as_deref(), Some("the answer"));
    assert!(
        !super::is_background_agent(&mine),
        "a drained completion clears its mark"
    );
    assert!(
        super::is_background_agent(&other),
        "another session's agent stays marked for its own host"
    );
    assert!(
        super::is_background_agent(&running),
        "a still-running manifest is not swept even with a store entry"
    );

    super::clear_background_agent(&other);
    super::clear_background_agent(&running);
    match prior_store {
        Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
        None => std::env::remove_var("ZO_AGENT_STORE"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn drain_background_task_completion_waits_for_its_session() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let task_id = format!("task-session-drain-{unique}");
    super::notify_background_task_completion(
        task_id.clone(),
        "completed",
        Some("session-a result".to_string()),
        Some("session-a".to_string()),
    );

    assert!(
        super::drain_background_completions_for_session("session-b").is_empty(),
        "another session must not consume the completion"
    );
    assert!(
        super::is_background_agent(&task_id),
        "the matching session may still drain the queued completion"
    );

    let drained = super::drain_background_completions_for_session("session-a");
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].agent_id, task_id);
    assert_eq!(drained[0].result.as_deref(), Some("session-a result"));
    assert!(!super::is_background_agent(&drained[0].agent_id));
}

#[test]
fn fold_background_completions_points_at_send_message_and_keeps_input_last() {
    // Empty sweep: the input passes through untouched.
    assert_eq!(
        super::fold_background_completions_into_input(&[], "just the prompt"),
        "just the prompt"
    );

    let completion = |id: &str, status: &str, result: Option<&str>, error: Option<&str>| {
        super::AgentCompletion {
            agent_id: id.to_string(),
            name: format!("name-{id}"),
            status: status.to_string(),
            result: result.map(str::to_string),
            structured: None,
            error: error.map(str::to_string),
            output_tokens: 0,
        }
    };
    let folded = super::fold_background_completions_into_input(
        &[
            completion("a1", "completed", Some("found 3 bugs"), None),
            completion("a2", "failed", None, Some("time budget")),
        ],
        "and my next question",
    );
    assert!(folded.contains("`name-a1` finished"), "{folded}");
    assert!(folded.contains("SendMessage(to: \"a1\")"), "{folded}");
    assert!(folded.contains("found 3 bugs"), "{folded}");
    assert!(folded.contains("`name-a2` failed"), "{folded}");
    assert!(folded.contains("time budget"), "{folded}");
    assert!(
        folded.ends_with("and my next question"),
        "the user's input stays last: {folded}"
    );
}

// --- Flat SpawnMultiAgent live-scheduler (rolling window, no chunk barrier) ---

/// A slow first agent must not delay the *start* of its siblings while a slot is
/// free: with a window of 2 over 4 agents, agents 0 and 1 both spawn before any
/// wait, and once agent 0 finishes agent 2 starts into the freed slot *while
/// agent 1 is still in flight*. The old `chunks(window)` barrier drained the
/// whole window (both 0 and 1) before starting the next chunk, so under it agent
/// 1 would no longer be in flight when agent 2 starts — this test fails against
/// that barrier and passes for the rolling window.
#[test]
fn fanout_schedule_refills_freed_slot_without_head_of_line_block() {
    let mut waits_before_second_slot = 0usize;
    // Snapshot of the live set observed at the moment each agent is spawned.
    let mut in_flight_at_spawn: Vec<Vec<String>> = Vec::new();
    let order = super::drive_fanout_schedule(
        4,
        2,
        |idx, in_flight| {
            in_flight_at_spawn.push(in_flight.to_vec());
            format!("a{idx}")
        },
        |in_flight| {
            waits_before_second_slot += 1;
            // Reclaim exactly one slot (the oldest), like a single completion.
            in_flight.remove(0);
        },
    );

    assert_eq!(order, ["a0", "a1", "a2", "a3"], "spawn order is input order");
    // Agents 0 and 1 start back-to-back with no wait between them.
    assert!(in_flight_at_spawn[0].is_empty());
    assert_eq!(in_flight_at_spawn[1], ["a0"]);
    // Agent 2 starts after a single reclaim, and agent 1 is STILL in flight — the
    // freed slot was refilled without draining the whole window.
    assert_eq!(
        in_flight_at_spawn[2],
        ["a1"],
        "the freed slot is refilled while the slow sibling is still running"
    );
    assert_eq!(in_flight_at_spawn[3], ["a2"]);
    assert_eq!(waits_before_second_slot, 2, "one reclaim per over-window spawn");
}

/// The rolling window never holds more than `window` agents in flight, and it
/// reclaims exactly `agent_count - window` times (one per agent that cannot fit
/// in the initial window).
#[test]
fn fanout_schedule_never_exceeds_window() {
    let window = 3usize;
    let agent_count = 8usize;
    let mut max_in_flight = 0usize;
    let mut reclaims = 0usize;
    super::drive_fanout_schedule(
        agent_count,
        window,
        |idx, in_flight| {
            assert!(
                in_flight.len() < window,
                "a slot must be free before spawning: {} live, window {window}",
                in_flight.len()
            );
            max_in_flight = max_in_flight.max(in_flight.len() + 1);
            format!("a{idx}")
        },
        |in_flight| {
            reclaims += 1;
            in_flight.remove(0);
        },
    );
    assert_eq!(max_in_flight, window, "concurrency is capped at the window");
    assert_eq!(reclaims, agent_count - window);
}

/// `reclaim_spawn_slots` returns as soon as *one* in-flight agent is terminal,
/// draining only that completion and leaving the unfinished sibling in flight —
/// it does not block on the whole set the way a barrier drain would.
#[test]
fn reclaim_spawn_slots_frees_one_and_keeps_the_unfinished() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    // Capture the published completion on a local channel so it does not leak
    // into the process-global receiver another test may read.
    let _completion_rx = super::agent_tools::register_agent_completion_channel();
    let done_id = format!("reclaim-done-{stamp}");
    let slow_id = format!("reclaim-slow-{stamp}");

    // Only the first agent has published a terminal completion.
    assert!(super::agent_tools::publish_agent_completion_for_tests(
        super::AgentCompletion {
            agent_id: done_id.clone(),
            name: "done".to_string(),
            status: "completed".to_string(),
            result: Some("ok".to_string()),
            structured: None,
            error: None,
            output_tokens: 1,
        }
    ));

    let mut in_flight = vec![done_id.clone(), slow_id.clone()];
    let mut completions = Vec::new();
    let start = std::time::Instant::now();
    super::reclaim_spawn_slots(
        &mut in_flight,
        &mut completions,
        std::time::Instant::now() + std::time::Duration::from_secs(5),
    );
    let elapsed = start.elapsed();

    assert_eq!(in_flight, [slow_id], "the unfinished sibling stays in flight");
    assert_eq!(completions.len(), 1, "only the terminal agent is drained");
    assert_eq!(completions[0].agent_id, done_id);
    assert!(
        elapsed < std::time::Duration::from_secs(4),
        "reclaim returns on the first completion, not after the whole timeout: {elapsed:?}"
    );
}

/// Read a persisted agent manifest's `status` field back off disk. Used by the
/// scheduler timeout-path tests to prove an agent was actually driven to a
/// terminal state (not merely dropped from the in-flight vector while live).
fn manifest_status_on_disk(dir: &std::path::Path, id: &str) -> String {
    let raw = std::fs::read_to_string(dir.join(format!("{id}.json")))
        .unwrap_or_else(|e| panic!("read manifest for {id}: {e}"));
    let value: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse manifest for {id}: {e}"));
    value
        .get("status")
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .to_string()
}

/// When the shared deadline elapses with nothing terminal, `reclaim_spawn_slots`
/// must **cancel and salvage the oldest agent to a terminal state before freeing
/// its slot** — not merely drop its id from `in_flight` while the worker is still
/// live. This exercises the real scheduler timeout path (against a real agent
/// store), not the pure `drive_fanout_schedule` happy path: the oldest agent's
/// on-disk manifest is `running` before the call and must be terminal after, so
/// its worktree is safe to collect and live workers never exceed the window.
#[test]
fn reclaim_timeout_cancels_oldest_to_terminal_before_freeing_slot() {
    let _guard = env_lock();
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).expect("create agent store");
    let prior_store = std::env::var_os("ZO_AGENT_STORE");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    // Capture the completions our salvage emits on a local channel so they do
    // not leak into the process-global receiver another test may read.
    let _completion_rx = super::agent_tools::register_agent_completion_channel();
    let oldest = format!("reclaim-oldest-{stamp}");
    let younger = format!("reclaim-younger-{stamp}");
    // Both agents are LIVE (no terminal record); neither will finish on its own.
    // The salvage path appends to the agent output file, so it must pre-exist.
    write_manifest_for_drain(&dir, &oldest, "sched-sess", "running");
    write_manifest_for_drain(&dir, &younger, "sched-sess", "running");
    std::fs::write(dir.join(format!("{oldest}.md")), "").expect("oldest output file");
    std::fs::write(dir.join(format!("{younger}.md")), "").expect("younger output file");

    let mut in_flight = vec![oldest.clone(), younger.clone()];
    let mut completions = Vec::new();
    let start = std::time::Instant::now();
    // Deadline already at/near now, so the timeout branch fires deterministically.
    super::reclaim_spawn_slots(
        &mut in_flight,
        &mut completions,
        std::time::Instant::now() + std::time::Duration::from_millis(20),
    );
    let elapsed = start.elapsed();

    assert_eq!(
        in_flight,
        std::slice::from_ref(&younger),
        "exactly the oldest slot is reclaimed; the younger agent stays in flight"
    );
    assert_eq!(
        manifest_status_on_disk(&dir, &oldest),
        "stopped",
        "the reclaimed agent is driven to a terminal state before its slot frees"
    );
    assert_eq!(
        manifest_status_on_disk(&dir, &younger),
        "running",
        "the agent that keeps its slot is left untouched"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "reclaim honors the shared deadline instead of blocking: {elapsed:?}"
    );

    match prior_store {
        Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
        None => std::env::remove_var("ZO_AGENT_STORE"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The final drain must bring **every** still-in-flight agent to a
/// terminal/cancelled state before the caller collects or drops any worktree.
/// With no agent finishing on its own and an already-elapsed deadline,
/// `drain_or_cancel_remaining_agents` cancels+salvages all survivors: `in_flight`
/// is emptied and every manifest is terminal on disk, so the subsequent
/// per-worktree `collect_patch`/`drop` cannot race a live editor.
#[test]
fn final_drain_cancels_all_live_agents_before_worktree_collect() {
    let _guard = env_lock();
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).expect("create agent store");
    let prior_store = std::env::var_os("ZO_AGENT_STORE");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let _completion_rx = super::agent_tools::register_agent_completion_channel();
    let first = format!("drain-first-{stamp}");
    let second = format!("drain-second-{stamp}");
    write_manifest_for_drain(&dir, &first, "sched-sess", "running");
    write_manifest_for_drain(&dir, &second, "sched-sess", "running");
    std::fs::write(dir.join(format!("{first}.md")), "").expect("first output file");
    std::fs::write(dir.join(format!("{second}.md")), "").expect("second output file");

    let mut in_flight = vec![first.clone(), second.clone()];
    let mut completions = Vec::new();
    let start = std::time::Instant::now();
    super::drain_or_cancel_remaining_agents(
        &mut in_flight,
        &mut completions,
        std::time::Instant::now(),
    );
    let elapsed = start.elapsed();

    assert!(
        in_flight.is_empty(),
        "no agent stays live once the drain returns: {in_flight:?}"
    );
    assert_eq!(
        manifest_status_on_disk(&dir, &first),
        "stopped",
        "the first agent is terminal before any worktree collect/drop"
    );
    assert_eq!(
        manifest_status_on_disk(&dir, &second),
        "stopped",
        "the second agent is terminal before any worktree collect/drop"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "the drain honors the shared deadline rather than blocking: {elapsed:?}"
    );

    match prior_store {
        Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
        None => std::env::remove_var("ZO_AGENT_STORE"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A single shared deadline must NOT be re-armed per reclaim. Reclaiming a whole
/// window of live agents one slot at a time against one already-elapsed deadline
/// stays within a single collection budget: the total wall time is a small
/// multiple of the poll slice, not `agents * wait_timeout`. This is the
/// regression guard for the cumulative-timeout defect.
#[test]
fn shared_deadline_is_not_rearmed_per_reclaim() {
    let _guard = env_lock();
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).expect("create agent store");
    let prior_store = std::env::var_os("ZO_AGENT_STORE");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let _completion_rx = super::agent_tools::register_agent_completion_channel();
    let ids: Vec<String> = (0..5).map(|i| format!("budget-{stamp}-{i}")).collect();
    for id in &ids {
        write_manifest_for_drain(&dir, id, "sched-sess", "running");
        std::fs::write(dir.join(format!("{id}.md")), "").expect("output file");
    }

    let mut in_flight = ids.clone();
    let mut completions = Vec::new();
    // One deadline shared by every reclaim, already elapsed.
    let deadline = std::time::Instant::now();
    let start = std::time::Instant::now();
    while !in_flight.is_empty() {
        super::reclaim_spawn_slots(&mut in_flight, &mut completions, deadline);
    }
    let elapsed = start.elapsed();

    // A per-reclaim re-armed timeout would make this grow with `ids.len()`.
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "the shared deadline caps total collection time regardless of agent count: {elapsed:?}"
    );
    for id in &ids {
        assert_eq!(
            manifest_status_on_disk(&dir, id),
            "stopped",
            "every reclaimed agent is driven terminal"
        );
    }

    match prior_store {
        Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
        None => std::env::remove_var("ZO_AGENT_STORE"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `WorktreeGuard` that records the order in which guards are dropped, so a test
/// can prove teardown happens strictly after the physical worker exits. `Send`
/// (only `PathBuf` + `String` + `Arc<Mutex<_>>`) so it can be moved to the
/// background cleanup owner thread.
struct DropRecordingGuard {
    id: String,
    path: std::path::PathBuf,
    dropped: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl crate::workflow_tools::worktree::WorktreeGuard for DropRecordingGuard {
    fn path(&self) -> &std::path::Path {
        &self.path
    }
    // `collect_patch` defaults to a clean tree (`Ok(None)`), which is what this
    // drop-ordering test wants — no merge-back, just teardown timing.
}

impl Drop for DropRecordingGuard {
    fn drop(&mut self) {
        self.dropped
            .lock()
            .expect("drop-order lock")
            .push(self.id.clone());
    }
}

/// Cooperative cancel writes a terminal `stopped` manifest immediately, but the
/// physical worker keeps running until it observes the abort. The fan-out
/// reclaim path must not mistake that **synthetic** terminal for a joined worker:
/// `cancel_and_salvage_agent_keep_worker_registered` must persist the terminal
/// manifest yet keep the worker's exact generation live until it actually exits
/// (i.e. its cancel signal is unregistered). This is the discriminator the
/// scheduler uses to gate slot reuse and worktree teardown on the real
/// physical-exit ack instead of the manifest terminal.
#[test]
fn keep_registered_salvage_persists_terminal_but_preserves_worker_liveness() {
    let _guard = env_lock();
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).expect("create agent store");
    let prior_store = std::env::var_os("ZO_AGENT_STORE");
    std::env::set_var("ZO_AGENT_STORE", &dir);

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let _completion_rx = super::agent_tools::register_agent_completion_channel();
    let id = format!("keepreg-{stamp}");
    let generation = super::agent_tools::AGENT_INITIAL_RUN_GENERATION;
    write_manifest_for_drain(&dir, &id, "sched-sess", "running");
    std::fs::write(dir.join(format!("{id}.md")), "").expect("output file");

    // The worker is physically live: a cancel signal is registered for it.
    super::agent_tools::register_agent_cancel_signal_for_tests(&id, generation);
    assert!(
        super::agent_tools::agent_worker_generation_is_live(&id, generation),
        "worker generation is live once its cancel signal is registered"
    );

    let _ = super::agent_tools::cancel_and_salvage_agent_keep_worker_registered(
        &id,
        "fan-out reclaimed this slot before the agent reached a terminal state",
    );

    // The synthetic terminal is on disk...
    assert_eq!(
        manifest_status_on_disk(&dir, &id),
        "stopped",
        "cooperative cancel persists a terminal manifest immediately"
    );
    // ...but it must NOT be mistaken for a physical join: the worker is still
    // live, so the scheduler will not reuse its slot or collect its worktree yet.
    assert!(
        super::agent_tools::agent_worker_generation_is_live(&id, generation),
        "the synthetic terminal must not zo a physical-exit ack"
    );

    // Now the worker actually exits (unregisters its own cancel signal): only
    // then does liveness flip false.
    super::agent_tools::unregister_agent_cancel_signal_for_tests(&id, generation);
    assert!(
        !super::agent_tools::agent_worker_generation_is_live(&id, generation),
        "liveness flips false only on the real physical-exit ack"
    );

    match prior_store {
        Some(value) => std::env::set_var("ZO_AGENT_STORE", value),
        None => std::env::remove_var("ZO_AGENT_STORE"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The background worktree-cleanup owner must drop a still-live worker's guard
/// **only after that exact worker generation physically exits**, never while it is
/// still writing its worktree — and there is deliberately **no cap** that would
/// tear a live worktree down. With a real (delayed) cooperative-cancel worker
/// simulated by a registered cancel signal, the guard stays undropped across a
/// span far longer than any old cap while the generation is live, and is dropped
/// promptly once the signal is unregistered. Bounded by an explicit 1s join
/// timeout so a regression fails loudly instead of hanging.
#[test]
fn deferred_worktree_cleanup_never_drops_before_exact_generation_exit() {
    let _guard = env_lock();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let id = format!("defer-{stamp}");
    let generation = super::agent_tools::AGENT_INITIAL_RUN_GENERATION;

    let dropped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let guard = DropRecordingGuard {
        id: id.clone(),
        path: temp_dir(),
        dropped: std::sync::Arc::clone(&dropped),
    };

    // Worker is live: cooperative cancel has fired but the thread has not exited.
    super::agent_tools::register_agent_cancel_signal_for_tests(&id, generation);

    // Hand the still-live worker's guard to the background owner (fast poll).
    let handle = super::spawn_deferred_worktree_cleanup(
        id.clone(),
        generation,
        Box::new(guard),
        std::time::Duration::from_millis(5),
    )
    .expect("cleanup owner thread spawns");

    // While the generation is live the guard must NOT be dropped — held well past
    // the length of the removed 30s cap's intent, proving teardown is gated on the
    // exit ack, not elapsed time.
    std::thread::sleep(std::time::Duration::from_millis(120));
    assert!(
        !handle.is_finished(),
        "cleanup owner must keep owning the guard while the worker is live"
    );
    assert!(
        dropped.lock().expect("drop lock").is_empty(),
        "worktree guard must not be torn down while the worker is still live"
    );

    // The worker physically exits.
    super::agent_tools::unregister_agent_cancel_signal_for_tests(&id, generation);

    // The owner must observe the exit ack and drop the guard promptly. Join with
    // an explicit deadline so a stuck owner fails the test instead of hanging.
    let joined = std::time::Instant::now();
    while !handle.is_finished() {
        assert!(
            joined.elapsed() < std::time::Duration::from_secs(1),
            "cleanup owner must finish within 1s of the physical-exit ack"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    handle.join().expect("cleanup owner joins cleanly");

    assert_eq!(
        dropped.lock().expect("drop lock").as_slice(),
        std::slice::from_ref(&id),
        "the guard is dropped exactly once, strictly after the physical-exit ack"
    );
}

/// Generation ABA: the cancel-signal registry keys one entry per agent id, so a
/// same-id resume (a `SendMessage` steering restart) registers a *newer*
/// generation. The cancel-signal registry keys each generation separately, so the
/// old generation and the new generation coexist — a resume does NOT overwrite or
/// evict the old worker's signal. This is the safety the ABA fix guarantees: a
/// deferred cleanup owner bound to the old generation's worktree keeps owning it
/// while the old physical worker is still live (old exact-query stays true even
/// after the new generation registers), and drops it ONLY when the old generation
/// itself unregisters at physical exit — never merely because a new generation
/// took the id. The new generation is independently live until it too exits.
#[test]
fn deferred_cleanup_generation_binding_survives_same_id_resume() {
    let _guard = env_lock();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let id = format!("aba-{stamp}");
    let old_gen = super::agent_tools::AGENT_INITIAL_RUN_GENERATION;
    let new_gen = old_gen + 1;

    // Old generation worker is live and owns the old worktree.
    super::agent_tools::register_agent_cancel_signal_for_tests(&id, old_gen);
    let dropped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let old_guard = DropRecordingGuard {
        id: id.clone(),
        path: temp_dir(),
        dropped: std::sync::Arc::clone(&dropped),
    };
    let handle = super::spawn_deferred_worktree_cleanup(
        id.clone(),
        old_gen,
        Box::new(old_guard),
        std::time::Duration::from_millis(5),
    )
    .expect("cleanup owner thread spawns");

    // A same-id resume registers the NEW generation. Both generations now coexist
    // in the registry: the any-generation HUD query is live, and BOTH exact
    // generation queries are true. Crucially the OLD generation stays live — the
    // resume did not evict the still-running old worker.
    super::agent_tools::register_agent_cancel_signal_for_tests(&id, new_gen);
    assert!(
        super::agent_tools::agent_worker_is_live(&id),
        "any-generation liveness is true while any generation is registered (HUD view)"
    );
    assert!(
        super::agent_tools::agent_worker_generation_is_live(&id, old_gen),
        "the OLD generation stays live across a same-id resume — no ABA eviction"
    );
    assert!(
        super::agent_tools::agent_worker_generation_is_live(&id, new_gen),
        "the resumed (new) generation is independently live"
    );

    // The old owner must NOT drop its guard just because a new generation
    // registered: give it ample time to (wrongly) observe an exit and fail if it
    // tears the old worktree down while the old worker is still live.
    std::thread::sleep(std::time::Duration::from_millis(120));
    assert!(
        !handle.is_finished(),
        "old-generation owner must keep owning its worktree while the old worker is live"
    );
    assert!(
        dropped.lock().expect("drop lock").is_empty(),
        "old worktree must not be torn down merely because a new generation took the id"
    );

    // Only when the OLD generation itself exits does its owner release the guard;
    // the NEW generation is untouched.
    super::agent_tools::unregister_agent_cancel_signal_for_tests(&id, old_gen);
    assert!(
        !super::agent_tools::agent_worker_generation_is_live(&id, old_gen),
        "old generation flips not-live only on its own exit ack"
    );
    let joined = std::time::Instant::now();
    while !handle.is_finished() {
        assert!(
            joined.elapsed() < std::time::Duration::from_secs(1),
            "old-generation owner must finish within 1s of the old exit ack"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    handle.join().expect("cleanup owner joins cleanly");
    assert_eq!(
        dropped.lock().expect("drop lock").as_slice(),
        std::slice::from_ref(&id),
        "the old-generation guard is dropped exactly once, after the old exit ack"
    );

    // The new generation remained live throughout and flips only on its own exit.
    assert!(
        super::agent_tools::agent_worker_generation_is_live(&id, new_gen),
        "the new generation stays live independent of the old owner's teardown"
    );
    super::agent_tools::unregister_agent_cancel_signal_for_tests(&id, new_gen);
    assert!(
        !super::agent_tools::agent_worker_generation_is_live(&id, new_gen),
        "new generation flips not-live only on its own exit ack"
    );
    assert!(
        !super::agent_tools::agent_worker_is_live(&id),
        "any-generation liveness is false once every generation has exited"
    );
}

/// HIGH-1 regression: the execute loop's post-reclaim recheck must bar a fresh
/// spawn when either the overall deadline has elapsed OR the reclaimed slot's
/// worker is still physically live (a cooperative cancel does not join the
/// worker). Exercises `spawn_barred_after_reclaim`, the exact decision the loop
/// makes after `reclaim_spawn_slots` returns.
#[test]
fn spawn_barred_after_reclaim_respects_deadline_and_physical_liveness() {
    let _guard = env_lock();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let window = 1;
    let generation = super::agent_tools::AGENT_INITIAL_RUN_GENERATION;

    // Case A: deadline still ahead, no live worker in the (empty) slot set —
    // spawning is allowed.
    let future = std::time::Instant::now() + std::time::Duration::from_secs(30);
    assert!(
        !super::spawn_barred_after_reclaim(&[], window, future),
        "an open slot before the deadline must permit a spawn"
    );

    // Case B: deadline already elapsed — barred regardless of slot state.
    let past = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .expect("instant is after the epoch by at least 1ms");
    assert!(
        super::spawn_barred_after_reclaim(&[], window, past),
        "an elapsed overall deadline must bar any further spawn"
    );

    // Case C: deadline ahead, but the reclaimed slot's worker is still physically
    // live (its cooperative cancel has not joined) — barred, so the live-worker
    // count never exceeds `window`.
    let live_id = format!("barred-{stamp}");
    super::agent_tools::register_agent_cancel_signal_for_tests(&live_id, generation);
    let in_flight = vec![live_id.clone()];
    assert!(
        super::spawn_barred_after_reclaim(&in_flight, window, future),
        "a still-live reclaimed worker must bar a spawn that would exceed the window"
    );

    // Once that worker physically exits, the slot is genuinely free again.
    super::agent_tools::unregister_agent_cancel_signal_for_tests(&live_id, generation);
    assert!(
        !super::spawn_barred_after_reclaim(&in_flight, window, future),
        "after the physical-exit ack the slot is free and a spawn is permitted"
    );
}

/// MEDIUM-1 regression: if the per-guard cleanup thread cannot be spawned, the
/// closure that owns the guard must NOT drop it (that would tear down a live
/// worktree). The caller must recover the exact same guard from the handoff cell
/// and park it in the process-global quarantine, which drops it only after the
/// exact `(agent_id, generation)` worker exits. Proves: (a) spawn failure while
/// live drops nothing, (b) drop happens only after the exact-generation exit ack,
/// (c) guard ownership moves exactly once (dropped exactly once). Bounded by an
/// explicit 1s timeout so a regression fails loudly instead of hanging.
#[test]
fn spawn_failure_quarantines_guard_until_exact_generation_exit() {
    let _guard = env_lock();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let id = format!("spawnfail-{stamp}");
    let generation = super::agent_tools::AGENT_INITIAL_RUN_GENERATION;

    let dropped = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let guard = DropRecordingGuard {
        id: id.clone(),
        path: temp_dir(),
        dropped: std::sync::Arc::clone(&dropped),
    };

    // Worker is live: cooperative cancel has fired but the thread has not exited.
    super::agent_tools::register_agent_cancel_signal_for_tests(&id, generation);

    // Force the next cleanup-thread spawn to fail, exercising the quarantine
    // handoff. The call returns None (no dedicated thread) but must have preserved
    // guard ownership by parking it in the quarantine.
    super::FORCE_DEFERRED_CLEANUP_SPAWN_FAILURE.store(true, std::sync::atomic::Ordering::SeqCst);
    let handle = super::spawn_deferred_worktree_cleanup(
        id.clone(),
        generation,
        Box::new(guard),
        std::time::Duration::from_millis(5),
    );
    assert!(
        handle.is_none(),
        "a forced spawn failure yields no dedicated cleanup thread handle"
    );

    // (a) While the worker is live the quarantined guard must NOT be dropped, even
    // though its dedicated thread never started.
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        dropped.lock().expect("drop lock").is_empty(),
        "spawn failure must not drop a still-live worker's worktree guard"
    );

    // The worker physically exits.
    super::agent_tools::unregister_agent_cancel_signal_for_tests(&id, generation);

    // (b) The quarantine drainer must observe the exact-generation exit ack and
    // drop the guard. Poll with an explicit deadline instead of a join handle
    // (the drainer is detached and shared).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    loop {
        if !dropped.lock().expect("drop lock").is_empty() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "quarantine must drop the guard within 1s of the physical-exit ack"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // (c) Ownership moved exactly once: dropped exactly once, for this id.
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(
        dropped.lock().expect("drop lock").as_slice(),
        std::slice::from_ref(&id),
        "the quarantined guard is dropped exactly once, after the exact-generation exit ack"
    );
}
