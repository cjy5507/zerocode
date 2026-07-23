//! Bin-target unit tests for `main.rs`, moved out verbatim (original
//! one-level indentation retained) so the production entry point stays
//! readable — the inline module body outweighed production code ~6:1.

use super::cli_args::format_unknown_slash_command;
use super::cli_args::{
    parse_args, resolve_model_alias, CliAction, CliInputFormat, CliOutputFormat,
};
use super::formatting::{
    format_compact_report, format_cost_report, format_model_report, format_model_switch_report,
    format_permissions_report, format_permissions_switch_report, format_resume_report,
    format_unknown_slash_command_message, render_resume_usage,
};
use super::git_helpers::{
    parse_git_status_branch, parse_git_status_metadata_for, parse_git_workspace_summary,
    GitWorkspaceSummary,
};
use super::resume::{run_resume_command, StatusUsage};
use super::session::{build_runtime_plugin_state_with_loader, LiveCli};
use super::{
    build_bughunter_prompt, build_council_prompt, build_distill_prompt, build_issue_prompt,
    build_pr_prompt, build_runtime_with_plugin_state, build_ultraplan_prompt,
    create_managed_session_handle, filter_tool_specs,
    format_commit_preflight_report, format_commit_skipped_report,
    format_status_report, format_tool_call_start, format_tool_result,
    permission_policy, print_help_to, push_output_block, render_config_report,
    render_diff_report_for, render_memory_report, render_repl_help, resolve_session_reference,
    response_to_events, slash_command_completion_candidates_with_sessions, status_context,
    validate_no_args, CliToolExecutor,
    DEFAULT_MODEL,
};
use crate::session_registry::SessionScope;
use api::{MessageResponse, OutputContentBlock, Usage};
use commands::resume_supported_slash_commands;
use commands::SlashCommand;
use plugins::{
    PluginManager, PluginManagerConfig, PluginTool, PluginToolDefinition, PluginToolPermission,
};
use runtime::{
    AssistantEvent, ConfigLoader, ContentBlock, ConversationMessage, MessageRole, PermissionMode,
    Session, ToolExecutor,
};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tools::GlobalToolRegistry;

fn registry_with_plugin_tool() -> GlobalToolRegistry {
    GlobalToolRegistry::with_plugin_tools(vec![PluginTool::new(
        "plugin-demo@external",
        "plugin-demo",
        PluginToolDefinition {
            name: "plugin_echo".to_string(),
            description: Some("Echo plugin payload".to_string()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"],
                "additionalProperties": false
            }),
        },
        "echo".to_string(),
        Vec::new(),
        PluginToolPermission::WorkspaceWrite,
        None,
    )])
    .expect("plugin tool registry should build")
}

fn temp_dir() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // Keep test sessions/state out of the developer's real ~/.zo.
    crate::isolate_global_zo_home_for_tests();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos();
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "zo-cli-{}-{}-{seq}",
        std::process::id(),
        nanos
    ))
}

fn git(args: &[&str], cwd: &Path) {
    let status = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .status()
        .expect("git command should run");
    assert!(
        status.success(),
        "git command failed: git {}",
        args.join(" ")
    );
}

fn env_lock() -> MutexGuard<'static, ()> {
    crate::test_env_lock()
}

fn with_current_dir<T>(cwd: &Path, f: impl FnOnce() -> T) -> T {
    let _guard = cwd_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = std::env::current_dir().expect("cwd should load");
    std::env::set_current_dir(cwd).expect("cwd should change");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::env::set_current_dir(previous).expect("cwd should restore");
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn with_env_vars<T>(vars: &[(&str, Option<std::ffi::OsString>)], f: impl FnOnce() -> T) -> T {
    let previous = vars
        .iter()
        .map(|(key, _)| (*key, std::env::var_os(key)))
        .collect::<Vec<_>>();
    for (key, value) in vars {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    for (key, value) in previous {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

#[test]
fn current_cli_cwd_prefers_shell_pwd_when_it_points_to_same_location() {
    let _guard = cwd_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let root = std::env::temp_dir().join(format!(
        "zo-logical-cwd-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let physical = root.join("physical");
    let logical = root.join("logical");
    std::fs::create_dir_all(&physical).expect("physical dir");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&physical, &logical).expect("logical link");

    let previous_dir = std::env::current_dir().expect("cwd should load");
    let previous_pwd = std::env::var_os("PWD");
    std::env::set_current_dir(&physical).expect("switch cwd");
    std::env::set_var("PWD", &logical);

    let resolved = super::current_cli_cwd().expect("cwd should resolve");

    std::env::set_current_dir(previous_dir).expect("cwd restore should succeed");
    match previous_pwd {
        Some(value) => std::env::set_var("PWD", value),
        None => std::env::remove_var("PWD"),
    }
    let _ = std::fs::remove_dir_all(root);

    assert_eq!(resolved, logical);
}

fn write_plugin_fixture(root: &Path, name: &str, include_hooks: bool, include_lifecycle: bool) {
    fs::create_dir_all(root.join(".claude-plugin")).expect("manifest dir");
    if include_hooks {
        fs::create_dir_all(root.join("hooks")).expect("hooks dir");
        fs::write(
            root.join("hooks").join("pre.sh"),
            "#!/bin/sh\nprintf 'plugin pre hook'\n",
        )
        .expect("write hook");
    }
    if include_lifecycle {
        fs::create_dir_all(root.join("lifecycle")).expect("lifecycle dir");
        fs::write(
            root.join("lifecycle").join("init.sh"),
            "#!/bin/sh\nprintf 'init\\n' >> lifecycle.log\n",
        )
        .expect("write init lifecycle");
        fs::write(
            root.join("lifecycle").join("shutdown.sh"),
            "#!/bin/sh\nprintf 'shutdown\\n' >> lifecycle.log\n",
        )
        .expect("write shutdown lifecycle");
    }

    let hooks = if include_hooks {
        ",\n  \"hooks\": {\n    \"PreToolUse\": [\"./hooks/pre.sh\"]\n  }"
    } else {
        ""
    };
    let lifecycle = if include_lifecycle {
        ",\n  \"lifecycle\": {\n    \"Init\": [\"./lifecycle/init.sh\"],\n    \"Shutdown\": [\"./lifecycle/shutdown.sh\"]\n  }"
    } else {
        ""
    };
    fs::write(
            root.join(".claude-plugin").join("plugin.json"),
            format!(
                "{{\n  \"name\": \"{name}\",\n  \"version\": \"1.0.0\",\n  \"description\": \"runtime plugin fixture\"{hooks}{lifecycle}\n}}"
            ),
        )
        .expect("write plugin manifest");
}
#[test]
fn defaults_to_repl_when_no_args() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    assert_eq!(
        parse_args(&[]).expect("args should parse"),
        CliAction::Repl {
            model: DEFAULT_MODEL.to_string(),
            model_pinned: false,
            allowed_tools: None,
            disallowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            mcp_config: None,
            inline: false,
        }
    );
}

#[test]
fn default_permission_mode_uses_project_config_when_env_is_unset() {
    let _guard = env_lock();
    let root = temp_dir();
    let cwd = root.join("project");
    let config_home = root.join("config-home");
    std::fs::create_dir_all(cwd.join(".zo")).expect("project config dir should exist");
    std::fs::create_dir_all(&config_home).expect("config home should exist");
    std::fs::write(
        cwd.join(".zo").join("settings.json"),
        r#"{"permissionMode":"acceptEdits"}"#,
    )
    .expect("project config should write");

    let original_config_home = std::env::var("ZO_CONFIG_HOME").ok();
    let original_permission_mode = std::env::var("ZO_PERMISSION_MODE").ok();
    std::env::set_var("ZO_CONFIG_HOME", &config_home);
    std::env::remove_var("ZO_PERMISSION_MODE");

    let resolved = with_current_dir(&cwd, super::default_permission_mode);

    match original_config_home {
        Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
        None => std::env::remove_var("ZO_CONFIG_HOME"),
    }
    match original_permission_mode {
        Some(value) => std::env::set_var("ZO_PERMISSION_MODE", value),
        None => std::env::remove_var("ZO_PERMISSION_MODE"),
    }
    std::fs::remove_dir_all(root).expect("temp config root should clean up");

    assert_eq!(resolved, PermissionMode::WorkspaceWrite);
}

#[test]
fn project_config_default_is_used_and_env_is_ignored() {
    let _guard = env_lock();
    let root = temp_dir();
    let cwd = root.join("project");
    let config_home = root.join("config-home");
    std::fs::create_dir_all(cwd.join(".zo")).expect("project config dir should exist");
    std::fs::create_dir_all(&config_home).expect("config home should exist");
    std::fs::write(
        cwd.join(".zo").join("settings.json"),
        r#"{"permissionMode":"acceptEdits"}"#,
    )
    .expect("project config should write");

    let original_config_home = std::env::var("ZO_CONFIG_HOME").ok();
    let original_permission_mode = std::env::var("ZO_PERMISSION_MODE").ok();
    std::env::set_var("ZO_CONFIG_HOME", &config_home);
    // The env override is intentionally NOT consulted any more: even a
    // conflicting `read-only` env must lose to the project's configured mode.
    std::env::set_var("ZO_PERMISSION_MODE", "read-only");

    let resolved = with_current_dir(&cwd, super::default_permission_mode);

    match original_config_home {
        Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
        None => std::env::remove_var("ZO_CONFIG_HOME"),
    }
    match original_permission_mode {
        Some(value) => std::env::set_var("ZO_PERMISSION_MODE", value),
        None => std::env::remove_var("ZO_PERMISSION_MODE"),
    }
    std::fs::remove_dir_all(root).expect("temp config root should clean up");

    // `acceptEdits` resolves to workspace-write; the env `read-only` is ignored.
    assert_eq!(resolved, PermissionMode::WorkspaceWrite);
}

#[test]
fn default_permission_mode_uses_read_only_fallback_for_home_cwd() {
    let _guard = env_lock();
    let root = temp_dir();
    let home = root.join("home");
    let config_home = home.join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home should exist");

    let resolved = with_env_vars(
        &[
            ("HOME", Some(home.as_os_str().to_os_string())),
            (
                "ZO_CONFIG_HOME",
                Some(config_home.as_os_str().to_os_string()),
            ),
            ("ZO_HOME", None),
            ("ZO_PERMISSION_MODE", None),
        ],
        || with_current_dir(&home, super::default_permission_mode),
    );

    std::fs::remove_dir_all(root).expect("temp config root should clean up");
    assert_eq!(resolved, PermissionMode::ReadOnly);
}

#[test]
fn configured_full_access_overrides_home_cwd_safe_fallback() {
    let _guard = env_lock();
    let root = temp_dir();
    let home = root.join("home");
    let config_home = home.join(".zo");
    std::fs::create_dir_all(&config_home).expect("config home should exist");
    std::fs::write(
        config_home.join("settings.json"),
        r#"{"permissionMode":"dontAsk"}"#,
    )
    .expect("global config should write");

    let resolved = with_env_vars(
        &[
            ("HOME", Some(home.as_os_str().to_os_string())),
            (
                "ZO_CONFIG_HOME",
                Some(config_home.as_os_str().to_os_string()),
            ),
            ("ZO_HOME", None),
            ("ZO_PERMISSION_MODE", None),
        ],
        || with_current_dir(&home, super::default_permission_mode),
    );

    std::fs::remove_dir_all(root).expect("temp config root should clean up");
    assert_eq!(resolved, PermissionMode::DangerFullAccess);
}

#[test]
fn default_permission_mode_uses_read_only_fallback_for_zo_config_home_parent() {
    let _guard = env_lock();
    let root = temp_dir();
    let protected_home = root.join("global-home");
    let config_home = protected_home.join(".zo");
    let unrelated_home = root.join("unrelated-home");
    std::fs::create_dir_all(&config_home).expect("config home should exist");
    std::fs::create_dir_all(&unrelated_home).expect("unrelated home should exist");

    let resolved = with_env_vars(
        &[
            ("HOME", Some(unrelated_home.as_os_str().to_os_string())),
            (
                "ZO_CONFIG_HOME",
                Some(config_home.as_os_str().to_os_string()),
            ),
            ("ZO_HOME", None),
            ("ZO_PERMISSION_MODE", None),
        ],
        || with_current_dir(&protected_home, super::default_permission_mode),
    );

    std::fs::remove_dir_all(root).expect("temp config root should clean up");
    assert_eq!(resolved, PermissionMode::ReadOnly);
}

#[test]
fn env_permission_mode_is_ignored_and_safe_fallback_wins() {
    let _guard = env_lock();
    let root = temp_dir();
    let protected_home = root.join("global-home");
    let config_home = protected_home.join(".zo");
    let unrelated_home = root.join("unrelated-home");
    std::fs::create_dir_all(&config_home).expect("config home should exist");
    std::fs::create_dir_all(&unrelated_home).expect("unrelated home should exist");

    let resolved = with_env_vars(
        &[
            ("HOME", Some(unrelated_home.as_os_str().to_os_string())),
            (
                "ZO_CONFIG_HOME",
                Some(config_home.as_os_str().to_os_string()),
            ),
            ("ZO_HOME", None),
            // The env override is intentionally NOT consulted: even a permissive
            // `danger-full-access` env must lose to the workspace-risk-aware
            // safe fallback for a protected (config-home) cwd.
            (
                "ZO_PERMISSION_MODE",
                Some(std::ffi::OsString::from("danger-full-access")),
            ),
        ],
        || with_current_dir(&protected_home, super::default_permission_mode),
    );

    std::fs::remove_dir_all(root).expect("temp config root should clean up");
    // No project config + protected cwd ⇒ safe fallback; the env is ignored.
    assert_eq!(resolved, PermissionMode::ReadOnly);
}

#[test]
fn parses_prompt_subcommand() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    let args = vec![
        "prompt".to_string(),
        "hello".to_string(),
        "world".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::Prompt {
            prompt: "hello world".to_string(),
            model_pinned: false,
            model: DEFAULT_MODEL.to_string(),
            output_format: CliOutputFormat::Text,
            allowed_tools: None,
            disallowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            input_format: CliInputFormat::Text,
            mcp_config: None,
            prefill: None,
            no_follow: false,
            session_id: None,
            fallback_model: None,
        }
    );
}

#[test]
fn parses_mcp_config_for_prompt_and_repl_actions() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");

    let prompt_args = vec![
        "--mcp-config".to_string(),
        "external-mcp.json".to_string(),
        "prompt".to_string(),
        "use".to_string(),
        "mcp".to_string(),
    ];
    match parse_args(&prompt_args).expect("prompt args should parse") {
        CliAction::Prompt {
            prompt, mcp_config, ..
        } => {
            assert_eq!(prompt, "use mcp");
            assert_eq!(mcp_config, Some(PathBuf::from("external-mcp.json")));
        }
        other => panic!("expected Prompt, got {other:?}"),
    }

    let repl_args = vec!["--mcp-config=external-mcp.json".to_string()];
    match parse_args(&repl_args).expect("repl args should parse") {
        CliAction::Repl { mcp_config, .. } => {
            assert_eq!(mcp_config, Some(PathBuf::from("external-mcp.json")));
        }
        other => panic!("expected Repl, got {other:?}"),
    }
}

#[test]
fn parses_headless_budget_flags() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    let args = vec![
        "--max-turns".to_string(),
        "8".to_string(),
        "--max-tool-calls=15".to_string(),
        "-p".to_string(),
        "analyze".to_string(),
    ];

    match parse_args(&args).expect("budget flags should parse") {
        CliAction::Prompt {
            max_turns,
            max_tool_calls,
            ..
        } => {
            assert_eq!(max_turns, Some(8));
            assert_eq!(max_tool_calls, Some(15));
        }
        other => panic!("expected Prompt, got {other:?}"),
    }
}

#[test]
fn rejects_zero_budget_flags() {
    let err = parse_args(&[
        "--max-tool-calls".to_string(),
        "0".to_string(),
        "-p".to_string(),
        "analyze".to_string(),
    ])
    .expect_err("zero tool-call budget should be rejected");
    assert!(err.contains("--max-tool-calls"), "message: {err}");
    assert!(err.contains("positive integer"), "message: {err}");

    let err = parse_args(&[
        "--max-turns=0".to_string(),
        "-p".to_string(),
        "analyze".to_string(),
    ])
    .expect_err("zero turn budget should be rejected");
    assert!(err.contains("--max-turns"), "message: {err}");
    assert!(err.contains("positive integer"), "message: {err}");
}

#[test]
fn parses_serve_with_default_bind() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    assert_eq!(
        parse_args(&["serve".to_string()]).expect("serve should parse"),
        CliAction::Serve {
            bind_addr: "127.0.0.1:8787".to_string(),
            model: DEFAULT_MODEL.to_string(),
            allowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
        }
    );
}

#[test]
fn parses_serve_port_and_bind_overrides() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    // --port sets the port on loopback.
    let CliAction::Serve { bind_addr, .. } = parse_args(&[
        "serve".to_string(),
        "--port".to_string(),
        "9100".to_string(),
    ])
    .expect("serve --port should parse") else {
        panic!("expected Serve");
    };
    assert_eq!(bind_addr, "127.0.0.1:9100");
    // --bind wins over the default and takes a full host:port.
    let CliAction::Serve { bind_addr, .. } =
        parse_args(&["serve".to_string(), "--bind=0.0.0.0:1234".to_string()])
            .expect("serve --bind should parse")
    else {
        panic!("expected Serve");
    };
    assert_eq!(bind_addr, "0.0.0.0:1234");
}

#[test]
fn rejects_serve_with_stray_positional_and_bad_port() {
    let _guard = env_lock();
    assert!(parse_args(&["serve".to_string(), "oops".to_string()]).is_err());
    assert!(parse_args(&[
        "serve".to_string(),
        "--port".to_string(),
        "not-a-number".to_string()
    ])
    .is_err());
}

#[test]
fn parses_attach_with_and_without_session_id() {
    let _guard = env_lock();
    assert_eq!(
        parse_args(&["attach".to_string(), "sess-42".to_string()]).expect("attach id parses"),
        CliAction::Attach {
            bind_addr: "127.0.0.1:8787".to_string(),
            session_id: Some("sess-42".to_string()),
            plain: false,
        }
    );
    assert_eq!(
        parse_args(&[
            "attach".to_string(),
            "--port".to_string(),
            "9100".to_string()
        ])
        .expect("attach no-id parses"),
        CliAction::Attach {
            bind_addr: "127.0.0.1:9100".to_string(),
            session_id: None,
            plain: false,
        }
    );
}

#[test]
fn parses_attach_plain_flag_anywhere() {
    let _guard = env_lock();
    // `--plain` selects the line client and is order-independent vs the id.
    assert_eq!(
        parse_args(&[
            "attach".to_string(),
            "--plain".to_string(),
            "sess-9".to_string()
        ])
        .expect("attach --plain parses"),
        CliAction::Attach {
            bind_addr: "127.0.0.1:8787".to_string(),
            session_id: Some("sess-9".to_string()),
            plain: true,
        }
    );
}

#[test]
fn rejects_attach_with_two_session_ids() {
    let _guard = env_lock();
    assert!(parse_args(&["attach".to_string(), "a".to_string(), "b".to_string()]).is_err());
}

#[test]
fn parses_bare_prompt_and_json_output_flag() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    let args = vec![
        "--output-format=json".to_string(),
        "--model".to_string(),
        "claude-opus".to_string(),
        "explain".to_string(),
        "this".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::Prompt {
            prompt: "explain this".to_string(),
            model_pinned: true,
            model: "claude-opus-4-8".to_string(),
            output_format: CliOutputFormat::Json,
            allowed_tools: None,
            disallowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            input_format: CliInputFormat::Text,
            mcp_config: None,
            prefill: None,
            no_follow: false,
            session_id: None,
            fallback_model: None,
        }
    );
}

#[test]
fn parses_streaming_output_format_aliases() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    // `stream-json` is the Claude Code parity name and `ndjson` the
    // Zo alias; both must reach the action as the same variant so the
    // help text (which now advertises stream-json) cannot drift from the
    // parser.
    for alias in ["stream-json", "ndjson"] {
        let args = vec![
            "--output-format".to_string(),
            alias.to_string(),
            "explain".to_string(),
            "this".to_string(),
        ];
        match parse_args(&args).expect("args should parse") {
            CliAction::Prompt { output_format, .. } => assert_eq!(
                output_format,
                CliOutputFormat::Ndjson,
                "alias {alias:?} should map to ndjson"
            ),
            other => panic!("expected prompt action for {alias:?}, got {other:?}"),
        }
    }
}

#[test]
fn accepts_stream_json_input_format_values() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    let ok = parse_args(&[
        "--input-format".to_string(),
        "text".to_string(),
        "hello".to_string(),
    ])
    .expect("text input format should parse");
    assert!(matches!(ok, CliAction::Prompt { .. }));

    for value in ["json", "stream-json", "ndjson"] {
        match parse_args(&[
            "--input-format".to_string(),
            value.to_string(),
            "hello".to_string(),
        ])
        .expect("streaming input format should parse")
        {
            CliAction::Prompt { input_format, .. } => {
                assert_eq!(input_format, CliInputFormat::StreamJson, "alias {value:?}");
            }
            other => panic!("expected prompt action for {value:?}, got {other:?}"),
        }
    }
}

#[test]
fn stream_json_input_format_after_print_requires_no_prompt() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    match parse_args(&[
        "-p".to_string(),
        "--input-format".to_string(),
        "stream-json".to_string(),
        "--output-format".to_string(),
        "stream-json".to_string(),
        "--max-turns".to_string(),
        "2".to_string(),
    ])
    .expect("stream-json stdin prompt should parse")
    {
        CliAction::Prompt {
            prompt,
            input_format,
            output_format,
            max_turns,
            ..
        } => {
            assert_eq!(prompt, "");
            assert_eq!(input_format, CliInputFormat::StreamJson);
            assert_eq!(output_format, CliOutputFormat::Ndjson);
            assert_eq!(max_turns, Some(2));
        }
        other => panic!("expected prompt action, got {other:?}"),
    }
}

#[test]
fn resolves_model_aliases_in_args() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    let args = vec![
        "--model".to_string(),
        "opus".to_string(),
        "explain".to_string(),
        "this".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::Prompt {
            prompt: "explain this".to_string(),
            model_pinned: true,
            model: "claude-opus-4-8".to_string(),
            output_format: CliOutputFormat::Text,
            allowed_tools: None,
            disallowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            input_format: CliInputFormat::Text,
            mcp_config: None,
            prefill: None,
            no_follow: false,
            session_id: None,
            fallback_model: None,
        }
    );
}

#[test]
fn resolves_known_model_aliases() {
    assert_eq!(resolve_model_alias("opus"), "claude-opus-4-8");
    assert_eq!(resolve_model_alias("sonnet"), "claude-sonnet-5");
    assert_eq!(resolve_model_alias("haiku"), "claude-haiku-4-5-20251001");
    assert_eq!(resolve_model_alias("claude-opus"), "claude-opus-4-8");
    assert_eq!(resolve_model_alias("claude-sonnet"), "claude-sonnet-5");
    assert_eq!(
        resolve_model_alias("claude-haiku"),
        "claude-haiku-4-5-20251001"
    );
    // Dot-separated versions are normalized to hyphens
    assert_eq!(resolve_model_alias("claude-opus-4.6"), "claude-opus-4-6");
    assert_eq!(
        resolve_model_alias("claude-sonnet-4.6"),
        "claude-sonnet-4-6"
    );
    assert_eq!(
        resolve_model_alias("claude-haiku-4.5-20251001"),
        "claude-haiku-4-5-20251001"
    );
    // Non-claude models are left unchanged
    assert_eq!(resolve_model_alias("grok-3"), "grok-3");
}

#[test]
fn parses_version_flags_without_initializing_prompt_mode() {
    assert_eq!(
        parse_args(&["--version".to_string()]).expect("args should parse"),
        CliAction::Version
    );
    assert_eq!(
        parse_args(&["-V".to_string()]).expect("args should parse"),
        CliAction::Version
    );
}

#[test]
fn parses_permission_mode_flag() {
    let args = vec!["--permission-mode=read-only".to_string()];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::Repl {
            model: DEFAULT_MODEL.to_string(),
            model_pinned: false,
            allowed_tools: None,
            disallowed_tools: None,
            permission_mode: PermissionMode::ReadOnly,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            mcp_config: None,
            inline: false,
        }
    );
}

#[test]
fn parses_allowed_tools_flags_with_aliases_and_lists() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    let args = vec![
        "--allowedTools".to_string(),
        "read,glob".to_string(),
        "--allowed-tools=write_file".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::Repl {
            model: DEFAULT_MODEL.to_string(),
            model_pinned: false,
            allowed_tools: Some(
                ["glob_search", "read_file", "write_file"]
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            ),
            disallowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            mcp_config: None,
            inline: false,
        }
    );
}

#[test]
fn parses_inline_for_the_main_interactive_command_only() {
    let action = parse_args(&[
        "--permission-mode=read-only".to_string(),
        "--inline".to_string(),
    ])
    .expect("inline repl parses");
    assert!(matches!(action, CliAction::Repl { inline: true, .. }));

    let error = parse_args(&["--inline".to_string(), "status".to_string()])
        .expect_err("inline is not a headless/subcommand flag");
    assert!(error.contains("main interactive command"));

    for args in [
        vec!["--inline".to_string(), "-p".to_string(), "hello".to_string()],
        vec!["-p".to_string(), "hello".to_string(), "--inline".to_string()],
    ] {
        let error = parse_args(&args).expect_err("inline must reject one-shot prompt mode");
        assert!(error.contains("main interactive command"));
    }
}

#[test]
fn rejects_unknown_allowed_tools() {
    let error = parse_args(&["--allowedTools".to_string(), "teleport".to_string()])
        .expect_err("tool should be rejected");
    assert!(error.contains("unsupported tool in --allowedTools: teleport"));
}

#[test]
fn parses_system_prompt_options() {
    let args = vec![
        "system-prompt".to_string(),
        "--cwd".to_string(),
        "/tmp/project".to_string(),
        "--date".to_string(),
        "2026-04-01".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::PrintSystemPrompt {
            cwd: PathBuf::from("/tmp/project"),
            date: "2026-04-01".to_string(),
        }
    );
}

#[test]
fn parses_login_and_logout_subcommands() {
    assert_eq!(
        parse_args(&["login".to_string()]).expect("login should parse"),
        CliAction::Login { provider: None }
    );
    assert_eq!(
        parse_args(&["logout".to_string()]).expect("logout should parse"),
        CliAction::Logout
    );
    assert_eq!(
        parse_args(&["init".to_string()]).expect("init should parse"),
        CliAction::Init
    );
    assert_eq!(
        parse_args(&["agents".to_string()]).expect("agents should parse"),
        CliAction::Agents { args: None }
    );
    assert_eq!(
        parse_args(&["mcp".to_string()]).expect("mcp should parse"),
        CliAction::Mcp { args: None }
    );
    assert_eq!(
        parse_args(&["skills".to_string()]).expect("skills should parse"),
        CliAction::Skills { args: None }
    );
    assert_eq!(
        parse_args(&["agents".to_string(), "--help".to_string()])
            .expect("agents help should parse"),
        CliAction::Agents {
            args: Some("--help".to_string())
        }
    );
}

#[test]
fn parses_single_word_command_aliases_without_falling_back_to_prompt_mode() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    assert_eq!(
        parse_args(&["help".to_string()]).expect("help should parse"),
        CliAction::Help
    );
    assert_eq!(
        parse_args(&["version".to_string()]).expect("version should parse"),
        CliAction::Version
    );
    assert_eq!(
        parse_args(&["status".to_string()]).expect("status should parse"),
        CliAction::Status {
            model: DEFAULT_MODEL.to_string(),
            permission_mode: PermissionMode::DangerFullAccess,
        }
    );
    assert_eq!(
        parse_args(&["sandbox".to_string()]).expect("sandbox should parse"),
        CliAction::Sandbox
    );
}

#[test]
fn single_word_slash_command_names_return_guidance_instead_of_hitting_prompt_mode() {
    let error = parse_args(&["cost".to_string()]).expect_err("cost should return guidance");
    assert!(error.contains("slash command"));
    assert!(error.contains("/cost"));
}

#[test]
fn multi_word_prompt_still_uses_shorthand_prompt_mode() {
    let _guard = env_lock();
    std::env::remove_var("ZO_PERMISSION_MODE");
    assert_eq!(
        parse_args(&["help".to_string(), "me".to_string(), "debug".to_string()])
            .expect("prompt shorthand should still work"),
        CliAction::Prompt {
            prompt: "help me debug".to_string(),
            model: DEFAULT_MODEL.to_string(),
            model_pinned: false,
            output_format: CliOutputFormat::Text,
            allowed_tools: None,
            disallowed_tools: None,
            permission_mode: PermissionMode::DangerFullAccess,
            max_turns: None,
            max_tool_calls: None,
            system_prompt: None,
            append_system_prompt: None,
            verbose: false,
            input_format: CliInputFormat::Text,
            mcp_config: None,
            prefill: None,
            no_follow: false,
            session_id: None,
            fallback_model: None,
        }
    );
}

#[test]
fn parses_direct_agents_mcp_and_skills_slash_commands() {
    assert_eq!(
        parse_args(&["/agents".to_string()]).expect("/agents should parse"),
        CliAction::Agents { args: None }
    );
    assert_eq!(
        parse_args(&["/mcp".to_string(), "show".to_string(), "demo".to_string()])
            .expect("/mcp show demo should parse"),
        CliAction::Mcp {
            args: Some("show demo".to_string())
        }
    );
    assert_eq!(
        parse_args(&["/skills".to_string()]).expect("/skills should parse"),
        CliAction::Skills { args: None }
    );
    assert_eq!(
        parse_args(&["/skills".to_string(), "help".to_string()])
            .expect("/skills help should parse"),
        CliAction::Skills {
            args: Some("help".to_string())
        }
    );
    assert_eq!(
        parse_args(&[
            "/skills".to_string(),
            "install".to_string(),
            "./fixtures/help-skill".to_string()
        ])
        .expect("/skills install should parse"),
        CliAction::Skills {
            args: Some("install ./fixtures/help-skill".to_string())
        }
    );
    let status_action =
        parse_args(&["/status".to_string()]).expect("/status should now be supported from the CLI");
    assert!(matches!(status_action, CliAction::SlashCommand { .. }));
}

#[test]
fn direct_slash_commands_surface_shared_validation_errors() {
    // /compact accepts an optional focus directive (CC parity), so args parse
    // rather than error; a no-arg command (/help) still surfaces the usage error.
    let help_error = parse_args(&["/help".to_string(), "now".to_string()])
        .expect_err("invalid /help shape should be rejected");
    assert!(help_error.contains("Unexpected arguments for /help."));
    assert!(help_error.contains("Usage            /help"));

    let plugins_error = parse_args(&[
        "/plugins".to_string(),
        "list".to_string(),
        "extra".to_string(),
    ])
    .expect_err("invalid /plugins list shape should be rejected");
    assert!(plugins_error.contains("Usage: /plugin list"));
    assert!(plugins_error.contains("Aliases          /plugins, /marketplace"));
}

#[test]
fn formats_unknown_slash_command_with_suggestions() {
    let report = format_unknown_slash_command_message("statsu");
    assert!(report.contains("unknown slash command: /statsu"));
    assert!(report.contains("Did you mean"));
    assert!(report.contains("Use /help"));
}

#[test]
fn parses_resume_flag_with_slash_command() {
    let args = vec![
        "--resume".to_string(),
        "session.jsonl".to_string(),
        "/compact".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::ResumeSession {
            session_path: PathBuf::from("session.jsonl"),
            from_turn: None,
            commands: vec!["/compact".to_string()],
        }
    );
}

#[test]
fn parses_resume_flag_without_path_as_latest_session() {
    assert_eq!(
        parse_args(&["--resume".to_string()]).expect("args should parse"),
        CliAction::ResumeSession {
            session_path: PathBuf::from("latest"),
            from_turn: None,
            commands: vec![],
        }
    );
    assert_eq!(
        parse_args(&["--resume".to_string(), "/status".to_string()])
            .expect("resume shortcut should parse"),
        CliAction::ResumeSession {
            session_path: PathBuf::from("latest"),
            from_turn: None,
            commands: vec!["/status".to_string()],
        }
    );
}

#[test]
fn parses_resume_flag_with_multiple_slash_commands() {
    let args = vec![
        "--resume".to_string(),
        "session.jsonl".to_string(),
        "/status".to_string(),
        "/compact".to_string(),
        "/cost".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::ResumeSession {
            session_path: PathBuf::from("session.jsonl"),
            from_turn: None,
            commands: vec![
                "/status".to_string(),
                "/compact".to_string(),
                "/cost".to_string(),
            ],
        }
    );
}

#[test]
fn rejects_unknown_options_with_helpful_guidance() {
    let error = parse_args(&["--resum".to_string()]).expect_err("unknown option should fail");
    assert!(error.contains("unknown option: --resum"));
    assert!(error.contains("Did you mean --resume?"));
    assert!(error.contains("zo --help"));
}

#[test]
fn parses_resume_flag_with_slash_command_arguments() {
    let args = vec![
        "--resume".to_string(),
        "session.jsonl".to_string(),
        "/export".to_string(),
        "notes.txt".to_string(),
        "/clear".to_string(),
        "--confirm".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::ResumeSession {
            session_path: PathBuf::from("session.jsonl"),
            from_turn: None,
            commands: vec![
                "/export notes.txt".to_string(),
                "/clear --confirm".to_string(),
            ],
        }
    );
}

#[test]
fn parses_resume_flag_with_absolute_export_path() {
    let args = vec![
        "--resume".to_string(),
        "session.jsonl".to_string(),
        "/export".to_string(),
        "/tmp/notes.txt".to_string(),
        "/status".to_string(),
    ];
    assert_eq!(
        parse_args(&args).expect("args should parse"),
        CliAction::ResumeSession {
            session_path: PathBuf::from("session.jsonl"),
            from_turn: None,
            commands: vec!["/export /tmp/notes.txt".to_string(), "/status".to_string()],
        }
    );
}

#[test]
fn filtered_tool_specs_respect_allowlist() {
    let allowed = ["read_file", "grep_search"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let filtered = filter_tool_specs(
        &GlobalToolRegistry::builtin(),
        "claude-sonnet-4-6",
        Some(&allowed),
    );
    let names = filtered
        .into_iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["read_file", "grep_search"]);
}

#[test]
fn filtered_tool_specs_defer_plugin_tools_until_activated() {
    // Plugin schemas are deferred from the default wire advertisement (P2
    // token diet); a ToolSearch load activates them for subsequent requests.
    let registry = registry_with_plugin_tool();
    let filtered = filter_tool_specs(&registry, "claude-sonnet-4-6", None);
    let names = filtered
        .into_iter()
        .map(|definition| definition.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"bash".to_string()));
    assert!(!names.contains(&"plugin_echo".to_string()));

    // The search result itself is irrelevant here — the call's activation
    // side effect (plugin schemas join the advertisement) is what's under test.
    let _ = registry.search("select:plugin_echo", 3, None, None);
    let names = filter_tool_specs(&registry, "claude-sonnet-4-6", None)
        .into_iter()
        .map(|definition| definition.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"plugin_echo".to_string()));
}

#[test]
fn permission_policy_uses_plugin_tool_permissions() {
    let feature_config = runtime::RuntimeFeatureConfig::default();
    let policy = permission_policy(
        PermissionMode::ReadOnly,
        &feature_config,
        &registry_with_plugin_tool(),
    )
    .expect("permission policy should build");
    let required = policy.required_mode_for("plugin_echo");
    assert_eq!(required, PermissionMode::WorkspaceWrite);
}

#[test]
fn shared_help_uses_resume_annotation_copy() {
    let help = commands::render_slash_command_help();
    assert!(help.contains("Slash commands"));
    assert!(help.contains("works with --resume SESSION.jsonl"));
}

#[test]
fn repl_help_includes_shared_commands_and_exit() {
    let help = render_repl_help();
    assert!(help.contains("REPL"));
    assert!(help.contains("/help"));
    assert!(help.contains("Complete commands, modes, and recent sessions"));
    assert!(help.contains("/status"));
    assert!(help.contains("/sandbox"));
    assert!(help.contains("/model [model]"));
    assert!(help.contains("/permissions [read-only|workspace-write|danger-full-access]"));
    assert!(help.contains("/clear [--confirm]"));
    assert!(help.contains("/cost"));
    assert!(help.contains("/resume <session-path>"));
    assert!(help.contains("/config [env|hooks|model|plugins]"));
    assert!(help.contains("/mcp [list|show <server>|auth [list|<server>]|logout <server>|help]"));
    assert!(help.contains("/memory"));
    assert!(help.contains("/init"));
    assert!(help.contains("/diff"));
    assert!(help.contains("/version"));
    assert!(help.contains("/export [file]"));
    assert!(help.contains("/session [list|switch <session-id>|fork [branch-name]]"));
    assert!(help.contains(
        "/plugin [list|install <path>|enable <name>|disable <name>|uninstall <id>|update <id>]"
    ));
    assert!(help.contains("aliases: /plugins, /marketplace"));
    assert!(help.contains("/agents"));
    assert!(help.contains("/skills"));
    assert!(help.contains("/exit"));
    assert!(help
        .contains("Auto-save            ~/.zo/projects/<project>/sessions/<session-id>.jsonl"));
    assert!(help.contains("Resume latest        /resume latest"));
}

#[test]
fn completion_candidates_include_workflow_shortcuts_and_dynamic_sessions() {
    let completions = slash_command_completion_candidates_with_sessions(
        "sonnet",
        Some("session-current"),
        &["session-old".to_string()],
        &[],
    );

    assert!(completions.contains(&"/model claude-sonnet-5".to_string()));
    assert!(completions.contains(&"/permissions workspace-write".to_string()));
    assert!(completions.contains(&"/session list".to_string()));
    assert!(completions.contains(&"/session switch session-current".to_string()));
    assert!(completions.contains(&"/resume session-old".to_string()));
    assert!(completions.contains(&"/mcp list".to_string()));
    assert!(completions.contains(&"/ultraplan ".to_string()));
    assert!(!completions.contains(&"/rename".to_string()));
    assert!(!completions.contains(&"/desktop".to_string()));
}

#[test]
fn completion_candidates_include_prompt_commands() {
    let prompt_commands = vec![commands::PromptCommandDef {
        name: "review-local".to_string(),
        description: Some("Review local changes".to_string()),
        argument_hint: Some("<scope>".to_string()),
        model: None,
        effort: None,
        body: "Review $ARGUMENTS".to_string(),
        allowed_tools: Vec::new(),
        path: PathBuf::from(".zo/commands/review-local.md"),
    }];

    let completions =
        slash_command_completion_candidates_with_sessions("sonnet", None, &[], &prompt_commands);

    assert!(completions.contains(&"/review-local".to_string()));
    assert!(completions.contains(&"/review-local ".to_string()));
}

#[test]
fn startup_banner_mentions_workflow_completions() {
    let _guard = env_lock();
    // Inject dummy credentials so LiveCli can construct without real Anthropic key
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-banner-test");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    let banner = with_current_dir(&root, || {
        LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
        )
        .expect("cli should initialize")
        .startup_banner()
    });

    assert!(banner.contains("/help"));
    assert!(banner.contains("Shift+Enter"));

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

#[test]
fn startup_screen_and_input_label_preserve_core_session_fields() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-startup-screen");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    let (screen, input_label, session_id) = with_current_dir(&root, || {
        let cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
        )
        .expect("cli should initialize");
        let session_id = cli.session.id.clone();
        (
            cli.startup_screen(Some(std::time::Duration::from_millis(42))),
            cli.input_box_label(),
            session_id,
        )
    });

    assert_eq!(screen.model, "claude-sonnet-4-6");
    assert_eq!(screen.permissions, "danger-full-access");
    assert_eq!(screen.session_id, session_id);
    assert_eq!(screen.startup_ms, Some(42));
    assert!(screen
        .autosave_path
        .ends_with(format!("{session_id}.jsonl")));
    assert!(
        input_label.contains("sonnet-4-6"),
        "input label should use model alias: {input_label}"
    );
    assert!(
        input_label.contains(&session_id[..8]),
        "input label should include short session id: {input_label}"
    );

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

#[test]
fn live_cli_resume_report_without_target_returns_usage() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-resume-usage");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    let report = with_current_dir(&root, || {
        let mut cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
        )
        .expect("cli should initialize");
        cli.resume_session_report(None)
            .expect("resume usage should render")
    });

    // With no args, /resume auto-loads the latest session (created by
    // LiveCli::new). If no session exists, it falls back to usage text.
    // In this test, LiveCli::new creates a session, so we get a resume
    // report or usage depending on session directory state.
    assert!(
        report.contains("Session resumed") || report.contains("Usage"),
        "expected resume or usage report, got: {report}"
    );

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

/// Regression: `/name`, `/deep`, `/auto`, `/hunks`, and `/remote` used to fall
/// through every headless REPL sub-dispatcher and surface the raw internal
/// "unhandled slash command" error. Each must now be handled — `/name` sets the
/// display name, `/deep` and `/auto` install the gate, `/hunks` and `/remote`
/// print an honest TUI-only notice — so `handle_repl_command` returns `Ok`.
#[test]
fn headless_repl_handles_previously_unhandled_slash_commands() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-repl-gaps");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    with_current_dir(&root, || {
        let mut cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
        )
        .expect("cli should initialize");

        for input in [
            "/name gaptest",
            "/deep off",
            "/auto off",
            "/hunks",
            "/remote status",
        ] {
            let command = SlashCommand::parse(input)
                .expect("parses cleanly")
                .expect("is a command");
            let quit = cli
                .handle_repl_command(command)
                .unwrap_or_else(|error| panic!("{input} should be handled, got: {error}"));
            assert!(!quit, "{input} must not quit the REPL");
        }

        // `/name gaptest` genuinely set (and persisted) the display name.
        assert_eq!(cli.runtime.session().name.as_deref(), Some("gaptest"));

        // Direct contract of the shared setter: show, then reject an over-long name.
        assert!(
            cli.set_display_name(None)
                .expect("show is Ok")
                .contains("gaptest")
        );
        let too_long = "x".repeat(commands::MAX_SESSION_NAME_CHARS + 1);
        assert!(cli.set_display_name(Some(&too_long)).is_err());
    });

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

#[test]
fn live_cli_session_reports_cover_fork_switch_and_resume() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-session-reports");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    with_current_dir(&root, || {
        let mut cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
        )
        .expect("cli should initialize");
        let original_session = cli.session.id.clone();

        let (list_report, listed) = cli
            .session_command_report(None, None)
            .expect("session list should render");
        assert!(!listed);
        assert!(list_report.contains(&original_session));

        let (fork_report, forked) = cli
            .session_command_report(Some("fork"), Some("incident-review"))
            .expect("session fork should succeed");
        assert!(forked);
        assert!(fork_report.contains("Session forked"));
        assert!(fork_report.contains("incident-review"));

        let forked_session = cli.session.id.clone();
        assert_ne!(forked_session, original_session);

        let (switch_report, switched) = cli
            .session_command_report(Some("switch"), Some(&original_session))
            .expect("session switch should succeed");
        assert!(switched);
        assert!(switch_report.contains("Session switched"));
        assert_eq!(cli.session.id, original_session);

        let resume_report = cli
            .resume_session_report(Some(&forked_session))
            .expect("resume should succeed");
        assert!(resume_report.contains("Session resumed"));
        assert_eq!(cli.session.id, forked_session);
    });

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

#[test]
fn live_cli_clear_session_report_preserves_resume_reference() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-clear-session");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    with_current_dir(&root, || {
        let mut cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
        )
        .expect("cli should initialize");

        let original_session = cli.session.id.clone();
        let dry_run = cli
            .clear_session_report(false)
            .expect("clear without confirm should render guidance");
        assert!(dry_run.contains("/clear --confirm"));
        assert_eq!(cli.session.id, original_session);

        let report = cli
            .clear_session_report(true)
            .expect("clear with confirm should succeed");
        assert!(report.contains("Session cleared"));
        assert!(report.contains(&format!("Previous session {original_session}")));
        assert!(report.contains(&format!("Resume previous  /resume {original_session}")));
        assert_ne!(cli.session.id, original_session);
    });

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

/// `apply_permission_change` must commit its state only after the fallible
/// policy build succeeds, so `LiveCli`'s mode, the shared tool-context permission
/// cell, and the live runtime policy never diverge. A clean policy-build-failure
/// injection would need a seam this constructor does not expose, so this pins
/// the observable transactional contract on the paths the seam supports: a
/// successful switch advances all three in lockstep, and an unchanged-mode call
/// is a no-op that reports without touching state.
#[test]
fn apply_permission_change_commits_mode_and_context_together() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-perm-change");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    with_current_dir(&root, || {
        let mut cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::ReadOnly,
        )
        .expect("cli should initialize");
        assert_eq!(cli.permission_mode, PermissionMode::ReadOnly);

        // Unchanged-mode call: reports, commits nothing new.
        let same = cli
            .apply_permission_change("read-only")
            .expect("no-op switch reports");
        assert!(same.contains("Read"));
        assert_eq!(cli.permission_mode, PermissionMode::ReadOnly);

        // Successful switch: LiveCli mode and the shared tool-context cell
        // advance together (the runtime policy is swapped in the same commit).
        cli.apply_permission_change("workspace-write")
            .expect("switch to workspace-write succeeds");
        assert_eq!(cli.permission_mode, PermissionMode::WorkspaceWrite);
        assert_eq!(
            cli.runtime
                .api_client()
                .tool_registry()
                .context()
                .session_permission_mode(),
            Some(runtime::PermissionMode::WorkspaceWrite),
            "tool-context permission cell commits with the mode"
        );

        // A rejected spelling fails before committing anything.
        let err = cli.apply_permission_change("bogus-mode");
        assert!(err.is_err(), "unknown mode is rejected");
        assert_eq!(
            cli.permission_mode,
            PermissionMode::WorkspaceWrite,
            "a rejected switch leaves the prior mode intact"
        );
    });

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

/// User-selected Plan must reach the model as a per-turn contract, and plain
/// `ReadOnly` must NOT. This exercises the runtime-facing state directly (via
/// `set_plan_selected` → `effective_system_prompt`), not model behavior:
///   - entering Plan appends the "plan is already active / do not call
///     `EnterPlanMode`" contract that prevents the duplicate write-gated denial,
///   - plain `ReadOnly` (never selecting Plan) receives no such contract, so it
///     is not mislabeled as Plan,
///   - leaving Plan clears the contract while the runtime permission mode the
///     user restored is untouched.
#[test]
fn live_cli_plan_selection_gates_the_per_turn_plan_contract() {
    let _guard = env_lock();
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-plan-contract");
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");

    with_current_dir(&root, || {
        // A plain read-only session (Plan never selected) carries no plan
        // contract — read-only must not be mislabeled as Plan.
        let mut cli = LiveCli::new(
            "claude-sonnet-4-6".to_string(),
            true,
            None,
            PermissionMode::ReadOnly,
        )
        .expect("cli should initialize");
        assert!(!cli.plan_selected(), "fresh session is not user-selected Plan");
        let plain_read_only = cli.effective_system_prompt();
        assert!(
            !plain_read_only
                .iter()
                .any(|segment| segment.contains("Plan mode is already active")),
            "plain ReadOnly must not receive the Plan contract"
        );

        // Selecting Plan (as Shift+Tab / `/plan on` do) injects the contract.
        cli.set_plan_selected(true);
        assert!(cli.plan_selected());
        let planning = cli.effective_system_prompt();
        let contract = planning
            .iter()
            .find(|segment| segment.contains("Plan mode is already active"))
            .expect("selecting Plan must inject the Plan contract");
        assert!(
            contract.contains("Do NOT call EnterPlanMode"),
            "Plan contract must forbid the write-gated re-entry tool: {contract}"
        );

        // Leaving Plan (as Shift+Tab off / `/plan off` do) clears the contract.
        cli.set_plan_selected(false);
        assert!(!cli.plan_selected());
        assert!(
            !cli.effective_system_prompt()
                .iter()
                .any(|segment| segment.contains("Plan mode is already active")),
            "leaving Plan must clear the Plan contract"
        );
    });

    fs::remove_dir_all(root).expect("cleanup temp dir");
    std::env::remove_var("ANTHROPIC_API_KEY");
}

#[test]
fn resume_supported_command_list_matches_expected_surface() {
    let names = resume_supported_slash_commands()
        .into_iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    // Public surface is intentionally narrower than the internal spec inventory.
    assert!(
        names.len() >= 25,
        "expected at least 25 resume-supported commands, got {}",
        names.len()
    );
    // Verify key resume commands still exist
    assert!(names.contains(&"help"));
    assert!(names.contains(&"status"));
    assert!(names.contains(&"compact"));
}

#[test]
fn resume_report_uses_sectioned_layout() {
    let report = format_resume_report("session.jsonl", 14, 6);
    assert!(report.contains("Session resumed"));
    assert!(report.contains("Session file     session.jsonl"));
    assert!(report.contains("Messages         14"));
    assert!(report.contains("Turns            6"));
}

#[test]
fn compact_report_uses_structured_output() {
    let compacted = format_compact_report(8, 5, false);
    assert!(compacted.contains("Compact"));
    assert!(compacted.contains("Result           compacted"));
    assert!(compacted.contains("Messages removed 8"));
    let skipped = format_compact_report(0, 3, true);
    assert!(skipped.contains("Result           skipped"));
}

#[test]
fn cost_report_uses_sectioned_layout() {
    let report = format_cost_report(runtime::TokenUsage {
        input_tokens: 20,
        output_tokens: 8,
        cache_creation_input_tokens: 3,
        cache_read_input_tokens: 1,
    });
    assert!(report.contains("Cost"));
    assert!(report.contains("Input tokens     20"));
    assert!(report.contains("Output tokens    8"));
    assert!(report.contains("Cache create     3"));
    assert!(report.contains("Cache read       1"));
    assert!(report.contains("Total tokens     32"));
}

#[test]
fn permissions_report_uses_sectioned_layout() {
    let report = format_permissions_report("workspace-write");
    assert!(report.contains("Permissions"));
    assert!(report.contains("Active mode      workspace-write"));
    assert!(report.contains("Modes"));
    assert!(report.contains("read-only          ○ available Read/search tools only"));
    assert!(report.contains("workspace-write    ● current   Edit files inside the workspace"));
    assert!(report.contains("danger-full-access ○ available Unrestricted tool access"));
}

#[test]
fn permissions_switch_report_is_structured() {
    let report = format_permissions_switch_report("read-only", "workspace-write");
    assert!(report.contains("Permissions updated"));
    assert!(report.contains("Result           mode switched"));
    assert!(report.contains("Previous mode    read-only"));
    assert!(report.contains("Active mode      workspace-write"));
    assert!(report.contains("Applies to       subsequent tool calls"));
}

#[test]
fn init_help_mentions_direct_subcommand() {
    let mut help = Vec::new();
    print_help_to(&mut help).expect("help should render");
    let help = String::from_utf8(help).expect("help should be utf8");
    assert!(help.contains("zo help"));
    assert!(help.contains("zo version"));
    assert!(help.contains("zo status"));
    assert!(help.contains("zo sandbox"));
    assert!(help.contains("zo init"));
    assert!(help.contains("zo agents"));
    assert!(help.contains("zo mcp"));
    assert!(help.contains("zo skills"));
    assert!(help.contains("zo /skills"));
}

#[test]
fn model_report_uses_sectioned_layout() {
    let report = format_model_report("claude-sonnet", 12, 4);
    assert!(report.contains("Model"));
    assert!(report.contains("Current model    claude-sonnet"));
    assert!(report.contains("Session messages 12"));
    assert!(report.contains("Switch models with /model <name>"));
}

#[test]
fn model_switch_report_preserves_context_summary() {
    let report = format_model_switch_report("claude-sonnet", "claude-opus", 9);
    assert!(report.contains("Model updated"));
    assert!(report.contains("Previous         claude-sonnet"));
    assert!(report.contains("Current          claude-opus"));
    assert!(report.contains("Preserved msgs   9"));
}

#[test]
fn status_line_reports_model_and_token_totals() {
    let status = format_status_report(
        "claude-sonnet",
        StatusUsage {
            message_count: 7,
            turns: 3,
            latest: runtime::TokenUsage {
                input_tokens: 5,
                output_tokens: 4,
                cache_creation_input_tokens: 1,
                cache_read_input_tokens: 0,
            },
            cumulative: runtime::TokenUsage {
                input_tokens: 20,
                output_tokens: 8,
                cache_creation_input_tokens: 2,
                cache_read_input_tokens: 1,
            },
            estimated_tokens: 128,
        },
        "workspace-write",
        &super::StatusContext {
            cwd: PathBuf::from("/tmp/project"),
            session_path: Some(PathBuf::from("session.jsonl")),
            loaded_config_files: 2,
            discovered_config_files: 3,
            instruction_file_count: 4,
            project_root: Some(PathBuf::from("/tmp")),
            git_branch: Some("main".to_string()),
            git_summary: GitWorkspaceSummary {
                changed_files: 3,
                staged_files: 1,
                unstaged_files: 1,
                untracked_files: 1,
                conflicted_files: 0,
            },
            sandbox_status: runtime::SandboxStatus::default(),
        },
    );
    assert!(status.contains("Status"));
    assert!(status.contains("Model            claude-sonnet"));
    assert!(status.contains("Permission mode  workspace-write"));
    assert!(status.contains("Messages         7"));
    assert!(status.contains("Latest total     10"));
    assert!(status.contains("Cumulative total 31"));
    assert!(status.contains("Cwd              /tmp/project"));
    assert!(status.contains("Project root     /tmp"));
    assert!(status.contains("Git branch       main"));
    assert!(status.contains("Git state        dirty · 3 files · 1 staged, 1 unstaged, 1 untracked"));
    assert!(status.contains("Changed files    3"));
    assert!(status.contains("Staged           1"));
    assert!(status.contains("Unstaged         1"));
    assert!(status.contains("Untracked        1"));
    assert!(status.contains("Session          session.jsonl"));
    assert!(status.contains("Config files     loaded 2/3"));
    assert!(status.contains("Instruction files 4"));
    assert!(status.contains("Suggested flow   /status → /diff → /commit"));
}

#[test]
fn commit_reports_surface_workspace_context() {
    let summary = GitWorkspaceSummary {
        changed_files: 2,
        staged_files: 1,
        unstaged_files: 1,
        untracked_files: 0,
        conflicted_files: 0,
    };

    let preflight = format_commit_preflight_report(Some("feature/ux"), summary);
    assert!(preflight.contains("Result           ready"));
    assert!(preflight.contains("Branch           feature/ux"));
    assert!(preflight.contains("Workspace        dirty · 2 files · 1 staged, 1 unstaged"));
    assert!(preflight
        .contains("Action           create a git commit from the current workspace changes"));
}

#[test]
fn commit_skipped_report_points_to_next_steps() {
    let report = format_commit_skipped_report();
    assert!(report.contains("Reason           no workspace changes"));
    assert!(
        report.contains("Action           create a git commit from the current workspace changes")
    );
    assert!(report.contains("/status to inspect context"));
    assert!(report.contains("/diff to inspect repo changes"));
}

#[test]
fn runtime_slash_prompts_queue_real_work() {
    let bughunter = build_bughunter_prompt(Some("runtime"));
    assert!(bughunter.contains("Hunt for real bugs in: runtime"));
    assert!(bughunter.contains("try to refute it yourself"));

    let ultraplan = build_ultraplan_prompt(Some("ship the release"));
    assert!(ultraplan.contains("execution plan for: ship the release"));
    assert!(ultraplan.contains("Do not start implementing"));

    let council = build_council_prompt(Some("choose provider routing"));
    assert!(council.contains("choose provider routing"));
    assert!(council.contains("SpawnMultiAgent"));
    assert!(council.contains("exactly three"));
    assert!(council.contains("Council"));
    assert!(council.contains("Do not include model names"));
    assert!(council.contains("llm_judge_allowed"));
    assert!(council.contains("llm_judge_call_limit"));
    assert!(council.contains("subagent_type: \"judge\""));
    assert!(council.contains("more than `llm_judge_call_limit`"));

    let distill = build_distill_prompt(Some("review loop"));
    assert!(distill.contains("review loop"));
    assert!(distill.contains("SkillDistill"));
    assert!(distill.contains("state: proposed"));
    assert!(distill.contains("Do not call `Skill`"));

    let pr = build_pr_prompt("feature/ux", Some("ready for review"));
    assert!(pr.contains("current branch `feature/ux`"));
    assert!(pr.contains("user context: ready for review"));
    assert!(pr.contains("gh pr create"));

    let issue = build_issue_prompt(Some("flaky test"));
    assert!(issue.contains("File a GitHub issue for: flaky test"));
    assert!(issue.contains("gh issue create"));
}

#[test]
fn no_arg_commands_reject_unexpected_arguments() {
    assert!(validate_no_args("/commit", None).is_ok());

    let error = validate_no_args("/commit", Some("now"))
        .expect_err("unexpected arguments should fail")
        .to_string();
    assert!(error.contains("/commit does not accept arguments"));
    assert!(error.contains("Received: now"));
}

#[test]
fn config_report_supports_section_views() {
    let report = render_config_report(Some("env")).expect("config report should render");
    assert!(report.contains("Merged section: env"));
    let plugins_report =
        render_config_report(Some("plugins")).expect("plugins config report should render");
    assert!(plugins_report.contains("Merged section: plugins"));
}

#[test]
fn memory_report_uses_sectioned_layout() {
    let report = render_memory_report().expect("memory report should render");
    assert!(report.contains("Memory"));
    assert!(report.contains("Working directory"));
    assert!(report.contains("Instruction files"));
    assert!(report.contains("Discovered files"));
}

#[test]
fn config_report_uses_sectioned_layout() {
    let report = render_config_report(None).expect("config report should render");
    assert!(report.contains("Config"));
    assert!(report.contains("Discovered files"));
    assert!(report.contains("Merged JSON"));
}

#[test]
fn parses_git_status_metadata() {
    let _guard = env_lock();
    let temp_root = temp_dir();
    fs::create_dir_all(&temp_root).expect("root dir");
    let (project_root, branch): (Option<std::path::PathBuf>, Option<String>) =
        parse_git_status_metadata_for(
            &temp_root,
            Some(
                "## rcc/cli...origin/rcc/cli
 M src/main.rs",
            ),
        );
    assert_eq!(branch.as_deref(), Some("rcc/cli"));
    assert!(project_root.is_none());
    fs::remove_dir_all(temp_root).expect("cleanup temp dir");
}

#[test]
fn parses_detached_head_from_status_snapshot() {
    let _guard = env_lock();
    assert_eq!(
        parse_git_status_branch(Some(
            "## HEAD (no branch)
 M src/main.rs"
        )),
        Some("detached HEAD".to_string())
    );
}

#[test]
fn parses_git_workspace_summary_counts() {
    let summary = parse_git_workspace_summary(Some(
        "## feature/ux
M  src/main.rs
 M README.md
?? notes.md
UU conflicted.rs",
    ));

    assert_eq!(
        summary,
        GitWorkspaceSummary {
            changed_files: 4,
            staged_files: 2,
            unstaged_files: 2,
            untracked_files: 1,
            conflicted_files: 1,
        }
    );
    assert_eq!(
        summary.headline(),
        "dirty · 4 files · 2 staged, 2 unstaged, 1 untracked, 1 conflicted"
    );
}

#[test]
fn render_diff_report_shows_clean_tree_for_committed_repo() {
    let _guard = env_lock();
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");
    git(&["init", "--quiet"], &root);
    git(&["config", "user.email", "tests@example.com"], &root);
    git(&["config", "user.name", "zo Tests"], &root);
    fs::write(root.join("tracked.txt"), "hello\n").expect("write file");
    git(&["add", "tracked.txt"], &root);
    git(&["commit", "-m", "init", "--quiet"], &root);

    let report = render_diff_report_for(&root).expect("diff report should render");
    assert!(report.contains("# Diff"));
    assert!(report.contains("### Result"));
    assert!(report.contains("clean working tree"));

    fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn render_diff_report_includes_staged_and_unstaged_sections() {
    let _guard = env_lock();
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");
    git(&["init", "--quiet"], &root);
    git(&["config", "user.email", "tests@example.com"], &root);
    git(&["config", "user.name", "zo Tests"], &root);
    fs::write(root.join("tracked.txt"), "hello\n").expect("write file");
    git(&["add", "tracked.txt"], &root);
    git(&["commit", "-m", "init", "--quiet"], &root);

    fs::write(root.join("tracked.txt"), "hello\nstaged\n").expect("update file");
    git(&["add", "tracked.txt"], &root);
    fs::write(root.join("tracked.txt"), "hello\nstaged\nunstaged\n").expect("update file twice");

    let report = render_diff_report_for(&root).expect("diff report should render");
    assert!(report.contains("Staged changes:"));
    assert!(report.contains("Unstaged changes:"));
    assert!(report.contains("```diff"));
    assert!(report.contains("1 file changed"));
    assert!(report.contains("tracked.txt"));

    fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn render_diff_report_omits_ignored_files() {
    let _guard = env_lock();
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");
    git(&["init", "--quiet"], &root);
    git(&["config", "user.email", "tests@example.com"], &root);
    git(&["config", "user.name", "zo Tests"], &root);
    fs::write(root.join(".gitignore"), ".omx/\nignored.txt\n").expect("write gitignore");
    fs::write(root.join("tracked.txt"), "hello\n").expect("write tracked");
    git(&["add", ".gitignore", "tracked.txt"], &root);
    git(&["commit", "-m", "init", "--quiet"], &root);
    fs::create_dir_all(root.join(".omx")).expect("write omx dir");
    fs::write(root.join(".omx").join("state.json"), "{}").expect("write ignored omx");
    fs::write(root.join("ignored.txt"), "secret\n").expect("write ignored file");
    fs::write(root.join("tracked.txt"), "hello\nworld\n").expect("write tracked change");

    let report = render_diff_report_for(&root).expect("diff report should render");
    assert!(report.contains("tracked.txt"));
    assert!(!report.contains("+++ b/ignored.txt"));
    assert!(!report.contains("+++ b/.omx/state.json"));

    fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn resume_diff_command_renders_report_for_saved_session() {
    let _guard = env_lock();
    let root = temp_dir();
    fs::create_dir_all(&root).expect("root dir");
    git(&["init", "--quiet"], &root);
    git(&["config", "user.email", "tests@example.com"], &root);
    git(&["config", "user.name", "zo Tests"], &root);
    fs::write(root.join("tracked.txt"), "hello\n").expect("write tracked");
    git(&["add", "tracked.txt"], &root);
    git(&["commit", "-m", "init", "--quiet"], &root);
    fs::write(root.join("tracked.txt"), "hello\nworld\n").expect("modify tracked");
    let session_path = root.join("session.json");
    Session::new()
        .save_to_path(&session_path)
        .expect("session should save");

    let session = Session::load_from_path(&session_path).expect("session should load");
    let outcome = with_current_dir(&root, || {
        run_resume_command(&session_path, &session, &SlashCommand::Diff)
            .expect("resume diff should work")
    });
    let message = outcome.message.expect("diff message should exist");
    assert!(message.contains("Unstaged changes:"));
    assert!(message.contains("```diff"));
    assert!(message.contains("tracked.txt"));

    fs::remove_dir_all(root).expect("cleanup temp dir");
}

#[test]
fn status_context_reads_real_workspace_metadata() {
    let context = status_context(None).expect("status context should load");
    assert!(context.cwd.is_absolute());
    assert!(context.discovered_config_files >= context.loaded_config_files);
    assert!(context.loaded_config_files <= context.discovered_config_files);
}

#[test]
fn clear_command_requires_explicit_confirmation_flag() {
    assert_eq!(
        SlashCommand::parse("/clear"),
        Ok(Some(SlashCommand::Clear { confirm: false }))
    );
    assert_eq!(
        SlashCommand::parse("/clear --confirm"),
        Ok(Some(SlashCommand::Clear { confirm: true }))
    );
}

#[test]
fn parses_resume_and_config_slash_commands() {
    assert_eq!(
        SlashCommand::parse("/resume saved-session.jsonl"),
        Ok(Some(SlashCommand::Resume {
            session_path: Some("saved-session.jsonl".to_string())
        }))
    );
    assert_eq!(
        SlashCommand::parse("/clear --confirm"),
        Ok(Some(SlashCommand::Clear { confirm: true }))
    );
    assert_eq!(
        SlashCommand::parse("/config"),
        Ok(Some(SlashCommand::Config { section: None }))
    );
    assert_eq!(
        SlashCommand::parse("/config env"),
        Ok(Some(SlashCommand::Config {
            section: Some("env".to_string())
        }))
    );
    assert_eq!(
        SlashCommand::parse("/memory"),
        Ok(Some(SlashCommand::Memory))
    );
    assert_eq!(SlashCommand::parse("/init"), Ok(Some(SlashCommand::Init)));
    assert_eq!(
        SlashCommand::parse("/session fork incident-review"),
        Ok(Some(SlashCommand::Session {
            action: Some("fork".to_string()),
            target: Some("incident-review".to_string())
        }))
    );
}

#[test]
fn help_mentions_jsonl_resume_examples() {
    let mut help = Vec::new();
    print_help_to(&mut help).expect("help should render");
    let help = String::from_utf8(help).expect("help should be utf8");
    assert!(help.contains("zo --resume [SESSION.jsonl|session-id|latest]"));
    assert!(help.contains("Use `latest` with --resume, /resume, or /session switch"));
    assert!(help.contains("zo --resume latest"));
    assert!(help.contains("zo --resume latest /status /diff /export notes.txt"));
}

#[test]
fn help_documents_output_and_input_formats() {
    let mut help = Vec::new();
    print_help_to(&mut help).expect("help should render");
    let help = String::from_utf8(help).expect("help should be utf8");
    // The parser accepts stream-json/ndjson output; help must say so.
    assert!(help.contains("stream-json"), "help: {help}");
    // Input format is documented with streaming aliases now that stdin NDJSON
    // is wired into the headless prompt path.
    assert!(help.contains("--input-format"), "help: {help}");
    assert!(
        help.contains("text or stream-json"),
        "help should disclose the supported input formats"
    );
}

#[test]
fn managed_sessions_default_to_jsonl_and_resolve_legacy_json() {
    let _env_guard = env_lock();
    let _guard = cwd_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let workspace = temp_workspace("session-resolution");
    std::fs::create_dir_all(&workspace).expect("workspace should create");
    let previous = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(&workspace).expect("switch cwd");

    let handle = create_managed_session_handle("session-alpha", SessionScope::Project)
        .expect("jsonl handle");
    assert!(handle.path.ends_with("session-alpha.jsonl"));

    let legacy_path = workspace.join(".zo/sessions/legacy.json");
    std::fs::create_dir_all(
        legacy_path
            .parent()
            .expect("legacy path should have parent directory"),
    )
    .expect("session dir should exist");
    Session::new()
        .with_persistence_path(legacy_path.clone())
        .save_to_path(&legacy_path)
        .expect("legacy session should save");

    let resolved = resolve_session_reference("legacy").expect("legacy session should resolve");
    assert_eq!(
        resolved
            .path
            .canonicalize()
            .expect("resolved path should exist"),
        legacy_path
            .canonicalize()
            .expect("legacy path should exist")
    );

    std::env::set_current_dir(previous).expect("restore cwd");
    std::fs::remove_dir_all(workspace).expect("workspace should clean up");
}

#[test]
fn latest_session_alias_resolves_most_recent_managed_session() {
    let _env_guard = env_lock();
    let _guard = cwd_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let workspace = temp_workspace("latest-session-alias");
    std::fs::create_dir_all(&workspace).expect("workspace should create");
    let previous = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(&workspace).expect("switch cwd");

    let older = create_managed_session_handle("session-older", SessionScope::Project)
        .expect("older handle");
    Session::new()
        .with_persistence_path(older.path.clone())
        .save_to_path(&older.path)
        .expect("older session should save");
    std::thread::sleep(Duration::from_millis(20));
    let newer = create_managed_session_handle("session-newer", SessionScope::Project)
        .expect("newer handle");
    Session::new()
        .with_persistence_path(newer.path.clone())
        .save_to_path(&newer.path)
        .expect("newer session should save");

    let resolved = resolve_session_reference("latest").expect("latest session should resolve");
    assert_eq!(
        resolved
            .path
            .canonicalize()
            .expect("resolved path should exist"),
        newer.path.canonicalize().expect("newer path should exist")
    );

    std::env::set_current_dir(previous).expect("restore cwd");
    std::fs::remove_dir_all(workspace).expect("workspace should clean up");
}

#[test]
fn unknown_slash_command_guidance_suggests_nearby_commands() {
    let message = format_unknown_slash_command("stats");
    assert!(message.contains("Unknown slash command: /stats"));
    assert!(message.contains("/status"));
    assert!(message.contains("/help"));
}

#[test]
fn resume_usage_mentions_latest_shortcut() {
    let usage = render_resume_usage();
    assert!(usage.contains("/resume <session-path|session-id|latest>"));
    assert!(usage.contains("/sessions/<session-id>.jsonl"));
    assert!(usage.contains("/session list"));
}

fn cwd_lock() -> &'static Mutex<()> {
    crate::test_cwd_lock()
}

fn temp_workspace(label: &str) -> PathBuf {
    // Keep test sessions/state out of the developer's real ~/.zo.
    crate::isolate_global_zo_home_for_tests();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("zo-cli-{label}-{nanos}"))
}

#[test]
fn init_template_mentions_detected_rust_workspace() {
    let _guard = cwd_lock()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let rendered = crate::init::render_init_context_md(&workspace_root);
    assert!(rendered.contains("# context.md"));
    assert!(rendered.contains("cargo clippy --workspace --all-targets -- -D warnings"));
}

#[test]
fn converts_tool_roundtrip_messages() {
    let messages = vec![
        ConversationMessage::user_text("hello"),
        ConversationMessage::assistant(vec![ContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "bash".to_string(),
            input: "{\"command\":\"pwd\"}".to_string(),
        }]),
        ConversationMessage {
            role: MessageRole::Tool,
            blocks: vec![ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                tool_name: "bash".to_string(),
                output: "ok".to_string(),
                is_error: false,
                images: Vec::new(),
            }],
            usage: None,
            thought_signature: None,
            reasoning_replay: None,
                    model: None,
        },
    ];

    let converted = super::convert_messages(&messages);
    assert_eq!(converted.len(), 3);
    assert_eq!(converted[1].role, "assistant");
    assert_eq!(converted[2].role, "user");
}
#[test]
fn repl_help_mentions_history_completion_and_multiline() {
    let help = render_repl_help();
    assert!(help.contains("Up/Down"));
    assert!(help.contains("Tab"));
    assert!(help.contains("Shift+Enter/Ctrl+J"));
}

#[test]
fn tool_rendering_helpers_compact_output() {
    let start = format_tool_call_start("read_file", r#"{"path":"src/main.rs"}"#);
    assert!(start.contains("read_file"));
    assert!(start.contains("src/main.rs"));

    let done = format_tool_result(
        "read_file",
        r#"{"file":{"filePath":"src/main.rs","content":"hello","numLines":1,"startLine":1,"totalLines":1}}"#,
        false,
    );
    assert!(done.contains("📄 Read src/main.rs"));
    assert!(done.contains("hello"));
}

#[test]
fn tool_rendering_truncates_large_read_output_for_display_only() {
    let content = (0..200)
        .map(|index| format!("line {index:03}"))
        .collect::<Vec<_>>()
        .join("\n");
    let output = json!({
        "file": {
            "filePath": "src/main.rs",
            "content": content,
            "numLines": 200,
            "startLine": 1,
            "totalLines": 200
        }
    })
    .to_string();

    let rendered = format_tool_result("read_file", &output, false);

    assert!(rendered.contains("line 000"));
    assert!(rendered.contains("line 079"));
    assert!(!rendered.contains("line 199"));
    assert!(rendered.contains("full result preserved in session"));
    assert!(output.contains("line 199"));
}

#[test]
fn tool_rendering_truncates_large_bash_output_for_display_only() {
    let stdout = (0..120)
        .map(|index| format!("stdout {index:03}"))
        .collect::<Vec<_>>()
        .join("\n");
    let output = json!({
        "stdout": stdout,
        "stderr": "",
        "returnCodeInterpretation": "completed successfully"
    })
    .to_string();

    let rendered = format_tool_result("bash", &output, false);

    assert!(rendered.contains("stdout 000"));
    assert!(rendered.contains("stdout 059"));
    assert!(!rendered.contains("stdout 119"));
    assert!(rendered.contains("full result preserved in session"));
    assert!(output.contains("stdout 119"));
}

#[test]
fn tool_rendering_truncates_generic_long_output_for_display_only() {
    let items = (0..120)
        .map(|index| format!("payload {index:03}"))
        .collect::<Vec<_>>();
    let output = json!({
        "summary": "plugin payload",
        "items": items,
    })
    .to_string();

    let rendered = format_tool_result("plugin_echo", &output, false);

    assert!(rendered.contains("plugin_echo"));
    assert!(rendered.contains("payload 000"));
    assert!(rendered.contains("payload 040"));
    assert!(!rendered.contains("payload 080"));
    assert!(!rendered.contains("payload 119"));
    assert!(rendered.contains("full result preserved in session"));
    assert!(output.contains("payload 119"));
}

#[test]
fn tool_rendering_truncates_raw_generic_output_for_display_only() {
    let output = (0..120)
        .map(|index| format!("raw {index:03}"))
        .collect::<Vec<_>>()
        .join("\n");

    let rendered = format_tool_result("plugin_echo", &output, false);

    assert!(rendered.contains("plugin_echo"));
    assert!(rendered.contains("raw 000"));
    assert!(rendered.contains("raw 059"));
    assert!(!rendered.contains("raw 119"));
    assert!(rendered.contains("full result preserved in session"));
    assert!(output.contains("raw 119"));
}

#[test]
fn push_output_block_renders_markdown_text() {
    let mut out = Vec::new();
    let mut events = Vec::new();
    let mut pending_tool = None;

    push_output_block(
        OutputContentBlock::Text {
            text: "# Heading".to_string(),
        },
        &mut out,
        &mut events,
        &mut pending_tool,
        false,
    )
    .expect("text block should render");

    let rendered = String::from_utf8(out).expect("utf8");
    assert!(rendered.contains("Heading"));
    assert!(rendered.contains('\u{1b}'));
}

#[test]
fn push_output_block_skips_empty_object_prefix_for_tool_streams() {
    let mut out = Vec::new();
    let mut events = Vec::new();
    let mut pending_tool = None;

    push_output_block(
        OutputContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "read_file".to_string(),
            input: json!({}),
        },
        &mut out,
        &mut events,
        &mut pending_tool,
        true,
    )
    .expect("tool block should accumulate");

    assert!(events.is_empty());
    assert_eq!(
        pending_tool,
        Some(("tool-1".to_string(), "read_file".to_string(), String::new(),))
    );
}

#[test]
fn response_to_events_preserves_empty_object_json_input_outside_streaming() {
    let mut out = Vec::new();
    let events = response_to_events(
        MessageResponse {
            id: "msg-1".to_string(),
            kind: "message".to_string(),
            model: "claude-opus-4-7".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::ToolUse {
                id: "tool-1".to_string(),
                name: "read_file".to_string(),
                input: json!({}),
            }],
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            request_id: None,
            thought_signature: None,
            reasoning_replay: None,
            context_management: None,
        },
        &mut out,
    )
    .expect("response conversion should succeed");

    assert!(matches!(
        &events[0],
        AssistantEvent::ToolUse { name, input, .. }
            if name == "read_file" && input == "{}"
    ));
}

#[test]
fn response_to_events_preserves_non_empty_json_input_outside_streaming() {
    let mut out = Vec::new();
    let events = response_to_events(
        MessageResponse {
            id: "msg-2".to_string(),
            kind: "message".to_string(),
            model: "claude-opus-4-7".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::ToolUse {
                id: "tool-2".to_string(),
                name: "read_file".to_string(),
                input: json!({ "path": "rust/Cargo.toml" }),
            }],
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            request_id: None,
            thought_signature: None,
            reasoning_replay: None,
            context_management: None,
        },
        &mut out,
    )
    .expect("response conversion should succeed");

    assert!(matches!(
        &events[0],
        AssistantEvent::ToolUse { name, input, .. }
            if name == "read_file" && input == "{\"path\":\"rust/Cargo.toml\"}"
    ));
}

#[test]
fn response_to_events_captures_thinking_blocks_before_text() {
    // The non-streaming fallback must capture a complete thinking block (so it is
    // stored and replayed verbatim on the next Anthropic request) in arrival order,
    // leading the turn before the text. The reasoning text is internal, so it is
    // NOT rendered to the terminal writer.
    let mut out = Vec::new();
    let events = response_to_events(
        MessageResponse {
            id: "msg-3".to_string(),
            kind: "message".to_string(),
            model: "claude-opus-4-7".to_string(),
            role: "assistant".to_string(),
            content: vec![
                OutputContentBlock::Thinking {
                    thinking: "step 1".to_string(),
                    signature: Some("sig_123".to_string()),
                },
                OutputContentBlock::Text {
                    text: "Final answer".to_string(),
                },
            ],
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            request_id: None,
            thought_signature: None,
            reasoning_replay: None,
            context_management: None,
        },
        &mut out,
    )
    .expect("response conversion should succeed");

    assert!(matches!(
        &events[0],
        AssistantEvent::Thinking { thinking, signature }
            if thinking == "step 1" && signature.as_deref() == Some("sig_123")
    ));
    assert!(matches!(
        &events[1],
        AssistantEvent::TextDelta(text) if text == "Final answer"
    ));
    // Reasoning text stays internal — only the answer is rendered to the terminal.
    assert!(!String::from_utf8(out).expect("utf8").contains("step 1"));
}

#[test]
fn response_to_events_emits_provider_state_for_thought_signature() {
    let mut out = Vec::new();
    let events = response_to_events(
        MessageResponse {
            id: "msg-signed".to_string(),
            kind: "message".to_string(),
            model: "gemini-3.5-flash".to_string(),
            role: "assistant".to_string(),
            content: vec![OutputContentBlock::Text {
                text: "Final answer".to_string(),
            }],
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
            request_id: None,
            thought_signature: Some("SIG_NON_STREAM".to_string()),
            reasoning_replay: None,
            context_management: None,
        },
        &mut out,
    )
    .expect("response conversion should succeed");

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantEvent::ProviderState(state)
            if state.as_gemini_thought_signature() == Some("SIG_NON_STREAM")
    )));
}

#[test]
fn build_runtime_plugin_state_merges_plugin_hooks_into_runtime_features() {
    let config_home = temp_dir();
    let workspace = temp_dir();
    let source_root = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    fs::create_dir_all(&source_root).expect("source root");
    write_plugin_fixture(&source_root, "hook-runtime-demo", true, false);

    let mut manager = PluginManager::new(PluginManagerConfig::new(&config_home));
    manager
        .install(source_root.to_str().expect("utf8 source path"))
        .expect("plugin install should succeed");
    let loader = ConfigLoader::new(&workspace, &config_home);
    let runtime_config = loader.load().expect("runtime config should load");
    let state =
        build_runtime_plugin_state_with_loader(&workspace, &loader, &runtime_config, None)
        .expect("plugin state should load");
    let pre_hooks = state.feature_config.hooks().pre_tool_use();
    assert_eq!(pre_hooks.len(), 1);
    let command = pre_hooks[0].command().trim_matches('\'');
    assert!(
        command.ends_with("hooks/pre.sh"),
        "expected installed plugin hook path, got {pre_hooks:?}"
    );

    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
    let _ = fs::remove_dir_all(source_root);
}

#[test]
#[allow(clippy::too_many_lines)]
fn build_runtime_plugin_state_discovers_mcp_tools_and_surfaces_pending_servers() {
    let _guard = env_lock();
    let previous_eager_mcp = std::env::var_os("ZO_EAGER_MCP");

    let config_home = temp_dir();
    let workspace = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    let script_path = workspace.join("fixture-mcp.py");
    write_mcp_server_fixture(&script_path);
    fs::write(
        config_home.join("settings.json"),
        format!(
            r#"{{
                  "mcpServers": {{
                    "alpha": {{
                      "command": "python3",
                      "args": ["{}"]
                    }},
                    "broken": {{
                      "command": "python3",
                      "args": ["-c", "import sys; sys.exit(0)"]
                    }}
                  }}
                }}"#,
            script_path.to_string_lossy()
        ),
    )
    .expect("write mcp settings");

    std::env::set_var("ZO_EAGER_MCP", "1");

    let loader = ConfigLoader::new(&workspace, &config_home);
    let runtime_config = loader.load().expect("runtime config should load");
    let state =
        build_runtime_plugin_state_with_loader(&workspace, &loader, &runtime_config, None)
        .expect("runtime plugin state should load");

    let allowed = state
        .tool_registry
        .normalize_allowed_tools(&["mcp__alpha__echo".to_string(), "MCPTool".to_string()])
        .expect("mcp tools should be allow-listable")
        .expect("allow-list should exist");
    assert!(allowed.contains("mcp__alpha__echo"));
    assert!(allowed.contains("MCPTool"));

    let mut executor = CliToolExecutor::new(
        None,
        false,
        state.tool_registry.clone(),
        state.mcp_state.clone(),
    );

    let tool_output = executor
        .execute("mcp__alpha__echo", r#"{"text":"hello"}"#)
        .expect("discovered mcp tool should execute");
    let tool_json: serde_json::Value =
        serde_json::from_str(&tool_output).expect("tool output should be json");
    assert_eq!(tool_json["structuredContent"]["echoed"], "hello");

    let wrapped_output = executor
        .execute(
            "MCPTool",
            r#"{"qualifiedName":"mcp__alpha__echo","arguments":{"text":"wrapped"}}"#,
        )
        .expect("generic mcp wrapper should execute");
    let wrapped_json: serde_json::Value =
        serde_json::from_str(&wrapped_output).expect("wrapped output should be json");
    assert_eq!(wrapped_json["structuredContent"]["echoed"], "wrapped");

    for (alias, text) in [
        ("alpha.echo", "dot-alias"),
        ("alpha_echo", "underscore-alias"),
        ("echo", "bare-unique-alias"),
    ] {
        let alias_output = executor
            .execute(
                "MCPTool",
                &format!(r#"{{"qualifiedName":"{alias}","arguments":{{"text":"{text}"}}}}"#),
            )
            .unwrap_or_else(|error| panic!("MCPTool alias {alias:?} should execute: {error}"));
        let alias_json: serde_json::Value =
            serde_json::from_str(&alias_output).expect("alias output should be json");
        assert_eq!(
            alias_json["structuredContent"]["echoed"], text,
            "MCPTool alias {alias:?} routes to mcp__alpha__echo"
        );
    }

    let search_output = executor
        .execute("ToolSearch", r#"{"query":"alpha echo","max_results":5}"#)
        .expect("tool search should execute");
    let search_json: serde_json::Value =
        serde_json::from_str(&search_output).expect("search output should be json");
    assert_eq!(search_json["matches"][0], "mcp__alpha__echo");
    assert_eq!(search_json["pending_mcp_servers"][0], "broken");
    assert_eq!(
        search_json["mcp_degraded"]["failed_servers"][0]["server_name"],
        "broken"
    );
    assert_eq!(
        search_json["mcp_degraded"]["failed_servers"][0]["phase"],
        "initialize_handshake"
    );
    assert_eq!(
        search_json["mcp_degraded"]["available_tools"][0],
        "mcp__alpha__echo"
    );

    let listed = executor
        .execute("ListMcpResourcesTool", r#"{"server":"alpha"}"#)
        .expect("resources should list");
    let listed_json: serde_json::Value =
        serde_json::from_str(&listed).expect("resource output should be json");
    assert_eq!(listed_json["resources"][0]["uri"], "file://guide.txt");

    let read = executor
        .execute(
            "ReadMcpResourceTool",
            r#"{"server":"alpha","uri":"file://guide.txt"}"#,
        )
        .expect("resource should read");
    let read_json: serde_json::Value =
        serde_json::from_str(&read).expect("resource read output should be json");
    assert_eq!(
        read_json["contents"][0]["text"],
        "contents for file://guide.txt"
    );

    for legacy_name in ["ListMcpResources", "ReadMcpResource", "MCP", "McpAuth"] {
        assert!(
            executor.execute(legacy_name, "{}").is_err(),
            "legacy MCP tool `{legacy_name}` must not route through the CLI executor"
        );
    }

    if let Some(mcp_state) = state.mcp_state {
        mcp_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .shutdown()
            .expect("mcp shutdown should succeed");
    }

    match previous_eager_mcp {
        Some(value) => std::env::set_var("ZO_EAGER_MCP", value),
        None => std::env::remove_var("ZO_EAGER_MCP"),
    }
    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
}

#[test]
fn deferred_mcp_discovery_splices_tools_without_eager_env() {
    let _guard = env_lock();
    let previous_eager_mcp = std::env::var_os("ZO_EAGER_MCP");

    let config_home = temp_dir();
    let workspace = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    let script_path = workspace.join("fixture-mcp.py");
    write_mcp_server_fixture(&script_path);
    fs::write(
        config_home.join("settings.json"),
        format!(
            r#"{{
                  "mcpServers": {{
                    "alpha": {{
                      "command": "python3",
                      "args": ["{}"]
                    }}
                  }}
                }}"#,
            script_path.to_string_lossy()
        ),
    )
    .expect("write mcp settings");

    std::env::remove_var("ZO_EAGER_MCP");

    let loader = ConfigLoader::new(&workspace, &config_home);
    let runtime_config = loader.load().expect("runtime config should load");
    let state =
        build_runtime_plugin_state_with_loader(&workspace, &loader, &runtime_config, None)
        .expect("runtime plugin state should load");

    assert!(state.tool_registry.has_runtime_tool("MCPTool"));
    assert!(
        !state.tool_registry.has_runtime_tool("mcp__alpha__echo"),
        "non-eager startup should advertise only MCP wrapper tools"
    );
    let mcp_state = state.mcp_state.clone().expect("mcp state should exist");
    crate::session::discover_pending_mcp_tools_now(&mcp_state, &state.tool_registry);
    assert!(state.tool_registry.has_runtime_tool("mcp__alpha__echo"));

    mcp_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .shutdown()
        .expect("mcp shutdown should succeed");

    match previous_eager_mcp {
        Some(value) => std::env::set_var("ZO_EAGER_MCP", value),
        None => std::env::remove_var("ZO_EAGER_MCP"),
    }
    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
}

// A mid-session runtime rebuild (`/reload`, `/resume`, `/permission`,
// Shift+Tab, model/effort change) funnels through `LiveCli::replace_runtime`,
// which swaps in a freshly built runtime whose MCP servers are all seeded
// `pending`. Startup kicks background discovery exactly once, so unless
// `replace_runtime` restarts it, every server on the replacement runtime stays
// stuck `Discovering` and its tools never surface. This drives a real
// non-eager `LiveCli` through `reload_context()` and asserts `mcp__alpha__echo`
// appears on the replacement registry — it fails if the
// `Self::start_mcp_discovery_for_runtime(&self.runtime)` restart is removed.
#[test]
fn reload_context_restarts_mcp_discovery_on_replacement_runtime() {
    let _guard = env_lock();
    let previous_key = std::env::var_os("ANTHROPIC_API_KEY");
    let previous_config_home = std::env::var_os("ZO_CONFIG_HOME");
    let previous_eager_mcp = std::env::var_os("ZO_EAGER_MCP");

    let config_home = temp_dir();
    let workspace = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    let script_path = workspace.join("fixture-mcp.py");
    write_mcp_server_fixture(&script_path);
    let mcp_config = workspace.join("external-mcp.json");
    fs::write(
        &mcp_config,
        format!(
            r#"{{
                  "mcpServers": {{
                    "alpha": {{
                      "command": "python3",
                      "args": ["{}"]
                    }}
                  }}
                }}"#,
            script_path.to_string_lossy()
        ),
    )
    .expect("write explicit mcp config");

    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-mcp-reload");
    std::env::set_var("ZO_CONFIG_HOME", &config_home);
    // Non-eager: startup advertises only the MCP wrapper tools and leaves the
    // `alpha` server `pending`, so `mcp__alpha__echo` can only appear once a
    // discovery pass runs against the replacement runtime.
    std::env::remove_var("ZO_EAGER_MCP");

    let mut cli = with_current_dir(&workspace, || {
        LiveCli::new_scoped_with_mcp_config(
            DEFAULT_MODEL.to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
            SessionScope::Ephemeral,
            Some(mcp_config.clone()),
        )
    })
    .expect("live cli should build with explicit mcp config");

    assert!(
        cli.runtime.mcp_state.is_some(),
        "--mcp-config should create runtime MCP state"
    );
    assert!(
        !cli.runtime
            .api_client()
            .tool_registry()
            .has_runtime_tool("mcp__alpha__echo"),
        "non-eager startup must leave alpha pending before any discovery pass"
    );

    cli.reload_context().expect("reload_context should succeed");

    // The replacement runtime must still own the configured `alpha` server so
    // discovery has something to find. During an active discovery pass the
    // server is briefly detached out of `server_names()`, so accept either the
    // still-pending name or an already-spliced tool as proof the config
    // survived the rebuild.
    let mcp_state = cli
        .runtime
        .mcp_state
        .clone()
        .expect("replacement runtime must keep MCP state after reload");

    // Poll the replacement runtime's registry under a strict timeout. Discovery
    // runs on a background thread, so allow it to land — but never hang.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut discovered = false;
    while std::time::Instant::now() < deadline {
        if cli
            .runtime
            .api_client()
            .tool_registry()
            .has_runtime_tool("mcp__alpha__echo")
        {
            discovered = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    if !discovered {
        // Surface the terminal status/error rather than only "no tool", so a
        // Failed/AuthPending discovery is diagnosable from the failure alone.
        let statuses = mcp_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .server_statuses();
        panic!(
            "reload_context() must restart MCP discovery on the replacement runtime; \
             mcp__alpha__echo never surfaced. Server statuses: {statuses:?}"
        );
    }

    cli.runtime.shutdown_mcp().expect("mcp shutdown");
    cli.runtime.shutdown_plugins().expect("plugin shutdown");

    match previous_key {
        Some(value) => std::env::set_var("ANTHROPIC_API_KEY", value),
        None => std::env::remove_var("ANTHROPIC_API_KEY"),
    }
    match previous_config_home {
        Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
        None => std::env::remove_var("ZO_CONFIG_HOME"),
    }
    match previous_eager_mcp {
        Some(value) => std::env::set_var("ZO_EAGER_MCP", value),
        None => std::env::remove_var("ZO_EAGER_MCP"),
    }
    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
}

#[test]
fn live_cli_explicit_mcp_config_wires_runtime_mcp_tools() {
    let _guard = env_lock();
    let previous_key = std::env::var_os("ANTHROPIC_API_KEY");
    let previous_config_home = std::env::var_os("ZO_CONFIG_HOME");
    let previous_eager_mcp = std::env::var_os("ZO_EAGER_MCP");

    let config_home = temp_dir();
    let workspace = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    let script_path = workspace.join("fixture-mcp.py");
    write_mcp_server_fixture(&script_path);
    let mcp_config = workspace.join("external-mcp.json");
    fs::write(
        &mcp_config,
        format!(
            r#"{{
                  "mcpServers": {{
                    "alpha": {{
                      "command": "python3",
                      "args": ["{}"]
                    }}
                  }}
                }}"#,
            script_path.to_string_lossy()
        ),
    )
    .expect("write explicit mcp config");

    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-mcp-config");
    std::env::set_var("ZO_CONFIG_HOME", &config_home);
    std::env::set_var("ZO_EAGER_MCP", "1");

    let mut cli = with_current_dir(&workspace, || {
        LiveCli::new_scoped_with_mcp_config(
            DEFAULT_MODEL.to_string(),
            true,
            None,
            PermissionMode::DangerFullAccess,
            SessionScope::Ephemeral,
            Some(mcp_config.clone()),
        )
    })
    .expect("live cli should build with explicit mcp config");

    assert!(
        cli.runtime.mcp_state.is_some(),
        "--mcp-config should create runtime MCP state"
    );
    let registry = cli.runtime.api_client().tool_registry();
    assert!(registry.has_runtime_tool("mcp__alpha__echo"));
    assert!(registry.has_runtime_tool("MCPTool"));
    cli.runtime.shutdown_mcp().expect("mcp shutdown");
    cli.runtime.shutdown_plugins().expect("plugin shutdown");

    match previous_key {
        Some(value) => std::env::set_var("ANTHROPIC_API_KEY", value),
        None => std::env::remove_var("ANTHROPIC_API_KEY"),
    }
    match previous_config_home {
        Some(value) => std::env::set_var("ZO_CONFIG_HOME", value),
        None => std::env::remove_var("ZO_CONFIG_HOME"),
    }
    match previous_eager_mcp {
        Some(value) => std::env::set_var("ZO_EAGER_MCP", value),
        None => std::env::remove_var("ZO_EAGER_MCP"),
    }
    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
}

#[test]
fn build_runtime_plugin_state_surfaces_degraded_mcp_servers_structurally() {
    let _guard = env_lock();
    let previous_eager_mcp = std::env::var_os("ZO_EAGER_MCP");

    let config_home = temp_dir();
    let workspace = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    fs::write(
        config_home.join("settings.json"),
        r#"{
              "mcpServers": {
                "remote": {
                  "url": "https://example.test/mcp"
                }
              }
            }"#,
    )
    .expect("write mcp settings");

    std::env::set_var("ZO_EAGER_MCP", "1");

    let loader = ConfigLoader::new(&workspace, &config_home);
    let runtime_config = loader.load().expect("runtime config should load");
    let state =
        build_runtime_plugin_state_with_loader(&workspace, &loader, &runtime_config, None)
        .expect("runtime plugin state should load");
    let mut executor = CliToolExecutor::new(
        None,
        false,
        state.tool_registry.clone(),
        state.mcp_state.clone(),
    );

    let search_output = executor
        .execute("ToolSearch", r#"{"query":"remote","max_results":5}"#)
        .expect("tool search should execute");
    let search_json: serde_json::Value =
        serde_json::from_str(&search_output).expect("search output should be json");
    assert_eq!(search_json["pending_mcp_servers"][0], "remote");
    assert_eq!(
        search_json["mcp_degraded"]["failed_servers"][0]["server_name"],
        "remote"
    );
    assert_eq!(
        search_json["mcp_degraded"]["failed_servers"][0]["phase"],
        "initialize_handshake"
    );
    assert!(
        search_json["mcp_degraded"]["failed_servers"][0]["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("example.test")
    );

    match previous_eager_mcp {
        Some(value) => std::env::set_var("ZO_EAGER_MCP", value),
        None => std::env::remove_var("ZO_EAGER_MCP"),
    }
    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
}

#[test]
fn build_runtime_runs_plugin_lifecycle_init_and_shutdown() {
    let config_home = temp_dir();
    // Inject a dummy API key so runtime construction succeeds without real credentials.
    // This test only exercises plugin lifecycle (init/shutdown), never calls the API.
    std::env::set_var("ANTHROPIC_API_KEY", "test-dummy-key-for-plugin-lifecycle");
    let workspace = temp_dir();
    let source_root = temp_dir();
    fs::create_dir_all(&config_home).expect("config home");
    fs::create_dir_all(&workspace).expect("workspace");
    fs::create_dir_all(&source_root).expect("source root");
    write_plugin_fixture(&source_root, "lifecycle-runtime-demo", false, true);

    let mut manager = PluginManager::new(PluginManagerConfig::new(&config_home));
    let install = manager
        .install(source_root.to_str().expect("utf8 source path"))
        .expect("plugin install should succeed");
    let log_path = install.install_path.join("lifecycle.log");
    let loader = ConfigLoader::new(&workspace, &config_home);
    let runtime_config = loader.load().expect("runtime config should load");
    let runtime_plugin_state =
        build_runtime_plugin_state_with_loader(&workspace, &loader, &runtime_config, None)
            .expect("plugin state should load");
    let mut runtime = build_runtime_with_plugin_state(
        Session::new(),
        "runtime-plugin-lifecycle",
        DEFAULT_MODEL.to_string(),
        vec!["test system prompt".to_string()],
        true,
        false,
        None,
        PermissionMode::DangerFullAccess,
        runtime_plugin_state,
        None,
        None,
        None,
    )
    .expect("runtime should build");

    assert_eq!(
        fs::read_to_string(&log_path).expect("init log should exist"),
        "init\n"
    );

    runtime
        .shutdown_plugins()
        .expect("plugin shutdown should succeed");

    assert_eq!(
        fs::read_to_string(&log_path).expect("shutdown log should exist"),
        "init\nshutdown\n"
    );

    let _ = fs::remove_dir_all(config_home);
    let _ = fs::remove_dir_all(workspace);
    let _ = fs::remove_dir_all(source_root);
    std::env::remove_var("ANTHROPIC_API_KEY");
}

// ===========================================================================
// L7c-3 commit 2: ndjson byte-equivalence test
//
// Pins the byte output of `session::write_ndjson_summary` against a frozen
// fixture. The fixture is auto-generated on the first run (or whenever the
// env var `UPDATE_FIXTURES=1` is set) and committed to disk; subsequent
// runs read the file and assert equality. Any change to the formatter that
// alters its bytes will fail this test, which is exactly the guard L7c-3
// needs while the legacy ANSI tool formatters and helpers are deleted in
// commits 7–9.
//
// The synthetic `TurnSummary` is intentionally minimal and deterministic:
// 1 assistant text response, 1 auto-compaction event, and a fixed usage
// counter. No UUIDs, no timestamps, no message ids, so no normalization is
// needed (the F2 normalize rules in the handoff become unnecessary at this
// synthesis fidelity).
// ===========================================================================

fn synthetic_turn_summary() -> runtime::TurnSummary {
    runtime::TurnSummary {
        assistant_messages: vec![ConversationMessage {
            role: MessageRole::Assistant,
            blocks: vec![ContentBlock::Text {
                text: "Hello from the byte-equivalence fixture.".to_string(),
            }],
            usage: None,
            thought_signature: None,
            reasoning_replay: None,
                    model: None,
        }],
        tool_results: vec![],
        prompt_cache_events: vec![],
        iterations: 1,
        usage: runtime::TokenUsage {
            input_tokens: 12,
            output_tokens: 7,
            cache_creation_input_tokens: 3,
            cache_read_input_tokens: 5,
        },
        turn_output_tokens: 7,
        auto_compaction: Some(runtime::AutoCompactionEvent {
            removed_message_count: 4,
            tokens_before: 0,
            tokens_after: 0,
        }),
        microcompact: None,
        deep_verification: None,
        verification_issues: Vec::new(),
        deep_verifier_parse: None,
        deep_verifier_model: None,
        budget_exhausted: None,
    }
}

#[test]
fn ndjson_byte_equivalence_against_fixture() {
    let summary = synthetic_turn_summary();
    let mut buf: Vec<u8> = Vec::new();
    // Fixed model + duration keep the fixture deterministic.
    crate::session::write_ndjson_summary(&summary, "sonnet", "test-session", 0, &mut buf)
        .expect("write_ndjson_summary should succeed");
    let actual = String::from_utf8(buf).expect("ndjson output should be utf8");

    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ndjson_baseline_basic.jsonl");

    let should_update = std::env::var("UPDATE_FIXTURES").is_ok();
    if should_update || !fixture_path.exists() {
        if let Some(parent) = fixture_path.parent() {
            fs::create_dir_all(parent).expect("create fixtures dir");
        }
        fs::write(&fixture_path, &actual).expect("write fixture");
    }

    let expected = fs::read_to_string(&fixture_path).expect("read fixture");
    assert_eq!(
        actual,
        expected,
        "ndjson byte output drifted from fixture {}; rerun with UPDATE_FIXTURES=1 if intentional",
        fixture_path.display()
    );
}

// End-to-end semantic check of the `stream-json` document (W11):
// every line is a typed event, it opens with a `system` init naming
// the model, and closes with a `result` event carrying duration and
// structured usage. Complements the opaque byte-fixture above.
#[test]
fn ndjson_summary_emits_typed_result_event() {
    let summary = synthetic_turn_summary();
    let mut buf: Vec<u8> = Vec::new();
    crate::session::write_ndjson_summary(&summary, "sonnet", "test-session", 42, &mut buf)
        .expect("write_ndjson_summary should succeed");
    let text = String::from_utf8(buf).expect("utf8");
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.len() >= 2, "expected at least init + result: {text}");

    for line in &lines {
        let value: serde_json::Value = serde_json::from_str(line).expect("each line is valid json");
        assert!(
            value.get("type").is_some(),
            "every event carries a type tag: {line}"
        );
    }

    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["type"], "system");
    assert!(
        first["text"]
            .as_str()
            .unwrap_or_default()
            .contains("sonnet"),
        "init event names the model: {first}"
    );

    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(last["type"], "result");
    assert_eq!(last["duration_ms"], 42);
    assert_eq!(last["iterations"], 1);
    assert_eq!(last["model"], "sonnet");
    assert_eq!(last["usage"]["input_tokens"], 12);
    assert_eq!(last["usage"]["output_tokens"], 7);
    assert!(
        last["estimated_cost"]
            .as_str()
            .unwrap_or_default()
            .starts_with('$'),
        "result carries a formatted cost: {last}"
    );
}

fn write_mcp_server_fixture(script_path: &Path) {
    let script = [
            "#!/usr/bin/env python3",
            "import json, sys",
            "",
            "def read_message():",
            "    while True:",
            "        line = sys.stdin.readline()",
            "        if not line:",
            "            return None",
            "        if line.strip():",
            "            return json.loads(line)",
            "",
            "def send_message(message):",
            r"    sys.stdout.buffer.write(json.dumps(message).encode() + b'\n')",
            "    sys.stdout.buffer.flush()",
            "",
            "while True:",
            "    request = read_message()",
            "    if request is None:",
            "        break",
            "    method = request.get('method', '')",
            "    if 'id' not in request:",
            "        continue",
            "    if method == 'initialize':",
            "        send_message({",
            "            'jsonrpc': '2.0',",
            "            'id': request['id'],",
            "            'result': {",
            "                'protocolVersion': request['params']['protocolVersion'],",
            "                'capabilities': {'tools': {}, 'resources': {}},",
            "                'serverInfo': {'name': 'fixture', 'version': '1.0.0'}",
            "            }",
            "        })",
            "    elif method == 'tools/list':",
            "        send_message({",
            "            'jsonrpc': '2.0',",
            "            'id': request['id'],",
            "            'result': {",
            "                'tools': [",
            "                    {",
            "                        'name': 'echo',",
            "                        'description': 'Echo from MCP fixture',",
            "                        'inputSchema': {",
            "                            'type': 'object',",
            "                            'properties': {'text': {'type': 'string'}},",
            "                            'required': ['text'],",
            "                            'additionalProperties': False",
            "                        },",
            "                        'annotations': {'readOnlyHint': True}",
            "                    }",
            "                ]",
            "            }",
            "        })",
            "    elif method == 'tools/call':",
            "        args = request['params'].get('arguments') or {}",
            "        send_message({",
            "            'jsonrpc': '2.0',",
            "            'id': request['id'],",
            "            'result': {",
            "                'content': [{'type': 'text', 'text': f\"echo:{args.get('text', '')}\"}],",
            "                'structuredContent': {'echoed': args.get('text', '')},",
            "                'isError': False",
            "            }",
            "        })",
            "    elif method == 'resources/list':",
            "        send_message({",
            "            'jsonrpc': '2.0',",
            "            'id': request['id'],",
            "            'result': {",
            "                'resources': [{'uri': 'file://guide.txt', 'name': 'guide', 'mimeType': 'text/plain'}]",
            "            }",
            "        })",
            "    elif method == 'resources/read':",
            "        uri = request['params']['uri']",
            "        send_message({",
            "            'jsonrpc': '2.0',",
            "            'id': request['id'],",
            "            'result': {",
            "                'contents': [{'uri': uri, 'mimeType': 'text/plain', 'text': f'contents for {uri}'}]",
            "            }",
            "        })",
            "    else:",
            "        send_message({",
            "            'jsonrpc': '2.0',",
            "            'id': request['id'],",
            "            'error': {'code': -32601, 'message': method}",
            "        })",
            "",
        ]
        .join("\n");
    fs::write(script_path, script).expect("mcp fixture script should write");
}
