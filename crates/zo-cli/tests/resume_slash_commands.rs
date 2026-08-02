use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use runtime::ContentBlock;
use runtime::Session;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn resumed_binary_accepts_slash_commands_with_arguments() {
    // given
    let temp_dir = unique_temp_dir("resume-slash-commands");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let session_path = temp_dir.join("session.jsonl");
    let export_path = temp_dir.join("notes.txt");

    let mut session = Session::new();
    session
        .push_user_text("ship the slash command harness")
        .expect("session write should succeed");
    session
        .save_to_path(&session_path)
        .expect("session should persist");

    // when
    let output = run_zo(
        &temp_dir,
        &[
            "--resume",
            session_path.to_str().expect("utf8 path"),
            "/export",
            export_path.to_str().expect("utf8 path"),
            "/clear",
            "--confirm",
        ],
    );

    // then
    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("Export"));
    assert!(stdout.contains("wrote transcript"));
    assert!(stdout.contains(export_path.to_str().expect("utf8 path")));
    assert!(stdout.contains("Session cleared"));
    assert!(stdout.contains("Mode             resumed session reset"));
    assert!(stdout.contains("Previous session"));
    assert!(stdout.contains("Resume previous"));
    assert!(stdout.contains("Backup           "));
    assert!(stdout.contains("Session file     "));

    let export = fs::read_to_string(&export_path).expect("export file should exist");
    assert!(export.contains("# Conversation Export"));
    assert!(export.contains("ship the slash command harness"));

    let restored = Session::load_from_path(&session_path).expect("cleared session should load");
    assert!(restored.messages.is_empty());

    let backup_path = stdout
        .lines()
        .find_map(|line| line.strip_prefix("  Backup           "))
        .map(PathBuf::from)
        .expect("clear output should include backup path");
    let backup = Session::load_from_path(&backup_path).expect("backup session should load");
    assert_eq!(backup.messages.len(), 1);
    assert!(matches!(
        backup.messages[0].blocks.first(),
        Some(ContentBlock::Text { text }) if text == "ship the slash command harness"
    ));
}

#[test]
fn status_command_applies_cli_flags_end_to_end() {
    // given
    let temp_dir = unique_temp_dir("status-command-flags");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    // when
    let output = run_zo(
        &temp_dir,
        &[
            "--model",
            "sonnet",
            "--permission-mode",
            "read-only",
            "status",
        ],
    );

    // then
    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("Status"));
    assert!(stdout.contains("Model            claude-sonnet-5"));
    assert!(stdout.contains("Permission mode  read-only"));
}

#[test]
fn resumed_config_command_loads_settings_files_end_to_end() {
    // given
    let temp_dir = unique_temp_dir("resume-config");
    let project_dir = temp_dir.join("project");
    let config_home = temp_dir.join("home").join(".zo");
    fs::create_dir_all(project_dir.join(".zo")).expect("project config dir should exist");
    fs::create_dir_all(&config_home).expect("config home should exist");

    let session_path = project_dir.join("session.jsonl");
    Session::new()
        .with_persistence_path(&session_path)
        .save_to_path(&session_path)
        .expect("session should persist");

    fs::write(config_home.join("settings.json"), r#"{"model":"haiku"}"#)
        .expect("user config should write");
    fs::write(
        project_dir.join(".zo").join("settings.local.json"),
        r#"{"model":"opus"}"#,
    )
    .expect("local config should write");

    // when
    let output = run_zo_with_env(
        &project_dir,
        &[
            "--resume",
            session_path.to_str().expect("utf8 path"),
            "/config",
            "model",
        ],
        &[(
            "ZO_CONFIG_HOME",
            config_home.to_str().expect("utf8 path"),
        )],
    );

    // then
    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("Config"));
    assert!(stdout.contains("Loaded files      2"));
    assert!(stdout.contains(
        config_home
            .join("settings.json")
            .to_str()
            .expect("utf8 path")
    ));
    assert!(stdout.contains(
        project_dir
            .join(".zo")
            .join("settings.local.json")
            .to_str()
            .expect("utf8 path")
    ));
    assert!(stdout.contains("Merged section: model"));
    assert!(stdout.contains("opus"));
}

#[test]
fn resume_latest_restores_the_most_recent_managed_session() {
    // given
    let temp_dir = unique_temp_dir("resume-latest");
    let project_dir = temp_dir.join("project");
    let sessions_dir = project_dir.join(".zo").join("sessions");
    fs::create_dir_all(&sessions_dir).expect("sessions dir should exist");

    // Use lexicographically ordered names as a deterministic tie-break fallback
    // in addition to the sleep below: on very fast filesystems the two writes
    // can otherwise land in the same mtime bucket and make `latest` flaky.
    let older_path = sessions_dir.join("session-aa-older.jsonl");
    let newer_path = sessions_dir.join("session-zz-newer.jsonl");

    let mut older = Session::new().with_persistence_path(&older_path);
    older
        .push_user_text("older session")
        .expect("older session write should succeed");
    older
        .save_to_path(&older_path)
        .expect("older session should persist");

    std::thread::sleep(Duration::from_millis(10));

    let mut newer = Session::new().with_persistence_path(&newer_path);
    newer
        .push_user_text("newer session")
        .expect("newer session write should succeed");
    newer
        .push_user_text("resume me")
        .expect("newer session write should succeed");
    newer
        .save_to_path(&newer_path)
        .expect("newer session should persist");

    // when
    let output = run_zo(&project_dir, &["--resume", "latest", "/status"]);

    // then
    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("Status"));
    assert!(stdout.contains("Messages         2"));
    assert!(stdout.contains(newer_path.to_str().expect("utf8 path")));
}

#[test]
fn resumed_copy_command_uses_clipboard_helper() {
    let temp_dir = unique_temp_dir("resume-copy");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let session_path = temp_dir.join("session.jsonl");
    let mut session = Session::new();
    session
        .push_user_text("copy this resumed payload")
        .expect("session write should succeed");
    session
        .save_to_path(&session_path)
        .expect("session should persist");

    let bin_dir = temp_dir.join("bin");
    fs::create_dir_all(&bin_dir).expect("bin dir");
    let capture_path = temp_dir.join("clipboard.txt");
    let clipboard_command = if cfg!(target_os = "macos") {
        "pbcopy"
    } else {
        "wl-copy"
    };
    let script_path = bin_dir.join(clipboard_command);
    fs::write(
        &script_path,
        format!("#!/bin/sh\ncat > \"{}\"\n", capture_path.display()),
    )
    .expect("write clipboard stub");
    let mut perms = fs::metadata(&script_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&script_path, perms).expect("chmod");

    let path_env = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    let output = run_zo_with_env(
        &temp_dir,
        &[
            "--resume",
            session_path.to_str().expect("utf8 path"),
            "/copy",
            "last",
        ],
        &[("PATH", &path_env)],
    );

    assert!(
        output.status.success(),
        "stdout:\n{}\n\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf8");
    assert!(stdout.contains("Copied to clipboard"));
    assert!(stdout.contains("Target           last"));
    let captured = fs::read_to_string(capture_path).expect("clipboard capture");
    assert!(captured.contains("copy this resumed payload"));
}

fn run_zo(current_dir: &Path, args: &[&str]) -> Output {
    run_zo_with_env(current_dir, args, &[])
}

fn run_zo_with_env(current_dir: &Path, args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_zo"));
    command.current_dir(current_dir).args(args);
    command.env_remove("ZO_HOME");
    let caller_sets_home = envs.iter().any(|(key, _)| *key == "HOME");
    let caller_sets_config_home = envs.iter().any(|(key, _)| *key == "ZO_CONFIG_HOME");
    if !caller_sets_home {
        command.env("HOME", current_dir.join(".home"));
    }
    if !caller_sets_home && !caller_sets_config_home {
        command.env("ZO_CONFIG_HOME", current_dir.join(".zo-home"));
    }
    for (key, value) in envs {
        command.env(key, value);
    }
    command.output().expect("zo should launch")
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_millis();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "zo-{label}-{}-{millis}-{counter}",
        std::process::id()
    ))
}
