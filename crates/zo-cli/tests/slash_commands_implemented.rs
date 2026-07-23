use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn slash_command_theme_persists_config() {
    let temp_dir = unique_temp_dir("theme-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let home_dir = temp_dir.join("home");
    fs::create_dir_all(home_dir.join(".zo")).expect("home config dir");

    let output = run_zo_with_env(&temp_dir, &["/theme", "dark"], &[("HOME", &home_dir)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Theme set to: dark"));

    let settings = fs::read_to_string(home_dir.join(".zo").join("settings.json"))
        .expect("theme settings should persist");
    assert!(settings.contains(r#""theme": "dark""#));

    let output = run_zo_with_env(&temp_dir, &["/theme"], &[("HOME", &home_dir)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Current theme: dark"));
    assert!(stderr.contains("Available: default, dark, light, high-contrast"));
}

#[test]
fn slash_command_advisor_toggles_persisted_mode() {
    let temp_dir = unique_temp_dir("advisor-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let home_dir = temp_dir.join("home");
    fs::create_dir_all(home_dir.join(".zo")).expect("home config dir");

    let output = run_zo_with_env(&temp_dir, &["/advisor"], &[("HOME", &home_dir)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Advisor mode enabled"));

    let settings = fs::read_to_string(home_dir.join(".zo").join("settings.json"))
        .expect("advisor settings should persist");
    assert!(settings.contains(r#""advisorModeEnabled": true"#));
}

#[test]
fn slash_command_thinkback_reports_recent_flow() {
    let temp_dir = unique_temp_dir("thinkback-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/thinkback"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Thinkback"));
    assert!(stderr.contains("Recent flow") || stderr.contains("no prior assistant activity"));
}

#[test]
fn slash_command_upgrade_reports_local_build_state() {
    let temp_dir = unique_temp_dir("upgrade-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/upgrade"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Upgrade"));
    assert!(stderr.contains("Version"));
}

#[test]
fn slash_command_plan_mock() {
    let temp_dir = unique_temp_dir("plan-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/plan", "on"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Enabled worktree-local plan mode override."));
    let local_settings = fs::read_to_string(temp_dir.join(".zo").join("settings.local.json"))
        .expect("plan mode settings should persist");
    assert!(local_settings.contains(r#""defaultMode": "plan""#));

    let output = run_zo(&temp_dir, &["/plan", "off"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Restored the prior worktree-local plan mode setting."));

    let output = run_zo(&temp_dir, &["/plan"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Planning mode is currently disabled."));
}

#[test]
fn headless_bare_tier_falls_back_to_text_listing() {
    let temp_dir = unique_temp_dir("tier-headless");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/tier"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Deep-tier pool (built-in default)"), "{stderr}");
    assert!(stderr.contains("1. "), "{stderr}");
    assert!(stderr.contains("Usage: /tier"), "{stderr}");
}

#[test]
fn slash_command_review_reports_real_diff_state() {
    let temp_dir = unique_temp_dir("review-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    init_git_repo(&temp_dir);
    fs::write(temp_dir.join("src.txt"), "before\n").expect("seed file");
    git(&temp_dir, &["add", "src.txt"]);
    git(&temp_dir, &["commit", "-m", "seed"]);
    fs::write(temp_dir.join("src.txt"), "before\nafter\n").expect("modify file");

    let output = run_zo(&temp_dir, &["/review"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Review"));
    assert!(stderr.contains("Method           git diff preflight"));
    assert!(stderr.contains("Unstaged diff stat"));
    assert!(stderr.contains("src.txt"));

    let output = run_zo(&temp_dir, &["/review", "src.txt"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Scope            src.txt"));
}

#[test]
fn slash_command_doctor_reports_environment_health() {
    let temp_dir = unique_temp_dir("doctor-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/doctor"]);
    assert!(output.status.success());
    // `/doctor` routes to the standalone engine before `LiveCli` startup, so it
    // prints to stdout (like `zo doctor`) rather than the REPL stderr path.
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(combined.contains("Doctor"), "{combined}");
    assert!(combined.contains("config/parse"), "{combined}");
    assert!(combined.contains("sandbox"), "{combined}");
    assert!(combined.contains("git"), "{combined}");
    assert!(
        combined.contains("Healthy") || combined.contains("Needs attention"),
        "{combined}"
    );
}

#[test]
fn slash_command_hooks_reports_configured_hooks() {
    let temp_dir = unique_temp_dir("hooks-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Hooks live in the trusted User config home: repo-committed Project hooks are
    // supply-chain gated (stripped), so a project-scope fixture would report zero.
    let config_home = temp_dir.join("config-home");
    fs::create_dir_all(&config_home).expect("config home should exist");
    fs::write(
        config_home.join("settings.json"),
        r#"{"hooks":{"PreToolUse":["./hooks/pre.sh"],"PostToolUse":["./hooks/post.sh"]}}"#,
    )
    .expect("write hook settings");

    let output = run_zo_with_env(&temp_dir, &["/hooks"], &[("ZO_CONFIG_HOME", config_home.as_path())]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Hooks"));
    assert!(stderr.contains("Counts            pre=1 post=1 failure=0"));
    assert!(stderr.contains("PreToolUse         ./hooks/pre.sh"));
    assert!(stderr.contains("PostToolUse        ./hooks/post.sh"));
}

#[test]
fn slash_command_tasks_reports_honest_empty_registry() {
    let temp_dir = unique_temp_dir("tasks-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/tasks"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Tasks"));
    assert!(stderr.contains("Count            0"));
    assert!(stderr.contains("no active tasks are currently registered in this process"));
}

#[test]
fn slash_command_rename_without_name_reports_usage() {
    let temp_dir = unique_temp_dir("deferred-mock");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/rename"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Usage: /rename <name>"));
}

#[test]
fn slash_command_share_reports_created_artifact() {
    let temp_dir = unique_temp_dir("share-deferred");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/share"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Share artifact created"));
}

#[test]
fn slash_command_hidden_brief_still_returns_precise_runtime_message() {
    let temp_dir = unique_temp_dir("brief-deferred");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let home_dir = temp_dir.join("home");
    fs::create_dir_all(home_dir.join(".zo")).expect("home config dir");

    let output = run_zo_with_env(&temp_dir, &["/brief"], &[("HOME", &home_dir)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Brief mode set output style to: concise"));

    // `outputStyle` persists per-project (settings.local.json, CC parity) —
    // not in the user-global settings file.
    let settings = fs::read_to_string(temp_dir.join(".zo").join("settings.local.json"))
        .expect("brief mode settings should persist");
    assert!(settings.contains(r#""outputStyle": "concise""#));
}

#[test]
fn slash_command_keybindings_reports_built_ins() {
    let temp_dir = unique_temp_dir("keybindings-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let home_dir = temp_dir.join("home");
    fs::create_dir_all(home_dir.join(".zo")).expect("home config dir");

    let output = run_zo_with_env(&temp_dir, &["/keybindings"], &[("HOME", &home_dir)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Keybindings"));
    assert!(stderr.contains("Built-ins"));
}

#[test]
fn slash_command_privacy_settings_reports_storage_locations() {
    let temp_dir = unique_temp_dir("privacy-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let home_dir = temp_dir.join("home");
    fs::create_dir_all(home_dir.join(".zo")).expect("home config dir");

    let output = run_zo_with_env(&temp_dir, &["/privacy-settings"], &[("HOME", &home_dir)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Privacy settings"));
    assert!(stderr.contains("Credentials path"));
    assert!(stderr.contains("OAuth token"));
}

#[test]
fn slash_command_release_notes_summarizes_git_history() {
    let temp_dir = unique_temp_dir("release-notes-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    init_git_repo(&temp_dir);
    fs::write(temp_dir.join("notes.txt"), "hello\n").expect("seed notes");
    git(&temp_dir, &["add", "notes.txt"]);
    git(&temp_dir, &["commit", "-m", "Add notes"]);

    let output = run_zo(&temp_dir, &["/release-notes"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Release notes"));
    assert!(stderr.contains("Add notes"));
}

#[test]
fn slash_command_insights_reports_session_metrics() {
    let temp_dir = unique_temp_dir("insights-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/insights"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Insights"));
    assert!(stderr.contains("Messages"));
    assert!(stderr.contains("Total tokens"));
}

#[test]
fn slash_command_rename_updates_session_id_and_path() {
    let temp_dir = unique_temp_dir("rename-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    // Sessions persist to the global per-user home now; pin it inside the temp
    // dir so the test neither pollutes nor depends on the real `~/.zo`.
    let config_home = temp_dir.join(".zo-home");

    let output = run_zo_with_env(
        &temp_dir,
        &["/rename", "Focus Session"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Session renamed"));
    assert!(stderr.contains("Current ID       focus-session"));
    assert!(stderr.contains("/projects/"));
    assert!(stderr.contains("/sessions/focus-session.jsonl"));
}

#[test]
fn slash_command_copy_last_uses_clipboard_helper() {
    let temp_dir = unique_temp_dir("copy-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let bin_dir = temp_dir.join("bin");
    fs::create_dir_all(&bin_dir).expect("bin dir");
    let capture_path = temp_dir.join("clipboard.txt");
    let script_path = bin_dir.join("pbcopy");
    fs::write(
        &script_path,
        format!("#!/bin/sh\ncat > \"{}\"\n", capture_path.display()),
    )
    .expect("write pbcopy stub");
    let mut perms = fs::metadata(&script_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&script_path, perms).expect("chmod");

    let path_env = PathBuf::from(format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    ));

    let output = run_zo_with_env(&temp_dir, &["/copy", "all"], &[("PATH", &path_env)]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Copied to clipboard"));
    assert!(stderr.contains("Target           all"));

    let captured = fs::read_to_string(capture_path).expect("clipboard capture");
    assert!(!captured.trim().is_empty());
}

#[test]
fn slash_command_share_writes_local_share_artifact() {
    let temp_dir = unique_temp_dir("share-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");

    let output = run_zo(&temp_dir, &["/share"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Share artifact created"));

    let share_dir = temp_dir.join(".zo").join("share");
    let entries = fs::read_dir(&share_dir)
        .expect("share dir should exist")
        .collect::<Result<Vec<_>, _>>()
        .expect("read share dir");
    assert_eq!(entries.len(), 1);
    let artifact = fs::read_to_string(entries[0].path()).expect("share artifact");
    assert!(artifact.contains("# Conversation Export"));
}

#[test]
fn slash_command_desktop_uses_opener_stub() {
    let temp_dir = unique_temp_dir("desktop-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    let bin_dir = temp_dir.join("bin");
    fs::create_dir_all(&bin_dir).expect("bin dir");
    let capture_path = temp_dir.join("opened.txt");
    let script_path = bin_dir.join("open");
    fs::write(
        &script_path,
        format!(
            "#!/bin/sh\nprintf '%s' \"$1\" > \"{}\"\n",
            capture_path.display()
        ),
    )
    .expect("write open stub");
    let mut perms = fs::metadata(&script_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&script_path, perms).expect("chmod");

    let path_env = PathBuf::from(format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    ));
    // Pin the global session home into the temp dir (sessions are now stored
    // under `~/.zo/projects/<slug>/sessions`), so the opener path is
    // deterministic and the test does not touch the real `~/.zo`.
    let config_home = temp_dir.join(".zo-home");
    let output = run_zo_with_env(
        &temp_dir,
        &["/desktop"],
        &[("PATH", &path_env), ("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Desktop open requested"));
    let opened = fs::read_to_string(capture_path).expect("desktop capture");
    assert!(opened.contains("/sessions/"));
}

#[test]
fn slash_command_security_review_reports_review_and_secret_scan() {
    let temp_dir = unique_temp_dir("security-review-real");
    fs::create_dir_all(&temp_dir).expect("temp dir should exist");
    init_git_repo(&temp_dir);
    fs::write(temp_dir.join("config.txt"), "api_key = demo-secret\n").expect("seed file");

    let output = run_zo(&temp_dir, &["/security-review"]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Security review"));
    assert!(stderr.contains("Review"));
    assert!(stderr.contains("Secrets scan"));
}

fn run_zo(current_dir: &Path, args: &[&str]) -> Output {
    run_zo_with_env(current_dir, args, &[])
}

fn run_zo_with_env(current_dir: &Path, args: &[&str], envs: &[(&str, &Path)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_zo"));
    command
        .current_dir(current_dir)
        .args(args)
        .env("ANTHROPIC_API_KEY", "test-key");
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

fn git(current_dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(current_dir)
        .args(args)
        .output()
        .expect("git should launch");
    assert!(
        output.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_git_repo(current_dir: &Path) {
    git(current_dir, &["init"]);
    git(current_dir, &["config", "user.email", "zo@example.com"]);
    git(current_dir, &["config", "user.name", "Zo"]);
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
