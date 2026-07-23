//! Process-level tests for the top-level `zo doctor` command and its
//! automatic-safe-repair contract. Every test isolates `HOME` /
//! `ZO_CONFIG_HOME` / `ZO_STATE_DIR` into a private temp tree so the
//! developer's real home is never touched.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn help_lists_doctor_command() {
    let temp_dir = unique_temp_dir("doctor-help");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let output = run_doctor(&temp_dir, &["--help"], &[]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("zo doctor"), "{stdout}");
    assert!(stdout.contains("--check"), "{stdout}");
}

#[test]
fn doctor_long_help_flag_prints_usage() {
    let temp_dir = unique_temp_dir("doctor-longhelp");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let output = run_doctor(&temp_dir, &["doctor", "--help"], &[]);
    assert!(output.status.success(), "doctor --help must exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("zo doctor"), "{stdout}");
    assert!(stdout.contains("--check"), "{stdout}");
    // The diagnosis itself must not run for a help request.
    assert!(!stdout.contains("Summary"), "{stdout}");
}

#[test]
fn doctor_short_help_flag_prints_usage() {
    let temp_dir = unique_temp_dir("doctor-shorthelp");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let output = run_doctor(&temp_dir, &["doctor", "-h"], &[]);
    assert!(output.status.success(), "doctor -h must exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("zo doctor"), "{stdout}");
    assert!(stdout.contains("--check"), "{stdout}");
}

#[test]
fn doctor_renders_report_with_summary() {
    let temp_dir = unique_temp_dir("doctor-summary");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let output = run_doctor(&temp_dir, &["doctor"], &[]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Doctor"), "{stdout}");
    assert!(stdout.contains("Version"), "{stdout}");
    assert!(stdout.contains("Summary"), "{stdout}");
    assert!(
        stdout.contains("Healthy") || stdout.contains("Needs attention"),
        "{stdout}"
    );
    // PASS/WARN/FAIL/FIXED convention is visible.
    assert!(
        stdout.contains("PASS") || stdout.contains("WARN") || stdout.contains("FAIL"),
        "{stdout}"
    );
}

#[test]
fn check_mode_does_not_create_missing_config_home() {
    let temp_dir = unique_temp_dir("doctor-check-nomut");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    // Point the global config home at a path that does NOT yet exist.
    let config_home = temp_dir.join("config-home-absent");
    assert!(!config_home.exists());

    let output = run_doctor(
        &temp_dir,
        &["doctor", "--check"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());

    // Read-only mode must not create the directory.
    assert!(
        !config_home.exists(),
        "--check must not create the config home"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("WARN"), "{stdout}");
    assert!(stdout.contains("config-home"), "{stdout}");
}

#[test]
fn missing_optional_settings_does_not_fail_config_parse() {
    let temp_dir = unique_temp_dir("doctor-no-settings");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let output = run_doctor(&temp_dir, &["doctor", "--check"], &[]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parse_line = stdout
        .lines()
        .find(|line| line.contains("config/parse"))
        .unwrap_or_default();
    assert!(parse_line.contains("PASS"), "{stdout}");
}

#[test]
fn default_mode_creates_missing_config_home_privately() {
    let temp_dir = unique_temp_dir("doctor-repair-create");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home-new");
    assert!(!config_home.exists());

    let output = run_doctor(&temp_dir, &["doctor"], &[("ZO_CONFIG_HOME", &config_home)]);
    assert!(output.status.success());

    assert!(config_home.is_dir(), "default mode must create the home");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            fs::metadata(&config_home).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FIXED"), "{stdout}");
}

#[test]
fn default_mode_creates_nested_state_dir_on_first_run() {
    // A fresh isolated home: only the config home is created by the runner's
    // env; the nested project state directory must be created wholesale and
    // must not remain a FAIL. Regression for the first-run recovery gap.
    let temp_dir = unique_temp_dir("doctor-first-run");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("fresh-home");
    let state_dir = temp_dir.join("fresh-state");
    assert!(!config_home.exists());

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home), ("ZO_STATE_DIR", &state_dir)],
    );
    assert!(output.status.success());

    // `<state-dir>/projects/<slug>/state` must exist after a default run.
    let projects = state_dir.join("projects");
    assert!(projects.is_dir(), "nested state dir must be created");
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The state-dir check line must be FIXED or PASS, never FAIL.
    let state_line = stdout
        .lines()
        .find(|line| line.contains("state-dir"))
        .unwrap_or_default();
    assert!(
        state_line.contains("FIXED") || state_line.contains("PASS"),
        "state-dir must not remain failed: {stdout}"
    );
}

#[cfg(unix)]
#[test]
fn default_mode_tightens_preexisting_broad_state_suffix_dirs() {
    // SUFFIX-TIGHTENING regression: `$ZO_STATE_DIR` and its `projects/` already
    // exist at 0o777, with `<slug>/state` missing below. Every Zo-owned suffix
    // directory (base, projects, slug, state) must be tightened to 0o700 and
    // the run reported FIXED, while the non-Zo parent of the base stays
    // unchanged. The pre-fix walker chmod'd only newly created components,
    // leaving the pre-existing broad base and `projects/` world-accessible.
    use std::os::unix::fs::PermissionsExt as _;
    let temp_dir = unique_temp_dir("doctor-broad-suffix");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    // A non-Zo parent that must never be chmod'd, holding a broad state base.
    let non_zo_parent = temp_dir.join("non-zo-parent");
    fs::create_dir(&non_zo_parent).expect("non-zo parent");
    fs::set_permissions(&non_zo_parent, fs::Permissions::from_mode(0o755)).expect("parent perms");
    let state_dir = non_zo_parent.join("state-base");
    fs::create_dir(&state_dir).expect("state base");
    fs::set_permissions(&state_dir, fs::Permissions::from_mode(0o777)).expect("broad base");
    let projects = state_dir.join("projects");
    fs::create_dir(&projects).expect("projects");
    fs::set_permissions(&projects, fs::Permissions::from_mode(0o777)).expect("broad projects");

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home), ("ZO_STATE_DIR", &state_dir)],
    );
    assert!(output.status.success());

    // Every Zo-owned suffix directory must now be owner-only 0o700.
    let slug_entry = fs::read_dir(&projects)
        .expect("read projects")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("a <slug> directory must be created under projects");
    let state_leaf = slug_entry.join("state");
    for dir in [&state_dir, &projects, &slug_entry, &state_leaf] {
        assert_eq!(
            fs::metadata(dir).expect("suffix meta").permissions().mode() & 0o777,
            0o700,
            "Zo-owned suffix dir {dir:?} must be tightened to 0o700"
        );
    }
    // The non-Zo parent must never be chmod'd.
    assert_eq!(
        fs::metadata(&non_zo_parent).expect("parent meta").permissions().mode() & 0o777,
        0o755,
        "non-Zo parent must remain unchanged"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let state_line = stdout
        .lines()
        .find(|line| line.contains("state-dir"))
        .unwrap_or_default();
    assert!(
        state_line.contains("FIXED"),
        "state-dir must be reported FIXED once the whole suffix is private: {stdout}"
    );
}

#[cfg(unix)]
fn assert_existing_state_suffix_is_repaired(label: &str, leaf_mode: u32) {
    use std::os::unix::fs::PermissionsExt as _;

    let temp_dir = unique_temp_dir(label);
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    let non_zo_parent = temp_dir.join("non-zo-parent");
    fs::create_dir(&non_zo_parent).expect("non-Zo parent");
    fs::set_permissions(&non_zo_parent, fs::Permissions::from_mode(0o755))
        .expect("parent mode");
    let state_dir = non_zo_parent.join("state-base");

    let first = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home), ("ZO_STATE_DIR", &state_dir)],
    );
    assert!(first.status.success());
    let projects = state_dir.join("projects");
    let slug = fs::read_dir(&projects)
        .expect("read projects")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("project slug directory");
    let state_leaf = slug.join("state");

    for dir in [&state_dir, &projects, &slug] {
        fs::set_permissions(dir, fs::Permissions::from_mode(0o777)).expect("broad ancestor");
    }
    fs::set_permissions(&state_leaf, fs::Permissions::from_mode(leaf_mode))
        .expect("leaf mode");

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home), ("ZO_STATE_DIR", &state_dir)],
    );
    assert!(output.status.success());
    for dir in [&state_dir, &projects, &slug, &state_leaf] {
        assert_eq!(
            fs::metadata(dir).expect("suffix metadata").permissions().mode() & 0o777,
            0o700,
            "existing Zo-owned suffix dir {} must be repaired",
            dir.display()
        );
    }
    assert_eq!(
        fs::metadata(&non_zo_parent).expect("parent metadata").permissions().mode() & 0o777,
        0o755,
        "non-Zo parent must remain unchanged"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let state_line = stdout
        .lines()
        .find(|line| line.contains("state-dir"))
        .unwrap_or_default();
    assert!(state_line.contains("FIXED"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn existing_private_state_leaf_does_not_hide_broad_owned_ancestors() {
    assert_existing_state_suffix_is_repaired("doctor-private-leaf-broad-ancestors", 0o700);
}

#[cfg(unix)]
#[test]
fn existing_broad_state_leaf_repairs_all_broad_owned_ancestors() {
    assert_existing_state_suffix_is_repaired("doctor-broad-leaf-broad-ancestors", 0o777);
}

#[cfg(unix)]
#[test]
fn symlinked_state_root_is_never_followed_or_modified() {
    let temp_dir = unique_temp_dir("doctor-state-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    let real_state = temp_dir.join("real-state");
    let state_link = temp_dir.join("state-link");
    fs::create_dir(&config_home).expect("config home");
    fs::create_dir(&real_state).expect("real state target");
    std::os::unix::fs::symlink(&real_state, &state_link).expect("state symlink");

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home), ("ZO_STATE_DIR", &state_link)],
    );
    assert!(output.status.success());
    assert_eq!(
        fs::read_dir(&real_state).expect("read target").count(),
        0,
        "doctor must not create state through a symlinked root"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let finding_line = stdout
        .lines()
        .find(|line| line.contains("state-dir"))
        .unwrap_or_default();
    assert!(finding_line.contains("FAIL"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn default_mode_tightens_overly_broad_config_home() {
    use std::os::unix::fs::PermissionsExt as _;
    let temp_dir = unique_temp_dir("doctor-repair-chmod");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home-broad");
    fs::create_dir(&config_home).expect("create home");
    fs::set_permissions(&config_home, fs::Permissions::from_mode(0o755)).expect("chmod broad");

    let output = run_doctor(&temp_dir, &["doctor"], &[("ZO_CONFIG_HOME", &config_home)]);
    assert!(output.status.success());

    assert_eq!(
        fs::metadata(&config_home).unwrap().permissions().mode() & 0o777,
        0o700,
        "default mode must tighten to owner-only"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FIXED"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn broad_settings_file_is_tightened_but_content_preserved() {
    use std::os::unix::fs::PermissionsExt as _;
    let temp_dir = unique_temp_dir("doctor-settings-chmod");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    let settings = config_home.join("settings.json");
    let original = r#"{"model":"opus"}"#;
    fs::write(&settings, original).expect("write settings");
    fs::set_permissions(&settings, fs::Permissions::from_mode(0o644)).expect("chmod broad");

    let output = run_doctor(&temp_dir, &["doctor"], &[("ZO_CONFIG_HOME", &config_home)]);
    assert!(output.status.success());

    assert_eq!(
        fs::metadata(&settings).unwrap().permissions().mode() & 0o777,
        0o600,
        "settings.json must be tightened to owner-only"
    );
    assert_eq!(
        fs::read_to_string(&settings).unwrap(),
        original,
        "settings content must never be rewritten"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FIXED"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn symlinked_settings_file_is_failed_and_not_followed() {
    let temp_dir = unique_temp_dir("doctor-settings-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    // A settings.json that is a symlink to a valid JSON file elsewhere.
    let target = temp_dir.join("outside-settings.json");
    fs::write(&target, "{}").expect("write target");
    std::os::unix::fs::symlink(&target, config_home.join("settings.json")).expect("symlink");

    let output = run_doctor(
        &temp_dir,
        &["doctor", "--check"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The unsafe symlink must be surfaced as a FAIL on the config/parse line.
    let parse_line = stdout
        .lines()
        .find(|line| line.contains("config/parse"))
        .unwrap_or_default();
    assert!(parse_line.contains("FAIL"), "{stdout}");
    // The symlink target must be left untouched (never followed/modified).
    assert!(target.exists(), "symlink target must remain");
}

#[cfg(unix)]
#[test]
fn non_regular_settings_target_is_never_modified() {
    let temp_dir = unique_temp_dir("doctor-settings-nonregular");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    // `settings.json` is a directory, not a regular file.
    let odd = config_home.join("settings.json");
    fs::create_dir(&odd).expect("mkdir settings.json");

    let output = run_doctor(
        &temp_dir,
        &["doctor", "--check"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parse_line = stdout
        .lines()
        .find(|line| line.contains("config/parse"))
        .unwrap_or_default();
    assert!(parse_line.contains("FAIL"), "{stdout}");
    assert!(odd.is_dir(), "non-regular target must remain a directory");
}

#[cfg(unix)]
#[test]
fn symlinked_credentials_are_not_read() {
    let temp_dir = unique_temp_dir("doctor-credentials-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    let target = temp_dir.join("outside-credentials.json");
    fs::write(
        &target,
        r#"{"oauth":{"accessToken":"secret-from-symlink","refreshToken":null,"expiresAt":null,"scopes":[]}}"#,
    )
    .expect("write credentials target");
    std::os::unix::fs::symlink(&target, config_home.join("credentials.json"))
        .expect("credentials symlink");

    let mut command = base_command(&temp_dir, &["doctor", "--check"]);
    command.env_remove("ANTHROPIC_API_KEY");
    command.env("ZO_CONFIG_HOME", &config_home);
    command.env("ZO_STATE_DIR", temp_dir.join("state"));
    command.env("HOME", temp_dir.join("home"));
    let output = command.output().expect("zo should launch");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let auth_line = stdout
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(auth_line.contains("FAIL"), "{stdout}");
    assert!(!stdout.contains("secret-from-symlink"), "{stdout}");
    assert!(target.exists(), "credentials target must remain untouched");
}

#[cfg(unix)]
#[test]
fn symlinked_project_config_dir_is_not_a_repair_root() {
    use std::os::unix::fs::PermissionsExt as _;
    let temp_dir = unique_temp_dir("doctor-local-root-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    let outside = temp_dir.join("outside-zo");
    fs::create_dir(&config_home).expect("config home");
    fs::create_dir(&outside).expect("outside zo dir");
    let settings = outside.join("settings.local.json");
    fs::write(&settings, "{}").expect("local settings");
    fs::set_permissions(&settings, fs::Permissions::from_mode(0o644)).expect("broad mode");
    std::os::unix::fs::symlink(&outside, temp_dir.join(".zo")).expect("project .zo symlink");

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    assert_eq!(
        fs::metadata(&settings).expect("target metadata").permissions().mode() & 0o777,
        0o644,
        "doctor must not chmod through a symlinked project .zo directory"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let files_line = stdout
        .lines()
        .find(|line| line.contains("config-files"))
        .unwrap_or_default();
    assert!(files_line.contains("FAIL"), "{stdout}");
}

#[test]
fn malformed_settings_is_reported_and_not_rewritten() {
    let temp_dir = unique_temp_dir("doctor-malformed");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir_all(&config_home).expect("config home");
    let settings = config_home.join("settings.json");
    let malformed = "{ this is : not json ";
    fs::write(&settings, malformed).expect("write malformed");

    let output = run_doctor(
        &temp_dir,
        &["doctor", "--check"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parse_line = stdout
        .lines()
        .find(|line| line.contains("config/parse"))
        .unwrap_or_default();
    assert!(parse_line.contains("FAIL"), "{stdout}");
    // Content must be left exactly as-is; the raw text must not leak either.
    assert_eq!(
        fs::read_to_string(&settings).unwrap(),
        malformed,
        "malformed settings must never be rewritten"
    );
    assert!(
        !stdout.contains("this is"),
        "raw settings source must not leak into output: {stdout}"
    );
}

#[test]
fn missing_mcp_executable_is_warned_without_spawning_it() {
    let temp_dir = unique_temp_dir("doctor-mcp-missing");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    // A trusted User-scope config declaring a stdio MCP server whose command
    // is a REAL executable script that, if spawned, would create a marker file.
    // Doctor must never spawn it, so the marker must never appear even though
    // the command resolves on disk.
    let config_home = temp_dir.join("config-home");
    fs::create_dir_all(&config_home).expect("config home");
    let marker = temp_dir.join("mcp-ran.marker");
    let script = temp_dir.join("fake-mcp.sh");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::write(
            &script,
            format!("#!/bin/sh\ntouch \"{}\"\n", marker.display()),
        )
        .expect("write script");
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).expect("chmod script");
    }
    #[cfg(not(unix))]
    {
        fs::write(&script, "").expect("write script");
    }
    let missing = temp_dir.join("definitely-not-here-mcp");
    // Use a resolvable command (the script) for one server, and a nonexistent
    // command for another so the FAIL path is also exercised.
    fs::write(
        config_home.join("settings.json"),
        format!(
            r#"{{"mcpServers":{{"resolvable":{{"command":"{}","args":[]}},"broken":{{"command":"{}","args":[]}}}}}}"#,
            script.display(),
            missing.display()
        ),
    )
    .expect("write mcp settings");

    let output = run_doctor(
        &temp_dir,
        &["doctor", "--check"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    assert!(
        !marker.exists(),
        "the MCP command must never be spawned (marker created)"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The unresolvable command yields a FAIL naming it.
    assert!(stdout.contains("FAIL"), "{stdout}");
    assert!(stdout.contains("broken"), "{stdout}");
}

#[test]
fn secrets_never_appear_in_output() {
    let temp_dir = unique_temp_dir("doctor-secrets");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let secret = "sk-ant-super-secret-token-value-zzz";

    let output = run_doctor(&temp_dir, &["doctor"], &[]);
    // The runner sets ANTHROPIC_API_KEY; override it with an identifiable secret.
    let output_with_secret = {
        let mut command = base_command(&temp_dir, &["doctor"]);
        command.env("ANTHROPIC_API_KEY", secret);
        command.env("ZO_CONFIG_HOME", temp_dir.join(".zo-home"));
        command.env("ZO_STATE_DIR", temp_dir.join(".zo-state"));
        command.output().expect("zo should launch")
    };
    assert!(output.status.success());
    assert!(output_with_secret.status.success());
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output_with_secret.stdout),
        String::from_utf8_lossy(&output_with_secret.stderr)
    );
    assert!(
        !combined.contains(secret),
        "doctor output must never echo a credential value: {combined}"
    );
    // Auth presence is still reported.
    assert!(combined.contains("auth"), "{combined}");
}

#[test]
fn unexpected_doctor_argument_is_rejected() {
    let temp_dir = unique_temp_dir("doctor-badarg");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let output = run_doctor(&temp_dir, &["doctor", "--bogus"], &[]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("doctor"), "{stderr}");
}

#[cfg(unix)]
#[test]
fn intermediate_symlink_ancestor_state_root_is_not_traversed() {
    // NO-FOLLOW security regression: an *intermediate* ancestor of the state
    // root is a symlink whose target already contains `projects/`. The unfixed
    // deepest-existing-ancestor logic would pick `state-link/projects` as a
    // trusted root and create/chmod through the symlink. Default `zo doctor`
    // must FAIL and create/chmod nothing in the target.
    use std::os::unix::fs::PermissionsExt as _;
    let temp_dir = unique_temp_dir("doctor-state-intermediate-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    // The real target already contains `projects/` (a pre-existing descendant),
    // so a naive "deepest existing ancestor" walk would trust it.
    let real_state = temp_dir.join("real-state");
    fs::create_dir(&real_state).expect("real state");
    let real_projects = real_state.join("projects");
    fs::create_dir(&real_projects).expect("real projects");
    fs::set_permissions(&real_projects, fs::Permissions::from_mode(0o777)).expect("broad perms");
    // The configured ZO_STATE_DIR is a symlink to that real target.
    let state_link = temp_dir.join("state-link");
    std::os::unix::fs::symlink(&real_state, &state_link).expect("state symlink");

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home), ("ZO_STATE_DIR", &state_link)],
    );
    assert!(output.status.success());

    // Nothing created below the symlink target's `projects/` (no `<slug>/state`).
    assert_eq!(
        fs::read_dir(&real_projects).expect("read projects").count(),
        0,
        "doctor must not create state through an intermediate symlinked ancestor"
    );
    // The pre-existing target's permissions must be untouched (never chmod'd).
    assert_eq!(
        fs::metadata(&real_projects).expect("meta").permissions().mode() & 0o777,
        0o777,
        "doctor must not chmod through a symlinked ancestor"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let state_report_line = stdout
        .lines()
        .find(|line| line.contains("state-dir"))
        .unwrap_or_default();
    assert!(state_report_line.contains("FAIL"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn intermediate_symlink_ancestor_config_home_is_not_traversed() {
    // The config-home repair path routes through the same absolute no-follow
    // walker, so an intermediate symlink ancestor of the config home must also
    // FAIL and create nothing through the symlink.
    use std::os::unix::fs::PermissionsExt as _;
    let temp_dir = unique_temp_dir("doctor-confighome-intermediate-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let real_parent = temp_dir.join("real-parent");
    fs::create_dir(&real_parent).expect("real parent");
    fs::set_permissions(&real_parent, fs::Permissions::from_mode(0o777)).expect("broad perms");
    let link_parent = temp_dir.join("link-parent");
    std::os::unix::fs::symlink(&real_parent, &link_parent).expect("parent symlink");
    // Config home descends through the symlinked ancestor.
    let config_home = link_parent.join("config-home");

    let output = run_doctor(&temp_dir, &["doctor"], &[("ZO_CONFIG_HOME", &config_home)]);
    assert!(output.status.success());

    // Nothing created through the symlink; the target parent untouched.
    assert!(
        !real_parent.join("config-home").exists(),
        "doctor must not create the config home through a symlinked ancestor"
    );
    assert_eq!(
        fs::metadata(&real_parent).expect("meta").permissions().mode() & 0o777,
        0o777,
        "doctor must not chmod through a symlinked ancestor"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.contains("config-home"))
        .unwrap_or_default();
    assert!(line.contains("FAIL"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn intermediate_symlink_ancestor_settings_file_is_not_chmodded() {
    use std::os::unix::fs::PermissionsExt as _;

    let temp_dir = unique_temp_dir("doctor-config-file-intermediate-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let real_parent = temp_dir.join("real-parent");
    let real_config = real_parent.join("config-home");
    fs::create_dir_all(&real_config).expect("real config home");
    let settings = real_config.join("settings.json");
    fs::write(&settings, "{}").expect("settings file");
    fs::set_permissions(&settings, fs::Permissions::from_mode(0o777)).expect("broad mode");

    let link_parent = temp_dir.join("link-parent");
    std::os::unix::fs::symlink(&real_parent, &link_parent).expect("parent symlink");
    let config_home = link_parent.join("config-home");

    let output = run_doctor(
        &temp_dir,
        &["doctor"],
        &[("ZO_CONFIG_HOME", &config_home)],
    );
    assert!(output.status.success());
    assert_eq!(
        fs::metadata(&settings).expect("target metadata").permissions().mode() & 0o777,
        0o777,
        "doctor must not chmod a settings file through an intermediate symlink"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.contains("config-files") && line.contains("FAIL"))
        .unwrap_or_default();
    assert!(!line.is_empty(), "{stdout}");
}

/// Every ambient credential variable doctor's auth check can consult, plus
/// custom-provider keys that could satisfy any generic detection. A provider-
/// only regression test must strip all of these so exactly one credential is
/// visible and no inherited value can mask the defect or leak into output.
#[cfg(unix)]
const ALL_AMBIENT_CREDENTIAL_VARS: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_ACCESS_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "XAI_API_KEY",
    "ZO_CUSTOM_OPENAI_API_KEY",
];

#[cfg(unix)]
fn clear_ambient_credentials(command: &mut Command) {
    for key in ALL_AMBIENT_CREDENTIAL_VARS {
        command.env_remove(key);
    }
}

/// Strip every ambient credential variable and pin isolated credential roots
/// (`ZO_CONFIG_HOME`, `HOME`, `ZO_STATE_DIR`, and `ZO_HOME` via `base_command`)
/// into the per-test tree, so the only credential doctor can see is the one the
/// test deliberately sets. Returns the isolated config home.
#[cfg(unix)]
fn isolated_auth_command(temp_dir: &Path, args: &[&str]) -> (Command, PathBuf) {
    let config_home = temp_dir.join("config-home");
    fs::create_dir_all(&config_home).expect("config home");
    let mut command = base_command(temp_dir, args);
    clear_ambient_credentials(&mut command);
    command.env("ZO_CONFIG_HOME", &config_home);
    command.env("HOME", temp_dir.join("home"));
    command.env("ZO_STATE_DIR", temp_dir.join("state"));
    (command, config_home)
}

/// Run `zo doctor --check` with exactly one provider credential set (every other
/// supported ambient credential stripped and credential roots isolated) and
/// assert the auth finding is PASS naming exactly `expected_label`, without the
/// false "no credentials" warning and without echoing the secret value.
#[cfg(unix)]
fn assert_single_provider_key_is_recognized(
    label: &str,
    env_key: &str,
    secret: &str,
    expected_label: &str,
) {
    let temp_dir = unique_temp_dir(&format!("doctor-auth-{label}"));
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let (mut command, _config_home) = isolated_auth_command(&temp_dir, &["doctor", "--check"]);
    command.env(env_key, secret);
    let output = command.output().expect("zo should launch");
    assert!(output.status.success());

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let auth_line = combined
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(
        auth_line.contains("PASS"),
        "{env_key} alone must be recognized as credentials: {combined}"
    );
    assert!(
        auth_line.contains(&format!("credentials present for {expected_label}")),
        "auth finding must name exactly `{expected_label}`: {combined}"
    );
    assert!(
        !auth_line.contains(','),
        "only `{expected_label}` may be reported in this isolated test: {combined}"
    );
    assert!(
        !combined.contains("no provider API key"),
        "single {env_key} must not yield the no-credentials warning: {combined}"
    );
    assert!(
        !combined.contains(secret),
        "doctor must never echo the {env_key} value: {combined}"
    );
}

#[cfg(unix)]
#[test]
fn openai_api_key_alone_is_recognized() {
    assert_single_provider_key_is_recognized(
        "openai",
        "OPENAI_API_KEY",
        "sk-openai-secret-xyz",
        "OpenAI",
    );
}

#[cfg(unix)]
#[test]
fn google_api_key_alone_is_recognized() {
    assert_single_provider_key_is_recognized(
        "google",
        "GOOGLE_API_KEY",
        "goog-secret-xyz",
        "Google",
    );
}

#[cfg(unix)]
#[test]
fn xai_api_key_alone_is_recognized() {
    assert_single_provider_key_is_recognized("xai", "XAI_API_KEY", "xai-secret-xyz", "xAI");
}

/// Write a `credentials.json` under an isolated config home carrying only the
/// given top-level OAuth key, and assert doctor reports auth PASS naming exactly
/// `expected_label`, without leaking the token and without the false
/// no-credentials warning. Every ambient credential is stripped so the saved
/// OAuth is the only credential doctor can observe.
#[cfg(unix)]
fn assert_saved_oauth_is_recognized(
    label: &str,
    oauth_key: &str,
    secret: &str,
    expected_label: &str,
) {
    let temp_dir = unique_temp_dir(&format!("doctor-oauth-{label}"));
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let (mut command, config_home) = isolated_auth_command(&temp_dir, &["doctor", "--check"]);
    let credentials = format!(
        r#"{{"{oauth_key}":{{"accessToken":"{secret}","refreshToken":null,"expiresAt":null,"scopes":[]}}}}"#
    );
    fs::write(config_home.join("credentials.json"), credentials).expect("write credentials");

    let output = command.output().expect("zo should launch");
    assert!(output.status.success());

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let auth_line = combined
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(
        auth_line.contains("PASS"),
        "saved {oauth_key} OAuth must be recognized: {combined}"
    );
    assert!(
        auth_line.contains(&format!("credentials present for {expected_label}")),
        "auth finding must name exactly `{expected_label}`: {combined}"
    );
    assert!(
        !auth_line.contains(','),
        "only `{expected_label}` may be reported in this isolated test: {combined}"
    );
    assert!(
        !combined.contains("no provider API key"),
        "saved {oauth_key} OAuth must not yield the no-credentials warning: {combined}"
    );
    assert!(
        !combined.contains(secret),
        "doctor must never echo the saved {oauth_key} token: {combined}"
    );
}

#[cfg(unix)]
#[test]
fn saved_openai_oauth_is_recognized() {
    assert_saved_oauth_is_recognized(
        "openai",
        "openai_oauth",
        "openai-oauth-secret-xyz",
        "OpenAI OAuth",
    );
}

#[cfg(unix)]
#[test]
fn saved_google_oauth_is_recognized() {
    assert_saved_oauth_is_recognized(
        "google",
        "google_code_assist_oauth",
        "google-oauth-secret-xyz",
        "Google OAuth",
    );
}

#[cfg(unix)]
#[test]
fn unsafe_credentials_store_fails_even_with_other_provider_env_key() {
    // A symlinked credentials.json must yield FAIL even when another provider's
    // env key (OpenAI) is present — an unsafe store is never silently masked.
    let temp_dir = unique_temp_dir("doctor-unsafe-creds-with-env");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let (mut command, config_home) = isolated_auth_command(&temp_dir, &["doctor", "--check"]);
    let target = temp_dir.join("outside-credentials.json");
    fs::write(
        &target,
        r#"{"openai_oauth":{"accessToken":"secret-from-symlink","refreshToken":null,"expiresAt":null,"scopes":[]}}"#,
    )
    .expect("write credentials target");
    std::os::unix::fs::symlink(&target, config_home.join("credentials.json"))
        .expect("credentials symlink");

    // Deliberately set one provider env key: the unsafe store must still FAIL.
    command.env("OPENAI_API_KEY", "sk-openai-present");
    let output = command.output().expect("zo should launch");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let auth_line = stdout
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(
        auth_line.contains("FAIL"),
        "unsafe credentials store must FAIL even with an env key: {stdout}"
    );
    assert!(!stdout.contains("secret-from-symlink"), "{stdout}");
    assert!(target.exists(), "credentials target must remain untouched");
}

#[cfg(unix)]
#[test]
fn intermediate_symlink_ancestor_credentials_store_is_not_read() {
    // NO-FOLLOW credentials regression: `ZO_CONFIG_HOME` descends through an
    // intermediate symlinked ancestor whose real target already holds a valid
    // `credentials.json` with saved OpenAI OAuth. The unfixed no-follow reader
    // canonicalizes the caller-supplied root and would follow the symlink,
    // reporting the OAuth as present. The fix opens every component `O_NOFOLLOW`
    // from `/`, so the store is unsafe: auth must FAIL and no token must leak.
    let temp_dir = unique_temp_dir("doctor-creds-intermediate-symlink");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let real_parent = temp_dir.join("real-parent");
    fs::create_dir(&real_parent).expect("real parent");
    let real_config = real_parent.join("config-home");
    fs::create_dir(&real_config).expect("real config");
    let secret = "openai-oauth-through-symlink-xyz";
    fs::write(
        real_config.join("credentials.json"),
        format!(
            r#"{{"openai_oauth":{{"accessToken":"{secret}","refreshToken":null,"expiresAt":null,"scopes":[]}}}}"#
        ),
    )
    .expect("write credentials");
    // The configured config home reaches the real one only through a symlinked
    // intermediate ancestor.
    let link_parent = temp_dir.join("link-parent");
    std::os::unix::fs::symlink(&real_parent, &link_parent).expect("parent symlink");
    let config_home = link_parent.join("config-home");

    let mut command = base_command(&temp_dir, &["doctor", "--check"]);
    clear_ambient_credentials(&mut command);
    command.env("ZO_CONFIG_HOME", &config_home);
    command.env("HOME", temp_dir.join("home"));
    command.env("ZO_STATE_DIR", temp_dir.join("state"));
    let output = command.output().expect("zo should launch");
    assert!(output.status.success());

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let auth_line = combined
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(
        auth_line.contains("FAIL"),
        "credentials reached through a symlinked ancestor must FAIL: {combined}"
    );
    assert!(
        !combined.contains(secret),
        "doctor must never read/echo credentials behind a symlinked ancestor: {combined}"
    );
}

#[cfg(unix)]
#[test]
fn repair_does_not_create_missing_non_zo_config_home_parent() {
    // REPAIR-BOUNDARY regression: an explicit `ZO_CONFIG_HOME` whose non-Zo
    // parent does not exist must FAIL rather than fabricating that parent. The
    // unfixed absolute walker `mkdirat`'d every missing component from `/`,
    // creating (and chmod'ing) the caller's `missing-parent`.
    let temp_dir = unique_temp_dir("doctor-missing-parent");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let existing = temp_dir.join("existing");
    fs::create_dir(&existing).expect("existing dir");
    let missing_parent = existing.join("missing-parent");
    let config_home = missing_parent.join("config-home");

    let output = run_doctor(&temp_dir, &["doctor"], &[("ZO_CONFIG_HOME", &config_home)]);
    assert!(output.status.success());

    assert!(
        !missing_parent.exists(),
        "doctor must not create the non-Zo missing parent"
    );
    assert!(
        !config_home.exists(),
        "doctor must not create the config home under a missing non-Zo parent"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.contains("config-home"))
        .unwrap_or_default();
    assert!(line.contains("FAIL"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn anthropic_auth_token_alone_is_recognized() {
    assert_single_provider_key_is_recognized(
        "anthropic-auth-token",
        "ANTHROPIC_AUTH_TOKEN",
        "anthropic-auth-token-secret-xyz",
        "Anthropic",
    );
}

#[cfg(unix)]
#[test]
fn google_access_token_alone_is_recognized() {
    assert_single_provider_key_is_recognized(
        "google-access-token",
        "GOOGLE_ACCESS_TOKEN",
        "google-access-token-secret-xyz",
        "Google",
    );
}

#[cfg(unix)]
#[test]
fn explicit_adc_file_alone_is_recognized() {
    // A readable ADC file pointed to by GOOGLE_APPLICATION_CREDENTIALS is a
    // valid Google credential; the unfixed inventory ignored it entirely.
    let temp_dir = unique_temp_dir("doctor-adc");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("config home");
    let adc = temp_dir.join("adc.json");
    let secret = "adc-refresh-secret-xyz";
    fs::write(
        &adc,
        format!(
            r#"{{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"{secret}"}}"#
        ),
    )
    .expect("write adc");

    let mut command = base_command(&temp_dir, &["doctor", "--check"]);
    clear_ambient_credentials(&mut command);
    command.env("ZO_CONFIG_HOME", &config_home);
    command.env("HOME", temp_dir.join("home"));
    command.env("ZO_STATE_DIR", temp_dir.join("state"));
    command.env("GOOGLE_APPLICATION_CREDENTIALS", &adc);
    let output = command.output().expect("zo should launch");
    assert!(output.status.success());

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let auth_line = combined
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(
        auth_line.contains("PASS"),
        "an explicit ADC file must be recognized as a Google credential: {combined}"
    );
    assert!(
        auth_line.contains("credentials present for Google") && !auth_line.contains(','),
        "ADC must identify exactly Google credentials: {combined}"
    );
    assert!(
        !combined.contains("no provider API key"),
        "an ADC file must not yield the no-credentials warning: {combined}"
    );
    assert!(
        !combined.contains(secret),
        "doctor must never echo ADC file contents: {combined}"
    );
}

#[cfg(unix)]
#[test]
fn saved_oauth_in_lower_zo_home_root_is_recognized() {
    // LOWER-ROOT regression: the primary `ZO_CONFIG_HOME` is empty but a lower
    // supported root (`ZO_HOME`) carries valid OpenAI OAuth. The credential
    // model reads the full root chain, so doctor must recognize it; the unfixed
    // single-primary-file probe reported the false no-credentials warning.
    let temp_dir = unique_temp_dir("doctor-lower-root-oauth");
    fs::create_dir_all(&temp_dir).expect("temp dir");
    let config_home = temp_dir.join("config-home");
    fs::create_dir(&config_home).expect("empty primary config home");
    let zo_home = temp_dir.join("zo-home");
    fs::create_dir(&zo_home).expect("zo home");
    let secret = "lower-root-openai-oauth-xyz";
    fs::write(
        zo_home.join("credentials.json"),
        format!(
            r#"{{"openai_oauth":{{"accessToken":"{secret}","refreshToken":null,"expiresAt":null,"scopes":[]}}}}"#
        ),
    )
    .expect("write lower-root credentials");

    let mut command = base_command(&temp_dir, &["doctor", "--check"]);
    clear_ambient_credentials(&mut command);
    command.env("ZO_CONFIG_HOME", &config_home);
    command.env("ZO_HOME", &zo_home);
    command.env("HOME", temp_dir.join("home"));
    command.env("ZO_STATE_DIR", temp_dir.join("state"));
    let output = command.output().expect("zo should launch");
    assert!(output.status.success());

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let auth_line = combined
        .lines()
        .find(|line| line.contains("auth"))
        .unwrap_or_default();
    assert!(
        auth_line.contains("PASS"),
        "saved OAuth in a lower supported root must be recognized: {combined}"
    );
    assert!(
        auth_line.contains("credentials present for OpenAI OAuth") && !auth_line.contains(','),
        "lower-root OAuth must identify exactly OpenAI OAuth: {combined}"
    );
    assert!(
        !combined.contains("no provider API key"),
        "lower-root OAuth must not yield the no-credentials warning: {combined}"
    );
    assert!(
        !combined.contains(secret),
        "doctor must never echo the lower-root token: {combined}"
    );
}

fn base_command(current_dir: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_zo"));
    command.current_dir(current_dir).args(args);
    command.env_remove("ZO_HOME");
    // Never inherit a developer-configured state directory: default-mode tests
    // create/chmod entries under it. Each caller that cares sets its own value.
    command.env_remove("ZO_STATE_DIR");
    command
}

fn run_doctor(current_dir: &Path, args: &[&str], envs: &[(&str, &Path)]) -> Output {
    let mut command = base_command(current_dir, args);
    command.env("ANTHROPIC_API_KEY", "test-key");
    let caller_sets_home = envs.iter().any(|(key, _)| *key == "HOME");
    let caller_sets_config_home = envs.iter().any(|(key, _)| *key == "ZO_CONFIG_HOME");
    let caller_sets_state_dir = envs.iter().any(|(key, _)| *key == "ZO_STATE_DIR");
    if !caller_sets_home {
        command.env("HOME", current_dir.join(".home"));
    }
    if !caller_sets_home && !caller_sets_config_home {
        command.env("ZO_CONFIG_HOME", current_dir.join(".zo-home"));
    }
    // Keep state creation inside the per-test tree unless the caller pins it.
    if !caller_sets_state_dir {
        command.env("ZO_STATE_DIR", current_dir.join(".zo-state"));
    }
    for (key, value) in envs {
        command.env(key, value);
    }
    command.output().expect("zo should launch")
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_millis();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "zo-{label}-{}-{millis}-{counter}",
        std::process::id()
    ))
}
